/* Merged tracer host side (cyclescope capture spec,
 * docs/en/design/capture.md).  One source builds for ARCH=sm_90a and
 * ARCH=sm_103a; the only architecture input is the ArchTable picked from
 * the device compute capability.
 *
 * Produces the capture directory inside $CYCLESCOPE_TRACE_DIR:
 *   launches.jsonl, functions.jsonl, functions/<function_id>.tsv,
 *   tensor_maps.jsonl, trace-<launch>.bin.zst (or .bin when zstd is
 *   unavailable), l2_footprint.{bin,json}, placement.jsonl.
 * run.json / app.log are owned by the capture driver, not the tracer.
 */
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <regex>
#include <string>
#include <cmath>
#include <deque>
#include <mutex>
#include <new>
#include <sched.h>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
/* The NVBit channel buffer lives in device memory and is drained by the
 * host thread with cudaMemcpyAsync.  NVBit's default (host-pinned,
 * zero-copy) makes every record a PCIe write whose arbitration favours
 * fixed TPCs: under instrumentation SMs ran 1.4-1.9x apart and hardware
 * dynamic schedulers (cluster launch control, atomic tile counters) handed
 * the fast ones more tiles, so traced work assignment differed from the
 * reference run (full-gpu-chain-report §20).  -DCYCLESCOPE_TRACER_HOST_CHANNEL
 * rebuilds the old layout for A/B only. */
#ifndef CYCLESCOPE_TRACER_HOST_CHANNEL
#define USE_ASYNC_STREAM 1
#endif
#include "utils/channel.hpp"
#include "record_format.h"


#include "uniform_register.h"
#include "tensor_map_operand.h"
#include "arch_table.h"
#include "arch_hopper.h"
#include "arch_blackwell.h"

#ifndef CHANNEL_SIZE
#define CHANNEL_SIZE (1l << 22)
#endif

/* Architecture table selected at the first context init (CC major 9 ->
 * hopper, 10 -> blackwell).  Everything arch-specific goes through it. */
static const ArchTable *g_arch = nullptr;
static const ArchTable *arch_table_default() {
#if defined(CYCLESCOPE_BUILD_BLACKWELL)
  return &kArchBlackwell;
#else
  return &kArchHopper;
#endif
}
static void select_arch_table(CUcontext ctx) {
  if (g_arch)
    return;
  g_arch = arch_table_default();
  CUdevice dev = -1;
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS && dev >= 0) {
    int major = 0;
    if (cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             dev) == CUDA_SUCCESS) {
      if (major == 9)
        g_arch = &kArchHopper;
      else if (major == 10)
        g_arch = &kArchBlackwell;
    }
  }
}

__global__ void flush_channel();
#ifndef CYCLESCOPE_NVBIT_HAS_SASS_BINARY
#define CYCLESCOPE_NVBIT_HAS_SASS_BINARY 1
#endif
#if !CYCLESCOPE_NVBIT_HAS_SASS_BINARY
/* NVBit without Instr::getSassBinary: the functions/<id>.tsv 16-byte
 * encoding column is read out of the live code segment instead. */
__global__ void read_code_bytes(const uint8_t *base, uint64_t offset,
                                uint8_t *out) {
  for (int i = 0; i < 16; ++i)
    out[i] = base[offset + i];
}
#endif

static ChannelHost channel_host;
/* the one device-visible state object (common.h V5DevState).  Its
 * ChannelDev member has a host-only implicit constructor, so the managed
 * object is raw storage, constructed in nvbit_tool_init. */
static __managed__ __align__(16) unsigned char
    v5_dev_storage[sizeof(V5DevState)];
#define V5_DEV (*reinterpret_cast<V5DevState *>(v5_dev_storage))
static pthread_t recv_thread;
static volatile bool recv_thread_started = false;
static volatile bool recv_thread_receiving = false;
static volatile bool skip_flag = false;
static bool first_call = true;
static int verbose = 0;

static std::string user_folder;
static std::string capture_launches = "all"; /* CYCLESCOPE_CAPTURE_LAUNCHES */
/* CYCLESCOPE_CAPTURE_LAUNCH_LIMIT=<n>: trace only launch ordinals <= n (later
 * launches keep their launches.jsonl identity with traced=false).
 * 0 = no limit. */
static uint64_t capture_launch_limit = 0;
static bool g_current_launch_traced = false;
static int capture_clusters = -1;            /* CYCLESCOPE_CAPTURE_CLUSTERS:
                                              * -1 = all (default) */
static bool capture_clusters_sampled = false; /* "sampled:<n>" form */
static std::vector<uint32_t> capture_cluster_ids; /* explicit linear ids */
static bool capture_mode_placement = false;  /* CYCLESCOPE_CAPTURE_MODE=placement */
static std::vector<std::string> extra_operand_opcodes;
/* Debug knobs for isolating instrumentation faults (not for captures):
 * CAPTURE_SKIP_OPCODES=<prefix,...> leaves matching instructions
 * uninstrumented; CAPTURE_NO_AFTER=1 drops the PHASECHK AFTER hook. */
static std::vector<std::string> g_skip_prefixes;
/* CAPTURE_IDX_RANGE=<lo>-<hi>: instrument only instruction indices in
 * [lo, hi) of every function (bisects which instrumented instruction
 * triggers a device fault). */
static int64_t g_idx_lo = -1, g_idx_hi = -1;
/* CAPTURE_BARE_OPCODES=<prefix,...>: instrument matching instructions with
 * an empty layout (no mref/operand capture) -- separates NVBit's own
 * instrumentation of the instruction from the tracer's operand reads. */
static std::vector<std::string> g_bare_prefixes;
/* CAPTURE_NO_MREF_OPCODES / CAPTURE_NO_RANK_OPCODES /
 * CAPTURE_NO_XRANK_OPCODES: drop the NVBit mrefs (and their RANK
 * entries), only the per-mref RANK entries, or the URANK/BRANK entries. */
static std::vector<std::string> g_nomref_prefixes, g_norank_prefixes,
    g_noxrank_prefixes, g_nour_prefixes, g_nor_prefixes;
static bool no_after_hook = false;
static bool no_gsync = false; /* CAPTURE_NO_GSYNC=1 drops the gsync hooks */
static bool have_zstd = false;
static bool no_operands = false;    /* CAPTURE_OPERAND_FAMILIES=none */
/* CUDA_STACK_SIZE: device printf + injected calls can exceed the 1 KB
 * default thread stack (CU-CUTE-01's ~760 device printfs).  v4 shipped the
 * same escape; the tracer default matches its production value, 0
 * disables. */
static uint32_t cuda_stack_size = 65536;
static int cuda_stack_size_applied = 0;

/* ------------------------------------------------------------------ */
/* prior-launch L2 footprint (CYCLESCOPE_L2_FOOTPRINT)                      */
/* ------------------------------------------------------------------ */
/* While a launch before the selected one runs, its instrumented MREF
 * sites append sector entries to the device ring through
 * g_footprint_cell; the selected launch's enter callback (device already
 * idle) drains the ring and writes l2_footprint.{bin,json}.  Disabled in
 * placement mode. */
static __managed__ footprint_cell_t g_footprint_cell;
static bool g_footprint_enabled = false;
static bool g_footprint_kernel = true;
static bool g_footprint_host_ops_on = true;
static int64_t g_footprint_entries_env = 0;
static uint64_t g_footprint_entries = 0; /* ring capacity, power of two */
static uint64_t *g_footprint_ring_dev = nullptr;
static unsigned long long *g_footprint_counter_dev = nullptr;
static unsigned long long g_prior_launches_recorded = 0;
static unsigned long long g_prior_launches_with_bulk_ops = 0;
/* Per-function fact: does it contain any TMA/bulk opcode (whose MREF
 * sites are intentionally not footprint-instrumented). */
static std::unordered_map<CUfunction, bool> g_func_has_bulk_ops;
/* Host-side memcpy/memset calls also leave L2 footprints; recorded at
 * API-exit and interleaved into the drain by launch-boundary position. */
struct FootprintHostOp {
  uint64_t addr;
  uint64_t bytes;
  uint32_t is_store; /* 1 = device destination, 0 = device source */
  uint64_t seq;      /* launch ordinal at record time (0 before launch 1) */
  uint64_t order;    /* API call order among recorded ops */
};
static std::vector<FootprintHostOp> g_footprint_host_ops;
static std::map<uint64_t, uint64_t> g_launch_boundaries;
static unsigned long long g_host_ops_unsupported = 0;
static std::set<std::pair<uint64_t, uint64_t>> g_footprint_dev_ranges;
/* Set once the selected launch callback has run; footprint recording
 * stops and later untraced launches are fully de-instrumented. */
static bool g_selected_launch_seen = false;

static bool opcode_is_bulk_transfer(const std::string &opcode) {
  static const char *kBulkPrefixes[] = {"UTMALDG", "UTMASTG", "UTMAREDG",
                                        "UTMAPF",  "UBLKCP",  "UBLKRED",
                                        "UTMACCTL"};
  for (const char *p : kBulkPrefixes)
    if (opcode.rfind(p, 0) == 0)
      return true;
  return false;
}

/* ------------------------------------------------------------------ */
/* launch identity / prior-launch argument comparison                   */
/* ------------------------------------------------------------------ */
static std::mutex g_prior_launch_mutex;
static std::map<std::string, uint32_t> g_prior_launch_counts;
/* Launch ordinal of the most recent launch with the same identity key,
 * so the footprint can be split at the end of the last identical launch
 * (warmup replay boundary). */
static std::map<std::string, uint64_t> g_prior_launch_last_seq;
/* Earlier launches of the same (ctx, function) for
 * function_prior_launches. */
static std::map<std::pair<CUcontext, CUfunction>, uint32_t>
    g_function_prior_counts;
/* Per-regex-token match counts for the regex@N selection form. */
static std::map<std::string, uint64_t> g_regex_match_counts;

struct LaunchArgumentBytes {
  bool exact = false;
  std::string encoded;
  std::vector<long long> arg_sizes;
  std::string arg_hex;
};

template <typename T>
static void append_launch_key_value(std::string &key, const T &value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

static void append_hex(std::string &out, const void *bytes, size_t count) {
  const char *digits = "0123456789abcdef";
  const auto *p = static_cast<const unsigned char *>(bytes);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(digits[p[i] >> 4]);
    out.push_back(digits[p[i] & 15]);
  }
}

static bool get_kernel_parameter_array(nvbit_api_cuda_t cbid, void *params,
                                       void ***kernel_params,
                                       void ***extra) {
  *kernel_params = nullptr;
  *extra = nullptr;
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
      cbid == API_CUDA_cuLaunchKernelEx) {
    const auto *launch = static_cast<const cuLaunchKernelEx_params *>(params);
    if (launch == nullptr)
      return false;
    *kernel_params = launch->kernelParams;
    *extra = launch->extra;
    return true;
  }
  if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
      cbid == API_CUDA_cuLaunchKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
      cbid == API_CUDA_cuLaunchCooperativeKernel) {
    const auto *launch = static_cast<const cuLaunchKernel_params *>(params);
    if (launch == nullptr)
      return false;
    *kernel_params = launch->kernelParams;
    *extra = launch->extra;
    return true;
  }
  if (cbid == API_CUDA_cuGraphAddKernelNode) {
    const auto *node = static_cast<const CUDA_KERNEL_NODE_PARAMS *>(params);
    if (node == nullptr)
      return false;
    *kernel_params = node->kernelParams;
    *extra = node->extra;
    return true;
  }
  return false;
}

static bool get_extra_argument_buffer(void **extra, const void **buffer,
                                      std::size_t *buffer_size) {
  if (extra == nullptr)
    return false;
  bool found_buffer = false;
  bool found_size = false;
  bool found_end = false;
  for (std::size_t pair = 0; pair < 32; ++pair) {
    void *key = extra[pair * 2];
    if (key == CU_LAUNCH_PARAM_END) {
      found_end = true;
      break;
    }
    void *value = extra[pair * 2 + 1];
    if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      *buffer = value;
      found_buffer = true;
    } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      if (value == nullptr)
        return false;
      *buffer_size = *static_cast<const std::size_t *>(value);
      found_size = true;
    }
  }
  return found_end && found_buffer && found_size &&
         (*buffer != nullptr || *buffer_size == 0);
}

static LaunchArgumentBytes launch_argument_bytes(
    CUcontext ctx, CUfunction func, nvbit_api_cuda_t cbid, void *params) {
  void **kernel_params = nullptr;
  void **extra = nullptr;
  if (!get_kernel_parameter_array(cbid, params, &kernel_params, &extra))
    return {};

  if (kernel_params != nullptr) {
    const auto sizes = nvbit_get_kernel_argument_sizes(ctx, func);
    LaunchArgumentBytes result;
    result.exact = true;
    result.encoded.push_back('P');
    const auto count = static_cast<std::uint64_t>(sizes.size());
    append_launch_key_value(result.encoded, count);
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index] < 0 ||
          (sizes[index] > 0 && kernel_params[index] == nullptr))
        return {};
      const auto size = static_cast<std::uint32_t>(sizes[index]);
      append_launch_key_value(result.encoded, size);
      result.arg_sizes.push_back(static_cast<long long>(sizes[index]));
      if (size != 0) {
        result.encoded.append(
            static_cast<const char *>(kernel_params[index]), size);
        append_hex(result.arg_hex, kernel_params[index], size);
      }
    }
    return result;
  }

  if (extra != nullptr) {
    const void *buffer = nullptr;
    std::size_t buffer_size = 0;
    if (!get_extra_argument_buffer(extra, &buffer, &buffer_size))
      return {};
    LaunchArgumentBytes result;
    result.exact = true;
    result.encoded.push_back('B');
    const auto size = static_cast<std::uint64_t>(buffer_size);
    append_launch_key_value(result.encoded, size);
    result.arg_sizes.push_back(static_cast<long long>(buffer_size));
    if (buffer_size != 0) {
      result.encoded.append(static_cast<const char *>(buffer), buffer_size);
      append_hex(result.arg_hex, buffer, buffer_size);
    }
    return result;
  }

  const auto sizes = nvbit_get_kernel_argument_sizes(ctx, func);
  if (!sizes.empty())
    return {};
  LaunchArgumentBytes result;
  result.exact = true;
  result.encoded.push_back('P');
  const std::uint64_t count = 0;
  append_launch_key_value(result.encoded, count);
  return result;
}

static bool launch_identity_key(
    CUcontext ctx, CUfunction func, nvbit_api_cuda_t cbid, void *params,
    const std::array<uint32_t, 3> &grid,
    const std::array<uint32_t, 3> &block,
    const std::array<uint32_t, 3> &cluster, uint32_t dynamic_smem,
    std::string *key) {
  const auto arguments = launch_argument_bytes(ctx, func, cbid, params);
  if (!arguments.exact)
    return false;
  const auto context_address =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(ctx));
  const auto function_address = nvbit_get_func_addr(ctx, func);
  append_launch_key_value(*key, context_address);
  append_launch_key_value(*key, function_address);
  for (int i = 0; i < 3; ++i) {
    append_launch_key_value(*key, grid[i]);
    append_launch_key_value(*key, block[i]);
    append_launch_key_value(*key, cluster[i]);
  }
  append_launch_key_value(*key, dynamic_smem);
  const auto argument_size =
      static_cast<std::uint64_t>(arguments.encoded.size());
  append_launch_key_value(*key, argument_size);
  key->append(arguments.encoded);
  return true;
}

static void record_matching_launch(
    const std::string *key, bool key_exact, std::uint32_t *prior_identical,
    uint64_t launch_seq, uint64_t *last_identical_seq) {
  std::lock_guard<std::mutex> lock(g_prior_launch_mutex);
  *last_identical_seq = 0;
  if (key_exact && key != nullptr) {
    auto found = g_prior_launch_counts.find(*key);
    if (found == g_prior_launch_counts.end()) {
      g_prior_launch_counts.emplace(*key, 1);
      *prior_identical = 0;
    } else {
      *prior_identical = found->second;
      if (found->second != UINT32_MAX)
        ++found->second;
      const auto last = g_prior_launch_last_seq.find(*key);
      if (last != g_prior_launch_last_seq.end())
        *last_identical_seq = last->second;
    }
    g_prior_launch_last_seq[*key] = launch_seq;
  }
}

/* ------------------------------------------------------------------ */
/* function registration (FnReg)                                       */
/* ------------------------------------------------------------------ */

struct FuncReg {
  uint64_t id = 0; /* 1-based, unique per run */
  CUcontext ctx = nullptr;
  CUmodule module = nullptr;
  uint64_t related_of = 0; /* entry reg id for callees, 0 for entries */
  uint64_t addr = 0;
  int instructions = 0;
  std::string name;
};

struct OpEntry {
  uint8_t kind = 0;   /* V5_OP_* */
  uint8_t aux = 0;    /* RANK/BRANK: mref index */
  int reg_num = -1;   /* R/UR register number (BRANK: the R part) */
  std::string name;   /* tsv token, e.g. R4 / UR9 / RANK0 / DESC / PRED */
  int slot = -1;      /* on-wire ops[] slot base (set during instrumentation) */
  int reg2_num = -1;  /* BRANK: the UR part (-1 = none) */
  int32_t disp = 0;   /* URANK/BRANK: immediate displacement */
  int aux_op_slot() const { return slot; }
};

/* v5.4 global-sync layout: a mechanical operand split for the four
 * opcode classes in gsync_class(); a shape that does not classify is
 * "unsupported" (logged once per static instruction, no gsync hooks,
 * the BEFORE record is unchanged). */
struct GsyncOperand {
  bool is_ureg = false;
  bool is_zero = false; /* RZ/URZ: the constant 0 */
  int base = -1;
  int words = 1;
};
struct GsyncLayout {
  int cls = 0; /* 0 none, 1 atom, 2 red, 3 ld_strong, 4 st_strong */
  bool supported = false;
  int before_words = 0; /* source words + pre_value words */
  int source_words = 0;
  int after_words = 0;
  int flags = 0;    /* bit0: pre_value appended after the sources */
  int pre_bytes = 0; /* 0/4/8 */
  std::vector<GsyncOperand> sources;
  std::vector<GsyncOperand> dests; /* non-zero destinations */
};

struct InstrLayout {
  int mref_count = 0;
  std::vector<OpEntry> ops;
  GsyncLayout gsync;
};

static std::unordered_map<CUfunction, FuncReg> g_func_regs;
static std::unordered_map<CUfunction, CUmodule> g_module_of_function;
static std::unordered_set<CUfunction> g_instrumented;
/* (reg<<32)|idx -> serialized tail layout, for the recv parser. */
static std::unordered_map<uint64_t, InstrLayout> g_layouts;
static uint64_t g_next_reg_id = 1;
static uint64_t g_launch_counter = 0;

static FILE *g_functions_jsonl = nullptr;
static FILE *g_launches_jsonl = nullptr;
static FILE *g_tensor_maps_jsonl = nullptr;

static void json_escape(FILE *f, const std::string &s) {
  fputc('"', f);
  for (char c : s) {
    if (c == '"' || c == '\\')
      fprintf(f, "\\%c", c);
    else if ((unsigned char)c < 0x20)
      fprintf(f, "\\u%04x", (unsigned char)c);
    else
      fputc(c, f);
  }
  fputc('"', f);
}

/* ------------------------------------------------------------------ */
/* operand layout builder                                              */
/* ------------------------------------------------------------------ */

static std::vector<std::string> g_operand_prefixes;
static std::vector<std::string> g_operand_exact;

static bool opcode_in_operand_family(const std::string &opcode) {
  for (const auto &prefix : g_operand_prefixes) {
    if (!prefix.empty() && opcode.rfind(prefix, 0) == 0)
      return true;
  }
  for (const auto &exact : g_operand_exact)
    if (opcode == exact)
      return true;
  return false;
}

/* Split SASS operand text (after the opcode token and optional guard) into
 * tokens and extract captured_operands entries.  Grammar facts only:
 * [R<n>]/[UR<n>] brackets contribute the register pair (n and n+1), bare
 * UR<n>/R<n> contribute one word, each NVBit MREF contributes RANK<k>. */
static void sass_tokens(const std::string &sass,
                        std::vector<std::string> &tokens) {
  /* drop a leading @P/@!P guard, then the opcode token itself */
  size_t pos = 0;
  auto skip_ws = [&] { while (pos < sass.size() && isspace(sass[pos])) ++pos; };
  skip_ws();
  if (pos < sass.size() && sass[pos] == '@') {
    while (pos < sass.size() && !isspace(sass[pos]))
      ++pos;
    skip_ws();
  }
  /* opcode token: up to first whitespace */
  while (pos < sass.size() && !isspace(sass[pos]))
    ++pos;
  const std::string rest = pos < sass.size() ? sass.substr(pos) : "";
  size_t start = 0;
  for (size_t i = 0; i <= rest.size(); ++i) {
    if (i == rest.size() || rest[i] == ',') {
      std::string token = rest.substr(start, i - start);
      /* trim */
      const auto b = token.find_first_not_of(" \t");
      const auto e = token.find_last_not_of(" \t;");
      if (b != std::string::npos)
        tokens.push_back(token.substr(b, e - b + 1));
      start = i + 1;
    }
  }
}

static const std::regex kRegBracket(R"(^\[(UR|R)([0-9]{1,3})([^\]]*)\]$)");
static const std::regex kBracketUreg(R"((?:^|\+)(UR)([0-9]{1,3}))");
static const std::regex kBracketImm(R"((?:^|\+)(0[xX][0-9a-fA-F]+|[0-9]+))");

/* Parse the "+URy" / "+imm" tail of an [Rx ...] or [URx ...] bracket. */
static void bracket_addends(const std::string &tail, int &ureg_num,
                            int32_t &disp) {
  ureg_num = -1;
  disp = 0;
  std::smatch m;
  if (std::regex_search(tail, m, kBracketUreg))
    ureg_num = std::stoi(m[2]);
  if (std::regex_search(tail, m, kBracketImm))
    disp = (int32_t)std::stoll(m[1], nullptr, 0);
}
static const std::regex kRegBare(R"(^(UR|R)([0-9]{1,3})$)");
/* c[0x0][UR5+0x240] / c[0x0][R2] : constant-bank operand with a register
 * addend (immediate-only forms need no capture). */
static const std::regex kConstBankReg(
    R"(^c\[0x[0-9a-fA-F]+\]\[(UR|R)([0-9]{1,3})(?:\+0x[0-9a-fA-F]+)?\]$)");
/* UTC*MMA descriptor operands: idesc[URx] (instruction descriptor word),
 * gdesc[URx] (64-bit global-descriptor pair URx,URx+1), tmem[URx]
 * (destination TMEM address word).  None match kRegBracket/kRegBare.
 * Only present on architectures whose table has descriptor tokens. */
static const std::regex kUdescBracket(R"(^(idesc|gdesc|tmem)\[UR([0-9]{1,3})\]$)");
/* UTMA* cache-policy operand: desc[URx] -> CPOL/CPOLH pair. */
static const std::regex kCpolBracket(R"(^desc\[UR([0-9]{1,3})\]$)");

static std::string opcode_root(const std::string &opcode) {
  const size_t dot = opcode.find('.');
  return dot == std::string::npos ? opcode : opcode.substr(0, dot);
}

static bool root_in(const std::string &opcode, const char *const *roots,
                    int count) {
  for (int i = 0; i < count; ++i)
    if (opcode.rfind(roots[i], 0) == 0)
      return true;
  return false;
}

static void build_layout(Instr *instr, const std::string &opcode,
                         bool family, InstrLayout &layout) {
  /* NVBit MREF count decides the mref_addr64 args (clamped at V5_MAX_MREF). */
  int num_mref = 0;
  const int nops = instr->getNumOperands();
  for (int i = 0; i < nops; ++i) {
    const InstrType::operand_t *op = instr->getOperand(i);
    if (op->type == InstrType::OperandType::MREF)
      ++num_mref;
  }
  layout.mref_count = std::min(num_mref, V5_MAX_MREF);
  /* SPACE entry: any instruction NVBit marks GENERIC-space with at least
   * one MREF carries the isspacep ballots of mref0 -- not family-gated. */
  if (layout.mref_count > 0 &&
      instr->getMemorySpace() == InstrType::MemorySpace::GENERIC)
    layout.ops.push_back({V5_OP_SPACE, 0, -1, "SPACE"});
  /* Constant-bank operands c[bank][REG+imm]: the register addend is the
   * only dynamic part of the constant address (every arch, every root;
   * the converter adds the immediate and the bank from the SASS token). */
  {
    std::vector<std::string> ctokens;
    if (instr->getSass())
      sass_tokens(instr->getSass(), ctokens);
    for (const auto &token : ctokens) {
      std::smatch m;
      if (!std::regex_match(token, m, kConstBankReg))
        continue;
      const std::string cls = m[1];
      const int base = std::stoi(m[2]);
      if (cls == "UR" &&
          !ureg_sampling_available(UregOperand{UregKind::Numeric, base}, 1))
        continue;
      bool dup = false;
      for (const auto &e : layout.ops)
        if (e.reg_num == base && e.kind == (cls == "UR" ? V5_OP_UR : V5_OP_R))
          dup = true;
      if (!dup)
        layout.ops.push_back({uint8_t(cls == "UR" ? V5_OP_UR : V5_OP_R), 0,
                              base, cls + std::to_string(base)});
    }
  }
  if (!family)
    return;

  std::vector<std::string> tokens;
  if (instr->getSass())
    sass_tokens(instr->getSass(), tokens);
  // UBLKRED's [URx] brackets (global destination, shared source) are
  // captured like UBLKCP's since 2026-11-08: the converter takes the
  // global-side base from the .G bracket's UR pair (the predecessor
  // skipped them, leaving the reduction with no global address:
  // full-gpu-chain-report.md §15.2)
  for (const auto &token : tokens) {
    std::smatch m;
    if (std::regex_match(token, m, kRegBracket)) {
      const std::string cls = m[1];
      const int base = std::stoi(m[2]);
      OpEntry a{uint8_t(cls == "UR" ? V5_OP_UR : V5_OP_R), 0, base,
                cls + std::to_string(base)};
      OpEntry b{uint8_t(cls == "UR" ? V5_OP_UR : V5_OP_R), 0, base + 1,
                cls + std::to_string(base + 1)};
      layout.ops.push_back(a);
      /* R(x+1) only for a 64-bit [Rx.64] pair: a 32-bit [Rx+...] bracket
       * has no partner register, and reading it at SYNCS.ARRIVE.*.RED
       * raised illegal-instruction in CU-76/CU-92/FA4-BWD kernels
       * (bisected with CAPTURE_NO_R_OPCODES).  UR pairs stay: UTMA/UBLKCP
       * companions live in UR(x+1). */
      // UR pairs stay -- except a bulk op's odd-numbered [URx]: a
      // 64-bit address pair is even-aligned, an odd base is the 32-bit
      // shared-side address with no partner (FA4 bwd `UBLKRED.G.S
      // [UR6], [UR5], UR9`: pushing (UR5, UR6) duplicated UR6 in the
      // layout and the converter lost the instruction's UR values)
      const bool bulk_op = opcode.rfind("UBLKCP", 0) == 0 ||
                           opcode.rfind("UBLKRED", 0) == 0;
      if ((cls == "UR" && !(bulk_op && (base & 1))) ||
          m[3].str().find(".64") != std::string::npos)
        layout.ops.push_back(b);
    } else if (std::regex_match(token, m, kRegBare)) {
      const std::string cls = m[1];
      const int base = std::stoi(m[2]);
      layout.ops.push_back({uint8_t(cls == "UR" ? V5_OP_UR : V5_OP_R), 0, base,
                            cls + std::to_string(base)});
      /* 64-bit bare data operands are real register pairs (n, n+1):
       * STS.64's data (Rn,Rn+1) and SYNCS.EXCH.64's mbarrier init state
       * (URn,URn+1 or Rn,Rn+1) -- the init state's high word carries the
       * initial phase bit, which the converter needs. */
      const bool data_pair64 =
          (opcode == "STS.64" && cls == "R") ||
          opcode.rfind("SYNCS.EXCH.64", 0) == 0;
      if (data_pair64)
        layout.ops.push_back({uint8_t(cls == "UR" ? V5_OP_UR : V5_OP_R), 0,
                              base + 1, cls + std::to_string(base + 1)});
    }
  }
  /* UTC*MMA descriptor values (arch table: descriptor tokens exist only
   * where the grammar carries them), appended after the generic operand
   * entries so existing layouts keep their order. */
  if (g_arch->has_descriptor_tokens)
    for (const auto &token : tokens) {
    std::smatch m;
    if (!std::regex_match(token, m, kUdescBracket))
      continue;
    const std::string what = m[1];
    const int base = std::stoi(m[2]);
    const char *label = what == "idesc" ? "IDESC" : (what == "gdesc" ? "GDESC" : "TMEMD");
    if (!ureg_sampling_available(UregOperand{UregKind::Numeric, base}, 1))
      continue;
    layout.ops.push_back({V5_OP_UR, 0, base, label});
    if (what == "gdesc" &&
        ureg_sampling_available(UregOperand{UregKind::Numeric, base + 1}, 1))
      layout.ops.push_back({V5_OP_UR, 0, base + 1, "GDESCH"});
  }
  /* UTMA* cache-policy operand desc[URx] -> CPOL/CPOLH (URx, URx+1). */
  if (opcode.rfind("UTMA", 0) == 0) {
    for (const auto &token : tokens) {
      std::smatch m;
      if (!std::regex_match(token, m, kCpolBracket))
        continue;
      const int base = std::stoi(m[1]);
      if (ureg_sampling_available(UregOperand{UregKind::Numeric, base}, 1))
        layout.ops.push_back({V5_OP_CPOL, 0, base, "CPOL"});
      if (ureg_sampling_available(
              UregOperand{UregKind::Numeric, base + 1}, 1))
        layout.ops.push_back({V5_OP_CPOLH, 0, base + 1, "CPOLH"});
    }
  }
  /* one RANK entry per NVBit MREF */
  for (int m = 0; m < layout.mref_count; ++m) {
    layout.ops.push_back({V5_OP_RANK, uint8_t(m), -1,
                          "RANK" + std::to_string(m)});
  }
  /* UTMA*: tensormap address operand -> DESC (host dedup id) */
  if (opcode.rfind("UTMA", 0) == 0) {
    int role = 0, map_ureg = -1;
    if (tma_map_operand_class(opcode, instr->getSass(), &role, &map_ureg) &&
        role == 1 && map_ureg >= 0 &&
        ureg_sampling_available(UregOperand{UregKind::Numeric, map_ureg}, 2)) {
      OpEntry desc{V5_OP_DESC, 0, map_ureg, "DESC"};
      layout.ops.push_back(desc);
    }
  }
  /* Owner rank of a barrier uniform operand (v4 rule: getctarank on the
   * operand's own shared::cluster address word).
   * UTMALDG/UTMAREDG/UTMAPF/UBLKCP pack dst|mbarrier into the first [URx]
   * operand's pair, so the barrier companion word is x+1; UTCBAR's [URx]
   * operand is the barrier itself, so the barrier word is x; SYNCS/ARRIVES
   * carry the mbarrier operand as the [URx] bracket, so the word is x. */
  {
    const bool barrier_op =
        root_in(opcode, g_arch->mbarrier_companion_roots,
                g_arch->mbarrier_companion_root_count) ||
        root_in(opcode, g_arch->mbarrier_self_roots,
                g_arch->mbarrier_self_root_count);
    int urank_num = -1;
    int32_t urank_disp = 0;
    if (barrier_op) {
      /* UTMAPF's two-bracket form ([URcoords], [URmap]) packs coordinates
       * in the first bracket, not dst|mbarrier -- only the one-bracket
       * form carries the barrier companion word (genaudit probe
       * utmapf.sass). */
      bool utmapf_multi_bracket = false;
      if (opcode.rfind("UTMAPF", 0) == 0) {
        int bracket_count = 0;
        for (const auto &token : tokens) {
          std::smatch cm;
          if (std::regex_match(token, cm, kRegBracket))
            ++bracket_count;
        }
        utmapf_multi_bracket = bracket_count >= 2;
      }
      int mref_i = -1;
      for (const auto &token : tokens) {
        std::smatch m;
        if (std::regex_match(token, m, kRegBracket)) {
          ++mref_i;
          if (m[1] == std::string("UR") && urank_num < 0 &&
              !utmapf_multi_bracket) {
            int _u = -1;
            bracket_addends(m[3], _u, urank_disp);
            urank_num = std::stoi(m[2]);
            if (root_in(opcode, g_arch->mbarrier_companion_roots,
                        g_arch->mbarrier_companion_root_count))
              ++urank_num;
          } else if (m[1] == std::string("R") &&
                     layout.ops.size() + 2 <= V5_MAX_OPS) {
            /* [Rx+URy+disp]: the mbarrier operand word is R+UR+disp;
             * the aux BRVAL slot immediately after carries UR/disp. */
            int ureg = -1;
            int32_t disp = 0;
            bracket_addends(m[3], ureg, disp);
            OpEntry be{V5_OP_BRANK, uint8_t(mref_i), std::stoi(m[2]),
                       "BRANK" + std::to_string(mref_i)};
            layout.ops.push_back(be);
            OpEntry aux{V5_OP_PRED, 0, -1,
                        "BRVAL" + std::to_string(mref_i)};
            aux.reg2_num = ureg;
            aux.disp = disp;
            layout.ops.push_back(aux);
          }
        }
      }
    }
    if (urank_num >= 0 &&
        layout.ops.size() + 2 <= V5_MAX_OPS &&
        ureg_sampling_available(
            UregOperand{UregKind::Numeric, urank_num}, 1)) {
      OpEntry ue{V5_OP_URANK, 0, urank_num,
                 "URANK" + std::to_string(urank_num)};
      layout.ops.push_back(ue);
      OpEntry aux{V5_OP_PRED, 0, -1, "BRVAL"};
      aux.disp = urank_disp;
      layout.ops.push_back(aux);
    }
    /* STAS [R<n>.64] packs the shared::cluster mbarrier word in R(n+1):
     * capture its owner rank directly (per-lane getctarank on the packed
     * word, disp 0) instead of relying on an earlier barrier op in the
     * warp having named the same address. */
    if (opcode.rfind("STAS", 0) == 0) {
      int mref_i = -1;
      for (const auto &token : tokens) {
        std::smatch m;
        if (!std::regex_match(token, m, kRegBracket))
          continue;
        ++mref_i;
        if (m[1] == std::string("R") &&
            m[3].str().find(".64") != std::string::npos &&
            layout.ops.size() + 2 <= V5_MAX_OPS) {
          OpEntry be{V5_OP_BRANK, uint8_t(mref_i), std::stoi(m[2]) + 1,
                     "BRANK" + std::to_string(mref_i)};
          layout.ops.push_back(be);
          OpEntry aux{V5_OP_PRED, 0, -1,
                      "BRVAL" + std::to_string(mref_i)};
          aux.reg2_num = -1;
          aux.disp = 0;
          layout.ops.push_back(aux);
          break;
        }
      }
    }
  }
  /* TMA coordinate bank: the first [URx] bracket is a contiguous block;
   * the coordinate base is first_ureg + the arch table's per-root offset
   * (hopper: dst|mbar pair/single-word conventions; blackwell: the
   * UTMAPF two-bracket exception).  Rank comes from the opcode's .ND
   * dimension suffix (<= 5). */
  {
    int rank = -1;
    {
      std::smatch rm;
      if (std::regex_search(opcode, rm, std::regex(R"(\.(\d)D)")))
        rank = std::stoi(rm[1]);
      if (opcode.find("IM2COL") != std::string::npos && rank > 0)
        rank += 2;  /* im2col packs two extra offset words */
    }
    if (rank > 0) {
      int bracket_count = 0;
      int first_ureg = -1;
      for (const auto &token : tokens) {
        std::smatch m;
        if (!std::regex_match(token, m, kRegBracket))
          continue;
        ++bracket_count;
        if (first_ureg < 0 && m[1] == "UR")
          first_ureg = std::stoi(m[2]);
      }
      const int offset = first_ureg >= 0
          ? g_arch->tma_coordinate_offset(opcode_root(opcode).c_str(),
                                          bracket_count)
          : -1;
      if (offset >= 0) {
        for (int c = 0; c < rank && c <= 6; ++c) {
          const int reg = first_ureg + offset + c;
          if (layout.ops.size() + 2 > V5_MAX_OPS) break;
          if (!ureg_sampling_available(UregOperand{UregKind::Numeric, reg}, 1))
            continue;
          /* named by register so the converter selects coordinates by
           * UR number under its own per-root offset rule */
          bool dup = false;
          for (const auto &e : layout.ops)
            if (e.kind == V5_OP_UR && e.reg_num == reg) dup = true;
          if (dup) continue;
          layout.ops.push_back({V5_OP_UR, 0, reg,
                                "UR" + std::to_string(reg)});
        }
      }
    }
  }
  /* PHASECHK AFTER pairing marker */
  if (opcode.rfind("SYNCS.PHASECHK", 0) == 0 ||
      opcode.find("PHASECHK") != std::string::npos) {
    layout.ops.push_back({V5_OP_PRED, 0, -1, "PRED"});
  }
  /* drop UR>=80 / unsampleable operands (converter treats them missing) */
  std::vector<OpEntry> kept;
  for (const auto &e : layout.ops) {
    if (e.kind == V5_OP_UR) {
      UregOperand uo{UregKind::Numeric, e.reg_num};
      if (!ureg_sampling_available(uo))
        continue;
    }
    kept.push_back(e);
    if (kept.size() >= V5_MAX_OPS)
      break;
  }
  layout.ops = kept;
}

/* Global-sync opcode classes (patterns anchored at the opcode start):
 * atom ^(ATOMG|ATOM)\b (ATOMS / UTCATOMSWS do not match),
 * red ^(REDG|RED)\b (REDUX / REDAS / REDS do not match),
 * ld_strong ^LD(G)?\..*STRONG\.(GPU|SYS),
 * st_strong ^ST(G)?\..*STRONG\.(GPU|SYS).  MEMBAR carries no value. */
static int gsync_class(const std::string &opcode) {
  static const std::regex atom_re(R"(^(ATOMG|ATOM)\b)");
  static const std::regex red_re(R"(^(REDG|RED)\b)");
  static const std::regex ld_re(R"(^LD(G)?\..*STRONG\.(GPU|SYS))");
  static const std::regex st_re(R"(^ST(G)?\..*STRONG\.(GPU|SYS))");
  if (std::regex_search(opcode, atom_re))
    return 1;
  if (std::regex_search(opcode, red_re))
    return 2;
  if (std::regex_search(opcode, ld_re))
    return 3;
  if (std::regex_search(opcode, st_re))
    return 4;
  return 0;
}

/* u32 words one gsync operand occupies: the opcode tokenized on '.', a
 * token containing "128" -> 4, containing "64" (64/U64/S64/F64) -> 2,
 * otherwise 1 (S32/F32/U8/U16/S16). */
static int gsync_operand_words(const std::string &opcode) {
  bool wide64 = false;
  size_t pos = 0;
  while (pos <= opcode.size()) {
    const size_t dot = opcode.find('.', pos);
    const std::string tok =
        opcode.substr(pos, dot == std::string::npos ? std::string::npos
                                                    : dot - pos);
    if (tok.find("128") != std::string::npos)
      return 4;
    if (tok.find("64") != std::string::npos)
      wide64 = true;
    if (dot == std::string::npos)
      break;
    pos = dot + 1;
  }
  return wide64 ? 2 : 1;
}

/* Mechanical operand split over the structured NVBit operands (no SASS
 * text parsing): exactly one MREF is required; MEM_DESC and
 * TMA_PARAM_HANDLE operands are skipped (sm_103a `desc[URx][Ry.64]`
 * addresses carry a MEM_DESC); a PRED operand is skipped only when it is
 * PT (the throwaway predicate sink of `ATOMG ... PT, R3, ...`).  REGs
 * before the MREF are destinations, REG/UREGs after it are sources;
 * anything else is unsupported. */
static void gsync_layout(Instr *instr, const std::string &opcode,
                         GsyncLayout &g) {
  g.cls = gsync_class(opcode);
  if (g.cls == 0)
    return;
  const int words = gsync_operand_words(opcode);
  const int nops = instr->getNumOperands();
  int mref_index = -1, mref_count = 0;
  for (int i = 0; i < nops; ++i) {
    if (instr->getOperand(i)->type == InstrType::OperandType::MREF) {
      if (mref_index < 0)
        mref_index = i;
      ++mref_count;
    }
  }
  if (mref_count != 1)
    return;
  using OT = InstrType::OperandType;
  std::vector<GsyncOperand> dests_all;
  bool ok = true;
  for (int i = 0; i < nops && ok; ++i) {
    const InstrType::operand_t *op = instr->getOperand(i);
    if (op->type == OT::MREF || op->type == OT::MEM_DESC ||
        op->type == OT::TMA_PARAM_HANDLE)
      continue;
    if (op->type == OT::PRED) {
      if (op->u.pred.num == InstrType::PT)
        continue;
      ok = false;
      break;
    }
    const bool is_reg = op->type == OT::REG;
    const bool is_ureg = op->type == OT::UREG;
    if (i < mref_index) {
      if (!is_reg) {
        ok = false;
        break;
      }
      GsyncOperand d;
      d.is_zero = op->u.reg.num >= InstrType::RZ;
      d.base = op->u.reg.num;
      d.words = words;
      dests_all.push_back(d);
    } else {
      if (!is_reg && !is_ureg) {
        ok = false;
        break;
      }
      GsyncOperand s;
      s.is_ureg = is_ureg;
      s.is_zero = is_reg ? op->u.reg.num >= InstrType::RZ
                         : op->u.reg.num >= InstrType::URZ;
      s.base = op->u.reg.num;
      s.words = words;
      /* a UR source the backend cannot sample fails the whole shape */
      if (is_ureg && !s.is_zero &&
          !ureg_sampling_available(
              UregOperand{UregKind::Numeric, s.base}, words)) {
        ok = false;
        break;
      }
      g.sources.push_back(s);
    }
  }
  if (!ok)
    return;
  for (const auto &d : dests_all) {
    if (!d.is_zero)
      g.dests.push_back(d);
  }
  for (const auto &s : g.sources)
    g.source_words += s.words;
  for (const auto &d : g.dests)
    g.after_words += d.words;
  /* pre_value: RED-class shapes keep no result (REDG, or ATOM with all
   * destinations RZ) -- the lane's own mref0 address is read right
   * before the instruction executes.  Only 4B/8B pre reads exist. */
  if ((g.cls == 1 || g.cls == 2) && g.dests.empty() &&
      (words == 1 || words == 2)) {
    g.flags |= 1;
    g.pre_bytes = words * 4;
  }
  g.before_words = g.source_words + (g.flags & 1 ? g.pre_bytes / 4 : 0);
  g.supported = g.before_words <= V5_GSYNC_MAX_WORDS &&
                g.after_words <= V5_GSYNC_MAX_WORDS;
}

/* Value args for one instrument_gsync call: the operand words in order
 * (RZ/URZ -> const 0, UREG -> ureg_val, REG -> reg_val), zero-padded to
 * V5_GSYNC_MAX_WORDS. */
static void emit_gsync_value_args(Instr *instr,
                                  const std::vector<GsyncOperand> &ops) {
  int arg = 0;
  for (const auto &operand : ops) {
    for (int word = 0; word < operand.words && arg < V5_GSYNC_MAX_WORDS;
         ++word, ++arg) {
      if (operand.is_zero)
        nvbit_add_call_arg_const_val32(instr, 0);
      else if (operand.is_ureg)
        nvbit_add_call_arg_ureg_val(instr, operand.base + word);
      else
        nvbit_add_call_arg_reg_val(instr, operand.base + word);
    }
  }
  for (; arg < V5_GSYNC_MAX_WORDS; ++arg)
    nvbit_add_call_arg_const_val32(instr, 0);
}

/* Parse a PHASECHK destination predicate token (P<n> or UP<n>) out of the
 * SASS operand list.  Returns {kind, num}: 0=PRED, 1=UPRED, -1 none. */
static std::pair<int, int> phasechk_dest(Instr *instr) {
  std::vector<std::string> tokens;
  if (instr->getSass())
    sass_tokens(instr->getSass(), tokens);
  static const std::regex pred_re(R"(^(U?P)([0-9])$)");
  for (const auto &token : tokens) {
    std::smatch m;
    if (std::regex_match(token, m, pred_re))
      return {m[1] == "UP" ? 1 : 0, std::stoi(m[2])};
  }
  return {-1, -1};
}

/* ------------------------------------------------------------------ */
/* static SASS table (functions/<reg>.tsv)                             */
/* ------------------------------------------------------------------ */

struct ByteCollector {
  std::vector<uint8_t> bytes;
};

static void instr_byte_cb(uint8_t b, void *userdata) {
  ((ByteCollector *)userdata)->bytes.push_back(b);
}

static void write_function_tables(CUcontext ctx, CUfunction func,
                                  const FuncReg &reg) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/functions/%llu.tsv", user_folder.c_str(),
           (unsigned long long)reg.id);
  FILE *f = fopen(path, "w");
  if (!f)
    return;
#if !CYCLESCOPE_NVBIT_HAS_SASS_BINARY
  /* NVBit without Instr::getSassBinary: a small kernel reads the code
   * segment at function address + pc (16 bytes per instruction). */
  static uint8_t *g_code_read_dev = nullptr;
  if (!g_code_read_dev) {
    if (cudaMalloc(&g_code_read_dev, 16) != cudaSuccess)
      g_code_read_dev = nullptr;
  }
#endif
  int idx = 0;
  for (auto instr : nvbit_get_instrs(ctx, func)) {
    ByteCollector bc;
#if CYCLESCOPE_NVBIT_HAS_SASS_BINARY
    instr->getSassBinary(instr_byte_cb, &bc);
#else
    bc.bytes.assign(16, 0);
    if (g_code_read_dev) {
      read_code_bytes<<<1, 1>>>((const uint8_t *)reg.addr,
                              (uint64_t)instr->getOffset(),
                              g_code_read_dev);
      cudaMemcpy(bc.bytes.data(), g_code_read_dev, 16,
                 cudaMemcpyDeviceToHost);
    }
#endif
    char hex[64];
    size_t n = std::min(bc.bytes.size(), sizeof(hex) / 2 - 1);
    for (size_t i = 0; i < n; ++i)
      snprintf(hex + 2 * i, 3, "%02x", bc.bytes[i]);
    hex[2 * n] = '\0';
    const uint64_t key = (reg.id << 32) | (uint32_t)idx;
    auto it = g_layouts.find(key);
    std::string ops_field;
    if (it != g_layouts.end()) {
      for (size_t i = 0; i < it->second.ops.size(); ++i) {
        if (i)
          ops_field += ',';
        ops_field += it->second.ops[i].name;
      }
    }
    const int mrefs = it != g_layouts.end() ? it->second.mref_count : 0;
    const char *sass = instr->getSass() ? instr->getSass() : "";
    /* sanitize tabs/newlines out of sass text */
    std::string clean;
    for (const char *p = sass; *p; ++p)
      clean += (*p == '\t' || *p == '\n') ? ' ' : *p;
    fprintf(f, "%d\t%llx\t%s\t%s\t%s\t%d\n", idx,
            (unsigned long long)instr->getOffset(), hex, clean.c_str(),
            ops_field.c_str(), mrefs);
    ++idx;
  }
  fclose(f);
}

/* ------------------------------------------------------------------ */
/* instrumentation                                                     */
/* ------------------------------------------------------------------ */

/* Guard predicate argument + its flip flag.  nvbit_add_call_arg_guard_pred_val
 * does not deliver the UPx value of a uniform guard: one VOTEU-set UP0
 * guarded instructions captured as both exec 0 and exec full (CU-CUTE-04),
 * so @UPx/@!UPx guards read the designated uniform predicate instead and
 * the injected code applies the negation. */
static void add_guard_pred_args(Instr *instr) {
  if (instr->hasPred() && instr->isPredUniform()) {
    nvbit_add_call_arg_upred_val_at(instr, instr->getPredNum());
    nvbit_add_call_arg_const_val32(instr, instr->isPredNeg() ? 1 : 0);
  } else {
    nvbit_add_call_arg_guard_pred_val(instr);
    nvbit_add_call_arg_const_val32(instr, 0);
  }
}

/* ptxas can stash a GPR in the return-PC register:
 *   RPCMOV.32 Rpc.LO, Rx ; ... ; RPCMOV.32 Rx, Rpc.LO
 * Every injected call is a CALL/RET that overwrites Rpc, so a call placed
 * anywhere inside such a window destroys the stashed value (CU-C78/C81 and
 * FA-BWD crashed with "unspecified launch failure"; bisected to the first
 * instrumented instruction after the save).  Records of the instructions
 * after the save up to and including the restore are therefore emitted in
 * program order from IPOINT_AFTER calls on the restore.  This is exact only
 * for straight-line, unguarded instructions without captured operands or
 * memory references; any other window aborts the capture. */
static std::string sass_body(Instr *instr) {
  std::string sass = instr->getSass() ? instr->getSass() : "";
  static const std::regex guard(R"(^\s*@!?U?P[T0-9]+\s+)");
  return std::regex_replace(sass, guard, "");
}
static bool rpc_save(Instr *instr) {
  static const std::regex re(R"(^RPCMOV\S*\s+Rpc\.)");
  return std::regex_search(sass_body(instr), re);
}
static bool rpc_restore(Instr *instr) {
  static const std::regex re(R"(^RPCMOV\S*\s+R[0-9]+\s*,\s*Rpc\.)");
  return std::regex_search(sass_body(instr), re);
}
[[noreturn]] static void rpc_window_fail(const FuncReg &reg, uint32_t idx,
                                         const char *why) {
  fprintf(stderr,
          "cyclescope-tracer: unsupported deferred-record window in %s at instruction "
          "%u: %s\n",
          reg.name.c_str(), idx, why);
  fflush(stderr);
  abort();
}
static void insert_deferred_record(Instr *site, const FuncReg &reg,
                                   uint32_t idx) {
  nvbit_insert_call(site, "instrument_inst", IPOINT_AFTER);
  nvbit_add_call_arg_const_val32(site, 1);
  nvbit_add_call_arg_const_val32(site, 0);
  nvbit_add_call_arg_const_val32(site, (uint32_t)reg.id);
  nvbit_add_call_arg_const_val32(site, idx);
  nvbit_add_call_arg_const_val64(site, (uint64_t)&V5_DEV);
  nvbit_add_call_arg_const_val32(site, 0);
  for (int m = 0; m < V5_MAX_MREF; ++m)
    nvbit_add_call_arg_const_val64(site, 0);
  nvbit_add_call_arg_const_val64(site, 0);
  nvbit_add_call_arg_const_val64(site, 0);
  nvbit_add_call_arg_const_val32(site, 0);
  nvbit_add_call_arg_const_val32(site, 0);
  for (int i = 0; i < V5_MAX_OPS; ++i)
    nvbit_add_call_arg_const_val32(site, 0);
}

/* ops_meta packing shared by the live, stashed and deferred record paths. */
static void pack_ops_meta(const InstrLayout &layout, uint64_t &ops_meta,
                          uint64_t &ops_meta_hi, int &desc_map_ureg) {
  ops_meta = ops_meta_hi = 0;
  desc_map_ureg = -1;
  for (size_t i = 0; i < layout.ops.size() && i < V5_MAX_OPS; ++i) {
    const OpEntry &e = layout.ops[i];
    const bool takes_arg = (e.kind == V5_OP_R || e.kind == V5_OP_UR ||
                            e.kind == V5_OP_URANK || e.kind == V5_OP_BRANK ||
                            e.kind == V5_OP_CPOL || e.kind == V5_OP_CPOLH);
    uint64_t &m = i < 8 ? ops_meta : ops_meta_hi;
    const int si = i < 8 ? i : i - 8;
    m |= (uint64_t)(e.kind & 0xF) << (8 * si);
    m |= (uint64_t)(e.aux & 0x7) << (8 * si + 4);
    m |= (uint64_t)(takes_arg ? 1 : 0) << (8 * si + 7);
    if (e.kind == V5_OP_DESC)
      desc_map_ureg = e.reg_num;
  }
}
/* Operand value args read at `instr` (registers as of that instruction). */
static void add_value_args(Instr *instr, const InstrLayout &layout) {
  for (size_t i = 0; i < layout.ops.size() && i < V5_MAX_OPS; ++i) {
    const OpEntry &e = layout.ops[i];
    if (e.kind == V5_OP_R || e.kind == V5_OP_BRANK)
      nvbit_add_call_arg_reg_val(instr, e.reg_num);
    else if (e.kind == V5_OP_UR || e.kind == V5_OP_URANK ||
             e.kind == V5_OP_CPOL || e.kind == V5_OP_CPOLH)
      add_ureg_word(instr, e.reg_num);
    else if (e.name.rfind("BRVAL", 0) == 0) {
      if (e.reg2_num >= 0)
        add_ureg_word(instr, e.reg2_num);
      else
        nvbit_add_call_arg_const_val32(instr, (uint32_t)e.disp);
    } else
      nvbit_add_call_arg_const_val32(instr, 0);
  }
  for (size_t i = layout.ops.size(); i < V5_MAX_OPS; ++i)
    nvbit_add_call_arg_const_val32(instr, 0);
}
/* Operand-bearing instructions inside a WARPSYNC.COLLECTIVE region (e.g.
 * BAR.SYNC R0, R0 whose operand is built inside the region and rewritten
 * after it; ML-MLDEC-H64, tracer-bisect4) cannot take the collective
 * instrument_inst call there.  A collective-free stash_ops call saves each
 * lane's operand words at the instruction itself; the record is emitted
 * after ENDCOLLECTIVE by instrument_inst_stashed from those words. */
static constexpr int kMaxStashPerWindow = V5_STASH_SLOTS;
static void insert_stash_call(Instr *instr, const InstrLayout &layout,
                              int slot) {
  nvbit_insert_call(instr, "stash_ops", IPOINT_BEFORE);
  nvbit_add_call_arg_const_val32(instr, (uint32_t)slot);
  add_value_args(instr, layout);
}
static void insert_stashed_record(Instr *site, const FuncReg &reg,
                                  uint32_t idx, const InstrLayout &layout,
                                  int slot) {
  uint64_t ops_meta, ops_meta_hi;
  int desc_map_ureg;
  pack_ops_meta(layout, ops_meta, ops_meta_hi, desc_map_ureg);
  nvbit_insert_call(site, "instrument_inst_stashed", IPOINT_AFTER);
  nvbit_add_call_arg_const_val32(site, 1);
  nvbit_add_call_arg_const_val32(site, 0);
  nvbit_add_call_arg_const_val32(site, (uint32_t)reg.id);
  nvbit_add_call_arg_const_val32(site, idx);
  nvbit_add_call_arg_const_val64(site, (uint64_t)&V5_DEV);
  nvbit_add_call_arg_const_val32(site, 0);
  for (int m = 0; m < V5_MAX_MREF; ++m)
    nvbit_add_call_arg_const_val64(site, 0);
  nvbit_add_call_arg_const_val64(site, ops_meta);
  nvbit_add_call_arg_const_val64(site, ops_meta_hi);
  nvbit_add_call_arg_const_val32(site, 0);
  nvbit_add_call_arg_const_val32(site, 0);
  nvbit_add_call_arg_const_val32(site, (uint32_t)slot);
}

static void instrument_function(CUcontext ctx, CUfunction func, FuncReg &reg) {
  const std::vector<Instr *> &all = nvbit_get_instrs(ctx, func);
  if (capture_mode_placement) {
    /* v6 placement mode: one instrument_place call at the function's
     * first instruction (entry) and at every EXIT instruction; nothing
     * else is instrumented, no operand layouts are built. */
    for (size_t i = 0; i < all.size(); ++i) {
      Instr *instr = all[i];
      const std::string opcode =
          instr->getOpcode() ? instr->getOpcode() : "";
      const bool is_exit = opcode.rfind("EXIT", 0) == 0;
      if (i != 0 && !is_exit)
        continue;
      nvbit_insert_call(instr, "instrument_place", IPOINT_BEFORE);
      nvbit_add_call_arg_const_val32(instr, i == 0 ? 0 : 1);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&V5_DEV);
    }
    reg.instructions = (int)all.size();
    return;
  }
  /* v6.1 CTA info: one call at the function's first instruction; all
   * warps execute it, the lead lane pushes a V5P_CTA chunk.  Inserted
   * before the idx-0 instrument_inst call so the chunk precedes that
   * instruction's HEAD chunk in channel order. */
  if (!all.empty()) {
    nvbit_insert_call(all[0], "instrument_cta_info", IPOINT_BEFORE);
    nvbit_add_call_arg_const_val32(all[0], (uint32_t)reg.id);
    nvbit_add_call_arg_const_val64(all[0], (uint64_t)&V5_DEV);
  }
  /* idx -> restore idx for every instruction in (save, restore] */
  std::map<uint32_t, uint32_t> deferred_to;
  /* idx -> stash slot for operand-bearing instructions of collective windows */
  std::map<uint32_t, int> stash_slot;
  for (size_t i = 0; i < all.size(); ++i) {
    if (!rpc_save(all[i]))
      continue;
    size_t j = i + 1;
    while (j < all.size() && !rpc_restore(all[j])) {
      if (rpc_save(all[j]))
        rpc_window_fail(reg, (uint32_t)all[j]->getIdx(), "nested save");
      ++j;
    }
    if (j == all.size())
      rpc_window_fail(reg, (uint32_t)all[i]->getIdx(), "no restore");
    for (size_t k = i + 1; k <= j; ++k)
      deferred_to[(uint32_t)all[k]->getIdx()] = (uint32_t)all[j]->getIdx();
  }
  /* WARPSYNC.COLLECTIVE ... ENDCOLLECTIVE: ptxas's divergent fallback for
   * warp-synchronous code (__syncwarp/SHFL/VOTE).  An injected call (its
   * own __activemask/__ballot_sync) inside the region hung CU-C78-C06
   * (tracer-bisect3: first failing point = the NOP inside such a region).
   * Straight-line regions without guards, captured operands, memory or
   * control flow get the same after-the-end records; other regions keep
   * the previous per-instruction instrumentation. */
  static const std::regex coll_begin(R"(^WARPSYNC\.COLLECTIVE\b)");
  static const std::regex coll_end(R"(^ENDCOLLECTIVE\b)");
  static const std::regex cflow(
      R"(^(BRA|BRX|BRXU|JMP|JMX|CALL|RET|EXIT|BSSY|BSYNC|BREAK|WARPSYNC|BPT|KILL|NANOSLEEP|YIELD)\b)");
  for (size_t i = 0; i < all.size(); ++i) {
    if (!std::regex_search(sass_body(all[i]), coll_begin))
      continue;
    size_t j = i + 1;
    bool ok = true;
    std::vector<uint32_t> family_idx;
    while (j < all.size() && !std::regex_search(sass_body(all[j]), coll_end)) {
      const std::string op = all[j]->getOpcode() ? all[j]->getOpcode() : "";
      if (opcode_in_operand_family(op))
        family_idx.push_back((uint32_t)all[j]->getIdx());
      if (all[j]->hasPred() || std::regex_search(op, cflow) ||
          (all[j]->getMemorySpace() != InstrType::MemorySpace::NONE &&
           all[j]->getMemorySpace() != InstrType::MemorySpace::CONSTANT))
        ok = false;
      ++j;
    }
    if (j == all.size() || all[j]->hasPred() ||
        family_idx.size() > (size_t)kMaxStashPerWindow)
      ok = false;
    if (ok)
      for (size_t f = 0; f < family_idx.size(); ++f)
        stash_slot[family_idx[f]] = (int)f;
    if (ok)
      for (size_t k = i + 1; k <= j; ++k)
        if (!deferred_to.count((uint32_t)all[k]->getIdx()))
          deferred_to[(uint32_t)all[k]->getIdx()] = (uint32_t)all[j]->getIdx();
    i = j;
  }
  std::vector<uint32_t> pending;
  for (auto instr : all) {
    const std::string opcode = instr->getOpcode() ? instr->getOpcode() : "";
    bool skipped = false;
    for (const auto &prefix : g_skip_prefixes)
      if (opcode.rfind(prefix, 0) == 0)
        skipped = true;
    if (skipped)
      continue;
    if (g_idx_lo >= 0 && ((int64_t)instr->getIdx() < g_idx_lo ||
                          (int64_t)instr->getIdx() >= g_idx_hi))
      continue;
    const bool family = opcode_in_operand_family(opcode);
    InstrLayout layout;
    bool bare = false;
    for (const auto &prefix : g_bare_prefixes)
      if (opcode.rfind(prefix, 0) == 0)
        bare = true;
    if (!bare)
      build_layout(instr, opcode, family, layout);
    /* Prior-launch footprint: every MREF site that resolves to global
     * memory gets the footprint callback, independent of the operand
     * family -- the ring covers the whole launch.  TMA/bulk sites are
     * skipped (their addresses come from descriptors/uniform registers,
     * not MREF lanes); deferred-window instructions take no calls at
     * all (a CALL inside an RPC stash window destroys the register). */
    if (opcode_is_bulk_transfer(opcode))
      g_func_has_bulk_ops[func] = true;
    if (g_footprint_enabled && g_footprint_kernel &&
        !opcode_is_bulk_transfer(opcode) &&
        deferred_to.find((uint32_t)instr->getIdx()) == deferred_to.end()) {
      const InstrType::MemorySpace space = instr->getMemorySpace();
      const bool generic = space == InstrType::MemorySpace::GENERIC;
      if (generic || space == InstrType::MemorySpace::GLOBAL ||
          space == InstrType::MemorySpace::GLOBAL_TO_SHARED) {
        const int32_t is_store =
            (instr->isStore() || opcode.rfind("ATOM", 0) == 0 ||
             opcode.rfind("RED", 0) == 0)
                ? 1
                : 0;
        int mref_ordinal = 0;
        for (int oi = 0; oi < instr->getNumOperands(); ++oi) {
          if (instr->getOperand(oi)->type != InstrType::OperandType::MREF)
            continue;
          nvbit_insert_call(instr, "instrument_l2_footprint",
                            IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(instr);
          nvbit_add_call_arg_mref_addr64(instr, mref_ordinal);
          nvbit_add_call_arg_const_val32(instr, is_store);
          nvbit_add_call_arg_const_val32(instr, generic ? 1 : 0);
          nvbit_add_call_arg_const_val64(instr,
                                         (uint64_t)&g_footprint_cell);
          ++mref_ordinal;
        }
      }
    }
    const auto matches = [&](const std::vector<std::string> &prefixes) {
      for (const auto &prefix : prefixes)
        if (opcode.rfind(prefix, 0) == 0)
          return true;
      return false;
    };
    const bool no_mref = matches(g_nomref_prefixes);
    const bool no_rank = no_mref || matches(g_norank_prefixes);
    const bool no_xrank = matches(g_noxrank_prefixes);
    const bool no_ur = matches(g_nour_prefixes);
    const bool no_r = matches(g_nor_prefixes);
    if (no_mref)
      layout.mref_count = 0;
    if (no_rank || no_xrank || no_ur || no_r) {
      std::vector<OpEntry> kept;
      for (const auto &e : layout.ops) {
        if (no_rank && e.kind == V5_OP_RANK)
          continue;
        if (no_ur && e.kind == V5_OP_UR)
          continue;
        if (no_r && e.kind == V5_OP_R)
          continue;
        if (no_xrank && (e.kind == V5_OP_URANK || e.kind == V5_OP_BRANK ||
                         e.name.rfind("BRVAL", 0) == 0))
          continue;
        kept.push_back(e);
      }
      layout.ops = kept;
    }
    if (!bare) {
      gsync_layout(instr, opcode, layout.gsync);
      if (layout.gsync.cls != 0 && !layout.gsync.supported)
        fprintf(stderr,
                "cyclescope-tracer gsync: unsupported operand shape reg=%u idx=%u "
                "sass=%s\n",
                (unsigned)reg.id, (unsigned)instr->getIdx(),
                instr->getSass() ? instr->getSass() : "");
    }
    const uint32_t idx = (uint32_t)instr->getIdx();
    for (size_t oi = 0; oi < layout.ops.size(); ++oi)
      layout.ops[oi].slot = (int)(oi * 32);
    g_layouts[(reg.id << 32) | idx] = layout;

    const auto deferred = deferred_to.find(idx);
    if (deferred != deferred_to.end()) {
      if (instr->hasPred())
        rpc_window_fail(reg, idx, "guarded instruction");
      const auto stash = stash_slot.find(idx);
      if (layout.mref_count != 0)
        rpc_window_fail(reg, idx, "instruction with memory references");
      if (!layout.ops.empty() && stash == stash_slot.end())
        rpc_window_fail(reg, idx, "instruction with captured operands");
      for (const auto &e : layout.ops)
        if (e.kind == V5_OP_DESC || e.kind == V5_OP_RANK)
          rpc_window_fail(reg, idx, "stashed operand kind needs live state");
      if (stash != stash_slot.end())
        insert_stash_call(instr, layout, stash->second);
      if (instr->getMemorySpace() != InstrType::MemorySpace::NONE &&
          instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT)
        rpc_window_fail(reg, idx, "memory instruction");
      static const std::regex branch(
          R"(^(BRA|BRX|BRXU|JMP|JMX|CALL|RET|EXIT|BSSY|BSYNC|BREAK|WARPSYNC|BPT|KILL|NANOSLEEP|YIELD)\b)");
      if (std::regex_search(opcode, branch))
        rpc_window_fail(reg, idx, "control flow");
      pending.push_back(idx);
      if (deferred->second == idx) {
        for (uint32_t k : pending) {
          const auto st = stash_slot.find(k);
          const auto &lk = g_layouts[(reg.id << 32) | k];
          if (st != stash_slot.end() && !lk.ops.empty())
            insert_stashed_record(instr, reg, k, lk, st->second);
          else
            insert_deferred_record(instr, reg, k);
        }
        pending.clear();
      }
      continue;
    }

    nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
    add_guard_pred_args(instr);
    nvbit_add_call_arg_const_val32(instr, (uint32_t)reg.id);
    nvbit_add_call_arg_const_val32(instr, idx);
    nvbit_add_call_arg_const_val64(instr, (uint64_t)&V5_DEV);
    nvbit_add_call_arg_const_val32(instr, layout.mref_count);
    for (int m = 0; m < V5_MAX_MREF; ++m) {
      if (m < layout.mref_count)
        nvbit_add_call_arg_mref_addr64(instr, m);
      else
        nvbit_add_call_arg_const_val64(instr, 0);
    }
    /* ops_meta: kind(3) | aux(4) | consumes-arg(1) per slot; slots 8..9
     * go into ops_meta_hi. */
    uint64_t ops_meta, ops_meta_hi;
    int desc_map_ureg;
    pack_ops_meta(layout, ops_meta, ops_meta_hi, desc_map_ureg);
    nvbit_add_call_arg_const_val64(instr, ops_meta);
    nvbit_add_call_arg_const_val64(instr, ops_meta_hi);
    /* DESC pointer args: the map UR pair values. */
    if (desc_map_ureg >= 0) {
      add_ureg_word(instr, desc_map_ureg);
      add_ureg_word(instr, desc_map_ureg + 1);
    } else {
      nvbit_add_call_arg_const_val32(instr, 0);
      nvbit_add_call_arg_const_val32(instr, 0);
    }
    /* operand values; BRANK/URANK aux slot carries the UR addend or disp */
    add_value_args(instr, layout);

    /* v5.4 global-sync value hooks: BEFORE carries the source operand
     * words + optional pre_value, AFTER the destination words.  Inserted
     * after the instrument_inst BEFORE call so the V5P_GSYNC chunk lands
     * behind this instruction's HEAD/MREF/OP chunks (NVBit runs the calls
     * at one instrumentation point in insertion order). */
    if (!no_gsync && layout.gsync.cls != 0 && layout.gsync.supported) {
      const GsyncLayout &g = layout.gsync;
      if (g.before_words > 0) {
        nvbit_insert_call(instr, "instrument_gsync", IPOINT_BEFORE);
        add_guard_pred_args(instr);
        nvbit_add_call_arg_const_val32(instr, (uint32_t)reg.id);
        nvbit_add_call_arg_const_val32(instr, idx);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&V5_DEV);
        nvbit_add_call_arg_const_val32(instr, 0); /* phase: before */
        nvbit_add_call_arg_const_val32(instr, g.before_words);
        nvbit_add_call_arg_const_val32(instr, g.flags);
        nvbit_add_call_arg_const_val32(instr, g.pre_bytes);
        nvbit_add_call_arg_mref_addr64(instr, 0);
        emit_gsync_value_args(instr, g.sources);
      }
      if (g.after_words > 0) {
        nvbit_insert_call(instr, "instrument_gsync", IPOINT_AFTER);
        add_guard_pred_args(instr);
        nvbit_add_call_arg_const_val32(instr, (uint32_t)reg.id);
        nvbit_add_call_arg_const_val32(instr, idx);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&V5_DEV);
        nvbit_add_call_arg_const_val32(instr, 1); /* phase: after */
        nvbit_add_call_arg_const_val32(instr, g.after_words);
        nvbit_add_call_arg_const_val32(instr, 0);
        nvbit_add_call_arg_const_val32(instr, 0);
        nvbit_add_call_arg_const_val64(instr, 0);
        emit_gsync_value_args(instr, g.dests);
      }
    }

    /* PHASECHK AFTER record (dest predicate ballot). */
    if (!no_after_hook && (opcode.rfind("SYNCS.PHASECHK", 0) == 0 ||
                           opcode.find("PHASECHK") != std::string::npos)) {
      const auto [dest_kind, dest_num] = phasechk_dest(instr);
      if (dest_kind >= 0) {
        nvbit_insert_call(instr, "instrument_phasechk_after", IPOINT_AFTER);
        add_guard_pred_args(instr);
        if (dest_kind == 0)
          nvbit_add_call_arg_pred_val_at(instr, dest_num);
        else
          nvbit_add_call_arg_upred_val_at(instr, dest_num);
        nvbit_add_call_arg_const_val32(instr, (uint32_t)reg.id);
        nvbit_add_call_arg_const_val32(instr, idx);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&V5_DEV);
      }
    }
  }
  if (!pending.empty())
    rpc_window_fail(reg, pending.front(), "window records not emitted");
  reg.instructions = (int)nvbit_get_instrs(ctx, func).size();
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  if (g_instrumented.count(func))
    return;
  std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
  related.push_back(func);
  uint64_t entry_id = 0;
  for (auto f : related) {
    if (!g_instrumented.insert(f).second)
      continue;
    FuncReg reg;
    reg.id = g_next_reg_id++;
    reg.ctx = ctx;
    reg.addr = (uint64_t)nvbit_get_func_addr(ctx, f);
    reg.related_of = (f == func) ? 0 : entry_id;
    reg.name = nvbit_get_func_name(ctx, f) ? nvbit_get_func_name(ctx, f) : "";
    reg.module = g_module_of_function.count(f) ? g_module_of_function[f] : nullptr;
    instrument_function(ctx, f, reg);
    g_func_regs[f] = reg;
    if (f == func)
      entry_id = reg.id;
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/functions", user_folder.c_str());
    mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    write_function_tables(ctx, f, reg);
    /* functions.jsonl row */
    if (g_functions_jsonl) {
      fprintf(g_functions_jsonl, "{\"function_id\": %llu, \"name\": ",
              (unsigned long long)reg.id);
      json_escape(g_functions_jsonl, reg.name);
      fprintf(g_functions_jsonl, ", \"ctx\": \"0x%lx\", \"module\": ",
              (unsigned long)(uintptr_t)ctx);
      if (reg.module)
        fprintf(g_functions_jsonl, "\"0x%lx\"",
                (unsigned long)(uintptr_t)reg.module);
      else
        fprintf(g_functions_jsonl, "null");
      fprintf(g_functions_jsonl,
              ", \"addr\": \"0x%llx\", \"instructions\": %d, \"related_of\": ",
              (unsigned long long)reg.addr, reg.instructions);
      if (reg.related_of)
        fprintf(g_functions_jsonl, "%llu",
                (unsigned long long)reg.related_of);
      else
        fprintf(g_functions_jsonl, "null");
      fprintf(g_functions_jsonl, "}\n");
      fflush(g_functions_jsonl);
    }
  }
}

/* ------------------------------------------------------------------ */
/* TMA encode interception -> tensor_maps.jsonl encode field              */
/* ------------------------------------------------------------------ */

struct TmaEncodeFacts {
  std::string api;
  int data_type = -1;
  std::vector<uint64_t> global_dim, global_strides, box_dim, element_strides;
  int interleave = -1, swizzle = -1, l2_promotion = -1, oob_fill = -1;
  /* im2col only: bounding-box corners, rank - 2 spatial entries each, in
   * the API array order (cuTensorMapEncodeIm2col pixelBox*Corner) */
  std::vector<int> pixel_box_lower_corner, pixel_box_upper_corner;
};

using TmaImage = std::array<uint8_t, 128>;
static std::map<TmaImage, TmaEncodeFacts> g_encode_facts;
static std::map<TmaImage, uint32_t> g_desc_ids;
static uint32_t g_next_desc_id = 1;
static std::mutex *g_tma_mutex = nullptr;

static std::string tma_image_hex(const TmaImage &image) {
  std::string hex;
  hex.reserve(256);
  char b[3];
  for (uint8_t v : image) {
    snprintf(b, sizeof(b), "%02x", v);
    hex += b;
  }
  return hex;
}

static void write_desc_jsonl(uint32_t id, const TmaImage &image) {
  if (!g_tensor_maps_jsonl)
    return;
  fprintf(g_tensor_maps_jsonl, "{\"id\": %u, \"image\": \"%s\", \"encode\": ",
          id, tma_image_hex(image).c_str());
  auto it = g_encode_facts.find(image);
  if (it == g_encode_facts.end()) {
    fprintf(g_tensor_maps_jsonl, "null}\n");
  } else {
    const TmaEncodeFacts &e = it->second;
    auto u64list = [](FILE *f, const std::vector<uint64_t> &v) {
      fputc('[', f);
      for (size_t i = 0; i < v.size(); ++i)
        fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)v[i]);
      fputc(']', f);
    };
    fprintf(g_tensor_maps_jsonl, "{\"api\": \"%s\", \"data_type\": %d, \"rank\": %zu, "
            "\"global_dim\": ", e.api.c_str(), e.data_type,
            e.global_dim.size());
    u64list(g_tensor_maps_jsonl, e.global_dim);
    fprintf(g_tensor_maps_jsonl, ", \"global_strides\": ");
    u64list(g_tensor_maps_jsonl, e.global_strides);
    fprintf(g_tensor_maps_jsonl, ", \"box_dim\": ");
    u64list(g_tensor_maps_jsonl, e.box_dim);
    fprintf(g_tensor_maps_jsonl, ", \"element_strides\": ");
    u64list(g_tensor_maps_jsonl, e.element_strides);
    fprintf(g_tensor_maps_jsonl,
            ", \"interleave\": %d, \"swizzle\": %d, \"l2_promotion\": %d, "
            "\"oob_fill\": %d",
            e.interleave, e.swizzle, e.l2_promotion, e.oob_fill);
    if (e.api == "im2col") {
      auto ilist = [](FILE *f, const std::vector<int> &v) {
        fputc('[', f);
        for (size_t i = 0; i < v.size(); ++i)
          fprintf(f, "%s%d", i ? "," : "", v[i]);
        fputc(']', f);
      };
      fprintf(g_tensor_maps_jsonl, ", \"pixel_box_lower_corner\": ");
      ilist(g_tensor_maps_jsonl, e.pixel_box_lower_corner);
      fprintf(g_tensor_maps_jsonl, ", \"pixel_box_upper_corner\": ");
      ilist(g_tensor_maps_jsonl, e.pixel_box_upper_corner);
    }
    fprintf(g_tensor_maps_jsonl, "}}\n");
  }
  fflush(g_tensor_maps_jsonl);
}

static uint32_t desc_id_of(const TmaImage &image) {
  auto it = g_desc_ids.find(image);
  if (it != g_desc_ids.end())
    return it->second;
  const uint32_t id = g_next_desc_id++;
  g_desc_ids[image] = id;
  write_desc_jsonl(id, image);
  return id;
}

static void tma_record_encode(const CUtensorMap *tensor_map, int data_type,
                              uint32_t rank, const uint64_t *gdim,
                              const uint64_t *gstrides, const uint32_t *bdim,
                              const uint32_t *estrides, uint32_t interleave,
                              uint32_t swizzle, uint32_t l2, uint32_t oob,
                              const char *api) {
  if (!tensor_map)
    return;
  TmaImage image;
  memcpy(image.data(), tensor_map, image.size());
  TmaEncodeFacts e;
  e.api = api;
  e.data_type = data_type;
  for (uint32_t i = 0; i < rank && i < 5; ++i) {
    e.global_dim.push_back(gdim[i]);
    if (i > 0)
      e.global_strides.push_back(gstrides[i - 1]);
    e.box_dim.push_back(bdim[i]);
    e.element_strides.push_back(estrides[i]);
  }
  e.interleave = interleave;
  e.swizzle = swizzle;
  e.l2_promotion = l2;
  e.oob_fill = oob;
  std::lock_guard<std::mutex> guard(*g_tma_mutex);
  g_encode_facts[image] = e;
  /* if this image was already assigned an id, rewrite its row */
  auto dit = g_desc_ids.find(image);
  if (dit != g_desc_ids.end())
    write_desc_jsonl(dit->second, image);
}

static void tma_intercept_encode(nvbit_api_cuda_t cbid, void *params,
                                 CUresult *pStatus) {
  if (pStatus && *pStatus != CUDA_SUCCESS)
    return;
  if (cbid == API_CUDA_cuTensorMapEncodeTiled) {
    auto *p = (cuTensorMapEncodeTiled_params *)params;
    tma_record_encode(p->tensorMap, p->tensorDataType, p->tensorRank,
                      p->globalDim, p->globalStrides, p->boxDim,
                      p->elementStrides, p->interleave, p->swizzle,
                      p->l2Promotion, p->oobFill, "tiled");
  } else if (cbid == API_CUDA_cuTensorMapEncodeIm2col) {
    auto *p = (cuTensorMapEncodeIm2col_params *)params;
    TmaImage image;
    memcpy(image.data(), p->tensorMap, image.size());
    TmaEncodeFacts e;
    e.api = "im2col";
    e.data_type = p->tensorDataType;
    for (uint32_t i = 0; i < p->tensorRank && i < 5; ++i) {
      e.global_dim.push_back(p->globalDim[i]);
      if (i > 0)
        e.global_strides.push_back(p->globalStrides[i - 1]);
      e.element_strides.push_back(p->elementStrides[i]);
    }
    e.box_dim.push_back(p->channelsPerPixel);
    e.box_dim.push_back(p->pixelsPerColumn);
    /* no corner arrays -> empty lists (a missing fact downstream) */
    for (uint32_t i = 0; i + 2 < p->tensorRank && i < 3 &&
                         p->pixelBoxLowerCorner && p->pixelBoxUpperCorner;
         ++i) {
      e.pixel_box_lower_corner.push_back(p->pixelBoxLowerCorner[i]);
      e.pixel_box_upper_corner.push_back(p->pixelBoxUpperCorner[i]);
    }
    e.interleave = p->interleave;
    e.swizzle = p->swizzle;
    e.l2_promotion = p->l2Promotion;
    e.oob_fill = p->oobFill;
    std::lock_guard<std::mutex> guard(*g_tma_mutex);
    g_encode_facts[image] = e;
    auto dit = g_desc_ids.find(image);
    if (dit != g_desc_ids.end())
      write_desc_jsonl(dit->second, image);
  }
}

/* replace-address: the host call mutates the image bytes; record the new
 * image carrying the previous image's encode facts. */
static void tma_intercept_host_replace(bool is_exit, void *params,
                                       CUresult *pStatus) {
  auto *p = static_cast<cuTensorMapReplaceAddress_params *>(params);
  static std::map<const void *, TmaImage> pending;
  if (!is_exit) {
    if (p->tensorMap) {
      TmaImage before;
      memcpy(before.data(), p->tensorMap, before.size());
      pending[p->tensorMap] = before;
    }
    return;
  }
  if (pStatus && *pStatus != CUDA_SUCCESS)
    return;
  auto it = pending.find(p->tensorMap);
  if (it == pending.end())
    return;
  TmaImage after;
  memcpy(after.data(), p->tensorMap, after.size());
  std::lock_guard<std::mutex> guard(*g_tma_mutex);
  auto e = g_encode_facts.find(it->second);
  if (e != g_encode_facts.end())
    g_encode_facts[after] = e->second;
  pending.erase(it);
}

/* private host-side encode entry reached via cuGetProcAddress */
static std::vector<std::pair<void *, void *>> g_tma_orig_fns;
using TmaEncodeTiledFn = decltype(&cuTensorMapEncodeTiled);
static TmaEncodeTiledFn g_orig_encode_tiled = nullptr;

extern "C" CUresult cuTensorMapEncodeTiled_hook(
    CUtensorMap *tensor_map, CUtensorMapDataType data_type, cuuint32_t rank,
    void *global_address, const cuuint64_t *global_dim,
    const cuuint64_t *global_strides, const cuuint32_t *box_dim,
    const cuuint32_t *element_strides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  CUresult rc = g_orig_encode_tiled
                    ? g_orig_encode_tiled(tensor_map, data_type, rank,
                                          global_address, global_dim,
                                          global_strides, box_dim,
                                          element_strides, interleave,
                                          swizzle, l2, oob)
                    : CUDA_ERROR_UNKNOWN;
  if (rc == CUDA_SUCCESS)
    tma_record_encode(tensor_map, data_type, rank, global_dim, global_strides,
                      box_dim, element_strides, interleave, swizzle, l2, oob,
                      "tiled");
  return rc;
}

static void tma_intercept_driver_entry(const char *symbol, void **pfn) {
  if ((strcmp(symbol, "cuTensorMapEncodeTiled") == 0 ||
       strcmp(symbol, "_cuTensorMapEncodeTiled") == 0) &&
      g_orig_encode_tiled == nullptr) {
    g_orig_encode_tiled = (TmaEncodeTiledFn)*pfn;
    *pfn = (void *)&cuTensorMapEncodeTiled_hook;
  }
}

/* ------------------------------------------------------------------ */
/* launch selection / cluster sampling                                 */
/* ------------------------------------------------------------------ */

/* Two-level selection: `launch_pattern_matches` is true whenever the
 * launch's identity matches the selection tokens ignoring the regex@N
 * occurrence gate (drives the prior-identical-launch counters, matching
 * the hopper semantics where identical counting covered every matching
 * launch); `launch_selected` additionally applies the ordinal/occurrence
 * gate (drives tracing). */
static bool launch_pattern_matches(uint64_t ordinal,
                                   const std::string &name,
                                   const std::string &mangled_name) {
  if (capture_launches == "all" || capture_launches.empty())
    return true;
  std::stringstream ss(capture_launches);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty())
      continue;
    bool digits = std::all_of(token.begin(), token.end(), ::isdigit);
    if (digits) {
      if (std::stoull(token) == ordinal)
        return true;
      continue;
    }
    std::string pattern = token;
    const size_t at = token.rfind('@');
    if (at != std::string::npos && at + 1 < token.size() &&
        std::all_of(token.begin() + at + 1, token.end(), ::isdigit))
      pattern = token.substr(0, at);
    try {
      std::regex::flag_type flags = std::regex::ECMAScript;
      if (pattern.rfind("(?i)", 0) == 0) {
        pattern = pattern.substr(4);
        flags |= std::regex::icase;
      }
      const std::regex re(pattern, flags);
      if (std::regex_match(name, re) ||
          (!mangled_name.empty() && std::regex_match(mangled_name, re)))
        return true;
    } catch (const std::regex_error &) {
    }
  }
  return false;
}

static bool launch_selected(uint64_t ordinal, const std::string &name,
                            const std::string &mangled_name) {
  if (capture_launches == "all" || capture_launches.empty())
    return true;
  std::stringstream ss(capture_launches);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty())
      continue;
    bool digits = std::all_of(token.begin(), token.end(), ::isdigit);
    if (digits) {
      if (std::stoull(token) == ordinal)
        return true;
      continue;
    }
    /* regex@N: the launch where the regex matches for the Nth time
     * (1-based, counted across the whole run). */
    uint64_t occurrence = 0;
    std::string pattern = token;
    const size_t at = token.rfind('@');
    if (at != std::string::npos && at + 1 < token.size() &&
        std::all_of(token.begin() + at + 1, token.end(), ::isdigit)) {
      occurrence = std::stoull(token.substr(at + 1));
      pattern = token.substr(0, at);
    }
    try {
      /* case specs write main-kernel regexes in Python syntax; a
       * leading (?i) maps to ECMAScript + icase.  The regex is tried
       * against both the demangled and the mangled name. */
      std::regex::flag_type flags = std::regex::ECMAScript;
      if (pattern.rfind("(?i)", 0) == 0) {
        pattern = pattern.substr(4);
        flags |= std::regex::icase;
      }
      const std::regex re(pattern, flags);
      if (std::regex_match(name, re) ||
          (!mangled_name.empty() && std::regex_match(mangled_name, re))) {
        if (occurrence == 0)
          return true;
        const uint64_t seen = ++g_regex_match_counts[token];
        if (seen == occurrence)
          return true;
      }
    } catch (const std::regex_error &) {
    }
  }
  return false;
}

/* uniform sample of `total` cluster indices, endpoints included. */
static std::vector<uint32_t> uniform_cluster_sample(uint32_t total, int n) {
  std::vector<uint32_t> out;
  if (n <= 0 || total == 0)
    return out;
  if ((uint32_t)n >= total) {
    out.resize(total);
    for (uint32_t i = 0; i < total; ++i)
      out[i] = i;
    return out;
  }
  for (int i = 0; i < n; ++i) {
    uint32_t idx = (uint32_t)llround((double)i * (total - 1) / (n - 1));
    out.push_back(idx);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/* ------------------------------------------------------------------ */
/* recv thread: parse channel records and write the trace file         */
/* ------------------------------------------------------------------ */

struct LaunchOut {
  FILE *f = nullptr;       /* trace file (or zstd pipe) */
  std::string path;
  bool via_pipe = false;
};

static LaunchOut g_launch_out;
static uint64_t g_records_written = 0;

/* Chunk reassembly: chunks of one record arrive in channel order per
 * (cta_x, warp) producer.  The head chunk opens a record; mref/op chunks
 * append in static layout order; a V5P_AFTER chunk is a complete record.
 * A new head chunk or stream end flushes the pending record. */
struct V5Pending {
  bool open = false;
  V5Head head{};
  std::vector<uint8_t> tail;
  const InstrLayout *layout = nullptr;
  size_t next_mref = 0;      /* mref chunks expected */
  size_t next_op = 0;        /* next operand-entry index expected */
};

using V5PendingKey =
    std::tuple<uint32_t, uint16_t, uint16_t, uint16_t>; /* cta xyz + warp */

static V5PendingKey pending_key(const V5ChunkHead *ch) {
  return {ch->cta_x, ch->cta_y, ch->cta_z, ch->warp};
}

static std::map<V5PendingKey, V5Pending> g_pending;
/* Guards g_pending, g_layouts and g_launch_out between the receive thread
 * and the launch hooks.  Unsynchronized flush/close at kernel exit while
 * the receiver still processed (or inserted into g_pending) corrupted the
 * heap after a few hundred launches (CU-C81-BW-1C-C01: "free(): invalid
 * pointer" / SIGSEGV mid-run, capture-r8-rpc). */
static std::mutex g_record_mutex;

static void write_record(const uint8_t *rec, size_t len) {
  if (g_launch_out.f)
    fwrite(rec, 1, len, g_launch_out.f);
  ++g_records_written;
}

static void flush_pending(V5Pending &pend) {
  if (pend.open) {
    std::vector<uint8_t> rec((const uint8_t *)&pend.head,
                             (const uint8_t *)&pend.head + sizeof(V5Head));
    rec.insert(rec.end(), pend.tail.begin(), pend.tail.end());
    write_record(rec.data(), rec.size());
  }
  pend = V5Pending();
}

/* Append a chunk body to the pending record's tail, honoring the static
 * operand layout (missing entries are skipped to keep the byte stream
 * aligned). */
static void append_op_chunk(V5Pending &pend, const V5OpBody *body,
                            size_t body_bytes) {
  const uint8_t entry = body->entry;
  const InstrLayout &layout = *pend.layout;
  while (pend.next_op < layout.ops.size() &&
         (int)pend.next_op < (int)entry) {
    /* a chunk for an earlier entry was never pushed (unsampleable) --
     * emit its zero-width part is impossible; entries are byte-ordered
     * so only advance. */
    ++pend.next_op;
  }
  if (pend.next_op >= layout.ops.size())
    return;
  const OpEntry &e = layout.ops[entry];
  const int exec_count = __builtin_popcount(pend.head.exec_mask);
  if (e.kind == V5_OP_R || e.kind == V5_OP_RANK || e.kind == V5_OP_BRANK) {
    const size_t want = (size_t)exec_count * 4;
    pend.tail.insert(pend.tail.end(), (const uint8_t *)body->vals,
                     (const uint8_t *)body->vals + std::min(want, body_bytes));
  } else if (e.kind == V5_OP_UR || e.kind == V5_OP_URANK ||
             e.kind == V5_OP_CPOL || e.kind == V5_OP_CPOLH) {
    pend.tail.insert(pend.tail.end(), (const uint8_t *)body->vals,
                     (const uint8_t *)body->vals + 4);
  } else if (e.kind == V5_OP_SPACE) {
    pend.tail.insert(pend.tail.end(), (const uint8_t *)body->vals,
                     (const uint8_t *)body->vals +
                         std::min((size_t)12, body_bytes));
  } else if (e.kind == V5_OP_DESC) {
    if (body_bytes >= 4 + 128 && body->vals[0] == 0xFFFFFFFFu) {
      TmaImage image;
      memcpy(image.data(), body->vals + 1, 128);
      const uint32_t id = desc_id_of(image);
      pend.tail.insert(pend.tail.end(), (const uint8_t *)&id,
                       (const uint8_t *)&id + 4);
    } else {
      const uint32_t zero = 0;
      pend.tail.insert(pend.tail.end(), (const uint8_t *)&zero,
                       (const uint8_t *)&zero + 4);
    }
  }
  ++pend.next_op;
}

/* Placement output: one placement.jsonl per capture directory, appended
 * per traced launch (each row carries its launch ordinal). */
static FILE *placement_out() {
  static FILE *f = nullptr;
  if (!f && capture_mode_placement)
    f = fopen((user_folder + "/placement.jsonl").c_str(), "a");
  return f;
}

static void write_placement_json(const V5ChunkHead *ch,
                                 const V5PlaceBody *pb) {
  FILE *f = placement_out();
  if (!f)
    return;
  fprintf(f,
          "{\"launch\": %u, \"cta\": [%u,%u,%u], \"warp\": %u, "
          "\"smid\": %u, \"warpid\": %u, \"globaltimer\": %llu, "
          "\"clock64\": %llu, \"kind\": \"%s\"}\n",
          ch->idx, ch->cta_x, ch->cta_y, ch->cta_z, ch->warp, pb->smid,
          pb->warpid, (unsigned long long)pb->globaltimer,
          (unsigned long long)pb->clock64,
          ch->kind == 0 ? "entry" : "exit");
  fflush(f);
  ++g_records_written;
}

/* v6 placement array (device side, see V5DevState::place_buf).  Grown on
 * demand; zeroed before every traced launch; copied out in
 * leave_kernel_launch and written as placement-<ord>.jsonl. */
static V5PlaceRec *g_place_dev = nullptr;
static uint32_t g_place_dev_capacity = 0;
static uint32_t g_place_records = 0;   /* records armed for the current launch */
static uint64_t g_place_ordinal = 0;

static void arm_placement_array(uint64_t ordinal,
                                const std::array<uint32_t, 3> &grid,
                                const std::array<uint32_t, 3> &block) {
  const uint64_t ctas = (uint64_t)grid[0] * grid[1] * grid[2];
  const uint32_t threads = block[0] * block[1] * block[2];
  const uint32_t wpc = (threads + 31u) / 32u;
  const uint64_t need = ctas * wpc * 2u;
  if (need == 0 || need > (1ull << 31)) {
    V5_DEV.place_buf = nullptr;
    V5_DEV.place_capacity = 0;
    g_place_records = 0;
    fprintf(stderr, "cyclescope-tracer placement: launch %llu grid too large (%llu records)\n",
            (unsigned long long)ordinal, (unsigned long long)need);
    return;
  }
  if (need > g_place_dev_capacity) {
    if (g_place_dev)
      cudaFree(g_place_dev);
    g_place_dev = nullptr;
    g_place_dev_capacity = 0;
    if (cudaMalloc(&g_place_dev, need * sizeof(V5PlaceRec)) != cudaSuccess) {
      fprintf(stderr, "cyclescope-tracer placement: cudaMalloc(%llu records) failed\n",
              (unsigned long long)need);
      g_place_dev = nullptr;
      V5_DEV.place_buf = nullptr;
      V5_DEV.place_capacity = 0;
      g_place_records = 0;
      return;
    }
    g_place_dev_capacity = (uint32_t)need;
  }
  cudaMemset(g_place_dev, 0, need * sizeof(V5PlaceRec));
  V5_DEV.place_buf = g_place_dev;
  V5_DEV.place_capacity = (uint32_t)need;
  V5_DEV.place_warps_per_cta = wpc;
  g_place_records = (uint32_t)need;
  g_place_ordinal = ordinal;
}

static void dump_placement_array() {
  if (g_place_records == 0 || g_place_dev == nullptr)
    return;
  std::vector<V5PlaceRec> host(g_place_records);
  if (cudaMemcpy(host.data(), g_place_dev, g_place_records * sizeof(V5PlaceRec),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "cyclescope-tracer placement: copy-out failed for launch %llu\n",
            (unsigned long long)g_place_ordinal);
    return;
  }
  FILE *f = placement_out();
  if (!f)
    return;
  for (uint32_t i = 0; i < g_place_records; ++i) {
    const V5PlaceRec &r = host[i];
    if (!r.valid)
      continue;
    fprintf(f,
            "{\"launch\": %llu, \"cta\": [%u,%u,%u], \"warp\": %u, "
            "\"smid\": %u, \"warpid\": %u, \"globaltimer\": %llu, "
            "\"clock64\": %llu, \"kind\": \"%s\"}\n",
            (unsigned long long)g_place_ordinal, r.cta_x, r.cta_y, r.cta_z,
            r.warp, r.smid, r.warpid, (unsigned long long)r.globaltimer,
            (unsigned long long)r.clock64, (i & 1u) == 0 ? "entry" : "exit");
    ++g_records_written;
  }
  fflush(f);  /* placement_out() caches the FILE*; closing it here left
                 a dangling stream that ctx_term's fflush aborted on */
  g_place_records = 0;
  V5_DEV.place_buf = nullptr;
  V5_DEV.place_capacity = 0;
}

static void process_chunk(const uint8_t *buf, size_t len) {
  if (len < sizeof(V5ChunkHead))
    return;
  const V5ChunkHead *ch = (const V5ChunkHead *)buf;
  if (ch->part == V5P_PLACE) {
    if (len >= sizeof(V5ChunkHead) + sizeof(V5PlaceBody))
      write_placement_json(ch, (const V5PlaceBody *)(buf +
                                                    sizeof(V5ChunkHead)));
    return;
  }
  const uint8_t *body = buf + sizeof(V5ChunkHead);
  const size_t body_len = len - sizeof(V5ChunkHead);
  if (ch->part == V5P_CTA) {
    /* one kind-3 file record per warp (32B V5Head + 40B V5CtaBody);
     * readers dedupe per CTA.  The chunk precedes the producer warp's
     * first instruction record, so flush that key's pending record. */
    V5Pending &pend = g_pending[pending_key(ch)];
    flush_pending(pend);
    if (body_len < sizeof(V5CtaBody))
      return;
    V5Head head{};
    head.cta_x = ch->cta_x;
    head.cta_y = ch->cta_y;
    head.cta_z = ch->cta_z;
    head.warp = ch->warp;
    head.sm = 0;
    head.kind = V5_KIND_CTA;
    head.function = ch->function;
    head.idx = ch->idx;
    head.active_mask = 0xFFFFFFFFu;
    head.exec_mask = 0xFFFFFFFFu;
    std::vector<uint8_t> rec((const uint8_t *)&head,
                             (const uint8_t *)&head + sizeof(V5Head));
    rec.insert(rec.end(), body, body + sizeof(V5CtaBody));
    write_record(rec.data(), rec.size());
    return;
  }
  V5Pending &pend = g_pending[pending_key(ch)];
  if (ch->part == V5P_AFTER) {
    /* Per-warp channel order: this warp's BEFORE record for the paired
     * PHASECHK precedes this chunk -- flush it so the AFTER lands right
     * after its BEFORE in the file. */
    flush_pending(pend);
    const V5AfterBody *ab = (const V5AfterBody *)body;
    V5Head head{};
    head.cta_x = ch->cta_x;
    head.cta_y = ab->cta_y;
    head.cta_z = ab->cta_z;
    head.warp = ch->warp;
    head.sm = ab->sm;
    head.kind = V5_KIND_AFTER;
    head.function = ch->function;
    head.idx = ch->idx;
    head.active_mask = ab->active_mask;
    head.exec_mask = 0;
    std::vector<uint8_t> rec((const uint8_t *)&head,
                             (const uint8_t *)&head + sizeof(V5Head));
    rec.insert(rec.end(), (const uint8_t *)&ab->ready_mask,
               (const uint8_t *)&ab->ready_mask + 4);
    write_record(rec.data(), rec.size());
    return;
  }
  if (ch->part == V5P_GSYNC) {
    /* Per-warp channel order: the BEFORE record for the instruction
     * this chunk belongs to precedes it -- flush the pending record so
     * the GSYNC record lands right after it in the file. */
    flush_pending(pend);
    const V5GsyncBody *gb = (const V5GsyncBody *)body;
    const size_t nvals =
        (size_t)__builtin_popcount(gb->mask) * gb->nwords;
    if (body_len < offsetof(V5GsyncBody, vals) + nvals * 4)
      return;
    V5Head head{};
    head.cta_x = ch->cta_x;
    head.cta_y = gb->cta_y;
    head.cta_z = gb->cta_z;
    head.warp = ch->warp;
    head.sm = gb->sm;
    head.kind = V5_KIND_GSYNC;
    head.function = ch->function;
    head.idx = ch->idx;
    head.active_mask = gb->mask;
    head.exec_mask = gb->mask;
    head.reserved = 0;
    std::vector<uint8_t> rec((const uint8_t *)&head,
                             (const uint8_t *)&head + sizeof(V5Head));
    const uint8_t tail4[4] = {gb->phase, gb->nwords, gb->flags, 0};
    rec.insert(rec.end(), tail4, tail4 + 4);
    rec.insert(rec.end(), (const uint8_t *)gb->vals,
               (const uint8_t *)gb->vals + nvals * 4);
    write_record(rec.data(), rec.size());
    return;
  }
  if (ch->part == V5P_HEAD) {
    flush_pending(pend);
    const V5HeadBody *hb = (const V5HeadBody *)body;
    pend.open = true;
    pend.head.cta_x = ch->cta_x;
    pend.head.cta_y = hb->cta_y;
    pend.head.cta_z = hb->cta_z;
    pend.head.warp = ch->warp;
    pend.head.sm = hb->sm;
    pend.head.kind = ch->kind;
    pend.head.function = ch->function;
    pend.head.idx = ch->idx;
    pend.head.active_mask = hb->active_mask;
    pend.head.exec_mask = hb->exec_mask;
    pend.head.reserved = hb->reserved; /* %warpid */
    const uint64_t lkey = ((uint64_t)ch->function << 32) | ch->idx;
    auto it = g_layouts.find(lkey);
    pend.layout = it != g_layouts.end() ? &it->second : nullptr;
    if (!pend.layout)
      fprintf(stderr, "cyclescope-tracer: no layout reg=%u idx=%u\n", ch->function,
              ch->idx);
    return;
  }
  if (!pend.open || !pend.layout)
    return;
  const InstrLayout &layout = *pend.layout;
  const int exec_count = __builtin_popcount(pend.head.exec_mask);
  if (ch->part == V5P_MREF) {
    const V5MrefBody *mb = (const V5MrefBody *)body;
    while (pend.next_mref < (size_t)layout.mref_count &&
           pend.next_mref < mb->mref) {
      /* an mref chunk was never pushed for this slot; nothing to append */
      ++pend.next_mref;
    }
    const size_t want = (size_t)exec_count * 8;
    const size_t have = body_len > 8 ? body_len - 8 : 0;
    pend.tail.insert(pend.tail.end(), (const uint8_t *)mb->addrs,
                     (const uint8_t *)mb->addrs + std::min(want, have));
    ++pend.next_mref;
    return;
  }
  if (ch->part == V5P_OP)
    append_op_chunk(pend, (const V5OpBody *)body,
                    body_len > 4 ? body_len - 4 : 0);
}

static void flush_all_pending() {
  for (auto &kv : g_pending)
    flush_pending(kv.second);
}

static void *recv_thread_fun(void *) {
  uint8_t *recv_buffer = (uint8_t *)malloc(CHANNEL_SIZE);
  while (recv_thread_started) {
    uint32_t num_recv_bytes = 0;
    if ((num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
      /* Each channel push lands contiguously; every chunk is delimited
       * by its V5ChunkHead plus a body length the host derives from the
       * pending record's layout and exec_mask. */
      std::lock_guard<std::mutex> guard(g_record_mutex);
      size_t pos = 0;
      while (pos + sizeof(V5ChunkHead) <= num_recv_bytes) {
        const V5ChunkHead *ch = (const V5ChunkHead *)(recv_buffer + pos);
        size_t body = 0;
        if (ch->part == V5P_HEAD) {
          body = sizeof(V5HeadBody);
        } else if (ch->part == V5P_AFTER) {
          body = sizeof(V5AfterBody);
        } else if (ch->part == V5P_PLACE) {
          body = sizeof(V5PlaceBody);
        } else if (ch->part == V5P_CTA) {
          body = sizeof(V5CtaBody);
        } else if (ch->part == V5P_GSYNC) {
          /* self-describing body: fixed header + popcount(mask)*nwords
           * u32 values; no static layout needed (V5P_OP-style lookup) */
          const V5GsyncBody *gb =
              (const V5GsyncBody *)(recv_buffer + pos +
                                    sizeof(V5ChunkHead));
          body = offsetof(V5GsyncBody, vals) +
                 (size_t)__builtin_popcount(gb->mask) * gb->nwords * 4;
        } else {
          auto pit = g_pending.find(pending_key(ch));
          const V5Pending *pend =
              pit != g_pending.end() && pit->second.open ? &pit->second
                                                        : nullptr;
          const int exec_count =
              pend ? __builtin_popcount(pend->head.exec_mask) : 0;
          if (ch->part == V5P_MREF) {
            body = 8 + (size_t)exec_count * 8;
          } else {
            /* V5P_OP body: u8 entry + 3 pad + entry bytes */
            const V5OpBody *ob =
                (const V5OpBody *)(recv_buffer + pos + sizeof(V5ChunkHead));
            size_t entry = 4;
            if (pend && pend->layout && ob->entry < pend->layout->ops.size()) {
              const OpEntry &e = pend->layout->ops[ob->entry];
              if (e.kind == V5_OP_R || e.kind == V5_OP_RANK ||
                  e.kind == V5_OP_BRANK)
                entry += (size_t)exec_count * 4;
              else if (e.kind == V5_OP_UR || e.kind == V5_OP_URANK ||
                       e.kind == V5_OP_CPOL || e.kind == V5_OP_CPOLH)
                entry += 4;
              else if (e.kind == V5_OP_SPACE)
                entry += 12;
              else if (e.kind == V5_OP_DESC)
                entry += ob->vals[0] == 0xFFFFFFFFu ? 4 + 128 : 4;
            }
            body = entry;
          }
        }
        const size_t chunk = sizeof(V5ChunkHead) + body;
        if (pos + chunk > num_recv_bytes) {
          fprintf(stderr, "cyclescope-tracer: partial chunk %zu < %zu\n",
                  num_recv_bytes - pos, chunk);
          break;
        }
        process_chunk(recv_buffer + pos, chunk);
        pos += chunk;
      }
      if (pos != num_recv_bytes)
        fprintf(stderr, "cyclescope-tracer: %zu undelimited bytes dropped\n",
                num_recv_bytes - pos);
    }
    sched_yield();
  }
  free(recv_buffer);
  return nullptr;
}

/* ------------------------------------------------------------------ */
/* launch hooks                                                        */
/* ------------------------------------------------------------------ */

static int get_attr_with_kernel_fallback(CUfunction func, int attr) {
  int value = -1;
  if (cuFuncGetAttribute(&value, (CUfunction_attribute)attr, func) !=
      CUDA_SUCCESS)
    return -1;
  return value;
}

static FILE *open_trace_stream(const std::string &path, bool *via_pipe) {
  *via_pipe = false;
  if (have_zstd) {
    std::string cmd = "zstd -q -T0 -f -o " + path + ".zst -";
    FILE *f = popen(cmd.c_str(), "w");
    if (f) {
      *via_pipe = true;
      return f;
    }
  }
  return fopen(path.c_str(), "wb");
}

static void close_trace_stream(LaunchOut &out) {
  if (!out.f)
    return;
  if (out.via_pipe)
    pclose(out.f);
  else
    fclose(out.f);
  out.f = nullptr;
}

/* v6 launch/device facts: queried with the driver API at launch time;
 * -1 serializes as null with a *_note reason.  Facts only describe what
 * the hardware/driver reports; nothing is invented. */
struct V6LaunchFacts {
  int occ_blocks_per_sm = -1;
  int occ_max_clusters = -1;
  const char *occ_clusters_note = "not a cluster launch";
  int max_dynamic_smem = -1;
  int preferred_smem_carveout = -1;
  int cluster_sched_policy = -1;
  int non_portable_cluster = -1;
  int required_cluster[3] = {-1, -1, -1};
  int preferred_cluster[3] = {-1, -1, -1};
};

static void json_int_or_null(FILE *f, int v) {
  if (v < 0)
    fprintf(f, "null");
  else
    fprintf(f, "%d", v);
}

static void json_str_or_null(FILE *f, const char *s) {
  if (!s || !s[0])
    fprintf(f, "null");
  else
    json_escape(f, s);
}

static bool g_device_facts_done = false;
static void write_device_json(CUcontext ctx) {
  if (g_device_facts_done)
    return;
  CUdevice dev = -1;
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS || dev < 0)
    return; /* retry at the next launch */
  g_device_facts_done = true;
  const auto attr = [dev](CUdevice_attribute a) -> int {
    int v = -1;
    if (cuDeviceGetAttribute(&v, a, dev) != CUDA_SUCCESS)
      return -1;
    return v;
  };
  char devname[256] = {0};
  if (cuDeviceGetName(devname, sizeof(devname) - 1, dev) != CUDA_SUCCESS)
    devname[0] = 0;
  int drv = -1;
  cuDriverGetVersion(&drv);
  /* NVBit's bundled cuda.h resolves the unversioned name to the legacy
   * unsigned-int entry point; call the 64-bit v2 symbol explicitly. */
  size_t total_mem = 0;
  cuDeviceTotalMem_v2(&total_mem, dev);
  FILE *f = fopen((user_folder + "/device.json").c_str(), "w");
  if (!f)
    return;
  fprintf(f, "{\"name\": ");
  json_str_or_null(f, devname);
  fprintf(f, ", \"sm_count\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
  fprintf(f, ", \"l2_cache_bytes\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE));
  fprintf(f, ", \"smem_per_sm_bytes\": ");
  json_int_or_null(
      f, attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR));
  fprintf(f, ", \"smem_per_block_bytes\": ");
  json_int_or_null(f,
                   attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
  fprintf(f, ", \"smem_per_block_optin_bytes\": ");
  json_int_or_null(
      f, attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN));
  fprintf(f, ", \"registers_per_sm\": ");
  json_int_or_null(
      f, attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR));
  fprintf(f, ", \"registers_per_block\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK));
  fprintf(f, ", \"max_threads_per_sm\": ");
  json_int_or_null(f,
                   attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR));
  fprintf(f, ", \"max_threads_per_block\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
  fprintf(f, ", \"clock_rate_khz\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE));
  fprintf(f, ", \"memory_clock_rate_khz\": ");
  json_int_or_null(f, attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE));
  fprintf(f, ", \"memory_bus_width_bits\": ");
  json_int_or_null(f,
                   attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH));
  fprintf(f, ", \"total_mem_bytes\": %llu",
          (unsigned long long)total_mem);
  if (drv > 0)
    fprintf(f, ", \"driver_version\": \"%d.%d\"", drv / 1000,
            (drv % 1000) / 10);
  else
    fprintf(f, ", \"driver_version\": null");
#ifdef CUDA_VERSION
  fprintf(f, ", \"cuda_version\": \"%d.%d\"", CUDA_VERSION / 1000,
          (CUDA_VERSION % 1000) / 10);
#else
  fprintf(f, ", \"cuda_version\": null");
#endif
  fprintf(f, "}\n");
  fclose(f);
}

/* Record one device-side touch of a host memory API (memcpy/memset).
 * Synchronous copies are already complete at API-exit; async copies are
 * attributed by call order -- the launch-boundary merge in
 * write_l2_footprint places them between the surrounding kernels'
 * device touches. */
static void record_footprint_op(uint64_t addr, uint64_t bytes,
                                int is_store) {
  if (addr == 0 || bytes == 0)
    return;
  g_footprint_host_ops.push_back(FootprintHostOp{
      addr, bytes, (uint32_t)is_store, g_launch_counter,
      (uint64_t)g_footprint_host_ops.size()});
}

/* Device-pointer classification for generic cuMemcpy/cuMemcpyAsync
 * without driver API queries: cuPointerGetAttribute inside an nvbit
 * callback re-enters an intercepted driver entry point and deadlocks.
 * Instead we keep intervals of live device allocations, updated on
 * cuMemAlloc and cuMemFree exits. */
static void dev_range_add(uint64_t addr, uint64_t bytes) {
  if (addr && bytes)
    g_footprint_dev_ranges.emplace(addr, addr + bytes);
}

static void dev_range_remove(uint64_t addr) {
  for (auto it = g_footprint_dev_ranges.lower_bound({addr, 0});
       it != g_footprint_dev_ranges.end() && it->first <= addr;) {
    auto cur = it++;
    if (addr >= cur->first && addr < cur->second)
      g_footprint_dev_ranges.erase(cur);
  }
}

static bool pointer_is_device(uint64_t p) {
  if (p == 0)
    return false;
  auto it = g_footprint_dev_ranges.upper_bound({p, UINT64_MAX});
  if (it == g_footprint_dev_ranges.begin())
    return false;
  --it;
  return p >= it->first && p < it->second;
}

static void record_footprint_mem_event(nvbit_api_cuda_t cbid, void *params,
                                       const CUresult *pStatus) {
  if (pStatus == NULL || *pStatus != CUDA_SUCCESS)
    return;
  switch (cbid) {
  case API_CUDA_cuMemAlloc_v2: {
    auto *p = (cuMemAlloc_v2_params *)params;
    if (p->dptr != NULL)
      dev_range_add(*p->dptr, p->bytesize);
  } break;
  case API_CUDA_cuMemAllocPitch_v2: {
    auto *p = (cuMemAllocPitch_v2_params *)params;
    if (p->dptr != NULL) {
      const uint64_t pitch =
          p->pPitch != NULL ? *p->pPitch : p->WidthInBytes;
      dev_range_add(*p->dptr, pitch * (uint64_t)p->Height);
    }
  } break;
  case API_CUDA_cuMemAllocAsync:
  case API_CUDA_cuMemAllocAsync_ptsz: {
    auto *p = (cuMemAllocAsync_params *)params;
    if (p->dptr != NULL)
      dev_range_add(*p->dptr, p->bytesize);
  } break;
  case API_CUDA_cuMemFree_v2: {
    auto *p = (cuMemFree_v2_params *)params;
    dev_range_remove(p->dptr);
  } break;
  case API_CUDA_cuMemFreeAsync:
  case API_CUDA_cuMemFreeAsync_ptsz: {
    auto *p = (cuMemFreeAsync_params *)params;
    dev_range_remove(p->dptr);
  } break;
  default:
    break;
  }
}

static void record_footprint_host_op(nvbit_api_cuda_t cbid, void *params) {
  switch (cbid) {
  case API_CUDA_cuMemcpyHtoD_v2:
  case API_CUDA_cuMemcpyHtoD_v2_ptds: {
    auto *p = (cuMemcpyHtoD_v2_params *)params;
    record_footprint_op(p->dstDevice, p->ByteCount, 1);
  } break;
  case API_CUDA_cuMemcpyHtoDAsync_v2:
  case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz: {
    auto *p = (cuMemcpyHtoDAsync_v2_params *)params;
    record_footprint_op(p->dstDevice, p->ByteCount, 1);
  } break;
  case API_CUDA_cuMemcpyDtoH_v2:
  case API_CUDA_cuMemcpyDtoH_v2_ptds: {
    auto *p = (cuMemcpyDtoH_v2_params *)params;
    record_footprint_op(p->srcDevice, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemcpyDtoHAsync_v2:
  case API_CUDA_cuMemcpyDtoHAsync_v2_ptsz: {
    auto *p = (cuMemcpyDtoHAsync_v2_params *)params;
    record_footprint_op(p->srcDevice, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemcpyDtoD_v2:
  case API_CUDA_cuMemcpyDtoD_v2_ptds: {
    auto *p = (cuMemcpyDtoD_v2_params *)params;
    record_footprint_op(p->dstDevice, p->ByteCount, 1);
    record_footprint_op(p->srcDevice, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemcpyDtoDAsync_v2:
  case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz: {
    auto *p = (cuMemcpyDtoDAsync_v2_params *)params;
    record_footprint_op(p->dstDevice, p->ByteCount, 1);
    record_footprint_op(p->srcDevice, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemcpy: {
    auto *p = (cuMemcpy_params *)params;
    if (pointer_is_device(p->dst))
      record_footprint_op(p->dst, p->ByteCount, 1);
    if (pointer_is_device(p->src))
      record_footprint_op(p->src, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemcpyAsync:
  case API_CUDA_cuMemcpyAsync_ptsz: {
    auto *p = (cuMemcpyAsync_params *)params;
    if (pointer_is_device(p->dst))
      record_footprint_op(p->dst, p->ByteCount, 1);
    if (pointer_is_device(p->src))
      record_footprint_op(p->src, p->ByteCount, 0);
  } break;
  case API_CUDA_cuMemsetD8_v2:
  case API_CUDA_cuMemsetD8_v2_ptds: {
    auto *p = (cuMemsetD8_v2_params *)params;
    record_footprint_op(p->dstDevice, p->N, 1);
  } break;
  case API_CUDA_cuMemsetD16_v2:
  case API_CUDA_cuMemsetD16_v2_ptds: {
    auto *p = (cuMemsetD16_v2_params *)params;
    record_footprint_op(p->dstDevice, p->N * 2, 1);
  } break;
  case API_CUDA_cuMemsetD32_v2:
  case API_CUDA_cuMemsetD32_v2_ptds: {
    auto *p = (cuMemsetD32_v2_params *)params;
    record_footprint_op(p->dstDevice, p->N * 4, 1);
  } break;
  case API_CUDA_cuMemsetD8Async:
  case API_CUDA_cuMemsetD8Async_ptsz: {
    auto *p = (cuMemsetD8Async_params *)params;
    record_footprint_op(p->dstDevice, p->N, 1);
  } break;
  case API_CUDA_cuMemsetD16Async:
  case API_CUDA_cuMemsetD16Async_ptsz: {
    auto *p = (cuMemsetD16Async_params *)params;
    record_footprint_op(p->dstDevice, p->N * 2, 1);
  } break;
  case API_CUDA_cuMemsetD32Async:
  case API_CUDA_cuMemsetD32Async_ptsz: {
    auto *p = (cuMemsetD32Async_params *)params;
    record_footprint_op(p->dstDevice, p->N * 4, 1);
  } break;
  default:
    break;
  }
}

/* Footprint entry encoding: u64 = (sector_address << 8) |
 * ((smid & 0xFF) << 1) | is_store, sector_address = addr & ~31.  Host-API
 * ops carry no SM; their smid field is 0xFF. */
#define FOOTPRINT_SMID_HOST 0xFFull
static uint64_t footprint_sector_key(uint64_t entry) {
  return entry & ~0xFFull;
}
static uint64_t footprint_host_entry(uint64_t sector, int is_store) {
  return (sector << 8) | (FOOTPRINT_SMID_HOST << 1) |
         (is_store ? 1ull : 0ull);
}

/* Drain the prior-launch footprint ring and write
 * l2_footprint.{bin,json}.  Called inside the selected launch's enter
 * callback where the device is already idle; after it the cell stays
 * inactive (g_selected_launch_seen gates any further prior launches). */
static void write_l2_footprint(uint64_t selected_seq,
                               uint64_t last_identical_seq) {
  g_footprint_cell.active = 0;
  cudaDeviceSynchronize();

  unsigned long long total = 0;
  if (g_footprint_counter_dev != NULL)
    cudaMemcpy(&total, g_footprint_counter_dev, sizeof(total),
               cudaMemcpyDeviceToHost);
  const uint64_t cap = g_footprint_entries;
  const uint64_t valid = (uint64_t)total < cap ? (uint64_t)total : cap;
  const bool wrapped = (uint64_t)total > cap;

  /* Recover ring contents oldest -> newest.  Without wrap the valid
   * slots are 0..total-1; with wrap they are the last cap writes at
   * positions (total-cap)..total-1 mod cap. */
  std::vector<uint64_t> ordered;
  if (valid > 0 && g_footprint_ring_dev != NULL) {
    ordered.resize(valid);
    if (!wrapped) {
      cudaMemcpy(ordered.data(), g_footprint_ring_dev, valid * 8,
                 cudaMemcpyDeviceToHost);
    } else {
      const uint64_t first = ((uint64_t)total - cap) & (cap - 1);
      const uint64_t tail = cap - first;
      cudaMemcpy(ordered.data(),
                 (uint64_t *)g_footprint_ring_dev + first, tail * 8,
                 cudaMemcpyDeviceToHost);
      if (first > 0)
        cudaMemcpy(ordered.data() + tail, g_footprint_ring_dev,
                   first * 8, cudaMemcpyDeviceToHost);
    }
  }

  int dev = 0;
  cudaGetDevice(&dev);
  int l2_bytes = 0;
  cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
  const uint64_t keep_limit = 2ull * (uint64_t)l2_bytes / 32ull;

  /* Host ops were only recorded to a host vector; merge them by
   * launch-boundary position: op at seq k sits at boundary[k+1], seq 0
   * at position 0, missing boundary uses the selected snapshot.  Descending
   * scan stops once keep_limit distinct sectors are kept. */
  const uint64_t base_idx = wrapped ? (uint64_t)total - cap : 0;
  std::vector<size_t> host_order(g_footprint_host_ops.size());
  for (size_t i = 0; i < host_order.size(); ++i)
    host_order[i] = i;
  const auto op_pos = [&](size_t i) -> uint64_t {
    const uint64_t k = g_footprint_host_ops[i].seq;
    if (k == 0)
      return 0;
    const auto it = g_launch_boundaries.find(k + 1);
    return it != g_launch_boundaries.end() ? it->second : (uint64_t)total;
  };
  std::sort(host_order.begin(), host_order.end(), [&](size_t a, size_t b) {
    const uint64_t pa = op_pos(a), pb = op_pos(b);
    return pa != pb ? pa > pb : a > b;
  });
  std::unordered_set<uint64_t> seen;
  std::vector<uint64_t> dedup;
  unsigned long long sectors_expanded = 0;
  bool stopped_early = false;
  size_t hp = 0;
  const size_t n_host = host_order.size();
  /* Warmup split: entries newer than the end of the last launch identical
   * to the selected one (ring position >= the boundary recorded at the
   * entry of the launch after it; host ops by seq >= that launch).  The
   * engine seeds [0, split) first, replays the identical launch's own
   * accesses, then seeds [split, kept). */
  uint64_t boundary_end = (uint64_t)total;
  if (last_identical_seq != 0) {
    const auto it = g_launch_boundaries.find(last_identical_seq + 1);
    if (it != g_launch_boundaries.end())
      boundary_end = it->second;
  }
  uint64_t newer_kept = 0;
  const auto emit_host = [&](const FootprintHostOp *op, uint64_t sector) {
    ++sectors_expanded;
    if (seen.insert(sector & ~31ull).second) {
      if (dedup.size() < keep_limit) {
        dedup.push_back(footprint_host_entry(sector, (int)op->is_store));
        if (last_identical_seq != 0 && op->seq >= last_identical_seq)
          ++newer_kept;
      } else {
        stopped_early = true;
      }
    }
  };
  for (int64_t ri = (int64_t)ordered.size() - 1; ri >= 0 && !stopped_early;
       --ri) {
    const uint64_t gpos = base_idx + (uint64_t)ri;
    while (hp < n_host && op_pos(host_order[hp]) > gpos) {
      const FootprintHostOp *op = &g_footprint_host_ops[host_order[hp]];
      ++hp;
      const uint64_t first = op->addr & ~31ull;
      const uint64_t nsec = (op->addr + op->bytes - first + 31) / 32;
      for (int64_t s = (int64_t)nsec - 1; s >= 0 && !stopped_early; --s)
        emit_host(op, first + (uint64_t)s * 32);
    }
    if (stopped_early)
      break;
    if (seen.insert(footprint_sector_key(ordered[ri])).second) {
      if (dedup.size() < keep_limit) {
        dedup.push_back(ordered[ri]);
        if (last_identical_seq != 0 && gpos >= boundary_end)
          ++newer_kept;
      } else {
        stopped_early = true;
      }
    }
  }
  if (!stopped_early) {
    for (; hp < n_host; ++hp) {
      const FootprintHostOp *op = &g_footprint_host_ops[host_order[hp]];
      const uint64_t first = op->addr & ~31ull;
      const uint64_t nsec = (op->addr + op->bytes - first + 31) / 32;
      for (int64_t s = (int64_t)nsec - 1; s >= 0 && !stopped_early; --s)
        emit_host(op, first + (uint64_t)s * 32);
    }
  }
  const uint64_t distinct = (uint64_t)seen.size();
  const bool truncated = stopped_early;
  std::reverse(dedup.begin(), dedup.end());
  const uint64_t kept = (uint64_t)dedup.size();
  const uint64_t warmup_split = kept - newer_kept;

  /* l2_footprint.bin: no header, raw u64 entries oldest -> newest (the
   * job container concatenates it directly). */
  const std::string bin_path = user_folder + "/l2_footprint.bin";
  FILE *bf = fopen(bin_path.c_str(), "wb");
  if (bf) {
    if (kept > 0)
      fwrite(dedup.data(), 8, kept, bf);
    fclose(bf);
  } else {
    fprintf(stderr, "cyclescope-tracer: missing_fact:l2_footprint_write (%s)\n",
            bin_path.c_str());
  }

  const std::string json_path = user_folder + "/l2_footprint.json";
  FILE *jf = fopen(json_path.c_str(), "w");
  if (jf) {
    fprintf(jf,
            "{\n"
            "  \"selected_launch\": %llu,\n"
            "  \"total_touches\": %llu,\n"
            "  \"ring_capacity\": %llu,\n"
            "  \"distinct\": %llu,\n"
            "  \"kept\": %llu,\n"
            "  \"warmup_split\": %llu,\n"
            "  \"wrapped\": %s,\n"
            "  \"l2_bytes\": %u,\n"
            "  \"keep_limit\": %llu,\n"
            "  \"truncated_to_keep_limit\": %s,\n"
            "  \"prior_launches_recorded\": %llu,\n"
            "  \"prior_launches_with_bulk_ops\": %llu,\n"
            "  \"host_ops\": %llu,\n"
            "  \"host_ops_unsupported\": %llu,\n"
            "  \"host_op_sectors_expanded\": %llu,\n"
            "  \"launch_boundaries_recorded\": %llu,\n"
            "  \"last_identical_launch\": %llu,\n"
            "  \"entries_env\": %lld\n"
            "}\n",
            (unsigned long long)selected_seq, (unsigned long long)total,
            (unsigned long long)cap, (unsigned long long)distinct,
            (unsigned long long)kept, (unsigned long long)warmup_split,
            wrapped ? "true" : "false", (unsigned)l2_bytes,
            (unsigned long long)keep_limit,
            truncated ? "true" : "false", g_prior_launches_recorded,
            g_prior_launches_with_bulk_ops,
            (unsigned long long)g_footprint_host_ops.size(),
            g_host_ops_unsupported, sectors_expanded,
            (unsigned long long)g_launch_boundaries.size(),
            (unsigned long long)last_identical_seq,
            (long long)g_footprint_entries_env);
    fclose(jf);
  } else {
    fprintf(stderr, "cyclescope-tracer: missing_fact:l2_footprint_write (%s)\n",
            json_path.c_str());
  }

  g_footprint_cell.active = 0;
}

static void write_launch_jsonl(uint64_t ordinal, const char *api,
                               const std::string &func_name,
                               const std::string &mangled_name,
                               uint64_t reg,
                               const std::array<uint32_t, 3> &grid,
                               const std::array<uint32_t, 3> &block,
                               const std::array<uint32_t, 3> &cluster,
                               const std::array<uint32_t, 3>
                                   &effective_cluster,
                               uint32_t smem_dynamic, uint32_t smem_static,
                               int regs, int local_bytes, CUstream stream,
                               bool traced,
                               const std::vector<std::array<uint32_t, 3>>
                                   &sampled,
                               const V6LaunchFacts &facts,
                               uint32_t prior_identical_launches,
                               const char *argument_bytes_status,
                               uint32_t function_prior_launches) {
  if (!g_launches_jsonl)
    return;
  fprintf(g_launches_jsonl,
          "{\"launch\": %llu, \"function\": ", (unsigned long long)ordinal);
  json_escape(g_launches_jsonl, func_name);
  fprintf(g_launches_jsonl, ", \"mangled\": ");
  json_escape(g_launches_jsonl, mangled_name);
  fprintf(g_launches_jsonl, ", \"function_id\": %llu, \"api\": \"%s\", \"grid\": "
          "[%u,%u,%u], \"block\": [%u,%u,%u], \"cluster\": [%u,%u,%u], "
          "\"effective_cluster\": [%u,%u,%u], "
          "\"smem_dynamic\": %u, \"smem_static\": %u, \"regs\": %d, "
          "\"local_bytes\": %d, \"stream\": \"0x%llx\", \"traced\": %s, "
          "\"occupancy_max_blocks_per_sm\": ",
          (unsigned long long)reg, api, grid[0], grid[1], grid[2], block[0],
          block[1], block[2], cluster[0], cluster[1], cluster[2],
          effective_cluster[0], effective_cluster[1], effective_cluster[2],
          smem_dynamic, smem_static, regs, local_bytes,
          (unsigned long long)(uintptr_t)stream, traced ? "true" : "false");
  json_int_or_null(g_launches_jsonl, facts.occ_blocks_per_sm);
  fprintf(g_launches_jsonl, ", \"occupancy_max_active_clusters\": ");
  json_int_or_null(g_launches_jsonl, facts.occ_max_clusters);
  fprintf(g_launches_jsonl, ", \"occupancy_clusters_note\": ");
  json_str_or_null(g_launches_jsonl, facts.occ_clusters_note);
  fprintf(g_launches_jsonl, ", \"max_dynamic_smem\": ");
  json_int_or_null(g_launches_jsonl, facts.max_dynamic_smem);
  fprintf(g_launches_jsonl, ", \"preferred_smem_carveout\": ");
  json_int_or_null(g_launches_jsonl, facts.preferred_smem_carveout);
  fprintf(g_launches_jsonl, ", \"cluster_scheduling_policy\": ");
  json_int_or_null(g_launches_jsonl, facts.cluster_sched_policy);
  fprintf(g_launches_jsonl, ", \"non_portable_cluster_allowed\": ");
  json_int_or_null(g_launches_jsonl, facts.non_portable_cluster);
  fprintf(g_launches_jsonl, ", \"required_cluster\": [");
  json_int_or_null(g_launches_jsonl, facts.required_cluster[0]);
  fprintf(g_launches_jsonl, ",");
  json_int_or_null(g_launches_jsonl, facts.required_cluster[1]);
  fprintf(g_launches_jsonl, ",");
  json_int_or_null(g_launches_jsonl, facts.required_cluster[2]);
  fprintf(g_launches_jsonl, "], \"preferred_cluster\": [");
  json_int_or_null(g_launches_jsonl, facts.preferred_cluster[0]);
  fprintf(g_launches_jsonl, ",");
  json_int_or_null(g_launches_jsonl, facts.preferred_cluster[1]);
  fprintf(g_launches_jsonl, ",");
  json_int_or_null(g_launches_jsonl, facts.preferred_cluster[2]);
  fprintf(g_launches_jsonl, "], \"sampled_clusters\": [");
  for (size_t i = 0; i < sampled.size(); ++i)
    fprintf(g_launches_jsonl, "%s[%u,%u,%u]", i ? "," : "", sampled[i][0],
            sampled[i][1], sampled[i][2]);
  fprintf(g_launches_jsonl,
          "], \"prior_identical_launches\": %u, "
          "\"argument_bytes_status\": \"%s\", "
          "\"function_prior_launches\": %u}\n",
          prior_identical_launches, argument_bytes_status,
          function_prior_launches);
  fflush(g_launches_jsonl);
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                nvbit_api_cuda_t cbid, void *params,
                                const char *api_name, bool stream_capture) {
  g_current_launch_traced = false;
  select_arch_table(ctx);
  if (cuda_stack_size > 0 && !cuda_stack_size_applied) {
    const cudaError_t serr =
        cudaDeviceSetLimit(cudaLimitStackSize, (size_t)cuda_stack_size);
    if (serr != cudaSuccess)
      fprintf(stderr, "cyclescope-tracer cudaDeviceSetLimit(stackSize=%u): %s\n",
              cuda_stack_size, cudaGetErrorString(serr));
    cuda_stack_size_applied = 1;
  }
  const uint64_t ordinal = ++g_launch_counter;
  V5_DEV.cur_launch = (int32_t)ordinal;
  /* Footprint boundary: after this sync the device is idle, so the ring
   * counter marks the end of all launches up to the previous one. */
  if (g_footprint_enabled && !g_selected_launch_seen && !stream_capture) {
    cudaDeviceSynchronize();
    unsigned long long pos = 0;
    if (g_footprint_counter_dev != NULL)
      cudaMemcpy(&pos, g_footprint_counter_dev, sizeof(pos),
                 cudaMemcpyDeviceToHost);
    g_launch_boundaries[ordinal] = pos;
  }
  std::array<uint32_t, 3> grid{0, 0, 0}, block{0, 0, 0}, cluster{0, 0, 0};
  uint32_t smem_dynamic = 0;
  CUstream stream = nullptr;

  std::array<uint32_t, 3> preferred_cluster{0, 0, 0};
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
      cbid == API_CUDA_cuLaunchKernelEx) {
    auto *p = (cuLaunchKernelEx_params *)params;
    grid = {p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ};
    block = {p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ};
    smem_dynamic = p->config->sharedMemBytes;
    stream = p->config->hStream;
    for (unsigned i = 0; i < p->config->numAttrs; ++i) {
      CUlaunchAttribute &a = p->config->attrs[i];
      if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
        cluster = {a.value.clusterDim.x, a.value.clusterDim.y,
                   a.value.clusterDim.z};
      if (a.id == CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION)
        preferred_cluster = {a.value.clusterDim.x, a.value.clusterDim.y,
                             a.value.clusterDim.z};
    }
  } else {
    auto *p = (cuLaunchKernel_params *)params;
    grid = {p->gridDimX, p->gridDimY, p->gridDimZ};
    block = {p->blockDimX, p->blockDimY, p->blockDimZ};
    smem_dynamic = p->sharedMemBytes;
    stream = p->hStream;
  }
  /* Effective cluster dims (S0.2 rule, same as the converter): the launch
   * attribute wins when nontrivial; a plain 1x1x1 launch falls back to
   * the function's required_cluster attributes (compiled-in cluster
   * size).  `cluster` keeps the launch-attribute value for launches.jsonl;
   * `eff_cluster` is what sampling and sampled_clusters use. */
  const bool cluster_attr =
      cluster[0] != 0 && !(cluster[0] == 1 && cluster[1] == 1 &&
                           cluster[2] == 1);
  std::array<uint32_t, 3> eff_cluster;
  if (cluster_attr) {
    eff_cluster = cluster;
  } else {
    int cx = get_attr_with_kernel_fallback(
        func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH);
    int cy = get_attr_with_kernel_fallback(
        func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT);
    int cz = get_attr_with_kernel_fallback(
        func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH);
    eff_cluster = (cx > 0)
        ? std::array<uint32_t, 3>{(uint32_t)cx, (uint32_t)cy,
                                  (uint32_t)cz}
        : std::array<uint32_t, 3>{1, 1, 1};
  }
  if (cluster[0] == 0)
    cluster = {1, 1, 1};  /* launches.jsonl "cluster" = launch attr */
  if (stream_capture) {
    /* graph-captured launches are identities only: no trace, no sampling */
  }

  const std::string name =
      nvbit_get_func_name(ctx, func) ? nvbit_get_func_name(ctx, func) : "";
  const char *mangled_raw = nvbit_get_func_name(ctx, func, true);
  const std::string mangled_name = mangled_raw ? mangled_raw : "";
  const bool pattern_match =
      !stream_capture && launch_pattern_matches(ordinal, name, mangled_name);
  const bool selection_match =
      !stream_capture && launch_selected(ordinal, name, mangled_name) &&
      (capture_launch_limit == 0 || ordinal <= capture_launch_limit);
  /* Identity key (ctx + function + geometry + argument bytes) and the
   * prior-launch counters; counted over the launches the selection
   * matches so prior_identical_launches is the warmup count. */
  const LaunchArgumentBytes launch_args =
      launch_argument_bytes(ctx, func, cbid, params);
  std::string launch_key;
  const bool key_exact =
      launch_args.exact &&
      launch_identity_key(ctx, func, cbid, params, grid, block, eff_cluster,
                          smem_dynamic, &launch_key);
  uint32_t prior_identical_launches = 0;
  uint64_t last_identical_launch = 0;
  if (pattern_match)
    record_matching_launch(key_exact ? &launch_key : nullptr, key_exact,
                           &prior_identical_launches, ordinal,
                           &last_identical_launch);
  const uint32_t function_prior_launches =
      g_function_prior_counts[{ctx, func}]++;
  const bool traced = selection_match;
  g_current_launch_traced = traced;
  {
    std::lock_guard<std::mutex> guard(g_record_mutex);
    instrument_function_if_needed(ctx, func);
  }
  const uint64_t reg = g_func_regs.count(func) ? g_func_regs[func].id : 0;
  if (traced && !g_selected_launch_seen) {
    g_selected_launch_seen = true;
    if (g_footprint_enabled)
      write_l2_footprint(ordinal, last_identical_launch);
  }

  std::vector<std::array<uint32_t, 3>> sampled;
  if (traced) {
    const int cd[3] = {(int)eff_cluster[0], (int)eff_cluster[1],
                       (int)eff_cluster[2]};
    const uint32_t nc[3] = {grid[0] / std::max(eff_cluster[0], 1u),
                            grid[1] / std::max(eff_cluster[1], 1u),
                            grid[2] / std::max(eff_cluster[2], 1u)};
    const uint32_t total = nc[0] * nc[1] * nc[2];
    const auto emit_ids = [&](const std::vector<uint32_t> &ids) {
      V5_DEV.sampling.count = (int)ids.size();
      V5_DEV.sampling.cdx = cd[0];
      V5_DEV.sampling.cdy = cd[1];
      V5_DEV.sampling.cdz = cd[2];
      V5_DEV.sampling.ncx = (int)nc[0];
      V5_DEV.sampling.ncy = (int)nc[1];
      for (size_t i = 0; i < ids.size() && i < V5_MAX_SAMPLED_CLUSTERS;
           ++i) {
        V5_DEV.sampling.ids[i] = ids[i];
        const uint32_t cz = nc[0] * nc[1] ? ids[i] / (nc[0] * nc[1]) : 0;
        const uint32_t cy = nc[0] ? (ids[i] / nc[0]) % nc[1] : 0;
        const uint32_t cx = nc[0] ? ids[i] % nc[0] : 0;
        sampled.push_back({cx, cy, cz});
      }
    };
    if (!capture_cluster_ids.empty()) {
      /* explicit linear cluster ids (CYCLESCOPE_CAPTURE_CLUSTERS=a,b,c) */
      std::vector<uint32_t> ids;
      for (uint32_t id : capture_cluster_ids)
        if (id < total)
          ids.push_back(id);
      emit_ids(ids);
    } else if (capture_clusters < 0) {
      /* all clusters: count<0 tells the device to trace every CTA, no
       * ids table (it is bounded by V5_MAX_SAMPLED_CLUSTERS).  The
       * launches.jsonl sampled_clusters list still enumerates every
       * cluster coordinate so the converter's per-cluster filter sees
       * the full set. */
      V5_DEV.sampling.count = -1;
      V5_DEV.sampling.cdx = cd[0];
      V5_DEV.sampling.cdy = cd[1];
      V5_DEV.sampling.cdz = cd[2];
      V5_DEV.sampling.ncx = (int)nc[0];
      V5_DEV.sampling.ncy = (int)nc[1];
      sampled.reserve(total);
      for (uint32_t cz = 0; cz < nc[2]; ++cz)
        for (uint32_t cy = 0; cy < nc[1]; ++cy)
          for (uint32_t cx = 0; cx < nc[0]; ++cx)
            sampled.push_back({cx, cy, cz});
    } else {
      emit_ids(uniform_cluster_sample(
          total, std::min(capture_clusters, V5_MAX_SAMPLED_CLUSTERS)));
    }
    V5_DEV.stop_report = false;
    if (capture_mode_placement)
      arm_placement_array(ordinal, grid, block);
    if (!capture_mode_placement) {
      /* placement mode writes placement.jsonl from the device
       * placement array and produces no trace-<n>.bin */
      char path[1024];
      snprintf(path, sizeof(path), "%s/trace-%llu.bin",
               user_folder.c_str(), (unsigned long long)ordinal);
      {
        std::lock_guard<std::mutex> guard(g_record_mutex);
        close_trace_stream(g_launch_out);
        g_launch_out.path = path;
        g_launch_out.f = open_trace_stream(path, &g_launch_out.via_pipe);
        /* 16-byte file header: magic + format version + header size,
         * then the record stream. */
        if (g_launch_out.f) {
          char head[16] = {'G', 'S', 'I', 'M', 'T', 'R', 'C', '\0'};
          uint32_t v[2] = {1u, 16u};
          fwrite(head, 1, 8, g_launch_out.f);
          fwrite(v, 4, 2, g_launch_out.f);
        }
      }
    }
    recv_thread_receiving = true;
    nvbit_enable_instrumented(ctx, func, true, true);
  } else {
    /* Launches before the selected one stay instrumented with
     * stop_report set: every trace callback early-returns and only
     * instrument_l2_footprint records.  Once the selected launch has
     * been seen, untraced launches are de-instrumented entirely. */
    const bool prior_footprint_launch =
        g_footprint_enabled && g_footprint_kernel &&
        !g_selected_launch_seen && !stream_capture;
    if (prior_footprint_launch) {
      g_footprint_cell.active = 1;
      ++g_prior_launches_recorded;
      if (g_func_has_bulk_ops.count(func) && g_func_has_bulk_ops[func])
        ++g_prior_launches_with_bulk_ops;
      V5_DEV.stop_report = true;
      nvbit_enable_instrumented(ctx, func, true, true);
    } else {
      V5_DEV.stop_report = true;
      nvbit_enable_instrumented(ctx, func, false, true);
    }
  }

  const int regs =
      get_attr_with_kernel_fallback(func, CU_FUNC_ATTRIBUTE_NUM_REGS);
  const int smem_static = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
  const int local_bytes = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);

  /* v6: launch/device facts (driver API; null + note when unavailable) */
  write_device_json(ctx);
  V6LaunchFacts facts;
  const int block_threads = (int)(block[0] * block[1] * block[2]);
  if (block_threads > 0) {
    // NVBit's instrumented function does not inherit the app's opt-in
    // dynamic-smem ceiling (48 KB default) — restore it before asking.
    const int mds = get_attr_with_kernel_fallback(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    if (mds > 0)
      cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                         mds);
    int nb = -1;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &nb, func, block_threads, smem_dynamic) == CUDA_SUCCESS)
      facts.occ_blocks_per_sm = nb;
  }
  if (cluster[0] * cluster[1] * cluster[2] > 1) {
    CUlaunchAttribute attr{};
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = cluster[0];
    attr.value.clusterDim.y = cluster[1];
    attr.value.clusterDim.z = cluster[2];
    CUlaunchConfig cfg{};
    cfg.gridDimX = grid[0];
    cfg.gridDimY = grid[1];
    cfg.gridDimZ = grid[2];
    cfg.blockDimX = block[0];
    cfg.blockDimY = block[1];
    cfg.blockDimZ = block[2];
    cfg.sharedMemBytes = smem_dynamic;
    cfg.hStream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int nc = -1;
    const CUresult orc = cuOccupancyMaxActiveClusters(&nc, func, &cfg);
    if (orc == CUDA_SUCCESS) {
      facts.occ_max_clusters = nc;
      facts.occ_clusters_note = nullptr;
    } else {
      facts.occ_clusters_note = "cuOccupancyMaxActiveClusters failed";
    }
  }
  facts.max_dynamic_smem = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
  facts.preferred_smem_carveout = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
  facts.cluster_sched_policy = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE);
  facts.non_portable_cluster = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED);
  facts.required_cluster[0] = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH);
  facts.required_cluster[1] = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT);
  facts.required_cluster[2] = get_attr_with_kernel_fallback(
      func, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH);
  if (preferred_cluster[0] != 0)
    for (int i = 0; i < 3; ++i)
      facts.preferred_cluster[i] = (int)preferred_cluster[i];
  write_launch_jsonl(ordinal, api_name, name, mangled_name, reg, grid,
                     block, cluster, eff_cluster,
                     smem_dynamic, (uint32_t)std::max(smem_static, 0), regs,
                     local_bytes, stream, traced, sampled, facts,
                     prior_identical_launches,
                     launch_args.exact ? "exact" : "unavailable",
                     function_prior_launches);
}

static cudaError_t sync_and_report(const char *where) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
    fprintf(stderr, "cyclescope-tracer %s: %s\n", where, cudaGetErrorString(e));
  return e;
}

static void leave_kernel_launch(CUcontext, CUfunction) {
  sync_and_report("leave_kernel_launch");
  /* An untraced launch pushes no records: skip the flush kernel and the
   * >=300 ms settle window (paid per launch in 1000-launch loops). */
  if (!g_current_launch_traced)
    return;
  if (capture_mode_placement) {
    /* placement mode: records sit in the device array; nothing went
     * through the channel */
    dump_placement_array();
    return;
  }
  skip_flag = true;
  flush_channel<<<1, 1>>>();
  sync_and_report("leave_kernel_launch.flush");
  skip_flag = false;
  /* Wait until no new record lands for a settle window; the device side is
   * already quiescent so a short window suffices. */
  auto idle_since = std::chrono::steady_clock::now();
  const auto deadline = idle_since + std::chrono::milliseconds(30000);
  while (std::chrono::steady_clock::now() < deadline) {
    const uint64_t before = g_records_written;
    for (int i = 0; i < 200; ++i) {
      sched_yield();
      if (g_records_written != before) {
        idle_since = std::chrono::steady_clock::now();
        break;
      }
    }
    if (std::chrono::steady_clock::now() - idle_since >
        std::chrono::milliseconds(300))
      break;
  }
  std::lock_guard<std::mutex> guard(g_record_mutex);
  flush_all_pending();
  close_trace_stream(g_launch_out);
}

/* ------------------------------------------------------------------ */
/* NVBit entry points                                                  */
/* ------------------------------------------------------------------ */

/* kernel launched by the host to flush the channel buffer; the
 * channel is the first member of the managed v5_dev state. */
__global__ void flush_channel() { V5_DEV.channel.flush(); }

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  const char *folder = getenv("CYCLESCOPE_TRACE_DIR");
  user_folder = folder ? folder : ".";
  const char *launches = getenv("CYCLESCOPE_CAPTURE_LAUNCHES");
  if (launches && launches[0])
    capture_launches = launches;
  if (const char *limit = getenv("CYCLESCOPE_CAPTURE_LAUNCH_LIMIT"))
    capture_launch_limit = strtoull(limit, nullptr, 10);
  const char *clusters = getenv("CYCLESCOPE_CAPTURE_CLUSTERS");
  if (clusters && clusters[0]) {
    if (strcmp(clusters, "all") == 0) {
      capture_clusters = -1;
    } else if (strncmp(clusters, "sampled:", 8) == 0) {
      capture_clusters_sampled = true;
      capture_clusters = atoi(clusters + 8);
    } else {
      /* comma-separated linear cluster ids */
      std::stringstream ss(clusters);
      std::string token;
      bool digits = true;
      while (std::getline(ss, token, ',')) {
        if (token.empty())
          continue;
        if (!std::all_of(token.begin(), token.end(), ::isdigit)) {
          digits = false;
          break;
        }
        capture_cluster_ids.push_back((uint32_t)std::stoul(token));
      }
      if (!digits) {
        fprintf(stderr,
                "cyclescope-tracer: unsupported CYCLESCOPE_CAPTURE_CLUSTERS=%s "
                "(all | sampled:<n> | <id,id,...>)\n",
                clusters);
        capture_cluster_ids.clear();
        capture_clusters = -1;
      }
    }
  }
  if (const char *mode = getenv("CYCLESCOPE_CAPTURE_MODE")) {
    if (strcmp(mode, "placement") == 0)
      capture_mode_placement = true;
  }
  {
    const char *fp = getenv("CYCLESCOPE_L2_FOOTPRINT");
    g_footprint_enabled = (!capture_mode_placement) &&
                          (fp == nullptr || fp[0] != '0');
    if (const char *n = getenv("CYCLESCOPE_L2_FOOTPRINT_ENTRIES"))
      g_footprint_entries_env = strtoll(n, nullptr, 10);
    else
      g_footprint_entries_env = 1 << 26;
    if (g_footprint_enabled) {
      const uint64_t entries = (uint64_t)g_footprint_entries_env;
      if (entries == 0 || (entries & (entries - 1)) != 0) {
        fprintf(stderr,
                "cyclescope-tracer: CYCLESCOPE_L2_FOOTPRINT_ENTRIES=%lld is not a "
                "positive power of two; disabling L2 footprint\n",
                (long long)g_footprint_entries_env);
        g_footprint_enabled = false;
      } else {
        g_footprint_entries = entries;
      }
    }
    if (const char *v = getenv("CYCLESCOPE_L2_FOOTPRINT_KERNEL"))
      g_footprint_kernel = v[0] != '0';
    if (const char *v = getenv("CYCLESCOPE_L2_FOOTPRINT_HOST_OPS"))
      g_footprint_host_ops_on = v[0] != '0';
  }
  if (const char *skip = getenv("CYCLESCOPE_TRACER_SKIP_OPCODES")) {
    std::stringstream ss(skip);
    std::string token;
    while (std::getline(ss, token, ','))
      if (!token.empty())
        g_skip_prefixes.push_back(token);
  }
  if (const char *range = getenv("CYCLESCOPE_TRACER_IDX_RANGE")) {
    long long lo = -1, hi = -1;
    if (sscanf(range, "%lld-%lld", &lo, &hi) == 2 && lo >= 0 && hi >= lo) {
      g_idx_lo = lo;
      g_idx_hi = hi;
    }
  }
  if (const char *bare = getenv("CYCLESCOPE_TRACER_BARE_OPCODES")) {
    std::stringstream ss(bare);
    std::string token;
    while (std::getline(ss, token, ','))
      if (!token.empty())
        g_bare_prefixes.push_back(token);
  }
  for (const auto &[name, dest] :
       std::vector<std::pair<const char *, std::vector<std::string> *>>{
           {"CYCLESCOPE_TRACER_NO_MREF_OPCODES", &g_nomref_prefixes},
           {"CYCLESCOPE_TRACER_NO_RANK_OPCODES", &g_norank_prefixes},
           {"CYCLESCOPE_TRACER_NO_XRANK_OPCODES", &g_noxrank_prefixes},
           {"CYCLESCOPE_TRACER_NO_UR_OPCODES", &g_nour_prefixes},
           {"CYCLESCOPE_TRACER_NO_R_OPCODES", &g_nor_prefixes}}) {
    if (const char *v = getenv(name)) {
      std::stringstream ss(v);
      std::string token;
      while (std::getline(ss, token, ','))
        if (!token.empty())
          dest->push_back(token);
    }
  }
  if (const char *na = getenv("CYCLESCOPE_TRACER_NO_AFTER"))
    no_after_hook = na[0] == '1';
  if (const char *ng = getenv("CYCLESCOPE_TRACER_NO_GLOBAL_SYNC"))
    no_gsync = ng[0] == '1';
  const char *extra = getenv("CYCLESCOPE_TRACER_OPERAND_OPCODES");
  /* Family prefixes come from the arch table (shared union); the
   * CYCLESCOPE_TRACER_OPERAND_FAMILIES/OPCODES debug knobs still override. */
  g_operand_prefixes.assign(
      kOperandFamilyPrefixes,
      kOperandFamilyPrefixes + K_OPERAND_FAMILY_COUNT);
  g_operand_exact.assign(kOperandFamilyExact,
                         kOperandFamilyExact + K_OPERAND_EXACT_COUNT);
  {
    const char *fam = getenv("CYCLESCOPE_TRACER_OPERAND_FAMILIES");
    if (fam && fam[0]) {
      g_operand_prefixes.clear();
      g_operand_exact.clear();
      if (strcmp(fam, "none") != 0) {
        std::stringstream ss(fam);
        std::string token;
        while (std::getline(ss, token, ','))
          if (!token.empty())
            g_operand_prefixes.push_back(token);
      } else {
        no_operands = true;
      }
    }
  }
  if (extra && extra[0]) {
    std::stringstream ss(extra);
    std::string token;
    while (std::getline(ss, token, ','))
      if (!token.empty())
        g_operand_prefixes.push_back(token);
  }
  const char *v = getenv("CYCLESCOPE_TRACER_VERBOSE");
  if (v)
    verbose = atoi(v);
  {
    const char *ss = getenv("CYCLESCOPE_CUDA_STACK_SIZE");
    if (ss && ss[0])
      cuda_stack_size = (uint32_t)strtoul(ss, nullptr, 0);
  }
  have_zstd = (system("command -v zstd >/dev/null 2>&1") == 0);
  g_tma_mutex = new std::mutex;
  mkdir(user_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  std::string fdir = user_folder + "/functions";
  mkdir(fdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  g_functions_jsonl =
      fopen((user_folder + "/functions.jsonl").c_str(), "w");
  g_launches_jsonl = fopen((user_folder + "/launches.jsonl").c_str(), "w");
  g_tensor_maps_jsonl = fopen((user_folder + "/tensor_maps.jsonl").c_str(), "w");
  printf("cyclescope-tracer output=%s\n", user_folder.c_str());
}

void nvbit_tool_init(CUcontext ctx) {
  select_arch_table(ctx);
  recv_thread_started = true;
  recv_thread_receiving = true;
  new (&V5_DEV) V5DevState();
  V5_DEV.sampling.count = -1;
  V5_DEV.stop_report = false;
  channel_host.init(0, CHANNEL_SIZE, &V5_DEV.channel, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, nullptr);
  if (g_footprint_enabled &&
      (g_footprint_kernel || g_footprint_host_ops_on) &&
      g_footprint_ring_dev == NULL) {
    cudaError_t err =
        cudaMalloc(&g_footprint_ring_dev, g_footprint_entries * 8);
    if (err == cudaSuccess)
      err = cudaMalloc(
          reinterpret_cast<void **>(&g_footprint_counter_dev), 8);
    if (err == cudaSuccess)
      err = cudaMemset(g_footprint_counter_dev, 0, 8);
    if (err == cudaSuccess) {
      g_footprint_cell.ring = (uint64_t *)g_footprint_ring_dev;
      g_footprint_cell.counter = g_footprint_counter_dev;
      g_footprint_cell.mask = g_footprint_entries - 1;
      g_footprint_cell.active = 0;
    } else {
      fprintf(stderr, "cyclescope-tracer: missing_fact:l2_footprint_alloc (%s)\n",
              cudaGetErrorString(err));
      g_footprint_enabled = false;
    }
  }
}

void nvbit_at_ctx_init(CUcontext ctx) { select_arch_table(ctx); }

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
  }
  {
    FILE *pf = placement_out();
    if (pf)
      fflush(pf);
  }
  for (auto it = g_func_regs.begin(); it != g_func_regs.end();) {
    if (it->second.ctx == ctx) {
      g_instrumented.erase(it->first);
      it = g_func_regs.erase(it);
    } else {
      ++it;
    }
  }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;
  switch (cbid) {
  case API_CUDA_cuLaunchKernel_ptsz:
  case API_CUDA_cuLaunchKernel:
  case API_CUDA_cuLaunchCooperativeKernel:
  case API_CUDA_cuLaunchCooperativeKernel_ptsz:
  case API_CUDA_cuLaunchKernelEx:
  case API_CUDA_cuLaunchKernelEx_ptsz:
  case API_CUDA_cuLaunchGridAsync: {
    CUfunction func;
    CUstream stream;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
      auto *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
      stream = p->config->hStream;
    } else if (cbid == API_CUDA_cuLaunchGridAsync) {
      auto *p = (cuLaunchGridAsync_params *)params;
      func = p->f;
      stream = p->hStream;
    } else {
      auto *p = (cuLaunchKernel_params *)params;
      func = p->f;
      stream = p->hStream;
    }
    cudaStreamCaptureStatus status;
    CUDA_SAFECALL(cudaStreamIsCapturing(stream, &status));
    if (!is_exit) {
      enter_kernel_launch(ctx, func, cbid, params, name,
                          status == cudaStreamCaptureStatusActive);
    } else if (status != cudaStreamCaptureStatusActive) {
      leave_kernel_launch(ctx, func);
    }
  } break;
  case API_CUDA_cuLaunch:
  case API_CUDA_cuLaunchGrid: {
    auto *p = (cuLaunch_params *)params;
    if (!is_exit)
      enter_kernel_launch(ctx, p->f, cbid, params, name, false);
    else
      leave_kernel_launch(ctx, p->f);
  } break;
  case API_CUDA_cuGraphLaunch: {
    if (is_exit) {
      auto *p = (cuGraphLaunch_params *)params;
      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      skip_flag = true;
      flush_channel<<<1, 1, 0, p->hStream>>>();
      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      skip_flag = false;
    }
  } break;
  case API_CUDA_cuMemcpyHtoD_v2:
  case API_CUDA_cuMemcpyHtoD_v2_ptds:
  case API_CUDA_cuMemcpyHtoDAsync_v2:
  case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz:
  case API_CUDA_cuMemcpyDtoD_v2:
  case API_CUDA_cuMemcpyDtoD_v2_ptds:
  case API_CUDA_cuMemcpyDtoDAsync_v2:
  case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
  case API_CUDA_cuMemcpyDtoH_v2:
  case API_CUDA_cuMemcpyDtoH_v2_ptds:
  case API_CUDA_cuMemcpyDtoHAsync_v2:
  case API_CUDA_cuMemcpyDtoHAsync_v2_ptsz:
  case API_CUDA_cuMemcpy:
  case API_CUDA_cuMemcpyAsync:
  case API_CUDA_cuMemcpyAsync_ptsz:
  case API_CUDA_cuMemsetD8_v2:
  case API_CUDA_cuMemsetD8_v2_ptds:
  case API_CUDA_cuMemsetD16_v2:
  case API_CUDA_cuMemsetD16_v2_ptds:
  case API_CUDA_cuMemsetD32_v2:
  case API_CUDA_cuMemsetD32_v2_ptds:
  case API_CUDA_cuMemsetD8Async:
  case API_CUDA_cuMemsetD8Async_ptsz:
  case API_CUDA_cuMemsetD16Async:
  case API_CUDA_cuMemsetD16Async_ptsz:
  case API_CUDA_cuMemsetD32Async:
  case API_CUDA_cuMemsetD32Async_ptsz: {
    if (is_exit && pStatus != NULL && *pStatus == CUDA_SUCCESS &&
        g_footprint_enabled && g_footprint_host_ops_on &&
        !g_selected_launch_seen) {
      record_footprint_host_op(cbid, params);
    }
  } break;
  case API_CUDA_cuMemAlloc_v2:
  case API_CUDA_cuMemAllocPitch_v2:
  case API_CUDA_cuMemAllocAsync:
  case API_CUDA_cuMemAllocAsync_ptsz:
  case API_CUDA_cuMemFree_v2:
  case API_CUDA_cuMemFreeAsync:
  case API_CUDA_cuMemFreeAsync_ptsz: {
    if (is_exit && g_footprint_enabled && g_footprint_host_ops_on &&
        !g_selected_launch_seen)
      record_footprint_mem_event(cbid, params, pStatus);
  } break;
  case API_CUDA_cuMemcpyPeer:
  case API_CUDA_cuMemcpyPeer_ptds:
  case API_CUDA_cuMemcpyPeerAsync:
  case API_CUDA_cuMemcpyPeerAsync_ptsz:
  case API_CUDA_cuMemcpy2D_v2:
  case API_CUDA_cuMemcpy2D_v2_ptds:
  case API_CUDA_cuMemcpy2DUnaligned_v2:
  case API_CUDA_cuMemcpy2DUnaligned_v2_ptds:
  case API_CUDA_cuMemcpy2DAsync_v2:
  case API_CUDA_cuMemcpy2DAsync_v2_ptsz:
  case API_CUDA_cuMemcpy3D_v2:
  case API_CUDA_cuMemcpy3D_v2_ptds:
  case API_CUDA_cuMemcpy3DAsync_v2:
  case API_CUDA_cuMemcpy3DAsync_v2_ptsz:
  case API_CUDA_cuMemcpy3DPeer:
  case API_CUDA_cuMemcpy3DPeer_ptds:
  case API_CUDA_cuMemcpy3DPeerAsync:
  case API_CUDA_cuMemcpy3DPeerAsync_ptsz:
  case API_CUDA_cuMemsetD2D8_v2:
  case API_CUDA_cuMemsetD2D8_v2_ptds:
  case API_CUDA_cuMemsetD2D16_v2:
  case API_CUDA_cuMemsetD2D16_v2_ptds:
  case API_CUDA_cuMemsetD2D32_v2:
  case API_CUDA_cuMemsetD2D32_v2_ptds:
  case API_CUDA_cuMemsetD2D8Async:
  case API_CUDA_cuMemsetD2D8Async_ptsz:
  case API_CUDA_cuMemsetD2D16Async:
  case API_CUDA_cuMemsetD2D16Async_ptsz:
  case API_CUDA_cuMemsetD2D32Async:
  case API_CUDA_cuMemsetD2D32Async_ptsz: {
    if (is_exit && pStatus != NULL && *pStatus == CUDA_SUCCESS &&
        g_footprint_enabled && g_footprint_host_ops_on &&
        !g_selected_launch_seen)
      ++g_host_ops_unsupported;
  } break;
  case API_CUDA_cuTensorMapEncodeTiled:
  case API_CUDA_cuTensorMapEncodeIm2col:
    if (is_exit)
      tma_intercept_encode(cbid, params, pStatus);
    break;
  case API_CUDA_cuTensorMapReplaceAddress:
    tma_intercept_host_replace(is_exit, params, pStatus);
    break;
  case API_CUDA_cuModuleGetFunction:
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      auto *p = (cuModuleGetFunction_params *)params;
      if (p->hfunc && *p->hfunc)
        g_module_of_function[*p->hfunc] = p->hmod;
    }
    break;
  case API_CUDA_cuModuleUnload:
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      CUmodule mod = ((cuModuleUnload_params *)params)->hmod;
      for (auto it = g_func_regs.begin(); it != g_func_regs.end();) {
        if (it->second.module == mod) {
          g_instrumented.erase(it->first);
          it = g_func_regs.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = g_module_of_function.begin();
           it != g_module_of_function.end();) {
        if (it->second == mod)
          it = g_module_of_function.erase(it);
        else
          ++it;
      }
    }
    break;
  case API_CUDA_cuGetProcAddress_v2:
  case API_CUDA_cuGetProcAddress:
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      auto *p = (cuGetProcAddress_params *)params;
      tma_intercept_driver_entry(p->symbol, p->pfn);
    }
    break;
  default:
    break;
  }
}
