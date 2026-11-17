// Mechanism-tree microbenchmarks: shared device/host helpers (B300 / sm_103a).
//
// Timing convention: every probe times a straight-line (compile-time
// unrolled) body between two SR_CLOCKLO reads, repeated `reps` times per
// launch; the first repetition of each launch is a warm-up and is dropped on
// the host.  Leaf values are derived from differences between variants that
// share the same surrounding code (chain slopes, variant-minus-baseline), so
// the clock-read and loop constants cancel; see calibration/mechtree/README.md.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define MB_CHECK(call)                                                       \
  do {                                                                       \
    const cudaError_t mb_err__ = (call);                                     \
    if (mb_err__ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(mb_err__));                            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------- device

// Set by timed-out waits (mb_mbar_wait_to & co); checked by the host after
// every launch.
__device__ std::uint32_t g_mb_err;

__device__ __forceinline__ std::uint64_t mb_clock() {
  std::uint64_t v;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(v)::"memory");
  return v;
}

// Timeout clock for give-up paths: %globaltimer (ns), so the SASS gate's
// SR_CLOCKLO windows only ever contain the two measurement reads.
__device__ __forceinline__ std::uint64_t mb_gtimer() {
  std::uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)::"memory");
  return v;
}

__device__ __forceinline__ std::uint32_t mb_smem_u32(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

// Opaque copy: keeps a value in a register so ptxas cannot rematerialize
// its computation (e.g. S2R SR_CgaCtaId) inside a timed window.
__device__ __forceinline__ std::uint32_t mb_opaque(std::uint32_t v) {
  std::uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(v));
  return r;
}

// Pin through shared memory: ptxas rematerializes plain movs and kernel
// parameters (LDCU) inside loops; a volatile shared load it cannot.  `slot`
// is a per-thread __shared__ word.
__device__ __forceinline__ std::uint32_t mb_pin(std::uint32_t v, std::uint32_t* slot) {
  const std::uint32_t a = mb_smem_u32(slot);
  std::uint32_t r;
  asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(r) : "r"(a) : "memory");
  return r;
}

__device__ __forceinline__ std::uint32_t mb_ctarank() {
  std::uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

// CUTLASS elect_one_sync form: one lane of the converged warp gets 1.
__device__ __forceinline__ std::uint32_t mb_elect_one() {
  std::uint32_t pred;
  asm volatile(
      "{\n .reg .pred P;\n .reg .b32 laneid;\n"
      "elect.sync _|P, 0xffffffff;\n"
      "selp.u32 %0, 1, 0, P;\n}"
      : "=r"(pred));
  return pred;
}

__device__ __forceinline__ std::uint32_t mb_smid() {
  std::uint32_t r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void mb_cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void mb_mbar_init(std::uint32_t mbar, std::uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count) : "memory");
}

__device__ __forceinline__ void mb_fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Spin until the phase with parity `parity` of the CTA-local mbarrier has
// completed (the SASS form is SYNCS.PHASECHK.TRANS64.TRYWAIT + BRA, the same
// shape CUTLASS/ptxas emit for pipeline consumer waits).
__device__ __forceinline__ void mb_mbar_wait(std::uint32_t mbar, std::uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "MB_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra MB_WAIT_%=;\n\t}\n" ::"r"(mbar),
      "r"(parity)
      : "memory");
}

// Same shape as mb_mbar_wait (TRYWAIT + branch, retries out of line), but
// the retry loop gives up after 2^22 polls and sets *err, so a missing
// arrival fails the variant instead of hanging the probe.
__device__ __forceinline__ void mb_mbar_wait_to(std::uint32_t mbar, std::uint32_t parity,
                                                std::uint32_t* err = &g_mb_err) {
  asm volatile(
      "{\n\t.reg .pred p, q;\n\t.reg .u32 n;\n\t"
      "mov.u32 n, 0;\n\t"
      "MB_WTO_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@p bra MB_WTO_DONE_%=;\n\t"
      "add.u32 n, n, 1;\n\t"
      "setp.lt.u32 q, n, 4194304;\n\t"
      "@q bra MB_WTO_%=;\n\t"
      "st.global.u32 [%2], 1;\n\t"
      "MB_WTO_DONE_%=:\n\t}\n" ::"r"(mbar),
      "r"(parity), "l"(err)
      : "memory");
}

// Non-suspending detector: spins on mbarrier.test_wait (SYNCS.PHASECHK
// without TRYWAIT), so a phase completion is seen within one poll of
// ~the test_wait result latency instead of on the try_wait suspend grid.
__device__ __forceinline__ void mb_mbar_spin_to(std::uint32_t mbar, std::uint32_t parity,
                                                std::uint32_t* err = &g_mb_err) {
  asm volatile(
      "{\n\t.reg .pred p, q;\n\t.reg .u32 n;\n\t"
      "mov.u32 n, 0;\n\t"
      "MB_SPN_%=:\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@p bra MB_SPN_DONE_%=;\n\t"
      "add.u32 n, n, 1;\n\t"
      "setp.lt.u32 q, n, 67108864;\n\t"
      "@q bra MB_SPN_%=;\n\t"
      "st.global.u32 [%2], 1;\n\t"
      "MB_SPN_DONE_%=:\n\t}\n" ::"r"(mbar),
      "r"(parity), "l"(err)
      : "memory");
}

__device__ __forceinline__ void mb_mbar_arrive(std::uint32_t mbar) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.shared::cta.b64 st, [%0];\n\t}\n" ::"r"(mbar)
      : "memory");
}

// tcgen05 allocation helpers (warp-collective).  sm_90a: tcgen05 does
// not exist; the helpers become no-ops so cta_group variants compile but
// produce no valid timing (recorded as unrealised per protocol).
#if __CUDA_ARCH__ >= 1000
template <int CG>
__device__ __forceinline__ void mb_tmem_alloc(std::uint32_t slot_smem, std::uint32_t ncols) {
  if constexpr (CG == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(ncols) : "memory");
  } else {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot_smem), "r"(ncols) : "memory");
  }
}

template <int CG>
__device__ __forceinline__ void mb_tmem_dealloc(std::uint32_t taddr, std::uint32_t ncols) {
  if constexpr (CG == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
  } else {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
  }
}

template <int CG>
__device__ __forceinline__ void mb_tmem_relinquish() {
  if constexpr (CG == 1) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  } else {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
}

__device__ __forceinline__ void mb_tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void mb_tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void mb_tmem_fill(std::uint32_t taddr, int ncols, std::uint32_t v) {
#pragma unroll 1
  for (int c = 0; c < ncols; c += 8) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};" ::"r"(taddr + static_cast<std::uint32_t>(c)),
        "r"(v)
        : "memory");
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

#else
template <int CG>
__device__ __forceinline__ void mb_tmem_alloc(std::uint32_t, std::uint32_t) {}
template <int CG>
__device__ __forceinline__ void mb_tmem_dealloc(std::uint32_t, std::uint32_t) {}
template <int CG>
__device__ __forceinline__ void mb_tmem_relinquish() {}
__device__ __forceinline__ void mb_tc_fence_before() {}
__device__ __forceinline__ void mb_tc_fence_after() {}
__device__ __forceinline__ void mb_tmem_fill(std::uint32_t, int, std::uint32_t) {}
#endif

// ---------------------------------------------------------------- host

struct MbDevice {
  std::string name, uuid;
  int sm_clock_khz{}, sm_count{}, major{}, minor{};
};

inline MbDevice mb_query_device() {
  MbDevice d;
  cudaDeviceProp p{};
  int device = 0;
  MB_CHECK(cudaGetDevice(&device));
  MB_CHECK(cudaGetDeviceProperties(&p, device));
  d.name = p.name;
  d.uuid = "GPU-";
  constexpr char hex[] = "0123456789abcdef";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) d.uuid += '-';
    const auto byte = static_cast<unsigned char>(p.uuid.bytes[i]);
    d.uuid += hex[byte >> 4];
    d.uuid += hex[byte & 15];
  }
  MB_CHECK(cudaDeviceGetAttribute(&d.sm_clock_khz, cudaDevAttrClockRate, device));
  d.sm_count = p.multiProcessorCount;
  d.major = p.major;
  d.minor = p.minor;
  return d;
}

struct MbArgs {
  std::string out_dir = ".";
  int launches = 8;
  int reps = 33;  // first repetition of each launch is a warm-up
  std::string only;  // optional variant-name substring filter
  std::string cells;  // optional cell-set selector (e.g. "32k")
  int family = 0;
  int forms = 0;    // 1 = operand-form subset (poffself/ppoff2 + ppoff/poff for the 8 classes)    // 0 all, 1 = natives only, 2 = patched only, 3 = list patched names
};

inline MbArgs mb_parse(int argc, char** argv) {
  MbArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string k(argv[i]);
    auto need = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--out") a.out_dir = need();
    else if (k == "--launches") a.launches = std::atoi(need());
    else if (k == "--reps") a.reps = std::atoi(need());
    else if (k == "--only") a.only = need();
    else if (k == "--cells") a.cells = need();
    else if (k == "--no-patched") a.family = 1;
    else if (k == "--patched-only") a.family = 2;
    else if (k == "--list-patched") a.family = 3;
    else if (k == "--forms") a.forms = 1;
    else {
      std::fprintf(stderr, "unknown argument %s\n", k.c_str());
      std::exit(2);
    }
  }
  // reps=1 is for replay captures only: the converter's CLOCKLO marker
  // requires exactly two dynamic SR_CLOCKLO reads per warp, so the traced
  // run times one rep (which the hw-side rep-0 drop leaves with n=0; the
  // untraced reference run still uses reps>=2)
  if (a.launches < 1 || a.reps < 1) std::exit(2);
  return a;
}

struct MbWriter {
  FILE* f{};
  std::string probe;
  MbDevice dev;
  int failures{};

  MbWriter(const MbArgs& a, const char* probe_name) : probe(probe_name), dev(mb_query_device()) {
    const std::string path = a.out_dir + "/" + probe + ".jsonl";
    f = std::fopen(path.c_str(), a.family == 2 ? "a" : "w");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", path.c_str());
      std::exit(1);
    }
  }
  ~MbWriter() {
    if (f) std::fclose(f);
  }

  // One row per (variant, u): raw per-repetition cycle counts.
  void row(const std::string& variant, int u, const std::string& params_json,
           const std::vector<double>& samples, const std::string& extra_json = "") {
    std::fprintf(f, "{\"schema\":\"mechtree.mb.v1\",\"probe\":\"%s\",\"variant\":\"%s\",\"u\":%d,",
                 probe.c_str(), variant.c_str(), u);
    std::fprintf(f, "\"params\":{%s},\"device\":\"%s\",\"gpu_uuid\":\"%s\",\"cc\":\"%d.%d\",\"sm_clock_khz\":%d,",
                 params_json.c_str(), dev.name.c_str(), dev.uuid.c_str(), dev.major, dev.minor, dev.sm_clock_khz);
    std::fprintf(f, "\"utc\":%lld,\"n\":%zu,\"samples\":[", static_cast<long long>(std::time(nullptr)),
                 samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
      std::fprintf(f, i ? ",%.1f" : "%.1f", samples[i]);
    std::fprintf(f, "]%s%s}\n", extra_json.empty() ? "" : ",", extra_json.c_str());
    std::fflush(f);
  }

  void fail(const std::string& variant, int u, const std::string& params_json, const char* why) {
    ++failures;
    std::fprintf(f, "{\"schema\":\"mechtree.mb.v1\",\"probe\":\"%s\",\"variant\":\"%s\",\"u\":%d,"
                    "\"params\":{%s},\"error\":\"%s\"}\n",
                 probe.c_str(), variant.c_str(), u, params_json.c_str(), why);
    std::fflush(f);
    std::fprintf(stderr, "FAIL %s u=%d: %s\n", variant.c_str(), u, why);
  }
};

// Copy the per-repetition records of one launch and append all but the
// warm-up repetition.  `stride` lets a kernel store several words per rep.
inline void mb_collect(const std::uint64_t* d_out, int reps, int stride, int word,
                       std::vector<double>& dst) {
  std::vector<std::uint64_t> h(static_cast<std::size_t>(reps) * stride);
  MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
  for (int r = 1; r < reps; ++r) dst.push_back(static_cast<double>(h[static_cast<std::size_t>(r) * stride + word]));
}

// Read-and-clear the device timeout flag.
inline bool mb_take_err() {
  std::uint32_t e = 0, z = 0;
  MB_CHECK(cudaMemcpyFromSymbol(&e, g_mb_err, sizeof(e)));
  MB_CHECK(cudaMemcpyToSymbol(g_mb_err, &z, sizeof(z)));
  return e != 0;
}

// --only accepts comma-separated substrings; prefix a part with '=' to match
// a complete variant name.
inline bool mb_selected(const MbArgs& a, const std::string& variant) {
  if (a.only.empty()) return true;
  std::size_t start = 0;
  while (start <= a.only.size()) {
    const std::size_t end = std::min(a.only.find(',', start), a.only.size());
    const std::string part = a.only.substr(start, end - start);
    const bool exact = !part.empty() && part.front() == '=';
    const std::string needle = exact ? part.substr(1) : part;
    if (!needle.empty() &&
        (exact ? variant == needle
               : variant.find(needle) != std::string::npos)) {
      return true;
    }
    start = end + 1;
  }
  return false;
}
