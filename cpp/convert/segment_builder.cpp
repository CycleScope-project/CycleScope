// C++ 3a segment builder — port of python/cyclescope/job/segment_builder.py
// emit order and dedup semantics preserved (byte-identity oracle).
#include "convert/segment_builder.hpp"
#include "convert/convert_rules.hpp"
#include "convert/semantics.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <cstdio>

namespace cyclescope_convert {

namespace {

constexpr std::uint32_t NONE32 = 0xFFFFFFFFu;
constexpr std::uint64_t SECTOR_BYTES = 32;
constexpr std::uint64_t FETCH_LINE_BYTES = 128;
constexpr std::uint32_t MACC_REMOTE = 0x20;
constexpr std::uint32_t MACC_UNIFORM = 0x10;
constexpr std::uint32_t MACC_L1_ALLOCATE = 0x200;

// segment row sizes (python job/segment_format.py)
constexpr std::size_t S_CTA = 56, S_WARP = 32, S_TMPL = 68, S_INST = 16,
                      S_REGS = 4, S_MACC = 48, S_MSEC = 16, S_CODE = 16,
                      S_META = 120, S_MBPI = 24, S_SPLT = 16, S_TIME = 40,
                      S_USES = 38, S_FACT = 12, S_SEMA = 60, S_MBOL = 4,
                      S_CLKM = 16;

// facts row sizes (python convert/facts_format.py)
constexpr std::size_t F_WARP = 40, F_DYN = 28, F_MREF = 28, F_MREX = 20,
                      F_ADDR = 8, F_STAT = 64, F_STAX = 24, F_OPTK = 4,
                      F_REGL = 4, F_CTAS = 48, F_OPS = 32, F_OPSX = 20,
                      F_GSYN = 40, F_GSYX = 8,
                      F_UKV = 13, F_LANE = 12, F_LVAL = 8, F_TMAP = 44,
                      F_TMDI = 8;

inline void put8(std::vector<std::uint8_t> &b, std::uint64_t v) {
  const auto n = b.size();
  b.resize(n + 8);
  std::memcpy(b.data() + n, &v, 8);   // little-endian host
}
inline void put4(std::vector<std::uint8_t> &b, std::uint32_t v) {
  const auto n = b.size();
  b.resize(n + 4);
  std::memcpy(b.data() + n, &v, 4);
}
inline void put2(std::vector<std::uint8_t> &b, std::uint16_t v) {
  b.push_back(v & 0xFF);
  b.push_back((v >> 8) & 0xFF);
}
inline std::string upper(std::string s) {
  for (auto &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}
inline void put1(std::vector<std::uint8_t> &b, std::uint8_t v) {
  b.push_back(v);
}
inline void put32s(std::vector<std::uint8_t> &b, const std::string &s) {
  b.insert(b.end(), s.begin(), s.begin() + std::min<std::size_t>(32, s.size()));
  b.resize(b.size() + (32 - std::min<std::size_t>(32, s.size())), 0);
}
inline std::uint32_t rd32(const std::uint8_t *p) {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline std::uint64_t rd64(const std::uint8_t *p) {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

std::uint32_t tag4(const char *t) {
  std::uint32_t v = 0;
  std::memcpy(&v, t, 4);
  return v;
}

// facts MREF codes -> emit codes
int space_code(int sp) {
  // facts _SPACE: global0 shared1 local2 constant3 generic4 param5
  // shared_cluster6 global_system7
  switch (sp) {
    case 0: return 0x0;
    case 1: return 0x1;
    case 2: return 0x2;
    case 3: return 0x3;
    case 4: return 0x4;
    case 5: return 0x4;    // param -> generic
    case 6: return 0x1;    // shared_cluster -> shared
    case 7: return 0x0;    // global_system -> global
    default: return 0x4;   // generic fallback
  }
}
int access_code(int ak) {
  // facts _AK: load0 store1 atomic2 async_copy3 async_reduce4 prefetch5
  // reduction6 other7
  switch (ak) {
    case 0: return 0x0;
    case 1: return 0x40;
    case 2: return 0x80;
    case 3: return 0xC0;
    case 4: return 0x100;
    case 5: return 0x140;
    default: return 0;
  }
}

std::string root_of(const std::string &opcode) {
  const auto d = opcode.find('.');
  return d == std::string::npos ? opcode : opcode.substr(0, d);
}

static std::uint64_t local_shared_address(std::uint64_t a) {
  return a & 0xFFFFFF;  // strip DSMEM rank tag (rank << 24)
}

}  // namespace

SegmentBuilder::SegmentBuilder(FactsFile &facts, FactsFile &programs,
                               const ConvertRules *rules,
                               const BuildMeta &meta,
                               const std::string &arch,
                               const TreeRt &rt, StaticCache *shared)
    : f_(facts), rules_(rules), meta_(meta),
      cache_(shared && facts.has_shared_statics() ? *shared : own_cache_),
      statics_(cache_.statics), rt_(&rt),
      statics_as_info_(cache_.infos), tmpl_key_prefix_(cache_.tmpl_key_prefix),
      reg_ids_(cache_.reg_ids), arch_name_(arch) {
  axctx_.static_axis_cache = &cache_.static_axis_cache;
  static const char *kTags[] = {"CTAS", "WARP", "TMPL", "INST", "REGS",
                                "SEMA", "MACC", "MSEC", "TIME", "FACT",
                                "USES", "SPLT", "MBPI", "MBOL", "CODE",
                                "CLKM", "AXIS", "PROG", "NODE", "GRD ",
                                "LEAF", "PARM", "PTS ", "META"};
  for (auto *t : kTags) tables_[tag4(t)];
  // program table bytes (copied verbatim); only FACT/USES are kept:
  // they are emitted with per-segment rows appended, while the other
  // seven shared tables live in program.tables under v2 and are never
  // read back from tables_
  for (auto *t : {"FACT", "USES"}) {
    std::size_t n = 0;
    if (const auto *p = programs.table(t, &n))
      tables_[tag4(t)].assign(p, p + n);
  }
  if (const auto *sm = programs.table("SCHM", nullptr)) {
    std::size_t sn = 0;
    programs.table("SCHM", &sn);
    for (std::size_t i = 0; i + 4 <= sn; i += 4)
      schm_target_[*(const std::uint16_t *)(sm + i)] =
          *(const std::uint8_t *)(sm + i + 2);
  }
  fact_count_ = tables_[tag4("FACT")].size() / 12;
  axctx_.n_ctas = (int)(t_.ctas_n / F_CTAS);
  axctx_.sem = &sem_ctx_;
  axctx_.tmaps = &sem_ctx_.tensor_maps;
  axctx_.rules = rules_;
  axctx_.arch = arch_name_;
  axctx_.arch_bw = arch_name_ == "blackwell";
  arch_kind_ = arch_name_ == "hopper" ? 1
                                      : arch_name_ == "blackwell" ? 2 : 0;
  axctx_.statics = &statics_as_info_;
  axctx_.block = {meta_.block[0], meta_.block[1],
                     meta_.block[2]};
  axctx_.registers_per_thread = meta_.registers_per_thread;
  axctx_.declared_math_warpgroups =
      meta_.declared_math_warpgroups;
  if (rt_->axis_index.count("operation_class"))
    for (auto &v :
         rt_->axes[rt_->axis_index.at("operation_class")].values)
      class_domain_.insert(v);
  if (const auto *s = programs.table("STRS", nullptr)) {
    std::size_t n = 0;
    programs.table("STRS", &n);
    strings_.assign(s, s + n);
    // python STRS_OFFSETS contains only names interned via str_off:
    // PROG.mechanism offsets + AXIS.name offsets — NOT the op-class
    // runs (which sit in STRS but are never in the dict).
    const char *base = reinterpret_cast<const char *>(s);
    if (const auto *p = programs.table("PROG", nullptr)) {
      std::size_t pn = 0;
      programs.table("PROG", &pn);
      for (std::size_t i = 0; i + 32 <= pn; i += 32) {
        const auto off = rd32(p + i);
        if (off < n) string_offsets_[base + off] = off;
      }
    }
    if (const auto *a = programs.table("AXIS", nullptr)) {
      std::size_t an = 0;
      programs.table("AXIS", &an);
      for (std::size_t i = 0; i + 4 <= an; i += 4) {
        const auto off = rd32(a + i);
        if (off < n) string_offsets_[base + off] = off;
      }
    }
  }
  t_.ctas = facts.table("CTAS", &t_.ctas_n);
  t_.warp = facts.table("WARP", &t_.warp_n);
  t_.dyn = facts.table("DYNI", &t_.dyn_n);
  t_.mref = facts.table("MREF", &t_.mref_n);
  t_.mrex = facts.table("MREX", &t_.mrex_n);
  t_.addr = facts.table("ADDR", nullptr);
  t_.stat = facts.table("STAT", &t_.stat_n);
  t_.stax = facts.table("STAX", nullptr);
  t_.optk = facts.table("OPTK", nullptr);
  t_.regl = facts.table("REGL", nullptr);
  t_.ops = facts.table("OPSZ", nullptr);
  t_.opsx = facts.table("OPSX", nullptr);
  t_.ukv = facts.table("UKVT", nullptr);
  t_.lane = facts.table("LANE", nullptr);
  t_.lval = facts.table("LVAL", nullptr);
  t_.tmap = facts.table("TMAP", nullptr);
  t_.tmdi = facts.table("TMDI", nullptr);
  t_.mbpi = facts.table("MBPI", &t_.mbpi_n);
  t_.gsyn = facts.table("GSYN", nullptr);
  t_.gsynx = facts.table("GSYX", nullptr);
  sem_ctx_.n_ctas = t_.ctas_n / F_CTAS;
  sem_ctx_.local_shared = &local_shared_address;
  {
    const auto *cx = t_.ctas;
    for (std::uint64_t i = 0; i + F_CTAS <= t_.ctas_n; i += F_CTAS) {
      // facts CTAS: cta_linear@0 .. cluster_rank@16
      sem_ctx_.cta_rank[rd32(cx + i)] = rd32(cx + i + 16);
    }
    // facts WARP: cta_linear@0
    for (std::uint64_t i = 0; i + F_WARP <= t_.warp_n; i += F_WARP)
      sem_ctx_.cta_warps[rd32(t_.warp + i)] += 1;
  }
  // tensor maps: TMAP <3I4Q> id, elem_bytes, rank, pad, gaddr,
  // dims_begin, strides_begin, box_begin (TMDI indices); each map's
  // TMDI run is contiguous (global_dim, stride_bytes, box_dim), so a
  // box ends at the next map's dims_begin (or the table end).
  if (t_.tmap) {
    std::size_t tn = 0;
    facts.table("TMAP", &tn);
    std::size_t tm_n = tn / F_TMAP;
    std::size_t tmdi_n = 0;
    facts.table("TMDI", &tmdi_n);
    for (std::size_t i = 0; i < tm_n; ++i) {
      const auto *tm = t_.tmap + i * F_TMAP;
      // <3I4Q> fields: id@0, elem@4, rank@8, gaddr@12, dims_begin@20,
      // strides_begin@28, box_begin@36 (TMDI indices)
      const auto tm_id = rd32(tm);
      const auto eb = rd32(tm + 4);
      const auto bb = rd64(tm + 36);
      std::uint64_t next = (i + 1 < tm_n)
          ? rd64(t_.tmap + (i + 1) * F_TMAP + 20)
          : tmdi_n / F_TMDI;
      std::vector<std::uint64_t> box;
      for (std::uint64_t j = bb; j < next; ++j)
        box.push_back(rd64(t_.tmdi + j * F_TMDI));
      sem_ctx_.tensor_maps[tm_id] = {eb, box};
    }
  }
  t_.ctas_n /= F_CTAS;
  t_.warp_n /= F_WARP;
  t_.dyn_n /= F_DYN;
  t_.mref_n /= F_MREF;
  // store-as-init prescan: shared addresses named by explicit barrier
  // ops anywhere in the unit (arch promote hooks consume this set;
  // e.g. blackwell STS.64 -> mbarrier_init).
  // Only captured barrier-word facts name barriers (mbarrier_address and
  // the per-lane mbarrier_words of executing instances); mref addresses
  // are not used (sm_103 NVBit reports SYNCS mrefs at +0x7FFF, and a
  // data word so "named" turns an ordinary STS into a rejected init).
  if (rules_ && !rules_->named_barrier_roots().empty()) {
    for (std::uint64_t di = 0; di < t_.dyn_n; ++di) {
      DynOps dd;
      build_dynops(di, dd);
      if ((dd.active_mask & dd.predicate_mask) == 0) continue;
      auto it = dd.uniform.find("mbarrier_address");
      if (it != dd.uniform.end())
        // named mbarrier words are keyed by local shared offsets;
        // strip the DSMEM rank tag (bug 4-5.1)
        sem_ctx_.mbarrier_addresses.insert(
            local_shared_address(it->second));
      auto lw = dd.lane_values.find("mbarrier_words");
      if (lw != dd.lane_values.end())
        for (auto a : lw->second)
          sem_ctx_.mbarrier_addresses.insert(local_shared_address(a));
    }
  }
  t_.mrex_n /= F_MREX;
  t_.stat_n /= F_STAT;
  if (statics_.size() < t_.stat_n) {
    statics_.resize(t_.stat_n);
    statics_as_info_.resize(t_.stat_n);
    tmpl_key_prefix_.clear();   // keyed by StaticRow address
  }
}

std::uint32_t SegmentBuilder::string_offset(const std::string &s) {
  auto it = string_offsets_.find(s);
  if (it != string_offsets_.end()) return it->second;
  const auto off = static_cast<std::uint32_t>(strings_.size());
  strings_.insert(strings_.end(), s.begin(), s.end());
  strings_.push_back(0);
  string_offsets_[s] = off;
  return off;
}

void SegmentBuilder::sector(std::uint64_t key, std::uint32_t bmask,
                            std::uint32_t lmask) {
  auto &m = tables_[tag4("MSEC")];
  put8(m, key * SECTOR_BYTES);
  put4(m, bmask);
  put4(m, lmask);
  msec_count_++;
}

void SegmentBuilder::emit_plain(std::uint64_t ab, std::uint64_t ac,
                                std::uint32_t width) {
  // lane->sector merge is per access (one MACC): an earlier
  // instruction's MSEC rows are not this MACC's rows, so sector reuse
  // across instructions must still emit a row here
  // width > 0 (local-space loads): precise byte mask like emit_store
  // -- a 4-byte LDL covers 4 bytes of its trace sector, not 32
  const std::uint64_t width_mask =
      width ? ((1ull << width) - 1) : 0xFFFFFFFFull;
  std::unordered_map<std::uint64_t, std::uint32_t> sector_index;
  auto &m = tables_[tag4("MSEC")];
  for (std::uint64_t l = 0; l < ac; ++l) {
    const auto a = rd64(t_.addr + (ab + l) * F_ADDR);
    const auto key = a / SECTOR_BYTES;
    const std::uint64_t bm = width_mask << (a % SECTOR_BYTES);
    auto it = sector_index.find(key);
    if (it == sector_index.end()) {
      sector_index[key] = static_cast<std::uint32_t>(msec_count_);
      sector(key, static_cast<std::uint32_t>(bm & 0xFFFFFFFFull),
             1u << l);
    } else {
      const auto base = static_cast<std::uint64_t>(it->second) * S_MSEC;
      if (width) {
        const std::uint32_t oldb = rd32(&m[base + 8]);
        const std::uint32_t nb =
            oldb | static_cast<std::uint32_t>(bm & 0xFFFFFFFFull);
        std::memcpy(&m[base + 8], &nb, 4);
      }
      const auto off = base + 12;
      const std::uint32_t old = rd32(&m[off]);
      std::uint32_t nv = old | (1u << l);
      std::memcpy(&m[off], &nv, 4);
    }
  }
}

void SegmentBuilder::emit_store(std::uint64_t /*mref_idx*/,
                                std::uint64_t ab, std::uint64_t ac,
                                std::uint32_t lane_mask,
                                std::uint32_t width) {
  const std::uint32_t mask =
      lane_mask ? lane_mask : ((1u << ac) - 1u);
  const std::uint64_t width_mask = (1ull << width) - 1;
  std::map<std::uint64_t, std::pair<std::uint64_t, std::uint32_t>> per;
  for (std::uint64_t l = 0; l < ac; ++l) {
    if (!((mask >> l) & 1)) continue;
    const auto a = rd64(t_.addr + (ab + l) * F_ADDR);
    const auto key = a / SECTOR_BYTES;
    auto &slot = per[key];
    slot.first |= width_mask << (a % SECTOR_BYTES);
    slot.second |= 1u << l;
  }
  for (auto &[key, v] : per)
    sector(key, static_cast<std::uint32_t>(v.first & 0xFFFFFFFFull),
           v.second);
}

void SegmentBuilder::emit_bulk(std::uint64_t ab, std::uint64_t ac,
                               std::uint32_t width) {
  const std::uint64_t a = ac ? rd64(t_.addr + ab * F_ADDR) : 0;
  const std::uint64_t size = std::max<std::uint64_t>(width, 1);
  for (std::uint64_t key = a / SECTOR_BYTES;
       key <= (a + size - 1) / SECTOR_BYTES; ++key)
    sector(key, 0xFFFFFFFFu, 0);
}

void SegmentBuilder::emit_access(const StaticRow &s, std::uint64_t mi) {
  const auto *mr = t_.mref + mi * F_MREF;
  const auto *mx = t_.mrex + mi * F_MREX;
  const int sp = static_cast<int>(rd32(mr + 0));
  const int ak = static_cast<int>(rd32(mr + 4));
  const std::uint32_t wb = rd32(mr + 8);
  const int rr = static_cast<int>(rd32(mr + 12));
  const std::uint32_t lm = rd32(mr + 16);
  const std::uint32_t ch = rd32(mr + 20);
  const std::uint64_t ab = rd32(mx + 8);
  const std::uint64_t ac = rd32(mx + 12);
  int flags = space_code(sp) | access_code(ak);
  if (rr >= 0) flags |= MACC_REMOTE;
  if (s.uniform_root) flags |= MACC_UNIFORM;
  if (ak == 3 && s.root == "LDGSTS" &&
      s.opcode.find("BYPASS") == std::string::npos)
    flags |= MACC_L1_ALLOCATE;
  const auto begin = msec_count_;
  if (s.utma_root) {
    for (std::uint64_t l = 0; l < ac; ++l) {
      const auto a = rd64(t_.addr + (ab + l) * F_ADDR);
      sector(a / SECTOR_BYTES, 0xFFFFFFFFu, 0);
    }
  } else if (ak == 1) {   // store
    emit_store(mi, ab, ac, lm, wb);
  } else if ((ak == 3 || ak == 4) && ac == 0) {   // async_copy/reduce
    emit_bulk(ab, ac, wb);
  } else {
    // local-space plain loads carry precise byte masks (the frame
    // interleave needs the real bytes); other spaces keep the
    // full-sector mask
    emit_plain(ab, ac, sp == 2 ? wb : 0);
  }
  auto &m = tables_[tag4("MACC")];
  put8(m, begin);
  put4(m, static_cast<std::uint32_t>(msec_count_ - begin));
  put4(m, static_cast<std::uint32_t>(flags));
  put8(m, 0);
  put8(m, 0);
  put8(m, NONE32);
  put4(m, 0);
  put4(m, ch);
  macc_count_++;
}

// operand token -> base name: leading - ~ ! and every | stripped,
// then the name up to the first '.' (python parity)
static std::string operand_base_(const std::string &tok) {
  std::string t = tok;
  while (!t.empty() &&
         (t[0] == '-' || t[0] == '~' || t[0] == '!' || t[0] == '|'))
    t.erase(t.begin());
  // abs bars also close after the name (|R9|.reuse)
  t.erase(std::remove(t.begin(), t.end(), '|'), t.end());
  const auto dot = t.find('.');
  return dot == std::string::npos ? t : t.substr(0, dot);
}

// base-name string -> REGS namespace id, or NONE32 for non-registers
// (same rule as register_id_ applied to the parsed name)
static std::uint32_t register_id_str_(const std::string &name) {
  static const struct { const char *prefix; std::uint32_t base; }
      kNamespaces[] = {{"UR", 256}, {"UP", 768}, {"R", 0},
                       {"P", 512}, {"B", 1024}};
  for (const auto &ns : kNamespaces) {
    const std::size_t lp = std::strlen(ns.prefix);
    if (name.compare(0, lp, ns.prefix) != 0) continue;
    if (lp < name.size() && name[lp] >= '0' && name[lp] <= '9') {
      std::uint32_t v = 0;
      std::size_t d = lp;
      for (; d < name.size() && name[d] >= '0' && name[d] <= '9'; ++d)
        v = v * 10 + (name[d] - '0');
      return ns.base + v;
    }
    return NONE32;
  }
  return NONE32;
}

static bool plain_pred_(const std::string &t) {
  // PT / UPT / P<n> / UP<n> without modifiers
  if (t == "PT" || t == "UPT") return true;
  std::size_t i = 0;
  if (t.size() >= 2 && t[0] == 'U' && t[1] == 'P') i = 2;
  else if (t.size() >= 2 && t[0] == 'P') i = 1;
  else return false;
  for (; i < t.size(); ++i)
    if (t[i] < '0' || t[i] > '9') return false;
  return true;
}

// operand tokens of the real source slots (control-word reuse bit
// order): after the register destinations and the plain secondary
// predicate destinations SASS prints right after them
static std::vector<std::string> source_slots_(
    const std::vector<std::string> &operands,
    std::size_t ndst) {
  std::vector<std::string> rest(
      operands.begin() + std::min(operands.size(), ndst),
      operands.end());
  std::size_t i = 0;
  while (i < rest.size() && plain_pred_(rest[i])) ++i;
  rest.erase(rest.begin(), rest.begin() + i);
  return rest;
}

// opcodes whose register operands are always 64-bit pairs; the
// f32x2 classes (FFMA2/FADD2/FMUL2) treat every operand as a pair
// except a plain ".F32"-suffixed broadcast scalar (python parity)
static bool pair_class_(const std::string &root) {
  static const char *k[] = {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX"};
  for (auto *r : k) if (root == r) return true;
  return false;
}
static bool f32x2_class_(const std::string &root) {
  return root == "FFMA2" || root == "FADD2" || root == "FMUL2";
}

// populate every StaticRow field the emit/scan loaders set; called
// for the pc-16 predecessor before its own emit may have run
void SegmentBuilder::ensure_static(std::uint32_t si) {
  if (statics_[si].opcode.empty()) fill_static(si);
}

// per-REGS-source-row flags (protocol v4 register_file.operand_read),
// identical rule to python _src_flags
std::vector<std::uint8_t>
SegmentBuilder::src_flags(const StaticRow &s) {
  const auto srcs = source_slots_(s.operands, s.dst_names.size());
  std::vector<const std::string *> reg_toks;
  for (auto &t : srcs)
    if (register_id_str_(operand_base_(t)) != NONE32)
      reg_toks.push_back(&t);
  std::vector<const std::string *> rows;
  for (auto &r : s.src_names)
    if (register_id_str_(operand_base_(r)) != NONE32)
      rows.push_back(&r);
  bool aligned = reg_toks.size() == rows.size();
  if (aligned)
    for (std::size_t k = 0; k < rows.size(); ++k)
      if (operand_base_(*reg_toks[k]) != operand_base_(*rows[k])) {
        aligned = false;
        break;
      }
  if (!aligned) {
    ++regs_flag_fallbacks_;
    return std::vector<std::uint8_t>(rows.size(), 0);
  }
  if (static_by_pc_.empty()) {
    for (std::uint32_t i = 0; i < statics_.size(); ++i) {
      const auto pc = rd64(t_.stat + i * F_STAT);
      if (pc) static_by_pc_[pc] = i;
    }
  }
  const StaticRow *pred = nullptr;
  auto pit = static_by_pc_.find(s.pc - 16);
  if (pit != static_by_pc_.end()) {
    ensure_static(pit->second);
    pred = &statics_[pit->second];
  }
  const auto pred_srcs =
      pred ? source_slots_(pred->operands, pred->dst_names.size())
           : std::vector<std::string>{};
  const std::string root_ = s.opcode.substr(0, s.opcode.find('.'));
  const bool pairc = pair_class_(root_), fx2c = f32x2_class_(root_);
  std::vector<std::uint8_t> flags;
  for (std::size_t j = 0; j < srcs.size(); ++j) {
    const auto &tok = srcs[j];
    if (register_id_str_(operand_base_(tok)) == NONE32) continue;
    std::uint8_t f = 0;
    if (pred && j < pred_srcs.size() &&
        ((pred->reuse_mask >> j) & 1) &&
        operand_base_(pred_srcs[j]) == operand_base_(tok))
      f |= 1;
    if (tok.find(".64") != std::string::npos ||
        tok.find(".F32x2") != std::string::npos || pairc ||
        (fx2c && tok.find(".F32") == std::string::npos))
      f |= 2;
    if ((s.reuse_mask >> j) & 1)
      f |= 4;
    flags.push_back(f);
  }
  return flags;
}

std::uint32_t SegmentBuilder::register_id_(std::uint32_t str_off) {
  auto it = reg_ids_.find(str_off);
  if (it != reg_ids_.end()) return it->second;
  const char *name = f_.sstr_at(str_off);
  std::uint32_t n = NONE32;
  if (name) {
    // REGS-row register namespace (must match python
    // segment_builder._register_id): R<n> -> n, UR<n> -> 256+n,
    // P<n> -> 512+n, UP<n> -> 768+n, B<n> -> 1024+n; only the prefix
    // and the digits right after it matter (R12.64, R10.H0_H0,
    // R2.reuse, UR4.64).  RZ/URZ/PT/UPT/SR_* and other names ->
    // NONE32: hard-wired or special registers carry no dependency, so
    // they emit no REGS row and do not enter the template key.
    static const struct { const char *prefix; std::uint32_t base; }
        kNamespaces[] = {{"UR", 256}, {"UP", 768}, {"R", 0},
                         {"P", 512}, {"B", 1024}};
    for (const auto &ns : kNamespaces) {
      const std::size_t lp = std::strlen(ns.prefix);
      if (std::strncmp(name, ns.prefix, lp) != 0) continue;
      const char *d = name + lp;
      if (*d >= '0' && *d <= '9') {
        std::uint32_t v = 0;
        for (; *d >= '0' && *d <= '9'; ++d) v = v * 10 + (*d - '0');
        n = ns.base + v;
      }
      break;
    }
  }
  reg_ids_[str_off] = n;
  return n;
}

std::pair<std::uint32_t, bool>
SegmentBuilder::template_index(const StaticRow &s,
                               const std::string &cls,
                               const std::vector<UseRow> *uses) {
  // key must match python: cls, source / destination register numbers,
  // has_predicate, memory_space, control word (read / write barrier,
  // wait mask, stall, yield), then the use rows.  The TMPL row carries
  // the register refs and scoreboard bits the engine tracks dependencies
  // with, so templates may only be shared by static instructions whose
  // operands and control word are identical.
  // the cls..yield prefix depends only on the static row (cls is a
  // deterministic function of it), so build it once per StaticRow
  const auto sflags = src_flags(s);
  auto &pfx = tmpl_key_prefix_[&s];
  if (pfx.empty()) {
    std::string k = cls + "\x1f";
    std::size_t fk = 0;
    for (auto r : s.src_regs) {
      const auto id = register_id_(r);
      if (id != NONE32)
        k += std::to_string(id) + ":" + std::to_string(sflags[fk++]) + ",";
    }
    k += "\x1f";
    for (auto r : s.dst_regs) {
      const auto id = register_id_(r);
      if (id != NONE32) k += std::to_string(id) + ",";
    }
    k += "\x1f";
    k += (s.has_predicate ? "1" : "0") + std::string("\x1f") +
        s.memory_space;
    k += "\x1f" + std::to_string(s.read_barrier) + "," +
        std::to_string(s.write_barrier) + "," +
        std::to_string(s.wait_mask & 0xFF) + "," +
        std::to_string(s.stall & 0xF) + "," +
        std::to_string(s.yield_hint & 1);
    pfx = std::move(k);
  }
  std::string key = pfx;
  if (uses) {
    for (auto &r : *uses) {
      key += "\x1f" + r.program + "\x1e";
      if (r.missing) continue;
      for (auto &[n, v] : r.facts) {
        key += n + "=";
        key += (v.tag == AxVal::kStr) ? ("s:" + v.s)
                                      : ("i:" + std::to_string(v.i));
        key += ";";
      }
    }
  }
  auto it = templates_.find(key);
  if (it != templates_.end()) return {it->second, false};
  const auto index = static_cast<std::uint32_t>(templates_.size());
  templates_[key] = index;
  auto &regs = tables_[tag4("REGS")];
  const std::uint64_t src_begin = regs.size() / S_REGS;
  std::uint32_t nsrc = 0;
  std::size_t fk = 0;
  for (auto r : s.src_regs) {
    const auto id = register_id_(r);
    if (id != NONE32) {
      put4(regs, id | (sflags[fk++] << 24));
      ++nsrc;
    }
  }
  const std::uint64_t dst_begin = regs.size() / S_REGS;
  std::uint32_t ndst = 0;
  for (auto r : s.dst_regs) {
    const auto id = register_id_(r);
    if (id != NONE32) {
      put4(regs, id);
      ++ndst;
    }
  }
  auto &tm = tables_[tag4("TMPL")];
  put4(tm, string_offset(cls));
  put4(tm, rules_ ? rules_->instruction_flags(
                        StaticInfo{s.opcode, s.operands, s.memory_space,
                                   s.has_predicate})
                  : 0);
  put4(tm, NONE32);
  put4(tm, NONE32);
  put8(tm, src_begin);
  put4(tm, nsrc);
  put8(tm, dst_begin);
  put4(tm, ndst);
  put1(tm, static_cast<std::uint8_t>(s.read_barrier));
  put1(tm, static_cast<std::uint8_t>(s.write_barrier));
  put1(tm, static_cast<std::uint8_t>(s.wait_mask & 0xFF));
  put1(tm, static_cast<std::uint8_t>((s.stall & 0xF) |
                                     ((s.yield_hint & 1) << 4)));
  put8(tm, 0);   // timing_begin
  put4(tm, 0);   // timing_count
  put4(tm, 0);   // semantic_kind
  put8(tm, 0);   // semantic_u64
  return {index, true};
}

void SegmentBuilder::fill_static(std::uint32_t si) {
  auto &s = statics_[si];
  const auto *st = t_.stat + si * F_STAT;
  const auto *sx = t_.stax + si * F_STAX;
  s.pc = rd64(st);
  s.code_address = rd64(st + 8);
  { const char *p = f_.sstr_at(rd32(st + 16)); s.opcode = p ? p : ""; }
  const auto ob = rd32(st + 28), oc = rd32(st + 32);
  const auto sb = rd32(st + 36), sc = rd32(st + 40);
  const auto db = rd32(st + 44), dc = rd32(st + 48);
  const auto pr = rd32(st + 60);
  { const char *p = f_.sstr_at(rd32(sx)); s.memory_space = p ? p : ""; }
  s.has_predicate = pr != 0;
  const auto pk = rd32(sx + 16);
  s.read_barrier = (pk >> 5) & 0xF;
  s.write_barrier = (pk >> 9) & 0xF;
  s.wait_mask = rd32(sx + 8);
  s.reuse_mask = rd32(sx + 12);
  s.stall = pk & 0xF;
  s.yield_hint = (pk >> 4) & 1;
  for (std::uint32_t j = 0; j < oc; ++j)
    { const char *p = f_.sstr_at(rd32(t_.optk + (ob + j) * F_OPTK)); s.operands.push_back(p ? p : ""); }
  for (std::uint32_t j = 0; j < sc; ++j) {
    const auto off = rd32(t_.regl + (sb + j) * F_REGL);
    s.src_regs.push_back(off);
    const char *p = f_.sstr_at(off);
    s.src_names.push_back(p ? p : "");
  }
  for (std::uint32_t j = 0; j < dc; ++j) {
    const auto off = rd32(t_.regl + (db + j) * F_REGL);
    s.dst_regs.push_back(off);
    const char *p = f_.sstr_at(off);
    s.dst_names.push_back(p ? p : "");
  }
  const auto rb = rd32(st + 52), rc = rd32(st + 56);
  {
    // RROL rows live in facts "RROL" table (IB)
    std::size_t rrol_n = 0;
    const std::uint8_t *rrol = f_.table("RROL", &rrol_n);
    if (rrol) {
      for (std::uint32_t j = 0; j < rc; ++j) {
        const auto *rr = rrol + (rb + j) * 5;
        const char *p = f_.sstr_at(rd32(rr));
        s.roles[p ? p : ""] = rr[4] ? "address" : "data";
      }
    }
  }
  statics_as_info_[si] = {s.opcode, s.operands, s.memory_space,
                          s.has_predicate, s.pc, s.dst_names,
                          s.src_names};
  const StaticInfo &info = statics_as_info_[si];
  s.root = root_of(s.opcode);
  s.uniform_root = s.root == "ULDC" || s.root == "LDCU";
  s.utma_root = s.root == "UTMALDG" || s.root == "UTMASTG" ||
      s.root == "UTMAREDG" || s.root == "UTMAPF";
  s.cls = rules_ ? rules_->operation_class(info) : s.opcode;
  s.classes = rules_ ? rules_->operation_classes(info)
                     : std::vector<std::string>{s.cls};
  try {
    s.sem_op = semantic_operation_of(arch_kind_, info);
  } catch (ConversionError &e) {
    s.sem_error = e.what();
  }
  s.src_set.assign(s.src_names.begin(), s.src_names.end());
  std::sort(s.src_set.begin(), s.src_set.end());
  s.src_set.erase(std::unique(s.src_set.begin(), s.src_set.end()),
                  s.src_set.end());
  s.dst_set.assign(s.dst_names.begin(), s.dst_names.end());
  std::sort(s.dst_set.begin(), s.dst_set.end());
  s.dst_set.erase(std::unique(s.dst_set.begin(), s.dst_set.end()),
                  s.dst_set.end());
}

void SegmentBuilder::emit_instance(std::uint64_t di, int cta, int wslot) {
  const auto *d = t_.dyn + di * F_DYN;
  const auto si = rd32(d);
  const auto mb = rd32(d + 12), mc = rd32(d + 16);
  if (si >= statics_.size()) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "bad si=%u di=%llu stat_n=%zu", si,
                  (unsigned long long)di, statics_.size());
    throw std::runtime_error(buf);
  }
  ensure_static(si);
  const auto &s = statics_[si];
  const StaticInfo &info = statics_as_info_[si];
  const std::string &cls = s.cls;
  DynOps dyn;
  build_dynops(di, dyn);
  if (dyn.has_gsyn) {
    const int gcls = global_sync_class(info.opcode);
    // segment GSYN <IBBBBQQQQ>: instance/lane/kind/class/flags/
    // address/after/before/sequence (40 B, python parity)
    auto &gt = tables_[tag4("GSYN")];
    put4(gt, static_cast<std::uint32_t>(inst_count_));
    put1(gt, static_cast<std::uint8_t>(dyn.gsyn_lane));
    put1(gt, static_cast<std::uint8_t>(dyn.gsyn_kind));
    put1(gt, static_cast<std::uint8_t>(gcls));
    put1(gt, static_cast<std::uint8_t>(
        dyn.gsyn_flags ? dyn.gsyn_flags
                       : (1u | (dyn.gsyn_before ? 2u : 0u))));
    put8(gt, dyn.gsyn_addr);
    put8(gt, dyn.gsyn_after);
    put8(gt, dyn.gsyn_before);
    put8(gt, dyn.gsyn_seq);
  }
  const bool executing = (dyn.active_mask & dyn.predicate_mask) != 0;
  axctx_.flow = &flows_[wslot];
  auto *flow = axctx_.flow;
  flow->reader = cur_reader_;
  const std::vector<std::string> &classes = s.classes;
  auto use_rows = program_facts(*rt_, info, &dyn, axctx_, cls, classes);
  std::vector<UseRow> ok_rows;
  for (auto &r : use_rows) {
    if (r.missing) {
      missing_facts_["missing_fact:" + r.missing_key + " (" +
                     s.opcode + "; " + r.program + ")"]++;
    } else {
      ok_rows.push_back(r);
    }
  }
  if (executing && !class_domain_.empty()) {
    bool in_domain = false;
    for (auto &c2 : classes)
      if (class_domain_.count(c2)) in_domain = true;
    if (!in_domain)
      unsupported_["unsupported:engine:mechanism_tree:operation_class=" +
                   cls + " (" + s.opcode + ")"]++;
    bool has_tmpl = false;
    std::string blocker = "none";
    for (auto &r : ok_rows) {
      const auto &p = rt_->progs[r.prog];
      if ((schm_target_.count(p.schema_id) ? schm_target_[p.schema_id]
                                          : 15) == 0)
        has_tmpl = true;
    }
    for (auto &r : use_rows)
      if (r.missing)
        blocker = r.missing_key.substr(r.missing_key.find(':') + 1);
    if (!has_tmpl)
      missing_facts_["missing_fact:program:instruction.service (" +
                     cls + "; " + blocker + ")"]++;
  }
  const auto [tmpl, created] =
      template_index(s, cls, &use_rows);
  if (created) {
    auto [tb, tc] = emit_uses_rows(ok_rows, tmpl, false, s.opcode);
    _patch_tmpl_timing(tmpl, tb, tc);
  } else {
    emit_uses_rows(ok_rows, tmpl, true, s.opcode);
  }
  const std::uint32_t semantics_index =
      emit_semantics(s, info, dyn, cta, wslot);
  std::uint32_t memory_index = NONE32;
  if (mc) {
    memory_index = static_cast<std::uint32_t>(macc_count_);
    for (std::uint32_t k = 0; k < mc; ++k) emit_access(s, mb + k);
  }
  const auto code_line = s.pc / FETCH_LINE_BYTES;
  auto &inst = tables_[tag4("INST")];
  put4(inst, tmpl);
  put4(inst, semantics_index);
  put4(inst, memory_index);
  put4(inst, static_cast<std::uint32_t>(code_line));
  if (s.code_address) {
    auto &code = tables_[tag4("CODE")];
    put4(code, static_cast<std::uint32_t>(inst_count_));
    put4(code, 0);
    put8(code, s.code_address);
  }
  inst_count_++;
  if (executing) {
    const auto op2 = upper(s.opcode);
    if (op2.rfind("LDGSTS", 0) == 0) flow->cpasync_count++;
    if (op2.rfind("ARRIVES", 0) == 0) flow->cpasync_count = 0;
    if (op2.rfind("LDGDEPBAR", 0) == 0) flow->cpasync_count = 0;
  }
}

void SegmentBuilder::_patch_tmpl_timing(std::uint32_t tmpl,
                                        std::uint32_t tb,
                                        std::uint32_t tc) {
  auto &tm = tables_[tag4("TMPL")];
  const auto base = (std::size_t)tmpl * 68;
  std::uint64_t b = tb;
  std::memcpy(tm.data() + base + 44, &b, 8);
  std::uint32_t c = tc;
  std::memcpy(tm.data() + base + 52, &c, 4);
}

void SegmentBuilder::emit_warp(std::uint64_t wi) {
  const auto *w = t_.warp + wi * F_WARP;
  const auto cl = rd32(w);
  const auto wic = rd32(w + 24);
  const auto ws = rd32(w + 20);
  const auto db = rd32(w + 28), dn = rd32(w + 32);
  auto &out = tables_[tag4("WARP")];
  // pack first, append after instances
  std::vector<std::uint8_t> row;
  put4(row, cl);
  put4(row, wic);
  put4(row, ws);
  put4(row, ws % 4);   // smsp = warp_slot % 4
  put8(row, inst_count_);
  put8(row, dn);
  auto &flow = flows_[ws];
  // first-reader prescan (python _scan_first_readers)
  std::map<std::uint64_t, std::pair<std::uint64_t, std::string>>
      readers;
  {
    std::unordered_map<std::string,
                       std::tuple<std::uint64_t, std::uint64_t,
                                  std::string>>
        pending;
    for (std::int64_t i = (std::int64_t)dn - 1; i >= 0; --i) {
      const auto *d = t_.dyn + (db + i) * F_DYN;
      if ((rd32(d + 4) & rd32(d + 8)) == 0) continue;
      const auto si = rd32(d);
      if (si >= statics_.size()) continue;
      ensure_static(si);
      const auto &st = statics_[si];
      // scan
      const std::pair<std::uint64_t, std::string> *best = nullptr;
      std::pair<std::uint64_t, std::string> bv;
      for (auto &reg : st.dst_names) {
        auto it = pending.find(reg);
        if (it != pending.end() &&
            (!best || std::get<0>(it->second) < std::get<0>(*best))) {
          bv = {std::get<1>(it->second), std::get<2>(it->second)};
          best = &bv;
        }
      }
      if (best) readers[(std::uint64_t)i] = *best;
      // sorted unique name sets (per-key assignments commute, so the
      // iteration order within each loop does not matter)
      const auto &srcs = st.src_set;
      const auto &dsts = st.dst_set;
      const auto in = [](const std::vector<std::string> &v,
                         const std::string &x) {
        return std::binary_search(v.begin(), v.end(), x);
      };
      for (auto &reg : dsts) {
        if (in(srcs, reg)) {
          auto ri = st.roles.find(reg);
          pending[reg] = {i, si, ri != st.roles.end() ? ri->second
                                                    : ""};
        } else {
          pending.erase(reg);
        }
      }
      for (auto &reg : srcs)
        if (!in(dsts, reg)) {
          auto ri = st.roles.find(reg);
          pending[reg] = {i, si, ri != st.roles.end() ? ri->second
                                                    : ""};
        }
    }
  }
  for (std::uint64_t j = 0; j < dn; ++j) {
    auto it = readers.find(j);
    if (it != readers.end()) {
      cur_reader_ = {true, it->second.first};
      cur_reader_role_ = it->second.second;
      cur_reader_role_none_ = it->second.second.empty();
    } else {
      cur_reader_ = {false, 0};
      cur_reader_role_.clear();
      cur_reader_role_none_ = false;
    }
    flow.reader = {cur_reader_.first,
                   cur_reader_.first ? cur_reader_.second : 0};
    flow.reader_role = cur_reader_role_;
    flow.reader_role_none = cur_reader_role_none_;
    if (j + 1 < dn) {
      // branch_taken axis: next dynamic instance's static pc
      // (read the STAT row directly: statics_[nsi] is filled lazily by
      // emit_instance, so a static first seen as the *next* instance
      // still has pc 0 here and every such BRA read as taken)
      const auto nsi = rd32(t_.dyn + (db + j + 1) * F_DYN);
      if (nsi >= statics_.size()) throw std::runtime_error("bad next si");
      axctx_.next_static_pc = rd64(t_.stat + nsi * F_STAT);
      axctx_.has_next_static_pc = true;
    } else {
      axctx_.has_next_static_pc = false;
    }
    emit_instance(db + j, cl, ws);
  }
  flow.reader = {false, 0};
  out.insert(out.end(), row.begin(), row.end());
}

std::vector<std::uint8_t> SegmentBuilder::build() {
  // group warps by cta in table order
  std::unordered_map<std::uint32_t, std::vector<std::uint64_t>> by_cta;
  for (std::uint64_t i = 0; i < t_.warp_n; ++i)
    by_cta[rd32(t_.warp + i * F_WARP)].push_back(i);
  std::uint64_t first_warp = 0;
  for (std::uint64_t ci = 0; ci < t_.ctas_n; ++ci) {
    const auto *c = t_.ctas + ci * F_CTAS;
    const auto cl = rd32(c);
    const auto &w = by_cta[cl];
    auto &out = tables_[tag4("CTAS")];
    put4(out, cl);
    put4(out, rd32(c + 4));    // x
    put4(out, rd32(c + 8));    // y
    put4(out, rd32(c + 12));   // z
    put4(out, rd32(c + 16));   // cluster_rank
    put4(out, meta_.cluster_shape[0]);
    put4(out, meta_.cluster_shape[1]);
    put4(out, meta_.cluster_shape[2]);
    put4(out, rd32(c + 20));   // sm_id
    put4(out, rd32(c + 24));   // peer_sm_id
    put4(out, rd32(c + 28));   // multicast
    put1(out, static_cast<std::uint8_t>(rd32(c + 32) & 0xFF));   // group
    put1(out, static_cast<std::uint8_t>(rd32(c + 36) & 0xFF));   // leader
    put2(out, static_cast<std::uint16_t>(rd32(c + 40) & 0xFFFF)); // tmem
    put4(out, static_cast<std::uint32_t>(first_warp));
    put4(out, static_cast<std::uint32_t>(w.size()));
    for (auto wi : w) emit_warp(wi);
    first_warp += w.size();
  }
  // segment MBPI: (cta_linear, object, address, arrival_count, pad)
  // facts MBPI <4i4I>: (cta_linear, address, arrival_count, ...)
  if (t_.mbpi) {
    const auto *mp = t_.mbpi;
    auto &mb = tables_[tag4("MBPI")];
    for (std::uint64_t i = 0; i < t_.mbpi_n / 32; ++i) {
      // facts MBPI <4i4I>: cta_linear@0, address@4, arrival_count@8
      const auto *r = mp + i * 32;
      const auto cta = rd32(r);
      const auto addr = rd32(r + 4);
      const auto arr = rd32(r + 8);
      const auto obj = sem_ctx_.mbarrier((int)cta, addr, "mbpi");
      put4(mb, cta);
      put4(mb, obj);
      put8(mb, addr);
      put4(mb, arr);
      put4(mb, 0);
    }
  }
  if (!sem_ctx_.mbarrier_objects.empty()) {
    auto &mo = tables_[tag4("MBOB")];
    std::vector<std::pair<std::pair<int, std::uint64_t>, std::uint32_t>>
        objs(sem_ctx_.mbarrier_objects.begin(),
             sem_ctx_.mbarrier_objects.end());
    std::sort(objs.begin(), objs.end(),
              [](auto &a, auto &b) { return a.second < b.second; });
    for (auto &[k, idx] : objs) {
      put4(mo, (std::uint32_t)k.first);
      put8(mo, k.second);
    }
  }
  auto &meta = tables_[tag4("META")];
  put32s(meta, meta_.tree_digest);
  put32s(meta, meta_.program_digest);
  put32s(meta, meta_.converter_digest);
  put4(meta, f_.cluster_id());
  put4(meta, static_cast<std::uint32_t>(t_.ctas_n));
  put4(meta, static_cast<std::uint32_t>(t_.warp_n));
  put8(meta, inst_count_);
  put4(meta, meta_.flags);
  // STRS
  auto &strs = strings_;
  // assemble GSIMSEG (REQUIRED_TABLES order)
  static const char *kOrder[] = {
      "META", "CTAS", "WARP", "TMPL", "INST", "REGS", "SEMA",
      "MACC", "MSEC", "TIME", "PROG", "NODE", "GRD ", "LEAF",
      "PARM", "PTS ", "FACT", "USES", "SPLT", "MBPI", "MBOL",
      "CODE", "CLKM", "STRS", "AXIS", "MBOB", "GSYN"};
  static const std::size_t kRec[] = {
      S_META, S_CTA, S_WARP, S_TMPL, S_INST, S_REGS, S_SEMA,
      S_MACC, S_MSEC, S_TIME, 32, 24, 24, 8, 48, 16,
      S_FACT, S_USES, S_SPLT, S_MBPI, S_MBOL, S_CODE, S_CLKM, 0, 4,
      12, 40};
  const std::size_t ntab0 = sizeof(kOrder) / sizeof(kOrder[0]);
  // python parity: optional tables (MBOB, GSYN) are only emitted when
  // non-empty
  std::vector<std::size_t> order;
  order.reserve(ntab0);
  for (std::size_t i = 0; i < ntab0; ++i) {
    if (std::string(kOrder[i]) == "MBOB" &&
        sem_ctx_.mbarrier_objects.empty())
      continue;
    if (std::string(kOrder[i]) == "GSYN" &&
        tables_[tag4("GSYN")].empty())
      continue;
    // segment v2: shared program tables live once in program.tables
    if (std::string(kOrder[i]) == "PROG" ||
        std::string(kOrder[i]) == "NODE" ||
        std::string(kOrder[i]) == "GRD " ||
        std::string(kOrder[i]) == "LEAF" ||
        std::string(kOrder[i]) == "PARM" ||
        std::string(kOrder[i]) == "PTS " ||
        std::string(kOrder[i]) == "AXIS")
      continue;
    order.push_back(i);
  }
  const std::size_t ntab = order.size();
  std::vector<std::uint8_t> out;
  const std::size_t hdr = 72, dent = 32;
  std::uint64_t cursor = hdr + ntab * dent;
  cursor = (cursor + 7) & ~7ull;
  out.resize(hdr);
  std::memcpy(out.data(), "GSIMSEG\0", 8);
  out.resize(cursor);
  std::vector<std::uint8_t> dir(ntab * dent);
  std::size_t body = cursor;
  for (std::size_t oi2 = 0; oi2 < ntab; ++oi2) {
    const std::size_t i = order[oi2];
    const auto &pl = (std::string(kOrder[i]) == "STRS")
                         ? strs
                         : tables_[tag4(kOrder[i])];
    const auto rec = kRec[i];
    const auto cnt = rec ? pl.size() / rec : pl.size();
    std::memcpy(dir.data() + oi2 * dent, kOrder[i], 4);
    std::uint32_t rb = static_cast<std::uint32_t>(rec);
    std::memcpy(dir.data() + oi2 * dent + 4, &rb, 4);
    std::uint64_t c = cnt, off = out.size();
    std::memcpy(dir.data() + oi2 * dent + 8, &c, 8);
    std::memcpy(dir.data() + oi2 * dent + 16, &off, 8);
    out.insert(out.end(), pl.begin(), pl.end());
    if (oi2 + 1 < ntab)          // python pads only between tables, not after
      out.resize((out.size() + 7) & ~7ull);
  }
  // header fields: magic,2,hdr_bytes,0,dir_off,count,flags,32s
  std::uint32_t v32;
  v32 = 2; std::memcpy(out.data() + 8, &v32, 4);
  v32 = static_cast<std::uint32_t>(hdr);
  std::memcpy(out.data() + 12, &v32, 4);
  std::uint64_t v64 = out.size();
  std::memcpy(out.data() + 16, &v64, 8);
  v64 = hdr;
  std::memcpy(out.data() + 24, &v64, 8);
  v32 = static_cast<std::uint32_t>(ntab);
  std::memcpy(out.data() + 32, &v32, 4);
  v32 = 0;
  std::memcpy(out.data() + 36, &v32, 4);
  std::memcpy(out.data() + 72, dir.data(), dir.size());
  (void)body;
  if (regs_flag_fallbacks_)
    fprintf(stderr,
            "note: regs flag fallback (flags 0) on %llu static "
            "instruction(s)\n",
            (unsigned long long)regs_flag_fallbacks_);
  return out;
}

}  // namespace cyclescope_convert

namespace cyclescope_convert {
// arch singletons defined in cpp/arch/<arch>/convert_rules.cpp
const ConvertRules *hopper_convert_rules();
const ConvertRules *blackwell_convert_rules();

const ConvertRules *convert_rules_for(const std::string &arch) {
  if (arch == "hopper") return hopper_convert_rules();
  if (arch == "blackwell") return blackwell_convert_rules();
  return nullptr;
}
}  // namespace cyclescope_convert

namespace cyclescope_convert {
namespace {
std::string where_hex(const std::string &opc, std::uint64_t pc) {
  char buf[48];
  std::snprintf(buf, sizeof buf, " @0x%llx", (unsigned long long)pc);
  return opc + buf;
}
}  // namespace

void SegmentBuilder::build_dynops(std::uint64_t di, DynOps &dyn) {
  const auto *d = t_.dyn + di * F_DYN;
  dyn.active_mask = rd32(d + 4);
  dyn.predicate_mask = rd32(d + 8);
  const auto oi = rd32(d + 20), mb = rd32(d + 12), mc = rd32(d + 16),
             gi = rd32(d + 24);
  if (gi != NONE32 && t_.gsyn) {
    const auto *g = t_.gsyn + gi * F_GSYN;
    dyn.gsyn_kind = rd32(g); dyn.gsyn_lane = rd32(g + 4);
    dyn.gsyn_addr = rd64(g + 8);
    dyn.gsyn_before = rd64(g + 16); dyn.gsyn_after = rd64(g + 24);
    dyn.gsyn_seq = rd64(g + 32);
    if (t_.gsynx) dyn.gsyn_flags = rd32(t_.gsynx + gi * F_GSYX + 4);
    dyn.has_gsyn = true;
  }
  for (std::uint32_t k = 0; k < mc; ++k) {
    const auto *mr = t_.mref + (mb + k) * F_MREF;
    const auto *mx = t_.mrex + (mb + k) * F_MREX;
    DynOps::Ref ref;
    ref.space = rd32(mr);
    ref.access_kind = rd32(mr + 4);
    ref.width = rd32(mr + 8);
    ref.lane_mask = rd32(mr + 16);
    ref.remote_rank = *(const std::int32_t *)(mr + 12);
    ref.destination_count = *(const std::int32_t *)(mr + 24);
    ref.multicast_mask = 0;
    ref.transfer_bytes = *(const std::int64_t *)(mx);
    ref.cut_rows = *(const std::int32_t *)(mx + 16);
    const auto ab = rd32(mx + 8), ac = rd32(mx + 12);
    for (std::uint32_t j = 0; j < ac; ++j)
      ref.addrs.push_back(rd64(t_.addr + (ab + j) * F_ADDR));
    dyn.memory.push_back(std::move(ref));
  }
  if (oi != NONE32 && t_.ops) {
    const auto *o = t_.ops + oi * F_OPS;
    const auto *ox = t_.opsx + oi * F_OPSX;
    const auto kb = rd32(o), kc = rd32(o + 4), lb = rd32(o + 8),
               rb = rd32(o + 16), rc = rd32(o + 20),
               arm = rd32(o + 24), flags = rd32(o + 28);
    const auto tm = rd32(ox);
    if (flags & 1) {
      dyn.after_ready_mask = arm;
      dyn.has_after_ready = true;
    }
    if (tm) {
      dyn.tensor_map = tm;
      dyn.has_tensor_map = true;
    }
    for (std::uint32_t j = 0; j < kc; ++j) {
      const auto *u = t_.ukv + (kb + j) * F_UKV;
      const auto ko = rd32(u);
      const auto v = rd64(u + 5);
      const char *p = f_.str_at(ko);
      if (u[4])
        dyn.uniform[p ? p : ""] = v;
      else {
        const char *vs = f_.str_at((std::uint32_t)v);
        dyn.uniform_str[p ? p : ""] = vs ? vs : "";
      }
    }
    // LANE rows: named per-lane values [lb, lb+lc) followed by the rank
    // entries [rb, rb+rc) (facts_format OPS); both are read
    for (std::int64_t j = (std::int64_t)lb; j < (std::int64_t)(rb + rc);
         ++j) {
      const auto *ln = t_.lane + j * F_LANE;
      const auto ko = rd32(ln), vb = rd32(ln + 4), vc = rd32(ln + 8);
      const int rank = (ko & 0x80000000u) ? (int)(ko & 0x7FFFFFFF)
                                          : (int)ko;
      std::vector<std::uint64_t> vals;
      for (std::uint32_t m2 = 0; m2 < vc; ++m2)
        vals.push_back(rd64(t_.lval + (vb + m2) * F_LVAL));
      if (ko & 0x80000000u) {
        for (auto v : vals)
          dyn.ranks[rank].push_back((std::uint32_t)v);
      } else {
        const char *p = f_.str_at(ko);
        dyn.lane_values[p ? p : ""] = vals;
      }
    }
  }
}

std::uint32_t SegmentBuilder::emit_semantics(const StaticRow &s,
                                             const StaticInfo &info,
                                             const DynOps &dyn, int cta,
                                             int wslot) {
  if (!s.sem_error.empty()) {
    missing_facts_[s.sem_error]++;
    return NONE32;
  }
  SemOp op = s.sem_op;
  SemRow row;
  try {
    if (rules_)
      op = rules_->promote_operation(info, dyn, sem_ctx_, op);
    if (op != SemOp::none)
      row = decode_semantics(info, dyn, sem_ctx_, op, cta, wslot,
                             arch_kind_);
  } catch (ConversionError &e) {
    missing_facts_[e.what()]++;
    return NONE32;
  }
  if (dyn.has_gsyn) {
    SemRow gsync = global_sync_row(info, dyn);
    if (gsync.kind != SemKind::none) {
      if (row.kind == SemKind::none)
        row = std::move(gsync);
      else
        unsupported_["unsupported:global_sync_semantics_conflict (" +
                     where_str(info) + ")"]++;
    }
  }
  if (row.kind == SemKind::none) return NONE32;
  std::uint32_t mbol_begin = 0;
  const auto mbol_n = static_cast<std::uint32_t>(row.object_list.size());
  if (mbol_n) {
    mbol_begin = static_cast<std::uint32_t>(mbol_count_);
    auto &mb = tables_[tag4("MBOL")];
    for (auto o : row.object_list) put4(mb, o);
    mbol_count_ += mbol_n;
  }
  const auto index = static_cast<std::uint32_t>(sema_count_);
  for (auto &[rank, credit, pad] : row.split) {
    auto &sp = tables_[tag4("SPLT")];
    put4(sp, index);
    put4(sp, rank);
    put4(sp, credit);
    put4(sp, pad);
  }
  auto has_list = mbol_n != 0;
  auto f2 = row.fields;
  if (has_list) f2["_has_list"] = 1;
  auto slots = pack_fields(row.kind, f2, mbol_begin, mbol_n);
  auto &se = tables_[tag4("SEMA")];
  put2(se, static_cast<std::uint16_t>(row.kind));
  put2(se, static_cast<std::uint16_t>(row.flags));
  put8(se, row.earliest_issue);
  for (auto v : slots) put8(se, v);
  sema_count_++;
  return index;
}
}  // namespace cyclescope_convert


namespace cyclescope_convert {

std::pair<std::uint32_t, std::uint32_t>
SegmentBuilder::emit_uses_rows(const std::vector<UseRow> &rows,
                               std::uint32_t template_idx,
                               bool instance_only,
                               const std::string &opcode) {
  const auto time_begin = time_count_;
  for (const auto &row : rows) {
    if (row.missing) continue;
    const auto &prog = rt_->progs[row.prog];
    const auto tit = schm_target_.count(prog.schema_id)
                         ? schm_target_.at(prog.schema_id)
                         : 15;
    const int target = tit;  // 0 tmpl, 1 timing, 2 instance, else
    if (instance_only && target != 2) continue;
    const std::uint64_t fact_begin = fact_count_;
    bool bad = false;
    std::string bad_axis;
    for (auto &[name, val] : row.facts) {
      const auto &ax = rt_->axes[rt_->axis_index.at(name)];
      std::uint64_t v = 0;
      if (ax.kind == 1) {
        const bool found = ax.index_of(val.s, v);
        if (!found) {
          bad = true;
          bad_axis = name;
          break;
        }
      } else {
        v = (std::uint64_t)val.i;
      }
      auto &ft = tables_[tag4("FACT")];
      put4(ft, string_offset(name));
      put8(ft, v);
      fact_count_++;
    }
    if (bad) {
      missing_facts_["missing_fact:axis:" + bad_axis + " (" +
                      opcode + " domain)"]++;
      auto &ft = tables_[tag4("FACT")];
      ft.resize(fact_begin * 12);
      fact_count_ = fact_begin;
      continue;
    }
    const auto fact_count = fact_count_ - fact_begin;
    std::uint32_t target_index;
    if (target == 1) {  // TimingRow
      auto &tm = tables_[tag4("TIME")];
      put4(tm, prog.timing_kind);
      put4(tm, 0);
      put4(tm, 0);
      put4(tm, 0);
      put8(tm, 0);
      { double z = 0.0; std::uint64_t bits; std::memcpy(&bits, &z, 8); put8(tm, bits); }
      put1(tm, 1);
      for (int i = 0; i < 7; ++i) put1(tm, 0);
      target_index = static_cast<std::uint32_t>(time_count_);
      time_count_++;
    } else if (target == 0) {
      target_index = template_idx;
    } else if (target == 2) {
      target_index = static_cast<std::uint32_t>(inst_count_);
    } else {
      target_index = 0;
    }
    auto &us = tables_[tag4("USES")];
    put4(us, static_cast<std::uint32_t>(row.node));
    put1(us, static_cast<std::uint8_t>(target));
    put1(us, 0); put1(us, 0); put1(us, 0);
    put8(us, target_index);
    put2(us, static_cast<std::uint16_t>(prog.schema_id));
    put8(us, prog.field_mask);
    put8(us, fact_begin);
    put4(us, static_cast<std::uint32_t>(fact_count));
  }
  return {static_cast<std::uint32_t>(time_begin),
          static_cast<std::uint32_t>(time_count_ - time_begin)};
}

}  // namespace cyclescope_convert
