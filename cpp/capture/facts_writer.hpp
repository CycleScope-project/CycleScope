// GSIMCAP02 cluster facts writer + GSIMCAPS1 statics bundle.
// Mirrors python/cyclescope/convert/facts_format.py:
//   FactsWriter / _UnitStream / write_statics_file byte-for-byte.
#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cyclescope_convert::capture {

constexpr std::uint32_t kN32 = 0xFFFFFFFFu;

// String interning identical to FactsWriter._s: blob starts with one
// NUL byte, each new string appended once, None maps to offset 0.
struct StrIntern {
  std::vector<std::uint8_t> buf{0};
  std::unordered_map<std::string, std::uint32_t> map;
  std::uint32_t intern(const std::string &s) {
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    std::uint32_t off = static_cast<std::uint32_t>(buf.size());
    buf.insert(buf.end(), s.begin(), s.end());
    buf.push_back(0);
    map.emplace(s, off);
    return off;
  }
};

// write_file: header <8sIIQQII32s = 72B, dir entries <IIQQQ = 32B,
// payloads 8-byte aligned.  magic = "GSIMCP2\0" (cluster) or
// "GSIMCPS\0" (statics).
std::vector<std::uint8_t> write_file(
    const char magic[8],
    const std::vector<std::tuple<std::string, std::uint32_t,
                                 std::vector<std::uint8_t>>> &tables);

// ---- statics bundle (write_statics_file) --------------------------------
struct StatRow {      // fields consumed by the STAT/STAX row packers
  std::uint64_t pc = 0, code_address = 0;
  std::string opcode, operation_class;
  std::vector<std::string> operands;
  std::vector<std::string> source_registers, destination_registers;
  std::vector<std::pair<std::string, bool>> register_roles;
  std::string predicate;
  bool has_predicate = false;
  std::string memory_space;
  bool has_memory_space = false;
  std::int32_t branch_target = -1;
  std::uint32_t wait_mask = 0, reuse_mask = 0, packed_ctrl = 0;
  bool predicate_inverted = false;
};

// tables: META STAT STAX OPTK REGL RROL STRS  (magic GSIMCPS)
std::vector<std::uint8_t> write_statics_file(
    const std::vector<StatRow> &statics);

// ---- cluster unit (StreamingFactsWriter, split_statics=True) ------------
struct DynRow {
  std::uint32_t static_index = 0, active_mask = 0, predicate_mask = 0;
  // per-instruction aux payloads, prepacked little-endian rows:
  std::vector<std::uint8_t> mref, mrefx, addr, ops, opsx, ukv,
      lane, lval, cord, desc, gsyn, gsyx;
  std::uint32_t mref_count = 0;
  bool has_ops = false, has_gsync = false;
};
struct WarpRow {
  std::uint32_t cta_linear = 0, x = 0, y = 0, z = 0;
  std::uint32_t sm_id = 0, warp_slot = 0;
  std::int32_t warp_in_cta = 0;
};
struct CtaRow {
  std::int32_t cta_linear = 0, x = 0, y = 0, z = 0;
  std::uint32_t cluster_rank = 0, sm_id = 0, peer_sm_id = 0,
      multicast_mask = 0, cta_group = 0, pair_leader = 0,
      tmem_columns = 0;
};

// tables (split_statics order): META CTAS WARP DYNI MREF MREX ADDR OPSZ
// OPSX(20) UKVT LANE LVAL CORD DESC GSYN GSYX TMAP TMDI MBPI STRS
// (magic GSIMCP2).  warps[i] owns dyns[dyn_begin, dyn_begin+count).
std::vector<std::uint8_t> write_cluster_file(
    StrIntern &strtab,
    std::uint32_t cluster_id,
    const std::vector<CtaRow> &ctas,
    const std::vector<WarpRow> &warps,
    const std::vector<std::vector<DynRow>> &warp_dyns,
    const std::vector<std::vector<std::uint8_t>> &tmap_rows,
    const std::vector<std::vector<std::uint8_t>> &tmdi_rows,
    const std::vector<std::vector<std::uint8_t>> &mbpi_rows,
    std::uint32_t n_stat,
    const std::uint8_t digest32[32]);

}  // namespace cyclescope_convert::capture
