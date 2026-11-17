#include "capture/facts_writer.hpp"

#include <cstring>

namespace cyclescope_convert::capture {
namespace {

void put8(std::vector<std::uint8_t> &v, std::uint64_t x) {
  for (int i = 0; i < 8; ++i) v.push_back(std::uint8_t(x >> (8 * i)));
}
void put4(std::vector<std::uint8_t> &v, std::uint32_t x) {
  for (int i = 0; i < 4; ++i) v.push_back(std::uint8_t(x >> (8 * i)));
}
void put1(std::vector<std::uint8_t> &v, std::uint8_t x) { v.push_back(x); }
void puti4(std::vector<std::uint8_t> &v, std::int32_t x) {
  put4(v, static_cast<std::uint32_t>(x));
}
void puti8(std::vector<std::uint8_t> &v, std::int64_t x) {
  put8(v, static_cast<std::uint64_t>(x));
}
void pad8(std::vector<std::uint8_t> &v) {
  while (v.size() % 8) v.push_back(0);
}

std::uint32_t tag4(const std::string &t) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= std::uint32_t(std::uint8_t(t[i])) << (8 * i);
  return v;
}

// ---- row packers (facts_format.py struct layouts) ------------------------
// META <6I4Q
void pack_META(std::vector<std::uint8_t> &v, std::uint32_t cluster_id,
               std::uint32_t n_ctas, std::uint32_t n_warps,
               std::uint32_t n_stat, std::uint32_t n_dyn,
               std::uint32_t n_mref, const std::uint8_t digest[32]) {
  put4(v, cluster_id); put4(v, n_ctas); put4(v, n_warps);
  put4(v, n_stat); put4(v, n_dyn); put4(v, n_mref);
  v.insert(v.end(), digest, digest + 32);
}
// CTAS <4i8I
void pack_CTAS(std::vector<std::uint8_t> &v, const CtaRow &c) {
  puti4(v, c.cta_linear); puti4(v, c.x); puti4(v, c.y); puti4(v, c.z);
  put4(v, c.cluster_rank); put4(v, c.sm_id); put4(v, c.peer_sm_id);
  put4(v, c.multicast_mask); put4(v, c.cta_group); put4(v, c.pair_leader);
  put4(v, c.tmem_columns); put4(v, 0);
}
// WARP <6I2i2I
void pack_WARP(std::vector<std::uint8_t> &v, const WarpRow &w,
               std::uint32_t dyn_begin, std::uint32_t dyn_count) {
  put4(v, w.cta_linear); put4(v, w.x); put4(v, w.y); put4(v, w.z);
  put4(v, w.sm_id); put4(v, w.warp_slot);
  puti4(v, w.warp_in_cta);
  put4(v, dyn_begin); put4(v, dyn_count); put4(v, 0);
}
// STAT <3Q10I
void pack_STAT(std::vector<std::uint8_t> &v, StrIntern &s,
               const StatRow &r, std::uint32_t ob, std::uint32_t on,
               std::uint32_t sb, std::uint32_t db, std::uint32_t rb) {
  put8(v, r.pc); put8(v, r.code_address); put8(v, s.intern(r.opcode));
  put4(v, s.intern(r.operation_class));
  put4(v, ob); put4(v, on);
  put4(v, sb); put4(v, static_cast<std::uint32_t>(r.source_registers.size()));
  put4(v, db);
  put4(v, static_cast<std::uint32_t>(r.destination_registers.size()));
  put4(v, rb);
  put4(v, static_cast<std::uint32_t>(r.register_roles.size()));
  put4(v, r.has_predicate ? s.intern(r.predicate) : 0);
}
// STAX <4i2I
void pack_STAX(std::vector<std::uint8_t> &v, StrIntern &s,
               const StatRow &r) {
  puti4(v, static_cast<std::int32_t>(r.has_memory_space ? s.intern(r.memory_space) : 0));
  puti4(v, r.branch_target);
  puti4(v, static_cast<std::int32_t>(r.wait_mask));
  puti4(v, static_cast<std::int32_t>(r.reuse_mask));
  put4(v, r.packed_ctrl);
  put4(v, r.predicate_inverted ? 1u : 0u);
}
// DYN <3I4I
void pack_DYN(std::vector<std::uint8_t> &v, const DynRow &d,
              std::uint32_t mref_begin, std::uint32_t ops_i,
              std::uint32_t gsync_i) {
  put4(v, d.static_index); put4(v, d.active_mask);
  put4(v, d.predicate_mask);
  put4(v, mref_begin); put4(v, d.mref_count);
  put4(v, ops_i); put4(v, gsync_i);
}

}  // namespace

std::vector<std::uint8_t> write_file(
    const char magic[8],
    const std::vector<std::tuple<std::string, std::uint32_t,
                                 std::vector<std::uint8_t>>> &tables) {
  const std::size_t n = tables.size();
  std::vector<std::uint8_t> dir;
  std::vector<std::uint8_t> blob;
  std::uint64_t cursor = (72 + n * 32 + 7) / 8 * 8;
  for (const auto &[tag, rec, payload] : tables) {
    const std::uint32_t count =
        rec ? static_cast<std::uint32_t>(payload.size() / rec)
            : static_cast<std::uint32_t>(payload.size());
    put4(dir, tag4(tag)); put4(dir, rec); put8(dir, count);
    put8(dir, cursor); put8(dir, 0);
    while (blob.size() + 72 + n * 32 + ((72 + n * 32) % 8 ?
           8 - (72 + n * 32) % 8 : 0) < cursor)
      blob.push_back(0);
    // pad inside blob to reach `cursor` relative to body start
    const std::size_t body_pad =
        static_cast<std::size_t>(cursor) - (72 + n * 32 +
        ((72 + n * 32) % 8 ? 8 - (72 + n * 32) % 8 : 0));
    while (blob.size() < body_pad) blob.push_back(0);
    blob.insert(blob.end(), payload.begin(), payload.end());
    cursor = (cursor + payload.size() + 7) / 8 * 8;
  }
  // assemble header
  std::vector<std::uint8_t> out;
  out.reserve(72 + dir.size() + blob.size());
  out.insert(out.end(), magic, magic + 8);
  put4(out, 1);                       // version
  put4(out, 72);                      // header_bytes
  put8(out, 0);                       // file_bytes (patched below)
  put8(out, 72);                      // directory_offset
  put4(out, static_cast<std::uint32_t>(n));  // table_count
  put4(out, 0);                       // flags
  out.insert(out.end(), 32, 0);       // digest
  out.insert(out.end(), dir.begin(), dir.end());
  // pad to first payload offset
  const std::uint64_t body_start = (72 + n * 32 + 7) / 8 * 8;
  while (out.size() < body_start) out.push_back(0);
  out.insert(out.end(), blob.begin(), blob.end());
  // patch file_bytes @16
  for (int i = 0; i < 8; ++i)
    out[16 + i] = std::uint8_t(out.size() >> (8 * i));
  return out;
}

std::vector<std::uint8_t> write_statics_file(
    const std::vector<StatRow> &statics) {
  StrIntern s;
  std::vector<std::uint8_t> stat, stax, optk, regl, rrol;
  for (const auto &r : statics) {
    const std::uint32_t ob = static_cast<std::uint32_t>(optk.size() / 4);
    for (const auto &o : r.operands) put4(optk, s.intern(o));
    const std::uint32_t sb = static_cast<std::uint32_t>(regl.size() / 4);
    for (const auto &x : r.source_registers) put4(regl, s.intern(x));
    const std::uint32_t db = static_cast<std::uint32_t>(regl.size() / 4);
    for (const auto &x : r.destination_registers) put4(regl, s.intern(x));
    const std::uint32_t rb = static_cast<std::uint32_t>(rrol.size() / 5);
    for (const auto &[name, is_addr] : r.register_roles) {
      put4(rrol, s.intern(name));
      put1(rrol, is_addr ? 1 : 0);
    }
    pack_STAT(stat, s, r, ob,
              static_cast<std::uint32_t>(optk.size() / 4 - ob), sb, db, rb);
    pack_STAX(stax, s, r);
  }
  std::vector<std::uint8_t> meta;
  const std::uint8_t zeros[32] = {0};
  pack_META(meta, kN32, 0, 0,
            static_cast<std::uint32_t>(statics.size()), 0, 0, zeros);
  return write_file("GSIMCPS", {
      {"META", 56, meta}, {"STAT", 64, stat}, {"STAX", 24, stax},
      {"OPTK", 4, optk}, {"REGL", 4, regl}, {"RROL", 5, rrol},
      {"STRS", 0, s.buf}});
}

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
    const std::uint8_t digest32[32]) {
  std::vector<std::uint8_t> cta_b, warp_b, dyn_b, mref_b, mrex_b, addr_b,
      ops_b, opsx_b, ukv_b, lane_b, lval_b, cord_b, desc_b, gsyn_b, gsyx_b;
  std::uint32_t n_dyn = 0, n_mref = 0;
  for (std::size_t w = 0; w < warps.size(); ++w) {
    const std::uint32_t db = n_dyn;
    for (const auto &d : warp_dyns[w]) {
      const std::uint32_t mb =
          static_cast<std::uint32_t>(mref_b.size() / 28);
      mref_b.insert(mref_b.end(), d.mref.begin(), d.mref.end());
      mrex_b.insert(mrex_b.end(), d.mrefx.begin(), d.mrefx.end());
      addr_b.insert(addr_b.end(), d.addr.begin(), d.addr.end());
      const std::uint32_t oi =
          d.has_ops ? static_cast<std::uint32_t>(ops_b.size() / 32) : kN32;
      ops_b.insert(ops_b.end(), d.ops.begin(), d.ops.end());
      opsx_b.insert(opsx_b.end(), d.opsx.begin(), d.opsx.end());
      ukv_b.insert(ukv_b.end(), d.ukv.begin(), d.ukv.end());
      lane_b.insert(lane_b.end(), d.lane.begin(), d.lane.end());
      lval_b.insert(lval_b.end(), d.lval.begin(), d.lval.end());
      cord_b.insert(cord_b.end(), d.cord.begin(), d.cord.end());
      desc_b.insert(desc_b.end(), d.desc.begin(), d.desc.end());
      const std::uint32_t gi =
          d.has_gsync ? static_cast<std::uint32_t>(gsyn_b.size() / 40)
                      : kN32;
      gsyn_b.insert(gsyn_b.end(), d.gsyn.begin(), d.gsyn.end());
      gsyx_b.insert(gsyx_b.end(), d.gsyx.begin(), d.gsyx.end());
      pack_DYN(dyn_b, d, mb, oi, gi);
      n_mref += d.mref_count;
      ++n_dyn;
    }
    pack_WARP(warp_b, warps[w], db, n_dyn - db);
  }
  for (const auto &c : ctas) pack_CTAS(cta_b, c);
  std::vector<std::uint8_t> tmap_b, tmdi_b, mbpi_b;
  for (const auto &r : tmap_rows)
    tmap_b.insert(tmap_b.end(), r.begin(), r.end());
  for (const auto &r : tmdi_rows)
    tmdi_b.insert(tmdi_b.end(), r.begin(), r.end());
  for (const auto &r : mbpi_rows)
    mbpi_b.insert(mbpi_b.end(), r.begin(), r.end());
  std::vector<std::uint8_t> meta;
  pack_META(meta, cluster_id, static_cast<std::uint32_t>(ctas.size()),
            static_cast<std::uint32_t>(warps.size()), n_stat, n_dyn,
            n_mref, digest32);
  return write_file("GSIMCP2", {
      {"META", 56, meta}, {"CTAS", 48, cta_b}, {"WARP", 40, warp_b},
      {"DYNI", 28, dyn_b}, {"MREF", 28, mref_b}, {"MREX", 20, mrex_b},
      {"ADDR", 8, addr_b}, {"OPSZ", 32, ops_b}, {"OPSX", 20, opsx_b},
      {"UKVT", 13, ukv_b}, {"LANE", 12, lane_b}, {"LVAL", 8, lval_b},
      {"CORD", 8, cord_b}, {"DESC", 8, desc_b}, {"GSYN", 40, gsyn_b},
      {"GSYX", 8, gsyx_b}, {"TMAP", 44, tmap_b}, {"TMDI", 8, tmdi_b},
      {"MBPI", 32, mbpi_b}, {"STRS", 0, strtab.buf}});
}

}  // namespace cyclescope_convert::capture
