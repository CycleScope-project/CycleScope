// tensor_map.py verbatim port — see header for the Python mapping.
#include "capture/tensor_map.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cyclescope_convert::capture {
namespace {

constexpr int kSector = 32;
[[noreturn]] void missing(const std::string &where, const std::string &w) {
  throw std::runtime_error("missing_fact:" + w + " (" + where + ")");
}

// hardware element code -> element bytes
const std::map<int, int> kElementBytes = {
    {0, 1}, {1, 2}, {2, 4}, {3, 4}, {4, 8}, {5, 8}, {6, 2}, {7, 4},
    {8, 4}, {9, 8}, {10, 2}};
// packed sub-byte element codes -> (logical, physical) bits
const std::map<int, std::pair<int, int>> kElementPackedBits = {{11, {4, 4}}};
// encode-path dtype -> (logical, physical) bits
const std::map<int, std::pair<int, int>> kPackedDtypeBits = {
    {13, {4, 4}}, {14, {4, 4}}, {15, {6, 8}}};
// dtype -> bytes (float-capable; 13/14/15 handled via packed)
const std::map<int, double> kTensorDtypeBytes = {
    {0, 1}, {1, 2}, {2, 4}, {3, 4}, {4, 8}, {5, 8}, {6, 2}, {7, 4},
    {8, 8}, {9, 2}, {10, 4}, {11, 4}, {12, 4}};

std::vector<std::uint8_t> unhex(const std::string &s, bool *ok) {
  std::vector<std::uint8_t> out;
  *ok = s.size() % 2 == 0;
  if (!*ok) return out;
  for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
    const auto nib = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    const int h = nib(s[i]), l = nib(s[i + 1]);
    if (h < 0 || l < 0) { *ok = false; return {}; }
    out.push_back(std::uint8_t(h * 16 + l));
  }
  return out;
}
// extract little-endian int from raw bytes starting at bit position
std::uint64_t getbits(const std::vector<std::uint8_t> &r, int bit, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) {
    const int b = bit + i;
    if (b / 8 < int(r.size()) && (r[b / 8] >> (b % 8)) & 1) v |= 1ull << i;
  }
  return v;
}

// minimal JSON: flat {"k": v, ...} for tensor_maps.jsonl rows
bool jstr(const std::string &line, const std::string &key,
          std::string *out) {
  const auto pos = line.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  auto q = line.find('"', line.find(':', pos) + 1);
  if (q == std::string::npos) return false;
  const auto e = line.find('"', q + 1);
  if (e == std::string::npos) return false;
  *out = line.substr(q + 1, e - q - 1);
  return true;
}
bool jnum(const std::string &line, const std::string &key,
          std::int64_t *out) {
  const auto pos = line.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  auto c = line.find(':', pos);
  if (c == std::string::npos) return false;
  *out = std::strtoll(line.c_str() + c + 1, nullptr, 10);
  return true;
}
// array of ints/doubles for "key": [a,b,...]
bool jvec(const std::string &line, const std::string &key,
          std::vector<std::int64_t> *out) {
  const auto pos = line.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  const auto a = line.find('[', pos);
  const auto b = line.find(']', pos);
  if (a == std::string::npos || b == std::string::npos || b < a)
    return false;
  const std::string inner = line.substr(a + 1, b - a - 1);
  std::istringstream ss(inner);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    const auto s = tok.find_first_not_of(" \t\"");
    if (s == std::string::npos) continue;
    const auto e = tok.find_last_not_of(" \t\"");
    out->push_back(std::strtoll(tok.c_str() + s, nullptr, 10));
    (void)e;
  }
  return true;
}
// nested object "key": {...} -> substring
bool jobj(const std::string &line, const std::string &key,
          std::string *out) {
  const auto pos = line.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  const auto a = line.find('{', pos);
  if (a == std::string::npos) return false;
  int depth = 0;
  for (std::size_t i = a; i < line.size(); ++i) {
    if (line[i] == '{') ++depth;
    if (line[i] == '}' && --depth == 0) {
      *out = line.substr(a, i - a + 1);
      return true;
    }
  }
  return false;
}

}  // namespace

std::map<int, TmEntry> load_tensor_maps(const std::string &case_dir) {
  std::map<int, TmEntry> out;
  std::ifstream in(case_dir + "/tensor_maps.jsonl");
  if (!in) return out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::int64_t id = -1;
    if (!jnum(line, "id", &id)) continue;
    TmEntry e;
    jstr(line, "image", &e.image);
    std::string enc;
    if (jobj(line, "encode", &enc)) {
      e.has_encode = true;
      for (const std::string &k :
           {"rank", "global_address", "data_type"}) {
        std::int64_t v;
        if (jnum(enc, k, &v)) e.encode_ints[k] = v;
      }
      std::string api;
      if (jstr(enc, "api", &api)) e.encode_strs["api"] = api;
      for (const std::string &k :
           {"box_dim", "global_dim", "global_strides", "element_strides",
            "pixel_box_lower_corner", "pixel_box_upper_corner"}) {
        std::vector<std::int64_t> v;
        if (jvec(enc, k, &v)) e.encode_vecs[k] = v;
      }
    }
    out[int(id)] = std::move(e);
  }
  return out;
}

std::optional<std::uint64_t> image_global_address(
    const std::string &image_hex) {
  bool ok;
  const auto raw = unhex(image_hex, &ok);
  if (!ok || raw.size() < 8) return std::nullopt;
  return getbits(raw, 0, 64);
}

std::optional<TmLayout> decode_tiled(const std::string &image_hex) {
  bool ok;
  const auto raw = unhex(image_hex, &ok);
  if (!ok || raw.size() < 0x40) return std::nullopt;
  const std::uint32_t control = std::uint32_t(getbits(raw, 64, 32));
  if (control & 1) {
    TmLayout l;
    l.mode = 1;
    l.global_address = getbits(raw, 0, 64);
    return l;
  }
  const int rank = int((control >> 4) & 7) + 1;
  const int code = int((control >> 7) & 0xF);
  const auto pit = kElementPackedBits.find(code);
  const auto eit = kElementBytes.find(code);
  if (pit == kElementPackedBits.end() && eit == kElementBytes.end())
    return std::nullopt;
  if (rank > 5) return std::nullopt;
  TmLayout l;
  l.mode = 0;
  l.global_address = getbits(raw, 0, 64);
  l.rank = rank;
  l.element_bytes = pit != kElementPackedBits.end() ? 1 : eit->second;
  l.interleave = int((control >> 11) & 3);
  l.swizzle = int((control >> 13) & 3);
  l.l2_promotion = int((control >> 17) & 3);
  for (int i = 0; i < rank - 1; ++i) {
    const std::uint64_t low = getbits(raw, 96 + 32 * i, 32);
    const std::uint64_t high = getbits(raw, 224 + 4 * i, 4);
    l.global_strides.push_back(((high << 32) | low) << 4);
  }
  for (int i = 0; i < rank; ++i) {
    l.box_dim.push_back(std::int64_t(getbits(raw, 440 + 8 * i, 8)) + 1);
    l.global_dim.push_back(
        std::int64_t(getbits(raw, 256 + 32 * i, 32)) + 1);
    l.element_strides.push_back(
        std::int64_t(getbits(raw, 416 + 3 * i, 3)) + 1);
  }
  if (pit != kElementPackedBits.end()) {
    const auto [logical, physical] = pit->second;
    if ((l.box_dim[0] * physical) % 8 || (l.global_dim[0] * logical) % 8)
      return std::nullopt;
    l.box_dim[0] = l.box_dim[0] * physical / 8;
    l.global_dim[0] = l.global_dim[0] * logical / 8;
  }
  return l;
}

std::optional<TmLayout> from_encode(const TmEntry &e) {
  if (!e.has_encode || !e.encode_vecs.count("box_dim")) return std::nullopt;
  const std::int64_t dtype =
      e.encode_ints.count("data_type") ? e.encode_ints.at("data_type") : -1;
  const auto pit = kPackedDtypeBits.find(int(dtype));
  const auto dit = kTensorDtypeBytes.find(int(dtype));
  double dtype_bytes = 0;
  if (pit == kPackedDtypeBits.end()) {
    if (dit == kTensorDtypeBytes.end() ||
        dit->second != double(std::int64_t(dit->second)))
      return std::nullopt;
    dtype_bytes = dit->second;
  }
  const int rank = e.encode_ints.count("rank")
      ? int(e.encode_ints.at("rank")) : int(e.encode_vecs.at("box_dim").size());
  TmLayout l;
  l.mode = e.encode_strs.count("api") &&
                   e.encode_strs.at("api") == "im2col" ? 1 : 0;
  // the tracer's encode hook does not record the base pointer: the
  // address lives only in the 128 B image (tensor_map_entry falls back
  // to the decoded image; neither -> missing_fact)
  if (e.encode_ints.count("global_address") &&
      e.encode_ints.at("global_address") != 0)
    l.global_address = std::uint64_t(e.encode_ints.at("global_address"));
  l.rank = rank;
  l.element_bytes = int(dtype_bytes);
  auto take = [&](const std::string &k, int n,
                  std::vector<std::int64_t> *v, int64_t fill = -1) {
    v->clear();
    const auto it = e.encode_vecs.find(k);
    for (int i = 0; i < n; ++i)
      if (it != e.encode_vecs.end() && i < int(it->second.size()))
        v->push_back(it->second[i]);
      else if (fill >= 0)
        v->push_back(fill);
  };
  take("box_dim", rank, &l.box_dim);
  take("global_dim", rank, &l.global_dim);
  take("element_strides", rank, &l.element_strides, 1);
  {
    const auto it = e.encode_vecs.find("global_strides");
    if (it != e.encode_vecs.end())
      for (int i = 0; i < rank - 1 && i < int(it->second.size()); ++i)
        l.global_strides.push_back(std::uint64_t(it->second[i]));
  }
  if (pit != kPackedDtypeBits.end()) {
    const auto [logical, physical] = pit->second;
    if ((l.box_dim[0] * physical) % 8 ||
        (!l.global_dim.empty() && (l.global_dim[0] * logical) % 8))
      return std::nullopt;
    l.box_dim[0] = l.box_dim[0] * physical / 8;
    if (!l.global_dim.empty())
      l.global_dim[0] = l.global_dim[0] * logical / 8;
    l.element_bytes = 1;
  }
  return l;
}

TmResult tensor_map_entry(const TmEntry &entry, const std::string &opcode,
                          const std::string &where) {
  std::string up = opcode;
  for (auto &c : up) c = char(std::toupper(c));
  std::smatch m;
  static const std::regex dre("\\.([0-9])D\\b");
  if (!std::regex_search(up, m, dre))
    missing(where, "tma rank from opcode " + opcode);
  const int rank = std::stoi(m[1]);
  const bool im2col = up.find(".IM2COL") != std::string::npos;
  const auto decoded = decode_tiled(entry.image);
  const auto encoded = from_encode(entry);
  const std::optional<TmLayout> layout =
      encoded ? encoded : decoded;
  if (!layout) missing(where, "tensor map layout");
  if (decoded && encoded && decoded->mode == 0 &&
      (decoded->rank != encoded->rank ||
       decoded->box_dim != encoded->box_dim ||
       decoded->global_dim != encoded->global_dim))
    missing(where, "tensor_map_decode");
  TmResult r;
  if (layout->mode == 1 || im2col) {
    if (entry.encode_strs.count("api") == 0 ||
        entry.encode_strs.at("api") != "im2col" ||
        !entry.encode_vecs.count("box_dim") ||
        !entry.encode_vecs.count("pixel_box_lower_corner") ||
        !entry.encode_vecs.count("pixel_box_upper_corner"))
      missing(where, "tensor_map_im2col_corners");
    const auto &lo = entry.encode_vecs.at("pixel_box_lower_corner");
    const auto &upv = entry.encode_vecs.at("pixel_box_upper_corner");
    if (int(lo.size()) != rank - 2 || int(upv.size()) != rank - 2)
      missing(where, "tensor_map_im2col_corners");
    if (!layout->element_bytes || *layout->element_bytes == 0)
      missing(where, "tensor_map_element_bytes");
    const auto &box = entry.encode_vecs.at("box_dim");
    std::uint64_t address = 0;
    if (entry.encode_ints.count("global_address"))
      address = std::uint64_t(entry.encode_ints.at("global_address"));
    if (!address) address = image_global_address(entry.image).value_or(0);
    if (!address) missing(where, "tensor_map_global_address");
    r.rank = rank;
    r.element_bytes = *layout->element_bytes;
    r.global_address = address;
    r.box_dim = {box[0], box[1]};
    r.api = "im2col";
    r.geometry = *layout;
    r.geometry.mode = 1;
    r.geometry.global_address = address;
    r.cpp = int(box[0]);
    r.ppc = int(box[1]);
    r.lower = lo;
    r.upper = upv;
    const auto &g = r.geometry;
    r.shared_bytes = r.global_bytes =
        std::uint64_t(r.cpp) * std::uint64_t(r.ppc) *
        std::uint64_t(*g.element_bytes);
    return r;
  }
  if (layout->rank != rank)
    missing(where, "tma encode rank contradicts opcode");
  std::optional<std::uint64_t> address = layout->global_address;
  if (!address) address = image_global_address(entry.image);
  if (!address || *address == 0)
    missing(where, "tensor_map_global_address");
  r.rank = rank;
  r.element_bytes = *layout->element_bytes;
  r.global_address = *address;
  r.global_dim = layout->global_dim;
  r.stride_bytes = layout->global_strides;
  r.box_dim = layout->box_dim;
  r.api = "tiled";
  r.geometry = *layout;
  r.geometry.global_address = *address;
  std::uint64_t box_bytes = r.element_bytes;
  for (const auto b : r.box_dim) box_bytes *= std::uint64_t(b);
  r.shared_bytes = r.global_bytes = box_bytes;
  return r;
}

TmRect tiled_box_rectangle(const TmLayout &g,
                           const std::vector<std::int64_t> &coords) {
  const int rank = g.rank;
  if (int(coords.size()) < rank)
    missing("tma_box", "coordinate count below tensor rank");
  const int elem = *g.element_bytes;
  std::vector<std::int64_t> strides = {elem};
  for (const auto s : g.global_strides) strides.push_back(std::int64_t(s));
  TmRect r;
  r.elem_bytes = elem;
  r.base = g.global_address.value_or(0);
  std::int64_t offset = 0;
  for (int k = 0; k < rank; ++k) {
    const int64_t es = g.element_strides[k];
    const int64_t n = (g.box_dim[k] + es - 1) / es;
    const std::int64_t c = coords[k];
    const int64_t dim = g.global_dim[k];
    const int64_t j_lo = c >= 0 ? 0 : (-c + es - 1) / es;
    const int64_t j_hi = c < dim ? std::min<int64_t>(n - 1, (dim - 1 - c) / es)
                                 : -1;
    const int64_t count = std::max<int64_t>(0, j_hi - j_lo + 1);
    r.counts.push_back(count);
    r.steps.push_back(es * strides[k]);
    offset += (c + j_lo * es) * strides[k];
  }
  std::uint64_t touched = 1;
  for (const auto c : r.counts) touched *= std::uint64_t(c);
  if (touched) r.base += std::uint64_t(offset);
  r.bytes = std::uint64_t(elem) * touched;
  return r;
}

std::vector<std::int64_t> tma_box_sectors(
    const TmLayout &g, const std::vector<std::int64_t> &coords) {
  const auto rect = tiled_box_rectangle(g, coords);
  if (rect.bytes == 0) return {};
  std::vector<std::int64_t> out;
  const int64_t inner_n =
      rect.steps[0] == rect.elem_bytes ? 1 : rect.counts[0];
  const int64_t row_span = rect.steps[0] == rect.elem_bytes
      ? rect.counts[0] * rect.elem_bytes : rect.elem_bytes;
  // iterate product of counts[1:]
  std::vector<int64_t> idx(rect.counts.size() - 1, 0);
  const int nd = int(rect.counts.size()) - 1;
  while (true) {
    std::int64_t row = std::int64_t(rect.base);
    for (int i = 0; i < nd; ++i) row += idx[i] * rect.steps[i + 1];
    for (int64_t j = 0; j < inner_n; ++j) {
      const std::int64_t start = row + j * rect.steps[0];
      for (std::int64_t s = start / kSector * kSector;
           s <= (start + row_span - 1) / kSector * kSector;
           s += kSector)
        out.push_back(s);
    }
    int i = 0;
    for (; i < nd; ++i)
      if (++idx[i] < rect.counts[i + 1]) break;
      else idx[i] = 0;
    if (i >= nd) break;
    if (nd <= 0) break;
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<std::int64_t> bulk_sectors(std::uint64_t address,
                                       std::uint64_t size) {
  if (size == 0) return {};
  std::vector<std::int64_t> out;
  const std::int64_t lo = std::int64_t(address) / kSector * kSector;
  const std::int64_t hi =
      std::int64_t(address + size - 1) / kSector * kSector;
  for (std::int64_t s = lo; s <= hi; s += kSector) out.push_back(s);
  return out;
}

std::vector<std::int64_t> im2col_offsets(int rank, std::uint64_t value) {
  const int width = rank == 3 ? 16 : rank == 4 ? 8 : 5;
  std::vector<std::int64_t> out;
  for (int i = 0; i < rank - 2; ++i)
    out.push_back(std::int64_t((value >> (i * width)) &
                               ((1ull << width) - 1)));
  return out;
}

std::int64_t s32(std::int64_t v) {
  v &= 0xFFFFFFFFll;
  return v & 0x80000000ll ? v - 0x100000000ll : v;
}

std::vector<std::int64_t> im2col_box_sectors(
    const TmResult &g, const std::vector<std::int64_t> &coords,
    const std::vector<std::int64_t> &offsets) {
  const int rank = g.geometry.rank;
  if (int(coords.size()) < rank)
    missing("im2col_box", "coordinate count below rank");
  const std::int64_t c = s32(coords[0]);
  std::vector<std::int64_t> spatial;
  for (int i = 0; i < rank - 2; ++i) spatial.push_back(s32(coords[1 + i]));
  const std::int64_t n = s32(coords[rank - 1]);
  const int ns = rank - 2;
  const auto &dim = g.geometry.global_dim;
  const auto &stride = g.geometry.global_strides;
  const auto &es = g.geometry.element_strides;
  if (es[0] != 1) missing("im2col_box", "tma_im2col_element_stride");
  const int64_t cpp = g.cpp, ppc = g.ppc;
  std::vector<std::int64_t> size(dim.begin() + 1, dim.begin() + rank - 1);
  std::vector<std::int64_t> P, q0;
  for (int i = 0; i < ns; ++i) {
    const int64_t p = (size[i] + g.upper[i] - g.lower[i] - 1) / es[i] + 1;
    const int64_t num = spatial[i] - g.lower[i];
    if (num % es[i] || num / es[i] < 0 || num / es[i] >= p)
      missing("im2col_box", "tma_im2col_coordinate");
    P.push_back(p);
    q0.push_back(num / es[i]);
  }
  int64_t L = n;
  for (int i = ns - 1; i >= 0; --i) L = q0[i] + P[i] * L;
  const int64_t elem = *g.geometry.element_bytes;
  const int64_t big_n = rank > 2 ? dim[rank - 1] : 1;
  std::vector<std::int64_t> out;
  std::set<std::int64_t> seen;
  for (int64_t k = 0; k < ppc; ++k) {
    int64_t rest = L + k;
    std::vector<std::int64_t> q;
    for (int i = 0; i < ns; ++i) {
      q.push_back(rest % P[i]);
      rest /= P[i];
    }
    const int64_t n_k = rest;
    if (n_k >= big_n) continue;
    std::vector<std::int64_t> a(ns);
    bool oob = false;
    for (int i = 0; i < ns; ++i) {
      a[i] = g.lower[i] + q[i] * es[i] + offsets[i];
      if (a[i] < 0 || a[i] >= size[i]) oob = true;
    }
    if (oob) continue;
    const int64_t c_lo = std::max<int64_t>(c, 0);
    const int64_t c_hi = std::min<int64_t>(c + cpp, dim[0]);
    if (c_lo >= c_hi) continue;
    std::int64_t addr = std::int64_t(g.geometry.global_address.value_or(0)) +
        c_lo * elem;
    for (int i = 0; i < ns; ++i) addr += a[i] * int64_t(stride[i]);
    addr += n_k * int64_t(stride[ns]);
    const int64_t end = addr + (c_hi - c_lo) * elem - 1;
    for (int64_t s = addr / kSector * kSector;
         s <= end / kSector * kSector; s += kSector)
      if (seen.insert(s).second) out.push_back(s);
  }
  return out;
}

}  // namespace cyclescope_convert::capture
