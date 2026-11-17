// tensor_maps.jsonl -> TensorMap + TMA box sectors/bytes
// (python/cyclescope/convert/tensor_map.py verbatim port).
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyclescope_convert::capture {

// decoded/encoded layout — a field-absent map mirroring Python dict
struct TmLayout {
  int mode = 0;                       // 0 tiled, 1 im2col sentinel
  // the image always carries the base pointer; the tracer's encode
  // record never does (optional mirrors the Python None)
  std::optional<std::uint64_t> global_address;
  int rank = 0;
  std::optional<int> element_bytes;
  int interleave = 0, swizzle = 0, l2_promotion = 0;
  std::vector<std::uint64_t> global_strides;
  std::vector<std::int64_t> global_dim;
  std::vector<int64_t> element_strides;
  std::vector<int64_t> box_dim;
};

// tensor_maps.jsonl row
struct TmEntry {
  std::string image;
  std::map<std::string, std::string> encode_strs;   // raw JSON-ish values
  std::map<std::string, std::vector<std::int64_t>> encode_vecs;
  std::map<std::string, std::int64_t> encode_ints;
  bool has_encode = false;
};

std::map<int, TmEntry> load_tensor_maps(const std::string &case_dir);

// base pointer from the image's first 8 bytes (layout decode not needed)
std::optional<std::uint64_t> image_global_address(const std::string &image_hex);
std::optional<TmLayout> decode_tiled(const std::string &image_hex);
std::optional<TmLayout> from_encode(const TmEntry &e);

struct TmResult {
  // model.TensorMap fields
  int rank = 0;
  int element_bytes = 0;
  std::uint64_t global_address = 0;
  std::vector<std::int64_t> global_dim;
  std::vector<std::uint64_t> stride_bytes;
  std::vector<std::int64_t> box_dim;
  // facts
  std::string api;
  std::uint64_t shared_bytes = 0, global_bytes = 0;
  // geometry
  TmLayout geometry;
  int cpp = 0, ppc = 0;
  std::vector<std::int64_t> lower, upper;
};

TmResult tensor_map_entry(const TmEntry &entry, const std::string &opcode,
                          const std::string &where);

struct TmRect {
  std::uint64_t base = 0;
  int elem_bytes = 0;
  std::vector<int64_t> counts, steps;
  std::uint64_t bytes = 0;
};

TmRect tiled_box_rectangle(const TmLayout &geometry,
                           const std::vector<std::int64_t> &coords);
std::vector<std::int64_t> tma_box_sectors(const TmLayout &geometry,
                                          const std::vector<std::int64_t> &coords);
std::vector<std::int64_t> bulk_sectors(std::uint64_t address,
                                       std::uint64_t size);
std::vector<std::int64_t> im2col_offsets(int rank, std::uint64_t value);
std::vector<std::int64_t> im2col_box_sectors(const TmResult &geometry,
                                             const std::vector<std::int64_t> &coords,
                                             const std::vector<std::int64_t> &offsets);
std::int64_t s32(std::int64_t v);

}  // namespace cyclescope_convert::capture
