// GSIMCAP01 facts reader — mmap views over fixed-layout tables.
// Mirrors python/cyclescope/convert/facts_format.py record layouts.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace cyclescope_convert {

struct RowView {
  const std::uint8_t *p = nullptr;
  template <typename T> T at(std::size_t off) const {
    T v; std::memcpy(&v, p + off, sizeof(T)); return v;
  }
  std::uint32_t u32(std::size_t off) const { return at<std::uint32_t>(off); }
  std::int32_t i32(std::size_t off) const { return at<std::int32_t>(off); }
  std::uint64_t u64(std::size_t off) const { return at<std::uint64_t>(off); }
};

// fact table tags / row sizes (4-byte tags)
namespace facts {
inline constexpr std::size_t META = 40, CTAS = 48, WARP = 40, STAT = 64,
    STAX = 24, DYN = 28, MREF = 24, MREX = 20, ADDR = 8, OPTK = 4,
    REGL = 4, RROL = 5, UKV = 13 /*packed*/, LANE = 12, LVAL = 8,
    CORD = 8, DESC = 8, OPS = 32, OPSX = 16 /*actual 20 per python*/,
    GSYN = 40, GSYX = 8, TMAP = 28, TMDI = 8, MBPI = 32;
}

class FactsFile {
public:
  FactsFile() = default;
  explicit FactsFile(const std::string &path) { open(path); }
  ~FactsFile() { close(); }
  FactsFile(const FactsFile &) = delete;
  FactsFile &operator=(const FactsFile &) = delete;

  void open(const std::string &path);
  void close();

  bool has_table(const char *tag) const;
  // raw table bytes (8-byte aligned payload)
  const std::uint8_t *table(const char *tag, std::size_t *size) const;
  RowView row(const char *tag, std::size_t size, std::size_t idx) const {
    std::size_t n = 0;
    const auto *t = table(tag, &n);
    if (!t || idx * size >= n)
      throw std::out_of_range(std::string("facts row ") + tag);
    return RowView{t + idx * size};
  }
  std::size_t rows(const char *tag, std::size_t size) const {
    std::size_t n = 0;
    const auto *t = table(tag, &n);
    return t ? n / size : 0;
  }
  const char *str_at(std::uint32_t off) const {
    if (off == 0xFFFFFFFFu) return nullptr;
    if (!strs_ || off >= strs_n_) return nullptr;
    return reinterpret_cast<const char *>(strs_ + off);
  }
  // statics.gsc STRS (GSIMCAP02): stat-row string offsets
  const char *sstr_at(std::uint32_t off) const {
    if (off == 0xFFFFFFFFu) return nullptr;
    if (!sstrs_ || off >= sstrs_n_) return nullptr;
    return reinterpret_cast<const char *>(sstrs_ + off);
  }
  std::uint32_t cluster_id() const { return cluster_id_; }
  // GSIMCAP02: STAT-family tables come from the shared statics.gsc
  bool has_shared_statics() const { return stat_map_ != nullptr; }
  const std::string &launch_path() const { return launch_path_; }

  const std::uint8_t *stat_table(const char *tag,
                                 std::size_t *size) const;
private:
  int fd_ = -1;
  const std::uint8_t *map_ = nullptr;
  std::size_t size_ = 0;
  // GSIMCAP02 companion: statics.gsc mmap (stat tables + static STRS)
  int sfd_ = -1;
  const std::uint8_t *stat_map_ = nullptr;
  // STRS views resolved once in open(): cluster-file strings and the
  // static-row strings (statics.gsc under v2, else the same table)
  const std::uint8_t *strs_ = nullptr, *sstrs_ = nullptr;
  std::size_t strs_n_ = 0, sstrs_n_ = 0;
  std::size_t stat_size_ = 0;
  std::unordered_map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>>
      stat_dir_;
  std::unordered_map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>> dir_;
  std::unordered_map<std::uint32_t, std::uint32_t> dir_rec_;
  std::uint32_t cluster_id_ = 0;
  std::string launch_path_;

  static std::uint32_t tag4(const char *tag) {
    std::uint32_t v = 0;
    std::memcpy(&v, tag, 4);
    return v;
  }
};

}  // namespace cyclescope_convert
