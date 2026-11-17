#include "convert/facts_reader.hpp"

namespace cyclescope_convert {

// header: <8sIIQQII32s = 72 bytes; directory entry: <IIQQQ = 32 bytes
static std::uint64_t r64(const std::uint8_t *p) {
  std::uint64_t v; std::memcpy(&v, p, 8); return v;
}
static std::uint32_t r32(const std::uint8_t *p) {
  std::uint32_t v; std::memcpy(&v, p, 4); return v;
}

void FactsFile::open(const std::string &path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("open " + path);
  size_ = static_cast<std::size_t>(::lseek(fd_, 0, SEEK_END));
  map_ = static_cast<const std::uint8_t *>(
      ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    throw std::runtime_error("mmap " + path);
  }
  const bool v2 = std::memcmp(map_, "GSIMCP2", 8) == 0;
  if (size_ < 72 ||
      (!v2 && std::memcmp(map_, "GSIMCAP0", 8) != 0 &&
       std::memcmp(map_, "GSIMPRG0", 8) != 0))
    throw std::runtime_error("bad magic: " + path);
  const auto dir_off = r64(map_ + 24);
  const auto dir_n = r64(map_ + 32);
  for (std::size_t i = 0; i < dir_n; ++i) {
    const auto *e = map_ + dir_off + i * 32;
    const auto rec = r32(e + 4);
    const auto cnt = r64(e + 8);
    const auto off = r64(e + 16);
    const auto tag = e[0] | (e[1] << 8) | (e[2] << 16) | (e[3] << 24);
    dir_[tag] = {off, rec ? cnt * rec : cnt};
    dir_rec_[tag] = rec;
  }
  // OPSZ grew a flags word (after_ready presence, GSIMCAP01); facts
  // written before that carry 28-byte rows and must be regenerated.
  if (has_table("OPSZ") && dir_rec_[tag4("OPSZ")] != facts::OPS)
    throw std::runtime_error(
        "unsupported:facts_format:opsz_record_bytes (regenerate facts)");
  // MREX grew a cut_rows word (tma_oob v2, GSIMCAP01/02); facts written
  // before that carry 16-byte rows and must be regenerated
  if (has_table("MREX") && dir_rec_[tag4("MREX")] != facts::MREX)
    throw std::runtime_error(
        "missing_fact:mrex_record_bytes (regenerate facts)");
  if (const auto *m = table("META", nullptr))
    cluster_id_ = r32(m);
  launch_path_ = path;
  // sibling launch.json sits next to the .gsc
  const auto slash = launch_path_.find_last_of('/');
  launch_path_ = (slash == std::string::npos)
                     ? "launch.json"
                     : launch_path_.substr(0, slash) + "/launch.json";
  if (v2) {
    // GSIMCAP02: STAT-family tables and static strings live in the
    // shared sibling statics.gsc (once per facts dir)
    const std::string sp = (slash == std::string::npos)
        ? "statics.gsc" : path.substr(0, slash) + "/statics.gsc";
    sfd_ = ::open(sp.c_str(), O_RDONLY);
    if (sfd_ < 0)
      throw std::runtime_error("missing_fact:facts_statics (" + sp + ")");
    stat_size_ = static_cast<std::size_t>(::lseek(sfd_, 0, SEEK_END));
    stat_map_ = static_cast<const std::uint8_t *>(
        ::mmap(nullptr, stat_size_, PROT_READ, MAP_PRIVATE, sfd_, 0));
    if (stat_map_ == MAP_FAILED || stat_size_ < 72 ||
        std::memcmp(stat_map_, "GSIMCPS", 8) != 0)
      throw std::runtime_error("bad statics: " + sp);
    const auto soff = r64(stat_map_ + 24);
    const auto sn = r64(stat_map_ + 32);
    for (std::size_t i = 0; i < sn; ++i) {
      const auto *e = stat_map_ + soff + i * 32;
      const auto rec = r32(e + 4);
      const auto cnt = r64(e + 8);
      const auto off = r64(e + 16);
      const auto tag = e[0] | (e[1] << 8) | (e[2] << 16) | (e[3] << 24);
      stat_dir_[tag] = {off, rec ? cnt * rec : cnt};
    }
  }
  strs_ = table("STRS", &strs_n_);
  if (!strs_) strs_n_ = 0;
  sstrs_ = stat_map_ ? stat_table("STRS", &sstrs_n_) : table("STRS", &sstrs_n_);
  if (!sstrs_) sstrs_n_ = 0;
}

void FactsFile::close() {
  if (map_) ::munmap(const_cast<std::uint8_t *>(map_), size_);
  map_ = nullptr;
  strs_ = sstrs_ = nullptr;
  strs_n_ = sstrs_n_ = 0;
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
  if (stat_map_) ::munmap(const_cast<std::uint8_t *>(stat_map_),
                          stat_size_);
  stat_map_ = nullptr;
  if (sfd_ >= 0) ::close(sfd_);
  sfd_ = -1;
}

const std::uint8_t *FactsFile::stat_table(const char *tag,
                                          std::size_t *size) const {
  auto it = stat_dir_.find(tag4(tag));
  if (it == stat_dir_.end()) {
    if (size) *size = 0;
    return nullptr;
  }
  if (size) *size = static_cast<std::size_t>(it->second.second);
  return stat_map_ + it->second.first;
}

bool FactsFile::has_table(const char *tag) const {
  return dir_.count(tag4(tag)) != 0;
}

const std::uint8_t *FactsFile::table(const char *tag,
                                     std::size_t *size) const {
  const auto it = dir_.find(tag4(tag));
  if (it == dir_.end()) {
    if (stat_map_) return stat_table(tag, size);
    return nullptr;
  }
  if (size) *size = static_cast<std::size_t>(it->second.second);
  return map_ + it->second.first;
}

}  // namespace cyclescope_convert
