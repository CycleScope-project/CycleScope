#include "capture/trace_reader.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cyclescope_convert::capture {
namespace {

[[noreturn]] void die(const std::string &m) {
  throw std::runtime_error("capture_trace: " + m);
}
bool starts(const std::string &s, const char *p) {
  return s.rfind(p, 0) == 0;
}
bool is_ur(const std::string &op) {
  return starts(op, "UR") && !starts(op, "URANK");
}

std::uint32_t r32(const std::uint8_t *p) {
  std::uint32_t v; std::memcpy(&v, p, 4); return v;
}
std::int32_t ri32(const std::uint8_t *p) {
  std::int32_t v; std::memcpy(&v, p, 4); return v;
}
std::uint64_t r64(const std::uint8_t *p) {
  std::uint64_t v; std::memcpy(&v, p, 8); return v;
}

// _ZstStream: `zstd -dc` pipe (or raw .bin), chunked fill
struct Stream {
  FILE *f = nullptr;
  bool via_pipe = false;
  std::vector<std::uint8_t> buf;
  std::size_t head = 0;         // consumed prefix; compact when large
  bool eof = false;
  std::uint64_t pos = 0;

  void open(const fs::path &p) {
    if (p.extension() == ".zst") {
      const std::string cmd = "zstd -dc '" + p.string() + "'";
      f = popen(cmd.c_str(), "r");
      via_pipe = true;
    } else {
      f = fopen(p.c_str(), "rb");
    }
    if (!f) die("cannot open " + p.string());
  }
  std::size_t avail() const { return buf.size() - head; }
  void compact() {
    if (head >= (1 << 20)) {
      buf.erase(buf.begin(), buf.begin() + static_cast<long>(head));
      head = 0;
    }
  }
  std::size_t fill(std::size_t need) {
    compact();
    while (avail() < need && !eof) {
      const std::size_t want =
          std::max<std::size_t>(1 << 20, need - avail());
      const std::size_t old = buf.size();
      buf.resize(old + want);
      const std::size_t got = fread(buf.data() + old, 1, want, f);
      buf.resize(old + got);
      if (!got) { eof = true; break; }
    }
    return avail() >= need ? need : avail();
  }
  std::size_t read(std::uint8_t *out, std::size_t n) {
    n = fill(n);
    std::memcpy(out, buf.data() + head, n);
    head += n;
    pos += n;
    return n;
  }
  std::size_t skip(std::uint64_t n) {
    std::size_t skipped = 0;
    std::vector<std::uint8_t> tmp(1 << 16);
    while (skipped < n) {
      const std::size_t got = read(tmp.data(), std::min<std::uint64_t>(
          tmp.size(), n - skipped));
      if (!got) break;
      skipped += got;
    }
    return skipped;
  }
  void close() {
    if (!f) return;
    if (via_pipe) pclose(f); else fclose(f);
    f = nullptr;
  }
  ~Stream() { close(); }
};

}  // namespace

Layouts load_layouts(const std::string &functions_dir) {
  Layouts layouts;
  if (!fs::is_directory(functions_dir))
    die("no functions dir " + functions_dir);
  for (const auto &e : fs::directory_iterator(functions_dir)) {
    if (e.path().extension() != ".tsv") continue;
    const std::uint32_t fid =
        static_cast<std::uint32_t>(std::stoul(e.path().stem().string()));
    std::ifstream fh(e.path());
    std::string line;
    while (std::getline(fh, line)) {
      if (line.empty()) continue;
      std::vector<std::string> parts;
      std::size_t b = 0;
      while (true) {
        const std::size_t t = line.find('\t', b);
        parts.push_back(line.substr(b, t == std::string::npos ? t : t - b));
        if (t == std::string::npos) break;
        b = t + 1;
      }
      if (parts.size() < 6) continue;
      std::vector<std::string> ops;
      if (!parts[4].empty()) {
        std::size_t c = 0;
        while (true) {
          const std::size_t t = parts[4].find(',', c);
          ops.push_back(parts[4].substr(
              c, t == std::string::npos ? t : t - c));
          if (t == std::string::npos) break;
          c = t + 1;
        }
      }
      layouts[fid][static_cast<std::uint32_t>(std::stoul(parts[0]))] =
          {std::move(ops),
           static_cast<std::uint32_t>(std::stoul(parts[5]))};
    }
  }
  return layouts;
}

std::uint64_t op_entry_len(const std::string &op, std::uint64_t exec) {
  if (op == "PRED" || starts(op, "BRVAL")) return 0;
  if (op == "SPACE") return 12;
  if (starts(op, "URANK") || starts(op, "TCOORD") || op == "DESC" ||
      op == "CPOL" || op == "CPOLH" || starts(op, "UR") ||
      op == "IDESC" || op == "GDESC" || op == "GDESCH" || op == "TMEMD")
    return 4;
  return 4 * exec;
}

std::uint64_t record_tail_len(const std::vector<std::string> &ops,
                              std::uint32_t mref_count,
                              std::uint64_t exec) {
  std::uint64_t n = 8ull * mref_count * exec;
  for (const auto &op : ops) n += op_entry_len(op, exec);
  return n;
}

void iter_records(
    const std::string &case_dir, int launch,
    const std::string &trace_path,
    const std::function<bool(std::uint32_t, std::uint32_t,
                             std::uint32_t)> &cluster_filter,
    std::map<std::array<std::uint32_t, 3>,
             std::array<std::uint32_t, 10>> *cta_info,
    std::vector<std::uint64_t> *gsync_seq,
    const std::function<void(DynRecord &&)> &emit) {
  const Layouts layouts = load_layouts(case_dir + "/functions");
  fs::path path;
  if (!trace_path.empty()) {
    path = trace_path;
  } else {
    path = fs::path(case_dir) / ("trace-" + std::to_string(launch) +
                                ".bin.zst");
    if (!fs::is_regular_file(path)) {
      const fs::path raw = fs::path(case_dir) /
          ("trace-" + std::to_string(launch) + ".bin");
      if (fs::is_regular_file(raw)) path = raw;
    }
  }
  Stream in;
  in.open(path);
  std::uint8_t fhdr[16];
  if (in.read(fhdr, 16) != 16) die("truncated file header: " + path.string());
  if (std::memcmp(fhdr, "GSIMTRC", 8) != 0)
    die("bad magic: " + path.string());

  const auto pending_key = [](std::uint32_t x, std::uint32_t y,
                              std::uint32_t z, std::uint32_t w) {
    // ((x<<21|y)<<21|z)<<21|w packs positive cta coords; warp << 44
    return (std::uint64_t(w) << 44) |
        (std::uint64_t(z & 0x7FF) << 33) |
        (std::uint64_t(y & 0x7FF) << 22) |
        (std::uint64_t(x & 0x3FFFFF) );
  };
  struct Slot { DynRecord rec; bool live = true; };
  std::map<std::uint64_t, Slot> pending;
  std::vector<std::uint64_t> order;   // insertion order (Python dict)
  std::uint64_t gseq = 0;
  std::size_t seq_i = 0;

  const auto flush_nonkey = [&](std::uint64_t keep_key) {
    for (const auto k : order) {
      if (k == keep_key) continue;
      auto it = pending.find(k);
      if (it != pending.end()) {
        emit(std::move(it->second.rec));
        pending.erase(it);
      }
    }
  };

  std::uint8_t hbuf[32];
  for (;;) {
    const std::size_t got = in.read(hbuf, 32);
    if (!got) break;
    if (got != 32) die("trailing head in " + path.string());
    Head h;
    std::memcpy(&h, hbuf, 32);
    const std::uint64_t key =
        pending_key(h.cta_x, h.cta_y, h.cta_z, h.warp);
    const bool keep =
        !cluster_filter || cluster_filter(h.cta_x, h.cta_y, h.cta_z);

    if (h.kind == 3) {          // CTA info (v6.1): 40B body
      std::uint8_t b[40];
      in.read(b, 40);
      if (cta_info) {
        const std::array<std::uint32_t, 3> ck{h.cta_x, h.cta_y, h.cta_z};
        if (!cta_info->count(ck)) {
          std::array<std::uint32_t, 10> v{};
          for (int i = 0; i < 10; ++i) std::memcpy(&v[i], b + i * 4, 4);
          (*cta_info)[ck] = v;
        }
      }
      continue;
    }
    if (h.kind == 1) {          // AFTER
      std::uint8_t b4[4];
      in.read(b4, 4);
      if (!keep) continue;
      const std::uint32_t ready = r32(b4);
      auto it = pending.find(key);
      if (it != pending.end() && it->second.rec.idx == h.idx &&
          !it->second.rec.has_after_ready) {
        it->second.rec.after_ready = ready;
        it->second.rec.has_after_ready = true;
      } else {
        flush_nonkey(key);
        if (!pending.count(key))
          die("AFTER idx=" + std::to_string(h.idx) + " has no instruction");
      }
      continue;
    }
    if (h.kind == 2) {          // GSYNC
      std::uint64_t seq;
      if (gsync_seq) {
        if (seq_i >= gsync_seq->size())
          die("more GSYNC records than gsync_seq entries");
        seq = (*gsync_seq)[seq_i++];
      } else {
        seq = gseq;
      }
      ++gseq;
      std::uint8_t t4[4];
      in.read(t4, 4);
      const std::uint8_t phase = t4[0], nwords = t4[1], flags = t4[2];
      int exec = 0;
      for (std::uint32_t m = h.exec_mask; m; m &= m - 1) ++exec;
      const std::size_t body_n =
          static_cast<std::size_t>(exec) * nwords * 4;
      if (!keep) { in.skip(body_n); continue; }
      std::vector<std::uint8_t> body(body_n);
      if (body_n) in.read(body.data(), body_n);
      DynRecord::GSync g;
      g.phase = phase; g.nwords = nwords; g.flags = flags;
      g.sequence = seq;
      for (int i = 0; i < exec; ++i) {
        std::vector<std::uint32_t> w(nwords);
        for (int j = 0; j < nwords; ++j)
          w[j] = r32(body.data() + (i * nwords + j) * 4);
        g.words.push_back(std::move(w));
      }
      auto it = pending.find(key);
      if (it == pending.end() || it->second.rec.idx != h.idx) {
        flush_nonkey(key);
        it = pending.find(key);
        if (it == pending.end())
          die("GSYNC idx=" + std::to_string(h.idx) + " has no instruction");
      }
      it->second.rec.gsync.push_back(std::move(g));
      continue;
    }
    if (h.kind != 0) die("unknown record kind " + std::to_string(h.kind));
    {                                            // INSTR
      auto pit = pending.find(key);
      if (pit != pending.end()) {
        emit(std::move(pit->second.rec));
        pit->second.live = false;
        pending.erase(pit);
      }
      auto lit = layouts.find(h.function);
      if (lit == layouts.end() ||
          !lit->second.count(h.idx))
        die("no layout function=" + std::to_string(h.function) +
            " idx=" + std::to_string(h.idx));
      const auto &[opnames, mref_count] = lit->second.at(h.idx);
      int exec = 0;
      for (std::uint32_t m = h.exec_mask; m; m &= m - 1) ++exec;
      if (!keep) {
        in.skip(record_tail_len(opnames, mref_count, exec));
        continue;
      }
      DynRecord rec;
      rec.launch = launch;
      rec.cta_x = h.cta_x; rec.cta_y = h.cta_y; rec.cta_z = h.cta_z;
      rec.warp = h.warp; rec.sm = h.sm; rec.warp_slot = h.warp_slot;
      rec.function_id = h.function; rec.idx = h.idx;
      rec.active_mask = h.active_mask; rec.exec_mask = h.exec_mask;
      std::vector<std::uint8_t> tail;
      for (std::uint32_t m_i = 0; m_i < mref_count; ++m_i) {
        tail.resize(static_cast<std::size_t>(exec) * 8);
        in.read(tail.data(), tail.size());
        std::vector<std::uint64_t> addrs(exec);
        for (int i = 0; i < exec; ++i)
          addrs[i] = r64(tail.data() + i * 8);
        rec.mrefs.push_back(std::move(addrs));
      }
      int gdesc_i = -1, tmemd_i = -1;
      const auto set_op = [&](const std::string &k,
                              std::vector<std::int64_t> v) {
        if (!rec.ops.count(k)) rec.ops_order.push_back(k);
        rec.ops[k] = std::move(v);
      };
      for (const auto &op : opnames) {
        const std::uint64_t n = op_entry_len(op, exec);
        if (!n) continue;
        tail.resize(n);
        in.read(tail.data(), n);
        if (op == "SPACE") {
          set_op(op, {r32(tail.data()), r32(tail.data() + 4),
                      r32(tail.data() + 8)});
        } else if (starts(op, "BRANK") || starts(op, "RANK")) {
          std::vector<std::int64_t> v;
          for (int i = 0; i < exec; ++i)
            v.push_back(ri32(tail.data() + i * 4));
          set_op(op, std::move(v));
        } else if (starts(op, "URANK") || op == "DESC" ||
                   op == "CPOL" || op == "CPOLH" || is_ur(op) ||
                   op == "IDESC" || op == "GDESC" || op == "GDESCH" ||
                   op == "TMEMD" || starts(op, "TCOORD")) {
          const std::uint32_t val = r32(tail.data());
          set_op(op, {val});
          if (op == "GDESC") {
            ++gdesc_i;
            set_op("GDESC" + std::to_string(gdesc_i), {val});
          } else if (op == "GDESCH" && gdesc_i >= 0) {
            set_op("GDESC" + std::to_string(gdesc_i) + "H", {val});
          } else if (op == "TMEMD") {
            ++tmemd_i;
            set_op("TMEMD" + std::to_string(tmemd_i), {val});
          }
        } else {   // R<n>
          std::vector<std::int64_t> v;
          for (int i = 0; i < exec; ++i)
            v.push_back(r32(tail.data() + i * 4));
          set_op(op, std::move(v));
        }
      }
      if (gdesc_i >= 0)
        set_op("GDESC_COUNT", {gdesc_i + 1});
      if (tmemd_i >= 0)
        set_op("TMEMD_COUNT", {tmemd_i + 1});
      order.push_back(key);
      pending[key] = Slot{std::move(rec), true};
    }
  }
  if (gsync_seq && seq_i < gsync_seq->size())
    die("gsync_seq has entries beyond stream end");
  for (const auto k : order) {
    auto it = pending.find(k);
    if (it != pending.end()) {
      emit(std::move(it->second.rec));
      pending.erase(it);
    }
  }
}

}  // namespace cyclescope_convert::capture
