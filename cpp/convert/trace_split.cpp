// cyclescope-trace-split — route a GSIMTRC record stream into per-shard
// files in one decompression pass, so each `convert --shard k/n`
// reads only its own records instead of rescanning the whole trace.
//
// usage: cyclescope-trace-split --trace <trace-N.bin.zst|.bin>
//        --functions <case>/functions --grid gx,gy,gz
//        --cluster cx,cy,cz --shards N --out <dir>
//
// Record layout mirrors python/cyclescope/convert/trace_reader.py exactly:
// 32B V5Head; INSTR tail = mref_count*exec_count*8 + sum(op tail);
// AFTER tail = 4B; GSYNC tail = 4B + exec_count*nwords*4.
// Every record (AFTER/GSYNC included) is routed by its own head's cta:
// cl = cta.x//cx + (cta.y//cy)*ncx + (cta.z//cz)*ncx*ncy,
// ncx = ceil(gx/cx), ncy = ceil(gy/cy); shard = cl % N.
// Outputs <dir>/trace-<launch>.shard<k>of<N>.bin.zst (verbatim 16B
// file header + raw record bytes, `zstd -1 -q` pipe) and
// <dir>/trace-<launch>.shard<k>of<N>.gsync_seq (u64 LE per GSYNC
// record in that shard: its ordinal among all GSYNC records of the
// original stream).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInstr = 0, kAfter = 1, kGsync = 2;
constexpr std::size_t kFileHeader = 16;
constexpr std::size_t kHeadSize = 32;
const char kMagic[8] = {'G', 'S', 'I', 'M', 'T', 'R', 'C', '\0'};

[[noreturn]] void die(const std::string &msg) {
  std::fprintf(stderr, "trace_split: %s\n", msg.c_str());
  std::exit(1);
}

struct Head {
  std::uint32_t cta_x;
  std::uint16_t cta_y, cta_z, warp;
  std::uint8_t sm, kind;
  std::uint32_t function, idx, active_mask, exec_mask, warp_slot;
};
static_assert(sizeof(Head) == 32, "Head must match the 32B V5Head");

struct Layout {              // per (function, idx): ops kinds + mrefs
  std::vector<std::string> ops;
  std::uint32_t mref_count;
};

std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::size_t b = 0;
  while (true) {
    std::size_t c = s.find(',', b);
    if (c == std::string::npos) { out.push_back(s.substr(b)); break; }
    out.push_back(s.substr(b, c - b));
    b = c + 1;
  }
  if (out.size() == 1 && out[0].empty()) out.clear();
  return out;
}

std::map<std::uint32_t, std::map<std::uint32_t, Layout>>
load_layouts(const fs::path &dir) {
  std::map<std::uint32_t, std::map<std::uint32_t, Layout>> layouts;
  if (!fs::is_directory(dir)) die("no functions dir " + dir.string());
  for (const auto &e : fs::directory_iterator(dir)) {
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
        std::size_t t = line.find('\t', b);
        if (t == std::string::npos) { parts.push_back(line.substr(b)); break; }
        parts.push_back(line.substr(b, t - b));
        b = t + 1;
      }
      if (parts.size() < 6) continue;
      Layout l;
      l.ops = split_csv(parts[4]);
      l.mref_count = static_cast<std::uint32_t>(std::stoul(parts[5]));
      layouts[fid][static_cast<std::uint32_t>(std::stoul(parts[0]))] = l;
    }
  }
  return layouts;
}

bool starts_with(const std::string &s, const char *p) {
  return s.rfind(p, 0) == 0;
}

std::uint64_t op_entry_len(const std::string &op, std::uint64_t exec) {
  if (op == "PRED" || starts_with(op, "BRVAL")) return 0;
  if (op == "SPACE") return 12;
  if (starts_with(op, "URANK") || starts_with(op, "TCOORD") ||
      op == "DESC" || op == "CPOL" || op == "CPOLH" ||
      starts_with(op, "UR") || op == "IDESC" || op == "GDESC" ||
      op == "GDESCH" || op == "TMEMD")
    return 4;
  return 4 * exec;  // RANK<k>, BRANK<k>, R<n>: per executing lane 4 bytes
}

struct Reader {
  FILE *f = nullptr;
  bool via_pipe = false;
  // set when the caller stops reading before EOF on purpose (probe
  // pass): the decompressor then dies of SIGPIPE, which is not a
  // truncated trace
  bool stopped_early = false;
  void open(const fs::path &p) {
    if (p.extension() == ".zst") {
      std::string cmd = "zstd -dc '" + p.string() + "'";
      f = popen(cmd.c_str(), "r");
      via_pipe = true;
    } else {
      f = std::fopen(p.c_str(), "rb");
    }
    if (!f) die("cannot open trace " + p.string());
  }
  std::size_t read(void *buf, std::size_t n) {
    return std::fread(buf, 1, n, f);
  }
  void read_exact(void *buf, std::size_t n, const char *what) {
    if (read(buf, n) != n)
      die(std::string("truncated trace: ") + what);
  }
  void skip(std::uint64_t n) {
    std::uint8_t tmp[8192];
    while (n) {
      const auto m = std::min<std::uint64_t>(n, sizeof tmp);
      if (read(tmp, m) != m) die("truncated trace: skip");
      n -= m;
    }
  }
  void close() {
    if (!f) return;
    if (via_pipe) {
      if (pclose(f) != 0 && !stopped_early)
        die("zstd -dc failed mid-stream (truncated trace?)");
    } else {
      std::fclose(f);
    }
    f = nullptr;
  }
  ~Reader() { close(); }
};

struct ShardWriter {
  FILE *z = nullptr;        // zstd pipe
  FILE *seq = nullptr;      // gsync_seq file
  std::vector<std::uint8_t> buf;
  std::uint64_t records = 0;
};

std::vector<int> parse_triple(const std::string &s, const char *name) {
  std::vector<int> v;
  std::size_t b = 0;
  while (v.size() < 3) {
    std::size_t c = s.find(',', b);
    v.push_back(std::stoi(s.substr(b, c == std::string::npos
                                        ? c : c - b)));
    if (c == std::string::npos) break;
    b = c + 1;
  }
  if (v.size() != 3) die(std::string("bad ") + name + " '" + s + "'");
  return v;
}

int launch_of(const fs::path &p) {
  std::string n = p.stem().string();          // trace-N  (or .bin stem)
  if (n.rfind("trace-", 0) == 0) {
    std::string t = n.substr(6);
    std::size_t dot = t.find('.');
    if (dot != std::string::npos) t = t.substr(0, dot);
    return std::stoi(t);
  }
  die("trace name not trace-<launch>: " + n);
}

}  // namespace

int main(int argc, char **argv) {
  fs::path trace, functions, out_dir;
  std::vector<int> grid, cluster;
  int nshards = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) die("missing value for " + a);
      return argv[i];
    };
    if (a == "--trace") trace = next();
    else if (a == "--functions") functions = next();
    else if (a == "--grid") grid = parse_triple(next(), "--grid");
    else if (a == "--cluster") cluster = parse_triple(next(), "--cluster");
    else if (a == "--shards") nshards = std::stoi(next());
    else if (a == "--out") out_dir = next();
    else die("unknown arg " + a);
  }
  if (trace.empty() || functions.empty() || grid.size() != 3 ||
      cluster.size() != 3 || nshards < 1 || out_dir.empty())
    die("usage: cyclescope-trace-split --trace T --functions D --grid x,y,z "
        "--cluster x,y,z --shards N --out D");

  auto layouts = load_layouts(functions);
  const int launch = launch_of(trace);
  const std::uint64_t cx = cluster[0] > 0 ? cluster[0] : 1;
  const std::uint64_t cy = cluster[1] > 0 ? cluster[1] : 1;
  const std::uint64_t cz = cluster[2] > 0 ? cluster[2] : 1;
  const std::uint64_t ncx = (grid[0] + cx - 1) / cx;
  const std::uint64_t ncy = (grid[1] + cy - 1) / cy;

  fs::create_directories(out_dir);
  std::vector<std::unique_ptr<ShardWriter>> shards(nshards);
  std::vector<std::uint8_t> fhdr(kFileHeader);
  for (int k = 0; k < nshards; ++k) {
    auto s = std::make_unique<ShardWriter>();
    std::string base = out_dir.string() + "/trace-" +
        std::to_string(launch) + ".shard" + std::to_string(k) + "of" +
        std::to_string(nshards);
    std::string cmd = "zstd -1 -q -o '" + base + ".bin.zst' -";
    s->z = popen(cmd.c_str(), "w");
    if (!s->z) die("cannot open zstd pipe for shard " + std::to_string(k));
    s->seq = std::fopen((base + ".gsync_seq").c_str(), "wb");
    if (!s->seq) die("cannot open " + base + ".gsync_seq");
    s->buf.reserve(1 << 21);
    shards[k] = std::move(s);
  }

  Reader in;
  in.open(trace);
  in.read_exact(fhdr.data(), kFileHeader, "file header");
  if (std::memcmp(fhdr.data(), kMagic, 8) != 0)
    die("bad magic in " + trace.string());
  for (auto &s : shards)
    std::fwrite(fhdr.data(), 1, kFileHeader, s->z);

  std::uint8_t head[kHeadSize];
  std::vector<std::uint8_t> tail;
  std::uint64_t gsync_seq = 0, total = 0;
  // first pass: does this trace carry kind-3 records at all? If not,
  // take the verbatim pre-kind-3 path (immediate emit, byte-identical
  // shard files to the original splitter).
  bool has_kind3 = false;
  {
    Reader probe;
    probe.open(trace);
    probe.skip(kFileHeader);
    std::uint8_t h2[kHeadSize];
    while (probe.read(h2, kHeadSize) == kHeadSize) {
      Head hh;
      std::memcpy(&hh, h2, kHeadSize);
      if (hh.kind == 3) {
        has_kind3 = true;
        probe.stopped_early = true;
        break;
      }
      std::uint64_t tl = 0;
      if (hh.kind == kAfter) tl = 4;
      else if (hh.kind == kGsync) {
        // the 4B (phase, nwords, flags, pad) head is consumed here;
        // only the exec_count*nwords*4 body remains to skip
        std::uint8_t t4[4];
        probe.read_exact(t4, 4, "gsync head");
        tl = (std::uint64_t)__builtin_popcount(hh.exec_mask) * t4[1] * 4;
      } else if (hh.kind == kInstr) {
        auto fi = layouts.find(hh.function);
        if (fi == layouts.end()) die("probe: no layout");
        auto ii = fi->second.find(hh.idx);
        if (ii == fi->second.end()) die("probe: no layout idx");
        const Layout &l = ii->second;
        const std::uint64_t ex = __builtin_popcount(hh.exec_mask);
        tl = 8ull * l.mref_count * ex;
        for (const auto &op : l.ops) tl += op_entry_len(op, ex);
      } else die("probe: unknown kind");
      probe.skip(tl);
    }
  }
  // kind-3 CTA records: cta (x,y,z) -> hardware cluster id linearized by
  // %nclusterid; when present it overrides the computed (cta/dims) cl.
  std::map<std::tuple<std::uint32_t, std::uint16_t, std::uint16_t>,
           std::uint64_t> cta_cluster;
  using CtaKey = std::tuple<std::uint32_t, std::uint16_t, std::uint16_t>;
  // records of a CTA that precede its kind-3 record: buffer them, then
  // flush to the hardware shard once the kind-3 mapping lands (mixed-
  // shape launches make attribute routing wrong); CTAs with no kind-3
  // flush at EOF to the attribute shard.
  // a buffered GSYNC record keeps the launch-wide ordinal it was
  // assigned when read (seq is only written for gsync records)
  struct Pending {
    std::vector<std::uint8_t> bytes;
    bool gsync = false;
    std::uint64_t seq = 0;
  };
  std::map<CtaKey, std::vector<Pending>> pending;
  std::uint64_t pending_total = 0;
  auto emit = [&](ShardWriter &s, const uint8_t *hd,
                  const uint8_t *tl, std::size_t tn, bool gs,
                  std::uint64_t seq) {
    s.buf.insert(s.buf.end(), hd, hd + kHeadSize);
    s.buf.insert(s.buf.end(), tl, tl + tn);
    if (gs && std::fwrite(&seq, sizeof(seq), 1, s.seq) != 1)
      die("gsync_seq write failed");
    if (s.buf.size() >= (1 << 21)) {
      std::fwrite(s.buf.data(), 1, s.buf.size(), s.z);
      s.buf.clear();
    }
    ++s.records;
  };
  while (true) {
    std::size_t got = in.read(head, kHeadSize);
    if (got == 0) break;
    if (got != kHeadSize) die("trailing partial head");
    Head h;
    std::memcpy(&h, head, kHeadSize);   // packed little-endian layout
    std::uint64_t cl =
        (h.cta_x / cx) + (h.cta_y / cy) * ncx + (h.cta_z / cz) * ncx * ncy;
    if (h.kind == 3) {
      // read the 40B body first to learn clusterid/nclusterid
      std::uint8_t b[40];
      in.read_exact(b, 40, "cta body");
      auto ru32 = [&](int i) {
        std::uint32_t v;
        std::memcpy(&v, b + i * 4, 4);
        return v;
      };
      const std::uint64_t cid = (std::uint64_t)ru32(4) +
          (std::uint64_t)ru32(5) * ru32(7) +
          (std::uint64_t)ru32(6) * ru32(7) * ru32(8);
      const CtaKey ck{h.cta_x, h.cta_y, h.cta_z};
      cta_cluster[ck] = cid;
      cl = cid;
      ShardWriter &s = *shards[static_cast<int>(cl % nshards)];
      emit(s, head, b, 40, false, 0);
      {
        auto it = pending.find(ck);
        if (it != pending.end()) {
          for (auto &rec : it->second)
            emit(s, rec.bytes.data(), rec.bytes.data() + kHeadSize,
                 rec.bytes.size() - kHeadSize, rec.gsync, rec.seq);
          pending.erase(it);
        }
      }
      ++total;
      continue;
    }
    const CtaKey ckey{h.cta_x, h.cta_y, h.cta_z};
    const auto ck = cta_cluster.find(ckey);
    if (ck != cta_cluster.end()) cl = ck->second;
    const bool mapped = ck != cta_cluster.end();
    std::uint64_t tail_len = 0;
    const std::uint64_t exec =
        static_cast<std::uint64_t>(__builtin_popcount(h.exec_mask));
    std::uint64_t gsync_words = 0;
    if (h.kind == kAfter) {
      tail_len = 4;
    } else if (h.kind == kGsync) {
      tail_len = 4;    // plus exec_count*nwords*4 after reading nwords
    } else if (h.kind == kInstr) {
      auto fi = layouts.find(h.function);
      if (fi == layouts.end())
        die("no layout function=" + std::to_string(h.function) +
            " idx=" + std::to_string(h.idx));
      auto ii = fi->second.find(h.idx);
      if (ii == fi->second.end())
        die("no layout function=" + std::to_string(h.function) +
            " idx=" + std::to_string(h.idx));
      const Layout &l = ii->second;
      tail_len = 8ull * l.mref_count * exec;
      for (const auto &op : l.ops) tail_len += op_entry_len(op, exec);
    } else {
      die("unknown record kind " + std::to_string(h.kind));
    }
    tail.resize(tail_len);
    if (h.kind == kGsync) {
      // 4B (phase,nwords,flags,pad) then exec_count*nwords*4
      std::uint8_t t4[4];
      in.read_exact(t4, 4, "gsync tail head");
      tail.resize(4);
      std::memcpy(tail.data(), t4, 4);
      gsync_words = exec * t4[1] * 4;
      tail.resize(4 + gsync_words);
      if (gsync_words)
        in.read_exact(tail.data() + 4, gsync_words, "gsync body");
    } else if (tail_len) {
      in.read_exact(tail.data(), tail_len, "record tail");
    }
    // buffer only when this trace actually carries kind-3 records —
    // old traces emit immediately, byte-identical to pre-kind-3 builds
    if (has_kind3 && !mapped) {
      ++pending_total;
      // kind-3 stream exists; this CTA's record precedes its CTA-info
      Pending rec;
      rec.bytes.reserve(kHeadSize + tail.size());
      rec.bytes.insert(rec.bytes.end(), head, head + kHeadSize);
      rec.bytes.insert(rec.bytes.end(), tail.begin(), tail.end());
      rec.gsync = h.kind == kGsync;
      rec.seq = gsync_seq;
      pending[ckey].push_back(std::move(rec));
      ++total;
      if (h.kind == kGsync) ++gsync_seq;
      continue;
    }
    const int shard = static_cast<int>(cl % nshards);
    ShardWriter &s = *shards[shard];
    emit(s, head, tail.data(), tail.size(), h.kind == kGsync, gsync_seq);
    ++total;
    if (h.kind == kGsync) ++gsync_seq;
  }
  // EOF: flush pending — CTAs never mapped by kind-3 go to their
  // attribute-path shard (same linearization as the routing above).
  for (auto &[ckey, recs] : pending) {
    const auto ck = cta_cluster.find(ckey);
    const std::uint64_t cl2 =
        ck != cta_cluster.end()
            ? ck->second
            : (std::get<0>(ckey) / cx) + (std::get<1>(ckey) / cy) * ncx +
                  (std::get<2>(ckey) / cz) * ncx * ncy;
    const int shard = static_cast<int>(cl2 % nshards);
    ShardWriter &s = *shards[shard];
    for (auto &rec : recs)
      emit(s, rec.bytes.data(), rec.bytes.data() + kHeadSize,
           rec.bytes.size() - kHeadSize, rec.gsync, rec.seq);
  }
  for (auto &s : shards) {
    if (!s->buf.empty())
      std::fwrite(s->buf.data(), 1, s->buf.size(), s->z);
    if (std::fclose(s->seq) != 0) die("gsync_seq close failed");
    if (pclose(s->z) != 0) die("zstd pipe failed");
  }
  for (int k = 0; k < nshards; ++k)
    std::printf("shard %d/%d: %llu records\n", k, nshards,
                (unsigned long long)shards[k]->records);
  std::printf("total: %llu records, %llu gsync\n",
              (unsigned long long)total, (unsigned long long)gsync_seq);
  return 0;
}
