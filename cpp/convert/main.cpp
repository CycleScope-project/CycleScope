// cyclescope-convert — GSIMCAP01 -> GSIMSEG (phase 3a: no semantics/uses)
// usage: cyclescope-convert --facts DIR --programs FILE --arch ARCH
//        --out SEGDIR [--converter-digest HEX] [--missing-json FILE]
//        [--verbose]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "convert/facts_reader.hpp"
#include "convert/segment_builder.hpp"

using namespace cyclescope_convert;

static std::string hex32(const std::string &hex) {
  std::string out(32, '\0');
  for (int i = 0; i < 32 && 2 * i + 1 < (int)hex.size(); ++i)
    out[i] = static_cast<char>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
  return out;
}

static std::vector<int> cluster_shape(const std::string &launch_json) {
  // minimal scan: "cluster": [x, y, z]
  auto p = launch_json.find("\"cluster\"");
  std::vector<int> v{1, 1, 1};
  if (p == std::string::npos) return v;
  p = launch_json.find('[', p);
  if (p == std::string::npos) return v;
  for (int i = 0; i < 3; ++i) {
    p = launch_json.find_first_of("0123456789-", p + 1);
    if (p == std::string::npos) return v;
    v[i] = std::stoi(launch_json.substr(p));
    p = launch_json.find_first_of(",]", p);
    if (launch_json[p] == ']') break;
  }
  return v;
}

static std::string json_str(const std::string &s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + '"';
}

int main(int argc, char **argv) {
  std::string facts_dir, programs, arch, out_dir, cdigest_hex,
      missing_path;
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--facts") facts_dir = next();
    else if (a == "--programs") programs = next();
    else if (a == "--arch") arch = next();
    else if (a == "--out") out_dir = next();
    else if (a == "--converter-digest") cdigest_hex = next();
    else if (a == "--missing-json") missing_path = next();
    else if (a == "--verbose") verbose = true;
    else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }
  if (facts_dir.empty() || programs.empty() || arch.empty() ||
      out_dir.empty()) {
    std::fprintf(stderr, "cyclescope-convert --facts DIR --programs F "
                         "--arch A --out DIR\n");
    return 2;
  }
  const auto *rules = convert_rules_for(arch);
  if (!rules) {
    std::fprintf(stderr, "unknown arch: %s\n", arch.c_str());
    return 2;
  }
  FactsFile prog(programs);
  BuildMeta meta;
  if (const auto *m = prog.table("META", nullptr)) {
    meta.tree_digest.assign(reinterpret_cast<const char *>(m), 32);
    meta.program_digest.assign(reinterpret_cast<const char *>(m + 32), 32);
  }
  meta.converter_digest = cdigest_hex.empty()
                              ? std::string(32, '\0')
                              : hex32(cdigest_hex);
  {
    // launch.json next to first .gsc
    std::string lj = facts_dir + "/launch.json";
    std::ifstream in(lj);
    std::string js((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
    auto sh = cluster_shape(js);
    {
      auto jnum = [&](const char *key) -> int {
        auto p = js.find(key);
        if (p == std::string::npos) return -1;
        p = js.find(':', p);
        p = js.find_first_of("-0123456789", p + 1);
        return p == std::string::npos ? -1 : std::stoi(js.substr(p));
      };
      auto bp = js.find("\"block\"");
      if (bp != std::string::npos) {
        auto q = js.find('[', bp);
        for (int i = 0; i < 3 && q != std::string::npos; ++i) {
          q = js.find_first_of("0123456789", q + 1);
          if (q == std::string::npos || js[q] == ']') break;
          meta.block[i] = std::stoi(js.substr(q));
          q = js.find_first_of(",]", q);
        }
      }
      meta.registers_per_thread = jnum("registers_per_thread");
      meta.declared_math_warpgroups = jnum("declared_math_warpgroups");
    }
  meta.cluster_shape[0] = sh[0];
    meta.cluster_shape[1] = sh[1];
    meta.cluster_shape[2] = sh[2];
  }
  std::vector<std::filesystem::path> gscs;
  for (const auto &de :
       std::filesystem::directory_iterator(facts_dir))
    if (de.path().extension() == ".gsc" &&
        de.path().filename() != "statics.gsc") gscs.push_back(de.path());
  std::sort(gscs.begin(), gscs.end());
  std::map<std::string, std::uint64_t> miss, unsup;
  TreeRt rt(prog);  // parsed once for all segments
  SegmentBuilder::StaticCache statics;   // shared statics.gsc rows
  for (const auto &gp : gscs) {
    FactsFile facts;
    try {
      facts.open(gp.string());
    } catch (std::exception &e) {
      std::fprintf(stderr, "facts %s: %s\n", gp.c_str(), e.what());
      return 1;
    }
    SegmentBuilder b(facts, prog, rules, meta, arch, rt, &statics);
    auto seg = b.build();
    char oname[1024];
    std::snprintf(oname, sizeof oname, "%s/%06d.seg", out_dir.c_str(),
                  facts.cluster_id());
    std::ofstream o(oname, std::ios::binary);
    o.write(reinterpret_cast<const char *>(seg.data()),
            (std::streamsize)seg.size());
    o.close();
    for (auto &[k, v] : b.missing_facts()) miss[k] += v;
    for (auto &[k, v] : b.unsupported()) unsup[k] += v;
    if (verbose)
      std::fprintf(stderr, "seg %s -> %s (%zu B)\n", gp.c_str(), oname,
                   seg.size());
  }
  {
    // segment v2: the seven shared program tables once per dir
    static const char *kProgTags[] = {"PROG", "NODE", "GRD ", "LEAF",
                                      "PARM", "PTS ", "AXIS"};
    static const std::uint32_t kProgRec[] = {32, 24, 24, 8, 48, 16, 4};
    const std::size_t npt = 7;
    std::vector<std::uint8_t> out;
    const std::size_t hdr = 72, dent = 32;
    std::uint64_t cursor = (hdr + npt * dent + 7) & ~7ull;
    out.resize(cursor);
    std::vector<std::uint8_t> dir(npt * dent);
    for (std::size_t i = 0; i < npt; ++i) {
      std::size_t n = 0;
      const auto *pl = prog.table(kProgTags[i], &n);
      if (!pl) { std::fprintf(stderr, "programs missing %s\n",
                              kProgTags[i]); return 1; }
      const auto rec = kProgRec[i];
      const std::uint64_t cnt = n / rec;
      std::memcpy(dir.data() + i * dent, kProgTags[i], 4);
      std::uint32_t rb = rec;
      std::memcpy(dir.data() + i * dent + 4, &rb, 4);
      std::uint64_t off = out.size();
      std::memcpy(dir.data() + i * dent + 8, &cnt, 8);
      std::memcpy(dir.data() + i * dent + 16, &off, 8);
      out.insert(out.end(), pl, pl + n);
      if (i + 1 < npt) out.resize((out.size() + 7) & ~7ull);
    }
    std::memcpy(out.data(), "GSIMPRG0", 8);
    std::uint32_t v32 = 1; std::memcpy(out.data() + 8, &v32, 4);
    v32 = 72; std::memcpy(out.data() + 12, &v32, 4);
    std::uint64_t v64 = out.size();
    std::memcpy(out.data() + 16, &v64, 8);
    v64 = hdr; std::memcpy(out.data() + 24, &v64, 8);
    v32 = static_cast<std::uint32_t>(npt);
    std::memcpy(out.data() + 32, &v32, 4);
    std::memcpy(out.data() + 72, dir.data(), dir.size());
    std::ofstream po(out_dir + "/program.tables", std::ios::binary);
    po.write(reinterpret_cast<const char *>(out.data()),
             (std::streamsize)out.size());
  }
  if (!missing_path.empty()) {
    std::ofstream o(missing_path);
    o << "{\"missing_facts\":{";
    bool first = true;
    for (auto &[k, v] : miss) {
      if (!first) o << ',';
      first = false;
      o << json_str(k) << ':' << v;
    }
    o << "},\"unsupported\":{";
    first = true;
    for (auto &[k, v] : unsup) {
      if (!first) o << ',';
      first = false;
      o << json_str(k) << ':' << v;
    }
    o << "}}\n";
  }
  return 0;
}
