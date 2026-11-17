// Container/segment reader producing the merged in-memory Job (spec §3).

#include "job_reader.hpp"

#include <openssl/sha.h>

#include <fstream>
#include <unordered_map>

namespace cyclescope {

// --- BinaryFile -----------------------------------------------------------

BinaryFile::BinaryFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw FormatError("cannot open " + path);
  const std::streamoff n = in.tellg();
  std::string data(static_cast<std::size_t>(n), '\0');
  in.seekg(0);
  if (n > 0 && !in.read(data.data(), n))
    throw FormatError("short read " + path);
  init(std::move(data));
}

BinaryFile::BinaryFile(std::string data, InMemory) {
  init(std::move(data));
}

BinaryFile::BinaryFile(std::string_view data, View) { init_view(data); }

std::uint8_t BinaryFile::peek_kind(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw FormatError("cannot open " + path);
  char magic[8] = {};
  if (!in.read(magic, 8)) throw FormatError("short file");
  static const char kMagics[][8] = {"GSIMSEG", "GSIMJOB", "GSIMRES",
                                   "GSIMJNL", "GSIMWLG", "GSIMPRG"};
  std::uint8_t k = 0;
  for (; k < 6; ++k)
    if (std::memcmp(magic, kMagics[k], 8) == 0) break;
  if (k == 6) throw FormatError("bad magic");
  return k;
}

void BinaryFile::init(std::string data) {
  data_ = std::move(data);
  init_view(data_);
}

void BinaryFile::init_view(std::string_view data) {
  view_data_ = data;
  if (data.size() < kHeaderBytes) throw FormatError("short file");
  FileHeader h;
  std::memcpy(&h, data.data(), sizeof h);
  static const char kMagics[][8] = {"GSIMSEG", "GSIMJOB", "GSIMRES",
                                   "GSIMJNL", "GSIMWLG", "GSIMPRG"};
  for (kind_ = 0; kind_ < 6; ++kind_)
    if (std::memcmp(h.magic, kMagics[kind_], 8) == 0) break;
  if (kind_ == 6) throw FormatError("bad magic");
  version_ = h.format_version;
  if (version_ < 1 || version_ > kSplitProgramVersion)
    throw FormatError("bad format_version");
  if (h.file_bytes != data.size()) throw FormatError("file_bytes mismatch");
  entries_.resize(h.table_count);
  std::memcpy(entries_.data(), data.data() + h.directory_offset,
              h.table_count * kDirectoryEntryBytes);
  std::unordered_map<std::uint32_t, bool> seen;
  for (const auto& e : entries_) {
    if (seen[e.tag]) throw FormatError("duplicate table tag");
    seen[e.tag] = true;
    std::uint64_t len =
        e.record_bytes ? e.record_bytes * e.record_count : e.record_count;
    if (e.offset + len > data.size())
      throw FormatError("table overruns file");
  }
}

const Table* BinaryFile::table(std::uint32_t t) const {
  view_.reset();
  for (const auto& e : entries_)
    if (e.tag == t) {
      view_.emplace(Table{e.record_bytes, e.record_count, e.offset});
      return &*view_;
    }
  return nullptr;
}

const Table& BinaryFile::require(std::uint32_t t) const {
  const Table* p = table(t);
  if (!p) throw FormatError("required table missing");
  return *p;
}

// --- table helpers ---------------------------------------------------------

namespace {

template <typename R>
std::vector<R> take(const BinaryFile& f, std::uint32_t t,
                    const BinaryFile* prog = nullptr) {
  if (prog && !f.table(t)) return prog->rows<R>(t);
  const Table& tb = f.require(t);
  if (tb.record_bytes != sizeof(R))
    throw FormatError("record size mismatch");
  std::vector<R> out(tb.record_count);
  if (tb.record_count)
    std::memcpy(out.data(), f.data().data() + tb.offset,
                tb.record_count * sizeof(R));
  return out;
}

}  // namespace

// --- segment load ----------------------------------------------------------



Job read_segment_file_data(const BinaryFile& f,
                           const BinaryFile* prog,
                           bool program_tables) {
  if (f.kind() != 0) throw FormatError("not a segment");
  Job job;
  job.meta = take<MetaRow>(f, tag("META"), prog)[0];
  job.ctas = take<CtaInfo>(f, tag("CTAS"), prog);
  job.warps = take<WarpInfo>(f, tag("WARP"), prog);
  job.templates = take<InstructionTemplate>(f, tag("TMPL"), prog);
  job.instances = take<InstructionInstance>(f, tag("INST"), prog);
  job.register_refs = take<RegisterRef>(f, tag("REGS"), prog);
  job.semantics = take<InstanceSemantics>(f, tag("SEMA"), prog);
  job.accesses = take<MemoryAccess>(f, tag("MACC"), prog);
  job.sectors = take<MemorySector>(f, tag("MSEC"), prog);
  job.timing = take<TimingRow>(f, tag("TIME"), prog);
  // shared program tables: only the first segment's copy is kept by
  // the container merge, later segments skip the copies
  if (program_tables) {
    job.programs = take<MechanismProgram>(f, tag("PROG"), prog);
    job.nodes = take<MechanismNode>(f, tag("NODE"), prog);
    job.guards = take<MechanismGuard>(f, tag("GRD "), prog);
    job.leaves = take<MechanismLeaf>(f, tag("LEAF"), prog);
    job.fields = take<ParameterField>(f, tag("PARM"), prog);
    job.points = take<PointEntry>(f, tag("PTS "), prog);
  }
  job.facts = take<FactRow>(f, tag("FACT"), prog);
  job.uses = take<ParameterUse>(f, tag("USES"), prog);
  job.splits = take<SplitArrive>(f, tag("SPLT"), prog);
  job.mbarrier_preinit = take<MbarrierPreinit>(f, tag("MBPI"), prog);
  job.mbarrier_objects = take<MbarrierObject>(f, tag("MBOL"), prog);
  if (f.table(tag("MBOB")))
    job.mbarrier_owners = take<MbarrierObjectRow>(f, tag("MBOB"), prog);
  job.code = take<CodeRow>(f, tag("CODE"), prog);
  job.clock_markers = take<ClockMarker>(f, tag("CLKM"), prog);
  job.strings = f.blob(tag("STRS"));
  if (program_tables &&
      (f.table(tag("AXIS")) || (prog && prog->table(tag("AXIS")))))
    job.axis_names = take<AxisRow>(f, tag("AXIS"), prog);
  // single-segment defaults; merge_parts rebuilds these for containers
  job.cta_segment.assign(job.ctas.size(), 0);
  job.segment_mbol_base.assign(1, 0);
  job.segment_instance_base.assign(1, 0);
  job.segment_mbob_base.assign(1, 0);
  return job;
}

Job read_segment_file(const std::string& path) {
  BinaryFile f(path);
  if (f.format_version() >= kSplitProgramVersion) {
    const auto slash = path.find_last_of('/');
    const std::string pp = (slash == std::string::npos)
        ? "program.tables" : path.substr(0, slash) + "/program.tables";
    BinaryFile prog(pp);
    return read_segment_file_data(f, &prog);
  }
  return read_segment_file_data(f, nullptr);
}

// --- merge (spec §3) -------------------------------------------------------

// incremental container merge state: META digest reference + per-kind
// bases + shared prefix extents + the CTA linear->dense map
struct MergeState {
  MetaRow meta0;
  std::size_t strs0 = 0, facts0 = 0;
  std::size_t inst = 0, tmpl = 0, sema = 0, macc = 0, msec = 0, regs = 0,
      tim = 0, warp = 0, cta = 0, mbol = 0, mbob = 0;
  std::size_t s = 0;
  std::unordered_map<std::uint32_t, std::uint32_t> linear_to_dense;
};

// first segment: META digest reference + shared tables move into job
void merge_begin(Job& job, MergeState& st, Job&& first) {
  st.meta0 = first.meta;
  job.meta = first.meta;
  job.programs = std::move(first.programs);
  job.nodes = std::move(first.nodes);
  job.guards = std::move(first.guards);
  job.leaves = std::move(first.leaves);
  job.fields = std::move(first.fields);
  job.points = std::move(first.points);
  job.facts = std::move(first.facts);
  job.strings = std::move(first.strings);
  job.axis_names = std::move(first.axis_names);
  st.strs0 = job.strings.size();
  st.facts0 = job.facts.size();
  job.ctas.clear();
  job.cta_segment.clear();
  job.segment_mbol_base.clear();
  job.segment_instance_base.clear();
  job.segment_mbob_base.clear();
  job.segment_instance_base.reserve(1024);
  job.segment_mbol_base.reserve(1024);
  job.segment_mbob_base.reserve(1024);
}

void merge_append(Job& job, MergeState& st, Job&& p) {
  const std::size_t s = st.s;
  std::size_t &inst=st.inst,&tmpl=st.tmpl,&sema=st.sema,&macc=st.macc,
      &msec=st.msec,&regs=st.regs,&tim=st.tim,&warp=st.warp,&cta=st.cta,
      &mbol=st.mbol,&mbob=st.mbob;
  const std::size_t strs0=st.strs0, facts0=st.facts0;
  auto& linear_to_dense = st.linear_to_dense;
  {
    job.segment_instance_base.push_back(inst);
    job.segment_mbol_base.push_back(mbol);
    job.segment_mbob_base.push_back(mbob);
    job.cta_segment.insert(job.cta_segment.end(), p.ctas.size(),
                           static_cast<std::uint32_t>(s));
    std::unordered_map<std::uint32_t, std::uint32_t> local;
    for (std::size_t j = 0; j < p.ctas.size(); ++j) {
      auto& c = p.ctas[j];
      const auto dense = static_cast<std::uint32_t>(cta + j);
      if (!local.emplace(c.cta_index, dense).second)
        throw FormatError("duplicate CTA id in a segment");
      linear_to_dense.emplace(c.cta_index, dense);  // first wins

      c.cta_index = dense;
      c.first_warp += warp;
    }
    const auto dense_of = [&](std::uint32_t id, const char* what) {
      auto it = local.find(id);
      if (it == local.end())
        throw FormatError(std::string(what) + " names a CTA outside its segment");
      return it->second;
    };
    for (auto& w : p.warps) {
      w.first_instance += inst;
      w.cta_index = dense_of(w.cta_index, "WARP");
    }
    for (auto& i : p.instances) {
      if (i.template_index != kNone32) i.template_index += tmpl;
      if (i.semantics != kNone32) i.semantics += sema;
      if (i.memory_access != kNone32) i.memory_access += macc;
    }
    for (auto& t : p.templates) {
      t.source_begin += regs;
      t.destination_begin += regs;
      t.timing_begin += tim;
    }
    for (auto& a : p.accesses) {
      a.sector_begin += msec;
      a.timing_begin += tim;
    }
    for (auto& sp : p.splits)
      if (sp.semantics != kNone32) sp.semantics += sema;
    for (auto& mb : p.mbarrier_preinit)
      mb.cta_index = dense_of(mb.cta_index, "MBPI");
    for (auto& ob : p.mbarrier_owners)
      ob.cta_index = dense_of(ob.cta_index, "MBOB");
    for (auto& cd : p.code)
      if (cd.instance != kNone32) cd.instance += inst;
    for (auto& cm : p.clock_markers) {
      cm.warp += warp;
      cm.instance += inst;
    }
    // STRS / FACT: every segment starts with the shared program-table
    // prefix (identical bytes / rows) and appends its own strings and
    // per-instance FACT rows.  Segment 0 is taken whole; for later
    // segments the common prefix is reused and the per-segment tail is
    // appended with the referring offsets rebased (TMPL.operation_class,
    // FACT.axis, USES.fact_begin).
    std::size_t k_strs = 0, k_fact = 0, strs_base = 0, fact_base = 0;
    if (s > 0) {
      // segment 0's tables were moved into job.*; their original extents
      // are strs0 / facts0
      const std::size_t lim = std::min(strs0, p.strings.size());
      while (k_strs < lim && job.strings[k_strs] == p.strings[k_strs])
        ++k_strs;
      while (k_strs > 0 && p.strings[k_strs - 1] != '\0') --k_strs;
      strs_base = job.strings.size();
      job.strings.append(p.strings, k_strs, std::string::npos);
      k_fact = p.facts.size();
      for (const auto& u : p.uses)
        if (u.fact_count > 0 && (u.target == 0 || u.target == 1 ||
                                 u.target == 2 || u.target == 3 ||
                                 u.target == 8))
          k_fact = std::min<std::size_t>(k_fact, u.fact_begin);
      if (k_fact > facts0)
        throw FormatError("segment FACT prefix longer than segment 0");
      static_assert(std::is_trivially_copyable_v<FactRow>);
      if (k_fact &&
          std::memcmp(p.facts.data(), job.facts.data(),
                      k_fact * sizeof(FactRow)))
        throw FormatError("segment FACT prefix differs from segment 0");
      fact_base = job.facts.size();
    }
    auto remap_str = [&](std::uint32_t off) -> std::uint32_t {
      if (s == 0 || off < k_strs) return off;
      return static_cast<std::uint32_t>(off - k_strs + strs_base);
    };
    if (s > 0) {
      for (auto& t : p.templates) t.operation_class = remap_str(t.operation_class);
      for (std::size_t i = k_fact; i < p.facts.size(); ++i) {
        FactRow fr = p.facts[i];
        fr.axis = remap_str(fr.axis);
        job.facts.push_back(fr);
      }
    }
    for (auto& u : p.uses) {
      // per-instance targets: template/timing/instance_timing/memory_access/
      // instance get target_index rebased; global targets keep segment 0
      std::uint64_t add = 0;
      switch (u.target) {
        case 0: add = tmpl; break;
        case 1: add = tim; break;
        case 2: add = inst; break;  // InstanceTiming targets instances
        case 3: add = macc; break;
        case 8: add = inst; break;
        default:
          if (s > 0) continue;  // global target from segment 0 only
      }
      u.target_index += add;
      if (s > 0) {
        if (u.fact_count > 0) {
          if (u.fact_begin >= k_fact)
            u.fact_begin += fact_base - k_fact;
          else if (u.fact_begin + u.fact_count > k_fact)
            throw FormatError("USES fact range straddles the FACT prefix");
        }
        job.uses.push_back(u);
      }
    }
    if (s == 0) job.uses = std::move(p.uses);
    job.warps.insert(job.warps.end(), p.warps.begin(), p.warps.end());
    job.ctas.insert(job.ctas.end(), p.ctas.begin(), p.ctas.end());
    job.templates.insert(job.templates.end(), p.templates.begin(),
                         p.templates.end());
    job.instances.insert(job.instances.end(), p.instances.begin(),
                         p.instances.end());
    job.register_refs.insert(job.register_refs.end(),
                             p.register_refs.begin(), p.register_refs.end());
    job.semantics.insert(job.semantics.end(), p.semantics.begin(),
                         p.semantics.end());
    job.accesses.insert(job.accesses.end(), p.accesses.begin(),
                        p.accesses.end());
    job.sectors.insert(job.sectors.end(), p.sectors.begin(),
                       p.sectors.end());
    job.timing.insert(job.timing.end(), p.timing.begin(), p.timing.end());
    job.splits.insert(job.splits.end(), p.splits.begin(), p.splits.end());
    job.mbarrier_preinit.insert(job.mbarrier_preinit.end(),
                                p.mbarrier_preinit.begin(),
                                p.mbarrier_preinit.end());
    job.mbarrier_objects.insert(job.mbarrier_objects.end(),
                                p.mbarrier_objects.begin(),
                                p.mbarrier_objects.end());
    job.mbarrier_owners.insert(job.mbarrier_owners.end(),
                               p.mbarrier_owners.begin(),
                               p.mbarrier_owners.end());
    job.code.insert(job.code.end(), p.code.begin(), p.code.end());
    job.clock_markers.insert(job.clock_markers.end(),
                             p.clock_markers.begin(), p.clock_markers.end());
    inst += p.instances.size();
    tmpl += p.templates.size();
    sema += p.semantics.size();
    macc += p.accesses.size();
    msec += p.sectors.size();
    regs += p.register_refs.size();
    tim += p.timing.size();
    warp += p.warps.size();
    cta += p.ctas.size();
    mbol += p.mbarrier_objects.size();
    mbob += p.mbarrier_owners.size();
  }
  ++st.s;
}

void merge_finish(Job& job, MergeState& st) {
  // GSYN instance is segment-local; rebase to the merged instance
  // index space (pairing happens in the engine, §13.10)
  for (auto& g : job.global_sync)
    if (g.segment < job.segment_instance_base.size())
      g.instance += job.segment_instance_base[g.segment];
  // PLCE rows name CTAs by launch-wide linear id; keep those present in
  // the job (partial launches carry rows for absent CTAs) as dense ids
  std::vector<PlacementRow> kept;
  kept.reserve(job.placements.size());
  for (auto p : job.placements) {
    auto it = st.linear_to_dense.find(p.cta);
    if (it == st.linear_to_dense.end()) continue;
    p.cta = it->second;
    kept.push_back(p);
  }
  job.placements = std::move(kept);
}

void merge_parts(Job& job, std::vector<Job> parts) {
  if (parts.empty()) throw FormatError("container has no segments");
  MergeState st;
  for (std::size_t i = 1; i < parts.size(); ++i) {
    const MetaRow& m = parts[i].meta, &m0 = parts[0].meta;
    if (std::memcmp(m.tree_digest, m0.tree_digest, 32) ||
        std::memcmp(m.program_digest, m0.program_digest, 32) ||
        std::memcmp(m.converter_digest, m0.converter_digest, 32))
      throw FormatError("segment META digests differ");
  }
  merge_begin(job, st, std::move(parts[0]));
  for (auto& p : parts) merge_append(job, st, std::move(p));
  merge_finish(job, st);
}

Job read_container_file(const std::string& path) {
  BinaryFile f(path);
  if (f.kind() != 1) throw FormatError("not a container");
  Job job;
  job.launch = take<LaunchInfo>(f, tag("LNCH"))[0];
  auto segd = take<SegmentDirectoryRow>(f, tag("SEGD"));
  job.placements = take<PlacementRow>(f, tag("PLCE"));
  job.global_sync = take<GlobalSyncRow>(f, tag("GSYN"));
  job.topology = take<SmTopologyRow>(f, tag("TOPO"));

  MergeState st;
  st.meta0.tree_digest[0] = 1;  // unset sentinel: first segment seeds it
  bool begun = false;
  // directory pre-pass: size the merged tables once instead of growing
  // them segment by segment (many-cluster jobs: thousands of appends)
  {
    auto total = [&](const char* t4) {
      std::uint64_t n = 0;
      for (const auto& row : segd) {
        if (row.byte_offset + row.byte_length > f.data().size()) continue;
        try {
          BinaryFile img(f.data().substr(row.byte_offset, row.byte_length),
                         BinaryFile::View{});
          if (const Table* t = img.table(tag(t4))) n += t->record_count;
        } catch (const FormatError&) {
          // malformed image: the main loop reports it as before
        }
      }
      return n;
    };
    job.instances.reserve(total("INST"));
    job.templates.reserve(total("TMPL"));
    job.register_refs.reserve(total("REGS"));
    job.semantics.reserve(total("SEMA"));
    job.accesses.reserve(total("MACC"));
    job.sectors.reserve(total("MSEC"));
    job.timing.reserve(total("TIME"));
    job.uses.reserve(total("USES"));
    job.facts.reserve(total("FACT"));
    job.warps.reserve(total("WARP"));
    job.ctas.reserve(total("CTAS"));
    job.code.reserve(total("CODE"));
  }
  for (const auto& row : segd) {
    if (row.byte_offset + row.byte_length > f.data().size())
      throw FormatError("segment image overruns file");
    std::uint8_t digest[32];
    SHA256(reinterpret_cast<const unsigned char*>(
               f.data().data() + row.byte_offset),
           row.byte_length, digest);
    if (std::memcmp(digest, row.segment_sha256, 32))
      throw FormatError("segment sha256 mismatch");
    BinaryFile img(f.data().substr(row.byte_offset, row.byte_length),
                   BinaryFile::View{});
    const BinaryFile* prog =
        (f.format_version() >= kSplitProgramVersion ||
         img.format_version() >= kSplitProgramVersion)
            ? &f : nullptr;
    Job p = read_segment_file_data(img, prog, !begun);
    if (!begun) {
      merge_begin(job, st, Job{});  // placeholder tables; real move below
      // merge shared tables from first segment
      st.meta0 = p.meta;
      job.meta = p.meta;
      job.programs = std::move(p.programs);
      job.nodes = std::move(p.nodes);
      job.guards = std::move(p.guards);
      job.leaves = std::move(p.leaves);
      job.fields = std::move(p.fields);
      job.points = std::move(p.points);
      job.facts = std::move(p.facts);
      job.strings = std::move(p.strings);
      job.axis_names = std::move(p.axis_names);
      st.strs0 = job.strings.size();
      st.facts0 = job.facts.size();
      begun = true;
    } else {
      const MetaRow& m = p.meta, &m0 = st.meta0;
      if (std::memcmp(m.tree_digest, m0.tree_digest, 32) ||
          std::memcmp(m.program_digest, m0.program_digest, 32) ||
          std::memcmp(m.converter_digest, m0.converter_digest, 32))
        throw FormatError("segment META digests differ");
    }
    merge_append(job, st, std::move(p));
  }
  if (!begun) throw FormatError("container has no segments");
  merge_finish(job, st);
  return job;
}

// --- structural validation --------------------------------------------------

std::vector<std::string> validate_job(const Job& job) {
  std::vector<std::string> errors;
  auto bad = [&errors](std::string m) { errors.push_back(std::move(m)); };
  const std::size_t ni = job.instances.size();
  // warps tile the instance table
  std::uint64_t covered = 0;
  for (const auto& w : job.warps) {
    if (w.first_instance + w.instance_count > ni)
      bad("warp instance range out of bounds");
    covered += w.instance_count;
    if (w.cta_index >= job.ctas.size()) bad("warp cta_index out of range");
    if (w.smsp >= 4) bad("warp smsp out of range");
  }
  if (covered != ni) bad("warp instance ranges do not tile INST");
  for (const auto& i : job.instances) {
    if (i.template_index != kNone32 && i.template_index >= job.templates.size())
      bad("instance template index out of range");
    if (i.semantics != kNone32 && i.semantics >= job.semantics.size())
      bad("instance semantics index out of range");
    if (i.memory_access != kNone32 &&
        i.memory_access >= job.accesses.size())
      bad("instance memory_access index out of range");
  }
  for (const auto& g : job.global_sync) {
    if (g.segment >= job.segment_instance_base.size())
      bad("global_sync.segment out of range");
    else if (g.instance >= job.instances.size())
      bad("global_sync.instance out of range");
    if (g.class_ < 1 || g.class_ > 4)
      bad("global_sync.class out of range");
  }
  for (const auto& t : job.templates) {
    if (t.source_begin + t.source_count > job.register_refs.size())
      bad("template source refs out of range");
    if (t.destination_begin + t.destination_count >
        job.register_refs.size())
      bad("template destination refs out of range");
    if (t.timing_begin + t.timing_count > job.timing.size())
      bad("template timing range out of range");
  }
  for (const auto& a : job.accesses)
    if (a.sector_begin + a.sector_count > job.sectors.size())
      bad("access sector range out of range");
  for (const auto& n : job.nodes) {
    if (n.child_begin + n.child_count > job.nodes.size())
      bad("node children out of range");
    if (n.guard_begin + n.guard_count > job.guards.size())
      bad("node guard range out of range");
    if (n.leaf != kNone32 && n.leaf >= job.leaves.size())
      bad("node leaf index out of range");
  }
  for (const auto& p : job.programs)
    if (p.root_node >= job.nodes.size())
      bad("program root_node out of range");
  for (const auto& l : job.leaves)
    if (l.parameter >= job.fields.size() && job.fields.size())
      bad("leaf parameter index out of range");
  for (const auto& g : job.guards)
    if (g.axis >= job.axis_names.size() && job.axis_names.size())
      bad("guard axis index out of range");
  for (const auto& fld : job.fields)
    if (fld.axis != kNone32 && fld.axis >= job.axis_names.size() &&
        job.axis_names.size())
      bad("parm axis index out of range");
  for (const auto& u : job.uses) {
    if (u.node >= job.nodes.size()) bad("uses node index out of range");
    if (u.fact_begin + u.fact_count > job.facts.size())
      bad("uses fact range out of range");
  }
  for (const auto& ob : job.mbarrier_owners)
    if (ob.cta_index >= job.ctas.size())
      bad("mbarrier owner cta_index out of range");
  // placements consistent with CTAS sm_id when present
  if (!job.placements.empty())
    for (const auto& p : job.placements)
      if (p.placement == 1 &&
          p.cta < job.ctas.size() &&
          job.ctas[p.cta].sm_id != p.sm_id)
        bad("placement sm_id inconsistent with CTAS");
  return errors;
}

}  // namespace cyclescope
