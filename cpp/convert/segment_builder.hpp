// C++ 3a segment builder — GSIMCAP01 -> GSIMSEG tables
// (CTAS/WARP/TMPL/INST/REGS/MACC/MSEC/CODE/META/STRS + copied program
// tables). Byte-identical to python/cyclescope/job/segment_builder.py run
// with runtime=None (no mechanism uses / semantics yet).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "convert/facts_reader.hpp"
#include "convert/convert_rules.hpp"
#include "convert/semantics.hpp"
#include "convert/axes.hpp"
#include "convert/traversal.hpp"
#include <memory>

namespace cyclescope_convert {

struct BuildMeta {
  std::string tree_digest;      // 32 bytes
  std::string program_digest;   // 32 bytes
  std::string converter_digest; // 32 bytes
  int cluster_shape[3] = {1, 1, 1};
  int block[3] = {0, 0, 0};
  int registers_per_thread = -1;
  int declared_math_warpgroups = -1;
  std::uint32_t flags = 0;
};

class SegmentBuilder {
public:
  struct StaticRow {
    std::uint64_t pc, code_address;
    std::string opcode, memory_space;
    std::vector<std::string> operands;
    bool has_predicate;
    int read_barrier, write_barrier, wait_mask;
    int stall = 0, yield_hint = 0;
    std::uint32_t reuse_mask = 0;
    std::vector<std::uint32_t> src_regs, dst_regs;
    std::vector<std::string> src_names, dst_names;
    std::map<std::string, std::string> roles;   // reg -> "data"|"address"
    // static-only derived values (filled once by fill_static)
    std::string root;                  // opcode up to the first '.'
    bool uniform_root = false;         // ULDC / LDCU
    bool utma_root = false;            // UTMALDG/UTMASTG/UTMAREDG/UTMAPF
    std::string cls;                   // rules_->operation_class
    std::vector<std::string> classes;  // rules_->operation_classes
    SemOp sem_op = SemOp::none;        // semantic_operation_of
    std::string sem_error;             // its ConversionError, if any
    // first-reader prescan: sorted unique source / destination names
    std::vector<std::string> src_set, dst_set;
  };
  // Static-row state shared by every segment of one facts dir (the
  // statics.gsc tables are identical for all clusters under GSIMCAP02):
  // decoded rows, StaticInfo views, register ids, template key
  // prefixes and the static-axis memo.  One per cyclescope-convert run;
  // pre-v2 facts (per-cluster STAT tables) get a private one.
  struct StaticCache {
    std::vector<StaticRow> statics;
    std::vector<StaticInfo> infos;
    std::unordered_map<std::uint32_t, std::uint32_t> reg_ids;
    std::unordered_map<const StaticRow *, std::string> tmpl_key_prefix;
    std::unordered_map<std::string,
                       std::unordered_map<std::uint64_t, AxVal>>
        static_axis_cache;
  };

  SegmentBuilder(FactsFile &facts, FactsFile &programs,
                 const ConvertRules *rules, const BuildMeta &meta,
                 const std::string &arch, const TreeRt &rt,
                 StaticCache *shared = nullptr);
  std::vector<std::uint8_t> build();

  const std::unordered_map<std::string, std::uint64_t> &missing_facts()
      const { return missing_facts_; }
  const std::unordered_map<std::string, std::uint64_t> &unsupported()
      const { return unsupported_; }

private:
  FactsFile &f_;
  const ConvertRules *rules_;
  BuildMeta meta_;

  // output tables (byte buffers in emit order)
  std::unordered_map<std::uint32_t, std::vector<std::uint8_t>> tables_;
  std::vector<std::uint8_t> strings_;
  std::unordered_map<std::string, std::uint32_t> string_offsets_;

  StaticCache own_cache_;        // used when no shared cache applies
  StaticCache &cache_;
  std::vector<StaticRow> &statics_;
  // fill statics_[si] / statics_as_info_[si] from the STAT/STAX rows
  // on first use
  void fill_static(std::uint32_t si);

  std::unordered_map<std::string, std::uint32_t> templates_;
  std::uint64_t inst_count_ = 0, macc_count_ = 0, msec_count_ = 0;
  // segment GSYN rows are packed into tables_["GSYN"] during
  // emit_instance (<IBBBBQQQQ, 40 B/row)
  std::unordered_map<std::string, std::uint64_t> missing_facts_;
  // protocol v4 operand_read: pc -> static index (reuse-served pass)
  std::unordered_map<std::uint64_t, std::uint32_t> static_by_pc_;
  std::uint64_t regs_flag_fallbacks_ = 0;
  std::vector<std::uint8_t> src_flags(const StaticRow &s);
  void ensure_static(std::uint32_t si);
  std::unordered_map<std::string, std::uint64_t> unsupported_;
  const TreeRt *rt_;  // shared per program set (built once by caller)
  std::vector<StaticInfo> &statics_as_info_;
  AxisCtx axctx_;
  std::map<int, FlowCtx> flows_;
  std::uint64_t fact_count_ = 0, time_count_ = 0;
  // template key: hash of (cls,src,dst,pred,space,uses facts)
  std::unordered_map<std::string, std::uint32_t> tmpl_uses_;
  // template_index key prefix (cls..yield_hint) per static row
  std::unordered_map<const StaticRow *, std::string> &tmpl_key_prefix_;
  std::map<std::uint64_t, std::uint32_t> schm_target_;  // schema_id->code
  std::set<std::string> class_domain_;
  SemanticsContext sem_ctx_;
  std::string arch_name_;
  int arch_kind_ = 0;  // 1 = hopper, 2 = blackwell
  std::uint64_t sema_count_ = 0, mbol_count_ = 0;

  // facts table views
  struct {
    const std::uint8_t *warp = nullptr, *dyn = nullptr, *mref = nullptr,
                       *mrex = nullptr, *addr = nullptr, *stat = nullptr,
                       *stax = nullptr, *optk = nullptr, *regl = nullptr,
                       *ctas = nullptr, *ops = nullptr, *opsx = nullptr,
                       *ukv = nullptr, *lane = nullptr, *lval = nullptr,
                       *tmap = nullptr, *tmdi = nullptr,
                       *mbpi = nullptr, *gsyn = nullptr, *gsynx = nullptr;
    std::size_t warp_n = 0, dyn_n = 0, mref_n = 0, mrex_n = 0, stat_n = 0,
                ctas_n = 0, mbpi_n = 0;
  } t_;

  std::uint32_t string_offset(const std::string &s);
  std::unordered_map<std::uint32_t, std::uint32_t> &reg_ids_;
  std::uint32_t register_id_(std::uint32_t str_off);
  void emit_warp(std::uint64_t wrow);
  void emit_instance(std::uint64_t drow, int cta, int wslot);
  std::pair<std::uint32_t, bool> template_index(
      const StaticRow &s, const std::string &cls,
      const std::vector<UseRow> *uses = nullptr);
  void emit_access(const StaticRow &s, std::uint64_t mref_idx);
  std::uint32_t emit_semantics(const StaticRow &s, const StaticInfo &info,
                               const DynOps &dyn, int cta, int wslot);
  void build_dynops(std::uint64_t di, DynOps &dyn);
  std::pair<bool, std::uint64_t> cur_reader_{false, 0};
  std::string cur_reader_role_;
  bool cur_reader_role_none_ = false;
  void _patch_tmpl_timing(std::uint32_t tmpl, std::uint32_t tb,
                          std::uint32_t tc);
  std::pair<std::uint32_t, std::uint32_t>
  emit_uses_rows(const std::vector<UseRow> &rows,
                 std::uint32_t template_idx, bool instance_only,
                 const std::string &opcode);
  void sector(std::uint64_t key, std::uint32_t bmask,
              std::uint32_t lmask);
  void emit_plain(std::uint64_t ab, std::uint64_t ac,
                  std::uint32_t width);
  void emit_store(std::uint64_t mref_idx, std::uint64_t ab,
                  std::uint64_t ac, std::uint32_t lane_mask,
                  std::uint32_t width);
  void emit_bulk(std::uint64_t ab, std::uint64_t ac, std::uint32_t width);

  template <typename... A>
  void pack(std::uint32_t tag, const char *fmt, A... args);
};

const ConvertRules *convert_rules_for(const std::string &arch);

}  // namespace cyclescope_convert
