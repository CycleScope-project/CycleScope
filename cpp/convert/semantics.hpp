// Semantics decode — port of convert/semantics_common.py +
// job/semantics.py (kinds, field slots, pack_fields).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "convert/convert_rules.hpp"

namespace cyclescope_convert {

// SemanticOperation (convert/semantics_common.py IntEnum)
enum class SemOp : int {
  none = 0,
  mbarrier_init = 1, mbarrier_arrive = 2, mbarrier_expect_tx = 3,
  mbarrier_wait = 4, mbarrier_inval = 5,
  cta_barrier_arrive = 6, cta_barrier_wait = 7,
  cluster_barrier_arrive = 8, cluster_barrier_wait = 9,
  tma_load = 10, bulk_group_issue = 11, bulk_group_commit = 12,
  bulk_group_wait = 13, async_copy_issue = 14, async_copy_commit = 15,
  async_copy_wait = 16, bulk_copy = 17, bulk_reduce = 18,
  shared_store_async = 19,
  descriptor_visibility_publish = 20,
  descriptor_visibility_store = 21,
  descriptor_visibility_error_check = 22,
  descriptor_visibility_acquire = 23,
  descriptor_visibility_proxy_fence = 24,
  nanosleep = 25, depbar_wait = 26, dependents_launch = 27,
  warpgroup_arrive = 32, warpgroup_wait = 33, wgmma = 34,
  umma_issue = 35, umma_commit = 36, tmem_alloc = 37, tmem_dealloc = 38,
  tmem_relinquish = 39, tmem_load = 40, tmem_store = 41, tmem_copy = 42,
  tmem_wait_store = 43, clc_query = 44,
};

// SemanticsKind (job/semantics.py)
enum class SemKind : int {
  none = 0,
  mbarrier_init = 1, mbarrier_arrive = 2, mbarrier_expect_tx = 3,
  mbarrier_wait = 4, mbarrier_inval = 5,
  cta_barrier_arrive = 6, cta_barrier_wait = 7,
  cluster_barrier_arrive = 8, cluster_barrier_wait = 9,
  tma_load = 10, bulk_group_issue = 11, bulk_group_commit = 12,
  bulk_group_wait = 13, async_copy_issue = 14, async_copy_commit = 15,
  async_copy_wait = 16, bulk_copy = 17, bulk_reduce = 18,
  warpgroup_arrive = 19, warpgroup_wait = 20, wgmma = 22,
  umma_issue = 23, umma_commit = 24, tmem_alloc = 25, tmem_dealloc = 26,
  tmem_relinquish = 27, tmem_load = 28, tmem_store = 29, tmem_copy = 30,
  tmem_wait_store = 31, clc_query = 32, shared_store_async = 33,
  global_sync = 34, reconvergence_start = 35, reconvergence_complete = 36,
  nanosleep = 37, descriptor_visibility = 38, clock_marker = 39,
  dependents_launch = 40,
};

constexpr std::uint64_t GEN_UNRESOLVED = 1ull << 63;
constexpr std::uint64_t SEMA_NO_OBJECT = 0xFFFFFFFFull;
constexpr std::uint64_t SEMA_UNKNOWN = 0xFFFFFFFFull;
constexpr int SCOPE_CTA = 0, SCOPE_CLUSTER = 1;
constexpr int SEMA_FLAG_AFTER_READY = 0x1, SEMA_FLAG_ALSO_ARRIVES = 0x2,
              SEMA_FLAG_IMPLICIT_COMMIT = 0x4,
              SEMA_FLAG_GROUP_OPENER = 0x8, SEMA_FLAG_GROUP_COMMIT = 0x10,
              SEMA_FLAG_STAT = 0x20, SEMA_FLAG_SYNCS = 0x40,
              SEMA_FLAG_CTA_PAIR = 0x80,
              SEMA_FLAG_COMPLETE_TX = 0x100,
              SEMA_FLAG_WAIT_ORIGIN = 0x200;
constexpr int DV_PUBLISH = 1, DV_STORE = 2, DV_ERROR_CHECK = 3,
              DV_ACQUIRE = 4, DV_PROXY_FENCE = 5;

struct ConversionError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SemRow {
  SemKind kind = SemKind::none;
  int flags = 0;
  std::map<std::string, std::uint64_t> fields;
  std::vector<std::uint32_t> object_list;   // empty = none
  std::vector<std::array<std::uint32_t, 3>> split;
  std::uint64_t earliest_issue = 0;
};

// dynamic-operand view (what decoders read)
struct DynOps {
  // GSYN observation (kinds 1..4), -1 when absent
  int gsyn_kind = 0, gsyn_lane = 0, gsyn_flags = 0;
  std::uint64_t gsyn_addr = 0, gsyn_after = 0, gsyn_before = 0,
                gsyn_seq = 0;
  bool has_gsyn = false;
  std::uint32_t active_mask = 0, predicate_mask = 0;
  std::map<std::string, std::uint64_t> uniform;
  std::map<std::string, std::string> uniform_str;
  std::map<int, std::vector<std::uint32_t>> ranks;
  std::uint32_t after_ready_mask = 0;
  bool has_after_ready = false;
  std::uint32_t tensor_map = 0;
  bool has_tensor_map = false;
  // memory refs: (space_code, access_kind_code, width, lane_mask,
  //   remote_rank, addr_begin, addr_count) — addresses resolved via
  //   the ADDR table pointer
  struct Ref {
    int space = 4, access_kind = 0;
    std::uint32_t width = 0, lane_mask = 0;
    long long remote_rank = -1, destination_count = -1,
              transfer_bytes = -1, cut_rows = -1;
    std::uint64_t multicast_mask = 0;
    std::vector<std::uint64_t> addrs;
  };
  std::vector<Ref> memory;
  std::map<std::string, std::vector<std::uint64_t>> lane_values;
};

// unit-level state
struct SemanticsContext {
  int n_ctas = 0;
  // tensor_maps: rank -> (element_bytes, box_dims)
  std::map<int, std::pair<std::uint32_t, std::vector<std::uint64_t>>>
      tensor_maps;
  std::map<std::pair<int, std::uint64_t>, std::uint32_t> mbarrier_objects;
  // cta_linear -> cluster_rank (remote DSMEM object resolution)
  std::map<int, int> cta_rank;
  std::map<int, int> cta_warps;   // cta_linear -> warp count (BAR default)
  // arch hook: strip DSMEM rank tag from a shared address
  std::uint64_t (*local_shared)(std::uint64_t) = nullptr;
  std::map<int, int> ldgsts_group, bulk_group;
  std::set<int> saw_lgdepbar, saw_utmaflush;
  std::map<int, int> bulk_open_issues;
  std::map<int, std::vector<std::pair<std::uint64_t, int>>> desc_stores;
  struct DescPending {
    std::vector<std::uint64_t> bases;
    std::set<std::string> checks;
    int acquired = 0;
  };
  std::map<int, DescPending> desc_pending;
  // arch extras
  std::map<int, int> wg_groups, wg_counts, wg_commits;
  // blackwell STS.64-as-mbarrier-init: shared addresses referenced by
  // explicit barrier instructions (segment_builder prescan)
  std::set<std::uint64_t> mbarrier_addresses;

  std::uint32_t mbarrier(int cta, std::uint64_t addr,
                         const std::string &where);
  int next_group(std::map<int, int> &t, int wslot);
  int current_group(std::map<int, int> &t, int wslot);
};

// decoders
// mbarrier object for (issuing cta, word): strips the DSMEM rank tag and
// honours the owner_rank fact (peer CTA's object)
std::uint32_t resolve_mbarrier(const DynOps &d, SemanticsContext &c,
                               int cta, std::uint64_t addr,
                               const char *where);

SemRow decode_semantics(const StaticInfo &st, const DynOps &dyn,
                        SemanticsContext &ctx, SemOp op, int cta,
                        int wslot, int arch_kind);

// arch semantic_operation lookup (returns null SemOp::none when no op;
// throws ConversionError "unsupported:semantic_modifier" for supported
// roots without a mapped modifier)
// arch_kind: 1 = hopper, 2 = blackwell (0 = unknown)
SemOp semantic_operation_of(int arch_kind,
                            const StaticInfo &st);
std::string where_str(const StaticInfo &st);

// STS / STS.64 onto a barrier word named by explicit barrier ops
// (SemanticsContext::mbarrier_addresses prescan) -> mbarrier_init; a hit
// without a decoded init value throws missing_fact:mbarrier_init_value.
SemOp promote_store_to_mbarrier_init(const StaticInfo &st, const DynOps &d,
                                     const SemanticsContext &ctx, SemOp op);

// GSYN class by opcode root: 1 atom / 2 red / 3 ld_strong / 4 st_strong,
// 0 unknown (python semantics_common.global_sync_class)
int global_sync_class(const std::string &opcode);

// kind-34 row for an instruction carrying a GSYN observation (classes
// 1..3; st_strong is a producer only): the engine's acquire wait and
// atomic-predecessor gate (docs/en/design/engine.md) are keyed on it.  Kind none
// when the instruction carries no row.
SemRow global_sync_row(const StaticInfo &st, const DynOps &d);

// pack_fields — SEMANTICS_FIELDS slots
std::array<std::uint64_t, 6> pack_fields(SemKind kind,
    const std::map<std::string, std::uint64_t> &fields,
    std::uint32_t mbol_begin, std::uint32_t mbol_count);

}  // namespace cyclescope_convert
