#pragma once
// Instance semantics kinds — mirror of python/cyclescope/job/semantics.py
// (docs/en/formats.md §5).  Keep the numeric ids in lockstep; this header
// is the core-side copy (cpp/convert has its own writer side).

#include <cstdint>

namespace cyclescope {

enum SemanticsKind : std::uint16_t {
  kSemaNone = 0,
  kSemaMbarrierInit = 1,
  kSemaMbarrierArrive = 2,
  kSemaMbarrierExpectTx = 3,
  kSemaMbarrierWait = 4,
  kSemaMbarrierInval = 5,
  kSemaCtaBarrierArrive = 6,
  kSemaCtaBarrierWait = 7,
  kSemaClusterBarrierArrive = 8,
  kSemaClusterBarrierWait = 9,
  kSemaTmaLoad = 10,
  kSemaBulkGroupIssue = 11,
  kSemaBulkGroupCommit = 12,
  kSemaBulkGroupWait = 13,
  kSemaAsyncCopyIssue = 14,
  kSemaAsyncCopyCommit = 15,
  kSemaAsyncCopyWait = 16,
  kSemaBulkCopy = 17,
  kSemaBulkReduce = 18,
  kSemaWarpgroupArrive = 19,
  kSemaWarpgroupWait = 20,
  kSemaWgmma = 22,
  kSemaUmmaIssue = 23,
  kSemaUmmaCommit = 24,
  kSemaTmemAlloc = 25,
  kSemaTmemDealloc = 26,
  kSemaTmemRelinquish = 27,
  kSemaTmemLoad = 28,
  kSemaTmemStore = 29,
  kSemaTmemCopy = 30,
  kSemaTmemWaitStore = 31,
  kSemaClcQuery = 32,
  kSemaSharedStoreAsync = 33,
  kSemaGlobalSync = 34,
  kSemaReconvergenceStart = 35,
  kSemaReconvergenceComplete = 36,
  kSemaNanosleep = 37,
  kSemaDescriptorVisibility = 38,
  kSemaClockMarker = 39,
  kSemaDependentsLaunch = 40,
};

// SEMA.flags bits (semantics.py SEMA_FLAG_*)
constexpr std::uint16_t kSemaFlagAfterReady = 0x1;
constexpr std::uint16_t kSemaFlagAlsoArrives = 0x2;
constexpr std::uint16_t kSemaFlagImplicitCommit = 0x4;
// wgmma group semantics (semantics.py SEMA_FLAG_GROUP_*)
constexpr std::uint16_t kSemaFlagGroupOpener = 0x8;
constexpr std::uint16_t kSemaFlagGroupCommit = 0x10;
// NANOSLEEP.SYNCS (semantics.py SEMA_FLAG_SYNCS): wakes on the last
// polled mbarrier's phase completion
constexpr std::uint16_t kSemaFlagSyncs = 0x40;
// TMA .2CTA (cta_group::2): transaction bytes post on the destination
// rank's pair leader (rank & ~1)
constexpr std::uint16_t kSemaFlagCtaPair = 0x80;
// mbarrier_arrive .A<n>TX: transaction_bytes are complete_tx
// (0 arrivals), not expect_tx
constexpr std::uint16_t kSemaFlagCompleteTx = 0x100;

constexpr std::uint32_t kSemaNoObject = 0xFFFFFFFFu;

// TIME.kind ids used by the sync step (timing_rows.py TimingKind)
constexpr std::uint32_t kTimingSharedStoreVisibility = 3;
constexpr std::uint32_t kTimingGlobalStoreVisibility = 4;
constexpr std::uint32_t kTimingNanosleepWarpDelay = 16;
constexpr std::uint32_t kTimingCtaBarrierRelease = 17;
constexpr std::uint32_t kTimingAsyncCopyMbarrierArrival = 18;
constexpr std::uint32_t kTimingTmaServiceStart = 21;
constexpr std::uint32_t kTimingMbarrierPhaseRelease = 22;
constexpr std::uint32_t kTimingTmaCompletion = 38;
constexpr std::uint32_t kTimingClusterBarrierRelease = 42;
constexpr std::uint32_t kTimingDescriptorRefetch = 43;
constexpr std::uint32_t kTimingMbarrierTrywaitTimeout = 61;
constexpr std::uint32_t kTimingTensorPipeBeat = 63;
constexpr std::uint32_t kTimingTensorResultLatency = 64;
constexpr std::uint32_t kTimingGlobalStoreIssueInterval = 65;
constexpr std::uint32_t kTimingSharedFenceDrain = 66;
constexpr std::uint32_t kTimingSharedStoreAcceptanceSmsp = 80;
constexpr std::uint32_t kTimingTmaReduceStore = 81;

}  // namespace cyclescope
