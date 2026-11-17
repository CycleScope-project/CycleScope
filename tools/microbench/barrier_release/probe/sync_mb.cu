// synchronization.mbarrier.release (kinds 22/23), tensor.umma.commit
// (kind 12, cta_group 1 and 2) and tensor.umma.commit.pair_skew (kind 45).
//
// One thread (lane 0 of warp 0) issues every timed operation.  Chains are
// compile-time unrolled, U in {1,2,4,8,16,32}; each op kernel runs in two
// modes with IDENTICAL SASS (the mode is a runtime argument):
//   wait   : target barrier A (count 1), parity flips every step, so the
//            try_wait blocks until the op completes the phase;
//   nowait : target barrier B (count 2^20-1, never completes), parity stays
//            1, so the same try_wait returns at once.
//   arr    U x { mbarrier.arrive [X] ; try_wait.parity [X], ph }
//   tx     U x { arrive.expect_tx [X],16 ; complete_tx [X],16 ; try_wait [X], ph }
//   c1     U x { tcgen05.commit.cta_group::1 [X] ; try_wait [X], ph }      (no MMA in flight)
//   c2     cluster 2, leader: U x { tcgen05.commit.cta_group::2.multicast 0b11 [X] ; try_wait [X], ph }
//   tw_only  U x { try_wait [B], 1 }           (nowait mode only: immediate try_wait latency)
//   arr_only U x { mbarrier.arrive [B] }        (nowait mode only: arrive issue cost)
// NOTE (wave1 hardware): wait == nowait to the cycle for arr/tx/c1 -- a
// same-thread op->try_wait chain never blocks, so these chains bound the
// issue path only; the release delay itself comes from probes/sync_xw.cu.
// Engine (engine.cpp release_mbarrier / TensorOp::UmmaCommit): a local arrive
// lands at its issue cycle; an empty commit arrives at issue + kind 12; a
// blocked waiter resumes at phase transition + kind 22/23.  With
// D = slope(wait) - slope(nowait) and G = issue-to-try_wait stall cycles
// along the straight SASS path (sass_check records G per kernel):
//   kind22/23 = D_arr|tx + G_arr|tx ;  kind12 = D_cN + G_cN - kind22.
// pair_skew / flag_rtt (cluster 2, one sample per repetition): the leader
// clocks its own commit-arrival release, then the peer's flag written right
// after the peer's own release; flag_rtt is the leader-clocked flag round
// trip used to remove the one-way flag latency (rtt/2).
#include "../sync_ops.cuh"

#include <utility>

enum Op { kArr, kTx, kC1, kC2, kPairSkew, kFlagRtt, kTwOnly, kArrOnly };

template <int OP, int U>
__device__ __forceinline__ void chain(std::uint32_t x, std::uint32_t& ph, std::uint32_t flip) {
#pragma unroll
  for (int i = 0; i < U; ++i) {
    if constexpr (OP == kArr || OP == kArrOnly) mb_mbar_arrive(x);
    else if constexpr (OP == kTx) tx_arrive_complete(x);
    else if constexpr (OP == kC1) commit1(x);
    else if constexpr (OP == kC2) commit2_mc(x, 0x3);
    if constexpr (OP == kC2) mb_mbar_wait_to(x, ph);
    else if constexpr (OP != kArrOnly) mb_mbar_wait(x, ph);
    ph ^= flip;
  }
}

// 1-CTA ops.  wait: (use_b=0, ph0=0, flip=1); nowait: (use_b=1, ph0=1, flip=0).
template <int OP, int U>
__global__ void __launch_bounds__(32, 1) k_sync1(std::uint64_t* out, int reps, int use_b, std::uint32_t ph0,
                                                  std::uint32_t flip) {
  __shared__ __align__(8) std::uint64_t mbar[2];
  const std::uint32_t a = mb_smem_u32(&mbar[0]), b = mb_smem_u32(&mbar[1]);
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(b, kNever);
    mb_fence_mbar_init();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const std::uint32_t x = use_b ? b : a;
    std::uint32_t ph = ph0;
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint64_t t0 = mb_clock();
      chain<OP, U>(x, ph, flip);
      const std::uint64_t t1 = mb_clock();
      out[rep] = t1 - t0;
    }
  }
  __syncwarp();
}

// cluster-2 ops; leader = rank 0.
template <int OP, int U>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(32, 1)
    k_sync2(std::uint64_t* out, int reps, int use_b, std::uint32_t ph0, std::uint32_t flip) {
  __shared__ __align__(8) std::uint64_t mbar[2];
  __shared__ std::uint32_t flag;
  const std::uint32_t rank = mb_ctarank();
  const std::uint32_t a = mb_smem_u32(&mbar[0]), b = mb_smem_u32(&mbar[1]);
  const std::uint32_t flag_local = mb_smem_u32(&flag);
  const std::uint32_t flag_other = mapa(flag_local, rank ^ 1u);
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(b, kNever);
    flag = 0;
    mb_fence_mbar_init();
  }
  __syncwarp();
  mb_cluster_sync();
  // tcgen05 cta_group::2 operations need the CTA pair's TMEM allocation
  // (a 2CTA commit without it never arrives); allocate the minimum.
  __shared__ std::uint32_t slot;
  mb_tmem_alloc<2>(mb_smem_u32(&slot), 32);
  mb_tmem_relinquish<2>();
  mb_tc_fence_before();
  __syncwarp();
  mb_tc_fence_after();
  if (threadIdx.x == 0) {
    std::uint32_t ph = ph0;
    if constexpr (OP == kC2) {
      const std::uint32_t x = use_b ? b : a;
      if (rank == 0) {
#pragma unroll 1
        for (int rep = 0; rep < reps; ++rep) {
          const std::uint64_t t0 = mb_clock();
          chain<OP, U>(x, ph, flip);
          const std::uint64_t t1 = mb_clock();
          out[rep] = t1 - t0;
        }
      }
      // The peer issues and waits on nothing: a peer waiter can fall two
      // phases behind the leader's chain and alias the parity.
    } else if constexpr (OP == kPairSkew) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        if (rank == 0) {
          const std::uint64_t t0 = mb_clock();
          commit2_mc(a, 0x3);
          mb_mbar_wait_to(a, ph);
          const std::uint64_t t_self = mb_clock();
          spin_local_to(flag_local, static_cast<std::uint32_t>(rep + 1));
          const std::uint64_t t_flag = mb_clock();
          out[2 * rep + 0] = t_self - t0;
          out[2 * rep + 1] = t_flag - t_self;
        } else {
          mb_mbar_wait_to(a, ph);
          st_cluster(flag_other, static_cast<std::uint32_t>(rep + 1));
        }
        ph ^= 1u;
      }
    } else if constexpr (OP == kFlagRtt) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        if (rank == 0) {
          const std::uint64_t t0 = mb_clock();
          st_cluster(flag_other, static_cast<std::uint32_t>(rep + 1));
          spin_local_to(flag_local, static_cast<std::uint32_t>(rep + 1));
          const std::uint64_t t1 = mb_clock();
          out[2 * rep + 0] = t1 - t0;
          out[2 * rep + 1] = 0;
        } else {
          spin_local_to(flag_local, static_cast<std::uint32_t>(rep + 1));
          st_cluster(flag_other, static_cast<std::uint32_t>(rep + 1));
        }
      }
    }
  }
  __syncwarp();
  mb_tc_fence_before();
  mb_cluster_sync();
  mb_tmem_dealloc<2>(slot, 32);
}

using Kern = void (*)(std::uint64_t*, int, int, std::uint32_t, std::uint32_t);
struct Case {
  std::string op;
  int u, cluster, words;
  Kern k;
};

template <int OP, bool C2, int... Us>
void add(std::vector<Case>& v, const char* name, std::integer_sequence<int, Us...>) {
  if constexpr (C2) (v.push_back({name, Us, 2, 1, &k_sync2<OP, Us>}), ...);
  else (v.push_back({name, Us, 1, 1, &k_sync1<OP, Us>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "sync_mb");
  using Us = std::integer_sequence<int, 1, 2, 4, 8, 16, 32>;
  std::vector<Case> cases;
  add<kArr, false>(cases, "arr", Us{});
  add<kTx, false>(cases, "tx", Us{});
  add<kC1, false>(cases, "c1", Us{});
  add<kC2, true>(cases, "c2", Us{});
  add<kTwOnly, false>(cases, "tw_only", Us{});
  add<kArrOnly, false>(cases, "arr_only", Us{});
  cases.push_back({"pair_skew", 1, 2, 2, &k_sync2<kPairSkew, 1>});
  cases.push_back({"flag_rtt", 1, 2, 2, &k_sync2<kFlagRtt, 1>});
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps * 2));
  for (const auto& c : cases) {
    const bool two_word = c.words == 2;
    const bool nowait_only = c.op == "tw_only" || c.op == "arr_only";
    for (int mode = nowait_only ? 1 : 0; mode < (two_word ? 1 : 2); ++mode) {
      const std::string variant = two_word ? c.op : c.op + (mode == 0 ? "_wait" : "_nowait");
      if (!mb_selected(args, variant)) continue;
      const int use_b = mode;
      const std::uint32_t ph0 = mode == 0 ? 0u : 1u, flip = mode == 0 ? 1u : 0u;
      std::vector<double> s0, s1;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        c.k<<<c.cluster, 32>>>(d_out, args.reps, use_b, ph0, flip);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        if (!two_word) {
          mb_collect(d_out, args.reps, 1, 0, s0);
        } else {
          mb_collect(d_out, args.reps, 2, 0, s0);
          mb_collect(d_out, args.reps, 2, 1, s1);
        }
      }
      const std::string params = "\"cluster\":" + std::to_string(c.cluster) + ",\"op\":\"" + c.op + "\"";
      if (timed_out) {
        w.fail(variant, c.u, params, "mbarrier wait timed out");
        continue;
      }
      if (c.op == "pair_skew") {
        w.row("pair_skew.self", 1, params, s0);
        w.row("pair_skew.flag_minus_self", 1, params, s1);
      } else {
        w.row(variant, c.u, params, s0);
      }
    }
  }
  return w.failures ? 1 : 0;
}
