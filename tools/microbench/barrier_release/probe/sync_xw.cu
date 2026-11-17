// Cross-warp mbarrier release: synchronization.mbarrier.release (kinds
// 22/23), tensor.umma.commit (kind 12, cta_group 1/2) and the cta_group::2
// peer-arrival skew (kind 45).
//
// A same-thread op->try_wait chain never blocks (sync_mb: wait == nowait),
// so the release delay only exists for a waiter that is already blocked.
// Here warp 1 (waiter) blocks in try_wait on barrier A; warp 0 (signaler)
// lets it settle for `delay` cycles, reads t_s and then completes A's phase:
//   arr  mbarrier.arrive [A]                      (A count 1)
//   tx   arrive.expect_tx [A],16 before the delay; complete_tx [A],16 at t_s
//   c1   tcgen05.commit.cta_group::1 [A]          (no MMA in flight)
//   c2   cluster 2: leader tcgen05.commit.cta_group::2.multicast 0b11 [A]
// The waiter reads t_w right after its wait returns (same SM clock as t_s).
// Two waiters: "try" = blocked mbarrier.try_wait (the CUTLASS consumer form;
// it observes the phase on its suspend/poll grid), "spin" = test_wait spin
// (fine detector).  The signaler delay is jittered over [delay, delay+2048)
// per (launch, rep) so the poll phase is uniformly randomized; E[K] is used.
// K = t_w - t_s.  Engine: local arrive lands at issue; an empty
// commit arrives at issue + kind 12; a blocked waiter resumes at phase
// completion + kind 22/23.  With o_s / o_w the stall cycles CS2R->op and
// try_wait->CS2R (sass_check) and r the immediate try_wait result latency
// (sync_mb tw_only):
//   kind22 = E[K_arr|try] - o_s - o_w - r ;  kind23 = E[K_tx|try] - ... ;
//   kind12 = (E[K_cN|spin] - o_s_cN) - (E[K_arr|spin] - o_s_arr)
// c2 also: the peer's waiter, on wake, stores a flag into the leader's
// shared memory; leader warp 2 reads t_f on the leader clock when it sees
// it.  t_f - t_w(leader) - one_way_flag (sync_mb flag_rtt / 2) estimates the
// peer-minus-leader arrival skew.
#include "../sync_ops.cuh"

#include <utility>

enum XOp { kXArr = 0, kXTx = 1, kXC1 = 2, kXC2 = 3 };

template <int OP>
__global__ void __launch_bounds__(96, 1) k_xw(std::uint64_t* out, int reps, std::uint32_t delay, int spin,
                                              std::uint32_t seed) {
  constexpr bool C2 = OP == kXC2;
  __shared__ __align__(8) std::uint64_t mbar[2];  // A, ACK
  __shared__ std::uint32_t flag, slot;
  const std::uint32_t rank = C2 ? mb_ctarank() : 0u;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar[0])), ack = mb_opaque(mb_smem_u32(&mbar[1]));
  const std::uint32_t flag_local = mb_opaque(mb_smem_u32(&flag));
  const std::uint32_t flag_leader = mb_opaque(C2 ? mapa(flag_local, 0) : flag_local);
  const std::uint32_t ack_leader = mb_opaque(C2 ? mapa(ack, 0) : ack);
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, C2 ? 2 : 1);
    flag = 0;
    mb_fence_mbar_init();
  }
  __syncthreads();
  if constexpr (C2) mb_cluster_sync();
  constexpr bool kTc = OP == kXC1 || OP == kXC2;
  if constexpr (kTc) {
    if (warp == 0) {
      mb_tmem_alloc<C2 ? 2 : 1>(mb_smem_u32(&slot), 32);
      mb_tmem_relinquish<C2 ? 2 : 1>();
    }
  }
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (C2) mb_cluster_sync();
  mb_tc_fence_after();
  if (lane == 0) {
    std::uint32_t ph = 0;
    if (warp == 1) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        if (spin) mb_mbar_spin_to(a, ph);
        else mb_mbar_wait_to(a, ph);
        const std::uint64_t t_w = mb_clock();
        if (rank == 0) out[3 * rep + 1] = t_w;
        if constexpr (C2) {
          if (rank == 1) st_cluster(flag_leader, static_cast<std::uint32_t>(rep + 1));
        }
        ph ^= 1u;
        arrive_cluster(ack_leader);
      }
    } else if (warp == 0 && rank == 0) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        if constexpr (OP == kXTx) tx_expect(a);
        // jitter over many poll periods randomizes the waiter's poll phase
        const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
        const std::uint32_t d = delay + ((h ^ (h >> 15)) & 2047u);
        const std::uint64_t td = mb_clock();
        while (mb_clock() - td < d) {
        }
        const std::uint64_t t_s = mb_clock();
        if constexpr (OP == kXArr) mb_mbar_arrive(a);
        else if constexpr (OP == kXTx) tx_complete(a);
        else if constexpr (OP == kXC1) commit1(a);
        else commit2_mc(a, 0x3);
        out[3 * rep + 0] = t_s;
        mb_mbar_wait_to(ack, ph);
        ph ^= 1u;
      }
    } else if (warp == 2 && C2 && rank == 0) {
#pragma unroll 1
      for (int rep = 0; rep < reps; ++rep) {
        spin_local_to(flag_local, static_cast<std::uint32_t>(rep + 1));
        out[3 * rep + 2] = mb_clock();
      }
    }
  }
  __syncwarp();
  mb_tc_fence_before();
  __syncthreads();
  if constexpr (C2) mb_cluster_sync();
  if constexpr (kTc) {
    if (warp == 0) mb_tmem_dealloc<C2 ? 2 : 1>(slot, 32);
  }
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "sync_xw");
  struct Case {
    const char* op;
    int cluster;
    void (*k)(std::uint64_t*, int, std::uint32_t, int, std::uint32_t);
  };
  const Case cases[] = {{"arr", 1, &k_xw<kXArr>}, {"tx", 1, &k_xw<kXTx>}, {"c1", 1, &k_xw<kXC1>}, {"c2", 2, &k_xw<kXC2>}};
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  struct Wcfg {
    int spin;
    std::uint32_t delay;
  };
  const Wcfg wcfgs[] = {{0, 1000u}, {0, 4000u}, {0, 16000u}, {1, 4000u}};
  for (const auto& c : cases) {
    for (const auto& wc : wcfgs) {
      const std::uint32_t delay = wc.delay;
      const int spin = wc.spin;
      const std::string variant = std::string("xw_") + c.op + (spin ? ".spin" : ".try") + ".d" + std::to_string(delay);
      if (!mb_selected(args, variant)) continue;
      const std::string params = std::string("\"op\":\"") + c.op + "\",\"cluster\":" + std::to_string(c.cluster) +
                                 ",\"delay\":" + std::to_string(delay) + ",\"waiter\":\"" + (spin ? "spin" : "try") + "\"";
      std::vector<double> k, f;
      bool timed_out = false;
      for (int l = 0; l < args.launches && !timed_out; ++l) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(c.cluster, 1, 1);
        cfg.blockDim = dim3(96, 1, 1);
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = c.cluster;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        MB_CHECK(cudaLaunchKernelEx(&cfg, c.k, d_out, args.reps, delay, spin, static_cast<std::uint32_t>(l)));
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        timed_out = mb_take_err();
        std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) {
          k.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 1] - h[3 * r + 0])));
          if (c.cluster == 2) f.push_back(static_cast<double>(static_cast<std::int64_t>(h[3 * r + 2] - h[3 * r + 1])));
        }
      }
      if (timed_out) {
        w.fail(variant, 1, params, "mbarrier wait timed out");
        continue;
      }
      w.row(variant, 1, params, k);
      if (c.cluster == 2) w.row(variant + ".peerflag_minus_leaderwake", 1, params, f);
    }
  }
  return w.failures ? 1 : 0;
}
