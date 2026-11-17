// arrive_status: result latency of an mbarrier arrive whose returned
// state is consumed (protocol.json).  One CTA on one SM; U dependent
// arrives, each writing its state into a register the next arrive's
// address computation reads (address = base + (state & mask), mask is a
// run-time 0 so ptxas keeps the dependence).  Forms: plain arrive,
// arrive with a register count, arrive_drop.  Controls: discard (state
// unread, addresses independent), phase (last arrive completes the
// phase), load (W warps on one / per-warp barriers).
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../../tma_cold/mb.cuh"

__shared__ __align__(8) std::uint64_t kMbar[8];

enum { F_ARRIVE = 0, F_ARRIVE_CNT = 1, F_ARRIVE_DROP = 2 };
enum { K_CHAIN = 0, K_DISCARD = 1, K_PHASE = 2, K_LOAD_SAME = 3,
       K_LOAD_OWN = 4 };

template <int F>
__device__ __forceinline__ std::uint64_t mb_arrive(std::uint32_t addr,
                                                 std::uint32_t count) {
  std::uint64_t s;
  if constexpr (F == F_ARRIVE)
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                 : "=l"(s) : "r"(addr) : "memory");
  else if constexpr (F == F_ARRIVE_CNT)
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(s) : "r"(addr), "r"(count) : "memory");
  else
    asm volatile("mbarrier.arrive_drop.shared::cta.b64 %0, [%1];"
                 : "=l"(s) : "r"(addr) : "memory");
  return s;
}

// KIND: K_CHAIN dependent addresses on this warp's own barrier;
// K_DISCARD independent addresses (each arrive its own barrier), state
// unread; K_PHASE like CHAIN but init count completes at the last
// arrive; K_LOAD_SAME all participating warps on kMbar[0]; K_LOAD_OWN
// warp w on kMbar[w].  init_cnt is the mbarrier.init count used at
// kernel entry.
template <int U, int F, int KIND>
__global__ void __launch_bounds__(128, 1) k_arr(
    int reps, std::uint64_t* out, int warps, unsigned mask,
    unsigned count, unsigned init_cnt) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp == 0 && lane == 0)
    for (int b = 0; b < 8; ++b)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                   "r"(static_cast<std::uint32_t>(
                       __cvta_generic_to_shared(&kMbar[b]))),
                   "r"(init_cnt) : "memory");
  __syncthreads();
  const std::uint32_t base = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(
          KIND == K_LOAD_SAME ? &kMbar[0] : &kMbar[warp & 7]));
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (warp < warps) {
      std::uint64_t st = 0;
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int k = 0; k < U; ++k) {
        const std::uint32_t a =
            KIND == K_DISCARD
                ? base + static_cast<std::uint32_t>((k * 8) & 56)
                : base + static_cast<std::uint32_t>(st & mask);
        st = mb_arrive<F>(a, count);
      }
      const unsigned v = static_cast<unsigned>(st);
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(v) : "memory");
      if (lane == 0) {
        out[warp * 2 * 33 + 2 * rep] = t1 - t0;
        out[warp * 2 * 33 + 2 * rep + 1] = v;
      }
    }
    __syncthreads();
  }
}

// --- v2: issue cost of arrives whose state is discarded -------------
// k_arr2<U, F2>: U back-to-back independent arrives of one PTX form,
// state discarded, each on a different barrier of this warp's own
// 16-barrier segment (kMbar2[warp*16 .. warp*16+15]).  F2:
//   0 arrive            mbarrier.arrive.shared::cta.b64 s, [a]
//   1 arrive_cnt        mbarrier.arrive.shared::cta.b64 s, [a], cnt
//   2 expect_tx         mbarrier.arrive.expect_tx.shared::cta.b64 s, [a], tx
//   3 expect_tx_only    mbarrier.expect_tx.shared::cta.b64 [a], tx
//   4 cl_arrive         mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [a]
//   5 cl_arrive_expect_tx  ...arrive.expect_tx...shared::cluster.b64 _, [a], tx
// Cluster forms address this CTA's own barrier through mapa; the kernel
// is launched with a 1- or 2-CTA cluster.
__shared__ __align__(8) std::uint64_t kMbar2[64];

enum { F2_ARRIVE = 0, F2_ARRIVE_CNT = 1, F2_EXPECT_TX = 2,
       F2_EXPECT_TX_ONLY = 3, F2_CL_ARRIVE = 4,
       F2_CL_ARRIVE_EXPECT_TX = 5 };

template <int U, int F2>
__global__ void __launch_bounds__(128, 1) k_arr2(
    int reps, std::uint64_t* out, int warps, unsigned txbytes,
    unsigned init_cnt) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int cta = blockIdx.x;
  if (warp == 0 && lane == 0)
    for (int b = 0; b < 64; ++b)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                   "r"(static_cast<std::uint32_t>(
                       __cvta_generic_to_shared(&kMbar2[b]))),
                   "r"(init_cnt) : "memory");
  __syncthreads();
  if constexpr (F2 >= F2_CL_ARRIVE)
    asm volatile("barrier.cluster.arrive.aligned;\n\t"
                 "barrier.cluster.wait.aligned;" ::: "memory");
  const std::uint32_t sbase = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(&kMbar2[warp * 16]));
  std::uint32_t base = sbase;
  if constexpr (F2 >= F2_CL_ARRIVE) {
    std::uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(base) : "r"(sbase), "r"(rank));
  }
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (warp < warps) {
      std::uint64_t st = 0;
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int k = 0; k < U; ++k) {
        const std::uint32_t a = base + static_cast<std::uint32_t>(k * 8);
        if constexpr (F2 == F2_ARRIVE)
          asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                       : "=l"(st) : "r"(a) : "memory");
        else if constexpr (F2 == F2_ARRIVE_CNT)
          asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;"
                       : "=l"(st) : "r"(a), "r"(txbytes) : "memory");
        else if constexpr (F2 == F2_EXPECT_TX)
          asm volatile(
              "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
              : "=l"(st) : "r"(a), "r"(txbytes) : "memory");
        else if constexpr (F2 == F2_EXPECT_TX_ONLY)
          asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                       :: "r"(a), "r"(txbytes) : "memory");
        else if constexpr (F2 == F2_CL_ARRIVE)
          asm volatile(
              "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _,"
              " [%0];" :: "r"(a) : "memory");
        else
          asm volatile(
              "mbarrier.arrive.expect_tx.relaxed.cluster.shared::"
              "cluster.b64 _, [%0], %1;" :: "r"(a), "r"(txbytes)
              : "memory");
      }
      const unsigned v = static_cast<unsigned>(st);
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(v) : "memory");
      if (lane == 0) {
        const int slot = cta * 4 + warp;
        out[slot * 2 * 33 + 2 * rep] = t1 - t0;
        out[slot * 2 * 33 + 2 * rep + 1] = v;
      }
    }
    __syncthreads();
  }
}

template <int U, int F2>
static void launch_k2(int reps, std::uint64_t* out, int warps,
                      unsigned txbytes, unsigned init_cnt, int cn) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attrs[1];
  cfg.gridDim = dim3(cn, 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = nullptr;
  if (cn > 1) {
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cn;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, k_arr2<U, F2>, reps, out, warps, txbytes,
                     init_cnt);
}

int main(int argc, char** argv) {
  std::string outp = "arrive_status.jsonl";
  std::string cell;
  int reps = 33, launches = 8, v2 = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--out" && i + 1 < argc)
      outp = argv[++i];
    else if (std::string(argv[i]) == "--cell" && i + 1 < argc)
      cell = argv[++i];
    else if (std::string(argv[i]) == "--launches" && i + 1 < argc)
      launches = std::stoi(argv[++i]);
    else if (std::string(argv[i]) == "--reps" && i + 1 < argc)
      reps = std::stoi(argv[++i]);
    else if (std::string(argv[i]) == "--v2")
      v2 = 1;
  }
  if (reps > 33 || reps < 1 || launches < 1) return 1;
  const MbDevice dev = mb_query_device();
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * 33 * 8));
  FILE* f = std::fopen(outp.c_str(), "a");
  if (!f) return 1;
  const char* fname[] = {"arrive", "arrive_cnt", "arrive_drop"};
  const char* kname[] = {"chain", "discard", "phase", "load_same",
                         "load_own"};
  auto run_cell = [&](const std::string& variant, int nslots, auto launch) {
    if (!cell.empty() && variant.find(cell) == std::string::npos)
      return;
    std::vector<double> s;
    for (int l = 0; l < launches; ++l) {
      launch();
      cudaError_t e = cudaGetLastError();
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      std::vector<std::uint64_t> h(2 * 33 * 8);
      MB_CHECK(cudaMemcpy(h.data(), d_out,
                          sizeof(std::uint64_t) * 2 * 33 * 8,
                          cudaMemcpyDeviceToHost));
      if (l == 0) continue;
      for (int w = 0; w < nslots; ++w)
        for (int r = 1; r < reps; ++r)
          s.push_back(static_cast<double>(h[w * 2 * 33 + 2 * r]));
    }
    std::fprintf(f,
                 "{\"schema\":\"cyclescope.microbench.arrive_status.v1\","
                 "\"probe\":\"arrive_status\",\"variant\":\"%s\","
                 "\"n\":%zu,\"samples\":[", variant.c_str(), s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
      std::fprintf(f, i ? ",%.1f" : "%.1f", s[i]);
    std::fprintf(f, "],\"gpu_name\":\"%s\",\"gpu_uuid\":\"%s\"}\n",
                 dev.name.c_str(), dev.uuid.c_str());
    std::fflush(f);
    std::fprintf(stderr, "%s: %zu samples\n", variant.c_str(), s.size());
  };

  using KF = void (*)(int, std::uint64_t*, int, unsigned, unsigned,
                      unsigned);
  auto pick = [](int u, int form, int kind) -> KF {
    const int ui = u == 1 ? 0 : u == 2 ? 1 : u == 4 ? 2
                   : u == 8 ? 3 : 4;
    const int key = ui * 15 + form * 5 + kind;
    switch (key) {
#define C(uu, ff, kk) \
  case uu * 15 + ff * 5 + kk: return k_arr< \
      uu == 0 ? 1 : uu == 1 ? 2 : uu == 2 ? 4 : uu == 3 ? 8 : 16, \
      ff == 0 ? F_ARRIVE : ff == 1 ? F_ARRIVE_CNT : F_ARRIVE_DROP, kk>
      C(0,0,0); C(0,0,1); C(0,1,0); C(0,1,1); C(0,2,0); C(0,2,1);
      C(1,0,0); C(1,0,1); C(1,1,0); C(1,1,1); C(1,2,0); C(1,2,1);
      C(2,0,0); C(2,0,1); C(2,0,2); C(2,0,3); C(2,0,4);
      C(2,1,0); C(2,1,1); C(2,2,0); C(2,2,1);
      C(3,0,0); C(3,0,1); C(3,1,0); C(3,1,1); C(3,2,0); C(3,2,1);
      C(4,0,0); C(4,0,1); C(4,0,2); C(4,0,3); C(4,0,4);
      C(4,1,0); C(4,1,1); C(4,2,0); C(4,2,1);
      default: return nullptr;
#undef C
    }
  };

  const unsigned mask = 0, count = 1, big = 0x40000;
  if (v2) {
    const char* f2name[] = {"arrive", "arrive_cnt", "expect_tx",
                            "expect_tx_only", "cl_arrive",
                            "cl_arrive_expect_tx"};
    auto pick2 = [](int u, int f2) -> void (*)(int, std::uint64_t*,
                                               int, unsigned,
                                               unsigned, int) {
      const int ui = u == 1 ? 0 : u == 2 ? 1 : u == 4 ? 2
                     : u == 8 ? 3 : 4;
      switch (ui * 6 + f2) {
#define C2(uu, ff) \
  case uu * 6 + ff: return launch_k2< \
      uu == 0 ? 1 : uu == 1 ? 2 : uu == 2 ? 4 : uu == 3 ? 8 : 16, ff>
        C2(0,0); C2(0,1); C2(0,2); C2(0,3); C2(0,4); C2(0,5);
        C2(1,0); C2(1,1); C2(1,2); C2(1,3); C2(1,4); C2(1,5);
        C2(2,0); C2(2,1); C2(2,2); C2(2,3); C2(2,4); C2(2,5);
        C2(3,0); C2(3,1); C2(3,2); C2(3,3); C2(3,4); C2(3,5);
        C2(4,0); C2(4,1); C2(4,2); C2(4,3); C2(4,4); C2(4,5);
        default: return nullptr;
#undef C2
      }
    };
    const unsigned txb = 8;
    const int cns[2] = {1, 2};
    for (int f2 = 0; f2 < 6; ++f2)
      for (int ci = 0; ci < (f2 >= F2_CL_ARRIVE ? 2 : 1); ++ci) {
        const int cn = cns[ci];
        for (int w : {1, 4})
          for (int u : {1, 2, 4, 8, 16}) {
            const std::string v = std::string("arrive_status.v2.") +
                f2name[f2] + ".c" + std::to_string(cn) + ".w" +
                std::to_string(w) + ".u" + std::to_string(u);
            auto kf = pick2(u, f2);
            run_cell(v, w * cn, [&] {
              kf(reps, d_out, w, txb, big, cn);
            });
          }
      }
    std::fclose(f);
    return 0;
  }
  for (int form = 0; form < 3; ++form)
    for (int u : {1, 2, 4, 8, 16})
      for (int kind = 0; kind < 2; ++kind) {
        const std::string v = std::string("arrive_status.") +
            fname[form] + "." + kname[kind] + ".w1.u" +
            std::to_string(u);
        KF kf = pick(u, form, kind);
        run_cell(v, 1, [&] {
          kf<<<1, 128>>>(reps, d_out, 1, mask, count, big);
        });
      }
  for (int u : {4, 16}) {
    const std::string v = "arrive_status.arrive.phase.w1.u" +
        std::to_string(u);
    KF kf = pick(u, 0, K_PHASE);
    run_cell(v, 1, [&] {
      kf<<<1, 128>>>(reps, d_out, 1, mask, count, u * 32);
    });
  }
  for (int w : {4})
    for (int kind : {K_LOAD_SAME, K_LOAD_OWN})
      for (int u : {4, 16}) {
        const std::string v = "arrive_status.arrive." +
            std::string(kname[kind]) + ".w" + std::to_string(w) + ".u" +
            std::to_string(u);
        KF kf = pick(u, 0, kind);
        run_cell(v, w, [&] {
          kf<<<1, 128>>>(reps, d_out, w, mask, count, big);
        });
      }
  std::fclose(f);
  return 0;
}
