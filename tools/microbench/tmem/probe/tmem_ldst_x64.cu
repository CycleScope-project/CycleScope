// tmem protocol r2 addendum: 16x32bx2 load x64 one-warp ld_wc / ld_thru.
// Verbatim port of blackwell-prototype probes/tmem_ldst.cu (ld_wc / ld_thru
// kernels + sampling loop), restricted to the x64 cell and its control
// shapes: 16x32bx2.x64 gets ld_thru K {1,2,3} (one-warp register window
// 192 dst regs), the controls keep their original K series.  st_w /
// st_thru / ldred / NW=4 variants are not part of this cell and are
// left out; the SASS gate only counts LDTM in the timed window.
#include "../mb.cuh"
#include "../../service_latency/probe/tmem_ops.cuh"

#include <type_traits>
#include <utility>
#include <vector>

enum Mode { kLdWc = 1, kLdThru = 2 };

__device__ __forceinline__ void wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <class S, int MODE, int U>
__device__ __forceinline__ std::uint32_t body(std::uint32_t tbase,
                                              std::uint32_t& acc) {
  std::uint32_t last = 0;
  if constexpr (MODE == kLdWc) {
    std::uint32_t r[S::NR];
#pragma unroll
    for (int i = 0; i < U; ++i) {
      S::ld(tbase, r);
      wait_ld();
      acc ^= r[0];
    }
  } else {  // kLdThru: K distinct-register loads, one wait, consume all
    std::uint32_t r[U][S::NR];
#pragma unroll
    for (int k = 0; k < U; ++k)
      S::ld(tbase + static_cast<std::uint32_t>(k * S::COLS), r[k]);
    wait_ld();
#pragma unroll
    for (int k = 0; k < U; ++k) acc ^= r[k][0];
  }
  return last;
}

template <class S, int MODE, int U>
__global__ void __launch_bounds__(32, 1) k_tm(std::uint64_t* out,
                                              std::uint32_t* sink,
                                              int reps) {
  __shared__ std::uint32_t slot;
  mb_tmem_alloc<1>(mb_smem_u32(&slot), 512);
  mb_tc_fence_before();
  __syncthreads();
  mb_tc_fence_after();
  const std::uint32_t tbase = slot;
  mb_tmem_relinquish<1>();
  mb_tmem_fill(tbase, 512, 0x3f800000u ^ threadIdx.x);
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t0 = mb_clock();
    const std::uint32_t last = body<S, MODE, U>(tbase, acc);
    const std::uint64_t t1 = mb_clock();
    acc += last;
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x & 31u] ^= acc;
  mb_tc_fence_before();
  __syncthreads();
  mb_tmem_dealloc<1>(slot, 512);
}

using Kern = void (*)(std::uint64_t*, std::uint32_t*, int);
struct Case {
  std::string variant, shape;
  int u;
  Kern k;
};

template <class S, int MODE, int... Us>
void add(std::vector<Case>& v, const char* variant,
         std::integer_sequence<int, Us...>) {
  (v.push_back({variant, S::NAME, Us, &k_tm<S, MODE, Us>}), ...);
}

template <class S>
void add_cell(std::vector<Case>& v) {
  add<S, kLdWc>(v, "ld_wc", std::integer_sequence<int, 1, 2, 4, 8>{});
  if constexpr (S::NR * 8 <= 64)
    add<S, kLdThru>(v, "ld_thru",
                    std::integer_sequence<int, 1, 2, 4, 8>{});
  else if constexpr (S::NR * 4 <= 64)
    add<S, kLdThru>(v, "ld_thru", std::integer_sequence<int, 1, 2, 4>{});
  else if constexpr (S::NR * 2 <= 64)
    add<S, kLdThru>(v, "ld_thru", std::integer_sequence<int, 1, 2>{});
  else   // NR == 64 (16x32bx2.x64): one-warp window holds K = 3
    add<S, kLdThru>(v, "ld_thru", std::integer_sequence<int, 1, 2, 3>{});
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "tmem_ldst_x64");
  std::vector<Case> cases;
  add_cell<Tm_16x32bx2_x64>(cases);   // the r2 cell: ld_thru K {1,2,3}
  // controls reproduce installed x16/x32 values of both shapes
  add_cell<Tm_16x32bx2_x16>(cases);
  add_cell<Tm_16x32bx2_x32>(cases);
  add_cell<Tm_32x32b_x16>(cases);
  add_cell<Tm_32x32b_x32>(cases);
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 32));
  MB_CHECK(cudaMemset(d_sink, 0, sizeof(std::uint32_t) * 32));
  for (const auto& c : cases) {
    const std::string full = c.variant + "." + c.shape;
    if (!mb_selected(args, full)) continue;
    std::vector<double> samples;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, 32>>>(d_out, d_sink, args.reps);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, samples);
    }
    const std::string params =
        "\"shape\":\"" + c.shape + "\",\"op\":\"ld\"";
    w.row(c.variant, c.u, params, samples);
  }
  return w.failures ? 1 : 0;
}
