// tmem protocol v1 (tools/microbench/tmem/protocol.json):
// b300 LDTM 32x32b.x128 issue_interval, by same-SMSP participant count.
// One CTA of 4*N warps; the N warps with warp_id % 4 == 0 (all on SMSP 0,
// all addressing TMEM lane quadrant 0) each issue one tcgen05.ld of the
// cell shape between two clock64 reads + tcgen05.wait::ld; the other warps
// only ride the per-rep bar.sync.  Per rep the window is
// max(t_e) - min(t_s) over the participants (same SM clock).
// Warp 0 allocates 128 TMEM columns once and fills them once up front.
#include "../mb.cuh"
#include "../../service_latency/probe/tmem_ops.cuh"

#include <string>
#include <vector>

template <class Shape>
__global__ void k_ldtm(std::uint64_t* out, std::uint32_t* sink, int reps) {
  __shared__ std::uint32_t alloc_slot[1];
  const int warp = threadIdx.x >> 5;
  const bool part = (warp % 4) == 0;
  if (warp == 0) {
    mb_tmem_alloc<1>(mb_smem_u32(alloc_slot), 128);
    mb_tmem_relinquish<1>();
  }
  __syncthreads();
  const std::uint32_t taddr = alloc_slot[0];
  if (warp == 0) mb_tmem_fill(taddr, 128, 0x3f800000u);
  __syncthreads();
  std::uint32_t acc = 0;
  for (int r = 0; r < reps; ++r) {
    __syncthreads();  // all 4N warps leave together
    if (part) {
      std::uint32_t v[Shape::NR];
      const std::uint64_t t_s = mb_clock();
      Shape::ld(taddr, v);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      const std::uint64_t t_e = mb_clock();
      out[static_cast<std::size_t>(r) * 64 + warp * 2 + 0] = t_s;
      out[static_cast<std::size_t>(r) * 64 + warp * 2 + 1] = t_e;
      for (int i = 0; i < Shape::NR; ++i) acc += v[i];
    }
  }
  __syncthreads();
  if (part && acc == 0xdeadbeefu) sink[warp] = acc;  // keep v live past t_e
  if (warp == 0) mb_tmem_dealloc<1>(taddr, 128);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "ldtm");
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps * 64));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 64));
  MB_CHECK(cudaMemset(d_sink, 0, sizeof(std::uint32_t) * 64));
  const int Ns[4] = {1, 2, 4, 8};
  auto run_shape = [&](auto shape_tag, const char* shape_name, auto k) {
    for (int n : Ns) {
      const std::string variant =
          std::string(shape_name) + ".n" + std::to_string(n);
      if (!mb_selected(args, variant)) continue;
      for (int l = 0; l < args.launches; ++l) {
        MB_CHECK(cudaMemset(d_out, 0, sizeof(std::uint64_t) * args.reps * 64));
        k<<<1, 32 * 4 * n>>>(d_out, d_sink, args.reps);
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess) {
          w.fail(variant, n,
                 "\"shape\":\"" + std::string(shape_name) +
                     "\",\"n\":" + std::to_string(n) +
                     ",\"launch\":" + std::to_string(l),
                 (std::string("launch:") + cudaGetErrorName(lerr)).c_str());
          continue;
        }
        MB_CHECK(cudaDeviceSynchronize());
        // per-rep windows: max(t_e) - min(t_s) over the n participant warps
        std::vector<std::uint64_t> h(static_cast<std::size_t>(args.reps) * 64);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost));
        std::vector<double> s;
        for (int r = 1; r < args.reps; ++r) {  // rep 0 = warm-up
          std::uint64_t tmin = ~0ull, tmax = 0;
          for (int p = 0; p < n; ++p) {
            const int widx = p * 4;  // participants are warps 0,4,...,4(n-1)
            const std::uint64_t ts = h[static_cast<std::size_t>(r) * 64 + widx * 2 + 0];
            const std::uint64_t te = h[static_cast<std::size_t>(r) * 64 + widx * 2 + 1];
            if (ts < tmin) tmin = ts;
            if (te > tmax) tmax = te;
          }
          s.push_back(static_cast<double>(tmax - tmin));
        }
        w.row(variant, n,
              "\"shape\":\"" + std::string(shape_name) +
                  "\",\"n\":" + std::to_string(n) +
                  ",\"launch\":" + std::to_string(l),
              s);
      }
    }
  };
  // x128 = target; x64 / x32 = controls (installed 32.0 / 16.0)
  run_shape(Tm_32x32b_x128{}, Tm_32x32b_x128::NAME, k_ldtm<Tm_32x32b_x128>);
  run_shape(Tm_32x32b_x64{}, Tm_32x32b_x64::NAME, k_ldtm<Tm_32x32b_x64>);
  run_shape(Tm_32x32b_x32{}, Tm_32x32b_x32::NAME, k_ldtm<Tm_32x32b_x32>);
  return w.failures ? 1 : 0;
}
