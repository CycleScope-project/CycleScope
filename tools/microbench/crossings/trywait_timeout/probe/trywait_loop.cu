// synchronization.mbarrier.trywait_timeout (crossings protocol-v2, replaces
// trywait.cu): the quantity the engine needs is the period of an unsatisfied
// try_wait.parity poll loop, not one instruction.
//
// Loop cells (warp 0 lane 0): N iterations of { mbarrier.try_wait.parity /
// mbarrier.test_wait.parity on a never-completing mbarrier phase; branch }
// written as one inline PTX block so the SASS window is exactly PHASECHK +
// loop control.  MODE 0 = test_wait (non-suspending control), MODE 1 =
// try_wait no hint, MODE 2/3 = try_wait with suspendTimeHint ns.
//
// Release cell (k_release<D>): warp 0 lane 0 sits in the no-hint
// while(!try_wait) loop; warp 1 lane 0 reads t_s0, spins D + jitter cycles
// (jitter = hash(launch, rep) % 2048, sync_xw form), reads t_s1, arrives.
// t_w = waiter's clock right after the loop exits.  K = t_w - t_s1 is the
// wake-up latency; t_w - (t_s0 + D_actual) is the same against the nominal
// release time.
#include "../mb.cuh"

#include <string>
#include <vector>

constexpr int kMbars = 64;

// N is a kernel argument (not a template constant): ptxas fully unrolls a
// PTX loop with a constant trip count (b300-r1 SASS: N PHASECHK, no BRA).
template <int MODE, std::uint32_t HINT>
__global__ void __launch_bounds__(64, 1) k_loop(std::uint64_t* out, int reps, int n) {
  __shared__ __align__(8) std::uint64_t mb[kMbars];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < kMbars; ++i) mb_mbar_init(mb_smem_u32(&mb[i]), 1);
    mb_fence_mbar_init();
  }
  __syncthreads();
  __shared__ std::uint32_t slot[2];
  if (warp == 0 && lane == 0) {
    const std::uint32_t base = mb_smem_u32(&mb[0]);
    // n pinned through shared memory (ptxas reloaded the parameter with an
    // LDC inside the loop) and the PTX loop marked nounroll (ptxas unrolled
    // the runtime-count loop by 8 in b300-r2)
    const std::uint32_t nn = mb_pin(static_cast<std::uint32_t>(n), &slot[0]);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) {
      const std::uint32_t bar = base + 8u * static_cast<std::uint32_t>(r & (kMbars - 1));
      const std::uint64_t t0 = mb_clock();
      if constexpr (MODE == 0) {
        asm volatile(
            "{\n\t.reg .pred p, q;\n\t.reg .b32 n;\n\t"
            "mov.u32 n, %0;\n\t"
            "MB_TL_%=:\n\t"
            ".pragma \"nounroll\";\n\t"
            "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
            "sub.u32 n, n, 1;\n\t"
            "setp.ne.u32 q, n, 0;\n\t"
            "@q bra MB_TL_%=;\n\t}" ::"r"(nn), "r"(bar)
            : "memory");
      } else if constexpr (HINT == 0) {
        asm volatile(
            "{\n\t.reg .pred p, q;\n\t.reg .b32 n;\n\t"
            "mov.u32 n, %0;\n\t"
            "MB_TL_%=:\n\t"
            ".pragma \"nounroll\";\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
            "sub.u32 n, n, 1;\n\t"
            "setp.ne.u32 q, n, 0;\n\t"
            "@q bra MB_TL_%=;\n\t}" ::"r"(nn), "r"(bar)
            : "memory");
      } else {
        asm volatile(
            "{\n\t.reg .pred p, q;\n\t.reg .b32 n;\n\t"
            "mov.u32 n, %0;\n\t"
            "MB_TL_%=:\n\t"
            ".pragma \"nounroll\";\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0, %2;\n\t"
            "sub.u32 n, n, 1;\n\t"
            "setp.ne.u32 q, n, 0;\n\t"
            "@q bra MB_TL_%=;\n\t}" ::"r"(nn), "r"(bar), "n"(HINT)
            : "memory");
      }
      const std::uint64_t t1 = mb_clock();
      out[r] = t1 - t0;
    }
  }
}

template <std::uint32_t D>
__global__ void __launch_bounds__(64, 1) k_release(std::uint64_t* out, int reps, std::uint32_t seed) {
  __shared__ __align__(8) std::uint64_t mb[kMbars];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    const int i = r & (kMbars - 1);
    if (threadIdx.x == 0) {
      mb_mbar_init(mb_smem_u32(&mb[i]), 1);
      mb_fence_mbar_init();
    }
    __syncthreads();
    const std::uint32_t bar = mb_smem_u32(&mb[i]);
    if (warp == 0 && lane == 0) {
      // no-hint suspending wait loop on the phase parity 0 of this rep's
      // fresh mbarrier; completes when warp 1 arrives.
      asm volatile(
          "{\n\t.reg .pred p;\n\t"
          "MB_REL_%=:\n\t"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
          "@!p bra MB_REL_%=;\n\t}" ::"r"(bar)
          : "memory");
      out[4 * r + 0] = mb_clock();  // t_w
    } else if (warp == 1 && lane == 0) {
      const std::uint64_t t_s0 = mb_clock();
      const std::uint32_t h = (static_cast<std::uint32_t>(r) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = D + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s1 = mb_clock();
      mb_mbar_arrive(bar);
      out[4 * r + 1] = t_s0;
      out[4 * r + 2] = t_s1;
      out[4 * r + 3] = d;
    }
    __syncthreads();
  }
}

using LoopKern = void (*)(std::uint64_t*, int, int);
struct LoopCase {
  const char* variant;
  int n;
  LoopKern k;
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "trywait");
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 4 * args.reps));

  LoopCase cases[12];
  {
    int i = 0;
    for (int n : {16, 64, 256}) cases[i++] = {"test", n, &k_loop<0, 0>};
    for (int n : {16, 64, 256}) cases[i++] = {"try.h0", n, &k_loop<1, 0>};
    for (int n : {16, 64, 256}) cases[i++] = {"try.h1000", n, &k_loop<2, 1000>};
    for (int n : {16, 64, 256}) cases[i++] = {"try.h100000", n, &k_loop<3, 100000>};
  }
  for (const auto& c : cases) {
    if (!mb_selected(args, c.variant)) continue;
    const std::string params = std::string("\"mode\":\"") + c.variant + "\",\"n\":" + std::to_string(c.n);
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      c.k<<<1, 64>>>(d_out, args.reps, c.n);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(c.variant, c.n, params, s);
  }

  // release cell: t_w - t_s1 and t_w - (t_s0 + D_actual) per rep
  for (std::uint32_t d : {2000u, 20000u}) {
    const std::string variant = "release.d" + std::to_string(d);
    if (!mb_selected(args, variant)) continue;
    auto kern = d == 2000u ? &k_release<2000> : &k_release<20000>;
    const std::string params = "\"delay\":" + std::to_string(d);
    std::vector<double> k1, k0;
    for (int l = 0; l < args.launches; ++l) {
      kern<<<1, 64>>>(d_out, args.reps, static_cast<std::uint32_t>(l));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(4 * static_cast<std::size_t>(args.reps));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r) {
        const std::int64_t t_w = static_cast<std::int64_t>(h[4 * r + 0]);
        const std::int64_t t_s0 = static_cast<std::int64_t>(h[4 * r + 1]);
        const std::int64_t t_s1 = static_cast<std::int64_t>(h[4 * r + 2]);
        const std::int64_t da = static_cast<std::int64_t>(h[4 * r + 3]);
        k1.push_back(static_cast<double>(t_w - t_s1));
        k0.push_back(static_cast<double>(t_w - (t_s0 + da)));
      }
    }
    w.row(variant, 1, params, k1);
    w.row(variant + ".minus_t_s0", 1, params, k0);
  }
  return w.failures ? 1 : 0;
}
