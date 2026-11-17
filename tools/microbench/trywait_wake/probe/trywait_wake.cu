// trywait_wake probe: how long after an mbarrier phase completes does a warp
// that is *suspended* in mbarrier.try_wait (the SYNCS.PHASECHK.TRANS64.TRYWAIT
// + BRA loop every pipelined kernel uses) resume?
//
// One CTA, 2 warps.  Per repetition:
//   both warps: bar.sync (jitter-free start), read clock64 base
//   warp 1 lane 0 (waiter):   waits on [X] parity ph in the chosen form, then
//                             t_w = clock64 (first instruction after the loop)
//   warp 0 lane 0 (arriver):  spins until clock64 >= base + D, then
//                             t_a = clock64; completes the phase (plain arrive,
//                             or arrive.expect_tx 16 + complete_tx 16)
//   wake = t_w - t_a (cycles); D in cycles chooses where in the waiter's
//   suspend the completion lands.
// Waiter forms:  trywait  = mbarrier.try_wait.parity (default suspend hint),
//                trywait_hint = try_wait.parity with suspendTimeHint 10 ms
//                (ptxas: TRYWAIT -> NANOSLEEP.SYNCS -> PHASECHK loop),
//                testwait = mbarrier.test_wait.parity spin (control: no suspend).
// jsonl row per cell (blackwell.mechanism_curve_point.v1), samples = wake cycles.
#include "../../tma_cold/mb.cuh"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

__device__ __forceinline__ void wait_trywait(std::uint32_t mbar, std::uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "TW_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra TW_%=;\n\t}\n" ::"r"(mbar), "r"(parity) : "memory");
}
__device__ __forceinline__ void wait_trywait_hint(std::uint32_t mbar, std::uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "TWH_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra TWH_%=;\n\t}\n" ::"r"(mbar), "r"(parity), "r"(10000000u) : "memory");
}
__device__ __forceinline__ void wait_testwait(std::uint32_t mbar, std::uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "TSW_%=:\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra TSW_%=;\n\t}\n" ::"r"(mbar), "r"(parity) : "memory");
}
__device__ __forceinline__ void arrive_expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void complete_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

struct Out { long long t_a, t_w; };

// form: 0 trywait, 1 trywait_hint, 2 testwait; completion: 0 arrive, 1 tx
__global__ void __launch_bounds__(64, 1) k_wake(int form, int completion, long long D, int reps, Out* out) {
  __shared__ __align__(8) std::uint64_t mbar;
  const std::uint32_t mb = mb_smem_u32(&mbar);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) { mb_mbar_init(mb, 1); mb_fence_mbar_init(); }
  __syncthreads();
  unsigned parity = 0;
  for (int r = 0; r < reps; ++r) {
    __syncthreads();
    const long long base = (long long)mb_clock();
    if (warp == 1) {
      if (lane == 0) {
        if (form == 0) wait_trywait(mb, parity);
        else if (form == 1) wait_trywait_hint(mb, parity);
        else wait_testwait(mb, parity);
        out[r].t_w = (long long)mb_clock();
      }
    } else if (lane == 0) {
      while ((long long)mb_clock() < base + D) { }
      const long long t_a = (long long)mb_clock();
      if (completion == 0) {
        mb_mbar_arrive(mb);
      } else {
        arrive_expect_tx(mb, 16);
        complete_tx(mb, 16);
      }
      out[r].t_a = t_a;
    }
    __syncwarp();
    parity ^= 1u;
    __syncthreads();
  }
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

int main(int argc, char** argv) {
  std::string out = "trywait_wake.jsonl", form = "trywait", completion = "arrive";
  long long D = 400; int reps = 256, launches = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--out") out = next();
    else if (a == "--form") form = next();
    else if (a == "--completion") completion = next();
    else if (a == "--delay") D = std::atoll(next());
    else if (a == "--reps") reps = std::atoi(next());
    else if (a == "--launches") launches = std::atoi(next());
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  const int f = form == "trywait" ? 0 : form == "trywait_hint" ? 1 : form == "testwait" ? 2 : -1;
  const int c = completion == "arrive" ? 0 : completion == "tx" ? 1 : -1;
  if (f < 0 || c < 0) { std::fprintf(stderr, "form trywait|trywait_hint|testwait, completion arrive|tx\n"); return 2; }
  const MbDevice dev = mb_query_device();
  Out* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(Out) * reps));
  std::vector<double> wake, wake_abs;
  std::string err;
  for (int l = 0; l < launches; ++l) {
    MB_CHECK(cudaMemset(d_out, 0, sizeof(Out) * reps));
    k_wake<<<1, 64>>>(f, c, D, reps, d_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    std::vector<Out> h(reps);
    MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(Out) * reps, cudaMemcpyDeviceToHost));
    for (int r = 1; r < reps; ++r) {  // rep 0 warms
      wake.push_back(static_cast<double>(h[r].t_w - h[r].t_a));
    }
  }
  FILE* fo = std::fopen(out.c_str(), "a");
  if (!fo) { std::fprintf(stderr, "cannot append %s\n", out.c_str()); return 1; }
  std::fprintf(fo,
               "{\"schema\":\"blackwell.mechanism_curve_point.v1\",\"family\":\"trywait_wake.%s.%s.D%lld\","
               "\"quantity\":\"latency\",\"cta_group\":1,\"x_name\":\"delay_cycles\",\"x\":%lld,"
               "\"y_name\":\"wake_minus_complete_cycles\",\"y_unit\":\"cycles\","
               "\"median\":%.2f,\"p10\":%.2f,\"p90\":%.2f,\"mean\":%.2f,\"n\":%zu,\"holdout\":false,"
               "\"form\":\"%s\",\"completion\":\"%s\",\"delay\":%lld,\"reps\":%d,\"launches\":%d,"
               "\"source\":\"trywait_wake.measured\",\"gpu_name\":\"%s\",\"sm_clock_khz\":%d",
               form.c_str(), completion.c_str(), D, D, pct(wake, 0.5), pct(wake, 0.1), pct(wake, 0.9),
               wake.empty() ? 0.0 : std::accumulate(wake.begin(), wake.end(), 0.0) / wake.size(), wake.size(),
               form.c_str(), completion.c_str(), D, reps, launches, dev.name.c_str(), dev.sm_clock_khz);
  if (!err.empty()) std::fprintf(fo, ",\"error\":\"%s\"", err.c_str());
  std::fprintf(fo, "}\n");
  std::fclose(fo);
  std::printf("trywait_wake %s %s D=%lld wake p10/50/90 = %.0f / %.0f / %.0f%s\n", form.c_str(), completion.c_str(), D,
              pct(wake, 0.1), pct(wake, 0.5), pct(wake, 0.9), err.empty() ? "" : " ERR");
  cudaFree(d_out);
  return err.empty() ? 0 : 1;
}
