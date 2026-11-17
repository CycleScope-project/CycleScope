// cp.async mbarrier arrival delay: synchronization.cp_async.arrival (timing
// kind 18, op class ARRIVES).
//
// Engine (engine.cpp ~4249): an ARRIVES arrival becomes visible to the
// mbarrier at issue + kind 18, independent of any LDGSTS completion.  With no
// cp.async pending, the hardware arrive-on fires as soon as the (empty) group
// completes, so the delay is the only difference from a plain arrive.
// Cross-warp, jittered phase, test_wait spin waiter (same protocol as tma.cu):
//   warp 0 lane 0: delay ; t_s ; MODE 0: mbarrier.arrive [A]
//                               MODE 1: cp.async.mbarrier.arrive.noinc [A]
//   warp 1 lane 0: test_wait spin [A] ; t_w ; arrive ACK
//   kind18 = E[t_w - t_s | MODE 1] - E[t_w - t_s | MODE 0]   (async_rules.arrival_delay)
// Launch l probes from CTA (41 l) mod #SMs of a one-wave grid (the rest exit at
// once), so the samples cover many SMs; %smid recorded.
#include "../mb.cuh"

#include <string>
#include <vector>

constexpr std::uint32_t kXwDelay = 3000;

// ACK arrive in a different SASS form than the measured plain arrive (as in tma.cu)
__device__ __forceinline__ void arrive_cluster(std::uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}

template <int MODE>
__global__ void __launch_bounds__(64, 1) k_arr(std::uint64_t* out, int reps, std::uint32_t seed, int probe_cta,
                                               std::uint32_t* smid_out) {
  if (static_cast<int>(blockIdx.x) != probe_cta) return;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar_s[0])), ack = mb_opaque(mb_smem_u32(&mbar_s[1]));
  if (threadIdx.x == 0) {
    *smid_out = mb_smid();
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
  }
  __syncthreads();
  std::uint32_t ph = 0;
  if (lane == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t d = kXwDelay + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < d) {
      }
      const std::uint64_t t_s = mb_clock();
      if constexpr (MODE == 0) {
        mb_mbar_arrive(a);
      } else {
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
      }
      const std::uint64_t t_e = mb_clock();
      out[3 * rep + 0] = t_s;
      out[3 * rep + 2] = t_e;
      mb_mbar_wait_to(ack, ph);
      ph ^= 1u;
    }
  } else if (lane == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[3 * rep + 1] = mb_clock();
      ph ^= 1u;
      arrive_cluster(ack);
    }
  }
  __syncwarp();
  __syncthreads();
}

using Kern = void (*)(std::uint64_t*, int, std::uint32_t, int, std::uint32_t*);

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "async");
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  std::uint64_t* d_out = nullptr;
  std::uint32_t *d_smid = nullptr, *d_err = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 3 * args.reps));
  MB_CHECK(cudaMalloc(&d_smid, sizeof(std::uint32_t)));
  MB_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&d_err), g_mb_err));
  const struct {
    const char* name;
    Kern f;
  } cases[] = {{"arr.arrive", &k_arr<0>}, {"arr.cpasync_noinc", &k_arr<1>}};
  for (const auto& c : cases) {
    if (!mb_selected(args, c.name)) continue;
    std::vector<double> wait, issue;
    std::string smids;
    for (int l = 0; l < args.launches; ++l) {
      MB_CHECK(cudaMemset(d_err, 0, sizeof(std::uint32_t)));
      const int probe = (41 * l) % nsm;
      c.f<<<nsm, 64>>>(d_out, args.reps, 0x9e3779b9u * static_cast<std::uint32_t>(l + 1), probe, d_smid);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::uint32_t err = 0, smid = 0;
      MB_CHECK(cudaMemcpy(&err, d_err, 4, cudaMemcpyDeviceToHost));
      MB_CHECK(cudaMemcpy(&smid, d_smid, 4, cudaMemcpyDeviceToHost));
      if (err) {
        std::fprintf(stderr, "[async] %s launch %d: wait timed out\n", c.name, l);
        ++w.failures;
        continue;
      }
      std::vector<std::uint64_t> h(3 * static_cast<std::size_t>(args.reps));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r) {
        wait.push_back(static_cast<double>(h[3 * r + 1] - h[3 * r + 0]));
        issue.push_back(static_cast<double>(h[3 * r + 2] - h[3 * r + 0]));
      }
      smids += (smids.empty() ? "" : ",") + std::to_string(smid);
    }
    w.row(c.name, 0, "\"metric\":\"t_w-t_s\",\"smids\":[" + smids + "]", wait);
    w.row(std::string(c.name) + ".issue", 0, "\"metric\":\"t_e-t_s\"", issue);
  }
  return w.failures ? 1 : 0;
}
