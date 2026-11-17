// memory.constant_cache (crossings protocol-v2, replaces the v1 ldc_chain):
// dependent LDC pointer chase through a 64 KiB __constant__ table holding a
// random cyclic permutation over positions 0, S, 2S, ... < F.  The next
// offset comes only from the loaded value (shl 2 + and 0xFFFF) and the
// constant address is formed in a register (mov.u64 t, g_tab; add), so the
// SASS is LDC R, c[0x3][R] register-indexed.
//
// Per launch the host rewrites the table (cudaMemcpyToSymbol), so rep 0 is
// the cold first traversal and reps >= 1 are warm.  Rows:
//   ldc.s<S>.f<F>.cold  u = U  samples = rep 0 of each launch (8)
//   ldc.s<S>.f<F>.warm  u = U  samples = reps >= 1 of all launches
// U in {64, 256}; S in {4, 64, 128, 256} B; F in {256 B .. 64 KiB}; cells
// with F/S < 2 are skipped (no room for a cycle distinct from stride order).
#include "../mb.cuh"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

constexpr int kElems = 16384;  // 64 KiB __constant__ limit
__constant__ std::uint32_t g_tab[kElems];

template <int U>
__global__ void __launch_bounds__(32, 1) k_ldc(std::uint64_t* out, int reps) {
  __shared__ std::uint32_t slot[32];
  if (threadIdx.x != 0) return;
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    // start offset 0 pinned through a volatile shared slot per rep: an
    // immediate 0 made the chain loop-invariant (b300-r1: ptxas hoisted it
    // out of the clock window and used the uniform LDCU path)
    std::uint64_t off = mb_pin(0u, &slot[threadIdx.x]);  // byte offset into g_tab, carried by the chain
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      asm volatile(
          "{\n\t.reg .u64 t;\n\t.reg .u32 v;\n\t"
          "mov.u64 t, g_tab;\n\t"
          "add.u64 t, t, %1;\n\t"
          "ld.const.u32 v, [t];\n\t"
          "mul.wide.u32 %0, v, 4;\n\t"
          "and.b64 %0, %0, 0xFFFF;\n\t}"
          : "=l"(off)
          : "l"(off)
          : "memory");
    }
    const std::uint64_t t1 = mb_clock();
    out[r] = t1 - t0;
    out[64 + r] = off;  // keep the chain alive
  }
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "ldc_chain");
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 128));
  std::mt19937_64 rng(0x9e3779b97f4a7c15ull);

  const int strides[] = {4, 64, 128, 256};
  const int footprints[] = {256, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  std::vector<std::uint32_t> host(kElems, 0);

  for (int s : strides) {
    for (int f : footprints) {
      const int links = f / s;
      if (links < 2 || f > kElems * 4) continue;
      const std::string base = "ldc.s" + std::to_string(s) + ".f" + std::to_string(f);
      if (!mb_selected(args, base + ".warm") && !mb_selected(args, base + ".cold")) continue;
      // random single cycle over the F/S element positions (stride S bytes)
      std::vector<std::uint32_t> order(links);
      std::iota(order.begin(), order.end(), 0u);
      std::shuffle(order.begin(), order.end(), rng);
      std::fill(host.begin(), host.end(), 0);
      for (int k = 0; k < links; ++k)
        host[order[k] * (s / 4)] = order[(k + 1) % links] * (s / 4);
      const std::string params = "\"stride_bytes\":" + std::to_string(s) +
                                 ",\"footprint_bytes\":" + std::to_string(f);
      for (int u : {64, 256}) {
        auto kern = u == 64 ? &k_ldc<64> : &k_ldc<256>;
        std::vector<double> cold, warm;
        for (int l = 0; l < args.launches; ++l) {
          // rewrite the table every launch: rep 0 is a fresh cold traversal
          MB_CHECK(cudaMemcpyToSymbol(g_tab, host.data(), sizeof(std::uint32_t) * kElems));
          kern<<<1, 32>>>(d_out, args.reps);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(128);
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
          if (args.reps > 0) cold.push_back(static_cast<double>(h[0]));
          for (int r = 1; r < args.reps; ++r) warm.push_back(static_cast<double>(h[r]));
        }
        w.row(base + ".cold", u, params, cold);
        w.row(base + ".warm", u, params, warm);
      }
    }
  }
  return w.failures ? 1 : 0;
}
