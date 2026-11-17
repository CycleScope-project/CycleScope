// memory.l1.residency (crossings protocol-v3): one CTA; lane 0 runs the
// memlat dependent ld.global.ca.u32 chase over a footprint F of 128 B lines
// in random cyclic order (each line's first word holds the next line
// number).  The kernel touches the first and last dynamic-smem words so the
// carveout is real.
//
// Per rep: one full warm traversal (untimed), t0, one full traversal, t1.
// out[rep] = t1 - t0; u = F/128 hops.  F in 4 KiB steps from 4 KiB to
// 256 KiB; the derivation finds the knee F* per cell.
//
// Cells (host side; one variant prefix per cell):
//   c<C>k.s0k.b32    carveout pinned: PreferredSharedMemoryCarveout =
//                   ceil(100*C/228), C in {0,8,16,32,64,100,132,164,196,228}
//                   KiB, dynamic S = 0, block 32
//   c228k.s{64,200}k.b32 / c100k.s64k.b32
//                   smem-vs-carveout: pinned carveout with dynamic S
//   cdef.s<S>k.b<B> default-hint diagnostic: carveout attribute reset to
//                   cudaSharedmemCarveoutDefault (-1), S in
//                   {0,32,64,128,192} KiB x block in {32,1024}
// Every cell records the requested carveout percent, carveout KiB, S,
// block, the kernel's static shared bytes (cudaFuncGetAttributes
// sharedSizeBytes) and cudaOccupancyMaxActiveBlocksPerMultiprocessor taken
// after the attributes are set.
#include "../mb.cuh"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

__device__ __forceinline__ std::uint32_t ld_ca(const std::uint32_t* p) {
  std::uint32_t v;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// launch_bounds(1024): the default-hint cells run blocks of 1024 threads;
// every thread other than lane 0 is inert.
__global__ void __launch_bounds__(1024, 1) k_l1cap(const std::uint32_t* buf, std::uint64_t* out, int hops,
                                                   int reps, std::uint32_t smem_bytes, std::uint32_t* sink) {
  extern __shared__ unsigned char smem[];
  const int lane = threadIdx.x;
  std::uint32_t t = 0;
  if (smem_bytes && lane == 0) {
    // prove the carveout is real: touch the first and last dynamic-smem words
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(mb_smem_u32(smem)), "r"(lane + 1) : "memory");
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(t) : "r"(mb_smem_u32(smem)) : "memory");
    std::uint32_t v2;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v2) : "r"(mb_smem_u32(smem + smem_bytes - 4)) : "memory");
    t ^= v2;
  }
  if (lane == 0) {
    std::uint32_t v = 0;
    // base pointer pinned in registers through volatile shared slots:
    // ptxas otherwise reloads the kernel parameter (LDC.64) inside the
    // timed chase loop (b300-r1 SASS gate)
    __shared__ std::uint32_t slot[2];
    const std::uint64_t pb = reinterpret_cast<std::uint64_t>(buf);
    const std::uint64_t base = (static_cast<std::uint64_t>(mb_pin(static_cast<std::uint32_t>(pb >> 32), &slot[1])) << 32) |
                               mb_pin(static_cast<std::uint32_t>(pb), &slot[0]);
    const std::uint32_t* bufp = reinterpret_cast<const std::uint32_t*>(base);
#pragma unroll 1
    for (int r = 0; r < reps; ++r) {
#pragma unroll 1
      for (int h = 0; h < hops; ++h) v = ld_ca(bufp + static_cast<std::size_t>(v) * 32);  // warm pass
      const std::uint64_t t0 = mb_clock();
#pragma unroll 1
      for (int h = 0; h < hops; ++h) v = ld_ca(bufp + static_cast<std::size_t>(v) * 32);
      const std::uint64_t t1 = mb_clock();
      out[r] = t1 - t0;
      t ^= v;
    }
  }
  if (lane == 0) sink[0] = t;
}

static std::vector<std::uint32_t> make_random_cycle(std::size_t n, std::mt19937_64& rng) {
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<std::uint32_t> chain(n);
  for (std::size_t k = 0; k < n; ++k) chain[order[k]] = order[(k + 1) % n];
  return chain;
}

struct Cell {
  std::string name;
  int carveout_pct;   // -1 = default hint (cudaSharedmemCarveoutDefault)
  int carveout_kib;   // requested configuration, -1 for the default cells
  int smem_kib;
  int block;
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "l1cap");
  const int fmax = 256 << 10;
  std::uint32_t* buf = nullptr;
  MB_CHECK(cudaMalloc(&buf, fmax));
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, 4));
  MB_CHECK(cudaFuncSetAttribute(k_l1cap, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 << 10));
  std::mt19937_64 rng(0x243f6a8885a308d3ull);
  std::vector<std::uint32_t> host(fmax / 4, 0);

  auto cell_params = [&](const Cell& c, std::uint32_t sbytes) {
    cudaFuncAttributes fa{};
    MB_CHECK(cudaFuncGetAttributes(&fa, k_l1cap));
    int occ = 0;
    MB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_l1cap, c.block, sbytes));
    return "\"requested_carveout_pct\":" + std::to_string(c.carveout_pct) +
           ",\"carveout_kib\":" + std::to_string(c.carveout_kib) +
           ",\"smem_bytes\":" + std::to_string(sbytes) +
           ",\"block\":" + std::to_string(c.block) +
           ",\"static_shared_bytes\":" + std::to_string(fa.sharedSizeBytes) +
           ",\"occupancy_blocks\":" + std::to_string(occ);
  };

  auto run_cell = [&](const Cell& c) {
    if (c.carveout_pct >= 0)
      MB_CHECK(cudaFuncSetAttribute(k_l1cap, cudaFuncAttributePreferredSharedMemoryCarveout, c.carveout_pct));
    else
      MB_CHECK(cudaFuncSetAttribute(k_l1cap, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutDefault));
    const std::uint32_t sbytes = static_cast<std::uint32_t>(c.smem_kib) << 10;
    const std::string params = cell_params(c, sbytes);
    for (int f = 4; f <= 256; f += 4) {
      const int fbytes = f << 10, hops = fbytes / 128;
      const std::string variant = "l1cap." + c.name + ".f" + std::to_string(f) + "k";
      if (!mb_selected(args, variant)) continue;
      // random single cycle over the F/128 lines; line i's first word holds
      // the next line number
      const auto chain = make_random_cycle(hops, rng);
      std::fill(host.begin(), host.end(), 0);
      for (int i = 0; i < hops; ++i) host[i * 32] = chain[i];
      MB_CHECK(cudaMemcpy(buf, host.data(), fbytes, cudaMemcpyHostToDevice));
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        k_l1cap<<<1, c.block, sbytes>>>(buf, d_out, hops, args.reps, sbytes, d_sink);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        mb_collect(d_out, args.reps, 1, 0, samples);
      }
      w.row(variant, hops, params + ",\"footprint_bytes\":" + std::to_string(fbytes), samples);
    }
  };

  // carveout cells first (attribute set per cell), then the smem-vs-carveout
  // cells, then the default-hint diagnostic cells (attribute left at -1).
  std::vector<Cell> cells;
  for (int c : {0, 8, 16, 32, 64, 100, 132, 164, 196, 228}) {
    const int pct = (100 * c + 227) / 228;  // ceil(100*C/228)
    cells.push_back({"c" + std::to_string(c) + "k.s0k.b32", pct, c, 0, 32});
  }
  cells.push_back({"c228k.s64k.b32", 100, 228, 64, 32});
  cells.push_back({"c228k.s200k.b32", 100, 228, 200, 32});
  cells.push_back({"c100k.s64k.b32", (100 * 100 + 227) / 228, 100, 64, 32});
  for (const Cell& c : cells) run_cell(c);
  for (int s : {0, 32, 64, 128, 192})
    for (int b : {32, 1024})
      run_cell({"cdef.s" + std::to_string(s) + "k.b" + std::to_string(b), -1, -1, s, b});
  return w.failures ? 1 : 0;
}
