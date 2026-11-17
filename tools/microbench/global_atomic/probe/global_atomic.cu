/*
 * global_atomic.cu -- global atomic throughput / latency and
 * release->acquire visibility latency across SMs (M3; see
 * tools/microbench/global_atomic_protocol.json).  Port of hopper-prototype
 * tools/microbench/global_atomic_microbench.cu to sm_103a: one CTA per SM,
 * nsm <= 148, __launch_bounds__(1024,1).  One kernel per form
 * (template<int FORM>) so the SASS gate can separate loop bodies.
 * New form chase_l2: one CTA, one lane, dependent 128 B pointer chase over
 * a warmed 4 MiB region -> L2-hit load latency in cycles/step.
 *
 * usage: global_atomic <form> <nsm> <threads> <iters|steps> <trials> [peer=N]
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)



enum Form { RED_SAME = 0, RED_WARP, RED_LANE, ATOM_SAME, ATOM_CHAIN, PINGPONG, CHASE_L2, ATOM_CHAIN_ADDR, PINGPONG_ADDR };

struct Rec { unsigned smid, pad; unsigned long long clk0, clk1, gt0, gt1, ops; };

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid_reg() {
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ void red_add(unsigned* p, unsigned v) {
  asm volatile("red.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_add(unsigned* p, unsigned v) {
  unsigned old; asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old; }
__device__ __forceinline__ void st_release(volatile unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned ld_acquire(volatile unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_cg(unsigned* p) {
  unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

// pingpong_addr (protocol amendment 2): flag pair at explicit buffer
// offsets so the one-way exchange latency can be swept over placements.
__global__ void __launch_bounds__(32) pingpong_addr_kernel(
    unsigned* flag_a, unsigned* flag_b, unsigned iters, int peer,
    Rec* rec, unsigned* sink) {
  if ((int)blockIdx.x != 0 && (int)blockIdx.x != peer) return;
  if (threadIdx.x != 0) return;
  Rec& r = rec[blockIdx.x];
  const unsigned long long c0 = clock64(), g0 = gtimer();
  volatile unsigned* a = flag_a;
  volatile unsigned* b = flag_b;
  if (blockIdx.x == 0) {
#pragma unroll 1
    for (unsigned i = 1; i <= iters; ++i) { st_release(a, i); while (ld_acquire(b) != i) { } }
  } else {
#pragma unroll 1
    for (unsigned i = 1; i <= iters; ++i) { while (ld_acquire(a) != i) { } st_release(b, i); }
  }
  r.smid = smid_reg(); r.clk0 = c0; r.gt0 = g0;
  r.clk1 = clock64(); r.gt1 = gtimer(); r.ops = iters;
  if (sink[0] == 0xffffffffu) sink[blockIdx.x] = 1;
}

template <int FORM>
__global__ void __launch_bounds__(1024, 1) atomic_kernel(unsigned* buf, unsigned iters,
                                                         int peer, Rec* rec, unsigned* sink,
                                                         int maxsm) {
  Rec& r = rec[blockIdx.x];
  if (threadIdx.x == 0) { r.smid = smid_reg(); r.ops = 0; r.clk0 = r.clk1 = clock64(); r.gt0 = r.gt1 = gtimer(); }
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned* same = buf;                                              // one line
  unsigned* warp_addr = buf + 32 * (1 + blockIdx.x * 64 + warp);    // own line per warp
  unsigned* lane_addr = buf + 32 * (1 + 64 * maxsm + (blockIdx.x * 1024 + threadIdx.x));
  unsigned acc = 0;
  if (FORM == ATOM_CHAIN || FORM == PINGPONG || FORM == CHASE_L2) {
    if (FORM == PINGPONG && (int)blockIdx.x != 0 && (int)blockIdx.x != peer) return;
    if (threadIdx.x != 0) return;
    const unsigned long long c0 = clock64(), g0 = gtimer();
    if (FORM == ATOM_CHAIN) {
      unsigned v = blockIdx.x + 1;
#pragma unroll 1
      for (unsigned i = 0; i < iters; ++i) v = atom_add(same, v & 7u) + 1u;
      acc = v;
    } else if (FORM == PINGPONG) {
      volatile unsigned* flag_a = buf + 32 * 4;   // written by CTA 0, read by peer
      volatile unsigned* flag_b = buf + 32 * 8;   // written by peer, read by CTA 0
      if (blockIdx.x == 0) {
#pragma unroll 1
        for (unsigned i = 1; i <= iters; ++i) { st_release(flag_a, i); while (ld_acquire(flag_b) != i) { } }
      } else {
#pragma unroll 1
        for (unsigned i = 1; i <= iters; ++i) { while (ld_acquire(flag_a) != i) { } st_release(flag_b, i); }
      }
    } else {  // CHASE_L2: buf holds a pointer-chase layout over the region
      unsigned idx = 0;
#pragma unroll 1
      for (unsigned i = 0; i < iters; ++i) idx = ld_cg(buf + idx);
      acc = idx;
    }
    r.clk0 = c0; r.gt0 = g0; r.clk1 = clock64(); r.gt1 = gtimer(); r.ops = iters;
    if (acc == 0xffffffffu) sink[blockIdx.x] = acc;
    return;
  }
  __syncthreads();
  const unsigned long long c0 = clock64(), g0 = gtimer();
#pragma unroll 1
  for (unsigned i = 0; i < iters; ++i) {
    if (FORM == RED_SAME) red_add(same, 1u);
    else if (FORM == RED_WARP) red_add(warp_addr, 1u);
    else if (FORM == RED_LANE) red_add(lane_addr, 1u);
    else acc += atom_add(same, 1u);
  }
  __threadfence();
  const int any = __syncthreads_or(acc == 0xffffffffu);
  if (threadIdx.x == 0) { r.clk0 = c0; r.gt0 = g0; r.clk1 = clock64(); r.gt1 = gtimer();
    r.ops = (unsigned long long)iters * blockDim.x; if (any) sink[0] = 1; }
  (void)lane;
}

static double median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s <red_same|red_warp|red_lane|atom_same|atom_chain|pingpong|chase_l2> <nsm> <threads> <iters|steps> <trials> [peer=N]\n", argv[0]); return 1; }
  const std::string fs = argv[1];
  const char* names[] = {"red_same", "red_warp", "red_lane", "atom_same", "atom_chain", "pingpong", "chase_l2", "atom_chain_addr", "pingpong_addr"};
  int form = -1; for (int i = 0; i < 9; ++i) if (fs == names[i]) form = i;
  const int nsm = atoi(argv[2]), threads = atoi(argv[3]); const unsigned iters = strtoul(argv[4], nullptr, 10); const int trials = atoi(argv[5]);
  int peer = nsm > 1 ? 1 : 0; for (int i = 6; i < argc; ++i) if (!strncmp(argv[i], "peer=", 5)) peer = atoi(argv[i] + 5);
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int dev_sm = prop.multiProcessorCount;
  if (form < 0 || nsm < 1 || nsm > dev_sm || threads < 32 || threads > 1024 || threads % 32 || trials < 1
      || (form == PINGPONG && (peer <= 0 || peer >= nsm))) { fprintf(stderr, "bad arguments\n"); return 1; }
  unsigned* buf = nullptr; Rec* rec = nullptr; unsigned* sink = nullptr;
  // max(region 4 MiB, lane/warp/same table for the device SM count) elements
  const size_t region_elems = (4ull << 20) / 4;
  const size_t table_elems = 32ull * (1 + 64 * dev_sm + (size_t)dev_sm * 1024 + 16);
  const size_t buf_elems = std::max(region_elems, table_elems);
  CUDA_CHECK(cudaMalloc(&buf, buf_elems * 4)); CUDA_CHECK(cudaMalloc(&rec, nsm * sizeof(Rec))); CUDA_CHECK(cudaMalloc(&sink, 4096));
  printf("META form=%s nsm=%d threads=%d iters=%u trials=%d peer=%d gpu=%s\n", fs.c_str(), nsm, threads, iters, trials, peer, prop.name);
  std::vector<Rec> recs(nsm); std::vector<double> agg, cyc_per_op, mhz_med;
  // atom_chain_addr (protocol amendment 1): dependent atom chain on 64
  // distinct 128 B lines spread over 1 GiB (line k at k*(16 MiB + 128 B));
  // 1 warmup + <trials> timed launches per line, median per line.
  if (form == ATOM_CHAIN_ADDR) {
    const size_t step_elems = ((16ull << 20) + 128) / 4;   // 4194336
    const int n_lines = 64;
    unsigned* buf2 = nullptr;
    CUDA_CHECK(cudaMalloc(&buf2, (size_t)n_lines * step_elems * 4));
    CUDA_CHECK(cudaMemset(buf2, 0, (size_t)n_lines * step_elems * 4));
    std::vector<double> line_cy, line_ns;
    for (int k = 0; k < n_lines; ++k) {
      unsigned* line = buf2 + (size_t)k * step_elems;
      std::vector<double> cy, ns;
      for (int t = 0; t <= trials; ++t) {
        atomic_kernel<ATOM_CHAIN><<<1, 32>>>(line, iters, 0, rec, sink, dev_sm);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        if (!t) continue;   // warmup
        CUDA_CHECK(cudaMemcpy(recs.data(), rec, sizeof(Rec), cudaMemcpyDeviceToHost));
        const Rec& x = recs[0];
        if (!x.ops) continue;
        cy.push_back(double(x.clk1 - x.clk0) / iters);
        ns.push_back(double(x.gt1 - x.gt0) / iters);
      }
      const double mcy = median(cy), mns = median(ns);
      printf("LINE k=%d cycles_per_link=%.2f ns_per_link=%.2f\n", k, mcy, mns);
      if (cy.size()) { line_cy.push_back(mcy); line_ns.push_back(mns); }
    }
    printf("RESULT agg_ops_per_ns=0.0000 cyc_per_op=%.2f mhz=%.1f\n",
           median(line_cy),
           median(line_cy) > 0 && median(line_ns) > 0
               ? median(line_cy) / median(line_ns) * 1e3 : 0.0);
    CUDA_CHECK(cudaFree(buf2)); CUDA_CHECK(cudaFree(sink)); CUDA_CHECK(cudaFree(rec)); CUDA_CHECK(cudaFree(buf));
    return 0;
  }
  // pingpong_addr (protocol amendment 2): 16 flag placements on the same
  // 1 GiB sweep buffer; placement k -> flag_a at line 2k, flag_b at
  // line 2k+1 (line i at offset i*(16 MiB + 128 B)); peer=8, 1 warmup +
  // <trials> launches per placement; LINE = one-way (RT/2) cycles.
  if (form == PINGPONG_ADDR) {
    const size_t step_elems = ((16ull << 20) + 128) / 4;
    const int n_placements = 16;
    unsigned* buf2 = nullptr;
    CUDA_CHECK(cudaMalloc(&buf2, (size_t)64 * step_elems * 4));
    CUDA_CHECK(cudaMemset(buf2, 0, (size_t)64 * step_elems * 4));
    std::vector<double> ow_cy, ow_ns;
    for (int k = 0; k < n_placements; ++k) {
      unsigned* fa = buf2 + (size_t)(2 * k) * step_elems;
      unsigned* fb = buf2 + (size_t)(2 * k + 1) * step_elems;
      std::vector<double> cy, ns;
      for (int t = 0; t <= trials; ++t) {
        pingpong_addr_kernel<<<nsm, 32>>>(fa, fb, iters, peer, rec, sink);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        if (!t) continue;
        CUDA_CHECK(cudaMemcpy(recs.data(), rec, nsm * sizeof(Rec), cudaMemcpyDeviceToHost));
        const Rec& x = recs[0];
        if (!x.ops) continue;
        cy.push_back(double(x.clk1 - x.clk0) / (2.0 * iters));
        ns.push_back(double(x.gt1 - x.gt0) / (2.0 * iters));
      }
      const double mcy = median(cy), mns = median(ns);
      printf("LINE k=%d one_way_cycles=%.2f one_way_ns=%.2f\n", k, mcy, mns);
      if (cy.size()) { ow_cy.push_back(mcy); ow_ns.push_back(mns); }
    }
    printf("RESULT agg_ops_per_ns=0.0000 cyc_per_op=%.2f mhz=%.1f\n",
           median(ow_cy),
           median(ow_cy) > 0 && median(ow_ns) > 0
               ? median(ow_cy) / median(ow_ns) * 1e3 : 0.0);
    CUDA_CHECK(cudaFree(buf2)); CUDA_CHECK(cudaFree(sink)); CUDA_CHECK(cudaFree(rec)); CUDA_CHECK(cudaFree(buf));
    return 0;
  }
  // chase_l2 host-side pointer layout: each 128 B line's word 0 holds the
  // index of the next line to visit (strided walk over the whole region).
  if (form == CHASE_L2) {
    const unsigned stride_lines = 32;            // 4 KiB step -> covers 4 MiB in 1024 lines
    const unsigned lines = (unsigned)(region_elems / 32);
    std::vector<unsigned> h(region_elems, 0);
    std::vector<unsigned> order;
    for (unsigned i = 0; i < lines; ++i) order.push_back(i);
    // stride permutation so consecutive steps are far apart in the region
    for (unsigned i = 0; i < lines; ++i) h[order[i] * 32] = order[(i + stride_lines) % lines] * 32;
    CUDA_CHECK(cudaMemcpy(buf, h.data(), region_elems * 4, cudaMemcpyHostToDevice));
    // warm the region once before timing (protocol cells warm=1)
    atomic_kernel<CHASE_L2><<<1, 32>>>(buf, (unsigned)lines, 0, rec, sink, dev_sm);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  }
  void (*kern)(unsigned*, unsigned, int, Rec*, unsigned*, int) = nullptr;
  switch (form) {
    case RED_SAME: kern = atomic_kernel<RED_SAME>; break;
    case RED_WARP: kern = atomic_kernel<RED_WARP>; break;
    case RED_LANE: kern = atomic_kernel<RED_LANE>; break;
    case ATOM_SAME: kern = atomic_kernel<ATOM_SAME>; break;
    case ATOM_CHAIN: kern = atomic_kernel<ATOM_CHAIN>; break;
    case PINGPONG: kern = atomic_kernel<PINGPONG>; break;
    case CHASE_L2: kern = atomic_kernel<CHASE_L2>; break;
  }
  for (int t = 0; t <= trials; ++t) {
    if (form != CHASE_L2) { CUDA_CHECK(cudaMemset(buf, 0, buf_elems * 4)); }
    CUDA_CHECK(cudaMemset(rec, 0, nsm * sizeof(Rec)));
    kern<<<nsm, threads>>>(buf, iters, peer, rec, sink, dev_sm);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(recs.data(), rec, nsm * sizeof(Rec), cudaMemcpyDeviceToHost));
    if (t == 0) { std::vector<unsigned> sm; for (auto& x : recs) sm.push_back(x.smid); std::sort(sm.begin(), sm.end());
      printf("PLACEMENT distinct_sm=%zu nsm=%d\n", (size_t)(std::unique(sm.begin(), sm.end()) - sm.begin()), nsm); continue; }
    unsigned long long total = 0, gmin = ~0ull, gmax = 0; std::vector<double> cpo, mhz;
    for (int c = 0; c < nsm; ++c) { const Rec& x = recs[c]; if (!x.ops) continue; const double dc = double(x.clk1 - x.clk0), dn = double(x.gt1 - x.gt0);
      printf("CTA trial=%d cta=%d smid=%u cycles=%llu ns=%llu ops=%llu cyc_per_op=%.2f mhz=%.1f\n", t, c, x.smid, (unsigned long long)dc, (unsigned long long)dn, x.ops, dc / x.ops, dn > 0 ? dc / dn * 1e3 : 0.0);
      total += x.ops; gmin = std::min(gmin, x.gt0); gmax = std::max(gmax, x.gt1); cpo.push_back(dc / x.ops); if (dn > 0) mhz.push_back(dc / dn * 1e3); }
    const double window = gmax > gmin ? double(gmax - gmin) : 0.0, a = window > 0 ? total / window : 0.0;
    agg.push_back(a); cyc_per_op.push_back(median(cpo)); mhz_med.push_back(median(mhz));
    printf("TRIAL %d total_ops=%llu window_ns=%.0f agg_ops_per_ns=%.4f cyc_per_op_med=%.2f mhz_med=%.1f\n", t, total, window, a, median(cpo), median(mhz));
  }
  printf("RESULT agg_ops_per_ns=%.4f cyc_per_op=%.2f mhz=%.1f\n", median(agg), median(cyc_per_op), median(mhz_med));
  CUDA_CHECK(cudaFree(sink)); CUDA_CHECK(cudaFree(rec)); CUDA_CHECK(cudaFree(buf));
  return 0;
}
