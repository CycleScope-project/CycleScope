// tools/microbench/row_gather/probe/row_gather.cu -- protocol.json of this directory.
// single: one 128-bit load per iteration; lane l reads row g = l % L at chunk
//   c = (l / L) % C.  spread far: row g is a random 4-KiB slot of the table;
//   adjacent: rows are consecutive 128-byte lines inside one random slot.
// block: the flash-MLA gather's first block -- 8 rows per warp (row = lane % 8,
//   chunk = lane / 8), row index random per (cta, warp, iteration, row), row
//   stride 656 B; loads at +0x200 (L2::128B) then +0x0..+0x140 step 0x40
//   (L2::256B); t_first after the use of the second load, t_all after all seven.
// Passes inside one launch: pass 0 (DRAM, table pushed out of the L2 by the
//   host-side k_evict over --evict-mib before every launch), then every warp
//   reads its own --spoil-mib region of a spoiler buffer (L1 spoiler), then
//   pass 1 repeats the same addresses (L2).
//   row_gather --mode single|block --hint plain|kernel --warps W --rows L
//       --chunk C --spread far|adjacent --grid G --iters 32 --launches 5
//       --table-mib 1024 --evict-mib 512 --spoil-mib 4
// stdout: META line; per launch and pass
//   "IT <launch> <pass> <it> n <samples> <p10> <p50> <p90>"   (single)
//   "ITF ..." / "ITA ..."                                    (block)
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
static const char* arg(int c, char** v, const char* n, const char* d) { for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], n)) return v[i + 1]; return d; }
__device__ __forceinline__ unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned h32(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
struct V4 { unsigned x, y, z, w; };
__device__ __forceinline__ V4 ld_plain(const char* p) {
  V4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ V4 ld_ltc256(const char* p) {
  V4 v;
  asm volatile("ld.global.nc.L1::evict_last.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ V4 ld_ltc128(const char* p) {
  V4 v;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void spoil_l1(const unsigned* spoil, size_t spoil_bytes, unsigned gw, unsigned& acc) {
  const unsigned* s = spoil + (size_t)gw * (spoil_bytes / 4);
  for (size_t i = 0; i < spoil_bytes / 4; i += 8) acc += s[i];   // 32 B stride
}
template <int HINT>
__device__ __forceinline__ void single_body(const char* table, size_t table_bytes,
                                            unsigned L, unsigned C, int adjacent,
                                            const unsigned* spoil, size_t spoil_bytes,
                                            int iters, unsigned long long* rec, unsigned* sink,
                                            int active_cta, int shared_slots, unsigned* smid) {
  if (active_cta >= 0 && (int)blockIdx.x != active_cta) return;
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const unsigned gw = blockIdx.x * nw + warp;
  const unsigned slots = (unsigned)(table_bytes >> 12);
  if (smid && lane == 0) {
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smid[gw] = sm + 1;
  }
  unsigned acc = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (int it = 0; it < iters; ++it) {
      const unsigned g = lane % L, c = (lane / L) % C;
      const char* p;
      if (adjacent) {
        const unsigned slot = h32(blockIdx.x * 7919u + warp * 104729u + (unsigned)it * 15485863u) % slots;
        p = table + (size_t)slot * 4096 + g * 128 + c * 16;
      } else {
        const unsigned h = shared_slots ? ((unsigned)it * 15485863u + g * 40503u)
                                        : (blockIdx.x * 7919u + warp * 104729u + (unsigned)it * 15485863u + g * 40503u);
        const unsigned slot = h32(h) % slots;
        p = table + (size_t)slot * 4096 + c * 16;
      }
      __syncwarp();
      const unsigned long long t0 = clk();
      const V4 v = HINT ? ld_ltc256(p) : ld_plain(p);
      acc += (v.x ^ v.w) + (v.y ^ v.z);   // all 4 words used -> ptxas keeps .128
      const unsigned long long t1 = clk();
      if (lane == 0) rec[((size_t)gw * 2 + pass) * iters + it] = t1 - t0;
    }
    if (pass == 0) spoil_l1(spoil, spoil_bytes, gw, acc);
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}
template <int HINT>
__device__ __forceinline__ void block_body(const char* table, size_t table_bytes,
                                           const unsigned* spoil, size_t spoil_bytes,
                                           int iters, unsigned long long* recf,
                                           unsigned long long* reca, unsigned* sink) {
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const unsigned gw = blockIdx.x * nw + warp;
  const unsigned row = lane & 7, chunk = lane >> 3;
  const unsigned nidx = (unsigned)(table_bytes / 656 - 2);
  unsigned acc = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (int it = 0; it < iters; ++it) {
      const unsigned idx = h32(blockIdx.x * 7919u + warp * 104729u + (unsigned)it * 15485863u + row * 40503u) % nidx;
      const char* p = table + (size_t)idx * 656 + chunk * 16;
      __syncwarp();
      const unsigned long long t0 = clk();
      const V4 v0 = HINT ? ld_ltc128(p + 0x200) : ld_plain(p + 0x200);
      const V4 v1 = HINT ? ld_ltc256(p + 0x000) : ld_plain(p + 0x000);
      const V4 v2 = HINT ? ld_ltc256(p + 0x040) : ld_plain(p + 0x040);
      const V4 v3 = HINT ? ld_ltc256(p + 0x080) : ld_plain(p + 0x080);
      const V4 v4 = HINT ? ld_ltc256(p + 0x0c0) : ld_plain(p + 0x0c0);
      const V4 v5 = HINT ? ld_ltc256(p + 0x100) : ld_plain(p + 0x100);
      const V4 v6 = HINT ? ld_ltc256(p + 0x140) : ld_plain(p + 0x140);
      acc += (v1.x ^ v1.w) + (v1.y ^ v1.z);   // second load only -> t_first
      const unsigned long long t1 = clk();
      acc += (v0.x ^ v0.w) + (v0.y ^ v0.z) + (v2.x ^ v2.w) + (v2.y ^ v2.z)
           + (v3.x ^ v3.w) + (v3.y ^ v3.z) + (v4.x ^ v4.w) + (v4.y ^ v4.z)
           + (v5.x ^ v5.w) + (v5.y ^ v5.z) + (v6.x ^ v6.w) + (v6.y ^ v6.z);
      const unsigned long long t2 = clk();
      if (lane == 0) {
        const size_t o = ((size_t)gw * 2 + pass) * iters + it;
        recf[o] = t1 - t0;
        reca[o] = t2 - t0;
      }
    }
    if (pass == 0) spoil_l1(spoil, spoil_bytes, gw, acc);
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}
__global__ void k_single_plain(const char* t, size_t tb, unsigned L, unsigned C, int adj,
                               const unsigned* s, size_t sb, int it, unsigned long long* r, unsigned* o,
                               int active_cta, int shared_slots, unsigned* smid) {
  single_body<0>(t, tb, L, C, adj, s, sb, it, r, o, active_cta, shared_slots, smid);
}
__global__ void k_single_hint(const char* t, size_t tb, unsigned L, unsigned C, int adj,
                              const unsigned* s, size_t sb, int it, unsigned long long* r, unsigned* o) {
  single_body<1>(t, tb, L, C, adj, s, sb, it, r, o, -1, 0, nullptr);
}
__global__ void k_block_plain(const char* t, size_t tb, const unsigned* s, size_t sb, int it,
                              unsigned long long* rf, unsigned long long* ra, unsigned* o) {
  block_body<0>(t, tb, s, sb, it, rf, ra, o);
}
__global__ void k_block_hint(const char* t, size_t tb, const unsigned* s, size_t sb, int it,
                             unsigned long long* rf, unsigned long long* ra, unsigned* o) {
  block_body<1>(t, tb, s, sb, it, rf, ra, o);
}
// run 2, pair cells: row A = the run-1 slot (hash of the iteration only),
// row B = A + delta; every lane reads the same 16 bytes; two waits recorded.
__global__ void k_pair(const char* table, size_t table_bytes, unsigned long long delta,
                       int iters, unsigned long long* rec, unsigned* sink) {
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const unsigned gw = blockIdx.x * nw + warp;
  const unsigned slots = (unsigned)(table_bytes >> 12);
  unsigned acc = 0;
  for (int it = 0; it < iters; ++it) {
    const unsigned slot = h32((unsigned)it * 15485863u) % slots;
    const char* pa = table + (size_t)slot * 4096;
    const char* pb = table + ((((size_t)slot * 4096 + delta) % table_bytes) & ~(size_t)15);
    __syncwarp();
    const unsigned long long t0 = clk();
    const V4 va = ld_plain(pa);
    acc += (va.x ^ va.w) + (va.y ^ va.z);
    const unsigned long long t1 = clk();
    const V4 vb = ld_plain(pb);
    acc += (vb.x ^ vb.w) + (vb.y ^ vb.z);
    const unsigned long long t2 = clk();
    if (lane == 0) {
      const size_t o = ((size_t)gw * iters + it) * 2;
      rec[o] = t1 - t0;
      rec[o + 1] = t2 - t1;
    }
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}
__global__ void k_evict(const uint4* buf, size_t n, unsigned* out) {
  unsigned s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += buf[i].x;
  if (threadIdx.x == 0) out[blockIdx.x] = s;
}
static void pct_line(const char* tag, int l, int pass, int it, std::vector<unsigned long long>& v) {
  std::sort(v.begin(), v.end());
  std::printf("%s %d %d %d n %zu %llu %llu %llu\n", tag, l, pass, it, v.size(),
              v[v.size() / 10], v[v.size() / 2], v[v.size() * 9 / 10]);
}
int main(int argc, char** argv) {
  const char* mode = arg(argc, argv, "--mode", "single");
  const char* hint = arg(argc, argv, "--hint", "plain");
  const char* spread = arg(argc, argv, "--spread", "far");
  const int W = atoi(arg(argc, argv, "--warps", "1"));
  const unsigned L = (unsigned)atoi(arg(argc, argv, "--rows", "8"));
  const unsigned C = (unsigned)atoi(arg(argc, argv, "--chunk", "1"));
  const int I = atoi(arg(argc, argv, "--iters", "32"));
  const int N = atoi(arg(argc, argv, "--launches", "5"));
  const size_t table_mib = (size_t)atoi(arg(argc, argv, "--table-mib", "1024"));
  const size_t evict = (size_t)atoi(arg(argc, argv, "--evict-mib", "512")) << 20;
  const size_t spoil = (size_t)atoi(arg(argc, argv, "--spoil-mib", "4")) << 20;
  const int single = !std::strcmp(mode, "single");
  const int pair = !std::strcmp(mode, "pair");
  const int khint = !std::strcmp(hint, "kernel");
  const int adjacent = !std::strcmp(spread, "adjacent");
  const int active_cta = atoi(arg(argc, argv, "--active-cta", "-1"));
  const int shared_slots = atoi(arg(argc, argv, "--shared-slots", "0"));
  const unsigned long long delta = strtoull(arg(argc, argv, "--delta", "0"), nullptr, 0);
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int G = atoi(arg(argc, argv, "--grid", "0")) ? atoi(arg(argc, argv, "--grid", "0")) : pr.multiProcessorCount;
  const size_t tb = table_mib << 20;
  char* table; unsigned* sbuf; unsigned long long* rec; unsigned* sink; unsigned* eout; uint4* ebuf; unsigned* smid;
  CK(cudaMalloc(&table, tb)); CK(cudaMemset(table, 1, tb));
  CK(cudaMalloc(&ebuf, evict)); CK(cudaMemset(ebuf, 2, evict));
  CK(cudaMalloc(&sbuf, (size_t)G * W * spoil)); CK(cudaMemset(sbuf, 3, (size_t)G * W * spoil));
  CK(cudaMalloc(&smid, (size_t)G * W * 4)); CK(cudaMemset(smid, 0, (size_t)G * W * 4));
  const size_t nrec = (size_t)G * W * 2 * I;
  CK(cudaMalloc(&rec, nrec * 16)); CK(cudaMalloc(&sink, G * 4)); CK(cudaMalloc(&eout, 4096 * 4));
  int khz = 0; CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  std::printf("META device=\"%s\" sms=%d clock_khz=%d mode=%s hint=%s warps=%d rows=%u chunk=%u spread=%s grid=%d iters=%d launches=%d table_mib=%zu evict_mib=%zu spoil_mib=%zu active_cta=%d shared_slots=%d delta=%llu\n",
              pr.name, pr.multiProcessorCount, khz, mode, hint, W, L, C, spread, G, I, N, table_mib, evict >> 20, spoil >> 20, active_cta, shared_slots, delta);
  std::vector<unsigned long long> h(2 * nrec);
  // run 3 (side map): one launch per active CTA of an all-SM grid,
  // L2 evicted before each, three sweeps; device code unchanged.
  if (!std::strcmp(mode, "sweep")) {
    for (int sw = 0; sw < 3; ++sw) {
      for (int c = 0; c < G; ++c) {
        k_evict<<<4096, 256>>>(ebuf, evict / 16, eout);
        CK(cudaDeviceSynchronize());
        CK(cudaMemset(smid, 0, (size_t)G * W * 4));
        k_single_plain<<<G, 32 * W>>>(table, tb, L, C, adjacent, sbuf, spoil, I,
                                      rec, sink, c, 1, smid);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), rec, nrec * 8, cudaMemcpyDeviceToHost));
        std::vector<unsigned> hs((size_t)G * W);
        CK(cudaMemcpy(hs.data(), smid, hs.size() * 4,
                      cudaMemcpyDeviceToHost));
        for (size_t w = 0; w < hs.size(); ++w) {
          if (!hs[w]) continue;
          const unsigned cta = (unsigned)(w / W);
          for (int pass = 0; pass < 2; ++pass)
            for (int it = 0; it < I; ++it)
              std::printf("RAWS %d %d %u %u %d %llu\n", sw, pass, cta,
                          hs[w] - 1, it, h[(w * 2 + pass) * I + it]);
        }
      }
    }
    return 0;
  }
  for (int l = 0; l < N; ++l) {
    k_evict<<<4096, 256>>>(ebuf, evict / 16, eout); CK(cudaDeviceSynchronize());
    if (pair) {
      k_pair<<<G, 32 * W>>>(table, tb, delta, I, rec, sink);
    } else if (single) {
      if (khint) k_single_hint<<<G, 32 * W>>>(table, tb, L, C, adjacent, sbuf, spoil, I, rec, sink);
      else       k_single_plain<<<G, 32 * W>>>(table, tb, L, C, adjacent, sbuf, spoil, I, rec, sink,
                                             active_cta, shared_slots, smid);
    } else {
      if (khint) k_block_hint<<<G, 32 * W>>>(table, tb, sbuf, spoil, I, rec, rec + nrec, sink);
      else       k_block_plain<<<G, 32 * W>>>(table, tb, sbuf, spoil, I, rec, rec + nrec, sink);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if (!single) CK(cudaMemcpy(h.data(), rec, nrec * 16, cudaMemcpyDeviceToHost));
    else CK(cudaMemcpy(h.data(), rec, nrec * 8, cudaMemcpyDeviceToHost));
    const size_t nw = (size_t)G * W;
    if (pair) {
      for (int it = 0; it < I; ++it)
        std::printf("RAWP %d %d %llu %llu\n", l, it, h[it * 2], h[it * 2 + 1]);
      continue;
    }
    if (active_cta >= 0 || shared_slots) {
      std::vector<unsigned> hs(nw);
      CK(cudaMemcpy(hs.data(), smid, nw * 4, cudaMemcpyDeviceToHost));
      for (size_t c = 0; c < nw; ++c) {
        if (!hs[c]) continue;
        const unsigned cta = (unsigned)(c / W);
        for (int pass = 0; pass < 2; ++pass)
          for (int it = 0; it < I; ++it)
            std::printf("RAWM %d %d %u %u %d %llu\n", l, pass, cta, hs[c] - 1, it, h[(c * 2 + pass) * I + it]);
      }
      continue;
    }
    for (int pass = 0; pass < 2; ++pass) {
      for (int it = 0; it < I; ++it) {
        if (single) {
          std::vector<unsigned long long> v;
          for (size_t c = 0; c < nw; ++c) v.push_back(h[(c * 2 + pass) * I + it]);
          pct_line("IT", l, pass, it, v);
        } else {
          std::vector<unsigned long long> vf, va;
          for (size_t c = 0; c < nw; ++c) {
            const size_t o = (c * 2 + pass) * I + it;
            vf.push_back(h[o]);
            va.push_back(h[nrec + o]);
          }
          pct_line("ITF", l, pass, it, vf);
          pct_line("ITA", l, pass, it, va);
        }
      }
    }
  }
  return 0;
}
