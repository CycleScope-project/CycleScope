// tools/microbench/alu_forms/probe/alu_forms.cu -- protocol.json of the same
// directory.  Harness of fp_stream_mix (one CTA of 4*W warps, warp w on SMSP
// w % 4, W in {1, 2, 4, 8}; window = U instructions fully unrolled on eight
// independent registers; bar.sync before every rep; lane 0 of every warp
// records t_s / t_w).  Integer window kinds (instruction I, register k = I & 7):
//   0 add_imm    d[k] = d[k] + 1                 (1 GPR source)
//   1 add_rr     d[k] = d[k] + a[k]              (2 distinct GPR sources)
//   2 add_rrr    d[k] = d[k] + a[k] + b[k]       (3 distinct GPR sources, C)
//   3 xor_rr     d[k] = d[k] ^ a[k]              (LOP3, 2 sources)
//   4 and_imm    d[k] = d[k] & 0x7fffffff        (LOP3, 1 source)
//   5 mad_rrr    d[k] = d[k] * a[k] + b[k]       (IMAD, 3 sources)
//   6 add_rs     d[k] = d[k] + s                 (2 sources, s the same register every time)
//   7 shl_imm    d[k] = d[k] << 1                (SHF, 1 source)
// Run 2 (addendum 1), visit j = I >> 3: 8 imm_ax (add 1 / xor 85), 9 rr_ax
// (add a[k] / xor b[k]), 10 imm_as (add 1 / shl 1)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

struct CellArgs { unsigned a[8], b[8], s; };

template <int V, int I>
__device__ __forceinline__ void unit(unsigned *d, const CellArgs &o) {
  const int k = I & 7;
  if constexpr (V == 0) {
    asm volatile("add.u32 %0, %0, 1;" : "+r"(d[k]));
  } else if constexpr (V == 1) {
    asm volatile("add.u32 %0, %0, %1;" : "+r"(d[k]) : "r"(o.a[k]));
  } else if constexpr (V == 2) {
    d[k] = d[k] + o.a[k] + o.b[k];
  } else if constexpr (V == 3) {
    asm volatile("xor.b32 %0, %0, %1;" : "+r"(d[k]) : "r"(o.a[k]));
  } else if constexpr (V == 4) {
    asm volatile("and.b32 %0, %0, 2147483647;" : "+r"(d[k]));
  } else if constexpr (V == 5) {
    asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(d[k]) : "r"(o.a[k]), "r"(o.b[k]));
  } else if constexpr (V == 6) {
    asm volatile("add.u32 %0, %0, %1;" : "+r"(d[k]) : "r"(o.s));
  } else if constexpr (V == 7) {
    asm volatile("shl.b32 %0, %0, 1;" : "+r"(d[k]));
  } else if constexpr (V == 8) {  // imm_ax
    if constexpr (((I >> 3) & 1) == 0) asm volatile("add.u32 %0, %0, 1;" : "+r"(d[k]));
    else asm volatile("xor.b32 %0, %0, 85;" : "+r"(d[k]));
  } else if constexpr (V == 9) {  // rr_ax
    if constexpr (((I >> 3) & 1) == 0) asm volatile("add.u32 %0, %0, %1;" : "+r"(d[k]) : "r"(o.a[k]));
    else asm volatile("xor.b32 %0, %0, %1;" : "+r"(d[k]) : "r"(o.b[k]));
  } else if constexpr (V == 10) {  // imm_as
    if constexpr (((I >> 3) & 1) == 0) asm volatile("add.u32 %0, %0, 1;" : "+r"(d[k]));
    else asm volatile("shl.b32 %0, %0, 1;" : "+r"(d[k]));
  }
}

template <int V, int U, int I, int N>
struct EmitBlock {
  static __device__ __forceinline__ void go(unsigned *d, const CellArgs &o) {
    if constexpr (I < U && N > 0) {
      unit<V, I>(d, o);
      EmitBlock<V, U, I + 1, N - 1>::go(d, o);
    }
  }
};
template <int V, int U, int I = 0>
struct EmitUnits {
  static __device__ __forceinline__ void go(unsigned *d, const CellArgs &o) {
    EmitBlock<V, U, I, 64>::go(d, o);
    if constexpr (I + 64 < U) EmitUnits<V, U, I + 64>::go(d, o);
  }
};

template <int V, int U>
__global__ void k_cell(std::uint64_t *out, unsigned *dval, int reps,
                       int wgroups, CellArgs args, const unsigned *zeros) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool active = warp < 4 * wgroups;
  // operands through a per-lane add of a loaded zero: non-uniform, in GPRs
  const unsigned z = zeros[threadIdx.x];
  CellArgs o;
  for (int i = 0; i < 8; ++i) { o.a[i] = args.a[i] + z; o.b[i] = args.b[i] + z; }
  o.s = args.s + z;
  unsigned d[8];
  for (int i = 0; i < 8; ++i) d[i] = 3u + warp + i + z;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (!active) continue;
    const std::uint64_t t_s = mb_clock();
    EmitUnits<V, U>::go(d, o);
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      out[(warp * reps + rep) * 2 + 0] = t_s;
      out[(warp * reps + rep) * 2 + 1] = t_w;
    }
  }
  if (threadIdx.x == 0) {
    unsigned s = 0;
    for (int i = 0; i < 8; ++i) s ^= d[i];
    dval[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, unsigned *, int, int, CellArgs, const unsigned *);
#define ROWS(V, NAME) \
  {NAME, 64, &k_cell<V, 64>}, {NAME, 128, &k_cell<V, 128>}, {NAME, 256, &k_cell<V, 256>},
static const struct { const char *variant; int u; CKern kern; } C_CELLS[] = {
  ROWS(0, "add_imm") ROWS(1, "add_rr") ROWS(2, "add_rrr") ROWS(3, "xor_rr")
  ROWS(4, "and_imm") ROWS(5, "mad_rrr") ROWS(6, "add_rs") ROWS(7, "shl_imm")
  ROWS(8, "imm_ax") ROWS(9, "rr_ax") ROWS(10, "imm_as")
};
static const int WGS[] = {1, 2, 4, 8};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  unsigned *d_val = nullptr, *d_zeros = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 32 * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_val, sizeof(unsigned) * 8));
  MB_CHECK(cudaMalloc(&d_zeros, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMemset(d_zeros, 0, sizeof(unsigned) * 1024));
  for (const auto &c : C_CELLS) {
    for (int wg : WGS) {
      const std::string variant = std::string(c.variant) + ".w" + std::to_string(wg);
      if (!mb_selected(args, variant) &&
          !mb_selected(args, std::string(c.variant) + ".cta.w" + std::to_string(wg)))
        continue;
      CellArgs a{};
      for (int i = 0; i < 8; ++i) { a.a[i] = 5u + 2u * i; a.b[i] = 11u + 3u * i; }
      a.s = 7u;
      std::vector<double> warp_samples, cta_samples;
      for (int l = 0; l < args.launches; ++l) {
        c.kern<<<1, 128 * wg>>>(d_out, d_val, args.reps, wg, a, d_zeros);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(), cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(4 * wg * args.reps * 2);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) {
          std::uint64_t ts_min = ~0ull, tw_max = 0;
          for (int wp = 0; wp < 4 * wg; ++wp) {
            const std::uint64_t ts = h[(wp * args.reps + r) * 2 + 0];
            const std::uint64_t tw = h[(wp * args.reps + r) * 2 + 1];
            warp_samples.push_back(static_cast<double>(tw - ts));
            ts_min = std::min(ts_min, ts);
            tw_max = std::max(tw_max, tw);
          }
          cta_samples.push_back(static_cast<double>(tw_max - ts_min));
        }
      }
      const std::string base = "\"variant\":\"" + std::string(c.variant) + "\",\"w\":" + std::to_string(wg);
      w.row(variant, c.u, base + ",\"window\":\"warp\"", warp_samples);
      w.row(std::string(c.variant) + ".cta.w" + std::to_string(wg), c.u, base + ",\"window\":\"cta\"", cta_samples);
    }
  }
  return 0;
}
