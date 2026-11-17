// tools/microbench/math_occupancy/probe/math_occupancy.cu -- protocol.json.
// One CTA on one SM: <<<1, 32*Wt>>> with Wt = 4*W; warp w sits on SMSP
// w % 4.  placement=one: only warps with w % 4 == 0 run the timed
// window (the rest exit after the first barrier); placement=four: all
// warps run it.  Window per rep: all-warp barrier; clock t0; the
// stream, fully unrolled, register operands only; clock t1; lane 0
// records t1 - t0.  rep 0 is dropped on the host.
// W = 16 needs Wt = 64 warps = 2048 threads > the 1024-thread CTA cap,
// so it runs as 2 co-resident 1024-thread CTAs per SM (grid = 2 x SM
// count, __launch_bounds__(1024, 2) keeps two CTAs resident per SM),
// aligned each rep by a global arrival barrier; every SM runs the same
// cell shape ('one' still puts W participants on SMSP 0 of each SM:
// warps w % 4 == 0 of both CTAs).
//   math_occupancy --out file.jsonl --launches L --reps R [--only s[,s]]
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); \
  std::exit(1); } } while (0)
__device__ __forceinline__ unsigned long long clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}

// stream step I of stream S.  d[8] = rotating integer destinations,
// f[8] = rotating float destinations, x = float chain state,
// a/b + c0/c1 + s[8] = loop-invariant operands.
// streams: 0 imad, 1 fadd_chain, 2 fmnmx_chain, 3 shfl_fadd, 4 lop3,
//          5 imad_lop3, 6 imad_fadd, 7 lop3_fadd, 8 imad_imadhi,
//          9 fadd_fmul, 10 fadd_ind, 11 imadhi, 12 imad_chain,
//          13 lop3_chain, 14 imadhi_chain, 15 fmul_chain
template <int S, int I>
__device__ __forceinline__ void mstep(unsigned* d, float* f, float& x,
                                      unsigned& xi, unsigned a,
                                      unsigned b, float c0, float c1,
                                      const unsigned* s) {
  if constexpr (S == 0) {  // imad: independent, 8 rotating destinations
    asm volatile("mad.lo.s32 %0, %1, %2, %0;"
                 : "+r"(d[I & 7]) : "r"(a), "r"(b));
  } else if constexpr (S == 1) {  // fadd_chain: one dependence chain
    asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(c0));
  } else if constexpr (S == 2) {  // fmnmx_chain: alternating constants
    asm volatile("max.f32 %0, %0, %1;" : "+f"(x) : "f"((I & 1) ? c1 : c0));
  } else if constexpr (S == 3) {  // shfl_fadd: shfl_xor + add pair
    float y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(y) : "f"(x), "n"(16 >> (I % 5)));
    asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
  } else if constexpr (S == 4) {  // lop3: a ^ s_i ^ d_j three-input LOP3
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
                 : "=r"(d[I & 7])
                 : "r"(a), "r"(s[I & 7]), "r"(d[(I + 1) & 7]));
  } else if constexpr (S == 5) {  // imad_lop3: serial alternating chain
    if constexpr ((I & 1) == 0)
      asm volatile("mad.lo.s32 %0, %1, %2, %0;"
                   : "+r"(xi) : "r"(a), "r"(b));
    else
      asm volatile("lop3.b32 %0, %1, %2, %0, 0x96;"
                   : "+r"(xi) : "r"(a), "r"(b));
  } else if constexpr (S == 6) {  // imad_fadd: serial alternating
    // chain across register banks via free bit reinterprets
    if constexpr ((I & 1) == 0)
      asm volatile("mad.lo.s32 %0, %1, %2, %3;"
                   : "=r"(xi) : "r"(a), "r"(b), "r"(__float_as_uint(x)));
    else
      asm volatile("add.f32 %0, %0, %1;"
                   : "+f"(x) : "f"(__uint_as_float(xi)));
  } else if constexpr (S == 7) {  // lop3_fadd: serial alternating chain
    if constexpr ((I & 1) == 0)
      asm volatile("lop3.b32 %0, %1, %2, %0, 0x96;"
                   : "+r"(xi) : "r"(a), "r"(__float_as_uint(x)));
    else
      asm volatile("add.f32 %0, %0, %1;"
                   : "+f"(x) : "f"(__uint_as_float(xi)));
  } else if constexpr (S == 8) {  // imad_imadhi: serial alternating
    if constexpr ((I & 1) == 0)
      asm volatile("mad.lo.s32 %0, %1, %2, %0;"
                   : "+r"(xi) : "r"(a), "r"(b));
    else
      asm volatile("mul.hi.s32 %0, %1, %0;" : "+r"(xi) : "r"(a));
  } else if constexpr (S == 9) {  // fadd_fmul: serial alternating
    // chain; -Xptxas -fmad=false keeps add/mul from contracting
    if constexpr ((I & 1) == 0)
      asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(c0));
    else
      asm volatile("mul.f32 %0, %0, %1;" : "+f"(x) : "f"(c1));
  } else if constexpr (S == 10) {  // fadd_ind: 8 independent accumulates
    asm volatile("add.f32 %0, %0, %1;" : "+f"(f[I & 7]) : "f"(c0));
  } else if constexpr (S == 11) {  // imadhi: independent mul.hi
    asm volatile("mul.hi.s32 %0, %1, %2;"
                 : "=r"(s[I & 7]) : "r"(a), "r"(s[(I + 1) & 7]));
  } else if constexpr (S == 12) {  // imad_chain: one serial chain
    asm volatile("mad.lo.s32 %0, %1, %2, %0;"
                 : "+r"(xi) : "r"(a), "r"(b));
  } else if constexpr (S == 13) {  // lop3_chain: one serial chain
    asm volatile("lop3.b32 %0, %1, %2, %0, 0x96;"
                 : "+r"(xi) : "r"(a), "r"(b));
  } else if constexpr (S == 14) {  // imadhi_chain: one serial chain
    asm volatile("mul.hi.s32 %0, %1, %0;" : "+r"(xi) : "r"(a));
  } else {  // S == 15 fmul_chain: one serial chain
    asm volatile("mul.f32 %0, %0, %1;" : "+f"(x) : "f"(c1));
  }
}

// two-level unroll: ptxas caps template recursion depth, 64-step inner
// blocks keep it at STEPS/64 + 64
template <int S, int I, int N>
struct MBlock {
  __device__ __forceinline__ static void go(unsigned* d, float* f,
                                            float& x, unsigned& xi,
                                            unsigned a, unsigned b,
                                            float c0, float c1,
                                            const unsigned* s) {
    if constexpr (N > 0) {
      mstep<S, I>(d, f, x, xi, a, b, c0, c1, s);
      MBlock<S, I + 1, N - 1>::go(d, f, x, xi, a, b, c0, c1, s);
    }
  }
};
template <int S, int I, int STEPS>
struct MUnroll {
  __device__ __forceinline__ static void go(unsigned* d, float* f,
                                            float& x, unsigned& xi,
                                            unsigned a, unsigned b,
                                            float c0, float c1,
                                            const unsigned* s) {
    MBlock<S, I, 64>::go(d, f, x, xi, a, b, c0, c1, s);
    if constexpr (I + 64 < STEPS)
      MUnroll<S, I + 64, STEPS>::go(d, f, x, xi, a, b, c0, c1, s);
  }
};

template <int S>
__global__ void __launch_bounds__(1024, 2) k_math_occ(
    unsigned long long* rec, unsigned* sink, unsigned* gsync,
    unsigned* gerr, int reps, int W, int four, int ctas,
    unsigned* gslot, unsigned* gpart, unsigned* partflag,
    unsigned ia, unsigned ib, unsigned a0,
    float c0, float c1) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned gw = blockIdx.x * (blockDim.x >> 5) + warp;
  const bool part = four || ((warp & 3) == 0);
  unsigned d[8], s[8];
  float f[8];
  for (int i = 0; i < 8; ++i) {
    d[i] = ib + 0x9e3779b9u * (i + 1) + gw;
    s[i] = a0 + 0x85ebca6bu * (i + 3) ^ (unsigned)lane;
    f[i] = c0 + 0.5f * (float)i + 0.25f * (float)lane;
  }
  unsigned a = ia ^ (0x9e3779b9u * (gw + 1));
  unsigned b = ib + 0x85ebca6bu * (lane + 1);
  unsigned xi = a0 + 0x27d4eb2fu * (gw + 7) + b;
  float x = c0 + 0.25f * (float)lane + (float)gw;
  asm volatile("" : "+r"(a), "+r"(b), "+r"(xi), "+f"(x), "+f"(c0),
                    "+f"(c1));
  for (int i = 0; i < 8; ++i) asm volatile("" : "+r"(s[i]), "+f"(f[i]));
  __syncthreads();  // the initial barrier every warp passes
  if (ctas > 1) {   // --ctas C: keep only the CTAs that landed on the
    // SM of the first claimant; count them in gpart (row_gather sweep)
    if (threadIdx.x == 0) {
      unsigned sm;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      atomicCAS(gslot, 0u, sm + 1);
      unsigned target = 0, spins = 0;
      while (!(target = *((volatile unsigned*)gslot)))
        if (++spins > 8000000000u) { *gerr = 2; break; }
      const unsigned p = sm + 1 == target;
      partflag[blockIdx.x] = p;
      if (p) atomicAdd(gpart, 1u);
    }
    __syncthreads();
    if (!partflag[blockIdx.x]) return;
    if (*((volatile unsigned*)gerr)) return;
  }
  if (!part) return;  // non-participating warps exit immediately
  // rep barrier arrivals: ctas > 1 -> only the C selected CTAs arrive;
  // v1 two-CTA mode (gridDim > 1, ctas == 1) -> every CTA arrives
  const unsigned barrier_ctas = ctas > 1 ? (unsigned)ctas : gridDim.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (gridDim.x > 1) {  // global rep barrier across co-resident CTAs
      if (threadIdx.x == 0) {
        const unsigned target = barrier_ctas * (unsigned)(rep + 1);
        atomicAdd(gsync, 1u);
        unsigned long long spins = 0;
        while (*((volatile unsigned*)gsync) < target)
          if (++spins > 8000000000ull) { *gerr = 1; break; }
      }
      __syncthreads();
      if (*((volatile unsigned*)gerr)) return;
    }
    const unsigned long long t0 = clk();
    MUnroll<S, 0, S == 3 ? 128 : 256>::go(d, f, x, xi, a, b, c0, c1, s);
    const unsigned long long t1 = clk();
    // every lane writes (divergence-free: no BSSY inside the window);
    // the host reads lane 0
    rec[((size_t)gw * 32 + lane) * reps + rep] = t1 - t0;
  }
  if (lane == 0) {
    unsigned acc = __float_as_uint(x);
    acc ^= xi;
    for (int i = 0; i < 8; ++i) acc ^= d[i] ^ s[i] ^ __float_as_uint(f[i]);
    sink[gw] = acc;
  }
}

static const char* SNAMES[] = {"imad", "fadd_chain", "fmnmx_chain", "shfl_fadd", "lop3", "imad_lop3", "imad_fadd", "lop3_fadd", "imad_imadhi", "fadd_fmul", "fadd_ind", "imadhi", "imad_chain", "lop3_chain", "imadhi_chain", "fmul_chain"};
constexpr int kReps = 33;

struct Cell {
  const char* sname;
  int W, four, ctas;
};

template <int S>
void run_cell(const Cell& c, FILE* f, unsigned long long* rec,
              unsigned* sink, unsigned* gsync, unsigned* gerr,
              unsigned* gslot, unsigned* gpart, unsigned* partflag,
              const std::string& only, int launches, int reps,
              int num_sm, const char* devname, int sm_clock_khz) {
  const unsigned maxgl = 2 * (unsigned)num_sm * 32 * 32;
  std::vector<unsigned long long> h((size_t)maxgl * reps);
  const int wt = 4 * c.W;
  const char* pname = c.four ? "four" : "one";
  const std::string variant = std::string(c.sname) + "." + pname + ".w" +
                              std::to_string(c.W) + ".c" +
                              std::to_string(c.ctas);
  const std::string family = std::string("math_occupancy.") + c.sname +
                             "." + pname + ".W" + std::to_string(c.W) +
                             ".C" + std::to_string(c.ctas);
  if (!only.empty() && variant.find(only) == std::string::npos &&
      family.find(only) == std::string::npos)
    return;
  const int nl = c.ctas > 1 ? launches * 4 : launches;
  std::vector<double> samples;
  unsigned err = 0, parts = 0, discarded = 0;
  std::vector<unsigned> pflag(2 * (unsigned)num_sm);
  for (int l = 0; l < nl; ++l) {
    CK(cudaMemset(gsync, 0, 4));
    CK(cudaMemset(gerr, 0, 4));
    unsigned this_err = 0;
    if (c.ctas == 1 && wt <= 32) {
      k_math_occ<S><<<1, 32 * wt>>>(rec, sink, gsync, gerr, reps, c.W,
                                    c.four, 1, gslot, gpart, partflag,
                                    2654435761u + l, 2166136261u - l,
                                    3266489917u, 1.5f + l, -2.25f - l);
    } else {
      const int grid = c.ctas > 1 ? c.ctas * num_sm : 2 * num_sm;
      if (c.ctas > 1) {
        CK(cudaMemset(gslot, 0, 4));
        CK(cudaMemset(gpart, 0, 4));
        CK(cudaMemset(partflag, 0, grid * 4));
      }
      k_math_occ<S><<<grid, 32 * (wt <= 32 ? wt : 32)>>>(
          rec, sink, gsync, gerr, reps, c.W, c.four, c.ctas, gslot,
          gpart, partflag, 2654435761u + l, 2166136261u - l,
          3266489917u, 1.5f + l, -2.25f - l);
      CK(cudaMemcpy(&this_err, gerr, 4, cudaMemcpyDeviceToHost));
      err |= this_err;
      if (c.ctas > 1) {
        CK(cudaMemcpy(&parts, gpart, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pflag.data(), partflag, grid * 4,
                      cudaMemcpyDeviceToHost));
      }
    }
    CK(cudaDeviceSynchronize());
    if (c.ctas > 1 && (this_err || (int)parts != c.ctas)) {
      ++discarded;
      continue;
    }
    CK(cudaMemcpy(h.data(), rec, h.size() * 8, cudaMemcpyDeviceToHost));
    const int cta_warps = wt <= 32 ? wt : 32;
    const int cta_n = c.ctas > 1 ? c.ctas * num_sm
                               : (wt <= 32 ? 1 : 2 * num_sm);
    for (int b = 0; b < cta_n; ++b) {
      if (c.ctas > 1 && !pflag[b]) continue;
      for (int w = 0; w < cta_warps; ++w) {
        const int gw = b * cta_warps + w;
        if (!c.four && (w & 3)) continue;
        for (int r = 1; r < reps; ++r)
          samples.push_back((double)h[((size_t)gw * 32) * reps + r]);
      }
    }
  }
  std::sort(samples.begin(), samples.end());
  auto q = [&](double p) {
    return samples.empty() ? 0.0
        : samples[(size_t)(p * (samples.size() - 1))];
  };
  const int actual = c.ctas > 1 ? (int)parts * c.W : c.W;
  std::fprintf(f,
      "{\"schema\":\"cyclescope.microbench_sample.v1\",\"family\":\"%s\","
      "\"stream\":\"%s\",\"placement\":\"%s\",\"warps_per_smsp\":%d,"
      "\"ctas\":%d,\"total_warps\":%d,\"p10\":%.1f,\"p50\":%.1f,"
      "\"p90\":%.1f,\"n\":%zu,\"gpu_name\":\"%s\",\"sm_clock_khz\":%d,"
      "\"participants\":%u,\"warps_per_smsp_actual\":%d,"
      "\"discarded_launches\":%u%s}\n",
      family.c_str(), c.sname, pname, c.W, c.ctas, wt, q(0.10),
      q(0.50), q(0.90), samples.size(), devname, sm_clock_khz,
      c.ctas > 1 ? parts : 1, actual, discarded,
      err ? ",\"error\":\"barrier timeout\"" : "");
  std::fflush(f);
}

template <int S>
void run_stream(const char* sname, FILE* f, unsigned long long* rec,
                unsigned* sink, unsigned* gsync, unsigned* gerr,
                unsigned* gslot, unsigned* gpart, unsigned* partflag,
                const std::string& only, int launches, int reps,
                int num_sm, const char* devname, int sm_clock_khz,
                const std::vector<int>& ws, const std::vector<int>& ctas,
                const std::vector<int>& fours) {
  for (int four : fours)
    for (int W : ws)
      for (int C : ctas)
        run_cell<S>({sname, W, four, C}, f, rec, sink, gsync, gerr,
                    gslot, gpart, partflag, only, launches, reps, num_sm,
                    devname, sm_clock_khz);
}

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p; ) {
    v.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return v;
}

int main(int argc, char** argv) {
  std::string out = "math_occupancy.jsonl", only;
  int launches = 8, reps = kReps;
  std::vector<int> ws{1, 2, 4, 8, 16}, ctas{1}, streams, fours{0, 1};
  for (int i = 1; i < argc; ++i) {
    const std::string k(argv[i]);
    auto need = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n",
                                        k.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (k == "--out") out = need();
    else if (k == "--launches") launches = std::atoi(need());
    else if (k == "--reps") reps = std::atoi(need());
    else if (k == "--only") only = need();
    else if (k == "--ws") ws = parse_list(need());
    else if (k == "--ctas") ctas = parse_list(need());
    else if (k == "--streams") streams = parse_list(need());
    else if (k == "--placements") fours = parse_list(need());
    else { std::fprintf(stderr, "unknown argument %s\n", k.c_str());
           std::exit(2); }
  }
  if (reps < 2 || reps > 4096 || launches < 1) return 2;
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, 0));
  int khz;
  CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  unsigned long long* rec;
  unsigned *sink, *gsync, *gerr, *gslot, *gpart, *partflag;
  CK(cudaMalloc(&rec, (size_t)2 * pr.multiProcessorCount * 32 * 32 * reps * 8));
  CK(cudaMalloc(&sink, 2 * pr.multiProcessorCount * 32 * 4));
  CK(cudaMalloc(&gsync, 4));
  CK(cudaMalloc(&gerr, 4));
  CK(cudaMalloc(&gslot, 4));
  CK(cudaMalloc(&gpart, 4));
  CK(cudaMalloc(&partflag, (size_t)2 * pr.multiProcessorCount * 4));
  FILE* f = std::fopen(out.c_str(), "a");
  if (!f) { std::fprintf(stderr, "cannot append %s\n", out.c_str());
            return 1; }
  const std::vector<int> sids =
      streams.empty() ? std::vector<int>{0, 1, 2, 3, 4} : streams;
  for (int S : sids) {
    switch (S) {
#define RUN(ID) run_stream<ID>(SNAMES[ID], f, rec, sink, gsync, gerr, \
      gslot, gpart, partflag, only, launches, reps, \
      pr.multiProcessorCount, pr.name, khz, ws, ctas, fours)
      case 0: RUN(0); break;
      case 1: RUN(1); break;
      case 2: RUN(2); break;
      case 3: RUN(3); break;
      case 4: RUN(4); break;
      case 5: RUN(5); break;
      case 6: RUN(6); break;
      case 7: RUN(7); break;
      case 8: RUN(8); break;
      case 9: RUN(9); break;
      case 10: RUN(10); break;
      case 11: RUN(11); break;
      case 12: RUN(12); break;
      case 13: RUN(13); break;
      case 14: RUN(14); break;
      case 15: RUN(15); break;
      default: std::fprintf(stderr, "bad stream %d\n", S); return 2;
    }
  }
  std::fclose(f);
  return 0;
}
