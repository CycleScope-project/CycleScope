// tools/microbench/store_flood/probe.cu -- protocol v1 (frozen
// tools/microbench/store_flood/protocol.json + addendum): the SM
// global-store path with a full store buffer.  One CTA of 32*W
// threads (W in {1,2,4,8}; warp w sits on SMSP w % 4), bar.sync +
// fence.sc.gpu before every repetition.  Window = 48 groups of 8
// back-to-back global stores each followed by a %clock64 read (49
// stamps per warp, lane 0 records them).  The store of group g,
// slot k, lane l writes base + ((g*8+k)*8 + warp)*4096 + (l mod L)*stride,
// so every request owns a fresh 4-KiB region and lanes l >= L fold
// onto lane l mod L (same address, same value).  Variants:
// st.global.u16 / .b32 / .v2.b32 / .v4.b32 / none (clock only).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "tensor_pipe/mb.cuh"

enum : int { V_NONE = 0, V_U16 = 2, V_B32 = 4, V_V2 = 8, V_V4 = 16 };
constexpr int kGroups = 48;              // 8 stores + 1 clock per group
constexpr int kStamps = kGroups + 1;

// One store at [p + OFF]; OFF is a compile-time immediate inside the
// PTX string so the window carries no address arithmetic but the
// per-group pointer bump.
#define MB_STG_AT(VV, OFF, P, S)                                          \
  do {                                                                    \
    if constexpr (VV == V_V4)                                             \
      asm volatile("st.global.v4.b32 [%0+" #OFF "], {%1, %2, %3, %4};"    \
                   :: "l"(P), "r"((S)[0]), "r"((S)[1]), "r"((S)[2]),      \
                      "r"((S)[3]) : "memory");                            \
    else if constexpr (VV == V_V2)                                        \
      asm volatile("st.global.v2.b32 [%0+" #OFF "], {%1, %2};"            \
                   :: "l"(P), "r"((S)[0]), "r"((S)[1]) : "memory");       \
    else if constexpr (VV == V_B32)                                       \
      asm volatile("st.global.b32 [%0+" #OFF "], %1;"                     \
                   :: "l"(P), "r"((S)[0]) : "memory");                    \
    else if constexpr (VV == V_U16)                                       \
      asm volatile("st.global.u16 [%0+" #OFF "], %1;"                     \
                   :: "l"(P), "r"((S)[0]) : "memory");                    \
  } while (0)

template <int V>
__device__ __forceinline__ void group8(std::uint64_t p, const unsigned *src) {
  MB_STG_AT(V, 0, p, src);
  MB_STG_AT(V, 32768, p, src);
  MB_STG_AT(V, 65536, p, src);
  MB_STG_AT(V, 98304, p, src);
  MB_STG_AT(V, 131072, p, src);
  MB_STG_AT(V, 163840, p, src);
  MB_STG_AT(V, 196608, p, src);
  MB_STG_AT(V, 229376, p, src);
}

// Window unrolled by template recursion: group G stores, then t[G],
// then the pointer advances 256 KiB (= 8 * 32 KiB request regions).
template <int V, int G>
struct Flood {
  static __device__ __forceinline__ void go(std::uint64_t p,
                                            std::uint64_t *t,
                                            const unsigned *src) {
    if constexpr (G <= kGroups) {
      group8<V>(p, src);
      t[G] = mb_clock();
      Flood<V, G + 1>::go(p + 262144, t, src);
    }
  }
};

template <int V>
__global__ void k_flood(std::uint64_t *out, unsigned *sink, char *base,
                        int stride, int lanes, int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned src[4] = {0x11223344u + threadIdx.x, 0x55667788u,
                           0x99aabbccu, 0xddeeff00u};
  std::uint64_t t[kStamps];
  const std::uint64_t p0 = reinterpret_cast<std::uint64_t>(base) +
                           warp * 4096 + (lane % lanes) * stride;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    // drain: every store of the previous repetition acknowledged
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    Flood<V, 1>::go(p0, t, src);
    asm volatile("" ::: "memory");
    if (lane == 0)
#pragma unroll
      for (int g = 0; g < kStamps; ++g)
        out[(warp * reps + rep) * kStamps + g] = t[g];
  }
  if (threadIdx.x == 0)
    sink[0] = static_cast<unsigned>(p0) + src[0];
}


// ---------------- protocol v2: multi-SM sets + epilogue geometries --------
// New kernel family k_floodx<GEO>: same 48 x 8-store window, plus a global
// arrival counter so every participating SM opens each window together.
// GEO: 0 none (control), 1 full16 (w16.s16.l32), 2 line32 (w4.s128.l32),
//      3 row4 (fresh 4 rows per request), 4 row4_refill.
// Layout: SM slot = position of %smid in the active set; slice = slot *
// (8 warps * 8 MiB).  Flood geos keep v1's interleaved mapping inside the
// SM slice (request r owns a fresh 32 KiB chunk, warp at +4 KiB).  row4 /
// row4_refill give every warp its own 8 MiB: lane l hits row l/8 (rows
// 4 KiB apart) at byte 2 * (l % 8) (+ 16 * k for refill).
enum : int { G_NONE = 0, G_FULL16 = 1, G_LINE32 = 2, G_ROW4 = 3,
             G_REFILL = 4, G_LACE = 5, G_EPI = 6, G_EPISEQ = 7 };

// protocol v3: epi request k covers rows 8*b + s + 2*(l % 4) of a warp's
// 32 rows (rows 8 KiB apart, blocks b = (k / 4) % 4 of 8 rows, row set
// s = k % 2, half h = (k / 2) % 2, pass p = k / 16).  The request's
// lane-uniform scalar offset; each lane adds its row / byte part.
constexpr std::uint64_t epi_req_off(int k) {
  return static_cast<std::uint64_t>((k / 4) % 4 * 8 + k % 2) * 8192u +
         32u * (k / 16) + 16u * ((k / 2) % 2);
}

__device__ __forceinline__ void stg_u16_at(std::uint64_t a, unsigned v) {
  asm volatile("st.global.u16 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
constexpr int kSmemCarve = 135168;   // > smem/SM / 2 -> one CTA per SM
constexpr int kWMax = 8;

template <int GEO, int G>
__device__ __forceinline__ void group8x(std::uint64_t p,
                                        const unsigned *src) {
  if constexpr (GEO == G_FULL16) {
    MB_STG_AT(V_V4, 0, p, src);
    MB_STG_AT(V_V4, 32768, p, src);
    MB_STG_AT(V_V4, 65536, p, src);
    MB_STG_AT(V_V4, 98304, p, src);
    MB_STG_AT(V_V4, 131072, p, src);
    MB_STG_AT(V_V4, 163840, p, src);
    MB_STG_AT(V_V4, 196608, p, src);
    MB_STG_AT(V_V4, 229376, p, src);
  } else if constexpr (GEO == G_LINE32) {
    MB_STG_AT(V_B32, 0, p, src);
    MB_STG_AT(V_B32, 32768, p, src);
    MB_STG_AT(V_B32, 65536, p, src);
    MB_STG_AT(V_B32, 98304, p, src);
    MB_STG_AT(V_B32, 131072, p, src);
    MB_STG_AT(V_B32, 163840, p, src);
    MB_STG_AT(V_B32, 196608, p, src);
    MB_STG_AT(V_B32, 229376, p, src);
  } else if constexpr (GEO == G_ROW4) {
    MB_STG_AT(V_U16, 0, p, src);
    MB_STG_AT(V_U16, 16384, p, src);
    MB_STG_AT(V_U16, 32768, p, src);
    MB_STG_AT(V_U16, 49152, p, src);
    MB_STG_AT(V_U16, 65536, p, src);
    MB_STG_AT(V_U16, 81920, p, src);
    MB_STG_AT(V_U16, 98304, p, src);
    MB_STG_AT(V_U16, 114688, p, src);
  } else if constexpr (GEO == G_LACE) {
    MB_STG_AT(V_U16, 0, p, src);
    MB_STG_AT(V_U16, 16384, p, src);
    MB_STG_AT(V_U16, 32768, p, src);
    MB_STG_AT(V_U16, 49152, p, src);
    MB_STG_AT(V_U16, 65536, p, src);
    MB_STG_AT(V_U16, 81920, p, src);
    MB_STG_AT(V_U16, 98304, p, src);
    MB_STG_AT(V_U16, 114688, p, src);
  } else if constexpr (GEO == G_EPI || GEO == G_EPISEQ) {
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 0), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 1), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 2), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 3), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 4), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 5), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 6), src[0]);
    stg_u16_at(p + epi_req_off((G - 1) * 8 + 7), src[0]);
  } else if constexpr (GEO == G_REFILL) {
    MB_STG_AT(V_U16, 0, p, src);
    MB_STG_AT(V_U16, 16, p, src);
    MB_STG_AT(V_U16, 32, p, src);
    MB_STG_AT(V_U16, 48, p, src);
    MB_STG_AT(V_U16, 64, p, src);
    MB_STG_AT(V_U16, 80, p, src);
    MB_STG_AT(V_U16, 96, p, src);
    MB_STG_AT(V_U16, 112, p, src);
  } else {
    asm volatile("" ::: "memory");
  }
}

template <int GEO>
__device__ __forceinline__ constexpr std::uint64_t group_bump() {
  return (GEO == G_ROW4 || GEO == G_LACE)
             ? 131072u
             : (GEO == G_REFILL ? 128u
                                : (GEO == G_EPI || GEO == G_EPISEQ ? 0u
                                                                   : 262144u));
}

template <int GEO, int G>
struct FloodX {
  static __device__ __forceinline__ void go(std::uint64_t p,
                                            std::uint64_t *t,
                                            const unsigned *src) {
    if constexpr (G <= kGroups) {
      group8x<GEO, G>(p, src);
      t[G] = mb_clock();
      FloodX<GEO, G + 1>::go(p + group_bump<GEO>(), t, src);
    }
  }
};

// arrived: global arrival counter (all participating warps, monotonically
// increasing; rep r waits for (r+1) * npart_warps).  parts[blockIdx.x]:
// participant marker (smid + 1), or smid | 0x8000_0000 on spin timeout.
template <int GEO>
__global__ void __launch_bounds__(32 * kWMax, 1) k_floodx(
    std::uint64_t *out, unsigned *sink, char *base, int stride, int lanes,
    int warps, int reps, unsigned *arrived, unsigned npart_warps,
    const unsigned char *active, const int *order, unsigned *parts,
    std::uint64_t rep_stride) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned smid = 0;
  int slot = 0;
  if (active) {
    smid = mb_smid();
    if (!active[smid & 255u]) {
      if (threadIdx.x == 0) parts[blockIdx.x] = 0;
      return;
    }
    slot = order[smid & 255u];
  }
  const unsigned src[4] = {0x11223344u + threadIdx.x, 0x55667788u,
                           0x99aabbccu, 0xddeeff00u};
  std::uint64_t t[kStamps];
  // warm: reps overwrite the same region (host memsets per launch).
  // virgin: rep r uses a disjoint rep_stride-sized area; each warp owns
  // reps * rep_stride bytes so no two reps or warps share an address.
  const std::uint64_t warp_span =
      rep_stride ? rep_stride * static_cast<std::uint64_t>(reps)
                 : 8u * 1024u * 1024u;
  const std::uint64_t slice =
      reinterpret_cast<std::uint64_t>(base) +
      static_cast<std::uint64_t>(slot) * kWMax * warp_span;
  std::uint64_t p0;
  if constexpr (GEO == G_ROW4 || GEO == G_REFILL) {
    p0 = slice + static_cast<std::uint64_t>(warp) * warp_span +
         static_cast<std::uint64_t>(lane / 8) * 4096u +
         static_cast<std::uint64_t>(2 * (lane % 8));
  } else if constexpr (GEO == G_LACE) {
    p0 = slice + static_cast<std::uint64_t>(warp) * warp_span +
         static_cast<std::uint64_t>(lane % 4) * 4096u +
         static_cast<std::uint64_t>(2 * (lane / 4));
  } else if constexpr (GEO == G_EPI) {
    p0 = slice + static_cast<std::uint64_t>(warp) * warp_span +
         static_cast<std::uint64_t>(lane % 4) * 16384u +
         static_cast<std::uint64_t>(2 * (lane / 4));
  } else if constexpr (GEO == G_EPISEQ) {
    p0 = slice + static_cast<std::uint64_t>(warp) * warp_span +
         static_cast<std::uint64_t>(lane / 8) * 16384u +
         static_cast<std::uint64_t>(2 * (lane % 8));
  } else {
    p0 = slice + warp * 4096 + (lane % lanes) * stride;
  }
  const int wglob = slot * kWMax + warp;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    bool stop = false;
    if (lane == 0) {   // all warps of all participating CTAs meet here
      const unsigned target = static_cast<unsigned>(rep + 1) * npart_warps;
      atomicAdd(arrived, 1u);
      const std::uint64_t deadline = mb_gtimer() + 10000000000ull;
      while (*((volatile unsigned *)arrived) < target &&
             mb_gtimer() < deadline) { }
      if (*((volatile unsigned *)arrived) < target) {
        if (active && warp == 0) parts[blockIdx.x] = smid | 0x80000000u;
        stop = true;
      }
    }
    stop = __shfl_sync(0xffffffffu, static_cast<int>(stop), 0) != 0;
    if (stop) break;
    __syncthreads();
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    FloodX<GEO, 1>::go(p0 + static_cast<std::uint64_t>(rep) * rep_stride, t,
                       src);
    asm volatile("" ::: "memory");
    // all lanes write the same stamps (no branch -> no BSSY inside the
    // window); values within a warp are uniform
#pragma unroll
    for (int g = 0; g < kStamps; ++g)
      out[(wglob * reps + rep) * kStamps + g] = t[g];
    __syncthreads();
  }
  if (active && threadIdx.x == 0 && !(parts[blockIdx.x] & 0x80000000u))
    parts[blockIdx.x] = smid + 1u;
  if (threadIdx.x == 0)
    sink[0] = static_cast<unsigned>(p0) + src[0];
}


// ---------------- protocol v4: k_bisect<MASK> ----------------
// v1 k_flood<V_B32> window (line32: width 4, stride 128, 32 lanes) with
// independent bits adding one k_floodx property each:
//   bit 0 GRID   - grid of one CTA per SM + large dynamic smem + active
//                  set of one SM (implies %smid read + active/order/parts)
//   bit 1 POLL   - per-window global arrival counter (lane0 atomicAdd +
//                  volatile spin to (rep+1)*npart, shfl of stop)
//   bit 2 STAMPS - all 32 lanes write the 49 stamps + trailing bar.sync
//   bit 3 SLICE  - v2 address base (slice = slot * 8 warps * 8 MiB)
//   bit 4 BOUNDS - __launch_bounds__(32*8, 1) + one %smid read (no grid);
//                  realized by emitting the kernel twice: k_bisect_b (with
//                  bounds) and k_bisect_n (without).
enum : int { BB_GRID = 1, BB_POLL = 2, BB_STAMPS = 4, BB_SLICE = 8,
             BB_BOUNDS = 16 };

template <int MASK>
__device__ __forceinline__ void bisect_body(
    std::uint64_t *out, unsigned *sink, char *base, int stride, int lanes,
    int warps, int reps, unsigned *arrived, unsigned npart_warps,
    const unsigned char *active, const int *order, unsigned *parts) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned smid = 0;
  int slot = 0;
  if (MASK & BB_GRID) {
    smid = mb_smid();
    if (!active[smid & 255u]) {
      if (threadIdx.x == 0) parts[blockIdx.x] = 0;
      return;
    }
    slot = order[smid & 255u];
  } else if (MASK & BB_BOUNDS) {
    smid = mb_smid();
    asm volatile("" ::"r"(smid));
  }
  const unsigned src[4] = {0x11223344u + threadIdx.x, 0x55667788u,
                           0x99aabbccu, 0xddeeff00u};
  std::uint64_t t[kStamps];
  std::uint64_t p0;
  if (MASK & BB_SLICE)
    p0 = reinterpret_cast<std::uint64_t>(base) +
         static_cast<std::uint64_t>(slot) * kWMax * 8u * 1024u * 1024u +
         static_cast<std::uint64_t>(warp) * 4096u +
         static_cast<std::uint64_t>(lane % lanes) * stride;
  else
    p0 = reinterpret_cast<std::uint64_t>(base) +
         static_cast<std::uint64_t>(warp) * 4096u +
         static_cast<std::uint64_t>(lane % lanes) * stride;
  const int wglob = slot * kWMax + warp;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    if constexpr (MASK & BB_POLL) {
      bool stop = false;
      if (lane == 0) {
        const unsigned target = static_cast<unsigned>(rep + 1) * npart_warps;
        atomicAdd(arrived, 1u);
        const std::uint64_t deadline = mb_gtimer() + 10000000000ull;
        while (*((volatile unsigned *)arrived) < target &&
               mb_gtimer() < deadline) { }
        if (*((volatile unsigned *)arrived) < target) {
          if ((MASK & BB_GRID) && warp == 0)
            parts[blockIdx.x] = smid | 0x80000000u;
          stop = true;
        }
      }
      stop = __shfl_sync(0xffffffffu, static_cast<int>(stop), 0) != 0;
      if (stop) break;
    }
    __syncthreads();
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    Flood<V_B32, 1>::go(p0, t, src);
    asm volatile("" ::: "memory");
    if constexpr (MASK & BB_STAMPS) {
#pragma unroll
      for (int g = 0; g < kStamps; ++g)
        out[(wglob * reps + rep) * kStamps + g] = t[g];
      __syncthreads();
    } else {
      // the t[kGroups] != 0 test (runtime-true) pins the lane-0 branch
      // after the last clock read; otherwise ptxas schedules its BSSY
      // inside the final window interval
      if (lane == 0 && t[kGroups] != 0)
#pragma unroll
        for (int g = 0; g < kStamps; ++g)
          out[(warp * reps + rep) * kStamps + g] = t[g];
    }
  }
  if ((MASK & BB_GRID) && threadIdx.x == 0 &&
      !(parts[blockIdx.x] & 0x80000000u))
    parts[blockIdx.x] = smid + 1u;
  if (threadIdx.x == 0)
    sink[0] = static_cast<unsigned>(p0) + src[0];
}

#define BISECT_ARGS                                             \
  std::uint64_t *out, unsigned *sink, char *base, int stride,   \
  int lanes, int warps, int reps, unsigned *arrived,            \
  unsigned npart_warps, const unsigned char *active,            \
  const int *order, unsigned *parts
#define BISECT_CALL                                                    \
  bisect_body<MASK>(out, sink, base, stride, lanes, warps, reps,       \
                    arrived, npart_warps, active, order, parts)

template <int MASK>
__global__ void __launch_bounds__(32 * kWMax, 1) k_bisect_b(BISECT_ARGS) {
  BISECT_CALL;
}
template <int MASK>
__global__ void k_bisect_n(BISECT_ARGS) {
  BISECT_CALL;
}

// ---------------- bridge cells (epi, single SM, W = 4) ----------------
// FLAGS bits: 1 filler (FMUL + F2FP + ISETP per store, value depends on
// them), 2 pred (@P store, runtime-true predicate from a pre-window
// compare), 4 lanedata (per-lane value + request index), 8 nofence (one
// window per launch, no fence.sc.gpu inside).  GROUPS: 48 or 16.
enum : int { B_FILLER = 1, B_PRED = 2, B_LANEDATA = 4, B_NOFENCE = 8 };

template <int FLAGS>
__device__ __forceinline__ void stg_bridge(std::uint64_t a, unsigned base_v,
                                           float fb, float fc, float kf,
                                           unsigned lane_v, int k,
                                           unsigned pv, unsigned &acc) {
  unsigned v = base_v;
  if constexpr (FLAGS & B_FILLER) {
    float fk;
    asm("add.f32 %0, %1, %2;" : "=f"(fk) : "f"(fc), "f"(kf));
    float f;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(f) : "f"(fb), "f"(fk));
    unsigned short h16;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h16) : "f"(f));
    v = h16;
    unsigned c;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %1, %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}" : "=r"(c)
        : "r"(lane_v + static_cast<unsigned>(k) + 256u), "r"(pv));
    acc ^= c;
  }
  if constexpr (FLAGS & B_LANEDATA) {
    v += lane_v + static_cast<unsigned>(k);
  }
  if constexpr (FLAGS & B_PRED) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t"
        "@p st.global.u16 [%0], %1;\n\t}" ::"l"(a), "r"(v), "r"(pv)
        : "memory");
  } else {
    stg_u16_at(a, v);
  }
}

template <int FLAGS, int G, int GMAX>
struct FloodB {
  static __device__ __forceinline__ void go(std::uint64_t p,
                                            std::uint64_t *t,
                                            unsigned base_v, float fb,
                                            float fc,
                                            unsigned lane_v, unsigned pv,
                                            unsigned &acc) {
    if constexpr (G <= GMAX) {
      const int k0 = (G - 1) * 8;
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 0), base_v, fb, fc,
                        static_cast<float>(k0 + 0), lane_v,
                        k0 + 0, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 1), base_v, fb, fc,
                        static_cast<float>(k0 + 1), lane_v,
                        k0 + 1, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 2), base_v, fb, fc,
                        static_cast<float>(k0 + 2), lane_v,
                        k0 + 2, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 3), base_v, fb, fc,
                        static_cast<float>(k0 + 3), lane_v,
                        k0 + 3, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 4), base_v, fb, fc,
                        static_cast<float>(k0 + 4), lane_v,
                        k0 + 4, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 5), base_v, fb, fc,
                        static_cast<float>(k0 + 5), lane_v,
                        k0 + 5, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 6), base_v, fb, fc,
                        static_cast<float>(k0 + 6), lane_v,
                        k0 + 6, pv, acc);
      stg_bridge<FLAGS>(p + epi_req_off(k0 + 7), base_v, fb, fc,
                        static_cast<float>(k0 + 7), lane_v,
                        k0 + 7, pv, acc);
      t[G] = mb_clock();
      FloodB<FLAGS, G + 1, GMAX>::go(p, t, base_v, fb, fc, lane_v, pv, acc);
    }
  }
};

template <int FLAGS, int GMAX>
__global__ void __launch_bounds__(32 * kWMax, 1) k_bridge(
    std::uint64_t *out, unsigned *sink, char *base, int warps, int reps,
    unsigned *arrived, unsigned npart_warps,
    const unsigned *predbuf) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned base_v = 0x11223344u + threadIdx.x;
  const float fb = 0.25f * static_cast<float>(base_v | 1u);
  const unsigned pv_in = predbuf[threadIdx.x];   // per-lane loaded, all 1s
  const float fc = 0.5f + 1.0e-6f * static_cast<float>(pv_in);
  const unsigned pv = pv_in;   // raw loaded value, range unknown to ptxas
  unsigned acc = 0;
  std::uint64_t t[kStamps];
  const std::uint64_t p0 =
      reinterpret_cast<std::uint64_t>(base) +
      static_cast<std::uint64_t>(warp) * 8u * 1024u * 1024u +
      static_cast<std::uint64_t>(lane % 4) * 16384u +
      static_cast<std::uint64_t>(2 * (lane / 4));
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    bool stop = false;
    if (lane == 0) {
      const unsigned target = static_cast<unsigned>(rep + 1) * npart_warps;
      atomicAdd(arrived, 1u);
      const std::uint64_t deadline = mb_gtimer() + 10000000000ull;
      while (*((volatile unsigned *)arrived) < target &&
             mb_gtimer() < deadline) { }
      if (*((volatile unsigned *)arrived) < target) stop = true;
    }
    stop = __shfl_sync(0xffffffffu, static_cast<int>(stop), 0) != 0;
    if (stop) break;
    __syncthreads();
    if constexpr (!(FLAGS & B_NOFENCE))
      asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    FloodB<FLAGS, 1, GMAX>::go(p0, t, base_v, fb, fc,
                               lane * 0x00000101u, pv, acc);
    asm volatile("" ::: "memory");
#pragma unroll
    for (int g = 0; g < kStamps; ++g)
      out[(warp * reps + rep) * kStamps + g] = t[g];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    sink[0] = static_cast<unsigned>(p0) + base_v + acc + pv;
}

#define FLOODX_KERNELS(X) \
  X(0) X(1) X(2) X(3) X(4)

static const struct { int geo; const char *name; int width; int stride;
                      int lanes; } kGeos[] = {
  {G_NONE,   "none",    0,  0,   32},
  {G_FULL16, "full16",  16, 16,  32},
  {G_LINE32, "line32",  4,  128, 32},
  {G_ROW4,   "row4",    2,  0,   32},
  {G_REFILL, "row4_refill", 2, 0, 32},
  {G_LACE,   "lace",    2,  0,   32},
  {G_EPI,    "epi",     2,  0,   32},
  {G_EPISEQ, "epi_seq", 2,  0,   32},
};

// Footprint of one repetition's window per warp, per geo (virgin mode
// gives every rep its own rep_stride bytes).
static std::uint64_t rep_stride_for(int geo) {
  switch (geo) {
    case G_ROW4: case G_LACE: return 384ull * 16384u;          // fresh 16 KiB / req
    case G_EPI: case G_EPISEQ: return 32ull * 8192u;           // 32 rows x 8 KiB
    case G_REFILL: return 16384u;                              // 4 rows x 4 KiB
    default: return 384ull * 32768u;                           // flood 32 KiB / req
  }
}

template <int GEO>
static void launch_x_one(std::uint64_t *d_out, unsigned *d_sink, char *d_base,
                         int stride, int lanes, int warps, int reps,
                         unsigned *d_arrived, unsigned npart,
                         const unsigned char *d_act, const int *d_order,
                         unsigned *d_parts, int ctas,
                         std::uint64_t rep_stride) {
  MB_CHECK(cudaFuncSetAttribute(
      k_floodx<GEO>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemCarve));
  k_floodx<GEO><<<ctas, 32 * warps, ctas > 1 ? kSmemCarve : 0>>>(
      d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart,
      d_act, d_order, d_parts, rep_stride);
}

static void launch_x(int geo, std::uint64_t *d_out, unsigned *d_sink,
                     char *d_base, int stride, int lanes, int warps, int reps,
                     unsigned *d_arrived, unsigned npart,
                     const unsigned char *d_act, const int *d_order,
                     unsigned *d_parts, int ctas, std::uint64_t rep_stride) {
  switch (geo) {
    case G_FULL16: launch_x_one<G_FULL16>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_LINE32: launch_x_one<G_LINE32>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_ROW4:   launch_x_one<G_ROW4>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_REFILL: launch_x_one<G_REFILL>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_LACE:   launch_x_one<G_LACE>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_EPI:    launch_x_one<G_EPI>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    case G_EPISEQ: launch_x_one<G_EPISEQ>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
    default:       launch_x_one<G_NONE>(d_out, d_sink, d_base, stride, lanes, warps, reps, d_arrived, npart, d_act, d_order, d_parts, ctas, rep_stride); break;
  }
}

// ---------------- protocol v5: k_flood5<V> ----------------
// v1 kernel body verbatim, plus two launch axes decided outside the timed
// window: the launch's dynamic shared-memory size (first/last byte touched
// once before the first window) and sibling CTAs (grid = all SMs, only the
// first CTA to win a global CAS runs the windows, the rest return at once;
// the winner's SM is recorded in sink[1]).
extern __shared__ char kdsm5[];

template <int V>
__global__ void k_flood5(std::uint64_t *out, unsigned *sink, char *base,
                         int stride, int lanes, int reps, unsigned *flag,
                         int smem) {
  if (flag != nullptr) {
    __shared__ unsigned go;
    if (threadIdx.x == 0) {
      unsigned smid;
      asm("mov.u32 %0, %%smid;" : "=r"(smid));
      go = atomicCAS(flag, 0u, 1u) == 0u ? smid + 1u : 0u;
    }
    __syncthreads();
    if (go == 0u) return;
    if (threadIdx.x == 0) sink[1] = go - 1u;
  }
  if (smem > 0 && threadIdx.x == 0) {
    kdsm5[0] = 1;
    kdsm5[smem - 1] = 1;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned src[4] = {0x11223344u + threadIdx.x, 0x55667788u,
                           0x99aabbccu, 0xddeeff00u};
  std::uint64_t t[kStamps];
  const std::uint64_t p0 = reinterpret_cast<std::uint64_t>(base) +
                           warp * 4096 + (lane % lanes) * stride;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    t[0] = mb_clock();
    Flood<V, 1>::go(p0, t, src);
    asm volatile("" ::: "memory");
    if (lane == 0)
#pragma unroll
      for (int g = 0; g < kStamps; ++g)
        out[(warp * reps + rep) * kStamps + g] = t[g];
  }
  if (threadIdx.x == 0)
    sink[0] = static_cast<unsigned>(p0) + src[0];
}

// Host mirror of the device address formula, relative to a warp region.
static std::uint64_t host_off(int geo, int lane, int k) {
  switch (geo) {
    case G_LACE:   return static_cast<std::uint64_t>(k) * 16384u +
                          (lane % 4) * 4096u + 2 * (lane / 4);
    case G_EPI:    return epi_req_off(k) + (lane % 4) * 16384u + 2 * (lane / 4);
    case G_EPISEQ: return epi_req_off(k) + (lane / 8) * 16384u + 2 * (lane % 8);
    default:       return static_cast<std::uint64_t>(k) * 16384u +
                          (lane / 8) * 4096u + 2 * (lane % 8);   // row4
  }
}

static double mb_pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

int main(int argc, char **argv) {
  // v2 arguments are stripped before mb_parse sees argv.
  std::string active_file, set_name = "sm1", geos, dest = "warm",
              dump_addr, bridge_name, smem_arg = "0",
              carveout_arg = "default", siblings_arg = "0";
  bool dest_given = false, bisect_mode = false, v5_mode = false;
  int xwarps = -1;
  std::vector<std::string> rest{argv[0]};
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto take = [&](std::string *d) { d->assign(argv[++i]); };
    if (a == "--active-file" && i + 1 < argc) { take(&active_file); continue; }
    if (a == "--set-name" && i + 1 < argc) { take(&set_name); continue; }
    if (a == "--geos" && i + 1 < argc) { take(&geos); continue; }
    if (a == "--warps" && i + 1 < argc) { xwarps = std::atoi(argv[++i]); continue; }
    if (a == "--dest" && i + 1 < argc) { take(&dest); dest_given = true; continue; }
    if (a == "--dump-addr" && i + 1 < argc) { take(&dump_addr); continue; }
    if (a == "--bridge" && i + 1 < argc) { take(&bridge_name); continue; }
    if (a == "--bisect") { bisect_mode = true; continue; }
    if (a == "--v5") { v5_mode = true; continue; }
    if (a == "--smem" && i + 1 < argc) { take(&smem_arg); continue; }
    if (a == "--carveout" && i + 1 < argc) { take(&carveout_arg); continue; }
    if (a == "--siblings" && i + 1 < argc) { take(&siblings_arg); continue; }
    rest.push_back(a);
  }
  const bool v2 = !active_file.empty() || !geos.empty();
  std::vector<char *> v1argv;
  for (auto &s : rest) v1argv.push_back(s.data());
  MbArgs args = mb_parse(static_cast<int>(v1argv.size()), v1argv.data());
  MbWriter w(args, "store_flood");
  if (!dump_addr.empty()) {   // host-side mirror of the device addressing
    for (std::size_t pos = 0; pos <= dump_addr.size();) {
      const auto comma = dump_addr.find(',', pos);
      const std::string tok =
          dump_addr.substr(pos, comma == std::string::npos ? comma
                                                           : comma - pos);
      int geo = -1;
      for (const auto &g : kGeos) if (tok == g.name) geo = g.geo;
      if (geo < 0) { std::fprintf(stderr, "unknown geo %s\n", tok.c_str()); return 2; }
      const std::string fp = args.out_dir + "/addr-" + tok + ".txt";
      FILE *f = std::fopen(fp.c_str(), "w");
      if (!f) { std::fprintf(stderr, "cannot open %s\n", fp.c_str()); return 2; }
      for (int k = 0; k < 20; ++k) {
        std::fprintf(f, "k=%3d:", k);
        for (int l = 0; l < 32; ++l)
          std::fprintf(f, " %llu",
                       static_cast<unsigned long long>(host_off(geo, l, k)));
        std::fprintf(f, "\n");
      }
      std::fclose(f);
      std::fprintf(stderr, "wrote %s\n", fp.c_str());
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  std::uint64_t *d_out = nullptr;
  unsigned *d_sink = nullptr;
  char *d_base = nullptr;
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned) * 8));

  if (v5_mode) {
    const MbDevice dev = mb_query_device();
    int devi = 0;
    MB_CHECK(cudaGetDevice(&devi));
    int optin = 0;
    MB_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, devi));
    const int grid = siblings_arg == "all" ? dev.sm_count : 1;
    const int W = 4;
    unsigned *d_flag = nullptr;
    MB_CHECK(cudaMalloc(&d_flag, sizeof(unsigned)));
    MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * W * args.reps *
                                kStamps));
    MB_CHECK(cudaMalloc(&d_base, 16u << 20));
    const struct { const char *geo; int v; int stride; int lanes; } gcs[] = {
      {"line32", V_B32, 128, 32},
      {"full16", V_V4, 16, 32},
      {"none",   V_NONE, 0, 32},
    };
    for (const auto &c : gcs) {
      const std::string variant =
          "v5." + std::string(c.geo) + ".s" + smem_arg + ".c" +
          carveout_arg + ".g" + siblings_arg;
      void (*kf)(std::uint64_t *, unsigned *, char *, int, int, int,
                 unsigned *, int) = nullptr;
      switch (c.v) {
        case V_B32:  kf = k_flood5<V_B32>;  break;
        case V_V4:   kf = k_flood5<V_V4>;   break;
        default:     kf = k_flood5<V_NONE>; break;
      }
      int smem = smem_arg == "max" ? optin : std::atoi(smem_arg.c_str());
      {   // the kernel's static __shared__ counts against the opt-in max
        cudaFuncAttributes fa{};
        MB_CHECK(cudaFuncGetAttributes(&fa, kf));
        smem = std::min(smem, optin - static_cast<int>(fa.sharedSizeBytes));
      }
      MB_CHECK(cudaFuncSetAttribute(
          kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      if (carveout_arg != "default")
        MB_CHECK(cudaFuncSetAttribute(
            kf, cudaFuncAttributePreferredSharedMemoryCarveout,
            std::atoi(carveout_arg.c_str())));
      std::vector<std::vector<double>> gaps(kStamps);
      unsigned hsm = ~0u;
      for (int l = 0; l < args.launches; ++l) {
        MB_CHECK(cudaMemset(d_base, 0, 16u << 20));
        MB_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned)));
        MB_CHECK(cudaMemset(d_sink, 0, sizeof(unsigned) * 8));
        kf<<<grid, 32 * W, smem>>>(d_out, d_sink, d_base, c.stride,
                                   c.lanes, args.reps,
                                   grid > 1 ? d_flag : nullptr, smem);
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        unsigned hs[8];
        MB_CHECK(cudaMemcpy(hs, d_sink, sizeof(hs),
                            cudaMemcpyDeviceToHost));
        hsm = hs[1];
        std::vector<std::uint64_t> h(W * args.reps * kStamps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          for (int wp = 0; wp < W; ++wp) {
            const std::uint64_t *row = &h[(wp * args.reps + r) * kStamps];
            for (int g = 1; g <= kGroups; ++g)
              gaps[g].push_back(static_cast<double>(row[g] - row[g - 1]));
          }
      }
      const std::string params =
          "\"geometry\":\"" + std::string(c.geo) +
          "\",\"smem_requested\":\"" + smem_arg +
          "\",\"smem_bytes\":" + std::to_string(smem) +
          ",\"carveout\":\"" + carveout_arg +
          "\",\"siblings\":\"" + siblings_arg +
          "\",\"grid\":" + std::to_string(grid) +
          ",\"warps\":" + std::to_string(W) +
          ",\"participant_sm\":" + std::to_string(hsm);
      for (int g = 1; g <= kGroups; ++g)
        w.row(variant, g, params, gaps[g]);
      std::fprintf(stderr, "%s: %d launches\n", variant.c_str(),
                   args.launches);
    }
    MB_CHECK(cudaFree(d_flag));
    MB_CHECK(cudaFree(d_base));
    MB_CHECK(cudaFree(d_out));
    MB_CHECK(cudaFree(d_sink));
    return 0;
  }

  if (bisect_mode) {
    const MbDevice dev = mb_query_device();
    const int W = 4;
    unsigned *d_arrived = nullptr, *d_parts = nullptr;
    unsigned char *d_act = nullptr;
    int *d_order = nullptr;
    MB_CHECK(cudaMalloc(&d_arrived, sizeof(unsigned)));
    MB_CHECK(cudaMalloc(&d_parts, sizeof(unsigned) * dev.sm_count));
    MB_CHECK(cudaMalloc(&d_act, 256));
    MB_CHECK(cudaMalloc(&d_order, sizeof(int) * 256));
    {
      std::vector<unsigned char> act(256, 0);
      std::vector<int> ord(256, -1);
      act[0] = 1; ord[0] = 0;                 // active set = {SM 0}
      MB_CHECK(cudaMemcpy(d_act, act.data(), 256, cudaMemcpyHostToDevice));
      MB_CHECK(cudaMemcpy(d_order, ord.data(), sizeof(int) * 256,
                          cudaMemcpyHostToDevice));
    }
    MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * kWMax * args.reps *
                                kStamps));
    MB_CHECK(cudaMalloc(&d_base, 64u << 20));   // max(SLICE 64 MiB, v1 16 MiB)
    using BK = void (*)(std::uint64_t *, unsigned *, char *, int, int, int,
                        int, unsigned *, unsigned, const unsigned char *,
                        const int *, unsigned *);
#define BKCELL(M) {M, (M) & BB_BOUNDS ? static_cast<BK>(k_bisect_b<M>)      \
                                      : static_cast<BK>(k_bisect_n<M>)}
    const struct { int mask; BK k; } cells[] = {
      BKCELL(0),  BKCELL(1),  BKCELL(2),  BKCELL(4),  BKCELL(8),
      BKCELL(16), BKCELL(31), BKCELL(30), BKCELL(29), BKCELL(27),
      BKCELL(23), BKCELL(15),
    };
    for (const auto &c : cells) {
      const std::string variant =
          "bisect.m" + std::to_string(c.mask) + ".x" + std::to_string(W);
      MB_CHECK(cudaFuncSetAttribute(
          c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemCarve));
      const bool grid = (c.mask & BB_GRID) != 0;
      std::vector<std::vector<double>> gaps(kStamps);
      int discarded = 0;
      for (int l = 0; l < args.launches; ++l) {
        MB_CHECK(cudaMemset(d_base, 0, 64u << 20));
        MB_CHECK(cudaMemset(d_arrived, 0, sizeof(unsigned)));
        if (grid) {
          MB_CHECK(cudaMemset(d_parts, 0, sizeof(unsigned) * dev.sm_count));
          c.k<<<dev.sm_count, 32 * W, kSmemCarve>>>(
              d_out, d_sink, d_base, 128, 32, W, args.reps, d_arrived,
              static_cast<unsigned>(W), d_act, d_order, d_parts);
        } else {
          c.k<<<1, 32 * W>>>(d_out, d_sink, d_base, 128, 32, W, args.reps,
                             d_arrived, static_cast<unsigned>(W), d_act,
                             d_order, d_parts);
        }
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        if (mb_take_err()) {
          std::fprintf(stderr, "launch %s: mbarrier wait timeout\n",
                       variant.c_str());
          break;
        }
        if (grid) {
          std::vector<unsigned> h(dev.sm_count);
          MB_CHECK(cudaMemcpy(h.data(), d_parts, h.size() * 4,
                              cudaMemcpyDeviceToHost));
          int parts = 0;
          for (unsigned v : h) if (v && !(v & 0x80000000u)) ++parts;
          if (parts != 1) { ++discarded; continue; }
        }
        std::vector<std::uint64_t> h(
            static_cast<std::size_t>(W) * args.reps * kStamps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          for (int wp = 0; wp < W; ++wp) {
            const std::uint64_t *row = &h[(wp * args.reps + r) * kStamps];
            for (int g = 1; g <= kGroups; ++g)
              gaps[g].push_back(static_cast<double>(row[g] - row[g - 1]));
          }
      }
      const std::string params =
          "\"mask\":" + std::to_string(c.mask) +
          ",\"width_bytes\":4,\"stride_bytes\":128,\"lanes\":32" +
          ",\"warps\":" + std::to_string(W);
      const std::string extra =
          "\"discarded_launches\":" + std::to_string(discarded);
      for (int g = 1; g <= kGroups; ++g)
        w.row(variant, g, params, gaps[g], extra);
      std::fprintf(stderr, "%s: discarded %d\n", variant.c_str(), discarded);
    }
    MB_CHECK(cudaFree(d_base));
    MB_CHECK(cudaFree(d_out));
    MB_CHECK(cudaFree(d_sink));
    MB_CHECK(cudaFree(d_arrived));
    MB_CHECK(cudaFree(d_parts));
    MB_CHECK(cudaFree(d_act));
    MB_CHECK(cudaFree(d_order));
    return 0;
  }
  if (!bridge_name.empty()) {   // bridge cells: epi, single SM, W = 4
    unsigned *d_arrived = nullptr;
    MB_CHECK(cudaMalloc(&d_arrived, sizeof(unsigned)));
    unsigned *d_predbuf = nullptr;
    MB_CHECK(cudaMalloc(&d_predbuf, sizeof(unsigned) * 32 * kWMax));
    std::vector<unsigned> hb(32 * kWMax, 1u);
    MB_CHECK(cudaMemcpy(d_predbuf, hb.data(), sizeof(unsigned) * 32 * kWMax,
                        cudaMemcpyHostToDevice));
    const std::size_t dest_b = static_cast<std::size_t>(kWMax) * 8u * 1024u * 1024u;
    MB_CHECK(cudaMalloc(&d_base, dest_b));
    MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * kWMax * 128 * kStamps));
    struct BC { const char *name; int flags; int groups; int launches; int reps;
                void (*k)(std::uint64_t *, unsigned *, char *, int, int,
                          unsigned *, unsigned, const unsigned *); };
    const BC cells[] = {
      {"none",   0, 48, 8, 17, k_bridge<0, 48>},
      {"filler", B_FILLER, 48, 8, 17, k_bridge<B_FILLER, 48>},
      // "pred" cell dropped: ptxas lowers the per-store @p guard to
      // BSSY/BRA branches instead of a predicated STG on sm_90a.
      {"lanedata", B_LANEDATA, 48, 8, 17, k_bridge<B_LANEDATA, 48>},
      {"once",   B_NOFENCE, 48, 64, 2, k_bridge<B_NOFENCE, 48>},
      {"short",  B_NOFENCE, 16, 64, 2, k_bridge<B_NOFENCE, 16>},
      {"all",    B_FILLER | B_LANEDATA | B_NOFENCE, 16, 64, 2,
       k_bridge<B_FILLER | B_LANEDATA | B_NOFENCE, 16>},
    };
    const int W = 4;
    std::string err;
    for (const auto &c : cells) {
      if (bridge_name != "all_cells" && bridge_name != c.name) continue;
      const std::string variant =
          std::string("epi.") + c.name + ".x" + std::to_string(W);
      MB_CHECK(cudaFuncSetAttribute(
          c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
      std::vector<std::vector<double>> gaps(kStamps);
      std::vector<double> totals;
      int fails = 0;
      for (int l = 0; l < c.launches; ++l) {
        MB_CHECK(cudaMemset(d_base, 0, dest_b));
        MB_CHECK(cudaMemset(d_arrived, 0, sizeof(unsigned)));
        c.k<<<1, 32 * W>>>(d_out, d_sink, d_base, W, c.reps, d_arrived,
                           static_cast<unsigned>(W), d_predbuf);
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
        if (mb_take_err()) { err = "mbarrier wait timeout"; break; }
        std::vector<std::uint64_t> h(
            static_cast<std::size_t>(kWMax) * c.reps * kStamps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < c.reps; ++r)
          for (int wp = 0; wp < W; ++wp) {
            const std::uint64_t *row = &h[(wp * c.reps + r) * kStamps];
            for (int g = 1; g <= c.groups; ++g)
              gaps[g].push_back(static_cast<double>(row[g] - row[g - 1]));
            totals.push_back(static_cast<double>(row[c.groups] - row[0]));
          }
      }
      const std::string params =
          "\"geometry\":\"epi\",\"bridge\":\"" + std::string(c.name) +
          "\",\"groups\":" + std::to_string(c.groups) +
          ",\"warps\":" + std::to_string(W) +
          ",\"launches\":" + std::to_string(c.launches);
      const std::string extra =
          "\"failed_launches\":" + std::to_string(fails);
      if (!err.empty()) { w.row(variant, 0, params, {}, extra); break; }
      w.row(variant, 0, params, totals, extra);   // u=0: total window cycles
      for (int g = 1; g <= c.groups; ++g)
        w.row(variant, g, params, gaps[g], extra);
      std::fprintf(stderr, "%s: %d launches\n", variant.c_str(), c.launches);
      if (bridge_name != "all_cells") break;
    }
    if (!err.empty()) {
      w.fail(bridge_name, 0, "\"geometry\":\"epi\"", err.c_str());
      std::fprintf(stderr, "bridge error: %s\n", err.c_str());
      return 1;
    }
    MB_CHECK(cudaFree(d_base));
    MB_CHECK(cudaFree(d_out));
    MB_CHECK(cudaFree(d_sink));
    MB_CHECK(cudaFree(d_arrived));
    MB_CHECK(cudaFree(d_predbuf));
    return 0;
  }
  if (v2) {
    const MbDevice dev = mb_query_device();
    unsigned char *d_act = nullptr;
    int *d_order = nullptr;
    unsigned *d_arrived = nullptr, *d_parts = nullptr;
    std::vector<int> set_ids;
    int n_active = 1, ctas = 1;
    if (!active_file.empty()) {   // grid covers every SM; CTA per SM via smem
      std::vector<unsigned char> act(256, 0);
      std::vector<int> ord(256, -1);
      FILE *af = std::fopen(active_file.c_str(), "r");
      if (!af) { std::fprintf(stderr, "cannot open %s\n", active_file.c_str()); return 2; }
      for (int v; std::fscanf(af, "%d", &v) == 1;)
        if (v >= 0 && v < 256 && !act[v]) { act[v] = 1; ord[v] = static_cast<int>(set_ids.size()); set_ids.push_back(v); }
      std::fclose(af);
      n_active = static_cast<int>(set_ids.size());
      ctas = dev.sm_count;
      MB_CHECK(cudaMalloc(&d_act, 256));
      MB_CHECK(cudaMemcpy(d_act, act.data(), 256, cudaMemcpyHostToDevice));
      MB_CHECK(cudaMalloc(&d_order, sizeof(int) * 256));
      MB_CHECK(cudaMemcpy(d_order, ord.data(), sizeof(int) * 256, cudaMemcpyHostToDevice));
      MB_CHECK(cudaMalloc(&d_parts, sizeof(unsigned) * ctas));
    }
    MB_CHECK(cudaMalloc(&d_arrived, sizeof(unsigned)));
    const std::size_t dest_bytes =
        static_cast<std::size_t>(n_active) * kWMax * 8u * 1024u * 1024u;
    MB_CHECK(cudaMalloc(&d_base, dest_bytes));
    MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * n_active * kWMax *
                                args.reps * kStamps));
    const int W = xwarps > 0 ? xwarps : 4;
    std::vector<int> geo_list;
    if (geos.empty()) geo_list = {G_FULL16, G_LINE32, G_ROW4};
    else {
      std::string s = geos;
      for (std::size_t pos = 0; pos <= s.size();) {
        const auto comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        bool found = false;
        for (const auto &g : kGeos)
          if (tok == g.name) { geo_list.push_back(g.geo); found = true; }
        if (!found && !tok.empty()) { std::fprintf(stderr, "unknown geo %s\n", tok.c_str()); return 2; }
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
    for (int gi : geo_list) {
      const auto &g = kGeos[gi];
      // v2 multi-SM / single-SM rows keep <geo>.<set>; v3 rows (explicit
      // --dest) are <geo>.<dest>.
      const std::string tag = dest_given ? dest : set_name;
      const std::string variant = std::string(g.name) + "." + tag + ".x" + std::to_string(W);
      if (!mb_selected(args, variant)) continue;
      std::vector<std::vector<double>> gaps(kStamps);
      std::vector<double> sm_steady;      // one median per participating SM
      int discarded = 0, launches_ok = 0;
      const bool virgin = dest == "virgin";
      const std::uint64_t rep_stride =
          virgin ? rep_stride_for(gi) : 0;
      const std::size_t vbytes = static_cast<std::size_t>(n_active) * kWMax *
                                 rep_stride * args.reps;
      for (int l = 0; l < args.launches; ++l) {
        char *d_lbase = d_base;
        if (virgin) {
          MB_CHECK(cudaMalloc(&d_lbase, vbytes));   // never written by host
        } else {
          MB_CHECK(cudaMemset(d_base, 0, dest_bytes));
        }
        MB_CHECK(cudaMemset(d_arrived, 0, sizeof(unsigned)));
        if (d_parts) MB_CHECK(cudaMemset(d_parts, 0, sizeof(unsigned) * ctas));
        launch_x(gi, d_out, d_sink, d_lbase, g.stride, g.lanes, W, args.reps,
                 d_arrived, static_cast<unsigned>(n_active * W), d_act,
                 d_order, d_parts, ctas, rep_stride);
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (virgin) MB_CHECK(cudaFree(d_lbase));
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(), cudaGetErrorString(e));
          break;
        }
        int parts = 0;
        if (d_parts) {
          std::vector<unsigned> h(ctas);
          MB_CHECK(cudaMemcpy(h.data(), d_parts, sizeof(unsigned) * ctas, cudaMemcpyDeviceToHost));
          for (unsigned v : h) if (v && !(v & 0x80000000u)) ++parts;
          if (parts != n_active) { ++discarded; continue; }
        }
        ++launches_ok;
        std::vector<std::uint64_t> h(static_cast<std::size_t>(n_active) * kWMax *
                                     args.reps * kStamps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<std::vector<double>> sm_gaps(n_active);
        for (int r = 1; r < args.reps; ++r)
          for (int wp = 0; wp < W; ++wp)
            for (int s = 0; s < n_active; ++s) {
              const int wg = s * kWMax + wp;
              const std::uint64_t *row = &h[(wg * args.reps + r) * kStamps];
              for (int g2 = 1; g2 <= kGroups; ++g2) {
                const double d = static_cast<double>(row[g2] - row[g2 - 1]);
                gaps[g2].push_back(d);
                if (g2 >= 33) sm_gaps[s].push_back(d);
              }
            }
        for (int s = 0; s < n_active; ++s)
          if (!sm_gaps[s].empty()) sm_steady.push_back(mb_pct(sm_gaps[s], 0.5));
      }
      // per-SM steady gap medians (groups 33..48), across warps and reps
      const std::string params =
          "\"geometry\":\"" + std::string(g.name) + "\"" +
          ",\"set\":\"" + set_name + "\"" +
          ",\"dest\":\"" + dest + "\"" +
          ",\"participants\":" + std::to_string(d_parts ? n_active : 1) +
          ",\"width_bytes\":" + std::to_string(g.width) +
          ",\"stride_bytes\":" + std::to_string(g.stride) +
          ",\"lanes\":" + std::to_string(g.lanes) +
          ",\"warps\":" + std::to_string(W);
      const std::string extra =
          "\"sm_steady\":{\"p10\":" + std::to_string(mb_pct(sm_steady, 0.1)) +
          ",\"p50\":" + std::to_string(mb_pct(sm_steady, 0.5)) +
          ",\"p90\":" + std::to_string(mb_pct(sm_steady, 0.9)) + "}" +
          ",\"discarded_launches\":" + std::to_string(discarded);
      for (int g2 = 1; g2 <= kGroups; ++g2) {
        std::vector<double> smp = gaps[g2];
        std::string ex = extra;
        if (smp.size() > 2000) {   // quantiles only; p50 always present
          ex += ",\"q\":{\"n\":" + std::to_string(smp.size()) +
                ",\"p10\":" + std::to_string(mb_pct(smp, 0.1)) +
                ",\"p50\":" + std::to_string(mb_pct(smp, 0.5)) +
                ",\"p90\":" + std::to_string(mb_pct(smp, 0.9)) + "}";
          smp = {mb_pct(smp, 0.1), mb_pct(smp, 0.5), mb_pct(smp, 0.9)};
        }
        w.row(variant, g2, params, smp, ex);
      }
      std::fprintf(stderr, "%s: launches %d ok, %d discarded\n",
                   variant.c_str(), launches_ok, discarded);
    }
    MB_CHECK(cudaFree(d_base));
    MB_CHECK(cudaFree(d_out));
    MB_CHECK(cudaFree(d_sink));
    return 0;
  }

  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 8 * args.reps * kStamps));
  MB_CHECK(cudaMalloc(&d_base, 16u << 20));  // 384 req * 32 KiB + warp/lane span < 16 MiB
  struct Cell { int v; int width; int stride; int lanes; int warps; };
  std::vector<Cell> cells;
  for (int stride : {4, 32, 64, 128})
    for (int lanes : {8, 16, 32})
      for (int warps : {1, 2, 4, 8})
        cells.push_back({V_B32, 4, stride, lanes, warps});
  for (int stride : {2, 32, 128})
    for (int warps : {2, 8}) cells.push_back({V_U16, 2, stride, 32, warps});
  for (int stride : {8, 32, 128})
    for (int warps : {2, 8}) cells.push_back({V_V2, 8, stride, 32, warps});
  for (int stride : {16, 32, 128})
    for (int warps : {2, 8}) cells.push_back({V_V4, 16, stride, 32, warps});
  for (int warps : {1, 8}) cells.push_back({V_NONE, 0, 0, 32, warps});
  for (const auto &c : cells) {
    const std::string variant =
        c.v == V_NONE
            ? "none.x" + std::to_string(c.warps)
            : "w" + std::to_string(c.width) + ".s" + std::to_string(c.stride) +
                  ".l" + std::to_string(c.lanes) + ".x" + std::to_string(c.warps);
    if (!mb_selected(args, variant)) continue;
    // samples per group: gap t[g] - t[g-1], one per (rep >= 1, warp)
    std::vector<std::vector<double>> gaps(kStamps);
    for (int l = 0; l < args.launches; ++l) {
      MB_CHECK(cudaMemset(d_base, 0, 16u << 20));
      switch (c.v) {
        case V_U16:
          k_flood<V_U16><<<1, 32 * c.warps>>>(d_out, d_sink, d_base, c.stride,
                                              c.lanes, args.reps);
          break;
        case V_B32:
          k_flood<V_B32><<<1, 32 * c.warps>>>(d_out, d_sink, d_base, c.stride,
                                              c.lanes, args.reps);
          break;
        case V_V2:
          k_flood<V_V2><<<1, 32 * c.warps>>>(d_out, d_sink, d_base, c.stride,
                                             c.lanes, args.reps);
          break;
        case V_V4:
          k_flood<V_V4><<<1, 32 * c.warps>>>(d_out, d_sink, d_base, c.stride,
                                             c.lanes, args.reps);
          break;
        default:
          k_flood<V_NONE><<<1, 32 * c.warps>>>(d_out, d_sink, d_base, c.stride,
                                               c.lanes, args.reps);
      }
      const cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(c.warps * args.reps * kStamps);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                          cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r)
        for (int wp = 0; wp < c.warps; ++wp) {
          const std::uint64_t *row = &h[(wp * args.reps + r) * kStamps];
          for (int g = 1; g <= kGroups; ++g)
            gaps[g].push_back(static_cast<double>(row[g] - row[g - 1]));
        }
    }
    const std::string params =
        "\"width_bytes\":" + std::to_string(c.width) +
        ",\"stride_bytes\":" + std::to_string(c.stride) +
        ",\"lanes\":" + std::to_string(c.lanes) +
        ",\"warps\":" + std::to_string(c.warps);
    for (int g = 1; g <= kGroups; ++g) w.row(variant, g, params, gaps[g]);
  }
  MB_CHECK(cudaFree(d_base));
  MB_CHECK(cudaFree(d_sink));
  MB_CHECK(cudaFree(d_out));
  return 0;
}
