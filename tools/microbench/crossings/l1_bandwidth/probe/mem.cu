// Global-load throughput per SM by memory level (memory.{l1,l2,dram}.service
// sector_bytes_per_cycle, memory.*.capacity outstanding_requests).
//
//   stream LVL VEC W   one CTA of W warps on one SM (or one CTA per SM on
//                      every SM: "all"), each warp issues K independent
//                      VEC-byte-per-lane loads (VEC = 4: LDG.E, 128 B per warp
//                      instruction = 4 sectors; VEC = 16: LDG.E.128, 512 B =
//                      16 sectors) between bar.sync's, 8 loads in flight per
//                      warp.  bytes per cycle per SM = W * K * 32 * VEC /
//                      slope-over-K-free cycles (slope over K).
//     plain (weak) ld.global -> LDG.E / LDG.E.128, the captures' common form
//     l1:   32 KiB per-CTA region, warmed (L1 hits)
//     l2:   4 MiB per-CTA region (128 KiB per CTA on all SMs), warmed: misses
//           a 256 KiB L1, hits L2
//     dram: fresh region of an 8 GiB buffer per launch
// Same bytes per cycle for VEC 4 and 16 -> a bandwidth limit; same warp
// instructions per cycle -> a per-request (outstanding) limit.
#include "../mb.cuh"

#include <string>
#include <tuple>
#include <vector>

template <int VEC>
struct Vec;
template <>
struct Vec<4> {
  using T = std::uint32_t;
};
template <>
struct Vec<16> {
  using T = uint4;
};

template <int LVL, int VEC>
__device__ __forceinline__ std::uint32_t ld(const void* p) {
  if constexpr (VEC == 4) {
    std::uint32_t v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
  } else {
    std::uint32_t a, b, c, d;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
    return a ^ b ^ c ^ d;
  }
}

// region: per-CTA base; span: bytes the warp walks (wraps); K loads per warp, D per batch.
// MAXT is the __launch_bounds__ block-size bound: warpq cells (W <= 8) use 256
// so ptxas can hold D = 32 LDG.E.128 results in registers.
template <int LVL, int VEC, int K, int D = 8, int MAXT = 1024>
__global__ void __launch_bounds__(MAXT, 1) k_stream(const unsigned char* buf, std::size_t region_bytes,
                                                    std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                    int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const unsigned char* base = buf + fresh_offset + static_cast<std::size_t>(blockIdx.x) * region_bytes;
  constexpr std::uint32_t kWarpBytes = 32 * VEC;
  // each warp walks its own power-of-two segment with immediate offsets
  // (LDG [R + j * kWarpBytes]): one pointer update per 8 loads, no per-load
  // address arithmetic
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  const unsigned char* wbase = base + static_cast<std::size_t>(warp) * seg + lane * VEC;
  const unsigned char* wseg = base + static_cast<std::size_t>(warp) * seg;
  std::uint32_t acc = 0;
  if (LVL != 2) {
    for (std::size_t o = 0; o < seg; o += kWarpBytes) acc ^= ld<LVL, VEC>(wbase + o);
  }
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; i += D) {
      const unsigned char* p = wbase + off;
      std::uint32_t v[D];
#pragma unroll
      for (int j = 0; j < D; ++j) v[j] = ld<LVL, VEC>(p + j * kWarpBytes);
#pragma unroll
      for (int j = 0; j < D; ++j) acc ^= v[j];
      off = (off + D * kWarpBytes) & (seg - 1);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[static_cast<std::size_t>(blockIdx.x) * reps + rep] = t1 - t0;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// stores: same walk with st.global (STG.E / STG.E.128); LVL 1 = 4 MiB region
// (stays in L2), LVL 2 = fresh region (written back to DRAM)
template <int LVL, int VEC, int K, int D = 8>
__global__ void __launch_bounds__(1024, 1) k_store(const unsigned char* buf, std::size_t region_bytes,
                                                   std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                   int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  unsigned char* base = const_cast<unsigned char*>(buf) + fresh_offset + static_cast<std::size_t>(blockIdx.x) * region_bytes;
  constexpr std::uint32_t kWarpBytes = 32 * VEC;
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  unsigned char* wbase = base + static_cast<std::size_t>(warp) * seg + lane * VEC;
  const std::uint32_t val = threadIdx.x;
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; i += D) {
      unsigned char* p = wbase + off;
#pragma unroll
      for (int j = 0; j < D; ++j) {
        if constexpr (VEC == 4)
          asm volatile("st.global.u32 [%0], %1;" ::"l"(p + j * kWarpBytes), "r"(val) : "memory");
        else
          asm volatile("st.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(p + j * kWarpBytes), "r"(val) : "memory");
      }
      off = (off + D * kWarpBytes) & (seg - 1);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[static_cast<std::size_t>(blockIdx.x) * reps + rep] = t1 - t0;
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = val;
}

// Global load whose first 32-bit word feeds the chain while the rest of the
// words must stay live so ptxas keeps the full-width access.
template <int LVL, int VEC>
__device__ __forceinline__ void ld4(const void* p, std::uint32_t& a, std::uint32_t& b, std::uint32_t& c,
                                    std::uint32_t& d) {
  if constexpr (VEC == 4) {
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(p) : "memory");
    b = c = d = 0;
  } else {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  }
}

// warp_outstanding r2 chain kernel (amendment A8): load i's address depends on
// z[(i-D) mod D] = first-word>>31 of load i-D, so at most D loads are in
// flight and ptxas cannot hoist past D (the buffer is memset to 0x01, so
// z == 0 but that is opaque).  One rep walks K loads then wraps off.
template <int LVL, int VEC, int D, int K, int STRIDE = 0>
__global__ void __launch_bounds__(256, 1) k_chain(const unsigned char* buf, std::size_t region_bytes,
                                                  std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                  int reps) {
  static_assert(LVL != 0 || K * 32 * VEC <= (32 << 10),
                "l1 chain rep must fit the 32 KiB region");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const unsigned char* base = buf + fresh_offset + static_cast<std::size_t>(blockIdx.x) * region_bytes;
  constexpr std::uint32_t kWarpBytes = 32 * VEC;
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  const unsigned char* wbase = base + static_cast<std::size_t>(warp) * seg + lane * VEC;
  const unsigned char* wseg = base + static_cast<std::size_t>(warp) * seg;
  std::uint32_t acc = 0;
  if (LVL != 2) {
    for (std::size_t o = 0; o < seg; o += kWarpBytes) acc ^= ld<LVL, VEC>(wbase + o);
  }
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    // z[] must stay in registers (an indexed array lands in local memory):
    // the chain is expanded over a compile-time index sequence with the D
    // carry slots in a tuple addressed by constexpr indices, and the K loads
    // are walked in blocks of D with a rolled outer loop so the scheduling
    // window ptxas sees stays ~2D loads (full-K expansion spills past 255
    // registers at high D).  Slots are explicit parameters, never
    // [&]-captured: capturing the tuple by reference makes it address-taken.
    auto z = []<std::size_t... J>(std::index_sequence<J...>) {
      return std::tuple{(static_cast<void>(J), std::uint32_t{0})...};
    }(std::make_index_sequence<D>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto step = [wbase, wseg, seg, lane, off](auto& zz, auto& aa, auto ic) {
        const unsigned char* p =
            STRIDE == 0
                ? wbase + off + ic * kWarpBytes + (ic >= D ? std::get<ic % D>(zz) : 0)
                : wseg + ((off + (ic * 4 + (lane >> 3)) * static_cast<std::size_t>(STRIDE) +
                           (ic >= D ? std::get<ic % D>(zz) : 0) + (lane & 7) * 16) &
                          (seg - 1));
        std::uint32_t a, b, c, d;
        ld4<LVL, VEC>(p, a, b, c, d);
        std::get<ic % D>(zz) = (a ^ b ^ c ^ d) >> 31;
        (void)aa;
      };
      (step(z, acc, std::integral_constant<std::size_t, I>{}), ...);
    }(std::make_index_sequence<K>{});
    acc ^= std::apply([](auto... v) { return (v ^ ...); }, z);
    off = (off + (STRIDE == 0 ? K * kWarpBytes : K * 4 * static_cast<std::size_t>(STRIDE))) & (seg - 1);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[static_cast<std::size_t>(blockIdx.x) * reps + rep] = t1 - t0;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// warp_outstanding r2 cp.async chain (amendment A8): one copy per group with
// wait_group D-1 before it -> exactly D copies in flight, sliding; a final
// wait_group 0 inside the window drains the tail.
template <int LVL, int D, int K>
__global__ void __launch_bounds__(256, 1) k_cpchain(const unsigned char* buf, std::size_t region_bytes,
                                                    std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                    int reps) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const unsigned char* base = buf + fresh_offset + static_cast<std::size_t>(blockIdx.x) * region_bytes;
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  const unsigned char* wbase = base + static_cast<std::size_t>(warp) * seg + lane * 16;
  unsigned char* wsmem = smem + static_cast<std::size_t>(warp) * D * 512;
  const std::uint32_t saddr = static_cast<std::uint32_t>(__cvta_generic_to_shared(wsmem + lane * 16));
  std::uint32_t acc = 0;
  if (LVL != 2) {
    for (std::size_t o = 0; o < seg; o += 512) acc ^= ld<LVL, 16>(wbase + o);
  }
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
      asm volatile("cp.async.wait_group %0;" ::"n"(D - 1) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(saddr + (i % D) * 512),
                   "l"(wbase + off + i * 512) : "memory");
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    off = (off + K * 512) & (seg - 1);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[static_cast<std::size_t>(blockIdx.x) * reps + rep] = t1 - t0;
  }
  __syncthreads();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc ^ *reinterpret_cast<volatile std::uint32_t*>(wsmem + lane * 16);
}

// warp_outstanding r3 staged-copies probe (A12): the corpus shape — G = 32
// copies per commit_group, S stages in flight (wait_group S-1 before each
// group), one wait_group 0 drain inside the window.
template <int LVL, int S, int K, int STRIDE = 0>
__global__ void __launch_bounds__(256, 1) k_cpstage(const unsigned char* buf, std::size_t region_bytes,
                                                    std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                    int reps) {
  extern __shared__ __align__(16) unsigned char smem[];
  constexpr int G = 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const unsigned char* base = buf + fresh_offset + static_cast<std::size_t>(blockIdx.x) * region_bytes;
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  const unsigned char* wbase = base + static_cast<std::size_t>(warp) * seg + lane * 16;
  const unsigned char* wseg = base + static_cast<std::size_t>(warp) * seg;
  unsigned char* wsmem = smem + static_cast<std::size_t>(warp) * S * G * 512;
  const std::uint32_t saddr = static_cast<std::uint32_t>(__cvta_generic_to_shared(wsmem + lane * 16));
  std::uint32_t acc = 0;
  if (LVL != 2) {
    for (std::size_t o = 0; o < seg; o += 512) acc ^= ld<LVL, 16>(wbase + o);
  }
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int g = 0; g < K / G; ++g) {
      asm volatile("cp.async.wait_group %0;" ::"n"(S - 1) : "memory");
#pragma unroll
      for (int j = 0; j < G; ++j) {
        const std::size_t i = static_cast<std::size_t>(g) * G + j;
        const unsigned char* src =
            STRIDE == 0
                ? wbase + off + i * 512
                : wseg + ((off + (i * 4 + (lane >> 3)) * static_cast<std::size_t>(STRIDE) +
                           (lane & 7) * 16) &
                          (seg - 1));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(saddr + ((g % S) * G + j) * 512),
                     "l"(src) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    off = (off + (STRIDE == 0 ? K * 512 : K * 4 * static_cast<std::size_t>(STRIDE))) & (seg - 1);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[static_cast<std::size_t>(blockIdx.x) * reps + rep] = t1 - t0;
  }
  __syncthreads();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc ^ *reinterpret_cast<volatile std::uint32_t*>(wsmem + lane * 16);
}

// A7 flush: 512 MiB of STG.E.128 across all SMs into a dedicated scratch
// buffer, run (and synchronised) before every dram launch so the fresh
// region is guaranteed L2-cold.
__global__ void __launch_bounds__(1024, 1) k_flush(unsigned char* buf, std::size_t bytes) {
  const std::size_t idx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x * 16;
  for (std::size_t o = idx; o < bytes; o += stride)
    asm volatile("st.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(buf + o), "r"(threadIdx.x) : "memory");
}

using Kern = void (*)(const unsigned char*, std::size_t, std::size_t, std::uint64_t*, std::uint32_t*, int);
struct Case {
  int lvl, vec, k;
  Kern f;
  bool store;
  int depth = 8;
  int kind = 0;      // 0 = stream/store, 1 = chain, 2 = cpchain
  int wmask = 15;    // warpq measured cells: allowed W subset of {1,2,4,8}
  int stride = 0;   // stride cells: bytes between the 4 chunks of a warp copy
};

template <int LVL, int VEC, int... Ks>
void add(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({LVL, VEC, Ks, &k_stream<LVL, VEC, Ks>, false}), ...);
}
template <int LVL, int VEC, int... Ks>
void adds(std::vector<Case>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({LVL, VEC, Ks, &k_store<LVL, VEC, Ks>, true}), ...);
}
// sm_dram_port_protocol cells: explicit in-flight depth D, K = {2D, 4D, 8D}
template <int LVL, int VEC, int D>
void add_depth(std::vector<Case>& v, bool store) {
  if (store) {
    v.push_back({LVL, VEC, 2 * D, &k_store<LVL, VEC, 2 * D, D>, true, D});
    v.push_back({LVL, VEC, 4 * D, &k_store<LVL, VEC, 4 * D, D>, true, D});
    v.push_back({LVL, VEC, 8 * D, &k_store<LVL, VEC, 8 * D, D>, true, D});
  } else {
    v.push_back({LVL, VEC, 2 * D, &k_stream<LVL, VEC, 2 * D, D>, false, D});
    v.push_back({LVL, VEC, 4 * D, &k_stream<LVL, VEC, 4 * D, D>, false, D});
    v.push_back({LVL, VEC, 8 * D, &k_stream<LVL, VEC, 8 * D, D>, false, D});
  }
}
// warp_outstanding_protocol r2 cells: pinned sliding-window chains (A8).
template <int LVL, int VEC, int D>
void add_chain(std::vector<Case>& v, int wmask) {
  v.push_back({LVL, VEC, 2 * D, &k_chain<LVL, VEC, D, 2 * D>, false, D, 1, wmask});
  v.push_back({LVL, VEC, 4 * D, &k_chain<LVL, VEC, D, 4 * D>, false, D, 1, wmask});
  v.push_back({LVL, VEC, 8 * D, &k_chain<LVL, VEC, D, 8 * D>, false, D, 1, wmask});
}
template <int LVL, int VEC, int... Ds>
void add_chain_v(std::vector<Case>& v, int wmask, std::integer_sequence<int, Ds...>) {
  (add_chain<LVL, VEC, Ds>(v, wmask), ...);
}
template <int LVL, int D>
void add_cpc(std::vector<Case>& v, int wmask = 15) {
  v.push_back({LVL, 16, 2 * D, &k_cpchain<LVL, D, 2 * D>, false, D, 2, wmask});
  v.push_back({LVL, 16, 4 * D, &k_cpchain<LVL, D, 4 * D>, false, D, 2, wmask});
  v.push_back({LVL, 16, 8 * D, &k_cpchain<LVL, D, 8 * D>, false, D, 2, wmask});
}
template <int LVL, int... Ds>
void add_cpc_l(std::vector<Case>& v, int wmask, std::integer_sequence<int, Ds...>) {
  (add_cpc<LVL, Ds>(v, wmask), ...);
}
// cpstage cells: K = {2,4,8} x 32 x S copies, depth param = S*32.
template <int LVL, int S>
void add_cps(std::vector<Case>& v, int wmask) {
  constexpr int G = 32;
  v.push_back({LVL, 16, 2 * G * S, &k_cpstage<LVL, S, 2 * G * S>, false, S * G, 3, wmask});
  v.push_back({LVL, 16, 4 * G * S, &k_cpstage<LVL, S, 4 * G * S>, false, S * G, 3, wmask});
  v.push_back({LVL, 16, 8 * G * S, &k_cpstage<LVL, S, 8 * G * S>, false, S * G, 3, wmask});
}
// dram_stride_protocol cells: cpstage S=3 G=32 and chain v16 D=32, W=4,
// STRIDE apart the four 128 B chunks of each warp instruction.
template <int LVL, int STRIDE>
void add_stride(std::vector<Case>& v, int wmask) {
  constexpr int S = 3, G = 32, D = 32;
  v.push_back({LVL, 16, 2 * G * S, &k_cpstage<LVL, S, 2 * G * S, STRIDE>, false, S * G, 3, wmask, STRIDE});
  v.push_back({LVL, 16, 4 * G * S, &k_cpstage<LVL, S, 4 * G * S, STRIDE>, false, S * G, 3, wmask, STRIDE});
  v.push_back({LVL, 16, 8 * G * S, &k_cpstage<LVL, S, 8 * G * S, STRIDE>, false, S * G, 3, wmask, STRIDE});
  v.push_back({LVL, 16, 2 * D, &k_chain<LVL, 16, D, 2 * D, STRIDE>, false, D, 1, wmask, STRIDE});
  v.push_back({LVL, 16, 4 * D, &k_chain<LVL, 16, D, 4 * D, STRIDE>, false, D, 1, wmask, STRIDE});
  v.push_back({LVL, 16, 8 * D, &k_chain<LVL, 16, D, 8 * D, STRIDE>, false, D, 1, wmask, STRIDE});
}
template <int LVL, int... STRIDES>
void add_stride_l(std::vector<Case>& v, int wmask, std::integer_sequence<int, STRIDES...>) {
  (add_stride<LVL, STRIDES>(v, wmask), ...);
}

// --cells smport: single-SM DRAM port plateau (tools/microbench/
// sm_dram_port_protocol.json).  Cells are (level, vec, warps, depth, all);
// the wave-3 default set below is untouched.
struct SmPortCell {
  int lvl, vec, warps, depth, all;
  bool store;
};
static const SmPortCell kSmPortCells[] = {
    {2, 16, 8, 8, 0, false},  {2, 16, 16, 8, 0, false}, {2, 16, 32, 8, 0, false},
    {2, 16, 8, 16, 0, false}, {2, 16, 16, 16, 0, false}, {2, 16, 8, 32, 0, false},
    {2, 4, 32, 8, 0, false},  {2, 4, 16, 16, 0, false},
    {2, 16, 32, 8, 0, true},  {2, 16, 16, 16, 0, true},  {2, 16, 8, 32, 0, true},
    {1, 16, 32, 8, 0, false}, {2, 16, 32, 8, 1, false},
    {1, 16, 32, 8, 0, true},  {2, 16, 32, 8, 1, true},
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "mem");
  int nsm = 0;
  MB_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const std::size_t dram_bytes = std::size_t(1) << 33;  // 8 GiB
  unsigned char* buf = nullptr;
  MB_CHECK(cudaMalloc(&buf, dram_bytes));
  MB_CHECK(cudaMemset(buf, 1, dram_bytes));
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * nsm * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * nsm * 1024));
  std::vector<Case> cases;
  const bool smport = args.cells == "smport";
  const bool warpq = args.cells == "warpq";
  const bool stride = args.cells == "stride";
  const std::size_t flush_bytes = std::size_t(512) << 20;
  unsigned char* scratch = nullptr;
  if (warpq || stride) MB_CHECK(cudaMalloc(&scratch, flush_bytes));
  if (warpq) {
    // warp_outstanding_protocol r2: chain (LDG sliding window) at W 1/2/4/8,
    // L1 overhead reference at W=1 only, cpchain (LDGSTS), plus the two w32
    // smport controls.
    const std::integer_sequence<int, 4, 8, 12, 16, 24, 32> ds;
    add_chain_v<1, 4>(cases, 15, ds);
    add_chain_v<1, 16>(cases, 15, ds);
    add_chain_v<2, 4>(cases, 15, ds);
    add_chain_v<2, 16>(cases, 15, ds);
    add_chain_v<0, 4>(cases, 1, std::integer_sequence<int, 4, 8>{});
    add_chain_v<0, 16>(cases, 1, std::integer_sequence<int, 4, 8>{});
    add_cpc_l<1>(cases, 15, std::integer_sequence<int, 8, 16, 32>{});
    add_cpc_l<2>(cases, 15, std::integer_sequence<int, 8, 16, 32>{});
    // r3/A12: deeper one-copy groups at W<=4 and staged 32-copy groups.
    // r3/A13 chain v16 D{48,64} dropped: 4 result regs x D + D carry slots
    // exceed 255 registers (D=48 static_depth 41-46, D=64 50-60 + spills).
    add_cpc_l<1>(cases, 7, std::integer_sequence<int, 48, 63>{});
    add_cpc_l<2>(cases, 7, std::integer_sequence<int, 48, 63>{});
    add_cps<1, 2>(cases, 7);
    add_cps<1, 3>(cases, 7);
    add_cps<1, 4>(cases, 3);
    add_cps<2, 2>(cases, 7);
    add_cps<2, 3>(cases, 7);
    add_cps<2, 4>(cases, 3);
    add_depth<1, 16, 8>(cases, false);
    add_depth<2, 16, 8>(cases, false);
  }
  if (stride) {
    const std::integer_sequence<int, 128, 512, 2048, 8192, 32768> ss;
    add_stride_l<1>(cases, 3, ss);
    add_stride_l<2>(cases, 3, ss);
  }
  if (smport) {
    add_depth<2, 16, 8>(cases, false);
    add_depth<2, 16, 16>(cases, false);
    add_depth<2, 16, 32>(cases, false);
    add_depth<2, 4, 8>(cases, false);
    add_depth<2, 4, 16>(cases, false);
    add_depth<2, 16, 8>(cases, true);
    add_depth<2, 16, 16>(cases, true);
    add_depth<2, 16, 32>(cases, true);
    add_depth<1, 16, 8>(cases, false);
    add_depth<1, 16, 8>(cases, true);
  }
  const bool wave3 = !smport && !warpq && !stride;
  if (wave3) add<0, 4>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) add<0, 16>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) add<1, 4>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) add<1, 16>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) add<2, 4>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) add<2, 16>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) adds<1, 16>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) adds<2, 16>(cases, std::integer_sequence<int, 16, 32, 64>{});
  if (wave3) adds<1, 4>(cases, std::integer_sequence<int, 16, 32, 64>{});
  const char* lname[] = {"l1", "l2", "dram"};
  std::size_t fresh = 0;
  for (int all : {0, 1}) {
    for (int wn : {1, 2, 4, 8, 16, 32}) {
      if (all && wn != 32) continue;
      for (const auto& c : cases) {
        // dram: large enough that the timed reps never wrap back onto data a
        // previous rep pulled into L2
        const std::size_t region = c.lvl == 0 ? (32u << 10) : c.lvl == 1 ? (all ? (128u << 10) : (4u << 20))
                                                                         : (std::size_t(all ? 40 : (c.stride ? 512 : 256)) << 20);
        if (smport) {
          bool want = false;
          for (const auto& cell : kSmPortCells)
            want |= cell.lvl == c.lvl && cell.vec == c.vec && cell.warps == wn && cell.depth == c.depth &&
                    cell.all == all && cell.store == c.store;
          if (!want) continue;
        } else if (warpq) {
          if (all) continue;
          if (c.kind != 0 ? !(c.wmask & wn) : wn != 32) continue;
        } else if (stride) {
          if (all || wn != 4) continue;
        } else if (c.store && !(wn == 32 || wn == 8)) {
          continue;
        }
        const std::string variant =
            c.stride
                ? "stride." + std::string(c.kind == 3 ? "cpstage" : "chain") + "." +
                      std::string(lname[c.lvl]) + ".s" + std::to_string(c.stride)
            : c.kind == 1
                ? "chain." + std::string(lname[c.lvl]) + ".v" + std::to_string(c.vec) + ".w" + std::to_string(wn) +
                      ".d" + std::to_string(c.depth)
            : c.kind == 2
                ? "cpchain." + std::string(lname[c.lvl]) + ".v16.w" + std::to_string(wn) + ".d" +
                      std::to_string(c.depth)
            : c.kind == 3
                ? "cpstage." + std::string(lname[c.lvl]) + ".v16.w" + std::to_string(wn) + ".s" +
                      std::to_string(c.depth / 32)
                : std::string(c.store ? "store." : "stream.") + lname[c.lvl] + ".v" + std::to_string(c.vec) + ".w" +
                      std::to_string(wn) + (smport || warpq ? ".d" + std::to_string(c.depth) : std::string()) +
                      (all ? ".all" : ".one");
        if (!mb_selected(args, variant)) continue;
        std::vector<double> s;
        const int grid = all ? nsm : 1;
        for (int l = 0; l < args.launches; ++l) {
          std::size_t off = 0;
          if (c.lvl == 2) {
            if (fresh + region * grid > dram_bytes) fresh = 0;
            off = fresh;
            fresh += region * grid;
          }
          // A7: flush 512 MiB of stores into scratch before every dram
          // launch so the fresh region is L2-cold.
          if ((warpq || stride) && c.lvl == 2) {
            k_flush<<<nsm * 4, 1024>>>(scratch, flush_bytes);
            MB_CHECK(cudaGetLastError());
            MB_CHECK(cudaDeviceSynchronize());
          }
          const std::size_t smem = c.kind == 2 ? static_cast<std::size_t>(wn) * c.depth * 512
                                   : c.kind == 3 ? static_cast<std::size_t>(wn) * (c.depth / 32) * 16384
                                                 : 0;
          if (c.kind >= 2) MB_CHECK(cudaFuncSetAttribute(c.f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
          c.f<<<grid, 32 * wn, smem>>>(buf, region, off, d_out, d_sink, args.reps);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(static_cast<std::size_t>(grid) * args.reps);
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
          for (int b = 0; b < grid; ++b)
            for (int r = 1; r < args.reps; ++r) s.push_back(static_cast<double>(h[static_cast<std::size_t>(b) * args.reps + r]));
        }
        w.row(variant, c.k,
              "\"level\":\"" + std::string(lname[c.lvl]) + "\",\"store\":" + std::to_string(c.store ? 1 : 0) +
                  ",\"vec\":" + std::to_string(c.vec) + ",\"warps\":" +
                  std::to_string(wn) + ",\"depth\":" + std::to_string(c.depth) + ",\"all_sm\":" + std::to_string(all) +
                  ",\"region_bytes\":" + std::to_string(region) +
                  (warpq ? ",\"kind\":\"" + std::string(c.kind == 1 ? "chain" : c.kind == 2 ? "cpchain"
                                                                                      : c.kind == 3 ? "cpstage" : "ldg") +
                               "\""
                         : "") +
                  (c.stride ? ",\"kind\":\"stride_" + std::string(c.kind == 3 ? "cpstage" : "chain") +
                                  "\",\"stride\":" + std::to_string(c.stride)
                            : ""),
              s);
      }
    }
  }
  return w.failures ? 1 : 0;
}
