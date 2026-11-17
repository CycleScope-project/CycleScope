// Local shared memory: dependency.result (timing kind 7, LDS result),
// storage.shared.issue (kind 5, per-SMSP issue interval by bank-conflict
// degree), storage.shared.acceptance_sm (kind 6, per-SM STS acceptance) and
// storage.shared.visibility (kind 3, STS -> CTA-barrier visibility).
//
//   lds_chain W   one warp, lane l reads its own W-byte word at base + l*W
//                 (degree W/4 by the converter's rule), U x { a = ld.shared [a] }
//                 -> slope = LDS issue -> dependent-LDS issue (engine: register
//                 ready = issue + kind 7)
//   issue OP D    every active warp: bar.sync ; t0 ; K independent OP
//                 (LDS to distinct registers / STS of a fixed value, same
//                 address) with degree-D lane addresses (D lanes on bank 0 at
//                 D rows, the rest conflict-free; D = 0: broadcast) ; bar.sync ; t1
//                 active = W warps on SMSP 0 only (warp % 4 == 0) or W warps on
//                 each of the 4 SMSPs.  slope over K / W = cycles per op per
//                 SMSP; all-SMSP vs one-SMSP separates an SM-level limit
//                 (kind 6) from the SMSP one (kind 5).
//   vis S NW      CTA of NW warps; warp 0 (last arriver, jittered) does S STS
//                 then bar.sync; t_w on warp 1.  Engine: barrier release waits
//                 for max(last arrival + kind 17, STS issue + kind 3), so
//                 kind 3 = E[t_w - t_s | S=1] - o_s(t_s->STS) - o_w when it
//                 exceeds the no-store release.
#include "../mb.cuh"

#include <string>
#include <utility>
#include <vector>

constexpr int kWords = 8192;  // 32 KiB

template <int W>
__device__ __forceinline__ std::uint32_t lds_w(std::uint32_t a) {
  std::uint32_t r0, r1, r2, r3;
  if constexpr (W == 4) {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(a) : "memory");
    return r0;
  } else if constexpr (W == 8) {
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
    return r0 + (r1 & 0u);
  } else {
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a)
                 : "memory");
    return r0 + ((r1 ^ r2 ^ r3) & 0u);
  }
}

template <int W, int U>
__global__ void __launch_bounds__(32, 1) k_lds_chain(std::uint64_t* out, std::uint32_t* sink, int reps) {
  __shared__ __align__(16) std::uint32_t buf[kWords];
  const std::uint32_t base = mb_smem_u32(buf);
  for (int i = threadIdx.x; i < kWords; i += 32) buf[i] = base + static_cast<std::uint32_t>(i) * 4u;
  __syncwarp();
  std::uint32_t a = base + threadIdx.x * W;
  std::uint32_t acc = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) a = lds_w<W>(a);
    acc += a;
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = acc;
}

// ldsm_chain X: one warp, U dependent ldmatrix; the next row address
// is derived from the loaded value (a = base + (r & mask)), same
// window rules as lds_chain.  X in {1,2,4}, trans and non-trans.
template <int X, int U, bool TRANSPOSED>
__global__ void __launch_bounds__(32, 1) k_ldsm_chain(
    std::uint64_t* out, std::uint32_t* sink, int reps) {
  constexpr int kWords = 8192;          // 32 KiB: inside static limit
  __shared__ __align__(128) std::uint32_t buf[kWords];
  const std::uint32_t base = mb_smem_u32(buf);
  const int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < kWords; i += 32)
    buf[i] = base + static_cast<std::uint32_t>(i) * 4u;
  __syncwarp();
  const std::uint32_t row =
      ((static_cast<std::uint32_t>(lane >> 3) % X) * 8 + (lane & 7));
  std::uint32_t a = base + row * 16u;
  std::uint32_t r[X];
  std::uint32_t acc = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncwarp();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < U; ++i) {
      if constexpr (X == 1) {
        if constexpr (TRANSPOSED)
          asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                       : "=r"(r[0]) : "r"(a) : "memory");
        else
          asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                       : "=r"(r[0]) : "r"(a) : "memory");
      } else if constexpr (X == 2) {
        if constexpr (TRANSPOSED)
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                       : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
        else
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                       : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
      } else {
        if constexpr (TRANSPOSED)
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                       : "r"(a) : "memory");
        else
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                       : "r"(a) : "memory");
      }
      a = base + (r[0] & (kWords * 4 - 16));
    }
    acc += r[0];
    const std::uint64_t t1 = mb_clock();
    if (lane == 0) out[rep] = t1 - t0;
  }
  sink[threadIdx.x] = acc;
}

// Degree-d pattern for any d in 1..32: lanes 0..d-1 hit bank 0 at d distinct
// rows, the other lanes one distinct bank each (row 0) -> max distinct words
// per bank = d (the converter's bank_conflict_degree).  d = 0: broadcast.
__device__ __forceinline__ std::uint32_t degree_word(int lane, int d) {
  if (d == 0) return 0u;
  if (lane < d) return static_cast<std::uint32_t>(lane * 32);
  return static_cast<std::uint32_t>(lane - d + 1);
}

template <int OP, int K>
__global__ void __launch_bounds__(1024, 1) k_issue(std::uint64_t* out, std::uint32_t* sink, int reps, int d,
                                                   int all_smsp) {
  __shared__ __align__(16) std::uint32_t buf[kWords];
  __shared__ std::uint32_t pin[1024];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < kWords; i += blockDim.x) buf[i] = static_cast<std::uint32_t>(i);
  __syncthreads();
  const bool active = all_smsp || (warp % 4) == 0;
  const std::uint32_t a = mb_pin(mb_smem_u32(buf) + degree_word(lane, d) * 4u, &pin[threadIdx.x]);
  std::uint32_t r[K];
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (active) {
      if constexpr (OP == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k)
          asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(r[k]) : "r"(a) : "memory");
      } else if constexpr (OP == 2) {
        // k_issue_lds_nv (protocol-v2 addendum): non-volatile loads at
        // a + k x 128 B (one 128 B row per k: same bank / degree pattern,
        // distinct addresses so ptxas neither merges nor orders them)
#pragma unroll
        for (int k = 0; k < K; ++k)
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r[k]) : "r"(a + k * 128u));
      } else {
#pragma unroll
        for (int k = 0; k < K; ++k)
          asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(a), "r"(acc) : "memory");
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    if constexpr (OP == 0 || OP == 2) {
#pragma unroll
      for (int k = 0; k < K; ++k) acc ^= r[k];
    }
  }
  sink[threadIdx.x] = acc;
}

// ldmatrix / stmatrix throughput (LDSM / STSM): every active warp issues K
// independent m8n8 x{1,2,4} .b16 operations; lane l supplies row (l & 7) of
// matrix (l >> 3) at a conflict-free 16 B row address (consecutive 16 B rows,
// one 128 B wavefront per matrix); operation k uses its own 512 B block
// (ptxas merges identical LDSM addresses).  Same all-SMSP / SMSP0-only split
// as k_issue.
template <int OP, int X, int K, bool TRANSPOSED>
__global__ void __launch_bounds__(128, 1) k_mtx(std::uint64_t* out, std::uint32_t* sink, int reps, int, int all_smsp) {
  constexpr int kMtxWords = 16384;
  extern __shared__ __align__(128) std::uint32_t buf[];
  __shared__ std::uint32_t pin[128];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < kMtxWords; i += blockDim.x) buf[i] = static_cast<std::uint32_t>(i);
  __syncthreads();
  const bool active = all_smsp || (warp % 4) == 0;
  const std::uint32_t row = static_cast<std::uint32_t>(((lane >> 3) % X) * 8 + (lane & 7));
  const std::uint32_t a = mb_pin(mb_smem_u32(buf) + row * 16u + static_cast<std::uint32_t>(warp & 1) * 16384u, &pin[threadIdx.x]);
  std::uint32_t r[K][X];
  std::uint32_t acc = threadIdx.x;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
    if (active) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if constexpr (OP == 0) {
          if constexpr (X == 1) {
            if constexpr (TRANSPOSED)
              asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                           : "=r"(r[k][0]) : "r"(a + k * 512u) : "memory");
            else
              asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                           : "=r"(r[k][0]) : "r"(a + k * 512u) : "memory");
          } else if constexpr (X == 2) {
            if constexpr (TRANSPOSED)
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                           : "=r"(r[k][0]), "=r"(r[k][1])
                           : "r"(a + k * 512u) : "memory");
            else
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                           : "=r"(r[k][0]), "=r"(r[k][1])
                           : "r"(a + k * 512u) : "memory");
          } else {
            if constexpr (TRANSPOSED)
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                           : "=r"(r[k][0]), "=r"(r[k][1]), "=r"(r[k][2]), "=r"(r[k][3])
                           : "r"(a + k * 512u) : "memory");
            else
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                           : "=r"(r[k][0]), "=r"(r[k][1]), "=r"(r[k][2]), "=r"(r[k][3])
                           : "r"(a + k * 512u) : "memory");
          }
        } else {
          if constexpr (X == 1) {
            if constexpr (TRANSPOSED)
              asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
            else
              asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
          } else if constexpr (X == 2) {
            if constexpr (TRANSPOSED)
              asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
            else
              asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
          } else {
            if constexpr (TRANSPOSED)
              asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %1, %1, %1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
            else
              asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %1, %1, %1};"
                           ::"r"(a + k * 512u), "r"(acc) : "memory");
          }
        }
      }
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    if constexpr (OP == 0) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int x = 0; x < X; ++x) acc ^= r[k][x];
    }
  }
  sink[threadIdx.x] = acc;
}

constexpr std::uint32_t kDelay = 3000;

template <int S>
__global__ void __launch_bounds__(128, 1) k_vis(std::uint64_t* out, int reps, std::uint32_t seed) {
  __shared__ __align__(16) std::uint32_t buf[1024];
  __shared__ std::uint32_t pin[128];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_pin(mb_smem_u32(buf) + static_cast<std::uint32_t>(lane) * 4u, &pin[threadIdx.x]);
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    std::uint64_t t_s = 0;
    if (warp == 0) {
      const std::uint32_t h = (static_cast<std::uint32_t>(rep) * 2654435761u) ^ (seed * 2246822519u);
      const std::uint32_t dly = kDelay + ((h ^ (h >> 15)) & 2047u);
      const std::uint64_t td = mb_clock();
      while (mb_clock() - td < dly) {
      }
      t_s = mb_clock();
#pragma unroll
      for (int s = 0; s < S; ++s)
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(a + static_cast<std::uint32_t>(s) * 128u), "r"(static_cast<std::uint32_t>(rep))
                     : "memory");
    }
    __syncthreads();
    const std::uint64_t t_w = mb_clock();
    if (lane == 0) {
      if (warp == 0) {
        out[4 * rep + 0] = t_s;
        out[4 * rep + 1] = t_w;
      }
      if (warp == 1) out[4 * rep + 2] = t_w;
    }
  }
}

using CKern = void (*)(std::uint64_t*, std::uint32_t*, int);
using IKern = void (*)(std::uint64_t*, std::uint32_t*, int, int, int);
using VKern = void (*)(std::uint64_t*, int, std::uint32_t);

template <int W, int... Us>
void add_chain(std::vector<std::pair<std::pair<int, int>, CKern>>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({{W, Us}, &k_lds_chain<W, Us>}), ...);
}

template <int OP, int... Ks>
void add_issue(std::vector<std::pair<std::pair<int, int>, IKern>>& v, std::integer_sequence<int, Ks...>) {
  (v.push_back({{OP, Ks}, &k_issue<OP, Ks>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "smem");
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 4 * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 1024));
  // --- LDS result chain
  std::vector<std::pair<std::pair<int, int>, CKern>> chains;
  add_chain<4>(chains, std::integer_sequence<int, 1, 2, 4, 8, 16, 32>{});
  add_chain<8>(chains, std::integer_sequence<int, 1, 2, 4, 8, 16, 32>{});
  add_chain<16>(chains, std::integer_sequence<int, 1, 2, 4, 8, 16, 32>{});
  for (const auto& [wu, k] : chains) {
    const std::string variant = "lds_chain.w" + std::to_string(wu.first);
    if (!mb_selected(args, variant)) continue;
    std::vector<double> s;
    for (int l = 0; l < args.launches; ++l) {
      k<<<1, 32>>>(d_out, d_sink, args.reps);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      mb_collect(d_out, args.reps, 1, 0, s);
    }
    w.row(variant, wu.second, "\"width_bytes\":" + std::to_string(wu.first), s);
  }
  // --- issue / acceptance
  std::vector<std::pair<std::pair<int, int>, IKern>> issues;
  add_issue<0>(issues, std::integer_sequence<int, 8, 16, 32>{});
  add_issue<1>(issues, std::integer_sequence<int, 8, 16, 32>{});
  add_issue<2>(issues, std::integer_sequence<int, 8, 16, 32>{});
  for (int all : {0, 1}) {
    for (int wps : {1, 4}) {
      for (int d = 0; d <= 32; ++d) {
        for (const auto& [ok, k] : issues) {
          const char* opn = ok.first == 1 ? "sts" : ok.first == 2 ? "lds_nv" : "lds";
          const std::string variant = std::string(opn) + ".d" + std::to_string(d) +
                                      ".wps" + std::to_string(wps) + (all ? ".all" : ".smsp0");
          if (!mb_selected(args, variant)) continue;
          std::vector<double> s;
          for (int l = 0; l < args.launches; ++l) {
            k<<<1, 128 * wps>>>(d_out, d_sink, args.reps, d, all);
            MB_CHECK(cudaGetLastError());
            MB_CHECK(cudaDeviceSynchronize());
            mb_collect(d_out, args.reps, 1, 0, s);
          }
          w.row(variant, ok.second,
                "\"op\":\"" + std::string(opn) + "\",\"degree_param\":" + std::to_string(d) +
                    ",\"wps\":" + std::to_string(wps) + ",\"all_smsp\":" + std::to_string(all),
                s);
        }
      }
    }
  }
  // --- ldmatrix / stmatrix throughput
  {
    using MKern = void (*)(std::uint64_t*, std::uint32_t*, int, int, int);
    struct M {
      std::string op;
      int x, k;
      bool transposed;
      MKern f;
    };
#define ADD_MTX(OP, NAME, X, K) \
    mtx.push_back({NAME, X, K, false, &k_mtx<OP, X, K, false>}); \
    mtx.push_back({NAME, X, K, true, &k_mtx<OP, X, K, true>})
    std::vector<M> mtx;
#ifndef SMEM_NO_MTX
    ADD_MTX(0, "ldsm", 1, 1);  ADD_MTX(0, "ldsm", 1, 4);
    ADD_MTX(0, "ldsm", 1, 8);  ADD_MTX(0, "ldsm", 1, 16);
    ADD_MTX(0, "ldsm", 1, 32); ADD_MTX(0, "ldsm", 1, 64);
    ADD_MTX(0, "ldsm", 2, 1);  ADD_MTX(0, "ldsm", 2, 4);
    ADD_MTX(0, "ldsm", 2, 8);  ADD_MTX(0, "ldsm", 2, 16);
    ADD_MTX(0, "ldsm", 2, 32); ADD_MTX(0, "ldsm", 2, 64);
    ADD_MTX(0, "ldsm", 4, 1);  ADD_MTX(0, "ldsm", 4, 4);
    ADD_MTX(0, "ldsm", 4, 8);  ADD_MTX(0, "ldsm", 4, 12);
    ADD_MTX(0, "ldsm", 4, 16); ADD_MTX(0, "ldsm", 4, 24);
    ADD_MTX(0, "ldsm", 4, 28);
    ADD_MTX(1, "stsm", 1, 1);  ADD_MTX(1, "stsm", 1, 4);
    ADD_MTX(1, "stsm", 1, 8);  ADD_MTX(1, "stsm", 1, 16);
    ADD_MTX(1, "stsm", 1, 32); ADD_MTX(1, "stsm", 1, 64);
    ADD_MTX(1, "stsm", 2, 1);  ADD_MTX(1, "stsm", 2, 4);
    ADD_MTX(1, "stsm", 2, 8);  ADD_MTX(1, "stsm", 2, 16);
    ADD_MTX(1, "stsm", 2, 32); ADD_MTX(1, "stsm", 2, 64);
    ADD_MTX(1, "stsm", 4, 1);  ADD_MTX(1, "stsm", 4, 4);
    ADD_MTX(1, "stsm", 4, 8);  ADD_MTX(1, "stsm", 4, 16);
    ADD_MTX(1, "stsm", 4, 32); ADD_MTX(1, "stsm", 4, 64);
#endif
#undef ADD_MTX
    for (int all : {0, 1}) {
      for (const auto& m : mtx) {
        const std::string variant =
            "matrix." + m.op + ".x" + std::to_string(m.x) +
            ".mt" + std::to_string(m.transposed) + ".wps1" +
            (all ? ".all" : ".smsp0");
        if (!mb_selected(args, variant)) continue;
        std::vector<double> s;
        for (int l = 0; l < args.launches; ++l) {
          cudaFuncSetAttribute(m.f, cudaFuncAttributeMaxDynamicSharedMemorySize, 67584);
          m.f<<<1, 128, 67584>>>(d_out, d_sink, args.reps, 0, all);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          mb_collect(d_out, args.reps, 1, 0, s);
        }
        w.row(variant, m.k,
              "\"op\":\"" + m.op + "\",\"x\":" + std::to_string(m.x) +
                  ",\"matrix_transposed\":" +
                  (m.transposed ? "1" : "0") +
                  ",\"wps\":1,\"all_smsp\":" + std::to_string(all),
              s);
      }
    }
  }
  // --- ldsm_chain X x{1,2,4}, trans/non-trans, U same set as lds_chain
  {
    struct LC { int x, u; bool tr; void* f; };
    std::vector<LC> lc;
#define ADD_LC(X_, ...)                                                {                                                                    int us[] = {__VA_ARGS__};                                          for (int u_ : us) {                                                  lc.push_back({X_, u_, false, nullptr});                            lc.push_back({X_, u_, true, nullptr});                           }                                                                }
#define LC_KERNEL(X, U)                                                (TRANSPOSED ? (void*)&k_ldsm_chain<X, U, true>                                 : (void*)&k_ldsm_chain<X, U, false>)
    const int us[] = {1, 2, 4, 8, 16};
    for (int u_ : us) {
      for (int x_ : {1, 2, 4}) {
        struct K5 { int x, u; };
        void (*fn_t)(std::uint64_t*, std::uint32_t*, int) = nullptr;
        void (*fn_n)(std::uint64_t*, std::uint32_t*, int) = nullptr;
        switch (x_) {
          case 1:
            switch (u_) {
              case 1: fn_n=&k_ldsm_chain<1,1,false>; fn_t=&k_ldsm_chain<1,1,true>; break;
              case 2: fn_n=&k_ldsm_chain<1,2,false>; fn_t=&k_ldsm_chain<1,2,true>; break;
              case 4: fn_n=&k_ldsm_chain<1,4,false>; fn_t=&k_ldsm_chain<1,4,true>; break;
              case 8: fn_n=&k_ldsm_chain<1,8,false>; fn_t=&k_ldsm_chain<1,8,true>; break;
              default: fn_n=&k_ldsm_chain<1,16,false>; fn_t=&k_ldsm_chain<1,16,true>; break;
            }
            break;
          case 2:
            switch (u_) {
              case 1: fn_n=&k_ldsm_chain<2,1,false>; fn_t=&k_ldsm_chain<2,1,true>; break;
              case 2: fn_n=&k_ldsm_chain<2,2,false>; fn_t=&k_ldsm_chain<2,2,true>; break;
              case 4: fn_n=&k_ldsm_chain<2,4,false>; fn_t=&k_ldsm_chain<2,4,true>; break;
              case 8: fn_n=&k_ldsm_chain<2,8,false>; fn_t=&k_ldsm_chain<2,8,true>; break;
              default: fn_n=&k_ldsm_chain<2,16,false>; fn_t=&k_ldsm_chain<2,16,true>; break;
            }
            break;
          default:
            switch (u_) {
              case 1: fn_n=&k_ldsm_chain<4,1,false>; fn_t=&k_ldsm_chain<4,1,true>; break;
              case 2: fn_n=&k_ldsm_chain<4,2,false>; fn_t=&k_ldsm_chain<4,2,true>; break;
              case 4: fn_n=&k_ldsm_chain<4,4,false>; fn_t=&k_ldsm_chain<4,4,true>; break;
              case 8: fn_n=&k_ldsm_chain<4,8,false>; fn_t=&k_ldsm_chain<4,8,true>; break;
              default: fn_n=&k_ldsm_chain<4,16,false>; fn_t=&k_ldsm_chain<4,16,true>; break;
            }
            break;
        }
        for (int tr = 0; tr < 2; ++tr) {
          auto fn = tr ? fn_t : fn_n;
          const std::string variant =
              "ldsm_chain.x" + std::to_string(x_) + ".mt" +
              std::to_string(tr) + ".u" + std::to_string(u_);
          if (!mb_selected(args, variant)) continue;
          std::vector<double> s2;
          for (int l = 0; l < args.launches; ++l) {
            fn<<<1, 32>>>(d_out, d_sink, args.reps);
            MB_CHECK(cudaGetLastError());
            MB_CHECK(cudaDeviceSynchronize());
            mb_collect(d_out, args.reps, 1, 0, s2);
          }
          w.row(variant, u_,
                "\"op\":\"ldsm_chain\",\"x\":" +
                    std::to_string(x_) +
                    ",\"matrix_transposed\":" + std::to_string(tr),
                s2);
        }
      }
    }
  }
  // --- store visibility at a CTA barrier
  const std::pair<int, VKern> vis[] = {{0, &k_vis<0>}, {1, &k_vis<1>}, {4, &k_vis<4>}};
  for (const auto& [s_cnt, k] : vis) {
    const std::string variant = "vis.s" + std::to_string(s_cnt);
    if (!mb_selected(args, variant)) continue;
    std::vector<double> k0, k1;
    for (int l = 0; l < args.launches; ++l) {
      k<<<1, 128>>>(d_out, args.reps, static_cast<std::uint32_t>(l));
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(4 * static_cast<std::size_t>(args.reps));
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int r = 1; r < args.reps; ++r) {
        k0.push_back(static_cast<double>(static_cast<std::int64_t>(h[4 * r + 1] - h[4 * r])));
        k1.push_back(static_cast<double>(static_cast<std::int64_t>(h[4 * r + 2] - h[4 * r])));
      }
    }
    w.row(variant + ".arriver", s_cnt, "\"stores\":" + std::to_string(s_cnt), k0);
    w.row(variant + ".warp1", s_cnt, "\"stores\":" + std::to_string(s_cnt), k1);
  }
  return w.failures ? 1 : 0;
}
