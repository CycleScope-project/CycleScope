// tools/microbench/lsu_accept/probe/lsu_accept.cu -- protocol v1 (frozen
// tools/microbench/lsu_accept/protocol.json).  One CTA of 32*W threads
// (W in {1,2,4,8}; warp w sits on SMSP w % 4), bar.sync before every rep,
// lane 0 of each warp records t_s / t_w around the window.  Window = D
// back-to-back global requests on D distinct 64-bit pointer registers
// (per-lane coalesced: lane * bytes; each request its own 128 B-aligned
// region, distinct per warp and per k) + one add.s64 on the LAST
// request's pointer register (WAR on its read barrier) + t_w; after
// t_w: cp.async.wait_group 0 / consume + bar.sync.  D = 0 emits only
// the WAR on an unrelated register (overhead cell).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

// variants
enum : int { V_LDGSTS16, V_LDGSTS8, V_LDGSTS4, V_LDG16, V_LDG4,
             V_STG16, V_MIX, V_LDS };

// one request on pointer p (global) / u (shared), destination regs r
template <int V>
__device__ __forceinline__ void request(std::uint64_t p, std::uint32_t u,
                                        unsigned &r0, unsigned &r1,
                                        unsigned &r2, unsigned &r3,
                                        const unsigned s0, const unsigned s1,
                                        const unsigned s2, const unsigned s3) {
  if constexpr (V == V_LDGSTS16 || V == V_MIX) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(u), "l"(p));
    if constexpr (V == V_MIX)
      asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(u));
  } else if constexpr (V == V_LDGSTS8) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                 :: "r"(u), "l"(p));
  } else if constexpr (V == V_LDGSTS4) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                 :: "r"(u), "l"(p));
  } else if constexpr (V == V_LDG16) {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(p));
  } else if constexpr (V == V_LDG4) {
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(r0) : "l"(p));
  } else if constexpr (V == V_STG16) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(s0), "r"(s1), "r"(s2), "r"(s3));
  } else if constexpr (V == V_LDS) {
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(u));
  }
}

template <int V, int I, int N>
struct ReqBlock {
  static __device__ __forceinline__ void go(std::uint64_t *p,
                                            std::uint32_t *u,
                                            unsigned *acc,
                                            const unsigned *src) {
    if constexpr (N > 0) {
      request<V>(p[I], u[I], acc[I * 4 + 0], acc[I * 4 + 1],
                 acc[I * 4 + 2], acc[I * 4 + 3],
                 src[0], src[1], src[2], src[3]);
      ReqBlock<V, I + 1, N - 1>::go(p, u, acc, src);
    }
  }
};

extern __shared__ __align__(16) char smem_dyn[];

template <int V, int D>
__device__ __forceinline__ void window(std::uint64_t *p, std::uint32_t *u,
                                       unsigned *acc, const unsigned *src) {
  ReqBlock<V, 0, D>::go(p, u, acc, src);
  // WAR on the last request's pointer register (D = 0: unrelated reg);
  // for the lds control the "pointer" is the 32-bit smem address
  if constexpr (V == V_LDS) {
    // WAR on the last request's smem address register; consuming the
    // last load's result keeps the request inside the window
    std::uint32_t last = u[D > 0 ? D - 1 : 0];
    asm volatile("add.u32 %0, %0, %1;" : "+r"(last)
                 : "r"(D > 0 ? acc[(D - 1) * 4] : 0u));
  } else {
    std::uint64_t last = p[D > 0 ? D - 1 : 0];
    asm volatile("add.s64 %0, %0, 512;" : "+l"(last));
  }
}

template <int V, int D>
__global__ void k_cell(std::uint64_t *out, unsigned *sink,
                       const char *base, const unsigned *uzeros,
                       const std::uint64_t *offs, int reps, int warps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned uz = uzeros[threadIdx.x];
  // pointer registers: per-lane opaque via a loaded zero (contention_v3
  // trick); region stride 32 lanes * 16 B per request; k-major so the
  // layout is W-independent
  std::uint64_t p[16];
  std::uint32_t u[16];
  unsigned acc[64];
  const unsigned src[4] = {0x11223344u + uz, 0x55667788u + uz,
                           0x99aabbccu + uz, 0xddeeff00u + uz};
  const std::uint32_t sm0 = mb_smem_u32(smem_dyn);
  for (int k = 0; k < 16; ++k) {
    // pointer offsets loaded from a global table: opaque, distinct
    // per (k, lane) values ptxas cannot refold into [Rbase+imm]
    p[k] = reinterpret_cast<std::uint64_t>(base) +
           offs[k * 256 + warp * 32 + lane] + uz;
    // 16 B-aligned smem region per (warp, k); the opaque per-k add
    // keeps the address in its own register
    u[k] = sm0;
    asm volatile("add.u32 %0, %0, %1;" : "+r"(u[k]) : "r"(
        uz + static_cast<std::uint32_t>((warp * 16 + k) * 512 +
                                        lane * 16)));
  }
  for (int i = 0; i < 64; ++i) acc[i] = uz + i;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t_s = mb_clock();
    window<V, D>(p, u, acc, src);
    // v2: lds/mix consume every LDS.128 result (XOR-fold 4 words into
    // one register) before t_w so ptxas keeps the loads in the window
    if constexpr (V == V_LDS || V == V_MIX) {
      for (int i = 0; i < 64; i += 4)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(acc[i])
                     : "r"(acc[i + 1]), "r"(acc[i + 2]));
    }
    // invisible use of every request's destination registers: emits
    // no instructions but forbids the SASS scheduler from sinking a
    // request below the closing clock read
    for (int i = 0; i < 4 * 16; ++i)
      asm volatile("" ::"r"(acc[i]));
    for (int k = 0; k < 16; ++k)
      asm volatile("" ::"l"(p[k]), "r"(u[k]));
    // memory barrier: requests cannot sink below the closing clock
    asm volatile("" ::: "memory");
    const std::uint64_t t_w = mb_clock();
    if constexpr (V == V_LDGSTS16 || V == V_LDGSTS8 || V == V_LDGSTS4 ||
                  V == V_MIX)
      asm volatile("cp.async.wait_group 0;");
    if (lane == 0) {
      out[(warp * reps + rep) * 2 + 0] = t_s;
      out[(warp * reps + rep) * 2 + 1] = t_w;
    }
  }
  if (threadIdx.x == 0) {
    unsigned s = 0;
    for (int i = 0; i < 64; ++i) s += acc[i];
    for (int k = 0; k < 16; ++k)
      s += static_cast<unsigned>(p[k] ^ u[k]);
    sink[0] = s;
  }
}

using CKern = void (*)(std::uint64_t *, unsigned *, const char *,
                       const unsigned *, const std::uint64_t *, int,
                       int);

#define KCELL(V, D) &k_cell<V, D>
#define KROW(V) \
    KCELL(V, 0), KCELL(V, 1), KCELL(V, 2), KCELL(V, 4), KCELL(V, 8), \
        KCELL(V, 16)
static const struct {
  const char *variant;
  CKern kern[6];
} C_CELLS[] = {
    {"ldgsts16", KROW(V_LDGSTS16)}, {"ldgsts8", KROW(V_LDGSTS8)},
    {"ldgsts4", KROW(V_LDGSTS4)},   {"ldg16", KROW(V_LDG16)},
    {"ldg4", KROW(V_LDG4)},         {"stg16", KROW(V_STG16)},
    {"mix", KROW(V_MIX)},           {"lds", KROW(V_LDS)},
};
static const int C_D[6] = {0, 1, 2, 4, 8, 16};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  unsigned *d_sink = nullptr, *d_uzeros = nullptr;
  char *d_base = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 8 * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned) * 8));
  MB_CHECK(cudaMalloc(&d_uzeros, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMemset(d_uzeros, 0, sizeof(unsigned) * 1024));
  MB_CHECK(cudaMalloc(&d_base, 1u << 20));  // 16 req * 8 w * 512 B < 1 MB
  std::uint64_t *d_offs = nullptr;
  MB_CHECK(cudaMalloc(&d_offs, sizeof(std::uint64_t) * 16 * 256));
  {
    std::vector<std::uint64_t> ho(16 * 256);
    for (int k = 0; k < 16; ++k)
      for (int i = 0; i < 256; ++i)
        ho[k * 256 + i] = static_cast<std::uint64_t>(k * 256 + i) * 16;
    MB_CHECK(cudaMemcpy(d_offs, ho.data(), ho.size() * 8,
                        cudaMemcpyHostToDevice));
  }
  const int smem_bytes = 8 * 16 * 512;      // 64 KiB: (warp,k) region
  for (const auto &c : C_CELLS) {
    for (int di = 0; di < 6; ++di)
      MB_CHECK(cudaFuncSetAttribute(
          reinterpret_cast<const void *>(c.kern[di]),
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    for (int wg : {1, 2, 4, 8}) {
      const std::string variant = std::string(c.variant) + ".w" +
                                  std::to_string(wg);
      for (int di = 0; di < 6; ++di) {
        if (!mb_selected(args, variant)) continue;
        std::vector<double> samples;
        for (int l = 0; l < args.launches; ++l) {
          c.kern[di]<<<1, 32 * wg, smem_bytes>>>(
              d_out, d_sink, d_base, d_uzeros, d_offs, args.reps, wg);
          const cudaError_t e = cudaGetLastError();
          if (e != cudaSuccess) {
            std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                         cudaGetErrorString(e));
            break;
          }
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(wg * args.reps * 2);
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                              cudaMemcpyDeviceToHost));
          for (int r = 1; r < args.reps; ++r)
            for (int wp = 0; wp < wg; ++wp) {
              const std::uint64_t ts =
                  h[(wp * args.reps + r) * 2 + 0];
              const std::uint64_t tw =
                  h[(wp * args.reps + r) * 2 + 1];
              samples.push_back(static_cast<double>(tw - ts));
            }
        }
        const std::string params =
            "\"variant\":\"" + std::string(c.variant) + "\",\"w\":" +
            std::to_string(wg) + ",\"d\":" + std::to_string(C_D[di]) +
            ",\"window\":\"warp\"";
        w.row(variant, C_D[di], params, samples);
      }
    }
  }
  return 0;
}
