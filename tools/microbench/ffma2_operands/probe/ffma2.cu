// ffma2_operands probe (protocol.json): the v6 k_thrx2 harness
// unchanged, only the op differs per cell.
#include "../mb.cuh"
#include <string>
#include <vector>

// F3: fma.rn.f32x2 d,a,b,c — three distinct chain pairs (v6 cell)
struct F3 {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long k) {
    asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                 : "+l"(x) : "l"(y), "l"(k));
  }
};
// F2a: fma(x, x, y) — two distinct pairs, addend == multiplicand
struct F2a {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long k) {
    asm volatile("fma.rn.f32x2 %0, %0, %0, %1;"
                 : "+l"(x) : "l"(y));
  }
};
// F2b: fma(x, y, k) — k one loop-invariant pair shared by all chains
struct F2b {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long k) {
    asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                 : "+l"(x) : "l"(y), "l"(k));
  }
};
// F1: fma(x, k1, k2) — only x changes across steps; k1/k2 invariant
struct F1 {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long k1, unsigned long long k2) {
    asm volatile("fma.rn.f32x2 %0, %0, %1, %2;"
                 : "+l"(x) : "l"(k1), "l"(k2));
  }
};
// A2: add.rn.f32x2 x, x, y — FADD2 control (expect ~1.99)
struct A2 {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long k) {
    asm volatile("add.rn.f32x2 %0, %0, %1;"
                 : "+l"(x) : "l"(y));
  }
};
// S3: scalar fma.rn.f32 with three distinct registers (low halves)
struct S3 {
  __device__ __forceinline__ static void op(
      unsigned long long& x, unsigned long long& y,
      unsigned long long z) {
    // scalar fma.rn.f32 with three distinct registers: x = x*y + z
    asm volatile("{\n\t.reg .f32 fx,fy,fz;\n\t"
                 "mov.b32 fx, %0;\n\tmov.b32 fy, %1;\n\tmov.b32 fz, %2;\n\t"
                 "fma.rn.f32 fx, fx, fy, fz;\n\tmov.b32 %0, fx;\n\t}"
                 : "+r"(*reinterpret_cast<std::uint32_t*>(&x))
                 : "r"(*(std::uint32_t*)&y), "r"(*(std::uint32_t*)&z));
  }
};

// k_thrx2 verbatim from alu7 (16 warps x 8 chains, one op per chain per K)
template <class O, int K>
__global__ void __launch_bounds__(512, 1)
k_thrx2(std::uint64_t* out, std::uint32_t* sink, int reps,
        std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  unsigned long long x[8], y[8];
  const unsigned long long k = s ^ 0x3f800000u;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&x[j])) : "r"(base + j * kStride));
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&y[j])) : "r"(base + (8 + j) * kStride));
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        O::op(x[j], y[j], k);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= (std::uint32_t)(x[j] ^ y[j]);
  sink[threadIdx.x] = f;
}

// F3 kernel: three distinct chain registers a,b,c (x,y,z arrays)
template <int K>
__global__ void __launch_bounds__(512, 1)
k_thr_f3(std::uint64_t* out, std::uint32_t* sink, int reps,
         std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  unsigned long long x[8], y[8], z[8];
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&x[j])) : "r"(base + j * kStride));
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&y[j])) : "r"(base + (8 + j) * kStride));
      z[j] = x[j] ^ (0x9e3779b9ull * (j + 1));
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
                     : "+l"(x[j]) : "l"(y[j]), "l"(z[j]), "l"(x[j]));
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= (std::uint32_t)(x[j] ^ y[j] ^ z[j]);
  sink[threadIdx.x] = f;
}

// F1 needs two invariant pairs: same shape with an extra param
template <int K>
__global__ void __launch_bounds__(512, 1)
k_thr_f1(std::uint64_t* out, std::uint32_t* sink, int reps,
         std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  unsigned long long x[8], y[8];
  const unsigned long long k1 = s ^ 0x3f800000u,
                         k2 = 0x3fb504f3ull;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&x[j])) : "r"(base + j * kStride));
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(*reinterpret_cast<std::uint32_t*>(&y[j])) : "r"(base + (8 + j) * kStride));
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        F1::op(x[j], y[j], k1, k2);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= (std::uint32_t)(x[j] ^ y[j]);
  sink[threadIdx.x] = f;
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "ffma2");
  std::uint64_t* d_out;
  std::uint32_t* d_sink;
  cudaMalloc(&d_out, args.reps * 8);
  cudaMalloc(&d_sink, 512 * 4);
  const char* names[] = {"thr.F3", "thr.F2a", "thr.F2b", "thr.F1",
                         "thr.A2", "thr.S3"};
  for (const char* n : names) {
    if (args.only.size() && std::string(n).find(args.only) ==
                            std::string::npos)
      continue;
    for (int k : {16, 32, 64}) {
      std::vector<double> s;
      for (int l = 0; l < args.launches; ++l) {
        const std::uint32_t seed = 0x9e3779b9u + l * 97;
        void (*fn)(std::uint64_t*, std::uint32_t*, int,
                   std::uint32_t) = nullptr;
        std::string v(n);
#define PK(O) (k == 16 ? &k_thrx2<O, 16> \
               : k == 32 ? &k_thrx2<O, 32> : &k_thrx2<O, 64>)
        if (v == "thr.F2a") fn = PK(F2a);
        else if (v == "thr.F2b") fn = PK(F2b);
        else if (v == "thr.A2") fn = PK(A2);
        else if (v == "thr.F3")
          fn = (k == 16 ? &k_thr_f3<16>
                : k == 32 ? &k_thr_f3<32> : &k_thr_f3<64>);
        else if (v == "thr.S3") fn = PK(S3);
        else fn = (k == 16 ? (void (*)(std::uint64_t*, std::uint32_t*,
                                      int, std::uint32_t))&k_thr_f1<16>
                   : k == 32 ? (void (*)(std::uint64_t*, std::uint32_t*,
                                         int, std::uint32_t))&k_thr_f1<32>
                             : (void (*)(std::uint64_t*, std::uint32_t*,
                                         int, std::uint32_t))&k_thr_f1<64>);
        fn<<<1, 512>>>(d_out, d_sink, args.reps, seed);
        const cudaError_t e1 = cudaGetLastError();
        const cudaError_t e2 = cudaDeviceSynchronize();
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
          cudaError_t e = e1 != cudaSuccess ? e1 : e2;
          std::fprintf(stdout, "FAIL %s/%d launch %d: %s\n", n, k, l,
                       cudaGetErrorName(e));
          w.row(std::string(n) + "_k" + std::to_string(k), k,
                "\"status\":\"unrealised:" +
                std::string(cudaGetErrorName(e)) + "\"", {});
          cudaGetLastError();
          break;
        }
        std::vector<std::uint64_t> h(args.reps);
        cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost);
        if (l == 0) continue;
        for (int r = 1; r < args.reps; ++r) s.push_back((double)h[r]);
      }
      if (s.size())
        w.row(std::string(n) + "_k" + std::to_string(k), k,
              "\"cell\":\"" + std::string(n) + "\"", s);
    }
  }
  return 0;
}
