// register_operands probe (protocol.json): k_thrx2 harness verbatim
// (16 warps x 8 chains, K in {16,32,64}, one target op per chain per K);
// cells = (class, source form, perturb) x {16,32,64}.
#include "../mb.cuh"
#include <string>
#include <vector>

// op signatures: op(x, y, z, k1, k2) — unused operands ignored per form.
#define OPDEF(NAME, BODY)                                                 \
  struct NAME {                                                           \
    __device__ __forceinline__ static void op(                            \
        unsigned long long& x, unsigned long long& y,                     \
        unsigned long long& z, unsigned long long k1,                     \
        unsigned long long k2) { BODY }                                   \
  };

// FFMA (fma.rn.f32, scalar; low halves of the u64 chain regs)
#define U32X unsigned int& x_ = *reinterpret_cast<std::uint32_t*>(&x), \
               &y_ = *reinterpret_cast<std::uint32_t*>(&y), \
               &z_ = *reinterpret_cast<std::uint32_t*>(&z); \
  const unsigned int k1_ = (std::uint32_t)k1, k2_ = (std::uint32_t)k2
#define F32OPS(N, A, B, C)                                                \
  OPDEF(N, {                                                              \
    U32X;                                                                 \
    asm volatile("{\n\t.reg .f32 fa,fb,fc;\n\t"                       \
                 "mov.b32 fa, " A ";\n\tmov.b32 fb, " B                 \
                 ";\n\tmov.b32 fc, " C ";\n\t"                        \
                 "fma.rn.f32 fa, fa, fb, fc;\n\tmov.b32 %0, fa;\n\t}" \
                 : "+r"(x_) : "r"(y_), "r"(z_), "r"(k1_), "r"(k2_));        })
F32OPS(FFMA_xxx, "%0", "%0", "%0")
F32OPS(FFMA_xkk, "%0", "%3", "%4")
F32OPS(FFMA_xyk, "%0", "%1", "%3")
F32OPS(FFMA_xyz, "%0", "%1", "%2")
// FFMA2 (fma.rn.f32x2, 64-bit pairs)
#if __CUDA_ARCH__ >= 1000
#define X2ASM(A, B, C) "fma.rn.f32x2 %0, " A ", " B ", " C ";"
#define A2ASM(A, B) "add.rn.f32x2 %0, " A ", " B ";"
#else
#define X2ASM(A, B, C) ""
#define A2ASM(A, B) ""
#endif
#define X2OPS(N, A, B, C)                                                 \
  OPDEF(N, {                                                              \
    asm volatile(X2ASM(A, B, C)                                           \
                 : "+l"(x) : "l"(y), "l"(z), "l"(k1), "l"(k2)); })

X2OPS(FFMA2_xxx, "%0", "%0", "%0")
X2OPS(FFMA2_xkk, "%0", "%3", "%4")
X2OPS(FFMA2_xyk, "%0", "%1", "%3")
X2OPS(FFMA2_xyz, "%0", "%1", "%2")
// FADD2 (add.rn.f32x2, two sources)
#define A2OPS(N, A, B)                                                    \
  OPDEF(N, {                                                              \
    asm volatile(A2ASM(A, B)                                              \
                 : "+l"(x) : "l"(y), "l"(z), "l"(k1), "l"(k2)); })
A2OPS(FADD2_xx, "%0", "%0")
A2OPS(FADD2_xk, "%0", "%3")
A2OPS(FADD2_xy, "%0", "%1")
// IMAD (mad.lo.u32 on low halves)
#define IMOPS(N, A, B, C)                                                 \
  OPDEF(N, {                                                              \
    U32X;                                                                 \
    asm volatile("mad.lo.u32 %0, " A ", " B ", " C ";"                    \
                 : "+r"(x_) : "r"(y_), "r"(z_), "r"(k1_), "r"(k2_));        })
IMOPS(IMAD_xxx, "%0", "%0", "%0")
IMOPS(IMAD_xkk, "%0", "%3", "%4")
IMOPS(IMAD_xyk, "%0", "%1", "%3")
IMOPS(IMAD_xyz, "%0", "%1", "%2")
// FADD (add.rn.f32, two scalar sources)
#define FAOPS(N, A, B)                                                    \
  OPDEF(N, {                                                              \
    U32X;                                                                 \
    asm volatile("{\n\t.reg .f32 fa,fb;\n\t"                          \
                 "mov.b32 fa, " A ";\n\tmov.b32 fb, " B                 \
                 ";\n\tadd.rn.f32 fa, fa, fb;\n\tmov.b32 %0, fa;\n\t}"                  : "+r"(x_) : "r"(y_), "r"(z_), "r"(k1_), "r"(k2_));        })
FAOPS(FADD_xx, "%0", "%0")
FAOPS(FADD_xk, "%0", "%3")
FAOPS(FADD_xy, "%0", "%1")
// k_thrx2 verbatim (16 warps x 8 chains); PERT adds one live register
// pair before the chains to perturb bank assignment.
template <class O, int K, int PERT>
__global__ void __launch_bounds__(512, 1)
k_thr(std::uint64_t* out, std::uint32_t* sink, int reps, std::uint32_t s) {
  __shared__ __align__(16) std::uint32_t pinb[512 * 16];
  const std::uint32_t base = mb_smem_u32(pinb) + threadIdx.x * 4u;
  constexpr std::uint32_t kStride = 512u * 4u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + j * kStride), "r"(s + threadIdx.x + j));
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(base + (8 + j) * kStride), "r"(s * 3u + j));
  }
  unsigned long long pert = PERT ? (s | 1ull) : 0;
  unsigned long long x[8], y[8], z[8];
  const unsigned long long k1 = s ^ 0x3f800000u, k2 = 0x3fb504f3ull;
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
        O::op(x[j], y[j], z[j], k1, k2);
      }
    }
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  std::uint32_t f = (std::uint32_t)pert;
#pragma unroll
  for (int j = 0; j < 8; ++j) f ^= (std::uint32_t)(x[j] ^ y[j] ^ z[j]);
  sink[threadIdx.x] = f;
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "regops");
  std::uint64_t* d_out;
  std::uint32_t* d_sink;
  cudaMalloc(&d_out, args.reps * 8);
  cudaMalloc(&d_sink, 512 * 4);
  struct Cell { const char* op; const char* name; };
  const Cell cells[] = {
      {"FFMA_xxx", "FFMA.xxx"}, {"FFMA_xkk", "FFMA.xkk"},
      {"FFMA_xyk", "FFMA.xyk"}, {"FFMA_xyz", "FFMA.xyz"},
      {"FFMA2_xxx", "FFMA2.xxx"}, {"FFMA2_xkk", "FFMA2.xkk"},
      {"FFMA2_xyk", "FFMA2.xyk"}, {"FFMA2_xyz", "FFMA2.xyz"},
      {"FADD2_xx", "FADD2.xx"}, {"FADD2_xk", "FADD2.xk"},
      {"FADD2_xy", "FADD2.xy"},
      {"IMAD_xxx", "IMAD.xxx"}, {"IMAD_xkk", "IMAD.xkk"},
      {"IMAD_xyk", "IMAD.xyk"}, {"IMAD_xyz", "IMAD.xyz"},
      {"FADD_xx", "FADD.xx"}, {"FADD_xk", "FADD.xk"},
      {"FADD_xy", "FADD.xy"},
  };
  for (const auto& c : cells) {
    for (int pert : {0, 1}) {
      const std::string nm =
          std::string("thr.") + c.name + (pert ? ".p1" : ".p0");
      if (args.only.size() && nm.find(args.only) == std::string::npos)
        continue;
      for (int k : {16, 32, 64}) {
        std::vector<double> sm;
        for (int l = 0; l < args.launches; ++l) {
          const std::uint32_t seed = 0x9e3779b9u + l * 97;
          void (*fn)(std::uint64_t*, std::uint32_t*, int,
                     std::uint32_t) = nullptr;
#define PICK(O, P) (k == 16 ? &k_thr<O, 16, P> \
                    : k == 32 ? &k_thr<O, 32, P> : &k_thr<O, 64, P>)
#define PICKP(O) (pert ? PICK(O, 1) : PICK(O, 0))
          if (!std::strcmp(c.op, "FFMA_xxx")) fn = PICKP(FFMA_xxx);
          else if (!std::strcmp(c.op, "FFMA_xkk")) fn = PICKP(FFMA_xkk);
          else if (!std::strcmp(c.op, "FFMA_xyk")) fn = PICKP(FFMA_xyk);
          else if (!std::strcmp(c.op, "FFMA_xyz")) fn = PICKP(FFMA_xyz);
          else if (!std::strcmp(c.op, "FFMA2_xxx")) fn = PICKP(FFMA2_xxx);
          else if (!std::strcmp(c.op, "FFMA2_xkk")) fn = PICKP(FFMA2_xkk);
          else if (!std::strcmp(c.op, "FFMA2_xyk")) fn = PICKP(FFMA2_xyk);
          else if (!std::strcmp(c.op, "FFMA2_xyz")) fn = PICKP(FFMA2_xyz);
          else if (!std::strcmp(c.op, "FADD2_xx")) fn = PICKP(FADD2_xx);
          else if (!std::strcmp(c.op, "FADD2_xk")) fn = PICKP(FADD2_xk);
          else if (!std::strcmp(c.op, "FADD2_xy")) fn = PICKP(FADD2_xy);
          else if (!std::strcmp(c.op, "IMAD_xxx")) fn = PICKP(IMAD_xxx);
          else if (!std::strcmp(c.op, "IMAD_xkk")) fn = PICKP(IMAD_xkk);
          else if (!std::strcmp(c.op, "IMAD_xyk")) fn = PICKP(IMAD_xyk);
          else if (!std::strcmp(c.op, "IMAD_xyz")) fn = PICKP(IMAD_xyz);
          else if (!std::strcmp(c.op, "FADD_xx")) fn = PICKP(FADD_xx);
          else if (!std::strcmp(c.op, "FADD_xk")) fn = PICKP(FADD_xk);
          else fn = PICKP(FADD_xy);
          fn<<<1, 512>>>(d_out, d_sink, args.reps, seed);
          const cudaError_t e1 = cudaGetLastError();
          const cudaError_t e2 = cudaDeviceSynchronize();
          if (e1 != cudaSuccess || e2 != cudaSuccess) {
            cudaError_t e = e1 != cudaSuccess ? e1 : e2;
            std::fprintf(stdout, "FAIL %s/%d launch %d: %s\n",
                         nm.c_str(), k, l, cudaGetErrorName(e));
            w.row(nm, k, "\"status\":\"unrealised:" +
                  std::string(cudaGetErrorName(e)) + "\"", {});
            cudaGetLastError();
            break;
          }
          std::vector<std::uint64_t> h(args.reps);
          cudaMemcpy(h.data(), d_out, h.size() * 8,
                     cudaMemcpyDeviceToHost);
          if (l == 0) continue;
          for (int r = 1; r < args.reps; ++r) sm.push_back((double)h[r]);
        }
        if (sm.size())
          w.row(nm, k, "\"cell\":\"" + nm + "\"", sm);
      }
    }
  }
  return 0;
}
