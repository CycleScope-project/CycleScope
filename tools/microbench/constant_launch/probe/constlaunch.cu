// memory.constant_cache bank-0 first-access probe (constant_launch
// protocol v1): t0 = clock as the kernel's first instruction, a dependent
// register-indexed chain of U loads idx = table[idx] (first traversal),
// t1, the same chain again, t2; lane 0 of warp 0 stores t0/t1/t2/idx and
// the host prints one LAUNCH line per launch.
//
//   BANK 0: table = by-value kernel parameter struct of 4096 B
//           (uint32 next[1024] + output pointer is the only other param).
//   BANK 3: table = 4096 B __constant__ array written by
//           cudaMemcpyToSymbol before the first launch.
//
// Variants cl.b<BANK>.s<S>.u<U>, BANK in {0,3}, S in {64,256},
// U in {4,8,16} (U*S <= 4096 B), W = 1 warp, grid 1.
#include "../mb.cuh"

#include <string>
#include <vector>

struct ClParams {
  std::uint32_t next[1024];
};

__constant__ std::uint32_t cl_table3[1024];

// Window pinning (crossings constant_cache uses the same devices): each
// chain load is one volatile-asm ld.param / ld.const carrying a real
// register dependency (byte offset off -> LDC result -> off), the chain
// start comes from a volatile shared slot BEFORE t0 so ptxas cannot
// prove it uniform (uniform => ULDC UR path, off the gate), and the
// whole timed body runs only on the threadIdx.x == 0 path — plain C++
// loads or empty asm barriers let ptxas hoist the CS2Rs out of the
// window (verified in the local build).
#define CL_LD0(off, base)                                                  \
  asm volatile(                                                            \
      "{\n\t.reg .u64 t;\n\t.reg .u32 v;\n\t"                              \
      "add.u64 t, %1, %2;\n\t"                                             \
      "ld.param.u32 v, [t];\n\t"                                           \
      "mul.wide.u32 %0, v, 4;\n\t}"                                        \
      : "=l"(off) : "l"(base), "l"(off) : "memory")
#define CL_LD3(off)                                                        \
  asm volatile(                                                            \
      "{\n\t.reg .u64 t;\n\t.reg .u32 v;\n\t"                              \
      "mov.u64 t, cl_table3;\n\t"                                          \
      "add.u64 t, t, %1;\n\t"                                              \
      "ld.const.u32 v, [t];\n\t"                                           \
      "mul.wide.u32 %0, v, 4;\n\t}"                                        \
      : "=l"(off) : "l"(off) : "memory")

__shared__ std::uint32_t cl_slot[32];

template <int U, int S>
__global__ void __launch_bounds__(32, 1) k_cl0(const ClParams p,
                                               std::uint64_t* out) {
  if (threadIdx.x != 0) return;
  std::uint64_t off = mb_pin(0u, &cl_slot[threadIdx.x]);
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int k = 0; k < U; ++k) CL_LD0(off, p.next);
  const std::uint64_t t1 = mb_clock();
#pragma unroll
  for (int k = 0; k < U; ++k) CL_LD0(off, p.next);
  const std::uint64_t t2 = mb_clock();
  out[0] = t0; out[1] = t1; out[2] = t2; out[3] = off >> 2;
}

template <int U, int S>
__global__ void __launch_bounds__(32, 1) k_cl3(std::uint64_t* out) {
  if (threadIdx.x != 0) return;
  std::uint64_t off = mb_pin(0u, &cl_slot[threadIdx.x]);
  const std::uint64_t t0 = mb_clock();
#pragma unroll
  for (int k = 0; k < U; ++k) CL_LD3(off);
  const std::uint64_t t1 = mb_clock();
#pragma unroll
  for (int k = 0; k < U; ++k) CL_LD3(off);
  const std::uint64_t t2 = mb_clock();
  out[0] = t0; out[1] = t1; out[2] = t2; out[3] = off >> 2;
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.reps != 1) {
    std::fprintf(stderr, "counter-only probe requires --reps 1\n");
    return 2;
  }
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, 4 * sizeof(std::uint64_t)));

  // chain: load k reads byte offset k*S = element k*S/4;
  // next[k*S/4] = (k+1)*S/4 for k < U-1, 0 for k = U-1
  const int us[] = {4, 8, 16};
  const int ss[] = {64, 256};
  auto run = [&](int bank, int s, int u, auto kfn, const char* tag) {
    char name[64];
    std::snprintf(name, sizeof(name), "cl.b%d.s%d.u%d", bank, s, u);
    if (!mb_selected(args, name)) return;
    ClParams h{};
    for (int k = 0; k < u; ++k)
      h.next[k * s / 4] = (k + 1 < u ? (k + 1) * s : 0) / 4;
    // the chain depends on (s, u): rewrite the bank-3 table per variant
    if (bank == 3)
      MB_CHECK(cudaMemcpyToSymbol(cl_table3, h.next, sizeof(h.next)));
    for (int l = 0; l < args.launches; ++l) {
      kfn(h, d_out);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      MB_CHECK(cudaGetLastError());
      std::uint64_t o[4];
      MB_CHECK(cudaMemcpy(o, d_out, sizeof(o), cudaMemcpyDeviceToHost));
      std::printf("LAUNCH %s variant=%s launch=%d t0=%llu t1=%llu t2=%llu"
                  " idx=%llu\n", tag, name, l,
                  (unsigned long long)o[0], (unsigned long long)o[1],
                  (unsigned long long)o[2], (unsigned long long)o[3]);
    }
  };

  for (int s : ss) {
    for (int u : us) {
      if (u * s > 4096) continue;
#define DISP(KNAME) \
      (u == 4  ? (s == 64 ? KNAME<4, 64>  : KNAME<4, 256>) \
       : u == 8  ? (s == 64 ? KNAME<8, 64>  : KNAME<8, 256>) \
       :         (s == 64 ? KNAME<16, 64> : KNAME<16, 256>))
      {
        auto f = DISP(k_cl0);
        run(0, s, u, [&](const ClParams& p, std::uint64_t* o) {
          f<<<1, 32>>>(p, o);
        }, "k_cl");
        auto g = DISP(k_cl3);
        run(3, s, u, [&](const ClParams&, std::uint64_t* o) {
          g<<<1, 32>>>(o);
        }, "k_cl");
      }
#undef DISP
    }
  }
  return 0;
}
