// memory.constant_launch protocol v3 probe: as v2 (16 dependent
// register-indexed constant loads per traversal, two traversals,
// CS2R SR_CLOCKLO before the first load and after every load,
// t_0 .. t_32) with an added uniform path.
//
//   vector cells cl.b<BANK>.s256: v2 kernels verbatim (control).
//   uniform cells cl.u.b<BANK>.s256: the same chain with the index in
//   uniform registers -- plain C++ indexing where the index is a
//   warp-uniform value (kernel-parameter seed), so ptxas emits
//   ULDC UR, c[0x<B>][UR(+imm)] on sm_90a and LDCU on sm_103a.  Every
//   thread of the warp runs the chain; NO threadIdx-dependent branch
//   anywhere (a tid branch breaks the clock interleave: ptxas hoists
//   the CS2Rs out of the windows).  Table holds element indices:
//   next[k*S/4] = (k+1)*S/4 for k < 15 and 0 for k = 15.
#include "../mb.cuh"

#include <string>
#include <vector>

struct ClParams {
  std::uint32_t next[1024];
};

__constant__ std::uint32_t cl_table3[1024];

// ---- vector path (v2, verbatim) ----
#define CL_LD0(a, base)                                                    \
  asm volatile(                                                            \
      "{\n\t.reg .u32 v;\n\t"                                              \
      "ld.param.u32 v, [%1];\n\t"                                          \
      "mad.wide.u32 %0, v, 4, %2;\n\t}"                                    \
      : "=l"(a) : "l"(a), "l"(base) : "memory")
#define CL_LD3(a, base)                                                    \
  asm volatile(                                                            \
      "{\n\t.reg .u32 v;\n\t"                                              \
      "ld.const.u32 v, [%1];\n\t"                                          \
      "mad.wide.u32 %0, v, 4, %2;\n\t}"                                    \
      : "=l"(a) : "l"(a), "l"(base) : "memory")

__shared__ std::uint32_t cl_slot[32];

template <int S>
__global__ void __launch_bounds__(32, 1) k_cl0(const ClParams p,
                                               std::uint64_t* out) {
  if (threadIdx.x != 0) return;
  std::uint64_t base;
  asm("mov.u64 %0, %1;" : "=l"(base) : "l"(p.next));
  std::uint64_t a = mb_pin((std::uint32_t)base, &cl_slot[threadIdx.x]);
  std::uint64_t t[33];
  t[0] = mb_clock();
#pragma unroll
  for (int k = 0; k < 16; ++k) { CL_LD0(a, base); t[k + 1] = mb_clock(); }
#pragma unroll
  for (int k = 0; k < 16; ++k) { CL_LD0(a, base); t[17 + k] = mb_clock(); }
  for (int i = 0; i < 33; ++i) out[i] = t[i];
  out[33] = a - base;
}

template <int S>
__global__ void __launch_bounds__(32, 1) k_cl3(std::uint64_t* out) {
  if (threadIdx.x != 0) return;
  std::uint64_t base;
  asm("mov.u64 %0, cl_table3;" : "=l"(base));
  std::uint64_t a = mb_pin((std::uint32_t)base, &cl_slot[threadIdx.x]);
  std::uint64_t t[33];
  t[0] = mb_clock();
#pragma unroll
  for (int k = 0; k < 16; ++k) { CL_LD3(a, base); t[k + 1] = mb_clock(); }
#pragma unroll
  for (int k = 0; k < 16; ++k) { CL_LD3(a, base); t[17 + k] = mb_clock(); }
  for (int i = 0; i < 33; ++i) out[i] = t[i];
  out[33] = a - base;
}

// ---- uniform path ----
// element-index chain in C++; the index lives in uniform registers so
// ptxas emits ULDC UR, c[0x<B>][UR] (sm_90a) / LDCU UR, c[0x<B>][UR]
// (sm_103a).  EVERY thread of the warp runs the chain; NO threadIdx-
// dependent branch and NO asm operand may touch the chain register
// (either one forces the whole chain onto the vector path).
// Chain start: x = TBL[(u32)out & 1023] -- warp-uniform, not a
// compile-time constant; the indexed seed load lands before t0
// because the out descriptor is materialized early, keeping the
// measured window clean of anything but the chain loads and clocks.
#define UCLK(t) asm volatile("mov.u64 %0, %clock64;" : "=l"(t) :: "memory")

template <int S>
__global__ void __launch_bounds__(32, 1) k_uc0(const ClParams p,
                                               std::uint64_t* out) {
  std::uint32_t x = p.next[(std::uint32_t)(std::uintptr_t)out & 1023];
  std::uint64_t t[33];
  asm volatile("" :: "l"(out) : "memory");
  UCLK(t[0]);
#pragma unroll
  for (int k = 0; k < 16; ++k) { x = p.next[x]; UCLK(t[k + 1]); }
#pragma unroll
  for (int k = 0; k < 16; ++k) { x = p.next[x]; UCLK(t[17 + k]); }
  for (int i = 0; i < 33; ++i) out[i] = t[i];
  out[33] = ((std::uint64_t)x << 32) | x;
}

template <int S>
__global__ void __launch_bounds__(32, 1) k_uc3(std::uint64_t* out) {
  std::uint32_t x = cl_table3[(std::uint32_t)(std::uintptr_t)out & 1023];
  std::uint64_t t[33];
  asm volatile("" :: "l"(out) : "memory");
  UCLK(t[0]);
#pragma unroll
  for (int k = 0; k < 16; ++k) { x = cl_table3[x]; UCLK(t[k + 1]); }
#pragma unroll
  for (int k = 0; k < 16; ++k) { x = cl_table3[x]; UCLK(t[17 + k]); }
  for (int i = 0; i < 33; ++i) out[i] = t[i];
  out[33] = ((std::uint64_t)x << 32) | x;
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.reps != 1) {
    std::fprintf(stderr, "counter-only probe requires --reps 1\n");
    return 2;
  }
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, 34 * sizeof(std::uint64_t)));

  auto launch_line = [&](const char* name, int l) {
    std::uint64_t o[34];
    MB_CHECK(cudaMemcpy(o, d_out, sizeof(o), cudaMemcpyDeviceToHost));
    std::printf("LAUNCH k_cl variant=%s launch=%d t=", name, l);
    for (int i = 0; i < 33; ++i)
      std::printf("%s%llu", i ? "," : "", (unsigned long long)o[i]);
    std::printf(" idx=%llu\n", (unsigned long long)o[33]);
  };

  auto run = [&](const char* name, auto kfn) {
    if (!mb_selected(args, name)) return;
    for (int l = 0; l < args.launches; ++l) {
      kfn();
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      MB_CHECK(cudaGetLastError());
      launch_line(name, l);
    }
  };

  const int s = 256;
  ClParams h{};
  for (int k = 0; k < 16; ++k)
    h.next[k * s / 4] = (k + 1 < 16 ? (k + 1) * s / 4 : 0);
  MB_CHECK(cudaMemcpyToSymbol(cl_table3, h.next, sizeof(h.next)));
  const ClParams p = h;

  run("cl.b0.s256", [&] { k_cl0<s><<<1, 32>>>(p, d_out); });
  run("cl.b3.s256", [&] { k_cl3<s><<<1, 32>>>(d_out); });
  run("cl.u.b0.s256", [&] { k_uc0<s><<<1, 32>>>(p, d_out); });
  run("cl.u.b3.s256", [&] { k_uc3<s><<<1, 32>>>(d_out); });
  return 0;
}
