// memory.constant_launch protocol v2 probe: U = 16 dependent
// register-indexed constant loads per traversal, two traversals, a
// CS2R SR_CLOCKLO before the first load and after every load
// (t_0 .. t_32).  The chain register is the absolute next load
// address; mad.wide.u32 v*4+base is the one allowed ALU step between
// consecutive LDCs (same op every pair, class recorded by the gate).
//
//   BANK 0: table = by-value kernel parameter struct (4096 B
//           uint32 next[1024]); BANK 3: 4096 B __constant__ array
//           written by cudaMemcpyToSymbol per variant before launches.
// Variants cl.b<BANK>.s<S>, BANK in {0,3}, S in {64,256}
// (16 * S <= 4096 B), W = 1 warp, grid 1.
// Chain: load k reads element k*S/4; next[k*S/4] = (k+1)*S/4 for
// k < 15 and 0 for k = 15 (traversal re-starts at element 0).
// The bases MUST be the space-local addresses (param / const), not
// the cvta generic ones a C++ cast would produce.
#include "../mb.cuh"

#include <string>
#include <vector>

struct ClParams {
  std::uint32_t next[1024];
};

__constant__ std::uint32_t cl_table3[1024];

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
  asm("mov.u64 %0, %1;" : "=l"(base) : "l"(p.next));  // param-space
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
  asm("mov.u64 %0, cl_table3;" : "=l"(base));   // const-space address
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

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  if (args.reps != 1) {
    std::fprintf(stderr, "counter-only probe requires --reps 1\n");
    return 2;
  }
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, 34 * sizeof(std::uint64_t)));

  const int ss[] = {64, 256};
  auto run = [&](int bank, int s, auto kfn) {
    char name[64];
    std::snprintf(name, sizeof(name), "cl.b%d.s%d", bank, s);
    if (!mb_selected(args, name)) return;
    ClParams h{};
    for (int k = 0; k < 16; ++k)
      h.next[k * s / 4] = (k + 1 < 16 ? (k + 1) * s / 4 : 0);
    if (bank == 3)
      MB_CHECK(cudaMemcpyToSymbol(cl_table3, h.next, sizeof(h.next)));
    for (int l = 0; l < args.launches; ++l) {
      kfn(h, d_out);
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      MB_CHECK(cudaGetLastError());
      std::uint64_t o[34];
      MB_CHECK(cudaMemcpy(o, d_out, sizeof(o), cudaMemcpyDeviceToHost));
      std::printf("LAUNCH k_cl variant=%s launch=%d t=", name, l);
      for (int i = 0; i < 33; ++i)
        std::printf("%s%llu", i ? "," : "", (unsigned long long)o[i]);
      std::printf(" idx=%llu\n", (unsigned long long)o[33]);
    }
  };

  for (int s : ss) {
    auto f = (s == 64 ? k_cl0<64> : k_cl0<256>);
    run(0, s, [&](const ClParams& p, std::uint64_t* o) {
      f<<<1, 32>>>(p, o);
    });
    auto g = (s == 64 ? k_cl3<64> : k_cl3<256>);
    run(3, s, [&](const ClParams&, std::uint64_t* o) {
      g<<<1, 32>>>(o);
    });
  }
  return 0;
}
