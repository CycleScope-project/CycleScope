// tools/microbench/control_branch/probe/branch.cu -- protocol.json v1.
// One CTA, W warps in {1,4} (W=4: one warp per SMSP), timing thread
// warp0 lane0, t_s/t_w clock64 around a window of U units, U in
// {16,32,64,128}, straight-line unless noted.
//   taken     : @p bra NEXT; <1 dead FFMA>; NEXT: FFMA chain step
//   nottaken  : @q bra FAR; FFMA chain step      (falls through)
//   uncond    : bra NEXT; <dead FFMA>; NEXT: FFMA chain step
//   taken_dep : setp p (from chain value); @p bra NEXT; dead; NEXT: FFMA
//   loop      : counted loop, body = 1 FFMA step + counter (back edge
//               taken U-1 times)
// Predicates p(true)/q(false) are setp'd once from a kernel argument
// before t_s; labels/%= keep ptxas from folding the branches.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "tensor_pipe/mb.cuh"

#define CHAIN  "fma.rn.f32 %0, %0, %1, %2;"
#define DEAD   "fma.rn.f32 %3, %3, %1, %2;"
#define XS(x)  #x
#define XI(x)  XS(x)

// bra.uni: the predicates are warp-uniform by construction, and .uni
// keeps ptxas from emitting BSSY reconvergence around each unit's
// branch (gate bans BSSY inside the window).  The skipped dead block
// carries fence.gpu so the region cannot be folded away.
template <int V, int I>
__device__ __forceinline__ void unit(float &d, float &dd, float a,
                                     float b) {
  // protocol-v2: the taken/uncond skipped block is a bare fence.gpu
  // (MEMBAR) so the measured BRA cannot be folded by ptxas; branches
  // are plain predicated BRA (not bra.uni).
  if constexpr (V == 0) {           // taken
    asm volatile("@P_t bra L%=; fence.gpu; L%=: " CHAIN
                 : "+f"(d), "+f"(dd) : "f"(a), "f"(b));
  } else if constexpr (V == 1) {    // nottaken: per-unit local target;
    // rotate 4 predicate names so the U bras are not mergeable
    if constexpr (I % 4 == 0)
      asm volatile("@P_f0 bra L%=;" CHAIN "L%=: " : "+f"(d) : "f"(a), "f"(b));
    else if constexpr (I % 4 == 1)
      asm volatile("@P_f1 bra L%=;" CHAIN "L%=: " : "+f"(d) : "f"(a), "f"(b));
    else if constexpr (I % 4 == 2)
      asm volatile("@P_f2 bra L%=;" CHAIN "L%=: " : "+f"(d) : "f"(a), "f"(b));
    else
      asm volatile("@P_f3 bra L%=;" CHAIN "L%=: " : "+f"(d) : "f"(a), "f"(b));
  } else if constexpr (V == 2) {    // uncond
    asm volatile("bra L%=; fence.gpu; L%=: " CHAIN
                 : "+f"(d), "+f"(dd) : "f"(a), "f"(b));
  } else if constexpr (V == 3) {    // taken_dep: setp on the chain value
    asm volatile("setp.ne.f32 P_d, %0, 0F00000000;"
                 "@P_d bra L%=; fence.gpu; L%=: " CHAIN
                 : "+f"(d), "+f"(dd) : "f"(a), "f"(b));
  }
}

template <int V, int U, int I = 0>
struct EmitUnits {
  static __device__ __forceinline__ void go(float &d, float &dd,
                                            float a, float b) {
    unit<V, I>(d, dd, a, b);
    if constexpr (I + 1 < U) EmitUnits<V, U, I + 1>::go(d, dd, a, b);
  }
};
// nottaken trampolines: unique out-of-window targets (never executed)


template <int V, int U>
__global__ void __launch_bounds__(128) k_branch(
    std::uint64_t *out, float *dval, int reps, int wgroups,
    std::uint32_t seed) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp >= wgroups) return;
  float d = 0.25f + warp, dd = 0.f, a = 1.0000001f, b = 0.5f;
  // predicate registers live for the whole function (PTX .reg scope)
  asm(".reg .pred P_t;\n\t.reg .pred P_d;");
  asm("setp.ne.u32 P_t, %0, 0;" ::"r"(seed));
  asm(".reg .pred P_f0;\n\t.reg .pred P_f1;\n\t.reg .pred P_f2;\n\t.reg .pred P_f3;");
  asm("setp.eq.u32 P_f0, %0, 0;\n\t"
      "setp.eq.u32 P_f1, %0, 0;\n\t"
      "setp.eq.u32 P_f2, %0, 0;\n\t"
      "setp.eq.u32 P_f3, %0, 0;" ::"r"(seed));
  int loop_cnt;  // opaque bound: ptxas cannot unroll a runtime count
  asm("or.b32 %0, %1, %2;" : "=r"(loop_cnt) : "r"(seed & ~0xFFFF), "r"(U));
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const std::uint64_t t_s = mb_clock();
    if constexpr (V == 4) {         // counted loop, U iterations
      int cnt = loop_cnt;
      asm volatile("LL%=: " CHAIN
                   "add.s32 %1, %1, -1;"
                   "setp.ne.s32 P_d, %1, 0;"
                   "@P_d bra LL%=;"
                   : "+f"(d), "+r"(cnt) : "f"(a), "f"(b));
    } else {
      EmitUnits<V, U>::go(d, dd, a, b);
    }

    const std::uint64_t t_w = mb_clock();
    if (lane == 0 && warp == 0) {
      out[2 * rep] = t_s; out[2 * rep + 1] = t_w;
      // keep the skipped dead FFMA chain observable per rep so ptxas
      // cannot dead-code-eliminate the block the branch skips
      dval[(rep & 31) + 1] = dd;
    }
  }
  if (threadIdx.x == 0) { dval[0] = d + dd; }
}

using BKern = void (*)(std::uint64_t *, float *, int, int, std::uint32_t);
struct BCell { const char *variant; int u; BKern kern; };
static const int B_US[4] = {16, 32, 64, 128};

#define ROW(V, NAME)                                                \
  {NAME, 16, &k_branch<V, 16>}, {NAME, 32, &k_branch<V, 32>},        \
  {NAME, 64, &k_branch<V, 64>}, {NAME, 128, &k_branch<V, 128>},
static const BCell B_CELLS[] = {
  ROW(0, "taken") ROW(1, "nottaken") ROW(2, "uncond")
  ROW(3, "taken_dep") ROW(4, "loop")
};
#undef ROW

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  std::string probe = argv[0];
  const std::size_t sl = probe.rfind('/');
  if (sl != std::string::npos) probe = probe.substr(sl + 1);
  MbWriter w(args, probe.c_str());
  std::uint64_t *d_out = nullptr;
  float *d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(float) * 40));
  for (const BCell &c : B_CELLS) {
    for (int wgroups = 1; wgroups <= 4; wgroups += 3) {
      std::string variant = std::string(c.variant) + ".u" +
                            std::to_string(c.u) + ".w" +
                            std::to_string(wgroups);
      if (!mb_selected(args, variant)) continue;
      std::vector<double> samples;
      for (int l = 0; l < args.launches; ++l) {
        c.kern<<<1, 32 * wgroups>>>(d_out, d_val, args.reps, wgroups,
                                    static_cast<std::uint32_t>(l) + 1u);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
          std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                       cudaGetErrorString(e));
          break;
        }
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(2 * static_cast<std::size_t>(args.reps));
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                            cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r)
          samples.push_back(static_cast<double>(
              static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
      }
      const std::string params = "\"variant\":\"" +
          std::string(c.variant) + "\",\"w\":" + std::to_string(wgroups);
      w.row(variant, c.u, params, samples);
    }
  }
  return 0;
}
