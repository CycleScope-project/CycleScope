// tools/microbench/wgmma_handoff/probe/handoff.cu — H100 (sm_90a).
//
// protocol.json: after wgmma.wait_group D completes group A (D later
// groups still pending), the first FFMA that reads the just-completed
// accumulator waits an extra handoff delay vs an ordinary register.
// One CTA of one warpgroup (128 threads); timing thread = warp 0 lane 0.
//   Per rep: wgmma.fence; wgmma -> acc set a; commit_group;
//            D x { wgmma -> acc set b; commit_group };
//            wgmma.wait_group D; t_w; x = fma.rn.f32(src, c, c);
//            y = add.f32(x, 1.0); t_e; wgmma.wait_group 0; fold.
//   src: acc cell = a[0]; control cell = a pinned seed register.
// Descriptor/op helpers come from the generated wgmma_impl.cuh (gen_wgmma.py).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "../../tensor_pipe/mb.cuh"

// no-swizzle K-major descriptor (same as tensor_pipe/probe/wgmma.cu).
__device__ __forceinline__ std::uint64_t w_desc(const void* smem_ptr) {
  const std::uint32_t addr =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_ptr));
  std::uint64_t desc = 0;
  desc |= static_cast<std::uint64_t>((addr >> 4) & 0x3fff);
  desc |= static_cast<std::uint64_t>((128 >> 4) & 0x3fff) << 16;
  desc |= static_cast<std::uint64_t>((256 >> 4) & 0x3fff) << 32;
  return desc;
}

// all kinds on so the CID enumeration matches gen_wgmma.py cell_list()
#include "wgmma_impl.cuh"

static constexpr int kSmemBytes = 48 * 1024;

// kernel: CID = cell id (wgmma_cells.json), D = group_depth, CTRL = 1 for
// the ordinary-register control cell, 0 for the accumulator cell.
template <int CID, int D, int CTRL>
__global__ void __launch_bounds__(128, 1) k_handoff(
    std::uint64_t* out, std::uint32_t* dval, int reps,
    std::uint32_t seed) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t base = (raw + 1023u) & ~1023u;
  unsigned char* smem = smem_raw + (base - raw);
  const std::uint64_t da = w_desc(smem), db = w_desc(smem + 16384);
  using T = typename WD<CID>::dty;
  static_assert(std::is_same<T, float>::value, "f32 acc cells only");
  T dda[WD<CID>::ND];
  T ddb[WD<CID>::ND];
  for (int i = 0; i < WD<CID>::ND; ++i) { dda[i] = T{}; ddb[i] = T{}; }
  __shared__ std::uint32_t pin_slot[1];
  const float src_ctrl =
      __int_as_float(mb_pin(seed ^ 0x9e3779b9u, pin_slot));
  // second fma operand pinned in a register: two immediate operands would
  // materialize an IMAD.MOV inside the timed window
  const float c_two = __int_as_float(mb_pin(0x40000000u, pin_slot));
  float acc = 0.f;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    WOp<CID>::op(dda, da, db);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#pragma unroll
    for (int g = 0; g < D; ++g) {
      WOp<CID>::op(ddb, da, db);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    }
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(D) : "memory");
    // consumer: FFMA reads a[0] (acc) or the pinned seed (control);
    // the FADD chain keeps the window to the two stated instructions
    float x, y;
    const float v = CTRL ? src_ctrl : dda[0];  // FFMA reads a[0] itself
    const std::uint64_t t_w = mb_clock();
    asm volatile("fma.rn.f32 %0, %1, %2, %2;"
                 : "=f"(x) : "f"(v), "f"(c_two));
    asm volatile("add.f32 %0, %1, 0f3F800000;" : "=f"(y) : "f"(x));
    const std::uint64_t t_e = mb_clock();
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (threadIdx.x == 0) {
      out[2 * rep] = t_w;
      out[2 * rep + 1] = t_e;
    }
    acc += x + y;
  }
  // fold every live register into a sink (keep accumulators + consumer live)
  if (threadIdx.x == 0) {
    std::uint32_t s = __float_as_uint(acc);
    for (int i = 0; i < WD<CID>::ND; ++i) {
      std::uint32_t w0, w1;
      std::memcpy(&w0, &dda[i], sizeof(w0) < sizeof(T) ? sizeof(w0) : sizeof(T));
      std::memcpy(&w1, &ddb[i], sizeof(w1) < sizeof(T) ? sizeof(w1) : sizeof(T));
      s += w0 + w1;
    }
    dval[0] = s;
  }
}

using HKern = void (*)(std::uint64_t*, std::uint32_t*, int, std::uint32_t);

struct HCell {
  int cid; const char* form; const char* cls; const char* kind;
  const char* acc; int k; int n; const char* src;
};

// CIDs = gen_wgmma.py cell_list() enumeration: per kind (order
// f16,f16a,bf16,tf32,e4m3e4m3,e4m3e5m2,e5m2e5m2,s8s8) x n in
// {8,16,...,256} x {ss,rs}: cid = kind_idx*64 + (n/8-1)*2 + (src==rs)
static const HCell HC[] = {
    {142, "hgmma_bf16_n64k16_ss",  "HGMMA", "bf16",     "f32", 16,  64, "ss"},
    {158, "hgmma_bf16_n128k16_ss", "HGMMA", "bf16",     "f32", 16, 128, "ss"},
    {159, "hgmma_bf16_n128k16_rs", "HGMMA", "bf16",     "f32", 16, 128, "rs"},
    {46,  "hgmma_f16_n192k16_ss",  "HGMMA", "f16",      "f32", 16, 192, "ss"},
    {222, "hgmma_tf32_n128k8_ss",  "HGMMA", "tf32",     "f32",  8, 128, "ss"},
    {286, "qgmma_e4m3_n128k32_ss", "QGMMA", "e4m3e4m3", "f32", 32, 128, "ss"},
    {288, "qgmma_e4m3_n136k32_ss", "QGMMA", "e4m3e4m3", "f32", 32, 136, "ss"},
    {302, "qgmma_e4m3_n192k32_ss", "QGMMA", "e4m3e4m3", "f32", 32, 192, "ss"},
};
static const int HC_N = sizeof(HC) / sizeof(HCell);

template <int CID, int D, int CTRL>
HKern kget() { return &k_handoff<CID, D, CTRL>; }

template <int CID>
struct OneCell {
  static void add(HKern t[4][2]) {
    t[0][0] = &k_handoff<CID, 0, 0>; t[0][1] = &k_handoff<CID, 0, 1>;
    t[1][0] = &k_handoff<CID, 1, 0>; t[1][1] = &k_handoff<CID, 1, 1>;
    t[2][0] = &k_handoff<CID, 2, 0>; t[2][1] = &k_handoff<CID, 2, 1>;
    t[3][0] = &k_handoff<CID, 3, 0>; t[3][1] = &k_handoff<CID, 3, 1>;
  }
};

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "handoff");
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_val = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * args.reps));
  MB_CHECK(cudaMalloc(&d_val, sizeof(std::uint32_t) * 2));
  HKern table[HC_N][4][2];
  OneCell<142>::add(table[0]);
  OneCell<158>::add(table[1]);
  OneCell<159>::add(table[2]);
  OneCell<46>::add(table[3]);
  OneCell<222>::add(table[4]);
  OneCell<286>::add(table[5]);
  OneCell<288>::add(table[6]);
  OneCell<302>::add(table[7]);
  for (int ci = 0; ci < HC_N; ++ci) {
    const HCell& c = HC[ci];
    for (int d = 0; d < 4; ++d) {
      for (int ctrl = 0; ctrl < 2; ++ctrl) {
        const std::string variant =
            std::string(c.form) + ".d" + std::to_string(d) +
            (ctrl ? ".ctrl" : ".acc");
        if (!mb_selected(args, variant)) continue;
        for (int l = 0; l < args.launches; ++l) {
          std::vector<double> samples;
          MB_CHECK(cudaFuncSetAttribute(
              table[ci][d][ctrl],
              cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
          table[ci][d][ctrl]<<<1, 128, kSmemBytes>>>(
              d_out, d_val, args.reps, static_cast<std::uint32_t>(l));
          const cudaError_t e = cudaGetLastError();
          if (e != cudaSuccess) {
            std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                         cudaGetErrorString(e));
            break;
          }
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(
              2 * static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                              cudaMemcpyDeviceToHost));
          for (int r = 1; r < args.reps; ++r)
            samples.push_back(static_cast<double>(
                static_cast<std::int64_t>(h[2 * r + 1] - h[2 * r])));
          const std::string params =
            "\"form\":\"" + std::string(c.form) + "\",\"class\":\"" + c.cls +
            "\",\"dtype\":\"" + c.kind + "\",\"acc\":\"" + c.acc +
            "\",\"src\":\"" + c.src + "\",\"m\":64,\"n\":" +
            std::to_string(c.n) + ",\"k\":" + std::to_string(c.k) +
            ",\"d\":" + std::to_string(d) + ",\"cell\":\"" +
            (ctrl ? "ctrl" : "acc") + "\",\"launch\":" +
            std::to_string(l);
          w.row(variant, d, params, samples);
        }
      }
    }
  }
  return 0;
}
