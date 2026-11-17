// storage.tma.descriptor_refetch (crossings protocol-v2), built on the
// tma_cold xw scheme: 1 CTA of 64 threads.  warp 0 lane 0 does
// expect_tx -> t_s -> one UTMALDG (SW128 bf16 box 64 x 128 B = 16 KiB,
// L2-hot tile warmed at kernel start) -> t_e; warp 1 lane 0 spins in
// test_wait and reads t_w.  K = t_w - t_s; .issue = t_e - t_s.
//
// Per rep r the map handle is maps[r] (an array of M = reps copies of one
// CUtensorMap in global memory, rewritten by the host before every
// launch), so each rep's "first" cell is the first use of that map in this
// launch.  Cells per rep:
//   first    expect_tx -> t_s -> UTMALDG(maps[r])
//   reuse    UTMALDG(maps[r]) again
//   acquire  fence.proxy.tensormap::generic.acquire.gpu [maps[r]],128,
//            then t_s -> UTMALDG(maps[r])
//   param    UTMALDG(__grid_constant__ map) -- rep 0 only: the param map's
//            first use is once per launch, so its samples come from the
//            launches (--launches 64).
#include "../mb.cuh"

#include <cuda.h>
#include <dlfcn.h>

#include <cstring>
#include <string>
#include <vector>

constexpr int kInner = 64;    // bf16 elements per box row (128 B, SWIZZLE_128B)
constexpr int kBoxRows = 128; // 16 KiB box
constexpr int kRows = 256;    // the whole tensor is one box column

__device__ __forceinline__ void tma_load(std::uint32_t dst, const CUtensorMap* map, int x, int y, std::uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(reinterpret_cast<std::uint64_t>(map)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void expect_tx(std::uint32_t mbar, std::uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tmap_acquire(const CUtensorMap* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(
                   reinterpret_cast<std::uint64_t>(map))
               : "memory");
}

// CELL: 0 first / 1 reuse / 2 acquire / 3 param (issue only; all four use the
// same barrier dance so the waiter code is one path).
__device__ __forceinline__ void issue_cell(std::uint32_t buf, const CUtensorMap* map, std::uint32_t a,
                                           std::uint64_t* out3) {
  expect_tx(a, kBoxRows * kInner * 2u);
  const std::uint64_t t_s = mb_clock();
  tma_load(buf, map, 0, 0, a);
  const std::uint64_t t_e = mb_clock();
  out3[0] = t_s;
  out3[2] = t_e;
}

__global__ void __launch_bounds__(64, 1) k_tmap(const __grid_constant__ CUtensorMap gmap, const CUtensorMap* maps,
                                                const std::uint32_t* gbuf, std::uint64_t* out, int reps) {
  extern __shared__ unsigned char smem_raw[];
  const std::uint32_t raw = mb_smem_u32(smem_raw);
  const std::uint32_t buf = (raw + 1023u) & ~1023u;
  __shared__ __align__(8) std::uint64_t mbar_s[2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::uint32_t a = mb_opaque(mb_smem_u32(&mbar_s[0])), ack = mb_opaque(mb_smem_u32(&mbar_s[1]));
  if (threadIdx.x == 0) {
    mb_mbar_init(a, 1);
    mb_mbar_init(ack, 1);
    mb_fence_mbar_init();
    // no prefetch.tensormap: the param cell measures the map's first use
  }
  // keep the tile L2-hot: warp 0 walks the whole 32 KiB global region once
  if (warp == 0) {
    std::uint32_t acc = 0;
    for (std::size_t o = lane * 4; o < static_cast<std::size_t>(kRows) * kInner * 2; o += 32 * 4)
      acc ^= __ldg(reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const unsigned char*>(gbuf) + o));
    if (acc == 0xdeadbeefu) out[0] = acc;  // never taken; keeps the walk alive
  }
  __syncthreads();
  std::uint32_t ph = 0;
  if (lane == 0 && warp == 0) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      const CUtensorMap* m = maps + rep;  // first use of this map in the launch
      issue_cell(buf, m, a, out + 16 * rep + 0);
      mb_mbar_wait_to(ack, ph);
      ph ^= 1u;
      issue_cell(buf, m, a, out + 16 * rep + 4);  // reuse
      mb_mbar_wait_to(ack, ph);
      ph ^= 1u;
      tmap_acquire(m);
      issue_cell(buf, m, a, out + 16 * rep + 8);  // acquire
      mb_mbar_wait_to(ack, ph);
      ph ^= 1u;
      if (rep == 0) {
        issue_cell(buf, &gmap, a, out + 16 * rep + 12);  // param
        mb_mbar_wait_to(ack, ph);
        ph ^= 1u;
      }
    }
  } else if (lane == 0 && warp == 1) {
#pragma unroll 1
    for (int rep = 0; rep < reps; ++rep) {
      mb_mbar_spin_to(a, ph);
      out[16 * rep + 1] = mb_clock();  // t_w first
      ph ^= 1u;
      mb_mbar_arrive(ack);
      mb_mbar_spin_to(a, ph);
      out[16 * rep + 5] = mb_clock();  // t_w reuse
      ph ^= 1u;
      mb_mbar_arrive(ack);
      mb_mbar_spin_to(a, ph);
      out[16 * rep + 9] = mb_clock();  // t_w acquire
      ph ^= 1u;
      mb_mbar_arrive(ack);
      if (rep == 0) {
        mb_mbar_spin_to(a, ph);
        out[16 * rep + 13] = mb_clock();  // t_w param
        ph ^= 1u;
        mb_mbar_arrive(ack);
      }
    }
  }
  __syncwarp();
  __syncthreads();
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn encoder() {
  void* sym = dlsym(RTLD_DEFAULT, "cuTensorMapEncodeTiled");
  if (!sym) sym = dlsym(dlopen("libcuda.so.1", RTLD_NOW), "cuTensorMapEncodeTiled");
  return reinterpret_cast<EncodeTiledFn>(sym);
}

static CUtensorMap make_map(void* g) {
  CUtensorMap m{};
  const cuuint64_t dim[2] = {static_cast<cuuint64_t>(kInner), static_cast<cuuint64_t>(kRows)};
  const cuuint64_t stride[1] = {static_cast<cuuint64_t>(kInner * 2)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kInner), static_cast<cuuint32_t>(kBoxRows)};
  const cuuint32_t es[2] = {1u, 1u};
  auto fn = encoder();
  if (!fn || fn(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, dim, stride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
    std::exit(2);
  }
  return m;
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "tmap");
  const std::size_t tensor_bytes = static_cast<std::size_t>(kRows) * kInner * 2;
  std::uint32_t* gbuf = nullptr;
  MB_CHECK(cudaMalloc(&gbuf, tensor_bytes));
  MB_CHECK(cudaMemset(gbuf, 0x3c, tensor_bytes));
  CUtensorMap* d_maps = nullptr;
  MB_CHECK(cudaMalloc(&d_maps, sizeof(CUtensorMap) * args.reps));
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 16 * args.reps));
  const CUtensorMap map = make_map(gbuf);
  // M = reps copies of the template map, 128 B each, rewritten per launch
  std::vector<CUtensorMap> maps(args.reps, map);
  const int dyn = 2 * 1024 + kBoxRows * kInner * 2;
  MB_CHECK(cudaFuncSetAttribute(k_tmap, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));

  const char* cells[4] = {"first", "reuse", "acquire", "param"};
  std::vector<double> k[4], issue[4];
  bool timed_out = false;
  for (int l = 0; l < args.launches && !timed_out; ++l) {
    MB_CHECK(cudaMemcpy(d_maps, maps.data(), sizeof(CUtensorMap) * args.reps, cudaMemcpyHostToDevice));
    k_tmap<<<1, 64, dyn>>>(map, d_maps, gbuf, d_out, args.reps);
    MB_CHECK(cudaGetLastError());
    MB_CHECK(cudaDeviceSynchronize());
    timed_out = mb_take_err();
    std::vector<std::uint64_t> h(16 * static_cast<std::size_t>(args.reps));
    MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
    for (int r = 1; r < args.reps; ++r) {
      for (int c = 0; c < 3; ++c) {
        k[c].push_back(static_cast<double>(static_cast<std::int64_t>(h[16 * r + 4 * c + 1] - h[16 * r + 4 * c + 0])));
        issue[c].push_back(static_cast<double>(static_cast<std::int64_t>(h[16 * r + 4 * c + 2] - h[16 * r + 4 * c + 0])));
      }
    }
    if (args.reps > 0) {
      // param: the grid-constant map's first use is rep 0 of every launch
      k[3].push_back(static_cast<double>(static_cast<std::int64_t>(h[13] - h[12])));
      issue[3].push_back(static_cast<double>(static_cast<std::int64_t>(h[14] - h[12])));
    }
  }
  if (timed_out) {
    for (int c = 0; c < 4; ++c) w.fail(std::string("tmap.") + cells[c], 1, std::string("\"cell\":\"") + cells[c] + "\"",
                                     "mbarrier wait timed out");
    return 1;
  }
  for (int c = 0; c < 4; ++c) {
    const std::string params = std::string("\"cell\":\"") + cells[c] + "\",\"box_bytes\":16384,\"l2\":\"hot\"";
    w.row(std::string("tmap.") + cells[c], 1, params, k[c]);
    w.row(std::string("tmap.") + cells[c] + ".issue", 1, params, issue[c]);
  }
  return w.failures ? 1 : 0;
}
