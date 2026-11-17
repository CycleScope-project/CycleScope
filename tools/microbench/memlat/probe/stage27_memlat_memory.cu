// Stage27 MemLat probe: memory-hierarchy dependent-load latency on B300S.
//
// Targets (shopping list P0-1 / P1-4):
//   memory.dram.service.root  dram.latency_cycles (installed 400, H100 588)
//   memory.l2.service.root    l2.latency_cycles   (installed  96, H100 287)
//   memory.l1.service.root    l1.latency_cycles   (installed  32, sanity)
//
// Points:
//   dram.chain      dependent pointer chase, 2GiB random 4KiB-page permutation,
//                   512MiB scratch memset evicts L2 before every sample;
//                   ld.global.cg primary + ld.global.cv cross-check
//                   (reproduces p0_rerun 860.78 high-confidence anchor).
//   dram.width      same at access width 32/64/128B (MLP=1).
//   dram.mlp        W independent chains in one thread, W in {1,2,4,8,16,32}:
//                   cycles per hop per chain -- the memory-level-parallelism
//                   curve; W=1 must reproduce dram.chain.
//   l2.chain        2MiB random permutation chain; FULL warm walk of the
//                   permutation inside the kernel before the clock bracket
//                   (p0_rerun fix: its random start page was often cold ->
//                   bimodal 310/780).  ld.global.cg, 128 timed hops.
//   l1.chain        8KiB random chain, full warm walk, ld.global.ca (default).
//
// Raw per-sample cycles/hop are appended as one jsonl row per point
// (blackwell.mechanism_curve_point.v1); per-sample values live in "samples".
#include "common.cuh"
#include "curves_io.cuh"

#include <numeric>
#include <random>

__global__ void init_chain_kernel(std::uint32_t* buf, const std::uint32_t* perm,
                                  std::uint64_t npages) {
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i < npages) buf[i * 1024] = perm[i];
}

// Random permutation remapped into a single cycle: chain[order[k]] = order[k+1].
static std::vector<std::uint32_t> make_random_cycle(std::size_t n, std::mt19937_64& rng) {
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<std::uint32_t> chain(n);
  for (std::size_t k = 0; k < n; ++k) chain[order[k]] = order[(k + 1) % n];
  return chain;
}

__device__ __forceinline__ std::uint32_t ld_cg_u32(const std::uint32_t* p) {
  std::uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ std::uint32_t ld_cv_u32(const std::uint32_t* p) {
  std::uint32_t v;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ std::uint64_t ld_cg_u64(const std::uint64_t* p) {
  std::uint64_t v;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ld_cg_v4(const std::uint32_t* p, std::uint32_t& a,
                                         std::uint32_t& b, std::uint32_t& c,
                                         std::uint32_t& d) {
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
}

// ---- dram.chain: single dependent chase (mode 0=cg, 1=cv) ------------------
__global__ void memlat_chase_kernel(const std::uint32_t* p, long long* out, int hops,
                                    int mode, std::uint32_t start) {
  std::uint32_t idx = start;
#pragma unroll 1
  for (int i = 0; i < 16; ++i) {  // warm the descriptor/L1TLB path off the clock
    idx = (mode == 1) ? ld_cv_u32(p + (std::uint64_t)idx * 1024)
                      : ld_cg_u32(p + (std::uint64_t)idx * 1024);
  }
  const std::uint64_t t0 = rdclk64();
#pragma unroll 1
  for (int i = 0; i < hops; ++i) {
    idx = (mode == 1) ? ld_cv_u32(p + (std::uint64_t)idx * 1024)
                      : ld_cg_u32(p + (std::uint64_t)idx * 1024);
  }
  const std::uint64_t t1 = rdclk64();
  if (threadIdx.x == 0) {
    out[0] = static_cast<long long>(t1 - t0);
    out[1] = static_cast<long long>(idx);
  }
}

// ---- dram.width: width w in {32,64,128}, MLP=1 dependent chain -------------
// Chain word layout: one word per 4KiB page (u32 chain) for w=32; for 64/128
// every loaded word is folded (xor) into the next index so ptxas cannot
// dead-code-eliminate the upper words of the wider access.
__global__ void memlat_width_kernel(const std::uint32_t* p, long long* out, int hops,
                                    int width, std::uint32_t start) {
  std::uint32_t idx = start;
#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    if (width == 128) {
      std::uint32_t a, b, c, d;
      ld_cg_v4(p + (std::uint64_t)idx * 1024, a, b, c, d);
      idx = (a ^ b ^ c ^ d) & 0x7FFFFu;  // fold to a valid page (pages=2^19)
    } else if (width == 64) {
      const std::uint64_t v = ld_cg_u64(reinterpret_cast<const std::uint64_t*>(
          p + (std::uint64_t)idx * 1024));
      idx = static_cast<std::uint32_t>((v ^ (v >> 32)) & 0x7FFFFu);
    } else {
      idx = ld_cg_u32(p + (std::uint64_t)idx * 1024);
    }
  }
  const std::uint64_t t0 = rdclk64();
#pragma unroll 1
  for (int i = 0; i < hops; ++i) {
    if (width == 128) {
      std::uint32_t a, b, c, d;
      ld_cg_v4(p + (std::uint64_t)idx * 1024, a, b, c, d);
      idx = (a ^ b ^ c ^ d) & 0x7FFFFu;  // fold to a valid page (pages=2^19)
    } else if (width == 64) {
      const std::uint64_t v = ld_cg_u64(reinterpret_cast<const std::uint64_t*>(
          p + (std::uint64_t)idx * 1024));
      idx = static_cast<std::uint32_t>((v ^ (v >> 32)) & 0x7FFFFu);
    } else {
      idx = ld_cg_u32(p + (std::uint64_t)idx * 1024);
    }
  }
  const std::uint64_t t1 = rdclk64();
  if (threadIdx.x == 0) {
    out[0] = static_cast<long long>(t1 - t0);
    out[1] = static_cast<long long>(idx);
  }
}

// ---- dram.mlp: W independent chains, one thread, H hops each ---------------
// Per-W template instantiations keep every chain in its own scalar register
// (a shared idx[32] array got demoted to local storage by ptxas and the SASS
// gate caught the collapsed LDG count).  elapsed/H = cycles per hop per chain
// at MLP=W.
template <int W>
__global__ void memlat_mlp_t_kernel(const std::uint32_t* p, long long* out, int hops,
                                    std::uint32_t start) {
  std::uint32_t idx[W];
#pragma unroll
  for (int w = 0; w < W; ++w) idx[w] = start + static_cast<std::uint32_t>(w * 4099);
#pragma unroll 1
  for (int i = 0; i < 4; ++i) {  // warm off the clock
#pragma unroll
    for (int w = 0; w < W; ++w) idx[w] = ld_cg_u32(p + (std::uint64_t)idx[w] * 1024);
  }
  const std::uint64_t t0 = rdclk64();
#pragma unroll 1
  for (int i = 0; i < hops; ++i) {
#pragma unroll
    for (int w = 0; w < W; ++w) idx[w] = ld_cg_u32(p + (std::uint64_t)idx[w] * 1024);
  }
  const std::uint64_t t1 = rdclk64();
  if (threadIdx.x == 0) {
    out[0] = static_cast<long long>(t1 - t0);
    // Every chain cursor must be observably stored: asm volatile keeps the
    // asm statement in PTX but ptxas still dead-code-eliminates non-volatile
    // ld.global results that are never consumed (the compile-only gate caught
    // exactly this: 2 surviving LDGs for a 32-chain kernel).
#pragma unroll
    for (int w = 0; w < W; ++w) out[1 + w] = static_cast<long long>(idx[w]);
  }
}

// ---- warm-walk chain kernel (L2 / L1) --------------------------------------
// Walks the entire permutation once (warm), clocks `hops` further hops.
__global__ void memlat_warmchase_kernel(const std::uint32_t* p, long long* out,
                                        int warm_hops, int hops, int l1_mode,
                                        std::uint32_t start) {
  std::uint32_t idx = start;
  const std::uint64_t stride = l1_mode ? 1 : 1024;  // words: 4B (l1) / 4KiB (l2)
#pragma unroll 1
  for (int i = 0; i < warm_hops; ++i) {
    if (l1_mode) {
      asm volatile("ld.global.ca.u32 %0, [%1];"
                   : "=r"(idx) : "l"(p + idx * stride) : "memory");
    } else {
      idx = ld_cg_u32(p + idx * stride);
    }
  }
  const std::uint64_t t0 = rdclk64();
#pragma unroll 1
  for (int i = 0; i < hops; ++i) {
    if (l1_mode) {
      asm volatile("ld.global.ca.u32 %0, [%1];"
                   : "=r"(idx) : "l"(p + idx * stride) : "memory");
    } else {
      idx = ld_cg_u32(p + idx * stride);
    }
  }
  const std::uint64_t t1 = rdclk64();
  if (threadIdx.x == 0) {
    out[0] = static_cast<long long>(t1 - t0);
    out[1] = static_cast<long long>(idx);
  }
}

static std::string arr_ll(const std::vector<long long>& xs) {
  char b[32];
  std::string s = "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) s += ",";
    std::snprintf(b, sizeof(b), "%lld", xs[i]);
    s += b;
  }
  s += "]";
  return s;
}

int main(int argc, char** argv) {
  std::string mode = "dram";
  const char* out = "stage27_memlat_memory.jsonl";
  int width = 32, chains = 1, samples = 32, warm = 4, variant = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto nexts = [&](std::string* d) {
      if (i + 1 >= argc) std::exit(2);
      *d = argv[++i];
    };
    auto nexti = [&](int* d) {
      if (i + 1 >= argc) std::exit(2);
      *d = std::atoi(argv[++i]);
    };
    if (a == "--mode") nexts(&mode);
    else if (a == "--width") nexti(&width);
    else if (a == "--chains") nexti(&chains);
    else if (a == "--samples") nexti(&samples);
    else if (a == "--warmup") nexti(&warm);
    else if (a == "--variant") nexti(&variant);  // dram: 0=cg 1=cv
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else return 2;
  }
  const DeviceInfo dev = query_device();
  std::mt19937_64 rng(0xB3005u);
  constexpr int kHops = 64;

  long long* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(long long)));
  long long h[2] = {0, 0};
  std::vector<double> per_hop;
  std::vector<long long> raw;
  std::string err;

  if (mode == "dram") {
    const std::size_t bytes = 2ull << 30;
    const std::uint64_t pages = bytes / 4096;
    const std::size_t scratch_bytes = 512ull << 20;
    auto chain = make_random_cycle(pages, rng);
    std::uint32_t *dramd = nullptr, *permd = nullptr, *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&dramd, bytes));
    CUDA_CHECK(cudaMalloc(&permd, pages * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cudaMemcpy(permd, chain.data(), pages * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    init_chain_kernel<<<(unsigned)((pages + 255) / 256), 256>>>(dramd, permd, pages);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const int cv = variant ? 1 : 0;
    auto evict = [&]() {
      CUDA_CHECK(cudaMemset(scratch, 0xA5, scratch_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto launch = [&](std::uint32_t start) {
      memlat_chase_kernel<<<1, 1>>>(dramd, d, kHops, cv, start);
      cudaError_t x = cudaGetLastError();
      if (x != cudaSuccess) return x;
      return cudaDeviceSynchronize();
    };
    evict();
    for (int i = 0; i < warm; ++i) launch(0);
    for (int s = 0; s < samples; ++s) {
      evict();
      cudaError_t e = launch(static_cast<std::uint32_t>(rng() % pages));
      if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
      CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
      raw.push_back(h[0]);
      per_hop.push_back(static_cast<double>(h[0]) / kHops);
    }
    cudaFree(dramd); cudaFree(permd); cudaFree(scratch);
    CurvePoint p;
    p.family = cv ? "stage27_memlat.dram.chain.cv" : "stage27_memlat.dram.chain.cg";
    p.quantity = "latency";
    p.x_name = "mlp_chains";
    p.x = 1;
    p.y_name = "cycles_per_hop";
    p.samples = per_hop;
    char b[2048];
    std::snprintf(b, sizeof(b),
                  "\"raw_elapsed_cycles\":%s,\"hops\":%d,\"working_set_bytes\":%zu,"
                  "\"evict\":\"512MiB memset per sample\",\"policy\":\"%s\","
                  "\"source\":\"stage27.memlat.measured\",\"gpu_name\":\"%s\","
                  "\"sm_clock_mhz\":%d,\"clock_locked\":false",
                  arr_ll(raw).c_str(), kHops, bytes, cv ? "cv" : "cg",
                  dev.name.c_str(), dev.sm_clock_mhz);
    p.extra_json = b;
    if (!err.empty()) p.error = err;
    append_jsonl(out, p);
    return err.empty() ? 0 : 1;
  }

  if (mode == "dram_width") {
    const std::size_t bytes = 2ull << 30;
    const std::uint64_t pages = bytes / 4096;
    const std::size_t scratch_bytes = 512ull << 20;
    auto chain = make_random_cycle(pages, rng);
    std::uint32_t *dramd = nullptr, *permd = nullptr, *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&dramd, bytes));
    CUDA_CHECK(cudaMalloc(&permd, pages * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cudaMemcpy(permd, chain.data(), pages * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    init_chain_kernel<<<(unsigned)((pages + 255) / 256), 256>>>(dramd, permd, pages);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto evict = [&]() {
      CUDA_CHECK(cudaMemset(scratch, 0xA5, scratch_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto launch = [&](std::uint32_t start) {
      memlat_width_kernel<<<1, 1>>>(dramd, d, kHops, width, start);
      cudaError_t x = cudaGetLastError();
      if (x != cudaSuccess) return x;
      return cudaDeviceSynchronize();
    };
    evict();
    for (int i = 0; i < warm; ++i) launch(0);
    for (int s = 0; s < samples; ++s) {
      evict();
      cudaError_t e = launch(static_cast<std::uint32_t>(rng() % pages));
      if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
      CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
      raw.push_back(h[0]);
      per_hop.push_back(static_cast<double>(h[0]) / kHops);
    }
    cudaFree(dramd); cudaFree(permd); cudaFree(scratch);
    CurvePoint p;
    p.family = "stage27_memlat.dram.width";
    p.quantity = "latency";
    p.x_name = "width_bytes";
    p.x = width;
    p.y_name = "cycles_per_hop";
    p.samples = per_hop;
    char b[2048];
    std::snprintf(b, sizeof(b),
                  "\"raw_elapsed_cycles\":%s,\"hops\":%d,\"working_set_bytes\":%zu,"
                  "\"evict\":\"512MiB memset per sample\",\"policy\":\"cg\","
                  "\"source\":\"stage27.memlat.measured\",\"gpu_name\":\"%s\","
                  "\"sm_clock_mhz\":%d,\"clock_locked\":false",
                  arr_ll(raw).c_str(), kHops, bytes, dev.name.c_str(), dev.sm_clock_mhz);
    p.extra_json = b;
    if (!err.empty()) p.error = err;
    append_jsonl(out, p);
    return err.empty() ? 0 : 1;
  }

  if (mode == "dram_mlp") {
    const std::size_t bytes = 2ull << 30;
    const std::uint64_t pages = bytes / 4096;
    const std::size_t scratch_bytes = 512ull << 20;
    auto chain = make_random_cycle(pages, rng);
    std::uint32_t *dramd = nullptr, *permd = nullptr, *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&dramd, bytes));
    CUDA_CHECK(cudaMalloc(&permd, pages * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cudaMemcpy(permd, chain.data(), pages * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    init_chain_kernel<<<(unsigned)((pages + 255) / 256), 256>>>(dramd, permd, pages);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const int mlp_hops = 16;
    auto evict = [&]() {
      CUDA_CHECK(cudaMemset(scratch, 0xA5, scratch_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto launch = [&](std::uint32_t start) {
      switch (chains) {
        case 1: memlat_mlp_t_kernel<1><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        case 2: memlat_mlp_t_kernel<2><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        case 4: memlat_mlp_t_kernel<4><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        case 8: memlat_mlp_t_kernel<8><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        case 16: memlat_mlp_t_kernel<16><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        case 32: memlat_mlp_t_kernel<32><<<1, 1>>>(dramd, d, mlp_hops, start); break;
        default: return cudaErrorInvalidValue;
      }
      cudaError_t x = cudaGetLastError();
      if (x != cudaSuccess) return x;
      return cudaDeviceSynchronize();
    };
    evict();
    for (int i = 0; i < warm; ++i) launch(0);
    for (int s = 0; s < samples; ++s) {
      evict();
      cudaError_t e = launch(static_cast<std::uint32_t>(rng() % (pages / 8)));
      if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
      CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
      raw.push_back(h[0]);
      per_hop.push_back(static_cast<double>(h[0]) / mlp_hops);
    }
    cudaFree(dramd); cudaFree(permd); cudaFree(scratch);
    CurvePoint p;
    p.family = "stage27_memlat.dram.mlp";
    p.quantity = "latency";
    p.x_name = "mlp_chains";
    p.x = chains;
    p.y_name = "cycles_per_hop_per_chain";
    p.samples = per_hop;
    char b[2048];
    std::snprintf(b, sizeof(b),
                  "\"raw_elapsed_cycles\":%s,\"hops\":%d,\"working_set_bytes\":%zu,"
                  "\"evict\":\"512MiB memset per sample\",\"policy\":\"cg\","
                  "\"source\":\"stage27.memlat.measured\",\"gpu_name\":\"%s\","
                  "\"sm_clock_mhz\":%d,\"clock_locked\":false",
                  arr_ll(raw).c_str(), mlp_hops, bytes, dev.name.c_str(),
                  dev.sm_clock_mhz);
    p.extra_json = b;
    if (!err.empty()) p.error = err;
    append_jsonl(out, p);
    return err.empty() ? 0 : 1;
  }

  if (mode == "l2" || mode == "l1") {
    const bool l1 = mode == "l1";
    const std::size_t bytes = l1 ? (8ull << 10) : (2ull << 20);
    // l2: 512 chain entries at 4KiB word-stride (buf[i*1024]) covering the
    // full 2MiB; l1: 2048 word-granular (4B) entries covering 8KiB.
    const std::uint64_t entries = l1 ? bytes / 4 : bytes / 4096;
    auto chain = make_random_cycle(entries, rng);
    std::uint32_t *bufd = nullptr, *permd = nullptr;
    CUDA_CHECK(cudaMalloc(&bufd, bytes));
    CUDA_CHECK(cudaMalloc(&permd, entries * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(permd, chain.data(), entries * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    if (l1) {
      // word i holds the next word index
      std::vector<std::uint32_t> host(entries, 0u);
      for (std::uint64_t i = 0; i < entries; ++i) host[i] = chain[i];
      CUDA_CHECK(cudaMemcpy(bufd, host.data(), bytes, cudaMemcpyHostToDevice));
    } else {
      init_chain_kernel<<<(unsigned)((entries + 255) / 256), 256>>>(bufd, permd,
                                                                    entries);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    const int warm_hops = (int)entries;  // full permutation walk
    const int hops = 128;
    auto launch = [&](std::uint32_t start) {
      memlat_warmchase_kernel<<<1, 1>>>(bufd, d, warm_hops, hops, l1 ? 1 : 0, start);
      cudaError_t x = cudaGetLastError();
      if (x != cudaSuccess) return x;
      return cudaDeviceSynchronize();
    };
    for (int i = 0; i < warm; ++i) launch(0);
    for (int s = 0; s < samples; ++s) {
      cudaError_t e = launch(static_cast<std::uint32_t>(rng() % entries));
      if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
      CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
      raw.push_back(h[0]);
      per_hop.push_back(static_cast<double>(h[0]) / hops);
    }
    cudaFree(bufd); cudaFree(permd);
    CurvePoint p;
    p.family = l1 ? "stage27_memlat.l1.chain" : "stage27_memlat.l2.chain";
    p.quantity = "latency";
    p.x_name = "mlp_chains";
    p.x = 1;
    p.y_name = "cycles_per_hop";
    p.samples = per_hop;
    char b[2048];
    std::snprintf(b, sizeof(b),
                  "\"raw_elapsed_cycles\":%s,\"hops\":%d,\"warm_hops\":%d,"
                  "\"working_set_bytes\":%zu,\"policy\":\"%s\","
                  "\"residency\":\"full in-kernel warm walk of the permutation\","
                  "\"source\":\"stage27.memlat.measured\",\"gpu_name\":\"%s\","
                  "\"sm_clock_mhz\":%d,\"clock_locked\":false",
                  arr_ll(raw).c_str(), hops, warm_hops, bytes, l1 ? "ca" : "cg",
                  dev.name.c_str(), dev.sm_clock_mhz);
    p.extra_json = b;
    if (!err.empty()) p.error = err;
    append_jsonl(out, p);
    return err.empty() ? 0 : 1;
  }

  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
