// Stage27 MemLat probe: per-opcode dependent issue->issue latency on B300S.
//
// Target (shopping list P0-2): instruction.service per-opcode result_latency
// (installed flat =1; H100 exact241 per-opcode 1..64, median 5, all "source").
//
// Measured quantity: a warp of 32 threads runs `links` back-to-back dependent
// ops (op i consumes op i-1's destination); thread0 clocks each round with
// clock64 brackets; value = elapsed/links.  A depth sweep (4 and 16) checks
// the constant is depth-independent (no per-link clock pollution).
//
// Per-op dedicated kernels so the SASS gate can verify the target mnemonic
// per function (stage27_sass_check convention).  An op whose SASS fails the
// gate never reaches the GPU.
//
// Notes on specific ops:
//   sel   : selp chain in one asm block; the enabling setp is issued once per
//           round (amortized over 16 links) so the chain is pure SEL.
//   isetp : setp+selp PAIR per link (SETP has no data destination to chain
//           through); reported per pair; ISETP derived as pair - sel.
//   mufu  : rcp.approx (MUFU.RCP) chain under instruction.service; EX2 has
//           its own leaf (instruction.sfu.ex2) and is measured separately.
//   ex2   : ex2.approx chain; values saturate to inf, latency unaffected.
//   ldg   : L2-resident chain (context only -- the engine already charges the
//           data path to dram/l2 service leaves; LDG result_latency stays 1).
//   lds   : shared address chain (context only, same reasoning).
#include "common.cuh"
#include "curves_io.cuh"

#include <numeric>
#include <random>

__global__ void memlat_init_chain_kernel(std::uint32_t* buf, const std::uint32_t* perm,
                                         std::uint64_t npages) {
  const std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
  if (i < npages) buf[i * 1024] = perm[i];
}

// One dependent-op link per case.  LINK<name> must keep the destination of the
// previous link feeding the next (the SASS gate + depth sweep enforce this).
#define LINK_FADD asm volatile("add.f32 %0, %0, %1;" : "+f"(fx) : "f"(fy));
#define LINK_FMUL asm volatile("mul.f32 %0, %0, %1;" : "+f"(fx) : "f"(fy));
#define LINK_FFMA asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fx) : "f"(fy), "f"(fz));
#define LINK_IMAD asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
#define LINK_LOP3 asm volatile("lop3.b32 %0, %0, %1, %2, 0xEA;" : "+r"(x) : "r"(y), "r"(z));
#define LINK_SHF asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(y));
#define LINK_MOV asm volatile("mov.b32 %0, %0;" : "+r"(x));
#define LINK_MUFU asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fx));
#define LINK_EX2 asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fx));
#define LINK_PRMT asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(x) : "r"(y));
#define LINK_IADD3 asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(y));

// 16 links per LINK_* expansion block (keeps the asm blocks modest); the
// runner issues ceil(depth/16) blocks per timed round and divides by
// (blocks*16).
#define LINKS16(macro) \
  macro macro macro macro macro macro macro macro macro macro macro macro macro macro macro macro

#define DEFINE_OP_KERNEL(NAME, LINK, IS_FLOAT)                                 \
  __global__ void memlat_op_##NAME##_kernel(long long* out, int depth,         \
                                            int rounds) {                      \
    const int tid = static_cast<int>(threadIdx.x);                             \
    std::uint32_t x = 0x9E3779B9u * static_cast<std::uint32_t>(tid + 1);       \
    std::uint32_t y = 0x85EBCA6Bu * static_cast<std::uint32_t>(tid + 3);       \
    std::uint32_t z = 0x1234567u * static_cast<std::uint32_t>(tid + 5);        \
    float fx = 1.0f + 0.001f * static_cast<float>(tid);                        \
    const float fy = 1.0000001f;                                               \
    const float fz = 0.9999999f;                                               \
    (void)x; (void)y; (void)z; (void)fx;                                       \
    _Pragma("unroll 1") for (int i = 0; i < 32; ++i) { LINK }                  \
    _Pragma("unroll 1") for (int r = 0; r < rounds; ++r) {                     \
      __syncwarp();                                                            \
      const std::uint64_t t0 = rdclk64();                                      \
      _Pragma("unroll 1") for (int i = 0; i < depth / 16; ++i) {               \
        LINKS16(LINK)                                                          \
      }                                                                        \
      if (depth % 16) {                                                        \
        _Pragma("unroll") for (int i = 0; i < (depth % 16); ++i) { LINK }      \
      }                                                                        \
      __syncwarp();                                                            \
      const std::uint64_t t1 = rdclk64();                                      \
      if (tid == 0) out[r] = static_cast<long long>(t1 - t0);                  \
    }                                                                          \
    if (IS_FLOAT ? (fx == 42.0f) : (x == 0xDEADBEEFu)) out[rounds] = 1;        \
  }

DEFINE_OP_KERNEL(fadd, LINK_FADD, 1)
DEFINE_OP_KERNEL(fmul, LINK_FMUL, 1)
DEFINE_OP_KERNEL(ffma, LINK_FFMA, 1)
DEFINE_OP_KERNEL(imad, LINK_IMAD, 0)
DEFINE_OP_KERNEL(lop3, LINK_LOP3, 0)
DEFINE_OP_KERNEL(shf, LINK_SHF, 0)
DEFINE_OP_KERNEL(mov, LINK_MOV, 0)
DEFINE_OP_KERNEL(mufu, LINK_MUFU, 1)
DEFINE_OP_KERNEL(ex2, LINK_EX2, 1)
DEFINE_OP_KERNEL(prmt, LINK_PRMT, 0)
DEFINE_OP_KERNEL(iadd3, LINK_IADD3, 0)

// sel: pure selp chain; enabling setp once per round inside the same block.
__global__ void memlat_op_sel_kernel(long long* out, int depth, int rounds) {
  const int tid = static_cast<int>(threadIdx.x);
  std::uint32_t x = 0x9E3779B9u * static_cast<std::uint32_t>(tid + 1);
  const std::uint32_t y = 0x85EBCA6Bu;
  const std::uint32_t z = 0x1234567u * static_cast<std::uint32_t>(tid + 5);
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    __syncwarp();
    const std::uint64_t t0 = rdclk64();
    asm volatile("{ .reg .pred p;\n"
                 "  setp.ne.b32 p, %1, 0;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "  selp.b32 %0, %0, %2, p;\n"
                 "}"
                 : "+r"(x) : "r"(z), "r"(y));
    __syncwarp();
    const std::uint64_t t1 = rdclk64();
    if (tid == 0) out[r] = static_cast<long long>(t1 - t0);
  }
  if (x == 0xDEADBEEFu) out[rounds] = 1;
}

// isetp: setp+selp pair per link (SETP result is a predicate; the SEL is the
// minimal consumer that carries the dependency to the next link).
__global__ void memlat_op_isetp_kernel(long long* out, int depth, int rounds) {
  const int tid = static_cast<int>(threadIdx.x);
  std::uint32_t x = 0x9E3779B9u * static_cast<std::uint32_t>(tid + 1);
  const std::uint32_t y = 0x85EBCA6Bu;
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    __syncwarp();
    const std::uint64_t t0 = rdclk64();
    asm volatile("{ .reg .pred p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "  setp.lt.u32 p, %0, %1;\n"
                 "  selp.b32 %0, %0, %1, p;\n"
                 "}"
                 : "+r"(x) : "r"(y));
    __syncwarp();
    const std::uint64_t t1 = rdclk64();
    if (tid == 0) out[r] = static_cast<long long>(t1 - t0);
  }
  if (x == 0xDEADBEEFu) out[rounds] = 1;
}

// ldg: L2-resident dependent chase (context measurement).
__global__ void memlat_op_ldg_kernel(const std::uint32_t* p, long long* out, int depth,
                                     int rounds, std::uint32_t start) {
  std::uint32_t idx = start;
#pragma unroll 1
  for (int i = 0; i < 512; ++i) {  // full warm walk of the 2MiB permutation
    asm volatile("ld.global.cg.u32 %0, [%1];"
                 : "=r"(idx) : "l"(p + (std::uint64_t)idx * 1024) : "memory");
  }
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    const std::uint64_t t0 = rdclk64();
#pragma unroll 1
    for (int i = 0; i < depth; ++i) {
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(idx) : "l"(p + (std::uint64_t)idx * 1024) : "memory");
    }
    const std::uint64_t t1 = rdclk64();
    if (threadIdx.x == 0) out[r] = static_cast<long long>(t1 - t0);
  }
  if (idx == 0xDEADBEEFu) out[rounds] = 1;
}

// lds: shared address chain, no bank conflict (context measurement).
__global__ void memlat_op_lds_kernel(long long* out, int depth, int rounds) {
  __shared__ std::uint32_t sm[4096];
  const int tid = static_cast<int>(threadIdx.x);
  const int off0 = tid, off1 = tid + 32;  // 32 distinct banks
  sm[off0] = smem_u32(&sm[off1]);
  sm[off1] = smem_u32(&sm[off0]);
  __syncthreads();
  std::uint32_t addr = smem_u32(&sm[off0]);
#pragma unroll 1
  for (int i = 0; i < 64; ++i) {
    asm volatile("ld.shared.b32 %0, [%0];" : "+r"(addr) ::"memory");
  }
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    __syncwarp();
    const std::uint64_t t0 = rdclk64();
#pragma unroll 1
    for (int i = 0; i < depth; ++i) {
      asm volatile("ld.shared.b32 %0, [%0];" : "+r"(addr) ::"memory");
    }
    __syncwarp();
    const std::uint64_t t1 = rdclk64();
    if (tid == 0) out[r] = static_cast<long long>(t1 - t0);
  }
  if (addr == 0xDEADBEEFu) out[rounds] = 1;
}

static const char* kOpNames[] = {"fadd", "fmul", "ffma", "imad",  "lop3", "shf",
                                 "sel",  "isetp", "mov",  "mufu",  "ex2",  "prmt",
                                 "iadd3", "ldg",  "lds"};

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
  std::string op = "fadd";
  const char* out = "stage27_memlat_ops.jsonl";
  int depth = 16, rounds = 8, launches = 8;
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
    if (a == "--op") nexts(&op);
    else if (a == "--depth") nexti(&depth);
    else if (a == "--rounds") nexti(&rounds);
    else if (a == "--launches") nexti(&launches);
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else return 2;
  }
  int op_id = -1;
  for (int i = 0; i < 15; ++i)
    if (op == kOpNames[i]) op_id = i;
  if (op_id < 0) {
    std::fprintf(stderr, "unknown op %s\n", op.c_str());
    return 2;
  }
  // sel is a fixed 16-link asm block; isetp is a fixed block of 16 (setp+selp)
  // pairs.  Normalize division accordingly (derivation in extra_json).
  int links = depth;
  if (op == "sel") links = 16;
  if (op == "isetp") links = 16;
  const DeviceInfo dev = query_device();

  long long* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, (rounds + 1) * sizeof(long long)));
  std::uint32_t* ldg_buf = nullptr;
  std::mt19937_64 rng(0xB3006u);
  std::uint32_t start = 0;
  if (op == "ldg") {
    // 2MiB random page permutation, single cycle (same build as memory probe)
    const std::size_t bytes = 2ull << 20;
    const std::uint64_t pages = bytes / 4096;
    std::vector<std::uint32_t> order(pages);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::uint32_t> perm(pages);
    for (std::uint64_t k = 0; k < pages; ++k) perm[order[k]] = order[(k + 1) % pages];
    std::uint32_t* permd = nullptr;
    CUDA_CHECK(cudaMalloc(&ldg_buf, bytes));
    CUDA_CHECK(cudaMalloc(&permd, pages * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(permd, perm.data(), pages * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    memlat_init_chain_kernel<<<(unsigned)((pages + 255) / 256), 256>>>(ldg_buf, permd,
                                                                       pages);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(permd);
  }
  auto launch = [&]() {
    if (op == "sel") {
      memlat_op_sel_kernel<<<1, 32>>>(d, links, rounds);
    } else if (op == "isetp") {
      memlat_op_isetp_kernel<<<1, 32>>>(d, links, rounds);
    } else if (op == "lds") {
      memlat_op_lds_kernel<<<1, 32>>>(d, links, rounds);
    } else if (op == "ldg") {
      start = static_cast<std::uint32_t>(rng() % 512);
      memlat_op_ldg_kernel<<<1, 1>>>(ldg_buf, d, links, rounds, start);
    } else {
      switch (op_id) {
        case 0: memlat_op_fadd_kernel<<<1, 32>>>(d, links, rounds); break;
        case 1: memlat_op_fmul_kernel<<<1, 32>>>(d, links, rounds); break;
        case 2: memlat_op_ffma_kernel<<<1, 32>>>(d, links, rounds); break;
        case 3: memlat_op_imad_kernel<<<1, 32>>>(d, links, rounds); break;
        case 4: memlat_op_lop3_kernel<<<1, 32>>>(d, links, rounds); break;
        case 5: memlat_op_shf_kernel<<<1, 32>>>(d, links, rounds); break;
        case 8: memlat_op_mov_kernel<<<1, 32>>>(d, links, rounds); break;
        case 9: memlat_op_mufu_kernel<<<1, 32>>>(d, links, rounds); break;
        case 10: memlat_op_ex2_kernel<<<1, 32>>>(d, links, rounds); break;
        case 11: memlat_op_prmt_kernel<<<1, 32>>>(d, links, rounds); break;
        case 12: memlat_op_iadd3_kernel<<<1, 32>>>(d, links, rounds); break;
        default: return cudaErrorInvalidDeviceFunction;
      }
    }
    cudaError_t x = cudaGetLastError();
    if (x != cudaSuccess) return x;
    return cudaDeviceSynchronize();
  };
  for (int i = 0; i < 2; ++i) launch();
  std::vector<double> per_link;
  std::vector<long long> raw;
  std::string err;
  for (int s = 0; s < launches; ++s) {
    cudaError_t e = launch();
    if (e != cudaSuccess) { err = cudaGetErrorString(e); break; }
    std::vector<long long> h(rounds + 1, 0);
    CUDA_CHECK(cudaMemcpy(h.data(), d, (rounds + 1) * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    for (int r = 0; r < rounds; ++r) {
      raw.push_back(h[r]);
      per_link.push_back(static_cast<double>(h[r]) / links);
    }
  }
  if (ldg_buf) cudaFree(ldg_buf);
  cudaFree(d);
  if (!err.empty()) {
    std::fprintf(stderr, "launch error: %s\n", err.c_str());
    return 1;
  }
  char fam[96];
  std::snprintf(fam, sizeof(fam), "stage27_memlat.ops.%s", op.c_str());
  CurvePoint p;
  p.family = fam;
  p.quantity = op == "ldg" || op == "lds" ? "context_chain" : "result_latency_chain";
  p.x_name = "links";
  p.x = links;
  p.y_name = "cycles_per_link";
  p.samples = per_link;
  char b[2048];
  std::snprintf(b, sizeof(b),
                "\"raw_elapsed_cycles\":%s,\"threads\":32,\"links\":%d,\"depth_arg\":%d,"
                "\"op_sass_target\":\"%s\",\"derivation_note\":\"%s\","
                "\"source\":\"stage27.memlat.measured\",\"gpu_name\":\"%s\","
                "\"sm_clock_mhz\":%d,\"clock_locked\":false",
                arr_ll(raw).c_str(), links, depth, op.c_str(),
                op == "isetp" ? "pair setp+selp per link; ISETP = pair - sel"
                              : (op == "sel" ? "selp chain, setp amortized per round"
                                             : "direct dependent chain"),
                dev.name.c_str(), dev.sm_clock_mhz);
  p.extra_json = b;
  append_jsonl(out, p);
  return 0;
}
