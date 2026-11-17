// load_after_copies probe (protocol.json): does an L1-hit load wait
// behind async copies (cp.async / LDGSTS) the SM has issued but whose
// data has not returned?
// One CTA, 2 warps.  Per repetition: 8 victim lines loaded twice (L1
// resident, all lanes of a load on one word); then K async copies of
// 16 B per lane (512 B per instruction, fresh 512-B source rows of a
// 24 MiB L2-resident table, destination a 1024-B-aligned shared-memory
// ring, one 512-B slot per copy) issued by warp 0 (MODE 0, same_warp)
// or warp 1 (MODE 1, other_warp: warp 0 waits on a barrier until warp 1
// has issued them); then warp 0 times 8 dependent victim loads between
// two clock64 reads.  No cp.async wait before the chain; afterwards the
// issuing warp waits for all copies and the CTA syncs.
// A second victim kind (k_lacl) reads a per-thread volatile local array
// (SASS LDL) instead of global victims.
#include "../../tma_cold/mb.cuh"
#include <algorithm>
#include <string>
#include <vector>

__device__ __forceinline__ std::uint32_t ldg(const unsigned char* p) {
  std::uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

enum { FORM_CA = 0, FORM_CG = 1 };
constexpr int kKMax = 32;
constexpr int kSlot = 512;                     // bytes per copy
__shared__ __align__(1024) unsigned char kRing[kKMax * kSlot];

template <int FORM>
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src) {
  if (FORM == FORM_CA)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 ::"r"(dst), "l"(src));
  else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 ::"r"(dst), "l"(src));
}

template <int FORM, int MODE>
__global__ void __launch_bounds__(64, 1) k_lac(
    const unsigned char* victims, const unsigned char* table, int K,
    int reps, std::uint64_t* out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned saddr =
      static_cast<unsigned>(__cvta_generic_to_shared(kRing));
  // disjoint source regions per warp; rows fresh inside a launch
  const unsigned char* wtab = table + warp * (12u << 20);
  std::uint32_t sink = 0;
  for (int rep = 0; rep < reps; ++rep) {
    if (warp == 0)
      for (int pass = 0; pass < 2; ++pass)
        for (int k = 0; k < 8; ++k) sink ^= ldg(victims + k * 4096);
    __syncthreads();
    if (warp == MODE)
      for (int k = 0; k < K; ++k)
        cp_async16<FORM>(saddr + k * kSlot + lane * 16,
                         wtab + (rep * 64 + k) * kSlot + lane * 16);
    if (MODE == 1) __syncthreads();
    if (warp == 0) {
      std::uint32_t off = sink & 0u;
      const std::uint64_t t0 = mb_clock();
      for (int k = 0; k < 8; ++k) off += ldg(victims + k * 4096 + off);
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(off) : "memory");
      out[2 * rep] = t1 - t0;              // all lanes, same value:
      out[2 * rep + 1] = off + sink;       // no divergent store branch
    }
    if (warp == MODE) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
  }
}

// local victim: 8 dependent reads of a per-thread volatile stack array
// (SASS must lower to LDL)
template <int FORM>
__global__ void __launch_bounds__(64, 1) k_lacl(
    const unsigned char* table, int K, int reps, std::uint64_t* out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned saddr =
      static_cast<unsigned>(__cvta_generic_to_shared(kRing));
  const unsigned char* wtab = table + warp * (12u << 20);
  volatile unsigned loc[64];
  for (int i = 0; i < 64; ++i) loc[i] = (i + 13) & 63;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (warp == 0)
      for (int k = 0; k < K; ++k)
        cp_async16<FORM>(saddr + k * kSlot + lane * 16,
                         wtab + (rep * 64 + k) * kSlot + lane * 16);
    if (warp == 0) {
      unsigned off = (lane + rep) & 63;
      const std::uint64_t t0 = mb_clock();
      unsigned v = loc[off];
      v = loc[v]; v = loc[v]; v = loc[v]; v = loc[v];
      v = loc[v]; v = loc[v]; v = loc[v];
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(v) : "memory");
      out[2 * rep] = t1 - t0;
      out[2 * rep + 1] = v;
    }
    if (warp == 0) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
  }
}

// ---- protocol v2: k_lacl2<N, TW, FORM>, local victim only ----
// N = chain length (unrolled; read i is 128 B after read i-1 so the n
// words touch n distinct 128-B stack lines; per-thread array is 1 KiB).
// TW = 1: the issuing warp waits for the copies (commit + wait_group 0)
// before the first clock read; TW = 0: chain opens right after issue.
// mode / stl are run-time: mode 1 = warp 1 issues, warp 0 waits on a
// barrier; stl 1 = the chain words are rewritten (STL) before the copies.
template <int N, int TW, int FORM>
__global__ void __launch_bounds__(64, 1) k_lacl2(
    const unsigned char* table, int K, int reps, std::uint64_t* out,
    int mode, int stl) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned saddr =
      static_cast<unsigned>(__cvta_generic_to_shared(kRing));
  const unsigned char* wtab = table + warp * (12u << 20);
  volatile unsigned loc[256];              // 1 KiB per-thread stack array
  for (int i = 0; i < 256; ++i) loc[i] = i;
  for (int rep = 0; rep < reps; ++rep) {
    const unsigned base = (lane + rep * 7) & 255;
    if (stl && warp == 0)
#pragma unroll
      for (int i = 0; i < N; ++i)
        loc[(base + i * 32) & 255] = (base + i * 32) & 255;
    __syncthreads();
    if (warp == mode)
      for (int k = 0; k < K; ++k)
        cp_async16<FORM>(saddr + k * kSlot + lane * 16,
                         wtab + (rep * 64 + k) * kSlot + lane * 16);
    if (warp == mode && TW) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    if (mode == 1) __syncthreads();
    if (warp == 0) {
      unsigned v = base;
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int i = 0; i < N; ++i) v = loc[(v + 32) & 255];
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(v) : "memory");
      out[2 * rep] = t1 - t0;
      out[2 * rep + 1] = v;
    }
    if (warp == mode && !TW) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
  }
}

// ---- protocol v3: k_lacl3<N, TW, FORM, A> ----
// Lane-uniform stack chain: per-thread volatile array of A words,
// loc[i] = (i + S) % A identical in every thread (S = 13 for A = 64,
// 37 for A = 256, coprime so consecutive reads hit different words);
// chain v = loc[v] starts at (rep * 7) % A - no lane term, so each LDL
// reads one frame word across the 32 lanes. --smem adds dynamic shared
// memory (touched once before the first window); the 16-KiB copy ring
// stays static.
template <int N, int TW, int FORM, int A>
__global__ void __launch_bounds__(64, 1) k_lacl3(
    const unsigned char* table, int K, int reps, std::uint64_t* out,
    int mode, int smem) {
  extern __shared__ unsigned char dsm[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (smem) {
    dsm[0] = 0;
    dsm[smem - 1] = 0;
  }
  const unsigned saddr =
      static_cast<unsigned>(__cvta_generic_to_shared(kRing));
  const unsigned char* wtab = table + warp * (12u << 20);
  const int S = A == 64 ? 13 : 37;
  volatile unsigned loc[A];
  for (int i = 0; i < A; ++i) loc[i] = (i + S) % A;
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    if (warp == mode)
      for (int k = 0; k < K; ++k)
        cp_async16<FORM>(saddr + k * kSlot + lane * 16,
                         wtab + (rep * 64 + k) * kSlot + lane * 16);
    if (warp == mode && TW) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    if (mode == 1) __syncthreads();
    if (warp == 0) {
      unsigned v = (rep * 7) % A;
      const std::uint64_t t0 = mb_clock();
#pragma unroll
      for (int i = 0; i < N; ++i) v = loc[v];
      std::uint64_t t1;
      asm volatile("mov.u64 %0, %%clock64;"
                   : "=l"(t1) : "r"(v) : "memory");
      out[2 * rep] = t1 - t0;
      out[2 * rep + 1] = v;
    }
    if (warp == mode && !TW) {
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
  }
}

// prewarm: read the whole 24 MiB source table once so it is L2 resident
__global__ void k_prewarm(const unsigned char* table, unsigned* sink) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  unsigned v = 0;
  for (std::size_t o = i; o < (24u << 20); o += 4u * gridDim.x * blockDim.x)
    v ^= reinterpret_cast<const unsigned*>(table)[o / 4];
  if (v == 0xdeadbeefu) sink[0] = v;
}

int main(int argc, char** argv) {
  std::string outp = "load_after_copies.jsonl";
  bool v2 = false, v3 = false;
  int reps = 9, launches = 16;
  int c_n = -1, c_tw = 0, c_form = 0, c_mode = 0, c_stl = 0, c_k = 0;
  int c_a = 64, c_smem = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto iv = [&](int d) { return i + 1 < argc ? std::atoi(argv[++i]) : d; };
    if (a == "--out" && i + 1 < argc) outp = argv[++i];
    else if (a == "--v2") v2 = true;
    else if (a == "--v3") v3 = true;
    else if (a == "--words") c_a = iv(c_a);
    else if (a == "--smem") c_smem = iv(c_smem);
    else if (a == "--launches") launches = iv(launches);
    else if (a == "--reps") reps = iv(reps);
    else if (a == "--chain") c_n = iv(c_n);
    else if (a == "--timing") {
      c_tw = i + 1 < argc && std::string(argv[i + 1]) == "after_wait";
      if (i + 1 < argc) ++i;
    }
    else if (a == "--form") {
      c_form = i + 1 < argc && std::string(argv[i + 1]) == "cg";
      if (i + 1 < argc) ++i;
    }
    else if (a == "--issuer") {
      c_mode = i + 1 < argc && std::string(argv[i + 1]) == "other_warp";
      if (i + 1 < argc) ++i;
    }
    else if (a == "--stl") c_stl = iv(c_stl);
    else if (a == "--k" || a == "--copies") c_k = iv(c_k);
  }
  const MbDevice dev = mb_query_device();
  unsigned char *v = nullptr, *t = nullptr;
  unsigned* d_sink = nullptr;
  std::uint64_t* d_out = nullptr;
  MB_CHECK(cudaMalloc(&v, 1 << 16)); MB_CHECK(cudaMemset(v, 0, 1 << 16));
  MB_CHECK(cudaMalloc(&t, 24u << 20)); MB_CHECK(cudaMemset(t, 0x5a, 24u << 20));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned)));
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * 2 * reps));
  FILE* f = std::fopen(outp.c_str(), "a");
  if (!f) return 1;
  const char* xname[] = {"ca", "cg"};
  const char* mname[] = {"same_warp", "other_warp"};
  auto run_cell = [&](const std::string& variant, auto launch) {
    std::vector<double> s;
    for (int l = 0; l < launches; ++l) {
      k_prewarm<<<64, 256>>>(t, d_sink);
      launch();
      cudaError_t e = cudaGetLastError();
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                     cudaGetErrorString(e));
        break;
      }
      std::vector<std::uint64_t> h(2 * reps);
      MB_CHECK(cudaMemcpy(h.data(), d_out, sizeof(std::uint64_t) * 2 * reps,
                          cudaMemcpyDeviceToHost));
      if (l == 0) continue;                    // discarded first launch
      for (int r = 1; r < reps; ++r)
        s.push_back(static_cast<double>(h[2 * r]));
    }
    std::fprintf(f, "{\"schema\":\"cyclescope.microbench.load_after_copies.v1\","
                    "\"probe\":\"load_after_copies\",\"variant\":\"%s\","
                    "\"n\":%zu,\"samples\":[", variant.c_str(), s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
      std::fprintf(f, i ? ",%.1f" : "%.1f", s[i]);
    std::fprintf(f, "],\"gpu_name\":\"%s\",\"gpu_uuid\":\"%s\"}\n",
                 dev.name.c_str(), dev.uuid.c_str());
    std::fflush(f);
    std::fprintf(stderr, "%s: %zu samples\n", variant.c_str(), s.size());
  };
  const int Ks[] = {0, 1, 2, 4, 8, 16, 32};
  if (!v2 && !v3 && c_n < 0)
  for (int form = 0; form < 2; ++form)
    for (int mode = 0; mode < 2; ++mode)
      for (int K : Ks) {
        const std::string variant = std::string("global.") + xname[form] +
            "." + mname[mode] + ".k" + std::to_string(K);
        run_cell(variant, [&] {
          if (form == FORM_CA && mode == 0)
            k_lac<FORM_CA, 0><<<1, 64>>>(v, t, K, reps, d_out);
          else if (form == FORM_CA)
            k_lac<FORM_CA, 1><<<1, 64>>>(v, t, K, reps, d_out);
          else if (mode == 0)
            k_lac<FORM_CG, 0><<<1, 64>>>(v, t, K, reps, d_out);
          else
            k_lac<FORM_CG, 1><<<1, 64>>>(v, t, K, reps, d_out);
        });
      }
  for (int K : {0, 4, 16}) {
    if (v2 || v3 || c_n > 0) break;
    const std::string variant =
        "local.ca.same_warp.k" + std::to_string(K);
    run_cell(variant, [&] {
      k_lacl<FORM_CA><<<1, 64>>>(t, K, reps, d_out);
    });
  }

  if ((v2 || c_n > 0) && !v3) {
    using K2 = void (*)(const unsigned char*, int, int, std::uint64_t*,
                        int, int);
    auto pick2 = [](int n, int tw, int form) -> K2 {
      const int ni = n == 1 ? 0 : n == 2 ? 1 : n == 4 ? 2 : 3;
      switch (ni * 4 + tw * 2 + form) {
        case 0: return k_lacl2<1, 0, FORM_CA>;
        case 1: return k_lacl2<1, 0, FORM_CG>;
        case 2: return k_lacl2<1, 1, FORM_CA>;
        case 3: return k_lacl2<1, 1, FORM_CG>;
        case 4: return k_lacl2<2, 0, FORM_CA>;
        case 5: return k_lacl2<2, 0, FORM_CG>;
        case 6: return k_lacl2<2, 1, FORM_CA>;
        case 7: return k_lacl2<2, 1, FORM_CG>;
        case 8: return k_lacl2<4, 0, FORM_CA>;
        case 9: return k_lacl2<4, 0, FORM_CG>;
        case 10: return k_lacl2<4, 1, FORM_CA>;
        case 11: return k_lacl2<4, 1, FORM_CG>;
        case 12: return k_lacl2<8, 0, FORM_CA>;
        case 13: return k_lacl2<8, 0, FORM_CG>;
        case 14: return k_lacl2<8, 1, FORM_CA>;
        default: return k_lacl2<8, 1, FORM_CG>;
      }
    };
    struct V2Cell { int n, tw, form, mode, stl, k; };
    const V2Cell cells_all[] = {
      // (a) n x timing at K {0,4,16}, ca same_warp stl0
      {1,0,0,0,0,0},{1,0,0,0,0,4},{1,0,0,0,0,16},
      {1,1,0,0,0,0},{1,1,0,0,0,4},{1,1,0,0,0,16},
      {2,0,0,0,0,0},{2,0,0,0,0,4},{2,0,0,0,0,16},
      {2,1,0,0,0,0},{2,1,0,0,0,4},{2,1,0,0,0,16},
      {4,0,0,0,0,0},{4,0,0,0,0,4},{4,0,0,0,0,16},
      {4,1,0,0,0,0},{4,1,0,0,0,4},{4,1,0,0,0,16},
      {8,0,0,0,0,0},{8,0,0,0,0,4},{8,0,0,0,0,16},
      {8,1,0,0,0,0},{8,1,0,0,0,4},{8,1,0,0,0,16},
      // (b) K sweep at n=8, ca same_warp stl0, both timings
      {8,0,0,0,0,1},{8,0,0,0,0,2},{8,0,0,0,0,8},{8,0,0,0,0,32},
      {8,1,0,0,0,1},{8,1,0,0,0,2},{8,1,0,0,0,8},{8,1,0,0,0,32},
      // (c) cg same_warp and ca other_warp, n=8 now stl0
      {8,0,1,0,0,0},{8,0,1,0,0,4},{8,0,1,0,0,16},
      {8,0,0,1,0,0},{8,0,0,1,0,4},{8,0,0,1,0,16},
      // (d) stl 1, ca same_warp, both timings
      {8,0,0,0,1,0},{8,0,0,0,1,4},{8,0,0,0,1,16},
      {8,1,0,0,1,0},{8,1,0,0,1,4},{8,1,0,0,1,16},
    };
    const char* twname[] = {"now", "after_wait"};
    std::vector<V2Cell> cells(
        cells_all, cells_all + sizeof(cells_all) / sizeof(cells_all[0]));
    if (c_n > 0)
      cells.assign(1, V2Cell{c_n, c_tw, c_form, c_mode, c_stl, c_k});
    for (const auto& c : cells) {
      const std::string variant =
          "v2.n" + std::to_string(c.n) + "." + twname[c.tw] + "." +
          xname[c.form] + "." + mname[c.mode] + ".s" +
          std::to_string(c.stl) + ".k" + std::to_string(c.k);
      K2 kf = pick2(c.n, c.tw, c.form);
      run_cell(variant, [&] {
        kf<<<1, 64>>>(t, c.k, reps, d_out, c.mode, c.stl);
      });
    }
  }

  if (v3) {
    using K3 = void (*)(const unsigned char*, int, int, std::uint64_t*,
                        int, int);
    auto pick3 = [](int n, int tw, int form, int a) -> K3 {
      const int ni = n == 2 ? 0 : 1, ai = a == 64 ? 0 : 1;
      switch ((ni * 2 + tw) * 4 + form * 2 + ai) {
        case 0: return k_lacl3<2, 0, FORM_CA, 64>;
        case 1: return k_lacl3<2, 0, FORM_CA, 256>;
        case 2: return k_lacl3<2, 0, FORM_CG, 64>;
        case 3: return k_lacl3<2, 0, FORM_CG, 256>;
        case 4: return k_lacl3<2, 1, FORM_CA, 64>;
        case 5: return k_lacl3<2, 1, FORM_CA, 256>;
        case 6: return k_lacl3<2, 1, FORM_CG, 64>;
        case 7: return k_lacl3<2, 1, FORM_CG, 256>;
        case 8: return k_lacl3<8, 0, FORM_CA, 64>;
        case 9: return k_lacl3<8, 0, FORM_CA, 256>;
        case 10: return k_lacl3<8, 0, FORM_CG, 64>;
        case 11: return k_lacl3<8, 0, FORM_CG, 256>;
        case 12: return k_lacl3<8, 1, FORM_CA, 64>;
        case 13: return k_lacl3<8, 1, FORM_CA, 256>;
        case 14: return k_lacl3<8, 1, FORM_CG, 64>;
        default: return k_lacl3<8, 1, FORM_CG, 256>;
      }
    };
    struct V3Cell { int n, a, tw, form, mode, smem, k; };
    const V3Cell cells_all3[] = {
      // (a) K sweep x timing, ca same_warp n8 a64 m0
      {8,64,0,0,0,0,0},{8,64,0,0,0,0,1},{8,64,0,0,0,0,2},{8,64,0,0,0,0,4},
      {8,64,0,0,0,0,8},{8,64,0,0,0,0,16},{8,64,0,0,0,0,32},
      {8,64,1,0,0,0,0},{8,64,1,0,0,0,1},{8,64,1,0,0,0,2},{8,64,1,0,0,0,4},
      {8,64,1,0,0,0,8},{8,64,1,0,0,0,16},{8,64,1,0,0,0,32},
      // (b) cg same_warp / ca other_warp, n8 a64 now
      {8,64,0,1,0,0,0},{8,64,0,1,0,0,4},{8,64,0,1,0,0,16},
      {8,64,0,0,1,0,0},{8,64,0,0,1,0,4},{8,64,0,0,1,0,16},
      // (c) n2 a64 / a256 n8, now ca same_warp
      {2,64,0,0,0,0,0},{2,64,0,0,0,0,4},{2,64,0,0,0,0,16},
      {8,256,0,0,0,0,0},{8,256,0,0,0,0,4},{8,256,0,0,0,0,16},
      // (d) smem 96 KiB / 200 KiB, now ca same_warp
      {8,64,0,0,0,98304,0},{8,64,0,0,0,98304,4},{8,64,0,0,0,98304,16},
      {8,64,0,0,0,204800,0},{8,64,0,0,0,204800,4},{8,64,0,0,0,204800,16},
    };
    const char* twname[] = {"now", "after_wait"};
    std::vector<V3Cell> cells(
        cells_all3, cells_all3 + sizeof(cells_all3) / sizeof(cells_all3[0]));
    if (c_n > 0)
      cells.assign(1, V3Cell{c_n, c_a, c_tw, c_form, c_mode, c_smem, c_k});
    for (const auto& c : cells) {
      const std::string variant =
          "v3.n" + std::to_string(c.n) + ".a" + std::to_string(c.a) + "." +
          twname[c.tw] + "." + xname[c.form] + "." + mname[c.mode] + ".m" +
          std::to_string(c.smem) + ".k" + std::to_string(c.k);
      // uniform control: first valid rep chain indices, all 32 lanes
      {
        const int S = c.a == 64 ? 13 : 37;
        int v = (1 * 7) % c.a;
        std::string path = outp.substr(0, outp.find_last_of("/\\") + 1) +
            "uniform-" + variant + ".txt";
        FILE* uf = std::fopen(path.c_str(), "w");
        if (uf) {
          for (int lane = 0; lane < 32; ++lane) {
            int w = v;
            std::fprintf(uf, "lane %d:", lane);
            for (int i = 0; i < c.n; ++i) {
              std::fprintf(uf, " %d", w);
              w = (w + S) % c.a;
            }
            std::fprintf(uf, "\n");
          }
          std::fclose(uf);
        }
      }
      K3 kf = pick3(c.n, c.tw, c.form, c.a);
      if (c.smem > 0)
        cudaFuncSetAttribute(kf,
            cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem);
      run_cell(variant, [&] {
        kf<<<1, 64, c.smem>>>(t, c.k, reps, d_out, c.mode, c.smem);
      });
    }
  }
  std::fclose(f);
  return 0;
}
