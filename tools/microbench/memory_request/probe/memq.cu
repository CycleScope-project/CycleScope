// Memory request structure (memq_rules.py): per-level 1-sector request rate
// and in-flight sweep (memory.{l1,l2,dram}.issue_interval_cycles /
// outstanding_semantics), store ack path (memory.layout.store_path), and
// 1-sector store request rate to distinct vs the same sector
// (memory.store_l2.issue_interval_cycles, memory.store_merge.store_request_accept_interval).
//
//   ld.<lvl>.d<D>.w<W>   one CTA of W warps on one SM; every warp instruction
//       is one 32 B sector (lane address = sector + (lane & 7) * 4); each warp
//       issues K loads in groups of D independent loads, the group is consumed
//       before the next one is issued, so W * D requests are in flight.
//       requests/cycle/SM = W * K / slope over K (memq_rules.request_rate).
//     l1:   ld.global.ca, 16 KiB per-CTA region, warmed (L1 hits)
//     l2:   ld.global.cg, 4 MiB region, warmed (L2 hits, L1 bypassed)
//     dram: ld.global.cg, fresh region of an 8 GiB buffer per launch
//   st.<same|dist>.w<W>  K 1-sector st.global.u32 per warp between bar.sync's;
//       dist: every store a new sector of a 4 MiB L2-resident region;
//       same: every store of a warp to the same sector.
//   ack.<cold|warm>  lane 0 of one warp: t0 ; st.global.u32 [line] ; fence.sc.gpu ; t1
//       cold: a line of a fresh 8 GiB-buffer region (not in L2), then warm:
//       the same line again right after.  64 lines per launch.
#include "../mb.cuh"

#include <algorithm>
#include <string>
#include <vector>

template <int LVL>
__device__ __forceinline__ std::uint32_t ld1(const void* p) {
  std::uint32_t v;
  if constexpr (LVL == 0)
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// v2 dependent chase: the level's region is an array of 32 B sectors
// whose eight words all hold the next sector index, written here by
// init_chain.  Chain c (c = warp * D + d) starts at sector 4c; each
// step issues D loads whose values are the next step's addresses, so
// exactly W * D requests are in flight.
__global__ void init_chain(std::uint32_t* region, std::size_t n_sectors,
                           std::uint32_t stride, std::size_t slab_sectors) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x
      + threadIdx.x;
  if (g >= n_sectors) return;
  const std::size_t s = g % slab_sectors;
  const std::uint32_t nxt = static_cast<std::uint32_t>(
      (s + stride) % slab_sectors);
  std::uint32_t* w = region + g * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = nxt;
}

template <int LVL, int D, int K>
__global__ void __launch_bounds__(1024, 1) k_ld(const unsigned char* buf, std::size_t region_bytes,
                                                std::size_t fresh_offset, std::uint64_t* out, std::uint32_t* sink,
                                                int reps, std::size_t rep_stride) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::size_t lane_off = static_cast<std::size_t>(lane & 7) * 4;
  std::uint32_t acc = 0;
  if (LVL != 2) {
    // warm the region once: every sector touched -> all hits later
    const unsigned char* base = buf + fresh_offset;
    for (std::size_t o = lane_off; o < region_bytes; o += 32)
      acc ^= ld1<LVL>(base + o);
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    const unsigned char* base = buf + fresh_offset +
        static_cast<std::size_t>(rep) * rep_stride;
    const std::uint32_t n_sectors =
        static_cast<std::uint32_t>(region_bytes / 32);
    std::uint32_t idx[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
      // chain c starts at sector 4c (mod region sectors: the l1
      // region is only 512 sectors)
      idx[d] = static_cast<std::uint32_t>(4 * (warp * D + d)) % n_sectors;
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int d = 0; d < D; ++d)
        idx[d] = ld1<LVL>(base + static_cast<std::size_t>(idx[d]) * 32 +
                          lane_off);
    }
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
#pragma unroll
    for (int d = 0; d < D; ++d) acc ^= idx[d];
  }
  sink[threadIdx.x] = acc;
}

template <int SAME, int K>
__global__ void __launch_bounds__(1024, 1) k_st(unsigned char* buf, std::size_t region_bytes, std::uint64_t* out,
                                                std::uint32_t* sink, int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const std::size_t seg = region_bytes / static_cast<std::size_t>(nwarps);
  unsigned char* wbase = buf + static_cast<std::size_t>(warp) * seg + (lane & 7) * 4;
  const std::uint32_t val = threadIdx.x;
  // SAME: per-store offsets that are zero at run time but opaque to ptxas
  // (volatile shared loads), so it cannot merge stores to one address
  __shared__ std::uint32_t zero_s[64];
  if (threadIdx.x < 64) zero_s[threadIdx.x] = 0;
  __syncthreads();
  std::uint32_t z[SAME ? K : 1];
#pragma unroll
  for (int i = 0; i < (SAME ? K : 1); ++i)
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(z[i]) : "r"(mb_smem_u32(&zero_s[i])) : "memory");
  std::size_t off = 0;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    const std::uint64_t t0 = mb_clock();
#pragma unroll
    for (int i = 0; i < K; ++i) {
      unsigned char* p = SAME ? wbase + z[SAME ? i : 0] : wbase + ((off + i * 32) & (seg - 1));
      asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(val + i) : "memory");
    }
    off = (off + K * 32) & (seg - 1);
    __syncthreads();
    const std::uint64_t t1 = mb_clock();
    if (threadIdx.x == 0) out[rep] = t1 - t0;
  }
  if (threadIdx.x == 0) sink[0] = val;
}

// out[2 * line + {0,1}] = cold, warm ack cycles; kept in shared memory until
// the end so no timing store is pending when the next fence waits
__global__ void __launch_bounds__(32, 1) k_ack(unsigned char* lines, int nlines, std::uint64_t* out) {
  __shared__ std::uint64_t t[2 * 64];
  if (threadIdx.x != 0) return;
#pragma unroll 1
  for (int l = 0; l < nlines; ++l) {
    unsigned char* p = lines + static_cast<std::size_t>(l) * 4096;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const std::uint64_t t0 = mb_clock();
      asm volatile("st.global.u32 [%0], %1;\n\tfence.sc.gpu;" ::"l"(p), "r"(l + pass) : "memory");
      const std::uint64_t t1 = mb_clock();
      t[2 * l + pass] = t1 - t0;
    }
  }
  for (int i = 0; i < 2 * nlines; ++i) out[i] = t[i];
}

using LdKern = void (*)(const unsigned char*, std::size_t, std::size_t, std::uint64_t*, std::uint32_t*, int, std::size_t);
using StKern = void (*)(unsigned char*, std::size_t, std::uint64_t*, std::uint32_t*, int);
struct LdCase {
  int lvl, d, k;
  LdKern f;
};
struct StCase {
  int same, k;
  StKern f;
};

template <int LVL, int D>
void add_ld(std::vector<LdCase>& v) {
  // K 64/128/256: memq-107's K 16/32/64 windows failed the 1% linearity gate
  v.push_back({LVL, D, 64, &k_ld<LVL, D, 64>});
  v.push_back({LVL, D, 128, &k_ld<LVL, D, 128>});
  v.push_back({LVL, D, 256, &k_ld<LVL, D, 256>});
  if constexpr (LVL == 1) {  // L2: memq-109 still failed the linearity gate at D 4/8
    v.push_back({LVL, D, 512, &k_ld<LVL, D, 512>});
    v.push_back({LVL, D, 1024, &k_ld<LVL, D, 1024>});
  }
}
template <int LVL>
void add_lvl(std::vector<LdCase>& v) {
  add_ld<LVL, 1>(v);
  add_ld<LVL, 2>(v);
  add_ld<LVL, 4>(v);
  add_ld<LVL, 8>(v);
  add_ld<LVL, 16>(v);
  add_ld<LVL, 32>(v);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "memq");
  const std::size_t dram_bytes = std::size_t(1) << 33;  // 8 GiB
  unsigned char* buf = nullptr;
  MB_CHECK(cudaMalloc(&buf, dram_bytes));
  MB_CHECK(cudaMemset(buf, 1, dram_bytes));
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * std::max(args.reps, 256)));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 1024));
  const char* lname[] = {"l1", "l2", "dram"};

  std::vector<LdCase> lds;
  add_lvl<0>(lds);
  add_lvl<1>(lds);
  add_lvl<2>(lds);
  // in-flight series: W = 1..32 at D = 1, then W = 32 at D = 2..16
  const std::pair<int, int> wd[] = {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {16, 1}, {32, 1},
                                    {32, 2}, {32, 4}, {32, 8}, {32, 16},
                                    {32, 32}};
  std::size_t fresh = std::size_t(1) << 30;  // dram regions start past the warmed regions
  for (const auto& [wn, d] : wd) {
    for (const auto& c : lds) {
      if (c.d != d) continue;
      const std::string variant = std::string("ld.") + lname[c.lvl] + ".d" + std::to_string(d) + ".w" + std::to_string(wn);
      if (!mb_selected(args, variant)) continue;
      const std::size_t region = c.lvl == 0 ? (16u << 10) : c.lvl == 1 ? (4u << 20) : (std::size_t(256) << 20);
      std::vector<double> s;
      // l1 / l2 chain regions sit past the dram slab stretch
      // (7 x 1088 MiB); written once per level (init is idempotent)
      const std::size_t small_base =
          std::size_t(7) * (1088ull << 20);
      if (c.lvl != 2) {
        const std::size_t loff = small_base +
            (c.lvl == 0 ? 0 : (16ull << 10));
        init_chain<<<(unsigned)((region / 32 + 255) / 256), 256>>>(
            reinterpret_cast<std::uint32_t*>(buf + loff),
            region / 32, c.lvl == 0 ? 4 : 8192, region / 32);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
      }
      for (int l = 0; l < args.launches; ++l) {
        std::size_t off = 0, rep_stride = 0;
        if (c.lvl != 2)
          off = small_base + (c.lvl == 0 ? 0 : (16ull << 10));
        if (c.lvl == 2) {
          // v2: launch l walks slabs of its own 1088 MiB stretch
          // (17 reps x 64 MiB), chain layout written once per launch
          off = static_cast<std::size_t>(l) * (1088ull << 20);
          rep_stride = 64ull << 20;
          const std::size_t slab_sec = rep_stride / 32;
          init_chain<<<(unsigned)((17 * slab_sec + 255) / 256), 256>>>(
              reinterpret_cast<std::uint32_t*>(buf + off),
              17 * slab_sec, 8192, slab_sec);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
        }
        c.f<<<1, 32 * wn>>>(buf, region, off, d_out, d_sink, args.reps,
                            rep_stride);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(args.reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) s.push_back(static_cast<double>(h[r]));
      }
      w.row(variant, c.k,
            "\"level\":\"" + std::string(lname[c.lvl]) + "\",\"d\":" + std::to_string(d) + ",\"warps\":" +
                std::to_string(wn) + ",\"inflight\":" + std::to_string(wn * d) + ",\"sectors_per_request\":1",
            s);
    }
  }

  const StCase sts[] = {{0, 16, &k_st<0, 16>}, {0, 32, &k_st<0, 32>}, {0, 64, &k_st<0, 64>},
                        {1, 16, &k_st<1, 16>}, {1, 32, &k_st<1, 32>}, {1, 64, &k_st<1, 64>}};
  for (int wn : {1, 4, 16, 32}) {
    for (const auto& c : sts) {
      const std::string variant = std::string("st.") + (c.same ? "same" : "dist") + ".w" + std::to_string(wn);
      if (!mb_selected(args, variant)) continue;
      std::vector<double> s;
      for (int l = 0; l < args.launches; ++l) {
        c.f<<<1, 32 * wn>>>(buf, std::size_t(4) << 20, d_out, d_sink, args.reps);
        MB_CHECK(cudaGetLastError());
        MB_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> h(args.reps);
        MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
        for (int r = 1; r < args.reps; ++r) s.push_back(static_cast<double>(h[r]));
      }
      w.row(variant, c.k, "\"same\":" + std::to_string(c.same) + ",\"warps\":" + std::to_string(wn) +
                              ",\"sectors_per_request\":1", s);
    }
  }

  if (mb_selected(args, "ack.cold") || mb_selected(args, "ack.warm")) {
    constexpr int kLines = 64;
    std::vector<double> cold, warm;
    for (int l = 0; l < args.launches; ++l) {
      if (fresh + kLines * 4096 > dram_bytes) fresh = std::size_t(1) << 30;
      k_ack<<<1, 32>>>(buf + fresh, kLines, d_out);
      fresh += std::size_t(64) << 20;  // next launch's lines far away (evicted from L2 by the ld.dram walk)
      MB_CHECK(cudaGetLastError());
      MB_CHECK(cudaDeviceSynchronize());
      std::vector<std::uint64_t> h(2 * kLines);
      MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int i = 0; i < kLines; ++i) {
        cold.push_back(static_cast<double>(h[2 * i]));
        warm.push_back(static_cast<double>(h[2 * i + 1]));
      }
    }
    w.row("ack.cold", 0, "\"lines\":64", cold);
    w.row("ack.warm", 0, "\"lines\":64", warm);
  }
  return w.failures ? 1 : 0;
}
