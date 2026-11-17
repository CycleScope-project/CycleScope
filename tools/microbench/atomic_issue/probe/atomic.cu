// storage.atomic.issue (atomic_issue protocol): per-SM atomic unit
// admission interval of one warp-wide atomic instruction, linear in
// atomic_same_address_lanes L.  One CTA of W warps (W = 1: SMSP 0 only;
// W = 4: one per SMSP); every warp: bar.sync ; t_s ; U independent atomics
// ; t_e ; bar.sync.  out[rep * 16 + warp] = t_e - t_s (lane 0 of each warp).
//
// OP: 0 red.global.add.u32 (REDG / RED.E.ADD), 1 atom.global.add.u32 with
// the returned values only consumed after t_e (ATOMG.E.ADD),
// 2 atom.shared.add.u32 (ATOMS.ADD), 3 red.shared.add.u32 (ATOMS / REDS
// form as compiled), 4 generic atom.add.u32 on a global pointer
// (ATOM.E.ADD).
//
// Addresses: lanes grouped in blocks of L consecutive lanes; block b
// targets base + b x 128 B (global: one L2-resident line per block, warmed
// before the window; each warp owns a 4 KiB region) or sbase + b x 4 B
// (shared: distinct banks within the warp's 128 B region).  All U atomics
// of a rep use the same per-lane address.  Variant atomic.<op>.L<L>.W<W>,
// u = U.
//
// r2 addendum (IND = 1, global ops only): instruction i of the U targets
// base + warp x 512 KiB + i x 4 KiB + b x 128 B, so no two instructions of
// a rep touch the same line (no same-line follow-on chain at L2); the
// whole 2 MiB is warmed into L2 before the window.  U in {16, 32, 64,
// 128}.  Variant atomic.<op>.L<L>.W<W>.indep.
//
// r3 addendum (IND = 1, block stride S bytes instead of 128): block b
// targets + b x S, so L = 1 with S in {4, 8, 16, 32, 64} puts the 32 lanes
// on S / 4 distinct lines (coalesced at S = 4) and S = 4 with L > 1 puts
// 32 / L distinct words in one line.  Variant atomic.<op>.L<L>.W<W>.S<S>
// (S = 128 is the .indep variant).
//
// r4 addendum: W = 16 (four warps per SMSP, 512 threads) for the
// independent-line cells with U in {16, 32, 64} (the register file does
// not hold 128 returned values per thread at 512 threads): the per-SM
// atomic path is saturated from U = 16 on, so the U sweep of returning
// atomics is steady where the W = 4 sweep was not.  kMaxWarps = 16.
#include "../mb.cuh"

#include <string>
#include <utility>
#include <vector>

// OP ids match the SASS-gate table in sass_check.py.
template <int OP>
__device__ __forceinline__ std::uint32_t atomic_op(std::uint64_t gaddr, std::uint32_t saddr, std::uint32_t v) {
  std::uint32_t r = 0;
  if constexpr (OP == 0)
    asm volatile("red.global.add.u32 [%0], %1;" ::"l"(gaddr), "r"(v) : "memory");
  else if constexpr (OP == 1)
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v) : "memory");
  else if constexpr (OP == 2)
    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v) : "memory");
  else if constexpr (OP == 3)
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(saddr), "r"(v) : "memory");
  else
    asm volatile("atom.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v) : "memory");
  return r;
}

constexpr int kWarpRegionBytes = 4096;          // r1: 32 lines per warp
constexpr int kIndepWarpRegionBytes = 512 * 1024;  // r2: 128 x 32 lines per warp
constexpr int kIndepStrideBytes = 4096;         // r2: per-instruction stride

constexpr int kMaxWarps = 16;

// U <= 64 kernels may run 16 warps (r4); U = 128 keeps the 4-warp bound so
// its 128 returned values stay in registers
template <int OP, int U, int IND>
__global__ void __launch_bounds__(U <= 64 ? 512 : 128, 1) k_atomic(std::uint32_t* gbuf, std::uint64_t* out, int reps, int l,
                                                   std::uint32_t* sink, int sbytes) {
  __shared__ __align__(128) std::uint32_t sbuf[kMaxWarps * 32];  // 128 B per warp
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const int b = lane / l;
  const bool shared_op = OP == 2 || OP == 3;
  const int region = IND ? kIndepWarpRegionBytes : kWarpRegionBytes;
  const std::uint64_t gaddr =
      reinterpret_cast<std::uint64_t>(gbuf) + static_cast<std::uint64_t>(warp) * region + b * sbytes;
  const std::uint32_t saddr = mb_smem_u32(sbuf) + warp * 128 + b * 4;
  // warm the warp's global lines into L2 before the timed reps
  if (!shared_op) {
    std::uint32_t acc = 0;
    const int words = region / 4;
    for (int o = lane; o < words; o += 32) acc ^= gbuf[warp * words + o];
    if (acc == 0xdeadbeefu) sink[1] = acc;
  }
  if (shared_op && threadIdx.x < kMaxWarps * 32) sbuf[threadIdx.x] = 0;
  __syncthreads();
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    asm volatile("bar.sync 0, %0;" ::"r"(nwarps * 32) : "memory");
    const std::uint64_t t_s = mb_clock();
    std::uint32_t r[U];
#pragma unroll
    for (int i = 0; i < U; ++i)
      r[i] = atomic_op<OP>(gaddr + (IND ? i * kIndepStrideBytes : 0), saddr, rep * U + i);
    const std::uint64_t t_e = mb_clock();
    asm volatile("bar.sync 0, %0;" ::"r"(nwarps * 32) : "memory");
    if (lane == 0) out[rep * kMaxWarps + warp] = t_e - t_s;
    std::uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < U; ++i) acc ^= r[i];  // returned values used after t_e
    if (acc == 0xdeadbeefu) sink[threadIdx.x] = acc;
  }
}

using Kern = void (*)(std::uint32_t*, std::uint64_t*, int, int, std::uint32_t*, int);
struct Case {
  int op, u, ind;
  Kern k;
};

template <int OP, int IND, int... Us>
void add(std::vector<Case>& v, std::integer_sequence<int, Us...>) {
  (v.push_back({OP, Us, IND, &k_atomic<OP, Us, IND>}), ...);
}

int main(int argc, char** argv) {
  const MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "atomic");
  std::uint32_t* gbuf = nullptr;
  MB_CHECK(cudaMalloc(&gbuf, static_cast<std::size_t>(kMaxWarps) * kIndepWarpRegionBytes));
  MB_CHECK(cudaMemset(gbuf, 0, static_cast<std::size_t>(kMaxWarps) * kIndepWarpRegionBytes));
  std::uint64_t* d_out = nullptr;
  std::uint32_t* d_sink = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * kMaxWarps * args.reps));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(std::uint32_t) * 32 * kMaxWarps));

  std::vector<Case> cases;
  const std::integer_sequence<int, 8, 16, 32, 64> us;
  add<0, 0>(cases, us);  // red.global
  add<1, 0>(cases, us);  // atom.global
  add<2, 0>(cases, us);  // atom.shared
  add<3, 0>(cases, us);  // red.shared
  add<4, 0>(cases, us);  // atom generic
  const std::integer_sequence<int, 16, 32, 64, 128> us2;  // r2 independent lines
  add<0, 1>(cases, us2);  // red.global indep
  add<1, 1>(cases, us2);  // atom.global indep
  add<4, 1>(cases, us2);  // atom generic indep
  const char* opname[] = {"redg", "atomg", "atoms", "reds", "atom"};
  // (l, sbytes) cells: r1 / r2 block stride 128 for every L; r3 lines sweep
  // L = 1 with S in {4, 8, 16, 32, 64}; r3 same-word sweep S = 4, L > 1
  std::vector<std::pair<int, int>> cells;
  for (int l : {1, 2, 4, 8, 16, 32}) cells.push_back({l, 128});
  for (int sb : {4, 8, 16, 32, 64}) cells.push_back({1, sb});
  for (int l : {2, 4, 8, 16, 32}) cells.push_back({l, 4});
  for (const auto& c : cases) {
    for (const auto& [l, sb] : cells) {
      if (sb != 128 && !c.ind) continue;  // r3 cells are independent-line only
      for (int wn : {1, 4, 16}) {
        if (wn == 16 && (!c.ind || c.u > 64)) continue;  // r4: W = 16 only for U <= 64 independent-line cells
        const std::string suffix = !c.ind ? "" : sb == 128 ? ".indep" : ".S" + std::to_string(sb);
        const std::string variant = std::string("atomic.") + opname[c.op] + ".L" + std::to_string(l) +
                                    ".W" + std::to_string(wn) + suffix;
        if (!mb_selected(args, variant)) continue;
        const std::string params = std::string("\"op\":\"") + opname[c.op] + "\",\"l\":" + std::to_string(l) +
                                   ",\"warps\":" + std::to_string(wn) + ",\"indep\":" + std::to_string(c.ind) +
                                   ",\"sbytes\":" + std::to_string(sb);
        std::vector<double> s;
        for (int lch = 0; lch < args.launches; ++lch) {
          c.k<<<1, 32 * wn>>>(gbuf, d_out, args.reps, l, d_sink, sb);
          MB_CHECK(cudaGetLastError());
          MB_CHECK(cudaDeviceSynchronize());
          std::vector<std::uint64_t> h(kMaxWarps * static_cast<std::size_t>(args.reps));
          MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost));
          for (int r = 1; r < args.reps; ++r)
            for (int wp = 0; wp < wn; ++wp)
              s.push_back(static_cast<double>(static_cast<std::int64_t>(h[r * kMaxWarps + wp])));
        }
        w.row(variant, c.u, params, s);
      }
    }
  }
  return w.failures ? 1 : 0;
}
