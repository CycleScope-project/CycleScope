// tools/microbench/store_hold/probe/store_hold.cu -- protocol v1
// (frozen tools/microbench/store_hold/protocol.json): how long an issued
// global store keeps its address / data registers unavailable to a later
// writer on the same warp.  One CTA, W warps; warp 0 is the timing warp;
// in W=4 cells warps 1..3 run the store_flood stream of the same
// geometry over the whole window.
//
// Timing window (warp 0): t0 = clock; K prior stores; the test store
// st [pa], va; ts = clock; X; t1 = clock.  Prior store #1 has its own
// pa1 / vd1 (it is the overwrite target of the ovw_prior cells); prior
// stores #2..K share one read-only address / data register pair and
// differ only by an immediate offset (protocol allows this at K = 40);
// the test store and X are in one asm block so their operand registers
// are bound and cannot be renamed.  Everything the overwrite / read
// produces is folded into acc after the last clock read and written to
// sink at kernel end.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "tensor_pipe/mb.cuh"

enum : int { V_B32 = 4, V_V4 = 16 };
enum : int { G_FULL16 = 0, G_LINE32 = 1 };
enum : int { X_NONE = 0, X_OVW_DATA = 1, X_OVW_ADDR = 2,
             X_OVW_PDATA = 3, X_OVW_PADDR = 4, X_RDATA = 5,
             X_RADDR = 6 };

template <int V>
__device__ __forceinline__ void grp8(std::uint64_t p, const unsigned *s) {
  if constexpr (V == V_V4) {
#define SH_STG16(O) \
    asm volatile("st.global.v4.b32 [%0+" #O "], {%1, %2, %3, %4};" \
                 ::"l"(p), "r"((s)[0]), "r"((s)[1]), "r"((s)[2]),    \
                    "r"((s)[3]) : "memory")
    SH_STG16(0); SH_STG16(32768); SH_STG16(65536); SH_STG16(98304);
    SH_STG16(131072); SH_STG16(163840); SH_STG16(196608); SH_STG16(229376);
#undef SH_STG16
  } else {
#define SH_STG4(O) \
    asm volatile("st.global.b32 [%0+" #O "], %1;" \
                 ::"l"(p), "r"((s)[0]) : "memory")
    SH_STG4(0); SH_STG4(32768); SH_STG4(65536); SH_STG4(98304);
    SH_STG4(131072); SH_STG4(163840); SH_STG4(196608); SH_STG4(229376);
#undef SH_STG4
  }
}

// the other warps' store_flood stream: 48 groups x 8 stores, no clocks
template <int V, int G>
struct FloodS {
  static __device__ __forceinline__ void go(std::uint64_t p,
                                            const unsigned *s) {
    if constexpr (G <= 48) {
      grp8<V>(p, s);
      FloodS<V, G + 1>::go(p + 262144, s);
    }
  }
};

static __device__ __forceinline__ void win_g0x0k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x0k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x0k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x0k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x0k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x0k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x0k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x0k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x0k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x0k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %5;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x1k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x1k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x1k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x1k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x1k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x1k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x1k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x1k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x1k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x1k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dp, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dp;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x2k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x2k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x2k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x2k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x2k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x2k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x2k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x2k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x2k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x2k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pp, %11, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pp;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x3k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x3k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x3k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x3k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x3k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x3k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x3k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x3k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x3k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x3k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 dq, %9, %10;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dq;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, dq;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x4k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x4k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x4k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x4k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x4k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x4k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x4k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x4k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x4k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x4k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 pq, %11, 9;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pq], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, pq;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x5k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x5k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x5k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x5k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x5k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x5k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x5k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x5k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x5k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x5k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u32 %15, dp, 1;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [%16], dp;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "xor.b32 %0, %0, %15;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x6k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x6k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x6k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x6k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g0x6k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"      "mov.b32 q1, %6;\n\t"      "mov.b32 q2, %7;\n\t"      "mov.b32 q3, %8;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"      "mov.b32 r1, %9;\n\t"      "mov.b32 r2, %10;\n\t"      "mov.b32 r3, %15;\n\t"
      "st.global.v4.b32 [pq], {dq, r1, r2, r3};\n\t"
      "st.global.v4.b32 [%11+65536], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+98304], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+131072], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+163840], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+196608], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+229376], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+262144], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+294912], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+327680], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+360448], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+393216], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+425984], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+458752], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+491520], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+524288], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+557056], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+589824], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+622592], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+655360], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+688128], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+720896], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+753664], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+786432], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+819200], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+851968], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+884736], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+917504], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+950272], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+983040], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1015808], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1048576], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1081344], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1114112], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1146880], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1179648], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1212416], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1245184], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1277952], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [%11+1310720], {%5, %6, %7, %8};\n\t"
      "st.global.v4.b32 [pp], {dp, q1, q2, q3};\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x6k0(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x6k1(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x6k4(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x6k16(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
static __device__ __forceinline__ void win_g1x6k40(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  asm volatile(
      "{\n\t"
      ".reg .u64 tt, uu, dd, pp, pq;\n\t"
      ".reg .u32 lo32, hi32, dp, dq, q1, q2, q3, r1, r2, r3;\n\t"
      "mov.b32 dp, %5;\n\t"
      "mov.b64 pp, %3;\n\t"
      "mov.b64 pq, %12;\n\t"
      "mov.b32 dq, %13;\n\t"
      "st.global.b32 [pq], dq;\n\t"
      "st.global.b32 [%11+65536], %13;\n\t"
      "st.global.b32 [%11+98304], %13;\n\t"
      "st.global.b32 [%11+131072], %13;\n\t"
      "st.global.b32 [%11+163840], %13;\n\t"
      "st.global.b32 [%11+196608], %13;\n\t"
      "st.global.b32 [%11+229376], %13;\n\t"
      "st.global.b32 [%11+262144], %13;\n\t"
      "st.global.b32 [%11+294912], %13;\n\t"
      "st.global.b32 [%11+327680], %13;\n\t"
      "st.global.b32 [%11+360448], %13;\n\t"
      "st.global.b32 [%11+393216], %13;\n\t"
      "st.global.b32 [%11+425984], %13;\n\t"
      "st.global.b32 [%11+458752], %13;\n\t"
      "st.global.b32 [%11+491520], %13;\n\t"
      "st.global.b32 [%11+524288], %13;\n\t"
      "st.global.b32 [%11+557056], %13;\n\t"
      "st.global.b32 [%11+589824], %13;\n\t"
      "st.global.b32 [%11+622592], %13;\n\t"
      "st.global.b32 [%11+655360], %13;\n\t"
      "st.global.b32 [%11+688128], %13;\n\t"
      "st.global.b32 [%11+720896], %13;\n\t"
      "st.global.b32 [%11+753664], %13;\n\t"
      "st.global.b32 [%11+786432], %13;\n\t"
      "st.global.b32 [%11+819200], %13;\n\t"
      "st.global.b32 [%11+851968], %13;\n\t"
      "st.global.b32 [%11+884736], %13;\n\t"
      "st.global.b32 [%11+917504], %13;\n\t"
      "st.global.b32 [%11+950272], %13;\n\t"
      "st.global.b32 [%11+983040], %13;\n\t"
      "st.global.b32 [%11+1015808], %13;\n\t"
      "st.global.b32 [%11+1048576], %13;\n\t"
      "st.global.b32 [%11+1081344], %13;\n\t"
      "st.global.b32 [%11+1114112], %13;\n\t"
      "st.global.b32 [%11+1146880], %13;\n\t"
      "st.global.b32 [%11+1179648], %13;\n\t"
      "st.global.b32 [%11+1212416], %13;\n\t"
      "st.global.b32 [%11+1245184], %13;\n\t"
      "st.global.b32 [%11+1277952], %13;\n\t"
      "st.global.b32 [%11+1310720], %13;\n\t"
      "st.global.b32 [pp], dp;\n\t"
      "mov.u64 tt, %%clock64;\n\t"
      "add.u64 %14, pp, 8;\n\t"
      "mov.u64 uu, %%clock64;\n\t"
      "st.global.b32 [pp], %13;\n\t"
      "sub.u64 dd, tt, %4;\n\t"
      "st.global.u64 [%1], dd;\n\t"
      "sub.u64 dd, uu, tt;\n\t"
      "st.global.u64 [%2], dd;\n\t"
      "mov.b64 {lo32, hi32}, %14;\n\t""xor.b32 %0, %0, lo32;\n\t"
      "}"
      : "+r"(acc)
      : "l"(d0), "l"(d1), "l"(pa), "l"(t0), "r"(va0), "r"(va1),
        "r"(va2), "r"(va3), "r"(vb), "r"(vc), "l"(pb), "l"(pa1),
        "r"(vd1), "l"(ry), "r"(rx), "l"(d2)
      : "memory");
}
template <int GEO, int X, int K>
static __device__ __forceinline__ void win_call(
    std::uint64_t *d0, std::uint64_t *d1, unsigned &acc,
    std::uint64_t t0, std::uint64_t pa, unsigned va0, unsigned va1,
    unsigned va2, unsigned va3, unsigned vb, unsigned vc,
    std::uint64_t pb, std::uint64_t pa1, unsigned vd1,
    std::uint64_t &ry, unsigned &rx, char *d2) {
  if constexpr (GEO == 0 && X == 0 && K == 0)
    win_g0x0k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 0 && K == 1)
    win_g0x0k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 0 && K == 4)
    win_g0x0k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 0 && K == 16)
    win_g0x0k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 0 && K == 40)
    win_g0x0k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 1 && K == 0)
    win_g0x1k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 1 && K == 1)
    win_g0x1k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 1 && K == 4)
    win_g0x1k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 1 && K == 16)
    win_g0x1k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 1 && K == 40)
    win_g0x1k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 2 && K == 0)
    win_g0x2k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 2 && K == 1)
    win_g0x2k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 2 && K == 4)
    win_g0x2k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 2 && K == 16)
    win_g0x2k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 2 && K == 40)
    win_g0x2k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 3 && K == 0)
    win_g0x3k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 3 && K == 1)
    win_g0x3k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 3 && K == 4)
    win_g0x3k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 3 && K == 16)
    win_g0x3k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 3 && K == 40)
    win_g0x3k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 4 && K == 0)
    win_g0x4k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 4 && K == 1)
    win_g0x4k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 4 && K == 4)
    win_g0x4k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 4 && K == 16)
    win_g0x4k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 4 && K == 40)
    win_g0x4k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 5 && K == 0)
    win_g0x5k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 5 && K == 1)
    win_g0x5k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 5 && K == 4)
    win_g0x5k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 5 && K == 16)
    win_g0x5k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 5 && K == 40)
    win_g0x5k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 6 && K == 0)
    win_g0x6k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 6 && K == 1)
    win_g0x6k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 6 && K == 4)
    win_g0x6k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 6 && K == 16)
    win_g0x6k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 0 && X == 6 && K == 40)
    win_g0x6k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 0 && K == 0)
    win_g1x0k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 0 && K == 1)
    win_g1x0k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 0 && K == 4)
    win_g1x0k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 0 && K == 16)
    win_g1x0k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 0 && K == 40)
    win_g1x0k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 1 && K == 0)
    win_g1x1k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 1 && K == 1)
    win_g1x1k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 1 && K == 4)
    win_g1x1k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 1 && K == 16)
    win_g1x1k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 1 && K == 40)
    win_g1x1k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 2 && K == 0)
    win_g1x2k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 2 && K == 1)
    win_g1x2k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 2 && K == 4)
    win_g1x2k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 2 && K == 16)
    win_g1x2k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 2 && K == 40)
    win_g1x2k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 3 && K == 0)
    win_g1x3k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 3 && K == 1)
    win_g1x3k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 3 && K == 4)
    win_g1x3k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 3 && K == 16)
    win_g1x3k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 3 && K == 40)
    win_g1x3k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 4 && K == 0)
    win_g1x4k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 4 && K == 1)
    win_g1x4k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 4 && K == 4)
    win_g1x4k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 4 && K == 16)
    win_g1x4k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 4 && K == 40)
    win_g1x4k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 5 && K == 0)
    win_g1x5k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 5 && K == 1)
    win_g1x5k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 5 && K == 4)
    win_g1x5k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 5 && K == 16)
    win_g1x5k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 5 && K == 40)
    win_g1x5k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 6 && K == 0)
    win_g1x6k0(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 6 && K == 1)
    win_g1x6k1(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 6 && K == 4)
    win_g1x6k4(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 6 && K == 16)
    win_g1x6k16(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
  if constexpr (GEO == 1 && X == 6 && K == 40)
    win_g1x6k40(d0,d1,acc,t0,pa,va0,va1,va2,va3,vb,vc,pb,pa1,vd1,ry,rx,d2);
}
// delay for warp 0 in W = 4 cells: dependent-add chain before t0 so the
// flood stream is in steady state when the window opens (~ a few
// thousand cycles; must stay before t0)
#define SH_DELAY \
    do { \
      _Pragma("unroll") \
      for (int i = 0; i < 1024; ++i) \
        asm volatile("add.u32 %0, %0, 1;" : "+r"(acc)); \
    } while (0)

template <int GEO, int X, int K, int W>
__global__ void __launch_bounds__(128, 1) k_hold(
    std::uint64_t *out, unsigned *sink, char *base, int reps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned acc = threadIdx.x;
  const int stride = (GEO == G_FULL16) ? 16 : 128;
  const unsigned src[4] = {0x11223344u + threadIdx.x, 0x55667788u,
                           0x99aabbccu, 0xddeeff00u};
  const std::uint64_t region = reinterpret_cast<std::uint64_t>(base) +
      static_cast<std::uint64_t>(lane) * stride;
  const unsigned vd1 = 0x77700000u + lane;
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    if (W > 1 && warp != 0) {
      const std::uint64_t pf =
          reinterpret_cast<std::uint64_t>(base) + 2u * 1024u * 1024u +
          static_cast<std::uint64_t>(warp) * 16u * 1024u * 1024u +
          static_cast<std::uint64_t>(lane) * stride;
      FloodS<GEO == G_FULL16 ? V_V4 : V_B32, 1>::go(pf, src);
    } else {
      std::uint64_t t0 = 0, ry = 0;
      unsigned rx = 0;
      const std::uint64_t pb = region;
      const std::uint64_t pa1 = pb + 32768;
      const std::uint64_t pa =
          pb + static_cast<std::uint64_t>(K + 1) * 32768u;
      const std::uint64_t *o = &out[rep * 2];
      if (W > 1) SH_DELAY;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
      win_call<GEO, X, K>(
          const_cast<std::uint64_t *>(o),
          const_cast<std::uint64_t *>(o + 1), acc, t0, pa,
          src[0], src[1], src[2], src[3], 0x0badf00du + lane,
          0x0cafe000u + lane, pb, pa1, vd1 ^ 0x5a5au, ry, rx,
          base + (62u << 20));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) sink[0] = acc + static_cast<unsigned>(region);
}








// all prior stores and the test store live in one asm block per cell
// (win_g<g>x<x>k<K>): prior store #1 first with its own pa1 / vd1,
// priors #2..K on the shared read-only pb / vb at immediate offsets.
// ------------------------------------------------------------------
// protocol v2 (loop form): k_loop<X, D, G>, W = 1 warp.  The loop body
// stores on loop-carried pointers p_0..p_D / data v_0..v_D, then applies
// X to each loop variable in rotation.  X = addr advances p_j by the
// 32 KiB request-region stride (each iteration writes new regions);
// X = data rewrites v[j][0] with pointers untouched (same lines every
// iteration); X = none leaves everything and advances a dummy counter
// D + 1 times so the body has the same instruction count.
enum { LX_NONE = 0, LX_ADDR = 1, LX_DATA = 2 };

template <int X, int D, int G>
__global__ void __launch_bounds__(32, 1) k_loop(
    std::uint64_t *out, unsigned *sink, char *base, int reps, int n) {
  const int lane = threadIdx.x & 31;
  const int stride = (G == G_FULL16) ? 16 : 128;
  std::uint64_t p[D + 1];
  unsigned v[D + 1][4];
  unsigned dm = lane;
  for (int j = 0; j <= D; ++j) {
    p[j] = reinterpret_cast<std::uint64_t>(base) +
           static_cast<std::uint64_t>(lane) * stride +
           static_cast<std::uint64_t>(j) * (24u * 32768u);
    v[j][0] = 0x1a2b0000u + lane + j;
    v[j][1] = 0x3c4d0000u + lane + j;
    v[j][2] = 0x5e6f0000u + lane + j;
    v[j][3] = 0x7a8b0000u + lane + j;
  }
#pragma unroll 1
  for (int rep = 0; rep < reps; ++rep) {
    __syncthreads();
    asm volatile("fence.sc.gpu;" ::: "memory");
    __syncthreads();
    std::uint64_t t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#pragma unroll 1
    for (int i = 0; i < n; ++i) {
#pragma unroll
      for (int j = 0; j <= D; ++j) {
        if (G == G_FULL16)
          asm volatile(
              "st.global.v4.b32 [%0], {%1,%2,%3,%4};"
              :: "l"(p[j]), "r"(v[j][0]), "r"(v[j][1]),
                 "r"(v[j][2]), "r"(v[j][3]) : "memory");
        else
          asm volatile(
              "st.global.b32 [%0], %1;"
              :: "l"(p[j]), "r"(v[j][0]) : "memory");
      }
#pragma unroll
      for (int j = 0; j <= D; ++j) {
        if (X == LX_ADDR)
          p[j] += 32768;
        else if (X == LX_DATA)
          v[j][0] += 1;
        else
          dm += 1;
      }
      asm volatile("" ::: "memory");
    }
    std::uint64_t t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (lane == 0) out[rep] = t1 - t0;
    __syncthreads();
  }
  if (lane == 0)
    sink[0] += dm + static_cast<unsigned>(p[0]) + v[0][0];
}

static const char *LXNAME[] = {"none", "addr", "data"};

using KFN2 = void (*)(std::uint64_t *, unsigned *, char *, int, int);

template <int X, int G>
static KFN2 dpick(int d) {
  switch (d) {
    case 0:  return k_loop<X, 0, G>;
    case 1:  return k_loop<X, 1, G>;
    default: return k_loop<X, 3, G>;
  }
}
static KFN2 lpick(int x, int d, int g) {
  switch (x) {
    case 0:
      return g == G_FULL16 ? dpick<0, G_FULL16>(d) : dpick<0, G_LINE32>(d);
    case 1:
      return g == G_FULL16 ? dpick<1, G_FULL16>(d) : dpick<1, G_LINE32>(d);
    default:
      return g == G_FULL16 ? dpick<2, G_FULL16>(d) : dpick<2, G_LINE32>(d);
  }
}

// ------------------------------------------------------------------
static double mb_pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

using KFN = void (*)(std::uint64_t *, unsigned *, char *, int);

template <int GEO, int X, int W>
static KFN kpick(int k) {
  switch (k) {
    case 0:  return k_hold<GEO, X, 0, W>;
    case 1:  return k_hold<GEO, X, 1, W>;
    case 4:  return k_hold<GEO, X, 4, W>;
    case 16: return k_hold<GEO, X, 16, W>;
    default: return k_hold<GEO, X, 40, W>;
  }
}
template <int GEO, int W>
static KFN xpick(int x, int k) {
  switch (x) {
    case 0: return kpick<GEO, 0, W>(k);
    case 1: return kpick<GEO, 1, W>(k);
    case 2: return kpick<GEO, 2, W>(k);
    case 3: return kpick<GEO, 3, W>(k);
    case 4: return kpick<GEO, 4, W>(k);
    case 5: return kpick<GEO, 5, W>(k);
    default: return kpick<GEO, 6, W>(k);
  }
}
static KFN pick(int g, int x, int k, int w) {
  if (w == 1)
    return g == G_FULL16 ? xpick<G_FULL16, 1>(x, k) : xpick<G_LINE32, 1>(x, k);
  return g == G_FULL16 ? xpick<G_FULL16, 4>(x, k) : xpick<G_LINE32, 4>(x, k);
}

static const char *XNAME[] = {"none", "ovw_data", "ovw_addr",
                             "ovw_prior_data", "ovw_prior_addr",
                             "read_data", "read_addr"};
static const char *GNAME[] = {"full16", "line32"};

int main(int argc, char **argv) {
  MbArgs args = mb_parse(argc, argv);
  MbWriter w(args, "store_hold");
  std::uint64_t *d_out = nullptr;
  unsigned *d_sink = nullptr;
  char *d_base = nullptr;
  MB_CHECK(cudaMalloc(&d_out, sizeof(std::uint64_t) * args.reps * 2));
  MB_CHECK(cudaMalloc(&d_sink, sizeof(unsigned) * 8));
  MB_CHECK(cudaMalloc(&d_base, 64u << 20));
  const bool v2 = std::getenv("SH_V2") != nullptr;
  if (v2) {
    const int Ds[] = {0, 1, 3};
    const int Ns[] = {4, 8, 16, 24};
    for (int x = 0; x < 3; ++x)
      for (int g = 0; g < 2; ++g)
        for (int d : Ds)
          for (int n : Ns) {
            const std::string variant = std::string("loop.") + LXNAME[x] +
                "." + GNAME[g] + ".d" + std::to_string(d) +
                ".n" + std::to_string(n);
            if (!mb_selected(args, variant)) continue;
            KFN2 kf = lpick(x, d, g);
            std::vector<double> u0;
            for (int l = 0; l < args.launches; ++l) {
              kf<<<1, 32>>>(d_out, d_sink, d_base, args.reps, n);
              cudaError_t e = cudaGetLastError();
              if (e == cudaSuccess) e = cudaDeviceSynchronize();
              if (e != cudaSuccess) {
                std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                             cudaGetErrorString(e));
                break;
              }
              std::vector<std::uint64_t> h(args.reps);
              MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                                  cudaMemcpyDeviceToHost));
              for (int r = 1; r < args.reps; ++r)
                u0.push_back(static_cast<double>(h[r]));
            }
            const std::string params =
                "\"geometry\":\"" + std::string(GNAME[g]) +
                "\",\"x\":\"" + LXNAME[x] +
                "\",\"d\":" + std::to_string(d) +
                ",\"n\":" + std::to_string(n) + ",\"warps\":1";
            w.row(variant, 0, params, u0);
            std::fprintf(stderr, "%s: %d launches\n", variant.c_str(),
                         args.launches);
          }
  } else {
  const int Ks[] = {0, 1, 4, 16, 40};
  for (int x = 0; x < 7; ++x)
    for (int g = 0; g < 2; ++g)
      for (int k : Ks)
        for (int W : {1, 4}) {
          if (k == 0 && (x == X_OVW_PDATA || x == X_OVW_PADDR)) continue;
          const std::string variant = std::string(XNAME[x]) + "." +
              GNAME[g] + ".k" + std::to_string(k) + ".x" + std::to_string(W);
          if (!mb_selected(args, variant)) continue;
          KFN kf = pick(g, x, k, W);
          MB_CHECK(cudaFuncSetAttribute(
              kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
          std::vector<double> u0, u1;
          for (int l = 0; l < args.launches; ++l) {
            MB_CHECK(cudaMemset(d_base, 0, 64u << 20));
            kf<<<1, 32 * W>>>(d_out, d_sink, d_base, args.reps);
            cudaError_t e = cudaGetLastError();
            if (e == cudaSuccess) e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
              std::fprintf(stderr, "launch %s: %s\n", variant.c_str(),
                           cudaGetErrorString(e));
              break;
            }
            std::vector<std::uint64_t> h(args.reps * 2);
            MB_CHECK(cudaMemcpy(h.data(), d_out, h.size() * 8,
                                cudaMemcpyDeviceToHost));
            for (int r = 1; r < args.reps; ++r) {
              u0.push_back(static_cast<double>(h[r * 2]));
              u1.push_back(static_cast<double>(h[r * 2 + 1]));
            }
          }
          const std::string params =
              "\"geometry\":\"" + std::string(GNAME[g]) +
              "\",\"x\":\"" + XNAME[x] +
              "\",\"k\":" + std::to_string(k) +
              ",\"warps\":" + std::to_string(W);
          w.row(variant, 0, params, u0);
          w.row(variant, 1, params, u1);
          std::fprintf(stderr, "%s: %d launches\n", variant.c_str(),
                       args.launches);
        }
  }
  MB_CHECK(cudaFree(d_base));
  MB_CHECK(cudaFree(d_out));
  MB_CHECK(cudaFree(d_sink));
  return 0;
}
