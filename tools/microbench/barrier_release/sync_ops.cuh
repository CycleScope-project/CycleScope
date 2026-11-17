// Shared mbarrier / commit helpers for the sync probes.
#pragma once

#include "mb.cuh"

constexpr std::uint32_t kNever = (1u << 20) - 1;  // arrival count that never completes

__device__ __forceinline__ void tx_arrive_complete(std::uint32_t mbar) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], 16;\n\t"
      "mbarrier.complete_tx.shared::cta.b64 [%0], 16;\n\t}\n" ::"r"(mbar)
      : "memory");
}

__device__ __forceinline__ void commit1(std::uint32_t mbar) {
#if __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
#else
  (void)mbar;  // sm_90a: tcgen05 does not exist; commit variants unrealised
#endif
}

__device__ __forceinline__ void commit2_mc(std::uint32_t mbar, std::uint16_t mask) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(mbar),
      "h"(mask)
      : "memory");
#else
  (void)mbar; (void)mask;  // sm_90a: unrealised
#endif
}

__device__ __forceinline__ std::uint32_t mapa(std::uint32_t local, std::uint32_t rank) {
  std::uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_cluster(std::uint32_t addr, std::uint32_t v) {
  asm volatile("st.volatile.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void spin_local(std::uint32_t addr, std::uint32_t want) {
  std::uint32_t v;
  do {
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  } while (v != want);
}

__device__ __forceinline__ void tx_expect(std::uint32_t mbar) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], 16;\n\t}\n" ::"r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tx_complete(std::uint32_t mbar) {
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], 16;" ::"r"(mbar) : "memory");
}

// arrive on an mbarrier given by its shared::cluster address (own or peer CTA)
__device__ __forceinline__ void arrive_cluster(std::uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}

__device__ __forceinline__ bool spin_local_to(std::uint32_t addr, std::uint32_t want,
                                              std::uint32_t* err = &g_mb_err,
                                              std::uint64_t limit = 1ull << 26) {
  const std::uint64_t t0 = mb_gtimer();
  std::uint32_t v;
  do {
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
    if (v != want && mb_gtimer() - t0 > limit) {
      atomicExch(err, 1u);
      return false;
    }
  } while (v != want);
  return true;
}
