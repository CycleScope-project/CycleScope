// SM103 microbench helpers. Host+device. No CUTLASS.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error__));                             \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__device__ __forceinline__ std::uint64_t rdclk64() {
  std::uint64_t v;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(v)::"memory");
  return v;
}

__device__ __forceinline__ void cluster_arrive_wait() {
  asm volatile(
      "barrier.cluster.arrive.aligned;\n\t"
      "barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ std::uint32_t cluster_ctarank() {
  std::uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

__device__ __forceinline__ std::uint64_t globaltimer() {
  std::uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)::"memory");
  return v;
}

__device__ __forceinline__ std::uint32_t smem_u32(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ std::uint32_t read_smid() {
  std::uint32_t v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}

struct DeviceInfo {
  std::string name;
  int sm_clock_mhz;
  int mem_clock_mhz;
  int sm_count;
  int major;
  int minor;
  std::size_t memory_bytes;
};

inline DeviceInfo query_device() {
  DeviceInfo d{};
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  d.name = p.name;
  int clk = 0, memclk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
  d.sm_clock_mhz = clk / 1000;
  d.mem_clock_mhz = memclk / 1000;
  d.sm_count = p.multiProcessorCount;
  d.major = p.major;
  d.minor = p.minor;
  d.memory_bytes = p.totalGlobalMem;
  return d;
}

inline double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  const double idx = p * static_cast<double>(xs.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(idx);
  const std::size_t hi = std::min(lo + 1, xs.size() - 1);
  const double w = idx - static_cast<double>(lo);
  return xs[lo] * (1.0 - w) + xs[hi] * w;
}

inline double median_of(const std::vector<double>& xs) {
  return percentile(xs, 0.5);
}

inline void print_json_array(FILE* f, const std::vector<double>& xs) {
  std::fputc('[', f);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) std::fputc(',', f);
    std::fprintf(f, "%.6f", xs[i]);
  }
  std::fputc(']', f);
}

inline void json_escape(FILE* f, const char* s) {
  for (; *s; ++s) {
    if (*s == '"' || *s == '\\') std::fputc('\\', f);
    std::fputc(*s, f);
  }
}

struct Variant {
  std::string key;
  std::vector<double> samples;
  std::string extra_json;
};

inline void write_probe_json(const char* path, const DeviceInfo& dev,
                             const char* name, const char* method,
                             const char* unit, const std::vector<Variant>& vars,
                             int primary = 0, const char* extra_top_json = nullptr) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path);
    std::exit(1);
  }
  const Variant& p = vars.at(static_cast<std::size_t>(primary));
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"schema\": \"blackwell.microbench.v1\",\n");
  std::fprintf(f, "  \"name\": \"%s\",\n", name);
  std::fprintf(f, "  \"device\": \"");
  json_escape(f, dev.name.c_str());
  std::fprintf(f, "\",\n");
  std::fprintf(f, "  \"sm_clock_mhz\": %d,\n", dev.sm_clock_mhz);
  std::fprintf(f, "  \"mem_clock_mhz\": %d,\n", dev.mem_clock_mhz);
  std::fprintf(f, "  \"sm_count\": %d,\n", dev.sm_count);
  std::fprintf(f, "  \"cc\": \"%d.%d\",\n", dev.major, dev.minor);
  std::fprintf(f, "  \"unit\": \"%s\",\n", unit);
  std::fprintf(f, "  \"method\": \"");
  json_escape(f, method);
  std::fprintf(f, "\",\n");
  std::fprintf(f, "  \"median\": %.6f,\n", median_of(p.samples));
  std::fprintf(f, "  \"p10\": %.6f,\n", percentile(p.samples, 0.10));
  std::fprintf(f, "  \"p90\": %.6f,\n", percentile(p.samples, 0.90));
  std::fprintf(f, "  \"samples\": ");
  print_json_array(f, p.samples);
  std::fprintf(f, ",\n  \"variants\": [\n");
  for (std::size_t i = 0; i < vars.size(); ++i) {
    const Variant& v = vars[i];
    std::fprintf(f,
                 "    {\"key\": \"%s\", \"median\": %.6f, \"p10\": %.6f, "
                 "\"p90\": %.6f, \"n\": %zu, \"samples\": ",
                 v.key.c_str(), median_of(v.samples), percentile(v.samples, 0.10),
                 percentile(v.samples, 0.90), v.samples.size());
    print_json_array(f, v.samples);
    if (!v.extra_json.empty()) {
      std::fprintf(f, ", %s}", v.extra_json.c_str());
    } else {
      std::fprintf(f, "}");
    }
    std::fprintf(f, i + 1 == vars.size() ? "\n" : ",\n");
  }
  if (extra_top_json && *extra_top_json) {
    std::fprintf(f, "  ],\n  %s\n}\n", extra_top_json);
  } else {
    std::fprintf(f, "  ]\n}\n");
  }
  std::fclose(f);
  std::printf("wrote %s median=%.3f n=%zu\n", path, median_of(p.samples),
              p.samples.size());
}

__device__ __forceinline__ std::uint64_t make_smem_desc(std::uint32_t smem_addr,
                                                        std::uint32_t lbo_bytes,
                                                        std::uint32_t sbo_bytes) {
  const std::uint64_t addr_enc =
      (static_cast<std::uint64_t>(smem_addr) & 0x3FFFFu) >> 4;
  const std::uint64_t lbo_enc =
      (static_cast<std::uint64_t>(lbo_bytes) & 0x3FFFFu) >> 4;
  const std::uint64_t sbo_enc =
      (static_cast<std::uint64_t>(sbo_bytes) & 0x3FFFFu) >> 4;
  std::uint64_t d = 0;
  d |= addr_enc << 0;
  d |= lbo_enc << 16;
  d |= sbo_enc << 32;
  d |= static_cast<std::uint64_t>(0b001) << 46;
  return d;
}

__device__ __forceinline__ std::uint32_t make_idesc(std::uint16_t M, std::uint16_t N,
                                                   std::uint8_t atype, std::uint8_t btype,
                                                   std::uint8_t dtype) {
  std::uint32_t d = 0;
  d |= (static_cast<std::uint32_t>(dtype) & 0x3) << 4;
  d |= (static_cast<std::uint32_t>(atype) & 0x7) << 7;
  d |= (static_cast<std::uint32_t>(btype) & 0x7) << 10;
  d |= (static_cast<std::uint32_t>(N >> 3) & 0x3F) << 17;
  d |= (static_cast<std::uint32_t>(M >> 4) & 0x1F) << 24;
  return d;
}

__device__ __forceinline__ void mbar_init(std::uint64_t* mbar, unsigned expected) {
  const std::uint32_t a = smem_u32(mbar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(expected)
               : "memory");
}

__device__ __forceinline__ int mbar_try_wait(std::uint64_t* mbar, unsigned phase,
                                             std::uint64_t timeout_cyc = 50000000ull) {
  const std::uint32_t a = smem_u32(mbar);
  const std::uint64_t t0 = rdclk64();
  int done = 0;
  while (!done) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(done)
        : "r"(a), "r"(phase)
        : "memory");
    if (rdclk64() - t0 > timeout_cyc) return 0;
  }
  return 1;
}
__device__ __forceinline__ void tmem_alloc(std::uint32_t* slot, int ncols) {
  const std::uint32_t a = smem_u32(slot);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(a),
      "r"(ncols)
      : "memory");
}

__device__ __forceinline__ void tmem_dealloc(std::uint32_t taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
               "r"(ncols)
               : "memory");
}

__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" :::
               "memory");
}

__device__ __forceinline__ void umma_commit(std::uint64_t* mbar) {
  const std::uint32_t a = smem_u32(mbar);
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(a)
      : "memory");
}

// ---- cta_group::2 (2-SM UMMA) helpers ----

__device__ __forceinline__ void mbar_init_cluster(std::uint64_t* mbar,
                                                  unsigned expected) {
  mbar_init(mbar, expected);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc2(std::uint32_t* slot, int ncols) {
  const std::uint32_t a = smem_u32(slot);
  asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(a),
      "r"(ncols)
      : "memory");
}

__device__ __forceinline__ void tmem_dealloc2(std::uint32_t taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
               "r"(ncols)
               : "memory");
}

__device__ __forceinline__ void tmem_relinquish2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" :::
               "memory");
}

// commit with multicast: arrives on the mbarrier at the same smem offset in
// every CTA named by cta_mask (cluster ranks).
__device__ __forceinline__ void umma_commit2_mc(std::uint64_t* mbar,
                                                std::uint16_t cta_mask) {
  const std::uint32_t a = smem_u32(mbar);
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
      ".multicast::cluster.b64 [%0], %1;" ::"r"(a),
      "h"(cta_mask)
      : "memory");
}
