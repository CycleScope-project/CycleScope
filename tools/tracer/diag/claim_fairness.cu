// claim_fairness: how evenly do co-resident CTAs claim work items from a
// shared atomic counter?  One persistent CTA per SM loops
//   item = atomicAdd(counter, 1); if (item >= n_items) exit; work(item)
// with a fixed per-item body (global loads + FMAs), then writes its claim
// count and %smid.  Run natively the counts are flat (+-1); run under the
// tracer the counts map each SM's speed under instrumentation, which is
// what a hardware dynamic scheduler (atomic tile counters, cluster launch
// control) turns into per-CTA work in a capture.
//
//   nvcc -O2 -arch=<sm> -o claim_fairness claim_fairness.cu
//   ./claim_fairness [items_per_cta=32] [work_iters=64] [ctas_per_sm=1] > claims.tsv
//
// Output: header line "# sms=<n> ctas=<n> items=<n> iters=<n> ms=<kernel ms>",
// then one line per CTA: cta<TAB>smid<TAB>claims.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)

__device__ __forceinline__ unsigned read_smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}

__global__ void claim_kernel(unsigned *counter, unsigned n_items, int work_iters,
                             const float *scratch, unsigned scratch_floats,
                             unsigned *claims, unsigned *smids, float *sink) {
  __shared__ unsigned s_item;
  unsigned my_claims = 0;
  float acc = (float)threadIdx.x;
  if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
  for (;;) {
    if (threadIdx.x == 0) s_item = atomicAdd(counter, 1u);
    __syncthreads();
    unsigned item = s_item;
    __syncthreads();
    if (item >= n_items) break;
    ++my_claims;
    unsigned base = (item * blockDim.x * 4u) % scratch_floats;
    for (int i = 0; i < work_iters; ++i) {
      unsigned off = (base + threadIdx.x + (unsigned)i * blockDim.x) % scratch_floats;
      acc = fmaf(acc, 0.999f, scratch[off]);
    }
  }
  if (threadIdx.x == 0) claims[blockIdx.x] = my_claims;
  if (acc == 12345.678f) sink[0] = acc;  // keep the work live
}

int main(int argc, char **argv) {
  unsigned items_per_cta = argc > 1 ? (unsigned)atoi(argv[1]) : 32;
  int work_iters = argc > 2 ? atoi(argv[2]) : 64;
  int ctas_per_sm = argc > 3 ? atoi(argv[3]) : 1;
  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  unsigned ctas = (unsigned)(sms * ctas_per_sm);
  unsigned n_items = items_per_cta * ctas;
  const unsigned block = 256;
  const unsigned scratch_floats = 1u << 22;  // 16 MiB, L2-resident

  unsigned *counter, *claims, *smids; float *scratch, *sink;
  CK(cudaMalloc(&counter, sizeof(unsigned)));
  CK(cudaMalloc(&claims, ctas * sizeof(unsigned)));
  CK(cudaMalloc(&smids, ctas * sizeof(unsigned)));
  CK(cudaMalloc(&scratch, scratch_floats * sizeof(float)));
  CK(cudaMalloc(&sink, sizeof(float)));
  CK(cudaMemset(counter, 0, sizeof(unsigned)));
  CK(cudaMemset(scratch, 0, scratch_floats * sizeof(float)));

  // all CTAs must be co-resident for the claim loop to be work stealing
  int max_per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm, claim_kernel, block, 0));
  if (max_per_sm < ctas_per_sm) {
    fprintf(stderr, "ctas_per_sm=%d exceeds occupancy %d\n", ctas_per_sm, max_per_sm);
    return 2;
  }

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  claim_kernel<<<ctas, block>>>(counter, n_items, work_iters, scratch, scratch_floats,
                                claims, smids, sink);
  CK(cudaEventRecord(t1));
  CK(cudaDeviceSynchronize());
  float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));

  std::vector<unsigned> h_claims(ctas), h_smids(ctas);
  CK(cudaMemcpy(h_claims.data(), claims, ctas * sizeof(unsigned), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_smids.data(), smids, ctas * sizeof(unsigned), cudaMemcpyDeviceToHost));
  printf("# sms=%d ctas=%u items=%u iters=%d ms=%.3f\n", sms, ctas, n_items, work_iters, ms);
  for (unsigned i = 0; i < ctas; ++i) printf("%u\t%u\t%u\n", i, h_smids[i], h_claims[i]);
  return 0;
}
