#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

/* Header-free GetProcAddress intercept. cuda.h's cuGetProcAddress arity
 * changes across CUDA 11/12/13; compiling this TU without cuda.h avoids
 * conflicting types with cluster_launch_trace.c. */

extern unsigned int fi_wrap_driver_symbol(const char *symbol, void **pfn);

static void *fi_libcuda(void) {
  void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY);
  return h;
}

static void *fi_next(const char *name) {
  void *s = dlsym(RTLD_NEXT, name);
  if (s) return s;
  void *h = fi_libcuda();
  return h ? dlsym(h, name) : NULL;
}

typedef unsigned int (*fi_proc_v2_t)(const char *, void **, int, unsigned long long, int *);

unsigned int cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                                 unsigned long long flags, int *symbolStatus) {
  static fi_proc_v2_t real;
  if (!real) real = (fi_proc_v2_t)fi_next("cuGetProcAddress_v2");
  if (!real) return 999u; /* CUDA_ERROR_UNKNOWN */
  unsigned int st = real(symbol, pfn, cudaVersion, flags, symbolStatus);
  if (st == 0) fi_wrap_driver_symbol(symbol, pfn);
  return st;
}

unsigned int cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                              unsigned long long flags, int *symbolStatus) {
  static fi_proc_v2_t real;
  if (!real) real = (fi_proc_v2_t)fi_next("cuGetProcAddress");
  if (real && real != (fi_proc_v2_t)cuGetProcAddress) {
    unsigned int st = real(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (st == 0) fi_wrap_driver_symbol(symbol, pfn);
    return st;
  }
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
}
