#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static FILE *g_log;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_seq;
static void *real_sym(const char *name);

static void fi_open_log(void) {
  const char *path = getenv("FLASHINFER_CLUSTER_LAUNCH_LOG");
  if (!path || !path[0]) return;
  g_log = fopen(path, "a");
  if (g_log) setvbuf(g_log, NULL, _IOLBF, 0);
}

static void fi_cluster_from_cuda_attrs(const cudaLaunchAttribute *attrs, unsigned n,
                                       unsigned *cx, unsigned *cy, unsigned *cz) {
  *cx = *cy = *cz = 1;
  if (!attrs) return;
  for (unsigned i = 0; i < n; ++i) {
    if (attrs[i].id == cudaLaunchAttributeClusterDimension) {
      *cx = attrs[i].val.clusterDim.x ? attrs[i].val.clusterDim.x : 1;
      *cy = attrs[i].val.clusterDim.y ? attrs[i].val.clusterDim.y : 1;
      *cz = attrs[i].val.clusterDim.z ? attrs[i].val.clusterDim.z : 1;
      return;
    }
  }
}

static void fi_cluster_from_cu_attrs(const CUlaunchAttribute *attrs, unsigned n,
                                     unsigned *cx, unsigned *cy, unsigned *cz) {
  *cx = *cy = *cz = 1;
  if (!attrs) return;
  for (unsigned i = 0; i < n; ++i) {
    if (attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      *cx = attrs[i].value.clusterDim.x;
      *cy = attrs[i].value.clusterDim.y;
      *cz = attrs[i].value.clusterDim.z;
      return;
    }
  }
}

/* Never call cudaFuncGetName from a launch interceptor: it sets a sticky
 * runtime error (Triton then reports CUDA unknown error) and re-entering
 * the runtime during cudaLaunchKernelExC SIGSEGV'd segment_gemm. */
static const char *fi_elf_sname(const void *func) {
  if (!func) return NULL;
  static uintptr_t cached_addr;
  static char cached[768];
  uintptr_t addr = (uintptr_t)func;
  if (cached_addr == addr && cached[0]) return cached;
  FILE *maps = fopen("/proc/self/maps", "r");
  if (!maps) return NULL;
  char line[512];
  char path[384] = {0};
  unsigned long start = 0;
  while (fgets(line, sizeof(line), maps)) {
    unsigned long s = 0, e = 0;
    char perms[8] = {0};
    if (sscanf(line, "%lx-%lx %7s", &s, &e, perms) < 3) continue;
    if (addr < s || addr >= e || !strchr(perms, 'x')) continue;
    char *slash = strchr(line, '/');
    if (!slash) break;
    strncpy(path, slash, sizeof(path) - 1);
    path[strcspn(path, "\n")] = 0;
    start = s;
    break;
  }
  fclose(maps);
  if (!path[0] || !start) return NULL;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;
  struct stat st;
  if (fstat(fd, &st) || st.st_size < 64) {
    close(fd);
    return NULL;
  }
  size_t sz = (size_t)st.st_size;
  void *map = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return NULL;
  const unsigned char *base = (const unsigned char *)map;
  if (base[0] != 0x7f || base[1] != 'E' || base[2] != 'L' || base[3] != 'F' || base[4] != 2) {
    munmap(map, sz);
    return NULL;
  }
  uint64_t shoff = 0;
  uint16_t shentsize = 0, shnum = 0;
  memcpy(&shoff, base + 40, 8);
  memcpy(&shentsize, base + 58, 2);
  memcpy(&shnum, base + 60, 2);
  if (!shoff || shentsize < 64 || shoff + (uint64_t)shentsize * shnum > sz) {
    munmap(map, sz);
    return NULL;
  }
  const char *best = NULL;
  uint64_t best_val = 0;
  uintptr_t file_off = addr - (uintptr_t)start;
  for (uint16_t i = 0; i < shnum; ++i) {
    const unsigned char *sh = base + shoff + (uint64_t)i * shentsize;
    uint32_t type = 0, link = 0;
    memcpy(&type, sh + 4, 4);
    if (type != 2) continue;
    uint64_t off = 0, size = 0, entsize = 0;
    memcpy(&off, sh + 24, 8);
    memcpy(&size, sh + 32, 8);
    memcpy(&link, sh + 40, 4);
    memcpy(&entsize, sh + 56, 8);
    if (!entsize) entsize = 24;
    if (off + size > sz || link >= shnum) continue;
    const unsigned char *strsh = base + shoff + (uint64_t)link * shentsize;
    uint64_t stroff = 0, strsz = 0;
    memcpy(&stroff, strsh + 24, 8);
    memcpy(&strsz, strsh + 32, 8);
    if (stroff + strsz > sz) continue;
    const char *strs = (const char *)base + stroff;
    uint64_t nsym = size / entsize;
    for (uint64_t sidx = 0; sidx < nsym; ++sidx) {
      const unsigned char *sym = base + off + sidx * entsize;
      uint32_t nameoff = 0;
      uint16_t shndx = 0;
      uint64_t val = 0, ss = 0;
      memcpy(&nameoff, sym, 4);
      unsigned char stype = (unsigned char)(sym[4] & 0xf);
      memcpy(&shndx, sym + 6, 2);
      memcpy(&val, sym + 8, 8);
      memcpy(&ss, sym + 16, 8);
      if (stype != 2 || shndx == 0 || !ss) continue;
      if (file_off < val || file_off >= val + ss) continue;
      if (nameoff >= strsz || !strs[nameoff]) continue;
      if (!best || val >= best_val) {
        best = strs + nameoff;
        best_val = val;
      }
    }
  }
  const char *out = NULL;
  if (best) {
    strncpy(cached, best, sizeof(cached) - 1);
    cached[sizeof(cached) - 1] = 0;
    cached_addr = addr;
    out = cached;
  }
  munmap(map, sz);
  return out;
}

static const char *fi_elf_sname_mod(const void *func, const char *path, const void *fbase) {
  if (!func || !path || !fbase) return NULL;
  static uintptr_t cached_addr;
  static char cached[768];
  uintptr_t addr = (uintptr_t)func;
  if (cached_addr == addr && cached[0]) return cached;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;
  struct stat st;
  if (fstat(fd, &st) || st.st_size < 64) {
    close(fd);
    return NULL;
  }
  size_t sz = (size_t)st.st_size;
  void *map = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return NULL;
  const unsigned char *base = (const unsigned char *)map;
  if (base[0] != 0x7f || base[1] != 'E' || base[2] != 'L' || base[3] != 'F' || base[4] != 2) {
    munmap(map, sz);
    return NULL;
  }
  uint64_t shoff = 0;
  uint16_t shentsize = 0, shnum = 0;
  memcpy(&shoff, base + 40, 8);
  memcpy(&shentsize, base + 58, 2);
  memcpy(&shnum, base + 60, 2);
  if (!shoff || shentsize < 64 || shoff + (uint64_t)shentsize * shnum > sz) {
    munmap(map, sz);
    return NULL;
  }
  const char *best = NULL;
  uint64_t best_val = 0;
  uintptr_t rel = addr - (uintptr_t)fbase;
  for (uint16_t i = 0; i < shnum; ++i) {
    const unsigned char *sh = base + shoff + (uint64_t)i * shentsize;
    uint32_t type = 0, link = 0;
    memcpy(&type, sh + 4, 4);
    if (type != 2 && type != 11) continue;
    uint64_t off = 0, size = 0, entsize = 0;
    memcpy(&off, sh + 24, 8);
    memcpy(&size, sh + 32, 8);
    memcpy(&link, sh + 40, 4);
    memcpy(&entsize, sh + 56, 8);
    if (!entsize) entsize = 24;
    if (off + size > sz || link >= shnum) continue;
    const unsigned char *strsh = base + shoff + (uint64_t)link * shentsize;
    uint64_t stroff = 0, strsz = 0;
    memcpy(&stroff, strsh + 24, 8);
    memcpy(&strsz, strsh + 32, 8);
    if (stroff + strsz > sz) continue;
    const char *strs = (const char *)base + stroff;
    uint64_t nsym = size / entsize;
    for (uint64_t sidx = 0; sidx < nsym; ++sidx) {
      const unsigned char *sym = base + off + sidx * entsize;
      uint32_t nameoff = 0;
      uint16_t shndx = 0;
      uint64_t val = 0, ss = 0;
      memcpy(&nameoff, sym, 4);
      unsigned char stype = (unsigned char)(sym[4] & 0xf);
      memcpy(&shndx, sym + 6, 2);
      memcpy(&val, sym + 8, 8);
      memcpy(&ss, sym + 16, 8);
      if (stype != 2 || shndx == 0) continue;
      uint64_t span = ss ? ss : 1;
      if (rel < val || rel >= val + span) continue;
      if (nameoff >= strsz || !strs[nameoff]) continue;
      if (!best || val >= best_val) {
        best = strs + nameoff;
        best_val = val;
      }
    }
  }
  const char *out = NULL;
  if (best) {
    strncpy(cached, best, sizeof(cached) - 1);
    cached[sizeof(cached) - 1] = 0;
    cached_addr = addr;
    out = cached;
  }
  munmap(map, sz);
  return out;
}

static const char *fi_cuda_name(const void *func) {
  if (!func) return NULL;
  Dl_info info;
  memset(&info, 0, sizeof(info));
  if (dladdr(func, &info)) {
    if (info.dli_sname) return info.dli_sname;
    const char *elf = fi_elf_sname_mod(func, info.dli_fname, info.dli_fbase);
    if (elf) return elf;
  }
  return fi_elf_sname(func);
}

static const char *fi_cuda_name_runtime(const void *func) { return fi_cuda_name(func); }

/* Driver-only. Do not call this from cudaLaunchKernelExC. */
static const char *fi_cu_name(CUfunction func) {
  if (!func) return NULL;
  typedef CUresult (*fn_t)(const char **, CUfunction);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cuFuncGetName");
  if (!real) return NULL;
  const char *name = NULL;
  if (real(&name, func) == CUDA_SUCCESS && name) return name;
  return NULL;
}

enum {
  FI_ID_cuLaunchKernel = 0,
  FI_ID_cuLaunchKernel_ptsz,
  FI_ID_cuLaunchKernelEx,
  FI_ID_cuLaunchKernelEx_ptsz,
  FI_ID_cuLaunchCooperativeKernel,
  FI_ID_cuModuleLaunchKernel,
  FI_ID_COUNT
};

static void *g_drv_saved[FI_ID_COUNT];

static void *fi_drv_real(int id, const char *name) {
  if (id >= 0 && id < FI_ID_COUNT && g_drv_saved[id]) return g_drv_saved[id];
  return real_sym(name);
}

static FILE *g_code_log;
static unsigned long long g_code_seq;
static __thread int g_enqueue_status = -1;
static __thread CUstream g_launch_stream;
static __thread const CUlaunchConfig *g_launch_config;

/* Borrow the caller's live host config only until this launch log is written.
 * Do not dump union padding or kernel arguments (uninitialized/private bytes).
 * All attributes emitted by DG's host config are represented losslessly. */
static void fi_log_config(FILE *out) {
  int complete = !g_launch_config || !g_launch_config->numAttrs || g_launch_config->attrs;
  fprintf(out, ",\"stream\":\"%p\",\"num_attrs\":%u,\"attributes\":[",
          (void *)g_launch_stream, g_launch_config ? g_launch_config->numAttrs : 0);
  if (g_launch_config && g_launch_config->attrs)
    for (unsigned i = 0; i < g_launch_config->numAttrs; ++i) {
      const CUlaunchAttribute *attr = &g_launch_config->attrs[i];
      fprintf(out, "%s{\"id\":%d", i ? "," : "", (int)attr->id);
      if (attr->id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
        fprintf(out, ",\"cluster_dimension\":[%u,%u,%u]", attr->value.clusterDim.x, attr->value.clusterDim.y, attr->value.clusterDim.z);
      else if (attr->id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION)
        fprintf(out, ",\"programmatic_stream_serialization_allowed\":%d", attr->value.programmaticStreamSerializationAllowed);
      else if (attr->id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE)
        fprintf(out, ",\"cooperative\":%d", attr->value.cooperative);
      else { complete = 0; fputs(",\"supported\":false", out); }
      fputc('}', out);
    }
  fprintf(out, "],\"config_complete\":%s", complete ? "true" : "false");
}

static void fi_json_string(FILE *out, const char *s) {
  fputc('"', out);
  if (s) for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
    if (*p == '"' || *p == '\\') { fputc('\\', out); fputc(*p, out); }
    else if (*p < 32) fprintf(out, "\\u%04x", (unsigned)*p);
    else fputc(*p, out);
  }
  fputc('"', out);
}

static int fi_driver_identity(CUfunction f, CUcontext *ctx, CUmodule *mod, const char **name) {
  typedef CUresult (*ctx_t)(CUcontext *);
  typedef CUresult (*mod_t)(CUmodule *, CUfunction);
  static ctx_t getctx;
  static mod_t getmod;
  if (!getctx) getctx = (ctx_t)real_sym("cuCtxGetCurrent");
  if (!getmod) getmod = (mod_t)real_sym("cuFuncGetModule");
  *name = fi_cu_name(f);
  return getctx && getmod && *name && **name &&
    getctx(ctx) == CUDA_SUCCESS && *ctx && getmod(mod, f) == CUDA_SUCCESS && *mod;
}

static void fi_open_code_log(void) {
  if (g_code_log) return;
  const char *path = getenv("FLASHINFER_LOADED_CODE_LOG");
  if (path && *path) {
    g_code_log = fopen(path, "a");
    if (g_code_log) setvbuf(g_code_log, NULL, _IOLBF, 0);
  }
}

int fi_deepgemm_bind(const char *path, void *library, void *module, void *function, const char *load_api) {
  CUcontext ctx = NULL;
  CUmodule actual = NULL;
  const char *name = NULL;
  int valid = path && *path && library && module && function &&
    fi_driver_identity((CUfunction)function, &ctx, &actual, &name) && actual == (CUmodule)module;
  pthread_mutex_lock(&g_mu);
  fi_open_code_log();
  if (g_code_log) {
    fprintf(g_code_log, "{\"event\":\"bind\",\"code_seq\":%llu,\"launch_cursor\":%d,\"pid\":%d,\"library\":\"%p\",\"module\":\"%p\",\"func\":\"%p\",\"context\":\"%p\",\"binding_valid\":%s,\"name_bytes\":%zu,\"name\":", ++g_code_seq, g_seq, (int)getpid(), library, module, function, (void *)ctx, valid ? "true" : "false", name ? strlen(name) : 0);
    fi_json_string(g_code_log, name);
    fputs(",\"loaded_path\":", g_code_log); fi_json_string(g_code_log, path);
    fputs(",\"load_api\":", g_code_log); fi_json_string(g_code_log, load_api);
    fputs("}\n", g_code_log);
    fflush(g_code_log);
  } else valid = 0;
  pthread_mutex_unlock(&g_mu);
  return valid ? 0 : -1;
}

void fi_deepgemm_unload(void *library, int status) {
  pthread_mutex_lock(&g_mu);
  fi_open_code_log();
  if (g_code_log) fprintf(g_code_log, "{\"event\":\"unload\",\"code_seq\":%llu,\"launch_cursor\":%d,\"pid\":%d,\"library\":\"%p\",\"status\":%d}\n", ++g_code_seq, g_seq, (int)getpid(), library, status);
  pthread_mutex_unlock(&g_mu);
}

/* DG attention two-bank producer fact: the host launcher hands one marker
 * bank pointer to the kernel it is about to enqueue. Recorded here, at
 * Args-construction time, with the current launch cursor and the host
 * thread, so the consumer can bind bank<->physical launch without any
 * runtime-class-name or ordering guess. Same loaded_code channel and cursor
 * discipline as fi_deepgemm_bind: the event's launch_cursor is the seq of
 * the one launch this thread enqueues next; pid+thread+immediate-successor
 * uniqueness are enforced consumer-side. Device code, marker intervals and
 * timing are untouched. */
void fi_deepgemm_attention_bank(int bank, const void *marker_ptr, const void *logits_ptr,
                                unsigned long long payload_bytes, unsigned long long marker_bytes) {
  pthread_mutex_lock(&g_mu);
  fi_open_code_log();
  if (g_code_log)
    fprintf(g_code_log, "{\"event\":\"marker_bank\",\"code_seq\":%llu,\"launch_cursor\":%d,\"pid\":%d,\"thread\":%llu,\"bank\":%d,\"marker_ptr\":%llu,\"logits_ptr\":%llu,\"payload_bytes\":%llu,\"marker_bytes\":%llu}\n",
            ++g_code_seq, g_seq, (int)getpid(), (unsigned long long)pthread_self(),
            bank, (unsigned long long)(uintptr_t)marker_ptr,
            (unsigned long long)(uintptr_t)logits_ptr, payload_bytes, marker_bytes);
  pthread_mutex_unlock(&g_mu);
}

/* DG attention marker-reset producer fact: the full7 allocation helper
 * zeroes the marker-bank region of the output storage via an ATen fill
 * kernel enqueued through the CUDA runtime API inside the trial window.
 * The host records the exact byte region it is about to zero, with the
 * same cursor/thread discipline, so the consumer can admit exactly that
 * one runtime launch as the marker reset -- any other runtime launch in
 * the window stays unattested and fails closed. */
void fi_deepgemm_attention_reset(const void *base, unsigned long long bytes) {
  pthread_mutex_lock(&g_mu);
  fi_open_code_log();
  if (g_code_log)
    fprintf(g_code_log, "{\"event\":\"marker_reset\",\"code_seq\":%llu,\"launch_cursor\":%d,\"pid\":%d,\"thread\":%llu,\"base\":%llu,\"bytes\":%llu}\n",
            ++g_code_seq, g_seq, (int)getpid(), (unsigned long long)pthread_self(),
            (unsigned long long)(uintptr_t)base, bytes);
  pthread_mutex_unlock(&g_mu);
}

uint64_t fi_launch_cursor(void) {
  pthread_mutex_lock(&g_mu);
  uint64_t value = (uint64_t)g_seq;
  if (g_log) fflush(g_log);
  if (g_code_log) fflush(g_code_log);
  pthread_mutex_unlock(&g_mu);
  return value;
}

static void fi_log(const char *api, const char *name, const void *func, unsigned gx,
                   unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                   unsigned cx, unsigned cy, unsigned cz, unsigned smem) {
  if (!g_log) { g_launch_config = NULL; g_launch_stream = NULL; return; }
  CUcontext ctx = NULL;
  CUmodule module = NULL;
  int driver = strncmp(api, "cuda", 4) != 0;
  int valid = driver && fi_driver_identity((CUfunction)func, &ctx, &module, &name);
  int geometry_valid = cx && cy && cz;
  if (driver) {
    typedef CUresult (*attr_t)(int *, CUfunction_attribute, CUfunction);
    static attr_t attr;
    if (!attr) attr = (attr_t)real_sym("cuFuncGetAttribute");
    int required[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
      if (!attr || attr(&required[i], (CUfunction_attribute)(11 + i), (CUfunction)func) != CUDA_SUCCESS) geometry_valid = 0;
    int explicit_cluster = 0;
    if (g_launch_config && g_launch_config->attrs)
      for (unsigned i = 0; i < g_launch_config->numAttrs; ++i)
        if (g_launch_config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) explicit_cluster = 1;
    if (required[0] || required[1] || required[2]) {
      if (!required[0] || !required[1] || !required[2]) geometry_valid = 0;
      else if (!explicit_cluster && cx == 1 && cy == 1 && cz == 1) { cx = required[0]; cy = required[1]; cz = required[2]; }
      else if (cx != (unsigned)required[0] || cy != (unsigned)required[1] || cz != (unsigned)required[2]) geometry_valid = 0;
    }
  }
  pthread_mutex_lock(&g_mu);
  fprintf(g_log, "{\"seq\":%d,\"pid\":%d,\"thread\":%llu,\"api\":\"%s\",\"func\":\"%p\",\"context\":\"%p\",\"module\":\"%p\",\"driver_identity\":%s,\"binding_valid\":%s,\"geometry_valid\":%s,\"enqueue_status\":%d,\"name_bytes\":%zu,\"name\":", ++g_seq, (int)getpid(), (unsigned long long)pthread_self(), api, func, (void *)ctx, (void *)module, driver ? "true" : "false", valid ? "true" : "false", geometry_valid ? "true" : "false", driver ? g_enqueue_status : -1, name ? strlen(name) : 0);
  fi_json_string(g_log, name);
  fprintf(g_log, ",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"cluster\":[%u,%u,%u],\"cta_count\":%llu,\"cluster_ctas\":%llu,\"dynamic_smem\":%u", gx, gy, gz, bx, by, bz, cx, cy, cz, (unsigned long long)gx*gy*gz, (unsigned long long)cx*cy*cz, smem);
  if (driver) fi_log_config(g_log);
  fputs("}\n", g_log);
  g_launch_config = NULL;
  g_launch_stream = NULL;
  pthread_mutex_unlock(&g_mu);
}

/* Same-process attach: FLASHINFER_CLUSTER_SMID_LOG is written by the target
 * kernel %%smid dump after those CTAs run. Bind the dump onto matching 2CTA
 * launch rows so launches.jsonl carries the two SM ids. Do not invent ids. */
static int fi_parse_smids(const char *dump, int *out, int maxn) {
  const char *p = strstr(dump, "\"smids\"");
  int n = 0;
  if (!p) return 0;
  p = strchr(p, '[');
  if (!p) return 0;
  p++;
  while (*p && *p != ']' && n < maxn) {
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    if (*p == ']' || !*p) break;
    char *end = NULL;
    int v = (int)strtol(p, &end, 10);
    if (end == p) break;
    int dup = 0;
    for (int i = 0; i < n; ++i)
      if (out[i] == v) dup = 1;
    if (!dup) out[n++] = v;
    p = end;
  }
  return n;
}

static int fi_nctaid_product(const char *dump) {
  const char *p = strstr(dump, "cluster_nctaid");
  int x = 1, y = 1, z = 1;
  if (!p) return 0;
  p = strchr(p, '[');
  if (!p) return 0;
  if (sscanf(p, "[%d,%d,%d", &x, &y, &z) < 1) return 0;
  if (x < 1) x = 1;
  if (y < 1) y = 1;
  if (z < 1) z = 1;
  return x * y * z;
}

static void fi_attach_smids_from_dump(void) {
  const char *logp = getenv("FLASHINFER_CLUSTER_LAUNCH_LOG");
  const char *smp = getenv("FLASHINFER_CLUSTER_SMID_LOG");
  char dump[2048];
  int smids[16];
  int nsm = 0;
  int ncta = 0;
  FILE *sf;
  FILE *lf;
  FILE *out;
  char tmp[768];
  char line[2048];
  char smid_json[256];
  int i;
  int njson;
  if (!logp || !logp[0] || !smp || !smp[0]) return;
  pthread_mutex_lock(&g_mu);
  if (g_log) {
    fflush(g_log);
    fclose(g_log);
    g_log = NULL;
  }
  pthread_mutex_unlock(&g_mu);
  sf = fopen(smp, "r");
  if (!sf) return;
  dump[0] = 0;
  if (!fgets(dump, sizeof(dump), sf)) {
    fclose(sf);
    return;
  }
  fclose(sf);
  nsm = fi_parse_smids(dump, smids, 16);
  if (nsm < 1) return;
  ncta = fi_nctaid_product(dump);
  njson = snprintf(smid_json, sizeof(smid_json), "\"smids\":[");
  for (i = 0; i < nsm && njson > 0 && (size_t)njson + 16 < sizeof(smid_json); ++i)
    njson += snprintf(smid_json + njson, sizeof(smid_json) - (size_t)njson,
                      "%s%d", i ? "," : "", smids[i]);
  snprintf(smid_json + njson, sizeof(smid_json) - (size_t)njson, "],\"sm_ids\":[");
  njson = (int)strlen(smid_json);
  for (i = 0; i < nsm && njson > 0 && (size_t)njson + 16 < sizeof(smid_json); ++i)
    njson += snprintf(smid_json + njson, sizeof(smid_json) - (size_t)njson,
                      "%s%d", i ? "," : "", smids[i]);
  snprintf(smid_json + njson, sizeof(smid_json) - (size_t)njson, "]");
  snprintf(tmp, sizeof(tmp), "%s.smidattach", logp);
  lf = fopen(logp, "r");
  if (!lf) return;
  out = fopen(tmp, "w");
  if (!out) {
    fclose(lf);
    return;
  }
  while (fgets(line, sizeof(line), lf)) {
    size_t len = strlen(line);
    int cluster2 = (strstr(line, "\"cluster_ctas\":2") != NULL);
    int ordinary2 = (strstr(line, "\"cta_count\":2") != NULL) &&
                    (strstr(line, "\"cluster_ctas\":1") != NULL);
    int match = (ncta == 2 && cluster2) || (ncta != 2 && ordinary2);
    if (match && strstr(line, "\"smids\"") == NULL && len > 3 && line[len - 1] == '\n' &&
        line[len - 2] == '}') {
      line[len - 2] = 0;
      fprintf(out, "%s,%s}\n", line, smid_json);
    } else {
      fputs(line, out);
    }
  }
  fclose(lf);
  fclose(out);
  rename(tmp, logp);
}

__attribute__((destructor)) static void fi_fini(void) { fi_attach_smids_from_dump(); }

static void *real_sym(const char *name) {
  /* RTLD_NEXT only would miss some CUDA 12 driver symbols. Triton then gets
   * CUDA unknown error from a NULL real cuDeviceGetAttribute. */
  void *sym = dlsym(RTLD_NEXT, name);
  if (sym) return sym;
  int driver = (strncmp(name, "cu", 2) == 0 && strncmp(name, "cuda", 4) != 0);
  const char *lib = driver ? "libcuda.so.1" : "libcudart.so";
  void *h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen(lib, RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args,
                             size_t smem, cudaStream_t stream) {
  typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchKernel");
  if (!g_log) fi_open_log();
  fi_log("cudaLaunchKernel", fi_cuda_name_runtime(func), func, grid.x, grid.y, grid.z,
         block.x, block.y, block.z, 1, 1, 1, (unsigned)smem);
  if (!real) return cudaErrorUnknown;
  return real(func, grid, block, args, smem, stream);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func,
                                void **args) {
  typedef cudaError_t (*fn_t)(const cudaLaunchConfig_t *, const void *, void **);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchKernelExC");
  if (!g_log) fi_open_log();
  unsigned cx = 1, cy = 1, cz = 1;
  if (config)
    fi_cluster_from_cuda_attrs(config->attrs, config->numAttrs, &cx, &cy, &cz);
  if (config)
    fi_log("cudaLaunchKernelExC", fi_cuda_name(func), func, config->gridDim.x,
           config->gridDim.y, config->gridDim.z, config->blockDim.x,
           config->blockDim.y, config->blockDim.z, cx, cy, cz,
           (unsigned)config->dynamicSmemBytes);
  if (!real) return cudaErrorUnknown;
  return real(config, func, args);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                CUstream stream, void **params, void **extra) {
  typedef CUresult (*fn_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void **, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuLaunchKernel, "cuLaunchKernel");
  if (!g_log) fi_open_log();
  CUresult st = real ? real(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_stream = stream;
  fi_log("cuLaunchKernel", fi_cu_name(f), (const void *)f, gx, gy, gz, bx, by, bz,
         1, 1, 1, smem);
  return st;
}

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                  void **params, void **extra) {
  typedef CUresult (*fn_t)(const CUlaunchConfig *, CUfunction, void **, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuLaunchKernelEx, "cuLaunchKernelEx");
  if (!g_log) fi_open_log();
  unsigned cx = 1, cy = 1, cz = 1;
  if (config)
    fi_cluster_from_cu_attrs(config->attrs, config->numAttrs, &cx, &cy, &cz);
  CUresult st = real ? real(config, f, params, extra) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_config = config;
  g_launch_stream = config ? config->hStream : NULL;
  if (config)
    fi_log("cuLaunchKernelEx", fi_cu_name(f), (const void *)f, config->gridDimX,
           config->gridDimY, config->gridDimZ, config->blockDimX, config->blockDimY,
           config->blockDimZ, cx, cy, cz, config->sharedMemBytes);
  return st;
}


static int fi_target_sms(void) {
  const char *t = getenv("FLASHINFER_CLUSTER_SM_TARGET_SMS");
  if (!t || !t[0]) return 0;
  int v = atoi(t);
  return v > 0 ? v : 0;
}

/* Hardcode multiprocessor-count attribute ids. Header enums from a different
 * CUDA than libcudart can alias max-shared-memory and break Triton.
 * Do NOT hook cudaGetDeviceProperties: cudaDeviceProp layout is not stable
 * across the image runtime vs wheel headers, and Triton queries it. segment_gemm
 * 2CTA shrink uses FLASHINFER_CLUSTER_SM_TARGET_SMS in group_gemm_sm90.cuh. */
enum { FI_ATTR_MULTIPROCESSOR_COUNT = 16 };

cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
  typedef cudaError_t (*fn_t)(int *, enum cudaDeviceAttr, int);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaDeviceGetAttribute");
  if (!real) return cudaErrorUnknown;
  cudaError_t st = real(value, attr, device);
  int target = fi_target_sms();
  if (st == cudaSuccess && target > 0 && value &&
      (int)attr == FI_ATTR_MULTIPROCESSOR_COUNT) {
    *value = target;
  }
  return st;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  typedef CUresult (*fn_t)(int *, CUdevice_attribute, CUdevice);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cuDeviceGetAttribute");
  if (!real) return CUDA_ERROR_UNKNOWN;
  CUresult st = real(pi, attrib, dev);
  int target = fi_target_sms();
  if (st == CUDA_SUCCESS && target > 0 && pi &&
      (int)attrib == FI_ATTR_MULTIPROCESSOR_COUNT) {
    *pi = target;
  }
  return st;
}


cudaError_t cudaLaunchKernel_ptsz(const void *func, dim3 grid, dim3 block, void **args,
                                  size_t smem, cudaStream_t stream) {
  typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchKernel_ptsz");
  if (!g_log) fi_open_log();
  fi_log("cudaLaunchKernel_ptsz", fi_cuda_name_runtime(func), func, grid.x, grid.y, grid.z,
         block.x, block.y, block.z, 1, 1, 1, (unsigned)smem);
  if (!real) return cudaErrorUnknown;
  return real(func, grid, block, args, smem, stream);
}

cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t *config, const void *func,
                                     void **args) {
  typedef cudaError_t (*fn_t)(const cudaLaunchConfig_t *, const void *, void **);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchKernelExC_ptsz");
  if (!g_log) fi_open_log();
  unsigned cx = 1, cy = 1, cz = 1;
  if (config)
    fi_cluster_from_cuda_attrs(config->attrs, config->numAttrs, &cx, &cy, &cz);
  if (config)
    fi_log("cudaLaunchKernelExC_ptsz", fi_cuda_name(func), func, config->gridDim.x,
           config->gridDim.y, config->gridDim.z, config->blockDim.x,
           config->blockDim.y, config->blockDim.z, cx, cy, cz,
           (unsigned)config->dynamicSmemBytes);
  if (!real) return cudaErrorUnknown;
  return real(config, func, args);
}

CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                     unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                     CUstream stream, void **params, void **extra) {
  typedef CUresult (*fn_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void **, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuLaunchKernel_ptsz, "cuLaunchKernel_ptsz");
  if (!g_log) fi_open_log();
  CUresult st = real ? real(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_stream = stream;
  fi_log("cuLaunchKernel_ptsz", fi_cu_name(f), (const void *)f, gx, gy, gz, bx, by, bz,
         1, 1, 1, smem);
  return st;
}

CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f,
                                       void **params, void **extra) {
  typedef CUresult (*fn_t)(const CUlaunchConfig *, CUfunction, void **, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuLaunchKernelEx_ptsz, "cuLaunchKernelEx_ptsz");
  if (!g_log) fi_open_log();
  unsigned cx = 1, cy = 1, cz = 1;
  if (config) fi_cluster_from_cu_attrs(config->attrs, config->numAttrs, &cx, &cy, &cz);
  CUresult st = real ? real(config, f, params, extra) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_config = config;
  g_launch_stream = config ? config->hStream : NULL;
  if (config)
    fi_log("cuLaunchKernelEx_ptsz", fi_cu_name(f), (const void *)f, config->gridDimX,
           config->gridDimY, config->gridDimZ, config->blockDimX, config->blockDimY,
           config->blockDimZ, cx, cy, cz, config->sharedMemBytes);
  return st;
}

/* RMSNorm/PDL and some AOT cubins use cooperative / module launch, not cudaLaunchKernel. */
cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 grid, dim3 block,
                                                  void **args, size_t smem, cudaStream_t stream) {
  typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchCooperativeKernel");
  if (!g_log) fi_open_log();
  fi_log("cudaLaunchCooperativeKernel", fi_cuda_name_runtime(func), func, grid.x, grid.y, grid.z,
         block.x, block.y, block.z, 1, 1, 1, (unsigned)smem);
  if (!real) return cudaErrorUnknown;
  return real(func, grid, block, args, smem, stream);
}

cudaError_t CUDARTAPI cudaLaunchCooperativeKernel_ptsz(const void *func, dim3 grid, dim3 block,
                                                       void **args, size_t smem,
                                                       cudaStream_t stream) {
  typedef cudaError_t (*fn_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static fn_t real;
  if (!real) real = (fn_t)real_sym("cudaLaunchCooperativeKernel_ptsz");
  if (!g_log) fi_open_log();
  fi_log("cudaLaunchCooperativeKernel_ptsz", fi_cuda_name_runtime(func), func, grid.x, grid.y, grid.z,
         block.x, block.y, block.z, 1, 1, 1, (unsigned)smem);
  if (!real) return cudaErrorUnknown;
  return real(func, grid, block, args, smem, stream);
}

CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                           unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                           CUstream stream, void **params) {
  typedef CUresult (*fn_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuLaunchCooperativeKernel, "cuLaunchCooperativeKernel");
  if (!g_log) fi_open_log();
  CUresult st = real ? real(f, gx, gy, gz, bx, by, bz, smem, stream, params) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_stream = stream;
  fi_log("cuLaunchCooperativeKernel", fi_cu_name(f), (const void *)f, gx, gy, gz, bx, by, bz,
         1, 1, 1, smem);
  return st;
}

CUresult CUDAAPI cuModuleLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                      unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                      CUstream stream, void **params, void **extra) {
  typedef CUresult (*fn_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, CUstream, void **, void **);
  fn_t real = (fn_t)fi_drv_real(FI_ID_cuModuleLaunchKernel, "cuModuleLaunchKernel");
  if (!g_log) fi_open_log();
  CUresult st = real ? real(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra) : CUDA_ERROR_UNKNOWN;
  g_enqueue_status = (int)st;
  g_launch_stream = stream;
  fi_log("cuModuleLaunchKernel", fi_cu_name(f), (const void *)f, gx, gy, gz, bx, by, bz,
         1, 1, 1, smem);
  return st;
}

/* Called from cluster_launch_getproc.c (no cuda.h) when cuGetProcAddress
 * resolves a launch symbol. Saves the real driver pfn, then substitutes our
 * wrapper so Triton/CuTe launches are logged. */
unsigned int fi_wrap_driver_symbol(const char *symbol, void **pfn) {
  if (!symbol || !pfn || !*pfn) return 0;
  int id = -1;
  void *wrapper = NULL;
  if (!strcmp(symbol, "cuLaunchKernel")) {
    id = FI_ID_cuLaunchKernel;
    wrapper = (void *)cuLaunchKernel;
  } else if (!strcmp(symbol, "cuLaunchKernel_ptsz")) {
    id = FI_ID_cuLaunchKernel_ptsz;
    wrapper = (void *)cuLaunchKernel_ptsz;
  } else if (!strcmp(symbol, "cuLaunchKernelEx")) {
    id = FI_ID_cuLaunchKernelEx;
    wrapper = (void *)cuLaunchKernelEx;
  } else if (!strcmp(symbol, "cuLaunchKernelEx_ptsz")) {
    id = FI_ID_cuLaunchKernelEx_ptsz;
    wrapper = (void *)cuLaunchKernelEx_ptsz;
  } else if (!strcmp(symbol, "cuLaunchCooperativeKernel")) {
    id = FI_ID_cuLaunchCooperativeKernel;
    wrapper = (void *)cuLaunchCooperativeKernel;
  } else if (!strcmp(symbol, "cuModuleLaunchKernel")) {
    id = FI_ID_cuModuleLaunchKernel;
    wrapper = (void *)cuModuleLaunchKernel;
  }
  if (id < 0 || !wrapper) return 0;
  if (*pfn == wrapper) return 0;
  pthread_mutex_lock(&g_mu);
  if (!g_drv_saved[id]) g_drv_saved[id] = *pfn;
  pthread_mutex_unlock(&g_mu);
  *pfn = wrapper;
  return 0;
}
