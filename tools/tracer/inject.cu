/* tracer_v5 device-side injected functions.
 *
 * Per executed instruction: one V5P_HEAD chunk, one V5P_MREF chunk per
 * MREF, one V5P_OP chunk per operand entry (R/UR/RANK/DESC).  PHASECHK
 * instructions additionally emit a V5P_AFTER chunk after the instruction.
 * All writes use fixed-stride indexed stores (nvcc/ptxas-safe); push sizes
 * cover only the populated prefix of each chunk.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "utils/utils.h"
/* same channel layout choice as tracer.cu (device buffer by default) */
#ifndef CYCLESCOPE_TRACER_HOST_CHANNEL
#define USE_ASYNC_STREAM 1
#endif
#include "utils/channel.hpp"
#include "record_format.h"

__device__ __forceinline__ bool v5_cta_sampled(uint64_t state_ptr, int4 cta) {
  const V5ClusterSampling *s = &((const V5DevState *)state_ptr)->sampling;
  if (s->count < 0)
    return true;
  if (s->cdx <= 0 || s->cdy <= 0 || s->cdz <= 0)
    return false;
  const int cx = cta.x / s->cdx;
  const int cy = cta.y / s->cdy;
  const int cz = cta.z / s->cdz;
  const uint32_t id = (uint32_t)cx + (uint32_t)cy * (uint32_t)s->ncx +
                      (uint32_t)cz * (uint32_t)s->ncx * (uint32_t)s->ncy;
  for (int i = 0; i < s->count; ++i) {
    if (s->ids[i] == id)
      return true;
  }
  return false;
}

__device__ __forceinline__ uint16_t v5_warp(void) {
  return (uint16_t)((threadIdx.z * blockDim.y * blockDim.x +
                     threadIdx.y * blockDim.x + threadIdx.x) /
                    32);
}

__device__ __forceinline__ void v5_chunk_head(V5ChunkHead *ch, uint8_t part,
                                            uint8_t kind, int4 cta,
                                            int32_t function_id, int32_t idx) {
  ch->cta_x = (uint32_t)cta.x;
  ch->cta_y = (uint16_t)cta.y;
  ch->cta_z = (uint16_t)cta.z;
  ch->warp = v5_warp();
  ch->part = part;
  ch->kind = kind;
  ch->function = (uint32_t)function_id;
  ch->idx = (uint32_t)idx;
  ch->pad = 0;
}

/* Copy a 128-byte image from generic space into dst.  Only called on an
 * operand proven to be a tensormap generic address (tma_map_role role 1);
 * the conservative canonical-48 boundary retained from diag9a/b keeps the
 * deref out of wild ranges, and .shared::cluster peers are never touched. */
__device__ __forceinline__ bool v5_copy_desc_image(uint8_t *dst,
                                                   uint64_t ptr) {
  if ((ptr >> 48) != 0)
    return false;
  int space = 0;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "isspacep.global p, %1;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(space)
               : "l"(ptr));
  if (space == 0) {
    int sspace = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "isspacep.shared::cta p, %1;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "}"
                 : "=r"(sspace)
                 : "r"((uint32_t)ptr));
    if (sspace == 0)
      return false;
  }
  const uint32_t *src = (const uint32_t *)ptr;
  uint32_t *d = (uint32_t *)dst;
  for (int i = 0; i < 32; ++i)
    d[i] = src[i];
  return true;
}

__device__ __forceinline__ uint64_t v5_globaltimer(void) {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

/* v6 placement record (CAPTURE_MODE=placement): the lead lane of each
 * warp stores one V5PlaceRec into the device placement array at function
 * entry (pkind 0) and at every EXIT instruction (pkind 1).  No channel,
 * no addresses/operands; the host copies the array out after the launch
 * and writes placement-<launch>.jsonl. */
static __device__ __forceinline__ int4 v5_tid_reg(void) {
  int4 r;
  asm("mov.u32 %0, %%tid.x;" : "=r"(r.x));
  asm("mov.u32 %0, %%tid.y;" : "=r"(r.y));
  asm("mov.u32 %0, %%tid.z;" : "=r"(r.z));
  r.w = 0;
  return r;
}
static __device__ __forceinline__ int4 v5_ntid_reg(void) {
  int4 r;
  asm("mov.u32 %0, %%ntid.x;" : "=r"(r.x));
  asm("mov.u32 %0, %%ntid.y;" : "=r"(r.y));
  asm("mov.u32 %0, %%ntid.z;" : "=r"(r.z));
  r.w = 0;
  return r;
}

extern "C" __device__ __noinline__ void
instrument_place(int32_t pkind, uint64_t channel_ptr) {
  const V5DevState *st = (const V5DevState *)channel_ptr;
  if (*((volatile bool *)&st->stop_report))
    return;
  const int4 cta = get_ctaid();
  if (!v5_cta_sampled(channel_ptr, cta))
    return;
  const uint32_t active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  if (laneid != __ffs(active_mask) - 1)
    return;
  V5PlaceRec *buf = st->place_buf;
  if (buf == nullptr)
    return;
  const int4 nct = get_nctaid();
  const int4 tid = v5_tid_reg();
  const int4 ntid = v5_ntid_reg();
  const uint32_t cta_lin =
      ((uint32_t)cta.z * (uint32_t)nct.y + (uint32_t)cta.y) * (uint32_t)nct.x +
      (uint32_t)cta.x;
  const uint32_t warp_in_cta =
      ((uint32_t)tid.z * (uint32_t)ntid.y * (uint32_t)ntid.x +
       (uint32_t)tid.y * (uint32_t)ntid.x + (uint32_t)tid.x) / 32u;
  const uint32_t slot =
      (cta_lin * st->place_warps_per_cta + warp_in_cta) * 2u + (uint32_t)pkind;
  if (slot >= st->place_capacity)
    return;
  V5PlaceRec r;
  r.globaltimer = v5_globaltimer();
  r.clock64 = clock64();
  r.smid = (uint32_t)get_smid();
  r.warpid = (uint32_t)get_warpid();
  r.cta_x = (uint32_t)cta.x;
  r.cta_y = (uint32_t)cta.y;
  r.cta_z = (uint32_t)cta.z;
  r.warp = warp_in_cta;
  r.valid = 1u;
  r.pad = 0u;
  /* a predicated-off EXIT site also fires (no guard on the call); the real
   * exit comes later and overwrites the slot */
  buf[slot] = r;
}

extern "C" __device__ __noinline__ void
instrument_inst(int pred, int32_t pred_flip, int32_t function_id, int32_t instr_idx,
                uint64_t channel_ptr,
                int32_t mref_count, uint64_t mref0, uint64_t mref1,
                uint64_t mref2, uint64_t mref3, uint64_t ops_meta,
                uint64_t ops_meta_hi, int32_t desc_low, int32_t desc_high, uint32_t v0, uint32_t v1,
                uint32_t v2, uint32_t v3, uint32_t v4, uint32_t v5,
                uint32_t v6, uint32_t v7, uint32_t v8, uint32_t v9,
                uint32_t v10, uint32_t v11, uint32_t v12, uint32_t v13,
                uint32_t v14, uint32_t v15) {
  /* stop_report (launches before the selected one run instrumented with
   * it set) is checked before any warp-collective ballot: a ballot in a
   * divergent region of an untraced kernel deadlocked
   * fusedBuildExpertMapsSortFirstTokenKernel (moe_gdn) in every run;
   * the predecessor's inject_funcs check it first for the same reason. */
  if (*((volatile bool *)&((V5DevState *)channel_ptr)->stop_report))
    return;
  const uint32_t active_mask = __ballot_sync(__activemask(), 1);
  /* pred_flip: 1 for a negated uniform guard (@!UPx), whose raw UPx value
   * arrives through nvbit_add_call_arg_upred_val_at (see tracer_tool.cu). */
  pred = (pred != 0) != (pred_flip != 0);
  const uint32_t exec_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  const int4 cta = get_ctaid();
  if (!v5_cta_sampled(channel_ptr, cta))
    return;

  ChannelDev *channel_dev = (ChannelDev *)channel_ptr;
  const uint64_t mref_addrs[V5_MAX_MREF] = {mref0, mref1, mref2, mref3};
  const uint32_t values[V5_MAX_OPS] = {v0, v1, v2,  v3,  v4,  v5,  v6,  v7,
                                       v8, v9, v10, v11, v12, v13, v14, v15};
  /* BRANK entries take a second dynamic at values[i + 1]: the layout
   * guarantees a BRVAL aux entry (kind PRED, zero-width tail) follows. */
  const int exec_count = __popc(exec_mask);

  /* head chunk */
  {
    struct {
      V5ChunkHead ch;
      V5HeadBody body;
    } hc;
    v5_chunk_head(&hc.ch, V5P_HEAD, V5_KIND_INSTR, cta, function_id, instr_idx);
    if (first_laneid == laneid) {
      hc.body.cta_y = (uint16_t)cta.y;
      hc.body.cta_z = (uint16_t)cta.z;
      hc.body.sm = (uint8_t)get_smid();
      hc.body.pad = 0;
      hc.body.active_mask = active_mask;
      hc.body.exec_mask = exec_mask;
      hc.body.reserved = (uint32_t)get_warpid();
      channel_dev->push(&hc, sizeof(hc));
    }
  }

  /* mref chunks: one per MREF operand */
  for (int m = 0; m < mref_count && m < V5_MAX_MREF; ++m) {
    struct {
      V5ChunkHead ch;
      V5MrefBody body;
    } mc2;
    v5_chunk_head(&mc2.ch, V5P_MREF, 0, cta, function_id, instr_idx);
    mc2.body.mref = (uint8_t)m;
    int slot = 0;
    for (int lane = 0; lane < 32; ++lane) {
      if (!(exec_mask & (1u << lane)))
        continue;
      const uint64_t addr = __shfl_sync(active_mask, mref_addrs[m], lane);
      if (first_laneid == laneid)
        mc2.body.addrs[slot++] = addr;
    }
    if (first_laneid == laneid)
      channel_dev->push(&mc2, sizeof(mc2.ch) + 8 + (uint32_t)exec_count * 8);
  }

  /* operand-entry chunks */
  for (int i = 0; i < V5_MAX_OPS; ++i) {
    const uint64_t meta = i < 8 ? ops_meta : ops_meta_hi;
    const int si = i < 8 ? i : i - 8;
    const uint32_t kind = (uint32_t)V5_OPMETA_KIND(meta, si);
    if (kind == 0)
      break;
    if (kind == V5_OP_PRED)
      continue;
    struct {
      V5ChunkHead ch;
      V5OpBody body;
    } oc;
    v5_chunk_head(&oc.ch, V5P_OP, 0, cta, function_id, instr_idx);
    oc.body.entry = (uint8_t)i;
    uint32_t body_bytes = 0;
    if (kind == V5_OP_R) {
      int slot = 0;
      for (int lane = 0; lane < 32; ++lane) {
        if (!(exec_mask & (1u << lane)))
          continue;
        const uint32_t value = __shfl_sync(active_mask, values[i], lane);
        if (first_laneid == laneid)
          oc.body.vals[slot++] = value;
      }
      body_bytes = (uint32_t)exec_count * 4;
    } else if (kind == V5_OP_UR || kind == V5_OP_CPOL ||
               kind == V5_OP_CPOLH) {
      if (first_laneid == laneid)
        oc.body.vals[0] = values[i]; /* UR values are warp-uniform */
      body_bytes = 4;
    } else if (kind == V5_OP_SPACE) {
      /* isspacep ballots of this lane's own mref0 address over the exec
       * lanes: generic/shared::cta/shared::cluster (sm_90+). */
      int is_global = 0, is_shared = 0, is_cluster = 0;
      if (pred) {
        const uint64_t addr = mref_addrs[0];
        asm volatile("{ .reg .pred pg, ps;\n\t"
                     "isspacep.global pg, %2;\n\t"
                     "isspacep.shared ps, %2;\n\t"
                     "selp.u32 %0, 1, 0, pg;\n\t"
                     "selp.u32 %1, 1, 0, ps; }"
                     : "=&r"(is_global), "=&r"(is_shared)
                     : "l"(addr));
        asm volatile("{ .reg .pred pcl;\n\t"
                     "isspacep.shared::cluster pcl, %1;\n\t"
                     "selp.u32 %0, 1, 0, pcl; }"
                     : "=&r"(is_cluster)
                     : "l"(addr));
      }
      const uint32_t gballot =
          __ballot_sync(active_mask, pred && is_global);
      const uint32_t sballot =
          __ballot_sync(active_mask, pred && is_shared);
      const uint32_t cballot =
          __ballot_sync(active_mask, pred && is_cluster);
      if (first_laneid == laneid) {
        oc.body.vals[0] = gballot;
        oc.body.vals[1] = sballot;
        oc.body.vals[2] = cballot;
      }
      body_bytes = 12;
    } else if (kind == V5_OP_RANK) {
      const int m = (int)V5_OPMETA_AUX(meta, si);
      int rank = -1;
      if (pred && m >= 0 && m < mref_count && m < V5_MAX_MREF) {
        const uint64_t addr = mref_addrs[m];
        /* getctarank.shared::cluster is only architecturally defined for
         * a shared::cta window operand; anything else is an unavailable
         * fact (-1), never a wild probe. */
        int is_shared = 0;
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "isspacep.shared::cta p, %1;\n\t"
                     "selp.b32 %0, 1, 0, p;\n\t"
                     "}"
                     : "=r"(is_shared)
                     : "r"((uint32_t)addr));
        if (is_shared)
          asm volatile("getctarank.shared::cluster.u32 %0, %1;"
                       : "=r"(rank)
                       : "r"((uint32_t)addr));
      }
      int slot = 0;
      for (int lane = 0; lane < 32; ++lane) {
        if (!(exec_mask & (1u << lane)))
          continue;
        const int32_t lane_rank = __shfl_sync(active_mask, rank, lane);
        if (first_laneid == laneid)
          ((int32_t *)oc.body.vals)[slot++] = lane_rank;
      }
      body_bytes = (uint32_t)exec_count * 4;
    } else if (kind == V5_OP_URANK) {
      /* Owner rank of the barrier/companion UR: v4's rule -- getctarank on
       * the captured UREG operand word, executed by the leader lane only,
       * no collectives.  The operand is architecturally a shared::cluster
       * address for the opcodes that carry URANK entries; v4 probed the
       * operand word unguarded the same way.  -1 marks an unavailable
       * fact.  (No mbarrier state read: a generic ld of the object returns
       * stale words -- v5.1 tried it, r4 read 0 on live barriers.) */
      int rank = -1;
      if (first_laneid == laneid && pred)
        asm volatile("getctarank.shared::cluster.u32 %0, %1;"
                     : "=r"(rank)
                     : "r"(values[i] + values[i + 1]));
      if (first_laneid == laneid)
        ((int32_t *)oc.body.vals)[0] = rank;
      body_bytes = 4;
    } else if (kind == V5_OP_BRANK) {
      /* [Rx+URy+disp] operand word: per-lane getctarank on each lane's own
       * composed 32-bit shared::cluster address (v4's sb_addr rule).  A
       * RED arrive can arm a different CTA replica per lane, so v5.3
       * keeps every executing lane's rank; predicated-off lanes are -1.
       * The shfl collect stays outside any per-lane branch. */
      int rank = -1;
      if (pred)
        asm volatile("getctarank.shared::cluster.u32 %0, %1;"
                     : "=r"(rank)
                     : "r"(values[i] + values[i + 1]));
      int slot = 0;
      for (int lane = 0; lane < 32; ++lane) {
        if (!(exec_mask & (1u << lane)))
          continue;
        const int32_t lane_rank = __shfl_sync(active_mask, rank, lane);
        if (first_laneid == laneid)
          ((int32_t *)oc.body.vals)[slot++] = lane_rank;
      }
      body_bytes = (uint32_t)exec_count * 4;
    } else if (kind == V5_OP_DESC) {
      uint32_t *image = oc.body.vals;
      bool ok = false;
      if (exec_mask != 0 && first_laneid == laneid) {
        const uint64_t ptr = (uint64_t)(uint32_t)desc_low |
                             ((uint64_t)(uint32_t)desc_high << 32);
        ok = ptr != 0 && v5_copy_desc_image((uint8_t *)(image + 1), ptr);
      }
      ok = __shfl_sync(active_mask, ok ? 1 : 0, first_laneid) != 0;
      if (first_laneid == laneid)
        image[0] = ok ? 0xFFFFFFFFu : 0;
      body_bytes = ok ? 4 + 128 : 4;
    }
    if (first_laneid == laneid)
      channel_dev->push(&oc, sizeof(oc.ch) + 4 + body_bytes);
  }
}

/* PHASECHK AFTER: one V5P_AFTER chunk carrying the dest-predicate ballot. */
extern "C" __device__ __noinline__ void
instrument_phasechk_after(int pred, int32_t pred_flip, int dest_pred,
                          int32_t function_id,
                          int32_t instr_idx, uint64_t channel_ptr) {
  if (*((volatile bool *)&((V5DevState *)channel_ptr)->stop_report))
    return;
  const uint32_t active_mask = __ballot_sync(__activemask(), 1);
  const int4 cta = get_ctaid();
  if (!v5_cta_sampled(channel_ptr, cta))
    return;
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  const uint32_t ready = __ballot_sync(active_mask, dest_pred);
  if (first_laneid == laneid) {
    struct {
      V5ChunkHead ch;
      V5AfterBody body;
    } ac;
    v5_chunk_head(&ac.ch, V5P_AFTER, V5_KIND_AFTER, cta, function_id, instr_idx);
    ac.body.cta_y = (uint16_t)cta.y;
    ac.body.cta_z = (uint16_t)cta.z;
    ac.body.sm = (uint8_t)get_smid();
    ac.body.pad = 0;
    ac.body.active_mask = active_mask;
    ac.body.ready_mask = ready;
    ((ChannelDev *)channel_ptr)->push(&ac, sizeof(ac));
  }
}

/* v5.4 global-sync values (V5P_GSYNC): phase 0 carries each executing
 * lane's source operand words followed by the optional pre_value words --
 * the lane's own mref0 address read right before the instruction
 * executes; phase 1 carries the destination register words.  Same
 * stop_report / sampling / active/exec-mask rules as instrument_inst. */
extern "C" __device__ __noinline__ void
instrument_gsync(int pred, int32_t pred_flip, int32_t function_id,
                 int32_t instr_idx, uint64_t channel_ptr, int32_t phase,
                 int32_t nwords, int32_t flags, int32_t pre_bytes,
                 uint64_t mref0, uint32_t v0, uint32_t v1, uint32_t v2,
                 uint32_t v3, uint32_t v4, uint32_t v5, uint32_t v6,
                 uint32_t v7) {
  if (*((volatile bool *)&((V5DevState *)channel_ptr)->stop_report))
    return;
  const uint32_t active_mask = __ballot_sync(__activemask(), 1);
  pred = (pred != 0) != (pred_flip != 0);
  const uint32_t exec_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  const int4 cta = get_ctaid();
  if (!v5_cta_sampled(channel_ptr, cta))
    return;
  if (exec_mask == 0 || nwords <= 0 || nwords > V5_GSYNC_MAX_WORDS)
    return;

  uint32_t lane_words[V5_GSYNC_MAX_WORDS] = {v0, v1, v2, v3,
                                             v4, v5, v6, v7};
  /* pre_value lives at the tail of the lane's word sequence (after the
   * nwords - pre_words source words), read only on pred-true lanes. */
  if (phase == 0 && (flags & 1) && pre_bytes > 0 && pred) {
    const int pre_words = pre_bytes / 4;
    if (pre_bytes == 4) {
      uint32_t value32 = 0;
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];"
                   : "=r"(value32)
                   : "l"(mref0)
                   : "memory");
      lane_words[nwords - pre_words] = value32;
    } else if (pre_bytes == 8) {
      uint64_t value64 = 0;
      asm volatile("ld.relaxed.gpu.u64 %0, [%1];"
                   : "=l"(value64)
                   : "l"(mref0)
                   : "memory");
      lane_words[nwords - 2] = (uint32_t)value64;
      lane_words[nwords - 1] = (uint32_t)(value64 >> 32);
    }
  }

  struct {
    V5ChunkHead ch;
    V5GsyncBody body;
  } gc;
  v5_chunk_head(&gc.ch, V5P_GSYNC, V5_KIND_GSYNC, cta, function_id, instr_idx);
  gc.body.cta_y = (uint16_t)cta.y;
  gc.body.cta_z = (uint16_t)cta.z;
  gc.body.sm = (uint8_t)get_smid();
  gc.body.phase = (uint8_t)phase;
  gc.body.nwords = (uint8_t)nwords;
  gc.body.flags = (uint8_t)flags;
  gc.body.mask = exec_mask;
  int slot = 0;
  for (int lane = 0; lane < 32; ++lane) {
    if (!(exec_mask & (1u << lane)))
      continue;
    for (int w = 0; w < nwords; ++w) {
      const uint32_t value = __shfl_sync(active_mask, lane_words[w], lane);
      if (first_laneid == laneid)
        gc.body.vals[slot * nwords + w] = value;
    }
    ++slot;
  }
  if (first_laneid == laneid)
    ((ChannelDev *)channel_ptr)
        ->push(&gc, sizeof(gc.ch) + 12 +
                        (uint32_t)__popc(exec_mask) * (uint32_t)nwords * 4);
}

/* CTA info record (v6.1): every warp of a sampled launch pushes one
 * V5P_CTA chunk at its instrumented function's first instruction; the
 * lead lane writes the %cluster* register values verbatim.  All warps
 * execute the call (no predication); the lead lane pushes. */
extern "C" __device__ __noinline__ void
instrument_cta_info(int32_t function_id, uint64_t channel_ptr) {
  const V5DevState *st = (const V5DevState *)channel_ptr;
  if (*((volatile bool *)&st->stop_report))
    return;
  const int4 cta = get_ctaid();
  if (!v5_cta_sampled(channel_ptr, cta))
    return;
  const uint32_t active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  if (laneid != __ffs(active_mask) - 1)
    return;
  struct {
    V5ChunkHead ch;
    V5CtaBody body;
  } cc;
  v5_chunk_head(&cc.ch, V5P_CTA, V5_KIND_CTA, cta, function_id, 0);
  uint32_t v;
#define V5_RD(reg)   asm volatile("mov.u32 %0, %" reg ";" : "=r"(v)); cc.body
  V5_RD("cluster_ctarank").cluster_ctarank = v;
  V5_RD("cluster_nctaid.x").cluster_nctaid_x = v;
  V5_RD("cluster_nctaid.y").cluster_nctaid_y = v;
  V5_RD("cluster_nctaid.z").cluster_nctaid_z = v;
  V5_RD("clusterid.x").clusterid_x = v;
  V5_RD("clusterid.y").clusterid_y = v;
  V5_RD("clusterid.z").clusterid_z = v;
  V5_RD("nclusterid.x").nclusterid_x = v;
  V5_RD("nclusterid.y").nclusterid_y = v;
  V5_RD("nclusterid.z").nclusterid_z = v;
#undef V5_RD
  ((ChannelDev *)channel_ptr)
      ->push(&cc, sizeof(cc.ch) + sizeof(cc.body));
}

/* Prior-launch L2 footprint: one ring entry per touched 32-byte sector
 * per warp-executed MREF while the host holds cell->active (launches
 * before the selected one); it deliberately has no sampling gate -- the
 * footprint covers every CTA of the launch.  Entry encoding:
 *   u64 = (sector_address << 8) | ((get_smid() & 0xFF) << 1) | is_store
 * with sector_address = addr & ~31.  Warp-aggregate write path: ballot
 * the executing lanes (generic-space MREFs additionally filter
 * non-global addresses per lane), __match_any picks one leader lane per
 * distinct sector, the lowest leader reserves n slots with a single
 * atomicAdd, and each leader writes its own slot at (base + rank) &
 * mask.  The ring can wrap; the host recovers the newest entries in
 * order. */
extern "C" __device__ __noinline__ void
instrument_l2_footprint(int pred, uint64_t mref_addr, int32_t is_store,
                        int32_t is_generic, uint64_t pcell) {
  footprint_cell_t *cell = reinterpret_cast<footprint_cell_t *>(pcell);
  if (!cell->active)
    return;
  const unsigned active_mask = __activemask();
  const int laneid = get_laneid();
  bool mine = pred != 0;
  if (is_generic && mine)
    mine = __isGlobal(reinterpret_cast<const void *>(mref_addr));
  const unsigned exec_mask = __ballot_sync(active_mask, mine);
  if (exec_mask == 0 || ((exec_mask >> laneid) & 1u) == 0)
    return;

  const uint64_t sector = mref_addr & ~31ull;
  const unsigned peers = __match_any_sync(exec_mask, sector);
  const bool is_leader = (__ffs(peers) - 1) == laneid;
  const unsigned leader_mask = __ballot_sync(exec_mask, is_leader);
  const unsigned n_leaders = __popc(leader_mask);
  const int first_leader = __ffs(leader_mask) - 1;
  unsigned long long base = 0;
  if (laneid == first_leader)
    base = atomicAdd(cell->counter, (unsigned long long)n_leaders);
  base = __shfl_sync(exec_mask, base, first_leader);
  if (is_leader) {
    const unsigned rank = __popc(leader_mask & ((1u << laneid) - 1u));
    cell->ring[(base + rank) & cell->mask] =
        (sector << 8) | (((uint64_t)get_smid() & 0xFFull) << 1) |
        (is_store ? 1ull : 0ull);
  }
}

/* Collective-free operand stash (see tracer_tool.cu insert_stash_call):
 * each lane stores its operand words for one instruction of a
 * WARPSYNC.COLLECTIVE region under its (SM, hardware warp) slot, which is
 * unique among resident warps; the record is emitted after ENDCOLLECTIVE. */
__device__ uint32_t v5_stash[V5_STASH_SMS][V5_STASH_WARPS][V5_STASH_SLOTS]
                            [V5_MAX_OPS][32];

extern "C" __device__ __noinline__ void
stash_ops(int32_t slot, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
          uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7, uint32_t v8,
          uint32_t v9, uint32_t v10, uint32_t v11, uint32_t v12,
          uint32_t v13, uint32_t v14, uint32_t v15) {
  const unsigned sm = get_smid(), warp = get_warpid();
  if (sm >= V5_STASH_SMS || warp >= V5_STASH_WARPS || slot < 0 ||
      slot >= V5_STASH_SLOTS)
    return;
  const int lane = get_laneid();
  const uint32_t values[V5_MAX_OPS] = {v0, v1, v2,  v3,  v4,  v5,  v6,  v7,
                                       v8, v9, v10, v11, v12, v13, v14, v15};
  for (int i = 0; i < V5_MAX_OPS; ++i)
    v5_stash[sm][warp][slot][i][lane] = values[i];
}

extern "C" __device__ __noinline__ void
instrument_inst_stashed(int pred, int32_t pred_flip, int32_t function_id,
                        int32_t instr_idx, uint64_t channel_ptr,
                        int32_t mref_count, uint64_t mref0, uint64_t mref1,
                        uint64_t mref2, uint64_t mref3, uint64_t ops_meta,
                        uint64_t ops_meta_hi, int32_t desc_low,
                        int32_t desc_high, int32_t slot) {
  const unsigned sm = get_smid(), warp = get_warpid();
  uint32_t v[V5_MAX_OPS] = {};
  if (sm < V5_STASH_SMS && warp < V5_STASH_WARPS && slot >= 0 &&
      slot < V5_STASH_SLOTS) {
    const int lane = get_laneid();
    for (int i = 0; i < V5_MAX_OPS; ++i)
      v[i] = v5_stash[sm][warp][slot][i][lane];
  }
  instrument_inst(pred, pred_flip, function_id, instr_idx, channel_ptr, mref_count,
                  mref0, mref1, mref2, mref3, ops_meta, ops_meta_hi, desc_low,
                  desc_high, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                  v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
}
