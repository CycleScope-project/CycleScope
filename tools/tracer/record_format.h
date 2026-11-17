/* Shared record/device-state layout of the merged tracer (capture spec
 * docs/en/design/capture.md).
 *
 * On the device->host channel every executed instruction emits one HEAD
 * chunk followed by one chunk per serialized part (each MREF, each operand
 * entry, the PHASECHK AFTER fact).  Every chunk carries a self-identifying
 * 24-byte V5ChunkHead; chunks of one record keep channel order per
 * (cta, warp) producer.  The host reassembles chunks into the variable
 * formats.md record layout (a 32-byte V5Head plus contiguous tail) before
 * writing trace-<launch>.bin.zst.
 *
 * Chunk bodies are written with fixed-stride indexed stores only -- a
 * moving byte pointer over a multi-KB local buffer makes ptxas explode
 * (33 GB, SIGKILL) on sm_103a.
 */
#include <stdint.h>

#define V5_MAX_MREF 4
#define V5_STASH_SLOTS 4
#define V5_STASH_SMS 160
#define V5_STASH_WARPS 64
#define V5_MAX_OPS 16
#define V5_MAX_SAMPLED_CLUSTERS 1024

/* kind values for RecHead.kind and V5P_HEAD chunk bodies */
#define V5_KIND_INSTR 0 /* BEFORE record */
#define V5_KIND_AFTER 1 /* PHASECHK AFTER record (tail = u32 ready_mask) */
#define V5_KIND_GSYNC 2 /* global-sync value record (V5P_GSYNC) */
#define V5_KIND_CTA 3   /* per-warp CTA info record (V5P_CTA): cluster
                         * membership facts read from %cluster* registers */

/* operand-entry kinds inside the serialized operand block; the host keeps
 * one OpsLayout per (reg, idx) so it can compute every tail length. */
#define V5_OP_R 1    /* per-exec-lane u32 */
#define V5_OP_UR 2   /* single u32 */
#define V5_OP_RANK 3 /* per-exec-lane i32 (getctarank of that lane's mref) */
#define V5_OP_DESC 4 /* u32 marker: 0 = no image; 0xFFFFFFFF then 128B image */
#define V5_OP_PRED 5 /* emits nothing in the BEFORE tail (AFTER carries it) */
#define V5_OP_URANK 6 /* one i32: getctarank.shared::cluster of a UR value */
#define V5_OP_BRANK 7 /* per-exec-lane i32: getctarank of [Rx+URy+disp] */
#define V5_OP_SPACE 8 /* 3 x u32: isspacep global/shared::cta/shared::cluster
                       * ballots over exec lanes of mref0 (GENERIC-space
                       * instructions with at least one MREF) */
#define V5_OP_CPOL 9  /* one u32: cache-policy UR low word (desc[URx]) */
#define V5_OP_CPOLH 10 /* one u32: cache-policy UR high word (desc[URx+1]) */

/* Per-operand-entry descriptor packed into the ops_meta const argument:
 * 8 bytes, one per entry (V5_MAX_OPS slots):
 *   bits[3:0] kind, bits[6:4] aux (RANK: mref index; others unused),
 *   bit[7] entry consumes one value arg */
#define V5_OPMETA_KIND(meta, i) (((meta) >> (8ull * (i))) & 0xFull)
#define V5_OPMETA_AUX(meta, i) (((meta) >> (8ull * (i) + 4)) & 0x7ull)

/* chunk part tags */
#define V5P_HEAD 0
#define V5P_MREF 1 /* body: u8 mref_index + exec*8 bytes of addresses */
#define V5P_OP 2   /* body: u8 entry_index + entry bytes per layout */
#define V5P_AFTER 3 /* PHASECHK AFTER: body = u8 cta_y,cta_z,sm + masks+ready */
/* v6 placement record: lead lane of each warp, at function entry and at
 * each EXIT.  ch->kind = 0 entry / 1 exit, ch->idx = launch ordinal. */
#define V5P_PLACE 4 /* body: V5PlaceBody (fixed size) */
/* v5.4 global-sync values: instrument_gsync pushes one chunk right after
 * its instruction's BEFORE record parts (the call is inserted behind the
 * instrument_inst BEFORE call, so the chunk lands after the HEAD/MREF/OP
 * chunks of the same instruction). */
#define V5P_GSYNC 5 /* body: V5GsyncBody (variable pushed prefix) */
#define V5_GSYNC_MAX_WORDS 8 /* max u32 words per lane, BEFORE and AFTER */
/* CTA info record (v6.1): instrument_cta_info at every instrumented
 * function's first instruction pushes one chunk per warp (lead lane).
 * The host writes one kind-3 file record per warp; readers dedupe per
 * CTA.  Emitted for every CTA of a sampled launch regardless of the
 * cluster-sampling gate (the converter needs the membership of every
 * CTA it sees; the gate keeps the instruction stream unchanged). */
#define V5P_CTA 6 /* body: V5CtaBody (fixed 40 bytes) */

/* v5.1: the reassembly key carries the full CTA id.  Without cta_y/cta_z
 * in the chunk head, records from different CTAs sharing (cta_x, warp)
 * interleaved into one record on grids whose y/z extents exceed 1
 * (CU-72A trace-1, W3PROBE trace-3). */
struct V5ChunkHead {
  uint32_t cta_x;
  uint16_t cta_y, cta_z;
  uint16_t warp; /* warp id within CTA */
  uint8_t part;
  uint8_t kind; /* V5_KIND_* (for part==V5P_HEAD; else 0) */
  uint32_t function;
  uint32_t idx;
  uint32_t pad;
}; /* 24 bytes */
static_assert(sizeof(V5ChunkHead) == 24, "V5ChunkHead must stay 24 bytes");

/* 32-byte file-record head (formats.md): the host writes this. */
struct V5Head {
  uint32_t cta_x;
  uint16_t cta_y, cta_z;
  uint16_t warp;
  uint8_t sm;
  uint8_t kind;
  uint32_t function;
  uint32_t idx;
  uint32_t active_mask;
  uint32_t exec_mask;
  uint32_t reserved; /* %warpid (SM-global warp id), sampled by leader lane */
};

/* part==V5P_HEAD body: the remaining head fields. */
struct V5HeadBody {
  uint16_t cta_y, cta_z;
  uint8_t sm;
  uint8_t pad;
  uint32_t active_mask;
  uint32_t exec_mask;
  uint32_t reserved;
}; /* 16 bytes; chunk = 32B total */

/* part==V5P_AFTER body */
struct V5AfterBody {
  uint16_t cta_y, cta_z;
  uint8_t sm;
  uint8_t pad;
  uint32_t active_mask;
  uint32_t ready_mask;
}; /* 12 bytes; chunk = 28B total */

struct V5MrefBody {
  uint8_t mref;
  uint8_t pad[7];
  uint64_t addrs[32];
}; /* 264 bytes; pushed prefix = 8 + exec*8 */

struct V5OpBody {
  uint8_t entry;
  uint8_t pad[3];
  uint32_t vals[32 + 32]; /* entry values; DESC image uses +32 words */
}; /* 264 bytes; pushed prefix = 4 + entry bytes */

/* part==V5P_GSYNC body (v5.4).  vals[slot * nwords + w]: lanes with a
 * value in ascending lane order, nwords u32 words each -- phase 0 carries
 * the source operand words followed by the optional pre_value words
 * (flags bit0), phase 1 the destination register words. */
struct V5GsyncBody {
  uint16_t cta_y, cta_z;
  uint8_t sm;    /* get_smid() */
  uint8_t phase; /* 0 before / 1 after */
  uint8_t nwords;
  uint8_t flags; /* bit0: pre_value words appended after the sources */
  uint32_t mask; /* lanes carrying values = exec_mask */
  uint32_t vals[32 * V5_GSYNC_MAX_WORDS];
}; /* pushed prefix = 12 + popcount(mask)*nwords*4 */

/* part==V5P_PLACE body (v6): per-warp placement fact, fixed size. */
/* part==V5P_CTA body (v6.1): the launch's actual cluster shape and this
 * CTA's membership, read from the %cluster* special registers.  For
 * non-cluster launches every field is 1/0 accordingly (ctarank 0,
 * nctaid/grid dims, clusterid = ctaid, nclusterid = grid dims). */
struct V5CtaBody {
  uint32_t cluster_ctarank;  /* %cluster_ctarank */
  uint32_t cluster_nctaid_x; /* %cluster_nctaid.x (CTAs per cluster) */
  uint32_t cluster_nctaid_y;
  uint32_t cluster_nctaid_z;
  uint32_t clusterid_x;      /* %clusterid.x (this CTA's cluster index) */
  uint32_t clusterid_y;
  uint32_t clusterid_z;
  uint32_t nclusterid_x;     /* %nclusterid.x (clusters per grid axis) */
  uint32_t nclusterid_y;
  uint32_t nclusterid_z;
}; /* 40 bytes; chunk = 64B total, file record = 32B head + 40B tail */

struct V5PlaceBody {
  uint64_t globaltimer; /* %globaltimer */
  uint64_t clock64;   /* clock64() */
  uint32_t smid;      /* %smid */
  uint32_t warpid;    /* %warpid */
}; /* 24 bytes; chunk = 48B total */

/* Host-side launch state the device checks cheaply before building a
 * record.  Laid out in __managed__ memory, updated by the launch hook. */
struct V5ClusterSampling {
  int32_t count; /* -1 = every cluster sampled */
  int32_t cdx, cdy, cdz; /* cluster dims (CTAs per cluster axis) */
  int32_t ncx, ncy;      /* cluster counts along grid x/y */
  uint32_t ids[V5_MAX_SAMPLED_CLUSTERS]; /* linear cluster ids */
};

/* All device-visible tool state lives in ONE __managed__ object defined in
 * tracer_tool.cu and reaches the injected functions through channel_ptr.
 * (v5.0-v5.2 declared these `static __managed__` here, so inject_funcs.cu
 * read its own never-written copy: count stayed -1 and every CTA was
 * traced.)  channel must stay the first member: the injected code casts
 * channel_ptr to both ChannelDev* and V5DevState*. */
/* v6 placement mode record (CAPTURE_MODE=placement).  Written with plain
 * stores into a device array indexed by (CTA linear id, warp in CTA, kind)
 * -- no channel push: 1000+ warps pushing through the ChannelDev lock at
 * kernel entry stretched the probed launch 4-8x (s0-smoke-r2). */
struct V5PlaceRec {
  uint64_t globaltimer; /* %globaltimer */
  uint64_t clock64;   /* clock64() */
  uint32_t smid;      /* %smid */
  uint32_t warpid;    /* %warpid */
  uint32_t cta_x, cta_y, cta_z;
  uint32_t warp;      /* warp index inside the CTA */
  uint32_t valid;     /* 1 once written */
  uint32_t pad;
}; /* 48 bytes */

struct V5DevState {
  ChannelDev channel;
  V5ClusterSampling sampling;
  bool stop_report;
  int32_t cur_launch; /* launch ordinal being captured (placement chunks
                       * carry it in ch->idx); written by the launch hook */
  /* v6 placement array for the launch being captured (device pointer,
   * capacity in records = CTAs * warps_per_cta * 2; slot = (cta_linear *
   * warps_per_cta + warp_in_cta) * 2 + kind). */
  V5PlaceRec *place_buf;
  uint32_t place_capacity;
  uint32_t place_warps_per_cta;
};

/* Prior-launch L2 footprint ring: while a launch before the selected one
 * runs, every global-sector touch appends one u64
 *   (sector_address << 8) | ((smid & 0xFF) << 1) | is_store
 * into a device ring.  The host drains the ring when the selected launch
 * is about to run.  The cell lives in __managed__ memory so the host
 * toggles `active` without extra copies; `ring`/`counter` are device
 * allocations. */
typedef struct {
  uint64_t *ring;
  unsigned long long *counter;
  uint64_t mask;      /* ring capacity - 1 (capacity is a power of two) */
  int32_t active;
} footprint_cell_t;
