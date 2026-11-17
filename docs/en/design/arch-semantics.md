# Hopper and Blackwell: what differs

[中文](../../zh/design/arch-semantics.md)

One pipeline serves both architectures. Architecture differences enter
in exactly four places: the tracer's arch table, the conversion rules,
the mechanism tree, and the engine's arch module. Everything else —
the record stream, the segment format, the simulator core — is
shared, and `cpp/core/` contains no device branches.

## 1. Where the difference lives

| layer | shared part | arch part |
|---|---|---|
| capture | `tools/tracer/tracer.cu`, `record_format.h` | the arch table (`arch_table.h`, `arch_hopper.h`, `arch_blackwell.h`): operand families, mbarrier-word rules, TMA coordinates, descriptor tokens |
| convert | `cpp/{capture,convert}/`, `python/cyclescope/convert/` | `cpp/arch/<arch>/{capture_rules,convert_rules}.cpp`, `python/cyclescope/arch/<arch>/` |
| mechanisms | `configs/mechanism_tree/common/` | `configs/mechanism_tree/{hopper,blackwell}/`, generated as `configs/trees/{h100,b300}.json` |
| engine | `cpp/core/` | `cpp/arch/{hopper,blackwell}/` behind `architecture_module.hpp` |

At run time the launch's `LNCH.device` picks the module
(`cpp/arch/registry.cpp`) and the tree the job was built with.

## 2. Shared semantics

Handled once, for both: mbarrier init/arrive/expect/wait phases, CTA
named barriers and the cluster barrier, TMA and bulk copies,
`cp.async` groups, generic memory accesses, and global-memory
synchronization. Their `SEMA` kinds and the engine state machines are
arch-free ([engine.md](engine.md) §5).

## 3. Hopper: warpgroups and WGMMA

Hopper's tensor core is driven by a **warpgroup** — four warps acting
as one collective — and returns results in registers. Conversion maps
the instructions onto semantic kinds
(`python/cyclescope/arch/hopper/semantics.py`):

| instruction | semantic kind |
|---|---|
| `WARPGROUP.ARRIVE` | `warpgroup_arrive` |
| `HGMMA` / `QGMMA` / `IGMMA` / `WGMMA` | `wgmma` (shape, dtypes, ss/rs source mode, accumulator registers) |
| `WARPGROUP.DEPBAR.LE`, `WARPGROUP.WAIT` | `warpgroup_wait` |
| `FENCE.VIEW.ASYNC.S` | `descriptor_visibility` (proxy fence) |

In the engine (`cpp/arch/hopper/`), the module runs the collective
launch rendezvous (all four warps must arrive before the op launches
onto the shared tensor pipe), the group FIFO on the warpgroup lane of
`AsyncGroups`, the accumulator handoff to the instructions that
consume the result, and the register-file issue steal while an op is
in flight. `proxy_fence` models the async-proxy drain a fence waits
for.

## 4. Blackwell: UMMA, tensor memory, CLC

Blackwell's tensor core is issued by a single thread and accumulates
in **tensor memory** (TMEM), a per-SM column budget CTAs allocate
from. Conversion maps
(`python/cyclescope/arch/blackwell/semantics.py`):

| instruction | semantic kind |
|---|---|
| `UTCHMMA` / `UTCQMMA` / `UTCOMMA` | `umma_issue` (kind, cta_group, A source, shape, descriptor) |
| `UTCBAR` | `umma_commit` (object, cta_group, mask, scope) |
| `UTCATOMSWS` (`FIND_AND_SET` / `AND` / `RELINQUISH` forms) | `tmem_alloc` / `tmem_dealloc` / `tmem_relinquish` |
| `LDTM` / `STTM` / `UTCCP` | `tmem_load` / `tmem_store` / `tmem_copy` |
| `FENCE.VIEW.ASYNC.T` | `tmem_wait_store` |
| `UGETNEXTWORKID` | `clc_query` |
| `STAS` | `shared_store_async` |

In the engine (`cpp/arch/blackwell/`), the module runs the UMMA pipe
beats and commit arrives (a `cta_group::2` op posts on the pair
leader), the TMEM column budget — an alloc blocks when the budget is
spent and a dealloc frees and wakes — TMEM load/store/copy timing,
and CLC query completion through its mbarrier.
