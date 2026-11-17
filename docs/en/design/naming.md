# Terms and naming rules

[中文](../../zh/design/naming.md)

One word per concept, everywhere: code, formats, configs and docs.
This page is the glossary that ties the concept words used across the
design notes to the names the code and the binary formats actually
use, plus the naming rules the code follows.

## 1. Glossary

| concept | name in code / formats |
|---|---|
| thread block | `CTA` (CTA id, `cta_index`, `%cluster*` ranks) |
| group of thread blocks that is captured and converted together | `cluster` → one `GSIMSEG` segment, `cluster_id` |
| everything belonging to one kernel launch | `launch`, `LNCH` row, container `GSIMJOB` (`job.bin`) |
| the tracer's per-instruction record | `record` in `trace-<launch>.bin.zst` |
| per-cluster converted facts | `GSIMCAP` facts file (`*.gsc`) + `launch.json` |
| a static instruction as the compiler wrote it | `StaticInstruction`; in a job, `TMPL` + `REGS` |
| one execution of it | `InstructionInstance`, `INST` |
| what an instance does to a barrier/group/copy | `InstanceSemantics`, `SEMA` kind + fields |
| the sectors a memory access touches | `MACC` / `MSEC` rows |
| a mechanism binding for one instance | `USES` row + `FACT` axis values |
| a leaf pick for an event | `TIME` row (kind) / `LEAF` + `PARM` |
| mechanism / axis / leaf | `PROG` program, `AXIS` + `GRD ` guard, `LEAF` |
| CTA→SM assignment | `placement`, `PLCE` row |
| SM interconnect layout | `sm topology`, `TOPO` rows |
| cross-CTA sync through global memory | `global sync`, `GSYN` rows |
| engine output | `result` `GSIMRES`, `issue journal` `GSIMJNL`, `wait ledger` `GSIMWLG` |
| failure classes | `missing_fact:<fact> (<where>)`, `unsupported:<reason> (<where>)` |
| leaf evidence | `measured` / `experimental` / `structural` / `placeholder` |
| leaf ownership | `cluster-local` / `gpu-shared` ([mechanism-tree.md](mechanism-tree.md) §4) |
| reference condition | `reference_sm_clock_khz`, `reference_cache_control` |

## 2. Naming rules the code follows

- English identifiers; documents `lower-kebab-case.md`; Python and C++
  files `snake_case`; JSON configs `lower_snake_case.json`.
- Python: modules and functions `snake_case`, classes `CapWords`,
  constants `UPPER_SNAKE`; package `cyclescope`.
- C++: types `CapWords`, functions and variables `snake_case`,
  constants `kCapWords`; namespaces `cyclescope` (with
  `cyclescope::core` for the generated parameter fields) and
  `cyclescope_convert` (capture/convert side).
- Table tags are four characters of upper-case ASCII (`TMPL`, `SEMA`,
  `GRD `); file magics are `GSIM` + class + version (`GSIMSEG`,
  `GSIMJOB`, `GSIMRES`, `GSIMCAP01`, `GSIMPRG0`).
- CLI: one entry `cyclescope <verb>`; binaries
  `cyclescope-{convert,capture-facts,trace-split,engine}`; environment
  variables `CYCLESCOPE_*`.
- Field names say units: `*_cycles`, `*_ns`, `*_bytes`,
  `*_per_cycle`, `*_per_ns`, `*_count`, `*_index`; booleans in
  positive form (`enabled`, `is_*`, `has_*`).
- A name says what a thing is, never where it came from: no dates,
  round numbers, person or case names, and no invented abbreviations.
