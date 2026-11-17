# Job files: segments and the container

[中文](../../zh/design/job-format.md)

A **segment** packages one cluster's time-independent facts together
with the mechanism programs that will time them. A **container**
(`job.bin`) packages one launch: its segments plus the launch-level
tables. The engine reads a container and nothing else.

## 1. One file shell

Both formats — and the result files — share one shell: a fixed
header, then fixed-width record tables located through a table
directory of `(tag, record size, count, offset)` entries. Unknown
tags are ignored, so tables can be added without breaking readers.
Byte layouts are the single source of truth in
[../formats.md](../formats.md); this page keeps to what each table is
*for*.

## 2. What a segment holds

`GSIMSEG` (one cluster), grouped by role:

| group | tables | contents |
|---|---|---|
| identity | `META`, `STRS` | cluster id, counts, and three digests: of the mechanism tree, the compiled programs and the converter |
| resident structure | `CTAS`, `WARP` | the cluster's CTAs with their ranks and SM ids, and each warp's slot and instruction span |
| instruction streams | `TMPL`, `INST`, `REGS`, `SEMA`, `SPLT`, `CLKM` | per template and per instance: what ran, its registers, and what it does to barriers and copies |
| memory | `MACC`, `MSEC` | the sectors each access touches |
| timing | `TIME`, `USES`, `FACT` | timing rows and the mechanism bindings with their captured axis values |
| barrier bookkeeping | `MBPI`, `MBOL`, `MBOB` | pre-initialized barriers, object lists for multi-target instructions, per-object owner CTAs |
| observations | `GSYN`, `CODE` | global-sync value observations and code addresses |

The mechanism programs a segment can use travel beside it: the seven
shared tables (`PROG`, `NODE`, `GRD `, `LEAF`, `PARM`, `PTS `,
`AXIS`) live once per segments directory in `program.tables`
(`GSIMPRG0`), and readers splice them in. A segment therefore stays
small, and one launch shares one program image.

## 3. What the container adds

`GSIMJOB` (one launch) adds the rows that only make sense at launch
scope:

| table | contents |
|---|---|
| `LNCH` | the launch facts: grid/block/cluster dims, shared-memory sizes, register count, SM count, L2 size, occupancy limits, the reference SM clock and cache-control condition, the device |
| `SEGD` | the segment directory: each segment's cluster id, byte range inside the file and sha256 |
| `PLCE` | CTA→SM placement: which SM and warp-slot base each CTA ran on |
| `GSYN` | the launch-wide merge of every segment's global-sync observations |
| `TOPO` | the SM topology: placement domains, L2 partitions, TPC pairs, GPCs |

The segment images themselves are appended after the tables, located
through `SEGD`. Merging renumbers CTAs to a dense launch-wide space
and rebases the per-segment indices ([../formats.md](../formats.md)
§4).

## 4. Build flow and integrity

`cyclescope build <segments-dir> --out job.bin` collects the segments
of a launch, checks that they share identical tree, program and
converter digests (a segment built from a different tree cannot be
merged silently), and writes the container. Payload and per-segment
sha256 digests are verified on read.

## 5. Cache and relink

Conversion is the expensive stage, so segments are cached
(`python/cyclescope/job/cache.py`): an entry is keyed by the capture,
the cluster, the converter source digest and the tree digest. A hit
reuses the bytes; an entry whose converter inputs match but whose
tree differs is **relinked** instead of reconverted
(`python/cyclescope/job/relink.py`, `cyclescope relink`): the program
tables are recompiled for the new tree and each `USES` row is
re-walked with its stored axis values — the captured facts are
unchanged. `cyclescope split` does the inverse of `build`, unpacking a
container back into a segments directory.

## 6. Results

The same shell carries the outputs: `GSIMRES` (result counters),
`GSIMJNL` (issue journal) and `GSIMWLG` (wait ledger). Their layouts
are in [../formats.md](../formats.md) §7.

## 7. In the code

| part | where |
|---|---|
| shared shell, header, directory | `python/cyclescope/io/{file_header,fixed_tables}.py`, `cpp/core/job_format.hpp`, `job_reader.{hpp,cpp}` |
| segment tables | `python/cyclescope/job/{segment_format,segment_builder,semantics,timing_rows,program_tables}.py`, `cpp/convert/segment_builder.{hpp,cpp}` |
| container write/merge/read | `python/cyclescope/job/container.py` |
| cache, relink, split | `python/cyclescope/job/{cache,relink}.py`, `cyclescope split` in `python/cyclescope/cli.py` |
| result writers | `cpp/core/result_writer.{hpp,cpp}` |
