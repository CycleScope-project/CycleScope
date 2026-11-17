# Capture and reference failures

[中文](../../zh/agents/capture-failures.md)

The app runs but capture or the ncu reference fails. Status strings
come from `python/cyclescope/capture/driver.py` and
`python/cyclescope/reference/ncu_runner.py`; the checker is
`python/cyclescope/capture/trace_check.py` (`trace_check.txt` in the
capture directory).

## 1. Check order

1. `captures/<case>/run.json`: `status`, `rc`, `utc_start`,
   `prepass`, `toolchain`, `tracer_sha256`.
2. `captures/<case>/app.log`: the application's own stdout/stderr.
3. `captures/<case>/trace_check.txt`: the `trace_check` verdict.
4. `captures/<case>/ncu_launches.jsonl`: the census (one row per
   launch) — needed by both selection and occupancy facts.

## 2. No traced launch / wrong launch

- `no traced launch (no trace-*.bin.zst, no launches.jsonl row with
  traced=true)` — the app never reached the selected launch: killed
  at `--timeout`, selection mismatch, or the launch was skipped.
  Check `launches.jsonl` for what actually ran.
- `selection_failed:<e>` — the regex/occurrence did not match the
  census (occurrence beyond the census count is reported as
  `occurrence N of <regex> beyond ncu census`). Count warm-ups.
- `launch_table_mismatch:{diff}` — a `launches.tsv` row pinned a pair
  the census does not have.
- `ambiguous_launch` — selection is ambiguous; tighten `kernel_regex`.

## 3. trace_check failure strings

`quick_check`/full check emit: `zstd_missing` (no `zstd` binary on
PATH — trace files are `trace-<launch>.bin.zst`), `zstd frame check
failed`, `bad trace header` (file not starting `GSIMTRC\0`), `missing
functions/<id>.tsv` (static table for the traced function absent or
empty), `missing trace-N.bin.zst`, `trace-N: <exception>` (record
stream undecodable — usually a truncated write), `FAIL load_static:
<exc>`.

## 4. App hangs or dies only under the tracer

1. Confirm it runs without the tracer (same argv/env/cwd, no
   `CUDA_INJECTION64_PATH`).
2. Narrow the instrumented surface with the tracer's diagnostic
   switches (all read by name in `tools/tracer/tracer.cu` /
   `inject.cu`):

   | variable | effect |
   |---|---|
   | `CYCLESCOPE_TRACE_DIR` | output directory |
   | `CYCLESCOPE_CAPTURE_LAUNCHES` | `all` | comma ordinals | regex |
   | `CYCLESCOPE_CAPTURE_CLUSTERS` | `all` (default) | `sampled:<n>` | cluster id list |
   | `CYCLESCOPE_CAPTURE_MODE` | `trace` | `placement` |
   | `CYCLESCOPE_CAPTURE_LAUNCH_LIMIT` | trace only launch ordinals ≤ n |
   | `CYCLESCOPE_CUDA_STACK_SIZE` | device stack size |
   | `CYCLESCOPE_TRACER_{SKIP,BARE,NO_MREF,NO_RANK,NO_XRANK,NO_UR,NO_R}_OPCODES` | drop opcode classes from instrumentation (bisect: remove classes until the hang disappears, then re-add) |
   | `CYCLESCOPE_TRACER_OPERAND_{OPCODES,FAMILIES}` | restrict operand capture |
   | `CYCLESCOPE_TRACER_NO_AFTER` | disable after-injection records |
   | `CYCLESCOPE_TRACER_NO_GLOBAL_SYNC` | disable global-sync records |
   | `CYCLESCOPE_TRACER_IDX_RANGE` | restrict recorded instruction indices |
   | `CYCLESCOPE_TRACER_VERBOSE` | tracer-side log |

   Bisect: hang present → halve the opcode classes; hang gone → the
   dropped half contains the trigger.
3. `CYCLESCOPE_L2_FOOTPRINT` is forced to `0` by the driver
   (cold cache); leave it that way.

## 5. Timeouts and process groups

`--timeout` (default 1800 s) bounds each run; the census pass is
capped at `min(timeout_s, 900)`, the prepass at
`min(timeout_s, 1200)`. A timeout kills the process group; a timed-out
trace usually ends as `no traced launch` or a truncated record stream
(`trace-N: …`).

## 6. Reference failures

- `ncu_missing` — no `ncu` on PATH; H100 uses `CYCLESCOPE_NCU` (the
  environment ships none — install per `configs/capture_environment.json`),
  B300 uses the environment's ncu.
- `ncu_timeout` (TIMEOUT_RC) / `ncu_failed:rc=<n>` — the profiled run
  failed; check `app.log` and the ncu log in the reference directory.
- `reference launch <k> not in the ncu census` — a `--captures`
  replay pointed at a launch the new census does not contain.
- `app_failed:triton_prepass` — the triton autotune prepass failed
  (log: `triton_prepass.log` in the workdir).
- `placement` run failures: mode `placement` writes
  `placement.jsonl`; a missing/empty file with placement-dependent
  cases → `missing_fact` downstream at build.
