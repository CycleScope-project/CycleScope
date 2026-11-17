# Capturing a new kernel

[中文](../../zh/agents/new-kernel.md)

How to capture and simulate a kernel that is not in the suites, for an
application that already runs on the machine.

## 1. Prerequisites

1. A provisioned tracer: `$CYCLESCOPE_ARTIFACT_ROOT/build/tracer/
   sm_90a|sm_103a/tracer.so` (`provision_tools.py build tracer-sm_90a`,
   see [../capture-tools.md](../capture-tools.md)).
2. Site config resolves the engine build (`configs/site.json` from
   `configs/site.example.json`, or `CYCLESCOPE_BUILD_DIR`) and
   `CYCLESCOPE_ARTIFACT_ROOT`.
3. The application runs standalone under its normal environment.

## 2. How a Hopper case is defined

Hopper cases come from `suites/hopper/<set>/cases.tsv` (one row per
case, tab-separated; resolution in
`python/cyclescope/capture/driver.py`):

| column | meaning |
|---|---|
| `set` | library/set label (informational grouping) |
| `case_id` | unique case name |
| `kernel_regex` | regex over the mangled kernel name; selects the traced launch |
| `occurrence` | in `kernel_regex@occurrence`, the Nth launch whose name matches the regex (1-based, counted across the whole run; the tracer tries the regex against both demangled and mangled names) |
| `argv_json` | argv as a JSON list with `"` doubled (`""`); paths may use the placeholders below |
| `cwd` | working directory, `-` for default |
| `env_json` | extra environment, JSON object, `-`/`{}` for none |
| `env_profile` | empty/`default` or `far` (wraps argv in `bash -c 'source suites/hopper/runners/environment_far.sh && exec "$@"'`) |
| `tma_recorder`, `baseline` | `-` in the shipped suite |
| `note` | free text (semicolon-separated metadata) |

Path forms inside `argv_json`/`env_json`: `$CYCLESCOPE_ARTIFACT_ROOT/
tools/<name>` for provisioned tools, `$CYCLESCOPE_ROOT` for repo
files, `<CAPTURE>` for the per-case capture directory (scratch /
runtime-generated inputs live there).

Launch selection priority (`_capture_selection` in
`python/cyclescope/capture/driver.py`): a `--reference` case key >
a `launches.tsv` table row (`resolved.launch_key`, an exact
`(kernel, occurrence)` pair) > the ncu census. Only the census leg
runs the unprofiled ncu prepass (`ncu_launches.jsonl`; needs `ncu`,
see `CYCLESCOPE_NCU`) — a reference key or a table row skips it. On
Hopper the census is only a cross-check that `occurrence` matches
exist (`occurrence N of <regex> beyond ncu census`); a missing census
is recorded, not fatal.

## 3. How a Blackwell case is defined

`suites/blackwell/<set>/cases.txt` maps `case_id → family`; the family
row lives in `suites/blackwell/<set>/cases_<family>.json`, and the
family's adapter in
`suites/blackwell/runners/common/capture_adapters.py`
(`capture_registry.json` routes case-id prefixes to adapters). A new
family needs adapter code; argv-level cases of an existing family only
need a JSON row. Not exercised on the published tree.

## 4. Worked Hopper example

1. Create `suites/hopper/local/cases.tsv` with header (copy the header
   of `suites/hopper/full_gpu/cases.tsv`) and one row:

   ```
   local	my_demo	my_kernel_regex	1	[""/path/to/app"",""--flag""]	-	{}	default	-	-	standalone binary
   ```

2. Dry-run the resolution (no GPU needed):

   ```
   python3 -m cyclescope.cli capture --suite hopper/local --case my_demo --out <scratch> --dry-run
   ```

   It prints a JSON record. Real output for the row above (scratch
   copy, `/bin/true` as the app):

   ```json
   {
     "suite": "hopper/local", "case": "my_demo",
     "argv": ["/bin/true"],
     "env": {"PYTHONPATH": ".../suites/hopper/runners/harness:..."},
     "cwd": "",
     "tracer": "<artifact root>/build/tracer/sm_90a/tracer.so",
     "mode": "both",
     "capture_format": "cyclescope.capture.v1",
     "launch_selection": "my_kernel_regex@1"
   }
   ```

   (`tracer` shown abbreviated — the real value is the provisioned
   tracer path.) Verify `argv`/`env`/`cwd` are exactly what the app
   needs (the driver injects the hopper harness dir into
   `PYTHONPATH`).
3. Run it on a GPU machine:
   `cyclescope run --suite hopper/local --case my_demo --out <out>`.
4. Judge the result: there is no baseline row, so `cyclescope report`
   gives verdict `n/a`; read `sim_cycles`/`hw_cycles` and the
   `ratio`/`err_pct` columns of `sim/my_demo/status.json`-driven
   report rows, and compare with neighbouring shipped cases' ratios.

## 5. Not supported today

- There is no "trace an arbitrary command line" entry point; a case
  always resolves through a suite row (`cases.tsv` on Hopper, family
  spec on Blackwell).
- The GPU stages of this example (capture + ncu reference on a real
  card) were not exercised on the published tree; the dry-run
  resolution was.
- A new Blackwell family needs adapter code; a Hopper standalone
  binary needs only a `cases.tsv` row.
