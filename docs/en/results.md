# Reading the results

[中文](../zh/results.md)

A run produces two kinds of output: the report row, which tells you
how close the simulated cycle count is to the hardware, and the
execution record, which tells you *why* — every instruction's issue
cycle and what each warp waited for.

## 1. The report

```bash
cyclescope report <run_out> [<run_out> ...]
    [--expected expected.tsv] [--tolerance PCT] [--out file]
cyclescope report --baseline <expected.tsv>
```

`report` reads every `sim/<case>/status.json` under each run output
directory and prints one row per case, tab-separated:

| column | meaning |
|---|---|
| `suite`, `case` | which case |
| `stage` | stage the run stopped at, or `done` |
| `class` | `ok` or the failure class |
| `issued` | instruction instances the engine issued |
| `sim_cycles` | simulated cycle count |
| `hw_cycles` | cycles the same launch took on the GPU (Nsight Compute) |
| `ratio` | `sim_cycles / hw_cycles` |
| `err_pct` | `abs(ratio - 1)` in percent |
| `base_sim` | the shipped baseline's `sim_cycles` |
| `delta_pct` | this run's distance from the baseline, signed percent |
| `placement` | the captured CTA→SM placement digest |
| `verdict` | one word — see below |

The verdict compares the run with the shipped `expected.tsv`:

- `ok` — the run finished and, on the same placement, is within
  `--tolerance` percent of the baseline (default 3 %).
- `fail` — the run did not finish cleanly, or finished beyond the
  tolerance. `report` exits 1 when any row fails.
- `listed` — the run finished but its placement differs from the
  baseline's, so the baseline number does not apply directly.
- `n/a` — the case is not in the expected table.

`--expected` points at a different expected table (repeatable);
without it the suite's own `expected.tsv` is used. `--out` writes the
table to a file. After the rows, one `#` block per suite prints an
accuracy line (`cases`, `mape`, `p90_abs_err`, `median_ratio` —
computed over the rows that finished with hardware cycles) and a
baseline line (`within`, `beyond`, `listed`, max |delta|).

`--baseline <expected.tsv>` prints the accuracy summary of a shipped
expected table alone — no run needed.

## 2. The execution record

The engine writes up to three files:

| file | magic | what it holds |
|---|---|---|
| `result.bin` | `GSIMRES` | totals: cycles, issued/completed, hard failures, per-SMSP active spans, wait cycles per bucket, resource waits, telemetry counters |
| `journal.bin` | `GSIMJNL` | one row per issued instance: warp, SMSP, instance, issue cycle, operation class, template, wait bucket, wait cycles, sync object |
| `wait.bin` | `GSIMWLG` | wait ledger: per warp and cause, total cycles and count |

`cyclescope run` writes `result.bin` under
`sim/<case>/out-<builder>/`; `--ledgers` adds `journal.bin` and
`wait.bin` in the same directory. By hand they come from
`cyclescope-engine simulate --journal <path> --wait-ledger <path>`.

The issue journals and wait ledgers of all 1,000 suite kernels are
published in [CycleScope-records](https://github.com/CycleScope-project/CycleScope-records),
so they can be studied without running anything.

The readers are in `cyclescope.job.result` — `read_result`,
`read_journal`, `read_wait_ledger`, each taking the file's bytes:

```python
from pathlib import Path
from cyclescope.job.result import (
    read_result, read_journal, read_wait_ledger)

res = read_result(Path("result.bin").read_bytes())
print("cycles:", res["cycles"], " issued:", res["issued"])

waits = {}
for row in read_journal(Path("journal.bin").read_bytes())["JRNL"]:
    if row["wait_bucket"] != 0xFFFFFFFF:          # 0xFFFFFFFF = no wait
        waits[row["wait_bucket"]] = \
            waits.get(row["wait_bucket"], 0) + row["wait_cycles"]
for bucket, cycles in sorted(waits.items()):
    print(f"bucket {bucket}: {cycles} cycles")
```

## 3. The wait buckets

The journal's `wait_bucket` and the ledger's `cause_kind` hold the
`WaitBucket` enum of `cpp/core/simulator.hpp` — `0xFFFFFFFF` means the
instruction did not wait:

| id | bucket | what the warp was waiting for |
|---|---|---|
| 0 | dispatch | for the CTA to be dispatched and the warp to enter |
| 1 | fetch | on instruction fetch |
| 2 | collective | on a barrier release or a suspended wake |
| 3 | semantic | on a semantic gate (a poll, an async group, an arch rule) |
| 4 | resource | on a busy unit, port, queue or issue interval |
| 5 | register | on a register value or the compiler's schedule |
| 6 | scheduler | not attributed to any check above: the initial cause before a warp's first failed check |

## 4. Where to look next

- [formats.md](formats.md) §7 — the byte layout of all three files.
- [engine.md](engine.md) — what the telemetry groups count.
- [design/engine.md](design/engine.md) — the model that produces them.
