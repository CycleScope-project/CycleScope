# Capturing a new library

[中文](../../zh/agents/new-library.md)

Bringing up a library or framework the repository has never captured,
in order: buildable → reproducible → case spec → first capture.

## 1. Make the assets reproducible

Every external asset enters through `configs/capture_tools.json`
(entry kinds: `git`, `pypi_wheels`, `download`, `build`, `composite`;
integrity by `tree_sha256`):

1. Add an entry: source (`url`/`rev` or wheels or download URL),
   `patches` (repo-relative patch files), `toolchain` (`h100`/`b300`
   for `build` entries — the build script runs inside that toolchain
   environment), `outputs`, `purpose`, `cases`.
2. Provision: `python3 scripts/provision/provision_tools.py fetch <name>`
   (or `build <name>` inside the toolchain environment), then
   `… verify <name>` (tree hash must match `tree_sha256`; re-pin with
   `scripts/provision/tree_sha256.py` when the entry legitimately
   changes). See [../capture-tools.md](../capture-tools.md).

## 2. Make it runnable

- Hopper: the case is an argv/env row in `cases.tsv` — a standalone
  binary, or a small harness script under `suites/hopper/runners/`
  (see how `environment_far.sh` is used by `env_profile: far`).
- Blackwell: a runner module under `suites/blackwell/runners/<family>/`
  plus a family adapter in `runners/common/capture_adapters.py` and a
  `cases_<family>.json` spec; new families need adapter code (not
  exercised on the published tree).

## 3. JIT-compiling libraries

- Write caches/workspaces under `<CAPTURE>` (the per-case scratch
  directory), never a shared path — parallel cases must not share a
  JIT cache.
- Cold JIT can take minutes: raise `cyclescope capture --timeout`
  (default 1800 s) and the `reference --timeout` accordingly.
- The ncu census prepass warms the cache before the traced run, so the
  reference sees the same code path; triton-family cases pin the
  autotune config via the prepass (`prepare_case` in
  `python/cyclescope/capture/driver.py`).

## 4. Launch-selection pitfalls

- Many launches: pick `kernel_regex` + `occurrence` against the ncu
  census (`ncu_launches.jsonl`), not by eyeball order.
- Warm-up launches: count them — `occurrence` is the Nth census match,
  including warm-ups.
- Main vs helper kernels: the regex must match exactly the kernel you
  want (mangled name); a too-wide regex silently picks a helper.
- Kernels that differ between runs (JIT name suffixes, autotune
  variants): pin the config first (prepass/autotune file), otherwise
  the census and the traced run see different kernels.

## 5. Usual first failures

| first failure | goes to |
|---|---|
| `no traced launch`, `selection_failed`, census miss | [capture-failures.md](capture-failures.md) §2 |
| `missing_fact:*` / `unsupported:*` on a new opcode or operand form | [convert-failures.md](convert-failures.md) |
| `unsupported:engine:*` / deadlock on a new synchronisation pattern | [engine-failures.md](engine-failures.md) |
| wrong timing, mechanism exists but leaf missing/placeholder | [missing-leaf.md](missing-leaf.md) |
| behaviour with no mechanism rule | [missing-mechanism.md](missing-mechanism.md) |

## 6. Reproducibility checklist

- [ ] the asset is in `configs/capture_tools.json` with a pinned
      `tree_sha256`, and `provision_tools.py verify <name>` passes
- [ ] the case spec is committed (`cases.tsv` row / family JSON +
      adapter), not a hand-edited command line
- [ ] JIT caches live under `<CAPTURE>`; cold-JIT fits inside
      `--timeout`
- [ ] `cyclescope capture --dry-run` shows the intended argv/env/cwd
      and `launch_selection`
- [ ] `trace_check` passes and `ncu_launches.jsonl` contains the
      selected launch
- [ ] the census and the traced run pick the same kernel (pinned
      autotune/JIT config if applicable)
