# Capturing on your own GPU: tools and setup

[中文](../zh/capture-tools.md)

Simulating a shipped capture needs nothing but the build. *Capturing*
a kernel needs more: the NVBit tracer, Nsight Compute for the hardware
reference, the CUDA toolchain, and the third-party sources the suite
kernels come from. None of that is vendored in this repository — every
external piece is registered by name in `configs/capture_tools.json`
and lives under `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>`.

## 1. Setting up a new machine

`scripts/provision/provision_tools.py` fetches, builds and checks
every registered entry:

```bash
export CYCLESCOPE_ARTIFACT_ROOT=<where tools live>
python3 scripts/provision/provision_tools.py plan          # the steps, in dependency order
python3 scripts/provision/provision_tools.py fetch --all   # or: fetch <name>...
python3 scripts/provision/provision_tools.py build <name>  # inside its toolchain environment
python3 scripts/provision/provision_tools.py verify --all
python3 scripts/provision/provision_tools.py bundle --out DIR   # export for an offline machine
```

In order: set the artifact root, read the `plan`, `fetch` everything
downloadable, run each `build` entry inside its registered toolchain
environment (`cd <clone> && scripts/provision/provision_tools.py build
<name>`), place the manual downloads, then `verify --all`. For a
machine without network, `bundle` exports the pristine results.

Notes:

- Set `CYCLESCOPE_PIP_INDEX_URL` to a reachable PyPI index if the
  default cannot be fetched; `CYCLESCOPE_TMP` controls the temporary
  directory used while fetching.
- Nsight Compute is a `manual` entry — the provisioner cannot download
  it; put it at `$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute` and
  let `verify` check it.
- `build` entries run inside their toolchain environment; building on
  the host does not produce the expected outputs.
- When builds run as root their outputs are root-owned;
  `CYCLESCOPE_OWNER` makes the provisioner chown them back.

## 2. Kinds of registry entry

Each entry in `configs/capture_tools.json` has a `kind` that selects
how it is produced (fields per entry: see the file):

| kind | what it is |
|---|---|
| `git` | an upstream checkout at a pinned revision, optional patches and submodule fetches |
| `pypi_wheels` | Python wheels from a locked requirements list |
| `download` | an archive or file set fetched and unpacked by checksum; `manual: true` means a human download |
| `build` | outputs of a registered build script running inside a toolchain |
| `composite` | symlinks assembling subtrees of other entries |
| `toolchain` | the environment a set of `build` entries runs inside |
| `overlay` | files laid over another entry's tree |

## 3. How a tree is checked

Every entry records a `tree_sha256`: all files and symlinks sorted by
path, each hashed with its content (or its link target), the whole
list hashed again. `verify` recomputes it — a tree passes only when
it matches, so a corrupted or partial provision cannot slip through.
`scripts/provision/tree_sha256.py` recomputes a digest by hand.

## 4. Things to know when capturing

- **JIT caches are rebuilt per case.** They are outputs, not inputs —
  each case gets an empty scratch cache. A cold JIT takes minutes on
  some cases, so a first capture may need a larger `--timeout` than
  the 5400-second default.
- **Nsight Compute differs per architecture.** On H100 the environment
  ships no ncu — point `CYCLESCOPE_NCU` at the provisioned
  `$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute` copy. On B300 use
  the environment's own `ncu`. The version each reference was timed
  with is recorded in the case's `timing.json`; exact versions are in
  [environment.md](environment.md).
- **Timeouts stop the whole process group.** Capture and reference
  children run in their own process group so a timeout cannot leave a
  profiler attached to a later run.

When a capture or a reference run fails, see
[agents/capture-failures.md](agents/capture-failures.md). To capture
a kernel or a library that is not in the suites, see
[agents/new-kernel.md](agents/new-kernel.md) and
[agents/new-library.md](agents/new-library.md).

## 5. Comparing captures and tool outputs

Two scripts, one line each:

- `scripts/provision/compare_tool_outputs.py` — diff a provisioned
  entry or build output against a reference tree.
- `scripts/pipeline/compare_captures.py` — diff two captures' static
  facts (launch list, functions, SASS, tensor maps).
