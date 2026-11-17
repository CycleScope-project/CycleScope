# Getting started

[中文](../zh/getting-started.md)

This page takes you from a fresh clone to a simulated kernel. You do
not need a GPU for it: the repository ships example captures, and
everything after capture runs on the CPU.

## 1. What you need

- Linux with a C++20 compiler, CMake 3.20 or newer, Python 3.10 or
  newer, and the OpenSSL development headers (libcrypto).
- The `zstd` command on `PATH`. Traces are stored compressed.
- A GPU only if you want to capture kernels yourself (section 5).

The versions the project was built and tested with are listed in
[environment.md](environment.md).

## 2. Build and install

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure    # optional: run the C++ tests
pip install -e .
```

The build produces four binaries in `build/`: `cyclescope-engine` (the
simulator) and `cyclescope-capture-facts`, `cyclescope-convert`,
`cyclescope-trace-split` (the conversion tools). `pip install -e .`
adds the `cyclescope` command, which drives them.

Install in editable mode from a clone, as above. The package reads
`configs/`, `suites/` and `scripts/` from the checkout, so a plain
`pip install` without the source tree does not work, and the command
tells you so.

To try the command without installing:

```bash
PYTHONPATH=python python3 -m cyclescope.cli --help
```

## 3. Simulate your first kernel

`examples/captures/` holds ten captured kernels, five per
architecture. Run one:

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example
cyclescope report out/example
```

`cyclescope run` converts the capture, builds the job file and
simulates it. `cyclescope report` prints one row per case. The columns
to read first:

| column | meaning |
|---|---|
| `sim_cycles` | cycles the simulation took |
| `hw_cycles` | cycles the same launch took on the real GPU, measured with Nsight Compute |
| `err_pct` | the difference between the two, in percent |
| `verdict` | `ok` when the run reproduces the shipped baseline |

For this case the two cycle counts are about 1 % apart. The other
examples and their expected numbers are listed in
[examples/README.md](../../examples/README.md). For Blackwell cases use
`--suite blackwell/full_gpu --captures examples/captures/blackwell`.

Everything the run wrote is under `out/example/sim/<case>/`. The
layout is described in [pipeline.md](pipeline.md).

## 4. Look inside the execution

The total cycle count is only the summary. Add `--ledgers` to also
write the issue journal and the wait ledger: when each instruction
issued, how long it waited, and what it waited for.

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example --ledgers
```

[results.md](results.md) shows how to read them.

## 5. Capture a kernel on your own GPU

Capture needs more than the simulator does: an NVIDIA GPU of a
supported architecture, the NVBit tracer, Nsight Compute for the
hardware reference, and the libraries the suite kernels come from.
[capture-tools.md](capture-tools.md) explains how to set these up.
With them in place, leave out `--captures` and the case is captured on
the local GPU first:

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_bf16_dense_02 \
    --out out/captured
```

## 6. Configuration

The tools find their directories through a small site file. The
defaults work for the steps above; you only need this when your build
directory, tracer or Nsight Compute live somewhere else.

Copy `configs/site.example.json` to `configs/site.json`, or point
`CYCLESCOPE_SITE_CONFIG` at your own copy. An environment variable
always wins over the file.

| key | environment variable | what it sets | default |
|---|---|---|---|
| `artifact_root` | `CYCLESCOPE_ARTIFACT_ROOT` | where tools, captures and caches go | `<repo>/artifacts` |
| `build_dir` | `CYCLESCOPE_BUILD_DIR` | where the built binaries are | `<repo>/build` |
| `tracer` | `CYCLESCOPE_TRACER` | the tracer library, per SM architecture | under the artifact root |
| `ncu` | `CYCLESCOPE_NCU` | the Nsight Compute binary | `ncu` on `PATH` |
| `nvdisasm` | `NVDISASM` | the `nvdisasm` binary the tracer uses | under the artifact root |

Print what is in effect:

```bash
PYTHONPATH=python python3 -m cyclescope.site --show
```

## Where to go next

- [pipeline.md](pipeline.md): the stages one by one, and every file a
  run writes.
- [results.md](results.md): reading the report and the execution
  record.
- [suites.md](suites.md): the 1,000 kernels the project is evaluated
  on.
- [design/README.md](design/README.md): how CycleScope works and why.
