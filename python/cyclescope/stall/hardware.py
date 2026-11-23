"""Hardware side: ncu per-launch stall counters and per-pc stall samples.

A reference case directory (``cyclescope reference --metrics ...
--source-stalls``) holds ``ncu_metrics.csv`` (raw page, one row per
launch), ``ncu_source.csv`` (SourceCounters page of the picked launch)
and ``timing.json`` (the picked launch ordinal).
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

COUNTER_PREFIX = "smsp__average_warp_latency_issue_stalled_"
COUNTER_SUFFIX = ".ratio"

# ncu_source.csv column `stall_<name>` -> class name (the counter name
# where a counter exists)
SAMPLE_CLASSES = {
    "barrier": "barrier",
    "branch_resolving": "branch_resolving",
    "dispatch": "dispatch_stall",
    "drain": "drain",
    "imc": "imc_miss",
    "lg": "lg_throttle",
    "long_sb": "long_scoreboard",
    "math": "math_pipe_throttle",
    "membar": "membar",
    "mio": "mio_throttle",
    "misc": "misc",
    "no_inst": "no_instruction",
    "not_selected": "not_selected",
    "selected": "selected",
    "short_sb": "short_scoreboard",
    "sleep": "sleeping",
    "tex": "tex_throttle",
    "wait": "wait",
}
# classes the counters report and the samples never do
COUNTER_ONLY_CLASSES = ("gmma",)
CLASSES = tuple(sorted(set(SAMPLE_CLASSES.values()))) + COUNTER_ONLY_CLASSES


def _number(text) -> float:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return math.nan


@dataclass
class HardwarePc:
    offset: int                     # bytes from the first listed address
    sass: str
    executed: float                 # warp-level instructions executed
    samples: dict[str, float] = field(default_factory=dict)

    @property
    def sample_count(self) -> float:
        return sum(self.samples.values())


@dataclass
class HardwareStalls:
    kernel: str
    cycles: float                   # sm__cycles_active.max
    instructions: float             # smsp__inst_executed.sum
    counters: dict[str, float]      # class -> cycles per warp
    pcs: list[HardwarePc]

    @property
    def residency_per_warp(self) -> float:
        return sum(self.counters.values())

    @property
    def warps(self) -> float:
        return self.instructions / self.counters["selected"]

    @property
    def sample_total(self) -> float:
        return sum(p.sample_count for p in self.pcs)

    def class_samples(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.pcs:
            for name, n in p.samples.items():
                out[name] = out.get(name, 0.0) + n
        return out

    @property
    def cycles_per_sample(self) -> float:
        """Pooled scale: cycles per warp one sample stands for, over the
        classes that have both a counter and samples.  Used for the
        classes that only the samples report (drain, branch_resolving,
        misc, membar); every other class has its own scale, see
        class_scales()."""
        samples = self.class_samples()
        counted = [c for c, n in samples.items()
                   if n > 0 and self.counters.get(c, 0.0) > 0]
        total = sum(samples[c] for c in counted)
        if total <= 0:
            raise ValueError("unsupported:stall_samples:none (ncu_source.csv)")
        return sum(self.counters[c] for c in counted) / total

    def class_scales(self) -> dict[str, float]:
        """Cycles per warp one sample of each sampled class stands for.

        The sampler takes about one sample per SM and interval whatever
        the number of resident warps (a launch with 32 times the warps
        yields about the same number of samples), so a sample taken
        while many warps are resident stands for more warp cycles than
        one taken while few are, and the classes are not sampled at one
        common rate.  The counters are exact per class (stalled cycles
        per launched warp), so a class with a counter is scaled by its
        own counter / its own samples: the samples only spread that
        class over its pcs.  A class without a counter takes the pooled
        scale."""
        pooled = self.cycles_per_sample
        scales = {}
        for name, n in self.class_samples().items():
            if n <= 0:
                continue
            counter = self.counters.get(name, 0.0)
            scales[name] = counter / n if counter > 0 else pooled
        return scales

    def unsampled(self) -> dict[str, float]:
        """Counter classes without a single sample: cycles per warp the
        samples cannot place on a pc (h100's gmma counter, and rare
        classes of short launches)."""
        samples = self.class_samples()
        return {c: v for c, v in self.counters.items()
                if v > 0 and not samples.get(c)}

    def reference_consistency(self) -> float:
        """Total variation distance between the counter class shares
        and the sample class shares without `drain`."""
        samples = dict(self.class_samples())
        samples.pop("drain", None)
        n, r = sum(samples.values()), self.residency_per_warp
        names = set(samples) | set(self.counters)
        return 0.5 * sum(abs(self.counters.get(c, 0.0) / r
                             - samples.get(c, 0.0) / n) for c in names)

    def selected_scale_ratio(self) -> float:
        """(scale of the `selected` class) / (pooled scale); 1 when the
        samples are proportional to warp residency.  nan without
        selected samples."""
        selected = self.class_samples().get("selected", 0.0)
        if selected <= 0:
            return math.nan
        return (self.counters["selected"] / selected) / self.cycles_per_sample

    def class_scale_ratios(self) -> dict[str, float]:
        """Per class with a counter and samples: its scale / the pooled
        scale."""
        pooled = self.cycles_per_sample
        samples = self.class_samples()
        return {c: scale / pooled for c, scale in self.class_scales().items()
                if self.counters.get(c, 0.0) > 0 and samples.get(c)}


def read_source_table(path: Path) -> tuple[str, list[HardwarePc]]:
    """(kernel name, per-pc rows) of an ncu SourceCounters csv."""
    text = Path(path).read_text(errors="replace")
    start = text.find('"Address"')
    if start < 0:
        raise ValueError(f"missing_fact:stall_sample_table ({path})")
    if text.find('"Address"', start + 1) >= 0:
        raise ValueError(f"unsupported:stall_sample_tables:multiple ({path})")
    kernel = ""
    kstart = text.rfind('"Kernel Name"', 0, start)
    if kstart >= 0:
        row = next(csv.reader([text[kstart:start].splitlines()[0]]))
        kernel = row[1] if len(row) > 1 else ""
    reader = csv.DictReader(io.StringIO(text[start:]))
    columns: dict[str, str] = {}
    for name in reader.fieldnames or []:
        if name.startswith("stall_") and "(Not Issued)" not in name:
            short = name[len("stall_"):]
            if short not in SAMPLE_CLASSES:
                raise ValueError(
                    f"unsupported:stall_sample_class:{short} ({path})")
            columns[name] = SAMPLE_CLASSES[short]
    if not columns:
        raise ValueError(f"missing_fact:stall_sample_columns ({path})")
    pcs: list[HardwarePc] = []
    base = None
    for row in reader:
        address = (row.get("Address") or "").strip()
        if not address.startswith("0x"):
            continue
        value = int(address, 16)
        if base is None:
            base = value
        samples = {}
        for name, cls in columns.items():
            n = _number(row.get(name))
            if n == n and n:
                samples[cls] = samples.get(cls, 0.0) + n
        executed = _number(row.get("Instructions Executed"))
        pcs.append(HardwarePc(offset=value - base,
                              sass=(row.get("Source") or "").strip(),
                              executed=executed if executed == executed
                              else 0.0, samples=samples))
    if not pcs:
        raise ValueError(f"missing_fact:stall_sample_rows ({path})")
    return kernel, pcs


def read_counter_row(path: Path, ordinal: int) -> dict[str, str]:
    """The launch row with ID == ordinal of an ncu raw-page csv."""
    text = Path(path).read_text(errors="replace")
    start = text.find('"ID"')
    if start < 0:
        raise ValueError(f"missing_fact:ncu_launch_table ({path})")
    for row in csv.DictReader(io.StringIO(text[start:])):
        if (row.get("ID") or "").strip() == str(ordinal):
            return row
    raise ValueError(f"missing_fact:ncu_launch_row:{ordinal} ({path})")


def load_hardware(case_dir: Path) -> HardwareStalls:
    case_dir = Path(case_dir)
    timing = json.loads((case_dir / "timing.json").read_text())
    ordinal = (timing.get("pick") or {}).get("ordinal")
    if ordinal is None:
        raise ValueError(
            f"missing_fact:reference_launch_ordinal ({case_dir}/timing.json)")
    row = read_counter_row(case_dir / "ncu_metrics.csv", int(ordinal))
    counters = {}
    for name, value in row.items():
        if name and name.startswith(COUNTER_PREFIX) and \
                name.endswith(COUNTER_SUFFIX):
            n = _number(value)
            if n == n:
                counters[name[len(COUNTER_PREFIX):-len(COUNTER_SUFFIX)]] = n
    if not counters.get("selected"):
        raise ValueError(
            f"missing_fact:stall_counter:selected ({case_dir}/ncu_metrics.csv)")
    for name in counters:
        if name not in CLASSES:
            raise ValueError(
                f"unsupported:stall_counter_class:{name} ({case_dir})")
    instructions = _number(row.get("smsp__inst_executed.sum"))
    cycles = _number(row.get("sm__cycles_active.max"))
    if not instructions == instructions or not cycles == cycles:
        raise ValueError(
            f"missing_fact:ncu_launch_metrics ({case_dir}/ncu_metrics.csv)")
    _, pcs = read_source_table(case_dir / "ncu_source.csv")
    return HardwareStalls(kernel=(row.get("Kernel Name") or "").strip(),
                          cycles=cycles, instructions=instructions,
                          counters=counters, pcs=pcs)
