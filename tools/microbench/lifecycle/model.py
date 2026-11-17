#!/usr/bin/env python3
"""kernel.lifecycle envelope derivation, protocol version 2
(tools/microbench/lifecycle/protocol-v2.json; version 1 is superseded).

Per launch (launches 0 and 1 of every variant dropped):
    env = sm__cycles_active.max - max over SMs of the marker span
          - fetch(probe kernel)
marker span of one SM = max t1 - min t0 over the warps the LAUNCH line
places on that SM (C = 2 puts one CTA per SM; clocks of different SMs
are not comparable, so the span is per SM and the max is taken).
fetch(probe kernel) = the installed instruction.fetch model's cost for the
code bytes from the kernel entry up to and including the first clock read
(the launch-prefetched head arrives at stream_l1 per 128 B line in the
engine, instruction_fetch.cpp launch_prefetch): stream_l1 x ceil(bytes/128),
with the bytes taken from the SASS of the build (k_life<L> entry to the
first CS2R SR_CLOCKLO) and stream_l1 / line_bytes from the generated tree
of the device.

envelope(W) = mean over launches and over L of the C = 1, S = 0 cells.
Acceptance per cell (protocol): se(mean env) / mean <= 3 % in both runs,
run agreement <= 5 %, |mean env(L = 4096) - mean env(L = 0)| <= 10 %
(addendum 2026-11-04: the verbatim probe's L axis is {0, 1024, 4096}).
The launch-level distribution (sorted samples, deciles, a 2-mode split)
is recorded per cell.

Usage: model.py --run-a <dir> --run-b <dir> --tree configs/trees/<dev>.json
                --sass <lifecyc.sass> --out model.json
"""
import argparse, csv, io, json, math, re, statistics
from pathlib import Path

LAUNCH_RE = re.compile(r"^LAUNCH k_life variant=life\.c(\d+)\.w(\d+)\.s(\d+)\.l(\d+) launch=(\d+) smid=([\d,]+) t0=([\d,]+) t1=([\d,]+)")
DROP = {0, 1}
SE_REL = 0.03      # se(mean env) / mean
RUN_AGREE = 0.05   # |a - b| / mean
L_AGREE = 0.10     # |env(L=4096) - env(L=0)| / env(L=0)
WS = (1, 4, 8, 16, 32)
LS = (0, 1024, 4096)


def parse_launches(path):
    """-> ordered list of (variant, launch, span_max_over_sms)."""
    out = []
    for line in Path(path).read_text().splitlines():
        m = LAUNCH_RE.match(line.strip())
        if not m:
            continue
        c, w, s, l, launch = map(int, m.groups()[:5])
        smids = [int(x) for x in m.group(6).split(",")]
        t0 = [int(x) for x in m.group(7).split(",")]
        t1 = [int(x) for x in m.group(8).split(",")]
        spans = []
        for b in range(c):  # CTA b -> warps b*w .. b*w+w-1, on smids[b]
            lo, hi = b * w, (b + 1) * w
            spans.append(max(t1[lo:hi]) - min(t0[lo:hi]))
        out.append((f"c{c}.w{w}.s{s}.l{l}", launch, max(spans), smids))
    return out


def parse_counters(path):
    text = Path(path).read_text()
    header = re.search(r'^(?:"ID"|ID),', text, re.MULTILINE)
    rows = list(csv.reader(io.StringIO(text[header.start():]))) if header else []
    names = rows[0] if rows else []
    idx = {k: names.index(k) for k in ("sm__cycles_active.max", "sm__cycles_elapsed.max") if k in names}
    act, ela = [], []
    for r in rows[2:]:
        if len(r) < len(names):
            continue
        act.append(int(float(r[idx["sm__cycles_active.max"]].replace(",", ""))))
        ela.append(int(float(r[idx["sm__cycles_elapsed.max"]].replace(",", ""))) if "sm__cycles_elapsed.max" in idx else 0)
    return act, ela


def prologue_lines(sass_path, line_bytes=128):
    """per k_life<L>: bytes from entry to the first CS2R SR_CLOCKLO inclusive -> lines."""
    out = {}
    cur = None
    for ln in Path(sass_path).read_text().splitlines():
        m = re.search(r"Function : (\S+)", ln)
        if m:
            cur = m.group(1); continue
        if cur is None or cur in out:
            continue
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?);", ln)
        if m and "CS2R" in m.group(2) and "SR_CLOCKLO" in m.group(2):
            addr = int(m.group(1), 16)
            out[cur] = {"prologue_bytes": addr + 16, "lines": math.ceil((addr + 16) / line_bytes)}
    return out


def fetch_model(tree_path):
    t = json.loads(Path(tree_path).read_text())
    for n in t["nodes"]:
        if n["id"] == "instruction.fetch.root":
            f = n["parameter"]["fields"]
            return {k: f[k]["model"]["value"] for k in ("stream_l1_cycles_per_line", "stream_cold_cycles_per_line",
                                                       "launch_prefetch_bytes", "line_bytes", "enabled")}
    raise SystemExit("instruction.fetch.root not in tree")


def fetch_cost(fm, lines):
    """engine form: prefetched head at stream_l1 per line, stream_cold beyond launch_prefetch_bytes."""
    if not fm["enabled"]:
        return 0.0
    head = fm["launch_prefetch_bytes"] // fm["line_bytes"]
    return fm["stream_l1_cycles_per_line"] * min(lines, head) + fm["stream_cold_cycles_per_line"] * max(0, lines - head)


def cells_of_run(run_dir, fetch_by_l):
    launches = parse_launches(Path(run_dir) / "launches.log")
    act, ela = parse_counters(Path(run_dir) / "counters.csv")
    warn = None
    if len(launches) != len(act):
        warn = f"{len(launches)} launches vs {len(act)} counter rows"
    cells = {}
    for (v, launch, span, smids), a, e in zip(launches, act, ela or [0] * len(launches)):
        if launch in DROP:
            continue
        l = int(v.split(".l")[1])
        c = cells.setdefault(v, {"env": [], "env_elapsed": [], "span": [], "active": [], "smid": []})
        c["env"].append(a - span - fetch_by_l[l])
        c["env_elapsed"].append(e - span - fetch_by_l[l])
        c["span"].append(span); c["active"].append(a); c["smid"].append(smids)
    out = {}
    for v, c in cells.items():
        env = c["env"]; n = len(env)
        if n < 4:
            continue
        mean = statistics.fmean(env)
        se = statistics.stdev(env) / math.sqrt(n)
        srt = sorted(env)
        # two-mode split: largest gap in the sorted samples
        gaps = [(srt[i + 1] - srt[i], i) for i in range(n - 1)]
        g, i = max(gaps)
        lo, hi = srt[:i + 1], srt[i + 1:]
        out[v] = {"mean": mean, "se": se, "se_rel": se / mean if mean else None, "n": n,
                  "median": statistics.median(env), "min": srt[0], "max": srt[-1],
                  "deciles": [srt[min(n - 1, round(q * (n - 1) / 10))] for q in range(11)],
                  "modes": {"gap": g, "low": {"n": len(lo), "mean": statistics.fmean(lo)},
                            "high": {"n": len(hi), "mean": statistics.fmean(hi)} if hi else None},
                  "mean_elapsed_form": statistics.fmean(c["env_elapsed"]),
                  "span_mean": statistics.fmean(c["span"]), "active_mean": statistics.fmean(c["active"]),
                  "samples": env}
    return out, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--sass", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fm = fetch_model(a.tree)
    pro = prologue_lines(a.sass, int(fm["line_bytes"]))
    fetch_by_l = {}
    for l in LS:
        k = [f for f in pro if re.match(rf"_Z6k_lifeILi{l}EEvPmf$", f)]
        if not k:
            raise SystemExit(f"k_life<{l}> not in SASS {a.sass}: {list(pro)[:5]}")
        fetch_by_l[l] = fetch_cost(fm, pro[k[0]]["lines"])
    runs = {}
    for tag, d in (("a", a.run_a), ("b", a.run_b)):
        runs[tag], warn = cells_of_run(d, fetch_by_l)
        if warn:
            runs[tag]["_warn"] = warn
    out = {"protocol": "tools/microbench/lifecycle/protocol-v2.json", "runs": {"a": a.run_a, "b": a.run_b},
           "fetch_model": fm, "prologue": pro, "fetch_by_l": fetch_by_l, "cells": {}, "install": {}}
    for v in sorted(set(runs["a"]) | set(runs["b"])):
        if v.startswith("_"):
            continue
        ca, cb = runs["a"].get(v), runs["b"].get(v)
        cell = {"run_a": ca, "run_b": cb}
        if ca and cb:
            m = (ca["mean"] + cb["mean"]) / 2
            cell["mean"] = m
            cell["run_agree"] = abs(ca["mean"] - cb["mean"]) / m if m else None
            cell["se_ok"] = (ca["se_rel"] or 1) <= SE_REL and (cb["se_rel"] or 1) <= SE_REL
        out["cells"][v] = cell
    # install column: C = 1, S = 0, per W: mean over launches and over L, averaged over the two runs
    for w in WS:
        per_l = {}
        for l in LS:
            c = out["cells"].get(f"c1.w{w}.s0.l{l}")
            if c and "mean" in c:
                per_l[l] = c
        if len(per_l) != len(LS):
            out["install"][str(w)] = {"missing": [l for l in LS if l not in per_l]}
            continue
        env_w = statistics.fmean(c["mean"] for c in per_l.values())
        l_agree = abs(per_l[4096]["mean"] - per_l[0]["mean"]) / per_l[0]["mean"]
        ok = all(c["se_ok"] for c in per_l.values()) and all(c["run_agree"] <= RUN_AGREE for c in per_l.values()) \
            and l_agree <= L_AGREE
        out["install"][str(w)] = {
            "envelope_cycles": env_w, "per_l_mean": {str(l): c["mean"] for l, c in per_l.items()},
            "per_l_run_agree": {str(l): c["run_agree"] for l, c in per_l.items()},
            "se_rel_max": max(x["se_rel"] for c in per_l.values() for x in (c["run_a"], c["run_b"])),
            "l_agree": l_agree, "verdict": "measured" if ok else "experimental"}
    out["points"] = [[w, out["install"][str(w)]["envelope_cycles"]] for w in WS if "envelope_cycles" in out["install"][str(w)]]
    out["verdict"] = ("measured" if out["points"] and all(out["install"][str(w)].get("verdict") == "measured" for w in WS)
                      else "experimental")
    Path(a.out).write_text(json.dumps(out, indent=1))
    for w in WS:
        i = out["install"][str(w)]
        if "envelope_cycles" in i:
            print(f"  W={w:2d}: env {i['envelope_cycles']:.1f} per-L {', '.join(f'{k}:{v:.0f}' for k, v in i['per_l_mean'].items())}"
                  f" se_rel_max {100*i['se_rel_max']:.1f}% l_agree {100*i['l_agree']:.1f}% -> {i['verdict']}")
        else:
            print(f"  W={w:2d}: missing {i['missing']}")
    print("  verdict:", out["verdict"], "fetch_by_l:", fetch_by_l)


if __name__ == "__main__":
    main()
