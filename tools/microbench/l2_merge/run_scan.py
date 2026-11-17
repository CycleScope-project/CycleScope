#!/usr/bin/env python3
"""l2_merge protocol: run probe/tma_scan (multi-SM TMA aggregate throughput,
hopper-prototype M6 m6r2 set) and summarise it.  Self-contained port of hopper-prototype
run_tma_scan.py --set m6r2 plus the parse / run_ncu / write_summary helpers
of run_l2_dram_scan.py.

nsm(G) = floor(SM count / G) * G (SM count from --sms or the binary's META
sms= line).  Every config runs --trials plain trials, then once under ncu
(--cache-control none --clock-control none, kernel regex tma_scan_kernel for
m6r2 / scan_kernel for m1r4, launch-skip 1 launch-count 1).  Writes
<out>/raw/, <out>/ncu/, <out>/results.json, <out>/summary.md and
<out>/m6_summary.md (m6r2) or <out>/m1_summary.md (m1r4, r = lts/l1tex).

usage: run_scan.py --bin BIN --out DIR [--trials 5] [--ncu NCU]
                   [--set m6r2|m1r4] [--sms N] [--only PREFIX,...]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

RE_KV = re.compile(r"(\w+)=([-\w.]+)")


def parse(text: str) -> dict:
    rec = {"meta": {}, "trials": [], "ctas": [], "result": {}, "hist": [], "histcheck": {}}
    for line in text.splitlines():
        if line.startswith("META "):
            rec["meta"] = dict(RE_KV.findall(line[5:]))
        elif line.startswith("PLACEMENT "):
            rec["placement"] = {k: int(v) for k, v in RE_KV.findall(line)}
        elif line.startswith("TRIAL "):
            rec["trials"].append({k: float(v) for k, v in RE_KV.findall(line)})
        elif line.startswith("CTA "):
            rec["ctas"].append({k: float(v) for k, v in RE_KV.findall(line)})
        elif line.startswith("RESULT "):
            rec["result"] = {k: float(v) for k, v in RE_KV.findall(line)}
        elif line.startswith("HIST "):
            kv = dict(RE_KV.findall(line))
            rec["hist"].append((int(kv["cycles_ge"]), int(kv["count"])))
        elif line.startswith("HISTCHECK "):
            rec["histcheck"] = {k: float(v) for k, v in RE_KV.findall(line)}
    # trial-to-trial spread of the aggregate
    rec["smids"] = sorted({int(c["smid"]) for c in rec["ctas"] if c.get("trial") == 1 and c.get("bytes", 0) > 0})
    rec["active_sms"] = len(rec["smids"])
    aggs = [t["agg_B_per_ns"] for t in rec["trials"] if t.get("agg_B_per_ns")]
    if len(aggs) >= 2:
        rec["agg_cv_pct"] = round(100 * statistics.pstdev(aggs) / statistics.mean(aggs), 2)
    lds = [t["ld_bytes"] for t in rec["trials"] if t.get("ld_bytes")]
    sts = [t["st_bytes"] for t in rec["trials"] if t.get("st_bytes")]
    if lds and sts and rec["result"].get("agg_B_per_ns"):
        tot = statistics.median(lds) + statistics.median(sts)
        rec["result"]["ld_B_per_ns"] = round(rec["result"]["agg_B_per_ns"] * statistics.median(lds) / tot, 2)
        rec["result"]["st_B_per_ns"] = round(rec["result"]["agg_B_per_ns"] * statistics.median(sts) / tot, 2)
    return rec


NCU_METRICS = ",".join([
    "sm__cycles_active.max", "sm__cycles_elapsed.avg.per_second", "gpu__time_duration.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum",
    "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
    "lts__t_sector_hit_rate.pct", "l1tex__t_bytes.sum",
])


def run_ncu(ncu: str, binary: str, args: list[str], clock: str, out_txt: Path,
            kernel_re: str = "tma_scan_kernel") -> dict:
    cmd = [ncu, "--clock-control", clock, "--cache-control", "none",
           "--kernel-name", f"regex:{kernel_re}", "--launch-skip", "1", "--launch-count", "1",
           "--metrics", NCU_METRICS, "--csv", binary, *args]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_txt.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    row = {"rc": proc.returncode, "seconds": round(time.monotonic() - t0, 1), "metrics": {}}
    start = proc.stdout.find('"ID"')
    if start < 0:
        return row
    for r in csv.DictReader(io.StringIO(proc.stdout[start:])):
        name = r.get("Metric Name")
        if name:
            row["metrics"][name] = {"value": r.get("Metric Value"), "unit": r.get("Metric Unit")}
    return row


def write_summary(out: Path, results: list[dict], ncu_rows: list[dict]) -> None:
    lines = ["# scan summary", ""]
    groups = [
        ("lockstep groups (G sweep)", "tma_l2g_24mb_g"),
        ("disjoint slices", "tma_l2s_24mb_"),
        ("lockstep groups (G sweep)", "ld_l2g_24mb_g"),
        ("disjoint slices", "ld_l2s_24mb_"),
    ]
    for title, prefix in groups:
        rows = [r for r in results if r["config"]["name"].startswith(prefix)]
        if not rows:
            continue
        lines += [f"## {title}", "",
                  "| config | nsm | active SMs | agg B/ns | per-SM B/cycle | MHz | window ms | CV% | placement |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in rows:
            res = r["result"]
            pl = r.get("placement") or {}
            act = r.get("active_sms") or r["config"]["nsm"]
            lines.append(
                f"| {r['config']['name']} | {r['config']['nsm']} | {act} | {res.get('agg_B_per_ns', 0):.1f} | "
                f"{res.get('per_sm_B_per_cycle', 0):.2f} | "
                f"{res.get('mhz', 0):.0f} | {res.get('window_ns', 0) / 1e6:.2f} | {r.get('agg_cv_pct', '')} | "
                f"{pl.get('distinct_sm', '?')}/{pl.get('nsm', '?')} |")
        lines.append("")
    if ncu_rows:
        lines += ["## ncu cross-check", "", "| config | clock | rc | " + " | ".join(NCU_METRICS.split(",")) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in NCU_METRICS.split(",")) + " |"]
        for row in ncu_rows:
            vals = [str((row["metrics"].get(m) or {}).get("value", "")) for m in NCU_METRICS.split(",")]
            lines.append(f"| {row['name']} | {row['clock']} | {row['rc']} | " + " | ".join(vals) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))


def detect_sms(binary: str, which: str = "m6r2") -> int:
    """SM count from the binary's META sms= line (tiny 1-SM run)."""
    argv = [binary, "ld", "1", "32", "1", "1", "1", "region=slice"] if which == "m1r4" \
        else [binary, "1", "8", "1", "1"]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"run_scan.py: SM-count probe failed rc={proc.returncode}: {proc.stderr[:400]}")
    m = re.search(r"^META .*?\bsms=(\d+)", proc.stdout, re.M)
    if not m:
        sys.exit("run_scan.py: no sms= in META line of the SM-count probe")
    return int(m.group(1))


def configs(trials: int, sms: int, which: str = "m6r2") -> list[dict]:
    out = []

    def nsm(g: int) -> int:
        return sms // g * g

    def add(name, nsm_, footprint, per_cta, **kv):
        args = [str(nsm_), str(footprint), str(per_cta), str(trials)]
        args += [f"{k}={v}" for k, v in kv.items()]
        out.append({"name": name, "mode": "tma", "nsm": nsm_, "threads": 32,
                    "footprint_mb": footprint, "per_cta_mb": per_cta, "kv": kv,
                    "args": args})

    if which == "m1r4":
        return configs_m1r4(trials, sms)

    # m6r2 (l2_merge protocol): merge ceiling and merge window on the TMA path.
    # (a) aligned-start lockstep groups, G sweep: lts/SM-side bytes vs G
    #     separates "at most N requesters per slice read" (ceil(G/N)/G
    #     steps) from a bandwidth-ratio floor.  nsm(G) = floor(sms/G)*G.
    for g in (2, 3, 4, 6, 8, 11, 12, 16, 22, 32, 44, 132):
        add(f"tma_l2g_24mb_g{g:03d}_sync", nsm(g), 24, 256, group=g, sync_start=1)
    # (b) controlled inter-request gap: member m of a pair starts
    #     m * shift cycles after the aligned start; lts/SM rises from the
    #     lockstep value to 1.0 where the gap exceeds the merge window.
    for d in (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096):
        add(f"tma_l2g_24mb_g002_shift{d:05d}", nsm(2), 24, 256, group=2, shift_cycles=d)
    for d in (128, 256, 512, 1024, 2048):
        add(f"tma_l2g_24mb_g004_shift{d:05d}", nsm(4), 24, 256, group=4, shift_cycles=d)
    # (c) references (unaligned lockstep / rotated / disjoint slices)
    add("tma_l2g_24mb_g002_lock", nsm(2), 24, 256, group=2)
    add("tma_l2g_24mb_g002_rot", nsm(2), 24, 256, group=2, group_rot=1)
    add(f"tma_l2s_24mb_n{sms:03d}", sms, 24, 256)
    return out


def configs_m1r4(trials: int, sms: int) -> list[dict]:
    """m1r4 (l2_merge protocol v2): delivered-bytes-per-cycle on the LDG path
    (probe/ldg_scan, <mode> <nsm> <threads> <footprint_mb> <per_cta_mb>
    <trials> [k=v...]).  nsm(G) = floor(sms/G)*G."""
    out = []

    def nsm(g: int) -> int:
        return sms // g * g

    def add(name, mode, nsm_, threads, footprint, per_cta, **kv):
        args = [mode, str(nsm_), str(threads), str(footprint), str(per_cta), str(trials)]
        args += [f"{k}={v}" for k, v in kv.items()]
        out.append({"name": name, "mode": mode, "nsm": nsm_, "threads": threads,
                    "footprint_mb": footprint, "per_cta_mb": per_cta, "kv": kv,
                    "args": args})

    for g in (2, 4, 6, 12, 22, 44, 66, 132):
        add(f"ld_l2g_24mb_g{g:03d}_lock", "ld", nsm(g), 1024, 24, 256, region="slice", group=g)
    for g in (2, 4, 12, 44):
        add(f"ld_l2g_24mb_g{g:03d}_lock_stride2mb", "ld", nsm(g), 1024, 24, 256,
            region="slice", group=g, slice_stride_mb=2)
    for g in (2, 4, 6, 12, 22, 44, 66, 132):
        add(f"ld_l2g_24mb_g{g:03d}_rot", "ld", nsm(g), 1024, 24, 256,
            region="slice", group=g, group_rot=1)
    add(f"ld_l2s_24mb_t1024_n{sms:03d}_base", "ld", sms, 1024, 24, 256, region="slice")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--ncu", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--sms", type=int, default=0)
    ap.add_argument("--set", default="m6r2", choices=["m6r2", "m1r4"])
    a = ap.parse_args()
    sms = a.sms or detect_sms(a.bin, a.set)
    print(f"run_scan.py: sms={sms}", flush=True)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in configs(a.trials, sms, a.set):
        if a.only and not any(cfg["name"].startswith(x) for x in a.only.split(",")):
            continue
        t0 = time.monotonic()
        proc = subprocess.run([a.bin, *cfg["args"]], capture_output=True, text=True)
        (out / "raw" / f"{cfg['name']}.txt").write_text(
            proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
        rec = {"config": cfg, "rc": proc.returncode, "seconds": round(time.monotonic() - t0, 1)}
        rec.update(parse(proc.stdout))
        results.append(rec)
        res = rec["result"]
        print(f"{cfg['name']}: rc={proc.returncode} agg={res.get('agg_B_per_ns')} B/ns "
              f"per_sm={res.get('per_sm_B_per_cycle')} B/cyc mhz={res.get('mhz')} "
              f"placement={rec.get('placement')}", flush=True)
    ncu_rows = []
    if a.ncu:
        (out / "ncu").mkdir(exist_ok=True)
        # trials position in the argv layout: 3 for TMA (<nsm> <fp> <per_cta>
        # <trials>), 5 for LDG (<mode> <nsm> <threads> <fp> <per_cta> <trials>)
        tidx, kernel_re = (3, "tma_scan_kernel") if a.set == "m6r2" else (5, "scan_kernel")
        for r in results:
            if r["rc"] != 0:
                continue
            args = list(r["config"]["args"]); args[tidx] = "1"
            row = run_ncu(a.ncu, a.bin, args, "none", out / "ncu" / f"{r['config']['name']}_none.txt",
                          kernel_re)
            row.update({"name": r["config"]["name"], "clock": "none"})
            ncu_rows.append(row)
            print(f"ncu {r['config']['name']}: rc={row['rc']}", flush=True)
    (out / "results.json").write_text(json.dumps(
        {"bin": a.bin, "trials": a.trials, "set": a.set, "sms": sms,
         "results": results, "ncu": ncu_rows},
        indent=1, sort_keys=True) + "\n")
    write_summary(out, results, ncu_rows)
    # lts/SM = ncu lts__t_bytes / SM-side bytes of one launch (slice-side
    # share of the delivered bytes); start spread = max start_ns of the
    # group members that carry no shift (alignment check, ns).  For m1r4 the
    # SM-side denominator is l1tex__t_bytes (r = lts/l1tex per protocol v2).
    lts = {row["name"]: row["metrics"].get("lts__t_bytes.sum", {}).get("value") for row in ncu_rows}
    l1 = {row["name"]: row["metrics"].get("l1tex__t_bytes.sum", {}).get("value") for row in ncu_rows}
    if a.set == "m1r4":
        lines = ["# M1 r4 LDG scan", "",
                 "| config | nsm | active SMs | agg B/ns | per-SM B/cycle | MHz | CV% | SM bytes | lts bytes | l1tex bytes | r=lts/l1tex |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in results:
            res = r["result"]
            name = r["config"]["name"]
            sm_bytes = next((t["total_bytes"] for t in r["trials"] if t.get("total_bytes")), 0)
            lts_bytes = float(str(lts.get(name) or "0").replace(",", ""))
            l1_bytes = float(str(l1.get(name) or "0").replace(",", ""))
            ratio = f"{lts_bytes / l1_bytes:.3f}" if l1_bytes and lts_bytes else ""
            lines.append(f"| {name} | {r['config']['nsm']} | {r.get('active_sms')} | {res.get('agg_B_per_ns', 0):.1f} | "
                         f"{res.get('per_sm_B_per_cycle', 0):.2f} | {res.get('mhz', 0):.0f} | {r.get('agg_cv_pct', '')} | "
                         f"{sm_bytes:.0f} | {lts_bytes:.0f} | {l1_bytes:.0f} | {ratio} |")
        (out / "m1_summary.md").write_text("\n".join(lines) + "\n")
        return 0
    lines = ["# M6 TMA scan", "",
             "| config | nsm | active SMs | agg B/ns | per-SM B/cycle | MHz | CV% | SM bytes | lts bytes | lts/SM | m0 start spread ns |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        res = r["result"]
        name = r["config"]["name"]
        sm_bytes = next((t["total_bytes"] for t in r["trials"] if t.get("total_bytes")), 0)
        lts_bytes = float(str(lts.get(name) or "0").replace(",", ""))
        ratio = f"{lts_bytes / sm_bytes:.3f}" if sm_bytes and lts_bytes else ""
        group = int(r["config"]["kv"].get("group", 1))
        m0 = [c.get("start_ns", 0) for c in r["ctas"] if c.get("trial") == 1 and int(c["cta"]) % group == 0]
        spread = f"{max(m0):.0f}" if m0 else ""
        lines.append(f"| {name} | {r['config']['nsm']} | {r.get('active_sms')} | {res.get('agg_B_per_ns', 0):.1f} | "
                     f"{res.get('per_sm_B_per_cycle', 0):.2f} | {res.get('mhz', 0):.0f} | {r.get('agg_cv_pct', '')} | "
                     f"{sm_bytes:.0f} | {lts_bytes:.0f} | {ratio} | {spread} |")
    (out / "m6_summary.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
