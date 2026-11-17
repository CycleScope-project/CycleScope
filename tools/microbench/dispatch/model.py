"""dispatch model (protocol.json derivation):
  aggregate_interval_ns: per launch, entries sorted by gt_in; backlog
    window = first gt_out to last gt_in; interval = window/entries;
    median over launches 1..5.
  kernel_slope_ns_per_cta: OLS of (last gt_out - first gt_in) over grid
    size across waves 2/8/32 at one ctas_per_sm.
  serial_ns_per_cta: c16 aggregate, two-run mean.
  first_wave_spread_ns: max-min gt_in over first wave per launch,
    median; first_wave_interval = spread/(CTAs-1) if spread > 2 ticks
    (tick = smallest positive distinct gt_in delta) else unresolved
    bound spread/(CTAs-1).
  tail: slot_free_to_entry median over launches.
"""
import json, re, statistics, sys
from pathlib import Path

def load(path):
    ctas, launches = [], {}
    for l in Path(path).read_text().splitlines():
        if l.startswith("CTA "):
            d = dict(t.split("=") for t in l[4:].split())
            ctas.append({k: (float(v) if k.startswith(("gt_", "clk_"))
                          else int(v)) for k, v in d.items()})
        elif l.startswith("LAUNCH "):
            d = dict(t.split("=") for t in l[7:].split())
            launches[int(d["launch"])] = float(d["host_ns"])
    return ctas, launches

def agg_interval(ctas):
    per_launch = {}
    for c in ctas:
        per_launch.setdefault(c["launch"], []).append(c)
    out = []
    for L in sorted(per_launch)[1:6]:
        rows = per_launch[L]
        ents = sorted(r["gt_in"] for r in rows)
        first_out = min(r["gt_out"] for r in rows)
        win = max(ents) - first_out
        inside = sum(1 for e in ents if e >= first_out)
        if inside:
            out.append(win / inside)
    return statistics.median(out) if out else None

def kernel_slope(ctas, launches):
    per_launch = {}
    for c in ctas:
        per_launch.setdefault(c["launch"], []).append(c)
    xs, ys = [], []
    for L in sorted(per_launch)[1:]:
        rows = per_launch[L]
        xs.append(len(rows))
        ys.append(max(r["gt_out"] for r in rows)
                  - min(r["gt_in"] for r in rows))
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) \
        / sum((x - mx) ** 2 for x in xs)

def first_wave(ctas, ctas_per_sm, sm_count):
    per_launch = {}
    for c in ctas:
        per_launch.setdefault(c["launch"], []).append(c)
    spreads, ticks = [], []
    n = sm_count * ctas_per_sm
    for L in sorted(per_launch)[1:]:
        rows = sorted(per_launch[L], key=lambda r: r["gt_in"])[:n]
        g = sorted(r["gt_in"] for r in rows)
        diffs = [b - a for a, b in zip(g, g[1:]) if b > a]
        tick = min(diffs) if diffs else 0
        spreads.append((g[-1] - g[0], tick))
    med_sp = statistics.median(s for s, _ in spreads)
    med_tk = statistics.median(t for _, t in spreads)
    return med_sp, med_tk, n

def tail_gap(path):
    gaps = []
    for l in Path(path).read_text().splitlines():
        if l.startswith("CTA "):
            d = dict(t.split("=") for t in l[4:].split())
            if int(d["launch"]) > 0:
                pass
    # tail: the extra CTA's gt_in - first-wave max gt_out per launch
    per_launch = {}
    for l in Path(path).read_text().splitlines():
        if l.startswith("CTA "):
            d = dict(t.split("=") for t in l[4:].split())
            per_launch.setdefault(int(d["launch"]), []).append(d)
    for L in sorted(per_launch)[1:]:
        rows = per_launch[L]
        first = sorted(rows, key=lambda r: float(r["gt_in"]))[:-1]
        last = sorted(rows, key=lambda r: float(r["gt_in"]))[-1]
        gaps.append(float(last["gt_in"])
                    - max(float(r["gt_out"]) for r in first))
    return statistics.median(gaps) if gaps else None

def tail_clock_gap(path, per_launch=False):
    """r4 addendum: extra CTA's clk_in minus the previous CTA's clk_out on
    the same smid, in ns via that SM's measured clock rate
    f = (clk_out - clk_in) / (gt_out - gt_in) of the previous CTA."""
    per_l = {}
    for l in Path(path).read_text().splitlines():
        if not l.startswith("CTA "):
            continue
        d = dict(t.split("=") for t in l[4:].split())
        per_l.setdefault(int(d["launch"]), []).append(d)
    gaps, dropped = [], 0
    for L in sorted(per_l)[1:10]:   # launches 1..9, launch 0 dropped
        rows = per_l[L]
        last = sorted(rows, key=lambda r: float(r["gt_in"]))[-1]
        smid = int(last["smid"])
        peers = [r for r in rows
                 if r is not last and int(r["smid"]) == smid]
        if len(peers) != 1:
            dropped += 1
            continue
        prev = peers[0]
        gap_cycles = float(last["clk_in"]) - float(prev["clk_out"])
        span_c = float(prev["clk_out"]) - float(prev["clk_in"])
        span_g = float(prev["gt_out"]) - float(prev["gt_in"])
        if span_g <= 0 or span_c <= 0:
            dropped += 1
            continue
        gaps.append(gap_cycles / (span_c / span_g))
    if per_launch:
        return gaps
    if dropped:
        print(f"tail_clock_gap: dropped {dropped} launches",
              file=sys.stderr)
    return statistics.median(gaps) if gaps else None


def model(root, device, sm_count):
    out = {}
    agg = {}
    for run in ("run-a", "run-b"):
        for c in (4, 16):
            vals, slopes = [], []
            for w in (2, 8, 32):
                ctas, L = load(
                    Path(root) / run / f"tput.c{c}.w{w}.txt")
                a = agg_interval(ctas)
                if a: vals.append(a)
            agg[f"c{c}_{run}"] = statistics.median(vals)
            # kernel slope across waves at this c
            allc = {}
            for w in (2, 8, 32):
                ctas, L = load(
                    Path(root) / run / f"tput.c{c}.w{w}.txt")
                for cc in ctas:
                    cc = dict(cc)
                    allc.setdefault(w, []).append(cc)
            xs, ys = [], []
            for w in (2, 8, 32):
                rows = allc[w]
                for L2 in sorted({r["launch"] for r in rows})[1:]:
                    rr = [r for r in rows if r["launch"] == L2]
                    xs.append(len(rr))
                    ys.append(max(r["gt_out"] for r in rr)
                              - min(r["gt_in"] for r in rr))
            mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
            agg[f"c{c}_{run}_slope"] = (
                sum((x-mx)*(y-my) for x, y in zip(xs, ys))
                / sum((x-mx)**2 for x in xs))
    out["aggregate_interval_ns"] = agg
    out["serial_ns_per_cta"] = (
        agg["c16_run-a"] + agg["c16_run-b"]) / 2
    out["kernel_slope_ns_per_cta_c16"] = statistics.median(
        [agg["c16_run-a_slope"], agg["c16_run-b_slope"]])
    for c in (4, 16):
        sp, tk, n = first_wave(
            [c2 for run in ("run-a", "run-b")
             for w in (2, 8, 32)
             for c2 in load(Path(root)/run/f"tput.c{c}.w{w}.txt")[0]],
            c, sm_count)
        out[f"first_wave_c{c}"] = {
            "spread_ns": sp, "tick_ns": tk, "ctas": n,
            "resolved": sp > 2 * tk,
            "bound_ns": sp / (n - 1)}
    gaps = [tail_gap(Path(root) / run / "tail.txt")
            for run in ("run-a", "run-b")]
    out["slot_free_to_entry_ns"] = gaps
    # r4 addendum: same-SM clock64 gap (gt ticks are 32 ns on h100)
    cg = {run: tail_clock_gap(Path(root) / run / "tail.txt")
          for run in ("run-a", "run-b")}
    out["slot_free_to_entry_clock_ns"] = cg
    # launch CV per run over the kept per-launch values
    cv = {}
    for run in ("run-a", "run-b"):
        vals = tail_clock_gap(Path(root) / run / "tail.txt",
                              per_launch=True)
        if vals and statistics.mean(vals) > 0:
            cv[run] = (statistics.stdev(vals) / statistics.mean(vals)
                       if len(vals) > 1 else 0.0)
        else:
            cv[run] = None
    out["slot_free_to_entry_clock_cv"] = cv
    # consistency: clock gap inside the same run's gt bracket (gt gap
    # resolves to >= one 32 ns tick; bracket = [gap, gap + tick))
    out["slot_free_to_entry_consistency"] = {
        run: {"clock_ns": cg[run], "gt_ns": gaps[i]}
        for i, run in enumerate(("run-a", "run-b"))}
    return out

if __name__ == "__main__":
    root, device = sys.argv[1], sys.argv[2]
    sm = 132 if device == "h100" else 148
    print(json.dumps(model(root, device, sm), indent=1))
