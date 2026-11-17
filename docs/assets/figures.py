#!/usr/bin/env python3
"""Draw the README figures (light and dark variants, text as paths).

    python3 docs/assets/figures.py [--journal DIR] [--png DIR]

Writes docs/assets/{logo,overview,accuracy}-{light,dark}.svg.  The accuracy
figure is drawn from suites/<arch>/full_gpu/expected.tsv.  With --journal
also record-{light,dark}.svg, the issue journal of one thread block; DIR is
the out-cpp directory of a `cyclescope run --ledgers` (the README shows the
example capture hopper/deepgemm_einsum_05).  Needs matplotlib and the Lato
font; --png also writes previews to DIR.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

RUNTIME, STATIC, MECH, RED = "#3B8FE0", "#F0721C", "#35A26B", "#D65050"
THEMES = {
    "light": {"fg": "#1F2328", "muted": "#59636E", "line": "#8C959F",
              "card": "#FFFFFF", "band": "#F3F5F7", "grid": "#D8DEE4"},
    "dark": {"fg": "#E6EDF3", "muted": "#9198A1", "line": "#6E7681",
             "card": "#161B22", "band": "#1C2129", "grid": "#30363D"},
}
plt.rcParams.update({"font.family": "Lato", "svg.fonttype": "path",
                     "svg.hashsalt": "cyclescope"})


def rbox(ax, x, y, w, h, *, fc, ec, lw=1.4, r=12, z=2, alpha=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def tint(ax, x, y, w, h, color, r=12):
    """A card: theme-independent translucent fill + coloured outline."""
    rbox(ax, x, y, w, h, fc=color, ec="none", r=r, alpha=0.11, z=2)
    rbox(ax, x, y, w, h, fc="none", ec=color, r=r, lw=1.6, z=3)


def arrow(ax, x0, y0, x1, y1, color, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.26",
                                color=color, lw=lw, shrinkA=0, shrinkB=0))


def overview(theme: str):
    t = THEMES[theme]
    fig = plt.figure(figsize=(12, 4.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 490)
    ax.axis("off")

    # ---- the three kinds of facts -------------------------------------
    cards = [
        (20, RUNTIME, "Runtime facts", "what every warp did",
         "captured by a tracer on the real GPU"),
        (415, STATIC, "Static facts", "what the compiler decided",
         "read from the kernel binary"),
        (810, MECH, "Mechanism facts", "how the hardware responds",
         "measured by microbenchmarks"),
    ]
    top, ch, cw = 300, 150, 370
    for x, c, title, what, source in cards:
        tint(ax, x, top, cw, ch, c)
        ax.text(x + 22, top + ch - 30, title, color=c, fontsize=19,
                fontweight="bold", va="center", zorder=5)
        ax.text(x + 22, top + ch - 62, what, color=t["fg"], fontsize=14.5,
                va="center", zorder=5)
        ax.text(x + 22, top + 20, source, color=t["muted"], fontsize=12.5,
                va="center", style="italic", zorder=5)
    # runtime glyph: per-warp instruction streams, dark = operands recorded
    gx, gy = 42, top + 44
    for row, marks in enumerate(((3, 9, 13, 19), (1, 6, 11, 16), (4, 8, 14, 20))):
        for i in range(22):
            ax.add_patch(Rectangle((gx + i * 14.6, gy + row * 12), 11, 8,
                                   fc=RUNTIME, ec="none", zorder=5,
                                   alpha=1.0 if i in marks else 0.38))
    # static glyph: opcode, registers, control word of one instruction
    gx, gy = 437, top + 44
    ax.add_patch(Rectangle((gx, gy + 24), 96, 8, fc=t["line"], ec="none", zorder=5))
    for i in range(4):
        ax.add_patch(Rectangle((gx + 108 + i * 19, gy + 24), 13, 8, fc="none",
                               ec=t["line"], lw=1.1, zorder=5))
    ax.add_patch(Rectangle((gx, gy + 12), 210, 8, fc=STATIC, ec="none",
                           alpha=0.8, zorder=5))
    ax.add_patch(Rectangle((gx, gy), 150, 8, fc=STATIC, ec="none",
                           alpha=0.38, zorder=5))
    # mechanism glyph: a tree whose leaves are measured values
    gx, gy = 834, top + 42
    for k, cx in enumerate((30, 100, 170)):
        ax.plot([gx + 100, gx + cx], [gy + 36, gy + 20], color=MECH, lw=1.3, zorder=5)
        for dx in (-18, 18):
            ax.plot([gx + cx, gx + cx + dx], [gy + 20, gy + 6], color=MECH,
                    lw=1.3, zorder=5)
            ax.add_patch(Rectangle((gx + cx + dx - 7, gy - 4), 14, 10, fc=MECH,
                                   ec="none", alpha=0.45 + 0.18 * k, zorder=6))
        ax.plot([gx + cx], [gy + 20], "o", color=MECH, ms=4.5, zorder=6)
    ax.plot([gx + 100], [gy + 36], "o", color=MECH, ms=4.5, zorder=6)

    # ---- scope labels ---------------------------------------------------
    ax.text(20 + (765) / 2, top + ch + 18, "ONCE PER KERNEL", color=t["muted"],
            fontsize=11.5, ha="center", va="center", fontweight="bold")
    ax.text(810 + cw / 2, top + ch + 18, "ONCE PER GPU MODEL", color=t["muted"],
            fontsize=11.5, ha="center", va="center", fontweight="bold")

    # ---- facts flow into the pipeline ----------------------------------
    rbox(ax, 4, 26, 1192, 194, fc=t["band"], ec="none", r=16, z=1)
    for x in (20 + cw / 2, 415 + cw / 2, 810 + cw / 2):
        arrow(ax, x, top - 4, x, 224, t["line"])

    # ---- pipeline ---------------------------------------------------------
    py, ph = 44, 158
    stages = [
        (20, 250, "Convert", "facts → one segment", "per group of thread blocks"),
        (310, 250, "Build", "segments → one job", "per kernel launch"),
        (600, 250, "Simulate", "the engine replays the job", "cycle by cycle"),
    ]
    for x, w, name, l1, l2 in stages:
        rbox(ax, x, py, w, ph, fc=t["card"], ec=t["line"], lw=1.4)
        ax.text(x + w / 2, py + ph - 34, name, color=t["fg"], fontsize=19,
                fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x + w / 2, py + 62, l1, color=t["fg"], fontsize=13.5,
                ha="center", va="center", zorder=5)
        ax.text(x + w / 2, py + 36, l2, color=t["muted"], fontsize=13,
                ha="center", va="center", zorder=5)
    for x in (270, 560, 850):
        arrow(ax, x + 6, py + ph / 2, x + 34, py + ph / 2, t["line"])
    # the execution record
    rx, rw = 890, 290
    rbox(ax, rx, py, rw, ph, fc=t["card"], ec=t["fg"], lw=2.2)
    ax.text(rx + rw / 2, py + ph - 34, "Execution record", color=t["fg"],
            fontsize=19, fontweight="bold", ha="center", va="center", zorder=5)
    rows = [
        ("warp", ((0, .16, RUNTIME), (.16, .45, STATIC), (.45, .62, RUNTIME),
                  (.62, .86, STATIC), (.86, 1, RUNTIME))),
        ("warp", ((0, .30, STATIC), (.30, .50, RUNTIME), (.50, .78, STATIC),
                  (.78, 1, RUNTIME))),
        ("resource", ((0, .28, MECH), (.28, .64, RED), (.64, 1, MECH))),
    ]
    bx, bw = rx + 92, 172
    for k, (lab, segs) in enumerate(rows):
        y = py + 84 - k * 21
        ax.text(bx - 10, y + 6, lab, color=t["muted"], fontsize=12, ha="right",
                va="center", zorder=5)
        for a, b, c in segs:
            ax.add_patch(Rectangle((bx + a * bw, y), (b - a) * bw - 1.5, 12,
                                   fc=c, ec="none", alpha=0.85, zorder=5))
    ax.text(rx + rw / 2, py + 19, "who issued, who waited, and why",
            color=t["muted"], fontsize=13, ha="center", va="center", zorder=5)
    return fig


def logo(theme: str):
    t = THEMES[theme]
    fig = plt.figure(figsize=(7.6, 1.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 760)
    ax.set_ylim(0, 170)
    ax.axis("off")
    # mark: three warps of an execution record
    x0, y0, w = 150, 43, 96
    rows = (((0, .30, RUNTIME), (.30, .72, STATIC), (.72, 1, RUNTIME)),
            ((0, .52, STATIC), (.52, .80, RUNTIME), (.80, 1, MECH)),
            ((0, .22, MECH), (.22, .60, RUNTIME), (.60, 1, STATIC)))
    for k, segs in enumerate(rows):
        for a, b, c in segs:
            rbox(ax, x0 + a * w, y0 + (2 - k) * 31, (b - a) * w - 5, 22, fc=c,
                 ec="none", r=5, z=3)
    ax.text(x0 + w + 22, 84, "CycleScope", color=t["fg"], fontsize=50,
            fontweight="black", va="center", ha="left")
    return fig


WAITS = (   # wait bucket id (docs/en/results.md), label, colour
    (5, "register value or compiler schedule", RUNTIME),
    (4, "busy unit, port or queue", MECH),
    (3, "semantic gate (poll, async group)", STATIC),
    (2, "barrier release", "#8E6BD8"),
    (1, "instruction fetch", "#9AA4AF"),
)


def timeline(theme: str, run_dir: Path, warps: int = 8):
    """Issue journal of the first thread block of one simulated launch."""
    import lzma
    import sys
    sys.path.insert(0, str(REPO / "python"))
    from cyclescope.job.result import read_journal, read_result
    t = THEMES[theme]
    jp = run_dir / "journal.bin"
    raw = jp.read_bytes() if jp.is_file() else \
        lzma.open(run_dir / "journal.bin.xz").read()
    cycles = read_result((run_dir / "result.bin").read_bytes())["cycles"]
    rows = [r for r in read_journal(raw)["JRNL"] if r["warp"] < warps]
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_axes([0.065, 0.3, 0.92, 0.6])
    ax.set_facecolor("none")
    for w in range(warps):
        y = warps - 1 - w
        mine = [r for r in rows if r["warp"] == w]
        for bucket, _label, color in WAITS:
            spans = [(r["issue_cycle"] - r["wait_cycles"], r["wait_cycles"])
                     for r in mine
                     if r["wait_bucket"] == bucket and r["wait_cycles"] > 0]
            ax.broken_barh(spans, (y - 0.36, 0.72), fc=color, ec="none",
                           zorder=3, rasterized=True)
        ax.vlines([r["issue_cycle"] for r in mine], y - 0.36, y + 0.36,
                  color=t["fg"], lw=0.35, alpha=0.75, zorder=4, rasterized=True)
    ax.set_xlim(0, cycles)
    ax.set_ylim(-0.6, warps - 0.4)
    ax.set_yticks(range(warps))
    ax.set_yticklabels([f"warp {warps - 1 - y}" for y in range(warps)])
    ax.tick_params(colors=t["muted"], labelsize=11.5, length=0)
    ax.tick_params(axis="x", length=4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(t["line"])
    ax.set_xlabel("cycle", color=t["fg"], fontsize=12.5, labelpad=4)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="none") for _b, _l, c in WAITS]
    handles.append(plt.Line2D([0], [0], color=t["fg"], lw=1.2))
    labels = ["waiting: " + label for _b, label, _c in WAITS] + \
        ["an instruction issues"]
    leg = fig.legend(handles, labels, loc="lower center", ncol=3,
                     frameon=False, fontsize=11.5, handlelength=1.3,
                     columnspacing=2.2, bbox_to_anchor=(0.52, 0.0))
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    return fig


def read_expected(arch: str):
    with (REPO / "suites" / arch / "full_gpu" / "expected.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return ([float(r["hw_cycles"]) for r in rows],
            [float(r["sim_cycles"]) for r in rows])


def accuracy(theme: str):
    t = THEMES[theme]
    data = {"H100": read_expected("hopper"), "B300": read_expected("blackwell")}
    colors = {"H100": RUNTIME, "B300": STATIC}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.7),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.14, wspace=0.2)
    for ax in (a, b):
        ax.set_facecolor("none")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["line"])
        ax.tick_params(colors=t["muted"], labelsize=11.5, length=4)
        ax.grid(True, color=t["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    # (a) parity
    lo = min(min(min(v) for v in d) for d in data.values()) * 0.7
    hi = max(max(max(v) for v in d) for d in data.values()) * 1.4
    a.plot([lo, hi], [lo, hi], color=t["fg"], lw=1.1, zorder=2)
    for f in (0.8, 1.2):
        a.plot([lo, hi], [lo * f, hi * f], color=t["line"], lw=0.9, ls=(0, (4, 3)),
               zorder=2)
    for dev, (hw, sim) in data.items():
        a.scatter(hw, sim, s=13, color=colors[dev], alpha=0.55, lw=0, zorder=3,
                  label=f"{dev}  ·  {len(hw)} kernels")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.set_xlabel("hardware cycles (Nsight Compute)", color=t["fg"], fontsize=12.5)
    a.set_ylabel("simulated cycles", color=t["fg"], fontsize=12.5)
    a.set_title("Simulated vs. measured, 1,000 kernels", color=t["fg"],
                fontsize=14, fontweight="bold", loc="left", pad=10)
    leg = a.legend(loc="upper left", frameon=False, fontsize=11.5,
                   handletextpad=0.2, borderaxespad=0.3, markerscale=1.8)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    a.text(lo * 2.6, lo * 2.6 * 0.8 * 0.86, "±20 %", color=t["muted"],
           fontsize=11, ha="left", va="top", rotation=38)
    # (b) signed error distribution
    edges = [x * 2.5 for x in range(-20, 21)]
    for dev, (hw, sim) in data.items():
        err = [(s / h - 1) * 100 for h, s in zip(hw, sim)]
        b.hist(err, bins=edges, color=colors[dev], alpha=0.2, zorder=3,
               histtype="stepfilled", lw=0)
        b.hist(err, bins=edges, color=colors[dev], zorder=4, histtype="step",
               lw=1.9, label=dev)
    b.axvline(0, color=t["fg"], lw=1.0, zorder=2)
    b.set_xlim(-50, 50)
    b.set_xlabel("simulated / hardware − 1   (%)", color=t["fg"], fontsize=12.5)
    b.set_ylabel("kernels", color=t["fg"], fontsize=12.5)
    b.set_title("Error distribution", color=t["fg"], fontsize=14,
                fontweight="bold", loc="left", pad=10)
    leg = b.legend(loc="upper right", frameon=False, fontsize=11.5)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path)
    ap.add_argument("--journal", type=Path)
    a = ap.parse_args()
    figures = [("logo", logo), ("overview", overview), ("accuracy", accuracy)]
    if a.journal:
        figures.append(("record", lambda theme: timeline(theme, a.journal)))
    for name, draw in figures:
        for theme in THEMES:
            fig = draw(theme)
            fig.savefig(OUT / f"{name}-{theme}.svg", transparent=True,
                        metadata={"Date": None}, dpi=220)
            if a.png:
                a.png.mkdir(parents=True, exist_ok=True)
                fig.savefig(a.png / f"{name}-{theme}.png", dpi=110,
                            facecolor="#FFFFFF" if theme == "light"
                            else "#0D1117")
            plt.close(fig)


if __name__ == "__main__":
    main()
