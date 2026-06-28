#!/usr/bin/env python3
"""Plot the selsweep selection-cost frontier.

Consumes the CSV that `trifle-bench selsweep` writes (columns
`arm,knob,N,k,recall,sigma_df_p50,sigma_df_p99,lat_p50_us,lat_p99_us`) and draws
recall@k against the work it cost — Σdf and/or p99 latency. One selsweep run can
now sweep a whole N ladder (`--docs 1000,5000,25000,...`), so the file carries
several N; pick how to view them with `--mode`:

  facet    (default) one panel-row per N, the two arms (t_max, df_budget)
           overlaid per work axis. The per-N frontier; better arm sits up-left.
  overlay  every N on shared axes (color = N, linestyle = arm). How the frontier
           shifts as the corpus grows.
  knee     the scaling analysis: for each N, the cheapest df_budget reaching
           ~its full recall (the knee), then optimal-df_budget vs N with the
           fraction budget*/N and a log-log fit — is the knee a constant slice
           of N? Prints the numbers to stderr.

The `>>`-appended workflow writes one header line per run; those repeats are
skipped on read, so a hand-concatenated file works the same as a single ladder.

    python3 plot_selsweep.py frontier.csv                      # -> selsweep.png
    python3 plot_selsweep.py frontier.csv --mode overlay
    python3 plot_selsweep.py frontier.csv --mode knee --knee-frac 0.98
    cat frontier.csv | python3 plot_selsweep.py - --out f.svg --show

Needs matplotlib (+ numpy, which it pulls in).
"""

import argparse
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless by default; --show flips to an interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Column order emitted by cmd_selsweep (see benchmarks/src/main.rs).
COLS = (
    "arm",
    "knob",
    "N",
    "k",
    "recall",
    "sigma_df_p50",
    "sigma_df_p99",
    "lat_p50_us",
    "lat_p99_us",
)

# The two work axes the frontier is read against, keyed by --x choice.
# (label, column, log-x) — Σdf spans the 0.005..1.0 fraction grid, so log; the
# latency range is narrow, so linear.
X_AXES = {
    "sigma": ("Σdf (p50)", "sigma_df_p50", True),
    "lat": ("p99 latency (µs)", "lat_p99_us", False),
}

# Per-arm linestyle + marker (color is free to carry N in the overlay/knee views).
ARM_STYLE = {
    "t_max": ("-", "o"),
    "df_budget": ("--", "s"),
}
# Per-arm color for the facet view, where each panel is a single N so arm owns color.
ARM_COLOR = {"t_max": "tab:blue", "df_budget": "tab:orange"}


def parse_rows(streams):
    """Yield typed row dicts from one or more open CSV streams, skipping the
    repeated `arm,knob,...` header lines a `>>`-appended frontier file carries."""
    rows = []
    for stream in streams:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("arm,knob"):
                continue
            parts = line.split(",")
            if len(parts) != len(COLS):
                continue  # tolerate stray lines (e.g. an accidental log echo)
            rec = dict(zip(COLS, parts))
            rows.append(
                {
                    "arm": rec["arm"],
                    "knob": int(rec["knob"]),
                    "N": int(rec["N"]),
                    "k": int(rec["k"]),
                    "recall": float(rec["recall"]),
                    "sigma_df_p50": float(rec["sigma_df_p50"]),
                    "sigma_df_p99": float(rec["sigma_df_p99"]),
                    "lat_p50_us": float(rec["lat_p50_us"]),
                    "lat_p99_us": float(rec["lat_p99_us"]),
                }
            )
    return rows


def rows_at_k(rows, k):
    at_k = [r for r in rows if r["k"] == k]
    if not at_k:
        have = sorted({r["k"] for r in rows})
        raise SystemExit(f"no rows with k={k}; the file has k in {have}")
    return at_k


def n_colors(ns):
    """A stable color per N, sampled along viridis (low N dark, high N bright)."""
    cmap = plt.cm.viridis
    if len(ns) == 1:
        return {ns[0]: cmap(0.5)}
    return {n: cmap(i / (len(ns) - 1)) for i, n in enumerate(ns)}


def draw_facet(rows, k, x_choices, title, annotate):
    """rows = distinct N, cols = chosen work axes; the two arms overlaid per cell."""
    at_k = rows_at_k(rows, k)
    ns = sorted({r["N"] for r in at_k})
    fig, axes = plt.subplots(
        len(ns),
        len(x_choices),
        figsize=(6.2 * len(x_choices), 3.8 * len(ns)),
        squeeze=False,
    )
    for ri, n in enumerate(ns):
        for ci, xc in enumerate(x_choices):
            ax = axes[ri][ci]
            xlabel, xcol, logx = X_AXES[xc]
            by_arm = defaultdict(list)
            for r in at_k:
                if r["N"] == n:
                    by_arm[r["arm"]].append((r[xcol], r["recall"], r["knob"]))
            for arm, pts in by_arm.items():
                pts.sort()
                ls, marker = ARM_STYLE.get(arm, ("-", "x"))
                color = ARM_COLOR.get(arm, "gray")
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    ls=ls,
                    marker=marker,
                    color=color,
                    label=arm,
                    alpha=0.9,
                )
                if annotate:
                    for x, y, knob in pts:
                        ax.annotate(
                            str(knob),
                            (x, y),
                            textcoords="offset points",
                            xytext=(3, 4),
                            fontsize=7,
                            color=color,
                            alpha=0.8,
                        )
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"recall@{k}")
            ax.set_title(f"N={n:,}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(title="arm", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def draw_overlay(rows, k, x_choices, title):
    """Every N on shared axes — color = N, linestyle = arm — one panel per work axis."""
    at_k = rows_at_k(rows, k)
    ns = sorted({r["N"] for r in at_k})
    colors = n_colors(ns)
    arms = [a for a in ("t_max", "df_budget") if any(r["arm"] == a for r in at_k)]

    fig, axes = plt.subplots(
        1, len(x_choices), figsize=(6.6 * len(x_choices), 4.6), squeeze=False
    )
    for ci, xc in enumerate(x_choices):
        ax = axes[0][ci]
        xlabel, xcol, logx = X_AXES[xc]
        for n in ns:
            for arm in arms:
                pts = sorted(
                    (r[xcol], r["recall"])
                    for r in at_k
                    if r["N"] == n and r["arm"] == arm
                )
                if not pts:
                    continue
                ls, marker = ARM_STYLE[arm]
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    ls=ls,
                    marker=marker,
                    ms=4,
                    color=colors[n],
                    alpha=0.9,
                )
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"recall@{k}")
        ax.grid(True, which="both", alpha=0.25)

    # Two legends: color -> N, linestyle -> arm.
    n_handles = [Line2D([], [], color=colors[n], lw=2, label=f"N={n:,}") for n in ns]
    arm_handles = [
        Line2D([], [], color="0.3", ls=ARM_STYLE[a][0], marker=ARM_STYLE[a][1], label=a)
        for a in arms
    ]
    leg1 = axes[0][0].legend(handles=n_handles, title="N", fontsize=8, loc="lower right")
    axes[0][0].add_artist(leg1)
    axes[0][-1].legend(handles=arm_handles, title="arm", fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def find_knees(at_k, knee_frac):
    """Per N, the cheapest df_budget reaching `knee_frac` of that arm's max recall@k.
    Returns a sorted list of dicts: n, budget, recall, rmax, sigma, frac."""
    ns = sorted({r["N"] for r in at_k if r["arm"] == "df_budget"})
    out = []
    for n in ns:
        pts = sorted(
            (r["knob"], r["recall"], r["sigma_df_p50"])
            for r in at_k
            if r["arm"] == "df_budget" and r["N"] == n
        )
        if not pts:
            continue
        rmax = max(r for _, r, _ in pts)
        target = knee_frac * rmax
        knee = next(((b, r, s) for b, r, s in pts if r >= target), None)
        if knee is None:
            continue
        budget, rec, sigma = knee
        out.append(
            {
                "n": n,
                "budget": budget,
                "recall": rec,
                "rmax": rmax,
                "sigma": sigma,
                "frac": budget / n,
            }
        )
    return out


def draw_knee(rows, k, knee_frac, title, annotate):
    """[left] df_budget frontiers across N with the knee ringed; [right] optimal
    df_budget vs N with the budget*/N fraction and a log-log fit."""
    at_k = rows_at_k(rows, k)
    knees = find_knees(at_k, knee_frac)
    if not knees:
        raise SystemExit("no df_budget rows to analyze (knee mode needs the df_budget arm)")
    ns = [kn["n"] for kn in knees]
    colors = n_colors(ns)

    fig, (axf, axk) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # Left: df_budget recall@k vs Σdf p50, one curve per N, knee point ringed.
    for kn in knees:
        n = kn["n"]
        pts = sorted(
            (r["sigma_df_p50"], r["recall"])
            for r in at_k
            if r["arm"] == "df_budget" and r["N"] == n
        )
        axf.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "--s",
            ms=4,
            color=colors[n],
            label=f"N={n:,}",
            alpha=0.9,
        )
        axf.scatter(
            [kn["sigma"]],
            [kn["recall"]],
            s=140,
            facecolors="none",
            edgecolors=colors[n],
            linewidths=1.8,
            zorder=5,
        )
    axf.set_xscale("log")
    axf.set_xlabel("Σdf (p50)")
    axf.set_ylabel(f"recall@{k}")
    axf.set_title(f"df_budget frontier per N (○ = knee ≥ {knee_frac:g}·max recall)")
    axf.grid(True, which="both", alpha=0.25)
    axf.legend(title="N", fontsize=8)

    # Right: optimal df_budget vs N (log-log), the budget*/N fraction, and the fit.
    xs = np.array([kn["n"] for kn in knees], dtype=float)
    ys = np.array([kn["budget"] for kn in knees], dtype=float)
    for kn in knees:
        axk.scatter([kn["n"]], [kn["budget"]], s=60, color=colors[kn["n"]], zorder=5)
        if annotate:
            axk.annotate(
                f"{kn['frac']:.3f}·N",
                (kn["n"], kn["budget"]),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=8,
            )
    axk.set_xscale("log")
    axk.set_yscale("log")
    axk.set_xlabel("N (corpus size)")
    axk.set_ylabel(f"optimal df_budget (≥ {knee_frac:g}·max recall@{k})")

    mean_frac = float(np.mean([kn["frac"] for kn in knees]))
    summary = [
        f"knee = cheapest df_budget reaching {knee_frac:g}× its max recall@{k}",
    ]
    for kn in knees:
        summary.append(
            f"  N={kn['n']:>9,}  budget*={kn['budget']:>9,}  "
            f"frac={kn['frac']:.3f}  recall={kn['recall']:.4f} (max {kn['rmax']:.4f})"
        )
    note = f"mean budget*/N = {mean_frac:.3f}"
    if len(knees) >= 2:
        slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
        grid = np.array([xs.min(), xs.max()])
        axk.plot(
            grid,
            np.exp(intercept) * grid**slope,
            color="0.3",
            lw=1.3,
            label=f"fit: budget* ∝ N^{slope:.2f}",
        )
        axk.plot(
            grid,
            mean_frac * grid,
            color="crimson",
            ls=":",
            lw=1.3,
            label=f"{mean_frac:.3f}·N",
        )
        axk.legend(fontsize=8)
        note += f";  fit slope = {slope:.2f} (1.0 ⇒ budget* ∝ N)"
    axk.set_title("optimal df_budget vs N")
    axk.grid(True, which="both", alpha=0.25)

    print("\n".join(summary), file=sys.stderr)
    print(note, file=sys.stderr)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Plot the selsweep selection-cost frontier (recall@k vs work)."
    )
    ap.add_argument(
        "csv",
        nargs="*",
        default=["-"],
        help="selsweep CSV file(s); '-' or empty reads stdin",
    )
    ap.add_argument("--out", default="selsweep.png", help="output image (.png/.svg/.pdf)")
    ap.add_argument("--k", type=int, default=10, help="recall cutoff to plot (default 10)")
    ap.add_argument(
        "--mode",
        choices=["facet", "overlay", "knee"],
        default="facet",
        help="facet per N (default), overlay all N, or the df_budget-vs-N knee analysis",
    )
    ap.add_argument(
        "--x",
        choices=["sigma", "lat", "both"],
        default="both",
        help="work axis for facet/overlay: Σdf, p99 latency, or both (default both)",
    )
    ap.add_argument(
        "--knee-frac",
        type=float,
        default=0.98,
        help="knee mode: fraction of an arm's max recall that counts as 'full' (default 0.98)",
    )
    ap.add_argument("--title", default=None, help="figure title")
    ap.add_argument(
        "--no-annotate",
        action="store_true",
        help="don't label points (knob values, or budget*/N fractions)",
    )
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    paths = [p for p in args.csv if p] or ["-"]
    opened = []
    try:
        streams = []
        for p in paths:
            if p == "-":
                streams.append(sys.stdin)
            else:
                f = open(p, encoding="utf-8")
                opened.append(f)
                streams.append(f)
        rows = parse_rows(streams)
    finally:
        for f in opened:
            f.close()

    if not rows:
        raise SystemExit("no data rows parsed; is this a selsweep CSV?")

    annotate = not args.no_annotate
    x_choices = ["sigma", "lat"] if args.x == "both" else [args.x]
    if args.mode == "knee":
        title = args.title or f"df_budget scaling — optimal budget vs N (recall@{args.k})"
        fig = draw_knee(rows, args.k, args.knee_frac, title, annotate)
    elif args.mode == "overlay":
        title = args.title or f"Selection-cost frontier across N — recall@{args.k}"
        fig = draw_overlay(rows, args.k, x_choices, title)
    else:
        title = args.title or f"Selection-cost frontier — recall@{args.k} (t_max vs df_budget)"
        fig = draw_facet(rows, args.k, x_choices, title, annotate)

    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", file=sys.stderr)
    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
