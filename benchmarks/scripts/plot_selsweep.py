#!/usr/bin/env python3
"""Plot the selsweep selection-cost frontier.

Consumes the CSV that `trifle-bench selsweep` writes (columns
`arm,knob,N,k,recall,sigma_df_p50,sigma_df_p99,lat_p50_us,lat_p99_us`) and draws
recall@k against the work it cost — Σdf and/or p99 latency — with the two
selection arms (`t_max`, `df_budget`) overlaid so you can read which knob buys
more recall per unit work. The better arm is the curve that sits up-and-left.

The scaling-ladder workflow appends several N runs into one file:

    for n in 1000 5000 25000 125000 625000; do
      trifle-bench selsweep --corpus geonames-all --docs "$n" >> frontier.csv
    done

so the file carries one header line per run; those repeats are skipped on read.
Each distinct N becomes its own row of panels.

    python3 plot_selsweep.py frontier.csv                 # -> selsweep.png
    python3 plot_selsweep.py frontier.csv --k 10 --x both
    cat frontier.csv | python3 plot_selsweep.py - --out frontier.svg --show

Needs matplotlib (the only non-stdlib dependency).
"""

import argparse
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless by default; --show flips to an interactive backend
import matplotlib.pyplot as plt  # noqa: E402

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

# A stable color + marker per arm (cells are already faceted by N, so arm is the
# only series within a cell).
ARM_STYLE = {
    "t_max": ("tab:blue", "o"),
    "df_budget": ("tab:orange", "s"),
}


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


def draw(rows, k, x_choices, title, annotate):
    """Build the figure: rows = distinct N, cols = chosen work axes."""
    at_k = [r for r in rows if r["k"] == k]
    if not at_k:
        have = sorted({r["k"] for r in rows})
        raise SystemExit(f"no rows with k={k}; the file has k in {have}")

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
            # Group this cell's points by arm, sorted along the work axis so the
            # polyline reads as a frontier.
            by_arm = defaultdict(list)
            for r in at_k:
                if r["N"] == n:
                    by_arm[r["arm"]].append((r[xcol], r["recall"], r["knob"]))

            for arm, pts in by_arm.items():
                pts.sort()
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                color, marker = ARM_STYLE.get(arm, ("gray", "x"))
                ax.plot(xs, ys, marker=marker, color=color, label=arm, alpha=0.9)
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
        "--x",
        choices=["sigma", "lat", "both"],
        default="both",
        help="work axis: Σdf, p99 latency, or both side by side (default both)",
    )
    ap.add_argument("--title", default=None, help="figure title")
    ap.add_argument(
        "--no-annotate",
        action="store_true",
        help="don't label each point with its knob value",
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

    title = args.title or f"Selection-cost frontier — recall@{args.k} (t_max vs df_budget)"
    x_choices = ["sigma", "lat"] if args.x == "both" else [args.x]
    fig = draw(rows, args.k, x_choices, title, annotate=not args.no_annotate)

    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", file=sys.stderr)
    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
