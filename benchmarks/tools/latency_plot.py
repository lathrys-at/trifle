#!/usr/bin/env python3
"""Sweep trifle's `perf` benchmark across a corpus-size ladder and plot speed + recall.

This is the analysis half of the speed+quality eval; the measurement half is the `perf`
subcommand of `trifle-benchmarks` with `--format json` (which this script drives). For each
index size `N` it runs one `perf` invocation that measures trifle at several rerank
**efforts** (Low/Medium/High) from a single index build, alongside the in-process SQLite
baselines, on *labeled* queries — emitting per (engine, effort) the p50/p90/p99/max latency,
throughput, **honest recall@k**, AND the raw per-query samples.

Two query regimes (`--corpus`):
  - **msmarco** (default): real MS MARCO dev queries scored against qrels (no typos) — the
    paraphrase regime. Baselines: FTS5-word BM25, FTS5-trigram OR-bag, LIKE.
  - **geonames-all / geonames-cities**: entity name + `--edits` typos — the *real* typo
    regime, where recall measures typo tolerance. Baselines: FTS5-trigram OR-bag, LIKE.
FTS5 is scored via the OR-bag `MATCH` (not phrase), so the baseline recall is fair.

It renders two figures:

  1. **latency_grouped.png** — one panel per `N`; within a panel, one bar *group* per
     alternative (trifle Low/Medium/High + each baseline), each group a p50/p90/p99 triple.
     Effort/engine is the color; percentile is the position (left→right) and the alpha. The
     **recall@k** and the **\\*max** latency are annotated above each group.
  2. **throughput_vs_N.png** — throughput (q/s) vs `N`, one line per alternative, with
     recall@k annotated above each point.

Plus a supplementary **latency_vs_N.png** (p50/p99 vs N, the latency-scaling story).

The raw JSON for every `N` is persisted (under `<out>/raw/`), together with a combined
`raw.json` and a tidy `summary.csv`, so the plots can be regenerated — or the analysis
changed entirely — WITHOUT re-running the benchmark (`--reuse-raw`).

Usage:
    # msmarco sweep (real dev queries; downloads ~1 GiB collection on first use) + plots
    python3 benchmarks/tools/latency_plot.py --queries 100 --seed 42

    # geonames typo regime (full gazetteer; downloads ~400 MB on first use)
    python3 benchmarks/tools/latency_plot.py --corpus geonames-all --edits 2

    # re-plot from previously captured raw data, no benchmark run
    python3 benchmarks/tools/latency_plot.py --reuse-raw
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

# The requested corpus-size ladder (×5 steps), and the default query sample count.
DEFAULT_DOCS = "1000,5000,25000,125000,625000,3125000"

# Series identity → (display label, color). A series key is `engine/effort` for trifle and
# the bare engine name for the baselines (which have no effort knob). trifle's three efforts
# get a green family (light→dark = Low→High); the baselines get distinct hues.
SERIES = {
    "trifle/low": ("trifle (Low)", "#74c476"),
    "trifle/medium": ("trifle (Medium)", "#31a354"),
    "trifle/high": ("trifle (High)", "#006d2c"),
    "fts5-word-bm25": ("FTS5 word (BM25)", "#756bb1"),
    "fts5-trigram-bm25": ("FTS5 trigram (BM25)", "#e6550d"),
    "like-scan": ("LIKE scan", "#3182bd"),
}
# A stable plotting order (left→right within a panel, top→bottom in legends).
SERIES_ORDER = list(SERIES.keys())

# Percentile bars within a group: (field-stem, alpha). Heights are monotone p50≤p90≤p99.
PCTS = [("p50", 1.0), ("p90", 0.68), ("p99", 0.42)]


def series_key(rec):
    """The series key for a JSON record: `engine/effort`, or bare `engine` for baselines."""
    eff = rec.get("effort")
    return f"{rec['engine']}/{eff}" if eff else rec["engine"]


def label_of(key):
    return SERIES.get(key, (key, "#888888"))[0]


def color_of(key):
    return SERIES.get(key, (key, "#888888"))[1]


def fmt_latency(ns):
    """Compact ns → µs/ms string, matching the Rust harness's `fmt_dur`."""
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.0f}µs"
    return f"{ns / 1_000_000:.2f}ms"


# ---- measurement: drive the perf subcommand ---------------------------------------------
def run_one(corpus, n, queries, k, seed, efforts, edits, warmup, max_like_n, max_tri_n):
    """Run one `perf --format json` invocation for index size `n`; return its parsed JSON
    object. `perf` measures latency + throughput + honest recall@k on labeled queries (real
    MS MARCO dev queries for msmarco; entity/snippet + `edits` typos otherwise), scoring FTS5
    via the fair OR-bag MATCH.

    The two genuinely-impractical-at-scale baselines are dropped above their thresholds
    (logged, never silent): `like-scan` (an O(N) full scan/query) above `max_like_n`, and
    `fts5-trigram-bm25` (the OR-bag MATCH explodes to ~seconds/query at millions of docs)
    above `max_tri_n`. trifle and `fts5-word-bm25` run at every N."""
    cmd = [
        "cargo", "run", "-q", "-p", "trifle-benchmarks", "--release", "--",
        "perf", "--corpus", corpus, "--docs", str(n), "--queries", str(queries),
        "--k", str(k), "--seed", str(seed), "--effort-sweep", efforts,
        "--edits", str(edits), "--warmup", str(warmup), "--format", "json",
    ]
    if n > max_like_n:
        cmd += ["--filter", "like-scan"]
        print(f"  N={n:,}: like-scan filtered (N > --max-like-n {max_like_n:,}; O(N) scan)",
              file=sys.stderr)
    if n > max_tri_n:
        cmd += ["--filter", "fts5-trigram-bm25"]
        print(f"  N={n:,}: fts5-trigram-bm25 filtered (N > --max-tri-n {max_tri_n:,}; "
              f"OR-bag MATCH ~seconds/query)", file=sys.stderr)
    print(f"  perf N={n:,} (corpus={corpus}, efforts={efforts}) ...", file=sys.stderr, flush=True)
    cp = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr[-4000:])
        raise SystemExit(f"latency failed for N={n} (corpus={corpus})")
    # stdout is exactly one JSON object (the `#` human header is suppressed in json mode).
    out = cp.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Be forgiving if anything leaked onto stdout: take the last brace-delimited line.
        line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("{")), None)
        if line is None:
            sys.stderr.write(cp.stdout[-2000:])
            raise SystemExit(f"could not parse latency JSON for N={n}")
        return json.loads(line)


def run_sweep(args, out):
    """Drive `latency` across the N ladder, persisting every raw object AS IT COMPLETES (so a
    later N that exhausts memory can't lose the earlier results) and updating the combined
    `raw.json` after each. A failed N is logged and skipped, never silently dropped."""
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs = [int(x) for x in args.docs.split(",")]
    raw, failed = {}, []
    for n in docs:
        try:
            obj = run_one(args.corpus, n, args.queries, args.k, args.seed,
                          args.efforts, args.edits, args.warmup, args.max_like_n, args.max_tri_n)
        except SystemExit as e:
            print(f"  !! N={n:,} FAILED ({e}); skipping and continuing", file=sys.stderr)
            failed.append(n)
            continue
        (raw_dir / f"perf-{args.corpus}-N{n}.json").write_text(json.dumps(obj, indent=2))
        raw[n] = obj
        # Rewrite the combined file after every N so a crash mid-sweep keeps what we have.
        (out / "raw.json").write_text(
            json.dumps({str(m): raw[m] for m in docs if m in raw}, indent=2))
    if failed:
        print(f"NOTE: {len(failed)} size(s) skipped after failure: "
              f"{', '.join(f'{m:,}' for m in failed)}", file=sys.stderr)
    if not raw:
        raise SystemExit("every N failed — nothing to plot")
    return raw


def load_raw(out):
    """Reload a prior sweep from `<out>/raw.json` (the `--reuse-raw` path)."""
    path = out / "raw.json"
    if not path.exists():
        raise SystemExit(f"--reuse-raw: no {path} (run the sweep once first)")
    blob = json.loads(path.read_text())
    return {int(n): obj for n, obj in blob.items()}


# ---- shape the raw objects into {n: {series_key: metrics}} -------------------------------
def tidy(raw):
    """{n: {series_key: {p50_ns,p90_ns,p99_ns,max_ns,mean_ns,recall,throughput_qps,n}}}."""
    data = {}
    for n, obj in raw.items():
        row = {}
        for rec in obj["records"]:
            ln = rec["latency_ns"]
            row[series_key(rec)] = {
                "p50_ns": ln["p50"], "p90_ns": ln["p90"], "p99_ns": ln["p99"],
                "max_ns": ln["max"], "mean_ns": ln["mean"], "n": ln["n"],
                "recall": rec["recall_at_k"], "recall_k": rec["recall_k"],
                "throughput_qps": rec["throughput_qps"],
            }
        data[n] = row
    return data


def present_series(row):
    """Series keys present in a row, in the canonical order (unknowns appended)."""
    known = [k for k in SERIES_ORDER if k in row]
    extra = [k for k in row if k not in SERIES]
    return known + extra


def write_csv(data, path):
    ns = sorted(data)
    cols = ["docs", "engine", "effort", "series", "p50_ns", "p90_ns", "p99_ns",
            "max_ns", "mean_ns", "throughput_qps", "recall_at_k", "recall_k", "n_samples"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n in ns:
            for key in present_series(data[n]):
                m = data[n][key]
                engine, _, effort = key.partition("/")
                w.writerow([n, engine, effort or "", key, m["p50_ns"], m["p90_ns"],
                            m["p99_ns"], m["max_ns"], f"{m['mean_ns']:.1f}",
                            f"{m['throughput_qps']:.2f}", f"{m['recall']:.4f}",
                            m["recall_k"], m["n"]])


# ---- plots ------------------------------------------------------------------------------
def plot_latency_grouped(data, ns, corpus, k, path):
    """One panel per N. Per panel: a bar group per alternative, p50/p90/p99 within. Color =
    series (effort/engine); percentile = position + alpha. recall@k and *max above each
    group."""
    cols = min(3, len(ns))
    rows = (len(ns) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.6 * rows), squeeze=False)
    group_w = 0.82
    for ax, n in zip(axes.flat, ns):
        keys = present_series(data[n])
        xs = np.arange(len(keys))
        bar_w = group_w / len(PCTS)
        lo, hi = float("inf"), 0.0
        for gi, key in enumerate(keys):
            m = data[n][key]
            color = color_of(key)
            for bi, (p, alpha) in enumerate(PCTS):
                val_us = m[f"{p}_ns"] / 1_000.0
                x = xs[gi] - group_w / 2 + bar_w * (bi + 0.5)
                ax.bar(x, val_us, width=bar_w * 0.92, color=color, alpha=alpha,
                       edgecolor="black", linewidth=0.3, zorder=3)
                lo, hi = min(lo, max(val_us, 1e-3)), max(hi, val_us)
            # recall@k and *max annotated above the group (the p99 bar is the tallest).
            top_us = m["p99_ns"] / 1_000.0
            ax.annotate(f"r {m['recall']:.2f}\n*{fmt_latency(m['max_ns'])}",
                        (xs[gi], top_us), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=7, linespacing=1.05)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(lo / 1.8, 1e-2), top=hi * 4.0)  # headroom for annotations
        ax.set_xticks(xs)
        ax.set_xticklabels([label_of(key) for key in keys], rotation=18, ha="right", fontsize=8)
        ax.set_title(f"N = {n:,} docs", fontsize=11)
        ax.set_ylabel("latency (µs, log)")
        ax.grid(axis="y", alpha=0.3, which="both", zorder=0)
    for ax in axes.flat[len(ns):]:
        ax.axis("off")

    # Two legends: percentile (alpha) and the recall/max annotation key.
    pct_handles = [mpatches.Patch(facecolor="#555555", alpha=a, edgecolor="black",
                                  linewidth=0.3, label=p) for p, a in PCTS]
    fig.legend(handles=pct_handles, title="percentile (left→right per group)",
               loc="upper right", ncol=3, fontsize=8, title_fontsize=8,
               bbox_to_anchor=(0.995, 0.998))
    fig.suptitle(f"Search latency by corpus size — {corpus}\n"
                 f"bar = p50/p90/p99 · annotation: r = recall@{k}, *= max latency",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_throughput(data, ns, corpus, k, path):
    """Throughput (q/s) vs N, one line per alternative; recall@k annotated above each point."""
    fig, ax = plt.subplots(figsize=(11, 7))
    all_keys = [key for key in SERIES_ORDER if any(key in data[n] for n in ns)]
    # The three trifle effort lines run close together; stagger their recall labels
    # vertically (by position among same-engine series) so they don't overprint.
    for si, key in enumerate(all_keys):
        engine = key.split("/")[0]
        same_engine = [j for j, kk in enumerate(all_keys) if kk.split("/")[0] == engine]
        rank = same_engine.index(si)  # 0,1,2,… within this engine's series
        dy = 8 + rank * 9
        xs, ys, recs = [], [], []
        for n in ns:
            if key in data[n]:
                xs.append(n)
                ys.append(data[n][key]["throughput_qps"])
                recs.append(data[n][key]["recall"])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", ms=5, color=color_of(key), label=label_of(key), zorder=3)
        for x, y, r in zip(xs, ys, recs):
            ax.annotate(f"{r:.2f}", (x, y), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6.5, color=color_of(key), zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size N (docs, log)")
    ax.set_ylabel("throughput (queries/s, log)")
    ax.set_title(f"Throughput vs corpus size — {corpus}\n"
                 f"(recall@{k} annotated above each point)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="alternative")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_latency_vs_n(data, ns, corpus, path):
    """Supplementary: p50 and p99 latency vs N (the flat-latency-as-N-grows story)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    all_keys = [key for key in SERIES_ORDER if any(key in data[n] for n in ns)]
    for ax, p in zip(axes.flat, ["p50", "p99"]):
        for key in all_keys:
            xs, ys = [], []
            for n in ns:
                if key in data[n]:
                    xs.append(n)
                    ys.append(data[n][key][f"{p}_ns"] / 1_000.0)
            if xs:
                ax.plot(xs, ys, marker="o", ms=5, color=color_of(key), label=label_of(key))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("corpus size N (docs, log)")
        ax.set_ylabel(f"{p} latency (µs, log)")
        ax.set_title(f"{p} latency vs N")
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=8, title="alternative")
    fig.suptitle(f"Latency scaling with corpus size — {corpus}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---- main -------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="msmarco",
                    help="query regime: msmarco (real dev queries) | geonames-all | "
                         "geonames-cities | synthetic (entity/snippet + typos) [msmarco]")
    ap.add_argument("--docs", default=DEFAULT_DOCS, help="comma-separated index sizes N")
    ap.add_argument("--queries", type=int, default=100, help="query samples per N [100]")
    ap.add_argument("--k", type=int, default=10, help="top-k cutoff (and recall@k) [10]")
    ap.add_argument("--seed", type=int, default=42, help="master seed [42]")
    ap.add_argument("--efforts", default="low,medium,high",
                    help="trifle efforts to sweep [low,medium,high]")
    ap.add_argument("--edits", type=int, default=2,
                    help="typos/query for the geonames/synthetic regimes (ignored for msmarco) [2]")
    ap.add_argument("--warmup", type=int, default=100, help="untimed warmup queries [100]")
    ap.add_argument("--max-like-n", type=int, default=625000,
                    help="drop like-scan above this N (O(N) scan impractical) [625000]")
    ap.add_argument("--max-tri-n", type=int, default=625000,
                    help="drop fts5-trigram-bm25 above this N (OR-bag MATCH ~seconds/query) [625000]")
    ap.add_argument("--out", default=None, help="output dir [benchmarks/reports/perf-<corpus>]")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="reuse <out>/raw.json (skip the benchmark, just re-plot)")
    args = ap.parse_args()

    out = Path(args.out or REPO / "benchmarks" / "reports" / f"perf-{args.corpus}")
    out.mkdir(parents=True, exist_ok=True)

    if args.reuse_raw:
        print(f"reusing raw data in {out}/raw.json", file=sys.stderr)
        raw = load_raw(out)
    else:
        print(f"sweeping corpus={args.corpus} queries={args.queries} seed={args.seed} "
              f"N={args.docs} efforts={args.efforts}", file=sys.stderr)
        raw = run_sweep(args, out)

    data = tidy(raw)
    ns = sorted(data)
    print(f"loaded N={ns}", file=sys.stderr)

    write_csv(data, out / "summary.csv")
    plot_latency_grouped(data, ns, args.corpus, args.k, out / "latency_grouped.png")
    plot_throughput(data, ns, args.corpus, args.k, out / "throughput_vs_N.png")
    plot_latency_vs_n(data, ns, args.corpus, out / "latency_vs_N.png")

    print(f"\nwrote: {out}/summary.csv, raw.json, raw/, "
          f"latency_grouped.png, throughput_vs_N.png, latency_vs_N.png")


if __name__ == "__main__":
    main()
