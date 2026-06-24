#!/usr/bin/env python3
"""Find the selection-cap (`t_max`) knee and test whether it scales with query length.

`t_max` is NOT a rerank-pool-style effort knob. Pool is a monotone, saturating budget you
spend (`p* ≈ c·√(k·N)`); `t_max` is a *knee you find*: raising it lifts the recall ceiling
(more kept trigrams catch docs whose rare anchors were beyond the old cap), but past the
knee the rarest-first pruner is adding *common* (high-DF) trigrams that cost latency
superlinearly and inject overlap noise that can push the true doc *down* (recall@small-k
falls — the "hump"). So `t_max` past its knee is a mistake, not an effort level.

Hypothesis under test (confirm or break — do not assume): the knee scales with QUERY LENGTH
(trigram count), ~independent of N and k — the same variable as the typo floor F, whose
ceiling t_max is. Predicted `t_max ≈ min(T_cap, a·tokens + b)`.

Method (per the handoff): drive `trifle-bench tmaxsweep` (recall at a *generous* pool, so
selection is the only bottleneck), bucket queries by length, and for each (N, length-bucket)
measure the recall CEILING and latency vs t_max separately, locate the knee, check whether it
moves with N/k, and track recall@small-k to expose the hump. Report LINEAR-space prediction
error (observed/predicted), not log-log R² — a 1.4× miss on a selection bound changes which
docs are reachable.

    python3 benchmarks/tools/tmax_knee.py --corpus msmarco --docs 20000,100000 --queries 500
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Recall@k cutoffs to evaluate from each query's recovered rank. CEIL_K is the "ceiling"
# (selection-reachable) cutoff; SMALL_K exposes the precision hump.
KS = [1, 5, 10, 50]
SMALL_K = 1
CEIL_K = 50
# Query-length (trigram-count) buckets. Edges are inclusive-low, exclusive-high.
LEN_EDGES = [4, 7, 10, 14, 19, 26, 36, 51, 9999]
KNEE_FRAC = 0.98   # knee = smallest t_max reaching this fraction of the bucket's max ceiling
MIN_BUCKET_Q = 15  # drop length buckets with fewer than this many distinct queries


# ---- driving the sweep ------------------------------------------------------
def run_sweep(corpus, n, queries, seed, edits, max_tmax, pool):
    cmd = [
        "cargo", "run", "-q", "-p", "trifle-benchmarks", "--release", "--",
        "tmaxsweep", "--corpus", corpus, "--docs", str(n), "--queries", str(queries),
        "--seed", str(seed), "--edits", str(edits), "--max-tmax", str(max_tmax),
        "--pool", str(pool),
    ]
    print(f"  [tmaxsweep] N={n} …", file=sys.stderr, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"tmaxsweep failed (N={n}):\n{r.stderr[-2000:]}")
    return pd.read_csv(io.StringIO(r.stdout))


def collect(args, out):
    csv = out / "tmax_raw.csv"
    if args.reuse_csv and csv.exists():
        print(f"reusing {csv}", file=sys.stderr)
        return pd.read_csv(csv)
    frames = []
    for n in args.docs:
        df = run_sweep(args.corpus, n, args.queries, args.seed, args.edits,
                       args.max_tmax, args.pool)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(csv, index=False)
    return df


# ---- analysis ---------------------------------------------------------------
def bucket_label(lo, hi):
    return f"{lo}+" if hi >= 9999 else f"{lo}-{hi - 1}"


def add_buckets(df):
    edges = LEN_EDGES
    idx = np.digitize(df.q_trigrams, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    df = df.copy()
    df["blo"] = [edges[i] for i in idx]
    df["bhi"] = [edges[i + 1] for i in idx]
    df["bucket"] = [bucket_label(edges[i], edges[i + 1]) for i in idx]
    return df


def recall_curve(g, k):
    """recall@k vs t_max for one (N, bucket) group: mean over its queries of 0<rank<=k."""
    hit = (g["rank"] > 0) & (g["rank"] <= k)
    r = g.assign(hit=hit).groupby("t_max")["hit"].mean()
    return r.index.values.astype(float), r.values.astype(float)


def knee(ts, rec, frac=KNEE_FRAC):
    """Smallest t_max reaching `frac` of the curve's max recall (the ceiling plateau)."""
    if len(rec) == 0 or np.max(rec) <= 0:
        return float("nan")
    tgt = frac * np.max(rec)
    hit = np.where(rec >= tgt)[0]
    return float(ts[hit[0]]) if len(hit) else float(ts[-1])


def analyse(df, out):
    df = add_buckets(df)
    Ns = sorted(df.N.unique())
    # Keep buckets with enough distinct queries (counted at a single t_max).
    t0 = df.t_max.min()
    bq = df[df.t_max == t0].groupby("bucket").size()
    buckets = [b for b in bq.index if bq[b] >= MIN_BUCKET_Q]
    # Order buckets by their low edge.
    buckets = sorted(buckets, key=lambda b: int(b.rstrip("+").split("-")[0]))

    rows = []          # per (N, bucket): knee at CEIL_K, mean length, hump info
    for N in Ns:
        for b in buckets:
            g = df[(df.N == N) & (df.bucket == b)]
            if g.empty:
                continue
            ts, rceil = recall_curve(g, CEIL_K)
            kn = knee(ts, rceil)
            _, rsmall = recall_curve(g, SMALL_K)
            # Hump: does recall@small-k fall after its peak (past the knee)?
            ipk = int(np.argmax(rsmall))
            peak = rsmall[ipk]
            tail_min = float(np.min(rsmall[ipk:])) if ipk < len(rsmall) else peak
            hump_drop = peak - tail_min          # absolute recall@1 lost past the peak
            rows.append(dict(
                N=int(N), bucket=b, mean_len=float(g.q_trigrams.mean()),
                n_q=int((g.t_max == t0).sum()),
                knee=kn, ceil_max=float(np.max(rceil)),
                r1_peak=float(peak), r1_peak_t=float(ts[ipk]), hump_drop=float(hump_drop),
            ))
    res = pd.DataFrame(rows)

    # --- knee vs N stability (the core hypothesis test) ---
    stab = []
    for b in buckets:
        sub = res[res.bucket == b]
        if sub.empty:
            continue
        knees = sub.knee.values
        stab.append(dict(
            bucket=b, mean_len=float(sub.mean_len.mean()),
            knee_med=float(np.median(knees)),
            knee_min=float(np.min(knees)), knee_max=float(np.max(knees)),
            knee_spread=float(np.max(knees) - np.min(knees)),
        ))
    stab = pd.DataFrame(stab)

    # --- fit knee ≈ min(T_cap, a·tokens + b), linear-space error ---
    fit = {}
    if len(stab) >= 3:
        x = stab.mean_len.values
        y = stab.knee_med.values
        # Cap = the saturated knee (median of the top third by length); fit the rising part.
        cap = float(np.median(np.sort(y)[-max(1, len(y) // 3):]))
        rising = y < cap - 1e-9
        if rising.sum() >= 2:
            a, b = np.polyfit(x[rising], y[rising], 1)
        else:
            a, b = np.polyfit(x, y, 1)
        pred = np.minimum(cap, a * x + b)
        ratio = y / np.maximum(pred, 1e-9)        # linear-space observed/predicted
        fit = dict(a=float(a), b=float(b), T_cap=float(cap),
                   ratio_min=float(np.min(ratio)), ratio_max=float(np.max(ratio)),
                   ratio_med=float(np.median(ratio)),
                   per_bucket=[dict(bucket=bk, mean_len=float(xl), knee=float(yl),
                                    pred=float(pp), ratio=float(rr))
                               for bk, xl, yl, pp, rr in
                               zip(stab.bucket, x, y, pred, ratio)])

    summary = dict(corpus=df.attrs.get("corpus", ""), Ns=[int(n) for n in Ns],
                   ceil_k=CEIL_K, small_k=SMALL_K, knee_frac=KNEE_FRAC,
                   buckets=buckets, fit=fit,
                   stability=stab.to_dict("records"),
                   per_N_bucket=res.to_dict("records"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return df, res, stab, fit, buckets, Ns


# ---- plots ------------------------------------------------------------------
def facet(df, buckets, Ns, k, ylabel, fname, out, title):
    ncol = min(4, len(buckets)) or 1
    nrow = (len(buckets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for i, b in enumerate(buckets):
        ax = axes[i // ncol][i % ncol]
        for N in Ns:
            g = df[(df.N == N) & (df.bucket == b)]
            if g.empty:
                continue
            ts, rec = recall_curve(g, k) if k else latency_curve(g)
            ax.plot(ts, rec, marker=".", label=f"N={N:,}")
        ax.set_title(f"len {b}")
        ax.set_xlabel("t_max")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(buckets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=110)
    plt.close(fig)


def latency_curve(g):
    r = g.groupby("t_max")["latency_us"].median()
    return r.index.values.astype(float), r.values.astype(float)


def plot_knee_fit(stab, fit, out):
    if stab.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(stab.mean_len, stab.knee_med,
                yerr=[stab.knee_med - stab.knee_min, stab.knee_max - stab.knee_med],
                fmt="o", capsize=3, label="knee (median over N; bar = N spread)")
    if fit:
        xs = np.linspace(stab.mean_len.min(), stab.mean_len.max(), 100)
        ax.plot(xs, np.minimum(fit["T_cap"], fit["a"] * xs + fit["b"]),
                "-", label=f"min({fit['T_cap']:.0f}, {fit['a']:.2f}·len + {fit['b']:.1f})")
    ax.set_xlabel("query length (mean trigram count in bucket)")
    ax.set_ylabel("knee t_max*")
    ax.set_title("t_max knee vs query length")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "knee_vs_length.png", dpi=120)
    plt.close(fig)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="t_max selection-knee analysis")
    ap.add_argument("--corpus", required=True,
                    help="synthetic | msmarco | geonames-cities | geonames-all")
    ap.add_argument("--docs", default="20000,100000",
                    help="comma-separated index sizes N")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edits", type=int, default=2)
    ap.add_argument("--max-tmax", type=int, default=64)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-csv", action="store_true")
    args = ap.parse_args()
    args.docs = [int(x) for x in args.docs.split(",")]
    out = Path(args.out or f"tmax-{args.corpus}")
    out.mkdir(parents=True, exist_ok=True)

    df = collect(args, out)
    df.attrs["corpus"] = args.corpus
    df, res, stab, fit, buckets, Ns = analyse(df, out)

    facet(df, buckets, Ns, CEIL_K, f"recall@{CEIL_K} (ceiling)", "ceiling_vs_tmax.png", out,
          "Recall ceiling vs t_max (selection-reachable, generous pool)")
    facet(df, buckets, Ns, SMALL_K, f"recall@{SMALL_K} (precision / hump)",
          "hump_vs_tmax.png", out, "Recall@1 vs t_max — does it fall past the knee? (the hump)")
    facet(df, buckets, Ns, None, "median latency (µs)", "latency_vs_tmax.png", out,
          "Latency vs t_max (cost side)")
    plot_knee_fit(stab, fit, out)

    # printed report
    print(f"\n=== t_max knee — corpus={args.corpus}  N={Ns} ===")
    print("\nKnee per length bucket (does it move with N?):")
    print(stab.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nHump check (recall@1 drop past its peak, per N×bucket):")
    h = res[["N", "bucket", "mean_len", "ceil_max", "r1_peak", "r1_peak_t", "hump_drop"]]
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if fit:
        print(f"\nFit: knee ≈ min({fit['T_cap']:.0f}, {fit['a']:.3f}·len + {fit['b']:.2f})")
        print(f"  linear-space observed/predicted ratio: "
              f"min={fit['ratio_min']:.2f} med={fit['ratio_med']:.2f} max={fit['ratio_max']:.2f}")
        print("  (ratio near 1.0 across buckets ⇒ knee is a per-query-length bound)")
    print(f"\nWrote plots + summary.json to {out}/")


if __name__ == "__main__":
    main()
