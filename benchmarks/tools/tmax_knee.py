#!/usr/bin/env python3
"""Characterize the selection cap `t_max`: where is its recall optimum, and what sets it?

`t_max` is NOT a rerank-pool-style effort knob. Pool is a monotone, saturating budget you
spend (`p* ≈ c·√(k·N)`); `t_max` is a knob with an interior OPTIMUM. Raising it lifts the
recall ceiling (more kept trigrams catch docs whose rare anchors were beyond the old cap),
but past the optimum the rarest-first pruner is adding *common* (high-DF) trigrams that cost
latency superlinearly and inject overlap noise that pushes the true doc DOWN — so recall
falls (the "hump"). `t_max` past its optimum is damage, not effort.

Because the recall-vs-t_max curve PEAKS rather than plateaus, this tool reports peak-based
statistics, not a saturation knee:
  - peak_t   : argmax_t recall@k(t)  — the recall-optimal t_max.
  - near_t   : the CHEAPEST t_max within `NEAR_EPS` recall of the peak (the practical
               operating point — don't pay latency for the last <1% recall).
  - hump     : peak_recall − recall@(max t_max)  (how much you lose by over-selecting).
All are computed per (N, query-length bucket, k), so we can see whether the optimum moves
with N (the handoff's N-independence hypothesis), with k, and with query length.

Method (per the handoff): `tmaxsweep` measures recall at a *generous* pool so selection is
the only bottleneck — and the generous pool is scaled with N (the `√(kN)` pool law), capped
to bound rerank cost. Report LINEAR-space prediction error (observed/predicted), not log-log
R²: a 1.4× miss on a selection bound changes which docs are reachable.

    python3 benchmarks/tools/tmax_knee.py --corpus msmarco \\
        --docs 1000,5000,10000,50000,100000,500000,1000000 --queries 500
"""

import argparse
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 5, 10, 50]          # recall@k cutoffs evaluated from each query's recovered rank
SMALL_K = 1                  # precision end (the hump shows here first)
CEIL_K = 50                  # large-k ceiling end (selection-reachable)
LEN_EDGES = [4, 7, 10, 14, 19, 26, 36, 51, 9999]   # query trigram-count buckets
NEAR_EPS = 0.01              # near-optimal = within this absolute recall of the peak
MIN_BUCKET_Q = 12            # drop length buckets with fewer than this many queries


# ---- generous pool that scales with N (so pool is not the bottleneck) -------
def generous_pool(n, cap):
    # The relevant doc's overlap-rank grows ~√(k·N) (the pool law). Use a multiple of that
    # at k = max KS so the rerank pool comfortably contains it, capped to bound rerank cost.
    p = round(2.0 * math.sqrt(max(KS) * n))
    return int(min(n, min(cap, max(1500, p))))


# ---- driving the sweep ------------------------------------------------------
def run_sweep(corpus, n, queries, seed, edits, max_tmax, pool):
    cmd = [
        "cargo", "run", "-q", "-p", "trifle-benchmarks", "--release", "--",
        "tmaxsweep", "--corpus", corpus, "--docs", str(n), "--queries", str(queries),
        "--seed", str(seed), "--edits", str(edits), "--max-tmax", str(max_tmax),
        "--pool", str(pool),
    ]
    print(f"  [tmaxsweep] N={n} pool={pool} …", file=sys.stderr, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"tmaxsweep failed (N={n}):\n{r.stderr[-2000:]}")
    df = pd.read_csv(io.StringIO(r.stdout))
    df["pool"] = pool
    return df


def collect(args, out):
    csv = out / "tmax_raw.csv"
    if args.reuse_csv and csv.exists():
        print(f"reusing {csv}", file=sys.stderr)
        return pd.read_csv(csv)
    frames = []
    for n in args.docs:
        pool = args.pool if args.pool > 0 else generous_pool(n, args.pool_cap)
        frames.append(run_sweep(args.corpus, n, args.queries, args.seed, args.edits,
                                args.max_tmax, pool))
        # Checkpoint after each N, so a failure at a large N keeps the smaller-N data.
        pd.concat(frames, ignore_index=True).to_csv(csv, index=False)
    return pd.concat(frames, ignore_index=True)


# ---- analysis ---------------------------------------------------------------
def bucket_label(lo, hi):
    return f"{lo}+" if hi >= 9999 else f"{lo}-{hi - 1}"


def add_buckets(df):
    e = LEN_EDGES
    idx = np.clip(np.digitize(df.q_trigrams, e, right=False) - 1, 0, len(e) - 2)
    df = df.copy()
    df["bucket"] = [bucket_label(e[i], e[i + 1]) for i in idx]
    df["blo"] = [e[i] for i in idx]
    return df


def recall_curve(g, k):
    hit = (g["rank"] > 0) & (g["rank"] <= k)
    r = g.assign(hit=hit).groupby("t_max")["hit"].mean().sort_index()
    return r.index.values.astype(float), r.values.astype(float)


def latency_curve(g):
    r = g.groupby("t_max")["latency_us"].median().sort_index()
    return r.index.values.astype(float), r.values.astype(float)


def optima(ts, rec):
    """peak recall, peak t, cheapest-near-optimal t, end recall (at max t_max)."""
    if len(rec) == 0 or np.max(rec) <= 0:
        return dict(peak=float("nan"), peak_t=float("nan"),
                    near_t=float("nan"), end=float("nan"))
    ipk = int(np.argmax(rec))
    peak = float(rec[ipk])
    near_i = np.where(rec >= peak - NEAR_EPS)[0]
    near_t = float(ts[near_i[0]]) if len(near_i) else float(ts[ipk])
    return dict(peak=peak, peak_t=float(ts[ipk]), near_t=near_t, end=float(rec[-1]))


def analyse(df, out):
    df = add_buckets(df)
    Ns = sorted(df.N.unique())
    t0 = df.t_max.min()
    bq = df[df.t_max == t0].groupby("bucket").size()
    buckets = sorted([b for b in bq.index if bq[b] >= MIN_BUCKET_Q],
                     key=lambda b: int(b.rstrip("+").split("-")[0]))

    rows = []
    for N in Ns:
        for b in buckets:
            g = df[(df.N == N) & (df.bucket == b)]
            if g.empty:
                continue
            tsl, lat = latency_curve(g)
            lat_floor = float(lat[0]) if len(lat) else float("nan")
            rec = {}
            for k in KS:
                ts, r = recall_curve(g, k)
                o = optima(ts, r)
                rec[k] = o
                # latency at the near-optimal t for this k
                o["lat_near"] = float(lat[np.searchsorted(tsl, o["near_t"])]
                                      if not math.isnan(o["near_t"]) and len(lat) else float("nan"))
            rows.append(dict(
                N=int(N), bucket=b, mean_len=float(g.q_trigrams.mean()),
                n_q=int((g.t_max == t0).sum()), pool=int(g["pool"].iloc[0]),
                lat_floor_us=lat_floor,
                **{f"peak_t@{k}": rec[k]["peak_t"] for k in KS},
                **{f"near_t@{k}": rec[k]["near_t"] for k in KS},
                **{f"peak@{k}": rec[k]["peak"] for k in KS},
                **{f"hump@{k}": rec[k]["peak"] - rec[k]["end"] for k in KS},
                lat_near50_us=rec[CEIL_K]["lat_near"],
            ))
    res = pd.DataFrame(rows)

    # N-independence of the operating point (near_t) per (bucket, k)
    stab = []
    for b in buckets:
        sub = res[res.bucket == b]
        if sub.empty:
            continue
        row = dict(bucket=b, mean_len=float(sub.mean_len.mean()))
        for k in KS:
            v = sub[f"near_t@{k}"].values
            row[f"near_t@{k}_med"] = float(np.nanmedian(v))
            row[f"near_t@{k}_spread"] = float(np.nanmax(v) - np.nanmin(v))
        stab.append(row)
    stab = pd.DataFrame(stab)

    # Fit near_t vs query length (linear-space error), at the ceiling k and at k=10.
    fits = {}
    for k in (10, CEIL_K):
        if len(stab) < 3:
            continue
        x = stab.mean_len.values
        y = stab[f"near_t@{k}_med"].values
        m = ~np.isnan(y)
        if m.sum() < 3:
            continue
        x, y = x[m], y[m]
        cap = float(np.median(np.sort(y)[-max(1, len(y) // 3):]))
        rising = y < cap - 1e-9
        a, b = np.polyfit(x[rising], y[rising], 1) if rising.sum() >= 2 else np.polyfit(x, y, 1)
        pred = np.minimum(cap, a * x + b)
        ratio = y / np.maximum(pred, 1e-9)
        fits[k] = dict(a=float(a), b=float(b), T_cap=float(cap),
                       ratio_min=float(np.min(ratio)), ratio_med=float(np.median(ratio)),
                       ratio_max=float(np.max(ratio)))

    summary = dict(corpus=df.attrs.get("corpus", ""), Ns=[int(n) for n in Ns],
                   ks=KS, near_eps=NEAR_EPS, buckets=buckets,
                   fits=fits, stability=stab.to_dict("records"),
                   per_N_bucket=res.to_dict("records"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return df, res, stab, fits, buckets, Ns


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
            ts, y = recall_curve(g, k) if k else latency_curve(g)
            ax.plot(ts, y, marker=".", label=f"N={N:,}")
        ax.set_title(f"len {b}")
        ax.set_xlabel("t_max")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(buckets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=110)
    plt.close(fig)


def plot_opt_vs_len(stab, fits, out):
    if stab.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k in KS:
        ax.plot(stab.mean_len, stab[f"near_t@{k}_med"], "o-", label=f"near-optimal t_max @k={k}")
    if CEIL_K in fits:
        f = fits[CEIL_K]
        xs = np.linspace(stab.mean_len.min(), stab.mean_len.max(), 100)
        ax.plot(xs, np.minimum(f["T_cap"], f["a"] * xs + f["b"]), "k--",
                label=f"fit@{CEIL_K}: min({f['T_cap']:.0f}, {f['a']:.2f}·len+{f['b']:.1f})")
    ax.set_xlabel("query length (mean trigram count in bucket)")
    ax.set_ylabel("near-optimal t_max")
    ax.set_title("Operating-point t_max vs query length, by k")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "opt_vs_length.png", dpi=120)
    plt.close(fig)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="t_max selection-optimum analysis")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--docs", default="1000,5000,10000,50000,100000,500000,1000000")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edits", type=int, default=2)
    ap.add_argument("--max-tmax", type=int, default=64)
    ap.add_argument("--pool", type=int, default=0,
                    help="fixed generous pool; 0 = scale with N (the default)")
    ap.add_argument("--pool-cap", type=int, default=12000,
                    help="cap on the per-N scaled pool, to bound rerank cost")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-csv", action="store_true")
    args = ap.parse_args()
    args.docs = [int(x) for x in args.docs.split(",")]
    out = Path(args.out or f"tmax-{args.corpus}")
    out.mkdir(parents=True, exist_ok=True)

    df = collect(args, out)
    df.attrs["corpus"] = args.corpus
    df, res, stab, fits, buckets, Ns = analyse(df, out)

    facet(df, buckets, Ns, CEIL_K, f"recall@{CEIL_K} (ceiling)", "ceiling_vs_tmax.png", out,
          f"Recall@{CEIL_K} vs t_max (selection ceiling, generous pool) — {args.corpus}")
    facet(df, buckets, Ns, SMALL_K, f"recall@{SMALL_K} (precision)", "hump_vs_tmax.png", out,
          f"Recall@{SMALL_K} vs t_max — the precision hump — {args.corpus}")
    facet(df, buckets, Ns, None, "median latency (µs)", "latency_vs_tmax.png", out,
          f"Latency vs t_max (cost side) — {args.corpus}")
    plot_opt_vs_len(stab, fits, out)

    pd.set_option("display.width", 200)
    print(f"\n=== t_max optimum — corpus={args.corpus}  N={Ns} ===")
    print("\nOperating point (cheapest t_max within "
          f"{NEAR_EPS:.0%} recall of the peak), median over N, with N-spread:")
    cols = ["bucket", "mean_len"] + [f"near_t@{k}_med" for k in KS] + \
           [f"near_t@{k}_spread" for k in KS]
    print(stab[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nPeak recall and hump (peak − recall@max-t_max), per N×bucket:")
    pc = ["N", "bucket", "mean_len", "pool", f"peak@{CEIL_K}", f"hump@{CEIL_K}",
          f"hump@{SMALL_K}", f"near_t@{CEIL_K}", "lat_floor_us", "lat_near50_us"]
    print(res[pc].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for k in (10, CEIL_K):
        if k in fits:
            f = fits[k]
            print(f"\nFit near_t@{k} ≈ min({f['T_cap']:.0f}, {f['a']:.3f}·len + {f['b']:.2f})  "
                  f"linear ratio min/med/max = {f['ratio_min']:.2f}/{f['ratio_med']:.2f}/{f['ratio_max']:.2f}")
    print(f"\nWrote plots + summary.json to {out}/")


if __name__ == "__main__":
    main()
