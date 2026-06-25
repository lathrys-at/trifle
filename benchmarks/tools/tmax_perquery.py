#!/usr/bin/env python3
"""Per-query knee distribution analysis (the core t_max statistic) — reanalysis only.

Mean recall over queries with *different individual optima* manufactures a flat plateau even
when every query has a sharp knee, so the mean-based optimum is biased toward "flat" and needs
huge query counts to locate. The per-query knee is symmetric and sample-efficient:

  t_enter(q,k) = smallest t_max at which q's relevant doc reaches rank <= k  (first recovered)
  t_exit(q,k)  = largest  t_max at which rank <= k                            (per-query hump:
                 if t_exit < max grid, the doc was recovered then DROPPED OUT — bounded-by-
                 correctness, measured directly and unsmeared)
  never-recovered queries are right-CENSORED — reported as a recovery rate, never dropped.

Answers:
  Q-EXIST  does the entry-knee depend on query length? slope + bootstrap CI (symmetric: a
           length law, a null, or underpowered are all reachable conclusions).
  Q-DRIFT  (preview) does the entry-knee center shift with N?  (full answer needs 500k/1M.)
  HUMP     re-audit "the hump grows with N" via the exit-knee / drop-out distribution.

Reads a raw tmaxsweep CSV: columns N,t_max,q_trigrams,rank,latency_us (one row per
(query,t_max); a query block is the run of rows between t_max resets within an N).

    python3 benchmarks/tools/tmax_perquery.py --csv /tmp/tmax-msmarco-wide/tmax_raw.csv \\
        --out /tmp/tmax-msmarco-wide/perquery
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 10, 50]
LEN_EDGES = [4, 7, 10, 14, 19, 26, 36, 51, 9999]
BOOT = 2000
RNG = np.random.default_rng(0)


def bucket_label(lo, hi):
    return f"{lo}+" if hi >= 9999 else f"{lo}-{hi - 1}"


def length_bucket(qt):
    i = int(np.clip(np.digitize([qt], LEN_EDGES, right=False)[0] - 1, 0, len(LEN_EDGES) - 2))
    return bucket_label(LEN_EDGES[i], LEN_EDGES[i + 1])


# ---- reconstruct per-query blocks + entry/exit knees ------------------------
def per_query_knees(df):
    rows = []
    for N, gN in df.groupby("N", sort=True):
        g = gN.reset_index(drop=True)
        qid = (g["t_max"].diff() < 0).cumsum()        # new query each time t_max resets
        for _, q in g.assign(qid=qid).groupby("qid"):
            q = q.sort_values("t_max")
            ts = q["t_max"].to_numpy()
            rank = q["rank"].to_numpy()
            qt = int(q["q_trigrams"].iloc[0])
            tmax_top = int(ts[-1])
            for k in KS:
                hit = (rank > 0) & (rank <= k)
                if hit.any():
                    enter = int(ts[np.argmax(hit)])
                    exit_ = int(ts[len(hit) - 1 - np.argmax(hit[::-1])])
                    rows.append(dict(N=int(N), qt=qt, bucket=length_bucket(qt), k=k,
                                     recovered=True, t_enter=enter, t_exit=exit_,
                                     dropped=exit_ < tmax_top))
                else:
                    rows.append(dict(N=int(N), qt=qt, bucket=length_bucket(qt), k=k,
                                     recovered=False, t_enter=np.nan, t_exit=np.nan,
                                     dropped=False))
    return pd.DataFrame(rows)


def boot_slope(x, y):
    """OLS slope of y~x with a percentile bootstrap 95% CI (resampling queries)."""
    if len(x) < 8:
        return dict(n=int(len(x)), slope=float("nan"), lo=float("nan"), hi=float("nan"))
    a = float(np.polyfit(x, y, 1)[0])
    idx = np.arange(len(x))
    bs = np.empty(BOOT)
    for i in range(BOOT):
        s = RNG.choice(idx, len(idx), replace=True)
        bs[i] = np.polyfit(x[s], y[s], 1)[0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=int(len(x)), slope=a, lo=float(lo), hi=float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ship-slope", type=float, default=0.10,
                    help="shippable length-law slope threshold (t_max per trigram); a null is "
                         "'powered' only if the slope CI excludes this")
    args = ap.parse_args()
    out = Path(args.out or (Path(args.csv).parent / "perquery"))
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    pq = per_query_knees(df)
    pq.to_csv(out / "per_query_knees.csv", index=False)
    Ns = sorted(pq.N.unique())
    buckets = sorted(pq.bucket.unique(), key=lambda b: int(b.rstrip("+").split("-")[0]))

    summary = {"csv": str(args.csv), "Ns": [int(n) for n in Ns], "ship_slope": args.ship_slope}

    # ---- recovery rate (censoring), per (N, k) ----
    rec = (pq.groupby(["N", "k"])["recovered"].mean().reset_index(name="recovery_rate"))
    summary["recovery_rate"] = rec.to_dict("records")

    # ---- Q-EXIST: entry-knee vs query length, slope + CI (pooled and per N) ----
    qexist = {}
    for k in KS:
        d = pq[(pq.k == k) & pq.recovered]
        pooled = boot_slope(d.qt.to_numpy(float), d.t_enter.to_numpy(float))
        per_n = {int(N): boot_slope(dd.qt.to_numpy(float), dd.t_enter.to_numpy(float))
                 for N, dd in d.groupby("N")}
        # implied spread of the knee across the observed length range, from the pooled slope
        if not np.isnan(pooled["slope"]):
            span = (d.qt.max() - d.qt.min())
            pooled["implied_knee_span"] = float(pooled["slope"] * span)
            pooled["len_range"] = [float(d.qt.min()), float(d.qt.max())]
            pooled["powered_null"] = bool(abs(pooled["lo"]) < args.ship_slope
                                          and abs(pooled["hi"]) < args.ship_slope)
            pooled["sloped"] = bool(pooled["lo"] > 0 or pooled["hi"] < 0)
        qexist[k] = dict(pooled=pooled, per_N=per_n)
    summary["Q_EXIST"] = qexist

    # ---- Q-DRIFT preview: entry-knee center vs N, per length bucket ----
    drift = {}
    for k in KS:
        rows = []
        for b in buckets:
            cell = []
            for N in Ns:
                d = pq[(pq.k == k) & pq.recovered & (pq.bucket == b) & (pq.N == N)]
                cell.append(dict(N=int(N), n=int(len(d)),
                                 med=float(d.t_enter.median()) if len(d) else float("nan"),
                                 iqr=float(d.t_enter.quantile(.75) - d.t_enter.quantile(.25))
                                 if len(d) else float("nan")))
            rows.append(dict(bucket=b, by_N=cell))
        drift[k] = rows
    summary["Q_DRIFT_preview"] = drift

    # ---- HUMP re-audit: per-query drop-out fraction vs N ----
    hump = {}
    for k in KS:
        d = pq[(pq.k == k) & pq.recovered]
        hump[k] = [dict(N=int(N), n=int(len(dd)),
                        drop_frac=float(dd.dropped.mean()),
                        exit_med=float(dd.t_exit.median()))
                   for N, dd in d.groupby("N")]
    summary["HUMP"] = hump

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- plots ----
    # entry-knee distribution per length bucket (one panel per k), N as box groups
    for k in KS:
        d = pq[(pq.k == k) & pq.recovered]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data, labels = [], []
        for b in buckets:
            v = d[d.bucket == b].t_enter.dropna().to_numpy()
            if len(v) >= 5:
                data.append(v)
                labels.append(f"{b}\n(n={len(v)})")
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_xlabel("query length bucket (trigram count)")
        ax.set_ylabel(f"entry-knee t_max @k={k}")
        ax.set_title(f"Per-query entry-knee distribution vs length — k={k} ({Path(args.csv).parent.name})")
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"entryknee_dist_k{k}.png", dpi=120)
        plt.close(fig)

    # entry-knee median vs N per length bucket (Q-DRIFT preview), at k=10
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in buckets:
        xs, ys = [], []
        for N in Ns:
            d = pq[(pq.k == 10) & pq.recovered & (pq.bucket == b) & (pq.N == N)]
            if len(d) >= 5:
                xs.append(N)
                ys.append(d.t_enter.median())
        if xs:
            ax.plot(xs, ys, "o-", label=f"len {b}")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median entry-knee t_max @k=10")
    ax.set_title("median entry-knee t_max vs N, by length")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "drift_preview_k10.png", dpi=120)
    plt.close(fig)

    # ---- printed report ----
    pd.set_option("display.width", 200)
    print(f"\n=== per-query knee analysis — {args.csv} ===")
    print(f"N: {Ns}   length buckets: {buckets}")
    print("\nRecovery rate (fraction of queries whose relevant doc is EVER in top-k), per N:")
    print(rec.pivot(index="N", columns="k", values="recovery_rate")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nQ-EXIST — entry-knee vs query length (pooled over N): slope [95% CI]  per trigram")
    print(f"  (shippable-slope threshold = ±{args.ship_slope}; 'powered null' ⇒ CI inside it)")
    for k in KS:
        p = qexist[k]["pooled"]
        verdict = ("SLOPED" if p.get("sloped") else
                   "powered-null" if p.get("powered_null") else "underpowered")
        print(f"  k={k:>2}: slope={p['slope']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]  "
              f"n={p['n']}  implied knee span over len {p.get('len_range')}"
              f" = {p.get('implied_knee_span', float('nan')):+.1f} t_max  →  {verdict}")

    print("\nQ-EXIST per N (slope [CI] at k=10) — is the length law present at each size?")
    for N, s in qexist[10]["per_N"].items():
        print(f"  N={N:>7}: slope={s['slope']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]  n={s['n']}")

    print("\nQ-DRIFT preview — median entry-knee @k=10 by (length bucket × N):")
    rowfmt = "  {:>7} " + " ".join(f"{b:>9}" for b in buckets)
    print(rowfmt.format("N", *[]))
    for N in Ns:
        cells = []
        for b in buckets:
            d = pq[(pq.k == 10) & pq.recovered & (pq.bucket == b) & (pq.N == N)]
            cells.append(f"{d.t_enter.median():.0f}(n{len(d)})" if len(d) >= 5 else "   -    ")
        print(f"  {N:>7} " + " ".join(f"{c:>9}" for c in cells))

    print("\nHUMP re-audit — per-query drop-out fraction (recovered then exits top-k by max t_max):")
    print(f"{'N':>8}" + "".join(f"  drop@{k}  exit_med@{k}" for k in KS))
    for N in Ns:
        s = f"{N:>8}"
        for k in KS:
            h = next((x for x in hump[k] if x["N"] == N), None)
            s += f"  {h['drop_frac']:>6.3f}  {h['exit_med']:>10.0f}" if h else "     -          -"
        print(s)
    print(f"\nWrote per_query_knees.csv + plots + summary.json to {out}/")


if __name__ == "__main__":
    main()
