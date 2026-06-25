#!/usr/bin/env python3
"""Calibrate trifle's rerank-pool-depth law `p(k, N)` on a corpus.

This is the analysis half of the calibration tool; the measurement half is the
`ranksweep` subcommand of `trifle-benchmarks` (which this script drives). It:

  1. sweeps the rerank pool depth across a grid of index sizes N (`--docs`) for the
     chosen corpus, building the recall@k(pool, N) matrix via `ranksweep`;
  2. fits the rising-regime power law  `p*_target ≈ c · k^a · N^b`  for each recall
     target (a fraction of the deep-pool recall *ceiling*);
  3. renders the manifold, the p*-vs-N scaling, and the power-law collapse to PNGs;
  4. emits the constant `c` at each `--targets` p* level (default 0.5, 0.9, 0.95, 0.99)
     with its spread, and maps them onto trifle's `Effort` ladder.

See `tools/README.md` for the full mathematical treatment (from Zipf's law) of *why*
the pool requirement is a power law in N, and what `c` means.

Usage:
    python3 benchmarks/tools/calibrate_pool.py --corpus msmarco --queries 500 --seed 42
    python3 benchmarks/tools/calibrate_pool.py --corpus geonames-cities --targets 0.9,0.99
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
# trifle's shipped Effort ladder (must match `Effort::coeff` in src/lib.rs); the tool
# reports which calibrated recall target each constant corresponds to, so drift is visible.
EFFORT_LADDER = {"Low": 0.03, "Medium (default)": 0.05, "High": 0.10, "Max": 0.30}


# ---- measurement: drive the ranksweep subcommand ----------------------------
def run_sweep(corpus, docs, queries, seed, edits, max_pool=None):
    """Return the recall@k(pool,N) matrix as a DataFrame by invoking `ranksweep`."""
    rows = []
    for n in docs:
        print(f"  ranksweep N={n:,} (corpus={corpus}) ...", file=sys.stderr, flush=True)
        cmd = ["cargo", "run", "-q", "-p", "trifle-benchmarks", "--release", "--",
               "ranksweep", "--corpus", corpus, "--docs", str(n),
               "--queries", str(queries), "--seed", str(seed), "--edits", str(edits)]
        if max_pool:
            cmd += ["--max-pool", str(max_pool)]
        cp = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stderr[-3000:])
            raise SystemExit(f"ranksweep failed for N={n} (corpus={corpus})")
        lines = [l for l in cp.stdout.splitlines() if l.strip()]
        rows += lines
    df = pd.DataFrame([r.split(",") for r in rows],
                      columns=["N", "edits", "pool", "k", "queries", "recall"]).astype(float)
    return df.astype({"N": int, "edits": int, "pool": int, "k": int, "queries": int})


# ---- analysis primitives ----------------------------------------------------
class Matrix:
    def __init__(self, df):
        self.df = df.drop_duplicates(["N", "pool", "k"])
        self.Ns = sorted(self.df.N.unique())
        self.Ks = sorted(self.df.k.unique())

    def curve(self, N, k):
        d = self.df[(self.df.N == N) & (self.df.k == k)].sort_values("pool")
        return d.pool.values * 1.0, d.recall.values * 1.0

    def ceiling(self, N, k):           # best recall observed (deepest pool, but max-guarded
                                       # so a noisy dip at the tail can't depress the target)
        _, R = self.curve(N, k)
        return float(np.max(R)) if len(R) else float("nan")

    def pstar(self, N, k, frac):       # smallest pool reaching frac*ceiling
        P, R = self.curve(N, k)
        if len(P) == 0:
            return float("nan")
        tgt = frac * self.ceiling(N, k)
        idx = np.where(R >= tgt)[0]
        if len(idx) == 0:
            return P[-1]
        i = idx[0]
        if i == 0:
            return P[0]
        r0, r1 = R[i - 1], R[i]
        t = (tgt - r0) / (r1 - r0) if r1 != r0 else 1.0
        return math.exp(np.log(P[i - 1]) + t * (np.log(P[i]) - np.log(P[i - 1])))


def fit_power_law(m, frac, rising_mult=1.3, c_mult=1.05):
    """The constant `c = p*/√(kN)` over points with any over-fetch (`p* > c_mult·k`), plus
    the `log p* = const + a·log k + b·log N` power-law fit over the cleaner rising regime
    (`p* > rising_mult·k`). Low targets sit in the floor (`p* ≈ k`) so they yield a `c` but
    not a power fit — reported as such rather than dropped."""
    pts = [(N, k, m.pstar(N, k, frac)) for N in m.Ns for k in m.Ks]
    F = pd.DataFrame(pts, columns=["N", "k", "p"])
    Fc = F[(F.p > c_mult * F.k) & (F.p > 0)]
    if len(Fc) == 0:
        return None
    c = (Fc.p / np.sqrt(Fc.k * Fc.N)).values
    out = dict(c_med=float(np.median(c)), c_p10=float(np.percentile(c, 10)),
               c_p90=float(np.percentile(c, 90)), n_c=int(len(Fc)),
               const=None, k_exp=None, N_exp=None, r2=None, n_fit=0)
    Fr = F[(F.p > rising_mult * F.k) & (F.p > 0)]
    if len(Fr) >= 4:
        X = np.c_[np.ones(len(Fr)), np.log(Fr.k), np.log(Fr.N)]
        y = np.log(Fr.p)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
        out.update(const=float(math.exp(beta[0])), k_exp=float(beta[1]),
                   N_exp=float(beta[2]), r2=float(r2), n_fit=int(len(Fr)))
    return out


# ---- plots ------------------------------------------------------------------
def plot_manifold(m, corpus, path):
    rows = (len(m.Ks) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4.5 * rows), squeeze=False)
    for ax, k in zip(axes.flat, m.Ks):
        for N in m.Ns:
            P, R = m.curve(N, k)
            if len(P):
                ax.plot(P, R, marker="o", ms=3, label=f"N={N:,}")
        ax.set_xscale("log"); ax.set_title(f"recall@{k} vs pool")
        ax.set_xlabel("rerank pool depth"); ax.set_ylabel("recall"); ax.grid(alpha=.3)
        ax.axvline(k, color="gray", ls=":", lw=1)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"recall vs rerank-pool depth — {corpus}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_pstar(m, frac, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in m.Ks:
        xs, ys = [], []
        for N in m.Ns:
            p = m.pstar(N, k, frac)
            if not math.isnan(p):
                xs.append(N); ys.append(p)
        ax.plot(xs, ys, marker="o", label=f"k={k}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (index size)"); ax.set_ylabel(f"p* (pool for {frac:.0%} of ceiling)")
    ax.set_title(f"p* vs N at {frac:.0%} of ceiling")
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_collapse(m, frac, fit, path):
    a, b = fit["k_exp"], fit["N_exp"]
    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = [], []
    for k in m.Ks:
        for N in m.Ns:
            p = m.pstar(N, k, frac)
            if p > 1.3 * k:
                x = (k ** a) * (N ** b); xs.append(x); ys.append(p)
        sel = [(((k ** a) * (N ** b)), m.pstar(N, k, frac)) for N in m.Ns if m.pstar(N, k, frac) > 1.3 * k]
        if sel:
            xv, yv = zip(*sel); ax.scatter(xv, yv, s=30, label=f"k={k}")
    xs, ys = np.array(xs), np.array(ys)
    c = np.sum(xs * ys) / np.sum(xs * xs)
    xx = np.array([xs.min(), xs.max()])
    ax.plot(xx, c * xx, "k--", label=f"p* = {c:.4f}·k^{a:.2f}·N^{b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"k^{a:.2f}·N^{b:.2f} (fitted predictor)")
    ax.set_ylabel(f"p* at {frac:.0%} of ceiling")
    ax.set_title(f"p* vs fitted predictor (R²={fit['r2']:.3f})")
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True,
                    help="synthetic | msmarco | geonames-cities | geonames-all")
    ap.add_argument("--queries", type=int, default=500, help="queries sampled per N [500]")
    ap.add_argument("--seed", type=int, default=42, help="master seed [42]")
    ap.add_argument("--edits", type=int, default=2, help="typos injected (synthetic/geonames) [2]")
    ap.add_argument("--docs", default="1000,5000,10000,50000,100000,500000,1000000",
                    help="comma-separated index sizes N")
    ap.add_argument("--targets", default="0.5,0.9,0.95,0.99",
                    help="comma-separated recall-fraction-of-ceiling targets for the c constants")
    ap.add_argument("--max-pool", type=int, default=None,
                    help="deepest rerank pool to sweep; raise past 2048 to push the ceiling "
                         "at very large N [ranksweep default: 2048]")
    ap.add_argument("--out", default=None, help="output dir [calibration-<corpus>]")
    ap.add_argument("--reuse-csv", action="store_true",
                    help="reuse an existing matrix.csv in the out dir (skip the sweep)")
    args = ap.parse_args()

    docs = [int(x) for x in args.docs.split(",")]
    targets = [float(x) for x in args.targets.split(",")]
    out = Path(args.out or f"calibration-{args.corpus}"); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "matrix.csv"

    if args.reuse_csv and csv_path.exists():
        print(f"reusing {csv_path}", file=sys.stderr)
        df = pd.read_csv(csv_path)
    else:
        print(f"sweeping corpus={args.corpus} queries={args.queries} seed={args.seed} "
              f"N={docs}", file=sys.stderr)
        df = run_sweep(args.corpus, docs, args.queries, args.seed, args.edits, args.max_pool)
        df.to_csv(csv_path, index=False)

    m = Matrix(df)
    print(f"loaded N={m.Ns} k={m.Ks}", file=sys.stderr)

    # plots (manifold once; p*/collapse at the 95% reference target if present, else max target)
    plot_manifold(m, args.corpus, out / "manifold.png")
    ref = 0.95 if 0.95 in targets else max(targets)
    ref_fit = fit_power_law(m, ref)
    plot_pstar(m, ref, out / "pstar_vs_N.png")
    if ref_fit and ref_fit["k_exp"] is not None:
        plot_collapse(m, ref, ref_fit, out / "collapse.png")

    # fits + constants at every target
    summary = {"corpus": args.corpus, "queries": args.queries, "seed": args.seed,
               "edits": args.edits, "N": m.Ns, "k": m.Ks, "targets": {}}
    print("\n=== p* = c·√(kN), and the rising-regime power-law fit ===")
    print(f"{'target':>7} {'c=p*/√(kN)':>11} {'c p10..p90':>16}   power fit (k^a·N^b)")
    for frac in sorted(targets):
        f = fit_power_law(m, frac)
        if not f:
            print(f"{frac:>6.0%}  (no over-fetch needed — p* == k)"); continue
        summary["targets"][f"{frac}"] = f
        fit = (f"k^{f['k_exp']:.2f}·N^{f['N_exp']:.2f}  R²={f['r2']:.3f}"
               if f["k_exp"] is not None else "floor regime (p≈k)")
        print(f"{frac:>6.0%} {f['c_med']:>11.4f}  {f['c_p10']:>6.3f}..{f['c_p90']:<6.3f}  {fit}")

    # map trifle's shipped Effort constants onto the calibrated targets
    print("\n=== trifle Effort ladder vs this calibration ===")
    print(f"{'level':>18} {'shipped c':>9}  → nearest calibrated target")
    cal = [(frac, summary['targets'][f'{frac}']['c_med']) for frac in sorted(targets)
           if f'{frac}' in summary['targets']]
    for level, c in EFFORT_LADDER.items():
        near = min(cal, key=lambda fc: abs(fc[1] - c)) if cal else None
        tag = f"~{near[0]:.0%} of ceiling (c_cal={near[1]:.3f})" if near else "n/a"
        print(f"{level:>18} {c:>9.2f}  → {tag}")
    summary["effort_ladder"] = EFFORT_LADDER

    def _json(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"not JSON serializable: {type(o)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    print(f"\nwrote: {out}/matrix.csv, manifold.png, pstar_vs_N.png, collapse.png, summary.json")


if __name__ == "__main__":
    main()
