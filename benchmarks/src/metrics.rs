//! Summaries — pure, dependency-free. Latency percentiles, throughput, recall@k,
//! and a plain u64 distribution (reused for the Σ-cardinality work-done curve).

use std::collections::HashSet;
use std::time::Duration;

/// Nearest-rank percentile of a *sorted* slice: the value at rank
/// `ceil(p/100 · n)` (1-based), clamped into range. `p` in `[0, 100]`.
fn nearest_rank(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// A distribution over `u64` samples (already collected, sorted on construction).
pub struct Dist {
    sorted: Vec<u64>,
}

impl Dist {
    pub fn new(mut samples: Vec<u64>) -> Self {
        samples.sort_unstable();
        Dist { sorted: samples }
    }
    pub fn count(&self) -> usize {
        self.sorted.len()
    }
    pub fn pct(&self, p: f64) -> u64 {
        nearest_rank(&self.sorted, p)
    }
    pub fn max(&self) -> u64 {
        self.sorted.last().copied().unwrap_or(0)
    }
    pub fn mean(&self) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let sum: u128 = self.sorted.iter().map(|&n| n as u128).sum();
        sum as f64 / self.sorted.len() as f64
    }
}

/// A latency distribution: nanosecond samples, summarized as `Duration`s.
pub struct Latency {
    dist: Dist,
}

impl Latency {
    pub fn from_durations(samples: Vec<Duration>) -> Self {
        Latency {
            dist: Dist::new(samples.iter().map(|d| d.as_nanos() as u64).collect()),
        }
    }
    pub fn p50(&self) -> Duration {
        Duration::from_nanos(self.dist.pct(50.0))
    }
    pub fn p90(&self) -> Duration {
        Duration::from_nanos(self.dist.pct(90.0))
    }
    pub fn p95(&self) -> Duration {
        Duration::from_nanos(self.dist.pct(95.0))
    }
    pub fn p99(&self) -> Duration {
        Duration::from_nanos(self.dist.pct(99.0))
    }
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.dist.max())
    }
}

/// Queries per second from a query count and a wall-clock elapsed time.
pub fn throughput(queries: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    queries as f64 / secs
}

/// **set-recall@k** over (possibly multi-passage, sparse) relevance judgments:
/// `mean over queries of |top-k(results[i]) ∩ relevant[i]| / |relevant[i]|`.
///
/// `relevant[i]` is the in-corpus judged-relevant id set for query `i`. Both engines
/// MUST be scored against the *same* `relevant` slice with the *same* `k` — that is the
/// symmetry contract; a per-engine label set or k is how a bogus delta sneaks in. A
/// query with an empty `relevant[i]` is undefined (0/0) and is **excluded from the
/// denominator identically for every engine**, never scored for one side only. The
/// metric truncates each result to `k` itself, so an engine that over-returns can't
/// inflate. The count actually scored is [`scored_queries`].
pub fn set_recall_at_k(results: &[Vec<i64>], relevant: &[Vec<i64>], k: usize) -> f64 {
    let mut scored = 0usize;
    let mut acc = 0.0f64;
    for (got, rel) in results.iter().zip(relevant.iter()) {
        if rel.is_empty() {
            continue;
        }
        scored += 1;
        let topk: HashSet<i64> = got.iter().copied().take(k).collect();
        let hits = rel.iter().filter(|r| topk.contains(r)).count();
        acc += hits as f64 / rel.len() as f64;
    }
    if scored == 0 {
        0.0
    } else {
        acc / scored as f64
    }
}

/// How many queries [`set_recall_at_k`] actually scores (those with ≥1 in-corpus
/// relevant id) — report this so a silently-shrunk query set is visible.
pub fn scored_queries(relevant: &[Vec<i64>]) -> usize {
    relevant.iter().filter(|r| !r.is_empty()).count()
}

/// **MRR@k**: mean over scored queries of `1 / rank` where `rank` (1-based) is the position of
/// the first relevant id within the top-`k` of `got`, or `0` if none appears. Same
/// empty-relevant exclusion contract as [`set_recall_at_k`].
pub fn mrr_at_k(results: &[Vec<i64>], relevant: &[Vec<i64>], k: usize) -> f64 {
    let mut scored = 0usize;
    let mut acc = 0.0f64;
    for (got, rel) in results.iter().zip(relevant.iter()) {
        if rel.is_empty() {
            continue;
        }
        scored += 1;
        let rels: HashSet<i64> = rel.iter().copied().collect();
        if let Some(pos) = got.iter().take(k).position(|id| rels.contains(id)) {
            acc += 1.0 / (pos as f64 + 1.0);
        }
    }
    if scored == 0 {
        0.0
    } else {
        acc / scored as f64
    }
}

/// **nDCG@k** with binary relevance: `DCG@k / IDCG@k` averaged over scored queries, where
/// `DCG = Σ_i rel(got[i]) / log2(i + 2)` over the top-`k` (0-based `i`) and `IDCG` is the DCG of
/// the ideal ranking (all relevant first). Same empty-relevant exclusion as [`set_recall_at_k`].
pub fn ndcg_at_k(results: &[Vec<i64>], relevant: &[Vec<i64>], k: usize) -> f64 {
    let mut scored = 0usize;
    let mut acc = 0.0f64;
    for (got, rel) in results.iter().zip(relevant.iter()) {
        if rel.is_empty() {
            continue;
        }
        scored += 1;
        let rels: HashSet<i64> = rel.iter().copied().collect();
        let dcg: f64 = got
            .iter()
            .take(k)
            .enumerate()
            .filter(|(_, id)| rels.contains(id))
            .map(|(i, _)| 1.0 / ((i as f64 + 2.0).log2()))
            .sum();
        let ideal_hits = rel.len().min(k);
        let idcg: f64 = (0..ideal_hits)
            .map(|i| 1.0 / ((i as f64 + 2.0).log2()))
            .sum();
        if idcg > 0.0 {
            acc += dcg / idcg;
        }
    }
    if scored == 0 {
        0.0
    } else {
        acc / scored as f64
    }
}

/// Render a `Duration` compactly in the largest unit that keeps it readable.
pub fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_endpoints_and_interior() {
        let xs: Vec<u64> = (1..=100).collect();
        assert_eq!(nearest_rank(&xs, 0.0), 1);
        assert_eq!(nearest_rank(&xs, 50.0), 50);
        assert_eq!(nearest_rank(&xs, 99.0), 99);
        assert_eq!(nearest_rank(&xs, 100.0), 100);
    }

    #[test]
    fn empty_dist_is_zero_not_panic() {
        let d = Dist::new(vec![]);
        assert_eq!(d.pct(50.0), 0);
        assert_eq!(d.max(), 0);
        assert_eq!(d.mean(), 0.0);
    }

    #[test]
    fn set_recall_singleton_is_classic_recall() {
        // one-element relevant sets behave like classic recall@k (the fuzzy eval's case)
        let results = vec![vec![1, 2, 3], vec![9, 8], vec![4]];
        let relevant = vec![vec![2], vec![7], vec![4]];
        assert!((set_recall_at_k(&results, &relevant, 10) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn set_recall_partial_multi_passage() {
        // 1 of 2 relevant retrieved -> 0.5 for that query
        let results = vec![vec![5, 1]];
        let relevant = vec![vec![1, 2]];
        assert!((set_recall_at_k(&results, &relevant, 10) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn set_recall_truncates_at_k_and_excludes_empty() {
        // target at rank 3 is excluded at k=2; the empty-relevant query is not counted
        let results = vec![vec![9, 8, 1], vec![1]];
        let relevant = vec![vec![1], vec![]];
        assert_eq!(set_recall_at_k(&results, &relevant, 2), 0.0);
        assert_eq!(scored_queries(&relevant), 1);
    }

    #[test]
    fn throughput_zero_elapsed_is_zero() {
        assert_eq!(throughput(10, Duration::ZERO), 0.0);
    }

    #[test]
    fn mrr_uses_first_relevant_rank_and_respects_k() {
        // first relevant (id 1) sits at rank 3 → 1/3; empty-relevant query excluded.
        let results = vec![vec![9, 8, 1], vec![1]];
        let relevant = vec![vec![1], vec![]];
        assert!((mrr_at_k(&results, &relevant, 10) - 1.0 / 3.0).abs() < 1e-9);
        // at k=2 the relevant id is past the cutoff → 0.
        assert_eq!(mrr_at_k(&results, &relevant, 2), 0.0);
    }

    #[test]
    fn ndcg_is_one_at_rank_one_and_discounts_below() {
        // relevant at rank 1 → perfect nDCG.
        assert!((ndcg_at_k(&[vec![1, 2, 3]], &[vec![1]], 10) - 1.0).abs() < 1e-9);
        // relevant at rank 2 → 1/log2(3) discounted against an ideal IDCG of 1.
        let want = (1.0 / 3.0_f64.log2()) / 1.0;
        assert!((ndcg_at_k(&[vec![9, 1]], &[vec![1]], 10) - want).abs() < 1e-9);
    }
}
