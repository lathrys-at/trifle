//! Summaries — pure, dependency-free. Latency percentiles, throughput, recall@k,
//! and a plain u64 distribution (reused for the Σ-cardinality work-done curve).

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

/// recall@k: the fraction of labeled queries whose ground-truth document id appears
/// in that query's returned id list. `results[i]` is the engine's answer for the
/// query whose relevant doc is `labels[i]`.
pub fn recall_at_k(results: &[Vec<i64>], labels: &[i64]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for (got, &want) in results.iter().zip(labels.iter()) {
        if got.contains(&want) {
            hits += 1;
        }
    }
    hits as f64 / labels.len() as f64
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
    fn recall_counts_membership() {
        let results = vec![vec![1, 2, 3], vec![9, 8], vec![4]];
        let labels = vec![2, 7, 4];
        assert!((recall_at_k(&results, &labels) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_zero_elapsed_is_zero() {
        assert_eq!(throughput(10, Duration::ZERO), 0.0);
    }
}
