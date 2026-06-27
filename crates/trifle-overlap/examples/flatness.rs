//! Flatness / scaling benchmark for `trifle-overlap` (CRoaring backend).
//!
//! Run with: `cargo run -p trifle-overlap --example flatness --release`
//!
//! Demonstrates the engine's real property: candidate generation is a fixed *count* of bitmap
//! operations (`O(k·log k)`, cardinality-independent), so wall-clock is sublinear in cardinality
//! (sparse regime) and flat in the dense bitmap-container regime — pulling away from a naive
//! per-id counter as postings densify. Experiments:
//!   - sparse regime (#1): build+top-10 vs cardinality (vs a naive HashMap overlap counter);
//!   - dense regime (#1b): one croaring container, vary density — BSI wall-clock stays flat;
//!   - scaling in `k` (#2): number of selected postings;
//!   - lazy early-stop (#3): top-10 vs full drain.
//!
//! Debug numbers are meaningless — always `--release`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use croaring::Bitmap;
use trifle_overlap::{Counter, Scored};

/// Tiny deterministic xorshift RNG (no `rand` dep).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
}

/// `k` postings over `[0, universe)`, each of cardinality `card`. `planted` ids are forced into
/// every posting (a shared high-overlap head to stream); the rest is random fill.
fn make_postings(k: usize, card: u32, universe: u32, planted: u32, seed: u64) -> Vec<Bitmap> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let mut bm = Bitmap::new();
            for id in 0..planted.min(card) {
                bm.add(id);
            }
            while bm.cardinality() < card as u64 {
                bm.add(planted + rng.below(universe - planted));
            }
            bm
        })
        .collect()
}

/// The naive alternative: count how many postings contain each id with a HashMap, then select the
/// top-`want`. `O(Σ cardinality)` — linear in posting size.
fn naive_topk(postings: &[Bitmap], want: usize) -> Vec<(u32, u32)> {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for p in postings {
        for id in p.iter() {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    let mut v: Vec<(u32, u32)> = counts.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(want);
    v
}

fn median<F: FnMut()>(iters: usize, mut f: F) -> Duration {
    f(); // warmup
    let mut ds: Vec<Duration> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed()
        })
        .collect();
    ds.sort_unstable();
    ds[ds.len() / 2]
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Build + pull the top-`TOPN` (the engine borrows the postings, so timing has no clone).
fn build_top(postings: &[Bitmap], topn: usize) {
    let counter = Counter::build(postings, 1.0, 2);
    let mut w = counter.walk();
    let mut n = 0;
    while n < topn && counter.advance(&mut w).is_some() {
        n += 1;
    }
    std::hint::black_box(n);
}

fn main() {
    const UNIVERSE: u32 = 4_000_000;
    const PLANTED: u32 = 2_000;
    const ITERS: usize = 9;
    const TOPN: usize = 10;

    println!("trifle-overlap flatness benchmark — CRoaring backend (release)\n");

    // ---- Experiment 1: sparse regime (fixed k) -------------------------------------------
    println!("== 1. sparse regime: build + top-{TOPN} vs cardinality (k = 8 postings) ==");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "card", "BSI (µs)", "naive (µs)", "speedup"
    );
    println!("{:->10}-+-{:->12}-+-{:->12}-+-{:->10}", "", "", "", "");
    for &card in &[2_000u32, 8_000, 32_000, 128_000, 512_000] {
        let postings = make_postings(8, card, UNIVERSE, PLANTED, 0xC0FFEE ^ card as u64);
        let bsi = median(ITERS, || build_top(&postings, TOPN));
        let naive = median(ITERS, || {
            std::hint::black_box(naive_topk(&postings, TOPN));
        });
        println!(
            "{:>10} | {:>12.1} | {:>12.1} | {:>9.1}x",
            card,
            us(bsi),
            us(naive),
            us(naive) / us(bsi).max(1e-9)
        );
    }
    println!(
        "\n  BSI grows SUBLINEARLY and pulls away from naive as density rises. trifle feeds the\n  \
         engine the RAREST (sparsest) postings by selection — the cheap left of this table.\n"
    );

    // ---- Experiment 1b: dense regime (one container) -------------------------------------
    println!("== 1b. dense regime: one croaring container, vary density (k = 8) ==");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "card", "BSI (µs)", "naive (µs)", "speedup"
    );
    println!("{:->10}-+-{:->12}-+-{:->12}-+-{:->10}", "", "", "", "");
    const DENSE_U: u32 = 65_536;
    for &card in &[4_000u32, 12_000, 24_000, 48_000] {
        let postings = make_postings(8, card, DENSE_U, 400, 0xDED5 ^ card as u64);
        let bsi = median(ITERS, || build_top(&postings, TOPN));
        let naive = median(ITERS, || {
            std::hint::black_box(naive_topk(&postings, TOPN));
        });
        println!(
            "{:>10} | {:>12.1} | {:>12.1} | {:>9.1}x",
            card,
            us(bsi),
            us(naive),
            us(naive) / us(bsi).max(1e-9)
        );
    }
    println!(
        "\n  Density rose 12x; BSI wall-clock stays ~flat (each posting is one fixed-width bitmap\n  \
         container — the op count is constant), while naive grows ~linearly in set bits.\n"
    );

    // ---- Experiment 2: scaling in k ------------------------------------------------------
    println!("== 2. build + top-{TOPN}, vs k (cardinality = 64k) ==");
    println!("{:>6} | {:>12}", "k", "BSI (µs)");
    println!("{:->6}-+-{:->12}", "", "");
    for &kk in &[2usize, 4, 8, 16, 24] {
        let postings = make_postings(kk, 64_000, UNIVERSE, PLANTED, 0xBEEF ^ kk as u64);
        let bsi = median(ITERS, || build_top(&postings, TOPN));
        println!("{:>6} | {:>12.1}", kk, us(bsi));
    }
    println!();

    // ---- Experiment 3: lazy early-stop (top-10 vs full drain) ----------------------------
    println!("== 3. lazy early-stop (k = 12, cardinality = 128k, all-weight-1) ==");
    let postings = make_postings(12, 128_000, UNIVERSE, PLANTED, 0xD00D);
    let top10 = median(ITERS, || build_top(&postings, TOPN));
    let drain_all = median(ITERS, || {
        let counter = Counter::build(&postings, 1.0, 2);
        let got: Vec<Scored> = counter.stream().collect();
        std::hint::black_box(got.len());
    });
    println!("  top-{TOPN}:   {:>10.1} µs", us(top10));
    println!("  full drain: {:>10.1} µs", us(drain_all));
    println!(
        "  (full drain is cheap because the all-weight-1 fast path takes overlap = score with no\n  \
         per-id probing; the walk is no longer a bottleneck.)\n"
    );

    // Sanity smoke test.
    let counter = Counter::build(&make_postings(6, 50_000, UNIVERSE, PLANTED, 1), 1.0, 2);
    let top: Vec<Scored> = counter.stream().take(5).collect();
    assert!(top.iter().all(|s| s.overlap >= 2));
    assert!(top.windows(2).all(|p| p[0].score >= p[1].score));
    println!("correctness smoke test passed (floor honored, score-descending).");
}
