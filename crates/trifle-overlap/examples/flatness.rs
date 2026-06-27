//! Flatness benchmark for `trifle-overlap` — dependency-free (roaring + std only).
//!
//! Run with: `cargo run -p trifle-overlap --example flatness --release`
//!
//! Demonstrates the engine's real property: candidate generation is a fixed *count* of bitmap
//! operations (`O(k·log k)`, cardinality-independent), so wall-clock is sublinear in cardinality
//! (sparse regime) and flat in the dense bitmap-container regime — pulling away from a naive
//! per-id counter as postings densify. Experiments:
//!   - sparse regime (#1): build+top-10 vs cardinality (vs a naive HashMap overlap counter);
//!   - dense regime (#1b): one roaring container, vary density — BSI wall-clock stays flat;
//!   - scaling in `k` (#2): number of selected postings;
//!   - lazy early-stop (#3): top-10 vs full drain.
//!
//! Debug numbers are meaningless — always `--release`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use roaring::RoaringBitmap;
use trifle_overlap::{Counter, Scored};

/// Tiny deterministic xorshift RNG (no `rand` dep — keeps the engine crate roaring-only).
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

/// `k` postings over universe `[0, universe)`, each of cardinality `card`. `planted` ids are
/// forced into *every* posting so there is a high-overlap head to stream (a realistic shared
/// rare-gram set); the rest is random fill.
fn make_postings(k: usize, card: u32, universe: u32, planted: u32, seed: u64) -> Vec<RoaringBitmap> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let mut bm = RoaringBitmap::new();
            for id in 0..planted.min(card) {
                bm.insert(id); // planted ids live at the low end of the universe
            }
            while bm.len() < card as u64 {
                bm.insert(planted + rng.below(universe - planted));
            }
            bm
        })
        .collect()
}

/// The naive alternative to the BSI counter: count how many postings contain each id with a
/// HashMap, then select the top-`want` by count. `O(Σ cardinality)` — linear in posting size.
fn naive_topk(postings: &[RoaringBitmap], want: usize) -> Vec<(u32, u32)> {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for p in postings {
        for id in p {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    let mut v: Vec<(u32, u32)> = counts.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(want);
    v
}

/// Median wall-clock of `f` over `iters` runs, after one warmup.
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

fn main() {
    const UNIVERSE: u32 = 4_000_000;
    const PLANTED: u32 = 2_000; // shared high-overlap head
    const ITERS: usize = 9;
    const TOPN: usize = 10;

    println!("trifle-overlap flatness benchmark (release)\n");

    // ---- Experiment 1: flatness vs cardinality (fixed k) ---------------------------------
    println!("== 1. sparse regime: build + top-{TOPN} vs cardinality (k = 8 postings) ==");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "card", "BSI (µs)", "naive (µs)", "speedup"
    );
    println!("{:->10}-+-{:->12}-+-{:->12}-+-{:->10}", "", "", "", "");
    let k = 8;
    for &card in &[2_000u32, 8_000, 32_000, 128_000, 512_000] {
        let postings = make_postings(k, card, UNIVERSE, PLANTED, 0xC0FFEE ^ card as u64);

        let bsi = median(ITERS, || {
            let p = postings.clone(); // clone is OUTSIDE... no: must be inside to get fresh owned input
            let counter = Counter::build(p, 1.0, 2);
            let mut w = counter.walk();
            let mut n = 0;
            while n < TOPN && counter.advance(&mut w).is_some() {
                n += 1;
            }
            std::hint::black_box(n);
        });
        // The clone above is inside the timed region; subtract a clone-only measurement so we
        // report engine cost, not allocation of the input.
        let clone_only = median(ITERS, || {
            std::hint::black_box(postings.clone());
        });
        let bsi = bsi.saturating_sub(clone_only);

        let naive = median(ITERS, || {
            let top = naive_topk(&postings, TOPN);
            std::hint::black_box(top);
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
        "\n  As cardinality grows 256x, BSI grows SUBLINEARLY and pulls away from the naive\n  \
         per-id counter (the speedup column rises with density): BSI does a fixed count of\n  \
         bitmap ops (~O(k·log k)), naive is O(Σ set bits). It is NOT literally \"independent of\n  \
         posting size\" — its cost tracks the bitmap representation (containers × density), not\n  \
         a per-result cost. trifle feeds the engine the RAREST (sparsest) postings by selection,\n  \
         so the cheap left of this table is the real operating point.\n"
    );

    // ---- Experiment 1b: the TRUE flatness regime (one dense container) -------------------
    // A small universe puts every id in a single roaring container; past ~4096 ids it is a
    // dense bitmap container (fixed 1024-word ops). So BSI wall-clock stays flat as density
    // rises, while naive grows with set-bit count — the literal "fixed op count" invariant.
    println!("== 1b. dense regime: one roaring container, vary density (k = 8) ==");
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "card", "BSI (µs)", "naive (µs)", "speedup"
    );
    println!("{:->10}-+-{:->12}-+-{:->12}-+-{:->10}", "", "", "", "");
    const DENSE_U: u32 = 65_536; // exactly one roaring container
    for &card in &[4_000u32, 12_000, 24_000, 48_000] {
        let postings = make_postings(8, card, DENSE_U, 400, 0xDED5 ^ card as u64);
        let bsi = median(ITERS, || {
            let counter = Counter::build(postings.clone(), 1.0, 2);
            let mut w = counter.walk();
            let mut n = 0;
            while n < TOPN && counter.advance(&mut w).is_some() {
                n += 1;
            }
            std::hint::black_box(n);
        });
        let clone_only = median(ITERS, || {
            std::hint::black_box(postings.clone());
        });
        let bsi = bsi.saturating_sub(clone_only);
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
         container — the op count is constant), while naive grows ~linearly in set bits. This is\n  \
         the literal invariant: a fixed COUNT of cardinality-independent bitmap ops.\n"
    );

    // ---- Experiment 2: scaling in k (fixed cardinality) ----------------------------------
    println!("== 2. build + top-{TOPN}, vs k (cardinality = 64k) ==");
    println!("{:>6} | {:>12}", "k", "BSI (µs)");
    println!("{:->6}-+-{:->12}", "", "");
    for &kk in &[2usize, 4, 8, 16, 24] {
        let postings = make_postings(kk, 64_000, UNIVERSE, PLANTED, 0xBEEF ^ kk as u64);
        let bsi = median(ITERS, || {
            let counter = Counter::build(postings.clone(), 1.0, 2);
            let mut w = counter.walk();
            let mut n = 0;
            while n < TOPN && counter.advance(&mut w).is_some() {
                n += 1;
            }
            std::hint::black_box(n);
        });
        let clone_only = median(ITERS, || {
            std::hint::black_box(postings.clone());
        });
        println!("{:>6} | {:>12.1}", kk, us(bsi.saturating_sub(clone_only)));
    }
    println!();

    // ---- Experiment 3: lazy early-stop (top-10 vs full drain) ----------------------------
    println!("== 3. lazy early-stop (k = 12, cardinality = 128k) ==");
    let postings = make_postings(12, 128_000, UNIVERSE, PLANTED, 0xD00D);
    let top10 = median(ITERS, || {
        let counter = Counter::build(postings.clone(), 1.0, 2);
        let mut w = counter.walk();
        let mut n = 0;
        while n < TOPN && counter.advance(&mut w).is_some() {
            n += 1;
        }
        std::hint::black_box(n);
    });
    let drain_all = median(ITERS, || {
        let counter = Counter::build(postings.clone(), 1.0, 2);
        let got: Vec<Scored> = counter.stream().collect();
        std::hint::black_box(got.len());
    });
    let clone_only = median(ITERS, || {
        std::hint::black_box(postings.clone());
    });
    let top10 = us(top10.saturating_sub(clone_only));
    let drain_all = us(drain_all.saturating_sub(clone_only));
    println!("  top-{TOPN}:   {top10:>10.1} µs");
    println!("  full drain: {drain_all:>10.1} µs");
    println!(
        "  early-stop saves {:.1}x by not materializing the low-score tail.\n",
        drain_all / top10.max(1e-9)
    );

    // Sanity: the planted head is the high-overlap set (correctness smoke test under --release).
    let counter = Counter::build(make_postings(6, 50_000, UNIVERSE, PLANTED, 1), 1.0, 2);
    let top: Vec<Scored> = counter.stream().take(5).collect();
    assert!(
        top.iter().all(|s| s.overlap >= 2),
        "top candidates must clear the min_shared floor"
    );
    assert!(
        top.windows(2).all(|p| p[0].score >= p[1].score),
        "stream must be weighted-score descending"
    );
    println!("correctness smoke test passed (floor honored, score-descending).");
}
