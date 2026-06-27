//! Deserialize-cost baseline for the zero-copy investigation.
//!
//! Run with: `cargo run -p trifle-overlap --example deserialize_cost --release`
//!
//! Quantifies the cost the zero-copy / frozen-view work targets: how much per-query time goes to
//! **deserializing posting containers out of stored blobs** into owned `RoaringBitmap`s, vs
//! operating on postings already in hand. The gap is the copy a zero-copy view would remove.
//!
//! This is backend-agnostic (pure `roaring` here) — it establishes the number any croaring +
//! frozen-view variant must beat. `--release` only.

use std::time::{Duration, Instant};

use roaring::RoaringBitmap;

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

fn make_postings(k: usize, card: u32, universe: u32, planted: u32, seed: u64) -> Vec<RoaringBitmap> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| {
            let mut bm = RoaringBitmap::new();
            for id in 0..planted.min(card) {
                bm.insert(id);
            }
            while bm.len() < card as u64 {
                bm.insert(planted + rng.below(universe - planted));
            }
            bm
        })
        .collect()
}

fn serialize_all(postings: &[RoaringBitmap]) -> Vec<Vec<u8>> {
    postings
        .iter()
        .map(|bm| {
            let mut buf = Vec::with_capacity(bm.serialized_size());
            bm.serialize_into(&mut buf).expect("serialize");
            buf
        })
        .collect()
}

fn median(iters: usize, mut f: impl FnMut()) -> Duration {
    f();
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
    const PLANTED: u32 = 2_000;
    const ITERS: usize = 50;
    const K: usize = 8;

    println!("trifle-overlap deserialize-cost baseline (release)\n");
    println!(
        "  per query: load K={K} selected postings + build the bit-sliced counter.\n  \
         'owned' = postings already in memory; 'deserialize' = decode from stored blobs first.\n"
    );
    println!(
        "{:>10} | {:>11} | {:>14} | {:>16} | {:>10}",
        "card", "blob KiB", "owned build µs", "deser.+build µs", "deser. %"
    );
    println!(
        "{:->10}-+-{:->11}-+-{:->14}-+-{:->16}-+-{:->10}",
        "", "", "", "", ""
    );

    for &card in &[2_000u32, 8_000, 32_000, 128_000, 512_000] {
        let postings = make_postings(K, card, UNIVERSE, PLANTED, 0x5EED ^ card as u64);
        let blobs = serialize_all(&postings);
        let blob_kib = blobs.iter().map(|b| b.len()).sum::<usize>() as f64 / 1024.0;

        // Baseline: postings already owned (clone simulates having them in hand), build BSI.
        let owned = median(ITERS, || {
            let p = postings.clone();
            let counter = trifle_overlap::Counter::build(p, 1.0, 2);
            let mut w = counter.walk();
            let mut n = 0;
            while n < 10 && counter.advance(&mut w).is_some() {
                n += 1;
            }
            std::hint::black_box(n);
        });
        let clone_only = median(ITERS, || {
            std::hint::black_box(postings.clone());
        });
        let owned = us(owned.saturating_sub(clone_only));

        // With deserialize: decode each selected posting from its blob, then build BSI.
        let deser = median(ITERS, || {
            let p: Vec<RoaringBitmap> = blobs
                .iter()
                .map(|b| RoaringBitmap::deserialize_from(b.as_slice()).expect("deser"))
                .collect();
            let counter = trifle_overlap::Counter::build(p, 1.0, 2);
            let mut w = counter.walk();
            let mut n = 0;
            while n < 10 && counter.advance(&mut w).is_some() {
                n += 1;
            }
            std::hint::black_box(n);
        });
        let deser = us(deser);

        let pct = (1.0 - owned / deser.max(1e-9)) * 100.0;
        println!(
            "{:>10} | {:>11.1} | {:>14.1} | {:>16.1} | {:>9.1}%",
            card, blob_kib, owned, deser, pct
        );
    }

    println!(
        "\n  The 'deser. %' column is the share of per-query time spent decoding posting containers\n  \
         out of stored blobs — the copy a zero-copy frozen view (croaring) aims to remove. It\n  \
         grows with cardinality (deserialize is O(containers·density)); trifle selects the rarest\n  \
         (sparsest) postings, so the realistic win is the upper-left, but a common token sneaking\n  \
         into the selection is exactly where zero-copy would pay off most."
    );
}
