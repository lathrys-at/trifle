//! A/B benchmark: does **croaring** (SIMD CRoaring) + **zero-copy Portable views** beat the
//! pure-Rust `roaring` crate for trifle's bit-sliced overlap build?
//!
//! Run: `cargo run --manifest-path crates/croaring-bsi-bench/Cargo.toml --release`
//!
//! Two levers, measured separately, with identical algorithms differing only in the bitmap
//! library:
//!   A. **plane math (SIMD):** build the weighted bit-sliced planes from inputs already in
//!      memory — isolates the AND/XOR/ANDNOT op speed (the dominant per-query cost).
//!   B. **zero-copy load:** build from stored portable blobs — roaring deserializes (copies)
//!      each posting; croaring constructs a transient `BitmapView` over the bytes (no copy).
//!
//! Both `roaring`-crate portable bytes and `croaring` portable bytes are byte-identical, so the
//! croaring view reads trifle's *existing* stored format with no migration.

use std::time::{Duration, Instant};

use croaring::{Bitmap, BitmapView, Portable};
use roaring::RoaringBitmap;

// ---- deterministic RNG (no dep) -----------------------------------------------------------
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

/// Per-posting df-anchored tier weight {1,2,3,4} (inlined copy of trifle_overlap::tier_weights
/// so this experimental crate stays standalone).
fn tier_weights(cards: &[u64], d: f64) -> Vec<u32> {
    let d = if d > 0.0 { d } else { 1.0 };
    let df_max = cards.iter().copied().max().unwrap_or(1).max(1) as f64;
    cards
        .iter()
        .map(|&c| {
            let df = c.max(1) as f64;
            let steps = ((df_max / df).log2() / d).round().max(0.0) as u32;
            1 + steps.min(3)
        })
        .collect()
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

fn portable_blobs(postings: &[RoaringBitmap]) -> Vec<Vec<u8>> {
    postings
        .iter()
        .map(|bm| {
            let mut v = Vec::with_capacity(bm.serialized_size());
            bm.serialize_into(&mut v).unwrap();
            v
        })
        .collect()
}

// ---- the BSI plane build, three ways (identical algorithm) ---------------------------------

fn add_weighted_roaring(acc: &mut Vec<RoaringBitmap>, posting: &RoaringBitmap, w: u32) {
    if posting.is_empty() {
        return;
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(RoaringBitmap::new());
            }
            let mut carry = &acc[start] & posting;
            acc[start] ^= posting;
            let mut level = start + 1;
            while !carry.is_empty() {
                while acc.len() <= level {
                    acc.push(RoaringBitmap::new());
                }
                let nc = &acc[level] & &carry;
                acc[level] ^= &carry;
                carry = nc;
                level += 1;
            }
        }
        bit += 1;
    }
}

fn add_weighted_croaring_owned(acc: &mut Vec<Bitmap>, posting: &Bitmap, w: u32) {
    if posting.is_empty() {
        return;
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(Bitmap::new());
            }
            let mut carry: Bitmap = posting & &acc[start];
            acc[start] ^= posting;
            let mut level = start + 1;
            while !carry.is_empty() {
                while acc.len() <= level {
                    acc.push(Bitmap::new());
                }
                let nc = &acc[level] & &carry;
                acc[level] ^= &carry;
                carry = nc;
                level += 1;
            }
        }
        bit += 1;
    }
}

/// Zero-copy: the posting operand is a transient `BitmapView` over the stored bytes — never
/// materialized into an owned `Bitmap`. Only the (small) carry is owned.
fn add_weighted_croaring_view(acc: &mut Vec<Bitmap>, view: &BitmapView<'_>, w: u32) {
    if view.is_empty() {
        return;
    }
    let mut bit = 0u32;
    while (w >> bit) != 0 {
        if (w >> bit) & 1 == 1 {
            let start = bit as usize;
            while acc.len() <= start {
                acc.push(Bitmap::new());
            }
            let mut carry: Bitmap = view & &acc[start]; // commuted AND &BitmapView & &Bitmap
            acc[start] ^= view; // in-place XOR against the view (no posting copy)
            let mut level = start + 1;
            while !carry.is_empty() {
                while acc.len() <= level {
                    acc.push(Bitmap::new());
                }
                let nc = &acc[level] & &carry;
                acc[level] ^= &carry;
                carry = nc;
                level += 1;
            }
        }
        bit += 1;
    }
}

fn build_roaring(postings: &[RoaringBitmap], weights: &[u32]) -> Vec<RoaringBitmap> {
    let mut acc = Vec::new();
    for (p, &w) in postings.iter().zip(weights) {
        add_weighted_roaring(&mut acc, p, w);
    }
    acc
}

fn build_croaring_owned(bms: &[Bitmap], weights: &[u32]) -> Vec<Bitmap> {
    let mut acc = Vec::new();
    for (b, &w) in bms.iter().zip(weights) {
        add_weighted_croaring_owned(&mut acc, b, w);
    }
    acc
}

fn build_croaring_view(blobs: &[Vec<u8>], weights: &[u32]) -> Vec<Bitmap> {
    let mut acc = Vec::new();
    for (blob, &w) in blobs.iter().zip(weights) {
        // SAFETY: `blob` outlives `view`; portable layout, alignment 1.
        let view = unsafe { BitmapView::deserialize::<Portable>(blob) };
        add_weighted_croaring_view(&mut acc, &view, w);
    }
    acc
}

// ---- correctness cross-check --------------------------------------------------------------

fn count_roaring(planes: &[RoaringBitmap], id: u32) -> u32 {
    planes
        .iter()
        .enumerate()
        .map(|(b, p)| (p.contains(id) as u32) << b)
        .sum()
}
fn count_croaring(planes: &[Bitmap], id: u32) -> u32 {
    planes
        .iter()
        .enumerate()
        .map(|(b, p)| (p.contains(id) as u32) << b)
        .sum()
}

// ---- timing -------------------------------------------------------------------------------

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

fn run_regime(name: &str, universe: u32, planted: u32, cards: &[u32]) {
    const K: usize = 8;
    const ITERS: usize = 60;
    println!("== {name} (k={K}) ==");
    println!(
        "{:>8} | {:>13} {:>13} {:>8} | {:>15} {:>15} {:>15}",
        "card", "A:roaring µs", "A:croar µs", "speedup", "B:roar des+bld", "B:croar view", "B:croar owned"
    );
    println!(
        "{:->8}-+-{:->13}-{:->13}-{:->8}-+-{:->15}-{:->15}-{:->15}",
        "", "", "", "", "", "", ""
    );
    for &card in cards {
        let postings = make_postings(K, card, universe, planted, 0xBADC0DE ^ card as u64);
        let weights = tier_weights(
            &postings.iter().map(RoaringBitmap::len).collect::<Vec<_>>(),
            1.0,
        );
        let blobs = portable_blobs(&postings);
        let cro_owned: Vec<Bitmap> = blobs
            .iter()
            .map(|b| Bitmap::try_deserialize::<Portable>(b).unwrap())
            .collect();

        // Correctness: roaring planes vs croaring-view planes must encode identical counts.
        {
            let pr = build_roaring(&postings, &weights);
            let pc = build_croaring_view(&blobs, &weights);
            for &id in &[0u32, 1, 7, 99, planted, planted.saturating_add(1)] {
                assert_eq!(
                    count_roaring(&pr, id),
                    count_croaring(&pc, id),
                    "count mismatch at id {id} (card {card})"
                );
            }
        }

        // A: plane math, inputs already in memory.
        let a_roaring = median(ITERS, || {
            std::hint::black_box(build_roaring(&postings, &weights));
        });
        let a_croaring = median(ITERS, || {
            std::hint::black_box(build_croaring_owned(&cro_owned, &weights));
        });

        // B: load + build from stored portable blobs.
        let b_roaring = median(ITERS, || {
            let p: Vec<RoaringBitmap> = blobs
                .iter()
                .map(|b| RoaringBitmap::deserialize_from(b.as_slice()).unwrap())
                .collect();
            std::hint::black_box(build_roaring(&p, &weights));
        });
        let b_view = median(ITERS, || {
            std::hint::black_box(build_croaring_view(&blobs, &weights));
        });
        let b_owned = median(ITERS, || {
            let p: Vec<Bitmap> = blobs
                .iter()
                .map(|b| Bitmap::try_deserialize::<Portable>(b).unwrap())
                .collect();
            std::hint::black_box(build_croaring_owned(&p, &weights));
        });

        println!(
            "{:>8} | {:>13.1} {:>13.1} {:>7.2}x | {:>15.1} {:>15.1} {:>15.1}",
            card,
            us(a_roaring),
            us(a_croaring),
            us(a_roaring) / us(a_croaring).max(1e-9),
            us(b_roaring),
            us(b_view),
            us(b_owned),
        );
    }
    println!();
}

fn main() {
    println!("croaring vs roaring — BSI build A/B (release)\n");
    println!("A = build weighted planes from in-memory inputs (plane-math / SIMD).");
    println!("B = build from stored portable blobs (roaring deserializes; croaring views).\n");
    run_regime(
        "sparse regime (4M universe)",
        4_000_000,
        2_000,
        &[2_000, 8_000, 32_000, 128_000, 512_000],
    );
    run_regime(
        "dense regime (one 65536 container)",
        65_536,
        400,
        &[4_000, 12_000, 24_000, 48_000],
    );
    println!(
        "Read: A:speedup = roaring/croaring on plane math (>1 ⇒ croaring SIMD wins). B columns\n\
         compare per-query load+build from blobs: croaring 'view' avoids the deserialize copy that\n\
         roaring 'des+bld' and croaring 'owned' both pay."
    );
}
