//! End-to-end query-latency benchmark for the lean storage+stream spike.
//!
//! Run with: `cargo run -p trifle-lean --example query_latency --release`
//!
//! Builds a synthetic corpus of small documents, then measures `matches()` latency (tokenize →
//! select → engine walk → per-chunk provenance over the flattened `seg` table → dedup → batched
//! hydrate) — and the same with an opt-in raw-SQL `key IN rarray(?)` filter. Demonstrates the
//! whole shape is viable and fast. `--release` only.

use std::rc::Rc;
use std::time::{Duration, Instant};

use rusqlite::ToSql;
use rusqlite::types::Value;
use trifle_lean::{Filter, LeanIndex, SearchOpts};

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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

const VOCAB: &[&str] = &[
    "quick", "brown", "fox", "lazy", "dog", "silver", "river", "mountain", "garden", "winter",
    "summer", "yellow", "purple", "copper", "marble", "thunder", "whisper", "ember", "cobalt",
    "harbor", "lantern", "meadow", "orchard", "pebble", "ripple", "saffron", "tundra", "violet",
    "willow", "zephyr", "almond", "basil", "cedar", "dahlia", "fennel",
];

fn phrase(rng: &mut Rng, words: usize) -> String {
    (0..words)
        .map(|_| VOCAB[rng.below(VOCAB.len())])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Inject a transposition typo into a query to exercise fuzzy matching.
fn typo(s: &str) -> String {
    let mut c: Vec<char> = s.chars().collect();
    if c.len() >= 4 {
        c.swap(1, 2);
    }
    c.into_iter().collect()
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

fn build(n: usize, seed: u64) -> LeanIndex {
    let idx = LeanIndex::open_in_memory().unwrap();
    let mut rng = Rng::new(seed);
    for key in 0..n {
        let words = 4 + rng.below(4);
        let text = phrase(&mut rng, words);
        idx.insert(key as i64, "body", &text).unwrap();
    }
    idx
}

fn main() {
    const ITERS: usize = 200;
    println!("trifle-lean end-to-end query latency (release)\n");
    println!(
        "{:>10} | {:>14} | {:>18} | {:>16}",
        "docs", "matches µs", "matches+filter µs", "stream top-10 µs"
    );
    println!("{:->10}-+-{:->14}-+-{:->18}-+-{:->16}", "", "", "", "");

    for &n in &[10_000usize, 50_000, 200_000] {
        let idx = build(n, 0xABCDEF ^ n as u64);
        let opts = SearchOpts::default();

        // Rotate over a set of typo'd 2-word queries so we don't measure one cached path.
        let queries: Vec<String> = {
            let mut rng = Rng::new(99);
            (0..32)
                .map(|_| {
                    let p = phrase(&mut rng, 2);
                    typo(&p)
                })
                .collect()
        };
        let mut qi = 0usize;
        let mut next_q = || {
            let q = queries[qi % queries.len()].clone();
            qi += 1;
            q
        };

        let plain = median(ITERS, || {
            let q = next_q();
            let hits = idx.matches(&q, opts, 10, None).unwrap();
            std::hint::black_box(hits.len());
        });

        // A `key IN rarray(?)` filter over half the corpus (the universal, staleness-free mode).
        let allowed: Rc<Vec<Value>> =
            Rc::new((0..n as i64).step_by(2).map(Value::Integer).collect());
        let filtered = median(ITERS, || {
            let q = next_q();
            let params: Vec<&dyn ToSql> = vec![&allowed];
            let f = Filter {
                fragment: "key IN rarray(?1)",
                params: &params,
            };
            let hits = idx.matches(&q, opts, 10, Some(&f)).unwrap();
            std::hint::black_box(hits.len());
        });

        let streamed = median(ITERS, || {
            let q = next_q();
            let mut s = idx.candidates(&q, opts).unwrap();
            let mut pool = Vec::new();
            for item in s.by_ref().take(10) {
                pool.push(item.unwrap());
            }
            let hits = s.hydrate(&pool).unwrap();
            std::hint::black_box(hits.len());
        });

        println!(
            "{:>10} | {:>14.1} | {:>18.1} | {:>16.1}",
            n,
            plain.as_secs_f64() * 1e6,
            filtered.as_secs_f64() * 1e6,
            streamed.as_secs_f64() * 1e6,
        );
    }
    println!(
        "\n  End-to-end: tokenize -> rarest-first select -> BSI walk -> per-chunk provenance over\n  \
         the flat seg table -> dedup-by-key -> batched hydrate. The opt-in `key IN rarray(?)`\n  \
         filter folds into the per-chunk provenance query (no separate pass).\n  \
         NOTE: the filter column passes a HALF-CORPUS key array (worst case for rarray\n  \
         marshaling — up to 100k entries at 200k docs). A realistic selective filter (small\n  \
         allowed-key set, or a co-located ATTACH join) is far cheaper; this is the §11 cost,\n  \
         shown honestly rather than hidden behind a tiny filter set."
    );
}
