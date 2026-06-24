//! Search latency scaling sweep (§10.2): trifle's central promise is *flatness* —
//! bit-sliced overlap is posting-size-independent and document-frequency reads are
//! PK seeks, so search latency should stay near-flat as the corpus grows. This bench
//! sweeps the corpus size and measures search latency so that curve is observable.
//!
//! Run with `cargo bench`. The sweep sizes are modest by default (building a corpus
//! dominates wall-clock); raise `SCALES` to reproduce the §10 curve at 100k–1M.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;
use trifle::{Config, Index, SearchOpts};

/// Corpus sizes to sweep (segment count). The architectural claim is that the
/// latency curve across these is near-flat.
const SCALES: &[usize] = &[1_000, 10_000, 50_000];

/// A small, fixed vocabulary so generated docs share trigrams realistically and
/// queries actually match across the corpus.
const VOCAB: &[&str] = &[
    "quick", "brown", "foxes", "jumping", "over", "lazy", "sleeping", "dogs", "river", "mountain",
    "forest", "quartz", "sphinx", "wizard", "vortex", "puzzle", "cipher", "amber", "cobalt",
    "lantern", "harbor", "meadow", "thunder", "whisper", "crimson", "velvet", "marble", "ancient",
    "distant", "silent", "golden", "frozen",
];

/// A tiny deterministic PRNG (xorshift64) — no rand dependency, reproducible runs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn word(&mut self) -> &'static str {
        VOCAB[(self.next() as usize) % VOCAB.len()]
    }
}

/// A doc of 4–8 words drawn from the vocabulary.
fn make_doc(rng: &mut Rng) -> String {
    let n = 4 + (rng.next() as usize % 5);
    (0..n).map(|_| rng.word()).collect::<Vec<_>>().join(" ")
}

/// Build a populated, compacted index of `n` segments in a fresh temp dir.
fn build(n: usize) -> (Index, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let index = Index::open_at(&dir.path().join("bench.db"), Config::default()).unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // One batch keeps it one transaction; chunk so memory stays bounded.
    let mut batch = Vec::with_capacity(4096);
    for doc_id in 0..n as i64 {
        batch.push(trifle::Segment::new(
            doc_id,
            "field",
            "body",
            make_doc(&mut rng),
        ));
        if batch.len() == 4096 {
            index.insert_batch(batch.drain(..)).unwrap();
        }
    }
    if !batch.is_empty() {
        index.insert_batch(batch).unwrap();
    }
    index.compact().unwrap(); // fold into bases — the steady-state read shape
    (index, dir)
}

/// A fixed query set: clean phrases and typo'd ones (the typo path is the point).
const QUERIES: &[&str] = &[
    "quick brown foxes",
    "lazy sleeping dogs",
    "quikc bronw foxs", // typos
    "anceint distnat",  // typos
    "quartz sphinx wizard",
];

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_scaling");
    group.measurement_time(Duration::from_secs(5));
    for &n in SCALES {
        let (index, _dir) = build(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                let q = QUERIES[i % QUERIES.len()];
                i += 1;
                let hits = index.search(black_box(q), SearchOpts::new(10)).unwrap();
                black_box(hits);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
