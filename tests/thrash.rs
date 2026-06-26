//! Property-based thrashing: randomized op sequences (upsert / remove / remove-segment /
//! compact / rebuild) applied in lockstep to the real index and an in-memory oracle, with
//! strong invariants checked after **every** op.
//!
//! This is the one deliberately expensive integration test. The oracle is the model of
//! expected state — `key -> {label -> text}`, mirroring trifle's v0.2 document/segment
//! model — and the invariants are the ones an independent model can verify without
//! reimplementing ranking:
//!
//! - segment count matches the oracle exactly (catches lost/leaked/double writes);
//! - `compact()` clears the delta backlog (fold correctness);
//! - every returned match is *faithful* — its `(key, label, text)` is a real segment
//!   currently in the oracle (catches phantom docs, stale text, monotonic-id reuse,
//!   swap/rebuild corruption);
//! - a segment's own text re-finds its doc (recall survives churn + compaction);
//! - every span is a valid char boundary into the returned text.
//!
//! Fixtures stay tiny on purpose (≤ 6 docs, ≤ 3 labels each, short texts) — a small state
//! space hit from many random angles finds interleaving bugs a big corpus would not.

mod common;
use common::*;

use std::collections::BTreeMap;

use proptest::prelude::*;
use trifle::store::Sidecar;
use trifle::tokenize::DefaultTokenizer;
use trifle::{Document, SearchOpts};

/// Real overlapping vocabulary, so searches actually match across docs.
const WORDS: &[&str] = &[
    "quick", "brown", "foxes", "jumps", "lazy", "dogs", "river", "mount", "quartz", "sphinx",
    "wizard", "vortex", "puzzle", "cipher", "amber", "cobalt",
];
const LABELS: &[&str] = &["a", "b", "c"];

/// `key -> {label -> text}`, mirroring trifle's v0.2 segment model (labels unique per doc).
type Oracle = BTreeMap<i64, BTreeMap<String, String>>;
type Idx = trifle::Index<DefaultTokenizer, Sidecar>;

#[derive(Debug, Clone)]
enum Op {
    /// Insert-or-replace the named labels (keeps a doc's other labels).
    Upsert {
        doc: i64,
        segs: Vec<(String, String)>,
    },
    /// Drop a whole document.
    Remove {
        doc: i64,
    },
    /// Drop one `(doc, label)` segment.
    RemoveSegment {
        doc: i64,
        label: String,
    },
    Compact,
    Rebuild {
        corpus: Vec<(i64, String, String)>,
    },
}

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(WORDS), 1..4).prop_map(|ws| ws.join(" "))
}
fn label() -> impl Strategy<Value = String> {
    prop::sample::select(LABELS).prop_map(String::from)
}
/// A non-empty set of **distinct** labels, each with its own text — `insert`/`upsert`
/// require distinct labels within one call (a doc holds each label once).
fn segs() -> impl Strategy<Value = Vec<(String, String)>> {
    prop::sample::subsequence(LABELS.to_vec(), 1..=LABELS.len())
        .prop_flat_map(|labels| {
            let n = labels.len();
            (Just(labels), prop::collection::vec(text(), n))
        })
        .prop_map(|(labels, texts)| labels.into_iter().map(String::from).zip(texts).collect())
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (1i64..=6, segs()).prop_map(|(doc, segs)| Op::Upsert { doc, segs }),
        2 => (1i64..=6).prop_map(|doc| Op::Remove { doc }),
        2 => (1i64..=6, label()).prop_map(|(doc, label)| Op::RemoveSegment { doc, label }),
        2 => Just(Op::Compact),
        1 => prop::collection::vec((1i64..=6, label(), text()), 0..6)
                .prop_map(|corpus| Op::Rebuild { corpus }),
    ]
}

/// Apply one op to both the index and the oracle, keeping them in lockstep.
fn apply(idx: &Idx, oracle: &mut Oracle, op: Op) {
    match op {
        Op::Upsert { doc, segs } => {
            let pairs: Vec<(&str, &str)> =
                segs.iter().map(|(l, t)| (l.as_str(), t.as_str())).collect();
            let mut w = idx.writer().unwrap();
            w.upsert(doc, &pairs).unwrap();
            w.commit().unwrap();
            // Upsert replaces named labels (last-wins within the batch) and keeps the rest.
            let entry = oracle.entry(doc).or_default();
            for (l, t) in &segs {
                entry.insert(l.clone(), t.clone());
            }
        }
        Op::Remove { doc } => {
            let mut w = idx.writer().unwrap();
            w.remove(doc).unwrap();
            w.commit().unwrap();
            oracle.remove(&doc);
        }
        Op::RemoveSegment { doc, label } => {
            let mut w = idx.writer().unwrap();
            w.remove_segment(doc, &label).unwrap();
            w.commit().unwrap();
            if let Some(labels) = oracle.get_mut(&doc) {
                labels.remove(&label);
                if labels.is_empty() {
                    oracle.remove(&doc);
                }
            }
        }
        Op::Compact => {
            idx.compact().unwrap();
            assert_eq!(
                idx.stats().unwrap().delta_backlog,
                0,
                "compact clears the backlog"
            );
        }
        Op::Rebuild { corpus } => {
            // Rebuild is verbatim, but labels are unique per doc and keys unique per doc:
            // collapse the corpus to one Document per doc (last-wins per label).
            let mut model = Oracle::new();
            for (d, l, t) in &corpus {
                model.entry(*d).or_default().insert(l.clone(), t.clone());
            }
            let docs: Vec<Document> = model
                .iter()
                .map(|(d, labels)| {
                    Document::new(
                        *d,
                        labels.iter().map(|(l, t)| (l.clone(), t.clone())).collect(),
                    )
                })
                .collect();
            idx.rebuild(docs).unwrap();
            *oracle = model;
        }
    }
}

/// Is `m` a real segment currently in the oracle? (snapshot mode -> text matches.)
fn faithful(oracle: &Oracle, m: &trifle::Match) -> bool {
    m.key
        .as_i64()
        .and_then(|d| oracle.get(&d))
        .and_then(|labels| labels.get(&m.label))
        .map(|t| t.as_str())
        == m.text.as_deref()
}

/// Check every invariant against the oracle.
fn check(idx: &Idx, oracle: &Oracle) {
    let expected_segments: u64 = oracle.values().map(|v| v.len() as u64).sum();
    assert_eq!(
        idx.stats().unwrap().segments,
        expected_segments,
        "segment count"
    );

    // Sample a few existing docs (deterministic BTreeMap order) and verify recall + that
    // every returned match is faithful and well-formed.
    let reader = idx.reader().unwrap();
    for (doc, labels) in oracle.iter().take(4) {
        let Some((_, txt)) = labels.iter().next() else {
            continue;
        };
        let hits = reader.search(txt, SearchOpts::new(100)).unwrap();
        assert!(
            hit(&hits, *doc),
            "a segment's own text must re-find its doc"
        );
        for m in &hits {
            assert!(
                faithful(oracle, m),
                "result {m:?} is not a live oracle segment"
            );
            if let (Some((lo, hi)), Some(text)) = (m.span, m.text.as_deref()) {
                assert!(
                    text.is_char_boundary(lo) && text.is_char_boundary(hi) && lo < hi,
                    "span {:?} invalid for {text:?}",
                    m.span
                );
            }
        }
    }
}

proptest! {
    // Sized to thrash hard while staying under a ~30s wall-clock budget on a debug
    // `cargo test` (the lane CI runs); the one deliberately heavy test.
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn thrashing_preserves_every_invariant(ops in prop::collection::vec(op_strategy(), 6..48)) {
        let h = Harness::new();
        let mut oracle = Oracle::new();
        for op in ops {
            apply(&h.index, &mut oracle, op);
            check(&h.index, &oracle);
        }
    }
}
