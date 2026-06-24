//! Property-based thrashing: randomized op sequences (insert / remove / compact /
//! rebuild) applied in lockstep to the real index and an in-memory oracle, with
//! strong invariants checked after **every** op.
//!
//! This is the one deliberately expensive integration test. The oracle is the model
//! of expected state — `(doc_id, source) -> [(ref, text)]` — and the invariants are
//! the ones an independent model can verify without reimplementing ranking:
//!
//! - segment count matches the oracle exactly (catches lost/leaked/double writes);
//! - `compact()` clears the delta backlog and changes no result (fold correctness);
//! - every returned match is *faithful* — its `(doc, source, ref, text)` is a real
//!   segment currently in the oracle (catches phantom docs, stale text, monotonic-id
//!   reuse, swap/rebuild corruption);
//! - a segment's own text re-finds its doc (recall survives churn + compaction);
//! - every span is a valid char boundary into the returned text.
//!
//! Fixtures stay tiny on purpose (≤ 6 docs, short texts) — a small state space hit
//! from many random angles finds interleaving bugs that a big corpus would not.

mod common;
use common::*;

use std::collections::BTreeMap;

use proptest::prelude::*;
use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{SearchOpts, Segment};

/// Real overlapping vocabulary, so searches actually match across docs.
const WORDS: &[&str] = &[
    "quick", "brown", "foxes", "jumps", "lazy", "dogs", "river", "mount", "quartz", "sphinx",
    "wizard", "vortex", "puzzle", "cipher", "amber", "cobalt",
];
const SOURCES: &[&str] = &["field", "ocr", "caption"];
const REFS: &[&str] = &["a", "b", "c"];

/// `(doc_id, source) -> [(ref, text)]`, mirroring trifle's segment model.
type Oracle = BTreeMap<(i64, String), Vec<(String, String)>>;
type Idx = trifle::Index<TrigramTokenizer, Sidecar>;

#[derive(Debug, Clone)]
enum Op {
    Insert {
        doc: i64,
        source: String,
        segs: Vec<(String, String)>,
    },
    Remove {
        doc: i64,
    },
    Compact,
    Rebuild {
        corpus: Vec<(i64, String, String, String)>,
    },
}

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(WORDS), 1..4).prop_map(|ws| ws.join(" "))
}
fn source() -> impl Strategy<Value = String> {
    prop::sample::select(SOURCES).prop_map(String::from)
}
fn refid() -> impl Strategy<Value = String> {
    prop::sample::select(REFS).prop_map(String::from)
}
fn segs() -> impl Strategy<Value = Vec<(String, String)>> {
    prop::collection::vec((refid(), text()), 1..4)
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (1i64..=6, source(), segs()).prop_map(|(doc, source, segs)| Op::Insert { doc, source, segs }),
        2 => (1i64..=6).prop_map(|doc| Op::Remove { doc }),
        2 => Just(Op::Compact),
        1 => prop::collection::vec((1i64..=6, source(), refid(), text()), 0..6)
                .prop_map(|corpus| Op::Rebuild { corpus }),
    ]
}

/// Apply one op to both the index and the oracle, keeping them in lockstep.
fn apply(idx: &Idx, oracle: &mut Oracle, op: Op) {
    match op {
        Op::Insert { doc, source, segs } => {
            let pairs: Vec<(&str, &str)> =
                segs.iter().map(|(r, t)| (r.as_str(), t.as_str())).collect();
            idx.insert(doc, &source, &pairs).unwrap();
            // Upsert replaces the whole (doc, source) group.
            oracle.insert((doc, source), segs);
        }
        Op::Remove { doc } => {
            idx.remove(doc).unwrap();
            oracle.retain(|(d, _), _| *d != doc);
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
            let segments: Vec<Segment> = corpus
                .iter()
                .map(|(d, s, r, t)| Segment::new(*d, s, r, t))
                .collect();
            idx.rebuild(segments).unwrap();
            // Rebuild is verbatim: every corpus item is a segment under its pair.
            oracle.clear();
            for (d, s, r, t) in corpus {
                oracle.entry((d, s)).or_default().push((r, t));
            }
        }
    }
}

/// Is `m` a real segment currently in the oracle? (snapshot mode -> text matches.)
fn faithful(oracle: &Oracle, m: &trifle::Match) -> bool {
    oracle
        .get(&(m.doc_id, m.source.clone()))
        .is_some_and(|segs| {
            segs.iter()
                .any(|(r, t)| *r == m.ref_ && Some(t.as_str()) == m.text.as_deref())
        })
}

/// Check every invariant against the oracle.
fn check(idx: &Idx, oracle: &Oracle) {
    let expected_segments: u64 = oracle.values().map(|v| v.len() as u64).sum();
    assert_eq!(
        idx.stats().unwrap().segments,
        expected_segments,
        "segment count"
    );

    // Sample a few existing segments (deterministic BTreeMap order) and verify recall
    // + that every returned match is faithful and well-formed.
    for ((doc, _source), segs) in oracle.iter().take(4) {
        let Some((_, txt)) = segs.first() else {
            continue;
        };
        let hits = idx.search(txt, SearchOpts::new(100)).unwrap();
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
    #![proptest_config(ProptestConfig::with_cases(160))]

    #[test]
    fn thrashing_preserves_every_invariant(ops in prop::collection::vec(op_strategy(), 4..40)) {
        let h = Harness::new();
        let mut oracle = Oracle::new();
        for op in ops {
            apply(&h.index, &mut oracle, op);
            check(&h.index, &oracle);
        }
    }
}
