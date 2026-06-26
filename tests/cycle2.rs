//! Cycle-2 audit regressions:
//! - C2-RA-1: `remove_segment` emptying a document must reap its `doc` row so its
//!   filterable payload can't leak into a later insert under the same key.
//! - D2: BM25+ length normalization ordering (short outranks long; binary tf).
//! - C2-A3: a custom [`Ranker`] can compute an idf-weighted score from the public API.

mod common;
use common::*;

use trifle::rank::{self, Candidates, QueryContext, Ranked, Ranker};
use trifle::rusqlite::Connection;
use trifle::rusqlite::types::Value;
use trifle::{Config, Document, Filter, FilterType, Schema, SearchOpts};

fn schema_with_deck() -> Schema {
    Schema::chunked()
        .text("body")
        .filterable("deck", FilterType::Int)
        .build()
        .unwrap()
}

/// How many `doc` rows carry `deck = deck` (reads the raw file, bypassing search).
fn doc_rows_with_deck(path: &std::path::Path, deck: i64) -> i64 {
    let c = Connection::open(path).unwrap();
    c.query_row("SELECT count(*) FROM doc WHERE deck = ?1", [deck], |r| {
        r.get(0)
    })
    .unwrap()
}

// ----- C2-RA-1: remove_segment-to-empty must not orphan the doc row / payload -----

#[test]
fn remove_segment_to_empty_reaps_the_doc_row_and_payload() {
    let h = Harness::with_schema(schema_with_deck(), Config::default());
    let path = h.db_path();
    {
        let mut w = h.index.writer().unwrap();
        w.insert_document(
            Document::new(5, vec![("body".into(), "alpha bravo charlie".into())])
                .with_payload(vec![("deck".into(), Value::Integer(3))]),
        )
        .unwrap();
        w.commit().unwrap();
    }
    assert_eq!(doc_rows_with_deck(&path, 3), 1, "payload was written");

    // Removing the document's only segment must reap the now-empty doc row.
    h.remove_segment(5, "body");
    assert_eq!(h.index.stats().unwrap().segments, 0);
    assert_eq!(
        doc_rows_with_deck(&path, 3),
        0,
        "the emptied doc row (and its filterable payload) is gone, not orphaned"
    );

    // A fresh logical document under the same key that declares NO deck must not inherit deck=3.
    {
        let mut w = h.index.writer().unwrap();
        w.insert(5, &[("body", "delta echo foxtrot")]).unwrap();
        w.commit().unwrap();
    }
    let filtered = h
        .search(
            "delta echo",
            SearchOpts::new(10).filter(&Filter::eq("deck", 3i64)),
        )
        .unwrap();
    assert!(!hit(&filtered, 5), "no stale deck=3 leak on reinsert");
    // ...yet it is findable without the filter.
    assert!(hit(
        &h.search("delta echo", SearchOpts::new(10)).unwrap(),
        5
    ));
}

/// Set up a doc with a deck=3 payload, then empty it via `remove_segment`; assert that a
/// `reinsert` declaring no deck does not inherit the old payload.
fn assert_reinsert_is_clean(reinsert: impl Fn(&Harness)) {
    let h = Harness::with_schema(schema_with_deck(), Config::default());
    {
        let mut w = h.index.writer().unwrap();
        w.insert_document(
            Document::new(5, vec![("body".into(), "alpha bravo charlie".into())])
                .with_payload(vec![("deck".into(), Value::Integer(3))]),
        )
        .unwrap();
        w.commit().unwrap();
    }
    h.remove_segment(5, "body");
    reinsert(&h);
    let filtered = h
        .search(
            "delta echo",
            SearchOpts::new(10).filter(&Filter::eq("deck", 3i64)),
        )
        .unwrap();
    assert!(!hit(&filtered, 5), "stale deck=3 leaked on reinsert");
    assert!(hit(
        &h.search("delta echo", SearchOpts::new(10)).unwrap(),
        5
    ));
}

#[test]
fn no_reinsert_api_inherits_a_reaped_docs_payload() {
    // C2-RA-1 leaked via insert, upsert, AND insert_document(no payload); the reap fix closes
    // all three, since no orphan doc row survives for any of them to reuse.
    assert_reinsert_is_clean(|h| {
        let mut w = h.index.writer().unwrap();
        w.insert(5, &[("body", "delta echo foxtrot")]).unwrap();
        w.commit().unwrap();
    });
    assert_reinsert_is_clean(|h| {
        let mut w = h.index.writer().unwrap();
        w.upsert(5, &[("body", "delta echo foxtrot")]).unwrap();
        w.commit().unwrap();
    });
    assert_reinsert_is_clean(|h| {
        let mut w = h.index.writer().unwrap();
        w.insert_document(Document::new(
            5,
            vec![("body".into(), "delta echo foxtrot".into())],
        ))
        .unwrap();
        w.commit().unwrap();
    });
}

#[test]
fn remove_segment_to_empty_matches_whole_remove() {
    // remove_segment of the last segment and remove(key) must leave the same residue.
    let mk = || {
        let h = Harness::with_schema(schema_with_deck(), Config::default());
        {
            let mut w = h.index.writer().unwrap();
            w.insert_document(
                Document::new(7, vec![("body".into(), "shared content here".into())])
                    .with_payload(vec![("deck".into(), Value::Integer(9))]),
            )
            .unwrap();
            w.commit().unwrap();
        }
        h
    };
    let a = mk();
    a.remove_segment(7, "body");
    let b = mk();
    b.remove(7);
    assert_eq!(doc_rows_with_deck(&a.db_path(), 9), 0);
    assert_eq!(doc_rows_with_deck(&b.db_path(), 9), 0);
    assert_eq!(
        a.index.stats().unwrap().segments,
        b.index.stats().unwrap().segments
    );
}

#[test]
fn removing_one_of_several_segments_keeps_the_doc_row() {
    // Reaping happens only on the LAST segment — a still-populated doc keeps its row + payload.
    let schema = Schema::chunked()
        .text("front")
        .text("back")
        .filterable("deck", FilterType::Int)
        .build()
        .unwrap();
    let h = Harness::with_schema(schema, Config::default());
    let path = h.db_path();
    {
        let mut w = h.index.writer().unwrap();
        w.insert_document(
            Document::new(
                1,
                vec![
                    ("front".into(), "alpha bravo".into()),
                    ("back".into(), "charlie delta".into()),
                ],
            )
            .with_payload(vec![("deck".into(), Value::Integer(4))]),
        )
        .unwrap();
        w.commit().unwrap();
    }
    h.remove_segment(1, "front");
    assert_eq!(
        doc_rows_with_deck(&path, 4),
        1,
        "the doc row survives while a segment remains"
    );
    assert!(hit(
        &h.search(
            "charlie delta",
            SearchOpts::new(10).filter(&Filter::eq("deck", 4i64))
        )
        .unwrap(),
        1
    ));
}

// ----- D2: BM25+ length-normalization ordering -----

#[test]
fn bm25_length_normalization_favors_the_short_segment() {
    let h = Harness::new();
    // Long doc inserted FIRST (lower internal id => would win a tie); short doc last.
    h.put(
        1,
        "field",
        "f",
        "zephyrqual aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn",
    );
    h.put(2, "field", "f", "zephyrqual");
    let hits = h.search("zephyrqual", SearchOpts::new(10)).unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(2),
        "the short verbatim segment ranks first under length normalization"
    );
}

#[test]
fn bm25_tf_is_binary_repetition_does_not_boost() {
    let h = Harness::new();
    h.put(1, "field", "f", "wxqzv plain"); // term once + filler
    h.put(2, "field", "f", "wxqzv wxqzv"); // term twice, equal length
    let hits = h.search("wxqzv", SearchOpts::new(10)).unwrap();
    assert_eq!(
        ids(&hits),
        [1, 2],
        "binary tf: a repeated term does not outrank a single occurrence"
    );
}

// ----- C2-A3: a custom Ranker can compute idf from the public API -----

/// A BM25+ ranker built ENTIRELY from the public `Candidate`/`QueryContext` surface —
/// [`Candidate::matched_terms`] (per-term `df`), [`Candidate::seg_len`] (`|d|`),
/// [`rank::idf`], and [`QueryContext::n_segments`]/[`QueryContext::avgdl`] — proving the API
/// exposes everything a precision-tier ranker needs (it previously could not see `df`).
struct CustomBm25;
impl Ranker for CustomBm25 {
    fn rank(&self, candidates: &Candidates<'_>, q: &QueryContext<'_>) -> Vec<Ranked> {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        const DELTA: f64 = 1.0;
        let n = q.n_segments.max(1);
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .map(|c| {
                let dl = (c.seg_len() as f64).max(1.0);
                let norm = if q.avgdl > 0.0 {
                    1.0 - B + B * dl / q.avgdl
                } else {
                    1.0
                };
                let tf_component = (K1 + 1.0) / (1.0 + K1 * norm) + DELTA;
                let idf_sum: f64 = c.matched_terms().map(|(_, df)| rank::idf(df, n)).sum();
                (c.index(), idf_sum * tf_component)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored
            .into_iter()
            .map(|(candidate, _)| Ranked { candidate })
            .collect()
    }
}

#[test]
fn custom_ranker_reproduces_bm25_ordering_from_public_api() {
    let h = Harness::new();
    h.put(
        1,
        "field",
        "f",
        "zephyrqual aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn",
    );
    h.put(2, "field", "f", "zephyrqual");
    // A user-supplied ranker, using only the public idf inputs, recovers the short segment —
    // the same result the built-in Bm25Ranker produces.
    let custom = h
        .index
        .reader()
        .unwrap()
        .search("zephyrqual", SearchOpts::new(10).ranker(&CustomBm25))
        .unwrap();
    let builtin = h.search("zephyrqual", SearchOpts::new(10)).unwrap();
    assert_eq!(custom[0].key.as_i64(), Some(2));
    assert_eq!(
        ids(&custom),
        ids(&builtin),
        "custom BM25 matches the built-in"
    );
}
