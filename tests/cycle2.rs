//! Cycle-2 audit regressions:
//! - C2-RA-1: `remove_segment` emptying a document must reap its `doc` row so its
//!   filterable payload can't leak into a later insert under the same key.
//! - Ranking: IDF-weighted overlap surfaces rare-gram matches; presence-not-frequency; the
//!   `D` weight-step knob; and a custom [`Ranker`] scoring from the public signals.

mod common;
use common::*;

use trifle::rank::{Candidates, QueryContext, Ranked, Ranker};
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

// ----- IDF-weighted overlap ranking -----

#[test]
fn idf_weighting_surfaces_the_rare_gram_match() {
    let h = Harness::new();
    // "commonword" is in many docs (high df); "rarezzq" in one (low df). Single-word docs, so
    // the query's space-boundary grams match neither — only the clean word grams overlap.
    for doc in 1..=20 {
        h.put(doc, "field", "f", "commonword");
    }
    h.put(50, "field", "f", "rarezzq");
    let q = "rarezzq commonword";

    // With weighting (default D=1), the rare match wins despite sharing FEWER grams than the
    // common docs — rarity, not raw overlap, decides.
    let weighted = h.search(q, SearchOpts::new(5)).unwrap();
    assert_eq!(
        weighted[0].key.as_i64(),
        Some(50),
        "IDF weighting surfaces the rare-gram match over higher-overlap common matches"
    );

    // Collapse the weighting (a huge D → every gram tier 1) and raw-overlap order returns:
    // a common doc (more shared grams, lowest id) now ranks first, burying the rare match.
    let flat = h
        .index
        .reader()
        .unwrap()
        .search(q, SearchOpts::new(5).weight_step(1e9))
        .unwrap();
    assert_ne!(
        flat[0].key.as_i64(),
        Some(50),
        "without weighting the rare match is buried by raw overlap"
    );
}

#[test]
fn overlap_counts_gram_presence_not_repetition() {
    let h = Harness::new();
    h.put(1, "field", "f", "wxqzv plain"); // the query term once
    h.put(2, "field", "f", "wxqzv wxqzv"); // the query term twice
    let hits = h.search("wxqzv", SearchOpts::new(10)).unwrap();
    // A posting records presence, not frequency, so repetition does not raise the score; the
    // two tie and fall back to insertion order.
    assert_eq!(
        ids(&hits),
        [1, 2],
        "repeating a term in a segment does not boost its overlap"
    );
}

#[test]
fn weight_step_d_widens_the_tiers() {
    // Same rare/common corpus; a large-but-finite D shrinks the rare gram's weight advantage
    // enough that the higher-overlap common doc overtakes it — the knob visibly moves ranking.
    let h = Harness::new();
    for doc in 1..=20 {
        h.put(doc, "field", "f", "commonword");
    }
    h.put(50, "field", "f", "rarezzq");
    let q = "rarezzq commonword";
    let sharp = h.search(q, SearchOpts::new(5)).unwrap(); // D = 1.0
    let blunt = h
        .index
        .reader()
        .unwrap()
        .search(q, SearchOpts::new(5).weight_step(50.0))
        .unwrap();
    assert_eq!(
        sharp[0].key.as_i64(),
        Some(50),
        "sharp D keeps the rare match on top"
    );
    assert_ne!(
        blunt[0].key.as_i64(),
        Some(50),
        "blunt D lets raw overlap win"
    );
}

// ----- a custom Ranker scores from the public signals (extension point) -----

/// A custom ranker that scores by summed inverse-document-frequency over the matched terms,
/// built ENTIRELY from the public surface — [`Candidate::matched_terms`] (per-term `df`) and
/// [`QueryContext::n_segments`] — proving a caller can compute a corpus-relative score
/// without any built-in BM25.
struct IdfSum;
impl Ranker for IdfSum {
    fn rank(&self, candidates: &Candidates<'_>, q: &QueryContext<'_>) -> Vec<Ranked> {
        let n = q.n_segments.max(1) as f64;
        let idf = |df: u64| {
            (1.0 + (n - df as f64 + 0.5) / (df as f64 + 0.5))
                .ln()
                .max(0.0)
        };
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .map(|c| {
                let s: f64 = c.matched_terms().map(|(_, df)| idf(df)).sum();
                (c.index(), s)
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
fn custom_ranker_can_score_from_public_signals() {
    let h = Harness::new();
    for doc in 1..=20 {
        h.put(doc, "field", "f", "commonword");
    }
    h.put(50, "field", "f", "rarezzq");
    // Over-fetch so the custom ranker sees the common docs alongside the rare one, then let it
    // score by summed idf from the public df signal.
    let hits = h
        .index
        .reader()
        .unwrap()
        .search(
            "rarezzq commonword",
            SearchOpts::new(5)
                .rerank(trifle::Effort::High)
                .ranker(&IdfSum),
        )
        .unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(50),
        "a custom idf ranker reads per-term df and ranks the rare match first"
    );
}
