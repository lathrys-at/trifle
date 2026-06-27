//! Cycle-2 audit regressions:
//! - C2-RA-1: `remove_segment` emptying a document must reap its `doc` row so its
//!   filterable payload can't leak into a later insert under the same key.
//! - Ranking: IDF-weighted overlap surfaces rare-gram matches; presence-not-frequency; the
//!   `D` weight-step knob; a custom [`Ranker`] scoring from the public signals; and the
//!   corpus-derived `D` hint surfaced through `stats()`.

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

// T5 (I12): the Tier-2 filter is now applied scoped to candidate ids, not by materializing
// every matching doc id in the corpus. The correctness contract is unchanged: a non-selective
// filter (one the whole result set satisfies) must not alter the results or their order, and a
// selective filter still narrows.
#[test]
fn scoped_filter_matches_the_unfiltered_result_set() {
    let h = Harness::with_schema(schema_with_deck(), Config::default());
    {
        let mut w = h.index.writer().unwrap();
        for d in 1..=12 {
            w.insert_document(
                Document::new(d, vec![("body".into(), "alpha bravo charlie".into())])
                    .with_payload(vec![("deck".into(), Value::Integer(7))]),
            )
            .unwrap();
        }
        w.commit().unwrap();
    }

    // Every doc has deck=7, so a deck=7 filter spans the whole corpus: the scoped filter must
    // return exactly the unfiltered result set, in the same order.
    let unfiltered = ids(&h.search("alpha bravo", SearchOpts::new(20)).unwrap());
    let broad = ids(&h
        .search(
            "alpha bravo",
            SearchOpts::new(20).filter(&Filter::eq("deck", 7i64)),
        )
        .unwrap());
    assert_eq!(broad.len(), 12, "all twelve docs match the broad filter");
    assert_eq!(
        unfiltered, broad,
        "a non-selective filter must not change the result set or order"
    );

    // A selective filter still narrows; a filter matching nothing returns nothing.
    assert!(
        h.search(
            "alpha bravo",
            SearchOpts::new(20).filter(&Filter::eq("deck", 99i64))
        )
        .unwrap()
        .is_empty(),
        "a filter matching no doc returns nothing"
    );
}

// T3 (A4): set_fields requires an existing document — it must NOT create a payload-only
// "ghost" doc row that no search can return, and must not resurrect a doc the C2-RA-1 reaping
// path removed.
#[test]
fn set_fields_requires_an_existing_document() {
    let h = Harness::with_schema(schema_with_deck(), Config::default());
    let path = h.db_path();

    // A key that was never inserted is rejected; no ghost row is staged.
    {
        let mut w = h.index.writer().unwrap();
        let err = w
            .set_fields(404, &[("deck", Value::Integer(1))])
            .unwrap_err();
        assert!(
            matches!(err, trifle::Error::InvalidInput(_)),
            "a fresh key must be rejected, got {err:?}"
        );
        w.commit().unwrap();
    }
    assert_eq!(
        doc_rows_with_deck(&path, 1),
        0,
        "no ghost doc row was created"
    );

    // On a real document (with a segment) set_fields succeeds and the payload is filterable.
    {
        let mut w = h.index.writer().unwrap();
        w.insert(7, &[("body", "alpha bravo charlie")]).unwrap();
        w.set_fields(7, &[("deck", Value::Integer(2))]).unwrap();
        w.commit().unwrap();
    }
    assert!(hit(
        &h.search(
            "alpha bravo",
            SearchOpts::new(10).filter(&Filter::eq("deck", 2i64))
        )
        .unwrap(),
        7
    ));

    // C2-RA-1 non-interaction: emptying the doc reaps its row, after which set_fields errors
    // again rather than recreating a ghost.
    h.remove_segment(7, "body");
    {
        let mut w = h.index.writer().unwrap();
        let err = w.set_fields(7, &[("deck", Value::Integer(3))]).unwrap_err();
        assert!(
            matches!(err, trifle::Error::InvalidInput(_)),
            "a reaped doc must not be resurrected by set_fields, got {err:?}"
        );
        w.commit().unwrap();
    }
    assert_eq!(
        doc_rows_with_deck(&path, 3),
        0,
        "no ghost row after the reap"
    );
}

// F1 (audit): empty-segment writes must NOT create a ghost doc row — the §8 fold-of-empty is a
// no-op — and must not back-door the T3 set_fields guard. insert_document with payload-but-no-
// segments is rejected, mirroring T3. (The insert/upsert family was the sibling path T3 missed.)
#[test]
fn empty_segment_writes_create_no_ghost_doc_row() {
    let h = Harness::with_schema(schema_with_deck(), Config::default());
    let path = h.db_path();
    let doc_rows = |p: &std::path::Path| -> i64 {
        Connection::open(p)
            .unwrap()
            .query_row("SELECT count(*) FROM doc", [], |r| r.get(0))
            .unwrap()
    };

    // (a) empty insert / upsert / insert_document(no payload) are no-ops — no doc row created.
    {
        let mut w = h.index.writer().unwrap();
        w.insert(1, &[]).unwrap();
        w.upsert(2, &[]).unwrap();
        w.insert_document(Document::new(3, vec![])).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(
        doc_rows(&path),
        0,
        "empty-segment writes created ghost doc rows"
    );

    // (b) insert_document with payload but no segments is rejected (mirrors T3).
    {
        let mut w = h.index.writer().unwrap();
        let err = w
            .insert_document(
                Document::new(4, vec![]).with_payload(vec![("deck".into(), Value::Integer(7))]),
            )
            .unwrap_err();
        assert!(
            matches!(err, trifle::Error::InvalidInput(_)),
            "a payload-only document must be rejected, got {err:?}"
        );
        w.commit().unwrap();
    }
    assert_eq!(
        doc_rows(&path),
        0,
        "a rejected payload-only doc left a ghost row"
    );

    // (c) the empty-insert back door around the T3 set_fields guard is closed.
    {
        let mut w = h.index.writer().unwrap();
        w.insert(5, &[]).unwrap();
        let err = w.set_fields(5, &[("deck", Value::Integer(9))]).unwrap_err();
        assert!(
            matches!(err, trifle::Error::InvalidInput(_)),
            "empty insert + set_fields bypassed the T3 guard, got {err:?}"
        );
        w.commit().unwrap();
    }
    assert_eq!(doc_rows(&path), 0);

    // (d) a real segment still works (sanity).
    {
        let mut w = h.index.writer().unwrap();
        w.insert(6, &[("body", "alpha bravo charlie")]).unwrap();
        w.commit().unwrap();
    }
    assert_eq!(doc_rows(&path), 1);
    assert!(hit(
        &h.search("alpha bravo", SearchOpts::new(10)).unwrap(),
        6
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

// ----- the weight-step (D) hint surfaced through stats() -----

#[test]
fn stats_suggests_a_weight_step_from_observed_band_spreads() {
    let h = Harness::new();
    for doc in 1..=20 {
        h.put(doc, "field", "f", "commonword"); // df 20 for its grams
    }
    h.put(50, "field", "f", "rarezzq"); // df 1 for its grams

    // No searches yet → no hint to give.
    assert!(h.index.stats().unwrap().weight_step_hint.is_none());

    // Run searches whose survivors span a wide band (df 20 vs df 1 ≈ 4.3 doublings).
    for _ in 0..5 {
        let _ = h.search("rarezzq commonword", SearchOpts::new(5)).unwrap();
    }
    let hint = h
        .index
        .stats()
        .unwrap()
        .weight_step_hint
        .expect("a hint after some searches");
    assert_eq!(hint.samples, 5, "one band-spread sample per search");
    assert!(
        hint.median_spread > 3.0 && hint.median_spread < 6.0,
        "median band-spread ≈ log2(20) doublings, got {}",
        hint.median_spread
    );
    // suggested = max(0.5, median/3).
    assert!((hint.suggested - (hint.median_spread / 3.0).max(0.5)).abs() < 1e-9);
    assert!(
        hint.iqr.0 <= hint.median_spread && hint.median_spread <= hint.iqr.1,
        "median lies within the IQR"
    );
}

// T15: a rebuild can change the df distribution, so it must drop the accumulated band-spread
// samples — the hint then reflects only the new corpus. A `compact` (a df-preserving fold)
// must NOT discard them.
#[test]
fn rebuild_resets_the_weight_step_hint() {
    let h = Harness::new();
    for doc in 1..=20 {
        h.put(doc, "field", "f", "commonword");
    }
    h.put(50, "field", "f", "rarezzq");
    for _ in 0..5 {
        let _ = h.search("rarezzq commonword", SearchOpts::new(5)).unwrap();
    }
    assert!(
        h.index.stats().unwrap().weight_step_hint.is_some(),
        "the searches built a hint"
    );

    // A compact folds deltas into bases but leaves every df unchanged → samples stay valid.
    h.index.compact().unwrap();
    assert!(
        h.index.stats().unwrap().weight_step_hint.is_some(),
        "compact must NOT discard band-spread samples"
    );

    // A rebuild reassigns the corpus and can shift the df distribution → samples are dropped.
    h.index
        .rebuild((1..=20).map(|d| Document::new(d, vec![("f".into(), "commonword".into())])))
        .unwrap();
    assert!(
        h.index.stats().unwrap().weight_step_hint.is_none(),
        "rebuild zeroes the band-spread histogram; no post-rebuild searches yet"
    );
}
