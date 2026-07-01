//! The streaming spine: lazy [`CandidateStream`], choose-then-hydrate, the per-candidate term
//! introspection, and the `batch == serial` invariant.

mod common;
use common::*;

use trifle::{Result, SearchOpts};

#[test]
fn candidates_are_provenance_only_then_hydrated() {
    let h = Harness::new();
    load_fixture(&h);
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();
    let mut stream = reader.candidates("quick brown fox", &opts).unwrap();

    // Pull a pool of candidates (propagating errors — never filter_map(Result::ok)).
    let pool: Vec<_> = stream
        .by_ref()
        .take(10)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert!(!pool.is_empty(), "the query matches fixture docs");
    // Candidates carry provenance + score components + the corrected ranking key, no text. Since
    // v0.4 M3 dropped the engine's `≥ 1` weight clamp, `score()` (the bit-sliced energy `E_acc`)
    // can fall below `overlap()` (a candidate matching only weight-0 commons has `score() == 0`),
    // so the old `score() ≥ overlap()` no longer holds. The stream now orders by the finite
    // corrected float, descending.
    assert!(pool.iter().all(|c| c.corrected_score().is_finite()));
    assert!(
        pool.windows(2)
            .all(|w| w[0].corrected_score() >= w[1].corrected_score()),
        "the stream yields candidates in corrected-score descending order"
    );

    // Hydrate only the top 2 (choose-then-hydrate).
    let keep: Vec<_> = pool.into_iter().take(2).collect();
    let hits = stream.hydrate(&keep).unwrap();
    assert_eq!(hits.len(), keep.len());
    assert!(hits.iter().all(|m| !m.text.is_empty()));
    assert_eq!(
        ids(&hits),
        keep.iter()
            .map(|c| c.key().as_i64().unwrap())
            .collect::<Vec<_>>(),
        "hydrate preserves the chosen order"
    );
}

#[test]
fn corpus_signals_and_term_introspection() {
    let h = Harness::new();
    load_fixture(&h);
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();
    let mut stream = reader.candidates("quick brown", &opts).unwrap();

    assert_eq!(stream.n_segments(), FIXTURE.len() as u64);
    assert!(stream.avgdl() > 0.0);
    // present_terms are the selected tokens that have a posting, each with its df.
    let present: Vec<(String, u64)> = stream
        .present_terms()
        .map(|(t, df)| (t.to_string(), df))
        .collect();
    assert!(!present.is_empty(), "the query has present trigrams");
    assert!(present.iter().all(|(_, df)| *df > 0));

    // The first candidate's matched_terms is a non-empty subset of present_terms.
    let first = stream.next().unwrap().unwrap();
    let matched: Vec<(String, u64)> = stream
        .matched_terms(&first)
        .map(|(t, df)| (t.to_string(), df))
        .collect();
    assert!(!matched.is_empty(), "a candidate shares ≥1 selected term");
    assert!(matched.iter().all(|m| present.contains(m)));
}

#[test]
fn collect_matches_is_the_eager_equivalent() {
    let h = Harness::new();
    load_fixture(&h);
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();

    let streamed = reader
        .candidates("quick brown fox", &opts)
        .unwrap()
        .collect_matches(5)
        .unwrap();
    let eager = reader.matches("quick brown fox", &opts, 5).unwrap();
    assert_eq!(streamed, eager, "collect_matches == matches");
}

#[test]
fn batch_equals_serial() {
    let h = Harness::new();
    load_fixture(&h);
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();

    let queries = ["quick brown fox", "lazy dog sleeps", "wizards jump"];
    let batch = reader.matches_batch(&queries, &opts, 10).unwrap();
    for (i, q) in queries.iter().enumerate() {
        let serial = reader.matches(q, &opts, 10).unwrap();
        assert_eq!(
            batch[i], serial,
            "query {q:?} ranks identically batch vs serial"
        );
    }
}

#[test]
fn a_stream_can_pull_then_filter_in_caller_code() {
    // The streaming model replaces the old `scope` closure: filter the candidate stream on the
    // caller's own predicate over key/label, then hydrate the survivors.
    let h = Harness::new();
    load_fixture(&h);
    let reader = h.index.reader().unwrap();
    let opts = SearchOpts::new();
    let mut stream = reader.candidates("quick", &opts).unwrap();

    let kept: Vec<_> = stream
        .by_ref()
        .filter(|c| {
            c.as_ref()
                .map(|c| c.key().as_i64() == Some(7))
                .unwrap_or(true)
        })
        .take(5)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let hits = stream.hydrate(&kept).unwrap();
    assert!(hits.iter().all(|m| m.key.as_i64() == Some(7)));
}
