//! Write lifecycle: upsert-replace, delete, monotonic ids, compaction, rebuild,
//! stats, and persistence across reopen.

mod common;
use common::*;
use trifle::store::Sidecar;
use trifle::tokenize::TrigramTokenizer;
use trifle::{Config, Index, SearchOpts};

fn finds(h: &Harness, q: &str, doc: i64) -> bool {
    hit(&h.index.search(q, SearchOpts::new(10)).unwrap(), doc)
}

#[test]
fn upsert_replaces_a_doc_sources_segments() {
    let h = Harness::new();
    h.put(1, "field", "f", "obsolete antiquated terminology");
    assert!(finds(&h, "obsolete antiquated", 1));
    h.put(1, "field", "f", "modern contemporary vocabulary");
    assert!(!finds(&h, "obsolete antiquated", 1), "old text gone");
    assert!(finds(&h, "modern contemporary", 1), "new text present");
}

#[test]
fn upsert_one_source_leaves_other_sources_intact() {
    let h = Harness::new();
    h.put(1, "field", "f", "alpha bravo charlie");
    h.put(1, "ocr", "scan", "delta echo foxtrot");
    // Replace only the "field" source.
    h.put(1, "field", "f", "golf hotel india");
    assert!(!finds(&h, "alpha bravo", 1));
    assert!(finds(&h, "golf hotel", 1));
    assert!(finds(&h, "delta echo", 1), "ocr source untouched");
}

#[test]
fn remove_deletes_every_source_of_a_doc() {
    let h = Harness::new();
    h.put(1, "field", "f", "alpha bravo charlie");
    h.put(1, "ocr", "scan", "delta echo foxtrot");
    h.index.remove(1).unwrap();
    assert!(!finds(&h, "alpha bravo", 1));
    assert!(!finds(&h, "delta echo", 1));
}

#[test]
fn removed_content_is_unfindable_before_any_compaction() {
    let h = Harness::new();
    h.put(1, "field", "f", "zugzwang quixotic jamboree");
    h.index.remove(1).unwrap();
    // No compact() called — the stale posting ids must still not surface (the seg
    // row is gone, so hydration drops them, and df was decremented).
    assert!(
        h.index
            .search("zugzwang quixotic", SearchOpts::new(10))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn freed_ids_are_not_reused_so_no_false_positives() {
    let h = Harness::new();
    h.put(1, "field", "f", "wxyzqj uniquetoken alpha");
    h.index.remove(1).unwrap();
    // A new doc gets a fresh monotonic id, never the freed one, so the lingering
    // "wxyzqj" posting entry can never come to mean this new segment.
    h.put(2, "field", "f", "entirely separate beta gamma delta");
    assert!(
        !finds(&h, "wxyzqj uniquetoken", 2),
        "no false positive on new doc"
    );
    assert!(
        finds(&h, "separate beta gamma", 2),
        "new doc itself is findable"
    );
}

#[test]
fn compact_preserves_results_and_clears_the_backlog() {
    let h = Harness::new();
    load_fixture(&h);
    h.index.remove(3).unwrap(); // delete one doc to create a fold-time purge
    let before = h.index.stats().unwrap();
    assert!(before.delta_backlog > 0, "writes leave a delta backlog");
    let stats = h.index.compact().unwrap();
    assert!(stats.tokens_folded > 0);
    let after = h.index.stats().unwrap();
    assert_eq!(after.delta_backlog, 0, "fold clears the backlog");
    // Search still correct after folding.
    assert!(finds(&h, "quick brown fox", 1));
    assert!(!finds(&h, "lorem ipsum dolor", 3), "deleted doc stays gone");
}

#[test]
fn search_works_across_base_and_delta_after_compact() {
    let h = Harness::new();
    h.put(1, "field", "f", "established baseline content");
    h.index.compact().unwrap(); // doc 1 now lives in the base
    h.put(2, "field", "f", "established baseline content"); // doc 2 lives in the delta
    let hits = h
        .index
        .search("established baseline", SearchOpts::new(10))
        .unwrap();
    assert!(
        hit(&hits, 1) && hit(&hits, 2),
        "base and delta both contribute"
    );
}

#[test]
fn rebuild_replaces_the_corpus_and_is_searchable() {
    let h = Harness::new();
    h.put(99, "field", "f", "stale preexisting data to be discarded");
    let corpus = FIXTURE
        .iter()
        .map(|(doc, text)| trifle::Segment::new(*doc, "field", "body", *text));
    h.index.rebuild(corpus).unwrap();
    assert!(
        !finds(&h, "stale preexisting", 99),
        "pre-rebuild data discarded"
    );
    assert!(
        finds(&h, "quick brown fox", 1),
        "rebuilt corpus is searchable"
    );
    assert_eq!(h.index.stats().unwrap().segments, FIXTURE.len() as u64);
}

#[test]
fn rebuild_reclaims_a_grown_id_space() {
    let h = Harness::new();
    // Churn to grow the monotonic id space.
    for i in 0..20 {
        h.put(
            1,
            "field",
            "f",
            &format!("revision number {i} of the document"),
        );
    }
    h.index
        .rebuild([trifle::Segment::new(
            1,
            "field",
            "body",
            "final canonical text",
        )])
        .unwrap();
    assert_eq!(h.index.stats().unwrap().segments, 1);
    assert!(finds(&h, "final canonical", 1));
}

#[test]
fn stats_report_segment_and_term_counts() {
    let h = Harness::new();
    assert_eq!(h.index.stats().unwrap().segments, 0);
    load_fixture(&h);
    let s = h.index.stats().unwrap();
    assert_eq!(s.segments, FIXTURE.len() as u64);
    assert!(s.terms > 0);
    assert!(s.disk_bytes > 0);
}

#[test]
fn data_survives_reopen_of_the_same_file() {
    let h = Harness::new();
    load_fixture(&h);
    let path = h.db_path();
    drop(h.index); // close the writer; keep the temp dir alive via `h.dir`
    let reopened: Index<TrigramTokenizer, Sidecar> =
        Index::open_at(&path, Config::default()).unwrap();
    let hits = reopened
        .search("quick brown fox", SearchOpts::new(10))
        .unwrap();
    assert!(hit(&hits, 1), "warm cache survives reopen");
    drop(h.dir);
}
