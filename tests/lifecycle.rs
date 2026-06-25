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
    // Fold first, so doc 3's ids live in the bases; only then does removing it give
    // the next fold real base ids to purge.
    h.index.compact().unwrap();
    h.index.remove(3).unwrap();
    let before = h.index.stats().unwrap();
    assert!(
        before.delta_backlog > 0,
        "the remove leaves a delta backlog"
    );
    let stats = h.index.compact().unwrap();
    assert!(stats.tokens_folded > 0);
    assert!(
        stats.ids_purged > 0,
        "the deleted doc's base ids are purged"
    );
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

#[test]
fn insert_batch_keeps_every_segment_of_a_doc_source_pair() {
    let h = Harness::new();
    // Two segments under the same (doc, source) must BOTH be content, not last-wins.
    h.index
        .insert_batch([
            trifle::Segment::new(1, "field", "front", "photosynthesis chlorophyll"),
            trifle::Segment::new(1, "field", "back", "cellular respiration mitochondria"),
        ])
        .unwrap();
    assert!(finds(&h, "photosynthesis chlorophyll", 1));
    assert!(finds(&h, "cellular respiration", 1));
}

#[test]
fn insert_batch_groups_nonadjacent_entries_of_the_same_pair() {
    let h = Harness::new();
    // The two (1,"field") segments are separated by a different group in iteration
    // order; grouping must still accumulate both.
    h.index
        .insert_batch([
            trifle::Segment::new(1, "field", "a", "alpha distinctive content"),
            trifle::Segment::new(2, "ocr", "z", "separate document entirely"),
            trifle::Segment::new(1, "field", "b", "omega distinctive content"),
        ])
        .unwrap();
    assert!(finds(&h, "alpha distinctive", 1));
    assert!(finds(&h, "omega distinctive", 1));
    assert!(finds(&h, "separate document", 2));
}

#[test]
fn insert_batch_replaces_only_the_groups_present() {
    let h = Harness::new();
    // Disjoint old/new vocabularies, so a stale hit can only mean the replace failed.
    h.put(1, "field", "f", "obsolete antiquated");
    h.put(2, "field", "f", "deprecated archaic");
    h.put(3, "field", "f", "untouched keepsake permanent");
    h.index
        .insert_batch([
            trifle::Segment::new(1, "field", "f", "renewed modernized"),
            trifle::Segment::new(2, "field", "f", "current freshest"),
        ])
        .unwrap();
    assert!(!finds(&h, "obsolete antiquated", 1) && !finds(&h, "deprecated archaic", 2));
    assert!(finds(&h, "renewed modernized", 1) && finds(&h, "current freshest", 2));
    assert!(finds(&h, "untouched keepsake", 3), "doc 3 untouched");
}

#[test]
fn remove_source_wipes_one_category_and_leaves_the_others() {
    let h = Harness::new();
    // Doc 1 has two categories; doc 2 shares the "ocr" source as a distractor.
    h.put(1, "ocr", "scan", "alpha ocr distinctive");
    h.put(1, "caption", "alt", "beta caption distinctive");
    h.put(2, "ocr", "scan", "gamma other document");

    h.index.remove_source(1, "ocr").unwrap();

    assert!(!finds(&h, "alpha ocr", 1), "doc 1 ocr category wiped");
    assert!(
        finds(&h, "beta caption", 1),
        "doc 1 caption category survives"
    );
    assert!(
        finds(&h, "gamma other", 2),
        "another doc's same source untouched"
    );
    assert_eq!(h.index.stats().unwrap().segments, 2);

    // Removing a (doc, source) pair with no segments is a no-op.
    h.index.remove_source(1, "ocr").unwrap();
    h.index.remove_source(99, "nope").unwrap();
    assert_eq!(h.index.stats().unwrap().segments, 2);
}

#[test]
fn rebuild_indexes_duplicate_doc_ids_verbatim() {
    let h = Harness::new();
    h.index
        .rebuild([
            trifle::Segment::new(1, "field", "a", "first cat segment"),
            trifle::Segment::new(1, "field", "b", "second dog segment"),
            trifle::Segment::new(1, "ocr", "x", "third bird segment"),
        ])
        .unwrap();
    // Rebuild is a verbatim reindex (no grouping/dedup) — all three segments land.
    assert_eq!(h.index.stats().unwrap().segments, 3);
    assert!(finds(&h, "first cat", 1) && finds(&h, "second dog", 1) && finds(&h, "third bird", 1));
}

#[test]
fn rebuild_on_an_empty_corpus_empties_and_stays_usable() {
    let h = Harness::new();
    load_fixture(&h);
    h.index.rebuild(std::iter::empty()).unwrap();
    assert_eq!(h.index.stats().unwrap().segments, 0);
    assert!(
        h.index
            .search("quick brown", SearchOpts::new(10))
            .unwrap()
            .is_empty()
    );
    // The swapped-in empty tables are functional: a fresh write + search works.
    h.put(1, "field", "f", "freshly inserted after empty rebuild");
    assert!(finds(&h, "freshly inserted", 1));
}

#[test]
fn rebuild_leaves_no_shadow_tables_behind() {
    let h = Harness::new();
    let corpus = FIXTURE
        .iter()
        .map(|(doc, text)| trifle::Segment::new(*doc, "field", "body", *text));
    h.index.rebuild(corpus).unwrap();
    // The atomic swap must drop/rename every shadow; none may linger.
    let path = h.db_path();
    drop(h.index);
    let raw = trifle::rusqlite::Connection::open(&path).unwrap();
    let leftover: i64 = raw
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE name LIKE '%\\_shadow' ESCAPE '\\'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(leftover, 0, "no *_shadow tables after a rebuild");
    drop(h.dir);
}

#[test]
fn a_rebuild_that_panics_mid_stream_leaves_the_live_index_intact() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let h = Harness::new();
    load_fixture(&h);
    // An iterator that yields a couple of segments then panics, simulating a source
    // that dies mid-pull. The shadow build is inside the rebuild transaction, so the
    // unwind must roll it back and leave the old index whole.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let corpus = std::iter::from_fn(move || {
            n += 1;
            match n {
                1 => Some(trifle::Segment::new(
                    100,
                    "field",
                    "f",
                    "doomed rebuild row",
                )),
                _ => panic!("source died mid-stream"),
            }
        });
        h.index.rebuild(corpus)
    }));
    assert!(result.is_err(), "the panic propagated");
    // The original corpus is still fully searchable; the doomed row never landed.
    assert!(finds(&h, "quick brown fox", 1));
    assert_eq!(h.index.stats().unwrap().segments, FIXTURE.len() as u64);
    assert!(!finds(&h, "doomed rebuild", 100));
}

#[test]
fn stats_segments_track_upsert_and_remove() {
    let h = Harness::new();
    h.put(1, "field", "f", "one segment here");
    assert_eq!(h.index.stats().unwrap().segments, 1);
    h.put(1, "field", "f", "replaced single segment"); // upsert: still 1
    assert_eq!(h.index.stats().unwrap().segments, 1);
    h.put(1, "ocr", "s", "a second source segment"); // +1
    assert_eq!(h.index.stats().unwrap().segments, 2);
    h.index.remove(1).unwrap(); // both gone
    assert_eq!(h.index.stats().unwrap().segments, 0);
}

#[test]
fn results_are_byte_identical_across_reopen_and_rebuild() {
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown fox jumps";
    let baseline = h.index.search(q, SearchOpts::new(10)).unwrap();

    // Reopen: the same file must produce the exact same ordered Vec<Match>.
    let path = h.db_path();
    drop(h.index);
    let reopened: Index<TrigramTokenizer, Sidecar> =
        Index::open_at(&path, Config::default()).unwrap();
    assert_eq!(reopened.search(q, SearchOpts::new(10)).unwrap(), baseline);

    // Rebuild reassigns dense ids, but ranking is by content, so the ordered doc-id
    // list must be unchanged (the stable tie-breaks do not depend on the ids).
    let corpus = FIXTURE
        .iter()
        .map(|(doc, text)| trifle::Segment::new(*doc, "field", "body", *text));
    reopened.rebuild(corpus).unwrap();
    let after = reopened.search(q, SearchOpts::new(10)).unwrap();
    assert_eq!(
        ids(&after),
        ids(&baseline),
        "ranking is stable across reindex"
    );
    drop(h.dir);
}
