//! Write lifecycle: upsert-replace, delete, monotonic ids, compaction, rebuild,
//! stats, and persistence across reopen.

mod common;
use common::*;
use trifle::store::Sidecar;
use trifle::tokenize::DefaultTokenizer;
use trifle::{Config, Document, Index, Schema, SearchOpts};

fn finds(h: &Harness, q: &str, doc: i64) -> bool {
    hit(&h.search(q, SearchOpts::new(10)).unwrap(), doc)
}

/// A one-segment `Document` under label `"body"`.
fn doc(key: i64, label: &str, text: &str) -> Document {
    Document::new(key, vec![(label.to_string(), text.to_string())])
}

#[test]
fn upsert_replaces_a_segment() {
    let h = Harness::new();
    h.put(1, "field", "f", "obsolete antiquated terminology");
    assert!(finds(&h, "obsolete antiquated", 1));
    h.put(1, "field", "f", "modern contemporary vocabulary");
    assert!(!finds(&h, "obsolete antiquated", 1), "old text gone");
    assert!(finds(&h, "modern contemporary", 1), "new text present");
}

#[test]
fn upsert_one_label_leaves_other_labels_intact() {
    let h = Harness::new();
    h.put(1, "field", "f", "alpha bravo charlie");
    h.put(1, "ocr", "scan", "delta echo foxtrot");
    // Replace only the "f" label.
    h.put(1, "field", "f", "golf hotel india");
    assert!(!finds(&h, "alpha bravo", 1));
    assert!(finds(&h, "golf hotel", 1));
    assert!(finds(&h, "delta echo", 1), "the other label is untouched");
}

#[test]
fn remove_deletes_every_label_of_a_doc() {
    let h = Harness::new();
    h.put(1, "field", "f", "alpha bravo charlie");
    h.put(1, "ocr", "scan", "delta echo foxtrot");
    h.remove(1);
    assert!(!finds(&h, "alpha bravo", 1));
    assert!(!finds(&h, "delta echo", 1));
}

#[test]
fn removed_content_is_unfindable_before_any_compaction() {
    let h = Harness::new();
    h.put(1, "field", "f", "zugzwang quixotic jamboree");
    h.remove(1);
    // No compact() called — the stale posting ids must still not surface (the seg row is
    // gone, so hydration drops them, and df was decremented).
    assert!(
        h.search("zugzwang quixotic", SearchOpts::new(10))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn freed_ids_are_not_reused_so_no_false_positives() {
    let h = Harness::new();
    h.put(1, "field", "f", "wxyzqj uniquetoken alpha");
    h.remove(1);
    // A new doc gets a fresh monotonic id, never the freed one, so the lingering "wxyzqj"
    // posting entry can never come to mean this new segment.
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
    // Fold first, so doc 3's ids live in the bases; only then does removing it give the
    // next fold real base ids to purge.
    h.index.compact().unwrap();
    h.remove(3);
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
    h.index.rebuild(fixture_docs()).unwrap();
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
        .rebuild([doc(1, "body", "final canonical text")])
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
    let reopened: Index<DefaultTokenizer, Sidecar> =
        Index::open_at(&path, Schema::flat(), Config::default()).unwrap();
    let hits = reopened
        .reader()
        .unwrap()
        .search("quick brown fox", SearchOpts::new(10))
        .unwrap();
    assert!(hit(&hits, 1), "warm cache survives reopen");
    drop(h.dir);
}

#[test]
fn insert_keeps_every_segment_of_a_doc() {
    let h = Harness::new();
    // Two segments (labels) of the same doc must BOTH be content, not last-wins.
    h.insert(
        1,
        &[
            ("front", "photosynthesis chlorophyll"),
            ("back", "cellular respiration mitochondria"),
        ],
    );
    assert!(finds(&h, "photosynthesis chlorophyll", 1));
    assert!(finds(&h, "cellular respiration", 1));
}

#[test]
fn writer_batch_accumulates_interleaved_docs() {
    let h = Harness::new();
    // Doc 1's two labels straddle a different doc in the same writer batch; the writer must
    // accumulate both segments of doc 1 and the unrelated doc 2.
    let mut w = h.index.writer().unwrap();
    w.insert(1, &[("a", "alpha distinctive content")]).unwrap();
    w.insert(2, &[("z", "separate document entirely")]).unwrap();
    w.insert_segment(1, "b", "omega distinctive content")
        .unwrap();
    w.commit().unwrap();
    assert!(finds(&h, "alpha distinctive", 1));
    assert!(finds(&h, "omega distinctive", 1));
    assert!(finds(&h, "separate document", 2));
}

#[test]
fn upsert_replaces_only_the_named_docs() {
    let h = Harness::new();
    // Disjoint old/new vocabularies, so a stale hit can only mean the replace failed.
    h.put(1, "field", "f", "obsolete antiquated");
    h.put(2, "field", "f", "deprecated archaic");
    h.put(3, "field", "f", "untouched keepsake permanent");
    let mut w = h.index.writer().unwrap();
    w.upsert(1, &[("f", "renewed modernized")]).unwrap();
    w.upsert(2, &[("f", "current freshest")]).unwrap();
    w.commit().unwrap();
    assert!(!finds(&h, "obsolete antiquated", 1) && !finds(&h, "deprecated archaic", 2));
    assert!(finds(&h, "renewed modernized", 1) && finds(&h, "current freshest", 2));
    assert!(finds(&h, "untouched keepsake", 3), "doc 3 untouched");
}

#[test]
fn remove_segment_wipes_one_label_and_leaves_the_others() {
    let h = Harness::new();
    // Doc 1 has two labels; doc 2 shares the "scan" label as a distractor.
    h.put(1, "ocr", "scan", "alpha ocr distinctive");
    h.put(1, "caption", "alt", "beta caption distinctive");
    h.put(2, "ocr", "scan", "gamma other document");

    h.remove_segment(1, "scan");

    assert!(!finds(&h, "alpha ocr", 1), "doc 1 scan label wiped");
    assert!(finds(&h, "beta caption", 1), "doc 1 alt label survives");
    assert!(
        finds(&h, "gamma other", 2),
        "another doc's same label untouched"
    );
    assert_eq!(h.index.stats().unwrap().segments, 2);

    // Removing a (doc, label) pair with no segment is a no-op.
    h.remove_segment(1, "scan");
    h.remove_segment(99, "nope");
    assert_eq!(h.index.stats().unwrap().segments, 2);
}

#[test]
fn rebuild_indexes_all_labels_of_a_doc() {
    let h = Harness::new();
    h.index
        .rebuild([Document::new(
            1,
            vec![
                ("a".to_string(), "first cat segment".to_string()),
                ("b".to_string(), "second dog segment".to_string()),
                ("x".to_string(), "third bird segment".to_string()),
            ],
        )])
        .unwrap();
    // All three labels of the doc land as distinct segments.
    assert_eq!(h.index.stats().unwrap().segments, 3);
    assert!(finds(&h, "first cat", 1) && finds(&h, "second dog", 1) && finds(&h, "third bird", 1));
}

#[test]
fn rebuild_on_an_empty_corpus_empties_and_stays_usable() {
    let h = Harness::new();
    load_fixture(&h);
    h.index.rebuild(std::iter::empty::<Document>()).unwrap();
    assert_eq!(h.index.stats().unwrap().segments, 0);
    assert!(
        h.search("quick brown", SearchOpts::new(10))
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
    h.index.rebuild(fixture_docs()).unwrap();
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
    // An iterator that yields a document then panics, simulating a source that dies
    // mid-pull. The shadow build is inside the rebuild transaction, so the unwind must roll
    // it back and leave the old index whole.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut n = 0;
        let corpus = std::iter::from_fn(move || {
            n += 1;
            match n {
                1 => Some(doc(100, "body", "doomed rebuild row")),
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
    h.put(1, "ocr", "s", "a second label segment"); // +1
    assert_eq!(h.index.stats().unwrap().segments, 2);
    h.remove(1); // both gone
    assert_eq!(h.index.stats().unwrap().segments, 0);
}

#[test]
fn results_are_byte_identical_across_reopen_and_rebuild() {
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown fox jumps";
    let baseline = h.search(q, SearchOpts::new(10)).unwrap();

    // Reopen: the same file must produce the exact same ordered Vec<Match>.
    let path = h.db_path();
    drop(h.index);
    let reopened: Index<DefaultTokenizer, Sidecar> =
        Index::open_at(&path, Schema::flat(), Config::default()).unwrap();
    assert_eq!(
        reopened
            .reader()
            .unwrap()
            .search(q, SearchOpts::new(10))
            .unwrap(),
        baseline
    );

    // Rebuild reassigns dense ids, but ranking is by content, so the ordered doc-id list
    // must be unchanged (the stable tie-breaks do not depend on the ids).
    reopened.rebuild(fixture_docs()).unwrap();
    let after = reopened
        .reader()
        .unwrap()
        .search(q, SearchOpts::new(10))
        .unwrap();
    assert_eq!(
        ids(&after),
        ids(&baseline),
        "ranking is stable across reindex"
    );
    drop(h.dir);
}
