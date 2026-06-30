//! The opt-in raw-SQL [`SqlFilter`] over the caller's live data: the universal `key IN rarray(?)`
//! mode, anonymous `?` placeholders, numbered `?1` reuse, and the fragment-first binding that
//! keeps caller placeholders from colliding with the candidate-scope param.

mod common;
use common::*;

use std::rc::Rc;
use trifle::rusqlite::ToSql;
use trifle::rusqlite::types::Value;
use trifle::{SearchOpts, SqlFilter};

/// The fixture corpus, restricted by the given filter.
fn fixture() -> Harness {
    let h = Harness::new();
    load_fixture(&h);
    h
}

/// Load `(id, owned-text)` docs under label `"f"` in one writer batch.
fn load_owned(h: &Harness, docs: &[(i64, String)]) {
    let mut w = h.index.writer().unwrap();
    for (id, text) in docs {
        w.upsert(*id, &[("f", text.as_str())]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn filter_excludes_even_the_top_float_candidate() {
    // The eager path scores the candidate union via `score_union` (M3), which routes every chunk —
    // both the walk and the §7 count-only recovery (these commons-only docs are count-only
    // candidates) — through `prov.lookup`, folding in the SqlFilter. So a filter that excludes the
    // HIGHEST-FLOAT candidate must still remove it from the eager `matches` results (the filter is
    // not bypassed by the full-drain + float-rank path, nor by count-only recovery).
    let h = Harness::new();
    let mut docs: Vec<(i64, String)> = Vec::new();
    for i in 1..=12 {
        docs.push((i, "kqxvz report".to_string())); // rare kqxvz (high energy + credit) + common report
    }
    for i in 13..=60 {
        docs.push((i, "report only".to_string())); // commons-only, lower float
    }
    load_owned(&h, &docs);

    // Sanity: unfiltered, a kqxvz doc carries the highest float and is present.
    let unfiltered = ids(&h
        .search_opts("kqxvz report", &SearchOpts::new().min_shared(1), 60)
        .unwrap());
    assert!(
        unfiltered.contains(&1),
        "sanity: top-float doc present unfiltered: {unfiltered:?}"
    );

    // Filter to {13, 14} — both commons-only docs, NOT the high-float kqxvz docs.
    let allowed: Rc<Vec<Value>> = Rc::new(vec![Value::Integer(13), Value::Integer(14)]);
    let params: Vec<&dyn ToSql> = vec![&allowed];
    let filter = SqlFilter::new("key IN rarray(?1)", &params);
    let keys = ids(&h
        .search_opts(
            "kqxvz report",
            &SearchOpts::new().min_shared(1).filter(filter),
            60,
        )
        .unwrap());
    assert!(
        keys.iter().all(|k| *k == 13 || *k == 14),
        "score_union must apply the filter: got {keys:?}"
    );
    for excluded in [1i64, 2, 12] {
        assert!(
            !keys.contains(&excluded),
            "the excluded high-float doc {excluded} must NOT appear: {keys:?}"
        );
    }
}

#[test]
fn key_in_rarray_restricts_to_an_allowed_set() {
    let h = fixture();
    // Docs 1, 2, 7, 8 all share "quick"; restrict to {2, 8}.
    let allowed: Rc<Vec<Value>> = Rc::new(vec![Value::Integer(2), Value::Integer(8)]);
    let params: Vec<&dyn ToSql> = vec![&allowed];
    let filter = SqlFilter::new("key IN rarray(?1)", &params);
    let hits = h
        .search_opts("quick", &SearchOpts::new().filter(filter), 10)
        .unwrap();
    let keys = ids(&hits);
    assert!(!keys.is_empty(), "the allowed docs still match");
    assert!(
        keys.iter().all(|k| *k == 2 || *k == 8),
        "filtered to {{2,8}}: {keys:?}"
    );
}

#[test]
fn anonymous_placeholder_binds_alongside_the_scope_param() {
    // `txt LIKE ?` is an anonymous placeholder; it must bind correctly even though trifle appends
    // the candidate-scope param after the fragment.
    let h = fixture();
    let pat = "%fox%";
    let params: Vec<&dyn ToSql> = vec![&pat];
    let filter = SqlFilter::new("txt LIKE ?", &params);
    let hits = h
        .search_opts("quick", &SearchOpts::new().filter(filter), 10)
        .unwrap();
    let keys = ids(&hits);
    // Only doc 1 ("...quick brown fox...") contains "fox" among the "quick" docs.
    assert!(keys.contains(&1), "got {keys:?}");
    assert!(keys.iter().all(|k| *k == 1), "only fox docs: {keys:?}");
}

#[test]
fn numbered_placeholder_reuse_does_not_collide_with_scope() {
    // `?1` used TWICE in the fragment must bind to the caller's param, not the scope rarray.
    let h = fixture();
    let kv = 7i64;
    let params: Vec<&dyn ToSql> = vec![&kv];
    let filter = SqlFilter::new("key = ?1 OR key = ?1", &params);
    let hits = h
        .search_opts("quick", &SearchOpts::new().filter(filter), 10)
        .unwrap();
    assert!(
        hits.iter().all(|m| m.key.as_i64() == Some(7)),
        "reused ?1 bound to 7: {:?}",
        ids(&hits)
    );
    assert!(hit(&hits, 7), "doc 7 ('how vexingly quick...') matches");
}

#[test]
fn an_empty_allowed_set_matches_nothing() {
    let h = fixture();
    let allowed: Rc<Vec<Value>> = Rc::new(vec![]);
    let params: Vec<&dyn ToSql> = vec![&allowed];
    let filter = SqlFilter::new("key IN rarray(?1)", &params);
    let hits = h
        .search_opts("quick", &SearchOpts::new().filter(filter), 10)
        .unwrap();
    assert!(hits.is_empty(), "an empty key set filters everything out");
}

#[test]
fn filter_does_not_reduce_the_floor_for_unfiltered_queries() {
    // Sanity: the same query without a filter returns the superset.
    let h = fixture();
    let unfiltered = ids(&h.search("quick", 10).unwrap());
    let allowed: Rc<Vec<Value>> = Rc::new(vec![Value::Integer(1)]);
    let params: Vec<&dyn ToSql> = vec![&allowed];
    let filtered = ids(&h
        .search_opts(
            "quick",
            &SearchOpts::new().filter(SqlFilter::new("key IN rarray(?1)", &params)),
            10,
        )
        .unwrap());
    for k in &filtered {
        assert!(unfiltered.contains(k), "filter only removes, never adds");
    }
}
