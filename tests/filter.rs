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
    // the candidate-scope param after the fragment (the F3 audit-footgun fix).
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
