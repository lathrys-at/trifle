//! Layer 3 — the opt-in raw-SQL filter over the caller's *live* data.
//!
//! trifle stores **no** filter attribute columns. The attributes callers filter on (`due`,
//! `reps`, `deck`, `tags`) are high-churn and change decoupled from text; a write-infrequent
//! derived cache that mirrored them would silently serve **stale** filter results, a
//! wrong-results bug the (shape-only) drift-reset cannot catch. Filtering the caller's live data
//! is staleness-free by construction.
//!
//! A [`SqlFilter`] is one trusted-constant predicate fragment plus its bound params. It folds
//! into the per-chunk provenance query as
//!
//! ```sql
//! SELECT id, key, label FROM seg WHERE (<fragment>) AND id IN rarray(?{N+1})
//! ```
//!
//! — **fragment textually first, the candidate-scope param last** (`?{N+1}`, `N = params.len()`,
//! trifle-computed) — so both numbered `?1..?N` *and* anonymous `?` placeholders in the fragment
//! bind correctly and no caller placeholder can collide with the scope param. The filter is
//! evaluated scoped to each bucket's candidate ids, so its cost is `O(candidates pulled)`, never
//! `O(corpus)`.
//!
//! Two zero-declaration modes:
//! - **Universal (any deployment):** compute an allowed-key set in your own source of truth and
//!   bind it — `fragment = "key IN rarray(?)"`, `params = [&key_array]`. Subsumes a scope
//!   predicate *and* arbitrary structured filters (run the structured query against your DB, pass
//!   the keys). The key array is cacheable across an as-you-type session.
//! - **Co-located join (Sidecar + an optional `ATTACH`):** attach your tables to trifle's read
//!   connections, then join directly — `"key IN (SELECT note_id FROM src.cards WHERE deck = ?)"`.
//!
//! **Field-scoping.** The provenance query exposes the segment's `label` (the text-field name), so
//! restricting a search to one field is just a filter — the supported idiom, no per-field API:
//!
//! ```
//! # use trifle::SqlFilter;
//! # use trifle::rusqlite::ToSql;
//! let field: &dyn ToSql = &"title";
//! let filter = SqlFilter { fragment: "label = ?1", params: &[field] };
//! # let _ = filter;
//! ```
//!
//! This scopes candidates to segments of that field. (First-class field-scoped candidate generation
//! — field-local idf, per-field channels — arrives with the field-aware index milestone, post-0.4;
//! until then this filter is the field-scoping mechanism.)

use rusqlite::ToSql;

/// An opt-in raw-SQL filter: a **trusted-constant** predicate fragment over the `seg` columns
/// (`id`, `key`, `label`, `txt`, `len`) — or co-located caller tables via `ATTACH` — plus its
/// bound params.
///
/// # Safety contract
///
/// The `fragment` is a **trusted compile-time constant** (your own code, like a prepared-
/// statement string), **not** built from untrusted input: it is spliced verbatim into the
/// `WHERE`, so it is the injection surface. Bind data through `params` only — never format values
/// into the fragment. Reference params as numbered `?1..?N` (reusable) or anonymous `?`; the
/// candidate-scope param trifle appends is always `?{N+1}`, so your placeholders never collide.
#[derive(Clone, Copy)]
pub struct SqlFilter<'a> {
    /// The trusted-constant predicate fragment (the `seg` columns are in scope).
    pub fragment: &'a str,
    /// The bound params, in `?1..?N` order. Bind **all** data here; never interpolate into the
    /// fragment.
    pub params: &'a [&'a dyn ToSql],
}

impl<'a> SqlFilter<'a> {
    /// A filter from a trusted-constant `fragment` and its `params`.
    pub fn new(fragment: &'a str, params: &'a [&'a dyn ToSql]) -> Self {
        SqlFilter { fragment, params }
    }
}
