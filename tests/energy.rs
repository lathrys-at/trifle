//! End-to-end guards for the v0.4 logit-idf energy weighting (derivation §2/§4/§7), exercised
//! through the public `matches`/`matches_batch` API. These are the panel's adversarial cases
//! turned into regression tests: `batch == serial`, the no-vanish recall floor (a `df = N` gram and
//! all-zero-weight tiny corpora still retrieve every match via the engine's `≥ 1` clamp), the
//! all-common count-only degradation, end-to-end rarity ranking, and the degenerate-knob fallback /
//! coarse-`Δ` guard reachable from `SearchOpts`.

mod common;
use common::*;
use trifle::tokenize::DefaultTokenizer;
use trifle::{Config, Index, Schema, SearchOpts};

/// Load `(id, text)` docs under label `"f"` in one writer batch (faster than per-doc commits).
fn load(h: &Harness, docs: &[(i64, &str)]) {
    let mut w = h.index.writer().unwrap();
    for (id, text) in docs {
        w.upsert(*id, &[("f", *text)]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn batch_equals_serial_ranking() {
    // The per-query energy weights derive only from this query's tokens + the shared snapshot, so a
    // query ranks identically run alone vs. mid-batch.
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown";
    let serial = ids(&h.search(q, 10).unwrap());
    assert!(!serial.is_empty(), "the probe query must hit something");
    let batch = h
        .search_batch(&["lazy dog", q, "five wizards"], 10)
        .unwrap();
    assert_eq!(
        ids(&batch[1]),
        serial,
        "q ranks identically serial vs. mid-batch (batch == serial)"
    );
}

#[test]
fn ubiquitous_gram_does_not_drop_documents() {
    // "alpha" trigrams sit in every segment (df = N) → energy −∞ → weight 0 → the engine's ≥ 1
    // clamp keeps them count-only, so no document vanishes.
    let h = Harness::new();
    load(
        &h,
        &[
            (1, "alpha one"),
            (2, "alpha two"),
            (3, "alpha three"),
            (4, "alpha four"),
            (5, "alpha five"),
        ],
    );
    let hits = h
        .search_opts("alpha", &SearchOpts::new().min_shared(1), 10)
        .unwrap();
    assert_eq!(hits.len(), 5, "a df = N gram still retrieves every segment");
}

#[test]
fn tiny_corpora_retrieve_every_match() {
    // For N = 1..=4 every energy is ≤ 0 (floor near corpus size) → all weights quantize to 0 →
    // engine ≥ 1 clamp → overlap-only ranking. Every matching doc must still come back.
    for n in 1..=4i64 {
        let h = Harness::new();
        let docs: Vec<(i64, &str)> = (1..=n).map(|i| (i, "quick brown fox")).collect();
        load(&h, &docs);
        let hits = h.search("quick brown fox", 10).unwrap();
        assert_eq!(
            hits.len() as i64,
            n,
            "N={n}: all-zero energy weights still retrieve every doc"
        );
    }
}

#[test]
fn all_common_query_degrades_to_count_only_not_empty() {
    // Every query gram is ubiquitous (df = N) → no rarity discrimination → the query degrades to a
    // count-and-length ranking (plane count floored at 1), never empty or a crash.
    let h = Harness::new();
    let docs: Vec<(i64, &str)> = (1..=6)
        .map(|i| (i, "the common shared words here"))
        .collect();
    load(&h, &docs);
    let hits = h.search("common shared words", 10).unwrap();
    assert_eq!(
        hits.len(),
        6,
        "all-common query degrades to count-only, never empty"
    );
}

#[test]
fn rare_gram_outranks_common_only_match() {
    // On a corpus large enough that a rare gram (df = 1, floored) quantizes well above a common gram
    // (df ≈ N → weight 1), the document holding the rare gram must rank first — energy promotes
    // rarity end to end.
    let h = Harness::new();
    let mut docs: Vec<(i64, &str)> = (2..=30).map(|i| (i, "common filler words")).collect();
    docs.push((1, "common qzjwx words"));
    load(&h, &docs);
    let hits = h
        .search_opts("qzjwx common", &SearchOpts::new().min_shared(1), 10)
        .unwrap();
    assert_eq!(
        hits[0].key.as_i64(),
        Some(1),
        "energy promotes the rare-gram match to the top"
    );
}

#[test]
fn degenerate_knobs_fall_back_to_defaults() {
    // ν/κ/Δ are reachable via the public SearchOpts builders; degenerate values (out-of-domain,
    // NaN, +∞) are sanitized to their defaults — no panic from the debug guards, and identical
    // ranking to the default search.
    let h = Harness::new();
    load_fixture(&h);
    let q = "quick brown";
    let baseline = ids(&h.search(q, 10).unwrap());
    assert!(!baseline.is_empty(), "the probe query must hit something");
    let degenerate = [
        SearchOpts::new().nu(0.0),
        SearchOpts::new().nu(-1.0),
        SearchOpts::new().nu(f64::NAN),
        SearchOpts::new().kappa(f64::NAN),
        SearchOpts::new().delta(f64::INFINITY),
    ];
    for opts in &degenerate {
        let hits = h.search_opts(q, opts, 10).unwrap();
        assert_eq!(
            ids(&hits),
            baseline,
            "a degenerate knob falls back to the default (same ranking)"
        );
    }
}

/// Position of doc `id` in result order, or `usize::MAX` if absent.
fn rank_of(order: &[i64], id: i64) -> usize {
    order.iter().position(|&d| d == id).unwrap_or(usize::MAX)
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
fn junk_only_match_does_not_outrank_a_real_gram() {
    // §9 junk-below-real, end to end. N=100 (df_min=10): a real word "kqxvz" in 12 docs (df=12 > 10
    // → NON-floored, a rare-but-real high-energy gram that earns the count credit), a junk word
    // "wjbhm" in 1 doc (df=1 ≤ 10 → FLOORED, E_max energy but NO credit). A doc matching only junk
    // must not out-rank a doc matching the real gram — the count credit, not the floor, restores
    // that order (the floor alone parks junk at the energy ceiling). The junk doc is still retrieved.
    let h = Harness::new();
    let mut docs: Vec<(i64, String)> = Vec::new();
    for i in 1..=12 {
        docs.push((i, "kqxvz neutral".to_string())); // real word ⇒ df(kqxvz grams)=12
    }
    for i in 13..=99 {
        docs.push((i, format!("padding number {i}"))); // bulk filler, no query grams
    }
    docs.push((100, "wjbhm neutral".to_string())); // junk word ⇒ df=1
    load_owned(&h, &docs);

    let hits = h
        .search_opts("kqxvz wjbhm", &SearchOpts::new().min_shared(1), 20)
        .unwrap();
    let order = ids(&hits);
    assert!(hit(&hits, 1), "the real-gram doc is retrieved");
    assert!(
        hit(&hits, 100),
        "the junk-only doc is retrieved too (junk is not dropped, just un-credited)"
    );
    assert!(
        rank_of(&order, 1) < rank_of(&order, 100),
        "the real-gram doc out-ranks junk-only (order = {order:?})"
    );
}

#[test]
fn concentration_cap_demotes_a_commons_heavy_offtopic_doc() {
    // §9 concentration cap, end to end. The query "kqxvz report system" is concentrated: one
    // dominant rare gram (kqxvz, df=12) amid two common words (report, system; df=50 each → 8
    // query-relative common grams). WITHOUT the cap, an off-topic doc matching the 8 commons earns
    // a flat 8·μ credit that out-weighs an on-topic doc matching only the one rare gram; the cap
    // shrinks μ so the discriminating gram wins.
    let h = Harness::new();
    let mut docs: Vec<(i64, String)> = Vec::new();
    for i in 1..=12 {
        docs.push((i, "kqxvz".to_string())); // on-topic rare word ⇒ df(kqxvz grams)=12
    }
    for i in 13..=62 {
        docs.push((i, "report system".to_string())); // commons ⇒ df(report)=df(system)=50
    }
    for i in 63..=100 {
        docs.push((i, format!("padding number {i}"))); // filler
    }
    load_owned(&h, &docs);

    let hits = h
        .search_opts("kqxvz report system", &SearchOpts::new().min_shared(1), 30)
        .unwrap();
    let order = ids(&hits);
    assert!(hit(&hits, 1), "the on-topic rare-gram doc is retrieved");
    assert!(
        hit(&hits, 13),
        "the off-topic commons-heavy doc is retrieved"
    );
    assert!(
        rank_of(&order, 1) < rank_of(&order, 13),
        "the cap keeps the rare-gram doc above the commons-heavy doc (order = {order:?})"
    );
}

#[test]
fn batch_equals_serial_under_the_count_credit() {
    // The credit μ and the §9 cap are pure functions of this query's grams + the shared
    // (σ, N, ν, κ, Δ) snapshot, so a credit-bearing query ranks identically alone vs mid-batch.
    let h = Harness::new();
    let mut docs: Vec<(i64, String)> = Vec::new();
    for i in 1..=12 {
        docs.push((i, "kqxvz report".to_string()));
    }
    for i in 13..=50 {
        docs.push((i, "report only".to_string()));
    }
    load_owned(&h, &docs);

    let q = "kqxvz report";
    let opts = SearchOpts::new().min_shared(1);
    let serial = ids(&h.search_opts(q, &opts, 20).unwrap());
    assert!(!serial.is_empty(), "the probe query must hit something");
    let batch = h
        .index
        .reader()
        .unwrap()
        .matches_batch(&["report only", q, "kqxvz"], &opts, 20)
        .unwrap();
    assert_eq!(
        ids(&batch[1]),
        serial,
        "credit-bearing q ranks identically serial vs. mid-batch (batch == serial)"
    );
}

#[test]
fn dedup_keeps_the_max_float_segment_per_key() {
    // drain_top_k dedups one candidate per KEY, keeping the MAX-float segment. Doc 1 has a
    // rare-gram segment (title "kqxvz", high float) and a common-gram segment (body "report",
    // low float); it must surface via the higher-float title segment.
    let h = Harness::new();
    let mut w = h.index.writer().unwrap();
    w.upsert(1, &[("title", "kqxvz"), ("body", "report")])
        .unwrap();
    for i in 2..=40 {
        w.upsert(i, &[("body", "report")]).unwrap(); // make "report" common (df high)
    }
    w.commit().unwrap();

    let hits = h
        .search_opts("kqxvz report", &SearchOpts::new().min_shared(1), 50)
        .unwrap();
    let m1 = hits
        .iter()
        .find(|m| m.key.as_i64() == Some(1))
        .expect("doc 1 retrieved");
    assert_eq!(
        m1.label, "title",
        "per-key dedup kept the higher-float (rare-gram) segment, not the common one"
    );
}

#[test]
fn drain_ordering_is_deterministic_across_runs() {
    // The float sort's total-order tiebreak (float desc → integer score desc → seg id asc) makes
    // the eager result order identical run to run — load-bearing for batch == serial and the
    // thrash oracle.
    let h = Harness::new();
    let mut docs: Vec<(i64, String)> = Vec::new();
    for i in 1..=12 {
        docs.push((i, "kqxvz report system".to_string()));
    }
    for i in 13..=62 {
        docs.push((i, "report system".to_string()));
    }
    load_owned(&h, &docs);

    let opts = SearchOpts::new().min_shared(1);
    let first = ids(&h.search_opts("kqxvz report system", &opts, 64).unwrap());
    for _ in 0..30 {
        let again = ids(&h.search_opts("kqxvz report system", &opts, 64).unwrap());
        assert_eq!(
            again, first,
            "the drain ordering must be identical across runs"
        );
    }
}

/// Open a flat index at `dir/trifle.db` with a caller-chosen `σ` (the rest default).
fn open_with_sigma(dir: &std::path::Path, sigma: f64) -> Index<DefaultTokenizer> {
    let cfg = Config {
        sigma,
        ..Config::default()
    };
    Index::open_at(&dir.join("trifle.db"), Schema::flat(), cfg).unwrap()
}

#[test]
fn sigma_config_edges_do_not_panic_and_fall_back() {
    // σ is an index-level Config field, sanitized to (0,1) at open. Every degenerate σ must open
    // without panic and rank identically to the sanitized default 0.9 (all fall back); a σ just
    // below 1 (huge-but-finite μ) must produce a finite, sane ranking (no overflow / NaN).
    let docs: Vec<(i64, String)> = {
        let mut v: Vec<(i64, String)> = (1..=12).map(|i| (i, "kqxvz report".to_string())).collect();
        v.extend((13..=50).map(|i| (i, "report only".to_string())));
        v
    };
    let load = |idx: &Index<DefaultTokenizer>| {
        let mut w = idx.writer().unwrap();
        for (id, t) in &docs {
            w.upsert(*id, &[("f", t.as_str())]).unwrap();
        }
        w.commit().unwrap();
    };
    let rank = |idx: &Index<DefaultTokenizer>| {
        ids(&idx
            .reader()
            .unwrap()
            .matches("kqxvz report", &SearchOpts::new().min_shared(1), 50)
            .unwrap())
    };

    let base_dir = tempfile::tempdir().unwrap();
    let base = open_with_sigma(base_dir.path(), 0.9);
    load(&base);
    let baseline = rank(&base);
    assert!(!baseline.is_empty(), "the probe query must hit something");

    for bad in [
        0.0,
        1.0,
        -0.1,
        1.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let idx = open_with_sigma(dir.path(), bad);
        load(&idx);
        assert_eq!(
            rank(&idx),
            baseline,
            "σ={bad} must fall back to the 0.9 ranking"
        );
    }

    // σ = 0.9999 is in-range and kept: μ ≈ ln(9999) ≈ 9.21, huge but finite — no overflow / NaN.
    let dir = tempfile::tempdir().unwrap();
    let idx = open_with_sigma(dir.path(), 0.9999);
    load(&idx);
    let got = rank(&idx);
    assert_eq!(got.len(), 50, "σ=0.9999 retrieves every doc, no NaN crash");
    assert!(
        rank_of(&got, 1) < rank_of(&got, 13),
        "σ=0.9999: the rare-gram docs still rank above commons-only: {got:?}"
    );
}

#[cfg(debug_assertions)]
#[test]
fn coarse_delta_trips_the_quantization_guard_in_debug() {
    // The §7 Δ < 2·E_floored guard is reachable from the public API: a deliberately coarse Δ on a
    // corpus large enough for the floor regime trips the debug_assert (debug builds only — it is
    // compiled out of release). Δ = 100 is a finite, positive value, so sanitization keeps it.
    let h = Harness::new();
    let docs: Vec<(i64, &str)> = (1..=40)
        .map(|i| (i, "assorted filler words and padding here"))
        .collect();
    load(&h, &docs);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = h.search_opts("assorted filler", &SearchOpts::new().delta(100.0), 10);
    }));
    assert!(res.is_err(), "coarse Δ = 100 trips the §7 guard at N = 40");
}
