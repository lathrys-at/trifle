//! `trifle-bench` — the benchmark harness.
//!
//! The harness drives trifle's streaming read API (`reader().matches(query, &SearchOpts,
//! limit)`) against the in-process SQLite baselines on the same corpus and queries, so a
//! comparison isolates the matching strategy from the store. The weighted-overlap order **is**
//! the ranking — there is no rerank pool, so there is no effort knob.
//!
//! Distinct benchmarks, kept separate because conflating them measures the wrong thing:
//!
//! - **`latency`** — search latency + throughput only. Same corpus/queries, trifle vs the
//!   FTS5-trigram (phrase) BM25 and `LIKE`-scan speed baselines. Reports p50/p90/p95/p99/max
//!   and throughput, serial / batched / concurrent (the read-pool fan-out is a distinct axis).
//!   A trifle-only in-corpus self-recall figure rides along as a sanity check.
//! - **`relevance`** — recall + latency on MS MARCO real dev queries + qrels, vs word-level
//!   BM25 (and the trigram cousin) and the `LIKE` floor. One timed pull at depth 50 per query;
//!   recall/MRR/nDCG @ {1,5,10,50} fall out of slicing that one ranked list.
//! - **`fuzzy`** — recall + latency on entity-name + injected-edit queries over a GeoNames
//!   corpus, vs FTS5 trigram-MATCH (OR-bag) and the `LIKE` floor. Same single-pull-at-50
//!   methodology; reports per edit-count.
//! - **`selsweep`** — the selection-cost frontier: recall@k vs Σdf (and vs p99 latency) for
//!   BOTH selection arms (`t_max` and `df_budget`), under the work-done collector. CSV/JSON.
//!
//! Plus two utilities:
//!
//! - **`profile`** — the work-done instrument: the Σ(kept-posting cardinality) distribution,
//!   the quantity whose growth with corpus size would flatten trifle's latency advantage.
//! - **`fetch`** — warm the pinned-corpus cache on a network machine before an offline run.
//!
//! Everything is driven by a master `--seed` so a run is byte-reproducible, and the size knobs
//! (`--docs`, `--queries`, …) let you trace how cost scales with corpus size. See
//! `benchmarks/README.md` for the corpora and how to run.

mod baselines;
mod corpus;
mod instrument;
mod metrics;
mod profile;
mod query;
mod rng;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use baselines::{Engine, Fts5, Fts5Word, Like, MatchMode, Trifle, Tuning};
use corpus::{Corpus, Entity};
use metrics::{
    Dist, Latency, fmt_dur, mrr_at_k, ndcg_at_k, scored_queries, set_recall_at_k, throughput,
};
use serde::Serialize;
use trifle::tokenize::{Tokenizer, TrigramTokenizer};

/// The default master seed (`0x5EED…`). Override with `--seed`.
const DEFAULT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// The fixed retrieval depth for the recall evals (`relevance`, `fuzzy`, `selsweep`): each
/// query is pulled **once** to this depth (SQLite baselines `LIMIT 50`), and every reported
/// `recall@k` / `MRR@k` / `nDCG@k` is a slice of that single ranked list.
const KMAX: usize = 50;

/// The cutoffs reported for every quality metric, all sliced from the one depth-[`KMAX`] pull.
const REPORT_KS: [usize; 4] = [1, 5, 10, 50];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        eprint!("{}", USAGE);
        return ExitCode::FAILURE;
    };
    let result = match command.as_str() {
        "latency" => cmd_latency(rest),
        "relevance" => cmd_relevance(rest),
        "fuzzy" => cmd_fuzzy(rest),
        "selsweep" => cmd_selsweep(rest),
        "dsweep" => cmd_dsweep(rest),
        "bench" => cmd_bench(rest),
        "write" => cmd_write(rest),
        "profile" => cmd_profile(rest),
        "fetch" => cmd_fetch(rest),
        "help" | "--help" | "-h" => {
            print!("{}", USAGE);
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command: {other}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
trifle-bench — benchmark harness

USAGE:
    trifle-bench <COMMAND> [OPTIONS]

COMMANDS:
    latency    Search latency + throughput, trifle vs in-process baselines (speed only)
    relevance  MS MARCO real dev queries+qrels: recall/MRR/nDCG@{1,5,10,50} + latency
    fuzzy      Entity name+edit recall/MRR/nDCG@{1,5,10,50} + latency, per edit-count
    selsweep   Selection-cost frontier: recall@k vs Σdf for both arms (t_max, df_budget)
    dsweep     Recall@{1,5,10,50} vs weight_step D, and the WeightStepHint vs the optimum
    bench      Engine microbench: trifle-overlap build/walk on synthetic bitmaps (no SQLite)
    write      Write throughput: incremental upsert/commit, rebuild, and compact cost
    profile    Σ(kept-posting cardinality) distribution — the work-done curve
    fetch      Download + verify the pinned corpus assets into the cache (no bench)
    help       Show this message

COMMON OPTIONS:
    --docs <N>                    Index size. For `relevance`, the distractor target
                                  (answers are always indexed) [default: 50000, fuzzy 200000]
    --queries <N>                 Queries [default: latency 2000, relevance 1000, fuzzy 2000]
    --seed <N>                    Master seed (decimal or 0x-hex); fixes corpus + query
                                  sampling for a byte-reproducible run [default: 0x5EED..]

SEARCH-TUNING (trifle only; `None` leaves the engine default):
    --min-shared <M>              Match floor m (shared rare tokens) [default: 2]
    --t-max <T>                   Selection cap t_max — rarest tokens kept [default: 12]
    --weight-step <D>             df-doublings per IDF weight step [default: 1.0]

ENGINE SELECTION (latency, relevance, fuzzy):
    --filter <engine>             Skip an engine; repeatable. Engines: trifle,
                                  fts5-trigram-bm25, fts5-word-bm25, like-scan.
                                  (trifle is the subject of `fuzzy` and cannot be filtered.)

LATENCY (speed only):
    --corpus <synthetic|msmarco|geonames-cities|geonames-all>  Corpus [default: synthetic]
                                  synthetic/msmarco = in-corpus snippets (0-2 typos);
                                  geonames-* = exact entity-name queries (no typos), the
                                  short-structured-segment scaling regime.
    --k <N>                       Top-k limit per search [default: 10]
    --warmup <N>                  Untimed warmup queries [default: 3]
    --repeat <N>                  Measured passes (samples accumulate) [default: 1]
    --batched                     One matches_batch call (shares posting/frequency reads)
    --concurrent <T>              Run trifle across T reader threads, each on its own
                                  index.reader() (the read-pool caller-fanout axis) [default: 0]
    --format <text|json>          Output format; json includes p95 [default: text; serial only]
    --instrument <xctrace|samply> Re-exec the run under a sampling profiler (no effect on the
                                  measured code). xctrace = Instruments (macOS); samply = X-plat.
    --instrument-out <dir>        Where to write the trace [.cache/bench/instruments]

    A trifle-only in-corpus self-recall figure (the snippet/name's own doc is the answer)
    rides along so the speed numbers carry a quality sanity check.

RELEVANCE (recall + latency; single depth-50 pull):
    Engines: trifle, fts5-word-bm25 (BM25), fts5-trigram-bm25 (OR-bag), like-scan.
    --warmup <N>                  Untimed warmup queries [default: 3]
    --format <text|json>          Output format [default: text]

FUZZY (recall + latency; single depth-50 pull):
    --corpus <geonames-cities|geonames-all>   Entity corpus [default: geonames-cities]
    --edits <N>                   Typos per query. Omit to run {1, 2} separately.
    --warmup <N>                  Untimed warmup queries [default: 3]
    --format <text|json>          Output format [default: text]

SELSWEEP (selection-cost frontier; trifle only):
    --corpus <geonames-all|geonames-cities|msmarco-relevance>  Labeled corpus [default:
                                  msmarco-relevance]
    --edits <N>                   Typos per query for geonames [default: 2]
    --max-tmax <T>                Top of the t_max grid (2,4,..,T) [default: 20]
    --format <csv|json>           Output format [default: csv]
    Columns: arm,knob,N,k,recall,sigma_df_p50,sigma_df_p99,lat_p50_us,lat_p99_us.
    N is per-run (--docs); sweep N externally over the geometric x5 ladder
    {1000,5000,25000,125000,625000} for the scaling frontier.

DSWEEP (recall vs the weight_step D; trifle only, single depth-50 pull):
    --corpus <msmarco-relevance|geonames-cities|geonames-all>  Labeled corpus [default:
                                  msmarco-relevance]
    --edits <N>                   Typos per query for geonames [default: 2]
    --steps <a,b,c>               The D grid [default: 0.5,1.0,1.5,2.0,3.0]
    Reports recall/MRR/nDCG@{1,5,10,50} per D, then the corpus WeightStepHint's suggested D
    against the recall@10-optimal D in the grid.

BENCH (trifle-overlap engine, synthetic bitmaps — no SQLite, no corpus):
    --postings <K>                Postings per query (selected tokens) [default: 10]
    --trials <N>                  Counters built per regime (build-time samples) [default: 300]
    --universe <N>                Id space the synthetic postings draw from [default: 1000000]
    Sweeps posting cardinality to expose the build's op-count flatness, contrasts the
    all-weight-1 fast path with mixed weights, and shallow top-k vs a full deep-pull walk.

WRITE (write-path throughput; trifle only):
    --corpus <synthetic|msmarco|geonames-cities|geonames-all>  Corpus [default: synthetic]
    --docs <N>                    Documents to index [default: 50000]
    --batch <N>                   Commit batch size for the incremental path [default: 1000]
    Reports incremental upsert/commit docs/s, rebuild docs/s, and compact() cost.

PROFILE:
    --corpus <synthetic|msmarco>  Corpus source [default: synthetic]
    --k <N>                       Top-k limit per search [default: 10]

EXAMPLES:
    trifle-bench fetch --corpus geonames-cities
    trifle-bench latency --docs 100000 --queries 5000 --seed 42
    trifle-bench latency --corpus geonames-all --docs 625000 --filter like-scan
    trifle-bench latency --corpus msmarco --docs 25000 --concurrent 8
    trifle-bench relevance --docs 100000 --queries 2000 --format json
    trifle-bench fuzzy --corpus geonames-cities --edits 1
    trifle-bench selsweep --corpus geonames-all --docs 125000
    trifle-bench profile --docs 1000000
";

// ----- argument parsing -------------------------------------------------------

/// A parsed flag set: `--key value` / `--key=value` valued flags and `--flag` booleans.
/// Deliberately tiny and std-only (the harness vendors no arg crate, in keeping with its
/// hand-rolled RNG and shell-out asset fetch).
struct Flags {
    /// Values per flag, accumulated in order so a flag may repeat (e.g. `--filter`). Scalar
    /// accessors take the last occurrence (last-wins); `values` returns all.
    valued: HashMap<String, Vec<String>>,
    bools: HashSet<String>,
}

impl Flags {
    fn parse(args: &[String], bool_flags: &[&str]) -> Result<Flags, String> {
        let bset: HashSet<&str> = bool_flags.iter().copied().collect();
        let mut valued: HashMap<String, Vec<String>> = HashMap::new();
        let mut bools = HashSet::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument: {a}"))?;
            if let Some((k, v)) = key.split_once('=') {
                valued.entry(k.to_string()).or_default().push(v.to_string());
                i += 1;
            } else if bset.contains(key) {
                bools.insert(key.to_string());
                i += 1;
            } else {
                let v = args
                    .get(i + 1)
                    .filter(|v| !v.starts_with("--"))
                    .ok_or_else(|| format!("--{key} expects a value"))?;
                valued.entry(key.to_string()).or_default().push(v.clone());
                i += 2;
            }
        }
        Ok(Flags { valued, bools })
    }

    /// Reject any flag not in `allowed` — a typo'd knob is an error, not a silent default.
    fn reject_unknown(&self, allowed: &[&str]) -> Result<(), String> {
        let set: HashSet<&str> = allowed.iter().copied().collect();
        for k in self.valued.keys().chain(self.bools.iter()) {
            if !set.contains(k.as_str()) {
                return Err(format!("unknown option: --{k}"));
            }
        }
        Ok(())
    }

    /// The last value given for `key` (last-wins for a repeated scalar flag).
    fn last(&self, key: &str) -> Option<&String> {
        self.valued.get(key).and_then(|v| v.last())
    }
    /// Every value given for a repeatable flag, in order (empty if never given).
    fn values(&self, key: &str) -> &[String] {
        self.valued.get(key).map(Vec::as_slice).unwrap_or(&[])
    }
    fn str(&self, key: &str, default: &str) -> String {
        self.last(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }
    fn has(&self, key: &str) -> bool {
        self.valued.contains_key(key)
    }
    fn flag(&self, key: &str) -> bool {
        self.bools.contains(key)
    }
    fn u64(&self, key: &str, default: u64) -> Result<u64, String> {
        match self.last(key) {
            Some(v) => parse_u64(v).ok_or_else(|| format!("--{key}: not an integer: {v}")),
            None => Ok(default),
        }
    }
    fn usize(&self, key: &str, default: usize) -> Result<usize, String> {
        Ok(self.u64(key, default as u64)? as usize)
    }
    fn opt_u32(&self, key: &str) -> Result<Option<u32>, String> {
        match self.last(key) {
            Some(v) => v
                .parse::<u32>()
                .map(Some)
                .map_err(|_| format!("--{key}: not a u32: {v}")),
            None => Ok(None),
        }
    }
    fn opt_u64(&self, key: &str) -> Result<Option<u64>, String> {
        match self.last(key) {
            Some(v) => parse_u64(v)
                .map(Some)
                .ok_or_else(|| format!("--{key}: not an integer: {v}")),
            None => Ok(None),
        }
    }
    fn opt_f64(&self, key: &str) -> Result<Option<f64>, String> {
        match self.last(key) {
            Some(v) => v
                .parse::<f64>()
                .map(Some)
                .map_err(|_| format!("--{key}: not a float: {v}")),
            None => Ok(None),
        }
    }
}

/// Parse a `u64` in decimal or `0x`-hex, ignoring `_` separators.
fn parse_u64(s: &str) -> Option<u64> {
    let t = s.replace('_', "");
    match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => t.parse::<u64>().ok(),
    }
}

// ----- shared setup -----------------------------------------------------------

/// Build the corpus named by `--corpus`, sized by `--docs`, seeded by `--seed` — the
/// `synthetic`/`msmarco` snippet corpora (used by `profile`; `latency` handles the geonames
/// variants itself).
fn build_corpus(flags: &Flags) -> Result<(Corpus, u64), String> {
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let docs = flags.usize("docs", 50_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    let corpus = match flags.str("corpus", "synthetic").as_str() {
        "synthetic" => corpus::synthetic(docs, seed),
        "msmarco" => corpus::msmarco(docs, seed).map_err(|e| format!("msmarco: {e}"))?,
        other => return Err(format!("unknown --corpus {other} (synthetic|msmarco)")),
    };
    Ok((corpus, seed))
}

/// The trifle search-strictness knobs for a run, from the shared `--min-shared`/`--t-max`/
/// `--weight-step` flags. Each `None` leaves trifle's engine default.
fn tuning(flags: &Flags) -> Result<Tuning, String> {
    Ok(Tuning {
        min_shared: flags.opt_u32("min-shared")?,
        t_max: flags.opt_u64("t-max")?.map(|v| v as usize),
        weight_step: flags.opt_f64("weight-step")?,
    })
}

const CORPUS_OPTS: &[&str] = &[
    "corpus",
    "docs",
    "seed",
    "min-shared",
    "t-max",
    "weight-step",
];

/// The engine identifiers accepted by `--filter`. These must match the strings each engine
/// returns from [`Engine::name`]. Not every command runs every engine (latency/fuzzy use the
/// trigram FTS5; relevance adds the word-level BM25).
const ENGINE_TRIFLE: &str = "trifle";
const ENGINE_FTS5: &str = "fts5-trigram-bm25";
const ENGINE_FTS5_WORD: &str = "fts5-word-bm25";
const ENGINE_LIKE: &str = "like-scan";
const ALL_ENGINES: [&str; 4] = [ENGINE_TRIFLE, ENGINE_FTS5, ENGINE_FTS5_WORD, ENGINE_LIKE];

/// The set of engines to skip, collected from repeated `--filter <engine>` flags. Each value
/// must name a known engine — a typo is an error, not a silent no-op.
fn skipped_engines(flags: &Flags) -> Result<HashSet<String>, String> {
    let mut skip = HashSet::new();
    for v in flags.values("filter") {
        if !ALL_ENGINES.contains(&v.as_str()) {
            return Err(format!(
                "--filter {v}: unknown engine (expected one of {})",
                ALL_ENGINES.join(", ")
            ));
        }
        skip.insert(v.clone());
    }
    Ok(skip)
}

/// Print the `# filter: skipping …` line when any engine is filtered out.
fn print_filter(skip: &HashSet<String>) {
    if !skip.is_empty() {
        let mut s: Vec<&str> = skip.iter().map(String::as_str).collect();
        s.sort_unstable();
        println!("# filter: skipping {}", s.join(", "));
    }
}

/// The instrumentation re-exec seam used by `latency`. Returns `Ok(true)` if it re-exec'd the
/// run under a profiler (the caller should stop), `Ok(false)` to run the benchmark normally.
/// The env guard keeps the profiled child from re-instrumenting.
fn maybe_instrument(flags: &Flags, args: &[String], subcommand: &str) -> Result<bool, String> {
    if let Some(name) = flags.last("instrument")
        && !instrument::is_inner()
    {
        let profiler = instrument::Profiler::parse(name)?;
        let out = flags.str("instrument-out", ".cache/bench/instruments");
        let passthru = instrument::strip_self_flags(args);
        let code = instrument::run(profiler, Path::new(&out), subcommand, &passthru)?;
        if code != 0 {
            return Err(format!("profiler exited with code {code}"));
        }
        return Ok(true);
    }
    Ok(false)
}

// ----- latency (speed only) ---------------------------------------------------

fn cmd_latency(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &["batched"])?;
    let mut allowed = CORPUS_OPTS.to_vec();
    allowed.extend([
        "queries",
        "k",
        "warmup",
        "repeat",
        "batched",
        "concurrent",
        "filter",
        "format",
        "instrument",
        "instrument-out",
    ]);
    flags.reject_unknown(&allowed)?;

    // Instrumentation seam: if asked to profile and we are NOT already the profiled child,
    // re-exec the whole run under the chosen sampler and stop here.
    if maybe_instrument(&flags, args, "latency")? {
        return Ok(());
    }

    let skip = skipped_engines(&flags)?;
    // latency runs the three speed-comparison engines (word-level BM25 is relevance-only).
    if [ENGINE_TRIFLE, ENGINE_FTS5, ENGINE_LIKE]
        .iter()
        .all(|e| skip.contains(*e))
    {
        return Err("all engines filtered out — nothing to run".into());
    }

    let corpus_name = flags.str("corpus", "synthetic");
    let k = flags.usize("k", 10)?;
    let warmup = flags.usize("warmup", 3)?;
    let repeat = flags.usize("repeat", 1)?.max(1);
    let batched = flags.flag("batched");
    let concurrent = flags.usize("concurrent", 0)?;
    let tuning = tuning(&flags)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;

    let json_mode = match flags.str("format", "text").as_str() {
        "text" => false,
        "json" => true,
        other => return Err(format!("--format {other} (text|json)")),
    };
    if json_mode && (batched || concurrent > 1) {
        return Err(
            "--format json is implemented for serial mode only (drop --batched / --concurrent)"
                .into(),
        );
    }

    // The two latency query regimes: snippet corpora (synthetic/msmarco) carry a 0/1/2 typo
    // mix; the geonames corpora are exact entity-name queries (no typos), a short-structured
    // scaling regime. Both expose a per-query relevant id for the trifle self-recall sanity.
    let inputs = latency_inputs(&flags, &corpus_name, seed)?;
    if inputs.qtexts.is_empty() {
        return Err("no queries generated (corpus too small or documents too short)".into());
    }
    let qtexts: Vec<&str> = inputs.qtexts.iter().map(String::as_str).collect();
    let relevant: Vec<Vec<i64>> = inputs.targets.iter().map(|&t| vec![t]).collect();
    let corpus = &inputs.corpus;

    let meta = RunMeta {
        corpus: &corpus_name,
        provenance: &corpus.provenance,
        docs: corpus.docs.len(),
        queries: qtexts.len(),
        k,
        seed,
        mix: inputs.mix,
        warmup: warmup.min(qtexts.len()),
        repeat,
        min_shared: tuning.min_shared,
        t_max: tuning.t_max,
    };
    let bench = Bench {
        qtexts: &qtexts,
        relevant: &relevant,
        k,
        warmup,
        repeat,
        batched,
    };

    // Concurrent mode (text only): trifle-only read-pool fan-out throughput + per-query p99.
    if concurrent > 1 {
        if skip.contains(ENGINE_TRIFLE) {
            return Err("concurrent mode runs trifle only, but it is filtered out".into());
        }
        print_run_header(&meta, &skip);
        println!(
            "# mode=concurrent threads={concurrent} (trifle only — the read-pool fanout axis)"
        );
        let trifle = Trifle::build(corpus, tuning);
        let recall = set_recall_at_k(&trifle.search_batch(&qtexts, k), &relevant, k);
        bench_concurrent(&trifle, &bench, concurrent, recall);
        return Ok(());
    }

    // Serial / batched: measure every engine into records, then render.
    let records = measure_engines(corpus, &bench, &skip, tuning);

    if json_mode {
        render_run_json(&meta, &records);
    } else {
        print_run_header(&meta, &skip);
        println!("# mode={}", if batched { "batched" } else { "serial" });
        render_run_text(&bench, &records);
        println!(
            "# (latency is one axis; durability, footprint, update cost, and semantics differ \
             per engine — see the comparison table in the project README)"
        );
    }
    Ok(())
}

/// The built corpus + generated latency queries (texts and their relevant ids) plus the typo
/// mix the regime produced.
struct LatencyInputs {
    corpus: Corpus,
    qtexts: Vec<String>,
    targets: Vec<i64>,
    mix: [usize; 3],
}

/// Build the latency corpus + queries for `corpus_name`. The snippet corpora draw in-corpus
/// snippets with a 0/1/2 typo mix ([`query::perf_queries`]); the geonames corpora emit exact
/// entity-name queries (zero edits) over short structured segments.
fn latency_inputs(flags: &Flags, corpus_name: &str, seed: u64) -> Result<LatencyInputs, String> {
    let docs = flags.usize("docs", 50_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    match corpus_name {
        "synthetic" | "msmarco" => {
            let n = flags.usize("queries", 2000)?;
            let corpus = match corpus_name {
                "synthetic" => corpus::synthetic(docs, seed),
                _ => corpus::msmarco(docs, seed).map_err(|e| format!("msmarco: {e}"))?,
            };
            let queries = query::perf_queries(&corpus, n, seed);
            let mut mix = [0usize; 3];
            for q in &queries {
                if let Some(slot) = mix.get_mut(q.edits) {
                    *slot += 1;
                }
            }
            let qtexts = queries.iter().map(|q| q.text.clone()).collect();
            let targets = queries.iter().map(|q| q.target).collect();
            Ok(LatencyInputs {
                corpus,
                qtexts,
                targets,
                mix,
            })
        }
        corpus::CORPUS_GEONAMES_CITIES | corpus::CORPUS_GEONAMES_ALL => {
            let n = flags.usize("queries", 2000)?;
            let ent = corpus::geonames(corpus_name, docs, n, seed)
                .map_err(|e| format!("geonames: {e}"))?;
            if ent.targets.is_empty() {
                return Err("no entities loaded (corpus empty)".into());
            }
            // Exact entity-name queries: zero edits, the short-structured-segment regime.
            let qs = query::fuzzy_queries(&ent.targets, 0, seed);
            let mix = [qs.len(), 0, 0];
            let qtexts = qs.iter().map(|q| q.text.clone()).collect();
            let targets = qs.iter().map(|q| q.target).collect();
            Ok(LatencyInputs {
                corpus: ent.corpus,
                qtexts,
                targets,
                mix,
            })
        }
        other => Err(format!(
            "unknown --corpus {other} (synthetic|msmarco|geonames-cities|geonames-all)"
        )),
    }
}

/// The fixed inputs to one latency measurement, shared across engines (bundled to keep the
/// measurement helpers below their argument-count lint bar).
struct Bench<'a> {
    qtexts: &'a [&'a str],
    relevant: &'a [Vec<i64>],
    k: usize,
    warmup: usize,
    repeat: usize,
    batched: bool,
}

/// One measured engine row. `samples_ns` is the raw per-query latency in call order (serial
/// mode); it is empty in batched mode, which times the whole set as one call. `recall` is
/// `Some` only for trifle (its in-corpus self-recall sanity figure — the snippet/name queries
/// make the FTS5 phrase baseline's recall a lie, so it is left unscored).
struct Record {
    engine: String,
    samples_ns: Vec<u64>,
    throughput_qps: f64,
    recall: Option<f64>,
}

/// Build, then measure, each non-filtered engine for `latency`: trifle (with its self-recall)
/// plus the FTS5-phrase and LIKE speed baselines (no recall — phrase-MATCH on the typo'd
/// snippet queries scores ~0 by construction, so a recall number there would misrepresent FTS5).
fn measure_engines(
    corpus: &Corpus,
    bench: &Bench,
    skip: &HashSet<String>,
    tuning: Tuning,
) -> Vec<Record> {
    let mut records = Vec::new();
    if !skip.contains(ENGINE_TRIFLE) {
        let trifle = Trifle::build(corpus, tuning);
        records.push(measure_one(bench, true, &trifle));
    }
    if !skip.contains(ENGINE_FTS5) {
        // Phrase mode: the latency *speed* baseline. No recall (see the doc above).
        match Fts5::build(corpus, MatchMode::Phrase) {
            Some(fts5) => records.push(measure_one(bench, false, &fts5)),
            None => eprintln!("note: FTS5 trigram unavailable in the linked SQLite — skipping"),
        }
    }
    if !skip.contains(ENGINE_LIKE) {
        let like = Like::build(corpus);
        records.push(measure_one(bench, false, &like));
    }
    records
}

/// Run one engine's measurement: untimed warmup, then the timed loop (per-query serial
/// samples, or a whole-set batched best-of-`repeat`), then — if `want_recall` — an untimed
/// recall@k pass over the *same* queries (`batch == serial`, so it matches the timed results).
fn measure_one(bench: &Bench, want_recall: bool, engine: &dyn Engine) -> Record {
    let qs = bench.qtexts;
    let w = bench.warmup.min(qs.len());
    if w > 0 {
        let _ = engine.search_batch(&qs[..w], bench.k);
    }
    let (samples_ns, throughput_qps) = if bench.batched {
        let mut best = Duration::MAX;
        for _ in 0..bench.repeat {
            let t = Instant::now();
            let _ = engine.search_batch(qs, bench.k);
            best = best.min(t.elapsed());
        }
        (Vec::new(), throughput(qs.len(), best))
    } else {
        let mut samples = Vec::with_capacity(qs.len() * bench.repeat);
        let wall = Instant::now();
        for _ in 0..bench.repeat {
            for q in qs {
                let t = Instant::now();
                let _ = engine.search(q, bench.k);
                samples.push(t.elapsed().as_nanos() as u64);
            }
        }
        (samples, throughput(qs.len() * bench.repeat, wall.elapsed()))
    };
    let recall = want_recall
        .then(|| set_recall_at_k(&engine.search_batch(qs, bench.k), bench.relevant, bench.k));
    Record {
        engine: engine.name().to_string(),
        samples_ns,
        throughput_qps,
        recall,
    }
}

/// One reader-pool fan-out measurement (trifle-only — the single-`Connection` baselines have
/// no read-pool analogue). `threads` workers share one `&Trifle`, each running its own
/// interleaved shard behind a start gate, every search opening its own pooled `index.reader()`.
/// Reports aggregate throughput AND the per-query p99 across all worker samples.
fn bench_concurrent(trifle: &Trifle, bench: &Bench, threads: usize, recall: f64) {
    let queries = bench.qtexts;
    let k = bench.k;
    let per_thread_warmup = (bench.warmup / threads).min(queries.len());

    // A start gate sized for the workers + this thread. Each worker first warms its own pooled
    // reader connection (the pool creates one lazily per caller), then waits on the gate; the
    // gate opens for everyone at one instant — no spawn drift, so the readers are genuinely
    // concurrent for the whole measured window.
    let gate = Barrier::new(threads + 1);
    let mut samples: Vec<u64> = Vec::new();
    let mut elapsed = Duration::ZERO;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let gate = &gate;
                scope.spawn(move || {
                    for j in 0..per_thread_warmup {
                        let _ = trifle.search(queries[(t + j) % queries.len()], k);
                    }
                    gate.wait(); // every reader released together
                    let mut local = Vec::new();
                    let mut i = t;
                    while i < queries.len() {
                        let s = Instant::now();
                        let _ = trifle.search(queries[i], k);
                        local.push(s.elapsed().as_nanos() as u64);
                        i += threads;
                    }
                    local
                })
            })
            .collect();
        gate.wait(); // opens the gate; the instant after is the true start
        let started = Instant::now();
        for h in handles {
            samples.extend(h.join().expect("worker thread panicked"));
        }
        elapsed = started.elapsed();
    });
    let lat = Latency::from_durations(samples.iter().map(|&ns| Duration::from_nanos(ns)).collect());
    println!(
        "{:>18}  conc({:>2})  {} queries in {} ({:.0} q/s aggregate)  per-query p99 {}  recall@{k} {:.3}",
        "trifle",
        threads,
        queries.len(),
        fmt_dur(elapsed),
        throughput(queries.len(), elapsed),
        fmt_dur(lat.p99()),
        recall,
    );
}

/// The `# …`-prefixed header for the human (text) output of the latency profile.
fn print_run_header(meta: &RunMeta, skip: &HashSet<String>) {
    println!("# latency — {}", meta.provenance);
    let m = meta.mix;
    println!(
        "# docs={} queries={} (0-edit={} 1-edit={} 2-edit={}) k={} warmup={} repeat={}",
        meta.docs, meta.queries, m[0], m[1], m[2], meta.k, meta.warmup, meta.repeat
    );
    print_filter(skip);
}

/// Render the measured records as aligned human-readable lines (one per engine). A baseline
/// with no recall prints a `—` in the recall column.
fn render_run_text(bench: &Bench, records: &[Record]) {
    let k = bench.k;
    for r in records {
        let recall = match r.recall {
            Some(v) => format!("recall@{k} {v:.3}"),
            None => format!("recall@{k}     —"),
        };
        if bench.batched {
            println!(
                "{:>18}  batched  {} queries  {recall}  ({:.0} q/s)",
                r.engine,
                bench.qtexts.len(),
                r.throughput_qps,
            );
        } else {
            let lat = Latency::from_durations(
                r.samples_ns
                    .iter()
                    .map(|&ns| Duration::from_nanos(ns))
                    .collect(),
            );
            println!(
                "{:>18}  serial   p50 {:>8} p90 {:>8} p95 {:>8} p99 {:>8} max {:>8}  {recall}  ({:.0} q/s)",
                r.engine,
                fmt_dur(lat.p50()),
                fmt_dur(lat.p90()),
                fmt_dur(lat.p95()),
                fmt_dur(lat.p99()),
                fmt_dur(lat.max()),
                r.throughput_qps,
            );
        }
    }
}

/// Run metadata carried into the latency human header and the machine-readable JSON.
struct RunMeta<'a> {
    corpus: &'a str,
    provenance: &'a str,
    docs: usize,
    queries: usize,
    k: usize,
    seed: u64,
    mix: [usize; 3],
    warmup: usize,
    repeat: usize,
    min_shared: Option<u32>,
    t_max: Option<usize>,
}

// ---- machine-readable (`--format json`) schema ------------------------------------------

#[derive(Serialize)]
struct TypoMix {
    e0: usize,
    e1: usize,
    e2: usize,
}

#[derive(Serialize)]
struct Conditions {
    git_commit: Option<String>,
    rustc: Option<String>,
    arch: &'static str,
    os: &'static str,
    profile: &'static str,
    cpus: usize,
}

#[derive(Serialize)]
struct LatencyNs {
    p50: u64,
    p90: u64,
    p95: u64,
    p99: u64,
    max: u64,
    mean: f64,
    n: usize,
}

/// Summarize a ns-sample slice into the JSON latency block (includes p95).
fn latency_ns(samples: &[u64]) -> LatencyNs {
    let d = Dist::new(samples.to_vec());
    LatencyNs {
        p50: d.pct(50.0),
        p90: d.pct(90.0),
        p95: d.pct(95.0),
        p99: d.pct(99.0),
        max: d.max(),
        mean: d.mean(),
        n: d.count(),
    }
}

#[derive(Serialize)]
struct RecordJson<'a> {
    engine: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    recall_at_k: Option<f64>,
    recall_k: usize,
    throughput_qps: f64,
    latency_ns: LatencyNs,
    samples_ns: &'a [u64],
}

#[derive(Serialize)]
struct RunJson<'a> {
    command: &'a str,
    corpus: &'a str,
    provenance: &'a str,
    docs: usize,
    queries: usize,
    k: usize,
    seed: u64,
    typo_mix: TypoMix,
    warmup: usize,
    repeat: usize,
    mode: &'a str,
    min_shared: Option<u32>,
    t_max: Option<usize>,
    conditions: Conditions,
    records: Vec<RecordJson<'a>>,
}

/// Emit the whole latency run as one compact JSON object on stdout (no `#` lines), for a
/// post-processor to capture and persist.
fn render_run_json(meta: &RunMeta, records: &[Record]) {
    let records: Vec<RecordJson> = records
        .iter()
        .map(|r| RecordJson {
            engine: &r.engine,
            recall_at_k: r.recall,
            recall_k: meta.k,
            throughput_qps: r.throughput_qps,
            latency_ns: latency_ns(&r.samples_ns),
            samples_ns: &r.samples_ns,
        })
        .collect();
    let obj = RunJson {
        command: "latency",
        corpus: meta.corpus,
        provenance: meta.provenance,
        docs: meta.docs,
        queries: meta.queries,
        k: meta.k,
        seed: meta.seed,
        typo_mix: TypoMix {
            e0: meta.mix[0],
            e1: meta.mix[1],
            e2: meta.mix[2],
        },
        warmup: meta.warmup,
        repeat: meta.repeat,
        mode: "serial",
        min_shared: meta.min_shared,
        t_max: meta.t_max,
        conditions: conditions(),
        records,
    };
    println!(
        "{}",
        serde_json::to_string(&obj).expect("serialize run json")
    );
}

/// Snapshot the run environment for the JSON `conditions` block (best-effort; git/rustc are
/// `None` if the tool is absent). These gate whether two runs are comparable.
fn conditions() -> Conditions {
    Conditions {
        git_commit: cmd_capture("git", &["rev-parse", "--short", "HEAD"]),
        rustc: cmd_capture("rustc", &["--version"]),
        arch: std::env::consts::ARCH,
        os: std::env::consts::OS,
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    }
}

/// Best-effort stdout of `prog args…`, trimmed; `None` on any failure or empty output.
fn cmd_capture(prog: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

// ----- quality measurement (shared by relevance + fuzzy) ----------------------

/// One quality-eval engine row: latency samples from the single depth-[`KMAX`] timed pull, the
/// throughput over that pull, and recall/MRR/nDCG at each [`REPORT_KS`] cutoff (aligned to
/// `REPORT_KS` by index).
struct QualityRecord {
    engine: String,
    samples_ns: Vec<u64>,
    throughput_qps: f64,
    recall: Vec<f64>,
    mrr: Vec<f64>,
    ndcg: Vec<f64>,
}

/// Run one engine's quality measurement: warmup, then a single timed depth-[`KMAX`] pull per
/// query, returning the per-cutoff metrics record AND the raw depth-50 result lists (so the
/// caller can tag trifle's misses without re-searching).
fn measure_quality(
    engine: &dyn Engine,
    qtexts: &[&str],
    relevant: &[Vec<i64>],
    warmup: usize,
) -> (QualityRecord, Vec<Vec<i64>>) {
    let w = warmup.min(qtexts.len());
    if w > 0 {
        let _ = engine.search_batch(&qtexts[..w], KMAX);
    }
    let mut results = Vec::with_capacity(qtexts.len());
    let mut samples = Vec::with_capacity(qtexts.len());
    let wall = Instant::now();
    for q in qtexts {
        let t = Instant::now();
        let r = engine.search(q, KMAX);
        samples.push(t.elapsed().as_nanos() as u64);
        results.push(r);
    }
    let throughput_qps = throughput(qtexts.len(), wall.elapsed());
    let recall = REPORT_KS
        .iter()
        .map(|&k| set_recall_at_k(&results, relevant, k))
        .collect();
    let mrr = REPORT_KS
        .iter()
        .map(|&k| mrr_at_k(&results, relevant, k))
        .collect();
    let ndcg = REPORT_KS
        .iter()
        .map(|&k| ndcg_at_k(&results, relevant, k))
        .collect();
    let rec = QualityRecord {
        engine: engine.name().to_string(),
        samples_ns: samples,
        throughput_qps,
        recall,
        mrr,
        ndcg,
    };
    (rec, results)
}

/// Render a quality record as a latency line plus a recall/MRR/nDCG triple, all from the one
/// depth-50 pull.
fn render_quality_record(r: &QualityRecord) {
    let lat = Latency::from_durations(
        r.samples_ns
            .iter()
            .map(|&ns| Duration::from_nanos(ns))
            .collect(),
    );
    println!(
        "{:>18}  p50 {:>8} p90 {:>8} p95 {:>8} p99 {:>8} max {:>8}  ({:.0} q/s)",
        r.engine,
        fmt_dur(lat.p50()),
        fmt_dur(lat.p90()),
        fmt_dur(lat.p95()),
        fmt_dur(lat.p99()),
        fmt_dur(lat.max()),
        r.throughput_qps,
    );
    print_metric_row("recall", &r.recall);
    print_metric_row("MRR", &r.mrr);
    print_metric_row("nDCG", &r.ndcg);
}

/// Print one `@k` metric row aligned under the engine line (values indexed by [`REPORT_KS`]).
fn print_metric_row(label: &str, values: &[f64]) {
    print!("{:>18}  {label:<6}", "");
    for (k, v) in REPORT_KS.iter().zip(values) {
        print!("  @{k}:{v:.3}");
    }
    println!();
}

// ---- quality JSON schema ----

#[derive(Serialize)]
struct Kv {
    k: usize,
    value: f64,
}

/// Pair each [`REPORT_KS`] cutoff with its metric value for JSON.
fn kv_list(values: &[f64]) -> Vec<Kv> {
    REPORT_KS
        .iter()
        .zip(values)
        .map(|(&k, &value)| Kv { k, value })
        .collect()
}

#[derive(Serialize)]
struct QualityRecordJson<'a> {
    engine: &'a str,
    throughput_qps: f64,
    latency_ns: LatencyNs,
    recall_at_k: Vec<Kv>,
    mrr_at_k: Vec<Kv>,
    ndcg_at_k: Vec<Kv>,
    samples_ns: &'a [u64],
}

fn quality_record_json(r: &QualityRecord) -> QualityRecordJson<'_> {
    QualityRecordJson {
        engine: &r.engine,
        throughput_qps: r.throughput_qps,
        latency_ns: latency_ns(&r.samples_ns),
        recall_at_k: kv_list(&r.recall),
        mrr_at_k: kv_list(&r.mrr),
        ndcg_at_k: kv_list(&r.ndcg),
        samples_ns: &r.samples_ns,
    }
}

// ----- recall-miss tagging (shared) -------------------------------------------

/// View a token as `&str`. `Token: Borrow<str>`, but every type also `Borrow`s as itself, so a
/// bare `t.borrow()` is ambiguous; the `-> &str` return pins the str view.
fn token_str<T: Borrow<str>>(t: &T) -> &str {
    t.borrow()
}

/// The distinct trigram set of `s` under trifle's tokenizer (NFC + lowercase).
fn trigram_set(tok: &TrigramTokenizer, s: &str) -> HashSet<String> {
    tok.tokenize(s).map(|t| token_str(&t).to_string()).collect()
}

/// Distinct-trigram overlap between two strings under trifle's tokenizer — the raw count
/// trifle's overlap counter sees, before df-selection.
fn shared_trigrams(tok: &TrigramTokenizer, a: &str, b: &str) -> usize {
    trigram_set(tok, a)
        .intersection(&trigram_set(tok, b))
        .count()
}

/// `id -> text` over a corpus's docs, for resolving a label id to its (target) text.
fn text_index(corpus: &Corpus) -> HashMap<i64, &str> {
    corpus
        .docs
        .iter()
        .map(|d| (d.id, d.text.as_str()))
        .collect()
}

/// The effective match floor `m` for a run (trifle's default is 2, clamped ≥ 1).
fn effective_m(tuning: Tuning) -> usize {
    tuning.min_shared.unwrap_or(2).max(1) as usize
}

/// Where a recall miss's fix lives, tagged cheaply from the shared-trigram count between each
/// missed query and its target — no internals, no re-search.
#[derive(Default)]
struct MissTally {
    /// No shared trigrams: overlap could never surface the target. The pruner/tokenizer is the
    /// ceiling — the most concerning bucket.
    selection: usize,
    /// Shares trigrams, but fewer than the match floor `m` — points at `m` (the strictness
    /// dial), not the ranking.
    floor: usize,
    /// Shares ≥ `m` trigrams (cleared the raw overlap floor) but ranked past k — a ranking gap.
    /// On long multi-word queries trifle selects only the rarest tokens, so a few here may
    /// instead be selection-pruned; the *definitive* signals are the other two buckets.
    ranking: usize,
}

impl MissTally {
    fn record(&mut self, shared: usize, m: usize) {
        if shared == 0 {
            self.selection += 1;
        } else if shared < m {
            self.floor += 1;
        } else {
            self.ranking += 1;
        }
    }
    fn total(&self) -> usize {
        self.selection + self.floor + self.ranking
    }
    fn line(&self) -> String {
        format!(
            "misses={} (selection/no-overlap {}, below-floor/m {}, ranking {})",
            self.total(),
            self.selection,
            self.floor,
            self.ranking
        )
    }
}

/// Tag trifle's recall misses at depth `k`. `results`/`relevant` are trifle's, in query order;
/// `text_of` resolves a relevant id to its target text; `m` is the match floor.
fn tag_misses(
    qtexts: &[&str],
    relevant: &[Vec<i64>],
    results: &[Vec<i64>],
    k: usize,
    text_of: &HashMap<i64, &str>,
    m: usize,
) -> MissTally {
    let tok = TrigramTokenizer::new();
    let mut tally = MissTally::default();
    for ((got, rel), q) in results.iter().zip(relevant).zip(qtexts) {
        if rel.is_empty() {
            continue;
        }
        let topk: HashSet<i64> = got.iter().copied().take(k).collect();
        if rel.iter().any(|r| topk.contains(r)) {
            continue; // a hit, not a miss
        }
        let shared = rel
            .iter()
            .filter_map(|r| text_of.get(r))
            .map(|t| shared_trigrams(&tok, q, t))
            .max()
            .unwrap_or(0);
        tally.record(shared, m);
    }
    tally
}

// ----- relevance (recall + latency) -------------------------------------------

fn cmd_relevance(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "docs",
        "queries",
        "seed",
        "min-shared",
        "t-max",
        "weight-step",
        "filter",
        "warmup",
        "format",
    ])?;
    let skip = skipped_engines(&flags)?;

    let docs = flags.usize("docs", 50_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    let n = flags.usize("queries", 1000)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let warmup = flags.usize("warmup", 3)?;
    let tuning = tuning(&flags)?;
    let json_mode = match flags.str("format", "text").as_str() {
        "text" => false,
        "json" => true,
        other => return Err(format!("--format {other} (text|json)")),
    };

    let rel = corpus::msmarco_relevance(docs, n, seed).map_err(|e| format!("msmarco: {e}"))?;
    let corpus = &rel.corpus;
    if rel.queries.is_empty() {
        return Err("no scored queries (no qrel-relevant passage made it into the corpus)".into());
    }
    let qtexts: Vec<&str> = rel.queries.iter().map(|q| q.text.as_str()).collect();
    // The identical in-corpus label set scored for EVERY engine (the symmetry contract).
    let relevant: Vec<Vec<i64>> = rel.queries.iter().map(|q| q.relevant.clone()).collect();
    let scored = scored_queries(&relevant);

    // trifle (the subject) first, so we can tag its misses; then the baselines.
    let mut records: Vec<QualityRecord> = Vec::new();
    let mut trifle_misses: Option<MissTally> = None;
    if !skip.contains(ENGINE_TRIFLE) {
        let trifle = Trifle::build(corpus, tuning);
        let (rec, res) = measure_quality(&trifle, &qtexts, &relevant, warmup);
        let text_of = text_index(corpus);
        trifle_misses = Some(tag_misses(
            &qtexts,
            &relevant,
            &res,
            KMAX,
            &text_of,
            effective_m(tuning),
        ));
        records.push(rec);
    }
    if !skip.contains(ENGINE_FTS5_WORD) {
        match Fts5Word::build(corpus) {
            Some(e) => records.push(measure_quality(&e, &qtexts, &relevant, warmup).0),
            None => eprintln!("note: FTS5 (word) unavailable in the linked SQLite — skipping"),
        }
    }
    if !skip.contains(ENGINE_FTS5) {
        match Fts5::build(corpus, MatchMode::TrigramOr) {
            Some(e) => records.push(measure_quality(&e, &qtexts, &relevant, warmup).0),
            None => eprintln!("note: FTS5 (trigram) unavailable in the linked SQLite — skipping"),
        }
    }
    if !skip.contains(ENGINE_LIKE) {
        let like = Like::build(corpus);
        records.push(measure_quality(&like, &qtexts, &relevant, warmup).0);
    }

    if json_mode {
        let obj = QualityRunJson {
            command: "relevance",
            corpus: "msmarco-relevance",
            provenance: &corpus.provenance,
            docs: corpus.docs.len(),
            queries: qtexts.len(),
            scored_queries: scored,
            seed,
            k_max: KMAX,
            ks: REPORT_KS.to_vec(),
            conditions: conditions(),
            records: records.iter().map(quality_record_json).collect(),
        };
        println!(
            "{}",
            serde_json::to_string(&obj).expect("serialize relevance json")
        );
    } else {
        println!(
            "# relevance (recall/MRR/nDCG @ {REPORT_KS:?}) — {}",
            corpus.provenance
        );
        println!(
            "# docs={} scored-queries={scored} depth={KMAX} (single pull, sliced) — sparse qrels \
             (~1 relevant/query): recall@k UNDERSTATES true recall",
            corpus.docs.len()
        );
        println!(
            "# real paraphrased queries (no guaranteed substring). baseline = word BM25 \
             (canonical) + trigram-bm25 cousin + LIKE floor."
        );
        print_filter(&skip);
        for r in &records {
            render_quality_record(r);
        }
        if let Some(t) = trifle_misses {
            println!("# trifle {}", t.line());
        }
    }
    Ok(())
}

/// The quality-eval JSON object (relevance — fuzzy has its own per-edit grouping).
#[derive(Serialize)]
struct QualityRunJson<'a> {
    command: &'a str,
    corpus: &'a str,
    provenance: &'a str,
    docs: usize,
    queries: usize,
    scored_queries: usize,
    seed: u64,
    k_max: usize,
    ks: Vec<usize>,
    conditions: Conditions,
    records: Vec<QualityRecordJson<'a>>,
}

// ----- fuzzy (recall + latency) -----------------------------------------------

fn cmd_fuzzy(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "corpus",
        "docs",
        "queries",
        "edits",
        "seed",
        "min-shared",
        "t-max",
        "weight-step",
        "filter",
        "warmup",
        "format",
    ])?;
    let skip = skipped_engines(&flags)?;
    if skip.contains(ENGINE_TRIFLE) {
        return Err("trifle is the subject of the fuzzy eval and cannot be filtered out".into());
    }

    let corpus_key = flags.str("corpus", corpus::CORPUS_GEONAMES_CITIES);
    if corpus_key != corpus::CORPUS_GEONAMES_CITIES && corpus_key != corpus::CORPUS_GEONAMES_ALL {
        return Err(format!(
            "unknown --corpus {corpus_key} ({} | {})",
            corpus::CORPUS_GEONAMES_CITIES,
            corpus::CORPUS_GEONAMES_ALL
        ));
    }
    let docs = flags.usize("docs", 200_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    let n_targets = flags.usize("queries", 2000)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let warmup = flags.usize("warmup", 3)?;
    let tuning = tuning(&flags)?;
    let edit_counts: Vec<usize> = if flags.has("edits") {
        vec![flags.usize("edits", 1)?]
    } else {
        vec![1, 2]
    };
    let json_mode = match flags.str("format", "text").as_str() {
        "text" => false,
        "json" => true,
        other => return Err(format!("--format {other} (text|json)")),
    };

    let ent = corpus::geonames(&corpus_key, docs, n_targets, seed)
        .map_err(|e| format!("geonames: {e}"))?;
    let corpus = &ent.corpus;
    if ent.targets.is_empty() {
        return Err("no entities loaded (corpus empty)".into());
    }

    let trifle = Trifle::build(corpus, tuning); // the subject — always built
    let fts5 = if skip.contains(ENGINE_FTS5) {
        None
    } else {
        let e = Fts5::build(corpus, MatchMode::TrigramOr);
        if e.is_none() {
            eprintln!("note: FTS5 (trigram) unavailable in the linked SQLite — skipping");
        }
        e
    };
    let like = (!skip.contains(ENGINE_LIKE)).then(|| Like::build(corpus));
    let text_of = text_index(corpus);
    let m = effective_m(tuning);
    let density = near_distractor_density(&trifle, &ent.targets, KMAX);

    let tok = TrigramTokenizer::new();
    // Owned groups (their `QualityRecord`s outlive the per-edit loop so the borrowed JSON
    // view can be built once at the end).
    let mut groups: Vec<FuzzyGroup> = Vec::new();

    if !json_mode {
        println!(
            "# fuzzy (recall/MRR/nDCG @ {REPORT_KS:?}) — {}",
            corpus.provenance
        );
        println!(
            "# depth={KMAX} (single pull, sliced). baseline = FTS5 trigram-MATCH (OR-bag, bm25) \
             + LIKE floor — NOT bm25-phrase (~0 on typos by construction)."
        );
        println!(
            "# CAVEAT: entity-name fuzzy is a FAVORABLE regime (short, structured, low-paraphrase). \
             Strong recall here validates the fuzzy MACHINERY; it does NOT transfer to prose fuzzy."
        );
        println!(
            "# near-distractor density = {density:.3} (fraction of targets whose clean name \
             surfaces another indexed entity; low => trivially-easy run, recall inflated)."
        );
        print_filter(&skip);
    }

    for edits in edit_counts {
        let qs = query::fuzzy_queries(&ent.targets, edits, seed);
        if qs.is_empty() {
            if !json_mode {
                println!("{edits:>6}  (no queries generated)");
            }
            continue;
        }
        let qtexts: Vec<&str> = qs.iter().map(|q| q.text.as_str()).collect();
        // Singleton relevant sets — the same path as relevance (MRR is the reciprocal rank of
        // the one relevant entity).
        let relevant: Vec<Vec<i64>> = qs.iter().map(|q| vec![q.target]).collect();
        // Trigram survival: avg fraction of the clean name's trigrams the edits leave.
        let mut surv = 0.0f64;
        for q in &qs {
            let clean = trigram_set(&tok, &q.clean);
            if !clean.is_empty() {
                surv += shared_trigrams(&tok, &q.text, &q.clean) as f64 / clean.len() as f64;
            }
        }
        let survival = surv / qs.len() as f64;

        let mut records: Vec<QualityRecord> = Vec::new();
        let (tr_rec, tr_res) = measure_quality(&trifle, &qtexts, &relevant, warmup);
        records.push(tr_rec);
        if let Some(e) = &fts5 {
            records.push(measure_quality(e, &qtexts, &relevant, warmup).0);
        }
        if let Some(e) = &like {
            records.push(measure_quality(e, &qtexts, &relevant, warmup).0);
        }

        if json_mode {
            groups.push(FuzzyGroup {
                edits,
                queries: qtexts.len(),
                scored_queries: scored_queries(&relevant),
                trigram_survival: survival,
                records,
            });
        } else {
            println!(
                "## edits={edits}  queries={}  trigram-survival={survival:.3}",
                qtexts.len()
            );
            for r in &records {
                render_quality_record(r);
            }
            let tally = tag_misses(&qtexts, &relevant, &tr_res, KMAX, &text_of, m);
            println!("# trifle edits={edits}: {}", tally.line());
        }
    }

    if json_mode {
        let groups_json: Vec<FuzzyGroupJson> = groups
            .iter()
            .map(|g| FuzzyGroupJson {
                edits: g.edits,
                queries: g.queries,
                scored_queries: g.scored_queries,
                trigram_survival: g.trigram_survival,
                records: g.records.iter().map(quality_record_json).collect(),
            })
            .collect();
        let obj = FuzzyRunJson {
            command: "fuzzy",
            corpus: &corpus_key,
            provenance: &corpus.provenance,
            docs: corpus.docs.len(),
            seed,
            k_max: KMAX,
            ks: REPORT_KS.to_vec(),
            near_distractor_density: density,
            conditions: conditions(),
            groups: groups_json,
        };
        println!(
            "{}",
            serde_json::to_string(&obj).expect("serialize fuzzy json")
        );
    }
    Ok(())
}

/// One per-edit-count group of the fuzzy run (owned): the edit count, its diagnostics, and a
/// quality record per engine.
struct FuzzyGroup {
    edits: usize,
    queries: usize,
    scored_queries: usize,
    trigram_survival: f64,
    records: Vec<QualityRecord>,
}

/// The borrowed JSON view of a [`FuzzyGroup`].
#[derive(Serialize)]
struct FuzzyGroupJson<'a> {
    edits: usize,
    queries: usize,
    scored_queries: usize,
    trigram_survival: f64,
    records: Vec<QualityRecordJson<'a>>,
}

#[derive(Serialize)]
struct FuzzyRunJson<'a> {
    command: &'a str,
    corpus: &'a str,
    provenance: &'a str,
    docs: usize,
    seed: u64,
    k_max: usize,
    ks: Vec<usize>,
    near_distractor_density: f64,
    conditions: Conditions,
    groups: Vec<FuzzyGroupJson<'a>>,
}

/// Fraction of targets whose *clean* name surfaces ≥1 other indexed entity in trifle — i.e.
/// has a near-match distractor present. A low value means the run is trivially easy (no
/// confusables sampled) and the recall numbers are inflated.
fn near_distractor_density(trifle: &Trifle, targets: &[Entity], k: usize) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let mut with = 0usize;
    for t in targets {
        if trifle
            .search(&t.name, k.max(2))
            .iter()
            .any(|&id| id != t.id)
        {
            with += 1;
        }
    }
    with as f64 / targets.len() as f64
}

// ----- selsweep (selection-cost frontier) -------------------------------------

/// A corpus paired with its labeled queries: each query's text and its relevant doc-id set
/// (one id for single-label corpora, the qrel set for msmarco).
type LabeledCorpus = (Corpus, Vec<(String, Vec<i64>)>);

/// Build a labeled corpus at size `n` for the sweep. Each corpus is a different query/label
/// regime: `msmarco` (real dev queries + qrels, relevant-set label), `geonames-*` (entity name
/// + typos, single label), and `synthetic` (in-corpus snippet + typos, single label).
fn labeled_corpus(
    which: &str,
    n: usize,
    q: usize,
    edits: usize,
    seed: u64,
) -> Result<LabeledCorpus, String> {
    Ok(match which {
        "synthetic" => {
            let c = corpus::synthetic(n, seed);
            let qs = query::labeled_snippets(&c, q, edits, seed)
                .into_iter()
                .map(|(t, id)| (t, vec![id]))
                .collect();
            (c, qs)
        }
        "msmarco" => {
            let rel = corpus::msmarco_relevance(n, q, seed).map_err(|e| format!("msmarco: {e}"))?;
            let qs = rel
                .queries
                .into_iter()
                .map(|r| (r.text, r.relevant))
                .collect();
            (rel.corpus, qs)
        }
        corpus::CORPUS_GEONAMES_CITIES | corpus::CORPUS_GEONAMES_ALL => {
            let ent = corpus::geonames(which, n, q, seed).map_err(|e| format!("geonames: {e}"))?;
            let qs = query::fuzzy_queries(&ent.targets, edits.max(1), seed)
                .into_iter()
                .map(|fq| (fq.text, vec![fq.target]))
                .collect();
            (ent.corpus, qs)
        }
        other => {
            return Err(format!(
                "unknown --corpus {other} (geonames-all|geonames-cities|msmarco-relevance|synthetic)"
            ));
        }
    })
}

/// The `t_max` grid for the selection arm: `2, 4, …, max` (and `max` itself if the step misses
/// it). Coarse enough to plot the recall/Σdf frontier without a per-query blowup.
fn tmax_grid(max: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (2..=max).step_by(2).collect();
    if v.is_empty() {
        v.push(max.max(1));
    } else if *v.last().unwrap() != max {
        v.push(max);
    }
    v
}

/// The `df_budget` grid for the work-cap arm, as fractions of `N` (each becomes a `Σdf` cap).
/// Dense at the low end where the recall curve bends; `1.0` is effectively uncapped.
const SEL_FRACS: &[f64] = &[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];

/// One CSV/JSON row of the frontier: a fixed `(arm, knob)` measured at one recall cutoff `k`.
/// `sigma_df_*` and `lat_*_us` are per-`(arm, knob)` aggregates (constant across the `k` rows).
#[derive(Serialize)]
struct SelRow {
    arm: &'static str,
    knob: u64,
    n: usize,
    k: usize,
    recall: f64,
    sigma_df_p50: u64,
    sigma_df_p99: u64,
    lat_p50_us: f64,
    lat_p99_us: f64,
}

#[derive(Serialize)]
struct SelRunJson<'a> {
    command: &'a str,
    corpus: &'a str,
    provenance: &'a str,
    docs: usize,
    queries: usize,
    seed: u64,
    conditions: Conditions,
    rows: Vec<SelRow>,
}

fn cmd_selsweep(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "corpus",
        "docs",
        "queries",
        "edits",
        "seed",
        "min-shared",
        "weight-step",
        "max-tmax",
        "format",
    ])?;
    let n = flags.usize("docs", 100_000)?;
    if n == 0 {
        return Err("--docs must be >= 1".into());
    }
    let q = flags.usize("queries", 500)?;
    let edits = flags.usize("edits", 2)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let max_tmax = flags.usize("max-tmax", 20)?.max(2);
    // Accept the explicit `msmarco-relevance` alias alongside the geonames keys.
    let which = match flags.str("corpus", "msmarco-relevance").as_str() {
        "msmarco-relevance" | "msmarco" => "msmarco".to_string(),
        other => other.to_string(),
    };
    let json_mode = match flags.str("format", "csv").as_str() {
        "csv" => false,
        "json" => true,
        other => return Err(format!("--format {other} (csv|json)")),
    };

    // Fixed tuning: `t_max`/`df_budget` are the swept variables, so they are NOT taken from the
    // flags here; `min-shared`/`weight-step` are the held-constant strictness dials.
    let fixed = Tuning {
        min_shared: flags.opt_u32("min-shared")?,
        t_max: None,
        weight_step: flags.opt_f64("weight-step")?,
    };

    let (corpus, queries) = labeled_corpus(&which, n, q, edits, seed)?;
    let ndocs = corpus.docs.len();
    if queries.is_empty() {
        return Err("no queries generated".into());
    }
    let qtexts: Vec<&str> = queries.iter().map(|(t, _)| t.as_str()).collect();
    let relevant: Vec<Vec<i64>> = queries.iter().map(|(_, r)| r.clone()).collect();
    let provenance = corpus.provenance.clone();
    let trifle = Trifle::build(&corpus, fixed);

    let tmaxes = tmax_grid(max_tmax);
    eprintln!(
        "selsweep[{which}]: N={ndocs} queries={} depth={KMAX} — t_max {tmaxes:?} + df_budget {SEL_FRACS:?}",
        qtexts.len()
    );

    let mut rows: Vec<SelRow> = Vec::new();
    // Arm 1 — t_max: the selection-count cap.
    for &t in &tmaxes {
        let (results, lats, sigma) = sweep_run(&qtexts, |query| trifle.search_tmax(query, KMAX, t));
        rows.extend(sel_rows(
            "t_max", t as u64, ndocs, &results, &relevant, &lats, sigma,
        ));
    }
    // Arm 2 — df_budget: the Σdf work cap, as fractions of N.
    for &frac in SEL_FRACS {
        let budget = ((frac * ndocs as f64).round() as u64).max(1);
        let (results, lats, sigma) = sweep_run(&qtexts, |query| {
            trifle.search_df_budget(query, KMAX, budget)
        });
        rows.extend(sel_rows(
            "df_budget",
            budget,
            ndocs,
            &results,
            &relevant,
            &lats,
            sigma,
        ));
    }

    if json_mode {
        let obj = SelRunJson {
            command: "selsweep",
            corpus: &which,
            provenance: &provenance,
            docs: ndocs,
            queries: qtexts.len(),
            seed,
            conditions: conditions(),
            rows,
        };
        println!(
            "{}",
            serde_json::to_string(&obj).expect("serialize selsweep json")
        );
    } else {
        println!("arm,knob,N,k,recall,sigma_df_p50,sigma_df_p99,lat_p50_us,lat_p99_us");
        for r in &rows {
            println!(
                "{},{},{},{},{:.4},{},{},{:.2},{:.2}",
                r.arm,
                r.knob,
                r.n,
                r.k,
                r.recall,
                r.sigma_df_p50,
                r.sigma_df_p99,
                r.lat_p50_us,
                r.lat_p99_us,
            );
        }
    }
    Ok(())
}

/// Run one swept knob over every query under the work-done collector, returning the depth-50
/// result lists, the per-query latency samples (ns), and the captured Σdf samples.
fn sweep_run(
    qtexts: &[&str],
    search: impl Fn(&str) -> Vec<i64>,
) -> (Vec<Vec<i64>>, Vec<u64>, Vec<u64>) {
    let ((results, lats), sigma) = profile::capture(|| {
        let mut results = Vec::with_capacity(qtexts.len());
        let mut lats = Vec::with_capacity(qtexts.len());
        for q in qtexts {
            let t = Instant::now();
            let ids = search(q);
            lats.push(t.elapsed().as_nanos() as u64);
            results.push(ids);
        }
        (results, lats)
    });
    (results, lats, sigma)
}

/// Turn one `(arm, knob)` measurement into one [`SelRow`] per [`REPORT_KS`] cutoff (the Σdf and
/// latency aggregates are shared across the rows; recall varies by `k`).
#[allow(clippy::too_many_arguments)]
fn sel_rows(
    arm: &'static str,
    knob: u64,
    n: usize,
    results: &[Vec<i64>],
    relevant: &[Vec<i64>],
    lats: &[u64],
    sigma: Vec<u64>,
) -> Vec<SelRow> {
    let sd = Dist::new(sigma);
    let ld = Dist::new(lats.to_vec());
    let (sigma_df_p50, sigma_df_p99) = (sd.pct(50.0), sd.pct(99.0));
    let (lat_p50_us, lat_p99_us) = (ld.pct(50.0) as f64 / 1000.0, ld.pct(99.0) as f64 / 1000.0);
    REPORT_KS
        .iter()
        .map(|&k| SelRow {
            arm,
            knob,
            n,
            k,
            recall: set_recall_at_k(results, relevant, k),
            sigma_df_p50,
            sigma_df_p99,
            lat_p50_us,
            lat_p99_us,
        })
        .collect()
}

// ----- profile ----------------------------------------------------------------

fn cmd_profile(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    let mut allowed = CORPUS_OPTS.to_vec();
    allowed.extend(["queries", "k"]);
    flags.reject_unknown(&allowed)?;

    let (corpus, seed) = build_corpus(&flags)?;
    let n = flags.usize("queries", 2000)?;
    let k = flags.usize("k", 10)?;
    let tuning = tuning(&flags)?;

    let queries = query::perf_queries(&corpus, n, seed);
    if queries.is_empty() {
        return Err("no queries generated (corpus too small or documents too short)".into());
    }
    let qtexts: Vec<&str> = queries.iter().map(|q| q.text.as_str()).collect();
    let trifle = Trifle::build(&corpus, tuning);

    let (_, samples) = profile::capture(|| {
        for q in &qtexts {
            let _ = trifle.search(q, k);
        }
    });
    let dist = Dist::new(samples);

    let ndocs = corpus.docs.len();
    let nq = qtexts.len();
    let ns = dist.count();
    println!(
        "# profile — Σ(kept-posting cardinality) per query — {}",
        corpus.provenance
    );
    println!("# docs={ndocs} queries={nq} (samples={ns})");
    println!(
        "Σ-cardinality  p50 {} · p90 {} · p99 {} · max {} · mean {:.0}",
        dist.pct(50.0),
        dist.pct(90.0),
        dist.pct(99.0),
        dist.max(),
        dist.mean(),
    );
    println!("# Correlate with the p99 of `latency`: if the tail tracks this curve, the");
    println!("# residual is big-bitset AND/XOR cost (expected). If not, look at hydration.");
    Ok(())
}

// ----- fetch ------------------------------------------------------------------

fn cmd_fetch(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&["corpus"])?;
    let which = flags.str("corpus", "synthetic");
    corpus::prefetch(&which).map_err(|e| format!("fetch: {e}"))
}

// ----- dsweep (recall vs weight_step D) ---------------------------------------

/// Parse a comma-separated `f64` list (the `--steps` grid).
fn parse_steps(s: &str) -> Result<Vec<f64>, String> {
    let v: Vec<f64> = s
        .split(',')
        .map(|t| t.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("--steps: not a comma-separated float list: {s}"))?;
    if v.is_empty() {
        return Err("--steps is empty".into());
    }
    Ok(v)
}

fn cmd_dsweep(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "corpus",
        "docs",
        "queries",
        "edits",
        "seed",
        "min-shared",
        "t-max",
        "steps",
    ])?;
    let n = flags.usize("docs", 100_000)?;
    if n == 0 {
        return Err("--docs must be >= 1".into());
    }
    let q = flags.usize("queries", 1000)?;
    let edits = flags.usize("edits", 2)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let which = match flags.str("corpus", "msmarco-relevance").as_str() {
        "msmarco-relevance" | "msmarco" => "msmarco".to_string(),
        other => other.to_string(),
    };
    let steps = parse_steps(&flags.str("steps", "0.5,1.0,1.5,2.0,3.0"))?;
    // `weight_step` is the swept variable, so it is NOT part of the held-constant tuning.
    let fixed = Tuning {
        min_shared: flags.opt_u32("min-shared")?,
        t_max: flags.opt_u64("t-max")?.map(|v| v as usize),
        weight_step: None,
    };
    let (corpus, queries) = labeled_corpus(&which, n, q, edits, seed)?;
    if queries.is_empty() {
        return Err("no queries generated".into());
    }
    let ndocs = corpus.docs.len();
    let qtexts: Vec<&str> = queries.iter().map(|(t, _)| t.as_str()).collect();
    let relevant: Vec<Vec<i64>> = queries.iter().map(|(_, r)| r.clone()).collect();
    let trifle = Trifle::build(&corpus, fixed);

    println!(
        "# dsweep[{which}] — recall vs weight_step D — N={ndocs} queries={} depth={KMAX}",
        qtexts.len()
    );
    println!("D       recall@1  @5      @10     @50     mrr@10  ndcg@10");
    let mut best = (f64::NAN, f64::NEG_INFINITY); // (D, recall@10)
    for &d in &steps {
        let results: Vec<Vec<i64>> = qtexts
            .iter()
            .map(|qq| trifle.search_weight_step(qq, KMAX, d))
            .collect();
        let r10 = set_recall_at_k(&results, &relevant, 10);
        println!(
            "{d:<7.2} {:<9.4} {:<7.4} {:<7.4} {:<7.4} {:<7.4} {:.4}",
            set_recall_at_k(&results, &relevant, 1),
            set_recall_at_k(&results, &relevant, 5),
            r10,
            set_recall_at_k(&results, &relevant, 50),
            mrr_at_k(&results, &relevant, 10),
            ndcg_at_k(&results, &relevant, 10),
        );
        if r10 > best.1 {
            best = (d, r10);
        }
    }
    println!("# scored-queries={}", scored_queries(&relevant));
    match trifle.weight_step_hint() {
        Some(h) => println!(
            "# WeightStepHint suggests D≈{h:.2}; recall@10-optimal in grid is D={:.2} (recall@10 {:.4})",
            best.0, best.1
        ),
        None => println!("# WeightStepHint: none (no informative band-spread sample)"),
    }
    Ok(())
}

// ----- bench (trifle-overlap engine, synthetic bitmaps) -----------------------

fn cmd_bench(args: &[String]) -> Result<(), String> {
    use croaring::Bitmap;
    use rng::Rng;
    use trifle_overlap::Counter;

    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&["postings", "trials", "universe", "seed"])?;
    let k = flags.usize("postings", 10)?.max(1);
    let trials = flags.usize("trials", 300)?.max(1);
    let universe = flags.usize("universe", 1_000_000)?.max(1024);
    let seed = flags.u64("seed", DEFAULT_SEED)?;

    // A k-posting query: `overlap` ids shared by all postings (so some ids clear the floor),
    // the rest unique; each posting padded to `card`.
    fn synth(k: usize, card: usize, universe: usize, overlap: usize, rng: &mut Rng) -> Vec<Bitmap> {
        let shared: Vec<u32> = (0..overlap.min(card))
            .map(|_| rng.below(universe) as u32)
            .collect();
        (0..k)
            .map(|_| {
                let mut b = Bitmap::new();
                for &s in &shared {
                    b.add(s);
                }
                while (b.cardinality() as usize) < card {
                    b.add(rng.below(universe) as u32);
                }
                b
            })
            .collect()
    }
    let p = |samples: Vec<u64>, q: f64| fmt_dur(Duration::from_nanos(Dist::new(samples).pct(q)));

    println!("# bench — trifle-overlap build/walk on synthetic bitmaps (no SQLite)");
    println!("# postings(k)={k} trials={trials} universe={universe}");

    // Build cost vs posting cardinality — the op-count flatness curve (all-weight-1).
    println!("\n## build cost vs posting cardinality (all-weight-1)");
    println!("card     build_p50   build_p99");
    for &card in &[16usize, 128, 1024, 8192, 65536] {
        if card >= universe {
            continue;
        }
        let mut rng = Rng::new(seed ^ card as u64);
        let mut s = Vec::with_capacity(trials);
        for _ in 0..trials {
            let bms = synth(k, card, universe, 4, &mut rng);
            let t = Instant::now();
            std::hint::black_box(Counter::build(&bms, 1.0, 2));
            s.push(t.elapsed().as_nanos() as u64);
        }
        println!("{card:<8} {:<11} {}", p(s.clone(), 50.0), p(s, 99.0));
    }

    // Uniform card 4096 (→ all weight 1, the fast path) vs geometric cards (→ mixed weights).
    // Weights derive from df, so mixed weights require unequal cardinalities — the geometric set
    // is therefore also lighter total work. This contrasts a realistic skewed-df query against
    // the worst-case uniform-dense set; it is NOT weight-mode isolated at equal work.
    println!("\n## build: uniform card 4096 (all-weight-1) vs geometric cards (mixed-weight)");
    let mut rng = Rng::new(seed ^ 0x0A11);
    let (mut uni, mut mix) = (Vec::with_capacity(trials), Vec::with_capacity(trials));
    for _ in 0..trials {
        let bms = synth(k, 4096, universe, 4, &mut rng);
        let t = Instant::now();
        std::hint::black_box(Counter::build(&bms, 1.0, 2));
        uni.push(t.elapsed().as_nanos() as u64);
        let mixed: Vec<Bitmap> = (0..k)
            .map(|i| {
                let card = (4096usize >> (i % 4)).max(1);
                let mut b = Bitmap::new();
                while (b.cardinality() as usize) < card {
                    b.add(rng.below(universe) as u32);
                }
                b
            })
            .collect();
        let t = Instant::now();
        std::hint::black_box(Counter::build(&mixed, 1.0, 2));
        mix.push(t.elapsed().as_nanos() as u64);
    }
    println!("uniform (all-weight-1)    build_p50 {}", p(uni, 50.0));
    println!("geometric (mixed-weight)  build_p50 {}", p(mix, 50.0));

    // Shallow top-10 vs a full deep-pull drain (exercises the bucket walk + read_many).
    println!("\n## walk: shallow top-10 vs full drain (card 4096, overlap 64)");
    let mut rng = Rng::new(seed ^ 0x0BEE);
    let (mut shallow, mut full) = (Vec::with_capacity(trials), Vec::with_capacity(trials));
    for _ in 0..trials {
        let bms = synth(k, 4096, universe, 64, &mut rng);
        let c = Counter::build(&bms, 1.0, 2);
        let t = Instant::now();
        let mut w = c.walk();
        for _ in 0..10 {
            if c.advance(&mut w).is_none() {
                break;
            }
        }
        shallow.push(t.elapsed().as_nanos() as u64);
        let t = Instant::now();
        let mut w = c.walk();
        let mut n = 0usize;
        while c.advance(&mut w).is_some() {
            n += 1;
        }
        std::hint::black_box(n);
        full.push(t.elapsed().as_nanos() as u64);
    }
    println!("top-10      walk_p50 {}", p(shallow, 50.0));
    println!("full-drain  walk_p50 {}", p(full, 50.0));
    Ok(())
}

// ----- write (write-path throughput) ------------------------------------------

fn cmd_write(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&["corpus", "docs", "seed", "batch"])?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let docs = flags.usize("docs", 50_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    let which = flags.str("corpus", "synthetic");
    // The write path only needs the documents, so geonames is built for its `corpus` (one
    // throwaway target) just like synthetic/msmarco.
    let corpus = match which.as_str() {
        "synthetic" => corpus::synthetic(docs, seed),
        "msmarco" => corpus::msmarco(docs, seed).map_err(|e| format!("msmarco: {e}"))?,
        "geonames-cities" | "geonames-all" => {
            corpus::geonames(&which, docs, 1, seed)
                .map_err(|e| format!("geonames: {e}"))?
                .corpus
        }
        other => {
            return Err(format!(
                "unknown --corpus {other} (synthetic|msmarco|geonames-cities|geonames-all)"
            ));
        }
    };
    let batch = flags.usize("batch", 1000)?.max(1);
    let ndocs = corpus.docs.len();
    let open = |dir: &Path| -> Result<trifle::Index<TrigramTokenizer>, String> {
        let store = trifle::store::Sidecar::open(dir.join("w.db")).map_err(|e| e.to_string())?;
        trifle::Index::open(
            store,
            TrigramTokenizer::new(),
            trifle::Schema::flat(),
            trifle::Config::default(),
        )
        .map_err(|e| e.to_string())
    };
    let rate = |docs: usize, d: Duration| docs as f64 / d.as_secs_f64().max(1e-9);

    println!(
        "# write — {} — docs={ndocs} batch={batch}",
        corpus.provenance
    );

    // (a) incremental upsert + commit, batched.
    let dir_a = tempfile::tempdir().map_err(|e| e.to_string())?;
    let idx_a = open(dir_a.path())?;
    let t = Instant::now();
    {
        let mut w = idx_a.writer().map_err(|e| e.to_string())?;
        for (i, d) in corpus.docs.iter().enumerate() {
            w.upsert(d.id, &[("body", d.text.as_str())])
                .map_err(|e| e.to_string())?;
            if (i + 1) % batch == 0 {
                w.commit().map_err(|e| e.to_string())?;
            }
        }
        w.commit().map_err(|e| e.to_string())?;
    }
    let t_upsert = t.elapsed();
    println!(
        "incremental upsert  {} ({:.0} docs/s)",
        fmt_dur(t_upsert),
        rate(ndocs, t_upsert)
    );

    // (c) compact the incrementally-built index (it carries the full delta backlog).
    let backlog = idx_a.stats().map_err(|e| e.to_string())?.delta_backlog;
    let t = Instant::now();
    let cs = idx_a.compact().map_err(|e| e.to_string())?;
    println!(
        "compact             {} (delta_backlog {backlog} -> 0, tokens_folded {})",
        fmt_dur(t.elapsed()),
        cs.tokens_folded
    );

    // (b) rebuild from the same corpus (the bulk path).
    let dir_b = tempfile::tempdir().map_err(|e| e.to_string())?;
    let idx_b = open(dir_b.path())?;
    let docs = corpus
        .docs
        .iter()
        .map(|d| trifle::Document::new(d.id, vec![("body".to_string(), d.text.clone())]));
    let t = Instant::now();
    idx_b.rebuild(docs).map_err(|e| e.to_string())?;
    let t_rebuild = t.elapsed();
    println!(
        "rebuild             {} ({:.0} docs/s)",
        fmt_dur(t_rebuild),
        rate(ndocs, t_rebuild)
    );
    Ok(())
}
