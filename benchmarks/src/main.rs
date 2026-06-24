//! `trifle-bench` — the benchmark harness.
//!
//! Three distinct benchmarks live here, kept separate because conflating them measures
//! the wrong thing:
//!
//! - **`latency`** — search latency + throughput. Same corpus, same queries, trifle
//!   vs the in-process SQLite baselines (FTS5-trigram BM25, `LIKE` scan). Reports
//!   p50/p90/p99/max and throughput, serial *and* concurrent (the read pool's
//!   parallelism is a distinct axis). No labels needed.
//! - **`relevance`** — recall@k on MS MARCO real dev queries + qrels, vs a word-level
//!   BM25 baseline (and the trigram-bm25 cousin). The paraphrase case: real queries
//!   share no guaranteed substring with their answer.
//! - **`fuzzy`** — recall@k on entity-name + injected-edit queries over a GeoNames
//!   corpus, vs FTS5 trigram-MATCH and the LIKE floor (never bm25-phrase). The typo
//!   case; reports 1- vs 2-edit recall separately.
//!
//! Both recall evals tag each miss selection / floor / ranking, to say whether a fix
//! lives in the pruner/`m` or the ranker.
//!
//! Plus two utilities:
//!
//! - **`profile`** — the work-done instrument: the Σ(kept-posting cardinality)
//!   distribution, the quantity whose growth with corpus size would flatten trifle's
//!   latency advantage.
//! - **`fetch`** — warm the pinned-corpus cache on a network machine before an
//!   offline run.
//!
//! Everything is driven by a master `--seed` so a run is byte-reproducible, and the
//! size knobs (`--docs`, `--queries`, …) let you trace how cost scales with corpus
//! size. See `benchmarks/README.md` for the corpora and how to run.

mod baselines;
mod corpus;
mod metrics;
mod profile;
mod query;
mod rng;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::process::ExitCode;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use baselines::{Engine, Fts5, Fts5Word, Like, MatchMode, Trifle, Tuning};
use corpus::{Corpus, Entity};
use metrics::{Dist, Latency, fmt_dur, scored_queries, set_recall_at_k, throughput};
use trifle::Effort;
use trifle::tokenize::{Tokenizer, TrigramTokenizer};

/// The default master seed (`0x5EED…`). Override with `--seed`.
const DEFAULT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

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
        "profile" => cmd_profile(rest),
        "ranksweep" => cmd_ranksweep(rest),
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
    latency    Search latency + throughput, trifle vs in-process baselines
    relevance  MS MARCO real dev queries+qrels: set-recall@k vs word BM25 (+trigram)
    fuzzy      Entity name+edit recall vs FTS5 trigram-MATCH / LIKE, per edit-count
    profile    Σ(kept-posting cardinality) distribution — the work-done curve
    ranksweep  recall@k vs rerank-pool depth CSV (backend for tools/calibrate_pool.py)
    fetch      Download + verify the pinned corpus assets into the cache (no bench)
    help       Show this message

COMMON OPTIONS:
    --docs <N>                    Index size. For `relevance`, the distractor target
                                  (answers are always indexed) [default: 50000, fuzzy 200000]
    --queries <N>                 Queries (relevance: sampled real queries; fuzzy: target
                                  entities) [default: latency 2000, relevance 1000, fuzzy 2000]
    --k <N>                       Top-k cutoff [default: 10]
    --seed <N>                    Master seed (decimal or 0x-hex); fixes corpus + query
                                  sampling for a byte-reproducible run [default: 0x5EED..]

SEARCH-TUNING (trifle only):
    --min-shared <M>              Match floor m (shared rare tokens) [default: engine]
    --breadth <B>                 Breadth budget B (recall/latency) [default: engine]
    --effort <none|low|medium|high|max>  Rerank effort (pool depth c·√(kN) + the BM25
                                  precision tier). Omit to use trifle's default (Medium)

ENGINE SELECTION (latency, relevance, fuzzy):
    --filter <engine>             Skip an engine; repeatable. Engines: trifle,
                                  fts5-trigram-bm25, fts5-word-bm25, like-scan.
                                  e.g. --filter like-scan (slow on huge corpora)

LATENCY:
    --corpus <synthetic|msmarco>  Corpus source [default: synthetic]
    --warmup <N>                  Untimed warmup queries [default: 200]
    --repeat <N>                  Measured passes (samples accumulate) [default: 1]
    --batched                     One search_batch call (shares posting/frequency reads)
    --concurrent <T>              Run trifle across T reader threads (read-pool axis) [default: 0]

FUZZY:
    --corpus <geonames-cities|geonames-all>   Entity corpus [default: geonames-cities]
    --edits <N>                   Typos per query. Omit to run {1, 2} separately.

PROFILE:
    --corpus <synthetic|msmarco>  Corpus source [default: synthetic]

EXAMPLES:
    trifle-bench fetch --corpus geonames-cities
    trifle-bench latency --docs 100000 --queries 5000 --seed 42
    trifle-bench latency --docs 2000000 --filter like-scan
    trifle-bench relevance --docs 100000 --queries 2000
    trifle-bench fuzzy --corpus geonames-cities --edits 1
    trifle-bench profile --docs 1000000
";

// ----- argument parsing -------------------------------------------------------

/// A parsed flag set: `--key value` / `--key=value` valued flags and `--flag`
/// booleans. Deliberately tiny and std-only (the harness vendors no arg crate, in
/// keeping with its hand-rolled RNG and shell-out asset fetch).
struct Flags {
    /// Values per flag, accumulated in order so a flag may repeat (e.g. `--filter`).
    /// Scalar accessors take the last occurrence (last-wins); `values` returns all.
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

    /// Reject any flag not in `allowed` — a typo'd knob is an error, not a silent
    /// default.
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

/// Build the corpus named by `--corpus`, sized by `--docs`, seeded by `--seed`.
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

fn tuning(flags: &Flags) -> Result<Tuning, String> {
    Ok(Tuning {
        min_shared: flags.opt_u32("min-shared")?,
        breadth: flags.opt_u64("breadth")?,
        effort: parse_effort(flags)?,
    })
}

/// Parse `--effort <none|low|medium|high|max>`; `None` (unset) leaves trifle's default.
fn parse_effort(flags: &Flags) -> Result<Option<Effort>, String> {
    if !flags.has("effort") {
        return Ok(None);
    }
    Ok(Some(match flags.str("effort", "").as_str() {
        "none" => Effort::None,
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "max" => Effort::Max,
        other => return Err(format!("--effort {other} (none|low|medium|high|max)")),
    }))
}

const CORPUS_OPTS: &[&str] = &["corpus", "docs", "seed", "min-shared", "breadth", "effort"];

/// The engine identifiers accepted by `--filter`. These must match the strings each
/// engine returns from `baselines::Engine::name()`. Not every command runs every
/// engine (latency/fuzzy use the trigram FTS5; relevance adds the word-level BM25).
const ENGINE_TRIFLE: &str = "trifle";
const ENGINE_FTS5: &str = "fts5-trigram-bm25";
const ENGINE_FTS5_WORD: &str = "fts5-word-bm25";
const ENGINE_LIKE: &str = "like-scan";
const ALL_ENGINES: [&str; 4] = [ENGINE_TRIFLE, ENGINE_FTS5, ENGINE_FTS5_WORD, ENGINE_LIKE];

/// The set of engines to skip, collected from repeated `--filter <engine>` flags.
/// Each value must name a known engine — a typo is an error, not a silent no-op (the
/// same strictness `reject_unknown` applies to option *names*).
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

// ----- latency ----------------------------------------------------------------

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
    ]);
    flags.reject_unknown(&allowed)?;

    let skip = skipped_engines(&flags)?;
    // latency runs the three speed-comparison engines (word-level BM25 is relevance-only).
    if [ENGINE_TRIFLE, ENGINE_FTS5, ENGINE_LIKE]
        .iter()
        .all(|e| skip.contains(*e))
    {
        return Err("all engines filtered out — nothing to run".into());
    }

    let (corpus, seed) = build_corpus(&flags)?;
    let n = flags.usize("queries", 2000)?;
    let k = flags.usize("k", 10)?;
    let warmup = flags.usize("warmup", 200)?;
    let repeat = flags.usize("repeat", 1)?.max(1);
    let batched = flags.flag("batched");
    let concurrent = flags.usize("concurrent", 0)?;
    let tuning = tuning(&flags)?;

    let queries = query::perf_queries(&corpus, n, seed);
    if queries.is_empty() {
        return Err("no queries generated (corpus too small or documents too short)".into());
    }
    let qtexts: Vec<&str> = queries.iter().map(|q| q.text.as_str()).collect();

    // Typo mix of the generated set (0/1/2 edits), so the run records what was asked.
    let mut mix = [0usize; 3];
    for q in &queries {
        if let Some(slot) = mix.get_mut(q.edits) {
            *slot += 1;
        }
    }
    let ndocs = corpus.docs.len();
    let nq = qtexts.len();
    println!("# latency — {}", corpus.provenance);
    println!(
        "# docs={ndocs} queries={nq} (0-edit={} 1-edit={} 2-edit={}) k={k} warmup={} repeat={repeat}",
        mix[0],
        mix[1],
        mix[2],
        warmup.min(nq),
    );
    print_filter(&skip);

    if concurrent > 1 {
        if skip.contains(ENGINE_TRIFLE) {
            return Err("concurrent mode runs trifle only, but it is filtered out".into());
        }
        println!("# mode=concurrent threads={concurrent} (trifle only — the read-pool axis)");
        bench_concurrent(&corpus, &qtexts, k, concurrent, tuning, warmup);
        return Ok(());
    }

    println!("# mode={}", if batched { "batched" } else { "serial" });
    if !skip.contains(ENGINE_TRIFLE) {
        let trifle = Trifle::build(&corpus, tuning);
        bench_engine(&trifle, &qtexts, k, warmup, repeat, batched);
    }
    if !skip.contains(ENGINE_FTS5) {
        // Phrase mode: the latency baseline is unchanged by the recall realignment.
        match Fts5::build(&corpus, MatchMode::Phrase) {
            Some(fts5) => bench_engine(&fts5, &qtexts, k, warmup, repeat, batched),
            None => eprintln!("note: FTS5 trigram unavailable in the linked SQLite — skipping"),
        }
    }
    if !skip.contains(ENGINE_LIKE) {
        let like = Like::build(&corpus);
        bench_engine(&like, &qtexts, k, warmup, repeat, batched);
    }
    println!(
        "# (latency is one axis; durability, footprint, update cost, and semantics differ \
         per engine — see benchmarks/README.md)"
    );
    Ok(())
}

fn bench_engine(
    engine: &dyn Engine,
    queries: &[&str],
    k: usize,
    warmup: usize,
    repeat: usize,
    batched: bool,
) {
    let w = warmup.min(queries.len());
    if w > 0 {
        let _ = engine.search_many(&queries[..w], k);
    }
    if batched {
        let mut best = Duration::MAX;
        for _ in 0..repeat {
            let t = Instant::now();
            let _ = engine.search_many(queries, k);
            best = best.min(t.elapsed());
        }
        println!(
            "{:>18}  batched   {} queries in {} ({:.0} q/s)",
            engine.name(),
            queries.len(),
            fmt_dur(best),
            throughput(queries.len(), best),
        );
    } else {
        let mut samples = Vec::with_capacity(queries.len() * repeat);
        let wall = Instant::now();
        for _ in 0..repeat {
            for q in queries {
                let t = Instant::now();
                let _ = engine.search(q, k);
                samples.push(t.elapsed());
            }
        }
        let total = wall.elapsed();
        let lat = Latency::from_durations(samples);
        println!(
            "{:>18}  serial    p50 {:>8} p90 {:>8} p99 {:>8} max {:>8} ({:.0} q/s)",
            engine.name(),
            fmt_dur(lat.p50()),
            fmt_dur(lat.p90()),
            fmt_dur(lat.p99()),
            fmt_dur(lat.max()),
            throughput(queries.len() * repeat, total),
        );
    }
}

/// Concurrent throughput for trifle: `threads` readers share one `&Index`, each
/// running an interleaved shard of the query set. The read pool is the thing under
/// test, so this is trifle-only (the single-`Connection` baselines have no analogue).
fn bench_concurrent(
    corpus: &Corpus,
    queries: &[&str],
    k: usize,
    threads: usize,
    tuning: Tuning,
    warmup: usize,
) {
    let trifle = Trifle::build(corpus, tuning);
    let per_thread_warmup = (warmup / threads).min(queries.len());

    // A start gate sized for the workers + this thread. Each worker first warms its
    // own pooled reader connection (the pool creates one lazily per caller), then
    // waits on the gate; the gate opens for everyone at one instant — no spawn drift,
    // so the readers are genuinely concurrent for the whole measured window. The
    // clock starts the moment the gate opens, and the scope's implicit join blocks
    // until the last reader finishes, so the window is release -> all-done.
    let gate = Barrier::new(threads + 1);
    let mut started = Instant::now(); // reassigned at the gate; placeholder for the borrow
    std::thread::scope(|scope| {
        for t in 0..threads {
            let trifle = &trifle;
            let gate = &gate;
            scope.spawn(move || {
                for j in 0..per_thread_warmup {
                    let _ = trifle.search(queries[(t + j) % queries.len()], k);
                }
                gate.wait(); // every reader released together
                let mut i = t;
                while i < queries.len() {
                    let _ = trifle.search(queries[i], k);
                    i += threads;
                }
            });
        }
        gate.wait(); // opens the gate; the instant after is the true start
        started = Instant::now();
    });
    let elapsed = started.elapsed();
    println!(
        "{:>18}  conc({:>2})  {} queries in {} ({:.0} q/s aggregate)",
        "trifle",
        threads,
        queries.len(),
        fmt_dur(elapsed),
        throughput(queries.len(), elapsed),
    );
}

// ----- recall evals (shared) --------------------------------------------------

/// View a token as `&str`. `Token: Borrow<str>`, but every type also `Borrow`s as
/// itself, so a bare `t.borrow()` is ambiguous; the `-> &str` return pins the str view.
fn token_str<T: Borrow<str>>(t: &T) -> &str {
    t.borrow()
}

/// The distinct trigram set of `s` under trifle's tokenizer (NFC + lowercase).
fn trigram_set(tok: &TrigramTokenizer, s: &str) -> HashSet<String> {
    tok.tokenize(s).map(|t| token_str(&t).to_string()).collect()
}

/// Distinct-trigram overlap between two strings under trifle's tokenizer — the raw
/// count trifle's overlap counter sees, before df-selection.
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

/// Where a recall miss's fix lives, tagged cheaply from the shared-trigram count
/// between each missed query and its target — no internals, no re-search.
#[derive(Default)]
struct MissTally {
    /// No shared trigrams: overlap could never surface the target. The pruner/tokenizer
    /// is the ceiling — the most concerning bucket.
    selection: usize,
    /// Shares trigrams, but fewer than the match floor `m` — points at `m`/`B` (the
    /// strictness dials), not the ranker.
    floor: usize,
    /// Shares ≥ `m` trigrams (cleared the raw overlap floor) but ranked past k — a
    /// ranking gap (better `Ranker` territory). On long multi-word queries trifle
    /// selects only the rarest tokens, so a few here may instead be selection-pruned;
    /// the *definitive* signals are the other two buckets.
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

/// Tag trifle's recall misses. `results`/`relevant` are trifle's, in query order;
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

/// Score one labeled engine column against the *shared* label set (the symmetry
/// contract: identical `relevant`, identical `k` for every engine).
fn recall_col<E: Engine>(engine: &E, qtexts: &[&str], relevant: &[Vec<i64>], k: usize) -> f64 {
    set_recall_at_k(&engine.search_many(qtexts, k), relevant, k)
}

// ----- relevance --------------------------------------------------------------

fn cmd_relevance(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "docs",
        "queries",
        "k",
        "seed",
        "min-shared",
        "breadth",
        "filter",
        "effort",
    ])?;
    let skip = skipped_engines(&flags)?;

    let docs = flags.usize("docs", 50_000)?;
    if docs == 0 {
        return Err("--docs must be >= 1".into());
    }
    let n = flags.usize("queries", 1000)?;
    let k = flags.usize("k", 10)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let tuning = tuning(&flags)?;

    let rel = corpus::msmarco_relevance(docs, n, seed).map_err(|e| format!("msmarco: {e}"))?;
    let corpus = &rel.corpus;
    if rel.queries.is_empty() {
        return Err("no scored queries (no qrel-relevant passage made it into the corpus)".into());
    }
    let qtexts: Vec<&str> = rel.queries.iter().map(|q| q.text.as_str()).collect();
    // The identical in-corpus label set scored for EVERY engine (the symmetry contract).
    let relevant: Vec<Vec<i64>> = rel.queries.iter().map(|q| q.relevant.clone()).collect();

    println!("# relevance (set-recall@{k}) — {}", corpus.provenance);
    println!(
        "# docs={} scored-queries={} (sparse qrels ~1 relevant/query: recall@k UNDERSTATES true recall)",
        corpus.docs.len(),
        scored_queries(&relevant)
    );
    println!(
        "# real paraphrased queries (no guaranteed substring). baseline = word BM25 (canonical) + trigram-bm25 cousin."
    );
    print_filter(&skip);
    println!("{:>20}  {:>10}", "engine", "recall");

    // trifle (the subject) — also drives the miss breakdown.
    if !skip.contains(ENGINE_TRIFLE) {
        let trifle = Trifle::build(corpus, tuning);
        let res = trifle.search_many(&qtexts, k);
        println!(
            "{:>20}  {:>10.3}",
            ENGINE_TRIFLE,
            set_recall_at_k(&res, &relevant, k)
        );
        let text_of = text_index(corpus);
        let tally = tag_misses(&qtexts, &relevant, &res, k, &text_of, effective_m(tuning));
        println!("# trifle {}", tally.line());
    }
    // Canonical word-level BM25.
    if !skip.contains(ENGINE_FTS5_WORD) {
        match Fts5Word::build(corpus) {
            Some(e) => println!(
                "{:>20}  {:>10.3}",
                ENGINE_FTS5_WORD,
                recall_col(&e, &qtexts, &relevant, k)
            ),
            None => eprintln!("note: FTS5 (word) unavailable in the linked SQLite"),
        }
    }
    // Same-tokenization trigram BM25 (OR-bag).
    if !skip.contains(ENGINE_FTS5) {
        match Fts5::build(corpus, MatchMode::TrigramOr) {
            Some(e) => println!(
                "{:>20}  {:>10.3}",
                ENGINE_FTS5,
                recall_col(&e, &qtexts, &relevant, k)
            ),
            None => eprintln!("note: FTS5 (trigram) unavailable in the linked SQLite"),
        }
    }
    Ok(())
}

// ----- fuzzy ------------------------------------------------------------------

fn cmd_fuzzy(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "corpus",
        "docs",
        "queries",
        "k",
        "edits",
        "seed",
        "min-shared",
        "breadth",
        "filter",
        "effort",
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
    let k = flags.usize("k", 10)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let tuning = tuning(&flags)?;
    let edit_counts: Vec<usize> = if flags.has("edits") {
        vec![flags.usize("edits", 1)?]
    } else {
        vec![1, 2]
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
        Fts5::build(corpus, MatchMode::TrigramOr)
    };
    if fts5.is_none() && !skip.contains(ENGINE_FTS5) {
        eprintln!("note: FTS5 (trigram) unavailable in the linked SQLite — column blank");
    }
    let like = (!skip.contains(ENGINE_LIKE)).then(|| Like::build(corpus));
    let text_of = text_index(corpus);
    let m = effective_m(tuning);

    println!("# fuzzy (recall@{k}) — {}", corpus.provenance);
    println!(
        "# baseline = FTS5 trigram-MATCH (OR-bag, bm25) + LIKE floor — NOT bm25-phrase (that scores ~0 on typos by construction)."
    );
    println!(
        "# CAVEAT: entity-name fuzzy is a FAVORABLE regime (short, structured, low-paraphrase). \
         Strong recall here validates the fuzzy MACHINERY; it does NOT transfer to prose fuzzy — \
         that is the `relevance` eval's job."
    );
    let density = near_distractor_density(&trifle, &ent.targets, k);
    println!(
        "# near-distractor density = {density:.3} (fraction of targets whose clean name surfaces \
         another indexed entity; low => trivially-easy run, recall inflated)."
    );
    print_filter(&skip);
    println!(
        "{:>6}  {:>8}  {:>9}  {:>10}  {:>10}  {:>10}",
        "edits", "queries", "survival", "trifle", "fts5-tri", "like"
    );

    let tok = TrigramTokenizer::new();
    for edits in edit_counts {
        let qs = query::fuzzy_queries(&ent.targets, edits, seed);
        if qs.is_empty() {
            println!("{edits:>6}  (no queries generated)");
            continue;
        }
        let qtexts: Vec<&str> = qs.iter().map(|q| q.text.as_str()).collect();
        // Singleton relevant sets — the same set-recall@k path as relevance.
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

        let tr_res = trifle.search_many(&qtexts, k);
        let tr = set_recall_at_k(&tr_res, &relevant, k);
        let ft_s = fts5
            .as_ref()
            .map(|e| format!("{:.3}", recall_col(e, &qtexts, &relevant, k)))
            .unwrap_or_else(|| "—".into());
        let lk_s = like
            .as_ref()
            .map(|e| format!("{:.3}", recall_col(e, &qtexts, &relevant, k)))
            .unwrap_or_else(|| "—".into());
        println!(
            "{edits:>6}  {:>8}  {survival:>9.3}  {tr:>10.3}  {ft_s:>10}  {lk_s:>10}",
            qtexts.len()
        );
        let tally = tag_misses(&qtexts, &relevant, &tr_res, k, &text_of, m);
        println!("# trifle edits={edits}: {}", tally.line());
    }
    Ok(())
}

/// Fraction of targets whose *clean* name surfaces ≥1 other indexed entity in trifle —
/// i.e. has a near-match distractor present. A low value means the run is trivially easy
/// (no confusables sampled) and the recall numbers are inflated.
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

/// Print the `# filter: skipping …` line when any engine is filtered out.
fn print_filter(skip: &HashSet<String>) {
    if !skip.is_empty() {
        let mut s: Vec<&str> = skip.iter().map(String::as_str).collect();
        s.sort_unstable();
        println!("# filter: skipping {}", s.join(", "));
    }
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

// ----- ranksweep (rerank-pool calibration backend) ----------------------------

/// A log-spaced pool-depth grid `1 … max` (≈×1.5 steps, dense at the low end where the
/// recall curve bends), for the calibration sweep.
fn pool_grid(max: usize) -> Vec<usize> {
    let mut v = vec![1usize];
    while *v.last().unwrap() < max {
        let last = *v.last().unwrap();
        let next = ((last as f64 * 1.5).round() as usize).max(last + 1);
        v.push(next.min(max));
    }
    v
}

/// The measurement backend for the rerank-pool calibration (`tools/calibrate_pool.py`).
///
/// Emits the recall@k vs rerank-pool-depth matrix for one `(corpus, --docs N, --queries,
/// --seed)` as CSV to stdout (`N,edits,pool,k,queries,recall`). Builds the index once,
/// then for each pool depth reranks exactly the top-`pool` overlap candidates (via
/// [`Trifle::search_pool`], which pins the pool with `Effort::None` and the explicit
/// BM25 reranker) — so recall@k for every `k <= pool` falls out of one pass. The labels:
/// `synthetic`/`geonames` carry a single relevant id (snippet/name + typos), `msmarco`
/// the qrel relevant-set. See `tools/README.md` for the model and how the constants fall
/// out of this matrix.
fn cmd_ranksweep(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&[
        "corpus",
        "docs",
        "queries",
        "edits",
        "seed",
        "min-shared",
        "breadth",
        "max-pool",
    ])?;
    let n = flags.usize("docs", 100_000)?;
    if n == 0 {
        return Err("--docs must be >= 1".into());
    }
    let q = flags.usize("queries", 500)?;
    let edits = flags.usize("edits", 2)?;
    let seed = flags.u64("seed", DEFAULT_SEED)?;
    let tuning = tuning(&flags)?;
    let which = flags.str("corpus", "synthetic");

    // Log-spaced pool depths up to `--max-pool` (resolution dense at the low end), capped
    // to the corpus size at run time. Raise it past the default 2048 to push the recall
    // ceiling at very large N (where 2048 hasn't saturated — see tools/README.md caveats).
    let max_pool = flags.usize("max-pool", 2048)?.max(1);
    let pools = pool_grid(max_pool);
    const KS: &[usize] = &[1, 5, 10, 20, 50, 100];

    // (corpus, labeled queries). Each corpus yields a different query/label regime, so
    // the calibration can check the p(k,N) law isn't an artifact of any one of them:
    //  - `synthetic`         — snippet+typo queries (single label); scales freely.
    //  - `msmarco`           — real dev queries + qrels (relevant-set label); real prose.
    //  - `geonames-cities/-all` — entity name + typos (single label); short structured text.
    let (corpus, queries): (Corpus, Vec<(String, Vec<i64>)>) = match which.as_str() {
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
            let ent = corpus::geonames(&which, n, q, seed).map_err(|e| format!("geonames: {e}"))?;
            let qs = query::fuzzy_queries(&ent.targets, edits.max(1), seed)
                .into_iter()
                .map(|fq| (fq.text, vec![fq.target]))
                .collect();
            (ent.corpus, qs)
        }
        other => {
            return Err(format!(
                "unknown --corpus {other} (synthetic|msmarco|geonames-cities|geonames-all)"
            ));
        }
    };
    let ndocs = corpus.docs.len();
    if queries.is_empty() {
        return Err("no queries generated".into());
    }
    let qn = queries.len();
    let trifle = Trifle::build(&corpus, tuning);
    eprintln!("ranksweep[{which}]: N={ndocs} queries={qn} edits={edits} — sweeping pools…");

    for &pool in &pools {
        if pool > ndocs {
            continue;
        }
        let mut hits = vec![0usize; KS.len()];
        for (text, relevant) in &queries {
            let ids = trifle.search_pool(text, pool); // top-`pool` overlap, BM25-reranked
            for (ki, &k) in KS.iter().enumerate() {
                if k <= pool && ids.iter().take(k).any(|id| relevant.contains(id)) {
                    hits[ki] += 1;
                }
            }
        }
        for (ki, &k) in KS.iter().enumerate() {
            if k <= pool {
                println!(
                    "{ndocs},{edits},{pool},{k},{qn},{:.4}",
                    hits[ki] as f64 / qn as f64
                );
            }
        }
    }
    Ok(())
}

// ----- fetch ------------------------------------------------------------------

fn cmd_fetch(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    flags.reject_unknown(&["corpus"])?;
    let which = flags.str("corpus", "synthetic");
    corpus::prefetch(&which).map_err(|e| format!("fetch: {e}"))
}
