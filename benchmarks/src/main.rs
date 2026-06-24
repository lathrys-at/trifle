//! `trifle-bench` — the design §10 benchmark harness.
//!
//! Two distinct benchmarks live here, because conflating them measures the wrong
//! thing (§10):
//!
//! - **`latency`** — the footrace (§10.1–10.2). Same corpus, same queries, trifle
//!   vs the in-process SQLite baselines (FTS5-trigram BM25, `LIKE` scan). Reports
//!   p50/p90/p99/max and throughput, serial *and* concurrent (the read pool's
//!   parallelism is a distinct axis). No labels needed.
//! - **`quality`** — recall@k of trifle vs the FTS5-trigram (BM25) baseline on
//!   typo-injected queries (§10.5), reported per edit-count.
//!
//! Plus two utilities:
//!
//! - **`profile`** — the §10.2 work-done instrument: the Σ(kept-posting cardinality)
//!   distribution, the quantity whose growth would break trifle's flatness claim.
//! - **`fetch`** — warm the pinned-corpus cache on a network machine before an
//!   offline run.
//!
//! Everything is driven by a master `--seed` so a run is byte-reproducible, and the
//! size knobs (`--docs`, `--queries`, …) let you trace the scaling sweep (§10.2).
//! See `benchmarks/README.md` for the matrix, the corpora, and how to run.

mod baselines;
mod corpus;
mod metrics;
mod profile;
mod query;
mod rng;

use std::collections::{HashMap, HashSet};
use std::process::ExitCode;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use baselines::{Engine, Fts5, Like, Trifle, Tuning};
use corpus::Corpus;
use metrics::{Dist, Latency, fmt_dur, recall_at_k, throughput};

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
        "quality" => cmd_quality(rest),
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
trifle-bench — design §10 benchmark harness

USAGE:
    trifle-bench <COMMAND> [OPTIONS]

COMMANDS:
    latency    Footrace: search latency + throughput, trifle vs in-process baselines
    quality    recall@k of trifle vs the FTS5-trigram (BM25) baseline, per edit-count
    profile    Σ(kept-posting cardinality) distribution — the §10.2 work-done curve
    fetch      Download + verify the pinned corpus assets into the cache (no bench)
    help       Show this message

CORPUS OPTIONS (latency, quality, profile):
    --corpus <synthetic|msmarco>  Corpus source                       [default: synthetic]
    --docs <N>                    Documents to index (the index size) [default: 50000]
    --seed <N>                    Master seed; drives BOTH corpus sampling and query
                                  generation. Same seed -> identical run. Accepts
                                  decimal or 0x-hex.                   [default: 0x5EED..]

SEARCH-TUNING OPTIONS (latency, quality, profile) — trifle only (§10.3):
    --min-shared <M>              Match floor m (shared rare tokens)   [default: engine]
    --breadth <B>                 Breadth budget B (recall/latency)    [default: engine]

ENGINE SELECTION (latency, quality):
    --filter <engine>             Skip an engine; repeatable. Engines:
                                  trifle, fts5-trigram-bm25, like-scan.
                                  e.g. --filter like-scan (slow on huge corpora)

LATENCY OPTIONS:
    --queries <N>                 Queries to run                       [default: 2000]
    --k <N>                       Top-k per query                      [default: 10]
    --warmup <N>                  Untimed warmup queries               [default: 200]
    --repeat <N>                  Measured passes (samples accumulate) [default: 1]
    --batched                     Issue all queries in ONE search_batch call (shares
                                  posting/frequency reads) instead of one search() each
    --concurrent <T>              Run trifle across T reader threads (the read-pool
                                  parallelism axis; baselines are serial-only) [default: 0]

QUALITY OPTIONS:
    --queries <N>                 Labeled queries per edit-count       [default: 1000]
    --k <N>                       Top-k recall cutoff                  [default: 10]
    --edits <N>                   Typos per query. Omit to sweep {0,1,2} (§10.3).

EXAMPLES:
    trifle-bench fetch --corpus synthetic
    trifle-bench latency --docs 100000 --queries 5000 --seed 42
    trifle-bench latency --docs 100000 --batched
    trifle-bench latency --docs 500000 --concurrent 8
    trifle-bench latency --docs 2000000 --filter like-scan
    trifle-bench quality --corpus msmarco --docs 100000
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
    })
}

const CORPUS_OPTS: &[&str] = &["corpus", "docs", "seed", "min-shared", "breadth"];

/// The engine identifiers accepted by `--filter`. These must match the strings each
/// engine returns from `baselines::Engine::name()`.
const ENGINE_TRIFLE: &str = "trifle";
const ENGINE_FTS5: &str = "fts5-trigram-bm25";
const ENGINE_LIKE: &str = "like-scan";
const ALL_ENGINES: [&str; 3] = [ENGINE_TRIFLE, ENGINE_FTS5, ENGINE_LIKE];

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
    if ALL_ENGINES.iter().all(|e| skip.contains(*e)) {
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
    if !skip.is_empty() {
        let mut names: Vec<&str> = skip.iter().map(String::as_str).collect();
        names.sort_unstable();
        println!("# filter: skipping {}", names.join(", "));
    }

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
        match Fts5::build(&corpus) {
            Some(fts5) => bench_engine(&fts5, &qtexts, k, warmup, repeat, batched),
            None => eprintln!("note: FTS5 trigram unavailable in the linked SQLite — skipping"),
        }
    }
    if !skip.contains(ENGINE_LIKE) {
        let like = Like::build(&corpus);
        bench_engine(&like, &qtexts, k, warmup, repeat, batched);
    }
    println!(
        "# (hidden axes — durability, footprint kind, update cost, semantics — in README §matrix)"
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

// ----- quality ----------------------------------------------------------------

fn cmd_quality(args: &[String]) -> Result<(), String> {
    let flags = Flags::parse(args, &[])?;
    let mut allowed = CORPUS_OPTS.to_vec();
    allowed.extend(["queries", "k", "edits", "filter"]);
    flags.reject_unknown(&allowed)?;

    let skip = skipped_engines(&flags)?;
    if skip.contains(ENGINE_TRIFLE) && skip.contains(ENGINE_FTS5) {
        return Err("both quality engines (trifle, fts5-trigram-bm25) filtered out".into());
    }

    let (corpus, seed) = build_corpus(&flags)?;
    let n = flags.usize("queries", 1000)?;
    let k = flags.usize("k", 10)?;
    let tuning = tuning(&flags)?;
    // A single --edits pins one count; otherwise sweep {0,1,2} (§10.3: report 1- and
    // 2-edit recall separately, with the exact-match floor as the 0-edit baseline).
    let edit_counts: Vec<usize> = if flags.has("edits") {
        vec![flags.usize("edits", 1)?]
    } else {
        vec![0, 1, 2]
    };

    let trifle = if skip.contains(ENGINE_TRIFLE) {
        None
    } else {
        Some(Trifle::build(&corpus, tuning))
    };
    let fts5 = if skip.contains(ENGINE_FTS5) {
        None
    } else {
        Fts5::build(&corpus)
    };
    if fts5.is_none() && !skip.contains(ENGINE_FTS5) {
        eprintln!("note: FTS5 trigram unavailable in the linked SQLite — BM25 column blank");
    }

    println!("# quality (recall@{k}) — {}", corpus.provenance);
    println!("# docs={} queries/edit={}", corpus.docs.len(), n);
    println!(
        "{:>6}  {:>8}  {:>10}  {:>10}",
        "edits", "queries", "trifle", "fts5-bm25"
    );
    for edits in edit_counts {
        let qs = query::quality_queries(&corpus, n, edits, seed);
        if qs.is_empty() {
            println!("{edits:>6}  {:>8}  (no queries generated)", 0);
            continue;
        }
        let labels: Vec<i64> = qs.iter().map(|q| q.source_doc).collect();
        let qtexts: Vec<&str> = qs.iter().map(|q| q.text.as_str()).collect();
        let recall = |e: &dyn Engine| recall_at_k(&e.search_many(&qtexts, k), &labels);
        let tr_s = trifle
            .as_ref()
            .map(|e| format!("{:.3}", recall(e)))
            .unwrap_or_else(|| "—".into());
        let fr_s = fts5
            .as_ref()
            .map(|e| format!("{:.3}", recall(e)))
            .unwrap_or_else(|| "—".into());
        println!("{edits:>6}  {:>8}  {tr_s:>10}  {fr_s:>10}", qtexts.len());
    }
    Ok(())
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
