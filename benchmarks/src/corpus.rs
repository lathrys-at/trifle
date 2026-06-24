//! Real-corpus loading, following shrike's asset pattern: a **pinned source**
//! (`sources/*.json`: url + sha256 + attribution) is committed, the **bytes are
//! never committed** — they download on demand into a gitignored cache and are
//! hash-verified — and an **offline fallback** keeps the harness runnable without
//! network (at reduced realism, loudly).
//!
//! Corpora, each modeling a different eval (design §10.4):
//! - **synthetic-from-wordlist** (default, latency): real English words sampled with
//!   a Zipfian frequency law, so character-trigram document frequencies look like real
//!   text. The latency sweep's corpus; a tiny vocabulary would collapse every trigram
//!   onto near-every document and measure a degenerate dense-posting regime instead.
//! - **msmarco** (latency): a deterministic subsample of MS MARCO passages.
//! - **msmarco relevance** ([`msmarco_relevance`]): the **real dev queries + qrels**
//!   relevance eval. The index is built *answers + distractors* — every judged-relevant
//!   passage for the sampled queries, plus random distractors — so the known answer is
//!   present and recall@k measures ranking it over the distractors (§10.4).
//! - **GeoNames entities** ([`geonames`]): the fuzzy/typo eval's home (§10.5). Entity
//!   names + injected edits, where "type a corrupted target name, find the target" is
//!   the faithful task. `geonames-cities` (~34k) and `geonames-all` (the full gazetteer).

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::rng::Rng;

/// One indexed document.
pub struct Document {
    pub id: i64,
    pub text: String,
}

/// A loaded corpus of short documents.
pub struct Corpus {
    pub docs: Vec<Document>,
    /// Human-readable description (source + realism caveat) for the report header.
    pub provenance: String,
}

/// One MS MARCO relevance query: the real query text and the judged-relevant passage
/// ids that are actually present in the indexed corpus (the eval's ground truth, after
/// filtering to in-corpus answers so scoring is symmetric).
pub struct RelQuery {
    pub text: String,
    pub relevant: Vec<i64>,
}

/// A built relevance eval: an answers+distractors [`Corpus`] plus the labeled real
/// queries that score against it.
pub struct Relevance {
    pub corpus: Corpus,
    pub queries: Vec<RelQuery>,
}

/// One entity target for the fuzzy eval: the indexed id and its (uncorrupted) name.
#[derive(Clone)]
pub struct Entity {
    pub id: i64,
    pub name: String,
}

/// A built entity corpus: every entity indexed (targets + near-match distractors) plus
/// the sampled target subset to generate name+edit queries from. Every target is in the
/// corpus, so its answer is always present.
pub struct EntityCorpus {
    pub corpus: Corpus,
    pub targets: Vec<Entity>,
}

/// A committed pinned-source manifest.
#[derive(Deserialize)]
pub struct Manifest {
    pub name: String,
    pub license: String,
    pub url: String,
    /// Expected SHA-256 (hex). Empty means "unpinned": the fetch prints the computed
    /// hash to pin back into the manifest, and proceeds with a warning.
    pub sha256: String,
}

const WORDS_MANIFEST: &str = include_str!("../sources/words_alpha.json");
const MSMARCO_MANIFEST: &str = include_str!("../sources/msmarco.json");
const MSMARCO_QUERIES_MANIFEST: &str = include_str!("../sources/msmarco-queries.json");
const MSMARCO_QRELS_MANIFEST: &str = include_str!("../sources/msmarco-qrels.json");
const GEONAMES_CITIES_MANIFEST: &str = include_str!("../sources/geonames-cities15000.json");
const GEONAMES_ALL_MANIFEST: &str = include_str!("../sources/geonames-all.json");

/// Per-corpus cache namespaces — also the `--corpus` values. Each corpus's assets
/// live under their own subdirectory so identically named files (a `collection.tar.gz`
/// archive, an extracted `collection.tsv`) from different corpora never stomp on each
/// other in the cache. The relevance eval's queries/qrels share the `msmarco`
/// namespace so they sit beside (and reuse) the already-extracted `collection.tsv`.
const CORPUS_SYNTHETIC: &str = "synthetic";
const CORPUS_MSMARCO: &str = "msmarco";
pub const CORPUS_GEONAMES_CITIES: &str = "geonames-cities";
pub const CORPUS_GEONAMES_ALL: &str = "geonames-all";

/// The GeoNames member file inside each corpus's zip.
fn geonames_member(corpus: &str) -> Option<&'static str> {
    match corpus {
        CORPUS_GEONAMES_CITIES => Some("cities15000.txt"),
        CORPUS_GEONAMES_ALL => Some("allCountries.txt"),
        _ => None,
    }
}

fn geonames_manifest(corpus: &str) -> Option<Manifest> {
    match corpus {
        CORPUS_GEONAMES_CITIES => Some(manifest(GEONAMES_CITIES_MANIFEST)),
        CORPUS_GEONAMES_ALL => Some(manifest(GEONAMES_ALL_MANIFEST)),
        _ => None,
    }
}

fn manifest(json: &str) -> Manifest {
    serde_json::from_str(json).expect("pinned-source manifest is valid JSON")
}

/// The gitignored download-cache root (repo-root `.cache/bench`).
fn cache_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/benchmarks`; the cache is repo-root `.cache`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("benchmarks/ has a parent")
        .join(".cache")
        .join("bench")
}

/// One corpus's cache subdirectory under the root. Namespacing per corpus isolates a
/// corpus's downloaded archive and any extracted artifacts, so two corpora that ship
/// identically named files cannot overwrite each other.
fn cache_dir(corpus: &str) -> PathBuf {
    cache_root().join(corpus)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    // Shell out to the ubiquitous `sha256sum` rather than vendor a hash impl into a
    // throwaway harness; the line is `<hex>  <path>`.
    let out = Command::new("sha256sum").arg(path).output()?;
    if !out.status.success() {
        return Err(io::Error::other("sha256sum failed"));
    }
    let line = String::from_utf8_lossy(&out.stdout);
    Ok(line
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string())
}

/// Ensure the pinned file is present + verified in `corpus`'s cache, downloading
/// once if absent. Returns the cached path. The proxy/CA are configured in the
/// environment, so a plain `curl` honors them.
///
/// Verification is strict: when the manifest pins a `sha256`, a cached *or* freshly
/// downloaded file whose hash differs is a hard error (see [`sha256_mismatch`]) — the
/// run stops rather than silently serving or re-downloading the wrong bytes. An empty
/// manifest `sha256` is "unpinned": the file is used as-is with a loud warning
/// carrying the computed hash to pin back into the manifest.
fn ensure(m: &Manifest, corpus: &str) -> io::Result<PathBuf> {
    let dir = cache_dir(corpus);
    std::fs::create_dir_all(&dir)?;
    let file_name = m.url.rsplit('/').next().unwrap_or("asset");
    let dest = dir.join(file_name);

    // A cached copy: verify it against the pin before reusing. A mismatch is fatal —
    // we do not silently re-download over it. An unpinned manifest cannot verify, so
    // reuse with a warning rather than re-fetch a multi-GiB archive every run.
    if dest.is_file() {
        if m.sha256.is_empty() {
            eprintln!(
                "WARNING: {} is unpinned; reusing cached {} without verification.\n  Pin its sha256 in the manifest for a verified, reproducible cache.",
                m.name,
                dest.display()
            );
            return Ok(dest);
        }
        let got = sha256_file(&dest)?;
        if got == m.sha256 {
            return Ok(dest);
        }
        return Err(io::Error::other(sha256_mismatch(m, &dest, &got)));
    }

    eprintln!("fetching {} -> {}", m.url, dest.display());
    let tmp = dest.with_extension("part");
    let status = Command::new("curl")
        .args(["-fSL", "--proto", "=https", "-o"])
        .arg(&tmp)
        .arg(&m.url)
        .status()?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(io::Error::other(format!("download failed: {}", m.url)));
    }

    let got = sha256_file(&tmp)?;
    if m.sha256.is_empty() {
        eprintln!(
            "WARNING: {} is unpinned. Computed sha256 = {got}\n  Pin it in the manifest for reproducibility.",
            m.name
        );
    } else if got != m.sha256 {
        let _ = std::fs::remove_file(&tmp);
        return Err(io::Error::other(sha256_mismatch(m, &dest, &got)));
    }
    std::fs::rename(&tmp, &dest)?; // atomic: a half-written download never looks complete
    Ok(dest)
}

/// A clear, actionable sha256-mismatch error. Either the upstream source legitimately
/// changed — update the manifest's `sha256` to the printed value — or the cached file
/// is corrupt, in which case deleting it and re-running re-downloads a clean copy.
fn sha256_mismatch(m: &Manifest, dest: &Path, got: &str) -> String {
    format!(
        "sha256 mismatch for {name}\n  expected: {expected}\n  got:      {got}\n  file:     {file}\n  \
         If the upstream source legitimately changed, update its \"sha256\" in benchmarks/sources/ to \
         the value above; otherwise the cached file is corrupt — delete it and re-run to re-download.",
        name = m.name,
        expected = m.sha256,
        file = dest.display(),
    )
}

/// Extract a single `member` file from `archive` into `dir` (idempotent — skips if
/// already extracted). Supports `.zip` (GeoNames, via `unzip`) and `.tar.gz`/`.tgz`
/// (MS MARCO, via `tar`); restricting to one member avoids unpacking a tarball's other
/// large files (e.g. queries.train.tsv). Returns the extracted member's path.
fn extract_member(archive: &Path, dir: &Path, member: &str) -> io::Result<PathBuf> {
    let out = dir.join(member);
    if out.is_file() {
        return Ok(out);
    }
    std::fs::create_dir_all(dir)?;
    eprintln!("extracting {member} from {} ...", archive.display());
    let name = archive.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let status = if name.ends_with(".zip") {
        Command::new("unzip")
            .arg("-oq")
            .arg(archive)
            .arg(member)
            .arg("-d")
            .arg(dir)
            .status()?
    } else {
        Command::new("tar")
            .arg("-xzf")
            .arg(archive)
            .arg("-C")
            .arg(dir)
            .arg(member)
            .status()?
    };
    if !status.success() {
        return Err(io::Error::other(format!("extraction of {member} failed")));
    }
    if !out.is_file() {
        return Err(io::Error::other(format!(
            "{member} not found after extracting {}",
            archive.display()
        )));
    }
    Ok(out)
}

/// In-place Fisher–Yates shuffle with the deterministic RNG (so a seed fixes the
/// sampled query/distractor subsets).
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}

/// Pre-download and hash-verify the assets a corpus needs, without building it —
/// the `fetch` subcommand, for warming the cache on a network machine before an
/// offline run.
pub fn prefetch(corpus: &str) -> io::Result<()> {
    // The manifests each corpus needs (the relevance eval pulls three).
    let assets: Vec<(Manifest, &str)> = match corpus {
        CORPUS_SYNTHETIC => vec![(manifest(WORDS_MANIFEST), CORPUS_SYNTHETIC)],
        CORPUS_MSMARCO => vec![(manifest(MSMARCO_MANIFEST), CORPUS_MSMARCO)],
        "msmarco-relevance" | "relevance" => vec![
            (manifest(MSMARCO_MANIFEST), CORPUS_MSMARCO),
            (manifest(MSMARCO_QUERIES_MANIFEST), CORPUS_MSMARCO),
            (manifest(MSMARCO_QRELS_MANIFEST), CORPUS_MSMARCO),
        ],
        CORPUS_GEONAMES_CITIES => {
            vec![(manifest(GEONAMES_CITIES_MANIFEST), CORPUS_GEONAMES_CITIES)]
        }
        CORPUS_GEONAMES_ALL => vec![(manifest(GEONAMES_ALL_MANIFEST), CORPUS_GEONAMES_ALL)],
        other => return Err(io::Error::other(format!("unknown corpus: {other}"))),
    };
    for (m, ns) in &assets {
        eprintln!("{} — {}", m.name, m.license);
        let p = ensure(m, ns)?;
        eprintln!("ready: {}", p.display());
    }
    Ok(())
}

/// The offline fallback vocabulary — a domain-flavored set used only when the
/// pinned wordlist is not cached, so the harness still runs (with reduced trigram
/// realism, loudly warned).
const FALLBACK_VOCAB: &[&str] = &[
    "mitochondria",
    "ribosome",
    "photosynthesis",
    "chlorophyll",
    "enzyme",
    "catalyst",
    "molecule",
    "covalent",
    "electron",
    "neutron",
    "isotope",
    "polymer",
    "entropy",
    "quantum",
    "relativity",
    "gravity",
    "velocity",
    "momentum",
    "frequency",
    "amplitude",
    "algorithm",
    "compiler",
    "recursion",
    "pointer",
    "iterator",
    "closure",
    "monomorphize",
    "parliament",
    "constitution",
    "sovereign",
    "diplomacy",
    "treaty",
    "embargo",
    "renaissance",
    "baroque",
    "impressionism",
    "symphony",
    "concerto",
    "cathedral",
    "ecosystem",
    "biodiversity",
    "photosphere",
    "stratosphere",
    "tectonic",
    "sediment",
    "hypothesis",
    "empirical",
    "correlation",
    "regression",
    "variance",
    "probability",
];

/// Load the active vocabulary: the cached pinned wordlist when present, else the
/// fallback (with a warning). Words are lowercased ASCII, length 3..=15 (the
/// trigram floor, dropping pathological ultra-long words).
fn load_vocabulary() -> Vec<String> {
    let m = manifest(WORDS_MANIFEST);
    let cached =
        cache_dir(CORPUS_SYNTHETIC).join(m.url.rsplit('/').next().unwrap_or("words_alpha.txt"));
    let raw = if cached.is_file() {
        std::fs::read_to_string(&cached).ok()
    } else {
        // Try to fetch; on failure (offline), fall back.
        match ensure(&m, CORPUS_SYNTHETIC).and_then(std::fs::read_to_string) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "WARNING: wordlist unavailable ({e}); using the small fallback vocabulary — trigram realism is reduced."
                );
                None
            }
        }
    };
    let source = raw.unwrap_or_else(|| FALLBACK_VOCAB.join("\n"));
    let mut words: Vec<String> = source
        .lines()
        .map(|w| w.trim().to_ascii_lowercase())
        .filter(|w| (3..=15).contains(&w.len()) && w.bytes().all(|b| b.is_ascii_lowercase()))
        .collect();
    words.dedup();
    words
}

/// Build a synthetic corpus of `n` documents from the real wordlist, sampling words
/// with a Zipfian frequency law so trigram document-frequencies look like real text.
pub fn synthetic(n: usize, seed: u64) -> Corpus {
    let vocab = load_vocabulary();
    assert!(!vocab.is_empty(), "vocabulary must be non-empty");
    // A Zipfian rank → weight table (weight ∝ 1/rank), normalized to a cumulative
    // distribution we binary-search per word draw. `s = 1.0` is classic Zipf.
    let cum = zipf_cumulative(vocab.len(), 1.0);
    let mut rng = Rng::new(seed);
    let docs = (0..n as i64)
        .map(|id| {
            let words = rng.range(6, 20); // varied document length
            let text = (0..words)
                .map(|_| vocab[zipf_sample(&cum, &mut rng)].as_str())
                .collect::<Vec<_>>()
                .join(" ");
            Document { id, text }
        })
        .collect();
    Corpus {
        docs,
        provenance: format!(
            "synthetic-from-wordlist ({} words, Zipfian s=1.0, seed {seed})",
            vocab.len()
        ),
    }
}

/// Load a deterministic subsample of `n` MS MARCO passages (real documents). The
/// pinned tarball is downloaded + extracted on demand; this requires network and
/// ~1 GiB of disk for the archive.
pub fn msmarco(n: usize, seed: u64) -> io::Result<Corpus> {
    let m = manifest(MSMARCO_MANIFEST);
    let archive = ensure(&m, CORPUS_MSMARCO)?;
    let tsv = extract_member(&archive, &cache_dir(CORPUS_MSMARCO), "collection.tsv")?;
    // Reservoir-free deterministic subsample: take every k-th line (k = total/n) so
    // the subsample spans the file rather than its prefix. `total` is the known line
    // count; we stream once, keeping `id\ttext` lines whose index hits the stride.
    let body = std::fs::read_to_string(&tsv)?;
    let total = body.lines().count();
    let stride = (total / n).max(1);
    let mut rng = Rng::new(seed);
    let offset = rng.below(stride); // jitter the start so runs differ if reseeded
    let docs = body
        .lines()
        .enumerate()
        .filter(|(i, _)| i % stride == offset)
        .take(n)
        .filter_map(|(_, line)| {
            let (id, text) = line.split_once('\t')?;
            Some(Document {
                id: id.parse().ok()?,
                text: text.to_string(),
            })
        })
        .collect::<Vec<_>>();
    Ok(Corpus {
        provenance: format!(
            "MS MARCO passages (subsample {} of {total}, seed {seed})",
            docs.len()
        ),
        docs,
    })
}

/// Build the MS MARCO **relevance** eval (§10.4): real dev queries scored against their
/// qrels, with an "answers + distractors" index so the known answer is always present.
/// Streams `collection.tsv` (~8.8M lines) once — keeping the qrel-relevant passages for
/// the sampled queries and reservoir-sampling distractors — so the ~3 GB file is never
/// loaded whole. `docs` is the distractor target (a floor): the corpus is all answers
/// plus up to `max(0, docs - |answers|)` distractors.
pub fn msmarco_relevance(docs: usize, n_queries: usize, seed: u64) -> io::Result<Relevance> {
    let ns = CORPUS_MSMARCO;
    let cache = cache_dir(ns);
    let collection = extract_member(
        &ensure(&manifest(MSMARCO_MANIFEST), ns)?,
        &cache,
        "collection.tsv",
    )?;
    let qrels_path = ensure(&manifest(MSMARCO_QRELS_MANIFEST), ns)?;
    let queries_path = extract_member(
        &ensure(&manifest(MSMARCO_QUERIES_MANIFEST), ns)?,
        &cache,
        "queries.dev.tsv",
    )?;

    // qrels: qid -> relevant pids (`qid 0 pid 1`, tab-separated).
    let mut qrels: HashMap<i64, Vec<i64>> = HashMap::new();
    for line in std::fs::read_to_string(&qrels_path)?.lines() {
        let mut f = line.split('\t');
        let (Some(qid), _, Some(pid)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        let (Ok(qid), Ok(pid)) = (qid.parse::<i64>(), pid.parse::<i64>()) else {
            continue;
        };
        qrels.entry(qid).or_default().push(pid);
    }

    // Dev query texts, only for qids that have qrels.
    let mut qtext: HashMap<i64, String> = HashMap::new();
    for line in std::fs::read_to_string(&queries_path)?.lines() {
        let Some((qid, text)) = line.split_once('\t') else {
            continue;
        };
        let Ok(qid) = qid.parse::<i64>() else {
            continue;
        };
        if qrels.contains_key(&qid) {
            qtext.insert(qid, text.to_string());
        }
    }

    // Sample n_queries qids deterministically (sort then seeded shuffle).
    let mut rng = Rng::new(seed);
    let mut chosen: Vec<i64> = qtext.keys().copied().collect();
    chosen.sort_unstable();
    shuffle(&mut chosen, &mut rng);
    chosen.truncate(n_queries.min(chosen.len()));

    // The passages that MUST be indexed (every sampled query's answers).
    let mut answer_pids: HashSet<i64> = HashSet::new();
    for q in &chosen {
        answer_pids.extend(qrels[q].iter().copied());
    }

    // One streaming pass: collect answer passages; reservoir-sample distractors (pool
    // size `docs`) from the non-answer stream so answers are never displaced.
    let mut answers: HashMap<i64, String> = HashMap::new();
    let mut pool: Vec<Document> = Vec::new();
    let mut seen_distractors = 0usize;
    for line in io::BufReader::new(std::fs::File::open(&collection)?).lines() {
        let line = line?;
        let Some((pid_s, text)) = line.split_once('\t') else {
            continue;
        };
        let Ok(pid) = pid_s.parse::<i64>() else {
            continue;
        };
        if answer_pids.contains(&pid) {
            answers.entry(pid).or_insert_with(|| text.to_string());
        } else if docs > 0 {
            seen_distractors += 1;
            if pool.len() < docs {
                pool.push(Document {
                    id: pid,
                    text: text.to_string(),
                });
            } else if rng.below(seen_distractors) < docs {
                let slot = rng.below(docs);
                pool[slot] = Document {
                    id: pid,
                    text: text.to_string(),
                };
            }
        }
    }

    // Effective corpus = all answers + up to (docs - |answers|) distractors.
    let n_distractors = docs.saturating_sub(answers.len());
    shuffle(&mut pool, &mut rng);
    pool.truncate(n_distractors);
    let n_dist = pool.len();
    let mut all: Vec<Document> = answers
        .into_iter()
        .map(|(id, text)| Document { id, text })
        .collect();
    let n_answers = all.len();
    all.extend(pool);
    all.sort_by_key(|d| d.id); // deterministic insertion order

    // Labels: each query's relevant pids that actually made it into the index, so both
    // engines score against the identical in-corpus label set. Drop 0-answer queries.
    let indexed: HashSet<i64> = all.iter().map(|d| d.id).collect();
    let mut queries: Vec<RelQuery> = Vec::new();
    for q in &chosen {
        let relevant: Vec<i64> = qrels[q]
            .iter()
            .copied()
            .filter(|p| indexed.contains(p))
            .collect();
        if relevant.is_empty() {
            continue;
        }
        queries.push(RelQuery {
            text: qtext[q].clone(),
            relevant,
        });
    }

    let provenance = format!(
        "MS MARCO dev (real queries+qrels; {n_answers} answers + {n_dist} distractors, \
         {} scored queries, seed {seed})",
        queries.len()
    );
    Ok(Relevance {
        corpus: Corpus {
            docs: all,
            provenance,
        },
        queries,
    })
}

/// Build a GeoNames entity corpus for the fuzzy eval (§10.5). Streams the dump,
/// reservoir-samples `docs` entities as the indexed corpus (col 1 = geonameid = doc id,
/// col 2 = name), and samples `n_targets` of *those* as query targets — so every
/// target's answer is indexed, and the other entities are the (naturally near-match)
/// distractors §10.5 requires.
pub fn geonames(
    corpus: &str,
    docs: usize,
    n_targets: usize,
    seed: u64,
) -> io::Result<EntityCorpus> {
    let member = geonames_member(corpus)
        .ok_or_else(|| io::Error::other(format!("unknown geonames corpus: {corpus}")))?;
    let m = geonames_manifest(corpus).expect("geonames_member implies a manifest");
    let txt = extract_member(&ensure(&m, corpus)?, &cache_dir(corpus), member)?;

    let mut rng = Rng::new(seed);
    let mut pool: Vec<Document> = Vec::new();
    let mut seen = 0usize;
    for line in io::BufReader::new(std::fs::File::open(&txt)?).lines() {
        let line = line?;
        let mut f = line.split('\t');
        let (Some(id), Some(name)) = (f.next(), f.next()) else {
            continue;
        };
        if name.chars().count() < 3 {
            continue; // below the trigram floor
        }
        let Ok(id) = id.parse::<i64>() else {
            continue;
        };
        if docs == 0 {
            break;
        }
        let doc = Document {
            id,
            text: name.to_string(),
        };
        seen += 1;
        if pool.len() < docs {
            pool.push(doc);
        } else if rng.below(seen) < docs {
            let slot = rng.below(docs);
            pool[slot] = doc;
        }
    }
    pool.sort_by_key(|d| d.id); // deterministic order

    // Sample target entities from the indexed pool (so every target is present).
    let mut idx: Vec<usize> = (0..pool.len()).collect();
    shuffle(&mut idx, &mut rng);
    idx.truncate(n_targets.min(pool.len()));
    let targets: Vec<Entity> = idx
        .iter()
        .map(|&i| Entity {
            id: pool[i].id,
            name: pool[i].text.clone(),
        })
        .collect();

    let provenance = format!(
        "GeoNames {corpus} ({} entities indexed, {} query targets, seed {seed})",
        pool.len(),
        targets.len()
    );
    Ok(EntityCorpus {
        corpus: Corpus {
            docs: pool,
            provenance,
        },
        targets,
    })
}

/// Cumulative Zipfian distribution over `n` ranks with exponent `s`: `cum[i]` is the
/// probability of drawing rank ≤ i, normalized to end at 1.0.
fn zipf_cumulative(n: usize, s: f64) -> Vec<f64> {
    let mut cum = Vec::with_capacity(n);
    let mut acc = 0.0;
    for rank in 1..=n {
        acc += 1.0 / (rank as f64).powf(s);
        cum.push(acc);
    }
    let total = acc;
    for c in &mut cum {
        *c /= total;
    }
    cum
}

/// Draw a rank from the cumulative table: the first index whose cumulative weight is
/// ≥ a uniform `[0,1)` sample.
fn zipf_sample(cum: &[f64], rng: &mut Rng) -> usize {
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    cum.partition_point(|&c| c < u).min(cum.len() - 1)
}
