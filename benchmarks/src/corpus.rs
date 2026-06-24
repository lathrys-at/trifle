//! Real-corpus loading, following shrike's asset pattern: a **pinned source**
//! (`sources/*.json`: url + sha256 + attribution) is committed, the **bytes are
//! never committed** — they download on demand into a gitignored cache and are
//! hash-verified — and an **offline fallback** keeps the harness runnable without
//! network (at reduced realism, loudly).
//!
//! Two corpora:
//! - **synthetic-from-wordlist** (default): documents are runs of real English
//!   words sampled with a Zipfian frequency law, so character-trigram document
//!   frequencies are distributed like real text — common trigrams in many docs,
//!   rare ones in few. This is what makes the latency sweep meaningful; a tiny
//!   vocabulary collapses every trigram onto near-every document and measures a
//!   degenerate dense-posting regime instead.
//! - **msmarco** (opt-in, real documents): a deterministic subsample of MS MARCO
//!   passages — real prose, real vocabulary and co-occurrence.

use std::io;
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

fn manifest(json: &str) -> Manifest {
    serde_json::from_str(json).expect("pinned-source manifest is valid JSON")
}

/// The gitignored download cache (repo-root `.cache/bench`).
fn cache_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/benchmarks`; the cache is repo-root `.cache`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("benchmarks/ has a parent")
        .join(".cache")
        .join("bench")
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

/// Ensure the pinned file is present + verified in the cache, downloading once if
/// absent. Returns the cached path. The proxy/CA are configured in the environment,
/// so a plain `curl` honors them.
fn ensure(m: &Manifest) -> io::Result<PathBuf> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let file_name = m.url.rsplit('/').next().unwrap_or("asset");
    let dest = dir.join(file_name);

    if dest.is_file() && !m.sha256.is_empty() && sha256_file(&dest)? == m.sha256 {
        return Ok(dest);
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
        return Err(io::Error::other(format!(
            "sha256 mismatch for {}: got {got}, expected {}",
            m.name, m.sha256
        )));
    }
    std::fs::rename(&tmp, &dest)?; // atomic: a half-written download never looks complete
    Ok(dest)
}

/// Pre-download and hash-verify the assets a corpus needs, without building it —
/// the `fetch` subcommand, for warming the cache on a network machine before an
/// offline run. `synthetic` needs the wordlist; `msmarco` needs the passage archive.
pub fn prefetch(corpus: &str) -> io::Result<()> {
    let (m, after) = match corpus {
        "synthetic" => (manifest(WORDS_MANIFEST), ""),
        "msmarco" => (
            manifest(MSMARCO_MANIFEST),
            " (run the `msmarco` corpus once to extract collection.tsv)",
        ),
        other => return Err(io::Error::other(format!("unknown corpus: {other}"))),
    };
    eprintln!("{} — {}", m.name, m.license);
    let p = ensure(&m)?;
    eprintln!("ready: {}{after}", p.display());
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
    let cached = cache_dir().join(m.url.rsplit('/').next().unwrap_or("words_alpha.txt"));
    let raw = if cached.is_file() {
        std::fs::read_to_string(&cached).ok()
    } else {
        // Try to fetch; on failure (offline), fall back.
        match ensure(&m).and_then(std::fs::read_to_string) {
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
    let archive = ensure(&m)?;
    // Extract collection.tsv next to the archive (idempotent).
    let dir = archive.parent().expect("cached archive has a parent");
    let tsv = dir.join("collection.tsv");
    if !tsv.is_file() {
        eprintln!("extracting {} ...", archive.display());
        let status = Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(dir)
            .status()?;
        if !status.success() {
            return Err(io::Error::other("tar extraction failed"));
        }
    }
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
