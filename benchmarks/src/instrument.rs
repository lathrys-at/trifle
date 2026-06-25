//! Profiler instrumentation seam — re-exec the bench under a Rust-only sampling profiler.
//!
//! Modeled on shrike's `instrument.py`: rather than wiring a profiler into the hot path,
//! the harness re-execs *itself* under an external sampler, so the measured region is the
//! real production code path, symbolized end to end. This is macOS-first (the project's
//! development platform): the default is Instruments' **Time Profiler** via `xcrun
//! xctrace`; **`samply`** is the cross-platform Rust sampler (Firefox-profiler JSON). Both
//! are language-agnostic *sampling* profilers — no `tracing`-feature build, no async
//! runtime, nothing trifle-specific — which is why they suit a Cargo-only Rust project.
//!
//! The re-exec recursion is broken by an env guard ([`GUARD_ENV`]): the profiler launches
//! `trifle-bench latency …` with the guard set, and that inner (profiled) process sees it
//! and runs the benchmark normally instead of re-instrumenting. The driver checks
//! [`is_inner`] before deciding to re-exec.
//!
//! This is a *hook*, orthogonal to the machine-readable `--format json` output: instrument
//! to see *where* the time goes; emit JSON to record *how much*.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Set on the profiled child so a re-exec doesn't recurse. Its presence means "you are the
/// inner run — just execute the benchmark."
const GUARD_ENV: &str = "TRIFLE_BENCH_INSTRUMENTED";

/// The supported Rust-friendly sampling profilers.
#[derive(Clone, Copy)]
pub enum Profiler {
    /// Apple Instruments' Time Profiler, driven headless by `xcrun xctrace record`
    /// (macOS only). Produces a `.trace` bundle openable in Instruments.app.
    Xctrace,
    /// `samply record` — a cross-platform sampling profiler that emits Firefox-profiler
    /// JSON (`samply load <file>` to view).
    Samply,
}

impl Profiler {
    /// Parse the `--instrument` value.
    pub fn parse(name: &str) -> Result<Profiler, String> {
        match name {
            "xctrace" | "instruments" => Ok(Profiler::Xctrace),
            "samply" => Ok(Profiler::Samply),
            other => Err(format!("unknown --instrument {other} (xctrace|samply)")),
        }
    }

    fn artifact_ext(self) -> &'static str {
        match self {
            Profiler::Xctrace => "trace",
            Profiler::Samply => "json.gz",
        }
    }

    fn display(self) -> &'static str {
        match self {
            Profiler::Xctrace => "xctrace (Instruments Time Profiler)",
            Profiler::Samply => "samply",
        }
    }
}

/// True when running *inside* an instrumented re-exec (the profiled child). The latency
/// driver checks this first: if set, it skips re-exec and runs the benchmark normally.
pub fn is_inner() -> bool {
    std::env::var_os(GUARD_ENV).is_some()
}

/// Re-exec the current binary under `profiler`, profiling `trifle-bench <subcommand>
/// <passthru…>`. `passthru` is the original argument list (after the subcommand) with the
/// `--instrument*` flags already stripped (see [`strip_self_flags`]). Writes the trace
/// artifact under `out_dir` (named for the run, suffixed with the pid to avoid clobbering a
/// prior `.trace`) and returns the profiler's exit code.
pub fn run(
    profiler: Profiler,
    out_dir: &Path,
    subcommand: &str,
    passthru: &[String],
) -> Result<i32, String> {
    let exe =
        std::env::current_exe().map_err(|e| format!("cannot resolve current executable: {e}"))?;
    std::fs::create_dir_all(out_dir).map_err(|e| {
        format!(
            "cannot create instrument out dir {}: {e}",
            out_dir.display()
        )
    })?;
    let stem = artifact_stem(subcommand, passthru);
    let artifact = out_dir.join(format!(
        "{stem}-{}.{}",
        std::process::id(),
        profiler.artifact_ext()
    ));

    let mut cmd = match profiler {
        Profiler::Xctrace => {
            ensure_tool("xcrun")?;
            let mut c = Command::new("xcrun");
            c.arg("xctrace")
                .arg("record")
                .arg("--template")
                .arg("Time Profiler")
                .arg("--no-prompt")
                .arg("--output")
                .arg(&artifact)
                // The launched grandchild needs the guard; xctrace forwards --env to it.
                .arg("--env")
                .arg(format!("{GUARD_ENV}=1"))
                .arg("--launch")
                .arg("--")
                .arg(&exe)
                .arg(subcommand)
                .args(passthru);
            c
        }
        Profiler::Samply => {
            ensure_tool("samply")?;
            let mut c = Command::new("samply");
            c.arg("record")
                .arg("--save-only")
                .arg("-o")
                .arg(&artifact)
                .arg("--")
                .arg(&exe)
                .arg(subcommand)
                .args(passthru);
            // samply runs the command as a direct child, inheriting our env.
            c.env(GUARD_ENV, "1");
            c
        }
    };

    eprintln!(
        "instrument: profiling `{subcommand} {}` under {} -> {}",
        passthru.join(" "),
        profiler.display(),
        artifact.display()
    );
    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch profiler ({}): {e}", profiler.display()))?;
    let code = status
        .code()
        .unwrap_or(if status.success() { 0 } else { 1 });
    if artifact.exists() {
        let opener = match profiler {
            Profiler::Xctrace => format!("open {}", artifact.display()),
            Profiler::Samply => format!("samply load {}", artifact.display()),
        };
        eprintln!(
            "instrument: wrote {} — view with: {opener}",
            artifact.display()
        );
    } else {
        eprintln!(
            "instrument: profiler exited {code} but no artifact at {} (is the profiler installed and permitted?)",
            artifact.display()
        );
    }
    Ok(code)
}

/// Drop this harness's own flags (`--instrument`, `--instrument-out`) and their values from
/// a `latency` argument list, so the profiled re-exec runs the same benchmark *without*
/// re-triggering instrumentation. Handles both `--flag value` and `--flag=value` forms.
pub fn strip_self_flags(args: &[String]) -> Vec<String> {
    const VALUED: [&str; 2] = ["--instrument", "--instrument-out"];
    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some((k, _)) = a.split_once('=')
            && VALUED.contains(&k)
        {
            i += 1;
            continue;
        }
        if VALUED.contains(&a.as_str()) {
            i += 2; // skip the flag and its separate value
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

/// A filename stem describing the run: the subcommand plus the
/// `--corpus`/`--docs`/`--effort`/`--effort-sweep` values if present, so concurrent traces
/// are distinguishable at a glance.
fn artifact_stem(subcommand: &str, passthru: &[String]) -> String {
    let mut parts = vec![subcommand.to_string()];
    for key in ["corpus", "docs", "effort", "effort-sweep"] {
        if let Some(v) = flag_value(passthru, key) {
            parts.push(format!("{key}{}", sanitize(&v)));
        }
    }
    parts.join("-")
}

/// The value of `--key` in `args` (`--key value` or `--key=value`), if present.
fn flag_value(args: &[String], key: &str) -> Option<String> {
    let dashed = format!("--{key}");
    let eq = format!("--{key}=");
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(rest) = a.strip_prefix(&eq) {
            return Some(rest.to_string());
        }
        if a == &dashed {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

/// Keep a value filename-safe (commas in `--effort-sweep` become `_`).
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Error out early with an actionable message if `tool` isn't on `PATH`.
fn ensure_tool(tool: &str) -> Result<(), String> {
    if which(tool).is_some() {
        Ok(())
    } else {
        Err(format!(
            "instrument: `{tool}` not found on PATH. Install it (xcrun ships with Xcode \
             command-line tools; `cargo install samply`) or pick the other --instrument backend."
        ))
    }
}

/// A minimal `which`: the first `PATH` entry containing an executable file named `tool`.
fn which(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(tool))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_removes_instrument_flags_both_forms() {
        let args = vec![
            "--docs".to_string(),
            "1000".to_string(),
            "--instrument".to_string(),
            "xctrace".to_string(),
            "--instrument-out=/tmp/t".to_string(),
            "--effort-sweep".to_string(),
            "low,high".to_string(),
        ];
        assert_eq!(
            strip_self_flags(&args),
            vec!["--docs", "1000", "--effort-sweep", "low,high"]
        );
    }

    #[test]
    fn stem_encodes_subcommand_docs_and_effort() {
        let args = vec![
            "--corpus".to_string(),
            "msmarco".to_string(),
            "--docs".to_string(),
            "5000".to_string(),
            "--effort-sweep".to_string(),
            "low,medium,high".to_string(),
        ];
        assert_eq!(
            artifact_stem("perf", &args),
            "perf-corpusmsmarco-docs5000-effort-sweeplow_medium_high"
        );
    }

    #[test]
    fn flag_value_reads_both_forms() {
        let a = vec!["--docs=10".to_string()];
        assert_eq!(flag_value(&a, "docs").as_deref(), Some("10"));
        let b = vec!["--docs".to_string(), "20".to_string()];
        assert_eq!(flag_value(&b, "docs").as_deref(), Some("20"));
        assert_eq!(flag_value(&b, "missing"), None);
    }
}
