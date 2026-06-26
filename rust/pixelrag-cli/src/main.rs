//! # pixelrag-cli — PixelRAG command-line interface and benchmark harness
//!
//! Per **ADR-264** (PixelRAG Rust port on ruvector substrate). This binary is the
//! operator-facing entry point for the visual-RAG pipeline: it ingests documents
//! (render → embed → index), runs retrieval queries, and drives the benchmark
//! harness that the darwin / MetaHarness optimization loop (ADR-256) consumes.
//!
//! ## Status (this file)
//! The `benchmark` subcommand is **fully wired**: a hand-rolled std arg parser (no
//! external crates) builds [`Cli`] and routes `benchmark` to [`bench::run`], which
//! builds the index via `pixelrag-core` + the deterministic `SyntheticEmbedder`,
//! runs the subset-fixture queries, scores recall@10 / NDCG@10 / MRR + latency /
//! build / memory, and writes a JSON report carrying the mandatory honesty label.
//! It is the only shipped subcommand (the darwin loop drives only `benchmark`).
//!
//! ## Exit codes
//! - `0`  success
//! - `1`  runtime error (handler failure)
//! - `2`  usage error (bad args / unknown subcommand)

mod bench;
mod bench_visual;

use std::path::PathBuf;
use std::process::ExitCode;

/// Top-level subcommand the user selected on the command line.
///
/// Built by the hand-rolled [`Cli::parse_args`] std parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Run the TEXT benchmark harness (all-MiniLM/synthetic over tile bytes).
    Benchmark(bench::BenchArgs),
    /// Run the VISUAL benchmark harness (real CLIP ViT-B/32 cross-modal,
    /// text→image retrieval over rendered document screenshots).
    BenchmarkVisual(bench_visual::VisualArgs),
    /// Print usage help and exit successfully.
    Help,
}

/// Parsed command line: the chosen [`Command`] plus the program name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    /// argv[0] (program name) for usage messages.
    pub program: String,
    /// The dispatched subcommand.
    pub command: Command,
}

impl Cli {
    /// Parse raw process arguments into a [`Cli`].
    ///
    /// Hand-rolled std parsing (no external crates) over the fixed darwin command
    /// surface. The first token after `argv[0]` selects the subcommand; the rest are
    /// `--flag value` pairs. `benchmark` is the only shipped subcommand.
    pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Self, UsageError> {
        let program = args.next().unwrap_or_else(|| "pixelrag-cli".to_string());
        let sub = match args.next() {
            Some(s) => s,
            None => return Err(UsageError { message: "missing subcommand".into() }),
        };
        let rest: Vec<String> = args.collect();
        let command = match sub.as_str() {
            "help" | "-h" | "--help" => Command::Help,
            "benchmark" | "bench" => {
                // `--mode visual` selects the CLIP visual path; `text` (default) is
                // the existing all-MiniLM/synthetic harness.
                match flag(&rest, "--mode").map(|s| s.trim().to_ascii_lowercase()) {
                    Some(m) if m == "visual" => {
                        Command::BenchmarkVisual(parse_benchmark_visual(&rest)?)
                    }
                    Some(m) if m == "text" => Command::Benchmark(parse_benchmark(&rest)?),
                    None => Command::Benchmark(parse_benchmark(&rest)?),
                    Some(other) => {
                        return Err(UsageError {
                            message: format!("unknown --mode '{other}' (use text|visual)"),
                        })
                    }
                }
            }
            other => {
                return Err(UsageError { message: format!("unknown subcommand '{other}'") })
            }
        };
        Ok(Cli { program, command })
    }

    /// Render the top-level usage/help text.
    pub fn usage() -> &'static str {
        "pixelrag-cli — PixelRAG visual-RAG CLI + benchmark harness (ADR-264)\n\
         \n\
         USAGE:\n\
         \x20 pixelrag-cli <SUBCOMMAND> [FLAGS]\n\
         \n\
         SUBCOMMANDS:\n\
         \x20 benchmark   Build the index from the subset fixture, run queries, and\n\
         \x20             score recall@10 / NDCG@10 / MRR + latency/build/memory.\n\
         \x20 help        Print this help.\n\
         \n\
         benchmark FLAGS:\n\
         \x20 --mode <text|visual>    text (default) = all-MiniLM/synthetic over tile bytes;\n\
         \x20                         visual = REAL CLIP ViT-B/32 text→image retrieval.\n\
         \n\
         benchmark --mode text FLAGS:\n\
         \x20 --predictions <path>    Output path for per-query ranked predictions JSON.\n\
         \x20 --ground-truth <path>   qrels JSON (e.g. tests/fixtures/pixelrag/ground-truth.json).\n\
         \x20 --queries <path>        Queries JSON (e.g. tests/fixtures/pixelrag/queries.json).\n\
         \x20 --metrics <list>        Comma list, e.g. ndcg,mrr,recall@10 (default).\n\
         \x20 --tiles <dir>           Tiles dir (default: tiles/ next to --ground-truth).\n\
         \x20 --report-out <path>     Report JSON (default: bench_output/pixelrag_bench.json).\n\
         \x20 --k <n>                 Retrieval cutoff (default 10).\n\
         \x20 --batch-size <n>        Override embedding batch size.\n\
         \x20 --index-backend <name>  hnsw (default) | ivf-flat | ivf-sq.\n\
         \x20 --embedder <name>       real (default, all-MiniLM-L6-v2 via Node sidecar) | synthetic.\n\
         \x20 --darwin-config <path>  Optional darwin genome JSON (read-only).\n\
         \n\
         benchmark --mode visual FLAGS:\n\
         \x20 --fixture-dir <dir>     Visual fixture dir (default tests/fixtures/pixelrag/visual)\n\
         \x20                         holding manifest.json/queries.json/ground-truth.json + images/.\n\
         \x20 --predictions <path>    Output path for per-query ranked predictions JSON.\n\
         \x20 --report-out <path>     Report JSON (default: bench_output/pixelrag_visual_bench.json).\n\
         \x20 --metrics <list>        Comma list, e.g. ndcg,mrr,recall@10 (default).\n\
         \x20 --k <n>                 Retrieval cutoff (default 10).\n\
         \x20 --index-backend <name>  hnsw (default) | ivf-flat.\n\
         \n\
         HONESTY (text): with --embedder real (default) the benchmark uses REAL all-MiniLM-L6-v2\n\
         semantic embeddings over a small real eval fixture — semantic retrieval, still a\n\
         tiny corpus vs full-scale. With --embedder synthetic it uses a DETERMINISTIC\n\
         NON-SEMANTIC embedder (plumbing validation only). Either way the fixture is tiny.\n\
         HONESTY (visual): real CLIP ViT-B/32 cross-modal embeddings (CPU/WASM); text→image\n\
         retrieval over rendered document screenshots — a real visual encoder; Qwen3-VL/ColPali\n\
         is the GPU upgrade. Tiny 8-doc corpus — honest about scale, not a SOTA claim.\n"
    }

    /// Dispatch the parsed subcommand to its handler and map results to an exit code.
    ///
    /// Routes `Command` → handler and translates `Result<_, _>` into an [`ExitCode`]
    /// (`0` ok, `1` runtime error, `2` usage). `benchmark` runs the full harness.
    pub fn dispatch(self) -> ExitCode {
        match self.command {
            Command::Help => {
                println!("{}", Cli::usage());
                ExitCode::SUCCESS
            }
            Command::Benchmark(args) => match bench::run(args) {
                Ok(_report) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("benchmark error: {}", e.message);
                    ExitCode::from(1)
                }
            },
            Command::BenchmarkVisual(args) => match bench_visual::run(args) {
                Ok(_report) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("visual benchmark error: {}", e.message);
                    ExitCode::from(1)
                }
            },
        }
    }
}

// ── std flag parsing helpers (replaced by clap in M1+) ───────────────────────

/// Pull the value following `--name` from a flat `--flag value` argument list.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().map(String::as_str);
        }
        // Support `--name=value`.
        if let Some(v) = a.strip_prefix(name).and_then(|s| s.strip_prefix('=')) {
            return Some(v);
        }
    }
    None
}

fn parse_benchmark(args: &[String]) -> Result<bench::BenchArgs, UsageError> {
    use pixelrag_core::config::IndexBackend;

    let ground_truth = flag(args, "--ground-truth")
        .map(PathBuf::from)
        .ok_or_else(|| UsageError { message: "benchmark requires --ground-truth".into() })?;
    let queries = flag(args, "--queries")
        .map(PathBuf::from)
        .ok_or_else(|| UsageError { message: "benchmark requires --queries".into() })?;
    let predictions = flag(args, "--predictions")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("bench_output/vidore_results.json"));

    let metrics = match flag(args, "--metrics") {
        Some(list) => {
            let mut out = Vec::new();
            for tok in list.split(',').filter(|s| !s.trim().is_empty()) {
                let m = bench::MetricKind::parse(tok)
                    .map_err(|e| UsageError { message: format!("bad --metrics token '{}'", e.token) })?;
                out.push(m);
            }
            out
        }
        None => Vec::new(),
    };

    let index_backend = match flag(args, "--index-backend") {
        Some(s) => Some(match s.to_ascii_lowercase().as_str() {
            "hnsw" => IndexBackend::Hnsw,
            "ivfflat" | "ivf_flat" | "ivf-flat" => IndexBackend::IvfFlat,
            "ivfsq" | "ivf_sq" | "ivf-sq" => IndexBackend::IvfSq,
            other => {
                return Err(UsageError { message: format!("unknown --index-backend '{other}'") })
            }
        }),
        None => None,
    };

    let embedder = match flag(args, "--embedder") {
        Some(s) => bench::EmbedderChoice::parse(s)
            .ok_or_else(|| UsageError { message: format!("unknown --embedder '{s}' (use real|synthetic)") })?,
        None => bench::EmbedderChoice::default(), // default: real semantic all-MiniLM
    };

    Ok(bench::BenchArgs {
        predictions,
        ground_truth,
        queries,
        tiles_dir: flag(args, "--tiles").map(PathBuf::from),
        metrics,
        report_out: flag(args, "--report-out").map(PathBuf::from),
        darwin_config: flag(args, "--darwin-config").map(PathBuf::from),
        k: flag(args, "--k").and_then(|s| s.parse().ok()).unwrap_or(10),
        batch_size: flag(args, "--batch-size").and_then(|s| s.parse().ok()),
        index_backend,
        embedder,
    })
}

fn parse_benchmark_visual(args: &[String]) -> Result<bench_visual::VisualArgs, UsageError> {
    use pixelrag_core::config::IndexBackend;

    let mut va = bench_visual::VisualArgs::default();

    if let Some(d) = flag(args, "--fixture-dir") {
        va.fixture_dir = PathBuf::from(d);
    }
    if let Some(p) = flag(args, "--predictions") {
        va.predictions = PathBuf::from(p);
    }
    va.report_out = flag(args, "--report-out").map(PathBuf::from);
    if let Some(kv) = flag(args, "--k").and_then(|s| s.parse().ok()) {
        va.k = kv;
    }
    if let Some(s) = flag(args, "--index-backend") {
        va.index_backend = Some(match s.to_ascii_lowercase().as_str() {
            "hnsw" => IndexBackend::Hnsw,
            "ivfflat" | "ivf_flat" | "ivf-flat" => IndexBackend::IvfFlat,
            "ivfsq" | "ivf_sq" | "ivf-sq" => IndexBackend::IvfSq,
            other => {
                return Err(UsageError { message: format!("unknown --index-backend '{other}'") })
            }
        });
    }
    if let Some(list) = flag(args, "--metrics") {
        let mut out = Vec::new();
        for tok in list.split(',').filter(|s| !s.trim().is_empty()) {
            let m = bench::MetricKind::parse(tok)
                .map_err(|e| UsageError { message: format!("bad --metrics token '{}'", e.token) })?;
            out.push(m);
        }
        va.metrics = out;
    }

    Ok(va)
}

/// Usage (argument) error — maps to exit code `2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError {
    /// Human-readable description of what was wrong with the invocation.
    pub message: String,
}

/// Process entry point.
///
/// `Cli::parse_args(std::env::args())` → on `UsageError` print the usage banner to
/// stderr and return exit code `2`; otherwise `cli.dispatch()`.
fn main() -> ExitCode {
    match Cli::parse_args(std::env::args()) {
        Ok(cli) => cli.dispatch(),
        Err(usage_err) => {
            eprintln!("usage error: {}\n", usage_err.message);
            eprintln!("{}", Cli::usage());
            ExitCode::from(2)
        }
    }
}
