//! # pixelrag-cli — PixelRAG command-line interface and benchmark harness
//!
//! Per **ADR-264** (PixelRAG Rust port on ruvector substrate). This binary is the
//! operator-facing entry point for the visual-RAG pipeline: it ingests documents
//! (render → embed → index), runs retrieval queries, and drives the benchmark
//! harness that the darwin / MetaHarness optimization loop (ADR-256) consumes.
//!
//! ## M1 status (this file)
//! The `benchmark` subcommand is **fully wired**: a hand-rolled std arg parser (no
//! external crates) builds [`Cli`] and routes `benchmark` to [`bench::run`], which
//! builds the index via `pixelrag-core` + the deterministic `SyntheticEmbedder`,
//! runs the subset-fixture queries, scores recall@10 / NDCG@10 / MRR + latency /
//! build / memory, and writes a JSON report carrying the mandatory honesty label.
//! `index` / `search` remain `unimplemented!` (the darwin loop drives only
//! `benchmark`); their signatures are preserved for a later milestone.
//!
//! ## M1+ plan
//! - Replace [`Cli::parse_args`] std parsing with `clap` derive.
//! - `index`  → `pixelrag_core::Pipeline::index` (render+embed+`ruvector-core` HNSW / `ruvector-rairs` IVF-SQ).
//! - `search` → `pixelrag_core::Pipeline::search` (AnnIndex::search + rabitq allowlist filter).
//!
//! ## Exit codes
//! - `0`  success
//! - `1`  runtime error (handler failure)
//! - `2`  usage error (bad args / unknown subcommand)

mod bench;

use std::path::PathBuf;
use std::process::ExitCode;

/// Top-level subcommand the user selected on the command line.
///
/// In M1 this enum is produced by `clap` derive; for M0 it is built by the
/// hand-rolled [`Cli::parse_args`] std parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Ingest documents into a PixelRAG index (render → embed → index).
    Index(IndexArgs),
    /// Retrieve the top-k visually-similar tiles for one or more queries.
    Search(SearchArgs),
    /// Run the benchmark harness (recall/NDCG/MRR + latency + memory).
    Benchmark(bench::BenchArgs),
    /// Print usage help and exit successfully.
    Help,
}

/// Arguments for the `index` subcommand.
///
/// Mirrors the ADR-264 benchmark command surface
/// (`--dataset`, `--output`, `--batch-size`, `--quantization`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IndexArgs {
    /// Named dataset to ingest (e.g. `vidore`), mutually exclusive with `doc_path`.
    pub dataset: Option<String>,
    /// Direct document path to ingest (file or directory).
    pub doc_path: Option<PathBuf>,
    /// Optional source URL to render+ingest (M2 `--from-url`).
    pub from_url: Option<String>,
    /// Destination index file (`*.pixelrag`, bincode in M2).
    pub output: PathBuf,
    /// Embedding batch size fed to the encoder.
    pub batch_size: usize,
    /// Quantization tier label (`2-bit` | `3-bit` | `4-bit`).
    pub quantization: String,
}

/// Arguments for the `search` subcommand.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SearchArgs {
    /// Index file produced by `index`.
    pub index: PathBuf,
    /// JSON file of queries (image paths or query-image references).
    pub queries: PathBuf,
    /// Number of results to return per query.
    pub k: usize,
    /// Where to write retrieval results as JSON.
    pub output: PathBuf,
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
    /// `--flag value` pairs. `benchmark` is fully wired; `index`/`search` accept their
    /// flags but their handlers are deferred.
    ///
    /// ## M1+ plan
    /// Replace entirely with `clap` derive (`#[derive(Parser)]`).
    pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Self, UsageError> {
        let program = args.next().unwrap_or_else(|| "pixelrag-cli".to_string());
        let sub = match args.next() {
            Some(s) => s,
            None => return Err(UsageError { message: "missing subcommand".into() }),
        };
        let rest: Vec<String> = args.collect();
        let command = match sub.as_str() {
            "help" | "-h" | "--help" => Command::Help,
            "index" => Command::Index(parse_index(&rest)?),
            "search" => Command::Search(parse_search(&rest)?),
            "benchmark" | "bench" => Command::Benchmark(parse_benchmark(&rest)?),
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
         \x20 index       (M1+, not yet wired) Ingest documents into a PixelRAG index.\n\
         \x20 search      (M1+, not yet wired) Retrieve top-k tiles for queries.\n\
         \x20 help        Print this help.\n\
         \n\
         benchmark FLAGS:\n\
         \x20 --predictions <path>    Output path for per-query ranked predictions JSON.\n\
         \x20 --ground-truth <path>   qrels JSON (e.g. tests/fixtures/pixelrag/ground-truth.json).\n\
         \x20 --queries <path>        Queries JSON (e.g. tests/fixtures/pixelrag/queries.json).\n\
         \x20 --metrics <list>        Comma list, e.g. ndcg,mrr,recall@10 (default).\n\
         \x20 --tiles <dir>           Tiles dir (default: tiles/ next to --ground-truth).\n\
         \x20 --report-out <path>     Report JSON (default: bench_output/pixelrag_bench.json).\n\
         \x20 --k <n>                 Retrieval cutoff (default 10).\n\
         \x20 --batch-size <n>        Override embedding batch size.\n\
         \x20 --index-backend <name>  hnsw (default) | ivf-flat | ivf-sq | turbovec.\n\
         \x20 --darwin-config <path>  Optional darwin genome JSON (read-only).\n\
         \n\
         HONESTY: the benchmark uses a DETERMINISTIC SYNTHETIC embedder on a TINY SUBSET\n\
         fixture. Every metric is plumbing validation, NOT semantic retrieval quality;\n\
         real recall requires Qwen3-VL-2B (blocked in this environment).\n"
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
            Command::Index(args) => match run_index(args) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e.message);
                    ExitCode::from(1)
                }
            },
            Command::Search(args) => match run_search(args) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {}", e.message);
                    ExitCode::from(1)
                }
            },
            Command::Benchmark(args) => match bench::run(args) {
                Ok(_report) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("benchmark error: {}", e.message);
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

fn parse_index(args: &[String]) -> Result<IndexArgs, UsageError> {
    Ok(IndexArgs {
        dataset: flag(args, "--dataset").map(String::from),
        doc_path: flag(args, "--doc").map(PathBuf::from),
        from_url: flag(args, "--from-url").map(String::from),
        output: flag(args, "--output").map(PathBuf::from).unwrap_or_default(),
        batch_size: flag(args, "--batch-size").and_then(|s| s.parse().ok()).unwrap_or(32),
        quantization: flag(args, "--quantization").unwrap_or("none").to_string(),
    })
}

fn parse_search(args: &[String]) -> Result<SearchArgs, UsageError> {
    Ok(SearchArgs {
        index: flag(args, "--index").map(PathBuf::from).unwrap_or_default(),
        queries: flag(args, "--queries").map(PathBuf::from).unwrap_or_default(),
        k: flag(args, "--k").and_then(|s| s.parse().ok()).unwrap_or(10),
        output: flag(args, "--output").map(PathBuf::from).unwrap_or_default(),
    })
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
            "turbovec" => IndexBackend::Turbovec,
            other => {
                return Err(UsageError { message: format!("unknown --index-backend '{other}'") })
            }
        }),
        None => None,
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
    })
}

/// Usage (argument) error — maps to exit code `2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError {
    /// Human-readable description of what was wrong with the invocation.
    pub message: String,
}

/// Runtime error from a handler — maps to exit code `1`.
#[derive(Debug)]
pub struct CliError {
    /// Human-readable description of the failure.
    pub message: String,
}

/// Execute the `index` subcommand.
///
/// ## Intended behavior (M1)
/// Construct a `pixelrag_core::Pipeline` from `Config`, resolve the dataset/doc
/// source, then drive render → `pixelrag_encoder::Embedder::embed_batch` →
/// `pixelrag_core::IndexAdapter` (wrapping `ruvector-core::HNSWIndex` in M1, or
/// `ruvector-rairs::IVFIndex` as the memory fallback), and persist the index to
/// `args.output`. Reports tiles indexed and build time.
pub fn run_index(_args: IndexArgs) -> Result<(), CliError> {
    unimplemented!(
        "M1: pixelrag_core::Pipeline::index — render+embed tiles, build \
         ruvector-core HNSW (or ruvector-rairs IVF-SQ) index, persist to --output"
    )
}

/// Execute the `search` subcommand.
///
/// ## Intended behavior (M1)
/// Load the persisted index, embed each query via `pixelrag_encoder`, call
/// `pixelrag_core::Pipeline::search` (which calls `ruvector_rabitq::AnnIndex::search`
/// with optional rabitq allowlist filtering), and write top-k hits to `args.output`
/// as JSON for the benchmark stage to score.
pub fn run_search(_args: SearchArgs) -> Result<(), CliError> {
    unimplemented!(
        "M1: pixelrag_core::Pipeline::search — embed queries, AnnIndex::search(k) \
         with rabitq allowlist filter, write results JSON"
    )
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
