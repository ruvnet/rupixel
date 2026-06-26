//! # bench — PixelRAG benchmark harness
//!
//! Per **ADR-264** §Validation and §MetaHarness/Darwin integration (ADR-256).
//!
//! This module is the benchmark entry point the **darwin harness drives**. Darwin
//! evolves *harness parameters* (quantization tier, batch size, embedding-cache
//! size, index backend, rerank strategy) — never the Rust source — deploys each
//! candidate, and scores it on the ViDoRe SUBSET fixture (`NDCG@10 × index memory`,
//! Pareto frontier). The output here is the per-run [`BenchReport`] darwin consumes.
//!
//! ## Metrics produced (ADR-264 §Metrics)
//! - Retrieval quality: [`RetrievalMetrics`] — recall@k, NDCG@k, MRR.
//! - Latency: [`LatencyMetrics`] — p50 / p95 / p99 for vec-sim search (rerank excluded).
//! - Index build time: seconds per 1000 docs (embed + add).
//! - Memory: index footprint per indexed tile (honest f32-originals estimate).
//!
//! ## HONESTY — read before trusting any number here
//!
//! There is **no real Qwen3-VL-Embedding-2B** in this environment (weights + GPU are
//! blocked). The `benchmark` subcommand builds the index with the **deterministic
//! synthetic embedder** ([`pixelrag_encoder::SyntheticEmbedder`]) over a **tiny
//! subset fixture** (`tests/fixtures/pixelrag/`). Therefore every recall / NDCG / MRR
//! number this harness emits measures **pipeline plumbing on a subset fixture, NOT
//! semantic retrieval quality**. The report carries [`HONESTY_LABEL`] on every run so
//! a downstream consumer (darwin / a human) can never mistake it for semantic recall.
//! Real recall requires Qwen3-VL-2B (blocked). Determinism (fixed seed) keeps the run
//! reproducible so darwin can compare candidate *parameters* fairly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pixelrag_core::config::{Config, IndexBackend};
use pixelrag_core::index::build_index;
use pixelrag_core::pipeline::Pipeline;
use pixelrag_core::search::SearchRequest;
use pixelrag_core::tile::Tiler;
use pixelrag_core::{embedding::EncoderEmbedder, Embedding};
use pixelrag_encoder::{Embedder as _, SidecarEmbedder, SyntheticEmbedder};

/// Honesty label for **synthetic** runs (ADR-264 honesty rule). Plumbing only — the
/// synthetic embedder is deterministic but encodes no meaning.
pub const HONESTY_LABEL: &str = "subset fixture + synthetic embeddings — plumbing validation, \
NOT semantic retrieval quality; real recall requires Qwen3-VL-2B (blocked)";

/// Honesty label for **real** runs (`--embedder real`): genuine semantic embeddings,
/// but still over a tiny fixture (this is honest about scale, not quality).
pub const HONESTY_LABEL_REAL: &str = "real all-MiniLM-L6-v2 semantic embeddings over a small \
real eval fixture — semantic retrieval, still a tiny corpus vs full-scale";

/// Embedding width for the synthetic plumbing embedder. Kept small so the fixture
/// run is cheap; the real encoders are far wider (1024 Qwen3-VL / 768 CLIP surrogate).
const SYNTHETIC_DIM: usize = 128;

/// Path to the Node embedding sidecar (`all-MiniLM-L6-v2`), resolved at compile time
/// against THIS crate's manifest dir (the encoder crate can't assume the CLI layout, so
/// the CLI owns the path and passes it to [`SidecarEmbedder::new`]).
fn sidecar_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sidecar").join("embed_sidecar.mjs")
}

/// Which embedder backend the bench drives. Selected by `--embedder` (default `Real`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedderChoice {
    /// Real semantic `all-MiniLM-L6-v2` via the Node sidecar (default).
    #[default]
    Real,
    /// Deterministic non-semantic plumbing embedder.
    Synthetic,
}

impl EmbedderChoice {
    /// Parse `--embedder real|synthetic` (case-insensitive). Defaults handled by caller.
    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "real" | "minilm" | "sidecar" => Some(EmbedderChoice::Real),
            "synthetic" | "synth" => Some(EmbedderChoice::Synthetic),
            _ => None,
        }
    }
}

/// Arguments for the `benchmark` subcommand.
///
/// Mirrors the ADR-264 / `.metaharness/bench.json` command:
/// `benchmark --predictions <p> --ground-truth <gt> --metrics ndcg,mrr,recall@10 --queries <q>`.
/// `--predictions` is an OUTPUT path: the harness builds the index, runs the queries,
/// and writes the per-query ranked tile ids there before scoring (there is no separate
/// `search` step in the darwin command). Extra flags are optional and default to the
/// `Config::default()` M1 harness so the bare darwin command runs unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BenchArgs {
    /// Where to write the per-query ranked predictions JSON (also re-read for scoring).
    pub predictions: PathBuf,
    /// Ground-truth relevance JSON (qrels) for the subset fixture.
    pub ground_truth: PathBuf,
    /// Queries JSON (`query_id` + `text`) for the subset fixture.
    pub queries: PathBuf,
    /// Tiles directory (one `*.txt` placeholder per tile). When `None`, resolved as
    /// the `tiles/` subdir next to `--ground-truth` (the fixture layout).
    pub tiles_dir: Option<PathBuf>,
    /// Metrics to compute (parsed from a comma list, e.g. `ndcg,mrr,recall@10`).
    pub metrics: Vec<MetricKind>,
    /// Where to write the full [`BenchReport`] JSON. Defaults to
    /// `bench_output/pixelrag_bench.json` when `None`.
    pub report_out: Option<PathBuf>,
    /// Optional darwin harness-genome JSON; read-only, never controls the runtime
    /// beyond selecting the harness parameters (backend/batch/cache) via [`Config`].
    pub darwin_config: Option<PathBuf>,
    /// `k` cutoff for retrieval (default 10).
    pub k: usize,
    /// Override the embedding batch size (else from [`Config`]).
    pub batch_size: Option<usize>,
    /// Override the index backend (`hnsw`); else from [`Config`].
    pub index_backend: Option<IndexBackend>,
    /// Which embedder backend to use (`--embedder real|synthetic`, default `Real`).
    pub embedder: EmbedderChoice,
}

/// A single benchmark metric the harness can compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Recall@k — fraction of relevant items retrieved in the top-k.
    Recall(usize),
    /// Normalized Discounted Cumulative Gain at cutoff k.
    Ndcg(usize),
    /// Mean Reciprocal Rank.
    Mrr,
}

impl MetricKind {
    /// Parse a single metric token (`"ndcg"`, `"mrr"`, `"recall@10"`, `"ndcg@5"`).
    ///
    /// Splits on `@` for a cutoff suffix; default cutoff = 10 when omitted.
    pub fn parse(token: &str) -> Result<Self, MetricParseError> {
        let token = token.trim();
        let (name, k) = match token.split_once('@') {
            Some((name, k_str)) => {
                let k = k_str
                    .parse::<usize>()
                    .map_err(|_| MetricParseError { token: token.to_string() })?;
                (name, k)
            }
            None => (token, 10),
        };
        match name.to_ascii_lowercase().as_str() {
            "recall" => Ok(MetricKind::Recall(k)),
            "ndcg" => Ok(MetricKind::Ndcg(k)),
            "mrr" => Ok(MetricKind::Mrr),
            _ => Err(MetricParseError { token: token.to_string() }),
        }
    }
}

/// Error returned when a `--metrics` token cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricParseError {
    /// The offending token.
    pub token: String,
}

/// Retrieval-quality metrics for one benchmark run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RetrievalMetrics {
    /// recall@k keyed by cutoff k (e.g. {10: 0.0}).
    pub recall_at_k: Vec<(usize, f64)>,
    /// NDCG@k keyed by cutoff k.
    pub ndcg_at_k: Vec<(usize, f64)>,
    /// Mean Reciprocal Rank across all queries.
    pub mrr: f64,
    /// Number of queries scored.
    pub num_queries: usize,
}

/// Search-latency distribution (vec-sim retrieval only; rerank excluded).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LatencyMetrics {
    /// 50th percentile latency, milliseconds.
    pub p50_ms: f64,
    /// 95th percentile latency, milliseconds.
    pub p95_ms: f64,
    /// 99th percentile latency, milliseconds.
    pub p99_ms: f64,
    /// Number of latency samples collected.
    pub samples: usize,
}

/// Index-build cost for one benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BuildMetrics {
    /// Wall-clock seconds per 1000 docs (embed + add).
    pub seconds_per_1k_docs: f64,
    /// Total documents (tiles) indexed.
    pub docs_indexed: usize,
    /// Raw wall-clock milliseconds to build the whole index.
    pub total_build_ms: f64,
}

/// Memory-footprint metrics for one benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MemoryMetrics {
    /// Honest index footprint, bytes (f32 originals + id bookkeeping in M1).
    pub index_bytes: u64,
    /// Bytes of index + embeddings + metadata per indexed tile.
    pub bytes_per_doc: f64,
}

/// Complete report for one benchmark run — the unit darwin scores.
///
/// `quantization` and `batch_size` echo the harness parameters darwin evolved, so a
/// Pareto frontier of `(config, metrics)` pairs can be assembled downstream.
/// `honesty` MUST always equal [`HONESTY_LABEL`] (ADR-264 honesty rule).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BenchReport {
    /// Dataset name (echoed from the queries fixture, e.g. `pixelrag-subset-v0`).
    pub dataset: String,
    /// Quantization tier label echoed from the harness config (`none` in M1).
    pub quantization: String,
    /// Embedding batch size echoed from the harness config.
    pub batch_size: usize,
    /// Index backend label echoed from the harness config.
    pub index_backend: String,
    /// Embedder kind label (always `synthetic` in this environment).
    pub embedder: String,
    /// Retrieval-quality metrics.
    pub retrieval: RetrievalMetrics,
    /// Search-latency metrics.
    pub latency: LatencyMetrics,
    /// Index-build metrics.
    pub build: BuildMetrics,
    /// Memory metrics.
    pub memory: MemoryMetrics,
    /// Mandatory honesty label — always [`HONESTY_LABEL`].
    pub honesty: String,
}

/// Run the benchmark harness end to end.
///
/// Builds the index from the subset fixture (tiles → synthetic embeddings →
/// `ruvector-core` HNSW via `pixelrag-core`), runs every fixture query (timing each
/// search with [`Instant`]), writes the per-query ranked predictions to
/// `args.predictions`, scores the requested [`MetricKind`]s against ground truth,
/// assembles a [`BenchReport`] (with build/latency/memory + the honesty label),
/// serializes it to `args.report_out` (default `bench_output/pixelrag_bench.json`),
/// and prints a summary. This is the function the darwin harness shells out to per
/// candidate.
pub fn run(args: BenchArgs) -> Result<BenchReport, BenchError> {
    // ── 1. Resolve config / harness parameters (darwin-removable) ─────────────
    let mut config = match &args.darwin_config {
        Some(path) => Config::from_darwin_json(path).unwrap_or_else(|_| Config::default()),
        None => Config::default(),
    };
    if let Some(bs) = args.batch_size {
        config.batch_size = bs.max(1);
    }
    if let Some(backend) = args.index_backend {
        config.index_backend = backend;
    }
    let k = if args.k == 0 { 10 } else { args.k };

    // ── 2. Load the subset fixture ────────────────────────────────────────────
    let tiles_dir = args.tiles_dir.clone().unwrap_or_else(|| default_tiles_dir(&args.ground_truth));
    let tiles = load_tiles(&tiles_dir)?; // Vec<(tile_id, bytes)>
    let queries = load_queries(&args.queries)?; // (dataset, Vec<(query_id, text)>)
    let ground_truth = GroundTruth::load(&args.ground_truth)?;

    if tiles.is_empty() {
        return Err(BenchError { message: format!("no tiles found under {}", tiles_dir.display()) });
    }

    // ── 3. Build the index and time it ────────────────────────────────────────
    // Select the embedder behind a `Box<dyn Embedder>` so the index/query path below
    // is identical for both backends; the embedding DIM is taken from the chosen
    // backend (384 for real all-MiniLM, 128 for synthetic) — never hardcoded.
    let embedder_inner: Box<dyn pixelrag_encoder::Embedder> = match args.embedder {
        EmbedderChoice::Real => Box::new(SidecarEmbedder::new(sidecar_path())),
        EmbedderChoice::Synthetic => Box::new(SyntheticEmbedder::new(SYNTHETIC_DIM)),
    };
    let embedding_dim = embedder_inner.embedding_dim();
    // Labels for predictions JSON, report, and summary — derived once from the choice.
    let (embedder_label, honesty): (String, &str) = match args.embedder {
        EmbedderChoice::Real => ("real all-MiniLM-L6-v2".to_string(), HONESTY_LABEL_REAL),
        EmbedderChoice::Synthetic => ("synthetic".to_string(), HONESTY_LABEL),
    };
    let embedder = EncoderEmbedder::new(embedder_inner);
    let index = build_index(config.index_backend, embedding_dim)
        .map_err(|e| BenchError { message: format!("build_index: {e}") })?;
    let tiler = Tiler::default();
    let mut pipeline = Pipeline::new(config.clone(), tiler, embedder, index)
        .map_err(|e| BenchError { message: format!("Pipeline::new: {e}") })?;

    // Each tile is ingested under its fixture string id ("tile-NNN"); the pipeline
    // records that as the hit's `metadata.doc_id`, which is exactly what ground-truth
    // references — so predictions are scored directly on the fixture's string ids.
    let build_start = Instant::now();
    for (tile_id, bytes) in &tiles {
        // One tile per "document": ingest its raw bytes as a single pre-rendered page.
        pipeline
            .ingest_rendered(tile_id, &[bytes.clone()])
            .map_err(|e| BenchError { message: format!("ingest {tile_id}: {e}") })?;
    }
    let total_build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let docs_indexed = tiles.len();

    // ── 4. Run every query, timing each search; collect predictions ───────────
    // A query "image" is the query TEXT bytes embedded by the SAME backend as the
    // tiles. For `real`, this is genuine semantic all-MiniLM retrieval; for
    // `synthetic`, the mapping is non-semantic (byte-overlap only) — hence the
    // synthetic honesty label.
    let query_embedder: Box<dyn pixelrag_encoder::Embedder> = match args.embedder {
        EmbedderChoice::Real => Box::new(SidecarEmbedder::new(sidecar_path())),
        EmbedderChoice::Synthetic => Box::new(SyntheticEmbedder::new(SYNTHETIC_DIM)),
    };
    let mut predictions = Predictions::default();
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(queries.1.len());
    let req = SearchRequest { k, allowlist: None, rerank: false };

    for (query_id, text) in &queries.1 {
        let q: Embedding = embed_query_text(query_embedder.as_ref(), text, embedding_dim)
            .map_err(|e| BenchError { message: format!("embed query {query_id}: {e}") })?;
        let t0 = Instant::now();
        let hits = pipeline
            .search(&q, &req)
            .map_err(|e| BenchError { message: format!("search {query_id}: {e}") })?;
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let ranked: Vec<String> = hits.into_iter().map(|h| h.metadata.doc_id).collect();
        predictions.by_query.push((query_id.clone(), ranked));
    }

    // ── 5. Persist predictions JSON (output of --predictions) ─────────────────
    write_predictions(&args.predictions, &queries.0, &predictions, &embedder_label, honesty)?;

    // ── 6. Score requested metrics ────────────────────────────────────────────
    // Default to ndcg@k/mrr/recall@k if no --metrics were supplied.
    let metrics = if args.metrics.is_empty() {
        vec![MetricKind::Ndcg(k), MetricKind::Mrr, MetricKind::Recall(k)]
    } else {
        args.metrics.clone()
    };

    let mut retrieval = RetrievalMetrics { num_queries: predictions.by_query.len(), ..Default::default() };
    for m in &metrics {
        match *m {
            MetricKind::Recall(kk) => {
                retrieval.recall_at_k.push((kk, recall_at_k(&predictions, &ground_truth, kk)));
            }
            MetricKind::Ndcg(kk) => {
                retrieval.ndcg_at_k.push((kk, ndcg_at_k(&predictions, &ground_truth, kk)));
            }
            MetricKind::Mrr => {
                retrieval.mrr = mrr(&predictions, &ground_truth);
            }
        }
    }

    // ── 7. Assemble report (latency / build / memory + honesty label) ─────────
    let latency = percentiles(&latencies_ms);
    // Honest index footprint, read straight from the live backend so it stays
    // correct across backends (HNSW = n*dim*4 + n*usize; IVF-Flat additionally
    // counts its k-means centroid table) instead of reconstructing one formula.
    let index_bytes = pipeline.index_memory_bytes() as u64;

    let build = BuildMetrics {
        seconds_per_1k_docs: if docs_indexed == 0 {
            0.0
        } else {
            (total_build_ms / 1000.0) / (docs_indexed as f64) * 1000.0
        },
        docs_indexed,
        total_build_ms,
    };
    let memory = MemoryMetrics {
        index_bytes,
        bytes_per_doc: if docs_indexed == 0 { 0.0 } else { index_bytes as f64 / docs_indexed as f64 },
    };

    let report = BenchReport {
        dataset: queries.0.clone(),
        quantization: "none".to_string(), // M1: f32 originals (no scalar quantization wired yet).
        batch_size: config.batch_size,
        index_backend: backend_label(config.index_backend),
        embedder: embedder_label.clone(),
        retrieval,
        latency,
        build,
        memory,
        honesty: honesty.to_string(),
    };

    // ── 8. Serialize report + print summary ───────────────────────────────────
    let report_path = args
        .report_out
        .clone()
        .unwrap_or_else(|| PathBuf::from("bench_output/pixelrag_bench.json"));
    write_report(&report_path, &report)?;
    print_summary(&report, &report_path);

    Ok(report)
}

/// Embed a query string via the selected embedder, treating the text bytes as a
/// `Gray8` 1×N tile (mirrors the `EncoderEmbedder` tile→image bridge in pixelrag-core).
///
/// `dim` is the chosen backend's [`pixelrag_encoder::Embedder::embedding_dim`], used only
/// for the defensive zero fallback so the vector width always matches the index.
fn embed_query_text(
    embedder: &dyn pixelrag_encoder::Embedder,
    text: &str,
    dim: usize,
) -> Result<Embedding, BenchError> {
    use pixelrag_encoder::{Image, PixelFormat};
    let bytes = text.as_bytes().to_vec();
    let width = bytes.len().max(1) as u32;
    let img = Image { pixels: bytes, width, height: 1, format: PixelFormat::Gray8 };
    match embedder.embed(&img) {
        Ok(e) => Ok(e.vector),
        // The real sidecar can genuinely fail (node missing / non-zero exit); surface
        // it. The synthetic embedder never fails, so this branch is real-backend only.
        Err(e) => {
            let _ = dim; // kept for the explicit zero-fallback contract if ever needed
            Err(BenchError { message: e.to_string() })
        }
    }
}

/// Default tiles dir: the `tiles/` sibling of the ground-truth file.
fn default_tiles_dir(ground_truth: &Path) -> PathBuf {
    ground_truth
        .parent()
        .map(|p| p.join("tiles"))
        .unwrap_or_else(|| PathBuf::from("tiles"))
}

/// Load placeholder tiles: every `*.txt` under `dir`, sorted by file stem, returning
/// `(tile_id, raw_bytes)`. The `tile_id` is the file stem (`tile-000`), which is what
/// ground-truth references.
fn load_tiles(dir: &Path) -> Result<Vec<(String, Vec<u8>)>, BenchError> {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    let read = std::fs::read_dir(dir)
        .map_err(|e| BenchError { message: format!("read tiles dir {}: {e}", dir.display()) })?;
    for ent in read {
        let ent = ent.map_err(|e| BenchError { message: format!("tiles entry: {e}") })?;
        let path = ent.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .ok_or_else(|| BenchError { message: format!("bad tile filename: {}", path.display()) })?;
        let bytes = std::fs::read(&path)
            .map_err(|e| BenchError { message: format!("read tile {}: {e}", path.display()) })?;
        entries.push((stem, bytes));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// Load queries JSON → `(dataset, Vec<(query_id, text)>)`.
fn load_queries(path: &Path) -> Result<(String, Vec<(String, String)>), BenchError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| BenchError { message: format!("read queries {}: {e}", path.display()) })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BenchError { message: format!("parse queries: {e}") })?;
    let dataset = json
        .get("dataset")
        .and_then(|v| v.as_str())
        .unwrap_or("pixelrag-subset")
        .to_string();
    let arr = json
        .get("queries")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BenchError { message: "queries.json missing 'queries' array".into() })?;
    let mut out = Vec::with_capacity(arr.len());
    for q in arr {
        let id = q
            .get("query_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BenchError { message: "query missing 'query_id'".into() })?
            .to_string();
        let text = q.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        out.push((id, text));
    }
    Ok((dataset, out))
}

/// Compute recall@k for a set of ranked predictions against ground-truth qrels.
///
/// For each query, recall@k = |relevant ∩ top-k| / |relevant|; averaged over queries
/// that have at least one relevant item.
pub fn recall_at_k(predictions: &Predictions, ground_truth: &GroundTruth, k: usize) -> f64 {
    let mut sum = 0.0;
    let mut scored = 0usize;
    for (qid, ranked) in &predictions.by_query {
        let Some(rel) = ground_truth.relevant_ids(qid) else { continue };
        if rel.is_empty() {
            continue;
        }
        let topk: std::collections::HashSet<&String> = ranked.iter().take(k).collect();
        let hit = rel.iter().filter(|r| topk.contains(r)).count();
        sum += hit as f64 / rel.len() as f64;
        scored += 1;
    }
    if scored == 0 {
        0.0
    } else {
        sum / scored as f64
    }
}

/// Compute NDCG@k for ranked predictions against graded/binary relevance.
///
/// DCG@k = Σ rel_i / log2(i+1) for i in 1..=k; normalized by ideal DCG@k.
pub fn ndcg_at_k(predictions: &Predictions, ground_truth: &GroundTruth, k: usize) -> f64 {
    let mut sum = 0.0;
    let mut scored = 0usize;
    for (qid, ranked) in &predictions.by_query {
        let Some(grades) = ground_truth.graded(qid) else { continue };
        if grades.is_empty() {
            continue;
        }
        // DCG over the predicted ranking.
        let mut dcg = 0.0;
        for (i, doc) in ranked.iter().take(k).enumerate() {
            let rel = grades.get(doc).copied().unwrap_or(0) as f64;
            if rel > 0.0 {
                dcg += rel / ((i as f64 + 2.0).log2());
            }
        }
        // Ideal DCG: relevances sorted descending.
        let mut ideal: Vec<u32> = grades.values().copied().collect();
        ideal.sort_unstable_by(|a, b| b.cmp(a));
        let mut idcg = 0.0;
        for (i, &rel) in ideal.iter().take(k).enumerate() {
            if rel > 0 {
                idcg += rel as f64 / ((i as f64 + 2.0).log2());
            }
        }
        if idcg > 0.0 {
            sum += dcg / idcg;
            scored += 1;
        }
    }
    if scored == 0 {
        0.0
    } else {
        sum / scored as f64
    }
}

/// Compute Mean Reciprocal Rank for ranked predictions.
///
/// MRR = mean over queries of 1 / rank-of-first-relevant (0 if none in list).
pub fn mrr(predictions: &Predictions, ground_truth: &GroundTruth) -> f64 {
    let mut sum = 0.0;
    let mut scored = 0usize;
    for (qid, ranked) in &predictions.by_query {
        let Some(rel) = ground_truth.relevant_ids(qid) else { continue };
        if rel.is_empty() {
            continue;
        }
        scored += 1;
        let relset: std::collections::HashSet<&String> = rel.iter().collect();
        for (i, doc) in ranked.iter().enumerate() {
            if relset.contains(doc) {
                sum += 1.0 / (i as f64 + 1.0);
                break;
            }
        }
    }
    if scored == 0 {
        0.0
    } else {
        sum / scored as f64
    }
}

/// Compute p50/p95/p99 from a slice of latency samples (milliseconds).
fn percentiles(samples_ms: &[f64]) -> LatencyMetrics {
    if samples_ms.is_empty() {
        return LatencyMetrics::default();
    }
    let mut s = samples_ms.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |p: f64| -> f64 {
        // Nearest-rank percentile.
        let rank = (p * (s.len() as f64)).ceil() as usize;
        let idx = rank.saturating_sub(1).min(s.len() - 1);
        s[idx]
    };
    LatencyMetrics {
        p50_ms: pick(0.50),
        p95_ms: pick(0.95),
        p99_ms: pick(0.99),
        samples: s.len(),
    }
}

fn backend_label(b: IndexBackend) -> String {
    match b {
        IndexBackend::Hnsw => "hnsw",
        IndexBackend::IvfFlat => "ivf-flat",
        IndexBackend::IvfSq => "ivf-sq",
    }
    .to_string()
}

/// Per-query ranked retrieval results.
///
/// Conceptually `query_id -> [tile_id ordered by descending score]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Predictions {
    /// Ranked `(query_id, ordered tile_ids)` entries.
    pub by_query: Vec<(String, Vec<String>)>,
}

impl Predictions {
    /// Load predictions from a `pixelrag-cli` results JSON file.
    ///
    /// Accepts either `{"results":[{"query_id":..,"tiles":[..]}]}` (what this harness
    /// writes) or a bare `{query_id: [tile_id, ...]}` object. Retained as public API
    /// for the future split `search` → `benchmark --predictions <existing>` flow; the
    /// current single-shot `run` scores the predictions it just generated in-memory.
    #[allow(dead_code)]
    pub fn load(path: &std::path::Path) -> Result<Self, BenchError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| BenchError { message: format!("read predictions {}: {e}", path.display()) })?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| BenchError { message: format!("parse predictions: {e}") })?;
        let mut out = Predictions::default();
        if let Some(arr) = json.get("results").and_then(|v| v.as_array()) {
            for r in arr {
                let qid = r.get("query_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let tiles = r
                    .get("tiles")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                out.by_query.push((qid, tiles));
            }
        } else if let Some(obj) = json.as_object() {
            for (qid, v) in obj {
                if let Some(a) = v.as_array() {
                    let tiles = a.iter().filter_map(|t| t.as_str().map(String::from)).collect();
                    out.by_query.push((qid.clone(), tiles));
                }
            }
        }
        Ok(out)
    }
}

/// Ground-truth relevance (qrels).
///
/// Conceptually `query_id -> ranked [tile_id]` (most relevant first). Graded relevance
/// is derived from rank position (first = highest grade) so NDCG rewards ordering.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroundTruth {
    /// `(query_id, ranked relevant tile_ids)` entries.
    pub by_query: Vec<(String, Vec<String>)>,
}

impl GroundTruth {
    /// Load ground-truth qrels JSON (`{"relevance":[{"query_id":..,"relevant":[..]}]}`).
    pub fn load(path: &std::path::Path) -> Result<Self, BenchError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| BenchError { message: format!("read ground-truth {}: {e}", path.display()) })?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| BenchError { message: format!("parse ground-truth: {e}") })?;
        let arr = json
            .get("relevance")
            .and_then(|v| v.as_array())
            .ok_or_else(|| BenchError { message: "ground-truth missing 'relevance' array".into() })?;
        let mut out = GroundTruth::default();
        for r in arr {
            let qid = r
                .get("query_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| BenchError { message: "relevance entry missing 'query_id'".into() })?
                .to_string();
            let rel = r
                .get("relevant")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                .unwrap_or_default();
            out.by_query.push((qid, rel));
        }
        Ok(out)
    }

    /// Ranked relevant tile ids for a query, if present.
    fn relevant_ids(&self, qid: &str) -> Option<&Vec<String>> {
        self.by_query.iter().find(|(q, _)| q == qid).map(|(_, r)| r)
    }

    /// Graded relevance map (`tile_id -> grade`) derived from rank: the first
    /// relevant tile gets the highest grade, decreasing by position.
    fn graded(&self, qid: &str) -> Option<HashMap<String, u32>> {
        let rel = self.relevant_ids(qid)?;
        let n = rel.len() as u32;
        let mut map = HashMap::new();
        for (i, tile) in rel.iter().enumerate() {
            map.insert(tile.clone(), n - i as u32);
        }
        Some(map)
    }
}

/// Write the per-query predictions JSON consumed by scoring + emitted to `--predictions`.
fn write_predictions(
    path: &Path,
    dataset: &str,
    predictions: &Predictions,
    embedder_label: &str,
    honesty: &str,
) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BenchError { message: format!("create {}: {e}", parent.display()) })?;
        }
    }
    let results: Vec<serde_json::Value> = predictions
        .by_query
        .iter()
        .map(|(qid, tiles)| {
            serde_json::json!({ "query_id": qid, "tiles": tiles })
        })
        .collect();
    let doc = serde_json::json!({
        "_honesty": honesty,
        "dataset": dataset,
        "embedder": embedder_label,
        "results": results,
    });
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| BenchError { message: format!("serialize predictions: {e}") })?;
    std::fs::write(path, text)
        .map_err(|e| BenchError { message: format!("write predictions {}: {e}", path.display()) })
}

/// Serialize the [`BenchReport`] to JSON (the unit darwin scores).
fn write_report(path: &Path, report: &BenchReport) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BenchError { message: format!("create {}: {e}", parent.display()) })?;
        }
    }
    let recall: Vec<serde_json::Value> = report
        .retrieval
        .recall_at_k
        .iter()
        .map(|(k, v)| serde_json::json!({ "k": k, "value": v }))
        .collect();
    let ndcg: Vec<serde_json::Value> = report
        .retrieval
        .ndcg_at_k
        .iter()
        .map(|(k, v)| serde_json::json!({ "k": k, "value": v }))
        .collect();
    let doc = serde_json::json!({
        "honesty": report.honesty,
        "dataset": report.dataset,
        "config": {
            "index_backend": report.index_backend,
            "batch_size": report.batch_size,
            "quantization": report.quantization,
            "embedder": report.embedder,
        },
        "retrieval": {
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "mrr": report.retrieval.mrr,
            "num_queries": report.retrieval.num_queries,
        },
        "latency_ms": {
            "p50": report.latency.p50_ms,
            "p95": report.latency.p95_ms,
            "p99": report.latency.p99_ms,
            "samples": report.latency.samples,
        },
        "build": {
            "seconds_per_1k_docs": report.build.seconds_per_1k_docs,
            "docs_indexed": report.build.docs_indexed,
            "total_build_ms": report.build.total_build_ms,
        },
        "memory": {
            "index_bytes": report.memory.index_bytes,
            "bytes_per_doc": report.memory.bytes_per_doc,
        },
    });
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| BenchError { message: format!("serialize report: {e}") })?;
    std::fs::write(path, text)
        .map_err(|e| BenchError { message: format!("write report {}: {e}", path.display()) })
}

/// Print a short human summary; always leads with the honesty label.
fn print_summary(report: &BenchReport, report_path: &Path) {
    println!("PixelRAG benchmark — {}", report.dataset);
    println!("HONESTY: {}", report.honesty);
    println!(
        "  config: backend={} batch={} quant={} embedder={}",
        report.index_backend, report.batch_size, report.quantization, report.embedder
    );
    for (k, v) in &report.retrieval.recall_at_k {
        println!("  recall@{k} = {v:.4}");
    }
    for (k, v) in &report.retrieval.ndcg_at_k {
        println!("  ndcg@{k}   = {v:.4}");
    }
    println!("  mrr       = {:.4} (n={})", report.retrieval.mrr, report.retrieval.num_queries);
    println!(
        "  latency   p50={:.3}ms p95={:.3}ms p99={:.3}ms (n={})",
        report.latency.p50_ms, report.latency.p95_ms, report.latency.p99_ms, report.latency.samples
    );
    println!(
        "  build     {:.2}ms total, {:.2}s/1k docs ({} docs)",
        report.build.total_build_ms, report.build.seconds_per_1k_docs, report.build.docs_indexed
    );
    println!(
        "  memory    {} index bytes, {:.1} bytes/doc",
        report.memory.index_bytes, report.memory.bytes_per_doc
    );
    println!("  report → {}", report_path.display());
}

/// Error type for the benchmark harness.
#[derive(Debug)]
pub struct BenchError {
    /// Human-readable description of the failure (I/O, parse, or metric error).
    pub message: String,
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for BenchError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn preds(rows: &[(&str, &[&str])]) -> Predictions {
        Predictions {
            by_query: rows
                .iter()
                .map(|(q, ts)| (q.to_string(), ts.iter().map(|t| t.to_string()).collect()))
                .collect(),
        }
    }

    fn gt(rows: &[(&str, &[&str])]) -> GroundTruth {
        GroundTruth {
            by_query: rows
                .iter()
                .map(|(q, ts)| (q.to_string(), ts.iter().map(|t| t.to_string()).collect()))
                .collect(),
        }
    }

    #[test]
    fn metric_parse() {
        assert_eq!(MetricKind::parse("recall@10").unwrap(), MetricKind::Recall(10));
        assert_eq!(MetricKind::parse("ndcg").unwrap(), MetricKind::Ndcg(10));
        assert_eq!(MetricKind::parse("ndcg@5").unwrap(), MetricKind::Ndcg(5));
        assert_eq!(MetricKind::parse("mrr").unwrap(), MetricKind::Mrr);
        assert!(MetricKind::parse("bogus").is_err());
    }

    #[test]
    fn recall_perfect_and_zero() {
        let p = preds(&[("q1", &["a", "b"]), ("q2", &["x"])]);
        let g = gt(&[("q1", &["a"]), ("q2", &["y"])]);
        // q1: a in top-k → 1.0; q2: y not retrieved → 0.0; mean = 0.5
        assert!((recall_at_k(&p, &g, 10) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn mrr_first_relevant_rank() {
        let p = preds(&[("q1", &["x", "a", "b"])]); // first relevant at rank 2
        let g = gt(&[("q1", &["a"])]);
        assert!((mrr(&p, &g) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        let p = preds(&[("q1", &["a", "b"])]);
        let g = gt(&[("q1", &["a", "b"])]); // ranked relevance a > b
        assert!((ndcg_at_k(&p, &g, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn percentiles_basic() {
        let l = percentiles(&[1.0, 2.0, 3.0, 4.0, 100.0]);
        assert_eq!(l.samples, 5);
        assert!(l.p99_ms >= l.p50_ms);
    }
}
