//! # bench_visual — REAL visual-RAG benchmark (CLIP ViT-B/32 cross-modal)
//!
//! Per **ADR-264** §Validation, the visual path. Unlike the text benchmark
//! ([`crate::bench`], all-MiniLM over tile bytes), this path exercises a genuine
//! **visual encoder**: it embeds rendered document **screenshots** and the query
//! **text** into one shared CLIP space, then does text→image retrieval over a
//! `ruvector` ANN index.
//!
//! ## Honesty
//!
//! These are REAL CLIP ViT-B/32 cross-modal embeddings (CPU/WASM via the verified
//! `clip_sidecar.mjs`). The corpus is a tiny 8-doc fixture — honest about scale,
//! not a SOTA visual-document claim. Qwen3-VL / ColPali is the GPU upgrade.
//!
//! ## Flow
//!
//! 1. Read `manifest.json`, `queries.json`, `ground-truth.json` from the fixture.
//! 2. Shell `clip_sidecar.mjs` **once** with `{images:[abs png paths], texts:[query texts]}`
//!    → `image_vectors` (one per doc, manifest order) + `text_vectors` (one per query).
//! 3. Build a `ruvector` index (`--index-backend hnsw|ivf-flat`) over the IMAGE
//!    vectors; the index id is the manifest position, mapped back to the doc-id.
//! 4. For each query text vector, `AnnIndex::search` top-k → top-1 / recall@k /
//!    ndcg@k / mrr against ground-truth.
//! 5. Emit a [`crate::bench::BenchReport`] with the CLIP embedder label + honesty.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use pixelrag_core::config::{Config, IndexBackend};
use pixelrag_core::index::build_index;

use crate::bench::{
    BenchError, BenchReport, BuildMetrics, GroundTruth, LatencyMetrics, MemoryMetrics, MetricKind,
    Predictions, RetrievalMetrics,
};

/// Honesty label for the REAL visual (CLIP) path — never elided from a report.
pub const HONESTY_LABEL_VISUAL: &str = "real CLIP ViT-B/32 cross-modal embeddings (CPU/WASM); \
text→image retrieval over rendered document screenshots — a real visual encoder; \
Qwen3-VL/ColPali is the GPU upgrade";

/// Embedder label echoed into the report config block.
pub const EMBEDDER_LABEL_VISUAL: &str = "clip-vit-base-patch32 (visual)";

/// Path to the CLIP visual sidecar, resolved at compile time against THIS crate's
/// manifest dir (the CLI owns the sidecar layout; the encoder crate cannot).
fn clip_sidecar_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sidecar").join("clip_sidecar.mjs")
}

/// Arguments for `benchmark --mode visual`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualArgs {
    /// Visual fixture directory holding `manifest.json` / `queries.json` /
    /// `ground-truth.json` and the screenshot images.
    pub fixture_dir: PathBuf,
    /// Where to write per-query ranked predictions JSON.
    pub predictions: PathBuf,
    /// Where to write the full [`BenchReport`] JSON.
    pub report_out: Option<PathBuf>,
    /// Metrics to compute (default ndcg@k / mrr / recall@k).
    pub metrics: Vec<MetricKind>,
    /// `k` cutoff for retrieval (default 10).
    pub k: usize,
    /// Index backend override (`hnsw` default, or `ivf-flat`).
    pub index_backend: Option<IndexBackend>,
}

impl Default for VisualArgs {
    fn default() -> Self {
        VisualArgs {
            fixture_dir: PathBuf::from("tests/fixtures/pixelrag/visual"),
            predictions: PathBuf::from("bench_output/pixelrag_visual_results.json"),
            report_out: None,
            metrics: Vec::new(),
            k: 10,
            index_backend: None,
        }
    }
}

/// One manifest entry: a document id + the absolute screenshot path.
#[derive(Debug, Clone)]
struct ManifestDoc {
    id: String,
    image_abs: PathBuf,
}

/// Compile-time path to the headless render sidecar (`render_sidecar.mjs`),
/// resolved against THIS crate's manifest dir — the single source of truth the
/// render-on-missing path hands to [`pixelrag_render::Renderer`].
fn render_sidecar_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sidecar").join("render_sidecar.mjs")
}

/// The deserialized CLIP sidecar response.
#[derive(serde::Deserialize)]
struct ClipResponse {
    #[allow(dead_code)]
    model: String,
    dim: usize,
    image_vectors: Vec<Vec<f32>>,
    text_vectors: Vec<Vec<f32>>,
}

/// Run the visual benchmark end to end and return the [`BenchReport`].
pub fn run(args: VisualArgs) -> Result<BenchReport, BenchError> {
    let k = if args.k == 0 { 10 } else { args.k };
    let backend = args.index_backend.unwrap_or(Config::default().index_backend);

    // ── 1. Load fixture (manifest / queries / ground-truth) ──────────────────
    let manifest_path = args.fixture_dir.join("manifest.json");
    let queries_path = args.fixture_dir.join("queries.json");
    let gt_path = args.fixture_dir.join("ground-truth.json");

    let docs = load_manifest(&manifest_path, &args.fixture_dir)?;
    let queries = load_visual_queries(&queries_path)?; // Vec<(query_id, text)>
    let ground_truth = load_visual_ground_truth(&gt_path)?;

    if docs.is_empty() {
        return Err(BenchError { message: format!("no docs in manifest {}", manifest_path.display()) });
    }
    if queries.is_empty() {
        return Err(BenchError { message: format!("no queries in {}", queries_path.display()) });
    }

    // ── 2. ONE CLIP sidecar round-trip: images + texts → shared-space vectors ─
    let images: Vec<String> = docs.iter().map(|d| d.image_abs.to_string_lossy().into_owned()).collect();
    let texts: Vec<String> = queries.iter().map(|(_, t)| t.clone()).collect();
    let clip = run_clip_sidecar(&images, &texts)?;

    if clip.image_vectors.len() != docs.len() {
        return Err(BenchError {
            message: format!(
                "CLIP returned {} image vectors for {} docs",
                clip.image_vectors.len(),
                docs.len()
            ),
        });
    }
    if clip.text_vectors.len() != queries.len() {
        return Err(BenchError {
            message: format!(
                "CLIP returned {} text vectors for {} queries",
                clip.text_vectors.len(),
                queries.len()
            ),
        });
    }
    let dim = clip.dim;
    if dim == 0 {
        return Err(BenchError { message: "CLIP reported dim=0".into() });
    }

    // ── 3. Build the ruvector index over the IMAGE vectors ───────────────────
    // External id = manifest position; mapped back to doc-id for scoring.
    let mut index = build_index(backend, dim)
        .map_err(|e| BenchError { message: format!("build_index: {e}") })?;
    let build_start = Instant::now();
    for (i, vec) in clip.image_vectors.iter().enumerate() {
        if vec.len() != dim {
            return Err(BenchError {
                message: format!("image vector {i} width {} != dim {}", vec.len(), dim),
            });
        }
        index
            .add(i, vec.clone())
            .map_err(|e| BenchError { message: format!("index add {i}: {e}") })?;
    }
    index
        .finalize()
        .map_err(|e| BenchError { message: format!("index finalize: {e}") })?;
    let total_build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let docs_indexed = docs.len();

    // ── 4. text→image retrieval per query; collect predictions + latency ─────
    let mut predictions = Predictions::default();
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(queries.len());
    for ((query_id, _text), qvec) in queries.iter().zip(clip.text_vectors.iter()) {
        if qvec.len() != dim {
            return Err(BenchError {
                message: format!("query '{query_id}' vector width {} != dim {}", qvec.len(), dim),
            });
        }
        let t0 = Instant::now();
        let hits = index
            .search(qvec, k)
            .map_err(|e| BenchError { message: format!("search {query_id}: {e}") })?;
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        // Map ANN ids (manifest positions) back to doc-ids.
        let ranked: Vec<String> = hits
            .into_iter()
            .filter_map(|h| docs.get(h.id).map(|d| d.id.clone()))
            .collect();
        predictions.by_query.push((query_id.clone(), ranked));
    }

    // ── 5. Persist predictions JSON ──────────────────────────────────────────
    write_visual_predictions(&args.predictions, &predictions)?;

    // ── 6. Score metrics (top-1 always; recall@k / ndcg@k / mrr) ─────────────
    let metrics = if args.metrics.is_empty() {
        vec![MetricKind::Ndcg(k), MetricKind::Mrr, MetricKind::Recall(k)]
    } else {
        args.metrics.clone()
    };
    let mut retrieval = RetrievalMetrics { num_queries: predictions.by_query.len(), ..Default::default() };
    // top-1 accuracy is reported as recall@1 (1 relevant doc per query in this fixture).
    let top1 = crate::bench::recall_at_k(&predictions, &ground_truth, 1);
    retrieval.recall_at_k.push((1, top1));
    for m in &metrics {
        match *m {
            MetricKind::Recall(kk) => {
                if kk != 1 {
                    retrieval
                        .recall_at_k
                        .push((kk, crate::bench::recall_at_k(&predictions, &ground_truth, kk)));
                }
            }
            MetricKind::Ndcg(kk) => {
                retrieval
                    .ndcg_at_k
                    .push((kk, crate::bench::ndcg_at_k(&predictions, &ground_truth, kk)));
            }
            MetricKind::Mrr => {
                retrieval.mrr = crate::bench::mrr(&predictions, &ground_truth);
            }
        }
    }

    // ── 7. Assemble report (latency / build / memory + honesty) ──────────────
    let latency = percentiles(&latencies_ms);
    let index_bytes = index.memory_bytes() as u64;
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
        dataset: "pixelrag-visual-subset".to_string(),
        quantization: "none".to_string(),
        batch_size: docs_indexed.max(1), // one CLIP round-trip embeds the whole corpus
        index_backend: backend_label(backend),
        embedder: EMBEDDER_LABEL_VISUAL.to_string(),
        retrieval,
        latency,
        build,
        memory,
        honesty: HONESTY_LABEL_VISUAL.to_string(),
    };

    // ── 8. Serialize report + print summary ──────────────────────────────────
    let report_path = args
        .report_out
        .clone()
        .unwrap_or_else(|| PathBuf::from("bench_output/pixelrag_visual_bench.json"));
    crate::bench::write_report_public(&report_path, &report)?;
    print_visual_summary(&report, top1, &report_path);

    Ok(report)
}

/// Shell `clip_sidecar.mjs` once with `{images, texts}` and parse the response.
fn run_clip_sidecar(images: &[String], texts: &[String]) -> Result<ClipResponse, BenchError> {
    let sidecar = match std::env::var_os("PIXELRAG_CLIP_SIDECAR") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => clip_sidecar_path(),
    };
    let node_bin = std::env::var("PIXELRAG_NODE").unwrap_or_else(|_| "node".to_string());

    let request = serde_json::json!({ "images": images, "texts": texts });
    let payload = serde_json::to_vec(&request)
        .map_err(|e| BenchError { message: format!("serialize CLIP request: {e}") })?;

    let mut child = Command::new(&node_bin)
        .arg(&sidecar)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BenchError {
            message: format!(
                "spawn `{} {}` failed: {e} (is Node installed and on PATH? set PIXELRAG_NODE)",
                node_bin,
                sidecar.display()
            ),
        })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| BenchError { message: "CLIP sidecar stdin unavailable".into() })?;
        stdin
            .write_all(&payload)
            .map_err(|e| BenchError { message: format!("write CLIP request: {e}") })?;
    }
    child.stdin = None; // EOF for the sidecar's readStdin()

    let output = child
        .wait_with_output()
        .map_err(|e| BenchError { message: format!("wait for CLIP sidecar: {e}") })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(BenchError {
            message: format!("CLIP sidecar exited with {}: {}", output.status, stderr.trim()),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    if line.is_empty() {
        return Err(BenchError { message: "CLIP sidecar produced empty stdout".into() });
    }
    serde_json::from_str(line)
        .map_err(|e| BenchError { message: format!("parse CLIP response JSON: {e}") })
}

/// Load `manifest.json` → ordered docs, resolving each image path to an absolute
/// path. The manifest `image` field may name a subdir that no longer matches the
/// on-disk layout, so we also fall back to `<fixture_dir>/images/<doc-id>.png`.
fn load_manifest(path: &Path, fixture_dir: &Path) -> Result<Vec<ManifestDoc>, BenchError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| BenchError { message: format!("read manifest {}: {e}", path.display()) })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BenchError { message: format!("parse manifest: {e}") })?;
    let arr = json
        .as_array()
        .ok_or_else(|| BenchError { message: "manifest must be a JSON array".into() })?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BenchError { message: "manifest entry missing 'id'".into() })?
            .to_string();
        let url = entry.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let image_rel = entry.get("image").and_then(|v| v.as_str()).unwrap_or("");

        // Candidate 1: the manifest's declared relative path under the fixture dir.
        let primary = fixture_dir.join(image_rel);
        // Candidate 2: the canonical on-disk layout `images/<id>.png`.
        let fallback = fixture_dir.join("images").join(format!("{id}.png"));
        let mut image_abs = if !image_rel.is_empty() && primary.exists() {
            primary
        } else if fallback.exists() {
            fallback
        } else if !image_rel.is_empty() {
            primary
        } else {
            fallback
        };

        // REAL render-on-missing: if the screenshot is absent on disk, render it
        // from the manifest URL via the headless `pixelrag-render` sidecar adaptor
        // (the same renderer that produced the committed fixture). This keeps the
        // visual path self-healing instead of failing inside the CLIP sidecar.
        if !image_abs.exists() && !url.is_empty() {
            let images_dir = fixture_dir.join("images");
            match pixelrag_render::render(render_sidecar_path(), &[url.as_str()], &images_dir) {
                Ok(rendered) => {
                    if let Some(first) = rendered.into_iter().next() {
                        // The sidecar names by input order (`doc-00`); move/keep it
                        // under the manifest id so subsequent runs hit the fallback.
                        let target = images_dir.join(format!("{id}.png"));
                        if first.path != target {
                            let _ = std::fs::rename(&first.path, &target);
                        }
                        image_abs = if target.exists() { target } else { first.path };
                    }
                }
                Err(e) => {
                    return Err(BenchError {
                        message: format!(
                            "image for '{id}' missing and render failed: {e}. Pre-render the \
                             fixture or place the PNG at {}",
                            image_abs.display()
                        ),
                    });
                }
            }
        }

        let image_abs = std::fs::canonicalize(&image_abs).unwrap_or(image_abs);
        out.push(ManifestDoc { id, image_abs });
    }
    Ok(out)
}

/// Load the visual `queries.json` — a bare array of `{query_id, text}`.
fn load_visual_queries(path: &Path) -> Result<Vec<(String, String)>, BenchError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| BenchError { message: format!("read queries {}: {e}", path.display()) })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BenchError { message: format!("parse queries: {e}") })?;
    let arr = json
        .as_array()
        .ok_or_else(|| BenchError { message: "visual queries.json must be a JSON array".into() })?;
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
    Ok(out)
}

/// Load the visual `ground-truth.json` — a bare `{query_id: [doc-id, …]}` object —
/// into the [`GroundTruth`] qrels structure the scoring helpers consume.
fn load_visual_ground_truth(path: &Path) -> Result<GroundTruth, BenchError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| BenchError { message: format!("read ground-truth {}: {e}", path.display()) })?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BenchError { message: format!("parse ground-truth: {e}") })?;
    let obj = json
        .as_object()
        .ok_or_else(|| BenchError { message: "visual ground-truth.json must be a JSON object".into() })?;
    let mut gt = GroundTruth::default();
    for (qid, v) in obj {
        let rel: Vec<String> = v
            .as_array()
            .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
            .unwrap_or_default();
        gt.by_query.push((qid.clone(), rel));
    }
    Ok(gt)
}

/// Write the per-query predictions JSON (mirrors the text bench's results shape).
fn write_visual_predictions(path: &Path, predictions: &Predictions) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BenchError { message: format!("create {}: {e}", parent.display()) })?;
        }
    }
    let results: Vec<serde_json::Value> = predictions
        .by_query
        .iter()
        .map(|(qid, docs)| serde_json::json!({ "query_id": qid, "docs": docs }))
        .collect();
    let doc = serde_json::json!({
        "_honesty": HONESTY_LABEL_VISUAL,
        "dataset": "pixelrag-visual-subset",
        "embedder": EMBEDDER_LABEL_VISUAL,
        "results": results,
    });
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| BenchError { message: format!("serialize predictions: {e}") })?;
    std::fs::write(path, text)
        .map_err(|e| BenchError { message: format!("write predictions {}: {e}", path.display()) })
}

/// p50/p95/p99 from latency samples (nearest-rank), local copy of the text bench's.
fn percentiles(samples_ms: &[f64]) -> LatencyMetrics {
    if samples_ms.is_empty() {
        return LatencyMetrics::default();
    }
    let mut s = samples_ms.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |p: f64| -> f64 {
        let rank = (p * (s.len() as f64)).ceil() as usize;
        let idx = rank.saturating_sub(1).min(s.len() - 1);
        s[idx]
    };
    LatencyMetrics { p50_ms: pick(0.50), p95_ms: pick(0.95), p99_ms: pick(0.99), samples: s.len() }
}

fn backend_label(b: IndexBackend) -> String {
    match b {
        IndexBackend::Hnsw => "hnsw",
        IndexBackend::IvfFlat => "ivf-flat",
        IndexBackend::IvfSq => "ivf-sq",
    }
    .to_string()
}

/// Print a short visual-bench summary, leading with the honesty label.
fn print_visual_summary(report: &BenchReport, top1: f64, report_path: &Path) {
    println!("PixelRAG VISUAL benchmark — {}", report.dataset);
    println!("HONESTY: {}", report.honesty);
    println!(
        "  config: backend={} embedder={} docs={}",
        report.index_backend, report.embedder, report.build.docs_indexed
    );
    println!("  top-1     = {top1:.4}");
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
        "  build     {:.2}ms total ({} docs)",
        report.build.total_build_ms, report.build.docs_indexed
    );
    println!(
        "  memory    {} index bytes, {:.1} bytes/doc",
        report.memory.index_bytes, report.memory.bytes_per_doc
    );
    println!("  report → {}", report_path.display());
}
