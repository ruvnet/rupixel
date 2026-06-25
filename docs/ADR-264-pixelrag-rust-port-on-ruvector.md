---
adr: 264
title: "PixelRAG Rust port on ruvector substrate — Visual retrieval-augmented generation with pixel-native indexing"
status: Proposed
date: 2026-06-25
authors: [claude-flow]
related: [ADR-254, ADR-255, ADR-256, ADR-194, ADR-155, ADR-260, ADR-262]
supersedes: []
tags: [visual-rag, retrieval, embedding, vision-encoder, fastcan, pixel-native, benchmark, darwin, metaharness]
---

# ADR-264 — PixelRAG Rust port on ruvector substrate

> **Provenance note.** This decision proposes a Rust port of
> [StarTrail-org/PixelRAG](https://github.com/StarTrail-org/PixelRAG) (Apache-2.0,
> ~5.3k GitHub stars) layered on this repo's existing ruvector substrate. PixelRAG
> is a *visual retrieval-augmented generation* system that renders documents
> (web pages, PDFs) to screenshots via `pixelshot` (Playwright/CDP + PDF libraries)
> and retrieves over visual embeddings instead of parsing to text. The port reuses
> `ruvector-rabitq`, `ruvector-core` (HNSW), and `ruvector-rairs` (IVF-SQ) for
> indexing; integrates `ruvector-cnn` for vision encoding; and proposes
> `ruvector-turbovec` (ADR-254) as an aspirational M2+ optimization for FastScan.

## Status

**Proposed.** This ADR defines the reuse boundary, crate layout, milestone
breakdown (M0–M3), and the benchmark harness to validate the port. It does NOT
yet contain measured performance numbers — those will be gathered during M1+.

## Context

### The gap

PixelRAG solves a real RAG problem: **traditional text-based document parsing
loses visual structure** (tables, charts, layout cues). PixelRAG preserves it by
rendering documents to screenshots and retrieving over visual embeddings instead
of text. The upstream Python implementation (StarTrail-org/PixelRAG, GitHub) is
mature and well-benchmarked on Wikipedia (8.28M pages) and visual-QA datasets
(ViDoRe, document-VQA style).

However:
1. **Pure Python is not production-grade for inference-heavy workloads.** The
   render → embed → index → retrieve → rerank → generate pipeline is I/O and
   compute-bound; Rust with SIMD and async I/O can 10–100× throughput/cost.
2. **ruvector already has mature ANN substrate.** We have HNSW (`ruvector-core`),
   IVF-SQ (`ruvector-rairs`), 1-bit quantization (`ruvector-rabitq`), and
   vision CNN encoders (`ruvector-cnn`). Reuse is a force multiplier. The
   proposed multi-bit FastScan tier (`ruvector-turbovec`, ADR-254) will
   further optimize pixel-tile indexing once shipped.
3. **No Rust visual-RAG reference exists.** Building one on ruvector unlocks
   pixel-native retrieval as a ruvector feature (not a standalone silo),
   composable with other index types and agents.

### What already exists (not duplication)

1. **`ruvector-core` (HNSW)**: Graph-based ANN index, production-proven, O(log n)
   search. **M1 primary backend** for pixel-tile indexing.
2. **`ruvector-rairs` (IVF-SQ, ADR-193)**: Inverted-list + optional scalar
   quantization. **M1 fallback** if HNSW memory footprint exceeds budget on
   large datasets. Supports pre-filtered search (allowlist).
3. **`ruvector-rabitq`**: 1-bit binary quantization + randomized Hadamard
   rotation. Provides the `AnnIndex` trait contract, `RandomRotation::HadamardSigned`,
   and `VectorKernel`/`KernelCaps` abstractions. **Reuse for consistency.**
4. **`ruvector-turbovec` (ADR-254, proposed)**: Multi-bit (2–4-bit) FastScan
   index. **Aspirational M2+ optimization** once landing; not yet shipped.
   Handles scalar quantization without f32 rerank. If ADR-254 ships before
   M2, swap from ruvector-core → turbovec for memory efficiency.
5. **`ruvector-cnn`**: Vision CNN/embedding crates for local inference.
   Foundation for vision encoding; port wraps ONNX/Candle encoder for
   Qwen3-VL-Embedding or CLIP surrogate.
6. **Async Tokio + hyper/tonic stacks**: Already integrated in ruvector ecosystem
   (e.g., `mcp-brain-server`). **Reuse for pixel-rag-serve HTTP tier.**

So this ADR is **not reimplementing a vector DB**; it is composing the existing
and in-flight ruvector substrate into a cohesive visual-RAG pipeline.

## Decision

Introduce a **PixelRAG Rust port as a 5-crate module layered on ruvector**,
following the upstream `pixelshot` pipeline: **render** → **embed** → **index** → **serve**.
Each crate ≤500 lines and reuses existing ruvector primitives without new
plumbing. M1 uses `ruvector-core` (HNSW) or `ruvector-rairs` (IVF-SQ); M2+ can
adopt `ruvector-turbovec` FastScan if ADR-254 lands.

### Reuse boundary

| Component | Upstream (PixelRAG Python) | Rust port action | Crate |
|---|---|---|---|
| Document rendering (web pages, PDFs → screenshots) | `pixelshot` (Playwright + CDP + pdf2image) | Port to headless-chrome crate or lazy-load from upstream Python service (v1) | `pixelrag-render` (M2) or skip with Python sidecar |
| Visual encoder (Qwen3-VL-Embedding-2B LoRA fine-tuned) | HuggingFace transformers + LoRA | ONNX runtime inference via `ruvector-cnn` wrapper, or Candle/burn for ONNX, or Python sidecar (v1) | `pixelrag-encoder` (new) with ONNX/candle backend |
| Tiling + metadata | Screenshot tiles (size per `pixelshot` config) | Embed directly in Rust; integrate with render output | `pixelrag-core` (new) |
| Index (M1) | FAISS (normalized index, ~217G for 8.28M pages — likely IVF-based) | **M1**: Use `ruvector-core` (HNSW) or `ruvector-rairs` (IVF-SQ). **M2+**: Swap to `ruvector-turbovec` (ADR-254) for 2–4-bit quantization if shipped. | reuse `ruvector-core::HNSWIndex` or `ruvector-rairs::IVFIndex` |
| Retrieval + filtering | FAISS search() + numpy metadata | Implement search via `AnnIndex` trait; add filtered retrieval (allowlist from ruvector-rabitq) | `pixelrag-core` |
| Reranking (optional cross-encoder) | LLM judge / clip-rerank | Wrap in Python sidecar or lightweight Rust scorer | `pixelrag-rerank` (optional) |
| HTTP server | FastAPI + Pydantic | HTTP server on Tokio; Tonic gRPC option; OpenAPI schema | `pixelrag-serve` (new) |

### Crate layout

```
crates/
├── pixelrag-core/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── pipeline.rs        # Render → embed → index → search orchestrator
│   │   ├── tile.rs            # Document → tile(s) logic; caching layer
│   │   ├── embedding.rs       # Encoder wrapper (ONNX or sidecar-call)
│   │   ├── index.rs           # AnnIndex adaptor wrapping turbovec
│   │   └── search.rs          # Retrieval + filtering + reranking hooks
│   └── Cargo.toml
├── pixelrag-encoder/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── onnx.rs            # ONNX runtime loading + batching
│   │   ├── model.rs           # Qwen3-VL-Embedding model instantiation
│   │   └── cache.rs           # LRU embedding cache for tiles
│   └── Cargo.toml
├── pixelrag-render/ (optional M2)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── playwright.rs      # Headless Chrome via playwright-rs (if porting)
│   │   ├── pdf.rs            # PDF → bitmap via pdfium-render
│   │   └── cache.rs          # Disk cache for rendered images
│   └── Cargo.toml
├── pixelrag-serve/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── http.rs           # Hyper-based HTTP API (OpenAPI compat)
│   │   ├── handlers.rs       # /index, /search, /health endpoints
│   │   └── config.rs         # Server config (port, model path, etc.)
│   └── Cargo.toml
└── pixelrag-cli/
    ├── src/
    │   ├── main.rs           # CLI for ingestion, search, benchmark
    │   └── bench.rs          # Benchmark harness (see Validation)
    └── Cargo.toml
```

### Dependencies & trait contracts

- **`pixelrag-core`** depends on:
  - `ruvector-core` (HNSW, M1 primary)
  - `ruvector-rairs` (IVF-SQ, M1 fallback)
  - `ruvector-turbovec` (optional, M2+ if ADR-254 ships)
  - `ruvector-rabitq` (trait + rotation reuse)
  - `tokio` (async orchestration)
  - `serde` / `bincode` (persistence)
- **`pixelrag-encoder`** depends on:
  - `ort` (ONNX Runtime, ~2MB binary)
  - `ndarray` (tensor handling)
  - LRU cache (e.g., `lru`)
- **`pixelrag-render`** (v2+) depends on:
  - `playwright-rs` or `headless-chrome` crate (conditional feature flag)
  - `pdfium-render` (PDF → image)
- **`pixelrag-serve`** depends on:
  - `pixelrag-core` + `pixelrag-encoder`
  - `hyper` + `tokio`
  - `serde_json` (JSON request/response)
  - optional `tonic` (gRPC)
- **`pixelrag-cli`** depends on:
  - all of the above
  - `clap` (argument parsing)

All crates implement or wrap existing ruvector traits:
- `pixelrag-core::IndexAdapter` → impl `ruvector_rabitq::AnnIndex`
- `pixelrag-encoder::Embedder` → generic trait, not constrained to ruvector
- Both register with the `ruvector-rulake` dispatcher (ADR-155) if used as
  pluggable modules.

### Milestones

1. **M0 — Integration scaffold** (Week 1)
   - Stub crates with Cargo.toml, lib.rs skeleton, and trait imports.
   - Verify Cargo workspace compiles cleanly.
   - Set up test fixtures: sample Wikipedia pages, ViDoRe dataset subset.
   - **Deliverable**: crate stubs, workspace green, test data in `tests/fixtures/`.

2. **M1 — Core pipeline (encode + index + search, HNSW/IVF backend)** (Week 2–3)
   - `pixelrag-encoder`: ONNX runtime loader, batch embedding of tiles.
     - Load Qwen3-VL-Embedding-2B (or CLIP ONNX surrogate) from HuggingFace ONNX.
     - Implement batched embed(tiles: Vec<Image>) → Vec<Embedding>.
     - Add embedding cache (LRU, ~100MB).
   - `pixelrag-core`:
     - Tile logic: document → Vec<(image, bounds, metadata)>.
     - Index adaptor: wrap `ruvector-core::HNSWIndex` or `ruvector-rairs::IVFIndex` as search backend.
     - Search: AnnIndex::search(query_embedding, k=10) + filtered allowlist support (from rabitq).
   - `pixelrag-cli`: Subcommands `index <doc_path>`, `search <query_image>`.
   - **Validation**: E2E on ViDoRe subset (100 docs, 50 queries) — measure
     embedding latency, index build time, recall@10 vs. Python baseline.
   - **Deliverable**: `cargo test -p pixelrag-*`, CLI passing integration tests.

3. **M2 — Rendering + M2+ FastScan optimization (conditional on ADR-254)** (Week 4–5)
   - `pixelrag-render`: Port upstream render pipeline (headless-chrome crate or
     lazy-load from Python service). Cache rendered images to disk.
   - `pixelrag-core`: Integrate render() → embed() → index() in one pipeline.
   - If `ruvector-turbovec` ships: evaluate swap from HNSW → FastScan SQ for
     recall@10 at 2-bit vs 3-bit vs 4-bit, using TQ+ calibration.
   - **Validation**: Full ViDoRe (1000 docs, 200 queries) — latency p50/p99,
     memory per doc, end-to-end index build time.
   - **Deliverable**: CLI `index --from-url https://...`, persistence to
     `*.pixelrag` file (bincode).

4. **M3 — HTTP server + reranking (optional)** (Week 6)
   - `pixelrag-serve`: Hyper-based REST server; OpenAPI schema.
   - Endpoints: `POST /index` (ingest), `POST /search` (retrieve), `GET /health`.
   - Optional reranking hook: pluggable cross-encoder or LLM judge (e.g., Claude
     API call for rerank).
   - **Validation**: Load test (100 req/s, 10 concurrent), latency p99, memory
     stability over 1h.
   - **Deliverable**: `cargo build --release -p pixelrag-serve`, Docker example.

### Honest embedding model strategy

The **visual encoder is the porting bottleneck.** Upstream PixelRAG uses
Qwen3-VL-Embedding (a fine-tuned LoRA on CLIP-style ViT). Three strategies:

1. **v1 (Conservative, no new ONNX work).** Call out to a Python sidecar.
   - Rust code spawns a Python subprocess (or HTTP call to a separate
     `pixelrag-encoder-py` service).
   - Tiles are serialized to disk/network; encoder runs in Python; embeddings
     return to Rust.
   - **Pro**: No new ONNX runtime integration; reuse upstream model weights.
   - **Con**: IPC latency, extra complexity. Not production-grade.
   - **Fallback if ONNX causes issues.**

2. **v2 (Recommended, phased).** Use ONNX Runtime Rust binding (`ort` crate).
   - M1: Load Qwen3-VL ONNX weights (export from HuggingFace or use CLIP ONNX
     surrogate).
   - Batch embedding on Tokio thread pool.
   - **Pro**: Single-binary Rust deployment; 10× throughput vs. Python sidecar.
   - **Con**: ONNX Runtime binary is large (~300MB); model weights are sizeable
     (1–2GB for ViT-L).
   - **Plan**: Quantize model to int8 or 4-bit using `ort`'s quantization tools.

3. **v3 (Longer-term, if ONNX hits issues).** Migrate to Candle or burn.
   - HuggingFace's Candle is a Rust ML framework; burn is a learner framework.
   - Both can load ONNX or HuggingFace safetensors directly.
   - **Pro**: Pure Rust, lightweight, composable with ruvector-cnn kernels.
   - **Con**: Requires porting model-loading code; longer dev time.
   - **Timeline**: Post-M3, only if ONNX causes production issues.

**Proposed path**: Commit to **v2 (ONNX Runtime via `ort`)** for M1–M2. If
binary size or model loading becomes a blocker during M2, fall back to v1
(Python sidecar) for M3 and iterate on v3 (Candle) as a post-release task.

Document this in `pixelrag-encoder/README.md` with clear migration guidance.

## Validation (benchmark plan)

**Datasets:**
- **ViDoRe** (Visual Document Retrieval): 495 docs, 5,191 questions. Evaluates
  retrieval over visual structure (tables, charts, layout).
- **Document-VQA** (if time permits): 12k docs, VQA-style queries.
- **Wikipedia subset** (PixelRAG's primary benchmark): 8.28M pages; sample 10k
  for dev/test.

**Metrics:**
- **Recall@k**: NDCG@10 and MRR (measure reranking quality).
- **Embedding throughput**: tokens/sec (tiles/sec) at various batch sizes.
- **Index build time**: seconds per 1000 docs (includes render + embed + write).
- **Search latency**: p50, p95, p99 for a retrieval query (vec-sim search only,
  excluding rerank).
- **Memory**: RSS per indexed doc (metadata + embeddings + index); memory
  efficiency at 2-bit vs 4-bit quantization.
- **Recall-cost tradeoff**: NDCG@10 vs. memory per doc at each quantization
  tier.

**Exact commands to produce benchmark results:**

```bash
# M1 milestone: encode + index + search
cd crates/pixelrag-cli
cargo build --release

# Index ViDoRe dataset
time ./target/release/pixelrag-cli index \
  --dataset vidore \
  --output ./bench_output/vidore.pixelrag \
  --batch-size 32 \
  --quantization 4-bit

# Run search on queries (50 queries × 10 results)
time ./target/release/pixelrag-cli search \
  --index ./bench_output/vidore.pixelrag \
  --queries ./tests/fixtures/vidore_queries.json \
  --k 10 \
  --output ./bench_output/vidore_results.json

# Compute recall metrics (vs. ground truth)
cargo run --release -p pixelrag-cli -- benchmark \
  --predictions ./bench_output/vidore_results.json \
  --ground-truth ./tests/fixtures/vidore_gt.json \
  --metrics ndcg,mrr,recall@10

# Memory profiling
valgrind --tool=massif --massif-out-file=./bench_output/massif.out \
  ./target/release/pixelrag-cli index \
  --dataset vidore \
  --output ./bench_output/vidore_massif.pixelrag \
  --batch-size 32

# Parse massif output for peak RSS
ms_print ./bench_output/massif.out | grep "peak"
```

**Optimization harness (darwin / MetaHarness integration, §Metaharness):**

The benchmark harness itself becomes a *learnable system* via `@metaharness/darwin`
v0.7.0:
- Darwin evolves the **harness parameters**: quantization tier (2/3/4-bit),
  batch size, embedding cache size, SIMD kernel selection (if M2+), reranking
  strategy.
- Each candidate harness is deployed and scored on ViDoRe (NDCG@10 × index
  memory, Pareto frontier).
- Darwin runs offline (not in-loop of the Rust port); it produces a `harness
  genome` (JSON config) and a `policy` (decision tree for parameter selection).
- This config is **optional** — it does not ship inside the Rust library. The
  port is fully usable without it; darwin is a *separate optimization loop*.

**Placeholder results (to be measured M1+):**

| Milestone | Quantization | Batch | Embed (ms/tile) | Index (s/1k docs) | Search p99 (ms) | NDCG@10 | Memory/doc (KB) |
|-----------|--------------|-------|-----------------|-------------------|-----------------|---------|-----------------|
| M0        | — | — | — | — | — | — | — |
| M1        | 4-bit | 32 | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* |
| M2        | 3-bit | 64 | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* |
| M2        | 2-bit | 64 | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* |
| M3        | 4-bit + rerank | 32 | (+ LLM) | — | *to measure* | *to measure* | — |

(Baseline: PixelRAG Python on same ViDoRe subset, measured in same environment.)

## Consequences

### Positive
- **Closes the visual-RAG gap in ruvector.** Visual retrieval becomes a
  first-class feature (not a silo), composable with other index types.
- **Reuse payoff.** Builds on `ruvector-turbovec` (ADR-254), `ruvector-cnn`,
  and HNSW. No new ANN library needed.
- **Production-ready throughput.** Rust async + SIMD should 10–100× the Python
  baseline, enabling pixel-RAG at scale (8M pages → <500ms per query on
  commodity hardware).
- **Zero required MetaHarness dependency.** Port ships as standalone Rust
  crates; darwin is *optional augmentation* (ADR-256), not runtime dep.
- **Path to other visual tasks.** Once pixel-native indexing is proven, crate
  can be extended for visual Q&A, layout understanding, table extraction
  (downstream work).

### Negative
- **ONNX Runtime complexity.** Embedding model (Qwen3-VL-Embedding-2B) must be
  ported to ONNX or sidecar. Public ONNX weights are uncertain; LoRA adaptation
  adds M1 risk. Mitigated by v1 fallback (Python sidecar) or CLIP ONNX surrogate.
- **Rendering is hard.** Porting Playwright + CDP to Rust (or wrapping
  headless-chrome crate) is non-trivial. M2 risk. Mitigation: stub with upstream
  Python service (v1) or precomputed tile cache.
- **Benchmark setup.** ViDoRe / document-VQA eval fixtures are new to ruvector.
  Requires test data download + metric implementation. M0–M1 risk.
- **Index backend swap in M2.** Contingent on ADR-254 landing; if delayed, M2
  remains on HNSW/IVF-SQ. Not a blocker — the port ships in M1 with HNSW.
- **Model weight licensing.** Qwen3-VL terms unclear; CLIP surrogate is a
  workaround. Clarify before M3 production serve. Not a blocker.

### Neutral
- Adds 4–5 new crates. ruvector already has 100+; workspace remains modular
  (features flags for pixelrag on demand).
- No changes to existing ruvector APIs; pixelrag-* crates are purely additive.
- M1–M2 focused on measurement; no preemptive optimizations (SIMD, kernel
  fusion) until validated.

## MetaHarness / Darwin integration (ADR-256)

**Proposal**: Use `@metaharness/darwin` v0.7.0 to **evolve the porting +
benchmarking harness**, not the Rust source code. Governance per ADR-256
("Borrowing metaharness concepts into npx ruvector …").

**What darwin does:**
- Freezes the Rust port's *source code* and *algorithm* (unchanged).
- Evolves the *harness parameters*: batch size, embedding cache size, index
  choice (HNSW vs. IVF-SQ), reranking threshold, etc.
- Each candidate harness is deployed in a sandbox; scored on ViDoRe NDCG@10 ×
  memory.
- Produces a Pareto frontier of `(config, metrics)` pairs.

**Constraint (ADR-256 enforcement):** MetaHarness is **removable**. If darwin
is unavailable or disabled:
1. The port still builds, indexes, and searches using the **default M1 harness**
   (HNSW index, batch=32, cache=100MB, no rerank).
2. Darwin's output (optimal config JSON) is *read-only* — it does not control
   the Rust runtime via environment variables or APIs.
3. Packaging (Docker, binary release) does **not** ship darwin.

**Explicit coding rule**: In `pixelrag-core/lib.rs`:
```rust
// darwin-generated configs are OPTIONAL (path can be None).
// If Config::from_darwin_json(path) fails, use Config::default().
pub struct Config {
    pub index_backend: IndexBackend,  // default: HNSW
    pub batch_size: usize,            // default: 32
    pub embedding_cache_mb: usize,    // default: 100
    pub darwin_config_path: Option<PathBuf>, // optional
}
impl Config {
    pub fn from_darwin_json(path: &Path) -> Result<Self> { /* ... */ }
    pub fn default() -> Self { /* hard-coded defaults */ }
}
```

This ensures the binary is usable and correct even if darwin is never run.

## Alternatives considered — PhotonLayer optical compression (out of scope)

**Question raised:** can [PhotonLayer](https://github.com/ruvnet/PhotonLayer)
(ADR-260, the learned-optical-frontend simulator) improve the *size* and
*quality* of this Rust port? **Verdict: no for the core port — category
mismatch — with one narrow, speculative privacy direction tracked as future
work.** This entry records the reasoning so it is not re-proposed.

**What PhotonLayer actually is** (per `docs/research/photonlayer/ASSESSMENT.md`,
measured on MNIST via `photonlayer-bench`): a task-trained *single optical layer
+ tiny digital decoder* that **compresses an image-sensing front end** — MNIST
1024 → 64 sensor pixels (16× fewer pixels, 16× fewer digital MACs) at **−2.35 pp
accuracy vs a matched full-image baseline**. Its own positioning is explicit:
*"competitive single-layer optical compression … **not a new accuracy SOTA**"*,
and it carries a documented optimizer ceiling (single-mask hill-climb converges
~2 pp short; analytic gradient descent is the roadmap, not shipped).

**Why it does not fit PixelRAG's size/quality goals:**

1. **Quality — no.** PhotonLayer is, by construction, a *lossy* compressor: it
   trades a quantified accuracy *loss* for sensor/compute savings. It cannot
   raise retrieval recall, and its assessment forbids "improves quality / beats
   SOTA / near-lossless" framing.
2. **Size — wrong tool, category mismatch.** It is validated on MNIST-class
   *image* compression, not on (a) embedding vectors or (b) high-resolution
   document screenshots — where the text/table/layout detail is exactly what
   visual retrieval depends on. PixelRAG's size costs (the ~217G embedding index;
   the 2B encoder) are already addressed here by the *right* tools: RaBitQ
   rotation + scalar quantization (`ruvector-rabitq`), OPQ/PQ, dimensionality
   reduction, and encoder int8/4-bit. A diffractive optical operator used as a
   generic vector projector is an exotic, off-label dependency that does a worse,
   harder-to-audit job than these.
3. **Domain mismatch.** PhotonLayer operates on 2D optical fields via Fraunhofer
   FFT propagation, not arbitrary 1D embedding vectors; bolting it onto the
   encoder front end would discard the rich ViT features the whole system exists
   to produce.

**The one genuinely PhotonLayer-shaped direction (speculative, not size/quality):**
a *privacy-preserving ingest* mode (cf. ADR-262) that indexes optical
*measurements* of screenshots so no readable image is stored. This matches
PhotonLayer's real wedge ("privacy by physics") but **costs retrieval recall**
and, per the assessment, only supports the claim *"no readable image is stored"*
until reconstruction leakage is quantified. Tracked as **optional future
research**, explicitly out of scope for M0–M3.

## Links

- **Upstream**: [StarTrail-org/PixelRAG](https://github.com/StarTrail-org/PixelRAG)
  (Apache-2.0, ~5.3k stars). README + `pixelshot` render + architecture at `docs/`.
- **PixelRAG datasets & benchmarks**:
  - ViDoRe (visual document retrieval): https://huggingface.co/datasets/jinaai/ViDoRe
  - Document-VQA: https://huggingface.co/datasets/naver-clova-ocr/docvqa
  - Wikipedia: 8.28M pages (upstream pre-built index ~217G; cf. PixelRAG README).
- **Embedding model**:
  - Qwen3-VL-Embedding-2B: https://huggingface.co/Qwen/Qwen3-VL-Embedding
  - CLIP ONNX surrogate (fallback): https://huggingface.co/openai/clip-vit-large-patch14
  - ONNX export guide: https://huggingface.co/docs/optimum/exporters/onnx
- **Related ADRs**:
  - ADR-254 (ruvector-turbovec, proposed FastScan quantization): M2+ optimization path.
  - ADR-255 (OIA model integration): embedding model sourcing reference.
  - ADR-256 (MetaHarness SDK evaluation, proposed): removable-augmentation governance.
  - ADR-194 (ruvector ONNX embedder API): ONNX ingestion reference.
  - ADR-260 / ADR-262 (PhotonLayer optical frontend / privacy verification):
    considered and ruled out for core size/quality — see "Alternatives considered" above.
  - ADR-155 (rulake datalake layer): dispatcher/composition contract.
  - ADR-193 (ruvector-rairs IVF-SQ): M1 fallback index backend.
- **Rust ecosystem**:
  - ONNX Runtime Rust binding (`ort`): https://github.com/pykeio/ort
  - Candle (alternative encoder): https://github.com/huggingface/candle
  - Tokio async runtime: https://tokio.rs
  - Hyper HTTP library: https://hyper.rs
  - Headless Chrome Rust crate: https://crates.io/crates/headless_chrome
  - pdfium-render (PDF → bitmap): https://github.com/ajryan/pdfium-render
- **Benchmarking & optimization**:
  - MetaHarness (darwin) npm: https://www.npmjs.com/package/@metaharness/darwin
  - ADR-256 (removability constraint): removable-augmentation governance for darwin integration.
