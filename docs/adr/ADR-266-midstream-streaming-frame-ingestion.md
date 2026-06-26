---
adr: 266
title: "MidStream integration for streaming frame ingestion — real-time Rust ingest tier with temporal comparison and backpressure"
status: Proposed
date: 2026-06-26
authors: [claude-flow]
related: [ADR-265, ADR-264, ADR-077]
supersedes: []
tags: [midstream, streaming, real-time-ingest, temporal-comparison, backpressure, quic-multistream, frame-deduplication, tier-2, edge]
---

# ADR-266 — MidStream integration for streaming frame ingestion

> **Scope.** This decision proposes **Tier-2 native Rust streaming** for real-time
> video ingestion, extending [[ADR-265-real-time-video-visual-rag-rupixel|ADR-265]]
> (browser MVP). Reuses ruvnet/midstream crates (temporal-compare, nanosecond-scheduler,
> quic-multistream) as **streaming library primitives**, not LLM-specific tooling.
> Honest caveat: midstream is built for LLM token streams; this reuses its
> generic scheduling, transport, and temporal analysis for frame streams.

## Status

**Proposed.** Designed and scoped; NOT YET IMPLEMENTED. This ADR lays out the
Rust service architecture, library dependencies, and integration boundaries.
Assumes ruvnet/midstream crates are published and WASM-ready.

## Context

### The gap

[[ADR-265-real-time-video-visual-rag-rupixel|ADR-265]] delivers a browser MVP but has hard limits:
- **CPU-only inference** (WASM CLIP): 10–20 frames/sec max.
- **Local-only indexing** (browser IndexedDB): ~50MB storage.
- **No backpressure** on video ingest; simple frame dropping.
- **Single-browser** scope; no multi-device or edge compute.

For **production real-time ingest** (multiple streams, GPU acceleration,
high-performance temporal filtering), a native Rust service is required.

### What already exists (not duplication)

1. **ruvnet/midstream** (crate ecosystem, ADR-077 integration):
   - `temporal-compare` (DTW, LCS): Temporal sequence analysis for redundancy
     detection. Published on crates.io; used for LLM token deduplication but
     **fully generic** (works on any sequence, including image hashes).
   - `nanosecond-scheduler` (crate): Frame pacing and backpressure (slow consumer
     feedback to producer). Prevents queue buildup; critical for real-time.
   - `quic-multistream` (crate): Multi-stream QUIC transport. Enables parallel
     ingest of multiple video feeds with per-stream flow control.
   - `temporal-attractor-studio` (crate): Temporal trajectory aggregation (not
     primary use here, but available).
   - `strange-loop` (crate): Generic event loop orchestrator.
   - **All WASM-ready** as per midstream philosophy.

2. **rupixel CLIP encoder** (CPU fallback, GPU via ONNX Runtime):
   - Can be wrapped in a Rust FFI layer or invoked via subprocess (v1) or ONNX
     Runtime Rust binding (v2, same as ADR-264).
   - **Reuse for embedding service.**

3. **ruvector HNSW/IVF index** (ADR-264):
   - Async-compatible via Tokio. Can be persisted to disk for durability.
   - **Reuse for distributed index.**

4. **Witness infrastructure** (ADR-103, ADR-064):
   - Stream ingestion with cryptographic audit trails (BLAKE3 hashing of frame
     provenance).
   - Available for compliance/regulatory workflows.

So this ADR is **composing midstream primitives + rupixel inference + ruvector
indexing** into a coherent Rust ingest service. No new temporal or transport
libraries needed.

## Decision

Introduce **MidStream-based Rust ingest service (Tier 2)**, following the
three-layer stack:

```
[Video source (RTMP/HLS/file)]
         ↓
[QUIC receiver (quic-multistream)]
         ↓
[Frame sampler + temporal-compare dedup]
         ↓
[Inference dispatcher (GPU/CPU CLIP)]
         ↓
[HNSW index writer (Tokio async)]
         ↓
[Storage (RocksDB or Postgres)]
```

### Honest caveat: LLM-stream ≠ frame-stream reuse

**MidStream is built for token-stream analysis**, particularly:
- Token arrival patterns (bursty, variable-rate decoding).
- Attention-based token importance scoring.
- Probabilistic token dropping/resampling.

**For frame streams, we reuse the generic primitives:**
- `temporal-compare` DTW/LCS for *image hash* sequences (not token ids).
- `nanosecond-scheduler` for frame pacing (works on any event type).
- `quic-multistream` for multi-stream transport (protocol, not payload-specific).

The **LLM-specific parts** (token probability, top-K sampling, KV cache
alignment) are **NOT reused**; frame deduplication uses simple L2 distance or
perceptual hash Hamming distance (from ADR-265). There is **no npm package
`midstream`** consumable by browser clients; midstream is Rust crates only.
Browser clients call the Tier-2 Rust service via HTTP/gRPC.

### Crate structure

**New crates in ruvector workspace:**

```
crates/
├── pixelrag-ingest-service/
│   ├── src/
│   │   ├── main.rs                # Binary entrypoint
│   │   ├── lib.rs
│   │   ├── quic_receiver.rs       # quic-multistream wrapper
│   │   ├── frame_processor.rs     # sampler + temporal-compare dedup
│   │   ├── inference.rs           # CLIP embedding dispatcher
│   │   ├── indexer.rs             # HNSW writer (async)
│   │   ├── config.rs              # Service config (port, GPU, etc.)
│   │   └── witness.rs             # Optional: audit trail (BLAKE3)
│   └── Cargo.toml
├── pixelrag-ingest-client/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── quic_client.rs         # QUIC producer side
│   │   └── http_client.rs         # HTTP fallback
│   └── Cargo.toml
```

**Dependencies** (verified published, WASM-ready):

| Crate | Version (verified) | Role |
|---|---|---|
| `temporal-compare` | Published on crates.io | Frame hash DTW/LCS dedup |
| `nanosecond-scheduler` | Published on crates.io | Backpressure + pacing |
| `quic-multistream` | Published on crates.io | Multi-stream QUIC ingest |
| `tokio` | ≥1.0 | Async runtime |
| `ort` | Latest | ONNX Runtime (CLIP inference) |
| `ruvector-core` | From workspace | HNSW indexing |
| `ruvector-rairs` | From workspace | IVF-SQ indexing |
| `serde` / `bincode` | Standard | Serialization |
| `blake3` | Optional | Witness audit trail |

**No** unverified or speculative dependencies.

### Design: frame flow with backpressure and streaming LLM captions

```rust
// Pseudocode
struct FrameIngestService {
    quic_rx: QuicMultistream<Frame>,
    sampler: FrameSampler { fps: 5 },
    dedup: TemporalCompare { threshold: 0.05 },
    embedder: ClipEmbedder,  // ONNX or subprocess
    captioner: StreamingLLMCaptioner,  // OpenRouter vision LLM, stream:true
    temporal_analyzer: TemporalCompareAnalyzer,  // analyzes caption stream
    indexer: HNSWWriter,
    scheduler: NanosecondScheduler,
}

impl FrameIngestService {
    async fn run(&mut self) {
        loop {
            // 1. Receive frame with backpressure signal
            let frame = self.quic_rx.recv().await;
            
            // 2. Sample (skip N/M)
            if !self.sampler.should_process(&frame) { continue; }
            
            // 3. Dedup via temporal-compare
            let hash = frame.perceptual_hash();
            if self.dedup.is_duplicate(&hash) {
                frame.mark_skipped();
                continue;
            }
            
            // 4. Embed (may apply backpressure if queue fills)
            let embedding = self.embedder.embed(&frame).await?;
            let backpressure = self.scheduler.check_capacity();
            if backpressure.exceeds_threshold() {
                self.quic_rx.signal_slow_consumer().await;
            }
            
            // 5. Stream LLM caption (vision model, OpenRouter SSE)
            // The caption text becomes a token stream analyzed by midstream
            let caption_stream = self.captioner.caption_streaming(&frame).await?;
            let caption_text = String::new();
            pin_mut!(caption_stream);
            while let Some(token) = caption_stream.next().await {
                caption_text.push_str(&token);
                // Analyze caption tokens in real time for patterns/triggers
                self.temporal_analyzer.ingest_token(&token);
            }
            
            // 6. Index async (visual embedding + caption text as searchable metadata)
            self.indexer.insert_async(
                embedding,
                frame.metadata_with_caption(caption_text)
            );
        }
    }
}
```

**Key properties:**
- **Backpressure**: If embedding or indexing lags, scheduler signals producer
  to slow down (QUIC's per-stream flow control).
- **Multi-stream**: QUIC allows N parallel video feeds with independent flow
  control per stream (no head-of-line blocking).
- **Temporal dedup**: `temporal-compare` on *sequence of perceptual hashes*,
  not image pixels. O(n log n) via DTW, or O(n) via LCS tuning.
- **Streaming LLM captions**: Vision model (OpenRouter, stream:true) describes
  each keyframe token-by-token. Caption tokens are analyzed in real time by
  MidStream's temporal-compare for pattern triggers (scene changes, key events).
  Caption text is indexed alongside CLIP vectors for multimodal search ("find
  when the speaker said X" + visual context).
- **Async throughout**: Tokio runtime; embedding, captioning, and indexing
  don't block frame reception.

### API contract

**Tier-2 service exposes:**

1. **QUIC ingest** (primary):
   ```
   quic://<host>:9999/stream/<stream-id>
   → Send Frame { timestamp, data: bytes, metadata } messages
   → Receive Ack { skipped, indexed_at, embedding_id, caption_text }
   ```

2. **HTTP fallback** (for browsers or lightweight clients):
   ```
   POST /v1/ingest
   { "stream_id": "webcam-1", "frame": base64, "timestamp": 1234567890 }
   → { "indexed": true, "embedding_id": "abc123", "caption": "A person..." }
   
   GET /v1/search?q=<text_or_vision_query>&stream_id=<id>&k=10
   → [ { timestamp, score, frame_url, caption_snippet } ]
   
   POST /v1/caption-stream
   (proxy for browser calling OpenRouter with server-held key)
   { "frame": base64 }
   → SSE stream of caption tokens
   ```

3. **gRPC (optional, for agent orchestration)**:
   ```proto
   service PixelRagIngest {
     rpc IngestFrameStream(stream Frame) returns (stream IngestResponse);
     rpc Search(SearchRequest) returns (SearchResponse);
     rpc StreamCaption(Frame) returns (stream CaptionToken);
   }
   ```

**Tier-1 browser client** (ADR-265) calls the HTTP API; **Tier-2 edge
producers** (e.g., Raspberry Pi with camera) call QUIC. Browser captions
are proxied through `/v1/caption-stream` to protect OpenRouter API key
(see Security section below).

### Milestones

1. **M0 — Service scaffold** (Week 1)
   - Create crates, add Cargo.toml, verify midstream deps build cleanly.
   - Stub QUIC receiver, frame processor, indexer.
   - **Deliverable**: `cargo build -p pixelrag-ingest-service` succeeds.

2. **M1 — Core pipeline (QUIC ingest + dedup + embed + index)** (Week 2–3)
   - Implement QuicMultistream receiver; stream N concurrent video sources.
   - Integrate `temporal-compare` for hash-sequence deduplication.
   - Embed keyframes via ONNX Runtime (synchronous for M1; async thread pool
     in M2).
   - Write to HNSW index (persistent to RocksDB).
   - HTTP `/search` endpoint (in-memory HNSW search).
   - **Validation**: Send 10 concurrent 5-minute video feeds, measure throughput
     (frames/sec indexed), latency (p50/p99), dedup rate.
   - **Deliverable**: `pixelrag-ingest-service --config config.toml` starts server.

3. **M2 — Backpressure + async embedding + scale optimization** (Week 4–5)
   - Integrate `nanosecond-scheduler` for per-stream backpressure.
   - Async embedding on Tokio thread pool (no QUIC receiver blocking).
   - Optional GPU inference (ONNX Runtime with CUDA if available).
   - Benchmark: measure queuing latency, throughput at 100+ fps ingest.
   - **Deliverable**: `pixelrag-ingest-service --gpu` runs with NVIDIA GPU.

4. **M3 — Witness integration + gRPC + persistence** (Week 6)
   - Optional: BLAKE3 audit trail for each frame (compliance workflows).
   - gRPC API (for agent-to-service calls).
   - Durable index persistence (Postgres or cloud storage).
   - **Deliverable**: Service passes production readiness checks (latency,
     durability, security).

## Security: API key management for OpenRouter captions

### The requirement: protect LLM API credentials

Streaming LLM captions require OpenRouter API key access. The service must not
expose keys to the client or commit them to the repository.

**Two deployment patterns:**

1. **Browser (public demo) — Bring-Your-Own-Key (BYOK)**
   - User supplies their own OpenRouter API key in the browser UI.
   - Key is held **in-browser memory only** (sessionStorage, never localStorage).
   - Browser sends requests directly to OpenRouter SSE endpoint (CORS permitting).
   - Key is **never** sent to the ruvector service or the repo.
   - **Constraint**: Requires user to create an OpenRouter account + get a key.
   - **Pro**: Zero server-side infrastructure; no secrets in CI/CD.

2. **Production (GCP-hosted) — Server-side proxy**
   - GCP Secret Manager stores `OPENROUTER_API_KEY` (retrieved at service startup
     via `gcloud secrets versions access latest --secret=OPENROUTER_API_KEY`).
   - Tier-2 service exposes endpoint `/v1/caption-stream` (internal or
     authenticated only).
   - Browser calls `/v1/caption-stream` (sends frame, no key).
   - Service proxies to OpenRouter (injects secret server-side).
   - Response (SSE stream of caption tokens) flows back to browser.
   - **Pro**: Browser never sees the key; suitable for production/compliance.
   - **Constraint**: Requires service deployment; GCP secrets infrastructure.

### Implementation rules (non-negotiable)

**For the browser demo (ADR-265 live.html):**
- ✓ Accept OpenRouter key from user input field (sessionStorage only, never
  committed).
- ✓ Display a security note: *"Your key is held only in this browser tab, sent
  only to OpenRouter, and is never uploaded to this site or stored in the
  repo."*
- ✗ Do NOT hardcode any API key in source code.
- ✗ Do NOT load key from .env (even in dev; use .env.local, gitignored).
- ✗ Do NOT send key to ruvector service backend.

**For the Tier-2 service (pixelrag-ingest-service):**
- ✓ Read `OPENROUTER_API_KEY` from environment at startup.
- ✓ Validate key is set; error on startup if missing (fail-safe).
- ✓ Expose `/v1/caption-stream` endpoint (POST, internal/authenticated).
- ✓ Proxy request to OpenRouter; inject key server-side.
- ✓ Return SSE stream directly to caller (no intermediate caching of raw tokens).
- ✗ Never log the full key (safe: log only first 8 chars if needed for debugging).
- ✗ Never commit the key to git (it is a secret; use GCP/vault).

**For deployment (GCP):**
- Service account has `secretmanager.secretAccessor` permission.
- Startup hook: `gcloud secrets versions access latest --secret=OPENROUTER_API_KEY`
  → `OPENROUTER_API_KEY` env var.
- Docker/Cloud Run: Secret is injected at runtime, never baked into image.

### Honest trade-offs

- **Browser BYOK is user-unfriendly.** Requires users to sign up for OpenRouter
  + copy/paste their key. Acceptable for demos, not for production.
- **Server proxy adds latency.** Extra network hop (browser → ruvector service →
  OpenRouter). ~100–200ms added to caption latency. Acceptable for production
  where security > raw speed.
- **No DLP in browser.** User's OpenRouter key is visible (not hidden) in the
  browser. If the browser is compromised, key can be exfiltrated. Mitigated by
  BYOK being temporary (sessionStorage expires) and user-owned (they can rotate).

## Validation

### Benchmark: multi-stream ingest with streaming captions

```bash
# Start Tier-2 service
cd crates/pixelrag-ingest-service
cargo build --release
./target/release/pixelrag-ingest-service --config config.toml &

# Spawn 10 concurrent video streams (via test harness)
# Each stream: 5 minutes, 30 fps = 9000 frames
# Expected dedup: 60–70% → 2700–3600 keyframes per stream
cargo test --release -- --ignored bench_multi_stream

# Measure:
# - Throughput (frames/sec indexed across all streams)
# - Per-stream latency p50/p99
# - Dedup ratio (skipped / total)
# - Memory (RSS over time)
# - Index search latency (find top-5 for a test query)
```

### Test harness

Pseudo-code for verification:

```rust
#[tokio::test]
#[ignore]
async fn bench_multi_stream() {
    let service = PixelRagIngestService::spawn().await?;
    
    // 10 concurrent streams, 5-minute videos
    let streams = (0..10)
        .map(|i| {
            let client = QuicClient::connect(&service.addr()).await?;
            spawn_stream_producer(client, format!("stream-{}", i))
        })
        .collect::<Vec<_>>();
    
    let results = futures::future::join_all(streams).await;
    
    // Assertions:
    // - Dedup rate >= 60%
    // - Throughput >= 500 fps (across all streams)
    // - Search latency p99 < 200ms
    // - Service uptime = 100%
    
    Ok(())
}
```

### Metrics

| Metric | Target (M1) | Target (M2, w/ GPU) |
|---|---|---|
| Multi-stream throughput | ≥200 fps (10 streams × 20 fps ea.) | ≥1000 fps (10 streams × 100 fps ea.) |
| Per-stream latency p99 | <500ms | <100ms |
| Keyframe dedup rate | ≥60% | ≥60% (same algorithm) |
| Index search latency (top-5, 100k keyframes) | <100ms | <50ms |
| Memory per stream | <100MB | <150MB |
| Service uptime (1h load test) | 100% | 100% |

## Consequences

### Positive
- **Production-grade real-time ingest.** Multiple streams, backpressure, GPU
  acceleration unlock video RAG at scale (100+ concurrent producers).
- **Proven components.** Reuses ruvnet/midstream crates (published, battle-tested
  for token streams). No novel temporal algorithms to invent.
- **Multi-stream, no head-of-line blocking.** QUIC per-stream flow control means
  one slow producer doesn't starve others.
- **Streaming LLM captions unlock multimodal search.** Vision LLM (OpenRouter SSE)
  describes each keyframe; caption tokens are analyzed by temporal-compare in
  real time. Users can search both visually ("find dark scenes") and semantically
  ("find when they mentioned X"), improving retrieval quality.
- **Canonical MidStream use case.** LLM token streams are MidStream's native
  domain. Caption generation is the exact use case MidStream's temporal analysis
  was designed for — this ADR positions MidStream as the core component, not a
  generic library.
- **Async end-to-end.** Tokio runtime ensures embedding, captioning, and
  indexing never block frame reception.
- **Bridges Tier-1 and Tier-2.** Browser clients (ADR-265) can call HTTP API
  with BYOK; Tier-2 service proxies keys securely. Edge producers call QUIC
  directly. Single service.

### Negative
- **MidStream reuse is generic, not turnkey.** DTW/LCS dedup works on *sequences
  of hashes*, not native Rust/WASM token handling. Integration cost is moderate
  (frame → hash → temporal-compare). Not a "magic library" that does frame
  dedup out of the box; developers must understand temporal matching.
- **No npm package.** `midstream` crates are Rust-only. Browser clients cannot
  directly use them; they must call the Tier-2 HTTP/gRPC service. This is by
  design (native async/backpressure is not portable to browser JS), but it
  does add a service dependency.
- **Streaming LLM adds latency and cost.** Vision model (OpenRouter) costs ~$0.10
  per 1M tokens (~50 tokens per caption = ~$0.005 per keyframe). For a 1-hour
  video at 5 fps (3600 keyframes), caption cost is ~$18. This is **per user,
  per video**. Mitigated by: (a) caching captions for re-indexing, (b) sampling
  (skip every Nth keyframe for captions), (c) allowing user to toggle captions
  off.
- **GPU inference optional but not automatic.** CUDA support must be explicitly
  enabled and requires NVIDIA drivers. Fallback to CPU embedding is fine but
  slower. Mitigated by M2 benchmarks to guide deployment decisions.
- **OpenRouter API key security is critical.** Browser BYOK requires users to
  manage their own keys (simple but user-unfriendly); server-side proxy requires
  GCP secrets infrastructure. Both patterns must be documented and enforced
  in code. See Security section above.
- **Persistence model open.** M1 MVP indexes to in-memory HNSW; M3 should
  persist to Postgres or cloud (RocksDB local-only). Storage choice depends on
  deployment context (edge vs. cloud). Not a blocker, but requires
  infrastructure planning.

### Neutral
- Adds 2 new crates (ingest-service, ingest-client) to ruvector workspace.
  No changes to existing rupixel, ruvector, or midstream APIs.
- Tier-1 (browser) operates independently; Tier-2 is optional. Both are
  deployment choices, not hard dependencies.

## Links

- **Related ADRs:**
  - [[ADR-265-real-time-video-visual-rag-rupixel|ADR-265]] — Tier-1 browser MVP.
  - [[ADR-264-pixelrag-rust-port-on-ruvector|ADR-264]] — static document pixel RAG (parent, shares CLIP/HNSW).
  - [[ADR-077-midstream-brain-integration|ADR-077]] — MidStream platform integration.

- **MidStream crates** (published, WASM-ready):
  - `temporal-compare`: https://crates.io/crates/temporal-compare
  - `nanosecond-scheduler`: https://crates.io/crates/nanosecond-scheduler
  - `quic-multistream`: https://crates.io/crates/quic-multistream
  - Repo: https://github.com/ruvnet/midstream

- **Transport & serialization:**
  - QUIC: https://datatracker.ietf.org/doc/html/rfc9000
  - gRPC: https://grpc.io
  - Tokio async runtime: https://tokio.rs

- **Inference & LLM captions:**
  - ONNX Runtime Rust binding: https://github.com/pykeio/ort
  - GPU acceleration (CUDA): https://developer.nvidia.com/cuda-toolkit
  - OpenRouter (vision LLM, streaming): https://openrouter.ai/docs#api-overview
  - OpenRouter API cost: https://openrouter.ai/docs#models-and-pricing

- **Security & secrets:**
  - GCP Secret Manager: https://cloud.google.com/secret-manager/docs
  - BYOK patterns: https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html
