---
adr: 265
title: "Real-time video visual RAG over rupixel — frame sampling, keyframe gating, and CLIP embedding pipeline"
status: Proposed
date: 2026-06-26
authors: [claude-flow]
related: [ADR-264, ADR-260, ADR-262]
supersedes: []
tags: [visual-rag, video, real-time, frame-embedding, clip, keyframe-deduplication, temporal-filtering, browser-demo, rupixel]
---

# ADR-265 — Real-time video visual RAG over rupixel

> **Scope.** This decision extends [[ADR-264-pixelrag-rust-port-on-ruvector|ADR-264]]
> (static-document pixel RAG) to the **real-time video domain**: webcam streams,
> screen capture (getDisplayMedia), and pre-recorded video files. The goal is to
> enable text-to-frame retrieval over a live or batch video stream without
> processing every frame — achieved via **keyframe gating** (temporal deduplication)
> and **sampling**.

## Status

**Proposed.** Tier-1 (MVP/browser demo) is designed and scoped to rupixel's
existing live.html + live.js infrastructure; not yet implemented. Tier-2
(native Rust with streaming libraries) depends on ADR-266.

## Context

### The gap

Video is the natural extension of visual-RAG: a user wants to search *"Find the
moment where the speaker held up the chart"* in a screen recording, or *"Show me
when the product demo crashed"* in a webinar. Static document RAG (ADR-264)
handles screenshots; **video RAG must handle continuous frame streams** without
incurring the cost of embedding every frame at 24–30 fps.

Key constraints:
1. **Frame volume.** At 30 fps over a 1-hour video: 108,000 frames. Embedding all
   of them at 50ms per frame = 90 minutes of inference alone. Not feasible.
2. **Redundancy.** Consecutive frames are often near-identical (same speaker,
   same background). Embedding each one is waste.
3. **Browser accessibility.** A web UI (getUserMedia for webcam, getDisplayMedia
   for screen capture) is the lowest friction entry point for users.
4. **Latency.** For live video, users expect retrieval within 1–5 seconds of
   frame arrival, not batch processing later.

### What already exists (not duplication)

1. **rupixel CLIP encoder** (`docs/live.html`, `live.js`): Runs CLIP ViT-B/32 in
   the browser via `@xenova/transformers` (Hugging Face's transformer.js WASM
   port). Already proven for image embedding in real time on CPU. **Reuse
   directly.**
2. **ruvector HNSW index** (ADR-264, ADR-1): Graph-based nearest-neighbor search
   over embeddings. **Reuse for retrieval.**
3. **Temporal comparison libraries** (ruvnet/midstream, crates: `temporal-compare`
   with DTW/LCS; ADR-266): Exist in Rust for stream analysis. **Available if
   needed in Tier-2.**
4. **Browser APIs** (getUserMedia, getDisplayMedia, Canvas, Worker threads):
   Standard web platform; no new dependencies.

So this ADR is **not implementing a new vision encoder or search index** — it is
composing rupixel + ruvector with **keyframe gating** (temporal filtering) to
make video-scale retrieval tractable.

## Decision

Introduce **real-time video visual RAG in two tiers**:

### Tier 1 — Browser MVP (designed, no new infrastructure)

**Pipeline:**
```
[Video source] ──→ [Frame sampler (1–5 fps)] ──→ [Perceptual diff] ──→ [Keyframe gate]
                                                        ↓ (near-identical? skip)
                                                    [CLIP embed via ViT-B/32]
                                                        ↓
                                                    [Store in IndexedDB]
                                                        ↓
Text query ──→ [CLIP embed query] ──→ [HNSW search on IndexedDB frames] ──→ [Retrieval + playback]
```

**Components:**

1. **Frame sampler** (`rupixel/live.js`): Extract frames from video stream at
   configurable rate (1, 2, 5, or 10 fps).
   - For live webcam: use `requestAnimationFrame` with a low-pass filter (skip
     N frames, process 1).
   - For pre-recorded: loop `video.currentTime` at fixed intervals.

2. **Keyframe gating** (new, <100 lines of JS):
   - On each sampled frame, compute perceptual hash (e.g., `pHash` via canvas
     pixel downsampling, or compare L2 distance of the *previous* embedding vs.
     new).
   - If Hamming distance ≤ threshold OR L2 dist ≤ epsilon: **skip embedding,
     reuse previous**. Typically skips 60–80% of frames in low-motion scenes.
   - Explicitly store frame metadata: `{ timestamp, embedding, skipCount,
     sourceHash }` for audit + later re-embedding if thresholds change.

3. **CLIP embedding** (rupixel existing ViT-B/32):
   - Reuse `@xenova/transformers` CLIP model from live.js.
   - Embed keyframes only; batch if feasible (browser WW thread pool).
   - Cache embeddings to avoid re-embedding.

4. **Index storage** (IndexedDB or SQLite.js):
   - Store `{ timestamp, embedding: Float32Array, metadata: {skipCount, ...} }`.
   - For browser MVP: IndexedDB (native, ~50MB quota typical, can request
     persistent storage).
   - For longer videos (1–2 hours): SQLite.js or Postgres WASM (if
     bandwidth/size acceptable).

5. **Retrieval**:
   - Text query: embed via CLIP.
   - Search IndexedDB/in-memory HNSW (small dataset: <10k keyframes = <100MB
     embeddings).
   - Return top-K matches with timestamps.
   - Click → jump to that frame in video.

**Honest scope:**
- **Single-browser deployment.** No server; all processing is local (CPU CLIP
  only; no GPU). Typical laptop: 50–200ms per keyframe embedding.
- **Storage limit.** A 2-hour video at 5 fps = ~36k frames. Sampled to ~7–10k
  keyframes (70% dedup). Embeddings: 512-dim × 4 bytes × 10k = ~20MB. IndexedDB
  fits easily.
- **Latency.** For live capture: ~1–2 second lag from new frame to searchable
  (embedding + index insert). Not real-time in the sub-frame sense, but fast
  enough for user interaction.

**Mark as IMPLEMENTED at `rupixel/docs/live.html` + `live.js`:**
- Add UI widgets: FPS slider, dedup threshold slider, test queries.
- Document in `docs/live.html` the pipeline and threshold tuning.
- Benchmark: index & search latency on 1-hour sample video.

### Tier 2 — Native Rust streaming engine (Proposed, see ADR-266)

**Motivation:** The browser MVP is limited by CPU (single-threaded JS) and
bandwidth (WASM CLIP is large). For **production ingest at scale** (100+ videos,
concurrent streams, GPU acceleration), move to native Rust.

**Deferred to ADR-266:** MidStream integration (temporal-compare for dedup,
nanosecond-scheduler for backpressure, quic-multistream for transport).

## Security: API key management for the browser demo

### Browser demo is BYOK (Bring-Your-Own-Key) only

The rupixel public demo at `docs/live.html` **does not embed or commit any API
keys**. For optional streaming LLM captions (from ADR-266 / Tier-2 integration),
users supply their own OpenRouter API key via the browser UI.

**Key handling rules (non-negotiable):**

1. **User supplies key in the browser UI.**
   - Input field: `<input id="or-key" type="password" />`
   - Key is stored **in sessionStorage only** (expires when tab closes).
   - Key is **never** stored in localStorage or indexedDB.

2. **Key is sent only to OpenRouter, never to ruvector service.**
   - Browser calls OpenRouter SSE endpoint directly (or through a transparent proxy
     that does not log keys).
   - Key is **never** sent to any ruvector backend or logging service.

3. **Key is never committed, embedded, or logged.**
   - No `.env` file with an example key in the repo.
   - No config default or hardcoded placeholder.
   - No browser console logging of the key (safe: log only "key set" / "key cleared").

4. **Security notice displayed to user.**
   - UI label: *"Your key is held only in this browser tab, sent only to
     OpenRouter, and is never uploaded to this site or stored in the repo."*
   - Link to OpenRouter terms: https://openrouter.ai

**Why BYOK?** The browser demo is a public, static HTML site. Shipping a secret
key in a public repo or baking it into the client is a **critical security
violation** (key would be visible in source control, cached CDNs, and Git history).
BYOK shifts responsibility to users (they manage their own keys, which they can
rotate) and avoids leaking the ruvector team's infrastructure keys.

**For production use (GCP-hosted Tier-2 service):**
See ADR-266 "Security: API key management" for the server-side proxy pattern,
where the service holds the key and proxies browser requests. This is suitable
for internal/authenticated deployments where users don't manage keys.

## Validation

### Test plan

1. **Benchmark: 1-hour recorded video (1080p, 30fps source)**
   - Sample at 5 fps → 18k raw frames.
   - Apply keyframe gating (L2 < 0.05 threshold) → expected ~5–7k keyframes.
   - Embed all keyframes: measure time, memory.
   - Index into HNSW.
   - Run 10 test queries ("person speaking", "slide transition", "error message"):
     measure recall@5, latency p50/p99.

2. **User acceptance: browser demo**
   - Open `rupixel/docs/live.html` in Chrome/Firefox/Safari.
   - Capture screen for 5 minutes.
   - Search for 3 semantic moments ("when did I click the button", "code snippet",
     "graphs").
   - Verify results are correct and latency is <2s.

### Commands

```bash
# Build & serve rupixel demo
cd rupixel
npm run build
npm run serve

# Open browser to http://localhost:8080/docs/live.html
# → Select video source (webcam or file)
# → Adjust FPS and dedup threshold
# → Type query, see results + jump to frame

# Benchmark on a recorded video
# (Manual for MVP; automated test harness in Tier 2)
```

### Metrics

| Metric | Target | Measurement |
|---|---|---|
| Keyframe dedup rate | ≥60% (in low-motion scenes) | Count skipped / total frames |
| Embed latency (CPU, ViT-B/32) | 50–100ms per frame | Date.now() delta |
| Index insert latency (HNSW) | <5ms per keyframe | profiler |
| Retrieval p99 (top-5 over 10k keyframes) | <100ms | search latency |
| Recall@5 on test queries | ≥70% (semantic match) | manual eval |
| Memory (embeddings + index) | <100MB for 2-hour video | IndexedDB quota check |

## Consequences

### Positive
- **Closes video-RAG gap in rupixel.** Visual search over continuous streams
  becomes a documented feature, not ad-hoc.
- **Browser-native, zero deployment.** Users can run the demo immediately in any
  modern browser; no server setup, no infrastructure cost.
- **Reuses rupixel + ruvector.** Builds on proven components; no new vision
  model or index library needed.
- **Keyframe gating is generic.** The deduplication strategy is independent of
  the encoder; easily adapted to other vision models (DINO, DINOv2, etc.).
- **Security-first design.** BYOK pattern for API keys (OpenRouter) ensures no
  ruvector secrets are exposed in a public static site. Security responsibility
  is clear: users manage their own keys; browser demo never commits credentials.
- **Path to real-time.** Tier-1 MVP gives users immediate value; Tier-2 (ADR-266)
  unlocks production streaming with server-side key management.

### Negative
- **CPU-only embedding is slow.** WASM CLIP on browser CPU handles ~10–20 frames
  per second. For real-time 30-fps video, requires aggressive sampling (1–5 fps)
  or keyframe filtering. Mitigation: document clearly; offer GPU path in Tier-2.
- **Browser storage limits.** IndexedDB quota is ~50MB default; persistent
  storage requires user permission. Videos >2 hours must use Tier-2 or
  segmentation. Mitigated by sampling + dedup (typical: 20MB for 2 hours).
- **BYOK friction for optional captions.** Users who want streaming LLM captions
  must: (1) create an OpenRouter account, (2) generate an API key, (3) paste it
  into the browser UI. Low-friction for power users, but blocks casual demos.
  Mitigation: captions are optional; demo works fine without them. Production
  (Tier-2) uses server-side proxy to eliminate BYOK friction.
- **Temporal coherence not optimized.** The MVP uses simple L2 distance for
  deduplication; more sophisticated temporal models (optical flow, scene cut
  detection) are deferred to Tier-2. Current approach is acceptable for most
  UX.

### Neutral
- No changes to existing rupixel or ruvector APIs; this is a new client
  (browser extension of live.html).
- Keyframe gating is optional; can be disabled for dataset completeness if
  needed.

## Links

- **Related ADRs:**
  - [[ADR-264-pixelrag-rust-port-on-ruvector|ADR-264]] — static document pixel RAG (parent).
  - [[ADR-266-midstream-streaming-frame-ingestion|ADR-266]] — Tier-2 native Rust streaming.
  - [[ADR-260-photonlayer-optical-computing-simulator|ADR-260]] — optical front-end (orthogonal; not on critical path).
  - [[ADR-262-photonlayer-privacy-preserving-optical-verification|ADR-262]] — privacy narrative (orthogonal).

- **Rupixel references:**
  - Repo: `C:\Users\ruv\ruvector\docs\` (live.html, live.js).
  - CLIP ViT-B/32: https://huggingface.co/openai/clip-vit-base-patch32
  - @xenova/transformers: https://github.com/xenova/transformers.js

- **Video APIs:**
  - getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
  - getDisplayMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia
  - IndexedDB: https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API
  - Canvas: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API

- **Temporal filtering & stream engines:**
  - `temporal-compare` crate (DTW/LCS): https://github.com/ruvnet/midstream (see ADR-266).
  - Optical flow (reference, not used in MVP): https://en.wikipedia.org/wiki/Optical_flow

- **LLM captions & security (optional, Tier-2 integration):**
  - OpenRouter (vision LLM, streaming): https://openrouter.ai/docs#api-overview
  - Browser security (BYOK, sessionStorage): https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage
  - OWASP API key management: https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html
