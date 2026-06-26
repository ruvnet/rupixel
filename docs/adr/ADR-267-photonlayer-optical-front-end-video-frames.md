---
adr: 267
title: "PhotonLayer optical front-end for video frames — experimental edge-sensor preprocessing (off critical path)"
status: Proposed
date: 2026-06-26
authors: [claude-flow]
related: [ADR-265, ADR-266, ADR-260, ADR-262]
supersedes: []
tags: [photonlayer, optical-preprocessing, sensor-compression, privacy, edge-sensing, experimental, optional, off-critical-path]
---

# ADR-267 — PhotonLayer optical front-end for video frames

> **Scope & honest positioning.** This decision explores **PhononLayer** (a
> learned-phase-mask optical front-end simulator, ADR-260) as an **optional
> research-only enhancement** to [[ADR-265-real-time-video-visual-rag-rupixel|ADR-265]]/[[ADR-266-midstream-streaming-frame-ingestion|ADR-266]],
> not on the critical path for Tier-1 or Tier-2 deployment. Explicitly **NOT a
> general vision encoder**, it is a task-trained lossy-compressor; reuses the
> assessment from `docs/research/photonlayer/ASSESSMENT.md` verbatim. This ADR
> records why PhotonLayer does **not** sit on the pixel-RAG retrieval path, and
> what narrow, speculative directions remain for future exploration.

## Status

**Proposed.** Marked **experimental & off critical path**. Implementation is
contingent on clear evidence that edge-sensor bandwidth or privacy constraints
justify the accuracy cost. No blockers on ADR-265/266; both ship without it.

## Context

### The gap (rhetorical)

Question: *Can PhotonLayer's optical compression reduce video **ingest bandwidth**
for real-time video RAG?* Or: *Can it add a privacy layer (compressed
measurements instead of readable frames)?*

**Short answer: No for core retrieval quality; Yes as a speculative future
direction for privacy and edge-sensor use cases (not sized in M0–M3).**

This ADR exists to prevent PhotonLayer from being re-proposed as a solution to
video RAG's size or quality challenges, where it doesn't fit.

### What PhotonLayer actually is (per ASSESSMENT.md)

**Deterministic measurement on MNIST (public dataset, seed 0x6e157, blind test
2000 samples):**

| Component | Metric | Value |
|---|---|---|
| Full-image baseline (tiny centroid decoder) | Sensor pixels / Digital MACs | 1024 / 10,240 |
| Full-image baseline | Accuracy (blind test) | 75.40% |
| **PhotonLayer compressed** (learned optical mask + pooled readout) | Sensor pixels / Digital MACs | 64 / 640 |
| **PhotonLayer compressed** | Accuracy (blind test) | **73.05%** |
| **Δ accuracy vs baseline** | — | **−2.35 pp** |
| Compression ratio | Sensor pixels | **16.0×** |

**Honest positioning (verbatim from ASSESSMENT.md):**

> *A task-trained single optical layer with a tiny digital decoder, classifying
> MNIST within ~1–2 pp of a **matched** full-image tiny-decoder baseline while
> using ≥16× fewer sensor pixels and ≥10× fewer digital MACs. This is
> **competitive single-layer optical compression** — trading a small, quantified
> accuracy margin for large sensor- and compute-savings — **not a new accuracy
> SOTA**; the multi-layer ~97–99% D2NN / optoelectronic regime is explicitly out
> of scope.*

**Documented optimizer ceiling:** Single-mask hill-climb (the M1 training method)
converges ~2 pp short of the acceptance threshold; closing that gap requires
**analytic gradient descent** (not yet shipped). The false choice of "ignore the
ceiling and claim a PASS anyway" is explicitly rejected in ASSESSMENT.md.

**MUST NOT claim** (from ASSESSMENT.md):
- "beats SOTA", "state-of-the-art MNIST" (real SOTA > 99.7%).
- "outperforms D2NNs" (different architecture class).
- Bare "≥16× compression" without the matched baseline context.
- "near-lossless" or "improves quality".

### What already exists (not duplication)

1. **PhotonLayer simulator** (crates/photonlayer-*, ADR-260):
   - Learned phase mask + Fraunhofer FFT propagation + tiny decoder.
   - Validated on MNIST. Ready for research.

2. **Video RAG toolchain** (ADR-265, ADR-266):
   - CLIP ViT-B/32 (general-purpose vision encoder, designed for high semantic
     capacity).
   - Temporal deduplication (L2 distance on embeddings, or perceptual hashing).
   - HNSW indexing (generic ANN over embedding vectors).

3. **Privacy-aware RAG narrative** (ADR-262):
   - Covers *conceptual* privacy (storage, access control, differential privacy).
   - Separate from PhotonLayer's "privacy by physics" claim.

So this ADR is **not introducing new technology** — it is asking: *Does
PhotonLayer belong in the video RAG critical path?* The answer is **no**.

## Decision

### PhotonLayer is out of scope for ADR-265/266 core delivery

**Rationale:**

1. **Category mismatch: lossy compression ≠ retrieval encoder.**
   - PhotonLayer is validated on **image compression** (MNIST pixel reduction,
     1024 → 64 sensor pixels). Document screenshots and video frames are
     different: they carry semantic information (text, tables, faces, layout)
     that lossy optical compression would destroy.
   - CLIP ViT-B/32 is a general-purpose vision encoder, trained to preserve
     semantic information across diverse images. It is **fit-for-purpose** for
     pixel-RAG retrieval.
   - PhotonLayer as a *preprocessing layer* before CLIP would:
     - **Degrade retrieval recall** (lossy compression before the encoder).
     - Add **complexity and audit burden** (two stages of lossy reduction).
     - Provide **no size benefit** (CLIP embeddings, not raw pixels, are indexed).

2. **Size cost is already addressed.**
   - Ingest: Keyframe gating (ADR-265) + temporal-compare dedup (ADR-266)
     achieves 60–70% frame skipping without quality loss.
   - Embeddings: CLIP ViT-B/32 → 512-dim × 4 bytes = 2KB per keyframe. For 10k
     keyframes (2-hour video) = ~20MB. Fits in browser IndexedDB.
   - Index: HNSW graph + vector storage is compact; 1-bit quantization
     (`ruvector-rabitq`) further reduces size if needed.
   - **Lesson: the right tools for each layer (dedup for frames, quantization
     for vectors) beats a single lossy operator bolted to the wrong stage.**

3. **Privacy narrative is orthogonal (see below).**

**Explicit architectural decision:**
- PhotonLayer **will not be on the CLIP input path** (critical retrieval path).
- PhotonLayer **remains an optional, separately-gated experiment** (see "Future
  work").
- If someone wants to use PhotonLayer, it must be:
  - **Explicitly labeled experimental** in docs and code.
  - **Benchmarked separately** (not conflated with core pixel-RAG metrics).
  - **Off by default** (no impact on M0–M3 milestones, no required dependencies).

## Alternative: Privacy-preserving optical ingest (speculative)

**One scenario where PhotonLayer *might* fit:** *"Ingest optical measurements
of videos (not readable images) and retrieve over those measurements, so no
readable frame is ever stored on disk."*

This is conceptually interesting but **requires entirely different validation**:

1. **Privacy claim**: "Measurement is not invertible to readable image" requires
   formal leakage testing (reconstruction attacks, attribute inference,
   membership inference). ASSESSMENT.md notes: *"No readable image is stored"
   is a safer claim than "privacy-preserving" **until leakage is quantified**.*
   **This work is not done.**

2. **Retrieval cost**: CLIP is trained on readable images. If you only ingest
   optical measurements, you must either:
   - Retrain CLIP on optical measurements (expensive, uncertain efficacy).
   - Embed the measurements differently (new model, unproven).
   - Accept lower retrieval recall (honest, but limits utility).

3. **Deployment model**: A sensor that outputs optical measurements (not images)
   is hardware-level (learned diffractive mask, SLM, or tunable metasurface).
   This is **beyond software scope** (see ASSESSMENT.md roadmap: "Hardware bridge"
   is post-simulator). For software-only video RAG, you always have readable
   pixels on the camera/display; the "optical compression happens at sensing
   time" narrative applies only if you control the hardware.

**Verdict on privacy direction:**
- **Speculative and deferred.** Worth exploring as future research (post-M3).
- **Not a blocker** on video RAG MVP. Browser demo (ADR-265) and Tier-2 service
  (ADR-266) are fully deployable without it.
- **If pursued**, must be in a separate ADR with clear privacy threat model,
  leakage quantification, and hardware requirements.
- **Storage location**: Future work → ADR-26x (not this ADR's scope).

## Honest assessment: why PhotonLayer is off-path

| Dimension | PhotonLayer | CLIP | Winner for pixel-RAG |
|---|---|---|---|
| **Purpose** | Lossy optical compression | General vision encoding | CLIP |
| **Validated on** | MNIST (32×32 images, 10-class classification) | 400M image–text pairs, diverse | CLIP |
| **Semantic capacity** | Lossy (−2.35 pp MNIST) | High (designed for diversity) | CLIP |
| **Size cost reduction** | 16× sensor pixels (not applicable to software RAG) | Quantization (1-bit, int8) is orthogonal | Both, on different layers |
| **Complexity** | Single mask + tiny decoder | Proven model, ONNX available | CLIP |
| **Privacy story** | "Measurements not invertible" (unproven) | Transparent; standard practices | Both, separately |
| **Deployment** | Hardware or software simulator | Software (ONNX, transformers.js) | CLIP |

**Clear verdict: CLIP is the right encoder for pixel-RAG retrieval.**

PhotonLayer has a narrow, speculative privacy direction, but only if:
1. Hardware/sensor integration is planned (not software-only video RAG).
2. Leakage is quantified (privacy claim is proved).
3. Retrieval recall cost is acceptable (−2 pp on MNIST; unknown on documents).

None of these hold for M0–M3 of ADR-265/266. PhotonLayer **may be valuable
research**, but not as a pixel-RAG dependency.

## Validation & future work

### If PhotonLayer is ever proposed for video RAG again, require:

1. **Leakage audit** (ASSESSMENT.md roadmap):
   - Linear reconstruction (SVD inversion).
   - Learned decoder (can a CNN invert the measurements?).
   - Diffusion-prior reconstruction (can a generative model invert?).
   - Membership inference (can you tell if a specific image was in training?).
   - Attribute leakage (gender, age, orientation from measurement?).
   - **Publish leak probabilities as risk metrics**, not "privacy-preserving"
     marketing.

2. **Retrieval cost benchmark**:
   - Run CLIP embedding on both readable frames *and* OpticalLayer reconstructed
     frames.
   - Compare HNSW retrieval recall on the two embedding sets.
   - If optical path << readable path, the cost is real and must be disclosed.

3. **Hardware roadmap** (ASSESSMENT.md):
   - Clear path from software simulator → printed mask → SLM lab prototype →
     lensless camera module → production sensing hardware.
   - Software-only video RAG does **not** get the "sensor compression" benefit
     — the optical layer must happen *at image capture time*, not in software.

4. **Separate namespace**:
   - If experiment proceeds, use **separate code paths** (not baked into
     rupixel/ruvector core).
   - Feature flag: `--feature optical-ingest` (off by default).
   - Docs must clearly label as experimental + link to ASSESSMENT.md.

### Non-goals for M0–M3

- Integrating PhotonLayer into CLIP's input path.
- Retraining CLIP on optical measurements.
- Hardware validation (sensor prototyping).
- Privacy leakage testing.

All of these are **future research**, not part of video RAG MVP.

## Consequences

### Positive
- **Clear boundary: PhotonLayer ≠ pixel-RAG encoder.** Prevents architectural
  confusion and feature creep into the critical path.
- **Honest positioning.** Acknowledges PhotonLayer's strengths (optical
  compression, determinism, privacy wedge) without overstating fit for this use
  case.
- **Preserves research direction.** Privacy-by-physics and optical-sensing
  narratives remain available for future ADRs, hardware partnerships, and
  scientific exploration — just not baked into video RAG M0–M3.

### Negative
- **Misses a speculative opportunity.** If privacy-preserving video RAG is a
  business priority, optical ingest could be differentiated. However, the cost
  (hardware integration, leakage testing, retraining) is high for an unproven
  direction; deferring is the right trade-off.
- **Optical compression narrative is siloed.** PhotonLayer remains a research
  artifact; mainstream developers will not know about it or use it. Could have
  larger impact if harder to find, but that's a marketing/org issue, not
  architectural.

### Neutral
- No impact on ADR-265/266 delivery, toolchain, or metrics.
- ADR-260 (PhotonLayer simulator) stands alone; this ADR just clarifies its
  role in the broader system.
- Future privacy-RAG direction is unblocked; ADR-267 is a refusal to couple,
  not a permanent "never".

## Links

- **Related ADRs:**
  - [[ADR-265-real-time-video-visual-rag-rupixel|ADR-265]] — Tier-1 browser pixel-RAG MVP.
  - [[ADR-266-midstream-streaming-frame-ingestion|ADR-266]] — Tier-2 streaming service.
  - [[ADR-260-photonlayer-optical-computing-simulator|ADR-260]] — PhotonLayer simulator & research.
  - [[ADR-262-photonlayer-privacy-preserving-optical-verification|ADR-262]] — Privacy narrative (separate from this ADR's scope).

- **PhotonLayer assessment & research:**
  - `docs/research/photonlayer/ASSESSMENT.md` — measured numbers, optimizer ceiling, honest positioning.
  - Crates: `crates/photonlayer-core/`, `crates/photonlayer-bench/`, etc.
  - Repo: https://github.com/ruvnet/PhotonLayer

- **Optical computing references:**
  - Wirth-Singh et al., *Compressed Meta-Optical Encoder for Image Classification*,
    arXiv:2406.06534, *Adv. Photonics Nexus* 4(2):026009 (2025).
  - Lin et al., *All-optical machine learning using diffractive deep neural networks*,
    *Science* 361:1004 (2018).
  - Privacy-aware meta-optics: *ACS Photonics* (2026).

- **CLIP & vision encoding:**
  - OpenAI CLIP: https://github.com/openai/CLIP
  - CLIP ONNX (for production): https://huggingface.co/openai/clip-vit-base-patch32
  - Qwen3-VL-Embedding: https://huggingface.co/Qwen/Qwen3-VL-Embedding

- **Privacy testing (reference for future work):**
  - Leakage quantification: https://arxiv.org/abs/1802.08686 (membership inference)
  - Reconstruction attacks: https://arxiv.org/abs/1906.08935 (DLG, federated learning)
  - Differential privacy: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
