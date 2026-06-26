# Benchmark — traditional (text) RAG vs visual RAG

A like-for-like comparison of the two retrieval paths in rupixel, on the **same
documents, the same queries, and the same ground truth** — only the *modality*
differs:

- **Traditional / text RAG** — `all-MiniLM-L6-v2` (384-d) embeds each page's
  **extracted text**; the text query is matched against text vectors.
- **Visual RAG** — `clip-vit-base-patch32` (512-d) embeds each page's **rendered
  screenshot**; the text query is matched against image vectors (cross-modal).

> **Honesty up front:** this corpus is small (8 documents) and **text-clean**
> (Wikipedia articles with a good text layer and topically distinct subjects).
> On data like this, *both* paths are expected to do well — and they do. This
> benchmark is here to show the comparison is **real and reproducible**, and to
> be honest about *where each modality actually wins*, not to manufacture a gap.

## Setup

- **Corpus:** 8 documents across 8 distinct topics (black holes, French
  Revolution, photosynthesis, espresso, TCP/IP, baroque music, sunflowers, the
  Great Barrier Reef). Each exists in **both** modalities:
  text in `tests/fixtures/pixelrag/compare/text/tiles/*.txt`, screenshot in
  `tests/fixtures/pixelrag/visual/images/*.png` (rendered with `pixelrag-render`).
- **Queries:** 8 paraphrase queries (one per topic) sharing meaning but little
  vocabulary with their target — so retrieval must be *semantic*, not keyword.
- **Ground truth:** 1 relevant document per query. **Index:** ruvector HNSW.
- **Embedders run on CPU/WASM** (no GPU): MiniLM and CLIP via the same
  transformers.js sidecars the demos use.

## Results (measured)

| Metric | Traditional text RAG (MiniLM) | Visual RAG (CLIP) |
|---|---:|---:|
| **top-1 accuracy** | **1.00** (8/8) | **1.00** (8/8)¹ |
| recall@10 | 1.00 | 1.00 |
| nDCG@10 | 1.00 | 1.00 |
| MRR | 1.00 | 1.00 |
| query latency p50 | 0.62 ms | 0.52 ms |
| embedding dim | 384 | 512 |
| model (quantized) | all-MiniLM-L6-v2 (~23 MB) | clip-vit-base-patch32 (~85 MB) |
| input it needs | a clean **text layer** | a **rendered image** (pixels) |
| pre-step required | text extraction / parse | page render (`pixelrag-render`) |

¹ **8/8 with the native (sharp) image preprocessing used by the Rust bench.** The
**in-browser** demo (canvas preprocessing) scores **7/8 top-1, MRR 0.94** — one
near-tie, where *"a vibrant underwater coral ecosystem"* ranks the coral-reef
page #2 behind photosynthesis (both green nature scenes; scores within 0.02).
Same model, different image resampling → the tie flips. Reproduce in your browser
at the [visual demo](https://ruvnet.github.io/rupixel/visual.html).

## What this does — and doesn't — show

**Accuracy ties here.** With distinct topics and a clean text layer, both
modalities retrieve perfectly. Accuracy alone does **not** separate them on this
corpus, and we don't pretend it does.

**The real trade-off is qualitative:**

| | Traditional text RAG | Visual RAG |
|---|---|---|
| Needs a usable text layer | **Yes** — breaks on scans, image-only PDFs, screenshots, charts | **No** — reads pixels directly |
| Preserves layout / tables / figures | No — flattened to a token stream | **Yes** — the page *is* the input |
| Fine-grained text understanding | **Strong** | Weaker (CLIP ViT-B/32 is a baseline) |
| Cost per doc | text parse (cheap) | render + larger model (heavier) |

So: **traditional RAG is the right default for clean, text-rich documents** —
it's cheap, fast, and strong. **Visual RAG earns its keep where text extraction
fails or loses structure** — scanned documents, complex layouts, tables, charts,
forms — which *this* corpus deliberately does not stress.

## Where visual RAG should win (next benchmark)

The honest next step is a corpus that breaks text extraction: scanned/image-only
pages, multi-column layouts, table- and chart-heavy documents. There, text RAG
degrades (or returns nothing) while visual RAG still retrieves. A
document-specialized visual encoder (**Qwen3-VL / ColPali**, GPU) would also lift
the visual numbers well above the CLIP-baseline used here. That comparison is
tracked as future work — we report only what we have measured.

## Reproduce

```bash
# from a ruvector checkout that includes the pixelrag crates
( cd crates/pixelrag-cli/sidecar && npm install )   # MiniLM + CLIP sidecars

# Traditional text RAG (MiniLM over extracted page text)
cargo run -p pixelrag-cli -- benchmark --mode text --embedder real \
  --ground-truth tests/fixtures/pixelrag/compare/text/ground-truth.json \
  --queries      tests/fixtures/pixelrag/compare/text/queries.json \
  --tiles        tests/fixtures/pixelrag/compare/text/tiles \
  --metrics ndcg,mrr,recall@10 --index-backend hnsw

# Visual RAG (CLIP over rendered screenshots, same 8 docs/queries)
cargo run -p pixelrag-cli -- benchmark --mode visual --index-backend hnsw
```

Both write JSON reports to `bench_output/`. The visual demo
([visual.html](https://ruvnet.github.io/rupixel/visual.html)) and text demo
([index.html](https://ruvnet.github.io/rupixel/)) run the same two models live in
your browser.
