# rupixel — pixel-native visual RAG, ported to Rust on ruvector

> **Retrieve over what a page *looks like*, not just its text.** `rupixel` is an
> early-stage **Rust port of [PixelRAG](https://github.com/StarTrail-org/PixelRAG)**
> — visual / pixel-native **retrieval-augmented generation** — layered on the
> [ruvector](https://github.com/ruvnet/ruvector) approximate-nearest-neighbor
> substrate (HNSW + IVF-Flat), with a [metaharness](https://www.npmjs.com/package/@metaharness/darwin)
> benchmark + evolution CLI.

[![npm](https://img.shields.io/npm/v/rupixel.svg)](https://www.npmjs.com/package/rupixel)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![status](https://img.shields.io/badge/status-early--stage%20(WIP)-orange.svg)](#project-status)

`rupixel` renders documents — web pages, PDFs, images — to **screenshot tiles**
and retrieves over **visual embeddings**, so tables, charts, and layout survive
retrieval instead of being flattened by an HTML/PDF text parser. It's the
"screenshot the page and search the picture" approach (à la ColPali / visual
document retrieval), implemented in Rust for throughput and small footprint.

```bash
npx rupixel            # what it is, status, links
npx rupixel doctor     # check your environment + harness
npx rupixel bench verify
```

---

## Project status

**This is an honest work-in-progress, not a finished product.** Read this before
you judge the numbers:

| Area | State |
|---|---|
| Rust crate scaffold (`core`/`encoder`/`render`/`serve`/`cli`) | ✅ builds in the ruvector workspace |
| Index pipeline (tile → embed → index → search) | ✅ runs end-to-end |
| ANN backends | ✅ **HNSW** + **IVF-Flat** (real `ruvector` backends) |
| Benchmark harness (metaharness/darwin) | ✅ wired + verifiable |
| Visual encoder | ⏳ **stub** — real `Qwen3-VL-Embedding-2B` needs model weights + GPU |
| Benchmarks | ⚠️ **synthetic embedder on a 6-doc subset fixture** |
| Document rendering (headless-chrome / PDF) | ⏳ stub (M2) |
| HTTP serve + rerank | ⏳ stub (M3) |

> ⚠️ **The current benchmark numbers validate *plumbing*, not *semantic retrieval
> quality*.** They are produced with a deterministic **synthetic** embedder over a
> tiny fixture — meaningful recall vs. the upstream Python baseline requires the
> real Qwen3-VL encoder and a real-scale corpus. We label this everywhere on
> purpose; we won't quote a recall number we didn't earn.

---

## Why pixel-native RAG?

Traditional RAG parses a document to text first, which **loses visual structure** —
multi-column layout, tables, charts, figures, stamps, handwriting. Pixel-native
RAG skips parsing: it renders each page to image tiles and embeds the *pixels*,
so the retriever sees the document the way a human does. Upstream PixelRAG ships a
pre-built index of **8.28M Wikipedia pages** and even accepts an **image as the
query**.

`rupixel` brings that pipeline to Rust on `ruvector`, aiming for production-grade
throughput and memory, with the index swappable between `ruvector` backends.

---

## Quickstart (`npx rupixel`)

The CLI needs only Node ≥ 18 and network access (it wraps `@metaharness/darwin`):

```bash
# project status + architecture + links
npx rupixel info

# environment check (node + darwin + bench suite)
npx rupixel doctor

# (re)generate and verify the darwin benchmark suite
npx rupixel bench create
npx rupixel bench verify

# evolve the harness toward the best (recall × memory) Pareto frontier
#   — meaningful once a real encoder + real-scale corpus are wired (see Roadmap)
npx rupixel evolve --generations 20 --children 12 --seed 42
```

`rupixel` is a thin, dependency-free wrapper. It does **not** compile Rust for
you — the Rust port builds inside the ruvector monorepo (see
[`rust/README.md`](./rust/README.md)).

---

## How it works

```
document ──render──▶ screenshot tiles ──embed──▶ vectors ──index──▶ ruvector ANN
                          (M2 stub)        (encoder)              (HNSW | IVF-Flat)
                                                                        │
query (text or image) ──embed──▶ vector ──────────search──────────────▶ top-k tiles ──▶ reader/LLM
```

- **Render** (`pixelrag-render`, M2): page/PDF → tiles (headless-chrome / pdfium). *stub.*
- **Embed** (`pixelrag-encoder`): `Qwen3-VL-Embedding-2B` (real path, stub) or a
  deterministic **synthetic** embedder for plumbing today.
- **Index/Search** (`pixelrag-core`): adaptor over `ruvector` — **HNSW**
  (`ruvector-core`) or **IVF-Flat** (`ruvector-rairs`), selectable at runtime.
- **Serve** (`pixelrag-serve`, M3): HTTP `/index` `/search` `/health`. *stub.*

### Crates

| Crate | Role |
|---|---|
| `pixelrag-core` | pipeline + tile logic + ANN index adaptor (HNSW / IVF-Flat) |
| `pixelrag-encoder` | visual encoder (ONNX/Qwen real path; synthetic plumbing path) |
| `pixelrag-render` | document → screenshot tiles (M2) |
| `pixelrag-serve` | HTTP retrieval API (M3) |
| `pixelrag-cli` | ingest / search / benchmark harness |

---

## Benchmark harness (metaharness / darwin)

The benchmark suite is **darwin-generated** (`.metaharness/bench.json`) and
integrity-checked — `npx rupixel bench verify` recomputes its `taskHash`. The
harness exposes the optimizable surface (index backend, batch size, cache size)
so `darwin evolve` can search a **Pareto frontier** of `(recall × memory × latency)`.

It is a **removable augmentation** — the Rust port builds, indexes, and searches
with no darwin dependency at runtime. See [`docs/BENCH.md`](./docs/BENCH.md).

Illustrative subset/synthetic run (HNSW vs IVF-Flat, 6-doc fixture — *plumbing only*):

| backend | search p50 | build | memory/doc |
|---|---|---|---|
| `hnsw` | ~0.20 ms | ~8 ms | 520 B |
| `ivf-flat` | ~0.02 ms | ~1 ms | 776 B |

*(IVF trades memory for speed — directionally plausible, noisy at this scale.)*

---

## Roadmap

- **M0 — scaffold** ✅ five crates, workspace-green, darwin harness.
- **M1 — pipeline** ✅ index adaptor over `ruvector`, runnable bench (synthetic).
- **M2 — render + optimize** 🚧 IVF-Flat backend ✅; render port + real Pareto frontier ⏳.
- **M3 — serve** ⏳ HTTP API + optional rerank.

**Blockers (out of scope for the CLI):** real `Qwen3-VL-Embedding-2B` weights +
GPU/ONNX; a real-scale corpus for meaningful recall. Full design + acceptance
criteria in [`docs/ADR-264-pixelrag-rust-port-on-ruvector.md`](./docs/ADR-264-pixelrag-rust-port-on-ruvector.md).

---

## Building the Rust port

The crates use `ruvector` path dependencies, so they build inside the ruvector
monorepo (or via the `external/rupixel` submodule), **not** standalone:

```bash
# from a ruvector checkout that includes these crates
cargo build -p pixelrag-core -p pixelrag-cli
cargo run -p pixelrag-cli -- benchmark \
  --ground-truth tests/fixtures/pixelrag/ground-truth.json \
  --queries tests/fixtures/pixelrag/queries.json \
  --metrics ndcg,mrr,recall@10 --index-backend ivf-flat
```

See [`rust/README.md`](./rust/README.md) for details.

---

## Credits & license

- **Upstream:** [StarTrail-org/PixelRAG](https://github.com/StarTrail-org/PixelRAG)
  (Apache-2.0) — the visual-RAG approach this project ports. `rupixel` is an
  independent Rust reimplementation on `ruvector`; all code here is original.
- **Substrate:** [ruvnet/ruvector](https://github.com/ruvnet/ruvector) — ANN
  indexes (HNSW, IVF-Flat) and the broader vector platform.
- **Harness:** [@metaharness/darwin](https://www.npmjs.com/package/@metaharness/darwin).

Licensed under [MIT](./LICENSE).
