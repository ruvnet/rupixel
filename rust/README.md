# rupixel — Rust crates

These five crates implement the PixelRAG → Rust port on the `ruvector` substrate.

```
pixelrag-core      pipeline + tile logic + ANN index adaptor (HNSW / IVF-Flat)
pixelrag-encoder   visual encoder (ONNX/Qwen real path; synthetic plumbing path)
pixelrag-render    document → screenshot tiles (M2 stub)
pixelrag-serve     HTTP retrieval API (M3 stub)
pixelrag-cli       ingest / search / benchmark harness
```

## Important: these build inside the ruvector monorepo, not standalone

The crates depend on `ruvector` crates via **path dependencies**, e.g.:

```toml
# pixelrag-core/Cargo.toml
ruvector-core  = { path = "../ruvector-core" }    # HNSW backend
ruvector-rairs = { path = "../ruvector-rairs" }   # IVF-Flat backend
```

So they compile when these crates sit alongside the `ruvector` crates — i.e. in a
[ruvnet/ruvector](https://github.com/ruvnet/ruvector) checkout (where they are
also registered in the workspace `members`), or via the `external/rupixel`
submodule wired into that workspace.

This directory is the **source-of-record snapshot** for the standalone `rupixel`
repo / npm package. The canonical, continuously-built copy lives in the ruvector
monorepo under `crates/pixelrag-*`. Treat the monorepo as authoritative if the
two ever drift.

## Build & bench (from a ruvector checkout)

```bash
cargo build -p pixelrag-core -p pixelrag-encoder -p pixelrag-render \
            -p pixelrag-serve -p pixelrag-cli

cargo run -p pixelrag-cli -- benchmark \
  --ground-truth tests/fixtures/pixelrag/ground-truth.json \
  --queries tests/fixtures/pixelrag/queries.json \
  --metrics ndcg,mrr,recall@10 \
  --index-backend hnsw      # or: ivf-flat
```

Current benchmarks use a **synthetic** embedder on a **subset** fixture —
plumbing validation, not semantic retrieval quality. See `../docs/ADR-264-*` and
`../docs/BENCH.md`.
