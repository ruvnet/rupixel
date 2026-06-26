# rupixel — Rust crates

These crates implement the PixelRAG → Rust port on the `ruvector` substrate.
All shipped code is real — no `unimplemented!()` stubs.

```
pixelrag-core      pipeline + tile logic + ANN index adaptor (HNSW / IVF-Flat)
pixelrag-encoder   real all-MiniLM-L6-v2 embedder (WASM/CPU sidecar)
pixelrag-cli       benchmark harness (recall/ndcg/mrr + latency/build/memory)
```

The `pixelrag-cli/sidecar/` folder holds the Node embedding sidecar
(`embed_sidecar.mjs` + `package.json`); run `npm install` there once before the
`--embedder real` benchmark. The visual encoder (Qwen3-VL over rendered
screenshots) is roadmap, not shipped stub code.

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
