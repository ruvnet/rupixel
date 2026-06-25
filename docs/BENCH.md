# PixelRAG — Darwin / MetaHarness optimization bench

> Scope: this document owns the **darwin harness** for the PixelRAG Rust port
> (ADR-264), not the crates. It explains how `@metaharness/darwin` evolves the
> *harness parameters* toward the best `(recall × memory)` Pareto frontier, why
> the harness is removable (ADR-256), and the known blockers.

## What darwin optimizes (and what it does NOT)

Darwin **freezes the Rust source code and algorithm** and evolves only the
*harness parameters* declared in `.metaharness/bench.json → evolveParameters`:

| Parameter | Axis | Default (M1 harness) |
|-----------|------|----------------------|
| `index_backend` | `ruvector-core` HNSW vs `ruvector-rairs` IVF-SQ | `hnsw` |
| `batch_size` | tile-embedding batch | `32` |
| `embedding_cache_mb` | LRU tile-embedding cache | `100` |
| `rerank_threshold` | score cutoff to fire the optional rerank | `0.0` (off) |
| `quantization` | `none` / `4-bit` (rabitq) / `3-bit` / `2-bit` (turbovec, M2+) | `4-bit` |

Each candidate config is scored on the **ViDoRe SUBSET fixture**
(`tests/fixtures/pixelrag/`) by the `pixelrag-cli benchmark` subcommand, which
emits recall@10 / NDCG@10 / MRR plus search p99 and memory/doc. Pass criteria pair
a **quality floor** (`recall@10 >= baseline.recallAt10 - baseline.epsilon`) with a
**resource budget** (search p99 + memory/doc), exactly as ADR-264 §Validation
specifies. Darwin returns a **Pareto frontier of `(config, metrics)`** trading
recall against memory.

## The exact `darwin evolve` invocation

Run from the repo root (`C:/Users/ruv/ruvector`). The suite must build first —
see Blockers.

**Suite integrity (important).** `darwin bench create .` stamps a `taskHash`
over the suite; `darwin bench verify` rejects any hand-edited suite as
"tampered". Therefore `.metaharness/bench.json` must be **darwin-generated, not
hand-authored**. The hand-enriched 6-task/5-param version is kept as
`.metaharness/bench.enriched.json` for reference only (it does NOT pass
`bench verify`). Verified canonical suite:

```bash
npx -y @metaharness/darwin@latest bench create .     # writes valid .metaharness/bench.json
npx -y @metaharness/darwin@latest bench verify ./.metaharness/bench.json   # => hash OK
```

Drive the port toward the best (recall × memory) frontier (real CLI surface —
objectives/constraints live in the suite tasks, not as flags):

```bash
npx -y @metaharness/darwin@latest evolve . \
  --bench ./.metaharness/bench.json \
  --selection pareto \
  --generations 20 \
  --children 12 \
  --seed 42 \
  --sandbox real \
  --mutator deterministic
```

- `--selection pareto` keeps the non-dominated `(recall, memory)` set rather than
  a single scalar winner.
- `--sandbox real` actually runs the bench per candidate; `mock` dry-runs the loop.
- `evolve` mutates the repo via its mutator — run it on a clean/committed tree so
  changes are reviewable; nothing in the Rust runtime reads darwin output
  automatically (removable, ADR-256).

To score a single configuration without evolving, run the project's own harness
directly (there is no `darwin bench run` subcommand):

```bash
cargo run -p pixelrag-cli -- benchmark \
  --ground-truth tests/fixtures/pixelrag/ground-truth.json \
  --queries tests/fixtures/pixelrag/queries.json \
  --metrics ndcg,mrr,recall@10
```

## Removability (ADR-256)

The harness is **fully removable**. Guarantees:

1. **The port builds, indexes, and searches without darwin** using
   `Config::default()` (HNSW, batch=32, cache=100MB, rerank off, 4-bit). The M0
   build gate (`task-0001`) only calls `cargo build` — no darwin dependency.
2. **Darwin output is read-only.** `genome.pixelrag.json` does NOT control the
   Rust runtime via env vars or APIs. A human optionally transcribes a chosen
   config into `Config::from_darwin_json(path)`; if that load fails, the code
   falls back to `Config::default()` (ADR-264 coding rule).
3. **Packaging excludes darwin.** Docker / binary release ship the crates only;
   `.metaharness/` and `@metaharness/darwin` are dev-only and can be deleted with
   zero impact on the shipped library.

## Known blockers

These gate a *real* `darwin evolve` run (the M0 deliverable is the suite +
fixtures + this doc, all offline-clean):

1. **M1 encoder weights / GPU.** True scoring needs the Qwen3-VL-Embedding-2B
   encoder (or a CLIP ONNX surrogate) and likely a GPU. Until M1 lands the
   encoder, the fixtures exercise harness *plumbing* only — recall numbers are
   placeholders, and `bench.json.baseline` is all zeros ("to measure").
2. **Corpus out of scope.** The upstream Wikipedia index is **8.28M pages / ~217G**
   and full ViDoRe is 495 docs / 5,191 questions. This bench uses a **6-tile / 5-query
   SUBSET** (`tests/fixtures/pixelrag/`) — explicitly NOT the upstream corpus. Do
   not commit the 217G index.
3. **CrowdStrike on this Windows host.** Freshly built bench binaries are sometimes
   killed by CrowdStrike before they run (see user memory: "Windows environment
   traps"). Allowlist `target/release/pixelrag-cli*` or run the evolve loop on a
   non-CrowdStrike host / CI shard.
4. **Buildable repo + suite required first.** `darwin evolve` cannot score a tree
   that does not compile. The M0 crates are `unimplemented!()` skeletons, so
   `task-0001` (build gate) passes but the benchmark tasks (`task-0002`…`0006`)
   will not produce real metrics until M1 fills in the encoder + index adaptor.
5. **2/3-bit quantization is conditional on ADR-254.** `quantization` values
   `2-bit`/`3-bit` need `ruvector-turbovec` FastScan, which is proposed, not
   shipped. Until then only `none` and `4-bit` (rabitq) are valid in `task-0006`.

## Files

- `.metaharness/bench.json` — the enriched darwin suite (5 evolve params, 6 tasks).
- `tests/fixtures/pixelrag/` — the subset fixture (tiles + queries + ground truth).
- This file — invocation, removability, blockers.
