# PixelRAG eval fixture — small but REAL semantic retrieval set

> **THIS IS A REAL (but TINY) SEMANTIC RETRIEVAL EVAL SET.** Unlike the previous
> 6-tile plumbing stub, this fixture is built so that **meaning, not keyword
> overlap, decides the answer**: every query is a paraphrase that shares few exact
> words with its relevant passages. A pure keyword/BM25 ranker — or the harness's
> non-semantic synthetic embedder — will score **well below 1.0** on recall@10 /
> NDCG@10, which is exactly what makes those metrics meaningful here.
>
> It is still **NOT** the upstream corpus, and we label that honestly:
> - Upstream Wikipedia index is **8.28M pages (~217G FAISS index)** — out of scope.
> - Full **ViDoRe** is 495 docs / 5,191 questions — out of scope here.
> - This set is **30 passages / 12 queries** — a sanity/regression target, not a
>   leaderboard. Absolute numbers from it must not be presented as semantic SOTA.

## What's in it

- **30 tiles** (`tiles/tile-000.txt` … `tiles/tile-029.txt`), each a short factual
  passage of 2–4 plain-text sentences.
- **6 distinct topics, 5 tiles each:**

  | Topic | Tile ids |
  |-------|----------|
  | Photosynthesis / biology      | `tile-000` … `tile-004` |
  | The French Revolution / history | `tile-005` … `tile-009` |
  | Black holes / astronomy       | `tile-010` … `tile-014` |
  | Espresso / coffee             | `tile-015` … `tile-019` |
  | TCP/IP networking             | `tile-020` … `tile-024` |
  | Baroque music                 | `tile-025` … `tile-029` |

- **12 queries** (`queries.json`), **2 per topic**, each a paraphrase that shares
  *meaning* but few exact words with its topic's tiles (forces semantic matching).
- **ground-truth.json** maps each `query_id` to the relevant tile ids — the whole
  same-topic cluster of 5, with the closest-matching tile ranked first so NDCG@10
  rewards correct ordering.

### Tile-id scheme

Zero-padded, three digits, contiguous: `tile-000` … `tile-029`. The id is the file
stem (the harness's `load_tiles` uses the stem directly), and ground-truth
references the same string ids. Query ids are zero-padded two digits: `q-01` … `q-12`.

## File schemas (harness-native)

The Rust bench (`crates/pixelrag-cli/src/bench.rs`) parses specific shapes, so the
two JSON files keep the harness-native object form rather than a bare array/map:

- **`queries.json`** — `{ "dataset", "queries": [ { "query_id", "text", "image" } ] }`.
  Only `text` drives retrieval in M0; `image` paths are placeholders.
- **`ground-truth.json`** — `{ "dataset", "k", "relevance": [ { "query_id", "relevant": [tile_id…] } ] }`.
  Conceptually this is the simple map `{ "q-01": ["tile-000", …], … }`, just wrapped
  in the `relevance` array the harness reads.

## How the harness uses it

`pixelrag-cli benchmark --predictions <results.json> --ground-truth ground-truth.json --queries queries.json`
builds an index from the tiles, runs each query, writes the ranked tile ids to
`--predictions`, and scores recall@10 / NDCG@10 / MRR against `ground-truth.json`.
darwin/MetaHarness (`.metaharness/bench.json`) scores those numbers; `epsilon` in
`bench.json.baseline` is the allowed recall regression vs the baseline.

### Honesty about the numbers

In this environment there is **no real Qwen3-VL-Embedding-2B** (weights + GPU
blocked), so the bundled `benchmark` subcommand still runs the **deterministic
synthetic embedder**, which is non-semantic. On that embedder this fixture will
report **low recall/NDCG** — that is expected and correct: it shows the corpus
*demands* semantics that the stub embedder cannot supply. Plug a real text/vision
embedder (Qwen3-VL, or any sentence encoder) into the pipeline and the same fixture
becomes a genuine — if small — semantic recall@10 / NDCG@10 measurement that lands
between the keyword floor and a perfect 1.0.

## Provenance / licensing

All 30 passages and 12 queries are original prose authored for this repo (no
upstream data copied). Real ViDoRe data is CC-licensed at
<https://huggingface.co/datasets/jinaai/ViDoRe>; download it separately for larger
runs — do not commit the 217G corpus.
