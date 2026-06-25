# PixelRAG benchmark fixtures — SUBSET, not the upstream corpus

> **THIS IS A TINY SUBSET FIXTURE.** It exists only so the darwin/MetaHarness
> suite (`.metaharness/bench.json`) and the `pixelrag-cli benchmark` subcommand
> have a runnable, offline, version-controllable target during M0.
>
> It is **NOT** the upstream PixelRAG corpus:
> - Upstream Wikipedia index is **8.28M pages (~217G FAISS index)** — out of scope.
> - Full **ViDoRe** is 495 docs / 5,191 questions — out of scope here.
>
> The "tiles" below are **placeholder text stand-ins** for rendered document
> screenshots. The real M1 pipeline embeds *pixel* tiles via the Qwen3-VL encoder;
> these `.txt` placeholders only exercise the harness plumbing (ingest → index →
> search → score) without GPU/model weights. Replace with a real ViDoRe subset
> (rendered screenshots) once the M1 encoder lands.

## Contents

| File | Role |
|------|------|
| `tiles/tile-000.txt` … `tile-005.txt` | 6 placeholder "tiles" (one document → one tile each). Text stands in for a rendered screenshot. |
| `queries.json` | A handful of queries, each with a `query_id`, `text`, and a placeholder `image` path. |
| `ground-truth.json` | For each `query_id`, the ranked list of relevant `tile_id`s used to compute recall@10 / NDCG@10 / MRR. |

## How the harness uses it

`pixelrag-cli benchmark --predictions <results.json> --ground-truth ground-truth.json`
compares the port's retrieved tile IDs against `ground-truth.json` to produce the
recall@10 / NDCG@10 / MRR numbers that darwin scores. The `epsilon` in
`bench.json.baseline` is the allowed recall regression vs the Python baseline.

## Provenance / licensing

These files are synthetic placeholders authored for this repo (no upstream data
copied). Real ViDoRe data is CC-licensed at
<https://huggingface.co/datasets/jinaai/ViDoRe>; download it separately for M1+
runs — do not commit the 217G corpus.
