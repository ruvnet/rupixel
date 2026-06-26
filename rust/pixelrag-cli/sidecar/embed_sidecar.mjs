#!/usr/bin/env node
/*
 * pixelrag embedding sidecar — REAL semantic embeddings for the Rust bench.
 *
 * Runs all-MiniLM-L6-v2 (sentence-transformers) via transformers.js — pure WASM,
 * CPU only, no GPU, no native onnxruntime. First run downloads the quantized
 * model (~30MB) from Hugging Face; subsequent runs use the local cache.
 *
 * Protocol (line-delimited JSON over stdin/stdout):
 *   stdin : {"texts": ["...", "..."]}            (one JSON object, then EOF)
 *   stdout: {"model":"all-MiniLM-L6-v2","dim":384,"vectors":[[...],[...]]}
 * Embeddings are mean-pooled and L2-normalized (cosine-ready).
 */
import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false; // always resolve from HF cache

const MODEL = 'Xenova/all-MiniLM-L6-v2';

function readStdin() {
  return new Promise((resolve, reject) => {
    let buf = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (d) => (buf += d));
    process.stdin.on('end', () => resolve(buf));
    process.stdin.on('error', reject);
  });
}

async function main() {
  const raw = await readStdin();
  let texts;
  try {
    texts = JSON.parse(raw).texts;
  } catch (e) {
    process.stderr.write(`sidecar: bad stdin JSON: ${e.message}\n`);
    process.exit(2);
  }
  if (!Array.isArray(texts) || texts.length === 0) {
    process.stderr.write('sidecar: "texts" must be a non-empty array\n');
    process.exit(2);
  }

  const extractor = await pipeline('feature-extraction', MODEL, { quantized: true });
  const vectors = [];
  for (const t of texts) {
    const out = await extractor(String(t), { pooling: 'mean', normalize: true });
    vectors.push(Array.from(out.data));
  }
  const dim = vectors[0]?.length ?? 0;
  process.stdout.write(JSON.stringify({ model: 'all-MiniLM-L6-v2', dim, vectors }) + '\n');
}

main().catch((e) => {
  process.stderr.write(`sidecar: ${e.stack || e}\n`);
  process.exit(1);
});
