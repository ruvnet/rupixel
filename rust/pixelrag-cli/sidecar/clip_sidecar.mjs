#!/usr/bin/env node
/*
 * CLIP visual embedding sidecar — REAL cross-modal embeddings for the visual path.
 * clip-vit-base-patch32 via transformers.js (pure WASM/CPU, no GPU). First run
 * downloads the quantized model from Hugging Face; then cached.
 *
 * Protocol (one JSON object on stdin, one on stdout):
 *   stdin : {"images": ["/abs/a.png", ...], "texts": ["query", ...]}   (either may be [])
 *   stdout: {"model":"clip-vit-base-patch32","dim":512,
 *            "image_vectors":[[...]], "text_vectors":[[...]]}
 * Vectors are L2-normalized → cosine = dot. Image and text share one space.
 */
import { AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage } from '@xenova/transformers';

const ID = 'Xenova/clip-vit-base-patch32';
const norm = (a) => { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); };

function readStdin() {
  return new Promise((res, rej) => { let b = ''; process.stdin.setEncoding('utf8');
    process.stdin.on('data', (d) => (b += d)); process.stdin.on('end', () => res(b)); process.stdin.on('error', rej); });
}

async function main() {
  const { images = [], texts = [] } = JSON.parse(await readStdin());
  const out = { model: 'clip-vit-base-patch32', dim: 512, image_vectors: [], text_vectors: [] };

  if (images.length) {
    const proc = await AutoProcessor.from_pretrained(ID);
    const vision = await CLIPVisionModelWithProjection.from_pretrained(ID, { quantized: true });
    for (const p of images) {
      const { image_embeds } = await vision(await proc(await RawImage.read(p)));
      out.image_vectors.push(norm(Array.from(image_embeds.data)));
    }
    out.dim = out.image_vectors[0]?.length ?? out.dim;
  }
  if (texts.length) {
    const tok = await AutoTokenizer.from_pretrained(ID);
    const text = await CLIPTextModelWithProjection.from_pretrained(ID, { quantized: true });
    const t = tok(texts, { padding: true, truncation: true });
    const { text_embeds } = await text(t);
    const dim = text_embeds.dims[1];
    for (let i = 0; i < texts.length; i++) out.text_vectors.push(norm(Array.from(text_embeds.data.slice(i * dim, (i + 1) * dim))));
    out.dim = dim;
  }
  process.stdout.write(JSON.stringify(out) + '\n');
}
main().catch((e) => { process.stderr.write(`clip-sidecar: ${e.stack || e}\n`); process.exit(1); });
