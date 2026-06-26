// rupixel — in-browser VISUAL RAG demo.
// Real CLIP ViT-B/32 running fully client-side via transformers.js (WASM/CPU).
// Text and image share ONE embedding space: a text query is cosine-ranked against
// real document screenshots. No server, no precomputed vectors — every screenshot
// is embedded live on page load, the query is embedded live on input.

import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Pull model weights from the HF CDN (no local model files in this repo).
env.allowLocalModels = false;

const MODEL_ID = "Xenova/clip-vit-base-patch32";

const qEl = document.getElementById("query");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const resultsEl = document.getElementById("results");

// CLIP components.
let tokenizer = null;          // text tokenizer
let textModel = null;          // CLIPTextModelWithProjection
let processor = null;          // image processor
let visionModel = null;        // CLIPVisionModelWithProjection

let corpus = [];               // [{ id, topic, url, image }]
let imageVectors = [];         // Float32Array[] aligned with corpus, L2-normalized

// ---- math helpers -------------------------------------------------------

function l2normalize(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const inv = 1 / (Math.sqrt(s) || 1);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}

// Both query and image vectors are L2-normalized, so dot product == cosine.
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// ---- embedding ----------------------------------------------------------

async function embedImage(path) {
  const image = await RawImage.read(path);
  const inputs = await processor(image);
  const { image_embeds } = await visionModel(inputs);
  return l2normalize(image_embeds.data); // Float32Array(512)
}

async function embedText(text) {
  const inputs = tokenizer(text, { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);
  return l2normalize(text_embeds.data); // Float32Array(512)
}

// ---- rendering ----------------------------------------------------------

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderResults(ranked) {
  if (!ranked.length) {
    resultsEl.innerHTML = '<p class="empty">Type a query to rank the document screenshots.</p>';
    return;
  }

  resultsEl.innerHTML = ranked
    .map((r, i) => {
      const best = i === 0;
      const scoreStr = r.score.toFixed(3);
      return `
        <article class="shot-card${best ? " best" : ""}">
          <div class="shot-thumb">
            <img src="${escapeHtml(r.image)}" alt="${escapeHtml(r.topic)} screenshot" loading="lazy" />
            <span class="shot-rank">#${i + 1}</span>
            ${best ? '<span class="best-badge">best match</span>' : ""}
          </div>
          <div class="shot-body">
            <div class="shot-head">
              <span class="shot-topic">${escapeHtml(r.topic)}</span>
              <span class="shot-score">${scoreStr}</span>
            </div>
            <span class="shot-id">${escapeHtml(r.id)}</span>
          </div>
        </article>`;
    })
    .join("");
}

// ---- search -------------------------------------------------------------

let searchToken = 0;

async function runSearch() {
  const query = qEl.value.trim();
  if (!query || !textModel) {
    renderResults([]);
    return;
  }

  const token = ++searchToken;
  const qVec = await embedText(query);
  if (token !== searchToken) return; // a newer keystroke superseded this one

  const ranked = corpus
    .map((doc, i) => ({ ...doc, score: dot(qVec, imageVectors[i]) }))
    .sort((a, b) => b.score - a.score);

  renderResults(ranked);
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ---- boot ---------------------------------------------------------------

function setStatus(text, cls) {
  statusText.innerHTML = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

async function init() {
  try {
    setStatus("Loading CLIP ViT-B/32 (quantized)…");
    // Load the four CLIP components in parallel from the HF CDN.
    [tokenizer, textModel, processor, visionModel] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID),
      CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { quantized: true }),
      AutoProcessor.from_pretrained(MODEL_ID),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { quantized: true }),
    ]);

    setStatus("Loading document manifest…");
    corpus = await fetch("corpus-img/manifest.json").then((r) => r.json());

    // Embed each screenshot sequentially to keep memory flat and show progress.
    imageVectors = [];
    for (let i = 0; i < corpus.length; i++) {
      setStatus(`Embedding ${i + 1} / ${corpus.length} document images…`);
      imageVectors.push(await embedImage(corpus[i].image));
    }

    const topics = new Set(corpus.map((d) => d.topic)).size;
    setStatus(
      `Ready — ${corpus.length} document screenshots across ${topics} topics, embedded locally with CLIP. Search runs on every keystroke.`,
      "ready"
    );

    qEl.disabled = false;
    qEl.focus();
    qEl.addEventListener("input", debounce(runSearch, 200));

    // Run the prefilled default query immediately for a screenshot-ready view.
    await runSearch();
  } catch (err) {
    console.error(err);
    setStatus(
      "Failed to load CLIP. Check your connection and reload — the demo needs the transformers.js CDN.",
      "error"
    );
  }
}

init();
