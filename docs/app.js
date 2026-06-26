// rupixel — in-browser semantic search demo.
// Runs all-MiniLM-L6-v2 fully client-side via transformers.js (WASM/CPU).
// No server, no precomputed vectors: the corpus is embedded live on page load.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Pull model weights from the HF CDN (no local model files in this repo).
env.allowLocalModels = false;

const qEl = document.getElementById("query");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const resultsEl = document.getElementById("results");

const TOP_K = 6;

let extractor = null;     // feature-extraction pipeline
let corpus = [];          // [{ id, topic, text }]
let corpusVectors = [];   // Float32Array[] aligned with corpus, L2-normalized

// ---- math helpers -------------------------------------------------------

// transformers.js with { pooling: "mean", normalize: true } already returns a
// unit vector, so dot product == cosine similarity.
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function embed(text) {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return out.data; // Float32Array(384)
}

// ---- rendering ----------------------------------------------------------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderResults(ranked) {
  if (!ranked.length) {
    resultsEl.innerHTML = '<p class="empty">Type a question to search the corpus.</p>';
    return;
  }

  // Scale bar widths relative to the top hit so the strongest match reads as full.
  const top = ranked[0].score || 1;

  resultsEl.innerHTML = ranked
    .map((r, i) => {
      const pct = Math.max(2, Math.round((r.score / top) * 100));
      const scoreStr = r.score.toFixed(3);
      return `
        <article class="card">
          <div class="card-head">
            <div class="card-meta">
              <span class="rank">#${i + 1}</span>
              <span class="topic">${escapeHtml(r.topic)}</span>
              <span class="tile-id">${escapeHtml(r.id)}</span>
            </div>
            <span class="score">${scoreStr}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" data-w="${pct}"></div></div>
          <p class="card-text">${escapeHtml(r.text)}</p>
        </article>`;
    })
    .join("");

  // Animate bars in on the next frame.
  requestAnimationFrame(() => {
    resultsEl.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

// ---- search -------------------------------------------------------------

let searchToken = 0;

async function runSearch() {
  const query = qEl.value.trim();
  if (!query || !extractor) {
    renderResults([]);
    return;
  }

  const token = ++searchToken;
  const qVec = await embed(query);
  if (token !== searchToken) return; // a newer keystroke superseded this one

  const ranked = corpus
    .map((doc, i) => ({ ...doc, score: dot(qVec, corpusVectors[i]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);

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
    setStatus("Loading all-MiniLM-L6-v2 (quantized, ~23&nbsp;MB)…");
    extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      quantized: true,
    });

    setStatus("Loading corpus…");
    corpus = await fetch("corpus.json").then((r) => r.json());

    setStatus(`Embedding ${corpus.length} passages in your browser…`);
    // Embed sequentially to keep memory flat and give clear progress.
    corpusVectors = [];
    for (let i = 0; i < corpus.length; i++) {
      corpusVectors.push(await embed(corpus[i].text));
    }

    const topics = new Set(corpus.map((d) => d.topic)).size;
    setStatus(
      `Ready — ${corpus.length} passages across ${topics} topics, embedded locally. Search runs on every keystroke.`,
      "ready"
    );

    qEl.disabled = false;
    qEl.focus();
    qEl.addEventListener("input", debounce(runSearch, 180));

    // Run the prefilled default query immediately for a screenshot-ready view.
    await runSearch();
  } catch (err) {
    console.error(err);
    setStatus(
      "Failed to load the model. Check your connection and reload — the demo needs the transformers.js CDN.",
      "error"
    );
  }
}

init();
