// rupixel — real-time video search, fully in the browser.
// A live feed (sample / webcam / screen) is sampled a few times a second; frames
// that are near-identical to the last kept one are skipped (a keyframe gate); the
// rest are embedded with CLIP ViT-B/32 (WASM/CPU). A text query is embedded the
// same way and cosine-ranked against the kept keyframes. No server, no upload.

import {
  AutoProcessor, AutoTokenizer,
  CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
  RawImage, env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

env.allowLocalModels = false;
const MODEL_ID = "Xenova/clip-vit-base-patch32";

// Prefer WebGPU (frames embed on the GPU); fall back to WASM/CPU automatically.
async function pickDevice() {
  if (navigator.gpu) {
    try { if (await navigator.gpu.requestAdapter()) return "webgpu"; } catch {}
  }
  return "wasm";
}

// --- DOM ---
const feedEl = document.getElementById("feed");
const sampleImg = document.getElementById("sample-frame");
const liveDot = document.getElementById("live-dot");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const qEl = document.getElementById("query");
const resultsEl = document.getElementById("results");
const kfCountEl = document.getElementById("kf-count");
const skipCountEl = document.getElementById("skip-count");
const btnToggle = document.getElementById("btn-toggle");
const btnPause = document.getElementById("btn-pause");
const describeOn = document.getElementById("describe-on");
const orKeyEl = document.getElementById("or-key");
const captionEl = document.getElementById("caption");
const ccOverlay = document.getElementById("cc-overlay");
const KEY_STORE = "rupixel_openrouter_key"; // sessionStorage: origin-scoped, cleared on tab close
const srcButtons = {
  sample: document.getElementById("src-sample"),
  webcam: document.getElementById("src-webcam"),
  screen: document.getElementById("src-screen"),
};

// --- CLIP ---
let tokenizer, textModel, processor, visionModel;

// --- state ---
let source = "sample";          // sample | webcam | screen
let running = false;
let stream = null;              // MediaStream for webcam/screen
let captureTimer = null;
let sampleTimer = null;
let sampleList = [];            // [{image, topic}]
let sampleIdx = 0;
let sampleCycle = 0;            // current sample-feed index (survives pause/resume)
let paused = false;
let lastSig = null;             // perceptual signature of last KEPT frame
let kept = 0, skipped = 0, startMs = 0;
const keyframes = [];           // { label, thumb, vec }
const MAX_KF = 36;
const SIG_THRESHOLD = 9;        // mean abs gray diff (0-255) above which a frame is "new"

const cap = document.createElement("canvas");      // full-ish capture
const capCtx = cap.getContext("2d", { willReadFrequently: true });
const sig = document.createElement("canvas");       // 16x16 signature
sig.width = sig.height = 16;
const sigCtx = sig.getContext("2d", { willReadFrequently: true });

function l2(v) { let s = 0; for (const x of v) s += x * x; const k = 1 / (Math.sqrt(s) || 1); return v.map((x) => x * k); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function setStatus(t, cls) { statusText.innerHTML = t; statusEl.className = "status" + (cls ? " " + cls : ""); }
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function clock(ms) { const s = Math.floor(ms / 1000); return `${String((s / 60) | 0).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; }

// signature: downscale to 16x16 gray, return Uint8Array(256)
function signatureFrom(srcEl, w, h) {
  sigCtx.drawImage(srcEl, 0, 0, w, h, 0, 0, 16, 16);
  const d = sigCtx.getImageData(0, 0, 16, 16).data;
  const g = new Uint8Array(256);
  for (let i = 0; i < 256; i++) g[i] = (d[i * 4] * 0.299 + d[i * 4 + 1] * 0.587 + d[i * 4 + 2] * 0.114) | 0;
  return g;
}
function sigDiff(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; }

async function embedImageFromCanvas() {
  const id = capCtx.getImageData(0, 0, cap.width, cap.height);
  const img = new RawImage(id.data, cap.width, cap.height, 4);
  const { image_embeds } = await visionModel(await processor(img));
  return l2(Array.from(image_embeds.data));
}
async function embedText(text) {
  const { text_embeds } = await textModel(tokenizer(text, { padding: true, truncation: true }));
  return l2(Array.from(text_embeds.data));
}

function activeSourceEl() {
  if (source === "sample") return { el: sampleImg, w: sampleImg.naturalWidth, h: sampleImg.naturalHeight };
  return { el: feedEl, w: feedEl.videoWidth, h: feedEl.videoHeight };
}

let embedding = false;
async function tick() {
  if (!running || embedding) return;
  const { el, w, h } = activeSourceEl();
  if (!w || !h) return;
  cap.width = Math.min(w, 384); cap.height = Math.round(cap.width * h / w);
  capCtx.drawImage(el, 0, 0, cap.width, cap.height);
  const s = signatureFrom(el, w, h);
  if (lastSig && sigDiff(s, lastSig) < SIG_THRESHOLD) { skipped++; skipCountEl.textContent = skipped; return; }
  lastSig = s;
  embedding = true;
  try {
    const vec = await embedImageFromCanvas();
    const thumb = cap.toDataURL("image/jpeg", 0.7);
    const label = source === "sample"
      ? (sampleList[(sampleIdx - 1 + sampleList.length) % sampleList.length]?.topic || `frame ${kept + 1}`)
      : `t = ${clock(performance.now() - startMs)}`;
    const kf = { label, thumb, vec, caption: "" };
    keyframes.push(kf);
    if (keyframes.length > MAX_KF) keyframes.shift();
    kept++; kfCountEl.textContent = kept;
    maybeDescribe(kf);
    if (qEl.value.trim()) runSearch();
  } finally { embedding = false; }
}

// --- optional: stream a one-sentence caption per keyframe via a vision LLM ---
// Security: the key is read live from the input and held only in this tab. It is
// sent ONLY to OpenRouter (or to a proxy URL you paste). It is never persisted to
// the repo, never logged, and never sent to this site. Paste an OpenRouter key
// (sk-or-…) for a direct call, OR paste a proxy URL (http…) that holds the key
// server-side (the secure pattern for a shared/GCP key).
// Most recent Chinese vision model at a strong price/performance point
// (Qwen3-VL flagship MoE, 235B total / 22B active, ~$0.20/Mtok). Cheaper
// alternatives on the same family: qwen/qwen3-vl-32b-instruct, qwen3-vl-8b-instruct.
const VISION_MODEL = "qwen/qwen3-vl-235b-a22b-instruct";
let describeWarned = false;

// Coalesced describer + scrolling history. On heavy motion many keyframes pass the
// gate; instead of describing each (which floods/overwrites), we keep ONE describe
// in flight and always describe the LATEST pending frame, dropping the intermediate
// ones. Finished captions are appended to a timestamped history, and the previous
// caption is passed as TEMPORAL CONTEXT so the model narrates change, not restate.
const captionLog = [];        // [{ ts, text, live, dropped }] newest-first
const MAX_LOG = 14;
let describePending = null;   // newest keyframe awaiting description
let describing = false;
let lastCaption = "";
let burstDropped = 0;         // frames superseded while a describe was in flight

// subtitle overlay: hold each completed caption on screen long enough to read,
// with a cross-fade, decoupled from the fast streaming transcript below.
const MIN_CC_MS = 4200;
let ccShownAt = 0, ccPending = null, ccTimer = null;

function renderCaptionLog() {
  if (!captionLog.length) { captionEl.hidden = true; return; }
  captionEl.hidden = false;
  captionEl.innerHTML = captionLog.map((c) =>
    `<div class="cap-entry${c.live ? " live" : ""}">` +
    `<span class="cap-ts">${esc(c.ts)}${c.dropped ? ` ·+${c.dropped}` : ""}</span>` +
    `<span class="cap-text">${esc(c.text) || "…"}</span></div>`).join("");
}

function showOverlay(text) {
  if (!text) return;
  ccPending = text;
  const wait = Math.max(0, MIN_CC_MS - (performance.now() - ccShownAt));
  clearTimeout(ccTimer);
  ccTimer = setTimeout(applyOverlay, wait);
}
function applyOverlay() {
  if (ccPending == null) return;
  const text = ccPending; ccPending = null;
  ccOverlay.classList.add("fading");           // fade out current line
  setTimeout(() => {
    ccOverlay.textContent = text;
    ccShownAt = performance.now();
    ccOverlay.classList.remove("fading");        // fade the new line in
  }, 240);
}

function maybeDescribe(kf) {
  if (!describeOn.checked) return;
  const cred = orKeyEl.value.trim();
  if (!cred) {
    if (!describeWarned) { setStatus("Auto-describe needs an OpenRouter key (or a proxy URL) in the field above.", "ready"); describeWarned = true; }
    return;
  }
  if (describing || describePending) burstDropped++; // this frame supersedes a queued/in-flight one
  describePending = kf;                              // always describe the freshest
  pumpDescribe();
}

async function pumpDescribe() {
  if (describing || !describePending) return;
  const cred = orKeyEl.value.trim(); if (!cred) return;
  const kf = describePending; describePending = null; describing = true;
  const entry = { ts: source === "sample" ? kf.label : `t=${clock(performance.now() - startMs)}`, text: "", live: true, dropped: burstDropped };
  burstDropped = 0;
  captionLog.unshift(entry);
  while (captionLog.length > MAX_LOG) captionLog.pop();
  renderCaptionLog();
  try { await describeOne(kf, cred, entry); }
  finally {
    entry.live = false; renderCaptionLog();
    describing = false;
    if (describePending) pumpDescribe(); // describe the newest frame that arrived during this one
  }
}

async function describeOne(kf, cred, entry) {
  const viaProxy = /^https?:\/\//i.test(cred);
  const url = viaProxy ? cred : "https://openrouter.ai/api/v1/chat/completions";
  const headers = { "Content-Type": "application/json" };
  if (!viaProxy) headers["Authorization"] = `Bearer ${cred}`; // BYO key → direct to OpenRouter
  const sys = "You narrate a live video feed. Describe ONLY the current frame in one short, concrete sentence. If it continues the previous moment, note what CHANGED instead of restating the scene.";
  const ctx = lastCaption ? `Previous moment: "${lastCaption}". ` : "";
  const body = {
    model: VISION_MODEL, stream: true,
    messages: [
      { role: "system", content: sys },
      { role: "user", content: [
        { type: "text", text: `${ctx}Describe the current frame.` },
        { type: "image_url", image_url: { url: kf.thumb } },
      ] },
    ],
  };
  try {
    const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
    if (!res.ok || !res.body) { entry.text = `describe failed (${res.status})`; renderCaptionLog(); return; }
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop();
      for (const line of lines) {
        const t = line.trim(); if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim(); if (data === "[DONE]") continue;
        try { const tok = JSON.parse(data).choices?.[0]?.delta?.content; if (tok) { kf.caption += tok; entry.text = kf.caption; renderCaptionLog(); } } catch {}
      }
    }
    if (kf.caption.trim()) { lastCaption = kf.caption.trim(); showOverlay(lastCaption); }
    if (qEl.value.trim()) runSearch(); // captions now searchable/shown on cards
  } catch (e) { entry.text = `describe error: ${e.message}`; renderCaptionLog(); }
}

function runSearch() {
  const q = qEl.value.trim();
  if (!q || !keyframes.length) { resultsEl.innerHTML = '<p class="empty">Keyframes will appear here as the feed changes. Type to rank them.</p>'; return; }
  embedText(q).then((qv) => {
    const ranked = keyframes.map((k) => ({ ...k, score: dot(qv, k.vec) })).sort((a, b) => b.score - a.score);
    resultsEl.innerHTML = ranked.map((r, i) => `
      <article class="shot-card${i === 0 ? " best" : ""}">
        <div class="shot-thumb"><img src="${r.thumb}" alt="keyframe" />
          <span class="shot-rank">#${i + 1}</span>${i === 0 ? '<span class="best-badge">best match</span>' : ""}</div>
        <div class="shot-body"><div class="shot-head">
          <span class="shot-topic">${esc(r.label)}</span><span class="shot-score">${r.score.toFixed(3)}</span>
        </div>${r.caption ? `<p class="shot-cap">${esc(r.caption)}</p>` : ""}</div>
      </article>`).join("");
  });
}

// --- sources ---
function stopStream() { if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; } }
function clearKeyframes() { keyframes.length = 0; kept = skipped = 0; lastSig = null; kfCountEl.textContent = "0"; skipCountEl.textContent = "0"; startMs = performance.now(); }

function startSampleCycle() {
  sampleImg.src = sampleList[sampleCycle].image;
  sampleIdx = (sampleCycle + 1) % sampleList.length;
  sampleTimer = setInterval(() => {
    sampleCycle = (sampleCycle + 1) % sampleList.length;
    sampleImg.src = sampleList[sampleCycle].image;
    sampleIdx = (sampleCycle + 1) % sampleList.length;
  }, 1200);
}

async function start() {
  clearKeyframes(); paused = false; sampleCycle = 0;
  if (source === "sample") {
    feedEl.hidden = true; sampleImg.hidden = false; liveDot.hidden = true;
    startSampleCycle();
  } else {
    sampleImg.hidden = true; feedEl.hidden = false;
    try {
      stream = source === "webcam"
        ? await navigator.mediaDevices.getUserMedia({ video: { width: 640 } })
        : await navigator.mediaDevices.getDisplayMedia({ video: true });
    } catch (e) { setStatus(`Could not open ${source}: ${e.message}. Try the sample feed.`, "error"); return; }
    feedEl.srcObject = stream; await feedEl.play(); liveDot.hidden = false;
  }
  running = true; btnToggle.textContent = "Stop"; btnToggle.classList.add("running");
  btnPause.disabled = false; btnPause.textContent = "⏸ Pause"; btnPause.classList.remove("active");
  captureTimer = setInterval(tick, 650);
  setStatus(source === "sample" ? "Sample feed running — watching for scene changes…" : "Live — watching the feed for scene changes…", "ready");
}

function pause() {
  if (!running || paused) return;
  paused = true;
  clearInterval(captureTimer); clearInterval(sampleTimer);
  if (stream && !feedEl.paused) feedEl.pause();
  liveDot.hidden = true;
  btnPause.textContent = "▶ Resume"; btnPause.classList.add("active");
  setStatus("Paused — the frame and captions stay up so you can read. Resume to continue.", "ready");
}
function resume() {
  if (!running || !paused) return;
  paused = false;
  if (source === "sample") startSampleCycle();
  else if (stream) { feedEl.play(); liveDot.hidden = false; }
  captureTimer = setInterval(tick, 650);
  btnPause.textContent = "⏸ Pause"; btnPause.classList.remove("active");
  setStatus("Resumed.", "ready");
}

function stop() {
  running = false; paused = false; clearInterval(captureTimer); clearInterval(sampleTimer); stopStream();
  liveDot.hidden = true; btnToggle.textContent = "Start"; btnToggle.classList.remove("running");
  btnPause.disabled = true; btnPause.textContent = "⏸ Pause"; btnPause.classList.remove("active");
  setStatus("Stopped.", "ready");
}
btnToggle.addEventListener("click", () => (running ? stop() : start()));
btnPause.addEventListener("click", () => (paused ? resume() : pause()));
for (const [k, b] of Object.entries(srcButtons)) {
  b.addEventListener("click", () => { if (running) stop(); source = k; Object.values(srcButtons).forEach((x) => x.classList.remove("active")); b.classList.add("active"); });
}

// --- boot ---
async function init() {
  try {
    const device = await pickDevice();
    const dtype = device === "webgpu" ? "fp32" : "q8"; // fp32 keeps CLIP ranking crisp on WebGPU
    setStatus(`Loading CLIP ViT-B/32 on <strong>${device.toUpperCase()}</strong>…`);
    [tokenizer, textModel, processor, visionModel] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID),
      CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { device, dtype }),
      AutoProcessor.from_pretrained(MODEL_ID),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { device, dtype }),
    ]);
    window.__rupixelDevice = device;
    sampleList = await fetch("corpus-img/manifest.json").then((r) => r.json()).then((m) => m.map((d) => ({ image: d.image, topic: d.topic })));
    setStatus(`Ready on <strong>${device.toUpperCase()}</strong> — pick a source and press Start (default is a sample feed).`, "ready");
    qEl.disabled = false; btnToggle.disabled = false;
    qEl.addEventListener("input", () => runSearch());

    // Secure client-side key storage: sessionStorage is origin-scoped and cleared
    // when the tab closes (not localStorage, which would persist on disk). The key
    // never leaves the browser except to OpenRouter / your proxy, and is never sent
    // to this site or committed. Restore it for this session and save on edit.
    try { orKeyEl.value = sessionStorage.getItem(KEY_STORE) || ""; } catch {}
    orKeyEl.addEventListener("input", () => {
      describeWarned = false;
      try {
        const v = orKeyEl.value.trim();
        if (v) sessionStorage.setItem(KEY_STORE, v); else sessionStorage.removeItem(KEY_STORE);
      } catch {}
    });
    await start(); // auto-start the sample feed so the demo is alive on load
  } catch (e) {
    console.error(e);
    setStatus("Failed to load CLIP — needs the transformers.js CDN. Reload to retry.", "error");
  }
}
init();
