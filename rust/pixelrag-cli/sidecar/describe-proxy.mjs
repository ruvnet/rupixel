#!/usr/bin/env node
/*
 * describe-proxy — SECURE server-side proxy for the live-video auto-describe.
 *
 * Holds the OpenRouter key SERVER-SIDE so a shared/GCP key never reaches the
 * browser or the repo. The key is read from the OPENROUTER_API_KEY environment
 * variable at runtime — it is NEVER hard-coded, logged, or written to disk.
 *
 * Run with the GCP-managed key (the key value never appears in your shell history
 * as a literal — it is piped from gcloud into the env):
 *
 *   OPENROUTER_API_KEY="$(gcloud secrets versions access latest --secret=OPENROUTER_API_KEY)" \
 *     node describe-proxy.mjs
 *
 * Then in the live demo (live.html), paste the proxy URL into the key field:
 *   http://localhost:8799/describe
 * The browser POSTs the request body here; this proxy adds the Authorization
 * header and streams OpenRouter's SSE response straight back. The key stays here.
 */
import http from "node:http";

const PORT = process.env.PORT || 8799;
const KEY = process.env.OPENROUTER_API_KEY;
if (!KEY) {
  console.error("describe-proxy: set OPENROUTER_API_KEY in the environment (e.g. from `gcloud secrets versions access`).");
  process.exit(1);
}
const ORIGIN = process.env.ALLOW_ORIGIN || "*"; // tighten for production

http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", ORIGIN);
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  if (req.method === "OPTIONS") { res.writeHead(204).end(); return; }
  if (req.method !== "POST") { res.writeHead(405).end("POST only"); return; }

  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", async () => {
    try {
      const upstream = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: { "Authorization": `Bearer ${KEY}`, "Content-Type": "application/json" },
        body, // forwarded verbatim (model, messages, stream:true)
      });
      res.writeHead(upstream.status, { "Content-Type": upstream.headers.get("content-type") || "text/event-stream" });
      const reader = upstream.body.getReader();
      for (;;) { const { value, done } = await reader.read(); if (done) break; res.write(value); }
      res.end();
    } catch (e) {
      res.writeHead(502).end(`proxy error: ${e.message}`); // never echoes the key
    }
  });
}).listen(PORT, () => console.log(`describe-proxy listening on http://localhost:${PORT}/describe (key from env, not logged)`));
