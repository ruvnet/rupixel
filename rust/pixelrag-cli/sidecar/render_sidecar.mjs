#!/usr/bin/env node
/*
 * Render sidecar — REAL document → screenshot rendering for the visual path.
 * Drives headless Chrome/Edge via puppeteer-core (needs a Chromium-family browser
 * installed; set PIXELRAG_BROWSER to its path, else common Edge/Chrome paths are tried).
 *
 * Protocol:
 *   stdin : {"urls":["https://…", …], "outDir":"/abs/dir", "width":1024, "height":768}
 *   stdout: {"images":[{"id":"doc-00","url":"…","path":"/abs/dir/doc-00.png"}, …]}
 */
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CANDIDATES = [
  process.env.PIXELRAG_BROWSER,
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);

function readStdin() {
  return new Promise((res, rej) => { let b = ''; process.stdin.setEncoding('utf8');
    process.stdin.on('data', (d) => (b += d)); process.stdin.on('end', () => res(b)); process.stdin.on('error', rej); });
}

async function main() {
  const { urls = [], outDir, width = 1024, height = 768 } = JSON.parse(await readStdin());
  if (!outDir) { process.stderr.write('render-sidecar: outDir required\n'); process.exit(2); }
  const exe = CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch { return false; } });
  if (!exe) { process.stderr.write('render-sidecar: no Chromium/Edge/Chrome found (set PIXELRAG_BROWSER)\n'); process.exit(3); }
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({ executablePath: exe, headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage'], defaultViewport: { width, height, deviceScaleFactor: 1 } });
  const images = [];
  try {
    for (let i = 0; i < urls.length; i++) {
      const id = `doc-${String(i).padStart(2, '0')}`;
      const file = path.join(outDir, `${id}.png`);
      const page = await browser.newPage();
      try {
        await page.goto(urls[i], { waitUntil: 'networkidle2', timeout: 45000 });
        await new Promise((r) => setTimeout(r, 700));
        await page.screenshot({ path: file, clip: { x: 0, y: 0, width, height } });
        images.push({ id, url: urls[i], path: file });
      } finally { await page.close(); }
    }
  } finally { await browser.close(); }
  process.stdout.write(JSON.stringify({ images }) + '\n');
}
main().catch((e) => { process.stderr.write(`render-sidecar: ${e.stack || e}\n`); process.exit(1); });
