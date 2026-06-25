#!/usr/bin/env node
/*
 * rupixel — CLI for the PixelRAG → Rust/ruvector port and its metaharness
 * benchmark harness.
 *
 * This CLI is a thin, dependency-free wrapper around `@metaharness/darwin`
 * (the benchmark/evolution harness) plus helpers for the Rust port. It works
 * standalone via `npx rupixel`. The Rust port itself builds inside the ruvector
 * monorepo (see rust/README.md) — this package does not compile Rust for you.
 *
 * Honest status: the port is early-stage (M0 scaffold + M1 plumbing + an
 * IVF-Flat backend). Current benchmarks use a SYNTHETIC embedder on a tiny
 * subset fixture — they validate plumbing, NOT semantic retrieval quality.
 */
'use strict';

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const VERSION = require('../package.json').version;
const DARWIN = '@metaharness/darwin@latest';

const C = process.stdout.isTTY
  ? { b: '\x1b[1m', d: '\x1b[2m', g: '\x1b[32m', y: '\x1b[33m', r: '\x1b[31m', x: '\x1b[0m' }
  : { b: '', d: '', g: '', y: '', r: '', x: '' };

function log(s = '') { process.stdout.write(s + '\n'); }

function banner() {
  log(`${C.b}rupixel${C.x} v${VERSION} ${C.d}— pixel-native visual RAG, ported to Rust on ruvector${C.x}`);
  log(`${C.y}status:${C.x} early-stage (M0 scaffold + M1 plumbing + IVF-Flat backend).`);
  log(`${C.d}benchmarks here use a synthetic embedder on a subset fixture — plumbing, not semantic quality.${C.x}`);
}

function help() {
  banner();
  log('');
  log(`${C.b}Usage:${C.x} npx rupixel <command> [args]`);
  log('');
  log(`  ${C.b}info${C.x}             Show project status, layout, and links (default)`);
  log(`  ${C.b}doctor${C.x}           Check node + @metaharness/darwin availability`);
  log(`  ${C.b}bench create${C.x}     Scaffold a darwin benchmark suite (.metaharness/bench.json)`);
  log(`  ${C.b}bench verify${C.x}     Verify the darwin suite integrity (taskHash)`);
  log(`  ${C.b}evolve${C.x} [args]    Run darwin evolve over the pixelrag suite (Pareto: recall × memory)`);
  log(`  ${C.b}version${C.x}          Print version`);
  log('');
  log(`${C.d}The Rust port builds inside the ruvector monorepo — see rust/README.md.${C.x}`);
  log(`${C.d}Docs: docs/ADR-264-pixelrag-rust-port-on-ruvector.md, docs/BENCH.md${C.x}`);
}

function darwin(args, opts = {}) {
  const r = spawnSync('npx', ['-y', DARWIN, ...args], { stdio: 'inherit', shell: process.platform === 'win32', ...opts });
  return r.status == null ? 1 : r.status;
}

function doctor() {
  banner();
  log('');
  log(`node            ${C.g}${process.version}${C.x}`);
  const probe = spawnSync('npx', ['-y', DARWIN, '--help'], { encoding: 'utf8', shell: process.platform === 'win32' });
  if (probe.status === 0) log(`@metaharness/darwin  ${C.g}reachable${C.x}`);
  else { log(`@metaharness/darwin  ${C.r}not reachable${C.x} (needs network / npm)`); return 1; }
  const suite = path.resolve(process.cwd(), '.metaharness', 'bench.json');
  log(`bench suite     ${fs.existsSync(suite) ? C.g + 'present' : C.y + 'missing (run: rupixel bench create)'}${C.x}`);
  return 0;
}

function info() {
  banner();
  log('');
  log(`${C.b}What it is:${C.x} a Rust port of PixelRAG (visual / pixel-native retrieval-augmented`);
  log(`generation) layered on the ruvector ANN substrate (HNSW + IVF-Flat). PixelRAG renders`);
  log(`documents to screenshot tiles and retrieves over visual embeddings instead of parsed text.`);
  log('');
  log(`${C.b}Done:${C.x} 5 Rust crates (core/encoder/render/serve/cli), darwin benchmark harness,`);
  log(`HNSW + IVF-Flat backends, runnable end-to-end on a subset fixture.`);
  log(`${C.b}Blocked / TODO:${C.x} real Qwen3-VL-Embedding-2B encoder (weights+GPU), render port,`);
  log(`full-corpus benchmark. See docs/ADR-264 for the roadmap.`);
  log('');
  log(`${C.b}Upstream:${C.x} https://github.com/StarTrail-org/PixelRAG (Apache-2.0)`);
  log(`${C.b}Substrate:${C.x} https://github.com/ruvnet/ruvector`);
  return 0;
}

function main(argv) {
  const [cmd, sub, ...rest] = argv;
  switch (cmd) {
    case undefined:
    case 'info': return info();
    case 'help': case '-h': case '--help': help(); return 0;
    case 'version': case '-v': case '--version': log(VERSION); return 0;
    case 'doctor': return doctor();
    case 'bench': {
      if (sub === 'create') return darwin(['bench', 'create', '.']);
      if (sub === 'verify') return darwin(['bench', 'verify', './.metaharness/bench.json']);
      log(`${C.r}usage:${C.x} rupixel bench <create|verify>`); return 2;
    }
    case 'evolve':
      return darwin(['evolve', '.', '--bench', './.metaharness/bench.json',
        '--selection', 'pareto', ...rest]);
    default:
      log(`${C.r}unknown command:${C.x} ${cmd}`); help(); return 2;
  }
}

process.exit(main(process.argv.slice(2)));
