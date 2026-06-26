//! Real Node sidecar embedder — `all-MiniLM-L6-v2` over transformers.js (ADR-264 v1).
//!
//! # This is the ONE genuinely-semantic backend that runs in this environment.
//!
//! Unlike [`crate::synthetic::SyntheticEmbedder`] (deterministic but **non-semantic**),
//! this backend produces
//! **real semantic embeddings**: it shells out to the verified Node sidecar
//! (`crates/pixelrag-cli/sidecar/embed_sidecar.mjs`), which runs
//! `Xenova/all-MiniLM-L6-v2` (sentence-transformers) via transformers.js — pure WASM /
//! CPU, no GPU, no native onnxruntime. Outputs are mean-pooled and L2-normalized, so
//! cosine == inner-product downstream (matching PixelRAG's normalized index contract).
//!
//! Verified semantics: `cos(cat, feline) ≈ 0.55`, `cos(cat, finance) ≈ 0.00`.
//!
//! ## Protocol (one round-trip per `embed_batch`)
//!
//! The Rust side spawns `node <sidecar.mjs>`, writes a single JSON object
//! `{"texts": ["..", ".."]}` to its stdin (then closes stdin → EOF), and reads
//! `{"model":"all-MiniLM-L6-v2","dim":384,"vectors":[[..384 f32..], ..]}` from stdout.
//! Vector order matches input `texts` order.
//!
//! ## Tile → text bridge
//!
//! In the M1 plumbing path, `pixelrag-core`'s `EncoderEmbedder` bridges each `Tile` to
//! a `Gray8` [`Image`] whose `pixels` are the tile's **UTF-8 text bytes** (not real
//! pixels). This backend therefore decodes each [`Image`] back to text via
//! [`String::from_utf8_lossy`] and embeds the text — which is exactly the real
//! all-MiniLM semantic signal we want for the bench.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{Embedder, EmbedderKind, Embedding, EncoderError, Image};

/// The embedding width emitted by `all-MiniLM-L6-v2`.
pub const MINILM_DIM: usize = 384;

/// Real Node sidecar embedder backed by `all-MiniLM-L6-v2` (transformers.js, WASM/CPU).
///
/// See the module docs. Each [`Embedder::embed_batch`] is one `node <sidecar>`
/// round-trip; tile bytes are interpreted as UTF-8 text and embedded semantically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SidecarEmbedder {
    /// Path to `embed_sidecar.mjs`. The CLI passes its own
    /// `<CARGO_MANIFEST_DIR>/sidecar/embed_sidecar.mjs`; the encoder crate can't assume
    /// the CLI's layout, so the path is injected via [`SidecarEmbedder::new`].
    sidecar_path: PathBuf,
    /// The `node` executable to spawn (defaults to `node` on `PATH`).
    node_bin: String,
    /// Expected/declared embedding width (`all-MiniLM-L6-v2` → 384). Used to size the
    /// ANN index up front and to validate the sidecar's response.
    dim: usize,
}

impl SidecarEmbedder {
    /// Construct a sidecar embedder that spawns `node <sidecar_path>`.
    ///
    /// `dim` is fixed at [`MINILM_DIM`] (384). The `node` binary is taken from the
    /// `PIXELRAG_NODE` env var if set, else `node` on `PATH`. The sidecar script path
    /// itself may be overridden by the `PIXELRAG_SIDECAR` env var (else the provided
    /// `sidecar_path` is used — the CLI passes its manifest-relative path).
    #[must_use]
    pub fn new(sidecar_path: impl Into<PathBuf>) -> Self {
        let sidecar_path = match std::env::var_os("PIXELRAG_SIDECAR") {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => sidecar_path.into(),
        };
        let node_bin = std::env::var("PIXELRAG_NODE").unwrap_or_else(|_| "node".to_string());
        Self { sidecar_path, node_bin, dim: MINILM_DIM }
    }

    /// The resolved sidecar script path this embedder will spawn.
    #[must_use]
    pub fn sidecar_path(&self) -> &Path {
        &self.sidecar_path
    }

    /// Decode the `pixels` of each [`Image`] to text (the tile bytes are UTF-8 text in
    /// the M1 bridge) and run one sidecar round-trip, returning vectors in input order.
    fn run_sidecar(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        let texts: Vec<String> = tiles
            .iter()
            .map(|img| String::from_utf8_lossy(&img.pixels).into_owned())
            .collect();

        let request = serde_json::json!({ "texts": texts });
        let payload = serde_json::to_vec(&request)
            .map_err(|e| EncoderError::Sidecar(format!("serialize request: {e}")))?;

        let mut child = Command::new(&self.node_bin)
            .arg(&self.sidecar_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                EncoderError::Sidecar(format!(
                    "spawn `{} {}` failed: {e} (is Node installed and on PATH? set PIXELRAG_NODE to override)",
                    self.node_bin,
                    self.sidecar_path.display()
                ))
            })?;

        // Write the request, then drop stdin so the sidecar sees EOF.
        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| EncoderError::Sidecar("sidecar stdin unavailable".into()))?;
            stdin
                .write_all(&payload)
                .map_err(|e| EncoderError::Sidecar(format!("write request to sidecar: {e}")))?;
        }
        child.stdin = None; // close stdin → EOF for the sidecar's readStdin()

        let output = child
            .wait_with_output()
            .map_err(|e| EncoderError::Sidecar(format!("wait for sidecar: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(EncoderError::Sidecar(format!(
                "sidecar exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // The sidecar prints one JSON line; tolerate trailing/leading whitespace.
        let line = stdout.trim();
        if line.is_empty() {
            return Err(EncoderError::Sidecar(
                "sidecar produced empty stdout (no JSON response)".into(),
            ));
        }
        let resp: SidecarResponse = serde_json::from_str(line).map_err(|e| {
            EncoderError::Sidecar(format!("parse sidecar response JSON: {e}"))
        })?;

        if resp.dim != self.dim {
            return Err(EncoderError::Sidecar(format!(
                "sidecar dim mismatch: expected {}, got {} (model={})",
                self.dim, resp.dim, resp.model
            )));
        }
        if resp.vectors.len() != tiles.len() {
            return Err(EncoderError::Sidecar(format!(
                "sidecar returned {} vectors for {} tiles",
                resp.vectors.len(),
                tiles.len()
            )));
        }

        let mut out = Vec::with_capacity(resp.vectors.len());
        for (i, vector) in resp.vectors.into_iter().enumerate() {
            if vector.len() != self.dim {
                return Err(EncoderError::Sidecar(format!(
                    "vector {i} has width {}, expected {}",
                    vector.len(),
                    self.dim
                )));
            }
            out.push(Embedding { vector, normalized: true });
        }
        Ok(out)
    }
}

/// Deserialized sidecar stdout: `{"model","dim","vectors"}`.
#[derive(serde::Deserialize)]
struct SidecarResponse {
    model: String,
    dim: usize,
    vectors: Vec<Vec<f32>>,
}

impl Embedder for SidecarEmbedder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        if tiles.is_empty() {
            return Ok(Vec::new());
        }
        self.run_sidecar(tiles)
    }

    fn kind(&self) -> EmbedderKind {
        EmbedderKind::Sidecar
    }
}
