//! # pixelrag-encoder — PixelRAG visual encoder wrapper
//!
//! M0 compiling skeleton for the visual encoder tier of the PixelRAG Rust port
//! (see **ADR-264 — PixelRAG Rust port on ruvector substrate**, §"Honest embedding
//! model strategy" and the `pixelrag-encoder` crate-layout entry).
//!
//! ## Role in the pipeline
//!
//! PixelRAG retrieves over **visual embeddings of document screenshot tiles** rather
//! than parsed text. This crate is the `embed` stage of the upstream
//! `render → embed → index → serve` pipeline: it turns rendered screenshot tiles
//! (`Image`) into fixed-width embedding vectors (`Embedding`) that `pixelrag-core`
//! then indexes via the ruvector ANN substrate (`ruvector-core` HNSW / `ruvector-rairs`
//! IVF-SQ / `ruvector-rabitq`).
//!
//! ## Encoder strategy (ADR-264 §encoder)
//!
//! Backends, selected at runtime via [`EmbedderKind`]:
//! - **M1 plumbing (the deterministic non-semantic backend):**
//!   [`synthetic::SyntheticEmbedder`] — a deterministic, seeded, **non-semantic**
//!   embedder that maps tile bytes → an L2-normalized f32 vector so the
//!   render→embed→cache→index→search pipeline runs WITHOUT the 2B model. The real
//!   Qwen3-VL-Embedding-2B weights + GPU are blocked here, so every metric derived
//!   from this backend MUST be labelled "subset fixture + synthetic embeddings —
//!   plumbing validation, NOT semantic retrieval quality; real recall requires
//!   Qwen3-VL-2B (blocked)". See the [`synthetic`] module honesty note.
//! - **v1 (the runnable real-semantic backend):** [`sidecar::SidecarEmbedder`] — shells
//!   out to the Node sidecar (`all-MiniLM-L6-v2` over transformers.js, pure WASM/CPU)
//!   and marshals tile text + embeddings over a stdin/stdout JSON protocol.
//!
//! All backends implement the crate-local [`Embedder`] trait (generic, NOT constrained
//! to a ruvector trait — per ADR-264 "Embedder → generic trait, not constrained to
//! ruvector"). The [`cache`] module fronts any `Embedder` with an LRU tile-embedding
//! cache to avoid re-encoding identical tiles.

#![forbid(unsafe_code)]

pub mod cache;
pub mod sidecar;
pub mod synthetic;

use std::fmt;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A rendered screenshot tile to be encoded.
///
/// Produced by `pixelrag-render` (M2) or by an upstream renderer; in M0 this is a
/// thin owned wrapper over raw decoded pixels plus geometry. The concrete pixel
/// layout (RGB8 vs RGBA8) is fixed by [`PixelFormat`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Decoded pixel bytes, row-major, `width * height * format.channels()` long.
    pub pixels: Vec<u8>,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
    /// Channel layout of `pixels`.
    pub format: PixelFormat,
}

/// Pixel channel layout for an [`Image`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 3 channels, 8 bits each (R, G, B).
    Rgb8,
    /// 4 channels, 8 bits each (R, G, B, A).
    Rgba8,
    /// 1 channel, 8 bits (luminance) — used by CLIP-style preprocessing variants.
    Gray8,
}

impl PixelFormat {
    /// Number of bytes per pixel for this layout.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// A dense visual embedding vector for a single tile.
///
/// Layout matches what `pixelrag-core` feeds to the ruvector `AnnIndex`: an
/// f32 vector of length [`Embedding::dim`]. Whether it is L2-normalized is recorded
/// in `normalized` (PixelRAG's FAISS index is normalized; the Rust port preserves
/// that contract so cosine == inner-product downstream).
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    /// Embedding components.
    pub vector: Vec<f32>,
    /// Whether `vector` has been L2-normalized.
    pub normalized: bool,
}

impl Embedding {
    /// Dimensionality of this embedding (`vector.len()`).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Stable content key for a tile, used by [`cache::EmbeddingCache`] as the LRU key.
///
/// In M1 this is a hash (e.g. blake3/xxhash) of the decoded pixels + preprocessing
/// params, so identical tiles produced by different documents share a cache slot.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey(pub [u8; 32]);

// ---------------------------------------------------------------------------
// Embedder trait + selector
// ---------------------------------------------------------------------------

/// The visual encoder contract for PixelRAG.
///
/// Deliberately generic and **not** tied to any ruvector trait (ADR-264). Any backend
/// — ONNX, Python sidecar, or a future Candle path — implements this so `pixelrag-core`
/// can hold a `Box<dyn Embedder>` without knowing the concrete model.
pub trait Embedder: Send + Sync {
    /// The fixed embedding dimensionality this encoder produces (e.g. 1024 for
    /// Qwen3-VL-Embedding-2B, 768 for CLIP ViT-L surrogate). Used by `pixelrag-core`
    /// to size the ANN index before any tile is seen.
    fn embedding_dim(&self) -> usize;

    /// Encode a single tile into an [`Embedding`].
    ///
    /// Preprocess → run the backend → optionally L2-normalize. Convenience wrapper
    /// over [`Embedder::embed_batch`].
    ///
    /// # Errors
    /// Returns [`EncoderError`] if preprocessing or inference fails.
    fn embed(&self, tile: &Image) -> Result<Embedding, EncoderError> {
        let mut out = self.embed_batch(std::slice::from_ref(tile))?;
        out.pop().ok_or(EncoderError::EmptyBatch)
    }

    /// Batched encode — the throughput-critical path.
    ///
    /// M1: stack tiles into one input tensor and run a single forward pass
    /// (ONNX) or one sidecar round-trip. Order of the returned vector matches
    /// the order of `tiles`.
    ///
    /// # Errors
    /// Returns [`EncoderError`] if preprocessing or inference fails.
    fn embed_batch(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError>;

    /// Which backend this is, for logging / config round-tripping.
    fn kind(&self) -> EmbedderKind;
}

/// Blanket impl so a `Box<dyn Embedder>` is itself an [`Embedder`].
///
/// Lets callers (e.g. the CLI bench) select a concrete backend at runtime
/// (`SidecarEmbedder` vs `SyntheticEmbedder`) behind one boxed type and feed it to a
/// generic consumer like `pixelrag_core::EncoderEmbedder<E>` without duplicating the
/// pipeline body per backend.
impl Embedder for Box<dyn Embedder> {
    fn embedding_dim(&self) -> usize {
        (**self).embedding_dim()
    }

    fn embed_batch(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        (**self).embed_batch(tiles)
    }

    fn kind(&self) -> EmbedderKind {
        (**self).kind()
    }
}

/// Identifies which encoder backend is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmbedderKind {
    /// [`synthetic::SyntheticEmbedder`] — deterministic, non-semantic M1 plumbing
    /// backend. Runnable while Qwen3-VL-2B is blocked.
    Synthetic,
    /// External-encoder sidecar over IPC, ADR-264 v1. The runnable real-semantic
    /// backend is [`sidecar::SidecarEmbedder`] (Node + `all-MiniLM-L6-v2`).
    Sidecar,
}

impl fmt::Display for EmbedderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EmbedderKind::Synthetic => "synthetic",
            EmbedderKind::Sidecar => "sidecar",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised by any [`Embedder`] backend or the cache layer.
///
/// M1 will likely back this with `thiserror`; in M0 it is a hand-written enum so the
/// crate stays std-only.
#[derive(Debug)]
pub enum EncoderError {
    /// The ONNX model file or runtime could not be loaded (path, version, opset).
    ModelLoad(String),
    /// Tile preprocessing failed (bad dimensions, unsupported [`PixelFormat`], resize error).
    Preprocess(String),
    /// The backend forward pass / inference call failed.
    Inference(String),
    /// The Python sidecar process or HTTP endpoint failed (spawn, transport, protocol).
    Sidecar(String),
    /// A batch operation was asked to return an embedding but produced none.
    EmptyBatch,
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncoderError::ModelLoad(m) => write!(f, "model load failed: {m}"),
            EncoderError::Preprocess(m) => write!(f, "tile preprocessing failed: {m}"),
            EncoderError::Inference(m) => write!(f, "inference failed: {m}"),
            EncoderError::Sidecar(m) => write!(f, "encoder sidecar failed: {m}"),
            EncoderError::EmptyBatch => write!(f, "batch produced no embeddings"),
        }
    }
}

impl std::error::Error for EncoderError {}

// Re-export the runnable real-semantic backend: `use pixelrag_encoder::SidecarEmbedder`
// (Node + all-MiniLM-L6-v2 over transformers.js).
pub use sidecar::SidecarEmbedder;
// Re-export the M1 plumbing backend: `use pixelrag_encoder::SyntheticEmbedder`.
pub use synthetic::SyntheticEmbedder;
