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
//! - **M1 plumbing (the only *runnable* backend in this environment):**
//!   [`synthetic::SyntheticEmbedder`] — a deterministic, seeded, **non-semantic**
//!   embedder that maps tile bytes → an L2-normalized f32 vector so the
//!   render→embed→cache→index→search pipeline runs WITHOUT the 2B model. The real
//!   Qwen3-VL-Embedding-2B weights + GPU are blocked here, so every metric derived
//!   from this backend MUST be labelled "subset fixture + synthetic embeddings —
//!   plumbing validation, NOT semantic retrieval quality; real recall requires
//!   Qwen3-VL-2B (blocked)". See the [`synthetic`] module honesty note.
//! - **v2 (recommended real path, stub):** [`OnnxEmbedder`] — ONNX Runtime (`ort`)
//!   loads a Qwen3-VL-Embedding-2B export or a CLIP ViT-L ONNX surrogate ([`model`]).
//!   Stays `unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")` — the `ort`
//!   dep is intentionally NOT added.
//! - **v1 (conservative fallback, stub):** [`SidecarEmbedder`] — shells out to an
//!   external Python encoder process / HTTP service and marshals tiles + embeddings
//!   over IPC.
//! - **v3 (post-M3, aspirational):** a pure-Rust Candle/burn path wired to
//!   `ruvector-cnn` kernels; not represented as a concrete type in M0.
//!
//! All backends implement the crate-local [`Embedder`] trait (generic, NOT constrained
//! to a ruvector trait — per ADR-264 "Embedder → generic trait, not constrained to
//! ruvector"). The [`cache`] module fronts any `Embedder` with an LRU tile-embedding
//! cache to avoid re-encoding identical tiles.
//!
//! ## M0 status
//!
//! Every backend method body is `unimplemented!("M1: …")`. Types, trait signatures,
//! and error variants are real and reflect the ADR; only behaviour is deferred to M1.

#![forbid(unsafe_code)]

pub mod cache;
pub mod model;
pub mod onnx;
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
    /// M1: preprocess (`model::Preprocessor`) → run the backend → optionally
    /// L2-normalize. Convenience wrapper over [`Embedder::embed_batch`].
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

/// Identifies which encoder backend is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmbedderKind {
    /// [`synthetic::SyntheticEmbedder`] — deterministic, non-semantic M1 plumbing
    /// backend. The only runnable backend while Qwen3-VL-2B is blocked.
    Synthetic,
    /// [`OnnxEmbedder`] — ONNX Runtime (`ort`), ADR-264 v2 (real path, stub).
    Onnx,
    /// [`SidecarEmbedder`] — external Python encoder, ADR-264 v1 fallback (stub).
    Sidecar,
}

impl fmt::Display for EmbedderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EmbedderKind::Synthetic => "synthetic",
            EmbedderKind::Onnx => "onnx",
            EmbedderKind::Sidecar => "sidecar",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Sidecar fallback backend (ADR-264 v1)
// ---------------------------------------------------------------------------

/// External-Python-encoder fallback ([`EmbedderKind::Sidecar`], ADR-264 v1).
///
/// Used when ONNX integration is unavailable or blocked: Rust serializes tiles to
/// the sidecar (subprocess stdin / a local HTTP endpoint), the Python `pixelrag-encoder-py`
/// service runs the real Qwen3-VL weights, and embeddings are marshaled back. Slower
/// (IPC latency) but reuses upstream model weights with zero new ONNX work.
#[derive(Clone, Debug)]
pub struct SidecarEmbedder {
    /// How to reach the encoder (subprocess command or HTTP URL).
    pub transport: SidecarTransport,
    /// Embedding dim the sidecar model emits (must match the Python encoder).
    pub embedding_dim: usize,
}

/// Transport used by [`SidecarEmbedder`] to reach the external encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SidecarTransport {
    /// Spawn a subprocess; tiles/embeddings flow over a stdin/stdout JSON protocol.
    /// `argv[0]` is the program, the rest are fixed args (e.g. `["python", "-m", "pixelrag_encoder_py"]`).
    Subprocess(Vec<String>),
    /// POST tiles to a long-running local HTTP encoder service at this base URL.
    Http(String),
}

impl SidecarEmbedder {
    /// Construct a sidecar embedder over the given transport, emitting `embedding_dim`-wide vectors.
    #[must_use]
    pub fn new(transport: SidecarTransport, embedding_dim: usize) -> Self {
        Self { transport, embedding_dim }
    }
}

impl Embedder for SidecarEmbedder {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn embed_batch(&self, _tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        // M1: serialize tiles → write to SidecarTransport (Subprocess via
        // std::process::Command stdin, or Http via the M1 HTTP client) → read back
        // JSON embeddings → deserialize into Vec<Embedding>. Reuses upstream Python
        // Qwen3-VL weights; no new ONNX integration. ADR-264 v1.
        unimplemented!("M1: marshal tiles to the Python encoder sidecar and read embeddings back")
    }

    fn kind(&self) -> EmbedderKind {
        EmbedderKind::Sidecar
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

// Re-export the primary backend so callers can `use pixelrag_encoder::OnnxEmbedder`.
pub use onnx::OnnxEmbedder;
// Re-export the M1 plumbing backend: `use pixelrag_encoder::SyntheticEmbedder`.
pub use synthetic::SyntheticEmbedder;
