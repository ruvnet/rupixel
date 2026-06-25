//! Generic visual-embedding abstraction.
//!
//! Per ADR-264 trait contracts: `pixelrag-core::Embedder` is a **generic trait,
//! not constrained to ruvector**. M1 implements it in `pixelrag-encoder` over
//! ONNX Runtime (`ort`) loading Qwen3-VL-Embedding-2B (or a CLIP ONNX surrogate),
//! with batched inference and an LRU tile cache. The v1 fallback (Python sidecar)
//! also implements this trait, so the pipeline is encoder-agnostic.

use crate::tile::Tile;
use crate::{Embedding, Result};

/// Embeds tiles into dense vectors. Implementations live OUTSIDE this crate
/// (`pixelrag-encoder` ONNX backend in M1, or a Python-sidecar shim in v1).
///
/// `Send + Sync` so the orchestrator can batch-embed across a Tokio thread pool.
pub trait Embedder: Send + Sync {
    /// Output embedding dimensionality (e.g. 768/1024 for ViT-L class encoders).
    /// Must match the `dim` of the [`crate::index`] backend.
    fn dim(&self) -> usize;

    /// Embed one tile.
    ///
    /// **M1**: decode `tile.image`, run a single-tile forward pass. Prefer
    /// [`Embedder::embed_batch`] for throughput.
    fn embed(&self, tile: &Tile) -> Result<Embedding>;

    /// Embed a batch of tiles (the throughput path).
    ///
    /// **M1**: decode + stack into an `ndarray` tensor of `batch_size`
    /// (from [`crate::config::Config`]), run one batched ONNX session call, split
    /// the output rows into one [`Embedding`] per input tile (order preserved).
    /// Consults the LRU cache in `pixelrag-encoder` to skip re-embedding.
    fn embed_batch(&self, tiles: &[Tile]) -> Result<Vec<Embedding>>;
}

/// A query embedder for the search side. In PixelRAG queries are themselves
/// rendered to an image (or a text→image surrogate), so the same encoder is used;
/// this thin trait lets callers pass a pre-rendered query tile or raw image.
pub trait QueryEmbedder: Send + Sync {
    /// Embed a query given its rendered image bytes.
    ///
    /// **M1**: same encoder forward pass as [`Embedder::embed`], producing a
    /// vector comparable in the [`crate::index`] backend's metric space.
    fn embed_query(&self, query_image: &[u8]) -> Result<Embedding>;
}

// ── pixelrag-encoder bridge ──────────────────────────────────────────────────

/// Adapts any `pixelrag_encoder::Embedder` (the canonical, ruvector-independent
/// encoder trait — ONNX in M1, Python sidecar in v1) to this crate's [`Embedder`].
///
/// The bridge owns the conversion from a core [`Tile`] (opaque `image: Vec<u8>`)
/// to a `pixelrag_encoder::Image`. In M1 the renderer is bypassed, so the tile
/// bytes are treated as a single-row `Gray8` buffer — sufficient for the
/// deterministic synthetic embedder used in plumbing validation. M2 replaces this
/// with the real decoded RGB tile from `pixelrag-render`.
pub struct EncoderEmbedder<E: pixelrag_encoder::Embedder> {
    inner: E,
}

impl<E: pixelrag_encoder::Embedder> EncoderEmbedder<E> {
    /// Wrap a `pixelrag_encoder::Embedder` so a [`crate::pipeline::Pipeline`] can
    /// drive it through the core [`Embedder`] contract.
    pub fn new(inner: E) -> Self {
        Self { inner }
    }

    fn to_image(tile: &Tile) -> pixelrag_encoder::Image {
        // M1 bridge: opaque tile bytes → Gray8 1×N image. No real decode; the
        // synthetic embedder hashes these bytes deterministically. M2 supplies a
        // decoded RGB image from the renderer instead.
        let width = tile.image.len() as u32;
        pixelrag_encoder::Image {
            pixels: tile.image.clone(),
            width,
            height: 1,
            format: pixelrag_encoder::PixelFormat::Gray8,
        }
    }
}

impl<E: pixelrag_encoder::Embedder> Embedder for EncoderEmbedder<E> {
    fn dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    fn embed(&self, tile: &Tile) -> Result<Embedding> {
        let img = Self::to_image(tile);
        self.inner
            .embed(&img)
            .map(|e| e.vector)
            .map_err(|e| crate::Error::Embedding(e.to_string()))
    }

    fn embed_batch(&self, tiles: &[Tile]) -> Result<Vec<Embedding>> {
        let imgs: Vec<pixelrag_encoder::Image> = tiles.iter().map(Self::to_image).collect();
        self.inner
            .embed_batch(&imgs)
            .map(|v| v.into_iter().map(|e| e.vector).collect())
            .map_err(|e| crate::Error::Embedding(e.to_string()))
    }
}
