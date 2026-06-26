//! # pixelrag-core — PixelRAG core orchestrator on the ruvector substrate
//!
//! Implements **ADR-264** ("PixelRAG Rust port on ruvector substrate"). This is
//! the orchestration crate that ties together the visual-RAG pipeline:
//!
//! ```text
//!   document ──render──▶ tiles ──embed──▶ embeddings ──index──▶ ANN ──search──▶ hits
//! ```
//!
//! Per the ADR reuse boundary, this crate is **not** a new vector DB. It composes
//! existing ruvector primitives:
//!
//! - [`index`] wraps `ruvector-core` (HNSW, M1 primary) / `ruvector-rairs`
//!   (IVF-Flat / IVF-SQ) behind a single [`index::AnnIndex`] adaptor.
//! - [`embedding`] defines a generic [`embedding::Embedder`] trait (NOT constrained
//!   to ruvector) wired to `pixelrag-encoder` in M1.
//! - [`tile`] turns a rendered document into tiles with bounds + metadata.
//! - [`search`] adds filtering (allowlist, reusing `ruvector-rabitq`) and rerank hooks.
//! - [`pipeline`] is the top-level orchestrator.
//! - [`config`] holds the runtime [`config::Config`] with a **removable** darwin
//!   augmentation path (ADR-256): the binary is fully usable when darwin is absent.
//!
#![forbid(unsafe_code)]

pub mod config;
pub mod embedding;
pub mod index;
pub mod pipeline;
pub mod search;
pub mod tile;

// ── crate-wide error / result ────────────────────────────────────────────────

/// Crate-wide error type for the PixelRAG core orchestrator.
///
/// M0 keeps this std-only (no `thiserror`). M1 may add `#[from]` conversions for
/// the underlying ruvector crate errors and `ort`/IO errors once those deps land.
#[derive(Debug)]
pub enum Error {
    /// A pipeline stage (render/embed/index/search) failed with a message.
    Pipeline(String),
    /// Tiling a document failed.
    Tile(String),
    /// The embedding backend failed (model load, inference, batching).
    Embedding(String),
    /// The ANN index backend failed (add/search/build/persist).
    Index(String),
    /// Configuration was invalid or a darwin genome could not be parsed.
    Config(String),
    /// An underlying I/O error (model files, *.pixelrag persistence, fixtures).
    Io(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Pipeline(m) => write!(f, "pixelrag pipeline error: {m}"),
            Error::Tile(m) => write!(f, "pixelrag tile error: {m}"),
            Error::Embedding(m) => write!(f, "pixelrag embedding error: {m}"),
            Error::Index(m) => write!(f, "pixelrag index error: {m}"),
            Error::Config(m) => write!(f, "pixelrag config error: {m}"),
            Error::Io(m) => write!(f, "pixelrag io error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

// ── shared lightweight types ─────────────────────────────────────────────────

/// A dense visual embedding for one tile.
///
/// M0 holds an owned `Vec<f32>`. M1 wires this to the `pixelrag-encoder` output;
/// it mirrors the vector contract consumed by `ruvector_rabitq::AnnIndex::add`.
pub type Embedding = Vec<f32>;

/// A single retrieval hit: the tile id and its score (squared-L2 distance, lower
/// is closer — mirrors `ruvector_rabitq::SearchResult` so the adaptor in
/// [`index`] is a zero-copy passthrough in M1).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// External tile id (assigned at index time; maps back to [`tile::Tile`]).
    pub id: usize,
    /// Estimated or exact squared-L2 distance to the query.
    pub score: f32,
}

#[cfg(test)]
mod plumbing_tests {
    //! End-to-end plumbing validation: tile → (synthetic) embed → HNSW index →
    //! search, with an allowlist filter.
    //!
    //! HONESTY: these use a DETERMINISTIC SYNTHETIC embedder on a tiny in-test
    //! fixture — plumbing validation, NOT semantic retrieval quality. Real recall
    //! requires Qwen3-VL-Embedding-2B (weights/GPU blocked in this environment).
    //! Determinism (fixed hashing, no RNG) makes the wiring reproducible.

    use crate::config::Config;
    use crate::embedding::EncoderEmbedder;
    use crate::index::build_index;
    use crate::pipeline::Pipeline;
    use crate::search::SearchRequest;
    use pixelrag_encoder::{Embedder as EncEmbedder, EmbedderKind, Image};

    const DIM: usize = 16;

    /// Deterministic synthetic embedder (NOT a real visual encoder). Maps tile
    /// bytes to a fixed unit vector via a stable byte hash — reproducible across
    /// runs with zero RNG. Used purely to exercise the pipeline plumbing.
    struct SyntheticEmbedder;

    impl SyntheticEmbedder {
        fn embed_bytes(bytes: &[u8]) -> crate::Embedding {
            let mut v = vec![0f32; DIM];
            for (i, b) in bytes.iter().enumerate() {
                v[i % DIM] += (*b as f32) + (i as f32) * 0.001;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut v {
                *x /= norm;
            }
            v
        }
    }

    impl EncEmbedder for SyntheticEmbedder {
        fn embedding_dim(&self) -> usize {
            DIM
        }
        fn embed_batch(
            &self,
            tiles: &[Image],
        ) -> Result<Vec<pixelrag_encoder::Embedding>, pixelrag_encoder::EncoderError> {
            Ok(tiles
                .iter()
                .map(|t| pixelrag_encoder::Embedding {
                    vector: Self::embed_bytes(&t.pixels),
                    normalized: true,
                })
                .collect())
        }
        fn kind(&self) -> EmbedderKind {
            EmbedderKind::Synthetic
        }
    }

    #[test]
    fn end_to_end_plumbing_synthetic() {
        let config = Config::default();
        let tiler = crate::tile::Tiler::default();
        let embedder = EncoderEmbedder::new(SyntheticEmbedder);
        let index = build_index(config.index_backend, DIM).unwrap();
        let mut pipeline = Pipeline::new(config, tiler, embedder, index).unwrap();

        // Two distinct text "documents" (synthetic embeddings, not semantic).
        let r1 = pipeline.ingest_rendered("doc-a", &[b"alpha beta gamma".to_vec()]).unwrap();
        let r2 = pipeline.ingest_rendered("doc-b", &[b"delta epsilon zeta".to_vec()]).unwrap();
        assert_eq!(r1.indexed, r1.tiles);
        assert_eq!(r2.indexed, r2.tiles);

        // Query equal to doc-a's tile must rank doc-a first (plumbing sanity only).
        let q = SyntheticEmbedder::embed_bytes(b"alpha beta gamma");
        let hits = pipeline.search(&q, &SearchRequest { k: 2, allowlist: None, rerank: false }).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].metadata.doc_id, "doc-a");

        // Allowlist filter: restrict to doc-b's id (id 1, inserted second).
        let filtered = pipeline
            .search(&q, &SearchRequest { k: 2, allowlist: Some(vec![1]), rerank: false })
            .unwrap();
        assert!(filtered.iter().all(|h| h.metadata.doc_id == "doc-b"));
    }
}
