//! Top-level orchestrator: render → embed → index → search.
//!
//! [`Pipeline`] ties the stages together per ADR-264:
//!
//! - **render** (M2): `pixelrag-render` produces page bitmaps from a doc/URL/PDF.
//!   In M1 the renderer is bypassed — pre-rendered pages are supplied directly.
//! - **embed**: [`crate::embedding::Embedder`] (from `pixelrag-encoder`).
//! - **index**: [`crate::index::AnnIndex`] over a ruvector backend.
//! - **search**: [`crate::search::Searcher`].
//!
//! The pipeline is generic over the [`crate::embedding::Embedder`] so M1 (ONNX)
//! and v1 (Python sidecar) implementations are interchangeable. It is driven by a
//! [`crate::config::Config`] whose darwin augmentation is optional (ADR-256).

use std::collections::HashMap;

use crate::config::Config;
use crate::embedding::Embedder;
use crate::index::AnnIndex;
use crate::search::{RetrievedTile, SearchRequest, Searcher};
use crate::tile::{Tile, TileMetadata, Tiler};
use crate::{Embedding, Error, Result, SearchResult};

/// Outcome of indexing one document: how many tiles were embedded + added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IngestReport {
    /// Tiles produced by the tiler for this document.
    pub tiles: usize,
    /// Tiles successfully embedded and inserted into the index.
    pub indexed: usize,
}

/// The render→embed→index→search orchestrator.
///
/// Owns the configured [`Tiler`], an [`Embedder`] `E`, the boxed [`AnnIndex`]
/// backend, and the [`Searcher`] view over it.
pub struct Pipeline<E: Embedder> {
    config: Config,
    tiler: Tiler,
    embedder: E,
    index: Box<dyn AnnIndex>,
    /// id → metadata for every indexed tile, used to enrich search hits.
    metadata: HashMap<usize, TileMetadata>,
    /// Monotonic external tile-id counter (insertion order).
    next_id: usize,
}

impl<E: Embedder> Pipeline<E> {
    /// Assemble a pipeline from its parts.
    ///
    /// **M1**: validates that `embedder.dim() == index.dim()` and that the
    /// `config` is valid ([`Config::validate`]) before returning.
    pub fn new(config: Config, tiler: Tiler, embedder: E, index: Box<dyn AnnIndex>) -> Result<Self> {
        config.validate()?;
        if embedder.dim() != index.dim() {
            return Err(Error::Pipeline(format!(
                "embedder dim {} != index dim {}",
                embedder.dim(),
                index.dim()
            )));
        }
        Ok(Self {
            config,
            tiler,
            embedder,
            index,
            metadata: HashMap::new(),
            next_id: 0,
        })
    }

    /// Read-only access to the active configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Honest byte footprint reported by the live index backend.
    ///
    /// The HNSW path is `n*dim*4 + n*size_of::<usize>()`; the IVF path adds its
    /// k-means centroid table. Callers (e.g. the bench memory metric) should use
    /// this rather than reconstructing a backend-specific formula, so footprints
    /// stay correct as backends differ.
    pub fn index_memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }

    /// Ingest one already-rendered document (M1 path; render is bypassed).
    ///
    /// **M1**: tile `pages` via [`Tiler::tile_document`], batch-embed via
    /// [`Embedder::embed_batch`] at `config.batch_size`, assign ids, and
    /// [`AnnIndex::add`] each embedding. Returns an [`IngestReport`].
    pub fn ingest_rendered(&mut self, doc_id: &str, pages: &[Vec<u8>]) -> Result<IngestReport> {
        let tiles = self.tiler.tile_document(doc_id, pages)?;
        self.index_tiles(&tiles)
    }

    /// Embed pre-tiled input directly (used by tests/benchmarks that supply
    /// their own tiles) and add to the index.
    ///
    /// **M1**: [`Embedder::embed_batch`] over `tiles` then [`AnnIndex::add`].
    pub fn index_tiles(&mut self, tiles: &[Tile]) -> Result<IngestReport> {
        if tiles.is_empty() {
            return Ok(IngestReport { tiles: 0, indexed: 0 });
        }
        let mut indexed = 0usize;
        // Honor config.batch_size: embed in fixed-size chunks (the throughput
        // path), preserving per-tile order so ids line up with metadata.
        for chunk in tiles.chunks(self.config.batch_size.max(1)) {
            let embeddings = self.embedder.embed_batch(chunk)?;
            if embeddings.len() != chunk.len() {
                return Err(Error::Embedding(format!(
                    "embed_batch returned {} vectors for {} tiles",
                    embeddings.len(),
                    chunk.len()
                )));
            }
            for (tile, embedding) in chunk.iter().zip(embeddings.into_iter()) {
                let id = self.next_id;
                self.next_id += 1;
                self.index.add(id, embedding)?;
                self.metadata.insert(id, tile.metadata.clone());
                indexed += 1;
            }
        }
        // Commit the batch. HNSW already inserted incrementally (no-op); IVF-style
        // backends buffer during `add` and do their train-then-add build here.
        // finalize() is idempotent, so calling it per ingest leaves the index
        // reflecting every tile after the final call.
        self.index.finalize()?;
        Ok(IngestReport { tiles: tiles.len(), indexed })
    }

    /// Retrieve over the index for an already-embedded query.
    ///
    /// **M1**: ANN search (filtered when `req.allowlist` is set) + metadata
    /// enrichment, run directly against the owned index.
    pub fn search(&self, query: &Embedding, req: &SearchRequest) -> Result<Vec<RetrievedTile>> {
        let raw: Vec<SearchResult> = match &req.allowlist {
            Some(allow) => self.index.search_filtered(query, req.k, allow)?,
            None => self.index.search(query, req.k)?,
        };
        Ok(raw
            .into_iter()
            .filter_map(|result| {
                self.metadata
                    .get(&result.id)
                    .cloned()
                    .map(|metadata| RetrievedTile { result, metadata })
            })
            .collect())
    }

    /// Build a standalone [`Searcher`] snapshot over a freshly-constructed index
    /// sharing this pipeline's metadata map.
    ///
    /// The [`Searcher`] owns its `Box<dyn AnnIndex>`, so this returns a searcher
    /// over an empty backend pre-loaded with the current id→metadata map; use
    /// [`Pipeline::search`] for retrieval against the live index. A borrowing
    /// `Searcher` view is deferred to a later milestone (would require a lifetime
    /// on `Searcher`).
    pub fn searcher(&self) -> Result<Searcher> {
        let index = crate::index::build_index(self.config.index_backend, self.index.dim())?;
        Ok(Searcher::new(index).with_metadata(self.metadata.clone()))
    }
}
