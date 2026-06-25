//! Retrieval, filtering, and rerank hooks.
//!
//! Sits on top of [`crate::index::AnnIndex`] to provide the user-facing retrieval
//! surface: vector search, optional allowlist pre-filtering (reusing the
//! `ruvector-rabitq` allowlist contract), and an *optional* rerank stage. Per
//! ADR-264 the reranker is pluggable (cross-encoder / CLIP-rerank / LLM judge,
//! e.g. a Claude API call) and lives behind a trait so the core never hard-depends
//! on any LLM. Reranking is off by default (the default M1 harness ships
//! `no rerank`).

use std::collections::HashMap;

use crate::index::AnnIndex;
use crate::tile::TileMetadata;
use crate::{Embedding, Result, SearchResult};

/// Parameters for a single retrieval request.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// `k` — number of nearest tiles to retrieve before reranking.
    pub k: usize,
    /// Optional allowlist of tile ids; `None` ⇒ search the whole index.
    pub allowlist: Option<Vec<usize>>,
    /// Whether to run the optional rerank stage over the retrieved candidates.
    pub rerank: bool,
}

impl Default for SearchRequest {
    fn default() -> Self {
        // Mirrors the default M1 harness: k=10, no filter, no rerank.
        SearchRequest { k: 10, allowlist: None, rerank: false }
    }
}

/// A retrieval hit enriched with its source-document metadata, returned to
/// callers (the raw [`SearchResult`] only carries id + score).
#[derive(Debug, Clone)]
pub struct RetrievedTile {
    /// The underlying id + score from the ANN backend.
    pub result: SearchResult,
    /// Provenance/localization metadata for the hit tile.
    pub metadata: TileMetadata,
}

/// Optional second-stage reranker over retrieved candidates.
///
/// Implementations are pluggable and external (cross-encoder, CLIP-rerank, or an
/// LLM judge such as a Claude API call). The core only sees this trait, keeping
/// the LLM dependency out of `pixelrag-core`.
pub trait Reranker: Send + Sync {
    /// Reorder `candidates` for the given query embedding, returning a new
    /// ordering (possibly truncated).
    ///
    /// **M3**: score each candidate against the query (cross-encoder forward pass
    /// or LLM judgment) and sort descending by rerank score.
    fn rerank(
        &self,
        query: &[f32],
        candidates: Vec<RetrievedTile>,
    ) -> Result<Vec<RetrievedTile>>;
}

/// The retrieval engine: an [`AnnIndex`] plus an optional [`Reranker`] and the
/// id→metadata map needed to enrich raw hits into [`RetrievedTile`]s.
///
/// M0 holds the boxed index behind the trait; M1 owns the metadata map and wires
/// the search path. M3 attaches an optional reranker.
pub struct Searcher {
    index: Box<dyn AnnIndex>,
    reranker: Option<Box<dyn Reranker>>,
    /// id → metadata, used to enrich raw [`SearchResult`]s into [`RetrievedTile`]s.
    /// Hits with no entry are dropped (an unknown id cannot be localized).
    metadata: HashMap<usize, TileMetadata>,
}

impl Searcher {
    /// Build a searcher around an index, with no reranker (default M1 harness).
    pub fn new(index: Box<dyn AnnIndex>) -> Self {
        Searcher { index, reranker: None, metadata: HashMap::new() }
    }

    /// Attach the id→metadata map used to enrich hits (populated at index time).
    pub fn with_metadata(mut self, metadata: HashMap<usize, TileMetadata>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach an optional [`Reranker`] (M3).
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Read access to the wrapped index (len/dim/memory inspection).
    pub fn index(&self) -> &dyn AnnIndex {
        self.index.as_ref()
    }

    /// Run a full retrieval: ANN search (filtered if `req.allowlist` is set),
    /// metadata enrichment, then optional rerank.
    ///
    /// **M1**: dispatch to [`AnnIndex::search`] or [`AnnIndex::search_filtered`]
    /// per `req`, map each [`SearchResult`] to its [`TileMetadata`].
    /// **M3**: if `req.rerank` and a [`Reranker`] is attached, run it over the
    /// candidates before returning.
    pub fn search(&self, query: &Embedding, req: &SearchRequest) -> Result<Vec<RetrievedTile>> {
        let raw: Vec<SearchResult> = match &req.allowlist {
            Some(allow) => self.index.search_filtered(query, req.k, allow)?,
            None => self.index.search(query, req.k)?,
        };

        // Enrich: drop hits with no known metadata (cannot be localized back to a
        // source region). The default M1 harness ships metadata for every indexed
        // tile, so this only filters genuinely unknown ids.
        let mut hits: Vec<RetrievedTile> = raw
            .into_iter()
            .filter_map(|result| {
                self.metadata
                    .get(&result.id)
                    .cloned()
                    .map(|metadata| RetrievedTile { result, metadata })
            })
            .collect();

        // M3: optional rerank pass. Off by default; a no-op when no reranker is
        // attached even if `req.rerank` is set.
        if req.rerank {
            if let Some(reranker) = &self.reranker {
                hits = reranker.rerank(query, hits)?;
            }
        }
        Ok(hits)
    }
}
