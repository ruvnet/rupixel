//! Request/response types and per-endpoint handler stubs (ADR-264, M3).
//!
//! Replaces upstream FastAPI route functions + Pydantic models. Each handler
//! takes the shared [`AppState`] (pipeline handle + config) plus a parsed
//! request type and returns a typed response. M3 wires these into the `hyper`
//! router in [`crate::http`] and the `pixelrag-core` pipeline.
//!
//! In M3 the request/response structs gain `#[derive(serde::Serialize,
//! serde::Deserialize)]`; for M0 they are plain std-only structs so the public
//! API surface is fixed and callers can already construct/inspect them.

use std::sync::Arc;

use crate::config::ServerConfig;

/// Shared, cheaply-cloneable application state passed to every handler.
///
/// M3: holds an `Arc<pixelrag_core::Pipeline>` (the render→embed→index→search
/// orchestrator) and an `Arc<pixelrag_encoder::Embedder>`. For M0 it carries
/// only the immutable [`ServerConfig`] so the handler signatures are stable.
#[derive(Clone)]
pub struct AppState {
    /// Immutable server configuration, shared across all requests.
    pub config: Arc<ServerConfig>,
    // M3: pub pipeline: Arc<pixelrag_core::Pipeline>,
    // M3: pub embedder: Arc<pixelrag_encoder::Embedder>,
}

impl AppState {
    /// Construct shared state from a validated [`ServerConfig`].
    ///
    /// M3: also opens the index (`pixelrag_core::Pipeline::open`) and loads the
    /// encoder model (`pixelrag_encoder::Embedder::load`).
    pub fn new(config: ServerConfig) -> crate::Result<Self> {
        unimplemented!(
            "M3: open pixelrag_core::Pipeline from config.index_path and load \
             pixelrag_encoder::Embedder from config.model_path; wrap in Arc"
        )
    }
}

// ---------------------------------------------------------------------------
// POST /index — ingest a document into the index.
// ---------------------------------------------------------------------------

/// Body of `POST /index`.
///
/// One of `url` or `image_b64` identifies the source document. M3: `#[serde]`.
#[derive(Debug, Clone, Default)]
pub struct IndexRequest {
    /// Document identifier stored alongside the tiles (returned by `/search`).
    pub doc_id: String,
    /// Source URL to render → embed → index (M2 `pixelrag-render` path).
    pub url: Option<String>,
    /// Inline base64-encoded image to embed → index directly (no render).
    pub image_b64: Option<String>,
    /// Optional free-form metadata persisted with the document's tiles.
    pub metadata: Option<String>,
}

/// Response of `POST /index`.
#[derive(Debug, Clone, Default)]
pub struct IndexResponse {
    /// Echoed document identifier.
    pub doc_id: String,
    /// Number of tiles embedded and inserted into the index.
    pub tiles_indexed: usize,
    /// Wall-clock ingest time in milliseconds (render + embed + insert).
    pub elapsed_ms: u64,
}

/// Handle `POST /index`: render (if `url`) → embed tiles → insert into backend.
///
/// M3: drive `pixelrag_core::Pipeline::index_document`, which renders via
/// `pixelrag-render`, embeds via `pixelrag-encoder`, and inserts into the
/// `ruvector-core::HNSWIndex` (or `ruvector-rairs::IVFIndex`) behind the
/// `AnnIndex` trait. Returns tile count + timing.
pub fn handle_index(_state: &AppState, _req: IndexRequest) -> crate::Result<IndexResponse> {
    unimplemented!(
        "M3: pipeline.index_document(req) -> render(url)?.embed_tiles()?.insert(); \
         map pixelrag_core::Error into ServeError::Pipeline"
    )
}

// ---------------------------------------------------------------------------
// POST /search — retrieve top-k tiles for a query.
// ---------------------------------------------------------------------------

/// Body of `POST /search`. Exactly one of `query_text` / `query_image_b64` set.
#[derive(Debug, Clone, Default)]
pub struct SearchRequest {
    /// Text query to embed and search (cross-modal retrieval).
    pub query_text: Option<String>,
    /// Base64-encoded query image to embed and search.
    pub query_image_b64: Option<String>,
    /// Number of results to return (default applied server-side if `None`).
    pub k: Option<usize>,
    /// Optional allowlist of `doc_id`s to restrict the search (filtered ANN).
    ///
    /// M3: wires to the rabitq-derived allowlist filter (ADR-264 retrieval row).
    pub allowlist: Option<Vec<String>>,
}

/// A single retrieval hit.
#[derive(Debug, Clone, Default)]
pub struct SearchHit {
    /// Owning document identifier.
    pub doc_id: String,
    /// Tile index within the document.
    pub tile_id: u32,
    /// Similarity score (higher is better; backend-dependent metric).
    pub score: f32,
}

/// Response of `POST /search`.
#[derive(Debug, Clone, Default)]
pub struct SearchResponse {
    /// Ranked hits, best first (length ≤ requested `k`).
    pub hits: Vec<SearchHit>,
    /// Vector-search latency in milliseconds (excludes optional rerank).
    pub elapsed_ms: u64,
}

/// Handle `POST /search`: embed query → ANN search → (optional) rerank.
///
/// M3: embed via `pixelrag-encoder`, call `pixelrag_core::Pipeline::search`
/// (which dispatches to `AnnIndex::search` with the optional rabitq allowlist),
/// then apply the optional cross-encoder/LLM rerank hook. Returns ranked hits.
pub fn handle_search(_state: &AppState, _req: SearchRequest) -> crate::Result<SearchResponse> {
    unimplemented!(
        "M3: embed query, pipeline.search(emb, k, allowlist) -> AnnIndex::search; \
         apply optional rerank hook; map errors into ServeError::Pipeline"
    )
}

// ---------------------------------------------------------------------------
// GET /health — liveness/readiness probe.
// ---------------------------------------------------------------------------

/// Response of `GET /health`.
#[derive(Debug, Clone, Default)]
pub struct HealthResponse {
    /// `"ok"` when the index and encoder are loaded and serving.
    pub status: String,
    /// Crate version (`env!("CARGO_PKG_VERSION")`) for build identification.
    pub version: String,
    /// Number of documents currently indexed (readiness signal).
    pub indexed_docs: usize,
}

/// Handle `GET /health`: report liveness + index readiness.
///
/// M3: read `pipeline.len()` for `indexed_docs`; stamp `version` from
/// `CARGO_PKG_VERSION`. Always returns `Ok` once the server is accepting.
pub fn handle_health(_state: &AppState) -> crate::Result<HealthResponse> {
    unimplemented!(
        "M3: return HealthResponse {{ status: \"ok\", version: env!(CARGO_PKG_VERSION), \
         indexed_docs: pipeline.len() }}"
    )
}
