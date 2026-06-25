//! `pixelrag-serve` — HTTP serve tier for the PixelRAG Rust port (ADR-264, M3).
//!
//! This crate exposes the visual-RAG pipeline (`pixelrag-core` + `pixelrag-encoder`)
//! over a REST API. Per the ADR-264 reuse boundary, it replaces the upstream
//! FastAPI + Pydantic server (`StarTrail-org/PixelRAG`) with a Tokio/`hyper`-based
//! server, with an optional `tonic` gRPC surface behind a feature flag.
//!
//! ## Endpoints (see [`handlers`])
//! - `POST /index`  — ingest a document; render → embed → index into the backend.
//! - `POST /search` — retrieve top-`k` tiles for a query (image or text).
//! - `GET  /health` — liveness/readiness probe.
//!
//! ## Module map
//! - [`config`]   — [`ServerConfig`], the bind address + pipeline wiring knobs.
//! - [`http`]     — [`Server`], the `hyper`-based listener + router stub.
//! - [`handlers`] — request/response types + per-endpoint handler stubs.
//!
//! ## M0 status
//! This is a **compiling skeleton only**. All function bodies are
//! `unimplemented!("M1: …")` / `unimplemented!("M3: …")` with doc comments
//! describing intended behavior and the real ruvector crate they will wire to.
//! `[dependencies]` is empty (std-only) so the workspace builds offline; the
//! intended `hyper`/`tokio`/`serde`/`pixelrag-core` deps are documented as
//! comments in `Cargo.toml` and wired in M3.
//!
//! Reference: `docs/adr/ADR-264-pixelrag-rust-port-on-ruvector.md` (§Crate layout,
//! §Milestones → M3).

#![forbid(unsafe_code)]

pub mod config;
pub mod handlers;
pub mod http;

pub use config::ServerConfig;
pub use handlers::{
    HealthResponse, IndexRequest, IndexResponse, SearchHit, SearchRequest, SearchResponse,
};
pub use http::Server;

/// Crate-wide error type for the serve tier.
///
/// M3: this will gain variants wrapping `pixelrag_core::Error`, `hyper::Error`,
/// and `serde_json::Error`. For M0 it is a std-only enum so callers can already
/// pattern-match on the intended failure modes.
#[derive(Debug)]
#[non_exhaustive]
pub enum ServeError {
    /// The request body could not be parsed into the expected type.
    BadRequest(String),
    /// A requested resource (e.g. an index) was not found.
    NotFound(String),
    /// The underlying `pixelrag-core` pipeline returned an error.
    Pipeline(String),
    /// Transport/IO failure (socket bind, accept, read/write).
    Transport(String),
}

impl core::fmt::Display for ServeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ServeError::BadRequest(m) => write!(f, "bad request: {m}"),
            ServeError::NotFound(m) => write!(f, "not found: {m}"),
            ServeError::Pipeline(m) => write!(f, "pipeline error: {m}"),
            ServeError::Transport(m) => write!(f, "transport error: {m}"),
        }
    }
}

impl std::error::Error for ServeError {}

/// Convenience result alias used across the serve tier.
pub type Result<T> = core::result::Result<T, ServeError>;
