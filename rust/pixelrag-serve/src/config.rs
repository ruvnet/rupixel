//! Server configuration for the PixelRAG serve tier (ADR-264, M3).
//!
//! Replaces upstream FastAPI settings/Pydantic `BaseSettings`. Carries the bind
//! address plus the knobs needed to wire the `pixelrag-core` pipeline (index
//! path, model path, batch size). Mirrors the `Config`-from-darwin pattern in
//! ADR-264 §MetaHarness: a darwin-generated JSON is **optional** and never
//! controls the runtime — defaults must always yield a usable server.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

/// Configuration for the HTTP server and its backing pipeline.
///
/// Construct via [`ServerConfig::default`] for the hard-coded M3 defaults, or
/// [`ServerConfig::from_env`] to layer environment overrides on top. A
/// darwin-tuned variant ([`ServerConfig::from_darwin_json`]) is optional and
/// falls back to defaults on any error (ADR-264 removability constraint).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address the HTTP listener binds to (default `127.0.0.1:8788`).
    pub bind: SocketAddr,
    /// Filesystem path to the persisted `*.pixelrag` index loaded at startup.
    ///
    /// `None` starts the server with an empty in-memory index; `POST /index`
    /// populates it. M3 wires this into `pixelrag_core::Pipeline::open`.
    pub index_path: Option<PathBuf>,
    /// Path to the ONNX/encoder model weights used by `/index`.
    ///
    /// M3 wires this into `pixelrag_encoder::Embedder` construction.
    pub model_path: Option<PathBuf>,
    /// Tile-embedding batch size for ingest (default `32`, per ADR-264 M1).
    pub batch_size: usize,
    /// Maximum number of concurrently served requests (back-pressure bound).
    pub max_concurrency: usize,
    /// Request body size cap in bytes (guards `/index` image uploads).
    pub max_body_bytes: usize,
    /// Optional path to a darwin-generated config JSON (ADR-264 §MetaHarness).
    ///
    /// Read-only optimization input; never required for correctness.
    pub darwin_config_path: Option<PathBuf>,
}

impl Default for ServerConfig {
    /// Hard-coded M3 defaults. Must always produce a correct, usable server
    /// even if darwin is never run (ADR-264 removability constraint).
    fn default() -> Self {
        ServerConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8788),
            index_path: None,
            model_path: None,
            batch_size: 32,
            max_concurrency: 64,
            max_body_bytes: 16 * 1024 * 1024,
            darwin_config_path: None,
        }
    }
}

impl ServerConfig {
    /// Build a config from process environment variables, layered over
    /// [`ServerConfig::default`].
    ///
    /// M3: read `PIXELRAG_BIND`, `PIXELRAG_INDEX`, `PIXELRAG_MODEL`,
    /// `PIXELRAG_BATCH_SIZE`, etc., validate them, and return a typed config.
    /// Unset variables retain their default values.
    pub fn from_env() -> crate::Result<Self> {
        unimplemented!(
            "M3: parse PIXELRAG_* env vars over ServerConfig::default(); \
             validate bind addr and numeric fields at the system boundary"
        )
    }

    /// Load an optional darwin-generated tuning JSON (ADR-264 §MetaHarness).
    ///
    /// M3: deserialize `{ batch_size, max_concurrency, … }` via `serde_json`
    /// and merge into `self`. On ANY error the caller must fall back to
    /// defaults — darwin output is read-only and never required.
    pub fn from_darwin_json(_path: &std::path::Path) -> crate::Result<Self> {
        unimplemented!(
            "M3: serde_json::from_reader into a partial overlay merged onto \
             ServerConfig::default(); errors are non-fatal (use defaults)"
        )
    }

    /// Validate field invariants (positive batch size, non-zero concurrency).
    ///
    /// M3: called once at startup after env/darwin merge; returns
    /// [`crate::ServeError::BadRequest`] on violation.
    pub fn validate(&self) -> crate::Result<()> {
        unimplemented!("M3: assert batch_size > 0, max_concurrency > 0, max_body_bytes > 0")
    }
}
