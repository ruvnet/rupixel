//! `hyper`-based HTTP server + router stub (ADR-264, M3).
//!
//! Replaces upstream FastAPI's Uvicorn server. Owns the Tokio listener, accepts
//! connections, routes `(method, path)` to the handlers in [`crate::handlers`],
//! and (de)serializes JSON bodies via `serde_json`. M3 brings in `hyper` 1.x
//! (`server`/`http1`), `http-body-util` for body collection, and `tokio` for
//! the runtime. M0 is a std-only skeleton fixing the public API.

use crate::config::ServerConfig;
use crate::handlers::AppState;

/// The PixelRAG HTTP server.
///
/// Holds the validated [`ServerConfig`] and shared [`AppState`]. M3: also holds
/// the bound `tokio::net::TcpListener` and a graceful-shutdown signal handle.
pub struct Server {
    config: ServerConfig,
    state: AppState,
}

impl Server {
    /// Build a server from config: validate it, then construct shared state
    /// (open index, load encoder).
    ///
    /// M3: `config.validate()?`, then `AppState::new(config)` to open the
    /// `pixelrag-core` pipeline and load the encoder model.
    pub fn new(_config: ServerConfig) -> crate::Result<Self> {
        unimplemented!(
            "M3: config.validate()?; let state = AppState::new(config.clone())?; \
             Ok(Server {{ config, state }})"
        )
    }

    /// Bind the listener and serve requests until shutdown.
    ///
    /// M3 (async): bind `tokio::net::TcpListener` to `config.bind`, accept in a
    /// loop, and serve each connection with `hyper::server::conn::http1` whose
    /// service routes via [`Server::route`]. Applies the `max_concurrency`
    /// back-pressure bound from config.
    pub fn serve(&self) -> crate::Result<()> {
        unimplemented!(
            "M3 (async): bind TcpListener(config.bind); accept loop; \
             hyper::server::conn::http1::Builder::serve_connection(io, service_fn(route)); \
             honor config.max_concurrency"
        )
    }

    /// Route a parsed `(method, path)` request to the matching handler.
    ///
    /// M3: the dispatch core. Match the table below; parse the body into the
    /// request type (capped at `config.max_body_bytes`); call the handler;
    /// serialize the response to a JSON `hyper::Response`. Unknown routes →
    /// `404`; bad bodies → `400` ([`crate::ServeError::BadRequest`]).
    ///
    /// | Method | Path      | Handler                              |
    /// |--------|-----------|--------------------------------------|
    /// | POST   | `/index`  | [`crate::handlers::handle_index`]    |
    /// | POST   | `/search` | [`crate::handlers::handle_search`]   |
    /// | GET    | `/health` | [`crate::handlers::handle_health`]   |
    pub fn route(_state: &AppState, _method: &str, _path: &str, _body: &[u8]) -> crate::Result<Vec<u8>> {
        unimplemented!(
            "M3: match (method, path): POST /index -> handle_index, POST /search -> \
             handle_search, GET /health -> handle_health; serde_json (de)serialize; \
             else ServeError::NotFound"
        )
    }

    /// Borrow the active server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Borrow the shared application state.
    pub fn state(&self) -> &AppState {
        &self.state
    }
}
