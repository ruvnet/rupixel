//! Runtime configuration for the PixelRAG core orchestrator.
//!
//! Implements the **removable darwin augmentation** contract from ADR-264
//! §"MetaHarness / Darwin integration" and §"Explicit coding rule":
//!
//! - [`Config::default`] returns the hard-coded **default M1 harness** (HNSW,
//!   batch=32, cache=100MB, no rerank). The binary is fully usable with this
//!   alone — darwin is never required.
//! - [`Config::from_darwin_json`] loads an *optional* darwin-evolved genome. If
//!   it fails (file missing, parse error), callers fall back to `default()`.
//!   The genome is **read-only** input — it does not control the runtime via env
//!   vars or APIs, per ADR-256 removability governance.

use std::path::{Path, PathBuf};

use crate::Result;

/// Which ANN backend the index adaptor wraps. Defaults to [`IndexBackend::Hnsw`]
/// (M1 primary). See ADR-264 reuse boundary table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBackend {
    /// `ruvector-core` HNSW — M1 primary backend.
    Hnsw,
    /// `ruvector-rairs` IVF-Flat — M1 train-then-add IVF backend (single
    /// assignment, flat list scan). The genuine second backend the darwin
    /// harness chooses between (`hnsw` vs `ivf-flat`).
    IvfFlat,
    /// `ruvector-rairs` IVF-SQ — M1 fallback when HNSW memory exceeds budget (ADR-193).
    IvfSq,
}

impl Default for IndexBackend {
    fn default() -> Self {
        IndexBackend::Hnsw
    }
}

/// Top-level runtime configuration.
///
/// Field set mirrors the ADR-264 "Explicit coding rule" snippet verbatim, plus
/// `embedding_cache_mb` and `batch_size` documented defaults.
#[derive(Debug, Clone)]
pub struct Config {
    /// ANN backend selection. Default: [`IndexBackend::Hnsw`].
    pub index_backend: IndexBackend,
    /// Tile embedding batch size handed to the encoder. Default: `32`.
    pub batch_size: usize,
    /// LRU embedding-cache budget in megabytes. Default: `100`.
    pub embedding_cache_mb: usize,
    /// Optional path to a darwin-evolved genome (JSON). `None` ⇒ pure defaults.
    pub darwin_config_path: Option<PathBuf>,
}

impl Config {
    /// Hard-coded **default M1 harness** — usable with zero darwin involvement.
    ///
    /// `HNSW` backend, `batch_size = 32`, `embedding_cache_mb = 100`,
    /// `darwin_config_path = None`. This is the guaranteed-correct baseline the
    /// ADR requires the binary to fall back to.
    pub fn default() -> Self {
        Config {
            index_backend: IndexBackend::Hnsw,
            batch_size: 32,
            embedding_cache_mb: 100,
            darwin_config_path: None,
        }
    }

    /// Load an OPTIONAL darwin-evolved genome (JSON) from `path` and merge it
    /// over [`Config::default`].
    ///
    /// **M1**: parse the JSON `(config, metrics)` genome produced by
    /// `@metaharness/darwin` v0.7.0 (via `serde_json`), validate ranges, and
    /// apply only the harness parameters (backend, batch, cache, rerank
    /// threshold). On any failure callers MUST fall back to [`Config::default`] —
    /// darwin is removable. Returns [`crate::Error::Config`] on parse/validation
    /// failure so the caller can decide to fall back.
    pub fn from_darwin_json(path: &Path) -> Result<Self> {
        use crate::Error;

        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("darwin genome read failed ({}): {e}", path.display())))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::Config(format!("darwin genome parse failed: {e}")))?;

        // The @metaharness/darwin genome is `{ "config": {...}, "metrics": {...} }`.
        // Only the harness `config` subtree is honored; `metrics` are read-only
        // provenance we never act on. A bare `{...}` (no wrapper) is also accepted.
        let cfg = json.get("config").unwrap_or(&json);

        // Start from the guaranteed-correct baseline and overlay only the
        // recognized, in-range fields. Unknown keys are ignored (forward-compat);
        // out-of-range values are a hard error so a bad genome never silently
        // degrades the harness — the caller falls back to default() on Err.
        let mut out = Config::default();

        if let Some(v) = cfg.get("index_backend") {
            let s = v
                .as_str()
                .ok_or_else(|| Error::Config("index_backend must be a string".into()))?;
            out.index_backend = match s.to_ascii_lowercase().as_str() {
                "hnsw" => IndexBackend::Hnsw,
                "ivfflat" | "ivf_flat" | "ivf-flat" => IndexBackend::IvfFlat,
                "ivfsq" | "ivf_sq" | "ivf-sq" => IndexBackend::IvfSq,
                other => return Err(Error::Config(format!("unknown index_backend '{other}'"))),
            };
        }
        if let Some(v) = cfg.get("batch_size") {
            out.batch_size = v
                .as_u64()
                .ok_or_else(|| Error::Config("batch_size must be a positive integer".into()))?
                as usize;
        }
        if let Some(v) = cfg.get("embedding_cache_mb") {
            out.embedding_cache_mb = v
                .as_u64()
                .ok_or_else(|| Error::Config("embedding_cache_mb must be an integer".into()))?
                as usize;
        }

        out.darwin_config_path = Some(path.to_path_buf());
        out.validate()?;
        Ok(out)
    }

    /// Validate that all fields are in-range (e.g. `batch_size > 0`,
    /// `embedding_cache_mb` within process budget).
    ///
    /// **M1**: enforce bounds and return [`crate::Error::Config`] on violation.
    pub fn validate(&self) -> Result<()> {
        use crate::Error;

        if self.batch_size == 0 {
            return Err(Error::Config("batch_size must be > 0".into()));
        }
        // A 32 GiB cap keeps an evolved genome from requesting an absurd cache
        // budget; 0 is allowed (caching disabled).
        if self.embedding_cache_mb > 32 * 1024 {
            return Err(Error::Config(format!(
                "embedding_cache_mb {} exceeds the 32768 MB ceiling",
                self.embedding_cache_mb
            )));
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::default()
    }
}
