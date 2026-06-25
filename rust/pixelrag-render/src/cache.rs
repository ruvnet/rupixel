//! Disk cache for rendered images / tiles.
//!
//! Rendering (headless Chrome navigation, PDFium rasterization) is the most
//! expensive stage of the pipeline. This module content-addresses render outputs
//! so an unchanged `(source, RenderConfig)` is never re-rendered — important for
//! re-indexing and for the benchmark harness (ADR-264 §Validation), which sweeps
//! tile/quantization params over a fixed document set.
//!
//! **Milestone:** M2 (ADR-264 §Milestones). M0 is a compiling skeleton.
//!
//! **Intended backend:** `std::fs` for the on-disk tile store, `sha2` for cache
//! keys, `serde`/`serde_json` for the manifest, `tokio::fs` for async I/O in the
//! serve path. None are deps yet.

use crate::{RenderConfig, RenderSource, RenderedImage, Result, Tile};
use std::path::{Path, PathBuf};

/// Content-addressed cache key for a `(source, config)` render request.
///
/// Backed by a hex digest (M2: SHA-256 over the canonicalized source + render
/// config). Two requests collide iff they would produce byte-identical tiles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// Derive a cache key from a render request.
    ///
    /// Intended behavior (M2): hash `(canonical source, RenderConfig fields)`
    /// with SHA-256 and hex-encode. `RenderConfig` is included so a tile-size or
    /// format change invalidates stale entries. Wires to the `sha2` crate.
    pub fn from_request(_source: &RenderSource, _config: &RenderConfig) -> Self {
        unimplemented!("M2: SHA-256 over canonical (source, RenderConfig) → hex key")
    }
}

/// One manifest entry recording a cached render.
///
/// Persisted (M2: as JSON via `serde`) so the cache survives process restarts
/// and the harness can audit hit/miss provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Key this entry is stored under.
    pub key: CacheKey,
    /// Number of tiles cached for this entry.
    pub tile_count: usize,
    /// Relative paths (under the cache root) of the stored tile images.
    pub tile_paths: Vec<PathBuf>,
    /// Unix-epoch seconds when this entry was written.
    pub created_at_secs: u64,
}

/// Disk-backed render cache rooted at a directory.
///
/// Layout (M2): `<root>/manifest.json` + `<root>/<key-prefix>/<key>/tile_<n>.<ext>`.
#[derive(Debug, Clone)]
pub struct RenderCache {
    /// Root directory for cached tiles + manifest.
    pub root: PathBuf,
    /// Soft cap on total cache size (bytes); 0 = unbounded. Eviction is LRU.
    pub max_bytes: u64,
}

impl RenderCache {
    /// Open (creating if absent) a render cache rooted at `root`.
    ///
    /// Intended behavior (M2): `create_dir_all(root)`, load `manifest.json` if
    /// present (else start empty). Maps I/O failures to
    /// [`crate::RenderError::Cache`].
    pub fn open(_root: &Path, _max_bytes: u64) -> Result<Self> {
        unimplemented!("M2: open/create cache root, load manifest.json via serde")
    }

    /// Look up tiles for a key. `Ok(None)` on a clean miss.
    ///
    /// Intended behavior (M2): resolve the manifest entry, read each tile image
    /// from disk, decode, and return [`Tile`]s. Wires to `std::fs` + `image`.
    pub fn get(&self, _key: &CacheKey) -> Result<Option<Vec<Tile>>> {
        unimplemented!("M2: read manifest entry + tile images from disk")
    }

    /// Store tiles under a key, updating the manifest and evicting if over cap.
    ///
    /// Intended behavior (M2): write each tile image to
    /// `<root>/<key>/tile_<n>.<ext>`, append a [`CacheEntry`], persist the
    /// manifest, and LRU-evict to honor `max_bytes`. Wires to `std::fs` + `serde`.
    pub fn put(&self, _key: &CacheKey, _tiles: &[Tile]) -> Result<()> {
        unimplemented!("M2: write tiles to disk, update + persist manifest, LRU-evict")
    }

    /// Whether a key is present without reading tile bytes.
    ///
    /// Intended behavior (M2): manifest membership check (no disk reads).
    pub fn contains(&self, _key: &CacheKey) -> bool {
        unimplemented!("M2: manifest membership check")
    }

    /// Remove a cached entry and its tile files.
    ///
    /// Intended behavior (M2): delete the key's tile directory and drop its
    /// manifest entry, then persist the manifest.
    pub fn evict(&self, _key: &CacheKey) -> Result<()> {
        unimplemented!("M2: delete key's tiles + manifest entry")
    }

    /// Drop every entry and reset the on-disk store.
    ///
    /// Intended behavior (M2): remove all tile directories and reset the
    /// manifest to empty. Used by the benchmark harness for clean-run isolation.
    pub fn clear(&self) -> Result<()> {
        unimplemented!("M2: remove all cached tiles + reset manifest")
    }
}

/// Side-channel helper: persist a standalone [`RenderedImage`] (pre-tiling) to
/// disk, e.g. for debugging or harness inspection.
///
/// Intended behavior (M2): write `image.bytes` to `path` (extension chosen from
/// `image.format`). Wires to `std::fs`.
pub fn write_image(_image: &RenderedImage, _path: &Path) -> Result<()> {
    unimplemented!("M2: write a RenderedImage to disk for inspection")
}
