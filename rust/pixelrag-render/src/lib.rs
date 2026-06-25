//! # pixelrag-render — document → screenshot-tile rendering (optional / M2)
//!
//! Per **ADR-264** (PixelRAG Rust port on ruvector substrate), this crate ports
//! the upstream `pixelshot` render stage: it turns documents (web pages, PDFs)
//! into **screenshot tiles** that the rest of the pipeline (`pixelrag-encoder` →
//! `pixelrag-core`) embeds and indexes. PixelRAG retrieves over *visual*
//! embeddings instead of parsed text, so faithful rendering — preserving tables,
//! charts, and layout — is the front of the whole pipeline.
//!
//! This crate is **optional and lands in M2** (ADR-264 §Milestones). M1 can run
//! on a precomputed tile cache or an upstream Python `pixelshot` sidecar; this
//! crate replaces that with a native Rust render path.
//!
//! ## Modules
//! - [`web`]   — headless-Chrome (CDP) / Playwright-CDP page rendering.
//! - [`pdf`]   — pdfium-backed PDF page rasterization.
//! - [`cache`] — content-addressed disk cache for rendered images/tiles.
//!
//! ## M0 status
//! Compiling **skeleton only**. All bodies are `unimplemented!("M1/M2: …")`.
//! No external crates (std-only); the intended M1/M2 deps (`headless_chrome`,
//! `pdfium-render`, `image`, `tokio`, `serde`, `sha2`) are documented as
//! comments in `Cargo.toml` and wired in at their milestone.
//!
//! ## Tile size note
//! Tile dimensions are **not** hardcoded. They come from [`RenderConfig`], which
//! mirrors upstream `pixelshot` config (ADR-264 reuse boundary: "Screenshot
//! tiles (size per `pixelshot` config)"). Defaults here are placeholders to be
//! reconciled with the upstream config during M2.

pub mod cache;
pub mod pdf;
pub mod web;

use std::fmt;
use std::path::PathBuf;

/// Crate-local result alias. M1 swaps the error for a `thiserror`-derived enum.
pub type Result<T> = std::result::Result<T, RenderError>;

/// Errors surfaced by the render stage.
///
/// M0 carries `String` payloads to stay std-only; M1 will replace these with
/// structured variants wrapping the underlying `headless_chrome` / `pdfium`
/// / `std::io` errors via `thiserror`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// Backend (browser/pdfium) failed to launch or attach.
    BackendUnavailable(String),
    /// The source (URL / PDF path / bytes) could not be loaded.
    SourceLoad(String),
    /// Rendering the page/document to a raster image failed.
    Render(String),
    /// Tiling a rendered page into sub-images failed.
    Tile(String),
    /// Disk-cache read/write/manifest error.
    Cache(String),
    /// Feature not yet implemented at this milestone.
    Unimplemented(&'static str),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::BackendUnavailable(m) => write!(f, "render backend unavailable: {m}"),
            RenderError::SourceLoad(m) => write!(f, "source load failed: {m}"),
            RenderError::Render(m) => write!(f, "render failed: {m}"),
            RenderError::Tile(m) => write!(f, "tiling failed: {m}"),
            RenderError::Cache(m) => write!(f, "cache error: {m}"),
            RenderError::Unimplemented(m) => write!(f, "not implemented: {m}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Image encoding for a rendered raster / tile.
///
/// PNG preserves crisp text/layout edges (lossless) — the visual detail
/// retrieval depends on; JPEG trades fidelity for smaller cache footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// Lossless PNG. Default for text/table/layout fidelity.
    Png,
    /// Lossy JPEG with the given quality (1–100).
    Jpeg(u8),
}

impl Default for ImageFormat {
    fn default() -> Self {
        ImageFormat::Png
    }
}

/// Render configuration mirroring upstream `pixelshot`.
///
/// **Tile size is configurable — not hardcoded** (ADR-264 reuse boundary). The
/// values here are placeholder defaults to be reconciled with the upstream
/// `pixelshot` config during M2; downstream callers (`pixelrag-core`) should
/// supply the authoritative config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderConfig {
    /// Tile width in pixels (per `pixelshot` config).
    pub tile_width: u32,
    /// Tile height in pixels (per `pixelshot` config).
    pub tile_height: u32,
    /// Overlap (px) between adjacent tiles, to avoid splitting content at seams.
    pub tile_overlap: u32,
    /// Page/viewport width used while rendering, before tiling.
    pub viewport_width: u32,
    /// Device pixel ratio (≥1 for hi-DPI crisper text). Stored ×100 to stay
    /// integral and `Eq` in M0; M1 may switch to `f32` once `image` lands.
    pub device_scale_x100: u32,
    /// Output encoding for tiles.
    pub format: ImageFormat,
}

impl Default for RenderConfig {
    /// Placeholder defaults — **reconcile with upstream `pixelshot` config in M2**.
    /// Deliberately not the legacy 256×256; the authoritative size is config-driven.
    fn default() -> Self {
        RenderConfig {
            tile_width: 0,
            tile_height: 0,
            tile_overlap: 0,
            viewport_width: 0,
            device_scale_x100: 100,
            format: ImageFormat::Png,
        }
    }
}

/// A single rendered raster image (a full page, pre-tiling), held in memory.
///
/// M1 replaces the raw byte buffer with an `image::RgbaImage` (or keeps encoded
/// bytes + format for cache round-trips).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedImage {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Encoded image bytes in [`RenderedImage::format`].
    pub bytes: Vec<u8>,
    /// Encoding of `bytes`.
    pub format: ImageFormat,
}

/// Pixel-space bounds of a tile within its source page (for metadata / provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBounds {
    /// Left edge (px) within the source page.
    pub x: u32,
    /// Top edge (px) within the source page.
    pub y: u32,
    /// Tile width (px).
    pub width: u32,
    /// Tile height (px).
    pub height: u32,
}

/// One screenshot tile produced from a rendered page.
///
/// This is the unit the encoder embeds. `page_index` + `bounds` give the
/// provenance `pixelrag-core` carries as retrieval metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    /// Zero-based source page index (0 for single-page web renders).
    pub page_index: u32,
    /// Position/size of this tile within the source page.
    pub bounds: TileBounds,
    /// Encoded tile image.
    pub image: RenderedImage,
}

/// Common contract for a render backend (web or PDF).
///
/// Implemented by [`web::WebRenderer`] and [`pdf::PdfRenderer`]. Kept synchronous
/// in M0 for a std-only skeleton; M1 introduces async variants on a Tokio pool
/// (ADR-264 §"Async Tokio").
pub trait Renderer {
    /// Render a source into one raster image per page.
    ///
    /// Web sources typically yield a single (tall) page; PDFs yield one per page.
    fn render(&self, source: &RenderSource, config: &RenderConfig) -> Result<Vec<RenderedImage>>;

    /// Split rendered page images into [`Tile`]s per `config` (size, overlap).
    ///
    /// Default-tiling lives in [`tile_pages`]; backends may override for
    /// backend-specific tiling (e.g., PDF text-region-aware splits).
    fn tile(&self, pages: &[RenderedImage], config: &RenderConfig) -> Result<Vec<Tile>> {
        tile_pages(pages, config)
    }
}

/// A document source to be rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderSource {
    /// Remote web page fetched and rendered headlessly.
    Url(String),
    /// Local HTML file path.
    HtmlFile(PathBuf),
    /// Local PDF file path.
    PdfFile(PathBuf),
    /// In-memory PDF bytes (e.g., streamed ingest).
    PdfBytes(Vec<u8>),
}

/// Default page→tile splitter shared by both backends.
///
/// Intended behavior (M2): for each page, slide a `tile_width × tile_height`
/// window with `tile_overlap` stride reduction, crop each window into a [`Tile`]
/// (carrying `bounds` + `page_index`), and re-encode in `config.format`. Wires
/// to the `image` crate for crop/encode once it lands.
pub fn tile_pages(_pages: &[RenderedImage], _config: &RenderConfig) -> Result<Vec<Tile>> {
    unimplemented!("M2: window-tile rendered pages per RenderConfig via the `image` crate")
}

/// Top-level convenience: dispatch a [`RenderSource`] to the right backend,
/// render → tile, consulting `cache` to skip recomputation.
///
/// Intended behavior (M2): compute a content-addressed key from
/// `(source, config)`, look it up in `cache`; on miss, pick
/// [`web::WebRenderer`] (URL/HTML) or [`pdf::PdfRenderer`] (PDF), render, tile,
/// store to cache, and return the tiles. This is the entry point
/// `pixelrag-core::pipeline` calls.
pub fn render_to_tiles(
    _source: &RenderSource,
    _config: &RenderConfig,
    _cache: &cache::RenderCache,
) -> Result<Vec<Tile>> {
    unimplemented!("M2: dispatch source→backend, render+tile, with cache lookup/store")
}
