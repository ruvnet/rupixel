//! PDF rasterization via pdfium.
//!
//! Ports the PDF side of upstream `pixelshot` (`pdf2image` / PDF libraries).
//! Loads a PDF (path or bytes), rasterizes each page to a bitmap at the
//! configured resolution, and emits one [`RenderedImage`] per page for
//! [`crate::tile_pages`] to split into tiles.
//!
//! **Milestone:** M2 (ADR-264 §Milestones). M0 is a compiling skeleton.
//!
//! **Intended backend:** `pdfium-render` (PDFium FFI). Renders each
//! `PdfPage` to an `image::DynamicImage` at a DPI derived from
//! `RenderConfig.device_scale_x100`. Not a dep yet.

use crate::{RenderConfig, RenderSource, RenderedImage, Renderer, Result};

/// PDFium-backed PDF page renderer.
///
/// In M1/M2 it will own the loaded `pdfium_render::Pdfium` library handle so the
/// native PDFium library is initialized once and reused across documents.
#[derive(Debug, Clone)]
pub struct PdfRenderer {
    /// Target render resolution (dots-per-inch) for page rasterization.
    /// Higher DPI sharpens text/table detail at the cost of larger bitmaps.
    pub render_dpi: u32,
    /// Optional page subset to render (zero-based). `None` = all pages.
    pub page_range: Option<(u32, u32)>,
}

impl Default for PdfRenderer {
    fn default() -> Self {
        PdfRenderer {
            render_dpi: 150,
            page_range: None,
        }
    }
}

impl PdfRenderer {
    /// Construct a PDF renderer with default DPI and full page range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize / bind the native PDFium library.
    ///
    /// Intended behavior (M2): `Pdfium::bind_to_system_library()` (or bundled),
    /// returning a ready handle. Maps failures to
    /// [`crate::RenderError::BackendUnavailable`].
    pub fn init_library(&self) -> Result<()> {
        unimplemented!("M2: bind PDFium via the `pdfium-render` crate")
    }

    /// Report the number of pages in a PDF source without rendering.
    ///
    /// Intended behavior (M2): load the document (path or bytes) and return its
    /// page count; used to plan batched rasterization. Wires to `pdfium-render`.
    pub fn page_count(&self, _source: &RenderSource) -> Result<u32> {
        unimplemented!("M2: load PDF and return page count via `pdfium-render`")
    }

    /// Rasterize a single page (zero-based) to a [`RenderedImage`].
    ///
    /// Intended behavior (M2): render page `page_index` at `render_dpi` scaled by
    /// `RenderConfig.device_scale_x100`, encode to `config.format`. Wires to
    /// `pdfium-render` + `image`.
    pub fn rasterize_page(
        &self,
        _source: &RenderSource,
        _page_index: u32,
        _config: &RenderConfig,
    ) -> Result<RenderedImage> {
        unimplemented!("M2: rasterize one PDF page via PDFium at render_dpi")
    }
}

impl Renderer for PdfRenderer {
    /// Rasterize every (in-range) page of a PDF source into page images.
    ///
    /// Intended behavior (M2): validate the source is `PdfFile`/`PdfBytes`,
    /// `init_library` if needed, then `rasterize_page` over `page_range` (or all
    /// pages), returning one [`RenderedImage`] per page in order.
    fn render(&self, _source: &RenderSource, _config: &RenderConfig) -> Result<Vec<RenderedImage>> {
        unimplemented!("M2: rasterize all PDF pages via PDFium → page images")
    }
}
