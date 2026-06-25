//! Document → tiles, with bounds and metadata.
//!
//! Per ADR-264 reuse boundary, tiling is implemented directly in Rust and
//! integrated with the render output (`pixelrag-render`, M2). In M1 the tiler
//! consumes already-rendered screenshots; in M2 it is fused into the
//! render→embed pipeline. A tile is the unit that gets embedded and indexed —
//! its [`TileBounds`] + [`TileMetadata`] let [`crate::search`] map a hit back to
//! a region of the source document.

use crate::Result;

/// Pixel-space bounds of a tile within its source document page.
///
/// Origin is top-left; all values are pixel coordinates in the rendered page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBounds {
    /// Left edge (px) of the tile within the page.
    pub x: u32,
    /// Top edge (px) of the tile within the page.
    pub y: u32,
    /// Tile width (px).
    pub width: u32,
    /// Tile height (px).
    pub height: u32,
}

/// Provenance + retrieval metadata carried alongside every tile.
///
/// M0 keeps this std-only. M1 derives `serde::{Serialize, Deserialize}` so tiles
/// persist into the `*.pixelrag` bincode artifact (M2) and round-trip through the
/// allowlist filter in [`crate::search`].
#[derive(Debug, Clone)]
pub struct TileMetadata {
    /// Source document identifier (URL, file path, or dataset doc id).
    pub doc_id: String,
    /// Zero-based page index within the source document.
    pub page: u32,
    /// Index of this tile within its page (row-major over the tiling grid).
    pub tile_index: u32,
}

/// A single screenshot tile: raw image bytes plus its bounds and metadata.
///
/// `image` is the encoded tile bitmap (e.g. PNG bytes from the renderer). The
/// embedder ([`crate::embedding::Embedder`]) decodes and embeds it; the indexer
/// assigns it an external id used by [`crate::SearchResult`].
#[derive(Debug, Clone)]
pub struct Tile {
    /// Encoded tile image bytes (decoded lazily by the encoder in M1).
    pub image: Vec<u8>,
    /// Where this tile sits in the source page.
    pub bounds: TileBounds,
    /// Provenance + retrieval metadata.
    pub metadata: TileMetadata,
}

/// Configuration for how a rendered document is split into tiles.
///
/// Mirrors upstream PixelRAG's `pixelshot` tile-size config. M1 wires these to
/// the renderer; M2 reads them from [`crate::config::Config`] / darwin genome.
#[derive(Debug, Clone, Copy)]
pub struct TileSpec {
    /// Target tile width in pixels.
    pub tile_width: u32,
    /// Target tile height in pixels.
    pub tile_height: u32,
    /// Pixel overlap between adjacent tiles (preserves cross-boundary structure).
    pub overlap: u32,
}

impl Default for TileSpec {
    fn default() -> Self {
        // Conservative placeholder; M1 calibrates against the encoder's input size.
        TileSpec { tile_width: 512, tile_height: 512, overlap: 0 }
    }
}

/// Splits a rendered document into tiles.
///
/// In M0 this is a skeleton. In M1 [`Tiler::tile_document`] takes the rendered
/// page bitmaps (one per page) and yields [`Tile`]s grid-split per [`TileSpec`],
/// stamping [`TileBounds`]/[`TileMetadata`] so retrieval hits are localizable.
#[derive(Debug, Clone, Default)]
pub struct Tiler {
    spec: TileSpec,
}

impl Tiler {
    /// Construct a tiler with the given [`TileSpec`].
    pub fn new(spec: TileSpec) -> Self {
        Tiler { spec }
    }

    /// The tile spec this tiler applies.
    pub fn spec(&self) -> TileSpec {
        self.spec
    }

    /// Split a rendered document into tiles.
    ///
    /// **M1**: `pages` are the rendered page bitmaps (PNG/raw bytes) for `doc_id`.
    /// Grid-split each page per [`TileSpec`] (with overlap), emit [`Tile`]s with
    /// correct [`TileBounds`] and [`TileMetadata`]. Integrates with
    /// `pixelrag-render` output in M2.
    pub fn tile_document(&self, doc_id: &str, pages: &[Vec<u8>]) -> Result<Vec<Tile>> {
        use crate::Error;

        if self.spec.tile_width == 0 || self.spec.tile_height == 0 {
            return Err(Error::Tile("TileSpec width/height must be > 0".into()));
        }

        // M1 path: `pages` are opaque byte buffers (a rendered page in M2, or a
        // synthetic fixture buffer today). We deterministically chunk each page's
        // bytes into tile-sized windows and stamp bounds/metadata so a hit is
        // localizable. No image decode/render here — that is M2/pixelrag-render.
        let chunk = (self.spec.tile_width as usize) * (self.spec.tile_height as usize);
        let mut tiles = Vec::new();
        for (page_idx, page) in pages.iter().enumerate() {
            // At least one tile per page even when the page is shorter than a
            // single window, so every page contributes a retrievable unit.
            let n_tiles = page.len().div_ceil(chunk).max(1);
            for tile_index in 0..n_tiles {
                let start = tile_index * chunk;
                let end = (start + chunk).min(page.len());
                let image = page.get(start..end).unwrap_or(&[]).to_vec();
                tiles.push(Tile {
                    image,
                    bounds: TileBounds {
                        x: 0,
                        y: tile_index as u32 * self.spec.tile_height,
                        width: self.spec.tile_width,
                        height: self.spec.tile_height,
                    },
                    metadata: TileMetadata {
                        doc_id: doc_id.to_string(),
                        page: page_idx as u32,
                        tile_index: tile_index as u32,
                    },
                });
            }
        }
        Ok(tiles)
    }

    /// Fixture convenience: split a plain-text "document" into deterministic
    /// tiles for plumbing validation.
    ///
    /// PixelRAG is visual-RAG; this does NOT render text to pixels. It exists only
    /// so the M1 fixture/bench can exercise tile → embed → index → search end to
    /// end with reproducible inputs. Each chunk of `chars_per_tile` UTF-8 bytes
    /// becomes one tile whose `image` is the raw text bytes (consumed by the
    /// deterministic synthetic embedder — NOT a real visual encoder).
    pub fn tile_text(&self, doc_id: &str, text: &str, chars_per_tile: usize) -> Result<Vec<Tile>> {
        use crate::Error;
        if chars_per_tile == 0 {
            return Err(Error::Tile("chars_per_tile must be > 0".into()));
        }
        let bytes = text.as_bytes();
        let n_tiles = bytes.len().div_ceil(chars_per_tile).max(1);
        let mut tiles = Vec::with_capacity(n_tiles);
        for tile_index in 0..n_tiles {
            let start = tile_index * chars_per_tile;
            let end = (start + chars_per_tile).min(bytes.len());
            tiles.push(Tile {
                image: bytes.get(start..end).unwrap_or(&[]).to_vec(),
                bounds: TileBounds {
                    x: 0,
                    y: tile_index as u32 * self.spec.tile_height,
                    width: self.spec.tile_width,
                    height: self.spec.tile_height,
                },
                metadata: TileMetadata {
                    doc_id: doc_id.to_string(),
                    page: 0,
                    tile_index: tile_index as u32,
                },
            });
        }
        Ok(tiles)
    }
}
