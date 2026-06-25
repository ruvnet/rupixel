//! Web-page rendering via headless Chrome (CDP) / Playwright-CDP.
//!
//! Ports the web side of upstream `pixelshot` (Playwright + Chrome DevTools
//! Protocol). Loads a URL or local HTML file in a headless browser, waits for
//! the page to settle, and captures a full-page screenshot that [`crate::tile_pages`]
//! later splits into tiles.
//!
//! **Milestone:** M2 (ADR-264 §Milestones). M0 is a compiling skeleton.
//!
//! **Intended backend:** `headless_chrome` crate (CDP `Page.captureScreenshot`),
//! with `playwright-rs` as an alternative for closer parity with upstream
//! `pixelshot`. Selected behind a feature flag in M2; neither is a dep yet.

use crate::{RenderConfig, RenderSource, RenderedImage, Renderer, Result};

/// Headless-browser web renderer.
///
/// Holds the launch/navigation policy (timeouts, JS-wait strategy, blocked
/// resource types). In M1/M2 it will own a pooled browser handle
/// (`headless_chrome::Browser`) so tabs are reused across renders.
#[derive(Debug, Clone)]
pub struct WebRenderer {
    /// Max milliseconds to wait for navigation + network idle before capture.
    pub nav_timeout_ms: u64,
    /// If true, wait for `networkidle` (no in-flight requests) before capture;
    /// else capture on `load`. Mirrors `pixelshot`'s wait strategy.
    pub wait_network_idle: bool,
    /// Block image/font/media loads to speed text/layout-only captures.
    pub block_heavy_resources: bool,
}

impl Default for WebRenderer {
    fn default() -> Self {
        WebRenderer {
            nav_timeout_ms: 30_000,
            wait_network_idle: true,
            block_heavy_resources: false,
        }
    }
}

impl WebRenderer {
    /// Construct a web renderer with default navigation policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Launch (or attach to a pooled) headless Chrome instance.
    ///
    /// Intended behavior (M2): launch `headless_chrome::Browser` with the
    /// viewport from `RenderConfig.viewport_width` / `device_scale_x100`,
    /// returning a ready handle. Errors map to
    /// [`crate::RenderError::BackendUnavailable`].
    pub fn launch(&self, _config: &RenderConfig) -> Result<()> {
        unimplemented!("M2: launch headless Chrome (CDP) via the `headless_chrome` crate")
    }

    /// Navigate to a single URL/HTML source and capture a full-page screenshot.
    ///
    /// Intended behavior (M2): open a tab, navigate, honor `nav_timeout_ms` /
    /// `wait_network_idle`, then `Page.captureScreenshot` (full page) and decode
    /// into a [`RenderedImage`]. Wires to `headless_chrome` + `image`.
    pub fn capture(&self, _source: &RenderSource, _config: &RenderConfig) -> Result<RenderedImage> {
        unimplemented!("M2: CDP navigate + full-page captureScreenshot")
    }
}

impl Renderer for WebRenderer {
    /// Render a web source into page images (typically one tall page).
    ///
    /// Intended behavior (M2): validate the source is `Url`/`HtmlFile`, `launch`
    /// if needed, `capture`, and return `vec![page]`. PDF sources are rejected
    /// here (routed to [`crate::pdf::PdfRenderer`] by [`crate::render_to_tiles`]).
    fn render(&self, _source: &RenderSource, _config: &RenderConfig) -> Result<Vec<RenderedImage>> {
        unimplemented!("M2: render web source via headless Chrome → page image(s)")
    }
}
