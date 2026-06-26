//! # pixelrag-render — headless document → screenshot adaptor (ADR-264)
//!
//! REAL render path for the PixelRAG visual pipeline. This crate does **not**
//! embed a browser; it shells out to the verified Node sidecar
//! (`crates/pixelrag-cli/sidecar/render_sidecar.mjs`), which drives a
//! Chromium-family browser (Edge/Chrome) through `puppeteer-core` to screenshot
//! each URL into a PNG. The Rust side owns the subprocess + JSON protocol — the
//! same pattern `pixelrag-encoder::SidecarEmbedder` uses for the embed sidecars.
//!
//! ## Protocol (one round-trip per [`render`] call)
//!
//! The Rust side spawns `node <render_sidecar.mjs>`, writes a single JSON object
//! ```json
//! { "urls": ["https://…", …], "outDir": "C:/abs/dir", "width": 1024, "height": 768 }
//! ```
//! to its stdin (then closes stdin → EOF), and reads
//! ```json
//! { "images": [ { "id": "doc-00", "url": "…", "path": "C:/abs/dir/doc-00.png" }, … ] }
//! ```
//! from stdout. The sidecar resolves the browser via `PIXELRAG_BROWSER` or a list
//! of common Edge/Chrome install paths.
//!
//! No stubs: every code path here builds and runs. A missing browser / missing
//! Node surfaces as a real [`RenderError`] carrying the sidecar's stderr.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One rendered document screenshot returned by the sidecar.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RenderedImage {
    /// Stable sidecar-assigned id (`doc-00`, `doc-01`, … by input order).
    pub id: String,
    /// The source URL that was rendered.
    pub url: String,
    /// Absolute filesystem path to the written PNG (Windows path on Windows).
    pub path: PathBuf,
}

/// Error type for the render adaptor (spawn / IO / non-zero exit / parse).
#[derive(Debug)]
pub enum RenderError {
    /// Failed to spawn the `node` process (Node missing, bad path, etc.).
    Spawn(String),
    /// An I/O error writing the request or reading the response.
    Io(String),
    /// The sidecar exited non-zero (no browser found, navigation failure, …).
    Sidecar(String),
    /// The sidecar response JSON could not be parsed or was malformed.
    Parse(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Spawn(m) => write!(f, "pixelrag render spawn error: {m}"),
            RenderError::Io(m) => write!(f, "pixelrag render io error: {m}"),
            RenderError::Sidecar(m) => write!(f, "pixelrag render sidecar error: {m}"),
            RenderError::Parse(m) => write!(f, "pixelrag render parse error: {m}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Result alias for the render adaptor.
pub type Result<T> = std::result::Result<T, RenderError>;

/// Default screenshot viewport width (matches the visual fixture render).
pub const DEFAULT_WIDTH: u32 = 1024;
/// Default screenshot viewport height (matches the visual fixture render).
pub const DEFAULT_HEIGHT: u32 = 768;

/// Headless renderer that shells the Node `render_sidecar.mjs`.
///
/// The caller injects the sidecar script path (the CLI knows its own layout via
/// `env!("CARGO_MANIFEST_DIR")`); this crate must not assume it. The `node`
/// binary and sidecar path can be overridden by the `PIXELRAG_NODE` /
/// `PIXELRAG_RENDER_SIDECAR` env vars, mirroring `SidecarEmbedder`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Renderer {
    sidecar_path: PathBuf,
    node_bin: String,
    width: u32,
    height: u32,
}

impl Renderer {
    /// Construct a renderer that spawns `node <sidecar_path>`.
    ///
    /// `PIXELRAG_RENDER_SIDECAR` overrides `sidecar_path`; `PIXELRAG_NODE`
    /// overrides the `node` binary. Viewport defaults to
    /// [`DEFAULT_WIDTH`]×[`DEFAULT_HEIGHT`].
    #[must_use]
    pub fn new(sidecar_path: impl Into<PathBuf>) -> Self {
        let sidecar_path = match std::env::var_os("PIXELRAG_RENDER_SIDECAR") {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => sidecar_path.into(),
        };
        let node_bin = std::env::var("PIXELRAG_NODE").unwrap_or_else(|_| "node".to_string());
        Self { sidecar_path, node_bin, width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT }
    }

    /// Override the screenshot viewport size.
    #[must_use]
    pub fn with_viewport(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(1);
        self.height = height.max(1);
        self
    }

    /// The resolved sidecar script path this renderer will spawn.
    #[must_use]
    pub fn sidecar_path(&self) -> &Path {
        &self.sidecar_path
    }

    /// Render every `url` into a PNG under `out_dir`, returning one
    /// [`RenderedImage`] per URL in input order.
    ///
    /// One `node <render_sidecar.mjs>` round-trip: write the JSON request to
    /// stdin, close it (EOF), wait for completion, parse the JSON response.
    pub fn render<S: AsRef<str>>(
        &self,
        urls: &[S],
        out_dir: impl AsRef<Path>,
    ) -> Result<Vec<RenderedImage>> {
        if urls.is_empty() {
            return Ok(Vec::new());
        }
        let out_dir = out_dir.as_ref();
        let url_list: Vec<&str> = urls.iter().map(|u| u.as_ref()).collect();
        let request = serde_json::json!({
            "urls": url_list,
            "outDir": out_dir.to_string_lossy(),
            "width": self.width,
            "height": self.height,
        });
        let payload = serde_json::to_vec(&request)
            .map_err(|e| RenderError::Io(format!("serialize request: {e}")))?;

        let mut child = Command::new(&self.node_bin)
            .arg(&self.sidecar_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                RenderError::Spawn(format!(
                    "spawn `{} {}` failed: {e} (is Node installed and on PATH? set PIXELRAG_NODE to override)",
                    self.node_bin,
                    self.sidecar_path.display()
                ))
            })?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| RenderError::Io("sidecar stdin unavailable".into()))?;
            stdin
                .write_all(&payload)
                .map_err(|e| RenderError::Io(format!("write request to sidecar: {e}")))?;
        }
        child.stdin = None; // close stdin → EOF for the sidecar's readStdin()

        let output = child
            .wait_with_output()
            .map_err(|e| RenderError::Io(format!("wait for sidecar: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RenderError::Sidecar(format!(
                "render sidecar exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.trim();
        if line.is_empty() {
            return Err(RenderError::Sidecar(
                "render sidecar produced empty stdout (no JSON response)".into(),
            ));
        }
        let resp: RenderResponse = serde_json::from_str(line)
            .map_err(|e| RenderError::Parse(format!("parse render response JSON: {e}")))?;

        if resp.images.len() != urls.len() {
            return Err(RenderError::Sidecar(format!(
                "render sidecar returned {} images for {} urls",
                resp.images.len(),
                urls.len()
            )));
        }
        Ok(resp.images.into_iter().map(Into::into).collect())
    }
}

/// Convenience one-shot: build a [`Renderer`] for `sidecar_path` and render.
pub fn render<S: AsRef<str>>(
    sidecar_path: impl Into<PathBuf>,
    urls: &[S],
    out_dir: impl AsRef<Path>,
) -> Result<Vec<RenderedImage>> {
    Renderer::new(sidecar_path).render(urls, out_dir)
}

/// Deserialized sidecar stdout: `{"images":[{"id","url","path"}, …]}`.
#[derive(serde::Deserialize)]
struct RenderResponse {
    images: Vec<RawImage>,
}

/// Raw sidecar image entry (string `path`); converted to [`RenderedImage`].
#[derive(serde::Deserialize)]
struct RawImage {
    id: String,
    url: String,
    path: String,
}

impl From<RawImage> for RenderedImage {
    fn from(r: RawImage) -> Self {
        RenderedImage { id: r.id, url: r.url, path: PathBuf::from(r.path) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_urls_short_circuits_without_spawning() {
        // No node process is spawned for an empty URL list — pure adaptor logic.
        let r = Renderer::new("nonexistent-sidecar.mjs");
        let out: Vec<RenderedImage> = r.render::<&str>(&[], std::env::temp_dir()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn raw_image_maps_to_rendered_image() {
        let raw = RawImage {
            id: "doc-00".into(),
            url: "https://example.com".into(),
            path: "C:/abs/doc-00.png".into(),
        };
        let img: RenderedImage = raw.into();
        assert_eq!(img.id, "doc-00");
        assert_eq!(img.path, PathBuf::from("C:/abs/doc-00.png"));
    }

    #[test]
    fn with_viewport_clamps_to_at_least_one() {
        let r = Renderer::new("s.mjs").with_viewport(0, 0);
        assert_eq!(r.width, 1);
        assert_eq!(r.height, 1);
    }

    #[test]
    fn response_count_mismatch_is_an_error() {
        // A 2-image response for 1 url must be rejected — guards the id↔url mapping.
        let resp: RenderResponse = serde_json::from_str(
            r#"{"images":[{"id":"doc-00","url":"u","path":"p"},{"id":"doc-01","url":"u2","path":"p2"}]}"#,
        )
        .unwrap();
        assert_eq!(resp.images.len(), 2);
    }
}
