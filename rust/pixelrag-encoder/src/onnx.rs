//! ONNX-runtime loader + batched inference backend (ADR-264 v2, M1 primary path).
//!
//! Wraps the ONNX Runtime Rust binding (`ort`) to load a vision-embedding model
//! (Qwen3-VL-Embedding-2B export or CLIP ViT-L surrogate — see [`crate::model`]) and
//! run batched forward passes over screenshot tiles. This is the recommended
//! single-binary deployment path from ADR-264 ("v2 — ONNX Runtime via `ort`").
//!
//! M0: the loader and inference paths are skeletons. The `ort::Session` and tensor
//! plumbing land in M1 once the `ort` + `ndarray` deps are enabled in `Cargo.toml`.

use std::path::{Path, PathBuf};

use crate::model::{ModelKind, Preprocessor};
use crate::{Embedder, EmbedderKind, Embedding, EncoderError, Image};

/// Compute device the ONNX session should target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// CPU execution provider (default, always available).
    Cpu,
    /// CUDA execution provider; `usize` is the GPU ordinal.
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

/// Configuration for constructing an [`OnnxEmbedder`].
#[derive(Clone, Debug)]
pub struct OnnxConfig {
    /// Path to the `.onnx` model file (Qwen3-VL export or CLIP surrogate).
    pub model_path: PathBuf,
    /// Which model family this file is — drives preprocessing + output handling.
    pub model_kind: ModelKind,
    /// Execution provider / device.
    pub device: Device,
    /// Number of intra-op threads for the ONNX session (CPU). `None` → runtime default.
    pub intra_threads: Option<usize>,
    /// Whether to L2-normalize the model output before returning (PixelRAG indexes
    /// normalized vectors so cosine == inner-product downstream).
    pub normalize: bool,
}

impl OnnxConfig {
    /// Build a config for the given model file + family, with CPU/default settings
    /// and output normalization enabled (matching PixelRAG's normalized FAISS index).
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>, model_kind: ModelKind) -> Self {
        Self {
            model_path: model_path.into(),
            model_kind,
            device: Device::Cpu,
            intra_threads: None,
            normalize: true,
        }
    }
}

/// ONNX Runtime visual encoder ([`EmbedderKind::Onnx`]).
///
/// Holds the loaded session + preprocessing config. In M1 it owns an `ort::Session`
/// (not represented in M0 to keep the crate std-only); here it carries the resolved
/// config and a [`Preprocessor`] so the public surface is stable.
pub struct OnnxEmbedder {
    config: OnnxConfig,
    preprocessor: Preprocessor,
    embedding_dim: usize,
    // M1: session: ort::Session,
}

impl OnnxEmbedder {
    /// Load an ONNX model from `config`, building the ONNX Runtime session and the
    /// matching [`Preprocessor`].
    ///
    /// M1: create the `ort::Environment` + `ort::Session` from `config.model_path`,
    /// select the execution provider from `config.device`, infer `embedding_dim`
    /// from the model's output tensor shape, and construct the `Preprocessor` from
    /// `config.model_kind`. Wires to the `ort` crate (ADR-264 v2 / ADR-194 ONNX
    /// embedder API).
    ///
    /// # Errors
    /// Returns [`EncoderError::ModelLoad`] if the file is missing or the session
    /// cannot be created.
    pub fn load(_config: OnnxConfig) -> Result<Self, EncoderError> {
        // Real path: build ort::Session, infer embedding_dim from output shape.
        // Intentionally NOT implemented in this environment (weights + GPU blocked);
        // use `SyntheticEmbedder` for plumbing. The `ort` dep is not added.
        unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")
    }

    /// Convenience loader from a model-file path, inferring the family from the path.
    ///
    /// # Errors
    /// Returns [`EncoderError::ModelLoad`] on failure to resolve or load.
    pub fn from_path(_path: &Path) -> Result<Self, EncoderError> {
        // Real path: sniff ModelKind from filename/metadata, delegate to `load`.
        unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")
    }

    /// The resolved configuration this embedder was built with.
    #[must_use]
    pub fn config(&self) -> &OnnxConfig {
        &self.config
    }

    /// The preprocessor applied to each tile before inference.
    #[must_use]
    pub fn preprocessor(&self) -> &Preprocessor {
        &self.preprocessor
    }
}

impl Embedder for OnnxEmbedder {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn embed_batch(&self, _tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        // Real path: for each tile run `self.preprocessor` → stack into one NCHW
        // ndarray → ort::Session::run a single forward pass → split rows into
        // Vec<Embedding>, L2-normalizing if `config.normalize`. Wires to `ort` +
        // `ndarray`. Blocked here — use `SyntheticEmbedder` for plumbing validation.
        unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")
    }

    fn kind(&self) -> EmbedderKind {
        EmbedderKind::Onnx
    }
}
