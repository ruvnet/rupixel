//! Model instantiation + preprocessing (ADR-264 §"Honest embedding model strategy").
//!
//! PixelRAG's upstream encoder is **Qwen3-VL-Embedding-2B** (a LoRA fine-tune of a
//! CLIP-style ViT). The Rust port targets an ONNX export of that model, with an
//! **OpenAI CLIP ViT-L/14 ONNX surrogate** as the licensing/availability fallback
//! (ADR-264 Links: model weight licensing is "unclear"; CLIP is the workaround).
//!
//! This module defines:
//! - [`ModelKind`]: which encoder family / which preprocessing recipe.
//! - [`Preprocessor`]: tile → normalized tensor (resize, center-crop, mean/std).
//! - [`ModelSpec`]: static metadata (input size, dim, mean/std) per family.
//!
//! M0: preprocessing math is deferred (`unimplemented!`); the specs are real values
//! drawn from the public model cards so downstream sizing code can compile against
//! stable constants in M1.

use crate::{EncoderError, Image};

/// Which visual-embedding model family is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// Qwen3-VL-Embedding-2B (upstream PixelRAG encoder, ONNX export). Primary target.
    Qwen3VlEmbedding2B,
    /// OpenAI CLIP ViT-L/14 (`clip-vit-large-patch14`) ONNX surrogate. Fallback when
    /// Qwen3-VL weights/licensing are unavailable (ADR-264 §encoder, Links).
    ClipVitLargePatch14,
}

impl ModelKind {
    /// Static spec (input geometry, dim, normalization constants) for this family.
    #[must_use]
    pub fn spec(self) -> ModelSpec {
        match self {
            // Qwen3-VL-Embedding-2B: 1024-d embedding, ViT-style 448px input.
            ModelKind::Qwen3VlEmbedding2B => ModelSpec {
                kind: self,
                input_size: 448,
                embedding_dim: 1024,
                // ImageNet-style normalization (placeholder pending exact model-card values, M1).
                mean: [0.481_454_66, 0.457_827_5, 0.408_210_72],
                std: [0.268_629_55, 0.261_302_6, 0.275_777_1],
            },
            // CLIP ViT-L/14: 768-d embedding, 224px input, CLIP normalization constants.
            ModelKind::ClipVitLargePatch14 => ModelSpec {
                kind: self,
                input_size: 224,
                embedding_dim: 768,
                mean: [0.481_454_66, 0.457_827_5, 0.408_210_72],
                std: [0.268_629_55, 0.261_302_6, 0.275_777_1],
            },
        }
    }
}

/// Static metadata describing a model family's I/O + preprocessing constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSpec {
    /// The family this spec describes.
    pub kind: ModelKind,
    /// Square input edge length in pixels the model expects (resize/crop target).
    pub input_size: u32,
    /// Output embedding dimensionality.
    pub embedding_dim: usize,
    /// Per-channel mean for normalization (R, G, B), in [0, 1] scale.
    pub mean: [f32; 3],
    /// Per-channel standard deviation for normalization (R, G, B).
    pub std: [f32; 3],
}

/// Turns a raw [`Image`] tile into the normalized tensor the model expects.
///
/// Holds the resolved [`ModelSpec`]. The transform (M1) is: optional RGBA→RGB,
/// resize to `spec.input_size`, center-crop, scale to [0,1], subtract `spec.mean`,
/// divide by `spec.std`, and emit CHW f32 — the standard CLIP/ViT preprocessing.
#[derive(Clone, Debug)]
pub struct Preprocessor {
    spec: ModelSpec,
}

impl Preprocessor {
    /// Construct a preprocessor for the given model family.
    #[must_use]
    pub fn new(kind: ModelKind) -> Self {
        Self { spec: kind.spec() }
    }

    /// The model spec this preprocessor enforces.
    #[must_use]
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Preprocess one tile into a flat CHW f32 tensor of length
    /// `3 * input_size * input_size`, ready to stack into an ONNX batch input.
    ///
    /// M1: resize + center-crop to `spec.input_size`, normalize with
    /// `spec.mean`/`spec.std`, lay out channel-first. Wires to `ndarray` (and an
    /// image-resize routine) in `onnx::OnnxEmbedder::embed_batch`.
    ///
    /// # Errors
    /// Returns [`EncoderError::Preprocess`] for unsupported formats or bad geometry.
    pub fn preprocess(&self, _tile: &Image) -> Result<Vec<f32>, EncoderError> {
        unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")
    }
}
