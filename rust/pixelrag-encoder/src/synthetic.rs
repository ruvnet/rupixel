//! Deterministic synthetic embedder (ADR-264 M1 plumbing validation).
//!
//! # HONESTY NOTE — read before using any metric this produces
//!
//! There is **no real Qwen3-VL-Embedding-2B** in this environment (weights + GPU are
//! blocked). The real encoder path lives in [`crate::onnx`] / [`crate::model`] and stays
//! an `unimplemented!("M1-real: needs Qwen3-VL-2B weights + ort")` stub.
//!
//! [`SyntheticEmbedder`] exists **only to exercise the pipeline plumbing** — tile →
//! embed → cache → index → search — without the 2B model. It maps tile *bytes* through
//! a tiny seeded PRNG into an L2-normalized f32 vector. The mapping is:
//!
//! - **deterministic** (same tile + same seed ⇒ identical vector, for reproducible runs),
//! - **content-sensitive** (different tile bytes ⇒ different vector, so the cache and
//!   index see distinct keys), and
//! - **completely non-semantic** — it encodes *no* visual meaning whatsoever.
//!
//! Therefore any recall / NDCG / MRR number computed over these embeddings measures
//! **plumbing correctness on a subset fixture, NOT semantic retrieval quality**. Real
//! recall requires Qwen3-VL-2B (blocked). Every bench metric MUST be labelled:
//!
//! > "subset fixture + synthetic embeddings — plumbing validation, NOT semantic
//! > retrieval quality; real recall requires Qwen3-VL-2B (blocked)".

use crate::{Embedder, EmbedderKind, Embedding, EncoderError, Image, PixelFormat};

/// The default embedding width for the synthetic embedder.
///
/// Kept small (128) so plumbing/index tests are cheap; the real encoders emit wider
/// vectors (1024 for Qwen3-VL, 768 for the CLIP surrogate — see [`crate::model`]).
pub const DEFAULT_SYNTHETIC_DIM: usize = 128;

/// Fixed default seed so synthetic runs are reproducible across machines.
pub const DEFAULT_SYNTHETIC_SEED: u64 = 0x5158_5658_5043_4452; // "QXVXPCDR" — arbitrary fixed salt

/// A deterministic, **non-semantic** embedder for pipeline plumbing validation.
///
/// See the module docs: this is NOT a real visual encoder. It hashes tile content into
/// a seeded PRNG and fills a fixed-width, L2-normalized f32 vector. Use it to validate
/// the render→embed→cache→index→search wiring while the real Qwen3-VL-2B path is blocked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyntheticEmbedder {
    dim: usize,
    seed: u64,
    normalize: bool,
}

impl Default for SyntheticEmbedder {
    /// 128-d, fixed default seed, L2-normalized (matching PixelRAG's normalized index).
    fn default() -> Self {
        Self {
            dim: DEFAULT_SYNTHETIC_DIM,
            seed: DEFAULT_SYNTHETIC_SEED,
            normalize: true,
        }
    }
}

impl SyntheticEmbedder {
    /// Construct with an explicit dimensionality, using the default seed and L2 norm.
    ///
    /// `dim` is clamped to at least 1 so the produced vector is always non-empty.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(1),
            ..Self::default()
        }
    }

    /// Construct with an explicit dimensionality **and** seed (for reproducible variants).
    #[must_use]
    pub fn with_seed(dim: usize, seed: u64) -> Self {
        Self {
            dim: dim.max(1),
            seed,
            normalize: true,
        }
    }

    /// Disable L2 normalization (off by default; the index contract wants normalized
    /// vectors, so this is only for plumbing experiments).
    #[must_use]
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// The fixed seed this embedder uses to derive vectors.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether outputs are L2-normalized.
    #[must_use]
    pub fn normalizes(&self) -> bool {
        self.normalize
    }

    /// Hash tile content (geometry + format + pixels) into a 64-bit seed via FNV-1a,
    /// mixed with this embedder's base `seed`.
    ///
    /// Deterministic and content-sensitive — the whole point of the synthetic path.
    fn content_seed(&self, tile: &Image) -> u64 {
        // FNV-1a 64-bit over a stable serialization of the tile.
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = FNV_OFFSET ^ self.seed;
        let mix = |b: u8, h: &mut u64| {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(FNV_PRIME);
        };
        // Geometry + format first so two same-byte buffers of different shape differ.
        for b in tile.width.to_le_bytes() {
            mix(b, &mut h);
        }
        for b in tile.height.to_le_bytes() {
            mix(b, &mut h);
        }
        mix(format_tag(tile.format), &mut h);
        for &b in &tile.pixels {
            mix(b, &mut h);
        }
        h
    }

    /// Produce the deterministic embedding for a single tile.
    fn embed_one(&self, tile: &Image) -> Embedding {
        let mut rng = SplitMix64::new(self.content_seed(tile));
        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            // Map the PRNG word into a symmetric f32 in roughly [-1, 1).
            vector.push(rng.next_unit_f32());
        }
        if self.normalize {
            l2_normalize(&mut vector);
        }
        Embedding {
            vector,
            normalized: self.normalize,
        }
    }
}

impl Embedder for SyntheticEmbedder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        // Synthetic: no real preprocessing/inference can fail, but keep the Result
        // contract so this is a drop-in for the real backends. Order matches `tiles`.
        Ok(tiles.iter().map(|t| self.embed_one(t)).collect())
    }

    fn kind(&self) -> EmbedderKind {
        EmbedderKind::Synthetic
    }
}

/// Stable single-byte tag for a [`PixelFormat`], folded into the content hash.
const fn format_tag(f: PixelFormat) -> u8 {
    match f {
        PixelFormat::Rgb8 => 1,
        PixelFormat::Rgba8 => 2,
        PixelFormat::Gray8 => 3,
    }
}

/// L2-normalize a vector in place. No-op (leaves zeros) if the norm is ~0.
fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > f32::EPSILON {
        let inv = 1.0 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Minimal SplitMix64 PRNG — std-only, deterministic, good enough for plumbing vectors.
///
/// This is NOT cryptographic and NOT a model; it only spreads a content seed into a
/// reproducible stream of f32 components.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next component as an f32 in roughly [-1, 1).
    fn next_unit_f32(&mut self) -> f32 {
        // Take the 24 high bits → exact float mantissa range [0, 1), then center to [-1, 1).
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        let u = (bits as f32) / 16_777_216.0_f32; // [0, 1)
        u * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(bytes: &[u8]) -> Image {
        Image {
            pixels: bytes.to_vec(),
            width: 2,
            height: 2,
            format: PixelFormat::Rgb8,
        }
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let e = SyntheticEmbedder::default();
        let a = e.embed(&tile(&[1, 2, 3, 4])).unwrap();
        let b = e.embed(&tile(&[1, 2, 3, 4])).unwrap();
        assert_eq!(a, b, "synthetic embedder must be deterministic");
        assert_eq!(a.dim(), DEFAULT_SYNTHETIC_DIM);
    }

    #[test]
    fn content_sensitive_different_input_different_output() {
        let e = SyntheticEmbedder::default();
        let a = e.embed(&tile(&[1, 2, 3, 4])).unwrap();
        let b = e.embed(&tile(&[4, 3, 2, 1])).unwrap();
        assert_ne!(a.vector, b.vector, "different tiles must map to different vectors");
    }

    #[test]
    fn output_is_l2_normalized_by_default() {
        let e = SyntheticEmbedder::default();
        let v = e.embed(&tile(&[9, 8, 7, 6, 5, 4])).unwrap();
        assert!(v.normalized);
        let norm: f32 = v.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
    }

    #[test]
    fn dim_is_configurable() {
        let e = SyntheticEmbedder::new(256);
        assert_eq!(e.embedding_dim(), 256);
        assert_eq!(e.embed(&tile(&[0, 1])).unwrap().dim(), 256);
    }

    #[test]
    fn seed_changes_vector() {
        let a = SyntheticEmbedder::with_seed(64, 1).embed(&tile(&[5, 5])).unwrap();
        let b = SyntheticEmbedder::with_seed(64, 2).embed(&tile(&[5, 5])).unwrap();
        assert_ne!(a.vector, b.vector, "different seeds must change the vector");
    }

    #[test]
    fn batch_matches_single() {
        let e = SyntheticEmbedder::default();
        let tiles = [tile(&[1]), tile(&[2]), tile(&[3])];
        let batch = e.embed_batch(&tiles).unwrap();
        for (i, t) in tiles.iter().enumerate() {
            assert_eq!(batch[i], e.embed(t).unwrap(), "batch order/content mismatch at {i}");
        }
    }
}
