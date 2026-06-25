//! LRU embedding cache for tiles (ADR-264 §M1: "Add embedding cache (LRU, ~100MB)").
//!
//! Encoding a screenshot tile through a ViT-scale model is the most expensive step in
//! the PixelRAG pipeline, and document corpora contain many repeated tiles (headers,
//! footers, blank margins, shared layout chrome). This module defines the
//! [`EmbeddingCache`] trait — keyed by [`TileKey`] (a content hash) — plus a
//! [`CachingEmbedder`] decorator that fronts any [`Embedder`] with a cache so repeated
//! tiles are encoded once.
//!
//! M1 plumbing: [`LruEmbeddingCache`] is a real, **std-only** LRU (a `HashMap` + a
//! monotonic access clock under a `Mutex` — no external `lru` crate). It enforces both a
//! byte budget and an entry-count cap, evicting the least-recently-used entry when over
//! budget. This is real enough to validate the cache-hit path in the pipeline; it is not
//! tuned for production throughput.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{Embedder, EmbedderKind, Embedding, EncoderError, Image, PixelFormat, TileKey};

/// A capacity-bounded, content-keyed cache of tile embeddings.
///
/// Implementors evict least-recently-used entries to stay within a memory budget
/// (`~100MB` per ADR-264 M1). Keyed by [`TileKey`], a stable hash of the tile pixels
/// so identical tiles across documents share one slot.
pub trait EmbeddingCache: Send + Sync {
    /// Look up a cached embedding, marking the entry most-recently-used on hit.
    fn get(&self, key: &TileKey) -> Option<Embedding>;

    /// Insert (or refresh) an embedding for `key`, evicting LRU entries if over budget.
    fn put(&self, key: TileKey, embedding: Embedding);

    /// Current number of cached entries.
    fn len(&self) -> usize;

    /// Whether the cache holds no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate resident size of the cache in bytes (for budget enforcement / metrics).
    fn approx_bytes(&self) -> usize;

    /// Drop all entries.
    fn clear(&self);
}

/// Sizing/eviction policy for [`LruEmbeddingCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    /// Soft memory budget in bytes (default derived from ~100MB per ADR-264 M1).
    pub max_bytes: usize,
    /// Hard cap on entry count (belt-and-suspenders alongside `max_bytes`).
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        // ADR-264 M1: LRU embedding cache ~100MB.
        Self {
            max_bytes: 100 * 1024 * 1024,
            max_entries: 100_000,
        }
    }
}

/// One stored entry: the embedding plus its last-access tick and byte cost.
#[derive(Clone, Debug)]
struct Entry {
    embedding: Embedding,
    last_access: u64,
    bytes: usize,
}

/// Interior mutable state for [`LruEmbeddingCache`], guarded by a single `Mutex`.
#[derive(Default)]
struct Inner {
    map: HashMap<TileKey, Entry>,
    /// Monotonic logical clock; each access/insert bumps it. Lowest value == LRU.
    clock: u64,
    bytes: usize,
}

/// Default LRU implementation of [`EmbeddingCache`] — std-only.
///
/// Backed by a `HashMap<TileKey, Entry>` plus a monotonic access clock under one
/// `Mutex`. `get` promotes the hit to most-recently-used by stamping it with the latest
/// clock value; `put` evicts the entry with the smallest stamp until both the byte
/// budget and entry cap are satisfied. O(n) eviction scan is acceptable for the M1
/// plumbing cache; a heap/intrusive-list upgrade is a later optimization.
pub struct LruEmbeddingCache {
    config: CacheConfig,
    inner: Mutex<Inner>,
}

impl LruEmbeddingCache {
    /// Create a cache with the given config.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Create a cache with [`CacheConfig::default`] (~100MB, ADR-264 M1).
    #[must_use]
    pub fn with_default_budget() -> Self {
        Self::new(CacheConfig::default())
    }

    /// The configuration this cache was built with.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Approximate resident byte cost of one embedding entry: the f32 vector payload
    /// plus the fixed [`TileKey`] + bookkeeping overhead.
    fn entry_bytes(embedding: &Embedding) -> usize {
        embedding.vector.len() * std::mem::size_of::<f32>()
            + std::mem::size_of::<TileKey>()
            + std::mem::size_of::<Entry>()
    }

    /// Evict least-recently-used entries until within both budgets. Caller holds the lock.
    fn evict_to_budget(&self, inner: &mut Inner) {
        while inner.bytes > self.config.max_bytes || inner.map.len() > self.config.max_entries {
            // Find the entry with the smallest access stamp (the LRU one).
            let victim = inner
                .map
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    if let Some(e) = inner.map.remove(&k) {
                        inner.bytes = inner.bytes.saturating_sub(e.bytes);
                    }
                }
                None => break, // empty map but still over a (zero) budget — nothing to evict
            }
        }
    }
}

impl EmbeddingCache for LruEmbeddingCache {
    fn get(&self, key: &TileKey) -> Option<Embedding> {
        let mut inner = self.inner.lock().expect("embedding cache mutex poisoned");
        inner.clock += 1;
        let tick = inner.clock;
        if let Some(entry) = inner.map.get_mut(key) {
            entry.last_access = tick; // promote to most-recently-used
            Some(entry.embedding.clone())
        } else {
            None
        }
    }

    fn put(&self, key: TileKey, embedding: Embedding) {
        let bytes = Self::entry_bytes(&embedding);
        let mut inner = self.inner.lock().expect("embedding cache mutex poisoned");
        inner.clock += 1;
        let tick = inner.clock;
        // Replace existing: subtract its old byte cost first.
        if let Some(old) = inner.map.remove(&key) {
            inner.bytes = inner.bytes.saturating_sub(old.bytes);
        }
        inner.bytes += bytes;
        inner.map.insert(
            key,
            Entry {
                embedding,
                last_access: tick,
                bytes,
            },
        );
        self.evict_to_budget(&mut inner);
    }

    fn len(&self) -> usize {
        self.inner.lock().expect("embedding cache mutex poisoned").map.len()
    }

    fn approx_bytes(&self) -> usize {
        self.inner.lock().expect("embedding cache mutex poisoned").bytes
    }

    fn clear(&self) {
        let mut inner = self.inner.lock().expect("embedding cache mutex poisoned");
        inner.map.clear();
        inner.bytes = 0;
        // Clock intentionally left monotonic so post-clear stamps stay ordered.
    }
}

/// An [`Embedder`] decorator that consults an [`EmbeddingCache`] before encoding.
///
/// Wraps any inner embedder + any cache. On `embed_batch`, it hashes each tile to a
/// [`TileKey`], serves cache hits directly, encodes only the misses through the inner
/// embedder, and back-fills the cache. This is what `pixelrag-core` actually holds so
/// the cache is transparent to the pipeline.
pub struct CachingEmbedder<E: Embedder, C: EmbeddingCache> {
    inner: E,
    cache: C,
}

impl<E: Embedder, C: EmbeddingCache> CachingEmbedder<E, C> {
    /// Wrap `inner` with `cache`.
    #[must_use]
    pub fn new(inner: E, cache: C) -> Self {
        Self { inner, cache }
    }

    /// Borrow the wrapped embedder.
    #[must_use]
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Borrow the cache.
    #[must_use]
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Compute the stable content key for a tile.
    ///
    /// M1 plumbing: a std-only FNV-1a 64-bit hash over the tile geometry, format, and
    /// pixels, expanded into a 32-byte [`TileKey`] via SplitMix64. Identical tiles map to
    /// identical keys so they share one cache slot. (A real build would use blake3/xxhash
    /// plus the preprocessing params; this is sufficient for the plumbing path.)
    fn tile_key(tile: &Image) -> TileKey {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = FNV_OFFSET;
        let mix = |b: u8, h: &mut u64| {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(FNV_PRIME);
        };
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
        // Expand the 64-bit digest into 32 bytes deterministically.
        let mut state = h;
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
        TileKey(out)
    }
}

impl<E: Embedder, C: EmbeddingCache> Embedder for CachingEmbedder<E, C> {
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    fn embed_batch(&self, tiles: &[Image]) -> Result<Vec<Embedding>, EncoderError> {
        // Key each tile, serve hits from the cache, encode only the misses, back-fill,
        // then reassemble outputs in the original tile order.
        let keys: Vec<TileKey> = tiles.iter().map(Self::tile_key).collect();

        // Slots hold either a hit (Some) or a placeholder to fill from miss results.
        let mut slots: Vec<Option<Embedding>> = Vec::with_capacity(tiles.len());
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_tiles: Vec<Image> = Vec::new();

        for (i, key) in keys.iter().enumerate() {
            match self.cache.get(key) {
                Some(hit) => slots.push(Some(hit)),
                None => {
                    slots.push(None);
                    miss_indices.push(i);
                    miss_tiles.push(tiles[i].clone());
                }
            }
        }

        if !miss_tiles.is_empty() {
            let encoded = self.inner.embed_batch(&miss_tiles)?;
            if encoded.len() != miss_tiles.len() {
                return Err(EncoderError::Inference(format!(
                    "inner embedder returned {} embeddings for {} tiles",
                    encoded.len(),
                    miss_tiles.len()
                )));
            }
            for (slot_idx, embedding) in miss_indices.into_iter().zip(encoded) {
                self.cache.put(keys[slot_idx].clone(), embedding.clone());
                slots[slot_idx] = Some(embedding);
            }
        }

        // All slots are now Some; collect in order.
        slots
            .into_iter()
            .map(|s| s.ok_or(EncoderError::EmptyBatch))
            .collect()
    }

    fn kind(&self) -> EmbedderKind {
        self.inner.kind()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::SyntheticEmbedder;

    fn tile(bytes: &[u8]) -> Image {
        Image {
            pixels: bytes.to_vec(),
            width: 2,
            height: 2,
            format: PixelFormat::Rgb8,
        }
    }

    fn embedding(dim: usize) -> Embedding {
        Embedding {
            vector: vec![0.1; dim],
            normalized: false,
        }
    }

    #[test]
    fn put_get_roundtrip() {
        let cache = LruEmbeddingCache::with_default_budget();
        let key = CachingEmbedder::<SyntheticEmbedder, LruEmbeddingCache>::tile_key(&tile(&[1, 2]));
        assert!(cache.get(&key).is_none());
        cache.put(key.clone(), embedding(8));
        assert_eq!(cache.get(&key), Some(embedding(8)));
        assert_eq!(cache.len(), 1);
        assert!(cache.approx_bytes() > 0);
    }

    #[test]
    fn evicts_lru_on_entry_cap() {
        let cache = LruEmbeddingCache::new(CacheConfig {
            max_bytes: usize::MAX,
            max_entries: 2,
        });
        let k = |n: u8| TileKey([n; 32]);
        cache.put(k(1), embedding(4));
        cache.put(k(2), embedding(4));
        // Touch k(1) so k(2) becomes LRU.
        let _ = cache.get(&k(1));
        cache.put(k(3), embedding(4)); // over cap → evict LRU (k2)
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&k(2)).is_none(), "k2 should have been evicted");
        assert!(cache.get(&k(1)).is_some());
        assert!(cache.get(&k(3)).is_some());
    }

    #[test]
    fn clear_resets() {
        let cache = LruEmbeddingCache::with_default_budget();
        cache.put(TileKey([7; 32]), embedding(4));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.approx_bytes(), 0);
    }

    #[test]
    fn caching_embedder_serves_hits_and_preserves_order() {
        let inner = SyntheticEmbedder::new(16);
        let cache = LruEmbeddingCache::with_default_budget();
        let caching = CachingEmbedder::new(inner, cache);

        let tiles = [tile(&[1]), tile(&[2]), tile(&[1])]; // index 0 and 2 identical
        let out1 = caching.embed_batch(&tiles).unwrap();
        // The two identical tiles must yield identical embeddings.
        assert_eq!(out1[0], out1[2]);
        // And match the bare synthetic embedder (order preserved).
        let bare = SyntheticEmbedder::new(16);
        assert_eq!(out1[0], bare.embed(&tile(&[1])).unwrap());
        assert_eq!(out1[1], bare.embed(&tile(&[2])).unwrap());

        // Second pass: everything is a cache hit; results identical.
        let out2 = caching.embed_batch(&tiles).unwrap();
        assert_eq!(out1, out2);
        // Two distinct tiles cached.
        assert_eq!(caching.cache().len(), 2);
    }

    #[test]
    fn identical_tiles_share_key() {
        type CE = CachingEmbedder<SyntheticEmbedder, LruEmbeddingCache>;
        let a = CE::tile_key(&tile(&[9, 9, 9]));
        let b = CE::tile_key(&tile(&[9, 9, 9]));
        let c = CE::tile_key(&tile(&[9, 9, 8]));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
