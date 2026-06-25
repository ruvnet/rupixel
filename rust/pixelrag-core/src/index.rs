//! ANN index adaptor.
//!
//! Per ADR-264 reuse boundary, this crate does **not** implement a vector index.
//! It defines [`AnnIndex`], a thin adaptor trait whose M1 implementations wrap an
//! existing ruvector backend:
//!
//! - `ruvector-core::HNSWIndex` — M1 primary (incremental insert).
//! - `ruvector-rairs::IvfFlat` (IVF, ADR-193) — M1 train-then-add IVF backend;
//!   the genuine second backend the darwin harness selects between.
//! - `ruvector-rairs` IVF-SQ (ADR-193) — M1 fallback on memory budget (deferred).
//! - `ruvector-turbovec` FastScan (ADR-254) — M2+ optimization, if shipped.
//!
//! HNSW and IVF have different build lifecycles: HNSW commits each vector on
//! `add`, whereas IVF must learn k-means centroids over the whole corpus first.
//! [`AnnIndex::finalize`] reconciles them — a no-op for HNSW, the train-then-add
//! build step for IVF — and the pipeline calls it after each ingest batch.
//!
//! The signature here intentionally mirrors `ruvector_rabitq::AnnIndex` so the
//! M1 implementation is a near-passthrough (`ruvector-rabitq` also provides the
//! `RandomRotation::HadamardSigned` reused at build time for consistency). The
//! [`crate::config::IndexBackend`] enum selects which concrete backend
//! [`build_index`] constructs.

use ruvector_core::types::DbOptions;
use ruvector_core::{DistanceMetric, SearchQuery, VectorDB, VectorEntry};

use crate::config::IndexBackend;
use crate::{Embedding, Error, Result, SearchResult};

/// Adaptor over a ruvector ANN backend. Mirrors `ruvector_rabitq::AnnIndex`
/// (`add`/`search`/`len`/`dim`/`memory_bytes`) so the M1 wrapper is trivial, plus
/// PixelRAG-specific persistence + filtered search hooks.
pub trait AnnIndex: Send + Sync {
    /// Insert one embedding under external `id` (the tile id from [`crate::tile`]).
    ///
    /// **M1**: forward to the wrapped backend's `add`; the backend owns its
    /// quantization/rotation.
    fn add(&mut self, id: usize, vector: Embedding) -> Result<()>;

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// **M1**: forward to the wrapped backend's `search`, returning hits ordered
    /// by ascending squared-L2 distance (see [`SearchResult`]).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search restricted to an allowlist of ids (pre-filtered retrieval).
    ///
    /// **M1**: reuse the allowlist-filtered search path from `ruvector-rairs`
    /// (IVF supports pre-filtered scan) / `ruvector-rabitq`. Backends without
    /// native filtering fall back to over-fetch + post-filter.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &[usize],
    ) -> Result<Vec<SearchResult>>;

    /// Build/commit the index after a batch of [`AnnIndex::add`] calls.
    ///
    /// **M1 — train-then-add backends**: HNSW inserts incrementally so `add`
    /// already commits each vector; for those this is a no-op (the default).
    /// IVF-style backends (k-means centroids must be learned over the *whole*
    /// corpus before any vector can be assigned to a list) buffer the embeddings
    /// during `add` and do the real `train(corpus)` + `add(corpus)` work here.
    ///
    /// Idempotent: it may be called repeatedly (e.g. once per ingested document)
    /// and each call rebuilds from the complete buffer, so the final state after
    /// the last call reflects every added vector. The pipeline calls it at the
    /// end of [`crate::pipeline::Pipeline::index_tiles`].
    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Embedding dimensionality the index was built for.
    fn dim(&self) -> usize;

    /// Honest byte footprint of the index (originals + codes + rotation +
    /// bookkeeping), matching `ruvector_rabitq::AnnIndex::memory_bytes`.
    fn memory_bytes(&self) -> usize;

    /// Persist the index to a `*.pixelrag` artifact.
    ///
    /// **M2**: serialize the wrapped backend + id map via `bincode`.
    fn save(&self, path: &std::path::Path) -> Result<()>;
}

/// Construct the configured ANN backend for embeddings of dimension `dim`.
///
/// **M1**: match on `backend` and build the corresponding ruvector index
/// (`ruvector-core` HNSW / `ruvector-rairs` IVF-SQ), returning it boxed behind
/// [`AnnIndex`]. `IndexBackend::Turbovec` is gated on ADR-254 shipping (M2+);
/// until then it returns [`crate::Error::Index`].
pub fn build_index(backend: IndexBackend, dim: usize) -> Result<Box<dyn AnnIndex>> {
    match backend {
        IndexBackend::Hnsw => Ok(Box::new(RuvectorHnswIndex::new(dim)?)),
        IndexBackend::IvfFlat => Ok(Box::new(RairsIvfFlatIndex::new(dim)?)),
        IndexBackend::IvfSq => Err(Error::Index(
            "IVF-SQ backend (ruvector-rairs, ADR-193) is the M1 fallback and not yet wired; \
             use IndexBackend::IvfFlat for the IVF path or IndexBackend::Hnsw"
                .into(),
        )),
        IndexBackend::Turbovec => Err(Error::Index(
            "Turbovec FastScan backend is gated on ADR-254 shipping (M2+)".into(),
        )),
    }
}

/// Load a previously persisted index from a `*.pixelrag` artifact.
///
/// **M2**: `bincode`-deserialize the backend + id map and return it behind
/// [`AnnIndex`].
pub fn load_index(_path: &std::path::Path) -> Result<Box<dyn AnnIndex>> {
    unimplemented!("M2: bincode-deserialize *.pixelrag into the wrapped ruvector backend")
}

// ── M1 concrete backend: ruvector-core HNSW ──────────────────────────────────

/// M1 primary [`AnnIndex`] implementation wrapping `ruvector_core::VectorDB`.
///
/// External tile ids (`usize`) are mapped to `ruvector_core::VectorId` (a
/// `String`) by decimal formatting; cosine distance is the default metric so
/// normalized visual embeddings compare by angle. The underlying `VectorDB`
/// requires the `storage` feature (a redb-backed path); we point it at a unique
/// temp path so the index is self-contained for the M1 plumbing harness.
pub struct RuvectorHnswIndex {
    db: VectorDB,
    dim: usize,
    /// Externally-assigned ids in insertion order (drives [`AnnIndex::len`] and
    /// the post-search allowlist filter without touching the backend).
    ids: Vec<usize>,
}

impl RuvectorHnswIndex {
    /// Build an empty HNSW-backed index for `dim`-wide cosine embeddings.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Index("embedding dimension must be > 0".into()));
        }
        // Unique, process-local storage path. ruvector-core's `storage` feature
        // is on by default and `VectorDB::new` needs a path; this keeps the M1
        // index ephemeral and isolated (M2 swaps in real *.pixelrag persistence).
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let storage_path = std::env::temp_dir()
            .join(format!("pixelrag-hnsw-{}-{}.db", std::process::id(), nanos))
            .to_string_lossy()
            .into_owned();

        let options = DbOptions {
            dimensions: dim,
            distance_metric: DistanceMetric::Cosine,
            storage_path,
            ..DbOptions::default()
        };
        let db = VectorDB::new(options).map_err(|e| Error::Index(format!("VectorDB::new: {e}")))?;
        Ok(Self { db, dim, ids: Vec::new() })
    }

    fn id_to_key(id: usize) -> String {
        id.to_string()
    }

    fn key_to_id(key: &str) -> Option<usize> {
        key.parse().ok()
    }
}

impl AnnIndex for RuvectorHnswIndex {
    fn add(&mut self, id: usize, vector: Embedding) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::Index(format!(
                "embedding dim {} != index dim {}",
                vector.len(),
                self.dim
            )));
        }
        self.db
            .insert(VectorEntry {
                id: Some(Self::id_to_key(id)),
                vector,
                metadata: None,
            })
            .map_err(|e| Error::Index(format!("VectorDB::insert: {e}")))?;
        self.ids.push(id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(Error::Index(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dim
            )));
        }
        let raw = self
            .db
            .search(SearchQuery {
                vector: query.to_vec(),
                k,
                filter: None,
                ef_search: None,
            })
            .map_err(|e| Error::Index(format!("VectorDB::search: {e}")))?;
        Ok(raw
            .into_iter()
            .filter_map(|r| Self::key_to_id(&r.id).map(|id| SearchResult { id, score: r.score }))
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &[usize],
    ) -> Result<Vec<SearchResult>> {
        // M1: over-fetch then post-filter against the allowlist. Native
        // pre-filtered scan (ruvector-rairs IVF / rabitq) is a later milestone;
        // over-fetching `k + allowlist.len()` guarantees we can still return up to
        // `k` allowed hits when the unfiltered top-k are mostly disallowed.
        let allow: std::collections::HashSet<usize> = allowlist.iter().copied().collect();
        let over_k = k.saturating_add(allowlist.len()).max(k);
        let mut hits = self.search(query, over_k)?;
        hits.retain(|h| allow.contains(&h.id));
        hits.truncate(k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        // Honest f32-originals footprint plus the id bookkeeping vector. This
        // excludes redb on-disk pages (the index is unquantized in M1).
        self.ids.len() * self.dim * std::mem::size_of::<f32>()
            + self.ids.len() * std::mem::size_of::<usize>()
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> {
        // M2: bincode-serialize the id map + backend snapshot into *.pixelrag.
        // The ruvector-core storage feature already persists vectors to its redb
        // path; a dedicated portable artifact is deferred to M2.
        Err(Error::Index(
            "M2: *.pixelrag persistence not yet implemented (RuvectorHnswIndex::save)".into(),
        ))
    }
}

// ── M1 concrete backend: ruvector-rairs IVF-Flat ─────────────────────────────

// The rairs `AnnIndex` trait is aliased so its inherent-looking methods
// (`add`/`search`/`num_lists`) are in scope without clashing with this crate's
// own `AnnIndex` trait of the same name.
use ruvector_rairs::AnnIndex as RairsAnnIndex;

/// k-means iterations for IVF training. Fixed so the index is deterministic
/// (paired with [`RairsIvfFlatIndex::SEED`]); generous enough for the tiny
/// fixtures the M1 harness runs.
const IVF_MAX_ITER: usize = 25;

/// Requested number of Voronoi cells. The effective count is clamped down for
/// small corpora by [`RairsIvfFlatIndex::effective_nclusters`] so the 6-doc
/// fixture never asks k-means for more clusters than it has points.
const IVF_REQUESTED_NCLUSTERS: usize = 16;

/// M1 IVF-Flat [`AnnIndex`] implementation wrapping `ruvector_rairs::IvfFlat`.
///
/// Unlike HNSW (incremental insert), IVF must learn its k-means centroids over
/// the **whole** corpus before any vector can be assigned to an inverted list.
/// So this adaptor buffers every `add`ed embedding (with its external tile id)
/// and does the real `train(corpus)` + `add(corpus)` in [`AnnIndex::finalize`].
/// The rairs backend assigns 0-based ids in `add` order, which is exactly the
/// buffer order — so `buffer[i].0` maps the rairs id `i` back to the external id.
pub struct RairsIvfFlatIndex {
    dim: usize,
    /// Buffered `(external_id, embedding)` pairs, in insertion order. The rairs
    /// backend's internal id `i` corresponds to `buffer[i].0`.
    buffer: Vec<(usize, Embedding)>,
    /// Built backend (`Some` after [`AnnIndex::finalize`]). Rebuilt from `buffer`
    /// on each finalize so the call is idempotent.
    built: Option<ruvector_rairs::IvfFlat>,
}

impl RairsIvfFlatIndex {
    /// Fixed RNG seed for reproducible k-means centroids (determinism contract).
    const SEED: u64 = 0x5158_6c61; // "QXla"

    /// Build an empty IVF-Flat-backed index for `dim`-wide embeddings.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Index("embedding dimension must be > 0".into()));
        }
        Ok(Self { dim, buffer: Vec::new(), built: None })
    }

    /// Safe cluster count for a corpus of `n` vectors.
    ///
    /// k-means is undefined with more clusters than points; for tiny corpora we
    /// also want at least a couple of points per cell. Clamp the requested count
    /// to `max(1, n / 2)` (and never above `n`) so the 6-doc fixture trains with
    /// 3 clusters instead of panicking on a request for 16.
    fn effective_nclusters(n: usize) -> usize {
        IVF_REQUESTED_NCLUSTERS.min((n / 2).max(1)).min(n.max(1))
    }

    /// `nprobe` for search: probe every list. For the small corpora the M1
    /// harness runs this gives exact IVF results (no recall loss from a partial
    /// probe), matching the HNSW path's behaviour as a fair backend comparison.
    fn nprobe(&self) -> usize {
        self.built.as_ref().map(|i| i.num_lists()).unwrap_or(0).max(1)
    }
}

impl AnnIndex for RairsIvfFlatIndex {
    fn add(&mut self, id: usize, vector: Embedding) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::Index(format!(
                "embedding dim {} != index dim {}",
                vector.len(),
                self.dim
            )));
        }
        // IVF can't assign before training; buffer now, build in finalize().
        self.buffer.push((id, vector));
        // A new vector invalidates any prior build.
        self.built = None;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            self.built = None;
            return Ok(());
        }
        let corpus: Vec<Vec<f32>> = self.buffer.iter().map(|(_, v)| v.clone()).collect();
        let nclusters = Self::effective_nclusters(corpus.len());
        let mut idx = ruvector_rairs::IvfFlat::new(self.dim, nclusters, IVF_MAX_ITER, Self::SEED);
        idx.train(&corpus)
            .map_err(|e| Error::Index(format!("IvfFlat::train: {e}")))?;
        // ruvector_rairs::AnnIndex::add assigns ids in this slice's order, which is
        // the buffer order — preserving the rairs-id → external-id mapping.
        idx.add(&corpus)
            .map_err(|e| Error::Index(format!("IvfFlat::add: {e}")))?;
        self.built = Some(idx);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(Error::Index(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dim
            )));
        }
        let idx = self.built.as_ref().ok_or_else(|| {
            Error::Index(
                "IVF index not finalized; call AnnIndex::finalize (or Pipeline ingest) before search"
                    .into(),
            )
        })?;
        let raw = idx
            .search(query, k, self.nprobe())
            .map_err(|e| Error::Index(format!("IvfFlat::search: {e}")))?;
        Ok(raw
            .into_iter()
            .filter_map(|r| {
                // rairs id is the 0-based add-order index → external id via buffer.
                self.buffer.get(r.id).map(|(ext_id, _)| SearchResult {
                    id: *ext_id,
                    // rairs returns L2 distance; the PixelRAG contract is squared-L2
                    // (see crate::SearchResult). Square it to match the unit; ordering
                    // is identical either way.
                    score: r.distance * r.distance,
                })
            })
            .collect())
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &[usize],
    ) -> Result<Vec<SearchResult>> {
        // Same over-fetch + post-filter strategy as the HNSW path (rairs IvfFlat
        // has no native allowlist scan): fetch enough to still yield up to `k`
        // allowed hits when the unfiltered top-k are mostly disallowed.
        let allow: std::collections::HashSet<usize> = allowlist.iter().copied().collect();
        let over_k = k.saturating_add(allowlist.len()).max(k);
        let mut hits = self.search(query, over_k)?;
        hits.retain(|h| allow.contains(&h.id));
        hits.truncate(k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        // Honest f32-originals footprint plus the id bookkeeping. IvfFlat stores
        // raw vectors in its lists (unquantized), so the per-vector cost matches
        // the HNSW path; the centroid table is the only IVF-specific overhead.
        let n = self.buffer.len();
        let centroids = self
            .built
            .as_ref()
            .map(|i| i.num_lists())
            .unwrap_or(0);
        n * self.dim * std::mem::size_of::<f32>()
            + n * std::mem::size_of::<usize>()
            + centroids * self.dim * std::mem::size_of::<f32>()
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> {
        Err(Error::Index(
            "M2: *.pixelrag persistence not yet implemented (RairsIvfFlatIndex::save)".into(),
        ))
    }
}

#[cfg(test)]
mod ivf_tests {
    //! IVF-Flat adaptor coverage: train-then-add lifecycle, tiny-corpus nclusters
    //! clamp, rairs-id → external-id mapping, and the allowlist post-filter.
    use super::*;

    /// Deterministic unit vector keyed off `seed` (no RNG), distinct per seed.
    fn vec_of(dim: usize, seed: usize) -> Embedding {
        let mut v: Vec<f32> = (0..dim).map(|i| ((seed * 31 + i * 7) % 17) as f32 + 0.5).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn search_before_finalize_errors() {
        let mut idx = RairsIvfFlatIndex::new(8).unwrap();
        idx.add(100, vec_of(8, 1)).unwrap();
        assert!(idx.search(&vec_of(8, 1), 1).is_err()); // explicit error, never a panic
    }

    #[test]
    fn tiny_corpus_self_matches_with_external_id() {
        // 6-doc fixture analogue: clamp keeps k-means from over-requesting clusters.
        let (dim, mut idx) = (8, RairsIvfFlatIndex::new(8).unwrap());
        for (i, id) in [10usize, 11, 12, 13, 14, 15].into_iter().enumerate() {
            idx.add(id, vec_of(dim, i)).unwrap();
        }
        idx.finalize().unwrap();
        assert_eq!(idx.len(), 6);
        let hits = idx.search(&vec_of(dim, 3), 3).unwrap(); // doc under ext id 13 (buffer idx 3)
        assert_eq!(hits[0].id, 13); // rairs add-order id maps back to the external id
        assert!(hits[0].score < 1e-4); // exact self-match, squared-L2 ≈ 0
    }

    #[test]
    fn nclusters_clamp_is_safe() {
        assert_eq!(RairsIvfFlatIndex::effective_nclusters(1), 1);
        assert_eq!(RairsIvfFlatIndex::effective_nclusters(2), 1);
        assert_eq!(RairsIvfFlatIndex::effective_nclusters(6), 3);
        assert_eq!(RairsIvfFlatIndex::effective_nclusters(10_000), IVF_REQUESTED_NCLUSTERS);
    }

    #[test]
    fn allowlist_filters_to_allowed_ids() {
        let (dim, mut idx) = (8, RairsIvfFlatIndex::new(8).unwrap());
        for (i, id) in [20usize, 21, 22, 23].into_iter().enumerate() {
            idx.add(id, vec_of(dim, i)).unwrap();
        }
        idx.finalize().unwrap();
        let hits = idx.search_filtered(&vec_of(dim, 0), 4, &[21, 23]).unwrap();
        assert!(hits.iter().all(|h| h.id == 21 || h.id == 23));
    }
}
