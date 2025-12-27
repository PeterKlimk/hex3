//! Zero-cost timing instrumentation for knn_clipping.
//!
//! When the `timing` feature is enabled, this module provides timing
//! infrastructure that measures and reports phase durations.
//! When disabled, all timing code compiles away to nothing.
//!
//! Usage:
//!   cargo run --release --features timing

use std::time::Duration;

#[cfg(feature = "timing")]
use rustc_hash::FxHashMap;

/// Histogram of neighbors processed at termination.
/// Buckets 0-47: exact counts for 1-48 neighbors (bucket i = i+1 neighbors)
/// Bucket 48: 49-64 neighbors
/// Bucket 49: 65-96 neighbors
/// Bucket 50: 97+ neighbors
#[cfg(feature = "timing")]
pub const NEIGHBOR_HIST_BUCKETS: usize = 51;

/// K-NN stage that a cell terminated at.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnnCellStage {
    /// Terminated during resume stage with given K value
    Resume(usize),
    /// Terminated during restart stage with given K value
    Restart(usize),
    /// Ran full scan as fallback
    FullScanFallback,
    /// Ran full scan for dead-cell recovery
    FullScanRecovery,
}

/// Sub-phase timings within cell construction.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct CellSubPhases {
    pub knn_query: Duration,
    pub packed_knn: Duration,
    pub clipping: Duration,
    pub certification: Duration,
    /// Live-dedup: per-vertex ownership checks and shard-local dedup work during cell build.
    pub key_dedup: Duration,
    /// Per-cell k-NN stage distribution (final stage used per cell).
    pub stage_counts: FxHashMap<KnnCellStage, u64>,
    /// Cells where the k-NN search loop exhausted (typically means it hit brute force).
    pub cells_knn_exhausted: u64,
    /// Histogram of neighbors processed at termination.
    pub neighbors_histogram: [u64; NEIGHBOR_HIST_BUCKETS],
}

#[cfg(feature = "timing")]
impl Default for CellSubPhases {
    fn default() -> Self {
        Self {
            knn_query: Duration::ZERO,
            packed_knn: Duration::ZERO,
            clipping: Duration::ZERO,
            certification: Duration::ZERO,
            key_dedup: Duration::ZERO,
            stage_counts: FxHashMap::default(),
            cells_knn_exhausted: 0,
            neighbors_histogram: [0; NEIGHBOR_HIST_BUCKETS],
        }
    }
}

/// Sub-phase timings within dedup.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct DedupSubPhases {
    /// Allocation and setup time.
    pub setup: Duration,
    /// Hash lookup and vertex deduplication.
    pub lookup: Duration,
    /// Per-cell index deduplication.
    pub cell_dedup: Duration,
    /// Live-dedup: overflow bucketing before flush.
    pub overflow_collect: Duration,
    /// Live-dedup: overflow flush into owner shards.
    pub overflow_flush: Duration,
    /// Live-dedup: concatenating shard vertex buffers.
    pub concat_vertices: Duration,
    /// Live-dedup: emitting cells in generator order (includes per-cell index dedup).
    pub emit_cells: Duration,
    /// Number of triplet keys processed.
    pub triplet_keys: u64,
    /// Number of support keys processed.
    pub support_keys: u64,
    /// Number of duplicate vertex indices removed during per-cell dedup.
    pub cell_dupes_removed: u64,
}

/// Dummy dedup sub-phases when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct DedupSubPhases;

/// Phase timings for the Voronoi algorithm.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct PhaseTimings {
    pub total: Duration,
    pub knn_build: Duration,
    pub cell_construction: Duration,
    pub cell_sub: CellSubPhases,
    pub dedup: Duration,
    pub dedup_sub: DedupSubPhases,
    pub assemble: Duration,
}

#[cfg(feature = "timing")]
impl PhaseTimings {
    pub fn report(&self, n: usize) {
        let total_ms = self.total.as_secs_f64() * 1000.0;
        let pct = |d: Duration| {
            if self.total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / self.total.as_secs_f64() * 100.0
            }
        };

        eprintln!("[timing] knn_clipping n={}", n);
        eprintln!(
            "  knn_build:         {:7.1}ms ({:4.1}%)",
            self.knn_build.as_secs_f64() * 1000.0,
            pct(self.knn_build)
        );
        eprintln!(
            "  cell_construction: {:7.1}ms ({:4.1}%)",
            self.cell_construction.as_secs_f64() * 1000.0,
            pct(self.cell_construction)
        );

        // Sub-phase breakdown: estimate wall time from CPU time using parent ratio
        let cpu_total = self.cell_sub.knn_query
            + self.cell_sub.packed_knn
            + self.cell_sub.clipping
            + self.cell_sub.certification
            + self.cell_sub.key_dedup;
        let cpu_total_secs = cpu_total.as_secs_f64();
        let wall_secs = self.cell_construction.as_secs_f64();

        // Ratio to convert CPU time to estimated wall time
        let cpu_to_wall = if cpu_total_secs > 0.0 {
            wall_secs / cpu_total_secs
        } else {
            1.0
        };
        let parallelism = if wall_secs > 0.0 {
            cpu_total_secs / wall_secs
        } else {
            1.0
        };

        let sub_pct = |d: Duration| {
            if cpu_total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / cpu_total_secs * 100.0
            }
        };
        let est_wall_ms = |d: Duration| d.as_secs_f64() * cpu_to_wall * 1000.0;

        eprintln!(
            "    knn_query:       {:7.1}ms ({:4.1}%)",
            est_wall_ms(self.cell_sub.knn_query),
            sub_pct(self.cell_sub.knn_query)
        );
        if self.cell_sub.packed_knn.as_nanos() > 0 {
            eprintln!(
                "    packed_knn:      {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.packed_knn),
                sub_pct(self.cell_sub.packed_knn)
            );
        }
        eprintln!(
            "    clipping:        {:7.1}ms ({:4.1}%)",
            est_wall_ms(self.cell_sub.clipping),
            sub_pct(self.cell_sub.clipping)
        );
        eprintln!(
            "    certification:   {:7.1}ms ({:4.1}%)",
            est_wall_ms(self.cell_sub.certification),
            sub_pct(self.cell_sub.certification)
        );
        if self.cell_sub.key_dedup.as_nanos() > 0 {
            eprintln!(
                "    key_dedup:       {:7.1}ms ({:4.1}%)",
                est_wall_ms(self.cell_sub.key_dedup),
                sub_pct(self.cell_sub.key_dedup)
            );
        }
        eprintln!("    ({:.1}x parallelism)", parallelism);

        // Collect and sort stage counts for display
        let total_cells: u64 = self.cell_sub.stage_counts.values().sum::<u64>().max(1);
        let pct_cells = |c: u64| c as f64 / total_cells as f64 * 100.0;

        // Separate resume, restart, and special stages
        let mut resume_stages: Vec<_> = self
            .cell_sub
            .stage_counts
            .iter()
            .filter_map(|(k, &v)| match k {
                KnnCellStage::Resume(n) => Some((*n, v)),
                _ => None,
            })
            .collect();
        resume_stages.sort_by_key(|(k, _)| *k);

        let mut restart_stages: Vec<_> = self
            .cell_sub
            .stage_counts
            .iter()
            .filter_map(|(k, &v)| match k {
                KnnCellStage::Restart(n) => Some((*n, v)),
                _ => None,
            })
            .collect();
        restart_stages.sort_by_key(|(k, _)| *k);

        let full_scan = self
            .cell_sub
            .stage_counts
            .get(&KnnCellStage::FullScanFallback)
            .copied()
            .unwrap_or(0);
        let recovery = self
            .cell_sub
            .stage_counts
            .get(&KnnCellStage::FullScanRecovery)
            .copied()
            .unwrap_or(0);

        // Build output string
        let mut stages_str = String::from("    knn_stages:");
        for (k, count) in &resume_stages {
            stages_str.push_str(&format!(" k{}={} ({:.1}%)", k, count, pct_cells(*count)));
        }
        for (k, count) in &restart_stages {
            stages_str.push_str(&format!(" K{}={} ({:.1}%)", k, count, pct_cells(*count)));
        }
        if full_scan > 0 {
            stages_str.push_str(&format!(" full_scan={} ({:.1}%)", full_scan, pct_cells(full_scan)));
        }
        if recovery > 0 {
            stages_str.push_str(&format!(" recovery={} ({:.1}%)", recovery, pct_cells(recovery)));
        }
        stages_str.push_str(&format!(
            " exhausted={} ({:.1}%)",
            self.cell_sub.cells_knn_exhausted,
            pct_cells(self.cell_sub.cells_knn_exhausted)
        ));
        eprintln!("{}", stages_str);

        // Neighbor histogram: compute percentiles
        let hist = &self.cell_sub.neighbors_histogram;
        let hist_total: u64 = hist.iter().sum();
        if hist_total > 0 {
            // Convert bucket index to neighbor count
            let bucket_to_neighbors = |bucket: usize| -> usize {
                if bucket < 48 {
                    bucket + 1 // buckets 0-47 = 1-48 neighbors
                } else if bucket == 48 {
                    64 // 49-64 range, report upper bound
                } else if bucket == 49 {
                    96 // 65-96 range
                } else {
                    97 // 97+ range
                }
            };

            // Find percentiles by scanning cumulative distribution
            let find_percentile = |p: f64| -> usize {
                let target = (hist_total as f64 * p) as u64;
                let mut cumulative = 0u64;
                for (bucket, &count) in hist.iter().enumerate() {
                    cumulative += count;
                    if cumulative >= target {
                        return bucket_to_neighbors(bucket);
                    }
                }
                bucket_to_neighbors(NEIGHBOR_HIST_BUCKETS - 1)
            };

            // Find max (last non-zero bucket)
            let max_bucket = hist
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &c)| c > 0)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let max_neighbors = bucket_to_neighbors(max_bucket);

            eprintln!(
                "    neighbors: p50={} p90={} p99={} max={}",
                find_percentile(0.50),
                find_percentile(0.90),
                find_percentile(0.99),
                max_neighbors,
            );

            // Dump non-zero buckets as: n=count (pct%)
            let mut detail = String::from("    neighbors_detail:");
            for (bucket, &count) in hist.iter().enumerate() {
                if count > 0 {
                    let n = bucket_to_neighbors(bucket);
                    let pct = count as f64 / hist_total as f64 * 100.0;
                    detail.push_str(&format!(" {}={:.1}%", n, pct));
                }
            }
            eprintln!("{}", detail);
        }

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            self.dedup.as_secs_f64() * 1000.0,
            pct(self.dedup)
        );

        // Dedup sub-phase breakdown
        let dedup_total = self.dedup_sub.setup + self.dedup_sub.lookup + self.dedup_sub.cell_dedup;
        if dedup_total.as_nanos() > 0 {
            let dedup_pct = |d: Duration| d.as_secs_f64() / dedup_total.as_secs_f64() * 100.0;
            eprintln!(
                "    setup:           {:7.1}ms ({:4.1}%)",
                self.dedup_sub.setup.as_secs_f64() * 1000.0,
                dedup_pct(self.dedup_sub.setup)
            );
            eprintln!(
                "    lookup:          {:7.1}ms ({:4.1}%)",
                self.dedup_sub.lookup.as_secs_f64() * 1000.0,
                dedup_pct(self.dedup_sub.lookup)
            );
            eprintln!(
                "    cell_dedup:      {:7.1}ms ({:4.1}%)",
                self.dedup_sub.cell_dedup.as_secs_f64() * 1000.0,
                dedup_pct(self.dedup_sub.cell_dedup)
            );

            // Optional: live-dedup breakdown (when populated).
            let live_total = self.dedup_sub.overflow_collect
                + self.dedup_sub.overflow_flush
                + self.dedup_sub.concat_vertices
                + self.dedup_sub.emit_cells;
            if live_total.as_nanos() > 0 {
                eprintln!(
                    "    live: overflow_collect={:7.1}ms overflow_flush={:7.1}ms concat_vertices={:7.1}ms emit_cells={:7.1}ms",
                    self.dedup_sub.overflow_collect.as_secs_f64() * 1000.0,
                    self.dedup_sub.overflow_flush.as_secs_f64() * 1000.0,
                    self.dedup_sub.concat_vertices.as_secs_f64() * 1000.0,
                    self.dedup_sub.emit_cells.as_secs_f64() * 1000.0,
                );
            }
            let total_keys = self.dedup_sub.triplet_keys + self.dedup_sub.support_keys;
            if total_keys > 0 {
                let key_pct = |k: u64| k as f64 / total_keys as f64 * 100.0;
                eprintln!(
                    "    keys: triplet={} ({:.1}%) support={} ({:.1}%)",
                    self.dedup_sub.triplet_keys,
                    key_pct(self.dedup_sub.triplet_keys),
                    self.dedup_sub.support_keys,
                    key_pct(self.dedup_sub.support_keys),
                );
            }
            if self.dedup_sub.cell_dupes_removed > 0 {
                if total_keys > 0 {
                    let dupe_rate =
                        self.dedup_sub.cell_dupes_removed as f64 / total_keys as f64 * 100.0;
                    eprintln!(
                        "    cell_dupes_removed: {} ({:.2}% of keys)",
                        self.dedup_sub.cell_dupes_removed,
                        dupe_rate,
                    );
                } else {
                    eprintln!(
                        "    cell_dupes_removed: {}",
                        self.dedup_sub.cell_dupes_removed
                    );
                }
            }
        }

        eprintln!(
            "  assemble:          {:7.1}ms ({:4.1}%)",
            self.assemble.as_secs_f64() * 1000.0,
            pct(self.assemble)
        );
        eprintln!("  total:             {:7.1}ms", total_ms);
    }
}

/// Dummy sub-phases when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CellSubPhases;

/// Dummy timings when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy)]
pub struct PhaseTimings;

#[cfg(not(feature = "timing"))]
impl PhaseTimings {
    #[inline(always)]
    pub fn report(&self, _n: usize) {}
}

/// Timer that tracks elapsed time when timing is enabled.
#[cfg(feature = "timing")]
pub struct Timer(std::time::Instant);

#[cfg(feature = "timing")]
impl Timer {
    #[inline]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Dummy timer when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
pub struct Timer;

#[cfg(not(feature = "timing"))]
impl Timer {
    #[inline(always)]
    pub fn start() -> Self {
        Self
    }

    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

/// Accumulator for cell sub-phase timings (used per-chunk, then merged).
#[cfg(feature = "timing")]
#[derive(Clone)]
pub struct CellSubAccum {
    pub knn_query: Duration,
    pub packed_knn: Duration,
    pub clipping: Duration,
    pub certification: Duration,
    pub key_dedup: Duration,
    pub stage_counts: FxHashMap<KnnCellStage, u64>,
    pub cells_knn_exhausted: u64,
    pub neighbors_histogram: [u64; NEIGHBOR_HIST_BUCKETS],
}

#[cfg(feature = "timing")]
impl Default for CellSubAccum {
    fn default() -> Self {
        Self {
            knn_query: Duration::ZERO,
            packed_knn: Duration::ZERO,
            clipping: Duration::ZERO,
            certification: Duration::ZERO,
            key_dedup: Duration::ZERO,
            stage_counts: FxHashMap::default(),
            cells_knn_exhausted: 0,
            neighbors_histogram: [0; NEIGHBOR_HIST_BUCKETS],
        }
    }
}

#[cfg(feature = "timing")]
impl CellSubAccum {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_knn(&mut self, d: Duration) {
        self.knn_query += d;
    }

    pub fn add_packed_knn(&mut self, d: Duration) {
        self.packed_knn += d;
    }

    pub fn add_clip(&mut self, d: Duration) {
        self.clipping += d;
    }

    pub fn add_cert(&mut self, d: Duration) {
        self.certification += d;
    }

    pub fn add_key_dedup(&mut self, d: Duration) {
        self.key_dedup += d;
    }

    pub fn add_cell_stage(
        &mut self,
        stage: KnnCellStage,
        knn_exhausted: bool,
        neighbors_processed: usize,
    ) {
        *self.stage_counts.entry(stage).or_insert(0) += 1;
        if knn_exhausted {
            self.cells_knn_exhausted += 1;
        }
        // Record histogram bucket
        let bucket = if neighbors_processed <= 48 {
            neighbors_processed.saturating_sub(1) // 1->0, 2->1, ..., 48->47
        } else if neighbors_processed <= 64 {
            48
        } else if neighbors_processed <= 96 {
            49
        } else {
            50
        };
        self.neighbors_histogram[bucket] += 1;
    }

    pub fn merge(&mut self, other: &CellSubAccum) {
        self.knn_query += other.knn_query;
        self.packed_knn += other.packed_knn;
        self.clipping += other.clipping;
        self.certification += other.certification;
        self.key_dedup += other.key_dedup;
        for (&stage, &count) in &other.stage_counts {
            *self.stage_counts.entry(stage).or_insert(0) += count;
        }
        self.cells_knn_exhausted += other.cells_knn_exhausted;
        for (i, &count) in other.neighbors_histogram.iter().enumerate() {
            self.neighbors_histogram[i] += count;
        }
    }

    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases {
            knn_query: self.knn_query,
            packed_knn: self.packed_knn,
            clipping: self.clipping,
            certification: self.certification,
            key_dedup: self.key_dedup,
            stage_counts: self.stage_counts,
            cells_knn_exhausted: self.cells_knn_exhausted,
            neighbors_histogram: self.neighbors_histogram,
        }
    }
}

/// Dummy accumulator when feature is disabled.
#[cfg(not(feature = "timing"))]
#[derive(Default, Clone, Copy)]
pub struct CellSubAccum;

#[cfg(not(feature = "timing"))]
impl CellSubAccum {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }
    #[inline(always)]
    pub fn add_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_packed_knn(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_clip(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cert(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_key_dedup(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cell_stage(
        &mut self,
        _stage: KnnCellStage,
        _knn_exhausted: bool,
        _neighbors_processed: usize,
    ) {
    }
    #[inline(always)]
    pub fn merge(&mut self, _other: &CellSubAccum) {}
    #[inline(always)]
    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases
    }
}

#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnCellStage {
    /// Terminated during resume stage with given K value
    Resume(usize),
    /// Terminated during restart stage with given K value
    Restart(usize),
    /// Ran full scan as fallback
    FullScanFallback,
    /// Ran full scan for dead-cell recovery
    FullScanRecovery,
}

/// Builder for collecting phase timings.
#[cfg(feature = "timing")]
pub struct TimingBuilder {
    t_start: std::time::Instant,
    knn_build: Duration,
    cell_construction: Duration,
    cell_sub: CellSubPhases,
    dedup: Duration,
    dedup_sub: DedupSubPhases,
    assemble: Duration,
}

#[cfg(feature = "timing")]
impl TimingBuilder {
    pub fn new() -> Self {
        Self {
            t_start: std::time::Instant::now(),
            knn_build: Duration::ZERO,
            cell_construction: Duration::ZERO,
            cell_sub: CellSubPhases::default(),
            dedup: Duration::ZERO,
            dedup_sub: DedupSubPhases::default(),
            assemble: Duration::ZERO,
        }
    }

    pub fn set_knn_build(&mut self, d: Duration) {
        self.knn_build = d;
    }

    pub fn set_cell_construction(&mut self, d: Duration, sub: CellSubPhases) {
        self.cell_construction = d;
        self.cell_sub = sub;
    }

    pub fn set_dedup(&mut self, d: Duration, sub: DedupSubPhases) {
        self.dedup = d;
        self.dedup_sub = sub;
    }

    pub fn set_assemble(&mut self, d: Duration) {
        self.assemble = d;
    }

    pub fn finish(self) -> PhaseTimings {
        PhaseTimings {
            total: self.t_start.elapsed(),
            knn_build: self.knn_build,
            cell_construction: self.cell_construction,
            cell_sub: self.cell_sub,
            dedup: self.dedup,
            dedup_sub: self.dedup_sub,
            assemble: self.assemble,
        }
    }
}

/// Dummy builder when feature is disabled.
#[cfg(not(feature = "timing"))]
pub struct TimingBuilder;

#[cfg(not(feature = "timing"))]
impl TimingBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn set_knn_build(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_cell_construction(&mut self, _d: Duration, _sub: CellSubPhases) {}

    #[inline(always)]
    pub fn set_dedup(&mut self, _d: Duration, _sub: DedupSubPhases) {}

    #[inline(always)]
    pub fn set_assemble(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn finish(self) -> PhaseTimings {
        PhaseTimings
    }
}
