//! Zero-cost timing instrumentation for gpu_voronoi.
//!
//! When the `timing` feature is enabled, this module provides timing
//! infrastructure that measures and reports phase durations.
//! When disabled, all timing code compiles away to nothing.
//!
//! Usage:
//!   cargo run --release --features timing

use std::time::Duration;

/// Sub-phase timings within cell construction.
#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default)]
pub struct CellSubPhases {
    pub knn_query: Duration,
    pub clipping: Duration,
    pub certification: Duration,
    /// Per-cell k-NN stage distribution (final stage used per cell).
    pub cells_k12: u64,
    pub cells_k24: u64,
    pub cells_k48: u64,
    /// Cells that ran an O(n) full scan due to k-NN exhaustion.
    pub cells_full_scan_fallback: u64,
    /// Cells that ran an O(n) full scan due to dead-cell recovery.
    pub cells_full_scan_recovery: u64,
    /// Cells where the k-NN search loop exhausted (typically means it hit brute force).
    pub cells_knn_exhausted: u64,
}

/// Phase timings for the Voronoi algorithm.
#[cfg(feature = "timing")]
#[derive(Debug, Clone)]
pub struct PhaseTimings {
    pub total: Duration,
    pub knn_build: Duration,
    pub cell_construction: Duration,
    pub cell_sub: CellSubPhases,
    pub dedup: Duration,
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

        eprintln!("[timing] gpu_voronoi n={}", n);
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
        let cpu_total =
            self.cell_sub.knn_query + self.cell_sub.clipping + self.cell_sub.certification;
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
        eprintln!("    ({:.1}x parallelism)", parallelism);

        let total_cells = (self.cell_sub.cells_k12
            + self.cell_sub.cells_k24
            + self.cell_sub.cells_k48
            + self.cell_sub.cells_full_scan_fallback
            + self.cell_sub.cells_full_scan_recovery)
            .max(1u64);
        let pct_cells = |c: u64| c as f64 / total_cells as f64 * 100.0;
        eprintln!(
            "    knn_stages: k12={} ({:.1}%) k24={} ({:.1}%) k48={} ({:.1}%) full_scan={} ({:.1}%) recovery_scan={} ({:.1}%) exhausted={} ({:.1}%)",
            self.cell_sub.cells_k12,
            pct_cells(self.cell_sub.cells_k12),
            self.cell_sub.cells_k24,
            pct_cells(self.cell_sub.cells_k24),
            self.cell_sub.cells_k48,
            pct_cells(self.cell_sub.cells_k48),
            self.cell_sub.cells_full_scan_fallback,
            pct_cells(self.cell_sub.cells_full_scan_fallback),
            self.cell_sub.cells_full_scan_recovery,
            pct_cells(self.cell_sub.cells_full_scan_recovery),
            self.cell_sub.cells_knn_exhausted,
            pct_cells(self.cell_sub.cells_knn_exhausted),
        );

        eprintln!(
            "  dedup:             {:7.1}ms ({:4.1}%)",
            self.dedup.as_secs_f64() * 1000.0,
            pct(self.dedup)
        );
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

/// Accumulator for timing multiple iterations of a phase.
#[cfg(feature = "timing")]
#[derive(Default)]
pub struct PhaseAccum {
    pub total: Duration,
}

#[cfg(feature = "timing")]
impl PhaseAccum {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add(&mut self, d: Duration) {
        self.total += d;
    }
}

/// Dummy accumulator when feature is disabled (zero-sized).
#[cfg(not(feature = "timing"))]
pub struct PhaseAccum;

#[cfg(not(feature = "timing"))]
impl PhaseAccum {
    #[inline(always)]
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn add(&mut self, _d: Duration) {}
}

/// Accumulator for cell sub-phase timings (used per-chunk, then merged).
#[cfg(feature = "timing")]
#[derive(Default, Clone)]
pub struct CellSubAccum {
    pub knn_query: Duration,
    pub clipping: Duration,
    pub certification: Duration,
    pub cells_k12: u64,
    pub cells_k24: u64,
    pub cells_k48: u64,
    pub cells_full_scan_fallback: u64,
    pub cells_full_scan_recovery: u64,
    pub cells_knn_exhausted: u64,
}

#[cfg(feature = "timing")]
impl CellSubAccum {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_knn(&mut self, d: Duration) {
        self.knn_query += d;
    }

    pub fn add_clip(&mut self, d: Duration) {
        self.clipping += d;
    }

    pub fn add_cert(&mut self, d: Duration) {
        self.certification += d;
    }

    pub fn add_cell_stage(&mut self, stage: KnnCellStage, knn_exhausted: bool) {
        match stage {
            KnnCellStage::K12 => self.cells_k12 += 1,
            KnnCellStage::K24 => self.cells_k24 += 1,
            KnnCellStage::K48 => self.cells_k48 += 1,
            KnnCellStage::FullScanFallback => self.cells_full_scan_fallback += 1,
            KnnCellStage::FullScanRecovery => self.cells_full_scan_recovery += 1,
        }
        if knn_exhausted {
            self.cells_knn_exhausted += 1;
        }
    }

    pub fn merge(&mut self, other: &CellSubAccum) {
        self.knn_query += other.knn_query;
        self.clipping += other.clipping;
        self.certification += other.certification;
        self.cells_k12 += other.cells_k12;
        self.cells_k24 += other.cells_k24;
        self.cells_k48 += other.cells_k48;
        self.cells_full_scan_fallback += other.cells_full_scan_fallback;
        self.cells_full_scan_recovery += other.cells_full_scan_recovery;
        self.cells_knn_exhausted += other.cells_knn_exhausted;
    }

    pub fn into_sub_phases(self) -> CellSubPhases {
        CellSubPhases {
            knn_query: self.knn_query,
            clipping: self.clipping,
            certification: self.certification,
            cells_k12: self.cells_k12,
            cells_k24: self.cells_k24,
            cells_k48: self.cells_k48,
            cells_full_scan_fallback: self.cells_full_scan_fallback,
            cells_full_scan_recovery: self.cells_full_scan_recovery,
            cells_knn_exhausted: self.cells_knn_exhausted,
        }
    }
}

#[cfg(feature = "timing")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnCellStage {
    K12,
    K24,
    K48,
    FullScanFallback,
    FullScanRecovery,
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
    pub fn add_clip(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cert(&mut self, _d: Duration) {}
    #[inline(always)]
    pub fn add_cell_stage(&mut self, _stage: KnnCellStage, _knn_exhausted: bool) {}
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
    K12,
    K24,
    K48,
    FullScanFallback,
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

    pub fn set_dedup(&mut self, d: Duration) {
        self.dedup = d;
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
    pub fn set_dedup(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn set_assemble(&mut self, _d: Duration) {}

    #[inline(always)]
    pub fn finish(self) -> PhaseTimings {
        PhaseTimings
    }
}
