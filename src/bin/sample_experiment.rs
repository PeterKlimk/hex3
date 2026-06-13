//! Standalone experiment for fine-mesh point sampling — runs *in a vacuum*,
//! with no world generation and no rendering. The goal is to compare candidate
//! sampling algorithms on perf and cell quality, so we can decide how to fix
//! the jagged/sliver cells around density transitions (see fine.rs).
//!
//! The density target is a synthetic continuous field on the sphere: a low
//! background with a few broad Gaussian "mountain" blobs and several thin
//! great-circle "river" arcs (the thin features are where slivers hurt most).
//! It omits the ocean floor but captures the land-detail contrast that drives
//! the artifacts. Max:min ratio is clamped to ~50:1 like the real prior.
//!
//! Algorithms compared:
//!   thin-piecewise  current production: Bernoulli thinning, density read from
//!                   the NEAREST coarse cell (piecewise-constant -> seams)
//!   thin-continuous Bernoulli thinning, density evaluated continuously
//!                   (isolates the density-interpolation fix)
//!   elimination     weighted sample elimination (Yuksel) -> adaptive blue noise
//!   relax-N         thin-continuous + N particle-repulsion relaxation passes
//!
//! Metrics (per algorithm):
//!   sample/tess time, cell count vs target,
//!   rho = nn_dist / sqrt(cell_area): min / p1 / p5 / median / CoV
//!         (blue noise -> tight & high; Poisson/slivers -> long tail toward 0),
//!   sliver fraction (rho < 0.4 and < 0.25),
//!   mean area CoV within log-spaced density bands (regularity, density removed),
//!   achieved vs intended coarse:fine area ratio (does density get respected).
//!
//! Usage:
//!   cargo run --release --bin sample_experiment -- --target 250000 --seed 12345
//!   cargo run --release --bin sample_experiment -- --target 2500000 --only elimination

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

use clap::Parser;
use glam::Vec3;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use hex3::geometry::SphericalVoronoi;
use hex3::world::Tessellation;

type Tree = ImmutableKdTree<f32, 3>;

const PHI: f32 = 1.618_034;
const FOUR_PI: f32 = 4.0 * std::f32::consts::PI;
const FIBONACCI_JITTER: f32 = 0.25;
const DENSITY_MAX_RATIO: f32 = 50.0;
const THIN_OVERSHOOT: f32 = 1.15;
const ELIM_OVERSAMPLE: f32 = 1.8;

#[derive(Parser)]
struct Cli {
    /// Target fine-cell count.
    #[arg(long, default_value_t = 250_000)]
    target: usize,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    /// Run only one algorithm (thin-piecewise | thin-continuous | elimination | relax).
    #[arg(long)]
    only: Option<String>,
    /// Relaxation passes for the relax algorithm.
    #[arg(long, default_value_t = 3)]
    relax_passes: usize,
    /// Coarse-cell count for the piecewise-constant density path (production: 100k).
    #[arg(long, default_value_t = 100_000)]
    coarse: usize,
    /// Cross-check degenerate cells against the exact convex-hull backend
    /// (qhull) on identical points. Use a modest --target; qhull is heavy.
    #[arg(long, default_value_t = false)]
    cross_check: bool,
}

fn main() {
    let cli = Cli::parse();
    let seed = cli.seed;
    let target = cli.target;

    let density = DensityField::new(seed);
    let mean_density = density.area_weighted_mean();
    println!(
        "density field: base={:.2} max={:.2} mean={:.3} intended_ratio={:.0}:1  ({} blobs, {} arcs)",
        density.base,
        density.max,
        mean_density,
        density.max / density.base,
        density.blobs.len(),
        density.arcs.len(),
    );
    println!(
        "target={} cells  seed={}  (rho is nn_dist / sqrt(cell_area); higher+tighter = more regular)\n",
        target, seed
    );

    if cli.cross_check {
        let pts = sample_thinning(seed, &density, mean_density, target, None);
        cross_check_backends(pts, &density, mean_density, target);
        return;
    }

    print_header();

    let want = |name: &str| {
        cli.only
            .as_deref()
            .map(|o| o.split(',').any(|t| t.trim() == name))
            .unwrap_or(true)
    };

    if want("thin-piecewise") {
        let coarse = build_coarse(&density, cli.coarse);
        println!("  (coarse density grid: {} cells)", cli.coarse);
        let t = Instant::now();
        let pts = sample_thinning(seed, &density, mean_density, target, Some(&coarse));
        let t_sample = t.elapsed();
        evaluate("thin-piecewise", pts, &density, mean_density, target, t_sample);
    }
    if want("thin-continuous") {
        let t = Instant::now();
        let pts = sample_thinning(seed, &density, mean_density, target, None);
        let t_sample = t.elapsed();
        evaluate("thin-continuous", pts, &density, mean_density, target, t_sample);
    }
    if want("elimination") {
        let t = Instant::now();
        let pts = sample_elimination(seed, &density, mean_density, target);
        let t_sample = t.elapsed();
        evaluate("elimination", pts, &density, mean_density, target, t_sample);
    }
    if want("relax") {
        let t = Instant::now();
        let pts = sample_relaxed(seed, &density, mean_density, target, cli.relax_passes);
        let t_sample = t.elapsed();
        evaluate(
            &format!("relax-{}", cli.relax_passes),
            pts,
            &density,
            mean_density,
            target,
            t_sample,
        );
    }
}

// ----------------------------------------------------------------------------
// Synthetic density field
// ----------------------------------------------------------------------------

struct DensityField {
    base: f32,
    max: f32,
    blobs: Vec<(Vec3, f32, f32)>, // center, amplitude, sigma (radians)
    arcs: Vec<(Vec3, f32, f32)>,  // great-circle normal, amplitude, sigma (radians)
}

impl DensityField {
    fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xDED1_7777);
        let blobs = (0..6)
            .map(|_| {
                let c = random_unit(&mut rng);
                let amp = rng.gen_range(20.0..45.0);
                let sigma = rng.gen_range(0.18..0.32);
                (c, amp, sigma)
            })
            .collect();
        let arcs = (0..8)
            .map(|_| {
                let n = random_unit(&mut rng);
                let amp = rng.gen_range(30.0..49.0);
                let sigma = rng.gen_range(0.012..0.028);
                (n, amp, sigma)
            })
            .collect();
        Self {
            base: 1.0,
            max: 50.0,
            blobs,
            arcs,
        }
    }

    fn value(&self, p: Vec3) -> f32 {
        let mut d = self.base;
        for &(c, amp, sigma) in &self.blobs {
            let ang = c.dot(p).clamp(-1.0, 1.0).acos();
            let t = ang / sigma;
            d += amp * (-t * t).exp();
        }
        for &(n, amp, sigma) in &self.arcs {
            // angular distance from p to the great circle with normal n
            let ang = (n.dot(p).clamp(-1.0, 1.0).abs()).asin();
            let t = ang / sigma;
            d += amp * (-t * t).exp();
        }
        d.clamp(self.base, self.max)
    }

    fn area_weighted_mean(&self) -> f32 {
        // Fibonacci points are ~equal-area, so a plain average estimates the
        // area-weighted mean.
        let n = 200_000usize;
        (0..n)
            .into_par_iter()
            .map(|i| self.value(fibonacci_point(i, n)))
            .sum::<f32>()
            / n as f32
    }
}

// ----------------------------------------------------------------------------
// Samplers
// ----------------------------------------------------------------------------

/// Bernoulli thinning of jittered-Fibonacci candidates. If `coarse` is given,
/// the acceptance density is read from the nearest coarse cell (reproducing the
/// piecewise-constant production path); otherwise it is evaluated continuously.
fn sample_thinning(
    seed: u64,
    density: &DensityField,
    mean_density: f32,
    target: usize,
    coarse: Option<&CoarseDensity>,
) -> Vec<Vec3> {
    let max_density = density.max;
    let candidate_count =
        ((target as f32 * max_density / mean_density) * THIN_OVERSHOOT).ceil() as usize;
    let mean_spacing = (FOUR_PI / candidate_count as f32).sqrt();
    let jitter = mean_spacing * FIBONACCI_JITTER;
    let sampling_seed = seed.wrapping_add(0x5A11);

    let mut accepted: Vec<Vec3> = (0..candidate_count)
        .into_par_iter()
        .filter_map(|i| {
            let p = jittered_fibonacci_point(i, candidate_count, jitter, sampling_seed);
            let d = match coarse {
                Some(c) => c.value_at(p),
                None => density.value(p),
            };
            let accept = (d / max_density).clamp(0.0, 1.0);
            (hash_unit(sampling_seed, i as u64, 1) <= accept).then_some(p)
        })
        .collect();

    // Top up if we undershot (rare with the overshoot factor).
    let mut rng = ChaCha8Rng::seed_from_u64(sampling_seed ^ 0xF111);
    while accepted.len() < target {
        let p = random_unit(&mut rng);
        let d = match coarse {
            Some(c) => c.value_at(p),
            None => density.value(p),
        };
        if rng.gen::<f32>() <= (d / max_density).clamp(0.0, 1.0) {
            accepted.push(p);
        }
    }

    // Shuffle then truncate so the discarded surplus isn't spatially biased.
    fisher_yates(&mut accepted, &mut rng);
    accepted.truncate(target);
    accepted
}

/// Weighted sample elimination (Yuksel 2015): over-generate density-proportional
/// candidates, then greedily remove the candidate with the highest "crowding"
/// weight until `target` remain. Produces adaptive blue noise with a soft
/// minimum-distance guarantee. Symmetric interaction radius keeps removal
/// updates cheap (bounded by the removed point's own, density-scaled radius).
fn sample_elimination(
    seed: u64,
    density: &DensityField,
    mean_density: f32,
    target: usize,
) -> Vec<Vec3> {
    let m = (target as f32 * ELIM_OVERSAMPLE).ceil() as usize;
    let candidates = sample_thinning(seed, density, mean_density, m, None);
    let n = candidates.len();

    let entries: Vec<[f32; 3]> = candidates.iter().map(|p| p.to_array()).collect();
    let tree = Tree::new_from_slice(&entries);

    // Per-candidate desired separation from the FINAL target density, and the
    // interaction cutoff d_max = 2*sep. Pair interaction uses min(d_max_i,d_max_j).
    let d_max: Vec<f32> = candidates
        .par_iter()
        .map(|&p| 2.0 * target_spacing(density.value(p), mean_density, target))
        .collect();
    let d_max_sq: Vec<f32> = d_max.iter().map(|d| d * d).collect();

    let contribution = |d: f32, dmaxij: f32| -> f32 {
        if d >= dmaxij {
            0.0
        } else {
            let x = 1.0 - d / dmaxij;
            // (1 - d/dmax)^8
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4
        }
    };

    // Initial crowding weights.
    let mut weight: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let q = entries[i];
            let mut w = 0.0f32;
            for nb in tree.within_unsorted::<SquaredEuclidean>(&q, d_max_sq[i]) {
                let j = nb.item as usize;
                if j == i {
                    continue;
                }
                let dmaxij = d_max[i].min(d_max[j]);
                w += contribution(nb.distance.sqrt(), dmaxij);
            }
            w
        })
        .collect();

    let mut alive = vec![true; n];
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(n);
    for i in 0..n {
        heap.push(HeapItem {
            weight: weight[i],
            idx: i,
        });
    }

    let mut alive_count = n;
    while alive_count > target {
        let Some(top) = heap.pop() else { break };
        let i = top.idx;
        if !alive[i] || top.weight != weight[i] {
            continue; // stale heap entry
        }
        alive[i] = false;
        alive_count -= 1;

        let q = entries[i];
        let neighbours = tree.within_unsorted::<SquaredEuclidean>(&q, d_max_sq[i]);
        for nb in neighbours {
            let j = nb.item as usize;
            if j == i || !alive[j] {
                continue;
            }
            let dmaxij = d_max[i].min(d_max[j]);
            let c = contribution(nb.distance.sqrt(), dmaxij);
            if c > 0.0 {
                weight[j] -= c;
                heap.push(HeapItem {
                    weight: weight[j],
                    idx: j,
                });
            }
        }
    }

    candidates
        .into_iter()
        .zip(alive)
        .filter_map(|(p, a)| a.then_some(p))
        .collect()
}

/// Thinning followed by N passes of density-aware particle repulsion. Each pass
/// pushes a point off neighbours closer than its target spacing, tangent to the
/// sphere (Jacobi update from a freshly built tree). No Voronoi recompute.
fn sample_relaxed(
    seed: u64,
    density: &DensityField,
    mean_density: f32,
    target: usize,
    passes: usize,
) -> Vec<Vec3> {
    let mut points = sample_thinning(seed, density, mean_density, target, None);

    for _ in 0..passes {
        let entries: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
        let tb = Instant::now();
        // Mutable KdTree: incremental build is ~20x faster than ImmutableKdTree's
        // optimal-layout build at millions of points, and we rebuild every pass.
        let mut tree = kiddo::KdTree::<f32, 3>::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            tree.add(e, i as u64);
        }
        let t_build = tb.elapsed();
        let tq = Instant::now();

        // Bounded k-nearest query (not radius-within): keeps per-point work
        // constant even where the input clumps, which is what blows up an
        // unbounded radius query on Poisson-clumped points.
        const RELAX_K: usize = 8;
        points = points
            .par_iter()
            .enumerate()
            .map(|(i, &p)| {
                let sep = target_spacing(density.value(p), mean_density, target);
                let mut push = Vec3::ZERO;
                for nb in tree.nearest_n::<SquaredEuclidean>(&entries[i], RELAX_K + 1) {
                    let j = nb.item as usize;
                    if j == i {
                        continue;
                    }
                    let d = nb.distance.sqrt();
                    if d > 1e-7 && d < sep {
                        let dir = (p - points[j]) / d;
                        push += dir * (sep - d) / sep;
                    }
                }
                if push == Vec3::ZERO {
                    return p;
                }
                // project the push onto the tangent plane and step
                let tangent = push - p * push.dot(p);
                (p + tangent * (0.4 * sep)).normalize()
            })
            .collect();
        eprintln!(
            "    relax pass: tree build {:.2?}, queries+move {:.2?}",
            t_build,
            tq.elapsed()
        );
    }

    points
}

/// Target nearest-neighbour spacing at a given density, for `target` total
/// points distributed with areal density proportional to the field.
fn target_spacing(density_value: f32, mean_density: f32, target: usize) -> f32 {
    // areal point density g = target * d / (mean * 4pi); expected area = 1/g.
    let g = target as f32 * density_value / (mean_density * FOUR_PI);
    (1.0 / g.max(1e-12)).sqrt()
}

// ----------------------------------------------------------------------------
// Coarse piecewise-constant density (production path)
// ----------------------------------------------------------------------------

struct CoarseDensity {
    tree: Tree,
    density: Vec<f32>,
}

impl CoarseDensity {
    fn value_at(&self, p: Vec3) -> f32 {
        let nn = self.tree.nearest_one::<SquaredEuclidean>(&p.to_array());
        self.density[nn.item as usize]
    }
}

fn build_coarse(density: &DensityField, coarse_cells: usize) -> CoarseDensity {
    let centers: Vec<Vec3> = (0..coarse_cells)
        .map(|i| fibonacci_point(i, coarse_cells))
        .collect();
    let entries: Vec<[f32; 3]> = centers.iter().map(|p| p.to_array()).collect();
    let tree = Tree::new_from_slice(&entries);
    // density prior sampled per coarse cell, then clamped to the max ratio
    let mut d: Vec<f32> = centers.par_iter().map(|&c| density.value(c)).collect();
    let min_d = d.iter().copied().fold(f32::INFINITY, f32::min).max(1e-6);
    let max_allowed = min_d * DENSITY_MAX_RATIO;
    for x in &mut d {
        *x = x.clamp(min_d, max_allowed);
    }
    CoarseDensity { tree, density: d }
}

// ----------------------------------------------------------------------------
// Backend cross-check: is a degenerate cell s2-voronoi's fault or the input's?
// ----------------------------------------------------------------------------

/// Tessellate identical points with BOTH backends (s2-voronoi knn-clipping and
/// the exact convex-hull/qhull duality) and cross-tabulate topological
/// degeneracy. A bounded Voronoi cell is always a >=3-gon; vertex count < 3 means
/// the backend failed to produce the polygon. The decisive number is
/// "knn-only": cells the exact backend resolves but s2-voronoi does not.
fn cross_check_backends(points: Vec<Vec3>, density: &DensityField, mean_density: f32, target: usize) {
    let n = points.len();
    println!("cross-check: {} points, both backends\n", n);

    let t = Instant::now();
    let knn = Tessellation::from_points_knn_clipping(points.clone());
    println!("  knn-clipping (s2-voronoi): {:.2?}", t.elapsed());

    // Is "area ~ 0" a real tiny cell or an f32 area-computation underflow?
    // Compare hex3's f32 cell_areas() against an f64 recompute on the SAME
    // (s2-voronoi) cells.
    let f32_areas = knn.cell_areas();
    let f32_tiny = f32_areas.iter().filter(|&&a| a <= 1e-9).count();
    let f64_tiny = (0..knn.num_cells())
        .into_par_iter()
        .filter(|&i| spherical_polygon_area_f64(&knn.voronoi, i) <= 1e-9)
        .count();
    let knn_lt3 = (0..knn.num_cells())
        .filter(|&i| knn.voronoi.cell(i).len() < 3)
        .count();
    println!(
        "  s2-voronoi cells: vertex-count<3 = {}, area<=1e-9 (f32 cell_areas) = {}, area<=1e-9 (f64 recompute) = {}",
        knn_lt3, f32_tiny, f64_tiny
    );

    let t = Instant::now();
    let ch = SphericalVoronoi::compute(&points);
    println!("  convex-hull (qhull, exact): {:.2?}", t.elapsed());

    // Align cell indices to the original points via nearest generator, in case a
    // backend reorders. (Both appear to preserve order, but don't assume it.)
    let knn_idx = align_to_points(&points, &knn.voronoi);
    let ch_idx = align_to_points(&points, &ch);

    // Degenerate = clipped to < 3 vertices (cannot be a real bounded cell).
    let knn_degen = degenerate_mask(&knn.voronoi, &knn_idx);
    let ch_degen = degenerate_mask(&ch, &ch_idx);

    // One tree on the points for the knn-only spacing diagnostic.
    let pt_entries: Vec<[f32; 3]> = points.iter().map(|p| p.to_array()).collect();
    let pt_tree = Tree::new_from_slice(&pt_entries);

    let mut both = 0usize;
    let mut knn_only = 0usize;
    let mut ch_only = 0usize;
    // For the knn-only cells (s2-voronoi's "fault"), how close-spaced are they?
    let mut knn_only_rho: Vec<f32> = Vec::new();
    for i in 0..n {
        match (knn_degen[i], ch_degen[i]) {
            (true, true) => both += 1,
            (true, false) => {
                knn_only += 1;
                let nn = pt_tree.nearest_n::<SquaredEuclidean>(&pt_entries[i], 2);
                let d = nn.get(1).map(|x| x.distance.sqrt()).unwrap_or(0.0);
                knn_only_rho.push(d / target_spacing(density.value(points[i]), mean_density, target));
            }
            (false, true) => ch_only += 1,
            (false, false) => {}
        }
    }

    let knn_total = knn_degen.iter().filter(|&&d| d).count();
    let ch_total = ch_degen.iter().filter(|&&d| d).count();
    knn_only_rho.sort_by(f32::total_cmp);
    let med_rho = knn_only_rho.get(knn_only_rho.len() / 2).copied().unwrap_or(f32::NAN);

    println!();
    println!("  degenerate (vertex count < 3):");
    println!("    s2-voronoi (knn): {} ({:.3}%)", knn_total, knn_total as f32 / n as f32 * 100.0);
    println!("    convex-hull     : {} ({:.3}%)", ch_total, ch_total as f32 / n as f32 * 100.0);
    println!("  cross-tab:");
    println!("    both degenerate           : {}", both);
    println!("    knn-only (s2-voronoi fails, exact OK): {}  <-- s2-voronoi 'wrong'", knn_only);
    println!("    convex-hull-only          : {}", ch_only);
    println!(
        "  of the knn-only cells: median rho = {:.2} (rho<<1 => still genuine slivers, just ones qhull resolved)",
        med_rho
    );

    // ---- "nearly correct" geometry check: do the two backends agree on f64
    // cell area where BOTH resolve the cell? s2-voronoi promises a valid
    // subdivision that is *nearly* the true Voronoi, with no robustness
    // guarantee on epsilon edges -- so we expect tight agreement on normal
    // cells and a tail of disagreement concentrated on tiny/close-pair cells.
    let knn_area: Vec<f64> = knn_idx
        .par_iter()
        .map(|&c| spherical_polygon_area_f64(&knn.voronoi, c))
        .collect();
    let ch_area: Vec<f64> = ch_idx
        .par_iter()
        .map(|&c| spherical_polygon_area_f64(&ch, c))
        .collect();

    // Points that qhull welded into a shared cell (ch_idx not injective).
    let mut seen = vec![false; ch.num_cells()];
    let mut ch_welded = 0usize;
    for &c in &ch_idx {
        if seen[c] {
            ch_welded += 1;
        }
        seen[c] = true;
    }

    // Relative area error per point, paired with the cell's spacing rho, only
    // where both backends resolve the cell. Stratifying by rho separates an
    // f32 vertex-precision effect (relative area error grows as cells shrink)
    // from a genuine algorithmic disagreement (uniform across cell sizes).
    let mut rel: Vec<(f64, f32)> = Vec::with_capacity(n);
    let mut signed_sum = 0.0f64;
    for i in 0..n {
        if knn_degen[i] || ch_degen[i] {
            continue;
        }
        let m = knn_area[i].max(ch_area[i]);
        if m <= 0.0 {
            continue;
        }
        let nn = pt_tree.nearest_n::<SquaredEuclidean>(&pt_entries[i], 2);
        let d = nn.get(1).map(|x| x.distance.sqrt()).unwrap_or(0.0);
        let rho = d / target_spacing(density.value(points[i]), mean_density, target);
        rel.push(((knn_area[i] - ch_area[i]).abs() / m, rho));
        signed_sum += (knn_area[i] - ch_area[i]) / m; // +ve => knn cells larger
    }
    let mut errs: Vec<f64> = rel.iter().map(|x| x.0).collect();
    errs.sort_by(f64::total_cmp);
    let pct = |p: f64| {
        errs.get(((errs.len().max(1) as f64 - 1.0) * p) as usize)
            .copied()
            .unwrap_or(f64::NAN)
    };
    let band_med = |lo: f32, hi: f32| {
        let mut v: Vec<f64> = rel
            .iter()
            .filter(|(_, r)| *r >= lo && *r < hi)
            .map(|x| x.0)
            .collect();
        v.sort_by(f64::total_cmp);
        let med = v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
        (v.len(), med)
    };

    println!();
    println!(
        "  geometry agreement (f64 cell area, {} cells both resolve):",
        rel.len()
    );
    println!(
        "    rel-err overall: median={:.1e} p90={:.1e} p99={:.1e} max={:.1e}; mean signed (knn-ch)={:+.1e}",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        errs.last().copied().unwrap_or(f64::NAN),
        signed_sum / rel.len().max(1) as f64,
    );
    for (lo, hi, label) in [
        (0.0f32, 0.3f32, "rho<0.3  (tiny)   "),
        (0.3, 0.6, "rho 0.3-0.6 (small)"),
        (0.6, 0.9, "rho 0.6-0.9 (normal)"),
        (0.9, f32::INFINITY, "rho>=0.9 (well-spaced)"),
    ] {
        let (count, med) = band_med(lo, hi);
        println!("    {label}: n={count:>7} median rel-err={med:.1e}");
    }
    println!("    (qhull welded {} points into shared cells)", ch_welded);
}

/// Spherical polygon area in f64 (Van Oosterom–Strackee per triangle), summed
/// from the generator. Robust where the f32 spherical_triangle_area underflows.
fn spherical_polygon_area_f64(vor: &SphericalVoronoi, cell_idx: usize) -> f64 {
    let cell = vor.cell(cell_idx);
    let n = cell.len();
    if n < 3 {
        return 0.0;
    }
    let g = vor.generators[cell_idx];
    let center = [g.x as f64, g.y as f64, g.z as f64];
    let mut area = 0.0f64;
    for i in 0..n {
        let v1 = vor.vertices[cell.vertex_indices[i] as usize];
        let v2 = vor.vertices[cell.vertex_indices[(i + 1) % n] as usize];
        let a = center;
        let b = [v1.x as f64, v1.y as f64, v1.z as f64];
        let c = [v2.x as f64, v2.y as f64, v2.z as f64];
        let cross = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        let triple = (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]).abs();
        let ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let bc = b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
        let ca = c[0] * a[0] + c[1] * a[1] + c[2] * a[2];
        area += 2.0 * triple.atan2(1.0 + ab + bc + ca);
    }
    area
}

fn align_to_points(points: &[Vec3], vor: &SphericalVoronoi) -> Vec<usize> {
    // Map original-point index -> cell index in this diagram, by nearest generator.
    let entries: Vec<[f32; 3]> = vor.generators.iter().map(|g| g.to_array()).collect();
    let tree = Tree::new_from_slice(&entries);
    points
        .par_iter()
        .map(|p| tree.nearest_one::<SquaredEuclidean>(&p.to_array()).item as usize)
        .collect()
}

fn degenerate_mask(vor: &SphericalVoronoi, idx: &[usize]) -> Vec<bool> {
    idx.iter().map(|&c| vor.cell(c).len() < 3).collect()
}

// ----------------------------------------------------------------------------
// Evaluation
// ----------------------------------------------------------------------------

fn evaluate(
    label: &str,
    points: Vec<Vec3>,
    density: &DensityField,
    mean_density: f32,
    target: usize,
    t_sample: Duration,
) {
    let n = points.len();

    let t = Instant::now();
    let tess = Tessellation::from_points_knn_clipping(points.clone());
    let t_tess = t.elapsed();

    let areas = tess.cell_areas();
    let cells = tess.num_cells();

    // nn distance via kd-tree (2nd nearest = nearest other point)
    let entries: Vec<[f32; 3]> = (0..cells).map(|i| tess.cell_center(i).to_array()).collect();
    let tree = Tree::new_from_slice(&entries);

    let cell_density: Vec<f32> = (0..cells)
        .into_par_iter()
        .map(|i| density.value(tess.cell_center(i)))
        .collect();

    // rho = nn_dist / target_spacing(density): tessellation-independent, so a
    // few degenerate (near-zero-area) clipping cells can't blow up the metric.
    // Ideal hexagonal blue noise ~1.07 with low spread; Poisson ~0.5 with a
    // long tail toward 0 (close pairs = slivers). Degenerate cells (area ~ 0)
    // are counted separately, not folded into the spacing stats.
    let degen = areas.iter().filter(|&&a| a <= 1e-9).count();
    // Raw nearest-neighbour chord distance per generator (2nd nearest = nearest
    // OTHER generator). Kept raw so we can probe the weld threshold directly.
    let nn_dist: Vec<f32> = (0..cells)
        .into_par_iter()
        .map(|i| {
            let nn = tree.nearest_n::<SquaredEuclidean>(&entries[i], 2);
            nn.get(1).map(|x| x.distance.sqrt()).unwrap_or(0.0)
        })
        .collect();
    let rho: Vec<f32> = (0..cells)
        .map(|i| nn_dist[i] / target_spacing(cell_density[i], mean_density, target))
        .collect();

    let mut rho_sorted = rho.clone();
    rho_sorted.sort_by(f32::total_cmp);
    let rho_min = rho_sorted.first().copied().unwrap_or(0.0);
    let rho_p1 = percentile(&rho_sorted, 0.01);
    let rho_p5 = percentile(&rho_sorted, 0.05);
    let rho_med = percentile(&rho_sorted, 0.50);
    let rho_cov = cov(&rho);

    // sliver = points much closer than their target spacing (ideal rho ~1.07)
    let sliver_50 = rho.iter().filter(|&&r| r < 0.50).count() as f32 / cells as f32;
    let sliver_30 = rho.iter().filter(|&&r| r < 0.30).count() as f32 / cells as f32;

    let (band_cov, area_ratio_achieved, area_ratio_intended) =
        band_stats(&areas, &cell_density);

    let count_err = (n as f32 - target as f32) / target as f32 * 100.0;

    println!(
        "{:<14} {:>7.0}ms {:>7.0}ms  {:>8} {:>+5.1}% {:>6}  {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2}  {:>5.1}% {:>5.1}%  {:>6.2}  {:>5.0}/{:<5.0}",
        label,
        t_sample.as_secs_f64() * 1000.0,
        t_tess.as_secs_f64() * 1000.0,
        cells,
        count_err,
        degen,
        rho_min,
        rho_p1,
        rho_p5,
        rho_med,
        rho_cov,
        sliver_50 * 100.0,
        sliver_30 * 100.0,
        band_cov,
        area_ratio_achieved,
        area_ratio_intended,
    );

    // Weld-vs-defect probe: is a degenerate (area~0) cell's generator actually
    // near-coincident with a neighbour (-> weld), or normally spaced (-> clipping
    // defect)? Compare global near-coincidence counts to the degenerate count,
    // and look at the nn-distance of the degenerate cells themselves.
    let below = |t: f32| nn_dist.iter().filter(|&&d| d < t).count();
    let mut degen_nn: Vec<f32> = (0..cells)
        .filter(|&i| areas[i] <= 1e-9)
        .map(|i| nn_dist[i])
        .collect();
    degen_nn.sort_by(f32::total_cmp);
    let degen_nn_med = degen_nn.get(degen_nn.len() / 2).copied().unwrap_or(f32::NAN);
    let degen_nn_max = degen_nn.last().copied().unwrap_or(f32::NAN);
    let degen_below_1e6 = degen_nn.iter().filter(|&&d| d < 1e-6).count();
    // Attribution: each degenerate cell's spacing RELATIVE to its local target.
    // rho << 1 => a genuine sliver (close-pair input, hex3's sampler).
    // rho ~ 1 => normally-spaced generator that came out degenerate anyway
    //            (s2-voronoi clipping robustness, independent of input quality).
    let mut degen_rho: Vec<f32> = (0..cells)
        .filter(|&i| areas[i] <= 1e-9)
        .map(|i| rho[i])
        .collect();
    degen_rho.sort_by(f32::total_cmp);
    let degen_rho_med = degen_rho.get(degen_rho.len() / 2).copied().unwrap_or(f32::NAN);
    let degen_rho_ge_07 = degen_rho.iter().filter(|&&r| r >= 0.7).count();
    println!(
        "    weld-probe: nn<1e-6={} nn<1e-5={} nn<1e-4={} | degen={} (nn<1e-6: {}, median nn={:.2e}, max nn={:.2e})",
        below(1e-6),
        below(1e-5),
        below(1e-4),
        degen,
        degen_below_1e6,
        degen_nn_med,
        degen_nn_max,
    );
    println!(
        "    attribution: of {} degenerate cells -> median rho={:.2}, with rho>=0.7 (normal spacing): {} ({:.1}%)",
        degen,
        degen_rho_med,
        degen_rho_ge_07,
        degen_rho_ge_07 as f32 / degen.max(1) as f32 * 100.0,
    );
}

fn print_header() {
    println!(
        "{:<14} {:>9} {:>9}  {:>8} {:>6} {:>6}  {:>5} {:>5} {:>5} {:>5} {:>5}  {:>6} {:>6}  {:>6}  {:>11}",
        "algo", "sample", "tess", "cells", "err", "degen", "rMin", "rP1", "rP5", "rMed", "rCoV",
        "slv<.5", "slv<.3", "bandCoV", "areaRatio a/i",
    );
    println!("{}", "-".repeat(128));
}

/// Mean per-band area CoV (density removed by binning), plus achieved vs
/// intended coarse:fine area ratio between the lowest and highest density bands.
fn band_stats(areas: &[f32], density: &[f32]) -> (f32, f32, f32) {
    const BANDS: usize = 8;
    let dmin = density.iter().copied().fold(f32::INFINITY, f32::min).max(1e-6);
    let dmax = density.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_min = dmin.ln();
    let log_span = (dmax.ln() - log_min).max(1e-6);

    let mut sum = vec![0.0f64; BANDS];
    let mut sumsq = vec![0.0f64; BANDS];
    let mut count = vec![0u64; BANDS];
    let mut dsum = vec![0.0f64; BANDS];

    for (&a, &d) in areas.iter().zip(density.iter()) {
        if a <= 1e-9 {
            continue; // degenerate clipping cell
        }
        let t = ((d.ln() - log_min) / log_span).clamp(0.0, 0.9999);
        let b = (t * BANDS as f32) as usize;
        sum[b] += a as f64;
        sumsq[b] += (a as f64) * (a as f64);
        count[b] += 1;
        dsum[b] += d as f64;
    }

    let mut cov_sum = 0.0f64;
    let mut cov_n = 0u32;
    for b in 0..BANDS {
        if count[b] < 16 {
            continue;
        }
        let mean = sum[b] / count[b] as f64;
        let var = (sumsq[b] / count[b] as f64 - mean * mean).max(0.0);
        if mean > 0.0 {
            cov_sum += var.sqrt() / mean;
            cov_n += 1;
        }
    }
    let band_cov = if cov_n > 0 {
        (cov_sum / cov_n as f64) as f32
    } else {
        f32::NAN
    };

    // lowest vs highest populated band
    let lo = (0..BANDS).find(|&b| count[b] >= 16);
    let hi = (0..BANDS).rev().find(|&b| count[b] >= 16);
    let (achieved, intended) = match (lo, hi) {
        (Some(lo), Some(hi)) if lo != hi => {
            let mean_lo = sum[lo] / count[lo] as f64;
            let mean_hi = sum[hi] / count[hi] as f64;
            let d_lo = dsum[lo] / count[lo] as f64;
            let d_hi = dsum[hi] / count[hi] as f64;
            ((mean_lo / mean_hi) as f32, (d_hi / d_lo) as f32)
        }
        _ => (f32::NAN, f32::NAN),
    };

    (band_cov, achieved, intended)
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

fn cov(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    if mean > 0.0 {
        (var.sqrt() / mean) as f32
    } else {
        0.0
    }
}

// ----------------------------------------------------------------------------
// Heap item with f32 ordering
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
struct HeapItem {
    weight: f32,
    idx: usize,
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then(self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ----------------------------------------------------------------------------
// Point helpers
// ----------------------------------------------------------------------------

fn fibonacci_point(i: usize, n: usize) -> Vec3 {
    let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = std::f32::consts::TAU * i as f32 / PHI;
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

fn jittered_fibonacci_point(i: usize, n: usize, jitter: f32, seed: u64) -> Vec3 {
    let mut p = fibonacci_point(i, n);
    if jitter > 0.0 {
        let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = p.cross(arbitrary).normalize();
        let v = p.cross(u);
        let angle = hash_unit(seed, i as u64, 0) * std::f32::consts::TAU;
        let tangent = u * angle.cos() + v * angle.sin();
        p = (p + tangent * jitter).normalize();
    }
    p
}

fn random_unit<R: Rng>(rng: &mut R) -> Vec3 {
    let y: f32 = rng.gen_range(-1.0..1.0);
    let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    let r = (1.0 - y * y).sqrt();
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

fn fisher_yates<R: Rng>(v: &mut [Vec3], rng: &mut R) {
    for i in (1..v.len()).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

fn hash_unit(seed: u64, index: u64, stream: u64) -> f32 {
    let value = splitmix64(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ stream);
    ((value >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
