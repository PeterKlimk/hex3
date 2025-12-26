# Plan: Extract `s2-voronoi` Crate + Refactor Hex3

## Goals

- Extract a publishable crate, `s2-voronoi`, that computes spherical (unit-sphere / S2) Voronoi diagrams
- Rename "gpu_voronoi" to "knn_clipping" (it's CPU-based, kNN + great-circle half-space clipping)
- Refactor first, then improve testing, then tune epsilons/errors, then add fallbacks, then stress test
- Clean up test infrastructure (currently scattered and mixed concerns)
- Keep hex3 free to reorganize (no external users yet)

## Non-goals (first pass)

- Moving Lloyd relaxation or point generation to s2-voronoi (defer)
- Moving adjacency computation to s2-voronoi (keep in hex3)
- Stable-only compilation (nice-to-have later)
- Splitting hex3 rendering/app into separate crates

## Workspace layout

```
hex3/                      # workspace root
├── Cargo.toml             # [workspace]
├── crates/
│   └── s2-voronoi/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs
│       │   ├── diagram.rs           # SphericalVoronoi + storage
│       │   ├── error.rs             # Error types (NEW)
│       │   ├── types.rs             # UnitVec3, UnitVec3Like trait
│       │   ├── knn_clipping/        # renamed from gpu_voronoi
│       │   │   ├── mod.rs
│       │   │   ├── cell_builder.rs
│       │   │   ├── ...
│       │   └── cube_grid/           # kNN spatial index
│       ├── benches/                 # criterion benchmarks (NEW)
│       │   └── voronoi_bench.rs
│       └── tests/                   # integration tests
│           └── correctness.rs
└── src/                   # hex3 app + worldgen + rendering
```

## Module placement decisions

| Current location | Destination | Notes |
|------------------|-------------|-------|
| `geometry/voronoi.rs` | s2-voronoi | Core SphericalVoronoi type |
| `geometry/gpu_voronoi/` | s2-voronoi/knn_clipping/ | Main algorithm, renamed |
| `geometry/cube_grid/` | s2-voronoi/cube_grid/ | kNN spatial index |
| `geometry/convex_hull.rs` | s2-voronoi (feature) | qhull backend, test-only |
| `geometry/validation.rs` | s2-voronoi (partial) | Public validation helpers |
| `geometry/sphere.rs` | hex3 | Point generation stays for now |
| `geometry/lloyd.rs` | hex3 | Lloyd relaxation stays for now |
| `geometry/mesh.rs` | hex3 | Map projection, rendering |
| `world/tessellation.rs` | hex3 | Uses s2-voronoi, owns adjacency |

## Naming decisions

- Crate: `s2-voronoi` (S2 = unit sphere)
- Backend rename: `gpu_voronoi` → `knn_clipping`
- CLI flag: `--gpu-voronoi` → `--voronoi-backend knn-clip`

## Public API

### Core types

```rust
/// Dependency-free point representation for stable ABI
#[repr(C)]
pub struct UnitVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Trait for zero-copy input from various math libraries
pub trait UnitVec3Like {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn z(&self) -> f32;
}

// Always implemented for:
impl UnitVec3Like for UnitVec3 { ... }
impl UnitVec3Like for [f32; 3] { ... }

// With `glam` feature:
impl UnitVec3Like for glam::Vec3 { ... }
```

### Diagram output

```rust
pub struct SphericalVoronoi {
    pub generators: Vec<UnitVec3>,
    pub vertices: Vec<UnitVec3>,
    cells: Vec<CellData>,        // internal
    cell_indices: Vec<u32>,      // internal
}

impl SphericalVoronoi {
    pub fn num_cells(&self) -> usize;
    pub fn cell(&self, index: usize) -> CellView<'_>;
    pub fn iter_cells(&self) -> impl Iterator<Item = CellView<'_>>;
}

pub struct CellView<'a> {
    pub generator_index: usize,
    pub vertex_indices: &'a [u32],
}
```

### Compute API

```rust
/// Compute with default settings (knn-clipping backend).
///
/// Returns a diagram plus diagnostics. Errors are reserved for invalid inputs
/// (e.g. insufficient points) or unrecoverable internal failures.
pub fn compute<P: UnitVec3Like>(points: &[P]) -> Result<VoronoiOutput, VoronoiError>;

/// Compute with explicit backend/config
pub fn compute_with<P: UnitVec3Like>(
    points: &[P],
    config: VoronoiConfig,
) -> Result<VoronoiOutput, VoronoiError>;

pub struct VoronoiConfig {
    pub termination: TerminationConfig,
    // future: knn schedule, preprocessing options
}

impl Default for VoronoiConfig { ... }

pub struct VoronoiOutput {
    pub diagram: SphericalVoronoi,
    pub diagnostics: VoronoiDiagnostics,
}

pub struct VoronoiDiagnostics {
    /// Cells with <3 vertices (effectively empty/invalid for adjacency).
    pub bad_cells: Vec<usize>,
    /// Cells with duplicate vertex indices (degenerate polygons).
    pub degenerate_cells: Vec<usize>,
    /// Cells with no neighbors after adjacency build (if computed externally).
    pub orphan_cells: Vec<usize>,
    /// (Optional) backend-specific counters, timing, termination histograms, etc.
}
```

### Error handling (NEW)

```rust
#[derive(Debug, thiserror::Error)]
pub enum VoronoiError {
    #[error("insufficient points: need at least 4, got {0}")]
    InsufficientPoints(usize),

    #[error("degenerate input: {0} coincident point pairs")]
    DegenerateInput(usize),

    #[error("computation failed: {0}")]
    ComputationFailed(String),
}
```

Note: In the initial extraction pass, “bad/degenerate cells” are expected for some inputs and
should generally be reported via `VoronoiDiagnostics` rather than returned as `Err`.

## Feature flags

| Feature | Description | Default |
|---------|-------------|---------|
| `glam` | `UnitVec3Like` impl for `glam::Vec3` | off |
| `timing` | Detailed timing instrumentation | off |
| `parallel` | Rayon parallelism in backends | on (default feature) |
| `qhull` | Convex hull backend (test/bench only) | off |

Note: glam is always an internal dependency (used for computation). The `glam` feature only controls the public trait impl.

## Implementation phases

### Phase 0: Sequence of work (big picture)

1. Refactor/extract with minimal behavioral change
2. Improve and reorganize tests (fast correctness first)
3. Revisit epsilons + error classification
4. Implement hierarchical fallback (higher precision) for uncertified/failed/degenerate cells
5. Stress test + benchmark across scales and input families

### Phase 1: Workspace setup
- Create `Cargo.toml` with `[workspace]`
- Create `crates/s2-voronoi/` skeleton
- Add core types: `UnitVec3`, `UnitVec3Like`, error types
- Verify `cargo build` works

### Phase 2: Move knn_clipping backend
- Move `geometry/gpu_voronoi/` → `s2-voronoi/src/knn_clipping/`
- Move `geometry/cube_grid/` → `s2-voronoi/src/cube_grid/`
- Update imports, replace `glam::Vec3` inputs with trait bounds
- Implement `compute()` and `compute_with()` API

### Phase 3: Integrate hex3
- Update hex3 `Cargo.toml` to depend on `s2-voronoi`
- Update `tessellation.rs` to use `s2_voronoi::compute()`
- Keep `build_adjacency` in hex3
- Update CLI: `--gpu-voronoi` → `--voronoi-backend`

### Phase 4: qhull backend (test-only)
- Move `convex_hull.rs` behind `qhull` feature
- Use only in tests/benches for ground-truth comparison
- Keep as dev-dependency intent

### Phase 5: Test restructuring (phase 1)

Current state:
- 15+ inline `#[cfg(test)]` modules scattered across files
- Massive test files (gpu_voronoi/tests.rs ~1500 lines)
- Mixed concerns: unit tests, stress tests, timing, validation
- Many `#[ignore]` tests that are really benchmarks

Target state:

**s2-voronoi tests (`crates/s2-voronoi/tests/`):**
```
tests/
├── api.rs           # Public API tests
├── correctness.rs   # Geometric invariants
└── edge_cases.rs    # Coincident points, small inputs
```

**s2-voronoi benchmarks (`crates/s2-voronoi/benches/`):**
```
benches/
└── voronoi_bench.rs  # criterion: 10k, 100k, 500k, 1M points
```

**hex3 integration tests (`tests/`):**
```
tests/
├── tessellation.rs   # Full pipeline: points → voronoi → adjacency
└── generation.rs     # World generation smoke tests
```

Test categories:
1. **Unit tests** (inline `#[cfg(test)]`) - small, fast, test one thing
2. **Integration tests** (`tests/`) - full pipeline, public API only
3. **Benchmarks** (`benches/`) - criterion, no assertions, timing only
4. **Property tests** (later phase) - geometric invariants once fallbacks are implemented and
   “success criteria” for degenerate inputs are well-defined.

### Phase 6: Epsilons + error classification (after tests)
- Define what constitutes:
  - “invalid input” (hard error)
  - “computable but degraded” (diagnostic)
  - “cell build failure requiring fallback” (recoverable)
- Audit and tune tolerances/epsilons with regression tests.

### Phase 7: Hierarchical fallback (precision ladder)

Goal: recover from uncertified / failed / degenerate cells by escalating precision:
1. f32 fast path (current)
2. f64 (already present for parts of the cell builder)
3. bigfloat
4. bigrational (most robust, slowest; likely last resort)

Implementation notes:
- Keep fallback internal to the backend initially; expose only diagnostics and final output.
- Add targeted tests that force each fallback tier to trigger.

### Phase 8: Stress tests + benchmarks
- Add/curate stress suites for:
  - near-duplicates
  - clustered inputs
  - very large N
  - adversarial seam/cubemap boundary cases
- Keep long-running stress tests `#[ignore]` and run them manually/CI nightly if desired.

### Phase 9: Documentation
- Doc comments on public API
- `//!` module docs explaining algorithm
- Examples in doc comments
- README.md for s2-voronoi crate

## Validation checklist

- [ ] `cargo build` passes in WSL2
- [ ] `cargo test` passes (both crates)
- [ ] `cargo test --release` passes
- [ ] `cargo clippy` clean
- [ ] `cargo doc` builds without warnings
- [ ] Windows: `cargo run --release` works (wgpu + compute shaders)
- [ ] Benchmarks run with criterion

## Open questions (resolved)

| Question | Decision |
|----------|----------|
| Normalize inputs or require unit vectors? | Debug-assert only (caller's responsibility) |
| Adjacency helper in s2-voronoi? | No, keep in hex3 |
| glam as internal or feature? | Internal dep, feature for trait impl |
| Lloyd/point gen in s2-voronoi? | No, keep in hex3 for now |
| qhull as real backend? | No, test-only |

## Future considerations (not this pass)

- Move Lloyd relaxation to s2-voronoi as optional convenience
- Move point generation (Fibonacci lattice) to s2-voronoi
- Adjacency helper as optional s2-voronoi feature
- SIMD acceleration with scalar fallback
- Publish to crates.io
