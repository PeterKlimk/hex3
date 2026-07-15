# Organization-owner planar capture and human-review amendment V0

**Date:** 2026-07-16

**Status:** executable preregistration for deterministic planar capture,
procedurally masked human observation and reveal; not implemented, not an arm
result and not a product-promotion verdict

**Parents:** [organization-owner comparison design V0](orogen-organization-owner-v0-2026-07-16.md),
[artifact/provenance amendment V0](orogen-organization-artifact-v0-2026-07-16.md),
[numerical/admission amendment V0](orogen-organization-numerical-v0-2026-07-16.md),
[evidence/projection amendment V0](orogen-organization-evidence-v0-2026-07-16.md),
[presentation policy](../presentation.md) and
[validation policy](../validation.md)

## Decision

Freeze a small deterministic CPU planar renderer and four immutable review
boundaries:

1. a withheld capture-authority root binding exact arm identities, source
   artifacts, tiles, sheets and the alias permutation;
2. separate reviewer-visible primary-4-km and resolution packets containing
   aliases and neutral instructions but no arm identities or native-G state;
3. immutable primary and resolution observation roots, each published before
   the next phase becomes available; and
4. a reveal root joining the authority, both observation roots, the alias key
   and separate G-authored provenance sheets.

An optional post-reveal interpretation is a fifth, separate authored record. It
cannot edit blind answers or the existing immutable
`OrganizationComparisonV0`, whose `human_review=MissingQuantity` remains an
accurate statement about that pre-presentation root. The reveal root records
that descriptive human evidence now exists; it does not rewrite that field to
`Supported` or turn ratings into a mechanical vote.

The renderer consumes the accepted planar mesh, frozen physical cell-mean
surface, shared-input forcing and independent S0/D0/reference-O0a evidence. It
does not call product `World`, read native H/C state, use O0b to decorate an
arm, select a camera from morphology or invoke particles. A separately owned
post-reveal G-provenance renderer may read only the fully validated native G
forest checkpoint. That exception cannot enter a procedurally masked common
tile.

The existing offscreen wgpu sweep is useful product-rendering precedent and
works on Linux, but is not this authority: it generates product `World`, uses
spherical/result-selected cameras and has backend-dependent pixels. V0 adds no
product renderer feature and makes no claim about interactive GPU fidelity.

## Scope corrections and narrow choices

- The common overlay means **reference O0a**, not cross-arm O0b. O0b remains
  numeric correspondence evidence and never becomes an identifying visual
  annotation.
- Per-cell forcing support comes from the accepted shared-input compiled
  segment stencils. It is arm-neutral input, not native terrain state.
- Planar Physical is exact vertical exaggeration `1.0`. Planar
  authentic/cartographic evaluates `0.04/(10.0/6371.0)` in that order and then
  requires binary64 bits `0x40397be76c8b4395`. Captions round this to `25.484x`;
  the decimal literal `25.484` (one ULP higher) is never the semantic value.
  The globe shader scale `0.04` itself is not used as a planar kilometre
  transform.
- North-up orthographic plan capture changes vertical exaggeration through the
  declared surface normal and hillshade. It has no perspective, parallax,
  horizon, cast shadows or result-dependent framing.
- Masking is procedural, not cryptographic. The parent requires a public
  literal seed, so a motivated reviewer can derive the six possible mappings.
  The reviewer-visible protocol omits the key and requires an explicit no-peek/
  no-derivation attestation; it does not claim double-blind evidence.
- A complete three-arm root produces A/B/C. If one of C/G lacks a valid required base rung
  evidence, the valid H pair may produce an exact A/B pairwise packet bound to
  its accepted pairwise-comparison root. H is mandatory and a one-arm packet is
  illegal. Every arm admitted by either comparison root has valid 8/4/2 base
  evidence; there is no 4-km-only review authority in V0.
- Physical, diagnostic and cartographic observations remain separate fields.
  No mean score, winner count, confidence-weighted rank or Pareto flag is
  derived from them.

## Registered identities and limits

```text
capture authority schema         orogen-owner-planar-capture-authority-v0
reviewer packet schema           orogen-owner-reviewer-packet-v0
review protocol                  orogen-owner-planar-review-protocol-v0
primary observation schema       orogen-owner-primary-observation-v0
resolution observation schema    orogen-owner-resolution-observation-v0
reveal schema                    orogen-owner-review-reveal-v0
interpretation schema            orogen-owner-post-reveal-interpretation-v0
capture failure schema           orogen-owner-capture-failure-v0
JSON projection family           orogen-owner-planar-review-json-v0
renderer contract                orogen-owner-cpu-planar-raster-v0
palette contract                 orogen-owner-planar-palettes-v0
legend contract                  no-raster-text-bound-utf8-sidecars-v0
PNG encoding                     png-0.17.16-rgba8-srgb-perceptual-best-no-filter-v0
semantic hash encoding           fnv1a64-bincode-fixint-le-v0
permutation contract             fnv1a64-chacha8-fisher-yates-v0
math repeat scope                identical-executable-target-os-cpu-v0

maximum capture authority bytes      512 MiB
maximum reviewer packet bytes        256 MiB
maximum observation/reveal bytes       8 MiB each
maximum interpretation bytes           1 MiB
maximum capture failure bytes           8 MiB
maximum one raw RGBA tile bytes        8 MiB
maximum one tile/provenance PNG bytes 32 MiB
maximum one sheet PNG bytes          512 MiB
maximum authority directory            4 GiB
maximum reviewer-packet directory       2 GiB
maximum capture bundle                  6 GiB
maximum observation/interpretation dir 64 MiB
maximum reveal directory                128 MiB
maximum semantic string bytes          4096
maximum reviewer IDs                     16
maximum aliases                           3
maximum tiles                           256
maximum sheets                           32
maximum narrative UTF-8 bytes          4096 per field
```

Rust declaration order is wire order; enum declaration order gives zero-based
discriminants. Fixed-integer little-endian bincode, finite canonical floats,
positive zero, bounded visitors, checked nested counts, trailing-byte rejection
and FNV root conventions are inherited from the evidence amendment. Decoders
never reserve an untrusted announced capacity and check raw caps before decode.

## Exact predecessor and cohort boundary

```rust
pub enum ReviewComparisonAuthorityV0 {
    Complete { comparison_hash: u64 },
    Pairwise {
        arm_a: OrganizationArmV0,
        arm_b: OrganizationArmV0,
        pairwise_comparison_hash: u64,
        excluded_arm: OrganizationArmV0,
        excluded_arm_nonadmission_hash: u64,
    },
}

pub struct CaptureArmBindingV0 {
    pub arm: OrganizationArmV0,
    pub base_8km_result_hash: u64,
    pub base_8km_evidence_hash: u64,
    pub base_4km_result_hash: u64,
    pub base_4km_evidence_hash: u64,
    pub base_2km_result_hash: u64,
    pub base_2km_evidence_hash: u64,
}
```

For complete authority, bindings are H/C/G. All eleven evidence roots and the
complete comparison validate before capture. For pairwise authority, bindings
are exactly H/C or H/G; the exact H/C or H/G pairwise comparison and every
8/4/2 base root it binds validate. The excluded arm is respectively G or C and
its hash binds the typed semantic base-run failure that prevented one required
8/4/2 base rung. A resource stop blocks review rather than authorizing fallback.
A valid excluded arm, common-
extractor/comparison failure, or mere absence of a complete root cannot
authorize pairwise fallback. Exactly one non-H arm must remain fully admitted;
if both non-H arms have genuine nonadmission, review is blocked. C/G
without H is illegal. The capture cohort is selected only
from semantic success roots; a typed invalid-arm failure is not rendered as a
terrain.

Full validation receives the accepted input bundle, every bound arm-run and
evidence directory, their required control/G-4 predecessors, and the complete
or pairwise comparison directory. It validates and rebuilds them under the
preceding amendments. The renderer reads:

- the exact stored 8/4/2 `LandscapeMesh` and final physical `f64` elevation;
- the matching common core, reference O0a and central projection binary;
- declarative segment geometry and compiled per-cell forcing stencils from the
  accepted input; and
- only for post-reveal G provenance, the validated native G forest checkpoint
  bound by G provenance.

Central JSON is insufficient because it omits member vectors. Native H/C trace,
flux, receiver and process arrays are forbidden. Native G is forbidden outside
the provenance builder. A source mutation that repairs outer hashes but changes
any drawable field rejects on full rebuild.

## Deterministic planar renderer

The renderer is a pure CPU function over validated values. It uses IEEE-754
binary64, one thread, round-to-nearest, no FMA, no SIMD-dependent reduction and
the evidence amendment's Neumaier recurrence where a sum is needed. Rendering
does not alter or cache back into any predecessor.

### View matrix and pixel coordinates

```rust
pub enum PlanarReviewViewV0 { FullDomain, Central, Transfer }

pub struct PlanarViewSpecV0 {
    pub view: PlanarReviewViewV0,
    pub source_extent_km: Option<[f64; 4]>,
    pub bounds_km: [f64; 4],
    pub width_px: u32,
    pub height_px: u32,
    pub north_up: bool,
    pub orthographic: bool,
}
```

View order and rasters are exact:

| view | bounds `[xmin,xmax,ymin,ymax]` km | raster |
|---|---|---:|
| `FullDomain` | minimal centred 3:2 expansion containing the stored control-volume vertex union across every admitted available resolution | 1440×960 |
| `Central` | `[-320,320,-160,160]` | 1440×720 |
| `Transfer` | `[-120,40,-72,72]` | 1200×1080 |

Require finite strict bounds. `source_extent_km` is the exact min/max union over
all admitted available resolution meshes only for `FullDomain` and `None`
otherwise. For FullDomain, set
`s=max((xmax-xmin)/1440,(ymax-ymin)/960)` and expand symmetrically about the
union centre to bounds of width `1440*s` and height `960*s`. This is the sole
padding rule: it preserves one kilometre per kilometre and never crops. The
fixed Central and Transfer bounds already match their raster aspects. For
zero-based pixel `(u,v)`, sample only its center:

```text
x = xmin + (u + 0.5) * (xmax-xmin) / width
y = ymax - (v + 0.5) * (ymax-ymin) / height
```

Thus +Y is north/up and +X east/right. Matrices, bounds and sample coordinates
are bit-equal across every arm and resolution in the capture. No multisampling
or antialiasing is used in authoritative tiles.

### Cell ownership and edge rules

Ownership geometry is the fully validated common core's
`EvaluationSurfaceGraphV0.cell_polygon_offsets` and
`cell_polygon_vertices_km`, bound by its common-core hash; `LandscapeMesh`
centres alone are not polygon authority. For each view, build a renderer-owned
64×64-pixel bin index. Visit polygons in ascending cell ID, transform their
finite vertex bounding box to continuous raster coordinates, and append the ID
to every clamped bin whose closed rectangle intersects that box. Because visit
order is ascending, each bin list is already sorted; the index is acceleration
only and a manufactured exhaustive scan must give identical owners.

At each pixel, query its bin, then test candidate polygons in ascending cell ID
using accepted binary64 vertex order. A point is inside under the half-open
top-left winding rule: top or left
edges include, bottom or right edges exclude in screen coordinates. A point
bit-equal to a shared vertex/edge that remains claimed by multiple polygons is
owned by the smallest cell ID. No owner yields background. Degenerate,
self-intersecting, clockwise-when-canonical-counterclockwise or overlapping
non-shared polygons are instrument failures, not holes painted over by a
nearest-cell fallback.

The exact orientation-independent test uses the even--odd ray rule. For each
directed screen-space edge `(x0,y0)->(x1,y1)`, first detect an exact collinear
point within its closed bounding box. Such a boundary point is included iff
`(y1-y0)<0` or `((y1-y0)==0 && (x1-x0)>0)`; otherwise it is excluded. For a
non-boundary point, toggle inside iff `(y0>py)!=(y1>py)` and
`px < x0+(py-y0)*(x1-x0)/(y1-y0)`, evaluated as written without FMA. This is
the only top-left predicate for polygons and triangles.

All cell-valued maps are piecewise constant at the owning cell. This preserves
the accepted cell-mean representation and does not invent a smooth physical
surface. Lines and markers use the subpixel rules below and never change cell
ownership.

### Pixel arithmetic and encoding

Palette anchors are stored as eight-bit sRGB. Interpolate anchor channels in
binary64 **linear-light** after `c=s/12.92` for `s<=0.04045` and
`c=((s+0.055)/1.055)^2.4` otherwise, where `s=channel/255`; clamp the normalized
scalar before interpolation. Encode with `s=12.92*c` for `c<=0.0031308` and
`s=1.055*c^(1/2.4)-0.055` otherwise, clamp, then round
`floor(255*s+0.5)`. Hillshade multiplies linear RGB before this encoding. Alpha is
straight, not premultiplied. Background is opaque `#f4f2ed`; missing values are
opaque `#ff00ff`. Any missing value in a required primary field is an instrument
failure; magenta is used only by manufactured null fixtures and optional
supplementary annotations.

Authoritative raw tiles are tightly packed row-major RGBA8. PNG uses the locked
`png` crate version in `Cargo.lock`, RGBA8, sRGB intent, `Compression::Best`,
`FilterType::NoFilter` and
`set_source_srgb(SrgbRenderingIntent::Perceptual)`, then `write_header` and one
`write_image_data` call. Chunk order is signature, IHDR, sRGB, one or more
consecutive IDAT chunks containing one zlib stream as emitted by locked png
0.17.16, then IEND; no other chunk is legal. PNG decode must reproduce raw RGBA exactly. Semantic
identity binds dimensions, raw-RGBA FNV and PNG byte FNV; backend-independent
raw pixels, not approximate image similarity, are the renderer authority. Here
backend-independent means independent of GPU/wgpu backend. Because `sqrt` and
`powf` are platform math, bit-repeat is claimed only for identical executable
bytes, target triple, OS/libc and CPU identifier recorded below. Cross-platform
images are presentation-equivalent candidates, not equal semantic roots.

For any palette, choose the lowest anchor interval `k` satisfying
`value<=anchor[k+1]` after range clamp, set
`u=(value-anchor[k])/(anchor[k+1]-anchor[k])`, and compute each linear channel
as `(1-u)*channel[k]+u*channel[k+1]` in that order without FMA. Return an exact
endpoint anchor directly. This applies to uniform Viridis and the nonuniform
hypsometric elevation anchors.

### Sidecar legends and primitive rasterization

No authoritative tile, sheet or provenance PNG contains rasterized text,
caption boxes, swatches or metadata. Exact row/column/view/layer identity,
ranges, exaggeration and legend wording live in the bound packet README, tile/
sheet specs and reveal sidecar. This deliberately avoids adding a custom font
renderer to a terrain-architecture discriminator. Magenta manufactured-null
pixels are the only in-image diagnostic annotation.

One-pixel lines use integer Bresenham after clipping to the tile. Wider lines
paint every pixel whose center has squared Euclidean distance at most
`(width_px/2)^2` from the clipped segment; ties include. Filled circles use the
same center-distance rule. Triangles use the top-left rule. Dashed lines use
eight pixels on, four off, with phase zero at the lexicographically smaller
endpoint. Primitive depth is explicit layer order only.

World endpoints first transform to continuous raster coordinates by the inverse
pixel-centre map. Bresenham endpoints round `floor(q+0.5)` after clipping.
Radius-four means centre distance `<=4`; the saddle triangle vertices relative
to its continuous anchor are `(0,-4),(-4,3),(4,3)`. A seven/nine-pixel square
contains pixel centres with maximum axis distance `<=3.5`/`<=4.5`; hollow
means the outer square minus the corresponding five/seven-pixel inner square.

Clip world segments with Liang--Barsky in boundary order left, right, bottom,
top. With `d=end-start`, use `(p,q)=(-dx,x0-xmin),(dx,xmax-x0),
(-dy,y0-ymin),(dy,ymax-y0)`, initial `[t0,t1]=[0,1]`; `p==0 && q<0` rejects,
`p==0` otherwise continues, `r=q/p`, `p<0` updates `t0=max(t0,r)`, `p>0`
updates `t1=min(t1,r)`, and `t0>t1` rejects. Equality includes the boundary.
Convert accepted endpoints to raster coordinates, round as above, then clamp
integer X to `[0,width-1]` and Y to `[0,height-1]`. Exact
one-pixel Bresenham is:

```text
dx=abs(x1-x0); sx=if x0<x1 {1} else {-1}
dy=-abs(y1-y0); sy=if y0<y1 {1} else {-1}; err=dx+dy
loop: paint(x0,y0); if x0==x1 && y0==y1 break
      e2=2*err
      if e2>=dy { err+=dy; x0+=sx }
      if e2<=dx { err+=dx; y0+=sy }
```

For dashed wide lines, project a candidate pixel centre onto the clipped
continuous segment, clamp to its endpoints, measure screen-pixel distance from
the lexicographically smaller clipped endpoint, and paint iff
`floor(distance) % 12 < 8`; zero-length segments use distance zero. One-pixel
dashes apply the same test to each Bresenham pixel centre. All arithmetic is
binary64 without FMA under the registered repeat scope.

The raster buffer remains opaque linear-light RGB in binary64 until final sRGB
encoding. Straight-alpha fill composition is exactly
`dst=src*a+dst*(1-a)` per channel in segment order, with no intermediate
eight-bit rounding. Opaque primitive colors are sRGB-decoded then replace the
covered linear pixel. Within each overlay family traverse accepted records by
their semantic ID ascending; faces use `(min_cell,max_cell,local_face)` order
and an identical shared face is drawn once. Reaches, portals, peaks and saddles
use reach/portal/peak/saddle ID. This order plus the family order below is the
only compositing order.

## Common layers, palettes and hillshade

```rust
pub enum CommonCaptureLayerV0 {
    PhysicalElevationFixed,
    PhysicalElevationSupplementary,
    PhysicalHillshade1x,
    PhysicalGradeFixed,
    ForcingIndependentEvidence,
    AuthenticCartographic,
}
```

The supplementary variant is present only under the frozen ladder below.
Every present layer is rendered for every alias/view/resolution with identical
settings.

### Fixed physical elevation

Normalize `t=(z_km+0.25)/4.25`, clamp to `[0,1]` and interpolate these Viridis
anchors at `t=[0,0.25,0.5,0.75,1]`:

```text
#440154 #3b528b #21918c #5ec962 #fde725
```

The sidecar legend states `physical elevation km; fixed [-0.25,4.0]`. Saturation is
computed over **source cells whose centers lie within the view bounds**, not
raster pixels: count and Neumaier physical area separately for `z<-0.25` and
`z>4.0`; equality is not saturated. Store eligible count/area and all four
under/over witnesses. No water styling appears in this diagnostic layer.

### Physical grade

Rebuild the per-cell planar physical gradient with the accepted
`reconstruct_mean_surface_gradient` over the stored mesh and final cell-mean
elevation; let `(gx,gy)` be its horizontal components and
`grade=sqrt(gx*gx+gy*gy)` in that evaluation order without FMA. Full validation
requires its weighted distribution to reproduce common evidence. Normalize
`grade/0.10`, clamp and interpolate in
linear light from `#ffffff` to `#111111`.
The sidecar legend states `physical grade; fixed [0,0.10]; angle=atan(grade)`. Report
over-range count and physical area; negative grade is invalid. Any unavailable
cell is an instrument failure; no unavailable count is published. No snow,
rock, biome, moisture or water category is applied.

### Physical and authentic shaded relief

For owning cell grade vector `(gx,gy)`, vertical exaggeration `E` gives the
upward normal:

```text
n = normalize([-E*gx, -E*gy, 1])
light = [-0.5, 0.5, f64::from_bits(0x3fe6a09e667f3bcd)]
shade = 0.35 + 0.65*max(0, dot(n,light))
```

Compute the normal denominator as
`sqrt((E*gx)*(E*gx)+(E*gy)*(E*gy)+1.0)` in that written order without FMA,
then divide each component. The stored light vector is exactly the three values
shown; it is the normalized `[-1,1,sqrt(2)]` result without runtime normalization
ambiguity and is copied bit-for-bit into reviewer disclosure.

`PhysicalHillshade1x` uses `E=1.0`, neutral land `#c8c3b8` and the shade above.
`AuthenticCartographic` uses `E=f64::from_bits(0x40397be76c8b4395)` and fixed
hypsometric anchors:

```text
z km      -0.25      0.0       1.0       2.0       3.0       4.0
color     #8fc4dc    #d8c9a3   #8fa56b   #8c795f   #b9aaa0   #f4f4f2
```

Interpolate by elevation in linear light and clip outside the endpoints. Cells
with `z<=0` use flat water `#8fc4dc`, shade `1.0`, plus a one-pixel `#4d87a0`
datum shoreline on faces whose neighbour is `z>0`; this is explicitly
cartographic datum water, not a modeled lake/ocean classification. Physical
hillshade uses no water flattening. Both sidecar legends state exact exaggeration and
lighting; there are no cast shadows, specular, snow, rock or moisture heuristics.

### Frozen supplementary range ladder

Primary fixed images are always retained. Before any reviewer packet is
assembled, inspect every admitted arm at every available resolution for one
view using the source-cell count and area witnesses. If any such arm/resolution
has over- or under-range **area fraction** greater than `0.005`, add one matched
supplementary elevation tile for every admitted arm and available resolution in
that view using the first ladder range whose two saturation fractions are each
`<=0.005` for every available arm/resolution:

```text
[-0.5,8.0] km
[-1.0,16.0] km
[-2.0,32.0] km
```

If the last still saturates, use it and retain the counts. Normalize the same
Viridis anchors over the chosen range. Thus supplementary presence and range
are one decision per view across both review phases, never per arm or spacing.
The sidecar legend says `supplementary; primary
fixed range retained`. Grade has no supplementary ladder.

## Forcing and independent-evidence overlay

The overlay base is the fixed elevation palette converted to linear light and
mixed `60%` toward neutral `#d8d6cf`. Draw in this exact order:

1. compiled forcing stencil fills;
2. reference-O0a backed faces and cross-section probes;
3. S0 primary/context footprint boundaries;
4. reference-scale D0 reaches and portals;
5. S0 peaks and saddles.

Forcing is identical across arms at one resolution. Let `w_s[i]` be the stored
nonnegative compiled stencil for segment `s`, and `wmax_s` its maximum in stored
cell order. A zero/nonfinite maximum is an instrument failure. Fill the owning
pixel with straight-alpha `min(0.55,0.55*w_s[i]/wmax_s)` using segment 0
`#e66101` then segment 1 `#5e3c99` under ordinary source-over compositing. This
normalization is arm-neutral per accepted resolution and stored in the tile
spec; it is not a terrain-derived scale.

Reference O0a uses only its accepted geometry:

- `LateralBoundaryCandidate` backed faces: one-pixel solid `#fdae61`;
- retained receiver/reach axis segments participating in a selected reference
  cross-section: one-pixel solid `#1b9e77`;
- each cross-section lateral ray from axis station to its stored finite first
  catchment exit: one-pixel dashed `#1b9e77`; censored/missing sides draw no
  invented endpoint; and
- no O0b partner, component, displacement or arm-to-arm label.

S0 boundaries are exact faces separating one branch's complete footprint from
another owner/outside. Primary highlands use two-pixel `#d73027`; context
highlands use one-pixel dashed `#d73027`. Peak anchors are radius-four filled
circles `#7f0000`; retained saddles are six-pixel top-left triangles
`#762a83`. These categories come from the projection cohort and do not select a
largest-N subset.

D0 uses exactly the `2,000 km2` reference scale. Draw every retained reach's
accepted centerline source-to-outlet in `#2166ac` with width
`1+min(strahler_order,3)` pixels. Draw every semantic portal as a seven-pixel
square `#053061`, including zero central contribution. Do not use native G
receivers or the product Major-river policy.

The sidecar legend names `input forcing`, `reference O0a`, `S0 primary/context`,
`D0 2,000 km2`, `peak`, `saddle` and `portal`. Line clipping occurs against the
view rectangle before pixel conversion. Geometry outside the view neither
wraps nor chooses a new anchor.

## Separate G-authored provenance layer

`build_g_provenance_sheet_v0` is a distinct capture-time withheld renderer. It accepts
the validated 4 km G native forest checkpoint plus G common D0 and draws the
same muted fixed-elevation base. First draw independent D0 as the solid blue
reference above. Then draw a G authored receiver segment only for a cell whose
validated G accumulated area is at least `2,000 km2`: two-pixel dashed
`#b2182b` from cell centre to receiver centre, or to the midpoint of its exact
portal face. Draw G portals as hollow nine-pixel `#b2182b` squares. The reveal sidecar
states `G authored forest - native provenance, not common evidence` and
`independent D0 - solid blue`.

This layer is generated for full/central/transfer at 4 km before review, stored
only in the withheld authority until reveal, omitted from primary/resolution reviewer
packets and never used to revise common-overlay ratings. H/C native state has no
analogous sheet.

## Tile and sheet identity

```rust
pub struct SaturationWitnessV0 {
    pub eligible_cell_count: u64,
    pub eligible_area_km2: f64,
    pub below_count: u64,
    pub below_area_km2: f64,
    pub above_count: u64,
    pub above_area_km2: f64,
}

pub struct CaptureTileSpecV0 {
    pub phase: ReviewPhaseV0,
    pub alias: ReviewAliasV0,
    pub spacing: ReviewSpacingV0,
    pub view: PlanarReviewViewV0,
    pub layer: CommonCaptureLayerV0,
    pub view_spec: PlanarViewSpecV0,
    pub value_range: Option<[f64; 2]>,
    pub vertical_exaggeration: Option<f64>,
    pub saturation: Option<SaturationWitnessV0>,
    pub forcing_segment_maxima: Option<[f64; 2]>,
    pub raw_rgba_hash: u64,
    pub png_hash: u64,
    pub filename: String,
}

pub enum ReviewAliasV0 { A, B, C }
pub enum ReviewSpacingV0 { Km8, Km4, Km2 }
pub enum ReviewPhaseV0 { Primary4Km, Resolution }

pub struct SheetSpecV0 {
    pub phase: ReviewPhaseV0,
    pub view: PlanarReviewViewV0,
    pub alias: Option<ReviewAliasV0>,
    pub row_layers: Vec<CommonCaptureLayerV0>,
    pub column_aliases: Vec<ReviewAliasV0>,
    pub column_spacings: Vec<ReviewSpacingV0>,
    pub source_tile_raw_rgba_hashes: Vec<u64>,
    pub raw_rgba_hash: u64,
    pub png_hash: u64,
    pub filename: String,
}
```

Tile option legality is exact:

| layer | `value_range` | `vertical_exaggeration` | `saturation` | `forcing_segment_maxima` |
|---|---|---|---|---|
| fixed elevation | `Some([-0.25,4])` | `None` | `Some` | `None` |
| supplementary elevation | chosen range | `None` | `Some` | `None` |
| Physical hillshade | `None` | `Some(1.0)` | `None` | `None` |
| physical grade | `Some([0,0.10])` | `None` | `Some`, below fields zero | `None` |
| forcing/evidence | `None` | `None` | `None` | exact two accepted stencil maxima |
| Authentic | `Some([-0.25,4])` | exact semantic E bits | `None` | `None` |

Any other option shape rejects. Supplementary tiles exist exactly under the
view-wide ladder rule. Primary sheets have `alias=None`, all admitted aliases
in alias order and `column_spacings=[Km4]`. Resolution sheets have
`alias=Some(a)`, `column_aliases=[a]` and spacings `[Km8,Km4,Km2]`. All vectors
have exact registered cardinality; no empty or extra column is legal.

Tile order is phase, view, alias, spacing, layer. Primary uses only 4 km;
resolution uses 8,4,2. Common layer order is fixed elevation, optional
supplement immediately after its fixed parent, Physical 1x, grade, forcing/
independent evidence, Authentic. Every admitted alias has all three rungs.

Primary produces three sheets, one per view. Rows are present common layers;
columns are A/B/C or A/B. Resolution produces one sheet per alias and view;
rows are the same present common layers and columns are `8 KM`, `4 KM`, `2 KM`.
All rows use their view's authoritative tile dimensions. A sheet with `R` rows,
`C` columns and tile size `W×H` has exact canvas
`(32+C*W+8*(C-1)) × (32+R*H+8*(R-1))`. Background is `#f4f2ed`; tile `(r,c)`
has top-left `(16+c*(W+8),16+r*(H+8))`. There is no final gutter, heading,
caption or column-label band. It copies authoritative tile RGBA without
resampling. Phase/view/row/column meaning is carried by filename, specs and
README only.

Individual tiles are authority; a sheet validator recomposes exact RGBA from
them and rejects a layout mismatch. `source_tile_raw_rgba_hashes` is
row-major with layer outer and displayed column inner; swapping a PNG hash,
tile order or raw hash rejects. PNGs contain no metadata. Filenames
are exact lowercase ASCII:

```text
tile-<phase>-<view>-<alias>-<spacing>-<layer>.png
sheet-primary-<view>.png
sheet-resolution-<alias>-<view>.png
g-provenance-4km-<view>.png
```

Tokens are `primary|resolution`, `full|central|transfer`, `a|b|c`,
`8km|4km|2km`, and
`elevation-fixed|elevation-supplementary|hillshade-1x|grade-fixed|forcing-independent|authentic`.
Exact README row labels are respectively `PHYSICAL ELEVATION - FIXED`, `PHYSICAL
ELEVATION - SUPPLEMENTARY`, `PHYSICAL HILLSHADE - 1X`, `PHYSICAL GRADE -
FIXED`, `FORCING + INDEPENDENT EVIDENCE`, and `AUTHENTIC CARTOGRAPHIC -
25.484X`. Primary columns are alias order; resolution columns are 8, 4, 2 km
order. The packet README states those orders and the view-specific row labels
exactly; supplementary is omitted from the list exactly when absent.

## Deterministic alias permutation and masking

Start with the admitted arms in enum order. Compute
`seed=fnv1a64(b"orogen-owner-review-order-v0")`, initialize
`rand_chacha 0.3 ChaCha8Rng::seed_from_u64(seed)`, then for `i=n-1..1` draw one
`RngCore::next_u64`, set `j=draw%(i+1)` and swap positions `i,j`. Position zero
is A, one B, two C. Modulo reduction and draw count are frozen. Result, arm and
evidence bytes are not inputs.

```rust
pub struct AliasBindingV0 {
    pub alias: ReviewAliasV0,
    pub arm: OrganizationArmV0,
    pub base_4km_result_hash: u64,
    pub base_4km_evidence_hash: u64,
}

pub struct PermutationCommitmentPreimageV0 {
    pub contract: String,
    pub literal_seed: String,
    pub bindings: Vec<AliasBindingV0>,
}
```

The authority stores bindings in alias order and the FNV of this preimage. The
reviewer packet stores only contract, literal-seed identifier, alias list and
commitment hash. This prevents accidental substitution but is trivially
brute-forceable over two/six permutations; it is explicitly not a secret hash.
The commitment domain is exactly
`orogen-owner-planar-review-v0/permutation-commitment`; hash its ASCII bytes then
the fixed-int little-endian bincode preimage.

Reviewer-visible files undergo an exact leak scan before publication. Reject
if any filename, sidecar string, prompt, JSON string or PNG ancillary chunk contains
the ASCII case-insensitive whole token `hold`, `carve`, `coupled`,
`coevolution`, `graph`, `reconstruction`, `native` or `forest`, or the adjacent
token phrase `arm h`, `arm c` or `arm g`. A token is a maximal ASCII
alphanumeric run, so `cartographic` is legal and standalone `graph` is not.
Also reject exact ASCII encodings of any arm-result/evidence hash or normalized
source attempt path as byte substrings. The neutral word `highland` is legal.
Only aliases appear in tiles and forms. G provenance is outside this scan
because it is never reviewer-visible before reveal.

## Capture authority and reviewer-packet schemas

```rust
pub struct OrganizationCaptureAuthorityV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub renderer_contract: String,
    pub input_bundle_hash: u64,
    pub comparison_authority: ReviewComparisonAuthorityV0,
    pub arm_bindings: Vec<CaptureArmBindingV0>,
    pub alias_bindings: Vec<AliasBindingV0>,
    pub permutation_commitment_hash: u64,
    pub renderer_source_revision: String,
    pub renderer_source_dirty: bool,
    pub renderer_executable_hash: u64,
    pub rustc_version: String,
    pub target_triple: String,
    pub os_libc_version: String,
    pub cpu_identifier: String,
    pub cargo_lock_hash: u64,
    pub thread_count: u32,
    pub views: Vec<PlanarViewSpecV0>,
    pub tiles: Vec<CaptureTileSpecV0>,
    pub primary_sheets: Vec<SheetSpecV0>,
    pub resolution_sheets: Vec<SheetSpecV0>,
    pub g_provenance_files: Vec<ReviewerFileBindingV0>,
    pub reviewer_id: String,
    pub primary_packet_hash: u64,
    pub resolution_packet_static_hash: u64,
    pub prompt_contract_hash: u64,
    pub derived_capture_authority_hash: u64,
}

pub struct ReviewerFileBindingV0 {
    pub filename: String,
    pub file_length: u64,
    pub file_hash: u64,
}

pub struct OrganizationReviewerPacketV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub protocol_version: String,
    pub phase: ReviewPhaseV0,
    pub reviewer_id: String,
    pub public_capture_id: u64,
    pub permutation_contract: String,
    pub permutation_literal_seed: String,
    pub permutation_commitment_hash: u64,
    pub aliases: Vec<ReviewAliasV0>,
    pub views: Vec<PlanarViewSpecV0>,
    pub presentation_disclosure: ReviewerPresentationDisclosureV0,
    pub sheets: Vec<ReviewerFileBindingV0>,
    pub form_template_hash: u64,
    pub prior_observation_hash: Option<u64>,
    pub derived_reviewer_packet_hash: u64,
}

pub struct ReviewerPresentationDisclosureV0 {
    pub physical_elevation_range_km: [f64; 2],
    pub physical_grade_range: [f64; 2],
    pub physical_exaggeration: f64,
    pub authentic_exaggeration: f64,
    pub light_vector: [f64; 3],
    pub fixed_camera_text: String,
    pub limitations: Vec<String>,
}
```

`public_capture_id` is FNV of renderer contract, input bundle hash, comparison-
authority kind without its semantic hash, admitted arm count and literal seed;
it is an opaque packet join, not an authentication token. Dirty source is
illegal for an active packet. Thread count is exactly one. G provenance files
are empty when G is not admitted and exactly three in view order when it is.

`renderer_source_revision` is the 40-lowercase-hex committed Git object;
`renderer_executable_hash` and `cargo_lock_hash` are file-byte FNVs.
`rustc_version` is `rustc -Vv` output with its one final LF removed. The V0
promotion build uses target `x86_64-unknown-linux-gnu`,
`-C target-cpu=x86-64-v2 -C target-feature=-fma`; `target_triple` and
`cpu_identifier` store those two registered strings. `os_libc_version` is the
first line of `ldd --version`, without LF. A different target/build scope needs
new renderer goldens and cannot claim equal pixels from this root.

The neutral limitations vector is exact:

```text
Cell-mean planar evidence; not the product globe renderer.
Physical 1x may be visually flat; numeric evidence remains authoritative.
Authentic uses declared 25.484x display rounding of the bound cartographic exaggeration and cannot establish physical success.
Aliases mask labels procedurally; do not derive or inspect the key before submission.
Judge only visible communication; do not infer undisclosed implementation depth.
```

`fixed_camera_text` is exactly `North-up orthographic plan; fixed shared
bounds; pixel-centre sampling; no perspective or result-dependent framing.`
`protocol_version` is the registered review-protocol string. The prompt
contract hash binds that string, exact prompts, response enum declarations,
ordering/invariants and the exact form-template constructor before any packet
is built.

The permutation literal is exactly `orogen-owner-review-order-v0` in both
`PermutationCommitmentPreimageV0.literal_seed` and packet
`permutation_literal_seed`; its FNV-derived `u64` initializes ChaCha and is never
stored in place of the literal.

Auxiliary hash preimages and domains are exact:

```rust
pub enum ReviewAuthorityKindV0 { Complete, PairwiseHC, PairwiseHG }

pub struct PublicCaptureIdPreimageV0 {
    pub renderer_contract: String,
    pub input_bundle_hash: u64,
    pub authority_kind: ReviewAuthorityKindV0,
    pub admitted_arm_count: u32,
    pub permutation_literal_seed: String,
}

pub struct PromptContractPreimageV0 {
    pub protocol_version: String,
    pub primary_prompts: Vec<String>,
    pub resolution_prompts: Vec<String>,
    pub response_variant_groups: Vec<Vec<String>>,
    pub invariant_strings: Vec<String>,
    pub primary_form_schema: String,
    pub resolution_form_schema: String,
    pub readme_constructor: String,
}

pub struct ReviewerPacketStaticPreimageV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub protocol_version: String,
    pub phase: ReviewPhaseV0,
    pub reviewer_id: String,
    pub public_capture_id: u64,
    pub permutation_contract: String,
    pub permutation_literal_seed: String,
    pub permutation_commitment_hash: u64,
    pub aliases: Vec<ReviewAliasV0>,
    pub views: Vec<PlanarViewSpecV0>,
    pub presentation_disclosure: ReviewerPresentationDisclosureV0,
    pub sheets: Vec<ReviewerFileBindingV0>,
    pub form_template_hash: u64,
}
```

Their domains are respectively
`orogen-owner-planar-review-v0/public-capture-id`,
`/prompt-contract` and `/resolution-packet-static` appended to the common
prefix. Hash exactly domain ASCII bytes then fixed-int little-endian bincode of
the preimage. `response_variant_groups` contains, in this order, the exact Rust
variant names for `ReviewClarityV0`, `ArtifactSeverityV0`,
`ReviewPreferenceV0`, `PriorExposureV0`, `ObservationProtocolStatusV0`,
`ResolutionStabilityV0` and `ArtifactResolutionTrendV0`. `invariant_strings`
is exactly:

```text
alias-major;views=FullDomain,Central,Transfer
preferences=nonempty-unique-admitted-enum-sorted-or-explicit-null-answer
notes=utf8-no-c0-except-lf-tab;maximum-bytes=4096
timestamps=canonical-utc-seconds;cross-root-nondecreasing
primary-row-channels=fixed+physical|independent|authentic
status-precedence=early-key-or-provenance;other-or-uncertain;declared-prior;valid
```

The form schema strings are `orogen-owner-primary-form-v0` and
`orogen-owner-resolution-form-v0`; the README constructor is
`orogen-owner-review-readme-v0`. The primary packet uses the complete packet
root preimage. `resolution_packet_static_hash` uses exactly the typed static
preimage above, not a field-name slice or concatenation convention.

The primary packet has no prior observation. The resolution packet binds the
validated primary-observation hash and cannot publish before it exists. Packet
files contain packet binary/JSON, exact form-template JSON, the phase's sheets,
neutral `README.txt` generated from the disclosure and a self-excluding run
envelope. They contain neither individual tiles nor authority hashes beyond the
permutation commitment.

`resolution_packet_static_hash` hashes the resolution packet fields from
`schema_version` through `form_template_hash`, omitting
`prior_observation_hash` and `derived_reviewer_packet_hash`. This avoids a
cycle: authority binds all resolution-visible bytes and static fields before
review, while the later packet adds the already-published primary-observation
hash. The primary packet is built first and its complete hash is stored in the
authority. `reviewer_id` is exactly `project-owner-primary-v0`, also appears in
both packets, and is frozen before the primary packet is opened.

### Exact form and README children

```rust
pub struct ReviewFormTemplateV0 {
    pub form_schema_version: String,
    pub protocol_version: String,
    pub phase: ReviewPhaseV0,
    pub reviewer_id: String,
    pub aliases: Vec<ReviewAliasV0>,
    pub views: Vec<PlanarReviewViewV0>,
    pub prompt_contract_hash: u64,
    pub prompt_keys: Vec<String>,
    pub prompts: Vec<String>,
    pub response_variant_groups: Vec<Vec<String>>,
    pub metadata_field_order: Vec<String>,
    pub metadata_prompts: Vec<String>,
    pub attestation_field_order: Vec<String>,
    pub attestation_prompts: Vec<String>,
}

pub struct PrimaryCompletedFormV0 {
    pub form_schema_version: String,
    pub reviewer_id: String,
    pub prior_exposure: PriorExposureV0,
    pub started_at_rfc3339: String,
    pub submitted_at_rfc3339: String,
    pub view_observations: Vec<PrimaryViewObservationV0>,
    pub alias_summaries: Vec<PrimaryAliasSummaryV0>,
    pub overall_preference: ReviewPreferenceV0,
    pub overall_notes: String,
    pub attestation: ReviewAttestationV0,
}

pub struct ResolutionCompletedFormV0 {
    pub form_schema_version: String,
    pub reviewer_id: String,
    pub started_at_rfc3339: String,
    pub submitted_at_rfc3339: String,
    pub view_observations: Vec<ResolutionViewObservationV0>,
    pub overall_resolution_preference: ReviewPreferenceV0,
    pub overall_notes: String,
    pub attestation: ReviewAttestationV0,
}
```

The template is canonical pretty JSON of `ReviewFormTemplateV0`, ending in one
LF. It contains no packet-hash placeholder, avoiding a packet/template cycle;
the submit command injects and validates the packet hash from its required
packet predecessor. `prompt_keys` are `primary-01` through `primary-08` then
`primary-alias-summary` and `primary-overall`, or `resolution-01` through
`resolution-03` then `resolution-overall`. `prompts` are the exact strings in
those sections in the same order. Metadata order is exactly the completed-form
editable metadata fields: primary uses `prior_exposure`,
`started_at_rfc3339`, `submitted_at_rfc3339`, `overall_notes`; resolution uses
the latter three. `metadata_prompts` is the parallel exact prior/start/submit/
notes string subset below. Auto-injected schema/reviewer fields are absent from
both vectors. Attestation order is the four
`ReviewAttestationV0` field names. Completed forms use serde's exact default
representation and declaration order, reject unknown/duplicate/missing fields,
and are converted to semantic observation roots; they are not retained.

`README.txt` is UTF-8 with LF only and exactly this constructor, substituting
the enum display tokens and joining limitations in their stored order:

```text
PLANAR REVIEW V0
REVIEWER: <reviewer_id>
PHASE: <PRIMARY 4 KM|RESOLUTION>
ALIASES: <comma-space alias list>
VIEWS: FULL, CENTRAL, TRANSFER
COLUMNS: <alias list|8 KM, 4 KM, 2 KM>
SHEETS:
- <sheet filename 0>
<one further '- <sheet filename>' line per bound sheet in stored order>
ROWS FULL: <comma-space row labels for Full>
ROWS CENTRAL: <comma-space row labels for Central>
ROWS TRANSFER: <comma-space row labels for Transfer>
LEGENDS:
- PHYSICAL ELEVATION - FIXED: viridis; km; range [-0.25,4.0]; values outside clamp; saturation witnesses in withheld authority
<for each view with supplement, in view order: '- PHYSICAL ELEVATION - SUPPLEMENTARY <FULL|CENTRAL|TRANSFER>: viridis; km; range <[-0.5,8.0]|[-1.0,16.0]|[-2.0,32.0]>; fixed primary retained'>
- PHYSICAL HILLSHADE - 1X: neutral land; E=1.0; light=northwest; no water flattening
- PHYSICAL GRADE - FIXED: white to black; range [0,0.10]; values above clamp
- FORCING + INDEPENDENT EVIDENCE: segment 0 orange fill; segment 1 purple fill; O0a orange/green lines; S0 primary/context red solid/dashed; D0 2,000 km2 blue reaches; peaks dark red circles; saddles purple triangles; portals dark blue squares
- AUTHENTIC CARTOGRAPHIC - 25.484X: hypsometric color; bound E displayed rounded to 25.484x; datum water flat blue; shoreline dark blue
CAMERA: <fixed_camera_text>
LIMITATIONS:
- <limitation 0>
- <limitation 1>
- <limitation 2>
- <limitation 3>
- <limitation 4>
INSTRUCTIONS:
- View sheets in listed view order.
- Use only the rows named by each prompt.
- Complete every field before submission.
- Do not inspect or derive the alias key before both observations are locked.
```

Angle-bracketed items are constructor slots, not literal output. Supplementary
lines are independently omitted/emitted by view in Full/Central/Transfer order
with that view's exact selected range token. There is exactly one final LF and
no blank line. The packet sheet bindings,
template file hash and regenerated disclosure bind every README substitution;
full validation regenerates all bytes.

Reviewer packet `run-envelope.json` deliberately does **not** inherit the
source-path/command fields of withheld artifact envelopes. Its exact neutral
projection is:

```rust
pub struct ReviewerFileBindingJsonV0 {
    pub filename: String,
    pub file_length: u64,
    pub file_hash: String,
}

pub struct ReviewerPacketEnvelopeV0 {
    pub envelope_schema_version: String,
    pub reviewer_packet_hash: String,
    pub public_capture_id: String,
    pub phase: ReviewPhaseV0,
    pub files: Vec<ReviewerFileBindingJsonV0>,
    pub publication_wall_milliseconds: u64,
}
```

The schema is `orogen-owner-reviewer-packet-envelope-v0`; `file_hash` is exactly
16 lowercase hex digits and all filename/string/cap rules are inherited from
the semantic binding. Files are every
other packet-subtree file in lexical relative-path order and exclude the
envelope. It contains no command, working directory, source path, source hash,
host or arm identity. This neutral envelope and every listed child pass the
same leak scan. Withheld capture/reveal publishers retain the full ordinary
envelope outside reviewer-visible packet subtrees.

## Human observation vocabulary and exact prompts

```rust
pub enum ReviewClarityV0 {
    NotAssessable,
    Absent,
    Weak,
    Mixed,
    Clear,
    Strong,
}

pub enum ArtifactSeverityV0 {
    NotAssessable,
    None,
    Minor,
    Noticeable,
    Severe,
    Dominant,
}

pub enum ReviewPreferenceV0 {
    Aliases(Vec<ReviewAliasV0>),
    NoVisibleDifference,
    NotAssessable,
}

pub enum PriorExposureV0 {
    None,
    ArchitectureDescriptionsOnly,
    EarlierUnmaskedOutputs,
    BothDescriptionsAndOutputs,
    Unsure,
}

pub enum ObservationProtocolStatusV0 {
    Valid,
    ValidWithDeclaredPriorExposure,
    EarlyRevealOrDerivedKey,
    OtherOrUncertainCompromise,
}

pub struct PrimaryViewObservationV0 {
    pub alias: ReviewAliasV0,
    pub view: PlanarReviewViewV0,
    pub differentiated_highlands: ReviewClarityV0,
    pub internal_crest_saddle_divide_structure: ReviewClarityV0,
    pub drainage_hierarchy: ReviewClarityV0,
    pub forcing_communication: ReviewClarityV0,
    pub physical_artifact_severity: ArtifactSeverityV0,
    pub authentic_artifact_severity: ArtifactSeverityV0,
    pub physical_diagnostic_usefulness: ReviewClarityV0,
    pub authentic_cartographic_usefulness: ReviewClarityV0,
    pub notes: String,
}

pub struct PrimaryAliasSummaryV0 {
    pub alias: ReviewAliasV0,
    pub overall_organization_usefulness: ReviewClarityV0,
    pub notes: String,
}

pub struct ReviewAttestationV0 {
    pub viewed_only_bound_packet_files: bool,
    pub did_not_open_or_derive_alias_key: bool,
    pub did_not_open_withheld_provenance: bool,
    pub completed_without_numeric_or_causal_result_summary: bool,
}

pub struct OrganizationPrimaryObservationV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub reviewer_id: String,
    pub reviewer_packet_hash: u64,
    pub public_capture_id: u64,
    pub prior_exposure: PriorExposureV0,
    pub started_at_rfc3339: String,
    pub submitted_at_rfc3339: String,
    pub view_observations: Vec<PrimaryViewObservationV0>,
    pub alias_summaries: Vec<PrimaryAliasSummaryV0>,
    pub overall_preference: ReviewPreferenceV0,
    pub overall_notes: String,
    pub attestation: ReviewAttestationV0,
    pub protocol_status: ObservationProtocolStatusV0,
    pub derived_primary_observation_hash: u64,
}
```

`ReviewClarityV0` anchors are exact: absent means no visible evidence of the
asked communication; weak means isolated/unclear; mixed means meaningful but
inconsistent; clear means readily legible with local limitations; strong means
dominant and coherent. These are ordered response labels, not values to average.
Artifact severity means none, minor local, noticeable but usable, severe enough
to mislead, or dominant enough to prevent judgment.

Primary form renders these exact prompts in field order for every alias then
Full/Central/Transfer:

1. `Using only fixed elevation and Physical 1x, how clearly are differentiated highlands visible rather than one universal smooth cap?`
2. `Using only fixed elevation, Physical 1x and physical grade, how clearly are internal crests, saddles and divides communicated?`
3. `Using only the forcing plus independent-evidence row, how clearly is drainage hierarchy communicated?`
4. `Using only the forcing plus independent-evidence row, how clearly does terrain organization communicate the linked forcing geometry?`
5. `In fixed elevation, Physical 1x and physical grade, how severe are faceting, terracing, seams, pillars or other physical-representation artifacts?`
6. `In Authentic, how severe are artifacts introduced or materially amplified by cartographic exaggeration?`
7. `How useful are fixed elevation, Physical 1x and physical grade for diagnosing the physical surface?`
8. `How useful is the declared Authentic 25.484x display layer as cartographic communication?`

Then ask per alias exactly `Overall, how clearly does this alias communicate
regional highland/drainage organization across the three views?` and:

```text
Which alias or tied aliases most clearly communicate regional highland/drainage organization?
```

Metadata and attestation labels are exact:

```text
Before opening this packet, what prior exposure did you have to architecture descriptions or unmasked outputs?
Review started at (UTC YYYY-MM-DDTHH:MM:SSZ):
Review submitted at (UTC YYYY-MM-DDTHH:MM:SSZ):
I viewed only files bound by this review packet.
I did not open or derive the alias key.
I did not open withheld post-reveal provenance files.
I completed this phase without a numeric or causal result summary.
Optional overall notes:
```

`NoVisibleDifference` and `NotAssessable` are first-class answers. The form
never says realistic, simulation, hack, physical owner, H, C or G. Notes are
optional valid UTF-8, reject NUL/other C0 controls except LF/TAB, and are capped
at 4096 bytes. Reviewer ID is printable ASCII 1--128 bytes and is registered
before packet access. RFC3339 times are observational provenance; require
canonical UTC `YYYY-MM-DDTHH:MM:SSZ` and nondecreasing order.

View records are alias-major then Full/Central/Transfer; summaries are alias
order. `ReviewPreferenceV0::Aliases` is nonempty, unique, admitted-only and
enum-sorted. The submitter derives `protocol_status` from
prior exposure and attestation; the completed form cannot select it directly.

## Resolution observation

```rust
pub enum ResolutionStabilityV0 {
    NotAssessable,
    Stable,
    MostlyStable,
    MateriallyReorganized,
}

pub enum ArtifactResolutionTrendV0 {
    NotAssessable,
    ConsistentlyImproves,
    MostlyImproves,
    Persists,
    MovesOrIntensifies,
}

pub struct ResolutionViewObservationV0 {
    pub alias: ReviewAliasV0,
    pub view: PlanarReviewViewV0,
    pub highland_drainage_stability: ResolutionStabilityV0,
    pub artifact_trend: ArtifactResolutionTrendV0,
    pub cartographic_communication_stability: ResolutionStabilityV0,
    pub notes: String,
}

pub struct OrganizationResolutionObservationV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub reviewer_id: String,
    pub reviewer_packet_hash: u64,
    pub primary_observation_hash: u64,
    pub public_capture_id: u64,
    pub started_at_rfc3339: String,
    pub submitted_at_rfc3339: String,
    pub view_observations: Vec<ResolutionViewObservationV0>,
    pub overall_resolution_preference: ReviewPreferenceV0,
    pub overall_notes: String,
    pub attestation: ReviewAttestationV0,
    pub protocol_status: ObservationProtocolStatusV0,
    pub derived_resolution_observation_hash: u64,
}
```

For every alias/view in order, prompts are:

1. `Does the same major highland and drainage organization persist across 8, 4 and 2 km?`
2. `Do faceting, terracing and seam artifacts improve consistently with resolution rather than move or intensify?`
3. `Does Authentic preserve the organization visible in fixed elevation, Physical 1x and the independent-evidence row without introducing a misleading resolution-dependent artifact?`

`Stable` means the same major organization with only local discretization changes;
`MostlyStable` permits visible local membership/shape changes;
`MateriallyReorganized` means a different major object layout. Artifact trend
labels have their ordinary literal meaning; `Persists` means no material
improvement, while `MovesOrIntensifies` includes a changed artifact grammar.
The overall prompt is exactly `Which alias or tied aliases best preserve the
same clearly communicated regional organization across 8, 4 and 2 km?` Primary answers are not shown alongside the resolution form
and cannot be edited.

Resolution view records use the same alias-major/view order and preference
rules. Its submitter derives `protocol_status` from the bound primary status and
the new attestation; the form cannot downgrade or select it directly.

V0 has exactly one reviewer: `project-owner-primary-v0`. A second reviewer,
consensus, majority vote or inter-rater average requires a new schema rather
than overloading this authored observation.

## Reveal and post-reveal interpretation

```rust
pub enum ReviewProtocolDispositionV0 {
    CompletedMaskedSequence,
    CompletedWithPriorExposure,
    ProtocolCompromised,
}

pub struct RevealedObservationBindingV0 {
    pub reviewer_id: String,
    pub primary_observation_hash: u64,
    pub resolution_observation_hash: u64,
}

pub struct OrganizationReviewRevealV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub capture_authority_hash: u64,
    pub primary_packet_hash: u64,
    pub resolution_packet_hash: u64,
    pub permutation_commitment_hash: u64,
    pub alias_bindings: Vec<AliasBindingV0>,
    pub observation: RevealedObservationBindingV0,
    pub g_provenance_files: Vec<ReviewerFileBindingV0>,
    pub g_provenance_legend_lines: Vec<String>,
    pub g_provenance_readme: Option<ReviewerFileBindingV0>,
    pub protocol_disposition: ReviewProtocolDispositionV0,
    pub revealed_at_rfc3339: String,
    pub derived_review_reveal_hash: u64,
}

pub enum ArchitectureRecommendationV0 {
    Arms(Vec<OrganizationArmV0>),
    NoRecommendation,
    InsufficientEvidence,
}

pub struct PostRevealInterpretationInputV0 {
    pub reviewer_id: String,
    pub recommendation: ArchitectureRecommendationV0,
    pub cited_evidence_hashes: Vec<u64>,
    pub interpretation: String,
    pub submitted_at_rfc3339: String,
    pub acknowledges_base_case_only: bool,
    pub acknowledges_no_product_promotion: bool,
}

pub struct OrganizationPostRevealInterpretationV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub reviewer_id: String,
    pub reveal_hash: u64,
    pub recommendation: ArchitectureRecommendationV0,
    pub cited_evidence_hashes: Vec<u64>,
    pub interpretation: String,
    pub submitted_at_rfc3339: String,
    pub acknowledges_base_case_only: bool,
    pub acknowledges_no_product_promotion: bool,
    pub derived_interpretation_hash: u64,
}
```

The completed-interpretation JSON is the recursive JSON mirror of
`PostRevealInterpretationInputV0`: `cited_evidence_hashes` contains 16-digit
lowercase hex strings, never JSON numbers, and converts to semantic `u64` only
after exact syntax, uniqueness, order and source-membership validation.

Reveal recomputes the permutation commitment and exact alias bindings, validates
the reviewer sequence and copies G provenance PNGs into the reveal directory.
It mechanically maps masked records to arms without altering answer values or
text. Any reviewer with early reveal/derived key, failed
attestation or uncertain/other exposure makes the protocol disposition compromised; the
record remains evidence about process failure and is never silently discarded.

When G is admitted, legend lines are exactly `G AUTHORED FOREST: dashed red;
support at least 2,000 km2; authored provenance, not common evidence` and
`INDEPENDENT D0: solid blue; common 2,000 km2 reference`, in that order.
`g-provenance-README.txt` is those lines joined by LF plus one final LF and its
binding is `Some`; without G both fields are empty/`None` and the file is absent.

Interpretation happens only after reveal and after the reviewer may inspect
numeric, correspondence, cost and G provenance evidence. Recommendation vectors
are nonempty, unique, enum-sorted subsets of arms in the reveal alias bindings.
`cited_evidence_hashes` is unique ascending and each value must occur in the
fully validated capture/comparison source graph, its cost file, either
observation root or revealed G-provenance file set; foreign hashes reject.
Interpretation is capped at 4096 bytes UTF-8. This
record may recommend the next architecture/audit conclusion but cannot promote
an arm: wet/dry and reorganization/R0 controls remain mandatory later work.

## Locked review state machine

The only legal state sequence is:

```text
FrozenComparison
  -> CaptureBundlePublished
  -> PrimaryObservationPublished
  -> ResolutionPacketPublished
  -> ResolutionObservationPublished
  -> RevealPublished
  -> [PostRevealInterpretationPublished]
```

No transition may be skipped or reversed. The primary packet is constructed
privately first so its hash can enter authority; authority is then constructed,
and the validated authority plus packet subtrees become visible in one atomic
capture-bundle rename. The reviewer records
`started_at_rfc3339` immediately before first opening it. The submitted primary
root is published atomically before the resolution packet is materialized or
opened. The resolution packet binds that primary root and all of its other
fields must reproduce the authority's static hash. Its observation is likewise
published before the alias binding or G-provenance directory is opened.

Require
`primary.started<=primary.submitted<=resolution.started<=resolution.submitted<=reveal.revealed`
and, when present, `reveal.revealed<=interpretation.submitted`, all at canonical
UTC-second precision. Equality is legal. Publication/opening order itself is a
reviewer attestation plus immutable path sequence, not claimed to be
machine-provable from wall-clock timestamps alone.

Opening a later phase early does not authorize deletion or restart. The
reviewer completes the remaining records where possible, declares the exposure
in `protocol_status`, and reveal publishes `ProtocolCompromised`. A clerical
error discovered before a submitted root exists may abandon the whole capture
attempt; it cannot replace a published packet or observation. A correction
after publication requires a new versioned protocol and new capture attempt,
while the old roots remain.

The reveal validator requires exact hash joins, reviewer identity, timestamps,
alias/view/field orders and attestation/status consistency. `Valid` requires
`PriorExposureV0::None` and every attestation boolean true.
`ValidWithDeclaredPriorExposure` requires one of the three non-`None`, non-
`Unsure` prior-exposure variants and every attestation boolean true. `Unsure`,
any false attestation or early key/provenance access requires a compromised
status. Derive one status by this descending precedence:
`EarlyRevealOrDerivedKey` when early key/provenance access occurred or either corresponding attestation is false;
`OtherOrUncertainCompromise` for `Unsure`, an unbound-file or numeric/causal
summary exposure, or any remaining false attestation;
`ValidWithDeclaredPriorExposure`; `Valid`. Multiple incidents choose the first
applicable row. Resolution combines newly declared incidents with the bound
primary record under the same precedence and cannot become less severe.

A wrong/modified packet rejects before observation publication and cannot be
laundered into a review status. Reviewer withdrawal leaves no complete
observation root, blocks subsequent phases and is recorded in the dated audit;
V0 does not fabricate all-`NotAssessable` answers for it.

## Hash preimages and deterministic JSON

Every root hash is FNV-1a over the registered domain string followed by the
bincode bytes of the same struct with its final `derived_*_hash` field omitted.
Child file hashes are FNV over file bytes. Root domains are exactly
`orogen-owner-planar-review-v0/capture-authority`, `/reviewer-packet`,
`/primary-observation`, `/resolution-observation`, `/review-reveal`,
`/post-reveal-interpretation` and `/capture-failure`, where each slash-prefixed
suffix is appended to the common prefix shown by the first string.
The permutation and static-packet hashes use their explicitly defined
preimages. Tile raw hashes bind tightly packed RGBA; PNG hashes bind encoded
bytes. Sheet raw hashes bind the recomposed RGBA, independently of PNG.

Binary is semantic authority. JSON inherits the evidence amendment's exact
projection rules: `from_semantic` is the sole constructor; declaration order is
field order; serde's default external enum tagging is used; no map, rename,
flatten, skip or omitted `None` is legal; pretty output ends in one LF. Each
root projection adds `json_schema_version` as its first field, preserves every
semantic field thereafter, renders `public_capture_id` and every integer field
whose name ends in `_hash` or `_hashes` as 16-digit lowercase hexadecimal
strings, and recursively applies that transform to nested structs, variants,
vectors, arrays and options. Other integers remain JSON integers and finite
floats remain round-trip JSON numbers. Exact JSON schema versions are
`orogen-owner-planar-capture-authority-json-v0`,
`orogen-owner-reviewer-packet-json-v0`,
`orogen-owner-primary-observation-json-v0`,
`orogen-owner-resolution-observation-json-v0`,
`orogen-owner-review-reveal-json-v0`,
`orogen-owner-post-reveal-interpretation-json-v0` and
`orogen-owner-capture-failure-json-v0`. Parsed JSON must equal a fresh
projection and byte regeneration must be exact.

`form-template.json` is a deterministic schema/prompt descriptor, not a
fillable response object and not a semantic result. It is generated from the
phase, aliases, views, exact prompts, enum variant spellings and reviewer ID; it
contains no packet-hash placeholder, free narrative, hidden field or JavaScript.
The submit command accepts a separate JSON value of the exact applicable
`*CompletedFormV0` type, rejects unknown or duplicate fields, converts it to the
semantic observation, validates every
choice/order/string rule and publishes both semantic binary and its canonical
JSON projection. The completed form itself is not retained, avoiding a second
authority for the same answers.

## Capture failure boundary

```rust
pub enum CaptureFailurePhaseV0 {
    PredecessorValidation,
    Rasterization,
    SheetAssembly,
    PacketAssembly,
    ObservationValidation,
    RevealAssembly,
    PostRevealInterpretation,
    Publication,
}

pub enum CaptureInvocationV0 {
    Capture,
    SubmitPrimary,
    ReleaseResolution,
    SubmitResolution,
    Reveal,
    Interpret,
}

pub enum CaptureFailureAuthorityV0 {
    ReplayableInstrument,
    ObservationalResource,
}

pub enum CaptureIndexFamilyV0 {
    Predecessor,
    Polygon,
    Pixel,
    Tile,
    Sheet,
    PacketFile,
    Prompt,
    Observation,
    AliasBinding,
    Citation,
}

pub enum CaptureFailureCauseV0 {
    InvalidPredecessor,
    InvalidGeometry,
    MissingRequiredField,
    NonCanonicalValue,
    HashMismatch,
    RasterMismatch,
    LeakDetected,
    InvalidReviewSequence,
    ResourceCeiling,
    InvariantFailure,
}

pub enum CaptureFailureWitnessV0 {
    None,
    Index { family: CaptureIndexFamilyV0, index: u64 },
    Hash { stored_hash: u64, recomputed_hash: u64 },
    File { relative_path: String },
    Resource { kind: OrganizationResourceKindV0, observed: u64, ceiling: u64 },
}

pub struct OrganizationCaptureFailureV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub invocation: CaptureInvocationV0,
    pub public_capture_id: Option<u64>,
    pub predecessor_hashes: Vec<u64>,
    pub authority: CaptureFailureAuthorityV0,
    pub phase: CaptureFailurePhaseV0,
    pub cause: CaptureFailureCauseV0,
    pub witness: CaptureFailureWitnessV0,
    pub derived_capture_failure_hash: u64,
}
```

Predecessor hashes are exact and ordered: `Capture` uses input, comparison,
then every semantic hash in the authority-dependent `--sources` traversal;
`SubmitPrimary` uses capture authority then primary packet;
`ReleaseResolution` appends primary observation; `SubmitResolution` appends
resolution packet; `Reveal` appends resolution observation; `Interpret`
appends reveal then cited hashes in unique ascending order. No other arity is
legal.

The legal ordinary matrix is:

| phase | causes | witness kinds |
|---|---|---|
| predecessor | invalid predecessor, noncanonical, hash mismatch, invariant | index, hash, file |
| raster | invalid geometry, missing field, noncanonical, raster mismatch, invariant | index, hash |
| sheet | raster mismatch, hash mismatch, invariant | index, hash, file |
| packet | leak, hash mismatch, noncanonical, invariant | index, hash, file |
| observation | invariant | index, file, none |
| reveal | invalid sequence, hash mismatch, invariant | index, hash, file |
| interpretation | invalid sequence, hash mismatch, noncanonical, invariant | index, hash, file |
| publication | invariant | file |

`ResourceCeiling` is legal at the active phase only with `Resource` and
`ObservationalResource`; every ordinary row is `ReplayableInstrument` and may
not use `Resource`. `None` is legal only for `InvariantFailure` when no narrower
witness exists. Raw-cap, decode, structure/order, scalar canonicality, local
hash, predecessor identity, deterministic rebuild, leak scan and publication
checks run in that precedence. For simultaneous failures choose phase order,
cause declaration order, then witness kind `Index,Hash,File,Resource,None`;
within it choose enum/index, `(stored_hash,recomputed_hash)`, lexical normalized
path or `(kind,observed,ceiling)` tuple order. Thus selection is total.

An invalid source, malformed completed form/interpretation or illegal
transition is always an external diagnostic and does not consume a canonical
successor target. It cannot manufacture a semantic failure under an unrelated
capture ID. Deterministic renderer, validator and leak-scan failures after
valid predecessors publish replayable failure roots. A measured resource stop
is observational and never implies numerical or arm invalidity. Protocol
compromise is represented in observations/reveal, not misclassified as a
renderer failure. Cooperative error publication contains no partial trusted
tiles or sheets.

## Canonical directories and atomic publication

The initial bundle alone uses a unique nonsemantic `<attempt-id>`. Every later
path key is the 16-digit lowercase semantic predecessor hash shown, which makes
each phase a single-successor claim. Exact successful file sets are:

```text
artifacts/orogen-owner-v0/review/capture/<attempt-id>/
  authority/capture-authority.bin
  authority/capture-authority.json
  authority/tiles/<canonical tile filenames>
  authority/sheets/<canonical primary and resolution sheet filenames>
  [authority/g-provenance/<canonical G provenance filenames>]
  primary-packet/reviewer-packet.bin
  primary-packet/reviewer-packet.json
  primary-packet/form-template.json
  primary-packet/README.txt
  primary-packet/sheets/<three primary sheet filenames>
  primary-packet/run-envelope.json
  run-envelope.json

artifacts/orogen-owner-v0/review/observation/primary/<capture-authority-hash>/
  primary-observation.bin
  primary-observation.json
  run-envelope.json

artifacts/orogen-owner-v0/review/packet/resolution/<primary-observation-hash>/
  reviewer-packet.bin
  reviewer-packet.json
  form-template.json
  README.txt
  sheets/<six or nine resolution sheet filenames>
  run-envelope.json

artifacts/orogen-owner-v0/review/observation/resolution/<resolution-packet-hash>/
  resolution-observation.bin
  resolution-observation.json
  run-envelope.json

artifacts/orogen-owner-v0/review/reveal/<resolution-observation-hash>/
  review-reveal.bin
  review-reveal.json
  g-provenance/<zero or three canonical provenance filenames>
  [g-provenance-README.txt]
  run-envelope.json

artifacts/orogen-owner-v0/review/interpretation/<review-reveal-hash>/
  post-reveal-interpretation.bin
  post-reveal-interpretation.json
  run-envelope.json
```

A pairwise resolution packet has six sheets; a complete packet has nine. The
capture bundle always contains three primary sheets and six/nine authority
resolution sheets. The authority and primary packet therefore become visible
in one parent-directory rename; there is no exposed unbound packet or second
rename. A semantic capture failure bundle instead contains exactly
`capture-failure.bin`, `capture-failure.json` and `run-envelope.json`.

Publication inherits the artifact amendment's absent target, owned sibling
lock, `create_new`, bounded decode, reread/full validation, `sync_all`, lexical
self-excluding envelope manifest, atomic rename and explicit stale-lock rules.
Targets never overwrite. A command derives and requires its exact canonical
target; an alternate path or existing same-predecessor target rejects, so
immutable roots cannot fork silently. Temp trees are private to the publisher. The capture
publisher validates every source root, rebuilds every raw tile twice in fresh
buffers, requires bit-identical results, decodes every PNG, recomposes every
sheet, runs the leak scan, atomically publishes and rereads the complete bundle,
then exposes the primary packet to the reviewer. Blind intermediate roots repeat
standalone canonical/chain validation; reveal repeats the full source and pixel
rebuild before unmasking rather than trusting paths or outer hashes.

The capture attempt, including primary-packet assembly, has a 30-minute, 2-GiB
wall/RSS ceiling, 4-GiB authority-subtree cap, 2-GiB packet-subtree cap and
6-GiB bundle cap. Resolution-packet, observation, reveal and interpretation
publishers each have a 5-minute, 1-GiB wall/RSS ceiling. Resolution packets
retain up to 2 GiB, observations/interpretations 64 MiB, and reveal 128 MiB, in
addition to registered byte caps. Human
viewing time is not timed as compute evidence. Release compilation is excluded;
full source validation, rasterization, encoding and publication are included.
External measurement remains final resource authority.

## Thin CLI and validation boundaries

After all non-active review infrastructure is implemented, build the promoted
binary once with the exact command below. Operational commands invoke those
bytes directly rather than allowing `cargo run` to rebuild them:

```text
RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=-fma" cargo build --release --bin orogen_owner_review
target/release/orogen_owner_review capture ...
target/release/orogen_owner_review submit-primary ...
target/release/orogen_owner_review release-resolution ...
target/release/orogen_owner_review submit-resolution ...
target/release/orogen_owner_review reveal ...
target/release/orogen_owner_review interpret ...
target/release/orogen_owner_review validate ...
```

Exact subcommand flags are:

```text
capture --input DIR --comparison DIR --sources JSON --output DIR
submit-primary --capture-bundle DIR --completed-form JSON --output DIR
release-resolution --capture-bundle DIR --primary-observation DIR --output DIR
submit-resolution --capture-bundle DIR --primary-observation DIR --resolution-packet DIR --completed-form JSON --output DIR
reveal --capture-bundle DIR --primary-observation DIR --resolution-packet DIR --resolution-observation DIR --sources JSON --output DIR
interpret --capture-bundle DIR --primary-observation DIR --resolution-packet DIR --resolution-observation DIR --reveal DIR --sources JSON --completed-interpretation JSON --output DIR
validate --kind capture|primary-observation|resolution-packet|resolution-observation|reveal|interpretation|capture-failure --artifact DIR --sources JSON --chain JSON
```

The nonsemantic path manifest is exact JSON in declaration order:

```rust
pub struct ReviewEvidencePathSlotV0 {
    pub identity: OrganizationArtifactIdentityV0,
    pub evidence_dir: String,
    pub arm_run_dir: String,
    pub run_predecessor_dir: Option<String>,
}

pub struct ReviewChainPathManifestV0 {
    pub capture_bundle_dir: String,
    pub primary_observation_dir: Option<String>,
    pub resolution_packet_dir: Option<String>,
    pub resolution_observation_dir: Option<String>,
    pub reveal_dir: Option<String>,
}

pub struct ReviewSourceManifestV0 {
    pub manifest_schema_version: String,
    pub input_dir: String,
    pub comparison_dir: String,
    pub evidence_slots: Vec<ReviewEvidencePathSlotV0>,
    pub cross_arm_surface_dirs: Vec<String>,
    pub numerical_discrepancy_dirs: Vec<String>,
    pub correspondence_dirs: Vec<String>,
    pub pairwise_comparison_dirs: Vec<String>,
    pub excluded_arm_failure_dir: Option<String>,
}
```

The schema is `orogen-owner-review-source-paths-v0`; paths are canonical UTF-8,
contain no NUL, are at most 4096 bytes, and are excluded from semantic hashes and reviewer-visible
envelopes. Complete has 11/3/2/22/3 slots and no excluded failure. Pairwise has
the evidence parent's exact H/C=8 or H/G=7 evidence slots, only that root's
displayed dependency subsets, and exactly one excluded semantic failure path.
Explicit command flags must equal manifest paths.
Raw manifest bytes are capped at 8 MiB before decode; every vector is capped by
the complete cardinalities just listed. Capture `--input`/`--comparison` must
equal the two manifest paths; reveal/interpret recover them from the manifest.

`--sources` is the evidence amendment's exact full-validation path manifest:
for complete review, its ordered 11 evidence/run/aligned-run-predecessor slots,
three cross-arm surface, two numerical-discrepancy, 22 correspondence and three
pairwise-comparison directories; for H/C or H/G, the exact eight or seven
evidence/run/aligned-predecessor slots and that pairwise root's displayed
dependencies, plus the excluded arm failure. Slots retain duplicates and use
the parent order. Commands reject the wrong authority-dependent arity, path
identity or an absent/extra key. `--output` must equal the canonical path
derived in the directory section (except the initial unique capture attempt)
and must not exist. Capture accepts no camera, palette, range, layer, alias,
seed, threshold or renderer flag. Capture and reveal require non-null sources
and full rebuild. Blind intermediate publishers accept no source paths and use
standalone canonical/chain validation against the immutable capture bundle;
their `validate` invocation supplies `--sources null` and a non-null typed
`ReviewChainPathManifestV0`. Capture uses `--chain null`; later kinds require
the exact predecessor prefix and reject later options without earlier ones.
Interpretation requires both non-null sources and chain for citation validation.
Capture-failure validation follows its invocation authority for both flags.
There is no flag that downgrades a required full
validation to standalone.

Standalone validation establishes canonical encoding, local invariants and
child-file integrity. Full validation additionally rebuilds every predecessor,
pixel, sheet, packet and join. Deterministic replay establishes declared
production by this renderer. None establishes scientific realism, product
render fidelity, global generality or causal sufficiency of an arm.

## Manufactured and regression gate

Before active-arm capture, tests must establish:

1. **Views and ownership:** exact FullDomain aspect padding, Central/Transfer
   pixel centres, north-up orientation, shared-edge/vertex top-left ownership,
   background, clipped cells and invalid polygon rejection.
2. **Raster arithmetic:** sRGB decode/encode anchors, rounding ties, hillshade
   normals at zero/affine grades, Bresenham/wide/dashed primitives and
   raw-RGBA repeat with FMA disabled under the registered executable scope.
3. **Physical layers:** range equality versus saturation, count/area witnesses,
   fixed grade, water only in Authentic, exact shoreline and no modeled-water
   implication.
4. **Supplement ladder:** no supplement, every threshold boundary, cohort-wide
   shared choice, exhausted last rung and identical rows across phases.
5. **Common overlay:** exact forcing normalization/compositing, O0a censoring,
   S0 primary/context/peak/saddle styles, D0 2,000-km2 reach/portal inclusion,
   clipping and layer order; O0b/native-state mutation cannot change pixels.
6. **G provenance:** only validated G-4 native forest can affect the separate
   three post-reveal images; support/portal boundaries and zero-G omission are
   exact; H/C have no native overlay.
7. **Sheets and PNG:** exact dimensions/copy offsets/sidecar labels,
   raw-to-PNG decode equality, byte repeat, no forbidden ancillary chunk
   and exact sheet recomposition.
8. **Masking and leaks:** exact two- and three-arm Fisher--Yates witnesses,
   commitment repeat, every forbidden token/hash/path in filenames, text, JSON
   and crafted PNG chunks rejects; masked packets contain no provenance tiles.
9. **Review sequence:** all legal responses, ties and not-assessable cases;
   string/time/order/cap checks; every illegal state
   transition and attestation/status combination; submitted roots are
   immutable.
10. **Reveal:** exact alias mapping without answer mutation, prior-exposure and
    compromise dispositions, G copy hashes, wrong-chain rejection and optional
    interpretation acknowledgements.
11. **Mutation and bounds:** repaired outer hashes after mutation of every
    predecessor, tile, sheet, packet, observation and reveal field reject;
    oversized/truncated/trailing/NaN/infinite/negative-zero/length inputs reject
    before allocation.
12. **Publication and CLI:** exact success/failure file sets, absent-target
    sentinel, ordinary-error cleanup, lock/temp handling, lexical envelopes,
    binary/JSON/PNG repeat and rejection of missing, extra or tuning flags.
13. **Known answers:** committed bincode bytes, FNV values and JSON bytes for all
    seven semantic roots plus permutation, public-ID, prompt and static-packet
    auxiliary hashes; exact two/three-arm template, README, neutral-envelope and
    optional-supplement bytes.
14. **Forms and time:** two/three-arm descriptor-to-completed-form-to-observation
    round trips; exact unknown/duplicate/missing fields; cross-root timestamp
    equality, one-second forward and every reversal.
15. **Transactions:** fault injection before and after the one capture-bundle
    rename, second-successor rejection for every later phase, canonical target
    mismatch and no visible packet from an uncommitted temp tree.
16. **Failure and caps:** every legal failure-matrix row and every illegal
    phase/cause/witness/authority combination; each raw/file/vector/directory
    cap exactly equal and one byte/item over, including three provenance PNGs.
17. **Interpretation:** foreign/non-admitted recommendation arms, empty/
    duplicate/unsorted ties, unknown/duplicate/unsorted citations and false
    acknowledgements reject.

Existing accepted shared-input, G0/S0, D0, O0a, O0b, arm artifact, numerical
and evidence gates remain prerequisites. These capture tests compose them and
do not change their answers.

Before any active H/C/G result is generated or viewed, implement all shared
non-active schema/artifact/evidence/review infrastructure, commit its source,
perform the exact promoted build, and publish a small dated renderer-promotion
audit with the executable hash plus manufactured raw-RGBA/PNG/sheet golden hashes. That
committed reference source resolves any residual library-level clipping or
encoding detail while outputs remain unavailable, so it cannot tune to terrain.
The active capture authority must name exactly that promoted revision and
executable scope; changing it requires a new amendment and new goldens before a
full campaign rerun.

## Campaign execution and stop boundary

After implementation passes all gates, execute the parent base campaign in its
registered order without inspecting presentation. Freeze and exact-repeat arm
and evidence roots first. Then:

1. build a complete H/C/G capture only from an accepted complete comparison;
   otherwise build the legal H/C or H/G capture only from its accepted pairwise
   comparison and the other arm's bound base nonadmission; every admitted arm
   has valid 8/4/2 base roots and a missing/failed complete reducer alone never
   permits fallback;
2. publish and lock capture authority plus the unopened primary packet;
3. complete and publish the primary observation;
4. release, complete and publish the resolution observation;
5. reveal aliases and separate G provenance, then inspect numeric,
   correspondence and cost evidence;
6. optionally publish the authored interpretation; and
7. write a dated audit and stop before response cases, tuning or product change.

This amendment completes item 4 of the parent executable stop boundary. Once it
is committed, implementation may add the shared semantic/artifact/evidence and
review infrastructure plus the three registered base arms. Likely narrow
helpers are:

```text
render_planar_capture_tile_v0
assemble_planar_review_sheet_v0
build_organization_capture_authority_v0
validate_organization_review_chain_v0
build_g_provenance_sheet_v0
```

That authorization does not choose an arm or freeze the project architecture.
Systems remain open to criticism and fundamental rework. It authorizes one
bounded discriminator designed to reveal whether current H, reduced causal C or
authentic graph-first G owns useful organization at justified cost. Wet/dry,
motion-reorganization/R0, global/product and downstream ecology/civilization
questions remain explicitly outside this base capture and must not be inferred
from it.
