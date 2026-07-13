# Metric registry

Status: schema established; implemented-metric inventory in progress.

The first code/document synthesis is recorded in
[the numerical instrumentation inventory](inventory/numerical-instrumentation.md).
It identifies current definitions and risks before individual stable IDs are
assigned.

This registry defines what Hex3 measurements mean and whether they may influence
decisions. It is not a target sheet. Initial entries will catalogue current code
and historical gates before any consolidation or new corpus harness is built.

## Entry schema

| Field | Required meaning |
|---|---|
| ID | Stable machine-friendly identifier |
| Name | Human-readable name |
| Status | current, provisional, historical, superseded or invalid |
| Class | invariant, field descriptor, feature, relationship, product indicator or gate |
| Claim | What the metric can legitimately tell us |
| Operational definition | Exact formula, mask/path/object construction and thresholds |
| Owner/stage | Source state and computed/viewed stage |
| Domain | Tectonics, terrain, climate, hydrology, erosion, semantics, presentation or performance |
| Units | Physical unit, normalized coordinate, ratio or count |
| Weighting | Area, volume, cell, length, object or other weighting |
| Aggregation | Per-cell, object, seed, ensemble and reported summaries |
| Resolution relation | Expected invariant, convergent, scale-dependent or unknown behavior |
| Control response | Perturbation that should move or preserve it |
| Confounders | Known ways it can mislead |
| Reference | Empirical/design comparison and its applicability |
| Decision role | Descriptive, warning, experiment evidence or promotion gate |
| Threshold rationale | None, heuristic, empirical range, numerical tolerance or policy |
| Goodhart risk | What direct optimization could damage or hide |
| Implementation | Current code/report location |
| Evidence | Audits/tests that establish or challenge it |

## Registration rules

- IDs describe meaning, not the current function name.
- Changed definitions receive a new versioned ID unless demonstrably equivalent.
- Cell-count and area-weighted variants are distinct metrics.
- Native elevation-per-radian slope and physical grade are distinct metrics.
- A feature metric must link to its object/mask definition.
- An absent threshold rationale is recorded as unknown, not reconstructed from
  a plausible story.
- Superseded and invalid metrics remain discoverable with their failure reason.
- Presentation metrics include profile and camera conditions in their identity.

## Initial inventory groups

The first population pass will cover:

1. topology, conservation, determinism and unit invariants;
2. terrain distributions, peaks, slope, relief, mountains and plateau masks;
3. tectonic work, range attribution and carrier-resolution behavior;
4. climate fields, aridity and spatial structure;
5. drainage, lakes, river topology, grading and erosion response;
6. coarse-to-fine and pre/post stage survival;
7. ecological proxy classifications and uncertainty;
8. generation/runtime/cache cost;
9. existing pass/warn/fail gates and retired diagnostic artifacts.

## Seed entries: high-risk and decision-relevant metrics

These entries establish identity and status; the next pass should move them to a
machine-readable registry and complete control/reference fields.

| ID | Status/class | Operational definition | Units / weighting | Decision role and principal confounder |
|---|---|---|---|---|
| `terrain.elevation.max_km.v1` | current field descriptor | maximum retained elevation converted with 10 km/unit | km; single cell extreme | anomaly description only; highly resolution/outlier sensitive |
| `terrain.land_area_fraction.v1` | current field descriptor | dry, non-submerged area divided by total sphere area on the retained surface | fraction; area-weighted | descriptive; datum and lake state affect it |
| `terrain.land_elevation_area_p50_km.v1` | current field descriptor | weighted median elevation over dry land | km; land-area weighted | stable distribution center; excludes submerged lake cells |
| `terrain.land_elevation_area_p90_km.v1` | current field descriptor | weighted 90th percentile elevation over dry land | km; land-area weighted | upper terrain envelope, not mountain morphology |
| `terrain.land_elevation_area_p99_km.v1` | current field descriptor | weighted 99th percentile elevation over dry land | km; land-area weighted | tail descriptor less fragile than maximum but still resolution-sensitive |
| `terrain.peak_guard_km.v1` | historical gate | warn above 12 km, fail above 14 km | km; worst cell per world | user-calibrated guard, not empirical Earth target; direct tuning can flatten legitimate extremes |
| `terrain.mountain_mask_1p5km.v1` | provisional feature mask | retained land elevation at least 1.5 km | area or cell fraction must be named | common range substrate; conflates elevated interiors and orogens |
| `terrain.significant_range_20kkm2.v1` | provisional feature component | connected 1.5-km mask component at least 20,000 km² | objects; km²/km | descriptive range sampling; threshold rationale remains heuristic |
| `terrain.summit_cap_500m.v1` | historical feature measurement | area in each significant range within 0.5 km of its own summit | km²; area-weighted object quantiles | broad-summit proxy; moves when one cell changes summit elevation |
| `terrain.flat_summit_cap_grade1pct.v1` | historical feature measurement | summit-cap cells whose steepest downhill neighboring edge has physical grade below 1% | percent of cap area | plateau proxy; cell-neighbor scale and cap definition affect result |
| `terrain.high_decile_downslope.v1` | current provisional descriptor | cells in top land-elevation decile and mountain mask, summarized by maximum downhill neighbor slope | native/physical slope variant must be named; sampled cells | roughness/plateau monitor, not a plateau object; adaptive sampling bias |
| `terrain.local_relief_radius_km.v1` | current field descriptor | max-minus-min elevation within declared physical radius; robust p95-p05 variant is separate | m or km; cell or area quantile named | scale-specific relief; low values can mean smooth terrain or missing resolved structure |
| `terrain.land_max_neighbor_grade_area_p50/p90/p99.v1` | current field descriptor | maximum absolute physical grade from each dry-land cell to any neighbor using stable chord-to-arc distance, then area-weighted quantile | physical grade; land-area weighted | one-edge scale shrinks/changes with resolution and is a roughness/allocation diagnostic, not a scale-independent slope distribution; dot/acos distance is forbidden because it collapses on near-coincident fine cells |
| `terrain.mountain_local_relief_r10km_sample_p90_m.v1` | current convergence descriptor | 90th percentile 10-km-radius max-minus-min relief over a deterministic sample of cells above 1.5 km | m; cell sample | short-scale resolved relief; adaptive sampling and single extrema remain confounders |
| `terrain.mountain_local_relief_r25km_sample_p50/p90_m.v1` | current convergence descriptor | median/90th percentile 25-km-radius max-minus-min relief over the same mountain sample | m; cell sample | fixed-scale convergence signal; not area-weighted and can be inflated by isolated extrema |
| `tectonics.material_relative_residual.v1` | current invariant | declared material-ledger closure residual divided by declared reference volume | dimensionless; global ledger | correctness gate once tolerance/reference are registered; says nothing about placement |
| `tectonics.carrier_peak_span.v1` | experimental gate | max-min peak elevation across 4096/8192/16384 carrier cells on fixed terrain; fail above 2 km | km; per-seed span | claim ambiguous between continuum convergence and procedural-scale robustness |
| `tectonics.carrier_land_span.v1` | experimental gate | max-min land coverage across carrier resolutions; fail above 2 percentage points | percentage points; area intended | sea-datum response can hide/redistribute deformation |
| `climate.relative_aridity_index.v1` | current field descriptor | precipitation divided by `0.2+0.8*clamped_temperature`, normalized to area-weighted land mean one | ratio; field is area-normalized | within-world stress pattern only; cannot compare absolute dryness across worlds |
| `hydrology.semantic_river_cells.v1` | current semantic feature | cells selected by declared `RiverThresholdPolicy`, product default minimum catchment 2,000 km² | cells plus approximate km/km² | renderer/audit shared population; grid-cell path and coarse floor affect geometry |
| `hydrology.lake_area_fraction_of_terrestrial.v1` | current field/object descriptor | retained lake-water area divided by all non-ocean terrestrial area, including lakes | fraction; area-weighted | distinguishes lake coverage from dry-land denominator; climate ratio and minimum lake rules apply |
| `hydrology.flow_fraction_of_max.v1` | provisional/default descriptor | land cells above 1% of world maximum flow | cells; count-weighted | incompatible with semantic river population and unstable to one extreme |
| `hydrology.seeded_face_path_cross_track_max_km.v1` | evaluated R1a gate | maximum absolute analytic cross-track coordinate over prescribed head plus selected shared-face midpoints and portal-face midpoint | km; path vertices | both R1a arms fail the frozen worst-case gate; resolved-valley geometry only, and the face midpoint cannot establish a subcell thalweg |
| `hydrology.seeded_face_path_arclength_error.v1` | evaluated R1a gate | absolute difference between selected-face polyline length and analytic reference length, divided by analytic length | ratio; one seeded path | R1a P0 passes V's arclength gate but not A's; M0 fails worst-case A and V arclength; endpoint/portal discretization remains explicit |
| `hydrology.seeded_face_path_backtracking_km.v1` | evaluated R1a gate | sum of positive upstream-coordinate increments along a nominally downstream selected-face polyline | km; one seeded path | all successful R1a A/V paths report zero, which does not rescue lateral drift or sink termination |
| `hydrology.seeded_path_portal_termination.v1` | evaluated R1a invariant | whether a prescribed-head trace reaches its required semantic portal rather than a sink, cycle, wrong portal or guard | boolean plus typed failure and validated partial prefix; one path | P0 fails 6/12 affine cases and M0 fails 4/12; conservative sink storage is not successful centreline termination |
| `hydrology.local_receiver_margin.v1` | evaluated R1a diagnostic | `(best-second)/max(|best|, epsilon)` reported separately for physical grade and MFD face fraction at every visited donor | ratio; path steps including typed failure prefixes | numerical dominance, not physical confidence; positive margins on failed affine paths rule out a tie explanation |
| `hydrology.local_receiver_tie_decision.v1` | evaluated R1a diagnostic | first rank key deciding best over runner-up: score, midpoint x/y/z, portal ID or combined face index; sole-face case separate | categorical; selected path faces | no R1a A/V success or failure prefix uses an exact-score/build-index tie; deterministic choice is not physical validation |
| `hydrology.r1_domain_receiver_conflict_fraction.v1` | current R1a precheck diagnostic | donors whose greatest physical-grade face differs from their greatest MFD-fraction face, divided by all eligible donors | fraction; one registered routed case | proves unequal geometry creates a discriminator but cannot replace a visited-cell witness or path scoring |
| `hydrology.r1_head_receiver_conflict.v1` | current R1a anti-alias subgate | count of prescribed heads whose P0 and M0 local winners differ | heads; one registered routed case or declared matrix sum | a head is necessarily visited by both arms, so a nonzero A/V matrix total is an existential visited-cell witness; says nothing about downstream validity or physical fit |
| `hydrology.r1_route_water_balance_relative_error.v1` | current R1a invariant | absolute source-minus-portal-minus-sink residual divided by total registered supply | ratio; one routed fixture case | global conservative closure only; local face, cell, portal and sink ledgers are checked separately and a closed-domain sink is not successful portal termination |
| `hydrology.r1_affine_generator_termination_repair.v1` | evaluated causal diagnostic | registered polygon-mean A termination failures whose paired generator-sampled A trace reaches the required portal under the otherwise unchanged algorithms | count; paired spacing × orientation × translation matrix, per receiver | zero for P0 and M0; alternate state placement is insufficient, while geometry remains censored unless both paired traces reach the portal |
| `mesh.voronoi_internal_face_width_quantiles_km.v1` | provisional fixture descriptor | p10/p50/p90 projected lengths of reciprocal two-vertex internal faces in the retained S2 cap | km; face-weighted | validates unequal geometry and mesh quality; not channel width |
| `mesh.voronoi_cell_area_quantiles_km2.v1` | provisional fixture descriptor | p10/p50/p90 projected polygon areas of retained cells in the guarded S2 cap | km²; cell-weighted | confirms positive unequal control-volume geometry; not a product-mesh area distribution |
| `mesh.voronoi_generator_centroid_offset_quantiles_km.v1` | provisional fixture descriptor | p50/p95 distance from each projected generator to its projected polygon area centroid | km; cell-weighted | exposes the approximation between polygon-mean state and generator-distance two-point operators; does not itself measure operator error |
| `mesh.tangent_projection_edge_distortion_max.v1` | provisional fixture gate | maximum absolute relative difference between a projected retained-edge length and the normalized unit-sphere great-circle arc of the same two vertices | ratio; worst internal or cut face | G0 threshold `0.001`; validates the local adapter, not planarization of a product globe |
| `mesh.tangent_projection_center_distance_distortion_max.v1` | provisional fixture descriptor | maximum absolute relative difference between projected and spherical generator-to-generator or generator-to-boundary-face-midpoint distance | ratio; worst internal or cut face | report-only companion to the two-point finite-volume geometry; no independent R1a gate |
| `mesh.tangent_projection_face_midpoint_error_max_km.v1` | provisional fixture descriptor | maximum distance between the mean of projected edge endpoints and the projection of their spherical great-circle midpoint | km; worst internal or cut face | reports the F0 midpoint convention's planarization effect, not unresolved physical crossing error |
| `mesh.tangent_projection_cell_area_distortion_max.v1` | provisional fixture descriptor | maximum per-cell absolute relative difference between projected polygon area and spherical cell area | ratio; worst retained cell | report-only local complement to the registered total-area gate; spherical reference inherits backend `f32` geometry |
| `mesh.tangent_projection_total_area_error.v1` | provisional fixture gate | absolute difference between summed projected retained polygon area and summed spherical retained-cell area, divided by spherical area | ratio; retained cap | G0 threshold `0.002`; the spherical reference inherits backend `f32` geometry |
| `erosion.river_profile_bow.v1` | historical/current descriptor | median normalized midpoint bow and percent concave over deduplicated long rivers, historically up to 50 and at least 200 km | ratio; object-weighted | useful aggregate grading evidence; selection and lake endpoints matter; no maintained gate |
| `erosion.population_slope_area_theta.v1` | superseded/invalid | population regression of local channel slope against drainage area | exponent; mixed channel samples | retained negative result only; lakes, mesh edges, bins and mixed regimes invalidate gating |
| `ecology.biome_transition_area.v1` | provisional semantic descriptor | land area with biome classification confidence below 0.20 | percent land area; area-weighted | exposes classifier ambiguity; threshold/calibration not promoted |
| `performance.stage_wall_time.v1` | current product indicator | wall time per declared pipeline stage and total in corpus artifacts | seconds; run/platform/build mode | initial cost evidence; process memory and hardware normalization remain absent |
| `mesh.active_cells.v1` | current run descriptor | number of cells exposed at the retained viewed/computed stage | count | required resolution context; never a quality metric |

### Immediate semantic splits required

- Create distinct IDs for cell-weighted and area-weighted elevation/climate
  quantiles rather than silently changing historical values.
- Register native elevation/radian, elevation/km and physical-grade slope as
  separate quantities.
- Give each local-relief radius and robust/non-robust estimator a parameterized
  identity.
- Replace generic “river” output labels with the selection policy/metric ID.
- Treat historical plateau proxies as competing hypotheses until a shared range/
  plateau semantic object is justified.

The populated registry may be split into machine-readable data and generated
tables once the inventory establishes the required fields. Until then, this
document is the authoritative schema and policy boundary.
