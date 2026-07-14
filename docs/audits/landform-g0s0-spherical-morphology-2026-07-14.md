# Landform G0/S0 spherical morphology audit

**Date:** 2026-07-14

**Verdict:** manufactured pass; product observation not begun

**Contract:** [G0/S0 executable contract](../research/landform-object-packet-g0s0-2026-07-14.md)

## Result

The preregistered spherical S0 seam now passes its bounded manufactured
checkpoint. Amendment F was committed before its implementation or outcome was
inspected. This validates the evidence instrument on controlled spherical
geometry; it does not evaluate product terrain or promote a terrain owner.

The end-to-end fixture uses a 2,048-cell synthetic product-Voronoi sphere
uniformly rescaled to a 100 km radius. Its prescribed local cap passes through
the physical G0 adapter, surface hierarchy and spherical morphology path. No
product elevation was supplied to the extractor.

## Manufactured evidence

The focused fixtures establish that:

- polygon-derived spherical centroid directions feed azimuthal-equidistant
  tangent-plane moments, with each projected polygon rescaled back to its
  authoritative physical cell area;
- the local cap returns finite local footprint, relief and summit-cap evidence
  end to end;
- tangent least-squares grade, global signless orientation and minor-arc
  distance behave covariantly under rigid rotation of the sphere;
- bucketed fixed-radius spherical relief agrees exactly with a brute-force
  great-circle scan on the manufactured comparison;
- inclusive great-circle radius membership and exact point-to-minor-arc
  distance cover interior and endpoint cases; and
- a deliberately nonlocal footprint reports `NonLocalGeometry` while
  retaining the structural and other independently defined evidence, rather
  than fabricating local tangent moments or erasing the object.

These are manufactured checks of geometry, covariance and evidence-preserving
failure semantics. They are not observations of mountain organization.

## Verification

- Focused spherical morphology tests: **5 passed**.
- Focused existing landforms tests: **19 passed**.
- `cargo test --lib --no-fail-fast`: **235 passed, 7 intentionally ignored**.
- `cargo build --bin hex3`: **passed**.
- `cargo fmt --check`: **passed**.
- `git diff --check`: **passed**.
- `cargo clippy` completed with pre-existing warnings only.

## Scope and interpretation

This checkpoint completes the manufactured spherical G0/S0 morphology seam
defined by amendment F: local polygon-centroid AEQD moments, rotation-covariant
grade/orientation/arc geometry, exact spherical relief selection and an
explicit nonlocal result that preserves other evidence.

Product elevation and product landforms remain uninspected. No H, C or G
surface has been measured, and this pass supports no inference about their
terrain quality, causal adequacy or comparative merit. Product-reference
observation requires a separate bounded checkpoint.
