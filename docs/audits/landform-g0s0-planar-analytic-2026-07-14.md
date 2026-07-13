# Landform G0/S0 planar analytic 8/4/2 audit

**Date:** 2026-07-14

**Verdict:** pass after evaluated sampling-support corrections

**Contract:** [G0/S0 executable contract](../research/landform-object-packet-g0s0-2026-07-14.md)

## Result

The frozen planar analytic matrix passes at nominal 8, 4 and 2 km. It observes
only prescribed continuous fields sampled at the preregistered cell centers;
no product terrain, H/C/G output or renderer state entered the extractor or
the oracles.

The matrix covers:

- the isolated cone, two-cone saddle, narrow/broad cap pair and linked
  four-cone sequence;
- identical expected topology and reference retention at every resolution;
- analytic peak labels without cross-resolution object correspondence;
- apex and merge-contact elevation/location bounds;
- strictly greater broad-cap area and low-threshold gentle fractions;
- orientation ambiguity, full cap grade validity and merge censoring;
- exact adapter/reference area closure and an actual-boundary 100 km buffer;
- both registered rectangles; and
- linked-root and all three losing-object area/ellipse accuracy.

The full audit is an ignored, explicit test because the 2 km domain contains
268,800 cells. It is not part of the ordinary fast library-test path.

## Evaluated contract corrections

Pre-execution amendment B froze missing fixture equations, the exact lattice
phase and center-sampling rule. It also replaced an unjustified monotonic
quadrature-error claim while retaining the final 2 km accuracy limits.

The first audit run stopped at the first failed location gate: the lower
two-cone saddle-cell polygon was `8.6698845 km` from the continuous contact on
the 8 km mesh. Amendment C corrected the measured object from that serialization
cell to the causal merge-support complex of saddle cells and strictly higher
face neighbors.

The resumed run passed that event and stopped at linked C–D, whose merge
complex was `8.7459305 km` from the contact. Amendment D supplied the missing
regular-hex covering radius, giving the scale-independent sampling bound
`spacing + circumradius` for contact distance and Lipschitz elevation error.
This bound follows the lattice geometry and was committed before 4 or 2 km ran;
it was not fitted to the observed miss.

## Morphology errors

Percent relative errors against independent continuous oracles:

| spacing | object | area | ellipse length | ellipse width |
|---:|---|---:|---:|---:|
| 8 km | linked root A | 0.0461% | 0.0621% | 0.1198% |
| 8 km | linked loser B | 9.4374% | 0.2248% | 8.7335% |
| 8 km | linked loser C | 9.9871% | 4.7171% | 5.0742% |
| 8 km | linked loser D | 8.2869% | 3.0013% | 4.6896% |
| 4 km | linked root A | 0.0101% | 0.0153% | 0.0111% |
| 4 km | linked loser B | 6.6073% | 3.1352% | 3.3541% |
| 4 km | linked loser C | 2.9661% | 0.8194% | 2.0817% |
| 4 km | linked loser D | 7.1499% | 3.5128% | 3.5766% |
| 2 km | linked root A | 0.0022% | 0.0107% | 0.0089% |
| 2 km | linked loser B | 3.0697% | 1.4002% | 1.6245% |
| 2 km | linked loser C | 1.3909% | 0.6709% | 0.7026% |
| 2 km | linked loser D | 3.3128% | 1.4667% | 1.8251% |

The 2 km linked maximum errors are `3.3128%` area, `1.4667%` length and
`1.8251%` width, within the frozen `5%`, `5%` and `7.5%` gates.

| spacing | rectangle | area | ellipse length | ellipse width |
|---:|---|---:|---:|---:|
| 8 km | axis aligned | 3.9230% | 0.0183% | 3.9406% |
| 8 km | rotated 30° | 0.4071% | 0.4044% | 0.1293% |
| 4 km | axis aligned | 1.8505% | 0.0046% | 1.8458% |
| 4 km | rotated 30° | 0.4071% | 0.4064% | 0.0324% |
| 2 km | axis aligned | 1.0363% | 0.0011% | 1.0374% |
| 2 km | rotated 30° | 0.4071% | 0.4069% | 0.0081% |

The non-monotone rotated length error illustrates why amendment B removed
strict per-step monotonicity while retaining a final-resolution accuracy gate.

## Verification

- Explicit release audit matrix: **passed** in `31.54 s`.
- Ordinary `cargo test --lib`: **228 passed, 7 ignored**.
- `cargo build --bin hex3`: **passed**.
- `cargo fmt --check`: **passed**.
- `git diff --check`: **passed**.

## Remaining G0/S0 scope

This pass does not establish the spherical product `Tessellation` adapter,
spherical morphology, explicit irregular-cap control volumes, every remaining
multiway/permutation/error fixture, serialized packet aggregates, or product
reference observation. D0, O0, R0 and all competitive H/C/G composition remain
out of scope.

The next bounded implementation is the spherical/product and irregular-cap G0
geometry seam, followed by the remaining adversarial fixture closure. Do not
begin terrain-arm tuning from these manufactured results.
