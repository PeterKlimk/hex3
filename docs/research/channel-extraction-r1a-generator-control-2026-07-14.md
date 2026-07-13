# R1a affine generator-point causal control

**Date:** 2026-07-14
**Status:** preregistered; not yet evaluated
**Parent experiment:** [Irregular-Voronoi seeded extraction R1a](channel-extraction-r1a-2026-07-13.md)
**Motivation:** [R1a path audit](../audits/channel-extraction-r1a-path-2026-07-14.md)

## Decision

Run one report-only paired intervention before implementing a new gradient,
flux or path operator. Replace the exact polygon-mean values of the affine A
control with exact values of the same affine surface at the retained Voronoi
generators. Change nothing else.

This is a falsification control for the diagnosed state-location mismatch. It
is not a physical terrain representation, a third receiver arm or a candidate
product mechanism. It cannot promote P0 or M0, override R1a's negative verdict,
or authorize reinterpreting conserved cell means as point samples.

## Frozen intervention

For registered case frame `(o, u, v)`, use the existing projected generator
`x_i = mesh.cell_center_km[i].xy` and construct

```text
s_i = (x_i - o) dot u
z_i = 1 km + 0.01 s_i
```

Only A receives this alternate sampling. Do not construct generator-point V or
B: doing so would mix the state-location question with nonlinear quadrature and
broad-corridor interpretation.

The paired polygon-mean and generator-point cases must share exactly:

- the guarded 8/4/2 km S2-Voronoi cap and projected mesh;
- prescribed head point, polygon owner and semantic portal `401`;
- cell areas and uniform runoff supply;
- the unfilled two-point face-grade algorithm, MFD-slope partition and boundary
  treatment;
- the P0/M0 score definitions, exact tie ordering and path-local recomputation;
- C0/F0 encodings, typed termination and all path metrics; and
- deterministic and conservative route audits.

Apply those computations independently to each elevation representation.
Numeric grades, fractions, ranks, routes, portal flux and sink storage are
expected to differ; freezing them would remove the intervention from the
operator being tested.

The generator-point construction must enter only before the shared route/case
assembly. The accepted exact-polygon-mean builder and its outputs must remain
bit-identical. Keep the control test-only or otherwise explicitly provenance-
wrapped so that its `Affine` config cannot be mistaken for registered physical
state.

## Matrix and reporting

Evaluate only A over the complete registered matrix:

```text
spacing h:       8, 4, 2 km
orientation:     0, 0.31 rad
translation:     0, 0.7 km
receiver:        P0 physical grade, M0 MFD fraction
representation:  exact polygon mean, generator point
```

For each paired trace report the exact typed outcome (`ReachedPortal`, `Sink`,
`Cycle` or `CellCountGuard`), F0 maximum absolute cross-track, relative
arclength error, backtracking, outlet error, cell count, exact-score ties,
build-index ties and the minimum visited P0/M0 margin. Malformed input, wrong
portal ownership or inconsistent cached ranks are invariant errors, not
experimental outcomes. Report route water closure and sink storage; a globally
conservative sink does not count as successful prescribed-head termination.

Preserve R1a's two distinct geometry summaries:

- for each representation independently, compute the ordinary four-case
  minimum, maximum and spread only from successful traces and report the number
  censored by termination, exactly as in the path audit; and
- for paired causal geometry, compare raw metrics only for registrations where
  both representations reach the portal. If either member fails, that pair's
  geometry is non-evaluable and its retained failure prefix is diagnostic only.

Do not claim a paired geometry-subgate transition unless all registrations
required by that subgate are successful in both representations. Incomplete
common-success subsets may show raw improvements but cannot support a gate-level
causal attribution.

Repeat every build and trace exactly. The cap, registered case, diagnostic case
and trace context must remain immutable. No outcome-quality assertion may be
introduced before observing the matrix.

## Frozen interpretation

Use R1a's already-frozen A F0 subgates only as common measuring instruments:

- all 12 prescribed-head traces reach portal `401`;
- worst 2 km cross-track is at most `3 km` and strictly below worst 8 km;
- worst 2 km relative arclength error is below `5%`;
- worst 2 km backtracking is at most `2 km`;
- the frozen four-registration robustness rule passes; and
- no selected face is decided by the final build-index key.

They do not become promotion gates for this control. Interpret each arm and
subgate separately:

1. For each registration, a polygon-mean termination failure paired with a
   generator-point `ReachedPortal` outcome shows that the polygon-mean/
   generator-geometry interaction is load-bearing for that termination under
   the otherwise unchanged stack.
2. A generator-point termination failure proves that changing state placement
   alone is insufficient; the unchanged operator, boundary treatment, receiver
   or their interaction remains load-bearing.
3. Attribute a numeric geometry-subgate transition only on a complete paired-
   success set as defined above. Otherwise report independent full-control
   gates and common-success raw metrics without a component-level causal claim.
4. If every generator-point A subgate passes for an arm, this single
   intervention is sufficient to repair that arm's aggregate affine outcome.
   It identifies the polygon-mean/generator-geometry mismatch as load-bearing,
   but does not by itself decompose face scoring, path selection, boundary
   handling or F0 geometry. It says nothing about V, sparse-centreline promotion
   or product terrain semantics.
5. If termination is repaired but generator-point geometry still fails, the
   mismatch explains the paired termination repair while the unchanged
   downstream stack remains inadequate; do not name a particular component
   without another intervention.
6. If geometry improves but termination still fails, report the paired paths
   and boundary interaction; do not average the categorical failure away.

Do not use a percentage-improvement threshold, retune the surface, relax the
existing gates or choose the visually better trace. The paired binary subgates
and disclosed raw metrics carry the causal conclusion.

## Stop rule and next rung

Stop after this matrix and write a dated audit. Do not implement a centroid
gradient in the same checkpoint.

Regardless of the outcome, generator values are not a promotable repair for
conservative polygon-mean state. The next physical candidate, if still
justified, is a local linear-exact gradient reconstructed from polygon means at
polygon-centroid geometry and projected onto actual face normals. A shared
conservative non-orthogonal face flux is a later rung for M0; continuous
`H(div)`/RT0 crossings remain ineligible until the underlying face fluxes are
trustworthy.
