# Channel centreline geometry and confluence basis

**Date:** 2026-07-13
**Status:** research synthesis governing extraction R1a

## Conclusion

A conservative face-flow field, a streamline reconstruction and a sparse river
network are three different objects.

- Face flux owns water accounting across finite-volume cells.
- A reconstructed streamline is one geometric interpretation of that flux
  inside cells.
- A river graph coalesces distributed flow into persistent reaches and can
  merge tributaries.

R0 blurred those objects. R1a must first decide a seeded receiver path in one
resolved valley and test its geometry without using cell-centre graph length.
A Y confluence is not a valid smooth-streamline oracle and is deferred.

## Why smooth streamlines do not form a merge-only Y

For a locally Lipschitz steady vector field, the initial-value problem has a
unique integral curve. If two integral curves meet, uniqueness makes them the
same curve on their common domain. Distinct trajectories therefore cannot meet
at finite time and then share a downstream suffix. The result applies to the
negative gradient of a smooth terrain surface; see the flow uniqueness result
in [MIT's vector-field notes](https://math.mit.edu/classes/18.101/fa07/pub/manifolds-4.pdf).

A discrete one-receiver graph can merge because it is a coarse graining. A
smooth terrain can also have a Y-shaped locus of transverse valley minima, but
that geomorphic skeleton is not a pair of merging particle streamlines. Its
junction is normally a scale-sensitive degeneracy, not an exact flow point.

This changes the experiment design:

- use smooth single-valley fixtures to score path direction and geometry;
- use an explicitly conservative graph patch to test merge/no-split topology;
- if a geomorphic Y is later required, validate it as a transverse-morphology
  oracle and give its junction an uncertainty region of order one cell; and
- never use a claimed smooth merging-gradient Y to select an extractor.

## What finite-volume flux identifies

Integrated face fluxes determine conservative transfer, not a unique velocity
field or crossing point inside a polygon. Any subcell path needs a disclosed
reconstruction assumption.

The cheapest honest convention is to connect the prescribed head through the
midpoints of selected shared faces to the outlet-face midpoint. It respects the
actual faces crossed and removes the known centre-to-centre detour, but it is a
geometric convention rather than recovered channel physics.

If that convention fails even affine-flow convergence, the principled upgrade
is a locally conservative `H(div)`/lowest-order Raviart–Thomas-style velocity
reconstruction followed by entry-to-exit integration. Such reconstructions are
designed to preserve face flux and reproduce simple flow fields; relevant
examples include [Klausen et al.](https://www.sintef.no/globalassets/project/geoscale/papers/klausen-2011.pdf)
and [Zhang et al.](https://doi.org/10.1029/2011WR011396). They still produce
streamlines, not a coalescing river graph.

| Representation | Role | Physical status | R1a status |
|---|---|---|---|
| Cell-centre neighbour polyline | Graph diagnostic | Mesh walk; length can be biased without convergence | report only |
| Selected-face midpoint polyline | Cheap subcell convention | Authentic numerical hack with explicit face crossings | eligible baseline |
| Conservative `H(div)` reconstruction | Interior flux/velocity interpretation | More physically based numerical reconstruction | escalation only |
| Persistent vector reaches | River topology and state owner | Explicit subgrid model | later lineage/C1 integration |
| Curvature/geodesic skeleton | DEM morphology extraction | Scale-dependent parameterization | ineligible at kilometre R1a |

## How established models avoid the conflation

Large-scale routing systems commonly carry rivers as explicit vector reaches
with their own topology, slope, length and contributing catchments rather than
deriving reach geometry from a cell-centre walk; [mizuRoute](https://doi.org/10.5194/gmd-9-2223-2016)
is one example.

On arbitrary meshes, prior vector rivers can be intersected with actual cell
faces while preserving upstream/downstream reach topology. Liao et al. compare
area difference, branching angle, sinuosity and reference length, but their
successful roughly 3–6 km confluence cases use prior flowlines and
flowline-guided meshes rather than spontaneous recovery of subgrid channels
([primary paper](https://www.osti.gov/pages/servlets/purl/1964150)). That is
evidence for separating vector-network ownership from mesh routing, not for
assuming Hex3 already knows a precise junction.

GeoNet combines nonlinear DEM filtering, curvature, contributing area and
geodesic paths at lidar scale
([Passalacqua et al.](https://doi.org/10.1029/2009JF001254)). Its extra scales
and endpoint choices are not free physical evidence at kilometre resolution.
Even much finer DEM-derived slopes and drainage attributes remain sensitive to
cell size ([Zhang and Montgomery](https://doi.org/10.1029/93WR03553)).

## Claims justified at Hex3 scale

At kilometre cells, a successful reduced model may justify:

- conservative runoff and outlet accounting;
- basin and major-reach topology;
- discharge/drainage-area ordering;
- a coarse reach corridor and confluence vicinity within `O(cell size)`; and
- stable semantic geometry whose uncertainty is explicit.

It does not identify exact thalweg position, bankfull width, banks, bars,
meanders or a subcell junction point. Smoothing, snapping, stream burning and
morphological skeletonization would be reduced parameterizations or
presentation choices unless finer evidence is introduced.

## Consequence for the architecture

The layered ownership decision remains sound but gains a sharper boundary:

1. MFD face flow owns instantaneous water.
2. A local receiver rule may provide a seeded graph path where a valley is
   resolved.
3. Face-crossing geometry is a derived interpretation with its own validation.
4. Merge topology, initiation, persistence and C1 state belong to the sparse
   network layer and cannot be proved by one smooth streamline fixture.

R1a therefore excludes Y and M1. It compares only physical-grade versus
dominant-integrated-flux local receivers on representative unequal face
geometry. A conservative Y topology patch and any smooth geomorphic-Y stress
test are later, separately preregistered rungs.
