> **Archived:** Historical design or experiment record. It does not define current architecture, defaults, or priorities. Start at [`docs/README.md`](../../README.md).

# Erosion: Valleys, Not Channels — regime gate → path-coupled thalweg

**Provenance.** Trio discussion on 2026-06-16 (user + Claude + an independent
`codex exec` voice), following the routing ladder. It started from the user's
observation that hex3 has effectively **two kinds of rivers**: the hydrology
**path** network (thin, sub-grid, rendered) and the **carved channels** erosion
leaves in the terrain. Two independent analyses converged; this records the
agreed geology, the endgame architecture, and a two-phase plan.

See also: [`erosion-routing-ladder.md`](erosion-routing-ladder.md),
[`erosion.md`](erosion.md), [`fine-mesh.md`](fine-mesh.md).

**Philosophy constraint (unchanged).** Add the *mechanism* with named constants
as playground knobs; do not tune constants to chase probe numbers. The user
judges maps; we review magnitudes. "Unreasonably physically inspired", not
Earth-accurate.

---

## The problem

Stream-power incision lowers a **whole cell**, so the cell becomes the river.
On the adaptive fine mesh, plains cells stay wide (~7 km for a flow-refined trunk,
up to ~20 km otherwise), so a carved channel reads as a **7–20 km-wide ditch** —
the "huge one-cell river". It's a **width / representation** bug, not
over-deepening (incision depth ~ `A^m/d` actually *shrinks* on wide cells; any
lowering of a wide cell just reads as a wide river). The model implicitly assumes
**cell-width = channel-width**, which is ~true on 1.5 km mountain cells and very
false on coarse plains cells. Final hydrology runs on the eroded surface, so the
path and the carved channel co-locate — same river, but the carved one is
wrong-scale.

## The geology (the settled framing)

- **Channels are sub-grid.** A river channel is meters to hundreds of meters wide;
  no planet-scale mesh resolves that, and real DEMs/LEMs don't either — a channel
  is a 1-D flow path with width from hydraulic geometry (`W ~ Q^0.5`, Leopold &
  Maddock 1953), not from mesh resolution. **Channels are path-owned.**
- **Erosion's job is the LAND the river runs through:** grading long profiles to
  concave river profiles, carving **valleys** where uplift is active, and
  **depositing** floodplains / fans / deltas / terraces in lowlands. So
  *"erosion can make valleys, not rivers" — and that is correct.* The path model
  is the river; erosion shapes the corridor.
- **Regime split (detachment vs transport limited; Whipple & Tucker 2002):**
  - **Mountains** — narrow V-valleys, bedrock, *detachment-limited incision*.
    Cell ≈ valley scale (fine), so cell-mean lowering ≈ valley incision. Keep it.
  - **Plains / lowlands** — broad flat floodplains, *transport-limited*. Alluvial
    rivers aggrade and meander (floodplains, levees, deltas, terraces); they do
    **not** carve canyons by default (they can incise after base-level fall).

## The endgame architecture — "two elevations"

Genuine unification of cell-terrain (valleys) and path-rivers (channels) needs a
second surface and a two-way link. This is the destination, deferred to Phase 2.

- **Cell surface** — valley / floodplain terrain, grid-scale (what we store now).
- **Path thalweg** — channel bed elevation + width + discharge + sediment load,
  along the 1-D network, sub-grid (new).
- **Forward coupling (terrain → path):** the path follows cell drainage; the
  valley floor sets where it goes. **Already exists** — hydrology runs on the
  eroded surface, so path and valley already co-locate. The gap is not *location*;
  it is sub-grid channel **state** + **features**.
- **Backward coupling (path → terrain):** the thalweg's stream-power profile
  drives the cell-mean terrain response, scaled by a **valley/confinement
  fraction**, with sub-grid features (levees, terraces, meander belt, distributary
  deltas) baked in as cell modulations.

**Codex's key trap (heed it).** Do NOT scale stream-power incision directly by
`channel_width / cell_width`. If uplift is applied to the whole cell while
incision only affects a sliver, the steady state demands absurd slopes and creates
resolution-transition artifacts. Instead the thalweg keeps a normal stream-power
profile; only the **amount transferred into cell-mean terrain** shrinks with
confinement. And the width that scales terrain is the **valley/floodplain width**
(broad-but-shallow on plains), not the channel width (which only sets the rendered
water).

## Two-phase plan

### Phase 1 — plains alluvial regime gate (now)

The cell-terrain response, without yet storing a separate thalweg. Full
detachment-limited incision in steep / high-relief / uplifting terrain; fade it
down in low-slope / low-relief / large-catchment plains, where the existing
transport-aware deposition + repose grading dominate → broad shallow corridors
instead of trenches. Rivers stay path-owned.

- **Mechanism:** a per-cell **confinement** factor `C ∈ [0,1]` (1 = confined
  bedrock → full incision; 0 = alluvial plain → no incision, deposition only),
  derived from slope (and/or local relief / drainage area). Scale the incision
  rate by `C`; let deposition fill the un-incised lowlands to a graded floodplain.
- **Playground knobs (Codex):** `bedrock_incision_confinement` (where C goes
  0→1), `alluvial_deposition_strength` (lowland aggradation), `valley_fraction_scale`
  (how much incision transfers to cell-mean — the Phase-2 transfer factor, here a
  scalar approximation).
- **Why first:** highest value/effort, fixes the visible ditch-on-plains, builds
  on the deposition code, and — crucially — **is the cell-terrain response layer
  Phase 2 reuses** (incise-in-mountains / aggrade-in-plains is needed there too).
  Not a detour; a prerequisite.
- **Validation:** maps (plains read as floodplains with the thin path on top, not
  trenches); `roughness_report` land-volume/elevation stays sane; the gate is
  behind 0-disable knobs for A/B.

### Phase 2 — explicit thalweg coupling (later)

Make the second surface real: store the channel thalweg (elevation / width /
discharge / sediment) on the path network; couple it back to cell terrain via the
valley-fraction transfer; add sub-grid features as modulations. The ambitious
"two elevations" model — more moving parts, easier to destabilize, so it sits on
top of a solid Phase-1 regime response.

## Non-goals / notes

- **Resolution is a separate quality upgrade, not the fix.** More cells (incl. a
  cheaper live Voronoi update in s2-voronoi) improve valleys, dissection, fans,
  and the intrinsic cell-scale mesh noise — worth doing for its own sake — but a
  700 m carved "river" is still the mesh pretending to be the wetted channel. It
  shrinks the wrong feature; it does not change the representation.
- MFD routing (ladder Rungs 2–3) is dormant and orthogonal to this.

## References

- Leopold & Maddock (1953). *The hydraulic geometry of stream channels…* USGS PP
  252. (Channel width `W ~ Q^b`, b ≈ 0.5.)
- Whipple & Tucker (2002). *Implications of sediment-flux-dependent river incision
  models…* JGR. (Detachment- vs transport-limited regimes.)
- Tucker & Hancock (2010). *Modelling landscape evolution.* ESPL. (Sub-grid
  channel/valley representation on coarse meshes.)
