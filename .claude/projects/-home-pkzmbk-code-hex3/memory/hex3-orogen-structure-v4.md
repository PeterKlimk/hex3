---
name: hex3-orogen-structure-v4
description: June 2026 — erosion-v4 OrogenStructure: structured-emergent uplift (O0) + n>1 incision = the VALIDATED mountain approach (beats painted P1 dunes + smooth dome). Coarse-asymmetry deferred (double-applies). On branch erosion-v3-emergent-orogens.
metadata:
  type: project
---

**erosion-v4 OrogenStructure — the validated mountain-generation approach** (branch `erosion-v3-emergent-orogens`, June 2026). Follows [[hex3-emergent-orogens-finding]] (pure-emergent-from-smooth-uplift FALSIFIED) and the painted P1 work ([[hex3-fine-synthesis-p1]], whose P1b "sand dunes" proved global structure can't be painted). Spec: `docs/specs/orogen-structure.md`.

**The principle (both first-principles reviews converged):** tectonics owns GLOBAL structure, erosion owns DISSECTION, noise owns LOCAL heterogeneity (rate, not height). Our mistake was painting global structure (P1b).

**VALIDATED LOCKED STATE (user blind-ranked it best vs painted + smooth-emergent):**
- **O0 structured-emergent uplift** (`emergent_structured`, `fine.rs` `compute_emergent_uplift_shape`): the emergent builder's uplift source = demoted forcing × asymmetric front profile (steep foreland / gentle hinterland, side from overriding plate) × along-strike SEGMENTATION (real arc-length CHAINING of convergent fronts, `chain_fronts`) × land-floor + volume-normalized. Gives asymmetric, segmented ranges.
- **n>1 stream-power incision** (`EROSION_N`, default still 1; use ~2): Newton-solved implicit step; the shape lever ("ranges not bumps"). n=1 was the roundness culprit.
- **channel_support lowered 30→4** (`EROSION_CHANNEL_SUPPORT_KM2`): the high massif was UNDER-dissected (spikes = hillslope, not channels); ~3-5 dissects it into ridge-valley. Smoothing (diffusivity/uplift_smooth) was the WRONG lever (domed the spikes).
- **hillshade render fix** (`slope_shading` uniform, face normal via dpdx/dpdy): prior visual judgments were on broken flat lighting.

**Knobs/sweeps:** `--emergent-lambda 0.5 --emergent-structured 1 --erosion-n 2 --erosion-hillslope-crit 200 --erosion-steps 200`, `--sweep-stack o0` (painted vs smooth-emergent vs structured A/B). Painted path stays behind knobs for A/B.

**O-coarse DEFERRED (codex-confirmed):** the asymmetric COARSE envelope (`coarse_asymmetry`, `features.rs asym_band`, for atmosphere/rain-shadow consistency) DOUBLE-APPLIES with O0's front profile → "inner massif + surrounding BARRIER" artifact. Decision B: O0 owns asymmetry, `coarse_asymmetry`=0 (default-off, guarded). Retained as scaffolding. Eventual fix = single-owner C/A hybrid (one shared OrogenFronts signed-distance product; coarse owns broad envelope, O0 drops its front-profile → demoted×segmentation; collision stays symmetric). Also a real secondary bug there: `compute_overriding_side` sign (nearest-midpoint) vs distance (Dijkstra) source-mismatch — fix before re-enabling.

**NOT yet merged to main.** Cache is at FINE_BASE_CACHE_VERSION 9. Per [[hex3-sweep-image-reading]], terrain aesthetics are the USER's call (numbers necessary-not-sufficient); the user judges via the Windows hillshade sweeps ([[hex3-sweep-harness]]).
