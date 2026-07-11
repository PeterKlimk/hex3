> **Research note:** Exploratory material, not current repository truth. See [`docs/gaps.md`](../gaps.md) for the maintained research agenda.

# Meso-structure design consultation — GPT 5.6 via codex exec (2026-07-10)

Context: relief-spectrum redesign (spec §11) — composed meso regime passed numeric
gates; user visual verdicts: 'dune-y … ridges too consistent', then irregularity
sweep 'generally similar / not sure what they should look like'. Consultation
prompt in git history; memo verbatim below. Verify claims before load-bearing use;
spot-checks of the cited refs (Hovius R≈2, Perron valley spacing) look right.

## 1. What 10–50 km mountain structure should look like

The blunt verdict: the longitudinal fold train is a good specialized prior for thin-skinned fold–thrust terrain, but it is the wrong default for an alpine collision belt. Increasing its irregularity will produce “less synthetic corduroy,” not “the Alps.”

There is also an unresolved amplitude issue. Going from 190 m to 580–650 m is a major improvement, but it has not closed an Earth-alpine target of 1500–3000 m. At 25 km, 600 m reads more like deeply dissected hills or Appalachian-scale relief than a high alpine massif. USGS puts typical Valley-and-Ridge local relief around 300 m; western Southern Alps transverse ridges reach roughly 1000–1500 m above intervening valleys. [USGS](https://www.usgs.gov/centers/florence-bascom-geoscience-center/science/appalachian-basin-geologic-mapping-project?field_pub_type_target_id=All&items_per_page=6&page=6), [Southern Alps study](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL078692)

At these wavelengths, different mountain types should look like this:

| Regime | Dominant 10–50 km signature | Drainage expression |
|---|---|---|
| Zagros/Jura/Valley-and-Ridge | Several long, strike-parallel ridges; fairly persistent spacing; coherent fold packets separated by transfer zones | Trellis drainage: long strike valleys, short cross-ridge tributaries, occasional major water gaps |
| Alpine collision core | Irregular massifs separated by deep passes and trunk valleys; one or a few axial divides, with branching side ridges | Transverse or oblique main valleys, dendritic tributaries, interdigitating valley heads |
| Southern Alps NZ wet flank | Short, steep ridges and catchments normal to the range; strong divide-to-front relief | Small, dendritic, mostly transverse western catchments |
| Glaciated alpine interior | Large trough systems, broad valley heads, hanging tributaries, blunt or truncated spurs | Drainage hierarchy inherited but strongly widened and reorganized by ice |
| Continental arc | Segmented axial highland plus discrete volcanic or intrusive massifs | Radial drainage locally, transverse dendritic drainage between centers |

The Appalachians are especially instructive: their repeated ridges are not merely a noisy uplift wave. Alternating resistant and weak strata produce persistent strike ridges and valleys, while streams form a trellis network with both longitudinal and transverse components. A sinusoidal uplift field without that trellis organization captures the silhouette but not the landform grammar. [USGS description](https://pubs.usgs.gov/ha/ha730/ch_l/L-text5.html)

The Southern Alps demonstrate the other grammar. West of the Main Divide, catchments are smaller, dendritic, and mostly perpendicular to the divide; the eastern side contains larger, partly inherited glacial valleys, some subparallel to structure. In other words, even one orogen contains both drainage-organized and structure-controlled provinces. [Herman and Braun](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JF000248)

For a generic “alpine” default, I would target approximately:

- 10–20% of meso variance in strike-parallel structural grain.
- 25–35% in massifs, saddles, transfer zones, and transverse/oblique segmentation.
- 50–60% in structure that is explicitly coherent with the eventual drainage hierarchy.

More importantly, distribute those components spatially. Put 50–70% longitudinal structure in a narrow outer fold–thrust apron, but only 0–15% in the alpine core. A global linear mixture is less convincing than a core-to-foreland facies transition.

A lay viewer primarily notices negative space:

- Deep valleys that branch and converge.
- Spurs projecting into those valleys.
- A skyline composed of unequal peak groups and saddles.
- Large valleys crossing or cutting into the range rather than every depression following strike.
- Hierarchy: one trunk valley, several tributary valleys, many short headwater hollows.

They do not generally perceive “more sophisticated phase decorrelation.” If the repeated strike rhythm remains the strongest gestalt, it still reads as dunes, ribs, or corduroy.

One further limit: cirques are mostly sub-kilometre to roughly kilometre-scale features—one 631-cirque inventory found mean dimensions around 0.65 × 0.72 km. They cannot exist geometrically on a 3.9 km mesh. [Cirque morphometry](https://www.sciencedirect.com/science/article/abs/pii/S0169555X13005989) At this resolution, aim for glacial trough systems and massif-scale valley heads, not literal cirques or arêtes.

## 2. Why the irregularity sweep looks similar

My ranking is:

1. **The dial changes realization more than visual class.**  
   All four cases retain the same dominant wavelength, strike alignment, amplitude envelope, and basic ridge count. Phase decorrelation changes which ridge connects to which, but a non-expert comparing separate renders sees “parallel ribbed mountains” in every case. Stationary textures with different phase can be numerically dissimilar while remaining stylistically identical.

2. **The mesh barely resolves the operations intended to carry the distinction.**  
   A 25 km wave has about 6.4 cell spacings. The 0.618λ octave is about 15.5 km, or four cells. Four cells can affect p95–p05 and slope, but cannot reliably describe crest wandering, bifurcation, tapered termination, or a believable saddle. Those are multi-sample shapes, not merely frequencies. I would regard eight cells per intended geomorphic object—roughly 30 km here—as a practical minimum.

3. **Hillshade preserves orientation and slope statistics more strongly than identity.**  
   With a fixed sun, all strike-parallel ridges present similar illuminated and shadowed faces. Moving a crest, breaking its phase, or swapping its neighbor often leaves the histogram and directional distribution of surface normals nearly unchanged. The eye therefore sees the same fabric.

4. **Erosion is probably a common transfer function, but not necessarily a complete washout.**  
   MFD plus \(n=2\) incision strongly selects drainage paths and creates a shared channel texture. Adjacent small valleys compete, capture area, and eliminate one another; numerical landscape models naturally evolve toward quasi-regular valley spacing through this competition. [Perron et al.](https://seismo.berkeley.edu/~kirchner/reprints/2008_85_Perron_valley_spacing.pdf) That can erase differences that do not alter major basin topology.

   However, your short epoch and 89% relief retention argue against “erosion erased everything” as the first explanation. Test it rather than assuming it.

5. **The montage probably asks the eye to perform the wrong task.**  
   Globe views are excellent for range envelopes and terrible for following a single 25 km crest through a termination. Even a zoom will obscure this if crops, cameras, or lighting differ.

The decisive diagnostic is a fixed crop with four aligned panels for each dial value:

- Band-passed uplift, 10–50 km.
- Band-passed final elevation.
- Extracted crest and major-channel objects.
- Identical hillshade.

Measure both uplift-to-height coherence and endpoint difference RMS. If the final elevation objects differ but hillshade does not, the visualization is hiding the change. If uplift differs but final crest/channel objects do not, erosion is absorbing it. If neither object population changes much, the dial is simply not changing the terrain class.

I would also remove the 15.5 km octave for this experiment. It is too close to representational failure. Move secondary organization upward, perhaps 35–60 km, and let erosion generate the smaller relief.

## 3. Object-level metrics

First remove the ≥75–100 km range envelope, then extract objects from a 10–50 km residual. On the irregular mesh, use graph-distance smoothing, MFD channels, watershed divides, and a graph-based crest or discrete-Morse skeleton. All lengths should use physical edge lengths and all aggregation should be cell-area weighted.

The numerical bands below are recommended initial calibration targets after Earth DEMs are degraded to your effective 3.9 km resolution. Except where noted, they are not universal published constants.

### 3.1 Ridge-track persistence plus spacing regularity

At cross-strike transects spaced along \(u\), detect crest intersections and link them between adjacent transects. Report:

- Cross-strike spacing CV.
- Median strike-parallel track length divided by local wavelength, \(L_{\rm track}/\lambda\).
- Termination, merger, and split counts per 100 km of strike.
- Fraction of crest length belonging to the largest few tracks.

Useful starting bands:

| Class | Spacing CV | Median \(L_{\rm track}/\lambda\) |
|---|---:|---:|
| Metronomic train | <0.15 | >8 |
| Natural fold packets | 0.20–0.45 | 2–8 |
| Alpine core | >0.45 or no stable spacing mode | <2 for strike-parallel tracks |

Spacing CV alone is insufficient. Some real fold belts are genuinely regular. The powerful discriminator is regular spacing combined with exceptional persistence and almost no birth/death. Natural folds occur in coherent packets, but terminate, plunge, merge, step across transfer zones, and change wavelength by domain. Their irregularity is non-stationary and geological, not homogeneous random decorrelation.

This would be my highest-priority ridge metric.

### 3.2 Joint ridge/channel orientation and network type

For every crest and channel segment, calculate \(\Delta\theta\) relative to strike and a nematic order parameter:

\[
S=\langle\cos(2\Delta\theta)\rangle,
\]

where \(+1\) is strike-parallel, \(-1\) is transverse, and 0 is isotropic or balanced.

Do this separately for:

- Crest segments.
- Low-order channels.
- High-order trunks.
- Each flank.

Also retain the complete angular distribution. A single \(S\) can hide the bimodality diagnostic of a trellis network.

Expected signatures:

- **Metronomic train:** crest \(S \gtrsim 0.75\); channels strongly transverse; narrow angular modes.
- **Natural fold belt:** crest \(S \approx 0.45–0.8\); channels clearly bimodal, with both strike-following reaches and cross-ridge water gaps.
- **Drainage-organized alpine:** core crest \(S \approx 0–0.35\); low-order channels broad and dendritic; high-order trunks commonly transverse or oblique, often \(S \approx -0.25\) to \(-0.65\).

Add tributary junction geometry, but do not use mean junction angle by itself. Empirical dendritic networks average roughly 58–71°, while measured trellis networks overlap at about 61–77°. Orientation sequence and drainage-area accumulation distinguish them better than angle alone. [Network comparison](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JF000781) Fluvial and debris-flow networks also show characteristic modes near 50° and 75°. [Junction-angle study](https://stars.library.ucf.edu/scopus2015/5836/)

This is the best single metric family for distinguishing “fold belt” from “alpine,” because it tests the relationship between positive and negative landforms.

### 3.3 Major-basin spacing, shape, and hierarchy

For each flank, identify basins reaching the mountain front and measure:

- Divide-to-front length \(L_f\).
- Spacing \(S_o\) between major outlets.
- \(R=L_f/S_o\).
- Outlet-spacing CV.
- Hack exponent from main-channel length versus basin area.
- Basin-size inequality and Strahler-order distribution.

Hovius measured \(R\) around 1.91–2.23 for most of 11 linear mountain belts, so major outlet spacing is commonly about half the divide-to-front distance. The Nepal Himalaya was less regular because frontal structures divert drainage. [Hovius](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2117.1996.tb00113.x)

That produces an important warning: regular valley spacing is not itself artificial. Drainage competition naturally regularizes outlets. A plausible alpine target is approximately:

- \(R\): 1.8–2.4.
- Outlet-spacing CV: roughly 0.20–0.45.
- Hack exponent: about 0.5–0.6.
- A broad, right-skewed basin-area distribution rather than equal repeated strips.

A metronomic construction may hit \(R\) while failing the basin hierarchy: every basin has comparable width, length, and tributary count. That equality is the artifact.

### 3.4 Spectral anisotropy is a guardrail, not the primary verdict

Keep 2-D spectral peakiness and anisotropy, but classify them as field diagnostics rather than object metrics. They are excellent for detecting a narrow wavelength spike and excessive strike fabric. They are weak for distinguishing phase-randomized versions of the same process, because phase changes can leave the power spectrum nearly unchanged.

I would rank the candidate metrics:

1. Joint crest/channel orientation and topology.
2. Ridge persistence, termination rate, and spacing CV.
3. Basin spacing plus hierarchy.
4. Spectral anisotropy/peakiness.

The strongest validation would be a small reference library—Zagros, Appalachian Valley-and-Ridge, western and eastern Southern Alps, European Alps, Caucasus—low-passed and resampled to the same effective resolution before computing these ranges.

## 4. What I would seed through uplift instead

I would retire the continuous fold train as the generic alpine meso prior. Keep it as a named “fold–thrust apron” component.

### 4.1 Massif-and-saddle uplift

Replace repeated cross-strike oscillation with irregular uplift massifs:

\[
G(u,v)=\sum_i a_i
\exp\left[
-\frac{(u-u_i)^2}{2L_{u,i}^2}
-\frac{(v-v_i)^2}{2L_{v,i}^2}
\right].
\]

Use:

- Irregular \(u_i\) spacing of about 25–60 km.
- \(L_u\) around 10–30 km.
- \(L_v\) around 8–20 km.
- Centers offset toward either flank, not all on \(v=0\).
- Heavy-tailed amplitudes: many ordinary massifs, a few dominant ones.
- Low-uplift saddles or gaps between selected neighbors.

Band-limit and mean-normalize it so it redistributes uplift rather than increasing total uplift. This creates peak groups, unequal passes, and places for trunk valleys to cross the range. It changes the object vocabulary from “parallel ridges” to “massifs separated by corridors.”

### 4.2 Seed the negative space: branching low-uplift corridors

At alpine scale, valleys are the visually controlling objects. Since erodibility does not remain coherent, encode a weak drainage scaffold as reduced uplift rate.

Construct independent branching corridor trees on the two flanks:

- Roots at the mountain front.
- Major-root spacing \(S_o \approx 0.45–0.52L_f\), consistent with the observed \(R\approx2\).
- Trunks mostly transverse but allowed 20–40° obliquity and occasional strike-following reaches.
- Tributary junctions centered broadly around 50–75°.
- Heads interdigitating near the divide rather than pairing symmetrically across it.
- Only 10–20% of trunks allowed to cross the whole belt as antecedent or structurally guided rivers.
- Corridor widths no smaller than roughly 8–15 km at this mesh.
- Uplift reduction around 10–25%, sufficient for \(n=2\) incision to amplify.

The inter-corridor ground automatically becomes branching spurs and ridges. That is much closer to how an alpine image reads than explicitly painting every spur as a ridge.

### 4.3 Best option: a two-stage, drainage-aware uplift pulse

This is my preferred construction under your channel constraint:

1. Run a longer, lower-amplitude burn-in using only the macro tectonic envelope and broadband perturbations.
2. Extract the stable order-3-and-higher drainage and divide network.
3. Smooth that network to 10–40 km scales.
4. Build a zero-mean uplift modifier: lower uplift along major trunks, higher uplift across broad interfluves and massif centers.
5. Apply the high-relief, short final erosion epoch with this modifier frozen.

This still injects structure exclusively through uplift rate. It simply lets erosion propose a hierarchical scaffold before the short epoch amplifies relief. Use one feedback pass, not continuous feedback, or the network will lock into exaggerated spokes.

It also resolves the central tension in the current design: a short epoch preserves imposed structure, but a mature alpine drainage hierarchy normally requires time to organize. Pre-organize the drainage, then use the short epoch.

### 4.4 Keep folds, but make them geological domains

Where folds are appropriate, stop making irregularity spatially uniform. Build packets:

- Two to six coherent ridges per packet.
- Packet lengths around 40–150 km.
- Abrupt wavelength or phase steps at transfer zones.
- Salients and embayments in the front.
- Local fold mergers, plunging noses, and en echelon offsets.
- Drainage that sometimes follows the longitudinal lows and sometimes breaches the packet.

That will look more natural than continuously decorrelating every adjacent ridge. Real structural irregularity has domains and discontinuities.

For an alpine collision default, my final meso recipe would be:

- 10–15% broken structural grain.
- 30–35% massif/saddle and transfer-zone field.
- 50–60% drainage-derived uplift scaffold.
- Fold-dominated structure confined to a separate foreland mask.

For an arc, replace most folds with discrete axial centers, overlapping massifs, and radial-to-transverse drainage. For a Zagros or Valley-and-Ridge preset, invert the weighting and embrace the longitudinal trains—but accompany them with an actual trellis drainage signature.

The fold-train work is therefore not a dead end. It is a successful biome-specific component that has been asked to serve as a universal mountain generator. The alpine look requires changing the dominant objects, not turning the same irregularity dial further.
tokens used
