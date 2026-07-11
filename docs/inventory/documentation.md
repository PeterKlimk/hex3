# Documentation inventory

This is a current-state inventory, not the replacement documentation architecture.
It accounts for every repository Markdown file found with `rg --files -g '*.md'`
on 2026-07-12. Dispositions are **provisional** until the code/system inventories
and project review are complete. Dates below describe the documents' evidence,
not an assertion that their code claims remain true.

**Migration note:** the canonical replacement set has since been created and the
provisional archive/research/generated dispositions applied. Paths in the tables
below intentionally record where files were found during inventory; use
[`../README.md`](../README.md) for the maintained structure.

## Inventory conventions

- **Current** means recently written or explicitly maintained; it does not mean
  code-verified by this inventory.
- **Historical** means useful evidence tied to a dated implementation or experiment.
- **Code check** says whether claims should be checked against the current tree before
  they become canonical. Even an implemented spec generally needs this check.
- Generated reports and experimental memos should not be treated as architecture.
- Proposed dispositions are deliberately conservative: archive means preserve as
  evidence outside the canonical reading path, not discard.

## Root context and policy

| File | Stated purpose | Freshness / status | Overlap or conflict | Code check | **Provisional disposition** |
|---|---|---|---|---|---|
| `AGENTS.md` | Repository context and operating instructions for coding assistants; includes architecture, pipeline, constants and controls. | Active policy supplied for this sprint, but much of its technical snapshot predates stage 4 and recent terrain work. | Substantially duplicates `CLAUDE.md`. Its three-stage controls, module list and named constants conflict with the richer/four-stage account in `CLAUDE.md` and likely with current code. | Yes, for every architecture/control/constant claim; policy such as WSL/Windows is separately authoritative. | **Rewrite**, keeping a short canonical assistant policy and links to new architecture/current-state docs. |
| `CLAUDE.md` | Claude-specific build, CLI, analysis, architecture and controls guide. | Apparently newer than `AGENTS.md`: includes crust, CLI/export, upper wind, erosion stage and fine-terrain modules. Still a manually maintained snapshot. | Duplicates almost all of `AGENTS.md`; tool-specific naming is the main unique aspect. Competing assistant guides are an immediate drift source. | Yes. | **Merge** durable technical content into canonical docs; retain only genuinely Claude-specific instructions, or delete if none remain. |
| `analysis_999.md` | Generated `Hex3 World Analysis` report for seed 999. | Undated generated artifact; parameters and pipeline version are not recorded sufficiently to establish comparability. | Same schema as the three reports under `docs/`; unexplained root placement. | Reproduction/version provenance needed, not source inspection alone. | **Archive or delete** after deciding a generated-report retention policy. Prefer reproducible outputs outside the canonical docs tree. |

There is no root `README.md`. A replacement set will likely need a concise human
entry point rather than asking readers to infer the project from assistant policy files.
`scripts/requirements.txt` was inspected by the file search but is a dependency manifest,
not documentation, so it is not assigned a documentation disposition.

## Canonical intent, roadmap and presentation candidates

| File | Stated purpose | Freshness / status | Overlap or conflict | Code check | **Provisional disposition** |
|---|---|---|---|---|---|
| `docs/thesis.md` | Project thesis: coherent causal worlds, physical grounding → semantic interpretation → cartographic spectacle, fidelity vocabulary and decision principles. | Current; created for this documentation sprint. Normative rather than implementation-descriptive. | Intentionally supersedes narrower slogans in old roadmaps. It reconciles physical grounding with explicit presentation exaggeration. | Only links/implementation examples, if later added. | **Retain** as the goal/thesis document; review wording after the inventories. |
| `docs/presentation.md` | Contract separating physical terrain/audits from cartographic exaggeration; defines relief presets, river width and validation rules. | Very recent and directly motivated by the relief-scale incident. | Strongly aligned with `docs/thesis.md`. Specific preset numbers and “default” status may drift; older documents often reason from render appearance without recording scale. | Yes: CLI flags, keys, defaults, numerical exaggerations and shared consumers. | **Retain and rewrite/expand** into the canonical presentation architecture/contract. |
| `docs/archive/roadmap-mountains-2026-07-11.md` (formerly `docs/roadmap.md`) | Historical roadmap/idea inventory centered on “good-looking mountains & mountain ranges,” with process rules and a dated baseline. | Archived during this documentation sprint and superseded by the new canonical `docs/roadmap.md`. | Its narrow goal and chronological status log conflict with the broader thesis; it remains evidence for terrain decisions. | Required before reusing any old current-state claim. | **Archived** as historical mountain-work context. |
| `docs/ideas.md` | Post-roadmap backlog of physical mechanisms, missing couplings and performance ideas. | June-era living backlog; explicitly sketches, not specs. Some listed work has since landed or been parked. | Overlaps old roadmap and erosion/orogen specs. Its “unreasonably physically inspired” philosophy is useful but narrower/less precise than the thesis vocabulary. | Yes for every “already computed,” current limitation and implementation-status claim. | **Merge** surviving ideas into a new backlog/gap/Pareto document; archive the snapshot. |
| `docs/physically-inspired-roadmap.md` | Survey replacing painted/ad-hoc mechanisms with physical principles. | Historical June roadmap; its opening says several mechanisms are current even though later docs say they landed or were superseded. | Overlaps `ideas.md`, `algorithm-review`, `cheap-wins`, `circulation`, `flexure`, erosion specs and thesis. | Yes. | **Archive after merge** of still-relevant rationale into system assessments and backlog. |

## Audits, reviews and generated evidence

These are valuable provenance but should sit below canonical architecture, with a
clear “historical result at revision/configuration” banner.

| File | Stated purpose | Freshness / status | Overlap or conflict | Code check | **Provisional disposition** |
|---|---|---|---|---|---|
| `docs/algorithm-review-2026-06.md` | Domain review of algorithm choices, missing couplings and a consolidated roadmap, based on seed 12345. | Historical June baseline. Its headline gaps (monolithic plate typing, no moisture, no erosion) were subsequently addressed. | Superseded by implementation plans and later audits; still useful as decision provenance. | Required before reusing any current-state claim. | **Archive** as historical review. |
| `docs/algorithm-audit-2026-06-15.md` | Code-level correctness/numerics/determinism audit at named branch/tip, including resolved and accepted findings. | Strongly versioned historical evidence; unusually explicit that docs/comments were untrusted. | Later improvement plans and fixes consume its findings. Not a current architecture source. | No for what was observed at the cited revision; yes before claiming a finding remains open/resolved now. | **Retain in audits/archive**, add revision/status metadata if reorganized. |
| `docs/density-audit-2026-06-15.md` | Empirical adaptive fine-mesh allocation and convergence audit. | Historical, reproducible methodology with open follow-ups. | Feeds fine-mesh and relief-spectrum decisions; numerical defaults likely changed. | Yes for current numbers/tool flags. | **Retain in audits/archive**. |
| `docs/erosion-validation-2026-06-15.md` | Validates river grading and records a misleading metric/measurement lesson. | Historical but durable methodological value. | Corrects earlier interpretations in density/erosion work; should not be mistaken for comprehensive erosion validation. | Yes if its diagnostic/tool status is presented as current. | **Retain in audits/archive**; summarize durable validation lessons canonically. |
| `docs/resolution-independence-2026-06-16.md` | Coarse- and fine-axis resolution tests, fixes, known issue and reproduction commands. | Historical snapshot after specific fixes. | Complements density and algorithm audits; “FIXED” is revision-relative. | Yes for current convergence and commands. | **Retain in audits/archive**. |
| `docs/audits/orogen-numeric-sweep-2026-07-11.md` | Render-independent plateau/interior-relief/structured-strength sweep and rejected default change. | Very recent historical experiment. | Closely coupled to terrain reset, relief redesign and thin-sheet work. Corrects render-driven diagnosis. | Only to confirm current flags/defaults; measurements remain evidence for recorded build/config. | **Retain as audit evidence**. |
| `docs/audits/tectonic-promotion-scorecard-2026-07-11.md` | Promotion gates and extensive follow-up for moving-carrier/lifecycle tectonic experiments. | Very recent; contains multiple falsifications and retained experimental rungs. | Overlaps `tectonic-time.md`, `thin-sheet-orogeny.md` and `terrain-reset.md`; status can diverge because results were appended chronologically. | Yes for current promotion/default status. | **Retain as audit evidence**, cross-link from a concise current experiment registry. |
| `docs/analysis-seed-12345.md` | Generated baseline analysis for seed 12345 with monolithic plate typing. | Historical pre-crust/moisture baseline. | Superseded experimentally by the two same-seed variants below. | Reproduction metadata needed. | **Archive** as labeled baseline or delete if reproducibility is inadequate. |
| `docs/analysis-seed-12345-crust.md` | Generated same-seed report after per-cell crust changes. | Historical intermediate artifact. | Nearly identical schema/content to other seed reports; filename is the only clear experiment label. | Reproduction metadata needed. | **Archive** with experiment metadata, otherwise delete. |
| `docs/analysis-seed-12345-moisture.md` | Generated same-seed report after moisture changes. | Historical intermediate artifact. | Same duplication/provenance issue. | Reproduction metadata needed. | **Archive** with experiment metadata, otherwise delete. |

## Specifications and implementation hand-offs

The `docs/specs/` directory is not one document type: it mixes prospective specs,
living implementation logs, completed hand-offs, research memos, roadmaps and
falsified experiments. Each file is accounted for below.

| File | Purpose and apparent status | Overlap / conflict and verification need | **Provisional disposition** |
|---|---|---|---|
| `docs/specs/cheap-wins.md` | Spec for land–ocean thermal contrast, temperature-dependent basin evaporation and spreading-rate-derived ocean age. No explicit completion banner. | Descends directly from `physically-inspired-roadmap.md`; verify whether each mechanism/default landed. | **Merge/rewrite** surviving items into system backlog or individual current specs; archive original. |
| `docs/specs/circulation.md` | Design for one meridional overturning circulation with integration decisions and validation. | Old root docs describe simpler atmosphere; verify implementation and current equations/knobs before treating as architecture. | **Archive after extracting** current atmospheric design into canonical system docs. |
| `docs/specs/drainage-integration.md` | Core basin integration/outlet-breaching pass; explicitly implemented and merged 2026-06-22, with outcome and success criteria. | Overlaps hydrology architecture and later erosion/base-level work. Implementation and defaults still require current-tree verification. | **Archive as implementation decision record**; summarize current mechanism canonically. |
| `docs/specs/erosion.md` | Original fine-mesh fluvial erosion spec; itself notes the fine mesh changed after drafting. | Foundational but superseded by v2/v3/escalations and implementation evolution. Verify all model details. | **Archive**; replace with current erosion-system architecture. |
| `docs/specs/fine-mesh.md` | Original adaptive fine-mesh refinement infrastructure spec with performance/validation contract. | Complements density audit and staging; likely predates current fine pipeline/cache. Verify fully. | **Archive after extraction** into current geometry/fine-stage architecture. |
| `docs/specs/erosion-v2.md` | Roadmap for fine-terrain synthesis and coupled evolution; phases and philosophy. | Overlaps fine synthesis, erosion v3 and later orogen reset; some assumptions are superseded. | **Archive** as design evolution. |
| `docs/specs/erosion-fine-synthesis.md` | Detailed P1 fine-terrain synthesis; all three rungs recorded as landed, with visual sign-off caveats. | Later docs reject/park portions of painted structure and reset the cumulative stack. Verify which rungs remain available/default. | **Archive as experiment/implementation record**. |
| `docs/specs/erosion-v3-emergent-orogens.md` | Attempts uplift-rate-forced emergent orogens; explicitly says premise falsified with current solver. | Leads to `orogen-structure.md` and later reset/time work; should not remain in active-spec reading path. | **Archive as falsified experiment**. |
| `docs/specs/orogen-structure.md` | Erosion-v4 ownership architecture; reviewed sound-with-fixes; coarse implementation later deferred. | Its global-tectonics/dissection split partly survives in `terrain-reset.md`, but details conflict with later parking decisions. Verify current flags/defaults. | **Archive after merging durable ownership rationale** into canonical terrain architecture. |
| `docs/specs/erosion-escalations.md` | Ordered post-tuning structural roadmap with gate results and traps. | Many rungs have their own specs/outcomes; status table is time-relative. | **Archive**, migrate genuinely open work into new backlog. |
| `docs/specs/erosion-uplift-smoothing.md` | Escalation #1 diagnosis, mechanism and implemented A/B outcome. | A bounded experiment within escalation history, not current architecture. Verify whether code remains enabled. | **Archive as experiment result**. |
| `docs/specs/erosion-routing-ladder.md` | Standing SFD→MFD diagnostic/implementation ladder; rung 0 done, later rungs/open decisions remain. | Overlaps hydrology/erosion gaps and includes explicit nonphysical off-ramps. Verify current routing before carrying forward. | **Merge** open Pareto-positive routing questions into roadmap; archive ladder. |
| `docs/specs/erosion-glacial-streampower.md` | Proposed glacial stream-power correction/ladders for a then-current “prickle.” | `improvement-plan-2026-06.md` says glacial v1 is done while this proposes a next model; current role/status unclear. | **Rewrite or archive** after code/system audit establishes current glacial model. |
| `docs/specs/erosion-valleys-not-channels.md` | Two-elevation architecture and staged plan for broad valleys versus rendered/physical channels. | Directly relevant to physical-versus-cartographic separation, but may conflate geomorphic valley state with channel presentation. Verify implementation status (no explicit landed banner). | **Reassess and merge** durable idea into hydrology/rendering roadmap; archive spec if inactive. |
| `docs/specs/improvement-plan-2026-06.md` | Post-erosion audit plan; phases 1–3 done, phase 4 partly done/deferred; also claims glacial and fault-front v1 done. | A historical status nexus overlapping audit, ideas and many erosion specs. All DONE claims require verification. | **Archive** after transferring unresolved items. |
| `docs/specs/improvement-plan-2026-06-16.md` | Second dated correctness/coupling improvement plan with same-day implementation status and deferred cases. | Overlaps `algorithm-audit` and the prior improvement plan; chronology is difficult to infer from filenames alone. Verify current resolution of every carried item. | **Archive**. |
| `docs/specs/flexure.md` | Detailed elastic plate-flexure implementation spec, diagnostics, tests and forbidden shortcuts. | Descends from physically inspired roadmap and feeds later orogen/basin ideas. No top-level completion status; verify implementation and whether flexure is trench-only. | **Archive after extracting** current flexure design; retain as ADR only if decisions remain exact. |
| `docs/specs/rain-shadow.md` | Fine-mesh downwind precipitation experiment; implemented default-off and reclassified from mountain-shape to drainage/climate feature. | Important falsification of visual-impact assumptions; overlaps climate and erosion architecture. Verify toggle/code still exists. | **Retain in experiment archive**; summarize current optional mechanism canonically. |
| `docs/specs/staging.md` | Stage navigation, steppable erosion, caching and invariants; records several DONE items and deferred scrubber. | Root guides disagree on stage count/navigation. Verify the stage model, keys, cache and current naming. | **Rewrite/extract** into canonical runtime pipeline and tooling docs; archive implementation log. |
| `docs/specs/relief-spectrum-redesign.md` | Large instrument-first mountain redesign log: draft v2 followed by implemented candidates, gates and addenda. | Internal status is contradictory: header says “no code yet,” later sections say implementations landed. Overlaps meso/A4, terrain reset and audits. | **Archive as experiment notebook**; extract current accepted/rejected mechanisms into an experiment registry. |
| `docs/specs/meso-design-consult-gpt56.md` | Verbatim external-model consultation on 10–50 km mountain grammar, with warning to verify claims. | Research memo, not repository truth; links and scientific comparisons need source verification before load-bearing use. | **Archive under research**, not specs; synthesize only verified findings. |
| `docs/specs/meso-a4-drainage-pulse.md` | Implementation hand-off for a two-stage drainage-aware uplift pulse; implemented/gate-validated, user visual pending. | Closely tied to relief redesign, but `terrain-reset.md` may park this mechanism. Verify current default/flag and whether visual review occurred. | **Archive as experiment record**. |
| `docs/specs/terrain-reset.md` | Recent product-baseline reset and physical ownership rule; parks accumulated mechanisms and makes geological time the next redesign. | One of the freshest current-state sources and aligns with the thesis, but is still terrain-specific and status-heavy. Verify exact default/parked flags. | **Retain for now**, then merge its durable ownership decisions into canonical architecture and archive the reset record. |
| `docs/specs/tectonic-time.md` | Extensive implementation ladder for boundary history/geological clocks; multiple implemented, falsified and unpromoted rungs; surface coupling pending. | Overlaps scorecard and thin-sheet docs. Header is current-looking but long appended history obscures product state. Verify modes/defaults and unresolved work. | **Split**: concise current experiment/status doc plus archived research/decision log. |
| `docs/specs/thin-sheet-orogeny.md` | Thin-sheet research prototype; width-aware yield tested/parked, prototype not promoted. | Direct companion to tectonic-time and scorecard; contains corrected measurement history. Verify experimental plumbing still exists. | **Archive as active research record** unless work resumes; link from experiment registry. |

## Major contradictions and terminology drift

These are documentation contradictions, not yet judgments about which code path is
correct.

1. **Project goal.** `docs/roadmap.md` defines the goal as good-looking mountains;
   `docs/physically-inspired-roadmap.md` and `docs/ideas.md` use “unreasonably
   physically inspired”; `docs/thesis.md` defines a broader procedural planet whose
   physical grounding serves causal authenticity, emergence, visual appeal and
   spectacle. The thesis should become authoritative; mountain quality is a major
   workstream, not the whole project.
2. **Realism vocabulary.** Existing documents alternate among “realistic,” “physical,”
   “physically inspired,” “Earth-plausible,” “authentic,” “painted,” and “hack” without
   stable definitions. `docs/thesis.md` introduces the proposed fidelity ladder
   (simulation-grade, physically based, authentic hack, visual hack) and should govern
   the replacements.
3. **Authentic means two things.** Older philosophy uses authenticity for causal model
   behavior; `docs/presentation.md` names a 25× relief preset `Authentic`. That name can
   misleadingly suggest physical scale. Consider “Cartographic” or always qualify it as
   “authentic presentation.”
4. **Pipeline stages.** `AGENTS.md` documents stages 1–3 ending at hydrosphere;
   `CLAUDE.md` and `docs/specs/staging.md` include stage 4 erosion and distinguish
   viewed versus computed stages, fine base and fine surface. A canonical stage/state
   model is required.
5. **Hydrology versus erosion ownership.** “Hydrosphere,” “hydrology,” “drainage,”
   “rivers,” “fine hydrology,” and erosion rerouting are used at different granularities.
   Some specs call drainage an input to erosion, others describe post-erosion hydrology,
   and staging invalidates coarse hydrology when fine data appears. Replacement docs
   need explicit state transitions and ownership.
6. **Tectonics versus terrain.** Root context describes direct feature fields producing
   elevation. Later documents variously assign global structure to tectonics, direct
   uplift to an erosion epoch, height to thickness/isostasy, or geological-time carrier
   models. `terrain-reset.md` establishes a temporary baseline, while
   `tectonic-time.md` explicitly says the physical-clock replacement is unpromoted.
7. **Spec status.** “Spec” currently covers proposal, completed implementation, active
   roadmap, rejected hypothesis and research transcript. `relief-spectrum-redesign.md`
   is the sharpest example: a draft/no-code header followed by implementation results.
8. **Baseline/default/landed/promoted.** These terms are not interchangeable. A mechanism
   may be implemented (`landed`), available behind a flag, numerically gate-valid,
   visually unreviewed, parked neutral, or product default. The new docs should track
   these as separate fields.
9. **Physical units and time.** Presentation now insists on physical elevation units,
   while `tectonic-time.md` says several core clocks and erosion maturity remain
   dimensionless. “Physically based” must therefore be assessed per subsystem and
   quantity, not asserted for the whole pipeline.
10. **Current versus generated evidence.** Seed reports, scorecards and sweeps quote
    exact numbers but usually lack a common metadata envelope (commit, constants/config,
    backend, mesh resolution, relief preset where relevant, and command). Results cannot
    safely be compared without one.

## Structural recommendations for the replacement set

These are inventory-derived, still provisional:

- Keep `docs/thesis.md` normative and implementation-light.
- Create one human entry point (`README.md`) and one concise contributor/assistant policy;
  avoid mirroring architecture in both `AGENTS.md` and `CLAUDE.md`.
- Separate **current architecture**, **stage/state pipeline**, **presentation contract**,
  **roadmap/gaps**, **system fidelity/Pareto assessment**, and **validation policy**.
- Introduce an experiment registry with explicit states: proposed, implemented behind a
  flag, numerically evaluated, visually evaluated, promoted, parked, falsified, removed.
- Move chronological audits, generated analyses, consultations and superseded specs into
  clear `archive/`, `audits/`, or `research/` locations only after the new canonical docs
  preserve their still-relevant conclusions.
- Give every future measurement report a reproducibility header: revision, seed(s), CLI,
  constants/config hash, backend, coarse/fine resolutions, stage, physical units, and
  presentation preset/camera when images are evidence.
