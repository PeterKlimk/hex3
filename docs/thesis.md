# Hex3 Project Thesis

Hex3 is a procedural planet system built to create worlds that are visually
compelling because they are grounded in coherent planetary processes.

The maintained implementation strategy for this thesis is the
[hybrid causal world-model decision](model-strategy.md). It states the truth
promised by each domain and prevents either maximal simulation or local metric
optimization from becoming the project goal by default.

It combines aspects of a globe simulator, a cartographic model, a strategy-game
world, and a real-time planet demo. Its target is not scientific prediction or
an exact reconstruction of Earth. Its target is a world whose large-scale form,
local detail, and visible relationships feel causally authentic: mountains,
climate, rivers, lakes, coasts, and future systems should arise from one another
in ways that are plausible, legible, and capable of producing surprises.

The desired response is both:

> I can understand why this world looks like this.

and:

> Wow, that's cool.

## Core philosophy: physical grounding to spectacle

Physical grounding is the generative substrate. Spectacle is a disciplined act
of interpretation.

Hex3 should preserve coherent physical state and causal relationships, then use
semantic interpretation and cartographic presentation to make their important
consequences visible. It should not corrupt the world model merely because real
mountains, rivers, or other features are too small to read at planetary scale.

This leads to three distinct architectural layers:

1. **World model** — internally consistent state and processes: geometry,
   elevation, crust, motion, climate, water, erosion, and their successors.
2. **Semantic model** — derived meaning: ranges, divides, river hierarchy,
   basins, margins, climate regions, and other features worth reasoning about
   or emphasizing.
3. **Presentation model** — visual communication: relief exaggeration, river
   width, color, lighting, visibility, symbols, and scale-dependent detail.

Presentation may be dramatic. It must remain explicit, inspectable, and unable
to feed back into simulation or validation unless that feedback is itself a
deliberately modeled process.

## Plausible causality, not maximal simulation

Hex3 is not trying to maximize the number of equations solved. It is trying to
maximize the coherence, emergence, and visible payoff obtained from its compute
and implementation budget.

A system earns its place when it contributes one or more of the following:

- important visible structure;
- meaningful downstream constraints or interactions;
- emergent variety that would be difficult to author directly;
- explanatory power: a feature exists for a comprehensible reason;
- reusable state that enables several later systems;
- diagnostic or interactive value.

Scientific sophistication without a worthwhile consequence is not inherently
valuable here. Conversely, a cheap approximation can be excellent when it
preserves the important causes, correlations, constraints, and distributions.
Compute should be spent where deeper simulation creates valuable emergence or
where many downstream systems depend on the result.

## Fidelity vocabulary

Every substantial system should state what kind of fidelity it claims. These
categories are tools, not a ranking in which everything must move upward.

- **Simulation-grade** — resolves enough of a governing system to support
  meaningful quantitative claims within a stated domain and error budget.
- **Physically based** — simplifies the governing system while retaining
  physically meaningful state, constraints, and causal relationships.
- **Authentic hack** — an engineered shortcut that reproduces the important
  consequences and correlations of a real process without simulating it fully.
- **Visual hack** — a presentation-only technique used for legibility, beauty,
  or emphasis; it makes no claim about physical state.
- **Unjustified hack** — a shortcut that hides a modeling failure, creates
  downstream contradictions, or has no clear benefit commensurate with its
  cost. These should be exposed, replaced, or explicitly accepted as debt.

Most of Hex3 should live in the physically based and authentic-hack categories.
Simulation-grade work is appropriate where its emergence or downstream value
justifies the cost. Visual hacks are welcome when they remain in the
presentation layer.

## Authenticity contract

The project uses the following contract when models and visuals disagree:

> Physical fields remain internally coherent and presentation-independent.
> Features may be semantically amplified and visually exaggerated, but every
> exaggeration is explicit, inspectable, and excluded from physical validation.

In particular:

- physical quantities should use defined units and scales wherever practical;
- generated state should be evaluated numerically and structurally, not inferred
  from an exaggerated render;
- visual comparisons must identify their presentation settings;
- true-scale and diagnostic views must remain available;
- rendering controls may change what is legible, never what physically exists;
- a surprising feature should be investigated across all three layers before
  the generating model is declared wrong.

Relief exaggeration and widened rivers are therefore legitimate cartography.
They become defects only when they are hidden, mistaken for modeled geometry,
or used as evidence to tune the physical system.

## Emergence and authored structure

Hex3 prefers systems that interact over independent layers that merely coexist.
Tectonics should influence elevation; elevation should organize circulation and
drainage; climate should affect water and erosion; erosion should respond to
geology and reshape later outcomes. The most valuable additions often connect
state that already exists rather than introduce another isolated field.

Randomness is an input to structure, not a substitute for structure. Noise is
useful for unresolved variation, irregularity, and texture when a coherent
process determines the feature's location, orientation, scale, or behavior.

Authored constraints and playful parameters are also legitimate. The test is
whether they steer a coherent system toward interesting worlds rather than
paint the desired answer after the fact.

## Multiple honest views of one world

The project need not choose a single visual interpretation. A useful set of
views may include:

- **physical** — true-scale geometry and quantities for inspection;
- **diagnostic** — fields, units, slopes, topology, and system boundaries;
- **cartographic** — the default legible globe or board-like world;
- **dramatic** — an optional showcase presentation that prioritizes spectacle.

Likewise, an actual mini-planet or a more categorical game world may be a valid
world configuration, but it should be modeled as such. Changing the fictional
scale of the planet should be a deliberate premise with coherent consequences,
not an invisible workaround for a rendering problem.

## Decision principles

When proposing, reviewing, or replacing a system, ask:

1. What visible or systemic outcome is it responsible for?
2. Which later systems consume its output?
3. What fidelity does it claim, and where does it intentionally depart from
   reality?
4. Does a cheaper model preserve the consequences that matter?
5. Does a deeper model unlock worthwhile emergence or several downstream uses?
6. Is the result sensitive, controllable, diagnosable, and resolution-aware?
7. Is a problem in the world model, semantic interpretation, or presentation?
8. What is the Pareto case for building this instead of another missing
   interaction or system?

Complexity is justified by benefit, not by realism alone. Simplification is
justified by preserved behavior, not by convenience alone.

## The architecture is provisional

Hex3 has no obligation to preserve a current system merely because it is
implemented, physically motivated, or expensive to replace. Every model,
stage boundary, representation and default remains open to criticism,
simplification, replacement or fundamental redesign.

Current architecture is evidence about where the project is, not a constraint
on where it may go. Compatibility and sunk implementation cost are practical
considerations, but they do not outrank causal coherence, emergence, visual
payoff or a clearly better compute/benefit trade.

Likewise, the current pipeline is incomplete by design. Later systems may build
new meaning and behavior from terrain, climate and water: oceans and ice,
sediment and soils, biomes, vegetation, ecology, resources, culture,
settlement and civilization are all plausible horizons. They should be added
because existing world state gives them meaningful causes and because their
outputs create new visible or interactive consequences—not merely to complete
a checklist of planet subsystems.

## Documentation and validation

Architecture documentation should describe the actual system rather than its
aspirations. Each major system should eventually record:

- purpose and visible payoff;
- inputs, outputs, ownership, and downstream consumers;
- implementation and stage status;
- fidelity category and physical basis;
- deliberate approximations and presentation treatments;
- compute and complexity cost;
- validations, failure modes, and diagnostic views;
- opportunities to retain, deepen, simplify, replace, connect, or omit it.

Roadmaps should distinguish missing foundations, missing couplings, fidelity
improvements, presentation work, and speculative ideas. Earth measurements and
scientific models are references and grounding evidence, not automatic targets.
Validation should combine quantitative structure, controlled comparisons, and
human visual judgment while avoiding the pretense that any one metric defines a
good world.

## North star

Hex3 succeeds when it produces distinctive, beautiful planets whose geography
feels related rather than assembled; when closer inspection reveals coherent
causes rather than arbitrary decoration; and when its simplifications buy more
emergence, clarity, and delight than their cost in authenticity.

The goal is not merely to imitate a planet. It is to build a system from which
convincing planets can happen.
