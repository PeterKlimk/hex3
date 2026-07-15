# Common planar evidence-core V0 audit

**Date:** 2026-07-15

**Verdict:** accepted as the bounded planar artifact-boundary checkpoint

**Contract:** [common planar evidence-core V0](../research/landform-common-core-v0-2026-07-15.md)

**Evaluated state:** dirty implementation worktree based on preregistration
commit `9c73dc4`; this audit and implementation are committed together after the
recorded gates pass

## What passed

The implementation adds a separately hashed common S0/D0 core, reference O0a
sidecar, ten-run sensitivity suite and core-backed O0b artifact. Frozen
`LandformObjectPacketCoreV0` and `ObjectCorrespondenceV0` remain unchanged.

The exact compatibility matrix passes:

- asymmetric-Y and isolated-four-cone V0 packets at 8/4/2 split, independently
  validate and materialize to exact original Rust values, bytes and hashes;
- valid linked-four-cone 4/8 packets do the same, while the historical 2 km D0
  ambiguity remains untouched;
- old and new O0b mechanical fields and work counts are bit-identical for
  asymmetric-Y and isolated-four-cone 4↔8 and 4↔2, plus linked-four-cone 4↔8;
- semantic artifact rejection and reversal pass through the new core boundary;
  the equal-elder and five-cell-remapping synthetic kernel gates remain passing
  shared-kernel evidence;
- wrong-core, namespace/configuration/order, predecessor, repaired-hash
  semantic-mutation, trailing-byte and finite retained-field mutation witnesses
  reject; and
- the registered asymmetric 4→8 and isolated 4→2 repeat builds are exact by
  value, bytes and hash.

The old and new O0b paths share one mechanical builder over a borrowed core
view. This makes equivalence structural rather than two implementations that
happen to agree at this checkpoint. The new schema and bytes do not cross-decode
as frozen V0; numeric 64-bit hash inequality is not used as an identity gate.

## Commands and results

All commands ran in WSL2 Linux `6.6.87.2-microsoft-standard-WSL2` on an AMD
Ryzen 5 3600 (6 cores/12 threads), with 10 GiB visible RAM. Release compilation
was complete before the timed cost command.

| command/gate | result |
|---|---|
| `cargo check --all-targets` | pass |
| `cargo test` | pass, 351 tests across library/binaries/integration plus 24 ignored audit/doc tests |
| `cargo clippy --all-targets` | pass with pre-existing warnings; no new production warning in this slice |
| bounded common-core shape/mutation tests | 3 pass |
| bounded core-backed O0b equivalence/rejection | pass |
| exact V0 decomposition matrix, release | pass, 466.80 s |
| old/new O0b direction and core-backed reversal matrix, release | pass, 321.34 s |
| deterministic repeat matrix, release | pass, 230.85 s |
| focused release cost audit under `/usr/bin/time -v` | pass, 157.27 s test / 163.76 s process |
| whole-process peak RSS | 726,804 KiB |

The focused audit stays below the preregistered ten-minute and 2 GiB ceilings.
The broad matrices are intentionally ignored tests because they rebuild exact
historical O0a payloads and are checkpoint evidence rather than routine unit
cost.

## Retained artifact cost

All byte counts are exact fixed-encoding serialized sizes. Correspondence is
from the frozen 4 km source to the listed isolated-four-cone target.

| spacing | cells | common core | reference O0a | ten sensitivities | materialized V0 | old/new O0b |
|---|---:|---:|---:|---:|---:|---:|
| 8 km | 5,400 | 4,882,740 | 1,411,401 | 14,454,292 | 20,749,641 | 117,888 / 117,898 |
| 4 km | 21,600 | 16,815,455 | 1,372,078 | 14,012,432 | 32,201,173 | 41,252 / 41,262 |
| 2 km | 86,400 | 63,446,482 | 2,489,906 | 25,175,792 | 91,113,388 | 54,840 / 54,850 |

The exact V0 identity is preserved. A consumer retaining common core plus
reference O0a keeps about 6.29/18.19/65.94 MB at 8/4/2 rather than
20.75/32.20/91.11 MB. This is a useful separation of optional audit evidence,
not a complete scaling solution.

## What dominates the core

| field/group | 8 km | 4 km | 2 km |
|---|---:|---:|---:|
| graph | 3,334,600 | 13,321,696 | 53,254,288 |
| physical elevation | 43,208 | 172,808 | 691,208 |
| scored mask | 5,408 | 21,608 | 86,408 |
| local runoff | 43,208 | 172,808 | 691,208 |
| S0 evidence | 16,188 | 53,980 | 201,924 |
| D0 evidence | 1,439,595 | 3,072,022 | 8,520,913 |

The graph grows from about 68.3% of the core at 8 km to 83.9% at 2 km. D0 is
the second substantive owner but falls from about 29.5% to 13.4%. The physical
arrays and S0 are comparatively small. Separating O0a sensitivities was the
right dependency correction; further O0a factorization is not the next Pareto
move. If real evaluations establish that core retention is too costly, inspect
reconstructible/shared geometry ownership before trimming causal arrays or
evidence.

## Timing interpretation

At 8/4/2, exact packet fixture assembly took 1.51/5.12/26.06 s; split took
1.81/6.31/32.21 s; full core-plus-sidecar validation took 1.02/3.54/18.21 s;
and exact V0 materialization took 1.73/5.85/28.46 s. These paths deliberately
rebuild predecessors. They are archive/audit operations, not a demonstrated
per-frame or interactive interchange cost.

Core-backed O0b took 0.90/1.40/4.67 s versus 0.38/0.49/1.47 s for the frozen V0
entry point in this audit because the new entry point fully validates the two
cores before building the shared mechanical result. Do not optimize away that
trust boundary on manufactured evidence. Repeated real comparison runs may
justify a separately preregistered already-validated session API later.

## Boundary and next step

This acceptance validates an evaluation artifact boundary. It does not validate
a linked terrain, choose H/C/G, add partial scored support, promote natural-kind
landform names or adapt product hydrology.

The next checkpoint is the linked shared-input manifest. It must bind the exact
mesh/phase, declarative and compiled deformation field, schedule and integrated
work, initial state, portals, runoff, material mask, scoring decision and
resource context. The organization comparison preregistration—not that
manifest—must resolve whole-graph V0 versus a new partial-scored population
identity and own the arm-neutral opportunity calibration.
