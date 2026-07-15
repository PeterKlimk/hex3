# Linked orogen shared-input V0 audit

**Date:** 2026-07-15

**Verdict:** accepted as the arm-neutral linked-case input identity

**Contract:** [linked orogen shared-input manifest V0](../research/orogen-linked-shared-input-v0-2026-07-15.md)

**Evaluated state:** dirty implementation worktree based on preregistration
commit `d7518af`; this audit and implementation are committed together after the
recorded gates pass

## What passed

The implementation materializes one immutable 8/4/2 km bundle containing the
exact mesh and portal records, declarative scenario, compiled support, forcing
witnesses, integrated work, raw initial elevation, local runoff, homogeneous
base-material membership and whole/central candidate masks. The frozen semantic
bundle hash is:

```text
0d6a4ab7aec24e68
```

The implementation also adds:

- one shared authority for the coordinate-defined parabolic initial surface,
  consumed by both the bundle builder and `c0_orogen_smoke`;
- a read-only export of the existing forcing compiler's support stencils rather
  than a second compiler;
- an exact same-platform replay decoder and a bounded stored-input decoder that
  validates the artifact without claiming cross-platform transcendental replay;
- deterministic binary and JSON projections with component, resolution and
  bundle hashes; and
- cooperatively collision-safe sibling publication between materializer
  invocations, with file and directory synchronization, post-write
  reread/validation and a nonsemantic run envelope. An unrelated writer is not
  covered by the cooperative lock.

The repaired-hash mutation matrix rejects finite one-ULP changes to every
registered declaration/input/witness class even after all affected nested and
outer hashes are recomputed. Canonical resolution and witness order, trailing
bytes, malformed length prefixes, negative zero in registered nonnegative
fields, nonzero horizontal velocity and semantic portal/mesh changes also
reject. Repeated full builds are identical by value, binary bytes, JSON bytes
and hash.

## Materialized identity

| spacing | cells | directed edges | central cells | support cells per segment | resolution hash |
|---:|---:|---:|---:|---:|---|
| 8 km | 11,040 | 65,394 | 3,680 | 635 | `b1c05ea3f174f296` |
| 4 km | 44,400 | 264,702 | 14,880 | 2,543 | `2ca511209aba8284` |
| 2 km | 177,600 | 1,062,202 | 58,880 | 10,181 | `3248b568c9eab45a` |

The generated control-volume areas are
611,898.9092978081/615,224.4468486015/615,224.4468473383 km2. The central
candidate areas are
203,966.30309932513/206,183.3281329458/203,966.30309921884 km2. These are
ordered sums over the actual generated meshes, not nominal rectangle claims.

Each north/south portal covers 958/959/959.5 projected km at 8/4/2 km. The
independent unsplit-face reconstruction confirms the two closed outboard
slivers caused by the staggered full-cell patch; they are not missing portal
records. Every positive-measure in-span face fragment has exactly one portal
owner.

The compiled cumulative work closes to
100,625.00000000012/100,625.00000000007/100,624.99999999958 km3 against the
analytic 100,625 km3 ledger. Both segment stencils area-integrate to one within
the preregistered tolerance. The raw initial surfaces remain low relief and
nonnegative, with 8/4/2 km minima of approximately
0.0004702/0.0001286/0.0000430 km and maxima of approximately
0.024291/0.024362/0.024358 km; portal-owner cells receive no pinning rewrite.

## Artifact and cost

The final registered release invocation wrote exactly:

| file | bytes | FNV-1a-64 |
|---|---:|---|
| `shared-input.bin` | 53,342,979 | `1521102dccb5efb6` |
| `manifest.json` | 11,470 | `77e6188d116cd92f` |
| `run-envelope.json` | 1,586 | nonsemantic/self-excluded |

On WSL2 Linux `6.6.87.2-microsoft-standard-WSL2`, an AMD Ryzen 5 3600 and
`rustc 1.96.0-nightly`, `/usr/bin/time -v` measured 10.91 s wall time and
467,824 KiB whole-process peak RSS. The run envelope recorded one materializer
thread, 12-way available parallelism and 10.596 s to completed prepublication
validation. This passes the preregistered two-minute and 1 GiB limits with a
large margin.

The materialized directory is generated evidence and remains ignored by Git.
Its semantic identity is secured in code by the V0 golden bundle hash; this
audit records the measured projection and resource envelope.

## Commands and results

| command/gate | result |
|---|---|
| `cargo fmt --all` | pass |
| `cargo check --all-targets` | pass |
| `cargo test` | pass, 353 tests across library/binaries/integration; 27 audit/doc tests intentionally ignored |
| three registered linked-input release tests | pass: determinism/replay, repaired mutation, synced publication and collision preservation |
| `cargo clippy --all-targets` | pass with the repository's pre-existing warnings; no warning in the new linked-input implementation |
| `git diff --check` | pass |
| registered release materialization under `/usr/bin/time -v` | pass, 10.91 s and 467,824 KiB peak RSS |

The `serde_json` dependency enables `float_roundtrip`. Without that parser mode,
one valid manifest decimal parsed one ULP away from its binary f64 authority and
created a false exact-replay failure; the projection now round-trips every
registered float exactly.

## Boundary and next step

This acceptance establishes common admissible inputs, not a terrain baseline.
It does not produce a final surface, choose H/C/G, define arm conversion or
chronology, select whole versus central scoring support, extract a landform,
judge a mountain or river, or promote the current forcing compiler as complete
tectonics. In particular, retained vergence and transfer links remain disclosed
metadata that the current compiler does not consume, and horizontal velocity
remains identically zero.

Stop at this boundary. The next document is the organization-owner comparison
preregistration. It must own arm-neutral opportunity calibration, each arm's
conversion and chronology, admission gates, the shared compute ceiling, the
accepted evaluation population, independent final-surface evidence and matched
human presentation before any H/C/G implementation proceeds.
