# Planet dossiers

Planet dossiers connect the project thesis to actual generated worlds. Each
dossier examines geographic objects, their claimed causes, their presentation
and the mechanisms that own them. They are cross-system evaluations, not
promotion scorecards or collections of attractive screenshots.

A dossier may remain preliminary when authoritative object extraction or
matched captures do not yet exist. Missing evidence is recorded rather than
replaced by aggregate metrics.

Current dossiers:

- [Seed 12345](seed-12345.md) — reference/high-relief world; first vertical
  slice establishing the method and identifying the next evidence packet.

CPU spatial packets are generated with the `dossier` binary. Their JSON
artifacts contain exact run manifests, selected-object coordinates and the
schema-v2 aggregate water-geography report: ocean/land components, shoreline,
basin/lake states, independent river roles, drainage-repair footprint and
consistency checks. Artifacts are ignored by Git; maintained interpretation
belongs in the dossier documents.
Matched relief captures use the Windows sweep renderer with repeated
`--sweep-target id:lat:lon` arguments and emit a `capture.json` sidecar.
