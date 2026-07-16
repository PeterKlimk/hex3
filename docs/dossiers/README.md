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
artifacts contain exact run manifests, selected-object coordinates, the
schema-v2 water-geography relationship graph and the schema-v3 diagnostic
conditional-climatology comparison within dossier schema v4. The comparison
holds supplied terrain and temperature
fixed, preserves total land runoff and reports the water geography induced by
the simpler latitude/elevation/ocean-distance projection. Artifacts are ignored
by Git; maintained interpretation belongs in the dossier documents.

Ordinary packets omit full fields. For a bounded spatial review, opt in to the
nested climatology schema-v3 arrays and render the paired map directly:

```bash
cargo run --release --bin dossier -- \
  --seed 12345 --cells 100000 --fine-max 1000000 \
  --include-climatology-spatial --output artifacts/dossiers/seed-12345.json
python3 scripts/render_climatology_null.py \
  artifacts/dossiers/seed-12345.json
```

The opt-in JSON is intentionally large and should be limited to predeclared
worlds; it is evaluation evidence, not persistent product state.

Matched relief captures use the Windows sweep renderer with repeated
`--sweep-target id:lat:lon` arguments and emit a `capture.json` sidecar.
