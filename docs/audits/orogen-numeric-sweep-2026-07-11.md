> **Dated audit evidence:** Results are revision/configuration specific and do not define current defaults.

# Orogen numeric sweep — 2026-07-11

All runs use the legacy orogen model, seed-specific 100,000-cell coarse worlds,
the full generated fine mesh, and the normal erosion defaults. No rendered image
measurements enter this report.

## Plateau metric

For each connected mountain component of at least 20,000 km², the audit measures:

- physical area within 0.5 km and 1.0 km of that component's own summit;
- the area-weighted fraction of its 0.5-km summit cap whose steepest downhill
  neighbor edge is below a 1% grade;
- p50 and p90 of those values across significant components.

This prevents a structured builder from appearing to remove a plateau merely by
fragmenting one massif into several components. Relief scale, lighting, coloring,
and image downscaling do not affect these measurements.

## Interior-relief sweep — seed 12345, uniform builder

`interior_relief` is a pre-erosion, zero-mean fBm height seed. One elevation unit
is approximately 10 km.

| interior relief | summit slope p50 | summit slope p90 | p95-p05 relief 10 km | 25 km | 100 km | maximum summit |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.000747 | 0.004539 | 19 m | 129 m | 564 m | 6.46 km |
| 0.040 | 0.000786 | 0.004437 | 24 m | 136 m | 561 m | 6.46 km |
| 0.080 | 0.000878 | 0.004385 | 30 m | 142 m | 566 m | 6.5 km |
| 0.120 | 0.001037 | 0.004606 | 35 m | 157 m | 575 m | 6.5 km |
| 0.160 | 0.001221 | 0.004645 | 42 m | 176 m | 581 m | 6.49 km |
| 0.240 | 0.001581 | 0.004668 | 50 m | 219 m | 630 m | 6.6 km |
| 0.320 | 0.001933 | 0.004878 | 56 m | 249 m | 683 m | 6.7 km |
| 0.640 | 0.002945 | 0.005969 | 68 m | 332 m | 898 m | 7.0 km |

The historical 0.005–0.04 range is largely erased by erosion. The parameter begins
to own final small-scale sharpness around 0.08–0.16. It does not economically fix
the broad plateau envelope.

## Structured-strength sweep — seed 12345, interior relief 0.005

| structured | summit slope p50 | relief 10 km | 25 km | 100 km | maximum summit | maximum target overshoot |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000747 | 19 m | 129 m | 564 m | 6.46 km | n/a |
| 0.25 | 0.000739 | 20 m | 125 m | 543 m | 6.43 km | 1.12 km |
| 0.50 | 0.000819 | 24 m | 138 m | 571 m | 6.99 km | 1.28 km |
| 0.75 | 0.000979 | 33 m | 157 m | 628 m | 8.03 km | 2.34 km (WARN) |
| 1.00 | 0.001398 | 43 m | 191 m | 744 m | 10.01 km | 4.32 km (FAIL) |

The response is nonlinear. Most plateau reduction arrives above 0.5, together with
rapid peak overshoot. Reducing rebuild gain from 1.2 to 1.0 at full structure lowers
the seed-12345 peak to 8.71 km and summit-slope p50 to 0.001135, but still overshoots
the coarse target by 3.04 km and reduces mountain-land coverage from 8.5% to 8.0%.

## Cross-seed plateau comparison

| seed | path | 0.5-km cap p50 | cap p90 | flat-cap p50 | flat-cap p90 | maximum summit | max target overshoot |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 12345 | uniform + relief 0.16 | 10,257 km² | 24,961 km² | 38.3% | 40.5% | 6.49 km | n/a |
| 12345 | full structured + relief 0.005 | 11,043 km² | 49,032 km² | 28.0% | 92.5% | 10.01 km | 4.32 km (FAIL) |
| 777 | uniform + relief 0.16 | 16,217 km² | 55,974 km² | 42.8% | 100.0% | 6.8 km | n/a |
| 777 | full structured + relief 0.005 | 3,383 km² | 30,789 km² | 7.6% | 100.0% | 8.6 km | 2.86 km (WARN) |
| 4242 | uniform + relief 0.16 | 19,460 km² | 239,028 km² | 30.4% | 94.2% | 6.4 km | n/a |
| 4242 | full structured + relief 0.005 | 17,428 km² | 45,400 km² | 57.8% | 87.9% | 8.7 km | 3.27 km (WARN) |

## Result

- Full structured uplift is genuinely intended to replace a uniformly rebuilt
  dome with asymmetric, segmented uplift, and it often reduces plateau area.
- It is not a general plateau cure: seed 12345 develops a worse p90 cap and a pillar,
  and every tested seed has a very flat upper-tail component.
- Its present normalization systematically couples plateau reduction to peak
  concentration. The tested maximum target overshoot is 2.86–4.32 km.
- Raising `interior_relief` to 0.16 improves local sharpness without materially
  raising peaks, but does not robustly eliminate broad plateaus.
- No tested candidate justifies a default change yet.

The next mechanism should change the structured allocation itself, not add more
render-driven relief: derive the front-normal support width from each coarse orogen
footprint and reduce along-strike concentration, then gate it on per-orogen cap area,
flat-cap fraction, land coverage, and coarse-target overshoot across seeds.
