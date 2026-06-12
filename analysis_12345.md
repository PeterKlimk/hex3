# Hex3 World Analysis

## Metadata

| Property | Value |
|----------|-------|
| Seed | 12345 |
| Stage | 3 |
| Cells | 100,000 |
| Plates | 14 (10 mixed continent/ocean) |
| Mean neighbor distance | 0.0126 rad |
| Mean cell area | 0.000126 sr |

## Coverage

| Type | Percentage |
|------|------------|
| Land (elevation >= 0) | 18.2% |
| Ocean (elevation < 0) | 81.8% |
| Continental crust | 30.0% |

## Islands (Oceanic Land)

| Property | Value |
|----------|-------|
| Coverage | 0.82% |
| Cell count | 818 |
| Elevation min | 0.000 |
| Elevation max | 0.147 |
| Elevation mean | 0.060 |

## Elevation

| Statistic | Value |
|-----------|-------|
| Minimum | -0.586 |
| Maximum | 0.610 |
| Mean (area-weighted) | -0.203 |
| Land mean | 0.056 |
| Ocean mean | -0.261 |

### Elevation Percentiles (area-weighted)

| Percentile | Elevation |
|------------|-----------|
| p10 | -0.391 |
| p25 | -0.340 |
| p50 | -0.279 |
| p75 | -0.020 |
| p90 | 0.026 |

### Elevation Histogram (area-weighted)

| Elevation Range | Ocean % | Land % | Total % |
|-----------------|---------|--------|---------|
| -0.53 to -0.47 | 0.3% | 0.0% | 0.3% |
| -0.47 to -0.41 | 6.4% | 0.0% | 6.4% |
| -0.41 to -0.35 | 15.9% | 0.0% | 15.9% |
| -0.35 to -0.29 | 24.8% | 0.0% | 24.8% |
| -0.29 to -0.23 | 10.8% | 0.0% | 10.8% |
| -0.23 to -0.17 | 3.2% | 0.0% | 3.2% |
| -0.17 to -0.11 | 3.0% | 0.0% | 3.0% |
| -0.11 to -0.05 | 5.0% | 0.0% | 5.0% |
| -0.05 to 0.01 | 12.3% | 3.7% | 15.9% |
| 0.01 to 0.07 | 0.0% | 11.3% | 11.3% |
| 0.07 to 0.13 | 0.0% | 1.4% | 1.4% |
| 0.13 to 0.19 | 0.0% | 0.4% | 0.4% |
| 0.19 to 0.25 | 0.0% | 0.3% | 0.3% |
| 0.25 to 0.31 | 0.0% | 0.4% | 0.4% |
| 0.31 to 0.37 | 0.0% | 0.4% | 0.4% |
| 0.37 to 0.43 | 0.0% | 0.2% | 0.2% |

### Hypsometric Curve (cumulative area vs elevation)

| Cumulative Area | Elevation |
|-----------------|-----------|
| 5% | 0.047 |
| 10% | 0.026 |
| 15% | 0.010 |
| 20% | -0.005 |
| 25% | -0.020 |
| 30% | -0.046 |
| 35% | -0.098 |
| 40% | -0.196 |
| 45% | -0.257 |
| 50% | -0.279 |
| 55% | -0.294 |
| 60% | -0.306 |
| 65% | -0.317 |
| 70% | -0.328 |
| 75% | -0.340 |
| 80% | -0.355 |
| 85% | -0.372 |
| 90% | -0.391 |
| 95% | -0.428 |
| 100% | -0.586 |

## Tectonic Features

| Feature | Max | Mean (non-zero) | Affected Area |
|---------|-----|-----------------|---------------|
| Trench | 0.145 | 0.022 | 6.5% |
| Arc | 0.400 | 0.121 | 12.1% |
| Ridge | 0.017 | 0.006 | 5.8% |
| Collision | 0.279 | 0.100 | 2.6% |
| Activity | 1.000 | 0.548 | 40.8% |

## Oceanic Ridge Distance (Thermal Subsidence Driver)

| Property | Value |
|----------|-------|
| Oceanic area with finite ridge distance | 90.8% |
| Oceanic area with no ridge on plate | 9.2% |
| Oceanic plates with ridges | 12 / 14 |
| Mean ridge distance (finite only) | 0.386 rad |
| Ridge distance p10 (finite only) | 0.046 rad |
| Ridge distance p25 (finite only) | 0.117 rad |
| Ridge distance p50 (finite only) | 0.257 rad |
| Ridge distance p75 (finite only) | 0.537 rad |
| Ridge distance p90 (finite only) | 0.990 rad |
| Max ridge distance (finite only) | 1.685 rad |

## Hydrology

| Property | Value |
|----------|-------|
| Lake coverage | 0.42% |
| Lake cells | 425 |

## Plates

| Property | Value |
|----------|-------|
| Largest plate | 29,261 cells |
| Smallest plate | 1,676 cells |
