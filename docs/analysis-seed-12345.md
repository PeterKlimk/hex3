# Hex3 World Analysis

## Metadata

| Property | Value |
|----------|-------|
| Seed | 12345 |
| Stage | 3 |
| Cells | 100,000 |
| Plates | 14 (1 continental, 13 oceanic) |
| Mean neighbor distance | 0.0126 rad |
| Mean cell area | 0.000126 sr |

## Coverage

| Type | Percentage |
|------|------------|
| Land (elevation >= 0) | 16.8% |
| Ocean (elevation < 0) | 83.2% |
| Continental crust | 29.3% |

## Islands (Oceanic Land)

| Property | Value |
|----------|-------|
| Coverage | 0.92% |
| Cell count | 921 |
| Elevation min | 0.000 |
| Elevation max | 0.141 |
| Elevation mean | 0.045 |

## Elevation

| Statistic | Value |
|-----------|-------|
| Minimum | -0.587 |
| Maximum | 0.313 |
| Mean (area-weighted) | -0.221 |
| Land mean | 0.026 |
| Ocean mean | -0.271 |

### Elevation Percentiles (area-weighted)

| Percentile | Elevation |
|------------|-----------|
| p10 | -0.396 |
| p25 | -0.355 |
| p50 | -0.297 |
| p75 | -0.020 |
| p90 | 0.017 |

### Elevation Histogram (area-weighted)

| Elevation Range | Ocean % | Land % | Total % |
|-----------------|---------|--------|---------|
| -0.59 to -0.54 | 0.1% | 0.0% | 0.1% |
| -0.54 to -0.50 | 0.3% | 0.0% | 0.3% |
| -0.50 to -0.45 | 3.3% | 0.0% | 3.3% |
| -0.45 to -0.41 | 4.5% | 0.0% | 4.5% |
| -0.41 to -0.36 | 13.6% | 0.0% | 13.6% |
| -0.36 to -0.32 | 20.5% | 0.0% | 20.5% |
| -0.32 to -0.27 | 14.1% | 0.0% | 14.1% |
| -0.27 to -0.23 | 4.7% | 0.0% | 4.7% |
| -0.23 to -0.18 | 1.9% | 0.0% | 1.9% |
| -0.18 to -0.14 | 1.9% | 0.0% | 1.9% |
| -0.14 to -0.09 | 2.3% | 0.0% | 2.3% |
| -0.09 to -0.05 | 3.1% | 0.0% | 3.1% |
| -0.05 to -0.00 | 12.0% | 0.0% | 12.0% |
| -0.00 to 0.04 | 1.0% | 14.7% | 15.7% |
| 0.04 to 0.09 | 0.0% | 1.6% | 1.6% |
| 0.09 to 0.13 | 0.0% | 0.2% | 0.2% |
| 0.13 to 0.18 | 0.0% | 0.1% | 0.1% |

### Hypsometric Curve (cumulative area vs elevation)

| Cumulative Area | Elevation |
|-----------------|-----------|
| 5% | 0.032 |
| 10% | 0.017 |
| 15% | 0.004 |
| 20% | -0.007 |
| 25% | -0.020 |
| 30% | -0.049 |
| 35% | -0.133 |
| 40% | -0.243 |
| 45% | -0.279 |
| 50% | -0.297 |
| 55% | -0.311 |
| 60% | -0.322 |
| 65% | -0.332 |
| 70% | -0.343 |
| 75% | -0.355 |
| 80% | -0.366 |
| 85% | -0.379 |
| 90% | -0.396 |
| 95% | -0.433 |
| 100% | -0.587 |

## Tectonic Features

| Feature | Max | Mean (non-zero) | Affected Area |
|---------|-----|-----------------|---------------|
| Trench | 0.161 | 0.026 | 7.8% |
| Arc | 0.400 | 0.137 | 13.9% |
| Ridge | 0.017 | 0.006 | 4.7% |
| Collision | 0.000 | 0.000 | 0.0% |
| Activity | 1.000 | 0.568 | 41.9% |

## Oceanic Ridge Distance (Thermal Subsidence Driver)

| Property | Value |
|----------|-------|
| Oceanic area with finite ridge distance | 93.2% |
| Oceanic area with no ridge on plate | 6.8% |
| Oceanic plates with ridges | 11 / 13 |
| Mean ridge distance (finite only) | 0.417 rad |
| Ridge distance p10 (finite only) | 0.063 rad |
| Ridge distance p25 (finite only) | 0.163 rad |
| Ridge distance p50 (finite only) | 0.354 rad |
| Ridge distance p75 (finite only) | 0.611 rad |
| Ridge distance p90 (finite only) | 0.868 rad |
| Max ridge distance (finite only) | 1.601 rad |

## Hydrology

| Property | Value |
|----------|-------|
| Lake coverage | 0.21% |
| Lake cells | 210 |

## Plates

| Property | Value |
|----------|-------|
| Largest plate | 29,261 cells |
| Smallest plate | 1,676 cells |
