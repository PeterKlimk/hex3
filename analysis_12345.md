# Hex3 World Analysis

## Metadata

| Property | Value |
|----------|-------|
| Seed | 12345 |
| Stage | 2 |
| Cells | 80,000 |
| Plates | 14 (6 continental, 8 oceanic) |
| Mean neighbor distance | 0.0140 rad |
| Mean cell area | 0.000157 sr |

## Coverage

| Type | Percentage |
|------|------------|
| Land (elevation >= 0) | 25.0% |
| Ocean (elevation < 0) | 75.0% |
| Continental crust | 28.5% |

## Islands (Oceanic Land)

| Property | Value |
|----------|-------|
| Coverage | 0.44% |
| Cell count | 352 |
| Elevation min | 0.000 |
| Elevation max | 0.146 |
| Elevation mean | 0.050 |

## Elevation

| Statistic | Value |
|-----------|-------|
| Minimum | -0.527 |
| Maximum | 0.418 |
| Mean (area-weighted) | -0.227 |
| Land mean | 0.058 |
| Ocean mean | -0.322 |

### Elevation Percentiles (area-weighted)

| Percentile | Elevation |
|------------|-----------|
| p10 | -0.421 |
| p25 | -0.367 |
| p50 | -0.320 |
| p75 | -0.000 |
| p90 | 0.054 |

### Elevation Histogram (area-weighted)

| Elevation Range | Ocean % | Land % | Total % |
|-----------------|---------|--------|---------|
| -0.53 to -0.48 | 0.1% | 0.0% | 0.1% |
| -0.48 to -0.43 | 7.5% | 0.0% | 7.5% |
| -0.43 to -0.39 | 10.4% | 0.0% | 10.4% |
| -0.39 to -0.34 | 23.5% | 0.0% | 23.5% |
| -0.34 to -0.29 | 17.9% | 0.0% | 17.9% |
| -0.29 to -0.24 | 4.8% | 0.0% | 4.8% |
| -0.24 to -0.20 | 1.6% | 0.0% | 1.6% |
| -0.20 to -0.15 | 1.5% | 0.0% | 1.5% |
| -0.15 to -0.10 | 1.5% | 0.0% | 1.5% |
| -0.10 to -0.05 | 1.6% | 0.0% | 1.6% |
| -0.05 to -0.01 | 3.3% | 0.0% | 3.3% |
| -0.01 to 0.04 | 1.2% | 10.9% | 12.1% |
| 0.04 to 0.09 | 0.0% | 9.4% | 9.4% |
| 0.09 to 0.13 | 0.0% | 2.6% | 2.6% |
| 0.13 to 0.18 | 0.0% | 1.1% | 1.1% |
| 0.18 to 0.23 | 0.0% | 0.7% | 0.7% |
| 0.23 to 0.28 | 0.0% | 0.3% | 0.3% |

### Hypsometric Curve (cumulative area vs elevation)

| Cumulative Area | Elevation |
|-----------------|-----------|
| 5% | 0.084 |
| 10% | 0.054 |
| 15% | 0.037 |
| 20% | 0.019 |
| 25% | -0.000 |
| 30% | -0.070 |
| 35% | -0.220 |
| 40% | -0.288 |
| 45% | -0.308 |
| 50% | -0.320 |
| 55% | -0.331 |
| 60% | -0.341 |
| 65% | -0.349 |
| 70% | -0.357 |
| 75% | -0.367 |
| 80% | -0.379 |
| 85% | -0.396 |
| 90% | -0.421 |
| 95% | -0.445 |
| 100% | -0.527 |

## Tectonic Features

| Feature | Max | Mean (non-zero) | Affected Area |
|---------|-----|-----------------|---------------|
| Trench | 0.092 | 0.015 | 5.3% |
| Arc | 0.400 | 0.122 | 12.6% |
| Ridge | 0.016 | 0.005 | 4.3% |
| Collision | 0.250 | 0.077 | 3.1% |
| Activity | 1.000 | 0.030 | 23.9% |

## Oceanic Ridge Distance (Thermal Subsidence Driver)

| Property | Value |
|----------|-------|
| Oceanic area with finite ridge distance | 87.5% |
| Oceanic area with no ridge on plate | 12.5% |
| Oceanic plates with ridges | 7 / 8 |
| Mean ridge distance (finite only) | 0.381 rad |
| Ridge distance p10 (finite only) | 0.060 rad |
| Ridge distance p25 (finite only) | 0.150 rad |
| Ridge distance p50 (finite only) | 0.310 rad |
| Ridge distance p75 (finite only) | 0.539 rad |
| Ridge distance p90 (finite only) | 0.828 rad |
| Max ridge distance (finite only) | 1.378 rad |

## Hydrology

| Property | Value |
|----------|-------|
| Lake coverage | 5.07% |
| Lake cells | 4,058 |

## Plates

| Property | Value |
|----------|-------|
| Largest plate | 18,068 cells |
| Smallest plate | 1 cells |
