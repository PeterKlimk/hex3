#!/usr/bin/env python3
"""
Hex3 Terrain Analysis Script

Generates diagnostic plots from hex3 world export files.

Usage:
    python analyze_terrain.py world_dump.json.gz
    python analyze_terrain.py world_dump.json.gz --show
    python analyze_terrain.py dump1.json.gz dump2.json.gz  # compare multiple
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_world(path: Path) -> dict:
    """Load world data from JSON or gzipped JSON."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def plot_elevation_histogram(data: dict, ax: plt.Axes, label: str = None):
    """Plot elevation histogram, separating land and ocean."""
    elevation = np.array(data["cells"]["elevation"])
    area = np.array(data["cells"]["area"])

    # Separate land (elevation >= 0) and ocean (elevation < 0)
    land_mask = elevation >= 0
    ocean_mask = elevation < 0

    land_elev = elevation[land_mask]
    land_area = area[land_mask]
    ocean_elev = elevation[ocean_mask]
    ocean_area = area[ocean_mask]

    # Create bins
    bins = np.linspace(elevation.min(), elevation.max(), 50)

    # Plot area-weighted histograms
    if len(ocean_elev) > 0:
        ax.hist(ocean_elev, bins=bins, weights=ocean_area, alpha=0.7,
                label=f"Ocean ({100*ocean_area.sum()/area.sum():.1f}%)", color="steelblue")
    if len(land_elev) > 0:
        ax.hist(land_elev, bins=bins, weights=land_area, alpha=0.7,
                label=f"Land ({100*land_area.sum()/area.sum():.1f}%)", color="olivedrab")

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Elevation")
    ax.set_ylabel("Area (steradians)")
    ax.legend()

    title = "Elevation Distribution"
    if label:
        title += f" ({label})"
    ax.set_title(title)


def plot_hypsometric_curve(data: dict, ax: plt.Axes, label: str = None):
    """Plot hypsometric curve (cumulative area vs elevation)."""
    elevation = np.array(data["cells"]["elevation"])
    area = np.array(data["cells"]["area"])

    # Sort by elevation (descending for traditional hypsometric curve)
    sorted_idx = np.argsort(-elevation)
    sorted_elev = elevation[sorted_idx]
    sorted_area = area[sorted_idx]

    # Cumulative area (normalized to 0-100%)
    cumulative_area = np.cumsum(sorted_area) / area.sum() * 100

    ax.plot(cumulative_area, sorted_elev, linewidth=2, label=label)
    ax.axhline(y=0, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.5, label="Sea level")
    ax.set_xlabel("Cumulative Area (%)")
    ax.set_ylabel("Elevation")
    ax.set_title("Hypsometric Curve")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()


def plot_elevation_by_latitude(data: dict, ax: plt.Axes):
    """Plot elevation vs latitude as a 2D histogram."""
    elevation = np.array(data["cells"]["elevation"])
    latitude = np.array(data["cells"]["latitude"])
    area = np.array(data["cells"]["area"])

    # Convert latitude to degrees
    lat_deg = np.degrees(latitude)

    # 2D histogram weighted by area
    h, xedges, yedges = np.histogram2d(
        lat_deg, elevation,
        bins=[50, 50],
        weights=area
    )

    # Plot as image
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    im = ax.imshow(h, extent=extent, origin="lower", aspect="auto", cmap="YlOrBr")
    ax.axvline(x=0, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Elevation")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Elevation by Latitude")
    plt.colorbar(im, ax=ax, label="Area")


def plot_feature_distributions(data: dict, axes: list):
    """Plot histograms of tectonic feature fields."""
    features = data["cells"]["features"]
    area = np.array(data["cells"]["area"])

    feature_names = ["trench", "arc", "ridge", "collision"]
    colors = ["purple", "crimson", "orange", "saddlebrown"]

    for ax, name, color in zip(axes, feature_names, colors):
        values = np.array(features[name])

        # Only plot non-zero values
        mask = values > 0.001
        if mask.sum() > 0:
            ax.hist(values[mask], bins=30, weights=area[mask], color=color, alpha=0.7)
            affected_pct = 100 * area[mask].sum() / area.sum()
            ax.set_title(f"{name.title()} ({affected_pct:.1f}% of area)")
        else:
            ax.set_title(f"{name.title()} (none)")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Area")


def plot_plate_statistics(data: dict, ax: plt.Axes):
    """Plot plate size distribution."""
    plates = data["plates"]

    # Sort by cell count
    plates_sorted = sorted(plates, key=lambda p: p["cell_count"], reverse=True)

    names = [f"P{p['id']}" for p in plates_sorted]
    sizes = [p["cell_count"] for p in plates_sorted]
    colors = ["olivedrab" if p["plate_type"] == "continental" else "steelblue"
              for p in plates_sorted]

    ax.bar(names, sizes, color=colors, alpha=0.7)
    ax.set_xlabel("Plate")
    ax.set_ylabel("Cell Count")
    ax.set_title("Plate Sizes (green=continental, blue=oceanic)")
    ax.tick_params(axis='x', rotation=45)


def plot_noise_distribution(data: dict, ax: plt.Axes):
    """Plot noise contribution distribution."""
    noise = np.array(data["cells"]["noise"]["combined"])
    area = np.array(data["cells"]["area"])

    ax.hist(noise, bins=50, weights=area, color="gray", alpha=0.7)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Noise Contribution")
    ax.set_ylabel("Area")
    ax.set_title(f"Noise Distribution (std={noise.std():.3f})")


def compute_histogram_bins(elevation: np.ndarray, area: np.ndarray, n_bins: int = 20) -> dict:
    """Compute elevation histogram bins for land and ocean."""
    land_mask = elevation >= 0
    ocean_mask = ~land_mask
    total_area = area.sum()

    # Use fixed bins from min to max
    bins = np.linspace(elevation.min(), elevation.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute area-weighted histograms
    land_hist, _ = np.histogram(elevation[land_mask], bins=bins, weights=area[land_mask])
    ocean_hist, _ = np.histogram(elevation[ocean_mask], bins=bins, weights=area[ocean_mask])

    # Normalize to percentage of total area
    land_pct = 100 * land_hist / total_area
    ocean_pct = 100 * ocean_hist / total_area

    return {
        "bin_min": bins[:-1].tolist(),
        "bin_max": bins[1:].tolist(),
        "bin_center": bin_centers.tolist(),
        "land_area_pct": land_pct.tolist(),
        "ocean_area_pct": ocean_pct.tolist(),
    }


def compute_hypsometric_curve(elevation: np.ndarray, area: np.ndarray, n_points: int = 20) -> dict:
    """Compute hypsometric curve data points."""
    # Sort by elevation (descending)
    sorted_idx = np.argsort(-elevation)
    sorted_elev = elevation[sorted_idx]
    sorted_area = area[sorted_idx]

    # Cumulative area (normalized to 0-100%)
    cumulative_area = np.cumsum(sorted_area) / area.sum() * 100

    # Sample at regular intervals
    sample_pcts = np.linspace(0, 100, n_points + 1)[1:]  # Skip 0%
    sample_elevs = []
    for pct in sample_pcts:
        idx = np.searchsorted(cumulative_area, pct)
        idx = min(idx, len(sorted_elev) - 1)
        sample_elevs.append(float(sorted_elev[idx]))

    return {
        "cumulative_area_pct": sample_pcts.tolist(),
        "elevation": sample_elevs,
    }


def compute_summary(data: dict) -> dict:
    """Compute summary statistics and return as a dictionary."""
    meta = data["metadata"]
    cells = data["cells"]

    elevation = np.array(cells["elevation"])
    area = np.array(cells["area"])
    plate_type = np.array(cells["plate_type"])

    land_mask = elevation >= 0
    ocean_mask = ~land_mask
    continental_mask = plate_type == 0

    total_area = area.sum()

    # Basic stats
    summary = {
        "seed": meta["seed"],
        "stage": meta["stage"],
        "num_cells": meta["num_cells"],
        "num_plates": meta["num_plates"],
        "mean_neighbor_dist": meta["mean_neighbor_dist"],
        "mean_cell_area": meta["mean_cell_area"],
    }

    # Histogram and hypsometric data
    summary["histogram"] = compute_histogram_bins(elevation, area)
    summary["hypsometric"] = compute_hypsometric_curve(elevation, area)

    # Coverage
    summary["land_coverage"] = area[land_mask].sum() / total_area
    summary["ocean_coverage"] = area[ocean_mask].sum() / total_area
    summary["continental_coverage"] = area[continental_mask].sum() / total_area

    # Island stats (oceanic crust with elevation >= 0)
    oceanic_mask = plate_type == 1
    island_mask = oceanic_mask & land_mask
    island_area = area[island_mask].sum()
    island_count = island_mask.sum()
    summary["islands"] = {
        "coverage": island_area / total_area,
        "cell_count": int(island_count),
    }
    if island_count > 0:
        island_elev = elevation[island_mask]
        island_weights = area[island_mask]
        summary["islands"]["elevation_mean"] = float(np.average(island_elev, weights=island_weights))
        summary["islands"]["elevation_max"] = float(island_elev.max())
        summary["islands"]["elevation_min"] = float(island_elev.min())

    # Elevation stats
    summary["elevation_min"] = float(elevation.min())
    summary["elevation_max"] = float(elevation.max())
    summary["elevation_mean"] = float(np.average(elevation, weights=area))

    if land_mask.sum() > 0:
        summary["land_elevation_mean"] = float(np.average(elevation[land_mask], weights=area[land_mask]))
    if ocean_mask.sum() > 0:
        summary["ocean_elevation_mean"] = float(np.average(elevation[ocean_mask], weights=area[ocean_mask]))

    # Elevation percentiles (area-weighted approximation via sorting)
    sorted_idx = np.argsort(elevation)
    sorted_elev = elevation[sorted_idx]
    cumulative_area = np.cumsum(area[sorted_idx]) / total_area

    percentiles = [10, 25, 50, 75, 90]
    summary["elevation_percentiles"] = {}
    for p in percentiles:
        idx = np.searchsorted(cumulative_area, p / 100)
        idx = min(idx, len(sorted_elev) - 1)
        summary["elevation_percentiles"][f"p{p}"] = float(sorted_elev[idx])

    # Feature stats
    features = cells["features"]
    summary["features"] = {}
    for name in ["trench", "arc", "ridge", "collision", "activity"]:
        values = np.array(features[name])
        mask = values > 0.001
        affected_area = area[mask].sum() / total_area if mask.sum() > 0 else 0
        summary["features"][name] = {
            "max": float(values.max()),
            "mean_nonzero": float(values[mask].mean()) if mask.sum() > 0 else 0,
            "affected_area_pct": affected_area * 100,
        }

    # Ridge distance stats (if exported)
    ridge_dist = features.get("ridge_distance")
    if ridge_dist is not None:
        ridge_dist = np.array(ridge_dist, dtype=np.float32)
        oceanic_cells = plate_type == 1
        oceanic_ridge_dist = ridge_dist[oceanic_cells]
        oceanic_area = area[oceanic_cells]

        finite_mask = np.isfinite(oceanic_ridge_dist)
        finite_area = oceanic_area[finite_mask].sum() if finite_mask.any() else 0.0
        oceanic_total_area = oceanic_area.sum()

        ridge_distance_summary = {
            "oceanic_has_ridge_area_pct": float(100.0 * finite_area / oceanic_total_area)
            if oceanic_total_area > 0
            else 0.0,
            "oceanic_no_ridge_area_pct": float(100.0 * (1.0 - finite_area / oceanic_total_area))
            if oceanic_total_area > 0
            else 0.0,
        }

        if finite_mask.any():
            finite_dist = oceanic_ridge_dist[finite_mask]
            finite_dist_area = oceanic_area[finite_mask]

            # Area-weighted percentiles for finite ridge-distance.
            # Sort by distance, then pick by cumulative area.
            idx = np.argsort(finite_dist)
            d_sorted = finite_dist[idx]
            a_sorted = finite_dist_area[idx]
            cum = np.cumsum(a_sorted) / a_sorted.sum()
            for p in [10, 25, 50, 75, 90]:
                j = np.searchsorted(cum, p / 100)
                j = min(j, len(d_sorted) - 1)
                ridge_distance_summary[f"oceanic_ridge_distance_p{p}"] = float(d_sorted[j])

            ridge_distance_summary["oceanic_ridge_distance_mean"] = float(
                np.average(finite_dist, weights=finite_dist_area)
            )
            ridge_distance_summary["oceanic_ridge_distance_max"] = float(d_sorted[-1])

        # Count oceanic plates with/without any ridge reachable inside the plate.
        plate_id = np.array(cells["plate_id"])
        oceanic_plate_ids = np.unique(plate_id[oceanic_cells])
        plates_with_ridge = 0
        for pid in oceanic_plate_ids:
            plate_mask = (plate_id == pid) & oceanic_cells
            if np.isfinite(ridge_dist[plate_mask]).any():
                plates_with_ridge += 1
        ridge_distance_summary["oceanic_plates_total"] = int(len(oceanic_plate_ids))
        ridge_distance_summary["oceanic_plates_with_ridge"] = int(plates_with_ridge)
        ridge_distance_summary["oceanic_plates_without_ridge"] = int(
            len(oceanic_plate_ids) - plates_with_ridge
        )

        summary["ridge_distance"] = ridge_distance_summary

    # Hydrology stats
    if cells.get("hydrology"):
        hydro = cells["hydrology"]
        is_lake = np.array(hydro["is_lake"])
        lake_area = area[is_lake].sum() if is_lake.sum() > 0 else 0
        summary["hydrology"] = {
            "lake_coverage": lake_area / total_area,
            "lake_cells": int(is_lake.sum()),
        }

    # Plate stats
    plates = data["plates"]
    continental_plates = [p for p in plates if p["plate_type"] == "continental"]
    oceanic_plates = [p for p in plates if p["plate_type"] == "oceanic"]
    summary["plates"] = {
        "continental_count": len(continental_plates),
        "oceanic_count": len(oceanic_plates),
        "largest_plate_cells": max(p["cell_count"] for p in plates),
        "smallest_plate_cells": min(p["cell_count"] for p in plates),
    }

    return summary


def print_summary(data: dict):
    """Print summary statistics to stdout."""
    s = compute_summary(data)

    print(f"\n{'='*60}")
    print(f"World Summary: seed={s['seed']}, stage={s['stage']}")
    print(f"{'='*60}")
    print(f"Cells: {s['num_cells']:,}")
    print(f"Plates: {s['num_plates']}")
    print(f"Mean neighbor distance: {s['mean_neighbor_dist']:.4f} rad")
    print(f"Mean cell area: {s['mean_cell_area']:.6f} sr")
    print()
    print(f"Land coverage: {100*s['land_coverage']:.1f}%")
    print(f"Continental coverage: {100*s['continental_coverage']:.1f}%")
    print()
    # Island stats
    islands = s.get("islands", {})
    print(f"Islands (oceanic land): {100*islands.get('coverage', 0):.2f}% ({islands.get('cell_count', 0):,} cells)")
    if islands.get("elevation_mean") is not None:
        print(f"  Elevation: [{islands['elevation_min']:.3f}, {islands['elevation_max']:.3f}], mean={islands['elevation_mean']:.3f}")
    print()
    print(f"Elevation range: [{s['elevation_min']:.3f}, {s['elevation_max']:.3f}]")
    print(f"Mean elevation: {s['elevation_mean']:.3f}")
    if "land_elevation_mean" in s:
        print(f"Land mean: {s['land_elevation_mean']:.3f}")
    if "ocean_elevation_mean" in s:
        print(f"Ocean mean: {s['ocean_elevation_mean']:.3f}")

    # Feature stats
    print()
    print("Feature max values:")
    for name in ["trench", "arc", "ridge", "collision", "activity"]:
        print(f"  {name}: {s['features'][name]['max']:.3f}")

    # Hydrology stats
    if "hydrology" in s:
        print()
        print(f"Lake coverage: {100*s['hydrology']['lake_coverage']:.2f}%")
        print(f"Lake cells: {s['hydrology']['lake_cells']:,}")


def write_summary_markdown(data: dict, output_path: Path):
    """Write summary statistics to a markdown file."""
    s = compute_summary(data)

    lines = [
        f"# Hex3 World Analysis",
        f"",
        f"## Metadata",
        f"",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Seed | {s['seed']} |",
        f"| Stage | {s['stage']} |",
        f"| Cells | {s['num_cells']:,} |",
        f"| Plates | {s['num_plates']} ({s['plates']['continental_count']} continental, {s['plates']['oceanic_count']} oceanic) |",
        f"| Mean neighbor distance | {s['mean_neighbor_dist']:.4f} rad |",
        f"| Mean cell area | {s['mean_cell_area']:.6f} sr |",
        f"",
        f"## Coverage",
        f"",
        f"| Type | Percentage |",
        f"|------|------------|",
        f"| Land (elevation >= 0) | {100*s['land_coverage']:.1f}% |",
        f"| Ocean (elevation < 0) | {100*s['ocean_coverage']:.1f}% |",
        f"| Continental crust | {100*s['continental_coverage']:.1f}% |",
        f"",
        f"## Islands (Oceanic Land)",
        f"",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Coverage | {100*s['islands']['coverage']:.2f}% |",
        f"| Cell count | {s['islands']['cell_count']:,} |",
    ]
    if s['islands'].get('elevation_mean') is not None:
        lines.extend([
            f"| Elevation min | {s['islands']['elevation_min']:.3f} |",
            f"| Elevation max | {s['islands']['elevation_max']:.3f} |",
            f"| Elevation mean | {s['islands']['elevation_mean']:.3f} |",
        ])
    lines.extend([
        f"",
        f"## Elevation",
        f"",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Minimum | {s['elevation_min']:.3f} |",
        f"| Maximum | {s['elevation_max']:.3f} |",
        f"| Mean (area-weighted) | {s['elevation_mean']:.3f} |",
    ])

    if "land_elevation_mean" in s:
        lines.append(f"| Land mean | {s['land_elevation_mean']:.3f} |")
    if "ocean_elevation_mean" in s:
        lines.append(f"| Ocean mean | {s['ocean_elevation_mean']:.3f} |")

    lines.extend([
        f"",
        f"### Elevation Percentiles (area-weighted)",
        f"",
        f"| Percentile | Elevation |",
        f"|------------|-----------|",
    ])
    for p, v in s["elevation_percentiles"].items():
        lines.append(f"| {p} | {v:.3f} |")

    # Histogram data
    h = s["histogram"]
    lines.extend([
        f"",
        f"### Elevation Histogram (area-weighted)",
        f"",
        f"| Elevation Range | Ocean % | Land % | Total % |",
        f"|-----------------|---------|--------|---------|",
    ])
    for i in range(len(h["bin_center"])):
        ocean = h["ocean_area_pct"][i]
        land = h["land_area_pct"][i]
        total = ocean + land
        if total > 0.1:  # Only show bins with >0.1% area
            lines.append(f"| {h['bin_min'][i]:.2f} to {h['bin_max'][i]:.2f} | {ocean:.1f}% | {land:.1f}% | {total:.1f}% |")

    # Hypsometric curve data
    hyp = s["hypsometric"]
    lines.extend([
        f"",
        f"### Hypsometric Curve (cumulative area vs elevation)",
        f"",
        f"| Cumulative Area | Elevation |",
        f"|-----------------|-----------|",
    ])
    for i in range(len(hyp["cumulative_area_pct"])):
        lines.append(f"| {hyp['cumulative_area_pct'][i]:.0f}% | {hyp['elevation'][i]:.3f} |")

    lines.extend([
        f"",
        f"## Tectonic Features",
        f"",
        f"| Feature | Max | Mean (non-zero) | Affected Area |",
        f"|---------|-----|-----------------|---------------|",
    ])
    for name in ["trench", "arc", "ridge", "collision", "activity"]:
        f = s["features"][name]
        lines.append(f"| {name.title()} | {f['max']:.3f} | {f['mean_nonzero']:.3f} | {f['affected_area_pct']:.1f}% |")

    if "ridge_distance" in s:
        rd = s["ridge_distance"]
        lines.extend([
            f"",
            f"## Oceanic Ridge Distance (Thermal Subsidence Driver)",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Oceanic area with finite ridge distance | {rd['oceanic_has_ridge_area_pct']:.1f}% |",
            f"| Oceanic area with no ridge on plate | {rd['oceanic_no_ridge_area_pct']:.1f}% |",
            f"| Oceanic plates with ridges | {rd['oceanic_plates_with_ridge']} / {rd['oceanic_plates_total']} |",
        ])
        if "oceanic_ridge_distance_mean" in rd:
            lines.append(f"| Mean ridge distance (finite only) | {rd['oceanic_ridge_distance_mean']:.3f} rad |")
        for p in [10, 25, 50, 75, 90]:
            k = f"oceanic_ridge_distance_p{p}"
            if k in rd:
                lines.append(f"| Ridge distance {k.split('_')[-1]} (finite only) | {rd[k]:.3f} rad |")
        if "oceanic_ridge_distance_max" in rd:
            lines.append(f"| Max ridge distance (finite only) | {rd['oceanic_ridge_distance_max']:.3f} rad |")

    if "hydrology" in s:
        lines.extend([
            f"",
            f"## Hydrology",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Lake coverage | {100*s['hydrology']['lake_coverage']:.2f}% |",
            f"| Lake cells | {s['hydrology']['lake_cells']:,} |",
        ])

    lines.extend([
        f"",
        f"## Plates",
        f"",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Largest plate | {s['plates']['largest_plate_cells']:,} cells |",
        f"| Smallest plate | {s['plates']['smallest_plate_cells']:,} cells |",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {output_path}")


def analyze_single(data: dict, output_dir: Path, show: bool):
    """Generate all plots for a single world."""
    meta = data["metadata"]

    print_summary(data)

    # Write markdown summary
    if output_dir:
        md_path = output_dir / f"analysis_{meta['seed']}.md"
        write_summary_markdown(data, md_path)

    # Create figure with subplots (3x3 grid)
    fig = plt.figure(figsize=(16, 14))

    # Row 1: Elevation analysis
    ax1 = fig.add_subplot(3, 3, 1)
    plot_elevation_histogram(data, ax1)

    ax2 = fig.add_subplot(3, 3, 2)
    plot_hypsometric_curve(data, ax2)

    ax3 = fig.add_subplot(3, 3, 3)
    plot_elevation_by_latitude(data, ax3)

    # Row 2: Tectonic features (trench, arc, ridge, collision)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6)
    ax7 = fig.add_subplot(3, 3, 7)
    plot_feature_distributions(data, [ax4, ax5, ax6, ax7])

    # Row 3: Plate statistics and noise
    ax8 = fig.add_subplot(3, 3, 8)
    plot_plate_statistics(data, ax8)

    ax9 = fig.add_subplot(3, 3, 9)
    plot_noise_distribution(data, ax9)

    fig.suptitle(f"Hex3 World Analysis - Seed {meta['seed']} ({meta['num_cells']:,} cells, Stage {meta['stage']})")
    plt.tight_layout()

    # Save or show
    if output_dir:
        output_path = output_dir / f"analysis_{meta['seed']}.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_comparison(data_list: list, labels: list, output_dir: Path, show: bool):
    """Compare multiple worlds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlay hypsometric curves
    for data, label in zip(data_list, labels):
        plot_hypsometric_curve(data, axes[0], label=label)
    axes[0].legend()

    # Overlay elevation histograms (just land/ocean ratio comparison)
    for data, label in zip(data_list, labels):
        elevation = np.array(data["cells"]["elevation"])
        area = np.array(data["cells"]["area"])
        bins = np.linspace(-0.5, 0.8, 50)
        axes[1].hist(elevation, bins=bins, weights=area, alpha=0.5, label=label)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Elevation")
    axes[1].set_ylabel("Area")
    axes[1].set_title("Elevation Distribution Comparison")
    axes[1].legend()

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "comparison.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hex3 world exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("files", nargs="+", type=Path, help="World export file(s)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for plots (default: current dir)")

    args = parser.parse_args()

    output_dir = args.output or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all files
    data_list = []
    labels = []
    for path in args.files:
        print(f"Loading {path}...")
        data = load_world(path)
        data_list.append(data)
        meta = data["metadata"]
        labels.append(f"seed={meta['seed']}, n={meta['num_cells']//1000}k")

    if len(data_list) == 1:
        analyze_single(data_list[0], output_dir, args.show)
    else:
        # Print summary for each
        for data in data_list:
            print_summary(data)
        # Generate comparison plots
        analyze_comparison(data_list, labels, output_dir, args.show)


if __name__ == "__main__":
    main()
