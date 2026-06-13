#!/usr/bin/env python3
"""Render exported world layers as equirectangular maps.

Gives a quick visual readout of any per-cell layer without running the app:

    python render_map.py world.json.gz                 # default layer grid
    python render_map.py world.json.gz -l precipitation elevation
    python render_map.py world.json.gz -o maps.png

Layers: elevation, crust, plates, precipitation, temperature, uplift, wind,
upper_wind, flow, moisture-adjacent fields fall back to a generic colormap if
unknown.
"""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm


def load_world(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


# Terrain-like colormap: deep ocean -> shelf -> lowland green -> mountain -> snow
TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "hex3_terrain",
    [
        (0.00, "#01051a"),  # abyss
        (0.42, "#0a2d52"),  # deep ocean
        (0.55, "#1d6f94"),  # shelf
        (0.58, "#2e7d32"),  # coast green
        (0.66, "#7a9e44"),  # plains
        (0.76, "#9c7a3c"),  # uplands
        (0.88, "#6e6258"),  # mountains
        (1.00, "#f5f7fa"),  # snow
    ],
)

PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "hex3_precip",
    ["#c2a36b", "#8c9a4e", "#2e7d44", "#1c6e63", "#1a4e7a"],
)


def layer_values(data: dict, layer: str):
    """Return (values, plot kwargs) for a named layer."""
    cells = data["cells"]
    atmosphere = cells.get("atmosphere") or {}
    hydrology = cells.get("hydrology") or {}

    if layer == "elevation":
        v = np.array(cells["elevation"])
        lim = max(abs(v.min()), abs(v.max()))
        return v, dict(cmap=TERRAIN_CMAP, vmin=-lim, vmax=lim)
    if layer == "crust":
        return np.array(cells["crust_type"]), dict(cmap="cividis_r", vmin=0, vmax=1)
    if layer == "plates":
        return np.array(cells["plate_id"]), dict(cmap="tab20")
    if layer == "precipitation":
        v = np.array(atmosphere.get("precipitation"))
        return v, dict(cmap=PRECIP_CMAP, vmin=0, vmax=2.5)
    if layer == "temperature":
        return np.array(atmosphere.get("temperature")), dict(cmap="coolwarm")
    if layer == "uplift":
        v = np.array(atmosphere.get("uplift"))
        lim = max(abs(np.nanmin(v)), abs(np.nanmax(v)), 1e-6)
        return v, dict(cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    if layer in ("wind", "upper_wind"):
        v = np.array(atmosphere.get(layer))
        lim = max(abs(np.nanmin(v)), abs(np.nanmax(v)), 1e-6)
        return v, dict(cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    if layer in ("wind_speed", "upper_wind_speed"):
        return np.array(atmosphere.get(layer)), dict(cmap="viridis", vmin=0)
    if layer == "flow":
        # Land-only river view: ocean cells get NaN so rivers stand out.
        v = np.array(hydrology.get("flow_accumulation", cells.get("elevation")), dtype=float)
        elev = np.array(cells["elevation"])
        v[elev < 0] = np.nan
        return v, dict(cmap="Blues", norm=LogNorm(vmin=1.0, vmax=max(np.nanmax(v), 10.0)))
    # Fallback: look it up in features / noise / atmosphere by name
    for group in (cells.get("features") or {}, cells.get("noise") or {}, atmosphere, hydrology):
        if layer in group:
            return np.array(group[layer]), dict(cmap="viridis")
    raise KeyError(f"unknown layer: {layer}")


def render(data: dict, layers: list[str], output: Path, dpi: int):
    cells = data["cells"]
    lat = np.degrees(np.array(cells["latitude"]))
    lon = np.degrees(np.array(cells["longitude"]))
    elevation = np.array(cells["elevation"])

    n = len(layers)
    ncols = 1 if n == 1 else 2
    nrows = (n + ncols - 1) // ncols
    panel_w = 16 if n == 1 else 8
    panel_h = 8.4 if n == 1 else 4.2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False
    )

    # Marker size tuned so ~100k points tile an 8x4 panel without gaps.
    size = max(0.3, 140000 / len(lat)) * (4.0 if n == 1 else 1.0)

    for ax, layer in zip(axes.flat, layers):
        v, kwargs = layer_values(data, layer)
        sc = ax.scatter(lon, lat, c=v, s=size, linewidths=0, **kwargs)
        # Coastline overlay for non-elevation layers: outline land cells faintly.
        if layer not in ("elevation", "crust"):
            coast = elevation >= 0
            ax.scatter(
                lon[coast], lat[coast], s=size * 0.25, c="black", alpha=0.12, linewidths=0
            )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(layer)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.8)

    for ax in axes.flat[n:]:
        ax.axis("off")

    meta = data["metadata"]
    fig.suptitle(f"Seed {meta['seed']} - {meta['num_cells']:,} cells, stage {meta['stage']}")
    plt.tight_layout()
    fig.savefig(output, dpi=dpi)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument(
        "-l", "--layers", nargs="+",
        default=["elevation", "precipitation", "crust", "temperature"],
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    data = load_world(args.world)
    output = args.output or Path(f"map_{data['metadata']['seed']}.png")
    render(data, args.layers, output, args.dpi)


if __name__ == "__main__":
    main()
