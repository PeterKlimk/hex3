#!/usr/bin/env python3
"""Render the optional dossier climatology-null spatial evidence.

The packet is emitted by:

    dossier --include-climatology-spatial --output seed.json

It holds terrain fixed and compares the product precipitation/hydrology with
the fitted latitude/elevation/ocean-distance null. This renderer deliberately
shows only that comparison; it is not a general world-export viewer.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
    TwoSlopeNorm,
)
from scipy.spatial import cKDTree


PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "hex3_precip",
    ["#c5aa72", "#91a45b", "#4f8b55", "#23756b", "#244f83"]
)
PRECIP_CMAP.set_bad("#dceaf2")
RESIDUAL_CMAP = plt.get_cmap("RdBu").copy()
RESIDUAL_CMAP.set_bad("#dceaf2")
FLOW_CMAP = plt.get_cmap("Blues").copy()
FLOW_CMAP.set_bad("#dceaf2")
RIVER_CHANGE_CMAP = ListedColormap(
    ["#eeeeee", "#68a9cf", "#154f83", "#d97732", "#8b4ba8"]
)
RIVER_CHANGE_LABELS = [
    "none in both",
    "same river",
    "same major",
    "product stronger",
    "null stronger",
]


def load_packet(path: Path) -> tuple[dict, dict]:
    with path.open("rt", encoding="utf-8") as handle:
        packet = json.load(handle)
    try:
        spatial = packet["climatology_null"]["spatial_evidence"]
    except KeyError as error:
        raise SystemExit(
            "dossier has no climatology spatial evidence; regenerate it with "
            "--include-climatology-spatial"
        ) from error
    return packet, spatial


def make_grid(nlon: int) -> tuple[np.ndarray, tuple[int, int]]:
    nlat = nlon // 2
    lon = (np.arange(nlon) + 0.5) / nlon * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (np.arange(nlat) + 0.5) / nlat * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    xyz = np.stack(
        [cos_lat * np.cos(lon_grid), np.sin(lat_grid), cos_lat * np.sin(lon_grid)],
        axis=-1,
    )
    return xyz.reshape(-1, 3), (nlat, nlon)


def robust_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, percentile)), 1e-9)


def river_change(product: np.ndarray, null: np.ndarray) -> np.ndarray:
    change = np.zeros(product.shape, dtype=np.uint8)
    same = product == null
    change[same & (product == 1)] = 1
    change[same & (product == 2)] = 2
    change[product > null] = 3
    change[null > product] = 4
    return change


def render(packet: dict, spatial: dict, output: Path, nlon: int, dpi: int) -> None:
    xyz = np.asarray(spatial["unit_xyz"], dtype=np.float64)
    grid_xyz, shape = make_grid(nlon)
    _, nearest = cKDTree(xyz).query(grid_xyz, k=1)

    def sample(name: str, dtype=np.float64) -> np.ndarray:
        return np.asarray(spatial[name], dtype=dtype)[nearest].reshape(shape)

    ocean = sample("ocean", dtype=bool)
    product_precip = sample("product_precipitation")
    null_precip = sample("null_precipitation")
    residual = product_precip - null_precip
    product_flow = sample("product_flow_accumulation")
    null_flow = sample("null_flow_accumulation")
    product_rivers = sample("product_river_class", dtype=np.uint8)
    null_rivers = sample("null_river_class", dtype=np.uint8)

    land = ~ocean
    product_precip[~land] = np.nan
    null_precip[~land] = np.nan
    residual[~land] = np.nan
    product_flow[~land] = np.nan
    null_flow[~land] = np.nan

    wet_values = np.concatenate([product_precip[land], null_precip[land]])
    precip_max = max(float(np.nanpercentile(wet_values, 99.0)), 1e-6)
    # The transported field has narrow wet coastal/orographic tails. A square-
    # root display keeps them visible without flattening the rest of the land;
    # the residual uses a tighter robust limit to reveal coherent broad signs.
    residual_limit = robust_limit(residual, percentile=97.0)
    positive_flow = np.concatenate(
        [product_flow[np.isfinite(product_flow)], null_flow[np.isfinite(null_flow)]]
    )
    flow_floor = max(float(np.percentile(positive_flow[positive_flow > 0], 1.0)), 1e-15)
    product_log_flow = np.log10(np.maximum(product_flow, flow_floor))
    null_log_flow = np.log10(np.maximum(null_flow, flow_floor))
    flow_values = np.concatenate(
        [product_log_flow[np.isfinite(product_log_flow)], null_log_flow[np.isfinite(null_log_flow)]]
    )
    flow_min, flow_max = np.percentile(flow_values, [1.0, 99.5])
    changes = river_change(product_rivers, null_rivers)
    changes[~land] = 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 8.7), constrained_layout=True)
    extent = (-180, 180, -90, 90)
    panels = [
        (
            product_precip,
            "product precipitation",
            dict(cmap=PRECIP_CMAP, norm=PowerNorm(gamma=0.5, vmin=0, vmax=precip_max)),
        ),
        (
            null_precip,
            "conditional-null precipitation",
            dict(cmap=PRECIP_CMAP, norm=PowerNorm(gamma=0.5, vmin=0, vmax=precip_max)),
        ),
        (
            residual,
            "product minus null precipitation",
            dict(
                cmap=RESIDUAL_CMAP,
                norm=TwoSlopeNorm(vcenter=0, vmin=-residual_limit, vmax=residual_limit),
            ),
        ),
        (
            product_log_flow,
            "product log10 flow accumulation",
            dict(cmap=FLOW_CMAP, vmin=flow_min, vmax=flow_max),
        ),
        (
            null_log_flow,
            "null log10 flow accumulation",
            dict(cmap=FLOW_CMAP, vmin=flow_min, vmax=flow_max),
        ),
        (
            changes,
            "river-class change",
            dict(cmap=RIVER_CHANGE_CMAP, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5)),
        ),
    ]

    for index, (axis, (values, title, kwargs)) in enumerate(zip(axes.flat, panels)):
        image = axis.imshow(values, extent=extent, origin="upper", interpolation="nearest", **kwargs)
        axis.contour(
            np.linspace(-180, 180, shape[1]),
            np.linspace(90, -90, shape[0]),
            land.astype(np.uint8),
            levels=[0.5],
            colors="black",
            linewidths=0.35,
            alpha=0.65,
        )
        axis.set_title(title)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_yticks([-60, -30, 0, 30, 60])
        if index == 5:
            colorbar = fig.colorbar(image, ax=axis, ticks=range(5), shrink=0.78)
            colorbar.ax.set_yticklabels(RIVER_CHANGE_LABELS)
        else:
            fig.colorbar(image, ax=axis, shrink=0.78)

    manifest = packet["manifest"]
    comparison = packet["climatology_null"]["precipitation"]
    fig.suptitle(
        f"Seed {manifest['seed']} · {manifest['active_cells']:,} cells · "
        f"null R² {comparison['explained_variance_fraction']:.3f}",
        fontsize=14,
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dossier", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--grid", type=int, default=720, help="longitude pixels; latitude is half")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    packet, spatial = load_packet(args.dossier)
    output = args.output or args.dossier.with_suffix(".png")
    render(packet, spatial, output, args.grid, args.dpi)


if __name__ == "__main__":
    main()
