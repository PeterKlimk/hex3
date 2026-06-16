#!/usr/bin/env python3
"""Cross-resolution comparison harness for hex3 world exports.

Tests *resolution independence*: given the same seed generated at several mesh
resolutions, does each simulation system converge to the same output as the
mesh is refined? Two complementary views:

  1. AGGREGATE METRICS (area-weighted, mesh-size-free): land fraction,
     hypsometric percentiles, continental fraction, feature-magnitude
     percentiles, climate means, lake fraction, peak river flow. These are
     intensive quantities that should be invariant. Reported per resolution
     with the drift vs the highest-resolution reference.

  2. SPATIAL FIELDS (rasterized): each per-cell field is sampled onto a common
     equirectangular grid (nearest cell, via a 3D KD-tree on the unit-sphere
     cell centers) and compared to the reference resolution cell-for-cell. This
     catches spatial *drift* (a mountain belt shifting, a rain shadow moving)
     that aggregate distributions miss. Reported as cos-lat-weighted normalized
     RMSE and Pearson correlation vs the reference.

A system is "resolution independent" if its aggregate metrics stop drifting and
its rasterized field converges (nRMSE -> 0, corr -> 1) as resolution rises.

Usage:
    python resolution_compare.py world_50k.json.gz world_100k.json.gz \\
        world_200k.json.gz world_400k.json.gz
    python resolution_compare.py *.json.gz --grid 720 --plot out_dir/
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_world(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def cell_xyz(data: dict) -> np.ndarray:
    """Reconstruct unit-sphere cell centers from exported lat/lon (radians)."""
    lat = np.asarray(data["cells"]["latitude"], dtype=np.float64)
    lon = np.asarray(data["cells"]["longitude"], dtype=np.float64)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), np.sin(lat), cl * np.sin(lon)], axis=1)


# --------------------------------------------------------------------------- #
# Area-weighted statistics
# --------------------------------------------------------------------------- #
def wmean(v, w):
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(v * w) / np.sum(w))


def wpct(v, w, q):
    """Area-weighted percentile(s). q in [0,1]."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(q, cw, v)


def aggregate_metrics(data: dict) -> dict:
    """Intensive, mesh-size-free metrics that should be resolution invariant."""
    cells = data["cells"]
    elev = np.asarray(cells["elevation"], dtype=np.float64)
    area = np.asarray(cells["area"], dtype=np.float64)
    crust = np.asarray(cells["crust_type"], dtype=np.int32)
    n = len(elev)
    land = elev >= 0.0
    m = {}

    m["land_frac"] = float(np.sum(area[land]) / np.sum(area))
    m["continental_frac"] = float(np.sum(area[crust == 0]) / np.sum(area))
    p5, p50, p95 = wpct(elev, area, [0.05, 0.50, 0.95])
    m["elev_p5"] = float(p5)
    m["elev_p50"] = float(p50)
    m["elev_p95"] = float(p95)
    m["elev_max"] = float(elev.max())
    m["elev_mean"] = wmean(elev, area)
    if land.any():
        la = area[land]
        m["land_elev_p50"] = float(wpct(elev[land], la, 0.50))
        m["land_elev_p95"] = float(wpct(elev[land], la, 0.95))

    # Tectonic feature magnitudes (p99 = how strong the strongest belts are).
    feat = cells.get("features", {})
    for key in ("trench", "arc", "ridge", "collision", "activity"):
        if key in feat:
            v = np.abs(np.asarray(feat[key], dtype=np.float64))
            m[f"feat_{key}_p99"] = float(wpct(v, area, 0.99))

    # Climate (stage 2+).
    atmo = cells.get("atmosphere")
    if atmo:
        temp = np.asarray(atmo["temperature"], dtype=np.float64)
        m["temp_mean_land"] = wmean(temp[land], area[land]) if land.any() else 0.0
        m["temp_mean_ocean"] = (
            wmean(temp[~land], area[~land]) if (~land).any() else 0.0
        )
        m["thermal_contrast"] = m["temp_mean_land"] - m["temp_mean_ocean"]
        if "precipitation" in atmo:
            precip = np.asarray(atmo["precipitation"], dtype=np.float64)
            if land.any():
                m["precip_mean_land"] = wmean(precip[land], area[land])
                m["precip_arid_frac_land"] = float(
                    np.sum(area[land & (precip < 0.35)]) / np.sum(area[land])
                )
        if "uplift" in atmo:
            up = np.asarray(atmo["uplift"], dtype=np.float64)
            m["uplift_p99"] = float(wpct(up, area, 0.99))

    # Hydrology (stage 3+).
    hyd = cells.get("hydrology")
    if hyd:
        if "is_lake" in hyd:
            lake = np.asarray(hyd["is_lake"], dtype=bool)
            nonocean = elev >= 0.0  # lakes sit on land
            denom = np.sum(area[nonocean])
            m["lake_frac_land"] = (
                float(np.sum(area[lake & nonocean]) / denom) if denom > 0 else 0.0
            )
        if "flow_accumulation" in hyd:
            flow = np.asarray(hyd["flow_accumulation"], dtype=np.float64)
            # Peak discharge normalized to mean cell area => mesh-independent
            # "upstream-cell-equivalents" (cf. flow_count_equiv in the app).
            mean_area = float(np.mean(area))
            m["max_flow_equiv"] = float(flow.max() / mean_area) if mean_area else 0.0

    m["_n"] = n
    return m


# --------------------------------------------------------------------------- #
# Rasterization to a common grid
# --------------------------------------------------------------------------- #
def make_grid(nlon: int):
    """Equirectangular grid; nlat = nlon // 2. Returns (xyz, coslat_weights)."""
    nlat = nlon // 2
    # Pixel centers.
    lon = (np.arange(nlon) + 0.5) / nlon * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(nlat) + 0.5) / nlat * np.pi
    lon_g, lat_g = np.meshgrid(lon, lat)
    cl = np.cos(lat_g)
    xyz = np.stack(
        [cl * np.cos(lon_g), np.sin(lat_g), cl * np.sin(lon_g)], axis=-1
    ).reshape(-1, 3)
    weights = cl.reshape(-1)  # area element ~ cos(lat)
    return xyz, weights, (nlat, nlon)


def sampler_for(data: dict, grid_xyz: np.ndarray):
    """Nearest-cell sampler: returns a function mapping a per-cell array -> grid."""
    tree = cKDTree(cell_xyz(data))
    _, idx = tree.query(grid_xyz, k=1)

    def sample(values) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[idx]

    return sample


def field_arrays(data: dict) -> dict:
    """Per-cell fields to rasterize and compare, keyed by name."""
    cells = data["cells"]
    out = {"elevation": cells["elevation"]}
    feat = cells.get("features", {})
    for key in ("trench", "arc", "ridge", "collision", "activity"):
        if key in feat:
            out[f"feat_{key}"] = feat[key]
    atmo = cells.get("atmosphere")
    if atmo:
        for key in ("temperature", "precipitation", "uplift"):
            if key in atmo:
                out[key] = atmo[key]
    hyd = cells.get("hydrology")
    if hyd and "flow_accumulation" in hyd:
        # log1p compresses the heavy tail so the metric isn't dominated by a few
        # trunk cells.
        out["log_flow"] = np.log1p(np.asarray(hyd["flow_accumulation"], float))
    return out


def compare_field(a: np.ndarray, ref: np.ndarray, w: np.ndarray):
    """cos-lat-weighted normalized RMSE and Pearson correlation."""
    wa = w / np.sum(w)
    ma = np.sum(wa * a)
    mr = np.sum(wa * ref)
    da, dr = a - ma, ref - mr
    cov = np.sum(wa * da * dr)
    va = np.sum(wa * da * da)
    vr = np.sum(wa * dr * dr)
    corr = cov / np.sqrt(va * vr) if va > 0 and vr > 0 else float("nan")
    rmse = np.sqrt(np.sum(wa * (a - ref) ** 2))
    nrmse = rmse / np.sqrt(vr) if vr > 0 else float("nan")
    return nrmse, corr


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(x):
    if isinstance(x, float):
        if x != x:
            return "  nan"
        if abs(x) >= 1000 or (x != 0 and abs(x) < 0.001):
            return f"{x:.2e}"
        return f"{x:+.4f}" if abs(x) < 100 else f"{x:.1f}"
    return str(x)


def print_aggregate_table(labels, metrics_list):
    ref = metrics_list[-1]  # highest resolution
    keys = [k for k in metrics_list[-1] if not k.startswith("_")]
    # Union of keys across files (some stages may differ), preserve ref order.
    for mm in metrics_list:
        for k in mm:
            if not k.startswith("_") and k not in keys:
                keys.append(k)

    w0 = max(len(k) for k in keys) + 1
    header = f"{'metric':<{w0}}" + "".join(f"{l:>14}" for l in labels)
    header += f"{'drift%':>10}"
    print("\n=== AGGREGATE METRICS (area-weighted; reference = " f"{labels[-1]}) ===")
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<{w0}}"
        vals = []
        for mm in metrics_list:
            v = mm.get(k)
            vals.append(v)
            row += f"{(fmt(v) if v is not None else '-'):>14}"
        # Drift of lowest-res vs reference, relative to reference scale.
        rv = ref.get(k)
        first = next((v for v in vals if v is not None), None)
        if rv is not None and first is not None:
            scale = max(abs(rv), 1e-9)
            drift = 100.0 * abs(first - rv) / scale
            row += f"{drift:>9.1f}%"
        else:
            row += f"{'-':>10}"
        print(row)


def coarse_invariance_summary(worlds, metrics_list):
    """Statistical-invariance test for the COARSE axis.

    Changing the coarse cell count resamples the sphere (Fibonacci lattice +
    jitter) and relocates the whole geography, so a cell-for-cell spatial
    comparison is meaningless across coarse resolutions. The meaningful question
    is statistical: does resolution move a metric more than *reseeding* does?

    For each metric we build a [seed x resolution] matrix and compare:
      - seed_band : typical spread across seeds at fixed resolution (the noise
                    floor from re-randomizing the world).
      - res_drift : spread of the seed-mean across resolution (systematic
                    movement attributable to resolution).
    ratio = res_drift / seed_band. ratio <~ 1 => resolution perturbs the metric
    no more than reseeding => RESOLUTION INDEPENDENT. ratio >> 1 => systematic
    resolution dependence worth investigating.
    """
    seeds = sorted({d["metadata"]["seed"] for _, d in worlds})
    res = sorted({d["metadata"]["num_cells"] for _, d in worlds})
    res_labels = [f"{n // 1000}k" if n >= 1000 else str(n) for n in res]
    # index: (seed, num_cells) -> metrics dict
    by_key = {}
    for (_, d), mm in zip(worlds, metrics_list):
        by_key[(d["metadata"]["seed"], d["metadata"]["num_cells"])] = mm

    keys = []
    for mm in metrics_list:
        for k in mm:
            if not k.startswith("_") and k not in keys:
                keys.append(k)

    if len(seeds) < 2:
        print("\n(Only one seed present — cannot separate resolution drift from "
              "seed noise. Re-run the sweep with >=2 seeds for the invariance test.)")
        return

    print(f"\n=== COARSE-AXIS STATISTICAL INVARIANCE "
          f"({len(seeds)} seeds x {len(res)} resolutions) ===")
    print("Per resolution: mean across seeds. ratio = (drift across resolution) / "
          "(spread across seeds).")
    print("ratio <~1 => resolution independent; ratio >>1 => resolution dependent.\n")
    w0 = max(len(k) for k in keys) + 1
    header = f"{'metric':<{w0}}" + "".join(f"{l:>11}" for l in res_labels)
    header += f"{'seedσ':>10}{'resΔ':>10}{'ratio':>8}  verdict"
    print(header)
    print("-" * len(header))

    for k in keys:
        # matrix[s][r]
        mat = np.full((len(seeds), len(res)), np.nan)
        for si, s in enumerate(seeds):
            for ri, n in enumerate(res):
                mm = by_key.get((s, n))
                if mm is not None and mm.get(k) is not None:
                    mat[si, ri] = mm[k]
        if np.all(np.isnan(mat)):
            continue
        seed_mean = np.nanmean(mat, axis=0)            # per resolution
        # seed spread at each resolution, then typical (median) -> noise floor.
        seed_spread = np.nanstd(mat, axis=0, ddof=0)
        seed_band = float(np.nanmedian(seed_spread))
        res_drift = float(np.nanmax(seed_mean) - np.nanmin(seed_mean))
        ratio = res_drift / seed_band if seed_band > 1e-12 else (
            float("inf") if res_drift > 1e-9 else 0.0)
        # Practical-significance gate: relative drift vs the metric's own scale.
        # Guards against flagging a metric whose drift AND seed noise are both
        # negligible (e.g. a target-pinned land fraction).
        scale = float(np.nanmedian(np.abs(seed_mean)))
        rel_drift = res_drift / scale if scale > 1e-9 else 0.0
        if rel_drift < 0.02:
            verdict = "indep"          # moves <2% of its own magnitude: flat
        elif ratio <= 1.5:
            verdict = "indep"          # resolution within reseed noise
        elif ratio <= 3.0:
            verdict = "watch"
        else:
            verdict = "DEPENDENT"
        row = f"{k:<{w0}}"
        for ri in range(len(res)):
            row += f"{fmt(float(seed_mean[ri])):>11}"
        row += f"{seed_band:>10.4f}{res_drift:>10.4f}"
        row += (f"{ratio:>8.1f}" if np.isfinite(ratio) else f"{'inf':>8}")
        row += f"  {verdict}"
        print(row)
    print("\nseedσ = median over resolutions of std-across-seeds (reseed noise floor).")
    print("resΔ  = range of the seed-mean across resolution (systematic drift).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--mode", choices=["coarse", "fine"], default="coarse",
                    help="coarse: statistical-invariance test across seeds "
                         "(geography changes with cell count, so spatial compare "
                         "is suppressed). fine: spatial cell-for-cell convergence "
                         "(coarse world fixed; adaptive mesh refined).")
    ap.add_argument("--grid", type=int, default=720,
                    help="Raster longitude resolution (nlat=grid/2). Default 720.")
    ap.add_argument("--plot", type=Path, default=None,
                    help="Directory to write difference maps (PNG) vs reference.")
    args = ap.parse_args()

    worlds = [(p, load_world(p)) for p in args.files]
    # Sort ascending by cell count so the reference is last (highest res).
    worlds.sort(key=lambda pw: pw[1]["metadata"]["num_cells"])
    labels = []
    for p, d in worlds:
        n = d["metadata"]["num_cells"]
        labels.append(f"{n//1000}k" if n >= 1000 else str(n))
    seeds = {d["metadata"]["seed"] for _, d in worlds}
    stages = {d["metadata"]["stage"] for _, d in worlds}
    print(f"Mode: {args.mode} | files: {len(worlds)} | seeds: {sorted(seeds)} "
          f"| stages: {sorted(stages)}")
    print("Resolutions (ascending): " + ", ".join(
        f"{l}({d[1]['metadata']['num_cells']})" for l, d in zip(labels, worlds)))

    metrics_list = [aggregate_metrics(d) for _, d in worlds]

    if args.mode == "coarse":
        # Per-seed aggregate tables (drift vs that seed's highest res), then the
        # cross-seed invariance summary which is the real verdict.
        for s in sorted(seeds):
            sub = [(p, d, mm) for (p, d), mm in zip(worlds, metrics_list)
                   if d["metadata"]["seed"] == s]
            if len(sub) > 1:
                print(f"\n----- seed {s} -----")
                slabels = [f"{d['metadata']['num_cells']//1000}k" for _, d, _ in sub]
                print_aggregate_table(slabels, [mm for _, _, mm in sub])
        coarse_invariance_summary(worlds, metrics_list)
        print("\nNote: spatial cell-for-cell comparison is intentionally skipped in "
              "coarse mode —\nchanging the coarse cell count resamples the sphere and "
              "moves the continents,\nso only statistical invariance is meaningful "
              "here. Use --mode fine for the\nspatial convergence test on the adaptive "
              "(erosion) mesh.")
        return

    # ---- fine mode: aggregate table + rasterized spatial convergence ----
    print_aggregate_table(labels, metrics_list)
    grid_xyz, gw, shape = make_grid(args.grid)
    _, ref_data = worlds[-1]
    ref_sampler = sampler_for(ref_data, grid_xyz)
    ref_fields = field_arrays(ref_data)
    ref_rasters = {k: ref_sampler(v) for k, v in ref_fields.items()}

    field_names = list(ref_fields.keys())
    print(f"\n=== SPATIAL FIELD CONVERGENCE vs {labels[-1]} "
          f"(grid {shape[1]}x{shape[0]}; nRMSE / corr) ===")
    print("lower nRMSE and corr->1 as resolution rises => converging\n")
    w0 = max(len(k) for k in field_names) + 1
    header = f"{'field':<{w0}}" + "".join(f"{l:>16}" for l in labels[:-1])
    print(header)
    print("-" * len(header))

    rows = {k: f"{k:<{w0}}" for k in field_names}
    diff_store = {}
    for (p, d), label in zip(worlds[:-1], labels[:-1]):
        sampler = sampler_for(d, grid_xyz)
        fields = field_arrays(d)
        for k in field_names:
            if k not in fields:
                rows[k] += f"{'-':>16}"
                continue
            ras = sampler(fields[k])
            nrmse, corr = compare_field(ras, ref_rasters[k], gw)
            rows[k] += f"{nrmse:>7.3f}/{corr:>6.3f}"
            if args.plot is not None:
                diff_store[(k, label)] = (ras - ref_rasters[k]).reshape(shape)
    for k in field_names:
        print(rows[k])

    if args.plot is not None:
        _write_diff_maps(args.plot, diff_store, shape, labels[-1])

    print("\nLegend: nRMSE = RMSE / (reference field std); corr = cos-lat-weighted "
          "Pearson r.\nA resolution-independent system shows nRMSE shrinking toward "
          "0 and corr toward 1\nas the coarser meshes approach the reference.")


def _write_diff_maps(outdir: Path, diff_store, shape, ref_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    for (k, label), diff in diff_store.items():
        lim = np.percentile(np.abs(diff), 99) or 1.0
        plt.figure(figsize=(10, 5))
        plt.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[-180, 180, -90, 90], aspect="auto")
        plt.colorbar(label=f"{k}: {label} - {ref_label}")
        plt.title(f"{k} difference: {label} vs {ref_label} (reference)")
        plt.tight_layout()
        path = outdir / f"diff_{k}_{label}.png"
        plt.savefig(path, dpi=90)
        plt.close()
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
