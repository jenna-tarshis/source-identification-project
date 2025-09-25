# tin_surface_from_arrivals.py
# Create a TIN (Delaunay) arrival-time surface for a chosen event
# using arrival_times_object.pkl produced by polynomial_fit.py.

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from datetime import datetime
from pyproj import Transformer

# ----------------------------- Config imports -----------------------------
from config import (
    pollutant,
    facility_id,
    start_dt as START_DT,
    end_dt as END_DT,
)

# ----------------------------- Paths & helpers -----------------------------
PLOT_ROOT = "plots"
TIN_SUBDIR = "tin_surfaces"

def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def _fmt_dt_for_dir(x) -> str:
    return pd.to_datetime(x).strftime("%Y-%m-%d_%H%M")

def analysis_dir() -> str:
    d = os.path.join(
        PLOT_ROOT,
        _safe_name(pollutant),
        _safe_name(facility_id),
        f"{_fmt_dt_for_dir(START_DT)}__{_fmt_dt_for_dir(END_DT)}",
    )
    os.makedirs(d, exist_ok=True)
    return d

def latlon_to_local_xy(lat, lon, ref_lat, ref_lon):
    # EPSG:4326 -> EPSG:3857 (meters), then shift by reference sensor
    fwd = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x, y = fwd.transform(lon, lat)
    xr, yr = fwd.transform(ref_lon, ref_lat)
    return x - xr, y - yr

# ----------------------------- Core logic -----------------------------
def load_facility_coordinates(facility_json_path):
    """
    Expect a JSON with at least: { "<sensor_id>": {"lat": ..., "lon": ...}, ... }
    If your schema is different, adapt the accessors below.
    """
    with open(facility_json_path, "r") as f:
        data = json.load(f)
    coords = {}
    for sid, meta in data.items():
        # Try common keys; adapt if your schema differs.
        lat = meta.get("lat") or meta.get("latitude")
        lon = meta.get("lon") or meta.get("longitude")
        if lat is None or lon is None:
            continue
        coords[str(sid)] = (float(lat), float(lon))
    return coords

def load_event_arrivals(arrival_times_pkl, event_id):
    """
    arrival_times_object.pkl structure:
      { event_id: [(sensor_id, (peak_value, peak_time)), ... (sorted by time)] }
    """
    with open(arrival_times_pkl, "rb") as f:
        d = pickle.load(f)
    if event_id not in d:
        raise KeyError(f"Event '{event_id}' not found in {arrival_times_pkl}. "
                       f"Available: {list(d.keys())[:10]} ...")
    return d[event_id]

def build_tin_and_plot(event_id,
                       event_peaks,
                       coords_map,
                       save_dir,
                       add_quivers=True,
                       n_levels=10):
    # ---------- assemble inputs ----------
    usable = [(sid, v_t) for sid, v_t in event_peaks if str(sid) in coords_map]
    if len(usable) < 3:
        raise ValueError("Need at least 3 sensors with coordinates for a TIN.")

    sids = [str(sid) for sid, _ in usable]
    lats = np.array([coords_map[sid][0] for sid in sids], dtype=float)
    lons = np.array([coords_map[sid][1] for sid in sids], dtype=float)
    peak_times = np.array([pd.to_datetime(vt[1]) for _, vt in usable])

    # reference = earliest arrival (for local XY)
    idx0 = int(np.argmin(peak_times))
    ref_lat, ref_lon = lats[idx0], lons[idx0]

    # lat/lon -> local meters
    xs, ys = [], []
    for la, lo in zip(lats, lons):
        x, y = latlon_to_local_xy(la, lo, ref_lat, ref_lon)
        xs.append(x); ys.append(y)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # arrival offsets (s since earliest)
    t0 = peak_times.min()
    zs = np.array([(pt - t0).total_seconds() for pt in peak_times], dtype=float)

    # TIN
    tri = mtri.Triangulation(xs, ys)

    # ---------- per-triangle plane fit -> slope magnitude ----------
    # z = a x + b y + c  =>  gradient = [a, b], slope magnitude = sqrt(a^2 + b^2) (s/m)
    slopes = np.empty(tri.triangles.shape[0], dtype=float)
    tri_cx = np.empty_like(slopes)  # centroids for quivers
    tri_cy = np.empty_like(slopes)
    gx = np.empty_like(slopes)
    gy = np.empty_like(slopes)

    for k, inds in enumerate(tri.triangles):
        X = np.c_[xs[inds], ys[inds], np.ones(3)]
        Z = zs[inds]
        a, b, c = np.linalg.lstsq(X, Z, rcond=None)[0]
        slopes[k] = np.hypot(a, b)
        tri_cx[k] = xs[inds].mean()
        tri_cy[k] = ys[inds].mean()
        gx[k] = a
        gy[k] = b

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # (A) Triangles shaded by slope (steep = dark purple)
    # Use 'Purples' so higher slopes map to darker purple.
    tpc = ax.tripcolor(tri, facecolors=slopes, shading="flat",
                       cmap="Purples", edgecolors="k", linewidth=0.15)
    cbar1 = fig.colorbar(tpc, ax=ax, pad=0.01)
    cbar1.set_label(r"Slope ‖∇z‖ (s m$^{-1}$)")

    # (B) Arrival-time contours on top (seconds)
    levels = np.linspace(float(zs.min()), float(zs.max()), n_levels)
    cs = ax.tricontour(tri, zs, levels=levels, colors="k", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%.0f s", fontsize=8, inline=True)

    # (C) Points colored by arrival time (earliest = purple, latest = white)
    sc = ax.scatter(xs, ys, c=zs, cmap="Purples_r", s=55, edgecolor="k", zorder=10)
    for sid, x, y in zip(sids, xs, ys):
        ax.text(x + 8, y + 8, sid, fontsize=8, color="k", alpha=0.95)
    cbar2 = fig.colorbar(sc, ax=ax, pad=0.04)
    cbar2.set_label("Arrival time offset (s)")

    # (D) Optional: per-triangle gradient arrows (direction of steepest increase)
    if add_quivers:
        # normalize arrows for display
        mag = np.hypot(gx, gy)
        mag[mag == 0] = 1.0
        scale = 80.0 / np.median(mag)
        ax.quiver(tri_cx, tri_cy, gx, gy, angles="xy", scale_units="xy",
                  scale=1/scale, color="white", width=0.003, alpha=0.9, zorder=11)

    ax.set_title(f"TIN Arrival-Time Surface — Event {event_id}\n"
                 "(Triangles shaded by slope; points colored by arrival offset)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{event_id}_tin_surface.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Build a TIN arrival-time surface for an event.")
    parser.add_argument("--event", required=True, help="Event ID (e.g., E1)")
    parser.add_argument("--arrivals", default="arrival_times_object.pkl",
                        help="Path to arrival_times_object.pkl")
    parser.add_argument("--facility_json", default=f"{facility_id}_facility_dict.json",
                        help="Facility JSON with sensor coordinates")
    parser.add_argument("--no_quivers", action="store_true", help="Disable gradient arrows")
    args = parser.parse_args()

    # Load inputs
    arrivals = load_event_arrivals(args.arrivals, args.event)
    coords_map = load_facility_coordinates(args.facility_json)

    # Output dir
    out_dir = os.path.join(analysis_dir(), TIN_SUBDIR)

    # Build + plot
    build_tin_and_plot(
        event_id=args.event,
        event_peaks=arrivals,
        coords_map=coords_map,
        save_dir=out_dir,
        add_quivers=not args.no_quivers,
    )

if __name__ == "__main__":
    main()
