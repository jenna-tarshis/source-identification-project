from __future__ import annotations

import os
import json
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import Delaunay
from pyproj import Transformer

from config import facility_id, pollutant as POLLUTANT, start_dt as START_DT, end_dt as END_DT

# ----------------------------- Plot folder helpers -----------------------------
PLOT_ROOT = "plots"
os.makedirs(PLOT_ROOT, exist_ok=True)

def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def _fmt_dt_for_dir(x) -> str:
    # incoming format like "2025-06-01 00:00:00" → "2025-06-01_0000"
    return pd.to_datetime(x).strftime("%Y-%m-%d_%H%M")

def _analysis_dir(pollutant: str, facility_id: str, start_dt: str, end_dt: str) -> str:
    """
    plots/<pollutant>/<facility_id>/<start-dt>__<end-dt>
    """
    d = os.path.join(
        PLOT_ROOT,
        _safe_name(pollutant),
        _safe_name(facility_id),
        f"{_fmt_dt_for_dir(start_dt)}__{_fmt_dt_for_dir(end_dt)}",
    )
    os.makedirs(d, exist_ok=True)
    return d

def _plot_path_for_event(event_id, equipment_data, facility_id: str) -> str:
    """
    plots/<pollutant>/<facility_id>/<start>__<end>/<event-start-date>_<event_id>.png
    """
    try:
        min_ts = min(ts for _, (_, ts) in equipment_data)
        ev_start_str = pd.to_datetime(min_ts).strftime("%Y-%m-%d")
    except Exception:
        ev_start_str = "unknown_date"

    base = _analysis_dir(POLLUTANT, facility_id, START_DT, END_DT)
    filename = f"{ev_start_str}_{_safe_name(event_id)}.png"
    return os.path.join(base, filename)

def upsert_event_origin_csv(result_dict, equipment_data, event_id: str, facility_id: str):
    """
    Upserts one row into:
      plots/<pollutant>/<facility_id>/<start>__<end>/estimated_origins.csv
    Uniqueness key: (date, event_id)
    """
    import csv

    # event start date used for key/filename
    try:
        min_ts = min(ts for _, (_, ts) in equipment_data)
        date_str = pd.to_datetime(min_ts).date().isoformat()
    except Exception:
        date_str = "unknown_date"

    lat, lon = map(float, result_dict["latlon_origin"])
    x_m, y_m = map(float, result_dict["meters_origin"])
    n_sensors = int(len(result_dict["x_vals"]))
    best_min_val = float(result_dict["multi_result"]["best_min_val"])

    base = _analysis_dir(POLLUTANT, facility_id, START_DT, END_DT)
    csv_path = os.path.join(base, "estimated_origins.csv")

    row = {
        "facility_id": facility_id,
        "pollutant": POLLUTANT,
        "analysis_start": START_DT,
        "analysis_end": END_DT,
        "event_id": str(event_id),
        "date": date_str,
        "origin_lat": lat,
        "origin_lon": lon,
        "origin_x_m": x_m,
        "origin_y_m": y_m,
        "n_sensors_used": n_sensors,
        "best_min_val_seconds": best_min_val,
    }
    fieldnames = list(row.keys())

    rows, updated = [], False
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = [dict(rr) for rr in r]
        for rr in rows:
            if rr.get("date") == row["date"] and rr.get("event_id") == row["event_id"]:
                rr.update({k: str(v) if isinstance(v, (float, int)) else v for k, v in row.items()})
                updated = True
                break
    if not updated:
        rows.append({k: str(v) if isinstance(v, (float, int)) else v for k, v in row.items()})

    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)

    print(f"[UPSERTED] {csv_path} :: ({'updated' if updated else 'inserted'}) {row['date']}_{row['event_id']}")



# ----------------------------- Coordinate Conversion --------------------------------

def latlon_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    x_ref, y_ref = transformer.transform(ref_lon, ref_lat)
    return x - x_ref, y - y_ref


def meters_to_latlon(x: float, y: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    x_ref, y_ref = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform(ref_lon, ref_lat)
    lon, lat = transformer.transform(x + x_ref, y + y_ref)
    return (lat, lon)

# --------------------------- Arrival Time Helpers -----------------------------

def _poly_segment_peak(
    df: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "datetime",
    degree: int = 2,
    n_dense: int = 1000,
) -> Tuple[float, pd.Timestamp]:
    """Return (peak_value, peak_timestamp) after polynomial fitting on one segment.

    Falls back to raw max if the segment is too short or degenerate.
    """
    df = df.dropna(subset=[time_col, value_col]).copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    if len(df) < degree + 1:
        idx = df[value_col].idxmax()
        return float(df.loc[idx, value_col]), pd.to_datetime(df.loc[idx, time_col])

    t0 = df[time_col].iloc[0]
    x = (df[time_col] - t0).dt.total_seconds().to_numpy()
    y = df[value_col].to_numpy()

    if not np.isfinite(x).all() or (x.max() - x.min() <= 0):
        idx = df[value_col].idxmax()
        return float(df.loc[idx, value_col]), pd.to_datetime(df.loc[idx, time_col])

    try:
        coefs = np.polyfit(x, y, deg=degree)
    except Exception:
        idx = df[value_col].idxmax()
        return float(df.loc[idx, value_col]), pd.to_datetime(df.loc[idx, time_col])

    poly = np.poly1d(coefs)
    x_dense = np.linspace(x.min(), x.max(), max(10, n_dense))
    y_dense = poly(x_dense)

    i = int(np.argmax(y_dense))
    peak_val = float(y_dense[i])
    peak_ts = t0 + pd.to_timedelta(float(x_dense[i]), unit="s")
    return peak_val, pd.to_datetime(peak_ts)


def _normalize_arrival_times(arrival_list: List[Tuple[str, Tuple[float, Any]]]):
    """Normalize arrival_times_object list into consistent [(eq, (val, ts))]."""
    norm = []
    for eq, (val, ts) in arrival_list:
        norm.append((str(eq), (float(val), pd.to_datetime(ts))))
    # Sort by time for downstream logic
    norm.sort(key=lambda x: x[1][1])
    return norm


def _normalize_grouped_event(
    grouped_events: Dict[str, Dict[str, pd.DataFrame]],
    event_id: str,
    method: str = "poly_peak",
    *,
    poly_degree: int = 2,
    poly_samples: int = 1000,
) -> List[Tuple[str, Tuple[float, pd.Timestamp]]]:
    """
    Convert one grouped event (mapping equipment_id -> segment df) to a list of
    (equipment_id, (peak_val, peak_ts)). Supported methods:
      - "peak": raw max within the segment
      - "start": first timestamp in the segment
      - "poly_peak": fitted polynomial peak within the segment
    """
    segs = grouped_events[event_id]
    items: List[Tuple[str, Tuple[float, pd.Timestamp]]] = []

    for eq, seg_df in segs.items():
        if seg_df is None or len(seg_df) == 0:
            continue
        df = seg_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "value"]).sort_values("datetime")
        if len(df) == 0:
            continue

        if method == "start":
            ts = df["datetime"].iloc[0]
            val = float(df["value"].iloc[0])
        elif method == "poly_peak":
            val, ts = _poly_segment_peak(
                df,
                value_col="value",
                time_col="datetime",
                degree=poly_degree,
                n_dense=poly_samples,
            )
        else:  # "peak"
            idx = df["value"].idxmax()
            val = float(df.loc[idx, "value"])
            ts = pd.to_datetime(df.loc[idx, "datetime"])  

        items.append((str(eq), (float(val), pd.to_datetime(ts))))

    items.sort(key=lambda x: x[1][1])  # sort by time
    return items

# --------------------------- Surface + Descent -----------------------------

def _interpolate_surface_with_fallback(x_vals, y_vals, z_vals, grid_n: int = 100):
    """Interpolate arrival surface with cubic→linear blend inside hull; NaN outside hull."""
    xi = np.linspace(x_vals.min(), x_vals.max(), grid_n)
    yi = np.linspace(y_vals.min(), y_vals.max(), grid_n)
    XI, YI = np.meshgrid(xi, yi)

    # Convex hull mask
    hull = Delaunay(np.column_stack([x_vals, y_vals]))
    pts = np.column_stack([XI.ravel(), YI.ravel()])
    in_hull = hull.find_simplex(pts) >= 0
    in_hull_mask = in_hull.reshape(XI.shape)

    # Interpolations
    zi_cubic  = griddata((x_vals, y_vals), z_vals, (XI, YI), method="cubic")
    zi_linear = griddata((x_vals, y_vals), z_vals, (XI, YI), method="linear")

    zi = zi_cubic.copy()
    nan_inside = np.isnan(zi) & in_hull_mask
    zi[nan_inside] = zi_linear[nan_inside]

    # Outside hull remains NaN
    zi[~in_hull_mask] = np.nan

    return xi, yi, zi

# (rest of the file unchanged, same gradient descent, plotting, main loop)



def _random_point_in_hull(hull: Delaunay) -> Tuple[float, float]:
    """Sample a random point inside the Delaunay hull by triangle area weighting."""
    simplices = hull.simplices
    pts = hull.points
    tri = pts[simplices]  # (ntri, 3, 2)
    areas = 0.5 * np.abs(
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    )
    areas = np.where(areas <= 0, 1e-12, areas)
    idx = np.random.choice(len(simplices), p=areas / areas.sum())
    a, b, c = tri[idx]
    r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    p = a + r1 * (b - a) + r2 * (c - a)
    return float(p[0]), float(p[1])


def multistart_gradient_descent(
    f_interp,
    xi,
    yi,
    hull: Delaunay,
    *,
    num_starts: int = 30,
    learning_rate: float = 5.0,
    max_iter: int = 800,
    grad_tol: float = 1e-4,
    max_step: float = 20.0,
    step_shrink: float = 0.5,
    min_step: float = 1e-3,
):
    def numerical_gradient(x, y, h: float = 1.0):
        probes = np.array([[x + h, y], [x - h, y], [x, y + h], [x, y - h]])
        if np.any(hull.find_simplex(probes) < 0):
            return np.array([np.nan, np.nan])
        fxh1, fxh2, fyh1, fyh2 = f_interp(probes)
        if np.any(np.isnan([fxh1, fxh2, fyh1, fyh2])):
            return np.array([np.nan, np.nan])
        return np.array([(fxh1 - fxh2) / (2 * h), (fyh1 - fyh2) / (2 * h)])

    all_paths, starts, ends = [], [], []
    results = []  # (val, (x,y), path, start_idx)

    for si in range(num_starts):
        x, y = _random_point_in_hull(hull)
        starts.append((x, y))
        path = [(x, y)]

        for _ in range(max_iter):
            g = numerical_gradient(x, y)
            if np.any(np.isnan(g)):
                break
            gnorm = float(np.linalg.norm(g))
            if gnorm < grad_tol:
                break

            step = min(learning_rate, max_step / max(gnorm, 1e-12))
            ok = False
            while step >= min_step:
                xn = x - step * g[0]
                yn = y - step * g[1]
                if hull.find_simplex([[xn, yn]]) >= 0 and not np.isnan(f_interp([[xn, yn]])[0]):
                    ok = True
                    break
                step *= step_shrink
            if not ok:
                break

            x, y = xn, yn
            path.append((x, y))

        val = f_interp([[x, y]])[0]
        all_paths.append(path)
        ends.append((x, y))
        if not np.isnan(val):
            results.append((val, (x, y), path, si))

    if not results:
        print("All descent attempts failed.")
        return None

    best = min(results, key=lambda t: t[0])
    best_val, best_xy, best_path, best_start_idx = best

    return {
        "all_paths": [np.array(p) for p in all_paths],
        "start_points": starts,
        "end_points": ends,
        "best_min_xy": best_xy,
        "best_min_val": best_val,
        "best_start_idx": best_start_idx,
    }


def run_gradient_descent_on_surface(
    equipment_data: List[Tuple[str, Tuple[float, pd.Timestamp]]],
    facility_dict: Dict[str, Dict[str, Any]],
    event_id: str = "Event",
):
    # Reference point: first equipment in list
    ref_id = equipment_data[0][0]
    ref_lat = facility_dict[ref_id]["lat"]
    ref_lon = facility_dict[ref_id]["lon"]

    # Build point arrays
    min_time = min(ts for _, (_, ts) in equipment_data)
    x_vals, y_vals, z_vals = [], [], []
    for eq_id, (_, ts) in equipment_data:
        if eq_id not in facility_dict or facility_dict[eq_id]["lat"] is None:
            continue
        lat, lon = facility_dict[eq_id]["lat"], facility_dict[eq_id]["lon"]
        x_m, y_m = latlon_to_meters(lat, lon, ref_lat, ref_lon)
        x_vals.append(x_m)
        y_vals.append(y_m)
        z_vals.append((pd.to_datetime(ts) - pd.to_datetime(min_time)).total_seconds())

    x_vals = np.array(x_vals); y_vals = np.array(y_vals); z_vals = np.array(z_vals)

    if len(x_vals) < 3:
        raise ValueError("Not enough points to interpolate a surface (need >= 3 sensors).")

    # Interpolate with robust fallback
    xi, yi, zi_grid = _interpolate_surface_with_fallback(x_vals, y_vals, z_vals, grid_n=100)

    f_interp = RegularGridInterpolator((xi, yi), zi_grid.T, bounds_error=False, fill_value=np.nan)

    # Delaunay hull for in-bounds checks & random starts
    hull = Delaunay(np.column_stack([x_vals, y_vals]))

    multi_result = multistart_gradient_descent(
        f_interp, xi, yi, hull,
        num_starts=30, learning_rate=5, max_iter=800, grad_tol=1e-4, max_step=20,
    )
    if multi_result is None:
        raise RuntimeError("Gradient descent failed to find a valid minimum.")

    best_idx = multi_result.get("best_start_idx", 0)
    path = multi_result["all_paths"][best_idx]
    x_min, y_min = multi_result["best_min_xy"]

    # Clamp to grid bounds for safety
    x_min = float(np.clip(x_min, xi[0], xi[-1]))
    y_min = float(np.clip(y_min, yi[0], yi[-1]))

    lat_min, lon_min = meters_to_latlon(x_min, y_min, ref_lat, ref_lon)

    return {
        "path": path,
        "meters_origin": (x_min, y_min),
        "latlon_origin": (lat_min, lon_min),
        "ref_latlon": (ref_lat, ref_lon),
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "event_id": event_id,
        "zi_grid": zi_grid,
        "xi_grid": xi,
        "yi_grid": yi,
        "multi_result": multi_result,
    }

# ---------------------------  Plot Gradient Descent Path on Arrival Surface -----------------------------

def plot_gradient_descent_surface(result_dict, equipment_data):
    zi = result_dict["zi_grid"]
    xi = result_dict["xi_grid"]
    yi = result_dict["yi_grid"]
    x_vals = result_dict["x_vals"]
    y_vals = result_dict["y_vals"]
    event_id = result_dict["event_id"]

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(xi, yi, zi, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Arrival Time (s)")

    # Multistart descent overlay
    multi = result_dict.get("multi_result")
    if multi:
        cmap = cm.get_cmap("tab10", len(multi["all_paths"]))
        for i, path in enumerate(multi["all_paths"]):
            path = np.array(path)
            color = cmap(i)
            ax.plot(path[:, 0], path[:, 1], '-', color=color, linewidth=2, alpha=0.9)
            ax.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=4)
            ax.plot(path[-1, 0], path[-1, 1], 'x', color=color, markersize=5)

        best_x, best_y = multi["best_min_xy"]
        ax.scatter(best_x, best_y, s=180, facecolors='none', edgecolors='yellow', linewidths=2)

    # Plot sensor locations and labels
    for (sid, _), x, y in zip(equipment_data, x_vals, y_vals):
        ax.plot(x, y, 'ko', markersize=4)
        ax.text(x, y, sid, fontsize=8, ha='left', va='bottom')

    ax.set_title(f"Gradient Descent — {event_id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    plt.tight_layout()

    # --- Save to nested subfolders by facility + date ---
    save_path = _plot_path_for_event(event_id, equipment_data, facility_id)
    plt.savefig(save_path, dpi=300)
    print(f"[SAVED] {save_path}")
    plt.show()

# ------------------------------------------------------ RUN -------------------------------------------------

if __name__ == "__main__":
    # Load facility metadata
    with open(f"{facility_id}_facility_dict.json", "r") as f:
        facility_dict = json.load(f)

    has_grouped = os.path.exists("grouped_events.pkl")
    has_arrival = os.path.exists("arrival_times_object.pkl")

    if not has_grouped and not has_arrival:
        raise FileNotFoundError("Neither grouped_events.pkl nor arrival_times_object.pkl found.")

    # CASE A: iterate all grouped events
    if has_grouped:
        with open("grouped_events.pkl", "rb") as f:
            grouped_events = pickle.load(f)  # dict[event_id -> {eq -> seg_df}}

        # Choose arrival method and parameters
        arrival_method = "poly_peak"  # "peak" | "start" | "poly_peak"
        poly_degree = 2
        poly_samples = 1000
        min_sensors = 3  # need at least 3 sensors to form a surface

        for event_id in sorted(grouped_events.keys()):
            seg_map = grouped_events[event_id]
            if not seg_map or len(seg_map) < min_sensors:
                print(f"[SKIP] {event_id}: Not enough sensors ({len(seg_map) if seg_map else 0}).")
                continue
            equipment_data = _normalize_grouped_event(
                grouped_events,
                event_id,
                method=arrival_method,
                poly_degree=poly_degree,
                poly_samples=poly_samples,
            )
            if len(equipment_data) < min_sensors:
                print(f"[SKIP] {event_id}: <{min_sensors} usable arrivals after cleaning.")
                continue

            try:
                result = run_gradient_descent_on_surface(equipment_data, facility_dict, event_id=event_id)
            except Exception as e:
                print(f"[FAIL] {event_id}: {e}")
                continue

            print(f"[OK] {event_id}: Estimated origin (lat, lon) = {result['latlon_origin']}")
            upsert_event_origin_csv(result, equipment_data, event_id, facility_id)
            plot_gradient_descent_surface(result, equipment_data)

    # CASE B: single event from SavGol peaks (legacy path)
    elif has_arrival:
        with open("arrival_times_object.pkl", "rb") as f:
            arrival_times_object = pickle.load(f)  # list[(eq, (val, ts))]
        equipment_data = _normalize_arrival_times(arrival_times_object)

        result = run_gradient_descent_on_surface(equipment_data, facility_dict, event_id="E1")
        print("Estimated origin (lat, lon):", result["latlon_origin"])
        #upsert_event_origin_csv(result, equipment_data, event_id, facility_id)
        plot_gradient_descent_surface(result, equipment_data)
