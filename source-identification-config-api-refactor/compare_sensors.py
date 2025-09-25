# CompareSensorsEVA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
plt.ion()

from plot_utils import plot_overlay_anomalies_for_all_sensors, plot_overlay_all_sensors

from config import (
    facility_id,
    pollutant,
    return_period_days,
    block_width_seconds,
    start_dt as START_DT,
    end_dt as END_DT,
)
from analysis_utils import (
    load_and_preprocess_api_csv,
    compute_block_maxima,
    assign_return_period_labels,
    get_anomaly_threshold,
    label_anomalies,
    extract_anomalous_segments,
    assign_event_groups,
    apply_rolling_avg,
)

# --------------- Output folder helpers (match surface_analysis) ---------------
PLOT_ROOT = "plots"
os.makedirs(PLOT_ROOT, exist_ok=True)

def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def _fmt_dt_for_dir(x) -> str:
    # "2025-06-01 00:00:00" -> "2025-06-01_0000"
    return pd.to_datetime(x).strftime("%Y-%m-%d_%H%M")

def _analysis_dir(pollutant: str, facility_id: str, start_dt: str, end_dt: str) -> str:
    """
    plots/<pollutant>/<facility_id>/<start>__<end>
    """
    d = os.path.join(
        PLOT_ROOT,
        _safe_name(pollutant),
        _safe_name(facility_id),
        f"{_fmt_dt_for_dir(start_dt)}__{_fmt_dt_for_dir(end_dt)}",
    )
    os.makedirs(d, exist_ok=True)
    return d

# --------------- Output grouped events function ---------------
def _write_grouped_events_csv(grouped_events: dict, out_dir: str, filename: str = "grouped events.csv"):
    """
    Flatten grouped_events to a CSV:
      Columns:
        event_id, event_start, event_end, n_sensors,
        equipment_id, rows, sensor_start, sensor_end
      One row per (event_id, equipment_id)
    """
    import csv

    rows = []
    for event_id, segments in grouped_events.items():
        # Collect per-sensor spans and compute event-wide span
        sensor_rows = []
        for eq, seg_df in (segments or {}).items():
            if seg_df is None or len(seg_df) == 0:
                continue
            dfi = seg_df.copy()
            dfi["datetime"] = pd.to_datetime(dfi["datetime"], errors="coerce")
            dfi = dfi.dropna(subset=["datetime"]).sort_values("datetime")
            if len(dfi) == 0:
                continue
            s_start = dfi["datetime"].iloc[0]
            s_end   = dfi["datetime"].iloc[-1]
            sensor_rows.append((str(eq), len(dfi), s_start, s_end))

        if not sensor_rows:
            # still emit an event line with no sensors if desired (skip by default)
            continue

        event_start = min(sr[2] for sr in sensor_rows)
        event_end   = max(sr[3] for sr in sensor_rows)
        n_sensors   = len(sensor_rows)

        # one CSV row per sensor with event-level metadata repeated
        for eq_id, n_rows, s_start, s_end in sensor_rows:
            rows.append({
                "event_id": event_id,
                "event_start": event_start.isoformat(),
                "event_end": event_end.isoformat(),
                "n_sensors": n_sensors,
                "equipment_id": eq_id,
                "rows": int(n_rows),
                "sensor_start": s_start.isoformat(),
                "sensor_end": s_end.isoformat(),
            })

    # Write CSV (overwrite on each run for this window)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, filename)
    fieldnames = ["event_id","event_start","event_end","n_sensors","equipment_id","rows","sensor_start","sensor_end"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVED] {csv_path}")



# --- LOAD FACILITY DICT to this file ---
sensor_file = f"{facility_id}_facility_dict.json"
if not os.path.exists(sensor_file):
    raise FileNotFoundError(f"Sensor metadata file {sensor_file} not found.")
with open(sensor_file, "r") as f:
    facility = json.load(f)
#print(json.dumps(facility, indent=4)) #pretty print

sensor_dfs = {}
for equipment in facility:
    fname = f"data/processed_samples/{facility_id}_{equipment}_{pollutant}.csv"
    print(fname)
    try:
        df = pd.read_csv(fname)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
        
        maxima_vals, _ = compute_block_maxima(df, "value", block_width_seconds)
        return_periods, sorted_maxima = assign_return_period_labels(maxima_vals, block_width_seconds)
        threshold = get_anomaly_threshold(sorted_maxima, return_periods, return_period_days)
        df = label_anomalies(df, "value", threshold)
        sensor_dfs[equipment] = df

    except Exception as e:
        print(f"Failed to load {fname}: {e}")
        continue

print(threshold)
print(sensor_dfs)

"""# --- Export grouped sensor dfs for downstream analysis ---
import pickle
with open("sensor_dfs_object.pkl", "wb") as f:
    pickle.dump(sensor_dfs, f)
"""
# --- Plot all sensors with anomalies ---
plot_overlay_anomalies_for_all_sensors(sensor_dfs, pollutant=pollutant, return_period_days=return_period_days)

# save plot to directory
fig = plot_overlay_anomalies_for_all_sensors(sensor_dfs, pollutant=pollutant, return_period_days=return_period_days)
# If the plotting util doesn't return a fig, fall back to current figure
if fig is None:
    fig = plt.gcf()

base = _analysis_dir(pollutant, facility_id, START_DT, END_DT)
overlay_path = os.path.join(base, "anomaly_overlay.png")
fig.savefig(overlay_path, dpi=300)
plt.close(fig)
print(f"[SAVED] {overlay_path}")


# --- Extract and group anomalous segments ---
all_anom_segments = {}
for label, df in sensor_dfs.items():
    anom_segments = extract_anomalous_segments(df, "Anomaly")
    all_anom_segments[label] = anom_segments
grouped_events = assign_event_groups(all_anom_segments)

# --- save grouped_events to .csv file ---
base = _analysis_dir(pollutant, facility_id, START_DT, END_DT)
_write_grouped_events_csv(grouped_events, base, filename="grouped events.csv")

# --- Print summary of Events ---
print("GROUPED EVENTS:")
for event_id, segments in grouped_events.items():
    print(f"\n{event_id} includes {len(segments)} sensors:")
    for label, seg_df in segments.items():
        print(f"  - {label}: {len(seg_df)} rows, {seg_df['datetime'].iloc[0]} → {seg_df['datetime'].iloc[-1]}")

# --- Export grouped events pkl ---
import pickle
with open("grouped_events.pkl", "wb") as f:
    pickle.dump(grouped_events, f)

plt.ioff()
plt.show()
