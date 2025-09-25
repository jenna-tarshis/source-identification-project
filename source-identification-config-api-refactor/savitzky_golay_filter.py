# smooth_peaks.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema

from config import facility_id, pollutant, data_dir, start_dt, end_dt

# ----------------------------------------------------------
# Process a single sensor file and extract peak
# ----------------------------------------------------------
def extract_smoothed_peak(df, window_length=5, polyorder=2):
    df = df.dropna(subset=["value"])
    if len(df) < window_length:
        return None, None

    signal = df["value"].values
    time_array = df["datetime"].values

    smoothed = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
    peak_indices = argrelextrema(smoothed, comparator=np.greater)[0]

    if len(peak_indices) == 0:
        return None, None

    peak_idx = peak_indices[np.argmax(smoothed[peak_indices])]
    return smoothed[peak_idx], time_array[peak_idx]

# ----------------------------------------------------------
# Main routine: loop through all sensors
# ----------------------------------------------------------
all_sensor_dfs = {}
peak_results = {}
WINDOW_LENGTH = 20
POLY_ORDER = 2

sensor_file = f"{facility_id}_facility_dict.json"
with open(sensor_file, "r") as f:
    facility = json.load(f)

for equipment in facility:
    fname = f"{data_dir}/{facility_id}_{equipment}_{pollutant}.csv"
    if not os.path.exists(fname):
        continue

    try:
        df = pd.read_csv(fname, parse_dates=["datetime"])
        df = df[(df["datetime"] >= pd.to_datetime(start_dt)) &
                (df["datetime"] <= pd.to_datetime(end_dt))]
        df = df.sort_values("datetime").reset_index(drop=True)
        all_sensor_dfs[equipment] = df

        peak_val, peak_time = extract_smoothed_peak(df, WINDOW_LENGTH, POLY_ORDER)
        if peak_time is not None:
            peak_results[equipment] = (peak_val, peak_time)

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue

# Sort by time of peak
peak_results_sorted = sorted(peak_results.items(), key=lambda x: x[1][1])

# ----------------------------------------------------------
# Overlay plot of smoothed signals
# ----------------------------------------------------------
plt.figure(figsize=(12, 6))
colors = plt.cm.get_cmap('tab10', len(all_sensor_dfs))

for idx, (sensor, df) in enumerate(all_sensor_dfs.items()):
    if len(df) < 5:
        continue
    signal = df["value"].values
    time_array = df["datetime"].values
    smoothed = savgol_filter(signal, WINDOW_LENGTH, POLY_ORDER)

    plt.plot(time_array, smoothed, label=f"{sensor}", color=colors(idx))

    if sensor in peak_results:
        plt.scatter(peak_results[sensor][1], peak_results[sensor][0],
                    color=colors(idx), marker="x")

plt.title("Signal Overlay and Peaks (Savitzky-Golay)")
plt.xlabel("Time")
plt.ylabel(f"{pollutant} Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Save objects
# ----------------------------------------------------------
with open("arrival_times_object.pkl", "wb") as f:
    pickle.dump(peak_results_sorted, f)

print("Smoothed peak extraction complete. Results saved.")
