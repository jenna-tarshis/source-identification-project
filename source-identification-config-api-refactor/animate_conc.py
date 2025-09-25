# animate_conc.py with wind overlay

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.interpolate import griddata
from config import facility_id, pollutant, data_dir, start_dt, end_dt

# --- Load facility metadata ---
facility_file = f"{facility_id}_facility_dict.json"
with open(facility_file, "r") as f:
    facility = json.load(f)

# --- Extract PM-2.5 sensor positions ---
sensor_positions = {}
for equip_id_str, meta in facility.items():
    equip_id = int(equip_id_str)
    if meta["lat"] is None or meta["lon"] is None:
        continue
    sensors = meta.get("sensors", {})
    if pollutant in sensors:
        sensor_positions[equip_id] = {
            "lat": meta["lat"],
            "lon": meta["lon"]
        }

# --- Load time series data from CSVs ---
dfs = []
for equip_id in sensor_positions:
    csv_path = os.path.join(data_dir, f"{facility_id}_{equip_id}_{pollutant}.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Missing data for equipment {equip_id}")
        continue
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df["equipment"] = equip_id
    dfs.append(df)

if not dfs:
    raise RuntimeError("No valid data files found.")

all_data = pd.concat(dfs)
all_data = all_data[(all_data["datetime"] >= pd.to_datetime(start_dt)) &
                    (all_data["datetime"] <= pd.to_datetime(end_dt))]

# --- Pivot to wide format ---
pivot = all_data.pivot(index="datetime", columns="equipment", values="value")
pivot = pivot.sort_index().interpolate().fillna(method="bfill").fillna(method="ffill")

# --- Load weather data ---
weather_df = pd.read_csv("weather_time_series.csv", parse_dates=["datetime"])
weather_df = weather_df.set_index("datetime").sort_index()
weather_df["wind_rad"] = np.deg2rad(weather_df["wind_direction_deg"])
weather_df["U"] = -weather_df["wind_speed_ms"] * np.sin(weather_df["wind_rad"])
weather_df["V"] = -weather_df["wind_speed_ms"] * np.cos(weather_df["wind_rad"])

# --- Main plot setup ---
equip_ids = pivot.columns
x = np.array([sensor_positions[e]["lon"] for e in equip_ids])
y = np.array([sensor_positions[e]["lat"] for e in equip_ids])
vmin = pivot.min().min()
vmax = pivot.max().max()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # room for slider and button

# Create grid for interpolation
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# Initialize surface plot with dummy data
zi = griddata((x, y), pivot.iloc[0].values, (xi, yi), method='linear')
surface = ax.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap="inferno", vmin=vmin, vmax=vmax, aspect='auto')
cbar = plt.colorbar(surface, ax=ax)
cbar.set_label(f"{pollutant} Concentration")

# Overlay sensor points
sc = ax.scatter(x, y, c=pivot.iloc[0].values, cmap="inferno", s=100, vmin=vmin, vmax=vmax, edgecolor='k')

# Annotate each sensor with its equipment ID
for i, equip_id in enumerate(equip_ids):
    ax.text(x[i] + 0.0001, y[i], str(equip_id), fontsize=8, ha="left", va="center")

# Wind vector placeholder (center of plot, thinner arrow)
center_x = (x.min() + x.max()) / 2
center_y = (y.min() + y.max()) / 2
wind_quiver = ax.quiver(center_x, center_y, 0, 0, color="cyan", scale=30, width=0.002)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# --- Draw frame ---
timestamp_list = pivot.index.to_list()
frame_idx = [0]  # use list to make mutable in closures


def draw_frame(i):
    timestamp = timestamp_list[i]
    values = pivot.iloc[i].values

    # Update interpolated surface
    zi = griddata((x, y), values, (xi, yi), method='linear')
    surface.set_data(zi)

    # Update sensor point colors
    sc.set_array(values)

    # Get matching wind vector (if available)
    if timestamp in weather_df.index:
        u = weather_df.loc[timestamp, "U"]
        v = weather_df.loc[timestamp, "V"]
        wind_quiver.set_UVC(u, v)

    ax.set_title(f"{pollutant} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
    fig.canvas.draw_idle()

# --- Initial frame ---
draw_frame(frame_idx[0])

# --- Slider widget ---
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(timestamp_list) - 1, valinit=0, valstep=1)


def on_slider_change(val):
    frame_idx[0] = int(val)
    draw_frame(frame_idx[0])

slider.on_changed(on_slider_change)

# --- Play/Pause Button ---
ax_button = plt.axes([0.8, 0.1, 0.1, 0.04])
button = Button(ax_button, 'Play')
playing = [False]


def toggle_play(event):
    playing[0] = not playing[0]
    button.label.set_text("Pause" if playing[0] else "Play")

button.on_clicked(toggle_play)

# --- Animation loop via timer ---

def update_loop(event):
    if playing[0]:
        frame_idx[0] = (frame_idx[0] + 1) % len(timestamp_list)
        slider.set_val(frame_idx[0])


# Start animation timer

timer = fig.canvas.new_timer(interval=300)
timer.add_callback(update_loop, None)
timer.start()

plt.show()

# --- Optional: Save animation ---
# ani.save("pm25_animation.mp4", writer="ffmpeg", fps=5)
