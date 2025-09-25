# planar_analysis_V2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.ion()

# import grouped events object from CompareSensorsEVA.py
import pickle
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

from analysis_utils import *
from config import *

#------------------------------------
# gets ref_conc from iterating peaks and looking forward in time
# handles more than 3 sensors by making groups of 3
# two main loops here: 2D plotting and 3D surface plotting
#------------------------------------

def get_valid_thresholds(event_data):
    """
    Iterates throgh event data (sensors and their anomalous segments) to find the 
    maximum and t_max of each sensor in the event.

    Returns a list of (threshold, other_sensors_dict) tuples.
    """
    max_vals = {}
    for sensor, df in event_data.items():
        max_val = df['PM-2.5'].max()
        time_of_max = df[df['PM-2.5'] == max_val]['time_seconds'].iloc[0]
        max_vals[sensor] = (max_val, time_of_max)

    # Sort sensors by max concentration, descending
    sorted_sensors = sorted(max_vals.items(), key=lambda x: x[1][0], reverse=True)
    candidate_thresholds = [(val[1][0], val[1][1]) for val in sorted_sensors]  # (threshold, time_of_peak)

    valid_thresholds = []   
    for threshold, time_ref in candidate_thresholds:  # for all the thresholds collected
        other_sensors = {}
        for sensor, df in event_data.items():           # iterate thruogh the sensors of the event
            df_above = df[(df['PM-2.5'] >= threshold) & (df['time_seconds'] >= time_ref)]    #check if that sensor hits the threshold
            if not df_above.empty:                      #if it does, add the time at which it does to other_sensors dict
                other_sensors[sensor] = df_above.iloc[0]['time_seconds']
        if len(other_sensors) >= 3:                     #only keep threshold if theres 3 or more sensors hitting it
            valid_thresholds.append((threshold, other_sensors))

    return valid_thresholds


def analyze_triangle_from_threshold(sensor_ids, other_sensors, ref_sensor):
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["May25June22"][ref_sensor])
    coords = []
    times = []

    for sid in sensor_ids:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["May25June22"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        coords.append((x, y))
        times.append(other_sensors[sid])

    coords = np.array(coords)
    vector, area = fit_plane_and_get_gradient(coords, times)
    return vector, coords, area

#----------------------------------------------------------------------------------
# Parameters
min_triangle_area = 50  # m², to filter out small/unreliable triangles
scale = 100  # scale up unit vector length for visibility

# Optional control to specify event and threshold index to analyze
selected_event = "E1"        # Set to None to process all events
selected_threshold_idx = 2   # Index starts at 0, set to None to process all thresholds in selected event
use_steepness_as_color = True  # Toggle between steepness (True) and arrival time (False)
"""
for event_id, event_data in grouped_events.items():
    if selected_event is not None and event_id != selected_event:
        continue

    valid_thresholds = get_valid_thresholds(event_data)
    print(event_id)
    print(valid_thresholds)
    event_vectors = []

    for i, (threshold, sensor_times) in enumerate(valid_thresholds):
        if selected_threshold_idx is not None and i != selected_threshold_idx:
            continue
        vectors = []
        eligible_sensors = list(sensor_times.keys())

        for tri in combinations(eligible_sensors, 3):
            vec, coords, area = analyze_triangle_from_threshold(tri, sensor_times, ref_sensor=eligible_sensors[0])
            if vec is None or area < min_triangle_area:
                continue
            vectors.append((tri, vec, coords, area))

        # Plotting
        plt.figure(figsize=(8, 8))
        for tri, vec, coords, area in vectors:
            centroid = np.mean(coords, axis=0)
            scaled_vec = vec * scale
            plt.quiver(centroid[0], centroid[1], scaled_vec[0], scaled_vec[1],
                       angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7, width=0.002)
            triangle_path = np.vstack([coords, coords[0]])
            plt.plot(triangle_path[:, 0], triangle_path[:, 1], 'k--', alpha=0.3)

        # Sensor dots
        ref_sensor = eligible_sensors[0]
        lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"][ref_sensor])
        for sid in event_data:
            lat, lon = get_sensor_position(facility_sensors["AltEn"][sid])
            x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
            plt.plot(x, y, 'ko')
            plt.text(x, y, sid, fontsize=8)

        plt.title(f"{event_id} — Plume Vectors at {threshold:.2f} µg/m³")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
"""
#------------------------------------------------------------------------------

# Process events
for event_id, event_data in grouped_events.items():
    if selected_event is not None and event_id != selected_event:
        continue

    valid_thresholds = get_valid_thresholds(event_data)
    print(event_id)
    print(valid_thresholds)

    for i, (threshold, sensor_times) in enumerate(valid_thresholds):
        if selected_threshold_idx is not None and i != selected_threshold_idx:
            continue

        vectors = []
        eligible_sensors = list(sensor_times.keys())

        x_vals, y_vals, z_vals = [], [], []
        ref_sensor = eligible_sensors[0]
        lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["May25June22"][ref_sensor])

        for sid in eligible_sensors:
            lat, lon = get_sensor_position(facility_sensors["AltEn"]["May25June22"][sid])
            x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(sensor_times[sid])

        x_vals, y_vals, z_vals = np.array(x_vals), np.array(y_vals), np.array(z_vals)

        # Create grid for surface
        xi = np.linspace(min(x_vals), max(x_vals), 100)
        yi = np.linspace(min(y_vals), max(y_vals), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')

        # Compute gradient and steepness
        dz_dx, dz_dy = np.gradient(zi, xi[0, :], yi[:, 0])
        steepness = np.sqrt(dz_dx**2 + dz_dy**2)

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if use_steepness_as_color:
            facecolors = cm.inferno(steepness / np.nanmax(steepness))
            colorbar_vals = steepness
            cbar_label = 'Steepness (s/m)'
        else:
            normed_zi = (zi - np.nanmin(zi)) / (np.nanmax(zi) - np.nanmin(zi))
            facecolors = cm.viridis(normed_zi)
            colorbar_vals = zi
            cbar_label = 'Arrival Time (s)'

        surf = ax.plot_surface(
            xi, yi, zi,
            facecolors=facecolors,
            rstride=1, cstride=1, antialiased=True, linewidth=0, shade=False
        )
        ax.scatter(x_vals, y_vals, z_vals, color='r', s=50, label='Sensors')

        ax.set_title(f"{event_id} — Arrival Time Surface at {threshold:.2f} µg/m³")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Time (s)")

        mappable = cm.ScalarMappable(cmap=cm.inferno if use_steepness_as_color else cm.viridis)
        mappable.set_array(colorbar_vals)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label=cbar_label)

        plt.tight_layout()
        plt.savefig(f"{event_id}_threshold_{i+1}_3Dsurface.png")

plt.ioff()
plt.show()



"""
# Parameters
min_triangle_area = 50  # m², to filter out small/unreliable triangles
all_vectors = {}  # Output container: {event_id: [(sensor_ids, vector, coords, area, threshold), ...]}
scale = 100  # scale up unit vector length for visibility

# Process each event
for event_id, event_data in grouped_events.items():
    valid_thresholds = get_valid_thresholds(event_data)
    event_vectors = []

    for threshold, other_sensors in valid_thresholds:
        eligible_sensors = list(other_sensors.keys())
        for tri in combinations(eligible_sensors, 3):
            vec, coords, area = analyze_triangle_from_threshold(tri, other_sensors, ref_sensor=eligible_sensors[0])
            if vec is None or area < min_triangle_area:
                continue
            event_vectors.append((tri, vec, coords, area, threshold))

        #all_vectors[event_id] = event_vectors

        # Plot 2D vector field for each event
        plt.figure(figsize=(8, 8))
        for tri, vec, coords, area, threshold in event_vectors:
            centroid = coords.mean(axis=0)
            scaled_vec = vec * scale
            plt.quiver(centroid[0], centroid[1], scaled_vec[0], scaled_vec[1],
                    angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7, width=0.002)
            triangle_path = np.vstack([coords, coords[0]])
            plt.plot(triangle_path[:, 0], triangle_path[:, 1], 'k--', alpha=0.3)

        # Plot sensor positions
        ref_sensor = list(event_data.keys())[0]
        lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"][ref_sensor])
        for sid in event_data:
            lat, lon = get_sensor_position(facility_sensors["AltEn"][sid])
            x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
            plt.plot(x, y, 'ko')
            plt.text(x, y, sid, fontsize=8)

        plt.title(f"Plume Vectors for {event_id}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"vector_field_{event_id}.png")

# Optional: export all vectors
with open("all_event_vectors.pkl", "wb") as f:
    pickle.dump(all_vectors, f)

plt.ioff()
plt.show()"""
