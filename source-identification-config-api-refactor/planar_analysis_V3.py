# planar_analysis_V3.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from numpy.polynomial.polynomial import Polynomial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.ion()

import pickle
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

from analysis_utils import *
from config import *

def fit_poly_and_find_peak(df, degree=2):
    x = df['time_seconds'].values
    y = df['PM-2.5'].values
    if len(x) < degree + 1:
        max_val = df['PM-2.5'].max()
        time_of_max = df[df['PM-2.5'] == max_val]['time_seconds'].iloc[0]
        return max_val, time_of_max

    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = poly(x_dense)
    idx_max = np.argmax(y_dense)
    return y_dense[idx_max], x_dense[idx_max]

def get_maxima(event_data):
    max_vals = {}
    for sensor, df in event_data.items():
        max_val, time_of_max = fit_poly_and_find_peak(df)
        max_vals[sensor] = (max_val, time_of_max)
    return sorted(max_vals.items(), key=lambda x: x[1][1])

def triangle_internal_angles(coords):
    a, b, c = coords
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)

    def angle(opposite, side1, side2):
        return np.degrees(np.arccos((side1**2 + side2**2 - opposite**2) / (2 * side1 * side2)))
    
    angle_A = angle(bc, ab, ca)
    angle_B = angle(ca, ab, bc)
    angle_C = angle(ab, bc, ca)
    return angle_A, angle_B, angle_C
def is_valid_triangle(coords, min_angle=20, max_angle=110):
    angles = triangle_internal_angles(coords)
    return all(min_angle <= angle <= max_angle for angle in angles)

def analyze_triangle_from_threshold(args):
    tri, peak_times, ref_sensor = args
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["May25June22"][ref_sensor])
    coords = []
    times = []

    for sid in tri:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["May25June22"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        coords.append((x, y))
        times.append(peak_times[sid])

    coords = np.array(coords)
    if not is_valid_triangle(coords):
        return None

    vec, area = fit_plane_and_get_gradient(coords, times)
    if vec is None or area < 50:
        return None

    return (tri, vec, coords, area)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    min_triangle_area = 50
    scale = 50

    for event_id, event_data in grouped_events.items():
        ordered_max_vals = get_maxima(event_data)
        sensors_in_event = [sensor for sensor, _ in ordered_max_vals]
        peak_times = {sensor: t_max for sensor, (_, t_max) in ordered_max_vals}
        ref_sensor = sensors_in_event[0]

        # Prepare triangle jobs
        triangle_args = [(tri, peak_times, ref_sensor) for tri in combinations(sensors_in_event, 3)]

        # Run in parallel
        with Pool() as pool:
            results = pool.map(analyze_triangle_from_threshold, triangle_args)

        vectors = [res for res in results if res is not None]

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
        lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["May25June22"][ref_sensor])
        for sid in event_data:
            lat, lon = get_sensor_position(facility_sensors["AltEn"]["May25June22"][sid])
            x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
            plt.plot(x, y, 'ko')
            plt.text(x, y, sid, fontsize=8)

        plt.title(f"{event_id} — Arrival Times of Peak Concentration at Participating Sensors")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()

    plt.ioff()
    plt.show()

"""
# 3D ---------------------------------------------------------------------
# Parameters
min_triangle_area = 50
scale = 50
use_steepness_as_color = False

for event_id, event_data in grouped_events.items():
    ordered_max_vals = get_maxima(event_data)
    sensors_in_event = [sensor for sensor, _ in ordered_max_vals]
    peak_times = {sensor: t_max for sensor, (_, t_max) in ordered_max_vals}

    x_vals, y_vals, z_vals = [], [], []
    ref_sensor = sensors_in_event[0]
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"][ref_sensor])

    for sid in sensors_in_event:
        lat, lon = get_sensor_position(facility_sensors["AltEn"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(peak_times[sid])

    x_vals, y_vals, z_vals = np.array(x_vals), np.array(y_vals), np.array(z_vals)

    xi = np.linspace(min(x_vals), max(x_vals), 100)
    yi = np.linspace(min(y_vals), max(y_vals), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')

    dz_dx, dz_dy = np.gradient(zi, xi[0, :], yi[:, 0])
    steepness = np.sqrt(dz_dx**2 + dz_dy**2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if use_steepness_as_color:
        facecolors = cm.inferno(steepness / np.nanmax(steepness))
        colorbar_vals = steepness
        cbar_label = 'Steepness (s/m)'
        cmap = cm.inferno
    else:
        normed_zi = (zi - np.nanmin(zi)) / (np.nanmax(zi) - np.nanmin(zi))
        facecolors = cm.viridis(normed_zi)
        colorbar_vals = zi
        cbar_label = 'Arrival Time (s)'
        cmap = cm.viridis

    surf = ax.plot_surface(xi, yi, zi, facecolors=facecolors, rstride=1, cstride=1,
                           antialiased=True, linewidth=0, shade=False)
    ax.scatter(x_vals, y_vals, z_vals, color='r', s=50, label='Sensors')
    # Label each sensor
    for sid, x, y, z in zip(sensors_in_event, x_vals, y_vals, z_vals):
        ax.text(x, y, z, sid, fontsize=8, color='black')

    ax.set_title(f"{event_id} — Peak Concentration Arrival Surface")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Time (s)")

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(colorbar_vals)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label=cbar_label)

    plt.tight_layout()
    plt.savefig(f"{event_id}_peak_3Dsurface.png")


def plot_poly_fit(sensor_id, df, degree=4):
    #Plots the raw PM-2.5 data and the fitted polynomial for a given sensor's dataframe.
    
    x = df['time_seconds'].values
    y = df['PM-2.5'].values

    # Fit polynomial
    if len(x) < degree + 1:
        print(f"Not enough data points to fit a degree-{degree} polynomial.")
        return

    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)

    # Dense range for smooth curve
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = poly(x_dense)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o', label='Raw Data', alpha=0.6)
    plt.plot(x_dense, y_dense, '-', label=f'Polynomial Fit (deg {degree})')
    plt.title(f"Polynomial Fit for Sensor {sensor_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("PM-2.5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_poly_fit("S8", event_data["S8"])

plt.ioff()
plt.show()

"""