import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from numpy.polynomial.polynomial import Polynomial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import pickle
from scipy.interpolate import RegularGridInterpolator

from analysis_utils import *
from config import *

# FIT 2 DEGREE POLYNOMIAL TO ANOMALOUS REGION + FIND PEAK
def fit_poly_and_find_peak(df, degree=2):
    x = df['time_seconds'].values
    y = df[pollutant].values
    if len(x) < degree + 1:
        max_val = df[pollutant].max()
        time_of_max = df[df[pollutant] == max_val]['time_seconds'].iloc[0]
        return max_val, time_of_max

    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = poly(x_dense)
    idx_max = np.argmax(y_dense)
    return y_dense[idx_max], x_dense[idx_max]

# APPLY fit_poly_and_find_peak() TO GET MAX AND TIME OF MAX
def get_maxima(event_data):
    max_vals = {}
    for sensor, df in event_data.items():
        max_val, time_of_max = fit_poly_and_find_peak(df)
        max_vals[sensor] = (max_val, time_of_max)
    return sorted(max_vals.items(), key=lambda x: x[1][1])

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_polynomial_fits(event_id, event_data, degree=2):
    """
    Plots PM2.5 time series for all sensors in the event with polynomial fits and their maxima.
    Uses a unique color per sensor for clarity.
    """
    plt.figure(figsize=(12, 6))
    
    sensor_ids = list(event_data.keys())
    colors = plt.cm.get_cmap('tab10', len(sensor_ids))  # color palette

    for idx, (sensor, df) in enumerate(event_data.items()):
        x = df['time_seconds'].values
        y = df[pollutant].values
        color = colors(idx)

        # Plot raw data (faint)
        plt.plot(df['Datetime'], y, linestyle='--', color=color, alpha=0.4)

        if len(x) < degree + 1:
            max_val = df[pollutant].max()
            time_of_max = df.loc[df[pollutant].idxmax(), 'Datetime']
            plt.scatter(time_of_max, max_val, color=color, marker='x', label=f"{sensor} peak")
            continue

        # Fit polynomial and evaluate densely
        coefs = np.polyfit(x, y, degree)
        poly = np.poly1d(coefs)
        x_dense = np.linspace(x.min(), x.max(), 1000)
        y_dense = poly(x_dense)
        idx_max = np.argmax(y_dense)
        max_val = y_dense[idx_max]
        time_of_max = pd.to_datetime(np.interp(x_dense[idx_max], x, df['Datetime'].astype(np.int64) // 10**9), unit='s')

        # Plot polynomial fit (bold)
        time_dense = pd.to_datetime(np.interp(x_dense, x, df['Datetime'].astype(np.int64) // 10**9), unit='s')
        plt.plot(time_dense, y_dense, color=color, label=f"{sensor} fit")

        # Plot max point
        plt.scatter(time_of_max, max_val, color=color, marker='x')

    plt.title(f"Polynomial Fits and Maxima for Event {event_id}")
    plt.xlabel("Datetime")
    plt.ylabel("PM-2.5 Concentration")
    plt.grid(True)
    plt.tight_layout()

    # Make legend show only 1 entry per sensor
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    clean_handles_labels = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l.split()[0]))]
    plt.legend(*zip(*clean_handles_labels))

    plt.show()

# COSINE LAW TO DETERMINE INTERNAL ANGLES OF TRIANGLE
def triangle_internal_angles(coords):
    a, b, c = coords
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    def angle(opposite, side1, side2):
        return np.degrees(np.arccos((side1**2 + side2**2 - opposite**2) / (2 * side1 * side2)))
    return angle(bc, ab, ca), angle(ca, ab, bc), angle(ab, bc, ca)

# CONSTRAINT ON INTERNAL ANGLES
def is_valid_triangle(coords, min_angle=45, max_angle=135):
    return all(min_angle <= angle <= max_angle for angle in triangle_internal_angles(coords))

# FIND SENSOR COORDINATES, CHECK TRI VALIDITY, FIT PLANE + GRADIENT
def analyze_triangle(args):
    tri, peak_times, ref_sensor = args
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][ref_sensor])
    coords = []
    times = []
    for sid in tri:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        coords.append((x, y))
        times.append(peak_times[sid])
    coords = np.array(coords)
    if not is_valid_triangle(coords):
        return None
    vec, area = fit_plane_and_get_gradient(coords, times)   # analysis_utils.py
    if vec is None or area < 50:
        return None
    return tri, vec, coords, area

#-------------------------------------------------------------------------------------------------------------------------------

def plot_2d_vectors(event_id, vectors, event_data, ref_sensor, scale=2000):
    plt.figure(figsize=(8, 8))
    all_x, all_y = [], []

    for tri, vec, coords, area in vectors:
        centroid = np.mean(coords, axis=0)
        # Normalize and scale vector
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        norm_vec = vec / norm
        scaled_vec = norm_vec * scale
        
        # Collect vector start and end for bounds
        all_x.extend([centroid[0], centroid[0] + scaled_vec[0]])
        all_y.extend([centroid[1], centroid[1] + scaled_vec[1]])

        plt.quiver(centroid[0], centroid[1], scaled_vec[0], scaled_vec[1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7, width=0.002)

        triangle_path = np.vstack([coords, coords[0]])
        plt.plot(triangle_path[:, 0], triangle_path[:, 1], 'k--', alpha=0.3)

    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][ref_sensor])
    for sid in event_data:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        plt.plot(x, y, 'ko')
        plt.text(x, y, sid, fontsize=8)
        all_x.append(x)
        all_y.append(y)

    # Set axis limits to show all vector tips
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    margin_x = 0.1 * (x_max - x_min)
    margin_y = 0.1 * (y_max - y_min)

    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)

    plt.title(f"{event_id} — 2D Gradient Vectors")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

#-------------------------------------------------------------------------------------------------------------------------------

def plot_3d_surface(event_id, event_data, peak_times, ref_sensor, use_steepness=False):
    x_vals, y_vals, z_vals = [], [], []
    for sid in peak_times:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][sid])
        x, y = latlon_to_meters(lat, lon, *get_sensor_position(facility_sensors["AltEn"]["Feb2024"][ref_sensor]))
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(peak_times[sid])
    x_vals, y_vals, z_vals = np.array(x_vals), np.array(y_vals), np.array(z_vals)
    # Skip if there are fewer than 4 points
    if len(x_vals) < 4:
        print(f"Skipping 3D plot for event {event_id} (not enough points: {len(x_vals)})")
        return

    xi, yi = np.linspace(x_vals.min(), x_vals.max(), 100), np.linspace(y_vals.min(), y_vals.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')
    dz_dx, dz_dy = np.gradient(zi, xi[0, :], yi[:, 0])
    steepness = np.sqrt(dz_dx**2 + dz_dy**2)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if use_steepness:
        norm = steepness / np.nanmax(steepness)
        facecolors = cm.inferno(norm)
        cmap = cm.inferno
        colorbar_vals = steepness
        cbar_label = 'Steepness (s/m)'
    else:
        norm = (zi - np.nanmin(zi)) / (np.nanmax(zi) - np.nanmin(zi))
        facecolors = cm.viridis(norm)
        cmap = cm.viridis
        colorbar_vals = zi
        cbar_label = 'Arrival Time (s)'

    surf = ax.plot_surface(xi, yi, zi, facecolors=facecolors, rstride=1, cstride=1, antialiased=True, linewidth=0, shade=False)
    ax.scatter(x_vals, y_vals, z_vals, color='r', s=50, label='Sensors')
    for sid, x, y, z in zip(peak_times.keys(), x_vals, y_vals, z_vals):
        ax.text(x, y, z, sid, fontsize=8, color='black')
    ax.set_title(f"{event_id} — 3D Arrival Surface")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Time (s)")
    ax.view_init(elev=90, azim=-90)  # Top-down view
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(colorbar_vals)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label=cbar_label)
    plt.tight_layout()
    #plt.savefig(f"{event_id}_3D_surface.png")

#-------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    with open("grouped_events.pkl", "rb") as f:
        grouped_events = pickle.load(f)

    #event_id = "E1"  
    #event_data = grouped_events[event_id]

    for event_id, event_data in grouped_events.items():
        print(event_id)

        ordered_max_vals = get_maxima(event_data)
        sensors_in_event = [sensor for sensor, _ in ordered_max_vals]
        peak_times = {sensor: t_max for sensor, (_, t_max) in ordered_max_vals}
        ref_sensor = sensors_in_event[0]

        triangle_args = [(tri, peak_times, ref_sensor) for tri in combinations(sensors_in_event, 3)]
        with Pool() as pool:
            results = pool.map(analyze_triangle, triangle_args)
        vectors = [res for res in results if res is not None]

        #plot_2d_vectors(event_id, vectors, event_data, ref_sensor)
        plot_3d_surface(event_id, event_data, peak_times, ref_sensor, use_steepness=False)
        #plot_surface_gradient_vectors(event_id, peak_times, ref_sensor, scale=50)
        plot_polynomial_fits(event_id, event_data)
        
    plt.show()

