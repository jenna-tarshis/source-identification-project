import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import pickle
from itertools import combinations

from planar_analysis_V4 import get_sensor_position, latlon_to_meters, get_maxima
from config import facility_sensors

# ---- Load event data ----
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

# --- Constants ---
R = 6371000  # Earth radius in meters
global_ref_sensor = "S7"
lat_global, lon_global = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][global_ref_sensor])

# --- Gradient descent functions ---
def numerical_gradient(f, x, y, h=1.0):
    fxh1 = f([[x + h, y]])[0]
    fxh2 = f([[x - h, y]])[0]
    fyh1 = f([[x, y + h]])[0]
    fyh2 = f([[x, y - h]])[0]
    dfdx = (fxh1 - fxh2) / (2 * h)
    dfdy = (fyh1 - fyh2) / (2 * h)
    return np.array([dfdx, dfdy])

def gradient_descent(f, x_init, y_init, learning_rate=10, max_iter=100, tol=1e-3):
    x, y = x_init, y_init
    path = [(x, y)]
    for _ in range(max_iter):
        grad = numerical_gradient(f, x, y)
        x_new, y_new = x - learning_rate * grad[0], y - learning_rate * grad[1]
        path.append((x_new, y_new))
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        x, y = x_new, y_new
    return (x, y), path

def plot_gradient_descent_surface(event_id, x_vals, y_vals, z_vals, path, ref_sensor):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Interpolation grid
    xi = np.linspace(x_vals.min(), x_vals.max(), 100)
    yi = np.linspace(y_vals.min(), y_vals.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi = griddata((x_vals, y_vals), z_vals, (xi_grid, yi_grid), method='cubic')

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(xi_grid, yi_grid, zi, levels=50, cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Arrival Time (s)")

    path = np.array(path)

    if len(path) == 1 or np.allclose(path[0], path[-1], atol=1e-2):
        ax.plot(path[0, 0], path[0, 1], 'yo', markersize=8, label="Start = Local Min")
    else:
        ax.plot(path[:, 0], path[:, 1], 'r-o', label="Descent path")
        ax.plot(path[0, 0], path[0, 1], 'go', label="Start point")
        ax.plot(path[-1, 0], path[-1, 1], 'yo', label="Local Min")


    # Plot sensor positions
    for x, y in zip(x_vals, y_vals):
        ax.plot(x, y, 'ko')
    for sid, x, y in zip(list(peak_times.keys()), x_vals, y_vals):
        ax.text(x, y, sid, fontsize=8)

    ax.set_title(f"Gradient Descent on Arrival Time Surface — Event {event_id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_summary_estimated_origins(estimates, sensor_ids, sensor_dict, reference_sensor="S9"):
    """
    estimates: list of tuples (event_id, x_local, y_local, ref_sensor)
    sensor_ids: list of all sensors to show on the map
    sensor_dict: facility_sensors["AltEn"]["May25June22"]
    reference_sensor: sensor to serve as the global (x=0,y=0) anchor
    """
    import matplotlib.pyplot as plt

    # Reference origin lat/lon
    lat_ref_global, lon_ref_global = get_sensor_position(sensor_dict[reference_sensor])

    # Plot sensor positions in global frame
    sensor_coords = {}
    for sid in sensor_ids:
        lat, lon = get_sensor_position(sensor_dict[sid])
        x, y = latlon_to_meters(lat, lon, lat_ref_global, lon_ref_global)
        sensor_coords[sid] = (x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    for sid, (x, y) in sensor_coords.items():
        ax.plot(x, y, 'ko')
        ax.text(x, y, sid, fontsize=9)

    plotted_legend = False

    # Plot estimated origins
    for event_id, x_local, y_local, ref_sensor in estimates:
        # Convert local to global using ref_sensor’s lat/lon
        lat_ref_event, lon_ref_event = map(float, get_sensor_position(sensor_dict[ref_sensor]))
        lat_global, lon_global = map(float, get_sensor_position(sensor_dict[reference_sensor]))

        # Convert local coords (x_local, y_local) in ref_sensor frame to lat/lon
        delta_lat = y_local / 111320
        delta_lon = x_local / (111320 * np.cos(np.radians(lat_ref_event)))
        lat_est = lat_ref_event + delta_lat
        lon_est = lon_ref_event + delta_lon

        # Project into global frame
        x_global, y_global = latlon_to_meters(lat_est, lon_est, lat_global, lon_global)

        if not plotted_legend:
            ax.plot(x_global, y_global, 'yo', markersize=8, label='Estimated Origin')
            plotted_legend = True
        else:
            ax.plot(x_global, y_global, 'yo', markersize=8)

        ax.text(x_global + 5, y_global + 5, f"{event_id}\n({x_local:.0f}, {y_local:.0f})", color='goldenrod', fontsize=8)


    ax.set_title("Estimated Origins from All Events (Global Frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()

estimated_origins_summary = []

# ---- Loop through events ----
for event_id, event_data in grouped_events.items():
    print(f"\n--- {event_id} ---")

    ordered_max_vals = get_maxima(event_data)
    if len(ordered_max_vals) < 4:
        print(f"Skipping {event_id}: not enough sensors")
        continue

    peak_times = {sensor: t_max for sensor, (_, t_max) in ordered_max_vals}
    ref_sensor = list(peak_times.keys())[0]
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][ref_sensor])

    # Convert coordinates to local frame
    x_vals, y_vals, z_vals = [], [], []
    for sid in peak_times:
        lat, lon = get_sensor_position(facility_sensors["AltEn"]["Feb2024"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(peak_times[sid])
    x_vals, y_vals, z_vals = np.array(x_vals), np.array(y_vals), np.array(z_vals)

    # Interpolation grid
    xi = np.linspace(x_vals.min(), x_vals.max(), 100)
    yi = np.linspace(y_vals.min(), y_vals.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    try:
        zi = griddata((x_vals, y_vals), z_vals, (xi_grid, yi_grid), method='cubic')
        arrival_func = RegularGridInterpolator((xi, yi), zi.T, bounds_error=False, fill_value=np.nan)
    except:
        print(f"Skipping {event_id}: surface interpolation failed")
        continue

    init_x = np.mean(x_vals)
    init_y = np.mean(y_vals)
    local_min, path = gradient_descent(arrival_func, init_x, init_y)
    (x_min, y_min) = local_min

    # Handle case where gradient descent didn’t move or returned NaN
    if np.isnan(x_min) or np.isnan(y_min) or np.allclose(path[0], path[-1], atol=1e-2):
        print(f"{event_id}: gradient descent did not move — start point is likely already the minimum")

        x_min, y_min = path[0]
        path = [path[0]]  # single-point path
        plot_gradient_descent_surface(event_id, x_vals, y_vals, z_vals, path, ref_sensor)
        continue

    plot_gradient_descent_surface(event_id, x_vals, y_vals, z_vals, path, ref_sensor)

    print(f"Estimated origin for {event_id}: x = {x_min:.2f}, y = {y_min:.2f}")
    print("Reference sensor:", ref_sensor)

    estimated_origins_summary.append((event_id, x_min, y_min, ref_sensor))


sensor_ids = list(facility_sensors["AltEn"]["Feb2024"].keys())
plot_summary_estimated_origins(estimated_origins_summary, sensor_ids, facility_sensors["AltEn"]["Feb2024"], global_ref_sensor)

