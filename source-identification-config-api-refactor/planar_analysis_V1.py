# planar_analysis_V1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
plt.ion()

# import grouped events object from CompareSensorsEVA.py
import pickle
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

from analysis_utils import *
from config import *

#------------------------------------
# PlumeTrackingIsoconc.py MODULARIZED, first version of planar analysis, limited to 3 sensors 
#------------------------------------

def analyze_triangle(sensor_ids, event_data, ref_sensor=None):
    """
    Combines all steps: get positions, reference concentration, times to threshold, and vector.
    Returns: (projected_vector, triangle_coords, area) or None if invalid.
    """
    if ref_sensor is None:
        ref_sensor = sensor_ids[0]

    threshold = get_event_start_threshold(event_data)                       # get reference conc
    times = get_times_to_threshold(sensor_ids, event_data, threshold)       # get times
    if times is None:                                                       
        return None, None, None

    # Get positions
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"][ref_sensor])
    coords = []
    for sid in sensor_ids:
        lat, lon = get_sensor_position(facility_sensors["AltEn"][sid])
        x, y = latlon_to_meters(lat, lon, lat_ref, lon_ref)
        coords.append((x, y))
    coords = np.array(coords)

    # Fit plane and get vector
    vector, area = fit_plane_and_get_gradient(coords, times)

    return vector, coords, area

#---------------------------------------------------------------------------------------------------
# load grouped_events object
with open("grouped_events.pkl", "rb") as f:
    grouped_events = pickle.load(f)

# parameters
min_triangle_area = 50      # m², to filter out small/unreliable triangles
all_vectors = {}            # Output container: {event_id: [(sensor_ids, vector, coords, area), ...]}
scale = 50                  # scale up unit vector length for visibility

# Process each event in grouped_events object
for event_id, event_data in grouped_events.items():
    sensor_ids = list(event_data.keys())    # extract sensor IDs
    if len(sensor_ids) < 3:
        continue  # not enough sensors for a triangle = exit

    vectors = []
    triangle_counter = 0
    max_triangles = 165

    # make triangle + fit plane + compute vector
    for tri in combinations(sensor_ids, 3):
        if triangle_counter >= max_triangles:
            break

        vec, coords, area = analyze_triangle(tri, event_data, ref_sensor=sensor_ids[0])
        if vec is None or area < min_triangle_area:
            continue

        vectors.append((tri, vec, coords, area))
        triangle_counter += 1

    # Plot 2D vector field
    plt.figure(figsize=(8, 8))
    for tri, vec, coords, area in vectors:
        centroid = coords.mean(axis=0)
        scaled_vec = vec * scale

        # Plot vector
        plt.quiver(centroid[0], centroid[1], scaled_vec[0], scaled_vec[1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.5, width=0.002)

        # Plot triangle outline
        #triangle_path = np.vstack([coords, coords[0]])  # close the triangle
        #plt.plot(triangle_path[:, 0], triangle_path[:, 1], 'k--', alpha=0.3)

    # Plot sensor positions (optional)
    lat_ref, lon_ref = get_sensor_position(facility_sensors["AltEn"][sensor_ids[0]])
    for sid in sensor_ids:
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

# export all vectors
with open("all_event_vectors.pkl", "wb") as f:
    pickle.dump(all_vectors, f)

plt.ioff()
plt.show()
