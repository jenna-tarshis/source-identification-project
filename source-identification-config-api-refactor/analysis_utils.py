# anaylsis_utils.py
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_api_csv(file_path):
    df = pd.read_csv(file_path)

    # Calculate seconds since first timestamp
    df["time_seconds"] = (df["Datetime"] - df["Datetime"].min()).dt.total_seconds()

    # Rename 'value' column to match pollutant
    pollutant_name = file_path.split("_")[-1].replace(".csv", "")
    df = df.rename(columns={"value": pollutant_name})

    return df

# NORMALIZE DATA
def normalize_df(df):
    scaler = StandardScaler()
    meta_cols = ['Datetime', 'time_seconds']
    features = df.drop(columns=[col for col in meta_cols if col in df.columns])
    features = features.apply(pd.to_numeric, errors='coerce')
    normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return pd.concat([df[meta_cols].reset_index(drop=True), normalized], axis=1)

# APPLY ROLLING AVERAGE
def apply_rolling_avg(df, window):
    meta_cols = ['datetime']  # columns to preserve
    features = df.drop(columns=[col for col in meta_cols if col in df.columns])
    features = features.apply(pd.to_numeric, errors='coerce')
    rolled = features.rolling(window=window, min_periods=1).mean()
    return pd.concat([df[meta_cols].reset_index(drop=True), rolled.reset_index(drop=True)], axis=1)


# -------------------------------------------------------------- EVA.py --------------------------------------------------------------

# COMPUTE BLOCK MAXIMA AND ANOMALIES
def compute_block_maxima(df, col, block_width_seconds):
    """
    Splits the time series into fixed-width datetime blocks and finds max value in each.

    Parameters:
    - df: pandas DataFrame with a 'datetime' column
    - col: column to find maxima in (usually 'value')
    - block_width_seconds: block width in seconds

    Returns:
    - maxima_values: list of max values in each block
    - maxima_blocks: list of DataFrames (each block's data)
    """
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    start_time = df["datetime"].min()
    end_time = df["datetime"].max()

    maxima_values = []
    maxima_blocks = []

    current_start = start_time

    while current_start < end_time:
        current_end = current_start + pd.Timedelta(seconds=block_width_seconds)
        block_df = df[(df["datetime"] >= current_start) & (df["datetime"] < current_end)]

        if not block_df.empty:
            max_val = block_df[col].max()
            maxima_values.append(max_val)
            maxima_blocks.append(block_df)

        current_start = current_end

    return maxima_values, maxima_blocks



def assign_return_period_labels(maxima, block_width_seconds):
    """
    Assign return period (in days) to each block maximum.

    Parameters:
    - maxima: list of block maximum values
    - block_width_seconds: size of each block in seconds

    Returns:
    - return_periods_in_days: list of return periods for each maximum (same order as sorted maxima)
    - maxima_sorted: list of maxima sorted in descending order
    """
    maxima_sorted = sorted(maxima, reverse=True)
    n = len(maxima_sorted)

    # Return period in units of blocks
    return_periods_in_blocks = [(n + 1) / (i + 1) for i in range(n)]

    # Convert block size to days
    block_width_days = block_width_seconds / (24 * 3600)
    return_periods_in_days = [rp * block_width_days for rp in return_periods_in_blocks]

    return return_periods_in_days, maxima_sorted


def get_anomaly_threshold(maxima, return_periods, threshold_rp):
    """
    Determine anomaly threshold based on a target return period.

    Parameters:
    - maxima: sorted list of block maxima
    - return_periods: corresponding list of return periods (same order as maxima)
    - threshold_rp: the minimum return period (in days) to consider as an anomaly

    Returns:
    - The lowest value among the maxima with a return period >= threshold_rp,
      or the highest overall maximum if none qualify
    """
    high_returns = [val for i, val in enumerate(maxima) if return_periods[i] >= threshold_rp]
    return min(high_returns) if high_returns else maxima[0]


def label_anomalies(df, col, threshold):
    """
    Add a boolean column indicating whether each row exceeds the anomaly threshold.

    Parameters:
    - df: DataFrame with pollutant data
    - col: name of the column to evaluate (e.g., 'value')
    - threshold: numeric threshold above which values are considered anomalies

    Returns:
    - DataFrame with a new column named '{col}_anomaly'
    """
    df['Anomaly'] = df[col] > threshold
    return df


# WRAPPER FOR ANOMALY DETECTION (via block-maxima approach)
def detect_anomalies_pipeline(df, pollutant_cols, return_days=5, block_width_seconds=6*3600):
    for col in pollutant_cols:
        maxima, _ = compute_block_maxima(df, col, block_width_seconds)
        rp, maxima_sorted = assign_return_period_labels(maxima)
        threshold = get_anomaly_threshold(maxima_sorted, rp, return_days)
        df = label_anomalies(df, col, threshold)
    return df

# -------------------------------------------------------------- CompareSensorsEVA.py --------------------------------------------------------------

# FIND CONINUOUS T/F SEGMENTS WITHIN ANOMALY COL
def extract_anomalous_segments(df, anom_col, min_duration=pd.Timedelta("30min")):
    df = df.copy()
    df['segment_num'] = (df[anom_col] != df[anom_col].shift()).cumsum()                 
    segments = {}
    seg_ID = 1
    for _, gdf in df[df[anom_col]].groupby('segment_num'):      # only look at the ANOMALOUS SEGMENTS
        start = gdf['datetime'].iloc[0]                                                    
        end = gdf['datetime'].iloc[-1]
        if end - start >= min_duration:     # ignore events less than 30 mins
            segments[(seg_ID)] = {
                'Start': start,
                'End': end,
                'Data': gdf
            }
            seg_ID += 1
    return segments

def assign_event_groups(all_segments, response_window=pd.Timedelta("1h")):
    flat_segments = []
    # Flatten the nested dict into a list of segment records
    for sensor, segment_info in all_segments.items():
        for segment_num, seg_data in segment_info.items():
            if isinstance(segment_num, int):  # segment_num is 1, 2, 3...
                flat_segments.append({
                    'Sensor': sensor,
                    'Start': seg_data['Start'],
                    'End': seg_data['End'],
                    'Data': seg_data['Data'],
                    'Assigned': False
                })

    # Sort by start time
    flat_segments = sorted(flat_segments, key=lambda x: x['Start'])

    events = {}
    event_id = 1
    i = 0
    while i < len(flat_segments):
        if flat_segments[i]['Assigned']:
            i += 1
            continue
        trigger = flat_segments[i]
        trigger_start = trigger['Start']
        trigger['Assigned'] = True
        event_segments = [trigger]

        # Group all other segments starting within the response window of the trigger
        for j in range(i + 1, len(flat_segments)):
            candidate = flat_segments[j]
            if candidate['Assigned']:
                continue
            if candidate['Start'] - trigger_start <= response_window:
                candidate['Assigned'] = True
                event_segments.append(candidate)
            else:
                break  # everything is sorted, so no more matches

        # Sort and store event
        event_segments = sorted(event_segments, key=lambda x: x['Start'])
        events[f"E{event_id}"] = {seg['Sensor']: seg['Data'] for seg in event_segments}
        event_id += 1
        i += 1  # move to the next potential trigger

    return events

# -------------------------------------------------------------- PlumeTrackingIsoconc.py --------------------------------------------------------------

"""# GET LAT/LON OF A SENSOR USING FILE_PATH --> MUST BE CALLED BEFORE LOAD_AND_PREPROCESS_CSV()
def get_sensor_position(file_path):
    df_raw = pd.read_csv(file_path, header=4)
    if df_raw.iloc[0].astype(str).str.contains("ug/m|\u00c2").any():
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    lat = df_raw['lat'].iloc[1]
    lon = df_raw['lon'].iloc[1]
    return lat, lon"""

# CONVERTS LATITUDE/LONGITUDE TO METERS (IN REFERENCE TO A CHOOSEN SENSOR)
def latlon_to_meters(lat, lon, lat_ref, lon_ref):
    """Convert lat/lon to meters relative to a reference point (a sensor)"""
    # Convert degrees to radians
    lat = float(lat)
    lon = float(lon)
    lat_ref = float(lat_ref)
    lon_ref = float(lon_ref)

    delta_lat = lat - lat_ref
    delta_lon = lon - lon_ref

    mean_lat_rad = math.radians((lat + lat_ref) / 2.0)
    meters_per_deg_lat = 111_320  # approx
    meters_per_deg_lon = 111_320 * math.cos(mean_lat_rad)

    x = delta_lon * meters_per_deg_lon  # east-west
    y = delta_lat * meters_per_deg_lat  # north-south
    return x, y

def get_event_start_threshold(event_data):
    """
    Returns the PM-2.5 concentration at the earliest moment across sensors in the event.
    """
    earliest_time = None
    threshold = None
    for sensor, df in event_data.items():
        first_time = df['Datetime'].iloc[0]
        if earliest_time is None or first_time < earliest_time:
            earliest_time = first_time
            threshold = df['PM-2.5'].iloc[0]
    return threshold

def get_times_to_threshold(sensor_ids, event_data, threshold):
    """
    For each sensor, find the first time the concentration reaches/exceeds the threshold.
    Returns: list of times (seconds since event start), or None if not reached
    """
    times = []
    for sid in sensor_ids:
        df = event_data[sid]
        df_above = df[df['PM-2.5'] >= threshold]
        if df_above.empty:
            return None  # Sensor didn’t reach the threshold → skip this triangle
        times.append(df_above.iloc[0]['time_seconds'])
    return times

def fit_plane_and_get_gradient(coords, times):
    """
    coords: 3x2 numpy array of (x, y) sensor positions
    times:  length-3 array of times to reach concentration
    Returns: 2D gradient vector (normalized or not), triangle area
    """
    A = np.c_[coords[:, 0], coords[:, 1], np.ones(3)]
    coeffs, _, _, _ = np.linalg.lstsq(A, times, rcond=None)
    a, b, c = coeffs
    gradient = np.array([a, b])  # units: seconds per meter
    projected_vector = -gradient  # negative to indicate direction of plume movement

    # projected_vector = projected_vector / np.linalg.norm(projected_vector)        # UNCOMMENT TO NORMLAIZE VECTOR

    # Area (shoelace formula)
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    return projected_vector, area

