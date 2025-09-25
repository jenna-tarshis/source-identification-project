# get_time_series.py
import os
import json
import requests
import pandas as pd
from datetime import datetime
from config import start_dt, end_dt, facility_id, pollutant

"""
Login
Use JSON file to iterate sensors of facility (filter by pollutant)
Save time series to .csv in data/processed_samples
"""

# --- CONFIG ---
EMAIL = "jenna.t@scentroid.com"
PASSWORD = "njw9qyr5FRZ-uhj4xwf"
LOGIN_URL = f"https://dashboard-api.sims3.scentroid.com/login?email={EMAIL}&password={PASSWORD}"
DATA_URL = "https://dashboard-api.sims3.scentroid.com/processed_samples/plot"
HEADERS = {"Content-Type": "application/json"}

# --- OUTPUT DIRECTORY ---
SAVE_DIR = "data/processed_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- CLEAR OLD FILES ---
for fname in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, fname)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to delete {file_path}: {e}")

# --- LOAD FACILITY DICT ---
sensor_file = f"{facility_id}_facility_dict.json"
with open(sensor_file, "r") as f:
    facility = json.load(f)
print(facility)

# --- TIME RANGE ---
start_dt = pd.to_datetime(start_dt)
end_dt = pd.to_datetime(end_dt)
start_epoch = int(start_dt.timestamp())
end_epoch = int(end_dt.timestamp())

# --- LOGIN ---
try:
    login_response = requests.post(LOGIN_URL)
    cookies = login_response.cookies
    print("Logged in successfully.")
except Exception as e:
    raise RuntimeError(f"Login failed: {e}")

# --- LOOP THROUGH EQUIPMENT + SENSORS ---
for equipment_id, meta in facility.items():
    sensors = meta.get("sensors", {})

    for pollutant_name, sensor_info in sensors.items():
        # Handle legacy or new format: if sensor_info is a list, unpack it
        if isinstance(sensor_info, list):
            sensor_id, sensitivity = sensor_info
        else:
            sensor_id = sensor_info
            sensitivity = 1.0  # default if not provided

        if pollutant and pollutant_name != pollutant:
            continue  # Skip if user specified a pollutant and this one doesn't match

        print(f"Retrieving data for Equipment {equipment_id} - {pollutant_name} (Sensor ID: {sensor_id})")

        payload = {
            "sensor_list": [str(sensor_id)],
            "start_time": start_epoch,
            "end_time": end_epoch
        }

        try:
            response = requests.post(DATA_URL, json=payload, headers=HEADERS, cookies=cookies)
            response.raise_for_status()
            data = response.json()
            series_list = data.get("series", [])

            if not series_list:
                print(f"No data returned for {equipment_id} - {pollutant_name}")
                continue

            for series in series_list:
                values = series.get("data", [])
                if not values:
                    print(f"No values for {equipment_id} - {pollutant_name}")
                    continue

                df = pd.DataFrame(values, columns=["timestamp_ms", "value"])
                df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
                df = df[["datetime", "value"]]

                #  sensitivity correction ---
                if sensitivity != 1.0:
                    if sensitivity > 1.0:
                        df["value"] = df["value"] / sensitivity
                        print(f"  → Divided by sensitivity {sensitivity}")
                    else:
                        df["value"] = df["value"] * (sensitivity)
                        print(f"  → Multiplied by {sensitivity} to scale for sensitivity {sensitivity}")

                filename = f"{facility_id}_{equipment_id}_{pollutant_name}.csv"
                save_path = os.path.join(SAVE_DIR, filename)
                df.to_csv(save_path, index=False)
                print(f"Saved to {save_path}")

        except Exception as e:
            print(f"Failed to retrieve or save data for {equipment_id} - {pollutant_name}: {e}")


