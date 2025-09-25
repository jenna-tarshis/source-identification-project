# get_weather_series.py
import requests
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from config import facility_id

# --- CONFIG ---
EMAIL = "jenna.t@scentroid.com"
PASSWORD = "njw9qyr5FRZ-uhj4xwf"
LOGIN_URL = f"https://dashboard-api.sims3.scentroid.com/login?email={EMAIL}&password={PASSWORD}"
WEATHER_URL = "https://dashboard-api.sims3.scentroid.com/weather/card"
HEADERS = {"Content-Type": "application/json"}

# --- LOGIN ---
login_response = requests.post(LOGIN_URL)
cookies = login_response.cookies
print("Logged in successfully.")

# --- COLLECT UNIQUE TIMESTAMPS FROM ALL PROCESSED FILES ---
data_folder = "data/processed_samples"
timestamps = set()

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_folder, file), parse_dates=["datetime"])
        timestamps.update(df["datetime"].dt.floor("min").unique())  # floor to minute to prevent duplicates

timestamps = sorted(timestamps)
print(f"Found {len(timestamps)} unique timestamps.")

# --- FETCH WEATHER FOR EACH TIMESTAMP ---
records = []

for dt in tqdm(timestamps):
    try:
        epoch = int(pd.to_datetime(dt).timestamp())

        payload = {
            "facility_id": facility_id,
            "time_epoch": epoch,
            "is_forecast": False
        }

        response = requests.post(WEATHER_URL, json=payload, headers=HEADERS, cookies=cookies)
        response.raise_for_status()

        weather_raw = response.json()
        weather_filtered = {
            "datetime": dt,
            "wind_direction": weather_raw.get("wind_direction"),
            "wind_speed_ms": weather_raw.get("wind_speed_ms"),
            "wind_direction_deg": weather_raw.get("wind_direction_deg"),
            #"temperature_c": weather_raw.get("temperature_c"),
            #"precipitation_mm": weather_raw.get("precipitation_mm")
        }
        records.append(weather_filtered)

    except Exception as e:
        print(f"Failed at {dt}: {e}")

df_weather = pd.DataFrame(records)
df_weather.to_csv("weather_time_series.csv", index=False)
print("Saved weather_time_series.csv")
