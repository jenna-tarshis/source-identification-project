# get_sensor_ids.py
import requests
import json
import os
from config import facility_id

"""
Login
Get all equipment in the facility (makes a list of full responses)
Using equipment id, get all sensors in each equipment
Using sensor ids, get short names of each pollutant/measurement
Store lat,lon,sensors (shortname and id) in facility dict
Save facility dict to JSON file 
"""

# --- CONFIG ---
EMAIL = "jenna.t@scentroid.com"
PASSWORD = "njw9qyr5FRZ-uhj4xwf"
LOGIN_URL = f"https://dashboard-api.sims3.scentroid.com/login?email={EMAIL}&password={PASSWORD}"
EQUIPMENT_URL = f"https://dashboard-api.sims3.scentroid.com/equipment/facility/{facility_id}"
SENSOR_URL = f"https://dashboard-api.sims3.scentroid.com/sensor/equipment/"
HEADERS = {"Content-Type": "application/json"}

# --- LOGIN TO SIMS3 ---
try:
    login_response = requests.post(LOGIN_URL)
    cookies = login_response.cookies
    print("Logged in successfully.")
except Exception as e:
    raise RuntimeError(f"Login failed: {e}")

# --- GET EQUIPMENT LIST ---
try:
    response = requests.get(f"https://dashboard-api.sims3.scentroid.com/equipment/facility/{facility_id}", headers=HEADERS, cookies=cookies)
    response.raise_for_status()
    equipment_list = response.json()
    #print(equipment_list)
    print(f"Retrieved {len(equipment_list)} equipment units.")
except Exception as e:
    raise RuntimeError(f"Failed to retrieve equipment list: {e}")

# --- GET SENSORS IN EACH EQUIPMENT ---
facility = {} 

for equipment in equipment_list:
    equipment_id = equipment['id']  # equipment ID

    try:
        response = requests.get(
            f"https://dashboard-api.sims3.scentroid.com/sensor/equipment/{equipment_id}?include_deleted=false&sort_by=packet_id",
            headers=HEADERS, cookies=cookies)
        response.raise_for_status()
        sensor_list = response.json()
        #print(sensor_list)
        print(f"Retrieved {len(sensor_list)} sensors for equipment {equipment_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve sensor list: {e}")
    
    sensor_names = {}

    for sensor in sensor_list:
        sensor_id = sensor['id']
        try:
            response = requests.get(
                f"https://dashboard-api.sims3.scentroid.com/sensor/{sensor_id}",
                headers=HEADERS, cookies=cookies)
            response.raise_for_status()
            sensor_info = response.json()
            short_name = sensor_info['sensor_type']['short_name']
            sensitivity = sensor_info['sensitivity']
            sensor_names[short_name] = [sensor_id, sensitivity]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve sensor info for sensor {sensor_id}: {e}")

    #add equipment to facility dictionary using its ID as key
    facility[equipment_id] = {
        'lat': equipment['lat'],
        'lon': equipment['lon'],
        'sensors': sensor_names
    }

print(json.dumps(facility, indent=4))  # pretty print

# --- SAVE TO JSON ---
json_filename = f"{facility_id}_facility_dict.json"
with open(json_filename, "w") as f:
    json.dump(facility, f, indent=4)

