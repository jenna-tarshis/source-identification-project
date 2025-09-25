import urllib.request
import json

try:
    with urllib.request.urlopen('https://data.api.xweather.com/airquality/dubai,?format=json&client_id=WYtSoVnTz3UcuflJxCYXE&client_secret=39u2QtZOLBCBqK8uwGarI1ffJ79U3eilKoTyTTdf') as response:
        data = response.read()
        json_data = json.loads(data)

        if 'success' not in json_data or json_data['success']:
            print(json.dumps(json_data, indent=2))
        else:
            print("Error: %s" % json_data['error']['description'])
except Exception as e:
    print(f"Request failed: {e}")