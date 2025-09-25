# config.py
import pandas as pd

#PARAMETERS
start_dt = "2025-06-01 00:00:00"
end_dt = "2025-06-08 00:00:00"
facility_id = 318
equipment = 1028
pollutant = "PM-2.5"
return_period_days = 1            # anomaly threshold in days
block_width_seconds = 4*3600      # block size for EVA 
data_dir = "data/processed_samples"  # path to processed sensor CSVs

#start_dt_analysis = "2025-04-16 14:00:00"
#end_dt_analysis = "2025-04-16 15:00:00"

ref_conc = 0