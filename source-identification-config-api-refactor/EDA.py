# %%
# Exploratory Data Analysis Script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import math
from ProcessTimeSeries import file_paths, load_and_preprocess_csv, normalize_dataframe
plt.ion()  # Enable interactive mode for VS Code Interactive Window

# %%
raw_df = load_and_preprocess_csv(file_paths["AltEn-1Week-_May25_June01"])     # load and preprocess data
normalized_df = normalize_dataframe(raw_df)                                 # create normalized version of df
pollutant = ['PM-2.5']                                                     # pick pollutant to analyse
window = 60                                                                 # define window for rollign average (in minutes here??)

#pollutant = normalized_df.columns.difference(['time_seconds']).tolist()
#pollutant = ['H2S-10', 'NH3-100', 'TVOC-P-3', 'PM-2.5', 'RH-INT', 'T-INT', 'P-BARO']

# %%
# Plot real concentration values of pollutant
plt.figure(figsize=(20, 4), dpi=400)
for col in pollutant:
    plt.plot(raw_df['time_seconds'], raw_df[col], label=col, color='darkred')
seconds_per_day = 86400
seconds_per_6_hour = 21600
max_time = raw_df['time_seconds'].max()
for hour_start in range(0, int(max_time), seconds_per_6_hour):
    if hour_start % seconds_per_day == 0:
        plt.axvline(x=hour_start, color='blue', linestyle='--', linewidth=1.2)
    else:
        plt.axvline(x=hour_start, color='lightgray', linestyle='--', linewidth=0.9)
plt.xlabel('Time [1e6 seconds]')
plt.ylabel('Real Concentration Value [ug/m^3]')
plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
plt.axvline(0, color='black', linestyle='-', linewidth=1.2)
plt.title(f'Real Concentration of {pollutant} Over Time')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot NORMALIZED time series of pollutant
plt.figure(figsize=(20, 4), dpi=400)
for col in pollutant:
    plt.plot(normalized_df['time_seconds'], normalized_df[col], label=col)
seconds_per_day = 86400
seconds_per_6_hour = 21600
max_time = normalized_df['time_seconds'].max()
for hour_start in range(0, int(max_time), seconds_per_6_hour):
    if hour_start % seconds_per_day == 0:
        plt.axvline(x=hour_start, color='blue', linestyle='--', linewidth=1.2)
    else:
        plt.axvline(x=hour_start, color='lightgray', linestyle='--', linewidth=0.9)
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Value (Z-score)')
plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
plt.axvline(0, color='black', linestyle='-', linewidth=1.2)
plt.title('Normalized Sensor Readings Over Time')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Rolling average of real pollutant concentration 
for col in pollutant:
    plt.figure(figsize=(20, 4), dpi=400)
    plt.plot(raw_df['time_seconds'], raw_df[col].rolling(window).mean(), label=f"{col} (Rolling Avg)", color='darkred')
    for t in range(0, int(max_time), seconds_per_6_hour):
        if t % seconds_per_day == 0:
            plt.axvline(x=t, color='blue', linestyle='--', linewidth=1.2)
        else:
            plt.axvline(x=t, color='purple', linestyle='dotted', linewidth=0.6)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rolling Avg Concentration')
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2)
    plt.title(f'Rolling Avg of {col} (window = {window}mins)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
# Rolling average of normalized pollutant concentration 
for col in pollutant:
    plt.figure(figsize=(20, 4), dpi=400)
    plt.plot(normalized_df['time_seconds'], normalized_df[col].rolling(window).mean(), label=f"{col} (Rolling Avg)", color='blue')
    for t in range(0, int(max_time), seconds_per_6_hour):
        if t % seconds_per_day == 0:
            plt.axvline(x=t, color='blue', linestyle='--', linewidth=1.2)
        else:
            plt.axvline(x=t, color='purple', linestyle='dotted', linewidth=0.6)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rolling Avg Concentration')
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2)
    plt.title(f'Rolling Avg of {col} (window = {window}s)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
# Plot histogram of pollutant concentration
plt.figure(figsize=(9,5))
plt.title(f'{pollutant} Distribution')
sns.histplot(normalized_df[col], kde=True)

# %%
