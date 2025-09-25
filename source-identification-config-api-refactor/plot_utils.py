# plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import get_cmap
import matplotlib.ticker as mticker  
import numpy as np
import pandas as pd

# ----------------------------------------------- EVA.py -----------------------------------------------
# Datetime is now used on the x-axis, but time in seconds is used for all calcs ...

def _add_time_gridlines(time_series):
    ax = plt.gca()
    if np.issubdtype(time_series.dtype, np.datetime64):
        # Clean datetime range
        min_time = pd.to_datetime(time_series.min())
        max_time = pd.to_datetime(time_series.max())
        # No automatic ticks — only manual lines
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_minor_locator(mticker.NullLocator())    # disable minor ticks
        # 6-hour intervals
        six_hour_range = pd.date_range(start=min_time.floor('6h'), end=max_time.ceil('6h'), freq='6h')
        for t in six_hour_range:
            is_midnight = t.hour == 0
            plt.axvline(x=t, color='blue' if is_midnight else 'lightgray',
                        linestyle='--' if is_midnight else ':',
                        linewidth=1.2 if is_midnight else 0.8)
        plt.gcf().autofmt_xdate()
    else:
        # Fall back to seconds-based gridlines
        seconds_per_day = 86400
        seconds_per_6_hour = 21600
        max_t = np.nanmax(time_series)

        for t in range(0, int(max_t) + 1, seconds_per_6_hour):
            is_midnight = (t % seconds_per_day == 0)
            plt.axvline(x=t, color='blue' if is_midnight else 'lightgray',
                        linestyle='--' if is_midnight else ':',
                        linewidth=1.2 if is_midnight else 0.8)


def plot_block_maxima(time, signal, maxima_times, maxima_vals, label):
    plt.figure(figsize=(12, 3), dpi=100)
    _add_time_gridlines(time)
    plt.plot(time, signal, label=label)
    plt.scatter(maxima_times, maxima_vals, color='red', label='Block Max')
    plt.xlabel("Datetime")
    plt.ylabel("Concentration")
    plt.legend()
    plt.tight_layout()

def plot_return_period(return_periods, maxima_vals, title):
    plt.figure(figsize=(10, 6))
    plt.plot(return_periods, maxima_vals, color='blue')
    plt.scatter(return_periods, maxima_vals, color='red')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlabel("Return Period")
    ax.set_ylabel("Block Max Concentration")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    # Custom ticks: include fractional days and full days
    ticks = [0.25, 0.5, 1, 2, 5, 10, 20]  # in days
    tick_labels = ["6 hr", "12 hr", "1 day", "2 days", "5 days", "10 days", "20 days"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    plt.tight_layout()
    plt.show()

def plot_anomaly_regions(time, signal, is_anomaly, threshold, return_period, ax=None, line_color='blue'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    
    ax.plot(time, signal, color=line_color, linewidth=1.5)
    ax.fill_between(time, signal, where=is_anomaly, color='red', alpha=0.3)
    ax.axhline(threshold, color='black', linestyle='--', linewidth=1)
    _add_time_gridlines(time)
    ax.set_title(f"Anomalous Regions (Threshold = {return_period} day return period = {threshold:.2f} ug/m3)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Concentration")
    return ax

# ---------------------------------- CompareSensorsEVA.py -----------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_overlay_anomalies_for_all_sensors(sensor_dfs, pollutant, return_period_days):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    cmap = get_cmap('tab10', len(sensor_dfs))

    for i, (sensor, df) in enumerate(sensor_dfs.items()):
        if df.empty or "value" not in df.columns or "Anomaly" not in df.columns:
            print(f"Skipping sensor {sensor} due to missing data.")
            continue

        time_col = "datetime" if "datetime" in df.columns else "Datetime"
        time = df[time_col]
        signal = df["value"].interpolate(limit_direction='both')
        is_anom = df["Anomaly"]

        line_color = cmap(i)
        ax.plot(time, signal, color=line_color, linewidth=1.5, label=sensor)
        ax.fill_between(time, signal, where=is_anom, color=line_color, alpha=0.3, label=None)

    ax.set_title(f"{pollutant} Anomalous Regions Overlay (Threshold = {return_period_days} day return period)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Concentration")
    ax.legend(title="Sensor", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True)
    plt.tight_layout()


def plot_overlay_all_sensors(sensor_dfs, pollutant):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    cmap = get_cmap('tab10', len(sensor_dfs))

    for i, (sensor, df) in enumerate(sensor_dfs.items()):
        if df.empty or "value" not in df.columns:
            print(f"Skipping sensor {sensor} due to missing data.")
            continue

        time_col = "datetime" if "datetime" in df.columns else "Datetime"
        time = df[time_col]
        signal = df["value"].interpolate(limit_direction='both')

        line_color = cmap(i)
        ax.plot(time, signal, color=line_color, linewidth=1.5, label=sensor)
        #ax.fill_between(time, signal, where=is_anom, color=line_color, alpha=0.3, label=None)

    ax.set_title(f"{pollutant} Sensor Overlay")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Concentration")
    ax.legend(title="Sensor", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True)
    plt.tight_layout()


    