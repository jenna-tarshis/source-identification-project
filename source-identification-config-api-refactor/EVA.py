# EVA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import (
    load_and_preprocess_api_csv,
    normalize_df,
    apply_rolling_avg,
    compute_block_maxima,
    assign_return_period_labels,
    get_anomaly_threshold,
    label_anomalies
)
from plot_utils import (
    plot_block_maxima,
    plot_return_period,
    plot_anomaly_regions
)

from config import (
    facility_id,
    pollutant,
    equipment,
    return_period_days,
    block_width_seconds
)

# -------------------- CONFIG --------------------
file_path = f"data/processed_samples/{facility_id}_{equipment}_{pollutant}.csv"
pollutant = "value"
#smoothing_window = 300  # seconds, adjust as needed
# ------------------------------------------------

df = pd.read_csv(file_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

#df = apply_rolling_avg(df, window=10)

# Compute EVA
maxima, maxima_blocks = compute_block_maxima(df, pollutant, block_width_seconds)
return_periods, maxima_sorted = assign_return_period_labels(maxima, block_width_seconds)
threshold = get_anomaly_threshold(maxima_sorted, return_periods, return_period_days)
df = label_anomalies(df, pollutant, threshold)

# Plotting
plot_block_maxima(df["datetime"], df[pollutant], [block["datetime"].iloc[df[pollutant].loc[block.index].argmax()] for block in maxima_blocks], maxima, label="PM-2.5")
plot_return_period(return_periods, maxima_sorted, title=f"Return Period Curve ({return_period_days}d threshold = {threshold:.2f})")
plot_anomaly_regions(df["datetime"], df[pollutant], df["Anomaly"], threshold, return_period_days)

plt.show()
