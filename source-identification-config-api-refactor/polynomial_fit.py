# polynomial_fit.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    pollutant,
    facility_id,
    start_dt as START_DT,
    end_dt as END_DT,
)

# ----------------------------- Settings -----------------------------
DEGREE = 2                 # polynomial degree for fit
N_DENSE = 1000             # points for smooth polynomial curve
PLOT_ROOT = "plots"        # same root used elsewhere

# ----------------------------- Helpers -----------------------------
def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def _fmt_dt_for_dir(x) -> str:
    return pd.to_datetime(x).strftime("%Y-%m-%d_%H%M")

def _analysis_dir(pollutant: str, facility_id: str, start_dt: str, end_dt: str) -> str:
    """
    plots/<pollutant>/<facility_id>/<start>__<end>
    """
    d = os.path.join(
        PLOT_ROOT,
        _safe_name(pollutant),
        _safe_name(facility_id),
        f"{_fmt_dt_for_dir(START_DT)}__{_fmt_dt_for_dir(END_DT)}",
    )
    os.makedirs(d, exist_ok=True)
    return d

def _fit_poly_return_peak(df: pd.DataFrame, degree: int):
    """
    Fit polynomial of given degree to df[['datetime','value']].
    Returns (peak_val, peak_time, time_dense, y_dense).
    Robust to short segments.
    """
    dfi = df.copy()
    dfi["datetime"] = pd.to_datetime(dfi["datetime"], errors="coerce")
    dfi = dfi.dropna(subset=["datetime"]).sort_values("datetime")
    if len(dfi) == 0:
        return None, None, None, None

    # seconds from segment start
    x = (dfi["datetime"] - dfi["datetime"].iloc[0]).dt.total_seconds().to_numpy()
    y = dfi["value"].to_numpy()

    if len(x) < degree + 1 or np.allclose(y, y[0]):
        # fall back to max of observed segment
        idx = int(np.nanargmax(y))
        return y[idx], dfi["datetime"].iloc[idx], dfi["datetime"], y

    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    x_dense = np.linspace(x.min(), x.max(), N_DENSE)
    y_dense = poly(x_dense)
    t0 = dfi["datetime"].iloc[0]
    time_dense = pd.to_datetime(t0) + pd.to_timedelta(x_dense, unit="s")

    peak_idx = int(np.nanargmax(y_dense))
    return y_dense[peak_idx], time_dense[peak_idx], time_dense, y_dense

# ----------------------------- Main -----------------------------
def main():
    # Load grouped_events.pkl produced by compare_sensors.py
    with open("grouped_events.pkl", "rb") as f:
        grouped_events = pickle.load(f)

    out_dir = _analysis_dir(pollutant, facility_id, START_DT, END_DT)
    polyfits_dir = os.path.join(out_dir, "polynomial_fits")
    os.makedirs(polyfits_dir, exist_ok=True)

    # Collect all peaks in a dict so we can also export an arrival_times-like object
    # Structure: {event_id: [(sensor_id, (peak_val, peak_time)), ... sorted by time]}
    all_event_peaks = {}

    # Iterate over events
    for event_id, sensor_segments in grouped_events.items():
        # Skip empty events
        valid_items = [(sid, seg) for sid, seg in (sensor_segments or {}).items()
                       if isinstance(seg, pd.DataFrame) and len(seg) > 0]
        if not valid_items:
            continue

        # Figure out event time window for the title
        seg_spans = []
        for _, seg in valid_items:
            seg = seg.copy()
            seg["datetime"] = pd.to_datetime(seg["datetime"], errors="coerce")
            seg = seg.dropna(subset=["datetime"]).sort_values("datetime")
            if len(seg):
                seg_spans.append((seg["datetime"].iloc[0], seg["datetime"].iloc[-1]))

        if seg_spans:
            event_start = min(s for s, _ in seg_spans)
            event_end   = max(e for _, e in seg_spans)
        else:
            event_start = event_end = None

        # Plot
        plt.figure(figsize=(12, 6))
        colors = plt.cm.get_cmap("tab10", len(valid_items))

        event_peaks = []
        for idx, (sensor_id, seg_df) in enumerate(valid_items):
            color = colors(idx)

            # Raw series
            seg_df = seg_df.copy()
            seg_df["datetime"] = pd.to_datetime(seg_df["datetime"], errors="coerce")
            seg_df = seg_df.dropna(subset=["datetime"]).sort_values("datetime")
            if len(seg_df) == 0:
                continue

            plt.plot(seg_df["datetime"], seg_df["value"], linestyle="--", alpha=0.35, color=color, label=f"{sensor_id} raw")

            # Fit + peak
            peak_val, peak_time, time_dense, y_dense = _fit_poly_return_peak(seg_df, DEGREE)
            if peak_val is not None and peak_time is not None:
                event_peaks.append((str(sensor_id), (float(peak_val), pd.to_datetime(peak_time))))
                if time_dense is not None and y_dense is not None:
                    plt.plot(time_dense, y_dense, color=color, linewidth=1.8, label=f"{sensor_id} fit")
                # Mark peak
                plt.scatter(peak_time, peak_val, color=color, marker="x", zorder=5)

        # Sort peaks by time and store
        event_peaks_sorted = sorted(event_peaks, key=lambda kv: kv[1][1])
        all_event_peaks[event_id] = event_peaks_sorted

        # Labels / title
        if event_start is not None and event_end is not None:
            title_dt = f"{event_start.strftime('%Y-%m-%d %H:%M')} → {event_end.strftime('%Y-%m-%d %H:%M')}"
        else:
            title_dt = "(unknown span)"

        plt.title(f"Polynomial Fits (deg={DEGREE}) for {pollutant} — Event {event_id}\n{title_dt}")
        plt.xlabel("Datetime")
        plt.ylabel(f"{pollutant} concentration")
        plt.grid(True)
        # Clean legend: one "raw" and one "fit" per sensor could be noisy—keep only the "fit" labels
        handles, labels = plt.gca().get_legend_handles_labels()
        keep = [(h, l) for (h, l) in zip(handles, labels) if l.endswith("fit")]
        if keep:
            plt.legend(*zip(*keep), fontsize=9, ncol=2)

        plt.tight_layout()

        # Save per-event figure + CSV of peaks
        fig_path = os.path.join(polyfits_dir, f"{event_id}_polynomial_fits.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[SAVED] {fig_path}")

        peaks_csv_path = os.path.join(polyfits_dir, f"{event_id}_polynomial_peaks.csv")
        pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "sensor_id": sid,
                    "peak_value": val_time[0],
                    "peak_time": pd.to_datetime(val_time[1]).isoformat(),
                }
                for sid, val_time in event_peaks_sorted
            ]
        ).to_csv(peaks_csv_path, index=False)
        print(f"[SAVED] {peaks_csv_path}")

    # Optional: export an arrival_times-like object for downstream use
    with open("arrival_times_object.pkl", "wb") as f:
        pickle.dump(all_event_peaks, f)
    print("[SAVED] arrival_times_object.pkl (dict: event_id -> sorted [(sensor_id, (peak_value, peak_time)), ...])")

if __name__ == "__main__":
    main()
