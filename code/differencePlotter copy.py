
"""
differencePlotter.py
------------------------
Generates time series plots of the difference between each sensor and its reference.
Now includes percentile-based y-axis limits so large outliers don’t distort y-axis limits.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
from scipy.signal import find_peaks

# ---------- CONFIGURATION ---------- #
SOURCE_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv")
EXPORT_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/differencePlots")
SUFFIX_MAP = {"humidity": "r", "pressure": "p", "temperature": "t"}
USE_HARDCODED_SUFFIX = True
REFERENCE_SENSORS = ["ref", "reference", "reference_rhpc_h"]
RESTRICT_TO_REFERENCE_TIME = True
PLOT_INDIVIDUAL_GROUPS = True

# Add these two flags to control which plots are generated:
PLOT_FULL_RESOLUTION = True      # Set to True to generate the full-resolution plot
PLOT_ONE_MIN_AVERAGE = True     # Set to True to generate the 1-minute average plot

# Add this flag to enable/disable the limited time plot
PLOT_ONE_HOUR = True  # Set to False to disable the limited time plot
PLOT_HOURS = .5        # How many hours to plot if PLOT_ONE_HOUR is True (supports decimals, e.g., 0.5 for 30 minutes, 1 for 1 hour)

BEGIN_EDGE_CUT_PERCENT = 0.03  # percent to cut from the beginning edge (for x-axis bounds only)
END_EDGE_CUT_PERCENT = 0.1    # percent to cut from the ending edge (for x-axis bounds only)

# Legend anchor tweaks (adjust these to nudge left/right legends)
LEFT_LEGEND_X = -0.08
RIGHT_LEGEND_X = 1.08

# ---------- HELPER FUNCTIONS ---------- #

def get_allowed_suffixes(instrument: str, use_hardcoded: bool) -> set:
    inst_lower = instrument.lower()
    if use_hardcoded and inst_lower in SUFFIX_MAP:
        return {SUFFIX_MAP[inst_lower]}
    first_letter = inst_lower[0]
    return {"r", "h"} if first_letter == "r" else {first_letter}

def add_edge_labels_if_needed(ax, time_vals, fmt="%d/%m/%y %H:%M", n_ticks: int | None = None):
    start = float(mdates.date2num(time_vals.min()))
    end = float(mdates.date2num(time_vals.max()))
    if end <= start:
        return
    raw_ticks = ax.get_xticks()
    current_count = max(4, min(10, len(raw_ticks))) if len(raw_ticks) >= 2 else 6
    if n_ticks is None:
        n_ticks = current_count
    new_ticks = np.linspace(start, end, n_ticks)
    ax.xaxis.set_major_locator(FixedLocator(new_ticks))
    ax.xaxis.set_major_formatter(DateFormatter(fmt))
    ax.grid(True, which="major", axis="x")
    fig = ax.get_figure()
    try:
        fig.canvas.draw()
    except Exception:
        pass
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_va("top")
    fig.subplots_adjust(bottom=0.18)

def load_csv_once(csv_path: Path, use_hardcoded=True, reference_sensors=None):
    if reference_sensors is None:
        reference_sensors = []
    try:
        # parse 'time' on read and avoid low_memory dtype guessing
        df = pd.read_csv(csv_path, parse_dates=["time"], low_memory=False)
    except Exception as e:
        print(f"⚠️ Could not read {csv_path}: {e}")
        return None, [], []
    if "time" not in df.columns:
        print(f"⚠️ Missing 'time' column in {csv_path}")
        return None, [], []
    # ensure datetime (parse_dates above should handle most cases)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    instrument = csv_path.parts[-3]
    allowed_suffixes = get_allowed_suffixes(instrument, use_hardcoded)
    all_sensors = [c for c in df.columns if c.lower() not in ("time", "epoch")]
    valid_sensors = [s for s in all_sensors if s[-1].lower() in allowed_suffixes or s.startswith("reference_")]
    references = [s for s in valid_sensors if any(ref.lower() in s.lower() for ref in reference_sensors)]
    return df, valid_sensors, references

def get_folder_y_limits(csv_files, use_hardcoded=True, reference_sensors=None, clip_quantile=0.999):
    if reference_sensors is None:
        reference_sensors = []
    all_diffs = []
    for csv_path in csv_files:
        df, valid_sensors, references = load_csv_once(csv_path, use_hardcoded, reference_sensors)
        if df is None or not valid_sensors or not references:
            continue
        ref = references[0]
        non_refs = [s for s in valid_sensors if s not in references]
        for sensor in non_refs:
            diff = pd.to_numeric(df[sensor], errors="coerce") - pd.to_numeric(df[ref], errors="coerce")
            diff = diff[np.isfinite(diff)]
            if diff.size > 0:
                all_diffs.append(diff)
    if not all_diffs:
        return -1, 1  # fallback
    combined = np.concatenate(all_diffs)
    # Clip extreme values using quantiles
    q_low, q_high = np.nanpercentile(combined, [0.5, 99.5])
    y_max = max(abs(q_high), abs(q_low))
    y_min = -y_max
    return y_min, y_max

def find_reference_temp(df):
    # Try to find a column that is a reference temperature
    temp_ref_names = ["ref", "reference", "temp_ref", "reference_temp", "temp_reference"]
    for col in df.columns:
        for ref in temp_ref_names:
            if ref in col.lower() and col.lower().endswith("t"):
                return col
    # fallback: any column ending with 't'
    temp_cols = [c for c in df.columns if c.lower().endswith("t")]
    return temp_cols[0] if temp_cols else None

# ---------- AXIS PRESETS (manual override) ---------- #
# Format: {"instrument_name": (ymin, ymax, step)}
AXIS_PRESETS = {
    # Example:
    "rh_test": (-5, 10, 5.0),
    "pressure": (-6.5, 1.5, 0.5)
    # "temperature": (-1, 1, 0.1),
}

def get_nice_axis_limits_and_ticks(data, n_ticks=6, instrument=None, force_no_preset=False):
    """
    Given a 1D array, return (ymin, ymax, yticks) with nice rounded limits and regular steps.
    Axis will start at a multiple of 0.5 if possible.
    If AXIS_PRESETS is set for the instrument, use those values unless force_no_preset is True.
    """
    if not force_no_preset and instrument is not None and instrument.lower() in AXIS_PRESETS:
        ymin, ymax, step = AXIS_PRESETS[instrument.lower()]
        yticks = np.arange(ymin, ymax + step * 0.5, step)
        return ymin, ymax, yticks

    finite_data = np.array(data)[np.isfinite(data)]
    if finite_data.size == 0:
        return -1, 1, np.arange(-1, 1.1, 0.5)
    dmin, dmax = np.min(finite_data), np.max(finite_data)
    # Symmetric about zero
    max_abs = max(abs(dmin), abs(dmax))
    # Find a nice step
    if max_abs <= 1:
        step = 0.1 if max_abs > 0.5 else 0.05 if max_abs > 0.2 else 0.02
    elif max_abs <= 2:
        step = 0.2
    elif max_abs <= 5:
        step = 0.5
    elif max_abs <= 10:
        step = 1
    elif max_abs <= 100:
        step = 5
    else:
        step = 2
    # For step >= 0.5, force axis to start at a multiple of 0.5
    if step >= 0.5:
        y_max = np.ceil(max_abs * 2) / 2  # round up to nearest 0.5
        y_min = -y_max
        yticks = np.arange(y_min, y_max + step * 0.5, step)
    else:
        y_max = np.ceil(max_abs / step) * step
        y_min = -y_max
        yticks = np.arange(y_min, y_max + step * 0.5, step)
    return y_min, y_max, yticks


def align_time_and_series(time_vals, *series_list):
    """
    Trim/align time_vals and provided series so they all have the same length.
    Returns (time_trimmed, [series_trimmed...]). Works with pandas Series or numpy arrays.
    """
    try:
        tv = pd.Series(time_vals).reset_index(drop=True)
    except Exception:
        tv = pd.Series(list(time_vals))
    lengths = [len(tv)] + [len(s) for s in series_list]
    min_len = min(lengths)
    tv_trim = tv.iloc[:min_len]
    trimmed = []
    for s in series_list:
        if hasattr(s, 'iloc'):
            trimmed.append(s.iloc[:min_len])
        else:
            trimmed.append(pd.Series(s).iloc[:min_len])
    return tv_trim, trimmed

# ---------- MAIN PLOTTING FUNCTION ---------- #

def process_csv(
    csv_path: Path,
    export_root: Path,
    y_min,
    y_max,
    use_hardcoded=True,
    reference_sensors=None,
    clip_quantile: float = 0.999,
    min_data_fraction: float = 0.5,
    n_y_ticks: int = 6,
    restrict_to_reference_time: bool = True,
    plot_individual_groups: bool = True
):
    instrument = csv_path.parts[-3]
    if "pre-test" in instrument.lower() or "pre-test" in csv_path.name.lower():
        print(f"⚠️ Skipping plot for {csv_path.name}: instrument or file contains 'pre-test'")
        return

    if reference_sensors is None:
        reference_sensors = []

    df, valid_sensors, references = load_csv_once(csv_path, use_hardcoded, reference_sensors)
    if df is None or not valid_sensors or not references:
        print(f"⚠️ No valid sensors or references in {csv_path}, skipping plot")
        return

    if restrict_to_reference_time and references:
        ref_times = pd.Series(dtype="datetime64[ns]")
        for ref in references:
            ref_times = pd.concat([ref_times, df.loc[df[ref].notna(), "time"]])
        if not ref_times.empty:
            ref_start = ref_times.min()
            ref_end = ref_times.max()
            df = df[(df["time"] >= ref_start) & (df["time"] <= ref_end)]
            time_vals = df["time"]
        else:
            print(f"⚠️ Reference columns are all NaN in {csv_path}, skipping plot")
            return
    else:
        time_vals = df["time"]


    non_reference_sensors = [s for s in valid_sensors if s not in references]
    filtered_non_refs = [s for s in non_reference_sensors if df[s].count() / len(df) >= min_data_fraction]
    valid_sensors = filtered_non_refs + [s for s in references if s in valid_sensors]
    references = [s for s in references if s in valid_sensors]
    if not valid_sensors or not references:
        print(f"⚠️ No valid sensors or references with sufficient data in {csv_path}, skipping plot")
        return

    out_dir = export_root / instrument
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure canonical index and aligned time/ref series before plotting
    if "time" in df.columns:
        df = df.reset_index(drop=True)
        time_vals = df["time"]
    else:
        time_vals = pd.Series(time_vals).reset_index(drop=True)
    ref_vals = pd.to_numeric(df[ref], errors="coerce")

    # --- Main difference plot (all sensors) ---
    plt.figure(figsize=(12, 6))
    ref = references[0]
    ref_vals = pd.to_numeric(df[ref], errors="coerce")
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    sensor_colors = {}
    all_diffs = []

    for sensor in valid_sensors:
        if sensor == ref:
            continue
        y_sensor = pd.to_numeric(df[sensor], errors="coerce")
        diff = y_sensor - ref_vals
        # ...existing code...
        # Simple moving average smoothing
        diff = diff.rolling(window=20, center=True, min_periods=1).mean()
        all_diffs.append(diff)
        sensor_colors[sensor] = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

    # --- Use improved axis logic for y-limits and ticks ---
    if all_diffs:
        diffs_concat = np.concatenate([d[np.isfinite(d)] for d in all_diffs if np.any(np.isfinite(d))])
        y_min_rounded, y_max_rounded, y_ticks = get_nice_axis_limits_and_ticks(diffs_concat, n_ticks=n_y_ticks, instrument=instrument)
    else:
        y_min_rounded, y_max_rounded, y_ticks = -1, 1, np.arange(-1, 1.1, 0.5)

    # Defensive alignment: trim time_vals and diffs to same minimum length
    try:
        time_trim, trimmed_diffs = align_time_and_series(time_vals, *all_diffs)
    except Exception:
        time_trim = pd.Series(time_vals).reset_index(drop=True)
        trimmed_diffs = [d.reset_index(drop=True).iloc[:len(time_trim)] if hasattr(d, 'reset_index') else pd.Series(d).iloc[:len(time_trim)] for d in all_diffs]

    diff_iter = iter(trimmed_diffs)
    for sensor in valid_sensors:
        if sensor == ref:
            continue
        diff = next(diff_iter)
        plt.plot(time_trim, diff, label=f"{sensor} - {ref}", alpha=0.8, zorder=1, color=sensor_colors[sensor])

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel(f"Difference from {ref}")
    plt.title(f"{instrument} - SENSOR DIFFERENCES ({csv_path.stem})")
    plt.ylim(y_min_rounded, y_max_rounded)
    plt.gca().set_yticks(y_ticks)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.xlim(time_vals.min(), time_vals.max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax, time_vals)

    # --- Legend placement and size adjustment ---
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(loc="best", frameon=True, fontsize=10, handlelength=2)
    plt.draw()

    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = out_dir / f"allSensorsDiff_{csv_path.stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    plt.close()

    # --- Temperature overlay difference plot ---
    temp_ref_col = find_reference_temp(df)
    if temp_ref_col is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        for sensor in valid_sensors:
            if sensor == ref:
                continue
            y_sensor = pd.to_numeric(df[sensor], errors="coerce")
            diff = y_sensor - ref_vals
            # ...existing code...
            ax1.plot(time_vals, diff, label=f"{sensor} - {ref}",
                     alpha=0.8, zorder=1, color=sensor_colors.get(sensor, None))
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"Difference from {ref}")
        ax1.set_title(f"{instrument} - SENSOR DIFFERENCES + REF TEMP ({csv_path.stem})")
        ax1.set_ylim(y_min_rounded, y_max_rounded)
        ax1.set_yticks(y_ticks)
        ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
        ax1.set_xlim(time_vals.min(), time_vals.max())
        plt.xticks(rotation=45, ha='right')
        add_edge_labels_if_needed(ax1, time_vals)

        # Overlay reference temperature (auto axis, no preset)
        temp_vals = pd.to_numeric(df[temp_ref_col], errors="coerce").values
        tmin, tmax = np.nanmin(temp_vals), np.nanmax(temp_vals)
        tmin_rounded = np.floor(tmin)
        tmax_rounded = np.ceil(tmax)
        n_temp_ticks = min(len(y_ticks), 8)
        if tmax_rounded - tmin_rounded < n_temp_ticks - 1:
            t_ticks = np.arange(tmin_rounded, tmax_rounded + 1)
        else:
            t_ticks = np.linspace(tmin_rounded, tmax_rounded, n_temp_ticks, dtype=int)
        ax2 = ax1.twinx()
        ax2.plot(time_vals, temp_vals, label=f"REF TEMP: {temp_ref_col}", linestyle="--", color="red", alpha=0.8, zorder=2)
        ax2.set_ylabel("Reference Temperature")
        ax2.set_ylim(tmin_rounded, tmax_rounded)
        ax2.set_yticks(t_ticks)
        if n_temp_ticks > 8:
            ax2.set_yticklabels([])

        ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.yaxis.grid(False)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2

        legend = ax2.legend(
            all_lines, all_labels,
            loc="center left", bbox_to_anchor=(RIGHT_LEGEND_X, 0.5),
            frameon=True, fontsize=10, handlelength=2, borderaxespad=0.0
        )
        plt.draw()

        plt.savefig(out_dir / f"allSensorsDiffTempOverlay_{csv_path.stem}.png", dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {out_dir / f'allSensorsDiffTempOverlay_{csv_path.stem}.png'}")
        plt.close()

    # --- Reference overlay difference plot ---
    plot_reference_overlay(
        df,
        valid_sensors,
        references,
        time_vals,
        instrument,
        out_dir,
        ref,
        ref_vals,
        sensor_colors,
        y_min_rounded,
        y_max_rounded,
        y_ticks,
        n_y_ticks,
        csv_path
    )

    # Group sensors by their first three letters (sensor type)
    sensor_type_map = {}
    for sensor in valid_sensors:
        if sensor == ref:
            continue
        sensor_type = sensor[:3]
        sensor_type_map.setdefault(sensor_type, []).append(sensor)

    avg_diff_dict = {}
    for sensor_type, sensors in sensor_type_map.items():
        non_ref_sensors = [s for s in sensors if s != ref]
        if not non_ref_sensors:
            continue
        sensor_data = [pd.to_numeric(df[s], errors="coerce") for s in non_ref_sensors]
        if not sensor_data:
            continue
        sensor_data = pd.concat(sensor_data, axis=1)
        avg_sensor = sensor_data.mean(axis=1)
        avg_diff = avg_sensor - ref_vals
        avg_diff_dict[sensor_type] = avg_diff

    # Plot average differences for each sensor type
    if avg_diff_dict:
        for sensor_type, avg_diff in avg_diff_dict.items():
            plt.figure(figsize=(12, 6))
            avg_diff_plot = avg_diff
            try:
                time_trim_avg, [avg_trim] = align_time_and_series(time_vals, avg_diff_plot.rolling(window=20, center=True, min_periods=1).mean())
            except Exception:
                time_trim_avg = pd.Series(time_vals).reset_index(drop=True)
                avg_trim = avg_diff_plot.rolling(window=20, center=True, min_periods=1).mean().reset_index(drop=True).iloc[:len(time_trim_avg)]
            plt.plot(time_trim_avg, avg_trim, label=f"{sensor_type} avg - {ref}")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Time")
            plt.ylabel(f"Difference from {ref}")
            plt.title(f"{instrument} - {sensor_type.upper()} SENSOR TYPE AVERAGED DIFFERENCE ({csv_path.stem})")
            plt.ylim(y_min_rounded, y_max_rounded)
            plt.gca().set_yticks(y_ticks)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
            plt.xlim(time_vals.min(), time_vals.max())
            plt.xticks(rotation=45, ha='right')
            add_edge_labels_if_needed(ax, time_vals)
            plt.legend(loc="best", frameon=True, fontsize=10, handlelength=2)
            plt.grid(True, linestyle="--", alpha=0.5)
            save_path = out_dir / f"{sensor_type.upper()}_allSensorTypeAvgDiff_{csv_path.stem}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {save_path}")
            plt.close()

def plot_reference_overlay(
    df,
    valid_sensors,
    references,
    time_vals,
    instrument,
    out_dir,
    ref,
    ref_vals,
    sensor_colors,
    y_min_rounded,
    y_max_rounded,
    y_ticks,
    n_y_ticks,
    csv_path
):
    # Only plot differences and overlay reference
    # keep as pandas Series for safer slicing/alignment
    ref_overlay_vals = pd.to_numeric(df[ref], errors="coerce")
    non_reference_sensors = [s for s in valid_sensors if s not in references]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Defensive alignment: compute diffs and trim with time_vals so x/y match
    diffs = []
    sensors_to_plot = []
    for sensor in non_reference_sensors:
        y_sensor = pd.to_numeric(df[sensor], errors="coerce")
        diff = y_sensor - ref_vals
        diffs.append(diff)
        sensors_to_plot.append(sensor)
    try:
        time_trim, diffs_trimmed = align_time_and_series(time_vals, *diffs)
    except Exception:
        time_trim = pd.Series(time_vals).reset_index(drop=True)
        diffs_trimmed = [d.reset_index(drop=True).iloc[:len(time_trim)] if hasattr(d, 'reset_index') else pd.Series(d).iloc[:len(time_trim)] for d in diffs]
    for sensor, diff in zip(sensors_to_plot, diffs_trimmed):
        ax1.plot(time_trim, diff, label=f"{sensor} - {ref}", alpha=0.8, zorder=1, color=sensor_colors.get(sensor, None))
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"Difference from {ref}")
    ax1.set_title(f"{instrument} - SENSOR DIFFERENCES + REF OVERLAY ({csv_path.stem})")
    ax1.set_ylim(y_min_rounded, y_max_rounded)
    ax1.set_yticks(y_ticks)
    ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax1.set_xlim(time_vals.min(), time_vals.max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax1, time_vals)

    # Overlay reference value (original, not smoothed)
    ax2 = ax1.twinx()
    # align reference overlay values to the time axis
    try:
        time_trim_ref, [ref_trim] = align_time_and_series(time_vals, ref_overlay_vals)
    except Exception:
        time_trim_ref = pd.Series(time_vals).reset_index(drop=True)
        ref_trim = ref_overlay_vals.reset_index(drop=True).iloc[:len(time_trim_ref)] if hasattr(ref_overlay_vals, 'reset_index') else pd.Series(ref_overlay_vals).iloc[:len(time_trim_ref)]
    ax2.plot(time_trim_ref, ref_trim, label=f"REF: {ref}", linestyle="--", color="red", alpha=0.8, zorder=2)
    # For humidity references, force the ref-axis to 10..90 with ticks every 10
    if instrument.lower() in ("humidity", "rh_test"):
        rmin_rounded = 10
        rmax_rounded = 90
        r_ticks = np.arange(10, 91, 10)
        # remove the axis label for humidity as requested
        ax2.set_ylabel("")
    else:
        rmin, rmax = np.nanmin(ref_overlay_vals), np.nanmax(ref_overlay_vals)
        rmin_rounded = np.floor(rmin)
        rmax_rounded = np.ceil(rmax)
        n_ref_ticks = min(len(y_ticks), 8)
        if rmax_rounded - rmin_rounded < n_ref_ticks - 1:
            r_ticks = np.arange(rmin_rounded, rmax_rounded + 1)
        else:
            r_ticks = np.linspace(rmin_rounded, rmax_rounded, n_ref_ticks, dtype=int)
        ax2.set_ylabel(f"Reference Value ({ref})")
    ax2.set_ylim(rmin_rounded, rmax_rounded)
    ax2.set_yticks(r_ticks)
    # keep default behavior of hiding tick labels only for crowded non-humidity axes
    if instrument.lower() not in ("humidity", "rh_test"):
        try:
            if n_ref_ticks > 8:
                ax2.set_yticklabels([])
        except Exception:
            pass

    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.yaxis.grid(False)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    legend = ax2.legend(
        all_lines, all_labels,
        loc="center left", bbox_to_anchor=(RIGHT_LEGEND_X, 0.5),
        frameon=True, fontsize=10, handlelength=2, borderaxespad=0.0
    )
    plt.draw()

    plt.savefig(out_dir / f"allSensorsDiffRefOverlay_{csv_path.stem}.png", dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out_dir / f'allSensorsDiffRefOverlay_{csv_path.stem}.png'}")
    plt.close()
def main(
    source_root: Path = SOURCE_ROOT,
    export_root: Path = EXPORT_ROOT,
    use_hardcoded: bool = USE_HARDCODED_SUFFIX,
    reference_sensors: list | None = REFERENCE_SENSORS,
    plot_individual_groups: bool = PLOT_INDIVIDUAL_GROUPS,
):
    """
    Find CSV files under source_root, compute folder-wide y-limits, and process each CSV.
    """
    source_root = Path(source_root)
    export_root = Path(export_root)
    csv_files = sorted(source_root.rglob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV files found under {source_root}")
        return

    # Compute sensible y-limits across all CSVs
    y_min, y_max = get_folder_y_limits(csv_files, use_hardcoded=use_hardcoded, reference_sensors=reference_sensors)
    print(f"Using y-limits: {y_min} to {y_max}")

    for csv_file in csv_files:
        try:
            process_csv(
                csv_file,
                export_root,
                y_min,
                y_max,
                use_hardcoded=use_hardcoded,
                reference_sensors=reference_sensors,
                restrict_to_reference_time=RESTRICT_TO_REFERENCE_TIME,
                plot_individual_groups=plot_individual_groups
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"⚠️ Error processing {csv_file}: {e}")

if __name__ == "__main__":
    main()
    # ...existing code...
