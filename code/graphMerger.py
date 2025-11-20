"""
graphMerger.py
------------------------
Generates time series plots for sensor CSV files.
- Filters sensors by instrument and suffix.
- Always includes reference sensors, even with missing data.
- Annotates flat, high-humidity regions with temperature.
- Optionally overlays temperature on pressure plots with dual y-axes.
- Automatically places the legend inside the plot, away from data.
- Handles missing and non-numeric data robustly.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates

# ---------- CONFIGURATION ---------- #
SOURCE_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv")
EXPORT_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/timeseriesPlots")
PLOT_PRESSURE_WITH_TEMP = True  # Overlay temperature on pressure plots if True
SUFFIX_MAP = {"humidity": "r", "pressure": "p", "temperature": "t"}
USE_HARDCODED_SUFFIX = True
REFERENCE_SENSORS = ["ref"]
RESTRICT_TO_REFERENCE_TIME = True
PLOT_INDIVIDUAL_GROUPS = False

# ---------- AXIS PRESETS (manual override) ---------- #
USE_AXIS_PRESETS = False  # Set to True to activate axis presets
AXIS_PRESETS = {
    # Example:
    # "humidity": (-6.5, 6.5, 0.5),
    # "pressure": (-6.5, 1.5, 0.5),
    # "temperature": (-1, 1, 0.1),
}

PLOT_LIMITED_TIME = True  # Set to True to enable limited time plot
PLOT_HOURS = .0166666666667             # How many hours to plot if PLOT_LIMITED_TIME is True (supports decimals, e.g., 0.5 for 30 minutes)

# ---------- HELPER FUNCTIONS ---------- #

def get_allowed_suffixes(instrument: str, use_hardcoded: bool) -> set:
    """
    Determine allowed sensor suffixes for a given instrument.
    """
    inst_lower = instrument.lower()
    if use_hardcoded and inst_lower in SUFFIX_MAP:
        return {SUFFIX_MAP[inst_lower]}
    first_letter = inst_lower[0]
    return {"r", "h"} if first_letter == "r" else {first_letter}

def add_edge_labels_if_needed(ax, time_vals, fmt="%d/%m/%y %H:%M", n_ticks: int | None = None):
    """
    Set evenly spaced x-axis ticks, including edges, and rotate labels.
    """
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
    """
    Load a CSV, parse time, and identify valid and reference sensors.
    Only sensors containing the REFERENCE_SENSORS tag are treated as references.
    """
    if reference_sensors is None:
        reference_sensors = []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Could not read {csv_path}: {e}")
        return None, [], []
    if "time" not in df.columns:
        print(f"⚠️ Missing 'time' column in {csv_path}")
        return None, [], []
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    instrument = csv_path.parts[-3]
    allowed_suffixes = get_allowed_suffixes(instrument, use_hardcoded)
    all_sensors = [c for c in df.columns if c.lower() not in ("time", "epoch")]
    valid_sensors = [s for s in all_sensors if s[-1].lower() in allowed_suffixes or s.startswith("reference_")]
    # Only use sensors with the REFERENCE_SENSORS tag as references
    references = [s for s in valid_sensors if any(ref.lower() in s.lower() for ref in reference_sensors)]
    return df, valid_sensors, references

def get_folder_y_limits(csv_files, use_hardcoded=True, reference_sensors=None, clip_quantile=0.999):
    """
    Compute global y-axis limits for all CSVs in a folder, using quantiles to clip outliers.
    """
    if reference_sensors is None:
        reference_sensors = []
    all_values = []
    for csv_path in csv_files:
        df, valid_sensors, references = load_csv_once(csv_path, use_hardcoded, reference_sensors)
        if df is None or not valid_sensors:
            continue
        min_data_fraction = 0.5
        non_reference_sensors = [s for s in valid_sensors if s not in references]
        filtered_non_refs = [s for s in non_reference_sensors if df[s].count() / len(df) >= min_data_fraction]
        valid_sensors = filtered_non_refs + [s for s in references if s in valid_sensors]
        references = [s for s in references if s in valid_sensors]
        if not valid_sensors:
            continue
        sensors_to_use = references + [s for s in valid_sensors if s not in references]
        combined_values = df[sensors_to_use].apply(pd.to_numeric, errors="coerce").values.flatten()
        combined_values = combined_values[np.isfinite(combined_values)]
        if combined_values.size > 0:
            all_values.append(combined_values)
    if not all_values:
        return None, None
    combined_all = np.concatenate(all_values)
    y_max = np.quantile(combined_all, clip_quantile)
    y_min = np.quantile(combined_all, 1 - clip_quantile)
    return y_min, y_max

def detect_flat_high_regions(y, threshold=85, window=100, std_tol=1.0, min_length=100):
    """
    Detect regions where y stays above threshold and is 'flat' (std < std_tol) for at least min_length points.
    Returns list of (start_idx, end_idx).
    """
    y = np.asarray(y, dtype=float)
    regions = []
    i = 0
    while i < len(y) - window:
        window_vals = y[i:i+window]
        if np.all(window_vals > threshold) and np.nanstd(window_vals) < std_tol:
            # Expand region as long as it stays flat and above threshold
            start = i
            while i < len(y) - window and np.all(y[i:i+window] > threshold) and np.nanstd(y[i:i+window]) < std_tol:
                i += 1
            end = i + window // 2
            if end - start >= min_length:
                regions.append((start, end))
        else:
            i += 1
    return regions

def choose_legend_location(ax):
    """
    Place the legend inside the plot, away from data.
    Uses 'best' for robust, automatic placement.
    """
    return "best", 1, 10, 2  # loc, ncol, fontsize, handlelength

def get_nice_axis_limits_and_ticks(data, n_ticks=6, instrument=None, force_no_preset=False):
    """
    Given a 1D array, return (ymin, ymax, yticks) with nice rounded limits and regular steps.
    Axis will start at a multiple of 0.5 if possible.
    If AXIS_PRESETS is set for the instrument and USE_AXIS_PRESETS is True, use those values unless force_no_preset is True.
    """
    if USE_AXIS_PRESETS and not force_no_preset and instrument is not None and instrument.lower() in AXIS_PRESETS:
        ymin, ymax, step = AXIS_PRESETS[instrument.lower()]
        yticks = np.arange(ymin, ymax + step * 0.5, step)
        return ymin, ymax, yticks

    finite_data = np.array(data)[np.isfinite(data)]
    if finite_data.size == 0:
        return -1, 1, np.arange(-1, 1.1, 0.5)
    dmin, dmax = np.min(finite_data), np.max(finite_data)
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

def plot_limited_time_allsensors(
    df,
    valid_sensors,
    references,
    time_vals,
    instrument,
    out_dir,
    sensor_colors,
    y_min_rounded,
    y_max_rounded,
    y_ticks,
    n_y_ticks,
    csv_path,
    hours=1
):
    t0 = time_vals.min()
    t1 = t0 + pd.Timedelta(hours=hours)
    mask = (time_vals >= t0) & (time_vals <= t1)
    if mask.sum() < 2:
        print(f"⚠️ Not enough data for {hours}-hour plot in {csv_path.name}")
        return

    plt.figure(figsize=(12, 6))
    for sensor in valid_sensors:
        y_sensor = pd.to_numeric(df[sensor], errors="coerce")
        plt.plot(
            time_vals[mask], y_sensor[mask],
            label=sensor,
            alpha=0.8, zorder=1, color=sensor_colors[sensor]
        )

    plt.xlabel("Time")
    plt.ylabel(f"{instrument} (all sensors)")
    # Format the title and file name to show minutes if not a whole hour
    if hours == 1:
        title_str = "1 HOUR"
        file_str = "1hour"
    elif float(hours).is_integer():
        title_str = f"{int(hours)} HOURS"
        file_str = f"{int(hours)}hour"
    else:
        minutes = int(hours * 60)
        title_str = f"{minutes} MINUTES"
        file_str = f"{minutes}min"
    plt.title(f"{instrument} - {title_str} ALL SENSORS ({csv_path.stem})")
    plt.ylim(y_min_rounded, y_max_rounded)
    plt.gca().set_yticks(y_ticks)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.xlim(time_vals[mask].min(), time_vals[mask].max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax, time_vals[mask])
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(loc="best", frameon=True, fontsize=10, handlelength=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = out_dir / f"allSensors_{file_str}_{csv_path.stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    plt.close()

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
    if reference_sensors is None:
        reference_sensors = []

    df, valid_sensors, references = load_csv_once(csv_path, use_hardcoded, reference_sensors)
    if df is None or not valid_sensors:
        return

    # Restrict to reference time range if requested
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

    # Filter non-reference sensors with too much missing data, always keep references
    non_reference_sensors = [s for s in valid_sensors if s not in references]
    filtered_non_refs = [s for s in non_reference_sensors if df[s].count() / len(df) >= min_data_fraction]
    valid_sensors = filtered_non_refs + [s for s in references if s in valid_sensors]
    references = [s for s in references if s in valid_sensors]
    if not valid_sensors:
        print(f"⚠️ No valid sensors with sufficient data in {csv_path}, skipping plot")
        return

    # Prepare output directory
    relative = csv_path.relative_to(SOURCE_ROOT)
    out_dir = export_root / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Main time series plot (original logic) ---
    plt.figure(figsize=(12, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    sensor_colors = {}
    for sensor in valid_sensors:
        if sensor in references:
            sensor_colors[sensor] = "red"
        else:
            sensor_colors[sensor] = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1

    for sensor in valid_sensors:
        if sensor.lower().startswith("temp"):
            continue  # skip temperature sensors in main plot
        y_numeric = pd.to_numeric(df[sensor], errors="coerce")
        y_plot = y_numeric
        if sensor in references:
            plt.plot(
                time_vals, y_plot,
                label=f"REF: {sensor}", linewidth=3,
                alpha=0.95, color=sensor_colors[sensor], zorder=10
            )
        else:
            plt.plot(
                time_vals, y_plot,
                label=sensor, alpha=0.8, zorder=1, color=sensor_colors[sensor]
            )

    # Annotate flat, high-humidity regions with temperature reference
    if "reference_rhpc_h" in df.columns and "temp_reference" in df.columns:
        y_ref = pd.to_numeric(df["reference_rhpc_h"], errors="coerce").values
        times = df["time"].values
        temp_vals = pd.to_numeric(df["temp_reference"], errors="coerce").values
        regions = detect_flat_high_regions(y_ref, threshold=85, window=100, std_tol=1.0, min_length=100)
        for (start, end) in regions:
            mid = (start + end) // 2
            if mid >= len(times):
                continue
            flat_time = times[mid]
            flat_temp = temp_vals[mid]
            flat_hum = y_ref[mid]
            label_y = y_min + 0.5 * (flat_hum - y_min)
            annotation = f"{int(round(flat_temp))}°C"
            if np.isfinite(flat_temp) and np.isfinite(flat_hum):
                plt.annotate(
                    annotation,
                    (flat_time, label_y),
                    textcoords="offset points",
                    xytext=(0, -30),
                    ha="center",
                    fontsize=12, color="blue",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.85)
                )

    if y_min < y_max:
        plt.ylim(y_min, y_max)
        y_ticks = np.linspace(y_min, y_max, n_y_ticks)
        plt.gca().set_yticks(y_ticks)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.xlim(time_vals.min(), time_vals.max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax, time_vals)

    plt.xlabel("Time")
    plt.ylabel(f"{csv_path.parts[-3]} (all sensors)")
    plt.title(f"{csv_path.parts[-3]} - ALL SENSORS ({csv_path.stem})")

    # Robust legend: always inside plot, away from lines
    handles, labels = ax.get_legend_handles_labels()
    ref_indices = [i for i, l in enumerate(labels) if l.startswith("REF:")]
    non_ref_indices = [i for i in range(len(labels)) if i not in ref_indices]
    new_order = ref_indices + non_ref_indices
    legend_handles = [handles[i] for i in new_order]
    legend_labels = [labels[i] for i in new_order]
    n_items = len(legend_handles)
    ncol = 1 if n_items <= 10 else 2
    fontsize = 10 if n_items <= 10 else 8
    plt.legend(
        legend_handles, legend_labels,
        loc="best", ncol=ncol, frameon=True, fontsize=fontsize, handlelength=2
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = out_dir / f"allSensors_{csv_path.stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    plt.close()

    # --- Limited time plot (extra plot only) ---
    if PLOT_LIMITED_TIME:
        # Use the same y_min/y_max/y_ticks as above for consistency
        plot_limited_time_allsensors(
            df,
            valid_sensors,
            references,
            time_vals,
            csv_path.parts[-3],
            out_dir,
            sensor_colors,
            y_min,
            y_max,
            np.linspace(y_min, y_max, n_y_ticks),
            n_y_ticks,
            csv_path,
            hours=PLOT_HOURS
        )

    # --- Optionally plot pressure with temperature overlay (original logic) ---
    if PLOT_PRESSURE_WITH_TEMP:
        pressure_sensors = [s for s in valid_sensors if s[-1].lower() == "p"]
        temp_sensors = [s for s in df.columns if s[-1].lower() == "t"]
        if pressure_sensors and temp_sensors:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            for sensor in pressure_sensors:
                y_numeric = pd.to_numeric(df[sensor], errors="coerce")
                color = sensor_colors.get(sensor, None)
                ax1.plot(df["time"], y_numeric, label=sensor, alpha=0.8, zorder=1, color=color)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Pressure")
            ax1.set_title(f"{csv_path.parts[-3]} - Pressure with Temperature Overlay ({csv_path.stem})")
            ax1.grid(True, linestyle="--", alpha=0.5)
            all_temps = np.concatenate([pd.to_numeric(df[s], errors="coerce").dropna().values for s in temp_sensors])
            tmin, tmax = np.nanmin(all_temps), np.nanmax(all_temps)
            ax2 = ax1.twinx()
            for sensor in temp_sensors:
                y_numeric = pd.to_numeric(df[sensor], errors="coerce")
                ax2.plot(df["time"], y_numeric, label=f"TEMP: {sensor}", linestyle="--", alpha=0.7, zorder=2)
            ax2.set_ylabel("Temperature")
            ax2.set_ylim(tmin, tmax)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            all_lines = lines1 + lines2
            all_labels = labels1 + labels2
            n_items = len(all_lines)
            ncol = 1 if n_items <= 10 else 2
            fontsize = 10 if n_items <= 10 else 8
            ax2.legend(
                all_lines, all_labels,
                loc="best", ncol=ncol, frameon=True, fontsize=fontsize, handlelength=2
            )
            save_path = out_dir / f"pressure_with_temp_{csv_path.stem}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {save_path}")
            plt.close()

    # --- Individual sensor type plots (by first three letters, no references) ---
    sensor_groups = {}
    for sensor in valid_sensors:
        if any(ref.lower() in sensor.lower() for ref in reference_sensors):
            continue  # skip references
        if len(sensor) < 3:
            continue
        group = sensor[:3].upper()
        sensor_groups.setdefault(group, []).append(sensor)

    for group, sensors_in_group in sensor_groups.items():
        plt.figure(figsize=(12, 6))
        color_idx = 0
        group_colors = {}
        for sensor in sensors_in_group:
            group_colors[sensor] = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1
        for sensor in sensors_in_group:
            y_numeric = pd.to_numeric(df[sensor], errors="coerce")
            y_plot = y_numeric
            plt.plot(time_vals, y_plot, label=sensor, alpha=0.8, zorder=1, color=group_colors[sensor])
        plt.xlabel("Time")
        plt.ylabel(f"{csv_path.parts[-3]} ({group})")
        plt.title(f"{csv_path.parts[-3]} - {group} SENSORS ({csv_path.stem})")
        plt.ylim(y_min, y_max)
        y_ticks = np.linspace(y_min, y_max, n_y_ticks)
        plt.gca().set_yticks(y_ticks)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
        plt.xlim(time_vals.min(), time_vals.max())
        plt.xticks(rotation=45, ha='right')
        add_edge_labels_if_needed(ax, time_vals)
        plt.legend(loc="best", frameon=True, fontsize=10, handlelength=2)
        plt.grid(True, linestyle="--", alpha=0.5)
        save_path = out_dir / f"group_{group}_{csv_path.stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

# ---------- MAIN SCRIPT ENTRY POINT ---------- #

def main(
    src_root: Path,
    export_root: Path,
    use_hardcoded=True,
    reference_sensors=None,
    restrict_to_reference_time=True,
    plot_individual_groups=True
):
    """
    Process all CSVs in the source root, grouped by folder.
    """
    csv_files = list(src_root.rglob("*.csv"))
    folder_groups = defaultdict(list)
    for f in csv_files:
        folder_groups[f.parent].append(f)
    for folder, files in folder_groups.items():
        y_min, y_max = get_folder_y_limits(files, use_hardcoded, reference_sensors)
        if y_min is None:
            continue
        print(f"\n📁 Folder={folder} | y-axis: {y_min:.2f} to {y_max:.2f}")
        for csv_file in files:
            process_csv(
                csv_file,
                export_root,
                y_min,
                y_max,
                use_hardcoded,
                reference_sensors,
                restrict_to_reference_time=restrict_to_reference_time,
                plot_individual_groups=plot_individual_groups
            )

if __name__ == "__main__":
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    main(
        SOURCE_ROOT,
        EXPORT_ROOT,
        USE_HARDCODED_SUFFIX,
        REFERENCE_SENSORS,
        restrict_to_reference_time=RESTRICT_TO_REFERENCE_TIME,
        plot_individual_groups=PLOT_INDIVIDUAL_GROUPS
    )
