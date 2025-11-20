"""
MultiGrapher: Generate line plots from sensor CSV files with consistent 
time alignment and axis scaling.

Workflow:
1. Walk through all folders under `csv_root`.
2. For each folder:
   - Compute the shared min/max time across all CSVs in that folder (for x-axis).
   - Compute the shared min/max y-values per sensor group (based on prefix+suffix).
3. For each CSV file, generate plots for its sensor columns that match
   the folder’s intended variable type.
4. Save output plots into a mirrored folder structure under `destination_root`.

Key sensor matching logic:
- Each CSV folder is assumed to represent one variable type 
  (e.g., `t...` = temperature, `p...` = pressure, `h...` = humidity).
- Sensor-to-folder matching:
  - **Suffix rule:** the last character of the sensor column 
    must match the folder type.
  - Special case for humidity:
    - If the sensor ends in `"h"`, it is valid in both `"h"` and `"r"` folders.
    - This allows humidity (`h`) and relative humidity (`r`) to be grouped together.
- Y-axis scaling:
  - Sensors are grouped by a **prefix+suffix key**:
    - Prefix = first 3 characters of the column name.
    - Suffix = last character of the column name.
  - Example:
    - `temp01_t` → scaling group = `tem_t`
    - `temp02_t` → scaling group = `tem_t`
    - `humid01_h` → scaling group = `hum_h`
    - `humid02_r` → scaling group = `hum_r`
  - This ensures sensors that measure the same variable type with similar IDs 
    share a consistent y-axis range.

Note:
- Missing values (`NaN` or sentinel ≤ -900) are dropped completely.
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import numpy as np
from math import ceil

# ---------------- Configuration ---------------- #
# Pull From:
csv_root = Path("/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv")
# Export To:
destination_root = Path("/Users/reesecoleman/Desktop/UCAR Data/data/exclusive_sensors")
destination_root.mkdir(parents=True, exist_ok=True)

preserve_nesting = True
# If True, output folder structure mirrors input structure.
# If False, all output files go directly into `destination_root`.

skip_non_matching_sensors = True
# If True, only plot sensors whose suffix character matches the folder type(first character).
# If False, plot all sensors in each CSV.
# ------------------------------------------------ #

def get_folder_time_range(folder_path: Path):
    # Return earliest and latest timestamps across all CSVs in a folder.
    min_time, max_time = None, None
    for csv_file in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, usecols=['time'], encoding='utf-8')
        except Exception:
            continue
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        if df.empty:
            continue
        folder_min = df['time'].min()
        folder_max = df['time'].max()
        if min_time is None or folder_min < min_time:
            min_time = folder_min
        if max_time is None or folder_max > max_time:
            max_time = folder_max
    return min_time, max_time


def get_folder_sensor_ranges(folder_path: Path):
    """
    Compute overall min/max values for each sensor group in a folder.
    Grouping key = first 3 chars (prefix) + last char (suffix).
    Example: temp01_t → group "tem_t"
    """
    sensor_ranges = {}

    for csv_file in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if 'time' not in df.columns:
            continue

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])

        sensor_columns = [col for col in df.columns if col not in ['time', 'epoch']]
        for col in sensor_columns:
            series = pd.to_numeric(df[col], errors="coerce")
            series.loc[series <= -900] = np.nan
            series = series.dropna()

            if series.empty:
                continue

            sensor_key = col[:3].lower() + "_" + col[-1].lower()
            col_min, col_max = series.min(), series.max()

            if sensor_key not in sensor_ranges:
                sensor_ranges[sensor_key] = (col_min, col_max)
            else:
                old_min, old_max = sensor_ranges[sensor_key]
                sensor_ranges[sensor_key] = (min(old_min, col_min), max(old_max, col_max))

    return sensor_ranges


def process_csv(csv_file, folder_min_time, folder_max_time, sensor_ranges, skip_non_matching_sensors=skip_non_matching_sensors):
    """Process a single CSV: filter sensors, align time, and generate plots."""
    print(f"Processing {csv_file}")
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if "time" not in df.columns:
        print(f"Skipping {csv_file}: no 'time' column")
        return

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        return

    # Clip to folder time range
    if folder_min_time is not None and folder_max_time is not None:
        folder_min_time = pd.to_datetime(folder_min_time)
        folder_max_time = pd.to_datetime(folder_max_time)
        df = df[(df["time"] >= folder_min_time) & (df["time"] <= folder_max_time)]
        if df.empty:
            return
    else:
        folder_min_time = df["time"].min()
        folder_max_time = df["time"].max()

    try:
        folder_name = csv_file.parents[1].name
    except IndexError:
        folder_name = csv_file.parent.name
    folder_first_letter = folder_name[0].lower() if folder_name else None

    df = df.sort_values("time").drop_duplicates(subset="time", keep="first")

    diffs = df["time"].diff().dt.total_seconds().dropna()
    median_seconds = max(1, int(diffs.median())) if not diffs.empty else 60
    total_seconds = int((folder_max_time - folder_min_time).total_seconds())
    if total_seconds <= 0:
        return

    max_points = 20000
    est_points = total_seconds / median_seconds
    if est_points > max_points:
        median_seconds = ceil(total_seconds / max_points)

    freq_str = "s"
    full_index = pd.date_range(start=folder_min_time, end=folder_max_time, freq=freq_str)

    # Tick marks
    num_ticks = 8
    tick_times = pd.date_range(start=folder_min_time, end=folder_max_time, periods=num_ticks)
    tick_nums = mdates.date2num(tick_times.to_numpy(dtype="datetime64[ns]"))
    pad_days = max(1, int(total_seconds * 0.002)) / 86400.0

    for sensor_column in df.columns:
        if sensor_column.lower() in ("time", "epoch"):
            continue

        # --- Folder matching logic (suffix only) ---
        if skip_non_matching_sensors and folder_first_letter:
            sensor_last = sensor_column[-1].lower()
            folder_first = folder_first_letter.lower()
            if sensor_last == "h":
                if folder_first not in ("h", "r"):
                    continue
            else:
                if sensor_last != folder_first:
                    continue
        # -------------------------------------------

        df_sensor = df[["time", sensor_column]].copy()
        df_sensor[sensor_column] = pd.to_numeric(df_sensor[sensor_column], errors="coerce")
        df_sensor.loc[df_sensor[sensor_column] <= -900, sensor_column] = np.nan

        if df_sensor[sensor_column].dropna().empty:
            continue

        df_sensor = df_sensor.sort_values("time").drop_duplicates(subset="time", keep="first")

        df_reindexed = (
            df_sensor.set_index("time")
            .reindex(full_index)    # preserves NaN for missing times
            .rename_axis("time")
            .reset_index()
        )

        valid_vals = df_reindexed[sensor_column].dropna()
        if valid_vals.empty:
            continue

        # --- Y-axis scaling using prefix+suffix ---
        sensor_key = sensor_column[:3].lower() + "_" + sensor_column[-1].lower()
        if sensor_key in sensor_ranges:
            y_min, y_max = sensor_ranges[sensor_key]
        else:
            y_min, y_max = valid_vals.min(), valid_vals.max()
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        # --- Observed values (real data only) ---
        observed_mask = ~df_reindexed[sensor_column].isna()
        x_obs = mdates.date2num(df_reindexed.loc[observed_mask, "time"].to_numpy(dtype="datetime64[ns]"))
        y_obs = df_reindexed.loc[observed_mask, sensor_column].to_numpy(dtype=float)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_obs, y_obs, color="blue", linewidth=1.5, alpha=0.9, label="observed")

        ax.set_title(f"{sensor_column} - {folder_name}")
        ax.set_ylabel(sensor_column)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6)

        ax.xaxis.set_major_locator(FixedLocator(tick_nums))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%y %H:%M"))
        ax.set_xlim(tick_nums[0] - pad_days, tick_nums[-1] + pad_days)

        fig.autofmt_xdate()
        ax.legend()

        if preserve_nesting:
            rel_path = csv_file.parent.relative_to(csv_root)
            out_dir = destination_root / rel_path
        else:
            out_dir = destination_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{csv_file.stem}_{sensor_column}.png"

        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved graph: {out_file}")


def main():
    # Main driver: walk folders, compute ranges, and generate plots.
    for folder in [csv_root] + [p for p in csv_root.rglob("*") if p.is_dir()]:
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            continue

        folder_min_time, folder_max_time = get_folder_time_range(folder)
        sensor_ranges = get_folder_sensor_ranges(folder)

        for csv_file in csv_files:
            process_csv(csv_file, folder_min_time, folder_max_time, sensor_ranges)


if __name__ == "__main__":
    main()
