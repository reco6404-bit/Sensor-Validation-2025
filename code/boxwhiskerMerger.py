"""boxwhiskerMerger.py

Generate readable box-and-whisker plots for each folder of formatted CSVs.

This script groups sensors by type (first three characters of the column
name) and plots a box for each group plus an optional REFERENCE box. It
places boxed textual annotations (Min, Max, IQR, Q1, Median, Mean) near each
box; arrows have been removed in favor of boxed text for cleaner visuals.

Quick usage (from project root):
    python3 code/boxwhiskerMerger.py

Output
 - PNG files are saved under `boxPlots/<Instrument>/<Folder>/boxplot_<csv_stem>.png`.

Design notes for a first-time reader
 - Sensor naming convention: the last character of a column name is treated
     as the sensor suffix (e.g., 't' for temp, 'p' for pressure, 'r' for RH).
 - Reference sensor detection uses substrings listed in `REFERENCE_KEYWORDS`.
 - `base_fontsize` is computed once and applied consistently to ticks, title,
     labels, and annotation text so the visual scale is predictable across plots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Configuration ---------- #
SOURCE_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv")
EXPORT_ROOT = Path("/Users/reesecoleman/Desktop/UCAR Data/boxPlots")
REFERENCE_KEYWORDS = ["ref"]  # Substrings that mark reference sensors

def is_reference(sensor_name):
    """Return True if sensor contains any reference substring."""
    return any(ref.lower() in str(sensor_name).lower() for ref in REFERENCE_KEYWORDS)

def get_sensor_type(sensor_name):
    """Return the first three letters as the sensor type/class."""
    return str(sensor_name)[:3].lower()

def get_instrument_suffix(instrument_name):
    """Return the instrument suffix (e.g., 'p' for pressure)."""
    return str(instrument_name)[0].lower()

def get_allowed_suffix(folder_name: str):
    """Allowed suffixes based on instrument folder name."""
    first_letter = folder_name[0].lower()
    if first_letter == "h":
        return ("h", "r")
    if first_letter == "r":
        return ("r", "h")
    return (first_letter,)


def get_unit_for_instrument(instrument_name: str):
    """Return an easily readable unit string for the given instrument folder/name.

    Recognized units:
    - Temperature -> °C
    - Pressure -> millibars
    - RH / Relative Humidity -> % RH
    Returns an empty string when unknown.
    """
    if not instrument_name:
        return ""
    name = str(instrument_name).lower()
    if "temp" in name or "temperature" in name:
        return "°C"
    if "press" in name or "pressure" in name or "mb" in name:
        return "millibars"
    if "rh" in name or "relative" in name or "humidity" in name:
        return "% RH"
    return ""

def process_csv(csv_path, instrument, folder, export_root):
    """
    For a single CSV, group sensors by type and instrument,
    and plot boxplots for each group and the reference.
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read {csv_path}: {e}")
        return

    # Drop non-sensor columns
    for col in ["time", "epoch"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    allowed_suffixes = get_allowed_suffix(instrument)
    # Clean and filter sensor columns
    sensor_cols = [c for c in df.columns if not is_reference(c)]
    sensor_groups = {}
    for col in sensor_cols:
        if not isinstance(col, str) or len(col) < 4:
            continue  # skip malformed names
        if col[-1].lower() not in allowed_suffixes:
            continue  # skip if not matching allowed suffix
        group = get_sensor_type(col)
        # Clean values: convert to numeric, NaN for ≤ -900
        vals = pd.to_numeric(df[col], errors="coerce")
        vals[vals <= -900] = np.nan
        sensor_groups.setdefault(group, []).append(vals)

    # Collect data for each group
    data, labels, colors = [], [], []
    for group, vals_list in sorted(sensor_groups.items()):
        vals = pd.concat(vals_list)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            continue
        data.append(vals)
        labels.append(group.upper())
        colors.append("#99ccff")  # blue for normal sensors

    # Add reference sensors as a separate box
    ref_cols = [c for c in df.columns if is_reference(c)]
    ref_vals_list = []
    for c in ref_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals[vals <= -900] = np.nan
        ref_vals_list.append(vals)
    if ref_vals_list:
        ref_vals = pd.concat(ref_vals_list)
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if len(ref_vals) >= 3:
            data.append(ref_vals)
            labels.append("REFERENCE")
            colors.append("red")

    if not data:
        print(f"⚠️ No valid sensor groups in {csv_path}")
        return

    # --- Custom box spacing positions ---
    n_boxes = len(data)
    box_spacing = 2.5
    x_positions = np.arange(1, n_boxes * box_spacing + 1, box_spacing)

    # --- Plot boxplot ---
    # Make boxes and plot elements larger for readability (increased sizing)
    plt.figure(figsize=(max(12, n_boxes * 6.0), 16))
    box = plt.boxplot(
        data,
        patch_artist=True,
        positions=x_positions,
        widths=1.8,  # wider boxes
        showmeans=True,
        boxprops=dict(linewidth=3.6),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        medianprops=dict(linewidth=3.2, color="black"),
        meanprops=dict(marker="D", markerfacecolor="black", markersize=10)
    )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # --- Adjust y limits with margin for labels ---
    all_min = min(np.min(vals) for vals in data)
    all_max = max(np.max(vals) for vals in data)
    y_range = all_max - all_min if all_max > all_min else 1
    y_margin = y_range * 0.25
    plt.ylim(all_min - y_margin, all_max + y_margin)

    # --- Tight x margins (no wasted space) ---
    label_margin = 1.2  # smaller margin to prevent squishing
    plt.xlim(x_positions[0] - label_margin, x_positions[-1] + label_margin)

    # Determine base font size (double the default matplotlib font size) and other
    # annotation offsets so they are available before tick/label calls.
    base_fontsize = int(plt.rcParams.get('font.size', 10) * 2)
    min_fontsize = 10
    min_gap = 0.35
    x_offset = 1.2      # right side label distance (scaled up)
    alt_x_offset = -1.2 # left side label distance

    plt.xticks(x_positions, labels, rotation=45, ha="right")
    # Make tick labels larger
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=base_fontsize)
    ax.tick_params(axis='y', labelsize=base_fontsize)

    # --- Annotate each box with non-overlapping labels ---
    from matplotlib.transforms import Bbox

    directions = [
        (0, 1),   # up
        (0, -1),  # down
        (1, 0),   # right
        (-1, 0),  # left
        (1, 1),   # up-right
        (-1, 1),  # up-left
        (1, -1),  # down-right
        (-1, -1), # down-left
    ]
    shift_amount_y = min_gap * 1.2
    shift_amount_x = 0.5

    for i, vals in enumerate(data):
        vals = np.sort(vals)
        minv, maxv = np.min(vals), np.max(vals)
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        meanv = np.mean(vals)
        x = x_positions[i]

        # --- Min/Max as simple labels, no arrows, no shifting ---
        plt.text(x, minv - y_range * 0.03, f"Min\n{minv:.2f}",
                 ha='center', va='top', fontsize=base_fontsize, color='blue',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=2.4))
        plt.text(x, maxv + y_range * 0.03, f"Max\n{maxv:.2f}",
                 ha='center', va='bottom', fontsize=base_fontsize, color='blue',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=2.4))

        # --- IQR inside the box ---
        iqr_y = (q1 + q3) / 2
        # Use a dark color (not yellow/orange) for IQR so it is legible on white
        iqr_color = 'black'
        plt.text(
            x, iqr_y, f"IQR\n{iqr:.2f}",
            ha='center', va='center', fontsize=base_fontsize, color=iqr_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=2.4)
        )

        # --- Q1, Median, Q3, Mean: alternate right/left, stack vertically, always point to stat ---
        # Dynamically order mean/median so mean is below median if mean < median, else above
        if meanv < median:
            label_defs = [
                ("Q1", q1, 'purple', 0.7, 'left'),
                ("Mean", meanv, 'darkgreen', -0.7, 'right'),
                ("Median", median, 'black', -0.7, 'right'),
                ("Q3", q3, 'purple', 0.7, 'left'),
            ]
        else:
            label_defs = [
                ("Q1", q1, 'purple', 0.7, 'left'),
                ("Median", median, 'black', -0.7, 'right'),
                ("Mean", meanv, 'darkgreen', -0.7, 'right'),
                ("Q3", q3, 'purple', 0.7, 'left'),
            ]
        n_labels = len(label_defs)
        stack_gap = y_range * 0.10  # vertical gap between stacked labels
        stack_center = (q1 + q3) / 2
        stack_positions = []
        for idx in range(n_labels):
            offset = (idx - (n_labels - 1) / 2) * stack_gap
            stack_positions.append(stack_center + offset)

        for (label, y0, color, x_offset, ha), y_annot in zip(label_defs, stack_positions):
            x_annot = x + x_offset
            # Ensure text color is not yellow (poor contrast on white)
            bad_colors = {"yellow", "#ffff00", "brightyellow", "gold"}
            color_safe = color if (isinstance(color, str) and color.lower() not in bad_colors) else "black"
            # Place the label near the statistic without an arrow connector.
            # Use text with a boxed background so it remains legible.
            plt.text(
                x_annot, y_annot,
                f"{label}\n{(iqr if label=='IQR' else y0):.2f}",
                ha=ha, va='center', fontsize=base_fontsize, color=color_safe,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=2.4)
            )

    # Include units in the y-label when available
    unit = get_unit_for_instrument(instrument)
    y_label = f"Sensor Value ({unit})" if unit else "Sensor Value"
    plt.title(f"{instrument} - {folder}\nSensor Value Distributions", fontsize=base_fontsize+2)
    plt.ylabel(y_label, fontsize=base_fontsize)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save plot
    out_dir = export_root / instrument / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"boxplot_{csv_path.stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved boxplot: {out_path}")


def main(src_root, export_root):
    """
    For each instrument and child folder, process all CSVs and generate boxplots.
    """
    csv_paths = list(src_root.rglob("*.csv"))
    print(f"🔎 Found {len(csv_paths)} CSV files.")
    for csv_path in csv_paths:
        try:
            instrument = csv_path.parents[1].name
            folder = csv_path.parent.name
        except IndexError:
            continue
        process_csv(csv_path, instrument, folder, export_root)

if __name__ == "__main__":
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    main(SOURCE_ROOT, EXPORT_ROOT)