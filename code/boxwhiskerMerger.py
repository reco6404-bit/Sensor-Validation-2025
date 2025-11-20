"""
boxwhiskerMerger.py
------------------------
For each instrument and child folder, generates box and whisker plots
grouped by sensor type (first 3 letters) and instrument (last letter).
Reference sensors are shown as a separate box.
Annotations for min, Q1, median, Q3, max, and IQR are placed with arrows,
alternating sides and nudging to avoid overlap.
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
    plt.figure(figsize=(max(10, n_boxes * 2.5), 7))
    box = plt.boxplot(
        data, 
        patch_artist=True,
        positions=x_positions,
        widths=0.7,
        showmeans=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),   # thinner whiskers
        capprops=dict(linewidth=0.8),       # thinner caps
        medianprops=dict(linewidth=1.2, color="orange"),
        meanprops=dict(marker="D", markerfacecolor="black", markersize=4)
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

    plt.xticks(x_positions, labels, rotation=45, ha="right")

    # --- Annotation offsets tuned for tighter layout ---
    x_offset = 0.6      # right side label distance
    alt_x_offset = -0.6 # left side label distance
    base_fontsize = 8
    min_fontsize = 5
    min_gap = 0.35

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
                 ha='center', va='top', fontsize=8, color='blue',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=1.2))
        plt.text(x, maxv + y_range * 0.03, f"Max\n{maxv:.2f}",
                 ha='center', va='bottom', fontsize=8, color='blue',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=1.2))

        # --- IQR inside the box ---
        iqr_y = (q1 + q3) / 2
        plt.text(
            x, iqr_y, f"IQR\n{iqr:.2f}",
            ha='center', va='center', fontsize=8, color='orange',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=1.2)
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
            plt.annotate(
                f"{label}\n{(iqr if label=='IQR' else y0):.2f}",
                xy=(x, y0),
                xytext=(x_annot, y_annot),
                ha=ha, va='center', fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', lw=1.2),
                arrowprops=dict(arrowstyle="->", color=color, lw=1)
            )

    plt.title(f"{instrument} - {folder}\nSensor Value Distributions")
    plt.ylabel("Sensor Value")
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