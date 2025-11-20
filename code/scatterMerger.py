"""
Generates scatterplots comparing all valid sensor pairs within each folder
(e.g., pre-test or cal-test) for each instrument. 
Reference sensors are labeled in axis, titles, and saved filenames(start with ref_ if the scatter includes a ref sensor). 
Rows with invalid data (NaN) 
are dropped and if there are only a few points, the sensor is not plotted.
Linear fit and R² are shown.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# ---------- Configuration ---------- #
SOURCE_ROOT = "/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv"
EXPORT_ROOT = "/Users/reesecoleman/Desktop/UCAR Data/scatterPlots"
REFERENCE_SENSORS = ["ref"]  # Substrings that mark reference sensors


# ---------- Helpers ---------- #
def get_allowed_suffix(folder_name: str):
    """Allowed suffixes based on instrument folder name."""
    first_letter = folder_name[0].lower()
    if first_letter == "h":
        return ("h", "r")
    if first_letter == "r":
        return ("r", "h")
    return (first_letter,)


def is_reference(sensor_name: str):
    """Return True if sensor contains any reference substring."""
    return any(ref.lower() in sensor_name.lower() for ref in REFERENCE_SENSORS)


def load_csv(file: Path):
    """Load CSV, clean values, and compute elapsed time."""
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")
        return None

    if "time" not in df.columns:
        print(f"⚠️ Missing 'time' column in {file}")
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    elapsed = (df["time"] - df["time"].min()).dt.total_seconds()

    sensor_cols = [c for c in df.columns if c.lower() not in ("time", "epoch")]
    for c in sensor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -900, c] = np.nan

    df = pd.concat([elapsed.rename("elapsed"), df[sensor_cols]], axis=1)
    df = df.dropna(how="all", subset=sensor_cols)  # Keep rows with at least one value
    return df


def plot_pair(x_vals, y_vals, x_label, y_label, out_path):
    """Generate and save a scatterplot for a sensor pair."""
    # Filter invalid values
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals, y_vals = x_vals[mask], y_vals[mask]

    if len(x_vals) < 3 or np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        print(f"⚠️ Skipped plot {out_path.name}: insufficient or degenerate data ({len(x_vals)} points)")
        return False

    try:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
    except np.linalg.LinAlgError as e:
        print(f"❌ Polyfit failed for {out_path.name}: {e}")
        return False

    y_pred = slope * x_vals + intercept
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r2_val = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, s=20, alpha=0.6, label="Data Points")
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    plt.plot(x_line, slope * x_line + intercept, color="red", linewidth=2,
             label=f"Fit: y={slope:.2f}x + {intercept:.2f}")
    plt.text(0.05, 0.93, f"R² = {r2_val:.3f}", transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Scatterplot: {x_label} vs {y_label}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot: {out_path}")
    return True


# ---------- Main ---------- #
def main(src_root: Path, export_root: Path):
    csv_paths = list(src_root.rglob("*.csv"))
    print(f"🔎 Found {len(csv_paths)} CSV files.")
    groups = {}
    for csv_path in csv_paths:
        try:
            instrument = csv_path.parents[1].name
            folder = csv_path.parent.name
        except IndexError:
            continue
        groups.setdefault((instrument, folder), []).append(csv_path)

    for (instrument, folder), files in groups.items():
        allowed_suffix = get_allowed_suffix(instrument)
        print(f"\n📁 Processing Instrument={instrument}, Folder={folder}, Allowed suffixes={allowed_suffix}")

        for csv_file in files:
            df = load_csv(csv_file)
            if df is None or df.empty:
                print(f"⚠️ Skipping {csv_file}: no valid data")
                continue

            sensors = [c for c in df.columns if c.lower() not in ("elapsed", "time", "epoch")]
            filtered = [s for s in sensors if s[-1].lower() in allowed_suffix]
            if len(filtered) < 2:
                print(f"⚠️ Skipping {csv_file}: not enough matching sensors")
                continue

            for s1, s2 in combinations(filtered, 2):
                ref1, ref2 = is_reference(s1), is_reference(s2)
                label1 = f"{s1} (Reference)" if ref1 else s1
                label2 = f"{s2} (Reference)" if ref2 else s2

                # If either sensor is a temperature sensor, align on temp_reference
                temp_ref_col = "temp_reference"
                is_temp_scatter = s1.endswith("t") or s2.endswith("t")
                if is_temp_scatter and temp_ref_col in df.columns:
                    temp_ref = pd.to_numeric(df[temp_ref_col], errors="coerce")
                    valid_mask = np.isfinite(temp_ref)
                    if not valid_mask.any():
                        print(f"⚠️ Skipping {s1} vs {s2} in {csv_file.name}: temp_reference is empty")
                        continue
                    vals1 = df[s1][valid_mask].values
                    vals2 = df[s2][valid_mask].values
                    # Also mask out NaNs in the sensors themselves
                    mask = np.isfinite(vals1) & np.isfinite(vals2)
                    vals1, vals2 = vals1[mask], vals2[mask]
                else:
                    # Use all available data, interpolate to common time
                    common_time = np.linspace(df["elapsed"].min(), df["elapsed"].max(), 500)
                    vals1 = np.interp(common_time, df["elapsed"], df[s1])
                    vals2 = np.interp(common_time, df["elapsed"], df[s2])

                ref_tag = "ref_" if (ref1 or ref2) else ""
                out_dir = export_root / instrument / folder
                plot_name = f"{ref_tag}{s1}_vs_{s2}{csv_file.stem}.png"
                plot_path = out_dir / plot_name

                success = plot_pair(vals1, vals2, label1, label2, plot_path)
                if not success:
                    print(f"⚠️ Ignored: {s1} vs {s2} in {csv_file.name}")


if __name__ == "__main__":
    src = Path(SOURCE_ROOT)
    exp = Path(EXPORT_ROOT)
    exp.mkdir(parents=True, exist_ok=True)
    main(src, exp)
