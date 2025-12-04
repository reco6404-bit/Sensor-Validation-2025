"""graphMerger.py

Lightweight utilities to generate time-series plots from formatted CSV files.
This module reads per-instrument CSVs, selects sensors and reference series,
computes sensible y-axis limits/ticks, and writes PNGs. The code is
organized as small helpers (CSV loading, tick generation, small plotting
conveniences) combined in `process_csv` which produces the primary plots.

The implementation is defensive: non-numeric data are ignored, plotting
functions align series lengths before plotting, and axis presets can be
used to force consistent tick intervals for specific instruments.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FixedLocator, MaxNLocator, FormatStrFormatter
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
USE_AXIS_PRESETS = True  # Set to True to activate axis presets
AXIS_PRESETS = {
    # Humidity instruments: force 10..100 with 10-unit ticks
    "rh_test": (10, 100, 10),
    "humidity": (10, 100, 10),
    # Pressure instruments: force 825..840 with 1-unit ticks
    "pressure": (825, 840, 1),
   # Temperature instruments: force -25..50 with 5-degree ticks
    "temperature_test": (-25, 50, 5),
}

# Temperature tick step (degrees) used when rounding temperature axes
TEMP_TICK_STEP = 5


def get_unit_for_instrument(instrument: str) -> str:
    """Return a short unit string for a given instrument name.
    Known mappings:
      - humidity / rh_test -> "% RH"
      - temperature / temperature_test -> "°C"
      - pressure -> "millibars"
    """
    if not instrument:
        return ""
    inst = instrument.lower()
    if "humidity" in inst or "rh_test" in inst or inst.startswith("rh"):
        return "% RH"
    if "temperature" in inst or inst.startswith("temp") or inst.endswith("_t"):
        return "°C"
    if "pressure" in inst or inst.startswith("press") or inst.startswith("lps"):
        return "millibars"
    return ""

PLOT_LIMITED_TIME = False  # Set to True to enable limited time plot
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
    # This helper was present historically. Legend placement is handled
    # inline in the plotting code using matplotlib's automatic placement
    # now. Keep a simple return for backward compatibility.
    return "best", 1, 10, 2  # loc, ncol, fontsize, handlelength

def get_nice_axis_limits_and_ticks(data, n_ticks=6, instrument=None, force_no_preset=False):
    """
    Given a 1D array, return (ymin, ymax, yticks) with nice rounded limits and regular steps.
    Axis will start at a multiple of 0.5 if possible.
    If AXIS_PRESETS is set for the instrument and USE_AXIS_PRESETS is True, use those values unless force_no_preset is True.
    """
    # Prefer using axis presets when configured
    if USE_AXIS_PRESETS and not force_no_preset and instrument is not None and instrument.lower() in AXIS_PRESETS:
        ymin, ymax, step = AXIS_PRESETS[instrument.lower()]
        yticks = np.arange(ymin, ymax + step * 0.5, step)
        return ymin, ymax, yticks

    finite_data = np.array(data)[np.isfinite(data)]
    if finite_data.size == 0:
        return -1, 1, np.arange(-1, 1.1, 0.5)
    dmin, dmax = float(np.min(finite_data)), float(np.max(finite_data))

    # If data are essentially zero, keep a small symmetric window
    if abs(dmax - dmin) < 1e-9 and abs(dmax) < 1e-6:
        return -1, 1, np.arange(-1, 1.1, 0.5)

    span = dmax - dmin

    # Prefer whole-number ticks for normal ranges; only use fractional ticks when
    # the data span is very small (< 1). This makes labeling proactive (whole numbers)
    # but allows fine steps for tiny ranges.
    fractional_steps = [0.1, 0.2, 0.5, 1.0]
    integer_steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # If the data span is very small (<1), allow fractional steps; otherwise prefer integer steps
    candidate_steps = fractional_steps if span < 1.0 else integer_steps

    chosen_step = None
    chosen_ymin = None
    chosen_ymax = None

    # Try steps in order to find a reasonable number of ticks (4..12)
    for step in candidate_steps:
        y_min = np.floor(dmin / step) * step
        y_max = np.ceil(dmax / step) * step
        if y_max <= y_min:
            y_max = y_min + step
        n_ticks_try = int(round((y_max - y_min) / step)) + 1
        if 4 <= n_ticks_try <= 12:
            chosen_step = step
            chosen_ymin = y_min
            chosen_ymax = y_max
            break

    # Fallback: if nothing matched, pick a sensible default (largest candidate)
    if chosen_ymin is None:
        step = candidate_steps[-1]
        chosen_step = step
        chosen_ymin = np.floor(dmin / step) * step
        chosen_ymax = np.ceil(dmax / step) * step
        if chosen_ymax <= chosen_ymin:
            chosen_ymax = chosen_ymin + step

    yticks = np.arange(chosen_ymin, chosen_ymax + chosen_step * 0.5, chosen_step)

    # Special-case behavior by instrument type for nicer axes
    if instrument is not None:
        inst = instrument.lower()
        try:
            # humidity instruments -> integer % RH ticks
            if "humidity" in inst or inst.startswith("rh") or "rh_test" in inst:
                ymin_i = int(math.floor(dmin))
                ymax_i = int(math.ceil(dmax))
                if ymin_i == ymax_i:
                    ymin_i -= 1
                    ymax_i += 1
                yticks = np.arange(ymin_i, ymax_i + 1, 1)
                return float(ymin_i), float(ymax_i), yticks.astype(int)
            # temperature instruments -> multiples of TEMP_TICK_STEP
            if "temp" in inst or inst.endswith("_t") or "temperature" in inst:
                step = TEMP_TICK_STEP
                ymin_t = int(math.floor(dmin / step) * step)
                ymax_t = int(math.ceil(dmax / step) * step)
                if ymax_t <= ymin_t:
                    ymax_t = ymin_t + step
                yticks = np.arange(ymin_t, ymax_t + step, step)
                max_ticks = 8
                if len(yticks) > max_ticks:
                    s = int(np.ceil(len(yticks) / float(max_ticks)))
                    yticks = yticks[::s]
                return float(ymin_t), float(ymax_t), yticks.astype(int)
        except Exception:
            pass

    # Ensure the last tick is at or above the data max so plotted lines sit below the top tick
    try:
        if len(yticks) == 0:
            yticks = np.array([chosen_ymin, chosen_ymax])
        if float(yticks[-1]) < float(dmax) - 1e-9:
            step = chosen_step if 'chosen_step' in locals() and chosen_step is not None else (yticks[-1] - yticks[-2] if len(yticks) >= 2 else (float(chosen_ymax) - float(chosen_ymin)))
            append_tick = float(yticks[-1]) + float(step)
            yticks = np.concatenate([yticks, np.array([append_tick])])
        if np.all(np.isfinite(yticks)) and np.all(np.isclose(yticks, np.round(yticks))):
            yticks = yticks.astype(int)
    except Exception:
        pass

    # Ensure returned ymax covers the highest tick (we may have appended a tick)
    try:
        chosen_ymin = float(chosen_ymin)
        chosen_ymax = float(chosen_ymax)
        if len(yticks) > 0:
            chosen_ymax = max(chosen_ymax, float(yticks[-1]))
            chosen_ymin = min(chosen_ymin, float(yticks[0]))
    except Exception:
        pass
    return float(chosen_ymin), float(chosen_ymax), yticks


def ensure_legible_yticks(ax, ticks=None, prefer_integer=True, max_ticks=8, preserve_ticks=False):
    """
    Ensure the axis `ax` has legible, whole-number y-ticks that do not overlap.
    - ticks: array-like of desired tick positions (may be numeric). If None, use current ticks.
    - prefer_integer: if True, round ticks to integers when they are close to integer values.
    - max_ticks: maximum number of ticks to display; will subsample evenly if needed.

    This function sets a FixedLocator on the axis to lock positions and applies an
    integer formatter when possible. It also reduces font-size slightly when many
    ticks are present to avoid crowding.
    """
    try:
        if ticks is None:
            current = np.array(ax.get_yticks(), dtype=float)
        else:
            current = np.array(ticks, dtype=float)

        # Prefer integer ticks when values are near integers
        if prefer_integer and np.all(np.isfinite(current)) and np.allclose(current, np.round(current), atol=1e-6):
            current = np.round(current).astype(int).astype(float)

        # If too many ticks, subsample evenly but keep the last tick
        # Unless preserve_ticks is True (used for manual axis presets such as temperature)
        if len(current) > max_ticks and not preserve_ticks:
            n = len(current)
            idxs = np.unique(np.round(np.linspace(0, n - 1, max_ticks)).astype(int))
            current = current[idxs]

        # If the last tick is below max plotted data, caller should ensure ticks include top; we only ensure formatting here.
        # Apply ticks and formatter
        try:
            ax.yaxis.set_major_locator(FixedLocator(current))
        except Exception:
            ax.set_yticks(current)

        # If integer-like, use integer formatter
        if prefer_integer and np.all(np.isclose(current, np.round(current), atol=1e-6)):
            try:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            except Exception:
                pass

        # Reduce label size slightly if many ticks
        try:
            if len(current) > 6:
                ax.tick_params(axis='y', labelsize=max(7, 10 - (len(current) - 6)))
            else:
                ax.tick_params(axis='y', labelsize=10)
        except Exception:
            pass
        return current
    except Exception:
        return None


def place_legend_smart(ax, fig=None, handles=None, labels=None, ncol=1, fontsize=10, handlelength=2, min_fontsize=6, prefer_outside_threshold=6):
    """
    Place a legend on `ax` but avoid overlapping plotted lines.
    Strategy:
      - Try matplotlib's 'best' placement first.
      - If the legend overlaps any plotted line (sampled), progressively shrink the font.
      - If shrinking doesn't help, move the legend outside the plot on the right and
        shrink to `min_fontsize`. Adjust subplots to make room.

    Returns the legend object.
    """
    if fig is None:
        fig = ax.get_figure()
    try:
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        n_labels = 0 if labels is None else len(labels)

        # If there are several labels, prefer placing the legend outside immediately
        if n_labels >= prefer_outside_threshold:
            try:
                # center vertically so it doesn't occlude tall peaks
                loc_out = "center left"
                bbox = (1.02, 0.5)
                # reduce fontsize a bit to keep compact
                fs = max(min_fontsize, int(fontsize * 0.9))
                legend = ax.legend(handles, labels, loc=loc_out, bbox_to_anchor=bbox, ncol=ncol, frameon=True, fontsize=fs, handlelength=handlelength)
                # make room for the legend: widen the right margin more for larger legends
                try:
                    if n_labels > 12:
                        fig.subplots_adjust(right=0.62)
                    else:
                        fig.subplots_adjust(right=0.72)
                except Exception:
                    pass
                return legend
            except Exception:
                pass

        # create initial legend with automatic placement
        legend = ax.legend(handles, labels, loc="best", ncol=ncol, frameon=True, fontsize=fontsize, handlelength=handlelength)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer)

            # decide which line objects to check: prefer given handles if they are Line2D
            lines_to_check = []
            for h in handles:
                try:
                    if hasattr(h, 'get_xdata') and hasattr(h, 'get_ydata'):
                        lines_to_check.append(h)
                except Exception:
                    continue
            if not lines_to_check:
                lines_to_check = ax.get_lines()

            # helper to test overlap
            def legend_overlaps():
                for line in lines_to_check:
                    try:
                        xdata = np.asarray(line.get_xdata(), dtype=float)
                        ydata = np.asarray(line.get_ydata(), dtype=float)
                    except Exception:
                        continue
                    if xdata.size == 0:
                        continue
                    idxs = np.linspace(0, xdata.size - 1, min(50, xdata.size)).astype(int)
                    pts = np.column_stack([xdata[idxs], ydata[idxs]])
                    disp = ax.transData.transform(pts)
                    xs, ys = disp[:, 0], disp[:, 1]
                    if np.any((xs >= legend_box.x0) & (xs <= legend_box.x1) & (ys >= legend_box.y0) & (ys <= legend_box.y1)):
                        return True
                return False

            if legend_overlaps():
                # try shrinking first
                for fs in range(int(fontsize) - 1, int(min_fontsize) - 1, -1):
                    legend.remove()
                    legend = ax.legend(handles, labels, loc="best", ncol=ncol, frameon=True, fontsize=fs, handlelength=handlelength)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    legend_box = legend.get_window_extent(renderer)
                    if not legend_overlaps():
                        return legend
                # still overlapping: create a separate legend axis at the right
                try:
                    legend.remove()
                except Exception:
                    pass
                try:
                    # create an axis on the right for the legend and draw there so it
                    # cannot overlap the data. Width depends on label count.
                    if n_labels > 12:
                        left = 0.60
                        width = 0.35
                    else:
                        left = 0.78
                        width = 0.22
                    leg_ax = fig.add_axes([left, 0.08, width, 0.84])
                    leg_ax.axis('off')
                    legend = leg_ax.legend(handles, labels, loc='center', ncol=ncol, frameon=True, fontsize=min_fontsize, handlelength=handlelength)
                    return legend
                except Exception:
                    # fallback to previous outside placement
                    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=ncol, frameon=True, fontsize=min_fontsize, handlelength=handlelength)
                    try:
                        fig.subplots_adjust(right=0.75)
                    except Exception:
                        pass
                    return legend
        except Exception:
            return legend
    except Exception:
        try:
            # fallback to a simple legend call
            return ax.legend()
        except Exception:
            return None
    return legend

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
    units = get_unit_for_instrument(instrument)
    ylabel = f"{instrument} ({units})" if units else f"{instrument} (all sensors)"
    plt.ylabel(ylabel)
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
    ax = plt.gca()
    ensure_legible_yticks(ax, y_ticks, prefer_integer=True, max_ticks=8)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.xlim(time_vals[mask].min(), time_vals[mask].max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax, time_vals[mask])
    handles, labels = ax.get_legend_handles_labels()
    n_items_lt = len(labels)
    ncol_lt = 1 if n_items_lt <= 10 else 2
    fontsize_lt = 10 if n_items_lt <= 10 else 8
    legend = place_legend_smart(ax, fig=plt.gcf(), handles=handles, labels=labels, ncol=ncol_lt, fontsize=fontsize_lt, handlelength=2, prefer_outside_threshold=4)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = out_dir / f"allSensors_{file_str}_{csv_path.stem}.png"
    # Safety: ensure top tick covers data in the limited-time view
    try:
        # collect plotted numeric values within mask
        vals = []
        for s in valid_sensors:
            arr = pd.to_numeric(df[s], errors='coerce')
            arrm = arr[mask].dropna().values
            if arrm.size:
                vals.append(arrm)
        if vals:
            data_max = float(np.nanmax(np.concatenate(vals)))
            current_ticks = np.array(y_ticks) if y_ticks is not None else np.array(plt.gca().get_yticks())
            if current_ticks.size > 0 and float(current_ticks[-1]) < data_max - 1e-9:
                try:
                    inst_key = instrument.lower() if instrument is not None else None
                    if USE_AXIS_PRESETS and inst_key in AXIS_PRESETS:
                        preset_step = float(AXIS_PRESETS[inst_key][2])
                        top_tick = float(math.ceil(data_max / preset_step) * preset_step)
                        if top_tick <= float(current_ticks[-1]):
                            top_tick = float(current_ticks[-1]) + abs(preset_step)
                        start_tick = float(current_ticks[0])
                        new_ticks = np.arange(start_tick, top_tick + preset_step * 0.1, preset_step)
                        plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                        ax = plt.gca()
                        ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=100, preserve_ticks=True)
                    else:
                        if len(current_ticks) >= 2:
                            step_local = float(current_ticks[-1] - current_ticks[-2])
                        else:
                            step_local = max(1.0, math.ceil(data_max) - float(current_ticks[-1]))
                        top_tick = float(math.ceil(data_max / step_local) * step_local)
                        if top_tick <= current_ticks[-1]:
                            top_tick = float(current_ticks[-1]) + abs(step_local)
                        new_ticks = np.concatenate([current_ticks, np.array([top_tick])])
                        plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                        ax = plt.gca()
                        ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=8)
                except Exception:
                    if len(current_ticks) >= 2:
                        step_local = float(current_ticks[-1] - current_ticks[-2])
                    else:
                        step_local = max(1.0, math.ceil(data_max) - float(current_ticks[-1]))
                    top_tick = float(math.ceil(data_max / step_local) * step_local)
                    if top_tick <= current_ticks[-1]:
                        top_tick = float(current_ticks[-1]) + abs(step_local)
                    new_ticks = np.concatenate([current_ticks, np.array([top_tick])])
                    plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                    ax = plt.gca()
                    ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=8)
    except Exception:
        pass
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

    # instrument name (folder) used for presets and labels
    instrument = csv_path.parts[-3]

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
    # Reserve red-like colors for reference series only. Matplotlib's default
    # cycle contains a red hex (#d62728) which we don't want used for non-ref
    # sensors so references remain visually distinct.
    RESERVED_RED_NAMES = {"red", "tab:red", "crimson", "firebrick", "maroon", "darkred"}
    RESERVED_RED_HEX = {"#d62728", "#ff0000", "#8b0000"}

    def pick_nonred_color(cycle, start_idx):
        """
        Pick the next color from the cycle that is not in the reserved-red sets.
        Returns (color, next_index).
        """
        n = len(cycle)
        for i in range(n):
            idx = (start_idx + i) % n
            c = cycle[idx]
            try:
                cl = c.lower()
            except Exception:
                cl = str(c)
            if cl in RESERVED_RED_NAMES or cl in RESERVED_RED_HEX:
                continue
            # return this color and the index after it
            return c, (idx + 1) % n
        # fallback: nothing suitable found, return the original next entry
        return cycle[start_idx % n], (start_idx + 1) % n

    for sensor in valid_sensors:
        if sensor in references:
            sensor_colors[sensor] = "red"
        else:
            color, color_idx = pick_nonred_color(color_cycle, color_idx)
            sensor_colors[sensor] = color

    # Compute nice y-limits and ticks based on available numeric data (prefer whole numbers)
    y_min_rounded, y_max_rounded, y_ticks = None, None, None
    try:
        combined_vals = []
        for s in valid_sensors:
            arr = pd.to_numeric(df[s], errors='coerce').dropna().values
            if arr.size:
                combined_vals.append(arr)
        if combined_vals:
            combined = np.concatenate(combined_vals)
            y_min_rounded, y_max_rounded, y_ticks = get_nice_axis_limits_and_ticks(combined, n_ticks=n_y_ticks, instrument=instrument)
    except Exception:
        y_min_rounded, y_max_rounded, y_ticks = None, None, None

    for sensor in valid_sensors:
        if sensor.lower().startswith("temp"):
            continue  # skip temperature sensors in main plot
        y_numeric = pd.to_numeric(df[sensor], errors="coerce")
        y_plot = y_numeric
        if sensor in references:
            # Use a short generic REF label in legends instead of the full column name
            plt.plot(
                time_vals, y_plot,
                label="REF", linewidth=3,
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

    # Use rounded y-limits/ticks when available; otherwise fall back to provided folder y_min/y_max
    if y_min_rounded is not None and y_max_rounded is not None:
        plt.ylim(y_min_rounded, y_max_rounded)
        ax = plt.gca()
        # enforce legible integer ticks
        # Preserve ticks for any instrument with an explicit AXIS_PRESET
        preserve_flag = False
        try:
            if USE_AXIS_PRESETS and instrument is not None and instrument.lower() in AXIS_PRESETS:
                preserve_flag = True
        except Exception:
            preserve_flag = False
        _res = ensure_legible_yticks(ax, y_ticks, prefer_integer=True, max_ticks=8, preserve_ticks=preserve_flag)
        if _res is not None:
            y_ticks = _res
    elif y_min < y_max:
        plt.ylim(y_min, y_max)
        y_ticks = np.linspace(y_min, y_max, n_y_ticks)
        ax = plt.gca()
        # For limited-time plots, preserve ticks when a matching preset exists
        preserve_flag = False
        try:
            if USE_AXIS_PRESETS and instrument is not None and instrument.lower() in AXIS_PRESETS:
                preset_step = AXIS_PRESETS[instrument.lower()][2]
                if int(preset_step) == int(TEMP_TICK_STEP):
                    preserve_flag = True
        except Exception:
            preserve_flag = False
        ensure_legible_yticks(ax, y_ticks, prefer_integer=True, max_ticks=8, preserve_ticks=preserve_flag)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    plt.xlim(time_vals.min(), time_vals.max())
    plt.xticks(rotation=45, ha='right')
    add_edge_labels_if_needed(ax, time_vals)

    plt.xlabel("Time")
    units = get_unit_for_instrument(instrument)
    ylabel = f"{instrument} ({units})" if units else f"{instrument} (all sensors)"
    plt.ylabel(ylabel)
    plt.title(f"{instrument} - ALL SENSORS ({csv_path.stem})")

    # Robust legend: always inside plot, away from lines
    handles, labels = ax.get_legend_handles_labels()
    # detect REF entries that may be labelled as 'REF' or 'REF: <name>'
    ref_indices = [i for i, l in enumerate(labels) if isinstance(l, str) and l.startswith("REF")]
    non_ref_indices = [i for i in range(len(labels)) if i not in ref_indices]
    new_order = ref_indices + non_ref_indices
    legend_handles = [handles[i] for i in new_order]
    legend_labels = [labels[i] for i in new_order]
    n_items = len(legend_handles)
    ncol = 1 if n_items <= 10 else 2
    fontsize = 10 if n_items <= 10 else 8
    place_legend_smart(ax, fig=plt.gcf(), handles=legend_handles, labels=legend_labels, ncol=ncol, fontsize=fontsize, handlelength=2, prefer_outside_threshold=4)

    plt.grid(True, linestyle="--", alpha=0.5)
    # Final safety: ensure the highest y tick is at or above the actual plotted data max
    try:
        if 'combined' in locals() and combined is not None and combined.size:
            data_max = float(np.nanmax(combined))
            # current ticks
            current_ticks = np.array(y_ticks) if y_ticks is not None else np.array(plt.gca().get_yticks())
            if current_ticks.size > 0 and float(current_ticks[-1]) < data_max - 1e-9:
                # If this instrument has an axis preset, expand the preset to the
                # next multiple of the preset step so ticks remain uniform.
                try:
                    inst_key = instrument.lower() if instrument is not None else None
                    if USE_AXIS_PRESETS and inst_key in AXIS_PRESETS:
                        preset_step = float(AXIS_PRESETS[inst_key][2])
                        top_tick = float(math.ceil(data_max / preset_step) * preset_step)
                        if top_tick <= float(current_ticks[-1]):
                            top_tick = float(current_ticks[-1]) + abs(preset_step)
                        # build a new tick array starting from the first current tick
                        start_tick = float(current_ticks[0])
                        new_ticks = np.arange(start_tick, top_tick + preset_step * 0.1, preset_step)
                        plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                        ax = plt.gca()
                        ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=100, preserve_ticks=True)
                    else:
                        # determine step (fallback to 1 if ambiguous)
                        if len(current_ticks) >= 2:
                            step_local = float(current_ticks[-1] - current_ticks[-2])
                        else:
                            step_local = max(1.0, math.ceil(data_max) - float(current_ticks[-1]))
                        top_tick = float(math.ceil(data_max / step_local) * step_local)
                        if top_tick <= current_ticks[-1]:
                            top_tick = float(current_ticks[-1]) + abs(step_local)
                        new_ticks = np.concatenate([current_ticks, np.array([top_tick])])
                        plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                        ax = plt.gca()
                        ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=8)
                except Exception:
                    # fallback to original behavior
                    if len(current_ticks) >= 2:
                        step_local = float(current_ticks[-1] - current_ticks[-2])
                    else:
                        step_local = max(1.0, math.ceil(data_max) - float(current_ticks[-1]))
                    top_tick = float(math.ceil(data_max / step_local) * step_local)
                    if top_tick <= current_ticks[-1]:
                        top_tick = float(current_ticks[-1]) + abs(step_local)
                    new_ticks = np.concatenate([current_ticks, np.array([top_tick])])
                    plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                    ax = plt.gca()
                    ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=8)
    except Exception:
        pass
    save_path = out_dir / f"allSensors_{csv_path.stem}.png"
    # Print axis diagnostics for requested instruments so we can verify presets
    try:
        # Print diagnostics for the instruments where we commonly use presets
        if instrument.lower() in ("rh_test", "humidity", "pressure", "temperature", "temperature_test"):
            ax = plt.gca()
            print(f"AXIS DIAGNOSTIC ({instrument}): ylim={ax.get_ylim()} ticks={ax.get_yticks()}")
    except Exception:
        pass
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
            instrument,
            out_dir,
            sensor_colors,
            y_min_rounded if y_min_rounded is not None else y_min,
            y_max_rounded if y_max_rounded is not None else y_max,
            y_ticks if y_ticks is not None else np.linspace(y_min, y_max, n_y_ticks),
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
            ax1.set_title(f"{instrument} - Pressure with Temperature Overlay ({csv_path.stem})")
            ax1.grid(True, linestyle="--", alpha=0.5)
            all_temps = np.concatenate([pd.to_numeric(df[s], errors="coerce").dropna().values for s in temp_sensors])
            tmin, tmax = np.nanmin(all_temps), np.nanmax(all_temps)
            ax2 = ax1.twinx()
            for sensor in temp_sensors:
                y_numeric = pd.to_numeric(df[sensor], errors="coerce")
                ax2.plot(df["time"], y_numeric, label=f"TEMP: {sensor}", linestyle="--", alpha=0.7, zorder=2)
            ax2.set_ylabel("Temperature")
            # Round temperature axis to nice multiples (use TEMP_TICK_STEP)
            try:
                step = TEMP_TICK_STEP
                tmin_rounded = int(math.floor(float(tmin) / step) * step)
                tmax_rounded = int(math.ceil(float(tmax) / step) * step)
                if tmax_rounded <= tmin_rounded:
                    tmax_rounded = tmin_rounded + step
                t_ticks = np.arange(tmin_rounded, tmax_rounded + step, step)
                max_ticks = 8
                if len(t_ticks) > max_ticks:
                    # If a temperature preset is in use with the configured TEMP_TICK_STEP,
                    # preserve the exact tick positions (do not subsample). Otherwise,
                    # pick up to `max_ticks` tick positions while guaranteeing the final
                    # (highest) tick is preserved after subsampling.
                    try:
                        if not (USE_AXIS_PRESETS and instrument is not None and instrument.lower() in AXIS_PRESETS and int(AXIS_PRESETS[instrument.lower()][2]) == int(TEMP_TICK_STEP)):
                            n = len(t_ticks)
                            idxs = np.unique(np.round(np.linspace(0, n - 1, max_ticks)).astype(int))
                            t_ticks = t_ticks[idxs]
                        # else: preserve t_ticks as-is for temperature presets
                    except Exception:
                        n = len(t_ticks)
                        idxs = np.unique(np.round(np.linspace(0, n - 1, max_ticks)).astype(int))
                        t_ticks = t_ticks[idxs]
                # Ensure the highest tick is at or above the actual data maximum.
                try:
                    data_max = float(np.nanmax(all_temps))
                    # If the last tick is below the data max, round the top tick UP
                    # to the next multiple of the step and append it if needed.
                    if float(t_ticks[-1]) < data_max - 1e-9:
                        # compute a rounded-up top tick (multiple of step)
                        top_tick = int(math.ceil(data_max / step) * step)
                        if top_tick <= int(t_ticks[-1]):
                            top_tick = int(t_ticks[-1]) + int(step)
                        # append if not already present
                        if top_tick not in t_ticks:
                            t_ticks = np.concatenate([t_ticks, np.array([top_tick])])
                            # also adjust the rounded axis endpoints
                            tmax_rounded = max(tmax_rounded, float(top_tick))
                except Exception:
                    pass

                ax2.set_ylim(tmin_rounded, tmax_rounded)
                # Force the exact tick locations via FixedLocator so they survive
                # any backend/layout adjustments at save time.
                try:
                    ax2.yaxis.set_major_locator(FixedLocator(t_ticks))
                    try:
                        ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                    except Exception:
                        pass
                except Exception:
                    try:
                        ax2.set_yticks(t_ticks)
                        try:
                            ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                ax2.set_ylim(tmin, tmax)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            all_lines = lines1 + lines2
            all_labels = labels1 + labels2
            n_items = len(all_lines)
            ncol = 1 if n_items <= 10 else 2
            fontsize = 10 if n_items <= 10 else 8
            place_legend_smart(ax2, fig=fig, handles=all_lines, labels=all_labels, ncol=ncol, fontsize=fontsize, handlelength=2)
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
            color, color_idx = pick_nonred_color(color_cycle, color_idx)
            group_colors[sensor] = color
        for sensor in sensors_in_group:
            y_numeric = pd.to_numeric(df[sensor], errors="coerce")
            y_plot = y_numeric
            plt.plot(time_vals, y_plot, label=sensor, alpha=0.8, zorder=1, color=group_colors[sensor])

        plt.xlabel("Time")
        units = get_unit_for_instrument(instrument)
        ylabel = f"{instrument} ({group}) ({units})" if units else f"{instrument} ({group})"
        plt.ylabel(ylabel)
        plt.title(f"{instrument} - {group} SENSORS ({csv_path.stem})")
        # Use rounded limits/ticks if computed earlier
        if y_min_rounded is not None and y_max_rounded is not None:
            plt.ylim(y_min_rounded, y_max_rounded)
            ax = plt.gca()
            ensure_legible_yticks(ax, y_ticks, prefer_integer=True, max_ticks=8)
        else:
            plt.ylim(y_min, y_max)
            y_ticks = np.linspace(y_min, y_max, n_y_ticks)
            ax = plt.gca()
            ensure_legible_yticks(ax, y_ticks, prefer_integer=True, max_ticks=8)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
        plt.xlim(time_vals.min(), time_vals.max())
        plt.xticks(rotation=45, ha='right')
        add_edge_labels_if_needed(ax, time_vals)
        place_legend_smart(plt.gca(), fig=plt.gcf(), ncol=1, fontsize=10, handlelength=2)
        plt.grid(True, linestyle="--", alpha=0.5)
        # Safety: ensure the highest tick covers plotted lines for this group
        try:
            vals = []
            for s in sensors_in_group:
                arr = pd.to_numeric(df[s], errors='coerce').dropna().values
                if arr.size:
                    vals.append(arr)
            if vals:
                data_max = float(np.nanmax(np.concatenate(vals)))
                current_ticks = np.array(y_ticks) if y_ticks is not None else np.array(plt.gca().get_yticks())
                if current_ticks.size > 0 and float(current_ticks[-1]) < data_max - 1e-9:
                    if len(current_ticks) >= 2:
                        step_local = float(current_ticks[-1] - current_ticks[-2])
                    else:
                        step_local = max(1.0, math.ceil(data_max) - float(current_ticks[-1]))
                    top_tick = float(math.ceil(data_max / step_local) * step_local)
                    if top_tick <= current_ticks[-1]:
                        top_tick = float(current_ticks[-1]) + abs(step_local)
                    new_ticks = np.concatenate([current_ticks, np.array([top_tick])])
                    plt.ylim(float(new_ticks[0]), float(new_ticks[-1]))
                    ax = plt.gca()
                    ensure_legible_yticks(ax, new_ticks, prefer_integer=True, max_ticks=8)
        except Exception:
            pass
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
        # Print a human-friendly rounded folder y-axis summary (integers or 5-degree steps for temperature)
        try:
            inst_name = folder.name.lower()
            if "temp" in inst_name or "temperature" in inst_name:
                step = TEMP_TICK_STEP
                pretty_min = int(math.floor(y_min / step) * step)
                pretty_max = int(math.ceil(y_max / step) * step)
            else:
                pretty_min = int(round(y_min))
                pretty_max = int(round(y_max))
            print(f"\n📁 Folder={folder} | y-axis: {pretty_min} to {pretty_max}")
        except Exception:
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
