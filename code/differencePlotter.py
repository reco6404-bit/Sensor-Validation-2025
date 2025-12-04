# Global spike threshold for difference jumps
spike_threshold = 0.5  # Change this value to control spike detection everywhere
# Global window size for masking around humidity label indices
masking_window_size = 3000  # Change this value to control the number of points masked before/after each label
# Hysteresis detection defaults (absolute difference in %RH and minimum duration in seconds)
HYSTERESIS_THRESHOLD = 1.0  # percent points of RH considered hysteresis
HYSTERESIS_MIN_DURATION_SECONDS = 10  # minimum duration to report a hysteresis period
"""differencePlotter.py

Produce difference plots (sensor - reference) from formatted CSV files.

What this script does:
 - For each CSV it finds a reference sensor (configured via `REFERENCE_SENSORS`) and
     computes sensor - reference differences for all non-reference sensors.
 - Special-case handling for humidity: detect and mask/interpolate spikes so
     the resulting plots show stable plateaus. Hysteresis detection is available
     and the script also exports per-sensor precision CSVs.
 - Axis presets in `AXIS_PRESETS` may be used to force consistent y-limits
     and tick steps for specific instruments (e.g., RH or temperature plots).

Usage
    Run from the project root:
        python3 code/differencePlotter.py

Outputs
    PNGs are written to `differencePlots/<Instrument>/` and a per-CSV
    sensor precisions CSV is exported alongside the images.

Implementation notes for new readers
 - This module is defensive: it avoids in-place interpolation of the input
     DataFrame and instead applies boolean masks at plotting time so raw data
     are preserved for reproducibility.
 - Many helpers (spike detection, tick calculation, legend placement) are
     intentionally small and documented near their definitions.
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
import math

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
END_EDGE_CUT_PERCENT = 0.12    # percent to cut from the ending edge (for x-axis bounds only)

# ---------- AXIS PRESETS (manual override) ---------- #
# Format: {"instrument_name": (ymin, ymax, step)}
AXIS_PRESETS = {
    # Example manual overrides (lowercase instrument keys):
    "rh_test": (-8, 10, 2.0),
    "pressure": (-6.5, 1.5, 0.5),
    "temperature": (-0.5, 0.5, 0.1),
}


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

# Temperature tick step (degrees) used when rounding temperature axes
TEMP_TICK_STEP = 5


def find_axis_preset_for_instrument(instrument: str):
    """Return a preset tuple (min, max, step) for `instrument` using flexible matching.
    Matching tries exact lowercase key, then substring/prefix matches against keys in AXIS_PRESETS.
    Returns None if no preset found.
    """
    if not instrument:
        return None
    inst = instrument.lower()
    # exact match
    if inst in AXIS_PRESETS:
        return AXIS_PRESETS[inst]
    # try substring/prefix matches (e.g., 'temperature_test' should match 'temperature')
    for k, v in AXIS_PRESETS.items():
        try:
            if k in inst or inst.startswith(k) or inst.endswith(k):
                return v
        except Exception:
            continue
    return None

def get_nice_axis_limits_and_ticks(data, n_ticks=6, instrument=None, force_no_preset=False):
    """
    Given a 1D array, return (ymin, ymax, yticks) with nice rounded limits and regular steps.
    Axis will start at a multiple of 0.5 if possible.
    If AXIS_PRESETS is set for the instrument, use those values unless force_no_preset is True.
    """
    if not force_no_preset and instrument is not None:
        preset = find_axis_preset_for_instrument(instrument)
        if preset is not None:
            ymin, ymax, step = preset
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
        # round extremes to multiples of 'step' but ensure we ROUND UP the top
        y_min = math.floor(dmin / step) * step
        y_max = math.ceil(dmax / step) * step
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
        chosen_ymin = math.floor(dmin / step) * step
        chosen_ymax = math.ceil(dmax / step) * step
        if chosen_ymax <= chosen_ymin:
            chosen_ymax = chosen_ymin + step

    # Build tick array
    # Build tick array and ensure top tick is at or above the true data max
    yticks = np.arange(chosen_ymin, chosen_ymax + chosen_step * 0.5, chosen_step)

    # Special-case: prefer whole-number ticks for humidity/RH instruments
    if instrument is not None:
        inst = instrument.lower()
        try:
            # humidity instrument -> integer % RH ticks
            if "humidity" in inst or inst.startswith("rh") or "rh_test" in inst:
                # Use integer %RH ticks and ensure we round the top tick UP so it
                # is never below the actual maximum value in the data.
                ymin_i = int(math.floor(dmin))
                ymax_i = int(math.ceil(dmax))
                if ymin_i == ymax_i:
                    ymin_i -= 1
                    ymax_i += 1
                # Ensure the returned ymax covers the actual max (round up)
                if float(ymax_i) < float(dmax) - 1e-9:
                    ymax_i = int(math.ceil(float(dmax)))
                yticks = np.arange(ymin_i, ymax_i + 1, 1)
                return float(ymin_i), float(ymax_i), yticks.astype(int)
            # temperature instrument -> use TEMP_TICK_STEP multiples
            if "temp" in inst or inst.endswith("_t") or "temperature" in inst:
                # Use fixed TEMP_TICK_STEP multiples and ROUND UP the top tick so
                # the highest tick is >= the actual maximum value.
                step = TEMP_TICK_STEP
                ymin_t = int(math.floor(dmin / step) * step)
                # Round up using math.ceil to ensure top tick >= dmax
                ymax_t = int(math.ceil(float(dmax) / float(step)) * step)
                if ymax_t <= ymin_t:
                    ymax_t = ymin_t + step
                yticks = np.arange(ymin_t, ymax_t + step, step)
                # limit number of ticks similar to general logic
                max_ticks = 8
                if len(yticks) > max_ticks:
                    s = int(np.ceil(len(yticks) / float(max_ticks)))
                    yticks = yticks[::s]
                return float(ymin_t), float(ymax_t), yticks.astype(int)
        except Exception:
            # on any error, fall back to the generic ticks below
            pass

    # Ensure the last tick is at or above the actual data max (avoid labels lower than plotted lines)
    try:
        if len(yticks) == 0:
            yticks = np.array([chosen_ymin, chosen_ymax])
        # If due to subsampling the last tick falls below the data max, append the rounded max
        if float(yticks[-1]) < float(dmax) - 1e-9:
            # determine step to append: prefer chosen_step when available
            step = chosen_step if 'chosen_step' in locals() and chosen_step is not None else (yticks[-1] - yticks[-2] if len(yticks) >= 2 else (float(chosen_ymax) - float(chosen_ymin)))
            append_tick = float(yticks[-1]) + float(step)
            # round append_tick to a sensible value
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


def align_time_and_series(time_vals, *series_list):
    """
    Trim/align time_vals and provided series so they all have the same length.
    Returns (time_trimmed, [series_trimmed...]). Works with pandas Series or numpy arrays.
    """
    # Convert time_vals to pandas Series if possible
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
            # numpy-like
            trimmed.append(pd.Series(s).iloc[:min_len])
    return tv_trim, trimmed


def safe_plot(ax, time_vals, y_vals, csv_path=None, sensor_name=None, **plot_kwargs):
    """
    Plot time vs y on ax after aligning lengths. If lengths cannot be aligned
    or remain mismatched, log a clear diagnostic including csv_path and
    sensor_name and skip plotting that series.
    """
    try:
        time_t, [y_t] = align_time_and_series(time_vals, y_vals)
    except Exception:
        # Best-effort conversion
        try:
            time_t = pd.Series(time_vals).reset_index(drop=True)
        except Exception:
            time_t = pd.Series(list(time_vals))
        try:
            y_t = pd.Series(y_vals).reset_index(drop=True)
        except Exception:
            y_t = pd.Series(list(y_vals))

    if len(time_t) != len(y_t):
        label = plot_kwargs.get('label') or sensor_name or '<series>'
        print(f"⚠️ Skipping plot for {label} in {csv_path}: time len={len(time_t)} y len={len(y_t)}")
        return

    # Plot and capture the Line2D object so we can report the actual color used
    try:
        lines = ax.plot(time_t, y_t, **plot_kwargs)
        # matplotlib returns a list of Line2D objects; report the first one's color
        # Do not print per-series plotted colors here to reduce console noise.
        # The function still returns the Line2D objects implicitly via matplotlib.
    except Exception as e:
        # If plotting fails for any reason, log and continue
        label = plot_kwargs.get('label') or sensor_name or '<series>'
        print(f"⚠️ Failed to plot {label} in {csv_path}: {e}")

# Helper to place legends outside the axes without overlapping the plot area.
def place_legends_outside(fig, ax_left, lines_left, labels_left, ax_right=None, lines_right=None, labels_right=None, left_anchor=-0.08, right_anchor=1.08, pad=0.02, min_plot_width=0.30):
    """
    Reserve symmetric margins and place left/right legends outside the plot.
    Returns (legend_left, legend_right).
    """
    legend_left = None
    legend_right = None
    try:
        if lines_left:
            legend_left = ax_left.legend(lines_left, labels_left, loc="center right", bbox_to_anchor=(left_anchor, 0.5), frameon=True, fontsize=10, handlelength=2)
        if ax_right is not None and lines_right:
            legend_right = ax_right.legend(lines_right, labels_right, loc="center left", bbox_to_anchor=(right_anchor, 0.5), frameon=True, fontsize=10, handlelength=2)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bbox_left = legend_left.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()) if legend_left is not None else None
        bbox_right = legend_right.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()) if legend_right is not None else None

        if bbox_left is None and bbox_right is None:
            return legend_left, legend_right

        w_left = bbox_left.width if bbox_left is not None else 0.0
        w_right = bbox_right.width if bbox_right is not None else 0.0
        max_legend_width = max(w_left, w_right)
        compact_pad = pad * 0.6
        desired_margin = max_legend_width + compact_pad

        min_left = 0.02
        left_needed = max(min_left, desired_margin)
        right_needed = min(0.98, 1.0 - desired_margin)

        if right_needed - left_needed < min_plot_width:
            center = 0.5
            left_needed = max(min_left, center - min_plot_width / 2)
            right_needed = min(0.98, center + min_plot_width / 2)

        try:
            fig.subplots_adjust(left=left_needed, right=right_needed)
        except Exception:
            pass

        if legend_left is not None:
            legend_left.remove()
            legend_left = ax_left.legend(lines_left, labels_left, loc="center right", bbox_to_anchor=(left_anchor, 0.5), frameon=True, fontsize=10, handlelength=2)
        if legend_right is not None:
            legend_right.remove()
            legend_right = ax_right.legend(lines_right, labels_right, loc="center left", bbox_to_anchor=(right_anchor, 0.5), frameon=True, fontsize=10, handlelength=2)
    except Exception:
        # fallback conservative margins
        try:
            if lines_left and lines_right:
                fig.subplots_adjust(left=0.18, right=0.82)
                if legend_left is not None:
                    legend_left.remove()
                legend_left = ax_left.legend(lines_left, labels_left, loc="center right", bbox_to_anchor=(-0.22, 0.5), frameon=True, fontsize=10, handlelength=2)
                if legend_right is not None:
                    legend_right.remove()
                legend_right = ax_right.legend(lines_right, labels_right, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, handlelength=2)
            else:
                if lines_left:
                    fig.subplots_adjust(left=0.22)
                    if legend_left is not None:
                        legend_left.remove()
                    legend_left = ax_left.legend(lines_left, labels_left, loc="center right", bbox_to_anchor=(-0.22, 0.5), frameon=True, fontsize=10, handlelength=2)
                if lines_right:
                    fig.subplots_adjust(right=0.78)
                    if legend_right is not None:
                        legend_right.remove()
                    legend_right = ax_right.legend(lines_right, labels_right, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, handlelength=2)
        except Exception:
            pass
    return legend_left, legend_right

# --- Spike removal helper: replace momentary large deviations with interpolated values ---
def detect_spikes(series, z_thresh=1.0, window=20):
    """
    Detect spike periods in a series based on deviation from rolling median.
    Returns list of (start_idx, end_idx) tuples for spike periods.
    """
    rolling_med = series.rolling(window=window, center=True).median()
    dev = (series - rolling_med).abs()
    spike_mask = dev > z_thresh

    from itertools import groupby
    spike_periods = []
    for k, g in groupby(enumerate(spike_mask), lambda x: x[1]):
        if k:  # spike
            group = list(g)
            start = group[0][0]
            end = group[-1][0]
            if end - start >= 0:  # at least 1 point
                spike_periods.append((start, end))
    return spike_periods


def _format_duration(start_ts, end_ts):
    try:
        duration = end_ts - start_ts
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    except Exception:
        return "unknown"


def print_spike_periods(spike_periods, time_index, sensor_name=None, csv_path=None):
    """
    Print human-readable spike periods with timestamps and durations.
    - spike_periods: list of (start_idx, end_idx)
    - time_index: pandas Series or array-like of timestamps (must indexable by integer positions)
    """
    # Intentionally silent by default to avoid noisy per-spike output during
    # bulk processing. If you want visible spike logging, replace this body
    # with a loop that prints timestamps/durations for each (start, end).
    return


def expand_spike_periods(series, spike_periods, z_thresh=1.0, window=20, max_expand=5000, time_index=None, merge_gap_seconds=10):
    """
    Expand each detected spike period outward until the series returns to baseline.
    Baseline is defined where abs(series - rolling_median) <= z_thresh.
    Returns a new list of (start_idx, end_idx) with expanded intervals (merged if overlapping).
    """
    if not spike_periods:
        return []
    try:
        s = pd.Series(series).reset_index(drop=True)
    except Exception:
        s = pd.Series(list(series))
    rolling_med = s.rolling(window=window, center=True, min_periods=1).median()
    dev = (s - rolling_med).abs()
    n = len(s)
    expanded = []
    for start, end in spike_periods:
        # expand backward
        b = start
        steps = 0
        while b > 0 and steps < max_expand:
            if not np.isfinite(dev.iloc[b - 1]) or dev.iloc[b - 1] > z_thresh:
                b -= 1
                steps += 1
                continue
            break
        # expand forward
        e = end
        steps = 0
        while e < n - 1 and steps < max_expand:
            if not np.isfinite(dev.iloc[e + 1]) or dev.iloc[e + 1] > z_thresh:
                e += 1
                steps += 1
                continue
            break
        expanded.append((max(0, b), min(n - 1, e)))
    # merge overlapping/adjacent intervals
    if not expanded:
        return []
    expanded_sorted = sorted(expanded, key=lambda x: x[0])
    merged = [expanded_sorted[0]]
    # Prepare time index if provided
    time_series = None
    if time_index is not None:
        try:
            time_series = pd.to_datetime(pd.Series(time_index).reset_index(drop=True))
            if len(time_series) != n:
                time_series = None
        except Exception:
            time_series = None

    for cur in expanded_sorted[1:]:
        last = merged[-1]
        # merge if overlapping or adjacent by index
        if cur[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], cur[1]))
            continue
        # if time series provided, merge if gap between intervals is small (seconds)
        if time_series is not None:
            try:
                last_end_time = time_series.iloc[last[1]]
                cur_start_time = time_series.iloc[cur[0]]
                gap = (cur_start_time - last_end_time).total_seconds()
                if gap <= merge_gap_seconds:
                    # merge across small time gap
                    merged[-1] = (last[0], max(last[1], cur[1]))
                    continue
            except Exception:
                pass
        # otherwise keep as separate interval
        merged.append(cur)
    return merged


def merge_time_intervals(intervals, time_index, merge_gap_seconds=1):
    """
    Merge a list of intervals given in index-space into timestamp intervals.
    intervals: list of (start_idx, end_idx, source)
    time_index: pandas Series/array-like of datetimes indexed by integer positions
    Returns list of (start_ts, end_ts, duration_timedelta, set_of_sources)
    """
    if not intervals:
        return []
    # Convert to timestamp intervals
    try:
        times = pd.to_datetime(pd.Series(time_index).reset_index(drop=True))
    except Exception:
        times = None
    ts_intervals = []
    for s, e, src in intervals:
        try:
            start_ts = times.iloc[s] if times is not None and s < len(times) else None
        except Exception:
            start_ts = None
        try:
            end_ts = times.iloc[e] if times is not None and e < len(times) else None
        except Exception:
            end_ts = None
        ts_intervals.append((start_ts, end_ts, src))

    # Sort by start timestamp (None values go last)
    ts_intervals = sorted([t for t in ts_intervals if t[0] is not None], key=lambda x: x[0])
    if not ts_intervals:
        return []
    merged = []
    cur_start, cur_end, cur_sources = ts_intervals[0][0], ts_intervals[0][1], {ts_intervals[0][2]}
    for st, en, src in ts_intervals[1:]:
        # If overlapping or within merge gap, merge
        gap = (st - cur_end).total_seconds() if cur_end is not None and st is not None else None
        if gap is not None and gap <= merge_gap_seconds:
            # extend end if needed
            if en is not None and (cur_end is None or en > cur_end):
                cur_end = en
            cur_sources.add(src)
        elif st <= cur_end:
            # overlapping
            if en is not None and en > cur_end:
                cur_end = en
            cur_sources.add(src)
        else:
            duration = cur_end - cur_start if cur_start is not None and cur_end is not None else None
            merged.append((cur_start, cur_end, duration, cur_sources.copy()))
            cur_start, cur_end, cur_sources = st, en, {src}
    duration = cur_end - cur_start if cur_start is not None and cur_end is not None else None
    merged.append((cur_start, cur_end, duration, cur_sources.copy()))
    return merged


def detect_hysteresis_periods(diff_df, time_index, threshold=HYSTERESIS_THRESHOLD, min_duration_seconds=HYSTERESIS_MIN_DURATION_SECONDS):
    """
    Detect hysteresis periods per sensor where the absolute difference from reference
    exceeds `threshold` for at least `min_duration_seconds`.

    Returns dict: {sensor_name: [(start_ts, end_ts, duration_timedelta, mean_diff, peak_diff), ...], ...}
    """
    results = {}
    try:
        times = pd.to_datetime(pd.Series(time_index).reset_index(drop=True))
    except Exception:
        times = None

    for col in diff_df.columns:
        try:
            series = pd.to_numeric(diff_df[col], errors="coerce")
        except Exception:
            series = pd.Series(diff_df[col])
        mask = series.abs() > float(threshold)
        periods = []
        if mask.any():
            from itertools import groupby
            for k, g in groupby(enumerate(mask.tolist()), lambda x: x[1]):
                if k:
                    group = list(g)
                    start = group[0][0]
                    end = group[-1][0]
                    # convert to timestamps if available
                    try:
                        start_ts = times.iloc[start] if times is not None and start < len(times) else None
                        end_ts = times.iloc[end] if times is not None and end < len(times) else None
                    except Exception:
                        start_ts = None
                        end_ts = None
                    # compute duration in seconds if timestamps available
                    duration = None
                    if start_ts is not None and end_ts is not None:
                        duration = end_ts - start_ts
                    else:
                        # fall back to sample count / unknown duration
                        duration = None
                    # filter by duration if possible, otherwise by sample count
                    ok = True
                    if duration is not None:
                        if duration.total_seconds() < min_duration_seconds:
                            ok = False
                    else:
                        # approximate: require at least 2 samples if time not available
                        if (end - start + 1) < 2:
                            ok = False
                    if not ok:
                        continue
                    # compute mean and peak diffs over the interval
                    try:
                        seg = series.iloc[start:end+1].dropna()
                        mean_diff = float(seg.abs().mean()) if not seg.empty else float('nan')
                        peak_diff = float(seg.abs().max()) if not seg.empty else float('nan')
                    except Exception:
                        mean_diff = float('nan')
                        peak_diff = float('nan')
                    periods.append((start_ts, end_ts, duration, mean_diff, peak_diff))
        if periods:
            results[col] = periods
    return results

def remove_spikes(series, window=20, z_thresh=1.0):
    """
    Remove spikes by detecting spike periods and interpolating over them.
    """
    spike_periods = detect_spikes(series, z_thresh=z_thresh, window=window)
    series_clean = series.copy()
    for start, end in spike_periods:
        series_clean.iloc[start:end+1] = np.nan
    series_clean = series_clean.interpolate(method='linear')
    return series_clean


def build_spike_mask(series, window=20, z_thresh=1.0):
    """
    Return a boolean mask (pandas Series) where True indicates the sample is part of a detected spike period.
    This does NOT interpolate or remove values; it only marks them so plotting code can display gaps.
    """
    try:
        s = pd.Series(series).reset_index(drop=True)
    except Exception:
        s = pd.Series(list(series))
    mask = np.zeros(len(s), dtype=bool)
    try:
        periods = detect_spikes(s, z_thresh=z_thresh, window=window)
        for start, end in periods:
            if start < 0:
                start = 0
            if end >= len(s):
                end = len(s) - 1
            mask[start:end + 1] = True
    except Exception:
        # On failure, return all-False mask
        pass
    return pd.Series(mask)

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
    # --- Apply per-instrument axis preset override early ---
    # If a preset exists for this instrument in AXIS_PRESETS, set initial
    # y_min/y_max/ticks here so later plotting branches inherit the same
    # bounds (mirrors behavior in graphMerger.py).
    y_min_rounded = None
    y_max_rounded = None
    y_ticks = None
    try:
        preset = find_axis_preset_for_instrument(instrument)
        if preset is not None:
            preset_min, preset_max, preset_step = preset
            y_min_rounded = float(preset_min)
            y_max_rounded = float(preset_max)
            y_ticks = np.arange(preset_min, preset_max + preset_step * 0.5, preset_step)
    except Exception:
        y_min_rounded = None
        y_max_rounded = None
        y_ticks = None
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

    # NOTE: spike masking will be computed below (after canonicalization and sensor filtering)
    # to avoid overwriting the original DataFrame with interpolated values. We want to
    # keep raw series in `df` and apply boolean masks when plotting so masked regions
    # appear as visible gaps across all humidity plots.

    # Ensure spike detection is applied to all difference calculations
    non_reference_sensors = [s for s in valid_sensors if s not in references]
    filtered_non_refs = [s for s in non_reference_sensors if df[s].count() / len(df) >= min_data_fraction]
    valid_sensors = filtered_non_refs + [s for s in references if s in valid_sensors]
    references = [s for s in references if s in valid_sensors]
    if not valid_sensors or not references:
        print(f"⚠️ No valid sensors or references with sufficient data in {csv_path}, skipping plot")
        return

    out_dir = export_root / instrument
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure dataframe index, time, and reference series are canonical and trimmed to same length
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
    # Colors that should be reserved for references only (names or common hexes)
    RESERVED_RED_NAMES = {"red", "tab:red", "crimson", "firebrick", "maroon", "darkred"}
    RESERVED_RED_HEX = {"#d62728", "#ff0000", "#8b0000"}

    def pick_nonred_color(cycle, start_idx):
        """Return (color, next_index) where color is the first non-reserved color
        found in cycle starting at start_idx. If all colors are reserved, return
        the original color at start_idx.
        """
        n = len(cycle)
        for offset in range(n):
            idx = (start_idx + offset) % n
            c = cycle[idx]
            try:
                c_lower = c.lower()
            except Exception:
                c_lower = str(c)
            if c_lower in RESERVED_RED_NAMES or c in RESERVED_RED_HEX:
                continue
            return c, idx + 1
        # fallback: nothing safe found, return the color at start_idx
        return cycle[start_idx % n], start_idx + 1
    sensor_colors = {}
    all_diffs = []
    sensor_precisions = {}
    # Collect spike/clip intervals from across processing so we can report the
    # consolidated cut periods later (passed into plot_reference_overlay).
    collected_cut_intervals = []  # list of (start_idx, end_idx, source)

    # Connect points across gaps for temperature instruments and for humidity plots
    # where we want interpolated connectors over removed spike regions.
    do_connect = True if ("temperature" in instrument.lower() or instrument.lower() in ("humidity", "rh_test")) else False
    for sensor in valid_sensors:
        if sensor == ref:
            continue
        y_sensor = pd.to_numeric(df[sensor], errors="coerce")
        diff = y_sensor - ref_vals
        # For humidity instruments: remove spikes by interpolation (original behavior)
        if instrument.lower() in ("humidity", "rh_test"):
            try:
                # detect individual spike periods and print durations
                spike_periods = detect_spikes(diff, z_thresh=spike_threshold, window=20)
                # Expand each detected spike until the series returns to baseline
                expanded_periods = expand_spike_periods(diff, spike_periods, z_thresh=spike_threshold, window=20, time_index=df["time"], merge_gap_seconds=10)
                # record expanded periods so they can be reported as CUT ranges later
                for s_idx, e_idx in expanded_periods:
                    collected_cut_intervals.append((s_idx, e_idx, sensor))
                print_spike_periods(expanded_periods, df["time"], sensor_name=sensor, csv_path=csv_path)
                diff = remove_spikes(diff, window=20, z_thresh=spike_threshold)
            except Exception:
                pass
        # Simple moving average smoothing (keeps NaNs so gaps are visible)
        diff = diff.rolling(window=20, center=True, min_periods=1).mean()
        # Compute a per-sensor precision measure: mean absolute difference from reference
        try:
            mean_abs = float(np.nanmean(np.abs(diff.values))) if diff is not None else float('nan')
        except Exception:
            mean_abs = float('nan')
        sensor_precisions[sensor] = mean_abs
        all_diffs.append(diff)
        # Pick a color but avoid reserved red shades for non-reference sensors
        chosen_color, color_idx = pick_nonred_color(color_cycle, color_idx)
        sensor_colors[sensor] = chosen_color

    # Export per-sensor precision values to CSV in the same output folder
    try:
        # Map common matplotlib cycle hex codes to friendly color names for CSV export.
        # These hex codes correspond to matplotlib's default color cycle and are
        # converted to readable names per user request.
        COLOR_NAME_MAP = {
            '#1f77b4': 'blue',
            '#ff7f0e': 'orange',
            '#2ca02c': 'green',
            '#d62728': 'red',
            '#9467bd': 'purple',
            '#8c564b': 'brown',
            '#e377c2': 'pink',
            '#7f7f7f': 'gray',
            '#bcbd22': 'olive',
            '#17becf': 'cyan'
        }

        def color_to_name(c):
            if c is None:
                return ''
            try:
                cs = str(c).lower()
            except Exception:
                return str(c)
            # If it's already a human-readable name (not a hex), return as-is
            if not cs.startswith('#'):
                return cs
            # Normalize and map known hex to names
            if cs in COLOR_NAME_MAP:
                return COLOR_NAME_MAP[cs]
            # Fallback: return original hex if unknown
            return cs

        prec_rows = []
        for s, p in sensor_precisions.items():
            color_value = sensor_colors.get(s)
            prec_rows.append({
                "sensor": s,
                "precision": float(p) if p is not None else float('nan'),
                "color": color_to_name(color_value),
            })
        prec_df = pd.DataFrame(prec_rows)
        prec_path = out_dir / f"{csv_path.stem}_sensor_precisions.csv"
        prec_df.to_csv(prec_path, index=False)
        print(f"✅ Exported precisions: {prec_path}")
    except Exception as e:
        print(f"⚠️ Failed to export precisions CSV for {csv_path}: {e}")

    # --- Use improved axis logic for y-limits and ticks (only if not preset) ---
    # If an axis preset was applied earlier (y_min_rounded is not None) do
    # not overwrite it; otherwise compute data-driven limits. For temperature
    # difference plots we choose a small symmetric range around zero where
    # appropriate (e.g. -1..1 with 0.2 ticks) so small differences are easy
    # to inspect.
    if y_min_rounded is None:
        if all_diffs:
            diffs_concat = np.concatenate([d[np.isfinite(d)] for d in all_diffs if np.any(np.isfinite(d))])
            # Compute raw min/max for custom per-instrument logic
            try:
                dmin = float(np.nanmin(diffs_concat))
                dmax = float(np.nanmax(diffs_concat))
            except Exception:
                dmin, dmax = -1.0, 1.0

            # Temperature-specific presets: prefer a small symmetric window
            # around zero when measured diffs are small. This yields examples
            # like -1..1 with 0.2 ticks when the data support that granularity.
            try:
                if "temperature" in instrument.lower():
                    max_abs = max(abs(dmin), abs(dmax))
                    if max_abs <= 1.0:
                        bound = 1.0
                        step = 0.2
                    elif max_abs <= 2.0:
                        bound = 2.0
                        step = 0.5
                    else:
                        # For larger spreads choose a sensible step from candidates
                        candidates = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
                        raw_step = (2.0 * max_abs) / float(max(4, n_y_ticks))
                        step = next((c for c in candidates if c >= raw_step), candidates[-1])
                        bound = math.ceil(max_abs / step) * step
                    y_min_rounded = -float(bound)
                    y_max_rounded = float(bound)
                    y_ticks = np.arange(y_min_rounded, y_max_rounded + step / 2.0, step)
                else:
                    y_min_rounded, y_max_rounded, y_ticks = get_nice_axis_limits_and_ticks(diffs_concat, n_ticks=n_y_ticks, instrument=instrument)
            except Exception:
                # fallback to generic helper on any failure
                y_min_rounded, y_max_rounded, y_ticks = get_nice_axis_limits_and_ticks(diffs_concat, n_ticks=n_y_ticks, instrument=instrument)
        else:
            y_min_rounded, y_max_rounded, y_ticks = -1, 1, np.arange(-1, 1.1, 0.5)
    # (plotting continues after trimmed_diffs is computed below)

    # Defensive alignment: trim time_vals and diffs to the same minimum length to avoid x/y dimension mismatch
    try:
        time_trim, trimmed_diffs = align_time_and_series(time_vals, *all_diffs)
    except Exception:
        time_trim = pd.Series(time_vals).reset_index(drop=True)
        trimmed_diffs = [d.reset_index(drop=True).iloc[:len(time_trim)] if hasattr(d, 'reset_index') else pd.Series(d).iloc[:len(time_trim)] for d in all_diffs]

    # Plot using the trimmed series
    diff_iter = iter(trimmed_diffs)
    ax_main = plt.gca()
    for sensor in valid_sensors:
        if sensor == ref:
            continue
        diff = next(diff_iter)
        precision = sensor_precisions.get(sensor, float('nan'))
        # Show mean absolute difference (precision) next to sensor in legend.
        # Draw an interpolated dotted line (labelled) to connect gaps, then
        # overplot the smoothed/actual series without a label so the legend
        # contains only one entry per sensor.
        # Use a short generic reference label in plots (do not embed the full column name)
        label = f"{sensor} - ref"
        try:
            diff_series = pd.Series(diff).reset_index(drop=True)
            diff_interp = diff_series.interpolate(method='linear', limit_direction='both')
        except Exception:
            diff_interp = diff
        if do_connect:
            # Interpolated dotted (unlabelled) for temperature/humidity instruments only
            safe_plot(ax_main, time_trim, diff_interp, csv_path=csv_path, sensor_name=sensor, label=None, linestyle=':', alpha=0.9, zorder=1, color=sensor_colors[sensor])
            # Original smoothed series as labeled solid line/markers (legend will show solid color)
            safe_plot(ax_main, time_trim, diff, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors[sensor])
        else:
            # For non-temperature instruments, plot only the smoothed series and label it
            safe_plot(ax_main, time_trim, diff, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors[sensor])

    try:
        plotted_fin = np.concatenate([np.asarray(d)[np.isfinite(d)] for d in trimmed_diffs if np.any(np.isfinite(d))])
    except Exception:
        plotted_fin = np.array([])
    if plotted_fin.size > 0:
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Time")
        units = get_unit_for_instrument(instrument)
        # Use generic 'ref' in axis labels to avoid long column names on the plot
        ylabel = f"Difference from ref ({units})" if units else f"Difference from ref"
        plt.ylabel(ylabel)
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
        fig = ax.get_figure()
        # Place legends outside symmetrically using the helper so they do not
        # overlap the plotting area (left-only in this plot; right legend unused)
        legend_left, legend_right = place_legends_outside(fig, ax, handles, labels, ax_right=None, lines_right=None, labels_right=None)
        plt.draw()

        plt.grid(True, linestyle="--", alpha=0.5)
        save_path = out_dir / f"{instrument}_allSensorsDiff_{csv_path.stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        # Print tick labels and plotted data min/max for diagnostics
        try:
            tick_min = y_ticks[0] if hasattr(y_ticks, '__len__') and len(y_ticks) > 0 else y_min_rounded
            tick_max = y_ticks[-1] if hasattr(y_ticks, '__len__') and len(y_ticks) > 0 else y_max_rounded
            data_min = float(np.nanmin(plotted_fin)) if plotted_fin.size > 0 else float('nan')
            data_max = float(np.nanmax(plotted_fin)) if plotted_fin.size > 0 else float('nan')
            print(f"TICKS (primary y): {tick_min} -> {tick_max} | DATA (primary y): {data_min:.2f} -> {data_max:.2f}")
        except Exception:
            pass
        plt.close()
    else:
        plt.close()
        print(f"⚠️ Skipping all-sensors plot for {csv_path.name}: no finite sensor differences to plot")

    # --- Temperature overlay difference plot ---
    temp_ref_col = find_reference_temp(df)
    if temp_ref_col is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # Defensive alignment for temp overlay
        temp_all = []
        for sensor in valid_sensors:
            if sensor == ref:
                continue
            y_sensor = pd.to_numeric(df[sensor], errors="coerce")
            diff = y_sensor - ref_vals
            if instrument.lower() in ("humidity", "rh_test"):
                try:
                    # detect spikes and report durations
                    spike_periods = detect_spikes(diff, z_thresh=spike_threshold, window=20)
                    expanded_periods = expand_spike_periods(diff, spike_periods, z_thresh=spike_threshold, window=20, time_index=df["time"], merge_gap_seconds=10)
                    for s_idx, e_idx in expanded_periods:
                        collected_cut_intervals.append((s_idx, e_idx, sensor))
                    print_spike_periods(expanded_periods, df["time"], sensor_name=sensor, csv_path=csv_path)
                    diff = remove_spikes(diff, window=20, z_thresh=spike_threshold)
                except Exception:
                    pass
            temp_all.append(diff)
        try:
            time_trim2, temp_trimmed = align_time_and_series(time_vals, *temp_all)
        except Exception:
            time_trim2 = pd.Series(time_vals).reset_index(drop=True)
            temp_trimmed = [d.reset_index(drop=True).iloc[:len(time_trim2)] if hasattr(d, 'reset_index') else pd.Series(d).iloc[:len(time_trim2)] for d in temp_all]
        for sensor, diff in zip([s for s in valid_sensors if s != ref], temp_trimmed):
            precision = sensor_precisions.get(sensor, float('nan'))
            label = f"{sensor} - ref"
            try:
                diff_series = pd.Series(diff).reset_index(drop=True)
                diff_interp = diff_series.interpolate(method='linear', limit_direction='both')
            except Exception:
                diff_interp = diff
            # Only connect points across gaps for temperature instrument plots
            if do_connect:
                # interpolated dotted unlabelled line (connector)
                safe_plot(ax1, time_trim2, diff_interp, csv_path=csv_path, sensor_name=sensor, label=None, linestyle=':', alpha=0.9, zorder=1, color=sensor_colors.get(sensor, None))
                # actual smoothed series as labeled solid line (legend will show solid color)
                safe_plot(ax1, time_trim2, diff, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors.get(sensor, None))
            else:
                # For non-temperature instruments, plot the smoothed series labeled
                safe_plot(ax1, time_trim2, diff, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors.get(sensor, None))
            if np.any(np.isfinite(np.asarray(diff))):
                has_temp_plot = True
        if not has_temp_plot:
            plt.close()
            print(f"⚠️ Skipping temperature-overlay for {csv_path.name}: no finite temperature differences to plot")
        else:
            ax1.axhline(0, color="black", linestyle="--", linewidth=1)
            ax1.set_xlabel("Time")
            units = get_unit_for_instrument(instrument)
            ylabel = f"Difference from ref ({units})" if units else f"Difference from ref"
            ax1.set_ylabel(ylabel)
            ax1.set_title(f"{instrument} - SENSOR DIFFERENCES + REF TEMP ({csv_path.stem})")
            ax1.set_ylim(y_min_rounded, y_max_rounded)
            ax1.set_yticks(y_ticks)
            ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
            ax1.set_xlim(time_vals.min(), time_vals.max())
            plt.xticks(rotation=45, ha='right')
            add_edge_labels_if_needed(ax1, time_vals)

    # Overlay reference temperature (auto axis, no preset)
        temp_vals = pd.to_numeric(df[temp_ref_col], errors="coerce")
        tmin, tmax = np.nanmin(temp_vals), np.nanmax(temp_vals)
        # Round temperature axis to multiples of 5 so tick labels are "nice"
        # (e.g. -20, -15, -10, -5, 0, 5...) and avoid odd tick endpoints like -11 or 9.
        tmin_rounded = int(math.floor(float(tmin) / 5.0) * 5)
        tmax_rounded = int(math.ceil(float(tmax) / 5.0) * 5)
        # Ensure the temperature axis always extends up to at least 50°C
        # (user request). This keeps tick endpoints nice (multiples of 5)
        # while guaranteeing a 50°C upper bound when appropriate.
        if tmax_rounded < 50:
            tmax_rounded = 50
        # ensure a non-zero span
        if tmin_rounded == tmax_rounded:
            tmin_rounded -= 5
            tmax_rounded += 5
        t_ticks = np.arange(tmin_rounded, tmax_rounded + 1, 5)
        # Limit the number of ticks so labels remain readable; pick a step
        # to reduce the ticks to at most max_ticks while keeping round values.
        max_ticks = 8
        if len(t_ticks) > max_ticks:
            # Choose up to `max_ticks` tick positions while ensuring the
            # highest tick (top of axis) is always included. Use evenly
            # spaced indices from 0..n-1 and pick unique integer positions
            # so we never drop the end tick via simple slicing.
            n = len(t_ticks)
            # linspace from 0 to n-1 with max_ticks points, round to nearest int
            idxs = np.unique(np.round(np.linspace(0, n - 1, max_ticks)).astype(int))
            t_ticks = t_ticks[idxs]
        ax2 = ax1.twinx()
        # align time and temp_vals
        try:
            time_trim3, [temp_trim] = align_time_and_series(time_vals, temp_vals)
        except Exception:
            time_trim3 = pd.Series(time_vals).reset_index(drop=True)
            temp_trim = temp_vals.reset_index(drop=True).iloc[:len(time_trim3)] if hasattr(temp_vals, 'reset_index') else pd.Series(temp_vals).iloc[:len(time_trim3)]
        # Connect temperature points with a dotted interpolated line so large gaps
        # are visually connected, but still show the original sampled points.
        try:
            temp_series = pd.Series(temp_trim).reset_index(drop=True)
            temp_interp = temp_series.interpolate(method='linear', limit_direction='both')
        except Exception:
            temp_interp = temp_trim
        # Plot interpolated dotted line (labelled)
        safe_plot(ax2, time_trim3, temp_interp, csv_path=csv_path, sensor_name=f"REF TEMP: {temp_ref_col}", label=f"REF TEMP: {temp_ref_col}", linestyle=':', color='red', alpha=0.9, zorder=2)
        # Overplot original points as unlabeled markers so actual samples remain visible
        safe_plot(ax2, time_trim3, temp_trim, csv_path=csv_path, sensor_name=f"REF TEMP: {temp_ref_col} (points)", label=None, linestyle='none', marker='o', markersize=3, color='red', alpha=0.8, zorder=3)
        ax2.set_ylabel("Reference Temperature (°C)")
        ax2.set_ylim(tmin_rounded, tmax_rounded)
        # Use FixedLocator to force the ticks we computed. This ensures matplotlib
        # does not auto-adjust the tick positions when rendering/saving.
        try:
            ax2.yaxis.set_major_locator(FixedLocator(t_ticks))
        except Exception:
            # fallback to set_yticks if FixedLocator fails for any reason
            try:
                ax2.set_yticks(t_ticks)
            except Exception:
                pass
        # keep tick labels visible (we already reduced tick count above)

        ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.yaxis.grid(False)

        # Place left/right legends outside the plot area so they don't overlay axes
        fig = ax1.get_figure()
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Reserve margins and place legends using the shared helper
        legend1, legend2 = place_legends_outside(fig, ax1, lines1, labels1, ax2, lines2, labels2)
        plt.draw()

        save_path = out_dir / f"{instrument}_allSensorsDiffTempOverlay_{csv_path.stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        # Diagnostic tick/data printing removed to reduce console noise.
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
        csv_path,
        sensor_precisions,
        initial_cut_intervals=collected_cut_intervals,
    )

    # NOTE: We no longer overwrite df with interpolated spike-removed series. Spike
    # masking is applied at plot time via `spike_masks` so every humidity plot shows
    # clipped regions consistently.

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
            # Choose a non-red color for this sensor-type average using the same
            # cycle-avoidance logic as used for individual sensors so we do not
            # accidentally use reserved red shades for averages.
            try:
                group_color, color_idx = pick_nonred_color(color_cycle, color_idx)
            except Exception:
                group_color = None
            # Apply spike detection and masking for humidity instruments
            avg_diff_plot = avg_diff
            if instrument.lower() in ("humidity", "rh_test"):
                try:
                    avg_diff_plot = remove_spikes(avg_diff_plot, window=20, z_thresh=spike_threshold)
                except Exception:
                    pass
            # Defensive align avg series with time axis
            try:
                time_trim_avg, [avg_trim] = align_time_and_series(time_vals, avg_diff_plot.rolling(window=20, center=True, min_periods=1).mean())
            except Exception:
                time_trim_avg = pd.Series(time_vals).reset_index(drop=True)
                avg_trim = avg_diff_plot.rolling(window=20, center=True, min_periods=1).mean().reset_index(drop=True).iloc[:len(time_trim_avg)]
            # Connect averages with interpolated dotted line only for temperature instruments.
            try:
                avg_series = pd.Series(avg_trim).reset_index(drop=True)
                avg_interp = avg_series.interpolate(method='linear', limit_direction='both')
            except Exception:
                avg_interp = avg_trim
            if do_connect:
                # Show connector as dotted but unlabeled, label the solid averaged series
                safe_plot(plt.gca(), time_trim_avg, avg_interp, csv_path=csv_path, sensor_name=f"{sensor_type} avg", label=None, linestyle=':', alpha=0.9, color=group_color)
                safe_plot(plt.gca(), time_trim_avg, avg_trim, csv_path=csv_path, sensor_name=f"{sensor_type} avg", label=f"{sensor_type} avg - ref", linestyle='-', alpha=0.9, color=group_color)
            else:
                # For non-temperature instruments, plot the averaged series labeled only
                safe_plot(plt.gca(), time_trim_avg, avg_trim, csv_path=csv_path, sensor_name=f"{sensor_type} avg", label=f"{sensor_type} avg - ref", linestyle='-', alpha=0.9, color=group_color)
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Time")
            units = get_unit_for_instrument(instrument)
            ylabel = f"Difference from ref ({units})" if units else f"Difference from ref"
            plt.ylabel(ylabel)
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
            save_path = out_dir / f"{instrument}_{sensor_type.upper()}_allSensorTypeAvgDiff_{csv_path.stem}.png"
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
    , sensor_precisions=None,
    initial_cut_intervals=None
):
    # (debug prints removed) spike_threshold in use: kept for internal logic
    """
    Plots all sensor differences from reference, with the reference value overlaid on a secondary y-axis.
    For humidity/rh_test instruments: detect steep increases in the difference (sensor - reference),
    replace (smooth) the large jumps by connecting the value at the start of one spike to the
    value at the start of the next spike, and plot the smoothed differences so spikes are not shown.
    """
    # Initialize axes placeholders (guard against code paths where twins aren't created)
    ax1 = None
    ax2 = None

    # Connect points across gaps for temperature instruments and for humidity plots
    # where we want interpolated connectors over removed spike regions.
    do_connect = True if ("temperature" in instrument.lower() or instrument.lower() in ("humidity", "rh_test")) else False

    # Prepare reference overlay values
    ref_overlay_vals = pd.to_numeric(df[ref], errors="coerce").values

    # Detect main hills (local maxima) for temperature labeling (full data)
    ignore_frac = 0.05
    n_full = len(ref_overlay_vals)
    start_idx_full = int(n_full * ignore_frac)
    end_idx_full = int(n_full * (1 - ignore_frac))
    ref_overlay_cropped = ref_overlay_vals[start_idx_full:end_idx_full]
    peaks, _ = find_peaks(ref_overlay_cropped, distance=300, prominence=2)
    peaks = peaks + start_idx_full

    temp_ref_col = find_reference_temp(df)
    temp_vals = pd.to_numeric(df[temp_ref_col], errors="coerce").values if temp_ref_col is not None else None

    # Prepare sensors and working copy
    non_reference_sensors = [s for s in valid_sensors if s not in references]
    df_clean = df.copy()
    # --- Apply edge cut to all data upfront ---
    n = len(df_clean)
    start_idx = int(BEGIN_EDGE_CUT_PERCENT * n)
    end_idx = n - int(END_EDGE_CUT_PERCENT * n)
    df_clean = df_clean.iloc[start_idx:end_idx].reset_index(drop=True)
    time_vals = time_vals.iloc[start_idx:end_idx].reset_index(drop=True) if hasattr(time_vals, 'iloc') else time_vals[start_idx:end_idx]
    ref_vals = ref_vals.iloc[start_idx:end_idx] if hasattr(ref_vals, 'iloc') else ref_vals[start_idx:end_idx]
    ref_overlay_vals = ref_overlay_vals[start_idx:end_idx] if 'ref_overlay_vals' in locals() and len(ref_overlay_vals) == n else ref_overlay_vals
    if temp_ref_col is not None and temp_vals is not None:
        temp_vals = temp_vals[start_idx:end_idx]

    # ...existing code...

    if instrument.lower() in ("humidity", "rh_test"):
        # --- Detect plateaus in reference humidity (used for masking only) ---
        filtered_plateaus = []
        ref_hum = (
            pd.to_numeric(df_clean[ref], errors="coerce")
            .interpolate()
            .rolling(window=11, center=True, min_periods=1)
            .mean()
            .values
        )
        step_tolerance = 0.02
        diffs = np.diff(ref_hum)
        change_points = np.where(np.abs(diffs) > step_tolerance)[0]
        step_boundaries = []
        if len(change_points) > 0:
            start = change_points[0]
            for i in range(1, len(change_points)):
                if change_points[i] - change_points[i - 1] > 50:
                    step_boundaries.append(start)
                    start = change_points[i]
            step_boundaries.append(start)
        plateaus = []
        # Build plateaus as (start_idx, end_idx, mean_value)
        for i in range(len(step_boundaries) - 1):
            s = step_boundaries[i]
            e = step_boundaries[i + 1]
            if e - s > 100:
                mean_val = np.mean(ref_hum[s:e])
                plateaus.append((s, e, mean_val))
        last_val = None
        for item in plateaus:
            idx, end_idx, val = item
            if last_val is None or abs(val - last_val) > step_tolerance / 2:
                filtered_plateaus.append((idx, end_idx, val))
                last_val = val

        # Mask at the exact indices used for humidity labeling (filtered_plateaus)
        mask_window = 1000  # number of points before and after each jump to mask (adjust as needed)
        label_times = []
        plateau_indices = []
        for idx, end_idx, hum_val in filtered_plateaus:
            if idx < len(df_clean["time"]):
                label_time = df_clean["time"].iloc[idx]
                label_times.append(label_time)
                plateau_indices.append((idx, end_idx))

        # (Previously printed plateau debug lines here.)
        # We now detect and report hysteresis periods per sensor later (after diff_df is built)
        # --- Create a DataFrame of difference values for all sensors ---
        diff_df = pd.DataFrame(index=df_clean.index)
        ref_numeric = pd.to_numeric(df_clean[ref], errors="coerce")

        # Collect all cut intervals (from spikes and plateau masks) so we can report
        # which time periods are being removed/masked from plots. Start with any
        # intervals detected earlier during main/temp processing (passed in).
        cut_intervals = list(initial_cut_intervals) if initial_cut_intervals else []  # list of (start_idx, end_idx, source)

        for sensor in non_reference_sensors:
            y_sensor = pd.to_numeric(df_clean[sensor], errors="coerce")
            # Apply spike interpolation removal for humidity instruments (original behavior)
            if instrument.lower() in ("humidity", "rh_test"):
                try:
                    # detect spikes on the raw sensor series and expand them
                    spike_periods = detect_spikes(y_sensor, z_thresh=spike_threshold, window=20)
                    expanded = expand_spike_periods(y_sensor, spike_periods, z_thresh=spike_threshold, window=20, time_index=df_clean["time"], merge_gap_seconds=10)
                    # Record expanded intervals with sensor as source
                    for s_idx, e_idx in expanded:
                        cut_intervals.append((s_idx, e_idx, sensor))
                    # The time index for df_clean starts at the edge-cut; align by using df_clean["time"]
                    print_spike_periods(expanded, df_clean["time"], sensor_name=sensor, csv_path=csv_path)
                    y_sensor = remove_spikes(y_sensor, window=20, z_thresh=spike_threshold)
                except Exception:
                    pass
            diff_df[sensor] = y_sensor - ref_numeric

        # For debug: print the difference value for the first non-reference sensor at each humidity label (plateau) time
        if non_reference_sensors:
            for idx, end_idx, hum_val in filtered_plateaus:
                if idx < len(diff_df):
                    start = idx
                    end = min(len(diff_df), idx + masking_window_size + 1)
                    # record plateau mask interval
                    cut_intervals.append((start, end - 1, 'PLATEAU'))
            for sensor in non_reference_sensors:
                mask = np.zeros(len(diff_df), dtype=bool)
                for idx, end_idx, hum_val in filtered_plateaus:
                    if idx < len(diff_df):
                        start = idx
                        end = min(len(diff_df), idx + masking_window_size + 1)
                        mask[start:end] = True
                # Mask these points in the difference DataFrame for plotting using .loc to avoid chained assignment
                diff_df.loc[mask, sensor] = np.nan

        # Now merge all cut_intervals (spikes + plateaus) into timestamped merged intervals
        try:
            merged_cut = merge_time_intervals(cut_intervals, df_clean["time"], merge_gap_seconds=1)
            # merged_cut detected; detailed CUT period printing suppressed to avoid
            # excessive console output. The merged_cut list is still available for
            # future logging or export if desired.
        except Exception:
            pass

        # Detect hysteresis periods on the (masked) difference DataFrame and print
        try:
            hysteresis = detect_hysteresis_periods(diff_df, df_clean["time"], threshold=HYSTERESIS_THRESHOLD, min_duration_seconds=HYSTERESIS_MIN_DURATION_SECONDS)
            # Hysteresis periods detected; detailed printing suppressed. The
            # `hysteresis` dict is still produced and can be exported to CSV by
            # a follow-up change if machine-readable output is desired.
        except Exception:
            pass

        # --- Plotting using df_clean so spikes are not graphed ---
        # Compute data-driven y-limits/ticks from the difference dataframe so
        # plots are centered on the actual spread of the data instead of a
        # hard-coded range. This produces tighter y-limits when the data
        # occupies a smaller range (e.g. most temperature diffs within -2..2).
        try:
            flat = diff_df.to_numpy().ravel()
            finite = flat[np.isfinite(flat)]
            if finite.size > 0:
                dmin = float(np.nanmin(finite))
                dmax = float(np.nanmax(finite))
                if dmin == dmax:
                    # Single-value series: provide a small pad so the line is visible
                    pad = max(abs(dmin) * 0.1, 0.5)
                else:
                    pad = max((dmax - dmin) * 0.1, 0.25)
                y_min_data = dmin - pad
                y_max_data = dmax + pad
                # Choose a reasonable tick step based on span
                span = y_max_data - y_min_data
                raw_step = span / 6.0 if span > 0 else 1.0
                if raw_step <= 0.1:
                    step = 0.05
                elif raw_step <= 0.25:
                    step = 0.1
                elif raw_step <= 0.5:
                    step = 0.25
                elif raw_step <= 1.0:
                    step = 0.5
                else:
                    # round to nearest 1, 2, 5, 10-like step
                    magnitude = 10 ** math.floor(math.log10(raw_step))
                    step = math.ceil(raw_step / magnitude) * magnitude
                # Build ticks aligned to step
                y_low_tick = math.floor(y_min_data / step) * step
                y_high_tick = math.ceil(y_max_data / step) * step
                y_ticks = np.arange(y_low_tick, y_high_tick + step / 2.0, step)
                # Ensure at least 3 ticks
                if len(y_ticks) < 3:
                    step = max(step / 2.0, 0.01)
                    y_ticks = np.arange(y_low_tick, y_high_tick + step / 2.0, step)
                y_min_rounded = float(y_ticks[0])
                y_max_rounded = float(y_ticks[-1])
        except Exception:
            # Fall back to previously computed folder-wide limits
            pass

        # (temperature-specific clamp removed — axis selection is handled earlier)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        for sensor in non_reference_sensors:
            diff_masked = diff_df[sensor]
            # Align time axis with diff_masked (in case of dropped/NaN rows)
            if "time" in df_clean.columns:
                time_for_plot = df_clean["time"]
            else:
                time_for_plot = np.arange(len(diff_masked))
            # Final defensive trim to ensure matched lengths
            try:
                time_for_plot, [diff_masked] = align_time_and_series(time_for_plot, diff_masked)
            except Exception:
                # fallback: convert and trim to min
                time_for_plot = pd.Series(time_for_plot).reset_index(drop=True)
                diff_masked = pd.Series(diff_masked).reset_index(drop=True).iloc[:len(time_for_plot)]
            # Apply smoothing so overlay matches main plots
            try:
                diff_masked = diff_masked.rolling(window=20, center=True, min_periods=1).mean()
            except Exception:
                pass
            precision = sensor_precisions.get(sensor, float('nan')) if sensor_precisions is not None else float('nan')
            label = f"{sensor} - ref"
            # Create interpolated dotted line to connect gaps and label it; then
            # plot the masked-smoothed series unlabeled underneath.
            try:
                diff_series = pd.Series(diff_masked).reset_index(drop=True)
                diff_interp = diff_series.interpolate(method='linear', limit_direction='both')
            except Exception:
                diff_interp = diff_masked
            if do_connect:
                # Interpolated dotted (unlabelled) for temperature/humidity instruments only
                safe_plot(ax1, time_for_plot, diff_interp, csv_path=csv_path, sensor_name=sensor, label=None, linestyle=':', alpha=0.9, zorder=1, color=sensor_colors.get(sensor, None))
                # Original masked-smoothed series as labeled solid line
                safe_plot(ax1, time_for_plot, diff_masked, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors.get(sensor, None))
            else:
                # For non-temperature instruments, plot the masked-smoothed series labeled only
                safe_plot(ax1, time_for_plot, diff_masked, csv_path=csv_path, sensor_name=sensor, label=label, linestyle='-', alpha=0.9, zorder=2, color=sensor_colors.get(sensor, None))

        # No need to adjust xlim; all data is already trimmed
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Time")
        units = get_unit_for_instrument(instrument)
        ylabel = f"Difference from ref ({units})" if units else f"Difference from ref"
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{instrument} - SENSOR DIFFERENCES + REF OVERLAY (SMOOTHED) ({csv_path.stem})")
        ax1.set_ylim(y_min_rounded, y_max_rounded)
        ax1.set_yticks(y_ticks)
        ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
        ax1.set_xlim(time_vals.min(), time_vals.max())
        plt.xticks(rotation=45, ha='right')
        add_edge_labels_if_needed(ax1, time_vals)

        # Overlay reference value (original, not smoothed)
        ax2 = ax1.twinx()
        # Ensure reference overlay is aligned with the time axis used for plotting
        try:
            time_trim_ref, [ref_trim] = align_time_and_series(time_vals, pd.Series(ref_overlay_vals))
        except Exception:
            time_trim_ref = pd.Series(time_vals).reset_index(drop=True)
            ref_trim = pd.Series(ref_overlay_vals).reset_index(drop=True).iloc[:len(time_trim_ref)]
        # Connect reference points with a dotted interpolated line so gaps are connected
        try:
            ref_series = pd.Series(ref_trim).reset_index(drop=True)
            ref_interp = ref_series.interpolate(method='linear', limit_direction='both')
        except Exception:
            ref_interp = ref_trim
        # Plot interpolated dotted reference (labelled)
        # Show a compact REF label (do not print full column name in the legend)
        safe_plot(ax2, time_trim_ref, ref_interp, csv_path=csv_path, sensor_name="REF", label="REF", linestyle=':', color='red', alpha=0.9, zorder=2)
        # Plot original reference samples as unlabeled markers
        safe_plot(ax2, time_trim_ref, ref_trim, csv_path=csv_path, sensor_name="REF", label=None, linestyle='none', marker='o', markersize=3, color='red', alpha=0.8, zorder=3)
    # Force ref axis to 0..100 with 10-unit ticks and set axis label with units
        rmin_rounded = 0
        rmax_rounded = 100
        r_ticks = np.arange(0, 101, 10)
        ax2.set_ylabel(f"Reference ({get_unit_for_instrument(instrument)})")
        ax2.set_ylim(rmin_rounded, rmax_rounded)
        ax2.set_yticks(r_ticks)

        # Center temperature labels at the bottom-middle of each detected humidity plateau
        if temp_ref_col is not None and temp_vals is not None:
            try:
                # Align temp_vals to the time base used for the reference overlay
                try:
                    time_for_temp, [temp_aligned] = align_time_and_series(time_trim_ref, temp_vals)
                except Exception:
                    time_for_temp = pd.Series(time_trim_ref).reset_index(drop=True)
                    temp_aligned = pd.Series(temp_vals).reset_index(drop=True).iloc[:len(time_for_temp)]

                # Prefer explicit flat plateaus near 90% humidity. Scan smoothed reference
                # for regions where humidity >= 89.5 and flat, then choose up to three
                ref_series = pd.Series(ref_trim).reset_index(drop=True)
                ref_smooth = ref_series.interpolate().rolling(window=11, center=True, min_periods=1).mean()
                try:
                    is_high = (ref_smooth >= 89.5).astype(int).values
                    from itertools import groupby
                    runs = []
                    for k, g in groupby(enumerate(is_high), lambda x: x[1]):
                        if k == 1:  # high region
                            group = list(g)
                            start = group[0][0]
                            end = group[-1][0]
                            length = end - start + 1
                            runs.append((start, end, length))
                    # keep runs that are reasonably long (at least 30 samples)
                    runs = [r for r in runs if r[2] >= 30]
                    # Sort by length (desc) and take up to 3
                    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)[:3]
                    chosen_indices = [int((s + e) // 2) for s, e, _ in runs_sorted]
                except Exception:
                    chosen_indices = []

                # If no suitable high-90 plateaus found, fallback to previously-detected plateaus
                if not chosen_indices:
                    plateau_sources = filtered_plateaus if 'filtered_plateaus' in locals() else []
                    for p in plateau_sources:
                        start_idx, end_idx, _ = p
                        chosen_indices.append(int((start_idx + end_idx) // 2))
                # Final fallback: three evenly spaced indices
                if not chosen_indices:
                    L = len(ref_series)
                    chosen_indices = [int(L * 0.25), int(L * 0.5), int(L * 0.75)]

                # Limit to three and annotate at bottom-middle of each hill
                for idx in chosen_indices[:3]:
                    if idx < 0 or idx >= len(ref_series):
                        continue
                    try:
                        t = time_for_temp.iloc[idx]
                        temp_val = float(temp_aligned.iloc[idx]) if idx < len(temp_aligned) else float('nan')
                        # Place label slightly below the plateau value (bottom middle)
                        ref_val_at_idx = float(ref_series.iloc[idx])
                        # Place the temperature label at a fixed humidity level (10%) — bottom middle of the hill
                        y_label = 10
                        ax2.plot([t], [ref_val_at_idx], marker='o', markersize=4, color='blue', alpha=0.9)
                        ax2.annotate(f"{temp_val:.0f}°C", (t, y_label), textcoords="offset points", xytext=(0, 0), ha='center', va='top', fontsize=9, color='blue', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=0.6, alpha=0.9))
                    except Exception:
                        continue
            except Exception:
                # Non-fatal: if alignment or detection fails, skip annotations
                pass

        ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.yaxis.grid(False)

    else:
        # Non-humidity instruments: compute data-driven y-limits based on
        # the actual sensor - reference differences, then plot.
        try:
            diffs = []
            for sensor in non_reference_sensors:
                y_sensor = pd.to_numeric(df_clean[sensor], errors="coerce")
                diffs.append((y_sensor - ref_numeric).to_numpy())
            if diffs:
                allvals = np.concatenate([d.ravel() for d in diffs])
                finite = allvals[np.isfinite(allvals)]
                if finite.size > 0:
                    dmin = float(np.nanmin(finite))
                    dmax = float(np.nanmax(finite))
                    if dmin == dmax:
                        pad = max(abs(dmin) * 0.1, 0.5)
                    else:
                        pad = max((dmax - dmin) * 0.1, 0.25)
                    y_min_data = dmin - pad
                    y_max_data = dmax + pad
                    span = y_max_data - y_min_data
                    raw_step = span / 6.0 if span > 0 else 1.0
                    if raw_step <= 0.1:
                        step = 0.05
                    elif raw_step <= 0.25:
                        step = 0.1
                    elif raw_step <= 0.5:
                        step = 0.25
                    elif raw_step <= 1.0:
                        step = 0.5
                    else:
                        magnitude = 10 ** math.floor(math.log10(raw_step))
                        step = math.ceil(raw_step / magnitude) * magnitude
                    y_low_tick = math.floor(y_min_data / step) * step
                    y_high_tick = math.ceil(y_max_data / step) * step
                    y_ticks = np.arange(y_low_tick, y_high_tick + step / 2.0, step)
                    y_min_rounded = float(y_ticks[0])
                    y_max_rounded = float(y_ticks[-1])
        except Exception:
            pass

        # (temperature-specific clamp removed — axis selection is handled earlier)

        # Non-humidity instruments: simple overlay plot (no smoothing)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        for sensor in non_reference_sensors:
            y_sensor = pd.to_numeric(df_clean[sensor], errors="coerce")
            diff = y_sensor - ref_vals
            safe_plot(ax1, time_vals, diff, csv_path=csv_path, sensor_name=sensor, label=f"{sensor} - ref", alpha=0.8, zorder=1, color=sensor_colors.get(sensor, None))
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Time")
        units = get_unit_for_instrument(instrument)
        ylabel = f"Difference from ref ({units})" if units else f"Difference from ref"
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{instrument} - SENSOR DIFFERENCES + REF OVERLAY ({csv_path.stem})")
        ax1.set_ylim(y_min_rounded, y_max_rounded)
        ax1.set_yticks(y_ticks)
        ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
        ax1.set_xlim(time_vals.min(), time_vals.max())
        plt.xticks(rotation=45, ha='right')
        add_edge_labels_if_needed(ax1, time_vals)

    # Common legend/finish for both branches: create separate legends for left/right axes
    lines1, labels1 = ([], []) if ax1 is None else ax1.get_legend_handles_labels()
    lines2, labels2 = ([], []) if ax2 is None else ax2.get_legend_handles_labels()
    legend_ax = ax2 if ax2 is not None else ax1
    if legend_ax is not None:
        fig = legend_ax.get_figure()
        # Use helper to reserve margins and place left/right legends outside the plot
        legend1, legend2 = place_legends_outside(fig, ax1, lines1, labels1, ax2, lines2, labels2)
    plt.draw()

    save_path = out_dir / f"{instrument}_allSensorsDiffRefOverlay_{csv_path.stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    # Diagnostic tick/data printing for reference overlay removed to keep
    # output compact. If you want machine-readable diagnostics, I can add
    # an optional CSV export for these metrics in the next change.
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
