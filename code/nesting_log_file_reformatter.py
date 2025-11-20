import re
import ast
import pandas as pd
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================
parent_origin = Path("/Users/reesecoleman/Desktop/UCAR Data/data/RawCollection")
data_destination = Path("/Users/reesecoleman/Desktop/UCAR Data/data/formattedcsv")
data_destination.mkdir(parents=True, exist_ok=True)

reference_files = [
    Path("/Users/reesecoleman/Desktop/UCAR Data/data/RawCollection/Temperature_Test/Cal_Test_Data/Super_therm_020425"),
    Path("/Users/reesecoleman/Desktop/UCAR Data/data/RawCollection/Pressure/Cal_Test_Data/3dpaws_pressure"),
    Path("/Users/reesecoleman/Desktop/UCAR Data/data/RawCollection/RH_Test/Cal_Test_Data/humidity_chamber_full_header.csv"),  # Only keep the CSV reference for humidity
]

LABVIEW_UNIX_OFFSET = 2082844800

# Set the averaging window (in minutes) for the resampled CSVs
RESAMPLE_MINUTES = .5  # Set to 1 for 1-minute average, 5 for 5-minute average, etc.

# ================================================================
# PARSERS
# ================================================================

def parse_line(s: str):
    s = s.strip()
    if not s:
        return None
    s = re.sub(r',\s*([}\]])', r'\1', s)
    s = re.sub(r"\*0*1(?=\d)", "", s)
    try:
        obj = ast.literal_eval(s)
        return dict(obj) if isinstance(obj, dict) else None
    except Exception:
        return None

def parse_supertherm(lines: list[str]) -> pd.DataFrame:
    start_time = pd.to_datetime(lines[0].replace("\ufeff", ""), format="%m/%d/%Y %I:%M %p")
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            labview_timestamp = float(parts[0])
            unix_timestamp = labview_timestamp - LABVIEW_UNIX_OFFSET
            t = pd.to_datetime(unix_timestamp, unit="s")
            ref_temp = float(parts[1])
            rows.append({"time": t, "reference": ref_temp})
        except Exception:
            continue
    return pd.DataFrame(rows)

def parse_3dpaws_pressure(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = pd.to_datetime(parts[0], utc=True).tz_convert(None)
            val_str = parts[1].replace("*0001", "")
            val = float(val_str)
            rows.append({"time": t, "reference": val})
        except Exception:
            continue
    return pd.DataFrame(rows)

def parse_humidity_chamber_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse humidity chamber CSV reference file.
    Only reads: Timestamp, %RH@PC, %RH@PcTc, CmbTemp.
    Converts LabVIEW time to Unix epoch for synchronization.
    """
    LABVIEW_UNIX_OFFSET = 2082844800

    # Print the first 5 rows for debugging
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [next(f) for _ in range(5)]
    print("DEBUG: First 5 rows of the CSV file:")
    for i, line in enumerate(raw_lines):
        print(f"Row {i}: {line.strip()}")

    # Read with header at row 2 (third row, zero-based)
    df = pd.read_csv(csv_path, header=2, usecols=["Timestamp", "%RH@PC", "%RH@PcTc", "CmbTemp"])
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    print("DEBUG: Parsed header columns (restricted):", df.columns.tolist())

    # Convert LabVIEW time to Unix epoch
    if "Timestamp" not in df.columns:
        raise KeyError(f"Could not find 'Timestamp' column in CSV. Columns: {df.columns.tolist()}")
    df["time"] = pd.to_numeric(df["Timestamp"], errors="coerce") - LABVIEW_UNIX_OFFSET
    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")

    # Only use CmbTemp for temp_reference, never any other column
    if "CmbTemp" not in df.columns:
        print("DEBUG: 'CmbTemp' column not found! All temp_reference values will be empty.")
        temp_reference = [None] * len(df)
    else:
        temp_reference = df["CmbTemp"]

    # Build cleaned dataframe
    out = pd.DataFrame({
        "time": df["time"],
        "reference_rhpc_h": df["%RH@PC"] if "%RH@PC" in df.columns else float("nan"),
        "reference_rhpctc_h": df["%RH@PcTc"] if "%RH@PcTc" in df.columns else float("nan"),
        "temp_reference": temp_reference,
    })

    print("DEBUG: Output DataFrame columns:", out.columns.tolist())
    print("DEBUG: First 3 rows of output DataFrame:")
    print(out.head(3))

    out = out.dropna(subset=["time"])
    out = out.sort_values("time")
    return out

def load_reference_file(ref_path: Path) -> pd.DataFrame:
    # Handle CSV reference file
    if ref_path.suffix.lower() == ".csv":
        ref_df = parse_humidity_chamber_csv(ref_path)
        # Save the temp_reference column for later use
        load_reference_file.saved_temp_reference = ref_df[["time", "temp_reference"]].copy()
        return ref_df
    with open(ref_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    first_line = lines[0]
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", first_line):
        if "super" in ref_path.name.lower():
            return parse_supertherm(lines)
        else:
            raise ValueError("No longer supported: 3dpaws_rh_test_sht_hdc_021825 or similar RH text reference files.")
    elif re.match(r"\d{4}-\d{2}-\d{2}T", first_line):
        return parse_3dpaws_pressure(lines)
    else:
        raise ValueError(f"Unknown reference file format: {ref_path}")

# Initialize the attribute to avoid AttributeError
load_reference_file.saved_temp_reference = None

# ================================================================
# CORE PROCESSING
# ================================================================

def process_one_instrument(
    log_paths: list[Path],
    out_name: str,
    destination: Path,
    relative_folder: Path,
    ref_df: pd.DataFrame | None = None,
    use_reference: bool = False,
):
    all_dfs = []
    headers: list[str] = []
    seen = set()

    folder_parts = [p.lower() for p in relative_folder.parts]
    if any("temp" in p for p in folder_parts):
        ref_suffix = "t"
    elif any("press" in p for p in folder_parts):
        ref_suffix = "p"
    elif any("humid" in p or "rh" in p for p in folder_parts):
        ref_suffix = "r"
    else:
        ref_suffix = "ref"

    for log in sorted(log_paths):
        print(f"Reading {log}")
        rows = []
        with open(log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                rec = parse_line(line)
                if rec is None:
                    continue
                rows.append(rec)
                for k in rec.keys():
                    if k not in seen:
                        headers.append(k)
                        seen.add(k)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.reindex(columns=headers)
        if "at" in df.columns and "time" not in df.columns:
            df["at"] = pd.to_datetime(df["at"], errors="coerce")
            df.rename(columns={"at": "time"}, inplace=True)
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        all_dfs.append(df)
        # Debug: print first 10 temp_reference values after reading this log
        if "temp_reference" in df.columns:
            print(f"DEBUG: First 10 temp_reference values after reading {log}:")
            print(df["temp_reference"].head(10).tolist())
        else:
            print(f"DEBUG: No temp_reference column found after reading {log}.")

    if not all_dfs:
        print(f"No records for {out_name}; skipping.")
        return

    out_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    if "time" in out_df.columns:
        out_df = out_df.dropna(subset=["time"]).copy()
        out_df.sort_values(by="time", inplace=True)

    if use_reference and ref_df is not None and not ref_df.empty:
        ref_df = ref_df.dropna(subset=["time"]).copy()
        ref_df.sort_values(by="time", inplace=True)
        if "time" in out_df.columns:
            out_df["time"] = pd.to_datetime(out_df["time"]).dt.tz_localize(None)
        if "time" in ref_df.columns:
            ref_df["time"] = pd.to_datetime(ref_df["time"]).dt.tz_localize(None)

        # Merge only RH columns (not temp_reference)
        if {"reference_rhpc_h", "reference_rhpctc_h", "temp_reference"}.issubset(ref_df.columns):
            print("DEBUG: Merging humidity reference columns into output file.")
            print("DEBUG: Reference DataFrame columns:", ref_df.columns.tolist())
            print("DEBUG: First 3 rows of reference DataFrame:")
            print(ref_df[["time", "reference_rhpc_h", "reference_rhpctc_h", "temp_reference"]].head(3))

            out_df = pd.merge_asof(
                out_df,
                ref_df[["time", "reference_rhpc_h", "reference_rhpctc_h"]],
                on="time",
                direction="nearest",
                tolerance=pd.Timedelta("5s"),
            )

            # temp_reference will be replaced below, so do not assign here

            # Ensure both columns exist, even if NaN
            if "reference_rhpc_h" not in out_df.columns:
                out_df["reference_rhpc_h"] = float("nan")
            if "reference_rhpctc_h" not in out_df.columns:
                out_df["reference_rhpctc_h"] = float("nan")
        else:
            out_df = pd.merge_asof(
                out_df,
                ref_df,
                on="time",
                direction="nearest",
                tolerance=pd.Timedelta("5s"),
            )
            # Rename reference column if present
            if "reference" in out_df.columns:
                new_ref_col = f"reference_{ref_suffix}"
                out_df.rename(columns={"reference": new_ref_col}, inplace=True)

    # === Replace temp_reference with saved column from reference CSV ===
    if load_reference_file.saved_temp_reference is not None:
        print("DEBUG: Replacing temp_reference column with saved values from reference CSV.")
        # Map by nearest time
        saved_temp_ref = load_reference_file.saved_temp_reference.copy()
        saved_temp_ref = saved_temp_ref.dropna(subset=["time"])
        saved_temp_ref = saved_temp_ref.sort_values("time")
        out_df["temp_reference"] = pd.merge_asof(
            out_df[["time"]].sort_values("time"),
            saved_temp_ref.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("5s"),
        )["temp_reference"].values
        print("DEBUG: First 10 temp_reference values after replacement:")
        print(out_df["temp_reference"].head(10).tolist())
    else:
        print("DEBUG: No saved temp_reference column found to replace.")

    # Final debug: show what column is being used for temp_reference in the output
    if "temp_reference" in out_df.columns:
        print("DEBUG: Final output contains 'temp_reference' column.")
        print("DEBUG: First 10 temp_reference values in output:")
        print(out_df["temp_reference"].head(10))
        print("DEBUG: Unique values in temp_reference:", out_df["temp_reference"].unique())
    else:
        print("DEBUG: 'temp_reference' column missing from final output!")

    # === Only keep temp_reference for humidity folders ===
    folder_parts = [p.lower() for p in relative_folder.parts]
    is_humidity = any("humid" in p or "rh" in p for p in folder_parts)
    if not is_humidity and "temp_reference" in out_df.columns:
        print("DEBUG: Dropping temp_reference column for non-humidity instrument.")
        out_df = out_df.drop(columns=["temp_reference"])

    out_dir = destination / relative_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows → {out_path}")

    # --- Write x-minute resampled CSV ---
    if "time" in out_df.columns:
        df_resample = out_df.copy()
        df_resample["time"] = pd.to_datetime(df_resample["time"], errors="coerce")
        df_resample = df_resample.dropna(subset=["time"]).set_index("time")
        # Use mean for numeric columns, first for non-numeric
        agg_dict = {col: "mean" for col in df_resample.columns}
        for col in df_resample.columns:
            if not pd.api.types.is_numeric_dtype(df_resample[col]):
                agg_dict[col] = "first"
        df_1min = df_resample.resample(f'{RESAMPLE_MINUTES}T').agg(agg_dict).reset_index()
        out_path_resample = out_dir / f"{RESAMPLE_MINUTES}minute_{out_name}.csv"
        df_1min.to_csv(out_path_resample, index=False)
        print(f"Wrote {len(df_1min)} rows → {out_path_resample}")
    else:
        print("No 'time' column found, skipping resample CSV.")

def main(parent: Path, destination: Path, reference_files: list[Path]):
    loaded_refs = {ref_path: load_reference_file(ref_path) for ref_path in reference_files}
    for folder in [parent] + [p for p in parent.rglob("*") if p.is_dir()]:
        log_paths = list(folder.glob("*.log"))
        if not log_paths:
            continue
        relative_folder = folder.relative_to(parent)
        out_name = folder.name.replace(" ", "_")
        ref_df = None
        use_reference = False
        for ref_path, df in loaded_refs.items():
            # Use the CSV reference for RH folders if present
            if (
                ("humid" in str(folder).lower() or "rh" in str(folder).lower())
                and ref_path.name == "humidity_chamber_full_header.csv"
            ):
                ref_df = df
                use_reference = True
                break
            elif folder in ref_path.parents:
                ref_df = df
                use_reference = True
                break
        process_one_instrument(
            log_paths, out_name, destination, relative_folder,
            ref_df=ref_df,
            use_reference=use_reference
        )

if __name__ == "__main__":
    main(parent_origin, data_destination, reference_files)
