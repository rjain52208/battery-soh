"""
CALCE CS2 Battery Data Loader
==============================
Loads real lithium-ion battery cycling data from CALCE CS2 Excel files
(University of Maryland). Falls back to synthetic data generation if
no raw CALCE files are found.

Expected directory structure:
    data/raw/CS2_33/CS2_33/*.xlsx
    data/raw/CS2_34/CS2_34/*.xlsx
    ...
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR = "data/raw"
NOMINAL_CAPACITY_AH = 1.1  # CS2 prismatic cell nominal capacity

CALCE_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]

# Health tier thresholds (for real battery data)
GOOD_THRESHOLD = 90.0       # SOH >= 90%  → Good
MODERATE_THRESHOLD = 80.0   # SOH >= 80%  → Moderate
                            # SOH <  80%  → Weak


# ---------------------------------------------------------------------------
# Health tier labeling
# ---------------------------------------------------------------------------

def label_health_tier(soh: float) -> str:
    """Assign a health tier label based on SOH percentage."""
    if soh >= GOOD_THRESHOLD:
        return "Good"
    elif soh >= MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Weak"


# ---------------------------------------------------------------------------
# Single cell loader
# ---------------------------------------------------------------------------

def _load_single_cell(cell_name: str, raw_dir: str) -> pd.DataFrame | None:
    """
    Load all Excel files for one CS2 cell and extract per-cycle summaries.

    CALCE Arbin Excel files have columns:
        Data_Point, Test_Time(s), Date_Time, Step_Time(s), Step_Index,
        Cycle_Index, Current(A), Voltage(V), Charge_Capacity(Ah),
        Discharge_Capacity(Ah), Charge_Energy(Wh), Discharge_Energy(Wh),
        dV/dt(V/s), Internal_Resistance(Ohm), ...

    Capacity columns are CUMULATIVE within each file, so per-cycle capacity
    is computed as the difference of max cumulative values between cycles.
    """
    # Find Excel files — handle nested directory (CS2_33/CS2_33/*.xlsx)
    patterns = [
        os.path.join(raw_dir, cell_name, cell_name, "*.xlsx"),
        os.path.join(raw_dir, cell_name, cell_name, "*.xls"),
        os.path.join(raw_dir, cell_name, "*.xlsx"),
        os.path.join(raw_dir, cell_name, "*.xls"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    files = sorted(set(files))

    if not files:
        return None

    all_cycles = []
    global_cycle_offset = 0  # running cycle counter across files

    for fpath in files:
        try:
            xl = pd.ExcelFile(fpath)
            # Data sheet is usually the second sheet (index 1) or named Channel_*
            data_sheet = None
            for sn in xl.sheet_names:
                if sn.startswith("Channel") or sn.lower() == "sheet1":
                    data_sheet = sn
                    break
            if data_sheet is None and len(xl.sheet_names) > 1:
                data_sheet = xl.sheet_names[1]
            elif data_sheet is None:
                data_sheet = xl.sheet_names[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_excel(fpath, sheet_name=data_sheet)

            # Validate required columns
            required = ["Cycle_Index", "Current(A)", "Voltage(V)",
                         "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"]
            if not all(c in df.columns for c in required):
                continue

            df = df.dropna(subset=["Cycle_Index"])
            df["Cycle_Index"] = df["Cycle_Index"].astype(int)

            # Process each cycle in this file
            for cycle_idx, cdf in df.groupby("Cycle_Index"):
                if len(cdf) < 5:  # skip trivial cycles
                    continue

                # Per-cycle discharge capacity: max cumulative in this cycle
                #   minus max cumulative from previous cycle
                cycle_max_discharge = cdf["Discharge_Capacity(Ah)"].max()
                cycle_max_charge = cdf["Charge_Capacity(Ah)"].max()

                # Voltage stats
                v_max = cdf["Voltage(V)"].max()
                v_min = cdf["Voltage(V)"].min()

                # Current stats
                current_mean = cdf["Current(A)"].mean()

                # Timing: step time
                if "Step_Time(s)" in cdf.columns:
                    charge_mask = cdf["Current(A)"] > 0.01
                    discharge_mask = cdf["Current(A)"] < -0.01
                    charge_time_s = cdf.loc[charge_mask, "Step_Time(s)"].max() if charge_mask.any() else 0
                    discharge_time_s = cdf.loc[discharge_mask, "Step_Time(s)"].max() if discharge_mask.any() else 0
                else:
                    charge_time_s = 0
                    discharge_time_s = 0

                # Internal resistance (if available)
                if "Internal_Resistance(Ohm)" in cdf.columns:
                    ir = cdf["Internal_Resistance(Ohm)"]
                    ir_nonzero = ir[ir > 0]
                    resistance = ir_nonzero.mean() if len(ir_nonzero) > 0 else 0.0
                else:
                    resistance = 0.0

                all_cycles.append({
                    "file_cycle": cycle_idx,
                    "cum_discharge_Ah": cycle_max_discharge,
                    "cum_charge_Ah": cycle_max_charge,
                    "charge_voltage_max": v_max,
                    "discharge_voltage_min": v_min,
                    "current_mean_A": current_mean,
                    "charge_time_s": charge_time_s if pd.notna(charge_time_s) else 0,
                    "discharge_time_s": discharge_time_s if pd.notna(discharge_time_s) else 0,
                    "internal_resistance_ohms": resistance,
                })

        except Exception as e:
            print(f"  [Warning] Skipping {os.path.basename(fpath)}: {e}")
            continue

    if not all_cycles:
        return None

    cycles_df = pd.DataFrame(all_cycles)

    # Compute per-cycle discharge capacity from cumulative values
    # The cumulative counter resets per file but increases within a file
    # We need per-cycle delta: diff of cumulative max discharge
    cycles_df["discharge_capacity_Ah"] = cycles_df["cum_discharge_Ah"].diff()
    cycles_df.loc[0, "discharge_capacity_Ah"] = cycles_df.loc[0, "cum_discharge_Ah"]

    cycles_df["charge_capacity_Ah"] = cycles_df["cum_charge_Ah"].diff()
    cycles_df.loc[0, "charge_capacity_Ah"] = cycles_df.loc[0, "cum_charge_Ah"]

    # Filter out negative or zero capacity (file boundary resets)
    cycles_df = cycles_df[cycles_df["discharge_capacity_Ah"] > 0.01].copy()
    cycles_df = cycles_df[cycles_df["charge_capacity_Ah"] > 0.01].copy()

    if len(cycles_df) == 0:
        return None

    # Assign global cycle numbers
    cycles_df = cycles_df.reset_index(drop=True)
    cycles_df["cycle_number"] = range(1, len(cycles_df) + 1)
    cycles_df["battery_id"] = cell_name

    # Compute SOH based on initial capacity
    initial_cap = cycles_df["discharge_capacity_Ah"].iloc[:3].mean()  # avg of first 3 cycles
    cycles_df["SOH_percent"] = (cycles_df["discharge_capacity_Ah"] / initial_cap * 100.0).clip(0, 110)

    # Assign health tiers
    cycles_df["health_tier"] = cycles_df["SOH_percent"].apply(label_health_tier)

    # Convert times to minutes
    cycles_df["charge_time_min"] = cycles_df["charge_time_s"] / 60.0
    cycles_df["discharge_time_min"] = cycles_df["discharge_time_s"] / 60.0

    # Select final columns
    result = cycles_df[[
        "battery_id", "cycle_number",
        "charge_capacity_Ah", "discharge_capacity_Ah",
        "charge_voltage_max", "discharge_voltage_min",
        "charge_time_min", "discharge_time_min",
        "internal_resistance_ohms",
        "SOH_percent", "health_tier",
    ]].copy()

    return result


# ---------------------------------------------------------------------------
# Synthetic fallback (simplified version)
# ---------------------------------------------------------------------------

def _generate_synthetic_fallback(output_dir: str) -> pd.DataFrame:
    """Generate synthetic data if no CALCE files are found."""
    print("[Data Loader] ⚠️  No CALCE CS2 files found in data/raw/")
    print("[Data Loader] Generating synthetic fallback data (6 batteries)...")

    rng = np.random.default_rng(42)
    frames = []

    for i, name in enumerate(CALCE_CELLS):
        n_cycles = rng.integers(300, 900)
        alpha = rng.uniform(0.01, 0.06)
        beta = rng.uniform(0.85, 1.15)
        noise_std = rng.uniform(0.3, 0.8)

        cycles = np.arange(1, n_cycles + 1)
        soh = 100.0 - alpha * np.power(cycles, beta) + rng.normal(0, noise_std, n_cycles)
        soh = np.clip(soh, 0, 110)

        discharge_cap = NOMINAL_CAPACITY_AH * (soh / 100.0) + rng.normal(0, 0.002, n_cycles)
        discharge_cap = np.clip(discharge_cap, 0.01, NOMINAL_CAPACITY_AH * 1.05)

        charge_cap = discharge_cap * (1.0 + 0.003 + 0.01 * (1 - soh / 100.0)) + rng.normal(0, 0.001, n_cycles)
        charge_cap = np.clip(charge_cap, discharge_cap, NOMINAL_CAPACITY_AH * 1.15)

        resistance = 0.09 * (1.0 + 1.5 * (1 - soh / 100.0)) + rng.normal(0, 0.001, n_cycles)

        frames.append(pd.DataFrame({
            "battery_id": name,
            "cycle_number": cycles,
            "charge_capacity_Ah": np.round(charge_cap, 5),
            "discharge_capacity_Ah": np.round(discharge_cap, 5),
            "charge_voltage_max": np.round(4.20 + rng.normal(0, 0.003, n_cycles), 4),
            "discharge_voltage_min": np.round(2.70 + 0.3 * (1 - soh / 100.0) + rng.normal(0, 0.005, n_cycles), 4),
            "charge_time_min": np.round((charge_cap / 0.55) * 60 + rng.normal(0, 0.5, n_cycles), 2),
            "discharge_time_min": np.round((discharge_cap / 0.55) * 60 + rng.normal(0, 0.5, n_cycles), 2),
            "internal_resistance_ohms": np.round(resistance, 5),
            "SOH_percent": np.round(soh, 3),
            "health_tier": [label_health_tier(s) for s in soh],
        }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_calce_data(
    raw_dir: str = RAW_DATA_DIR,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Load CALCE CS2 battery data from Excel files, or fall back to synthetic.

    Returns a DataFrame with per-cycle measurements for all cells.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Try loading real CALCE data
    all_frames = []
    for cell in CALCE_CELLS:
        cell_dir = os.path.join(raw_dir, cell)
        if os.path.isdir(cell_dir):
            print(f"  Loading {cell}...", end=" ", flush=True)
            result = _load_single_cell(cell, raw_dir)
            if result is not None:
                print(f"{len(result)} cycles")
                all_frames.append(result)
            else:
                print("no valid data found")
        # else: silently skip missing cells

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True)
        print(f"\n[Data Loader] Loaded {len(df):,} total cycles from {len(all_frames)} CALCE cells")
    else:
        df = _generate_synthetic_fallback(output_dir)

    # Save combined dataset
    out_path = os.path.join(output_dir, "calce_battery_data.csv")
    df.to_csv(out_path, index=False)
    print(f"[Data Loader] Saved → {out_path}")

    # Print tier distribution (last cycle per battery)
    last = df.groupby("battery_id").last()
    dist = last["health_tier"].value_counts().to_dict()
    print(f"  Tier distribution (last cycle): {dist}")

    return df


if __name__ == "__main__":
    load_calce_data()
