"""
Feature Engineering for Battery SOH Prediction
================================================
Computes per-cycle rolling-window features from raw cycle data,
producing one row per cycle for downstream classification and clustering.

Each cycle gets features computed from the preceding N cycles as context,
which allows training with thousands of samples instead of one-per-battery.
"""

import numpy as np
import pandas as pd
import os


WINDOW_SIZE = 20   # Rolling window of recent cycles for trend features
VARIANCE_WINDOW = 10
MIN_HISTORY = 10   # Minimum cycles of history before generating features


def _slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Compute the OLS slope of y vs x (default: integer indices)."""
    if len(y) < 2:
        return 0.0
    if x is None:
        x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _engineer_per_cycle_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window features for each cycle of a single battery.

    Returns one row per cycle (after MIN_HISTORY warmup), with the feature
    context window drawn from the preceding WINDOW_SIZE cycles.
    """
    group = group.sort_values("cycle_number").reset_index(drop=True)
    records = []

    for i in range(MIN_HISTORY, len(group)):
        window_start = max(0, i - WINDOW_SIZE)
        window = group.iloc[window_start:i + 1]
        var_window = group.iloc[max(0, i - VARIANCE_WINDOW):i + 1]
        current = group.iloc[i]

        cycles_w = window["cycle_number"].values.astype(float)
        discharge_w = window["discharge_capacity_Ah"].values
        charge_w = window["charge_capacity_Ah"].values

        # Resistance features (if available)
        if "internal_resistance_ohms" in window.columns:
            resistance_w = window["internal_resistance_ohms"].values
            resistance_growth_rate = _slope(resistance_w, cycles_w) if (resistance_w > 0).any() else 0.0
        else:
            resistance_growth_rate = 0.0

        # Voltage features
        v_max_w = window["charge_voltage_max"].values
        v_min_w = window["discharge_voltage_min"].values
        voltage_delta = v_max_w - v_min_w
        voltage_delta_trend = _slope(voltage_delta, cycles_w)

        # Core features
        capacity_fade_rate = _slope(discharge_w, cycles_w)
        avg_charge_discharge_ratio = float(np.mean(charge_w / np.clip(discharge_w, 1e-6, None)))
        cycle_count = int(current["cycle_number"])
        current_discharge_capacity = float(current["discharge_capacity_Ah"])
        current_SOH = float(current["SOH_percent"])
        capacity_variance_last_10 = float(var_window["discharge_capacity_Ah"].var())

        # Energy efficiency
        discharge_energy = discharge_w * v_min_w
        charge_energy = charge_w * v_max_w
        energy_efficiency = float(np.mean(discharge_energy / np.clip(charge_energy, 1e-6, None)))

        records.append({
            "battery_id": current["battery_id"],
            "cycle_number": cycle_count,
            "capacity_fade_rate": capacity_fade_rate,
            "resistance_growth_rate": resistance_growth_rate,
            "average_charge_discharge_ratio": avg_charge_discharge_ratio,
            "voltage_delta_trend": voltage_delta_trend,
            "cycle_count": cycle_count,
            "current_discharge_capacity": current_discharge_capacity,
            "current_SOH": current_SOH,
            "capacity_variance_last_10_cycles": capacity_variance_last_10,
            "energy_efficiency": energy_efficiency,
            "health_tier": current["health_tier"],
        })

    return pd.DataFrame(records)


def run_feature_engineering(
    input_path: str = "data/calce_battery_data.csv",
    output_dir: str = "data",
) -> pd.DataFrame:
    """Load raw cycle data, compute per-cycle features, and save to CSV."""
    df = pd.read_csv(input_path)
    print(f"[Feature Engineering] Loaded {len(df):,} rows from {input_path}")

    frames = []
    for _bid, group in df.groupby("battery_id"):
        feat = _engineer_per_cycle_features(group)
        if len(feat) > 0:
            frames.append(feat)

    features_df = pd.concat(frames, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "engineered_features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"[Feature Engineering] Saved {len(features_df):,} feature rows → {out_path}")
    print(f"  Health tier distribution: {features_df['health_tier'].value_counts().to_dict()}")
    return features_df


if __name__ == "__main__":
    run_feature_engineering()
