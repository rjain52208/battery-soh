#!/usr/bin/env python3
"""
Battery SOH Prediction Pipeline — Orchestrator
=================================================
Runs the full pipeline: data generation → feature engineering →
classification training → clustering. All results are logged to MLflow.
"""

import sys
import os
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.data_loader import load_calce_data
from src.feature_engineering import run_feature_engineering
from src.train_classifiers import train_classifiers
from src.clustering import run_clustering


def main():
    print("=" * 70)
    print("  ⚡  Battery SOH Prediction Pipeline")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Load CALCE battery data (falls back to synthetic if no raw files)
    print("\n" + "─" * 50)
    print("  Step 1/4: Load CALCE Data")
    print("─" * 50)
    load_calce_data(raw_dir="data/raw", output_dir="data")

    # Step 2: Feature engineering
    print("\n" + "─" * 50)
    print("  Step 2/4: Feature Engineering")
    print("─" * 50)
    run_feature_engineering(input_path="data/calce_battery_data.csv", output_dir="data")

    # Step 3: Train classifiers
    print("\n" + "─" * 50)
    print("  Step 3/4: Classification (XGBoost + Logistic Regression)")
    print("─" * 50)
    results = train_classifiers(
        features_path="data/engineered_features.csv",
        models_dir="models",
        mlruns_dir="mlruns",
    )

    # Step 4: Clustering
    print("\n" + "─" * 50)
    print("  Step 4/4: K-Means Clustering")
    print("─" * 50)
    run_clustering(
        features_path="data/engineered_features.csv",
        output_dir="data",
        models_dir="models",
        mlruns_dir="mlruns",
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ✅  Pipeline complete in {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  XGBoost accuracy:  {results['xgboost']['accuracy']:.1%}")
    print(f"  LogReg accuracy:   {results['logistic_regression']['accuracy']:.1%}")
    print(f"\n  📊 Launch dashboard:   python src/dashboard.py")
    print(f"  📈 Launch MLflow UI:   mlflow ui --backend-store-uri mlruns")
    print(f"     Then open http://localhost:8050 (dashboard)")
    print(f"     and http://localhost:5000 (MLflow)")


if __name__ == "__main__":
    main()
