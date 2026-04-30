"""
Classification Pipeline for Battery Health Tier Prediction
============================================================
Trains XGBoost and Logistic Regression classifiers with hyperparameter
tuning, evaluates performance, and logs everything to MLflow.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "capacity_fade_rate",
    "resistance_growth_rate",
    "average_charge_discharge_ratio",
    "voltage_delta_trend",
    "cycle_count",
    "current_discharge_capacity",
    "current_SOH",
    "capacity_variance_last_10_cycles",
    "energy_efficiency",
]

TIER_ORDER = ["Good", "Moderate", "Weak"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true, y_pred, display_labels, title, filepath):
    """Save a confusion matrix plot to disk."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    return cm


def _save_feature_importance(model, feature_names, filepath):
    """Save a horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], color="#58a6ff")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def _log_classification_metrics(y_true, y_pred, labels, prefix=""):
    """Log per-class and overall metrics to MLflow."""
    acc = accuracy_score(y_true, y_pred)
    mlflow.log_metric(f"{prefix}accuracy", acc)

    for avg in ["macro", "weighted"]:
        mlflow.log_metric(f"{prefix}precision_{avg}", precision_score(y_true, y_pred, average=avg, zero_division=0))
        mlflow.log_metric(f"{prefix}recall_{avg}", recall_score(y_true, y_pred, average=avg, zero_division=0))
        mlflow.log_metric(f"{prefix}f1_{avg}", f1_score(y_true, y_pred, average=avg, zero_division=0))

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    for cls_name in labels:
        if cls_name in report:
            for metric_name in ["precision", "recall", "f1-score"]:
                safe_cls = cls_name.replace("-", "_").replace(" ", "_")
                mlflow.log_metric(f"{prefix}{safe_cls}_{metric_name.replace('-', '_')}", report[cls_name][metric_name])

    return acc, report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifiers(
    features_path: str = "data/engineered_features.csv",
    models_dir: str = "models",
    mlruns_dir: str = "mlruns",
) -> dict:
    """Train XGBoost and Logistic Regression, log to MLflow, save artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(features_path)
    print(f"[Classification] Loaded {len(df)} samples from {features_path}")

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    le.fit(TIER_ORDER)
    y = le.transform(df["health_tier"].values)
    labels = list(le.classes_)
    groups = df["battery_id"].values

    # --- Group-aware split (all cycles from one battery stay together) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Train batteries: {sorted(set(groups[train_idx]))}")
    print(f"  Test batteries:  {sorted(set(groups[test_idx]))}")

    # --- MLflow setup ---
    mlflow.set_tracking_uri(mlruns_dir)
    mlflow.set_experiment("Battery_SOH_Prediction")

    results = {}

    # =====================================================================
    # 1) Logistic Regression (baseline)
    # =====================================================================
    print("\n--- Logistic Regression (baseline) ---")
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("C", 1.0)

        lr_acc, lr_report = _log_classification_metrics(y_test, y_pred_lr, labels, prefix="lr_")

        # Confusion matrix
        cm_path = os.path.join(models_dir, "lr_confusion_matrix.png")
        _save_confusion_matrix(y_test, y_pred_lr, labels, "Logistic Regression — Confusion Matrix", cm_path)
        mlflow.log_artifact(cm_path)

        # Save model
        lr_model_path = os.path.join(models_dir, "logistic_regression.joblib")
        joblib.dump({"model": lr, "scaler": scaler, "label_encoder": le}, lr_model_path)
        mlflow.log_artifact(lr_model_path)
        mlflow.sklearn.log_model(lr, "lr_model")

        print(f"  Accuracy: {lr_acc:.4f}")
        print(classification_report(y_test, y_pred_lr, target_names=labels, zero_division=0))
        results["logistic_regression"] = {"accuracy": lr_acc, "report": lr_report}

    # =====================================================================
    # 2) XGBoost with RandomizedSearchCV
    # =====================================================================
    print("\n--- XGBoost with Hyperparameter Tuning ---")
    with mlflow.start_run(run_name="XGBoost_Tuned"):
        xgb_base = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        param_distributions = {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300, 500],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        }

        search = RandomizedSearchCV(
            xgb_base,
            param_distributions=param_distributions,
            n_iter=40,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_xgb = search.best_estimator_
        y_pred_xgb = best_xgb.predict(X_test)

        # Log all best params
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 40)
        mlflow.log_param("cv_folds", 5)
        for k, v in search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_score", search.best_score_)

        xgb_acc, xgb_report = _log_classification_metrics(y_test, y_pred_xgb, labels, prefix="xgb_")

        # Confusion matrix
        cm_path = os.path.join(models_dir, "xgb_confusion_matrix.png")
        xgb_cm = _save_confusion_matrix(y_test, y_pred_xgb, labels, "XGBoost — Confusion Matrix", cm_path)
        mlflow.log_artifact(cm_path)

        # Feature importance
        fi_path = os.path.join(models_dir, "xgb_feature_importance.png")
        _save_feature_importance(best_xgb, FEATURE_COLS, fi_path)
        mlflow.log_artifact(fi_path)

        # Save model
        xgb_model_path = os.path.join(models_dir, "xgboost_model.joblib")
        joblib.dump({"model": best_xgb, "label_encoder": le, "feature_cols": FEATURE_COLS}, xgb_model_path)
        mlflow.log_artifact(xgb_model_path)
        mlflow.xgboost.log_model(best_xgb, "xgb_model")

        print(f"  Best CV F1 (macro): {search.best_score_:.4f}")
        print(f"  Test Accuracy: {xgb_acc:.4f}")
        print(f"  Best Params: {search.best_params_}")
        print(classification_report(y_test, y_pred_xgb, target_names=labels, zero_division=0))
        results["xgboost"] = {
            "accuracy": xgb_acc,
            "report": xgb_report,
            "best_params": search.best_params_,
            "confusion_matrix": xgb_cm.tolist(),
            "feature_importances": dict(zip(FEATURE_COLS, best_xgb.feature_importances_.tolist())),
        }

    # --- Save predictions for dashboard ---
    # Predict on ALL data for the drilldown table
    y_all_pred = le.inverse_transform(best_xgb.predict(df[FEATURE_COLS].values))
    df["predicted_tier"] = y_all_pred
    pred_path = os.path.join("data", "predictions.csv")
    df.to_csv(pred_path, index=False)
    print(f"\n[Classification] Predictions saved → {pred_path}")

    # Save confusion matrix data as JSON for dashboard
    cm_data = {
        "labels": labels,
        "xgb_confusion_matrix": xgb_cm.tolist(),
        "xgb_accuracy": xgb_acc,
        "lr_accuracy": lr_acc,
        "feature_importances": results["xgboost"]["feature_importances"],
    }
    cm_json_path = os.path.join("data", "classification_results.json")
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[Classification] Results JSON saved → {cm_json_path}")

    return results


if __name__ == "__main__":
    train_classifiers()
