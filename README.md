# Battery State-of-Health (SOH) Prediction Pipeline

An end-to-end machine learning pipeline that predicts lithium-ion battery health tiers from real-world cycling data. The project ingests raw measurements from the CALCE CS2 battery dataset, engineers degradation-aware features, trains an XGBoost classifier achieving 99.5% accuracy, and surfaces results through an interactive Streamlit dashboard.

**Live dashboard:** [battery-soh.streamlit.app](https://battery-soh.streamlit.app)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (data loading, features, models, clusters)
python run_pipeline.py

# Launch the dashboard locally
streamlit run streamlit_app.py
```

---

## Dataset

**Source:** [CALCE Battery Research Group](https://calce.umd.edu/battery-data), University of Maryland

The CS2 dataset contains cycling data from six prismatic lithium-ion cells (CS2_33 through CS2_38) with a nominal capacity of 1.1 Ah. Cells were cycled between 753 and 1,051 times under controlled lab conditions until significant degradation occurred.

- **Total:** 5,515 charge/discharge cycles across 6 cells
- **Per cycle:** discharge capacity, charge capacity, voltage extremes, charge/discharge times, internal resistance
- **SOH:** computed as current capacity / initial capacity (average of first 3 cycles)

### Health Tier Definitions

| Tier | SOH Range | Meaning |
|---|---|---|
| Good | >= 90% | Operating normally |
| Moderate | 80 - 90% | Aging, needs monitoring |
| Weak | < 80% | Near end-of-life, replace |

**Fleet summary at final cycle:** 0 Good, 2 Moderate (CS2_33, CS2_35), 4 Weak (CS2_34, CS2_36, CS2_37, CS2_38)

---

## Dashboard

The Streamlit dashboard is organized into four sections:

### 1. At a Glance
Six KPI cards showing fleet size, count per health tier, fleet-average SOH (44.8%), and a color-coded gauge. The red gauge signals that most cells were cycled to near-failure in this aging study.

### 2. SOH Distribution
Histogram of SOH across all 5,515 cycle measurements (not just final states). The tall green bars near 100% represent early healthy life; the red left tail shows degradation.

### 3. Battery Explorer
- **Status table** showing each battery's latest-cycle snapshot with risk flags
- **Degradation curves** with 10-cycle rolling-mean smoothing and multi-battery comparison
- **RUL forecast** with linear projection to the 80% Weak threshold

### 4. Model Performance
- **Feature importance** bar chart — discharge capacity (0.486) and SOH (0.406) dominate
- **Confusion matrix** on held-out batteries CS2_33/CS2_34 (1,576 predictions, 8 errors)
- **PCA cluster visualization** with toggle between K-Means assignments and actual tiers

---

## Model Results

| Model | Metric | Value | Notes |
|---|---|---|---|
| XGBoost | Test Accuracy | 99.5% | 40-iter RandomizedSearchCV, 5-fold CV |
| XGBoost | F1 (macro) | 0.99 | Near-perfect per-class precision/recall |
| Logistic Regression | Test Accuracy | 94.7% | Baseline comparison |
| K-Means (k=3) | Adjusted Rand Index | 0.32 | Unsupervised, no label access |

**Train/test split:** Group-aware (GroupShuffleSplit) — trained on CS2_35/36/37/38, tested on CS2_33/34. No data leakage between batteries.

---

## Project Structure

```
battery_soh_project/
├── data/
│   ├── calce_battery_data.csv        # 5,515 rows - processed cycle data
│   ├── engineered_features.csv       # 5,455 rows - per-cycle features
│   ├── predictions.csv               # Features + model predictions
│   ├── cluster_results.csv           # Features + cluster assignments + PCA
│   ├── classification_results.json   # Model metrics and confusion matrix
│   └── cluster_metadata.json         # Cluster info and ARI score
├── src/
│   ├── data_loader.py                # CALCE data ingestion and SOH computation
│   ├── feature_engineering.py        # Rolling-window feature extraction
│   ├── train_classifiers.py          # XGBoost + LogReg with MLflow logging
│   └── clustering.py                 # K-Means + PCA with MLflow logging
├── models/                           # Saved plots (confusion matrix, etc.)
├── docs/
│   └── Battery_SOH_Report.pdf        # Detailed project report
├── .streamlit/config.toml            # Light theme config
├── streamlit_app.py                  # Interactive dashboard
├── run_pipeline.py                   # One-command pipeline orchestrator
└── requirements.txt
```

### Pipeline Flow

```
data_loader.py --> calce_battery_data.csv (5,515 rows)
       |
       v
feature_engineering.py --> engineered_features.csv (5,455 rows)
       |
       |---> train_classifiers.py --> XGBoost (99.5%) + LogReg (94.7%)
       |
       |---> clustering.py --> K-Means (k=3, ARI=0.32)
       |
       v
streamlit_app.py --> Interactive dashboard
```

---

## Feature Engineering

Nine features computed per cycle using rolling windows (window=10):

| Feature | Description |
|---|---|
| capacity_fade_rate | Rolling slope of discharge capacity |
| resistance_growth_rate | Rolling slope of internal resistance |
| charge_discharge_ratio | Charge capacity / discharge capacity |
| voltage_delta_trend | Rolling slope of voltage range |
| cycle_count | Cycle number |
| current_discharge_capacity | Raw discharge capacity (Ah) |
| current_SOH | SOH at this cycle (%) |
| capacity_variance | Rolling variance of discharge capacity |
| energy_efficiency | Discharge energy / charge energy |

---

## Tech Stack

- **Data:** Python, Pandas, NumPy
- **ML:** Scikit-learn, XGBoost
- **Tracking:** MLflow
- **Dashboard:** Streamlit, Plotly
- **Data Source:** CALCE CS2 (University of Maryland)
