# ⚡ Battery State-of-Health (SOH) Prediction Pipeline

A complete end-to-end machine learning pipeline that predicts lithium-ion battery health using synthetic cycle data. The project generates realistic battery degradation data, engineers predictive features, trains classification and clustering models, and presents everything through a polished interactive dashboard.

---

## 🚀 Quick Start

```bash
cd battery_soh_project

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (data → features → models → clusters)
python run_pipeline.py

# Launch the interactive dashboard
python src/dashboard.py
# → Open http://localhost:8050

# (Optional) View MLflow experiment logs
mlflow ui --backend-store-uri mlruns
# → Open http://localhost:5000
```

---

## 📊 Understanding the Project Through the Dashboard

The dashboard is the central window into the entire pipeline. Each panel answers a specific question about the battery fleet. Here's a visual walkthrough of what each section shows, why it matters, and what's happening behind the scenes.

### Panel 1 — Fleet Overview KPIs & SOH Distribution

![Fleet KPIs, SOH histogram, and feature importance](docs/screenshots/01_kpis_and_overview.png)

This top section gives you the **executive summary** of the entire fleet at a glance.

#### KPI Cards (Top Row)

| Card | What It Shows | Why It Matters |
|---|---|---|
| **Total Batteries** | The fleet size (200 batteries) | Sets the scale of the analysis |
| **Healthy** (green) | Count and percentage of batteries with SOH ≥ 80% | These batteries are operating normally — no action needed |
| **Degraded** (amber) | Count and percentage with SOH between 60%–80% | ⚠️ These need monitoring — approaching replacement threshold |
| **End-of-Life** (red) | Count and percentage with SOH < 60% | 🔴 Critical — these should be replaced immediately |
| **Avg SOH** | Fleet-wide average State-of-Health | A single number to judge overall fleet condition |
| **Gauge** | Visual indicator of fleet health | Quick pass/fail visual — green = fleet is mostly healthy |

> **Behind the scenes:** These numbers come from `data_generation.py` (200 synthetic batteries with realistic degradation curves) and `feature_engineering.py` (which extracts the latest SOH for each battery and assigns a health tier).

#### SOH Distribution Histogram (Bottom Left)

The histogram shows **how SOH values are distributed across all 200 batteries**. Each bar is color-coded:
- 🟢 **Green** = Healthy (SOH ≥ 80%)
- 🟡 **Amber** = Degraded (60% ≤ SOH < 80%)
- 🔴 **Red** = End-of-Life (SOH < 60%)

**What to look for:** A healthy fleet will have most bars clustered near 100%. A long left tail with red/amber bars indicates aging batteries that need attention. This chart helps fleet managers understand the *shape* of degradation across the fleet — is it concentrated in a few bad batteries, or is the whole fleet slowly aging?

#### XGBoost Feature Importances (Bottom Right)

This horizontal bar chart shows **which battery measurements matter most for predicting health tier**. The longer the bar, the more that feature contributed to the XGBoost model's decisions.

In our results:
- **`current_discharge_capacity`** is the #1 predictor — the amount of charge a battery can deliver is the most direct indicator of health
- **`current_SOH`** and **`average_charge_discharge_ratio`** are strong signals — the ratio between charge-in vs. charge-out reveals coulombic efficiency loss
- **`cycle_count`**, **`resistance_growth_rate`** contribute less — age alone isn't as predictive as actual capacity measurements

> **Behind the scenes:** These importances are extracted from the trained XGBoost classifier in `train_classifiers.py`. The model was tuned with RandomizedSearchCV (40 iterations, 5-fold CV) and achieved **97.5% accuracy**.

---

### Panel 2 — Degradation Curves

![Discharge capacity degradation curves with multi-battery selection](docs/screenshots/02_degradation_curves.png)

This panel is the **lifecycle view** — it shows how each battery's discharge capacity decreases over hundreds of charge/discharge cycles.

#### What You're Seeing

- **X-axis**: Cycle number (how many times the battery has been charged/discharged)
- **Y-axis**: Discharge capacity in Amp-hours (Ah) — how much energy the battery can deliver per cycle
- **Each line**: One battery's degradation trajectory
- **Dashed lines**: Average trendlines per health tier (green dashed = avg of Healthy batteries, amber dashed = avg of Degraded, red dashed = avg of End-of-Life)

#### How to Read It

- A **flat green line near 3.0 Ah** = a healthy battery holding its capacity well
- A **steeply declining red line** = a battery that has lost significant capacity and is near end-of-life
- The **gap between individual lines and their tier's average** tells you if a specific battery is degrading faster or slower than peers in its health class

#### Interactivity

- Use the **dropdown** at the top to select which batteries to compare (you can select multiple)
- **Click a row in the Battery Drilldown table** (below) to add that battery to this chart with a highlighted, thicker line

> **Behind the scenes:** The raw cycle data comes from `data_generation.py`, which simulates each battery using a non-linear degradation model: `SOH = 100 - α·cycle^β + noise`. Different batteries get different α and β values, creating realistic variation in degradation rates.

---

### Panel 3 — Cluster Visualization & Confusion Matrix

![PCA cluster scatter plot and XGBoost confusion matrix heatmap](docs/screenshots/03_clusters_confusion.png)

This section combines **unsupervised learning** (K-Means clustering) with **supervised learning** (XGBoost classification) to give a complete picture of battery health patterns.

#### Cluster Visualization (Left) — *"What patterns does the data naturally form?"*

This is a **2D PCA scatter plot** where each dot represents one battery. The 9 engineered features (capacity fade rate, resistance growth, energy efficiency, etc.) are compressed into 2 dimensions using Principal Component Analysis so we can visualize them.

- **Colors = K-Means cluster assignments** (Cluster 0, 1, 2)
- Toggle the radio button to **"Actual Health Tier"** to see the same dots colored by their true health labels — this lets you visually compare how well the unsupervised clusters match reality

**What to look for:**
- The tight blue cluster on the right = healthy batteries (high capacity, low resistance)
- The spread-out amber dots in the middle-left = degraded batteries
- The two red outlier dots at top-left = extreme end-of-life batteries

> **Behind the scenes:** `clustering.py` runs K-Means with k=3 on the scaled feature matrix and reports an **Adjusted Rand Index of 0.57**, meaning the clusters have meaningful (though imperfect) alignment with the true health tiers. The imperfection makes sense — unsupervised clustering doesn't have access to labels, yet still discovers the same degradation patterns.

#### Confusion Matrix Heatmap (Right) — *"How accurate are the model's predictions?"*

This is the **XGBoost model's report card**. Each cell shows:
- **Count**: How many batteries fell into that prediction vs actual category
- **Percentage**: What fraction of that actual class was correctly identified

Reading the matrix:
- **Diagonal cells** (top-left to bottom-right) = correct predictions ✅
- **Off-diagonal cells** = misclassifications ❌

Our results show:
- **Healthy**: 30/30 correct (100%) — the model never mistakes a healthy battery
- **End-of-Life**: 3/3 correct (100%) — critical batteries are always caught
- **Degraded**: 6/7 correct (85.7%) — one degraded battery was misclassified as End-of-Life (a conservative error — it's better to flag a degraded battery as worse than it is, rather than miss it)

Overall accuracy: **97.5%** (vs. 92.5% for the Logistic Regression baseline).

> **Behind the scenes:** `train_classifiers.py` trains both models, logs everything to MLflow (parameters, metrics, artifacts), and saves the confusion matrix data for the dashboard.

---

### Panel 4 — Battery Drilldown Table & Degradation Forecast

![Battery drilldown table with risk flags and SOH forecast chart](docs/screenshots/04_table_forecast.png)

This is the **operations panel** — where you'd go to investigate individual batteries and plan maintenance.

#### Battery Drilldown Table (Top)

A **searchable, sortable, filterable table** of every battery in the fleet with:

| Column | What It Tells You |
|---|---|
| **Battery Id** | Unique identifier |
| **Current SOH** | Latest state-of-health percentage — the battery's "health score" |
| **Predicted Tier** | What the XGBoost model *thinks* the health tier is |
| **Health Tier** | The actual (ground-truth) health tier |
| **Cycle Count** | Total charge/discharge cycles completed |
| **Capacity Fade Rate** | How fast capacity is declining per cycle (more negative = worse) |
| **Risk Flag** | Automated alert: 🔴 **Critical** (SOH < 65%), ⚠️ **Rapid Fade** (steep decline), ✅ **Stable** |

**Highlighted rows in red** = End-of-Life batteries that need immediate attention.

You can:
- **Filter** any column using the filter boxes below the headers
- **Sort** by clicking column headers (e.g., sort by SOH ascending to see worst batteries first)
- **Click a radio button** on any row to select it — this updates both the degradation curves chart AND the forecast chart below

#### Degradation Trend Forecast (Bottom) — *"When will this battery die?"*

When you select a battery from the table, this chart shows:
- **Solid line**: Historical SOH over the battery's actual cycles
- **Dashed gray line**: Linear extrapolation projecting future SOH
- **Red dotted line at 60%**: The End-of-Life threshold
- **Amber dotted line at 80%**: The Degraded threshold
- **Title annotation**: **Estimated RUL (Remaining Useful Life)** — how many more cycles until SOH drops to 60%

In the screenshot, BAT_0002 shows:
- SOH started near 100% and has declined steadily over ~410 cycles
- It crossed the 80% degraded threshold around cycle 250
- It crossed the 60% end-of-life threshold around cycle 350
- The forecast shows **~0 cycles remaining** — this battery has already reached end-of-life

> **Behind the scenes:** The forecast uses a simple linear regression fit on the battery's SOH history to project forward. While real-world forecasting would use more sophisticated models (e.g., Kalman filters, LSTMs), linear extrapolation gives a practical first-order estimate of remaining useful life.

---

## 🏗️ Project Architecture

```
battery_soh_project/
├── data/                          # Generated data files
│   ├── synthetic_battery_data.csv # 60,704 rows × 12 columns (raw cycle data)
│   ├── engineered_features.csv    # 200 rows × 11 columns (one per battery)
│   ├── predictions.csv            # Features + XGBoost predictions
│   ├── cluster_results.csv        # Features + cluster assignments + PCA coords
│   ├── classification_results.json
│   └── cluster_metadata.json
├── models/                        # Saved model artifacts
│   ├── xgboost_model.joblib       # Trained XGBoost classifier
│   ├── logistic_regression.joblib # Trained LogReg baseline
│   └── *.png                      # Confusion matrix & feature importance plots
├── src/
│   ├── data_generation.py         # Synthetic battery data generator
│   ├── feature_engineering.py     # Per-battery feature extraction
│   ├── train_classifiers.py       # XGBoost + LogReg + MLflow logging
│   ├── clustering.py              # K-Means + PCA + MLflow logging
│   └── dashboard.py               # Interactive Plotly Dash dashboard
├── docs/screenshots/              # Dashboard screenshots for this README
├── mlruns/                        # MLflow experiment tracking data
├── run_pipeline.py                # One-command pipeline orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Pipeline Flow

```
data_generation.py ──→ synthetic_battery_data.csv (60K rows)
        │
        ▼
feature_engineering.py ──→ engineered_features.csv (200 rows)
        │
        ├──→ train_classifiers.py ──→ XGBoost (97.5%) + LogReg (92.5%)
        │                            └──→ MLflow logging
        │
        └──→ clustering.py ──→ K-Means (k=3, ARI=0.57)
                               └──→ MLflow logging
        │
        ▼
dashboard.py ──→ Interactive visualization of all results
```

---

## 🔬 Model Results Summary

| Model | Accuracy | F1 (macro) | Purpose |
|---|---|---|---|
| **XGBoost** | **97.5%** | 0.93 | Primary classifier — predicts health tier |
| Logistic Regression | 92.5% | 0.92 | Baseline comparison |
| K-Means (k=3) | ARI = 0.57 | — | Unsupervised pattern discovery |

### Health Tier Definitions

| Tier | SOH Range | Action |
|---|---|---|
| 🟢 Healthy | ≥ 80% | No action needed |
| 🟡 Degraded | 60% – 79% | Schedule monitoring and eventual replacement |
| 🔴 End-of-Life | < 60% | Replace immediately |

---

## ⚙️ Running Individual Components

```bash
# Generate synthetic battery data only
python src/data_generation.py

# Run feature engineering only (requires data to exist)
python src/feature_engineering.py

# Train classifiers only (requires features to exist)
python src/train_classifiers.py

# Run clustering only (requires features to exist)
python src/clustering.py

# Launch dashboard only (requires all pipeline outputs)
python src/dashboard.py
```

---

## 📦 Dependencies

- Python 3.10+
- numpy, pandas — data manipulation
- scikit-learn — preprocessing, clustering, metrics
- xgboost — gradient boosting classifier
- mlflow — experiment tracking
- plotly, dash, dash-bootstrap-components — interactive dashboard
- matplotlib — static plots for MLflow artifacts
