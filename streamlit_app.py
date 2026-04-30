"""
Battery SOH Dashboard — Streamlit version
==========================================
Single-page light-theme dashboard for CALCE CS2 battery health analysis.
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery SOH Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C = {
    "bg":      "#f8f9fa",
    "surface": "#ffffff",
    "border":  "#e2e8f0",
    "text":    "#1e293b",
    "muted":   "#64748b",
    "good":    "#16a34a",
    "moderate":"#d97706",
    "weak":    "#dc2626",
    "accent":  "#2563eb",
    "light_accent": "#eff6ff",
}

TIER_COLORS = {"Good": C["good"], "Moderate": C["moderate"], "Weak": C["weak"]}
TIER_ORDER  = ["Good", "Moderate", "Weak"]

CHART = dict(
    template="plotly_white",
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, system-ui, sans-serif", size=12),
)
DEFAULT_MARGIN = dict(l=50, r=30, t=40, b=40)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
h1, h2, h3, h4, h5, h6 { color: #1e293b; }
div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 16px 20px; box-shadow: none;
}
div[data-testid="stMetric"] label { font-size: 0.72rem; font-weight: 600;
    color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.75rem; font-weight: 700; }
.section-label {
    font-weight: 700; font-size: 1rem; color: #1e293b;
    margin-bottom: 12px; margin-top: 24px;
}
.panel-desc {
    color: #64748b; font-size: 0.82rem; line-height: 1.55; margin-bottom: 16px;
}
.stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    raw      = pd.read_csv("data/calce_battery_data.csv")
    features = pd.read_csv("data/engineered_features.csv")
    preds    = pd.read_csv("data/predictions.csv")
    clusters = pd.read_csv("data/cluster_results.csv")
    with open("data/classification_results.json") as f:
        cls = json.load(f)
    with open("data/cluster_metadata.json") as f:
        meta = json.load(f)
    return raw, features, preds, clusters, cls, meta

raw, features, preds, clusters, cls, meta = load_data()

battery_ids = sorted(raw["battery_id"].unique())

# Per-battery snapshot (latest cycle)
snap = preds.sort_values("cycle_number").groupby("battery_id").last().reset_index()
snap["risk_flag"] = snap.apply(
    lambda r: "Critical" if r["current_SOH"] < 80
    else ("Rapid Fade" if r["capacity_fade_rate"] < -0.005 else "Stable"),
    axis=1,
)
total       = len(snap)
tier_counts = snap["health_tier"].value_counts()
avg_soh     = snap["current_SOH"].mean()
soh_color   = C["good"] if avg_soh >= 90 else (C["moderate"] if avg_soh >= 80 else C["weak"])

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
muted = C["muted"]
st.markdown(
    f"<h2 style='margin-bottom:0'>Battery Fleet Health</h2>"
    f"<span style='font-size:0.82rem; color:{muted}'>CALCE CS2  |  6 cells  |  5,515 cycles</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Section 1: At a Glance
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">At a Glance</div>', unsafe_allow_html=True)

cols = st.columns(6)
with cols[0]:
    st.metric("Fleet Size", total, help="Total batteries tracked")
with cols[1]:
    good_n = tier_counts.get("Good", 0)
    st.metric("Good", good_n, help="SOH >= 90%")
with cols[2]:
    mod_n = tier_counts.get("Moderate", 0)
    st.metric("Moderate", mod_n, help="SOH 80-90%")
with cols[3]:
    weak_n = tier_counts.get("Weak", 0)
    st.metric("Weak", weak_n, help="SOH < 80%")
with cols[4]:
    st.metric("Avg SOH", f"{avg_soh:.1f}%", help="Fleet-wide average")
with cols[5]:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_soh,
        number={"suffix": "%", "font": {"size": 28, "color": soh_color, "weight": 700}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=C["muted"]),
            bar=dict(color=soh_color, thickness=0.25),
            bgcolor="#f1f5f9",
            bordercolor=C["border"],
            steps=[
                dict(range=[0, 80],  color="#fee2e2"),
                dict(range=[80, 90], color="#fef3c7"),
                dict(range=[90, 100],color="#dcfce7"),
            ],
        ),
    ))
    fig_gauge.update_layout(
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"]), height=160,
        margin=dict(l=30, r=30, t=10, b=0),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Section 2: SOH Distribution
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">SOH Distribution</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-desc">This histogram counts every individual charge/discharge cycle, '
    "not just the final state of each battery. Early-life cycles cluster near 100% (green), "
    "while later cycles shift left as batteries degrade. Every battery starts healthy, so the "
    "large green region represents the first several hundred cycles of each cell's life.</div>",
    unsafe_allow_html=True,
)

fig_hist = px.histogram(
    features, x="current_SOH", color="health_tier",
    color_discrete_map=TIER_COLORS, nbins=35,
    category_orders={"health_tier": TIER_ORDER},
    labels={"current_SOH": "SOH (%)", "health_tier": "Tier"},
)
fig_hist.update_layout(
    **CHART, title=None, height=300, bargap=0.06,
    margin=DEFAULT_MARGIN,
    legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    yaxis_title="Cycle count",
)
fig_hist.update_traces(marker_line_width=0)
st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Section 3: Battery Explorer
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Battery Explorer</div>', unsafe_allow_html=True)

col_table, col_charts = st.columns([5, 7])

with col_table:
    st.markdown("**Status table**")
    st.markdown(
        '<div class="panel-desc">Each row shows one battery at its most recent cycle. '
        "Click a battery ID in the table to load its degradation history and forecast.</div>",
        unsafe_allow_html=True,
    )

    TABLE_COLS = [
        "battery_id", "current_SOH", "predicted_tier",
        "health_tier", "cycle_count", "capacity_fade_rate", "risk_flag",
    ]
    display_df = snap[TABLE_COLS].copy()
    display_df.columns = [c.replace("_", " ").title() for c in TABLE_COLS]
    display_df = display_df.round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    selected_battery = st.selectbox(
        "Select battery for detail view",
        options=battery_ids,
        index=0,
    )

with col_charts:
    # Degradation curves
    st.markdown("**Discharge capacity over time**")
    st.markdown(
        '<div class="panel-desc">Each line traces a battery\'s capacity per cycle. '
        "A flat line means the battery is holding up; a steep drop means rapid aging.</div>",
        unsafe_allow_html=True,
    )
    selected_batteries = st.multiselect(
        "Batteries to compare",
        options=battery_ids,
        default=battery_ids[:3],
    )

    if selected_batteries:
        fig_deg = go.Figure()
        for bid in selected_batteries:
            bdata = raw[raw["battery_id"] == bid].sort_values("cycle_number")
            tier  = bdata["health_tier"].iloc[-1]
            color = TIER_COLORS.get(tier, C["accent"])
            smoothed = bdata["discharge_capacity_Ah"].rolling(
                window=10, min_periods=1, center=True
            ).mean()
            fig_deg.add_trace(go.Scatter(
                x=bdata["cycle_number"], y=smoothed, mode="lines",
                name=f"{bid} ({tier})",
                line=dict(
                    color=color,
                    width=2.5 if bid == selected_battery else 1.5,
                ),
                opacity=1.0 if bid == selected_battery else 0.6,
            ))
        fig_deg.update_layout(
            **CHART, title=None, height=300,
            xaxis_title="Cycle", yaxis_title="Capacity (Ah)",
            legend=dict(font=dict(size=10), orientation="h", y=-0.25),
            margin=dict(l=50, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_deg, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Select at least one battery above.")

    # Forecast
    st.markdown("**Remaining useful life forecast**")
    st.markdown(
        '<div class="panel-desc">SOH history and a linear projection of when the selected '
        "battery will cross the 80% (Weak) threshold.</div>",
        unsafe_allow_html=True,
    )

    bdata  = raw[raw["battery_id"] == selected_battery].sort_values("cycle_number")
    cycles = bdata["cycle_number"].values
    soh    = bdata["SOH_percent"].values
    tier   = bdata["health_tier"].iloc[-1]

    slope, intercept = np.polyfit(cycles, soh, 1)
    cur_cycle = cycles[-1]
    if slope < 0:
        rul_cycle = (80.0 - intercept) / slope
        remaining = max(0, rul_cycle - cur_cycle)
    else:
        rul_cycle = cur_cycle + 500
        remaining = float("inf")
    proj_end    = min(int(rul_cycle) + 50, cur_cycle + 600)
    proj_cycles = np.arange(cur_cycle, proj_end)
    proj_soh    = slope * proj_cycles + intercept
    rul_label   = f"~{remaining:.0f} cycles left" if remaining != float("inf") else "stable"

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=cycles, y=soh, mode="lines", name="Actual SOH",
        line=dict(color=TIER_COLORS.get(tier, C["accent"]), width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=proj_cycles, y=proj_soh, mode="lines", name="Forecast",
        line=dict(color=C["muted"], width=1.5, dash="dash"),
    ))
    fig_fc.add_hline(y=80, line_dash="dot", line_color=C["weak"], opacity=0.6,
                      annotation_text="Weak (80%)", annotation_font_color=C["weak"],
                      annotation_font_size=11)
    fig_fc.add_hline(y=90, line_dash="dot", line_color=C["moderate"], opacity=0.4,
                      annotation_text="Moderate (90%)", annotation_font_color=C["moderate"],
                      annotation_font_size=11)
    fig_fc.update_layout(
        **CHART, title=None, height=260,
        xaxis_title="Cycle", yaxis_title="SOH (%)",
        yaxis_range=[0, 112],
        legend=dict(font=dict(size=10), orientation="h", y=-0.35),
        margin=dict(l=50, r=20, t=10, b=55),
        annotations=[dict(
            text=f"{selected_battery}  |  {rul_label}",
            x=0.01, y=0.97, xref="paper", yref="paper",
            showarrow=False,
            font=dict(color=C["text"], size=11, family="Inter"),
            bgcolor="rgba(241,245,249,0.9)",
            borderpad=4,
        )],
    )
    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Section 4: Model Performance
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

# Feature importance
st.markdown("**Feature importance (XGBoost)**")
st.markdown(
    '<div class="panel-desc">Which measurements matter most when the classifier decides a '
    "battery's health tier. Longer bars indicate stronger predictors. Discharge capacity and "
    "current SOH dominate because they directly measure how much energy the cell can still deliver.</div>",
    unsafe_allow_html=True,
)

fi = cls.get("feature_importances", {})
fi_sorted = sorted(fi.items(), key=lambda x: x[1])
fi_names  = [x[0].replace("_", " ").title() for x in fi_sorted]
fi_vals   = [x[1] for x in fi_sorted]

fig_fi = go.Figure(go.Bar(
    x=fi_vals, y=fi_names, orientation="h",
    marker_color=C["accent"], marker_line_width=0,
    text=[f"{v:.3f}" for v in fi_vals],
    textposition="outside",
    textfont=dict(size=11, color=C["muted"]),
))
fig_fi.update_layout(
    **CHART, title=None, height=320,
    margin=dict(l=180, r=50, t=10, b=40),
)
st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

# Confusion matrix + Clusters
col_cm, col_cl = st.columns(2)

with col_cm:
    st.markdown(f"**Confusion matrix  --  test accuracy {cls['xgb_accuracy']:.1%}**")
    st.markdown(
        '<div class="panel-desc">Evaluated on held-out batteries CS2_33 and CS2_34 '
        "(the model never trained on these). Each cell counts cycle-level predictions. "
        "The 828 'Good' predictions are the early-life cycles when SOH was above 90%.</div>",
        unsafe_allow_html=True,
    )

    cm        = np.array(cls["xgb_confusion_matrix"])
    cm_labels = cls["labels"]
    cm_pct    = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
    cm_text   = [[f"{cm[i][j]}  ({cm_pct[i][j]}%)"
                  for j in range(len(cm_labels))]
                 for i in range(len(cm_labels))]

    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=cm_labels, y=cm_labels,
        text=cm_text, texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[[0, "#f0f4ff"], [1, C["accent"]]],
        showscale=False, xgap=3, ygap=3,
    ))
    fig_cm.update_layout(
        **CHART, title=None, height=320,
        margin=dict(l=80, r=30, t=10, b=60),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

with col_cl:
    ari = meta.get("adjusted_rand_index", 0)
    st.markdown(f"**Unsupervised clusters  --  ARI {ari:.2f}**")
    st.markdown(
        '<div class="panel-desc">K-Means (k=3) groups cycle measurements by feature similarity '
        "without seeing the health labels. Toggle between cluster assignments and actual tiers "
        f"to compare. An ARI of {ari:.2f} means partial but meaningful alignment.</div>",
        unsafe_allow_html=True,
    )

    color_by = st.radio(
        "Colour by",
        options=["K-Means cluster", "Actual tier"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if color_by == "K-Means cluster":
        fig_cl = px.scatter(
            clusters, x="pca_x", y="pca_y",
            color=clusters["cluster"].astype(str),
            hover_data=["battery_id", "current_SOH", "health_tier"],
            color_discrete_sequence=[C["accent"], C["moderate"], C["weak"]],
            labels={"pca_x": "PC 1", "pca_y": "PC 2", "color": "Cluster"},
        )
    else:
        fig_cl = px.scatter(
            clusters, x="pca_x", y="pca_y",
            color="health_tier",
            color_discrete_map=TIER_COLORS,
            category_orders={"health_tier": TIER_ORDER},
            hover_data=["battery_id", "current_SOH", "cluster"],
            labels={"pca_x": "PC 1", "pca_y": "PC 2"},
        )
    fig_cl.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0)))
    fig_cl.update_layout(
        **CHART, title=None, height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_cl, use_container_width=True, config={"displayModeBar": False})
