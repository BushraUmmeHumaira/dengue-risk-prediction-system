
from pathlib import Path
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="DengueGuard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import build_features
from src.predict import predict_risk


# ==================================================
# DESIGN
# ==================================================


# ==================================================
# PREMIUM DARK DASHBOARD THEME
# ==================================================

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
 background:#0D1117 !important; color:#E6EDF3 !important;
}
.block-container { max-width:1360px; padding-top:2.2rem; padding-bottom:3rem; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
 background:#111820 !important; border-right:1px solid #27323E !important;
}
section[data-testid="stSidebar"] { min-width:275px; }
h1,h2,h3,h4,h5,h6 {color:#F0F6FC !important;letter-spacing:-.025em;}
p, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {color:#C4D0DC;}
[data-testid="stCaptionContainer"] p {color:#92A3B4 !important;}
[data-testid="stMetric"] {background:#161D26 !important;border:1px solid #293644 !important;
 border-radius:14px;padding:17px; min-width:0;}
[data-testid="stMetricLabel"] p {color:#9BAFC0 !important;font-size:.84rem;}
[data-testid="stMetricValue"] {color:#F0F6FC !important; font-size:1.65rem !important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#161D26 !important;
 border:1px solid #293644 !important; border-radius:14px !important;}
/* Make form controls match the site instead of using brown/red defaults. */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
 background:#1B2632 !important;border:1px solid #354657 !important;
 color:#F0F6FC !important;border-radius:10px !important;}
div[data-baseweb="select"] span, div[data-baseweb="select"] input,
div[data-baseweb="select"] svg {color:#E6EDF3 !important;fill:#E6EDF3 !important;}
div[data-baseweb="tag"] {background:#164C51 !important;border-radius:6px !important;}
div[data-baseweb="tag"] span {color:#C8FFF7 !important;}
div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
 background:#1B2632 !important;color:#E6EDF3 !important;}
.stTabs [data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #2D3B49;}
.stTabs [data-baseweb="tab"] {color:#AABAC9 !important;font-weight:600;padding:10px 14px;}
.stTabs [aria-selected="true"] {color:#51D4C7 !important;border-bottom:2px solid #51D4C7 !important;}
[data-testid="stAlert"] {border-radius:10px;}
[data-testid="stDataFrame"] {border:1px solid #293644;border-radius:10px;}
[data-testid="stProgressBar"] div[role="progressbar"] {background:#2EC4B6 !important;}
hr {border-color:#293644 !important;}
/* Plotly charts are styled separately using fig.update_layout. */
/* Dengue Watch-inspired layout, preserving charcoal color scheme */
[data-testid="stHeader"] {background:transparent !important;}
.block-container {max-width:1320px !important;padding-top:1.3rem !important;}
.dg-hero {padding:38px 40px 40px; margin:4px 0 30px; border-radius:22px;
 background:linear-gradient(120deg,#17252B 0%,#121B24 57%,#151C26 100%);
 border:1px solid #2A3D45; position:relative; overflow:hidden;}
.dg-hero:after {content:"";position:absolute;width:290px;height:290px;right:-100px;top:-155px;
 border-radius:50%;border:55px solid rgba(57,201,186,.055);pointer-events:none;}
.dg-eyebrow {font-size:11px;letter-spacing:2px;font-weight:800;color:#59D8CA;margin-bottom:20px;}
.dg-hero h1 {font-size:clamp(34px,4.2vw,58px)!important;line-height:1.09;
 letter-spacing:-.045em;margin:0 0 18px;color:#F3F8FC!important;}
.dg-hero h1 span {color:#51D4C7;}
.dg-hero p {font-size:15px;line-height:1.8;color:#B6C8D3!important;margin:0;}
[data-testid="stMetric"] {background:#151C25!important;min-height:125px;box-shadow:0 7px 24px rgba(0,0,0,.12);}
[data-testid="stMetricValue"] {font-size:1.55rem!important;white-space:normal!important;overflow-wrap:anywhere;}
.stTabs [data-baseweb="tab-list"] {gap:4px;overflow-x:auto;padding:5px 0 0;}
.stTabs [data-baseweb="tab"] {padding:12px 17px!important;white-space:nowrap;}
.stTabs [data-baseweb="tab-panel"] {padding-top:22px;}
section[data-testid="stSidebar"] {min-width:285px;}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {font-size:1.45rem!important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#151C25!important;}
[data-testid="stSlider"] [role="slider"] {background:#51D4C7!important;}
[data-testid="stAlert"] {background:#182732!important;}
[data-testid="stPlotlyChart"] {border:1px solid #293644;border-radius:16px;overflow:hidden;background:#151C25;}


/* DENGUEGUARD MOTION LAYER — CSS-only; prediction logic unchanged */
@keyframes dg-hero-arrive {
  from { opacity:0; transform:translateY(16px); }
  to { opacity:1; transform:translateY(0); }
}
@keyframes dg-aura-drift {
  0%,100% { transform:translate3d(0,0,0) scale(1); opacity:.55; }
  50% { transform:translate3d(-22px,18px,0) scale(1.13); opacity:.9; }
}
@keyframes dg-panel-arrive {
  from {opacity:0;transform:translateY(8px)}
  to {opacity:1;transform:translateY(0)}
}
.dg-hero {
  isolation:isolate;
  animation:dg-hero-arrive .8s cubic-bezier(.2,.8,.2,1) both;
  transition:border-color .35s ease,box-shadow .35s ease,transform .35s ease;
}
.dg-hero::before {
  content:"";position:absolute;z-index:-1;pointer-events:none;
  width:370px;height:370px;right:-90px;bottom:-235px;border-radius:50%;
  background:radial-gradient(circle,rgba(46,196,182,.23),transparent 68%);
  filter:blur(14px);animation:dg-aura-drift 9s ease-in-out infinite;
}
.dg-hero:hover {border-color:#3F7778;box-shadow:0 20px 60px rgba(0,0,0,.2),0 0 36px rgba(46,196,182,.07);}
/* Native Streamlit cards: hover transforms do not replace their widgets. */
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stPlotlyChart"] {
  position:relative;
  transition:transform .32s cubic-bezier(.2,.8,.2,1),
    box-shadow .32s ease,border-color .32s ease !important;
  will-change:auto;
}
[data-testid="stMetric"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stPlotlyChart"]:hover {
  transform:translateY(-5px) scale(1.012);
  border-color:#438F8B !important;
  box-shadow:0 16px 36px rgba(0,0,0,.28),0 0 24px rgba(46,196,182,.10) !important;
  z-index:2;
}
/* Slightly stronger lift for individual ward and city cards. */
[data-testid="stVerticalBlockBorderWrapper"]:hover [data-testid="stMetricValue"] {
  color:#76E8DC !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius:10px 10px 0 0;
  transition:background .25s ease,color .25s ease,transform .25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  background:rgba(46,196,182,.09) !important;
  color:#7BE9DD !important;
  transform:translateY(-2px);
}
.stTabs [data-baseweb="tab-panel"] {
  animation:dg-panel-arrive .36s ease-out both;
}
.stButton > button, [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  transition:transform .24s ease,box-shadow .24s ease,border-color .24s ease !important;
}
.stButton > button:hover {
  transform:translateY(-3px);
  box-shadow:0 10px 24px rgba(46,196,182,.17) !important;
  border-color:#4ED3C5 !important;
}
.stButton > button:active {transform:translateY(0) scale(.985);}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
  transition:border-color .22s ease,box-shadow .22s ease;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
[data-testid="stNumberInput"] div[data-baseweb="input"] > div:hover {
  border-color:#4C9C96 !important;
  box-shadow:0 0 0 2px rgba(46,196,182,.08);
}
/* Keep pop-out cards from causing sideways scrolling on narrow screens. */
@media (max-width:700px) {
  [data-testid="stMetric"]:hover,
  [data-testid="stVerticalBlockBorderWrapper"]:hover,
  [data-testid="stPlotlyChart"]:hover {transform:translateY(-3px);}
  .dg-hero {padding:28px 24px 30px;}
}
@media (prefers-reduced-motion:reduce) {
  .dg-hero,.dg-hero::before,.stTabs [data-baseweb="tab-panel"] {animation:none !important;}
  .dg-hero,[data-testid="stMetric"],
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stPlotlyChart"],.stTabs [data-baseweb="tab"],
  .stButton > button {transition:none !important;}
  [data-testid="stMetric"]:hover,
  [data-testid="stVerticalBlockBorderWrapper"]:hover,
  [data-testid="stPlotlyChart"]:hover,
  .stTabs [data-baseweb="tab"]:hover,
  .stButton > button:hover {transform:none !important;}
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def load_data(horizon):

    predictions = pd.read_csv(
        MODEL_DIR / f"test_predictions_{horizon}w.csv",
        parse_dates=["date"]
    )

    results = pd.read_csv(
        MODEL_DIR / f"results_{horizon}w.csv"
    )

    importance = pd.read_csv(
        MODEL_DIR / f"feature_importance_{horizon}w.csv"
    )

    metadata = pd.read_json(
        MODEL_DIR / f"{horizon}w" / "meta.json",
        typ="series"
    )

    return predictions, results, importance, metadata


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def ward_sort_key(value):
    """Sort ward names naturally by their numeric identifier."""
    match = re.search(r"(\d+)", str(value))

    if match:
        return int(match.group(1))

    return str(value)


def format_model_name(name):
    return str(name).replace("_", " ")


def get_target_column(horizon):
    return f"target_date_{horizon}w"



# ==================================================
# CHART STYLE
# ==================================================
def polish_chart(fig, height=440):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151C25",
        plot_bgcolor="#151C25",
        font=dict(color="#DCE7F2", size=13),
        title_font=dict(color="#F0F6FC", size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C5D5E3")),
        margin=dict(l=45, r=35, t=65, b=55),
        height=height,
        coloraxis_colorbar=dict(tickfont=dict(color="#C5D5E3")),
    )
    fig.update_xaxes(gridcolor="#293644", zerolinecolor="#293644",
                     tickfont=dict(color="#B8C8D8"), title_font=dict(color="#C5D5E3"))
    fig.update_yaxes(gridcolor="#293644", zerolinecolor="#293644",
                     tickfont=dict(color="#B8C8D8"), title_font=dict(color="#C5D5E3"))
    return fig


# ==================================================
# HEADER
# ==================================================

st.markdown("""<div class="dg-hero"><div class="dg-eyebrow">● DENGUE INTELLIGENCE PLATFORM</div>
<h1>Know the risk.<br><span>Protect the city.</span></h1>
<p>Ward-level dengue risk analysis across Chattogram, Gazipur and Narayanganj.<br>
Explore historical model predictions, weather drivers and evaluation results.</p>
</div>""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR CONTROLS
# ==================================================

st.sidebar.markdown("# 🦟 DengueGuard")
st.sidebar.caption("WARD-LEVEL EARLY WARNING · HISTORICAL DATA")

st.sidebar.divider()

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [2, 4],
    format_func=lambda x: f"{x}-Week Forecast"
)

try:
    preds, results, imp, metadata = load_data(horizon)

except Exception as error:
    st.error(
        "Unable to load the project data. "
        "Please check that the model files are present."
    )
    st.exception(error)
    st.stop()


# Model selection

best_model = metadata["best_model"]

model_options = results["model"].tolist()

if best_model not in model_options:
    st.error("The best model listed in metadata was not found in results.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Machine Learning Model",
    model_options,
    index=model_options.index(best_model),
    format_func=format_model_name
)


# City selection

available_cities = sorted(
    preds["city"].dropna().unique().tolist()    
    )

cities = st.sidebar.multiselect(
    "Select Cities",
    available_cities,
    default=available_cities
)


# Prediction date

dates = sorted(
    preds["date"].dropna().dt.date.unique().tolist()
)

if not dates:
    st.error("No prediction dates are available.")
    st.stop()

selected_date = st.sidebar.select_slider(
    "Prediction Week",
    options=dates,
    value=dates[-1]
)


st.sidebar.divider()

st.sidebar.markdown("**ACTIVE FILTERS**")
st.sidebar.caption(f"Forecast: {horizon} weeks")
st.sidebar.caption(f"Model: {format_model_name(model_choice)}")
st.sidebar.caption(f"Prediction week: {selected_date}")


# ==================================================
# FILTER PREDICTIONS
# ==================================================

probability_column = f"prob_{model_choice}"

if probability_column not in preds.columns:
    st.error(
        f"Could not find the prediction column: {probability_column}"
    )
    st.write("Available prediction columns:")
    st.write([
        column for column in preds.columns
        if column.startswith("prob_")
    ])
    st.stop()


week_df = preds[
    (preds["date"].dt.date == selected_date)
    & (preds["city"].isin(cities))
].copy()
week_df["probability"] = pd.to_numeric(
    week_df[probability_column],
    errors="coerce"
)

week_df = week_df.dropna(
    subset=["probability"]
)

week_df["predicted_high_risk"] = (
    week_df["probability"] >= 0.5
).astype(int)

week_df["risk"] = week_df["predicted_high_risk"].map({
    1: "High Risk",
    0: "Low Risk"
})

week_df = week_df.sort_values(
    ["city", "ward_id"],
    key=lambda column: (
        column.map(ward_sort_key)
        if column.name == "ward_id"
        else column
    )
)


# ==================================================
# OVERVIEW METRICS
# ==================================================

st.markdown("### Risk at a glance")

if week_df.empty:

    st.warning(
        "No prediction records found for the selected city and week."
    )

else:

    metric1, metric2, metric3, metric4 = st.columns(4)

    total_wards = len(week_df)

    high_risk_count = int(
        (week_df["predicted_high_risk"] == 1).sum()
    )

    average_probability = week_df["probability"].mean()
    highest_risk = week_df.loc[
        week_df["probability"].idxmax()
    ]

    metric1.metric(
        "Wards Analysed",
        total_wards
    )

    metric2.metric(
        "High-Risk Wards",
        high_risk_count
    )

    metric3.metric(
        "Average Risk Probability",
        f"{average_probability:.1%}"
    )

    metric4.metric(
        "Highest-Risk Ward",
        f"Ward {ward_sort_key(highest_risk['ward_id']):02d}" if isinstance(ward_sort_key(highest_risk["ward_id"]), int) else str(highest_risk["ward_id"]).split("_")[-1].lstrip("0") or "0"
    )


st.caption(
    f"Prediction origin: {selected_date}  |  "
    f"Forecast horizon: {horizon} weeks  |  "
    f"Selected model: {format_model_name(model_choice)}"
)

st.divider()


# ==================================================
# DASHBOARD NAVIGATION
# ==================================================

tab1, tab_city, tab2, tab3, tab4, tab_dynamic = st.tabs([
    "◉  Ward explorer",
    "▦  City overview",
    "◈  Model comparison",
    "◇  Weather drivers",
    "↗  Historical trends",
    "⚡  Dynamic prediction"
])


# ==================================================
# TAB 1: INDIVIDUAL WARD CARDS
# ==================================================

with tab1:

    st.markdown("## Explore wards")

    st.caption(
        "Each ward has its own prediction card. "
        "Select cities from the sidebar to filter the results."
         )

    if week_df.empty:

        st.info("No ward predictions available.")

    else:

        for city in sorted(week_df["city"].unique()):
            city_df = week_df[week_df["city"] == city].copy()
            city_df["ward_number"] = pd.to_numeric(
                city_df["ward_id"].astype(str).str.extract(r"(\d+)$")[0],
                errors="coerce"
            )
            city_df = city_df.sort_values("ward_number", na_position="last")
            st.markdown(f"### {city}")
            st.caption(f"{len(city_df)} wards · {horizon}-week prediction")

            # Every ward has its own card. No HTML strings are rendered.
            rows = list(city_df.iterrows())
            for row_start in range(0, len(rows), 3):
                columns = st.columns(3, gap="medium")
                for col, (_, ward) in zip(columns, rows[row_start:row_start + 3]):
                    probability = float(ward["probability"])
                    ward_number = ward["ward_number"]
                    ward_label = (
                        f"Ward {int(ward_number):02d}"
                        if pd.notna(ward_number) else str(ward["ward_id"])
                    )
                    with col:
                        with st.container(border=True):
                            st.caption(str(ward["ward_id"]))
                            st.markdown(f"#### {ward_label}")
                            st.metric("Model risk score", f"{probability:.2%}")
                            if probability >= 0.5:
                                st.markdown(":red[**HIGH RISK**]")
                            else:
                                st.markdown(":teal[**LOW RISK**]")
                            st.progress(max(0.0, min(1.0, probability)))
            st.divider()


        # Ward comparison chart

        st.markdown("### Compare ward probabilities")

        chart_df = week_df.copy()

        chart_df["Ward"] = (
            chart_df["ward_id"].astype(str)
        )
        fig = px.bar(
            chart_df,
            x="Ward",
            y="probability",
            color="risk",
            facet_col="city",
            facet_col_wrap=1,
            color_discrete_map={
                "High Risk": "#F27C78",
                "Low Risk": "#2EC4B6"
            },         
    
            hover_data=[
                "city",
                "ward_id",
                "probability"
            ],
            title="Predicted Risk Probability by Ward"
        )

        fig.update_layout(
        
            template="plotly_dark",
            paper_bgcolor="#151C25",
            plot_bgcolor="#151C25",
            font_color="#E7EEF5",
            yaxis_title="Risk Probability",
            xaxis_title="Individual Ward",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            height=max(450, len(chart_df) * 22)
        )

        fig.update_xaxes(
            tickangle=-35
        )

        polish_chart(fig, height=max(440, len(chart_df) * 25))
        st.plotly_chart(fig, width="stretch")


        # Ward details table

        st.markdown("### Detailed ward records")

        display = week_df[
            [
                "city",
                "ward_id",
                "probability",
                "risk",
                "predicted_high_risk",
                "actual_high_risk"
            ]
        ].copy()
        display.columns = [
            "City",
            "Ward",
            "Risk Probability",
            "Risk Status",
            "Predicted High Risk",
            "Actual High Risk"
        ]

        st.dataframe(
            display.style.format({
                "Risk Probability": "{:.2%}"
            }),
            width="stretch",
            hide_index=True
        )


# ==================================================
# CITY OVERVIEW — AGGREGATED SAVED PREDICTIONS
# ==================================================
with tab_city:
    st.markdown("## City overview")
    st.caption("Compare selected cities using the same saved historical predictions and model threshold.")
    if week_df.empty:
        st.info("Select at least one city with available records.")
    else:
        city_summary = (week_df.groupby("city", as_index=False)
                        .agg(wards=("ward_id", "count"),
                             high_risk_wards=("predicted_high_risk", "sum"),
                             mean_probability=("probability", "mean")))
        city_summary["high_risk_share"] = city_summary["high_risk_wards"] / city_summary["wards"]
        city_cols = st.columns(len(city_summary))
        for col, (_, city_row) in zip(city_cols, city_summary.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {city_row['city']}")
                    st.metric("Average risk score", f"{city_row['mean_probability']:.1%}")
                    st.caption(f"{int(city_row['high_risk_wards'])} high-risk / {int(city_row['wards'])} wards")
        city_chart = px.bar(
            city_summary, x="city", y="mean_probability", color="city",
            color_discrete_sequence=["#39C9BA", "#74A9E8", "#B39DF1"],
            text=city_summary["mean_probability"].map(lambda x: f"{x:.1%}"),
            title="Average predicted risk by city"
        )
        city_chart.update_traces(textposition="outside", cliponaxis=False)
        city_chart.update_layout(showlegend=False, yaxis_tickformat=".0%",
                                 yaxis_title="Mean predicted probability", xaxis_title="")
        polish_chart(city_chart, height=410)
        st.plotly_chart(city_chart, width="stretch")
        st.dataframe(city_summary.rename(columns={
            "city":"City", "wards":"Wards", "high_risk_wards":"High-risk wards",
            "mean_probability":"Average risk", "high_risk_share":"High-risk share"
        }).style.format({"Average risk":"{:.2%}", "High-risk share":"{:.1%}"}),
            hide_index=True, width="stretch")


# ==================================================
# TAB 2: MODEL PERFORMANCE
# ==================================================

with tab2:

    st.markdown("## Compare models")

    st.caption(
        "Compare the five trained models using their "
        "validation and test performance."
    )

    performance = results.copy()

    performance["model"] = (
        performance["model"].apply(format_model_name)
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    available_metrics = [
        column for column in metric_columns
        if column in performance.columns
    ]

    format_columns = {
        column: "{:.2%}"
        for column in available_metrics
    }
    if "validation_roc_auc" in performance.columns:
        format_columns["validation_roc_auc"] = "{:.2%}"

    st.dataframe(
        performance.style.format(format_columns),
        width="stretch",
        hide_index=True
    )

    if available_metrics:

        chart_data = performance.melt(
            id_vars="model",
            value_vars=available_metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="model",
            barmode="group",
            title="Comparison of Machine Learning Models"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151C25",
            plot_bgcolor="#151C25",
            font_color="#E6EDF3",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            legend_title="ML Model"
        )

        polish_chart(fig, height=460)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Best Performing Model")

    st.info(
        f"{format_model_name(best_model)} "
        f"is the best model for the {horizon}-week forecast "
        "according to the saved model metadata."
    )

    st.caption(
        "The best model may differ between the 2-week and 4-week horizons."
    )
    # ==================================================
# TAB 3: FEATURE IMPORTANCE
# ==================================================

with tab3:

    st.markdown("## Weather & risk drivers")

    st.caption(
        "Feature importance shows which input variables "
        "contributed most to the saved best model for this horizon."
    )

    if imp.empty:

        st.warning("No feature importance data is available.")

    else:

        top = imp.head(12).copy()

        top["importance"] = pd.to_numeric(
            top["importance"],
            errors="coerce"
        )

        top = top.dropna(
            subset=["importance"]
        )

        top = top.sort_values(
            "importance",
            ascending=True
        )

        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 12 Important Features",
            color="importance",
            color_continuous_scale=[
                "#164A53",
                "#238B8B",
                "#2EC4B6"
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151C25",
            plot_bgcolor="#151C25",
            font_color="#E6EDF3",
            xaxis_title="Feature Importance",
            yaxis_title="Feature",
            coloraxis_showscale=False
            )

        polish_chart(fig, height=490)
        fig.update_layout(margin=dict(l=165, r=30, t=65, b=55))
        st.plotly_chart(fig, width="stretch")

        st.info(
            "Feature importance indicates model contribution, "
            "not proof that a variable causes dengue."
        )


# ==================================================
# TAB 4: HISTORICAL EVALUATION
# ==================================================

with tab4:

    st.markdown("## Historical trends")

    st.caption(
        "Compare model predictions with actual high-risk labels "
        "from the historical test dataset."
    )

    selected = preds[
        preds["city"].isin(cities)
    ].copy()

    selected["probability"] = pd.to_numeric(
        selected[probability_column],
        errors="coerce"
    )

    selected = selected.dropna(
        subset=["probability"]
    )

    if selected.empty:

        st.warning(
            "No historical predictions are available for these cities."
        )

    else:

        selected["predicted_high_risk"] = (
            selected["probability"] >= 0.5
        ).astype(int)

        trend = selected.groupby("date").agg(
            predicted=(
                "predicted_high_risk",
                "mean"
            ),
            actual=(
                "actual_high_risk",
                "mean"
            )
        ).reset_index()
        trend_long = trend.melt(
            id_vars="date",
            var_name="Type",
            value_name="High Risk Share"
        )

        fig = px.line(
            trend_long,
            x="date",
            y="High Risk Share",
            color="Type",
            markers=True,
            color_discrete_map={
                "predicted": "#2EC4B6",
                "actual": "#FFB86B"
            },
            title="Predicted vs Actual High-Risk Ward Share"
        )

        

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151C25",
            plot_bgcolor="#151C25",
            font=dict(color="#E6EDF3", size=13),
            title=dict(
                text="Predicted vs Actual High-Risk Ward Share",
                font=dict(size=19, color="#FFFFFF")
            ),
            xaxis=dict(
                title="Prediction Week",
                title_font=dict(color="#C9D1D9"),
                tickfont=dict(color="#C9D1D9"),
                showgrid=False,
                linecolor="#30363D"
            ),
            yaxis=dict(
                title="High-Risk Ward Share",
                title_font=dict(color="#C9D1D9"),
                tickfont=dict(color="#C9D1D9"),
                tickformat=".0%",
                range=[0, 1],
                gridcolor="#30363D",
                zeroline=False
            ),
            legend=dict(
                title="Comparison",
                font=dict(color="#E6EDF3"),
                bgcolor="rgba(0,0,0,0)"
            ),
            height=480,
            margin=dict(l=35, r=35, t=75, b=40)
        )

        polish_chart(fig, height=490)
        st.plotly_chart(fig, width="stretch")

        # Historical classification results

        actual = selected["actual_high_risk"].astype(int)
        predicted = selected["predicted_high_risk"]

        correct = int(
            (actual == predicted).sum()
        )

        incorrect = int(
            (actual != predicted).sum()
        )

        false_alarms = int(
            ((predicted == 1) & (actual == 0)).sum()
        )

        missed_risks = int(
            ((predicted == 0) & (actual == 1)).sum()
        )

        accuracy = (
            correct / len(selected)
            if len(selected) else 0
        )

        st.markdown("### Historical Verification")

        a, b, c, d = st.columns(4)

        a.metric("Correct Predictions", correct)
        b.metric("Incorrect Predictions", incorrect)
        c.metric("False Alarms", false_alarms)
        d.metric("Missed High Risks", missed_risks)

        st.metric(
            "Historical Accuracy",
            f"{accuracy:.2%}"
        )

        st.caption(
            "These are historical test-set results, not a guarantee "
            "of future prediction accuracy."
        )

    st.warning(
        "The available prediction dataset covers historical test "
        "observations from 2024–2025. This dashboard does not "
        "provide live forecasts for 2026."
    )
# ==================================================
# TAB 6: INTERACTIVE HISTORICAL WHAT-IF INFERENCE
# ==================================================

@st.cache_data(show_spinner=False)
def load_raw_ward_history():
    raw = pd.read_csv(ROOT / "data" / "clean_dataset.csv", parse_dates=["date"])
    return raw.sort_values(["city", "ward_id", "date"]).reset_index(drop=True)


with tab_dynamic:
    st.markdown("## Dynamic ward risk prediction")
    st.caption(
        "Run the saved, validation-selected ML model on a historical ward-week "
        "and adjust observed weather values to explore a what-if scenario. "
        "This is not a live forecast or a causal estimate."
    )

    try:
        raw_history = load_raw_ward_history()
        eligible_history, _ = build_features(raw_history)
        # This historical demo uses rows with complete lags and known evaluation labels.
        available_cities_dynamic = sorted(eligible_history["city"].unique().tolist())

        with st.container(border=True):
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                dynamic_city = st.selectbox("City", available_cities_dynamic, key="dynamic_city")
            city_records = eligible_history[eligible_history["city"] == dynamic_city]
            with dc2:
                dynamic_ward = st.selectbox(
                    "Ward", sorted(city_records["ward_id"].unique().tolist(), key=ward_sort_key),
                    key="dynamic_ward"
                )
            ward_records = city_records[city_records["ward_id"] == dynamic_ward]
            historical_dates = sorted(ward_records["date"].dt.date.unique().tolist())
            with dc3:
                dynamic_date = st.selectbox(
                    "Historical observation week", historical_dates,
                    index=len(historical_dates) - 1, key="dynamic_date"
                )

            original_row = raw_history.loc[
                (raw_history["city"] == dynamic_city)
                & (raw_history["ward_id"] == dynamic_ward)
                & (raw_history["date"].dt.date == dynamic_date)
            ].iloc[0]

            w1, w2, w3 = st.columns(3)
            with w1:
                rainfall = st.number_input(
                    "Rainfall (mm)", min_value=0.0,
                    value=float(original_row["rainfall_mm"]), step=1.0,
                    key=f"rain_{dynamic_city}_{dynamic_ward}_{dynamic_date}"
                )
            with w2:
                avg_temp = st.number_input(
                    "Average temperature (°C)", min_value=-20.0, max_value=60.0,
                    value=float(original_row["avg_temp_c"]), step=0.5,
                    key=f"temp_{dynamic_city}_{dynamic_ward}_{dynamic_date}"
                )
            with w3:
                humidity = st.number_input(
                    "Relative humidity (%)", min_value=0.0, max_value=100.0,
                    value=float(original_row["relative_humidity_pct"]), step=1.0,
                    key=f"humidity_{dynamic_city}_{dynamic_ward}_{dynamic_date}"
                )

            run_dynamic = st.button("Generate model prediction", type="primary")

        if run_dynamic:
            # Rebuild engineered features using exactly the training feature logic.
            # Only the selected week's current weather is edited. Prior weeks and
            # their historical dengue cases remain unchanged.
            scenario = raw_history.copy()
            row_mask = (
                (scenario["city"] == dynamic_city)
                & (scenario["ward_id"] == dynamic_ward)
                & (scenario["date"].dt.date == dynamic_date)
            )
            scenario.loc[row_mask, "rainfall_mm"] = rainfall
            scenario.loc[row_mask, "avg_temp_c"] = avg_temp
            scenario.loc[row_mask, "relative_humidity_pct"] = humidity

            scenario_features, _ = build_features(scenario)
            scenario_row = scenario_features.loc[
                (scenario_features["city"] == dynamic_city)
                & (scenario_features["ward_id"] == dynamic_ward)
                & (scenario_features["date"].dt.date == dynamic_date)
            ]
            baseline_row = eligible_history.loc[
                (eligible_history["city"] == dynamic_city)
                & (eligible_history["ward_id"] == dynamic_ward)
                & (eligible_history["date"].dt.date == dynamic_date)
            ]
            if scenario_row.empty or baseline_row.empty:
                st.error("The selected historical record does not have enough complete features.")
            else:
                try:
                    baseline = predict_risk(baseline_row.iloc[0].to_dict(), horizon=horizon)
                    result = predict_risk(scenario_row.iloc[0].to_dict(), horizon=horizon)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Scenario risk probability", f"{result['estimated_probability']:.1%}")
                    c2.metric(
                        "Change from observed weather",
                        f"{(result['estimated_probability'] - baseline['estimated_probability']) * 100:+.1f} pp"
                    )
                    c3.metric(
                        "Predicted risk tier",
                        "HIGH RISK" if result["predicted_high_risk"] else "LOW RISK"
                    )
                    st.progress(max(0.0, min(1.0, result["estimated_probability"])))
                    st.caption(
                        f"Model: {format_model_name(result['model_name'])} | "
                        f"Horizon: {horizon} weeks | "
                        f"Baseline probability: {baseline['estimated_probability']:.1%} | "
                        "High-risk classification cutoff: 50%"
                    )
                    st.info(
                        "The model uses historical dengue and weather lags for this ward. "
                        "Changing a weather value does not establish that it causes "
                        "the predicted risk difference."
                    )
                except Exception as inference_error:
                    st.error("Could not generate a prediction using the saved model.")
                    st.exception(inference_error)
    except Exception as dynamic_error:
        st.error("Dynamic prediction data could not be loaded.")
        st.exception(dynamic_error)


    # ==================================================
# FOOTER
# ==================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption("🦟 DengueGuard")
    st.caption("University Machine Learning Project")

with footer_right:
    st.caption(
        "Historical model evaluation | "
        "Not an official public-health warning system"
    )

