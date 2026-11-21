"""Streamlit dashboard: dengue risk probability by ward."""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dengue Risk Dashboard", page_icon="🦟", layout="wide")

REQUIRED = ["model/test_predictions.csv", "model/results.csv", "model/feature_importance.csv"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Run `python src/train.py` first.")
    st.stop()

RISK_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a020", "High": "#d62f2f"}


@st.cache_data
def load():
    preds = pd.read_csv("model/test_predictions.csv", parse_dates=["date"])
    results = pd.read_csv("model/results.csv")
    imp = pd.read_csv("model/feature_importance.csv")
    return preds, results, imp


def risk_level(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


preds, results, imp = load()
model_names = [c.replace("prob_", "") for c in preds.columns if c.startswith("prob_")]
best_model = results.iloc[0]["model"]

# ---------------- Header
st.title("🦟 Dengue Risk Dashboard")
st.caption("Weekly dengue risk for 20 wards in Chattogram, Gazipur and Narayanganj, "
           "predicted with machine learning models trained on 2016-2023 data.")
with st.expander("How to read this dashboard"):
    st.markdown(
        "- **Probability** is the model's estimate that a ward has **4 or more dengue cases** in the selected week.\n"
        "- **Risk level:** Low is below 30%, Medium is 30% to 60%, High is 60% or more.\n"
        "- Predictions shown are for **2024-2025**, weeks the models never saw during training.\n"
        "- This is a decision-support tool for planning, not a diagnosis.")

# ---------------- Sidebar
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Model", model_names, index=model_names.index(best_model))
all_cities = sorted(preds["city"].unique())
cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities)
dates = sorted(preds["date"].dt.date.unique())
peak_week = preds.groupby("date")[f"prob_{best_model}"].mean().idxmax().date()  # most active week
week = st.sidebar.select_slider("Week", options=dates, value=peak_week)
default_week = preds.groupby("date")[f"prob_{best_model}"].std().idxmax().date()  # week with the most mixed risk levels
week = st.sidebar.select_slider("Week", options=dates, value=default_week)
preds["probability"] = preds[f"prob_{model_choice}"]
preds["risk"] = preds["probability"].apply(risk_level)
week_df = preds[(preds["date"].dt.date == week) & (preds["city"].isin(cities))]
week_df = week_df.sort_values("probability", ascending=False)
if week_df.empty:
    st.warning("Select at least one city.")
    st.stop()

# ---------------- Summary cards
top = week_df.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wards shown", len(week_df))
c2.metric("High-risk wards", int((week_df["risk"] == "High").sum()))
c3.metric("Medium-risk wards", int((week_df["risk"] == "Medium").sum()))
c4.metric("Highest-risk ward", top["ward_id"], f"{top['probability']:.0%}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Risk by ward", "Trends", "Model comparison", "What drives risk", "About"])

# ---------------- Tab 1: risk by ward
with tab1:
    st.subheader(f"Dengue risk by ward, week of {week}")
    fig = px.bar(week_df, x="ward_id", y="probability", color="risk",
                 color_discrete_map=RISK_COLORS, category_orders={"risk": ["High", "Medium", "Low"]},
                 hover_data={"city": True, "dengue_cases": True, "probability": ":.1%"})
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="Ward",
                      yaxis_title="Probability of high risk", legend_title="Risk level")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([2, 1])
    with left:
        table = week_df[["ward_id", "city", "probability", "risk", "dengue_cases"]].copy()
        table["probability"] = table["probability"] * 100
        table = table.rename(columns={"dengue_cases": "actual_cases"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "probability": st.column_config.ProgressColumn("Probability", format="%.0f%%", min_value=0, max_value=100)})
        st.download_button("Download this table (CSV)", table.to_csv(index=False), f"risk_{week}.csv", "text/csv")
    with right:
        st.markdown("**Average risk by city**")
        city_avg = week_df.groupby("city")["probability"].mean().reset_index()
        st.plotly_chart(px.bar(city_avg, x="city", y="probability").update_layout(
            yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="", yaxis_title=""), width="stretch")

# ---------------- Tab 2: trends
with tab2:
    ward = st.selectbox("Ward", sorted(preds[preds["city"].isin(cities)]["ward_id"].unique()))
    wdf = preds[preds["ward_id"] == ward].sort_values("date")
    st.markdown("**Predicted probability of high risk**")
    st.plotly_chart(px.line(wdf, x="date", y="probability").update_layout(
        yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="", yaxis_title=""), width="stretch")
    st.markdown("**Actual weekly dengue cases**")
    st.plotly_chart(px.bar(wdf, x="date", y="dengue_cases").update_layout(
        xaxis_title="", yaxis_title="Cases"), width="stretch")

# ---------------- Tab 3: model comparison
with tab3:
    st.subheader("Models compared on unseen 2024-2025 data")
    st.dataframe(results, hide_index=True, width="stretch")
    long = results.melt(id_vars="model", var_name="metric", value_name="score")
    st.plotly_chart(px.bar(long, x="metric", y="score", color="model", barmode="group"), width="stretch")
    st.caption(f"Best model by ROC-AUC: **{best_model}**. Recall is the share of real high-risk weeks "
               "the model catches, which matters most for early warning.")

# ---------------- Tab 4: feature importance
with tab4:
    st.subheader("Most important factors (XGBoost)")
    top12 = imp.head(12).sort_values("importance")
    st.plotly_chart(px.bar(top12, x="importance", y="feature", orientation="h"), width="stretch")
    st.caption("Rainfall and humidity over the previous weeks matter most, which fits how "
               "mosquitoes breed after rain.")

# ---------------- Tab 5: about
with tab5:
    st.markdown(
        "### Pipeline\n"
        "Raw dengue CSV → data cleaning and validation → feature engineering → train 5 models → "
        "evaluate and select best → ward risk probabilities → this dashboard.\n\n"
        "### Data\n"
        "Weekly records for 20 wards (2016-2025): weather, ward conditions and dengue cases.\n\n"
        "### Models\n"
        "Logistic Regression, Random Forest, XGBoost, LightGBM and a neural network (MLP). "
        "No LLM is used in the prediction.\n\n"
        "### Limits\n"
        "A high-risk week means 4 or more cases. Predictions are estimates and should support, "
        "not replace, public health judgment.")
