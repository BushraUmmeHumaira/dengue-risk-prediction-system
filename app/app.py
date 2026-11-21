"""Streamlit UI: dengue risk probability by ward."""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dengue Risk Prediction", page_icon="🦟", layout="wide")


@st.cache_data
def load():
    preds = pd.read_csv("model/test_predictions.csv", parse_dates=["date"])
    results = pd.read_csv("model/results.csv")
    imp = pd.read_csv("model/feature_importance.csv")
    return preds, results, imp


preds, results, imp = load()
model_names = [c.replace("prob_", "") for c in preds.columns if c.startswith("prob_")]
best_model = results.iloc[0]["model"]


def risk_level(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


st.title("🦟 Dengue Risk Prediction System")
st.caption("Probability of a high-risk dengue week for each ward in Chattogram, Gazipur and Narayanganj")

# ---- Sidebar controls
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Model", model_names, index=model_names.index(best_model))
cities = st.sidebar.multiselect("Cities", sorted(preds["city"].unique()),
                                default=sorted(preds["city"].unique()))
dates = sorted(preds["date"].dt.date.unique())
week = st.sidebar.select_slider("Week", options=dates, value=dates[-1])

week_df = preds[(preds["date"].dt.date == week) & (preds["city"].isin(cities))].copy()
week_df["probability"] = week_df[f"prob_{model_choice}"]
week_df["risk"] = week_df["probability"].apply(risk_level)
week_df = week_df.sort_values("probability", ascending=False)

# ---- Top numbers
c1, c2, c3 = st.columns(3)
c1.metric("Wards shown", len(week_df))
c2.metric("High-risk wards", int((week_df["risk"] == "High").sum()))
c3.metric("Highest-risk ward",
          week_df.iloc[0]["ward_id"] if len(week_df) else "-",
          f"{week_df.iloc[0]['probability']:.0%}" if len(week_df) else "")

tab1, tab2, tab3, tab4 = st.tabs(["Risk by ward", "Model comparison",
                                  "What drives risk", "Predicted vs actual"])

with tab1:
    st.subheader(f"Dengue probability by ward, week of {week}")
    fig = px.bar(week_df, x="ward_id", y="probability", color="probability",
                 color_continuous_scale="YlOrRd", range_color=(0, 1),
                 hover_data=["city", "risk", "dengue_cases"])
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Ward", yaxis_title="Probability of high risk")
    st.plotly_chart(fig, width="stretch")
    table = week_df[["ward_id", "city", "probability", "risk", "dengue_cases"]].rename(
        columns={"dengue_cases": "actual_cases"})
    st.dataframe(table.style.format({"probability": "{:.1%}"}), width="stretch", hide_index=True)

with tab2:
    st.subheader("Models compared on unseen 2024-2025 data")
    st.dataframe(results, width="stretch", hide_index=True)
    long = results.melt(id_vars="model", var_name="metric", value_name="score")
    st.plotly_chart(px.bar(long, x="metric", y="score", color="model", barmode="group"),
                    width="stretch")

with tab3:
    st.subheader("Most important factors (XGBoost)")
    top = imp.head(12).sort_values("importance")
    st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h"),
                    width="stretch")

with tab4:
    st.subheader("Average predicted probability vs actual high-risk share over time")
    sel = preds[preds["city"].isin(cities)].copy()
    sel["probability"] = sel[f"prob_{model_choice}"]
    trend = sel.groupby("date").agg(predicted=("probability", "mean"),
                                    actual=("high_risk", "mean")).reset_index()
    st.plotly_chart(px.line(trend, x="date", y=["predicted", "actual"]), width="stretch")
