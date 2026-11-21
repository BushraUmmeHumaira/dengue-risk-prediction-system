"""Load the dengue dataset and build features."""
import pandas as pd

DATA_PATH = "data/dengue_dataset.csv"
HIGH_RISK_THRESHOLD = 4      # weekly cases >= 4 -> "high risk" (about the 75th percentile)
USE_CASE_LAGS = True         # True: forecast using past cases; False: weather + ward info only
SPLIT_DATE = "2024-01-01"    # train before this date, test from this date

WEATHER_COLS = ["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df, use_case_lags=USE_CASE_LAGS):
    df = df.copy()
    g = df.groupby("ward_id")

    # Calendar features (dengue is seasonal)
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Lagged weather (mosquito breeding follows rain and heat by some weeks)
    for col in WEATHER_COLS:
        for lag in (1, 2, 3, 4):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
    df["rain_sum_4w"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(4).sum())
    df["rain_sum_8w"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(8).sum())

    # Past cases (only previous weeks, never the current one)
    if use_case_lags:
        for lag in (1, 2, 3, 4):
            df[f"cases_lag{lag}"] = g["dengue_cases"].shift(lag)
        df["cases_mean_4w"] = g["dengue_cases"].transform(lambda s: s.shift(1).rolling(4).mean())

    # City as numbers
    df["city_code"] = df["city"].astype("category").cat.codes

    # Target: 1 = high dengue risk this week
    df["high_risk"] = (df["dengue_cases"] >= HIGH_RISK_THRESHOLD).astype(int)

    df = df.dropna().reset_index(drop=True)  # drops the first weeks that have no lags
    feature_cols = [c for c in df.columns
                    if c not in ["date", "city", "ward_id", "dengue_cases", "high_risk"]]
    return df, feature_cols


def time_split(df):
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    return train, test
