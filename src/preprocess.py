"""Load the cleaned dengue dataset and build features (feature engineering)."""
import numpy as np
import pandas as pd

DATA_PATH = "data/clean_dataset.csv"
HIGH_RISK_THRESHOLD = 4        # weekly cases >= 4 -> "high risk" (about the 75th percentile)
USE_CASE_LAGS = True           # True: forecast using past cases; False: weather + ward info only
USE_EXTRA_FEATURES = True      # extra engineered features (see build_features)
SPLIT_DATE = "2024-01-01"      # train before this date, test from this date

WEATHER_COLS = ["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
NON_FEATURES = ["date", "city", "ward_id", "dengue_cases", "high_risk"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df, use_case_lags=USE_CASE_LAGS, use_extra=USE_EXTRA_FEATURES):
    """No feature uses the current week's dengue_cases (the target). Case features use
    earlier weeks only (shift(1) or more). Weather of the current week is allowed because
    it is an input, not an outcome."""
    df = df.copy()
    g = df.groupby("ward_id")

    # 1. Calendar features (dengue is seasonal)
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 2. Lagged weather (mosquito breeding follows weather with a delay)
    for col in WEATHER_COLS:
        for lag in (1, 2, 3, 4):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

    # 3. Rolling rainfall totals
    df["rain_sum_4w"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(4).sum())
    df["rain_sum_8w"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(8).sum())

    # 4. Past cases (previous weeks only)
    if use_case_lags:
        for lag in (1, 2, 3, 4):
            df[f"cases_lag{lag}"] = g["dengue_cases"].shift(lag)
        df["cases_mean_4w"] = g["dengue_cases"].transform(lambda s: s.shift(1).rolling(4).mean())

    # 5. Extra engineered features
    if use_extra:
        df["monsoon"] = df["month"].between(5, 10).astype(int)
        df["temp_range"] = df["max_temp_c"] - df["min_temp_c"]
        df["temp_x_humidity"] = df["avg_temp_c"] * df["relative_humidity_pct"] / 100
        df["temp_mean_4w"] = g["avg_temp_c"].transform(lambda s: s.shift(1).rolling(4).mean())
        df["humidity_mean_4w"] = g["relative_humidity_pct"].transform(lambda s: s.shift(1).rolling(4).mean())
        df["rain_sum_12w"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(12).sum())
        df["log_rain_lag2"] = np.log1p(g["rainfall_mm"].shift(2))
        if use_case_lags:
            df["cases_trend"] = g["dengue_cases"].shift(1) - g["dengue_cases"].shift(2)
            df["cases_mean_8w"] = g["dengue_cases"].transform(lambda s: s.shift(1).rolling(8).mean())

    # 6. City as numbers
    df["city_code"] = df["city"].astype("category").cat.codes

    # Target: 1 = high dengue risk this week
    df["high_risk"] = (df["dengue_cases"] >= HIGH_RISK_THRESHOLD).astype(int)

    df = df.dropna().reset_index(drop=True)  # drops the first weeks that have no lags
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    return df, feature_cols


def time_split(df):
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    return train, test