
"""Load dengue data and build leakage-aware forecasting features."""
import numpy as np
import pandas as pd

DATA_PATH = "data/clean_dataset.csv"
HIGH_RISK_THRESHOLD = 4
USE_CASE_LAGS = True
USE_EXTRA_FEATURES = True
SPLIT_DATE = "2024-01-01"

WEATHER_COLS = [
    "rainfall_mm",
    "avg_temp_c",
    "relative_humidity_pct",
]

STATIC_COLS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
]

GROUP_COLS = ["city", "ward_id"]

# Columns that must never be used as model inputs.
NON_FEATURES = [
    "date",
    "city",
    "ward_id",
    "dengue_cases",
    "target_date_2w",
    "target_date_4w",
    "high_risk_2w",
    "high_risk_4w",
]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return (
        df.sort_values(GROUP_COLS + ["date"])
        .reset_index(drop=True)
    )


def build_features(df, use_case_lags=USE_CASE_LAGS,
                   use_extra=USE_EXTRA_FEATURES):

    df = df.copy()
    g = df.groupby(GROUP_COLS, sort=False)

    # 1. Calendar features
    df["month"] = df["date"].dt.month
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    # 2. Historical weather features
    for col in WEATHER_COLS:
        for lag in (1, 2, 3, 4):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

    # 3. Historical rainfall totals
    df["rain_sum_4w"] = g["rainfall_mm"].transform(
        lambda s: s.shift(1).rolling(4).sum()
    )
    df["rain_sum_8w"] = g["rainfall_mm"].transform(
        lambda s: s.shift(1).rolling(8).sum()
    )

    # 4. Historical dengue cases
    if use_case_lags:
        for lag in (1, 2, 3, 4):
            df[f"cases_lag{lag}"] = g["dengue_cases"].shift(lag)

        df["cases_mean_4w"] = g["dengue_cases"].transform(
            lambda s: s.shift(1).rolling(4).mean()
        )

    # 5. Additional engineered features
    if use_extra:
        df["monsoon"] = df["month"].between(5, 10).astype(int)
        df["temp_range"] = (
            df["max_temp_c"] - df["min_temp_c"]
        )
        df["temp_x_humidity"] = (
            df["avg_temp_c"]
            * df["relative_humidity_pct"] / 100
        )

        df["temp_mean_4w"] = g["avg_temp_c"].transform(
            lambda s: s.shift(1).rolling(4).mean()
        )
        df["humidity_mean_4w"] = g[
            "relative_humidity_pct"
        ].transform(
            lambda s: s.shift(1).rolling(4).mean()
        )
        df["rain_sum_12w"] = g["rainfall_mm"].transform(
            lambda s: s.shift(1).rolling(12).sum()
        )
        df["log_rain_lag2"] = np.log1p(
            g["rainfall_mm"].shift(2)
        )

        if use_case_lags:
            df["cases_trend"] = (
                g["dengue_cases"].shift(1)
                - g["dengue_cases"].shift(2)
            )
            df["cases_mean_8w"] = g["dengue_cases"].transform(
                lambda s: s.shift(1).rolling(8).mean()
            )

    # 6. City encoding
    df["city_code"] = df["city"].astype("category").cat.codes

    # 7. Future forecasting targets
    # shift(-2) = dengue cases two weeks after the prediction date.
    # shift(-4) = dengue cases four weeks after the prediction date.

    df["target_date_2w"] = g["date"].shift(-2)
    df["target_date_4w"] = g["date"].shift(-4)

    cases_2w = g["dengue_cases"].shift(-2)
    cases_4w = g["dengue_cases"].shift(-4)

    df["high_risk_2w"] = (
        cases_2w >= HIGH_RISK_THRESHOLD
    ).astype(int)

    df["high_risk_4w"] = (
        cases_4w >= HIGH_RISK_THRESHOLD
    ).astype(int)

    # Remove rows without sufficient historical features
    # or future labels for both forecasting horizons.
    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURES
    ]

    required_cols = feature_cols + [
        "high_risk_2w",
        "high_risk_4w",
        "target_date_2w",
        "target_date_4w",
    ]

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df, feature_cols


def time_split(df, horizon):
    """Split by prediction date and keep training labels before test period."""

    if horizon not in (2, 4):
        raise ValueError("Horizon must be 2 or 4 weeks.")

    target_date_col = f"target_date_{horizon}w"

    # Training outcomes must occur before the test period.
    train = df[
        (df["date"] < SPLIT_DATE)
        & (df[target_date_col] < SPLIT_DATE)
    ].copy()

    # Test predictions originate on or after the split date.
    test = df[df["date"] >= SPLIT_DATE].copy()

    return train, test
