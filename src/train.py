
"""Train and evaluate dengue risk forecasting models at 2 and 4 weeks."""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_data, build_features, time_split


MODEL_DIR = Path("model")
VALIDATION_DATE = "2022-01-01"
TEST_DATE = "2024-01-01"


def get_models():
    return {
        "Logistic_Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),

        "Neural_Network_MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        ),
    }


def evaluate_model(y_true, predictions, probabilities):
    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": auc,
    }


def train_horizon(df, features, horizon):

    print(f"\n{'=' * 55}")
    print(f"TRAINING {horizon}-WEEK FORECAST")
    print(f"{'=' * 55}")

    target = f"high_risk_{horizon}w"
    target_date = f"target_date_{horizon}w"

    # Final untouched test set: 2024 onwards.
    _, test = time_split(df, horizon)

    # Training period: before 2022.
    train = df[
        (df["date"] < VALIDATION_DATE)
        & (df[target_date] < VALIDATION_DATE)
    ].copy()

    # Validation period: 2022-2023.
    validation = df[
        (df["date"] >= VALIDATION_DATE)
        & (df["date"] < TEST_DATE)
        & (df[target_date] < TEST_DATE)
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty training, validation, or test data "
            f"for {horizon}-week forecasting."
        )

    X_train = train[features]
    y_train = train[target]

    X_val = validation[features]
    y_val = validation[target]

    X_test = test[features]
    y_test = test[target]

    if y_train.nunique() < 2:
        raise ValueError("Training data contains only one risk class.")

    if y_val.nunique() < 2:
        raise ValueError("Validation data contains only one risk class.")

    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Testing rows: {len(test)}")
    print(f"Features: {len(features)}")
    print(f"Training high-risk share: {y_train.mean():.3f}")
    print(f"Validation high-risk share: {y_val.mean():.3f}")
    print(f"Testing high-risk share: {y_test.mean():.3f}")

    horizon_dir = MODEL_DIR / f"{horizon}w"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    validation_results = []
    final_results = []
    probabilities_by_model = {}

    # First stage: choose the model using validation data.
    for name, model in get_models().items():

        print(f"\nValidation: {name}")

        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]

        try:
            val_auc = roc_auc_score(y_val, val_probabilities)
        except ValueError:
            val_auc = float("nan")

        validation_results.append({
            "model": name,
            "validation_roc_auc": val_auc
        })

        print(f"Validation ROC-AUC: {val_auc:.4f}")

    validation_df = pd.DataFrame(validation_results)

    valid_validation = validation_df.dropna(
        subset=["validation_roc_auc"]
    )

    if valid_validation.empty:
        raise ValueError(
            f"No valid validation ROC-AUC for {horizon}-week forecast."
        )

    # Select the model without looking at test performance.
    best_model_name = valid_validation.sort_values(
        "validation_roc_auc",
        ascending=False
    ).iloc[0]["model"]

    print(f"\nSelected using validation: {best_model_name}")

    # Second stage: refit all candidates on training + validation.
    train_val = pd.concat(
        [train, validation],
        ignore_index=True
    )

    X_train_val = train_val[features]
    y_train_val = train_val[target]

    final_models = get_models()

    for name, model in final_models.items():

        print(f"\nFinal test evaluation: {name}")

        model.fit(X_train_val, y_train_val)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = evaluate_model(
            y_test,
            predictions,
            probabilities
        )

        val_auc = validation_df.loc[
            validation_df["model"] == name,
            "validation_roc_auc"
        ].iloc[0]

        final_results.append({
            "model": name,
            "validation_roc_auc": val_auc,
            **metrics,
            "selected": name == best_model_name
        })

        probabilities_by_model[name] = probabilities

        joblib.dump(
            model,
            horizon_dir / f"{name}.joblib"
        )

        print(
            f"Test Accuracy: {metrics['accuracy']:.3f} | "
            f"Test F1: {metrics['f1']:.3f} | "
            f"Test ROC-AUC: {metrics['roc_auc']:.3f}"
        )

    # Sort by validation ROC-AUC, not test ROC-AUC.
    results_df = pd.DataFrame(final_results)

    results_df = results_df.sort_values(
        "validation_roc_auc",
        ascending=False,
        na_position="last"
    )

    results_df.to_csv(
        MODEL_DIR / f"results_{horizon}w.csv",
        index=False
    )

    print(f"\nSelected {horizon}-week model: {best_model_name}")

    # Save predictions from the validation-selected model.
    output = test[
        ["date", "city", "ward_id", target_date, target]
    ].copy()

    output = output.rename(columns={
        target: "actual_high_risk"
    })

    output["probability"] = probabilities_by_model[
        best_model_name
    ]

    output["predicted_high_risk"] = (
        output["probability"] >= 0.5
    ).astype(int)

    for name, probabilities in probabilities_by_model.items():
        output[f"prob_{name}"] = probabilities

    output.to_csv(
        MODEL_DIR / f"test_predictions_{horizon}w.csv",
        index=False
    )

    metadata = {
        "forecast_horizon_weeks": horizon,
        "best_model": best_model_name,
        "selection_metric": "validation_roc_auc",
        "validation_start": VALIDATION_DATE,
        "test_start": TEST_DATE,
        "features": features,
        "target": target,
        "target_date_column": target_date,
        "threshold": 4,
    }

    with open(
        horizon_dir / "meta.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    return results_df


def main():

    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading dengue dataset...")

    df, features = build_features(load_data())

    print(f"Prepared rows: {len(df)}")
    print(f"Feature count: {len(features)}")

    results_2w = train_horizon(df, features, 2)
    results_4w = train_horizon(df, features, 4)

    print("\n" + "=" * 55)
    print("FORECAST TRAINING COMPLETED")
    print("=" * 55)

    print("\n2-week model comparison:")
    print(results_2w.to_string(index=False))

    print("\n4-week model comparison:")
    print(results_4w.to_string(index=False))


if __name__ == "__main__":
    main()
