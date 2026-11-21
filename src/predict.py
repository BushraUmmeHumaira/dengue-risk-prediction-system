import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model's binary classification cutoff
CLASSIFICATION_CUTOFF = 0.5


@lru_cache(maxsize=2)
def load_forecast_model(horizon):
    """
    Load the selected model and metadata for a forecast horizon.
    Supported horizons: 2 or 4 weeks.
    """
    if horizon not in (2, 4):
        raise ValueError("Horizon must be either 2 or 4 weeks.")

    model_dir = PROJECT_ROOT / "model" / f"{horizon}w"

    metadata_path = model_dir / "meta.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = metadata["best_model"]
    model_path = model_dir / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    return model, metadata


def predict_risk(features, horizon=2):
    """
    Predict dengue high-risk status using prepared features.

    features must be a dictionary containing all 39 model features.
    The input should represent one ward's latest observation.
    """

    model, metadata = load_forecast_model(horizon)

    required_features = metadata["features"]

    # Check for missing features
    missing = [
        feature for feature in required_features
        if feature not in features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    # Arrange features in the exact training order
    input_df = pd.DataFrame(
        [{feature: features[feature] for feature in required_features}]
    )

    # Ensure all model inputs are numeric
    input_df = input_df.apply(pd.to_numeric, errors="coerce")

    if input_df.isnull().any().any():
        invalid = input_df.columns[
            input_df.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing or invalid numeric values in: {invalid}"
        )

    # Get estimated probability of the positive class
    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain positive class 1."
        )

    positive_index = classes.index(1)
    estimated_probability = float(
        probabilities[positive_index]
    )

    # Classification using the model's standard 0.5 cutoff
    predicted_high_risk = (
        estimated_probability >= CLASSIFICATION_CUTOFF
    )

    return {
        "forecast_horizon_weeks": horizon,
        "model_name": metadata["best_model"],
        "estimated_probability": estimated_probability,
        "predicted_high_risk": predicted_high_risk,
        "classification_cutoff": CLASSIFICATION_CUTOFF,
        "high_risk_case_threshold": metadata["threshold"]
    }