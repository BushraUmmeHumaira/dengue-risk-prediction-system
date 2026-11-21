
import pandas as pd

from src.predict import load_forecast_model


def get_feature_importance(horizon):
    model, metadata = load_forecast_model(horizon)
    features = metadata["features"]

    if horizon == 2:
        # Logistic Regression: absolute standardized coefficients
        classifier = model.named_steps["logisticregression"]
        importance = abs(classifier.coef_[0])
        method = "Absolute Logistic Regression coefficient"

    elif horizon == 4:
        # XGBoost: built-in feature importance
        importance = model.feature_importances_
        method = "XGBoost feature importance"

    else:
        raise ValueError("Horizon must be 2 or 4 weeks.")

    report = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    report = report.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    report["rank"] = report.index + 1
    report["method"] = method

    return report


if __name__ == "__main__":
    for horizon in (2, 4):
        report = get_feature_importance(horizon)

        output_path = (
            f"model/feature_importance_{horizon}w.csv"
        )

        report.to_csv(output_path, index=False)

        print(f"\n{horizon}-WEEK FEATURE IMPORTANCE")
        print(report.head(10).to_string(index=False))
        print(f"Saved to {output_path}")
