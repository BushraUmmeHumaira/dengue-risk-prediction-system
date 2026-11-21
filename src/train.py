"""Train several models, compare them, save results."""
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess import load_data, build_features, time_split


def get_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1),
        "Neural Network (MLP)": make_pipeline(StandardScaler(),
                                              MLPClassifier(hidden_layer_sizes=(64, 32),
                                                            max_iter=500, early_stopping=True,
                                                            random_state=42)),
    }


def main():
    df, features = build_features(load_data())
    train, test = time_split(df)
    X_train, y_train = train[features], train["high_risk"]
    X_test, y_test = test[features], test["high_risk"]
    print(f"Train rows: {len(train)}, Test rows: {len(test)}, Features: {len(features)}")
    print(f"High-risk share  train: {y_train.mean():.2f}  test: {y_test.mean():.2f}")

    results, probs = [], {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, p),
        })
        probs[name] = p
        file_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"model/{file_name}.joblib")
        print(f"{name:22s} acc={results[-1]['accuracy']:.3f}  f1={results[-1]['f1']:.3f}  auc={results[-1]['roc_auc']:.3f}")

    res = pd.DataFrame(results).sort_values("roc_auc", ascending=False).round(4)
    res.to_csv("model/results.csv", index=False)
    best = res.iloc[0]["model"]
    print("\nBest model:", best)

    # Predictions for the test period (the UI reads this file)
    out = test[["date", "city", "ward_id", "dengue_cases", "high_risk"]].copy()
    out["probability"] = probs[best]
    out["predicted_high_risk"] = (out["probability"] >= 0.5).astype(int)
    for name, p in probs.items():
        out[f"prob_{name}"] = p
    out.to_csv("model/test_predictions.csv", index=False)

    # Feature importance from XGBoost (shows which factors drive risk)
    xgb = get_models()["XGBoost"].fit(X_train, y_train)
    imp = pd.DataFrame({"feature": features, "importance": xgb.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv("model/feature_importance.csv", index=False)

    json.dump({"best_model": best, "features": features}, open("model/meta.json", "w"))


if __name__ == "__main__":
    main()
