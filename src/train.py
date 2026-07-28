"""Fit three models, score by ROC-AUC, persist the best."""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from .preprocess import load, split

MODELS = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "xgboost": XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        eval_metric="logloss",
        random_state=42,
    ),
}


def run() -> str:
    X_tr, X_te, y_tr, y_te = split(load())
    scores = {}
    for name, model in MODELS.items():
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        scores[name] = auc
        print(f"{name:14s} ROC-AUC {auc:.3f}")

    best = max(scores, key=scores.get)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": MODELS[best], "columns": list(X_tr.columns)}, "models/best.pkl")
    print(f"best: {best} ({scores[best]:.3f}) -> models/best.pkl")
    return best


if __name__ == "__main__":
    run()
