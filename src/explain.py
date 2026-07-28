"""SHAP summary for the saved model."""
import os

import joblib
import matplotlib.pyplot as plt
import shap

from .preprocess import load, split


def run() -> None:
    X_tr, X_te, y_tr, y_te = split(load())
    model = joblib.load("models/best.pkl")["model"]

    explainer = shap.Explainer(model, X_tr)
    values = explainer(X_te)

    os.makedirs("reports", exist_ok=True)
    shap.summary_plot(values, X_te, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=120)
    plt.close()
    print("wrote reports/shap_summary.png")


if __name__ == "__main__":
    run()
