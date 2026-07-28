"""Score the saved model and write ROC + confusion-matrix plots."""
import os

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

from .preprocess import load, split


def run() -> None:
    X_tr, X_te, y_tr, y_te = split(load())
    model = joblib.load("models/best.pkl")["model"]

    print(classification_report(y_te, model.predict(X_te)))

    os.makedirs("reports", exist_ok=True)
    RocCurveDisplay.from_estimator(model, X_te, y_te)
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=120)
    plt.close()

    ConfusionMatrixDisplay.from_estimator(model, X_te, y_te)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=120)
    plt.close()
    print("wrote reports/roc_curve.png, reports/confusion_matrix.png")


if __name__ == "__main__":
    run()
