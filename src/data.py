"""Generate a synthetic customer table so the repo runs without external data.

Churn is driven by tenure, monthly charges and contract type, plus noise — enough
signal for models to separate but not trivially so.
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = "data/customers.csv"


def generate(n: int = 8000) -> pd.DataFrame:
    tenure = RNG.integers(0, 72, n)
    monthly = RNG.normal(70, 25, n).clip(15, 130)
    contract = RNG.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = RNG.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.35, 0.20])
    support = RNG.choice([0, 1], n, p=[0.6, 0.4])

    contract_risk = np.select(
        [contract == "month-to-month", contract == "one-year"], [2.0, 0.5], default=-1.2
    )
    logit = (
        -1.0
        + contract_risk
        - 0.06 * tenure
        + 0.025 * (monthly - 70)
        + 0.8 * (internet == "fiber")
        - 1.0 * support
        + RNG.normal(0, 0.35, n)
    )
    churn = (1 / (1 + np.exp(-logit))) > RNG.random(n)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly.round(2),
            "contract": contract,
            "internet": internet,
            "tech_support": support,
            "churn": churn.astype(int),
        }
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT} | churn rate {df.churn.mean():.2%}")
