"""Load, encode and split the customer table."""
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = ["contract", "internet"]
TARGET = "churn"


def load(path: str = "data/customers.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the categoricals, drop the target's leakage-free view unchanged."""
    return pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    enc = encode(df)
    X = enc.drop(columns=[TARGET])
    y = enc[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
