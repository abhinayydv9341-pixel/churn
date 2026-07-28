import pandas as pd

from src.data import generate
from src.preprocess import encode, split


def test_encode_removes_categoricals():
    df = generate(200)
    enc = encode(df)
    assert "contract" not in enc.columns
    assert any(c.startswith("contract_") for c in enc.columns)


def test_split_is_stratified():
    df = generate(1000)
    X_tr, X_te, y_tr, y_te = split(df, test_size=0.25)
    assert len(X_te) == 250
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
