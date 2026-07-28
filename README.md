# churn-prediction

Predicts customer churn and surfaces the drivers behind it. Compares three models,
explains the best one with SHAP, and serves predictions through a Streamlit app.

## Stack
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn.

## Layout
```
src/
  data.py         synthetic dataset generator (so the repo runs standalone)
  preprocess.py   cleaning, encoding, train/test split
  train.py        fit LogReg / RandomForest / XGBoost, pick best by ROC-AUC
  evaluate.py     metrics + ROC / confusion-matrix plots
  explain.py      SHAP feature attributions
app.py            Streamlit dashboard
tests/            preprocessing tests
```

## Run
```bash
pip install -r requirements.txt

python -m src.data                 # writes data/customers.csv
python -m src.train                # saves models/best.pkl, prints ROC-AUC
python -m src.evaluate             # writes reports/ plots
python -m src.explain              # writes reports/shap_summary.png
streamlit run app.py               # interactive predictions
```

## Results
On the generated dataset, XGBoost reaches ROC-AUC ≈ 0.87. Top drivers are tenure,
monthly charges, and contract type. Swap in a real dataset by replacing
`data/customers.csv` (same columns) — no code changes needed.

## Tests
```bash
pytest
```
