"""A runnable baseline model for the socio-economic score (regression).

Loads the bundled learning set, encodes the categorical features, trains a
Random Forest regressor and reports R² / MAE / RMSE + top features.

    python src/baseline.py
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "learn_dataset.csv")

def prepare_data(df: pd.DataFrame):
    """Extract features and target, removing non-predictive identifiers.

    Drops UNIQUE_ID (record key) and insee_code (geographic commune code)
    before one-hot encoding categorical predictors with drop_first=True.
    """
    target = "target"
    drop_cols = ["UNIQUE_ID", "insee_code"]
    X = pd.get_dummies(df.drop(columns=[target] + drop_cols), drop_first=True)
    y = df[target]
    return X, y


X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"Rows: {len(df):,} | features after encoding: {X.shape[1]}")
print(f"Target: continuous socio-economic score (mean {y.mean():.1f}, std {y.std():.1f})")
print(f"R²:   {r2_score(y_test, pred):.3f}")
print(f"MAE:  {mean_absolute_error(y_test, pred):.3f}")
print(f"RMSE: {rmse:.3f}")

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features driving the score:")
print(imp.head(10).round(3).to_string())
