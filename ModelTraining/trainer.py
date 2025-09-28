# Trains per-axis univariate linear regressions and persists only the coefficients
# Uses those coefficients to add predictions and residuals to a Dataframe
# The existing code is easy to initialize because it only stores the value in a variable called model. -> Save the value to a file.
# Exact-zero filtering - same as privios codebase

from sklearn.linear_model import LinearRegression
import numpy as np, pandas as pd
from pathlib import Path
import joblib, json

class LinearAxisTrainer:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}

        # For each axis train with linerregression and save coefficient & intercept in models.pkl.
    def fit(self, train_df: pd.DataFrame, axes: list[str]) -> None:
        X_full = train_df[["time_s"]].values
        for k in axes:
            y = train_df[k].values
            # EXCLUDE exact zeros for this axis (to match the notebook)
            mask = np.isfinite(y) & (y != 0)
            X = X_full[mask]
            y = y[mask]
            lr = LinearRegression().fit(X, y)
            self.models[k] = {"coef": float(lr.coef_[0]), "intercept": float(lr.intercept_)}

        joblib.dump(self.models, self.out_dir / "models.pkl")

        # Calculate using the saved coef & intercept values ​​and add <axis>_pred, <axis>_res columns.
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        t = out[["time_s"]].values
        for k, m in self.models.items():
            yhat = m["coef"] * t.ravel() + m["intercept"]
            out[f"{k}_pred"] = yhat
            out[f"{k}_res"]  = out[k].values - yhat
        return out
