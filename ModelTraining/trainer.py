# from pathlib import Path
# from dataclasses import dataclass
# from typing import Dict
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from joblib import dump

# @dataclass
# class AxisModel:
#     """Stores per-axis linear coefficients."""
#     coef: float
#     intercept: float

# class LinearAxisTrainer:
#     """Fits a separate univariate linear regression per axis (y = a*t + b)."""
#     def __init__(self, out_dir: str):
#         self.out_dir = Path(out_dir)
#         self.out_dir.mkdir(parents=True, exist_ok=True)
#         self.models: Dict[str, AxisModel] = {}

#     def fit(self, df_train: pd.DataFrame, axes: list[str]) -> None:
#         X = df_train[["time_s"]].values
#         for k in axes:
#             y = df_train[k].values
#             lr = LinearRegression().fit(X, y)
#             self.models[k] = AxisModel(float(lr.coef_[0]), float(lr.intercept_))
#         dump(self.models, self.out_dir / "models.pkl")

#     def predict(self, df: pd.DataFrame) -> pd.DataFrame:
#         if not self.models:
#             raise RuntimeError("Models are not fitted. Call fit() first.")
#         out = df.copy()
#         for k, m in self.models.items():
#             out[f"{k}_pred"] = m.coef * out["time_s"] + m.intercept
#             out[f"{k}_res"]  = out[k] - out[f"{k}_pred"]
#         return out
from sklearn.linear_model import LinearRegression
import numpy as np, pandas as pd
from pathlib import Path
import joblib, json

class LinearAxisTrainer:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}

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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        t = out[["time_s"]].values
        for k, m in self.models.items():
            yhat = m["coef"] * t.ravel() + m["intercept"]
            out[f"{k}_pred"] = yhat
            out[f"{k}_res"]  = out[k].values - yhat
        return out
