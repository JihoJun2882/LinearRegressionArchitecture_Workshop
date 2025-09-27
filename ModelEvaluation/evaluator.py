from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class Evaluator:
    """Computes metrics and persists simple diagnostic plots."""
    def __init__(self, plot_dir: str):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def metrics(self, df_pred: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
        rows = []
        for k in axes:
            y = df_pred[k].values
            yhat = df_pred[f"{k}_pred"].values
            rows.append({
                "axis": k,
                "r2": r2_score(y, yhat),
                "mae": mean_absolute_error(y, yhat),
                "rmse": mean_squared_error(y, yhat) ** 0.5,
            })
        return pd.DataFrame(rows)

    def plots(self, df_pred: pd.DataFrame, axes: list[str]) -> None:
        for k in axes:
            # Scatter + regression line
            fig = plt.figure(figsize=(8, 4))
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2'])
            c_data = colors[0]  # data color
            c_line = colors[1]  # line color

            plt.scatter(df_pred["time_s"], df_pred[k], s=10, alpha=0.7, label="data", c=c_data)
            plt.plot(df_pred["time_s"], df_pred[f"{k}_pred"], label="fit", color=c_line, linewidth=1.5)
            plt.xlabel("time (s)"); plt.ylabel(k); plt.legend(); plt.tight_layout()
            fig.savefig(self.plot_dir / f"{k}_fit.png"); plt.close(fig)

            # Residual series
            fig = plt.figure(figsize=(8, 3))
            plt.scatter(df_pred["time_s"], df_pred[f"{k}_res"], s=10, alpha=0.8, label="residuals", c=c_data)
            plt.axhline(0, color=c_line, linestyle="--", linewidth=1)
            plt.tight_layout(); fig.savefig(self.plot_dir / f"{k}_residual.png"); plt.close(fig)
