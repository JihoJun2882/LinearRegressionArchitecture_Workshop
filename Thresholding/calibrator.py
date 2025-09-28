# Calculate per-axis residual-based thresholds and dwell times (in steps), then writes them to thresholds.json
# The threshold is calculated in the same way as the existing code. The difference lies in saving it to a file.

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np, pandas as pd, json

class ThresholdCalibrator:
    """Residual-based thresholds per axis with trim + quantile policy."""

    def __init__(self, out_dir: str, policy: Optional[Dict] = None):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.th: Dict[str, Dict] = {}
        policy = policy or {}

        # Percentiles (0–100) for MinC/MaxC
        self.minc_pct = float(policy.get("minc_percentile", 95))
        self.maxc_pct = float(policy.get("maxc_percentile", 99))

        # Trim settings
        self.trim_top = float(policy.get("trim_top_ratio", 0.0))          # e.g., 0.02
        self.min_pos_for_trim = int(policy.get("min_pos_for_trim", 20))
        self.use_mad_fallback = bool(policy.get("use_mad_fallback", True))

        # Defaults and run-length quantiles
        self.alert_seconds_default = float(policy.get("alert_seconds_default", 10.0))
        self.error_seconds_default = float(policy.get("error_seconds_default", 3.0))
        self.alert_quantile = float(policy.get("alert_quantile", 0.90))  
        self.error_quantile = float(policy.get("error_quantile", 0.50))

    @staticmethod
    def _run_lengths(mask: np.ndarray) -> List[int]:
        """Return lengths of consecutive True segments."""
        if mask.size == 0: return []
        runs, r = [], 0
        for v in mask:
            if v: r += 1
            else:
                if r: runs.append(r)
                r = 0
        if r: runs.append(r)
        return runs

    def _trimmed_pos(self, pos: np.ndarray) -> np.ndarray:
        """Optionally trim the top fraction of positive residuals."""
        if len(pos) >= self.min_pos_for_trim and self.trim_top > 0:
            s = np.sort(pos)
            cut = max(1, int(np.floor(len(s) * (1 - self.trim_top))))
            return s[:cut]
        return pos

    def fit(self, df_pred: pd.DataFrame, axes: List[str], dt_seconds: float) -> None:
        step_from_sec = lambda s: max(1, int(round(s / max(dt_seconds, 1e-9))))
        default_alert_steps = step_from_sec(self.alert_seconds_default)
        default_error_steps = step_from_sec(self.error_seconds_default)

        for k in axes:
            res = df_pred[f"{k}_res"].values
            pos = res[res > 0]

            # MinC/MaxC with trim or robust fallback
            if len(pos) < self.min_pos_for_trim and self.use_mad_fallback and len(pos) > 0:
                # robust fallback if too few positives
                med = float(np.median(pos)); mad = float(np.median(np.abs(pos - med)))
                minc = med + 2.0 * mad
                maxc = med + 3.0 * mad
                method = "median+MAD"
            else:
                pos_eff = self._trimmed_pos(pos)
                minc = float(np.percentile(pos_eff, self.minc_pct)) if len(pos_eff) else 0.0
                maxc = float(np.percentile(pos_eff, self.maxc_pct)) if len(pos_eff) else 0.0
                method = f"trim_top={self.trim_top}"

            # Run-length distributions
            alert_mask = res >= minc
            error_mask = res >= maxc
            run_alert = self._run_lengths(alert_mask)
            run_error = self._run_lengths(error_mask)

            # Alerts: use high quantile (>= default)
            if run_alert:
                q_alert = int(np.ceil(np.quantile(run_alert, self.alert_quantile)))
                T_long_steps = max(default_alert_steps, q_alert)
            else:
                T_long_steps = default_alert_steps

            # Errors: use median (>= default)
            if run_error:
                q_error = int(np.ceil(np.quantile(run_error, self.error_quantile)))
                T_short_steps = max(default_error_steps, q_error)
            else:
                T_short_steps = default_error_steps

            self.th[k] = {
                "MinC": float(minc), "MaxC": float(maxc),
                "T_long_steps": int(T_long_steps), "T_short_steps": int(T_short_steps),
                "dt_seconds": float(dt_seconds),
                "meta": {"method": method}
            }

        with open(self.out_dir / "thresholds.json", "w") as f:
            json.dump(self.th, f, indent=2)
