# This function profiles a time column by converting it to seconds (handling either datetime or numeric).
# Computing the median inter-sample interval (dt_seconds).
# It returns an EDAResult containing that interval, the number of rows, and the earliest/latest timestamps as strings.

import pandas as pd
from dataclasses import dataclass
import pandas.api.types as ptypes

@dataclass
class EDAResult:
    dt_seconds: float
    n_rows: int
    head_ts: str
    tail_ts: str

class Analyzer:
    def basic_profile(self, df: pd.DataFrame, time_col: str) -> EDAResult:
        if ptypes.is_datetime64_any_dtype(df[time_col]):
            ts_sec = df[time_col].view('int64') / 1e9
        else:
            ts_sec = pd.to_numeric(df[time_col], errors='coerce')
        if len(ts_sec) < 2:
            return EDAResult(0.0, len(df), str(df[time_col].min()), str(df[time_col].max()))
        dt = float(ts_sec.diff().dropna().median())
        return EDAResult(dt_seconds=dt, n_rows=len(df),
                         head_ts=str(df[time_col].min()), tail_ts=str(df[time_col].max()))
