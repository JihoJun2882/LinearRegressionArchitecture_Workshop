import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas.api.types as ptypes

@dataclass
class PrepStats:
    time0: float
    def to_json(self): return asdict(self)

class Preprocessor:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stats: PrepStats | None = None

    def fit_transform(self, df: pd.DataFrame, time_col: str, axes: list[str], interpolate: bool = True) -> pd.DataFrame:
        df = df.copy()
        if interpolate:
            df[axes] = df[axes].interpolate(limit_direction="both")
        if ptypes.is_datetime64_any_dtype(df[time_col]):
            t = df[time_col].view('int64') / 1e9
        else:
            t = pd.to_numeric(df[time_col], errors='coerce')
        t0 = float(t.iloc[0])
        df['time_s'] = t - t0
        self.stats = PrepStats(time0=t0)
        with open(self.out_dir / 'prep_stats.json', 'w') as f:
            json.dump(self.stats.to_json(), f, indent=2)
        return df[['time_s', *axes]]
