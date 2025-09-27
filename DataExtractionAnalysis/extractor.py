from dataclasses import dataclass
from typing import List, Sequence
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

@dataclass
class DBConfig:
    """Configuration to locate the DB URL in environment variables."""
    url_env: str

class DBExtractor:
    """Loads a time window of data from a relational DB using SQLAlchemy.
    Supports both WIDE and LONG schemas.

    - WIDE:  time_col + axes columns (e.g., axis1..axis8)
    - LONG:  time_col, axis_id_col, value_col  → pivoted to WIDE in pandas
    """
    def __init__(self, cfg: DBConfig):
        load_dotenv()
        url = os.getenv(cfg.url_env)
        if not url:
            raise RuntimeError(f"Environment variable {cfg.url_env} is not set.")
        self.engine = create_engine(url)

    def load_window(self, table: str, time_col: str, axes: Sequence,
                    start_val, end_val,
                    axis_id_col: str | None = None,
                    value_col: str | None = None) -> pd.DataFrame:
        """Load a time slice.
        If `axes` are strings → treat as WIDE schema.
        If `axes` are numbers → treat as LONG schema and pivot to WIDE.
        `start_val` / `end_val` may be numeric (for float ts) or ISO strings (for TIMESTAMP).
        """
        # Detect schema mode from the type of `axes` entries
        wide_mode = all(isinstance(a, str) for a in axes)
        long_mode = all(isinstance(a, (int, float)) for a in axes)

        if wide_mode:
            cols = ", ".join([time_col] + list(axes))
            q = f"""            SELECT {cols}
            FROM {table}
            WHERE {time_col} >= {self._literal(start_val)} AND {time_col} < {self._literal(end_val)}
            ORDER BY {time_col} ASC
            """
            df = pd.read_sql_query(q, self.engine, parse_dates=[time_col], coerce_float=True)
            return df

        if long_mode:
            if not axis_id_col or not value_col:
                raise ValueError("axis_id_col and value_col must be provided for LONG schema.")
            cols = ", ".join([time_col, axis_id_col, value_col])
            q = f"""            SELECT {cols}
            FROM {table}
            WHERE {time_col} >= {self._literal(start_val)} AND {time_col} < {self._literal(end_val)}
            ORDER BY {time_col} ASC, {axis_id_col} ASC
            """
            df_long = pd.read_sql_query(q, self.engine, coerce_float=True)
            # Pivot to WIDE
            df_wide = df_long.pivot(index=time_col, columns=axis_id_col, values=value_col).reset_index()
            # Ensure consistent column names: axis1..axisN
            rename_map = {aid: f"axis{int(aid)}" for aid in df_wide.columns if aid != time_col}
            df_wide = df_wide.rename(columns=rename_map)
            # Sort columns by axis order
            axis_names = [f"axis{int(a)}" for a in axes]
            return df_wide[[time_col] + axis_names]

        raise ValueError("`axes` must be either all strings (WIDE) or all numbers (LONG).")

    def _literal(self, v):
        """Best-effort SQL literal for numbers or ISO strings."""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
        # Assume string; wrap in single quotes
        return f"'{v}'"
