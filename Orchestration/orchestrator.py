import os, yaml, pandas as pd
import io
from dataclasses import dataclass
from typing import Any, Dict

from DataExtractionAnalysis.extractor import DBExtractor, DBConfig
from DataExtractionAnalysis.analyzer  import Analyzer
from DataPreparation.preprocessor     import Preprocessor
from ModelSelection.selector          import ModelSelector
from ModelTraining.trainer            import LinearAxisTrainer
from ModelEvaluation.evaluator        import Evaluator
from Thresholding.calibrator          import ThresholdCalibrator
from ModelRegistry.registry           import ModelRegistry

@dataclass
class OrchestrationResult:
    raw_shape: tuple
    train_shape: tuple
    metrics: pd.DataFrame
    thresholds: Dict[str, Any]
    artifact_dir: str

class Orchestrator:
    """High-level pipeline runner that mirrors the ML box in the diagram."""
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with io.open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback for UTF-8 with BOM or mixed encodings
            with io.open(config_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

        self.cfg = yaml.safe_load(content)

        self.DB_URL_ENV = self.cfg["database"]["url_env"]
        self.TABLE      = self.cfg["data"]["table"]
        self.TIME_COL   = self.cfg["data"]["time_col"]
        self.AXES       = self.cfg["data"]["axes"]
        self.T0         = self.cfg["data"]["train_start"]
        self.T1         = self.cfg["data"]["train_end"]
        self.AXIS_ID    = self.cfg["data"].get("axis_id_col")
        self.VALUE_COL  = self.cfg["data"].get("value_col")
        self.TH_POLICY  = self.cfg.get("thresholds", {})

        self.PLOT_DIR   = self.cfg["plots"]["dir"]
        self.REG_DIR    = self.cfg["registry"]["root_dir"]
        self.VER_TAG    = self.cfg["registry"]["version_tag"]
        self.MODEL_TYPE = self.cfg["model"]["type"]

    def run_all(self) -> OrchestrationResult:
        # 1) Extract
        ext = DBExtractor(DBConfig(url_env=self.DB_URL_ENV))
        axis_mode_long = all(isinstance(a, (int, float)) for a in self.AXES)
        if axis_mode_long:
            raw = ext.load_window(self.TABLE, self.TIME_COL, self.AXES, self.T0, self.T1,
                                  axis_id_col=self.AXIS_ID, value_col=self.VALUE_COL)
        else:
            raw = ext.load_window(self.TABLE, self.TIME_COL, self.AXES, self.T0, self.T1)

        ana = Analyzer()
        eda = ana.basic_profile(raw, self.TIME_COL)

        # 2) Prepare
        prep = Preprocessor(out_dir="out")
        train_df = prep.fit_transform(raw, self.TIME_COL,
                                      [c for c in raw.columns if c != self.TIME_COL],
                                      interpolate=self.cfg["prep"]["interpolate"])

        # 3) Select
        ms = ModelSelector()
        choice = ms.choose(self.MODEL_TYPE)

        # 4) Train
        trainer = LinearAxisTrainer(out_dir="out")
        axes_cols = [c for c in train_df.columns if c != "time_s"]
        trainer.fit(train_df, axes_cols)
        pred_train = trainer.predict(train_df)

        # 5) Evaluate
        ev = Evaluator(plot_dir=self.PLOT_DIR)
        metrics = ev.metrics(pred_train, axes_cols)
        ev.plots(pred_train, axes_cols)

        # 6) Threshold
        cal = ThresholdCalibrator(out_dir="out", policy=self.TH_POLICY)
        cal.fit(pred_train, axes_cols, dt_seconds=eda.dt_seconds)

        # 7) Registry
        reg = ModelRegistry(root_dir=self.REG_DIR)
        meta = {
            "model": choice.name,
            "data": {"table": self.TABLE, "time_range": [self.T0, self.T1], "axes": list(self.AXES)},
            "prep": {"interpolate": self.cfg["prep"]["interpolate"]},
        }
        reg.save(
            version_tag=self.VER_TAG,
            models_pkl_path="out/models.pkl",
            prep_stats_path="out/prep_stats.json",
            thresholds_path="out/thresholds.json",
            metrics_df=metrics,
            meta=meta,
        )
        return OrchestrationResult(
            raw_shape=raw.shape,
            train_shape=train_df.shape,
            metrics=metrics,
            thresholds=cal.th,
            artifact_dir=os.path.join(self.REG_DIR, self.VER_TAG),
        )
