# Linear Regression Architecture Workshop

# ML → MLOps Scaffold (DB → ML Box → Model Registry)

This repository implements the **ML block** in your diagram end‑to‑end:
1) **Data Extraction & Analysis** → 2) **Data Preparation** → 3) **Model Selection**
→ 4) **Model Training** → 5) **Model Evaluation & Validation** → 6) **Thresholding**
→ 7) **Model Registry** (versioned artifacts). Orchestration ties all steps together.

> **Main entry points**: `00_Main.ipynb` (Jupyter) or `python orchestrate.py --config config.yaml` (CLI).  
> All code and comments are in **English** as requested.

---

## Quick Start

```bash
# 1) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure connection & pipeline
#    - Copy .env.example → .env and set DB_URL
#    - Edit config.yaml (table/time/axes/window/threshold policy)

# 4) Run
python main.py                       # default config
python main.py --config path/to/config.yaml  # custom config
```

**Environment**  
`.env` must include `DB_URL` (SQLAlchemy string). Example for PostgreSQL:
```
postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
```

---

## Configuration (`config.yaml`)

```yaml
database:
  url_env: DB_URL            # env var to read DB URL

data:
  table: stream_samples      # table or view name
  time_col: ts               # time column (float seconds or TIMESTAMP)
  # Axes: choose ONE of the following modes
  # - WIDE schema: provide column names as strings
  #   axes: ["axis1","axis2","axis3","axis4","axis5","axis6","axis7","axis8"]
  # - LONG schema: provide axis IDs as numbers and the id/value columns
  #   axes: [1,2,3,4,5,6,7,8]
  axis_id_col: axis_id       # used only in LONG mode
  value_col: value           # used only in LONG mode

  # Training window (ISO strings if TIMESTAMP; numbers if float seconds)
  train_start: "2025-09-01T00:00:00"
  train_end:   "2025-09-05T00:00:00"

prep:
  interpolate: false         # keep false to exactly match the original notebook

model:
  type: linear               # linear regression per axis (y = a*t + b)

thresholds:
  # Residual → positive side percentiles (0–100)
  minc_percentile: 75
  maxc_percentile: 95

  # Trim the top fraction before computing percentiles (robust to outliers)
  trim_top_ratio: 0.02       # e.g., 2% of largest positive residuals removed
  min_pos_for_trim: 20       # trimming only if enough positive samples
  use_mad_fallback: true     # fallback to median+MAD when positives are too few

  # Default sustain durations (seconds) converted to steps with dt_seconds
  alert_seconds_default: 5.0
  error_seconds_default: 3.0

  # Run-length quantiles (0–1) to *raise above* defaults when sustained longer
  alert_quantile: 0.80       # Alerts require longer sustain (80th percentile)
  error_quantile: 0.50       # Errors use the median run-length

registry:
  root_dir: "ModelRegistry/artifacts"
  version_tag: "v1"

plots:
  dir: "out"
```

> **Tip:** For an 80/20 split like your original notebook, set `data.train_end` to the **80% sample-based timestamp**.  
> Example (ISO): `"2022-10-18T06:18:17.080Z"`.

---

## Folder‑by‑folder Responsibilities

```
project-root/
├─ DataExtractionAnalysis/
│  ├─ extractor.py  → DB read (SQLAlchemy). Supports WIDE (ts+axis1..N) or LONG (ts, axis_id, value → pivot to axis1..N).
│  └─ analyzer.py   → Basic profiling: median sampling interval (dt_seconds), row counts, time range.
│
├─ DataPreparation/
│  └─ preprocessor.py → Optional interpolation; derive normalized time axis `time_s` (epoch seconds, zeroed at start).
│
├─ ModelSelection/
│  └─ selector.py    → Choose model family. This scaffold implements `linear` (per-axis univariate regression).
│
├─ ModelTraining/
│  └─ trainer.py     → Fit per-axis LinearRegression (y = a*t + b). Excludes rows where that axis value == 0 (like the original notebook).
│                      Saves coefficients to `out/models.pkl`. Adds `{axis}_pred` and `{axis}_res` on predict().
│
├─ ModelEvaluation/
│  └─ evaluator.py   → Compute metrics (R², MAE, RMSE) and save plots.
│                      - Fit plot: scatter(data) + fitted line in different colors.
│                      - Residual plot: **scatter only** for residuals + dashed zero‑line in another color.
│                      Artifacts saved to `out/`.
│
├─ Thresholding/
│  └─ calibrator.py  → Positive residuals → (optional) trim top fraction → compute MinC/MaxC percentiles.
│                      Run-length policy (config‑driven):
│                        • Start from defaults (Alert=5s, Error=3s).
│                        • Alerts use 0.80‑quantile; Errors use median (0.50).
│                        • Raise T above defaults only if observed runs are longer.
│                      Saves `out/thresholds.json` per axis.
│
├─ ModelRegistry/
│  └─ registry.py    → Versioned save of artifacts to `ModelRegistry/artifacts/<version_tag>/`:
│                      `models.pkl`, `prep_stats.json`, `thresholds.json`, `metrics.csv`, `meta.yaml`.
│                      Maintains a `latest/` pointer (symlink or copy). 
│
├─ Orchestration/
│  └─ orchestrator.py → High‑level runner for the entire ML box.
│                       Robust UTF‑8 config loading (Windows safe). LONG/WIDE detection.
│                       Returns OrchestrationResult (shapes, metrics DF, thresholds dict, artifact dir).
│
├─ main.py            → calls the Orchestrator and shows metrics/thresholds.
├─ TrainedMachineLearningModel/ → Placeholder package for future utilities.
├─ out/               → Plots and temporary outputs.
└─ ModelRegistry/artifacts/ → Versioned registry and `latest/` pointer.
```

---

## Outputs & Where to Find Them

- **Plots**: `out/axis*_fit.png`, `out/axis*_residual.png`  
- **Metrics**: `ModelRegistry/artifacts/<version_tag>/metrics.csv`  
- **Thresholds**: `ModelRegistry/artifacts/<version_tag>/thresholds.json`  
- **Model**: `ModelRegistry/artifacts/<version_tag>/models.pkl`  
- **Metadata**: `ModelRegistry/artifacts/<version_tag>/meta.yaml`  
- **Latest pointer**: `ModelRegistry/artifacts/latest/`

Example thresholds entry:
```json
{
  "axis1": {
    "MinC": 7.31,
    "MaxC": 19.13,
    "T_long_steps": 3,
    "T_short_steps": 2,
    "dt_seconds": 1.891,
    "policy": {
      "minc_percentile": 75.0,
      "maxc_percentile": 95.0,
      "alert_seconds_default": 5.0,
      "error_seconds_default": 3.0,
      "alert_quantile": 0.8,
      "error_quantile": 0.5
    }
  }
}
```

---

## Reproducing Your Original Notebook

To align with the original assignment exactly:
- `prep.interpolate: false`  
- `thresholds.minc_percentile: 75`, `maxc_percentile: 95`  
- `thresholds.trim_top_ratio: 0.02`, `min_pos_for_trim: 20`, `use_mad_fallback: true`  
- `thresholds.alert_quantile: 0.80`, `error_quantile: 0.50`  
- `data.train_end`: 80% sample‑based split timestamp (you computed earlier)

The trainer already **excludes y==0 rows per axis**, matching your notebook behavior.

---

## Troubleshooting

- **UnicodeDecodeError on Windows**: the orchestrator reads YAML with UTF‑8/UTF‑8‑SIG. Ensure `config.yaml` is saved as UTF‑8.  
- **`squared=False` not supported**: Older scikit‑learn. This code computes RMSE as `sqrt(MSE)` for compatibility.  
- **Percentiles don’t match**: Verify the same window, interpolation flag, trimming settings, and LONG/WIDE mapping.  
- **DB query errors**: Ensure `.env` has a valid `DB_URL` and the table/column names in `config.yaml` are correct.

---

## Notes / Next Steps

- You can add a serving layer (e.g., FastAPI) that loads `latest/` artifacts and performs online residual + run‑length checks.
- For more accurate modeling, consider piecewise linear or polynomial fits, or model per operating mode.
