# Save the model as a file. (Parts not implemented in existing tasks)
# In existing tasks, it was saved as a variable such as model, not a file.

from pathlib import Path
from joblib import dump
import yaml, shutil, datetime as dt

class ModelRegistry:
    """Stores versioned ML artifacts and maintains a `latest` pointer."""
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, version_tag: str, models_pkl_path: str,
             prep_stats_path: str, thresholds_path: str,
             metrics_df, meta: dict) -> None:
        vdir = self.root / version_tag
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir(parents=True)

        # Copy artifacts
        shutil.copy2(models_pkl_path, vdir / "models.pkl")
        shutil.copy2(prep_stats_path,  vdir / "prep_stats.json")
        shutil.copy2(thresholds_path,  vdir / "thresholds.json")
        metrics_df.to_csv(vdir / "metrics.csv", index=False)

        # Meta
        meta = dict(meta)
        meta.setdefault("saved_at", dt.datetime.utcnow().isoformat() + "Z")
        with open(vdir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

        # Update 'latest' pointer (symlink if possible, else copy)
        latest = self.root / "latest"
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                shutil.rmtree(latest, ignore_errors=True)
        try:
            latest.symlink_to(vdir, target_is_directory=True)
        except Exception:
            # Platforms without symlink permissions
            shutil.copytree(vdir, latest)
