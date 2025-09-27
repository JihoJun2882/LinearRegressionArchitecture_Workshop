# main.py — entry point with a tiny CLI
import argparse
from Orchestration.orchestrator import Orchestrator

#CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    orch = Orchestrator(args.config)
    res = orch.run_all()

    # Summary
    print("Artifacts:", res.artifact_dir)
    print(f"Raw: {res.raw_shape[0]}x{res.raw_shape[1]}  |  Train: {res.train_shape[0]}x{res.train_shape[1]}")

    # Metrics (one line per axis)
    print("\nMetrics:")
    for row in res.metrics.to_dict(orient="records"):
        print(f"  {row['axis']}: R2={row['r2']:.3f}  MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}")

    # Thresholds (first 8)
    print("\nThresholds (first 8):")
    for i, (axis, v) in enumerate(sorted(res.thresholds.items())):
        if i >= 8: break
        print(f"  {axis}: MinC={v['MinC']:.3f}  MaxC={v['MaxC']:.3f}  "
              f"T_long={v['T_long_steps']} steps  T_short={v['T_short_steps']} steps")