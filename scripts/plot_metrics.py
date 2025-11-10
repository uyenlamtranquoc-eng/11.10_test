import os
import sys
import glob
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt


def find_run_dirs(root: str) -> List[str]:
    pattern = os.path.join(root, "*", "train_metrics.csv")
    files = glob.glob(pattern)
    return sorted(list({os.path.dirname(p) for p in files}))


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def plot_run(df: pd.DataFrame, run_dir: str) -> None:
    df = ensure_columns(df, [
        "ep",
        "reward",
        "avg_queue_veh",
        "throughput_veh_per_hour",
        "avg_delay_norm",
        "avg_stops_norm",
        "avg_speed_fluct_norm",
    ])

    ep = df["ep"].values if "ep" in df.columns else list(range(len(df)))

    figs = [
        ("reward", "Episode Reward"),
        ("avg_queue_veh", "Avg Queue (veh)"),
        ("throughput_veh_per_hour", "Throughput (veh/h)"),
        ("avg_delay_norm", "Avg Delay (norm)"),
        ("avg_stops_norm", "Avg Stops (norm)"),
        ("avg_speed_fluct_norm", "Avg Speed Fluct. (norm)"),
    ]

    for col, title in figs:
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(ep, df[col].values, label=col, color="#1f77b4")
            plt.xlabel("Episode")
            plt.ylabel(title)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(run_dir, f"{col}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"[plot] saved {out_path}")
        except Exception as e:
            print(f"[warn] plotting {run_dir} {col} failed: {e}")


def summarize_runs(run_dirs: List[str], out_dir: str) -> None:
    rows: List[Dict] = []
    for rd in run_dirs:
        csv_path = os.path.join(rd, "train_metrics.csv")
        df = safe_read_csv(csv_path)
        if df.empty:
            continue
        df = ensure_columns(df, [
            "reward",
            "avg_queue_veh",
            "throughput_veh_per_hour",
            "avg_delay_norm",
            "avg_stops_norm",
            "avg_speed_fluct_norm",
        ])
        try:
            rows.append({
                "run_dir": os.path.basename(rd),
                "episodes": int(len(df)),
                "reward_mean": float(df["reward"].mean()),
                "queue_mean": float(df["avg_queue_veh"].mean()),
                "throughput_mean": float(df["throughput_veh_per_hour"].mean()),
                "delay_mean": float(df["avg_delay_norm"].mean()),
                "stops_mean": float(df["avg_stops_norm"].mean()),
                "smooth_mean": float(df["avg_speed_fluct_norm"].mean()),
            })
        except Exception as e:
            print(f"[warn] summarize {rd} failed: {e}")

    if not rows:
        print("[summary] no rows, skip summary table")
        return

    out_path = os.path.join(out_dir, "summary_table.csv")
    try:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[summary] saved {out_path}")
    except Exception as e:
        print(f"[warn] writing summary table failed: {e}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "outputs")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"[error] outputs dir not found: {root}")
        sys.exit(1)
    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"[info] no runs found under {root}")
        sys.exit(0)
    for rd in run_dirs:
        csv_path = os.path.join(rd, "train_metrics.csv")
        df = safe_read_csv(csv_path)
        if df.empty:
            continue
        plot_run(df, rd)
    summarize_runs(run_dirs, root)


if __name__ == "__main__":
    main()