"""Ablation study: isolate each BINN_R pipeline mechanism, on all datasets.

For every dataset (D1, D2, D3), trains the tuned BINN_R baseline plus every
ablation variant in ``config.ABLATION_VARIANTS`` for the fixed 10 seeds
(``config.SEEDS``), then reports the test-R2 Mean +/- Std delta and relative
contribution of each mechanism, saved to ``results/``.

Bug fix vs. the original notebooks: the "w/o Radiation Conversion" ablation
(``is_conv=False``) was previously read into a local variable but never
actually forwarded to ``load_data(...)``, so it silently had zero effect on
any reported result. ``train.run_experiment`` threads ``is_conv`` end-to-end,
so this ablation now behaves as originally intended.

Usage:
    python experiments/run_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from train import run_experiment


def run_ablation_for_dataset(dataset_key: str) -> pd.DataFrame:
    """Run the BINN_R baseline and every ablation variant for one dataset."""
    baseline = run_experiment(dataset_key, "BINN_R")
    baseline_r2, baseline_r2_std = baseline["test_summary"][0], baseline["test_summary"][4]
    print(f"[{dataset_key}] Baseline BINN_R Test R2: {baseline_r2:.3f} +/- {baseline_r2_std:.3f}")

    rows = [{
        "Dataset": dataset_key,
        "Ablated Module": "None (Full BINN_R)",
        "Test_R2": f"{baseline_r2:.3f} +/- {baseline_r2_std:.3f}",
        "R2_Delta": "+0.000",
        "Contribution_%": "0.0%",
    }]

    for name, overrides in config.ABLATION_VARIANTS:
        print(f"[{dataset_key}] Evaluating ablation: {name}")
        result = run_experiment(dataset_key, "BINN_R", **overrides)
        r2, r2_std = result["test_summary"][0], result["test_summary"][4]
        delta = r2 - baseline_r2
        contribution_pct = (delta / baseline_r2) * 100 if baseline_r2 else 0.0
        rows.append({
            "Dataset": dataset_key,
            "Ablated Module": name,
            "Test_R2": f"{r2:.3f} +/- {r2_std:.3f}",
            "R2_Delta": f"{delta:+.3f}",
            "Contribution_%": f"{contribution_pct:+.1f}%",
        })

    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for dataset_key in config.DATASET_KEYS:
        df = run_ablation_for_dataset(dataset_key)
        out_path = config.RESULTS_DIR / f"ablation_{dataset_key}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        frames.append(df)

    combined_path = config.RESULTS_DIR / "ablation_all_datasets.csv"
    pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
    print(f"Saved combined ablation table to {combined_path}")


if __name__ == "__main__":
    main()
