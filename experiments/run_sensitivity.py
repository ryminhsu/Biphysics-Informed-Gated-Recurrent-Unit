"""Sensitivity analysis for BINN_R, restricted to dataset D3 (2024 test split).

Two independent sweeps around the D3 tuned baseline, each run for the fixed
10 seeds (``config.SEEDS``):
    1. Hyperparameter sensitivity: lookback / alpha_ / lr_bio / lr
       (``config.HYPERPARAM_SENSITIVITY_GRID``).
    2. Biophysical-parameter sensitivity: every ModVege-style physiological
       parameter perturbed by +/- 20% (``config.BIO_PARAM_PERTURBATION_PCT``).

Outputs two CSVs plus a two-panel tornado plot under ``results/``.

Usage:
    python experiments/run_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from train import run_experiment

DATASET_KEY: str = config.SENSITIVITY_DATASET_KEY  # "D3"


def _impact_pct(value: float, baseline: float) -> float:
    return (value - baseline) / baseline * 100 if baseline else 0.0


def run_hyperparameter_sensitivity(baseline_r2: float) -> pd.DataFrame:
    """Sweep each hyperparameter in ``config.HYPERPARAM_SENSITIVITY_GRID`` independently."""
    rows = []
    for param, values in config.HYPERPARAM_SENSITIVITY_GRID.items():
        for value in values:
            print(f"[{DATASET_KEY}] Hyperparameter sweep: {param} = {value}")
            result = run_experiment(DATASET_KEY, "BINN_R", **{param: value})
            r2 = result["test_summary"][0]
            rows.append({
                "Parameter": param,
                "Tested_Value": value,
                "Mean_Test_R2": f"{r2:.3f}",
                "R2_Delta": f"{r2 - baseline_r2:+.3f}",
                "R2_Impact_%": f"{_impact_pct(r2, baseline_r2):+.1f}%",
            })
    return pd.DataFrame(rows)


def run_bio_param_sensitivity(baseline_r2: float) -> pd.DataFrame:
    """Perturb each BINN_R physiological parameter by +/- ``config.BIO_PARAM_PERTURBATION_PCT``."""
    base_bio_params = config.DATASET_CONFIGS[DATASET_KEY].bio_params
    pct = config.BIO_PARAM_PERTURBATION_PCT
    rows = []

    for param, base_value in base_bio_params.items():
        low = round(base_value * (1 - pct), 4)
        high = round(base_value * (1 + pct), 4)
        for direction, value in [(f"Low (-{pct:.0%})", low), (f"High (+{pct:.0%})", high)]:
            print(f"[{DATASET_KEY}] Biophysical sweep: {param} {direction} = {value}")
            result = run_experiment(DATASET_KEY, "BINN_R", bio_param_overrides={param: value})
            r2 = result["test_summary"][0]
            rows.append({
                "Parameter": param,
                "Direction": direction,
                "Baseline_Value": base_value,
                "Tested_Value": value,
                "Mean_Test_R2": f"{r2:.3f}",
                "R2_Delta": f"{r2 - baseline_r2:+.3f}",
                "R2_Impact_%": f"{_impact_pct(r2, baseline_r2):+.1f}%",
            })
    return pd.DataFrame(rows)


def plot_tornado(df_hyper: pd.DataFrame, df_bio: pd.DataFrame, out_path: Path) -> None:
    """Save a two-panel tornado plot summarizing both sensitivity sweeps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    for ax, df, title in [(ax1, df_hyper, "Hyperparameters"), (ax2, df_bio, "Biophysical Parameters")]:
        impacts = df.copy()
        impacts["Impact"] = impacts["R2_Impact_%"].str.rstrip("%").astype(float)
        summary = (
            impacts.groupby("Parameter")["Impact"]
            .agg(["min", "max"])
            .assign(max_abs=lambda d: d[["min", "max"]].abs().max(axis=1))
            .sort_values("max_abs")
        )
        y_pos = np.arange(len(summary))
        ax.barh(y_pos, summary["min"].clip(upper=0), color="#E63946", alpha=0.85, label="Negative impact")
        ax.barh(y_pos, summary["max"].clip(lower=0), color="#457B9D", alpha=0.85, label="Positive impact")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Test R2 impact (%)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"BINN_R Sensitivity Analysis ({DATASET_KEY}, test split)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    baseline = run_experiment(DATASET_KEY, "BINN_R")
    baseline_r2 = baseline["test_summary"][0]
    print(f"[{DATASET_KEY}] Baseline BINN_R Test R2: {baseline_r2:.3f}")

    df_hyper = run_hyperparameter_sensitivity(baseline_r2)
    hyper_path = config.RESULTS_DIR / f"sensitivity_hyperparameters_{DATASET_KEY}.csv"
    df_hyper.to_csv(hyper_path, index=False)
    print(f"Saved {hyper_path}")

    df_bio = run_bio_param_sensitivity(baseline_r2)
    bio_path = config.RESULTS_DIR / f"sensitivity_bio_params_{DATASET_KEY}.csv"
    df_bio.to_csv(bio_path, index=False)
    print(f"Saved {bio_path}")

    figure_path = config.FIGURES_DIR / f"sensitivity_tornado_{DATASET_KEY}.png"
    plot_tornado(df_hyper, df_bio, figure_path)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
