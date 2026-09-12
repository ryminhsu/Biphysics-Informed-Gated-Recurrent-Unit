"""Temporal true-vs-predicted visualizations for every dataset (D1 / D2 / D3).

Loads the representative BINN_R checkpoint trained by ``train.py`` for each
dataset (rebuilding the exact data pipeline from the checkpoint's JSON
metadata sidecar) and produces, under ``results/figures/``, a per-paddock
time-series line plot (best / median / worst test-R2 paddocks), shaded by
train/validation/test period -- the temporal-prediction figure used in the
paper.

Usage:
    python visualization/plot_results.py
    python visualization/plot_results.py --datasets D3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from src import dataset, models, utils

MODEL_TYPE = "BINN_R"


def _load_model_and_data(dataset_key: str):
    """Rebuild the data pipeline and load the trained checkpoint for ``dataset_key``."""
    dataset_cfg = config.DATASET_CONFIGS[dataset_key]
    meta_path = config.CHECKPOINT_DIR / f"{dataset_key}_{MODEL_TYPE}.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {dataset_key}/{MODEL_TYPE}. "
            f"Run `python train.py --dataset {dataset_key} --model {MODEL_TYPE}` first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    run_params = meta["run_params"]
    split = dataset_cfg.date_split

    df_obs, df_priors, df_biovars = dataset.load_raw_tables()
    active_paddocks, _, num_paddocks, _, _ = dataset.load_data(
        df_obs, df_priors, df_biovars, dataset_cfg.year,
        is_conv=run_params["is_conv"], num_paddocks_to_keep=config.NUM_PADDOCKS_TO_KEEP,
    )

    _, _, _, paddock_datasets = dataset.create_datasets(
        active_paddocks, lookback=run_params["lookback"], forecast=config.FORECAST_HORIZON,
        train_start=split.train_start, train_end=split.train_end,
        val_start=split.val_start, val_end=split.val_end,
        test_start=split.test_start, test_end=split.test_end,
        extra_data=run_params["extra_data"],
        scaling_level=run_params["scaling_level"], is_seq=True,
    )

    model = models.build_dl_model(MODEL_TYPE, meta["input_dim"], config.FORECAST_HORIZON, num_paddocks, dataset_cfg)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"{dataset_key}_{MODEL_TYPE}.pt", map_location="cpu"))
    model.eval()

    return model, paddock_datasets


def _predict_split(model, paddock_datasets: Dict, pid: int, split: str) -> Tuple[np.ndarray, np.ndarray, list]:
    split_data = paddock_datasets[pid][split]
    X, y = split_data["X"], split_data["y"]
    dates = pd.to_datetime(split_data["date"]).tolist()
    if len(y) == 0:
        return np.array([]), np.array([]), dates

    with torch.no_grad():
        y_pred, _ = model(X, torch.full((len(y),), pid, dtype=torch.long))
    d_min, d_max = dataset.get_label_bounds(paddock_datasets, pid)
    y_pred_inv = dataset.inverse_transform_label(y_pred.numpy().reshape(-1, 1), d_min, d_max).flatten()
    y_true_inv = dataset.inverse_transform_label(y.numpy().reshape(-1, 1), d_min, d_max).flatten()
    return y_true_inv, y_pred_inv, dates


def plot_paddock_timeseries(dataset_key: str) -> Path:
    """Save a 3-panel (best/median/worst paddock) true-vs-predicted line plot."""
    model, paddock_datasets = _load_model_and_data(dataset_key)

    paddock_r2 = []
    for pid in paddock_datasets:
        y_true, y_pred, _ = _predict_split(model, paddock_datasets, pid, "test")
        if len(y_true) == 0:
            continue
        r2, _, _, _ = utils.model_eval(y_true, y_pred)
        paddock_r2.append((pid, r2))
    paddock_r2.sort(key=lambda t: t[1])

    worst, best = paddock_r2[0], paddock_r2[-1]
    median_r2 = float(np.median([r for _, r in paddock_r2]))
    median = min(paddock_r2, key=lambda t: abs(t[1] - median_r2))
    targets = [best, median, worst]
    titles = ["(A) Best Paddock", "(B) Median Paddock", "(C) Worst Paddock"]
    split_colors = {"train": "#F1F5F9", "val": "#E0F2FE", "test": "#FEF3C7"}

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), dpi=150)
    for ax, (pid, r2), title in zip(axes, targets, titles):
        all_dates, all_true, all_pred = [], [], []
        for split in ("train", "val", "test"):
            y_true, y_pred, dates = _predict_split(model, paddock_datasets, pid, split)
            if len(dates) == 0:
                continue
            ax.axvspan(dates[0], dates[-1], color=split_colors[split], alpha=0.6, zorder=0)
            all_dates.extend(dates)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        ax.plot(all_dates, all_true, color="grey", label="Observed AGB", linewidth=1.2, zorder=2)
        ax.plot(all_dates, all_pred, color="#2E86C1", label=f"{MODEL_TYPE} Prediction", linewidth=1.2, zorder=3)
        ax.set_title(f"{title} (Test R2={r2:.3f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("AGB")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{dataset_key}: Representative Paddock Predictions ({MODEL_TYPE})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = config.FIGURES_DIR / f"timeseries_{dataset_key}_{MODEL_TYPE}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot temporal predictions for one or more datasets.")
    parser.add_argument("--datasets", nargs="+", choices=config.DATASET_KEYS, default=config.DATASET_KEYS)
    args = parser.parse_args()

    for dataset_key in args.datasets:
        try:
            print(f"Saved {plot_paddock_timeseries(dataset_key)}")
        except FileNotFoundError as exc:
            print(f"[Skip {dataset_key}] {exc}")


if __name__ == "__main__":
    main()
