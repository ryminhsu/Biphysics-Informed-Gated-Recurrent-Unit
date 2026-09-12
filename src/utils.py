"""Reusable, dependency-light helpers: reproducibility, scaling, metrics.

This module sits at the bottom of the project's import graph (it does not
depend on ``src.dataset`` or ``src.models``) so that both of those modules
-- and any training/experiment script -- can share the same scaling and
evaluation logic without duplication.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ArrayLike = Union[np.ndarray, torch.Tensor, float]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed every RNG (``random``, ``numpy``, ``torch``) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# MinMax scaling helpers
# --------------------------------------------------------------------------- #
# The project exclusively uses MinMaxScaler(feature_range=(-1, 1)); the
# scaler's forward/inverse transform is reimplemented here (rather than
# calling back into the sklearn object) because the biophysics loss needs to
# apply it directly to torch tensors during training.
def minmax_scale(value: ArrayLike, value_min: ArrayLike, value_max: ArrayLike) -> ArrayLike:
    """Scale ``value`` from ``[value_min, value_max]`` to ``[-1, 1]``."""
    return 2 * (value - value_min) / (value_max - value_min) - 1


def minmax_inverse(value: ArrayLike, value_min: ArrayLike, value_max: ArrayLike) -> ArrayLike:
    """Invert :func:`minmax_scale`, mapping ``[-1, 1]`` back to original units."""
    return (value + 1) / 2 * (value_max - value_min) + value_min


# --------------------------------------------------------------------------- #
# Regression metrics
# --------------------------------------------------------------------------- #
def cal_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error, expressed as a percentage."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def model_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Return ``(R2, RMSE, MAE, SMAPE)`` for a set of predictions."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = cal_smape(y_true, y_pred)
    return r2, rmse, mae, smape


def print_metrics_table(results: Sequence[Tuple[float, float, float, float]], titles: Sequence[str]) -> None:
    """Pretty-print a list of ``(R2, RMSE, MAE, SMAPE)`` tuples."""
    table = PrettyTable()
    table.field_names = ["Set", "R2", "RMSE", "MAE", "SMAPE"]
    for (r2, rmse, mae, smape), title in zip(results, titles):
        table.add_row([title, round(r2, 4), round(rmse, 2), round(mae, 2), round(smape, 3)])
    print(table)


def evaluate_model(
    model,
    x_sets: Sequence[torch.Tensor],
    y_sets: Sequence[torch.Tensor],
    titles: Sequence[str],
    scalers: Sequence[Tuple[np.ndarray, np.ndarray]],
    is_normalized: bool = False,
    paddock_ids: Optional[Sequence[Optional[torch.Tensor]]] = None,
    verbose: bool = True,
) -> Tuple[List[Tuple[float, float, float, float]], List[np.ndarray], List[np.ndarray]]:
    """Evaluate a (DL or classical-ML) model on one or more data splits.

    This performs a single forward pass / ``predict`` call per split -- it is
    meant to be invoked *after* training completes (e.g. on the validation
    set for model selection, or once on the test set for final reporting),
    never inside the per-epoch training loop.

    Returns:
        results: ``(R2, RMSE, MAE, SMAPE)`` per split, in original units.
        y_true_list, y_pred_list: Inverse-transformed arrays per split.
    """
    is_pytorch = isinstance(model, torch.nn.Module)
    if is_pytorch:
        model.eval()

    results, y_true_list, y_pred_list = [], [], []
    p_ids = paddock_ids if paddock_ids is not None else [None] * len(x_sets)

    for X, y_true_raw, scaler_bounds, pid in zip(x_sets, y_sets, scalers, p_ids):
        if is_pytorch:
            with torch.no_grad():
                output = model(X, pid) if pid is not None else model(X)
                y_pred = output[0] if isinstance(output, (tuple, list)) else output
                y_pred = y_pred.cpu().numpy()
        else:
            y_pred = model.predict(X)

        y_pred = np.asarray(y_pred).reshape(-1, 1)
        y_true = np.asarray(y_true_raw).reshape(-1, 1)

        if not is_normalized:
            data_min, data_max = np.asarray(scaler_bounds[0]), np.asarray(scaler_bounds[1])
            y_pred = minmax_inverse(y_pred, data_min, data_max)
            y_true = minmax_inverse(y_true, data_min, data_max)

        metrics = model_eval(y_true, y_pred)
        results.append(metrics)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    if verbose:
        print_metrics_table(results, titles)

    return results, y_true_list, y_pred_list


# --------------------------------------------------------------------------- #
# Multi-seed aggregation helpers
# --------------------------------------------------------------------------- #
def pack_mean_std(
    r2: Sequence[float], rmse: Sequence[float], mae: Sequence[float], smape: Sequence[float]
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return ``(mean_r2, mean_rmse, mean_mae, mean_smape, std_r2, std_rmse, std_mae, std_smape)``."""
    return (
        float(np.mean(r2)), float(np.mean(rmse)), float(np.mean(mae)), float(np.mean(smape)),
        float(np.std(r2, ddof=1)), float(np.std(rmse, ddof=1)),
        float(np.std(mae, ddof=1)), float(np.std(smape, ddof=1)),
    )


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    """Format ``mean +/- std`` with a fixed number of decimal places."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def pick_representative_run(metric_values: Sequence[float]) -> int:
    """Return the index of the run whose metric is closest to the run mean.

    Used to select a single "typical" model (out of N independent seeded
    runs) to persist to disk / use for downstream visualization, without
    cherry-picking the best-performing run.
    """
    values = np.asarray(metric_values, dtype=float)
    mean_value = values.mean()
    return int(np.argmin(np.abs(values - mean_value)))
