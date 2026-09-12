"""Main training entry point for the BINN forage-biomass forecasting project.

For a given dataset (D1/D2/D3) and model type, trains 10 independent runs
using the fixed seeds in ``config.SEEDS`` and the dataset's tuned
hyperparameters, then reports:
    * Validation and test set metrics as Mean +/- Std across the 10 runs.
    * The validation R2 of the biophysics ("ModVege-style") sub-model, when
      applicable (BINN_R / BINN_C).
    * A single representative run -- selected as the run whose *validation*
      R2 is closest to the 10-run mean -- whose weights are saved to
      ``./checkpoints/``.

Test-set information never influences training or checkpoint selection: the
test split is evaluated exactly once per run, strictly after
``src.models.model_train`` has already restored its best-validation-loss
weights, purely for reporting.

Usage:
    python train.py --dataset D3 --model BINN_R
    python train.py --dataset all --model BINN_R --compare-baselines
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, TensorDataset

import config
from src import dataset, models, utils


def _resolve_run_hyperparams(
    model_type: str, dataset_cfg: config.DatasetConfig, **overrides: Any
) -> Dict[str, Any]:
    """Resolve the effective hyperparameters for one training run.

    Mirrors the original notebooks' default-resolution logic: BINN_C draws
    its ``alpha_``/``lr_bio`` defaults from ``dataset_cfg.binn_c_overrides``,
    while every other model type (including BINN_R) draws from
    ``dataset_cfg.hyper_params``. Explicit ``overrides`` (used by the
    ablation and sensitivity experiments) always take precedence.
    """
    extra_data = overrides.get("extra_data", config.FEATURE_COLUMNS)
    if model_type == "GC-LSTM":
        extra_data = extra_data + config.BIOVAR_COLUMNS

    physics_defaults = (
        dataset_cfg.binn_c_overrides if model_type == "BINN_C" else dataset_cfg.hyper_params
    )

    return {
        "extra_data": extra_data,
        "scaling_level": overrides.get("scaling_level", "paddock"),
        "is_conv": overrides.get("is_conv", True),
        "lookback": overrides.get("lookback", dataset_cfg.hyper_params["lookback"]),
        "alpha_": overrides.get("alpha_", physics_defaults["alpha_"]),
        "lr_bio": overrides.get("lr_bio", physics_defaults["lr_bio"]),
        "lr": overrides.get("lr", dataset_cfg.hyper_params["lr"]),
    }


def run_experiment(
    dataset_key: str,
    model_type: str,
    seeds: Optional[List[int]] = None,
    bio_param_overrides: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Train ``model_type`` on dataset ``dataset_key`` for every seed in ``seeds``.

    Args:
        bio_param_overrides: BINN_R-only overrides for the initial
            physiological parameter values (used by the biophysical
            sensitivity sweep).
        **overrides: Any of ``extra_data``, ``scaling_level``, ``is_conv``,
            ``lookback``, ``alpha_``, ``lr_bio``, ``lr`` -- used by the
            ablation and hyperparameter-sensitivity sweeps.

    Returns:
        A result dict with per-run metrics, Mean +/- Std summaries, and the
        representative run's weights/estimator, ready for
        ``save_checkpoint`` or further aggregation.
    """
    dataset_cfg = config.DATASET_CONFIGS[dataset_key]
    seeds = seeds or config.SEEDS
    run_params = _resolve_run_hyperparams(model_type, dataset_cfg, **overrides)
    is_dl = model_type in config.DL_MODEL_TYPES
    split = dataset_cfg.date_split

    df_obs, df_priors, df_biovars = dataset.load_raw_tables()
    active_paddocks, paddock_priors, num_paddocks, _, _ = dataset.load_data(
        df_obs, df_priors, df_biovars, dataset_cfg.year,
        is_conv=run_params["is_conv"], num_paddocks_to_keep=config.NUM_PADDOCKS_TO_KEEP,
    )

    val_metrics_per_run: List[tuple] = []
    test_metrics_per_run: List[tuple] = []
    val_biop_r2_per_run: List[Optional[float]] = []
    dl_state_dicts: List[Dict[str, torch.Tensor]] = []
    ml_estimators: List[Any] = []
    input_dim: Optional[int] = None

    for seed in seeds:
        utils.set_seed(seed)

        train_data, val_data, test_data, paddock_datasets = dataset.create_datasets(
            active_paddocks, lookback=run_params["lookback"], forecast=config.FORECAST_HORIZON,
            train_start=split.train_start, train_end=split.train_end,
            val_start=split.val_start, val_end=split.val_end,
            test_start=split.test_start, test_end=split.test_end,
            extra_data=run_params["extra_data"],
            scaling_level=run_params["scaling_level"], is_seq=is_dl,
        )

        X_val, y_val, sc_val, pid_val = dataset.get_eval_data(paddock_datasets, "val")
        X_test, y_test, sc_test, pid_test = dataset.get_eval_data(paddock_datasets, "test")

        if is_dl:
            input_dim = train_data[0].size(-1)
            bio_flag = model_type in ("BINN_R", "BINN_C")
            model = models.build_dl_model(
                model_type, input_dim, config.FORECAST_HORIZON, num_paddocks, dataset_cfg,
                paddock_priors=paddock_priors if model_type == "BINN_C" else None,
                bio_param_overrides=bio_param_overrides,
            )

            # --- Train (train + validation splits only; see module docstring) ---
            models.model_train(
                model, num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
                train_data=train_data, val_data=val_data, bio_flag=bio_flag,
                lr=run_params["lr"], lr_bio=run_params["lr_bio"], alpha_=run_params["alpha_"],
                is_printed=verbose, paddock_datasets=paddock_datasets, model_type=model_type,
            )

            # --- Post-training validation pass: recover the biophysics R2 of
            # the restored best checkpoint (honest use of val split, not a
            # training-loop leak). ---
            val_loader = DataLoader(TensorDataset(*val_data), batch_size=config.BATCH_SIZE, shuffle=False)
            _, _, val_biop_r2 = models.valid_one_epoch(
                model, val_loader, bio_flag, run_params["alpha_"], paddock_datasets,
            )

            # --- Single, post-hoc evaluation of val + test splits ---
            results, _, _ = utils.evaluate_model(
                model, [X_val, X_test], [y_val, y_test], ["Val", "Test"], [sc_val, sc_test],
                paddock_ids=[pid_val, pid_test], verbose=verbose,
            )
            dl_state_dicts.append(copy.deepcopy(model.state_dict()))
        else:
            model = models.build_ml_model(model_type)
            model.fit(train_data[0], train_data[1])
            results, _, _ = utils.evaluate_model(
                model, [X_val, X_test], [y_val, y_test], ["Val", "Test"], [sc_val, sc_test],
                verbose=verbose,
            )
            ml_estimators.append(copy.deepcopy(model))
            val_biop_r2 = None

        val_metrics_per_run.append(results[0])
        test_metrics_per_run.append(results[1])
        val_biop_r2_per_run.append(val_biop_r2)

    val_r2 = [m[0] for m in val_metrics_per_run]
    val_summary = utils.pack_mean_std(
        val_r2, [m[1] for m in val_metrics_per_run],
        [m[2] for m in val_metrics_per_run], [m[3] for m in val_metrics_per_run],
    )
    test_summary = utils.pack_mean_std(
        [m[0] for m in test_metrics_per_run], [m[1] for m in test_metrics_per_run],
        [m[2] for m in test_metrics_per_run], [m[3] for m in test_metrics_per_run],
    )

    # Representative run: closest to the mean *validation* R2. Selecting by
    # validation (rather than test) performance keeps the test split purely
    # observational, even at this post-hoc "which run to keep" stage.
    rep_idx = utils.pick_representative_run(val_r2)
    finite_biop = [v for v in val_biop_r2_per_run if v is not None]

    return {
        "dataset_key": dataset_key,
        "model_type": model_type,
        "seeds": seeds,
        "run_params": run_params,
        "input_dim": input_dim,
        "num_paddocks": num_paddocks,
        "is_dl": is_dl,
        "val_metrics_per_run": val_metrics_per_run,
        "test_metrics_per_run": test_metrics_per_run,
        "val_biop_r2_per_run": val_biop_r2_per_run,
        "val_biop_r2_mean": float(sum(finite_biop) / len(finite_biop)) if finite_biop else None,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "representative_index": rep_idx,
        "representative_seed": seeds[rep_idx],
        "representative_state_dict": dl_state_dicts[rep_idx] if is_dl else None,
        "representative_estimator": ml_estimators[rep_idx] if not is_dl else None,
    }


def save_checkpoint(result: Dict[str, Any]) -> None:
    """Persist the representative run's weights and a JSON metadata sidecar.

    Deep-learning checkpoints are saved as ``{dataset}_{model}.pt``
    (``state_dict``); classical ML estimators as ``{dataset}_{model}.pkl``.
    The metadata sidecar records everything ``eval_pretrained.py`` and
    ``visualization/plot_results.py`` need to rebuild the exact data
    pipeline and model architecture.
    """
    stem = f"{result['dataset_key']}_{result['model_type']}"
    if result["is_dl"]:
        torch.save(result["representative_state_dict"], config.CHECKPOINT_DIR / f"{stem}.pt")
    else:
        with open(config.CHECKPOINT_DIR / f"{stem}.pkl", "wb") as f:
            pickle.dump(result["representative_estimator"], f)

    metadata = {
        "dataset_key": result["dataset_key"],
        "model_type": result["model_type"],
        "seeds": result["seeds"],
        "run_params": result["run_params"],
        "input_dim": result["input_dim"],
        "num_paddocks": result["num_paddocks"],
        "representative_seed": result["representative_seed"],
        "val_summary": result["val_summary"],
        "test_summary": result["test_summary"],
        "val_biop_r2_mean": result["val_biop_r2_mean"],
    }
    with open(config.CHECKPOINT_DIR / f"{stem}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def print_summary(result: Dict[str, Any]) -> None:
    """Print the Val/Test Mean +/- Std summary table for one experiment."""
    v = result["val_summary"]
    t = result["test_summary"]
    print(f"\n[{result['dataset_key']} | {result['model_type']}] over {len(result['seeds'])} seeds:")
    print(f"  Val  R2={utils.format_mean_std(v[0], v[4])}  RMSE={utils.format_mean_std(v[1], v[5], 2)}  "
          f"MAE={utils.format_mean_std(v[2], v[6], 2)}  SMAPE={utils.format_mean_std(v[3], v[7])}")
    print(f"  Test R2={utils.format_mean_std(t[0], t[4])}  RMSE={utils.format_mean_std(t[1], t[5], 2)}  "
          f"MAE={utils.format_mean_std(t[2], t[6], 2)}  SMAPE={utils.format_mean_std(t[3], t[7])}")
    if result["val_biop_r2_mean"] is not None:
        print(f"  Val Biophysics-head R2 (mean): {result['val_biop_r2_mean']:.3f}")
    print(f"  Representative run: seed={result['representative_seed']} "
          f"(closest to mean Val R2) -> saved to ./checkpoints/{result['dataset_key']}_{result['model_type']}.*")


def _comparison_row(result: Dict[str, Any], p_value: str) -> Dict[str, Any]:
    t = result["test_summary"]
    return {
        "Dataset": result["dataset_key"],
        "Model": result["model_type"],
        "Test_R2": utils.format_mean_std(t[0], t[4]),
        "Test_RMSE": utils.format_mean_std(t[1], t[5], 2),
        "Test_MAE": utils.format_mean_std(t[2], t[6], 2),
        "Test_SMAPE": utils.format_mean_std(t[3], t[7]),
        "P_Value_vs_Reference": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BINN / baselines with 10 fixed seeds.")
    parser.add_argument("--dataset", choices=[*config.DATASET_KEYS, "all"], default="D3")
    parser.add_argument("--model", choices=config.ALL_MODEL_TYPES, default="BINN_R")
    parser.add_argument(
        "--compare-baselines", action="store_true",
        help="Also train every other model type and report a paired t-test of test-R2 against --model.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch training progress.")
    args = parser.parse_args()

    dataset_keys = config.DATASET_KEYS if args.dataset == "all" else [args.dataset]

    for dataset_key in dataset_keys:
        print(f"\n===== Dataset {dataset_key} (year {config.DATASET_CONFIGS[dataset_key].year}) =====")
        reference_result = run_experiment(dataset_key, args.model, verbose=args.verbose)
        print_summary(reference_result)
        save_checkpoint(reference_result)

        if args.compare_baselines:
            reference_test_r2 = [m[0] for m in reference_result["test_metrics_per_run"]]
            rows = [_comparison_row(reference_result, p_value="-")]

            for baseline_type in [m for m in config.ALL_MODEL_TYPES if m != args.model]:
                baseline_result = run_experiment(dataset_key, baseline_type, verbose=args.verbose)
                print_summary(baseline_result)
                save_checkpoint(baseline_result)
                _, p_value = ttest_rel(
                    reference_test_r2, [m[0] for m in baseline_result["test_metrics_per_run"]]
                )
                rows.append(_comparison_row(baseline_result, p_value=f"{p_value:.4f}"))

            out_path = config.RESULTS_DIR / f"model_comparison_{dataset_key}.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"\nSaved model comparison table to {out_path}")


if __name__ == "__main__":
    main()
