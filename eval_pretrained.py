"""Quick evaluation of an already-trained checkpoint on the held-out test split.

Rebuilds the exact data pipeline used at training time (from the checkpoint's
JSON metadata sidecar), loads the saved weights/estimator from
``./checkpoints/``, and reports test-set metrics. No training happens here.

Usage:
    python eval_pretrained.py --dataset D3 --model BINN_R
"""

from __future__ import annotations

import argparse
import json
import pickle
from typing import Any, Dict, Tuple

import torch

import config
from src import dataset, models, utils


def _load_metadata(dataset_key: str, model_type: str) -> Dict[str, Any]:
    meta_path = config.CHECKPOINT_DIR / f"{dataset_key}_{model_type}.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No checkpoint metadata at {meta_path}. "
            f"Run `python train.py --dataset {dataset_key} --model {model_type}` first."
        )
    with open(meta_path) as f:
        return json.load(f)


def evaluate_pretrained(
    dataset_key: str, model_type: str, verbose: bool = True
) -> Tuple[float, float, float, float]:
    """Load the saved checkpoint for (dataset_key, model_type) and evaluate it on the test split.

    Returns:
        ``(R2, RMSE, MAE, SMAPE)`` computed on the test split, in original AGB units.
    """
    dataset_cfg = config.DATASET_CONFIGS[dataset_key]
    meta = _load_metadata(dataset_key, model_type)
    run_params = meta["run_params"]
    is_dl = model_type in config.DL_MODEL_TYPES
    split = dataset_cfg.date_split

    df_obs, df_priors, df_biovars = dataset.load_raw_tables()
    active_paddocks, paddock_priors, num_paddocks, _, _ = dataset.load_data(
        df_obs, df_priors, df_biovars, dataset_cfg.year,
        is_conv=run_params["is_conv"], num_paddocks_to_keep=config.NUM_PADDOCKS_TO_KEEP,
    )

    _, _, _, paddock_datasets = dataset.create_datasets(
        active_paddocks, lookback=run_params["lookback"], forecast=config.FORECAST_HORIZON,
        train_start=split.train_start, train_end=split.train_end,
        val_start=split.val_start, val_end=split.val_end,
        test_start=split.test_start, test_end=split.test_end,
        extra_data=run_params["extra_data"],
        scaling_level=run_params["scaling_level"], is_seq=is_dl,
    )
    X_test, y_test, sc_test, pid_test = dataset.get_eval_data(paddock_datasets, "test")

    stem = f"{dataset_key}_{model_type}"
    if is_dl:
        model = models.build_dl_model(
            model_type, meta["input_dim"], config.FORECAST_HORIZON, num_paddocks, dataset_cfg,
            paddock_priors=paddock_priors if model_type == "BINN_C" else None,
        )
        model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"{stem}.pt", map_location="cpu"))
        results, _, _ = utils.evaluate_model(
            model, [X_test], [y_test], ["Test"], [sc_test],
            paddock_ids=[pid_test], verbose=verbose,
        )
    else:
        with open(config.CHECKPOINT_DIR / f"{stem}.pkl", "rb") as f:
            model = pickle.load(f)
        results, _, _ = utils.evaluate_model(
            model, [X_test], [y_test], ["Test"], [sc_test], verbose=verbose,
        )

    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the test split.")
    parser.add_argument("--dataset", choices=config.DATASET_KEYS, default="D3")
    parser.add_argument("--model", choices=config.ALL_MODEL_TYPES, default="BINN_R")
    args = parser.parse_args()
    evaluate_pretrained(args.dataset, args.model)


if __name__ == "__main__":
    main()
