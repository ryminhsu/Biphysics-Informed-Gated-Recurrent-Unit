"""Data loading, cleaning, and time-series windowing for the BINN pipeline.

The pipeline turns three raw CSV tables (daily/15-day climate observations,
calibrated ModVege biophysical priors, and ModVege state variables) into
paddock-wise sliding-window tensors ready for sequence models (BINN, GRU,
GC-LSTM, ...) or flattened feature matrices ready for classical ML models.

All scalers are fit on the training split only, per paddock (or globally,
for the "w/o Paddock-wise Scaling" ablation), to prevent any leakage from
the validation/test windows into the feature normalization statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import config
from src.utils import minmax_inverse

DateWindow = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Raw table loading
# --------------------------------------------------------------------------- #
def load_raw_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw CSV inputs referenced in ``config.py``.

    Returns:
        (df_observations, df_calibrated_priors, df_biophysical_variables)
    """
    df_obs = pd.read_csv(config.OBSERVATION_CSV)
    df_priors = pd.read_csv(config.CALIBRATED_PRIORS_CSV)
    df_biovars = pd.read_csv(config.BIOPHYSICAL_VARIABLES_CSV)
    df_obs["OBSERVATION_DATE"] = pd.to_datetime(df_obs["OBSERVATION_DATE"])
    df_biovars["OBSERVATION_DATE"] = pd.to_datetime(df_biovars["OBSERVATION_DATE"])
    return df_obs, df_priors, df_biovars


# --------------------------------------------------------------------------- #
# Paddock-level assembly
# --------------------------------------------------------------------------- #
def load_data(
    df_obs: pd.DataFrame,
    df_priors: pd.DataFrame,
    df_biovars: pd.DataFrame,
    target_year: int,
    is_conv: bool = True,
    num_paddocks_to_keep: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], int, LabelEncoder, List]:
    """Merge climate observations with calibrated biophysical priors and state variables.

    Args:
        df_obs: Raw per-paddock climate/AGB observation table.
        df_priors: Calibrated ModVege physiological parameters per
            paddock/year (``config.BIOPARAM_COLUMNS``).
        df_biovars: ModVege state variables per paddock/date
            (``config.BIOVAR_COLUMNS``), consumed by the GC-LSTM baseline.
        target_year: Year used to look up the calibrated priors
            (``df_priors["YEAR"] == target_year``); for D1/D2/D3 this always
            matches the rolling-origin test year.
        is_conv: If True, convert RADIATION (MJ/m^2, all-wave) to PAR
            (MJ/m^2) via the standard 0.45 conversion factor.
        num_paddocks_to_keep: Optionally subsample to the first N unique
            paddock IDs (used to build the fixed 94-paddock cohort).

    Returns:
        active_paddocks: Long-format frame with one row per paddock/date.
        paddock_priors: ``{bioparam_name: np.ndarray[num_paddocks]}`` used to
            initialize BINN_C's per-paddock physiological embeddings.
        num_paddocks: Number of unique paddocks retained after merging.
        paddock_encoder: Fitted ``LabelEncoder`` mapping PADDOCK_ID -> index.
        paddock_id_list: Original PADDOCK_ID strings, ordered by index.
    """
    df_obs = df_obs.copy()
    df_obs["OBSERVATION_DATE"] = pd.to_datetime(df_obs["OBSERVATION_DATE"])

    # Exclude 2017 and drop the first (partial/warm-up) 2018 observation per paddock.
    df_obs = df_obs[df_obs["OBSERVATION_DATE"].dt.year >= 2018].reset_index(drop=True)
    mask_2018 = df_obs["OBSERVATION_DATE"].dt.year == 2018
    first_2018_indices = (
        df_obs[mask_2018].groupby("PADDOCK_ID")["OBSERVATION_DATE"].idxmin().values
    )
    df_obs = df_obs.drop(first_2018_indices).reset_index(drop=True)

    if num_paddocks_to_keep is not None:
        unique_paddocks = df_obs["PADDOCK_ID"].unique()[:num_paddocks_to_keep]
        df_obs = df_obs[df_obs["PADDOCK_ID"].isin(unique_paddocks)].reset_index(drop=True)

    df_priors_year = df_priors[df_priors["YEAR"] == target_year].copy()

    merged_df = df_obs.merge(df_priors_year, how="inner", on="PADDOCK_ID")
    merged_df = merged_df.merge(df_biovars, how="inner", on=["PADDOCK_ID", "OBSERVATION_DATE"])

    paddock_encoder = LabelEncoder()
    merged_df["PADDOCK_ID_ENCODED"] = paddock_encoder.fit_transform(merged_df["PADDOCK_ID"])
    num_paddocks = len(paddock_encoder.classes_)
    paddock_id_list = paddock_encoder.classes_.tolist()

    if is_conv:
        merged_df["RADIATION"] *= 0.45
        merged_df["15D_AVG_RADIATION"] *= 0.45

    train_cols = (
        config.LABEL_COLUMN
        + config.BIOFEATURE_COLUMNS
        + config.FEATURE_COLUMNS
        + config.EXTRA_DATA_COLUMNS
        + config.BIOVAR_COLUMNS
        + ["OBSERVATION_DATE", "PADDOCK_ID", "PADDOCK_ID_ENCODED", "SumT"]
    )
    active_paddocks = merged_df[train_cols].copy()

    df_paddock_priors = (
        merged_df[["PADDOCK_ID_ENCODED"] + config.BIOPARAM_COLUMNS]
        .groupby("PADDOCK_ID_ENCODED")
        .first()
        .sort_index()
    )
    paddock_priors = {
        col: df_paddock_priors[col].values.astype(np.float32)
        for col in config.BIOPARAM_COLUMNS
    }

    return active_paddocks, paddock_priors, num_paddocks, paddock_encoder, paddock_id_list


# --------------------------------------------------------------------------- #
# Sliding-window construction
# --------------------------------------------------------------------------- #
def prepare_lstm_data(
    data: Sequence[np.ndarray], lookback: int, forecast: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a fixed-size window over a single paddock's time series.

    Args:
        data: ``(feature_matrix, biophysics_matrix)`` for one paddock, each
            shaped ``(num_timesteps, num_columns)``. ``feature_matrix``
            column 0 must be the label column.
        lookback: Number of past timesteps fed to the model.
        forecast: Number of future timesteps to predict.

    Returns:
        X: ``(num_windows, lookback, num_features)``
        y: ``(num_windows, forecast)`` -- label column only.
        bp: ``(num_windows, lookback, num_bio_columns)``
    """
    features, bio = data
    X, y, bp = [], [], []
    for i in range(len(features) - lookback - forecast + 1):
        X.append(features[i : i + lookback])
        y.append(features[i + lookback : i + lookback + forecast, 0])
        bp.append(bio[i : i + lookback])
    return np.array(X), np.array(y), np.array(bp)


def date_mask(dates: pd.Series, start: DateWindow, end: DateWindow) -> pd.Series:
    """Boolean mask selecting dates within ``[start, end]`` (year, month) bounds."""
    start_date = pd.Timestamp(year=start[0], month=start[1], day=1)
    end_date = pd.Timestamp(year=end[0], month=end[1], day=1) + pd.offsets.MonthEnd(0)
    return (dates >= start_date) & (dates <= end_date)


def data_preprocess(
    group: pd.DataFrame,
    input_features: List[str],
    train_start: DateWindow,
    train_end: DateWindow,
    val_start: DateWindow,
    val_end: DateWindow,
    test_start: DateWindow,
    test_end: DateWindow,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Fit a MinMaxScaler on a single paddock's training window and apply it.

    The scaler is fit exclusively on rows inside ``[train_start, train_end]``
    to prevent validation/test statistics from leaking into normalization.
    """
    train_mask = date_mask(group["OBSERVATION_DATE"], train_start, train_end)

    scaler = MinMaxScaler(feature_range=config.SCALE_FEATURE_RANGE)
    scaler.fit(group.loc[train_mask, input_features])
    group[input_features] = scaler.transform(group[input_features])
    return group.sort_values("OBSERVATION_DATE"), scaler


PaddockSplit = Dict[str, torch.Tensor]
PaddockDatasets = Dict[int, Dict[str, object]]


def create_datasets(
    active_paddocks: pd.DataFrame,
    lookback: int,
    forecast: int,
    train_start: DateWindow,
    train_end: DateWindow,
    val_start: DateWindow,
    val_end: DateWindow,
    test_start: DateWindow,
    test_end: DateWindow,
    extra_data: Optional[List[str]] = None,
    scaling_level: str = "paddock",
    is_seq: bool = True,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    PaddockDatasets,
]:
    """Build global train/val/test tensors plus a per-paddock dataset index.

    Args:
        scaling_level: ``"paddock"`` fits one scaler per paddock (default);
            ``"global"`` fits a single scaler across all paddocks using the
            pooled training window (the "w/o Paddock-wise Scaling" ablation).
        is_seq: If False, flattens each window to ``(lookback * num_features,)``
            for classical ML models that do not consume sequences.

    Returns:
        train_data, val_data, test_data: Each a ``(X, y, bio_params)`` tuple
            of tensors pooled across all paddocks.
        paddock_datasets: ``{paddock_id_encoded: {"scaler", "train", "val",
            "test", "all"}}`` used for paddock-wise inverse-scaling and
            evaluation/visualization.
    """
    extra_data = extra_data or []
    df = active_paddocks.copy()
    input_features = config.LABEL_COLUMN + config.BIOFEATURE_COLUMNS + extra_data

    global_scaler: Optional[MinMaxScaler] = None
    if scaling_level == "global":
        train_mask_all = date_mask(df["OBSERVATION_DATE"], train_start, train_end)
        global_scaler = MinMaxScaler(feature_range=config.SCALE_FEATURE_RANGE)
        global_scaler.fit(df.loc[train_mask_all, input_features])

    x_train_all, y_train_all, bp_train_all = [], [], []
    x_val_all, y_val_all, bp_val_all = [], [], []
    x_test_all, y_test_all, bp_test_all = [], [], []
    paddock_datasets: PaddockDatasets = {}

    for paddock_id, group in df.groupby("PADDOCK_ID_ENCODED"):
        group = group.reset_index(drop=True)

        if scaling_level == "paddock":
            processed_group, current_scaler = data_preprocess(
                group, input_features, train_start, train_end,
                val_start, val_end, test_start, test_end,
            )
        else:
            group[input_features] = global_scaler.transform(group[input_features])
            processed_group = group.sort_values("OBSERVATION_DATE")
            current_scaler = global_scaler

        cm_data = processed_group[input_features].values
        bp_data = processed_group[["SumT", "PADDOCK_ID_ENCODED"]].values

        X, y, bp_seq = prepare_lstm_data([cm_data, bp_data], lookback, forecast)
        seq_start_dates = processed_group["OBSERVATION_DATE"].iloc[lookback : lookback + len(X)]

        if not is_seq:
            X = X.reshape(X.shape[0], -1)

        t_mask = date_mask(seq_start_dates, train_start, train_end)
        v_mask = date_mask(seq_start_dates, val_start, val_end)
        s_mask = date_mask(seq_start_dates, test_start, test_end)

        paddock_datasets[paddock_id] = {
            "scaler": current_scaler,
            "train": {
                "X": torch.tensor(X[t_mask], dtype=torch.float32),
                "y": torch.tensor(y[t_mask], dtype=torch.float32),
                "date": seq_start_dates[t_mask],
            },
            "val": {
                "X": torch.tensor(X[v_mask], dtype=torch.float32),
                "y": torch.tensor(y[v_mask], dtype=torch.float32),
                "date": seq_start_dates[v_mask],
            },
            "test": {
                "X": torch.tensor(X[s_mask], dtype=torch.float32),
                "y": torch.tensor(y[s_mask], dtype=torch.float32),
                "date": seq_start_dates[s_mask],
            },
            "all": {
                "X": torch.tensor(X, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "date": seq_start_dates,
            },
        }

        if any(t_mask):
            x_train_all.append(X[t_mask]); y_train_all.append(y[t_mask]); bp_train_all.append(bp_seq[t_mask])
        if any(v_mask):
            x_val_all.append(X[v_mask]); y_val_all.append(y[v_mask]); bp_val_all.append(bp_seq[v_mask])
        if any(s_mask):
            x_test_all.append(X[s_mask]); y_test_all.append(y[s_mask]); bp_test_all.append(bp_seq[s_mask])

    train_data = (
        torch.tensor(np.concatenate(x_train_all), dtype=torch.float32),
        torch.tensor(np.concatenate(y_train_all), dtype=torch.float32),
        torch.tensor(np.concatenate(bp_train_all), dtype=torch.float32),
    )
    val_data = (
        torch.tensor(np.concatenate(x_val_all), dtype=torch.float32),
        torch.tensor(np.concatenate(y_val_all), dtype=torch.float32),
        torch.tensor(np.concatenate(bp_val_all), dtype=torch.float32),
    )
    test_data = (
        torch.tensor(np.concatenate(x_test_all), dtype=torch.float32),
        torch.tensor(np.concatenate(y_test_all), dtype=torch.float32),
        torch.tensor(np.concatenate(bp_test_all), dtype=torch.float32),
    )

    return train_data, val_data, test_data, paddock_datasets


# --------------------------------------------------------------------------- #
# Evaluation-time data extraction
# --------------------------------------------------------------------------- #
def get_eval_data(
    paddock_datasets: PaddockDatasets, split: str
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[np.ndarray, np.ndarray], torch.Tensor]:
    """Concatenate a given split ("train"/"val"/"test") across all paddocks.

    Returns:
        X, y: Pooled tensors for the requested split.
        (data_min, data_max): Per-row MinMaxScaler bounds for the label
            column, used to inverse-transform predictions back to the
            original AGB units.
        paddock_ids: Encoded paddock ID for every row (needed by BINN_C,
            which looks up per-paddock physiological embeddings).
    """
    X_list, y_list, paddock_ids, data_min, data_max = [], [], [], [], []

    for pid, data in paddock_datasets.items():
        X_list.append(data[split]["X"])
        y_list.append(data[split]["y"])
        scaler = data["scaler"]
        n_rows = data[split]["y"].shape[0]
        paddock_ids.extend(np.repeat(pid, n_rows))
        data_min.extend(np.repeat(scaler.data_min_[0], n_rows))
        data_max.extend(np.repeat(scaler.data_max_[0], n_rows))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    paddock_ids_tensor = torch.tensor(paddock_ids)
    data_min_arr = np.array(data_min)[:, np.newaxis]
    data_max_arr = np.array(data_max)[:, np.newaxis]

    return X, y, (data_min_arr, data_max_arr), paddock_ids_tensor


def get_label_bounds(paddock_datasets: PaddockDatasets, paddock_id: int) -> Tuple[float, float]:
    """Return ``(data_min, data_max)`` of the label column for one paddock."""
    scaler = paddock_datasets[paddock_id]["scaler"]
    return float(scaler.data_min_[0]), float(scaler.data_max_[0])


def inverse_transform_label(values: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    """Undo MinMax scaling for a paddock's label column."""
    return minmax_inverse(values, data_min, data_max)
