"""Global configuration for the BINN forage biomass forecasting project.

This module is the single source of truth for:
    * filesystem paths (raw data, checkpoints, results)
    * reproducibility settings (fixed seeds)
    * the shared feature/column schema
    * per-dataset (D1/D2/D3) temporal splits and tuned hyperparameters

D1, D2 and D3 correspond to the three annual rolling-origin evaluation
windows that were previously implemented as three near-duplicate notebooks
(BINN-2022.ipynb, BINN-2023.ipynb, BINN-2024.ipynb). Only the *best*
(already fine-tuned) hyperparameters are kept here; the Optuna search
procedures that originally produced them have been removed from this
open-source release.
"""

from __future__ import annotations

import os

# PyTorch and XGBoost each bundle their own OpenMP runtime; on macOS in
# particular, importing both in the same process can otherwise abort with a
# native OpenMP "duplicate runtime" crash. These must be set before
# ``xgboost`` is first imported anywhere in the project (``config`` is
# imported first by every entry point, so this is early enough).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# Raw CSV inputs (identical across D1/D2/D3; only the temporal split differs).
OBSERVATION_CSV: Path = DATA_DIR / "nrm1010_phalaris_20122025.csv"
CALIBRATED_PRIORS_CSV: Path = (
    DATA_DIR / "calibrated_biophyscial_properties(phalaris with GREEN)_15D_Cross.csv"
)
BIOPHYSICAL_VARIABLES_CSV: Path = DATA_DIR / "paddock_biophysical_variables_2024 (All).csv"

for _dir in (CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# 10 fixed seeds used for every repeated-run experiment (training, ablation,
# sensitivity analysis) so that results are exactly reproducible.
SEEDS: List[int] = list(range(42, 52))
NUM_RUNS: int = len(SEEDS)

# --------------------------------------------------------------------------- #
# Shared feature / column schema
# --------------------------------------------------------------------------- #
# 15-day rolling-average climate drivers (satellite/paddock level).
FEATURE_COLUMNS: List[str] = [
    "15D_AVG_DAILY_RAIN",
    "15D_AVG_MAX_TEMP",
    "15D_AVG_MIN_TEMP",
    "15D_AVG_RADIATION",
    "15D_AVG_ET_TALL_CROP",
    "15D_AVG_RH_TMAX",
    "15D_AVG_RH_TMIN",
]
# Daily climate drivers used by the biophysics (ModVege-style) growth module.
BIOFEATURE_COLUMNS: List[str] = [
    "DAILY_RAIN",
    "MAX_TEMP",
    "MIN_TEMP",
    "RADIATION",
    "ET_TALL_CROP",
    "RH_TMAX",
    "RH_TMIN",
]
# Calibrated ModVege physiological parameters (paddock-wise priors).
BIOPARAM_COLUMNS: List[str] = [
    "RUEmax",
    "alpha_PAR",
    "T0",
    "T1",
    "T2",
    "SLA",
    "K_DV",
    "gammaGV",
    "percentLAM",
    "ST1",
    "ST2",
    "minSEA",
    "maxSEA",
    "NI",
    "WHC",
    "WR",
]
# Auxiliary ModVege state variables, only consumed by the GC-LSTM baseline.
BIOVAR_COLUMNS: List[str] = ["GRO", "PGRO", "SEN", "GV_AGE", "ENV"]
EXTRA_DATA_COLUMNS: List[str] = ["NONGREEN"]
LABEL_COLUMN: List[str] = ["GREEN"]

# Fixed cohort size: the first N unique paddock IDs in the raw CSV.
NUM_PADDOCKS_TO_KEEP: int = 94

# --------------------------------------------------------------------------- #
# Shared training / evaluation settings
# --------------------------------------------------------------------------- #
FORECAST_HORIZON: int = 1
NUM_EPOCHS: int = 12
BATCH_SIZE: int = 128
SCALE_FEATURE_RANGE: Tuple[float, float] = (-1.0, 1.0)  # MinMaxScaler range


DateWindow = Tuple[int, int]


@dataclass(frozen=True)
class DateSplit:
    """(year, month) boundaries -- inclusive -- for train/val/test windows."""

    train_start: DateWindow
    train_end: DateWindow
    val_start: DateWindow
    val_end: DateWindow
    test_start: DateWindow
    test_end: DateWindow


@dataclass(frozen=True)
class BaselineArchParams:
    """Tuned architecture hyperparameters for every competing baseline."""

    bilstm: Dict[str, Any]
    gc_lstm: Dict[str, Any]
    cnn1d: Dict[str, Any]
    rnn_tcn: Dict[str, Any]


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that varies between D1 / D2 / D3."""

    key: str
    year: int  # rolling-origin target (= test) year; also the ModVege priors lookup year
    date_split: DateSplit
    model_params: Dict[str, Any]       # BINN GRU backbone (hidden size / dropout)
    hyper_params: Dict[str, Any]       # lookback, alpha_, lr, lr_bio for BINN_R
    bio_params: Dict[str, Any]         # BINN_R physiological initial values
    binn_c_overrides: Dict[str, Any]   # alpha_, lr_bio, age, wr, gt for BINN_C
    baselines: BaselineArchParams


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "D1": DatasetConfig(
        key="D1",
        year=2022,
        date_split=DateSplit(
            train_start=(2018, 1), train_end=(2020, 12),
            val_start=(2021, 1), val_end=(2021, 12),
            test_start=(2022, 1), test_end=(2022, 12),
        ),
        model_params={"hidden_size_1": 16, "dropout_rate": 0.10},
        hyper_params={"lookback": 3, "alpha_": 1.0, "lr_bio": 0.005, "lr": 0.004},
        bio_params={
            "RUEmax": 2.6, "alpha_PAR": 0.045, "T0": 4.8, "T1": 14.4, "T2": 24.0,
            "SLA": 0.025, "K_DV": 0.0016, "gammaGV": 0.4, "percentLAM": 0.68,
            "ST1": 700, "ST2": 1350, "maxSEA": 1.3, "NI": 1.0,
            "age": 1.1, "wr": 0.6, "gt": 0.51,
        },
        binn_c_overrides={"alpha_": 1.0, "lr_bio": 0.015, "age": 1.1, "wr": 0.72, "gt": 0.52},
        baselines=BaselineArchParams(
            bilstm={"hidden_size_1": 32, "dropout_rate": 0.11},
            gc_lstm={
                "init_w": 0.8, "climate_encoder_size": 20, "bio_encoder_size": 8,
                "embed_dim": 24, "fc_hidden_size": 12, "dropout_rate": 0.05,
            },
            cnn1d={
                "out_channels_1": 16, "out_channels_2": 48, "out_channels_3": 40,
                "kernel_size": 2, "fc_hidden_size": 20, "dropout_rate": 0.10,
            },
            rnn_tcn={
                "lstm_hidden": 24, "tcn_channels": [32, 32], "kernel_size": 2,
                "fc_hidden": 12, "dropout": 0.05,
            },
        ),
    ),
    "D2": DatasetConfig(
        key="D2",
        year=2023,
        date_split=DateSplit(
            train_start=(2018, 1), train_end=(2021, 12),
            val_start=(2022, 1), val_end=(2022, 12),
            test_start=(2023, 1), test_end=(2023, 12),
        ),
        model_params={"hidden_size_1": 26, "dropout_rate": 0.11},
        hyper_params={"lookback": 10, "alpha_": 0.4, "lr_bio": 0.02, "lr": 0.01},
        bio_params={
            "RUEmax": 2.85, "alpha_PAR": 0.045, "T0": 4.7, "T1": 14.4, "T2": 27.7,
            "SLA": 0.033, "K_DV": 0.0011, "gammaGV": 0.42, "percentLAM": 0.75,
            "ST1": 700, "ST2": 1620, "maxSEA": 1.3, "NI": 1.03,
            "age": 1.09, "wr": 0.68, "gt": 1.49,
        },
        binn_c_overrides={"alpha_": 0.4, "lr_bio": 0.012, "age": 1.8, "wr": 0.5, "gt": 0.8},
        baselines=BaselineArchParams(
            bilstm={"hidden_size_1": 12, "dropout_rate": 0.07},
            gc_lstm={
                "init_w": 0.95, "climate_encoder_size": 24, "bio_encoder_size": 4,
                "embed_dim": 24, "fc_hidden_size": 12, "dropout_rate": 0.09,
            },
            cnn1d={
                "out_channels_1": 18, "out_channels_2": 16, "out_channels_3": 56,
                "kernel_size": 3, "fc_hidden_size": 24, "dropout_rate": 0.05,
            },
            rnn_tcn={
                "lstm_hidden": 32, "tcn_channels": [28, 28], "kernel_size": 3,
                "fc_hidden": 24, "dropout": 0.18,
            },
        ),
    ),
    "D3": DatasetConfig(
        key="D3",
        year=2024,
        date_split=DateSplit(
            train_start=(2018, 1), train_end=(2022, 12),
            val_start=(2023, 1), val_end=(2023, 12),
            test_start=(2024, 1), test_end=(2024, 12),
        ),
        model_params={"hidden_size_1": 28, "dropout_rate": 0.10},
        hyper_params={"lookback": 4, "alpha_": 0.6, "lr_bio": 0.005, "lr": 0.006},
        bio_params={
            "RUEmax": 2.7, "alpha_PAR": 0.045, "T0": 4.0, "T1": 12.0, "T2": 20.0,
            "SLA": 0.03, "K_DV": 0.0029, "gammaGV": 0.4, "percentLAM": 0.68,
            "ST1": 700, "ST2": 1620, "maxSEA": 1.3, "NI": 0.9,
            "age": 1.32, "wr": 0.16, "gt": 1.44,
        },
        binn_c_overrides={"alpha_": 0.6, "lr_bio": 0.005, "age": 2.88, "wr": 0.27, "gt": 0.9},
        baselines=BaselineArchParams(
            bilstm={"hidden_size_1": 44, "dropout_rate": 0.15},
            gc_lstm={
                "init_w": 0.4, "climate_encoder_size": 28, "bio_encoder_size": 4,
                "embed_dim": 20, "fc_hidden_size": 12, "dropout_rate": 0.05,
            },
            cnn1d={
                "out_channels_1": 18, "out_channels_2": 20, "out_channels_3": 52,
                "kernel_size": 3, "fc_hidden_size": 28, "dropout_rate": 0.10,
            },
            rnn_tcn={
                "lstm_hidden": 12, "tcn_channels": [16, 16], "kernel_size": 2,
                "fc_hidden": 16, "dropout": 0.10,
            },
        ),
    ),
}

DATASET_KEYS: List[str] = list(DATASET_CONFIGS.keys())

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# Deep-learning model identifiers understood by ``src.models.build_dl_model``.
DL_MODEL_TYPES: List[str] = [
    "BINN_R", "BINN_C", "GRU", "GC-LSTM", "RNN-TCN", "1D-CNN", "BI-LSTM",
]
# Classical machine-learning baselines (scikit-learn / XGBoost).
ML_MODEL_TYPES: List[str] = ["lr", "rf", "xgb", "bagging"]
ALL_MODEL_TYPES: List[str] = DL_MODEL_TYPES + ML_MODEL_TYPES

# --------------------------------------------------------------------------- #
# Ablation study configuration (``experiments/run_ablation.py``)
# --------------------------------------------------------------------------- #
# Each entry perturbs a single mechanism of the BINN_R pipeline relative to
# its tuned baseline configuration.
ABLATION_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("w/o Biophysics-Driven Optimization", {"lr_bio": 0.0}),
    ("w/o Paddock-wise Scaling", {"scaling_level": "global"}),
    ("w/o Radiation Conversion", {"is_conv": False}),
    ("w/o 15-Day Aggregated Climate Features", {"extra_data": []}),
]

# --------------------------------------------------------------------------- #
# Sensitivity analysis configuration (``experiments/run_sensitivity.py``)
# --------------------------------------------------------------------------- #
# Restricted, per the project scope, to dataset D3 (2024 test split).
SENSITIVITY_DATASET_KEY: str = "D3"

HYPERPARAM_SENSITIVITY_GRID: Dict[str, List[float]] = {
    "lookback": [2, 4, 6, 8, 10, 12],
    "alpha_": [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0],
    "lr_bio": [0.003, 0.005, 0.01, 0.012, 0.015, 0.02],
    "lr": [0.002, 0.004, 0.006, 0.008, 0.01, 0.015],
}

# Relative perturbation applied to each biophysical parameter (+/- 20%).
BIO_PARAM_PERTURBATION_PCT: float = 0.20
