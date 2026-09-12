"""Model architectures, the biophysics-informed loss, and the training loop.

Contains:
    * BINN -- the proposed Biophysics-Informed Neural Network. It combines a
      GRU backbone with a per-paddock, ModVege-inspired growth/senescence
      module. Two variants are supported for ablation purposes:
        - "BINN_R": physiological parameters share one global, randomly
          (or default-) initialized embedding across all paddocks.
        - "BINN_C": physiological parameters are initialized per-paddock
          from field-calibrated ModVege priors.
    * Six literature baselines (GRU, BiLSTM, GC-LSTM, 1D-CNN, RNN-TCN) and
      four classical ML baselines (Linear Regression, Random Forest,
      Bagging, XGBoost).
    * ``build_dl_model`` / ``build_ml_model``: ablation-aware factories that
      resolve a model_type + ``config.DatasetConfig`` into a ready-to-train
      model instance.
    * The physics-informed loss (``complete_biophysics_loss``) and the
      shared train/validate loop (``train_one_epoch``, ``valid_one_epoch``,
      ``model_train``).

Strict train/val/test separation: ``model_train`` only ever consumes the
train and validation splits. Test-set evaluation happens exactly once,
after training has finished, in the calling script (``train.py``).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import config
from src.utils import minmax_inverse, minmax_scale

# --------------------------------------------------------------------------- #
# Classical ML baselines
# --------------------------------------------------------------------------- #
def get_linear_reg() -> LinearRegression:
    return LinearRegression()


def get_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=800, learning_rate=0.04, max_depth=3,
        subsample=0.8, colsample_bytree=1.0, reg_alpha=3.9, reg_lambda=1.0,
    )


def get_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=750, max_depth=13, min_samples_split=11,
        min_samples_leaf=2, max_features="sqrt",
    )


def get_bagging() -> BaggingRegressor:
    base = DecisionTreeRegressor(
        max_depth=12, min_samples_split=4, min_samples_leaf=6, random_state=42
    )
    return BaggingRegressor(
        estimator=base, n_estimators=50, max_samples=0.5,
        max_features=0.9, n_jobs=None, random_state=42,
    )


def build_ml_model(model_type: str):
    """Factory for classical ML baselines: 'lr' | 'rf' | 'xgb' | 'bagging'."""
    return {"lr": get_linear_reg, "rf": get_rf, "xgb": get_xgb, "bagging": get_bagging}[model_type]()


# --------------------------------------------------------------------------- #
# Literature baseline architectures
# --------------------------------------------------------------------------- #
class GRU(nn.Module):
    """Plain 2-layer GRU regressor (no biophysics module)."""

    def __init__(self, input_size: int, forecast: int, hidden_size_1: int = 24, dropout_rate: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size_1, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size_1, forecast)

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out), None


class BiLSTM(nn.Module):
    """Bi-directional stacked LSTM.

    Reference: Joshi et al. (2025), "An explainable Bi-LSTM model for winter
    wheat yield prediction." Frontiers in Plant Science, 15, 1491493.
    """

    def __init__(self, input_size: int, forecast: int = 1, hidden_size_1: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size_1 * 2, hidden_size_1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size_1 * 2, forecast)

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        out, _ = self.bilstm1(x)
        out, _ = self.bilstm2(out)
        last = self.dropout(out[:, -1, :])
        return self.fc(last), None


def _logit(p: float, eps: float = 1e-6) -> float:
    import math

    return math.log(p / (1 - p))


class GC_LSTM(nn.Module):
    """Gated climate/biology dual-encoder LSTM with a learned mixing weight.

    Reference: Srivastava et al. (2022), "Winter wheat yield prediction with
    genotype and weather data..." (dual-branch architecture adaptation).
    """

    def __init__(
        self, input_size: int, forecast: int, init_w: float,
        climate_encoder_size: int = 24, bio_encoder_size: int = 10, embed_dim: int = 24,
        fc_hidden_size: int = 14, dropout_rate: float = 0.1, climate_input_size: int = 15,
    ):
        super().__init__()
        self.climate_input_size = climate_input_size
        self.climate_lstm = nn.LSTM(climate_input_size, climate_encoder_size, batch_first=True)
        self.bio_lstm = nn.LSTM(input_size - climate_input_size, bio_encoder_size, batch_first=True)
        self.climate_proj = nn.Linear(climate_encoder_size, embed_dim)
        self.bio_proj = nn.Linear(bio_encoder_size, embed_dim)
        self.drop_climate = nn.Dropout(dropout_rate)
        self.drop_bio = nn.Dropout(dropout_rate)
        self.act = nn.ELU()
        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_size), nn.Linear(fc_hidden_size, forecast)
        )
        self.alpha = nn.Parameter(torch.tensor(_logit(init_w), dtype=torch.float32))

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        climate = x[:, :, : self.climate_input_size]
        bio = x[:, :, self.climate_input_size :]
        c_seq, _ = self.climate_lstm(climate)
        b_seq, _ = self.bio_lstm(bio)
        c_feat = self.drop_climate(self.act(self.climate_proj(c_seq[:, -1, :])))
        b_feat = self.drop_bio(self.act(self.bio_proj(b_seq[:, -1, :])))
        y_c, y_b = self.shared_head(c_feat), self.shared_head(b_feat)
        w = torch.sigmoid(self.alpha)
        return w * y_c + (1.0 - w) * y_b, None


class MultiChannelClimateCNN(nn.Module):
    """1D-CNN over stacked climate channels.

    Reference: Srivastava et al. (2022), "Winter wheat yield prediction
    using convolutional neural networks from environmental and phenological
    data." Scientific Reports, 12(1), 3215.
    """

    def __init__(
        self, in_channels: int = 15, forecast: int = 1, out_channels_1: int = 16,
        out_channels_2: int = 32, out_channels_3: int = 48, kernel_size: int = 3,
        fc_hidden_size: int = 24, dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.cnn_features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_1, kernel_size, padding="same"), nn.ReLU(),
            nn.Conv1d(out_channels_1, out_channels_2, kernel_size, padding="same"), nn.ReLU(),
            nn.Conv1d(out_channels_2, out_channels_3, kernel_size, padding="same"), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.final_fc = nn.Sequential(
            nn.Linear(out_channels_3, fc_hidden_size), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(fc_hidden_size, forecast),
        )

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return self.final_fc(self.cnn_features(x)), None


class DilatedCausalResidualBlock(nn.Module):
    """TCN residual block with weight-normalized dilated causal convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation))
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        chomp = out.size(-1) - x.size(-1)
        if chomp > 0:
            out = out[:, :, :-chomp]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class RNN_TCN(nn.Module):
    """LSTM encoder feeding a stack of dilated causal TCN blocks.

    Reference: Gong et al. (2021), "Deep learning based prediction on
    greenhouse crop yield combined TCN and RNN." Sensors, 21(13), 4537.
    """

    def __init__(
        self, in_features: int = 15, lstm_hidden: int = 32, tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 2, fc_hidden: int = 16, dropout: float = 0.2,
    ):
        super().__init__()
        tcn_channels = tcn_channels or [32, 32]
        self.lstm = nn.LSTM(in_features, lstm_hidden, num_layers=1, batch_first=True)
        blocks, in_ch = [], lstm_hidden
        for i, out_ch in enumerate(tcn_channels):
            blocks.append(DilatedCausalResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(tcn_channels[-1], fc_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        lstm_out, _ = self.lstm(x)
        tcn_out = self.tcn(lstm_out.transpose(1, 2))
        return self.fc_block(self.flatten(self.pool(tcn_out))), None


# --------------------------------------------------------------------------- #
# BINN: the proposed biophysics-informed model
# --------------------------------------------------------------------------- #
# Physically plausible bounds for each ModVege-inspired parameter, shared by
# every paddock's learned embedding. minSEA is not a free parameter: it is
# always derived as ``2.0 - maxSEA`` (a symmetry constraint enforced inside
# the loss), so only maxSEA is optimized.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "RUEmax": (1.0, 3.0), "alpha_PAR": (0.035, 0.055), "T0": (2.0, 5.0),
    "T1": (8.0, 15.0), "T2": (18.0, 28.0), "SLA": (0.019, 0.033),
    "K_DV": (0.001, 0.004), "gammaGV": (0.2, 0.6), "percentLAM": (0.58, 0.78),
    "ST1": (600.0, 1000.0), "ST2": (1200.0, 1850.0), "maxSEA": (1.0, 1.5),
    "NI": (0.35, 1.2), "age": (1.0, 3.0), "wr": (0.0, 1.0), "gt": (0.5, 1.5),
}


class BINN(nn.Module):
    """Biophysics-Informed Neural Network.

    A 2-layer GRU predicts the next-step AGB directly (the "data-driven"
    branch), while a per-paddock ModVege-style growth/senescence module
    (parameterized by learnable, sigmoid-constrained physiological
    parameters) produces a second, mechanistic prediction ``y_biop``. The
    two branches are combined by :func:`complete_biophysics_loss`.

    Args:
        paddock_priors: If provided (BINN_C), each physiological parameter's
            per-paddock embedding is initialized from field-calibrated
            values; otherwise (BINN_R) every paddock shares the same
            initial value given by the corresponding keyword argument.
    """

    def __init__(
        self, input_size: int, forecast: int, hidden_size_1: int = 24, dropout_rate: float = 0.1,
        num_paddocks: Optional[int] = None, paddock_priors: Optional[Dict[str, np.ndarray]] = None,
        RUEmax: float = 3.0, alpha_PAR: float = 0.045, T0: float = 4.0, T1: float = 10.0, T2: float = 20.0,
        SLA: float = 0.025, K_DV: float = 0.002, gammaGV: float = 0.4, percentLAM: float = 0.68,
        ST1: float = 700.0, ST2: float = 1350.0, maxSEA: float = 1.3, NI: float = 0.85,
        age: float = 2.0, wr: float = 0.8, gt: float = 1.5,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size_1, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size_1, forecast)

        self.num_paddocks = num_paddocks
        self.param_ranges = dict(PARAM_RANGES)
        self.initial_physical_values = {
            "RUEmax": RUEmax, "alpha_PAR": alpha_PAR, "T0": T0, "T1": T1, "T2": T2,
            "SLA": SLA, "K_DV": K_DV, "gammaGV": gammaGV, "percentLAM": percentLAM,
            "ST1": ST1, "ST2": ST2, "maxSEA": maxSEA, "NI": NI,
            "age": age, "wr": wr, "gt": gt,
        }

        def to_latent(v, v_min, v_max):
            p = np.clip((v - v_min) / (v_max - v_min), 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        self.unconstrained_params = nn.ModuleDict()
        for name, initial_v in self.initial_physical_values.items():
            v_min, v_max = self.param_ranges[name]
            embedding_layer = nn.Embedding(self.num_paddocks, 1)

            if paddock_priors and name in paddock_priors:
                u_init = to_latent(paddock_priors[name], v_min, v_max)
                with torch.no_grad():
                    embedding_layer.weight.copy_(torch.from_numpy(u_init).float().view(-1, 1))
            else:
                u_init_scalar = to_latent(initial_v, v_min, v_max)
                with torch.no_grad():
                    embedding_layer.weight.fill_(float(u_init_scalar))

            self.unconstrained_params[name] = embedding_layer

    def forward(self, x: torch.Tensor, paddock_ids: Optional[torch.Tensor] = None):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)

        paddock_indices = paddock_ids.long()
        trainable_params = {
            name: layer(paddock_indices) for name, layer in self.unconstrained_params.items()
        }
        trainable_params["param_ranges"] = self.param_ranges
        return out, trainable_params


# --------------------------------------------------------------------------- #
# Model factory (ablation-aware)
# --------------------------------------------------------------------------- #
def build_dl_model(
    model_type: str, input_dim: int, forecast: int, num_paddocks: int,
    dataset_cfg: "config.DatasetConfig", paddock_priors: Optional[Dict[str, np.ndarray]] = None,
    bio_param_overrides: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """Instantiate a deep-learning model using ``dataset_cfg``'s tuned hyperparameters.

    Args:
        model_type: One of ``config.DL_MODEL_TYPES``.
        bio_param_overrides: Optional per-call overrides merged on top of
            ``dataset_cfg.bio_params`` (used by BINN_R only; e.g. for the
            biophysical-parameter sensitivity analysis).
    """
    if model_type == "BINN_R":
        bio_params = {**dataset_cfg.bio_params, **(bio_param_overrides or {})}
        return BINN(
            input_size=input_dim, forecast=forecast, num_paddocks=num_paddocks,
            paddock_priors=None, **dataset_cfg.model_params, **bio_params,
        )
    if model_type == "BINN_C":
        c = dataset_cfg.binn_c_overrides
        return BINN(
            input_size=input_dim, forecast=forecast, num_paddocks=num_paddocks,
            paddock_priors=paddock_priors, **dataset_cfg.model_params,
            age=c["age"], wr=c["wr"], gt=c["gt"],
        )
    if model_type == "GRU":
        return GRU(input_size=input_dim, forecast=forecast, **dataset_cfg.model_params)
    if model_type == "BI-LSTM":
        return BiLSTM(input_size=input_dim, forecast=forecast, **dataset_cfg.baselines.bilstm)
    if model_type == "GC-LSTM":
        return GC_LSTM(input_size=input_dim, forecast=forecast, **dataset_cfg.baselines.gc_lstm)
    if model_type == "1D-CNN":
        return MultiChannelClimateCNN(in_channels=input_dim, forecast=forecast, **dataset_cfg.baselines.cnn1d)
    if model_type == "RNN-TCN":
        return RNN_TCN(in_features=input_dim, **dataset_cfg.baselines.rnn_tcn)
    raise ValueError(f"Unknown deep-learning model_type: {model_type!r}")


# --------------------------------------------------------------------------- #
# Biophysics (ModVege-style) growth/senescence module
# --------------------------------------------------------------------------- #
def fclai(SLA: torch.Tensor, AGB: torch.Tensor, percent_lam: torch.Tensor) -> torch.Tensor:
    """Leaf Area Index from specific leaf area, above-ground biomass, and lamina fraction."""
    return SLA * (AGB * 0.1) * percent_lam


def fPARi(PARi: torch.Tensor, alpha_PAR: torch.Tensor) -> torch.Tensor:
    """Light-interception attenuation factor as a function of incident PAR."""
    return torch.where(PARi < 5, 1.0, torch.clamp(1 - alpha_PAR * (PARi - 5), min=0.0))


def fT(T: torch.Tensor, T0: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear temperature limitation factor (0 below T0/above 40C, 1 within [T1, T2))."""
    result = torch.zeros_like(T)
    result = torch.where((T < T0) | (T >= 40), 0.0, result)
    result = torch.where((T >= T0) & (T < T1), (T - T0) / (T1 - T0), result)
    result = torch.where((T >= T1) & (T < T2), 1.0, result)
    result = torch.where((T >= T2) & (T < 40), (40 - T) / (40 - T2), result)
    return result


def fsea(maxSEA: torch.Tensor, minSEA: torch.Tensor, SumT: torch.Tensor, ST2: torch.Tensor, ST1: torch.Tensor) -> torch.Tensor:
    """Seasonal growth-effect multiplier driven by cumulative thermal time (SumT)."""
    result = torch.zeros_like(SumT)
    result = torch.where((SumT < 200) | (SumT >= ST2), minSEA, result)
    result = torch.where(
        (SumT >= 200) & (SumT < ST1 - 200),
        minSEA + (maxSEA - minSEA) * (SumT - 200) / (ST1 - 400), result,
    )
    result = torch.where((SumT >= ST1 - 200) & (SumT < ST1 - 100), maxSEA, result)
    result = torch.where(
        (SumT >= ST1 - 100) & (SumT < ST2),
        maxSEA + (minSEA - maxSEA) * (SumT - ST1 + 100) / (ST2 - ST1 + 100), result,
    )
    return result


def constrain_param(unconstrained_u: torch.Tensor, param_name: str, ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """Sigmoid-constrain a latent (unbounded) parameter into its physical range."""
    v_min, v_max = ranges[param_name]
    return v_min + (v_max - v_min) * torch.sigmoid(unconstrained_u)


def get_biophysics_features(
    X: torch.Tensor, bp: torch.Tensor, paddock_datasets,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Inverse-scale the model input window back into physical climate/AGB units.

    ``X`` columns are, by construction (see ``src.dataset.create_datasets``):
    ``[AGB(label), DAILY_RAIN, MAX_TEMP, MIN_TEMP, RADIATION(PAR), ET_TALL_CROP, RH_TMAX, RH_TMIN, ...]``.
    """
    device = X.device
    data_min_list, data_max_list = [], []
    for i in range(X.size(0)):
        paddock_id_encoded = bp[i, 0, -1].item()
        scaler = paddock_datasets[paddock_id_encoded]["scaler"]
        data_min_list.append(scaler.data_min_)
        data_max_list.append(scaler.data_max_)

    data_min = torch.tensor(np.array(data_min_list), dtype=torch.float32, device=device).unsqueeze(1)
    data_max = torch.tensor(np.array(data_max_list), dtype=torch.float32, device=device).unsqueeze(1)

    AGB = minmax_inverse(X[:, :, 0], data_min[:, :, 0], data_max[:, :, 0])
    PP = minmax_inverse(X[:, :, 1], data_min[:, :, 1], data_max[:, :, 1])
    MAX_TEMP = minmax_inverse(X[:, :, 2], data_min[:, :, 2], data_max[:, :, 2])
    MIN_TEMP = minmax_inverse(X[:, :, 3], data_min[:, :, 3], data_max[:, :, 3])
    PARi = minmax_inverse(X[:, :, 4], data_min[:, :, 4], data_max[:, :, 4])
    PET = minmax_inverse(X[:, :, 5], data_min[:, :, 5], data_max[:, :, 5])

    climate_params = {
        "AGB": AGB, "OAGB": X[:, :, 0], "PP": PP,
        "AVG_TEMP": (MAX_TEMP + MIN_TEMP) / 2, "PARi": PARi, "PET": PET,
    }
    return climate_params, data_min, data_max


def complete_biophysics_loss(
    y_pred: torch.Tensor, y: torch.Tensor, climate_params: Dict[str, torch.Tensor],
    bio_params: torch.Tensor, alpha_: float, data_min: torch.Tensor, data_max: torch.Tensor,
    tr_params: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the combined data + biophysics-consistency loss.

    The total loss is ``data_loss + biop_loss + alpha_ * match_loss``, where
    ``data_loss`` is the GRU head's MSE against the ground truth,
    ``biop_loss`` is the mechanistic (ModVege-style) head's MSE against the
    ground truth, and ``match_loss`` pulls the GRU head towards the
    (detached) mechanistic prediction.

    Returns:
        total_loss: Scalar loss used for backpropagation.
        y_biop: The mechanistic (ModVege-style) AGB prediction, scaled to
            match ``y_pred``/``y`` (used to additionally report a
            "biophysics R2" during training/validation).
    """
    ranges = tr_params["param_ranges"]
    RUEmax = constrain_param(tr_params["RUEmax"], "RUEmax", ranges)
    alpha_PAR = constrain_param(tr_params["alpha_PAR"], "alpha_PAR", ranges)
    T0 = constrain_param(tr_params["T0"], "T0", ranges)
    T1 = constrain_param(tr_params["T1"], "T1", ranges)
    T2 = constrain_param(tr_params["T2"], "T2", ranges)
    SLA = constrain_param(tr_params["SLA"], "SLA", ranges)
    K_DV = constrain_param(tr_params["K_DV"], "K_DV", ranges)
    gammaGV = constrain_param(tr_params["gammaGV"], "gammaGV", ranges)
    percentLAM = constrain_param(tr_params["percentLAM"], "percentLAM", ranges)
    ST1 = constrain_param(tr_params["ST1"], "ST1", ranges)
    ST2 = constrain_param(tr_params["ST2"], "ST2", ranges)
    NI = constrain_param(tr_params["NI"], "NI", ranges)
    maxSEA = constrain_param(tr_params["maxSEA"], "maxSEA", ranges)
    minSEA = 2.0 - maxSEA  # symmetry constraint (see PARAM_RANGES docstring)
    gt = constrain_param(tr_params["gt"], "gt", ranges)
    age = constrain_param(tr_params["age"], "age", ranges)
    wr = constrain_param(tr_params["wr"], "wr", ranges)

    SumT = bio_params[:, :, -2]
    AGB = climate_params["AGB"]
    PARi = climate_params["PARi"]
    TEMP = climate_params["AVG_TEMP"]

    data_loss = nn.MSELoss()(y_pred, y)

    LAI = fclai(SLA, AGB, percentLAM)
    ENV = NI * fPARi(PARi, alpha_PAR) * fT(TEMP, T0, T1, T2) * wr
    PGRO = PARi * RUEmax * (1 - torch.exp(-0.6 * LAI)) * 10
    SEA = fsea(maxSEA, minSEA, SumT, ST2, ST1)
    GRO = PGRO * SEA * ENV

    SEN_BASE = (1 - gammaGV) * K_DV * AGB
    SEN = torch.where(
        TEMP > T0, SEN_BASE * TEMP * age,
        torch.where(TEMP < 0, SEN_BASE * torch.abs(TEMP), torch.zeros_like(TEMP)),
    )

    growth_rates = (GRO - SEN).mean(-1) * 15 * gt.squeeze(1)
    y_biop = AGB[:, -1] + growth_rates
    y_biop = minmax_scale(y_biop, data_min[:, 0, 0], data_max[:, 0, 0]).view(-1, 1)

    biop_loss = nn.MSELoss()(y_biop, y)
    match_loss = nn.MSELoss()(y_pred, y_biop.detach())
    total_loss = data_loss + biop_loss + alpha_ * match_loss

    return total_loss, y_biop


# --------------------------------------------------------------------------- #
# Training / validation loop
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    bio_flag: bool, alpha_: float, paddock_datasets,
) -> Tuple[float, float, Optional[float]]:
    """Run one full training epoch. Returns ``(avg_loss, train_r2, train_biop_r2)``."""
    model.train()
    total_loss, all_preds, all_targets, all_biops = 0.0, [], [], []

    for X, y, bp in loader:
        optimizer.zero_grad()
        paddock_ids = bp[:, 0, -1].long()
        y_pred, tr_params = model(X, paddock_ids)

        if bio_flag:
            climate_params, data_min, data_max = get_biophysics_features(X, bp, paddock_datasets)
            loss, y_biop = complete_biophysics_loss(
                y_pred, y, climate_params, bp, alpha_, data_min, data_max, tr_params
            )
            all_biops.append(y_biop.detach().cpu())
        else:
            loss = nn.MSELoss()(y_pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(y_pred.detach().cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / len(loader)
    y_pred_all = torch.cat(all_preds).numpy()
    y_true_all = torch.cat(all_targets).numpy()
    train_r2 = r2_score(y_true_all, y_pred_all)

    train_biop_r2 = None
    if bio_flag and all_biops:
        train_biop_r2 = r2_score(y_true_all, torch.cat(all_biops).numpy())

    return avg_loss, train_r2, train_biop_r2


def valid_one_epoch(
    model: nn.Module, loader: DataLoader, bio_flag: bool, alpha_: float,
    paddock_datasets,
) -> Tuple[float, float, Optional[float]]:
    """Evaluate the model (no gradient updates) on a given loader.

    Used exclusively for the validation split during training, and for a
    single post-training pass over the validation/test splits.
    Returns ``(avg_loss, r2, biop_r2)``.
    """
    model.eval()
    total_loss, all_preds, all_targets, all_biops = 0.0, [], [], []

    with torch.no_grad():
        for X, y, bp in loader:
            paddock_ids = bp[:, 0, -1].long()
            y_pred, tr_params = model(X, paddock_ids)

            if bio_flag:
                climate_params, data_min, data_max = get_biophysics_features(X, bp, paddock_datasets)
                loss, y_biop = complete_biophysics_loss(
                    y_pred, y, climate_params, bp, alpha_, data_min, data_max, tr_params
                )
                all_biops.append(y_biop.cpu())
            else:
                loss = nn.MSELoss()(y_pred, y)

            total_loss += loss.item()
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(loader)
    y_pred_all = torch.cat(all_preds).numpy()
    y_true_all = torch.cat(all_targets).numpy()
    r2 = r2_score(y_true_all, y_pred_all)

    biop_r2 = None
    if bio_flag and all_biops:
        biop_r2 = r2_score(y_true_all, torch.cat(all_biops).numpy())

    return avg_loss, r2, biop_r2


def model_train(
    model: nn.Module, num_epochs: int, batch_size: int,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bio_flag: bool, lr: float, paddock_datasets, alpha_: float = 0.0,
    is_printed: bool = True,
    lr_bio: float = 0.0, model_type: str = "BINN", weight_decay: float = 0.0, bio_weight_decay: float = 0.0,
) -> Tuple[List[float], List[float], List[float], List[Optional[float]]]:
    """Train ``model`` for ``num_epochs``, restoring the best validation-loss checkpoint.

    IMPORTANT (no test-set leakage): this function never sees the test
    split. Only ``train_data`` and ``val_data`` are consumed here; the
    lowest-validation-loss checkpoint is restored before returning, and the
    (single, post-hoc) test-set evaluation is left entirely to the caller.

    Returns:
        train_losses, val_losses: Per-epoch loss curves.
        val_r2_history: Per-epoch validation R2 of the main (data-driven) head.
        val_biop_r2_history: Per-epoch validation R2 of the biophysics head
            (``None`` per epoch when ``bio_flag`` is False).
    """
    if model_type in ("BINN", "BINN_R", "BINN_C"):
        if is_printed:
            print(f"BINN optimization: dual parameter groups (lr={lr}, lr_bio={lr_bio})")
        bio_params_list = [p for layer in model.unconstrained_params.values() for p in layer.parameters()]
        standard_params = [p for n, p in model.named_parameters() if "unconstrained" not in n]
        optimizer = torch.optim.AdamW([
            {"params": standard_params, "lr": lr, "weight_decay": weight_decay},
            {"params": bio_params_list, "lr": lr_bio, "weight_decay": bio_weight_decay},
        ])
        for param in model.unconstrained_params.parameters():
            param.requires_grad = True
    else:
        if is_printed:
            print(f"Standard optimization for {model_type} (lr={lr})")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batch_size, shuffle=False)

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_r2_history: List[float] = []
    val_biop_r2_history: List[Optional[float]] = []

    lowest_val_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):
        train_loss, train_r2, train_biop_r2 = train_one_epoch(
            model, train_loader, optimizer, bio_flag, alpha_, paddock_datasets
        )
        train_losses.append(train_loss)

        val_loss, val_r2, val_biop_r2 = valid_one_epoch(
            model, val_loader, bio_flag, alpha_, paddock_datasets
        )
        val_losses.append(val_loss)
        val_r2_history.append(val_r2)
        val_biop_r2_history.append(val_biop_r2)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        if is_printed:
            msg = (
                f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.3f} | Train R2: {train_r2:.3f}"
            )
            if train_biop_r2 is not None:
                msg += f" (BioP: {train_biop_r2:.3f})"
            msg += f" | Val Loss: {val_loss:.3f} | Val R2: {val_r2:.3f}"
            if val_biop_r2 is not None:
                msg += f" (BioP: {val_biop_r2:.3f})"
            print(msg)

    if best_weights is not None:
        model.load_state_dict(best_weights)
        if is_printed:
            print(f"Restored best checkpoint (lowest Val Loss: {lowest_val_loss:.3f}).")

    return train_losses, val_losses, val_r2_history, val_biop_r2_history
