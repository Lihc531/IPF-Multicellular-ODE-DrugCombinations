#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_v1B1_second_batch_ablation_runner.py

Second-batch ablation runner for confirming the final Neural ODE model.

This script focuses on architectural and dynamical ablations that were not
included in the first batch:

    B0_full_reference
        Full v1B.1 reference model.

    B1_no_timeaware
        Remove explicit pseudo-time input t from ODE function:
            dz/dt = f(z, c)
        instead of:
            dz/dt = f(z, t, c)

    B2_no_pair_specific_delta_t
        Use a batch-average integration interval instead of pair-specific dt.

    B3_solver_dopri5
        Use adaptive dopri5 solver instead of rk4.

    B4_latent_dim_6
        Test lower latent dimension.

    B5_latent_dim_10
        Test higher latent dimension.

    B6_weak_direction
        Use weaker direction constraints.

    B7_strong_direction
        Use stronger direction constraints.

Main purpose:
    Decide whether final model should remain:
        time-aware + pair-specific-dt + rk4 + latent_dim=8 + full v1B.1 constraints

Input:
    04_cell_state_matrix_v1B_filtered.csv

PyCharm usage:
    1. Modify BASE_CONFIG["input_csv"].
    2. Modify BASE_CONFIG["outdir"].
    3. Run this script directly.

Main aggregate outputs:
    second_batch_summary_metrics.csv
    second_batch_stage_directionality.csv
    second_batch_disease_directionality.csv
    second_batch_celltype_directionality.csv
    second_batch_celltype_cluster_directionality.csv
    second_batch_stage4_terminal_check.csv
    second_batch_top10_fibrosis_clusters_nstates20.csv
    second_batch_top10_terminal_clusters_nstates20.csv
"""

import os
import json
import random
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchdiffeq import odeint
except ImportError as exc:
    raise ImportError(
        "torchdiffeq is not installed. Install it in your current PyCharm / conda environment:\n"
        "pip install torchdiffeq"
    ) from exc


# ============================================================
# 0. BASE CONFIG
# ============================================================
BASE_CONFIG = {
    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------
    "input_csv": r"E:\Rproject\results\cell_state_feature_matrix\tables\04_cell_state_matrix_v1B_filtered.csv",
    "outdir": r"E:\PythonProject\results\neural_ode_v1B1_second_batch_ablation",

    # --------------------------------------------------------
    # Main model identity
    # --------------------------------------------------------
    "group_mode": "celltype_cluster",
    "condition_mode": "celltype_cluster",

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------
    "min_cells": 10,
    "min_pairs_per_group": 10,

    # --------------------------------------------------------
    # Model settings
    # --------------------------------------------------------
    "latent_dim": 8,
    "hidden_dim": 64,
    "dropout": 0.05,
    "batch_size": 64,
    "epochs": 500,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "test_size": 0.2,
    "solver": "rk4",
    "seed": 123,
    "force_cpu": False,

    # Key architecture switches
    "use_time_input": True,
    "use_pair_specific_delta_t": True,

    # For adaptive solvers like dopri5. Ignored by rk4.
    "ode_rtol": 1e-4,
    "ode_atol": 1e-5,

    # --------------------------------------------------------
    # Direction vectors
    # --------------------------------------------------------
    "terminal_top_quantile": 0.90,
    "fibrosis_top_quantile": 0.90,
    "projection_delta_t": 0.05,

    # --------------------------------------------------------
    # Direction loss target weights
    # --------------------------------------------------------
    "lambda_axis_direction": 0.05,
    "lambda_fibrosis_direction": 0.05,
    "lambda_terminal_early": 0.02,
    "lambda_terminal_late": 0.04,
    "lambda_late_stability": 0.02,

    # --------------------------------------------------------
    # Direction loss warm-up
    # --------------------------------------------------------
    "direction_warmup_start_epoch": 50,
    "direction_warmup_end_epoch": 150,

    # --------------------------------------------------------
    # Stage-aware weights
    # --------------------------------------------------------
    "stage_weights": {
        "Stage_1_low": {
            "axis": 1.25,
            "fibrosis": 1.25,
            "terminal": 1.00,
            "late_terminal": 0.00,
            "late_stability": 0.00,
        },
        "Stage_2_intermediate_low": {
            "axis": 1.00,
            "fibrosis": 1.00,
            "terminal": 1.00,
            "late_terminal": 0.00,
            "late_stability": 0.00,
        },
        "Stage_3_intermediate_high": {
            "axis": 0.35,
            "fibrosis": 0.35,
            "terminal": 0.50,
            "late_terminal": 0.00,
            "late_stability": 0.00,
        },
        "Stage_4_high": {
            "axis": 0.00,
            "fibrosis": 0.00,
            "terminal": 0.00,
            "late_terminal": 1.00,
            "late_stability": 1.00,
        },
    },
}


def build_second_batch_configs(base: Dict) -> List[Dict]:
    ablations = []

    # B0: reference full v1B.1
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B0_full_reference",
        "ablation_family": "reference",
    })
    ablations.append(cfg)

    # B1: no explicit pseudo-time input
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B1_no_timeaware",
        "ablation_family": "time_input",
        "use_time_input": False,
    })
    ablations.append(cfg)

    # B2: no pair-specific delta_t
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B2_no_pair_specific_delta_t",
        "ablation_family": "delta_t",
        "use_pair_specific_delta_t": False,
    })
    ablations.append(cfg)

    # B3: adaptive solver dopri5
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B3_solver_dopri5",
        "ablation_family": "solver",
        "solver": "dopri5",
        "ode_rtol": 1e-4,
        "ode_atol": 1e-5,
    })
    ablations.append(cfg)

    # B4: lower latent dimension
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B4_latent_dim_6",
        "ablation_family": "latent_dim",
        "latent_dim": 6,
    })
    ablations.append(cfg)

    # B5: higher latent dimension
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B5_latent_dim_10",
        "ablation_family": "latent_dim",
        "latent_dim": 10,
    })
    ablations.append(cfg)

    # B6: weaker biological constraints
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B6_weak_direction",
        "ablation_family": "direction_strength",
        "lambda_axis_direction": 0.025,
        "lambda_fibrosis_direction": 0.025,
        "lambda_terminal_early": 0.010,
        "lambda_terminal_late": 0.020,
        "lambda_late_stability": 0.010,
    })
    ablations.append(cfg)

    # B7: stronger biological constraints
    cfg = deepcopy(base)
    cfg.update({
        "ablation_name": "B7_strong_direction",
        "ablation_family": "direction_strength",
        "lambda_axis_direction": 0.100,
        "lambda_fibrosis_direction": 0.100,
        "lambda_terminal_early": 0.040,
        "lambda_terminal_late": 0.080,
        "lambda_late_stability": 0.040,
    })
    ablations.append(cfg)

    return ablations


# ============================================================
# 1. Utilities
# ============================================================
def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def cosine_similarity_to_vector(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    return (matrix @ vector) / (matrix_norm * vector_norm + eps)


def warmup_multiplier(epoch: int, start_epoch: int, end_epoch: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return float((epoch - start_epoch) / max(1, end_epoch - start_epoch))


# ============================================================
# 2. Data preparation
# ============================================================
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_exact = {
        "mean_axis_v2_score",
        "mean_cell_multicellular_progression_score",
        "mean_nCount_RNA",
        "mean_nFeature_RNA",
        "mean_percent_mt",
        "mean_percent_ribo",
        "mean_doublet_score",
        "n_cells",
        "n_cells_sample",
        "fraction_in_sample",
    }

    feature_cols = []
    for col in numeric_cols:
        if col in exclude_exact:
            continue
        if not col.startswith("mean_"):
            continue
        if col.endswith("_score_0_1") or col.endswith("_composite_score"):
            feature_cols.append(col)

    feature_cols = sorted(set(feature_cols))
    if len(feature_cols) == 0:
        raise ValueError("No valid latent feature columns found.")
    return feature_cols


def find_fibrosis_column(df: pd.DataFrame) -> str:
    candidates = [
        "mean_fibrosis_composite_score",
        "mean_fibrosis_core_score_0_1",
        "mean_ECM_remodeling_score_0_1",
        "mean_collagen_formation_score_0_1",
        "mean_myofibroblast_activation_score_0_1",
        "mean_TGF_beta_signaling_score_0_1",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No fibrosis-related column found. Expected mean_fibrosis_composite_score or related module score."
    )


def prepare_dataframe(input_csv: str, min_cells: int) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input file not found:\n{input_csv}\n"
            "Please modify BASE_CONFIG['input_csv'] to the real file path."
        )

    df = pd.read_csv(input_csv)

    required = [
        "cell_state_id",
        "sample_id",
        "Disease_Identity",
        "CellType_Category",
        "celltype_marker_based",
        "cluster_default",
        "axis_v2_stage",
        "n_cells",
        "fraction_in_sample",
        "mean_axis_v2_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["mean_axis_v2_score"] = pd.to_numeric(df["mean_axis_v2_score"], errors="coerce")
    df["n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce")
    df = df[np.isfinite(df["mean_axis_v2_score"])].copy()
    df = df[np.isfinite(df["n_cells"])].copy()
    df = df[df["n_cells"] >= min_cells].copy()

    if df.shape[0] < 100:
        raise ValueError(f"Too few states after filtering: {df.shape[0]}. Try lowering min_cells.")

    df = df.sort_values(
        ["celltype_marker_based", "cluster_default", "sample_id", "mean_axis_v2_score"]
    ).reset_index(drop=True)

    return df


def make_condition_matrix(df: pd.DataFrame, condition_mode: str):
    if condition_mode == "celltype":
        condition_df = df[["celltype_marker_based"]].astype(str)
        condition_prefix = ["celltype_marker_based"]
    elif condition_mode == "celltype_cluster":
        condition_df = df[["celltype_marker_based", "cluster_default"]].astype(str)
        condition_prefix = ["celltype_marker_based", "cluster_default"]
    else:
        raise ValueError("condition_mode must be 'celltype' or 'celltype_cluster'.")

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    cond = encoder.fit_transform(condition_df)
    condition_feature_names = encoder.get_feature_names_out(condition_prefix)
    return cond, condition_feature_names


def build_stage_weight_matrix(df: pd.DataFrame, stage_weights: Dict) -> pd.DataFrame:
    rows = []
    for stage in df["axis_v2_stage"].astype(str).values:
        w = stage_weights.get(
            stage,
            {
                "axis": 0.50,
                "fibrosis": 0.50,
                "terminal": 0.50,
                "late_terminal": 0.00,
                "late_stability": 0.00,
            },
        )
        rows.append(
            {
                "stage_weight_axis": float(w["axis"]),
                "stage_weight_fibrosis": float(w["fibrosis"]),
                "stage_weight_terminal": float(w["terminal"]),
                "stage_weight_late_terminal": float(w["late_terminal"]),
                "stage_weight_late_stability": float(w["late_stability"]),
            }
        )
    return pd.DataFrame(rows)


def compute_latent_state(df: pd.DataFrame, config: Dict, tab_dir: str):
    feature_cols = select_feature_columns(df)
    fibrosis_col = find_fibrosis_column(df)

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    n_components = min(config["latent_dim"], X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_components < 2:
        raise ValueError("Too few components available for PCA.")

    pca = PCA(n_components=n_components, random_state=config["seed"])
    z = pca.fit_transform(X_scaled)

    pd.DataFrame({"feature_col": feature_cols}).to_csv(
        os.path.join(tab_dir, f"latentdim{config['latent_dim']}_features_used.csv"),
        index=False,
    )
    pd.DataFrame({
        "latent_dim": [f"PC{i + 1}" for i in range(z.shape[1])],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }).to_csv(
        os.path.join(tab_dir, f"latentdim{config['latent_dim']}_pca_explained_variance.csv"),
        index=False,
    )

    return z, pca, feature_cols, fibrosis_col


# ============================================================
# 3. Transition pairs
# ============================================================
def build_transition_pairs(
    df: pd.DataFrame,
    group_mode: str,
    min_delta_t: float = 1e-4,
) -> pd.DataFrame:
    if group_mode == "celltype":
        group_cols = ["celltype_marker_based"]
    elif group_mode == "celltype_cluster":
        group_cols = ["celltype_marker_based", "cluster_default"]
    else:
        raise ValueError("group_mode must be 'celltype' or 'celltype_cluster'.")

    pairs = []

    for group_values, g in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        transition_group = "__".join([str(x) for x in group_values])

        g = g.sort_values("mean_axis_v2_score").reset_index()
        if g.shape[0] < 2:
            continue

        for i in range(g.shape[0] - 1):
            row0 = g.iloc[i]
            row1 = g.iloc[i + 1]

            t0 = float(row0["mean_axis_v2_score"])
            t1 = float(row1["mean_axis_v2_score"])

            if not np.isfinite(t0) or not np.isfinite(t1):
                continue
            if t1 <= t0:
                continue
            if (t1 - t0) < min_delta_t:
                continue

            pair = {
                "idx0": int(row0["index"]),
                "idx1": int(row1["index"]),
                "cell_state_id_0": row0["cell_state_id"],
                "cell_state_id_1": row1["cell_state_id"],
                "sample_id_0": row0["sample_id"],
                "sample_id_1": row1["sample_id"],
                "Disease_Identity_0": row0["Disease_Identity"],
                "Disease_Identity_1": row1["Disease_Identity"],
                "celltype_marker_based": row0["celltype_marker_based"],
                "cluster_default_0": row0["cluster_default"],
                "cluster_default_1": row1["cluster_default"],
                "axis_v2_stage_0": row0["axis_v2_stage"],
                "axis_v2_stage_1": row1["axis_v2_stage"],
                "transition_group": transition_group,
                "t0": t0,
                "t1": t1,
                "delta_t": t1 - t0,
                "n_cells_0": row0["n_cells"],
                "n_cells_1": row1["n_cells"],
            }
            pairs.append(pair)

    pair_df = pd.DataFrame(pairs)
    if pair_df.shape[0] < 50:
        warnings.warn(
            f"Only {pair_df.shape[0]} transition pairs were built. "
            "The transition graph may be sparse."
        )
    return pair_df


def filter_sparse_transition_groups(pair_df: pd.DataFrame, min_pairs_per_group: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_count = (
        pair_df.groupby("transition_group")
        .size()
        .reset_index(name="n_transition_pairs")
        .sort_values("n_transition_pairs", ascending=False)
    )
    pair_count["used_for_training"] = pair_count["n_transition_pairs"] >= min_pairs_per_group

    keep_groups = set(pair_count.loc[pair_count["used_for_training"], "transition_group"].astype(str))
    filtered = pair_df[pair_df["transition_group"].astype(str).isin(keep_groups)].copy().reset_index(drop=True)

    return filtered, pair_count


class TransitionPairDataset(Dataset):
    def __init__(
        self,
        pair_df: pd.DataFrame,
        z: np.ndarray,
        cond: np.ndarray,
        stage_weight_df: pd.DataFrame,
    ):
        self.pair_df = pair_df.reset_index(drop=True)
        self.z = z.astype(np.float32)
        self.cond = cond.astype(np.float32)
        self.stage_weight_df = stage_weight_df.reset_index(drop=True).astype(np.float32)

    def __len__(self) -> int:
        return self.pair_df.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.pair_df.iloc[idx]
        idx0 = int(row["idx0"])
        idx1 = int(row["idx1"])
        w = self.stage_weight_df.iloc[idx0]

        return {
            "z0": torch.tensor(self.z[idx0], dtype=torch.float32),
            "z1": torch.tensor(self.z[idx1], dtype=torch.float32),
            "c": torch.tensor(self.cond[idx0], dtype=torch.float32),
            "t0": torch.tensor(np.float32(row["t0"]), dtype=torch.float32),
            "t1": torch.tensor(np.float32(row["t1"]), dtype=torch.float32),
            "idx0": torch.tensor(idx0, dtype=torch.long),
            "idx1": torch.tensor(idx1, dtype=torch.long),
            "stage_weight_axis": torch.tensor(w["stage_weight_axis"], dtype=torch.float32),
            "stage_weight_fibrosis": torch.tensor(w["stage_weight_fibrosis"], dtype=torch.float32),
            "stage_weight_terminal": torch.tensor(w["stage_weight_terminal"], dtype=torch.float32),
            "stage_weight_late_terminal": torch.tensor(w["stage_weight_late_terminal"], dtype=torch.float32),
            "stage_weight_late_stability": torch.tensor(w["stage_weight_late_stability"], dtype=torch.float32),
        }


# ============================================================
# 4. Neural ODE
# ============================================================
class TimeAwareODEFunc(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        use_time_input: bool = True,
    ):
        super().__init__()
        self.use_time_input = use_time_input
        input_dim = latent_dim + cond_dim + (1 if use_time_input else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.cond = None

    def set_condition(self, cond: torch.Tensor) -> None:
        self.cond = cond

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.cond is None:
            raise RuntimeError("Condition is not set before ODE integration.")
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.cond.shape[0] != z.shape[0]:
            raise RuntimeError(
                f"Condition batch size {self.cond.shape[0]} does not match z batch size {z.shape[0]}."
            )

        if self.use_time_input:
            t_value = t.expand(z.shape[0], 1).to(z.device)
            x = torch.cat([z, t_value, self.cond], dim=1)
        else:
            x = torch.cat([z, self.cond], dim=1)

        return self.net(x)


class ConditionalNeuralODE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        solver: str = "rk4",
        use_pair_specific_delta_t: bool = True,
        use_time_input: bool = True,
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-5,
    ):
        super().__init__()
        self.func = TimeAwareODEFunc(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_time_input=use_time_input,
        )
        self.solver = solver
        self.use_pair_specific_delta_t = use_pair_specific_delta_t
        self.use_time_input = use_time_input
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

    def _odeint(self, func, z0, ts):
        if self.solver in ["dopri5", "adams", "bdf"]:
            return odeint(func, z0, ts, method=self.solver, rtol=self.ode_rtol, atol=self.ode_atol)
        return odeint(func, z0, ts, method=self.solver)

    def forward(self, z0: torch.Tensor, c: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        self.func.set_condition(c)
        delta_t = (t1 - t0).to(z0.device)
        if torch.any(delta_t <= 0):
            raise RuntimeError("Non-positive delta_t in batch.")

        if self.use_pair_specific_delta_t:
            base_func = self.func

            class ScaledFunc(nn.Module):
                def __init__(self, func, delta):
                    super().__init__()
                    self.func = func
                    self.delta = delta

                def forward(self, tau, z):
                    dz = self.func(tau, z)
                    return dz * self.delta.view(-1, 1)

            ts = torch.tensor([0.0, 1.0], dtype=torch.float32, device=z0.device)
            zt = self._odeint(ScaledFunc(base_func, delta_t), z0, ts)
            return zt[-1]

        delta_mean = delta_t.mean()
        ts = torch.stack([
            torch.tensor(0.0, device=z0.device),
            delta_mean.to(z0.device),
        ])
        zt = self._odeint(self.func, z0, ts)
        return zt[-1]

    def integrate_delta(self, z0: torch.Tensor, c: torch.Tensor, delta_t: float) -> torch.Tensor:
        self.func.set_condition(c)
        ts = torch.tensor([0.0, float(delta_t)], dtype=torch.float32, device=z0.device)
        zt = self._odeint(self.func, z0, ts)
        return zt[-1]

    def velocity(self, z: torch.Tensor, c: torch.Tensor, t_value: float = 0.0) -> torch.Tensor:
        self.func.set_condition(c)
        t = torch.tensor(float(t_value), dtype=torch.float32, device=z.device)
        return self.func(t, z)


# ============================================================
# 5. Direction vectors and losses
# ============================================================
def build_direction_vectors(
    df: pd.DataFrame,
    z: np.ndarray,
    fibrosis_col: str,
    terminal_top_quantile: float,
    fibrosis_top_quantile: float,
) -> Dict:
    axis = df["mean_axis_v2_score"].astype(float).values
    fibrosis = pd.to_numeric(df[fibrosis_col], errors="coerce").astype(float).values

    valid_fib = np.isfinite(fibrosis)
    if valid_fib.sum() < 20:
        raise ValueError(f"Too few valid fibrosis values in {fibrosis_col}.")

    axis_low_cut = np.quantile(axis, 1.0 - terminal_top_quantile)
    axis_high_cut = np.quantile(axis, terminal_top_quantile)
    fib_low_cut = np.quantile(fibrosis[valid_fib], 1.0 - fibrosis_top_quantile)
    fib_high_cut = np.quantile(fibrosis[valid_fib], fibrosis_top_quantile)

    low_axis_idx = axis <= axis_low_cut
    high_axis_idx = axis >= axis_high_cut
    low_fib_idx = fibrosis <= fib_low_cut
    high_fib_idx = fibrosis >= fib_high_cut
    terminal_idx = high_axis_idx | high_fib_idx

    if low_axis_idx.sum() < 5 or high_axis_idx.sum() < 5:
        raise ValueError("Too few high/low axis states for axis direction.")
    if low_fib_idx.sum() < 5 or high_fib_idx.sum() < 5:
        raise ValueError("Too few high/low fibrosis states for fibrosis direction.")
    if terminal_idx.sum() < 5:
        raise ValueError("Too few terminal prototype states.")

    axis_direction = z[high_axis_idx].mean(axis=0) - z[low_axis_idx].mean(axis=0)
    fibrosis_direction = z[high_fib_idx].mean(axis=0) - z[low_fib_idx].mean(axis=0)
    terminal_prototype = z[terminal_idx].mean(axis=0)

    return {
        "axis_direction": normalize_vector(axis_direction),
        "fibrosis_direction": normalize_vector(fibrosis_direction),
        "terminal_prototype": terminal_prototype,
        "axis_low_cut": float(axis_low_cut),
        "axis_high_cut": float(axis_high_cut),
        "fibrosis_low_cut": float(fib_low_cut),
        "fibrosis_high_cut": float(fib_high_cut),
        "low_axis_n": int(low_axis_idx.sum()),
        "high_axis_n": int(high_axis_idx.sum()),
        "low_fibrosis_n": int(low_fib_idx.sum()),
        "high_fibrosis_n": int(high_fib_idx.sum()),
        "terminal_n": int(terminal_idx.sum()),
    }


def weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * w).sum() / (w.sum() + eps)


def stageaware_direction_constrained_loss(
    z0: torch.Tensor,
    z1_pred: torch.Tensor,
    axis_direction: torch.Tensor,
    fibrosis_direction: torch.Tensor,
    terminal_prototype: torch.Tensor,
    stage_weight_axis: torch.Tensor,
    stage_weight_fibrosis: torch.Tensor,
    stage_weight_terminal: torch.Tensor,
    stage_weight_late_terminal: torch.Tensor,
    stage_weight_late_stability: torch.Tensor,
    lambda_axis: float,
    lambda_fibrosis: float,
    lambda_terminal_early: float,
    lambda_terminal_late: float,
    lambda_late_stability: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    delta_pred = z1_pred - z0

    axis_proj = torch.sum(delta_pred * axis_direction.view(1, -1), dim=1)
    fibrosis_proj = torch.sum(delta_pred * fibrosis_direction.view(1, -1), dim=1)

    loss_axis = weighted_mean(torch.relu(-axis_proj), stage_weight_axis)
    loss_fibrosis = weighted_mean(torch.relu(-fibrosis_proj), stage_weight_fibrosis)

    dist_before = torch.norm(z0 - terminal_prototype.view(1, -1), dim=1)
    dist_after = torch.norm(z1_pred - terminal_prototype.view(1, -1), dim=1)
    terminal_violation = torch.relu(dist_after - dist_before)

    loss_terminal_early = weighted_mean(terminal_violation, stage_weight_terminal)
    loss_terminal_late = weighted_mean(terminal_violation, stage_weight_late_terminal)

    delta_norm = torch.norm(delta_pred, dim=1)
    loss_late_stability = weighted_mean(delta_norm, stage_weight_late_stability)

    total_direction_loss = (
        lambda_axis * loss_axis
        + lambda_fibrosis * loss_fibrosis
        + lambda_terminal_early * loss_terminal_early
        + lambda_terminal_late * loss_terminal_late
        + lambda_late_stability * loss_late_stability
    )

    stats = {
        "loss_axis_direction": float(loss_axis.detach().cpu().item()),
        "loss_fibrosis_direction": float(loss_fibrosis.detach().cpu().item()),
        "loss_terminal_early": float(loss_terminal_early.detach().cpu().item()),
        "loss_terminal_late": float(loss_terminal_late.detach().cpu().item()),
        "loss_late_stability": float(loss_late_stability.detach().cpu().item()),
    }

    return total_direction_loss, stats


# ============================================================
# 6. Training and evaluation
# ============================================================
def train_one_epoch(
    model: ConditionalNeuralODE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    axis_direction_t: torch.Tensor,
    fibrosis_direction_t: torch.Tensor,
    terminal_prototype_t: torch.Tensor,
    config: Dict,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    mse = nn.MSELoss()

    wm = warmup_multiplier(
        epoch,
        start_epoch=config["direction_warmup_start_epoch"],
        end_epoch=config["direction_warmup_end_epoch"],
    )

    lambda_axis = wm * config["lambda_axis_direction"]
    lambda_fibrosis = wm * config["lambda_fibrosis_direction"]
    lambda_terminal_early = wm * config["lambda_terminal_early"]
    lambda_terminal_late = wm * config["lambda_terminal_late"]
    lambda_late_stability = wm * config["lambda_late_stability"]

    total_losses = []
    pred_losses = []
    axis_losses = []
    fibrosis_losses = []
    terminal_early_losses = []
    terminal_late_losses = []
    late_stability_losses = []

    for batch in loader:
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        c = batch["c"].to(device)
        t0 = batch["t0"].to(device)
        t1 = batch["t1"].to(device)

        stage_weight_axis = batch["stage_weight_axis"].to(device)
        stage_weight_fibrosis = batch["stage_weight_fibrosis"].to(device)
        stage_weight_terminal = batch["stage_weight_terminal"].to(device)
        stage_weight_late_terminal = batch["stage_weight_late_terminal"].to(device)
        stage_weight_late_stability = batch["stage_weight_late_stability"].to(device)

        optimizer.zero_grad()
        z1_pred = model(z0, c, t0, t1)

        pred_loss = mse(z1_pred, z1)
        dir_loss, dir_stats = stageaware_direction_constrained_loss(
            z0=z0,
            z1_pred=z1_pred,
            axis_direction=axis_direction_t,
            fibrosis_direction=fibrosis_direction_t,
            terminal_prototype=terminal_prototype_t,
            stage_weight_axis=stage_weight_axis,
            stage_weight_fibrosis=stage_weight_fibrosis,
            stage_weight_terminal=stage_weight_terminal,
            stage_weight_late_terminal=stage_weight_late_terminal,
            stage_weight_late_stability=stage_weight_late_stability,
            lambda_axis=lambda_axis,
            lambda_fibrosis=lambda_fibrosis,
            lambda_terminal_early=lambda_terminal_early,
            lambda_terminal_late=lambda_terminal_late,
            lambda_late_stability=lambda_late_stability,
        )

        loss = pred_loss + dir_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_losses.append(float(loss.detach().cpu().item()))
        pred_losses.append(float(pred_loss.detach().cpu().item()))
        axis_losses.append(dir_stats["loss_axis_direction"])
        fibrosis_losses.append(dir_stats["loss_fibrosis_direction"])
        terminal_early_losses.append(dir_stats["loss_terminal_early"])
        terminal_late_losses.append(dir_stats["loss_terminal_late"])
        late_stability_losses.append(dir_stats["loss_late_stability"])

    return {
        "direction_warmup_multiplier": float(wm),
        "effective_lambda_axis_direction": float(lambda_axis),
        "effective_lambda_fibrosis_direction": float(lambda_fibrosis),
        "effective_lambda_terminal_early": float(lambda_terminal_early),
        "effective_lambda_terminal_late": float(lambda_terminal_late),
        "effective_lambda_late_stability": float(lambda_late_stability),
        "train_total_loss": float(np.mean(total_losses)),
        "train_prediction_loss": float(np.mean(pred_losses)),
        "train_axis_direction_loss": float(np.mean(axis_losses)),
        "train_fibrosis_direction_loss": float(np.mean(fibrosis_losses)),
        "train_terminal_early_loss": float(np.mean(terminal_early_losses)),
        "train_terminal_late_loss": float(np.mean(terminal_late_losses)),
        "train_late_stability_loss": float(np.mean(late_stability_losses)),
    }


@torch.no_grad()
def evaluate(model: ConditionalNeuralODE, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    mse = nn.MSELoss()
    y_true = []
    y_pred = []
    losses = []

    for batch in loader:
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        c = batch["c"].to(device)
        t0 = batch["t0"].to(device)
        t1 = batch["t1"].to(device)

        pred = model(z0, c, t0, t1)
        loss = mse(pred, z1)

        losses.append(float(loss.detach().cpu().item()))
        y_true.append(z1.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    return {
        "loss": float(np.mean(losses)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def compute_refined_velocity_scores(
    model: ConditionalNeuralODE,
    z: np.ndarray,
    cond: np.ndarray,
    df: pd.DataFrame,
    direction_info: Dict,
    stage_weight_df: pd.DataFrame,
    device: torch.device,
    projection_delta_t: float,
    batch_size: int = 256,
) -> pd.DataFrame:
    model.eval()

    axis_direction = direction_info["axis_direction"]
    fibrosis_direction = direction_info["fibrosis_direction"]
    terminal_prototype = direction_info["terminal_prototype"]

    velocity_list = []
    z_after_list = []
    t_values = df["mean_axis_v2_score"].astype(float).values

    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        z_batch = torch.tensor(z[start:end], dtype=torch.float32).to(device)
        c_batch = torch.tensor(cond[start:end], dtype=torch.float32).to(device)

        t_mean = float(np.mean(t_values[start:end]))
        vel = model.velocity(z_batch, c_batch, t_value=t_mean)
        z_after = model.integrate_delta(z_batch, c_batch, delta_t=projection_delta_t)

        velocity_list.append(vel.cpu().numpy())
        z_after_list.append(z_after.cpu().numpy())

    velocity = np.vstack(velocity_list)
    z_after = np.vstack(z_after_list)

    velocity_norm = np.linalg.norm(velocity, axis=1)
    velocity_axis_projection = velocity @ axis_direction
    velocity_fibrosis_projection = velocity @ fibrosis_direction
    velocity_axis_cosine = cosine_similarity_to_vector(velocity, axis_direction)
    velocity_fibrosis_cosine = cosine_similarity_to_vector(velocity, fibrosis_direction)

    dist_before = np.linalg.norm(z - terminal_prototype.reshape(1, -1), axis=1)
    dist_after = np.linalg.norm(z_after - terminal_prototype.reshape(1, -1), axis=1)
    terminal_similarity_before = -dist_before
    terminal_similarity_after = -dist_after
    delta_terminal_similarity = terminal_similarity_after - terminal_similarity_before

    out_cols = [
        "cell_state_id",
        "sample_id",
        "Disease_Identity",
        "CellType_Category",
        "celltype_marker_based",
        "cluster_default",
        "axis_v2_stage",
        "n_cells",
        "fraction_in_sample",
        "mean_axis_v2_score",
    ]

    out = df[out_cols].copy()
    out = pd.concat([out, stage_weight_df.reset_index(drop=True)], axis=1)

    out["velocity_norm"] = velocity_norm
    out["velocity_axis_projection"] = velocity_axis_projection
    out["velocity_axis_cosine"] = velocity_axis_cosine
    out["velocity_fibrosis_projection"] = velocity_fibrosis_projection
    out["velocity_fibrosis_cosine"] = velocity_fibrosis_cosine
    out["terminal_similarity_before"] = terminal_similarity_before
    out["terminal_similarity_after"] = terminal_similarity_after
    out["delta_terminal_similarity"] = delta_terminal_similarity
    out["moves_toward_axis"] = out["velocity_axis_projection"] > 0
    out["moves_toward_fibrosis"] = out["velocity_fibrosis_projection"] > 0
    out["moves_toward_terminal"] = out["delta_terminal_similarity"] > 0

    for i in range(velocity.shape[1]):
        out[f"velocity_latent_{i + 1}"] = velocity[:, i]

    return out


# ============================================================
# 7. Plot helpers
# ============================================================
def plot_loss_curve(history: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(history["epoch"], history["train_total_loss"], label="train total")
    plt.plot(history["epoch"], history["train_prediction_loss"], label="train prediction")
    plt.plot(history["epoch"], history["val_loss"], label="validation prediction")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prediction_scatter(eval_result: Dict, out_path: str, title: str) -> None:
    y_true = eval_result["y_true"].ravel()
    y_pred = eval_result["y_pred"].ravel()
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lim_min = min(np.min(y_true), np.min(y_pred))
    lim_max = max(np.max(y_true), np.max(y_pred))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("Observed latent z1")
    plt.ylabel("Predicted latent z1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 8. Summary helpers
# ============================================================
def summarize_velocity(velocity_df: pd.DataFrame, ablation_name: str) -> Dict[str, pd.DataFrame]:
    summary_by_celltype = (
        velocity_df
        .groupby("celltype_marker_based")
        .agg(
            n_states=("cell_state_id", "count"),
            mean_axis_v2=("mean_axis_v2_score", "mean"),
            mean_velocity_norm=("velocity_norm", "mean"),
            median_velocity_norm=("velocity_norm", "median"),
            mean_axis_projection=("velocity_axis_projection", "mean"),
            mean_axis_cosine=("velocity_axis_cosine", "mean"),
            pct_moves_toward_axis=("moves_toward_axis", "mean"),
            mean_fibrosis_projection=("velocity_fibrosis_projection", "mean"),
            mean_fibrosis_cosine=("velocity_fibrosis_cosine", "mean"),
            pct_moves_toward_fibrosis=("moves_toward_fibrosis", "mean"),
            mean_delta_terminal_similarity=("delta_terminal_similarity", "mean"),
            pct_moves_toward_terminal=("moves_toward_terminal", "mean"),
        )
        .reset_index()
        .sort_values("mean_fibrosis_projection", ascending=False)
    )
    summary_by_celltype.insert(0, "ablation_name", ablation_name)

    summary_by_stage = (
        velocity_df
        .groupby("axis_v2_stage")
        .agg(
            n_states=("cell_state_id", "count"),
            mean_axis_v2=("mean_axis_v2_score", "mean"),
            mean_velocity_norm=("velocity_norm", "mean"),
            mean_axis_projection=("velocity_axis_projection", "mean"),
            pct_moves_toward_axis=("moves_toward_axis", "mean"),
            mean_fibrosis_projection=("velocity_fibrosis_projection", "mean"),
            pct_moves_toward_fibrosis=("moves_toward_fibrosis", "mean"),
            mean_delta_terminal_similarity=("delta_terminal_similarity", "mean"),
            pct_moves_toward_terminal=("moves_toward_terminal", "mean"),
        )
        .reset_index()
        .sort_values("mean_axis_v2")
    )
    summary_by_stage.insert(0, "ablation_name", ablation_name)

    summary_by_disease = (
        velocity_df
        .groupby("Disease_Identity")
        .agg(
            n_states=("cell_state_id", "count"),
            mean_axis_v2=("mean_axis_v2_score", "mean"),
            mean_velocity_norm=("velocity_norm", "mean"),
            mean_axis_projection=("velocity_axis_projection", "mean"),
            pct_moves_toward_axis=("moves_toward_axis", "mean"),
            mean_fibrosis_projection=("velocity_fibrosis_projection", "mean"),
            pct_moves_toward_fibrosis=("moves_toward_fibrosis", "mean"),
            mean_delta_terminal_similarity=("delta_terminal_similarity", "mean"),
            pct_moves_toward_terminal=("moves_toward_terminal", "mean"),
        )
        .reset_index()
        .sort_values("mean_axis_v2")
    )
    summary_by_disease.insert(0, "ablation_name", ablation_name)

    summary_by_cluster = (
        velocity_df
        .groupby(["celltype_marker_based", "cluster_default"])
        .agg(
            n_states=("cell_state_id", "count"),
            mean_axis_v2=("mean_axis_v2_score", "mean"),
            mean_velocity_norm=("velocity_norm", "mean"),
            mean_axis_projection=("velocity_axis_projection", "mean"),
            pct_moves_toward_axis=("moves_toward_axis", "mean"),
            mean_fibrosis_projection=("velocity_fibrosis_projection", "mean"),
            pct_moves_toward_fibrosis=("moves_toward_fibrosis", "mean"),
            mean_delta_terminal_similarity=("delta_terminal_similarity", "mean"),
            pct_moves_toward_terminal=("moves_toward_terminal", "mean"),
        )
        .reset_index()
        .sort_values("mean_fibrosis_projection", ascending=False)
    )
    summary_by_cluster.insert(0, "ablation_name", ablation_name)

    return {
        "celltype": summary_by_celltype,
        "stage": summary_by_stage,
        "disease": summary_by_disease,
        "cluster": summary_by_cluster,
    }


# ============================================================
# 9. Single ablation run
# ============================================================
def run_single_ablation(
    config: Dict,
    df: pd.DataFrame,
    device: torch.device,
    base_outdir: str,
    shared_pair_df_all: pd.DataFrame,
    shared_pair_df: pd.DataFrame,
    shared_pair_count_by_group: pd.DataFrame,
) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    ablation_name = config["ablation_name"]

    outdir = os.path.join(base_outdir, ablation_name)
    tab_dir = os.path.join(outdir, "tables")
    fig_dir = os.path.join(outdir, "figures")
    model_dir = os.path.join(outdir, "models")
    ensure_dir(outdir)
    ensure_dir(tab_dir)
    ensure_dir(fig_dir)
    ensure_dir(model_dir)

    print("\n" + "=" * 80)
    print("Running second-batch ablation:", ablation_name)
    print("family:", config["ablation_family"])
    print("latent_dim:", config["latent_dim"])
    print("solver:", config["solver"])
    print("use_time_input:", config["use_time_input"])
    print("use_pair_specific_delta_t:", config["use_pair_specific_delta_t"])

    save_json(config, os.path.join(tab_dir, "00_ablation_config.json"))

    # Each latent-dim ablation must recompute PCA and direction vectors.
    z, pca, feature_cols, fibrosis_col = compute_latent_state(df, config, tab_dir)

    direction_info = build_direction_vectors(
        df=df,
        z=z,
        fibrosis_col=fibrosis_col,
        terminal_top_quantile=config["terminal_top_quantile"],
        fibrosis_top_quantile=config["fibrosis_top_quantile"],
    )
    direction_info_to_save = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in direction_info.items()
    }
    save_json(direction_info_to_save, os.path.join(tab_dir, "01_direction_vectors_and_thresholds.json"))

    cond, condition_feature_names = make_condition_matrix(df, config["condition_mode"])
    pd.DataFrame({"condition_feature": condition_feature_names}).to_csv(
        os.path.join(tab_dir, "02_condition_features.csv"), index=False
    )

    stage_weight_df = build_stage_weight_matrix(df, config["stage_weights"])
    pd.concat(
        [
            df[["cell_state_id", "celltype_marker_based", "cluster_default", "axis_v2_stage"]].reset_index(drop=True),
            stage_weight_df.reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(os.path.join(tab_dir, "03_stage_aware_constraint_weights.csv"), index=False)

    # Reuse the same transition pairs for all second-batch runs.
    pair_df_all = shared_pair_df_all.copy()
    pair_df = shared_pair_df.copy()
    pair_count_by_group = shared_pair_count_by_group.copy()

    pair_df_all.to_csv(os.path.join(tab_dir, "04_transition_pairs_all_before_group_filter.csv"), index=False)
    pair_count_by_group.to_csv(os.path.join(tab_dir, "05_transition_pair_count_by_group.csv"), index=False)
    pair_df.to_csv(os.path.join(tab_dir, "06_transition_pairs_used_for_training.csv"), index=False)

    if pair_df.shape[0] < 20:
        raise ValueError(
            f"{ablation_name}: too few transition pairs after filtering: {pair_df.shape[0]}."
        )

    group_counts = pair_df["transition_group"].value_counts()
    can_stratify = pair_df["transition_group"].nunique() > 1 and group_counts.min() >= 2
    stratify_labels = pair_df["transition_group"] if can_stratify else None

    train_pairs, val_pairs = train_test_split(
        pair_df,
        test_size=config["test_size"],
        random_state=config["seed"],
        stratify=stratify_labels,
    )

    train_dataset = TransitionPairDataset(
        train_pairs,
        z=z,
        cond=cond,
        stage_weight_df=stage_weight_df,
    )
    val_dataset = TransitionPairDataset(
        val_pairs,
        z=z,
        cond=cond,
        stage_weight_df=stage_weight_df,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    axis_direction_t = torch.tensor(direction_info["axis_direction"], dtype=torch.float32).to(device)
    fibrosis_direction_t = torch.tensor(direction_info["fibrosis_direction"], dtype=torch.float32).to(device)
    terminal_prototype_t = torch.tensor(direction_info["terminal_prototype"], dtype=torch.float32).to(device)

    model = ConditionalNeuralODE(
        latent_dim=z.shape[1],
        cond_dim=cond.shape[1],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        solver=config["solver"],
        use_pair_specific_delta_t=config["use_pair_specific_delta_t"],
        use_time_input=config["use_time_input"],
        ode_rtol=config["ode_rtol"],
        ode_atol=config["ode_atol"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history = []
    best_val_loss = np.inf
    best_model_path = os.path.join(model_dir, f"best_{ablation_name}.pt")

    for epoch in range(1, config["epochs"] + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            axis_direction_t=axis_direction_t,
            fibrosis_direction_t=fibrosis_direction_t,
            terminal_prototype_t=terminal_prototype_t,
            config=config,
            epoch=epoch,
        )
        val_result = evaluate(model, val_loader, device=device)
        val_loss = val_result["loss"]

        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_loss,
            "val_mse": val_result["mse"],
            "val_r2": val_result["r2"],
        }
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "latent_dim": int(z.shape[1]),
                "cond_dim": int(cond.shape[1]),
                "feature_cols": feature_cols,
                "condition_feature_names": condition_feature_names.tolist(),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "fibrosis_col": fibrosis_col,
            }, best_model_path)

        if epoch == 1 or epoch % 25 == 0 or epoch == config["epochs"]:
            print(
                f"{ablation_name} | Epoch {epoch:04d} | "
                f"warmup={train_stats['direction_warmup_multiplier']:.2f} | "
                f"train_pred={train_stats['train_prediction_loss']:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_r2={val_result['r2']:.4f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(tab_dir, "07_training_history.csv"), index=False)

    plot_loss_curve(
        history_df,
        os.path.join(fig_dir, "01_training_loss_curve.pdf"),
        title=f"{ablation_name}: training loss",
    )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    train_eval = evaluate(model, train_loader, device=device)
    val_eval = evaluate(model, val_loader, device=device)

    plot_prediction_scatter(
        val_eval,
        os.path.join(fig_dir, "02_validation_observed_vs_predicted.pdf"),
        title=f"{ablation_name}: observed vs predicted",
    )

    velocity_df = compute_refined_velocity_scores(
        model=model,
        z=z,
        cond=cond,
        df=df,
        direction_info=direction_info,
        stage_weight_df=stage_weight_df,
        device=device,
        projection_delta_t=config["projection_delta_t"],
        batch_size=config["batch_size"],
    )
    velocity_df.to_csv(os.path.join(tab_dir, "08_state_velocity_scores.csv"), index=False)

    summaries = summarize_velocity(velocity_df, ablation_name=ablation_name)
    summaries["celltype"].to_csv(os.path.join(tab_dir, "09_directionality_by_celltype.csv"), index=False)
    summaries["stage"].to_csv(os.path.join(tab_dir, "10_directionality_by_axis_stage.csv"), index=False)
    summaries["disease"].to_csv(os.path.join(tab_dir, "11_directionality_by_disease.csv"), index=False)
    summaries["cluster"].to_csv(os.path.join(tab_dir, "12_directionality_by_celltype_cluster.csv"), index=False)

    val_pairs_out = val_pairs.reset_index(drop=True).copy()
    y_true = val_eval["y_true"]
    y_pred = val_eval["y_pred"]
    val_pairs_out["prediction_mse"] = np.mean((y_true - y_pred) ** 2, axis=1)
    val_pairs_out["prediction_l2_distance"] = np.linalg.norm(y_true - y_pred, axis=1)
    val_pairs_out.to_csv(os.path.join(tab_dir, "13_validation_pair_predictions.csv"), index=False)

    final_metrics = {
        "ablation_name": ablation_name,
        "ablation_family": config["ablation_family"],
        "device": str(device),
        "n_states": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "latent_dim": int(z.shape[1]),
        "pca_total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "n_transition_pairs_before_group_filter": int(pair_df_all.shape[0]),
        "n_transition_pairs": int(pair_df.shape[0]),
        "n_train_pairs": int(train_pairs.shape[0]),
        "n_val_pairs": int(val_pairs.shape[0]),
        "n_groups_before_filter": int(pair_count_by_group.shape[0]),
        "n_groups_used_for_training": int(pair_count_by_group["used_for_training"].sum()),
        "min_pairs_per_group": int(config["min_pairs_per_group"]),
        "best_val_loss": float(best_val_loss),
        "train_mse": float(train_eval["mse"]),
        "train_r2": float(train_eval["r2"]),
        "val_mse": float(val_eval["mse"]),
        "val_r2": float(val_eval["r2"]),
        "group_mode": config["group_mode"],
        "condition_mode": config["condition_mode"],
        "solver": config["solver"],
        "use_time_input": bool(config["use_time_input"]),
        "use_pair_specific_delta_t": bool(config["use_pair_specific_delta_t"]),
        "ode_rtol": float(config["ode_rtol"]),
        "ode_atol": float(config["ode_atol"]),
        "fibrosis_direction_column": fibrosis_col,
        "lambda_axis_direction": float(config["lambda_axis_direction"]),
        "lambda_fibrosis_direction": float(config["lambda_fibrosis_direction"]),
        "lambda_terminal_early": float(config["lambda_terminal_early"]),
        "lambda_terminal_late": float(config["lambda_terminal_late"]),
        "lambda_late_stability": float(config["lambda_late_stability"]),
        "direction_warmup_start_epoch": int(config["direction_warmup_start_epoch"]),
        "direction_warmup_end_epoch": int(config["direction_warmup_end_epoch"]),
    }
    save_json(final_metrics, os.path.join(tab_dir, "14_final_metrics.json"))

    print("Finished:", ablation_name)
    print("  val_r2:", final_metrics["val_r2"])
    print("  val_mse:", final_metrics["val_mse"])

    return final_metrics, summaries


# ============================================================
# 10. Main runner
# ============================================================
def run_second_batch_runner(base_config: Dict) -> None:
    set_seed(base_config["seed"])

    outdir = base_config["outdir"]
    tab_dir = os.path.join(outdir, "_aggregate_tables")
    ensure_dir(outdir)
    ensure_dir(tab_dir)

    device = get_device(force_cpu=base_config["force_cpu"])
    print("Device:", device)
    print("Input:", base_config["input_csv"])
    print("Output:", outdir)

    df = prepare_dataframe(base_config["input_csv"], min_cells=base_config["min_cells"])
    print("Loaded states:", df.shape)

    # Shared transition graph for all second-batch ablations.
    pair_df_all = build_transition_pairs(df, group_mode=base_config["group_mode"])
    pair_df, pair_count_by_group = filter_sparse_transition_groups(
        pair_df_all,
        min_pairs_per_group=base_config["min_pairs_per_group"],
    )

    pair_df_all.to_csv(os.path.join(tab_dir, "00_transition_pairs_all_before_group_filter.csv"), index=False)
    pair_count_by_group.to_csv(os.path.join(tab_dir, "00_transition_pair_count_by_group.csv"), index=False)
    pair_df.to_csv(os.path.join(tab_dir, "00_transition_pairs_used_for_training.csv"), index=False)

    print("Transition pairs before filtering:", pair_df_all.shape[0])
    print("Transition pairs used:", pair_df.shape[0])
    print("Groups before filtering:", pair_count_by_group.shape[0])
    print("Groups used:", int(pair_count_by_group["used_for_training"].sum()))

    ablation_configs = build_second_batch_configs(base_config)
    save_json(
        {cfg["ablation_name"]: cfg for cfg in ablation_configs},
        os.path.join(tab_dir, "00_all_second_batch_ablation_configs.json"),
    )

    all_metrics = []
    all_stage = []
    all_disease = []
    all_celltype = []
    all_cluster = []

    for cfg in ablation_configs:
        set_seed(cfg["seed"])
        metrics, summaries = run_single_ablation(
            config=cfg,
            df=df,
            device=device,
            base_outdir=outdir,
            shared_pair_df_all=pair_df_all,
            shared_pair_df=pair_df,
            shared_pair_count_by_group=pair_count_by_group,
        )
        all_metrics.append(metrics)
        all_stage.append(summaries["stage"])
        all_disease.append(summaries["disease"])
        all_celltype.append(summaries["celltype"])
        all_cluster.append(summaries["cluster"])

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(tab_dir, "second_batch_summary_metrics_partial.csv"),
            index=False,
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(tab_dir, "second_batch_summary_metrics.csv"), index=False)

    stage_df = pd.concat(all_stage, axis=0, ignore_index=True)
    disease_df = pd.concat(all_disease, axis=0, ignore_index=True)
    celltype_df = pd.concat(all_celltype, axis=0, ignore_index=True)
    cluster_df = pd.concat(all_cluster, axis=0, ignore_index=True)

    stage_df.to_csv(os.path.join(tab_dir, "second_batch_stage_directionality.csv"), index=False)
    disease_df.to_csv(os.path.join(tab_dir, "second_batch_disease_directionality.csv"), index=False)
    celltype_df.to_csv(os.path.join(tab_dir, "second_batch_celltype_directionality.csv"), index=False)
    cluster_df.to_csv(os.path.join(tab_dir, "second_batch_celltype_cluster_directionality.csv"), index=False)

    stage4 = stage_df[stage_df["axis_v2_stage"].astype(str) == "Stage_4_high"].copy()
    stage4.to_csv(os.path.join(tab_dir, "second_batch_stage4_terminal_check.csv"), index=False)

    top_clusters = (
        cluster_df[cluster_df["n_states"] >= 20]
        .sort_values(["ablation_name", "mean_fibrosis_projection"], ascending=[True, False])
        .groupby("ablation_name")
        .head(10)
        .reset_index(drop=True)
    )
    top_clusters.to_csv(os.path.join(tab_dir, "second_batch_top10_fibrosis_clusters_nstates20.csv"), index=False)

    top_terminal_clusters = (
        cluster_df[cluster_df["n_states"] >= 20]
        .sort_values(["ablation_name", "mean_delta_terminal_similarity"], ascending=[True, False])
        .groupby("ablation_name")
        .head(10)
        .reset_index(drop=True)
    )
    top_terminal_clusters.to_csv(os.path.join(tab_dir, "second_batch_top10_terminal_clusters_nstates20.csv"), index=False)

    print("\nSecond-batch ablations finished.")
    print("Aggregate outputs:")
    print("  ", os.path.join(tab_dir, "second_batch_summary_metrics.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_stage_directionality.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_disease_directionality.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_celltype_directionality.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_celltype_cluster_directionality.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_stage4_terminal_check.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_top10_fibrosis_clusters_nstates20.csv"))
    print("  ", os.path.join(tab_dir, "second_batch_top10_terminal_clusters_nstates20.csv"))


if __name__ == "__main__":
    run_second_batch_runner(BASE_CONFIG)