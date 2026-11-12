#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_OT_COMBINATION_BASELINE_AND_FLOW_MATCHING_FINAL.py
====================================================

用途
----
读取脚本28产生的复方候选空间，在一个脚本中完成：
1. 基于目标参考组合分布的距离基线与Sinkhorn OT基线；
2. 基于minibatch最优传输耦合的Conditional Flow Matching（OT-CFM）；
3. ODE正向/反向积分、组合评分、内部留出诊断和模型保存；
4. 为脚本30的基线比较与消融实验提供可复用函数。

重要边界
--------
目标分布来自脚本28的“非冗余计算优先级A”组合，并非真实双药协同标签。
因此本脚本输出的是“向当前高优先级计算分布靠近”的运输与一致性评分，不能
直接解释为Bliss、Loewe、ZIP、HSA协同、联合安全性、最佳剂量或临床有效性。

最低输入
--------
INPUT_ROOT/
  02_pair_candidate_space_all.csv
  03a_pair_priority_A_nonredundant.csv

Python依赖
----------
pandas, numpy, scipy, scikit-learn, torch, matplotlib
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn


SCRIPT_VERSION = "29.3.0-FINAL-OT-BASELINE-OT-CFM"
EPS = 1e-12

# =============================================================================
# PyCharm用户主要修改区
# =============================================================================
INPUT_ROOT = r"E:\PythonProject\results\38_COMBINATION_CANDIDATE_SPACE"
OUTPUT_ROOT = r"E:\PythonProject\results\39_OT_FLOW_MATCHING_COMBINATION_MODEL"
EXISTING_OUTPUT_POLICY = "overwrite"  # overwrite | timestamp | error
DEVICE = "auto"                       # auto | cpu | cuda
RANDOM_SEED = 20260719

PAIR_SPACE_FILE = "02_pair_candidate_space_all.csv"
TARGET_REFERENCE_FILE = "03a_pair_priority_A_nonredundant.csv"

TARGET_PAIR_TIER = "COMPUTATIONAL_PAIR_PRIORITY_A"
SOURCE_PAIR_TIERS = {
    "COMPUTATIONAL_PAIR_PRIORITY_B",
    "COMPUTATIONAL_PAIR_EXPLORATORY",
}
EXCLUDED_PAIR_TIER = "EXCLUDED"

CONTINUOUS_FEATURES: List[str] = [
    "drug_a_member_quality",
    "drug_b_member_quality",
    "member_strength_score",
    "query_evidence_score",
    "mechanism_union_score",
    "mechanism_gain_score",
    "source_robustness_score",
    "target_complementarity_score",
    "context_coverage_score",
    "n_members_with_sigcom_query_evidence",
    "n_primary_positive_union",
    "n_primary_exclusive_positive",
    "n_primary_overlapping_positive",
    "n_primary_source_robust_union",
    "n_primary_direction_conflicts",
    "n_primary_source_conflicts",
    "drug_a_mechanism_mean_normalized",
    "drug_b_mechanism_mean_normalized",
    "mechanism_gain_raw",
    "n_union_targets_main",
    "n_shared_targets_main",
    "target_jaccard_main",
    "n_aligned_direct_ot_targets_union",
]
BOOLEAN_FEATURES: List[str] = [
    "same_drug_family",
    "high_target_redundancy",
    "both_members_mechanism_unstable",
]

# 脚本30按这些机制模块进行成组消融。
FEATURE_GROUPS: Dict[str, List[str]] = {
    "member_quality": [
        "drug_a_member_quality",
        "drug_b_member_quality",
        "member_strength_score",
    ],
    "query_evidence": [
        "query_evidence_score",
        "n_members_with_sigcom_query_evidence",
    ],
    "mechanism_union": [
        "mechanism_union_score",
        "n_primary_positive_union",
        "n_primary_exclusive_positive",
        "n_primary_overlapping_positive",
    ],
    "mechanism_gain": [
        "mechanism_gain_score",
        "mechanism_gain_raw",
        "drug_a_mechanism_mean_normalized",
        "drug_b_mechanism_mean_normalized",
    ],
    "source_robustness": [
        "source_robustness_score",
        "n_primary_source_robust_union",
        "n_primary_source_conflicts",
    ],
    "target_complementarity": [
        "target_complementarity_score",
        "n_union_targets_main",
        "n_shared_targets_main",
        "target_jaccard_main",
        "n_aligned_direct_ot_targets_union",
        "same_drug_family",
        "high_target_redundancy",
    ],
    "context_and_conflict": [
        "context_coverage_score",
        "n_primary_direction_conflicts",
        "both_members_mechanism_unstable",
    ],
}

MODEL_INPUT_FORBIDDEN = {
    "pair_candidate_score",
    "pair_tier",
    "pair_global_rank",
    "pair_tier_reason",
    "pair_tier_order",
    "pair_exclusion_reasons",
}
# =============================================================================


@dataclass
class RobustTransform:
    columns: List[str]
    median: np.ndarray
    scale: np.ndarray
    clip: float = 8.0

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        z = (matrix - self.median) / self.scale
        return np.clip(z, -self.clip, self.clip)

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        return matrix * self.scale + self.median

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "clip": float(self.clip),
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "RobustTransform":
        return cls(
            columns=list(data["columns"]),
            median=np.asarray(data["median"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            clip=float(data.get("clip", 8.0)),
        )


@dataclass
class OTConfig:
    epsilon: float = 0.15
    max_iter: int = 2500
    tolerance: float = 1e-8


@dataclass
class FlowConfig:
    hidden_dim: int = 128
    n_hidden_blocks: int = 3
    batch_size: int = 64
    train_steps: int = 1500
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    validate_every: int = 50
    early_stopping_patience: int = 12
    min_delta: float = 1e-5
    coupling_mode: str = "ot"       # ot | random | nearest
    train_cache_batches: int = 160
    validation_cache_batches: int = 24
    ode_steps: int = 50
    ode_method: str = "rk4"         # rk4 | euler
    inference_batch_size: int = 256
    target_train_fraction: float = 0.80


class TimeConditionedVectorField(nn.Module):
    """用于表格特征的时间条件残差向量场 v_theta(x,t)。"""

    def __init__(self, dim: int, hidden_dim: int, n_hidden_blocks: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden_blocks = int(n_hidden_blocks)
        self.time_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.input_layer = nn.Linear(dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_hidden_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        te = self.time_net(t)
        h = self.input_layer(torch.cat([x, te], dim=1))
        for block in self.blocks:
            h = h + block(h)
        return self.output_layer(h)


def normalize_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str) -> torch.device:
    requested = normalize_text(requested).casefold()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda'，但当前环境没有可用CUDA。")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("DEVICE只能是auto、cpu或cuda。")
    return torch.device(requested)


def choose_output_dir(path: Path, policy: str) -> Path:
    policy = normalize_text(policy).casefold()
    if path.exists() and any(path.iterdir()):
        if policy == "overwrite":
            shutil.rmtree(path)
        elif policy == "timestamp":
            path = path.parent / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif policy == "error":
            raise FileExistsError(f"输出目录非空：{path}")
        else:
            raise ValueError(f"未知EXISTING_OUTPUT_POLICY：{policy}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def bool_to_float(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1.0, "1": 1.0, "yes": 1.0, "y": 1.0,
        "false": 0.0, "0": 0.0, "no": 0.0, "n": 0.0, "": 0.0,
    }
    text = series.fillna("").astype("string").str.strip().str.casefold()
    return text.map(mapping).fillna(0.0).astype(float)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{label}缺少字段：{missing}")


def load_pair_inputs(
    input_root: Path,
    pair_space_file: str = PAIR_SPACE_FILE,
    target_reference_file: str = TARGET_REFERENCE_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_path = input_root / pair_space_file
    target_path = input_root / target_reference_file
    if not pair_path.exists():
        raise FileNotFoundError(f"找不到复方候选空间：{pair_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"找不到非冗余A参考组合：{target_path}")

    pairs = pd.read_csv(pair_path, low_memory=False)
    target_reference = pd.read_csv(target_path, low_memory=False)
    if pairs.empty:
        raise ValueError("复方候选空间为空。")
    if target_reference.empty:
        raise ValueError("非冗余A参考组合为空。")

    required = [
        "pair_id", "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name",
        "pair_tier", "pair_candidate_score", *CONTINUOUS_FEATURES, *BOOLEAN_FEATURES,
    ]
    require_columns(pairs, required, pair_space_file)
    require_columns(target_reference, ["pair_id"], target_reference_file)

    pairs = pairs.drop_duplicates("pair_id", keep="first").reset_index(drop=True)
    target_reference = target_reference.drop_duplicates("pair_id", keep="first").reset_index(drop=True)
    return pairs, target_reference


def prepare_model_frame(
    pairs: pd.DataFrame,
    target_reference: pd.DataFrame,
) -> pd.DataFrame:
    frame = pairs.copy()
    target_ids = set(target_reference["pair_id"].astype(str))
    frame["pair_tier"] = frame["pair_tier"].fillna("").astype(str)
    frame["is_excluded_pair"] = frame["pair_tier"].eq(EXCLUDED_PAIR_TIER)
    frame["is_target_reference"] = frame["pair_id"].astype(str).isin(target_ids)
    frame["is_source_distribution"] = frame["pair_tier"].isin(SOURCE_PAIR_TIERS)
    valid = frame.loc[~frame["is_excluded_pair"]].copy()
    valid = valid.reset_index(drop=False).rename(columns={"index": "original_row_index"})
    if len(valid) < 20:
        raise ValueError("未排除组合少于20个，无法稳定训练。")
    if int(valid["is_target_reference"].sum()) < 8:
        raise ValueError("非冗余目标参考组合少于8个，无法稳定训练。")
    if int(valid["is_source_distribution"].sum()) < 20:
        raise ValueError("B级/探索性源组合少于20个，无法稳定训练。")
    return valid


def selected_feature_columns(
    include_columns: Optional[Sequence[str]] = None,
    exclude_groups: Optional[Sequence[str]] = None,
) -> List[str]:
    available = CONTINUOUS_FEATURES + BOOLEAN_FEATURES
    if include_columns is None:
        columns = list(available)
    else:
        unknown = sorted(set(include_columns) - set(available))
        if unknown:
            raise KeyError(f"未知特征：{unknown}")
        columns = [column for column in available if column in set(include_columns)]

    for group in exclude_groups or []:
        if group not in FEATURE_GROUPS:
            raise KeyError(f"未知特征组：{group}")
        remove = set(FEATURE_GROUPS[group])
        columns = [column for column in columns if column not in remove]

    forbidden = sorted(set(columns) & MODEL_INPUT_FORBIDDEN)
    if forbidden:
        raise ValueError(f"模型输入包含禁止的结果泄漏字段：{forbidden}")
    if len(columns) < 2:
        raise ValueError("剩余特征少于2个。")
    return columns


def build_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    imputation_values: Optional[Mapping[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    require_columns(frame, feature_columns, "模型数据")
    output = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column in BOOLEAN_FEATURES:
            output[column] = bool_to_float(frame[column])
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce")

    if imputation_values is None:
        imputation = {
            column: float(output[column].median()) if output[column].notna().any() else 0.0
            for column in feature_columns
        }
    else:
        imputation = {column: float(imputation_values[column]) for column in feature_columns}
    output = output.fillna(imputation).fillna(0.0).astype(float)
    return output, imputation


def remove_zero_variance_columns(
    feature_frame: pd.DataFrame,
    variance_floor: float = 1e-12,
) -> Tuple[pd.DataFrame, List[str]]:
    variance = feature_frame.var(axis=0, ddof=0)
    kept = variance[variance > variance_floor].index.tolist()
    dropped = variance[variance <= variance_floor].index.tolist()
    if len(kept) < 2:
        raise ValueError("移除零方差特征后不足2个特征。")
    return feature_frame[kept].copy(), dropped


def fit_robust_transform(matrix: np.ndarray, columns: Sequence[str]) -> RobustTransform:
    matrix = np.asarray(matrix, dtype=np.float64)
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    return RobustTransform(list(columns), median.astype(float), scale.astype(float))


def squared_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    cost = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
    return cost / max(x.shape[1], 1)


def empirical_scale(values: np.ndarray, default: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return float(default)
    value = float(np.median(finite))
    return value if np.isfinite(value) and value > EPS else float(default)


def sinkhorn_plan(
    cost: np.ndarray,
    config: OTConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """均匀边际、熵正则的Sinkhorn OT。"""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        raise ValueError("OT代价矩阵为空。")
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)

    positive = cost[np.isfinite(cost) & (cost > 0)]
    cost_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(cost_scale) or cost_scale <= EPS:
        cost_scale = 1.0
    normalized_cost = cost / cost_scale
    epsilon = max(float(config.epsilon), 1e-4)
    kernel = np.exp(-np.clip(normalized_cost / epsilon, 0.0, 700.0))
    kernel = np.maximum(kernel, 1e-300)

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    residual = np.inf
    completed = 0
    for iteration in range(int(config.max_iter)):
        old_u = u.copy()
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        residual = float(np.max(np.abs(u - old_u)))
        completed = iteration + 1
        if residual < config.tolerance:
            break

    plan = (u[:, None] * kernel) * v[None, :]
    plan_sum = float(plan.sum())
    if not np.isfinite(plan_sum) or plan_sum <= 0:
        raise FloatingPointError("Sinkhorn运输计划数值失败。")
    plan /= plan_sum
    diagnostics = {
        "iterations": completed,
        "last_u_residual": residual,
        "row_marginal_max_error": float(np.max(np.abs(plan.sum(axis=1) - a))),
        "column_marginal_max_error": float(np.max(np.abs(plan.sum(axis=0) - b))),
        "cost_scale": cost_scale,
        "epsilon_on_scaled_cost": epsilon,
    }
    return plan, diagnostics


def compute_distance_baselines(
    x_all: np.ndarray,
    x_target_train: np.ndarray,
) -> pd.DataFrame:
    cost = squared_cost(x_all, x_target_train)
    nearest_cost = np.min(cost, axis=1)
    centroid = np.mean(x_target_train, axis=0, keepdims=True)
    centroid_cost = squared_cost(x_all, centroid).reshape(-1)
    nearest_score = np.exp(-nearest_cost / empirical_scale(nearest_cost))
    centroid_score = np.exp(-centroid_cost / empirical_scale(centroid_cost))
    return pd.DataFrame(
        {
            "nearest_target_cost": nearest_cost,
            "nearest_target_score": np.clip(nearest_score, 0.0, 1.0),
            "target_centroid_cost": centroid_cost,
            "target_centroid_score": np.clip(centroid_score, 0.0, 1.0),
        }
    )


def compute_ot_baseline(
    x_all: np.ndarray,
    x_target_train: np.ndarray,
    config: OTConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    cost = squared_cost(x_all, x_target_train)
    plan, diagnostics = sinkhorn_plan(cost, config)
    row_mass = plan.sum(axis=1, keepdims=True)
    conditional = plan / np.maximum(row_mass, 1e-300)
    barycentric = conditional @ x_target_train
    transport_cost = np.sum(conditional * cost, axis=1)
    nearest_cost = np.min(cost, axis=1)
    barycentric_distance = np.mean((x_all - barycentric) ** 2, axis=1)

    transport_similarity = np.exp(-transport_cost / empirical_scale(transport_cost))
    nearest_similarity = np.exp(-nearest_cost / empirical_scale(nearest_cost))
    barycentric_similarity = np.exp(
        -barycentric_distance / empirical_scale(barycentric_distance)
    )
    ot_score = (
        0.45 * transport_similarity
        + 0.35 * nearest_similarity
        + 0.20 * barycentric_similarity
    )
    result = pd.DataFrame(
        {
            "ot_transport_cost": transport_cost,
            "ot_nearest_target_cost": nearest_cost,
            "ot_barycentric_distance": barycentric_distance,
            "ot_transport_similarity": transport_similarity,
            "ot_nearest_target_similarity": nearest_similarity,
            "ot_barycentric_similarity": barycentric_similarity,
            "ot_baseline_score": np.clip(ot_score, 0.0, 1.0),
        }
    )
    diagnostics.update(
        {
            "n_rows_scored": int(len(x_all)),
            "n_target_train": int(len(x_target_train)),
            "transport_cost_scale": empirical_scale(transport_cost),
            "nearest_cost_scale": empirical_scale(nearest_cost),
            "barycentric_distance_scale": empirical_scale(barycentric_distance),
        }
    )
    return result, diagnostics, barycentric


def split_source_target_indices(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    seed: int,
    train_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = np.asarray(source_indices, dtype=int).copy()
    target = np.asarray(target_indices, dtype=int).copy()
    rng.shuffle(source)
    rng.shuffle(target)

    def split_one(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_train = int(round(len(values) * train_fraction))
        n_train = min(max(n_train, 2), len(values) - 1)
        return values[:n_train], values[n_train:]

    source_train, source_holdout = split_one(source)
    target_train, target_holdout = split_one(target)
    return source_train, source_holdout, target_train, target_holdout


def _pair_batch(
    x0: np.ndarray,
    x1: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = normalize_text(mode).casefold()
    if len(x0) != len(x1):
        raise ValueError("配对批次必须等长。")
    if mode == "random":
        return x0, x1[rng.permutation(len(x1))]

    cost = squared_cost(x0, x1)
    if mode == "ot":
        row_ind, col_ind = linear_sum_assignment(cost)
        order = np.argsort(row_ind)
        return x0[row_ind[order]], x1[col_ind[order]]
    if mode == "nearest":
        nearest = np.argmin(cost, axis=1)
        return x0, x1[nearest]
    raise ValueError("coupling_mode只能是ot、random或nearest。")


def build_coupling_cache(
    source: np.ndarray,
    target: np.ndarray,
    n_batches: int,
    batch_size: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        raise ValueError("源分布或目标分布为空。")
    paired_x0: List[np.ndarray] = []
    paired_x1: List[np.ndarray] = []
    for _ in range(int(n_batches)):
        source_idx = rng.integers(0, len(source), size=batch_size)
        target_idx = rng.integers(0, len(target), size=batch_size)
        x0 = source[source_idx].astype(np.float32, copy=False)
        x1 = target[target_idx].astype(np.float32, copy=False)
        p0, p1 = _pair_batch(x0, x1, mode, rng)
        paired_x0.append(p0)
        paired_x1.append(p1)
    return np.vstack(paired_x0), np.vstack(paired_x1)


def cfm_loss_paired(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    t = torch.rand((x0.shape[0], 1), device=x0.device, dtype=x0.dtype)
    xt = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted_velocity = model(xt, t)
    return torch.mean((predicted_velocity - target_velocity) ** 2)


def evaluate_cfm_cache(
    model: nn.Module,
    x0_cache: np.ndarray,
    x1_cache: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    batch_size: int,
    repeats: int = 5,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            idx = rng.integers(0, len(x0_cache), size=min(batch_size, len(x0_cache)))
            x0 = torch.as_tensor(x0_cache[idx], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(x1_cache[idx], dtype=torch.float32, device=device)
            losses.append(float(cfm_loss_paired(model, x0, x1).item()))
    return float(np.mean(losses))


def train_flow_matching(
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_validation: np.ndarray,
    target_validation: np.ndarray,
    device: torch.device,
    seed: int,
    config: FlowConfig,
    verbose: bool = True,
) -> Tuple[TimeConditionedVectorField, pd.DataFrame, Dict[str, Any]]:
    set_seed(seed)
    model = TimeConditionedVectorField(
        dim=source_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_hidden_blocks=config.n_hidden_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.train_steps, 1),
        eta_min=config.learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"[FM] 构建{config.coupling_mode}耦合缓存……")
    train_x0, train_x1 = build_coupling_cache(
        source_train,
        target_train,
        config.train_cache_batches,
        config.batch_size,
        config.coupling_mode,
        rng,
    )
    val_x0, val_x1 = build_coupling_cache(
        source_validation,
        target_validation,
        config.validation_cache_batches,
        config.batch_size,
        config.coupling_mode,
        rng,
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_monitor = float("inf")
    patience = 0

    for step in range(1, config.train_steps + 1):
        model.train()
        idx = rng.integers(0, len(train_x0), size=config.batch_size)
        x0 = torch.as_tensor(train_x0[idx], dtype=torch.float32, device=device)
        x1 = torch.as_tensor(train_x1[idx], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = cfm_loss_paired(model, x0, x1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Flow Matching损失非有限值，step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        should_validate = (
            step == 1
            or step % config.validate_every == 0
            or step == config.train_steps
        )
        if should_validate:
            validation_loss = evaluate_cfm_cache(
                model,
                val_x0,
                val_x1,
                device,
                rng,
                config.batch_size,
            )
            train_loss = float(loss.item())
            history.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if verbose:
                print(
                    f"[FM] step={step:4d} train={train_loss:.6f} "
                    f"validation={validation_loss:.6f}"
                )
            monitor = validation_loss if np.isfinite(validation_loss) else train_loss
            if monitor < best_monitor - config.min_delta:
                best_monitor = monitor
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    if verbose:
                        print(f"[FM] early stopping at step={step}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    diagnostics = {
        "best_validation_loss": best_monitor if np.isfinite(best_monitor) else None,
        "steps_completed": int(history[-1]["step"] if history else 0),
        "early_stopped": bool(history and history[-1]["step"] < config.train_steps),
        "device": str(device),
        "dimension": int(source_train.shape[1]),
        "flow_config": asdict(config),
        "n_train_cache_pairs": int(len(train_x0)),
        "n_validation_cache_pairs": int(len(val_x0)),
    }
    return model, pd.DataFrame(history), diagnostics


def _vector_field(
    model: nn.Module,
    x: torch.Tensor,
    t_scalar: float,
) -> torch.Tensor:
    t = torch.full((x.shape[0], 1), float(t_scalar), device=x.device, dtype=x.dtype)
    return model(x, t)


def integrate_flow_batch(
    model: nn.Module,
    x0: torch.Tensor,
    config: FlowConfig,
    reverse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    method = normalize_text(config.ode_method).casefold()
    if method not in {"rk4", "euler"}:
        raise ValueError("ode_method只能是rk4或euler。")
    direction = -1.0 if reverse else 1.0
    dt = direction / config.ode_steps
    x = x0.clone()
    path_length = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    energy = torch.zeros_like(path_length)

    with torch.no_grad():
        for step in range(config.ode_steps):
            if reverse:
                t = 1.0 - step / config.ode_steps
            else:
                t = step / config.ode_steps
            if method == "euler":
                velocity = _vector_field(model, x, t)
                delta = dt * velocity
            else:
                k1 = _vector_field(model, x, t)
                k2 = _vector_field(model, x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = _vector_field(model, x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = _vector_field(model, x + dt * k3, t + dt)
                velocity = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                delta = dt * velocity
            x = x + delta
            path_length += torch.linalg.norm(delta, dim=1)
            energy += torch.sum(velocity * velocity, dim=1) * abs(dt)

    displacement = torch.linalg.norm(x - x0, dim=1)
    straightness = displacement / torch.clamp(path_length, min=1e-8)
    return x, path_length, energy, straightness


def integrate_all(
    model: nn.Module,
    x_all: np.ndarray,
    device: torch.device,
    config: FlowConfig,
    reverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    endpoints: List[np.ndarray] = []
    lengths: List[np.ndarray] = []
    energies: List[np.ndarray] = []
    straightness: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(x_all), config.inference_batch_size):
        batch = torch.as_tensor(
            x_all[start : start + config.inference_batch_size],
            dtype=torch.float32,
            device=device,
        )
        endpoint, length, energy, straight = integrate_flow_batch(
            model, batch, config, reverse=reverse
        )
        endpoints.append(endpoint.cpu().numpy())
        lengths.append(length.cpu().numpy())
        energies.append(energy.cpu().numpy())
        straightness.append(straight.cpu().numpy())
    return (
        np.vstack(endpoints),
        np.concatenate(lengths),
        np.concatenate(energies),
        np.concatenate(straightness),
    )


def compute_flow_scores(
    x_all: np.ndarray,
    forward_endpoints: np.ndarray,
    reverse_endpoints: np.ndarray,
    x_target_train: np.ndarray,
    x_source_train: np.ndarray,
    forward_length: np.ndarray,
    forward_energy: np.ndarray,
    forward_straightness: np.ndarray,
    reverse_length: np.ndarray,
    reverse_energy: np.ndarray,
    reverse_straightness: np.ndarray,
) -> pd.DataFrame:
    initial_target_cost = np.min(squared_cost(x_all, x_target_train), axis=1)
    terminal_target_cost = np.min(squared_cost(forward_endpoints, x_target_train), axis=1)
    reverse_source_cost = np.min(squared_cost(reverse_endpoints, x_source_train), axis=1)
    forward_displacement = np.linalg.norm(forward_endpoints - x_all, axis=1) / math.sqrt(x_all.shape[1])
    reverse_displacement = np.linalg.norm(reverse_endpoints - x_all, axis=1) / math.sqrt(x_all.shape[1])

    initial_similarity = np.exp(-initial_target_cost / empirical_scale(initial_target_cost))
    terminal_similarity = np.exp(-terminal_target_cost / empirical_scale(terminal_target_cost))
    reverse_source_similarity = np.exp(-reverse_source_cost / empirical_scale(reverse_source_cost))
    low_forward_effort = np.exp(-forward_displacement / empirical_scale(forward_displacement))
    low_forward_energy = np.exp(-forward_energy / empirical_scale(forward_energy))
    transport_improvement = np.clip(
        (initial_target_cost - terminal_target_cost)
        / np.maximum(initial_target_cost, 1e-8),
        -1.0,
        1.0,
    )
    transport_improvement_score = (transport_improvement + 1.0) / 2.0

    flow_score = (
        0.25 * initial_similarity
        + 0.25 * terminal_similarity
        + 0.15 * reverse_source_similarity
        + 0.15 * low_forward_effort
        + 0.10 * low_forward_energy
        + 0.05 * np.clip(forward_straightness, 0.0, 1.0)
        + 0.05 * transport_improvement_score
    )
    return pd.DataFrame(
        {
            "fm_initial_nearest_target_cost": initial_target_cost,
            "fm_terminal_nearest_target_cost": terminal_target_cost,
            "fm_reverse_nearest_source_cost": reverse_source_cost,
            "fm_forward_displacement_norm": forward_displacement,
            "fm_reverse_displacement_norm": reverse_displacement,
            "fm_forward_path_length": forward_length,
            "fm_reverse_path_length": reverse_length,
            "fm_forward_path_energy": forward_energy,
            "fm_reverse_path_energy": reverse_energy,
            "fm_forward_path_straightness": forward_straightness,
            "fm_reverse_path_straightness": reverse_straightness,
            "fm_transport_improvement_fraction": transport_improvement,
            "fm_initial_target_similarity": initial_similarity,
            "fm_terminal_target_similarity": terminal_similarity,
            "fm_reverse_source_similarity": reverse_source_similarity,
            "fm_low_forward_effort_similarity": low_forward_effort,
            "fm_low_forward_energy_similarity": low_forward_energy,
            "flow_matching_score": np.clip(flow_score, 0.0, 1.0),
        }
    )


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_holdout_scores(
    target_holdout_indices: np.ndarray,
    source_holdout_indices: np.ndarray,
    score_frame: pd.DataFrame,
    score_columns: Sequence[str],
) -> Dict[str, Any]:
    indices = np.concatenate([target_holdout_indices, source_holdout_indices])
    labels = np.concatenate(
        [
            np.ones(len(target_holdout_indices), dtype=int),
            np.zeros(len(source_holdout_indices), dtype=int),
        ]
    )
    output: Dict[str, Any] = {
        "n_target_holdout": int(len(target_holdout_indices)),
        "n_source_holdout": int(len(source_holdout_indices)),
        "interpretation": "internal_computational_target_recovery_only",
    }
    for column in score_columns:
        output[column] = safe_binary_metrics(
            labels,
            score_frame.iloc[indices][column].to_numpy(float),
        )
    return output


def rank_descending(values: pd.Series) -> pd.Series:
    return values.rank(method="first", ascending=False).astype(int)


def save_pca_diagnostic(
    x_all: np.ndarray,
    forward_endpoints: np.ndarray,
    target_mask: np.ndarray,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    combined = np.vstack([x_all, forward_endpoints])
    coordinates = pca.fit_transform(combined)
    n = len(x_all)
    start = coordinates[:n]
    endpoint = coordinates[n:]
    plt.figure(figsize=(9, 7))
    plt.scatter(
        start[~target_mask, 0], start[~target_mask, 1],
        s=14, alpha=0.45, label="B/Exploratory start",
    )
    plt.scatter(
        start[target_mask, 0], start[target_mask, 1],
        s=24, alpha=0.75, label="Priority A reference",
    )
    plt.scatter(
        endpoint[:, 0], endpoint[:, 1],
        s=12, alpha=0.30, marker="x", label="Forward endpoint",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("OT-CFM pair-feature transport diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def save_training_plot(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history["step"], history["train_loss"], label="Train")
    plt.plot(history["step"], history["validation_loss"], label="Validation")
    plt.xlabel("Training step")
    plt.ylabel("Flow Matching loss")
    plt.title("OT-CFM training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def train_and_score_once(
    valid: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
    device: torch.device,
    ot_config: OTConfig,
    flow_config: FlowConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """脚本29与脚本30共享的单次划分、训练和评分入口。"""
    feature_frame, imputation = build_feature_frame(valid, feature_columns)
    feature_frame, zero_variance = remove_zero_variance_columns(feature_frame)
    final_columns = feature_frame.columns.tolist()

    target_indices = np.flatnonzero(valid["is_target_reference"].to_numpy(bool))
    source_indices = np.flatnonzero(valid["is_source_distribution"].to_numpy(bool))
    source_train_idx, source_holdout_idx, target_train_idx, target_holdout_idx = (
        split_source_target_indices(
            source_indices,
            target_indices,
            seed,
            flow_config.target_train_fraction,
        )
    )

    # 变换仅在训练源+训练目标上拟合，避免留出泄漏。
    transform_fit_idx = np.concatenate([source_train_idx, target_train_idx])
    transform = fit_robust_transform(
        feature_frame.iloc[transform_fit_idx].to_numpy(float),
        final_columns,
    )
    x_all = transform.transform(feature_frame.to_numpy(float)).astype(np.float32)

    x_source_train = x_all[source_train_idx]
    x_source_holdout = x_all[source_holdout_idx]
    x_target_train = x_all[target_train_idx]
    x_target_holdout = x_all[target_holdout_idx]

    distance_scores = compute_distance_baselines(x_all, x_target_train)
    ot_scores, ot_diagnostics, barycentric = compute_ot_baseline(
        x_all, x_target_train, ot_config
    )
    model, history, training_diagnostics = train_flow_matching(
        x_source_train,
        x_target_train,
        x_source_holdout,
        x_target_holdout,
        device,
        seed,
        flow_config,
        verbose=verbose,
    )
    forward_endpoints, forward_length, forward_energy, forward_straightness = integrate_all(
        model, x_all, device, flow_config, reverse=False
    )
    reverse_endpoints, reverse_length, reverse_energy, reverse_straightness = integrate_all(
        model, x_all, device, flow_config, reverse=True
    )
    flow_scores = compute_flow_scores(
        x_all,
        forward_endpoints,
        reverse_endpoints,
        x_target_train,
        x_source_train,
        forward_length,
        forward_energy,
        forward_straightness,
        reverse_length,
        reverse_energy,
        reverse_straightness,
    )

    scores = pd.concat(
        [distance_scores.reset_index(drop=True), ot_scores, flow_scores], axis=1
    )
    scores["ot_fm_model_score"] = (
        0.45 * scores["ot_baseline_score"]
        + 0.55 * scores["flow_matching_score"]
    )
    original = pd.to_numeric(valid["pair_candidate_score"], errors="coerce").fillna(0.0)
    original = np.clip(original.to_numpy(float), 0.0, 1.0)
    scores["original_pair_candidate_score"] = original
    scores["ot_fm_consensus_score"] = (
        0.70 * scores["ot_fm_model_score"]
        + 0.30 * original
    )
    metrics = evaluate_holdout_scores(
        target_holdout_idx,
        source_holdout_idx,
        scores,
        [
            "nearest_target_score",
            "target_centroid_score",
            "ot_baseline_score",
            "flow_matching_score",
            "ot_fm_model_score",
            "ot_fm_consensus_score",
        ],
    )
    return {
        "feature_columns": final_columns,
        "imputation_values": {key: imputation[key] for key in final_columns},
        "zero_variance_dropped": zero_variance,
        "transform": transform,
        "x_all": x_all,
        "source_train_indices": source_train_idx,
        "source_holdout_indices": source_holdout_idx,
        "target_train_indices": target_train_idx,
        "target_holdout_indices": target_holdout_idx,
        "distance_scores": distance_scores,
        "ot_scores": ot_scores,
        "flow_scores": flow_scores,
        "scores": scores,
        "ot_barycentric": barycentric,
        "model": model,
        "history": history,
        "forward_endpoints": forward_endpoints,
        "reverse_endpoints": reverse_endpoints,
        "ot_diagnostics": ot_diagnostics,
        "training_diagnostics": training_diagnostics,
        "holdout_metrics": metrics,
        "ot_config": copy.deepcopy(ot_config),
        "flow_config": copy.deepcopy(flow_config),
        "seed": int(seed),
    }


def write_methods_file(
    path: Path,
    feature_columns: Sequence[str],
    ot_config: OTConfig,
    flow_config: FlowConfig,
) -> None:
    text = f"""脚本版本：{SCRIPT_VERSION}

模型定义
========
目标分布：03a_pair_priority_A_nonredundant.csv中的非冗余计算优先级A组合。
源分布：脚本28中的计算优先级B与探索性组合。
排除组合：脚本28中pair_tier=EXCLUDED的组合。

泄漏控制
========
模型输入明确排除pair_candidate_score、pair_tier、pair_global_rank、pair_tier_reason
及其他脚本28结果字段。Robust变换仅在训练源样本和训练目标样本上拟合。

OT基线
======
Sinkhorn epsilon={ot_config.epsilon}，max_iter={ot_config.max_iter}，
tolerance={ot_config.tolerance}。输出运输代价、最近目标距离、OT重心距离和
OT基线分数。

Flow Matching
=============
耦合方式：{flow_config.coupling_mode}
条件路径：x_t=(1-t)x_0+t x_1
目标速度：u_t=x_1-x_0
损失：E||v_theta(x_t,t)-u_t||^2
网络：hidden_dim={flow_config.hidden_dim}，residual_blocks={flow_config.n_hidden_blocks}
训练：steps={flow_config.train_steps}，batch={flow_config.batch_size}
ODE：{flow_config.ode_method.upper()}，steps={flow_config.ode_steps}

输入特征
========
""" + "\n".join(f"- {column}" for column in feature_columns) + """

解释边界
========
1. 内部AUC/AP只反映对脚本28计算目标的恢复能力，不是外部生物学验证。
2. Flow Matching在连续特征空间运输，不直接生成新药物身份。
3. 输出不等于协同、联合安全、最佳剂量或临床有效性证据。
4. 获得真实组合剂量-反应或双药扰动表达后，应替换目标定义并做药物家族外部留出。
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(
    input_root: Path,
    output_root: Path,
    seed: int = RANDOM_SEED,
    device_name: str = DEVICE,
    output_policy: str = EXISTING_OUTPUT_POLICY,
    ot_config: Optional[OTConfig] = None,
    flow_config: Optional[FlowConfig] = None,
) -> Path:
    set_seed(seed)
    device = resolve_device(device_name)
    ot_config = ot_config or OTConfig()
    flow_config = flow_config or FlowConfig()
    output_root = choose_output_dir(output_root, output_policy)

    pairs, target_reference = load_pair_inputs(input_root)
    valid = prepare_model_frame(pairs, target_reference)
    columns = selected_feature_columns()
    result = train_and_score_once(
        valid,
        columns,
        seed,
        device,
        ot_config,
        flow_config,
        verbose=True,
    )

    score_frame = pd.concat(
        [valid.reset_index(drop=True), result["scores"].reset_index(drop=True)],
        axis=1,
    )
    for column in [
        "nearest_target_score",
        "target_centroid_score",
        "ot_baseline_score",
        "flow_matching_score",
        "ot_fm_model_score",
        "ot_fm_consensus_score",
    ]:
        score_frame[f"{column}_rank"] = rank_descending(score_frame[column])
    score_frame["model_interpretation_boundary"] = (
        "computational_transport_to_priority_distribution_not_synergy_evidence"
    )
    score_frame = score_frame.sort_values(
        ["ot_fm_consensus_score_rank", "pair_id"]
    )

    score_frame.to_csv(output_root / "01_ot_fm_combination_scores_all.csv", index=False)
    score_frame.head(50).to_csv(output_root / "02_top50_ot_fm_consensus.csv", index=False)
    score_frame[score_frame["is_target_reference"]].to_csv(
        output_root / "03_target_reference_rescored.csv", index=False
    )
    result["history"].to_csv(
        output_root / "04_flow_matching_training_history.csv", index=False
    )

    transform: RobustTransform = result["transform"]
    forward_original = transform.inverse_transform(result["forward_endpoints"])
    reverse_original = transform.inverse_transform(result["reverse_endpoints"])
    barycentric_original = transform.inverse_transform(result["ot_barycentric"])

    def endpoint_frame(values: np.ndarray, prefix: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            values,
            columns=[f"{prefix}__{column}" for column in result["feature_columns"]],
        )
        frame.insert(0, "pair_id", valid["pair_id"].astype(str).to_numpy())
        return frame

    endpoint_frame(forward_original, "forward_endpoint").to_csv(
        output_root / "05_forward_flow_endpoints_original_feature_scale.csv", index=False
    )
    endpoint_frame(reverse_original, "reverse_endpoint").to_csv(
        output_root / "06_reverse_flow_endpoints_original_feature_scale.csv", index=False
    )
    endpoint_frame(barycentric_original, "ot_barycenter").to_csv(
        output_root / "07_ot_barycentric_projections.csv", index=False
    )

    checkpoint = {
        "script_version": SCRIPT_VERSION,
        "model_state_dict": result["model"].state_dict(),
        "feature_columns": result["feature_columns"],
        "imputation_values": result["imputation_values"],
        "robust_transform": transform.to_jsonable(),
        "ot_config": asdict(ot_config),
        "flow_config": asdict(flow_config),
        "seed": seed,
    }
    torch.save(checkpoint, output_root / "08_ot_cfm_model_checkpoint.pt")

    save_pca_diagnostic(
        result["x_all"],
        result["forward_endpoints"],
        valid["is_target_reference"].to_numpy(bool),
        output_root / "09_ot_cfm_pca_diagnostic.png",
    )
    save_training_plot(
        result["history"],
        output_root / "10_flow_matching_training_history.png",
    )
    write_methods_file(
        output_root / "11_methods_and_interpretation.txt",
        result["feature_columns"],
        ot_config,
        flow_config,
    )

    qc = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "run_time": datetime.now().isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "device": str(device),
        "n_pair_rows_input": int(len(pairs)),
        "n_excluded_pair_rows": int((pairs["pair_tier"].astype(str) == EXCLUDED_PAIR_TIER).sum()),
        "n_valid_pairs_modeled": int(len(valid)),
        "n_target_reference": int(valid["is_target_reference"].sum()),
        "n_source_distribution": int(valid["is_source_distribution"].sum()),
        "n_features": int(len(result["feature_columns"])),
        "feature_columns": result["feature_columns"],
        "zero_variance_dropped": result["zero_variance_dropped"],
        "split": {
            "n_source_train": int(len(result["source_train_indices"])),
            "n_source_holdout": int(len(result["source_holdout_indices"])),
            "n_target_train": int(len(result["target_train_indices"])),
            "n_target_holdout": int(len(result["target_holdout_indices"])),
        },
        "ot_diagnostics": result["ot_diagnostics"],
        "flow_training_diagnostics": result["training_diagnostics"],
        "holdout_diagnostics": result["holdout_metrics"],
        "top10_pairs": score_frame.head(10)[
            ["pair_id", "ot_fm_consensus_score", "ot_fm_consensus_score_rank"]
        ].to_dict("records"),
        "interpretation_boundary": (
            "internal_computational_transport_model_not_experimental_synergy_validation"
        ),
    }
    (output_root / "12_model_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    transform_payload = transform.to_jsonable()
    transform_payload["imputation_values"] = result["imputation_values"]
    (output_root / "13_feature_transform.json").write_text(
        json.dumps(transform_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n完成。主要结果：")
    print(output_root / "01_ot_fm_combination_scores_all.csv")
    print(output_root / "02_top50_ot_fm_consensus.csv")
    print(output_root / "12_model_qc.json")
    print("重要：当前目标是脚本28计算优先级，不是真实实验协同标签。")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OT复方基线与OT-CFM训练")
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--device", default=DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-steps", type=int, default=FlowConfig.train_steps)
    parser.add_argument("--batch-size", type=int, default=FlowConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=FlowConfig.hidden_dim)
    parser.add_argument("--hidden-blocks", type=int, default=FlowConfig.n_hidden_blocks)
    parser.add_argument("--ode-steps", type=int, default=FlowConfig.ode_steps)
    parser.add_argument("--coupling", default="ot", choices=["ot", "random", "nearest"])
    parser.add_argument("--output-policy", default=EXISTING_OUTPUT_POLICY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_config = FlowConfig(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_hidden_blocks=args.hidden_blocks,
        ode_steps=args.ode_steps,
        coupling_mode=args.coupling,
    )
    run_pipeline(
        Path(os.path.expanduser(os.path.expandvars(args.input_root))),
        Path(os.path.expanduser(os.path.expandvars(args.output_root))),
        seed=args.seed,
        device_name=args.device,
        output_policy=args.output_policy,
        flow_config=flow_config,
    )


if __name__ == "__main__":
    main()
