#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35a_run_ot_disease_reversal_v3_robust.py

Subject-balanced, model-specific optimal-transport (OT) disease-reversal analysis
for PPI-informed driver candidates.

Main design choices
-------------------
1. Runs separate disease x cell-type models (default: IPF/COPD x Macrophage/Dendritic).
2. Uses Stage_3_intermediate_high as the disease source and matched Control cells as target.
3. Derives disease_direction and reversal_direction from each model's expression data.
4. Excludes driver genes from OT geometry by default to reduce circularity.
5. Gives each subject equal total mass so subjects with more cells do not dominate.
6. Produces strict Tier-A primary and Tier-A+B sensitivity outputs.
7. Uses local OT pairing with a permutation null instead of the algebraically
   constrained global balanced-OT mean shift.
8. Adds multi-seed stability, repeated leakage-controlled subject validation,
   leave-one-subject-out influence analysis, and strict consensus.
9. Adds whole-pipeline subject-label permutations and OT/geometry sensitivity grids.
10. Adds transport-plan quality gates and resumable robustness runs.
11. Uses a numerically stable internal log-domain balanced Sinkhorn solver; POT is
   only needed if optional unbalanced OT is selected.

Input options
-------------
A. H5AD:
   - h5ad_path
   - optional expression_layer or use_raw

B. Wide expression matrix + metadata:
   - expression_matrix_path: rows=observations, columns=genes, first column=index
   - metadata_path: rows=observations, first column=index

The script is intended for active-progression disease reversal. It must not be
reported as terminal-state reversal unless terminal-state source models are run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


SCRIPT_VERSION = "3.0.0"
EPS = 1e-12


DEFAULT_CONFIG: Dict[str, Any] = {
    "input_mode": "h5ad",  # h5ad | wide_matrix
    "h5ad_path": r"E:\\PythonProject\\data\\integrated_scRNA.h5ad",
    "expression_matrix_path": "",
    "metadata_path": "",
    "tier_a_driver_path": r"E:\\PythonProject\\results\\19a_ot_reversal_tier_a_driver_input.csv",
    "supported_driver_path": r"E:\\PythonProject\\results\\19_ot_reversal_driver_input.csv",
    "output_root": r"E:\\PythonProject\\results\\robustness_runs_v3\\08_ot_disease_reversal",
    "analysis_id": "OT_DISEASE_REVERSAL_v3_ROBUST",
    "existing_output_policy": "resume",  # error | timestamp | overwrite | resume
    "expression": {
        "expression_layer": None,
        "use_raw": False,
        "input_is_log1p": True,
        "normalization_target_sum": 10000.0,
        "gene_symbol_column_in_var": None,
    },
    "metadata_columns": {
        "disease": "Disease_Identity",
        "celltype": "celltype_marker_based",
        "stage": "axis_v2_stage",
        "subject": "Subject_Identity",
        "cluster": "cluster_default",
    },
    "models": [
        {
            "model_id": "IPF_Macrophage_Stage3_to_Control",
            "source_disease": ["IPF"],
            "source_celltype": ["Macrophage"],
            "source_stage": ["Stage_3_intermediate_high"],
            "target_disease": ["Control"],
            "target_celltype": ["Macrophage"],
            "target_stage": None,
        },
        {
            "model_id": "IPF_Dendritic_Stage3_to_Control",
            "source_disease": ["IPF"],
            "source_celltype": ["Dendritic"],
            "source_stage": ["Stage_3_intermediate_high"],
            "target_disease": ["Control"],
            "target_celltype": ["Dendritic"],
            "target_stage": None,
        },
        {
            "model_id": "COPD_Macrophage_Stage3_to_Control",
            "source_disease": ["COPD"],
            "source_celltype": ["Macrophage"],
            "source_stage": ["Stage_3_intermediate_high"],
            "target_disease": ["Control"],
            "target_celltype": ["Macrophage"],
            "target_stage": None,
        },
        {
            "model_id": "COPD_Dendritic_Stage3_to_Control",
            "source_disease": ["COPD"],
            "source_celltype": ["Dendritic"],
            "source_stage": ["Stage_3_intermediate_high"],
            "target_disease": ["Control"],
            "target_celltype": ["Dendritic"],
            "target_stage": None,
        },
    ],
    "matching": {
        "case_sensitive": False,
        "strip_whitespace": True,
    },
    "sampling": {
        "max_source_observations": 1500,
        "max_target_observations": 1500,
        "min_observations_per_arm": 100,
        "min_subjects_per_arm": 3,
        "balance_subjects": True,
        "stratify_within_subject_by_cluster": True,
        "random_seed": 123,
    },
    "geometry": {
        "n_variable_genes": 2000,
        "exclude_driver_genes": True,
        "n_pcs": 30,
        "scale_clip": 10.0,
    },
    "ot": {
        "method": "balanced_sinkhorn_log",  # balanced_sinkhorn_log | unbalanced_sinkhorn
        "regularization": 0.05,
        "unbalanced_marginal_regularization": 1.0,
        "max_iterations": 5000,
        "stop_threshold": 1e-8,
        "check_every": 25,
        "cost_normalization": "median_positive",  # median_positive | max | none
        "cost_metric": "sqeuclidean",  # sqeuclidean | euclidean | cosine | correlation
        "save_transport_plan": False,
        "transport_plan_qc": {
            "enabled": True,
            "require_for_single_run_eligibility": True,
            "min_mean_effective_target_neighbors": 2.0,
            "max_mean_dominant_target_weight": 0.95,
            "min_normalized_row_entropy": 0.05,
            "max_normalized_row_entropy": 0.98,
            "min_target_mass_coverage": 0.90,
        },
    },
    "direction": {
        "min_abs_disease_effect_logexpr": 0.05,
        "min_abs_ot_alignment_z": 0.05,
    },
    "bootstrap": {
        "n_iterations": 500,
        "confidence_level": 0.95,
        "random_seed": 456,
    },
    # Legacy ranking settings are retained for configuration compatibility, but
    # v3 eligibility and ranking use strict_filtering/ranking_v2 below.
    "ranking": {
        "ppi_prior_weight": 0.45,
        "ot_alignment_weight": 0.35,
        "source_subject_consistency_weight": 0.20,
        "min_source_subject_consistency": 0.50,
        "require_positive_bootstrap_lower_ci": False,
    },
    "strict_filtering": {
        "min_abs_disease_effect_z": 0.10,
        "min_abs_ot_local_pairing_corr": 0.10,
        "max_ot_local_pairing_fdr": 0.10,
        "min_source_subject_direction_consistency": 0.75,
        "require_positive_subject_bootstrap_ci": True,
        "require_permutation_significance": True,
    },
    "ranking_v2": {
        "ppi_prior_weight": 0.35,
        "disease_effect_weight": 0.25,
        "ot_local_pairing_weight": 0.25,
        "source_subject_consistency_weight": 0.15,
    },
    "permutation": {
        "n_iterations": 200,
        "random_seed": 789,
    },
    "robustness": {
        "enabled": True,
        "sampling_seeds": [123, 223, 323],
        "min_seed_eligibility_rate": 0.66,
        "require_direction_agreement": True,
        "permutation_iterations_per_repeat": 100,
        "subject_split_replication": {
            "enabled": False,
            "validation_fraction": 0.30,
            "random_seed": 901,
            "min_subjects_per_arm": 3,
            "require_for_final_eligibility": False,
        },
        "repeated_subject_validation": {
            "enabled": True,
            "n_repeats": 20,
            "min_repeats_completed": 15,
            "validation_fraction": 0.30,
            "random_seed": 1901,
            "stratify_subject_columns": [],
            "min_subjects_per_arm": 3,
            "min_validation_eligibility_rate": 0.70,
            "min_direction_agreement_rate": 0.90,
            "permutation_iterations_per_repeat": 100,
            "require_for_final_eligibility": True,
        },
        "leave_one_subject_out": {
            "enabled": True,
            "max_subjects_per_arm": 0,
            "min_runs_completed_fraction": 0.90,
            "random_seed": 2901,
            "permutation_iterations_per_run": 75,
            "min_eligibility_rate": 0.70,
            "require_no_direction_flip": True,
            "require_for_final_eligibility": True,
        },
        "sensitivity_grid": {
            "enabled": True,
            "min_eligibility_rate": 0.60,
            "min_direction_agreement_rate": 0.90,
            "min_settings_completed": 5,
            "require_for_final_eligibility": True,
            "settings": [
                {"setting_id": "base", "ot_regularization": 0.05, "cost_metric": "sqeuclidean", "n_pcs": 30, "n_variable_genes": 2000},
                {"setting_id": "eps_003", "ot_regularization": 0.03, "cost_metric": "sqeuclidean", "n_pcs": 30, "n_variable_genes": 2000},
                {"setting_id": "eps_010", "ot_regularization": 0.10, "cost_metric": "sqeuclidean", "n_pcs": 30, "n_variable_genes": 2000},
                {"setting_id": "cosine", "ot_regularization": 0.05, "cost_metric": "cosine", "n_pcs": 30, "n_variable_genes": 2000},
                {"setting_id": "pcs_20", "ot_regularization": 0.05, "cost_metric": "sqeuclidean", "n_pcs": 20, "n_variable_genes": 2000},
                {"setting_id": "genes_1000", "ot_regularization": 0.05, "cost_metric": "sqeuclidean", "n_pcs": 30, "n_variable_genes": 1000}
            ]
        },
        "pipeline_label_permutation": {
            "enabled": True,
            "n_iterations": 100,
            "min_iterations_completed": 90,
            "random_seed": 3901,
            "block_columns": [],
            "local_pairing_iterations": 50,
            "max_pipeline_fdr": 0.10,
            "require_for_final_eligibility": True,
        },
        "resume_completed_runs": True,
    },
    "consensus": {
        "min_models_eligible": 3,
        "min_model_eligibility_rate": 0.75,
        "require_direction_consistency": True,
        "require_all_eligible_ci_positive": True,
        "require_seed_stability": True,
        "require_subject_split_replication": False,
        "require_repeated_subject_validation": True,
        "require_loso_stability": True,
        "require_sensitivity_stability": True,
        "require_pipeline_permutation_significance": True,
        "require_partial_conjunction_significance": True,
        "max_partial_conjunction_fdr": 0.10,
    },
    "strict_missing_models": False,
}


@dataclass
class ExpressionDataset:
    X: Any
    obs: pd.DataFrame
    var_names: np.ndarray
    source_description: str


@dataclass
class DriverInputs:
    all_supported: pd.DataFrame
    tier_a_genes: set
    supported_genes: set
    gene_column: str


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: Any, config_dir: Path, required: bool = True) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        if required:
            raise FileNotFoundError("A required path is empty.")
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def choose_output_dir(root: Path, analysis_id: str, policy: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / analysis_id
    policy = str(policy).strip().lower()
    if not target.exists() or not any(target.iterdir()):
        target.mkdir(parents=True, exist_ok=True)
        return target
    if policy == "error":
        raise FileExistsError(
            f"Output directory exists and is non-empty: {target}. "
            "Use existing_output_policy='timestamp' or 'overwrite'."
        )
    if policy == "timestamp":
        target = root / f"{analysis_id}_{_now_stamp()}"
        target.mkdir(parents=True, exist_ok=False)
        return target
    if policy == "overwrite":
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=False)
        return target
    if policy == "resume":
        target.mkdir(parents=True, exist_ok=True)
        return target
    raise ValueError(f"Unknown existing_output_policy: {policy!r}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def detect_gene_column(df: pd.DataFrame) -> str:
    candidates = [
        "gene_symbol",
        "gene",
        "Gene",
        "symbol",
        "gene_name",
        "features",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find a gene column. Available columns: {list(df.columns)[:30]}")


def normalize_gene_symbols(values: Iterable[Any]) -> pd.Series:
    return pd.Series(values, dtype="string").str.strip().str.upper()


def load_driver_inputs(tier_a_path: Path, supported_path: Path) -> DriverInputs:
    tier_a = read_table(tier_a_path)
    supported = read_table(supported_path)
    tier_a_gene_col = detect_gene_column(tier_a)
    supported_gene_col = detect_gene_column(supported)

    tier_a = tier_a.copy()
    supported = supported.copy()
    tier_a["gene_symbol"] = normalize_gene_symbols(tier_a[tier_a_gene_col])
    supported["gene_symbol"] = normalize_gene_symbols(supported[supported_gene_col])
    tier_a = tier_a[tier_a["gene_symbol"].notna() & (tier_a["gene_symbol"] != "")]
    supported = supported[supported["gene_symbol"].notna() & (supported["gene_symbol"] != "")]
    tier_a = tier_a.drop_duplicates("gene_symbol", keep="first")
    supported = supported.drop_duplicates("gene_symbol", keep="first")

    tier_a_genes = set(tier_a["gene_symbol"])
    supported_genes = set(supported["gene_symbol"]) | tier_a_genes

    # Prefer the supported table because it contains the complete Tier A+B set.
    all_supported = supported.copy()
    missing_tier_a = tier_a[~tier_a["gene_symbol"].isin(all_supported["gene_symbol"])]
    if not missing_tier_a.empty:
        all_supported = pd.concat([all_supported, missing_tier_a], ignore_index=True, sort=False)
    all_supported["is_tier_a_primary"] = all_supported["gene_symbol"].isin(tier_a_genes)
    all_supported["is_tier_ab_supported"] = all_supported["gene_symbol"].isin(supported_genes)

    if "ppi_driver_tier" not in all_supported.columns:
        all_supported["ppi_driver_tier"] = np.where(
            all_supported["is_tier_a_primary"],
            "TIER_A_HIGH_CONFIDENCE",
            "TIER_B_ROBUST_BUT_CAUTION",
        )

    return DriverInputs(
        all_supported=all_supported,
        tier_a_genes=tier_a_genes,
        supported_genes=supported_genes,
        gene_column="gene_symbol",
    )


def make_unique_gene_names(values: Sequence[Any]) -> np.ndarray:
    names = normalize_gene_symbols(values).fillna("").to_numpy(dtype=str)
    seen: Dict[str, int] = {}
    output: List[str] = []
    for name in names:
        base = name if name else "UNNAMED_GENE"
        count = seen.get(base, 0)
        output.append(base if count == 0 else f"{base}__DUP{count}")
        seen[base] = count + 1
    return np.asarray(output, dtype=str)


def load_expression_dataset(config: Mapping[str, Any], config_dir: Path) -> Tuple[ExpressionDataset, Dict[str, Any]]:
    mode = str(config.get("input_mode", "h5ad")).strip().lower()
    expr_cfg = config["expression"]
    manifest: Dict[str, Any] = {"input_mode": mode}

    if mode == "h5ad":
        h5ad_path = resolve_path(config.get("h5ad_path"), config_dir, required=True)
        assert h5ad_path is not None
        try:
            import anndata as ad  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "H5AD input requires anndata. Install with: pip install anndata"
            ) from exc
        adata = ad.read_h5ad(h5ad_path)
        use_raw = bool(expr_cfg.get("use_raw", False))
        layer = expr_cfg.get("expression_layer")
        if use_raw:
            if adata.raw is None:
                raise ValueError("expression.use_raw=True but adata.raw is absent.")
            X = adata.raw.X
            var = adata.raw.var.copy()
            var_names_original = np.asarray(adata.raw.var_names.astype(str))
            source_desc = f"{h5ad_path}::raw.X"
        elif layer:
            if layer not in adata.layers:
                raise KeyError(f"Layer {layer!r} is absent. Available: {list(adata.layers.keys())}")
            X = adata.layers[layer]
            var = adata.var.copy()
            var_names_original = np.asarray(adata.var_names.astype(str))
            source_desc = f"{h5ad_path}::layers[{layer!r}]"
        else:
            X = adata.X
            var = adata.var.copy()
            var_names_original = np.asarray(adata.var_names.astype(str))
            source_desc = f"{h5ad_path}::X"

        gene_col = expr_cfg.get("gene_symbol_column_in_var")
        if gene_col:
            if gene_col not in var.columns:
                raise KeyError(f"gene_symbol_column_in_var={gene_col!r} is absent from adata.var")
            var_names = make_unique_gene_names(var[gene_col].astype(str).to_numpy())
        else:
            var_names = make_unique_gene_names(var_names_original)

        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        manifest.update(
            {
                "h5ad_path": str(h5ad_path),
                "h5ad_sha256": sha256_file(h5ad_path),
                "expression_source": source_desc,
                "n_observations": int(adata.n_obs),
                "n_genes": int(len(var_names)),
            }
        )
        return ExpressionDataset(X=X, obs=obs, var_names=var_names, source_description=source_desc), manifest

    if mode == "wide_matrix":
        matrix_path = resolve_path(config.get("expression_matrix_path"), config_dir, required=True)
        metadata_path = resolve_path(config.get("metadata_path"), config_dir, required=True)
        assert matrix_path is not None and metadata_path is not None
        expr = read_table(matrix_path)
        meta = read_table(metadata_path)
        if expr.shape[1] < 2:
            raise ValueError("Wide expression matrix must contain an ID column and at least one gene column.")
        if meta.shape[1] < 2:
            raise ValueError("Metadata must contain an ID column and metadata columns.")
        expr_id_col = expr.columns[0]
        meta_id_col = meta.columns[0]
        expr = expr.set_index(expr_id_col)
        meta = meta.set_index(meta_id_col)
        expr.index = expr.index.astype(str)
        meta.index = meta.index.astype(str)
        common = expr.index.intersection(meta.index)
        if len(common) == 0:
            raise ValueError("Expression matrix and metadata have no shared observation IDs.")
        expr = expr.loc[common]
        meta = meta.loc[common]
        numeric = expr.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            bad = int(numeric.isna().sum().sum())
            raise ValueError(f"Expression matrix contains {bad} non-numeric or missing values.")
        X = numeric.to_numpy(dtype=np.float64, copy=False)
        var_names = make_unique_gene_names(numeric.columns.astype(str).to_numpy())
        manifest.update(
            {
                "expression_matrix_path": str(matrix_path),
                "expression_matrix_sha256": sha256_file(matrix_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "n_observations": int(len(common)),
                "n_genes": int(len(var_names)),
            }
        )
        return ExpressionDataset(
            X=X,
            obs=meta,
            var_names=var_names,
            source_description=f"{matrix_path} + {metadata_path}",
        ), manifest

    raise ValueError(f"Unknown input_mode: {mode!r}")


def validate_metadata(obs: pd.DataFrame, columns: Mapping[str, str]) -> None:
    required = [columns[k] for k in ("disease", "celltype", "stage", "subject")]
    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise KeyError(f"Missing required metadata columns: {missing}. Available: {list(obs.columns)}")


def _normalize_label(value: Any, case_sensitive: bool, strip_whitespace: bool) -> str:
    text = "" if pd.isna(value) else str(value)
    if strip_whitespace:
        text = text.strip()
    return text if case_sensitive else text.casefold()


def make_match_mask(series: pd.Series, allowed: Optional[Sequence[Any]], matching: Mapping[str, Any]) -> np.ndarray:
    if allowed is None:
        return np.ones(len(series), dtype=bool)
    if isinstance(allowed, (str, bytes)):
        allowed = [allowed]
    case_sensitive = bool(matching.get("case_sensitive", False))
    strip_whitespace = bool(matching.get("strip_whitespace", True))
    allowed_norm = {
        _normalize_label(x, case_sensitive, strip_whitespace)
        for x in allowed
    }
    values = series.map(lambda x: _normalize_label(x, case_sensitive, strip_whitespace))
    return values.isin(allowed_norm).to_numpy(dtype=bool)


def build_model_masks(
    obs: pd.DataFrame,
    model: Mapping[str, Any],
    columns: Mapping[str, str],
    matching: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    disease_col = columns["disease"]
    celltype_col = columns["celltype"]
    stage_col = columns["stage"]
    source = (
        make_match_mask(obs[disease_col], model.get("source_disease"), matching)
        & make_match_mask(obs[celltype_col], model.get("source_celltype"), matching)
        & make_match_mask(obs[stage_col], model.get("source_stage"), matching)
    )
    target = (
        make_match_mask(obs[disease_col], model.get("target_disease"), matching)
        & make_match_mask(obs[celltype_col], model.get("target_celltype"), matching)
        & make_match_mask(obs[stage_col], model.get("target_stage"), matching)
    )
    return source, target


def _sample_within_subject(
    obs: pd.DataFrame,
    indices: np.ndarray,
    quota: int,
    cluster_col: Optional[str],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(indices) <= quota:
        return np.asarray(indices, dtype=int)
    if cluster_col is None or cluster_col not in obs.columns:
        return np.sort(rng.choice(indices, size=quota, replace=False))

    sub = obs.iloc[indices]
    groups = sub.groupby(cluster_col, dropna=False, observed=False).indices
    selected: List[int] = []
    remaining_quota = quota
    remaining_n = len(indices)
    items = list(groups.items())
    for pos, (_, local_positions) in enumerate(items):
        global_indices = indices[np.asarray(local_positions, dtype=int)]
        if pos == len(items) - 1:
            take = min(remaining_quota, len(global_indices))
        else:
            proportional = int(round(quota * len(global_indices) / max(len(indices), 1)))
            take = min(len(global_indices), max(1, proportional))
            take = min(take, remaining_quota)
        if take > 0:
            chosen = global_indices if take == len(global_indices) else rng.choice(global_indices, size=take, replace=False)
            selected.extend(np.asarray(chosen, dtype=int).tolist())
            remaining_quota -= take
            remaining_n -= len(global_indices)
        if remaining_quota <= 0:
            break

    if len(selected) < quota:
        remaining = np.setdiff1d(indices, np.asarray(selected, dtype=int), assume_unique=False)
        add_n = min(quota - len(selected), len(remaining))
        if add_n > 0:
            selected.extend(rng.choice(remaining, size=add_n, replace=False).tolist())
    return np.sort(np.asarray(selected[:quota], dtype=int))


def subject_balanced_sample(
    obs: pd.DataFrame,
    eligible_indices: np.ndarray,
    max_n: int,
    subject_col: str,
    cluster_col: Optional[str],
    balance_subjects: bool,
    stratify_cluster: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    eligible_indices = np.asarray(eligible_indices, dtype=int)
    if len(eligible_indices) <= max_n:
        return np.sort(eligible_indices)
    if not balance_subjects or subject_col not in obs.columns:
        return np.sort(rng.choice(eligible_indices, size=max_n, replace=False))

    subset = obs.iloc[eligible_indices]
    subject_values = subset[subject_col].astype("string").fillna("__MISSING_SUBJECT__")
    subject_groups: Dict[str, np.ndarray] = {}
    for subject, local_positions in subject_values.groupby(subject_values, observed=False).groups.items():
        local_positions_arr = subset.index.get_indexer_for(local_positions)
        subject_groups[str(subject)] = eligible_indices[local_positions_arr]

    subjects = sorted(subject_groups)
    n_subjects = len(subjects)
    base = max_n // max(n_subjects, 1)
    remainder = max_n % max(n_subjects, 1)
    selected: List[int] = []
    leftovers: List[int] = []
    for i, subject in enumerate(subjects):
        group_idx = subject_groups[subject]
        quota = base + (1 if i < remainder else 0)
        quota = min(quota, len(group_idx))
        chosen = _sample_within_subject(
            obs,
            group_idx,
            quota,
            cluster_col if stratify_cluster else None,
            rng,
        )
        selected.extend(chosen.tolist())
        leftovers.extend(np.setdiff1d(group_idx, chosen, assume_unique=False).tolist())

    if len(selected) < max_n and leftovers:
        add_n = min(max_n - len(selected), len(leftovers))
        selected.extend(rng.choice(np.asarray(leftovers, dtype=int), size=add_n, replace=False).tolist())
    return np.sort(np.asarray(selected[:max_n], dtype=int))


def make_subject_balanced_weights(obs: pd.DataFrame, indices: np.ndarray, subject_col: str) -> np.ndarray:
    n = len(indices)
    if n == 0:
        return np.array([], dtype=float)
    if subject_col not in obs.columns:
        return np.full(n, 1.0 / n, dtype=float)
    subjects = obs.iloc[indices][subject_col].astype("string").fillna("__MISSING_SUBJECT__").to_numpy()
    unique, counts = np.unique(subjects, return_counts=True)
    n_subjects = len(unique)
    count_map = dict(zip(unique, counts))
    weights = np.asarray([1.0 / n_subjects / count_map[s] for s in subjects], dtype=float)
    weights /= weights.sum()
    return weights


def subset_rows_cols(X: Any, rows: np.ndarray, cols: Optional[np.ndarray] = None) -> Any:
    result = X[rows]
    if cols is not None:
        result = result[:, cols]
    return result


def normalize_log1p_matrix(X: Any, input_is_log1p: bool, target_sum: float) -> Any:
    if input_is_log1p:
        if sparse.issparse(X):
            return X.astype(np.float64, copy=True)
        return np.asarray(X, dtype=np.float64)

    if sparse.issparse(X):
        out = X.tocsr().astype(np.float64, copy=True)
        totals = np.asarray(out.sum(axis=1)).ravel()
        factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        out = sparse.diags(factors) @ out
        out.data = np.log1p(out.data)
        return out

    out = np.asarray(X, dtype=np.float64).copy()
    totals = out.sum(axis=1)
    factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    out *= factors[:, None]
    np.log1p(out, out=out)
    return out


def matrix_mean_variance(X: Any) -> Tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean**2, 0.0)
        return mean, var
    arr = np.asarray(X, dtype=np.float64)
    return arr.mean(axis=0), arr.var(axis=0, ddof=0)


def as_dense(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def select_geometry_genes(
    X_combined: Any,
    var_names: np.ndarray,
    driver_genes: set,
    n_variable_genes: int,
    exclude_driver_genes: bool,
) -> np.ndarray:
    _, variance = matrix_mean_variance(X_combined)
    valid = np.isfinite(variance) & (variance > 0)
    if exclude_driver_genes:
        valid &= ~np.isin(var_names, np.asarray(sorted(driver_genes), dtype=str))
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        raise ValueError("No non-constant genes are available for OT geometry.")
    order = candidates[np.argsort(variance[candidates])[::-1]]
    return order[: min(n_variable_genes, len(order))]


def standardize_and_pca(
    X: np.ndarray,
    n_pcs: int,
    clip: float,
    random_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    keep = np.isfinite(stds) & (stds > EPS)
    if keep.sum() < 2:
        raise ValueError("Fewer than two variable geometry genes remain after scaling.")
    scaled = (X[:, keep] - means[keep]) / stds[keep]
    if clip and clip > 0:
        scaled = np.clip(scaled, -clip, clip)
    n_components = min(int(n_pcs), scaled.shape[0] - 1, scaled.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Not enough observations/genes for PCA.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_seed)
    embedding = pca.fit_transform(scaled)
    qc = {
        "n_scaled_genes": int(keep.sum()),
        "n_pcs": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio_first10": pca.explained_variance_ratio_[:10].tolist(),
    }
    return embedding, qc


def fit_and_apply_standardize_pca(
    X_fit: np.ndarray,
    X_apply: np.ndarray,
    n_pcs: int,
    clip: float,
    random_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit scaling/PCA only on X_fit and transform X_apply without leakage."""
    X_fit = np.asarray(X_fit, dtype=float)
    X_apply = np.asarray(X_apply, dtype=float)
    means = X_fit.mean(axis=0)
    stds = X_fit.std(axis=0, ddof=0)
    keep = np.isfinite(stds) & (stds > EPS)
    if int(keep.sum()) < 2:
        raise ValueError("Fewer than two variable geometry genes remain in training data.")
    fit_scaled = (X_fit[:, keep] - means[keep]) / stds[keep]
    apply_scaled = (X_apply[:, keep] - means[keep]) / stds[keep]
    if clip and clip > 0:
        fit_scaled = np.clip(fit_scaled, -clip, clip)
        apply_scaled = np.clip(apply_scaled, -clip, clip)
    n_components = min(int(n_pcs), fit_scaled.shape[0] - 1, fit_scaled.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Not enough training observations/genes for PCA.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_seed)
    pca.fit(fit_scaled)
    embedding = pca.transform(apply_scaled)
    qc = {
        "fit_mode": "training_only_transform",
        "n_fit_observations": int(X_fit.shape[0]),
        "n_apply_observations": int(X_apply.shape[0]),
        "n_scaled_genes": int(keep.sum()),
        "n_pcs": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio_first10": pca.explained_variance_ratio_[:10].tolist(),
    }
    return embedding, qc


def compute_transport_plan_qc(
    gamma: np.ndarray,
    row_normalized: np.ndarray,
    target_weights: np.ndarray,
    qc_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarize concentration/diffusion of the transport plan and apply gates."""
    row_entropy = -np.sum(
        row_normalized * np.log(np.maximum(row_normalized, EPS)), axis=1
    )
    max_entropy = math.log(max(row_normalized.shape[1], 2))
    normalized_row_entropy = row_entropy / max(max_entropy, EPS)
    effective_neighbors = np.exp(row_entropy)
    dominant_weight = np.max(row_normalized, axis=1)
    transported_target_mass = gamma.sum(axis=0)
    target_weights = np.asarray(target_weights, dtype=float)
    expected_positive = target_weights > EPS
    covered = transported_target_mass > (EPS * max(float(gamma.sum()), 1.0))
    target_coverage = float(
        np.sum(covered & expected_positive) / max(int(np.sum(expected_positive)), 1)
    )
    metrics = {
        "mean_effective_target_neighbors": float(np.mean(effective_neighbors)),
        "median_effective_target_neighbors": float(np.median(effective_neighbors)),
        "mean_dominant_target_weight": float(np.mean(dominant_weight)),
        "max_dominant_target_weight": float(np.max(dominant_weight)),
        "mean_normalized_row_entropy": float(np.mean(normalized_row_entropy)),
        "median_normalized_row_entropy": float(np.median(normalized_row_entropy)),
        "target_mass_coverage": target_coverage,
    }
    if not bool(qc_cfg.get("enabled", True)):
        metrics["passed"] = True
        metrics["failed_gates"] = []
        return metrics
    failed: List[str] = []
    if metrics["mean_effective_target_neighbors"] < float(
        qc_cfg.get("min_mean_effective_target_neighbors", 2.0)
    ):
        failed.append("TOO_CONCENTRATED_EFFECTIVE_NEIGHBORS")
    if metrics["mean_dominant_target_weight"] > float(
        qc_cfg.get("max_mean_dominant_target_weight", 0.95)
    ):
        failed.append("TOO_CONCENTRATED_DOMINANT_WEIGHT")
    if metrics["mean_normalized_row_entropy"] < float(
        qc_cfg.get("min_normalized_row_entropy", 0.05)
    ):
        failed.append("ROW_ENTROPY_TOO_LOW")
    if metrics["mean_normalized_row_entropy"] > float(
        qc_cfg.get("max_normalized_row_entropy", 0.98)
    ):
        failed.append("ROW_ENTROPY_TOO_HIGH")
    if metrics["target_mass_coverage"] < float(
        qc_cfg.get("min_target_mass_coverage", 0.90)
    ):
        failed.append("TARGET_MASS_COVERAGE_TOO_LOW")
    metrics["passed"] = not failed
    metrics["failed_gates"] = failed
    return metrics


def normalize_cost_matrix(cost: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    mode = str(mode).strip().lower()
    positive = cost[np.isfinite(cost) & (cost > 0)]
    if len(positive) == 0:
        raise ValueError("OT cost matrix has no positive finite entries.")
    if mode == "median_positive":
        scale = float(np.median(positive))
    elif mode == "max":
        scale = float(np.max(positive))
    elif mode == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unknown cost_normalization: {mode!r}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid OT cost scale: {scale}")
    return cost / scale, scale


def balanced_sinkhorn_log(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float,
    max_iter: int,
    stop_thr: float,
    check_every: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if reg <= 0:
        raise ValueError("OT regularization must be > 0.")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Balanced Sinkhorn requires strictly positive source/target weights.")
    log_a = np.log(a)
    log_b = np.log(b)
    log_kernel = -cost / reg
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)
    converged = False
    error = math.inf
    iteration = 0

    for iteration in range(1, max_iter + 1):
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        if iteration == 1 or iteration % check_every == 0 or iteration == max_iter:
            log_gamma = log_u[:, None] + log_kernel + log_v[None, :]
            gamma = np.exp(log_gamma)
            row_err = float(np.max(np.abs(gamma.sum(axis=1) - a)))
            col_err = float(np.max(np.abs(gamma.sum(axis=0) - b)))
            error = max(row_err, col_err)
            if np.isfinite(error) and error <= stop_thr:
                converged = True
                break

    log_gamma = log_u[:, None] + log_kernel + log_v[None, :]
    gamma = np.exp(log_gamma)
    log_info = {
        "solver": "internal_balanced_sinkhorn_log",
        "converged": bool(converged),
        "iterations": int(iteration),
        "marginal_max_abs_error": float(error),
        "transported_mass": float(gamma.sum()),
    }
    return gamma, log_info


def solve_ot(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    ot_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = str(ot_cfg.get("method", "balanced_sinkhorn_log")).strip().lower()
    reg = float(ot_cfg["regularization"])
    max_iter = int(ot_cfg["max_iterations"])
    stop_thr = float(ot_cfg["stop_threshold"])
    check_every = int(ot_cfg.get("check_every", 25))
    if method == "balanced_sinkhorn_log":
        return balanced_sinkhorn_log(a, b, cost, reg, max_iter, stop_thr, check_every)
    if method == "unbalanced_sinkhorn":
        try:
            import ot  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Unbalanced OT requires POT. Install with: pip install POT, "
                "or use ot.method='balanced_sinkhorn_log'."
            ) from exc
        reg_m = float(ot_cfg.get("unbalanced_marginal_regularization", 1.0))
        gamma, log = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            cost,
            reg,
            reg_m,
            method="sinkhorn_stabilized",
            numItermax=max_iter,
            stopThr=stop_thr,
            log=True,
        )
        gamma = np.asarray(gamma, dtype=float)
        err_values = log.get("err", []) if isinstance(log, Mapping) else []
        last_error = float(err_values[-1]) if len(err_values) else np.nan
        return gamma, {
            "solver": "POT_unbalanced_sinkhorn_stabilized",
            "converged": bool(np.isfinite(last_error) and last_error <= stop_thr),
            "iterations": int(len(err_values)),
            "marginal_max_abs_error": last_error,
            "transported_mass": float(gamma.sum()),
        }
    raise ValueError(f"Unknown OT method: {method!r}")


def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), EPS)
    return weights @ X


def weighted_variance(X: np.ndarray, weights: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), EPS)
    if mean is None:
        mean = weighted_mean(X, weights)
    return weights @ ((X - mean) ** 2)


def percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True)


def get_ppi_prior_percentile(driver_table: pd.DataFrame) -> pd.Series:
    for col in ["mean_driver_score_percentile", "primary_driver_score_percentile"]:
        if col in driver_table.columns:
            values = pd.to_numeric(driver_table[col], errors="coerce")
            if values.notna().any():
                return values.clip(0, 1).fillna(values.median())
    for col in ["primary_driver_score", "mean_driver_score", "gene_seed_evidence_score_first"]:
        if col in driver_table.columns:
            values = pd.to_numeric(driver_table[col], errors="coerce")
            if values.notna().any():
                return percentile_rank(values).fillna(0.5)
    if "ppi_driver_rank" in driver_table.columns:
        ranks = pd.to_numeric(driver_table["ppi_driver_rank"], errors="coerce")
        return (-ranks).rank(method="average", pct=True).fillna(0.5)
    return pd.Series(np.full(len(driver_table), 0.5), index=driver_table.index)


def bootstrap_subject_alignment_ci(
    subject_alignment: np.ndarray,
    n_iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-safe subject bootstrap CIs, evaluated only for nonempty genes."""
    subject_alignment = np.asarray(subject_alignment, dtype=float)
    n_subjects, n_genes = subject_alignment.shape
    lower = np.full(n_genes, np.nan, dtype=float)
    upper = np.full(n_genes, np.nan, dtype=float)
    if n_subjects == 0 or n_iterations <= 0:
        return lower, upper

    valid_genes = np.any(np.isfinite(subject_alignment), axis=0)
    if not np.any(valid_genes):
        return lower, upper
    values = subject_alignment[:, valid_genes]

    if n_subjects == 1:
        lower[valid_genes] = values[0]
        upper[valid_genes] = values[0]
        return lower, upper

    boot = np.full((n_iterations, values.shape[1]), np.nan, dtype=np.float32)
    for i in range(n_iterations):
        sampled = values[rng.integers(0, n_subjects, size=n_subjects)]
        counts = np.sum(np.isfinite(sampled), axis=0)
        sums = np.nansum(sampled, axis=0)
        boot[i] = np.divide(
            sums,
            counts,
            out=np.full(values.shape[1], np.nan, dtype=float),
            where=counts > 0,
        )

    alpha = (1.0 - confidence_level) / 2.0
    for j in range(boot.shape[1]):
        finite = boot[:, j][np.isfinite(boot[:, j])]
        if finite.size:
            lower[np.flatnonzero(valid_genes)[j]] = float(np.quantile(finite, alpha))
            upper[np.flatnonzero(valid_genes)[j]] = float(np.quantile(finite, 1.0 - alpha))
    return lower, upper


def bootstrap_two_arm_subject_effect_ci(
    source_subject_means: np.ndarray,
    target_subject_means: np.ndarray,
    direction_sign: np.ndarray,
    pooled_sd: np.ndarray,
    unresolved: np.ndarray,
    n_iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap disease-effect direction across source and target subjects."""
    source_subject_means = np.asarray(source_subject_means, dtype=float)
    target_subject_means = np.asarray(target_subject_means, dtype=float)
    n_source, n_genes = source_subject_means.shape
    n_target = target_subject_means.shape[0]
    lower = np.full(n_genes, np.nan, dtype=float)
    upper = np.full(n_genes, np.nan, dtype=float)
    if n_source == 0 or n_target == 0 or n_iterations <= 0:
        return lower, upper
    boot = np.full((n_iterations, n_genes), np.nan, dtype=np.float32)
    for i in range(int(n_iterations)):
        src = source_subject_means[rng.integers(0, n_source, size=n_source)].mean(axis=0)
        tgt = target_subject_means[rng.integers(0, n_target, size=n_target)].mean(axis=0)
        aligned = direction_sign * ((src - tgt) / pooled_sd)
        aligned[unresolved] = np.nan
        boot[i] = aligned
    alpha = (1.0 - confidence_level) / 2.0
    for j in range(n_genes):
        finite = boot[:, j][np.isfinite(boot[:, j])]
        if finite.size:
            lower[j] = float(np.quantile(finite, alpha))
            upper[j] = float(np.quantile(finite, 1.0 - alpha))
    return lower, upper


def weighted_column_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Column-wise weighted Pearson correlation without scipy loops."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"Correlation matrices must have the same shape: {X.shape} vs {Y.shape}")
    n = X.shape[0]
    if weights is None:
        w = np.full(n, 1.0 / max(n, 1), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have one value per observation")
        w = w / max(float(w.sum()), EPS)
    mx = w @ X
    my = w @ Y
    xc = X - mx
    yc = Y - my
    cov = w @ (xc * yc)
    vx = w @ (xc * xc)
    vy = w @ (yc * yc)
    denom = np.sqrt(np.maximum(vx * vy, 0.0))
    return np.divide(cov, denom, out=np.full(X.shape[1], np.nan), where=denom > EPS)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction preserving NaNs."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    if not np.any(valid):
        return adjusted
    p = np.clip(pvalues[valid], 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(q, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def permutation_local_pairing_test(
    X_source: np.ndarray,
    mapped_target: np.ndarray,
    weights: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test whether OT creates non-random local source-target pairing.

    The null permutes mapped target profiles across source cells. This preserves
    the mapped-target distribution and the global disease-control mean shift,
    while destroying the cell-level association supplied by the OT geometry.
    """
    observed = weighted_column_correlation(X_source, mapped_target, weights)
    pvalue = np.full(observed.shape, np.nan, dtype=float)
    valid = np.isfinite(observed)
    if n_iterations <= 0 or not np.any(valid):
        return observed, pvalue, benjamini_hochberg(pvalue)

    exceed = np.zeros(observed.shape, dtype=np.int64)
    n = X_source.shape[0]
    for _ in range(int(n_iterations)):
        permuted = mapped_target[rng.permutation(n)]
        null_corr = weighted_column_correlation(X_source, permuted, weights)
        comparable = valid & np.isfinite(null_corr)
        exceed[comparable] += null_corr[comparable] >= observed[comparable]
    pvalue[valid] = (exceed[valid] + 1.0) / (int(n_iterations) + 1.0)
    return observed, pvalue, benjamini_hochberg(pvalue)


def finite_positive_fraction(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    valid_n = np.sum(np.isfinite(matrix), axis=0)
    positive_n = np.sum(np.isfinite(matrix) & (matrix > 0), axis=0)
    return np.divide(
        positive_n,
        valid_n,
        out=np.full(matrix.shape[1], np.nan, dtype=float),
        where=valid_n > 0,
    )


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)



def finite_numeric_stat(values: Any, statistic: str) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(np.mean(arr))
    if statistic == "median":
        return float(np.median(arr))
    if statistic == "min":
        return float(np.min(arr))
    if statistic == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown statistic: {statistic}")

def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return safe.strip("_") or "model"


def summarize_unique_values(series: pd.Series, max_items: int = 20) -> List[str]:
    values = series.astype("string").fillna("<NA>").value_counts(dropna=False)
    return [f"{idx}: {int(count)}" for idx, count in values.head(max_items).items()]


def run_one_model(
    dataset: ExpressionDataset,
    drivers: DriverInputs,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    geometry_fit_dataset: Optional[ExpressionDataset] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    model_id = str(model["model_id"])
    model_dir = output_dir / "models" / sanitize_filename(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    columns = config["metadata_columns"]
    source_mask, target_mask = build_model_masks(dataset.obs, model, columns, config["matching"])
    source_all = np.flatnonzero(source_mask)
    target_all = np.flatnonzero(target_mask)
    subject_col = columns["subject"]
    cluster_col = columns.get("cluster")

    raw_source_subjects = dataset.obs.iloc[source_all][subject_col].astype("string").nunique(dropna=True)
    raw_target_subjects = dataset.obs.iloc[target_all][subject_col].astype("string").nunique(dropna=True)
    sampling_cfg = config["sampling"]
    min_obs = int(sampling_cfg["min_observations_per_arm"])
    min_subjects = int(sampling_cfg["min_subjects_per_arm"])

    base_qc: Dict[str, Any] = {
        "model_id": model_id,
        "status": "pending",
        "source_definition": {
            "disease": model.get("source_disease"),
            "celltype": model.get("source_celltype"),
            "stage": model.get("source_stage"),
        },
        "target_definition": {
            "disease": model.get("target_disease"),
            "celltype": model.get("target_celltype"),
            "stage": model.get("target_stage"),
        },
        "n_source_observations_raw": int(len(source_all)),
        "n_target_observations_raw": int(len(target_all)),
        "n_source_subjects_raw": int(raw_source_subjects),
        "n_target_subjects_raw": int(raw_target_subjects),
        "sampling_seed": int(sampling_cfg["random_seed"]),
    }

    if (
        len(source_all) < min_obs
        or len(target_all) < min_obs
        or raw_source_subjects < min_subjects
        or raw_target_subjects < min_subjects
    ):
        base_qc.update(
            {
                "status": "skipped_insufficient_data",
                "reason": (
                    f"Required >= {min_obs} observations and >= {min_subjects} subjects per arm; "
                    f"observed source={len(source_all)}/{raw_source_subjects}, "
                    f"target={len(target_all)}/{raw_target_subjects}."
                ),
            }
        )
        write_json(model_dir / "03_ot_model_qc.json", base_qc)
        return None, base_qc

    rng = np.random.default_rng(int(sampling_cfg["random_seed"]))
    source_idx = subject_balanced_sample(
        dataset.obs,
        source_all,
        int(sampling_cfg["max_source_observations"]),
        subject_col,
        cluster_col,
        bool(sampling_cfg.get("balance_subjects", True)),
        bool(sampling_cfg.get("stratify_within_subject_by_cluster", True)),
        rng,
    )
    target_idx = subject_balanced_sample(
        dataset.obs,
        target_all,
        int(sampling_cfg["max_target_observations"]),
        subject_col,
        cluster_col,
        bool(sampling_cfg.get("balance_subjects", True)),
        bool(sampling_cfg.get("stratify_within_subject_by_cluster", True)),
        rng,
    )
    combined_idx = np.concatenate([source_idx, target_idx])

    sampled = dataset.obs.iloc[combined_idx].copy()
    sampled.insert(0, "observation_id", sampled.index.astype(str))
    sampled.insert(1, "ot_arm", ["SOURCE_DISEASE"] * len(source_idx) + ["TARGET_CONTROL"] * len(target_idx))
    sampled.to_csv(model_dir / "01_sampled_observations.csv", index=False)

    expr_cfg = config["expression"]
    X_combined_raw = subset_rows_cols(dataset.X, combined_idx)
    X_combined = normalize_log1p_matrix(
        X_combined_raw,
        bool(expr_cfg.get("input_is_log1p", True)),
        float(expr_cfg.get("normalization_target_sum", 10000.0)),
    )

    driver_gene_set = drivers.supported_genes
    geometry_cfg = config["geometry"]
    geometry_fit_mode = "analysis_dataset"
    X_geometry_fit_source = X_combined
    geometry_var_names = dataset.var_names

    if geometry_fit_dataset is not None:
        if not np.array_equal(geometry_fit_dataset.var_names, dataset.var_names):
            raise ValueError("geometry_fit_dataset must have identical var_names/order.")
        fit_source_mask, fit_target_mask = build_model_masks(
            geometry_fit_dataset.obs, model, columns, config["matching"]
        )
        fit_source_all = np.flatnonzero(fit_source_mask)
        fit_target_all = np.flatnonzero(fit_target_mask)
        fit_source_subjects = geometry_fit_dataset.obs.iloc[fit_source_all][subject_col].astype("string").nunique(dropna=True)
        fit_target_subjects = geometry_fit_dataset.obs.iloc[fit_target_all][subject_col].astype("string").nunique(dropna=True)
        if (
            len(fit_source_all) < min_obs or len(fit_target_all) < min_obs
            or fit_source_subjects < min_subjects or fit_target_subjects < min_subjects
        ):
            raise ValueError("Insufficient training observations/subjects for leakage-controlled geometry fit.")
        fit_rng = np.random.default_rng(
            stable_seed(int(sampling_cfg["random_seed"]), model_id + "|geometry_fit")
        )
        fit_source_idx = subject_balanced_sample(
            geometry_fit_dataset.obs, fit_source_all,
            int(sampling_cfg["max_source_observations"]), subject_col, cluster_col,
            bool(sampling_cfg.get("balance_subjects", True)),
            bool(sampling_cfg.get("stratify_within_subject_by_cluster", True)), fit_rng,
        )
        fit_target_idx = subject_balanced_sample(
            geometry_fit_dataset.obs, fit_target_all,
            int(sampling_cfg["max_target_observations"]), subject_col, cluster_col,
            bool(sampling_cfg.get("balance_subjects", True)),
            bool(sampling_cfg.get("stratify_within_subject_by_cluster", True)), fit_rng,
        )
        fit_combined_idx = np.concatenate([fit_source_idx, fit_target_idx])
        X_fit_raw = subset_rows_cols(geometry_fit_dataset.X, fit_combined_idx)
        X_geometry_fit_source = normalize_log1p_matrix(
            X_fit_raw,
            bool(expr_cfg.get("input_is_log1p", True)),
            float(expr_cfg.get("normalization_target_sum", 10000.0)),
        )
        geometry_fit_mode = "disjoint_training_subjects"
        base_qc["geometry_training_n_source_observations"] = int(len(fit_source_idx))
        base_qc["geometry_training_n_target_observations"] = int(len(fit_target_idx))
        base_qc["geometry_training_n_source_subjects"] = int(fit_source_subjects)
        base_qc["geometry_training_n_target_subjects"] = int(fit_target_subjects)

    geometry_cols = select_geometry_genes(
        X_geometry_fit_source,
        geometry_var_names,
        driver_gene_set,
        int(geometry_cfg["n_variable_genes"]),
        bool(geometry_cfg.get("exclude_driver_genes", True)),
    )
    pd.DataFrame(
        {
            "gene_symbol": dataset.var_names[geometry_cols],
            "excluded_driver_gene": np.isin(dataset.var_names[geometry_cols], list(driver_gene_set)),
            "geometry_fit_mode": geometry_fit_mode,
        }
    ).to_csv(model_dir / "02_geometry_genes.csv", index=False)

    X_geometry_apply = as_dense(X_combined[:, geometry_cols])
    if geometry_fit_dataset is None:
        embedding, pca_qc = standardize_and_pca(
            X_geometry_apply,
            int(geometry_cfg["n_pcs"]),
            float(geometry_cfg.get("scale_clip", 10.0)),
            int(sampling_cfg["random_seed"]),
        )
        pca_qc["fit_mode"] = geometry_fit_mode
    else:
        X_geometry_fit = as_dense(X_geometry_fit_source[:, geometry_cols])
        embedding, pca_qc = fit_and_apply_standardize_pca(
            X_geometry_fit,
            X_geometry_apply,
            int(geometry_cfg["n_pcs"]),
            float(geometry_cfg.get("scale_clip", 10.0)),
            int(sampling_cfg["random_seed"]),
        )
    n_source = len(source_idx)
    Z_source = embedding[:n_source]
    Z_target = embedding[n_source:]

    cost_metric = str(config["ot"].get("cost_metric", "sqeuclidean"))
    cost_raw = pairwise_distances(Z_source, Z_target, metric=cost_metric)
    cost, cost_scale = normalize_cost_matrix(
        cost_raw, config["ot"].get("cost_normalization", "median_positive")
    )
    a = make_subject_balanced_weights(dataset.obs, source_idx, subject_col)
    b = make_subject_balanced_weights(dataset.obs, target_idx, subject_col)
    gamma, solver_qc = solve_ot(a, b, cost, config["ot"])
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 0) or gamma.sum() <= 0:
        raise ValueError(f"Model {model_id}: invalid OT transport plan.")

    row_mass = gamma.sum(axis=1)
    row_normalized = gamma / np.maximum(row_mass[:, None], EPS)
    transport_cost = float(np.sum(gamma * cost))
    plan_prob = gamma / max(gamma.sum(), EPS)
    plan_entropy = float(-np.sum(plan_prob * np.log(np.maximum(plan_prob, EPS))))
    plan_qc_cfg = config["ot"].get("transport_plan_qc", {})
    transport_plan_qc = compute_transport_plan_qc(gamma, row_normalized, b, plan_qc_cfg)

    var_lookup = {gene: idx for idx, gene in enumerate(dataset.var_names)}
    candidate_table = drivers.all_supported.copy()
    candidate_table["ppi_prior_percentile"] = get_ppi_prior_percentile(candidate_table)
    candidate_table = candidate_table[candidate_table["gene_symbol"].isin(var_lookup)].copy()
    missing_genes = sorted(drivers.supported_genes - set(candidate_table["gene_symbol"]))
    candidate_table = candidate_table.drop_duplicates("gene_symbol", keep="first")
    candidate_genes = candidate_table["gene_symbol"].to_numpy(dtype=str)
    candidate_cols = np.asarray([var_lookup[g] for g in candidate_genes], dtype=int)
    X_driver = as_dense(X_combined[:, candidate_cols])
    X_source = X_driver[:n_source]
    X_target = X_driver[n_source:]

    mapped_target = row_normalized @ X_target
    per_source_shift = mapped_target - X_source
    source_mean = weighted_mean(X_source, a)
    target_mean = weighted_mean(X_target, b)
    disease_effect = source_mean - target_mean
    pooled_var = 0.5 * (
        weighted_variance(X_source, a, source_mean)
        + weighted_variance(X_target, b, target_mean)
    )
    pooled_sd = np.sqrt(np.maximum(pooled_var, EPS))
    disease_effect_z = disease_effect / pooled_sd

    # This global OT shift is retained for diagnostics only. Under balanced OT,
    # its weighted mean is algebraically fixed by the marginals and equals
    # target_mean - source_mean; it must not be used as independent OT evidence.
    ot_global_shift = weighted_mean(per_source_shift, a)
    global_shift_identity_error = np.abs(ot_global_shift + disease_effect)

    min_effect = float(config["direction"]["min_abs_disease_effect_logexpr"])
    disease_direction = np.where(
        disease_effect > min_effect,
        "UP_IN_DISEASE",
        np.where(disease_effect < -min_effect, "DOWN_IN_DISEASE", "NEUTRAL_OR_UNRESOLVED"),
    )
    reversal_direction = np.where(
        disease_direction == "UP_IN_DISEASE",
        "DOWN",
        np.where(disease_direction == "DOWN_IN_DISEASE", "UP", "UNRESOLVED"),
    )
    direction_sign = np.sign(disease_effect)

    permutation_cfg = config.get("permutation", {})
    perm_rng = np.random.default_rng(
        stable_seed(int(permutation_cfg.get("random_seed", 789)), model_id + f"|{sampling_cfg['random_seed']}")
    )
    local_pairing_corr, local_pairing_p, local_pairing_fdr = permutation_local_pairing_test(
        X_source,
        mapped_target,
        a,
        int(permutation_cfg.get("n_iterations", 200)),
        perm_rng,
    )

    # Replication is checked on both arms. Source subjects are compared with the
    # weighted target mean, and target subjects with the weighted source mean.
    source_subjects = (
        dataset.obs.iloc[source_idx][subject_col]
        .astype("string").fillna("__MISSING_SOURCE_SUBJECT__").to_numpy()
    )
    target_subjects = (
        dataset.obs.iloc[target_idx][subject_col]
        .astype("string").fillna("__MISSING_TARGET_SUBJECT__").to_numpy()
    )
    unique_source_subjects = np.unique(source_subjects)
    unique_target_subjects = np.unique(target_subjects)
    source_subject_means = np.vstack([
        np.mean(X_source[source_subjects == subject], axis=0)
        for subject in unique_source_subjects
    ])
    target_subject_means = np.vstack([
        np.mean(X_target[target_subjects == subject], axis=0)
        for subject in unique_target_subjects
    ])
    unresolved = disease_direction == "NEUTRAL_OR_UNRESOLVED"
    source_direction_matrix = direction_sign * (
        (source_subject_means - target_mean[None, :]) / pooled_sd[None, :]
    )
    target_direction_matrix = direction_sign * (
        (source_mean[None, :] - target_subject_means) / pooled_sd[None, :]
    )
    source_direction_matrix[:, unresolved] = np.nan
    target_direction_matrix[:, unresolved] = np.nan
    source_subject_consistency = finite_positive_fraction(source_direction_matrix)
    target_subject_consistency = finite_positive_fraction(target_direction_matrix)
    two_arm_subject_consistency = np.fmin(
        source_subject_consistency, target_subject_consistency
    )

    subject_rows: List[pd.DataFrame] = []
    for pos, subject in enumerate(unique_source_subjects):
        subject_rows.append(pd.DataFrame({
            "model_id": model_id,
            "arm": "SOURCE_DISEASE",
            "subject": str(subject),
            "gene_symbol": candidate_genes,
            "subject_direction_alignment_z": source_direction_matrix[pos],
            "subject_direction_concordant": source_direction_matrix[pos] > 0,
        }))
    for pos, subject in enumerate(unique_target_subjects):
        subject_rows.append(pd.DataFrame({
            "model_id": model_id,
            "arm": "TARGET_CONTROL",
            "subject": str(subject),
            "gene_symbol": candidate_genes,
            "subject_direction_alignment_z": target_direction_matrix[pos],
            "subject_direction_concordant": target_direction_matrix[pos] > 0,
        }))
    pd.concat(subject_rows, ignore_index=True).to_csv(
        model_dir / "05_two_arm_subject_direction_scores.csv", index=False
    )

    bootstrap_cfg = config["bootstrap"]
    boot_rng = np.random.default_rng(
        stable_seed(int(bootstrap_cfg["random_seed"]), model_id + f"|{sampling_cfg['random_seed']}")
    )
    boot_lower, boot_upper = bootstrap_two_arm_subject_effect_ci(
        source_subject_means,
        target_subject_means,
        direction_sign,
        pooled_sd,
        unresolved,
        int(bootstrap_cfg["n_iterations"]),
        float(bootstrap_cfg["confidence_level"]),
        boot_rng,
    )

    results = candidate_table.copy().reset_index(drop=True)
    results.insert(0, "model_id", model_id)
    results["sampling_seed"] = int(sampling_cfg["random_seed"])
    results["gene_in_expression"] = True
    results["source_mean_logexpr"] = source_mean
    results["target_mean_logexpr"] = target_mean
    results["disease_effect_logexpr"] = disease_effect
    results["disease_effect_z"] = disease_effect_z
    results["disease_direction"] = disease_direction
    results["reversal_direction"] = reversal_direction
    results["ot_global_shift_toward_control_logexpr"] = ot_global_shift
    results["ot_global_shift_identity_error"] = global_shift_identity_error
    results["ot_global_shift_used_for_eligibility"] = False
    results["ot_local_pairing_corr"] = local_pairing_corr
    results["ot_local_pairing_permutation_pvalue"] = local_pairing_p
    results["ot_local_pairing_fdr"] = local_pairing_fdr
    results["transport_plan_qc_passed"] = bool(transport_plan_qc.get("passed", True))
    results["transport_mean_effective_target_neighbors"] = float(transport_plan_qc.get("mean_effective_target_neighbors", np.nan))
    results["transport_mean_dominant_target_weight"] = float(transport_plan_qc.get("mean_dominant_target_weight", np.nan))
    results["transport_mean_normalized_row_entropy"] = float(transport_plan_qc.get("mean_normalized_row_entropy", np.nan))
    results["transport_target_mass_coverage"] = float(transport_plan_qc.get("target_mass_coverage", np.nan))
    results["source_subject_direction_consistency"] = source_subject_consistency
    results["target_subject_direction_consistency"] = target_subject_consistency
    results["two_arm_subject_direction_consistency"] = two_arm_subject_consistency
    results["source_subject_direction_ci_lower"] = boot_lower
    results["source_subject_direction_ci_upper"] = boot_upper
    # Backward-compatible aliases, now explicitly based on subject disease effects.
    results["source_subject_alignment_ci_lower"] = boot_lower
    results["source_subject_alignment_ci_upper"] = boot_upper
    results["n_source_subjects_for_consistency"] = len(unique_source_subjects)
    results["n_target_subjects_for_consistency"] = len(unique_target_subjects)

    results["disease_effect_strength_percentile"] = percentile_rank(
        results["disease_effect_z"].abs()
    )
    results["ot_local_pairing_percentile"] = percentile_rank(
        results["ot_local_pairing_corr"]
    )

    strict_cfg = config["strict_filtering"]
    directional = results["disease_direction"] != "NEUTRAL_OR_UNRESOLVED"
    effect_strong = (
        results["disease_effect_z"].abs()
        >= float(strict_cfg["min_abs_disease_effect_z"])
    )
    pairing_strong = (
        results["ot_local_pairing_corr"]
        >= float(strict_cfg["min_abs_ot_local_pairing_corr"])
    )
    pairing_significant = (
        results["ot_local_pairing_fdr"]
        <= float(strict_cfg["max_ot_local_pairing_fdr"])
    )
    consistent = (
        results["two_arm_subject_direction_consistency"]
        >= float(strict_cfg["min_source_subject_direction_consistency"])
    )
    ci_ok = results["source_subject_direction_ci_lower"] > 0

    if not bool(strict_cfg.get("require_permutation_significance", True)):
        pairing_significant = pd.Series(True, index=results.index)
    if not bool(strict_cfg.get("require_positive_subject_bootstrap_ci", True)):
        ci_ok = pd.Series(True, index=results.index)

    plan_qc_ok = pd.Series(bool(transport_plan_qc.get("passed", True)), index=results.index)
    if not bool(plan_qc_cfg.get("require_for_single_run_eligibility", True)):
        plan_qc_ok = pd.Series(True, index=results.index)
    eligible = directional & effect_strong & pairing_strong & pairing_significant & consistent & ci_ok & plan_qc_ok
    results["ot_reversal_eligible_single_seed"] = eligible
    results["ot_reversal_eligible"] = eligible
    results["ot_reversal_ineligibility_reason"] = np.select(
        [
            ~directional,
            directional & ~effect_strong,
            directional & effect_strong & ~pairing_strong,
            directional & effect_strong & pairing_strong & ~pairing_significant,
            directional & effect_strong & pairing_strong & pairing_significant & ~consistent,
            directional & effect_strong & pairing_strong & pairing_significant & consistent & ~ci_ok,
            directional & effect_strong & pairing_strong & pairing_significant & consistent & ci_ok & ~plan_qc_ok,
        ],
        [
            "DISEASE_DIRECTION_UNRESOLVED",
            "DISEASE_EFFECT_TOO_SMALL",
            "OT_LOCAL_PAIRING_TOO_WEAK",
            "OT_LOCAL_PAIRING_NOT_SIGNIFICANT_AFTER_FDR",
            "LOW_TWO_ARM_SUBJECT_DIRECTION_CONSISTENCY",
            "TWO_ARM_SUBJECT_BOOTSTRAP_CI_NOT_POSITIVE",
            "TRANSPORT_PLAN_QC_FAILED",
        ],
        default="",
    )

    rank_cfg = config["ranking_v2"]
    weights = np.asarray(
        [
            float(rank_cfg["ppi_prior_weight"]),
            float(rank_cfg["disease_effect_weight"]),
            float(rank_cfg["ot_local_pairing_weight"]),
            float(rank_cfg["source_subject_consistency_weight"]),
        ]
    )
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"v3 ranking weights must sum to 1.0; observed {weights.sum():.6f}")
    composite = (
        weights[0] * results["ppi_prior_percentile"].astype(float)
        + weights[1] * results["disease_effect_strength_percentile"].fillna(0.0).astype(float)
        + weights[2] * results["ot_local_pairing_percentile"].fillna(0.0).astype(float)
        + weights[3] * results["two_arm_subject_direction_consistency"].fillna(0.0).astype(float)
    )
    results["ot_reversal_priority_score_single_seed"] = composite.where(eligible, np.nan)
    results["ot_reversal_priority_score"] = results["ot_reversal_priority_score_single_seed"]
    results["ot_reversal_rank_within_model"] = results["ot_reversal_priority_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")
    results["analysis_scope"] = "ACTIVE_PROGRESSION_STAGE3_TO_CONTROL"
    results["evidence_interpretation"] = "OT_INFORMED_PRIORITIZATION_NOT_CAUSAL_REVERSAL_PROOF"
    results["is_primary_tier_a_candidate"] = results["gene_symbol"].isin(drivers.tier_a_genes)
    results["is_supported_tier_ab_candidate"] = results["gene_symbol"].isin(drivers.supported_genes)

    results = results.sort_values(
        ["ot_reversal_eligible", "ot_reversal_priority_score", "ppi_prior_percentile"],
        ascending=[False, False, False],
        na_position="last",
    )
    results.to_csv(model_dir / "04_driver_reversal_scores.csv", index=False)
    primary = results[results["is_primary_tier_a_candidate"] & results["ot_reversal_eligible"]].copy()
    sensitivity = results[results["is_supported_tier_ab_candidate"] & results["ot_reversal_eligible"]].copy()
    primary.to_csv(model_dir / "06_primary_tier_a_shortlist.csv", index=False)
    sensitivity.to_csv(model_dir / "07_supported_tier_ab_shortlist.csv", index=False)

    if bool(config["ot"].get("save_transport_plan", False)):
        np.savez_compressed(
            model_dir / "transport_plan.npz",
            gamma=gamma,
            source_observation_ids=dataset.obs.index[source_idx].astype(str).to_numpy(),
            target_observation_ids=dataset.obs.index[target_idx].astype(str).to_numpy(),
        )

    base_qc.update(
        {
            "status": "ok" if solver_qc.get("converged") else "warning_ot_not_fully_converged",
            "n_source_observations_sampled": int(len(source_idx)),
            "n_target_observations_sampled": int(len(target_idx)),
            "n_source_subjects_sampled": int(dataset.obs.iloc[source_idx][subject_col].nunique(dropna=True)),
            "n_target_subjects_sampled": int(dataset.obs.iloc[target_idx][subject_col].nunique(dropna=True)),
            "subject_consistency_definition": "minimum of source-arm and target-arm direction consistency",
            "n_supported_driver_genes_input": int(len(drivers.supported_genes)),
            "n_supported_driver_genes_in_expression": int(len(candidate_genes)),
            "n_supported_driver_genes_missing_expression": int(len(missing_genes)),
            "missing_driver_genes": missing_genes[:100],
            "n_geometry_genes": int(len(geometry_cols)),
            "driver_genes_excluded_from_geometry": bool(geometry_cfg.get("exclude_driver_genes", True)),
            "pca": pca_qc,
            "cost_scale": float(cost_scale),
            "cost_min": float(np.min(cost)),
            "cost_median": float(np.median(cost)),
            "cost_max": float(np.max(cost)),
            "transport_cost_normalized": transport_cost,
            "transport_plan_entropy": plan_entropy,
            "transport_plan_qc": transport_plan_qc,
            "cost_metric": cost_metric,
            "geometry_fit_mode": geometry_fit_mode,
            "solver": solver_qc,
            "global_ot_shift_identity_max_abs_error": float(np.nanmax(global_shift_identity_error)),
            "global_ot_shift_used_for_eligibility": False,
            "permutation_iterations": int(permutation_cfg.get("n_iterations", 200)),
            "n_directional_genes": int(directional.sum()),
            "n_reversal_eligible_all_supported": int(results["ot_reversal_eligible"].sum()),
            "n_reversal_eligible_tier_a": int(primary.shape[0]),
            "n_reversal_eligible_tier_ab": int(sensitivity.shape[0]),
            "eligible_fraction_among_directional": float(
                results["ot_reversal_eligible"].sum() / max(int(directional.sum()), 1)
            ),
            "disease_direction_counts": results["disease_direction"].value_counts(dropna=False).to_dict(),
        }
    )
    write_json(model_dir / "03_ot_model_qc.json", base_qc)
    return results, base_qc


def aggregate_seed_stability(
    seed_results: Sequence[pd.DataFrame],
    min_rate: float,
    require_direction_agreement: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for frame in seed_results:
        cols = [
            "gene_symbol", "sampling_seed", "disease_direction",
            "ot_reversal_eligible_single_seed", "ot_reversal_priority_score_single_seed",
            "ot_local_pairing_corr", "ot_local_pairing_fdr", "disease_effect_z",
            "two_arm_subject_direction_consistency",
        ]
        frames.append(frame[cols].copy())
    combined = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for gene, group in combined.groupby("gene_symbol", sort=False):
        directional = group[group["disease_direction"] != "NEUTRAL_OR_UNRESOLVED"]
        direction_values = directional["disease_direction"].dropna().astype(str)
        direction_agreement = bool(direction_values.nunique() <= 1 and len(direction_values) > 0)
        direction_consensus = direction_values.mode().iloc[0] if len(direction_values) else "UNRESOLVED"
        eligible = group["ot_reversal_eligible_single_seed"].fillna(False).astype(bool)
        rate = float(eligible.mean()) if len(group) else 0.0
        robust = rate >= float(min_rate)
        if require_direction_agreement:
            robust = robust and direction_agreement
        rows.append(
            {
                "gene_symbol": gene,
                "n_sampling_seeds_evaluated": int(group["sampling_seed"].nunique()),
                "seed_eligibility_rate": rate,
                "seed_direction_agreement": direction_agreement,
                "seed_direction_consensus": direction_consensus,
                "seed_median_priority_score": finite_numeric_stat(
                    group["ot_reversal_priority_score_single_seed"], "median"
                ),
                "seed_median_ot_local_pairing_corr": finite_numeric_stat(
                    group["ot_local_pairing_corr"], "median"
                ),
                "seed_max_ot_local_pairing_fdr": finite_numeric_stat(
                    group["ot_local_pairing_fdr"], "max"
                ),
                "seed_median_abs_disease_effect_z": finite_numeric_stat(
                    pd.to_numeric(group["disease_effect_z"], errors="coerce").abs(), "median"
                ),
                "seed_min_subject_direction_consistency": finite_numeric_stat(
                    group["two_arm_subject_direction_consistency"], "min"
                ),
                "robust_across_sampling_seeds": bool(robust),
            }
        )
    return pd.DataFrame(rows)


def make_subject_split_replication_dataset(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[Optional[ExpressionDataset], Dict[str, Any]]:
    split_cfg = config["robustness"]["subject_split_replication"]
    columns = config["metadata_columns"]
    source_mask, target_mask = build_model_masks(dataset.obs, model, columns, config["matching"])
    subject_col = columns["subject"]
    source_subjects = sorted(
        dataset.obs.loc[source_mask, subject_col].astype("string").dropna().unique().tolist()
    )
    target_subjects = sorted(
        dataset.obs.loc[target_mask, subject_col].astype("string").dropna().unique().tolist()
    )
    min_subjects = int(split_cfg.get("min_subjects_per_arm", 3))
    fraction = float(split_cfg.get("validation_fraction", 0.30))
    meta = {
        "enabled": True,
        "validation_fraction": fraction,
        "n_source_subjects_available": len(source_subjects),
        "n_target_subjects_available": len(target_subjects),
    }
    if len(source_subjects) < min_subjects or len(target_subjects) < min_subjects:
        meta["status"] = "skipped_insufficient_subjects"
        return None, meta

    model_id = str(model.get("model_id", "model"))
    rng = np.random.default_rng(stable_seed(int(split_cfg.get("random_seed", 901)), model_id))
    n_source_val = min(len(source_subjects), max(min_subjects, int(round(len(source_subjects) * fraction))))
    n_target_val = min(len(target_subjects), max(min_subjects, int(round(len(target_subjects) * fraction))))
    source_val = sorted(rng.choice(np.asarray(source_subjects), size=n_source_val, replace=False).tolist())
    target_val = sorted(rng.choice(np.asarray(target_subjects), size=n_target_val, replace=False).tolist())

    subj = dataset.obs[subject_col].astype("string")
    selected = (source_mask & subj.isin(source_val).to_numpy()) | (
        target_mask & subj.isin(target_val).to_numpy()
    )
    rows = np.flatnonzero(selected)
    if rows.size == 0:
        meta["status"] = "skipped_no_observations"
        return None, meta
    X_sub = subset_rows_cols(dataset.X, rows)
    obs_sub = dataset.obs.iloc[rows].copy()
    meta.update(
        {
            "status": "ok",
            "source_validation_subjects": source_val,
            "target_validation_subjects": target_val,
            "n_validation_observations": int(len(rows)),
        }
    )
    return ExpressionDataset(
        X=X_sub,
        obs=obs_sub,
        var_names=dataset.var_names,
        source_description=dataset.source_description + "::subject_split_replication",
    ), meta



def subset_expression_dataset(
    dataset: ExpressionDataset,
    rows: np.ndarray,
    description_suffix: str,
) -> ExpressionDataset:
    rows = np.asarray(rows, dtype=int)
    return ExpressionDataset(
        X=subset_rows_cols(dataset.X, rows),
        obs=dataset.obs.iloc[rows].copy(),
        var_names=dataset.var_names,
        source_description=dataset.source_description + description_suffix,
    )


def dataset_subset_fingerprint(dataset: Optional[ExpressionDataset]) -> Optional[str]:
    if dataset is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(dataset.source_description).encode("utf-8"))
    digest.update(str(dataset.X.shape).encode("utf-8"))
    obs_hash = pd.util.hash_pandas_object(
        pd.Series(dataset.obs.index.astype(str), dtype="string"), index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(obs_hash.tobytes())
    digest.update(np.asarray(dataset.var_names, dtype=str).astype("U").tobytes())
    return digest.hexdigest()


def model_run_signature(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    geometry_fit_dataset: Optional[ExpressionDataset],
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "model": model,
        "config": config,
        "analysis_dataset_fingerprint": dataset_subset_fingerprint(dataset),
        "geometry_fit_dataset_fingerprint": dataset_subset_fingerprint(geometry_fit_dataset),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_cached_model_run(
    run_root: Path,
    model_id: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    model_dir = run_root / "models" / sanitize_filename(model_id)
    result_path = model_dir / "04_driver_reversal_scores.csv"
    qc_path = model_dir / "03_ot_model_qc.json"
    if not result_path.exists() or not qc_path.exists():
        return None, None
    try:
        result = pd.read_csv(result_path)
        with qc_path.open("r", encoding="utf-8") as handle:
            qc = json.load(handle)
        if str(qc.get("status", "")).startswith("failed"):
            return None, None
        return result, qc
    except Exception:
        return None, None


def run_model_with_resume(
    dataset: ExpressionDataset,
    drivers: DriverInputs,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    run_root: Path,
    resume: bool,
    geometry_fit_dataset: Optional[ExpressionDataset] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    model_id = str(model["model_id"])
    model_dir = run_root / "models" / sanitize_filename(model_id)
    signature_path = model_dir / "00_run_signature.json"
    signature = model_run_signature(dataset, model, config, geometry_fit_dataset)
    if resume and signature_path.exists():
        try:
            with signature_path.open("r", encoding="utf-8") as handle:
                cached_signature = json.load(handle).get("signature")
        except Exception:
            cached_signature = None
        if cached_signature == signature:
            cached_result, cached_qc = load_cached_model_run(run_root, model_id)
            if cached_result is not None and cached_qc is not None:
                cached_qc = dict(cached_qc)
                cached_qc["loaded_from_cache"] = True
                cached_qc["cache_signature"] = signature
                return cached_result, cached_qc
    result, qc = run_one_model(
        dataset,
        drivers,
        model,
        config,
        run_root,
        geometry_fit_dataset=geometry_fit_dataset,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(signature_path, {
        "signature": signature,
        "script_version": SCRIPT_VERSION,
        "model_id": model_id,
        "created_at": datetime.now().isoformat(),
    })
    qc = dict(qc)
    qc["loaded_from_cache"] = False
    qc["cache_signature"] = signature
    return result, qc


def model_subject_lists(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    columns = config["metadata_columns"]
    subject_col = columns["subject"]
    source_mask, target_mask = build_model_masks(
        dataset.obs, model, columns, config["matching"]
    )
    source_subjects = sorted(
        dataset.obs.loc[source_mask, subject_col]
        .astype("string").dropna().astype(str).unique().tolist()
    )
    target_subjects = sorted(
        dataset.obs.loc[target_mask, subject_col]
        .astype("string").dropna().astype(str).unique().tolist()
    )
    return source_subjects, target_subjects, source_mask, target_mask


def make_disjoint_subject_validation_split(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    repeat_index: int,
) -> Tuple[Optional[ExpressionDataset], Optional[ExpressionDataset], Dict[str, Any]]:
    cfg = config["robustness"]["repeated_subject_validation"]
    source_subjects, target_subjects, source_mask, target_mask = model_subject_lists(
        dataset, model, config
    )
    min_subjects = int(cfg.get("min_subjects_per_arm", 3))
    fraction = float(cfg.get("validation_fraction", 0.30))
    model_id = str(model.get("model_id", "model"))
    meta: Dict[str, Any] = {
        "repeat_index": int(repeat_index),
        "validation_fraction": fraction,
        "n_source_subjects_available": len(source_subjects),
        "n_target_subjects_available": len(target_subjects),
    }
    if len(source_subjects) < 2 * min_subjects or len(target_subjects) < 2 * min_subjects:
        meta["status"] = "skipped_insufficient_subjects_for_disjoint_train_validation"
        return None, None, meta

    def n_validation(total: int) -> int:
        proposed = max(min_subjects, int(round(total * fraction)))
        return min(proposed, total - min_subjects)

    n_source_val = n_validation(len(source_subjects))
    n_target_val = n_validation(len(target_subjects))
    rng = np.random.default_rng(
        stable_seed(int(cfg.get("random_seed", 1901)), f"{model_id}|repeat={repeat_index}")
    )
    stratify_columns = [str(x) for x in cfg.get("stratify_subject_columns", [])]
    missing_strata = [col for col in stratify_columns if col not in dataset.obs.columns]
    if missing_strata:
        raise KeyError(f"Subject-validation stratification columns are absent: {missing_strata}")
    subject_col = config["metadata_columns"]["subject"]

    def choose_validation_subjects(
        subjects_in_arm: List[str], arm_mask: np.ndarray, n_take: int
    ) -> List[str]:
        if not stratify_columns:
            return sorted(
                rng.choice(np.asarray(subjects_in_arm), size=n_take, replace=False).tolist()
            )
        arm_obs = dataset.obs.loc[arm_mask, [subject_col] + stratify_columns].copy()
        arm_obs[subject_col] = arm_obs[subject_col].astype("string").astype(str)
        subject_strata: Dict[str, Tuple[str, ...]] = {}
        for subject, subject_frame in arm_obs.groupby(subject_col, observed=False):
            values: List[str] = []
            for col in stratify_columns:
                mode = subject_frame[col].astype("string").fillna("<NA>").mode(dropna=False)
                values.append(str(mode.iloc[0]) if len(mode) else "<NA>")
            subject_strata[str(subject)] = tuple(values)
        groups: Dict[Tuple[str, ...], List[str]] = {}
        for subject in subjects_in_arm:
            groups.setdefault(subject_strata[str(subject)], []).append(str(subject))
        selected: List[str] = []
        remaining: List[str] = []
        for _, group_subjects in sorted(groups.items(), key=lambda x: str(x[0])):
            proposed = int(round(n_take * len(group_subjects) / len(subjects_in_arm)))
            proposed = min(proposed, len(group_subjects))
            if proposed > 0:
                chosen = rng.choice(
                    np.asarray(group_subjects), size=proposed, replace=False
                ).tolist()
                selected.extend(str(x) for x in chosen)
                remaining.extend(sorted(set(group_subjects) - set(chosen)))
            else:
                remaining.extend(group_subjects)
        if len(selected) < n_take:
            add = rng.choice(
                np.asarray(remaining), size=n_take - len(selected), replace=False
            ).tolist()
            selected.extend(str(x) for x in add)
        elif len(selected) > n_take:
            selected = rng.choice(
                np.asarray(selected), size=n_take, replace=False
            ).tolist()
        return sorted(selected)

    source_val = choose_validation_subjects(source_subjects, source_mask, n_source_val)
    target_val = choose_validation_subjects(target_subjects, target_mask, n_target_val)
    source_train = sorted(set(source_subjects) - set(source_val))
    target_train = sorted(set(target_subjects) - set(target_val))
    subject_values = dataset.obs[subject_col].astype("string").astype(str)

    validation_rows = np.flatnonzero(
        (source_mask & subject_values.isin(source_val).to_numpy())
        | (target_mask & subject_values.isin(target_val).to_numpy())
    )
    training_rows = np.flatnonzero(
        (source_mask & subject_values.isin(source_train).to_numpy())
        | (target_mask & subject_values.isin(target_train).to_numpy())
    )
    if len(validation_rows) == 0 or len(training_rows) == 0:
        meta["status"] = "skipped_empty_split"
        return None, None, meta
    meta.update({
        "status": "ok",
        "source_training_subjects": source_train,
        "target_training_subjects": target_train,
        "source_validation_subjects": source_val,
        "target_validation_subjects": target_val,
        "n_training_observations": int(len(training_rows)),
        "n_validation_observations": int(len(validation_rows)),
        "stratify_subject_columns": stratify_columns,
    })
    return (
        subset_expression_dataset(dataset, training_rows, f"::outer_train_{repeat_index}"),
        subset_expression_dataset(dataset, validation_rows, f"::outer_validation_{repeat_index}"),
        meta,
    )


def aggregate_repeated_subject_validation(
    run_results: Sequence[pd.DataFrame],
    base_result: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    base_direction = base_result.set_index("gene_symbol")["disease_direction"].astype(str).to_dict()
    frames: List[pd.DataFrame] = []
    for repeat_index, result in enumerate(run_results):
        frame = result[[
            "gene_symbol", "disease_direction", "ot_reversal_eligible_single_seed",
            "ot_reversal_priority_score_single_seed", "ot_reversal_rank_within_model",
            "disease_effect_z", "ot_local_pairing_corr", "ot_local_pairing_fdr",
            "two_arm_subject_direction_consistency",
        ]].copy()
        frame["validation_repeat"] = int(repeat_index)
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"gene_symbol": base_result["gene_symbol"]})
    combined = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for gene, group in combined.groupby("gene_symbol", sort=False):
        base_dir = base_direction.get(str(gene), "NEUTRAL_OR_UNRESOLVED")
        eligible = group["ot_reversal_eligible_single_seed"].fillna(False).astype(bool)
        directions = group["disease_direction"].fillna("NEUTRAL_OR_UNRESOLVED").astype(str)
        direction_matches = directions == base_dir
        directional = directions != "NEUTRAL_OR_UNRESOLVED"
        direction_agreement_rate = float(direction_matches.mean()) if len(group) else 0.0
        directional_agreement_rate = float(
            direction_matches[directional].mean()
        ) if directional.any() else 0.0
        eligibility_rate = float(eligible.mean()) if len(group) else 0.0
        completed_repeats = int(group["validation_repeat"].nunique())
        supported = (
            completed_repeats >= int(cfg.get("min_repeats_completed", 1))
            and eligibility_rate >= float(cfg.get("min_validation_eligibility_rate", 0.70))
            and direction_agreement_rate >= float(cfg.get("min_direction_agreement_rate", 0.90))
        )
        ranks = pd.to_numeric(group["ot_reversal_rank_within_model"], errors="coerce")
        rows.append({
            "gene_symbol": gene,
            "n_outer_validation_repeats_completed": completed_repeats,
            "outer_validation_eligibility_rate": eligibility_rate,
            "outer_validation_direction_agreement_rate": direction_agreement_rate,
            "outer_validation_directional_only_agreement_rate": directional_agreement_rate,
            "outer_validation_median_rank": finite_numeric_stat(ranks, "median"),
            "outer_validation_rank_q25": float(ranks.quantile(0.25)) if ranks.notna().any() else np.nan,
            "outer_validation_rank_q75": float(ranks.quantile(0.75)) if ranks.notna().any() else np.nan,
            "outer_validation_median_abs_disease_effect_z": finite_numeric_stat(
                pd.to_numeric(group["disease_effect_z"], errors="coerce").abs(), "median"
            ),
            "outer_validation_median_ot_local_pairing_corr": finite_numeric_stat(
                group["ot_local_pairing_corr"], "median"
            ),
            "outer_validation_max_ot_local_pairing_fdr": finite_numeric_stat(
                group["ot_local_pairing_fdr"], "max"
            ),
            "outer_validation_min_subject_consistency": finite_numeric_stat(
                group["two_arm_subject_direction_consistency"], "min"
            ),
            "repeated_subject_validation_supported": bool(supported),
        })
    return pd.DataFrame(rows)


def make_loso_dataset(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    omitted_subject: str,
    omitted_arm: str,
) -> ExpressionDataset:
    source_subjects, target_subjects, source_mask, target_mask = model_subject_lists(
        dataset, model, config
    )
    subject_col = config["metadata_columns"]["subject"]
    subjects = dataset.obs[subject_col].astype("string").astype(str)
    selected = source_mask | target_mask
    if omitted_arm == "SOURCE_DISEASE":
        selected &= ~(source_mask & (subjects == str(omitted_subject)).to_numpy())
    elif omitted_arm == "TARGET_CONTROL":
        selected &= ~(target_mask & (subjects == str(omitted_subject)).to_numpy())
    else:
        raise ValueError(f"Unknown omitted_arm: {omitted_arm}")
    rows = np.flatnonzero(selected)
    return subset_expression_dataset(
        dataset, rows, f"::loso_{omitted_arm}_{sanitize_filename(omitted_subject)}"
    )


def aggregate_loso_stability(
    run_results: Sequence[Tuple[str, str, pd.DataFrame]],
    base_result: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    base_direction = base_result.set_index("gene_symbol")["disease_direction"].astype(str).to_dict()
    frames: List[pd.DataFrame] = []
    for omitted_arm, omitted_subject, result in run_results:
        frame = result[[
            "gene_symbol", "disease_direction", "ot_reversal_eligible_single_seed",
            "ot_reversal_priority_score_single_seed", "disease_effect_z",
        ]].copy()
        frame["omitted_arm"] = omitted_arm
        frame["omitted_subject"] = omitted_subject
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"gene_symbol": base_result["gene_symbol"]})
    combined = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for gene, group in combined.groupby("gene_symbol", sort=False):
        base_dir = base_direction.get(str(gene), "NEUTRAL_OR_UNRESOLVED")
        eligible = group["ot_reversal_eligible_single_seed"].fillna(False).astype(bool)
        direction = group["disease_direction"].fillna("NEUTRAL_OR_UNRESOLVED").astype(str)
        flips = (direction != "NEUTRAL_OR_UNRESOLVED") & (direction != base_dir)
        eligibility_rate = float(eligible.mean()) if len(group) else 0.0
        no_flip = not bool(flips.any())
        expected_runs = int(cfg.get("_expected_runs", len(group)))
        completed_fraction = float(len(group) / max(expected_runs, 1))
        supported = (
            completed_fraction >= float(cfg.get("min_runs_completed_fraction", 0.90))
            and eligibility_rate >= float(cfg.get("min_eligibility_rate", 0.70))
        )
        if bool(cfg.get("require_no_direction_flip", True)):
            supported = supported and no_flip
        rows.append({
            "gene_symbol": gene,
            "n_loso_runs_completed": int(len(group)),
            "n_loso_runs_expected": expected_runs,
            "loso_runs_completed_fraction": completed_fraction,
            "loso_eligibility_rate": eligibility_rate,
            "loso_direction_flip_count": int(flips.sum()),
            "loso_no_direction_flip": bool(no_flip),
            "loso_median_priority_score": finite_numeric_stat(
                group["ot_reversal_priority_score_single_seed"], "median"
            ),
            "loso_min_abs_disease_effect_z": finite_numeric_stat(
                pd.to_numeric(group["disease_effect_z"], errors="coerce").abs(), "min"
            ),
            "loso_stability_supported": bool(supported),
        })
    return pd.DataFrame(rows)


def aggregate_sensitivity_grid(
    run_results: Sequence[Tuple[str, pd.DataFrame]],
    base_result: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    base_direction = base_result.set_index("gene_symbol")["disease_direction"].astype(str).to_dict()
    frames: List[pd.DataFrame] = []
    for setting_id, result in run_results:
        frame = result[[
            "gene_symbol", "disease_direction", "ot_reversal_eligible_single_seed",
            "ot_reversal_priority_score_single_seed", "ot_local_pairing_corr",
            "disease_effect_z", "transport_plan_qc_passed",
        ]].copy()
        frame["sensitivity_setting_id"] = setting_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"gene_symbol": base_result["gene_symbol"]})
    combined = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for gene, group in combined.groupby("gene_symbol", sort=False):
        base_dir = base_direction.get(str(gene), "NEUTRAL_OR_UNRESOLVED")
        eligible = group["ot_reversal_eligible_single_seed"].fillna(False).astype(bool)
        directions = group["disease_direction"].fillna("NEUTRAL_OR_UNRESOLVED").astype(str)
        direction_agreement = float((directions == base_dir).mean()) if len(group) else 0.0
        eligibility_rate = float(eligible.mean()) if len(group) else 0.0
        completed_settings = int(group["sensitivity_setting_id"].nunique())
        supported = (
            completed_settings >= int(cfg.get("min_settings_completed", 1))
            and eligibility_rate >= float(cfg.get("min_eligibility_rate", 0.60))
            and direction_agreement >= float(cfg.get("min_direction_agreement_rate", 0.90))
        )
        rows.append({
            "gene_symbol": gene,
            "n_sensitivity_settings_completed": completed_settings,
            "sensitivity_eligibility_rate": eligibility_rate,
            "sensitivity_direction_agreement_rate": direction_agreement,
            "sensitivity_plan_qc_pass_rate": float(
                group["transport_plan_qc_passed"].fillna(False).astype(bool).mean()
            ),
            "sensitivity_median_priority_score": finite_numeric_stat(
                group["ot_reversal_priority_score_single_seed"], "median"
            ),
            "sensitivity_median_ot_local_pairing_corr": finite_numeric_stat(
                group["ot_local_pairing_corr"], "median"
            ),
            "sensitivity_min_abs_disease_effect_z": finite_numeric_stat(
                pd.to_numeric(group["disease_effect_z"], errors="coerce").abs(), "min"
            ),
            "sensitivity_grid_supported": bool(supported),
        })
    return pd.DataFrame(rows)


def make_subject_label_permuted_dataset(
    dataset: ExpressionDataset,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    iteration: int,
) -> Tuple[ExpressionDataset, Dict[str, Any]]:
    cfg = config["robustness"]["pipeline_label_permutation"]
    source_subjects, target_subjects, source_mask, target_mask = model_subject_lists(
        dataset, model, config
    )
    overlap = set(source_subjects) & set(target_subjects)
    if overlap:
        raise ValueError(
            "Subject-label permutation requires disjoint source/target subjects; overlap: "
            + ", ".join(sorted(overlap)[:10])
        )
    all_subjects = source_subjects + target_subjects
    model_id = str(model.get("model_id", "model"))
    rng = np.random.default_rng(
        stable_seed(int(cfg.get("random_seed", 3901)), f"{model_id}|perm={iteration}")
    )
    subject_col = config["metadata_columns"]["subject"]
    subjects = dataset.obs[subject_col].astype("string").astype(str)
    union_mask = source_mask | target_mask
    rows = np.flatnonzero(union_mask)
    block_columns = [str(x) for x in cfg.get("block_columns", [])]
    missing_blocks = [col for col in block_columns if col not in dataset.obs.columns]
    if missing_blocks:
        raise KeyError(f"Permutation block columns are absent: {missing_blocks}")

    original_source = set(source_subjects)
    perm_source: set = set()
    block_summary: List[Dict[str, Any]] = []
    if not block_columns:
        n_source = len(source_subjects)
        perm_source = set(
            rng.choice(np.asarray(all_subjects), size=n_source, replace=False).tolist()
        )
        block_summary.append({
            "block": "ALL_SUBJECTS",
            "n_subjects": len(all_subjects),
            "n_source_preserved": n_source,
        })
    else:
        union_obs = dataset.obs.loc[union_mask, [subject_col] + block_columns].copy()
        union_obs[subject_col] = union_obs[subject_col].astype("string").astype(str)
        subject_blocks: Dict[str, Tuple[str, ...]] = {}
        for subject, subject_frame in union_obs.groupby(subject_col, observed=False):
            values: List[str] = []
            for col in block_columns:
                mode = subject_frame[col].astype("string").fillna("<NA>").mode(dropna=False)
                values.append(str(mode.iloc[0]) if len(mode) else "<NA>")
            subject_blocks[str(subject)] = tuple(values)
        block_to_subjects: Dict[Tuple[str, ...], List[str]] = {}
        for subject in all_subjects:
            block_to_subjects.setdefault(subject_blocks[str(subject)], []).append(str(subject))
        for block, block_subjects in sorted(block_to_subjects.items(), key=lambda x: str(x[0])):
            n_source_block = sum(subject in original_source for subject in block_subjects)
            if n_source_block <= 0:
                chosen = set()
            elif n_source_block >= len(block_subjects):
                chosen = set(block_subjects)
            else:
                chosen = set(
                    rng.choice(
                        np.asarray(block_subjects), size=n_source_block, replace=False
                    ).tolist()
                )
            perm_source.update(chosen)
            block_summary.append({
                "block": dict(zip(block_columns, block)),
                "n_subjects": len(block_subjects),
                "n_source_preserved": n_source_block,
                "n_exchangeable_subjects": (
                    len(block_subjects) if 0 < n_source_block < len(block_subjects) else 0
                ),
            })
    total_exchangeable = int(sum(x.get("n_exchangeable_subjects", x.get("n_subjects", 0)) for x in block_summary))
    if total_exchangeable <= 0:
        raise ValueError(
            "Subject-label permutation has no exchangeable subjects within configured blocks."
        )
    perm_target = set(all_subjects) - perm_source
    permuted = subset_expression_dataset(dataset, rows, f"::label_permutation_{iteration}")
    obs = permuted.obs.copy()
    perm_subjects = obs[subject_col].astype("string").astype(str)
    source_assignment = perm_subjects.isin(perm_source).to_numpy()
    columns = config["metadata_columns"]

    def first_value(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        if isinstance(value, (str, bytes)):
            return str(value)
        values = list(value)
        return str(values[0]) if values else fallback

    source_disease = first_value(model.get("source_disease"), "DISEASE")
    target_disease = first_value(model.get("target_disease"), "CONTROL")
    source_celltype = first_value(model.get("source_celltype"), "SOURCE_CELLTYPE")
    target_celltype = first_value(model.get("target_celltype"), source_celltype)
    source_stage = first_value(model.get("source_stage"), "SOURCE_STAGE")
    target_stage_value = model.get("target_stage")
    target_stage = None if target_stage_value is None else first_value(target_stage_value, "TARGET_STAGE")

    obs.loc[source_assignment, columns["disease"]] = source_disease
    obs.loc[~source_assignment, columns["disease"]] = target_disease
    obs.loc[source_assignment, columns["celltype"]] = source_celltype
    obs.loc[~source_assignment, columns["celltype"]] = target_celltype
    obs.loc[source_assignment, columns["stage"]] = source_stage
    if target_stage is not None:
        obs.loc[~source_assignment, columns["stage"]] = target_stage
    permuted.obs = obs
    meta = {
        "iteration": int(iteration),
        "n_source_subjects": len(perm_source),
        "n_target_subjects": len(perm_target),
        "permuted_source_subjects": sorted(perm_source),
        "permuted_target_subjects": sorted(perm_target),
        "block_columns": block_columns,
        "block_summary": block_summary,
    }
    return permuted, meta


def aggregate_pipeline_label_permutation(
    permutation_results: Sequence[pd.DataFrame],
    base_result: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    genes = base_result["gene_symbol"].astype(str).tolist()
    observed_score = pd.to_numeric(
        base_result.set_index("gene_symbol")["ot_reversal_priority_score_single_seed"],
        errors="coerce",
    ).reindex(genes)
    observed_eligible = base_result.set_index("gene_symbol")[
        "ot_reversal_eligible_single_seed"
    ].reindex(genes).fillna(False).astype(bool)
    null_scores = np.zeros((len(permutation_results), len(genes)), dtype=float)
    null_eligible = np.zeros((len(permutation_results), len(genes)), dtype=bool)
    null_shortlist_sizes: List[int] = []
    for i, result in enumerate(permutation_results):
        indexed = result.set_index("gene_symbol")
        scores = pd.to_numeric(
            indexed.get("ot_reversal_priority_score_single_seed"), errors="coerce"
        ).reindex(genes).fillna(0.0)
        eligibility = indexed.get("ot_reversal_eligible_single_seed").reindex(genes).fillna(False).astype(bool)
        null_scores[i] = scores.to_numpy(dtype=float)
        null_eligible[i] = eligibility.to_numpy(dtype=bool)
        null_shortlist_sizes.append(int(eligibility.sum()))
    pvalues = np.full(len(genes), np.nan, dtype=float)
    if len(permutation_results) > 0:
        obs = observed_score.fillna(0.0).to_numpy(dtype=float)
        for j in range(len(genes)):
            if observed_eligible.iloc[j]:
                pvalues[j] = (
                    1.0 + float(np.sum(null_scores[:, j] >= obs[j]))
                ) / (len(permutation_results) + 1.0)
    fdr = benjamini_hochberg(pvalues)
    enough_iterations = len(permutation_results) >= int(cfg.get("min_iterations_completed", 1))
    supported = (
        enough_iterations
        & np.isfinite(fdr)
        & (fdr <= float(cfg.get("max_pipeline_fdr", 0.10)))
    )
    table = pd.DataFrame({
        "gene_symbol": genes,
        "n_pipeline_label_permutations_completed": len(permutation_results),
        "pipeline_permutation_selection_frequency": (
            null_eligible.mean(axis=0) if len(permutation_results) else np.nan
        ),
        "pipeline_permutation_priority_pvalue": pvalues,
        "pipeline_permutation_priority_fdr": fdr,
        "pipeline_permutation_supported": supported,
    })
    observed_shortlist_size = int(observed_eligible.sum())
    shortlist_size_pvalue = (
        (1.0 + float(np.sum(np.asarray(null_shortlist_sizes) >= observed_shortlist_size)))
        / (len(null_shortlist_sizes) + 1.0)
        if null_shortlist_sizes else np.nan
    )
    qc = {
        "n_iterations_completed": len(permutation_results),
        "minimum_iterations_required": int(cfg.get("min_iterations_completed", 1)),
        "iterations_complete_enough": bool(enough_iterations),
        "null_shortlist_size_mean": float(np.mean(null_shortlist_sizes)) if null_shortlist_sizes else np.nan,
        "null_shortlist_size_median": float(np.median(null_shortlist_sizes)) if null_shortlist_sizes else np.nan,
        "null_shortlist_size_max": int(np.max(null_shortlist_sizes)) if null_shortlist_sizes else None,
        "null_shortlist_sizes": null_shortlist_sizes,
        "observed_single_seed_shortlist_size": observed_shortlist_size,
        "observed_shortlist_size_empirical_pvalue": shortlist_size_pvalue,
    }
    return table, qc


def rewrite_final_model_outputs(
    output_dir: Path,
    model_id: str,
    results: pd.DataFrame,
) -> None:
    model_dir = output_dir / "models" / sanitize_filename(model_id)
    results = results.sort_values(
        ["ot_reversal_eligible", "ot_reversal_priority_score", "ppi_prior_percentile"],
        ascending=[False, False, False],
        na_position="last",
    )
    results["ot_reversal_rank_within_model"] = results["ot_reversal_priority_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")
    results.to_csv(model_dir / "04_driver_reversal_scores.csv", index=False)
    results[
        results["is_primary_tier_a_candidate"] & results["ot_reversal_eligible"]
    ].to_csv(model_dir / "06_primary_tier_a_shortlist.csv", index=False)
    results[
        results["is_supported_tier_ab_candidate"] & results["ot_reversal_eligible"]
    ].to_csv(model_dir / "07_supported_tier_ab_shortlist.csv", index=False)

def build_consensus(
    all_results: pd.DataFrame,
    drivers: DriverInputs,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    consensus_cfg = config["consensus"]
    rows: List[Dict[str, Any]] = []
    for gene, group in all_results.groupby("gene_symbol", sort=False):
        eligible = group[group["ot_reversal_eligible"].fillna(False)]
        directions = eligible["disease_direction"].dropna().astype(str)
        n_up = int((directions == "UP_IN_DISEASE").sum())
        n_down = int((directions == "DOWN_IN_DISEASE").sum())
        direction_consistent = bool((n_up > 0) ^ (n_down > 0))
        if n_up > 0 and n_down == 0:
            consensus_direction = "UP_IN_DISEASE"
            consensus_reversal = "DOWN"
        elif n_down > 0 and n_up == 0:
            consensus_direction = "DOWN_IN_DISEASE"
            consensus_reversal = "UP"
        elif n_up > 0 and n_down > 0:
            consensus_direction = "MIXED_ACROSS_MODELS"
            consensus_reversal = "MODEL_SPECIFIC"
        else:
            consensus_direction = "UNRESOLVED"
            consensus_reversal = "UNRESOLVED"

        n_models = int(group["model_id"].nunique())
        n_eligible = int(eligible["model_id"].nunique())
        eligibility_rate = float(n_eligible / max(n_models, 1))
        all_ci_positive = bool(
            n_eligible > 0
            and (pd.to_numeric(eligible["source_subject_direction_ci_lower"], errors="coerce") > 0).all()
        )
        all_seed_stable = bool(
            n_eligible > 0
            and (
                eligible["robust_across_sampling_seeds"].fillna(False).all()
                if "robust_across_sampling_seeds" in eligible.columns else True
            )
        )
        all_split_supported = bool(
            n_eligible > 0
            and (
                eligible["subject_split_replication_supported"].fillna(False).all()
                if "subject_split_replication_supported" in eligible.columns else True
            )
        )
        all_repeated_validation_supported = bool(
            n_eligible > 0
            and (
                eligible["repeated_subject_validation_supported"].fillna(False).all()
                if "repeated_subject_validation_supported" in eligible.columns else True
            )
        )
        all_loso_supported = bool(
            n_eligible > 0
            and (
                eligible["loso_stability_supported"].fillna(False).all()
                if "loso_stability_supported" in eligible.columns else True
            )
        )
        all_sensitivity_supported = bool(
            n_eligible > 0
            and (
                eligible["sensitivity_grid_supported"].fillna(False).all()
                if "sensitivity_grid_supported" in eligible.columns else True
            )
        )
        all_pipeline_permutation_supported = bool(
            n_eligible > 0
            and (
                eligible["pipeline_permutation_supported"].fillna(False).all()
                if "pipeline_permutation_supported" in eligible.columns else True
            )
        )
        strict = (
            n_eligible >= int(consensus_cfg["min_models_eligible"])
            and eligibility_rate >= float(consensus_cfg["min_model_eligibility_rate"])
        )
        if bool(consensus_cfg.get("require_direction_consistency", True)):
            strict = strict and direction_consistent
        if bool(consensus_cfg.get("require_all_eligible_ci_positive", True)):
            strict = strict and all_ci_positive
        if bool(consensus_cfg.get("require_seed_stability", True)):
            strict = strict and all_seed_stable
        if bool(consensus_cfg.get("require_subject_split_replication", False)):
            strict = strict and all_split_supported
        if bool(consensus_cfg.get("require_repeated_subject_validation", True)):
            strict = strict and all_repeated_validation_supported
        if bool(consensus_cfg.get("require_loso_stability", True)):
            strict = strict and all_loso_supported
        if bool(consensus_cfg.get("require_sensitivity_stability", True)):
            strict = strict and all_sensitivity_supported
        if bool(consensus_cfg.get("require_pipeline_permutation_significance", True)):
            strict = strict and all_pipeline_permutation_supported

        best_row = group.sort_values(
            "ot_reversal_priority_score", ascending=False, na_position="last"
        ).iloc[0]
        mean_score = pd.to_numeric(
            eligible["ot_reversal_priority_score"], errors="coerce"
        ).mean()
        partial_r = int(consensus_cfg.get("min_models_eligible", 1))
        if "pipeline_permutation_priority_pvalue" in group.columns and n_models >= partial_r:
            model_pvalues = pd.to_numeric(
                group.drop_duplicates("model_id")["pipeline_permutation_priority_pvalue"],
                errors="coerce",
            ).fillna(1.0).clip(0.0, 1.0).sort_values().to_numpy(dtype=float)
            partial_conjunction_pvalue = float(
                min(1.0, (n_models - partial_r + 1) * model_pvalues[partial_r - 1])
            )
        else:
            partial_conjunction_pvalue = np.nan
        rows.append(
            {
                "gene_symbol": gene,
                "ppi_driver_tier": best_row.get("ppi_driver_tier", ""),
                "is_primary_tier_a_candidate": bool(gene in drivers.tier_a_genes),
                "n_models_evaluated": n_models,
                "n_models_reversal_eligible": n_eligible,
                "model_eligibility_rate": eligibility_rate,
                "eligible_models": ";".join(sorted(eligible["model_id"].astype(str).unique())),
                "n_up_in_disease_models": n_up,
                "n_down_in_disease_models": n_down,
                "direction_consistent_across_eligible_models": direction_consistent,
                "consensus_disease_direction": consensus_direction,
                "consensus_reversal_direction": consensus_reversal,
                "all_eligible_subject_cis_positive": all_ci_positive,
                "all_eligible_models_seed_stable": all_seed_stable,
                "all_eligible_models_subject_split_supported": all_split_supported,
                "all_eligible_models_repeated_subject_validation_supported": all_repeated_validation_supported,
                "all_eligible_models_loso_supported": all_loso_supported,
                "all_eligible_models_sensitivity_supported": all_sensitivity_supported,
                "all_eligible_models_pipeline_permutation_supported": all_pipeline_permutation_supported,
                "strict_consensus_eligible": bool(strict),
                "partial_conjunction_r": partial_r,
                "partial_conjunction_pvalue": partial_conjunction_pvalue,
                "mean_abs_disease_effect_z": pd.to_numeric(
                    eligible["disease_effect_z"], errors="coerce"
                ).abs().mean(),
                "mean_ot_local_pairing_corr": pd.to_numeric(
                    eligible["ot_local_pairing_corr"], errors="coerce"
                ).mean(),
                "max_ot_local_pairing_fdr": pd.to_numeric(
                    eligible["ot_local_pairing_fdr"], errors="coerce"
                ).max(),
                "mean_two_arm_subject_consistency": pd.to_numeric(
                    eligible["two_arm_subject_direction_consistency"], errors="coerce"
                ).mean(),
                "mean_outer_validation_eligibility_rate": (
                    pd.to_numeric(eligible["outer_validation_eligibility_rate"], errors="coerce").mean()
                    if "outer_validation_eligibility_rate" in eligible.columns else np.nan
                ),
                "mean_loso_eligibility_rate": (
                    pd.to_numeric(eligible["loso_eligibility_rate"], errors="coerce").mean()
                    if "loso_eligibility_rate" in eligible.columns else np.nan
                ),
                "mean_sensitivity_eligibility_rate": (
                    pd.to_numeric(eligible["sensitivity_eligibility_rate"], errors="coerce").mean()
                    if "sensitivity_eligibility_rate" in eligible.columns else np.nan
                ),
                "max_pipeline_permutation_fdr": (
                    pd.to_numeric(eligible["pipeline_permutation_priority_fdr"], errors="coerce").max()
                    if "pipeline_permutation_priority_fdr" in eligible.columns else np.nan
                ),
                "mean_ot_reversal_priority_score": mean_score,
                "best_ot_reversal_priority_score": pd.to_numeric(
                    eligible["ot_reversal_priority_score"], errors="coerce"
                ).max(),
                "best_model": best_row.get("model_id", ""),
                "support_scope": (
                    "STRICT_CROSS_MODEL" if strict else
                    "CROSS_MODEL_MIXED_DIRECTION" if n_eligible >= 2 and not direction_consistent else
                    "CROSS_MODEL_NOT_STRICT" if n_eligible >= 2 else
                    "MODEL_SPECIFIC" if n_eligible == 1 else
                    "NO_ELIGIBLE_MODEL"
                ),
            }
        )
    consensus = pd.DataFrame(rows)
    consensus["partial_conjunction_fdr"] = benjamini_hochberg(
        pd.to_numeric(consensus["partial_conjunction_pvalue"], errors="coerce").to_numpy(dtype=float)
    )
    if bool(consensus_cfg.get("require_partial_conjunction_significance", True)):
        consensus["strict_consensus_eligible"] = (
            consensus["strict_consensus_eligible"].fillna(False)
            & (
                pd.to_numeric(consensus["partial_conjunction_fdr"], errors="coerce")
                <= float(consensus_cfg.get("max_partial_conjunction_fdr", 0.10))
            )
        )
    consensus["consensus_priority_score"] = (
        consensus["mean_ot_reversal_priority_score"].fillna(0.0)
        * np.sqrt(consensus["model_eligibility_rate"].fillna(0.0))
    ).where(consensus["strict_consensus_eligible"], np.nan)
    consensus["consensus_rank"] = consensus["consensus_priority_score"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype("Int64")
    consensus = consensus.sort_values(
        ["strict_consensus_eligible", "n_models_reversal_eligible", "consensus_priority_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    primary = consensus[consensus["is_primary_tier_a_candidate"]].copy()
    sensitivity = consensus.copy()
    return primary, sensitivity

def validate_v3_configuration(config: Mapping[str, Any]) -> None:
    """Fail early on internally inconsistent robustness settings."""
    errors: List[str] = []

    def rate(name: str, value: Any) -> None:
        try:
            x = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric")
            return
        if not 0.0 <= x <= 1.0:
            errors.append(f"{name} must be between 0 and 1; observed {x}")

    policy = str(config.get("existing_output_policy", "resume")).lower()
    if policy not in {"error", "timestamp", "overwrite", "resume"}:
        errors.append(f"existing_output_policy is invalid: {policy}")
    ot_cfg = config.get("ot", {})
    if str(ot_cfg.get("cost_metric", "sqeuclidean")) not in {
        "sqeuclidean", "euclidean", "cosine", "correlation"
    }:
        errors.append("ot.cost_metric must be sqeuclidean, euclidean, cosine, or correlation")
    if float(ot_cfg.get("regularization", 0.0)) <= 0:
        errors.append("ot.regularization must be > 0")
    plan_cfg = ot_cfg.get("transport_plan_qc", {})
    rate("ot.transport_plan_qc.min_normalized_row_entropy", plan_cfg.get("min_normalized_row_entropy", 0.0))
    rate("ot.transport_plan_qc.max_normalized_row_entropy", plan_cfg.get("max_normalized_row_entropy", 1.0))
    rate("ot.transport_plan_qc.max_mean_dominant_target_weight", plan_cfg.get("max_mean_dominant_target_weight", 1.0))
    rate("ot.transport_plan_qc.min_target_mass_coverage", plan_cfg.get("min_target_mass_coverage", 0.0))
    if float(plan_cfg.get("min_normalized_row_entropy", 0.0)) > float(
        plan_cfg.get("max_normalized_row_entropy", 1.0)
    ):
        errors.append("transport-plan minimum entropy exceeds maximum entropy")

    robustness = config.get("robustness", {})
    rate("robustness.min_seed_eligibility_rate", robustness.get("min_seed_eligibility_rate", 0.0))
    repeated = robustness.get("repeated_subject_validation", {})
    n_repeats = int(repeated.get("n_repeats", 0))
    min_repeats = int(repeated.get("min_repeats_completed", 0))
    if bool(repeated.get("enabled", False)) and n_repeats <= 0:
        errors.append("repeated_subject_validation.n_repeats must be > 0 when enabled")
    if min_repeats > n_repeats:
        errors.append("repeated_subject_validation.min_repeats_completed exceeds n_repeats")
    rate("repeated_subject_validation.validation_fraction", repeated.get("validation_fraction", 0.0))
    rate("repeated_subject_validation.min_validation_eligibility_rate", repeated.get("min_validation_eligibility_rate", 0.0))
    rate("repeated_subject_validation.min_direction_agreement_rate", repeated.get("min_direction_agreement_rate", 0.0))

    loso = robustness.get("leave_one_subject_out", {})
    rate("leave_one_subject_out.min_runs_completed_fraction", loso.get("min_runs_completed_fraction", 0.0))
    rate("leave_one_subject_out.min_eligibility_rate", loso.get("min_eligibility_rate", 0.0))

    sensitivity = robustness.get("sensitivity_grid", {})
    settings = list(sensitivity.get("settings", []))
    min_settings = int(sensitivity.get("min_settings_completed", 0))
    if bool(sensitivity.get("enabled", False)) and not settings:
        errors.append("sensitivity_grid.settings is empty while sensitivity grid is enabled")
    if min_settings > len(settings):
        errors.append("sensitivity_grid.min_settings_completed exceeds number of settings")
    setting_ids = [str(x.get("setting_id", "")) for x in settings]
    if len(setting_ids) != len(set(setting_ids)):
        errors.append("sensitivity_grid setting_id values must be unique")
    rate("sensitivity_grid.min_eligibility_rate", sensitivity.get("min_eligibility_rate", 0.0))
    rate("sensitivity_grid.min_direction_agreement_rate", sensitivity.get("min_direction_agreement_rate", 0.0))

    pipeline_perm = robustness.get("pipeline_label_permutation", {})
    block_columns = [str(x) for x in pipeline_perm.get("block_columns", [])]
    metadata_columns = config.get("metadata_columns", {})
    forbidden_blocks = {
        str(metadata_columns.get("disease", "")),
        str(metadata_columns.get("stage", "")),
    } & set(block_columns)
    if forbidden_blocks:
        errors.append(
            "pipeline_label_permutation.block_columns must not contain disease/stage defining columns: "
            + ", ".join(sorted(forbidden_blocks))
        )
    n_perm = int(pipeline_perm.get("n_iterations", 0))
    min_perm = int(pipeline_perm.get("min_iterations_completed", 0))
    if bool(pipeline_perm.get("enabled", False)) and n_perm <= 0:
        errors.append("pipeline_label_permutation.n_iterations must be > 0 when enabled")
    if min_perm > n_perm:
        errors.append("pipeline_label_permutation.min_iterations_completed exceeds n_iterations")
    rate("pipeline_label_permutation.max_pipeline_fdr", pipeline_perm.get("max_pipeline_fdr", 1.0))

    consensus = config.get("consensus", {})
    rate("consensus.min_model_eligibility_rate", consensus.get("min_model_eligibility_rate", 0.0))
    rate("consensus.max_partial_conjunction_fdr", consensus.get("max_partial_conjunction_fdr", 1.0))
    if int(consensus.get("min_models_eligible", 1)) > len(config.get("models", [])):
        errors.append("consensus.min_models_eligible exceeds number of configured models")

    if errors:
        raise ValueError("Invalid v3 configuration:\n- " + "\n- ".join(errors))


def write_methods_file(path: Path, config: Mapping[str, Any]) -> None:
    strict = config["strict_filtering"]
    robustness = config.get("robustness", {})
    repeated = robustness.get("repeated_subject_validation", {})
    loso = robustness.get("leave_one_subject_out", {})
    sensitivity = robustness.get("sensitivity_grid", {})
    pipeline_perm = robustness.get("pipeline_label_permutation", {})
    text = f"""OT disease-reversal methods and score definition (v3 robust)
================================================================

Analysis scope
--------------
This workflow evaluates active-progression Stage 3 disease states against matched
Control states. It is an OT-informed prioritization analysis, not proof of causal
intervention or terminal-state reversal.

Core OT correction
------------------
Under balanced OT, the weighted global mapped-target mean is fixed by the target
marginal. The global mean OT shift is therefore algebraically redundant with the
disease effect and is retained only as a numerical identity check.

Single-run evidence
-------------------
Disease direction is estimated with equal subject mass. OT-specific evidence is the
weighted cell-level correlation between source expression and the OT-mapped control
barycenter. Its null permutes mapped control profiles across source cells, preserving
the global source-control shift while destroying local OT correspondence. Candidate
p-values are adjusted by Benjamini-Hochberg FDR.

A single run requires:
1. |disease_effect_logexpr| >= {config['direction']['min_abs_disease_effect_logexpr']};
2. |disease_effect_z| >= {strict['min_abs_disease_effect_z']};
3. OT local-pairing correlation >= {strict['min_abs_ot_local_pairing_corr']};
4. OT local-pairing FDR <= {strict['max_ot_local_pairing_fdr']};
5. two-arm subject direction consistency >= {strict['min_source_subject_direction_consistency']};
6. a positive two-arm subject-bootstrap lower CI when enabled;
7. transport-plan concentration/diffusion QC when enabled.

Leakage-controlled repeated subject validation
----------------------------------------------
Each repeat uses disjoint training and validation subjects. Geometry genes, scaling
parameters, and PCA are fitted only on training subjects and then applied to validation
subjects. The default requests {repeated.get('n_repeats', 0)} repeats and requires at
least {repeated.get('min_repeats_completed', 0)} completed repeats, validation
eligibility rate >= {repeated.get('min_validation_eligibility_rate', 0)}, and direction
agreement rate >= {repeated.get('min_direction_agreement_rate', 0)}.

Additional robustness gates
---------------------------
- Multi-seed sampling stability with direction agreement.
- Leave-one-subject-out analysis on both disease and control arms; minimum eligibility
  rate {loso.get('min_eligibility_rate', 0)} and no direction flip when configured.
- OT/geometry sensitivity grid across regularization, distance metric, PCA dimension,
  and geometry-gene count; minimum setting eligibility rate
  {sensitivity.get('min_eligibility_rate', 0)}.
- Whole-pipeline subject-label permutations. Each null iteration repeats geometry
  selection, PCA, OT, local permutation testing, FDR, and strict filtering. The default
  requests {pipeline_perm.get('n_iterations', 0)} iterations and requires per-gene
  pipeline FDR <= {pipeline_perm.get('max_pipeline_fdr', 1)}.
- Resumable robustness subruns when existing_output_policy='resume'.

Priority and cross-model inference
----------------------------------
The score combines PPI prior, disease-effect strength, OT local-pairing strength, and
two-arm subject consistency. Final cross-model consensus additionally requires all
enabled robustness gates and a Bonferroni partial-conjunction test for evidence in at
least the configured minimum number of models. Mixed-direction genes remain
model-specific and are not assigned a universal intervention direction.
"""
    path.write_text(text, encoding="utf-8")

def run_pipeline(config: Mapping[str, Any], config_path: Path) -> Path:
    started = time.time()
    config_dir = config_path.parent.resolve()
    output_root = resolve_path(config["output_root"], config_dir, required=False)
    assert output_root is not None
    output_dir = choose_output_dir(
        output_root,
        str(config["analysis_id"]),
        str(config.get("existing_output_policy", "resume")),
    )

    tier_a_path = resolve_path(config["tier_a_driver_path"], config_dir, required=True)
    supported_path = resolve_path(config["supported_driver_path"], config_dir, required=True)
    assert tier_a_path is not None and supported_path is not None
    drivers = load_driver_inputs(tier_a_path, supported_path)
    dataset, input_manifest = load_expression_dataset(config, config_dir)
    validate_metadata(dataset.obs, config["metadata_columns"])
    if isinstance(config, MutableMapping):
        config["_runtime_input_fingerprint"] = hashlib.sha256(
            json.dumps(input_manifest, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest()

    manifest: Dict[str, Any] = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "run_started": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path.resolve()),
        "config": config,
        "tier_a_driver_path": str(tier_a_path),
        "tier_a_driver_sha256": sha256_file(tier_a_path),
        "supported_driver_path": str(supported_path),
        "supported_driver_sha256": sha256_file(supported_path),
        "n_tier_a_driver_genes": len(drivers.tier_a_genes),
        "n_tier_ab_supported_genes": len(drivers.supported_genes),
        "expression_input": input_manifest,
        "v3_key_change": (
            "Adds leakage-controlled repeated subject validation, LOSO influence analysis, "
            "OT/geometry sensitivity, transport-plan QC, whole-pipeline subject-label nulls, "
            "and partial-conjunction cross-model inference."
        ),
    }
    write_json(output_dir / "00_input_manifest.json", manifest)

    driver_qc = drivers.all_supported[[
        col for col in [
            "gene_symbol", "ppi_driver_tier", "is_tier_a_primary", "is_tier_ab_supported",
            "primary_driver_score", "mean_driver_score", "mean_driver_score_percentile",
            "ppi_driver_rank",
        ] if col in drivers.all_supported.columns
    ]].copy()
    driver_qc["present_in_expression"] = driver_qc["gene_symbol"].isin(set(dataset.var_names))
    driver_qc["looks_like_ensembl_id"] = driver_qc["gene_symbol"].astype(str).str.match(
        r"^ENSG[0-9]+(?:\.[0-9]+)?$", case=False, na=False
    )
    driver_qc.to_csv(output_dir / "01_driver_input_qc.csv", index=False)

    model_results: List[pd.DataFrame] = []
    model_qc_rows: List[Dict[str, Any]] = []
    failed_models: List[Dict[str, Any]] = []
    robustness_cfg = config.get("robustness", {})
    robustness_enabled = bool(robustness_cfg.get("enabled", False))
    resume_runs = bool(robustness_cfg.get("resume_completed_runs", True))
    configured_seeds = [int(x) for x in robustness_cfg.get("sampling_seeds", [])]
    primary_seed = int(config["sampling"]["random_seed"])
    seeds = list(dict.fromkeys([primary_seed] + configured_seeds)) if robustness_enabled else [primary_seed]

    for model in config["models"]:
        model_id = str(model.get("model_id", "unnamed_model"))
        print(f"[MODEL] {model_id}", flush=True)
        try:
            primary_config = copy.deepcopy(config)
            primary_config["sampling"]["random_seed"] = primary_seed
            primary_result, primary_qc = run_one_model(
                dataset, drivers, model, primary_config, output_dir
            )
            if primary_result is None:
                model_qc_rows.append(primary_qc)
                continue

            seed_results: List[pd.DataFrame] = [primary_result.copy()]
            for seed in seeds:
                if seed == primary_seed:
                    continue
                print(f"  [SEED REPEAT] {model_id}: {seed}", flush=True)
                repeat_config = copy.deepcopy(config)
                repeat_config["sampling"]["random_seed"] = int(seed)
                repeat_config["permutation"]["n_iterations"] = int(
                    robustness_cfg.get("permutation_iterations_per_repeat", 100)
                )
                repeat_root = output_dir / "robustness" / f"seed_{seed}"
                repeat_result, _ = run_model_with_resume(
                    dataset, drivers, model, repeat_config, repeat_root, resume_runs
                )
                if repeat_result is not None:
                    seed_results.append(repeat_result.copy())

            stability = aggregate_seed_stability(
                seed_results,
                float(robustness_cfg.get("min_seed_eligibility_rate", 0.66)),
                bool(robustness_cfg.get("require_direction_agreement", True)),
            )
            model_dir = output_dir / "models" / sanitize_filename(model_id)
            stability.to_csv(model_dir / "08_sampling_seed_stability.csv", index=False)
            final_result = primary_result.merge(stability, on="gene_symbol", how="left")
            final_result["ot_reversal_eligible_primary_seed"] = final_result[
                "ot_reversal_eligible_single_seed"
            ]
            final_result["ot_reversal_eligible"] = final_result[
                "robust_across_sampling_seeds"
            ].fillna(False)
            final_result["ot_reversal_priority_score"] = final_result[
                "seed_median_priority_score"
            ].where(final_result["ot_reversal_eligible"], np.nan)

            split_cfg = robustness_cfg.get("subject_split_replication", {})
            split_meta: Dict[str, Any] = {"enabled": False, "status": "disabled"}
            if robustness_enabled and bool(split_cfg.get("enabled", False)):
                print(f"  [SUBJECT SPLIT] {model_id}", flush=True)
                split_dataset, split_meta = make_subject_split_replication_dataset(
                    dataset, model, config
                )
                if split_dataset is not None:
                    split_config = copy.deepcopy(config)
                    split_config["sampling"]["random_seed"] = stable_seed(
                        int(split_cfg.get("random_seed", 901)), model_id + "|sampling"
                    )
                    split_config["sampling"]["min_subjects_per_arm"] = int(
                        split_cfg.get("min_subjects_per_arm", 3)
                    )
                    split_config["permutation"]["n_iterations"] = int(
                        robustness_cfg.get("permutation_iterations_per_repeat", 100)
                    )
                    split_root = output_dir / "robustness" / "subject_split_replication"
                    split_result, split_qc = run_model_with_resume(
                        split_dataset, drivers, model, split_config, split_root, resume_runs
                    )
                    split_meta["model_qc"] = split_qc
                    if split_result is not None:
                        split_cols = split_result[[
                            "gene_symbol", "disease_direction", "ot_reversal_eligible_single_seed",
                            "ot_reversal_priority_score_single_seed", "ot_local_pairing_corr",
                            "ot_local_pairing_fdr", "two_arm_subject_direction_consistency",
                        ]].copy()
                        split_cols = split_cols.rename(columns={
                            "disease_direction": "subject_split_disease_direction",
                            "ot_reversal_eligible_single_seed": "subject_split_eligible",
                            "ot_reversal_priority_score_single_seed": "subject_split_priority_score",
                            "ot_local_pairing_corr": "subject_split_ot_local_pairing_corr",
                            "ot_local_pairing_fdr": "subject_split_ot_local_pairing_fdr",
                            "two_arm_subject_direction_consistency": "subject_split_subject_consistency",
                        })
                        final_result = final_result.merge(split_cols, on="gene_symbol", how="left")
                        final_result["subject_split_direction_matches"] = (
                            final_result["subject_split_disease_direction"]
                            == final_result["seed_direction_consensus"]
                        )
                        final_result["subject_split_replication_supported"] = (
                            final_result["subject_split_eligible"].fillna(False)
                            & final_result["subject_split_direction_matches"].fillna(False)
                        )
                    else:
                        final_result["subject_split_replication_supported"] = False
                else:
                    final_result["subject_split_replication_supported"] = False
            else:
                final_result["subject_split_replication_supported"] = True

            if bool(split_cfg.get("require_for_final_eligibility", True)):
                final_result["ot_reversal_eligible"] = (
                    final_result["ot_reversal_eligible"].fillna(False)
                    & final_result["subject_split_replication_supported"].fillna(False)
                )
            # Repeated disjoint subject validation. Geometry selection, scaling, and PCA
            # are fitted only on training subjects and applied to validation subjects.
            repeated_cfg = robustness_cfg.get("repeated_subject_validation", {})
            repeated_meta: Dict[str, Any] = {"enabled": False, "status": "disabled"}
            if robustness_enabled and bool(repeated_cfg.get("enabled", False)):
                print(f"  [REPEATED SUBJECT VALIDATION] {model_id}", flush=True)
                repeated_runs: List[pd.DataFrame] = []
                split_records: List[Dict[str, Any]] = []
                n_repeats = int(repeated_cfg.get("n_repeats", 20))
                for repeat_index in range(n_repeats):
                    train_dataset, validation_dataset, split_record = make_disjoint_subject_validation_split(
                        dataset, model, config, repeat_index
                    )
                    split_records.append(split_record)
                    if train_dataset is None or validation_dataset is None:
                        continue
                    validation_config = copy.deepcopy(config)
                    validation_config["sampling"]["random_seed"] = stable_seed(
                        int(repeated_cfg.get("random_seed", 1901)),
                        f"{model_id}|outer_validation_sampling={repeat_index}",
                    )
                    validation_config["sampling"]["min_subjects_per_arm"] = int(
                        repeated_cfg.get("min_subjects_per_arm", 3)
                    )
                    validation_config["permutation"]["n_iterations"] = int(
                        repeated_cfg.get("permutation_iterations_per_repeat", 100)
                    )
                    repeat_root = (
                        output_dir / "robustness" / "repeated_subject_validation"
                        / f"repeat_{repeat_index:03d}"
                    )
                    validation_result, validation_qc = run_model_with_resume(
                        validation_dataset,
                        drivers,
                        model,
                        validation_config,
                        repeat_root,
                        resume_runs,
                        geometry_fit_dataset=train_dataset,
                    )
                    split_record["model_qc"] = validation_qc
                    if validation_result is not None:
                        repeated_runs.append(validation_result)
                repeated_table = aggregate_repeated_subject_validation(
                    repeated_runs, primary_result, repeated_cfg
                )
                repeated_table.to_csv(
                    model_dir / "10_repeated_subject_validation_stability.csv", index=False
                )
                repeated_meta = {
                    "enabled": True,
                    "status": "ok" if repeated_runs else "no_completed_repeats",
                    "n_repeats_requested": n_repeats,
                    "n_repeats_completed": len(repeated_runs),
                    "splits": split_records,
                }
                write_json(
                    model_dir / "11_repeated_subject_validation.json", repeated_meta
                )
                final_result = final_result.merge(repeated_table, on="gene_symbol", how="left")
                final_result["repeated_subject_validation_supported"] = final_result[
                    "repeated_subject_validation_supported"
                ].fillna(False)
            else:
                final_result["repeated_subject_validation_supported"] = True

            if bool(repeated_cfg.get("require_for_final_eligibility", True)):
                final_result["ot_reversal_eligible"] = (
                    final_result["ot_reversal_eligible"].fillna(False)
                    & final_result["repeated_subject_validation_supported"].fillna(False)
                )

            # Leave-one-subject-out influence analysis on both disease and control arms.
            loso_cfg = robustness_cfg.get("leave_one_subject_out", {})
            loso_meta: Dict[str, Any] = {"enabled": False, "status": "disabled"}
            if robustness_enabled and bool(loso_cfg.get("enabled", False)):
                print(f"  [LOSO] {model_id}", flush=True)
                source_subjects, target_subjects, _, _ = model_subject_lists(
                    dataset, model, config
                )
                max_per_arm = int(loso_cfg.get("max_subjects_per_arm", 12))
                loso_rng = np.random.default_rng(
                    stable_seed(int(loso_cfg.get("random_seed", 2901)), model_id)
                )
                if max_per_arm > 0 and len(source_subjects) > max_per_arm:
                    source_subjects = sorted(
                        loso_rng.choice(np.asarray(source_subjects), size=max_per_arm, replace=False).tolist()
                    )
                if max_per_arm > 0 and len(target_subjects) > max_per_arm:
                    target_subjects = sorted(
                        loso_rng.choice(np.asarray(target_subjects), size=max_per_arm, replace=False).tolist()
                    )
                loso_runs: List[Tuple[str, str, pd.DataFrame]] = []
                loso_records: List[Dict[str, Any]] = []
                for omitted_arm, subjects_to_omit in [
                    ("SOURCE_DISEASE", source_subjects),
                    ("TARGET_CONTROL", target_subjects),
                ]:
                    for omitted_subject in subjects_to_omit:
                        loso_dataset = make_loso_dataset(
                            dataset, model, config, omitted_subject, omitted_arm
                        )
                        loso_config = copy.deepcopy(config)
                        loso_config["sampling"]["random_seed"] = stable_seed(
                            int(loso_cfg.get("random_seed", 2901)),
                            f"{model_id}|{omitted_arm}|{omitted_subject}",
                        )
                        loso_config["permutation"]["n_iterations"] = int(
                            loso_cfg.get("permutation_iterations_per_run", 75)
                        )
                        loso_root = (
                            output_dir / "robustness" / "leave_one_subject_out"
                            / sanitize_filename(omitted_arm)
                            / sanitize_filename(omitted_subject)
                        )
                        loso_result, loso_qc = run_model_with_resume(
                            loso_dataset, drivers, model, loso_config,
                            loso_root, resume_runs
                        )
                        loso_records.append({
                            "omitted_arm": omitted_arm,
                            "omitted_subject": omitted_subject,
                            "model_qc": loso_qc,
                        })
                        if loso_result is not None:
                            loso_runs.append((omitted_arm, omitted_subject, loso_result))
                loso_aggregate_cfg = dict(loso_cfg)
                loso_aggregate_cfg["_expected_runs"] = len(source_subjects) + len(target_subjects)
                loso_table = aggregate_loso_stability(
                    loso_runs, primary_result, loso_aggregate_cfg
                )
                loso_table.to_csv(model_dir / "12_loso_stability.csv", index=False)
                loso_meta = {
                    "enabled": True,
                    "status": "ok" if loso_runs else "no_completed_runs",
                    "n_runs_completed": len(loso_runs),
                    "runs": loso_records,
                }
                write_json(model_dir / "13_loso_run_qc.json", loso_meta)
                final_result = final_result.merge(loso_table, on="gene_symbol", how="left")
                final_result["loso_stability_supported"] = final_result[
                    "loso_stability_supported"
                ].fillna(False)
            else:
                final_result["loso_stability_supported"] = True

            if bool(loso_cfg.get("require_for_final_eligibility", True)):
                final_result["ot_reversal_eligible"] = (
                    final_result["ot_reversal_eligible"].fillna(False)
                    & final_result["loso_stability_supported"].fillna(False)
                )

            # OT/geometry sensitivity grid.
            sensitivity_cfg = robustness_cfg.get("sensitivity_grid", {})
            sensitivity_meta: Dict[str, Any] = {"enabled": False, "status": "disabled"}
            if robustness_enabled and bool(sensitivity_cfg.get("enabled", False)):
                print(f"  [SENSITIVITY GRID] {model_id}", flush=True)
                sensitivity_runs: List[Tuple[str, pd.DataFrame]] = []
                sensitivity_records: List[Dict[str, Any]] = []
                for setting in sensitivity_cfg.get("settings", []):
                    setting_id = str(setting.get("setting_id", f"setting_{len(sensitivity_runs)}"))
                    setting_config = copy.deepcopy(config)
                    setting_config["ot"]["regularization"] = float(
                        setting.get("ot_regularization", config["ot"]["regularization"])
                    )
                    setting_config["ot"]["cost_metric"] = str(
                        setting.get("cost_metric", config["ot"].get("cost_metric", "sqeuclidean"))
                    )
                    setting_config["ot"]["method"] = str(
                        setting.get("ot_method", config["ot"].get("method", "balanced_sinkhorn_log"))
                    )
                    setting_config["geometry"]["n_pcs"] = int(
                        setting.get("n_pcs", config["geometry"]["n_pcs"])
                    )
                    setting_config["geometry"]["n_variable_genes"] = int(
                        setting.get("n_variable_genes", config["geometry"]["n_variable_genes"])
                    )
                    setting_config["sampling"]["random_seed"] = stable_seed(
                        primary_seed, f"{model_id}|sensitivity|{setting_id}"
                    )
                    setting_config["permutation"]["n_iterations"] = int(
                        robustness_cfg.get("permutation_iterations_per_repeat", 100)
                    )
                    sensitivity_root = (
                        output_dir / "robustness" / "sensitivity_grid"
                        / sanitize_filename(setting_id)
                    )
                    setting_result, setting_qc = run_model_with_resume(
                        dataset, drivers, model, setting_config,
                        sensitivity_root, resume_runs
                    )
                    sensitivity_records.append({
                        "setting_id": setting_id,
                        "setting": setting,
                        "model_qc": setting_qc,
                    })
                    if setting_result is not None:
                        sensitivity_runs.append((setting_id, setting_result))
                sensitivity_table = aggregate_sensitivity_grid(
                    sensitivity_runs, primary_result, sensitivity_cfg
                )
                sensitivity_table.to_csv(
                    model_dir / "14_ot_geometry_sensitivity.csv", index=False
                )
                sensitivity_meta = {
                    "enabled": True,
                    "status": "ok" if sensitivity_runs else "no_completed_settings",
                    "n_settings_requested": len(sensitivity_cfg.get("settings", [])),
                    "n_settings_completed": len(sensitivity_runs),
                    "settings": sensitivity_records,
                }
                write_json(model_dir / "15_ot_geometry_sensitivity_qc.json", sensitivity_meta)
                final_result = final_result.merge(sensitivity_table, on="gene_symbol", how="left")
                final_result["sensitivity_grid_supported"] = final_result[
                    "sensitivity_grid_supported"
                ].fillna(False)
            else:
                final_result["sensitivity_grid_supported"] = True

            if bool(sensitivity_cfg.get("require_for_final_eligibility", True)):
                final_result["ot_reversal_eligible"] = (
                    final_result["ot_reversal_eligible"].fillna(False)
                    & final_result["sensitivity_grid_supported"].fillna(False)
                )

            # Whole-pipeline subject-label permutations. Each null run repeats
            # geometry selection, PCA, OT, scoring, FDR, and strict filtering.
            pipeline_perm_cfg = robustness_cfg.get("pipeline_label_permutation", {})
            pipeline_perm_qc: Dict[str, Any] = {"enabled": False, "status": "disabled"}
            if robustness_enabled and bool(pipeline_perm_cfg.get("enabled", False)):
                print(f"  [PIPELINE LABEL PERMUTATION] {model_id}", flush=True)
                permutation_runs: List[pd.DataFrame] = []
                permutation_records: List[Dict[str, Any]] = []
                n_pipeline_permutations = int(pipeline_perm_cfg.get("n_iterations", 100))
                for permutation_index in range(n_pipeline_permutations):
                    permuted_dataset, permutation_meta = make_subject_label_permuted_dataset(
                        dataset, model, config, permutation_index
                    )
                    permutation_config = copy.deepcopy(config)
                    permutation_config["sampling"]["random_seed"] = stable_seed(
                        int(pipeline_perm_cfg.get("random_seed", 3901)),
                        f"{model_id}|pipeline_perm_sampling={permutation_index}",
                    )
                    permutation_config["permutation"]["n_iterations"] = int(
                        pipeline_perm_cfg.get("local_pairing_iterations", 50)
                    )
                    permutation_root = (
                        output_dir / "robustness" / "pipeline_label_permutation"
                        / f"permutation_{permutation_index:04d}"
                    )
                    permutation_result, permutation_qc = run_model_with_resume(
                        permuted_dataset, drivers, model, permutation_config,
                        permutation_root, resume_runs
                    )
                    permutation_meta["model_qc"] = permutation_qc
                    permutation_records.append(permutation_meta)
                    if permutation_result is not None:
                        permutation_runs.append(permutation_result)
                pipeline_perm_table, pipeline_perm_summary = aggregate_pipeline_label_permutation(
                    permutation_runs, primary_result, pipeline_perm_cfg
                )
                pipeline_perm_table.to_csv(
                    model_dir / "16_pipeline_label_permutation.csv", index=False
                )
                pipeline_perm_qc = {
                    "enabled": True,
                    "status": "ok" if permutation_runs else "no_completed_permutations",
                    **pipeline_perm_summary,
                    "permutations": permutation_records,
                }
                write_json(
                    model_dir / "17_pipeline_label_permutation_qc.json", pipeline_perm_qc
                )
                final_result = final_result.merge(
                    pipeline_perm_table, on="gene_symbol", how="left"
                )
                final_result["pipeline_permutation_supported"] = final_result[
                    "pipeline_permutation_supported"
                ].fillna(False)
            else:
                final_result["pipeline_permutation_supported"] = True

            if bool(pipeline_perm_cfg.get("require_for_final_eligibility", True)):
                final_result["ot_reversal_eligible"] = (
                    final_result["ot_reversal_eligible"].fillna(False)
                    & final_result["pipeline_permutation_supported"].fillna(False)
                )

            final_result["ot_reversal_priority_score"] = final_result[
                "seed_median_priority_score"
            ].where(final_result["ot_reversal_eligible"], np.nan)
            final_result["final_robustness_status"] = np.select(
                [
                    ~final_result["robust_across_sampling_seeds"].fillna(False),
                    final_result["robust_across_sampling_seeds"].fillna(False)
                    & ~final_result["subject_split_replication_supported"].fillna(False),
                    final_result["robust_across_sampling_seeds"].fillna(False)
                    & final_result["subject_split_replication_supported"].fillna(False)
                    & ~final_result["repeated_subject_validation_supported"].fillna(False),
                    final_result["robust_across_sampling_seeds"].fillna(False)
                    & final_result["subject_split_replication_supported"].fillna(False)
                    & final_result["repeated_subject_validation_supported"].fillna(False)
                    & ~final_result["loso_stability_supported"].fillna(False),
                    final_result["robust_across_sampling_seeds"].fillna(False)
                    & final_result["subject_split_replication_supported"].fillna(False)
                    & final_result["repeated_subject_validation_supported"].fillna(False)
                    & final_result["loso_stability_supported"].fillna(False)
                    & ~final_result["sensitivity_grid_supported"].fillna(False),
                    final_result["robust_across_sampling_seeds"].fillna(False)
                    & final_result["subject_split_replication_supported"].fillna(False)
                    & final_result["repeated_subject_validation_supported"].fillna(False)
                    & final_result["loso_stability_supported"].fillna(False)
                    & final_result["sensitivity_grid_supported"].fillna(False)
                    & ~final_result["pipeline_permutation_supported"].fillna(False),
                ],
                [
                    "UNSTABLE_ACROSS_SAMPLING_SEEDS",
                    "NOT_REPLICATED_IN_LEGACY_SUBJECT_SPLIT",
                    "NOT_REPLICATED_IN_REPEATED_SUBJECT_VALIDATION",
                    "INFLUENTIAL_SINGLE_SUBJECT",
                    "SENSITIVE_TO_OT_OR_GEOMETRY_SETTINGS",
                    "NOT_SIGNIFICANT_IN_WHOLE_PIPELINE_NULL",
                ],
                default="ROBUST_V3",
            )

            rewrite_final_model_outputs(output_dir, model_id, final_result)
            write_json(model_dir / "09_subject_split_replication.json", split_meta)

            directional_n = int(
                (final_result["disease_direction"] != "NEUTRAL_OR_UNRESOLVED").sum()
            )
            primary_qc.update(
                {
                    "n_sampling_seeds_completed": int(len(seed_results)),
                    "sampling_seeds_completed": sorted(
                        {int(x["sampling_seed"].iloc[0]) for x in seed_results}
                    ),
                    "n_seed_robust_genes": int(
                        final_result["robust_across_sampling_seeds"].fillna(False).sum()
                    ),
                    "subject_split_replication": split_meta,
                    "repeated_subject_validation": repeated_meta,
                    "leave_one_subject_out": loso_meta,
                    "sensitivity_grid": sensitivity_meta,
                    "pipeline_label_permutation": pipeline_perm_qc,
                    "n_repeated_validation_supported_genes": int(
                        final_result["repeated_subject_validation_supported"].fillna(False).sum()
                    ),
                    "n_loso_supported_genes": int(
                        final_result["loso_stability_supported"].fillna(False).sum()
                    ),
                    "n_sensitivity_supported_genes": int(
                        final_result["sensitivity_grid_supported"].fillna(False).sum()
                    ),
                    "n_pipeline_permutation_supported_genes": int(
                        final_result["pipeline_permutation_supported"].fillna(False).sum()
                    ),
                    "n_final_robust_eligible_all_supported": int(
                        final_result["ot_reversal_eligible"].fillna(False).sum()
                    ),
                    "final_eligible_fraction_among_directional": float(
                        final_result["ot_reversal_eligible"].fillna(False).sum()
                        / max(directional_n, 1)
                    ),
                }
            )
            write_json(model_dir / "03_ot_model_qc.json", primary_qc)
            model_qc_rows.append(primary_qc)
            model_results.append(final_result)
        except Exception as exc:
            failure = {
                "model_id": model_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed_models.append(failure)
            model_qc_rows.append(failure)
            model_dir = output_dir / "models" / sanitize_filename(model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            write_json(model_dir / "03_ot_model_qc.json", failure)
            if bool(config.get("strict_missing_models", False)):
                raise
            warnings.warn(f"Model {model_id} failed: {exc}")

    model_qc_df = pd.DataFrame(model_qc_rows)
    model_qc_df.to_csv(output_dir / "15_model_qc_summary.csv", index=False)
    if not model_results:
        run_qc = {
            "status": "failed_no_models_completed",
            "n_models_requested": len(config["models"]),
            "n_models_completed": 0,
            "failed_models": failed_models,
        }
        write_json(output_dir / "16_run_qc_report.json", run_qc)
        raise RuntimeError("No OT model completed. Inspect model QC files.")

    all_results = pd.concat(model_results, ignore_index=True, sort=False)
    all_results.to_csv(output_dir / "10_all_model_driver_scores.csv", index=False)
    primary_consensus, sensitivity_consensus = build_consensus(all_results, drivers, config)
    primary_consensus.to_csv(output_dir / "11_primary_tier_a_ot_consensus.csv", index=False)
    sensitivity_consensus.to_csv(output_dir / "12_supported_tier_ab_ot_consensus.csv", index=False)

    primary_directional = all_results[
        all_results["is_primary_tier_a_candidate"] & all_results["ot_reversal_eligible"]
    ].copy()
    sensitivity_directional = all_results[
        all_results["is_supported_tier_ab_candidate"] & all_results["ot_reversal_eligible"]
    ].copy()
    primary_directional.to_csv(output_dir / "13_directional_driver_input_primary_tier_a.csv", index=False)
    sensitivity_directional.to_csv(output_dir / "14_directional_driver_input_sensitivity_tier_ab.csv", index=False)

    primary_consensus[primary_consensus["strict_consensus_eligible"]].to_csv(
        output_dir / "18_primary_tier_a_strict_consensus_shortlist.csv", index=False
    )
    sensitivity_consensus[sensitivity_consensus["strict_consensus_eligible"]].to_csv(
        output_dir / "19_supported_tier_ab_strict_consensus_shortlist.csv", index=False
    )
    sensitivity_consensus[
        sensitivity_consensus["consensus_disease_direction"] == "MIXED_ACROSS_MODELS"
    ].to_csv(output_dir / "20_mixed_direction_model_specific_genes.csv", index=False)
    write_methods_file(output_dir / "17_methods_and_score_definition.txt", config)

    status_values = [str(x.get("status", "")) for x in model_qc_rows]
    completed = sum(s in {"ok", "warning_ot_not_fully_converged"} for s in status_values)
    warning_models = [
        x.get("model_id") for x in model_qc_rows
        if x.get("status") == "warning_ot_not_fully_converged"
    ]
    skipped = [
        x.get("model_id") for x in model_qc_rows
        if str(x.get("status", "")).startswith("skipped")
    ]
    directional_total = int(
        (all_results["disease_direction"] != "NEUTRAL_OR_UNRESOLVED").sum()
    )
    eligible_total = int(all_results["ot_reversal_eligible"].fillna(False).sum())
    overall_eligible_fraction = float(eligible_total / max(directional_total, 1))
    run_qc = {
        "status": "ok" if completed == len(config["models"]) and not warning_models else "ok_with_warnings",
        "analysis_scope": "ACTIVE_PROGRESSION_STAGE3_TO_CONTROL",
        "method_version": SCRIPT_VERSION,
        "n_models_requested": len(config["models"]),
        "n_models_completed": completed,
        "n_models_with_convergence_warning": len(warning_models),
        "models_with_convergence_warning": warning_models,
        "n_models_skipped": len(skipped),
        "models_skipped": skipped,
        "n_models_failed": len(failed_models),
        "failed_models": failed_models,
        "n_all_model_gene_rows": int(len(all_results)),
        "n_directional_model_gene_rows": directional_total,
        "n_final_eligible_model_gene_rows": eligible_total,
        "final_eligible_fraction_among_directional": overall_eligible_fraction,
        "eligibility_selectivity_warning": bool(overall_eligible_fraction > 0.50),
        "n_primary_tier_a_directional_rows": int(len(primary_directional)),
        "n_sensitivity_tier_ab_directional_rows": int(len(sensitivity_directional)),
        "n_unique_primary_tier_a_genes_eligible_any_model": int(primary_directional["gene_symbol"].nunique()),
        "n_unique_tier_ab_genes_eligible_any_model": int(sensitivity_directional["gene_symbol"].nunique()),
        "n_primary_strict_consensus_genes": int(primary_consensus["strict_consensus_eligible"].sum()),
        "n_supported_strict_consensus_genes": int(sensitivity_consensus["strict_consensus_eligible"].sum()),
        "n_mixed_direction_genes": int(
            (sensitivity_consensus["consensus_disease_direction"] == "MIXED_ACROSS_MODELS").sum()
        ),
        "elapsed_seconds": float(time.time() - started),
        "important_interpretation": (
            "v3 adds leakage-controlled repeated subject validation, LOSO, OT/geometry "
            "sensitivity, transport-plan QC, whole-pipeline subject-label permutations, "
            "and partial-conjunction consensus. Results remain prioritization evidence, "
            "not causal intervention estimates."
        ),
    }
    write_json(output_dir / "16_run_qc_report.json", run_qc)
    print(f"[DONE] Output: {output_dir}", flush=True)
    return output_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-balanced OT disease-reversal analysis.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("35a_ot_paths.json")),
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write the default JSON config to --config and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if args.write_default_config:
        write_json(config_path, DEFAULT_CONFIG)
        print(f"Wrote default config: {config_path}")
        return
    if not config_path.exists():
        write_json(config_path, DEFAULT_CONFIG)
        raise FileNotFoundError(
            f"Config did not exist, so a template was created at {config_path}. "
            "Edit the paths/metadata labels and run again."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    config = deep_update(copy.deepcopy(DEFAULT_CONFIG), user_config)
    validate_v3_configuration(config)
    run_pipeline(config, config_path)


if __name__ == "__main__":
    main()