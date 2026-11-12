#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36a_finalize_ot_for_drug_pairing.py

Finalize OT disease-reversal outputs without changing any OT eligibility gate.
The script converts model-level robust results into auditable hierarchical
signatures for downstream drug pairing.

Key principles
--------------
1. `ot_reversal_eligible` and `final_robustness_status` are treated as frozen.
2. The original strict 3/4-model consensus is retained and never redefined.
3. Additional pairwise scopes are descriptive replication strata:
   - Dendritic cross-disease
   - Macrophage cross-disease
   - IPF cross-cell-type
   - COPD cross-cell-type
4. Tier A is the primary analysis; Tier A+B is sensitivity analysis.
5. Model-specific signatures remain model-specific and are not mislabeled as
   global disease-reversal targets.

Input
-----
Set OT_FINAL_VALIDATION_PATH near the top of this script to a directory or ZIP containing at least:
- 10_all_model_driver_scores.csv
- 11_primary_tier_a_ot_consensus.csv
- 12_supported_tier_ab_ot_consensus.csv
- 15_model_qc_summary.csv
- 16_run_qc_report.json

Output
------
Set OUTPUT_ROOT near the top of this script. The finalized directory contains compact QC, fixed exploratory ranking, hierarchical
signature files, a drug-pairing signature master table, candidate levels, and
methods text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.1.0-PYCHARM"
EPS = 1e-12

# =============================================================================
# PYCHARM USER SETTINGS — EDIT PATHS HERE
# =============================================================================
# Windows path example: r"E:\PythonProject\results\file.zip"
# You may also use forward slashes: "E:/PythonProject/results/file.zip"
# Relative paths are resolved relative to this .py file.

# Required: 35a final validation ZIP or its extracted directory.
OT_FINAL_VALIDATION_PATH = r"E:\PythonProject\results\OT_DISEASE_REVERSAL_v3_FINAL_VALIDATION.zip"

# Optional: CSV mapping Ensembl-like IDs to gene symbols.
# Leave as r"" to disable. Expected columns are described below.
GENE_ID_MAPPING_PATH = r""

# Required output directory. It will be created automatically.
OUTPUT_ROOT = r"E:\PythonProject\results\36a_OT_FINALIZED_FOR_DRUG_PAIRING"

# How to handle a non-empty output directory: "overwrite", "timestamp", or "error".
EXISTING_OUTPUT_POLICY = "overwrite"
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "ot_final_validation_path": "OT_DISEASE_REVERSAL_v3_FINAL_VALIDATION.zip",
        "gene_id_mapping_path": "",
    },
    "output": {
        "output_root": "36a_OT_FINALIZED_FOR_DRUG_PAIRING",
        "existing_output_policy": "overwrite",  # error | overwrite | timestamp
    },
    "eligibility": {
        "eligible_column": "ot_reversal_eligible",
        "required_final_status": "ROBUST_V3",
        "require_transport_plan_qc": True,
        "require_direction_resolved": True,
    },
    "tiers": {
        "tier_a_label": "TIER_A_HIGH_CONFIDENCE",
        "tier_b_label": "TIER_B_ROBUST_BUT_CAUTION",
        "tier_a_weight": 1.0,
        "tier_b_weight": 0.5,
    },
    "model_definitions": {
        "IPF_Macrophage_Stage3_to_Control": {
            "disease": "IPF",
            "celltype": "Macrophage",
            "stage": "Stage_3_intermediate_high",
        },
        "IPF_Dendritic_Stage3_to_Control": {
            "disease": "IPF",
            "celltype": "Dendritic",
            "stage": "Stage_3_intermediate_high",
        },
        "COPD_Macrophage_Stage3_to_Control": {
            "disease": "COPD",
            "celltype": "Macrophage",
            "stage": "Stage_3_intermediate_high",
        },
        "COPD_Dendritic_Stage3_to_Control": {
            "disease": "COPD",
            "celltype": "Dendritic",
            "stage": "Stage_3_intermediate_high",
        },
    },
    "pairwise_scopes": [
        {
            "scope_id": "DENDRITIC_CROSS_DISEASE",
            "scope_name": "Dendritic cross-disease",
            "required_models": [
                "IPF_Dendritic_Stage3_to_Control",
                "COPD_Dendritic_Stage3_to_Control",
            ],
            "disease_scope": "IPF;COPD",
            "celltype_scope": "Dendritic",
        },
        {
            "scope_id": "MACROPHAGE_CROSS_DISEASE",
            "scope_name": "Macrophage cross-disease",
            "required_models": [
                "IPF_Macrophage_Stage3_to_Control",
                "COPD_Macrophage_Stage3_to_Control",
            ],
            "disease_scope": "IPF;COPD",
            "celltype_scope": "Macrophage",
        },
        {
            "scope_id": "IPF_CROSS_CELLTYPE",
            "scope_name": "IPF cross-cell-type",
            "required_models": [
                "IPF_Macrophage_Stage3_to_Control",
                "IPF_Dendritic_Stage3_to_Control",
            ],
            "disease_scope": "IPF",
            "celltype_scope": "Macrophage;Dendritic",
        },
        {
            "scope_id": "COPD_CROSS_CELLTYPE",
            "scope_name": "COPD cross-cell-type",
            "required_models": [
                "COPD_Macrophage_Stage3_to_Control",
                "COPD_Dendritic_Stage3_to_Control",
            ],
            "disease_scope": "COPD",
            "celltype_scope": "Macrophage;Dendritic",
        },
    ],
    "signature": {
        "include_empty_signatures_in_registry": True,
        "write_one_file_per_signature": True,
        "minimum_models_for_exploratory_cross_model": 2,
        "model_support_bonus_per_extra_model": 0.25,
        "max_model_support_bonus": 0.50,
        "priority_floor": 0.25,
    },
    "candidate_levels": {
        "level_1": "Tier A replicated in a predefined cross-scope or original strict consensus",
        "level_2": "Tier B replicated in a predefined cross-scope, or Tier A robust in one model",
        "level_3": "Tier B robust in one model only",
    },
}


@dataclass
class InputBundle:
    root: Path
    temporary_root: Optional[Path]
    source_path: Path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(obj: Any) -> Any:
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
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        deep_update(config, user_config)
    return config


def resolve_path(value: Any, base_dir: Path, required: bool = True) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        if required:
            raise FileNotFoundError("Required path is empty.")
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def choose_output_dir(root: Path, policy: str) -> Path:
    policy = str(policy).strip().lower()
    if root.exists() and any(root.iterdir()):
        if policy == "error":
            raise FileExistsError(f"Output directory is non-empty: {root}")
        if policy == "overwrite":
            shutil.rmtree(root)
        elif policy == "timestamp":
            root = root.parent / f"{root.name}_{now_stamp()}"
        else:
            raise ValueError(f"Unknown output policy: {policy}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def open_input_bundle(path: Path) -> InputBundle:
    if path.is_dir():
        return InputBundle(root=path, temporary_root=None, source_path=path)
    if path.suffix.lower() != ".zip":
        raise ValueError("OT final-validation input must be a directory or ZIP file.")
    temp_root = Path(tempfile.mkdtemp(prefix="ot_finalize_"))
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(temp_root)
    return InputBundle(root=temp_root, temporary_root=temp_root, source_path=path)


def find_required_file(root: Path, basename: str) -> Path:
    matches = [p for p in root.rglob(basename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Required OT output absent: {basename}")
    if len(matches) > 1:
        matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    return series.astype("string").str.strip().str.lower().map(mapping).fillna(False).astype(bool)


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not math.isnan(default):
        out = out.fillna(default)
    return out


def normalize_gene_symbol(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def load_gene_mapping(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    source_col = None
    target_col = None
    for candidate in ("source_id", "ensembl_id", "gene_id", "input_gene"):
        if candidate in lower:
            source_col = lower[candidate]
            break
    for candidate in ("gene_symbol", "mapped_gene_symbol", "symbol"):
        if candidate in lower:
            target_col = lower[candidate]
            break
    if source_col is None or target_col is None:
        raise KeyError(
            "Gene mapping requires source_id/ensembl_id/gene_id and gene_symbol columns."
        )
    mapping: Dict[str, str] = {}
    for src, dst in zip(df[source_col], df[target_col]):
        src_n = normalize_gene_symbol(src)
        dst_n = normalize_gene_symbol(dst)
        if src_n and dst_n:
            mapping[src_n] = dst_n
    return mapping


def apply_gene_mapping(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["gene_symbol_original"] = out["gene_symbol"].map(normalize_gene_symbol)
    out["gene_symbol"] = out["gene_symbol_original"].map(lambda x: mapping.get(x, x))
    out["gene_id_was_mapped"] = out["gene_symbol"] != out["gene_symbol_original"]
    return out


def validate_score_table(df: pd.DataFrame) -> None:
    required = {
        "model_id",
        "gene_symbol",
        "ppi_driver_tier",
        "disease_direction",
        "reversal_direction",
        "ot_reversal_eligible",
        "final_robustness_status",
        "ot_reversal_priority_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"10_all_model_driver_scores.csv missing columns: {missing}")


def model_metadata(model_id: str, definitions: Mapping[str, Any]) -> Dict[str, str]:
    if model_id in definitions:
        item = definitions[model_id]
        return {
            "disease": str(item.get("disease", "")),
            "celltype": str(item.get("celltype", "")),
            "stage": str(item.get("stage", "")),
        }
    disease = "IPF" if "IPF" in model_id.upper() else "COPD" if "COPD" in model_id.upper() else "UNKNOWN"
    celltype = "Macrophage" if "MACROPHAGE" in model_id.upper() else "Dendritic" if "DENDRITIC" in model_id.upper() else "UNKNOWN"
    return {"disease": disease, "celltype": celltype, "stage": "UNKNOWN"}


def filter_frozen_eligible(scores: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["eligibility"]
    eligible_col = str(cfg.get("eligible_column", "ot_reversal_eligible"))
    if eligible_col not in scores.columns:
        raise KeyError(f"Eligibility column absent: {eligible_col}")
    mask = safe_bool(scores[eligible_col])

    required_status = str(cfg.get("required_final_status", "")).strip()
    if required_status:
        mask &= scores["final_robustness_status"].astype("string").eq(required_status)

    if bool(cfg.get("require_transport_plan_qc", True)) and "transport_plan_qc_passed" in scores.columns:
        mask &= safe_bool(scores["transport_plan_qc_passed"])

    if bool(cfg.get("require_direction_resolved", True)):
        mask &= scores["disease_direction"].isin(["UP_IN_DISEASE", "DOWN_IN_DISEASE"])
        mask &= scores["reversal_direction"].isin(["UP", "DOWN"])

    eligible = scores.loc[mask].copy()
    excluded = scores.loc[~mask].copy()
    return eligible, excluded


def direction_consistent(group: pd.DataFrame) -> bool:
    directions = set(group["reversal_direction"].dropna().astype(str))
    return len(directions) == 1 and len(directions) > 0


def aggregate_gene_level(eligible: pd.DataFrame, all_models: Sequence[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    metric_cols = [
        "disease_effect_z",
        "ot_local_pairing_corr",
        "ot_local_pairing_fdr",
        "two_arm_subject_direction_consistency",
        "source_subject_direction_ci_lower",
        "seed_eligibility_rate",
        "outer_validation_eligibility_rate",
        "loso_eligibility_rate",
        "sensitivity_eligibility_rate",
        "pipeline_permutation_priority_fdr",
        "ot_reversal_priority_score",
    ]
    for gene, group in eligible.groupby("gene_symbol", sort=False):
        models = sorted(group["model_id"].astype(str).unique())
        disease_dirs = sorted(group["disease_direction"].dropna().astype(str).unique())
        reversal_dirs = sorted(group["reversal_direction"].dropna().astype(str).unique())
        tiers = group["ppi_driver_tier"].astype(str)
        tier = "TIER_A_HIGH_CONFIDENCE" if (tiers == "TIER_A_HIGH_CONFIDENCE").any() else tiers.iloc[0]
        row: Dict[str, Any] = {
            "gene_symbol": gene,
            "ppi_driver_tier": tier,
            "is_tier_a_primary": tier == "TIER_A_HIGH_CONFIDENCE",
            "n_models_evaluated": len(all_models),
            "n_models_eligible": len(models),
            "model_eligibility_rate": len(models) / max(len(all_models), 1),
            "eligible_models": ";".join(models),
            "disease_direction_consistent": len(disease_dirs) == 1,
            "reversal_direction_consistent": len(reversal_dirs) == 1,
            "consensus_disease_direction": disease_dirs[0] if len(disease_dirs) == 1 else "MODEL_SPECIFIC",
            "consensus_reversal_direction": reversal_dirs[0] if len(reversal_dirs) == 1 else "MODEL_SPECIFIC",
        }
        for col in metric_cols:
            if col in group.columns:
                values = numeric(group[col])
                row[f"mean_{col}"] = float(values.mean()) if values.notna().any() else np.nan
                row[f"min_{col}"] = float(values.min()) if values.notna().any() else np.nan
                row[f"max_{col}"] = float(values.max()) if values.notna().any() else np.nan
        if "ot_reversal_priority_score" in group.columns:
            values = numeric(group["ot_reversal_priority_score"])
            if values.notna().any():
                best_idx = values.idxmax()
                row["best_model"] = str(group.loc[best_idx, "model_id"])
                row["best_priority_score"] = float(values.loc[best_idx])
            else:
                row["best_model"] = ""
                row["best_priority_score"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def validation_composite(row: Mapping[str, Any]) -> float:
    values: List[float] = []
    for col in (
        "mean_outer_validation_eligibility_rate",
        "mean_loso_eligibility_rate",
        "mean_sensitivity_eligibility_rate",
        "mean_seed_eligibility_rate",
    ):
        value = row.get(col, np.nan)
        if pd.notna(value):
            values.append(float(np.clip(value, 0.0, 1.0)))
    return float(np.mean(values)) if values else 0.0


def add_exploratory_consensus_score(gene_summary: pd.DataFrame, n_models: int) -> pd.DataFrame:
    out = gene_summary.copy()
    if out.empty:
        out["exploratory_consensus_score"] = pd.Series(dtype=float)
        out["exploratory_consensus_rank"] = pd.Series(dtype="Int64")
        return out
    tier_score = np.where(out["is_tier_a_primary"], 1.0, 0.6)
    support_score = numeric(out["n_models_eligible"], 0.0) / max(n_models, 1)
    priority_col = "mean_ot_reversal_priority_score"
    priority_score = numeric(out[priority_col], 0.0).clip(0.0, 1.0) if priority_col in out.columns else pd.Series(0.0, index=out.index)
    validation = out.apply(validation_composite, axis=1)
    direction_score = out["reversal_direction_consistent"].astype(float)
    out["exploratory_consensus_score"] = (
        0.30 * tier_score
        + 0.30 * support_score
        + 0.25 * priority_score
        + 0.10 * validation
        + 0.05 * direction_score
    )
    out = out.sort_values(
        ["n_models_eligible", "is_tier_a_primary", "exploratory_consensus_score", "gene_symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["exploratory_consensus_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["ranking_interpretation"] = (
        "Descriptive prioritization only; does not replace the original strict consensus test."
    )
    return out


def create_signature_rows(
    eligible: pd.DataFrame,
    signature_id: str,
    signature_name: str,
    signature_class: str,
    analysis_role: str,
    required_models: Sequence[str],
    disease_scope: str,
    celltype_scope: str,
    tier_mode: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    required_models = list(required_models)
    tier_a_label = config["tiers"]["tier_a_label"]
    tier_b_label = config["tiers"]["tier_b_label"]
    allowed_tiers = [tier_a_label] if tier_mode == "TIER_A" else [tier_a_label, tier_b_label]
    subset = eligible[
        eligible["model_id"].isin(required_models)
        & eligible["ppi_driver_tier"].isin(allowed_tiers)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for gene, group in subset.groupby("gene_symbol", sort=False):
        present_models = set(group["model_id"].astype(str))
        if not set(required_models).issubset(present_models):
            continue
        group = group[group["model_id"].isin(required_models)].copy()
        if not direction_consistent(group):
            continue
        tier = tier_a_label if (group["ppi_driver_tier"] == tier_a_label).any() else tier_b_label
        if tier_mode == "TIER_A" and tier != tier_a_label:
            continue
        priority = numeric(group["ot_reversal_priority_score"])
        mean_priority = float(priority.mean()) if priority.notna().any() else 0.0
        tier_weight = float(config["tiers"]["tier_a_weight"] if tier == tier_a_label else config["tiers"]["tier_b_weight"])
        bonus_per = float(config["signature"].get("model_support_bonus_per_extra_model", 0.25))
        max_bonus = float(config["signature"].get("max_model_support_bonus", 0.50))
        support_bonus = min(max_bonus, max(0, len(required_models) - 1) * bonus_per)
        priority_floor = float(config["signature"].get("priority_floor", 0.25))
        weight_raw = tier_weight * (priority_floor + (1.0 - priority_floor) * np.clip(mean_priority, 0.0, 1.0)) * (1.0 + support_bonus)
        records.append(
            {
                "signature_id": signature_id,
                "signature_name": signature_name,
                "signature_class": signature_class,
                "analysis_role": analysis_role,
                "tier_mode": tier_mode,
                "model_scope": ";".join(required_models),
                "disease_scope": disease_scope,
                "celltype_scope": celltype_scope,
                "gene_symbol": gene,
                "ppi_driver_tier": tier,
                "n_models_required": len(required_models),
                "n_models_supporting_gene": len(present_models.intersection(required_models)),
                "eligible_models": ";".join(sorted(present_models.intersection(required_models))),
                "disease_direction": str(group["disease_direction"].iloc[0]),
                "reversal_direction": str(group["reversal_direction"].iloc[0]),
                "mean_abs_disease_effect_z": float(numeric(group["disease_effect_z"]).abs().mean()),
                "mean_ot_local_pairing_corr": float(numeric(group["ot_local_pairing_corr"]).mean()),
                "max_ot_local_pairing_fdr": float(numeric(group["ot_local_pairing_fdr"]).max()),
                "mean_subject_direction_consistency": float(numeric(group["two_arm_subject_direction_consistency"]).mean()),
                "mean_outer_validation_eligibility_rate": float(numeric(group["outer_validation_eligibility_rate"]).mean()),
                "mean_loso_eligibility_rate": float(numeric(group["loso_eligibility_rate"]).mean()),
                "mean_sensitivity_eligibility_rate": float(numeric(group["sensitivity_eligibility_rate"]).mean()),
                "max_pipeline_permutation_fdr": float(numeric(group["pipeline_permutation_priority_fdr"]).max()),
                "mean_ot_reversal_priority_score": mean_priority,
                "tier_weight": tier_weight,
                "gene_weight_raw": float(weight_raw),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    total = float(out["gene_weight_raw"].sum())
    out["gene_weight_normalized"] = out["gene_weight_raw"] / total if total > 0 else 1.0 / len(out)
    out = out.sort_values(
        ["ppi_driver_tier", "gene_weight_raw", "gene_symbol"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    out["signature_gene_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def create_model_signature_rows(
    eligible: pd.DataFrame,
    model_id: str,
    metadata: Mapping[str, str],
    tier_mode: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    tier_a_label = config["tiers"]["tier_a_label"]
    tier_b_label = config["tiers"]["tier_b_label"]
    allowed = [tier_a_label] if tier_mode == "TIER_A" else [tier_a_label, tier_b_label]
    group = eligible[(eligible["model_id"] == model_id) & eligible["ppi_driver_tier"].isin(allowed)].copy()
    if group.empty:
        return pd.DataFrame()
    signature_id = f"MODEL_{model_id}__{tier_mode}"
    signature_name = f"{metadata['disease']} {metadata['celltype']} robust reversal ({tier_mode})"
    records: List[Dict[str, Any]] = []
    for _, row in group.iterrows():
        tier = str(row["ppi_driver_tier"])
        tier_weight = float(config["tiers"]["tier_a_weight"] if tier == tier_a_label else config["tiers"]["tier_b_weight"])
        priority = float(pd.to_numeric(pd.Series([row.get("ot_reversal_priority_score")]), errors="coerce").iloc[0])
        if not np.isfinite(priority):
            priority = 0.0
        priority_floor = float(config["signature"].get("priority_floor", 0.25))
        weight_raw = tier_weight * (priority_floor + (1.0 - priority_floor) * np.clip(priority, 0.0, 1.0))
        records.append(
            {
                "signature_id": signature_id,
                "signature_name": signature_name,
                "signature_class": "MODEL_SPECIFIC",
                "analysis_role": "PRIMARY" if tier_mode == "TIER_A" else "SENSITIVITY",
                "tier_mode": tier_mode,
                "model_scope": model_id,
                "disease_scope": metadata["disease"],
                "celltype_scope": metadata["celltype"],
                "gene_symbol": row["gene_symbol"],
                "ppi_driver_tier": tier,
                "n_models_required": 1,
                "n_models_supporting_gene": 1,
                "eligible_models": model_id,
                "disease_direction": row["disease_direction"],
                "reversal_direction": row["reversal_direction"],
                "mean_abs_disease_effect_z": abs(float(row.get("disease_effect_z", np.nan))),
                "mean_ot_local_pairing_corr": float(row.get("ot_local_pairing_corr", np.nan)),
                "max_ot_local_pairing_fdr": float(row.get("ot_local_pairing_fdr", np.nan)),
                "mean_subject_direction_consistency": float(row.get("two_arm_subject_direction_consistency", np.nan)),
                "mean_outer_validation_eligibility_rate": float(row.get("outer_validation_eligibility_rate", np.nan)),
                "mean_loso_eligibility_rate": float(row.get("loso_eligibility_rate", np.nan)),
                "mean_sensitivity_eligibility_rate": float(row.get("sensitivity_eligibility_rate", np.nan)),
                "max_pipeline_permutation_fdr": float(row.get("pipeline_permutation_priority_fdr", np.nan)),
                "mean_ot_reversal_priority_score": priority,
                "tier_weight": tier_weight,
                "gene_weight_raw": float(weight_raw),
            }
        )
    out = pd.DataFrame(records)
    total = float(out["gene_weight_raw"].sum())
    out["gene_weight_normalized"] = out["gene_weight_raw"] / total if total > 0 else 1.0 / len(out)
    out = out.sort_values(["ppi_driver_tier", "gene_weight_raw", "gene_symbol"], ascending=[True, False, True]).reset_index(drop=True)
    out["signature_gene_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def build_candidate_universe(
    gene_summary: pd.DataFrame,
    signatures: pd.DataFrame,
    strict_primary: pd.DataFrame,
    strict_supported: pd.DataFrame,
    tier_a_label: str,
) -> pd.DataFrame:
    universe = gene_summary.copy()
    if universe.empty:
        return universe

    membership: Dict[str, List[str]] = {}
    primary_cross: set = set()
    sensitivity_cross: set = set()
    model_primary: set = set()
    model_sensitivity: set = set()
    if not signatures.empty:
        for gene, group in signatures.groupby("gene_symbol"):
            ids = sorted(group["signature_id"].astype(str).unique())
            membership[str(gene)] = ids
            for _, row in group.iterrows():
                signature_class = str(row["signature_class"])
                role = str(row["analysis_role"])
                if signature_class == "PAIRWISE_CROSS_SCOPE" and role == "PRIMARY":
                    primary_cross.add(str(gene))
                elif signature_class == "PAIRWISE_CROSS_SCOPE" and role == "SENSITIVITY":
                    sensitivity_cross.add(str(gene))
                elif signature_class == "MODEL_SPECIFIC" and role == "PRIMARY":
                    model_primary.add(str(gene))
                elif signature_class == "MODEL_SPECIFIC" and role == "SENSITIVITY":
                    model_sensitivity.add(str(gene))

    strict_primary_genes = set(strict_primary.get("gene_symbol", pd.Series(dtype=str)).astype(str))
    strict_supported_genes = set(strict_supported.get("gene_symbol", pd.Series(dtype=str)).astype(str))
    universe["signature_memberships"] = universe["gene_symbol"].map(lambda g: ";".join(membership.get(str(g), [])))
    universe["in_original_primary_strict_consensus"] = universe["gene_symbol"].isin(strict_primary_genes)
    universe["in_original_supported_strict_consensus"] = universe["gene_symbol"].isin(strict_supported_genes)
    universe["in_primary_cross_scope_signature"] = universe["gene_symbol"].isin(primary_cross)
    universe["in_sensitivity_cross_scope_signature"] = universe["gene_symbol"].isin(sensitivity_cross)
    universe["in_model_specific_primary_signature"] = universe["gene_symbol"].isin(model_primary)
    universe["in_model_specific_sensitivity_signature"] = universe["gene_symbol"].isin(model_sensitivity)

    def assign_level(row: pd.Series) -> str:
        if bool(row["in_original_primary_strict_consensus"]) or bool(row["in_primary_cross_scope_signature"]):
            return "LEVEL_1_REPLICATED_TIER_A"
        if bool(row["in_original_supported_strict_consensus"]) or bool(row["in_sensitivity_cross_scope_signature"]):
            return "LEVEL_2_REPLICATED_TIER_AB"
        if str(row["ppi_driver_tier"]) == tier_a_label:
            return "LEVEL_2_MODEL_SPECIFIC_TIER_A"
        return "LEVEL_3_MODEL_SPECIFIC_TIER_B"

    universe["candidate_level"] = universe.apply(assign_level, axis=1)
    order = {
        "LEVEL_1_REPLICATED_TIER_A": 1,
        "LEVEL_2_REPLICATED_TIER_AB": 2,
        "LEVEL_2_MODEL_SPECIFIC_TIER_A": 3,
        "LEVEL_3_MODEL_SPECIFIC_TIER_B": 4,
    }
    universe["candidate_level_order"] = universe["candidate_level"].map(order).fillna(99).astype(int)
    universe["recommended_drug_direction"] = universe["consensus_reversal_direction"]
    universe["eligible_for_drug_pairing"] = universe["recommended_drug_direction"].isin(["UP", "DOWN"])
    universe = universe.sort_values(
        ["candidate_level_order", "n_models_eligible", "exploratory_consensus_score", "gene_symbol"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    universe["candidate_universe_rank"] = np.arange(1, len(universe) + 1, dtype=int)
    return universe


def compact_model_qc(model_qc: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "model_id",
        "status",
        "n_source_observations_available",
        "n_target_observations_available",
        "n_source_subjects",
        "n_target_subjects",
        "n_source_observations_sampled",
        "n_target_observations_sampled",
        "n_supported_driver_genes_in_expression",
        "n_directional_genes",
        "n_reversal_eligible_all_supported",
        "n_final_robust_eligible_all_supported",
        "n_final_robust_eligible_tier_a",
        "eligible_fraction_among_directional",
    ]
    present = [c for c in preferred if c in model_qc.columns]
    remaining = [c for c in model_qc.columns if c not in present]
    return model_qc[present + remaining]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_methods(path: Path, registry: pd.DataFrame, qc: Mapping[str, Any]) -> None:
    lines = [
        "OT FINALIZATION FOR DRUG PAIRING",
        "================================",
        "",
        f"Finalization script version: {SCRIPT_VERSION}",
        "",
        "Frozen eligibility rule",
        "-----------------------",
        "No OT result was re-estimated and no statistical threshold was relaxed.",
        "Only rows already marked ot_reversal_eligible=True and ROBUST_V3 were retained.",
        "The original strict 3/4-model consensus remains the formal global consensus.",
        "Pairwise cross-scope signatures are descriptive replication strata and must not be",
        "reported as substitutes for the original strict consensus.",
        "",
        "Signature hierarchy",
        "-------------------",
        "PRIMARY signatures contain Tier A genes only.",
        "SENSITIVITY signatures contain Tier A+B genes.",
        "Pairwise cross-scope signatures require a gene to be robust in every predefined",
        "model in that scope and to have a consistent reversal direction.",
        "Model-specific signatures retain genes robust in one disease-cell-type model.",
        "",
        "Drug-pairing gene weights",
        "-------------------------",
        "Tier A weight = 1.0; Tier B weight = 0.5. The tier weight is multiplied by a",
        "bounded model-level OT priority term and, for cross-scope signatures, a fixed",
        "replication bonus. Weights are normalized to sum to 1 within each signature.",
        "These weights are prioritization weights, not causal effect estimates.",
        "",
        "Interpretation",
        "--------------",
        "LEVEL_1: Tier A replicated in a predefined pairwise scope or original strict consensus.",
        "LEVEL_2: replicated Tier A+B support or model-specific Tier A support.",
        "LEVEL_3: model-specific Tier B support.",
        "Mixed-direction genes are excluded from unified drug-direction signatures and should",
        "remain model-specific.",
        "",
        "Run summary",
        "-----------",
        f"Frozen robust model-gene rows: {qc.get('n_robust_model_gene_rows', 0)}",
        f"Unique robust Tier A genes: {qc.get('n_unique_tier_a_genes', 0)}",
        f"Unique robust Tier A+B genes: {qc.get('n_unique_tier_ab_genes', 0)}",
        f"Original strict Tier A consensus genes: {qc.get('n_original_primary_strict_consensus', 0)}",
        f"Original strict Tier A+B consensus genes: {qc.get('n_original_supported_strict_consensus', 0)}",
        "",
        "Signature registry",
        "------------------",
    ]
    for _, row in registry.iterrows():
        lines.append(
            f"{row['signature_id']}: {int(row['n_genes'])} genes; "
            f"class={row['signature_class']}; role={row['analysis_role']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Build the analysis settings entirely from this script. No JSON config is used.
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["ot_final_validation_path"] = OT_FINAL_VALIDATION_PATH
    config["input"]["gene_id_mapping_path"] = GENE_ID_MAPPING_PATH
    config["output"]["output_root"] = OUTPUT_ROOT
    config["output"]["existing_output_policy"] = EXISTING_OUTPUT_POLICY

    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(OT_FINAL_VALIDATION_PATH, script_dir, required=True)
    assert input_path is not None

    mapping_path: Optional[Path] = None
    if str(GENE_ID_MAPPING_PATH).strip():
        mapping_path = resolve_path(GENE_ID_MAPPING_PATH, script_dir, required=True)

    output_root = resolve_path(OUTPUT_ROOT, script_dir, required=False)
    assert output_root is not None
    output_root = choose_output_dir(output_root, EXISTING_OUTPUT_POLICY)

    print("=" * 78)
    print("36a OT finalization (PyCharm direct-path version)")
    print(f"Input : {input_path}")
    print(f"Mapping: {mapping_path if mapping_path else 'DISABLED'}")
    print(f"Output: {output_root}")
    print("=" * 78)

    bundle = open_input_bundle(input_path)
    try:
        file_paths = {
            "scores": find_required_file(bundle.root, "10_all_model_driver_scores.csv"),
            "primary_consensus": find_required_file(bundle.root, "11_primary_tier_a_ot_consensus.csv"),
            "supported_consensus": find_required_file(bundle.root, "12_supported_tier_ab_ot_consensus.csv"),
            "model_qc": find_required_file(bundle.root, "15_model_qc_summary.csv"),
            "run_qc": find_required_file(bundle.root, "16_run_qc_report.json"),
            "primary_strict": find_required_file(bundle.root, "18_primary_tier_a_strict_consensus_shortlist.csv"),
            "supported_strict": find_required_file(bundle.root, "19_supported_tier_ab_strict_consensus_shortlist.csv"),
            "mixed_direction": find_required_file(bundle.root, "20_mixed_direction_model_specific_genes.csv"),
        }
        scores = pd.read_csv(file_paths["scores"])
        primary_consensus = pd.read_csv(file_paths["primary_consensus"])
        supported_consensus = pd.read_csv(file_paths["supported_consensus"])
        model_qc = pd.read_csv(file_paths["model_qc"])
        primary_strict = pd.read_csv(file_paths["primary_strict"])
        supported_strict = pd.read_csv(file_paths["supported_strict"])
        mixed_direction_original = pd.read_csv(file_paths["mixed_direction"])
        with file_paths["run_qc"].open("r", encoding="utf-8") as handle:
            run_qc = json.load(handle)

        validate_score_table(scores)
        scores["gene_symbol"] = scores["gene_symbol"].map(normalize_gene_symbol)
        gene_mapping = load_gene_mapping(mapping_path)
        scores = apply_gene_mapping(scores, gene_mapping)

        definitions = config["model_definitions"]
        scores["disease"] = scores["model_id"].map(lambda x: model_metadata(str(x), definitions)["disease"])
        scores["celltype"] = scores["model_id"].map(lambda x: model_metadata(str(x), definitions)["celltype"])
        scores["stage"] = scores["model_id"].map(lambda x: model_metadata(str(x), definitions)["stage"])

        eligible, excluded = filter_frozen_eligible(scores, config)
        all_models = sorted(scores["model_id"].astype(str).unique())
        tier_a_label = config["tiers"]["tier_a_label"]

        gene_summary = aggregate_gene_level(eligible, all_models)
        gene_summary = add_exploratory_consensus_score(gene_summary, len(all_models))

        cross_signature_frames: List[pd.DataFrame] = []
        registry_rows: List[Dict[str, Any]] = []
        for scope in config["pairwise_scopes"]:
            for tier_mode, role in (("TIER_A", "PRIMARY"), ("TIER_AB", "SENSITIVITY")):
                signature_id = f"{scope['scope_id']}__{tier_mode}"
                frame = create_signature_rows(
                    eligible=eligible,
                    signature_id=signature_id,
                    signature_name=f"{scope['scope_name']} ({tier_mode})",
                    signature_class="PAIRWISE_CROSS_SCOPE",
                    analysis_role=role,
                    required_models=scope["required_models"],
                    disease_scope=scope["disease_scope"],
                    celltype_scope=scope["celltype_scope"],
                    tier_mode=tier_mode,
                    config=config,
                )
                if not frame.empty:
                    cross_signature_frames.append(frame)
                registry_rows.append(
                    {
                        "signature_id": signature_id,
                        "signature_name": f"{scope['scope_name']} ({tier_mode})",
                        "signature_class": "PAIRWISE_CROSS_SCOPE",
                        "analysis_role": role,
                        "tier_mode": tier_mode,
                        "model_scope": ";".join(scope["required_models"]),
                        "disease_scope": scope["disease_scope"],
                        "celltype_scope": scope["celltype_scope"],
                        "n_genes": int(len(frame)),
                        "is_empty": bool(frame.empty),
                    }
                )

        model_signature_frames: List[pd.DataFrame] = []
        for model_id in all_models:
            meta = model_metadata(model_id, definitions)
            for tier_mode, role in (("TIER_A", "PRIMARY"), ("TIER_AB", "SENSITIVITY")):
                frame = create_model_signature_rows(eligible, model_id, meta, tier_mode, config)
                signature_id = f"MODEL_{model_id}__{tier_mode}"
                if not frame.empty:
                    model_signature_frames.append(frame)
                registry_rows.append(
                    {
                        "signature_id": signature_id,
                        "signature_name": f"{meta['disease']} {meta['celltype']} robust reversal ({tier_mode})",
                        "signature_class": "MODEL_SPECIFIC",
                        "analysis_role": role,
                        "tier_mode": tier_mode,
                        "model_scope": model_id,
                        "disease_scope": meta["disease"],
                        "celltype_scope": meta["celltype"],
                        "n_genes": int(len(frame)),
                        "is_empty": bool(frame.empty),
                    }
                )

        cross_signatures = pd.concat(cross_signature_frames, ignore_index=True, sort=False) if cross_signature_frames else pd.DataFrame()
        model_signatures = pd.concat(model_signature_frames, ignore_index=True, sort=False) if model_signature_frames else pd.DataFrame()
        signatures = pd.concat([cross_signatures, model_signatures], ignore_index=True, sort=False) if (not cross_signatures.empty or not model_signatures.empty) else pd.DataFrame()
        registry = pd.DataFrame(registry_rows)

        candidate_universe = build_candidate_universe(
            gene_summary=gene_summary,
            signatures=signatures,
            strict_primary=primary_strict,
            strict_supported=supported_strict,
            tier_a_label=tier_a_label,
        )

        unresolved_mask = scores["gene_symbol"].astype(str).str.match(r"^ENSG\d+", case=False, na=False)
        unresolved = (
            scores.loc[unresolved_mask, ["gene_symbol_original", "gene_symbol"]]
            .drop_duplicates()
            .sort_values("gene_symbol")
            .reset_index(drop=True)
        )
        unresolved["recommended_action"] = "Provide an Ensembl-to-gene-symbol mapping before final publication or external database matching."

        # Output files.
        compact_model_qc(model_qc).to_csv(output_root / "01_model_qc_compact.csv", index=False)
        eligible.to_csv(output_root / "02_robust_model_gene_pairs.csv", index=False)
        eligible[eligible["ppi_driver_tier"] == tier_a_label].to_csv(output_root / "03_primary_tier_a_any_model.csv", index=False)
        eligible.to_csv(output_root / "04_supported_tier_ab_any_model.csv", index=False)
        gene_summary.to_csv(output_root / "05_exploratory_consensus_all_robust_genes.csv", index=False)
        cross_signatures.to_csv(output_root / "06_cross_scope_signatures_long.csv", index=False)
        model_signatures.to_csv(output_root / "07_model_specific_signatures_long.csv", index=False)
        signatures.to_csv(output_root / "08_drug_pairing_signatures_long.csv", index=False)
        registry.to_csv(output_root / "09_signature_registry.csv", index=False)
        candidate_universe.to_csv(output_root / "10_candidate_screening_universe.csv", index=False)
        unresolved.to_csv(output_root / "11_unresolved_gene_ids.csv", index=False)
        excluded.to_csv(output_root / "12_nonrobust_rows_excluded_from_drug_pairing.csv", index=False)
        mixed_direction_original.to_csv(output_root / "13_original_mixed_direction_table.csv", index=False)

        # One file per non-empty signature.
        signature_dir = output_root / "signatures"
        signature_dir.mkdir(parents=True, exist_ok=True)
        if not signatures.empty and bool(config["signature"].get("write_one_file_per_signature", True)):
            for signature_id, frame in signatures.groupby("signature_id", sort=False):
                frame.to_csv(signature_dir / f"{safe_filename(str(signature_id))}.csv", index=False)

        lists_dir = output_root / "candidate_lists"
        lists_dir.mkdir(parents=True, exist_ok=True)
        if not candidate_universe.empty:
            for level, frame in candidate_universe.groupby("candidate_level", sort=False):
                genes = frame.loc[frame["eligible_for_drug_pairing"], "gene_symbol"].astype(str).tolist()
                (lists_dir / f"{safe_filename(str(level))}.txt").write_text("\n".join(genes) + ("\n" if genes else ""), encoding="utf-8")

        qc = {
            "status": "ok",
            "script_version": SCRIPT_VERSION,
            "source_run_status": run_qc.get("status"),
            "source_method_version": run_qc.get("method_version"),
            "source_analysis_scope": run_qc.get("analysis_scope"),
            "n_models": len(all_models),
            "models": all_models,
            "n_input_model_gene_rows": int(len(scores)),
            "n_robust_model_gene_rows": int(len(eligible)),
            "n_excluded_model_gene_rows": int(len(excluded)),
            "n_unique_tier_a_genes": int(eligible.loc[eligible["ppi_driver_tier"] == tier_a_label, "gene_symbol"].nunique()),
            "n_unique_tier_ab_genes": int(eligible["gene_symbol"].nunique()),
            "n_original_primary_strict_consensus": int(len(primary_strict)),
            "n_original_supported_strict_consensus": int(len(supported_strict)),
            "n_original_mixed_direction_genes": int(len(mixed_direction_original)),
            "n_pairwise_cross_scope_signature_rows": int(len(cross_signatures)),
            "n_model_specific_signature_rows": int(len(model_signatures)),
            "n_total_signature_rows": int(len(signatures)),
            "n_nonempty_signatures": int((registry["n_genes"] > 0).sum()),
            "n_empty_signatures": int((registry["n_genes"] == 0).sum()),
            "n_level_1_genes": int((candidate_universe.get("candidate_level", pd.Series(dtype=str)) == "LEVEL_1_REPLICATED_TIER_A").sum()),
            "n_level_2_replicated_genes": int((candidate_universe.get("candidate_level", pd.Series(dtype=str)) == "LEVEL_2_REPLICATED_TIER_AB").sum()),
            "n_level_2_model_specific_tier_a_genes": int((candidate_universe.get("candidate_level", pd.Series(dtype=str)) == "LEVEL_2_MODEL_SPECIFIC_TIER_A").sum()),
            "n_level_3_model_specific_tier_b_genes": int((candidate_universe.get("candidate_level", pd.Series(dtype=str)) == "LEVEL_3_MODEL_SPECIFIC_TIER_B").sum()),
            "n_unresolved_ensembl_like_ids": int(len(unresolved)),
            "strict_consensus_interpretation": "The original strict 3/4-model result is retained unchanged.",
            "pairwise_scope_interpretation": "Descriptive predefined replication strata; not a replacement for strict global consensus.",
        }
        write_json(output_root / "14_ot_finalization_qc.json", qc)

        manifest = {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "run_time": datetime.now().isoformat(),
            "input_source": str(input_path),
            "input_source_sha256": sha256_file(input_path) if input_path.is_file() else None,
            "input_files": {key: str(value) for key, value in file_paths.items()},
            "gene_mapping_path": str(mapping_path) if mapping_path else None,
            "gene_mapping_entries": len(gene_mapping),
            "output_root": str(output_root),
            "config": config,
        }
        write_json(output_root / "00_input_manifest.json", manifest)
        write_methods(output_root / "15_methods_and_interpretation.txt", registry, qc)

        print(f"OT finalization completed: {output_root}")
        print(json.dumps(qc, ensure_ascii=False, indent=2, default=json_default))
    finally:
        if bundle.temporary_root is not None:
            shutil.rmtree(bundle.temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
