#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
37a_run_drug_pairing_sigcom_FINAL.py

Final source-aware drug-prioritisation pipeline for the merged/global SigCom
Up/Down query generated from robust OT disease-reversal signatures.

Final design principles
-----------------------
1. The merged SigCom query is global evidence and is scored exactly once per
   drug; it is never duplicated across the individual OT signatures.
2. Strict two-arm FDR and best-arm FDR are retained separately. One-arm support
   is reported explicitly and is never relabelled as strict evidence.
3. Drug-target evidence is analysed in a main source-cleaned track and a
   downweighted sensitivity track. Suspicious-source-only mechanisms cannot
   become source-robust main mechanisms.
4. Conflicting activation/inhibition evidence is consolidated and audited.
5. Network scores are calibrated by propagated influence mass. Missing score
   components contribute zero and are not re-normalised to 100 percent.
6. Missing safety, pulmonary-risk, and availability information is reported as
   missing rather than interpreted as evidence of safety.
7. The script contains one effective main() entry point and writes full input,
   parameter, source-policy, and query-statistics provenance.

This program performs computational prioritisation only. Pharmacology, exposure,
dose, route, toxicity, and experimental validation remain necessary before a
candidate can be described as translationally supported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12
SCRIPT_VERSION = "3.1.0-FINAL-GLOBAL-QUERY-SOURCE-AWARE"
FINAL_SCRIPT_SCHEMA_VERSION = "37a-final-schema-1"

# =============================================================================
# PYCHARM USER SETTINGS — EDIT PATHS HERE
# =============================================================================
SIGNATURE_PATH = (
    r"E:\PythonProject\results\36a_OT_FINALIZED_FOR_DRUG_PAIRING"
    r"\08_drug_pairing_signatures_long.csv"
)
DRUG_PERTURBATION_PATH = r"E:\PythonProject\drug_data\drug_perturbation_signature.csv"
DRUG_TARGET_PATH = r"E:\PythonProject\drug_data\drug_target_actions.csv"
DRUG_METADATA_PATH = r"E:\PythonProject\drug_data\drug_metadata.csv"
NETWORK_EDGES_PATH = r"E:\PythonProject\drug_data\gene_network_edges.csv"
GENE_ALIAS_MAPPING_PATH = r"E:\PythonProject\drug_data\gene_alias_mapping.csv"

OUTPUT_ROOT = r"E:\PythonProject\results\37a_DRUG_PAIRING_RESULTS"
EXISTING_OUTPUT_POLICY = "overwrite"  # overwrite | timestamp | error
PERTURBATION_MODE = "auto"            # auto | signature_level | gene_level
ALLOW_TRUE_TEMPLATE_ROWS = False

# Final SigCom input behaviour.
# When an explicit is_reverser column exists, rows marked False are excluded.
SOURCE_REVERSERS_ONLY = True
# A warning is recorded when a drug has discordant query-score signs across
# multiple contexts; the rows are retained for transparent aggregation.
MIN_QUERY_CONTEXT_SIGN_CONSISTENCY = 0.50
# =============================================================================


DEFAULT_CONFIG: Dict[str, Any] = {
    "filters": {
        "minimum_gene_context_consistency": 0.50,
        "minimum_perturbation_evidence_weight": 0.0,
        "maximum_qvalue": 1.0,
        "exclude_high_pulmonary_risk": True,
        "high_risk_terms": [
            "high",
            "severe",
            "contraindicated",
            "pulmonary toxicity",
            "lung toxicity",
        ],
    },
    "effect": {
        "effect_scale": 1.0,
        "cap_effect_magnitude": 3.0,
        "max_qvalue_weight": 4.0,
        "direction_only_effect_magnitude": 1.0,
    },
    "target_action_map": {
        "inhibitor": -1,
        "antagonist": -1,
        "blocker": -1,
        "degrader": -1,
        "suppressor": -1,
        "downregulator": -1,
        "negative modulator": -1,
        "agonist": 1,
        "activator": 1,
        "inducer": 1,
        "upregulator": 1,
        "positive modulator": 1,
        "binder": 0,
        "unknown": 0,
    },
    "network": {
        "enabled": True,
        "restart_probability": 0.35,
        "n_iterations": 40,
        "convergence_tolerance": 1e-7,
        "batch_size": 128,
        "minimum_edge_weight": 0.0,
    },
    "scoring": {
        "component_weights_gene_level": {
            "expression_reversal_score": 0.45,
            "direct_target_alignment_score": 0.20,
            "network_reversal_score": 0.15,
            "clinical_score": 0.08,
            "safety_score": 0.08,
            "availability_score": 0.04,
        },
        "component_weights_signature_level": {
            "signature_query_reversal_score": 0.50,
            "direct_target_alignment_score": 0.20,
            "network_reversal_score": 0.10,
            "clinical_score": 0.08,
            "safety_score": 0.08,
            "availability_score": 0.04,
        },
        "signature_role_weights": {
            "PAIRWISE_CROSS_SCOPE|PRIMARY": 2.0,
            "PAIRWISE_CROSS_SCOPE|SENSITIVITY": 1.0,
            "MODEL_SPECIFIC|PRIMARY": 1.0,
            "MODEL_SPECIFIC|SENSITIVITY": 0.5,
        },
        "minimum_matched_genes": 1,
        "minimum_weighted_coverage": 0.10,
        "minimum_reversal_fraction": 0.55,
        "maximum_conflict_fraction": 0.45,
        "positive_signature_score": 0.25,
        "strong_conflict_signature_score": -0.25,
        "coverage_multiplier_floor": 0.50,
    },
    "candidate_tiers": {
        "gene_level": {
            "tier_a_min_global_score": 0.55,
            "tier_a_min_primary_positive_signatures": 1,
            "tier_a_min_tier_a_genes_matched": 1,
            "tier_b_min_global_score": 0.40,
            "tier_b_min_positive_signatures": 1,
            "exploratory_min_global_score": 0.0,
        },
        "signature_level": {
            "tier_a_min_global_score": 0.55,
            "tier_a_min_query_score": 0.60,
            "tier_a_max_qvalue": 0.05,
            "tier_b_min_global_score": 0.35,
            "tier_b_min_query_score": 0.35,
            "tier_b_max_qvalue": 0.25,
            "exploratory_min_global_score": 0.0,
        },
    },
}


def normalize_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_gene(value: Any) -> str:
    return normalize_text(value).upper()


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if not math.isnan(default):
        result = result.fillna(default)
    return result


def read_csv_safe(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_path(value: Any, base: Path, required: bool = False) -> Optional[Path]:
    if value is None or not str(value).strip():
        if required:
            raise FileNotFoundError("Required path is empty.")
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Input not found: {path}")
        return None
    return path


def resolve_optional_input_path(value: Any, base: Path, label: str) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def choose_output_dir(path: Path, policy: str) -> Path:
    policy = str(policy).strip().lower()
    if path.exists() and any(path.iterdir()):
        if policy == "error":
            raise FileExistsError(path)
        if policy == "overwrite":
            shutil.rmtree(path)
        elif policy == "timestamp":
            path = path.parent / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            raise ValueError(f"Unknown output policy: {policy}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if np.isfinite(value) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing(lower: Mapping[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        column = lower.get(name.casefold())
        if column is not None:
            return column
    return None


def source_column(df: pd.DataFrame) -> Optional[str]:
    return {str(c).casefold(): c for c in df.columns}.get("source")


def true_template_mask(
    df: pd.DataFrame,
    drug_id_col: str,
    gene_col: Optional[str] = None,
    action_col: Optional[str] = None,
) -> pd.Series:
    """Conservatively detect genuine template rows.

    ``EXAMPLE_`` by itself is not sufficient because some public pharmacology
    sources use that prefix for real patent compounds.
    """
    if df.empty or drug_id_col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    prefix = (
        df[drug_id_col]
        .astype("string")
        .str.strip()
        .str.startswith("EXAMPLE_", na=False)
    )

    src_col = source_column(df)
    if src_col is None:
        source_text = pd.Series("", index=df.index, dtype="string")
    else:
        source_text = (
            df[src_col]
            .fillna("")
            .astype("string")
            .str.strip()
            .str.casefold()
        )
    explicit_source = source_text.str.contains(
        r"\b(?:example|template|placeholder|demo|dummy)\b",
        regex=True,
        na=False,
    )

    explicit_fields = pd.Series(False, index=df.index, dtype=bool)
    for col in [gene_col, action_col]:
        if col and col in df.columns:
            text = (
                df[col]
                .fillna("")
                .astype("string")
                .str.strip()
                .str.casefold()
            )
            explicit_fields |= text.str.contains(
                r"^(?:example|template|placeholder|demo|dummy)(?:_|$)",
                regex=True,
                na=False,
            )

    declaration = pd.Series(False, index=df.index, dtype=bool)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        text = (
            df[col]
            .fillna("")
            .astype("string")
            .str.strip()
            .str.casefold()
        )
        declaration |= text.str.contains(
            r"(?:template row|example row|placeholder row|dummy row)",
            regex=True,
            na=False,
        )

    return prefix & (explicit_source | explicit_fields | declaration)


def reject_true_template_rows(
    df: pd.DataFrame,
    drug_id_col: str,
    allow: bool,
    gene_col: Optional[str] = None,
    action_col: Optional[str] = None,
) -> None:
    if allow or df.empty:
        return
    mask = true_template_mask(df, drug_id_col, gene_col, action_col)
    if bool(mask.any()):
        examples = df.loc[mask, drug_id_col].astype(str).head(5).tolist()
        raise ValueError(
            "True template rows detected. Remove them or set "
            "ALLOW_TRUE_TEMPLATE_ROWS=True for testing only. "
            f"Examples: {examples}"
        )


def load_alias_mapping(path: Optional[Path]) -> Dict[str, str]:
    df = read_csv_safe(path)
    if df.empty:
        return {}
    lower = {str(c).casefold(): c for c in df.columns}
    alias_col = first_existing(lower, ["alias", "source_id", "input_gene"])
    symbol_col = first_existing(
        lower, ["gene_symbol", "symbol", "mapped_gene_symbol"]
    )
    if alias_col is None or symbol_col is None:
        raise KeyError(
            "Alias mapping requires alias/source_id and gene_symbol columns."
        )
    return {
        normalize_gene(alias): normalize_gene(symbol)
        for alias, symbol in zip(df[alias_col], df[symbol_col])
        if normalize_gene(alias) and normalize_gene(symbol)
    }


def validate_signatures(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signature_id",
        "signature_name",
        "signature_class",
        "analysis_role",
        "gene_symbol",
        "ppi_driver_tier",
        "reversal_direction",
        "gene_weight_normalized",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Signature file missing columns: {missing}")

    out = df.copy()
    for col, default in {
        "tier_mode": "",
        "disease_scope": "",
        "celltype_scope": "",
    }.items():
        if col not in out.columns:
            out[col] = default

    out["gene_symbol"] = out["gene_symbol"].map(normalize_gene)
    out["reversal_direction"] = (
        out["reversal_direction"].fillna("").astype("string").str.upper().str.strip()
    )
    out = out[
        out["reversal_direction"].isin(["UP", "DOWN"])
        & out["gene_symbol"].ne("")
    ].copy()
    out["desired_sign"] = np.where(out["reversal_direction"].eq("UP"), 1.0, -1.0)
    out["gene_weight_normalized"] = numeric(
        out["gene_weight_normalized"], 0.0
    ).clip(lower=0.0)

    for _, idx in out.groupby("signature_id").groups.items():
        total = float(out.loc[idx, "gene_weight_normalized"].sum())
        out.loc[idx, "gene_weight_normalized"] = (
            out.loc[idx, "gene_weight_normalized"] / total
            if total > 0
            else 1.0 / len(idx)
        )

    if out.empty:
        raise ValueError("No valid OT signature rows remain after validation.")
    return out.reset_index(drop=True)


def detect_perturbation_mode(df: pd.DataFrame, requested: str) -> str:
    requested = requested.strip().lower()
    if requested in {"gene_level", "signature_level"}:
        return requested
    if requested != "auto":
        raise ValueError(
            "PERTURBATION_MODE must be auto, gene_level, or signature_level."
        )

    cols = {str(c).casefold() for c in df.columns}
    has_gene = "gene_symbol" in cols and (
        "signed_effect" in cols or "effect_direction" in cols
    )
    has_query = bool(
        {"query_reversal_score", "z_score_sum", "reversal_score"} & cols
    )
    if has_gene:
        return "gene_level"
    if has_query:
        return "signature_level"
    raise KeyError(
        "Cannot detect perturbation mode. Expected gene_symbol plus "
        "signed_effect/effect_direction, or query_reversal_score/z_score_sum."
    )


def load_gene_level_perturbations(
    df: pd.DataFrame,
    alias: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    lower = {str(c).casefold(): c for c in df.columns}
    drug_id = lower.get("drug_id")
    gene = lower.get("gene_symbol")
    if drug_id is None or gene is None:
        raise KeyError("Gene-level perturbation table needs drug_id and gene_symbol.")

    reject_true_template_rows(
        df,
        drug_id,
        ALLOW_TRUE_TEMPLATE_ROWS,
        gene_col=gene,
    )

    drug_name = lower.get("drug_name")
    context = lower.get("context_id")
    source = lower.get("source")
    signed_col = lower.get("signed_effect")
    direction_col = lower.get("effect_direction")
    weight_col = lower.get("evidence_weight")
    q_col = lower.get("qvalue") or lower.get("qvalue_strict")

    out = pd.DataFrame(index=df.index)
    out["drug_id"] = df[drug_id].map(normalize_text)
    out["drug_name"] = (
        df[drug_name].map(normalize_text) if drug_name else out["drug_id"]
    )
    out["gene_symbol"] = df[gene].map(normalize_gene).map(
        lambda x: alias.get(x, x)
    )
    out["context_id"] = (
        df[context].map(normalize_text) if context else "UNSPECIFIED"
    )
    out["source"] = df[source].map(normalize_text) if source else "UNSPECIFIED"

    signed = (
        numeric(df[signed_col])
        if signed_col
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    if direction_col:
        direction = (
            df[direction_col].fillna("").astype("string").str.upper().str.strip()
        )
        sign = direction.map(
            {"UP": 1.0, "DOWN": -1.0, "INCREASE": 1.0, "DECREASE": -1.0}
        )
        magnitude = float(cfg["effect"]["direction_only_effect_magnitude"])
        signed = signed.fillna(sign * magnitude)

    out["signed_effect"] = signed
    out["evidence_weight"] = (
        numeric(df[weight_col], 1.0).clip(lower=0.0) if weight_col else 1.0
    )
    out["qvalue"] = (
        numeric(df[q_col], 1.0).clip(lower=EPS, upper=1.0) if q_col else 1.0
    )

    out = out[
        out["drug_id"].ne("")
        & out["gene_symbol"].ne("")
        & np.isfinite(out["signed_effect"])
    ].copy()
    out = out[
        (out["qvalue"] <= float(cfg["filters"]["maximum_qvalue"]))
        & (
            out["evidence_weight"]
            >= float(cfg["filters"]["minimum_perturbation_evidence_weight"])
        )
    ]

    q_weight = (-np.log10(out["qvalue"].clip(lower=EPS))).clip(
        upper=float(cfg["effect"]["max_qvalue_weight"])
    )
    out["combined_weight"] = out["evidence_weight"] * (1.0 + q_weight)
    return out.reset_index(drop=True)


def aggregate_gene_level(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    scale = max(float(cfg["effect"]["effect_scale"]), EPS)
    cap = float(cfg["effect"]["cap_effect_magnitude"])
    min_consistency = float(cfg["filters"]["minimum_gene_context_consistency"])

    for (drug_id, drug_name, gene), group in df.groupby(
        ["drug_id", "drug_name", "gene_symbol"], dropna=False
    ):
        weights = numeric(group["combined_weight"], 1.0).to_numpy(float)
        effects = numeric(group["signed_effect"]).to_numpy(float)
        valid = np.isfinite(weights) & (weights > 0) & np.isfinite(effects)
        if not valid.any():
            continue
        weights = weights[valid]
        effects = effects[valid]
        mean_effect = float(np.average(effects, weights=weights))
        consistency = float(
            abs(np.sum(weights * np.sign(effects))) / max(np.sum(weights), EPS)
        )
        if consistency < min_consistency:
            continue

        records.append(
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "gene_symbol": gene,
                "aggregated_signed_effect": mean_effect,
                "bounded_signed_effect": float(
                    np.tanh(np.clip(mean_effect / scale, -cap, cap))
                ),
                "context_sign_consistency": consistency,
                "n_context_rows": int(len(group)),
                "n_unique_contexts": int(group["context_id"].nunique()),
                "evidence_weight_sum": float(weights.sum()),
                "sources": ";".join(sorted(set(group["source"].astype(str)))),
            }
        )

    return pd.DataFrame(records)


def robust_signed_score(raw: pd.Series) -> pd.Series:
    values = numeric(raw)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=raw.index, dtype=float)

    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs <= 1.000001:
        return values.clip(-1.0, 1.0)

    scale = float(np.nanpercentile(np.abs(finite), 90))
    if not np.isfinite(scale) or scale <= EPS:
        scale = max(float(np.nanstd(finite)), 1.0)
    return np.tanh(values / scale)


def load_signature_level_perturbations(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    lower = {str(c).casefold(): c for c in df.columns}
    drug_id_col = first_existing(
        lower, ["drug_id", "perturbagen_id", "compound_id"]
    )
    if drug_id_col is None:
        raise KeyError("Signature-level perturbation table needs drug_id.")

    reject_true_template_rows(df, drug_id_col, ALLOW_TRUE_TEMPLATE_ROWS)

    drug_name_col = first_existing(
        lower, ["drug_name", "perturbagen_name", "compound_name"]
    )
    score_col = first_existing(
        lower, ["query_reversal_score", "reversal_score"]
    )
    z_col = first_existing(lower, ["z_score_sum", "zscore_sum"])
    if score_col is None and z_col is None:
        raise KeyError(
            "Signature-level table needs query_reversal_score or z_score_sum."
        )

    q_col = first_existing(lower, ["qvalue_strict", "qvalue", "q_value", "fdr"])
    weight_col = first_existing(lower, ["evidence_weight", "weight"])
    sig_id_col = first_existing(
        lower, ["perturbation_signature_id", "signature_id", "sig_id"]
    )
    context_col = first_existing(lower, ["context_id", "context"])
    cell_col = first_existing(lower, ["cell_line", "cell_id", "cell"])
    dose_col = first_existing(lower, ["dose", "pert_dose"])
    time_col = first_existing(lower, ["timepoint", "time", "pert_time"])
    source_col = first_existing(lower, ["source", "database"])

    out = pd.DataFrame(index=df.index)
    out["drug_id"] = df[drug_id_col].map(normalize_text)
    out["drug_name"] = (
        df[drug_name_col].map(normalize_text)
        if drug_name_col
        else out["drug_id"]
    )

    raw_score = numeric(df[score_col]) if score_col else -numeric(df[z_col])
    out["query_reversal_score_raw"] = raw_score
    out["query_reversal_score_normalized"] = robust_signed_score(raw_score)
    out["qvalue"] = (
        numeric(df[q_col], 1.0).clip(lower=EPS, upper=1.0) if q_col else 1.0
    )
    out["evidence_weight"] = (
        numeric(df[weight_col], 1.0).clip(lower=0.0) if weight_col else 1.0
    )
    out["perturbation_signature_id"] = (
        df[sig_id_col].map(normalize_text) if sig_id_col else ""
    )
    out["context_id"] = df[context_col].map(normalize_text) if context_col else ""
    out["cell_line"] = df[cell_col].map(normalize_text) if cell_col else ""
    out["dose"] = df[dose_col].map(normalize_text) if dose_col else ""
    out["timepoint"] = df[time_col].map(normalize_text) if time_col else ""
    out["source"] = df[source_col].map(normalize_text) if source_col else "SigCom"

    out = out[
        out["drug_id"].ne("")
        & np.isfinite(out["query_reversal_score_normalized"])
    ].copy()
    out = out[
        (out["qvalue"] <= float(cfg["filters"]["maximum_qvalue"]))
        & (
            out["evidence_weight"]
            >= float(cfg["filters"]["minimum_perturbation_evidence_weight"])
        )
    ]

    q_weight = (-np.log10(out["qvalue"].clip(lower=EPS))).clip(
        upper=float(cfg["effect"]["max_qvalue_weight"])
    )
    out["combined_weight"] = out["evidence_weight"] * (1.0 + q_weight)
    return out.reset_index(drop=True)


def aggregate_signature_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name), group in df.groupby(
        ["drug_id", "drug_name"], dropna=False
    ):
        scores = numeric(group["query_reversal_score_normalized"]).to_numpy(float)
        weights = numeric(group["combined_weight"], 1.0).to_numpy(float)
        valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue

        scores = scores[valid]
        weights = weights[valid]
        weighted_mean = float(np.average(scores, weights=weights))
        best_idx = group["query_reversal_score_normalized"].idxmax()
        best = group.loc[best_idx]

        records.append(
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "signature_query_reversal_score": weighted_mean,
                "signature_query_best_score": float(
                    best["query_reversal_score_normalized"]
                ),
                "signature_query_qvalue": float(
                    numeric(pd.Series([best["qvalue"]]), 1.0).iloc[0]
                ),
                "n_query_context_rows": int(len(group)),
                "n_query_contexts": int(
                    group.loc[
                        group["context_id"]
                        .fillna("")
                        .astype("string")
                        .str.strip()
                        .ne(""),
                        "context_id",
                    ].nunique()
                ),
                "best_perturbation_signature_id": normalize_text(
                    best["perturbation_signature_id"]
                ),
                "best_query_context_id": normalize_text(best["context_id"]),
                "best_cell_line": normalize_text(best["cell_line"]),
                "best_dose": normalize_text(best["dose"]),
                "best_timepoint": normalize_text(best["timepoint"]),
                "query_sources": ";".join(sorted(set(group["source"].astype(str)))),
            }
        )

    return pd.DataFrame(records)


def action_sign(value: Any, mapping: Mapping[str, Any]) -> float:
    text = normalize_text(value).casefold()
    if text in mapping:
        return float(mapping[text])
    for key, sign in mapping.items():
        if text and str(key).casefold() in text:
            return float(sign)
    return 0.0


def load_targets(
    path: Optional[Path],
    alias: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = read_csv_safe(path)
    qc: Dict[str, Any] = {
        "rows_input": int(len(df)),
        "example_prefix_rows_kept": 0,
        "true_template_rows": 0,
        "blank_drug_id_repaired": 0,
        "rows_dropped_missing_identity_or_gene": 0,
        "rows_output": 0,
        "unknown_action_rows": 0,
    }
    if df.empty:
        return pd.DataFrame(), qc

    lower = {str(c).casefold(): c for c in df.columns}
    drug_id_col = lower.get("drug_id")
    gene_col = lower.get("gene_symbol")
    action_col = lower.get("action")
    if drug_id_col is None or gene_col is None or action_col is None:
        raise KeyError(
            "Drug target table requires drug_id, gene_symbol, and action."
        )

    template = true_template_mask(df, drug_id_col, gene_col, action_col)
    qc["true_template_rows"] = int(template.sum())
    if template.any() and not ALLOW_TRUE_TEMPLATE_ROWS:
        df = df.loc[~template].copy()

    example_prefix = (
        df[drug_id_col]
        .astype("string")
        .str.strip()
        .str.startswith("EXAMPLE_", na=False)
    )
    qc["example_prefix_rows_kept"] = int(example_prefix.sum())

    drug_name_col = lower.get("drug_name")
    concept_col = first_existing(
        lower, ["drug_concept_id", "concept_id", "drugbank_id"]
    )
    weight_col = lower.get("evidence_weight")
    source_col = lower.get("source")

    raw_id = df[drug_id_col].map(normalize_text)
    fallback_concept = (
        df[concept_col].map(normalize_text)
        if concept_col
        else pd.Series("", index=df.index, dtype="string")
    )
    fallback_name = (
        df[drug_name_col].map(normalize_text)
        if drug_name_col
        else pd.Series("", index=df.index, dtype="string")
    )

    repaired = raw_id.eq("") & fallback_concept.ne("")
    qc["blank_drug_id_repaired"] = int(repaired.sum())

    stable_id = raw_id.where(raw_id.ne(""), fallback_concept)
    stable_id = stable_id.where(
        stable_id.ne(""),
        fallback_name.map(lambda x: f"NAME:{x}" if x else ""),
    )

    out = pd.DataFrame(index=df.index)
    out["drug_id"] = stable_id
    out["drug_name"] = fallback_name.where(fallback_name.ne(""), stable_id)
    out["drug_concept_id"] = fallback_concept
    out["gene_symbol"] = df[gene_col].map(normalize_gene).map(
        lambda x: alias.get(x, x)
    )
    out["action"] = df[action_col].map(normalize_text)
    out["evidence_weight"] = (
        numeric(df[weight_col], 1.0).clip(lower=0.0) if weight_col else 1.0
    )
    out["source"] = (
        df[source_col].map(normalize_text) if source_col else "UNSPECIFIED"
    )
    out["action_sign"] = out["action"].map(
        lambda x: action_sign(x, cfg["target_action_map"])
    )

    valid = out["drug_id"].ne("") & out["gene_symbol"].ne("")
    qc["rows_dropped_missing_identity_or_gene"] = int((~valid).sum())
    out = out.loc[valid].copy()
    qc["unknown_action_rows"] = int(out["action_sign"].eq(0).sum())

    out = out.drop_duplicates(
        ["drug_id", "gene_symbol", "action", "source"], keep="first"
    ).reset_index(drop=True)

    qc["rows_output"] = int(len(out))
    qc["unique_drugs"] = int(out["drug_id"].nunique())
    qc["unique_genes"] = int(out["gene_symbol"].nunique())
    return out, qc


def load_metadata(path: Optional[Path]) -> pd.DataFrame:
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    lower = {str(c).casefold(): c for c in df.columns}
    id_col = lower.get("drug_id")
    if id_col is None:
        raise KeyError("Drug metadata needs drug_id.")

    name_col = lower.get("drug_name")
    out = pd.DataFrame({"drug_id": df[id_col].map(normalize_text)})
    out["drug_name"] = (
        df[name_col].map(normalize_text) if name_col else out["drug_id"]
    )

    for col in [
        "approval_status",
        "pulmonary_risk",
        "notes",
        "clinical_score",
        "safety_score",
        "availability_score",
    ]:
        src = lower.get(col)
        if col.endswith("_score"):
            out[col] = numeric(df[src]).clip(0.0, 1.0) if src else np.nan
        else:
            out[col] = df[src].map(normalize_text) if src else ""

    approval_map = {
        "approved": 1.0,
        "phase 4": 0.95,
        "phase 3": 0.85,
        "phase 2": 0.70,
        "phase 1": 0.55,
        "clinical": 0.65,
        "preclinical": 0.35,
        "experimental": 0.20,
    }
    missing = out["clinical_score"].isna()
    if missing.any():
        inferred = out.loc[missing, "approval_status"].str.casefold().map(
            lambda x: max(
                [value for key, value in approval_map.items() if key in x],
                default=np.nan,
            )
        )
        out.loc[missing, "clinical_score"] = inferred

    return out[out["drug_id"].ne("")].drop_duplicates("drug_id", keep="first")


def load_network(
    path: Optional[Path],
    alias: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    lower = {str(c).casefold(): c for c in df.columns}
    gene_a = lower.get("gene_a")
    gene_b = lower.get("gene_b")
    if gene_a is None or gene_b is None:
        raise KeyError("Network table needs gene_a and gene_b.")

    weight_col = lower.get("weight")
    out = pd.DataFrame(
        {
            "gene_a": df[gene_a].map(normalize_gene).map(
                lambda x: alias.get(x, x)
            ),
            "gene_b": df[gene_b].map(normalize_gene).map(
                lambda x: alias.get(x, x)
            ),
            "weight": (
                numeric(df[weight_col], 1.0).clip(lower=0.0)
                if weight_col
                else 1.0
            ),
        }
    )

    min_weight = float(cfg["network"]["minimum_edge_weight"])
    return out[
        out["gene_a"].ne("")
        & out["gene_b"].ne("")
        & out["gene_a"].ne(out["gene_b"])
        & (out["weight"] >= min_weight)
    ].reset_index(drop=True)


def score_expression(
    signatures: pd.DataFrame,
    perturbations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if perturbations.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = signatures.merge(perturbations, on="gene_symbol", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["directional_match"] = (
        merged["desired_sign"] * np.sign(merged["aggregated_signed_effect"])
    )
    merged["gene_reversal_score"] = (
        merged["desired_sign"] * merged["bounded_signed_effect"]
    )
    merged["weighted_gene_reversal"] = (
        merged["gene_weight_normalized"] * merged["gene_reversal_score"]
    )

    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name, signature_id), group in merged.groupby(
        ["drug_id", "drug_name", "signature_id"], dropna=False
    ):
        weights = numeric(group["gene_weight_normalized"], 0.0).to_numpy(float)
        matched_weight = float(weights.sum())
        positive_weight = float(
            weights[group["directional_match"].to_numpy() > 0].sum()
        )
        negative_weight = float(
            weights[group["directional_match"].to_numpy() < 0].sum()
        )
        tier_a = group["ppi_driver_tier"].eq("TIER_A_HIGH_CONFIDENCE")

        records.append(
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "signature_id": signature_id,
                "expression_reversal_score": float(
                    group["weighted_gene_reversal"].sum()
                    / max(matched_weight, EPS)
                ),
                "weighted_coverage": matched_weight,
                "reversal_fraction": positive_weight / max(matched_weight, EPS),
                "conflict_fraction": negative_weight / max(matched_weight, EPS),
                "n_matched_genes": int(group["gene_symbol"].nunique()),
                "n_tier_a_genes_matched": int(
                    group.loc[tier_a, "gene_symbol"].nunique()
                ),
                "matched_genes": ";".join(sorted(group["gene_symbol"].unique())),
            }
        )

    return pd.DataFrame(records), merged


def score_direct_targets(
    signatures: pd.DataFrame,
    targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if targets.empty:
        return pd.DataFrame(), pd.DataFrame()

    directional = targets[targets["action_sign"].ne(0)].copy()
    if directional.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = signatures.merge(
        directional,
        on="gene_symbol",
        how="inner",
        suffixes=("", "_target"),
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["target_alignment"] = merged["desired_sign"] * merged["action_sign"]
    merged["target_weighted_alignment"] = (
        merged["gene_weight_normalized"]
        * merged["target_alignment"]
        * numeric(merged["evidence_weight"], 1.0)
    )

    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name, signature_id), group in merged.groupby(
        ["drug_id", "drug_name", "signature_id"], dropna=False
    ):
        denominator = float(
            (
                group["gene_weight_normalized"]
                * numeric(group["evidence_weight"], 1.0)
            ).sum()
        )
        records.append(
            {
                "drug_id": drug_id,
                "drug_name": drug_name,
                "signature_id": signature_id,
                "direct_target_alignment_score": float(
                    group["target_weighted_alignment"].sum()
                    / max(denominator, EPS)
                ),
                "n_direct_signature_targets": int(
                    group["gene_symbol"].nunique()
                ),
                "n_direct_tier_a_targets": int(
                    group.loc[
                        group["ppi_driver_tier"].eq("TIER_A_HIGH_CONFIDENCE"),
                        "gene_symbol",
                    ].nunique()
                ),
                "direct_signature_targets": ";".join(
                    sorted(group["gene_symbol"].unique())
                ),
            }
        )

    return pd.DataFrame(records), merged


def network_scores(
    signatures: pd.DataFrame,
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if targets.empty or edges.empty or not bool(cfg["network"]["enabled"]):
        return pd.DataFrame()

    signed = targets[targets["action_sign"].ne(0)].copy()
    if signed.empty:
        return pd.DataFrame()

    nodes = sorted(
        set(edges["gene_a"])
        | set(edges["gene_b"])
        | set(signatures["gene_symbol"])
        | set(signed["gene_symbol"])
    )
    index = {gene: i for i, gene in enumerate(nodes)}

    rows = edges["gene_a"].map(index).to_numpy(int)
    cols = edges["gene_b"].map(index).to_numpy(int)
    values = numeric(edges["weight"], 1.0).to_numpy(float)
    adjacency = sparse.coo_matrix(
        (
            np.r_[values, values],
            (np.r_[rows, cols], np.r_[cols, rows]),
        ),
        shape=(len(nodes), len(nodes)),
    ).tocsr()

    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0,
    )
    transition_t = (sparse.diags(inverse) @ adjacency).T.tocsr()

    drug_groups = list(signed.groupby("drug_id", sort=False))
    signature_groups = {
        signature_id: group
        for signature_id, group in signatures.groupby("signature_id", sort=False)
    }

    restart = float(cfg["network"]["restart_probability"])
    n_iterations = int(cfg["network"]["n_iterations"])
    tolerance = float(cfg["network"]["convergence_tolerance"])
    batch_size = int(cfg["network"]["batch_size"])
    records: List[Dict[str, Any]] = []

    for start in range(0, len(drug_groups), batch_size):
        batch = drug_groups[start : start + batch_size]
        seeds = np.zeros((len(nodes), len(batch)), dtype=float)
        drug_ids: List[str] = []
        drug_names: List[str] = []

        for j, (drug_id, group) in enumerate(batch):
            drug_ids.append(str(drug_id))
            drug_names.append(str(group["drug_name"].iloc[0]))
            for _, row in group.iterrows():
                node_idx = index.get(row["gene_symbol"])
                if node_idx is not None:
                    seeds[node_idx, j] += (
                        float(row["action_sign"]) * float(row["evidence_weight"])
                    )
            denominator = np.sum(np.abs(seeds[:, j]))
            if denominator > 0:
                seeds[:, j] /= denominator

        influence = seeds.copy()
        for _ in range(n_iterations):
            updated = restart * seeds + (1.0 - restart) * (transition_t @ influence)
            if np.max(np.abs(updated - influence)) < tolerance:
                influence = updated
                break
            influence = updated

        for j, _ in enumerate(batch):
            for signature_id, group in signature_groups.items():
                genes = [gene for gene in group["gene_symbol"] if gene in index]
                if not genes:
                    continue
                idx = np.array([index[gene] for gene in genes], dtype=int)
                lookup = group.set_index("gene_symbol")
                desired = lookup.loc[genes, "desired_sign"].to_numpy(float)
                weights = lookup.loc[genes, "gene_weight_normalized"].to_numpy(float)
                influence_values = influence[idx, j]
                mass = float(np.sum(weights * np.abs(influence_values)))
                score = (
                    float(np.sum(weights * desired * influence_values) / mass)
                    if mass > 0
                    else np.nan
                )
                records.append(
                    {
                        "drug_id": drug_ids[j],
                        "drug_name": drug_names[j],
                        "signature_id": signature_id,
                        "network_reversal_score": score,
                        "network_signature_influence_mass": mass,
                    }
                )

    return pd.DataFrame(records)


def combine_scores(
    signatures: pd.DataFrame,
    perturbation_mode: str,
    expression: pd.DataFrame,
    query: pd.DataFrame,
    direct: pd.DataFrame,
    network: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    registry_columns = [
        "signature_id",
        "signature_name",
        "signature_class",
        "analysis_role",
        "tier_mode",
        "disease_scope",
        "celltype_scope",
    ]
    registry = signatures[registry_columns].drop_duplicates()

    drug_frames: List[pd.DataFrame] = []
    for frame in [expression, query, direct, network, metadata]:
        if frame is not None and not frame.empty:
            drug_frames.append(frame[["drug_id", "drug_name"]].drop_duplicates())
    if not drug_frames:
        raise ValueError("No usable drug evidence rows were loaded.")

    drugs = pd.concat(drug_frames, ignore_index=True).drop_duplicates(
        "drug_id", keep="first"
    )
    base = (
        drugs.assign(_k=1)
        .merge(registry.assign(_k=1), on="_k")
        .drop(columns="_k")
    )

    for frame, suffix in [
        (expression, "_expr"),
        (direct, "_target"),
        (network, "_network"),
    ]:
        if frame is not None and not frame.empty:
            base = base.merge(
                frame,
                on=["drug_id", "signature_id"],
                how="left",
                suffixes=("", suffix),
            )
            name_col = f"drug_name{suffix}"
            if name_col in base.columns:
                base["drug_name"] = (
                    base["drug_name"].replace("", np.nan).fillna(base[name_col])
                )
                base = base.drop(columns=name_col)

    if query is not None and not query.empty:
        base = base.merge(
            query.drop(columns="drug_name", errors="ignore"),
            on="drug_id",
            how="left",
        )

    if metadata is not None and not metadata.empty:
        base = base.merge(
            metadata.drop(columns="drug_name", errors="ignore"),
            on="drug_id",
            how="left",
        )

    base["evidence_mode"] = perturbation_mode
    base["global_query_replicated_across_ot_signatures"] = (
        perturbation_mode == "signature_level"
    )

    mechanistic_columns = [
        "expression_reversal_score",
        "signature_query_reversal_score",
        "direct_target_alignment_score",
        "network_reversal_score",
    ]
    for col in mechanistic_columns:
        if col not in base.columns:
            base[col] = np.nan
    base["has_mechanistic_evidence"] = base[mechanistic_columns].notna().any(axis=1)

    weight_key = (
        "component_weights_signature_level"
        if perturbation_mode == "signature_level"
        else "component_weights_gene_level"
    )
    component_weights = cfg["scoring"][weight_key]
    score_sum = np.zeros(len(base), dtype=float)
    weight_sum = np.zeros(len(base), dtype=float)

    for col, weight in component_weights.items():
        if col not in base.columns:
            base[col] = np.nan
        values = numeric(base[col]).to_numpy(float)
        valid = np.isfinite(values)
        score_sum[valid] += float(weight) * values[valid]
        weight_sum[valid] += float(weight)

    base["available_component_weight"] = weight_sum
    base["raw_integrated_score"] = np.divide(
        score_sum,
        weight_sum,
        out=np.full_like(score_sum, np.nan),
        where=weight_sum > 0,
    )
    base.loc[
        ~base["has_mechanistic_evidence"], "raw_integrated_score"
    ] = np.nan
    base["integrated_signature_score"] = base["raw_integrated_score"]

    if perturbation_mode == "gene_level":
        coverage = numeric(
            base.get("weighted_coverage", pd.Series(0.0, index=base.index)), 0.0
        ).clip(0.0, 1.0)
        floor = float(cfg["scoring"]["coverage_multiplier_floor"])
        base["coverage_multiplier"] = floor + (1.0 - floor) * coverage
        has_expression = base["expression_reversal_score"].notna()
        base.loc[has_expression, "integrated_signature_score"] *= base.loc[
            has_expression, "coverage_multiplier"
        ]

        min_genes = int(cfg["scoring"]["minimum_matched_genes"])
        min_coverage = float(cfg["scoring"]["minimum_weighted_coverage"])
        min_reversal = float(cfg["scoring"]["minimum_reversal_fraction"])
        max_conflict = float(cfg["scoring"]["maximum_conflict_fraction"])

        n_matched = numeric(
            base.get("n_matched_genes", pd.Series(0, index=base.index)), 0
        )
        weighted_coverage = numeric(
            base.get("weighted_coverage", pd.Series(0.0, index=base.index)), 0
        )
        reversal_fraction = numeric(
            base.get("reversal_fraction", pd.Series(0.0, index=base.index)), 0
        )
        conflict_fraction = numeric(
            base.get("conflict_fraction", pd.Series(1.0, index=base.index)), 1
        )

        quality_gate = base["expression_reversal_score"].isna() | (
            (n_matched >= min_genes)
            & (weighted_coverage >= min_coverage)
            & (reversal_fraction >= min_reversal)
            & (conflict_fraction <= max_conflict)
        )
        base["expression_quality_gate_pass"] = quality_gate
        base.loc[~quality_gate, "integrated_signature_score"] = np.minimum(
            base.loc[~quality_gate, "integrated_signature_score"].fillna(0),
            0,
        )
    else:
        base["coverage_multiplier"] = np.nan
        base["expression_quality_gate_pass"] = np.nan

    role_key = (
        base["signature_class"].astype(str)
        + "|"
        + base["analysis_role"].astype(str)
    )
    base["signature_role_weight"] = role_key.map(
        cfg["scoring"]["signature_role_weights"]
    ).fillna(0.5)
    return base


def normalize_pulmonary_risk_series(
    frame: pd.DataFrame,
    column: str = "pulmonary_risk",
) -> pd.Series:
    """Return a guaranteed string Series for safe keyword matching.

    This function is the direct fix for the reported float/NaN iteration error.
    It safely handles absent columns, NaN, None, numbers, booleans, and strings.
    """
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return (
        frame[column]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.casefold()
    )


def global_summary(
    long_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    perturbation_mode: str,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    positive_cut = float(cfg["scoring"]["positive_signature_score"])
    conflict_cut = float(cfg["scoring"]["strong_conflict_signature_score"])
    records: List[Dict[str, Any]] = []

    for drug_id, group in long_scores.groupby("drug_id", sort=False):
        scores = numeric(group["integrated_signature_score"])
        role_weights = numeric(group["signature_role_weight"], 0.5)
        valid = scores.notna()
        global_score = (
            float(np.average(scores[valid], weights=role_weights[valid]))
            if valid.any()
            else np.nan
        )
        primary = group["analysis_role"].eq("PRIMARY")

        row: Dict[str, Any] = {
            "drug_id": drug_id,
            "drug_name": (
                str(group["drug_name"].dropna().iloc[0])
                if group["drug_name"].notna().any()
                else str(drug_id)
            ),
            "evidence_mode": perturbation_mode,
            "global_drug_pairing_score": global_score,
            "n_signatures_evaluated": int(valid.sum()),
            "n_positive_signatures": int((scores >= positive_cut).sum()),
            "n_strong_conflict_signatures": int((scores <= conflict_cut).sum()),
            "n_primary_positive_signatures": int(
                ((scores >= positive_cut) & primary).sum()
            ),
            "best_signature_score": float(scores.max()) if valid.any() else np.nan,
            "worst_signature_score": float(scores.min()) if valid.any() else np.nan,
            "max_tier_a_genes_matched": int(
                max(
                    numeric(
                        group.get(
                            "n_tier_a_genes_matched",
                            pd.Series(0, index=group.index),
                        ),
                        0,
                    ).max(),
                    numeric(
                        group.get(
                            "n_direct_tier_a_targets",
                            pd.Series(0, index=group.index),
                        ),
                        0,
                    ).max(),
                )
            ),
            "best_signature_id": (
                str(group.loc[scores.idxmax(), "signature_id"])
                if valid.any()
                else ""
            ),
        }

        if "signature_query_reversal_score" in group.columns:
            non_missing = numeric(group["signature_query_reversal_score"]).dropna()
            row["signature_query_reversal_score"] = (
                float(non_missing.iloc[0]) if not non_missing.empty else np.nan
            )
        if "signature_query_qvalue" in group.columns:
            non_missing = numeric(group["signature_query_qvalue"]).dropna()
            row["signature_query_qvalue"] = (
                float(non_missing.iloc[0]) if not non_missing.empty else np.nan
            )

        for col in [
            "best_perturbation_signature_id",
            "best_query_context_id",
            "best_cell_line",
            "best_dose",
            "best_timepoint",
        ]:
            row[col] = (
                normalize_text(group[col].dropna().iloc[0])
                if col in group.columns and group[col].notna().any()
                else ""
            )

        records.append(row)

    out = pd.DataFrame(records)
    if not metadata.empty:
        out = out.merge(
            metadata.drop(columns="drug_name", errors="ignore"),
            on="drug_id",
            how="left",
        )

    # ---------------------------------------------------------------------
    # FIX: pulmonary_risk may contain NaN (float), None, numbers, or strings.
    # Normalize to a guaranteed StringDtype Series before ``term in value``.
    # ---------------------------------------------------------------------
    high_terms = [
        normalize_text(term).casefold()
        for term in cfg["filters"]["high_risk_terms"]
        if normalize_text(term)
    ]
    risk = normalize_pulmonary_risk_series(out)
    out["high_pulmonary_risk"] = (
        risk.map(
            lambda value: bool(value)
            and any(term in value for term in high_terms)
        )
        .fillna(False)
        .astype(bool)
    )

    if bool(cfg["filters"]["exclude_high_pulmonary_risk"]):
        out["hard_safety_exclusion"] = out["high_pulmonary_risk"].astype(bool)
    else:
        out["hard_safety_exclusion"] = False

    tiers = cfg["candidate_tiers"][perturbation_mode]

    def assign_candidate(row: pd.Series) -> Tuple[str, str]:
        if bool(row["hard_safety_exclusion"]):
            return "EXCLUDED", "High pulmonary-risk metadata"

        score = row["global_drug_pairing_score"]
        if pd.isna(score):
            return "EXCLUDED", "No usable score components"

        if perturbation_mode == "signature_level":
            query_score = row.get("signature_query_reversal_score", np.nan)
            qvalue = row.get("signature_query_qvalue", np.nan)
            q_ok_a = pd.isna(qvalue) or qvalue <= tiers["tier_a_max_qvalue"]
            q_ok_b = pd.isna(qvalue) or qvalue <= tiers["tier_b_max_qvalue"]

            if (
                score >= tiers["tier_a_min_global_score"]
                and pd.notna(query_score)
                and query_score >= tiers["tier_a_min_query_score"]
                and q_ok_a
            ):
                return (
                    "FINAL_CANDIDATE_A",
                    "Strong SigCom reversal with integrated support",
                )

            if (
                score >= tiers["tier_b_min_global_score"]
                and pd.notna(query_score)
                and query_score >= tiers["tier_b_min_query_score"]
                and q_ok_b
            ):
                return (
                    "FINAL_CANDIDATE_B",
                    "Moderate SigCom reversal with supporting evidence",
                )
        else:
            if (
                score >= tiers["tier_a_min_global_score"]
                and row["n_primary_positive_signatures"]
                >= tiers["tier_a_min_primary_positive_signatures"]
                and row["max_tier_a_genes_matched"]
                >= tiers["tier_a_min_tier_a_genes_matched"]
                and row["n_strong_conflict_signatures"] == 0
            ):
                return (
                    "FINAL_CANDIDATE_A",
                    "High integrated score with primary-signature support",
                )

            if (
                score >= tiers["tier_b_min_global_score"]
                and row["n_positive_signatures"]
                >= tiers["tier_b_min_positive_signatures"]
            ):
                return "FINAL_CANDIDATE_B", "Positive integrated evidence"

        if score > tiers["exploratory_min_global_score"]:
            return "EXPLORATORY", "Positive but below final thresholds"
        return "EXCLUDED", "Non-positive or conflicting evidence"

    assigned = out.apply(assign_candidate, axis=1, result_type="expand")
    out["candidate_tier"] = assigned[0]
    out["candidate_tier_reason"] = assigned[1]

    order = {
        "FINAL_CANDIDATE_A": 1,
        "FINAL_CANDIDATE_B": 2,
        "EXPLORATORY": 3,
        "EXCLUDED": 4,
    }
    out["candidate_tier_order"] = out["candidate_tier"].map(order)
    out = out.sort_values(
        ["candidate_tier_order", "global_drug_pairing_score", "drug_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    out["drug_global_rank"] = np.arange(1, len(out) + 1)
    return out


def write_empty_or_frame(
    frame: pd.DataFrame,
    path: Path,
    columns: Optional[List[str]] = None,
) -> None:
    if frame is None or frame.empty:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def write_methods(
    path: Path,
    perturbation_mode: str,
    qc: Mapping[str, Any],
) -> None:
    text = f"""DRUG PAIRING SCORE DEFINITION
=============================

Script version: {SCRIPT_VERSION}
Perturbation mode: {perturbation_mode}

Integrated compatibility fixes
------------------------------
1. Identifiers beginning with EXAMPLE_ are not automatically discarded. Some
   public pharmacology sources use such names for real patent compounds.
2. Missing target drug_id values use drug_concept_id, then NAME:<drug_name>.
3. pulmonary_risk values are normalized using fillna(''), StringDtype,
   strip(), and casefold() before substring matching. This safely handles NaN,
   None, numeric values, booleans, and strings.

Run counts
----------
OT signatures: {qc.get('n_signatures', 0)}
Target rows loaded: {qc.get('n_direct_target_rows', 0)}
Target EXAMPLE_ prefix rows retained: {qc.get('target_qc', {}).get('example_prefix_rows_kept', 0)}
Blank target IDs repaired: {qc.get('target_qc', {}).get('blank_drug_id_repaired', 0)}
Drugs evaluated: {qc.get('n_drugs_evaluated', 0)}

Interpretation
--------------
Computational prioritization only. Manual pharmacology, exposure, route,
cell-type relevance, and safety review are required.
"""
    path.write_text(text, encoding="utf-8")




# =============================================================================
# V2 OVERRIDES: GLOBAL QUERY IS SCORED ONCE; MECHANISTIC TRACKS ARE SEPARATE
# =============================================================================

# Additional PyCharm settings. The six input paths and OUTPUT_ROOT above remain
# the only paths that normally need editing.
NETWORK_DRUG_SCOPE = "query_and_direct_signature_targets"  # recommended | all_directional_targets
NETWORK_MASS_REFERENCE_QUANTILE = 0.75
PRIMARY_QUERY_WEIGHT = 0.70
PRIMARY_MECHANISM_WEIGHT = 0.20
PRIMARY_CONTEXT_WEIGHT = 0.05
PRIMARY_CLINICAL_WEIGHT = 0.05

# Strict final tiers remain statistically conservative. When the two-sided /
# two-arm strict q value fails, the script creates clearly labelled priority
# reverser outputs rather than silently relaxing the q-value threshold.
V2_TIERS = {
    "final_a_min_score": 0.65,
    "final_a_max_strict_q": 0.05,
    "final_a_min_mechanism": 0.10,
    "final_b_min_score": 0.55,
    "final_b_max_strict_q": 0.25,
    "priority_min_score": 0.50,
    "priority_max_best_arm_q": 0.05,
    "priority_mechanism_min": 0.08,
    "positive_mechanism_cut": 0.10,
    "negative_mechanism_cut": -0.10,
}


def normalize_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return normalize_text(value).casefold() in {"1", "true", "yes", "y", "t"}


def load_signature_level_perturbations(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Standardise the merged/global SigCom result table.

    The function keeps strict two-arm and best-arm statistics separate. When
    explicit arm FDR columns are available, missing strict/best-arm columns are
    derived as max(FDR_up, FDR_down) and min(FDR_up, FDR_down), respectively.
    Rows explicitly marked as non-reversers are excluded when
    SOURCE_REVERSERS_ONLY=True.
    """
    if df.empty:
        return pd.DataFrame()

    lower = {str(c).casefold(): c for c in df.columns}
    drug_id_col = first_existing(lower, ["drug_id", "perturbagen_id", "compound_id"])
    if drug_id_col is None:
        raise KeyError("Signature-level perturbation table needs drug_id.")
    reject_true_template_rows(df, drug_id_col, ALLOW_TRUE_TEMPLATE_ROWS)

    drug_name_col = first_existing(lower, ["drug_name", "perturbagen_name", "compound_name"])
    score_col = first_existing(lower, ["query_reversal_score", "reversal_score"])
    z_sum_col = first_existing(lower, ["z_score_sum", "zscore_sum"])
    if score_col is None and z_sum_col is None:
        raise KeyError("Signature-level table needs query_reversal_score or z_score_sum.")

    q_strict_col = first_existing(lower, ["qvalue_strict", "strict_qvalue", "qvalue", "q_value", "fdr"])
    q_best_col = first_existing(lower, ["qvalue_best_arm", "best_arm_qvalue"])
    fdr_up_col = first_existing(lower, ["fdr_up", "qvalue_up"])
    fdr_down_col = first_existing(lower, ["fdr_down", "qvalue_down"])
    if q_strict_col is None and not (fdr_up_col and fdr_down_col):
        raise KeyError(
            "Final 37a requires qvalue_strict/qvalue, or both fdr_up and fdr_down."
        )

    z_up_col = first_existing(lower, ["z_score_up", "zscore_up"])
    z_down_col = first_existing(lower, ["z_score_down", "zscore_down"])
    p_up_col = first_existing(lower, ["p_value_up", "pvalue_up"])
    p_down_col = first_existing(lower, ["p_value_down", "pvalue_down"])
    weight_col = first_existing(lower, ["evidence_weight", "weight"])
    sig_id_col = first_existing(lower, ["perturbation_signature_id", "signature_id", "sig_id"])
    context_col = first_existing(lower, ["context_id", "context"])
    cell_col = first_existing(lower, ["cell_line", "cell_id", "cell"])
    tissue_col = first_existing(lower, ["tissue"])
    disease_col = first_existing(lower, ["disease"])
    dose_col = first_existing(lower, ["dose", "pert_dose"])
    time_col = first_existing(lower, ["timepoint", "time", "pert_time"])
    source_col = first_existing(lower, ["source", "database"])
    is_reverser_col = first_existing(lower, ["is_reverser"])

    out = pd.DataFrame(index=df.index)
    out["source_row_number"] = np.arange(2, len(df) + 2)
    out["drug_id"] = df[drug_id_col].map(normalize_text)
    out["drug_name"] = df[drug_name_col].map(normalize_text) if drug_name_col else out["drug_id"]

    raw_score = numeric(df[score_col]) if score_col else -numeric(df[z_sum_col])
    out["query_reversal_score_raw"] = raw_score
    out["query_reversal_score_normalized"] = robust_signed_score(raw_score)

    fdr_up = numeric(df[fdr_up_col], 1.0).clip(EPS, 1.0) if fdr_up_col else pd.Series(np.nan, index=df.index)
    fdr_down = numeric(df[fdr_down_col], 1.0).clip(EPS, 1.0) if fdr_down_col else pd.Series(np.nan, index=df.index)
    out["fdr_up"] = fdr_up
    out["fdr_down"] = fdr_down

    if q_strict_col:
        out["qvalue_strict"] = numeric(df[q_strict_col], 1.0).clip(EPS, 1.0)
        strict_provenance = f"column:{q_strict_col}"
    else:
        out["qvalue_strict"] = pd.concat([fdr_up, fdr_down], axis=1).max(axis=1).clip(EPS, 1.0)
        strict_provenance = "derived:max(fdr_up,fdr_down)"

    if q_best_col:
        out["qvalue_best_arm"] = numeric(df[q_best_col], 1.0).clip(EPS, 1.0)
        best_provenance = f"column:{q_best_col}"
    elif fdr_up_col and fdr_down_col:
        out["qvalue_best_arm"] = pd.concat([fdr_up, fdr_down], axis=1).min(axis=1).clip(EPS, 1.0)
        best_provenance = "derived:min(fdr_up,fdr_down)"
    else:
        out["qvalue_best_arm"] = out["qvalue_strict"]
        best_provenance = "fallback:qvalue_strict"

    out["qvalue_strict_provenance"] = strict_provenance
    out["qvalue_best_arm_provenance"] = best_provenance
    out["z_score_up"] = numeric(df[z_up_col]) if z_up_col else np.nan
    out["z_score_down"] = numeric(df[z_down_col]) if z_down_col else np.nan
    out["p_value_up"] = numeric(df[p_up_col], 1.0).clip(EPS, 1.0) if p_up_col else np.nan
    out["p_value_down"] = numeric(df[p_down_col], 1.0).clip(EPS, 1.0) if p_down_col else np.nan
    out["evidence_weight"] = numeric(df[weight_col], 1.0).clip(lower=0.0) if weight_col else 1.0
    out["is_reverser_source"] = (
        df[is_reverser_col].map(normalize_bool)
        if is_reverser_col
        else out["query_reversal_score_normalized"].gt(0)
    )
    out["perturbation_signature_id"] = df[sig_id_col].map(normalize_text) if sig_id_col else ""
    out["context_id"] = df[context_col].map(normalize_text) if context_col else ""
    out["cell_line"] = df[cell_col].map(normalize_text) if cell_col else ""
    out["tissue"] = df[tissue_col].map(normalize_text) if tissue_col else ""
    out["disease"] = df[disease_col].map(normalize_text) if disease_col else ""
    out["dose"] = df[dose_col].map(normalize_text) if dose_col else ""
    out["timepoint"] = df[time_col].map(normalize_text) if time_col else ""
    out["source"] = df[source_col].map(normalize_text) if source_col else "SigCom"

    valid = out["drug_id"].ne("") & np.isfinite(out["query_reversal_score_normalized"])
    out = out.loc[valid].copy()
    if SOURCE_REVERSERS_ONLY:
        out = out.loc[out["is_reverser_source"]].copy()
    out = out[
        (out["qvalue_strict"] <= float(cfg["filters"]["maximum_qvalue"]))
        & (out["evidence_weight"] >= float(cfg["filters"]["minimum_perturbation_evidence_weight"]))
    ].copy()

    q_weight = (-np.log10(out["qvalue_best_arm"].clip(lower=EPS))).clip(
        upper=float(cfg["effect"]["max_qvalue_weight"])
    )
    out["combined_weight"] = out["evidence_weight"] * (1.0 + q_weight)
    return out.reset_index(drop=True)


def aggregate_signature_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated SigCom contexts without mixing statistics across rows.

    The weighted mean score uses all retained contexts. Candidate-tier q-values,
    z-scores, dose, time and cell context are taken from one internally coherent
    best-context row, rather than taking a minimum q-value from a different row.
    Across-context minima are retained in separate audit columns.
    """
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name), group in df.groupby(["drug_id", "drug_name"], dropna=False):
        group = group.copy()
        scores = numeric(group["query_reversal_score_normalized"]).to_numpy(float)
        weights = numeric(group["combined_weight"], 1.0).to_numpy(float)
        valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue

        weighted_mean = float(np.average(scores[valid], weights=weights[valid]))
        sign_consistency = float(
            np.sum(weights[valid] * (np.sign(scores[valid]) == np.sign(weighted_mean)))
            / max(np.sum(weights[valid]), EPS)
        )

        group["_score_sort"] = numeric(group["query_reversal_score_normalized"], -np.inf)
        group["_best_q_sort"] = numeric(group["qvalue_best_arm"], 1.0)
        group["_strict_q_sort"] = numeric(group["qvalue_strict"], 1.0)
        group["_weight_sort"] = numeric(group["combined_weight"], 0.0)
        best = group.sort_values(
            ["_score_sort", "_best_q_sort", "_strict_q_sort", "_weight_sort", "source_row_number"],
            ascending=[False, True, True, False, True],
        ).iloc[0]

        zu = abs(float(best["z_score_up"])) if pd.notna(best["z_score_up"]) else np.nan
        zd = abs(float(best["z_score_down"])) if pd.notna(best["z_score_down"]) else np.nan
        arm_balance = (
            min(zu, zd) / max(zu, zd, EPS)
            if np.isfinite(zu) and np.isfinite(zd)
            else np.nan
        )

        records.append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "signature_query_reversal_score": weighted_mean,
            "signature_query_best_score": float(best["query_reversal_score_normalized"]),
            "signature_query_score_raw": float(best["query_reversal_score_raw"]),
            "signature_query_qvalue_strict": float(best["qvalue_strict"]),
            "signature_query_qvalue_best_arm": float(best["qvalue_best_arm"]),
            "minimum_qvalue_strict_across_contexts": float(numeric(group["qvalue_strict"], 1.0).min()),
            "minimum_qvalue_best_arm_across_contexts": float(numeric(group["qvalue_best_arm"], 1.0).min()),
            "qvalue_strict_provenance": normalize_text(best["qvalue_strict_provenance"]),
            "qvalue_best_arm_provenance": normalize_text(best["qvalue_best_arm_provenance"]),
            "best_fdr_up": float(best["fdr_up"]) if pd.notna(best["fdr_up"]) else np.nan,
            "best_fdr_down": float(best["fdr_down"]) if pd.notna(best["fdr_down"]) else np.nan,
            "best_z_score_up": float(best["z_score_up"]) if pd.notna(best["z_score_up"]) else np.nan,
            "best_z_score_down": float(best["z_score_down"]) if pd.notna(best["z_score_down"]) else np.nan,
            "query_arm_balance": arm_balance,
            "query_context_sign_consistency": sign_consistency,
            "query_context_consistency_warning": sign_consistency < MIN_QUERY_CONTEXT_SIGN_CONSISTENCY,
            "strict_two_arm_significant_005": bool(float(best["qvalue_strict"]) <= 0.05),
            "strict_two_arm_significant_025": bool(float(best["qvalue_strict"]) <= 0.25),
            "best_arm_significant_005": bool(float(best["qvalue_best_arm"]) <= 0.05),
            "n_strict_contexts_005": int((numeric(group["qvalue_strict"], 1.0) <= 0.05).sum()),
            "n_best_arm_contexts_005": int((numeric(group["qvalue_best_arm"], 1.0) <= 0.05).sum()),
            "n_query_context_rows": int(len(group)),
            "n_query_contexts": int(group.loc[group["context_id"].ne(""), "context_id"].nunique()),
            "best_source_row_number": int(best["source_row_number"]),
            "best_perturbation_signature_id": normalize_text(best["perturbation_signature_id"]),
            "best_query_context_id": normalize_text(best["context_id"]),
            "best_cell_line": normalize_text(best["cell_line"]),
            "best_tissue": normalize_text(best["tissue"]),
            "best_disease": normalize_text(best["disease"]),
            "best_dose": normalize_text(best["dose"]),
            "best_timepoint": normalize_text(best["timepoint"]),
            "query_sources": ";".join(sorted(set(group["source"].astype(str)))),
        })
    return pd.DataFrame(records)


def load_metadata(path: Optional[Path]) -> pd.DataFrame:
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()
    lower = {str(c).casefold(): c for c in df.columns}
    id_col = lower.get("drug_id")
    if id_col is None:
        raise KeyError("Drug metadata needs drug_id.")
    name_col = lower.get("drug_name")
    out = pd.DataFrame({"drug_id": df[id_col].map(normalize_text)})
    out["drug_name"] = df[name_col].map(normalize_text) if name_col else out["drug_id"]
    text_cols = [
        "approval_status", "pulmonary_risk", "notes", "entity_quality_categories",
        "hard_block_reasons", "drug_family_group",
    ]
    for col in text_cols:
        src = lower.get(col)
        out[col] = df[src].map(normalize_text) if src else ""
    for col in ["clinical_score", "safety_score", "availability_score"]:
        src = lower.get(col)
        out[col] = numeric(df[src]).clip(0.0, 1.0) if src else np.nan
    hard_src = lower.get("hard_block_main_table")
    out["source_hard_block"] = df[hard_src].map(normalize_bool) if hard_src else False

    approval_map = {
        "approved": 1.0, "phase 4": 0.95, "phase 3": 0.85,
        "phase 2": 0.70, "phase 1": 0.55, "clinical": 0.65,
        "preclinical": 0.35, "experimental": 0.20,
    }
    missing = out["clinical_score"].isna()
    if missing.any():
        inferred = out.loc[missing, "approval_status"].str.casefold().map(
            lambda x: max([value for key, value in approval_map.items() if key in x], default=np.nan)
        )
        out.loc[missing, "clinical_score"] = inferred
    out["translation_metadata_complete"] = out[["clinical_score", "safety_score", "availability_score"]].notna().all(axis=1)
    return out[out["drug_id"].ne("")].drop_duplicates("drug_id", keep="first")



def score_direct_targets(
    signatures: pd.DataFrame,
    targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score direct targets with coverage/evidence calibration.

    The legacy ratio could equal +1 from a single low-coverage edge because the
    evidence weight cancelled between numerator and denominator. V2 retains the
    raw direction ratio but multiplies it by signature-gene coverage and mean
    evidence quality.
    """
    if targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    directional = targets[targets["action_sign"].ne(0)].copy()
    if directional.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = signatures.merge(directional, on="gene_symbol", how="inner", suffixes=("", "_target"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["target_alignment"] = merged["desired_sign"] * merged["action_sign"]
    merged["edge_mass"] = merged["gene_weight_normalized"] * numeric(merged["evidence_weight"], 1.0).clip(0.0, 1.0)
    merged["target_weighted_alignment"] = merged["edge_mass"] * merged["target_alignment"]
    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name, signature_id), group in merged.groupby(["drug_id", "drug_name", "signature_id"], dropna=False):
        # Avoid counting multiple databases/actions for the same gene as extra signature coverage.
        gene_rows = []
        for gene, gg in group.groupby("gene_symbol", sort=False):
            edge_mass = numeric(gg["edge_mass"], 0.0)
            best_idx = edge_mass.idxmax()
            gene_rows.append(gg.loc[best_idx])
        unique = pd.DataFrame(gene_rows)
        denominator = float(numeric(unique["edge_mass"], 0.0).sum())
        raw = float(numeric(unique["target_weighted_alignment"], 0.0).sum() / max(denominator, EPS))
        coverage = float(numeric(unique["gene_weight_normalized"], 0.0).sum())
        mean_quality = float(np.average(
            numeric(unique["evidence_weight"], 0.0).clip(0.0, 1.0),
            weights=np.maximum(numeric(unique["gene_weight_normalized"], 0.0), EPS),
        ))
        reliability = float(np.sqrt(np.clip(coverage, 0.0, 1.0)) * np.sqrt(np.clip(mean_quality, 0.0, 1.0)))
        records.append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "signature_id": signature_id,
            "direct_target_alignment_score": raw,
            "direct_target_gene_weight_coverage": coverage,
            "direct_target_mean_evidence_quality": mean_quality,
            "direct_target_reliability": reliability,
            "direct_target_effective_score": raw * reliability,
            "n_direct_signature_targets": int(unique["gene_symbol"].nunique()),
            "n_direct_tier_a_targets": int(unique.loc[unique["ppi_driver_tier"].eq("TIER_A_HIGH_CONFIDENCE"), "gene_symbol"].nunique()),
            "direct_signature_targets": ";".join(sorted(unique["gene_symbol"].unique())),
        })
    return pd.DataFrame(records), merged

def calibrate_network_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Dampen direction-only network scores by empirical influence mass.

    Raw propagation scores are ratios and can equal +/-1 even when only a tiny
    amount of propagated mass reaches the OT signature. Reliability is
    therefore mass / (mass + signature-specific reference mass), where the
    reference is the configured positive-mass quantile.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["network_mass_reference"] = np.nan
    out["network_mass_reliability"] = 0.0
    out["network_mass_percentile"] = 0.0
    for signature_id, idx in out.groupby("signature_id").groups.items():
        mass = numeric(out.loc[idx, "network_signature_influence_mass"], 0.0).clip(lower=0.0)
        positive = mass[mass > 0]
        if positive.empty:
            reference = EPS
        else:
            reference = float(positive.quantile(NETWORK_MASS_REFERENCE_QUANTILE))
            reference = max(reference, EPS)
        reliability = mass / (mass + reference)
        percentile = mass.rank(method="average", pct=True).fillna(0.0)
        out.loc[idx, "network_mass_reference"] = reference
        out.loc[idx, "network_mass_reliability"] = reliability
        out.loc[idx, "network_mass_percentile"] = percentile
    out["network_effective_score"] = (
        numeric(out["network_reversal_score"]) * numeric(out["network_mass_reliability"], 0.0)
    )
    return out


def context_relevance_score(row: pd.Series) -> float:
    text = " ".join([
        normalize_text(row.get("best_tissue", "")),
        normalize_text(row.get("best_cell_line", "")),
        normalize_text(row.get("best_disease", "")),
    ]).casefold()
    if any(term in text for term in ["lung", "pulmonary", "bronch", "alveolar", "thp1", "thp-1", "monocyte", "macrophage"]):
        return 1.0
    if any(term in text for term in ["blood", "immune", "hematopo", "lymph", "myeloid"]):
        return 0.80
    if text.strip():
        return 0.50
    return 0.35


def build_mechanistic_long(
    signatures: pd.DataFrame,
    query_drugs: pd.DataFrame,
    direct_scores: pd.DataFrame,
    network_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    registry_cols = [
        "signature_id", "signature_name", "signature_class", "analysis_role",
        "tier_mode", "disease_scope", "celltype_scope",
    ]
    registry = signatures[registry_cols].drop_duplicates()
    frames = []
    for frame in [query_drugs, direct_scores, network_scores_df]:
        if frame is not None and not frame.empty:
            frames.append(frame[["drug_id", "drug_name"]].drop_duplicates())
    drugs = pd.concat(frames, ignore_index=True).drop_duplicates("drug_id", keep="first")
    base = drugs.assign(_k=1).merge(registry.assign(_k=1), on="_k").drop(columns="_k")
    if not direct_scores.empty:
        base = base.merge(direct_scores.drop(columns="drug_name", errors="ignore"), on=["drug_id", "signature_id"], how="left")
    if not network_scores_df.empty:
        base = base.merge(network_scores_df.drop(columns="drug_name", errors="ignore"), on=["drug_id", "signature_id"], how="left")
    for col in ["direct_target_alignment_score", "direct_target_effective_score", "direct_target_gene_weight_coverage", "direct_target_reliability", "network_reversal_score", "network_effective_score", "network_signature_influence_mass", "network_mass_reliability", "network_mass_percentile"]:
        if col not in base.columns:
            base[col] = np.nan
    base["has_direct_signature_target"] = base["direct_target_effective_score"].notna()
    base["has_network_evidence"] = base["network_effective_score"].notna()
    # Fixed weights: missing evidence contributes zero; it is never renormalized to 100%.
    base["mechanistic_signature_score"] = (
        0.70 * numeric(base["direct_target_effective_score"], 0.0).clip(-1.0, 1.0)
        + 0.30 * numeric(base["network_effective_score"], 0.0).clip(-1.0, 1.0)
    )
    role_key = base["signature_class"].astype(str) + "|" + base["analysis_role"].astype(str)
    base["signature_role_weight"] = role_key.map(DEFAULT_CONFIG["scoring"]["signature_role_weights"]).fillna(0.5)
    base["global_query_counted_in_signature_score"] = False
    return base


def summarize_mechanistic(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for drug_id, group in long_df.groupby("drug_id", sort=False):
        primary = group["analysis_role"].eq("PRIMARY")
        g = group.loc[primary].copy()
        if g.empty:
            g = group.copy()
        scores = numeric(g["mechanistic_signature_score"], 0.0)
        weights = numeric(g["signature_role_weight"], 0.5)
        positive = scores >= V2_TIERS["positive_mechanism_cut"]
        negative = scores <= V2_TIERS["negative_mechanism_cut"]
        direct_tier_a = numeric(g.get("n_direct_tier_a_targets", pd.Series(0, index=g.index)), 0)
        row = {
            "drug_id": drug_id,
            "drug_name": normalize_text(group["drug_name"].iloc[0]),
            "mechanistic_support_score": float(max(scores.max(), 0.0)),
            "mean_primary_mechanistic_score": float(np.average(scores, weights=weights)) if len(scores) else 0.0,
            "best_primary_mechanistic_score": float(scores.max()) if len(scores) else 0.0,
            "worst_primary_mechanistic_score": float(scores.min()) if len(scores) else 0.0,
            "n_primary_mechanistic_positive": int(positive.sum()),
            "n_primary_mechanistic_conflict": int(negative.sum()),
            "n_primary_direct_supported": int(g["has_direct_signature_target"].sum()),
            "n_primary_network_supported": int(g["has_network_evidence"].sum()),
            "max_direct_tier_a_targets": int(direct_tier_a.max()) if len(direct_tier_a) else 0,
            "best_mechanistic_signature_id": str(g.loc[scores.idxmax(), "signature_id"]) if len(scores) else "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_safety_flags(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    risk = normalize_pulmonary_risk_series(out)
    high_terms = [normalize_text(t).casefold() for t in cfg["filters"]["high_risk_terms"] if normalize_text(t)]
    out["high_pulmonary_risk"] = risk.map(lambda v: bool(v) and any(t in v for t in high_terms)).fillna(False).astype(bool)
    if "source_hard_block" not in out.columns:
        out["source_hard_block"] = False
    out["source_hard_block"] = out["source_hard_block"].map(normalize_bool).astype(bool)
    out["hard_safety_or_entity_exclusion"] = out["source_hard_block"] | (
        out["high_pulmonary_risk"] if bool(cfg["filters"]["exclude_high_pulmonary_risk"]) else False
    )
    return out


def build_primary_query_summary(
    query: pd.DataFrame,
    mechanism: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    out = query.copy()
    if not mechanism.empty:
        out = out.merge(mechanism.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    if not metadata.empty:
        out = out.merge(metadata.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    for col, default in {
        "mechanistic_support_score": 0.0,
        "n_primary_mechanistic_positive": 0,
        "n_primary_mechanistic_conflict": 0,
        "max_direct_tier_a_targets": 0,
    }.items():
        if col not in out.columns:
            out[col] = default
        out[col] = numeric(out[col], default)
    out["context_relevance_score"] = out.apply(context_relevance_score, axis=1)
    out["clinical_component"] = numeric(out.get("clinical_score", pd.Series(np.nan, index=out.index))).fillna(0.0)
    out["primary_query_global_score"] = (
        PRIMARY_QUERY_WEIGHT * numeric(out["signature_query_reversal_score"], 0.0).clip(-1.0, 1.0)
        + PRIMARY_MECHANISM_WEIGHT * numeric(out["mechanistic_support_score"], 0.0).clip(0.0, 1.0)
        + PRIMARY_CONTEXT_WEIGHT * numeric(out["context_relevance_score"], 0.0).clip(0.0, 1.0)
        + PRIMARY_CLINICAL_WEIGHT * numeric(out["clinical_component"], 0.0).clip(0.0, 1.0)
    )
    out = add_safety_flags(out, cfg)

    def support_class(r: pd.Series) -> str:
        if r["signature_query_qvalue_strict"] <= 0.05:
            return "STRICT_TWO_ARM_FDR_0.05"
        if r["signature_query_qvalue_strict"] <= 0.25:
            return "STRICT_TWO_ARM_FDR_0.25"
        if r["signature_query_qvalue_best_arm"] <= 0.05:
            return "ONE_ARM_SUPPORTED_ONLY"
        return "SCORE_ONLY"
    out["query_statistical_support_class"] = out.apply(support_class, axis=1)

    def tier(r: pd.Series) -> Tuple[str, str]:
        if bool(r["hard_safety_or_entity_exclusion"]):
            return "EXCLUDED", normalize_text(r.get("hard_block_reasons", "")) or "Safety/entity hard block"
        score = float(r["primary_query_global_score"])
        strict_q = float(r["signature_query_qvalue_strict"])
        best_q = float(r["signature_query_qvalue_best_arm"])
        mech = float(r["mechanistic_support_score"])
        if score >= V2_TIERS["final_a_min_score"] and strict_q <= V2_TIERS["final_a_max_strict_q"] and mech >= V2_TIERS["final_a_min_mechanism"]:
            return "FINAL_CANDIDATE_A", "Strict two-arm SigCom support plus OT-linked mechanism"
        if score >= V2_TIERS["final_b_min_score"] and strict_q <= V2_TIERS["final_b_max_strict_q"]:
            return "FINAL_CANDIDATE_B", "Moderate strict two-arm SigCom support"
        if score >= V2_TIERS["priority_min_score"] and best_q <= V2_TIERS["priority_max_best_arm_q"]:
            if float(r.get("n_primary_direct_supported", 0)) > 0 and mech >= V2_TIERS["priority_mechanism_min"]:
                return "PRIORITY_REVERSER_DIRECT_SUPPORTED", "Strong one-arm SigCom evidence; strict two-arm FDR failed; direct OT-target support present"
            if float(r.get("n_primary_network_supported", 0)) > 0 and mech >= V2_TIERS["priority_mechanism_min"]:
                return "PRIORITY_REVERSER_NETWORK_SUPPORTED", "Strong one-arm SigCom evidence; strict two-arm FDR failed; calibrated network support present"
            return "PRIORITY_REVERSER_QUERY_ONLY", "Strong one-arm SigCom evidence; strict two-arm FDR failed; no mapped OT mechanism"
        if score > 0:
            return "EXPLORATORY", "Positive global-query score below priority criteria"
        return "EXCLUDED", "Non-positive score"

    assigned = out.apply(tier, axis=1, result_type="expand")
    out["candidate_tier"] = assigned[0]
    out["candidate_tier_reason"] = assigned[1]
    order = {
        "FINAL_CANDIDATE_A": 1,
        "FINAL_CANDIDATE_B": 2,
        "PRIORITY_REVERSER_DIRECT_SUPPORTED": 3,
        "PRIORITY_REVERSER_NETWORK_SUPPORTED": 4,
        "PRIORITY_REVERSER_QUERY_ONLY": 5,
        "EXPLORATORY": 6,
        "EXCLUDED": 7,
    }
    out["candidate_tier_order"] = out["candidate_tier"].map(order)
    out = out.sort_values(["candidate_tier_order", "primary_query_global_score", "signature_query_qvalue_strict", "drug_name"], ascending=[True, False, True, True]).reset_index(drop=True)
    out["drug_global_rank"] = np.arange(1, len(out) + 1)
    out["screening_track"] = "PRIMARY_SIGCOM_GLOBAL_QUERY"
    return out


def build_target_mechanism_summary(mechanistic_long: pd.DataFrame, metadata: pd.DataFrame, query_ids: set[str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    if mechanistic_long.empty:
        return pd.DataFrame()
    mech = summarize_mechanistic(mechanistic_long)
    direct_info = mechanistic_long.groupby("drug_id", as_index=False).agg(
        n_signatures_with_direct_target=("has_direct_signature_target", "sum"),
        n_signatures_with_network=("has_network_evidence", "sum"),
        max_network_mass_reliability=("network_mass_reliability", "max"),
        max_network_effective_score=("network_effective_score", "max"),
    )
    out = mech.merge(direct_info, on="drug_id", how="left")
    out = out[numeric(out["n_signatures_with_direct_target"], 0) > 0].copy()
    if not metadata.empty:
        out = out.merge(metadata.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    out["present_in_sigcom_query"] = out["drug_id"].isin(query_ids)
    out = add_safety_flags(out, cfg)
    out["target_mechanism_score"] = (
        0.80 * numeric(out["mechanistic_support_score"], 0.0).clip(0.0, 1.0)
        + 0.10 * (numeric(out["max_direct_tier_a_targets"], 0.0) > 0).astype(float)
        + 0.10 * numeric(out.get("clinical_score", pd.Series(np.nan, index=out.index))).fillna(0.0).clip(0.0, 1.0)
    )
    out["mechanism_candidate_class"] = np.select(
        [
            out["hard_safety_or_entity_exclusion"],
            (out["mechanistic_support_score"] >= 0.25) & (out["max_direct_tier_a_targets"] >= 1),
            out["mechanistic_support_score"] >= 0.10,
        ],
        ["EXCLUDED", "TARGET_MECHANISM_PRIORITY", "TARGET_MECHANISM_EXPLORATORY"],
        default="LOW_OR_CONFLICTING_MECHANISM",
    )
    mechanism_order = {
        "TARGET_MECHANISM_PRIORITY": 1,
        "TARGET_MECHANISM_EXPLORATORY": 2,
        "LOW_OR_CONFLICTING_MECHANISM": 3,
        "EXCLUDED": 4,
    }
    out["mechanism_candidate_order"] = out["mechanism_candidate_class"].map(mechanism_order).fillna(9)
    out = out.sort_values(["mechanism_candidate_order", "target_mechanism_score", "drug_name"], ascending=[True, False, True]).reset_index(drop=True)
    out["target_mechanism_rank"] = np.arange(1, len(out) + 1)
    out["screening_track"] = "TARGET_MECHANISM_DISCOVERY"
    return out




# =============================================================================
# V3 OVERRIDES: SOURCE-AWARE TARGET QC, DUAL-TRACK ROBUSTNESS, ONE-ARM TIERS
# =============================================================================

# -----------------------------------------------------------------------------
# Additional PyCharm settings. Normally only the paths at the top of this file
# need editing. These settings control the new source-sensitivity workflow.
# -----------------------------------------------------------------------------
SUSPICIOUS_TARGET_SOURCES = {"guidetopharmacology"}
MAIN_EXCLUDE_SUSPICIOUS_ONLY_TARGET_ROWS = True
SENSITIVITY_SUSPICIOUS_ONLY_MULTIPLIER = 0.10
SENSITIVITY_MIXED_SOURCE_MULTIPLIER = 0.70
MIN_TARGET_DIRECTION_CONSISTENCY = 0.60

# Confirmed hard blocks remain exclusions. Heuristic name flags are changed to
# manual-review flags so they cannot automatically remove real drugs.
CONFIRMED_HARD_BLOCK_REASONS = {
    "experimental_compound",
    "non_drug_entity",
}
HEURISTIC_MANUAL_REVIEW_REASONS = {
    "fluoroquinolone_like",
    "broad_cytotoxic_or_antineoplastic_name",
    "low_translational_quality_entity",
    "generic_or_vague_entity",
}

# Main ranking weights. Missing metadata contributes zero and is explicitly
# reported as missing; it is never interpreted as evidence of safety.
V3_PRIMARY_QUERY_WEIGHT = 0.75
V3_PRIMARY_MECHANISM_WEIGHT = 0.15
V3_PRIMARY_CONTEXT_WEIGHT = 0.05
V3_PRIMARY_CLINICAL_WEIGHT = 0.05

V3_TIERS = {
    "final_a_min_score": 0.65,
    "final_a_max_strict_q": 0.05,
    "final_a_min_mechanism": 0.10,
    "final_a_require_translation_complete": True,
    "final_b_min_score": 0.55,
    "final_b_max_strict_q": 0.25,
    "one_arm_a_min_score": 0.60,
    "one_arm_a_min_query_score": 0.70,
    "one_arm_a_max_best_q": 0.01,
    "one_arm_a_min_arm_balance": 0.20,
    "one_arm_b_min_score": 0.55,
    "one_arm_b_min_query_score": 0.65,
    "one_arm_b_max_best_q": 0.05,
    "one_arm_b_min_arm_balance": 0.15,
    "mechanism_positive_cut": 0.08,
    "mechanism_sensitivity_positive_cut": 0.05,
    "mechanism_conflict_cut": -0.08,
}


def _source_tokens(value: Any) -> List[str]:
    """Split a multi-source field into normalized exact source tokens."""
    text = normalize_text(value)
    if not text:
        return ["unspecified"]
    tokens = [t.strip().casefold() for t in re.split(r"[;,|]+", text) if t.strip()]
    return sorted(set(tokens)) or ["unspecified"]


def annotate_target_source_policy(targets: pd.DataFrame) -> pd.DataFrame:
    """Annotate raw target rows for main and sensitivity source policies."""
    if targets.empty:
        return targets.copy()
    out = targets.copy()
    token_lists = out["source"].map(_source_tokens)
    out["source_tokens_normalized"] = token_lists.map(lambda xs: ";".join(xs))
    out["n_source_tokens"] = token_lists.map(len).astype(int)
    out["n_suspicious_source_tokens"] = token_lists.map(
        lambda xs: sum(t in SUSPICIOUS_TARGET_SOURCES for t in xs)
    ).astype(int)
    out["n_non_suspicious_source_tokens"] = (
        out["n_source_tokens"] - out["n_suspicious_source_tokens"]
    ).astype(int)
    out["has_suspicious_source"] = out["n_suspicious_source_tokens"].gt(0)
    out["suspicious_source_only"] = (
        out["has_suspicious_source"] & out["n_non_suspicious_source_tokens"].eq(0)
    )
    out["mixed_suspicious_source"] = (
        out["has_suspicious_source"] & out["n_non_suspicious_source_tokens"].gt(0)
    )
    out["main_source_multiplier"] = np.where(
        out["suspicious_source_only"] & MAIN_EXCLUDE_SUSPICIOUS_ONLY_TARGET_ROWS,
        0.0,
        1.0,
    )
    out["sensitivity_source_multiplier"] = np.select(
        [out["suspicious_source_only"], out["mixed_suspicious_source"]],
        [SENSITIVITY_SUSPICIOUS_ONLY_MULTIPLIER, SENSITIVITY_MIXED_SOURCE_MULTIPLIER],
        default=1.0,
    ).astype(float)
    return out


def consolidate_target_track(
    annotated: pd.DataFrame,
    track: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one signed consensus action per drug-gene pair.

    Conflicting activation and inhibition evidence is combined by source-weighted
    vote. Low-consistency pairs are retained in a conflict table but excluded
    from direct and network scoring.
    """
    if annotated.empty:
        return pd.DataFrame(), pd.DataFrame()
    if track not in {"main", "sensitivity"}:
        raise ValueError("track must be 'main' or 'sensitivity'")
    multiplier_col = f"{track}_source_multiplier"
    work = annotated.copy()
    work["source_adjusted_weight"] = (
        numeric(work["evidence_weight"], 0.0).clip(0.0, 1.0)
        * numeric(work[multiplier_col], 0.0).clip(0.0, 1.0)
    )
    work = work[work["action_sign"].ne(0) & work["source_adjusted_weight"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for (drug_id, gene_symbol), group in work.groupby(["drug_id", "gene_symbol"], sort=False):
        positive = float(group.loc[group["action_sign"].gt(0), "source_adjusted_weight"].sum())
        negative = float(group.loc[group["action_sign"].lt(0), "source_adjusted_weight"].sum())
        total = positive + negative
        net = positive - negative
        consistency = abs(net) / max(total, EPS)
        consensus_sign = float(np.sign(net)) if consistency >= MIN_TARGET_DIRECTION_CONSISTENCY else 0.0
        winning_weight = max(positive, negative)
        compressed_quality = float(1.0 - math.exp(-winning_weight))
        source_tokens = sorted({
            token
            for value in group["source_tokens_normalized"]
            for token in normalize_text(value).split(";")
            if token
        })
        non_suspicious_tokens = [t for t in source_tokens if t not in SUSPICIOUS_TARGET_SOURCES]
        suspicious_tokens = [t for t in source_tokens if t in SUSPICIOUS_TARGET_SOURCES]
        drug_name_values = group["drug_name"].map(normalize_text)
        drug_name = next((x for x in drug_name_values if x), normalize_text(drug_id))
        concept_values = group.get("drug_concept_id", pd.Series("", index=group.index)).map(normalize_text)
        drug_concept_id = next((x for x in concept_values if x), "")
        source_conflict = positive > 0 and negative > 0
        low_consistency = consistency < MIN_TARGET_DIRECTION_CONSISTENCY
        records.append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "drug_concept_id": drug_concept_id,
            "gene_symbol": gene_symbol,
            "action": (
                "consensus_activation" if consensus_sign > 0
                else "consensus_inhibition" if consensus_sign < 0
                else "direction_conflict"
            ),
            "action_sign": consensus_sign,
            "evidence_weight": compressed_quality,
            "raw_positive_support": positive,
            "raw_negative_support": negative,
            "direction_consistency": consistency,
            "source_conflict_flag": source_conflict,
            "low_direction_consistency_flag": low_consistency,
            "source_track": track,
            "source": ";".join(source_tokens),
            "n_source_tokens": len(source_tokens),
            "n_non_suspicious_source_tokens": len(non_suspicious_tokens),
            "n_suspicious_source_tokens": len(suspicious_tokens),
            "suspicious_source_only_consensus": len(non_suspicious_tokens) == 0 and len(suspicious_tokens) > 0,
            "n_raw_rows": int(len(group)),
        })
    consolidated = pd.DataFrame(records)
    # Pairs with both positive and negative evidence remain auditable, but they
    # are excluded from scoring only when the net direction is insufficiently
    # consistent. A strong majority direction is retained with its consistency
    # penalty already included in the effective score.
    consolidated["scoring_exclusion_flag"] = consolidated["low_direction_consistency_flag"]
    conflicts = consolidated[consolidated["scoring_exclusion_flag"]].copy()
    return consolidated.reset_index(drop=True), conflicts.reset_index(drop=True)


def load_metadata(path: Optional[Path]) -> pd.DataFrame:
    """Load metadata while separating confirmed exclusions from heuristics."""
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()
    lower = {str(c).casefold(): c for c in df.columns}
    id_col = lower.get("drug_id")
    if id_col is None:
        raise KeyError("Drug metadata needs drug_id.")
    name_col = lower.get("drug_name")
    out = pd.DataFrame({"drug_id": df[id_col].map(normalize_text)})
    out["drug_name"] = df[name_col].map(normalize_text) if name_col else out["drug_id"]
    for col in [
        "approval_status", "pulmonary_risk", "notes", "entity_quality_categories",
        "hard_block_reasons", "drug_family_group",
    ]:
        src = lower.get(col)
        out[col] = df[src].map(normalize_text) if src else ""
    for col in ["clinical_score", "safety_score", "availability_score"]:
        src = lower.get(col)
        out[col] = numeric(df[src]).clip(0.0, 1.0) if src else np.nan
    raw_hard_col = lower.get("hard_block_main_table")
    out["source_hard_block_original"] = (
        df[raw_hard_col].map(normalize_bool) if raw_hard_col else False
    )

    def classify_block(row: pd.Series) -> Tuple[bool, bool, str]:
        reasons = {
            x.strip().casefold()
            for x in re.split(r"[;,|]+", normalize_text(row.get("hard_block_reasons", "")))
            if x.strip()
        }
        confirmed = bool(reasons & CONFIRMED_HARD_BLOCK_REASONS)
        heuristic = bool(reasons & HEURISTIC_MANUAL_REVIEW_REASONS)
        if bool(row.get("source_hard_block_original", False)) and not reasons:
            confirmed = True
        review_reasons = sorted(reasons & HEURISTIC_MANUAL_REVIEW_REASONS)
        return confirmed, heuristic, ";".join(review_reasons)

    block_info = out.apply(classify_block, axis=1, result_type="expand")
    out["source_hard_block"] = block_info[0].astype(bool)
    out["manual_review_flag"] = block_info[1].astype(bool)
    out["manual_review_reasons"] = block_info[2].astype(str)

    approval_map = {
        "approved": 1.0, "phase 4": 0.95, "phase 3": 0.85,
        "phase 2": 0.70, "phase 1": 0.55, "clinical": 0.65,
        "preclinical": 0.35, "experimental": 0.20,
    }
    missing = out["clinical_score"].isna()
    if missing.any():
        inferred = out.loc[missing, "approval_status"].str.casefold().map(
            lambda x: max([v for k, v in approval_map.items() if k in x], default=np.nan)
        )
        out.loc[missing, "clinical_score"] = inferred
    out["safety_data_available"] = out["safety_score"].notna() | out["pulmonary_risk"].map(normalize_text).ne("")
    out["translation_metadata_complete"] = out[
        ["clinical_score", "safety_score", "availability_score"]
    ].notna().all(axis=1) & out["pulmonary_risk"].map(normalize_text).ne("")
    return out[out["drug_id"].ne("")].drop_duplicates("drug_id", keep="first")


def score_direct_targets_source_aware(
    signatures: pd.DataFrame,
    targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    directional = targets[targets["action_sign"].ne(0)].copy()
    if directional.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = signatures.merge(directional, on="gene_symbol", how="inner", suffixes=("", "_target"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["target_alignment"] = merged["desired_sign"] * merged["action_sign"]
    merged["edge_mass"] = (
        merged["gene_weight_normalized"]
        * numeric(merged["evidence_weight"], 0.0).clip(0.0, 1.0)
        * numeric(merged["direction_consistency"], 0.0).clip(0.0, 1.0)
    )
    merged["target_weighted_alignment"] = merged["edge_mass"] * merged["target_alignment"]
    records: List[Dict[str, Any]] = []
    for (drug_id, drug_name, signature_id), group in merged.groupby(
        ["drug_id", "drug_name", "signature_id"], dropna=False
    ):
        denominator = float(numeric(group["edge_mass"], 0.0).sum())
        raw = float(numeric(group["target_weighted_alignment"], 0.0).sum() / max(denominator, EPS))
        coverage = float(numeric(group["gene_weight_normalized"], 0.0).sum())
        mean_quality = float(np.average(
            numeric(group["evidence_weight"], 0.0).clip(0.0, 1.0),
            weights=np.maximum(numeric(group["gene_weight_normalized"], 0.0), EPS),
        ))
        mean_consistency = float(np.average(
            numeric(group["direction_consistency"], 0.0).clip(0.0, 1.0),
            weights=np.maximum(numeric(group["gene_weight_normalized"], 0.0), EPS),
        ))
        reliability = float(
            np.sqrt(np.clip(coverage, 0.0, 1.0))
            * np.sqrt(np.clip(mean_quality, 0.0, 1.0))
            * np.sqrt(np.clip(mean_consistency, 0.0, 1.0))
        )
        records.append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "signature_id": signature_id,
            "source_track": normalize_text(group["source_track"].iloc[0]),
            "direct_target_alignment_score": raw,
            "direct_target_gene_weight_coverage": coverage,
            "direct_target_mean_evidence_quality": mean_quality,
            "direct_target_mean_direction_consistency": mean_consistency,
            "direct_target_reliability": reliability,
            "direct_target_effective_score": raw * reliability,
            "n_direct_signature_targets": int(group["gene_symbol"].nunique()),
            "n_direct_tier_a_targets": int(group.loc[
                group["ppi_driver_tier"].eq("TIER_A_HIGH_CONFIDENCE"), "gene_symbol"
            ].nunique()),
            "n_direct_non_suspicious_sources": int(
                numeric(group["n_non_suspicious_source_tokens"], 0).max()
            ),
            "n_direct_suspicious_sources": int(
                numeric(group["n_suspicious_source_tokens"], 0).max()
            ),
            "direct_signature_targets": ";".join(sorted(group["gene_symbol"].unique())),
            "direct_target_sources": ";".join(sorted(set(group["source"].astype(str)))),
        })
    return pd.DataFrame(records), merged


def _build_track_long(
    signatures: pd.DataFrame,
    query_drugs: pd.DataFrame,
    direct_scores: pd.DataFrame,
    network_scores_df: pd.DataFrame,
    track: str,
) -> pd.DataFrame:
    registry_cols = [
        "signature_id", "signature_name", "signature_class", "analysis_role",
        "tier_mode", "disease_scope", "celltype_scope",
    ]
    registry = signatures[registry_cols].drop_duplicates()
    frames: List[pd.DataFrame] = []
    for frame in [query_drugs, direct_scores, network_scores_df]:
        if frame is not None and not frame.empty:
            frames.append(frame[["drug_id", "drug_name"]].drop_duplicates())
    if not frames:
        return pd.DataFrame()
    drugs = pd.concat(frames, ignore_index=True).drop_duplicates("drug_id", keep="first")
    base = drugs.assign(_k=1).merge(registry.assign(_k=1), on="_k").drop(columns="_k")
    if direct_scores is not None and not direct_scores.empty:
        base = base.merge(
            direct_scores.drop(columns="drug_name", errors="ignore"),
            on=["drug_id", "signature_id"], how="left",
        )
    if network_scores_df is not None and not network_scores_df.empty:
        base = base.merge(
            network_scores_df.drop(columns="drug_name", errors="ignore"),
            on=["drug_id", "signature_id"], how="left",
            suffixes=("", "_network"),
        )
    for col in [
        "direct_target_effective_score", "direct_target_alignment_score",
        "direct_target_gene_weight_coverage", "direct_target_reliability",
        "n_direct_tier_a_targets", "n_direct_non_suspicious_sources",
        "network_reversal_score", "network_effective_score",
        "network_signature_influence_mass", "network_mass_reliability",
        "network_mass_percentile",
    ]:
        if col not in base.columns:
            base[col] = np.nan
    base["has_direct_signature_target"] = base["direct_target_effective_score"].notna()
    base["has_network_evidence"] = base["network_effective_score"].notna()
    base["mechanistic_signature_score"] = (
        0.70 * numeric(base["direct_target_effective_score"], 0.0).clip(-1.0, 1.0)
        + 0.30 * numeric(base["network_effective_score"], 0.0).clip(-1.0, 1.0)
    )
    role_key = base["signature_class"].astype(str) + "|" + base["analysis_role"].astype(str)
    base["signature_role_weight"] = role_key.map(
        DEFAULT_CONFIG["scoring"]["signature_role_weights"]
    ).fillna(0.5)
    base["target_source_track"] = track
    return base


def build_mechanistic_dual_track(
    signatures: pd.DataFrame,
    query_drugs: pd.DataFrame,
    direct_main: pd.DataFrame,
    network_main: pd.DataFrame,
    direct_sensitivity: pd.DataFrame,
    network_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    main = _build_track_long(signatures, query_drugs, direct_main, network_main, "main")
    sens = _build_track_long(
        signatures, query_drugs, direct_sensitivity, network_sensitivity, "sensitivity"
    )
    keys = [
        "drug_id", "signature_id", "signature_name", "signature_class",
        "analysis_role", "tier_mode", "disease_scope", "celltype_scope",
    ]
    if main.empty:
        return pd.DataFrame()
    keep_main = keys + [c for c in main.columns if c not in keys and c != "drug_name"]
    out = main[keep_main].copy()
    out.insert(1, "drug_name", main["drug_name"].values)
    if not sens.empty:
        sens_cols = [c for c in sens.columns if c not in keys + ["drug_name"]]
        sens_renamed = sens[keys + ["drug_name"] + sens_cols].rename(
            columns={
                "drug_name": "drug_name_sensitivity",
                **{c: f"{c}_sensitivity" for c in sens_cols},
            }
        )
        out = out.merge(sens_renamed, on=keys, how="outer")
        if "drug_name_sensitivity" in out.columns:
            out["drug_name"] = (
                out["drug_name"].replace("", np.nan)
                .fillna(out["drug_name_sensitivity"])
                .fillna(out["drug_id"].astype(str))
            )
    for col in [
        "mechanistic_signature_score", "mechanistic_signature_score_sensitivity",
        "direct_target_effective_score", "direct_target_effective_score_sensitivity",
        "network_effective_score", "network_effective_score_sensitivity",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    main_score = numeric(out["mechanistic_signature_score"], 0.0)
    sens_score = numeric(out["mechanistic_signature_score_sensitivity"], 0.0)
    out["mechanism_source_delta"] = sens_score - main_score
    out["mechanism_source_robust"] = (
        (main_score >= V3_TIERS["mechanism_positive_cut"])
        & (sens_score >= V3_TIERS["mechanism_sensitivity_positive_cut"])
    )
    out["mechanism_sensitivity_only_suspicious"] = (
        (main_score < V3_TIERS["mechanism_sensitivity_positive_cut"])
        & (sens_score >= V3_TIERS["mechanism_positive_cut"])
    )
    out["mechanism_source_conflict"] = (
        (main_score > V3_TIERS["mechanism_positive_cut"])
        & (sens_score < V3_TIERS["mechanism_conflict_cut"])
    )
    return out


def build_target_conflict_summary(
    signatures: pd.DataFrame,
    main_conflicts: pd.DataFrame,
    sensitivity_conflicts: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for track, conflicts in [("main", main_conflicts), ("sensitivity", sensitivity_conflicts)]:
        if conflicts.empty:
            continue
        merged = signatures[["signature_id", "analysis_role", "gene_symbol"]].merge(
            conflicts, on="gene_symbol", how="inner"
        )
        if merged.empty:
            continue
        agg = merged.groupby("drug_id", as_index=False).agg(
            **{
                f"n_ot_target_conflicts_{track}": ("gene_symbol", "nunique"),
                f"conflicting_ot_target_genes_{track}": (
                    "gene_symbol", lambda s: ";".join(sorted(set(s)))
                ),
            }
        )
        frames.append(agg)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="drug_id", how="outer")
    return out


def summarize_mechanistic_v3(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for drug_id, group in long_df.groupby("drug_id", sort=False):
        primary = group[group["analysis_role"].eq("PRIMARY")].copy()
        if primary.empty:
            primary = group.copy()
        main_scores = numeric(primary["mechanistic_signature_score"], 0.0)
        sens_scores = numeric(primary["mechanistic_signature_score_sensitivity"], 0.0)
        robust = primary["mechanism_source_robust"].fillna(False)
        suspicious_only = primary["mechanism_sensitivity_only_suspicious"].fillna(False)
        direct_main = numeric(primary.get("direct_target_effective_score", pd.Series(np.nan, index=primary.index)))
        network_main = numeric(primary.get("network_effective_score", pd.Series(np.nan, index=primary.index)))
        direct_sens = numeric(primary.get("direct_target_effective_score_sensitivity", pd.Series(np.nan, index=primary.index)))
        network_sens = numeric(primary.get("network_effective_score_sensitivity", pd.Series(np.nan, index=primary.index)))
        tier_a = numeric(primary.get("n_direct_tier_a_targets", pd.Series(0, index=primary.index)), 0)
        non_susp_sources = numeric(
            primary.get("n_direct_non_suspicious_sources", pd.Series(0, index=primary.index)), 0
        )
        rows.append({
            "drug_id": drug_id,
            "drug_name": normalize_text(group["drug_name"].dropna().iloc[0]) if group["drug_name"].notna().any() else str(drug_id),
            "mechanistic_support_score_main": float(max(main_scores.max(), 0.0)),
            "mechanistic_support_score_sensitivity": float(max(sens_scores.max(), 0.0)),
            "mean_primary_mechanistic_score_main": float(main_scores.mean()),
            "mean_primary_mechanistic_score_sensitivity": float(sens_scores.mean()),
            "n_primary_source_robust_signatures": int(robust.sum()),
            "n_primary_suspicious_only_signatures": int(suspicious_only.sum()),
            "n_primary_direct_supported_main": int((direct_main >= 0.05).sum()),
            "n_primary_network_supported_main": int((network_main >= 0.05).sum()),
            "n_primary_direct_supported_sensitivity": int((direct_sens >= 0.05).sum()),
            "n_primary_network_supported_sensitivity": int((network_sens >= 0.05).sum()),
            "n_primary_main_mechanistic_conflicts": int((main_scores <= V3_TIERS["mechanism_conflict_cut"]).sum()),
            "max_direct_tier_a_targets_main": int(tier_a.max()) if len(tier_a) else 0,
            "max_non_suspicious_direct_sources": int(non_susp_sources.max()) if len(non_susp_sources) else 0,
            "best_mechanistic_signature_id_main": str(primary.loc[main_scores.idxmax(), "signature_id"]),
        })
    out = pd.DataFrame(rows)
    out["mechanism_source_stable"] = (
        out["n_primary_source_robust_signatures"].gt(0)
        & out["n_primary_main_mechanistic_conflicts"].eq(0)
    )
    out["mechanism_sensitivity_delta"] = (
        out["mechanistic_support_score_sensitivity"] - out["mechanistic_support_score_main"]
    )
    return out


def add_safety_flags(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    risk = normalize_pulmonary_risk_series(out)
    high_terms = [
        normalize_text(t).casefold()
        for t in cfg["filters"]["high_risk_terms"]
        if normalize_text(t)
    ]
    out["high_pulmonary_risk"] = risk.map(
        lambda v: bool(v) and any(t in v for t in high_terms)
    ).fillna(False).astype(bool)
    if "source_hard_block" not in out.columns:
        out["source_hard_block"] = False
    if "manual_review_flag" not in out.columns:
        out["manual_review_flag"] = False
    out["source_hard_block"] = out["source_hard_block"].map(normalize_bool).astype(bool)
    out["manual_review_flag"] = out["manual_review_flag"].map(normalize_bool).astype(bool)
    out["hard_safety_or_entity_exclusion"] = out["source_hard_block"] | (
        out["high_pulmonary_risk"]
        if bool(cfg["filters"]["exclude_high_pulmonary_risk"])
        else False
    )
    if "translation_metadata_complete" not in out.columns:
        out["translation_metadata_complete"] = False
    out["translation_review_required"] = ~out["translation_metadata_complete"].map(normalize_bool)
    return out


def build_primary_query_summary_v3(
    query: pd.DataFrame,
    mechanism: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    out = query.copy()
    if not mechanism.empty:
        out = out.merge(mechanism.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    if not conflict_summary.empty:
        out = out.merge(conflict_summary, on="drug_id", how="left")
    if not metadata.empty:
        out = out.merge(metadata.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    numeric_defaults = {
        "mechanistic_support_score_main": 0.0,
        "mechanistic_support_score_sensitivity": 0.0,
        "n_primary_source_robust_signatures": 0,
        "n_primary_suspicious_only_signatures": 0,
        "n_primary_direct_supported_main": 0,
        "n_primary_network_supported_main": 0,
        "n_primary_main_mechanistic_conflicts": 0,
        "max_direct_tier_a_targets_main": 0,
        "n_ot_target_conflicts_main": 0,
        "n_ot_target_conflicts_sensitivity": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = numeric(out[col], default)
    if "mechanism_source_stable" not in out.columns:
        out["mechanism_source_stable"] = False
    out["mechanism_source_stable"] = out["mechanism_source_stable"].map(normalize_bool)
    out["context_relevance_score"] = out.apply(context_relevance_score, axis=1)
    out["clinical_component"] = numeric(
        out.get("clinical_score", pd.Series(np.nan, index=out.index))
    ).fillna(0.0)
    out["primary_query_global_score"] = (
        V3_PRIMARY_QUERY_WEIGHT * numeric(out["signature_query_reversal_score"], 0.0).clip(-1.0, 1.0)
        + V3_PRIMARY_MECHANISM_WEIGHT * numeric(out["mechanistic_support_score_main"], 0.0).clip(0.0, 1.0)
        + V3_PRIMARY_CONTEXT_WEIGHT * numeric(out["context_relevance_score"], 0.0).clip(0.0, 1.0)
        + V3_PRIMARY_CLINICAL_WEIGHT * numeric(out["clinical_component"], 0.0).clip(0.0, 1.0)
    )
    out = add_safety_flags(out, cfg)
    out["target_source_conflict_present"] = out["n_ot_target_conflicts_main"].gt(0)
    out["mechanism_support_class"] = np.select(
        [
            out["mechanism_source_stable"],
            out["mechanistic_support_score_main"].ge(V3_TIERS["mechanism_positive_cut"]),
            out["n_primary_suspicious_only_signatures"].gt(0),
        ],
        [
            "SOURCE_ROBUST_MAIN_MECHANISM",
            "MAIN_MECHANISM_NOT_SOURCE_ROBUST",
            "SENSITIVITY_ONLY_SUSPICIOUS_SOURCE",
        ],
        default="NO_MAPPED_OT_MECHANISM",
    )

    def query_support(r: pd.Series) -> str:
        if float(r["signature_query_qvalue_strict"]) <= 0.05:
            return "STRICT_TWO_ARM_FDR_0.05"
        if float(r["signature_query_qvalue_strict"]) <= 0.25:
            return "STRICT_TWO_ARM_FDR_0.25"
        if float(r["signature_query_qvalue_best_arm"]) <= 0.05:
            return "ONE_ARM_SUPPORTED_ONLY"
        return "SCORE_ONLY"
    out["query_statistical_support_class"] = out.apply(query_support, axis=1)

    def tier(r: pd.Series) -> Tuple[str, str]:
        if bool(r["hard_safety_or_entity_exclusion"]):
            return "EXCLUDED", normalize_text(r.get("hard_block_reasons", "")) or "Confirmed safety/entity exclusion"
        score = float(r["primary_query_global_score"])
        query_score = float(r["signature_query_reversal_score"])
        strict_q = float(r["signature_query_qvalue_strict"])
        best_q = float(r["signature_query_qvalue_best_arm"])
        arm_balance = float(r["query_arm_balance"]) if pd.notna(r["query_arm_balance"]) else 0.0
        mech = float(r["mechanistic_support_score_main"])
        source_stable = bool(r["mechanism_source_stable"])
        conflict = bool(r["target_source_conflict_present"]) or int(r["n_primary_main_mechanistic_conflicts"]) > 0
        translation_complete = bool(r.get("translation_metadata_complete", False))

        final_a_base = (
            score >= V3_TIERS["final_a_min_score"]
            and strict_q <= V3_TIERS["final_a_max_strict_q"]
            and mech >= V3_TIERS["final_a_min_mechanism"]
            and source_stable and not conflict
        )
        if final_a_base and (
            translation_complete or not V3_TIERS["final_a_require_translation_complete"]
        ):
            return "FINAL_CANDIDATE_A", "Strict two-arm support, source-robust mechanism, and complete translation metadata"
        if final_a_base:
            return "STRICT_CANDIDATE_PENDING_TRANSLATION_REVIEW", "Strict statistical/mechanistic support but safety or availability metadata is incomplete"
        if score >= V3_TIERS["final_b_min_score"] and strict_q <= V3_TIERS["final_b_max_strict_q"] and not conflict:
            return "STRICT_CANDIDATE_B_PENDING_REVIEW", "Moderate strict two-arm support; translation review required"

        one_arm_a = (
            score >= V3_TIERS["one_arm_a_min_score"]
            and query_score >= V3_TIERS["one_arm_a_min_query_score"]
            and best_q <= V3_TIERS["one_arm_a_max_best_q"]
            and arm_balance >= V3_TIERS["one_arm_a_min_arm_balance"]
            and not conflict
        )
        if one_arm_a and source_stable:
            return "ONE_ARM_PRIORITY_A_SOURCE_ROBUST", "Strong one-arm SigCom support with source-robust OT mechanism"
        if (
            one_arm_a
            and float(r["context_relevance_score"]) >= 0.80
            and int(r.get("n_primary_suspicious_only_signatures", 0)) == 0
            and not bool(r.get("manual_review_flag", False))
        ):
            return "ONE_ARM_PRIORITY_A_CONTEXT_RELEVANT", "Strong one-arm SigCom support in lung/immune-relevant context; no suspicious-only mechanism claim"

        one_arm_b = (
            score >= V3_TIERS["one_arm_b_min_score"]
            and query_score >= V3_TIERS["one_arm_b_min_query_score"]
            and best_q <= V3_TIERS["one_arm_b_max_best_q"]
            and arm_balance >= V3_TIERS["one_arm_b_min_arm_balance"]
        )
        if one_arm_b:
            if int(r.get("n_primary_suspicious_only_signatures", 0)) > 0 and mech < V3_TIERS["mechanism_positive_cut"]:
                return "ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM", "One-arm support; apparent mechanism exists only in the downweighted suspicious-source sensitivity track"
            if mech >= V3_TIERS["mechanism_positive_cut"] and not source_stable:
                return "ONE_ARM_PRIORITY_B_MECHANISM_UNCERTAIN", "One-arm support with mechanism sensitive to target-source policy"
            return "ONE_ARM_PRIORITY_B", "Moderate one-arm SigCom support; no source-robust OT mechanism"
        if best_q <= 0.05 and score > 0:
            return "ONE_ARM_EXPLORATORY", "Single-arm evidence below priority balance/score criteria"
        if score > 0:
            return "EXPLORATORY", "Positive score without statistical priority support"
        return "EXCLUDED", "Non-positive score"

    assigned = out.apply(tier, axis=1, result_type="expand")
    out["candidate_tier"] = assigned[0]
    out["candidate_tier_reason"] = assigned[1]
    order = {
        "FINAL_CANDIDATE_A": 1,
        "STRICT_CANDIDATE_PENDING_TRANSLATION_REVIEW": 2,
        "STRICT_CANDIDATE_B_PENDING_REVIEW": 3,
        "ONE_ARM_PRIORITY_A_SOURCE_ROBUST": 4,
        "ONE_ARM_PRIORITY_A_CONTEXT_RELEVANT": 5,
        "ONE_ARM_PRIORITY_B": 6,
        "ONE_ARM_PRIORITY_B_MECHANISM_UNCERTAIN": 7,
        "ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM": 8,
        "ONE_ARM_EXPLORATORY": 9,
        "EXPLORATORY": 10,
        "EXCLUDED": 11,
    }
    out["candidate_tier_order"] = out["candidate_tier"].map(order).fillna(99)
    out = out.sort_values(
        ["candidate_tier_order", "primary_query_global_score", "signature_query_qvalue_best_arm", "drug_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    out["drug_global_rank"] = np.arange(1, len(out) + 1)
    out["screening_track"] = "PRIMARY_SIGCOM_GLOBAL_QUERY_SOURCE_AWARE"
    return out


def build_target_mechanism_summary_v3(
    mechanistic_long: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    query_ids: set[str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if mechanistic_long.empty:
        return pd.DataFrame()
    mech = summarize_mechanistic_v3(mechanistic_long)
    direct_presence = mechanistic_long.groupby("drug_id", as_index=False).agg(
        n_signatures_with_direct_main=("direct_target_effective_score", lambda s: int(numeric(s).notna().sum())),
        n_signatures_with_direct_sensitivity=("direct_target_effective_score_sensitivity", lambda s: int(numeric(s).notna().sum())),
        max_network_mass_reliability_main=("network_mass_reliability", "max"),
        max_network_effective_score_main=("network_effective_score", "max"),
    )
    out = mech.merge(direct_presence, on="drug_id", how="left")
    out = out[(numeric(out["n_signatures_with_direct_main"], 0) > 0) |
              (numeric(out["n_signatures_with_direct_sensitivity"], 0) > 0)].copy()
    if not conflict_summary.empty:
        out = out.merge(conflict_summary, on="drug_id", how="left")
    if not metadata.empty:
        out = out.merge(metadata.drop(columns="drug_name", errors="ignore"), on="drug_id", how="left")
    out["present_in_sigcom_query"] = out["drug_id"].isin(query_ids)
    out = add_safety_flags(out, cfg)
    for col in ["n_ot_target_conflicts_main", "n_ot_target_conflicts_sensitivity"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = numeric(out[col], 0)
    out["target_mechanism_score"] = (
        0.75 * numeric(out["mechanistic_support_score_main"], 0.0).clip(0.0, 1.0)
        + 0.10 * out["mechanism_source_stable"].map(normalize_bool).astype(float)
        + 0.10 * (numeric(out["max_direct_tier_a_targets_main"], 0) > 0).astype(float)
        + 0.05 * numeric(out.get("clinical_score", pd.Series(np.nan, index=out.index))).fillna(0.0).clip(0.0, 1.0)
    )
    conflict = out["n_ot_target_conflicts_main"].gt(0)
    out["mechanism_candidate_class"] = np.select(
        [
            out["hard_safety_or_entity_exclusion"],
            conflict,
            out["mechanism_source_stable"]
                & (out["mechanistic_support_score_main"] >= 0.25)
                & (out["max_direct_tier_a_targets_main"] >= 1),
            out["mechanism_source_stable"]
                & (out["mechanistic_support_score_main"] >= 0.10),
            (out["mechanistic_support_score_main"] < 0.05)
                & (out["mechanistic_support_score_sensitivity"] >= 0.10),
        ],
        [
            "EXCLUDED",
            "TARGET_ACTION_CONFLICT_REVIEW",
            "TARGET_MECHANISM_PRIORITY_SOURCE_ROBUST",
            "TARGET_MECHANISM_EXPLORATORY_SOURCE_ROBUST",
            "TARGET_MECHANISM_SENSITIVITY_ONLY_SUSPICIOUS_SOURCE",
        ],
        default="LOW_OR_CONFLICTING_MECHANISM",
    )
    order = {
        "TARGET_MECHANISM_PRIORITY_SOURCE_ROBUST": 1,
        "TARGET_MECHANISM_EXPLORATORY_SOURCE_ROBUST": 2,
        "TARGET_ACTION_CONFLICT_REVIEW": 3,
        "TARGET_MECHANISM_SENSITIVITY_ONLY_SUSPICIOUS_SOURCE": 4,
        "LOW_OR_CONFLICTING_MECHANISM": 5,
        "EXCLUDED": 6,
    }
    out["mechanism_candidate_order"] = out["mechanism_candidate_class"].map(order).fillna(99)
    out = out.sort_values(
        ["mechanism_candidate_order", "target_mechanism_score", "drug_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    out["target_mechanism_rank"] = np.arange(1, len(out) + 1)
    out["screening_track"] = "TARGET_MECHANISM_DISCOVERY_SOURCE_AWARE"
    return out


def _target_source_qc(annotated: pd.DataFrame, main: pd.DataFrame, sens: pd.DataFrame,
                      main_conflicts: pd.DataFrame, sens_conflicts: pd.DataFrame) -> Dict[str, Any]:
    if annotated.empty:
        return {}
    directional = annotated[annotated["action_sign"].ne(0)]
    return {
        "raw_rows": int(len(annotated)),
        "directional_raw_rows": int(len(directional)),
        "suspicious_source_only_rows": int(annotated["suspicious_source_only"].sum()),
        "suspicious_source_only_directional_rows": int(directional["suspicious_source_only"].sum()),
        "mixed_suspicious_source_rows": int(annotated["mixed_suspicious_source"].sum()),
        "main_rows_excluded_by_source_policy": int(
            (annotated["main_source_multiplier"] <= 0).sum()
        ),
        "main_consolidated_pairs": int(len(main)),
        "sensitivity_consolidated_pairs": int(len(sens)),
        "main_direction_conflict_pairs": int(len(main_conflicts)),
        "sensitivity_direction_conflict_pairs": int(len(sens_conflicts)),
        "main_scored_directional_pairs": int(main["action_sign"].ne(0).sum()) if not main.empty else 0,
        "sensitivity_scored_directional_pairs": int(sens["action_sign"].ne(0).sum()) if not sens.empty else 0,
    }


def _query_input_qc(raw: pd.DataFrame, aggregated: pd.DataFrame) -> Dict[str, Any]:
    return {
        "source_reversers_only": bool(SOURCE_REVERSERS_ONLY),
        "rows_retained": int(len(raw)),
        "unique_drugs": int(aggregated["drug_id"].nunique()) if not aggregated.empty else 0,
        "drugs_with_multiple_context_rows": int((aggregated.get("n_query_context_rows", pd.Series(dtype=float)) > 1).sum()) if not aggregated.empty else 0,
        "context_consistency_warnings": int(aggregated.get("query_context_consistency_warning", pd.Series(False, index=aggregated.index)).fillna(False).sum()) if not aggregated.empty else 0,
        "strict_q_provenance": sorted(set(raw.get("qvalue_strict_provenance", pd.Series(dtype=str)).astype(str))) if not raw.empty else [],
        "best_arm_q_provenance": sorted(set(raw.get("qvalue_best_arm_provenance", pd.Series(dtype=str)).astype(str))) if not raw.empty else [],
        "z_arm_columns_available": bool(
            not raw.empty
            and raw.get("z_score_up", pd.Series(np.nan, index=raw.index)).notna().any()
            and raw.get("z_score_down", pd.Series(np.nan, index=raw.index)).notna().any()
        ),
    }


def main() -> None:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    base = Path(__file__).resolve().parent
    signature_path = resolve_path(SIGNATURE_PATH, base, required=True)
    perturbation_path = resolve_optional_input_path(DRUG_PERTURBATION_PATH, base, "Drug perturbation table")
    target_path = resolve_optional_input_path(DRUG_TARGET_PATH, base, "Drug target table")
    metadata_path = resolve_optional_input_path(DRUG_METADATA_PATH, base, "Drug metadata table")
    network_path = resolve_optional_input_path(NETWORK_EDGES_PATH, base, "Gene network edge table")
    alias_path = resolve_optional_input_path(GENE_ALIAS_MAPPING_PATH, base, "Gene alias mapping table")
    output = Path(os.path.expanduser(os.path.expandvars(OUTPUT_ROOT)))
    if not output.is_absolute():
        output = (base / output).resolve()
    output = choose_output_dir(output, EXISTING_OUTPUT_POLICY)

    print("=" * 78)
    print("37a Drug pairing v3 - source-aware dual-track target robustness")
    print(f"Script       : {SCRIPT_VERSION}")
    print(f"Signatures   : {signature_path}")
    print(f"Perturbation : {perturbation_path if perturbation_path else 'DISABLED'}")
    print(f"Targets      : {target_path if target_path else 'DISABLED'}")
    print(f"Metadata     : {metadata_path if metadata_path else 'DISABLED'}")
    print(f"Network      : {network_path if network_path else 'DISABLED'}")
    print(f"Output       : {output}")
    print("=" * 78)

    signatures = validate_signatures(read_csv_safe(signature_path))
    alias = load_alias_mapping(alias_path)
    signatures["gene_symbol"] = signatures["gene_symbol"].map(lambda x: alias.get(x, x))
    perturbation_df = read_csv_safe(perturbation_path)
    if perturbation_df.empty:
        raise ValueError("This v3 workflow requires the SigCom signature-level perturbation table.")
    perturbation_mode = detect_perturbation_mode(perturbation_df, PERTURBATION_MODE)
    if perturbation_mode != "signature_level":
        raise ValueError("This replacement v3 script is designed for signature-level SigCom input.")
    query_raw = load_signature_level_perturbations(perturbation_df, cfg)
    if query_raw.empty:
        raise ValueError("No usable SigCom reverser rows remain after final input filtering.")
    query_aggregated = aggregate_signature_level(query_raw)
    if query_aggregated.empty or query_aggregated["drug_id"].duplicated().any():
        raise ValueError("SigCom aggregation failed to produce one unique row per drug_id.")
    query_input_qc = _query_input_qc(query_raw, query_aggregated)

    raw_targets, target_qc_base = load_targets(target_path, alias, cfg)
    annotated_targets = annotate_target_source_policy(raw_targets)
    targets_main, conflicts_main = consolidate_target_track(annotated_targets, "main")
    targets_sensitivity, conflicts_sensitivity = consolidate_target_track(
        annotated_targets, "sensitivity"
    )
    metadata = load_metadata(metadata_path)
    edges = load_network(network_path, alias, cfg)

    direct_main, detail_main = score_direct_targets_source_aware(signatures, targets_main)
    direct_sens, detail_sens = score_direct_targets_source_aware(signatures, targets_sensitivity)
    query_ids = set(query_aggregated["drug_id"])
    direct_main_ids = set(direct_main["drug_id"]) if not direct_main.empty else set()
    direct_sens_ids = set(direct_sens["drug_id"]) if not direct_sens.empty else set()

    if NETWORK_DRUG_SCOPE == "query_and_direct_signature_targets":
        main_allowed = query_ids | direct_main_ids
        sens_allowed = query_ids | direct_sens_ids
        network_targets_main = targets_main[targets_main["drug_id"].isin(main_allowed)].copy()
        network_targets_sens = targets_sensitivity[
            targets_sensitivity["drug_id"].isin(sens_allowed)
        ].copy()
    elif NETWORK_DRUG_SCOPE == "all_directional_targets":
        network_targets_main = targets_main.copy()
        network_targets_sens = targets_sensitivity.copy()
    else:
        raise ValueError(f"Unknown NETWORK_DRUG_SCOPE: {NETWORK_DRUG_SCOPE}")

    network_raw_main = network_scores(signatures, network_targets_main, edges, cfg)
    network_raw_sens = network_scores(signatures, network_targets_sens, edges, cfg)
    network_main = calibrate_network_scores(network_raw_main)
    network_sens = calibrate_network_scores(network_raw_sens)

    mechanistic_long = build_mechanistic_dual_track(
        signatures, query_aggregated,
        direct_main, network_main,
        direct_sens, network_sens,
    )
    mechanism_summary = summarize_mechanistic_v3(mechanistic_long)
    conflict_summary = build_target_conflict_summary(
        signatures, conflicts_main, conflicts_sensitivity
    )
    primary = build_primary_query_summary_v3(
        query_aggregated, mechanism_summary, conflict_summary, metadata, cfg
    )
    target_discovery = build_target_mechanism_summary_v3(
        mechanistic_long, conflict_summary, metadata, query_ids, cfg
    )

    registry = signatures.groupby(
        ["signature_id", "signature_name", "signature_class", "analysis_role"],
        as_index=False,
    ).agg(
        n_genes=("gene_symbol", "nunique"),
        total_gene_weight=("gene_weight_normalized", "sum"),
        n_tier_a_genes=(
            "ppi_driver_tier",
            lambda s: int((s == "TIER_A_HIGH_CONFIDENCE").sum()),
        ),
    )
    registry.to_csv(output / "01_signature_qc.csv", index=False)
    write_empty_or_frame(query_raw, output / "02b_signature_query_rows_standardized.csv")
    write_empty_or_frame(query_aggregated, output / "02c_drug_signature_query_aggregated.csv")
    write_empty_or_frame(annotated_targets, output / "02d_drug_target_actions_source_annotated.csv")
    write_empty_or_frame(targets_main, output / "02e_targets_main_consolidated.csv")
    write_empty_or_frame(targets_sensitivity, output / "02f_targets_sensitivity_consolidated.csv")
    write_empty_or_frame(conflicts_main, output / "02g_target_action_conflicts_main.csv")
    write_empty_or_frame(conflicts_sensitivity, output / "02h_target_action_conflicts_sensitivity.csv")
    write_empty_or_frame(
        annotated_targets[annotated_targets["suspicious_source_only"]].copy(),
        output / "02i_suspicious_source_only_raw_rows.csv",
    )
    disagreement_audit = pd.concat([
        targets_main[targets_main["source_conflict_flag"]].assign(audit_track="main"),
        targets_sensitivity[targets_sensitivity["source_conflict_flag"]].assign(audit_track="sensitivity"),
    ], ignore_index=True)
    write_empty_or_frame(disagreement_audit, output / "02j_source_direction_disagreement_audit.csv")
    write_empty_or_frame(detail_main, output / "04a_direct_target_alignment_main_detail.csv")
    write_empty_or_frame(detail_sens, output / "04b_direct_target_alignment_sensitivity_detail.csv")
    write_empty_or_frame(network_main, output / "05a_network_propagation_main.csv")
    write_empty_or_frame(network_sens, output / "05b_network_propagation_sensitivity.csv")
    write_empty_or_frame(mechanistic_long, output / "06_dual_track_mechanistic_scores_long.csv")
    write_empty_or_frame(mechanism_summary, output / "06b_mechanism_source_robustness_summary.csv")
    primary.to_csv(output / "07_final_candidate_screening.csv", index=False)

    # Strict candidates and honest one-arm tiers.
    primary[primary["candidate_tier"].eq("FINAL_CANDIDATE_A")].to_csv(
        output / "08_final_candidate_A.csv", index=False
    )
    primary[primary["candidate_tier"].str.startswith("STRICT_CANDIDATE", na=False)].to_csv(
        output / "09_strict_candidates_pending_review.csv", index=False
    )
    one_arm = primary[primary["candidate_tier"].str.startswith("ONE_ARM_PRIORITY", na=False)].copy()
    one_arm.to_csv(output / "10_one_arm_priority_reversers.csv", index=False)
    primary[primary["candidate_tier"].eq("ONE_ARM_PRIORITY_A_SOURCE_ROBUST")].to_csv(
        output / "10a_one_arm_priority_A_source_robust.csv", index=False
    )
    primary[primary["candidate_tier"].eq("ONE_ARM_PRIORITY_A_CONTEXT_RELEVANT")].to_csv(
        output / "10b_one_arm_priority_A_context_relevant.csv", index=False
    )
    primary[primary["candidate_tier"].isin([
        "ONE_ARM_PRIORITY_B",
        "ONE_ARM_PRIORITY_B_MECHANISM_UNCERTAIN",
        "ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM",
    ])].to_csv(output / "10c_one_arm_priority_B.csv", index=False)
    primary[primary["candidate_tier"].eq(
        "ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM"
    )].to_csv(output / "10c2_one_arm_priority_B_suspicious_mechanism.csv", index=False)
    primary[primary["candidate_tier"].isin(["ONE_ARM_EXPLORATORY", "EXPLORATORY"])].to_csv(
        output / "10d_exploratory_query_candidates.csv", index=False
    )
    primary[primary["candidate_tier"].eq("EXCLUDED")].to_csv(
        output / "11_excluded_candidates.csv", index=False
    )
    primary[primary["manual_review_flag"].fillna(False)].to_csv(
        output / "11b_manual_review_not_auto_excluded.csv", index=False
    )
    target_discovery.to_csv(output / "14_target_mechanism_candidates.csv", index=False)
    target_discovery[target_discovery["mechanism_candidate_class"].eq(
        "TARGET_MECHANISM_PRIORITY_SOURCE_ROBUST"
    )].to_csv(output / "14a_target_mechanism_priority_source_robust.csv", index=False)
    target_discovery[target_discovery["mechanism_candidate_class"].eq(
        "TARGET_MECHANISM_SENSITIVITY_ONLY_SUSPICIOUS_SOURCE"
    )].to_csv(output / "14b_suspicious_source_only_mechanisms.csv", index=False)
    target_discovery[target_discovery["mechanism_candidate_class"].eq(
        "TARGET_ACTION_CONFLICT_REVIEW"
    )].to_csv(output / "14c_target_action_conflict_review.csv", index=False)

    per_signature = output / "per_signature"
    per_signature.mkdir(exist_ok=True)
    for signature_id, frame in mechanistic_long.groupby("signature_id"):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(signature_id)).strip("_") or "signature"
        frame.sort_values("mechanistic_signature_score", ascending=False).to_csv(
            per_signature / f"{safe_name}.csv", index=False
        )

    source_qc = _target_source_qc(
        annotated_targets, targets_main, targets_sensitivity,
        conflicts_main, conflicts_sensitivity,
    )
    q_strict = numeric(query_aggregated["signature_query_qvalue_strict"])
    q_best = numeric(query_aggregated["signature_query_qvalue_best_arm"])
    qc = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "perturbation_mode": perturbation_mode,
        "global_query_scored_once": True,
        "global_query_replicated_across_ot_signatures": False,
        "target_source_policy": {
            "main": "exclude suspicious-source-only rows",
            "sensitivity": (
                f"suspicious-only x{SENSITIVITY_SUSPICIOUS_ONLY_MULTIPLIER}; "
                f"mixed-source x{SENSITIVITY_MIXED_SOURCE_MULTIPLIER}"
            ),
            "suspicious_sources": sorted(SUSPICIOUS_TARGET_SOURCES),
            "minimum_direction_consistency": MIN_TARGET_DIRECTION_CONSISTENCY,
        },
        "network_missing_components_renormalized": False,
        "network_drug_scope": NETWORK_DRUG_SCOPE,
        "network_mass_calibration": f"mass/(mass+signature_q{NETWORK_MASS_REFERENCE_QUANTILE})",
        "n_signatures": int(signatures["signature_id"].nunique()),
        "n_signature_gene_rows": int(len(signatures)),
        "query_input_qc": query_input_qc,
        "n_signature_query_rows_loaded": int(len(query_raw)),
        "n_signature_query_drugs_aggregated": int(len(query_aggregated)),
        "n_query_drugs_with_main_target_match": int(len(query_ids & set(targets_main["drug_id"]))),
        "n_query_drugs_with_sensitivity_target_match": int(len(query_ids & set(targets_sensitivity["drug_id"]))),
        "n_network_rows_main": int(len(network_main)),
        "n_network_rows_sensitivity": int(len(network_sens)),
        "n_primary_drugs_evaluated": int(len(primary)),
        "n_final_candidate_a": int(primary["candidate_tier"].eq("FINAL_CANDIDATE_A").sum()),
        "n_strict_pending_review": int(primary["candidate_tier"].str.startswith("STRICT_CANDIDATE", na=False).sum()),
        "n_one_arm_priority_a_source_robust": int(primary["candidate_tier"].eq("ONE_ARM_PRIORITY_A_SOURCE_ROBUST").sum()),
        "n_one_arm_priority_a_context_relevant": int(primary["candidate_tier"].eq("ONE_ARM_PRIORITY_A_CONTEXT_RELEVANT").sum()),
        "n_one_arm_priority_b": int(primary["candidate_tier"].isin([
            "ONE_ARM_PRIORITY_B",
            "ONE_ARM_PRIORITY_B_MECHANISM_UNCERTAIN",
            "ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM",
        ]).sum()),
        "n_one_arm_priority_b_suspicious_mechanism": int(
            primary["candidate_tier"].eq("ONE_ARM_PRIORITY_B_SUSPICIOUS_MECHANISM").sum()
        ),
        "n_confirmed_hard_blocks": int(primary["source_hard_block"].fillna(False).sum()),
        "n_manual_review_flags": int(primary["manual_review_flag"].fillna(False).sum()),
        "n_translation_metadata_complete": int(primary["translation_metadata_complete"].fillna(False).sum()),
        "strict_q_min": float(q_strict.min()),
        "strict_q_max": float(q_strict.max()),
        "best_arm_q_min": float(q_best.min()),
        "n_strict_q_le_005": int((q_strict <= 0.05).sum()),
        "n_strict_q_le_025": int((q_strict <= 0.25).sum()),
        "n_best_arm_q_le_005": int((q_best <= 0.05).sum()),
        "target_qc_base": target_qc_base,
        "target_source_qc": source_qc,
        "interpretation": (
            "Final tiers remain strict. One-arm evidence is hierarchically reported. "
            "Mechanistic priority requires support that survives exclusion of suspicious-only target sources."
        ),
    }
    write_json(output / "12_drug_pairing_qc.json", qc)

    methods = f"""37a DRUG PAIRING V3 METHODS AND INTERPRETATION
================================================
Script version: {SCRIPT_VERSION}

Major repairs
-------------
1. GuideToPharmacology-only target rows are excluded from the main mechanism
   analysis. They remain in a sensitivity track at weight
   {SENSITIVITY_SUSPICIOUS_ONLY_MULTIPLIER:.2f}. Mixed-source rows receive
   sensitivity multiplier {SENSITIVITY_MIXED_SOURCE_MULTIPLIER:.2f}.
2. Drug-gene actions are consolidated across sources. Activation/inhibition
   conflicts with direction consistency below {MIN_TARGET_DIRECTION_CONSISTENCY:.2f}
   are excluded from direct and network scoring and written to conflict tables.
3. Main and sensitivity direct/network analyses are both computed. A mechanism
   is called source-robust only when it remains positive in both tracks and is
   non-conflicting in the main track.
4. Network direction remains calibrated by propagated influence mass. Missing
   components are never re-normalized to 100 percent.
5. The global SigCom query is counted once per drug. Strict qvalue and best-arm
   qvalue remain separate. One-arm evidence receives explicit A/B/exploratory
   labels instead of being relabelled as a strict final candidate.
6. Heuristic entity flags such as fluoroquinolone_like trigger manual review,
   not automatic exclusion. Confirmed experimental/non-drug flags remain hard
   exclusions.
7. Missing pulmonary-risk and safety fields are reported as missing. They are
   not interpreted as evidence of safety, and Final A can require complete
   translation metadata.

Primary interpretation boundary
-------------------------------
The current global query is directionally imbalanced. Strict two-arm FDR remains
required for Final A/B. One-arm priority outputs are intended for shortlist
construction and follow-up, not as statistically strict final candidates.
"""
    (output / "13_methods_and_score_definition.txt").write_text(methods, encoding="utf-8")
    write_json(output / "00_input_manifest.json", {
        "run_time": datetime.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "signature_path": str(signature_path),
        "signature_sha256": sha256_file(signature_path),
        "drug_perturbation_path": str(perturbation_path),
        "drug_perturbation_sha256": sha256_file(perturbation_path),
        "drug_target_path": str(target_path) if target_path else None,
        "drug_target_sha256": sha256_file(target_path),
        "drug_metadata_path": str(metadata_path) if metadata_path else None,
        "drug_metadata_sha256": sha256_file(metadata_path),
        "network_edges_path": str(network_path) if network_path else None,
        "network_edges_sha256": sha256_file(network_path),
        "gene_alias_mapping_path": str(alias_path) if alias_path else None,
        "gene_alias_mapping_sha256": sha256_file(alias_path),
        "final_script_schema_version": FINAL_SCRIPT_SCHEMA_VERSION,
        "effective_settings": {
            "source_reversers_only": SOURCE_REVERSERS_ONLY,
            "minimum_query_context_sign_consistency": MIN_QUERY_CONTEXT_SIGN_CONSISTENCY,
            "network_drug_scope": NETWORK_DRUG_SCOPE,
            "network_mass_reference_quantile": NETWORK_MASS_REFERENCE_QUANTILE,
            "main_exclude_suspicious_only_target_rows": MAIN_EXCLUDE_SUSPICIOUS_ONLY_TARGET_ROWS,
            "sensitivity_suspicious_only_multiplier": SENSITIVITY_SUSPICIOUS_ONLY_MULTIPLIER,
            "sensitivity_mixed_source_multiplier": SENSITIVITY_MIXED_SOURCE_MULTIPLIER,
            "minimum_target_direction_consistency": MIN_TARGET_DIRECTION_CONSISTENCY,
        },
        "network_drug_scope": NETWORK_DRUG_SCOPE,
        "v3_tiers": V3_TIERS,
        "suspicious_target_sources": sorted(SUSPICIOUS_TARGET_SOURCES),
    })
    print(f"Drug pairing v3 completed: {output}")
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[37a FINAL ERROR] {type(exc).__name__}: {exc}")
        raise
