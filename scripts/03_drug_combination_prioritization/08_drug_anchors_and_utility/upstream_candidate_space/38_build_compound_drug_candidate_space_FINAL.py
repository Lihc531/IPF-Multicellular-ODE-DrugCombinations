#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_build_compound_drug_candidate_space_FINAL.py

基于 37a 的最终输出构建双药复方候选空间。

设计边界
--------
1. 本脚本构建的是“计算优先级候选空间”，不是药物协同作用证据。
2. 组合必须至少包含一个来源稳健的单臂 A 级候选作为锚点。
3. 全局 SigCom 查询证据只评价单药逆转质量，不在组合层面重复累加为
   两份独立疾病 signature 证据。
4. 组合优先级综合考虑：
   - 两个成员各自的查询质量；
   - 主分析轨中的 OT signature 机制覆盖；
   - 机制互补增益；
   - 来源稳健性；
   - 靶点非冗余性；
   - 疾病与细胞类型覆盖；
   - 机制方向冲突、靶点反向冲突和同家族冗余。
5. 缺失安全性、暴露、剂量、给药途径和相互作用信息时，只标记为待复核，
   不把“缺失”解释为“安全”。

主要输入（均来自同一个 37a 输出目录）
---------------------------------------
- 07_final_candidate_screening.csv
- 06_dual_track_mechanistic_scores_long.csv
- 02e_targets_main_consolidated.csv
- 04a_direct_target_alignment_main_detail.csv
- 01_signature_qc.csv

主要输出
--------
01_candidate_pool.csv
02_pair_candidate_space_all.csv
03_pair_priority_A.csv
03a_pair_priority_A_nonredundant.csv
04_pair_priority_B.csv
04a_pair_priority_B_nonredundant.csv
05_pair_exploratory.csv
06_excluded_pairs.csv
07_pair_signature_coverage_long.csv
08_pair_target_overlap_audit.csv
09_pair_space_qc.json
10_methods_and_score_definition.txt

依赖：pandas, numpy
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "28.2.0-FINAL-QUERY-ANCHOR-MECHANISM-PARTNER-PAIR-SPACE"
EPS = 1e-12

# =============================================================================
# PYCHARM 用户设置：通常只修改这里
# =============================================================================
INPUT_ROOT = r"E:\PythonProject\results\37a_DRUG_PAIRING_RESULTS"
OUTPUT_ROOT = r"E:\PythonProject\results\38_COMBINATION_CANDIDATE_SPACE"
EXISTING_OUTPUT_POLICY = "overwrite"  # overwrite | timestamp | error

# 复方空间必须至少包含一个来源稳健的 A 级锚点。
ANCHOR_TIERS = {
    "ONE_ARM_PRIORITY_A_SOURCE_ROBUST",
}

# 允许进入搭档池的层级。默认排除可疑机制、纯探索性和已排除药物。
PARTNER_TIERS = {
    "ONE_ARM_PRIORITY_A_SOURCE_ROBUST",
    "ONE_ARM_PRIORITY_A_CONTEXT_RELEVANT",
    "ONE_ARM_PRIORITY_B",
    "ONE_ARM_PRIORITY_B_MECHANISM_UNCERTAIN",
}

# 允许作为“机制补充搭档”进入复方空间，但不能单独作为 SigCom 逆转锚点。
TARGET_MECHANISM_PARTNER_CLASSES = {
    "TARGET_MECHANISM_PRIORITY_SOURCE_ROBUST",
}

# None 表示保留全部符合条件的搭档；设置整数可仅保留综合排名靠前者。
MAX_PARTNER_POOL_SIZE: Optional[int] = None

REQUIRE_AT_LEAST_ONE_ANCHOR = True
EXCLUDE_HARD_BLOCKED_DRUGS = True
EXCLUDE_MANUAL_REVIEW_DRUGS = True
EXCLUDE_SAME_FAMILY_PAIRS = True
EXCLUDE_OPPOSITE_TARGET_ACTION_PAIRS = True
MAX_ALLOWED_PRIMARY_MECHANISM_CONFLICTS = 1

# 37a 中定义的主要机制阈值。
MECHANISM_POSITIVE_CUT = 0.08
MECHANISM_CONFLICT_CUT = -0.08
MECHANISM_SCORE_CAP = 0.20
MECHANISM_GAIN_REFERENCE = 0.08

# 组合候选层级阈值。
PAIR_A_MIN_SCORE = 0.65
PAIR_A_MIN_MECHANISM_UNION = 0.45
PAIR_A_MIN_POSITIVE_PRIMARY_SIGNATURES = 3
PAIR_A_MIN_SOURCE_ROBUST_SIGNATURES = 2
PAIR_A_MIN_WEAKER_MEMBER_QUALITY = 0.35

PAIR_B_MIN_SCORE = 0.50
PAIR_B_MIN_MECHANISM_UNION = 0.25
PAIR_B_MIN_POSITIVE_PRIMARY_SIGNATURES = 2

# 组合总分权重，总和必须为 1。
PAIR_SCORE_WEIGHTS = {
    "member_strength_score": 0.25,
    "query_evidence_score": 0.15,
    "mechanism_union_score": 0.25,
    "mechanism_gain_score": 0.15,
    "source_robustness_score": 0.10,
    "target_complementarity_score": 0.05,
    "context_coverage_score": 0.05,
}

# 罚分；硬性排除在评分后单独执行。
PENALTIES = {
    "per_primary_mechanism_conflict": 0.05,
    "same_family": 0.10,
    "high_target_redundancy": 0.08,
    "both_members_mechanism_unstable": 0.05,
}
TARGET_REDUNDANCY_JACCARD_CUT = 0.70
# =============================================================================


REQUIRED_FILES = {
    "screening": "07_final_candidate_screening.csv",
    "mechanism": "06_dual_track_mechanistic_scores_long.csv",
    "targets": "02e_targets_main_consolidated.csv",
    "direct_detail": "04a_direct_target_alignment_main_detail.csv",
    "signature_qc": "01_signature_qc.csv",
    "mechanism_priority": "14a_target_mechanism_priority_source_robust.csv",
}


@dataclass(frozen=True)
class DrugTargetProfile:
    genes: frozenset[str]
    signs: Mapping[str, float]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_gene(value: Any) -> str:
    return normalize_text(value).upper()


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if np.isfinite(default):
        out = out.fillna(default)
    return out


def bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "": default,
    }
    text = series.fillna("").astype("string").str.strip().str.casefold()
    values = [bool(mapping.get(str(value), default)) for value in text.to_numpy()]
    return pd.Series(values, index=series.index, dtype=bool)


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def linear_scale(value: Any, lower: float, upper: float) -> float:
    x = float(value) if pd.notna(value) else np.nan
    if not np.isfinite(x) or upper <= lower:
        return 0.0
    return clip01((x - lower) / (upper - lower))


def qvalue_strength(value: Any, cap_log10: float = 6.0) -> float:
    q = float(value) if pd.notna(value) else np.nan
    if not np.isfinite(q) or q <= 0:
        return 1.0 if q == 0 else 0.0
    q = min(max(q, 1e-300), 1.0)
    return clip01(-math.log10(q) / cap_log10)


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
            raise ValueError(f"未知 EXISTING_OUTPUT_POLICY：{policy}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label} 缺少字段：{missing}")


def read_inputs(input_root: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_FILES.items():
        path = input_root / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少 37a 输出文件：{path}")
        frames[key] = pd.read_csv(path, low_memory=False)
    return frames


def validate_inputs(frames: Mapping[str, pd.DataFrame]) -> None:
    screening = frames["screening"]
    mechanism = frames["mechanism"]
    targets = frames["targets"]
    direct = frames["direct_detail"]
    sig_qc = frames["signature_qc"]
    mechanism_priority = frames["mechanism_priority"]

    require_columns(
        screening,
        [
            "drug_id",
            "drug_name",
            "candidate_tier",
            "drug_global_rank",
            "primary_query_global_score",
            "signature_query_reversal_score",
            "signature_query_qvalue_best_arm",
            "query_arm_balance",
            "mechanistic_support_score_main",
            "mechanism_source_stable",
            "context_relevance_score",
            "hard_safety_or_entity_exclusion",
            "manual_review_flag",
        ],
        "07_final_candidate_screening.csv",
    )
    require_columns(
        mechanism,
        [
            "drug_id",
            "signature_id",
            "analysis_role",
            "disease_scope",
            "celltype_scope",
            "mechanistic_signature_score",
            "mechanism_source_robust",
            "mechanism_source_conflict",
        ],
        "06_dual_track_mechanistic_scores_long.csv",
    )
    require_columns(
        targets,
        [
            "drug_id",
            "gene_symbol",
            "action_sign",
            "source_conflict_flag",
            "scoring_exclusion_flag",
        ],
        "02e_targets_main_consolidated.csv",
    )
    require_columns(
        direct,
        ["drug_id", "signature_id", "gene_symbol", "target_alignment"],
        "04a_direct_target_alignment_main_detail.csv",
    )
    require_columns(
        sig_qc,
        ["signature_id", "analysis_role", "n_genes", "n_tier_a_genes"],
        "01_signature_qc.csv",
    )
    require_columns(
        mechanism_priority,
        [
            "drug_id",
            "drug_name",
            "target_mechanism_score",
            "mechanistic_support_score_main",
            "mechanism_source_stable",
            "mechanism_candidate_class",
            "target_mechanism_rank",
            "hard_safety_or_entity_exclusion",
            "manual_review_flag",
        ],
        "14a_target_mechanism_priority_source_robust.csv",
    )

    if screening["drug_id"].duplicated().any():
        dup = screening.loc[screening["drug_id"].duplicated(False), "drug_id"].head().tolist()
        raise ValueError(f"07 中 drug_id 不唯一：{dup}")

    weight_sum = sum(float(v) for v in PAIR_SCORE_WEIGHTS.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"PAIR_SCORE_WEIGHTS 总和必须为 1，当前为 {weight_sum}")


def normalize_member_features(
    screening: pd.DataFrame,
    mechanism_priority: pd.DataFrame,
) -> pd.DataFrame:
    query = screening.copy()
    query["candidate_role"] = "QUERY_REVERSER"
    query["present_in_sigcom_query"] = True
    if "target_mechanism_score" not in query.columns:
        query["target_mechanism_score"] = np.nan
    if "mechanism_candidate_class" not in query.columns:
        query["mechanism_candidate_class"] = ""

    # 来源稳健的靶点机制型候选可作为搭档进入，但不作为查询逆转锚点。
    mech = mechanism_priority.copy()
    mech["drug_id"] = mech["drug_id"].map(normalize_text)
    mech = mech[~mech["drug_id"].isin(set(query["drug_id"].map(normalize_text)))].copy()
    mech["candidate_role"] = "TARGET_MECHANISM_PARTNER"
    mech["present_in_sigcom_query"] = False
    mech["candidate_tier"] = mech["mechanism_candidate_class"].map(normalize_text)
    mech["drug_global_rank"] = (
        numeric(mech["target_mechanism_rank"], 999999).astype(int) + 100000
    )

    # 将机制型搭档扩展到与 query 表一致的最小字段集合。
    template_columns = set(query.columns) | set(mech.columns)
    for col in template_columns:
        if col not in query.columns:
            query[col] = np.nan
        if col not in mech.columns:
            mech[col] = np.nan

    out = pd.concat(
        [query[list(template_columns)], mech[list(template_columns)]],
        ignore_index=True,
        sort=False,
    )
    out["drug_id"] = out["drug_id"].map(normalize_text)
    out["drug_name"] = out["drug_name"].map(normalize_text)
    out["candidate_tier"] = out["candidate_tier"].map(normalize_text)
    out["candidate_role"] = out["candidate_role"].map(normalize_text)
    out["mechanism_candidate_class"] = out["mechanism_candidate_class"].map(normalize_text)
    out["present_in_sigcom_query"] = bool_series(out["present_in_sigcom_query"])

    out["hard_safety_or_entity_exclusion"] = bool_series(
        out["hard_safety_or_entity_exclusion"]
    )
    out["manual_review_flag"] = bool_series(out["manual_review_flag"])
    out["mechanism_source_stable"] = bool_series(out["mechanism_source_stable"])
    if "translation_metadata_complete" not in out.columns:
        out["translation_metadata_complete"] = False
    else:
        out["translation_metadata_complete"] = bool_series(
            out["translation_metadata_complete"]
        )

    out["global_score_norm"] = out["primary_query_global_score"].map(
        lambda x: linear_scale(x, 0.55, 0.70)
    )
    out["query_reversal_norm"] = out["signature_query_reversal_score"].map(
        lambda x: linear_scale(x, 0.65, 0.80)
    )
    out["best_arm_q_strength"] = out["signature_query_qvalue_best_arm"].map(
        qvalue_strength
    )
    out["arm_balance_norm"] = out["query_arm_balance"].map(
        lambda x: linear_scale(x, 0.10, 0.60)
    )
    out["mechanism_main_norm"] = out["mechanistic_support_score_main"].map(
        lambda x: linear_scale(x, 0.0, 0.70)
    )
    out["target_mechanism_norm"] = out["target_mechanism_score"].map(
        lambda x: linear_scale(x, 0.0, 0.75)
    )
    out["context_relevance_norm"] = numeric(
        out["context_relevance_score"], 0.0
    ).clip(0.0, 1.0)

    query_mask = out["present_in_sigcom_query"]
    out["member_quality_score"] = 0.0
    out.loc[query_mask, "member_quality_score"] = (
        0.40 * out.loc[query_mask, "global_score_norm"]
        + 0.20 * out.loc[query_mask, "query_reversal_norm"]
        + 0.15 * out.loc[query_mask, "best_arm_q_strength"]
        + 0.10 * out.loc[query_mask, "arm_balance_norm"]
        + 0.10 * out.loc[query_mask, "mechanism_main_norm"]
        + 0.05 * out.loc[query_mask, "context_relevance_norm"]
    )

    mech_mask = ~query_mask
    robust_count_norm = numeric(
        out.get(
            "n_primary_source_robust_signatures",
            pd.Series(0.0, index=out.index),
        ),
        0.0,
    ).clip(lower=0.0, upper=6.0) / 6.0
    direct_count_norm = numeric(
        out.get(
            "n_primary_direct_supported_main",
            pd.Series(0.0, index=out.index),
        ),
        0.0,
    ).clip(lower=0.0, upper=3.0) / 3.0
    out.loc[mech_mask, "member_quality_score"] = (
        0.45 * out.loc[mech_mask, "target_mechanism_norm"]
        + 0.30 * out.loc[mech_mask, "mechanism_main_norm"]
        + 0.15 * robust_count_norm.loc[mech_mask]
        + 0.10 * direct_count_norm.loc[mech_mask]
    )
    out["member_quality_score"] = numeric(
        out["member_quality_score"], 0.0
    ).clip(0.0, 1.0)

    out["anchor_eligible"] = (
        out["present_in_sigcom_query"]
        & out["candidate_tier"].isin(ANCHOR_TIERS)
    )
    out["partner_eligible"] = (
        out["candidate_tier"].isin(PARTNER_TIERS)
        | out["mechanism_candidate_class"].isin(TARGET_MECHANISM_PARTNER_CLASSES)
    )

    if EXCLUDE_HARD_BLOCKED_DRUGS:
        out.loc[out["hard_safety_or_entity_exclusion"], ["anchor_eligible", "partner_eligible"]] = False
    if EXCLUDE_MANUAL_REVIEW_DRUGS:
        out.loc[out["manual_review_flag"], ["anchor_eligible", "partner_eligible"]] = False

    out = out.copy()  # defragment after adding derived columns
    out["pool_exclusion_reason"] = ""
    allowed_tier_or_class = (
        out["candidate_tier"].isin(PARTNER_TIERS)
        | out["mechanism_candidate_class"].isin(TARGET_MECHANISM_PARTNER_CLASSES)
    )
    out.loc[~allowed_tier_or_class, "pool_exclusion_reason"] += "tier_or_mechanism_class_not_allowed;"
    if EXCLUDE_HARD_BLOCKED_DRUGS:
        out.loc[out["hard_safety_or_entity_exclusion"], "pool_exclusion_reason"] += "hard_blocked_drug;"
    if EXCLUDE_MANUAL_REVIEW_DRUGS:
        out.loc[out["manual_review_flag"], "pool_exclusion_reason"] += "manual_review_drug;"

    if MAX_PARTNER_POOL_SIZE is not None:
        n = int(MAX_PARTNER_POOL_SIZE)
        if n <= 0:
            raise ValueError("MAX_PARTNER_POOL_SIZE 必须为正整数或 None")
        eligible = out[out["partner_eligible"]].sort_values(
            ["anchor_eligible", "member_quality_score", "drug_global_rank"],
            ascending=[False, False, True],
        )
        keep_ids = set(eligible.head(n)["drug_id"]) | set(
            out.loc[out["anchor_eligible"], "drug_id"]
        )
        mask = out["partner_eligible"] & ~out["drug_id"].isin(keep_ids)
        out.loc[mask, "partner_eligible"] = False
        out.loc[mask, "pool_exclusion_reason"] += "beyond_partner_pool_limit;"

    if out["drug_id"].duplicated().any():
        dup = out.loc[out["drug_id"].duplicated(False), "drug_id"].tolist()
        raise ValueError(f"合并查询与机制型候选后 drug_id 重复：{dup[:10]}")
    return out


def build_signature_registry(
    mechanism: pd.DataFrame,
    signature_qc: pd.DataFrame,
) -> pd.DataFrame:
    primary = signature_qc[signature_qc["analysis_role"].eq("PRIMARY")].copy()
    primary = primary.drop_duplicates("signature_id")
    # signature_name 同时存在于两个输入表时，避免 pandas 自动生成 _x/_y 后缀。
    primary = primary.drop(
        columns=["signature_name", "signature_class", "tier_mode", "disease_scope", "celltype_scope"],
        errors="ignore",
    )

    context = (
        mechanism[mechanism["analysis_role"].eq("PRIMARY")]
        [["signature_id", "signature_name", "disease_scope", "celltype_scope", "signature_role_weight"]]
        .drop_duplicates("signature_id")
    )
    registry = primary.merge(context, on="signature_id", how="left")
    registry["signature_role_weight"] = numeric(
        registry.get("signature_role_weight", pd.Series(1.0, index=registry.index)),
        1.0,
    ).clip(lower=0.0)
    if float(registry["signature_role_weight"].sum()) <= 0:
        registry["signature_role_weight"] = 1.0
    return registry.reset_index(drop=True)


def build_mechanism_maps(
    mechanism: pd.DataFrame,
    primary_signature_ids: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, bool]],
    Dict[str, Dict[str, bool]],
]:
    primary = mechanism[
        mechanism["analysis_role"].eq("PRIMARY")
        & mechanism["signature_id"].isin(primary_signature_ids)
    ].copy()
    primary["mechanistic_signature_score"] = numeric(
        primary["mechanistic_signature_score"], 0.0
    )
    primary["mechanism_source_robust"] = bool_series(primary["mechanism_source_robust"])
    primary["mechanism_source_conflict"] = bool_series(primary["mechanism_source_conflict"])

    score_map: Dict[str, Dict[str, float]] = {}
    robust_map: Dict[str, Dict[str, bool]] = {}
    conflict_map: Dict[str, Dict[str, bool]] = {}
    for drug_id, group in primary.groupby("drug_id", sort=False):
        score_map[str(drug_id)] = dict(
            zip(group["signature_id"].astype(str), group["mechanistic_signature_score"].astype(float))
        )
        robust_map[str(drug_id)] = dict(
            zip(group["signature_id"].astype(str), group["mechanism_source_robust"].astype(bool))
        )
        conflict_map[str(drug_id)] = dict(
            zip(group["signature_id"].astype(str), group["mechanism_source_conflict"].astype(bool))
        )
    return score_map, robust_map, conflict_map


def build_target_profiles(targets: pd.DataFrame) -> Dict[str, DrugTargetProfile]:
    work = targets.copy()
    work["gene_symbol"] = work["gene_symbol"].map(normalize_gene)
    work["action_sign"] = numeric(work["action_sign"], 0.0)
    work["source_conflict_flag"] = bool_series(work["source_conflict_flag"])
    work["scoring_exclusion_flag"] = bool_series(work["scoring_exclusion_flag"])
    work = work[
        work["gene_symbol"].ne("")
        & work["action_sign"].ne(0)
        & ~work["source_conflict_flag"]
        & ~work["scoring_exclusion_flag"]
    ].copy()

    profiles: Dict[str, DrugTargetProfile] = {}
    for drug_id, group in work.groupby("drug_id", sort=False):
        signs = {
            str(row.gene_symbol): float(np.sign(row.action_sign))
            for row in group.itertuples(index=False)
        }
        profiles[str(drug_id)] = DrugTargetProfile(
            genes=frozenset(signs),
            signs=signs,
        )
    return profiles


def build_aligned_direct_target_map(direct_detail: pd.DataFrame) -> Dict[str, Set[str]]:
    work = direct_detail.copy()
    work["gene_symbol"] = work["gene_symbol"].map(normalize_gene)
    work["target_alignment"] = numeric(work["target_alignment"], 0.0)
    work = work[work["gene_symbol"].ne("") & work["target_alignment"].gt(0)]
    return {
        str(drug_id): set(group["gene_symbol"])
        for drug_id, group in work.groupby("drug_id", sort=False)
    }


def split_tokens(value: Any) -> Set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    tokens: Set[str] = set()
    for part in text.replace("|", ";").replace(",", ";").split(";"):
        item = part.strip()
        if item:
            tokens.add(item)
    return tokens


def family_value(row: pd.Series) -> str:
    return normalize_text(row.get("drug_family_group", "")).casefold()


def target_action_profile(profile: DrugTargetProfile) -> str:
    if not profile.genes:
        return "NO_DIRECTIONAL_TARGET_PROFILE"
    return ";".join(
        f"{gene}:{int(np.sign(profile.signs.get(gene, 0.0))):+d}"
        for gene in sorted(profile.genes)
    )


def pair_id(drug_a: str, drug_b: str) -> str:
    a, b = sorted([drug_a, drug_b])
    return f"{a}__PLUS__{b}"


def calculate_pair(
    a: pd.Series,
    b: pd.Series,
    registry: pd.DataFrame,
    mech_scores: Mapping[str, Mapping[str, float]],
    mech_robust: Mapping[str, Mapping[str, bool]],
    mech_conflicts: Mapping[str, Mapping[str, bool]],
    target_profiles: Mapping[str, DrugTargetProfile],
    aligned_direct_targets: Mapping[str, Set[str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    drug_a = str(a["drug_id"])
    drug_b = str(b["drug_id"])
    name_a = str(a["drug_name"])
    name_b = str(b["drug_name"])

    quality_a = float(a["member_quality_score"])
    quality_b = float(b["member_quality_score"])
    member_strength = 0.60 * np.mean([quality_a, quality_b]) + 0.40 * min(quality_a, quality_b)

    query_member_scores: List[float] = []
    for member in [a, b]:
        if bool(member.get("present_in_sigcom_query", False)):
            query_member_scores.append(
                float(
                    np.mean(
                        [
                            float(member["query_reversal_norm"]),
                            float(member["best_arm_q_strength"]),
                            float(member["arm_balance_norm"]),
                        ]
                    )
                )
            )
    n_query_members = len(query_member_scores)
    if n_query_members == 2:
        query_evidence = (
            0.60 * float(np.mean(query_member_scores))
            + 0.40 * float(min(query_member_scores))
        )
    elif n_query_members == 1:
        # 机制型搭档没有 SigCom 证据；保留锚点查询证据但施加轻度覆盖折减。
        query_evidence = 0.875 * query_member_scores[0]
    else:
        query_evidence = 0.0

    score_a_map = mech_scores.get(drug_a, {})
    score_b_map = mech_scores.get(drug_b, {})
    robust_a_map = mech_robust.get(drug_a, {})
    robust_b_map = mech_robust.get(drug_b, {})
    conflict_a_map = mech_conflicts.get(drug_a, {})
    conflict_b_map = mech_conflicts.get(drug_b, {})

    coverage_rows: List[Dict[str, Any]] = []
    union_scaled: List[float] = []
    a_scaled: List[float] = []
    b_scaled: List[float] = []
    role_weights: List[float] = []
    positive_union_count = 0
    exclusive_positive_count = 0
    overlapping_positive_count = 0
    robust_union_count = 0
    direction_conflict_count = 0
    source_conflict_count = 0
    disease_covered: Set[str] = set()
    celltypes_covered: Set[str] = set()

    for sig in registry.itertuples(index=False):
        sid = str(sig.signature_id)
        sa = float(score_a_map.get(sid, 0.0))
        sb = float(score_b_map.get(sid, 0.0))
        pa = sa >= MECHANISM_POSITIVE_CUT
        pb = sb >= MECHANISM_POSITIVE_CUT
        na = sa <= MECHANISM_CONFLICT_CUT
        nb = sb <= MECHANISM_CONFLICT_CUT
        union_raw = max(0.0, sa, sb)
        union_norm = clip01(union_raw / MECHANISM_SCORE_CAP)
        a_norm = clip01(max(0.0, sa) / MECHANISM_SCORE_CAP)
        b_norm = clip01(max(0.0, sb) / MECHANISM_SCORE_CAP)
        weight = max(float(sig.signature_role_weight), EPS)

        union_scaled.append(union_norm)
        a_scaled.append(a_norm)
        b_scaled.append(b_norm)
        role_weights.append(weight)

        if pa or pb:
            positive_union_count += 1
            disease_covered |= split_tokens(sig.disease_scope)
            celltypes_covered |= split_tokens(sig.celltype_scope)
        if pa ^ pb:
            exclusive_positive_count += 1
        if pa and pb:
            overlapping_positive_count += 1
        if (pa and bool(robust_a_map.get(sid, False))) or (
            pb and bool(robust_b_map.get(sid, False))
        ):
            robust_union_count += 1
        if (pa and nb) or (pb and na):
            direction_conflict_count += 1
        if bool(conflict_a_map.get(sid, False)) or bool(conflict_b_map.get(sid, False)):
            source_conflict_count += 1

        supported_by = (
            "BOTH" if pa and pb else "A_ONLY" if pa else "B_ONLY" if pb else "NONE"
        )
        coverage_rows.append(
            {
                "pair_id": pair_id(drug_a, drug_b),
                "drug_a_id": drug_a,
                "drug_b_id": drug_b,
                "signature_id": sid,
                "signature_name": normalize_text(sig.signature_name),
                "disease_scope": normalize_text(sig.disease_scope),
                "celltype_scope": normalize_text(sig.celltype_scope),
                "drug_a_mechanistic_score": sa,
                "drug_b_mechanistic_score": sb,
                "pair_union_mechanistic_score_raw": union_raw,
                "pair_union_mechanistic_score_normalized": union_norm,
                "supported_by": supported_by,
                "drug_a_source_robust": bool(robust_a_map.get(sid, False)),
                "drug_b_source_robust": bool(robust_b_map.get(sid, False)),
                "direction_conflict": bool((pa and nb) or (pb and na)),
            }
        )

    weights_arr = np.asarray(role_weights, dtype=float)
    mechanism_union_score = float(np.average(union_scaled, weights=weights_arr))
    drug_a_mech_mean = float(np.average(a_scaled, weights=weights_arr))
    drug_b_mech_mean = float(np.average(b_scaled, weights=weights_arr))
    mechanism_gain_raw = max(0.0, mechanism_union_score - max(drug_a_mech_mean, drug_b_mech_mean))
    mechanism_gain_score = clip01(mechanism_gain_raw / MECHANISM_GAIN_REFERENCE)

    n_primary = max(len(registry), 1)
    source_robustness_score = clip01(robust_union_count / max(PAIR_A_MIN_SOURCE_ROBUST_SIGNATURES, 1))

    profile_a = target_profiles.get(drug_a, DrugTargetProfile(frozenset(), {}))
    profile_b = target_profiles.get(drug_b, DrugTargetProfile(frozenset(), {}))
    targets_a = set(profile_a.genes)
    targets_b = set(profile_b.genes)
    target_profile_a = target_action_profile(profile_a)
    target_profile_b = target_action_profile(profile_b)
    overlap = targets_a & targets_b
    union_targets = targets_a | targets_b
    same_action_overlap = {
        gene
        for gene in overlap
        if np.sign(profile_a.signs.get(gene, 0.0)) == np.sign(profile_b.signs.get(gene, 0.0))
    }
    opposite_action_overlap = {
        gene
        for gene in overlap
        if np.sign(profile_a.signs.get(gene, 0.0)) == -np.sign(profile_b.signs.get(gene, 0.0))
    }
    target_jaccard = len(overlap) / len(union_targets) if union_targets else np.nan
    if targets_a and targets_b:
        target_complementarity = 1.0 - float(target_jaccard)
        target_evidence_status = "BOTH_AVAILABLE"
    elif targets_a or targets_b:
        target_complementarity = 0.25
        target_evidence_status = "ONE_AVAILABLE"
    else:
        target_complementarity = 0.0
        target_evidence_status = "BOTH_MISSING"

    direct_a = aligned_direct_targets.get(drug_a, set())
    direct_b = aligned_direct_targets.get(drug_b, set())
    aligned_direct_union = direct_a | direct_b

    disease_norm = {x.casefold() for x in disease_covered}
    cell_norm = {x.casefold() for x in celltypes_covered}
    disease_completeness = np.mean(
        ["ipf" in disease_norm, "copd" in disease_norm]
    )
    celltype_completeness = np.mean(
        ["dendritic" in cell_norm, "macrophage" in cell_norm]
    )
    exclusive_fraction = exclusive_positive_count / n_primary
    context_coverage_score = clip01(
        0.40 * disease_completeness
        + 0.40 * celltype_completeness
        + 0.20 * exclusive_fraction
    )

    family_a = family_value(a)
    family_b = family_value(b)
    same_family = bool(family_a and family_b and family_a == family_b)
    high_target_redundancy = bool(
        np.isfinite(target_jaccard)
        and target_jaccard >= TARGET_REDUNDANCY_JACCARD_CUT
    )
    both_mechanism_unstable = not (
        bool(a["mechanism_source_stable"]) or bool(b["mechanism_source_stable"])
    )

    component_values = {
        "member_strength_score": clip01(member_strength),
        "query_evidence_score": clip01(query_evidence),
        "mechanism_union_score": clip01(mechanism_union_score),
        "mechanism_gain_score": clip01(mechanism_gain_score),
        "source_robustness_score": clip01(source_robustness_score),
        "target_complementarity_score": clip01(target_complementarity),
        "context_coverage_score": clip01(context_coverage_score),
    }
    base_score = sum(
        PAIR_SCORE_WEIGHTS[key] * component_values[key]
        for key in PAIR_SCORE_WEIGHTS
    )

    penalty = 0.0
    penalty += PENALTIES["per_primary_mechanism_conflict"] * direction_conflict_count
    if same_family:
        penalty += PENALTIES["same_family"]
    if high_target_redundancy:
        penalty += PENALTIES["high_target_redundancy"]
    if both_mechanism_unstable:
        penalty += PENALTIES["both_members_mechanism_unstable"]
    final_score = clip01(base_score - penalty)

    has_anchor = bool(a["anchor_eligible"] or b["anchor_eligible"])
    exclusion_reasons: List[str] = []
    if REQUIRE_AT_LEAST_ONE_ANCHOR and not has_anchor:
        exclusion_reasons.append("no_source_robust_A_anchor")
    if EXCLUDE_SAME_FAMILY_PAIRS and same_family:
        exclusion_reasons.append("same_drug_family")
    if EXCLUDE_OPPOSITE_TARGET_ACTION_PAIRS and opposite_action_overlap:
        exclusion_reasons.append("opposite_actions_on_shared_targets")
    if direction_conflict_count > MAX_ALLOWED_PRIMARY_MECHANISM_CONFLICTS:
        exclusion_reasons.append("too_many_primary_mechanism_direction_conflicts")

    excluded = bool(exclusion_reasons)
    weaker_quality = min(quality_a, quality_b)

    if excluded:
        pair_tier = "EXCLUDED"
        tier_reason = ";".join(exclusion_reasons)
        tier_order = 9
    elif (
        final_score >= PAIR_A_MIN_SCORE
        and mechanism_union_score >= PAIR_A_MIN_MECHANISM_UNION
        and positive_union_count >= PAIR_A_MIN_POSITIVE_PRIMARY_SIGNATURES
        and robust_union_count >= PAIR_A_MIN_SOURCE_ROBUST_SIGNATURES
        and weaker_quality >= PAIR_A_MIN_WEAKER_MEMBER_QUALITY
    ):
        pair_tier = "COMPUTATIONAL_PAIR_PRIORITY_A"
        tier_reason = "high_pair_score_and_source_robust_mechanistic_union"
        tier_order = 1
    elif (
        final_score >= PAIR_B_MIN_SCORE
        and mechanism_union_score >= PAIR_B_MIN_MECHANISM_UNION
        and positive_union_count >= PAIR_B_MIN_POSITIVE_PRIMARY_SIGNATURES
    ):
        pair_tier = "COMPUTATIONAL_PAIR_PRIORITY_B"
        tier_reason = "moderate_pair_score_and_mechanistic_union"
        tier_order = 2
    else:
        pair_tier = "COMPUTATIONAL_PAIR_EXPLORATORY"
        tier_reason = "insufficient_for_A_or_B_but_retained_for_audit"
        tier_order = 3

    translation_complete_a = bool(a.get("translation_metadata_complete", False))
    translation_complete_b = bool(b.get("translation_metadata_complete", False))
    translation_review_required = not (translation_complete_a and translation_complete_b)

    role_a = normalize_text(a.get("candidate_role", ""))
    role_b = normalize_text(b.get("candidate_role", ""))
    if role_a == "TARGET_MECHANISM_PARTNER" and role_b != "TARGET_MECHANISM_PARTNER":
        pair_nonredundant_group = f"{drug_b}|MECH_PROFILE|{target_profile_a}"
    elif role_b == "TARGET_MECHANISM_PARTNER" and role_a != "TARGET_MECHANISM_PARTNER":
        pair_nonredundant_group = f"{drug_a}|MECH_PROFILE|{target_profile_b}"
    else:
        pair_nonredundant_group = pair_id(drug_a, drug_b)

    row: Dict[str, Any] = {
        "pair_id": pair_id(drug_a, drug_b),
        "drug_a_id": drug_a,
        "drug_a_name": name_a,
        "drug_a_role": role_a,
        "drug_a_tier": str(a["candidate_tier"]),
        "drug_a_rank": int(a["drug_global_rank"]),
        "drug_a_member_quality": quality_a,
        "drug_b_id": drug_b,
        "drug_b_name": name_b,
        "drug_b_role": role_b,
        "drug_b_tier": str(b["candidate_tier"]),
        "drug_b_rank": int(b["drug_global_rank"]),
        "drug_b_member_quality": quality_b,
        "has_source_robust_A_anchor": has_anchor,
        "n_members_with_sigcom_query_evidence": n_query_members,
        **component_values,
        "pair_score_before_penalty": base_score,
        "pair_penalty": penalty,
        "pair_candidate_score": final_score,
        "n_primary_signatures": n_primary,
        "n_primary_positive_union": positive_union_count,
        "n_primary_exclusive_positive": exclusive_positive_count,
        "n_primary_overlapping_positive": overlapping_positive_count,
        "n_primary_source_robust_union": robust_union_count,
        "n_primary_direction_conflicts": direction_conflict_count,
        "n_primary_source_conflicts": source_conflict_count,
        "drug_a_mechanism_mean_normalized": drug_a_mech_mean,
        "drug_b_mechanism_mean_normalized": drug_b_mech_mean,
        "mechanism_gain_raw": mechanism_gain_raw,
        "diseases_covered": ";".join(sorted(disease_covered)),
        "celltypes_covered": ";".join(sorted(celltypes_covered)),
        "drug_a_target_action_profile": target_profile_a,
        "drug_b_target_action_profile": target_profile_b,
        "pair_nonredundant_group": pair_nonredundant_group,
        "n_drug_a_targets_main": len(targets_a),
        "n_drug_b_targets_main": len(targets_b),
        "n_union_targets_main": len(union_targets),
        "n_shared_targets_main": len(overlap),
        "target_jaccard_main": target_jaccard,
        "shared_targets_main": ";".join(sorted(overlap)),
        "shared_targets_same_action": ";".join(sorted(same_action_overlap)),
        "shared_targets_opposite_action": ";".join(sorted(opposite_action_overlap)),
        "target_evidence_status": target_evidence_status,
        "n_aligned_direct_ot_targets_union": len(aligned_direct_union),
        "aligned_direct_ot_targets_union": ";".join(sorted(aligned_direct_union)),
        "same_drug_family": same_family,
        "high_target_redundancy": high_target_redundancy,
        "both_members_mechanism_unstable": both_mechanism_unstable,
        "translation_review_required": translation_review_required,
        "pair_tier": pair_tier,
        "pair_tier_reason": tier_reason,
        "pair_tier_order": tier_order,
        "pair_exclusion_reasons": ";".join(exclusion_reasons),
        "interpretation_boundary": "computational_candidate_space_not_synergy_or_safety_evidence",
    }

    target_audit = {
        "pair_id": row["pair_id"],
        "drug_a_id": drug_a,
        "drug_b_id": drug_b,
        "drug_a_target_action_profile": target_profile_a,
        "drug_b_target_action_profile": target_profile_b,
        "pair_nonredundant_group": pair_nonredundant_group,
        "n_drug_a_targets_main": len(targets_a),
        "n_drug_b_targets_main": len(targets_b),
        "n_union_targets_main": len(union_targets),
        "n_shared_targets_main": len(overlap),
        "target_jaccard_main": target_jaccard,
        "shared_targets_main": ";".join(sorted(overlap)),
        "shared_targets_same_action": ";".join(sorted(same_action_overlap)),
        "shared_targets_opposite_action": ";".join(sorted(opposite_action_overlap)),
        "target_evidence_status": target_evidence_status,
        "same_drug_family": same_family,
        "high_target_redundancy": high_target_redundancy,
    }
    return row, coverage_rows, target_audit


def write_methods(path: Path) -> None:
    text = f"""28 COMBINATION CANDIDATE SPACE METHODS
======================================
Script version: {SCRIPT_VERSION}

Interpretation boundary
-----------------------
This output is a computational drug-pair candidate space. It is not evidence of
pharmacological synergy, clinical efficacy, exposure compatibility, or safety.

Pool construction
-----------------
Anchor tiers: {sorted(ANCHOR_TIERS)}
Partner tiers: {sorted(PARTNER_TIERS)}
Target-mechanism partner classes: {sorted(TARGET_MECHANISM_PARTNER_CLASSES)}
Require at least one anchor: {REQUIRE_AT_LEAST_ONE_ANCHOR}
Exclude hard-blocked drugs: {EXCLUDE_HARD_BLOCKED_DRUGS}
Exclude manual-review drugs: {EXCLUDE_MANUAL_REVIEW_DRUGS}
Exclude same-family pairs: {EXCLUDE_SAME_FAMILY_PAIRS}
Exclude opposite actions on shared targets: {EXCLUDE_OPPOSITE_TARGET_ACTION_PAIRS}

Scoring components
------------------
{json.dumps(PAIR_SCORE_WEIGHTS, ensure_ascii=False, indent=2)}

Mechanism definitions
---------------------
Positive primary-signature mechanism score: >= {MECHANISM_POSITIVE_CUT}
Conflicting primary-signature mechanism score: <= {MECHANISM_CONFLICT_CUT}
Mechanism score normalization cap: {MECHANISM_SCORE_CAP}
Mechanism gain reference: {MECHANISM_GAIN_REFERENCE}

Pair A threshold
----------------
score >= {PAIR_A_MIN_SCORE}
mechanism union >= {PAIR_A_MIN_MECHANISM_UNION}
positive primary signatures >= {PAIR_A_MIN_POSITIVE_PRIMARY_SIGNATURES}
source-robust primary signatures >= {PAIR_A_MIN_SOURCE_ROBUST_SIGNATURES}
weaker member quality >= {PAIR_A_MIN_WEAKER_MEMBER_QUALITY}

Pair B threshold
----------------
score >= {PAIR_B_MIN_SCORE}
mechanism union >= {PAIR_B_MIN_MECHANISM_UNION}
positive primary signatures >= {PAIR_B_MIN_POSITIVE_PRIMARY_SIGNATURES}

Important
---------
The global SigCom query is shared by all drugs and is not treated as two
independent signatures at the pair level. Mechanism complementarity is derived
from the six PRIMARY OT signatures in 37a. Missing translation metadata triggers
review and is never interpreted as evidence of safety.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    input_root = Path(os.path.expanduser(os.path.expandvars(INPUT_ROOT)))
    output_root = choose_output_dir(
        Path(os.path.expanduser(os.path.expandvars(OUTPUT_ROOT))),
        EXISTING_OUTPUT_POLICY,
    )

    if not input_root.exists():
        raise FileNotFoundError(f"找不到 37a 输出目录：{input_root}")

    frames = read_inputs(input_root)
    validate_inputs(frames)

    candidates = normalize_member_features(
        frames["screening"],
        frames["mechanism_priority"],
    )
    registry = build_signature_registry(frames["mechanism"], frames["signature_qc"])
    primary_ids = registry["signature_id"].astype(str).tolist()
    mech_scores, mech_robust, mech_conflicts = build_mechanism_maps(
        frames["mechanism"], primary_ids
    )
    target_profiles = build_target_profiles(frames["targets"])
    aligned_direct_targets = build_aligned_direct_target_map(frames["direct_detail"])

    pool = candidates[candidates["partner_eligible"]].copy()
    pool = pool.sort_values(
        ["anchor_eligible", "member_quality_score", "drug_global_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    if pool.empty:
        raise ValueError("搭档池为空。请检查候选层级或排除设置。")
    if REQUIRE_AT_LEAST_ONE_ANCHOR and not bool(pool["anchor_eligible"].any()):
        raise ValueError("没有可用的来源稳健 A 级锚点。")

    pair_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    target_audits: List[Dict[str, Any]] = []

    for i, j in combinations(range(len(pool)), 2):
        a = pool.iloc[i]
        b = pool.iloc[j]
        if REQUIRE_AT_LEAST_ONE_ANCHOR and not (
            bool(a["anchor_eligible"]) or bool(b["anchor_eligible"])
        ):
            continue
        row, coverage, target_audit = calculate_pair(
            a,
            b,
            registry,
            mech_scores,
            mech_robust,
            mech_conflicts,
            target_profiles,
            aligned_direct_targets,
        )
        pair_rows.append(row)
        coverage_rows.extend(coverage)
        target_audits.append(target_audit)

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise ValueError("未生成任何组合。")

    pair_df = pair_df.sort_values(
        ["pair_tier_order", "pair_candidate_score", "mechanism_union_score", "member_strength_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    pair_df["pair_global_rank"] = np.arange(1, len(pair_df) + 1)

    coverage_df = pd.DataFrame(coverage_rows)
    target_audit_df = pd.DataFrame(target_audits)

    # 保持候选池字段精简且可追溯。
    pool_columns = [
        "drug_id",
        "drug_name",
        "candidate_role",
        "candidate_tier",
        "mechanism_candidate_class",
        "present_in_sigcom_query",
        "drug_global_rank",
        "target_mechanism_score",
        "primary_query_global_score",
        "signature_query_reversal_score",
        "signature_query_qvalue_strict",
        "signature_query_qvalue_best_arm",
        "query_arm_balance",
        "mechanistic_support_score_main",
        "mechanistic_support_score_sensitivity",
        "mechanism_source_stable",
        "context_relevance_score",
        "member_quality_score",
        "anchor_eligible",
        "partner_eligible",
        "hard_safety_or_entity_exclusion",
        "manual_review_flag",
        "translation_metadata_complete",
        "drug_family_group",
        "pool_exclusion_reason",
    ]
    pool_columns = [c for c in pool_columns if c in candidates.columns]
    candidates[pool_columns].sort_values(
        ["partner_eligible", "anchor_eligible", "member_quality_score"],
        ascending=[False, False, False],
    ).to_csv(output_root / "01_candidate_pool.csv", index=False)

    pair_df.to_csv(output_root / "02_pair_candidate_space_all.csv", index=False)
    priority_a = pair_df[pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_PRIORITY_A")].copy()
    priority_b = pair_df[pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_PRIORITY_B")].copy()
    priority_a.to_csv(output_root / "03_pair_priority_A.csv", index=False)
    priority_a.drop_duplicates("pair_nonredundant_group", keep="first").to_csv(
        output_root / "03a_pair_priority_A_nonredundant.csv", index=False
    )
    priority_b.to_csv(output_root / "04_pair_priority_B.csv", index=False)
    priority_b.drop_duplicates("pair_nonredundant_group", keep="first").to_csv(
        output_root / "04a_pair_priority_B_nonredundant.csv", index=False
    )
    pair_df[pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_EXPLORATORY")].to_csv(
        output_root / "05_pair_exploratory.csv", index=False
    )
    pair_df[pair_df["pair_tier"].eq("EXCLUDED")].to_csv(
        output_root / "06_excluded_pairs.csv", index=False
    )
    coverage_df.to_csv(output_root / "07_pair_signature_coverage_long.csv", index=False)
    target_audit_df.to_csv(output_root / "08_pair_target_overlap_audit.csv", index=False)

    qc = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "run_time": datetime.now().isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "n_screened_drugs_input": int(len(candidates)),
        "n_partner_pool_drugs": int(pool["drug_id"].nunique()),
        "n_query_reverser_pool_drugs": int(
            pool.loc[pool["candidate_role"].eq("QUERY_REVERSER"), "drug_id"].nunique()
        ),
        "n_target_mechanism_partner_drugs": int(
            pool.loc[pool["candidate_role"].eq("TARGET_MECHANISM_PARTNER"), "drug_id"].nunique()
        ),
        "n_anchor_drugs": int(pool.loc[pool["anchor_eligible"], "drug_id"].nunique()),
        "n_primary_signatures": int(len(registry)),
        "n_pairs_generated": int(len(pair_df)),
        "n_pair_priority_A": int(pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_PRIORITY_A").sum()),
        "n_pair_priority_A_nonredundant": int(
            priority_a["pair_nonredundant_group"].nunique()
        ),
        "n_pair_priority_B": int(pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_PRIORITY_B").sum()),
        "n_pair_priority_B_nonredundant": int(
            priority_b["pair_nonredundant_group"].nunique()
        ),
        "n_pair_exploratory": int(pair_df["pair_tier"].eq("COMPUTATIONAL_PAIR_EXPLORATORY").sum()),
        "n_pairs_excluded": int(pair_df["pair_tier"].eq("EXCLUDED").sum()),
        "n_pairs_translation_review_required": int(pair_df["translation_review_required"].sum()),
        "n_pairs_with_opposite_shared_target_actions": int(
            pair_df["shared_targets_opposite_action"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "n_pairs_same_family": int(pair_df["same_drug_family"].sum()),
        "n_pairs_high_target_redundancy": int(pair_df["high_target_redundancy"].sum()),
        "score_min": float(pair_df["pair_candidate_score"].min()),
        "score_max": float(pair_df["pair_candidate_score"].max()),
        "interpretation": (
            "Computational pair-candidate space only. Priority labels require "
            "pharmacology, exposure, drug-drug interaction, route, dose, and safety review."
        ),
    }
    (output_root / "09_pair_space_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_methods(output_root / "10_methods_and_score_definition.txt")

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print("\n完成。优先查看：")
    print(output_root / "03a_pair_priority_A_nonredundant.csv")
    print(output_root / "04a_pair_priority_B_nonredundant.csv")
    print(output_root / "09_pair_space_qc.json")


if __name__ == "__main__":
    main()
