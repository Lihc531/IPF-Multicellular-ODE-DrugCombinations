# -*- coding: utf-8 -*-
"""
脚本41（修正版）：药物组合最终多证据筛选
==========================================

修正重点（v41.1.0）
-------------------
1. 修复 translation_review_required 全量为 True 时，全部组合被转入人工复核、
   最终短名单为空的问题。
2. 默认采用 adaptive 翻译/身份复核策略：
   - 当 translation_review_required 在 >=95% 组合中为 True 时，判定该字段为
     “全局提示、无逐组合区分度”；此时不再单独阻断候选。
   - 但药物ID或名称缺失时，仍强制进入人工复核。
3. 保留原始字段，并新增：
   translation_review_raw、identity_fields_missing、
   translation_review_effective、translation_flag_global_suppressed。
4. QC增强：PASS=0或shortlist=0时，status不再错误显示ok，而是failed_screening。
5. 最终短名单仍仅从PASS的非冗余代表中选择；本修正不放宽硬排除、证据分歧、
   方向冲突、来源冲突等其他规则。

解释边界
--------
输出是计算优先级，不是Bliss/Loewe/ZIP/HSA协同分数，不代表联合安全、最佳剂量、
药代相容或临床有效性。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "41.1.0-FINAL-MULTI-EVIDENCE-SCREENING-FIXED"

# ============================================================
# 可直接在 PyCharm 中修改的默认路径
# ============================================================
SCRIPT39_ROOT = Path(r"E:\PythonProject\results\39_OT_FLOW_MATCHING_COMBINATION_MODEL")
SCRIPT40_ROOT = Path(r"E:\PythonProject\results\40_OT_FLOW_MATCHING_ABLATION")
OUTPUT_ROOT = Path(r"E:\PythonProject\results\41_FINAL_COMBINATION_SCREENING_FIXED")


@dataclass(frozen=True)
class ScreeningConfig:
    # 多证据权重；每个组合按实际可用证据自动重新归一化。
    weight_original_rule: float = 0.25
    weight_target_centroid: float = 0.25
    weight_sinkhorn_ot: float = 0.30
    weight_flow_stability: float = 0.20

    # 证据阈值。
    high_evidence_threshold: float = 0.80
    moderate_evidence_threshold: float = 0.65
    priority_percentile_threshold: float = 0.70
    discordance_review_threshold: float = 0.50

    # 冲突阈值。
    direction_conflict_review_threshold: int = 2
    source_conflict_review_threshold: int = 2

    # 人工复核惩罚仅用于排序，不替代人工复核。
    review_penalty_per_flag: float = 0.03
    max_review_penalty: float = 0.12

    # 翻译/药物身份复核策略。
    # adaptive: 全局无区分度时只保留真实身份字段缺失复核（推荐）
    # always: 始终按上游translation_review_required逐组合复核
    # missing_identity_only: 仅ID/名称缺失时复核
    # ignore: 完全忽略该类复核（不推荐）
    translation_review_mode: str = "adaptive"
    translation_global_flag_threshold: float = 0.95

    # 短名单。
    shortlist_size: int = 20
    confirmatory_quota: int = 8
    model_enhanced_quota: int = 5
    exploratory_quota: int = 5
    max_appearances_per_drug: int = 4

    # balanced / strict / permissive
    filter_policy: str = "balanced"

    # abort / overwrite
    output_policy: str = "abort"


EVIDENCE_COLUMNS = {
    "rule": "original_pair_candidate_score",
    "centroid": "target_centroid_score",
    "ot": "ot_baseline_score",
}

DERIVED_COLUMNS_TO_DROP = {
    "experiment",
    "flow_mean_rank",
    "flow_rank_std",
    "flow_top10_frequency",
    "flow_top20_frequency",
    "flow_top50_frequency",
    "flow_rank_percentile",
    "flow_seed_score_mean",
    "flow_seed_score_std",
    "flow_seed_rank_mean_from_long",
    "flow_seed_rank_std_from_long",
    "flow_seed_count",
    "flow_seed_score_percentile",
    "flow_stability_score",
    "selected_flow_experiment",
    "translation_review_raw",
    "identity_fields_missing",
    "translation_review_effective",
    "translation_flag_global_suppressed",
    "review_reasons_pre_score",
    "is_hard_excluded",
    "review_flag_count_pre_score",
    "rule_evidence_percentile",
    "centroid_evidence_percentile",
    "ot_evidence_percentile",
    "flow_evidence_percentile",
    "raw_consensus_score",
    "available_evidence_weight",
    "n_available_evidence",
    "evidence_min",
    "evidence_max",
    "evidence_spread",
    "evidence_agreement",
    "high_evidence_count",
    "moderate_evidence_count",
    "transport_high_count",
    "evidence_discordance_flag",
    "review_flag_count",
    "is_manual_review",
    "review_penalty",
    "final_priority_score",
    "final_rank_eligible",
    "final_priority_percentile",
    "final_rank_all",
    "candidate_track",
    "review_status",
    "screening_rationale",
    "hard_exclusion_reasons",
    "review_reasons",
}


# ============================================================
# 参数与通用函数
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="修正版最终多证据、非冗余药物组合筛选。"
    )
    parser.add_argument("--script39-root", "--script29-root", dest="script39_root", type=Path, default=SCRIPT39_ROOT)
    parser.add_argument("--script40-root", "--script30-root", dest="script40_root", type=Path, default=SCRIPT40_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--filter-policy",
        choices=["balanced", "strict", "permissive"],
        default="balanced",
    )
    parser.add_argument(
        "--translation-review-mode",
        choices=["adaptive", "always", "missing_identity_only", "ignore"],
        default="adaptive",
        help="推荐adaptive；修复全量translation_review_required阻断问题。",
    )
    parser.add_argument(
        "--translation-global-flag-threshold",
        type=float,
        default=0.95,
        help="adaptive模式下，原始翻译标记达到该比例时视为全局无区分度提示。",
    )
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument("--confirmatory-quota", type=int, default=8)
    parser.add_argument("--model-enhanced-quota", type=int, default=5)
    parser.add_argument("--exploratory-quota", type=int, default=5)
    parser.add_argument("--max-appearances-per-drug", type=int, default=4)
    parser.add_argument(
        "--output-policy", choices=["abort", "overwrite"], default="abort"
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, policy: str) -> None:
    if path.exists() and any(path.iterdir()):
        if policy == "abort":
            raise FileExistsError(
                f"输出目录非空：{path}\n请更换目录或设置 --output-policy overwrite。"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件：{path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"文件为空：{path}")
    return df


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label}缺少必要字段：{missing}")


def first_existing(candidates: Sequence[str], available: Iterable[str]) -> str | None:
    available_set = set(available)
    return next((c for c in candidates if c in available_set), None)


def as_bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    s = df[column]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(int(default)).ne(0)
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "是": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
                "否": False,
                "nan": default,
                "none": default,
                "": default,
            }
        )
    )
    return mapped.fillna(default).astype(bool)


def numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_name(value: Any) -> str:
    text = safe_text(value).upper()
    for old, new in [("-", "_"), (" ", "_"), ("/", "_"), ("+", "_")]:
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def high_is_good_percentile(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna()
    if valid.sum() == 0:
        return out
    if x.loc[valid].nunique(dropna=True) == 1:
        out.loc[valid] = 0.5
    else:
        out.loc[valid] = x.loc[valid].rank(method="average", pct=True)
    return out


def low_is_good_rank_percentile(mean_rank: pd.Series, n_items: int) -> pd.Series:
    rank = pd.to_numeric(mean_rank, errors="coerce")
    if n_items <= 1:
        return pd.Series(1.0, index=rank.index)
    return (1.0 - (rank - 1.0) / float(n_items - 1)).clip(0.0, 1.0)


def join_reasons(flags: Sequence[tuple[pd.Series, str]], index: pd.Index) -> pd.Series:
    values: list[str] = []
    for idx in index:
        reasons = [label for mask, label in flags if bool(mask.loc[idx])]
        values.append(";".join(reasons))
    return pd.Series(values, index=index, dtype="object")


def prepare_base_input(df: pd.DataFrame) -> pd.DataFrame:
    """允许误将旧版01_final_scores_all_pairs.csv作为输入时自动清理派生列。"""
    drop_cols = [c for c in DERIVED_COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# ============================================================
# Flow多随机种子证据
# ============================================================
def extract_best_flow_experiment(config: Mapping[str, Any]) -> str | None:
    for key in ["best_flow_experiment", "best_flow_model", "best_flow_model_name", "best_flow"]:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            for subkey in ["experiment", "name", "model"]:
                subvalue = value.get(subkey)
                if isinstance(subvalue, str) and subvalue.strip():
                    return subvalue.strip()
    return None


def choose_flow_experiment(frequency: pd.DataFrame, best_config: Mapping[str, Any]) -> str:
    require_columns(frequency, ["experiment", "pair_id"], "07_topk_selection_frequency.csv")
    available = frequency["experiment"].dropna().astype(str).unique().tolist()
    preferred = extract_best_flow_experiment(best_config)
    for candidate in [preferred, "cfm_nearest_pair", "cfm_ot_full"]:
        if candidate and candidate in available:
            return candidate
    cfm = sorted(x for x in available if x.startswith("cfm_"))
    if not cfm:
        raise ValueError("Top-K频率文件中未找到CFM实验。")
    return cfm[0]


def aggregate_flow_long_scores(long_scores: pd.DataFrame, experiment: str) -> pd.DataFrame:
    if long_scores.empty:
        return pd.DataFrame(columns=["pair_id"])
    exp_col = first_existing(["experiment", "model", "method"], long_scores.columns)
    pair_col = first_existing(["pair_id", "combination_id"], long_scores.columns)
    seed_col = first_existing(["seed", "random_seed"], long_scores.columns)
    score_col = first_existing(
        ["score", "model_score", "prediction_score", "pair_score", "flow_score"],
        long_scores.columns,
    )
    rank_col = first_existing(["rank", "score_rank", "model_rank"], long_scores.columns)
    if exp_col is None or pair_col is None:
        return pd.DataFrame(columns=["pair_id"])

    part = long_scores.loc[long_scores[exp_col].astype(str).eq(experiment)].copy()
    if part.empty:
        return pd.DataFrame(columns=["pair_id"])

    agg_spec: dict[str, tuple[str, str]] = {}
    if score_col is not None:
        part[score_col] = pd.to_numeric(part[score_col], errors="coerce")
        agg_spec["flow_seed_score_mean"] = (score_col, "mean")
        agg_spec["flow_seed_score_std"] = (score_col, "std")
    if rank_col is not None:
        part[rank_col] = pd.to_numeric(part[rank_col], errors="coerce")
        agg_spec["flow_seed_rank_mean_from_long"] = (rank_col, "mean")
        agg_spec["flow_seed_rank_std_from_long"] = (rank_col, "std")
    if seed_col is not None:
        agg_spec["flow_seed_count"] = (seed_col, "nunique")
    else:
        part["__row_count"] = 1
        agg_spec["flow_seed_count"] = ("__row_count", "sum")

    if not agg_spec:
        return pd.DataFrame(columns=["pair_id"])
    out = part.groupby(pair_col, as_index=False).agg(**agg_spec)
    return out.rename(columns={pair_col: "pair_id"})


def build_flow_evidence(
    frequency: pd.DataFrame,
    long_scores: pd.DataFrame,
    experiment: str,
    n_items: int,
) -> pd.DataFrame:
    required = [
        "experiment",
        "pair_id",
        "mean_rank",
        "rank_std",
        "top10_frequency",
        "top20_frequency",
        "top50_frequency",
    ]
    require_columns(frequency, required, "07_topk_selection_frequency.csv")
    part = frequency.loc[frequency["experiment"].astype(str).eq(experiment), required].copy()
    if part.empty:
        raise ValueError(f"未找到Flow实验Top-K频率：{experiment}")
    if part["pair_id"].duplicated().any():
        raise ValueError(f"Flow频率存在重复pair_id：{experiment}")

    for col in ["mean_rank", "rank_std", "top10_frequency", "top20_frequency", "top50_frequency"]:
        part[col] = pd.to_numeric(part[col], errors="coerce")
    part = part.rename(
        columns={
            "mean_rank": "flow_mean_rank",
            "rank_std": "flow_rank_std",
            "top10_frequency": "flow_top10_frequency",
            "top20_frequency": "flow_top20_frequency",
            "top50_frequency": "flow_top50_frequency",
        }
    )
    part["flow_rank_percentile"] = low_is_good_rank_percentile(part["flow_mean_rank"], n_items)

    long_agg = aggregate_flow_long_scores(long_scores, experiment)
    if not long_agg.empty:
        part = part.merge(long_agg, on="pair_id", how="left", validate="one_to_one")
        if "flow_seed_score_mean" in part.columns:
            part["flow_seed_score_percentile"] = high_is_good_percentile(part["flow_seed_score_mean"])

    components: list[tuple[str, float]] = [
        ("flow_rank_percentile", 0.55),
        ("flow_top20_frequency", 0.25),
        ("flow_top50_frequency", 0.15),
        ("flow_top10_frequency", 0.05),
    ]
    if "flow_seed_score_percentile" in part.columns:
        components = [
            ("flow_rank_percentile", 0.45),
            ("flow_seed_score_percentile", 0.15),
            ("flow_top20_frequency", 0.20),
            ("flow_top50_frequency", 0.15),
            ("flow_top10_frequency", 0.05),
        ]

    numerator = pd.Series(0.0, index=part.index)
    denominator = pd.Series(0.0, index=part.index)
    for col, weight in components:
        values = pd.to_numeric(part[col], errors="coerce")
        valid = values.notna()
        numerator.loc[valid] += weight * values.loc[valid]
        denominator.loc[valid] += weight
    part["flow_stability_score"] = numerator.div(denominator.replace(0, np.nan)).clip(0, 1)
    part["selected_flow_experiment"] = experiment
    return part


# ============================================================
# 修正后的身份/翻译复核逻辑
# ============================================================
def build_translation_review_flags(
    df: pd.DataFrame,
    config: ScreeningConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    raw = as_bool_series(df, "translation_review_required")

    missing_masks: list[pd.Series] = []
    for col in ["drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name"]:
        if col in df.columns:
            missing_masks.append(df[col].isna() | df[col].astype(str).str.strip().isin(["", "nan", "None"]))
        else:
            # 缺少整个关键字段时视为身份信息不完整。
            missing_masks.append(pd.Series(True, index=df.index, dtype=bool))
    identity_missing = missing_masks[0].copy()
    for mask in missing_masks[1:]:
        identity_missing |= mask

    raw_true_fraction = float(raw.mean()) if len(raw) else 0.0
    globally_uninformative = (
        len(raw) > 0
        and raw_true_fraction >= config.translation_global_flag_threshold
    )

    mode = config.translation_review_mode
    if mode == "adaptive":
        if globally_uninformative:
            effective = identity_missing.copy()
        else:
            effective = raw | identity_missing
    elif mode == "always":
        effective = raw | identity_missing
    elif mode == "missing_identity_only":
        effective = identity_missing.copy()
    elif mode == "ignore":
        effective = pd.Series(False, index=df.index, dtype=bool)
    else:
        raise ValueError(f"未知translation_review_mode：{mode}")

    suppressed = raw & ~effective & globally_uninformative

    diagnostics = {
        "mode": mode,
        "raw_true_count": int(raw.sum()),
        "raw_true_fraction": raw_true_fraction,
        "identity_missing_count": int(identity_missing.sum()),
        "effective_review_count": int(effective.sum()),
        "global_flag_threshold": float(config.translation_global_flag_threshold),
        "globally_uninformative": bool(globally_uninformative),
        "suppressed_count": int(suppressed.sum()),
    }

    df["translation_review_raw"] = raw
    df["identity_fields_missing"] = identity_missing
    df["translation_review_effective"] = effective
    df["translation_flag_global_suppressed"] = suppressed
    return effective, diagnostics


# ============================================================
# 过滤、评分和候选分层
# ============================================================
def apply_quality_flags(
    df: pd.DataFrame,
    config: ScreeningConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    excluded = as_bool_series(out, "is_excluded_pair")
    if "pair_tier" in out.columns:
        excluded |= out["pair_tier"].astype(str).str.upper().str.contains("EXCLUDED", na=False)

    same_family = as_bool_series(out, "same_drug_family")
    high_redundancy = as_bool_series(out, "high_target_redundancy")
    both_unstable = as_bool_series(out, "both_members_mechanism_unstable")
    translation_effective, translation_diagnostics = build_translation_review_flags(out, config)
    missing_flow = out["flow_stability_score"].isna()

    direction_conflicts = numeric_series(out, "n_primary_direction_conflicts", default=0).fillna(0)
    source_conflicts = numeric_series(out, "n_primary_source_conflicts", default=0).fillna(0)
    direction_review = direction_conflicts.ge(config.direction_conflict_review_threshold)
    source_review = source_conflicts.ge(config.source_conflict_review_threshold)

    hard_flags: list[tuple[pd.Series, str]] = [(excluded, "原始排除组合")]
    policy_flags = [
        (same_family, "同药物家族"),
        (high_redundancy, "高靶点冗余"),
        (both_unstable, "双方机制均不稳定"),
    ]
    soft_flags = [
        (translation_effective, "药物身份/翻译需复核"),
        (direction_review, "方向冲突较多"),
        (source_review, "来源冲突较多"),
        (missing_flow, "缺少Flow多种子稳定性"),
    ]
    review_flags: list[tuple[pd.Series, str]] = []

    if config.filter_policy == "balanced":
        hard_flags.extend(policy_flags)
        review_flags.extend(soft_flags)
    elif config.filter_policy == "strict":
        hard_flags.extend(policy_flags + soft_flags)
    elif config.filter_policy == "permissive":
        review_flags.extend(policy_flags + soft_flags)
    else:
        raise ValueError(f"未知filter_policy：{config.filter_policy}")

    out["hard_exclusion_reasons"] = join_reasons(hard_flags, out.index)
    out["review_reasons_pre_score"] = join_reasons(review_flags, out.index)
    out["is_hard_excluded"] = out["hard_exclusion_reasons"].ne("")
    out["review_flag_count_pre_score"] = pd.Series(0, index=out.index, dtype=int)
    for mask, _ in review_flags:
        out["review_flag_count_pre_score"] += mask.astype(int)
    return out, translation_diagnostics


def weighted_available_mean(
    df: pd.DataFrame,
    weighted_columns: Sequence[tuple[str, float]],
) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for col, weight in weighted_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.notna()
        numerator.loc[valid] += weight * values.loc[valid]
        denominator.loc[valid] += weight
    return numerator.div(denominator.replace(0, np.nan)), denominator


def add_evidence_scores(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    out = df.copy()
    for label, source in EVIDENCE_COLUMNS.items():
        out[f"{label}_evidence_percentile"] = high_is_good_percentile(out[source])
    out["flow_evidence_percentile"] = pd.to_numeric(out["flow_stability_score"], errors="coerce").clip(0, 1)

    weighted_columns = [
        ("rule_evidence_percentile", config.weight_original_rule),
        ("centroid_evidence_percentile", config.weight_target_centroid),
        ("ot_evidence_percentile", config.weight_sinkhorn_ot),
        ("flow_evidence_percentile", config.weight_flow_stability),
    ]
    out["raw_consensus_score"], out["available_evidence_weight"] = weighted_available_mean(out, weighted_columns)

    evidence_cols = [c for c, _ in weighted_columns]
    matrix = out[evidence_cols].apply(pd.to_numeric, errors="coerce")
    out["n_available_evidence"] = matrix.notna().sum(axis=1)
    out["evidence_min"] = matrix.min(axis=1, skipna=True)
    out["evidence_max"] = matrix.max(axis=1, skipna=True)
    out["evidence_spread"] = out["evidence_max"] - out["evidence_min"]
    out["evidence_agreement"] = (1.0 - matrix.std(axis=1, skipna=True)).clip(0, 1)
    out["high_evidence_count"] = matrix.ge(config.high_evidence_threshold).sum(axis=1)
    out["moderate_evidence_count"] = matrix.ge(config.moderate_evidence_threshold).sum(axis=1)
    out["transport_high_count"] = out[
        ["centroid_evidence_percentile", "ot_evidence_percentile", "flow_evidence_percentile"]
    ].ge(config.high_evidence_threshold).sum(axis=1)

    discordant = out["evidence_spread"].gt(config.discordance_review_threshold)
    out["evidence_discordance_flag"] = discordant
    out["review_reasons"] = out["review_reasons_pre_score"].copy()
    add_discordance = discordant & ~out["is_hard_excluded"]
    empty = add_discordance & out["review_reasons"].eq("")
    nonempty = add_discordance & out["review_reasons"].ne("")
    out.loc[empty, "review_reasons"] = "多证据分歧较大"
    out.loc[nonempty, "review_reasons"] = out.loc[nonempty, "review_reasons"] + ";多证据分歧较大"

    out["review_flag_count"] = out["review_flag_count_pre_score"] + add_discordance.astype(int)
    out["is_manual_review"] = out["review_reasons"].ne("") & ~out["is_hard_excluded"]
    out["review_penalty"] = (
        out["review_flag_count"] * config.review_penalty_per_flag
    ).clip(upper=config.max_review_penalty)
    out["final_priority_score"] = (out["raw_consensus_score"] - out["review_penalty"]).clip(0, 1)

    eligible = ~out["is_hard_excluded"]
    out["final_rank_eligible"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["final_priority_percentile"] = np.nan
    if eligible.any():
        out.loc[eligible, "final_rank_eligible"] = (
            out.loc[eligible, "final_priority_score"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
        out.loc[eligible, "final_priority_percentile"] = high_is_good_percentile(
            out.loc[eligible, "final_priority_score"]
        )

    out = out.sort_values(
        ["is_hard_excluded", "final_priority_score", "pair_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["final_rank_all"] = np.arange(1, len(out) + 1)
    return out


def add_candidate_tracks(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    out = df.copy()
    tier = out.get("pair_tier", pd.Series("", index=out.index)).astype(str).str.upper()
    tier_a = tier.str.contains("PRIORITY_A", na=False)
    tier_non_a = ~tier_a

    final_high = out["final_priority_percentile"].ge(config.priority_percentile_threshold)
    rule_high = out["rule_evidence_percentile"].ge(config.high_evidence_threshold)
    rule_not_high = out["rule_evidence_percentile"].lt(config.high_evidence_threshold)
    centroid_supported = out["centroid_evidence_percentile"].ge(config.moderate_evidence_threshold)
    ot_supported = out["ot_evidence_percentile"].ge(config.moderate_evidence_threshold)
    multi_supported = out["moderate_evidence_count"].ge(2)
    transport_high = out["transport_high_count"].ge(2)

    track = pd.Series("RESERVE", index=out.index, dtype="object")
    track.loc[out["is_hard_excluded"]] = "EXCLUDED"
    active = ~out["is_hard_excluded"]

    confirmatory = active & tier_a & final_high & rule_high & centroid_supported & ot_supported & multi_supported
    track.loc[confirmatory] = "CONFIRMATORY"

    model_enhanced = active & final_high & rule_not_high & transport_high & ~confirmatory
    track.loc[model_enhanced] = "MODEL_ENHANCED"

    exploratory = active & tier_non_a & final_high & multi_supported & ~confirmatory & ~model_enhanced
    track.loc[exploratory] = "EXPLORATORY"

    consensus = active & final_high & multi_supported & track.eq("RESERVE")
    track.loc[consensus] = "CONSENSUS_PRIORITY"

    out["candidate_track"] = track
    out["review_status"] = np.select(
        [out["is_hard_excluded"], out["is_manual_review"]],
        ["HARD_EXCLUDED", "MANUAL_REVIEW"],
        default="PASS",
    )

    rationale: list[str] = []
    for row in out.itertuples(index=False):
        supported: list[str] = []
        for label, col in [
            ("规则", "rule_evidence_percentile"),
            ("质心", "centroid_evidence_percentile"),
            ("OT", "ot_evidence_percentile"),
            ("Flow稳定性", "flow_evidence_percentile"),
        ]:
            value = getattr(row, col)
            if pd.notna(value) and float(value) >= config.high_evidence_threshold:
                supported.append(label)
        text = f"高位证据：{','.join(supported) if supported else '无'}"
        if getattr(row, "translation_flag_global_suppressed", False):
            text += "；上游翻译标记为全局无区分度提示，未作为逐组合阻断项"
        if getattr(row, "review_reasons"):
            text += f"；复核：{getattr(row, 'review_reasons')}"
        if getattr(row, "hard_exclusion_reasons"):
            text += f"；排除：{getattr(row, 'hard_exclusion_reasons')}"
        rationale.append(text)
    out["screening_rationale"] = rationale
    return out


# ============================================================
# 去冗余与短名单
# ============================================================
def build_group_key(df: pd.DataFrame) -> pd.Series:
    if "pair_nonredundant_group" in df.columns:
        raw = df["pair_nonredundant_group"].map(safe_text)
    else:
        raw = pd.Series("", index=df.index)
    fallback = "PAIR::" + df["pair_id"].astype(str)
    return raw.where(raw.ne(""), fallback)


def choose_nonredundant_representatives(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = scored.copy()
    work["nonredundant_group_key"] = build_group_key(work)
    work["canonical_reference_preference"] = as_bool_series(work, "is_target_reference")
    work = work.sort_values(
        [
            "nonredundant_group_key",
            "is_hard_excluded",
            "is_manual_review",
            "final_priority_score",
            "canonical_reference_preference",
            "pair_id",
        ],
        ascending=[True, True, True, False, False, True],
        kind="mergesort",
    )
    representatives = work.groupby("nonredundant_group_key", as_index=False, sort=False).head(1).copy()
    representatives["is_nonredundant_representative"] = True

    rep_map = representatives.set_index("nonredundant_group_key")["pair_id"].to_dict()
    group_size = work.groupby("nonredundant_group_key")["pair_id"].size().to_dict()
    group_members = work.groupby("nonredundant_group_key")["pair_id"].apply(
        lambda x: ";".join(x.astype(str))
    ).to_dict()
    representatives["nonredundant_group_size"] = representatives["nonredundant_group_key"].map(group_size)
    representatives["nonredundant_group_members"] = representatives["nonredundant_group_key"].map(group_members)

    membership = work[
        ["nonredundant_group_key", "pair_id", "final_priority_score", "review_status", "candidate_track"]
    ].copy()
    membership["representative_pair_id"] = membership["nonredundant_group_key"].map(rep_map)
    membership["is_representative"] = membership["pair_id"].eq(membership["representative_pair_id"])

    representatives = representatives.sort_values(
        ["is_hard_excluded", "final_priority_score", "pair_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    representatives["nonredundant_rank_all"] = np.arange(1, len(representatives) + 1)
    pass_mask = representatives["review_status"].eq("PASS")
    representatives["nonredundant_rank_pass"] = pd.Series(pd.NA, index=representatives.index, dtype="Int64")
    representatives.loc[pass_mask, "nonredundant_rank_pass"] = np.arange(1, int(pass_mask.sum()) + 1)
    return representatives, membership


def get_drug_tokens(row: pd.Series) -> tuple[str, str]:
    a_value = row.get("drug_a_id", "")
    b_value = row.get("drug_b_id", "")
    if not safe_text(a_value):
        a_value = row.get("drug_a_name", "")
    if not safe_text(b_value):
        b_value = row.get("drug_b_name", "")
    return normalize_name(a_value), normalize_name(b_value)


def greedy_add(
    pool: pd.DataFrame,
    selected_indices: list[int],
    target_count: int,
    drug_counts: dict[str, int],
    max_per_drug: int,
) -> list[int]:
    added: list[int] = []
    for idx, row in pool.iterrows():
        if len(selected_indices) >= target_count:
            break
        if idx in selected_indices:
            continue
        a, b = get_drug_tokens(row)
        if max_per_drug > 0:
            if (a and drug_counts.get(a, 0) >= max_per_drug) or (b and drug_counts.get(b, 0) >= max_per_drug):
                continue
        selected_indices.append(idx)
        added.append(idx)
        if a:
            drug_counts[a] = drug_counts.get(a, 0) + 1
        if b:
            drug_counts[b] = drug_counts.get(b, 0) + 1
    return added


def select_shortlist(representatives: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    eligible = representatives.loc[representatives["review_status"].eq("PASS")].copy()
    eligible = eligible.loc[~eligible["candidate_track"].isin(["EXCLUDED", "RESERVE"])]
    eligible = eligible.sort_values(
        ["final_priority_score", "evidence_agreement", "pair_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    selected: list[int] = []
    selection_phase: dict[int, str] = {}
    drug_counts: dict[str, int] = {}
    quota_plan = [
        ("CONFIRMATORY", config.confirmatory_quota),
        ("MODEL_ENHANCED", config.model_enhanced_quota),
        ("EXPLORATORY", config.exploratory_quota),
    ]
    for track, quota in quota_plan:
        target = min(config.shortlist_size, len(selected) + max(0, quota))
        added = greedy_add(
            eligible.loc[eligible["candidate_track"].eq(track)],
            selected,
            target,
            drug_counts,
            config.max_appearances_per_drug,
        )
        for idx in added:
            selection_phase[idx] = f"QUOTA_{track}"

    added = greedy_add(
        eligible, selected, config.shortlist_size, drug_counts, config.max_appearances_per_drug
    )
    for idx in added:
        selection_phase[idx] = "SCORE_FILL"

    # 仅放宽单药重复上限，不放宽PASS、硬排除或人工复核条件。
    if len(selected) < config.shortlist_size and config.max_appearances_per_drug > 0:
        added = greedy_add(eligible, selected, config.shortlist_size, drug_counts, max_per_drug=0)
        for idx in added:
            selection_phase[idx] = "DRUG_CAP_RELAXED_FILL"

    shortlist = eligible.loc[selected].copy() if selected else eligible.head(0).copy()
    shortlist["shortlist_selection_phase"] = shortlist.index.map(selection_phase)
    shortlist = shortlist.sort_values(
        ["final_priority_score", "evidence_agreement", "pair_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1))
    return shortlist


# ============================================================
# 报告、图形与QC
# ============================================================
def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "original_pair_candidate_score",
        "target_centroid_score",
        "ot_baseline_score",
        "flow_stability_score",
        "raw_consensus_score",
        "final_priority_score",
    ]
    available = [c for c in columns if c in df.columns]
    corr = df[available].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    return corr.reset_index().rename(columns={"index": "metric"})


def make_shortlist_plot(shortlist: pd.DataFrame, path: Path) -> None:
    if shortlist.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] 无法生成图：{exc}")
        return
    plot_df = shortlist.head(30).sort_values("final_priority_score", ascending=True)
    height = max(5.0, 0.34 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(plot_df["pair_id"].astype(str), plot_df["final_priority_score"])
    ax.set_xlabel("Final multi-evidence priority score")
    ax.set_ylabel("Drug combination")
    ax.set_title("Final nonredundant shortlist")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def methods_text(
    config: ScreeningConfig,
    selected_flow: str,
    n_pairs: int,
    n_groups: int,
    translation_diagnostics: Mapping[str, Any],
) -> str:
    return f"""脚本版本：{SCRIPT_VERSION}

筛选目的
========
整合原始规则、目标质心、Sinkhorn OT和Flow多随机种子稳定性，输出非冗余最终候选。
结果仍是计算优先级，不是真实双药协同、联合安全或临床有效性证据。

输入规模
========
组合数：{n_pairs}
非冗余组数：{n_groups}
采用Flow实验：{selected_flow}

权重
====
原始规则：{config.weight_original_rule:.2f}
目标质心：{config.weight_target_centroid:.2f}
Sinkhorn OT：{config.weight_sinkhorn_ot:.2f}
Flow稳定性：{config.weight_flow_stability:.2f}

翻译/身份复核修正
=================
模式：{config.translation_review_mode}
原始translation_review_required阳性比例：{translation_diagnostics.get('raw_true_fraction', 0):.3f}
是否判定为全局无区分度提示：{translation_diagnostics.get('globally_uninformative', False)}
被抑制的全局标记数量：{translation_diagnostics.get('suppressed_count', 0)}
实际身份/翻译人工复核数量：{translation_diagnostics.get('effective_review_count', 0)}

adaptive模式规则：当原始translation_review_required在至少
{config.translation_global_flag_threshold:.0%}组合中为True时，不再将该字段本身作为逐组合阻断条件；
但drug_a_id、drug_b_id、drug_a_name或drug_b_name缺失仍进入人工复核。原始字段不会删除，
并通过translation_review_raw及translation_flag_global_suppressed保留审计信息。

过滤策略
========
{config.filter_policy}
- balanced：原始排除、同药物家族、高靶点冗余、双方机制均不稳定为硬排除；
  有区分度的身份问题、方向/来源冲突、Flow缺失和证据分歧进入人工复核。
- strict：上述人工复核项也作为硬排除。
- permissive：仅原始EXCLUDED硬排除，其余标记为人工复核。

监督模型处理
============
逻辑回归、随机森林和梯度提升不进入最终共识，避免重复学习脚本28/38构造的计算标签。

解释边界
========
1. 最终分数不是Bliss、Loewe、ZIP或HSA协同分数。
2. 不代表联合安全、暴露可达、药代相容或推荐剂量。
3. 实验前必须复核药物身份、盐型/同族、机制方向、安全性和相互作用。
4. 获得真实组合实验标签后，应重新做外部评价与校准。
"""


def create_qc(
    scores: pd.DataFrame,
    representatives: pd.DataFrame,
    shortlist: pd.DataFrame,
    selected_flow: str,
    config: ScreeningConfig,
    source_paths: Mapping[str, str],
    translation_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    if scores["pair_id"].duplicated().any():
        errors.append("pair_id存在重复。")
    missing_flow_fraction = float(scores["flow_stability_score"].isna().mean())
    if missing_flow_fraction > 0:
        warnings.append(f"{missing_flow_fraction:.1%}组合缺少Flow稳定性。")

    pass_pairs = int(scores["review_status"].eq("PASS").sum())
    pass_reps = int(representatives["review_status"].eq("PASS").sum())
    if pass_pairs == 0:
        errors.append("PASS组合为0，筛选流程无法产生可直接进入短名单的候选。")
    if pass_reps == 0:
        errors.append("PASS非冗余代表为0。")
    if len(shortlist) == 0:
        errors.append("最终短名单为空。")
    elif len(shortlist) < config.shortlist_size:
        warnings.append(
            f"短名单仅{len(shortlist)}个，少于设定的{config.shortlist_size}个；通过筛选条件的高优先级代表不足。"
        )

    if scores["n_available_evidence"].min() < 3:
        warnings.append("部分组合可用证据少于3项。")

    if translation_diagnostics.get("globally_uninformative", False):
        warnings.append(
            "上游translation_review_required近乎全量为True，已按adaptive策略视为全局无区分度提示；原始标记保留但未逐组合阻断。"
        )

    status = "failed_screening" if errors else ("warning" if warnings else "ok")
    return {
        "script_version": SCRIPT_VERSION,
        "status": status,
        "source_paths": dict(source_paths),
        "selected_flow_experiment": selected_flow,
        "configuration": asdict(config),
        "translation_review_diagnostics": dict(translation_diagnostics),
        "counts": {
            "all_pairs": int(len(scores)),
            "hard_excluded_pairs": int(scores["is_hard_excluded"].sum()),
            "manual_review_pairs": int(scores["is_manual_review"].sum()),
            "pass_pairs": pass_pairs,
            "nonredundant_groups": int(len(representatives)),
            "nonredundant_pass_representatives": pass_reps,
            "shortlist": int(len(shortlist)),
            "shortlist_drug_cap_relaxed_fill": int(
                shortlist.get("shortlist_selection_phase", pd.Series(dtype=str))
                .eq("DRUG_CAP_RELAXED_FILL")
                .sum()
            ),
        },
        "candidate_track_counts_all_pairs": {
            str(k): int(v) for k, v in scores["candidate_track"].value_counts().items()
        },
        "candidate_track_counts_representatives": {
            str(k): int(v) for k, v in representatives["candidate_track"].value_counts().items()
        },
        "shortlist_track_counts": {
            str(k): int(v) for k, v in shortlist["candidate_track"].value_counts().items()
        },
        "missing_flow_fraction": missing_flow_fraction,
        "errors": errors,
        "warnings": warnings,
        "interpretation_boundary": (
            "内部计算优先级筛选；不是实验协同、联合安全、剂量或临床有效性证据。"
        ),
    }


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    args = parse_args()
    config = ScreeningConfig(
        filter_policy=args.filter_policy,
        translation_review_mode=args.translation_review_mode,
        translation_global_flag_threshold=args.translation_global_flag_threshold,
        shortlist_size=args.shortlist_size,
        confirmatory_quota=args.confirmatory_quota,
        model_enhanced_quota=args.model_enhanced_quota,
        exploratory_quota=args.exploratory_quota,
        max_appearances_per_drug=args.max_appearances_per_drug,
        output_policy=args.output_policy,
    )

    if not (0.5 <= config.translation_global_flag_threshold <= 1.0):
        raise ValueError("translation_global_flag_threshold应在0.5到1.0之间。")
    if config.shortlist_size <= 0:
        raise ValueError("shortlist_size必须大于0。")
    if min(config.confirmatory_quota, config.model_enhanced_quota, config.exploratory_quota) < 0:
        raise ValueError("候选配额不能为负。")

    path_scores = args.script39_root / "01_ot_fm_combination_scores_all.csv"
    path_frequency = args.script40_root / "07_topk_selection_frequency.csv"
    path_long = args.script40_root / "08_pair_scores_all_models_long.csv"
    path_best = args.script40_root / "09_best_model_configuration.json"

    base = prepare_base_input(read_csv_required(path_scores))
    frequency = read_csv_required(path_frequency)
    long_scores = pd.read_csv(path_long, low_memory=False) if path_long.exists() else pd.DataFrame()
    best_config = read_json_optional(path_best)

    require_columns(base, ["pair_id", *EVIDENCE_COLUMNS.values()], "01_ot_fm_combination_scores_all.csv")
    if base["pair_id"].isna().any():
        raise ValueError("基础评分文件存在空pair_id。")
    if base["pair_id"].duplicated().any():
        examples = base.loc[base["pair_id"].duplicated(keep=False), "pair_id"].head(10).tolist()
        raise ValueError(f"基础评分文件存在重复pair_id，例如：{examples}")

    selected_flow = choose_flow_experiment(frequency, best_config)
    print(f"[INFO] 使用Flow多种子实验：{selected_flow}")

    flow = build_flow_evidence(frequency, long_scores, selected_flow, len(base))
    merged = base.merge(flow, on="pair_id", how="left", validate="one_to_one")
    merged, translation_diagnostics = apply_quality_flags(merged, config)
    merged = add_evidence_scores(merged, config)
    merged = add_candidate_tracks(merged, config)

    representatives, membership = choose_nonredundant_representatives(merged)
    shortlist = select_shortlist(representatives, config)

    ensure_output_dir(args.output_root, config.output_policy)

    key_cols = [
        "final_rank_all",
        "final_rank_eligible",
        "pair_id",
        "candidate_track",
        "review_status",
        "final_priority_score",
        "raw_consensus_score",
        "review_penalty",
        "final_priority_percentile",
        "rule_evidence_percentile",
        "centroid_evidence_percentile",
        "ot_evidence_percentile",
        "flow_evidence_percentile",
        "high_evidence_count",
        "moderate_evidence_count",
        "evidence_agreement",
        "evidence_spread",
        "selected_flow_experiment",
        "flow_mean_rank",
        "flow_rank_std",
        "flow_top10_frequency",
        "flow_top20_frequency",
        "flow_top50_frequency",
        "translation_review_raw",
        "identity_fields_missing",
        "translation_review_effective",
        "translation_flag_global_suppressed",
        "hard_exclusion_reasons",
        "review_reasons",
        "screening_rationale",
    ]
    key_cols = [c for c in key_cols if c in merged.columns]
    all_output = merged[key_cols + [c for c in merged.columns if c not in key_cols]]

    rep_key = [
        "nonredundant_rank_all",
        "nonredundant_rank_pass",
        "pair_id",
        "nonredundant_group_key",
        "nonredundant_group_size",
        "candidate_track",
        "review_status",
        "final_priority_score",
        "evidence_agreement",
        "translation_review_raw",
        "translation_review_effective",
        "translation_flag_global_suppressed",
        "hard_exclusion_reasons",
        "review_reasons",
        "nonredundant_group_members",
    ]
    rep_key = [c for c in rep_key if c in representatives.columns]
    rep_output = representatives[rep_key + [c for c in representatives.columns if c not in rep_key]]

    write_csv(all_output, args.output_root / "01_final_scores_all_pairs.csv")
    write_csv(rep_output, args.output_root / "02_nonredundant_representatives.csv")
    write_csv(shortlist, args.output_root / "03_final_shortlist.csv")
    write_csv(
        rep_output.loc[rep_output["candidate_track"].eq("CONFIRMATORY") & rep_output["review_status"].eq("PASS")],
        args.output_root / "04_confirmatory_candidates.csv",
    )
    write_csv(
        rep_output.loc[rep_output["candidate_track"].eq("MODEL_ENHANCED") & rep_output["review_status"].eq("PASS")],
        args.output_root / "05_model_enhanced_candidates.csv",
    )
    write_csv(
        rep_output.loc[rep_output["candidate_track"].eq("EXPLORATORY") & rep_output["review_status"].eq("PASS")],
        args.output_root / "06_exploratory_candidates.csv",
    )
    write_csv(
        rep_output.loc[rep_output["review_status"].eq("MANUAL_REVIEW")],
        args.output_root / "07_manual_review_candidates.csv",
    )
    write_csv(
        rep_output.loc[rep_output["review_status"].eq("HARD_EXCLUDED")],
        args.output_root / "08_hard_excluded_candidates.csv",
    )
    write_csv(membership, args.output_root / "09_nonredundant_group_membership.csv")
    write_csv(build_correlation_table(merged), args.output_root / "10_evidence_spearman_correlation.csv")

    source_paths = {
        "script39_scores": str(path_scores),
        "script40_topk_frequency": str(path_frequency),
        "script40_long_scores": str(path_long),
        "script40_best_configuration": str(path_best),
    }
    qc = create_qc(
        merged,
        representatives,
        shortlist,
        selected_flow,
        config,
        source_paths,
        translation_diagnostics,
    )
    write_json(qc, args.output_root / "11_final_screening_qc.json")
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "selected_flow_experiment": selected_flow,
            "configuration": asdict(config),
            "translation_review_diagnostics": translation_diagnostics,
            "evidence_definition": {
                "original_rule": EVIDENCE_COLUMNS["rule"],
                "target_centroid": EVIDENCE_COLUMNS["centroid"],
                "sinkhorn_ot": EVIDENCE_COLUMNS["ot"],
                "flow": "多随机种子平均排名与Top-K进入频率",
            },
        },
        args.output_root / "12_frozen_final_screening_configuration.json",
    )
    (args.output_root / "13_methods_and_interpretation.txt").write_text(
        methods_text(config, selected_flow, len(merged), len(representatives), translation_diagnostics),
        encoding="utf-8",
    )
    make_shortlist_plot(shortlist, args.output_root / "14_final_shortlist.png")

    print("\n完成。")
    print(f"全部组合：{len(merged)}")
    print(f"PASS组合：{int(merged['review_status'].eq('PASS').sum())}")
    print(f"人工复核组合：{int(merged['review_status'].eq('MANUAL_REVIEW').sum())}")
    print(f"非冗余代表：{len(representatives)}")
    print(f"最终短名单：{len(shortlist)}")
    print(f"QC状态：{qc['status']}")
    print(f"输出目录：{args.output_root}")
    print("重要：结果是计算优先级，不是真实协同或联合安全结论。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
