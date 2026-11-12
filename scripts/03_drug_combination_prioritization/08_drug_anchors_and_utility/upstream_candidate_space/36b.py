#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36a：将 SigCom LINCS 导出的 metadata.tsv 一次性整理为 37a 可用输入。

本版本针对以下实际 metadata.tsv 表头：
Name, Perturbagen, Perturbation Type, Dose, Tissue, Cell Line, Disease,
Timepoint, Data Level, Creation Time, z-score (up), z-score (down),
log p-val (Fisher), log p-val (avg), p-value (up), z-score (sum),
p-value (down), FDR (up), FDR (down),
p-value Bonferroni (up), p-value Bonferroni (down)

重要说明
--------
metadata.tsv 是“扰动 signature 级”的检索结果，不是逐基因表达矩阵。
因此本脚本不会伪造 gene_symbol 或逐基因 signed_effect，而是输出：
  1. drug_perturbation_signature.csv：全部 signature 的标准化结果；
  2. drug_best_signature.csv：每个药物最优 signature；
  3. drug_metadata.csv：每个药物的聚合信息；
  4. drug_target_actions.csv：若 DGIdb 文件可用则生成，否则输出空表；
  5. gene_network_edges.csv：若 PPI 文件可用则生成，否则输出空表；
  6. gene_alias_mapping.csv：若 DGIdb genes.tsv 可用则生成，否则输出空表；
  7. metadata_repair_audit.csv：metadata 列修复记录；
  8. metadata_rejected_rows.csv：无法修复的行；
  9. preparation_qc.json 与 37a_input_manifest.json。

metadata.tsv 中部分行会因 Tissue/Disease 为空而少列。本脚本依据固定的
Creation Time、Timepoint、Data Level 与数值尾部结构自动恢复缺失位置。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS：在 PyCharm 中通常只需检查这里
# =============================================================================

METADATA_TSV_PATH = r"E:\PythonProject\drug_data\metadata.tsv"
OUTPUT_DIR = r"E:\PythonProject\drug_data"

# 以下文件均为可选。存在时会一起整理，不存在时不会阻止 metadata 处理。
DGIDB_CLEAN_INTERACTIONS_PATH = (
    r"E:\PythonProject\dgidb_drug_target_outputs\01_dgidb_interactions_clean.csv"
)
DGIDB_PAIR_TABLE_PATH = (
    r"E:\PythonProject\dgidb_drug_target_outputs\02_drug_target_pairs_for_ppi.csv"
)
DRUG_ENTITY_QC_PATH = (
    r"E:\PythonProject\results\ot_disease_reversal_score_v3_5"
    r"\tables\04c_drug_entity_quality_qc_v5.csv"
)
PPI_EDGE_PATH = r"E:\PythonProject\16b_PPI net\string_ppi_edges_700.csv"
DGIDB_GENES_TSV_PATH = r"E:\PythonProject\dgidb\genes.tsv"

OVERWRITE = True

# 是否仅保留 Chemical。你的当前文件全部为 Chemical，默认仍保留显式过滤。
KEEP_PERTURBATION_TYPES: list[str] = ["Chemical"]

# 可选阈值。None 表示不过滤，保留全部下载结果。
MAX_STRICT_QVALUE: Optional[float] = None
MIN_REVERSAL_SCORE: Optional[float] = None
MIN_DATA_LEVEL: Optional[float] = None

# 每个药物的“最佳 signature”排序规则：
# 先按 query_reversal_score 降序，再按 qvalue_strict 升序、数据层级降序。
# query_reversal_score = -z_score_sum；原始 z_score_sum 越负，逆转越强。
# =============================================================================


EXPECTED_METADATA_COLUMNS = [
    "Name",
    "Perturbagen",
    "Perturbation Type",
    "Dose",
    "Tissue",
    "Cell Line",
    "Disease",
    "Timepoint",
    "Data Level",
    "Creation Time",
    "z-score (up)",
    "z-score (down)",
    "log p-val (Fisher)",
    "log p-val (avg)",
    "p-value (up)",
    "z-score (sum)",
    "p-value (down)",
    "FDR (up)",
    "FDR (down)",
    "p-value Bonferroni (up)",
    "p-value Bonferroni (down)",
]

NUMERIC_METADATA_COLUMNS = [
    "Data Level",
    "z-score (up)",
    "z-score (down)",
    "log p-val (Fisher)",
    "log p-val (avg)",
    "p-value (up)",
    "z-score (sum)",
    "p-value (down)",
    "FDR (up)",
    "FDR (down)",
    "p-value Bonferroni (up)",
    "p-value Bonferroni (down)",
]

NEGATIVE_TERMS = (
    "inhibitor",
    "inhibits",
    "inhibition",
    "antagonist",
    "blocker",
    "degrader",
    "suppressor",
    "suppress",
    "downregulator",
    "negative modulator",
    "inverse agonist",
    "antisense",
)
POSITIVE_TERMS = (
    "agonist",
    "activator",
    "activation",
    "stimulator",
    "positive modulator",
    "inducer",
    "potentiator",
    "upregulator",
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip()).upper()


def normalize_col_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def find_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    lookup = {normalize_col_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"未找到列。候选：{list(candidates)}；实际：{list(df.columns)}"
        )
    return None


def optional_path(value: str | Path | None) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def stable_drug_id(name: object) -> str:
    """
    使用规范化药物名生成跨 metadata/DGIdb 可复用 ID。
    若名称不能形成有效 slug，则使用短哈希。
    """
    clean = normalize_text(name)
    slug = re.sub(r"[^A-Z0-9]+", "_", clean.upper()).strip("_")
    if slug:
        return slug
    digest = hashlib.sha1(clean.casefold().encode("utf-8")).hexdigest()[:12]
    return f"DRUG_{digest.upper()}"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in {"true", "t", "1", "yes", "y"}


def classify_action(value: object) -> str:
    text = normalize_text(value).casefold()
    has_negative = any(term in text for term in NEGATIVE_TERMS)
    has_positive = any(term in text for term in POSITIVE_TERMS)
    if has_negative and not has_positive:
        return "inhibitor"
    if has_positive and not has_negative:
        return "activator"
    return "unknown"


def join_unique(values: Iterable[object], separator: str = ";") -> str:
    unique = sorted(
        {normalize_text(v) for v in values if normalize_text(v)},
        key=str.casefold,
    )
    return separator.join(unique)


def parse_number_and_unit(value: object) -> tuple[float, str]:
    text = normalize_text(value)
    match = re.match(
        r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*([^\d\s].*)?$",
        text,
    )
    if not match:
        return np.nan, ""
    number = pd.to_numeric(match.group(1), errors="coerce")
    unit = normalize_text(match.group(2) or "")
    return float(number) if pd.notna(number) else np.nan, unit


def parse_timepoint_hours(value: object) -> float:
    number, unit = parse_number_and_unit(value)
    if pd.isna(number):
        return np.nan
    unit_key = unit.casefold()
    if unit_key in {"h", "hr", "hrs", "hour", "hours"}:
        return float(number)
    if unit_key in {"d", "day", "days"}:
        return float(number) * 24.0
    if unit_key in {"min", "mins", "minute", "minutes"}:
        return float(number) / 60.0
    return np.nan


def parse_metadata_tsv(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    读取并修复 SigCom metadata.tsv。

    观察到的合法布局：
    - 21列：Tissue、Cell Line、Disease 均存在；
    - 20列：Disease 为空且导出时省略；
    - 19列：Tissue 与 Disease 均为空，只保留 Cell Line。

    为避免只依赖总列数，本函数首先定位 Creation Time，然后将其前面的
    最后两个字段识别为 Timepoint 和 Data Level。
    """
    repaired_rows: list[list[str]] = []
    audit_rows: list[dict] = []
    rejected_rows: list[dict] = []

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("metadata.tsv 为空。") from exc

        normalized_header = [normalize_col_name(c) for c in raw_header]
        expected_normalized = [
            normalize_col_name(c) for c in EXPECTED_METADATA_COLUMNS
        ]
        if normalized_header != expected_normalized:
            raise ValueError(
                "metadata.tsv 表头与当前脚本预期不一致。\n"
                f"实际表头：{raw_header}\n"
                f"预期表头：{EXPECTED_METADATA_COLUMNS}"
            )

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not normalize_text(x) for x in row):
                continue

            raw_field_count = len(row)
            try:
                creation_index = next(
                    idx
                    for idx, value in enumerate(row[4:], start=4)
                    if re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}",
                        normalize_text(value),
                    )
                )
                fixed_left = row[:4]
                context_fields = row[4:creation_index]
                fixed_tail = row[creation_index:]

                # Creation Time + 11 个数值字段。
                if len(fixed_tail) != 12:
                    raise ValueError(
                        f"Creation Time 后应有12个字段，实际{len(fixed_tail)}个"
                    )

                # context_fields = [可选 Tissue/Cell Line/Disease] + Timepoint + Data Level
                if len(context_fields) < 3 or len(context_fields) > 5:
                    raise ValueError(
                        f"Creation Time 前的上下文字段数异常：{len(context_fields)}"
                    )

                timepoint = context_fields[-2]
                data_level = context_fields[-1]
                optional_context = context_fields[:-2]

                tissue = ""
                cell_line = ""
                disease = ""
                repair_action = "none"

                if len(optional_context) == 3:
                    tissue, cell_line, disease = optional_context
                elif len(optional_context) == 2:
                    tissue, cell_line = optional_context
                    repair_action = "insert_missing_disease"
                elif len(optional_context) == 1:
                    cell_line = optional_context[0]
                    repair_action = "insert_missing_tissue_and_disease"
                else:
                    raise ValueError(
                        "无法识别 Tissue/Cell Line/Disease 的位置"
                    )

                repaired = (
                    fixed_left
                    + [tissue, cell_line, disease, timepoint, data_level]
                    + fixed_tail
                )
                if len(repaired) != len(EXPECTED_METADATA_COLUMNS):
                    raise ValueError(
                        f"修复后列数为{len(repaired)}，预期21"
                    )

                repaired_rows.append(repaired)
                audit_rows.append(
                    {
                        "line_number": line_number,
                        "signature_name": normalize_text(row[0]),
                        "raw_field_count": raw_field_count,
                        "repair_action": repair_action,
                        "tissue_after_repair": normalize_text(tissue),
                        "cell_line_after_repair": normalize_text(cell_line),
                        "disease_after_repair": normalize_text(disease),
                    }
                )
            except Exception as exc:
                rejected_rows.append(
                    {
                        "line_number": line_number,
                        "raw_field_count": raw_field_count,
                        "reason": str(exc),
                        "raw_row": "\t".join(row),
                    }
                )

    repaired_df = pd.DataFrame(
        repaired_rows,
        columns=EXPECTED_METADATA_COLUMNS,
    )
    audit_df = pd.DataFrame(audit_rows)
    rejected_df = pd.DataFrame(
        rejected_rows,
        columns=["line_number", "raw_field_count", "reason", "raw_row"],
    )

    if repaired_df.empty:
        raise ValueError("没有成功读取任何 metadata 记录。")

    for column in NUMERIC_METADATA_COLUMNS:
        repaired_df[column] = pd.to_numeric(
            repaired_df[column],
            errors="coerce",
        )

    invalid_numeric = repaired_df[NUMERIC_METADATA_COLUMNS].isna().any(axis=1)
    if invalid_numeric.any():
        for idx, row in repaired_df[invalid_numeric].iterrows():
            rejected_rows.append(
                {
                    "line_number": int(audit_df.loc[idx, "line_number"]),
                    "raw_field_count": int(
                        audit_df.loc[idx, "raw_field_count"]
                    ),
                    "reason": "至少一个必要数值字段无法转换",
                    "raw_row": "",
                }
            )
        repaired_df = repaired_df[~invalid_numeric].copy()
        audit_df = audit_df[~invalid_numeric].copy()
        rejected_df = pd.DataFrame(
            rejected_rows,
            columns=["line_number", "raw_field_count", "reason", "raw_row"],
        )

    info = {
        "source_path": str(path),
        "raw_data_rows": int(len(audit_df) + len(rejected_df)),
        "accepted_rows": int(len(repaired_df)),
        "rejected_rows": int(len(rejected_df)),
        "repair_counts": (
            audit_df["repair_action"].value_counts().to_dict()
            if len(audit_df)
            else {}
        ),
    }
    return repaired_df, audit_df, rejected_df, info


def standardize_perturbation_signatures(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=raw.index)

    out["record_level"] = "signature"
    out["drug_id"] = raw["Perturbagen"].map(stable_drug_id)
    out["drug_name"] = raw["Perturbagen"].map(normalize_text)
    out["perturbation_signature_id"] = raw["Name"].map(normalize_text)
    out["context_id"] = out["perturbation_signature_id"]
    out["perturbation_type"] = raw["Perturbation Type"].map(normalize_text)
    out["dose"] = raw["Dose"].map(normalize_text)

    parsed_dose = out["dose"].map(parse_number_and_unit)
    out["dose_value"] = [item[0] for item in parsed_dose]
    out["dose_unit"] = [item[1] for item in parsed_dose]

    out["tissue"] = raw["Tissue"].map(normalize_text)
    out["cell_line"] = raw["Cell Line"].map(normalize_text)
    out["disease"] = raw["Disease"].map(normalize_text)
    out["timepoint"] = raw["Timepoint"].map(normalize_text)
    out["timepoint_hours"] = out["timepoint"].map(parse_timepoint_hours)
    out["data_level"] = pd.to_numeric(raw["Data Level"], errors="coerce")
    out["creation_time"] = raw["Creation Time"].map(normalize_text)

    rename_numeric = {
        "z-score (up)": "z_score_up",
        "z-score (down)": "z_score_down",
        "z-score (sum)": "z_score_sum",
        "log p-val (Fisher)": "log_p_fisher",
        "log p-val (avg)": "log_p_average",
        "p-value (up)": "p_value_up",
        "p-value (down)": "p_value_down",
        "FDR (up)": "fdr_up",
        "FDR (down)": "fdr_down",
        "p-value Bonferroni (up)": "bonferroni_up",
        "p-value Bonferroni (down)": "bonferroni_down",
    }
    for old, new in rename_numeric.items():
        out[new] = pd.to_numeric(raw[old], errors="coerce")

    # 对当前“寻找 reverser”的结果，z_score_sum 越负，逆转越强。
    out["query_reversal_score"] = -out["z_score_sum"]
    out["is_reverser"] = out["z_score_sum"] < 0

    # 严格 qvalue 要求上下两个方向都可靠，因此取较大 FDR。
    out["qvalue_strict"] = out[["fdr_up", "fdr_down"]].max(axis=1)
    out["qvalue_best_arm"] = out[["fdr_up", "fdr_down"]].min(axis=1)
    out["qvalue"] = out["qvalue_strict"]

    # 兼容 37a 的 0-1 权重：仅作为本次导出结果内部的相对排序权重。
    # 不把它解释为实验效应大小。
    out["evidence_weight"] = out["query_reversal_score"].rank(
        method="average",
        pct=True,
        ascending=True,
    )
    out["source"] = "SigCom LINCS metadata.tsv"

    # 基础清理。
    valid = (
        out["drug_id"].ne("")
        & out["drug_name"].ne("")
        & out["perturbation_signature_id"].ne("")
        & out["z_score_sum"].notna()
    )
    out = out[valid].copy()

    keep_types = {
        normalize_text(value).casefold()
        for value in KEEP_PERTURBATION_TYPES
        if normalize_text(value)
    }
    if keep_types:
        out = out[
            out["perturbation_type"].str.casefold().isin(keep_types)
        ].copy()

    if MAX_STRICT_QVALUE is not None:
        out = out[out["qvalue_strict"] <= float(MAX_STRICT_QVALUE)].copy()
    if MIN_REVERSAL_SCORE is not None:
        out = out[
            out["query_reversal_score"] >= float(MIN_REVERSAL_SCORE)
        ].copy()
    if MIN_DATA_LEVEL is not None:
        out = out[out["data_level"] >= float(MIN_DATA_LEVEL)].copy()

    out = out.sort_values(
        [
            "query_reversal_score",
            "qvalue_strict",
            "data_level",
            "drug_name",
            "perturbation_signature_id",
        ],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)

    out["global_rank"] = np.arange(1, len(out) + 1)

    info = {
        "n_signature_rows": int(len(out)),
        "n_unique_drugs": int(out["drug_id"].nunique()),
        "n_unique_cell_lines": int(out["cell_line"].nunique()),
        "n_unique_tissues_nonempty": int(
            out.loc[out["tissue"].ne(""), "tissue"].nunique()
        ),
        "n_reverser_rows": int(out["is_reverser"].sum()),
        "data_levels": sorted(
            pd.to_numeric(out["data_level"], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        ),
    }
    return out, info


def select_best_signature_per_drug(
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    ordered = signatures.sort_values(
        [
            "drug_id",
            "query_reversal_score",
            "qvalue_strict",
            "data_level",
            "perturbation_signature_id",
        ],
        ascending=[True, False, True, False, True],
    )
    best = ordered.drop_duplicates("drug_id", keep="first").copy()
    best = best.sort_values(
        ["query_reversal_score", "qvalue_strict", "drug_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best["drug_rank"] = np.arange(1, len(best) + 1)
    return best


def make_signature_drug_metadata(
    signatures: pd.DataFrame,
    best: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        signatures.groupby(["drug_id", "drug_name"], as_index=False)
        .agg(
            n_perturbation_signatures=(
                "perturbation_signature_id",
                "nunique",
            ),
            perturbation_types=("perturbation_type", join_unique),
            doses=("dose", join_unique),
            tissues=("tissue", join_unique),
            cell_lines=("cell_line", join_unique),
            diseases=("disease", join_unique),
            timepoints=("timepoint", join_unique),
            max_data_level=("data_level", "max"),
            mean_reversal_score=("query_reversal_score", "mean"),
            median_reversal_score=("query_reversal_score", "median"),
            max_reversal_score=("query_reversal_score", "max"),
            min_qvalue_strict=("qvalue_strict", "min"),
            min_fdr_up=("fdr_up", "min"),
            min_fdr_down=("fdr_down", "min"),
        )
    )

    best_cols = best[
        [
            "drug_id",
            "perturbation_signature_id",
            "context_id",
            "dose",
            "tissue",
            "cell_line",
            "disease",
            "timepoint",
            "data_level",
            "query_reversal_score",
            "qvalue_strict",
            "fdr_up",
            "fdr_down",
            "drug_rank",
        ]
    ].rename(
        columns={
            "perturbation_signature_id": "best_signature_id",
            "context_id": "best_context_id",
            "dose": "best_dose",
            "tissue": "best_tissue",
            "cell_line": "best_cell_line",
            "disease": "best_disease",
            "timepoint": "best_timepoint",
            "data_level": "best_data_level",
            "query_reversal_score": "best_reversal_score",
            "qvalue_strict": "best_qvalue_strict",
            "fdr_up": "best_fdr_up",
            "fdr_down": "best_fdr_down",
        }
    )

    meta = grouped.merge(best_cols, on="drug_id", how="left")
    meta["approval_status"] = "unknown"
    meta["clinical_score"] = np.nan
    meta["safety_score"] = np.nan
    meta["availability_score"] = np.nan
    meta["pulmonary_risk"] = ""
    meta["notes"] = (
        "SigCom signature-level result; no gene-level signed effect"
    )
    return meta


def load_optional_drug_source(
    clean_path_value: str,
    pair_path_value: str,
) -> tuple[Optional[pd.DataFrame], Optional[Path], dict]:
    clean_path = optional_path(clean_path_value)
    pair_path = optional_path(pair_path_value)

    selected = clean_path or pair_path
    if selected is None:
        return None, None, {
            "status": "not_provided",
            "message": (
                "DGIdb source not found; drug_target_actions.csv will be "
                "header-only."
            ),
        }

    return (
        pd.read_csv(selected, low_memory=False),
        selected,
        {"status": "loaded", "source_path": str(selected)},
    )


def make_targets_and_dgidb_metadata(
    raw: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_columns = [
        "drug_id",
        "drug_name",
        "drug_concept_id",
        "gene_symbol",
        "action",
        "evidence_weight",
        "source",
    ]
    meta_columns = [
        "drug_id",
        "drug_name",
        "approval_status",
        "clinical_score",
        "safety_score",
        "availability_score",
        "pulmonary_risk",
        "notes",
    ]

    if raw is None or raw.empty:
        return (
            pd.DataFrame(columns=target_columns),
            pd.DataFrame(columns=meta_columns),
            {"status": "not_provided", "n_target_rows": 0},
        )

    drug_col = find_col(
        raw,
        ["drug_name", "drug", "drug_claim_name"],
    )
    gene_col = find_col(
        raw,
        ["gene", "gene_symbol", "target_gene"],
    )
    action_col = find_col(
        raw,
        [
            "interaction_type",
            "interaction_types",
            "action",
            "mechanism_of_action",
        ],
        required=False,
    )
    source_col = find_col(
        raw,
        ["source", "interaction_source_db_name"],
        required=False,
    )
    concept_col = find_col(
        raw,
        ["drug_concept_id", "concept_id", "drug_id"],
        required=False,
    )
    score_col = find_col(
        raw,
        [
            "interaction_score",
            "max_interaction_score",
            "mean_interaction_score",
        ],
        required=False,
    )
    approved_col = find_col(raw, ["approved"], required=False)
    immuno_col = find_col(raw, ["immunotherapy"], required=False)
    antineo_col = find_col(raw, ["anti_neoplastic"], required=False)

    work = pd.DataFrame(index=raw.index)
    work["drug_name"] = raw[drug_col].map(normalize_text)
    work["drug_id"] = work["drug_name"].map(stable_drug_id)
    work["drug_concept_id"] = (
        raw[concept_col].map(normalize_text) if concept_col else ""
    )
    work["gene_symbol"] = raw[gene_col].map(normalize_gene)
    work["action_raw"] = (
        raw[action_col].map(normalize_text) if action_col else "unknown"
    )
    work["action"] = work["action_raw"].map(classify_action)
    work["source"] = (
        raw[source_col].map(normalize_text)
        if source_col
        else "UNSPECIFIED"
    )
    work["interaction_score"] = (
        pd.to_numeric(raw[score_col], errors="coerce")
        if score_col
        else np.nan
    )
    work["approved"] = (
        raw[approved_col].map(parse_bool) if approved_col else False
    )
    work["immunotherapy"] = (
        raw[immuno_col].map(parse_bool) if immuno_col else False
    )
    work["anti_neoplastic"] = (
        raw[antineo_col].map(parse_bool) if antineo_col else False
    )

    work = work[
        work["drug_id"].ne("") & work["gene_symbol"].ne("")
    ].copy()

    grouped = work.groupby(
        [
            "drug_id",
            "drug_name",
            "drug_concept_id",
            "gene_symbol",
            "action",
        ],
        dropna=False,
    )
    target = (
        grouped.agg(
            source=("source", join_unique),
            n_sources=(
                "source",
                lambda values: len(
                    {
                        normalize_text(v)
                        for v in values
                        if normalize_text(v)
                    }
                ),
            ),
            n_records=("gene_symbol", "size"),
            max_interaction_score=("interaction_score", "max"),
        )
        .reset_index()
    )

    score = pd.to_numeric(
        target["max_interaction_score"],
        errors="coerce",
    )
    score_component = score.fillna(0.0).clip(0.0, 1.0)
    source_component = (
        np.log1p(target["n_sources"].clip(lower=0)) / math.log(5)
    ).clip(0.0, 1.0)
    record_component = (
        np.log1p(target["n_records"].clip(lower=0)) / math.log(10)
    ).clip(0.0, 1.0)

    target["evidence_weight"] = (
        0.60 * score_component
        + 0.30 * source_component
        + 0.10 * record_component
    ).clip(0.0, 1.0)
    target = target[target_columns].copy()

    dgidb_meta = (
        work.groupby(["drug_id", "drug_name"], as_index=False)
        .agg(
            approved=("approved", "max"),
            immunotherapy=("immunotherapy", "max"),
            anti_neoplastic=("anti_neoplastic", "max"),
            n_target_genes=("gene_symbol", "nunique"),
        )
    )
    dgidb_meta["approval_status"] = np.where(
        dgidb_meta["approved"],
        "approved",
        "unknown",
    )
    dgidb_meta["clinical_score"] = np.nan
    dgidb_meta["safety_score"] = np.nan
    dgidb_meta["availability_score"] = np.nan
    dgidb_meta["pulmonary_risk"] = ""
    dgidb_meta["notes"] = dgidb_meta.apply(
        lambda row: ";".join(
            item
            for item, keep in [
                (
                    "immunotherapy",
                    bool(row["immunotherapy"]),
                ),
                (
                    "anti_neoplastic",
                    bool(row["anti_neoplastic"]),
                ),
                (
                    f"n_target_genes={int(row['n_target_genes'])}",
                    True,
                ),
            ]
            if keep
        ),
        axis=1,
    )
    dgidb_meta = dgidb_meta[meta_columns].copy()

    info = {
        "status": "ok",
        "n_source_rows": int(len(raw)),
        "n_target_rows": int(len(target)),
        "n_unique_drugs": int(target["drug_id"].nunique()),
        "n_directional_target_rows": int(
            target["action"].isin(["inhibitor", "activator"]).sum()
        ),
        "n_unknown_target_rows": int(
            target["action"].eq("unknown").sum()
        ),
    }
    return target, dgidb_meta, info


def merge_drug_metadata(
    signature_meta: pd.DataFrame,
    dgidb_meta: pd.DataFrame,
    entity_qc_path_value: str,
) -> pd.DataFrame:
    meta = signature_meta.copy()

    if not dgidb_meta.empty:
        suffix_cols = [
            "approval_status",
            "clinical_score",
            "safety_score",
            "availability_score",
            "pulmonary_risk",
            "notes",
        ]
        extra = dgidb_meta[
            ["drug_id"] + suffix_cols
        ].rename(
            columns={column: f"dgidb_{column}" for column in suffix_cols}
        )
        meta = meta.merge(extra, on="drug_id", how="left")

        for column in suffix_cols:
            dgidb_column = f"dgidb_{column}"
            if column == "notes":
                left = meta[column].fillna("").astype(str).str.strip("; ")
                right = (
                    meta[dgidb_column]
                    .fillna("")
                    .astype(str)
                    .str.strip("; ")
                )
                meta[column] = np.where(
                    right.ne(""),
                    np.where(
                        left.ne(""),
                        left + ";" + right,
                        right,
                    ),
                    left,
                )
            else:
                meta[column] = meta[dgidb_column].combine_first(meta[column])
            meta = meta.drop(columns=[dgidb_column])

    qc_path = optional_path(entity_qc_path_value)
    if qc_path is not None:
        qc = pd.read_csv(qc_path, low_memory=False)
        qc_drug_col = find_col(
            qc,
            ["drug_clean", "drug_name", "drug"],
            required=False,
        )
        if qc_drug_col:
            qc = qc.copy()
            qc["drug_id"] = qc[qc_drug_col].map(stable_drug_id)
            keep = [
                column
                for column in [
                    "drug_id",
                    "entity_quality_categories",
                    "hard_block_reasons",
                    "hard_block_main_table",
                    "drug_family_group",
                ]
                if column in qc.columns
            ]
            if len(keep) > 1:
                qc = qc[keep].drop_duplicates("drug_id")
                meta = meta.merge(qc, on="drug_id", how="left")

                note_columns = [
                    column
                    for column in [
                        "entity_quality_categories",
                        "hard_block_reasons",
                        "drug_family_group",
                    ]
                    if column in meta.columns
                ]
                if note_columns:
                    extra_notes = (
                        meta[note_columns]
                        .fillna("")
                        .astype(str)
                        .agg(";".join, axis=1)
                        .str.strip(";")
                    )
                    meta["notes"] = np.where(
                        extra_notes.ne(""),
                        meta["notes"].fillna("").astype(str)
                        + ";"
                        + extra_notes,
                        meta["notes"],
                    )
                    meta["notes"] = (
                        meta["notes"]
                        .fillna("")
                        .astype(str)
                        .str.strip(";")
                    )

    return meta.sort_values(
        ["drug_rank", "drug_name"],
        ascending=[True, True],
    ).reset_index(drop=True)


def make_network(path_value: str) -> tuple[pd.DataFrame, dict]:
    columns = ["gene_a", "gene_b", "weight"]
    path = optional_path(path_value)
    if path is None:
        return (
            pd.DataFrame(columns=columns),
            {
                "status": "not_provided",
                "n_output_edges": 0,
            },
        )

    raw = pd.read_csv(path, low_memory=False)
    a_col = find_col(
        raw,
        [
            "gene_a",
            "geneA",
            "source",
            "protein1",
            "protein_a",
            "node1",
            "preferredName_A",
        ],
    )
    b_col = find_col(
        raw,
        [
            "gene_b",
            "geneB",
            "target",
            "protein2",
            "protein_b",
            "node2",
            "preferredName_B",
        ],
    )
    weight_col = find_col(
        raw,
        [
            "weight",
            "score",
            "combined_score",
            "confidence",
            "edge_weight",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "gene_a": raw[a_col].map(normalize_gene),
            "gene_b": raw[b_col].map(normalize_gene),
            "weight": (
                pd.to_numeric(raw[weight_col], errors="coerce")
                if weight_col
                else 1.0
            ),
        }
    )
    out = out.dropna(subset=["weight"])

    if len(out) and float(out["weight"].max()) > 10:
        out["weight"] = out["weight"] / 1000.0

    out = out[
        out["gene_a"].ne("")
        & out["gene_b"].ne("")
        & out["gene_a"].ne(out["gene_b"])
        & out["weight"].gt(0)
    ].copy()

    left = out[["gene_a", "gene_b"]].min(axis=1)
    right = out[["gene_a", "gene_b"]].max(axis=1)
    out["gene_a"] = left
    out["gene_b"] = right
    out = (
        out.groupby(["gene_a", "gene_b"], as_index=False)["weight"]
        .max()
        .sort_values(["gene_a", "gene_b"])
        .reset_index(drop=True)
    )

    return out, {
        "status": "ok",
        "source_path": str(path),
        "n_input_rows": int(len(raw)),
        "n_output_edges": int(len(out)),
        "n_unique_genes": int(
            len(set(out["gene_a"]) | set(out["gene_b"]))
        ),
    }


def make_alias_mapping(path_value: str) -> tuple[pd.DataFrame, dict]:
    columns = ["alias", "gene_symbol", "mapping_source"]
    path = optional_path(path_value)
    if path is None:
        return (
            pd.DataFrame(columns=columns),
            {"status": "not_provided", "n_aliases": 0},
        )

    raw = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    alias_col = find_col(
        raw,
        ["gene_claim_name", "alias", "source_id"],
    )
    symbol_col = find_col(
        raw,
        ["gene_name", "gene_symbol", "symbol"],
    )

    out = pd.DataFrame(
        {
            "alias": raw[alias_col].map(normalize_gene),
            "gene_symbol": raw[symbol_col].map(normalize_gene),
        }
    )
    out = out[
        out["alias"].ne("")
        & out["gene_symbol"].ne("")
        & out["alias"].ne(out["gene_symbol"])
    ].copy()

    counts = (
        out.groupby(["alias", "gene_symbol"])
        .size()
        .reset_index(name="n")
        .sort_values(
            ["alias", "n", "gene_symbol"],
            ascending=[True, False, True],
        )
        .drop_duplicates("alias")
    )
    result = counts[["alias", "gene_symbol"]].copy()
    result["mapping_source"] = "DGIdb genes.tsv"

    return result, {
        "status": "ok",
        "source_path": str(path),
        "n_aliases": int(len(result)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SigCom metadata.tsv into standardized 37a inputs."
        )
    )
    parser.add_argument(
        "--metadata",
        default=METADATA_TSV_PATH,
        help="Path to metadata.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--dgidb-clean",
        default=DGIDB_CLEAN_INTERACTIONS_PATH,
    )
    parser.add_argument(
        "--dgidb-pairs",
        default=DGIDB_PAIR_TABLE_PATH,
    )
    parser.add_argument(
        "--drug-entity-qc",
        default=DRUG_ENTITY_QC_PATH,
    )
    parser.add_argument(
        "--ppi",
        default=PPI_EDGE_PATH,
    )
    parser.add_argument(
        "--dgidb-genes",
        default=DGIDB_GENES_TSV_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"找不到 metadata.tsv：{metadata_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "perturbation_signatures": (
            output_dir / "drug_perturbation_signature.csv"
        ),
        "best_signatures": output_dir / "drug_best_signature.csv",
        "drug_metadata": output_dir / "drug_metadata.csv",
        "drug_targets": output_dir / "drug_target_actions.csv",
        "network": output_dir / "gene_network_edges.csv",
        "aliases": output_dir / "gene_alias_mapping.csv",
        "repair_audit": output_dir / "metadata_repair_audit.csv",
        "rejected_rows": output_dir / "metadata_rejected_rows.csv",
        "qc": output_dir / "preparation_qc.json",
        "manifest": output_dir / "37a_input_manifest.json",
    }

    if not OVERWRITE:
        existing = [
            str(path)
            for path in output_paths.values()
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "输出文件已存在且 OVERWRITE=False：\n"
                + "\n".join(existing)
            )

    repaired, repair_audit, rejected, repair_info = (
        parse_metadata_tsv(metadata_path)
    )
    signatures, signature_info = standardize_perturbation_signatures(
        repaired
    )
    if signatures.empty:
        raise ValueError("筛选后没有任何 perturbation signature。")

    best = select_best_signature_per_drug(signatures)
    signature_meta = make_signature_drug_metadata(signatures, best)

    raw_targets, target_source_path, source_info = (
        load_optional_drug_source(
            args.dgidb_clean,
            args.dgidb_pairs,
        )
    )
    targets, dgidb_meta, target_info = (
        make_targets_and_dgidb_metadata(raw_targets)
    )

    drug_metadata = merge_drug_metadata(
        signature_meta,
        dgidb_meta,
        args.drug_entity_qc,
    )
    network, network_info = make_network(args.ppi)
    aliases, alias_info = make_alias_mapping(args.dgidb_genes)

    signatures.to_csv(
        output_paths["perturbation_signatures"],
        index=False,
        encoding="utf-8-sig",
    )
    best.to_csv(
        output_paths["best_signatures"],
        index=False,
        encoding="utf-8-sig",
    )
    drug_metadata.to_csv(
        output_paths["drug_metadata"],
        index=False,
        encoding="utf-8-sig",
    )
    targets.to_csv(
        output_paths["drug_targets"],
        index=False,
        encoding="utf-8-sig",
    )
    network.to_csv(
        output_paths["network"],
        index=False,
        encoding="utf-8-sig",
    )
    aliases.to_csv(
        output_paths["aliases"],
        index=False,
        encoding="utf-8-sig",
    )
    repair_audit.to_csv(
        output_paths["repair_audit"],
        index=False,
        encoding="utf-8-sig",
    )
    rejected.to_csv(
        output_paths["rejected_rows"],
        index=False,
        encoding="utf-8-sig",
    )

    qc = {
        "status": "ok",
        "metadata": repair_info,
        "perturbation_signatures": signature_info,
        "drug_targets": {
            **source_info,
            **target_info,
            "selected_source_path": (
                str(target_source_path)
                if target_source_path is not None
                else None
            ),
        },
        "network": network_info,
        "aliases": alias_info,
        "important_note": (
            "metadata.tsv contains signature-level enrichment statistics, "
            "not per-gene drug-versus-vehicle effects. No gene-level "
            "signed_effect was fabricated."
        ),
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
        },
    }
    output_paths["qc"].write_text(
        json.dumps(qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "input_mode": "signature_level",
        "primary_37a_inputs": {
            "drug_perturbation_signature": str(
                output_paths["perturbation_signatures"]
            ),
            "drug_best_signature": str(
                output_paths["best_signatures"]
            ),
            "drug_metadata": str(
                output_paths["drug_metadata"]
            ),
            "drug_target_actions": str(
                output_paths["drug_targets"]
            ),
            "gene_network_edges": str(
                output_paths["network"]
            ),
            "gene_alias_mapping": str(
                output_paths["aliases"]
            ),
        },
        "recommended_signature_fields": {
            "drug_id": "drug_id",
            "drug_name": "drug_name",
            "signature_id": "perturbation_signature_id",
            "context_id": "context_id",
            "reversal_score": "query_reversal_score",
            "evidence_weight": "evidence_weight",
            "qvalue": "qvalue_strict",
            "cell_line": "cell_line",
            "dose": "dose",
            "timepoint": "timepoint",
            "data_level": "data_level",
        },
        "schema_warning": (
            "Do not interpret query_reversal_score as a per-gene "
            "signed_effect. It is a signature-level query score."
        ),
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print("\n完成。37a 优先读取：")
    print(output_paths["perturbation_signatures"])
    print(output_paths["best_signatures"])
    print(output_paths["drug_metadata"])
    print(output_paths["manifest"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
