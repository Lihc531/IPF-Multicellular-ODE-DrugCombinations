#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36a_extract_global_sigcom_up_down_FINAL.py

从 35b/36a 输出的 08_drug_pairing_signatures_long.csv 中，构建一次性上传
SigCom LINCS Up/Down Signature Search 的全局合并疾病签名。

方向约定
--------
SigCom 的 Up Genes / Down Genes 表示疾病或待逆转状态本身的表达方向。

若输入 reversal_direction 表示“期望药物造成的方向”：
    reversal_direction == DOWN  -> 疾病中偏高 -> SigCom Up Genes
    reversal_direction == UP    -> 疾病中偏低 -> SigCom Down Genes

主要输出
--------
sigcom_all_up.txt
sigcom_all_down.txt
sigcom_all_query_copy_paste.txt
sigcom_all_query.csv
gene_direction_support_long.csv
global_direction_conflicts.csv
global_query_qc.json
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2.0.0-FINAL-GLOBAL-MERGED-SIGCOM-QUERY"

# =============================================================================
# 用户设置：通常只修改这里
# =============================================================================
INPUT_SIGNATURE_PATH = (
    r"E:\PythonProject\results\36a_OT_FINALIZED_FOR_DRUG_PAIRING"
    r"\08_drug_pairing_signatures_long.csv"
)
OUTPUT_DIR = r"E:\PythonProject\drug_data\sigcom_global_query"

# reversal_direction：输入方向表示希望药物造成的逆转方向，需要反转为疾病方向
# disease_direction：输入方向已经表示疾病中的 UP/DOWN，不再反转
INPUT_DIRECTION_MODE = "reversal_direction"
DIRECTION_COLUMN = ""  # 留空自动识别

MIN_GENE_WEIGHT = 0.0
ALLOWED_ANALYSIS_ROLES: list[str] = []
ALLOWED_PPI_TIERS: list[str] = []

# 同一基因在不同 signature 中落入相反方向时：
# net_weight：比较两个方向累计权重，保留权重更大的方向；完全相等则排除
# exclude：两个方向均排除
GLOBAL_CONFLICT_POLICY = "net_weight"

# 是否给 PRIMARY/SENSITIVITY 不同权重。为保持原始全局汇总逻辑，默认关闭。
USE_ANALYSIS_ROLE_WEIGHTS = False
ANALYSIS_ROLE_WEIGHTS = {
    "primary": 1.0,
    "sensitivity": 0.5,
}
DEFAULT_ANALYSIS_ROLE_WEIGHT = 1.0

# 可选：每个最终方向最多保留多少个基因；None 表示全部保留
TOP_N_PER_DIRECTION: Optional[int] = None
OVERWRITE = True
# =============================================================================


UP_TOKENS = {
    "UP", "UPREGULATED", "UP_REGULATED", "INCREASE", "INCREASED",
    "ACTIVATE", "ACTIVATED", "POSITIVE", "+", "1", "上调", "升高", "增加",
}
DOWN_TOKENS = {
    "DOWN", "DOWNREGULATED", "DOWN_REGULATED", "DECREASE", "DECREASED",
    "INHIBIT", "INHIBITED", "NEGATIVE", "-", "-1", "下调", "降低", "减少",
}


def normalize_col_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    normalized = {normalize_col_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"未找到必需列。候选列：{list(candidates)}；当前列：{list(df.columns)}"
        )
    return None


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", "", str(value).strip().upper())
    if text in {"", "NAN", "NONE", "NA", "NULL"}:
        return ""
    return text


def normalize_direction(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[\s\-]+", "_", text)
    if text in UP_TOKENS:
        return "UP"
    if text in DOWN_TOKENS:
        return "DOWN"
    if re.search(r"(^|_)UP($|_)|UPREGULAT|INCREAS|ACTIVAT|POSITIVE", text):
        return "UP"
    if re.search(r"(^|_)DOWN($|_)|DOWNREGULAT|DECREAS|INHIBIT|NEGATIVE", text):
        return "DOWN"
    return ""


def clean_allowed(values: Iterable[str]) -> set[str]:
    return {str(v).strip().casefold() for v in values if str(v).strip()}


def write_lines(path: Path, values: Iterable[str]) -> None:
    values = list(values)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        if not OVERWRITE:
            raise FileExistsError(f"输出目录已有文件且 OVERWRITE=False：{path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def role_weight(value: object) -> float:
    if not USE_ANALYSIS_ROLE_WEIGHTS:
        return 1.0
    key = str(value).strip().casefold()
    return float(ANALYSIS_ROLE_WEIGHTS.get(key, DEFAULT_ANALYSIS_ROLE_WEIGHT))


def main() -> None:
    input_path = Path(INPUT_SIGNATURE_PATH)
    output_dir = Path(OUTPUT_DIR)
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{input_path}\n请修改脚本顶部 INPUT_SIGNATURE_PATH。"
        )
    prepare_output_dir(output_dir)

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError("输入 CSV 为空。")

    signature_id_col = find_column(
        df, ["signature_id", "signatureid", "sig_id", "signature"], required=False
    )
    signature_name_col = find_column(
        df, ["signature_name", "signaturename", "sig_name", "name"], required=False
    )
    gene_col = find_column(df, ["gene_symbol", "genesymbol", "gene", "symbol"])

    if DIRECTION_COLUMN.strip():
        direction_col = find_column(df, [DIRECTION_COLUMN])
    elif INPUT_DIRECTION_MODE == "reversal_direction":
        direction_col = find_column(
            df,
            ["reversal_direction", "reversaldirection", "desired_drug_direction", "drug_direction"],
        )
    elif INPUT_DIRECTION_MODE == "disease_direction":
        direction_col = find_column(
            df,
            ["disease_direction", "diseasedirection", "expression_direction", "direction", "regulation"],
        )
    else:
        raise ValueError(
            "INPUT_DIRECTION_MODE 只能是 reversal_direction 或 disease_direction。"
        )

    weight_col = find_column(
        df,
        ["gene_weight_normalized", "geneweightnormalized", "gene_weight", "weight", "score"],
        required=False,
    )
    analysis_role_col = find_column(df, ["analysis_role", "analysisrole"], required=False)
    ppi_tier_col = find_column(
        df, ["ppi_driver_tier", "ppidrivertier", "driver_tier", "tier"], required=False
    )
    signature_class_col = find_column(
        df, ["signature_class", "signatureclass"], required=False
    )
    disease_scope_col = find_column(df, ["disease_scope", "diseasescope"], required=False)
    celltype_scope_col = find_column(df, ["celltype_scope", "celltypescope"], required=False)
    tier_mode_col = find_column(df, ["tier_mode", "tiermode"], required=False)

    if signature_id_col is None:
        df["_signature_id"] = (
            df[signature_name_col].fillna("").astype(str).str.strip()
            if signature_name_col
            else "OT_GLOBAL_SOURCE"
        )
        signature_id_col = "_signature_id"
    if signature_name_col is None:
        df["_signature_name"] = df[signature_id_col].fillna("").astype(str).str.strip()
        signature_name_col = "_signature_name"

    work = pd.DataFrame(index=df.index)
    work["source_row_number"] = np.arange(2, len(df) + 2)
    work["signature_id"] = df[signature_id_col].fillna("").astype(str).str.strip()
    work["signature_name"] = df[signature_name_col].fillna("").astype(str).str.strip()
    work["gene_symbol"] = df[gene_col].map(normalize_gene)
    work["input_direction_raw"] = df[direction_col]
    work["input_direction"] = df[direction_col].map(normalize_direction)
    work["gene_weight"] = (
        pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        if weight_col
        else 1.0
    )
    work["analysis_role"] = (
        df[analysis_role_col].fillna("").astype(str).str.strip()
        if analysis_role_col else ""
    )
    work["ppi_driver_tier"] = (
        df[ppi_tier_col].fillna("").astype(str).str.strip()
        if ppi_tier_col else ""
    )
    work["signature_class"] = (
        df[signature_class_col].fillna("").astype(str).str.strip()
        if signature_class_col else ""
    )
    work["disease_scope"] = (
        df[disease_scope_col].fillna("").astype(str).str.strip()
        if disease_scope_col else ""
    )
    work["celltype_scope"] = (
        df[celltype_scope_col].fillna("").astype(str).str.strip()
        if celltype_scope_col else ""
    )
    work["tier_mode"] = (
        df[tier_mode_col].fillna("").astype(str).str.strip()
        if tier_mode_col else ""
    )

    if INPUT_DIRECTION_MODE == "reversal_direction":
        work["sigcom_query_direction"] = work["input_direction"].map(
            {"DOWN": "UP", "UP": "DOWN"}
        )
    else:
        work["sigcom_query_direction"] = work["input_direction"]

    invalid_gene = work["gene_symbol"].eq("")
    invalid_direction = ~work["sigcom_query_direction"].isin(["UP", "DOWN"])
    invalid_signature = work["signature_id"].eq("")
    below_weight = work["gene_weight"] < float(MIN_GENE_WEIGHT)

    excluded_parts = []
    invalid = invalid_gene | invalid_direction | invalid_signature | below_weight
    if invalid.any():
        tmp = work.loc[invalid].copy()
        reasons = np.full(len(tmp), "", dtype=object)
        idx = tmp.index
        reasons = pd.Series(reasons, index=idx)
        reasons.loc[invalid_gene.loc[idx]] += "invalid_gene;"
        reasons.loc[invalid_direction.loc[idx]] += "invalid_direction;"
        reasons.loc[invalid_signature.loc[idx]] += "invalid_signature;"
        reasons.loc[below_weight.loc[idx]] += "below_min_weight;"
        tmp["exclusion_reason"] = reasons
        excluded_parts.append(tmp)
    work = work.loc[~invalid].copy()

    allowed_roles = clean_allowed(ALLOWED_ANALYSIS_ROLES)
    if allowed_roles:
        if analysis_role_col is None:
            raise KeyError("设置了 ALLOWED_ANALYSIS_ROLES，但输入没有 analysis_role 列。")
        keep = work["analysis_role"].str.casefold().isin(allowed_roles)
        tmp = work.loc[~keep].copy()
        tmp["exclusion_reason"] = "analysis_role_filter;"
        excluded_parts.append(tmp)
        work = work.loc[keep].copy()

    allowed_tiers = clean_allowed(ALLOWED_PPI_TIERS)
    if allowed_tiers:
        if ppi_tier_col is None:
            raise KeyError("设置了 ALLOWED_PPI_TIERS，但输入没有 PPI tier 列。")
        keep = work["ppi_driver_tier"].str.casefold().isin(allowed_tiers)
        tmp = work.loc[~keep].copy()
        tmp["exclusion_reason"] = "ppi_tier_filter;"
        excluded_parts.append(tmp)
        work = work.loc[keep].copy()

    if work.empty:
        raise ValueError("过滤后没有可用基因。")

    # 同一 signature 内、同一基因、同一方向仅保留权重最高的一行。
    work = work.sort_values(
        ["signature_id", "gene_symbol", "sigcom_query_direction", "gene_weight", "source_row_number"],
        ascending=[True, True, True, False, True],
    )
    duplicate_mask = work.duplicated(
        ["signature_id", "gene_symbol", "sigcom_query_direction"], keep="first"
    )
    if duplicate_mask.any():
        tmp = work.loc[duplicate_mask].copy()
        tmp["exclusion_reason"] = "duplicate_same_signature_direction;"
        excluded_parts.append(tmp)
    work = work.loc[~duplicate_mask].copy()

    # 同一 signature 内方向冲突一律排除，避免单个 signature 自相矛盾。
    within_counts = work.groupby(["signature_id", "gene_symbol"])["sigcom_query_direction"].nunique()
    within_conflict_keys = set(within_counts[within_counts > 1].index)
    if within_conflict_keys:
        mask = work.apply(
            lambda r: (r["signature_id"], r["gene_symbol"]) in within_conflict_keys,
            axis=1,
        )
        tmp = work.loc[mask].copy()
        tmp["exclusion_reason"] = "within_signature_direction_conflict;"
        excluded_parts.append(tmp)
        work = work.loc[~mask].copy()

    work["analysis_role_weight"] = work["analysis_role"].map(role_weight)
    work["effective_gene_weight"] = work["gene_weight"] * work["analysis_role_weight"]

    # 先按 gene × direction 汇总支持。
    direction_support = (
        work.groupby(["gene_symbol", "sigcom_query_direction"], as_index=False)
        .agg(
            total_weight=("effective_gene_weight", "sum"),
            raw_gene_weight_sum=("gene_weight", "sum"),
            max_gene_weight=("gene_weight", "max"),
            n_supporting_signatures=("signature_id", "nunique"),
            n_primary_signatures=("analysis_role", lambda s: int(s.str.casefold().eq("primary").sum())),
            source_signatures=("signature_id", lambda s: ";".join(sorted(set(map(str, s))))),
            source_signature_names=("signature_name", lambda s: ";".join(sorted(set(x for x in map(str, s) if x)))),
            signature_classes=("signature_class", lambda s: ";".join(sorted(set(x for x in map(str, s) if x)))),
            disease_scopes=("disease_scope", lambda s: ";".join(sorted(set(x for x in map(str, s) if x)))),
            celltype_scopes=("celltype_scope", lambda s: ";".join(sorted(set(x for x in map(str, s) if x)))),
            ppi_tiers=("ppi_driver_tier", lambda s: ";".join(sorted(set(x for x in map(str, s) if x)))),
        )
    )

    final_rows = []
    conflict_rows = []
    for gene, group in direction_support.groupby("gene_symbol", sort=False):
        by_dir = {r["sigcom_query_direction"]: r for _, r in group.iterrows()}
        up = by_dir.get("UP")
        down = by_dir.get("DOWN")

        if up is not None and down is not None:
            up_weight = float(up["total_weight"])
            down_weight = float(down["total_weight"])
            retained = None
            decision = "excluded_both"
            if GLOBAL_CONFLICT_POLICY == "net_weight":
                if not math.isclose(up_weight, down_weight, rel_tol=1e-12, abs_tol=1e-12):
                    retained = up if up_weight > down_weight else down
                    decision = f"retained_{retained['sigcom_query_direction'].lower()}_by_net_weight"
            elif GLOBAL_CONFLICT_POLICY != "exclude":
                raise ValueError("GLOBAL_CONFLICT_POLICY 只能是 net_weight 或 exclude。")

            conflict_rows.append({
                "gene_symbol": gene,
                "up_total_weight": up_weight,
                "down_total_weight": down_weight,
                "up_supporting_signatures": int(up["n_supporting_signatures"]),
                "down_supporting_signatures": int(down["n_supporting_signatures"]),
                "policy": GLOBAL_CONFLICT_POLICY,
                "decision": decision,
                "retained_direction": "" if retained is None else retained["sigcom_query_direction"],
                "up_source_signatures": up["source_signatures"],
                "down_source_signatures": down["source_signatures"],
            })
            if retained is None:
                continue
            row = retained.to_dict()
            row["direction_conflict_present"] = True
            row["opposite_direction_weight"] = min(up_weight, down_weight)
            row["net_direction_weight"] = abs(up_weight - down_weight)
            final_rows.append(row)
        else:
            retained = up if up is not None else down
            row = retained.to_dict()
            row["direction_conflict_present"] = False
            row["opposite_direction_weight"] = 0.0
            row["net_direction_weight"] = float(row["total_weight"])
            final_rows.append(row)

    final = pd.DataFrame(final_rows)
    if final.empty:
        raise ValueError("全局方向冲突处理后没有可上传基因。")

    final = final.sort_values(
        ["sigcom_query_direction", "net_direction_weight", "n_supporting_signatures", "max_gene_weight", "gene_symbol"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    final["rank_within_direction"] = final.groupby("sigcom_query_direction").cumcount() + 1

    if TOP_N_PER_DIRECTION is not None:
        if int(TOP_N_PER_DIRECTION) <= 0:
            raise ValueError("TOP_N_PER_DIRECTION 必须为正整数或 None。")
        final = final.loc[final["rank_within_direction"] <= int(TOP_N_PER_DIRECTION)].copy()

    up = final.loc[final["sigcom_query_direction"].eq("UP")].sort_values("rank_within_direction")
    down = final.loc[final["sigcom_query_direction"].eq("DOWN")].sort_values("rank_within_direction")
    up_genes = up["gene_symbol"].tolist()
    down_genes = down["gene_symbol"].tolist()

    write_lines(output_dir / "sigcom_all_up.txt", up_genes)
    write_lines(output_dir / "sigcom_all_down.txt", down_genes)

    copy_text = "\n".join([
        "SIGCOM UP GENES（疾病中上调；粘贴到网页 Up Genes）",
        "=" * 72,
        "\n".join(up_genes) if up_genes else "[EMPTY]",
        "",
        "SIGCOM DOWN GENES（疾病中下调；粘贴到网页 Down Genes）",
        "=" * 72,
        "\n".join(down_genes) if down_genes else "[EMPTY]",
        "",
    ])
    (output_dir / "sigcom_all_query_copy_paste.txt").write_text(copy_text, encoding="utf-8")

    final_columns = [
        "gene_symbol", "sigcom_query_direction", "rank_within_direction",
        "total_weight", "net_direction_weight", "opposite_direction_weight",
        "max_gene_weight", "n_supporting_signatures", "n_primary_signatures",
        "direction_conflict_present", "source_signatures", "source_signature_names",
        "signature_classes", "disease_scopes", "celltype_scopes", "ppi_tiers",
    ]
    final[final_columns].to_csv(output_dir / "sigcom_all_query.csv", index=False)
    work.sort_values(
        ["gene_symbol", "sigcom_query_direction", "signature_id"]
    ).to_csv(output_dir / "gene_direction_support_long.csv", index=False)
    pd.DataFrame(conflict_rows).to_csv(output_dir / "global_direction_conflicts.csv", index=False)

    excluded = pd.concat(excluded_parts, ignore_index=True) if excluded_parts else pd.DataFrame()
    excluded.to_csv(output_dir / "excluded_rows.csv", index=False)

    qc = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "input_direction_mode": INPUT_DIRECTION_MODE,
        "direction_column": direction_col,
        "gene_column": gene_col,
        "weight_column": weight_col,
        "global_conflict_policy": GLOBAL_CONFLICT_POLICY,
        "use_analysis_role_weights": USE_ANALYSIS_ROLE_WEIGHTS,
        "input_rows": int(len(df)),
        "retained_source_rows": int(len(work)),
        "n_source_signatures": int(work["signature_id"].nunique()),
        "n_unique_source_genes": int(work["gene_symbol"].nunique()),
        "n_global_direction_conflicts": int(len(conflict_rows)),
        "n_conflicts_retained_by_weight": int(sum(bool(x["retained_direction"]) for x in conflict_rows)),
        "n_conflicts_excluded": int(sum(not bool(x["retained_direction"]) for x in conflict_rows)),
        "n_sigcom_up_genes": int(len(up_genes)),
        "n_sigcom_down_genes": int(len(down_genes)),
        "n_final_unique_genes": int(final["gene_symbol"].nunique()),
        "overlap_between_final_up_down": int(len(set(up_genes) & set(down_genes))),
        "main_upload_files": {
            "up": str(output_dir / "sigcom_all_up.txt"),
            "down": str(output_dir / "sigcom_all_down.txt"),
        },
    }
    (output_dir / "global_query_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""SigCom 全局合并查询说明
========================
脚本版本：{SCRIPT_VERSION}

1. sigcom_all_up.txt
   粘贴到网页 Up Genes；表示疾病/待逆转状态中上调的基因。

2. sigcom_all_down.txt
   粘贴到网页 Down Genes；表示疾病/待逆转状态中下调的基因。

3. 当前方向模式：{INPUT_DIRECTION_MODE}
4. 当前跨 signature 冲突策略：{GLOBAL_CONFLICT_POLICY}

重要边界：
该文件把所有 OT signatures 合并成一次全局查询。它适合生成统一药物扰动
候选池，但会弱化疾病、细胞类型和 signature 层级的独立性。因此后续 37a
将全局 SigCom 分数只计算一次，OT-specific 机制仍由靶点和网络轨道评估。
"""
    (output_dir / "README_SigCom_global_query.txt").write_text(readme, encoding="utf-8")

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print("\n完成。网页上传文件：")
    print(output_dir / "sigcom_all_up.txt")
    print(output_dir / "sigcom_all_down.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
