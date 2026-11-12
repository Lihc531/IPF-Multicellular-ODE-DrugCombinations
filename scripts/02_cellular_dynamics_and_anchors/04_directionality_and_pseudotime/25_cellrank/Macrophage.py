#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import sys
import re

import pandas as pd
import numpy as np
from scipy.io import mmread


def check_and_install_packages():
    required_packages = [
        ("scanpy", "scanpy"),
        ("anndata", "anndata"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    ]

    print("正在检查依赖环境...")
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            print(f"✗ 缺失包: {import_name}，正在尝试安装...")
            try:
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--user",
                    "--no-cache-dir",
                    package_name,
                ])
                print(f"✓ 成功安装: {package_name}")
            except Exception as e:
                print(f"✗ 安装失败: {package_name}")
                print(f"  错误: {e}")
                print("  请手动安装缺失的包后重试。")
                sys.exit(1)


# =========================
# CONFIG：配置区
# =========================

EXPORT_DIR = r"G:\package\result\Rresults\All_cells_full_genes"
OUT_DIR = r"G:\package\result\Pythonresult\Macrophage"
AXIS_METADATA_CSV = r"E:\package\\24 export_small_exon_intron_from_tar\input\03_cell_metadata_with_axis_v2_module_scores.csv"
GENE_ID_TO_SYMBOL_CSV = r"G:\package\ensembl_to_symbol.csv"

GENE_ID_COL_IN_MAPPING = "gene_id"
GENE_SYMBOL_COL_IN_MAPPING = "gene_symbol"
USE_GENE_SYMBOL_AS_VARNAMES = True

# Macrophage 细胞数很大时，不建议默认直接跑 CellRank GPCCA，
# 否则容易因 dense transition matrix 导致长期停滞或内存爆炸。
# 先完整跑 DPT + diagnostics；若 diagnostics 稳定，再设置为 True 并使用子集。
RUN_CELLRANK = False
RUN_SCVELO_VELOCITY = True

# 如果后续重新打开 RUN_CELLRANK=True，请只在子集上运行。
CELLRANK_USE_SUBSAMPLE = False
CELLRANK_MAX_CELLS = 10000
CELLRANK_RANDOM_SEED = 123
CELLRANK_STRATIFY_COLS = [
    "Disease_Identity",
    "axis_v2_stage",
    "Subclass_Cell_Identity",
    "Manuscript_Identity",
]

N_PCS = 30
N_NEIGHBORS = 15
N_TOP_GENES = 2000

ROOT_MODE = "lowest_fibrosis"

AXIS_CELL_ID_COL = "cell_id"
AXIS_SAMPLE_COL = "sample_id"
AXIS_BARCODE_COL = "CellBarcode_Identity"

AXIS_SCORE_COL = "axis_v2_score"
FIBROSIS_SCORE_COL = "fibrosis_composite_score"

ROOT_LOW_QUANTILE = 0.02

SAVE_H5AD = True
SAVE_H5AD_AFTER_SCVELO = False

RUN_EXTRA_DIAGNOSTICS = True

# =========================
# TARGET CELL / SUBSET CONFIG
# =========================
# 做其他细胞时，优先修改这 4 个配置：
#   1) TARGET_CELLTYPE
#   2) OUT_DIR
#   3) EXPORT_DIR
#   4) MARKER_GENES
TARGET_CELLTYPE = "Macrophage"
CELLTYPE_COL = "celltype_marker_based"

# subtype 排除：
# Macrophage 运行时建议排除 Outlier / MT-tRNAs / CellCycle / 非 macrophage 混入细胞。
# 做 Endothelial 时可把下面的 terms 改成 ["Lymphatic"] 或设置 EXCLUDE_SUBTYPES=False。
EXCLUDE_SUBTYPES = True

# 包含型排除：只要 subtype 注释中包含这些词，就会被排除。
EXCLUDE_CONTAINS_TERMS = [
    "Outlier",
    "MT-tRNAs",
    "CellCycle",
    "Dendritic",
    "Innate_lymphoid",
    "cDC",
    "pDC",
    "Monocyte",
]

# 精确型排除：只有 subtype 注释完全等于这些词，才会被排除。
# 注意不要把 "T" 放进包含型排除，否则所有包含字母 T 的 subtype 都可能被误伤。
EXCLUDE_EXACT_TERMS = [
    "T",
    "NK",
    "Mast",
    "T_Regulatory",
    "T_Cytotoxic",
    "T_Regulatory_A",
    "T_Regulatory_B",
    "T_Cytotoxic_A",
    "T_Cytotoxic_B",
    "T_Cytotoxic_C",
]

# 兼容旧代码变量名，避免 NameError。
# 后续逻辑主要使用 EXCLUDE_CONTAINS_TERMS / EXCLUDE_EXACT_TERMS。
EXCLUDE_SUBTYPE_TERMS = EXCLUDE_CONTAINS_TERMS

# 脚本会自动在这些列里寻找 subtype 注释；找到后再执行排除。
SUBTYPE_COL_CANDIDATES = [
    "Subclass_Cell_Identity",
    "Manuscript_Identity",
    "Celltype_Identity",
    "celltype_fine",
    "celltype_subtype",
    "subtype",
    "annotation",
    "Annotation",
]

# 用于检查 pseudotime 是否被 QC / count / sample / subtype 驱动
DIAGNOSTIC_COLOR_COLS = [
    "dpt_pseudotime",
    "axis_v2_score",
    "fibrosis_composite_score",
    "Disease_Identity",
    "sample_id",
    "Subject_Identity",
    "total_counts_raw",
    "spliced_counts_raw",
    "unspliced_counts_raw",
    "axis_v2_stage",
    "celltype_marker_based",
    "Subclass_Cell_Identity",
    "Manuscript_Identity",
]

# Marker genes：做其他细胞时改这里。
# 当前配置为 Macrophage markers。
# 如果改回 Endothelial，请使用：
# ["PECAM1", "VWF", "KDR", "FLT1", "CLDN5", "RAMP2", "CA4", "ACKR1",
#  "SELE", "VCAM1", "ICAM1", "COL15A1", "PLVAP", "EMCN", "ESAM",
#  "PROX1", "PDPN", "LYVE1", "FLT4"]
MARKER_GENES = [
    "LYZ", "LST1", "TYROBP", "FCER1G",
    "CD68", "CD163", "MRC1", "MARCO",
    "FABP4", "APOE", "C1QA", "C1QB", "C1QC",
    "S100A8", "S100A9", "FCN1",
    "IL1B", "TNF", "CCL2", "CCL3", "CCL4",
    "MKI67", "TOP2A",
]

# 用 top pseudotime 细胞定义“末端区域”，用于统计它来自哪些 sample / disease / subtype
TERMINAL_PSEUDOTIME_QUANTILE = 0.95

ALLOW_BARCODE_ONLY_MERGE = False

MIN_METADATA_MATCHED_CELLS = 100

# CellRank 输出解释增强
TOP_FATE_CELLS_N = 50
SAVE_CELLRANK_TOP_CELLS = True
SAVE_CELLRANK_ASSIGNMENTS = True


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_index_unique(names):
    """
    让 index / gene symbol 唯一化，避免 AnnData var_names 重复。

    例如：
    ACTB, ACTB -> ACTB, ACTB-1
    """
    seen = {}
    unique_names = []

    for name in names:
        name = str(name).strip()

        if name == "" or name.lower() == "nan":
            name = "unknown"

        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}-{seen[name]}")

    return pd.Index(unique_names)

def sanitize_dataframe_for_h5ad(df):
    df = df.copy()

    for col in df.columns:
        try:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)
        except Exception:
            df[col] = df[col].astype(str)

    return df


def sanitize_anndata_for_h5ad(adata):
    try:
        import anndata as ad
        ad.settings.allow_write_nullable_strings = True
    except Exception:
        pass

    adata.obs = sanitize_dataframe_for_h5ad(adata.obs)
    adata.var = sanitize_dataframe_for_h5ad(adata.var)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    return adata


def normalize_sample_id(x):
    """
    示例：
    GSM4058947_135I-b -> 135I-b
    135I-b -> 135I-b
    """
    s = str(x).strip()
    s = re.sub(r"^GSM\d+_", "", s)
    return s


def normalize_barcode(x):
    """
    示例：
    AAACCTGAGAAGATTC-1 -> AAACCTGAGAAGATTC
    001C_AAACCTGAGAAGATTC -> AAACCTGAGAAGATTC
    GSM4058905_244C__AAACCTGAGAAGATTC -> AAACCTGAGAAGATTC
    """
    s = str(x).strip()

    if "__" in s:
        s = s.split("__")[-1]

    if "_" in s:
        s = s.split("_")[-1]

    s = re.sub(r"-1$", "", s)

    return s


def add_merge_keys_obs(obs):
    obs = obs.copy()

    if "sample_id" not in obs.columns or "original_barcode" not in obs.columns:
        raise ValueError(
            "cells.csv / adata.obs 必须包含 sample_id 和 original_barcode。"
            "请先检查 R 导出的 cells.csv。"
        )

    obs["_sample_norm"] = obs["sample_id"].map(normalize_sample_id)
    obs["_barcode_norm"] = obs["original_barcode"].map(normalize_barcode)
    obs["_key_sample_barcode_norm"] = (
        obs["_sample_norm"].astype(str)
        + "__"
        + obs["_barcode_norm"].astype(str)
    )

    return obs


def add_merge_keys_meta(meta):
    meta = meta.copy()

    if AXIS_SAMPLE_COL not in meta.columns:
        raise ValueError(f"axis metadata 中找不到列：{AXIS_SAMPLE_COL}")

    if AXIS_BARCODE_COL not in meta.columns:
        raise ValueError(f"axis metadata 中找不到列：{AXIS_BARCODE_COL}")

    meta["_sample_norm"] = meta[AXIS_SAMPLE_COL].map(normalize_sample_id)
    meta["_barcode_norm"] = meta[AXIS_BARCODE_COL].map(normalize_barcode)
    meta["_key_sample_barcode_norm"] = (
        meta["_sample_norm"].astype(str)
        + "__"
        + meta["_barcode_norm"].astype(str)
    )

    return meta


def merge_meta_columns_into_obs(obs, meta_sub):
    """
    一次性 concat，避免反复 obs[col] = ... 造成 DataFrame fragmentation warning。
    """
    obs = obs.copy()
    meta_sub = meta_sub.copy()

    new_cols = [col for col in meta_sub.columns if col not in obs.columns]

    if len(new_cols) == 0:
        return obs

    meta_new = meta_sub[new_cols].copy()
    meta_new.index = obs.index

    obs = pd.concat([obs, meta_new], axis=1)

    return obs


def find_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]


def get_subtype_columns(obs):
    return find_existing_columns(obs, SUBTYPE_COL_CANDIDATES)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# 读取 R 导出文件
# ============================================================

def read_exported_mtx(export_dir):
    export_dir = Path(export_dir)

    total_path = export_dir / "total_counts.mtx"
    spliced_path = export_dir / "spliced_exon.mtx"
    unspliced_path = export_dir / "unspliced_intron.mtx"
    genes_path = export_dir / "genes.csv"
    cells_path = export_dir / "cells.csv"

    required = [
        total_path,
        spliced_path,
        unspliced_path,
        genes_path,
        cells_path,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing exported files:\n" + "\n".join(missing)
        )

    print("Reading Matrix Market files...")

    # R 导出通常是 genes x cells；AnnData 需要 cells x genes。
    X = mmread(str(total_path)).tocsr().T
    spliced = mmread(str(spliced_path)).tocsr().T
    unspliced = mmread(str(unspliced_path)).tocsr().T

    genes = pd.read_csv(genes_path, sep=None, engine="python")
    cells = pd.read_csv(cells_path, sep=None, engine="python")

    print(f"X shape cells x genes: {X.shape}")
    print(f"spliced shape cells x genes: {spliced.shape}")
    print(f"unspliced shape cells x genes: {unspliced.shape}")
    print(f"genes rows: {genes.shape[0]}")
    print(f"cells rows: {cells.shape[0]}")

    return X, spliced, unspliced, genes, cells


# ============================================================
# 构建 AnnData
# ============================================================

def build_anndata(X, spliced, unspliced, genes, cells):
    import anndata as ad

    obs = cells.copy()

    if "cell_id" not in obs.columns:
        obs["cell_id"] = [f"cell_{i}" for i in range(obs.shape[0])]

    obs["cell_id"] = obs["cell_id"].astype(str)
    obs.index = pd.Index(obs["cell_id"].values.astype(str))
    obs.index.name = None

    var = genes.copy()

    if "gene_name" in var.columns:
        gene_names = var["gene_name"].astype(str)
    elif "gene_id" in var.columns:
        gene_names = var["gene_id"].astype(str)
    else:
        gene_names = pd.Series([f"gene_{i}" for i in range(var.shape[0])])

    gene_names = make_index_unique(gene_names)

    var.index = gene_names
    var.index.name = None

    obs = sanitize_dataframe_for_h5ad(obs)
    var = sanitize_dataframe_for_h5ad(var)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced.copy()

    adata.obs["total_counts_raw"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["spliced_counts_raw"] = np.asarray(
        adata.layers["spliced"].sum(axis=1)
    ).ravel()
    adata.obs["unspliced_counts_raw"] = np.asarray(
        adata.layers["unspliced"].sum(axis=1)
    ).ravel()

    return adata

def strip_ensembl_version(x):
    """
    去除 Ensembl ID 版本号。

    例如：
    ENSG00000110680.15 -> ENSG00000110680
    ENSG00000110680    -> ENSG00000110680
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s == "" or s.lower() == "nan":
        return ""

    return s.split(".")[0]


def make_index_unique(names):
    """
    让 gene symbol 唯一化，避免 AnnData var_names 重复。

    例如：
    ACTB, ACTB -> ACTB, ACTB-1
    """
    seen = {}
    unique_names = []

    for name in names:
        name = str(name).strip()

        if name == "" or name.lower() == "nan":
            name = "unknown"

        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}-{seen[name]}")

    return unique_names


def apply_gene_symbol_mapping(adata):
    """
    将 adata.var_names 中的 Ensembl ID 映射为 gene symbol。

    功能：
    1. 自动去除 Ensembl 版本号，例如 ENSG00000110680.15 -> ENSG00000110680
    2. 从 GENE_ID_TO_SYMBOL_CSV 读取 gene_id -> gene_symbol
    3. 在 adata.var 中保留原始 ID 和映射结果
    4. 可选择将 adata.var_names 替换成 gene symbol
    """

    mapping_path = Path(GENE_ID_TO_SYMBOL_CSV)

    print("\n========== Gene symbol mapping ==========")

    if not mapping_path.exists():
        print(f"[Warning] Gene mapping file not found:")
        print(f"          {GENE_ID_TO_SYMBOL_CSV}")
        print("          Skip gene symbol mapping.")
        return adata

    print(f"Mapping file: {GENE_ID_TO_SYMBOL_CSV}")

    mapping = pd.read_csv(mapping_path, sep=None, engine="python")

    required_cols = [
        GENE_ID_COL_IN_MAPPING,
        GENE_SYMBOL_COL_IN_MAPPING,
    ]

    for col in required_cols:
        if col not in mapping.columns:
            raise ValueError(
                f"Mapping file does not contain required column: {col}\n"
                f"Current columns: {list(mapping.columns)}"
            )

    mapping = mapping[
        [
            GENE_ID_COL_IN_MAPPING,
            GENE_SYMBOL_COL_IN_MAPPING,
        ]
    ].copy()

    mapping[GENE_ID_COL_IN_MAPPING] = (
        mapping[GENE_ID_COL_IN_MAPPING]
        .map(strip_ensembl_version)
    )

    mapping[GENE_SYMBOL_COL_IN_MAPPING] = (
        mapping[GENE_SYMBOL_COL_IN_MAPPING]
        .astype(str)
        .str.strip()
    )

    mapping = mapping.dropna()

    mapping = mapping[
        mapping[GENE_ID_COL_IN_MAPPING] != ""
    ]

    mapping = mapping[
        mapping[GENE_SYMBOL_COL_IN_MAPPING] != ""
    ]

    mapping = mapping[
        mapping[GENE_SYMBOL_COL_IN_MAPPING].str.lower() != "nan"
    ]

    mapping = mapping.drop_duplicates(
        subset=[GENE_ID_COL_IN_MAPPING],
        keep="first",
    )

    id_to_symbol = dict(
        zip(
            mapping[GENE_ID_COL_IN_MAPPING],
            mapping[GENE_SYMBOL_COL_IN_MAPPING],
        )
    )

    original_var_names = pd.Index(adata.var_names.astype(str))
    ensembl_clean = original_var_names.map(strip_ensembl_version)

    mapped_symbols = pd.Series(ensembl_clean).map(id_to_symbol)

    n_total = int(adata.n_vars)
    n_mapped = int(mapped_symbols.notna().sum())
    mapping_rate = n_mapped / n_total * 100 if n_total > 0 else 0

    print(f"Total genes: {n_total}")
    print(f"Mapped genes: {n_mapped}")
    print(f"Mapping rate: {mapping_rate:.2f}%")

    adata.var["ensembl_id_original"] = original_var_names.values
    adata.var["ensembl_id_clean"] = ensembl_clean.values
    adata.var["gene_symbol_mapped"] = mapped_symbols.fillna("").values

    if USE_GENE_SYMBOL_AS_VARNAMES:
        new_var_names = []

        for ens_id, gene_symbol in zip(ensembl_clean, mapped_symbols):
            if pd.notna(gene_symbol) and str(gene_symbol).strip() != "":
                new_var_names.append(str(gene_symbol).strip())
            else:
                # 没有映射到 symbol 的基因，保留 Ensembl ID
                new_var_names.append(str(ens_id).strip())

        new_var_names = make_index_unique(new_var_names)

        adata.var_names = new_var_names

        print("adata.var_names has been replaced by gene symbols where available.")
        print("First 20 adata.var_names after mapping:")
        print(list(adata.var_names[:20]))

    else:
        print("USE_GENE_SYMBOL_AS_VARNAMES = False")
        print("adata.var_names was not replaced.")

    return adata

# ============================================================
# gene name 检查
# ============================================================

def save_gene_name_check(adata, out_dir):
    """
    输出 gene name 检查文件，用于确认 Ensembl ID 是否已经成功映射为 gene symbol。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(5000, adata.n_vars)

    df = pd.DataFrame({
        "adata_var_names": adata.var_names[:n_show],
    })

    if "ensembl_id_original" in adata.var.columns:
        df["ensembl_id_original"] = adata.var["ensembl_id_original"].values[:n_show]

    if "ensembl_id_clean" in adata.var.columns:
        df["ensembl_id_clean"] = adata.var["ensembl_id_clean"].values[:n_show]

    if "gene_symbol_mapped" in adata.var.columns:
        df["gene_symbol_mapped"] = adata.var["gene_symbol_mapped"].values[:n_show]

    out_file = out_dir / "gene_name_check.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved gene name check file: {out_file}")

def report_marker_gene_availability(adata, marker_genes, out_dir):
    """
    检查 marker genes 是否存在于 adata.var_names 中。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    var_names = set(adata.var_names.astype(str))

    rows = []

    for gene in marker_genes:
        rows.append({
            "marker_gene": gene,
            "found_in_adata_var_names": gene in var_names,
        })

    df = pd.DataFrame(rows)

    out_file = out_dir / "marker_gene_resolution_report.csv"
    df.to_csv(out_file, index=False)

    n_found = int(df["found_in_adata_var_names"].sum())
    n_total = int(len(df))

    print("\n========== Marker gene availability ==========")
    print(f"Found marker genes: {n_found} / {n_total}")
    print(df.to_string(index=False))
    print(f"Saved marker gene report: {out_file}")

    return df

# ============================================================
# 合并 axis_v2 metadata
# ============================================================

def merge_axis_metadata(adata):
    """
    匹配顺序：
    1. direct cell_id；
    2. normalized sample_id + barcode；
    3. barcode-only 默认禁止。
    """

    axis_path = Path(AXIS_METADATA_CSV)

    if not axis_path.exists():
        raise FileNotFoundError(
            f"AXIS_METADATA_CSV not found: {AXIS_METADATA_CSV}"
        )

    meta = pd.read_csv(axis_path, sep=None, engine="python")

    print("\n========== Merge axis metadata ==========")
    print(f"Axis metadata: {AXIS_METADATA_CSV}")
    print(f"Axis metadata shape: {meta.shape}")

    obs = adata.obs.copy()

    for col in obs.columns:
        if obs[col].dtype == "object":
            obs[col] = obs[col].astype(str)

    for col in meta.columns:
        if meta[col].dtype == "object":
            meta[col] = meta[col].astype(str)

    # --------------------------------------------------------
    # 1. 直接用 cell_id 匹配
    # --------------------------------------------------------
    if AXIS_CELL_ID_COL in meta.columns:
        meta_by_cell = meta.drop_duplicates(
            subset=[AXIS_CELL_ID_COL]
        ).copy()

        meta_by_cell.index = meta_by_cell[AXIS_CELL_ID_COL].astype(str)

        overlap = obs.index.intersection(meta_by_cell.index)

        print(f"Direct cell_id overlap: {len(overlap)} / {adata.n_obs}")

        if len(overlap) >= MIN_METADATA_MATCHED_CELLS:
            meta_sub = meta_by_cell.reindex(obs.index)
            matched = int(meta_sub[AXIS_CELL_ID_COL].notna().sum())

            print(f"Direct cell_id matched cells: {matched}")

            obs = merge_meta_columns_into_obs(obs, meta_sub)
            adata.obs = obs

            print("Merged axis metadata by direct cell_id.")
            return adata

    # --------------------------------------------------------
    # 2. 规范化 sample_id + barcode 匹配
    # --------------------------------------------------------
    obs2 = add_merge_keys_obs(obs)
    meta2 = add_merge_keys_meta(meta)

    meta_by_key = meta2.drop_duplicates(
        subset=["_key_sample_barcode_norm"]
    ).copy()

    meta_by_key.index = meta_by_key["_key_sample_barcode_norm"].astype(str)

    overlap = pd.Index(
        obs2["_key_sample_barcode_norm"].values
    ).intersection(meta_by_key.index)

    print(f"normalized sample_id + barcode overlap: {len(overlap)} / {adata.n_obs}")

    if len(overlap) >= MIN_METADATA_MATCHED_CELLS:
        meta_sub = meta_by_key.reindex(
            obs2["_key_sample_barcode_norm"].values
        )

        matched = int(meta_sub["_key_sample_barcode_norm"].notna().sum())
        print(f"normalized sample_id + barcode matched cells: {matched}")

        obs2 = merge_meta_columns_into_obs(obs2, meta_sub)

        obs2.drop(
            columns=[
                "_sample_norm",
                "_barcode_norm",
                "_key_sample_barcode_norm",
            ],
            inplace=True,
            errors="ignore",
        )

        adata.obs = obs2

        print("Merged axis metadata by normalized sample_id + barcode.")
        return adata

    # --------------------------------------------------------
    # 3. barcode-only 匹配：默认禁止
    # --------------------------------------------------------
    if not ALLOW_BARCODE_ONLY_MERGE:
        raise ValueError(
            "\nMetadata merge failed safely.\n"
            "Direct cell_id 和 normalized sample_id + barcode 都没有足够 overlap。\n"
            "脚本已停止，避免使用危险的 barcode-only merge。\n\n"
            "请检查：\n"
            "1. cells.csv 中 sample_id / original_barcode 的格式；\n"
            "2. axis metadata 中 sample_id / CellBarcode_Identity 的格式；\n"
            "3. 是否需要把 AXIS_SAMPLE_COL 或 AXIS_BARCODE_COL 改成正确列名。\n"
        )

    print(
        "WARNING: ALLOW_BARCODE_ONLY_MERGE = True. "
        "Barcode-only merge may be unsafe."
    )

    obs3 = obs.copy()
    obs3["_barcode_key"] = obs3["original_barcode"].map(normalize_barcode)

    meta3 = meta.copy()
    meta3["_barcode_key"] = meta3[AXIS_BARCODE_COL].map(normalize_barcode)

    meta_by_barcode = meta3.drop_duplicates(subset=["_barcode_key"]).copy()
    meta_by_barcode.index = meta_by_barcode["_barcode_key"].astype(str)

    overlap = pd.Index(obs3["_barcode_key"]).intersection(meta_by_barcode.index)

    print(f"Barcode-only overlap: {len(overlap)} / {adata.n_obs}")

    if len(overlap) >= MIN_METADATA_MATCHED_CELLS:
        meta_sub = meta_by_barcode.reindex(obs3["_barcode_key"].values)
        obs3 = merge_meta_columns_into_obs(obs3, meta_sub)

        obs3.drop(columns=["_barcode_key"], inplace=True, errors="ignore")

        adata.obs = obs3

        print("Merged axis metadata by barcode-only.")
        return adata

    raise ValueError(
        "Could not merge axis metadata. "
        "No reliable overlap found."
    )


# ============================================================
# 过滤目标细胞类型 / 排除 subtype
# ============================================================

def filter_target_celltype(adata):
    print("\n========== Filter target cell type ==========")

    if CELLTYPE_COL not in adata.obs.columns:
        raise ValueError(
            f"{CELLTYPE_COL} not found in adata.obs. "
            "说明 metadata 没有正确合并，不能继续筛选目标细胞类型。"
        )

    celltype = adata.obs[CELLTYPE_COL].astype(str)

    print("Cell type counts before filtering:")
    print(celltype.value_counts(dropna=False).head(30))

    mask = celltype == TARGET_CELLTYPE

    n_target = int(mask.sum())
    n_total = int(adata.n_obs)

    print(f"Filtering target cell type: {TARGET_CELLTYPE}")
    print(f"Target cells: {n_target} / {n_total}")

    if n_target == 0:
        raise ValueError(
            f"No cells found for TARGET_CELLTYPE = {TARGET_CELLTYPE!r}. "
            f"Please check {CELLTYPE_COL} values."
        )

    adata = adata[mask].copy()

    print("Cell type counts after filtering:")
    print(adata.obs[CELLTYPE_COL].astype(str).value_counts(dropna=False))

    return adata


def exclude_unwanted_subtypes(adata, out_dir):
    """
    排除不希望进入轨迹分析的 subtype。

    使用两类规则：
    1. EXCLUDE_CONTAINS_TERMS：包含型匹配，例如 Outlier / MT-tRNAs / CellCycle；
    2. EXCLUDE_EXACT_TERMS：精确匹配，例如 T / NK / Mast，避免 "T" 误伤所有含字母 T 的名称。
    """
    if not EXCLUDE_SUBTYPES:
        print("\nSubtype exclusion skipped: EXCLUDE_SUBTYPES = False")
        return adata

    print("\n========== Exclude unwanted subtypes ==========")

    subtype_cols = get_subtype_columns(adata.obs)

    if len(subtype_cols) == 0:
        print("No subtype annotation columns found. Skipping subtype exclusion.")
        return adata

    print("Subtype annotation columns found:")
    print(subtype_cols)

    print(f"EXCLUDE_CONTAINS_TERMS: {EXCLUDE_CONTAINS_TERMS}")
    print(f"EXCLUDE_EXACT_TERMS: {EXCLUDE_EXACT_TERMS}")

    exclude_mask = pd.Series(False, index=adata.obs.index)

    for col in subtype_cols:
        values = adata.obs[col].astype(str)

        print(f"\nTop values in {col}:")
        print(values.value_counts(dropna=False).head(30))

        # 包含型排除
        for term in EXCLUDE_CONTAINS_TERMS:
            term_mask = values.str.contains(
                term,
                case=False,
                na=False,
                regex=False,
            )
            exclude_mask = exclude_mask | term_mask

        # 精确型排除
        if len(EXCLUDE_EXACT_TERMS) > 0:
            exact_mask = values.isin(EXCLUDE_EXACT_TERMS)
            exclude_mask = exclude_mask | exact_mask

    n_exclude = int(exclude_mask.sum())
    n_total = int(adata.n_obs)

    print(f"Cells matched for exclusion: {n_exclude} / {n_total}")

    if n_exclude == 0:
        print("No cells excluded.")
        return adata

    excluded = adata.obs.loc[exclude_mask].copy()
    excluded_path = Path(out_dir) / "excluded_subtype_cells.csv"
    excluded.to_csv(excluded_path, encoding="utf-8-sig")
    print(f"Saved excluded cells: {excluded_path}")

    adata = adata[~exclude_mask].copy()

    print(f"After subtype exclusion: {adata.n_obs} cells")

    for col in subtype_cols:
        if col in adata.obs.columns:
            print(f"\nTop values in {col} after exclusion:")
            print(adata.obs[col].astype(str).value_counts(dropna=False).head(30))

    return adata


def assert_required_metadata(adata):
    required_cols = [
        AXIS_SCORE_COL,
        FIBROSIS_SCORE_COL,
        CELLTYPE_COL,
    ]

    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Required metadata column not found after merge/filter: {col}"
            )

    valid_axis = pd.to_numeric(
        adata.obs[AXIS_SCORE_COL],
        errors="coerce",
    ).notna().sum()

    print(f"Valid {AXIS_SCORE_COL} cells after filtering: {valid_axis} / {adata.n_obs}")

    if valid_axis < MIN_METADATA_MATCHED_CELLS:
        raise ValueError(
            f"Too few valid {AXIS_SCORE_COL} values after filtering. "
            "metadata merge likely failed."
        )


# ============================================================
# Scanpy preprocessing
# ============================================================

def preprocess_scanpy(adata):
    import scanpy as sc

    print("\n========== Scanpy preprocessing ==========")

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 保存全基因 log-normalized 表达，方便 marker gene UMAP。
    adata.raw = adata.copy()

    n_top = min(N_TOP_GENES, adata.n_vars)

    if n_top >= 100:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top,
            flavor="seurat",
        )

        adata = adata[:, adata.var["highly_variable"]].copy()

        print(f"After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    else:
        print("Too few genes for HVG selection; skipping HVG.")

    sc.pp.scale(adata, max_value=10)

    n_comps = min(
        N_PCS,
        adata.n_obs - 1,
        adata.n_vars - 1,
    )

    if n_comps < 2:
        raise ValueError("Too few cells or genes after filtering for PCA.")

    sc.tl.pca(adata, n_comps=n_comps)

    sc.pp.neighbors(
        adata,
        n_neighbors=min(N_NEIGHBORS, adata.n_obs - 1),
        n_pcs=min(N_PCS, adata.obsm["X_pca"].shape[1]),
    )

    sc.tl.umap(adata)

    return adata


# ============================================================
# root 选择 + diffusion pseudotime
# ============================================================

def choose_root_cell(adata):
    if CELLTYPE_COL in adata.obs.columns:
        celltypes = set(adata.obs[CELLTYPE_COL].astype(str).unique())
        if celltypes != {TARGET_CELLTYPE}:
            raise ValueError(
                f"Root selection stopped: adata contains non-target cell types: {celltypes}. "
                f"Expected only {TARGET_CELLTYPE}."
            )

    if ROOT_MODE == "lowest_total_counts":
        root_idx = int(np.argmin(adata.obs["total_counts_raw"].values))

    elif ROOT_MODE == "lowest_axis_v2":
        if AXIS_SCORE_COL not in adata.obs.columns:
            raise ValueError(
                f"{AXIS_SCORE_COL} not found in adata.obs. "
                "Please merge axis metadata first."
            )

        score = pd.to_numeric(
            adata.obs[AXIS_SCORE_COL],
            errors="coerce",
        )

        if score.notna().sum() == 0:
            raise ValueError(f"No valid numeric values found in {AXIS_SCORE_COL}.")

        cutoff = score.quantile(ROOT_LOW_QUANTILE)
        candidate = score <= cutoff

        if candidate.sum() == 0:
            root_idx = int(np.nanargmin(score.values))
        else:
            candidate_indices = np.where(candidate.values)[0]

            if "X_diffmap" in adata.obsm:
                diff1 = adata.obsm["X_diffmap"][:, 1]
                root_idx = int(
                    candidate_indices[
                        np.argmin(diff1[candidate_indices])
                    ]
                )
            else:
                root_idx = int(candidate_indices[0])

    elif ROOT_MODE == "lowest_fibrosis":
        if FIBROSIS_SCORE_COL not in adata.obs.columns:
            raise ValueError(
                f"{FIBROSIS_SCORE_COL} not found in adata.obs. "
                "Please merge fibrosis metadata first."
            )

        score = pd.to_numeric(
            adata.obs[FIBROSIS_SCORE_COL],
            errors="coerce",
        )

        if score.notna().sum() == 0:
            raise ValueError(f"No valid numeric values found in {FIBROSIS_SCORE_COL}.")

        cutoff = score.quantile(ROOT_LOW_QUANTILE)
        candidate = score <= cutoff

        if candidate.sum() == 0:
            root_idx = int(np.nanargmin(score.values))
        else:
            candidate_indices = np.where(candidate.values)[0]

            if "X_diffmap" in adata.obsm:
                diff1 = adata.obsm["X_diffmap"][:, 1]
                root_idx = int(
                    candidate_indices[
                        np.argmin(diff1[candidate_indices])
                    ]
                )
            else:
                root_idx = int(candidate_indices[0])

    elif ROOT_MODE == "first_cell":
        root_idx = 0

    else:
        raise ValueError(f"Unknown ROOT_MODE: {ROOT_MODE}")

    return root_idx


def run_pseudotime(adata):
    import scanpy as sc

    print("\n========== Diffusion pseudotime ==========")

    sc.tl.diffmap(adata)

    root_idx = choose_root_cell(adata)

    adata.uns["iroot"] = root_idx

    root_cell = adata.obs_names[root_idx]

    print(f"Root mode: {ROOT_MODE}")
    print(f"Root cell: {root_cell}")

    if AXIS_SCORE_COL in adata.obs.columns:
        print(f"Root {AXIS_SCORE_COL}: {adata.obs.iloc[root_idx][AXIS_SCORE_COL]}")

    if FIBROSIS_SCORE_COL in adata.obs.columns:
        print(f"Root {FIBROSIS_SCORE_COL}: {adata.obs.iloc[root_idx][FIBROSIS_SCORE_COL]}")

    if CELLTYPE_COL in adata.obs.columns:
        print(f"Root {CELLTYPE_COL}: {adata.obs.iloc[root_idx][CELLTYPE_COL]}")

    subtype_cols = get_subtype_columns(adata.obs)
    for col in subtype_cols:
        print(f"Root {col}: {adata.obs.iloc[root_idx][col]}")

    sc.tl.dpt(adata)

    adata.obs["dpt_pseudotime"] = adata.obs["dpt_pseudotime"].astype(float)

    root_report = pd.DataFrame({
        "root_mode": [ROOT_MODE],
        "root_cell": [root_cell],
        "root_index": [root_idx],
        "root_low_quantile": [ROOT_LOW_QUANTILE],
    })

    for col in [
        AXIS_SCORE_COL,
        FIBROSIS_SCORE_COL,
        "sample_id",
        "original_barcode",
        CELLTYPE_COL,
        "Disease_Identity",
        "axis_v2_stage",
        "multicellular_axis_stage",
    ] + subtype_cols:
        if col in adata.obs.columns:
            root_report[col] = [adata.obs.iloc[root_idx][col]]

    root_report.to_csv(
        Path(OUT_DIR) / "root_cell_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return adata


# ============================================================
# 画图
# ============================================================

def try_plot_basic(adata, out_dir):
    import scanpy as sc
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    color_list = [
        "dpt_pseudotime",
        "total_counts_raw",
        "spliced_counts_raw",
        "unspliced_counts_raw",
    ]

    for col in [
        AXIS_SCORE_COL,
        FIBROSIS_SCORE_COL,
        "axis_v2_stage",
        "Disease_Identity",
        CELLTYPE_COL,
    ] + get_subtype_columns(adata.obs):
        if col in adata.obs.columns and col not in color_list:
            color_list.append(col)

    try:
        sc.pl.umap(
            adata,
            color=color_list,
            show=False,
            wspace=0.4,
        )

        out_png = fig_dir / "umap_basic_qc_pseudotime.png"

        plt.savefig(
            out_png,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved figure: {out_png}")

    except Exception as e:
        print(f"Plotting skipped due to error: {e}")


# ============================================================
# Extra diagnostics for pseudotime interpretation
# ============================================================

def safe_spearman(x, y):
    from scipy.stats import spearmanr

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = x.notna() & y.notna()

    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())

    rho, pval = spearmanr(x[mask], y[mask])

    return float(rho), float(pval), int(mask.sum())


def save_pseudotime_diagnostics(adata, out_dir):
    """
    保存 pseudotime 与 axis/QC/disease/sample/subtype 的关系表。
    用于判断 pseudotime 是否主要被 QC 或某个 sample/subtype 驱动。
    """

    if not RUN_EXTRA_DIAGNOSTICS:
        print("\nExtra diagnostics skipped: RUN_EXTRA_DIAGNOSTICS = False")
        return

    print("\n========== Extra pseudotime diagnostics ==========")

    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    obs = adata.obs.copy()

    if "dpt_pseudotime" not in obs.columns:
        print("No dpt_pseudotime found. Skipping diagnostics.")
        return

    # --------------------------------------------------------
    # 1. Spearman correlations
    # --------------------------------------------------------
    corr_cols = [
        AXIS_SCORE_COL,
        FIBROSIS_SCORE_COL,
        "total_counts_raw",
        "spliced_counts_raw",
        "unspliced_counts_raw",
    ]

    corr_rows = []

    for col in corr_cols:
        if col in obs.columns:
            rho, pval, n = safe_spearman(obs["dpt_pseudotime"], obs[col])
            corr_rows.append({
                "x": "dpt_pseudotime",
                "y": col,
                "spearman_rho": rho,
                "p_value": pval,
                "n_cells": n,
            })

    corr_df = pd.DataFrame(corr_rows)

    corr_path = diag_dir / "pseudotime_spearman_correlations.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {corr_path}")
    print(corr_df)

    # --------------------------------------------------------
    # 2. Group summaries
    # --------------------------------------------------------
    group_cols = [
        "Disease_Identity",
        "axis_v2_stage",
        "sample_id",
        "Subject_Identity",
    ] + get_subtype_columns(obs)

    # 去重保序
    group_cols = list(dict.fromkeys(group_cols))

    for group_col in group_cols:
        if group_col not in obs.columns:
            continue

        tmp = obs.copy()
        tmp["dpt_pseudotime"] = safe_numeric(tmp["dpt_pseudotime"])

        if AXIS_SCORE_COL in tmp.columns:
            tmp[AXIS_SCORE_COL] = safe_numeric(tmp[AXIS_SCORE_COL])
        else:
            tmp[AXIS_SCORE_COL] = np.nan

        if FIBROSIS_SCORE_COL in tmp.columns:
            tmp[FIBROSIS_SCORE_COL] = safe_numeric(tmp[FIBROSIS_SCORE_COL])
        else:
            tmp[FIBROSIS_SCORE_COL] = np.nan

        summary = (
            tmp
            .groupby(group_col, dropna=False)
            .agg(
                n_cells=("dpt_pseudotime", "size"),
                pseudotime_mean=("dpt_pseudotime", "mean"),
                pseudotime_median=("dpt_pseudotime", "median"),
                pseudotime_min=("dpt_pseudotime", "min"),
                pseudotime_max=("dpt_pseudotime", "max"),
                axis_v2_mean=(AXIS_SCORE_COL, "mean"),
                axis_v2_median=(AXIS_SCORE_COL, "median"),
                fibrosis_mean=(FIBROSIS_SCORE_COL, "mean"),
                fibrosis_median=(FIBROSIS_SCORE_COL, "median"),
            )
            .reset_index()
            .sort_values(
                ["pseudotime_median", "n_cells"],
                ascending=[False, False],
            )
        )

        out_path = diag_dir / f"pseudotime_summary_by_{group_col}.csv"
        summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")

    # --------------------------------------------------------
    # 3. Terminal pseudotime region composition
    # --------------------------------------------------------
    pt = safe_numeric(obs["dpt_pseudotime"])
    cutoff = pt.quantile(TERMINAL_PSEUDOTIME_QUANTILE)

    terminal = obs.loc[pt >= cutoff].copy()
    terminal["terminal_pseudotime_cutoff"] = cutoff

    terminal_path = diag_dir / "terminal_high_pseudotime_cells.csv"
    terminal.to_csv(terminal_path, encoding="utf-8-sig")
    print(f"Saved: {terminal_path}")
    print(f"Terminal high-pseudotime cells: {terminal.shape[0]} / {obs.shape[0]}")
    print(f"Terminal pseudotime cutoff: {cutoff}")

    for group_col in group_cols:
        if group_col not in terminal.columns:
            continue

        comp = (
            terminal[group_col]
            .astype(str)
            .value_counts(dropna=False)
            .reset_index()
        )
        comp.columns = [group_col, "n_terminal_cells"]
        comp["fraction_of_terminal"] = (
            comp["n_terminal_cells"] / comp["n_terminal_cells"].sum()
        )

        comp_path = diag_dir / f"terminal_composition_by_{group_col}.csv"
        comp.to_csv(comp_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {comp_path}")

def save_terminal_enrichment(
    adata,
    out_dir,
    group_col,
    pseudotime_col="dpt_pseudotime",
    q=0.95,
):
    """
    Compare terminal high-pseudotime composition with overall composition.

    This helps determine whether a group is truly enriched in the
    terminal pseudotime region, rather than simply being abundant overall.
    """

    obs = adata.obs.copy()

    if pseudotime_col not in obs.columns:
        print(f"[Warning] {pseudotime_col} not found. Skip terminal enrichment.")
        return None

    if group_col not in obs.columns:
        print(f"[Warning] {group_col} not found. Skip terminal enrichment.")
        return None

    pt = pd.to_numeric(obs[pseudotime_col], errors="coerce")

    if pt.notna().sum() < 10:
        print(f"[Warning] Too few valid {pseudotime_col} values. Skip terminal enrichment.")
        return None

    cutoff = pt.quantile(q)

    terminal = obs.loc[pt >= cutoff].copy()

    if terminal.shape[0] == 0:
        print("[Warning] No terminal cells found. Skip terminal enrichment.")
        return None

    overall_counts = (
        obs[group_col]
        .astype(str)
        .value_counts(dropna=False)
    )

    terminal_counts = (
        terminal[group_col]
        .astype(str)
        .value_counts(dropna=False)
    )

    df = pd.DataFrame({
        "overall_cells": overall_counts,
        "terminal_cells": terminal_counts,
    }).fillna(0)

    df["overall_cells"] = df["overall_cells"].astype(int)
    df["terminal_cells"] = df["terminal_cells"].astype(int)

    df["overall_fraction"] = (
        df["overall_cells"] / df["overall_cells"].sum()
    )

    df["terminal_fraction"] = (
        df["terminal_cells"] / df["terminal_cells"].sum()
    )

    df["fold_enrichment"] = (
        df["terminal_fraction"] / df["overall_fraction"]
    )

    df["terminal_minus_overall_fraction"] = (
        df["terminal_fraction"] - df["overall_fraction"]
    )

    df = (
        df
        .reset_index()
        .rename(columns={"index": group_col})
        .sort_values("fold_enrichment", ascending=False)
    )

    out_path = (
        Path(out_dir)
        / "diagnostics"
        / f"terminal_enrichment_by_{group_col}.csv"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved terminal enrichment: {out_path}")
    print(df.head(20))

    return df

def resolve_marker_genes(adata, marker_genes):
    """
    更稳的 marker gene 查找：
    1. 先在 adata.raw.var_names 或 adata.var_names 中大小写不敏感匹配；
    2. 如果 var 中存在 gene_name / gene_symbol / symbol 等列，也尝试匹配；
    3. 返回 Scanpy 可用于 sc.pl.umap(color=...) 的实际基因名。
    """

    if adata.raw is not None:
        var_names = list(adata.raw.var_names.astype(str))
        var_df = adata.raw.var.copy()
    else:
        var_names = list(adata.var_names.astype(str))
        var_df = adata.var.copy()

    upper_to_varname = {g.upper(): g for g in var_names}

    resolved = []
    missing = []

    marker_upper = [g.upper() for g in marker_genes]

    for gene, gene_upper in zip(marker_genes, marker_upper):
        if gene_upper in upper_to_varname:
            resolved.append(upper_to_varname[gene_upper])
        else:
            missing.append(gene)

    # 如果 var_names 找不到，再尝试 var 中的 gene symbol 列
    candidate_symbol_cols = [
        "gene_name",
        "gene_symbol",
        "symbol",
        "Gene",
        "gene",
        "features",
        "feature_name",
    ]

    symbol_cols = [c for c in candidate_symbol_cols if c in var_df.columns]

    if len(symbol_cols) > 0 and len(missing) > 0:
        for col in symbol_cols:
            symbol_series = var_df[col].astype(str)
            symbol_upper_to_index = {}

            for idx, val in symbol_series.items():
                val_upper = val.upper()
                if val_upper not in symbol_upper_to_index:
                    symbol_upper_to_index[val_upper] = idx

            still_missing = []

            for gene in missing:
                gene_upper = gene.upper()
                if gene_upper in symbol_upper_to_index:
                    resolved.append(symbol_upper_to_index[gene_upper])
                else:
                    still_missing.append(gene)

            missing = still_missing

    # 去重保序
    resolved = list(dict.fromkeys(resolved))

    return resolved, missing


def try_plot_diagnostics(adata, out_dir):
    """
    额外画 UMAP，用于判断 pseudotime 分支是否由 sample/QC/marker/subtype 驱动。
    """

    if not RUN_EXTRA_DIAGNOSTICS:
        return

    import scanpy as sc
    import matplotlib.pyplot as plt

    print("\n========== Plot extra diagnostics ==========")

    fig_dir = Path(out_dir) / "figures" / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. metadata / QC UMAP
    # --------------------------------------------------------
    color_cols = []

    for col in DIAGNOSTIC_COLOR_COLS + get_subtype_columns(adata.obs):
        if col in adata.obs.columns and col not in color_cols:
            color_cols.append(col)

    if len(color_cols) > 0:
        try:
            sc.pl.umap(
                adata,
                color=color_cols,
                show=False,
                wspace=0.4,
            )

            out_png = fig_dir / "umap_metadata_qc_diagnostics.png"
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved figure: {out_png}")

        except Exception as e:
            print(f"Metadata/QC diagnostic plotting skipped: {e}")

    # --------------------------------------------------------
    # 2. Marker gene UMAP
    # --------------------------------------------------------
    available_genes, missing_genes = resolve_marker_genes(adata, MARKER_GENES)

    marker_report = pd.DataFrame({
        "marker_gene": MARKER_GENES,
        "found": [g not in missing_genes for g in MARKER_GENES],
    })
    marker_report_path = Path(out_dir) / "diagnostics" / "marker_gene_resolution_report.csv"
    marker_report.to_csv(marker_report_path, index=False, encoding="utf-8-sig")
    print(f"Saved marker gene resolution report: {marker_report_path}")

    if len(available_genes) > 0:
        try:
            print("Available marker genes:")
            print(available_genes)
            print("Missing marker genes:")
            print(missing_genes)

            sc.pl.umap(
                adata,
                color=available_genes,
                show=False,
                wspace=0.4,
                use_raw=True if adata.raw is not None else False,
            )

            out_png = fig_dir / "umap_marker_genes.png"
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved figure: {out_png}")

        except Exception as e:
            print(f"Marker gene diagnostic plotting skipped: {e}")
    else:
        print("No marker genes found after case-insensitive and symbol-column matching.")


# ============================================================
# CellRank 兼容函数
# ============================================================

def cellrank_compute_terminal_states_compat(g):
    if hasattr(g, "compute_terminal_states"):
        return g.compute_terminal_states()

    if hasattr(g, "predict_terminal_states"):
        return g.predict_terminal_states()

    print(
        "This CellRank GPCCA object has no compute_terminal_states() "
        "or predict_terminal_states(). Skipping terminal state step."
    )

    return None


def cellrank_compute_fate_probabilities_compat(g):
    if hasattr(g, "compute_fate_probabilities"):
        return g.compute_fate_probabilities()

    print(
        "This CellRank GPCCA object has no compute_fate_probabilities(). "
        "Skipping fate probabilities."
    )

    return None


def infer_cellrank_state_names(g, n_states):
    """
    尝试从 CellRank estimator 中获取 state 名称。
    如果获取失败，则返回 state_0, state_1, ...
    """
    colnames = None

    for attr in ["terminal_states", "macrostates"]:
        try:
            states = getattr(g, attr, None)

            if states is None:
                continue

            if hasattr(states, "cat"):
                colnames = list(states.cat.categories.astype(str))
            elif hasattr(states, "names"):
                colnames = list(states.names.astype(str))
            else:
                tmp = pd.Series(states).astype(str)
                vals = [x for x in tmp.unique() if x.lower() != "nan"]
                if len(vals) == n_states:
                    colnames = vals

            if colnames is not None and len(colnames) == n_states:
                break

        except Exception:
            pass

    if colnames is None or len(colnames) != n_states:
        colnames = [f"state_{i}" for i in range(n_states)]

    return colnames


def save_fate_probabilities_if_present(g, adata, out_path):
    try:
        fp = getattr(g, "fate_probabilities", None)

        if fp is None:
            print("No fate_probabilities attribute found.")
            return False, None

        arr = np.asarray(fp)

        colnames = infer_cellrank_state_names(g, arr.shape[1])

        df = pd.DataFrame(
            arr,
            index=adata.obs_names,
            columns=colnames,
        )

        df.index.name = "cell_id"

        df.to_csv(
            out_path,
            encoding="utf-8-sig",
        )

        print(f"Saved fate probabilities: {out_path}")
        print("Fate probability columns:")
        print(colnames)

        return True, df

    except Exception as e:
        print(f"Could not save fate probabilities: {e}")
        return False, None


def get_cellrank_meta_cols(adata):
    meta_cols = [
        "dpt_pseudotime",
        AXIS_SCORE_COL,
        FIBROSIS_SCORE_COL,
        "Disease_Identity",
        "sample_id",
        "Subject_Identity",
        "axis_v2_stage",
        CELLTYPE_COL,
    ] + get_subtype_columns(adata.obs)

    meta_cols = [c for c in meta_cols if c in adata.obs.columns]
    meta_cols = list(dict.fromkeys(meta_cols))

    return meta_cols


def save_cellrank_assignments(fp_df, adata, out_path):
    if fp_df is None:
        return False

    try:
        assignments = pd.DataFrame(index=fp_df.index)
        assignments.index.name = "cell_id"

        assignments["cellrank_state"] = fp_df.idxmax(axis=1)
        assignments["cellrank_max_probability"] = fp_df.max(axis=1)

        meta_cols = get_cellrank_meta_cols(adata)

        out_df = assignments.join(adata.obs[meta_cols], how="left")

        out_df.to_csv(out_path, encoding="utf-8-sig")

        print(f"Saved CellRank assignments: {out_path}")

        return True

    except Exception as e:
        print(f"Could not save CellRank assignments: {e}")
        return False


def save_top_fate_cells(fp_df, adata, out_path, top_n=TOP_FATE_CELLS_N):
    if fp_df is None:
        return False

    try:
        meta_cols = get_cellrank_meta_cols(adata)

        rows = []

        for state in fp_df.columns:
            top_cells = fp_df[state].sort_values(ascending=False).head(top_n)

            tmp = adata.obs.loc[top_cells.index, meta_cols].copy()
            tmp.insert(0, "cell_id", top_cells.index)
            tmp.insert(1, "state", state)
            tmp.insert(2, "fate_probability", top_cells.values)

            rows.append(tmp)

        out_df = pd.concat(rows, axis=0)

        out_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Saved top fate cells: {out_path}")

        return True

    except Exception as e:
        print(f"Could not save top fate cells: {e}")
        return False



def make_cellrank_subset(adata, max_cells=5000, seed=123, stratify_cols=None):
    """
    为 CellRank GPCCA 构建子集，避免在十万级细胞上 dense transition matrix。

    策略：
    - 如果细胞数 <= max_cells，直接复制；
    - 否则按 pseudotime 分箱，并结合 disease/stage/subtype 等列做分层抽样；
    - 保留早期、中期、末端 pseudotime 的代表性。
    """
    if adata.n_obs <= max_cells:
        print(f"CellRank subset not needed: {adata.n_obs} <= {max_cells}")
        return adata.copy()

    rng = np.random.default_rng(seed)
    obs = adata.obs.copy()

    if "dpt_pseudotime" in obs.columns:
        pt = pd.to_numeric(obs["dpt_pseudotime"], errors="coerce")

        if pt.notna().sum() >= 10:
            obs["_cellrank_pseudotime_bin"] = pd.qcut(
                pt.rank(method="first"),
                q=10,
                labels=False,
                duplicates="drop",
            ).astype(str)
        else:
            obs["_cellrank_pseudotime_bin"] = "all"
    else:
        obs["_cellrank_pseudotime_bin"] = "all"

    if stratify_cols is None:
        stratify_cols = []

    use_cols = [c for c in stratify_cols if c in obs.columns]
    use_cols.append("_cellrank_pseudotime_bin")

    obs["_cellrank_stratum"] = obs[use_cols].astype(str).agg("|".join, axis=1)

    selected = []
    total = obs.shape[0]

    for stratum, sub in obs.groupby("_cellrank_stratum", observed=False):
        idx = sub.index.to_numpy()
        n_pick = max(1, int(round(max_cells * len(idx) / total)))
        n_pick = min(n_pick, len(idx))

        picked = rng.choice(idx, size=n_pick, replace=False)
        selected.extend(picked.tolist())

    if len(selected) > max_cells:
        selected = rng.choice(
            np.array(selected),
            size=max_cells,
            replace=False,
        ).tolist()

    print(f"CellRank subset: {len(selected)} / {adata.n_obs} cells")

    return adata[selected, :].copy()


# ============================================================
# CellRank 主线
# ============================================================

def run_cellrank_optional(adata, out_dir):
    if not RUN_CELLRANK:
        print("\nCellRank skipped: RUN_CELLRANK = False")
        return adata, []

    print("\n========== Optional CellRank test ==========")

    try:
        from cellrank.kernels import ConnectivityKernel, PseudotimeKernel
        from cellrank.estimators import GPCCA
    except Exception as e:
        print("CellRank is not installed or cannot be imported.")
        print(f"Error: {e}")
        print("Install with: pip install cellrank")

        return adata, []

    if CELLRANK_USE_SUBSAMPLE:
        adata = make_cellrank_subset(
            adata,
            max_cells=CELLRANK_MAX_CELLS,
            seed=CELLRANK_RANDOM_SEED,
            stratify_cols=CELLRANK_STRATIFY_COLS,
        )

        # 子集上重新计算邻居图，避免沿用全量对象的图结构。
        try:
            import scanpy as sc

            n_comps = min(N_PCS, adata.n_obs - 1, adata.n_vars - 1)

            if n_comps >= 2:
                sc.tl.pca(adata, n_comps=n_comps)

                sc.pp.neighbors(
                    adata,
                    n_neighbors=min(N_NEIGHBORS, adata.n_obs - 1),
                    n_pcs=min(N_PCS, adata.obsm["X_pca"].shape[1]),
                )
            else:
                print("Warning: too few cells/genes to recompute CellRank subset PCA.")

        except Exception as e:
            print(f"Warning: could not recompute neighbors for CellRank subset: {e}")

    saved_outputs = []

    n_states = min(3, max(2, adata.n_obs // 500))

    print(f"CellRank input shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Using n_states={n_states}")

    # --------------------------------------------------------
    # 1. ConnectivityKernel
    # --------------------------------------------------------
    try:
        print("Running ConnectivityKernel...")

        ck = ConnectivityKernel(adata)
        ck.compute_transition_matrix()

        print("Running GPCCA on ConnectivityKernel...")

        g = GPCCA(ck)

        if hasattr(g, "compute_macrostates"):
            try:
                g.compute_macrostates(
                    n_states=n_states,
                    cluster_key=None,
                )
            except TypeError:
                g.compute_macrostates(
                    n_states=n_states,
                )
        else:
            print("GPCCA has no compute_macrostates().")

        try:
            cellrank_compute_terminal_states_compat(g)
            cellrank_compute_fate_probabilities_compat(g)

            out_path = Path(out_dir) / "cellrank_connectivity_fate_probabilities.csv"

            ok, fp_df = save_fate_probabilities_if_present(g, adata, out_path)

            if ok:
                saved_outputs.append(out_path)

                if SAVE_CELLRANK_ASSIGNMENTS:
                    assign_path = Path(out_dir) / "cellrank_connectivity_assignments.csv"
                    if save_cellrank_assignments(fp_df, adata, assign_path):
                        saved_outputs.append(assign_path)

                if SAVE_CELLRANK_TOP_CELLS:
                    top_path = Path(out_dir) / "cellrank_connectivity_top_fate_cells.csv"
                    if save_top_fate_cells(fp_df, adata, top_path):
                        saved_outputs.append(top_path)

        except Exception as e:
            print(f"ConnectivityKernel terminal/fate step skipped: {e}")

        print("ConnectivityKernel CellRank test finished or partially finished.")

    except Exception as e:
        print("ConnectivityKernel CellRank test did not complete.")
        print(f"Error: {e}")

    # --------------------------------------------------------
    # 2. PseudotimeKernel
    # --------------------------------------------------------
    try:
        print("Running PseudotimeKernel...")

        pk = PseudotimeKernel(
            adata,
            time_key="dpt_pseudotime",
        )

        pk.compute_transition_matrix()

        print("Running GPCCA on PseudotimeKernel...")

        g2 = GPCCA(pk)

        if hasattr(g2, "compute_macrostates"):
            try:
                g2.compute_macrostates(
                    n_states=n_states,
                    cluster_key=None,
                )
            except TypeError:
                g2.compute_macrostates(
                    n_states=n_states,
                )
        else:
            print("GPCCA has no compute_macrostates().")

        try:
            cellrank_compute_terminal_states_compat(g2)
            cellrank_compute_fate_probabilities_compat(g2)

            out_path = Path(out_dir) / "cellrank_pseudotime_fate_probabilities.csv"

            ok, fp_df = save_fate_probabilities_if_present(g2, adata, out_path)

            if ok:
                saved_outputs.append(out_path)

                if SAVE_CELLRANK_ASSIGNMENTS:
                    assign_path = Path(out_dir) / "cellrank_pseudotime_assignments.csv"
                    if save_cellrank_assignments(fp_df, adata, assign_path):
                        saved_outputs.append(assign_path)

                if SAVE_CELLRANK_TOP_CELLS:
                    top_path = Path(out_dir) / "cellrank_pseudotime_top_fate_cells.csv"
                    if save_top_fate_cells(fp_df, adata, top_path):
                        saved_outputs.append(top_path)

        except Exception as e:
            print(f"PseudotimeKernel terminal/fate step skipped: {e}")

        print("PseudotimeKernel CellRank test finished or partially finished.")

    except Exception as e:
        print("PseudotimeKernel CellRank test did not complete.")
        print(f"Error: {e}")

    return adata, saved_outputs


# ============================================================
# scVelo 占位
# ============================================================

def run_scvelo_optional(adata_raw, out_dir):
    if not RUN_SCVELO_VELOCITY:
        print("\nscVelo skipped: RUN_SCVELO_VELOCITY = False")
        return None

    print(
        "This formal main script does not run scVelo. "
        "Please use a separate exploratory scVelo script."
    )

    return None


# ============================================================
# h5ad 保存
# ============================================================

def safe_write_h5ad(adata, path):
    try:
        adata = sanitize_anndata_for_h5ad(adata)
        adata.write_h5ad(path)
        print(f"Saved h5ad: {path}")

    except Exception as e:
        print(f"h5ad saving skipped due to error: {e}")
        print("Core CSV outputs are still usable.")


# ============================================================
# main
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    print("========== CellRank / axis_v2 main workflow ==========")
    print(f"EXPORT_DIR: {EXPORT_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"ROOT_MODE: {ROOT_MODE}")
    print(f"AXIS_METADATA_CSV: {AXIS_METADATA_CSV}")
    print(f"TARGET_CELLTYPE: {TARGET_CELLTYPE}")
    print(f"ALLOW_BARCODE_ONLY_MERGE: {ALLOW_BARCODE_ONLY_MERGE}")
    print(f"EXCLUDE_SUBTYPES: {EXCLUDE_SUBTYPES}")
    print(f"EXCLUDE_CONTAINS_TERMS: {EXCLUDE_CONTAINS_TERMS}")
    print(f"EXCLUDE_EXACT_TERMS: {EXCLUDE_EXACT_TERMS}")
    print(f"RUN_CELLRANK: {RUN_CELLRANK}")
    print(f"CELLRANK_USE_SUBSAMPLE: {CELLRANK_USE_SUBSAMPLE}")
    print(f"CELLRANK_MAX_CELLS: {CELLRANK_MAX_CELLS}")

    X, spliced, unspliced, genes, cells = read_exported_mtx(EXPORT_DIR)

    adata_raw = build_anndata(
        X,
        spliced,
        unspliced,
        genes,
        cells,
    )

    adata_raw = apply_gene_symbol_mapping(adata_raw)

    save_gene_name_check(adata_raw, OUT_DIR)

    report_marker_gene_availability(
        adata_raw,
        MARKER_GENES,
        OUT_DIR,
    )

    build_report = pd.DataFrame({
        "n_cells": [adata_raw.n_obs],
        "n_genes": [adata_raw.n_vars],
        "total_counts_sum": [
            float(adata_raw.obs["total_counts_raw"].sum())
        ],
        "spliced_counts_sum": [
            float(adata_raw.obs["spliced_counts_raw"].sum())
        ],
        "unspliced_counts_sum": [
            float(adata_raw.obs["unspliced_counts_raw"].sum())
        ],
    })

    build_report.to_csv(
        Path(OUT_DIR) / "anndata_build_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== AnnData build report ==========")
    print(build_report)

    if ROOT_MODE in [
        "lowest_axis_v2",
        "lowest_fibrosis",
    ]:
        adata_raw = merge_axis_metadata(adata_raw)

    adata_raw = filter_target_celltype(adata_raw)

    # 针对 Endothelial：可排除 Lymphatic-Endothelial 后重跑 vascular endothelial trajectory。
    adata_raw = exclude_unwanted_subtypes(adata_raw, OUT_DIR)

    assert_required_metadata(adata_raw)

    adata = adata_raw.copy()

    adata = preprocess_scanpy(adata)

    adata = run_pseudotime(adata)

    obs_out = Path(OUT_DIR) / "adata_obs_after_pseudotime.csv"

    adata.obs.to_csv(
        obs_out,
        encoding="utf-8-sig",
    )

    print(f"Saved obs after pseudotime: {obs_out}")

    try_plot_basic(adata, OUT_DIR)

    save_pseudotime_diagnostics(adata, OUT_DIR)

    for col in [
        "Disease_Identity",
        "axis_v2_stage",
        "sample_id",
        "Subject_Identity",
        "Subclass_Cell_Identity",
        "Manuscript_Identity",
    ]:
        if col in adata.obs.columns:
            save_terminal_enrichment(
                adata,
                OUT_DIR,
                group_col=col,
                q=TERMINAL_PSEUDOTIME_QUANTILE,
            )

    try_plot_diagnostics(adata, OUT_DIR)

    adata, cellrank_outputs = run_cellrank_optional(adata, OUT_DIR)

    _ = run_scvelo_optional(adata_raw, OUT_DIR)

    if SAVE_H5AD:
        safe_write_h5ad(
            adata,
            Path(OUT_DIR) / "processed_pseudotime_cellrank.h5ad",
        )
    else:
        print("\nh5ad skipped: SAVE_H5AD = False")

    print("\n========== Finished ==========")

    print("Core outputs:")
    print(f"  {Path(OUT_DIR) / 'anndata_build_report.csv'}")
    print(f"  {Path(OUT_DIR) / 'adata_obs_after_pseudotime.csv'}")
    print(f"  {Path(OUT_DIR) / 'root_cell_report.csv'}")
    print(f"  {Path(OUT_DIR) / 'gene_name_check.csv'}")
    print(f"  {Path(OUT_DIR) / 'figures'}")
    print(f"  {Path(OUT_DIR) / 'diagnostics'}")

    print("CellRank outputs:")

    if cellrank_outputs:
        for p in cellrank_outputs:
            print(f"  {p}")
    else:
        print("  No CellRank output was generated.")


if __name__ == "__main__":
    main()