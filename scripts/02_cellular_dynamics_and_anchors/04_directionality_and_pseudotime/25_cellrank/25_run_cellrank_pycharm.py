#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cellrank_axis_v2_main.py

正式主线版 CellRank / pseudotime 脚本

功能：
  1. 读取 R 导出的：
       total_counts.mtx
       spliced_exon.mtx
       unspliced_intron.mtx
       genes.csv
       cells.csv

  2. 构建 AnnData

  3. 合并 axis_v2 metadata

  4. 使用 axis_v2_score 最低区域作为 root

  5. 跑：
       Scanpy preprocessing
       PCA / neighbors / UMAP
       diffusion pseudotime
       CellRank ConnectivityKernel
       CellRank PseudotimeKernel

  6. 输出：
       anndata_build_report.csv
       adata_obs_after_pseudotime.csv
       root_cell_report.csv
       cellrank_connectivity_fate_probabilities.csv
       cellrank_pseudotime_fate_probabilities.csv
       figures/umap_basic_qc_pseudotime.png

注意：
  本脚本是正式主线版，默认：
    RUN_SCVELO_VELOCITY = False
    SAVE_H5AD = False

  scVelo / VelocityKernel 建议后续作为探索性补充单独跑。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG：每个 cell type 正式运行时主要修改这里
# ============================================================

# R 导出的目录，里面必须有：
# total_counts.mtx
# spliced_exon.mtx
# unspliced_intron.mtx
# genes.csv
# cells.csv
EXPORT_DIR = r"E:\package\result\Rresults\Endothelial"

# Python 输出结果目录
OUT_DIR = r"E:\package\result\Pythonresult\Endothelial"

# axis_v2 metadata 文件
AXIS_METADATA_CSV = r"E:\package\metadata\03_cell_metadata_with_axis_v2_module_scores.csv"

# 正式运行推荐使用 lowest_axis_v2
ROOT_MODE = "lowest_axis_v2"

# 可选：
# ROOT_MODE = "lowest_axis_v2"
# ROOT_MODE = "lowest_fibrosis"
# ROOT_MODE = "lowest_total_counts"
# ROOT_MODE = "first_cell"

# axis metadata 里的列名
AXIS_CELL_ID_COL = "cell_id"
AXIS_SAMPLE_COL = "sample_id"

# 如果你的 metadata 里 barcode 列叫 CellBarcode_Identity，就保持这样
AXIS_BARCODE_COL = "CellBarcode_Identity"

# 用于正式 root 的分数列
AXIS_SCORE_COL = "axis_v2_score"
FIBROSIS_SCORE_COL = "fibrosis_composite_score"

# 使用 axis_v2_score 最低多少比例的细胞作为 root 候选区域
# 0.02 = 最低 2%
ROOT_LOW_QUANTILE = 0.02

# Scanpy / CellRank 参数
RUN_CELLRANK = True

RUN_SCVELO_VELOCITY = False

SAVE_H5AD = False

N_PCS = 30

N_NEIGHBORS = 15

N_TOP_GENES = 2000


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_index_unique(values):
    """
    避免 gene name 重复。
    """
    seen = {}
    out = []

    for v in values:
        v = str(v)

        if v not in seen:
            seen[v] = 0
            out.append(v)
        else:
            seen[v] += 1
            out.append(f"{v}-{seen[v]}")

    return pd.Index(out)


def sanitize_dataframe_for_h5ad(df):
    """
    如果未来需要保存 h5ad，可减少 object/category 报错。
    当前脚本默认 SAVE_H5AD = False。
    """
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

    # R 导出通常是 genes x cells
    # AnnData 需要 cells x genes
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
        obs["cell_id"] = [
            f"cell_{i}" for i in range(obs.shape[0])
        ]

    obs["cell_id"] = obs["cell_id"].astype(str)
    obs.index = pd.Index(obs["cell_id"].values.astype(str))
    obs.index.name = None

    var = genes.copy()

    if "gene_name" in var.columns:
        gene_names = var["gene_name"].astype(str)
    elif "gene_id" in var.columns:
        gene_names = var["gene_id"].astype(str)
    else:
        gene_names = pd.Series(
            [f"gene_{i}" for i in range(var.shape[0])]
        )

    gene_names = make_index_unique(gene_names)

    var.index = gene_names
    var.index.name = None

    obs = sanitize_dataframe_for_h5ad(obs)
    var = sanitize_dataframe_for_h5ad(var)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced.copy()

    adata.obs["total_counts_raw"] = np.asarray(
        adata.X.sum(axis=1)
    ).ravel()

    adata.obs["spliced_counts_raw"] = np.asarray(
        adata.layers["spliced"].sum(axis=1)
    ).ravel()

    adata.obs["unspliced_counts_raw"] = np.asarray(
        adata.layers["unspliced"].sum(axis=1)
    ).ravel()

    return adata


# ============================================================
# 合并 axis_v2 metadata
# ============================================================

def merge_axis_metadata(adata):
    """
    将 axis_v2 / fibrosis metadata 合并到 adata.obs。

    当前 cells.csv 格式示例：
      cell_id = GSM4058905_244C__AAACCTGAGAAGATTC
      sample_id = GSM4058905_244C
      original_barcode = AAACCTGAGAAGATTC

    匹配顺序：
      1. 直接用 cell_id 匹配
      2. 用 sample_id + original_barcode 匹配
      3. 用 original_barcode 和 metadata barcode 后缀匹配

    第 3 种 barcode-only 匹配可能有重复风险，因此只有在前两种失败时才使用。
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

        meta_by_cell.index = meta_by_cell[
            AXIS_CELL_ID_COL
        ].astype(str)

        overlap = obs.index.intersection(meta_by_cell.index)

        print(
            f"Direct cell_id overlap: {len(overlap)} / {adata.n_obs}"
        )

        if len(overlap) > 0:
            meta_sub = meta_by_cell.reindex(obs.index)

            for col in meta_sub.columns:
                if col not in obs.columns:
                    obs[col] = meta_sub[col].values

            adata.obs = obs

            print("Merged axis metadata by direct cell_id.")

            return adata

    # --------------------------------------------------------
    # 2. 用 sample_id + original_barcode 匹配
    # --------------------------------------------------------
    if "sample_id" in obs.columns and "original_barcode" in obs.columns:
        if AXIS_SAMPLE_COL in meta.columns and AXIS_BARCODE_COL in meta.columns:

            obs["_merge_key_sample_barcode"] = (
                obs["sample_id"].astype(str)
                + "__"
                + obs["original_barcode"].astype(str)
            )

            # metadata 里的 barcode 可能是：
            #   AAACCTGAGAAGATTC
            # 或：
            #   001C_AAACCTGAGAAGATTC
            # 所以构造两种 key
            meta["_barcode_raw"] = meta[AXIS_BARCODE_COL].astype(str)
            meta["_barcode_suffix"] = (
                meta["_barcode_raw"].astype(str).str.split("_").str[-1]
            )

            meta["_merge_key_sample_barcode"] = (
                meta[AXIS_SAMPLE_COL].astype(str)
                + "__"
                + meta["_barcode_suffix"].astype(str)
            )

            meta_by_key = meta.drop_duplicates(
                subset=["_merge_key_sample_barcode"]
            ).copy()

            meta_by_key.index = meta_by_key[
                "_merge_key_sample_barcode"
            ].astype(str)

            overlap = pd.Index(
                obs["_merge_key_sample_barcode"]
            ).intersection(meta_by_key.index)

            print(
                f"sample_id + barcode overlap: {len(overlap)} / {adata.n_obs}"
            )

            if len(overlap) > 0:
                meta_sub = meta_by_key.reindex(
                    obs["_merge_key_sample_barcode"].values
                )

                for col in meta_sub.columns:
                    if col not in obs.columns:
                        obs[col] = meta_sub[col].values

                obs.drop(
                    columns=["_merge_key_sample_barcode"],
                    inplace=True,
                    errors="ignore",
                )

                adata.obs = obs

                print("Merged axis metadata by sample_id + barcode.")

                return adata

            obs.drop(
                columns=["_merge_key_sample_barcode"],
                inplace=True,
                errors="ignore",
            )

    # --------------------------------------------------------
    # 3. 最后尝试 barcode-only 匹配
    # --------------------------------------------------------
    if "original_barcode" in obs.columns and AXIS_BARCODE_COL in meta.columns:

        obs["_barcode_key"] = obs["original_barcode"].astype(str)

        meta["_barcode_key"] = (
            meta[AXIS_BARCODE_COL].astype(str).str.split("_").str[-1]
        )

        meta_by_barcode = meta.drop_duplicates(
            subset=["_barcode_key"]
        ).copy()

        meta_by_barcode.index = meta_by_barcode[
            "_barcode_key"
        ].astype(str)

        overlap = pd.Index(
            obs["_barcode_key"]
        ).intersection(meta_by_barcode.index)

        print(
            f"Barcode-only overlap: {len(overlap)} / {adata.n_obs}"
        )

        if len(overlap) > 0:
            print(
                "WARNING: Using barcode-only merge. "
                "This may be unsafe if barcodes are reused across samples."
            )

            meta_sub = meta_by_barcode.reindex(
                obs["_barcode_key"].values
            )

            for col in meta_sub.columns:
                if col not in obs.columns:
                    obs[col] = meta_sub[col].values

            obs.drop(
                columns=["_barcode_key"],
                inplace=True,
                errors="ignore",
            )

            adata.obs = obs

            print("Merged axis metadata by barcode-only.")

            return adata

        obs.drop(
            columns=["_barcode_key"],
            inplace=True,
            errors="ignore",
        )

    raise ValueError(
        "Could not merge axis metadata. "
        "No overlap found by cell_id, sample_id+barcode, or barcode-only. "
        "Please check metadata sample_id/barcode naming."
    )


# ============================================================
# Scanpy preprocessing
# ============================================================

def preprocess_scanpy(adata):
    import scanpy as sc

    print("\n========== Scanpy preprocessing ==========")

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=3)

    print(
        f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes"
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(N_TOP_GENES, adata.n_vars)

    if n_top >= 100:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top,
            flavor="seurat",
        )

        adata = adata[:, adata.var["highly_variable"]].copy()

        print(
            f"After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes"
        )
    else:
        print("Too few genes for HVG selection; skipping HVG.")

    sc.pp.scale(adata, max_value=10)

    n_comps = min(
        N_PCS,
        adata.n_obs - 1,
        adata.n_vars - 1,
    )

    if n_comps < 2:
        raise ValueError(
            "Too few cells or genes after filtering for PCA."
        )

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
    """
    根据 ROOT_MODE 选择 root cell index。
    """

    if ROOT_MODE == "lowest_total_counts":
        root_idx = int(
            np.argmin(adata.obs["total_counts_raw"].values)
        )

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
            raise ValueError(
                f"No valid numeric values found in {AXIS_SCORE_COL}."
            )

        cutoff = score.quantile(ROOT_LOW_QUANTILE)
        candidate = score <= cutoff

        if candidate.sum() == 0:
            root_idx = int(np.nanargmin(score.values))
        else:
            candidate_indices = np.where(candidate.values)[0]

            # 如果 diffmap 已经存在，在最低 axis_v2 区域中，
            # 选择 diffusion component 1 最小的细胞作为 root。
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
            raise ValueError(
                f"No valid numeric values found in {FIBROSIS_SCORE_COL}."
            )

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
        print(
            f"Root {AXIS_SCORE_COL}: "
            f"{adata.obs.iloc[root_idx][AXIS_SCORE_COL]}"
        )

    if FIBROSIS_SCORE_COL in adata.obs.columns:
        print(
            f"Root {FIBROSIS_SCORE_COL}: "
            f"{adata.obs.iloc[root_idx][FIBROSIS_SCORE_COL]}"
        )

    sc.tl.dpt(adata)

    adata.obs["dpt_pseudotime"] = adata.obs[
        "dpt_pseudotime"
    ].astype(float)

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
        "celltype_marker_based",
        "Disease_Identity",
        "axis_v2_stage",
        "multicellular_axis_stage",
    ]:
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
        "celltype_marker_based",
    ]:
        if col in adata.obs.columns:
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


def save_fate_probabilities_if_present(g, adata, out_path):
    try:
        fp = getattr(g, "fate_probabilities", None)

        if fp is None:
            print("No fate_probabilities attribute found.")
            return

        arr = np.asarray(fp)

        df = pd.DataFrame(
            arr,
            index=adata.obs_names,
        )

        df.index.name = "cell_id"

        df.to_csv(
            out_path,
            encoding="utf-8-sig",
        )

        print(f"Saved fate probabilities: {out_path}")

    except Exception as e:
        print(f"Could not save fate probabilities: {e}")


# ============================================================
# CellRank 主线
# ============================================================

def run_cellrank_optional(adata, out_dir):
    if not RUN_CELLRANK:
        print("\nCellRank skipped: RUN_CELLRANK = False")
        return adata

    print("\n========== Optional CellRank test ==========")

    try:
        from cellrank.kernels import ConnectivityKernel, PseudotimeKernel
        from cellrank.estimators import GPCCA
    except Exception as e:
        print("CellRank is not installed or cannot be imported.")
        print(f"Error: {e}")
        print("Install with: pip install cellrank")
        return adata

    n_states = min(3, max(2, adata.n_obs // 500))

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

            save_fate_probabilities_if_present(
                g,
                adata,
                Path(out_dir)
                / "cellrank_connectivity_fate_probabilities.csv",
            )

        except Exception as e:
            print(
                f"ConnectivityKernel terminal/fate step skipped: {e}"
            )

        print(
            "ConnectivityKernel CellRank test finished or partially finished."
        )

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

            save_fate_probabilities_if_present(
                g2,
                adata,
                Path(out_dir)
                / "cellrank_pseudotime_fate_probabilities.csv",
            )

        except Exception as e:
            print(
                f"PseudotimeKernel terminal/fate step skipped: {e}"
            )

        print(
            "PseudotimeKernel CellRank test finished or partially finished."
        )

    except Exception as e:
        print("PseudotimeKernel CellRank test did not complete.")
        print(f"Error: {e}")

    return adata


# ============================================================
# scVelo 占位：正式主线默认关闭
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
# h5ad 保存：默认关闭
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

    X, spliced, unspliced, genes, cells = read_exported_mtx(EXPORT_DIR)

    adata_raw = build_anndata(
        X,
        spliced,
        unspliced,
        genes,
        cells,
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

    # 正式 root 需要合并 axis metadata
    if ROOT_MODE in [
        "lowest_axis_v2",
        "lowest_fibrosis",
    ]:
        adata_raw = merge_axis_metadata(adata_raw)

    # pseudotime / connectivity 使用 processed adata
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

    adata = run_cellrank_optional(adata, OUT_DIR)

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
    print(f"  {Path(OUT_DIR) / 'figures'}")

    print("CellRank outputs:")

    print(
        f"  {Path(OUT_DIR) / 'cellrank_connectivity_fate_probabilities.csv'}"
    )

    print(
        f"  {Path(OUT_DIR) / 'cellrank_pseudotime_fate_probabilities.csv'}"
    )


if __name__ == "__main__":
    main()