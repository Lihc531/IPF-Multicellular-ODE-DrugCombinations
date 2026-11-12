#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_integrate_cellrank_directionality_non_refined_celltypes.py

Purpose
-------
Integrate current CellRank / DPT directionality support into final driver tables,
excluding cell types that will be rerun later:
    Fibroblast
    AT1_AT2
    Endothelial

This script does NOT rerun CellRank.
It annotates existing driver candidates with current directionality evidence.

Inputs
------
From:
    E:/PythonProject/results/ppi_informed_driver_refinement_layers1_2_v1B1/tables/

Required:
    15_final_driver_shortlist_top50.csv
    16_tierA_primary_driver_candidates.csv
    17_mechanism_grouped_driver_table.csv
    18_supplementary_watchlist_candidates.csv

Optional:
    11d_high_priority_druggable_driver_candidates.csv

Outputs
-------
    19_driver_directionality_support_non_refined_celltypes.csv
    20_driver_mechanism_interpretation_non_refined_celltypes.csv
    21_driver_result_narrative_table.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = Path(
    r"E:\PythonProject\results"
    r"\ppi_informed_driver_refinement_layers1_2_v1B1"
    r"\tables"
)

OUT_DIR = BASE_DIR

# These cell types will be rerun later and are NOT used as current directionality evidence.
EXCLUDED_RERUN_CELLTYPES = {
    "Fibroblast",
    "AT1_AT2",
    "Endothelial",
}

# Current CellRank/DPT evidence summary from your previous analysis.
# This is intentionally conservative.
CELLRANK_EVIDENCE = {
    "Macrophage": {
        "usable": True,
        "support_type": "IPF-associated myeloid terminal-like trajectory",
        "strength": "moderate",
        "supports_axis": "weak",
        "supports_fibrosis": "no",
        "interpretation": (
            "Macrophage DPT/CellRank identified terminal-like states enriched in IPF, "
            "but not aligned with fibrosis composite score; best interpreted as an "
            "IPF-associated inflammatory/myeloid transition."
        ),
    },
    "Plasma": {
        "usable": True,
        "support_type": "IPF-enriched humoral immune terminal-like trajectory",
        "strength": "moderate",
        "supports_axis": "weak_positive",
        "supports_fibrosis": "weak_positive",
        "interpretation": (
            "Plasma-cell trajectory showed IPF-enriched terminal-like states, "
            "supporting humoral immune remodeling."
        ),
    },
    "T_cell": {
        "usable": True,
        "support_type": "IPF-enriched T-cell terminal-like trajectory",
        "strength": "moderate",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "T-cell terminal-like states were enriched in IPF, supporting disease-associated "
            "lymphoid immune remodeling rather than a fibrosis-aligned continuous trajectory."
        ),
    },
    "B_cell": {
        "usable": True,
        "support_type": "weak B-cell immune remodeling trajectory",
        "strength": "weak_to_moderate",
        "supports_axis": "weak",
        "supports_fibrosis": "weak",
        "interpretation": (
            "B-cell result provides weak-to-moderate support for immune remodeling and is best "
            "used as supplementary evidence."
        ),
    },
    "Smooth_muscle": {
        "usable": True,
        "support_type": "weak mesenchymal/vascular support",
        "strength": "weak",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "Smooth muscle result showed weak directionality and should only be used as "
            "supplementary or context evidence."
        ),
    },
    "NK": {
        "usable": True,
        "support_type": "weak NK immune context",
        "strength": "weak",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "NK result showed weak directionality and is not recommended as a major validation line."
        ),
    },
    "Epithelial": {
        "usable": False,
        "support_type": "current broad epithelial CellRank not used as strong evidence",
        "strength": "not_used",
        "supports_axis": "not_used",
        "supports_fibrosis": "not_used",
        "interpretation": (
            "Current broad epithelial CellRank is not used as strong directionality evidence; "
            "epithelial interpretation should rely on the epithelial injury-remodeling axis "
            "and GSE159354 projection validation."
        ),
    },
}


# Gene-level mapping to current usable directionality evidence.
# Excludes Fibroblast / AT1_AT2 / Endothelial evidence because those will be rerun.
GENE_DIRECTIONALITY_MAP = {
    # Macrophage / myeloid
    "CCL18": ["Macrophage"],
    "SPP1": ["Macrophage"],
    "CD74": ["Macrophage", "T_cell", "Plasma", "B_cell"],
    "LPL": ["Macrophage"],
    "S100A10": ["Macrophage"],
    "FOS": ["Macrophage"],
    "APOE": ["Macrophage"],
    "TREM2": ["Macrophage"],
    "C1QA": ["Macrophage"],
    "C1QB": ["Macrophage"],
    "C1QC": ["Macrophage"],
    "LGALS3": ["Macrophage"],

    # Lymphoid / humoral immune
    "CCL5": ["T_cell"],
    "CXCL13": ["B_cell", "Plasma"],
    "CXCL10": ["T_cell"],
    "IL32": ["T_cell"],
    "HLA-DRA": ["Macrophage", "B_cell", "Plasma"],
    "HLA-DRB1": ["Macrophage", "B_cell", "Plasma"],

    # Epithelial injury genes are not using current AT1_AT2 CellRank,
    # but can be supported by epithelial axis / GSE159354.
    "MMP7": ["Epithelial_axis_validation"],
    "KRT17": ["Epithelial_axis_validation"],
    "CDKN2A": ["Epithelial_axis_validation"],

    # Smooth muscle / vascular context, weak current support
    "ACTA2": ["Smooth_muscle"],
    "TAGLN": ["Smooth_muscle"],
    "MYH11": ["Smooth_muscle"],
    "CNN1": ["Smooth_muscle"],
}


EPITHELIAL_AXIS_SUPPORTED = {"MMP7", "KRT17", "CDKN2A"}

MECHANISM_LABELS = {
    "Macrophage": "macrophage_IPF_associated_myeloid_transition",
    "T_cell": "T_cell_IPF_terminal_like_immune_remodeling",
    "Plasma": "plasma_humoral_immune_remodeling",
    "B_cell": "B_cell_immune_remodeling",
    "Smooth_muscle": "smooth_muscle_weak_context_support",
    "NK": "NK_weak_context_support",
    "Epithelial_axis_validation": "epithelial_injury_axis_validation",
}


# ============================================================
# 1. Utilities
# ============================================================

def clean_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "gene" in df.columns:
        df["gene"] = df["gene"].map(clean_gene)
    return df


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[optional missing] {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "gene" in df.columns:
        df["gene"] = df["gene"].map(clean_gene)
    return df


def as_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    return s.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])


def ensure_bool(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = False
        df[c] = as_bool(df[c])
    return df


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def directionality_support_for_gene(gene: str):
    gene = clean_gene(gene)
    supports = GENE_DIRECTIONALITY_MAP.get(gene, [])

    labels = []
    strengths = []
    notes = []

    for support in supports:
        if support == "Epithelial_axis_validation":
            labels.append(MECHANISM_LABELS[support])
            strengths.append("moderate")
            notes.append(
                "Supported by epithelial injury-remodeling axis / validation rather than current broad CellRank."
            )
            continue

        ev = CELLRANK_EVIDENCE.get(support)
        if ev is None:
            continue

        if ev["usable"]:
            labels.append(MECHANISM_LABELS.get(support, support))
            strengths.append(ev["strength"])
            notes.append(ev["interpretation"])

    if not labels:
        return {
            "current_directionality_support": "no_direct_support_in_current_non_refined_celltypes",
            "current_directionality_strength": "none",
            "directionality_supporting_celltypes": "",
            "directionality_interpretation_note": "",
        }

    # summarize strength
    if "moderate" in strengths:
        final_strength = "moderate"
    elif "weak_to_moderate" in strengths:
        final_strength = "weak_to_moderate"
    elif "weak" in strengths:
        final_strength = "weak"
    else:
        final_strength = "context"

    return {
        "current_directionality_support": ";".join(sorted(set(labels))),
        "current_directionality_strength": final_strength,
        "directionality_supporting_celltypes": ";".join(supports),
        "directionality_interpretation_note": " | ".join(sorted(set(notes))),
    }


def mechanism_interpretation(row: pd.Series) -> str:
    gene = clean_gene(row.get("gene", ""))

    if gene in {"CCL18", "SPP1", "CD74", "LPL", "S100A10", "FOS", "APOE", "TREM2", "LGALS3"}:
        return (
            f"{gene} links the PPI-informed driver network to IPF-associated macrophage/myeloid remodeling. "
            "Current CellRank/DPT evidence supports an IPF-enriched myeloid terminal-like state, "
            "but this should be interpreted as inflammatory/myeloid progression rather than a direct fibrosis-terminal trajectory."
        )

    if gene in {"CCL5", "CXCL10", "IL32"}:
        return (
            f"{gene} is best interpreted as a lymphoid immune remodeling driver. "
            "Current T-cell/NK-related directionality support is context-level and should be used as supplementary evidence."
        )

    if gene in {"CXCL13", "CD74", "HLA-DRA", "HLA-DRB1"}:
        return (
            f"{gene} supports B/plasma/antigen-presentation immune remodeling. "
            "Plasma and B-cell CellRank results provide supplementary support for IPF-associated humoral immune states."
        )

    if gene in {"MMP7", "KRT17", "CDKN2A"}:
        return (
            f"{gene} should be interpreted through the epithelial injury-remodeling axis and external epithelial validation. "
            "Current broad epithelial/AT1_AT2 CellRank should not be used as the primary directionality evidence until rerun."
        )

    if gene in {"ACTA2", "TAGLN", "MYH11", "CNN1"}:
        return (
            f"{gene} is a smooth-muscle/pericyte or mesenchymal context candidate. "
            "Current Smooth_muscle CellRank support is weak and should be treated as supplementary."
        )

    return (
        f"{gene} remains a PPI/ODE-informed driver candidate, but current non-refined CellRank evidence "
        "does not provide direct strong directionality support. Interpret using ODE, dynamic module, PPI, and druggability evidence."
    )


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "gene",
        "driver_score_rank",
        "ppi_informed_driver_score",
        "refined_priority_tier",
        "refined_driver_category",
        "mechanism_group",
        "current_directionality_support",
        "current_directionality_strength",
        "directionality_supporting_celltypes",
        "directionality_interpretation_note",
        "mechanism_interpretation",
        "top_celltype",
        "n_celltypes",
        "n_states",
        "active_progression_driver",
        "terminal_state_maintenance_driver",
        "shared_profibrotic_hub",
        "celltype_specific_driver",
        "high_priority_druggable_driver",
        "approved_druggable_driver",
        "anti_neoplastic_druggable_driver",
        "progression_driver_subscore",
        "terminal_maintenance_subscore",
        "network_hub_subscore",
        "neural_ode_direction_score_norm",
        "ppi_centrality_score_norm",
        "druggability_score_norm",
        "drug_names",
    ]
    existing = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    return df[existing + rest]


# ============================================================
# 2. Main
# ============================================================

def main():
    print("BASE_DIR:", BASE_DIR)

    top50 = read_csv_required(BASE_DIR / "15_final_driver_shortlist_top50.csv")
    tier_a = read_csv_required(BASE_DIR / "16_tierA_primary_driver_candidates.csv")
    mech = read_csv_required(BASE_DIR / "17_mechanism_grouped_driver_table.csv")
    watch = read_csv_optional(BASE_DIR / "18_supplementary_watchlist_candidates.csv")
    drug = read_csv_optional(BASE_DIR / "11d_high_priority_druggable_driver_candidates.csv")

    bool_cols = [
        "active_progression_driver",
        "terminal_state_maintenance_driver",
        "shared_profibrotic_hub",
        "celltype_specific_driver",
        "high_priority_druggable_driver",
        "approved_druggable_driver",
        "anti_neoplastic_druggable_driver",
    ]
    num_cols = [
        "ppi_informed_driver_score",
        "progression_driver_subscore",
        "terminal_maintenance_subscore",
        "network_hub_subscore",
        "neural_ode_direction_score_norm",
        "ppi_centrality_score_norm",
        "druggability_score_norm",
        "n_celltypes",
        "n_states",
    ]

    frames = []
    for label, df in [
        ("top50", top50),
        ("tierA", tier_a),
        ("mechanism", mech),
        ("watchlist", watch),
        ("druggable", drug),
    ]:
        if df.empty:
            continue
        df = df.copy()
        df["source_table"] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["gene"] = combined["gene"].map(clean_gene)

    # Collapse duplicated genes across source tables.
    combined["_source_priority"] = combined["source_table"].map({
        "tierA": 1,
        "top50": 2,
        "mechanism": 3,
        "druggable": 4,
        "watchlist": 5,
    }).fillna(9)

    combined = combined.sort_values(
        ["gene", "_source_priority", "ppi_informed_driver_score"],
        ascending=[True, True, False],
    ).drop_duplicates("gene", keep="first")

    combined = ensure_bool(combined, bool_cols)
    combined = ensure_numeric(combined, num_cols)

    support_df = pd.DataFrame(
        [directionality_support_for_gene(g) for g in combined["gene"]]
    )
    combined = pd.concat([combined.reset_index(drop=True), support_df], axis=1)

    combined["mechanism_interpretation"] = combined.apply(mechanism_interpretation, axis=1)

    # --------------------------------------------------------
    # 19. Full non-refined-celltype directionality support table
    # --------------------------------------------------------

    out19 = combined.sort_values(
        ["current_directionality_strength", "ppi_informed_driver_score"],
        ascending=[True, False],
    ).copy()

    # Custom strength order
    strength_order = {
        "moderate": 1,
        "weak_to_moderate": 2,
        "weak": 3,
        "context": 4,
        "none": 5,
    }
    out19["_strength_order"] = out19["current_directionality_strength"].map(strength_order).fillna(9)
    out19 = out19.sort_values(
        ["_strength_order", "ppi_informed_driver_score"],
        ascending=[True, False],
    ).drop(columns=["_strength_order"])

    out19 = reorder(out19)
    out19.to_csv(OUT_DIR / "19_driver_directionality_support_non_refined_celltypes.csv", index=False)

    # --------------------------------------------------------
    # 20. Mechanism interpretation table
    # --------------------------------------------------------

    # Keep genes with any current support, plus Tier A genes.
    tier_a_genes = set(tier_a["gene"].map(clean_gene))
    out20 = combined[
        (combined["current_directionality_strength"] != "none")
        | (combined["gene"].isin(tier_a_genes))
    ].copy()

    out20 = out20.sort_values(
        ["current_directionality_strength", "ppi_informed_driver_score"],
        ascending=[True, False],
    )
    out20 = reorder(out20)
    out20.to_csv(OUT_DIR / "20_driver_mechanism_interpretation_non_refined_celltypes.csv", index=False)

    # --------------------------------------------------------
    # 21. Result narrative table
    # --------------------------------------------------------

    narrative_rows = []
    for _, row in out20.iterrows():
        gene = row["gene"]
        support = row["current_directionality_support"]
        strength = row["current_directionality_strength"]
        category = row.get("refined_driver_category", "")
        tier = row.get("refined_priority_tier", "")
        score = row.get("ppi_informed_driver_score", np.nan)
        interp = row.get("mechanism_interpretation", "")

        narrative = (
            f"{gene} was prioritized as a {tier} candidate "
            f"(PPI-informed score={score:.3f}) with category evidence [{category}]. "
            f"Current directionality support: {support} ({strength}). {interp}"
        )

        narrative_rows.append({
            "gene": gene,
            "ppi_informed_driver_score": score,
            "refined_priority_tier": tier,
            "refined_driver_category": category,
            "current_directionality_support": support,
            "current_directionality_strength": strength,
            "result_sentence": narrative,
        })

    out21 = pd.DataFrame(narrative_rows)
    out21.to_csv(OUT_DIR / "21_driver_result_narrative_table.csv", index=False)

    print("\nDone. Generated:")
    print("  19_driver_directionality_support_non_refined_celltypes.csv")
    print("  20_driver_mechanism_interpretation_non_refined_celltypes.csv")
    print("  21_driver_result_narrative_table.csv")

    print("\nCounts:")
    print("  combined genes:", combined.shape[0])
    print("  supported genes:", int((combined["current_directionality_strength"] != "none").sum()))
    print("  table 19:", out19.shape)
    print("  table 20:", out20.shape)
    print("  table 21:", out21.shape)


if __name__ == "__main__":
    main()