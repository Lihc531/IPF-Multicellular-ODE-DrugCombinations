#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare DGIdb raw TSV exports into drug-target tables for PPI-informed driver refinement.

Expected input files from DGIdb download:
  - interactions.tsv
  - genes.tsv
  - drugs.tsv
  - categories.tsv

Main output for the PPI refinement script:
  - 02_drug_target_pairs_for_ppi.csv
    Required columns: gene, drug_name
    Extra columns: interaction_type, source, approved, immunotherapy, anti_neoplastic, interaction_score

Optional output for gene-level driver annotation:
  - 04_gene_target_annotation_for_ppi.csv
    One row per gene with n_drugs, drug_names, target_sources, and category summaries.

Usage:
  python 24_prepare_dgidb_drug_target_table.py \
      --dgidb_dir E:/PythonProject/dgidb \
      --outdir E:/PythonProject/results/dgidb_drug_targets

If your files are in the same directory as this script, simply run:
  python 24_prepare_dgidb_drug_target_table.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def read_tsv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"Optional file not found, skipped: {path}")
        return None
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig", low_memory=False)


def clean_symbol(x: object) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", "", x)
    return x.upper()


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def parse_bool(x: object) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}


def join_unique(values: Iterable[object], max_items: int = 80) -> str:
    vals = []
    for v in values:
        s = clean_text(v)
        if s and s.lower() not in {"nan", "none", "na"}:
            vals.append(s)
    vals = sorted(set(vals))
    if len(vals) > max_items:
        return ";".join(vals[:max_items]) + f";...(+{len(vals) - max_items})"
    return ";".join(vals)


def build_gene_name_map(genes: Optional[pd.DataFrame]) -> dict[str, str]:
    if genes is None or genes.empty:
        return {}
    df = genes.copy()
    for col in ["gene_claim_name", "gene_name"]:
        if col not in df.columns:
            return {}
    df["claim_clean"] = df["gene_claim_name"].map(clean_symbol)
    df["gene_clean"] = df["gene_name"].map(clean_symbol)
    df = df[(df["claim_clean"] != "") & (df["gene_clean"] != "")]
    # Prefer the most frequent mapping for each claim name.
    mapping = (
        df.groupby(["claim_clean", "gene_clean"])
        .size()
        .reset_index(name="n")
        .sort_values(["claim_clean", "n"], ascending=[True, False])
        .drop_duplicates("claim_clean")
    )
    return dict(zip(mapping["claim_clean"], mapping["gene_clean"]))


def build_drug_name_map(drugs: Optional[pd.DataFrame]) -> dict[str, str]:
    if drugs is None or drugs.empty:
        return {}
    df = drugs.copy()
    for col in ["drug_claim_name", "drug_name"]:
        if col not in df.columns:
            return {}
    df["claim_clean"] = df["drug_claim_name"].map(clean_text).str.upper()
    df["drug_clean"] = df["drug_name"].map(clean_text)
    df = df[(df["claim_clean"] != "") & (df["drug_clean"] != "")]
    mapping = (
        df.groupby(["claim_clean", "drug_clean"])
        .size()
        .reset_index(name="n")
        .sort_values(["claim_clean", "n"], ascending=[True, False])
        .drop_duplicates("claim_clean")
    )
    return dict(zip(mapping["claim_clean"], mapping["drug_clean"]))


def build_category_table(categories: Optional[pd.DataFrame]) -> pd.DataFrame:
    if categories is None or categories.empty:
        return pd.DataFrame(columns=["gene", "gene_categories", "category_sources"])
    df = categories.copy()
    # DGIdb categories.tsv usually has: name, name-2, source_db_name, source_db_version.
    if "name" not in df.columns or "name-2" not in df.columns:
        return pd.DataFrame(columns=["gene", "gene_categories", "category_sources"])
    df["gene"] = df["name"].map(clean_symbol)
    df["category"] = df["name-2"].map(clean_text)
    if "source_db_name" not in df.columns:
        df["source_db_name"] = "unknown"
    df = df[(df["gene"] != "") & (df["category"] != "")]
    out = (
        df.groupby("gene")
        .agg(
            gene_categories=("category", join_unique),
            category_sources=("source_db_name", join_unique),
        )
        .reset_index()
    )
    return out


def prepare_interactions(
    interactions: pd.DataFrame,
    gene_map: dict[str, str],
    drug_map: dict[str, str],
    min_interaction_score: Optional[float] = None,
    approved_only: bool = False,
) -> pd.DataFrame:
    required = {"gene_claim_name", "drug_claim_name"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"interactions.tsv is missing required columns: {missing}")

    df = interactions.copy()

    # Standard gene symbol: prefer DGIdb mapped gene_name; fallback to mapped claim; fallback to claim itself.
    if "gene_name" not in df.columns:
        df["gene_name"] = ""
    df["gene"] = df["gene_name"].map(clean_symbol)
    claim_gene = df["gene_claim_name"].map(clean_symbol)
    mapped_gene = claim_gene.map(gene_map).fillna("")
    df.loc[df["gene"] == "", "gene"] = mapped_gene[df["gene"] == ""]
    df.loc[df["gene"] == "", "gene"] = claim_gene[df["gene"] == ""]

    # Standard drug name: prefer DGIdb mapped drug_name; fallback to mapped claim; fallback to claim itself.
    if "drug_name" not in df.columns:
        df["drug_name"] = ""
    df["drug_name_std"] = df["drug_name"].map(clean_text)
    claim_drug = df["drug_claim_name"].map(clean_text)
    mapped_drug = claim_drug.str.upper().map(drug_map).fillna("")
    df.loc[df["drug_name_std"] == "", "drug_name_std"] = mapped_drug[df["drug_name_std"] == ""]
    df.loc[df["drug_name_std"] == "", "drug_name_std"] = claim_drug[df["drug_name_std"] == ""]

    # Canonical columns.
    if "interaction_source_db_name" not in df.columns:
        df["interaction_source_db_name"] = "unknown"
    if "interaction_type" not in df.columns:
        df["interaction_type"] = "unknown"
    if "interaction_score" not in df.columns:
        df["interaction_score"] = np.nan
    if "approved" not in df.columns:
        df["approved"] = False
    if "immunotherapy" not in df.columns:
        df["immunotherapy"] = False
    if "anti_neoplastic" not in df.columns:
        df["anti_neoplastic"] = False

    df["interaction_score"] = pd.to_numeric(df["interaction_score"], errors="coerce")
    df["source"] = df["interaction_source_db_name"].map(clean_text)
    df["interaction_type"] = df["interaction_type"].map(clean_text)
    df.loc[df["interaction_type"] == "", "interaction_type"] = "unspecified"
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        df[col] = df[col].map(parse_bool)

    df = df[(df["gene"] != "") & (df["drug_name_std"] != "")].copy()
    # Replace original DGIdb drug_name column with the standardized drug_name to avoid duplicate columns.
    if "drug_name" in df.columns:
        df = df.drop(columns=["drug_name"])
    df = df.rename(columns={"drug_name_std": "drug_name"})

    if min_interaction_score is not None:
        df = df[df["interaction_score"].fillna(-np.inf) >= float(min_interaction_score)].copy()
    if approved_only:
        df = df[df["approved"]].copy()

    keep = [
        "gene",
        "drug_name",
        "interaction_type",
        "interaction_score",
        "source",
        "approved",
        "immunotherapy",
        "anti_neoplastic",
        "gene_claim_name",
        "gene_concept_id" if "gene_concept_id" in df.columns else None,
        "drug_claim_name",
        "drug_concept_id" if "drug_concept_id" in df.columns else None,
    ]
    keep = [c for c in keep if c is not None and c in df.columns]
    out = df[keep].drop_duplicates().reset_index(drop=True)
    return out


def aggregate_gene_drug_pairs(clean: pd.DataFrame, categories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # One row per gene-drug pair, suitable for PPI script drug_target_csv.
    pairs = (
        clean.groupby(["gene", "drug_name"])
        .agg(
            interaction_type=("interaction_type", join_unique),
            source=("source", join_unique),
            n_sources=("source", lambda x: len(set(clean_text(v) for v in x if clean_text(v)))),
            n_records=("source", "size"),
            max_interaction_score=("interaction_score", "max"),
            mean_interaction_score=("interaction_score", "mean"),
            approved=("approved", "max"),
            immunotherapy=("immunotherapy", "max"),
            anti_neoplastic=("anti_neoplastic", "max"),
        )
        .reset_index()
    )
    pairs["max_interaction_score"] = pairs["max_interaction_score"].fillna(0.0)
    pairs["mean_interaction_score"] = pairs["mean_interaction_score"].fillna(0.0)
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        pairs[col] = pairs[col].astype(bool)

    # A simple prioritization score for drug-target evidence, not a biological efficacy score.
    pairs["dgidb_target_evidence_score"] = (
        0.35 * pairs["approved"].astype(float)
        + 0.20 * pairs["anti_neoplastic"].astype(float)
        + 0.15 * pairs["immunotherapy"].astype(float)
        + 0.15 * np.log1p(pairs["n_sources"]) / np.log(5)
        + 0.15 * pairs["max_interaction_score"].clip(0, 1)
    ).clip(0, 1)

    # One row per gene for annotation/summary.
    gene_summary = (
        pairs.groupby("gene")
        .agg(
            n_drugs=("drug_name", "nunique"),
            n_approved_drugs=("approved", "sum"),
            n_anti_neoplastic_drugs=("anti_neoplastic", "sum"),
            n_immunotherapy_drugs=("immunotherapy", "sum"),
            drug_names=("drug_name", join_unique),
            approved_drug_names=("drug_name", lambda x: ""),
            target_sources=("source", join_unique),
            interaction_types=("interaction_type", join_unique),
            max_dgidb_target_evidence_score=("dgidb_target_evidence_score", "max"),
            mean_dgidb_target_evidence_score=("dgidb_target_evidence_score", "mean"),
        )
        .reset_index()
    )

    # Fill approved drug names separately to avoid mixing groupby columns.
    approved_names = (
        pairs[pairs["approved"]]
        .groupby("gene")["drug_name"]
        .apply(join_unique)
        .reset_index(name="approved_drug_names")
    )
    anti_names = (
        pairs[pairs["anti_neoplastic"]]
        .groupby("gene")["drug_name"]
        .apply(join_unique)
        .reset_index(name="anti_neoplastic_drug_names")
    )
    gene_summary = gene_summary.drop(columns=["approved_drug_names"]).merge(approved_names, on="gene", how="left")
    gene_summary = gene_summary.merge(anti_names, on="gene", how="left")
    gene_summary["approved_drug_names"] = gene_summary["approved_drug_names"].fillna("")
    gene_summary["anti_neoplastic_drug_names"] = gene_summary["anti_neoplastic_drug_names"].fillna("")
    gene_summary["is_druggable"] = gene_summary["n_drugs"] > 0

    if not categories.empty:
        pairs = pairs.merge(categories, on="gene", how="left")
        gene_summary = gene_summary.merge(categories, on="gene", how="left")
    for df in [pairs, gene_summary]:
        for col in ["gene_categories", "category_sources"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

    top_gene_summary = gene_summary.sort_values(
        ["n_approved_drugs", "max_dgidb_target_evidence_score", "n_drugs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return pairs, gene_summary, top_gene_summary


def maybe_filter_to_driver_genes(pairs: pd.DataFrame, driver_gene_file: Optional[Path], outdir: Path) -> Optional[pd.DataFrame]:
    if driver_gene_file is None or not driver_gene_file.exists():
        return None
    driver_df = pd.read_csv(driver_gene_file, dtype=str, low_memory=False)
    gene_col = None
    for c in ["gene", "Gene", "symbol", "gene_symbol", "hgnc_symbol"]:
        if c in driver_df.columns:
            gene_col = c
            break
    if gene_col is None:
        print(f"Driver gene file provided, but no gene column was detected: {driver_gene_file}")
        return None
    driver_genes = set(driver_df[gene_col].map(clean_symbol))
    driver_genes.discard("")
    sub = pairs[pairs["gene"].isin(driver_genes)].copy()
    sub.to_csv(outdir / "06_drug_target_pairs_for_scored_driver_genes.csv", index=False)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgidb_dir", default=str(Path(__file__).resolve().parent), help="Directory containing DGIdb TSV files.")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "dgidb_drug_target_outputs"), help="Output directory.")
    parser.add_argument("--min_interaction_score", type=float, default=None, help="Optional minimum DGIdb interaction score.")
    parser.add_argument("--approved_only", action="store_true", help="Keep only approved drugs.")
    parser.add_argument("--driver_gene_file", default="", help="Optional PPI driver score CSV to subset drug-target pairs to scored genes.")
    args = parser.parse_args()

    dgidb_dir = Path(args.dgidb_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading DGIdb tables...")
    interactions = read_tsv(dgidb_dir / "interactions.tsv", required=True)
    genes = read_tsv(dgidb_dir / "genes.tsv", required=False)
    drugs = read_tsv(dgidb_dir / "drugs.tsv", required=False)
    categories = read_tsv(dgidb_dir / "categories.tsv", required=False)

    print("Building standard name maps...")
    gene_map = build_gene_name_map(genes)
    drug_map = build_drug_name_map(drugs)
    category_table = build_category_table(categories)

    print("Cleaning interaction records...")
    clean = prepare_interactions(
        interactions,
        gene_map=gene_map,
        drug_map=drug_map,
        min_interaction_score=args.min_interaction_score,
        approved_only=args.approved_only,
    )
    clean.to_csv(outdir / "01_dgidb_interactions_clean.csv", index=False)

    print("Aggregating gene-drug pairs and gene-level summaries...")
    pairs, gene_summary, top_gene_summary = aggregate_gene_drug_pairs(clean, category_table)

    # This is the file to use as drug_target_csv in the PPI refinement script.
    ppi_cols = [
        "gene",
        "drug_name",
        "interaction_type",
        "source",
        "approved",
        "immunotherapy",
        "anti_neoplastic",
        "max_interaction_score",
        "mean_interaction_score",
        "n_sources",
        "n_records",
        "dgidb_target_evidence_score",
    ]
    extra_cols = [c for c in ["gene_categories", "category_sources"] if c in pairs.columns]
    pairs[ppi_cols + extra_cols].to_csv(outdir / "02_drug_target_pairs_for_ppi.csv", index=False)
    pairs.to_csv(outdir / "03_gene_drug_pair_summary_full.csv", index=False)
    gene_summary.to_csv(outdir / "04_gene_target_annotation_for_ppi.csv", index=False)
    top_gene_summary.to_csv(outdir / "05_top_druggable_genes_by_dgidb.csv", index=False)

    subset = None
    driver_gene_file = Path(args.driver_gene_file) if args.driver_gene_file else None
    if driver_gene_file:
        subset = maybe_filter_to_driver_genes(pairs, driver_gene_file, outdir)

    summary = {
        "input_rows": {
            "interactions": int(interactions.shape[0]),
            "genes": int(genes.shape[0]) if genes is not None else 0,
            "drugs": int(drugs.shape[0]) if drugs is not None else 0,
            "categories": int(categories.shape[0]) if categories is not None else 0,
        },
        "clean_interaction_rows": int(clean.shape[0]),
        "unique_genes_with_drug_interactions": int(pairs["gene"].nunique()),
        "unique_drugs_with_gene_interactions": int(pairs["drug_name"].nunique()),
        "unique_gene_drug_pairs": int(pairs.shape[0]),
        "approved_gene_drug_pairs": int(pairs["approved"].sum()),
        "anti_neoplastic_gene_drug_pairs": int(pairs["anti_neoplastic"].sum()),
        "immunotherapy_gene_drug_pairs": int(pairs["immunotherapy"].sum()),
        "driver_gene_subset_pairs": int(subset.shape[0]) if subset is not None else None,
        "outputs": {
            "ppi_drug_target_csv": str(outdir / "02_drug_target_pairs_for_ppi.csv"),
            "gene_target_annotation": str(outdir / "04_gene_target_annotation_for_ppi.csv"),
            "top_druggable_genes": str(outdir / "05_top_druggable_genes_by_dgidb.csv"),
        },
    }
    with open(outdir / "00_dgidb_prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()