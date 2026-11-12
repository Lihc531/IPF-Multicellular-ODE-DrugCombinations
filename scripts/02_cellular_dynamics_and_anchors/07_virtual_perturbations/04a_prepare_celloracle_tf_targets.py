#!/usr/bin/env python3
"""Build an anchor-gene -> upstream-TF manifest for CellOracle.

The R network perturbation branch (03) keeps the original key-driver genes such
as FN1, VWF or CD74. CellOracle, however, perturbs transcription factors. This
script creates the missing mapping layer without overwriting the original
``targets.csv`` files.

Selection evidence in the default ``main_discovery_only`` mode:
  1. TF is a promoter-prior regulator of the anchor gene in CellOracle's human
     promoter base-GRN;
  2. TF is present and expressed in the prepared MAIN discovery input;
  3. optional MAIN fitted-Link support;
  4. optional support from the original PPI/dynamic candidate table.

The validation dataset is never used to rank or select TFs in the default mode.
It is audited only after the frozen MAIN-derived target list has been created.

Typical workflow:
  python 04a_prepare_celloracle_tf_targets.py --apply

Outputs:
  virtual_perturbation_inputs/supporting_tables/
      CellOracle_human_promoter_TF_universe.csv
  virtual_perturbation_results/01_target_manifest/
      celloracle_tf_candidate_review.csv
      celloracle_tf_target_manifest.csv
  virtual_perturbation_results/03_celloracle_inputs/<dataset>__<celltype>/
      targets_celloracle.csv                 (with --apply)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

try:
    import celloracle as co
except ModuleNotFoundError as exc:
    raise SystemExit(
        "CellOracle is not installed in this interpreter.\n"
        f"Interpreter: {sys.executable}\n"
        "Run this script inside the same conda environment used for "
        "05_run_celloracle_ko_oe.py."
    ) from exc


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE_ROOT = Path(os.getenv("VP_PIPELINE_ROOT", str(SCRIPT_ROOT)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create CellOracle upstream-TF targets mapped to original anchor genes."
    )
    p.add_argument("--pipeline-root", default=str(DEFAULT_PIPELINE_ROOT))
    p.add_argument("--input-root", default=os.getenv("VP_INPUT_ROOT"))
    p.add_argument("--output-root", default=os.getenv("VP_OUTPUT_ROOT"))
    p.add_argument("--top-n", type=int, default=None,
                   help="TFs per anchor. Default: 3 in pilot, 5 in full.")
    p.add_argument("--min-expressing-cells", type=int, default=25)
    p.add_argument("--celltype", action="append", default=[],
                   help="Optional cell-type filter; repeat for multiple cell types.")
    p.add_argument("--apply", action="store_true",
                   help="Write targets_celloracle.csv into prepared input folders.")
    p.add_argument("--tf-universe-only", action="store_true",
                   help="Only export the CellOracle TF universe and exit.")
    p.add_argument("--no-fallback", action="store_true",
                   help="Do not fall back to expressed candidate TFs when no direct prior TF is found.")
    p.add_argument("--use-fitted-links", action="store_true",
                   help="Use existing fitted CellOracle Links as additional TF-ranking evidence. "
                        "In main_discovery_only mode only MAIN Links are used.")
    p.add_argument("--selection-mode", choices=["main_discovery_only", "legacy_cross_dataset"],
                   default="main_discovery_only",
                   help="Default prevents validation leakage. legacy_cross_dataset is provided only for audit/reproduction.")
    p.add_argument("--reuse-frozen", action="store_true",
                   help="Reuse an existing frozen MAIN-derived manifest instead of reselecting TFs.")
    p.add_argument("--overwrite-frozen", action="store_true",
                   help="Allow replacing an existing frozen manifest.")
    p.add_argument("--ignore-fitted-links", action="store_true",
                   help="Deprecated compatibility flag; fitted Links are already ignored by default.")
    return p.parse_args()


def resolve_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    pipeline = Path(args.pipeline_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve() if args.input_root else pipeline / "virtual_perturbation_inputs"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else pipeline / "virtual_perturbation_results"
    return pipeline, input_root, output_root


def load_base_grn() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    print("Loading CellOracle human promoter base-GRN...")
    base = co.data.load_human_promoter_base_GRN()
    if not isinstance(base, pd.DataFrame):
        base = pd.DataFrame(base)
    metadata_columns = {"peak_id", "gene_short_name"}
    tf_columns = [str(c) for c in base.columns if str(c) not in metadata_columns]
    if not tf_columns:
        raise RuntimeError("No TF columns were detected in the CellOracle promoter base-GRN.")
    tf_case = {x.upper(): x for x in tf_columns}
    return base, tf_columns, tf_case


def direct_prior_tfs(base: pd.DataFrame, tf_columns: list[str], anchor: str) -> set[str]:
    if "gene_short_name" not in base.columns:
        return set()
    rows = base[base["gene_short_name"].astype(str).str.upper() == anchor.upper()]
    if rows.empty:
        return set()
    values = rows[tf_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    active = (values != 0).any(axis=0)
    return set(active.index[active].astype(str))


def read_job(job_dir: Path) -> dict[str, object]:
    required = ["counts_genes_by_cells.mtx", "genes.tsv", "barcodes.tsv", "cell_metadata.csv"]
    missing = [x for x in required if not (job_dir / x).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete prepared job {job_dir}; missing: {', '.join(missing)}")
    genes = pd.read_csv(job_dir / "genes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    cells = pd.read_csv(job_dir / "barcodes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    meta = pd.read_csv(job_dir / "cell_metadata.csv")
    if "cell_id" not in meta.columns or "disease_for_perturbation" not in meta.columns:
        raise ValueError(f"cell_metadata.csv lacks cell_id or disease_for_perturbation: {job_dir}")
    meta = meta.set_index("cell_id").loc[cells]
    counts = mmread(job_dir / "counts_genes_by_cells.mtx").tocsr()
    if counts.shape != (len(genes), len(cells)):
        raise ValueError(f"Matrix shape mismatch in {job_dir}: {counts.shape}")
    return {
        "genes": genes,
        "gene_index": {g.upper(): i for i, g in enumerate(genes)},
        "cells": cells,
        "meta": meta,
        "counts": counts,
    }


def expression_stats(job: dict[str, object], genes: list[str]) -> pd.DataFrame:
    meta: pd.DataFrame = job["meta"]  # type: ignore[assignment]
    counts = job["counts"]
    gene_index: dict[str, int] = job["gene_index"]  # type: ignore[assignment]
    rows: list[dict[str, object]] = []
    conditions = meta["disease_for_perturbation"].astype(str).to_numpy()
    for gene in genes:
        idx = gene_index.get(gene.upper())
        row: dict[str, object] = {"gene": gene, "present": idx is not None}
        for condition in ("Control", "IPF"):
            mask = conditions == condition
            n_total = int(mask.sum())
            if idx is None or n_total == 0:
                n_expr = 0
            else:
                values = counts[idx, mask]
                n_expr = int(values.getnnz()) if sparse.issparse(values) else int(np.sum(np.asarray(values) > 0))
            row[f"{condition}_n_cells"] = n_total
            row[f"{condition}_n_expr"] = n_expr
            row[f"{condition}_expr_fraction"] = n_expr / n_total if n_total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)




def load_fitted_link_coefficients(output_root: Path, dataset: str, celltype: str, anchor: str) -> dict[str, float]:
    """Return TF -> strongest fitted coefficient to anchor across available conditions."""
    path = output_root / "04_celloracle" / f"{dataset}__{celltype}" / "links.celloracle.links"
    if not path.is_file():
        return {}
    try:
        links = co.load_hdf5(str(path))
    except Exception as exc:
        print(f"WARNING: could not load fitted Links {path}: {exc}")
        return {}
    dictionaries = []
    if hasattr(links, "filtered_links") and isinstance(links.filtered_links, dict):
        dictionaries.append(links.filtered_links)
    if hasattr(links, "links_dict") and isinstance(links.links_dict, dict):
        dictionaries.append(links.links_dict)
    result: dict[str, float] = {}
    for d in dictionaries:
        for condition, frame in d.items():
            if not isinstance(frame, pd.DataFrame) or not {"source", "target"}.issubset(frame.columns):
                continue
            hit = frame[frame["target"].astype(str).str.upper() == anchor.upper()].copy()
            if hit.empty:
                continue
            coef_col = "coef_mean" if "coef_mean" in hit.columns else ("coef" if "coef" in hit.columns else None)
            if coef_col is None:
                continue
            hit[coef_col] = pd.to_numeric(hit[coef_col], errors="coerce")
            hit = hit.dropna(subset=[coef_col])
            for row in hit.itertuples(index=False):
                tf = str(getattr(row, "source"))
                coef = float(getattr(row, coef_col))
                old = result.get(tf.upper())
                if old is None or abs(coef) > abs(old):
                    result[tf.upper()] = coef
        if result:
            # Prefer filtered links; use unfiltered links only when no filtered hit exists.
            break
    return result

def candidate_support_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["celltype", "gene", "candidate_score", "candidate_supported"])
    x = pd.read_csv(path)
    if not {"celltype", "gene"}.issubset(x.columns):
        return pd.DataFrame(columns=["celltype", "gene", "candidate_score", "candidate_supported"])
    score_col = "key_driver_score" if "key_driver_score" in x.columns else (
        "dynamic_score" if "dynamic_score" in x.columns else None
    )
    x = x.copy()
    x["celltype"] = x["celltype"].astype(str).str.strip()
    x["gene"] = x["gene"].astype(str).str.strip()
    x["candidate_score"] = pd.to_numeric(x[score_col], errors="coerce") if score_col else 0.0
    x["candidate_score"] = x["candidate_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.sort_values("candidate_score", ascending=False).drop_duplicates(["celltype", "gene"])
    x["candidate_supported"] = True
    return x[["celltype", "gene", "candidate_score", "candidate_supported"]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_freeze_metadata(target_dir: Path, manifest_path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "selection_mode": args.selection_mode,
        "selection_dataset": "main" if args.selection_mode == "main_discovery_only" else "main+validation",
        "validation_used_for_selection": args.selection_mode != "main_discovery_only",
        "use_fitted_links": bool(args.use_fitted_links and not args.ignore_fitted_links),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (target_dir / "celloracle_tf_target_manifest.freeze.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.min_expressing_cells < 1:
        raise SystemExit("--min-expressing-cells must be >= 1.")
    pipeline_root, input_root, output_root = resolve_roots(args)
    support_dir = input_root / "supporting_tables"
    target_dir = output_root / "01_target_manifest"
    prepared_root = output_root / "03_celloracle_inputs"
    support_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = target_dir / "celloracle_tf_target_manifest.frozen.csv"
    if args.reuse_frozen:
        if not frozen_path.is_file():
            raise SystemExit(f"Frozen TF manifest not found: {frozen_path}")
        frozen = pd.read_csv(frozen_path)
        required = {"celltype", "anchor_gene", "tf_gene"}
        if not required.issubset(frozen.columns):
            raise SystemExit(f"Frozen manifest lacks required columns: {sorted(required)}")
        if args.celltype:
            frozen = frozen[frozen["celltype"].astype(str).isin(set(args.celltype))].copy()
        if frozen.empty:
            raise SystemExit("No frozen TF rows remained after filtering.")
        selected_manifest = frozen.copy()
        selected_path = target_dir / "celloracle_tf_target_manifest.csv"
        if not args.celltype:
            # Keep the manifest consumed by step 06 identical to the explicitly
            # reused frozen manifest. This prevents expected/output key drift.
            selected_manifest.to_csv(selected_path, index=False)
            print(f"Synchronized selected manifest from frozen manifest: {selected_path}")
        print(f"Reusing frozen MAIN-derived TF manifest: {frozen_path}")
        if not args.apply:
            print("Preview only. Add --apply to write targets_celloracle.csv files.")
            return
        written = 0
        application_rows = []
        prepared_root = output_root / "03_celloracle_inputs"
        for job_dir in sorted(prepared_root.glob("*__*")):
            if not job_dir.is_dir():
                continue
            parts = job_dir.name.split("__", 1)
            if len(parts) != 2:
                continue
            dataset, ct = parts
            rows = selected_manifest[selected_manifest["celltype"].astype(str) == ct].copy()
            if rows.empty:
                continue
            rows["dataset"] = dataset
            rows["gene"] = rows["tf_gene"]
            out_path = job_dir / "targets_celloracle.csv"
            rows.to_csv(out_path, index=False)
            application_rows.append({
                "job": job_dir.name,
                "dataset": dataset,
                "celltype": ct,
                "n_anchor_tf_pairs": int(rows[["celltype", "anchor_gene", "tf_gene"]].drop_duplicates().shape[0]),
                "target_table": str(out_path),
                "target_table_sha256": sha256_file(out_path),
            })
            written += 1
        pd.DataFrame(application_rows).to_csv(target_dir / "celloracle_target_application_audit.csv", index=False)
        print(f"Created targets_celloracle.csv in {written} prepared jobs from frozen manifest.")
        return

    base, tf_columns, tf_case = load_base_grn()
    universe_path = support_dir / "CellOracle_human_promoter_TF_universe.csv"
    pd.DataFrame({"gene": sorted(tf_columns)}).to_csv(universe_path, index=False)
    print(f"TF universe ({len(tf_columns)} genes): {universe_path}")
    if args.tf_universe_only:
        return

    manifest_path = target_dir / "target_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Original target manifest not found: {manifest_path}\nRun 02_build_target_manifest.R first.")
    if not prepared_root.is_dir():
        raise SystemExit(f"Prepared CellOracle input root not found: {prepared_root}\nRun 04_export_celloracle_inputs.R first.")

    anchors = pd.read_csv(manifest_path)
    if not {"celltype", "gene"}.issubset(anchors.columns):
        raise SystemExit("target_manifest.csv must contain celltype and gene columns.")
    anchors = anchors.copy()
    anchors["celltype"] = anchors["celltype"].astype(str).str.strip()
    anchors["anchor_gene"] = anchors["gene"].astype(str).str.strip()
    if args.celltype:
        anchors = anchors[anchors["celltype"].isin(set(args.celltype))]
    if anchors.empty:
        raise SystemExit("No anchors remained after filtering.")

    analysis_mode = os.getenv("VP_ANALYSIS_MODE", "pilot").lower()
    top_n = args.top_n if args.top_n is not None else (3 if analysis_mode == "pilot" else 5)
    if top_n < 1:
        raise SystemExit("--top-n must be >= 1.")

    candidate_path = support_dir / "07_final_ppi_informed_key_driver_candidates.csv"
    candidate_support = candidate_support_table(candidate_path)
    job_cache: dict[tuple[str, str], dict[str, object]] = {}
    for dataset in ("main", "validation"):
        for ct in sorted(anchors["celltype"].unique()):
            job_dir = prepared_root / f"{dataset}__{ct}"
            if job_dir.is_dir():
                job_cache[(dataset, ct)] = read_job(job_dir)

    review_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []

    for anchor_row in anchors.itertuples(index=False):
        ct = str(anchor_row.celltype)
        anchor = str(anchor_row.anchor_gene)
        main_job = job_cache.get(("main", ct))
        validation_job = job_cache.get(("validation", ct))
        if main_job is None or validation_job is None:
            print(f"WARNING: skipping {ct}/{anchor}; main or validation prepared job is missing.")
            continue

        main_genes = set(main_job["gene_index"].keys())  # type: ignore[index]
        validation_genes = set(validation_job["gene_index"].keys())  # type: ignore[index]
        if args.selection_mode == "main_discovery_only":
            tf_pool_upper = main_genes & set(tf_case.keys())
        else:
            tf_pool_upper = main_genes & validation_genes & set(tf_case.keys())

        direct = direct_prior_tfs(base, tf_columns, anchor)
        direct_upper = {x.upper() for x in direct}
        use_fitted_links = bool(args.use_fitted_links and not args.ignore_fitted_links)
        main_link_map = load_fitted_link_coefficients(
            output_root, "main", ct, anchor
        ) if use_fitted_links else {}
        validation_link_map = {}
        if args.selection_mode == "legacy_cross_dataset" and use_fitted_links:
            validation_link_map = load_fitted_link_coefficients(
                output_root, "validation", ct, anchor
            )

        if not tf_pool_upper:
            print(f"WARNING: no CellOracle TFs for {ct}/{anchor} in selection space.")
            continue
        pool = [tf_case[u] for u in sorted(tf_pool_upper)]
        main_stats = expression_stats(main_job, pool).add_prefix("main_").rename(columns={"main_gene": "gene"})
        validation_stats = expression_stats(validation_job, pool).add_prefix("validation_").rename(columns={"validation_gene": "gene"})
        table = main_stats.merge(validation_stats, on="gene", how="left")
        table.insert(0, "anchor_gene", anchor)
        table.insert(0, "celltype", ct)
        table["selection_mode"] = args.selection_mode
        table["selection_dataset"] = "main" if args.selection_mode == "main_discovery_only" else "main+validation"
        table["validation_used_for_selection"] = args.selection_mode != "main_discovery_only"
        table["direct_promoter_prior"] = table["gene"].str.upper().isin(direct_upper)
        table["main_fitted_link_coef"] = pd.to_numeric(
            table["gene"].str.upper().map(main_link_map), errors="coerce"
        )
        table["validation_fitted_link_coef"] = pd.to_numeric(
            table["gene"].str.upper().map(validation_link_map), errors="coerce"
        )
        table["main_fitted_link_supported"] = table["main_fitted_link_coef"].notna()
        table["validation_fitted_link_supported"] = table["validation_fitted_link_coef"].notna()
        table["fitted_link_supported"] = (
            table["main_fitted_link_supported"]
            if args.selection_mode == "main_discovery_only"
            else (table["main_fitted_link_supported"] | table["validation_fitted_link_supported"])
        )
        table["cross_dataset_fitted_link_supported"] = (
            table["main_fitted_link_supported"] & table["validation_fitted_link_supported"]
        )
        table["fitted_link_direction_concordant"] = np.where(
            table["cross_dataset_fitted_link_supported"],
            np.sign(table["main_fitted_link_coef"]) == np.sign(table["validation_fitted_link_coef"]),
            False,
        )
        if args.selection_mode == "main_discovery_only":
            table["fitted_link_abs_strength"] = table["main_fitted_link_coef"].abs().fillna(0.0)
            table["min_expr_cells"] = table[["main_Control_n_expr", "main_IPF_n_expr"]].min(axis=1)
            table["min_expr_fraction"] = table[["main_Control_expr_fraction", "main_IPF_expr_fraction"]].min(axis=1)
        else:
            table["fitted_link_abs_strength"] = table[[
                "main_fitted_link_coef", "validation_fitted_link_coef"
            ]].abs().max(axis=1, skipna=True).fillna(0.0)
            table["min_expr_cells"] = table[[
                "main_Control_n_expr", "main_IPF_n_expr",
                "validation_Control_n_expr", "validation_IPF_n_expr",
            ]].min(axis=1)
            table["min_expr_fraction"] = table[[
                "main_Control_expr_fraction", "main_IPF_expr_fraction",
                "validation_Control_expr_fraction", "validation_IPF_expr_fraction",
            ]].min(axis=1)
        table["expression_eligible"] = table["min_expr_cells"] >= args.min_expressing_cells
        table["validation_expression_audit_pass"] = (
            table[["validation_Control_n_expr", "validation_IPF_n_expr"]].min(axis=1).fillna(0)
            >= args.min_expressing_cells
        )

        support = candidate_support[candidate_support["celltype"] == ct].copy()
        support["gene_upper"] = support["gene"].str.upper()
        support = support.drop_duplicates("gene_upper")
        support_map = support.set_index("gene_upper") if not support.empty else pd.DataFrame()
        table["candidate_supported"] = table["gene"].str.upper().map(
            support_map["candidate_supported"] if not support.empty else {}
        ).fillna(False).astype(bool)
        table["candidate_score"] = pd.to_numeric(
            table["gene"].str.upper().map(support_map["candidate_score"] if not support.empty else {}),
            errors="coerce",
        ).fillna(0.0)
        if table["candidate_score"].nunique() > 1:
            table["candidate_score_percentile"] = table["candidate_score"].rank(pct=True)
        else:
            table["candidate_score_percentile"] = 0.0

        if args.selection_mode == "main_discovery_only":
            table["selection_score"] = (
                120.0 * table["main_fitted_link_supported"].astype(float)
                + 15.0 * np.log1p(table["main_fitted_link_coef"].abs().fillna(0.0))
                + 100.0 * table["direct_promoter_prior"].astype(float)
                + 20.0 * table["candidate_supported"].astype(float)
                + 10.0 * table["candidate_score_percentile"]
                + table["min_expr_fraction"]
            )
            table["selection_class"] = np.select(
                [table["main_fitted_link_supported"], table["direct_promoter_prior"], table["candidate_supported"]],
                ["main_fitted_link", "direct_promoter_prior", "candidate_TF_fallback"],
                default="expressed_TF_fallback",
            )
        else:
            table["selection_score"] = (
                160.0 * table["cross_dataset_fitted_link_supported"].astype(float)
                + 80.0 * table["fitted_link_supported"].astype(float)
                + 30.0 * table["fitted_link_direction_concordant"].astype(float)
                + 10.0 * np.log1p(table["fitted_link_abs_strength"])
                + 100.0 * table["direct_promoter_prior"].astype(float)
                + 20.0 * table["candidate_supported"].astype(float)
                + 10.0 * table["candidate_score_percentile"]
                + table["min_expr_fraction"]
            )
            table["selection_class"] = np.select(
                [table["cross_dataset_fitted_link_supported"], table["fitted_link_supported"], table["direct_promoter_prior"], table["candidate_supported"]],
                ["cross_dataset_fitted_link", "single_dataset_fitted_link", "direct_promoter_prior", "candidate_TF_fallback"],
                default="expressed_TF_fallback",
            )

        table["selected"] = False
        table["tf_rank_within_anchor"] = pd.NA
        eligible = table[
            table["expression_eligible"]
            & (table["fitted_link_supported"] | table["direct_promoter_prior"])
        ].copy()
        if eligible.empty and not args.no_fallback:
            eligible = table[table["expression_eligible"] & table["candidate_supported"]].copy()
        if eligible.empty and not args.no_fallback:
            eligible = table[table["expression_eligible"]].copy()
        eligible = eligible.sort_values(
            ["selection_score", "min_expr_cells", "gene"],
            ascending=[False, False, True],
            kind="stable",
        ).head(top_n)

        if eligible.empty:
            print(f"WARNING: no eligible upstream TF for {ct}/{anchor}.")
        else:
            selected_genes = eligible["gene"].tolist()
            table.loc[table["gene"].isin(selected_genes), "selected"] = True
            rank_map = {g: i + 1 for i, g in enumerate(selected_genes)}
            table.loc[table["selected"], "tf_rank_within_anchor"] = table.loc[table["selected"], "gene"].map(rank_map)
            selected = table[table["selected"]].copy()
            selected = selected.rename(columns={"gene": "tf_gene"})
            selected["target_id"] = selected["celltype"] + "__" + selected["anchor_gene"] + "__" + selected["tf_gene"]
            selected["selection_source"] = "MAIN_discovery_CellOracle_promoter_prior_expression_PPI"
            for col in ("biological_role", "axis_direction", "ko_condition", "oe_condition", "primary_test"):
                if hasattr(anchor_row, col):
                    selected[col] = getattr(anchor_row, col)
            selected_rows.append(selected)
            print(f"{ct} / {anchor}: " + ", ".join(selected_genes))

        review_rows.append(table)

    review = pd.concat(review_rows, ignore_index=True) if review_rows else pd.DataFrame()
    selected_manifest = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    review_path = target_dir / "celloracle_tf_candidate_review.csv"
    selected_path = target_dir / "celloracle_tf_target_manifest.csv"
    review.to_csv(review_path, index=False)
    selected_manifest.to_csv(selected_path, index=False)
    if selected_manifest.empty:
        raise SystemExit("No CellOracle TF targets were selected. Review celloracle_tf_candidate_review.csv.")
    if frozen_path.exists() and not args.overwrite_frozen:
        raise SystemExit(
            f"Frozen manifest already exists: {frozen_path}. "
            "Use --reuse-frozen or --overwrite-frozen explicitly."
        )
    selected_manifest.to_csv(frozen_path, index=False)
    write_freeze_metadata(target_dir, frozen_path, args)
    print(f"Candidate review: {review_path}")
    print(f"Selected TF manifest: {selected_path}")

    if not args.apply:
        print("Preview only. Re-run with --apply after reviewing the selected manifest.")
        return

    written = 0
    application_rows = []
    for (dataset, ct), _job in job_cache.items():
        rows = selected_manifest[selected_manifest["celltype"] == ct].copy()
        if rows.empty:
            continue
        rows.insert(0, "dataset", dataset)
        # Backward-compatible gene column: this is the TF being perturbed.
        rows["gene"] = rows["tf_gene"]
        preferred = [
            "target_id", "dataset", "celltype", "anchor_gene", "tf_gene", "gene",
            "tf_rank_within_anchor", "selection_class", "selection_score",
            "direct_promoter_prior", "fitted_link_supported",
            "cross_dataset_fitted_link_supported", "fitted_link_direction_concordant",
            "main_fitted_link_coef", "validation_fitted_link_coef",
            "candidate_supported", "candidate_score",
            "selection_mode", "selection_dataset", "validation_used_for_selection",
            "validation_expression_audit_pass",
            "min_expr_cells", "min_expr_fraction", "biological_role", "axis_direction",
            "ko_condition", "oe_condition", "primary_test", "selection_source",
        ]
        cols = [c for c in preferred if c in rows.columns] + [c for c in rows.columns if c not in preferred]
        out_path = prepared_root / f"{dataset}__{ct}" / "targets_celloracle.csv"
        rows[cols].to_csv(out_path, index=False)
        application_rows.append({
            "job": f"{dataset}__{ct}",
            "dataset": dataset,
            "celltype": ct,
            "n_anchor_tf_pairs": int(rows[["celltype", "anchor_gene", "tf_gene"]].drop_duplicates().shape[0]),
            "target_table": str(out_path),
            "target_table_sha256": sha256_file(out_path),
        })
        print(f"Wrote {out_path}")
        written += 1
    pd.DataFrame(application_rows).to_csv(target_dir / "celloracle_target_application_audit.csv", index=False)
    print(f"Created targets_celloracle.csv in {written} prepared jobs.")


if __name__ == "__main__":
    main()
