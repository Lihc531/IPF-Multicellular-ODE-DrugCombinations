#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_make_robustness_run_manifest_v1.py

Create a staged robustness-run manifest following the original project logic:

R axis / cell-state matrix
-> Neural ODE directionality / ablation
-> PPI-informed driver evidence
-> OT disease reversal
-> drug pairing / final screening
-> entity-QC-aware prioritization

This script only writes a manifest. It does not execute heavy analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def j(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def add_run(rows: List[Dict[str, Any]],
            phase_order: int,
            phase: str,
            run_id: str,
            script_language: str,
            source_script: str,
            overrides: Dict[str, Any],
            description: str,
            expected_outputs: List[str] | None = None,
            run_after: str | None = None,
            heavy: bool = True) -> None:
    rows.append({
        "phase_order": phase_order,
        "phase": phase,
        "run_id": run_id,
        "script_language": script_language,
        "source_script": source_script,
        "overrides_json": j(overrides),
        "description": description,
        "expected_outputs_json": j({"files": expected_outputs or []}),
        "run_after": run_after or "",
        "heavy": bool(heavy),
    })


def build_manifest() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Phase 1. Multicellular axis and axis_v2 robustness
    # ------------------------------------------------------------------
    add_run(rows, 1, "axis_v1", "AXISV1_baseline", "R",
            "13_build_multicellular_disease_progression_axis_nomultiplet.R",
            {"R_ASSIGN": {
                "w_sample": "0.50",
                "w_celltype": "0.25",
                "w_cluster": "0.25",
                "out_dir": 'file.path(project_dir, "results", "robustness", "axis_v1", "AXISV1_baseline")'
            }},
            "Baseline multicellular disease progression axis.")

    for run_id, ws, wc, wk, desc in [
        ("AXISV1_sample_heavy", "0.60", "0.20", "0.20", "Sample-composition-heavy axis."),
        ("AXISV1_celltype_heavy", "0.40", "0.40", "0.20", "Celltype-enrichment-heavy axis."),
        ("AXISV1_cluster_heavy", "0.40", "0.20", "0.40", "Cluster-enrichment-heavy axis."),
        ("AXISV1_equal_weight", "0.34", "0.33", "0.33", "Equal-weight axis components."),
    ]:
        add_run(rows, 1, "axis_v1", run_id, "R",
                "13_build_multicellular_disease_progression_axis_nomultiplet.R",
                {"R_ASSIGN": {
                    "w_sample": ws,
                    "w_celltype": wc,
                    "w_cluster": wk,
                    "out_dir": f'file.path(project_dir, "results", "robustness", "axis_v1", "{run_id}")'
                }},
                desc)

    add_run(rows, 2, "axis_v2", "AXISV2_baseline", "R",
            "14_add_fibrosis_module_pathway_scores_to_axis.R",
            {"R_ASSIGN": {
                "w_axis_v1": "0.50",
                "w_fibrosis": "0.30",
                "w_injury_inflammation": "0.20",
                "out_dir": 'file.path(project_dir, "results", "robustness", "axis_v2", "AXISV2_baseline")'
            }},
            "Baseline axis_v2 with multicellular axis, fibrosis, and injury/inflammation components.")

    for run_id, wa, wf, wi, desc in [
        ("AXISV2_axisv1_heavy", "0.65", "0.20", "0.15", "Axis-v1-heavy axis_v2."),
        ("AXISV2_fibrosis_heavy", "0.40", "0.45", "0.15", "Fibrosis-heavy axis_v2."),
        ("AXISV2_injury_heavy", "0.40", "0.20", "0.40", "Epithelial injury/inflammation-heavy axis_v2."),
        ("AXISV2_equal_weight", "0.34", "0.33", "0.33", "Equal-weight axis_v2 components."),
    ]:
        add_run(rows, 2, "axis_v2", run_id, "R",
                "14_add_fibrosis_module_pathway_scores_to_axis.R",
                {"R_ASSIGN": {
                    "w_axis_v1": wa,
                    "w_fibrosis": wf,
                    "w_injury_inflammation": wi,
                    "out_dir": f'file.path(project_dir, "results", "robustness", "axis_v2", "{run_id}")'
                }},
                desc)

    # ------------------------------------------------------------------
    # Phase 2. Cell-state feature matrix robustness
    # ------------------------------------------------------------------
    add_run(rows, 3, "cell_state_matrix", "CELLSTATE_baseline", "R",
            "18b_build_cell_state_feature_matrix_axis_v2.R",
            {"R_ASSIGN": {
                "min_cells_state_v1A": "20",
                "min_cells_state_v1B": "10",
                "n_axis_quantiles": "5",
                "out_dir": 'file.path(project_dir, "results", "robustness", "cell_state_matrix", "CELLSTATE_baseline")'
            }},
            "Baseline cell-state matrix construction.")

    for run_id, min_v1b, bins, desc in [
        ("CELLSTATE_min8_bins5", "8", "5", "Relaxed v1B state minimum cell count."),
        ("CELLSTATE_min15_bins5", "15", "5", "Stricter v1B state minimum cell count."),
        ("CELLSTATE_min10_bins4", "10", "4", "Fewer axis quantile bins."),
        ("CELLSTATE_min10_bins6", "10", "6", "More axis quantile bins."),
    ]:
        add_run(rows, 3, "cell_state_matrix", run_id, "R",
                "18b_build_cell_state_feature_matrix_axis_v2.R",
                {"R_ASSIGN": {
                    "min_cells_state_v1A": "20",
                    "min_cells_state_v1B": min_v1b,
                    "n_axis_quantiles": bins,
                    "out_dir": f'file.path(project_dir, "results", "robustness", "cell_state_matrix", "{run_id}")'
                }},
                desc)

    # ------------------------------------------------------------------
    # Phase 3. Neural ODE directionality robustness
    # Existing scripts already implement ablation batches; manifest registers them.
    # ------------------------------------------------------------------
    add_run(rows, 4, "neural_ode_ablation", "ODE_A_ablation_set", "PY",
            "22a_ablation1-5.py",
            {"PY_CONFIG": {
                "outdir": r"E:\PythonProject\results\robustness\neural_ode\ODE_A_ablation_set"
            }},
            "Existing first Neural ODE ablation batch: no-direction, no-warmup, no-stage-aware, etc.")

    add_run(rows, 4, "neural_ode_ablation", "ODE_B_ablation_set", "PY",
            "22b_ablation.py",
            {"PY_CONFIG": {
                "outdir": r"E:\PythonProject\results\robustness\neural_ode\ODE_B_ablation_set"
            }},
            "Existing second Neural ODE ablation batch: time-aware, solver, latent dimension, weak/strong direction.")

    # ------------------------------------------------------------------
    # Phase 4. PPI-informed driver robustness
    # ------------------------------------------------------------------
    add_run(rows, 5, "ppi_driver", "DRIVER_baseline", "PY",
            "23b_PPI-informed driver refinement.py",
            {"PY_CONFIG": {
                "outdir": r"E:\PythonProject\results\robustness\ppi_driver\DRIVER_baseline"
            }},
            "Baseline PPI-informed driver scoring.")

    for run_id, weights, desc in [
        ("DRIVER_no_ode_direction", {
            "dynamic_trend_score_norm": 1.20,
            "module_membership_score_norm": 1.00,
            "ppi_centrality_score_norm": 1.10,
            "celltype_specificity_score_norm": 0.80,
            "disease_stage_association_score_norm": 1.00,
            "neural_ode_direction_score_norm": 0.00,
            "patient_consistency_score_norm": 0.80,
            "druggability_score_norm": 0.60,
        }, "Driver scoring without Neural ODE directionality component."),
        ("DRIVER_no_ppi", {
            "dynamic_trend_score_norm": 1.20,
            "module_membership_score_norm": 1.00,
            "ppi_centrality_score_norm": 0.00,
            "celltype_specificity_score_norm": 0.80,
            "disease_stage_association_score_norm": 1.00,
            "neural_ode_direction_score_norm": 1.30,
            "patient_consistency_score_norm": 0.80,
            "druggability_score_norm": 0.60,
        }, "Driver scoring without PPI centrality component."),
        ("DRIVER_no_druggability", {
            "dynamic_trend_score_norm": 1.20,
            "module_membership_score_norm": 1.00,
            "ppi_centrality_score_norm": 1.10,
            "celltype_specificity_score_norm": 0.80,
            "disease_stage_association_score_norm": 1.00,
            "neural_ode_direction_score_norm": 1.30,
            "patient_consistency_score_norm": 0.80,
            "druggability_score_norm": 0.00,
        }, "Driver scoring without druggability component."),
        ("DRIVER_equal_weight", {
            "dynamic_trend_score_norm": 1.00,
            "module_membership_score_norm": 1.00,
            "ppi_centrality_score_norm": 1.00,
            "celltype_specificity_score_norm": 1.00,
            "disease_stage_association_score_norm": 1.00,
            "neural_ode_direction_score_norm": 1.00,
            "patient_consistency_score_norm": 1.00,
            "druggability_score_norm": 1.00,
        }, "Equal-weight driver components."),
    ]:
        add_run(rows, 5, "ppi_driver", run_id, "PY",
                "23b_PPI-informed driver refinement.py",
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ppi_driver\{run_id}",
                    "weights": weights
                }},
                desc)

    # ------------------------------------------------------------------
    # Phase 5. OT disease reversal robustness
    # ------------------------------------------------------------------
    base_ot_script = "29_ot_disease_reversal_score.py"
    add_run(rows, 6, "ot_reversal", "OT_baseline", "PY",
            base_ot_script,
            {"PY_CONFIG": {
                "outdir": r"E:\PythonProject\results\robustness\ot_reversal\OT_baseline",
                "sinkhorn_reg": 0.08,
                "random_seed": 20260708,
                "max_cells_per_group_for_ot": 220,
                "healthy_axis_quantile": 0.25,
                "disease_axis_quantile": 0.75,
                "max_feature_abs_corr": 0.95,
                "top_single_drugs_for_pairing": 80,
                "max_drugs_per_family_for_pairing": 2,
            }},
            "Baseline OT disease reversal run.")

    for reg in [0.05, 0.10, 0.12]:
        add_run(rows, 6, "ot_reversal", f"OT_sinkhorn_{str(reg).replace('.', '')}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_reversal\OT_sinkhorn_{str(reg).replace('.', '')}",
                    "sinkhorn_reg": reg
                }},
                f"Sinkhorn regularization sensitivity: {reg}")

    for seed in [20260709, 20260710, 20260711, 20260712]:
        add_run(rows, 6, "ot_reversal", f"OT_seed_{seed}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_reversal\OT_seed_{seed}",
                    "random_seed": seed
                }},
                f"Cell subsampling random seed sensitivity: {seed}")

    for n in [150, 300]:
        add_run(rows, 6, "ot_reversal", f"OT_maxcells_{n}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_reversal\OT_maxcells_{n}",
                    "max_cells_per_group_for_ot": n
                }},
                f"Maximum cells per OT group sensitivity: {n}")

    for lo, hi in [(0.20, 0.80), (0.30, 0.70)]:
        add_run(rows, 6, "ot_reversal", f"OT_quantile_{int(lo*100)}_{int(hi*100)}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_reversal\OT_quantile_{int(lo*100)}_{int(hi*100)}",
                    "healthy_axis_quantile": lo,
                    "disease_axis_quantile": hi
                }},
                f"Healthy/disease state quantile sensitivity: {lo}/{hi}")

    for corr in [0.90, 0.98]:
        add_run(rows, 6, "ot_reversal", f"OT_featurecorr_{int(corr*100)}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_reversal\OT_featurecorr_{int(corr*100)}",
                    "max_feature_abs_corr": corr
                }},
                f"OT feature correlation-pruning sensitivity: {corr}")

    for pool in [60, 100]:
        add_run(rows, 6, "ot_pairing", f"PAIR_pool_{pool}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_pairing\PAIR_pool_{pool}",
                    "top_single_drugs_for_pairing": pool
                }},
                f"Top single-drug pool size sensitivity: {pool}")

    for fam in [1, 3]:
        add_run(rows, 6, "ot_pairing", f"PAIR_familycap_{fam}", "PY",
                base_ot_script,
                {"PY_CONFIG": {
                    "outdir": rf"E:\PythonProject\results\robustness\ot_pairing\PAIR_familycap_{fam}",
                    "max_drugs_per_family_for_pairing": fam
                }},
                f"Maximum drugs per family sensitivity: {fam}")

    add_run(rows, 6, "ot_pairing", "PAIR_no_family_collapse", "PY",
            base_ot_script,
            {"PY_CONFIG": {
                "outdir": r"E:\PythonProject\results\robustness\ot_pairing\PAIR_no_family_collapse",
                "collapse_redundant_drug_families_for_pairing": False
            }},
            "Drug-family collapse disabled.")

    # ------------------------------------------------------------------
    # Phase 7. Final screening robustness.
    # This is computed by 34_final_screening_robustness_v1.py and does not require
    # this batch patcher. Register it in the manifest for completeness.
    # ------------------------------------------------------------------
    add_run(rows, 7, "final_screening", "FINAL_screening_sensitivity", "PY",
            "34_final_screening_robustness_v1.py",
            {"INPUT_TABLES": "Use baseline OT tables directory."},
            "Downstream final screening sensitivity: weights, table size, max drug appearances, QC penalties.",
            heavy=False)

    df = pd.DataFrame(rows).sort_values(["phase_order", "phase", "run_id"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create robustness run manifest.")
    parser.add_argument("--output-dir", default="robustness_manifest_v1",
                        help="Directory where manifest files are written.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_manifest()
    csv_path = out / "01_robustness_run_manifest.csv"
    json_path = out / "01_robustness_run_manifest.json"
    md_path = out / "01_robustness_run_manifest_README.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    md_path.write_text(
        "# Robustness run manifest\n\n"
        "Run order follows the biological pipeline:\n\n"
        "1. axis_v1\n2. axis_v2\n3. cell_state_matrix\n4. neural_ode_ablation\n"
        "5. ppi_driver\n6. ot_reversal / ot_pairing\n7. final_screening\n\n"
        "Use 32_run_patched_robustness_scripts_v1.py to create patched copies and optionally execute them.\n",
        encoding="utf-8"
    )

    print(f"[OK] Wrote manifest: {csv_path}")
    print(f"[OK] Wrote manifest JSON: {json_path}")
    print(f"[OK] Number of planned runs: {len(df)}")
    print(df.groupby(["phase_order", "phase"]).size().reset_index(name="n_runs").to_string(index=False))


if __name__ == "__main__":
    main()
