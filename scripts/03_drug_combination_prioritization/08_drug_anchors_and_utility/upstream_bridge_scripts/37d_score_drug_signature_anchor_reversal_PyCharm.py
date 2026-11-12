#!/usr/bin/env python3
"""Score drug signatures against disease-axis genes, CellOracle TFs and anchors.

This script uses signed gene-level perturbation effects. The TF score reflects
TF-gene expression direction unless an explicit TF-activity table is supplied;
it should therefore be interpreted as computational direction support, not TF
activity measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PyCharm 路径设置区：只修改这里即可
# =============================================================================
SIGNATURES_FILE = Path(r"D:\YOUR_PROJECT\outputs\36c_gene_level_signatures\drug_gene_level_signatures_aggregated.csv")
AXIS_WEIGHTS_FILE = Path(r"D:\YOUR_PROJECT\inputs\axis_weights.csv")
CELLORACLE_SUMMARY_FILE = Path(r"D:\YOUR_PROJECT\inputs\celloracle_tf_summary.csv")
OUTPUT_DIR = Path(r"D:\YOUR_PROJECT\outputs\37d_signature_reversal")

# 可选输入；没有时设为 None
BRIDGE_PATHS_FILE = None  # 例如 Path(r"D:\YOUR_PROJECT\outputs\37c_signed_paths\drug_target_tf_anchor_paths.csv")
ANCHOR_MODULES_FILE = None
TF_ACTIVITY_FILE = None
# =============================================================================


def pick(columns, aliases):
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    return next((lookup[x] for x in aliases if x in lookup), None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    aa, bb = a[ok], b[ok]
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 0 else np.nan


def parse_args():
    p = argparse.ArgumentParser(description="Score drug signature reversal of disease anchors and TF programs.")
    p.add_argument("--signatures", type=Path, default=SIGNATURES_FILE)
    p.add_argument("--axis-weights", type=Path, default=AXIS_WEIGHTS_FILE)
    p.add_argument("--celloracle-summary", type=Path, default=CELLORACLE_SUMMARY_FILE)
    p.add_argument("--bridge-paths", type=Path, default=BRIDGE_PATHS_FILE)
    p.add_argument("--anchor-modules", type=Path, default=ANCHOR_MODULES_FILE)
    p.add_argument("--tf-activity", type=Path, default=TF_ACTIVITY_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.signatures, "Signature table")
    require_file(args.axis_weights, "Axis weights")
    require_file(args.celloracle_summary, "CellOracle summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sig = pd.read_csv(args.signatures, low_memory=False)
    axis = pd.read_csv(args.axis_weights, low_memory=False)
    co = pd.read_csv(args.celloracle_summary, low_memory=False)
    gene_col = pick(sig.columns, ["gene", "gene_symbol"]); effect_col = pick(sig.columns, ["effect", "zscore", "logfc"])
    drug_col = pick(sig.columns, ["drug_id", "drug", "drug_name"])
    if not gene_col or not effect_col or not drug_col:
        raise SystemExit("Signature table requires drug, gene and effect columns.")
    sig = sig.copy(); sig["drug_id"] = sig[drug_col].astype(str); sig["gene"] = sig[gene_col].astype(str).str.upper(); sig["effect"] = pd.to_numeric(sig[effect_col], errors="coerce")
    sig = sig.dropna(subset=["effect"])
    axis_gene = pick(axis.columns, ["gene", "gene_symbol"]); axis_weight = pick(axis.columns, ["spearman_axis", "axis_weight", "weight"])
    if not axis_gene or not axis_weight or "celltype" not in axis.columns:
        raise SystemExit("Axis table requires celltype, gene and spearman_axis/weight.")
    axis = axis.copy(); axis["gene"] = axis[axis_gene].astype(str).str.upper(); axis["axis_weight"] = pd.to_numeric(axis[axis_weight], errors="coerce")
    tf_col = pick(co.columns, ["tf_gene", "gene"]); anchor_col = pick(co.columns, ["anchor_gene", "anchor"])
    if not tf_col or not anchor_col or "celltype" not in co.columns:
        raise SystemExit("CellOracle summary requires celltype, tf_gene and anchor_gene.")
    co = co.copy(); co["tf_gene"] = co[tf_col].astype(str).str.upper(); co["anchor_gene"] = co[anchor_col].astype(str).str.upper()
    role_sign = pd.Series(1, index=co.index, dtype=int)
    if "disease_role_sign" in co.columns:
        role_sign = np.sign(pd.to_numeric(co["disease_role_sign"], errors="coerce").fillna(1)).astype(int)
    elif "biological_role" in co.columns:
        role_sign = np.where(co["biological_role"].astype(str).str.lower().str.contains("decreas|protect|healthy"), -1, 1)
    co["desired_tf_effect_sign"] = -role_sign
    co_score_col = pick(co.columns, ["primary_reversal_score", "tf_cross_dataset_evidence_score"])
    co["tf_weight"] = pd.to_numeric(co[co_score_col], errors="coerce").rank(pct=True).fillna(0.5) if co_score_col else 0.5

    modules = pd.DataFrame()
    if args.anchor_modules and args.anchor_modules.exists():
        modules = pd.read_csv(args.anchor_modules, low_memory=False)
        req = {"celltype", "anchor_gene", "gene"}
        if not req.issubset(modules.columns):
            raise SystemExit("Anchor module table requires celltype, anchor_gene and gene.")
        modules["gene"] = modules["gene"].astype(str).str.upper(); modules["anchor_gene"] = modules["anchor_gene"].astype(str).str.upper()
        modules["weight"] = pd.to_numeric(modules.get("weight", 1.0), errors="coerce").fillna(1.0)

    tf_activity = pd.DataFrame()
    if args.tf_activity and args.tf_activity.exists():
        tf_activity = pd.read_csv(args.tf_activity, low_memory=False)
        tf_activity["tf_gene"] = tf_activity["tf_gene"].astype(str).str.upper()
        tf_activity["effect"] = pd.to_numeric(tf_activity["effect"], errors="coerce")

    rows = []
    for drug_id, ds in sig.groupby("drug_id"):
        effect_map = ds.groupby("gene")["effect"].median()
        for celltype, aw in axis.groupby("celltype"):
            merged = aw[["gene", "axis_weight"]].copy(); merged["effect"] = merged["gene"].map(effect_map)
            axis_rev = -cosine(merged["effect"].to_numpy(float), merged["axis_weight"].to_numpy(float))
            tf_rows = co[co["celltype"].astype(str) == str(celltype)]
            for anchor, tfa in tf_rows.groupby("anchor_gene"):
                desired = tfa.set_index("tf_gene")["desired_tf_effect_sign"]
                weights = tfa.set_index("tf_gene")["tf_weight"]
                if not tf_activity.empty:
                    ta = tf_activity[(tf_activity["drug_id"].astype(str) == str(drug_id)) & (tf_activity["celltype"].astype(str) == str(celltype))]
                    tf_effect = ta.groupby("tf_gene")["effect"].median()
                    tf_source = "explicit_tf_activity"
                else:
                    tf_effect = effect_map
                    tf_source = "tf_gene_expression_fallback"
                common_tf = desired.index.intersection(tf_effect.index)
                if len(common_tf):
                    match = (np.sign(tf_effect.loc[common_tf].to_numpy(float)) == desired.loc[common_tf].to_numpy(int)).astype(float)
                    tf_score = float(np.average(match, weights=weights.loc[common_tf].to_numpy(float))) if weights.loc[common_tf].sum() > 0 else float(match.mean())
                else:
                    tf_score = np.nan
                anchor_effect = effect_map.get(anchor, np.nan)
                anchor_gene_reversal = float(np.sign(anchor_effect) == -1) if np.isfinite(anchor_effect) else np.nan
                module_score = np.nan; module_n = 0
                if not modules.empty:
                    mm = modules[(modules["celltype"].astype(str) == str(celltype)) & (modules["anchor_gene"] == anchor)].copy()
                    mm["effect"] = mm["gene"].map(effect_map); mm = mm.dropna(subset=["effect"])
                    module_n = len(mm)
                    if module_n:
                        module_score = -cosine(mm["effect"].to_numpy(float), mm["weight"].to_numpy(float))
                vals = [axis_rev, tf_score, module_score, anchor_gene_reversal]
                coverage = float(np.mean([np.isfinite(v) for v in vals]))
                composite = float(np.nanmean(vals)) if any(np.isfinite(v) for v in vals) else np.nan
                rows.append({
                    "drug_id": drug_id, "celltype": celltype, "anchor_gene": anchor,
                    "axis_reversal_score": axis_rev, "tf_direction_score": tf_score,
                    "tf_score_source": tf_source, "anchor_gene_reversal": anchor_gene_reversal,
                    "anchor_module_reversal_score": module_score, "anchor_module_n_genes": module_n,
                    "signature_bridge_score_raw": composite, "signature_coverage": coverage,
                    "signature_bridge_score": composite * np.sqrt(coverage) if np.isfinite(composite) else np.nan,
                    "evidence_type": "public_or_external_signature_computational_support",
                })
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "drug_signature_anchor_reversal.csv", index=False)
    summary = out.groupby("drug_id", as_index=False).agg(
        best_signature_bridge_score=("signature_bridge_score", "max"),
        mean_signature_bridge_score=("signature_bridge_score", "mean"),
        n_anchor_contexts=("anchor_gene", "size"), mean_signature_coverage=("signature_coverage", "mean"),
    )
    summary.to_csv(args.output_dir / "drug_signature_reversal_summary.csv", index=False)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"n_rows": int(len(out)), "n_drugs": int(out["drug_id"].nunique()),
                   "tf_activity_used": bool(not tf_activity.empty)}, fh, indent=2)
    print(f"Drug signature reversal scores written to {args.output_dir}")


if __name__ == "__main__":
    main()
