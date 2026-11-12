#!/usr/bin/env python3
"""Merge cell-type, signed-path and signature evidence into a drug-anchor master table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PyCharm 路径设置区：只修改这里即可
# =============================================================================
CELLTYPE_SUPPORT_FILE = Path(r"D:\YOUR_PROJECT\inputs\drug_target_celltype_support.csv")
SIGNED_PATHS_FILE = Path(r"D:\YOUR_PROJECT\outputs\37c_signed_paths\drug_target_tf_anchor_paths.csv")
SIGNATURE_SCORES_FILE = Path(r"D:\YOUR_PROJECT\outputs\37d_signature_reversal\drug_signature_anchor_reversal.csv")
OUTPUT_DIR = Path(r"D:\YOUR_PROJECT\outputs\37e_drug_anchor_master")

TIER_A_MIN = 0.62
TIER_B_MIN = 0.42
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Build single-drug mechanism bridge evidence tiers.")
    p.add_argument("--celltype-support", type=Path, default=CELLTYPE_SUPPORT_FILE)
    p.add_argument("--signed-paths", type=Path, default=SIGNED_PATHS_FILE)
    p.add_argument("--signature-scores", type=Path, default=SIGNATURE_SCORES_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--tier-a-min", type=float, default=TIER_A_MIN)
    p.add_argument("--tier-b-min", type=float, default=TIER_B_MIN)
    return p.parse_args()


def clip(s):
    return pd.to_numeric(s, errors="coerce").clip(0, 1)


def require_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.celltype_support, "Celltype support")
    require_file(args.signed_paths, "Signed paths")
    require_file(args.signature_scores, "Signature scores")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(args.celltype_support, low_memory=False)
    paths = pd.read_csv(args.signed_paths, low_memory=False)
    sig = pd.read_csv(args.signature_scores, low_memory=False)
    required_support = {"drug_id", "target_gene", "celltype"}
    required_paths = {"drug_id", "target_gene", "celltype", "anchor_gene"}
    required_sig = {"drug_id", "celltype", "anchor_gene"}
    if not required_support.issubset(support.columns):
        raise SystemExit(f"Celltype support lacks {sorted(required_support - set(support.columns))}")
    if not required_paths.issubset(paths.columns):
        raise SystemExit(f"Signed paths lacks {sorted(required_paths - set(paths.columns))}")
    if not required_sig.issubset(sig.columns):
        raise SystemExit(f"Signature score table lacks {sorted(required_sig - set(sig.columns))}")

    path_score_col = "path_bridge_score" if "path_bridge_score" in paths.columns else None
    if path_score_col:
        paths = paths.sort_values(path_score_col, ascending=False)
    best_paths = paths.drop_duplicates(["drug_id", "target_gene", "celltype", "anchor_gene"], keep="first")
    merged = best_paths.merge(
        support,
        on=["drug_id", "target_gene", "celltype"], how="left", suffixes=("", "_support"), validate="many_to_one"
    ).merge(
        sig,
        on=["drug_id", "celltype", "anchor_gene"], how="left", suffixes=("", "_signature"), validate="many_to_one"
    )

    cell = clip(merged.get("celltype_support_score", np.nan))
    path = clip(merged.get("path_bridge_score", np.nan))
    sig_score = clip(merged.get("signature_bridge_score", np.nan))
    celloracle = pd.to_numeric(merged.get("celloracle_score", np.nan), errors="coerce")
    celloracle = celloracle.rank(pct=True) if celloracle.notna().any() else pd.Series(np.nan, index=merged.index)
    direction = merged.get("direction_compatible", False)
    if not isinstance(direction, pd.Series):
        direction = pd.Series(direction, index=merged.index)
    direction = direction.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    validated = merged.get("celloracle_direction_validated", False)
    if not isinstance(validated, pd.Series):
        validated = pd.Series(validated, index=merged.index)
    validated = validated.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)

    evidence = pd.DataFrame({
        "celltype": cell,
        "signed_path": path,
        "signature": sig_score,
        "celloracle": celloracle,
        "direction": direction,
        "celloracle_validated": validated,
    })
    weights = pd.Series({"celltype": 0.18, "signed_path": 0.22, "signature": 0.24,
                         "celloracle": 0.16, "direction": 0.12, "celloracle_validated": 0.08})
    avail = evidence.notna().mul(weights, axis=1).sum(axis=1)
    raw = evidence.fillna(0).mul(weights, axis=1).sum(axis=1) / avail.replace(0, np.nan)
    coverage = avail / weights.sum()
    merged["bridge_evidence_coverage"] = coverage
    merged["drug_anchor_bridge_score_raw"] = raw
    merged["drug_anchor_bridge_score"] = raw * np.sqrt(coverage)

    cell_supported = merged.get("celltype_supported", False)
    if not isinstance(cell_supported, pd.Series):
        cell_supported = pd.Series(cell_supported, index=merged.index)
    cell_supported = cell_supported.astype(str).str.lower().isin(["true", "1", "yes"])
    signature_available = pd.to_numeric(merged.get("signature_bridge_score", np.nan), errors="coerce").notna()
    direction_ok = direction.astype(bool)
    co_ok = validated.astype(bool)
    score = merged["drug_anchor_bridge_score"]
    merged["evidence_tier"] = np.select(
        [
            (score >= args.tier_a_min) & cell_supported & direction_ok & co_ok & signature_available,
            (score >= args.tier_b_min) & cell_supported & (direction_ok | signature_available),
        ],
        ["Tier_A_directional_mechanism_bridge", "Tier_B_phenotypic_or_partial_bridge"],
        default="Tier_C_target_or_structure_only",
    )
    merged["experimental_validation_claim_allowed"] = False
    merged["interpretation"] = np.select(
        [merged["evidence_tier"].str.startswith("Tier_A"), merged["evidence_tier"].str.startswith("Tier_B")],
        ["multi_layer_computational_mechanism_support", "partial_computational_reversal_support"],
        default="insufficient_anchor_bridge_support",
    )
    merged = merged.sort_values(["evidence_tier", "drug_anchor_bridge_score"], ascending=[True, False])
    merged.to_csv(args.output_dir / "drug_anchor_bridge_master.csv", index=False)

    drug_summary = merged.groupby("drug_id", as_index=False).agg(
        drug_anchor_bridge_score=("drug_anchor_bridge_score", "max"),
        mean_bridge_score=("drug_anchor_bridge_score", "mean"),
        n_anchors_supported=("anchor_gene", "nunique"),
        n_celltypes_supported=("celltype", "nunique"),
        n_targets_supported=("target_gene", "nunique"),
        n_tier_a=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier_A").sum())),
        n_tier_b=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier_B").sum())),
        evidence_coverage=("bridge_evidence_coverage", "mean"),
    )
    best_tier = merged.groupby("drug_id")["evidence_tier"].first().rename("best_evidence_tier")
    anchors = merged.groupby("drug_id")["anchor_gene"].agg(lambda s: ";".join(sorted(set(map(str, s))))).rename("validated_anchor_union")
    cells = merged.groupby("drug_id")["celltype"].agg(lambda s: ";".join(sorted(set(map(str, s))))).rename("bridge_celltype_union")
    drug_summary = drug_summary.merge(best_tier, on="drug_id").merge(anchors, on="drug_id").merge(cells, on="drug_id")
    drug_summary.to_csv(args.output_dir / "drug_anchor_bridge_summary.csv", index=False)

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"n_context_rows": int(len(merged)), "n_drugs": int(merged["drug_id"].nunique()),
                   "tier_a_min": args.tier_a_min, "tier_b_min": args.tier_b_min}, fh, indent=2)
    print(f"Drug-anchor bridge master written to {args.output_dir}")


if __name__ == "__main__":
    main()
