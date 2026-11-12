#!/usr/bin/env python3
"""Quantify a pair's incremental mechanistic coverage over its best single member.

This is not a Bliss/Loewe/ZIP/HSA synergy analysis. Without observed dual-drug
responses it estimates incremental anchor/cell-type/module coverage and member
balance only.
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
PAIRS_FILE = Path(r"D:\YOUR_PROJECT\inputs\drug_pairs.csv")
DRUG_BRIDGE_SUMMARY_FILE = Path(r"D:\YOUR_PROJECT\outputs\37e_drug_anchor_master\drug_anchor_bridge_summary.csv")
OUTPUT_DIR = Path(r"D:\YOUR_PROJECT\outputs\38b_pair_increment")
# =============================================================================


def pick(cols, aliases):
    lookup = {str(c).lower(): str(c) for c in cols}
    return next((lookup[x] for x in aliases if x in lookup), None)


def split_set(x) -> set[str]:
    if pd.isna(x):
        return set()
    return {y.strip() for y in str(x).replace(",", ";").split(";") if y.strip()}


def parse_args():
    p = argparse.ArgumentParser(description="Compare predicted pair coverage with best single drug.")
    p.add_argument("--pairs", type=Path, default=PAIRS_FILE)
    p.add_argument("--drug-bridge-summary", type=Path, default=DRUG_BRIDGE_SUMMARY_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.pairs, "Drug pair table")
    require_file(args.drug_bridge_summary, "Drug bridge summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs, low_memory=False)
    drugs = pd.read_csv(args.drug_bridge_summary, low_memory=False)
    pair_id = pick(pairs.columns, ["pair_id", "combination_id"])
    a_col = pick(pairs.columns, ["drug_a_id", "drug1_id", "drug_a", "drug1", "compound_a"])
    b_col = pick(pairs.columns, ["drug_b_id", "drug2_id", "drug_b", "drug2", "compound_b"])
    d_col = pick(drugs.columns, ["drug_id", "drug", "drug_name"])
    if not a_col or not b_col or not d_col:
        raise SystemExit("Could not detect pair member columns or drug_id.")
    if not pair_id:
        pairs["pair_id"] = pairs[a_col].astype(str) + "__" + pairs[b_col].astype(str)
        pair_id = "pair_id"
    score_col = pick(drugs.columns, ["drug_anchor_bridge_score", "best_bridge_score", "final_bridge_score"])
    if not score_col:
        raise SystemExit("Drug bridge summary lacks a bridge score.")
    lookup = drugs.set_index(drugs[d_col].astype(str))

    rows = []
    for r in pairs.itertuples(index=False):
        rd = r._asdict(); a = str(rd[a_col]); b = str(rd[b_col])
        da = lookup.loc[a] if a in lookup.index else None
        db = lookup.loc[b] if b in lookup.index else None
        if isinstance(da, pd.DataFrame): da = da.iloc[0]
        if isinstance(db, pd.DataFrame): db = db.iloc[0]
        sa = float(pd.to_numeric(da[score_col], errors="coerce")) if da is not None else np.nan
        sb = float(pd.to_numeric(db[score_col], errors="coerce")) if db is not None else np.nan
        aa = split_set(da.get("validated_anchor_union", "")) if da is not None else set()
        ab = split_set(db.get("validated_anchor_union", "")) if db is not None else set()
        ca = split_set(da.get("bridge_celltype_union", "")) if da is not None else set()
        cb = split_set(db.get("bridge_celltype_union", "")) if db is not None else set()
        anchor_union = aa | ab; cell_union = ca | cb
        best_anchor = max(len(aa), len(ab)); best_cell = max(len(ca), len(cb))
        inc_anchor = len(anchor_union) - best_anchor; inc_cell = len(cell_union) - best_cell
        weaker = np.nanmin([sa, sb]) if np.isfinite(sa) or np.isfinite(sb) else np.nan
        stronger = np.nanmax([sa, sb]) if np.isfinite(sa) or np.isfinite(sb) else np.nan
        complement = 0.5 * min(1.0, inc_anchor / 2.0) + 0.5 * min(1.0, inc_cell / 2.0)
        member_balance = weaker if np.isfinite(weaker) else 0.0
        pair_score = 0.45 * complement + 0.35 * member_balance + 0.20 * (stronger if np.isfinite(stronger) else 0.0)
        direction_conflict = int(bool(aa & ab) and sa < 0.4 and sb < 0.4) if np.isfinite(sa) and np.isfinite(sb) else 0
        pair_score = max(0.0, pair_score - 0.20 * direction_conflict)
        rows.append({
            "pair_id": rd[pair_id], "drug_a_id": a, "drug_b_id": b,
            "drug_a_bridge_score": sa, "drug_b_bridge_score": sb,
            "best_single_bridge_score": stronger, "weaker_member_bridge_score": weaker,
            "validated_anchor_union": ";".join(sorted(anchor_union)),
            "bridge_celltype_union": ";".join(sorted(cell_union)),
            "n_validated_anchors_covered": len(anchor_union), "n_celltypes_with_bridge": len(cell_union),
            "incremental_anchor_coverage": inc_anchor, "incremental_celltype_coverage": inc_cell,
            "anchor_complementarity_gain": complement,
            "bridge_direction_conflict_count": direction_conflict,
            "pair_increment_score": pair_score,
            "increment_evidence_type": "mechanistic_coverage_increment_not_pharmacologic_synergy",
            "pair_minus_best_single": np.nan,
        })
    out = pd.DataFrame(rows).sort_values("pair_increment_score", ascending=False)
    out.to_csv(args.output_dir / "pair_increment_over_best_single.csv", index=False)
    out.to_csv(args.output_dir / "pair_vs_best_single.csv", index=False)
    out.to_csv(args.output_dir / "pair_mechanistic_complementarity.csv", index=False)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"n_pairs": int(len(out)), "interpretation": "mechanistic complementarity, not synergy"}, fh, indent=2)
    print(f"Pair increment table written to {args.output_dir}")


if __name__ == "__main__":
    main()
