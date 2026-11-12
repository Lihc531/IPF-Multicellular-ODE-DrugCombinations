#!/usr/bin/env python3
"""Build signed drug-target-TF-anchor paths for each cell type.

This independent PyCharm version keeps every input and output path in the
settings block below. It accepts command-line overrides but does not require a
shared project configuration file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# =============================================================================
# PyCharm 路径设置区：只修改这里即可
# =============================================================================
TARGET_CELLTYPE_FILE = Path(r"D:\YOUR_PROJECT\inputs\drug_target_celltype_support.csv")
SIGNED_NETWORK_FILE = Path(r"D:\YOUR_PROJECT\inputs\signed_causal_network.csv")
CELLORACLE_SUMMARY_FILE = Path(r"D:\YOUR_PROJECT\inputs\celloracle_tf_summary.csv")
OUTPUT_DIR = Path(r"D:\YOUR_PROJECT\outputs\37c_signed_paths")

MAX_PATH_LENGTH = 3
MIN_CELLTYPE_SCORE = 0.35
MAX_PATHS_PER_PAIR = 20
# =============================================================================


def pick(columns, aliases):
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    return next((lookup[x] for x in aliases if x in lookup), None)


def parse_sign(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    positive = {"1", "+1", "+", "positive", "activate", "activates", "activation", "agonist", "up", "increase"}
    negative = {"-1", "-", "negative", "inhibit", "inhibits", "inhibition", "antagonist", "down", "decrease"}
    if text in positive:
        return 1.0
    if text in negative:
        return -1.0
    try:
        number = float(text)
        return float(np.sign(number)) if number != 0 else np.nan
    except ValueError:
        return np.nan


def action_sign(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if any(x in text for x in ["inhib", "antagon", "block", "degrad", "suppress", "negative"]):
        return -1.0
    if any(x in text for x in ["agon", "activ", "stimulat", "positive"]):
        return 1.0
    return parse_sign(value)


def parse_args():
    p = argparse.ArgumentParser(description="Build signed drug-target-TF-anchor paths.")
    p.add_argument("--target-celltype", type=Path, default=TARGET_CELLTYPE_FILE)
    p.add_argument("--signed-network", type=Path, default=SIGNED_NETWORK_FILE)
    p.add_argument("--celloracle-summary", type=Path, default=CELLORACLE_SUMMARY_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--max-path-length", type=int, default=MAX_PATH_LENGTH)
    p.add_argument("--min-celltype-score", type=float, default=MIN_CELLTYPE_SCORE)
    p.add_argument("--max-paths-per-pair", type=int, default=MAX_PATHS_PER_PAIR)
    return p.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def main():
    args = parse_args()
    require_file(args.target_celltype, "Target-celltype table")
    require_file(args.signed_network, "Signed network")
    require_file(args.celloracle_summary, "CellOracle summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    support = pd.read_csv(args.target_celltype, low_memory=False)
    net = pd.read_csv(args.signed_network, low_memory=False)
    co = pd.read_csv(args.celloracle_summary, low_memory=False)

    required_support = {"drug_id", "target_gene", "celltype"}
    if not required_support.issubset(support.columns):
        raise SystemExit(f"Target-celltype table lacks {sorted(required_support - set(support.columns))}")

    source_col = pick(net.columns, ["source", "from", "upstream", "regulator", "source_gene"])
    target_col = pick(net.columns, ["target", "to", "downstream", "target_gene"])
    sign_col = pick(net.columns, ["sign", "effect", "direction", "interaction_sign"])
    conf_col = pick(net.columns, ["confidence", "score", "weight", "evidence_score"])
    if not source_col or not target_col or not sign_col:
        raise SystemExit("Signed network requires source, target and sign columns.")

    tf_col = pick(co.columns, ["tf_gene", "gene"])
    anchor_col = pick(co.columns, ["anchor_gene", "anchor"])
    if not tf_col or not anchor_col or "celltype" not in co.columns:
        raise SystemExit("CellOracle summary requires celltype, tf_gene and anchor_gene.")

    support = support.copy()
    support["drug_id"] = support["drug_id"].astype(str)
    support["target_gene"] = support["target_gene"].astype(str).str.upper()
    cell_score_col = pick(support.columns, ["celltype_support_score", "support_score", "score"])
    support["celltype_support_score"] = pd.to_numeric(support[cell_score_col], errors="coerce") if cell_score_col else 1.0
    support = support[support["celltype_support_score"].fillna(0) >= args.min_celltype_score].copy()
    action_col = pick(support.columns, ["action", "mechanism", "moa", "target_action", "effect"])
    support["drug_target_effect_sign"] = support[action_col].map(action_sign) if action_col else np.nan

    net = net.copy()
    net["source_gene"] = net[source_col].astype(str).str.upper()
    net["target_gene"] = net[target_col].astype(str).str.upper()
    net["edge_sign"] = net[sign_col].map(parse_sign)
    net["edge_confidence"] = pd.to_numeric(net[conf_col], errors="coerce").clip(0, 1) if conf_col else 1.0
    net = net.dropna(subset=["edge_sign"])

    graph = nx.DiGraph()
    for row in net.itertuples(index=False):
        old = graph.get_edge_data(row.source_gene, row.target_gene)
        confidence = float(row.edge_confidence) if np.isfinite(row.edge_confidence) else 0.5
        if old is None or confidence > old.get("confidence", -1):
            graph.add_edge(row.source_gene, row.target_gene, sign=int(row.edge_sign), confidence=confidence)

    co = co.copy()
    co["tf_gene"] = co[tf_col].astype(str).str.upper()
    co["anchor_gene"] = co[anchor_col].astype(str).str.upper()
    co_score_col = pick(co.columns, ["primary_reversal_score", "tf_cross_dataset_evidence_score", "celloracle_score"])
    co["celloracle_score"] = pd.to_numeric(co[co_score_col], errors="coerce") if co_score_col else np.nan
    if co["celloracle_score"].notna().any():
        co["celloracle_score"] = co["celloracle_score"].rank(pct=True)
    else:
        co["celloracle_score"] = 0.5

    role_sign = pd.Series(1, index=co.index, dtype=int)
    if "disease_role_sign" in co.columns:
        role_sign = np.sign(pd.to_numeric(co["disease_role_sign"], errors="coerce").fillna(1)).astype(int)
    elif "biological_role" in co.columns:
        role_sign = np.where(co["biological_role"].astype(str).str.lower().str.contains("decreas|protect|healthy"), -1, 1)
    co["desired_tf_effect_sign"] = -role_sign

    validation_col = pick(co.columns, ["primary_direction_pass_both", "celloracle_direction_validated", "direction_validated"])
    if validation_col:
        co["celloracle_direction_validated"] = co[validation_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        co["celloracle_direction_validated"] = False

    rows = []
    for sr in support.itertuples(index=False):
        drug_id = str(sr.drug_id)
        target_gene = str(sr.target_gene)
        celltype = str(sr.celltype)
        target_effect_sign = float(sr.drug_target_effect_sign) if np.isfinite(sr.drug_target_effect_sign) else np.nan
        tf_rows = co[co["celltype"].astype(str) == celltype]
        for cr in tf_rows.itertuples(index=False):
            tf_gene = str(cr.tf_gene)
            anchor_gene = str(cr.anchor_gene)
            if target_gene not in graph or tf_gene not in graph:
                continue
            try:
                paths = nx.all_simple_paths(graph, target_gene, tf_gene, cutoff=args.max_path_length)
                count = 0
                for path in paths:
                    if len(path) < 2:
                        continue
                    edge_signs = []
                    confidences = []
                    for u, v in zip(path[:-1], path[1:]):
                        edge = graph[u][v]
                        edge_signs.append(int(edge["sign"]))
                        confidences.append(float(edge["confidence"]))
                    path_sign = int(np.prod(edge_signs))
                    propagated_tf_sign = int(np.sign(target_effect_sign * path_sign)) if np.isfinite(target_effect_sign) else np.nan
                    desired_sign = int(cr.desired_tf_effect_sign)
                    direction_compatible = bool(propagated_tf_sign == desired_sign) if np.isfinite(propagated_tf_sign) else False
                    path_confidence = float(np.prod(confidences) ** (1 / len(confidences))) if confidences else np.nan
                    cell_score = float(sr.celltype_support_score) if np.isfinite(sr.celltype_support_score) else np.nan
                    co_score = float(cr.celloracle_score) if np.isfinite(cr.celloracle_score) else np.nan
                    components = [cell_score, path_confidence, co_score, float(direction_compatible)]
                    finite = [x for x in components if np.isfinite(x)]
                    raw_score = float(np.mean(finite)) if finite else np.nan
                    coverage = len(finite) / len(components)
                    bridge_score = raw_score * np.sqrt(coverage) if np.isfinite(raw_score) else np.nan
                    rows.append({
                        "drug_id": drug_id,
                        "target_gene": target_gene,
                        "celltype": celltype,
                        "tf_gene": tf_gene,
                        "anchor_gene": anchor_gene,
                        "path_nodes": ";".join(path),
                        "path_length": len(path) - 1,
                        "path_sign": path_sign,
                        "drug_target_effect_sign": target_effect_sign,
                        "propagated_tf_effect_sign": propagated_tf_sign,
                        "desired_tf_effect_sign": desired_sign,
                        "direction_compatible": direction_compatible,
                        "celloracle_direction_validated": bool(cr.celloracle_direction_validated),
                        "celltype_support_score": cell_score,
                        "path_confidence": path_confidence,
                        "celloracle_score": co_score,
                        "path_bridge_score_raw": raw_score,
                        "path_bridge_coverage": coverage,
                        "path_bridge_score": bridge_score,
                    })
                    count += 1
                    if count >= args.max_paths_per_pair:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("path_bridge_score", ascending=False)
    out.to_csv(args.output_dir / "drug_target_tf_anchor_paths.csv", index=False)

    if out.empty:
        summary = pd.DataFrame(columns=["drug_id", "n_paths", "n_targets", "n_tfs", "n_anchors", "best_path_bridge_score"])
    else:
        summary = out.groupby("drug_id", as_index=False).agg(
            n_paths=("path_nodes", "size"),
            n_targets=("target_gene", "nunique"),
            n_tfs=("tf_gene", "nunique"),
            n_anchors=("anchor_gene", "nunique"),
            best_path_bridge_score=("path_bridge_score", "max"),
        )
    summary.to_csv(args.output_dir / "drug_target_tf_anchor_path_summary.csv", index=False)

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "n_paths": int(len(out)),
            "n_drugs": int(out["drug_id"].nunique()) if not out.empty else 0,
            "max_path_length": args.max_path_length,
            "min_celltype_score": args.min_celltype_score,
            "max_paths_per_pair": args.max_paths_per_pair,
        }, fh, ensure_ascii=False, indent=2)
    print(f"Signed target-TF-anchor paths written to {args.output_dir}")


if __name__ == "__main__":
    main()
