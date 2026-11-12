#!/usr/bin/env python3
"""Standardize gene-level drug perturbation signatures for mechanism bridging.

Accepted input is a long CSV/TSV containing at least a drug identifier/name, a
gene symbol and a signed effect (logFC, z-score, signed_effect or equivalent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# =============================================================================
# PyCharm 路径设置区：只修改这里即可
# =============================================================================
INPUT_FILE = Path(r"D:\YOUR_PROJECT\inputs\gene_level_drug_signatures.csv")
OUTPUT_DIR = Path(r"D:\YOUR_PROJECT\outputs\36c_gene_level_signatures")

# 本步骤参数
INPUT_SEPARATOR = None       # None=按扩展名自动判断；也可设为 "," 或 "\t"
MIN_ABS_EFFECT = 0.0
AGGREGATE_METHOD = "median"  # "none"、"median" 或 "mean"

# 若列名无法自动识别，可在此精确指定；通常保持 None。
DRUG_COLUMN = None
DRUG_ID_COLUMN = None
GENE_COLUMN = None
EFFECT_COLUMN = None
SIGNATURE_ID_COLUMN = None
CELL_LINE_COLUMN = None
DOSE_COLUMN = None
TIME_COLUMN = None
QUALITY_COLUMN = None
# =============================================================================

ALIASES = {
    "drug": ["drug", "drug_name", "perturbagen", "pert_name", "compound", "compound_name"],
    "drug_id": ["drug_id", "pert_id", "compound_id", "cid"],
    "gene": ["gene", "gene_symbol", "symbol", "gene_name"],
    "effect": ["effect", "signed_effect", "logfc", "log2fc", "zscore", "z_score", "score", "lfc"],
    "signature_id": ["signature_id", "perturbation_signature_id", "sig_id", "signature", "id"],
    "cell_line": ["cell_line", "cell", "cell_id", "celltype", "cell_type"],
    "dose": ["dose", "pert_dose", "concentration"],
    "time": ["time", "timepoint", "timepoint_hours", "pert_time", "duration"],
    "quality": ["quality", "quality_score", "evidence_weight", "qc_score", "distil_cc_q75", "tas"],
}


def first_match(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    return next((lookup[x] for x in candidates if x in lookup), None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize signed gene-level drug signatures.")
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--sep", default=INPUT_SEPARATOR)
    defaults = {
        "drug": DRUG_COLUMN, "drug_id": DRUG_ID_COLUMN, "gene": GENE_COLUMN,
        "effect": EFFECT_COLUMN, "signature_id": SIGNATURE_ID_COLUMN,
        "cell_line": CELL_LINE_COLUMN, "dose": DOSE_COLUMN,
        "time": TIME_COLUMN, "quality": QUALITY_COLUMN,
    }
    for key in ("drug", "drug_id", "gene", "effect", "signature_id", "cell_line", "dose", "time", "quality"):
        p.add_argument(f"--{key.replace('_','-')}-col", dest=f"{key}_col", default=defaults[key])
    p.add_argument("--min-abs-effect", type=float, default=MIN_ABS_EFFECT)
    p.add_argument("--aggregate", choices=["none", "median", "mean"], default=AGGREGATE_METHOD)
    return p.parse_args()


def read_table(path: Path, sep: str | None) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_table(args.input, args.sep)
    if raw.empty:
        raise SystemExit("Input signature table is empty.")

    mapping: dict[str, str | None] = {}
    for key, aliases in ALIASES.items():
        explicit = getattr(args, f"{key}_col")
        mapping[key] = explicit if explicit else first_match(raw.columns, aliases)
    if mapping["drug"] is None and mapping["drug_id"] is None:
        raise SystemExit("Could not detect a drug name or drug ID column.")
    if mapping["gene"] is None or mapping["effect"] is None:
        raise SystemExit("Could not detect gene and signed-effect columns.")

    out = pd.DataFrame(index=raw.index)
    out["drug"] = raw[mapping["drug"]].astype(str).str.strip() if mapping["drug"] else raw[mapping["drug_id"]].astype(str).str.strip()
    out["drug_id"] = raw[mapping["drug_id"]].astype(str).str.strip() if mapping["drug_id"] else out["drug"]
    out["gene"] = raw[mapping["gene"]].astype(str).str.strip().str.upper()
    out["effect"] = pd.to_numeric(raw[mapping["effect"]], errors="coerce")
    for key in ("signature_id", "cell_line", "dose", "time"):
        out[key] = raw[mapping[key]].astype(str).str.strip() if mapping[key] else ""
    out["quality_score"] = pd.to_numeric(raw[mapping["quality"]], errors="coerce") if mapping["quality"] else np.nan
    out = out.replace({"": np.nan}).dropna(subset=["drug", "gene", "effect"])
    out = out[np.isfinite(out["effect"]) & (out["effect"].abs() >= args.min_abs_effect)].copy()
    out["effect_direction"] = np.sign(out["effect"]).astype(int)
    out["source_file"] = str(args.input.resolve())

    out.to_csv(args.output_dir / "drug_gene_level_signatures_long.csv.gz", index=False, compression="gzip")

    group_cols = ["drug_id", "drug", "gene"]
    if args.aggregate == "none":
        aggregated = out.copy()
    else:
        fn = "median" if args.aggregate == "median" else "mean"
        aggregated = out.groupby(group_cols, as_index=False).agg(
            effect=("effect", fn),
            effect_abs_median=("effect", lambda s: float(np.nanmedian(np.abs(s)))),
            n_signatures=("effect", "size"),
            direction_consistency=("effect", lambda s: float(abs(np.nanmean(np.sign(s))))),
            quality_score=("quality_score", "mean"),
        )
        aggregated["effect_direction"] = np.sign(aggregated["effect"]).astype(int)
    aggregated.to_csv(args.output_dir / "drug_gene_level_signatures_aggregated.csv", index=False)

    qc = out.groupby(["drug_id", "drug"], as_index=False).agg(
        n_rows=("gene", "size"), n_genes=("gene", "nunique"),
        n_signatures=("signature_id", lambda s: int(pd.Series(s).replace("", np.nan).nunique())),
        median_abs_effect=("effect", lambda s: float(np.nanmedian(np.abs(s)))),
        mean_quality=("quality_score", "mean"),
    )
    qc.to_csv(args.output_dir / "drug_signature_gene_level_qc.csv", index=False)
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"input": str(args.input.resolve()), "column_mapping": mapping,
                   "n_rows": int(len(out)), "n_drugs": int(out["drug_id"].nunique()),
                   "aggregation": args.aggregate}, fh, ensure_ascii=False, indent=2)
    print(f"Wrote standardized signatures to {args.output_dir}")


if __name__ == "__main__":
    main()
