#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_build_ode_robustness_weights_v1.py

Purpose
-------
Convert completed Neural ODE robustness outputs into reusable reliability
weights and caution flags for downstream analyses:

    ODE / directionality
    -> PPI-informed driver
    -> OT disease reversal
    -> drug pairing
    -> final candidate screening

This script does NOT rerun Neural ODE. It reads the existing seed-stability
summary and, optionally, threshold perturbation runs, then writes compact
celltype/cluster/stage/disease reliability tables.

Expected upstream inputs
------------------------
1) Required:
   SEED_STABILITY_SUMMARY_v2 directory containing:
     09_celltype_stability_summary.csv
     11_cluster_stability_summary.csv
     05_stage_stability_summary.csv
     07_disease_stability_summary.csv
     12_celltype_seed_sensitivity_flags.csv
     13_cluster_seed_sensitivity_flags.csv
     15_overall_seed_stability_assessment.json

2) Optional:
   Neural ODE threshold run directories:
     ODE_CELLSTATE_min10_baseline
     ODE_CELLSTATE_min8
     ODE_CELLSTATE_min15

The optional threshold runs allow this script to add threshold-sensitivity
metrics from 14/15/16/17 directionality tables.

Outputs
-------
ODE_ROBUSTNESS_WEIGHTS/
  01_ode_overall_status.json
  02_celltype_ode_reliability_weights.csv
  03_cluster_ode_reliability_weights.csv
  04_stage_ode_reliability_weights.csv
  05_disease_ode_reliability_weights.csv
  06_threshold_celltype_stability_summary.csv
  07_threshold_cluster_stability_summary.csv
  08_threshold_stage_stability_summary.csv
  09_threshold_disease_stability_summary.csv
  10_ode_reliability_readme.md
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

CONFIG: Dict[str, Any] = {
    # Directory produced by the corrected seed summary script.
    "seed_summary_dir":
        r"E:\PythonProject\results\robustness_runs_v3\05_neural_ode_seed_robustness\SEED_STABILITY_SUMMARY_v2",

    # Optional threshold perturbation run directories. Set missing paths to "" or
    # leave them unchanged if unavailable; the script will skip missing paths.
    "threshold_runs": {
        "min10_baseline":
            r"E:\PythonProject\results\robustness_runs_v3\04_neural_ode_cellstate_threshold\ODE_CELLSTATE_min10_baseline",
        "min8":
            r"E:\PythonProject\results\robustness_runs_v3\04_neural_ode_cellstate_threshold\ODE_CELLSTATE_min8",
        "min15":
            r"E:\PythonProject\results\robustness_runs_v3\04_neural_ode_cellstate_threshold\ODE_CELLSTATE_min15",
    },

    "output_dir":
        r"E:\PythonProject\results\robustness_runs_v3\06_downstream_robustness\ODE_ROBUSTNESS_WEIGHTS",

    # error: stop if output dir is non-empty
    # timestamp: create a timestamped sibling dir
    # overwrite_summary_only: overwrite output files inside this summary folder
    "existing_output_policy": "error",

    "thresholds": {
        "celltype_seed_flag_pp": 15.0,
        "cluster_seed_flag_pp": 20.0,
        "stage_seed_flag_pp": 15.0,
        "disease_seed_flag_pp": 15.0,
        "celltype_threshold_flag_pp": 15.0,
        "cluster_threshold_flag_pp": 20.0,
        "stage_threshold_flag_pp": 15.0,
        "disease_threshold_flag_pp": 15.0,
        "min_celltype_states_for_strong_claim": 30,
        "min_cluster_states_for_strong_claim": 20,
        "small_group_weight_floor": 0.25,
        "unstable_weight_floor": 0.35,
    },

    # Metrics to use when computing directionality stability.
    "directionality_metrics": [
        "pct_moves_toward_axis",
        "pct_moves_toward_fibrosis",
        "pct_moves_toward_terminal",
    ],
}


DIRECTIONALITY_FILES = {
    "celltype": "14_directionality_by_celltype.csv",
    "stage": "15_directionality_by_axis_stage.csv",
    "disease": "16_directionality_by_disease.csv",
    "cluster": "17_directionality_by_celltype_cluster.csv",
}


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for x in path.rglob("*") if x.is_file())


def choose_output_dir(base: Path, policy: str) -> Path:
    policy = policy.strip().lower()
    if not base.exists() or count_files(base) == 0:
        base.mkdir(parents=True, exist_ok=True)
        return base
    if policy == "timestamp":
        target = base.parent / f"{base.name}_{timestamp_tag()}"
        target.mkdir(parents=True, exist_ok=False)
        return target
    if policy == "overwrite_summary_only":
        base.mkdir(parents=True, exist_ok=True)
        return base
    raise FileExistsError(
        f"Output directory already exists and is non-empty; nothing was overwritten:\n{base}"
    )


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)


def json_read_optional(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_pct_range(value: Any) -> float:
    """Return a percentage-point range. Values <= 1 are interpreted as fractions."""
    if pd.isna(value):
        return float("nan")
    x = float(value)
    return x * 100.0 if abs(x) <= 1.000001 else x


def stability_weight_from_pp_range(pp_range: float, flag_threshold: float, floor: float) -> float:
    if not math.isfinite(pp_range):
        return 1.0
    if flag_threshold <= 0:
        return 1.0
    # Smoothly decays from 1.0 at range=0 to floor at range>=2*threshold.
    scaled = max(0.0, 1.0 - pp_range / (2.0 * flag_threshold))
    return max(float(floor), float(scaled))


def infer_group_columns(table_kind: str, df: pd.DataFrame) -> List[str]:
    candidates = {
        "celltype": [["celltype_marker_based"], ["celltype"]],
        "cluster": [
            ["celltype_marker_based", "cluster_default"],
            ["celltype", "cluster"],
            ["celltype_marker_based", "cluster"],
        ],
        "stage": [["axis_v2_stage"], ["stage"]],
        "disease": [["Disease_Identity"], ["disease"]],
    }
    for cols in candidates.get(table_kind, []):
        if all(c in df.columns for c in cols):
            return cols
    # fallback: non-numeric columns
    nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return nonnum[:2]


def summarize_threshold_runs(table_kind: str, threshold_runs: Mapping[str, str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    filename = DIRECTIONALITY_FILES[table_kind]
    for run_name, run_dir_str in threshold_runs.items():
        if not run_dir_str:
            continue
        run_dir = Path(run_dir_str)
        candidates = [
            run_dir / "outputs" / "tables" / filename,
            run_dir / "tables" / filename,
            run_dir / filename,
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        df = pd.read_csv(path, low_memory=False)
        df.insert(0, "threshold_run", run_name)
        frames.append(df)

    if len(frames) < 2:
        return pd.DataFrame()

    group_cols = infer_group_columns(table_kind, frames[0])
    common = set.intersection(*(set(f.columns) for f in frames))
    group_cols = [c for c in group_cols if c in common]
    metric_cols = [
        c for c in CONFIG["directionality_metrics"]
        if c in common and all(pd.api.types.is_numeric_dtype(f[c]) for f in frames)
    ]
    if not group_cols or not metric_cols:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    records: List[Dict[str, Any]] = []
    grouper: Any = group_cols if len(group_cols) > 1 else group_cols[0]
    for keys, part in long_df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        n_states = pd.to_numeric(part["n_states"], errors="coerce") if "n_states" in part else pd.Series(dtype=float)
        for metric in metric_cols:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            raw_range = float(values.max() - values.min())
            pp_range = normalize_pct_range(raw_range)
            records.append({
                **base,
                "metric": metric,
                "n_threshold_runs": int(values.size),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "range": raw_range,
                "range_percentage_points": pp_range,
                "mean_n_states": float(n_states.mean()) if not n_states.empty else float("nan"),
            })
    return pd.DataFrame(records)


def max_pp_for_metrics(summary: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=list(group_cols) + ["max_range_percentage_points"])
    metric_subset = summary[summary["metric"].isin(CONFIG["directionality_metrics"])].copy()
    if metric_subset.empty:
        return pd.DataFrame(columns=list(group_cols) + ["max_range_percentage_points"])
    metric_subset["range_percentage_points"] = metric_subset["range_percentage_points"].map(normalize_pct_range)
    out = (
        metric_subset
        .groupby(list(group_cols), dropna=False)["range_percentage_points"]
        .max()
        .reset_index()
        .rename(columns={"range_percentage_points": "max_range_percentage_points"})
    )
    return out


def first_numeric_metric(summary: pd.DataFrame, group_cols: Sequence[str], metric: str, out_name: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=list(group_cols) + [out_name])
    part = summary[summary["metric"] == metric].copy()
    if part.empty:
        return pd.DataFrame(columns=list(group_cols) + [out_name])
    # Use mean across seeds/thresholds as the representative value.
    cols = list(group_cols) + ["mean"]
    return part[cols].rename(columns={"mean": out_name})


def build_reliability_table(
    table_kind: str,
    seed_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    group_cols: Sequence[str],
    seed_flag_pp: float,
    threshold_flag_pp: float,
    min_states: Optional[int],
    unstable_floor: float,
    small_group_floor: float,
) -> pd.DataFrame:
    seed_pp = max_pp_for_metrics(seed_summary, group_cols).rename(
        columns={"max_range_percentage_points": "max_seed_range_pp"}
    )
    thr_pp = max_pp_for_metrics(threshold_summary, group_cols).rename(
        columns={"max_range_percentage_points": "max_threshold_range_pp"}
    )

    # Representative state counts.
    n_states_seed = first_numeric_metric(seed_summary, group_cols, "n_states", "mean_n_states_seed")
    n_states_thr = first_numeric_metric(threshold_summary, group_cols, "n_states", "mean_n_states_threshold")

    # Start from all groups seen in either seed or threshold summary.
    keys = []
    for df in [seed_pp, thr_pp, n_states_seed, n_states_thr]:
        if not df.empty:
            keys.append(df[list(group_cols)])
    if not keys:
        return pd.DataFrame()
    base = pd.concat(keys, ignore_index=True).drop_duplicates()

    out = base.merge(seed_pp, on=list(group_cols), how="left")
    out = out.merge(thr_pp, on=list(group_cols), how="left")
    out = out.merge(n_states_seed, on=list(group_cols), how="left")
    out = out.merge(n_states_thr, on=list(group_cols), how="left")

    out["max_seed_range_pp"] = out["max_seed_range_pp"].fillna(0.0)
    out["max_threshold_range_pp"] = out["max_threshold_range_pp"].fillna(0.0)
    out["mean_n_states"] = out[["mean_n_states_seed", "mean_n_states_threshold"]].mean(axis=1, skipna=True)

    out["seed_unstable"] = out["max_seed_range_pp"] >= float(seed_flag_pp)
    out["threshold_unstable"] = out["max_threshold_range_pp"] >= float(threshold_flag_pp)
    out["small_group"] = False
    if min_states is not None:
        out["small_group"] = pd.to_numeric(out["mean_n_states"], errors="coerce").fillna(10**9) < float(min_states)

    seed_w = out["max_seed_range_pp"].map(lambda x: stability_weight_from_pp_range(float(x), seed_flag_pp, unstable_floor))
    thr_w = out["max_threshold_range_pp"].map(lambda x: stability_weight_from_pp_range(float(x), threshold_flag_pp, unstable_floor))
    size_w = np.where(out["small_group"], float(small_group_floor), 1.0)

    out["ode_seed_stability_weight"] = seed_w
    out["ode_threshold_stability_weight"] = thr_w
    out["ode_size_weight"] = size_w
    out["ode_reliability_weight"] = (out["ode_seed_stability_weight"] * out["ode_threshold_stability_weight"] * out["ode_size_weight"]).clip(0.05, 1.0)

    def tier(row: pd.Series) -> str:
        if bool(row["small_group"]):
            return "LOW_SMALL_GROUP"
        if bool(row["seed_unstable"]) and bool(row["threshold_unstable"]):
            return "LOW_SEED_AND_THRESHOLD_SENSITIVE"
        if bool(row["seed_unstable"]):
            return "CAUTION_SEED_SENSITIVE"
        if bool(row["threshold_unstable"]):
            return "CAUTION_THRESHOLD_SENSITIVE"
        return "HIGH_RELIABILITY"

    out["ode_reliability_tier"] = out.apply(tier, axis=1)
    out = out.sort_values(["ode_reliability_weight"] + list(group_cols), ascending=[False] + [True] * len(group_cols))
    return out


# =============================================================================
# 3. MAIN
# =============================================================================

def main() -> None:
    seed_dir = Path(CONFIG["seed_summary_dir"])
    output_dir = choose_output_dir(Path(CONFIG["output_dir"]), str(CONFIG["existing_output_policy"]))
    thresholds = CONFIG["thresholds"]

    overall = json_read_optional(seed_dir / "15_overall_seed_stability_assessment.json")
    json_dump(output_dir / "01_ode_overall_status.json", {
        "source_seed_summary_dir": str(seed_dir),
        "source_threshold_runs": CONFIG["threshold_runs"],
        "seed_summary_verdict": overall.get("verdict"),
        "seed_summary_observed": overall.get("observed", {}),
        "seed_summary_checks": overall.get("checks", {}),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "Reliability weights are intended for downstream evidence weighting, not for removing rows automatically.",
    })

    # Required seed-stability summaries.
    celltype_seed = read_csv_required(seed_dir / "09_celltype_stability_summary.csv")
    cluster_seed = read_csv_required(seed_dir / "11_cluster_stability_summary.csv")
    stage_seed = read_csv_required(seed_dir / "05_stage_stability_summary.csv")
    disease_seed = read_csv_required(seed_dir / "07_disease_stability_summary.csv")

    # Optional threshold summaries.
    threshold_summaries: Dict[str, pd.DataFrame] = {}
    for kind in ["celltype", "cluster", "stage", "disease"]:
        threshold_summaries[kind] = summarize_threshold_runs(kind, CONFIG["threshold_runs"])

    threshold_summaries["celltype"].to_csv(output_dir / "06_threshold_celltype_stability_summary.csv", index=False, encoding="utf-8-sig")
    threshold_summaries["cluster"].to_csv(output_dir / "07_threshold_cluster_stability_summary.csv", index=False, encoding="utf-8-sig")
    threshold_summaries["stage"].to_csv(output_dir / "08_threshold_stage_stability_summary.csv", index=False, encoding="utf-8-sig")
    threshold_summaries["disease"].to_csv(output_dir / "09_threshold_disease_stability_summary.csv", index=False, encoding="utf-8-sig")

    celltype_weights = build_reliability_table(
        "celltype",
        celltype_seed,
        threshold_summaries["celltype"],
        ["celltype_marker_based"],
        float(thresholds["celltype_seed_flag_pp"]),
        float(thresholds["celltype_threshold_flag_pp"]),
        int(thresholds["min_celltype_states_for_strong_claim"]),
        float(thresholds["unstable_weight_floor"]),
        float(thresholds["small_group_weight_floor"]),
    )
    cluster_weights = build_reliability_table(
        "cluster",
        cluster_seed,
        threshold_summaries["cluster"],
        ["celltype_marker_based", "cluster_default"],
        float(thresholds["cluster_seed_flag_pp"]),
        float(thresholds["cluster_threshold_flag_pp"]),
        int(thresholds["min_cluster_states_for_strong_claim"]),
        float(thresholds["unstable_weight_floor"]),
        float(thresholds["small_group_weight_floor"]),
    )
    stage_group = ["axis_v2_stage"] if "axis_v2_stage" in stage_seed.columns else infer_group_columns("stage", stage_seed)
    disease_group = ["Disease_Identity"] if "Disease_Identity" in disease_seed.columns else infer_group_columns("disease", disease_seed)

    stage_weights = build_reliability_table(
        "stage",
        stage_seed,
        threshold_summaries["stage"],
        stage_group,
        float(thresholds["stage_seed_flag_pp"]),
        float(thresholds["stage_threshold_flag_pp"]),
        None,
        float(thresholds["unstable_weight_floor"]),
        float(thresholds["small_group_weight_floor"]),
    )
    disease_weights = build_reliability_table(
        "disease",
        disease_seed,
        threshold_summaries["disease"],
        disease_group,
        float(thresholds["disease_seed_flag_pp"]),
        float(thresholds["disease_threshold_flag_pp"]),
        None,
        float(thresholds["unstable_weight_floor"]),
        float(thresholds["small_group_weight_floor"]),
    )

    celltype_weights.to_csv(output_dir / "02_celltype_ode_reliability_weights.csv", index=False, encoding="utf-8-sig")
    cluster_weights.to_csv(output_dir / "03_cluster_ode_reliability_weights.csv", index=False, encoding="utf-8-sig")
    stage_weights.to_csv(output_dir / "04_stage_ode_reliability_weights.csv", index=False, encoding="utf-8-sig")
    disease_weights.to_csv(output_dir / "05_disease_ode_reliability_weights.csv", index=False, encoding="utf-8-sig")

    md = f"""# ODE reliability weights for downstream robustness

Created: {datetime.now().astimezone().isoformat(timespec='seconds')}

## Seed summary verdict

`{overall.get('verdict', 'NA')}`

## How to use these tables

- `ode_reliability_weight` is a multiplicative evidence weight between 0 and 1.
- `HIGH_RELIABILITY` groups can support primary downstream interpretation.
- `CAUTION_*` groups can be retained but should be labelled as sensitivity-aware evidence.
- `LOW_SMALL_GROUP` groups should not drive strong biological conclusions alone.

## Files

- `02_celltype_ode_reliability_weights.csv`
- `03_cluster_ode_reliability_weights.csv`
- `04_stage_ode_reliability_weights.csv`
- `05_disease_ode_reliability_weights.csv`

These are intended to be joined into PPI driver, OT reversal, drug pairing, and final candidate screening outputs.
"""
    (output_dir / "10_ode_reliability_readme.md").write_text(md, encoding="utf-8")

    print("ODE reliability weights completed.")
    print("Output:", output_dir)
    for name, df in [
        ("celltype", celltype_weights),
        ("cluster", cluster_weights),
        ("stage", stage_weights),
        ("disease", disease_weights),
    ]:
        print(f"{name}: {df.shape[0]} rows")
        if "ode_reliability_tier" in df:
            print(df["ode_reliability_tier"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
