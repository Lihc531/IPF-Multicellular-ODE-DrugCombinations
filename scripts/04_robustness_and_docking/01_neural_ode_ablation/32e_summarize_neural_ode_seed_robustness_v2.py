#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
37_summarize_neural_ode_seed_robustness_v2.py

Re-summarize Neural ODE random-seed robustness.

v2 fix:
- state-level pairwise consistency no longer joins only by `cell_state_id`,
  because a small number of rows can share the same `cell_state_id`.
- It uses a composite state key:
  cell_state_id + sample_id + Disease_Identity + celltype_marker_based +
  cluster_default + axis_v2_stage + n_cells + fraction_in_sample +
  mean_axis_v2_score.
- If the composite key is unavailable, it falls back to a deterministic row key
  and reports that fallback in the output.

Run after seed456 and seed789 have completed. seed123 is read from the existing
min10 baseline directory.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


CONFIG: Dict[str, Any] = {
    "seed_123_run_dir":
        r"E:\PythonProject\results\robustness_runs_v3\04_neural_ode_cellstate_threshold\ODE_CELLSTATE_min10_baseline",

    "seed_robustness_root":
        r"E:\PythonProject\results\robustness_runs_v3\05_neural_ode_seed_robustness",

    "summary_output_dir":
        r"E:\PythonProject\results\robustness_runs_v3\05_neural_ode_seed_robustness\SEED_STABILITY_SUMMARY_v2",

    # error | timestamp
    "existing_output_policy": "error",

    "thresholds": {
        "max_val_r2_range": 0.05,
        "max_relative_val_mse_range": 0.15,
        "max_stage_percentage_point_range": 15.0,
        "max_disease_percentage_point_range": 15.0,
        "celltype_flag_percentage_point_range": 15.0,
        "cluster_flag_percentage_point_range": 20.0,
        "min_state_projection_spearman": 0.70,
        "min_state_projection_sign_agreement": 0.70,
    },
}

RUNS: Dict[str, Dict[str, Any]] = {
    "seed_123": {"seed": 123, "absolute_run_dir": CONFIG["seed_123_run_dir"]},
    "seed_456": {"seed": 456, "relative_run_dir": "ODE_SEED_456"},
    "seed_789": {"seed": 789, "relative_run_dir": "ODE_SEED_789"},
}

TABLES = {
    "metrics": "11_final_metrics.json",
    "velocity": "13_direction_constrained_state_velocity_scores.csv",
    "celltype": "14_directionality_by_celltype.csv",
    "stage": "15_directionality_by_axis_stage.csv",
    "disease": "16_directionality_by_disease.csv",
    "cluster": "17_directionality_by_celltype_cluster.csv",
    "validation": "18_validation_pair_predictions.csv",
}

GROUP_COLUMN_CANDIDATES: Dict[str, List[List[str]]] = {
    "stage": [["axis_v2_stage"], ["stage"]],
    "disease": [["Disease_Identity"], ["disease"]],
    "celltype": [["celltype_marker_based"], ["celltype"]],
    "cluster": [
        ["celltype_marker_based", "cluster_default"],
        ["celltype", "cluster"],
        ["celltype_marker_based", "cluster"],
    ],
}

COMPOSITE_STATE_KEY = [
    "cell_state_id",
    "sample_id",
    "Disease_Identity",
    "celltype_marker_based",
    "cluster_default",
    "axis_v2_stage",
    "n_cells",
    "fraction_in_sample",
    "mean_axis_v2_score",
]


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for x in path.rglob("*") if x.is_file())


def choose_output_dir(path: Path) -> Path:
    if not path.exists() or count_files(path) == 0:
        path.mkdir(parents=True, exist_ok=True)
        return path
    policy = str(CONFIG["existing_output_policy"]).lower().strip()
    if policy == "timestamp":
        out = path.parent / f"{path.name}_{timestamp_tag()}"
        out.mkdir(parents=True, exist_ok=False)
        return out
    raise FileExistsError(
        f"Summary output directory exists and is non-empty; nothing was overwritten:\n{path}"
    )


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, list):
        return {prefix: json.dumps(obj, ensure_ascii=False)}
    return {prefix: obj}


def resolve_run_dir(spec: Mapping[str, Any]) -> Path:
    if "absolute_run_dir" in spec:
        return Path(str(spec["absolute_run_dir"]))
    return Path(str(CONFIG["seed_robustness_root"])) / str(spec["relative_run_dir"])


def resolve_tables_dir(run_dir: Path) -> Path:
    for candidate in [run_dir / "outputs" / "tables", run_dir / "tables"]:
        if candidate.is_dir():
            return candidate
    raise NotADirectoryError(f"No tables directory found under: {run_dir}")


def prepare_run_data() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for run_name, spec in RUNS.items():
        run_dir = resolve_run_dir(spec)
        tables_dir = resolve_tables_dir(run_dir)
        data: Dict[str, Any] = {
            "seed": int(spec["seed"]),
            "run_dir": run_dir,
            "tables_dir": tables_dir,
        }
        missing = []
        for key, filename in TABLES.items():
            p = tables_dir / filename
            data[key] = p
            if not p.is_file() or p.stat().st_size == 0:
                missing.append(filename)
        if missing:
            raise FileNotFoundError(f"{run_name} missing required tables: {missing}")
        out[run_name] = data
    return out


def combine_csv(run_data: Mapping[str, Dict[str, Any]], table_key: str) -> pd.DataFrame:
    frames = []
    for run_name, data in run_data.items():
        df = pd.read_csv(data[table_key], low_memory=False)
        df.insert(0, "seed", int(data["seed"]))
        df.insert(0, "run_name", run_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def percentage_point_range(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    r = float(x.max() - x.min())
    return r * 100.0 if float(x.abs().max()) <= 1.000001 else r


def summarize_grouped_numeric(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in set(group_cols) | {"seed"}
    ]
    records: List[Dict[str, Any]] = []
    grouper = list(group_cols) if len(group_cols) > 1 else group_cols[0]
    for keys, part in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in numeric_cols:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            records.append({
                **base,
                "metric": metric,
                "n_seeds": int(values.size),
                "mean": mean,
                "sd": sd,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
                "cv_abs": float(abs(sd / mean)) if mean != 0 else float("nan"),
                "range_percentage_points": percentage_point_range(values)
                    if ("pct" in metric.lower() or "percent" in metric.lower())
                    else float("nan"),
            })
    return pd.DataFrame(records)


def pick_group_cols(table_key: str, df: pd.DataFrame) -> List[str]:
    common = set(df.columns)
    for candidate in GROUP_COLUMN_CANDIDATES.get(table_key, []):
        if all(c in common for c in candidate):
            return candidate
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in non_numeric if c not in {"run_name"}]
    if not non_numeric:
        raise RuntimeError(f"Could not infer group columns for {table_key}")
    return non_numeric[:1]


def build_state_key(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str], str]:
    available = [c for c in COMPOSITE_STATE_KEY if c in df.columns]
    if len(available) == len(COMPOSITE_STATE_KEY) and df.duplicated(available).sum() == 0:
        return df.copy(), available, "composite_state_key"

    # Fallback: keep order after stable sorting by all available metadata columns.
    # This should rarely be needed; it is reported in the audit output.
    fallback_cols = [c for c in available if c in df.columns]
    temp = df.copy()
    if fallback_cols:
        temp = temp.sort_values(fallback_cols, kind="mergesort").reset_index(drop=True)
    else:
        temp = temp.reset_index(drop=True)
    temp["_state_row_key"] = np.arange(len(temp))
    return temp, ["_state_row_key"], "row_order_fallback"


def state_level_consistency(run_data: Mapping[str, Dict[str, Any]]) -> pd.DataFrame:
    frames: Dict[str, pd.DataFrame] = {}
    key_cols_by_run: Dict[str, List[str]] = {}
    key_mode_by_run: Dict[str, str] = {}

    for run_name, data in run_data.items():
        raw = pd.read_csv(data["velocity"], low_memory=False)
        keyed, key_cols, key_mode = build_state_key(raw)
        frames[run_name] = keyed
        key_cols_by_run[run_name] = key_cols
        key_mode_by_run[run_name] = key_mode

    # Use composite key only if all runs support it; otherwise use fallback row key.
    if all(mode == "composite_state_key" for mode in key_mode_by_run.values()):
        join_cols = COMPOSITE_STATE_KEY
        key_mode = "composite_state_key"
    else:
        # Rebuild all as row fallback consistently.
        frames = {}
        for run_name, data in run_data.items():
            raw = pd.read_csv(data["velocity"], low_memory=False)
            keyed = raw.copy().reset_index(drop=True)
            keyed["_state_row_key"] = np.arange(len(keyed))
            frames[run_name] = keyed
        join_cols = ["_state_row_key"]
        key_mode = "row_order_fallback"

    common_cols = set.intersection(*(set(df.columns) for df in frames.values()))
    numeric_metrics = [
        c for c in common_cols
        if c not in join_cols and pd.api.types.is_numeric_dtype(frames[next(iter(frames))][c])
        and any(tok in c.lower() for tok in ["projection", "cosine", "similarity", "velocity_norm"])
    ]
    numeric_metrics = sorted(numeric_metrics)

    run_names = list(frames)
    records: List[Dict[str, Any]] = []
    for i in range(len(run_names)):
        for j in range(i + 1, len(run_names)):
            left_name, right_name = run_names[i], run_names[j]
            left = frames[left_name][join_cols + numeric_metrics]
            right = frames[right_name][join_cols + numeric_metrics]
            merged = left.merge(right, on=join_cols, how="inner", suffixes=("_left", "_right"))
            for metric in numeric_metrics:
                x = pd.to_numeric(merged[f"{metric}_left"], errors="coerce")
                y = pd.to_numeric(merged[f"{metric}_right"], errors="coerce")
                valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if len(x) < 3:
                    continue
                records.append({
                    "run_left": left_name,
                    "run_right": right_name,
                    "metric": metric,
                    "key_mode": key_mode,
                    "n_common_states": int(len(x)),
                    "pearson": float(x.corr(y, method="pearson")),
                    "spearman": float(x.rank(method="average").corr(y.rank(method="average"))),
                    "sign_agreement": float((np.sign(x.to_numpy()) == np.sign(y.to_numpy())).mean()),
                    "mean_absolute_difference": float(np.mean(np.abs(x.to_numpy() - y.to_numpy()))),
                })
    return pd.DataFrame(records)


def metric_series(metrics_df: pd.DataFrame, exact_or_partial_names: Sequence[str]) -> pd.Series | None:
    for name in exact_or_partial_names:
        if name in metrics_df.columns:
            return pd.to_numeric(metrics_df[name], errors="coerce")
    for col in metrics_df.columns:
        low = col.lower()
        if any(x.lower() in low for x in exact_or_partial_names):
            return pd.to_numeric(metrics_df[col], errors="coerce")
    return None


def main() -> None:
    outdir = choose_output_dir(Path(str(CONFIG["summary_output_dir"])))
    run_data = prepare_run_data()

    integrity = []
    for run_name, data in run_data.items():
        status_path = Path(data["run_dir"]) / "run_status.json"
        status = "not_found"
        returncode = None
        if status_path.is_file():
            obj = read_json(status_path)
            status = obj.get("status", "unknown")
            returncode = obj.get("returncode")
        integrity.append({
            "run_name": run_name,
            "seed": data["seed"],
            "run_dir": str(data["run_dir"]),
            "tables_dir": str(data["tables_dir"]),
            "run_status": status,
            "returncode": returncode,
            "integrity_ok": status in {"ok", "not_found"},
        })
    integrity_df = pd.DataFrame(integrity)
    integrity_df.to_csv(outdir / "01_run_integrity_check.csv", index=False, encoding="utf-8-sig")

    metrics_records = []
    for run_name, data in run_data.items():
        metrics_records.append({
            "run_name": run_name,
            "seed": data["seed"],
            **flatten(read_json(data["metrics"])),
        })
    metrics_df = pd.DataFrame(metrics_records).sort_values("seed")
    metrics_df.to_csv(outdir / "02_final_metrics_across_seeds.csv", index=False, encoding="utf-8-sig")

    rows = []
    for col in [c for c in metrics_df.select_dtypes(include=[np.number]).columns if c != "seed"]:
        vals = pd.to_numeric(metrics_df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        mean = float(vals.mean())
        rows.append({
            "metric": col,
            "n_seeds": int(len(vals)),
            "mean": mean,
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
            "relative_range_to_mean": float((vals.max() - vals.min()) / abs(mean)) if mean else float("nan"),
        })
    pd.DataFrame(rows).to_csv(outdir / "03_final_metrics_stability_summary.csv", index=False, encoding="utf-8-sig")

    summaries: Dict[str, pd.DataFrame] = {}
    for key, long_name, summary_name in [
        ("stage", "04_stage_directionality_across_seeds.csv", "05_stage_stability_summary.csv"),
        ("disease", "06_disease_directionality_across_seeds.csv", "07_disease_stability_summary.csv"),
        ("celltype", "08_celltype_directionality_across_seeds.csv", "09_celltype_stability_summary.csv"),
        ("cluster", "10_cluster_directionality_across_seeds.csv", "11_cluster_stability_summary.csv"),
    ]:
        long_df = combine_csv(run_data, key)
        group_cols = pick_group_cols(key, long_df)
        summary_df = summarize_grouped_numeric(long_df, group_cols)
        long_df.to_csv(outdir / long_name, index=False, encoding="utf-8-sig")
        summary_df.to_csv(outdir / summary_name, index=False, encoding="utf-8-sig")
        summaries[key] = summary_df

    th = CONFIG["thresholds"]
    cell_flags = summaries["celltype"]
    cell_flags = cell_flags[
        cell_flags["range_percentage_points"].notna()
        & (cell_flags["range_percentage_points"] >= float(th["celltype_flag_percentage_point_range"]))
    ].sort_values("range_percentage_points", ascending=False)
    cell_flags.to_csv(outdir / "12_celltype_seed_sensitivity_flags.csv", index=False, encoding="utf-8-sig")

    cluster_flags = summaries["cluster"]
    cluster_flags = cluster_flags[
        cluster_flags["range_percentage_points"].notna()
        & (cluster_flags["range_percentage_points"] >= float(th["cluster_flag_percentage_point_range"]))
    ].sort_values("range_percentage_points", ascending=False)
    cluster_flags.to_csv(outdir / "13_cluster_seed_sensitivity_flags.csv", index=False, encoding="utf-8-sig")

    state_cons = state_level_consistency(run_data)
    state_cons.to_csv(outdir / "14_state_level_pairwise_consistency.csv", index=False, encoding="utf-8-sig")

    val_r2 = metric_series(metrics_df, ["val_r2", "validation_r2"])
    val_mse = metric_series(metrics_df, ["val_mse", "validation_mse"])
    val_r2_range = float(val_r2.max() - val_r2.min()) if val_r2 is not None else float("nan")
    val_mse_relative_range = (
        float((val_mse.max() - val_mse.min()) / abs(val_mse.mean()))
        if val_mse is not None and val_mse.mean() != 0 else float("nan")
    )

    stage_pp = summaries["stage"]["range_percentage_points"].dropna()
    disease_pp = summaries["disease"]["range_percentage_points"].dropna()
    max_stage_pp = float(stage_pp.max()) if not stage_pp.empty else float("nan")
    max_disease_pp = float(disease_pp.max()) if not disease_pp.empty else float("nan")

    projection_rows = state_cons[
        state_cons["metric"].str.contains("projection|cosine|similarity", case=False, regex=True)
    ]
    median_spearman = float(projection_rows["spearman"].median()) if not projection_rows.empty else float("nan")
    median_sign = float(projection_rows["sign_agreement"].median()) if not projection_rows.empty else float("nan")

    checks = {
        "all_runs_complete": bool(integrity_df["integrity_ok"].all()),
        "val_r2_stable": bool(val_r2_range <= float(th["max_val_r2_range"])) if math.isfinite(val_r2_range) else False,
        "val_mse_stable": bool(val_mse_relative_range <= float(th["max_relative_val_mse_range"])) if math.isfinite(val_mse_relative_range) else False,
        "stage_directionality_stable": bool(max_stage_pp <= float(th["max_stage_percentage_point_range"])) if math.isfinite(max_stage_pp) else False,
        "disease_directionality_stable": bool(max_disease_pp <= float(th["max_disease_percentage_point_range"])) if math.isfinite(max_disease_pp) else False,
        "state_projection_rank_stable": bool(median_spearman >= float(th["min_state_projection_spearman"])) if math.isfinite(median_spearman) else False,
        "state_projection_sign_stable": bool(median_sign >= float(th["min_state_projection_sign_agreement"])) if math.isfinite(median_sign) else False,
    }

    if all(checks.values()):
        verdict = "PASS_GLOBAL_SEED_STABILITY"
    elif checks["all_runs_complete"] and checks["val_r2_stable"] and checks["val_mse_stable"]:
        verdict = "CAUTION_DIRECTIONALITY_SEED_SENSITIVITY"
    else:
        verdict = "FAIL_OR_REVIEW_REQUIRED"

    assessment = {
        "verdict": verdict,
        "operational_thresholds": th,
        "observed": {
            "val_r2_range": val_r2_range,
            "val_mse_relative_range": val_mse_relative_range,
            "max_stage_percentage_point_range": max_stage_pp,
            "max_disease_percentage_point_range": max_disease_pp,
            "median_state_projection_spearman": median_spearman,
            "median_state_projection_sign_agreement": median_sign,
            "n_celltype_sensitivity_flags": int(len(cell_flags)),
            "n_cluster_sensitivity_flags": int(len(cluster_flags)),
            "state_consistency_key_mode": state_cons["key_mode"].iloc[0] if not state_cons.empty else "not_available",
            "state_consistency_n_common_states_unique": int(state_cons["n_common_states"].median()) if not state_cons.empty else None,
        },
        "checks": checks,
        "interpretation_note": (
            "Global fit metrics are assessed separately from directional biological stability. "
            "Use flagged cell types/clusters as cautionary rather than primary robust conclusions."
        ),
    }
    json_dump(outdir / "15_overall_seed_stability_assessment.json", assessment)

    print("Seed robustness v2 summary completed.")
    print("Output:", outdir)
    print("Verdict:", verdict)
    print(json.dumps(assessment["observed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
