#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CELLSTATE minimum-cell threshold robustness through the formal 21a Neural ODE.

This launcher does not modify the original 21a script. For each run, it creates
an isolated copy, patches only input_csv/outdir/min_cells/seed, executes it in a
separate Python process, saves logs and metadata, and verifies key outputs.

Run design:
- ODE_CELLSTATE_min10_baseline <- CELLSTATE_baseline, min_cells=10
- ODE_CELLSTATE_min8           <- CELLSTATE_min8_bins5, min_cells=8
- ODE_CELLSTATE_min15          <- CELLSTATE_min15_bins5, min_cells=15

Quantile-bin runs (bins4/bins6) are not propagated because formal 21a uses the
stage-based 04_cell_state_matrix_v1B_filtered.csv and axis_v2_stage.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pprint
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# PYCHARM CONFIGURATION
# =============================================================================
CONFIG: Dict[str, Any] = {
    # The launcher searches these folders recursively for the exact 21a filename.
    "script_roots": [
        r"E:\PythonProject",
        r"E:\PythonProject\Python脚本",
    ],
    "source_script_name":
        "B! 21a_v1B_celltype_time_aware_direction_constrained_neural_ode.py",

    # V3 CELLSTATE robustness output root.
    "cellstate_runs_root":
        r"E:\PythonProject\results\robustness_runs_v3\03_cell_state_matrix",

    # A new dedicated output root; it does not overlap the formal analysis folder.
    "output_root":
        r"E:\PythonProject\results\robustness_runs_v3\04_neural_ode_cellstate_threshold",

    "python_bin":
        r"E:\Anaconda3\Anaconda\envs\pytorch_fixed\python.exe",
    "python_project_root": r"E:\PythonProject",

    # First run baseline only. After baseline passes, change to min8/min15 or all 3.
    "run_targets": [
        "ODE_CELLSTATE_min8",
    "ODE_CELLSTATE_min15",
    ],

    "execute": True,
    "seed": 123,
    "stop_on_error": True,

    # Never overwrite an existing non-empty run folder.
    # error: stop; skip: preserve and skip; timestamp: create a new suffixed folder.
    "existing_output_policy": "error",
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "ODE_CELLSTATE_min10_baseline": {
        "cellstate_run_id": "CELLSTATE_baseline",
        "min_cells": 10,
    },
    "ODE_CELLSTATE_min8": {
        "cellstate_run_id": "CELLSTATE_min8_bins5",
        "min_cells": 8,
    },
    "ODE_CELLSTATE_min15": {
        "cellstate_run_id": "CELLSTATE_min15_bins5",
        "min_cells": 15,
    },
}

INPUT_FILENAME = "04_cell_state_matrix_v1B_filtered.csv"
REQUIRED_OUTPUTS = [
    "tables/01_features_used_for_latent_state.csv",
    "tables/07_transition_pairs_all_before_group_filter.csv",
    "tables/08_transition_pair_count_by_group.csv",
    "tables/09_transition_pairs_used_for_training.csv",
    "tables/10_training_history.csv",
    "tables/11_final_metrics.json",
    "tables/13_direction_constrained_state_velocity_scores.csv",
    "tables/14_directionality_by_celltype.csv",
    "tables/15_directionality_by_axis_stage.csv",
    "tables/16_directionality_by_disease.csv",
    "tables/17_directionality_by_celltype_cluster.csv",
    "tables/18_validation_pair_predictions.csv",
]


@dataclass
class RunStatus:
    run_id: str
    status: str
    returncode: Optional[int]
    source_script: str
    input_csv: str
    prepared_script: str
    run_dir: str
    output_dir: str
    n_output_files: int
    missing_outputs: List[str]
    started_at: str
    finished_at: str
    message: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def time_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def find_source(name: str, roots: Sequence[Path]) -> Tuple[Path, List[Path]]:
    hits: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            hits.append(direct.resolve())
        hits.extend(p.resolve() for p in root.rglob(name) if p.is_file())
    hits = sorted(set(hits), key=lambda p: (len(p.parts), len(str(p)), str(p).lower()))
    if not hits:
        raise FileNotFoundError(f"Source script not found: {name}; roots={list(map(str, roots))}")
    return hits[0], hits


def find_input(root: Path, run_id: str) -> Tuple[Path, List[Path]]:
    preferred = root / run_id / "outputs" / "tables" / INPUT_FILENAME
    hits: List[Path] = [preferred.resolve()] if preferred.is_file() else []
    run_dir = root / run_id
    if run_dir.is_dir():
        hits.extend(p.resolve() for p in run_dir.rglob(INPUT_FILENAME) if p.is_file())
    if not hits and root.parent.is_dir():
        hits.extend(
            p.resolve() for p in root.parent.rglob(INPUT_FILENAME)
            if run_id.lower() in str(p).lower()
        )
    hits = sorted(set(hits), key=lambda p: (len(p.parts), len(str(p)), str(p).lower()))
    if not hits:
        raise FileNotFoundError(f"Input for {run_id} not found. Expected near: {preferred}")
    return hits[0], hits


def choose_run_dir(base: Path, policy: str) -> Optional[Path]:
    if not base.exists() or count_files(base) == 0:
        base.mkdir(parents=True, exist_ok=True)
        return base
    policy = policy.lower().strip()
    if policy == "skip":
        return None
    if policy == "timestamp":
        new = base.parent / f"{base.name}_{time_tag()}"
        new.mkdir(parents=True, exist_ok=False)
        return new
    if policy == "error":
        raise FileExistsError(
            f"Existing non-empty run directory was preserved:\n{base}\n"
            "Use 'skip' or 'timestamp', or select another run target."
        )
    raise ValueError(f"Unknown existing_output_policy: {policy}")


# =============================================================================
# Safe AST-based CONFIG patching
# =============================================================================
def _offset(text: str, lineno: int, col_bytes: int) -> int:
    lines = text.splitlines(keepends=True)
    base = sum(len(x) for x in lines[:lineno - 1])
    prefix = lines[lineno - 1].encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore")
    return base + len(prefix)


def _config_dict(tree: ast.AST) -> Tuple[str, ast.Dict]:
    found: List[Tuple[str, ast.Dict]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CONFIG", "BASE_CONFIG"}:
                    found.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"CONFIG", "BASE_CONFIG"} and isinstance(node.value, ast.Dict):
                found.append((node.target.id, node.value))
    if not found:
        raise RuntimeError("No top-level CONFIG or BASE_CONFIG dictionary found in 21a source.")
    found.sort(key=lambda x: 0 if x[0] == "CONFIG" else 1)
    return found[0]


def patch_config(text: str, values: Mapping[str, Any]) -> Tuple[str, Dict[str, Dict[str, str]]]:
    tree = ast.parse(text)
    config_name, node = _config_dict(tree)
    key_map: Dict[str, ast.AST] = {}
    for key_node, value_node in zip(node.keys, node.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key_map[key_node.value] = value_node

    edits = []
    report: Dict[str, Dict[str, str]] = {}
    for key, value in values.items():
        n = key_map.get(key)
        if n is None:
            raise KeyError(f"{config_name} key not found in source: {key}")
        start = _offset(text, n.lineno, n.col_offset)
        end = _offset(text, n.end_lineno, n.end_col_offset)
        old = text[start:end]
        new = pprint.pformat(str(value) if isinstance(value, Path) else value, sort_dicts=False)
        edits.append((start, end, new, key, old))

    patched = text
    for start, end, new, key, old in sorted(edits, key=lambda x: x[0], reverse=True):
        patched = patched[:start] + new + patched[end:]
        report[key] = {"old": old, "new": new}
    ast.parse(patched)
    return patched, report


# =============================================================================
# Input QC and process execution
# =============================================================================
def input_qc(path: Path, expected_min: int) -> Dict[str, Any]:
    required = {
        "cell_state_id", "sample_id", "Disease_Identity", "celltype_marker_based",
        "cluster_default", "axis_v2_stage", "n_cells", "mean_axis_v2_score",
    }
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    df = pd.read_csv(
        path,
        usecols=["cell_state_id", "Disease_Identity", "celltype_marker_based",
                 "cluster_default", "axis_v2_stage", "n_cells"],
        low_memory=False,
    )
    n = pd.to_numeric(df["n_cells"], errors="coerce")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "file_size_bytes": path.stat().st_size,
        "n_rows": int(len(df)),
        "n_unique_states": int(df["cell_state_id"].nunique()),
        "min_n_cells": float(n.min()),
        "max_n_cells": float(n.max()),
        "n_rows_below_expected_min": int((n < expected_min).sum()),
        "n_diseases": int(df["Disease_Identity"].nunique()),
        "n_celltypes": int(df["celltype_marker_based"].nunique()),
        "n_celltype_clusters": int(df[["celltype_marker_based", "cluster_default"]].drop_duplicates().shape[0]),
        "stage_counts": {str(k): int(v) for k, v in df["axis_v2_stage"].value_counts(dropna=False).items()},
    }


def stream(command: List[str], cwd: Path, out_log: Path, err_log: Path) -> int:
    proc = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=os.environ.copy(),
    )

    def pump(pipe, path: Path, prefix: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            assert pipe is not None
            for line in iter(pipe.readline, ""):
                print(prefix + line, end="")
                f.write(line)
                f.flush()
            pipe.close()

    t1 = threading.Thread(target=pump, args=(proc.stdout, out_log, ""), daemon=True)
    t2 = threading.Thread(target=pump, args=(proc.stderr, err_log, "[STDERR] "), daemon=True)
    t1.start(); t2.start()
    code = proc.wait()
    t1.join(); t2.join()
    return int(code)


def missing_outputs(output_dir: Path) -> List[str]:
    result = []
    for rel in REQUIRED_OUTPUTS:
        p = output_dir / Path(rel)
        if not p.is_file() or p.stat().st_size == 0:
            result.append(rel)
    return result


def run_one(run_id: str, source: Path, source_hits: Sequence[Path]) -> RunStatus:
    started = now_iso()
    scenario = SCENARIOS[run_id]
    base = Path(CONFIG["output_root"]) / run_id
    run_dir = choose_run_dir(base, str(CONFIG["existing_output_policy"]))
    if run_dir is None:
        return RunStatus(run_id, "skipped_existing", None, str(source), "", "", str(base),
                         str(base / "outputs"), count_files(base), [], started, now_iso(),
                         "Existing output preserved and skipped.")

    output_dir = run_dir / "outputs"
    meta_dir = run_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    input_csv, input_hits = find_input(Path(CONFIG["cellstate_runs_root"]), scenario["cellstate_run_id"])
    qc = input_qc(input_csv, int(scenario["min_cells"]))
    dump_json(meta_dir / "input_qc.json", qc)

    values = {
        "input_csv": str(input_csv),
        "outdir": str(output_dir),
        "min_cells": int(scenario["min_cells"]),
        "seed": int(CONFIG["seed"]),
    }
    text = source.read_text(encoding="utf-8-sig")
    patched, patch_report = patch_config(text, values)
    prepared = run_dir / "prepared_formal_21a_neural_ode.py"
    prepared.write_text(patched, encoding="utf-8")
    compile(patched, str(prepared), "exec")

    dump_json(meta_dir / "patch_report.json", patch_report)
    dump_json(meta_dir / "run_plan.json", {
        "run_id": run_id,
        "cellstate_run_id": scenario["cellstate_run_id"],
        "source_script": str(source),
        "source_script_sha256": sha256(source),
        "source_candidates": [str(p) for p in source_hits],
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256(input_csv),
        "input_candidates": [str(p) for p in input_hits],
        "output_dir": str(output_dir),
        "patched_values": values,
        "other_model_parameters": "unchanged from formal 21a source",
        "created_at": now_iso(),
    })

    if not bool(CONFIG["execute"]):
        return RunStatus(run_id, "prepared_only", None, str(source), str(input_csv), str(prepared),
                         str(run_dir), str(output_dir), 0, REQUIRED_OUTPUTS.copy(),
                         started, now_iso(), "Prepared and syntax-checked; execute=False.")

    python_bin = Path(CONFIG["python_bin"])
    project_root = Path(CONFIG["python_project_root"])
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"Python project root not found: {project_root}")

    print("\n" + "=" * 90)
    print(f"RUN: {run_id}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_dir}")
    print(f"min_cells={scenario['min_cells']}, seed={CONFIG['seed']}")
    print("=" * 90)
    code = stream([str(python_bin), str(prepared)], project_root,
                  run_dir / "stdout.log", run_dir / "stderr.log")
    missing = missing_outputs(output_dir)
    if code == 0 and not missing:
        status, message = "ok", "Completed; all required outputs verified."
    elif code == 0:
        status, message = "incomplete_outputs", "Return code 0 but required outputs are missing."
    else:
        status, message = "failed", f"Neural ODE returned exit code {code}."
    return RunStatus(run_id, status, code, str(source), str(input_csv), str(prepared),
                     str(run_dir), str(output_dir), count_files(output_dir), missing,
                     started, now_iso(), message)


def main() -> None:
    roots = [Path(x) for x in CONFIG["script_roots"]]
    source, source_hits = find_source(str(CONFIG["source_script_name"]), roots)
    print("Formal source:", source)
    print("Source SHA256:", sha256(source))

    targets = list(CONFIG["run_targets"])
    unknown = [x for x in targets if x not in SCENARIOS]
    if not targets:
        raise ValueError("run_targets is empty.")
    if unknown:
        raise KeyError(f"Unknown run targets: {unknown}")

    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[RunStatus] = []

    for run_id in targets:
        try:
            result = run_one(run_id, source, source_hits)
        except Exception as exc:
            result = RunStatus(run_id, "runner_error", None, str(source), "", "",
                               str(output_root / run_id), str(output_root / run_id / "outputs"),
                               count_files(output_root / run_id / "outputs"), [], now_iso(), now_iso(),
                               f"{type(exc).__name__}: {exc}")
            print(f"[ERROR] {run_id}: {result.message}", file=sys.stderr)
        results.append(result)
        dump_json(Path(result.run_dir) / "run_status.json", asdict(result))
        print(f"[{result.status}] {run_id}: {result.message}")
        if result.missing_outputs:
            print("Missing outputs:")
            for x in result.missing_outputs:
                print(" -", x)
        if result.status not in {"ok", "prepared_only", "skipped_existing"} and CONFIG["stop_on_error"]:
            break

    summary = pd.DataFrame([asdict(x) for x in results])
    summary_path = output_root / f"run_batch_summary_{time_tag()}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\nBatch summary:")
    print(summary[["run_id", "status", "returncode", "n_output_files"]].to_string(index=False))
    print("Saved:", summary_path)

    bad = summary[~summary["status"].isin(["ok", "prepared_only", "skipped_existing"])]
    if not bad.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
