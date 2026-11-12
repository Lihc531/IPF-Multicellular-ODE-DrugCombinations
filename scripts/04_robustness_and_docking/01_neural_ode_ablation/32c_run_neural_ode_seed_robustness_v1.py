#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_run_neural_ode_seed_robustness_v1.py

Run random-seed robustness experiments for the formal v1B.1 Neural ODE model.

Design
------
The completed min10/seed123 baseline is retained as the reference run. This
script creates only two new runs using the SAME baseline CELLSTATE input and
SAME model configuration, changing only the random seed:

    ODE_SEED_456
    ODE_SEED_789

The original formal 21a source script is never modified. Each run receives a
patched copy in an isolated output directory. Existing non-empty run folders
are preserved by default.
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
# 1. USER CONFIGURATION
# =============================================================================

CONFIG: Dict[str, Any] = {
    # Locations where the formal 21a source script may exist.
    "script_roots": [
        r"E:\PythonProject",
        r"E:\PythonProject\Python脚本",
    ],

    "source_script_name":
        "B! 21a_v1B_celltype_time_aware_direction_constrained_neural_ode.py",

    # Fixed CELLSTATE baseline input. This must be the same min10/bins5 input
    # used by ODE_CELLSTATE_min10_baseline.
    "input_csv":
        r"E:\PythonProject\results\robustness_runs_v3\03_cell_state_matrix\CELLSTATE_baseline\outputs\tables\04_cell_state_matrix_v1B_filtered.csv",

    # New root dedicated to random-seed robustness. It does not overlap the
    # formal analysis or the threshold-robustness outputs.
    "output_root":
        r"E:\PythonProject\results\robustness_runs_v3\05_neural_ode_seed_robustness",

    # Python environment containing torch, sklearn and torchdiffeq.
    "python_bin":
        r"E:\Anaconda3\Anaconda\envs\pytorch_fixed\python.exe",

    "python_project_root": r"E:\PythonProject",

    # seed=123 already exists as ODE_CELLSTATE_min10_baseline. Run only the two
    # additional seeds here. ODE_SEED_123_REPEAT is available but disabled.
    "run_targets": [
        "ODE_SEED_456",
        "ODE_SEED_789",
    ],

    # True executes training. False only prepares and syntax-checks scripts.
    "execute": True,

    # error: preserve existing non-empty folder and stop (recommended)
    # skip: preserve and skip
    # timestamp: write to a new timestamped folder
    "existing_output_policy": "error",

    "stop_on_error": True,
    "validate_input": True,

    # The baseline matrix and the formal model both use min_cells=10.
    "min_cells": 10,

    "required_output_files": [
        r"tables\01_features_used_for_latent_state.csv",
        r"tables\02_pca_explained_variance.csv",
        r"tables\03_pca_feature_loadings.csv",
        r"tables\04_condition_features.csv",
        r"tables\05_stage_aware_constraint_weights.csv",
        r"tables\06_direction_vectors_and_thresholds.json",
        r"tables\07_transition_pairs_all_before_group_filter.csv",
        r"tables\08_transition_pair_count_by_group.csv",
        r"tables\09_transition_pairs_used_for_training.csv",
        r"tables\10_training_history.csv",
        r"tables\11_final_metrics.json",
        r"tables\12_latent_states.csv",
        r"tables\13_direction_constrained_state_velocity_scores.csv",
        r"tables\14_directionality_by_celltype.csv",
        r"tables\15_directionality_by_axis_stage.csv",
        r"tables\16_directionality_by_disease.csv",
        r"tables\17_directionality_by_celltype_cluster.csv",
        r"tables\18_validation_pair_predictions.csv",
    ],
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "ODE_SEED_456": {
        "seed": 456,
        "description": "Formal v1B.1 model; baseline min10 input; random seed 456.",
    },
    "ODE_SEED_789": {
        "seed": 789,
        "description": "Formal v1B.1 model; baseline min10 input; random seed 789.",
    },
    "ODE_SEED_123_REPEAT": {
        "seed": 123,
        "description": "Optional independent repeat of the baseline seed 123 run.",
    },
}


# =============================================================================
# 2. HELPERS
# =============================================================================

@dataclass
class RunStatus:
    run_id: str
    seed: int
    status: str
    returncode: Optional[int]
    source_script: str
    prepared_script: str
    input_csv: str
    output_dir: str
    run_dir: str
    n_output_files: int
    missing_required_outputs: List[str]
    started_at: str
    finished_at: str
    message: str = ""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def resolve_source_script(name: str, roots: Sequence[Path]) -> Tuple[Path, List[Path]]:
    hits: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            hits.append(direct.resolve())
        hits.extend(p.resolve() for p in root.rglob(name) if p.is_file())

    unique = sorted(set(hits), key=lambda p: (len(p.parts), len(str(p)), str(p).lower()))
    if not unique:
        raise FileNotFoundError(
            f"Formal source script not found: {name}\n"
            f"Searched: {[str(x) for x in roots]}"
        )
    return unique[0], unique


def choose_run_dir(base: Path, policy: str) -> Optional[Path]:
    if not base.exists() or count_files(base) == 0:
        base.mkdir(parents=True, exist_ok=True)
        return base

    policy = policy.strip().lower()
    if policy == "skip":
        return None
    if policy == "timestamp":
        target = base.parent / f"{base.name}_{timestamp_tag()}"
        target.mkdir(parents=True, exist_ok=False)
        return target
    if policy == "error":
        raise FileExistsError(
            f"Existing non-empty run directory was preserved:\n{base}\n"
            "Use 'skip' or 'timestamp' only after inspecting the existing result."
        )
    raise ValueError(f"Unknown existing_output_policy: {policy}")


# =============================================================================
# 3. SAFE AST CONFIG PATCHING
# =============================================================================


def _position_to_offset(text: str, lineno: int, col_offset_bytes: int) -> int:
    lines = text.splitlines(keepends=True)
    base = sum(len(line) for line in lines[: lineno - 1])
    line = lines[lineno - 1]
    prefix = line.encode("utf-8")[:col_offset_bytes].decode("utf-8", errors="ignore")
    return base + len(prefix)


def _find_top_level_config(tree: ast.AST) -> Tuple[str, ast.Dict]:
    candidates: List[Tuple[str, ast.Dict]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CONFIG", "BASE_CONFIG"}:
                    candidates.append((target.id, node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Dict)
            and node.target.id in {"CONFIG", "BASE_CONFIG"}
        ):
            candidates.append((node.target.id, node.value))

    if not candidates:
        raise RuntimeError("No top-level CONFIG or BASE_CONFIG dictionary found.")
    candidates.sort(key=lambda item: 0 if item[0] == "CONFIG" else 1)
    return candidates[0]


def _dict_key_map(node: ast.Dict) -> Dict[str, ast.AST]:
    result: Dict[str, ast.AST] = {}
    for key_node, value_node in zip(node.keys, node.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            result[key_node.value] = value_node
    return result


def python_literal(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, dict):
        return pprint.pformat(value, sort_dicts=False, width=100)
    return repr(value)


def patch_source_config(
    source_text: str,
    replacements: Mapping[str, Any],
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    tree = ast.parse(source_text)
    config_name, config_node = _find_top_level_config(tree)
    key_map = _dict_key_map(config_node)

    edits: List[Tuple[int, int, str, str, str]] = []
    report: Dict[str, Dict[str, str]] = {}

    for key, value in replacements.items():
        node = key_map.get(key)
        if node is None:
            raise KeyError(f"Required {config_name} key missing: {key}")
        attrs = ("lineno", "col_offset", "end_lineno", "end_col_offset")
        if not all(hasattr(node, attr) for attr in attrs):
            raise RuntimeError(f"AST source location unavailable for key: {key}")

        start = _position_to_offset(source_text, node.lineno, node.col_offset)
        end = _position_to_offset(source_text, node.end_lineno, node.end_col_offset)
        old_text = source_text[start:end]
        new_text = python_literal(value)
        edits.append((start, end, new_text, key, old_text))

    patched = source_text
    for start, end, new_text, key, old_text in sorted(edits, key=lambda x: x[0], reverse=True):
        patched = patched[:start] + new_text + patched[end:]
        report[key] = {"old": old_text, "new": new_text}

    ast.parse(patched)
    return patched, report


# =============================================================================
# 4. INPUT AND PROCESS VALIDATION
# =============================================================================


def validate_cellstate_csv(path: Path, expected_min_cells: int) -> Dict[str, Any]:
    required_columns = {
        "cell_state_id",
        "sample_id",
        "Disease_Identity",
        "celltype_marker_based",
        "cluster_default",
        "axis_v2_stage",
        "n_cells",
        "mean_axis_v2_score",
    }

    header = pd.read_csv(path, nrows=0)
    missing = sorted(required_columns - set(header.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    qc = pd.read_csv(
        path,
        usecols=[
            "cell_state_id",
            "Disease_Identity",
            "celltype_marker_based",
            "cluster_default",
            "axis_v2_stage",
            "n_cells",
        ],
        low_memory=False,
    )
    n_cells = pd.to_numeric(qc["n_cells"], errors="coerce")
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "n_rows": int(len(qc)),
        "n_unique_state_ids": int(qc["cell_state_id"].nunique()),
        "min_n_cells_observed": float(n_cells.min()),
        "max_n_cells_observed": float(n_cells.max()),
        "n_rows_below_expected_min": int((n_cells < expected_min_cells).sum()),
        "n_diseases": int(qc["Disease_Identity"].nunique()),
        "n_celltypes": int(qc["celltype_marker_based"].nunique()),
        "n_celltype_clusters": int(
            qc[["celltype_marker_based", "cluster_default"]].drop_duplicates().shape[0]
        ),
        "stage_counts": {
            str(k): int(v)
            for k, v in qc["axis_v2_stage"].value_counts(dropna=False).items()
        },
    }


def stream_process(command: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )

    def pump(stream, path: Path, prefix: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            assert stream is not None
            for line in iter(stream.readline, ""):
                print(f"{prefix}{line}", end="")
                handle.write(line)
                handle.flush()
            stream.close()

    out_thread = threading.Thread(
        target=pump, args=(process.stdout, stdout_path, ""), daemon=True
    )
    err_thread = threading.Thread(
        target=pump, args=(process.stderr, stderr_path, "[STDERR] "), daemon=True
    )
    out_thread.start()
    err_thread.start()
    returncode = process.wait()
    out_thread.join()
    err_thread.join()
    return int(returncode)


def verify_outputs(output_dir: Path, required_files: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for rel in required_files:
        target = output_dir / Path(rel)
        if not target.is_file() or target.stat().st_size == 0:
            missing.append(rel)
    return missing


# =============================================================================
# 5. ONE RUN
# =============================================================================


def run_one(
    run_id: str,
    scenario: Mapping[str, Any],
    source_script: Path,
    source_candidates: Sequence[Path],
) -> RunStatus:
    started = now_iso()
    seed = int(scenario["seed"])
    output_root = Path(CONFIG["output_root"])
    requested_run_dir = output_root / run_id
    run_dir = choose_run_dir(requested_run_dir, str(CONFIG["existing_output_policy"]))

    if run_dir is None:
        return RunStatus(
            run_id=run_id,
            seed=seed,
            status="skipped_existing",
            returncode=None,
            source_script=str(source_script),
            prepared_script="",
            input_csv=str(CONFIG["input_csv"]),
            output_dir=str(requested_run_dir / "outputs"),
            run_dir=str(requested_run_dir),
            n_output_files=count_files(requested_run_dir),
            missing_required_outputs=[],
            started_at=started,
            finished_at=now_iso(),
            message="Existing non-empty run directory was preserved and skipped.",
        )

    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(CONFIG["input_csv"])
    if not input_csv.is_file():
        raise FileNotFoundError(f"Baseline CELLSTATE input not found: {input_csv}")

    input_qc: Dict[str, Any] = {}
    if bool(CONFIG["validate_input"]):
        input_qc = validate_cellstate_csv(input_csv, int(CONFIG["min_cells"]))
        json_dump(metadata_dir / "input_qc.json", input_qc)

    replacements = {
        "input_csv": str(input_csv),
        "outdir": str(output_dir),
        "min_cells": int(CONFIG["min_cells"]),
        "seed": seed,
    }

    source_text = source_script.read_text(encoding="utf-8-sig")
    patched_text, patch_report = patch_source_config(source_text, replacements)

    prepared_script = run_dir / "prepared_formal_21a_neural_ode.py"
    prepared_script.write_text(patched_text, encoding="utf-8")
    compile(patched_text, str(prepared_script), "exec")

    json_dump(
        metadata_dir / "run_plan.json",
        {
            "run_id": run_id,
            "description": scenario["description"],
            "source_script_selected": str(source_script),
            "source_script_candidates": [str(x) for x in source_candidates],
            "source_script_sha256": sha256_file(source_script),
            "input_csv": str(input_csv),
            "input_csv_sha256": sha256_file(input_csv),
            "output_dir": str(output_dir),
            "patched_config": replacements,
            "all_other_model_settings": "unchanged from formal 21a source",
            "created_at": now_iso(),
        },
    )
    json_dump(metadata_dir / "patch_report.json", patch_report)

    if not bool(CONFIG["execute"]):
        return RunStatus(
            run_id=run_id,
            seed=seed,
            status="prepared_only",
            returncode=None,
            source_script=str(source_script),
            prepared_script=str(prepared_script),
            input_csv=str(input_csv),
            output_dir=str(output_dir),
            run_dir=str(run_dir),
            n_output_files=count_files(output_dir),
            missing_required_outputs=list(CONFIG["required_output_files"]),
            started_at=started,
            finished_at=now_iso(),
            message="Prepared and syntax-checked; execution disabled.",
        )

    python_bin = Path(CONFIG["python_bin"])
    project_root = Path(CONFIG["python_project_root"])
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"Python project root not found: {project_root}")

    print("\n" + "=" * 88)
    print(f"RUN: {run_id}")
    print(f"Seed:   {seed}")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_dir}")
    print("=" * 88)

    returncode = stream_process(
        [str(python_bin), str(prepared_script)],
        cwd=project_root,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
    )

    missing = verify_outputs(output_dir, CONFIG["required_output_files"])
    if returncode == 0 and not missing:
        status = "ok"
        message = "Run completed and all required outputs were verified."
    elif returncode == 0:
        status = "incomplete_outputs"
        message = "Process returned 0, but required outputs are missing or empty."
    else:
        status = "failed"
        message = f"Neural ODE returned non-zero exit code: {returncode}"

    return RunStatus(
        run_id=run_id,
        seed=seed,
        status=status,
        returncode=returncode,
        source_script=str(source_script),
        prepared_script=str(prepared_script),
        input_csv=str(input_csv),
        output_dir=str(output_dir),
        run_dir=str(run_dir),
        n_output_files=count_files(output_dir),
        missing_required_outputs=missing,
        started_at=started,
        finished_at=now_iso(),
        message=message,
    )


# =============================================================================
# 6. MAIN
# =============================================================================


def main() -> None:
    roots = [Path(x) for x in CONFIG["script_roots"]]
    source_script, source_candidates = resolve_source_script(
        str(CONFIG["source_script_name"]), roots
    )

    print("Selected formal Neural ODE source:")
    print(" ", source_script)
    print("SHA256:", sha256_file(source_script))

    targets = list(CONFIG["run_targets"])
    if not targets:
        raise ValueError("CONFIG['run_targets'] is empty.")
    unknown = [x for x in targets if x not in SCENARIOS]
    if unknown:
        raise KeyError(f"Unknown run_targets: {unknown}; allowed: {sorted(SCENARIOS)}")

    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    statuses: List[RunStatus] = []
    for run_id in targets:
        started = now_iso()
        try:
            result = run_one(run_id, SCENARIOS[run_id], source_script, source_candidates)
        except Exception as exc:
            result = RunStatus(
                run_id=run_id,
                seed=int(SCENARIOS[run_id]["seed"]),
                status="runner_error",
                returncode=None,
                source_script=str(source_script),
                prepared_script="",
                input_csv=str(CONFIG["input_csv"]),
                output_dir=str(output_root / run_id / "outputs"),
                run_dir=str(output_root / run_id),
                n_output_files=count_files(output_root / run_id / "outputs"),
                missing_required_outputs=[],
                started_at=started,
                finished_at=now_iso(),
                message=f"{type(exc).__name__}: {exc}",
            )
            print(f"\n[ERROR] {run_id}: {result.message}", file=sys.stderr)

        statuses.append(result)
        json_dump(Path(result.run_dir) / "run_status.json", asdict(result))
        print(f"\n[{result.status}] {run_id}: {result.message}")

        if (
            result.status not in {"ok", "prepared_only", "skipped_existing"}
            and bool(CONFIG["stop_on_error"])
        ):
            break

    summary = pd.DataFrame([asdict(x) for x in statuses])
    summary_path = output_root / f"run_batch_summary_{timestamp_tag()}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nBatch summary:")
    if not summary.empty:
        print(summary[["run_id", "seed", "status", "returncode", "n_output_files"]].to_string(index=False))
    print("Saved:", summary_path)

    failed = summary[~summary["status"].isin(["ok", "prepared_only", "skipped_existing"])]
    if not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
