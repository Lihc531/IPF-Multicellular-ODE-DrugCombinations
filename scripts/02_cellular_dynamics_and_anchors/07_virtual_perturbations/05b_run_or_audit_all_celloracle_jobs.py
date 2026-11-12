#!/usr/bin/env python3
"""Manifest-aware audit and resume runner for all CellOracle jobs.

This version closes a gap in the previous job-level audit: the presence of two
aggregate output files is not enough to prove that every frozen anchor-TF test
was simulated. Completion now requires all of the following:

1. per-job ``targets_celloracle.csv`` matches the canonical TF manifest;
2. run metadata matches the current input signature;
3. every expected Control/IPF x KO/OE key exists in both output tables;
4. low-expression pairs are recorded as non-evaluable, while hard eligibility failures and runtime skips remain blockers.

Audit only:
    python 05b_run_or_audit_all_celloracle_jobs.py

Synchronize target tables and rerun incomplete dataset-celltype jobs:
    python 05b_run_or_audit_all_celloracle_jobs.py --run

Restrict reruns using the long missing-job table written by step 06:
    python 05b_run_or_audit_all_celloracle_jobs.py --run \
      --missing-jobs-csv celloracle_missing_jobs_long.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = Path(os.getenv("VP_PIPELINE_ROOT", str(SCRIPT_ROOT))).resolve()
OUTPUT_ROOT = Path(
    os.getenv("VP_OUTPUT_ROOT", str(PIPELINE_ROOT / "virtual_perturbation_results"))
).resolve()
INPUT_ROOT = OUTPUT_ROOT / "03_celloracle_inputs"
RESULT_ROOT = OUTPUT_ROOT / "04_celloracle"
TARGET_ROOT = OUTPUT_ROOT / "01_target_manifest"
CANONICAL_MANIFEST = TARGET_ROOT / "celloracle_tf_target_manifest.csv"
FROZEN_MANIFEST = TARGET_ROOT / "celloracle_tf_target_manifest.frozen.csv"
AUDIT_PATH = RESULT_ROOT / "celloracle_job_completion_audit.csv"
MISSING_TEST_PATH = RESULT_ROOT / "celloracle_output_tests_missing.csv"
NON_EVALUABLE_TEST_PATH = RESULT_ROOT / "celloracle_output_tests_non_evaluable.csv"
HARD_INELIGIBLE_PATH = RESULT_ROOT / "celloracle_hard_ineligible_pairs.csv"
RUNNER = SCRIPT_ROOT / "05_run_celloracle_ko_oe.py"
PAIR_COLS = ["celltype", "anchor_gene", "tf_gene"]
TEST_COLS = ["dataset", "celltype", "condition", "anchor_gene", "tf_gene", "perturbation"]
REQUIRED_CONDITIONS = ("Control", "IPF")
PERTURBATIONS = ("KO", "OE")
NON_EVALUABLE_REASONS = {"TF_below_expression_threshold"}
COMPLETED_STATUSES = {"completed_with_perturbation_results", "completed_no_evaluable_targets"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manifest-aware audit/resume for CellOracle jobs.")
    p.add_argument("--run", action="store_true", help="Run incomplete jobs after auditing.")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--min-expressing-cells", type=int, default=25)
    p.add_argument("--job", action="append", default=[], help="Restrict to a dataset__celltype job; repeatable.")
    p.add_argument(
        "--missing-jobs-csv",
        default=None,
        help="Optional celloracle_missing_jobs_long.csv from step 06; reruns unique dataset__celltype jobs.",
    )
    p.add_argument(
        "--no-sync-targets",
        action="store_true",
        help="Do not synchronize per-job targets_celloracle.csv from the canonical manifest before --run.",
    )
    return p.parse_args()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_signature(input_dir: Path) -> str | None:
    paths = [
        input_dir / "genes.tsv",
        input_dir / "barcodes.tsv",
        input_dir / "cell_metadata.csv",
        input_dir / "targets_celloracle.csv",
    ]
    if not all(path.is_file() for path in paths):
        return None
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(file_digest(path).encode("ascii"))
    return h.hexdigest()


def load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "invalid_run_metadata"}


def normalize_manifest(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    out = frame.copy()
    if "tf_gene" not in out.columns and "gene" in out.columns:
        out["tf_gene"] = out["gene"]
    if "anchor_gene" not in out.columns and "tf_gene" in out.columns:
        out["anchor_gene"] = out["tf_gene"]
    missing = [column for column in PAIR_COLS if column not in out.columns]
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")
    for column in PAIR_COLS:
        out[column] = out[column].astype(str).str.strip()
    out = out[(out["celltype"] != "") & (out["anchor_gene"] != "") & (out["tf_gene"] != "")]
    return out.drop_duplicates(PAIR_COLS).reset_index(drop=True)


def read_canonical_manifest() -> tuple[pd.DataFrame, Path]:
    if not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Canonical TF manifest not found: {CANONICAL_MANIFEST}. "
            "Run 04a_prepare_celloracle_tf_targets.py --apply first."
        )
    canonical = normalize_manifest(pd.read_csv(CANONICAL_MANIFEST), CANONICAL_MANIFEST)
    if canonical.empty:
        raise ValueError(f"Canonical TF manifest is empty: {CANONICAL_MANIFEST}")
    if FROZEN_MANIFEST.is_file():
        frozen = normalize_manifest(pd.read_csv(FROZEN_MANIFEST), FROZEN_MANIFEST)
        a = set(map(tuple, canonical[PAIR_COLS].to_numpy()))
        b = set(map(tuple, frozen[PAIR_COLS].to_numpy()))
        if a != b:
            print(
                "WARNING: selected and frozen TF manifests differ. "
                "The selected manifest is used because step 06 uses it as the expected test set.\n"
                f"  selected={len(a)} pairs; frozen={len(b)} pairs; "
                f"selected_only={len(a-b)}; frozen_only={len(b-a)}"
            )
    return canonical, CANONICAL_MANIFEST


def expected_jobs() -> list[str]:
    manifest_path = INPUT_ROOT / "celloracle_input_manifest.csv"
    if manifest_path.is_file():
        frame = pd.read_csv(manifest_path)
        if {"dataset", "celltype"}.issubset(frame.columns):
            return sorted((frame["dataset"].astype(str) + "__" + frame["celltype"].astype(str)).unique())
    required = {"counts_genes_by_cells.mtx", "genes.tsv", "barcodes.tsv", "cell_metadata.csv"}
    if not INPUT_ROOT.is_dir():
        return []
    return sorted(
        path.name for path in INPUT_ROOT.iterdir()
        if path.is_dir() and required.issubset({child.name for child in path.iterdir()})
    )


def jobs_from_missing_csv(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if not {"dataset", "celltype"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain dataset and celltype columns.")
    return sorted((frame["dataset"].astype(str) + "__" + frame["celltype"].astype(str)).unique())


def select_jobs(all_jobs: list[str], args: argparse.Namespace) -> list[str]:
    requested = set(args.job)
    if args.missing_jobs_csv:
        requested.update(jobs_from_missing_csv(Path(args.missing_jobs_csv).expanduser().resolve()))
    if not requested:
        return all_jobs
    unknown = sorted(requested.difference(all_jobs))
    if unknown:
        raise ValueError(f"Requested jobs are not present in prepared inputs: {unknown}")
    return [job for job in all_jobs if job in requested]


def sync_targets(jobs: list[str], canonical: pd.DataFrame) -> None:
    audit_rows: list[dict[str, object]] = []
    for job in jobs:
        if "__" not in job:
            raise ValueError(f"Invalid job name: {job}")
        dataset, celltype = job.split("__", 1)
        input_dir = INPUT_ROOT / job
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Prepared input directory not found: {input_dir}")
        rows = canonical[canonical["celltype"].eq(celltype)].copy()
        if rows.empty:
            raise ValueError(f"Canonical manifest contains no TF pairs for {celltype}.")
        rows["dataset"] = dataset
        rows["gene"] = rows["tf_gene"]
        preferred = ["target_id", "dataset", "celltype", "anchor_gene", "tf_gene", "gene"]
        columns = [column for column in preferred if column in rows.columns]
        columns += [column for column in rows.columns if column not in columns]
        target_path = input_dir / "targets_celloracle.csv"
        old_sha = file_digest(target_path) if target_path.is_file() else ""
        rows[columns].to_csv(target_path, index=False)
        new_sha = file_digest(target_path)
        audit_rows.append({
            "job": job,
            "dataset": dataset,
            "celltype": celltype,
            "n_pairs_written": int(len(rows)),
            "target_table": str(target_path),
            "old_sha256": old_sha,
            "new_sha256": new_sha,
            "changed": old_sha != new_sha,
        })
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(RESULT_ROOT / "celloracle_target_sync_audit.csv", index=False)


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if frame.empty:
        return set()
    return set(map(tuple, frame[PAIR_COLS].astype(str).to_numpy()))


def read_test_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    if not path.is_file():
        return set()
    try:
        header = pd.read_csv(path, nrows=0)
        if not set(TEST_COLS).issubset(header.columns):
            return set()
        frame = pd.read_csv(path, usecols=TEST_COLS)
    except Exception as exc:
        print(f"WARNING: could not read output keys from {path}: {exc}")
        return set()
    for column in TEST_COLS:
        frame[column] = frame[column].astype(str).str.strip()
    return set(map(tuple, frame[TEST_COLS].drop_duplicates().to_numpy()))


def expected_test_set(dataset: str, celltype: str, pairs: pd.DataFrame) -> set[tuple[str, str, str, str, str, str]]:
    result: set[tuple[str, str, str, str, str, str]] = set()
    for row in pairs.itertuples(index=False):
        for condition in REQUIRED_CONDITIONS:
            for perturbation in PERTURBATIONS:
                result.add((dataset, celltype, condition, str(row.anchor_gene), str(row.tf_gene), perturbation))
    return result


def format_test_key(key: tuple[str, str, str, str, str, str]) -> dict[str, str]:
    return dict(zip(TEST_COLS, key))


def read_skipped_pairs(
    path: Path,
    canonical_set: set[tuple[str, str, str]],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]], pd.DataFrame]:
    """Split skipped canonical pairs into formal non-evaluable and hard failures."""
    if not path.is_file():
        empty = pd.DataFrame(columns=[*PAIR_COLS, "reason"])
        return set(), set(), empty
    try:
        raw = pd.read_csv(path)
        normalized = normalize_manifest(raw, path)
        if "reason" in raw.columns:
            reason_map = (
                raw.assign(
                    celltype=raw["celltype"].astype(str).str.strip(),
                    anchor_gene=raw["anchor_gene"].astype(str).str.strip(),
                    tf_gene=(raw["tf_gene"] if "tf_gene" in raw.columns else raw["gene"]).astype(str).str.strip(),
                    reason=raw["reason"].astype(str).str.strip(),
                )
                [[*PAIR_COLS, "reason"]]
                .drop_duplicates(PAIR_COLS)
            )
        else:
            reason_map = normalized.assign(reason="unknown_ineligible_reason")[[*PAIR_COLS, "reason"]]
        reason_map = reason_map[
            reason_map[PAIR_COLS].apply(tuple, axis=1).isin(canonical_set)
        ].copy()
        non_eval = set(
            map(
                tuple,
                reason_map.loc[reason_map["reason"].isin(NON_EVALUABLE_REASONS), PAIR_COLS].to_numpy(),
            )
        )
        hard = set(
            map(
                tuple,
                reason_map.loc[~reason_map["reason"].isin(NON_EVALUABLE_REASONS), PAIR_COLS].to_numpy(),
            )
        )
        return non_eval, hard, reason_map
    except Exception as exc:
        print(f"WARNING: could not classify skipped targets from {path}: {exc}")
        # An unreadable skipped file cannot be treated as a benign exclusion.
        return set(), set(canonical_set), pd.DataFrame(columns=[*PAIR_COLS, "reason"])


def audit(jobs: list[str], canonical: pd.DataFrame, required_min_expressing_cells: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    non_evaluable_rows: list[dict[str, object]] = []
    hard_ineligible_rows: list[dict[str, object]] = []

    for job in jobs:
        input_dir = INPUT_ROOT / job
        output_dir = RESULT_ROOT / job
        dataset, celltype = job.split("__", 1) if "__" in job else ("unknown", job)
        target_path = input_dir / "targets_celloracle.csv"
        canonical_pairs = canonical[canonical["celltype"].eq(celltype)][PAIR_COLS].drop_duplicates()
        canonical_set = pair_set(canonical_pairs)

        if target_path.is_file():
            try:
                target_frame = normalize_manifest(pd.read_csv(target_path), target_path)
                target_set = pair_set(target_frame)
            except Exception as exc:
                print(f"WARNING: invalid target table {target_path}: {exc}")
                target_set = set()
        else:
            target_set = set()

        missing_target_pairs = canonical_set.difference(target_set)
        extra_target_pairs = target_set.difference(canonical_set)
        target_manifest_matches = target_set == canonical_set and bool(canonical_set)

        metadata = load_metadata(output_dir / "run_metadata.json")
        status = str(metadata.get("status", "not_run"))
        current_signature = input_signature(input_dir)
        stored_signature = metadata.get("input_signature_sha256")
        signature_matches = bool(current_signature and stored_signature == current_signature)
        try:
            stored_min_expressing_cells = int(metadata.get("min_expressing_cells"))
        except (TypeError, ValueError):
            stored_min_expressing_cells = None
        threshold_matches = stored_min_expressing_cells == int(required_min_expressing_cells)

        gene_file = output_dir / "celloracle_gene_deltas.csv.gz"
        summary_file = output_dir / "celloracle_perturbation_summary.csv"
        canonical_tests = expected_test_set(dataset, celltype, canonical_pairs)
        gene_tests = read_test_keys(gene_file)
        summary_tests = read_test_keys(summary_file)

        skipped_path = output_dir / "skipped_targets.csv"
        non_evaluable_pairs, hard_ineligible_pairs, skipped_detail = read_skipped_pairs(
            skipped_path, canonical_set
        )
        non_evaluable_tests = expected_test_set(
            dataset,
            celltype,
            canonical_pairs[
                canonical_pairs[PAIR_COLS].apply(tuple, axis=1).isin(non_evaluable_pairs)
            ],
        )
        evaluable_tests = canonical_tests.difference(non_evaluable_tests)
        missing_gene_tests = evaluable_tests.difference(gene_tests)
        missing_summary_tests = evaluable_tests.difference(summary_tests)

        runtime_skipped_path = output_dir / "runtime_skipped_targets.csv"
        runtime_skipped_pairs: set[tuple[str, str, str]] = set()
        if runtime_skipped_path.is_file():
            try:
                runtime_skipped = normalize_manifest(pd.read_csv(runtime_skipped_path), runtime_skipped_path)
                runtime_skipped_pairs = pair_set(runtime_skipped).intersection(canonical_set)
            except Exception:
                runtime_skipped_pairs = set(canonical_set)

        completed = (
            status in COMPLETED_STATUSES
            and target_manifest_matches
            and signature_matches
            and threshold_matches
            and gene_file.is_file()
            and summary_file.is_file()
            and not missing_gene_tests
            and not missing_summary_tests
            and not hard_ineligible_pairs
            and not runtime_skipped_pairs
        )

        reasons: list[str] = []
        notes: list[str] = []
        if not target_manifest_matches:
            reasons.append("target_manifest_mismatch")
        if not signature_matches:
            reasons.append("input_signature_mismatch")
        if not threshold_matches:
            reasons.append(
                f"min_expressing_cells_mismatch={stored_min_expressing_cells}!={required_min_expressing_cells}"
            )
        if status not in COMPLETED_STATUSES:
            reasons.append(f"run_status={status}")
        if missing_gene_tests:
            reasons.append(f"missing_evaluable_gene_tests={len(missing_gene_tests)}")
        if missing_summary_tests:
            reasons.append(f"missing_evaluable_summary_tests={len(missing_summary_tests)}")
        if hard_ineligible_pairs:
            reasons.append(f"hard_ineligible_pairs={len(hard_ineligible_pairs)}")
        if runtime_skipped_pairs:
            reasons.append(f"runtime_skipped_pairs={len(runtime_skipped_pairs)}")
        if non_evaluable_pairs:
            notes.append(f"non_evaluable_low_expression_pairs={len(non_evaluable_pairs)}")

        rows.append({
            "job": job,
            "dataset": dataset,
            "celltype": celltype,
            "has_prepared_input": input_dir.is_dir(),
            "has_targets_celloracle": target_path.is_file(),
            "canonical_pair_count": len(canonical_set),
            "target_pair_count": len(target_set),
            "missing_target_pair_count": len(missing_target_pairs),
            "extra_target_pair_count": len(extra_target_pairs),
            "target_manifest_matches": target_manifest_matches,
            "run_status": status,
            "input_signature_matches": signature_matches,
            "stored_min_expressing_cells": stored_min_expressing_cells,
            "required_min_expressing_cells": int(required_min_expressing_cells),
            "min_expressing_cells_matches": threshold_matches,
            "n_targets_requested_metadata": metadata.get("n_targets_requested"),
            "n_tf_targets_eligible_metadata": metadata.get("n_tf_targets_eligible"),
            "n_tf_targets_simulated_metadata": metadata.get("n_tf_targets_simulated"),
            "n_canonical_tests": len(canonical_tests),
            "n_evaluable_tests": len(evaluable_tests),
            "n_non_evaluable_tests": len(non_evaluable_tests),
            # Backward-compatible name: formal expected tests are evaluable tests.
            "n_expected_tests": len(evaluable_tests),
            "n_gene_tests_present": len(evaluable_tests.intersection(gene_tests)),
            "n_summary_tests_present": len(evaluable_tests.intersection(summary_tests)),
            "n_missing_gene_tests": len(missing_gene_tests),
            "n_missing_summary_tests": len(missing_summary_tests),
            "n_non_evaluable_expected_pairs": len(non_evaluable_pairs),
            "n_hard_ineligible_expected_pairs": len(hard_ineligible_pairs),
            "n_ineligible_expected_pairs": len(hard_ineligible_pairs),
            "n_runtime_skipped_expected_pairs": len(runtime_skipped_pairs),
            "has_gene_deltas": gene_file.is_file(),
            "has_perturbation_summary": summary_file.is_file(),
            "completed": completed,
            "audit_note": ";".join(notes),
            "incomplete_reason": ";".join(reasons),
        })

        reason_by_pair = {
            (str(r.celltype), str(r.anchor_gene), str(r.tf_gene)): str(r.reason)
            for r in skipped_detail.itertuples(index=False)
        }
        for key in sorted(non_evaluable_tests):
            item = format_test_key(key)
            pair = (item["celltype"], item["anchor_gene"], item["tf_gene"])
            item.update({
                "job": job,
                "evaluable": False,
                "non_evaluable_reason": reason_by_pair.get(pair, "TF_below_expression_threshold"),
            })
            non_evaluable_rows.append(item)

        for pair in sorted(hard_ineligible_pairs):
            hard_ineligible_rows.append({
                "job": job,
                "dataset": dataset,
                "celltype": pair[0],
                "anchor_gene": pair[1],
                "tf_gene": pair[2],
                "reason": reason_by_pair.get(pair, "unknown_ineligible_reason"),
            })

        for key in sorted(missing_gene_tests.union(missing_summary_tests)):
            item = format_test_key(key)
            item.update({
                "job": job,
                "evaluable": True,
                "gene_delta_present": key in gene_tests,
                "summary_present": key in summary_tests,
            })
            missing_rows.append(item)

    frame = pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(AUDIT_PATH, index=False)
    pd.DataFrame(
        missing_rows,
        columns=["job", *TEST_COLS, "evaluable", "gene_delta_present", "summary_present"],
    ).to_csv(MISSING_TEST_PATH, index=False)
    pd.DataFrame(
        non_evaluable_rows,
        columns=["job", *TEST_COLS, "evaluable", "non_evaluable_reason"],
    ).to_csv(NON_EVALUABLE_TEST_PATH, index=False)
    pd.DataFrame(
        hard_ineligible_rows,
        columns=["job", "dataset", "celltype", "anchor_gene", "tf_gene", "reason"],
    ).to_csv(HARD_INELIGIBLE_PATH, index=False)
    return frame


def run_one(job: str, args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        str(RUNNER),
        "--job",
        job,
        "--n-jobs",
        str(args.n_jobs),
        "--min-expressing-cells",
        str(args.min_expressing_cells),
    ]
    print(f"Preflight: {job}")
    subprocess.run(base + ["--check-only"], check=True)
    command = base + (["--force-rebuild"] if args.force_rebuild else [])
    print(f"Running: {job}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    canonical, canonical_path = read_canonical_manifest()
    all_jobs = expected_jobs()
    if not all_jobs:
        raise SystemExit(f"No prepared CellOracle jobs found under {INPUT_ROOT}")
    jobs = select_jobs(all_jobs, args)
    print(f"Canonical TF manifest: {canonical_path} ({len(canonical)} anchor-TF pairs)")
    print(f"Dataset-celltype jobs selected: {len(jobs)}")

    if args.run and not args.no_sync_targets:
        print("Synchronizing targets_celloracle.csv from the canonical manifest...")
        sync_targets(jobs, canonical)

    first = audit(jobs, canonical, args.min_expressing_cells)
    print(f"Audit written: {AUDIT_PATH}")
    display_cols = [
        "job", "canonical_pair_count", "target_pair_count",
        "n_canonical_tests", "n_evaluable_tests", "n_non_evaluable_tests",
        "n_missing_summary_tests", "completed", "audit_note", "incomplete_reason",
    ]
    print(first[display_cols].to_string(index=False))

    if not args.run:
        incomplete = first[~first["completed"]]
        if len(incomplete):
            print(f"Incomplete dataset-celltype jobs: {len(incomplete)}/{len(first)}")
        return

    for row in first.itertuples(index=False):
        if bool(row.completed):
            print(f"Skipping completed job: {row.job}")
            continue
        if not bool(row.has_targets_celloracle):
            raise SystemExit(f"Missing targets_celloracle.csv for {row.job} after synchronization.")
        run_one(str(row.job), args)

    final = audit(jobs, canonical, args.min_expressing_cells)
    incomplete = final[~final["completed"]]
    if len(incomplete):
        raise SystemExit(
            f"CellOracle remains incomplete for {len(incomplete)} dataset-celltype jobs. "
            f"Review {AUDIT_PATH} and {MISSING_TEST_PATH}."
        )
    print("All selected CellOracle jobs contain every evaluable test; low-expression non-evaluable tests are recorded separately.")


if __name__ == "__main__":
    main()
