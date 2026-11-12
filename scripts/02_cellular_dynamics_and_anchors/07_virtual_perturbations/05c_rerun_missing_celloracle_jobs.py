#!/usr/bin/env python3
"""Rerun unique dataset__celltype CellOracle jobs from step-06 missing keys.

The long table may contain many missing output keys, but the current CellOracle
runner operates at dataset-celltype granularity and rewrites aggregate outputs.
Low-expression canonical targets are handled as formally non-evaluable by steps
05/05b/06 and must not be resubmitted as missing work.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
ORCHESTRATOR = SCRIPT_ROOT / "05b_run_or_audit_all_celloracle_jobs.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-jobs-csv", required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--min-expressing-cells", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    path = Path(args.missing_jobs_csv).expanduser().resolve()
    frame = pd.read_csv(path)
    if not {"dataset", "celltype"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain dataset and celltype columns.")
    if frame.empty:
        print(f"No evaluable missing jobs are listed in {path}. Nothing to rerun.")
        return
    jobs = sorted((frame["dataset"].astype(str) + "__" + frame["celltype"].astype(str)).unique())
    print(f"Missing output keys: {len(frame)}")
    print(f"Unique dataset-celltype executor jobs: {len(jobs)}")
    for job in jobs:
        print(job)

    command = [
        sys.executable,
        str(ORCHESTRATOR),
        "--missing-jobs-csv",
        str(path),
        "--n-jobs",
        str(args.n_jobs),
        "--min-expressing-cells",
        str(args.min_expressing_cells),
    ]
    if not args.audit_only:
        command.append("--run")
    if args.force_rebuild:
        command.append("--force-rebuild")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
