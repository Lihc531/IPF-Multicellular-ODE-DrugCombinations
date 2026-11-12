#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal multi-seed AutoDock Vina validation layered onto the original workflow.

Subcommands:
  audit         Validate PDBQT files, box definitions, native controls and Vina version.
  run           Execute fixed-seed formal docking with resumable complete logs.
  summarize     Summarize affinity and receptor-frame pose consistency; validate redocking.
  interactions  Audit representative-pose contacts without over-counting multiple MODEL blocks.
  grade         Assign structure-level Vina evidence grades and re-aggregate to drug pairs.
  sync          Copy representative poses into the original results/pairs layout for B-style figures.

The pipeline evaluates structural plausibility. It is not a binding assay and it
cannot establish Bliss/Loewe/ZIP/HSA synergy for a drug pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_SEEDS = [104729, 130363, 155921, 196613, 262147]
AFFINITY_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+")
VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "selected", "success"}


STANDARD_VINA_MODES = {
    "", "standard", "standard_vina", "standard_non_covalent", "noncovalent",
    "non_covalent", "cross_docking", "native_redocking",
}
SPECIAL_DOCKING_TOKENS = (
    "covalent", "reference_only", "manual_review", "unsupported",
    "metal_coordination", "macrocycle_special", "flexible_receptor_required",
)


def normalize_docking_mode(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_special_mode(value: str) -> bool:
    if not value or value in STANDARD_VINA_MODES:
        return False
    return any(value == token or value.startswith(token + "_") for token in SPECIAL_DOCKING_TOKENS)


def standard_vina_mode_supported(row: object) -> tuple[bool, str]:
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    mode = normalize_docking_mode(get("docking_mode", ""))
    dtype = normalize_docking_mode(get("docking_type", ""))
    if _is_special_mode(mode) or _is_special_mode(dtype):
        return False, mode or dtype or "special_protocol_required"
    if mode in STANDARD_VINA_MODES or not mode:
        return True, mode or "standard_non_covalent"
    return False, mode or "unknown_docking_mode"


def safe_id(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_.") or "task"


def resolve(project: Path, value: object) -> Path:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return Path("")
    text = os.path.expandvars(os.path.expanduser(text))
    p = Path(text)
    return p if p.is_absolute() else project / p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(project: Path, config_name: str) -> dict:
    path = resolve(project, config_name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def config_value(cfg: dict, key: str, default):
    value = cfg.get(key, default)
    return default if value in (None, "") else value


def vina_executable(project: Path, cfg: dict, override: str = "") -> Path:
    raw = override or str(cfg.get("vina_exe", "")).strip()
    if raw:
        p = resolve(project, raw)
        if p.exists():
            return p
    found = shutil.which(raw or "vina") or shutil.which("vina.exe")
    return Path(found) if found else resolve(project, raw or "vina")


def vina_version(executable: Path) -> tuple[str, Optional[tuple[int, int, int]]]:
    try:
        cp = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=20)
        text = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
    except Exception as exc:
        return f"unavailable: {exc}", None
    m = VERSION_RE.search(text)
    return text or "unknown", tuple(map(int, m.groups())) if m else None


def parse_seeds(text: object) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    values = [x.strip() for x in str(text or "").replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in values] if values else list(DEFAULT_SEEDS)


def first_model_atoms(path: Path, heavy_only: bool = False) -> list[dict]:
    atoms: list[dict] = []
    if not path.is_file():
        return atoms
    model_seen = False
    current_model = 1
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MODEL"):
            model_seen = True
            try:
                current_model = int(line.split()[1])
            except Exception:
                current_model = 1
            if current_model > 1:
                break
            continue
        if line.startswith("ENDMDL") and (not model_seen or current_model == 1):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except Exception:
            parts = line.split()
            try:
                x, y, z = map(float, parts[6:9])
            except Exception:
                continue
        atom_type = line.split()[-1] if line.split() else ""
        element = re.sub(r"[^A-Za-z]", "", atom_type).upper()
        if not element:
            element = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()
        if heavy_only and element.startswith("H"):
            continue
        charge = np.nan
        try:
            charge = float(line[70:76])
        except Exception:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    charge = float(parts[-2])
                except Exception:
                    pass
        atoms.append({
            "serial": line[6:11].strip(),
            "atom_name": line[12:16].strip(),
            "resn": line[17:20].strip(),
            "chain": line[21:22].strip(),
            "resi": line[22:26].strip(),
            "x": x, "y": y, "z": z,
            "charge": charge,
            "atom_type": atom_type,
            "element": element,
        })
    return atoms


def atom_signature(atoms: Sequence[dict]) -> tuple[str, ...]:
    return tuple(f"{a['atom_name']}|{a['atom_type']}" for a in atoms)


def coords(atoms: Sequence[dict]) -> np.ndarray:
    return np.asarray([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)


def direct_rmsd(a_atoms: Sequence[dict], b_atoms: Sequence[dict]) -> float:
    """RMSD in the receptor coordinate frame; no ligand self-superposition."""
    if not a_atoms or len(a_atoms) != len(b_atoms):
        return np.nan
    if atom_signature(a_atoms) != atom_signature(b_atoms):
        return np.nan
    a = coords(a_atoms); b = coords(b_atoms)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def best_affinity(log_path: Path, pose_path: Optional[Path] = None) -> float:
    values: list[float] = []
    for path in [log_path, pose_path]:
        if not path or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = AFFINITY_RE.match(line)
            if m:
                values.append(float(m.group(1)))
            if "REMARK VINA RESULT:" in line:
                try:
                    values.append(float(line.split("RESULT:", 1)[1].split()[0]))
                except Exception:
                    pass
    return min(values) if values else np.nan


def box_values(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    values = []
    for c in ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]:
        values.append(float(pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]))
    return tuple(values)  # type: ignore[return-value]


def atoms_inside_box(atoms: Sequence[dict], box: tuple[float, float, float, float, float, float]) -> bool:
    if not atoms:
        return False
    cx, cy, cz, sx, sy, sz = box
    xyz = coords(atoms)
    mins = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2])
    maxs = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
    return bool(np.all((xyz >= mins) & (xyz <= maxs)))


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def cmd_audit(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    manifest_path = resolve(project, args.manifest)
    output_dir = resolve(project, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {"pair_id", "receptor", "ligand", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
    ensure_columns(manifest, required, "formal manifest")
    if "enabled" in manifest.columns:
        enabled_mask = manifest["enabled"].map(truthy).astype(bool)
        manifest = manifest.loc[enabled_mask].copy()
    if manifest.empty:
        extra_columns = [
            "audit_status", "fatal_reasons", "warnings", "normalized_docking_mode",
            "standard_vina_mode_supported", "receptor_resolved", "ligand_resolved",
            "native_ligand_resolved", "receptor_sha256", "ligand_sha256",
            "native_ligand_sha256", "n_receptor_atoms", "n_ligand_atoms", "vina_version",
        ]
        empty_audit = pd.DataFrame(columns=list(manifest.columns) + [c for c in extra_columns if c not in manifest.columns])
        audit_file = output_dir / "docking_input_audit.csv"
        formal_file = output_dir / "formal_docking_manifest_audited.csv"
        empty_audit.to_csv(audit_file, index=False, encoding="utf-8-sig")
        manifest.to_csv(formal_file, index=False, encoding="utf-8-sig")
        metadata = {
            "n_input": 0, "n_pass": 0, "n_fail": 0,
            "status": "blocked_empty_manifest",
            "message": "No enabled formal docking tasks. Prepare ligands, update mappings, and rebuild the formal manifest first.",
            "standard_vina_scope": "non-covalent small-molecule docking only",
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f"Audit: {audit_file}\nAudited manifest: {formal_file}")
        return 2
    exe = vina_executable(project, cfg, args.vina)
    version_text, version_tuple = vina_version(exe)
    min_version = tuple(map(int, str(config_value(cfg, "formal_min_vina_version", "1.2.5")).split(".")))
    rows = []
    audited = []
    for _, row in manifest.iterrows():
        pair_id = safe_id(row["pair_id"])
        fatal: list[str] = []
        warnings: list[str] = []
        mode_supported, normalized_mode = standard_vina_mode_supported(row)
        if not mode_supported:
            fatal.append(f"unsupported_standard_vina_mode:{normalized_mode}")
        rec = resolve(project, row["receptor"])
        lig = resolve(project, row["ligand"])
        native = resolve(project, row.get("native_ligand_pdbqt", ""))
        try:
            box = box_values(row)
        except Exception:
            box = (np.nan,) * 6  # type: ignore[assignment]
            fatal.append("invalid_box_values")
        if not rec.is_file(): fatal.append("receptor_missing")
        if not lig.is_file(): fatal.append("ligand_missing")
        sizes = np.asarray(box[3:], dtype=float)
        if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            fatal.append("invalid_box_size")
        elif np.any(sizes > 30):
            warnings.append("large_search_box_consider_higher_exhaustiveness")
        rec_atoms = first_model_atoms(rec) if rec.is_file() else []
        lig_atoms = first_model_atoms(lig) if lig.is_file() else []
        if rec.is_file() and not rec_atoms: fatal.append("receptor_has_no_atoms")
        if lig.is_file() and not lig_atoms: fatal.append("ligand_has_no_atoms")
        lig_text = lig.read_text(encoding="utf-8", errors="ignore") if lig.is_file() else ""
        if lig.is_file() and "ROOT" not in lig_text: warnings.append("ligand_ROOT_not_detected")
        if lig.is_file() and "TORSDOF" not in lig_text: warnings.append("ligand_TORSDOF_not_detected")
        if lig_atoms and all(not np.isfinite(a["charge"]) for a in lig_atoms):
            warnings.append("ligand_partial_charges_not_parsed")
        docking_type = str(row.get("docking_type", "cross_docking"))
        native_required = "native" in docking_type.lower()
        if native_required and not native.is_file():
            fatal.append("native_reference_missing")
        native_atoms = first_model_atoms(native, heavy_only=True) if native.is_file() else []
        if native.is_file() and not native_atoms:
            fatal.append("native_reference_has_no_atoms")
        if native_atoms and np.all(np.isfinite(box)) and not atoms_inside_box(native_atoms, box):
            fatal.append("native_ligand_not_fully_inside_box")
        if version_tuple is None:
            warnings.append("vina_version_not_parsed")
        elif version_tuple < min_version:
            fatal.append(f"vina_version_below_{'.'.join(map(str, min_version))}")
        if not exe.exists() and shutil.which(str(exe)) is None:
            fatal.append("vina_executable_missing")
        status = "pass" if not fatal else "fail"
        audit_row = row.to_dict()
        audit_row.update({
            "pair_id": pair_id,
            "audit_status": status,
            "fatal_reasons": ";".join(fatal),
            "warnings": ";".join(warnings),
            "normalized_docking_mode": normalized_mode,
            "standard_vina_mode_supported": mode_supported,
            "receptor_resolved": str(rec.resolve()) if rec.exists() else str(rec),
            "ligand_resolved": str(lig.resolve()) if lig.exists() else str(lig),
            "native_ligand_resolved": str(native.resolve()) if native.exists() else (str(native) if str(native) not in {"", "."} else ""),
            "receptor_sha256": file_sha256(rec) if rec.is_file() else "",
            "ligand_sha256": file_sha256(lig) if lig.is_file() else "",
            "native_ligand_sha256": file_sha256(native) if native.is_file() else "",
            "n_receptor_atoms": len(rec_atoms),
            "n_ligand_atoms": len(lig_atoms),
            "vina_version": version_text,
        })
        rows.append(audit_row)
        if status == "pass": audited.append(audit_row)
    audit_df = pd.DataFrame(rows)
    pass_df = pd.DataFrame(audited)
    audit_file = output_dir / "docking_input_audit.csv"
    formal_file = output_dir / "formal_docking_manifest_audited.csv"
    audit_df.to_csv(audit_file, index=False, encoding="utf-8-sig")
    pass_df.to_csv(formal_file, index=False, encoding="utf-8-sig")
    metadata = {
        "n_input": int(len(manifest)), "n_pass": int(len(pass_df)), "n_fail": int(len(manifest) - len(pass_df)),
        "vina_executable": str(exe), "vina_version": version_text,
        "formal_min_vina_version": ".".join(map(str, min_version)),
        "n_unsupported_special_modes": int(audit_df.get("standard_vina_mode_supported", pd.Series(dtype=bool)).eq(False).sum()),
        "standard_vina_scope": "non-covalent small-molecule docking only",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Audit: {audit_file}\nAudited manifest: {formal_file}")
    return 1 if len(pass_df) != len(manifest) else 0


def run_one(job: dict) -> dict:
    start = time.time()
    pair_id = job["pair_id"]; seed = job["seed"]
    out_dir: Path = job["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = out_dir / f"{pair_id}_out.pdbqt"
    log = out_dir / f"{pair_id}.log"
    command = job["command"] + ["--out", str(pose)]
    status = "dry_run"; rc = None; message = "dry run only"
    if job["dry_run"]:
        log.write_text("DRY RUN\n" + subprocess.list2cmdline(command), encoding="utf-8")
    elif job["resume"] and pose.is_file() and np.isfinite(best_affinity(log, pose)):
        status = "skipped_existing"; message = "existing complete result"
    else:
        try:
            cp = subprocess.run(command, capture_output=True, text=True, timeout=job["timeout"])
            rc = cp.returncode
            log.write_text((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else ""), encoding="utf-8", errors="replace")
            if rc == 0 and pose.is_file() and np.isfinite(best_affinity(log, pose)):
                status = "success"; message = "completed"
            else:
                status = "failed"; message = f"Vina exit={rc}; pose={pose.exists()}"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"; message = f"timeout after {job['timeout']} s"
            log.write_text(str(exc), encoding="utf-8", errors="replace")
        except Exception as exc:
            status = "failed"; message = str(exc)
            log.write_text(str(exc), encoding="utf-8", errors="replace")
    return {
        "pair_id": pair_id, "seed": seed, "status": status, "return_code": rc,
        "elapsed_seconds": round(time.time() - start, 3), "pose_file": str(pose),
        "log_file": str(log), "command": subprocess.list2cmdline(command), "message": message,
    }


def cmd_run(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    manifest_path = resolve(project, args.manifest)
    output_dir = resolve(project, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    skipped_rows = []
    if "audit_status" in manifest.columns:
        manifest = manifest[manifest["audit_status"].astype(str).str.lower().eq("pass")].copy()
    if not manifest.empty:
        mode_checks = manifest.apply(lambda row: standard_vina_mode_supported(row)[0], axis=1)
        unsupported = manifest[~mode_checks].copy()
        for _, row in unsupported.iterrows():
            _, mode = standard_vina_mode_supported(row)
            skipped_rows.append({
                "pair_id": safe_id(row.get("pair_id", "task")), "seed": np.nan,
                "status": "skipped_unsupported_docking_mode", "return_code": np.nan,
                "elapsed_seconds": 0.0, "pose_file": "", "log_file": "", "command": "",
                "message": f"standard Vina pipeline does not support docking_mode={mode}",
            })
        manifest = manifest[mode_checks].copy()
    exe = vina_executable(project, cfg, args.vina)
    seeds = parse_seeds(args.seeds or config_value(cfg, "formal_seeds", DEFAULT_SEEDS))
    exhaustiveness = int(args.exhaustiveness or config_value(cfg, "formal_exhaustiveness", 32))
    num_modes = int(args.num_modes or config_value(cfg, "formal_num_modes", 20))
    energy_range = float(args.energy_range or config_value(cfg, "formal_energy_range", 5))
    cpu = int(args.cpu or config_value(cfg, "formal_cpu", 4))
    jobs = int(args.jobs or config_value(cfg, "formal_jobs", 1))
    timeout = int(args.timeout or config_value(cfg, "docking_timeout_seconds", 3600))
    jobs = max(1, jobs)
    job_list = []
    for _, row in manifest.iterrows():
        pair_id = safe_id(row["pair_id"])
        receptor = str(row.get("receptor_resolved", row.get("receptor", "")))
        ligand = str(row.get("ligand_resolved", row.get("ligand", "")))
        for seed in seeds:
            command = [
                str(exe), "--receptor", receptor, "--ligand", ligand,
                "--center_x", str(row["center_x"]), "--center_y", str(row["center_y"]), "--center_z", str(row["center_z"]),
                "--size_x", str(row["size_x"]), "--size_y", str(row["size_y"]), "--size_z", str(row["size_z"]),
                "--seed", str(seed), "--exhaustiveness", str(exhaustiveness),
                "--num_modes", str(num_modes), "--energy_range", str(energy_range),
                "--cpu", str(cpu), "--verbosity", "1",
            ]
            job_list.append({
                "pair_id": pair_id, "seed": seed,
                "out_dir": output_dir / pair_id / f"seed_{seed}", "command": command,
                "dry_run": args.dry_run, "resume": args.resume, "timeout": timeout,
            })
    results = list(skipped_rows)
    if jobs == 1:
        for i, job in enumerate(job_list, 1):
            print(f"[{i}/{len(job_list)}] {job['pair_id']} seed={job['seed']}")
            result = run_one(job); results.append(result); print(f"  {result['status']}: {result['message']}")
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_map = {pool.submit(run_one, job): job for job in job_list}
            for i, future in enumerate(as_completed(future_map), 1):
                result = future.result(); results.append(result)
                print(f"[{i}/{len(job_list)}] {result['pair_id']} seed={result['seed']} {result['status']}")
    run_columns = ["pair_id", "seed", "status", "return_code", "elapsed_seconds", "pose_file", "log_file", "command", "message"]
    run_df = pd.DataFrame(results, columns=run_columns)
    if not run_df.empty:
        run_df = run_df.sort_values(["pair_id", "seed"], na_position="last")
    run_file = output_dir / "multiseed_run_manifest.csv"
    run_df.to_csv(run_file, index=False, encoding="utf-8-sig")
    version_text, _ = vina_version(exe)
    metadata = {
        "n_tasks": int(manifest["pair_id"].nunique()), "n_runs": int(len(run_df)),
        "seeds": seeds, "exhaustiveness": exhaustiveness, "num_modes": num_modes,
        "energy_range": energy_range, "cpu_per_run": cpu, "parallel_jobs": jobs,
        "dry_run": bool(args.dry_run), "resume": bool(args.resume),
        "n_special_mode_skipped": int(len(skipped_rows)),
        "vina_executable": str(exe), "vina_version": version_text,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    blocking = run_df["status"].isin(["failed", "timeout", "skipped_unsupported_docking_mode"]) if not run_df.empty else pd.Series(dtype=bool)
    return 1 if blocking.any() else 0


def pairwise_pose_metrics(group: pd.DataFrame, threshold: float) -> tuple[str, float, float, dict[str, float]]:
    valid = []
    for _, row in group.iterrows():
        path = Path(str(row["pose_file"]))
        atoms = first_model_atoms(path, heavy_only=True)
        if atoms:
            valid.append((str(row["seed"]), path, atoms))
    if not valid:
        return "", np.nan, 0.0, {}
    if len(valid) == 1:
        seed, path, _ = valid[0]
        return str(path), 0.0, 1.0, {seed: 0.0}
    matrix = np.full((len(valid), len(valid)), np.nan)
    for i in range(len(valid)):
        matrix[i, i] = 0.0
        for j in range(i + 1, len(valid)):
            value = direct_rmsd(valid[i][2], valid[j][2])
            matrix[i, j] = matrix[j, i] = value
    medians = np.nanmedian(matrix, axis=1)
    if np.all(~np.isfinite(medians)):
        return "", np.nan, 0.0, {}
    medoid_idx = int(np.nanargmin(medians))
    distances = matrix[medoid_idx]
    fraction = float(np.mean(distances[np.isfinite(distances)] <= threshold)) if np.isfinite(distances).any() else 0.0
    by_seed = {valid[i][0]: float(distances[i]) if np.isfinite(distances[i]) else np.nan for i in range(len(valid))}
    return str(valid[medoid_idx][1]), float(medians[medoid_idx]), fraction, by_seed


def cmd_summarize(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    run_path = resolve(project, args.run_manifest)
    manifest_path = resolve(project, args.docking_manifest)
    output_dir = resolve(project, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(run_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    runs["best_affinity_kcal_mol"] = [best_affinity(Path(str(l)), Path(str(p))) for l, p in zip(runs["log_file"], runs["pose_file"])]
    pose_threshold = float(args.pose_rmsd_threshold or config_value(cfg, "formal_pose_rmsd_threshold", 2.0))
    redock_threshold = float(args.redocking_rmsd_threshold or config_value(cfg, "formal_redocking_rmsd_threshold", 2.0))
    min_completion = float(args.min_seed_completion or config_value(cfg, "formal_min_seed_completion", 0.8))
    max_iqr = float(args.max_affinity_iqr or config_value(cfg, "formal_max_affinity_iqr", 1.0))
    min_cluster = float(args.min_pose_cluster_fraction or config_value(cfg, "formal_min_pose_cluster_fraction", 0.6))
    redock_fraction_required = float(args.min_redocking_fraction or config_value(cfg, "formal_min_redocking_fraction", 0.6))

    manifest_index = manifest.set_index(manifest["pair_id"].astype(str), drop=False)
    pose_rows = []
    summary_rows = []
    for pair_id, group in runs.groupby("pair_id", sort=False):
        mrow = manifest_index.loc[str(pair_id)] if str(pair_id) in manifest_index.index else None
        if isinstance(mrow, pd.DataFrame): mrow = mrow.iloc[0]
        rep_pose, medoid_median_rmsd, cluster_fraction, medoid_dist = pairwise_pose_metrics(group, pose_threshold)
        native_path = Path("")
        if mrow is not None:
            native_value = mrow.get("native_ligand_resolved", mrow.get("native_ligand_pdbqt", ""))
            if str(native_value).strip() and str(native_value).strip().lower() != "nan":
                native_path = Path(str(native_value))
        native_atoms = first_model_atoms(native_path, heavy_only=True) if native_path.is_file() else []
        redocking_values = []
        for _, row in group.iterrows():
            pose_path = Path(str(row["pose_file"]))
            pose_atoms = first_model_atoms(pose_path, heavy_only=True)
            rr = direct_rmsd(native_atoms, pose_atoms) if native_atoms else np.nan
            redocking_values.append(rr)
            pose_rows.append({
                **row.to_dict(),
                "medoid_distance_angstrom": medoid_dist.get(str(row["seed"]), np.nan),
                "is_representative_medoid": bool(rep_pose and str(pose_path) == rep_pose),
                "redocking_rmsd_angstrom": rr,
                "redocking_rmsd_method": "same_order_direct_receptor_frame" if np.isfinite(rr) else "not_available_or_atom_mapping_mismatch",
            })
        affinities = pd.to_numeric(group["best_affinity_kcal_mol"], errors="coerce")
        n_seeds = int(group["seed"].nunique())
        n_success = int(affinities.notna().sum())
        iqr = float(affinities.quantile(0.75) - affinities.quantile(0.25)) if n_success else np.nan
        redocking_array = np.asarray(redocking_values, dtype=float)
        redocking_fraction = float(np.mean(redocking_array[np.isfinite(redocking_array)] <= redock_threshold)) if np.isfinite(redocking_array).any() else np.nan
        summary = {
            "pair_id": pair_id,
            "n_seeds": n_seeds, "n_success": n_success,
            "seed_completion_fraction": n_success / n_seeds if n_seeds else 0,
            "best_affinity": float(affinities.min()) if n_success else np.nan,
            "median_affinity": float(affinities.median()) if n_success else np.nan,
            "affinity_iqr": iqr,
            "affinity_sd": float(affinities.std()) if n_success > 1 else 0.0 if n_success == 1 else np.nan,
            "representative_pose_file": rep_pose,
            "medoid_median_rmsd_angstrom": medoid_median_rmsd,
            "medoid_cluster_fraction_within_threshold": cluster_fraction,
            "pose_rmsd_threshold_angstrom": pose_threshold,
            "best_redocking_rmsd": float(np.nanmin(redocking_array)) if np.isfinite(redocking_array).any() else np.nan,
            "median_redocking_rmsd": float(np.nanmedian(redocking_array)) if np.isfinite(redocking_array).any() else np.nan,
            "redocking_fraction_within_threshold": redocking_fraction,
        }
        summary["multiseed_stable"] = bool(
            summary["seed_completion_fraction"] >= min_completion
            and np.isfinite(iqr) and iqr <= max_iqr
            and cluster_fraction >= min_cluster
        )
        summary["redocking_pose_recovered"] = bool(
            np.isfinite(summary["best_redocking_rmsd"])
            and summary["best_redocking_rmsd"] <= redock_threshold
            and np.isfinite(redocking_fraction)
            and redocking_fraction >= redock_fraction_required
        )
        if mrow is not None:
            for c in ["target_gene", "pdb_id", "receptor_id", "drug_id", "drug", "docking_type",
                      "receptor_resolved", "ligand_resolved", "source_pair_ids", "source_pair_memberships",
                      "member_roles", "bridge_score", "evidence_tier"]:
                if c in mrow.index: summary[c] = mrow[c]
        summary_rows.append(summary)
    pose_df = pd.DataFrame(pose_rows)
    summary_df = pd.DataFrame(summary_rows)
    pose_df.to_csv(output_dir / "multiseed_pose_level_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "vina_multiseed_summary.csv", index=False, encoding="utf-8-sig")

    native = summary_df[summary_df.get("docking_type", pd.Series(index=summary_df.index, dtype=str)).astype(str).str.contains("native", case=False, na=False)].copy()
    target_rows = []
    if not native.empty:
        for receptor_id, g in native.groupby("receptor_id", dropna=False):
            best = pd.to_numeric(g["best_redocking_rmsd"], errors="coerce").min()
            frac = pd.to_numeric(g["redocking_fraction_within_threshold"], errors="coerce").max()
            passed = bool(np.isfinite(best) and best <= redock_threshold and np.isfinite(frac) and frac >= redock_fraction_required)
            target_rows.append({
                "receptor_id": receptor_id,
                "target_gene": ";".join(sorted(set(g.get("target_gene", pd.Series(dtype=str)).dropna().astype(str)))),
                "pdb_id": ";".join(sorted(set(g.get("pdb_id", pd.Series(dtype=str)).dropna().astype(str)))),
                "n_native_redocking_tasks": int(len(g)),
                "best_redocking_rmsd": best,
                "best_redocking_fraction_within_threshold": frac,
                "target_redocking_pass": passed,
                "redocking_rmsd_threshold_angstrom": redock_threshold,
            })
    target_df = pd.DataFrame(target_rows, columns=[
        "receptor_id", "target_gene", "pdb_id", "n_native_redocking_tasks",
        "best_redocking_rmsd", "best_redocking_fraction_within_threshold",
        "target_redocking_pass", "redocking_rmsd_threshold_angstrom",
    ])
    target_df.to_csv(output_dir / "target_redocking_validation.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "n_tasks": int(len(summary_df)), "n_stable": int(summary_df.get("multiseed_stable", False).sum()),
        "n_target_structures_with_redocking": int(len(target_df)),
        "n_target_structures_redocking_pass": int(target_df.get("target_redocking_pass", pd.Series(dtype=bool)).sum()),
        "thresholds": {
            "min_seed_completion": min_completion, "max_affinity_iqr": max_iqr,
            "pose_rmsd_threshold_angstrom": pose_threshold, "min_pose_cluster_fraction": min_cluster,
            "redocking_rmsd_threshold_angstrom": redock_threshold,
            "min_redocking_fraction": redock_fraction_required,
        },
        "rmsd_note": "RMSD is calculated directly in the receptor coordinate frame. No ligand self-superposition is applied. Same atom order/type is required; symmetry-aware external validation is recommended for publication.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def polar_element(atom: dict) -> bool:
    return atom["element"].startswith(("N", "O", "S"))


def cmd_interactions(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    summary_path = resolve(project, args.multiseed_summary)
    manifest_path = resolve(project, args.docking_manifest)
    output_dir = resolve(project, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    merged = summary.merge(manifest, on="pair_id", how="left", suffixes=("", "_manifest"))
    contact_cutoff = float(args.contact_cutoff)
    polar_cutoff = float(args.polar_cutoff)
    clash_cutoff = float(args.clash_cutoff)
    rows = []
    long_rows = []
    for _, row in merged.iterrows():
        pair_id = safe_id(row["pair_id"])
        pose = Path(str(row.get("representative_pose_file", "")))
        receptor = Path(str(row.get("receptor_resolved", row.get("receptor", ""))))
        lig_atoms = first_model_atoms(pose, heavy_only=True)
        rec_atoms = first_model_atoms(receptor, heavy_only=True)
        residue_min: dict[tuple[str, str, str], float] = {}
        polar_pairs = 0; clashes = 0; atom_contacts = 0
        if lig_atoms and rec_atoms:
            lig_xyz = coords(lig_atoms); rec_xyz = coords(rec_atoms)
            for i, ra in enumerate(rec_atoms):
                distances = np.sqrt(np.sum((lig_xyz - rec_xyz[i]) ** 2, axis=1))
                min_d = float(np.min(distances))
                key = (ra["resn"], ra["chain"], ra["resi"])
                if min_d <= contact_cutoff:
                    residue_min[key] = min(residue_min.get(key, np.inf), min_d)
                    atom_contacts += int(np.sum(distances <= contact_cutoff))
                clashes += int(np.sum(distances < clash_cutoff))
                if polar_element(ra):
                    for j, d in enumerate(distances):
                        if d <= polar_cutoff and polar_element(lig_atoms[j]):
                            polar_pairs += 1
            for (resn, chain, resi), distance in sorted(residue_min.items(), key=lambda x: x[1]):
                long_rows.append({
                    "pair_id": pair_id, "resn": resn, "chain": chain, "resi": resi,
                    "minimum_distance_angstrom": distance,
                })
        residue_labels = [f"{resn}{resi}{'_'+chain if chain else ''}" for resn, chain, resi in sorted(residue_min)]
        rows.append({
            "pair_id": pair_id,
            "representative_pose_file": str(pose),
            "receptor_file": str(receptor),
            "n_contact_residues": len(residue_min),
            "contact_residues": ";".join(residue_labels),
            "n_atom_contacts_within_cutoff": atom_contacts,
            "n_potential_polar_contacts": polar_pairs,
            "n_steric_clash_warnings": clashes,
            "contact_cutoff_angstrom": contact_cutoff,
            "polar_contact_cutoff_angstrom": polar_cutoff,
            "clash_cutoff_angstrom": clash_cutoff,
            "interaction_note": "Distance-only contact audit; potential polar contacts are not angle-validated hydrogen bonds.",
        })
    pd.DataFrame(rows).to_csv(output_dir / "vina_interaction_fingerprint_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long_rows).to_csv(output_dir / "vina_contact_residues_long.csv", index=False, encoding="utf-8-sig")
    metadata = {"n_tasks": len(rows), "contact_cutoff": contact_cutoff, "polar_cutoff": polar_cutoff, "clash_cutoff": clash_cutoff}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def best_grade(values: Iterable[str]) -> str:
    vals = [str(v) for v in values if str(v) in GRADE_ORDER]
    return min(vals, key=lambda x: GRADE_ORDER[x]) if vals else "D"


def parse_memberships(value: object) -> list[tuple[str, str]]:
    result = []
    if pd.isna(value): return result
    for token in str(value).split(";"):
        token = token.strip()
        if not token: continue
        pair, sep, role = token.rpartition("|")
        if sep and role in {"A", "B"}:
            result.append((pair, role))
    return result


def cmd_grade(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    summary = pd.read_csv(resolve(project, args.multiseed_summary), low_memory=False)
    audit = pd.read_csv(resolve(project, args.input_audit), low_memory=False)
    manifest = pd.read_csv(resolve(project, args.docking_manifest), low_memory=False)
    target_path = resolve(project, args.target_validation)
    target = pd.read_csv(target_path, low_memory=False) if target_path.is_file() else pd.DataFrame(columns=["receptor_id", "target_redocking_pass"])
    interaction_path = resolve(project, args.interaction_summary)
    interactions = pd.read_csv(interaction_path, low_memory=False) if interaction_path.is_file() else pd.DataFrame()
    output_dir = resolve(project, args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    x = manifest.copy()
    audit_cols = [c for c in ["audit_status", "fatal_reasons", "warnings"] if c in audit.columns and c not in x.columns]
    if audit_cols:
        x = x.merge(audit[["pair_id"] + audit_cols], on="pair_id", how="left")
    summary_cols = [c for c in summary.columns if c == "pair_id" or c not in x.columns]
    x = x.merge(summary[summary_cols], on="pair_id", how="left")
    if not target.empty and "receptor_id" in x.columns:
        target_merge = target[[c for c in ["receptor_id", "target_redocking_pass", "best_redocking_rmsd", "best_redocking_fraction_within_threshold"] if c in target.columns]].copy()
        target_merge = target_merge.rename(columns={
            "best_redocking_rmsd": "target_best_redocking_rmsd",
            "best_redocking_fraction_within_threshold": "target_best_redocking_fraction_within_threshold",
        })
        x = x.merge(target_merge, on="receptor_id", how="left")
    if not interactions.empty:
        interaction_cols = [c for c in interactions.columns if c == "pair_id" or c not in x.columns]
        x = x.merge(interactions[interaction_cols], on="pair_id", how="left")
    audit_pass = x["audit_status"].astype(str).str.lower().eq("pass") if "audit_status" in x.columns else pd.Series(False, index=x.index)
    stable = x["multiseed_stable"].fillna(False).map(truthy) if "multiseed_stable" in x.columns else pd.Series(False, index=x.index)
    target_pass = x["target_redocking_pass"].fillna(False).map(truthy) if "target_redocking_pass" in x.columns else pd.Series(False, index=x.index)
    x["standard_vina_mode_supported"] = x.apply(lambda row: standard_vina_mode_supported(row)[0], axis=1)
    x["normalized_docking_mode"] = x.apply(lambda row: standard_vina_mode_supported(row)[1], axis=1)
    eligible = x["standard_vina_mode_supported"].fillna(False)
    x["vina_grade"] = np.select(
        [eligible & audit_pass & stable & target_pass, eligible & audit_pass & stable, eligible & audit_pass],
        ["A", "B", "C"], default="D",
    )
    grade_score = x["vina_grade"].map({"A": 1.0, "B": 0.75, "C": 0.40, "D": 0.0})
    completion = pd.to_numeric(x.get("seed_completion_fraction", 0), errors="coerce").fillna(0).clip(0, 1)
    cluster = pd.to_numeric(x.get("medoid_cluster_fraction_within_threshold", 0), errors="coerce").fillna(0).clip(0, 1)
    iqr = pd.to_numeric(x.get("affinity_iqr", np.nan), errors="coerce")
    consistency = (1 - iqr / 2).clip(0, 1).fillna(0)
    x["vina_quality_score"] = (0.55 * grade_score + 0.15 * completion + 0.15 * cluster + 0.15 * consistency).clip(0, 1)
    x["claim_boundary"] = np.where(
        x["standard_vina_mode_supported"],
        "structural_plausibility_only_not_experimental_binding_or_drug_pair_synergy",
        "unsupported_by_standard_vina_requires_separate_validated_protocol",
    )
    x["median_affinity_within_receptor_rank"] = x.groupby("receptor_id")["median_affinity"].rank(method="min", ascending=True, na_option="bottom") if "receptor_id" in x.columns else np.nan
    x["median_affinity_within_receptor_percentile"] = x.groupby("receptor_id")["median_affinity"].rank(pct=True, ascending=False) if "receptor_id" in x.columns else np.nan
    final_file = output_dir / "vina_formal_evidence.csv"
    x.to_csv(final_file, index=False, encoding="utf-8-sig")

    # Re-aggregate structural support to upstream drug-pair memberships.
    membership_rows = []
    for _, row in x.iterrows():
        if str(row.get("docking_type", "")).lower().find("native") >= 0:
            continue
        for source_pair, role in parse_memberships(row.get("source_pair_memberships", "")):
            membership_rows.append({
                "source_pair_id": source_pair, "member_role": role,
                "docking_task_id": row["pair_id"], "drug_id": row.get("drug_id", ""),
                "target_gene": row.get("target_gene", ""), "pdb_id": row.get("pdb_id", ""),
                "receptor_id": row.get("receptor_id", ""), "vina_grade": row["vina_grade"],
                "vina_quality_score": row["vina_quality_score"], "median_affinity": row.get("median_affinity", np.nan),
                "target_redocking_pass": row.get("target_redocking_pass", False),
                "multiseed_stable": row.get("multiseed_stable", False),
            })
    membership_df = pd.DataFrame(membership_rows)
    membership_df.to_csv(output_dir / "drug_pair_member_vina_support_long.csv", index=False, encoding="utf-8-sig")
    pair_rows = []
    if not membership_df.empty:
        for source_pair, g in membership_df.groupby("source_pair_id", sort=False):
            role_summary = {}
            for role in ["A", "B"]:
                rg = g[g["member_role"] == role]
                role_summary[role] = {
                    "present": not rg.empty,
                    "best_grade": best_grade(rg["vina_grade"]) if not rg.empty else "D",
                    "supported": bool(rg["vina_grade"].isin(["A", "B"]).any()) if not rg.empty else False,
                    "validated": bool(rg["vina_grade"].eq("A").any()) if not rg.empty else False,
                    "n_tasks": int(len(rg)),
                    "best_quality": float(pd.to_numeric(rg["vina_quality_score"], errors="coerce").max()) if not rg.empty else np.nan,
                    "supported_targets": ";".join(sorted(set(rg.loc[rg["vina_grade"].isin(["A", "B"]), "target_gene"].dropna().astype(str)))),
                }
            both_supported = role_summary["A"]["supported"] and role_summary["B"]["supported"]
            both_validated = role_summary["A"]["validated"] and role_summary["B"]["validated"]
            support_class = (
                "both_members_validated_structure_support" if both_validated else
                "both_members_structural_plausibility" if both_supported else
                "single_member_structural_support" if role_summary["A"]["supported"] or role_summary["B"]["supported"] else
                "no_formal_structural_support"
            )
            pair_rows.append({
                "pair_id": source_pair,
                "drug_a_best_vina_grade": role_summary["A"]["best_grade"],
                "drug_b_best_vina_grade": role_summary["B"]["best_grade"],
                "drug_a_structurally_supported": role_summary["A"]["supported"],
                "drug_b_structurally_supported": role_summary["B"]["supported"],
                "both_members_structurally_supported": both_supported,
                "both_members_target_redocking_validated": both_validated,
                "drug_a_best_vina_quality": role_summary["A"]["best_quality"],
                "drug_b_best_vina_quality": role_summary["B"]["best_quality"],
                "drug_a_supported_targets": role_summary["A"]["supported_targets"],
                "drug_b_supported_targets": role_summary["B"]["supported_targets"],
                "n_drug_a_docking_tasks": role_summary["A"]["n_tasks"],
                "n_drug_b_docking_tasks": role_summary["B"]["n_tasks"],
                "pair_vina_support_class": support_class,
                "pair_claim_boundary": "member-level structural plausibility; not combination binding or pharmacologic synergy",
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_candidate_path = resolve(project, args.pair_candidates)
    if pair_candidate_path.is_file() and not pair_df.empty:
        candidates = pd.read_csv(pair_candidate_path, low_memory=False)
        candidate_id = "pair_id" if "pair_id" in candidates.columns else "combination_id" if "combination_id" in candidates.columns else None
        if candidate_id:
            pair_df = candidates.merge(pair_df, left_on=candidate_id, right_on="pair_id", how="left", suffixes=("", "_vina"))
    pair_df.to_csv(output_dir / "drug_pair_vina_bridge_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "n_docking_tasks": int(len(x)), "grade_counts": x["vina_grade"].value_counts(dropna=False).to_dict(),
        "n_source_pairs": int(len(pair_df)),
        "n_source_pairs_both_members_supported": int(pair_df.get("both_members_structurally_supported", pd.Series(dtype=bool)).fillna(False).sum()) if not pair_df.empty else 0,
        "affinity_comparison_rule": "Affinities are ranked only within the same receptor_id; raw values should not be compared across different receptor structures.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Formal evidence: {final_file}")
    return 0


def update_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def cmd_sync(args: argparse.Namespace, project: Path, cfg: dict) -> int:
    evidence_path = resolve(project, args.formal_evidence)
    evidence = pd.read_csv(evidence_path, low_memory=False)
    if not args.include_native:
        evidence = evidence[~evidence.get("docking_type", "").astype(str).str.contains("native", case=False, na=False)].copy()
    if args.grade_filter:
        allowed = {x.strip().upper() for x in args.grade_filter.split(",") if x.strip()}
        evidence = evidence[evidence["vina_grade"].astype(str).str.upper().isin(allowed)].copy()
    pairs_root = project / "results" / "pairs"
    summary_rows = []
    viz_rows = []
    copied = 0
    for _, row in evidence.iterrows():
        task_id = safe_id(row["pair_id"])
        pose = Path(str(row.get("representative_pose_file", "")))
        receptor = Path(str(row.get("receptor_resolved", row.get("receptor", ""))))
        if not pose.is_file() or not receptor.is_file():
            continue
        pair_dir = pairs_root / task_id
        docking_dir = pair_dir / "docking"; images_dir = pair_dir / "images"
        docking_dir.mkdir(parents=True, exist_ok=True); images_dir.mkdir(parents=True, exist_ok=True)
        target_pose = docking_dir / f"{task_id}_out.pdbqt"
        shutil.copy2(pose, target_pose)
        metadata = {
            "pair_id": task_id, "receptor_path": str(receptor.resolve()), "pose_file": str(target_pose.resolve()),
            "ligand_path": str(row.get("ligand_resolved", row.get("ligand", ""))),
            "representative_pose_source": str(pose), "vina_grade": row.get("vina_grade", ""),
            "vina_quality_score": row.get("vina_quality_score", ""),
            "median_affinity": row.get("median_affinity", ""),
            "source_pair_ids": row.get("source_pair_ids", ""),
            "formal_note": "Representative medoid pose from formal multi-seed docking.",
        }
        (pair_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append({
            "pair_id": task_id, "status": "formal_representative",
            "best_affinity_kcal_mol": row.get("median_affinity", ""),
            "receptor": str(receptor.resolve()), "ligand": str(row.get("ligand_resolved", row.get("ligand", ""))),
            "center_x": row.get("center_x", ""), "center_y": row.get("center_y", ""), "center_z": row.get("center_z", ""),
            "size_x": row.get("size_x", ""), "size_y": row.get("size_y", ""), "size_z": row.get("size_z", ""),
            "exhaustiveness": config_value(cfg, "formal_exhaustiveness", 32),
            "num_modes": config_value(cfg, "formal_num_modes", 20),
            "cpu": config_value(cfg, "formal_cpu", 4), "energy_range": config_value(cfg, "formal_energy_range", 5),
            "pose_file": str(target_pose.resolve()), "log_file": "", "elapsed_seconds": "",
            "message": f"formal grade {row.get('vina_grade','')}; plotted affinity uses multi-seed median",
        })
        viz_rows.append({
            "enabled": 1, "pair_id": task_id, "receptor": str(receptor.resolve()),
            "ligand": str(row.get("ligand_resolved", row.get("ligand", ""))),
            "center_x": row.get("center_x", ""), "center_y": row.get("center_y", ""), "center_z": row.get("center_z", ""),
            "size_x": row.get("size_x", ""), "size_y": row.get("size_y", ""), "size_z": row.get("size_z", ""),
            "exhaustiveness": config_value(cfg, "formal_exhaustiveness", 32),
            "num_modes": config_value(cfg, "formal_num_modes", 20), "cpu": config_value(cfg, "formal_cpu", 4),
            "energy_range": config_value(cfg, "formal_energy_range", 5),
            "notes": f"formal grade {row.get('vina_grade','')}; source pairs={row.get('source_pair_ids','')}",
        })
        copied += 1
    summary_fields = [
        "pair_id", "status", "best_affinity_kcal_mol", "receptor", "ligand",
        "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
        "exhaustiveness", "num_modes", "cpu", "energy_range",
        "pose_file", "log_file", "elapsed_seconds", "message",
    ]
    viz_fields = [
        "enabled", "pair_id", "receptor", "ligand", "center_x", "center_y", "center_z",
        "size_x", "size_y", "size_z", "exhaustiveness", "num_modes", "cpu", "energy_range", "notes",
    ]
    update_csv(project / "results" / "summary_vina_results.csv", summary_rows, summary_fields)
    update_csv(project / "docking_pairs_formal_visualization.csv", viz_rows, viz_fields)
    print(f"Synced representative poses: {copied}")
    print("Visualization table: docking_pairs_formal_visualization.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Formal Vina pipeline integrated with the original Windows workflow.")
    p.add_argument("--project", default=".")
    p.add_argument("--config", default="config.json")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--manifest", default="docking_pairs_formal.csv")
    a.add_argument("--output-dir", default="results/formal/01_audit")
    a.add_argument("--vina", default="")

    r = sub.add_parser("run")
    r.add_argument("--manifest", default="results/formal/01_audit/formal_docking_manifest_audited.csv")
    r.add_argument("--output-dir", default="results/formal/02_multiseed")
    r.add_argument("--vina", default="")
    r.add_argument("--seeds", default="")
    r.add_argument("--exhaustiveness", type=int, default=0)
    r.add_argument("--num-modes", type=int, default=0)
    r.add_argument("--energy-range", type=float, default=0)
    r.add_argument("--cpu", type=int, default=0)
    r.add_argument("--jobs", type=int, default=0)
    r.add_argument("--timeout", type=int, default=0)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--resume", action="store_true")

    s = sub.add_parser("summarize")
    s.add_argument("--run-manifest", default="results/formal/02_multiseed/multiseed_run_manifest.csv")
    s.add_argument("--docking-manifest", default="results/formal/01_audit/formal_docking_manifest_audited.csv")
    s.add_argument("--output-dir", default="results/formal/03_summary")
    s.add_argument("--pose-rmsd-threshold", type=float, default=0)
    s.add_argument("--redocking-rmsd-threshold", type=float, default=0)
    s.add_argument("--min-seed-completion", type=float, default=0)
    s.add_argument("--max-affinity-iqr", type=float, default=0)
    s.add_argument("--min-pose-cluster-fraction", type=float, default=0)
    s.add_argument("--min-redocking-fraction", type=float, default=0)

    i = sub.add_parser("interactions")
    i.add_argument("--multiseed-summary", default="results/formal/03_summary/vina_multiseed_summary.csv")
    i.add_argument("--docking-manifest", default="results/formal/01_audit/formal_docking_manifest_audited.csv")
    i.add_argument("--output-dir", default="results/formal/04_interactions")
    i.add_argument("--contact-cutoff", type=float, default=4.0)
    i.add_argument("--polar-cutoff", type=float, default=3.5)
    i.add_argument("--clash-cutoff", type=float, default=2.0)

    g = sub.add_parser("grade")
    g.add_argument("--multiseed-summary", default="results/formal/03_summary/vina_multiseed_summary.csv")
    g.add_argument("--input-audit", default="results/formal/01_audit/docking_input_audit.csv")
    g.add_argument("--docking-manifest", default="results/formal/01_audit/formal_docking_manifest_audited.csv")
    g.add_argument("--target-validation", default="results/formal/03_summary/target_redocking_validation.csv")
    g.add_argument("--interaction-summary", default="results/formal/04_interactions/vina_interaction_fingerprint_summary.csv")
    g.add_argument("--pair-candidates", default="")
    g.add_argument("--output-dir", default="results/formal/05_final")

    y = sub.add_parser("sync")
    y.add_argument("--formal-evidence", default="results/formal/05_final/vina_formal_evidence.csv")
    y.add_argument("--grade-filter", default="A,B")
    y.add_argument("--include-native", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    project = Path(args.project).resolve()
    cfg = load_config(project, args.config)
    if args.command == "audit": return cmd_audit(args, project, cfg)
    if args.command == "run": return cmd_run(args, project, cfg)
    if args.command == "summarize": return cmd_summarize(args, project, cfg)
    if args.command == "interactions": return cmd_interactions(args, project, cfg)
    if args.command == "grade": return cmd_grade(args, project, cfg)
    if args.command == "sync": return cmd_sync(args, project, cfg)
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
