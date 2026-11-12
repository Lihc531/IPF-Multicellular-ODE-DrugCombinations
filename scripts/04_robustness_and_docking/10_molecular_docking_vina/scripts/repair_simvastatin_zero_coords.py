#!/usr/bin/env python3
"""Repair SIMVASTATIN_ACID ligand preparation when Open Babel produced zero coordinates.

This script uses a bundled, pre-generated 3D seed structure for the deprotonated
(open-acid) form expected at pH 7.4. It converts that SDF to MOL2 with the user's
Open Babel, validates the resulting coordinates and bond lengths, then invokes
MGLTools prepare_ligand4.py to create the final PDBQT.

Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TARGET_STEM = "SIMVASTATIN_ACID_pH7p4"
EXPECTED_PDBQTS = (
    "SELUMETINIB_pH7p4.pdbqt",
    "MAP2K1_4U7Z_3EW_native.pdbqt",
    "SIMVASTATIN_ACID_pH7p4.pdbqt",
    "HMGCR_1HW9_SIM_native.pdbqt",
)


class RepairError(RuntimeError):
    pass


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RepairError(f"Cannot parse config JSON: {path}\n{exc}") from exc


def _nested_get(data: Dict, dotted: str):
    cur = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _cfg_values(config: Dict, keys: Iterable[str]) -> List[str]:
    vals: List[str] = []
    for key in keys:
        value = _nested_get(config, key)
        if isinstance(value, str) and value.strip():
            vals.append(os.path.expandvars(value.strip().strip('"')))
    return vals


def _first_existing_file(candidates: Iterable[Optional[str]]) -> Optional[Path]:
    seen = set()
    for raw in candidates:
        if not raw:
            continue
        raw = os.path.expandvars(str(raw).strip().strip('"'))
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        p = Path(raw)
        if p.is_file():
            return p.resolve()
    return None


def _find_obabel(cli: Optional[str], config: Dict) -> Path:
    candidates: List[Optional[str]] = [cli]
    candidates += _cfg_values(config, (
        "obabel_exe", "openbabel_exe", "tools.obabel_exe", "tools.openbabel_exe",
        "openbabel.obabel_exe", "executables.obabel", "paths.obabel_exe",
    ))
    which = shutil.which("obabel")
    candidates.append(which)
    found = _first_existing_file(candidates)
    if found:
        return found
    raise RepairError(
        "Open Babel executable was not found. Set config.json -> obabel_exe, "
        "or pass --obabel E:\\path\\to\\obabel.exe"
    )


def _mgl_dir_values(config: Dict) -> List[str]:
    return _cfg_values(config, (
        "mgltools_dir", "mgltools_home", "tools.mgltools_dir",
        "mgltools.dir", "paths.mgltools_dir",
    ))


def _find_mgl_python(cli: Optional[str], config: Dict) -> Path:
    candidates: List[Optional[str]] = [cli]
    candidates += _cfg_values(config, (
        "mgl_python", "mgltools_python", "tools.mgl_python",
        "mgltools.python", "paths.mgl_python",
    ))
    for root in _mgl_dir_values(config):
        r = Path(root)
        candidates += [
            str(r / "python.exe"),
            str(r / "bin" / "python.exe"),
            str(r / "MGLToolsPckgs" / "python.exe"),
        ]
    found = _first_existing_file(candidates)
    if found:
        return found
    raise RepairError(
        "MGLTools Python was not found. Set config.json -> mgltools_dir, "
        "or pass --mgl-python E:\\MGLTools-1.5.7\\python.exe"
    )


def _find_prepare_ligand(cli: Optional[str], config: Dict, mgl_python: Path) -> Path:
    candidates: List[Optional[str]] = [cli]
    candidates += _cfg_values(config, (
        "prepare_ligand", "prepare_ligand4", "tools.prepare_ligand",
        "mgltools.prepare_ligand", "paths.prepare_ligand",
    ))
    roots: List[Path] = [Path(x) for x in _mgl_dir_values(config)]
    roots += [mgl_python.parent, mgl_python.parent.parent]
    for r in roots:
        candidates += [
            str(r / "Lib" / "site-packages" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py"),
            str(r / "lib" / "site-packages" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py"),
            str(r / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py"),
        ]
    found = _first_existing_file(candidates)
    if found:
        return found
    raise RepairError(
        "prepare_ligand4.py was not found. Pass --prepare-ligand with its full path."
    )


def _run(cmd: Sequence[str], cwd: Path, log_path: Path, label: str) -> subprocess.CompletedProcess:
    cwd.mkdir(parents=True, exist_ok=True)
    shown = subprocess.list2cmdline([str(x) for x in cmd])
    print(f"[{label}] {shown}")
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"\n===== {label} =====\nCOMMAND\n{shown}\n\nSTDOUT\n{proc.stdout}\nSTDERR\n{proc.stderr}\nEXIT_CODE\n{proc.returncode}\n")
    if proc.returncode != 0:
        raise RepairError(f"{label} failed with exit code {proc.returncode}. See {log_path}")
    return proc


def _parse_mol2(path: Path) -> Tuple[Dict[int, Tuple[float, float, float]], List[Tuple[int, int]]]:
    if not path.exists() or path.stat().st_size < 100:
        raise RepairError(f"MOL2 was not created or is empty: {path}")
    atoms: Dict[int, Tuple[float, float, float]] = {}
    bonds: List[Tuple[int, int]] = []
    section = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("@<TRIPOS>"):
            section = line.upper()
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if section == "@<TRIPOS>ATOM" and len(parts) >= 6:
            try:
                idx = int(parts[0])
                xyz = (float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError as exc:
                raise RepairError(f"Invalid MOL2 atom line: {raw}") from exc
            atoms[idx] = xyz
        elif section == "@<TRIPOS>BOND" and len(parts) >= 4:
            try:
                bonds.append((int(parts[1]), int(parts[2])))
            except ValueError as exc:
                raise RepairError(f"Invalid MOL2 bond line: {raw}") from exc
    return atoms, bonds


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _validate_mol2(path: Path) -> Dict[str, float]:
    atoms, bonds = _parse_mol2(path)
    if len(atoms) < 31:
        raise RepairError(f"MOL2 has too few atoms ({len(atoms)}); expected at least 31.")
    if len(bonds) < 30:
        raise RepairError(f"MOL2 has too few bonds ({len(bonds)}).")
    coords = list(atoms.values())
    if any(not all(math.isfinite(v) for v in xyz) for xyz in coords):
        raise RepairError("MOL2 contains NaN or infinite coordinates.")
    spans = [max(x[i] for x in coords) - min(x[i] for x in coords) for i in range(3)]
    diameter = max(_distance(a, b) for a in coords for b in coords)
    if diameter < 2.0 or max(spans) < 2.0:
        raise RepairError(
            f"MOL2 coordinates are degenerate (span={spans}, diameter={diameter:.4f})."
        )
    zero_atoms = sum(1 for xyz in coords if max(abs(v) for v in xyz) < 1e-8)
    min_bond = 999.0
    max_bond = 0.0
    bad: List[Tuple[int, int, float]] = []
    for i, j in bonds:
        if i not in atoms or j not in atoms:
            raise RepairError(f"MOL2 bond references a missing atom: {i}-{j}")
        d = _distance(atoms[i], atoms[j])
        min_bond = min(min_bond, d)
        max_bond = max(max_bond, d)
        if d < 0.20:
            bad.append((i, j, d))
    if bad:
        sample = ", ".join(f"{i}-{j}:{d:.4f}A" for i, j, d in bad[:8])
        raise RepairError(f"MOL2 contains zero/near-zero bond lengths: {sample}")
    return {
        "atom_count": float(len(atoms)),
        "bond_count": float(len(bonds)),
        "diameter": diameter,
        "min_bond": min_bond,
        "max_bond": max_bond,
        "origin_atom_count": float(zero_atoms),
        "span_x": spans[0], "span_y": spans[1], "span_z": spans[2],
    }


def _validate_pdbqt(path: Path) -> int:
    if not path.exists() or path.stat().st_size < 200:
        raise RepairError(f"PDBQT was not created or is empty: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms = [x for x in lines if x.startswith("ATOM") or x.startswith("HETATM")]
    if len(atoms) < 25:
        raise RepairError(f"PDBQT has too few atom records ({len(atoms)}).")
    if not any(x.startswith("ROOT") for x in lines):
        raise RepairError("PDBQT does not contain a ROOT record; ligand preparation may be incomplete.")
    return len(atoms)


def _backup(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".", help="Project root")
    ap.add_argument("--config", default="config.json", help="Config path, relative to project")
    ap.add_argument("--obabel", help="Full path to obabel.exe")
    ap.add_argument("--mgl-python", help="Full path to MGLTools python.exe")
    ap.add_argument("--prepare-ligand", help="Full path to prepare_ligand4.py")
    args = ap.parse_args(argv)

    project = Path(args.project).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    config = _read_json(config_path)

    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    seed_candidates = [
        project / "inputs" / "ligands" / f"{TARGET_STEM}_3D_seed.sdf",
        package_root / "inputs" / "ligands" / f"{TARGET_STEM}_3D_seed.sdf",
    ]
    seed = next((p for p in seed_candidates if p.is_file()), None)
    if seed is None:
        raise RepairError("Bundled 3D seed SDF is missing: inputs/ligands/SIMVASTATIN_ACID_pH7p4_3D_seed.sdf")

    obabel = _find_obabel(args.obabel, config)
    mgl_python = _find_mgl_python(args.mgl_python, config)
    prepare_ligand = _find_prepare_ligand(args.prepare_ligand, config, mgl_python)

    output_dir = project / "prepared" / "ligands"
    work_dir = project / "logs" / "formal_ligand_prep" / "simvastatin_zero_coord_rescue"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = project / "logs" / "formal_ligand_prep" / "SIMVASTATIN_ACID_zero_coord_rescue.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "SIMVASTATIN_ACID zero-coordinate rescue\n"
        f"timestamp={_dt.datetime.now().isoformat()}\n"
        f"project={project}\nseed={seed}\nobabel={obabel}\n"
        f"mgl_python={mgl_python}\nprepare_ligand={prepare_ligand}\n",
        encoding="utf-8",
    )

    tag = _now_tag()
    backup_dir = output_dir / "zero_coordinate_backup" / tag
    final_mol2 = output_dir / f"{TARGET_STEM}.mol2"
    final_pdbqt = output_dir / f"{TARGET_STEM}.pdbqt"
    for p in (final_mol2, final_pdbqt):
        _backup(p, backup_dir)

    work_sdf = work_dir / f"{TARGET_STEM}_3D_seed.sdf"
    work_mol2 = work_dir / f"{TARGET_STEM}.mol2"
    work_pdbqt = work_dir / f"{TARGET_STEM}.pdbqt"
    for p in (work_sdf, work_mol2, work_pdbqt):
        if p.exists():
            p.unlink()
    shutil.copy2(seed, work_sdf)

    print("============================================================")
    print("SIMVASTATIN_ACID zero-coordinate rescue")
    print("============================================================")
    print(f"Project       : {project}")
    print(f"3D seed       : {seed}")
    print(f"Open Babel    : {obabel}")
    print(f"MGL Python    : {mgl_python}")
    print(f"prepare_ligand: {prepare_ligand}")
    print()

    # Do not call --gen3d here. The bundled SDF already contains validated 3D coordinates.
    _run(
        [str(obabel), str(work_sdf), "-O", str(work_mol2), "--partialcharge", "gasteiger"],
        work_dir, log_path, "OPEN_BABEL_SDF_TO_MOL2",
    )
    metrics = _validate_mol2(work_mol2)
    print(
        "[OK] MOL2 geometry: atoms={atom_count:.0f}, bonds={bond_count:.0f}, "
        "diameter={diameter:.3f} A, min_bond={min_bond:.3f} A, "
        "span=({span_x:.3f}, {span_y:.3f}, {span_z:.3f}) A".format(**metrics)
    )

    _run(
        [str(mgl_python), str(prepare_ligand), "-l", str(work_mol2),
         "-o", str(work_pdbqt), "-A", "hydrogens", "-v"],
        work_dir, log_path, "MGLTOOLS_PREPARE_LIGAND",
    )
    pdbqt_atoms = _validate_pdbqt(work_pdbqt)

    shutil.copy2(work_mol2, final_mol2)
    shutil.copy2(work_pdbqt, final_pdbqt)
    shutil.copy2(work_sdf, output_dir / f"{TARGET_STEM}_3D_seed.sdf")

    report = {
        "status": "ok",
        "timestamp": _dt.datetime.now().isoformat(),
        "method": "bundled_deprotonated_3D_seed_to_OpenBabel_MOL2_to_MGLTools_PDBQT",
        "target": TARGET_STEM,
        "formal_charge": -1,
        "protonation_note": "carboxylate form appropriate for pH 7.4",
        "mol2_validation": metrics,
        "pdbqt_atom_records": pdbqt_atoms,
        "output_mol2": str(final_mol2),
        "output_pdbqt": str(final_pdbqt),
        "log": str(log_path),
    }
    (output_dir / "SIMVASTATIN_ACID_zero_coord_repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] PDBQT atom records: {pdbqt_atoms}")
    print(f"[OK] Installed: {final_pdbqt}")
    print(f"[OK] Log: {log_path}")
    print()
    missing = [name for name in EXPECTED_PDBQTS if not (output_dir / name).is_file()]
    if missing:
        print("[WARN] The SIMVASTATIN file is repaired, but these other formal outputs are missing:")
        for name in missing:
            print(f"       {output_dir / name}")
        print("       Run the normal preparation step once (without --force), or repair the missing inputs separately.")
    else:
        print("[OK] All four formal/native-control PDBQT files are present.")
        print("     Continue with: 13_build_formal_tasks_from_bridge.bat")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n[CANCELLED]", file=sys.stderr)
        raise SystemExit(130)
