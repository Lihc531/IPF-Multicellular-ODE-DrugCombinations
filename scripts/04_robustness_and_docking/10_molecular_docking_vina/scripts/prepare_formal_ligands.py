#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare formal standard-Vina ligands and native-redocking controls.

This script intentionally handles only ordinary non-covalent Vina tasks:
  * SELUMETINIB from inputs/ligands/SELUMETINIB.sdf
  * MAP2K1/4U7Z native ligand 3EW, chain A residue 502
  * SIMVASTATIN ACID derived from the verified 1HW9 SIM component
  * HMGCR/1HW9 native SIM ligand, chain C residue 1

BORTEZOMIB is never converted to a standard-Vina PDBQT by this script because
its boronic acid/PSMB5 interaction requires a separately validated covalent or
boron-compatible protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


# wwPDB/RCSB Chemical Component Dictionary (CCD), component SIM.
# Use the stereospecific chemical definition directly. Do not reconstruct
# bond orders/stereochemistry by round-tripping the crystallographic PDB
# residue through PDB -> SMILES.
SIMVASTATIN_ACID_CCD_ISOMERIC_SMILES = (
    "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)"
    "[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]12"
)


@dataclass(frozen=True)
class ResidueSpec:
    source: Path
    resname: str
    chain: str
    resseq: int
    output: Path
    label: str


def resolve(project: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else project / path


def load_config(project: Path, config_name: str) -> dict:
    path = resolve(project, config_name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def executable_from_config(project: Path, explicit: str, configured: str, fallback: str) -> Path:
    candidates: list[str] = [explicit, configured, fallback]
    for value in candidates:
        value = str(value or "").strip()
        if not value:
            continue
        candidate = resolve(project, value)
        if candidate.is_file():
            return candidate
        found = shutil.which(value)
        if found:
            return Path(found)
    return Path(str(explicit or configured or fallback))


def locate_mgltools(project: Path, cfg: dict, mgl_python: str, prepare_ligand: str) -> tuple[Path, Path]:
    root = resolve(project, str(cfg.get("mgltools_dir", ""))) if cfg.get("mgltools_dir") else Path("")
    python_candidates = [
        mgl_python,
        str(root / "python.exe") if str(root) not in {"", "."} else "",
        str(root / "bin" / "python.exe") if str(root) not in {"", "."} else "",
        str(root / "bin" / "MGLPython.exe") if str(root) not in {"", "."} else "",
        str(root / "bin" / "pythonsh.exe") if str(root) not in {"", "."} else "",
        str(root / "bin" / "pythonsh") if str(root) not in {"", "."} else "",
    ]
    script_candidates = [
        prepare_ligand,
        str(root / "Lib" / "site-packages" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py") if str(root) not in {"", "."} else "",
        str(root / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_ligand4.py") if str(root) not in {"", "."} else "",
    ]

    mglpy = next((resolve(project, p) for p in python_candidates if p and resolve(project, p).is_file()), Path(""))
    prep = next((resolve(project, p) for p in script_candidates if p and resolve(project, p).is_file()), Path(""))
    return mglpy, prep


def atom_serial(line: str) -> int | None:
    try:
        return int(line[6:11])
    except (ValueError, IndexError):
        return None


def extract_residue(spec: ResidueSpec, force: bool = False) -> dict:
    if not spec.source.is_file():
        raise FileNotFoundError(f"Missing source PDB for {spec.label}: {spec.source}")
    if spec.output.is_file() and spec.output.stat().st_size > 0 and not force:
        return {"path": str(spec.output), "status": "existing", "atom_count": count_pdb_atoms(spec.output)}

    selected: list[str] = []
    selected_serials: set[int] = set()
    source_lines = spec.source.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in source_lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip().upper() != spec.resname.upper():
            continue
        if line[21:22].strip().upper() != spec.chain.upper():
            continue
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        if resseq != spec.resseq:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        selected.append(line)
        serial = atom_serial(line)
        if serial is not None:
            selected_serials.add(serial)

    if not selected:
        raise RuntimeError(
            f"Could not find {spec.resname} chain {spec.chain} residue {spec.resseq} in {spec.source}"
        )

    conect_lines: list[str] = []
    for line in source_lines:
        if not line.startswith("CONECT"):
            continue
        values: list[int] = []
        for start in range(6, len(line), 5):
            token = line[start:start + 5].strip()
            if token.isdigit():
                values.append(int(token))
        if not values or values[0] not in selected_serials:
            continue
        neighbors = [v for v in values[1:] if v in selected_serials]
        if neighbors:
            conect_lines.append("CONECT" + f"{values[0]:5d}" + "".join(f"{v:5d}" for v in neighbors))

    spec.output.parent.mkdir(parents=True, exist_ok=True)
    contents = [
        f"REMARK Extracted {spec.label}",
        f"REMARK Source: {spec.source}",
        *selected,
        *conect_lines,
        "END",
    ]
    spec.output.write_text("\n".join(contents) + "\n", encoding="ascii", errors="replace")
    return {"path": str(spec.output), "status": "extracted", "atom_count": len(selected)}


def count_pdb_atoms(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
               if line.startswith(("ATOM  ", "HETATM")))


def write_text_if_changed(path: Path, text: str) -> bool:
    """Write text and return True only when the on-disk content changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text if text.endswith("\n") else text + "\n"
    if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == normalized:
        return False
    path.write_text(normalized, encoding="utf-8")
    return True


def run_command(command: Sequence[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(map(str, command)),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        errors="replace",
    )
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(list(map(str, command))) + "\n\nSTDOUT\n" +
        (completed.stdout or "") + "\nSTDERR\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def run_if_needed(command: Sequence[str], output: Path, log_path: Path, force: bool, cwd: Path | None = None) -> str:
    if output.is_file() and output.stat().st_size > 0 and not force:
        return "existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    run_command(command, log_path, cwd=cwd)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"Command returned success but output is missing or empty: {output}")
    return "generated"


def prepare_pdbqt(mgl_python: Path, prepare_script: Path, mol2: Path, pdbqt: Path,
                   log_path: Path, force: bool) -> str:
    # MGLTools/MolKit on Windows can lose directory components. Run inside the
    # output directory and pass only file names.
    if mol2.parent.resolve() != pdbqt.parent.resolve():
        raise ValueError("MOL2 and PDBQT must be in the same directory for the MGLTools compatibility workaround")
    command = [
        str(mgl_python), str(prepare_script),
        "-l", mol2.name, "-o", pdbqt.name,
        "-A", "hydrogens", "-v",
    ]
    return run_if_needed(command, pdbqt, log_path, force=force, cwd=mol2.parent)


def validate_pdbqt(path: Path, min_atoms: int = 10) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"PDBQT missing or empty: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if len(atom_lines) < min_atoms:
        raise RuntimeError(f"Too few atoms in {path}: {len(atom_lines)}")
    if not any(line.startswith("ROOT") for line in lines):
        raise RuntimeError(f"ROOT record not found in ligand PDBQT: {path}")
    if not any(line.startswith("TORSDOF") for line in lines):
        raise RuntimeError(f"TORSDOF record not found in ligand PDBQT: {path}")
    atom_types = [line.split()[-1].upper() for line in atom_lines if line.split()]
    if "B" in atom_types:
        raise RuntimeError(f"Unsupported boron atom type detected in standard-Vina ligand: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "atom_count": len(atom_lines),
        "atom_types": sorted(set(atom_types)),
    }


def update_csv(path: Path, key_column: str, updates: dict[str, dict[str, object]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        rows = list(reader)
        fieldnames = reader.fieldnames
    seen: set[str] = set()
    for row in rows:
        key = str(row.get(key_column, "")).strip().upper()
        if key in updates:
            for column, value in updates[key].items():
                if column not in fieldnames:
                    raise RuntimeError(f"Column {column!r} not found in {path}")
                row[column] = str(value)
            seen.add(key)
    missing = sorted(set(updates) - seen)
    if missing:
        raise RuntimeError(f"Rows not found in {path} for {key_column}: {missing}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def enable_formal_maps(project: Path) -> None:
    ligand_map = project / "bridge_integration" / "ligand_map.csv"
    structure_map = project / "bridge_integration" / "target_structure_map.csv"
    update_csv(ligand_map, "drug_id", {
        "SELUMETINIB": {
            "enabled": 1,
            "ligand": "prepared/ligands/SELUMETINIB_pH7p4.pdbqt",
            "protonation_state": "pH7.4_major_state",
            "ligand_source": "inputs/ligands/SELUMETINIB.sdf; prepared_by_scripts/prepare_formal_ligands.py",
            "docking_mode": "standard_non_covalent",
            "standard_vina_eligible": 1,
            "special_handling": "",
            "notes": "由正式配体脚本从已提供3D SDF制备；运行正式审计前仍需检查立体化学、质子化和PDBQT原子类型",
        },
        "SIMVASTATIN": {
            "enabled": 1,
            "drug_name": "Simvastatin acid",
            "ligand": "prepared/ligands/SIMVASTATIN_ACID_pH7p4.pdbqt",
            "protonation_state": "open_acid_major_state_pH7.4",
            "ligand_source": "1HW9 SIM chain C resi 1; independent 3D conformer generated by formal ligand script",
            "docking_mode": "standard_non_covalent",
            "standard_vina_eligible": 1,
            "special_handling": "verified_open_acid_source",
            "notes": "正式输入为1HW9验证的开环SIMVASTATIN ACID化学结构；不得用原闭环内酯SDF/PDBQT替代",
        },
        "BORTEZOMIB": {
            "enabled": 0,
            "docking_mode": "covalent_required",
            "standard_vina_eligible": 0,
            "special_handling": "defer_from_standard_vina",
        },
    })
    update_csv(structure_map, "target_gene", {
        "MAP2K1": {
            "enabled": 1,
            "pdb_id": "4U7Z",
            "receptor": "prepared/receptors/4U7Z.pdbqt",
            "center_x": "-33.218", "center_y": "29.280", "center_z": "14.879",
            "size_x": "21.799", "size_y": "21.332", "size_z": "20.000",
            "native_ligand_pdbqt": "prepared/ligands/MAP2K1_4U7Z_3EW_native.pdbqt",
            "structure_priority": 1,
            "native_ligand_protonation_state": "experimental_3EW_state",
            "docking_mode": "standard_non_covalent",
            "standard_vina_eligible": 1,
            "special_handling": "",
            "notes": "Human MAP2K1/MEK1 4U7Z cocrystal; box derived from 3EW chain A resi 502",
        },
        "HMGCR": {
            "enabled": 1,
            "pdb_id": "1HW9",
            "receptor": "prepared/receptors/1HW9.pdbqt",
            "center_x": "21.155", "center_y": "-32.225", "center_z": "26.620",
            "size_x": "20.788", "size_y": "20.226", "size_z": "20.965",
            "native_ligand_pdbqt": "prepared/ligands/HMGCR_1HW9_SIM_native.pdbqt",
            "structure_priority": 1,
            "native_ligand_protonation_state": "experimental_simvastatin_acid_state_pH7.4",
            "docking_mode": "standard_non_covalent",
            "standard_vina_eligible": 1,
            "special_handling": "",
            "notes": "Human HMGCR 1HW9 SIMVASTATIN ACID site; box derived from SIM chain C resi 1",
        },
        "PSMB5": {
            "enabled": 0,
            "receptor": "prepared/receptors/5LF3.pdbqt",
            "docking_mode": "covalent_required",
            "standard_vina_eligible": 0,
            "special_handling": "reference_only_or_covalent_protocol",
        },
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="Project root containing config.json")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--obabel", default="", help="Override Open Babel executable")
    parser.add_argument("--mgl-python", default="", help="Override MGLTools Python executable")
    parser.add_argument("--prepare-ligand", default="", help="Override prepare_ligand4.py")
    parser.add_argument("--force", action="store_true", help="Rebuild existing intermediate and final ligand files")
    parser.add_argument("--extract-only", action="store_true", help="Only extract verified native ligands; do not call external tools")
    parser.add_argument("--no-map-update", action="store_true", help="Do not update ligand_map.csv and target_structure_map.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    cfg = load_config(project, args.config)
    ligand_dir = project / "prepared" / "ligands"
    source_dir = ligand_dir / "formal_sources"
    log_dir = project / "logs" / "formal_ligand_prep"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    native_specs = [
        ResidueSpec(
            source=project / "inputs" / "receptors" / "4U7Z.pdb",
            resname="3EW", chain="A", resseq=502,
            output=source_dir / "MAP2K1_4U7Z_3EW_native.pdb",
            label="4U7Z 3EW chain A residue 502",
        ),
        ResidueSpec(
            source=project / "inputs" / "receptors" / "1HW9.pdb",
            resname="SIM", chain="C", resseq=1,
            output=source_dir / "HMGCR_1HW9_SIM_native.pdb",
            label="1HW9 SIMVASTATIN ACID chain C residue 1",
        ),
    ]
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "scope": "standard non-covalent Vina ligands only",
        "bortezomib": "excluded; covalent/boron-compatible protocol required",
        "extractions": [],
        "outputs": [],
    }
    for spec in native_specs:
        report["extractions"].append(extract_residue(spec, force=args.force))  # type: ignore[union-attr]

    if args.extract_only:
        report["status"] = "extract_only_complete"
        report_path = source_dir / "native_source_extraction_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Native ligand sources extracted. Report: {report_path}")
        return 0

    obabel = executable_from_config(project, args.obabel, str(cfg.get("obabel_exe", "")), "obabel")
    mglpy, prepare_script = locate_mgltools(project, cfg, args.mgl_python, args.prepare_ligand)
    missing_tools = []
    if not obabel.is_file() and shutil.which(str(obabel)) is None:
        missing_tools.append(f"Open Babel: {obabel}")
    if not mglpy.is_file():
        missing_tools.append(f"MGLTools Python: {mglpy}")
    if not prepare_script.is_file():
        missing_tools.append(f"prepare_ligand4.py: {prepare_script}")
    if missing_tools:
        print("[FATAL] Required ligand-preparation tools were not found:", file=sys.stderr)
        for item in missing_tools:
            print(f"  - {item}", file=sys.stderr)
        print("Activate the intended environment or update config.json/command-line overrides.", file=sys.stderr)
        return 2

    sel_sdf = project / "inputs" / "ligands" / "SELUMETINIB.sdf"
    if not sel_sdf.is_file():
        raise FileNotFoundError(f"Missing SELUMETINIB SDF: {sel_sdf}")

    sel_mol2 = ligand_dir / "SELUMETINIB_pH7p4.mol2"
    sel_pdbqt = ligand_dir / "SELUMETINIB_pH7p4.pdbqt"
    run_if_needed(
        [str(obabel), "-isdf", str(sel_sdf), "-omol2", "-O", str(sel_mol2), "-p", "7.4"],
        sel_mol2, log_dir / "SELUMETINIB_obabel.log", args.force,
    )
    prepare_pdbqt(mglpy, prepare_script, sel_mol2, sel_pdbqt,
                  log_dir / "SELUMETINIB_mgltools.log", args.force)

    ew_pdb = source_dir / "MAP2K1_4U7Z_3EW_native.pdb"
    ew_mol2 = ligand_dir / "MAP2K1_4U7Z_3EW_native.mol2"
    ew_pdbqt = ligand_dir / "MAP2K1_4U7Z_3EW_native.pdbqt"
    run_if_needed(
        [str(obabel), "-ipdb", str(ew_pdb), "-omol2", "-O", str(ew_mol2), "-h"],
        ew_mol2, log_dir / "MAP2K1_4U7Z_3EW_native_obabel.log", args.force,
    )
    prepare_pdbqt(mglpy, prepare_script, ew_mol2, ew_pdbqt,
                  log_dir / "MAP2K1_4U7Z_3EW_native_mgltools.log", args.force)

    sim_native_pdb = source_dir / "HMGCR_1HW9_SIM_native.pdb"
    sim_native_mol2 = ligand_dir / "HMGCR_1HW9_SIM_native.mol2"
    sim_native_pdbqt = ligand_dir / "HMGCR_1HW9_SIM_native.pdbqt"
    run_if_needed(
        [str(obabel), "-ipdb", str(sim_native_pdb), "-omol2", "-O", str(sim_native_mol2), "-p", "7.4"],
        sim_native_mol2, log_dir / "HMGCR_1HW9_SIM_native_obabel.log", args.force,
    )
    prepare_pdbqt(mglpy, prepare_script, sim_native_mol2, sim_native_pdbqt,
                  log_dir / "HMGCR_1HW9_SIM_native_mgltools.log", args.force)

    # Generate a separate 3D conformer for cross-docking rather than reusing the
    # exact crystallographic coordinates as the starting pose.
    sim_smi = source_dir / "SIMVASTATIN_ACID_from_1HW9.smi"
    sim_sdf = source_dir / "SIMVASTATIN_ACID_pH7p4_generated.sdf"
    sim_mol2 = ligand_dir / "SIMVASTATIN_ACID_pH7p4.mol2"
    sim_pdbqt = ligand_dir / "SIMVASTATIN_ACID_pH7p4.pdbqt"
    # Use the authoritative stereospecific CCD definition. A PDB HETATM
    # residue carries coordinates/connectivity but is not a reliable source of
    # bond orders or stereochemistry for rebuilding an independent conformer.
    smiles_changed = write_text_if_changed(
        sim_smi, SIMVASTATIN_ACID_CCD_ISOMERIC_SMILES
    )
    sim_force = args.force or smiles_changed
    run_if_needed(
        [str(obabel), "-ismi", str(sim_smi), "-osdf", "-O", str(sim_sdf), "--gen3d", "-p", "7.4"],
        sim_sdf, log_dir / "SIMVASTATIN_ACID_gen3d.log", sim_force,
    )
    run_if_needed(
        [str(obabel), "-isdf", str(sim_sdf), "-omol2", "-O", str(sim_mol2), "-p", "7.4"],
        sim_mol2, log_dir / "SIMVASTATIN_ACID_mol2.log", sim_force,
    )
    prepare_pdbqt(mglpy, prepare_script, sim_mol2, sim_pdbqt,
                  log_dir / "SIMVASTATIN_ACID_mgltools.log", sim_force)

    for output in (sel_pdbqt, ew_pdbqt, sim_pdbqt, sim_native_pdbqt):
        report["outputs"].append(validate_pdbqt(output, min_atoms=20))  # type: ignore[union-attr]

    if not args.no_map_update:
        enable_formal_maps(project)
        report["maps_updated"] = [
            "bridge_integration/ligand_map.csv",
            "bridge_integration/target_structure_map.csv",
        ]

    report["status"] = "success"
    report["tools"] = {
        "obabel": str(obabel),
        "mgl_python": str(mglpy),
        "prepare_ligand4": str(prepare_script),
    }
    report_path = ligand_dir / "formal_ligand_preparation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Formal standard-Vina ligand preparation completed.")
    for output in report["outputs"]:  # type: ignore[assignment]
        print(f"  - {output['path']}")
    print(f"Report: {report_path}")
    print("Next: run 13_build_formal_tasks_from_bridge.bat, then 14_audit_formal_inputs.bat.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
