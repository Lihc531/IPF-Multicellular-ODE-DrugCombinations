from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    resn: str
    chain: str
    resi: str
    x: float
    y: float
    z: float
    element: str

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class ResidueContact:
    resn: str
    chain: str
    resi: str
    min_distance: float
    receptor_atom_serial: int
    receptor_atom_name: str
    ligand_atom_serial: int
    ligand_atom_name: str

    @property
    def label(self) -> str:
        return f"{self.resn}{self.resi}"


@dataclass
class PairSpec:
    pair_id: str
    receptor: str
    ligand_input: str
    pose: str
    affinity: str
    output_dir: str
    overview_pml: str
    pocket_pml: str
    manual_pml: str
    overview_png: str
    pocket_png: str
    composite_png: str
    contacts: list[ResidueContact]

    def to_json(self) -> dict:
        data = asdict(self)
        data["contacts"] = [asdict(c) | {"label": c.label} for c in self.contacts]
        return data


def project_root_from_arg(project: str | Path | None) -> Path:
    root = Path(project or ".").expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project directory does not exist: {root}")
    return root


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_existing_path(project: Path, raw: str, fallbacks: Sequence[Path] = ()) -> Path:
    raw = (raw or "").strip().strip('"')
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else project / p)
        if p.name:
            candidates.extend([
                project / "prepared" / "receptors" / p.name,
                project / "prepared" / "ligands" / p.name,
                project / "inputs" / "receptors" / p.name,
                project / "inputs" / "ligands" / p.name,
            ])
    candidates.extend(fallbacks)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    attempted = "\n  ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not resolve file. Tried:\n  {attempted}")


def locate_visualization_csv(project: Path) -> Path:
    preferred = project / "docking_pairs_formal_visualization.csv"
    fallback = project / "docking_pairs.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Neither docking_pairs_formal_visualization.csv nor docking_pairs.csv exists."
    )


def read_enabled_pairs(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    enabled: list[dict[str, str]] = []
    for row in rows:
        flag = str(row.get("enabled", "1")).strip().lower()
        if flag in TRUE_VALUES:
            enabled.append({k: (v or "").strip() for k, v in row.items()})
    return enabled


def affinity_map(project: Path) -> dict[str, str]:
    candidates = [
        project / "results" / "summary_vina_results.csv",
        project / "results" / "formal" / "03_summary" / "vina_multiseed_summary.csv",
        project / "results" / "formal" / "05_final" / "vina_formal_evidence.csv",
    ]
    mapping: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pair_id = (row.get("pair_id") or row.get("task_id") or "").strip()
                if not pair_id:
                    continue
                value = (
                    row.get("median_affinity")
                    or row.get("affinity_median")
                    or row.get("affinity")
                    or row.get("vina_affinity")
                    or ""
                )
                if str(value).strip():
                    mapping[pair_id] = str(value).strip()
    return mapping


def _guess_element(atom_name: str, line: str) -> str:
    parts = line.split()
    if parts:
        tail = parts[-1].strip()
        if tail and len(tail) <= 3 and tail[0].isalpha():
            # PDBQT atom type is normally the last token.
            token = "".join(ch for ch in tail if ch.isalpha())
            if token:
                if token.upper().startswith("CL"):
                    return "Cl"
                if token.upper().startswith("BR"):
                    return "Br"
                return token[0].upper()
    stripped = "".join(ch for ch in atom_name if ch.isalpha())
    return stripped[0].upper() if stripped else "C"


def parse_pdbqt_first_model(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    in_first_model = True
    saw_model = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    in_first_model = False
                else:
                    saw_model = True
                    in_first_model = True
                continue
            if record == "ENDMDL" and saw_model:
                break
            if not in_first_model or record not in {"ATOM", "HETATM"}:
                continue
            try:
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resn = line[17:20].strip() or "UNK"
                chain = line[21:22].strip()
                resi = line[22:27].strip() or "0"
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    serial = int(parts[1])
                    name = parts[2]
                    resn = parts[3]
                    chain = parts[4] if len(parts[4]) == 1 and not parts[4].lstrip("-").isdigit() else ""
                    offset = 1 if chain else 0
                    resi = parts[4 + offset]
                    x, y, z = map(float, parts[5 + offset:8 + offset])
                except Exception:
                    continue
            atoms.append(
                Atom(
                    serial=serial,
                    name=name,
                    resn=resn,
                    chain=chain,
                    resi=resi,
                    x=x,
                    y=y,
                    z=z,
                    element=_guess_element(name, line),
                )
            )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records were parsed from {path}")
    return atoms


def distance(a: Atom, b: Atom) -> float:
    return math.dist(a.xyz, b.xyz)


def find_residue_contacts(
    receptor_atoms: Sequence[Atom],
    ligand_atoms: Sequence[Atom],
    cutoff: float = 4.0,
    max_residues: int = 18,
) -> list[ResidueContact]:
    ligand_heavy = [a for a in ligand_atoms if a.element.upper() != "H"] or list(ligand_atoms)
    receptor_heavy = [a for a in receptor_atoms if a.element.upper() != "H"] or list(receptor_atoms)
    best: dict[tuple[str, str, str], tuple[float, Atom, Atom]] = {}
    cutoff_sq = cutoff * cutoff
    for ra in receptor_heavy:
        if ra.resn.upper() in {"HOH", "WAT", "DOD"}:
            continue
        for la in ligand_heavy:
            dx = ra.x - la.x
            dy = ra.y - la.y
            dz = ra.z - la.z
            dsq = dx * dx + dy * dy + dz * dz
            if dsq > cutoff_sq:
                continue
            key = (ra.resn, ra.chain, ra.resi)
            previous = best.get(key)
            if previous is None or dsq < previous[0]:
                best[key] = (dsq, ra, la)
    contacts = [
        ResidueContact(
            resn=key[0],
            chain=key[1],
            resi=key[2],
            min_distance=math.sqrt(value[0]),
            receptor_atom_serial=value[1].serial,
            receptor_atom_name=value[1].name,
            ligand_atom_serial=value[2].serial,
            ligand_atom_name=value[2].name,
        )
        for key, value in best.items()
    ]
    contacts.sort(key=lambda item: (item.min_distance, item.chain, item.resi, item.resn))
    return contacts[:max_residues]


def pymol_selection_for_residue(contact: ResidueContact, object_name: str = "receptor") -> str:
    parts = [object_name, f"resi {contact.resi}", f"resn {contact.resn}"]
    if contact.chain:
        parts.append(f"chain {contact.chain}")
    return " and ".join(parts)


def to_pymol_path(path: Path) -> str:
    return path.resolve().as_posix().replace('"', '\\"')


def locate_pose(project: Path, pair_id: str, ligand_input: Path) -> Path:
    candidates = [
        project / "results" / "pairs" / pair_id / "docking" / f"{pair_id}_out.pdbqt",
        project / "results" / "pairs" / pair_id / "docking" / f"{pair_id}__out.pdbqt",
        project / "results" / "formal" / "03_summary" / "representative_poses" / f"{pair_id}.pdbqt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Representative pose is missing for " + pair_id + ".\nExpected, for example:\n  "
        + str(candidates[0])
        + "\nRun 19_sync_formal_poses_for_Bstyle.bat before figure generation.\n"
        + "The input ligand is deliberately not used as a substitute."
    )


def resolve_pymol(project: Path) -> list[str] | None:
    config = load_json(project / "config.json")
    configured = str(config.get("pymol_exe", "")).strip().strip('"')
    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        candidates.append(p if p.is_absolute() else project / p)
    env_value = os.environ.get("PYMOL_EXE", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path(r"E:\pymol\Scripts\pymol.exe"),
            Path(r"E:\PyMOL\PyMOL.exe"),
            Path(r"E:\Anaconda3\Anaconda\envs\pymol\Scripts\pymol.exe"),
            Path(r"C:\Program Files\PyMOL\PyMOL.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() in {".bat", ".cmd"}:
                return ["cmd.exe", "/d", "/s", "/c", "call", str(candidate.resolve())]
            return [str(candidate.resolve())]
    found = shutil.which("pymol") or shutil.which("pymol.exe")
    if found:
        return [found]
    return None


def make_pair_specs(project: Path, cutoff: float = 4.0) -> list[PairSpec]:
    csv_path = locate_visualization_csv(project)
    rows = read_enabled_pairs(csv_path)
    affinities = affinity_map(project)
    specs: list[PairSpec] = []
    for row in rows:
        pair_id = (row.get("pair_id") or row.get("task_id") or "").strip()
        if not pair_id:
            continue
        receptor = resolve_existing_path(project, row.get("receptor", ""))
        ligand_input = resolve_existing_path(project, row.get("ligand", ""))
        pose = locate_pose(project, pair_id, ligand_input)
        receptor_atoms = parse_pdbqt_first_model(receptor)
        ligand_atoms = parse_pdbqt_first_model(pose)
        contacts = find_residue_contacts(receptor_atoms, ligand_atoms, cutoff=cutoff)
        if not contacts:
            raise RuntimeError(
                f"No receptor residues were found within {cutoff:.1f} A of the representative pose for {pair_id}."
            )
        out_dir = project / "results" / "pairs" / pair_id / "figures_bstyle"
        out_dir.mkdir(parents=True, exist_ok=True)
        specs.append(
            PairSpec(
                pair_id=pair_id,
                receptor=str(receptor),
                ligand_input=str(ligand_input),
                pose=str(pose),
                affinity=affinities.get(pair_id, ""),
                output_dir=str(out_dir),
                overview_pml=str(out_dir / f"{pair_id}__overview.pml"),
                pocket_pml=str(out_dir / f"{pair_id}__pocket.pml"),
                manual_pml=str(out_dir / f"{pair_id}__manual_review.pml"),
                overview_png=str(out_dir / f"{pair_id}__overview.png"),
                pocket_png=str(out_dir / f"{pair_id}__pocket.png"),
                composite_png=str(out_dir / f"{pair_id}__composite_B.png"),
                contacts=contacts,
            )
        )
    return specs


def write_manifest(project: Path, specs: Iterable[PairSpec]) -> Path:
    manifest_dir = project / "visualization_manual"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "bstyle_scene_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "project": str(project),
                "pairs": [spec.to_json() for spec in specs],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return path


def load_manifest(project: Path) -> dict:
    path = project / "visualization_manual" / "bstyle_scene_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\nRun 05_generate_pymol_scripts.bat first."
        )
    return load_json(path)
