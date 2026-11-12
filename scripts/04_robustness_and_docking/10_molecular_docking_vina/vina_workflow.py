#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained AutoDock Vina workflow for Windows projects.

The script intentionally keeps all project paths relative to the project root and
supports the following commands:
  init, check, prepare, extract-centers, calculate-center, dock, make-pymol,
  organize-images, export-plot-data, plot-scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_DIRS = (
    "inputs/receptors",
    "inputs/ligands",
    "prepared/receptors",
    "prepared/ligands",
    "centers",
    "results/pairs",
    "results/final_figures",
    "results/plot_data",
    "logs",
)

PAIR_COLUMNS = [
    "enabled", "pair_id", "receptor", "ligand",
    "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z",
    "exhaustiveness", "num_modes", "cpu", "energy_range", "notes",
]

SUMMARY_COLUMNS = [
    "pair_id", "status", "best_affinity_kcal_mol", "receptor", "ligand",
    "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
    "exhaustiveness", "num_modes", "cpu", "energy_range",
    "pose_file", "log_file", "elapsed_seconds", "message",
]

COMMON_WATER = {"HOH", "WAT", "H2O", "DOD"}
COMMON_IONS = {
    "NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI",
    "CD", "HG", "SO4", "PO4", "NO3", "IOD", "BR",
}


@dataclass
class WorkflowConfig:
    project: Path
    config_path: Path
    data: dict

    @property
    def vina_exe(self) -> Path:
        return resolve_config_path(self.project, self.data.get("vina_exe", ""))

    @property
    def mgltools_dir(self) -> Path:
        return resolve_config_path(self.project, self.data.get("mgltools_dir", ""))

    @property
    def pymol_exe(self) -> Path:
        return resolve_config_path(self.project, self.data.get("pymol_exe", ""))

    @property
    def obabel_exe(self) -> Optional[Path]:
        raw = str(self.data.get("obabel_exe", "")).strip()
        if raw:
            return resolve_config_path(self.project, raw)
        found = shutil.which("obabel") or shutil.which("obabel.exe")
        return Path(found) if found else None

    def default(self, key: str, fallback):
        value = self.data.get(key, fallback)
        return fallback if value in (None, "") else value


def resolve_config_path(project: Path, raw: object) -> Path:
    text = str(raw or "").strip().strip('"').strip("'")
    text = os.path.expandvars(os.path.expanduser(text))
    path = Path(text)
    if text and not path.is_absolute():
        path = project / path
    return path.resolve() if text else Path("")


def load_config(project: Path, config_name: str) -> WorkflowConfig:
    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = project / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"config.json 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列：{exc.msg}"
        ) from exc
    return WorkflowConfig(project=project, config_path=config_path, data=data)


def ensure_project_dirs(project: Path) -> None:
    for rel in PROJECT_DIRS:
        (project / rel).mkdir(parents=True, exist_ok=True)


def bool_value(value: object, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "n", "off", "disabled", "skip"}


def float_value(value: object, default: float) -> float:
    text = str(value or "").strip()
    return default if not text else float(text)


def int_value(value: object, default: int) -> int:
    text = str(value or "").strip()
    return default if not text else int(float(text))


def safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return cleaned.strip("_.") or "pair"


def discover_files(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    suffixes_l = {s.lower() for s in suffixes}
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes_l)


def run_command(
    command: Sequence[str],
    cwd: Path,
    log_file: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in command)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"COMMAND: {printable}\n\n")
            return subprocess.run(
                list(map(str, command)), cwd=str(cwd), stdout=log,
                stderr=subprocess.STDOUT, text=True, timeout=timeout,
            )
    return subprocess.run(
        list(map(str, command)), cwd=str(cwd), capture_output=True,
        text=True, timeout=timeout,
    )


def find_mgltools_python(mgltools_dir: Path) -> Optional[Path]:
    candidates = [
        mgltools_dir / "pythonsh.exe",
        mgltools_dir / "bin" / "pythonsh.exe",
        mgltools_dir / "python.exe",
    ]
    return next((p for p in candidates if p.exists()), None)


def find_mgltools_script(mgltools_dir: Path, name: str) -> Optional[Path]:
    candidates = [
        mgltools_dir / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / name,
        mgltools_dir / "Lib" / "site-packages" / "AutoDockTools" / "Utilities24" / name,
        mgltools_dir / name,
    ]
    return next((p for p in candidates if p.exists()), None)


def check_environment(cfg: WorkflowConfig) -> int:
    ensure_project_dirs(cfg.project)
    checks: List[Tuple[str, bool, str]] = []
    checks.append(("Python", True, sys.executable))
    checks.append(("config.json", cfg.config_path.exists(), str(cfg.config_path)))
    checks.append(("AutoDock Vina", cfg.vina_exe.exists(), str(cfg.vina_exe)))
    checks.append(("MGLTools directory", cfg.mgltools_dir.exists(), str(cfg.mgltools_dir)))
    pythonsh = find_mgltools_python(cfg.mgltools_dir) if cfg.mgltools_dir.exists() else None
    rec_script = find_mgltools_script(cfg.mgltools_dir, "prepare_receptor4.py") if cfg.mgltools_dir.exists() else None
    lig_script = find_mgltools_script(cfg.mgltools_dir, "prepare_ligand4.py") if cfg.mgltools_dir.exists() else None
    checks.append(("MGLTools python", bool(pythonsh), str(pythonsh or "未找到")))
    checks.append(("prepare_receptor4.py", bool(rec_script), str(rec_script or "未找到")))
    checks.append(("prepare_ligand4.py", bool(lig_script), str(lig_script or "未找到")))
    checks.append(("PyMOL", cfg.pymol_exe.exists(), str(cfg.pymol_exe)))
    obabel = cfg.obabel_exe
    checks.append(("Open Babel（非 PDB/MOL2 配体需要）", bool(obabel and obabel.exists()), str(obabel or "未找到")))

    print("=" * 76)
    print("AutoDock Vina 项目环境检查")
    print("=" * 76)
    hard_fail = False
    for name, ok, detail in checks:
        marker = "OK" if ok else "缺失"
        print(f"[{marker:4s}] {name}: {detail}")
        if name in {"AutoDock Vina", "MGLTools directory", "MGLTools python", "prepare_receptor4.py", "prepare_ligand4.py"} and not ok:
            hard_fail = True
    print("-" * 76)
    print(f"受体输入目录：{cfg.project / 'inputs/receptors'}")
    print(f"配体输入目录：{cfg.project / 'inputs/ligands'}")
    print(f"任务表：{cfg.project / 'docking_pairs.csv'}")
    print("=" * 76)
    return 1 if hard_fail else 0


def pdb_atom_fields(line: str) -> Optional[dict]:
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        return {
            "record": line[0:6].strip(),
            "atom": line[12:16].strip(),
            "altloc": line[16:17].strip(),
            "resn": line[17:20].strip(),
            "chain": line[21:22].strip(),
            "resi": line[22:26].strip(),
            "icode": line[26:27].strip(),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
        }
    except (ValueError, IndexError):
        return None


def clean_receptor_pdb(source: Path, output: Path, keep_hetero: bool = False) -> None:
    lines: List[str] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ATOM  "):
            if line[16:17] not in {" ", "A"}:
                continue
            lines.append(line)
        elif keep_hetero and line.startswith("HETATM"):
            fields = pdb_atom_fields(line)
            if not fields or fields["resn"].upper() in COMMON_WATER:
                continue
            lines.append(line)
        elif line.startswith(("TER", "END")):
            lines.append(line)
    if not lines or not any(line.startswith("ATOM  ") for line in lines):
        raise ValueError(f"受体文件中没有可用 ATOM 记录：{source}")
    if not lines[-1].startswith("END"):
        lines.append("END")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")


def prepare_receptor(source: Path, target: Path, cfg: WorkflowConfig, force: bool) -> Tuple[bool, str]:
    if target.exists() and not force:
        return True, "已存在，跳过"
    if source.suffix.lower() == ".pdbqt":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True, "复制现有 PDBQT"
    pythonsh = find_mgltools_python(cfg.mgltools_dir)
    script = find_mgltools_script(cfg.mgltools_dir, "prepare_receptor4.py")
    if not pythonsh or not script:
        return False, "未找到 MGLTools receptor preparation 工具"
    cleaned = target.with_name(target.stem + "_clean.pdb")
    clean_receptor_pdb(source, cleaned, keep_hetero=bool_value(cfg.data.get("keep_receptor_hetero"), False))
    log = cfg.project / "logs" / f"prepare_receptor_{safe_id(source.stem)}.log"
    command = [str(pythonsh), str(script), "-r", str(cleaned), "-o", str(target), "-A", "hydrogens"]
    cp = run_command(command, cfg.project, log)
    return cp.returncode == 0 and target.exists(), f"日志：{log}"


def prepare_ligand(source: Path, target: Path, cfg: WorkflowConfig, force: bool) -> Tuple[bool, str]:
    if target.exists() and not force:
        return True, "已存在，跳过"
    if source.suffix.lower() == ".pdbqt":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True, "复制现有 PDBQT"
    pythonsh = find_mgltools_python(cfg.mgltools_dir)
    script = find_mgltools_script(cfg.mgltools_dir, "prepare_ligand4.py")
    if not pythonsh or not script:
        return False, "未找到 MGLTools ligand preparation 工具"
    working_input = source
    temp_mol2 = target.with_name(target.stem + "_for_mgltools.mol2")
    if source.suffix.lower() not in {".pdb", ".mol2"}:
        obabel = cfg.obabel_exe
        if not obabel or not obabel.exists():
            return False, f"{source.suffix} 配体需要 Open Babel，但未配置 obabel_exe"
        convert_log = cfg.project / "logs" / f"obabel_{safe_id(source.stem)}.log"
        cp = run_command([str(obabel), str(source), "-O", str(temp_mol2), "-h"], cfg.project, convert_log)
        if cp.returncode != 0 or not temp_mol2.exists():
            return False, f"Open Babel 转换失败，日志：{convert_log}"
        working_input = temp_mol2
    target.parent.mkdir(parents=True, exist_ok=True)
    log = cfg.project / "logs" / f"prepare_ligand_{safe_id(source.stem)}.log"
    command = [str(pythonsh), str(script), "-l", str(working_input), "-o", str(target), "-A", "hydrogens"]
    cp = run_command(command, cfg.project, log)
    return cp.returncode == 0 and target.exists(), f"日志：{log}"


def prepare_all(cfg: WorkflowConfig, force: bool = False) -> int:
    ensure_project_dirs(cfg.project)
    receptor_sources = discover_files(cfg.project / "inputs/receptors", {".pdb", ".pdbqt"})
    ligand_sources = discover_files(cfg.project / "inputs/ligands", {".pdb", ".pdbqt", ".mol2", ".sdf", ".mol"})
    print(f"发现受体文件：{len(receptor_sources)}")
    print(f"发现配体文件：{len(ligand_sources)}")
    failures = 0
    for source in receptor_sources:
        target = cfg.project / "prepared/receptors" / f"{source.stem}.pdbqt"
        ok, message = prepare_receptor(source, target, cfg, force)
        print(f"[{'OK' if ok else 'FAIL'}] receptor {source.name} -> {target.name}；{message}")
        failures += 0 if ok else 1
    for source in ligand_sources:
        target = cfg.project / "prepared/ligands" / f"{source.stem}.pdbqt"
        ok, message = prepare_ligand(source, target, cfg, force)
        print(f"[{'OK' if ok else 'FAIL'}] ligand {source.name} -> {target.name}；{message}")
        failures += 0 if ok else 1
    if not receptor_sources and not ligand_sources:
        print("没有输入文件。请将受体放入 inputs/receptors，将配体放入 inputs/ligands。")
        return 1
    return 1 if failures else 0


def hetero_groups(pdb_path: Path) -> Dict[Tuple[str, str, str], List[Tuple[float, float, float]]]:
    groups: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = pdb_atom_fields(line)
        if not fields or fields["record"] != "HETATM":
            continue
        resn = fields["resn"].upper()
        if resn in COMMON_WATER or resn in COMMON_IONS:
            continue
        key = (fields["chain"] or "_", fields["resi"], resn)
        groups.setdefault(key, []).append((fields["x"], fields["y"], fields["z"]))
    return groups


def centroid_and_size(coords: Sequence[Tuple[float, float, float]], padding: float, min_size: float) -> Tuple[float, float, float, float, float, float]:
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    zs = [p[2] for p in coords]
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    sx = max(min_size, max(xs) - min(xs) + 2 * padding)
    sy = max(min_size, max(ys) - min(ys) + 2 * padding)
    sz = max(min_size, max(zs) - min(zs) + 2 * padding)
    return cx, cy, cz, sx, sy, sz


def extract_centers(cfg: WorkflowConfig, padding: float, min_size: float) -> int:
    sources = discover_files(cfg.project / "inputs/receptors", {".pdb"})
    rows: List[dict] = []
    for pdb in sources:
        groups = hetero_groups(pdb)
        for (chain, resi, resn), coords in groups.items():
            cx, cy, cz, sx, sy, sz = centroid_and_size(coords, padding, min_size)
            rows.append({
                "receptor": pdb.stem, "source_pdb": str(pdb.relative_to(cfg.project)),
                "chain": chain, "resi": resi, "resn": resn, "atom_count": len(coords),
                "center_x": f"{cx:.3f}", "center_y": f"{cy:.3f}", "center_z": f"{cz:.3f}",
                "size_x": f"{sx:.3f}", "size_y": f"{sy:.3f}", "size_z": f"{sz:.3f}",
            })
    out = cfg.project / "centers" / "cocrystal_centers.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["receptor", "source_pdb", "chain", "resi", "resn", "atom_count", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"共晶/异源小分子中心数：{len(rows)}")
    print(f"输出：{out}")
    if not rows:
        print("未发现可用 HETATM 小分子。可改用 03_calculate_CDC25A_center.bat 手动指定残基。")
    return 0


def parse_residue_tokens(text: str) -> List[Tuple[Optional[str], str]]:
    tokens: List[Tuple[Optional[str], str]] = []
    for raw in re.split(r"[,;\s]+", text.strip()):
        if not raw:
            continue
        if ":" in raw:
            chain, resi = raw.split(":", 1)
            tokens.append((chain.strip() or None, resi.strip()))
        else:
            tokens.append((None, raw.strip()))
    return tokens


def calculate_center_from_selection(
    pdb_path: Path,
    residues: str,
    atom_names: str,
    padding: float,
    min_size: float,
) -> Tuple[float, float, float, float, float, float, int]:
    residue_tokens = parse_residue_tokens(residues)
    atoms_filter = {x.upper() for x in re.split(r"[,;\s]+", atom_names.strip()) if x.strip()}
    coords: List[Tuple[float, float, float]] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = pdb_atom_fields(line)
        if not fields:
            continue
        match_residue = not residue_tokens or any(
            fields["resi"] == resi and (chain is None or fields["chain"] == chain)
            for chain, resi in residue_tokens
        )
        match_atom = not atoms_filter or fields["atom"].upper() in atoms_filter
        if match_residue and match_atom:
            coords.append((fields["x"], fields["y"], fields["z"]))
    if not coords:
        raise ValueError("没有找到符合残基/原子筛选条件的坐标。")
    values = centroid_and_size(coords, padding, min_size)
    return (*values, len(coords))


def interactive_calculate_center(cfg: WorkflowConfig, args: argparse.Namespace) -> int:
    receptor_input = args.receptor or input("请输入受体 PDB 文件名或路径：").strip()
    pdb_path = resolve_input_path(cfg.project, receptor_input, "receptor", {".pdb"}, prepared=False)
    residues = args.residues if args.residues is not None else input("请输入位点残基，例如 A:430,A:431,A:436：").strip()
    atom_names = args.atoms if args.atoms is not None else input("可选原子名筛选，例如 CA,CB；直接回车表示全部原子：").strip()
    cx, cy, cz, sx, sy, sz, count = calculate_center_from_selection(
        pdb_path, residues, atom_names, args.padding, args.min_size
    )
    row = {
        "receptor": pdb_path.stem, "source_pdb": str(pdb_path), "residues": residues,
        "atom_names": atom_names, "atom_count": count,
        "center_x": f"{cx:.3f}", "center_y": f"{cy:.3f}", "center_z": f"{cz:.3f}",
        "size_x": f"{sx:.3f}", "size_y": f"{sy:.3f}", "size_z": f"{sz:.3f}",
    }
    out = cfg.project / "centers" / f"{safe_id(pdb_path.stem)}_manual_center.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print("\n建议写入 docking_pairs.csv 的参数：")
    print(f"center_x={cx:.3f}, center_y={cy:.3f}, center_z={cz:.3f}")
    print(f"size_x={sx:.3f}, size_y={sy:.3f}, size_z={sz:.3f}")
    print(f"使用原子数：{count}")
    print(f"输出：{out}")
    return 0


def read_pairs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到任务表：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    normalized = []
    for row in rows:
        clean = {str(k or "").strip(): str(v or "").strip() for k, v in row.items()}
        if not any(clean.values()):
            continue
        normalized.append(clean)
    return normalized


def resolve_input_path(
    project: Path,
    value: str,
    kind: str,
    suffixes: Iterable[str],
    prepared: bool = True,
) -> Path:
    value = str(value).strip().strip('"').strip("'")
    direct = Path(os.path.expandvars(os.path.expanduser(value)))
    if not direct.is_absolute():
        direct = project / direct
    if direct.exists():
        return direct.resolve()
    stem = Path(value).stem
    candidates: List[Path] = []
    if kind == "receptor":
        dirs = [project / "prepared/receptors", project / "inputs/receptors"] if prepared else [project / "inputs/receptors"]
    else:
        dirs = [project / "prepared/ligands", project / "inputs/ligands"] if prepared else [project / "inputs/ligands"]
    for directory in dirs:
        for suffix in suffixes:
            candidates.append(directory / f"{stem}{suffix}")
    found = next((p for p in candidates if p.exists()), None)
    if found:
        return found.resolve()
    raise FileNotFoundError(f"无法解析 {kind} 文件“{value}”。检查输入目录和任务表拼写。")


def pair_parameters(row: dict, cfg: WorkflowConfig) -> dict:
    required = ["pair_id", "receptor", "ligand", "center_x", "center_y", "center_z"]
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"缺少字段：{', '.join(missing)}")
    default_box = float(cfg.default("default_box_size", 22.0))
    return {
        "pair_id": safe_id(row["pair_id"]),
        "receptor": row["receptor"],
        "ligand": row["ligand"],
        "center_x": float(row["center_x"]),
        "center_y": float(row["center_y"]),
        "center_z": float(row["center_z"]),
        "size_x": float_value(row.get("size_x"), default_box),
        "size_y": float_value(row.get("size_y"), default_box),
        "size_z": float_value(row.get("size_z"), default_box),
        "exhaustiveness": int_value(row.get("exhaustiveness"), int(cfg.default("default_exhaustiveness", 8))),
        "num_modes": int_value(row.get("num_modes"), int(cfg.default("default_num_modes", 9))),
        "cpu": int_value(row.get("cpu"), int(cfg.default("default_cpu", 4))),
        "energy_range": float_value(row.get("energy_range"), float(cfg.default("default_energy_range", 3.0))),
        "notes": row.get("notes", ""),
    }


def parse_best_affinity(log_file: Path) -> Optional[float]:
    if not log_file.exists():
        return None
    pattern = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def write_vina_config(params: dict, receptor: Path, ligand: Path, pose: Path, log: Path, config_file: Path) -> None:
    lines = [
        f"receptor = {receptor}", f"ligand = {ligand}",
        f"center_x = {params['center_x']}", f"center_y = {params['center_y']}", f"center_z = {params['center_z']}",
        f"size_x = {params['size_x']}", f"size_y = {params['size_y']}", f"size_z = {params['size_z']}",
        f"exhaustiveness = {params['exhaustiveness']}", f"num_modes = {params['num_modes']}",
        f"cpu = {params['cpu']}", f"energy_range = {params['energy_range']}",
        f"out = {pose}", f"log = {log}",
    ]
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary(summary_path: Path, records: List[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def dock_pairs(cfg: WorkflowConfig, pairs_file: Path, dry_run: bool, force: bool, only: str) -> int:
    rows = read_pairs(pairs_file)
    selected = [row for row in rows if bool_value(row.get("enabled"), True)]
    if only:
        only_set = {safe_id(x) for x in re.split(r"[,;\s]+", only) if x.strip()}
        selected = [row for row in selected if safe_id(row.get("pair_id", "")) in only_set]
    if not selected:
        print("没有启用的对接任务。请检查 docking_pairs.csv 的 enabled 和 pair_id。")
        return 1
    if not cfg.vina_exe.exists() and not dry_run:
        raise FileNotFoundError(f"Vina 不存在：{cfg.vina_exe}")
    summary_path = cfg.project / "results" / "summary_vina_results.csv"
    existing: Dict[str, dict] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing = {row.get("pair_id", ""): row for row in csv.DictReader(f)}
    records: Dict[str, dict] = dict(existing)
    failures = 0
    timeout = int(cfg.default("docking_timeout_seconds", 3600))
    for index, row in enumerate(selected, start=1):
        start = time.time()
        pair_id = safe_id(row.get("pair_id", f"pair_{index}"))
        print(f"\n[{index}/{len(selected)}] {pair_id}")
        try:
            params = pair_parameters(row, cfg)
            receptor = resolve_input_path(cfg.project, params["receptor"], "receptor", {".pdbqt"}, prepared=True)
            ligand = resolve_input_path(cfg.project, params["ligand"], "ligand", {".pdbqt"}, prepared=True)
            pair_dir = cfg.project / "results" / "pairs" / pair_id
            docking_dir = pair_dir / "docking"
            images_dir = pair_dir / "images"
            docking_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            pose = docking_dir / f"{pair_id}_out.pdbqt"
            vina_log = docking_dir / f"{pair_id}_vina.log"
            vina_config = docking_dir / f"{pair_id}_vina_config.txt"
            metadata = pair_dir / "run_metadata.json"
            write_vina_config(params, receptor, ligand, pose, vina_log, vina_config)
            meta = dict(params)
            meta.update({
                "receptor_path": str(receptor), "ligand_path": str(ligand),
                "pose_file": str(pose), "vina_log": str(vina_log), "vina_config": str(vina_config),
            })
            metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            if dry_run:
                status, message, affinity = "dry_run", "仅检查，不运行 Vina", None
                print(f"  receptor: {receptor}")
                print(f"  ligand:   {ligand}")
                print(f"  config:   {vina_config}")
            elif pose.exists() and vina_log.exists() and not force:
                status, message = "skipped", "已有结果，未覆盖"
                affinity = parse_best_affinity(vina_log)
            else:
                command = [str(cfg.vina_exe), "--config", str(vina_config)]
                cp = run_command(command, cfg.project, timeout=timeout)
                # Vina may ignore log= in config depending on version; persist console output.
                console_text = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
                if not vina_log.exists() or vina_log.stat().st_size == 0:
                    vina_log.write_text(console_text, encoding="utf-8", errors="replace")
                affinity = parse_best_affinity(vina_log)
                if cp.returncode == 0 and pose.exists():
                    status, message = "success", "对接完成"
                else:
                    status, message = "failed", f"Vina 退出码 {cp.returncode}"
                    failures += 1
            elapsed = round(time.time() - start, 2)
            record = {
                "pair_id": pair_id, "status": status,
                "best_affinity_kcal_mol": "" if affinity is None else f"{affinity:.3f}",
                "receptor": str(receptor), "ligand": str(ligand),
                "pose_file": str(pose), "log_file": str(vina_log),
                "elapsed_seconds": elapsed, "message": message,
                **{k: params[k] for k in ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "exhaustiveness", "num_modes", "cpu", "energy_range"]},
            }
            records[pair_id] = record
            print(f"  {status}: {message}; affinity={record['best_affinity_kcal_mol'] or 'NA'}")
        except Exception as exc:
            failures += 1
            records[pair_id] = {
                "pair_id": pair_id, "status": "failed", "best_affinity_kcal_mol": "",
                "elapsed_seconds": round(time.time() - start, 2), "message": str(exc),
            }
            print(f"  FAIL: {exc}")
        update_summary(summary_path, list(records.values()))
    print(f"\n汇总表：{summary_path}")
    return 1 if failures else 0


def write_base_pymol_scripts(cfg: WorkflowConfig, pairs_file: Path) -> int:
    rows = read_pairs(pairs_file)
    count = 0
    for row in rows:
        if not bool_value(row.get("enabled"), True):
            continue
        pair_id = safe_id(row.get("pair_id", ""))
        if not pair_id:
            continue
        pair_dir = cfg.project / "results" / "pairs" / pair_id
        metadata_file = pair_dir / "run_metadata.json"
        if metadata_file.exists():
            meta = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
            receptor = Path(meta["receptor_path"])
            pose = Path(meta["pose_file"])
        else:
            try:
                params = pair_parameters(row, cfg)
                receptor = resolve_input_path(cfg.project, params["receptor"], "receptor", {".pdbqt"}, prepared=True)
                pose = pair_dir / "docking" / f"{pair_id}_out.pdbqt"
            except Exception as exc:
                print(f"[SKIP] {pair_id}: {exc}")
                continue
        if not receptor.exists() or not pose.exists():
            print(f"[SKIP] {pair_id}: 缺少 receptor 或 pose")
            continue
        images_dir = pair_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        pml = images_dir / f"{pair_id}_export_images.pml"
        pml.write_text(
            "reinitialize\n"
            f'load "{str(receptor).replace(os.sep, "/")}", receptor\n'
            f'load "{str(pose).replace(os.sep, "/")}", docked_ligand\n'
            "hide everything\nshow cartoon, receptor\nshow sticks, docked_ligand\n"
            "bg_color white\norient receptor\nzoom receptor, 8\n",
            encoding="ascii", errors="ignore",
        )
        count += 1
        print(f"[OK] {pml}")
    print(f"生成基础 PyMOL 脚本：{count} 组")
    return 0 if count else 1


def organize_images(cfg: WorkflowConfig) -> int:
    dest = cfg.project / "results" / "final_figures"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    patterns = ["*__composite_B.png", "*__overview_raw.png", "*__pocket_raw.png"]
    for pattern in patterns:
        for source in (cfg.project / "results" / "pairs").rglob(pattern):
            target = dest / source.name
            shutil.copy2(source, target)
            copied += 1
    print(f"归档图片：{copied} 张 -> {dest}")
    return 0


def export_plot_data(cfg: WorkflowConfig) -> int:
    summary = cfg.project / "results" / "summary_vina_results.csv"
    if not summary.exists():
        raise FileNotFoundError(f"未找到：{summary}")
    rows = []
    with summary.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            score = str(row.get("best_affinity_kcal_mol", "")).strip()
            if not score:
                continue
            try:
                value = float(score)
            except ValueError:
                continue
            rows.append({
                "pair_id": row.get("pair_id", ""),
                "affinity_kcal_mol": value,
                "status": row.get("status", ""),
                "receptor": row.get("receptor", ""),
                "ligand": row.get("ligand", ""),
            })
    rows.sort(key=lambda x: x["affinity_kcal_mol"])
    out = cfg.project / "results" / "plot_data" / "vina_affinity_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "pair_id", "affinity_kcal_mol", "status", "receptor", "ligand"])
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})
    print(f"绘图数据：{out}")
    return 0


def plot_scores(cfg: WorkflowConfig) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，请运行：python -m pip install -r requirements.txt") from exc
    data_file = cfg.project / "results" / "plot_data" / "vina_affinity_ranking.csv"
    if not data_file.exists():
        export_plot_data(cfg)
    with data_file.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("没有可绘制的 affinity 数据。")
    labels = [row["pair_id"] for row in rows]
    values = [float(row["affinity_kcal_mol"]) for row in rows]
    fig_h = max(4.5, 0.42 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    y = list(range(len(rows)))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Vina affinity (kcal/mol)")
    ax.set_title("AutoDock Vina affinity ranking")
    ax.axvline(0, linewidth=0.8)
    for yi, value in zip(y, values):
        ax.text(value, yi, f" {value:.2f}", va="center", ha="right" if value < 0 else "left")
    fig.tight_layout()
    out = cfg.project / "results" / "plot_data" / "vina_affinity_ranking.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"排序图：{out}")
    return 0


def init_templates(project: Path) -> int:
    ensure_project_dirs(project)
    pairs = project / "docking_pairs.csv"
    blank = project / "docking_pairs_blank_template.csv"
    if not blank.exists():
        with blank.open("w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=PAIR_COLUMNS).writeheader()
    if not pairs.exists():
        with pairs.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "enabled": "0", "pair_id": "example_pair", "receptor": "receptor_name",
                "ligand": "ligand_name", "center_x": "0", "center_y": "0", "center_z": "0",
                "size_x": "22", "size_y": "22", "size_z": "22",
                "exhaustiveness": "8", "num_modes": "9", "cpu": "4", "energy_range": "3",
                "notes": "将 enabled 改为 1 后才会运行",
            })
    print(f"项目目录已初始化：{project}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoDock Vina complete workflow")
    parser.add_argument("--project", default=".", help="项目根目录")
    parser.add_argument("--config", default="config.json", help="配置文件")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")
    sub.add_parser("check")
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--force", action="store_true")
    p_extract = sub.add_parser("extract-centers")
    p_extract.add_argument("--padding", type=float, default=6.0)
    p_extract.add_argument("--min-size", type=float, default=20.0)
    p_center = sub.add_parser("calculate-center")
    p_center.add_argument("--receptor", default="")
    p_center.add_argument("--residues", default=None)
    p_center.add_argument("--atoms", default=None)
    p_center.add_argument("--padding", type=float, default=6.0)
    p_center.add_argument("--min-size", type=float, default=20.0)
    p_dock = sub.add_parser("dock")
    p_dock.add_argument("--pairs", default="docking_pairs.csv")
    p_dock.add_argument("--dry-run", action="store_true")
    p_dock.add_argument("--force", action="store_true")
    p_dock.add_argument("--only", default="")
    p_pymol = sub.add_parser("make-pymol")
    p_pymol.add_argument("--pairs", default="docking_pairs.csv")
    sub.add_parser("organize-images")
    sub.add_parser("export-plot-data")
    sub.add_parser("plot-scores")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project = Path(args.project).resolve()
    if args.command == "init":
        return init_templates(project)
    cfg = load_config(project, args.config)
    ensure_project_dirs(project)
    if args.command == "check":
        return check_environment(cfg)
    if args.command == "prepare":
        return prepare_all(cfg, force=args.force)
    if args.command == "extract-centers":
        return extract_centers(cfg, args.padding, args.min_size)
    if args.command == "calculate-center":
        return interactive_calculate_center(cfg, args)
    if args.command == "dock":
        pairs = Path(args.pairs)
        if not pairs.is_absolute():
            pairs = project / pairs
        return dock_pairs(cfg, pairs, args.dry_run, args.force, args.only)
    if args.command == "make-pymol":
        pairs = Path(args.pairs)
        if not pairs.is_absolute():
            pairs = project / pairs
        return write_base_pymol_scripts(cfg, pairs)
    if args.command == "organize-images":
        return organize_images(cfg)
    if args.command == "export-plot-data":
        return export_plot_data(cfg)
    if args.command == "plot-scores":
        return plot_scores(cfg)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户取消。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{exc}")
        raise SystemExit(1)
