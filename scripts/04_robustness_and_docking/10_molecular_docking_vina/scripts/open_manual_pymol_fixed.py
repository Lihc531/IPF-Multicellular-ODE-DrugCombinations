from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def norm_path(path: Path) -> str:
    return path.resolve().as_posix()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}: {exc}") from exc


def resolve_path(project: Path, value: str) -> Path:
    value = (value or "").strip().strip('"')
    if not value:
        return Path()
    p = Path(value.replace("\\", os.sep))
    return p if p.is_absolute() else project / p


def find_pymol(project: Path, config: dict) -> Path:
    configured = str(config.get("pymol_exe", "")).strip()
    candidates: list[Path] = []
    if configured:
        p = resolve_path(project, configured)
        if p.suffix.lower() == ".exe":
            candidates.append(p)

    candidates.extend(
        Path(text)
        for text in (
            r"E:\pymol\Scripts\pymol.exe",
            r"E:\PyMOL\PyMOL.exe",
            r"E:\Anaconda3\Anaconda\envs\pymol\Scripts\pymol.exe",
            r"C:\Program Files\PyMOL\PyMOL.exe",
        )
    )
    from_path = shutil.which("pymol.exe") or shutil.which("pymol")
    if from_path:
        candidates.append(Path(from_path))

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()

    pretty = "\n".join(f"  {c}" for c in candidates)
    raise RuntimeError(
        "PyMOL executable was not found. Set config.json pymol_exe to the real "
        f"pymol.exe path. Checked:\n{pretty}"
    )


def read_tasks(csv_path: Path, project: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise RuntimeError(f"Task table not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("enabled", "1")).strip() not in {"0", "false", "False"}
        ]

    tasks: list[dict[str, str]] = []
    for row in rows:
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            continue
        receptor = resolve_path(project, row.get("receptor", ""))
        pose = project / "results" / "pairs" / pair_id / "docking" / f"{pair_id}_out.pdbqt"
        tasks.append(
            {
                **row,
                "pair_id": pair_id,
                "receptor_abs": str(receptor.resolve()),
                "pose_abs": str(pose.resolve()),
            }
        )
    if not tasks:
        raise RuntimeError(f"No enabled formal visualization tasks in {csv_path}")
    return tasks


def affinity_map(project: Path) -> dict[str, str]:
    path = project / "results" / "summary_vina_results.csv"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pair_id = (row.get("pair_id") or row.get("task_id") or "").strip()
                val = (
                    row.get("affinity")
                    or row.get("median_affinity")
                    or row.get("median_affinity_kcal_mol")
                    or row.get("best_affinity")
                    or ""
                )
                if pair_id and str(val).strip():
                    out[pair_id] = str(val).strip()
    except Exception:
        return {}
    return out


def choose_task(tasks: list[dict[str, str]], affinities: dict[str, str]) -> dict[str, str] | None:
    print("Available formal docking figures:")
    for index, task in enumerate(tasks, start=1):
        affinity = affinities.get(task["pair_id"], "")
        suffix = f"  affinity={affinity}" if affinity else ""
        print(f"[{index:2d}] {task['pair_id']}{suffix}")
    print("[ 0] Exit")
    while True:
        raw = input("Select number: ").strip()
        try:
            number = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if number == 0:
            return None
        if 1 <= number <= len(tasks):
            return tasks[number - 1]
        print("Number out of range.")


def write_pymol_python(project: Path, task: dict[str, str]) -> Path:
    pair_id = task["pair_id"]
    receptor = Path(task["receptor_abs"])
    pose = Path(task["pose_abs"])
    if not receptor.is_file():
        raise RuntimeError(f"Receptor not found: {receptor}")
    if not pose.is_file():
        raise RuntimeError(f"Formal representative pose not found: {pose}")

    figure_dir = project / "results" / "pairs" / pair_id / "figures_bstyle"
    figure_dir.mkdir(parents=True, exist_ok=True)
    overview = figure_dir / f"{pair_id}__overview.png"
    pocket = figure_dir / f"{pair_id}__pocket.png"
    session = figure_dir / f"{pair_id}__manual_view.pse"
    script_dir = project / "visualization_manual" / "generated"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{pair_id}__manual_view.py"

    template = r'''from pymol import cmd

PAIR_ID = __PAIR_ID__
RECEPTOR = __RECEPTOR__
POSE = __POSE__
OVERVIEW = __OVERVIEW__
POCKET = __POCKET__
SESSION = __SESSION__

cmd.reinitialize()
cmd.bg_color("white")
cmd.set("orthoscopic", 1)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1)
cmd.set("ray_shadows", 0)
cmd.set("depth_cue", 0)
cmd.set("specular", 0.18)
cmd.set("shininess", 18)
cmd.set("ambient", 0.52)
cmd.set("direct", 0.45)
cmd.set("reflect", 0.10)
cmd.set_color("protein_green", [0.72, 0.94, 0.72])
cmd.set_color("protein_surface", [0.88, 0.96, 0.89])
cmd.set_color("pocket_carbon", [0.78, 0.59, 0.38])
cmd.set_color("ligand_carbon", [0.00, 0.78, 0.82])

cmd.load(RECEPTOR, "receptor")
cmd.load(POSE, "ligand")
cmd.frame(1)
cmd.remove("solvent")
cmd.hide("everything", "all")

# Automatic overview image. Python API avoids the quoted-path PML bug.
cmd.show("sticks", "ligand")
cmd.set("stick_radius", 0.22, "ligand")
cmd.color("ligand_carbon", "ligand and elem C")
cmd.color("blue", "ligand and elem N")
cmd.color("red", "ligand and elem O")
cmd.color("yellow", "ligand and elem S")
cmd.color("orange", "ligand and elem P")
cmd.color("green", "ligand and elem F+Cl+Br+I")
cmd.show("surface", "receptor")
cmd.color("protein_surface", "receptor")
cmd.set("transparency", 0.26, "receptor")
cmd.set("surface_quality", 1)
cmd.set("two_sided_lighting", 1)
cmd.orient("receptor")
cmd.zoom("receptor", 7.0, complete=1)
cmd.center("receptor")
cmd.png(OVERVIEW, width=760, height=840, dpi=300, ray=1, quiet=0)

# Prepare an editable pocket view and keep PyMOL open.
cmd.hide("everything", "all")
cmd.show("cartoon", "receptor")
cmd.color("protein_green", "receptor")
cmd.select("pocket", "byres (receptor within 4.0 of ligand)")
cmd.show("sticks", "pocket")
cmd.set("stick_radius", 0.18, "pocket")
cmd.color("pocket_carbon", "pocket and elem C")
cmd.color("blue", "pocket and elem N")
cmd.color("red", "pocket and elem O")
cmd.color("yellow", "pocket and elem S")
cmd.show("sticks", "ligand")
cmd.set("stick_radius", 0.24, "ligand")
cmd.color("ligand_carbon", "ligand and elem C")
cmd.color("blue", "ligand and elem N")
cmd.color("red", "ligand and elem O")
cmd.color("yellow", "ligand and elem S")
cmd.color("orange", "ligand and elem P")
cmd.color("green", "ligand and elem F+Cl+Br+I")
cmd.orient("pocket or ligand")
cmd.zoom("pocket or ligand", 4.0, complete=1)
cmd.center("ligand")


def save_pocket():
    cmd.png(POCKET, width=1160, height=840, dpi=300, ray=1, quiet=0)
    print("[OK] Pocket image saved:", POCKET)


def save_manual_session():
    cmd.save(SESSION)
    print("[OK] PyMOL session saved:", SESSION)


def reset_pocket():
    cmd.orient("pocket or ligand")
    cmd.zoom("pocket or ligand", 4.0, complete=1)
    cmd.center("ligand")


cmd.extend("save_pocket", save_pocket)
cmd.extend("save_manual_session", save_manual_session)
cmd.extend("reset_pocket", reset_pocket)

print("============================================================")
print("Manual pocket view:", PAIR_ID)
print("Adjust the view with the mouse, then type: save_pocket")
print("Optional session backup: save_manual_session")
print("Reset automatic pocket view: reset_pocket")
print("After saving, close PyMOL and run 12_compose_bstyle_figures.bat")
print("============================================================")
'''
    replacements = {
        "__PAIR_ID__": repr(pair_id),
        "__RECEPTOR__": repr(norm_path(receptor)),
        "__POSE__": repr(norm_path(pose)),
        "__OVERVIEW__": repr(norm_path(overview)),
        "__POCKET__": repr(norm_path(pocket)),
        "__SESSION__": repr(norm_path(session)),
    }
    code = template
    for key, value in replacements.items():
        code = code.replace(key, value)
    script_path.write_text(code, encoding="utf-8", newline="\n")
    return script_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    # Defensive cleanup for Windows BAT arguments. A quoted path ending in a
    # backslash can be parsed as a literal trailing quote (for example Safe").
    raw_project = str(args.project).strip().strip('"').rstrip('"')
    project = Path(raw_project).resolve()

    config = read_json(project / "config.json")
    pymol_exe = find_pymol(project, config)
    tasks = read_tasks(project / "docking_pairs_formal_visualization.csv", project)
    task = choose_task(tasks, affinity_map(project))
    if task is None:
        return 0

    script = write_pymol_python(project, task)
    print(f"[OK] PyMOL executable: {pymol_exe}")
    print(f"[OK] Manual script: {script}")
    print("[INFO] PyMOL will stay open. Adjust the view and type save_pocket.")
    subprocess.Popen([str(pymol_exe), "-r", str(script)], cwd=str(project))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
