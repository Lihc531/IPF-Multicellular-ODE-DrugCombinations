from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bstyle_common import (
    PairSpec,
    load_manifest,
    make_pair_specs,
    project_root_from_arg,
    pymol_selection_for_residue,
    resolve_pymol,
    to_pymol_path,
    write_manifest,
)


OVERVIEW_SIZE = (760, 840)
POCKET_SIZE = (1160, 840)


def pml_header() -> list[str]:
    return [
        "reinitialize",
        "set quiet, 1",
        "bg_color white",
        "set orthoscopic, on",
        "set antialias, 2",
        "set ray_trace_mode, 1",
        "set ray_shadows, off",
        "set depth_cue, 0",
        "set specular, 0.18",
        "set shininess, 18",
        "set ambient, 0.52",
        "set direct, 0.45",
        "set reflect, 0.10",
        "set_color protein_green, [0.72, 0.94, 0.72]",
        "set_color protein_surface, [0.88, 0.96, 0.89]",
        "set_color pocket_carbon, [0.78, 0.59, 0.38]",
        "set_color ligand_carbon, [0.00, 0.78, 0.82]",
        "set_color leader_gray, [0.45, 0.45, 0.45]",
    ]


def load_commands(spec: PairSpec) -> list[str]:
    receptor = to_pymol_path(Path(spec.receptor))
    pose = to_pymol_path(Path(spec.pose))
    return [
        f'load "{receptor}", receptor',
        f'load "{pose}", ligand',
        "frame 1",
        "remove solvent",
        "hide everything, all",
        "show sticks, ligand",
        "set stick_radius, 0.22, ligand",
        "color ligand_carbon, ligand and elem C",
        "color blue, ligand and elem N",
        "color red, ligand and elem O",
        "color yellow, ligand and elem S",
        "color orange, ligand and elem P",
        "color green, ligand and elem F+Cl+Br+I",
    ]


def make_overview_pml(spec: PairSpec) -> str:
    output = to_pymol_path(Path(spec.overview_png))
    lines = pml_header() + load_commands(spec)
    lines += [
        "show surface, receptor",
        "color protein_surface, receptor",
        "set transparency, 0.26, receptor",
        "set surface_quality, 1",
        "set two_sided_lighting, on",
        "orient receptor",
        "zoom receptor, 7.0, complete=1",
        "center receptor",
        f'png "{output}", width={OVERVIEW_SIZE[0]}, height={OVERVIEW_SIZE[1]}, dpi=300, ray=1',
        "quit",
    ]
    return "\n".join(lines) + "\n"


def contact_selection(spec: PairSpec) -> str:
    selections = [f"({pymol_selection_for_residue(c)})" for c in spec.contacts]
    return " or ".join(selections)


def label_layout_code(spec: PairSpec, label_count: int = 5) -> str:
    chosen = spec.contacts[:label_count]
    payload = [
        {
            "label": c.label,
            "selection": pymol_selection_for_residue(c),
            "serial": c.receptor_atom_serial,
        }
        for c in chosen
    ]
    data = json.dumps(payload)
    # Screen-plane offsets, arranged to emulate the supplied reference figure.
    offsets = [(-4.5, 4.2), (0.0, 5.0), (4.5, 4.2), (-3.4, -4.4), (3.4, -4.4), (0.0, -5.2)]
    offsets_json = json.dumps(offsets)
    return f'''python
from pymol import cmd
import math
contacts = {data}
offsets = {offsets_json}
model = cmd.get_model("ligand")
coords = [a.coord for a in model.atom]
center = [sum(p[i] for p in coords) / len(coords) for i in range(3)]
view = cmd.get_view()
right = [view[0], view[1], view[2]]
up = [view[3], view[4], view[5]]
for i, item in enumerate(contacts, start=1):
    dx, dy = offsets[i - 1]
    pos = [center[j] + dx * right[j] + dy * up[j] for j in range(3)]
    anchor = f"label_anchor_{{i}}"
    cmd.pseudoatom(anchor, pos=pos, label=item["label"])
    atom_sel = f'(receptor and id {{item["serial"]}})'
    cmd.distance(f"leader_{{i}}", anchor, atom_sel)
cmd.hide("nonbonded", "label_anchor_*")
cmd.show("labels", "label_anchor_*")
cmd.hide("labels", "leader_*")
cmd.set("dash_color", "leader_gray", "leader_*")
cmd.set("dash_width", 2.0, "leader_*")
cmd.set("dash_gap", 0.28, "leader_*")
cmd.set("dash_length", 0.18, "leader_*")
cmd.set("label_color", "gray20", "label_anchor_*")
cmd.set("label_bg_color", "white", "label_anchor_*")
cmd.set("label_outline_color", "gray70", "label_anchor_*")
cmd.set("label_font_id", 7, "label_anchor_*")
cmd.set("label_size", 18, "label_anchor_*")
cmd.zoom("ligand or pocket or label_anchor_*", buffer=0.8, complete=1)
python end'''


def make_pocket_pml(spec: PairSpec, quit_after: bool = True, render: bool = True) -> str:
    output = to_pymol_path(Path(spec.pocket_png))
    pocket = contact_selection(spec)
    lines = pml_header() + load_commands(spec)
    lines += [
        f"select pocket, {pocket}",
        "show cartoon, receptor",
        "color protein_green, receptor",
        "set cartoon_transparency, 0.35, receptor",
        "show sticks, pocket",
        "set stick_radius, 0.17, pocket",
        "color pocket_carbon, pocket and elem C",
        "color blue, pocket and elem N",
        "color red, pocket and elem O",
        "color yellow, pocket and elem S",
        "color orange, pocket and elem P",
        "color green, pocket and elem F+Cl+Br+I",
        "orient ligand or pocket",
        "zoom ligand or pocket, 3.3, complete=1",
        "center ligand",
        label_layout_code(spec),
    ]
    if render:
        lines.append(
            f'png "{output}", width={POCKET_SIZE[0]}, height={POCKET_SIZE[1]}, dpi=300, ray=1'
        )
    if not quit_after:
        lines += [
            "set quiet, 0",
            "# Manual review scene: adjust the view, then save with the PyMOL png command.",
        ]
    else:
        lines.append("quit")
    return "\n".join(lines) + "\n"


def write_assets(specs: list[PairSpec]) -> None:
    for spec in specs:
        Path(spec.overview_pml).write_text(make_overview_pml(spec), encoding="utf-8", newline="\n")
        Path(spec.pocket_pml).write_text(make_pocket_pml(spec), encoding="utf-8", newline="\n")
        Path(spec.manual_pml).write_text(
            make_pocket_pml(spec, quit_after=False, render=False), encoding="utf-8", newline="\n"
        )


def run_pymol(project: Path, specs: list[PairSpec]) -> None:
    launcher = resolve_pymol(project)
    if not launcher:
        raise FileNotFoundError(
            "PyMOL executable was not found. Set config.json -> pymol_exe to the real pymol.exe path."
        )
    for spec in specs:
        for pml in (spec.overview_pml, spec.pocket_pml):
            command = launcher + ["-cq", pml]
            print("[PYMOL]", " ".join(f'"{part}"' if " " in part else part for part in command))
            completed = subprocess.run(command, cwd=project)
            if completed.returncode != 0:
                raise RuntimeError(f"PyMOL failed with exit code {completed.returncode}: {pml}")
        print(f"[OK] Rendered panels: {spec.pair_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate B-style PyMOL overview and pocket scripts from formal representative poses."
    )
    parser.add_argument("--project", default=".", help="Project root directory")
    parser.add_argument("--render", action="store_true", help="Render overview and pocket PNGs immediately")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Residue contact cutoff in Angstrom")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = project_root_from_arg(args.project)
    specs = make_pair_specs(project, cutoff=args.cutoff)
    if not specs:
        raise RuntimeError("No enabled visualization tasks were found.")
    write_assets(specs)
    manifest = write_manifest(project, specs)
    print(f"[OK] Generated PyMOL assets for {len(specs)} pair(s).")
    print(f"[OK] Manifest: {manifest}")
    for spec in specs:
        labels = ", ".join(c.label for c in spec.contacts[:5])
        print(f"  - {spec.pair_id}: labels={labels}")
    if args.render:
        run_pymol(project, specs)
    else:
        print("[INFO] Rendering skipped. Use --render or run 11_open_manual_pymol_menu.bat.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
