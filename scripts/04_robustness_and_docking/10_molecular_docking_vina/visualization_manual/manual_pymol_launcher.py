from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bstyle_common import load_manifest, project_root_from_arg, resolve_pymol


def choose_pair(pairs: list[dict]) -> dict | None:
    print("\nAvailable formal docking figures:")
    for index, pair in enumerate(pairs, start=1):
        affinity = str(pair.get("affinity", "")).strip()
        suffix = f"  affinity={affinity}" if affinity else ""
        print(f"[{index:2d}] {pair['pair_id']}{suffix}")
    print("[ 0] Exit")
    while True:
        raw = input("Select number: ").strip()
        if raw == "0":
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= value <= len(pairs):
            return pairs[value - 1]
        print("Selection out of range.")


def choose_scene() -> str | None:
    print("\nScene:")
    print("[1] Pocket close-up for manual review")
    print("[2] Whole-protein overview")
    print("[0] Back")
    while True:
        raw = input("Select scene: ").strip()
        if raw == "0":
            return None
        if raw == "1":
            return "manual_pml"
        if raw == "2":
            return "overview_pml"
        print("Please enter 0, 1, or 2.")


def main() -> int:
    project = project_root_from_arg(Path(__file__).resolve().parents[1])
    manifest = load_manifest(project)
    pairs = list(manifest.get("pairs", []))
    if not pairs:
        raise RuntimeError("The B-style manifest contains no pairs.")
    launcher = resolve_pymol(project)
    if not launcher:
        raise FileNotFoundError(
            "PyMOL executable was not found. Set config.json -> pymol_exe to the real pymol.exe path."
        )
    while True:
        pair = choose_pair(pairs)
        if pair is None:
            return 0
        scene_key = choose_scene()
        if scene_key is None:
            continue
        pml = Path(pair[scene_key])
        if not pml.exists():
            raise FileNotFoundError(f"PyMOL script not found: {pml}")
        command = launcher + [str(pml)]
        print("[OPEN]", " ".join(f'"{part}"' if " " in part else part for part in command))
        # Popen is intentional: the menu remains responsive and no BAT-to-BAT call is required.
        subprocess.Popen(command, cwd=project)
        print(f"[OK] Opened: {pair['pair_id']}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
