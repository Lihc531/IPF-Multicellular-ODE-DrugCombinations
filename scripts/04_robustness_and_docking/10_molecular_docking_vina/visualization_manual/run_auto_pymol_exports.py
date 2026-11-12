#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG = ROOT / "config.json"
PAIRS_ROOT = ROOT / "results" / "pairs"
LOG = ROOT / "logs" / "pymol_export_log.txt"


def load_pymol() -> Path:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    raw = str(cfg.get("pymol_exe", "")).strip().strip('"').strip("'")
    if not raw:
        raise ValueError("config.json 中缺少 pymol_exe。")
    raw = os.path.expandvars(os.path.expanduser(raw))
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"PyMOL 不存在：{path}")
    return path.resolve()


def run_pml(executable: Path, pml: Path, handle) -> int:
    if executable.suffix.lower() in {".bat", ".cmd"}:
        launch_dir = ROOT / "results" / "_auto_launchers"
        launch_dir.mkdir(parents=True, exist_ok=True)
        launcher = launch_dir / f"launch_{uuid.uuid4().hex}.cmd"
        launcher.write_text(
            f'@echo off\r\ncall "{executable}" -cq "{pml}"\r\nexit /b %errorlevel%\r\n',
            encoding="utf-8",
        )
        cp = subprocess.run(["cmd.exe", "/d", "/c", str(launcher)], cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        launcher.unlink(missing_ok=True)
        return cp.returncode
    cp = subprocess.run([str(executable), "-cq", str(pml)], cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    return cp.returncode


def main() -> int:
    executable = load_pymol()
    pml_files = sorted(PAIRS_ROOT.rglob("*_export_images.pml"))
    if not pml_files:
        raise FileNotFoundError("没有自动导出 PML。请先运行 05_generate_pymol_scripts.bat。")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with LOG.open("w", encoding="utf-8", errors="replace") as log:
        for index, pml in enumerate(pml_files, 1):
            print(f"[{index}/{len(pml_files)}] {pml.stem}")
            log.write(f"\n===== {pml} =====\n")
            code = run_pml(executable, pml, log)
            log.write(f"Exit code: {code}\n")
            failures += 1 if code else 0
        log.write("\n===== COMPOSE =====\n")
        cp = subprocess.run(
            ["python", str(ROOT / "compose_pymol_figures_bstyle.py")],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
        )
        failures += 1 if cp.returncode else 0
    print(f"日志：{LOG}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}")
        raise SystemExit(1)
