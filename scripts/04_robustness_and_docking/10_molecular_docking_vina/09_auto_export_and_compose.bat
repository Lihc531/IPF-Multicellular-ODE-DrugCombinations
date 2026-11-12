@echo off
chcp 65001 >nul
cd /d "%~dp0"
python visualization_manual\run_auto_pymol_exports.py
pause
