@echo off
chcp 65001 >nul
cd /d "%~dp0"
python formal_vina_pipeline.py --project . --config config.json audit --manifest docking_pairs_formal.csv --output-dir results/formal/01_audit
pause
