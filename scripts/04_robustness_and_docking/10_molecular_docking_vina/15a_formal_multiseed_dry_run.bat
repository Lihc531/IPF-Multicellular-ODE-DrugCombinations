@echo off
chcp 65001 >nul
cd /d "%~dp0"
python formal_vina_pipeline.py --project . --config config.json run --manifest results/formal/01_audit/formal_docking_manifest_audited.csv --output-dir results/formal/02_multiseed_dry_run --dry-run
pause
