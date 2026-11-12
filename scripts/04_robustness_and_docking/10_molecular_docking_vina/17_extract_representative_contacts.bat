@echo off
chcp 65001 >nul
cd /d "%~dp0"
python formal_vina_pipeline.py --project . --config config.json interactions --multiseed-summary results/formal/03_summary/vina_multiseed_summary.csv --docking-manifest results/formal/01_audit/formal_docking_manifest_audited.csv --output-dir results/formal/04_interactions
pause
