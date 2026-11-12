@echo off
chcp 65001 >nul
cd /d "%~dp0"
python formal_vina_pipeline.py --project . --config config.json sync --formal-evidence results/formal/05_final/vina_formal_evidence.csv --grade-filter A,B
pause
