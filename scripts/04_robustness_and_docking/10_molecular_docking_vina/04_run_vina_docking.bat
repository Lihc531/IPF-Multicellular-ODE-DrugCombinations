@echo off
chcp 65001 >nul
cd /d "%~dp0"
python vina_workflow.py --project . --config config.json dock --pairs docking_pairs.csv
pause
