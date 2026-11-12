@echo off
chcp 65001 >nul
cd /d "%~dp0"
python vina_workflow.py --project . --config config.json extract-centers --padding 6 --min-size 20
pause
