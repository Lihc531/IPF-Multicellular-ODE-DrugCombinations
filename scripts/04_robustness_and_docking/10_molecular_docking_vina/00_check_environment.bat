@echo off
chcp 65001 >nul

call E:\Anaconda3\Anaconda\condabin\conda.bat activate vina_env

cd /d "%~dp0"
python vina_workflow.py --project . --config config.json check

echo.
pause