@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo Generate and render B-style PyMOL panels
echo ============================================================
python "visualization_manual\generate_manual_pymol_assets.py" --project "." --render
if errorlevel 1 (
  echo.
  echo [FAIL] B-style PyMOL asset generation failed.
  echo Check config.json pymol_exe and confirm script 19 has synced the representative pose.
  pause
  exit /b 1
)

echo.
echo [OK] Overview and pocket panels were rendered.
echo Next: 12_compose_bstyle_figures.bat
pause
endlocal
