@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo Compose B-style overview plus pocket figures
echo ============================================================
python "visualization_manual\compose_bstyle_figures.py" --project "."
if errorlevel 1 (
  echo.
  echo [FAIL] Figure composition failed.
  echo If Pillow is missing, run: python -m pip install Pillow
  pause
  exit /b 1
)

echo.
echo [OK] Composite B-style figures completed.
pause
endlocal
