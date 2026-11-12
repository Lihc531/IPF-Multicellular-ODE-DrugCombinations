@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "REPAIR=scripts\repair_simvastatin_zero_coords.py"
set "LOG=logs\formal_ligand_prep\SIMVASTATIN_ACID_zero_coord_rescue.log"

echo ============================================================
echo Repair SIMVASTATIN_ACID zero-coordinate MOL2
echo ============================================================
echo.
echo Project directory:
echo   %CD%
echo.

if not exist "%REPAIR%" (
  echo [FATAL] Missing %REPAIR%
  echo Extract the complete repair package into the project root.
  goto :fail
)

if not exist "inputs\ligands\SIMVASTATIN_ACID_pH7p4_3D_seed.sdf" (
  echo [FATAL] Missing bundled 3D seed SDF.
  echo Extract the complete repair package into the project root.
  goto :fail
)

if not exist "config.json" (
  echo [FATAL] Missing config.json in %CD%
  goto :fail
)

where python >nul 2>&1
if errorlevel 1 (
  echo [FATAL] System Python was not found.
  goto :fail
)

python "%REPAIR%" --project . --config config.json
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
  echo [OK] SIMVASTATIN_ACID zero-coordinate repair completed.
  echo.
  echo Do NOT rerun the old 01c script with --force, because it will recreate
  echo the bad zero-coordinate MOL2 before this rescue runs.
  echo.
  echo If all four PDBQT files are present, continue with:
  echo   13_build_formal_tasks_from_bridge.bat
  goto :done
)

echo [FAIL] Repair failed with exit code %EXITCODE%.
if exist "%LOG%" (
  echo.
  echo ---------------- LOG TAIL ----------------
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Get-Content -LiteralPath '%CD%\%LOG%' -Tail 160"
  echo ---------------- END LOG TAIL ----------------
)
goto :done

:fail
set "EXITCODE=2"

:done
echo.
pause
exit /b %EXITCODE%
