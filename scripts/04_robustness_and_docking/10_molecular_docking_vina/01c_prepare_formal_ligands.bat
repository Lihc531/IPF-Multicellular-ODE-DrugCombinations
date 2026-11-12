@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "LOG=logs\formal_ligand_prep\SIMVASTATIN_ACID_mgltools.log"
set "SCRIPT=scripts\prepare_formal_ligands.py"

echo ============================================================
echo Prepare formal ligands - SIMVASTATIN force/diagnostic mode
echo ============================================================
echo.
echo Project directory:
echo   %CD%
echo.

if not exist "%SCRIPT%" (
  echo [FATAL] Missing %SCRIPT%
  echo Extract the complete fixed package into this project directory.
  goto :fail2
)

if not exist "config.json" (
  echo [FATAL] Missing config.json in:
  echo   %CD%
  goto :fail2
)

if not exist "FIX_SIMVASTATIN_MGLTOOLS.md" (
  echo [WARN] FIX_SIMVASTATIN_MGLTOOLS.md was not found.
  echo [WARN] This may still be the old project or an incomplete overwrite.
  echo.
)

where python >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Python was not found in the current terminal.
  echo Activate the intended environment first, then rerun this BAT file.
  goto :fail2
)

echo Python executable:
where python
echo.
python --version
echo.

echo Running the fixed preparation script with --force ...
echo.
python "%SCRIPT%" --project . --config config.json --force
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
  echo [OK] Formal ligand preparation completed.
  echo.
  echo Expected files:
  echo   prepared\ligands\SELUMETINIB_pH7p4.pdbqt
  echo   prepared\ligands\MAP2K1_4U7Z_3EW_native.pdbqt
  echo   prepared\ligands\SIMVASTATIN_ACID_pH7p4.pdbqt
  echo   prepared\ligands\HMGCR_1HW9_SIM_native.pdbqt
  echo.
  echo Next run: 13_build_formal_tasks_from_bridge.bat
  goto :done
)

echo [FAIL] Formal ligand preparation failed with exit code %EXITCODE%.
echo.

if exist "%LOG%" (
  echo ---------------- LOG TAIL: %LOG% ----------------
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Get-Content -LiteralPath '%CD%\%LOG%' -Tail 120"
  echo ---------------- END LOG TAIL ----------------
) else (
  echo [WARN] Expected log was not found:
  echo   %CD%\%LOG%
)

echo.
echo Interpretation:
echo   - If Python reports "unrecognized arguments: --force", the old
  echo     scripts\prepare_formal_ligands.py is still installed.
echo   - If the log mentions MolKit, bond order, atom type, or MOL2 parsing,
echo     confirm that the fixed script replaced the old script and that old
  echo     SIMVASTATIN intermediate files were rebuilt.
echo   - If the log says a file/path is missing, correct mgltools_dir and
  echo     obabel_exe in config.json.
echo.
goto :done

:fail2
set "EXITCODE=2"

:done
echo.
pause
exit /b %EXITCODE%
