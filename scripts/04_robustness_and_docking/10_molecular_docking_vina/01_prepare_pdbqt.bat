@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
title Prepare standard non-covalent ligand PDBQT

rem Resolve the project directory from this BAT file location.
for %%I in ("%~dp0.") do set "PROJECT=%%~fI"

set "INPUT_DIR=!PROJECT!\inputs\ligands"
set "OUTPUT_DIR=!PROJECT!\prepared\ligands"
set "LOG_DIR=!PROJECT!\logs"

set "OBABEL=E:\Anaconda3\Anaconda\envs\obabel_env\Library\bin\obabel.exe"
set "MGLPY=E:\MGLTools-1.5.7\python.exe"
set "PREPARE_LIGAND=E:\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_ligand4.py"

if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"
if not exist "!LOG_DIR!" mkdir "!LOG_DIR!"

rem Stop immediately when a required executable/script is missing.
if not exist "!OBABEL!" goto :MISSING_OBABEL
if not exist "!MGLPY!" goto :MISSING_MGLPY
if not exist "!PREPARE_LIGAND!" goto :MISSING_PREPARE_LIGAND

echo ============================================================
echo Prepare standard non-covalent ligands
echo ============================================================
echo Project: !PROJECT!
echo.

set /a FAILURES=0

call :PREPARE_ONE SELUMETINIB
if errorlevel 1 set /a FAILURES+=1

echo ------------------------------------------------------------
echo SIMVASTATIN
echo ------------------------------------------------------------
echo [SKIP] The supplied SIMVASTATIN.sdf is the closed lactone form.
echo        Formal HMGCR docking requires open SIMVASTATIN ACID.
echo        Run 01c_prepare_formal_ligands.bat instead.

echo.
echo ============================================================
echo BORTEZOMIB
echo ============================================================
echo [SKIP] BORTEZOMIB contains boron and requires a special
echo        covalent or boron-compatible docking workflow.
echo        It is not included in standard Vina preparation.

echo.
echo ============================================================
echo Preparation finished
echo ============================================================
echo Output directory:
echo !OUTPUT_DIR!
echo.

if !FAILURES! GTR 0 (
    echo [FAIL] !FAILURES! standard ligand^(s^) failed.
    echo Please review the corresponding files under:
    echo !LOG_DIR!
    echo.
    pause
    exit /b 1
)

echo [OK] All standard non-covalent ligands are ready.
echo.
pause
exit /b 0


:PREPARE_ONE
setlocal EnableDelayedExpansion
set "NAME=%~1"
set "SDF=!INPUT_DIR!\!NAME!.sdf"
set "MOL2=!OUTPUT_DIR!\!NAME!_for_mgltools.mol2"
set "PDBQT=!OUTPUT_DIR!\!NAME!.pdbqt"
set "OBABEL_LOG=!LOG_DIR!\prepare_ligand_!NAME!_obabel.log"
set "MGL_LOG=!LOG_DIR!\prepare_ligand_!NAME!_mgltools.log"

echo ------------------------------------------------------------
echo Preparing !NAME!
echo ------------------------------------------------------------

rem Remove zero-byte remnants so they are not mistaken for valid outputs.
if exist "!PDBQT!" (
    for %%F in ("!PDBQT!") do if %%~zF LEQ 0 del /q "!PDBQT!" >nul 2>&1
)
if exist "!MOL2!" (
    for %%F in ("!MOL2!") do if %%~zF LEQ 0 del /q "!MOL2!" >nul 2>&1
)

if exist "!PDBQT!" (
    echo [OK] !NAME!.pdbqt already exists, skipping.
    endlocal & exit /b 0
)

if exist "!MOL2!" (
    echo [OK] Existing MOL2 found:
    echo      !MOL2!
) else (
    if not exist "!SDF!" (
        echo [FAIL] Source SDF not found:
        echo        !SDF!
        endlocal & exit /b 1
    )

    echo Converting SDF to MOL2 with Open Babel...
    "!OBABEL!" -isdf "!SDF!" -omol2 -O "!MOL2!" -h > "!OBABEL_LOG!" 2>&1
    set "OBABEL_RC=!ERRORLEVEL!"
    type "!OBABEL_LOG!"

    if not "!OBABEL_RC!"=="0" (
        echo [FAIL] Open Babel failed for !NAME! with exit code !OBABEL_RC!.
        echo        Log: !OBABEL_LOG!
        if exist "!MOL2!" del /q "!MOL2!" >nul 2>&1
        endlocal & exit /b 1
    )

    if not exist "!MOL2!" (
        echo [FAIL] Open Babel returned success but MOL2 was not generated:
        echo        !MOL2!
        echo        Log: !OBABEL_LOG!
        endlocal & exit /b 1
    )

    echo [OK] MOL2 generated:
    echo      !MOL2!
)

echo Converting MOL2 to PDBQT with MGLTools...

rem MGLTools/MolKit may discard the directory part of -l on Windows.
rem Run it from OUTPUT_DIR and deliberately pass only the file names.
pushd "!OUTPUT_DIR!" >nul
"!MGLPY!" "!PREPARE_LIGAND!" -l "!NAME!_for_mgltools.mol2" -o "!NAME!.pdbqt" -A hydrogens > "!MGL_LOG!" 2>&1
set "MGL_RC=!ERRORLEVEL!"
popd >nul

type "!MGL_LOG!"

if not "!MGL_RC!"=="0" (
    echo [FAIL] MGLTools failed for !NAME! with exit code !MGL_RC!.
    echo        Log: !MGL_LOG!
    if exist "!PDBQT!" del /q "!PDBQT!" >nul 2>&1
    endlocal & exit /b 1
)

if not exist "!PDBQT!" (
    echo [FAIL] MGLTools returned success but PDBQT was not generated:
    echo        !PDBQT!
    echo        Log: !MGL_LOG!
    endlocal & exit /b 1
)

for %%F in ("!PDBQT!") do set "PDBQT_SIZE=%%~zF"
if !PDBQT_SIZE! LEQ 0 (
    echo [FAIL] Generated PDBQT is empty:
    echo        !PDBQT!
    del /q "!PDBQT!" >nul 2>&1
    endlocal & exit /b 1
)

echo [OK] Generated:
echo      !PDBQT!
echo      Size: !PDBQT_SIZE! bytes
echo.
endlocal & exit /b 0


:MISSING_OBABEL
echo [FATAL] Open Babel was not found:
echo         !OBABEL!
echo.
pause
exit /b 1

:MISSING_MGLPY
echo [FATAL] MGLTools Python was not found:
echo         !MGLPY!
echo.
pause
exit /b 1

:MISSING_PREPARE_LIGAND
echo [FATAL] prepare_ligand4.py was not found:
echo         !PREPARE_LIGAND!
echo.
pause
exit /b 1
