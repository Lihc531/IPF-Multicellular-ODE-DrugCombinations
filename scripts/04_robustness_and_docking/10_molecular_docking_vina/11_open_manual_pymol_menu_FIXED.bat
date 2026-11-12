@echo off
setlocal

rem Resolve the project directory without a trailing backslash.
rem Passing "%~dp0" directly can turn the closing quote into a literal quote
rem because %~dp0 always ends with a backslash.
for %%I in ("%~dp0.") do set "PROJECT=%%~fI"
cd /d "%PROJECT%"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  pause
  exit /b 1
)

python "%PROJECT%\scripts\open_manual_pymol_fixed.py" --project "%PROJECT%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] Manual PyMOL launcher failed with exit code %RC%.
  pause
)
exit /b %RC%
