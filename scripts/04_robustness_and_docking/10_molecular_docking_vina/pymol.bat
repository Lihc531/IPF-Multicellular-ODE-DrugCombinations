@echo off
setlocal

set "PYMOL_ENV=E:\pymol"
set "PATH=%PYMOL_ENV%;%PYMOL_ENV%\Library\bin;%PYMOL_ENV%\Scripts;%PATH%"

start "" "%PYMOL_ENV%\Scripts\pymol.exe"

endlocal