@echo off
chcp 65001 >nul
cd /d "%~dp0"
call 16_summarize_and_validate_redocking.bat
if errorlevel 1 exit /b 1
call 17_extract_representative_contacts.bat
if errorlevel 1 exit /b 1
call 18_grade_and_merge_bridge_evidence.bat
if errorlevel 1 exit /b 1
call 19_sync_formal_poses_for_Bstyle.bat
pause
