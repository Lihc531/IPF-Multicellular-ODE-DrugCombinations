@echo off
chcp 65001 >nul
cd /d "%~dp0"
call bridge_integration\build_tasks_example.bat
