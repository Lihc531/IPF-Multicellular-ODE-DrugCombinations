@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 示例残基输入格式：A:430,A:431,A:436
echo 不指定原子名时直接回车即可。
echo.
python vina_workflow.py --project . --config config.json calculate-center --padding 6 --min-size 20
pause
