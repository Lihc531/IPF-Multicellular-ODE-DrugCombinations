@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

:MENU
cls
echo =====================================================================
echo AutoDock Vina 原流程主干 + 当前课题正式多种子/桥接证据融合版
echo =====================================================================
echo.
echo ---------- 原流程：准备、预跑、排错与制图 ----------
echo [0] 初始化目录和模板
echo [1] 检查环境
echo [2] 预处理 receptor / ligand PDBQT
echo [3] 提取共晶配体中心
echo [4] 按残基计算位点中心
echo [5] 原单次流程 dry run（仅预检查）
echo [6] 原单次 Vina 对接（仅预跑/排错，不作为最终证据）
echo.
echo ---------- 当前课题：桥接候选到正式 Vina ----------
echo [P] 准备正式配体与原配体重对接文件
echo [D] 从37e/38b候选构建 drug-target-PDB 正式任务
echo [E] 正式输入审计
echo [F] 五种子正式命令 dry run
echo [G] 五种子正式运行（断点续跑）
echo [H] 多种子汇总 + 原配体重对接验证
echo [I] 代表位姿接触审计
echo [J] Vina分级并回并药物组合证据
echo [K] 同步Grade A/B代表位姿到原B-style制图目录
echo.
echo ---------- 原B-style制图 ----------
echo [7] 生成 PyMOL 制图脚本
echo [8] 打开人工选角度菜单
echo [9] 合成最终 B-style 图片
echo [A] 全自动 PyMOL 导出并合成
echo [B] 导出并绘制 affinity 排序图
echo [C] 汇总图片到 results\final_figures
echo [Q] 退出
echo =====================================================================
set /p choice=请输入选项：

if /I "%choice%"=="0" call 00_initialize_project.bat
if /I "%choice%"=="1" call 00_check_environment.bat
if /I "%choice%"=="2" call 01_prepare_pdbqt.bat
if /I "%choice%"=="3" call 02_extract_cocrystal_centers.bat
if /I "%choice%"=="4" call 03_calculate_CDC25A_center.bat
if /I "%choice%"=="5" call 04a_dry_run_check.bat
if /I "%choice%"=="6" call 04_run_vina_docking.bat
if /I "%choice%"=="P" call 01c_prepare_formal_ligands.bat
if /I "%choice%"=="D" call 13_build_formal_tasks_from_bridge.bat
if /I "%choice%"=="E" call 14_audit_formal_inputs.bat
if /I "%choice%"=="F" call 15a_formal_multiseed_dry_run.bat
if /I "%choice%"=="G" call 15_run_formal_multiseed.bat
if /I "%choice%"=="H" call 16_summarize_and_validate_redocking.bat
if /I "%choice%"=="I" call 17_extract_representative_contacts.bat
if /I "%choice%"=="J" call 18_grade_and_merge_bridge_evidence.bat
if /I "%choice%"=="K" call 19_sync_formal_poses_for_Bstyle.bat
if /I "%choice%"=="7" call 05_generate_pymol_scripts.bat
if /I "%choice%"=="8" call 11_open_manual_pymol_menu.bat
if /I "%choice%"=="9" call 12_compose_bstyle_figures.bat
if /I "%choice%"=="A" call 09_auto_export_and_compose.bat
if /I "%choice%"=="B" call 07_export_plot_data.bat & call 08_plot_vina_scores.bat
if /I "%choice%"=="C" call 06_organize_single_images.bat
if /I "%choice%"=="Q" goto END
goto MENU

:END
endlocal
exit /b 0
