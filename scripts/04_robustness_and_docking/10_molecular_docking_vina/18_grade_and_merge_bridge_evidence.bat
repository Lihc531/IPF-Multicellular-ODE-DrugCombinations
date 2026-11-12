@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PAIR_CANDIDATES=E:\PythonProject\results\38b_PAIR_VS_SINGLE\pair_proxy_signature_increment_audit.csv"
python formal_vina_pipeline.py --project . --config config.json grade --multiseed-summary results/formal/03_summary/vina_multiseed_summary.csv --input-audit results/formal/01_audit/docking_input_audit.csv --docking-manifest results/formal/01_audit/formal_docking_manifest_audited.csv --target-validation results/formal/03_summary/target_redocking_validation.csv --interaction-summary results/formal/04_interactions/vina_interaction_fingerprint_summary.csv --pair-candidates "%PAIR_CANDIDATES%" --output-dir results/formal/05_final
pause
