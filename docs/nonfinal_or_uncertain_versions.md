# 未纳入或需注意的非最终/重复版本

- `r/15b_celltype_dynamic_gene_modules_axis_v2.R`：与 FIXED2 同样包含 true dynamic_gene_csv 输出，但未选择；FIXED2 文件名更明确为修正版。
- `r/15_celltype_dynamic_gene_modules_axis_v2.R`：旧动态模块脚本，不生成 upload.zip 中的 12/13 true ODE-matched dynamic_gene_csv。
- `r/18_build_cell_state_feature_matrix_axis_v2.R`：保留 18b 作为更完整版本，18 未纳入。
- `py/B! 21a_v1B_celltype_time_aware_direction_constrained_neural_ode.py`：与 21a2 哈希完全一致；为避免重复，仅纳入 21a2。
- `py/B! 21b_v1B_celltype_cluster_time_aware_direction_constrained_neural_ode.py`：早于正式 v1B.1/21a2 的主模型，未纳入最终版。
- `py/19_train_neural_ode_cell_state_matrix_v1A.py`：Neural ODE 原型/v1A，未纳入最终版。
- `py/20_refine_neural_ode_with_directionality.py`：旧 refined ODE，未纳入最终版。
- `py/A！20a_Refine_Neural_Ode_With_Directionality.py`：旧 refined ODE 备份/阶段版本，未纳入最终版。
- `py/A！20b_time_aware_direction_constrained_neural_ode.py`：早期 time-aware ODE，未纳入最终版。
- `py/23b_PPI-informed driver refinement.py`：旧 PPI refinement；upload.zip 的最终 PPI 输出由 34a v2 匹配。
- `py/24_generate_final_driver_shortlists.py`：旧 driver shortlist 生成；最终 PPI/OT 交付由 34a/35b 匹配。
- `py/29_ot_disease_reversal_score.py`：旧 OT disease-reversal 版本；最终 06_OT 输出由 35a/35b 匹配。
- `py/29_OT_COMBINATION_BASELINE_AND_FLOW_MATCHING_FINAL.py`：旧编号的 Flow/baseline 脚本；最终打包采用 39/40/41。
- `py/30_final_ot_candidate_screening_v1.py`：旧候选筛选支线；最终正式候选以 Module 08 pair_vs_best_single 和 41.1 补充输出为准。
- `py/32_run_patched_robustness_scripts_v1.py`：robustness runner 旧版；采用 v3。
- `vina/11_open_manual_pymol_menu.bat`：旧 launcher；采用 11_open_manual_pymol_menu_FIXED.bat。

## 项目早期预处理脚本的版本边界（本次新增）

正式主队列上游主线采用：`03b_doublet_detection.R` → `03c_normalize_after_doublet.R` → `04_integration_clustering_singlet.R`（必要时以 `04b` 低内存marker脚本补充）→ `05_celltype_annotation_singlet.R` → `06_export_for_modeling_singlet.R` → `07_remove_multiplet_rerun_main_pipeline.R`。

以下早期文件**不纳入正式主线**：

- `03_qc_filter_normalize.R`：明确写有“Skip doublet detection for now”，将所有细胞临时设为singlet，已被 `03b` + `03c` 取代。
- `03d_sctransform_after_doublet.R`、`04_integration_clustering_sct.R`、`补充检查.R`：SCTransform可选支线；正式稿方法采用LogNormalize + Harmony，不用SCT覆盖正式singlet分支。
- `04_integration_clustering.R`、`05_celltype_annotation.R`、`06_export_for_modeling.R`：较早的非singlet对象支线，已被对应 `_singlet` 版本取代。
- 早期 `README.md`：推荐顺序对应旧版“先跳过doublet”的流程，仅作历史参考，不作为正式运行说明。

`02_build_seurat_object.R` 被保留在“数据获取与raw_object_溯源”目录，而不是标记为完全冻结的正式RAW解析脚本：它确实写出 `ipf_raw_seurat.rds`，但代码明确假定通用Matrix Market布局，并提示实际GEO文件名可能需要手工调整。
