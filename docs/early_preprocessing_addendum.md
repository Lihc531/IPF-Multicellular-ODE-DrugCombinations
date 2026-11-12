# GSE136831早期预处理补充归档说明

本次仅把用于Methods v0.5补证的早期脚本加入现有四模块归档，不修改脚本内容。

### 正式主线

`ipf_raw_seurat.rds` → `03b_doublet_detection.R` → `ipf_qc_singlet_seurat.rds` → `03c_normalize_after_doublet.R` → `ipf_qc_singlet_normalized_seurat.rds` → `04_integration_clustering_singlet.R` → `ipf_integrated_singlet_seurat.rds` → `05_celltype_annotation_singlet.R` → `ipf_annotated_singlet_seurat.rds` → `07_remove_multiplet_rerun_main_pipeline.R` → `ipf_nomultiplet_light_metadata_embeddings.rds`。

`06_export_for_modeling_singlet.R`用于singlet对象的metadata、embedding、组成及后续建模输入导出。`04b_find_markers_singlet_low_memory.R`用于低内存marker补跑。

### 仍需注意

1. `02_build_seurat_object.R`虽然写出 `ipf_raw_seurat.rds`，但脚本明确写明其假定Matrix Market格式并可能需要调整文件pattern，因此不将其描述为完全冻结的GSE136831 RAW解析实现。
2. `03b`中没有人工doublet score cutoff；采用scDblFinder自动分类结果。细胞数<500的sample/library被直接标记为singlet。
3. 本次早期包没有 `results/qc/doublet_summary_by_sample.csv` 实际结果，因此不能从脚本归档中反推逐样本doublet比例。
4. SCT和pre-scDblFinder旧支线未纳入正式主线。
