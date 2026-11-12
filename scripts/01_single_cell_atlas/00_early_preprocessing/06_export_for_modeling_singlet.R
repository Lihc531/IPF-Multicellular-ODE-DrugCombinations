# 06_export_for_modeling_singlet.R
# Export singlet-clean object results

library(Seurat)
library(dplyr)
library(Matrix)

project_dir <- getwd()

seu <- readRDS(file.path(project_dir, "ipf_annotated_singlet_seurat.rds"))

dir.create(file.path(project_dir, "metadata"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "tables"), recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 1. Metadata
# -----------------------------
write.csv(
  seu@meta.data,
  file.path(project_dir, "metadata", "cell_metadata_clean_singlet.csv")
)

# -----------------------------
# 2. Embeddings
# -----------------------------
umap <- Embeddings(seu, "umap")

write.csv(
  umap,
  file.path(project_dir, "results", "tables", "singlet_umap_embedding.csv")
)

if ("harmony" %in% Reductions(seu)) {
  harmony_emb <- Embeddings(seu, "harmony")
  
  write.csv(
    harmony_emb,
    file.path(project_dir, "results", "tables", "singlet_harmony_embedding.csv")
  )
}

# -----------------------------
# 3. Cell fraction by sample and cell type
# -----------------------------
if (all(c("sample_id", "celltype_marker_based") %in% colnames(seu@meta.data))) {
  frac <- seu@meta.data %>%
    count(sample_id, Disease_Identity, celltype_marker_based, name = "n_cells") %>%
    group_by(sample_id) %>%
    mutate(fraction = n_cells / sum(n_cells)) %>%
    ungroup()
  
  write.csv(
    frac,
    file.path(project_dir, "results", "tables", "singlet_cell_fraction_by_sample.csv"),
    row.names = FALSE
  )
}

# -----------------------------
# 4. Average expression by cell type
# -----------------------------
if (all(c("sample_id", "celltype_marker_based") %in% colnames(seu@meta.data))) {
  frac <- seu@meta.data %>%
    dplyr::count(sample_id, Disease_Identity, celltype_marker_based, name = "n_cells") %>%
    dplyr::group_by(sample_id) %>%
    dplyr::mutate(fraction = n_cells / sum(n_cells)) %>%
    dplyr::ungroup()
  
  write.csv(
    frac,
    file.path(project_dir, "results", "tables", "singlet_cell_fraction_by_sample.csv"),
    row.names = FALSE
  )
}

# -----------------------------
# 5. Low-memory average expression by cell type
#    Do NOT JoinLayers globally
# -----------------------------
DefaultAssay(seu) <- "RNA"

celltype_col <- "celltype_marker_based"

if (!celltype_col %in% colnames(seu@meta.data)) {
  stop("celltype_marker_based column not found in metadata.")
}

# 使用 marker genes 或 variable genes，避免对所有 4 万多个基因求均值
features_use <- VariableFeatures(seu)

if (length(features_use) == 0) {
  features_use <- rownames(seu)
}

features_use <- head(features_use, 3000)

# 检查 RNA assay 里的 data layers
rna_layers <- Layers(seu[["RNA"]])
data_layers <- grep("^data", rna_layers, value = TRUE)

if (length(data_layers) == 0) {
  stop("No data layers found. Please run NormalizeData first.")
}

avg_list <- list()

for (ly in data_layers) {
  message("Processing layer: ", ly)
  
  mat <- LayerData(
    object = seu,
    assay = "RNA",
    layer = ly
  )
  
  features_now <- intersect(features_use, rownames(mat))
  mat <- mat[features_now, , drop = FALSE]
  
  cells_now <- colnames(mat)
  meta_now <- seu@meta.data[cells_now, , drop = FALSE]
  
  groups <- as.character(meta_now[[celltype_col]])
  groups[is.na(groups)] <- "Unknown"
  
  group_levels <- sort(unique(groups))
  
  avg_layer <- sapply(group_levels, function(g) {
    Matrix::rowMeans(mat[, groups == g, drop = FALSE])
  })
  
  if (is.vector(avg_layer)) {
    avg_layer <- matrix(
      avg_layer,
      ncol = 1,
      dimnames = list(features_now, group_levels)
    )
  }
  
  avg_list[[ly]] <- avg_layer
  
  rm(mat, avg_layer)
  gc()
}

# -----------------------------
# 6. Combine layer-level average expression
# -----------------------------
all_genes <- Reduce(union, lapply(avg_list, rownames))
all_groups <- Reduce(union, lapply(avg_list, colnames))

sum_mat <- matrix(
  0,
  nrow = length(all_genes),
  ncol = length(all_groups),
  dimnames = list(all_genes, all_groups)
)

count_mat <- matrix(
  0,
  nrow = length(all_genes),
  ncol = length(all_groups),
  dimnames = list(all_genes, all_groups)
)

for (ly in names(avg_list)) {
  avg_layer <- avg_list[[ly]]
  
  genes <- rownames(avg_layer)
  groups <- colnames(avg_layer)
  
  # 当前 layer 每个 celltype 的细胞数
  mat_cells <- LayerData(
    object = seu,
    assay = "RNA",
    layer = ly
  )
  
  cells_now <- colnames(mat_cells)
  meta_now <- seu@meta.data[cells_now, , drop = FALSE]
  group_now <- as.character(meta_now[[celltype_col]])
  group_now[is.na(group_now)] <- "Unknown"
  
  n_by_group <- table(group_now)
  
  for (g in groups) {
    n_g <- as.numeric(n_by_group[g])
    
    sum_mat[genes, g] <- sum_mat[genes, g] + avg_layer[genes, g] * n_g
    count_mat[genes, g] <- count_mat[genes, g] + n_g
  }
  
  rm(mat_cells)
  gc()
}

avg_final <- sum_mat / count_mat
avg_final[is.nan(avg_final)] <- NA

write.csv(
  avg_final,
  file.path(project_dir, "results", "tables", "singlet_average_expression_by_celltype_low_memory.csv")
)

message("Low-memory average expression export finished.")

# -----------------------------
# 7. Skip full normalized matrix export
# -----------------------------
message("Skipping full normalized expression matrix export to avoid memory overflow.")

seu <- readRDS("ipf_annotated_singlet_seurat.rds")

dim(seu)
table(seu$sample_id)
table(seu$Disease_Identity)
table(seu$doublet_class)
table(seu$celltype_marker_based, useNA = "ifany")
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$CellType_Category)),
  "results/tables/singlet_cluster_vs_CellType_Category.csv"
)

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$celltype_marker_based)),
  "results/tables/singlet_cluster_vs_marker_based_celltype.csv"
)

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$Disease_Identity)),
  "results/tables/singlet_cluster_vs_Disease_Identity.csv"
)

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$sample_id)),
  "results/tables/singlet_cluster_vs_sample_id.csv"
)

cluster_qc <- seu@meta.data %>%
  group_by(cluster_default) %>%
  summarise(
    n_cells = n(),
    pct_original_multiplet = ifelse(
      "CellType_Category" %in% colnames(seu@meta.data),
      mean(CellType_Category == "Multiplet") * 100,
      NA
    ),
    median_nFeature_RNA = median(nFeature_RNA),
    median_nCount_RNA = median(nCount_RNA),
    median_percent_mt = median(percent.mt),
    .groups = "drop"
  )

write.csv(
  cluster_qc,
  "results/tables/singlet_cluster_qc_summary.csv",
  row.names = FALSE
)

cluster_qc
