# 07_export_nomultiplet_without_subset.R
# Low-memory removal of residual Multiplet without Seurat subset()
# This script does NOT create a new full Seurat object.
# It exports filtered metadata, embeddings, cell fractions, cross tables, and UMAP plots.

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(ggplot2)
})

project_dir <- getwd()

dir.create(file.path(project_dir, "metadata"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "figures"), recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 1. Read annotated object
# -----------------------------
seu <- readRDS(file.path(project_dir, "ipf_annotated_singlet_seurat.rds"))

message("Original object:")
print(dim(seu))

meta <- seu@meta.data

if (!"CellType_Category" %in% colnames(meta)) {
  stop("CellType_Category column not found.")
}

message("Original CellType_Category:")
print(table(meta$CellType_Category, useNA = "ifany"))

# -----------------------------
# 2. Define cells to keep
# -----------------------------
cells_keep <- rownames(meta)[
  is.na(meta$CellType_Category) | meta$CellType_Category != "Multiplet"
]

cells_remove <- rownames(meta)[
  !is.na(meta$CellType_Category) & meta$CellType_Category == "Multiplet"
]

message("Cells kept: ", length(cells_keep))
message("Cells removed as Multiplet: ", length(cells_remove))

writeLines(
  cells_keep,
  con = file.path(project_dir, "metadata", "cells_keep_nomultiplet.txt")
)

writeLines(
  cells_remove,
  con = file.path(project_dir, "metadata", "cells_removed_multiplet.txt")
)

# -----------------------------
# 3. Filter metadata only
# -----------------------------
meta_nomultiplet <- meta[cells_keep, , drop = FALSE]

message("Filtered CellType_Category:")
print(table(meta_nomultiplet$CellType_Category, useNA = "ifany"))

write.csv(
  meta_nomultiplet,
  file.path(project_dir, "metadata", "cell_metadata_annotated_singlet_nomultiplet.csv")
)

# -----------------------------
# 4. Export UMAP embedding without subsetting Seurat object
# -----------------------------
if ("umap" %in% Reductions(seu)) {
  umap <- Embeddings(seu, "umap")
  umap_nomultiplet <- umap[cells_keep, , drop = FALSE]
  
  write.csv(
    umap_nomultiplet,
    file.path(project_dir, "results", "tables", "nomultiplet_umap_embedding.csv")
  )
} else {
  warning("UMAP reduction not found.")
  umap_nomultiplet <- NULL
}

# -----------------------------
# 5. Export Harmony embedding without subsetting Seurat object
# -----------------------------
if ("harmony" %in% Reductions(seu)) {
  harmony_emb <- Embeddings(seu, "harmony")
  harmony_nomultiplet <- harmony_emb[cells_keep, , drop = FALSE]
  
  write.csv(
    harmony_nomultiplet,
    file.path(project_dir, "results", "tables", "nomultiplet_harmony_embedding.csv")
  )
  
  rm(harmony_emb, harmony_nomultiplet)
  gc()
}

# After embeddings are extracted, remove full Seurat object from memory
rm(seu)
gc()

# -----------------------------
# 6. Export cell fraction by sample
# -----------------------------
if (all(c("sample_id", "Disease_Identity", "celltype_marker_based") %in% colnames(meta_nomultiplet))) {
  frac <- meta_nomultiplet %>%
    dplyr::count(sample_id, Disease_Identity, celltype_marker_based, name = "n_cells") %>%
    dplyr::group_by(sample_id) %>%
    dplyr::mutate(fraction = n_cells / sum(n_cells)) %>%
    dplyr::ungroup()
  
  write.csv(
    frac,
    file.path(project_dir, "results", "tables", "nomultiplet_cell_fraction_by_sample.csv"),
    row.names = FALSE
  )
}

# -----------------------------
# 7. Export disease-level cell fraction summary
# -----------------------------
if (all(c("Disease_Identity", "celltype_marker_based") %in% colnames(meta_nomultiplet))) {
  frac_disease <- meta_nomultiplet %>%
    dplyr::count(Disease_Identity, celltype_marker_based, name = "n_cells") %>%
    dplyr::group_by(Disease_Identity) %>%
    dplyr::mutate(fraction = n_cells / sum(n_cells)) %>%
    dplyr::ungroup()
  
  write.csv(
    frac_disease,
    file.path(project_dir, "results", "tables", "nomultiplet_cell_fraction_by_disease.csv"),
    row.names = FALSE
  )
}

# -----------------------------
# 8. Export cross tables
# -----------------------------
if (all(c("cluster_default", "CellType_Category") %in% colnames(meta_nomultiplet))) {
  write.csv(
    as.data.frame.matrix(table(meta_nomultiplet$cluster_default, meta_nomultiplet$CellType_Category)),
    file.path(project_dir, "results", "tables", "nomultiplet_cluster_vs_CellType_Category.csv")
  )
}

if (all(c("cluster_default", "celltype_marker_based") %in% colnames(meta_nomultiplet))) {
  write.csv(
    as.data.frame.matrix(table(meta_nomultiplet$cluster_default, meta_nomultiplet$celltype_marker_based)),
    file.path(project_dir, "results", "tables", "nomultiplet_cluster_vs_marker_based_celltype.csv")
  )
}

if (all(c("cluster_default", "Disease_Identity") %in% colnames(meta_nomultiplet))) {
  write.csv(
    as.data.frame.matrix(table(meta_nomultiplet$cluster_default, meta_nomultiplet$Disease_Identity)),
    file.path(project_dir, "results", "tables", "nomultiplet_cluster_vs_Disease_Identity.csv")
  )
}

if (all(c("cluster_default", "sample_id") %in% colnames(meta_nomultiplet))) {
  write.csv(
    as.data.frame.matrix(table(meta_nomultiplet$cluster_default, meta_nomultiplet$sample_id)),
    file.path(project_dir, "results", "tables", "nomultiplet_cluster_vs_sample_id.csv")
  )
}

if (all(c("celltype_marker_based", "SingleR_label_cluster_level") %in% colnames(meta_nomultiplet))) {
  write.csv(
    as.data.frame.matrix(table(meta_nomultiplet$celltype_marker_based, meta_nomultiplet$SingleR_label_cluster_level)),
    file.path(project_dir, "results", "tables", "nomultiplet_marker_based_vs_SingleR_label.csv")
  )
}

# -----------------------------
# 9. Cluster QC summary
# -----------------------------
if ("cluster_default" %in% colnames(meta_nomultiplet)) {
  cluster_qc <- meta_nomultiplet %>%
    dplyr::group_by(cluster_default) %>%
    dplyr::summarise(
      n_cells = dplyr::n(),
      pct_original_multiplet = mean(CellType_Category == "Multiplet", na.rm = TRUE) * 100,
      median_nFeature_RNA = if ("nFeature_RNA" %in% colnames(meta_nomultiplet)) median(nFeature_RNA, na.rm = TRUE) else NA,
      median_nCount_RNA = if ("nCount_RNA" %in% colnames(meta_nomultiplet)) median(nCount_RNA, na.rm = TRUE) else NA,
      median_percent_mt = if ("percent.mt" %in% colnames(meta_nomultiplet)) median(percent.mt, na.rm = TRUE) else NA,
      .groups = "drop"
    )
  
  write.csv(
    cluster_qc,
    file.path(project_dir, "results", "tables", "nomultiplet_cluster_qc_summary.csv"),
    row.names = FALSE
  )
}

# -----------------------------
# 10. Plot UMAPs using ggplot, not Seurat DimPlot
# -----------------------------
if (!is.null(umap_nomultiplet)) {
  plot_df <- as.data.frame(umap_nomultiplet)
  plot_df$cell_id <- rownames(plot_df)
  
  # Standardize UMAP column names
  colnames(plot_df)[1:2] <- c("UMAP_1", "UMAP_2")
  
  plot_df <- cbind(
    plot_df,
    meta_nomultiplet[plot_df$cell_id, , drop = FALSE]
  )
  
  if ("cluster_default" %in% colnames(plot_df)) {
    p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = as.factor(cluster_default))) +
      geom_point(size = 0.05, alpha = 0.6) +
      theme_classic() +
      labs(color = "cluster_default")
    
    ggsave(
      file.path(project_dir, "results", "figures", "nomultiplet_umap_clusters_existing_embedding_ggplot.pdf"),
      p,
      width = 8,
      height = 6,
      limitsize = FALSE
    )
  }
  
  if ("celltype_marker_based" %in% colnames(plot_df)) {
    p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = celltype_marker_based)) +
      geom_point(size = 0.05, alpha = 0.6) +
      theme_classic() +
      labs(color = "celltype_marker_based")
    
    ggsave(
      file.path(project_dir, "results", "figures", "nomultiplet_umap_celltype_marker_based_existing_embedding_ggplot.pdf"),
      p,
      width = 10,
      height = 8,
      limitsize = FALSE
    )
  }
  
  if ("SingleR_label_cluster_level" %in% colnames(plot_df)) {
    p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = SingleR_label_cluster_level)) +
      geom_point(size = 0.05, alpha = 0.6) +
      theme_classic() +
      labs(color = "SingleR_label_cluster_level")
    
    ggsave(
      file.path(project_dir, "results", "figures", "nomultiplet_umap_singleR_label_cluster_level_existing_embedding_ggplot.pdf"),
      p,
      width = 10,
      height = 8,
      limitsize = FALSE
    )
  }
  
  if ("Disease_Identity" %in% colnames(plot_df)) {
    p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = Disease_Identity)) +
      geom_point(size = 0.05, alpha = 0.6) +
      theme_classic() +
      labs(color = "Disease_Identity")
    
    ggsave(
      file.path(project_dir, "results", "figures", "nomultiplet_umap_disease_identity_existing_embedding_ggplot.pdf"),
      p,
      width = 8,
      height = 6,
      limitsize = FALSE
    )
  }
  
  if ("sample_id" %in% colnames(plot_df)) {
    p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = sample_id)) +
      geom_point(size = 0.05, alpha = 0.5) +
      theme_classic() +
      labs(color = "sample_id")
    
    ggsave(
      file.path(project_dir, "results", "figures", "nomultiplet_umap_sample_id_existing_embedding_ggplot.pdf"),
      p,
      width = 12,
      height = 8,
      limitsize = FALSE
    )
  }
}

# -----------------------------
# 11. Save lightweight analysis object
# -----------------------------
nomultiplet_light <- list(
  metadata = meta_nomultiplet,
  umap = umap_nomultiplet,
  cells_keep = cells_keep,
  cells_removed_multiplet = cells_remove
)

saveRDS(
  nomultiplet_light,
  file.path(project_dir, "ipf_nomultiplet_light_metadata_embeddings.rds")
)

message("Done.")
message("No Seurat subset was performed.")
message("Saved:")
message("  metadata/cell_metadata_annotated_singlet_nomultiplet.csv")
message("  metadata/cells_keep_nomultiplet.txt")
message("  metadata/cells_removed_multiplet.txt")
message("  results/tables/nomultiplet_cell_fraction_by_sample.csv")
message("  results/tables/nomultiplet_umap_embedding.csv")
message("  ipf_nomultiplet_light_metadata_embeddings.rds")

message("Final expected cell number:")
message(length(cells_keep))