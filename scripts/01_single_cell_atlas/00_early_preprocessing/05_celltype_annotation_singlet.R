# 05_celltype_annotation_singlet.R
# Marker-based and reference-assisted annotation for singlet object

library(Seurat)
library(dplyr)
library(ggplot2)
library(SingleR)
library(celldex)
library(SingleCellExperiment)

project_dir <- getwd()

seu <- readRDS(file.path(project_dir, "ipf_integrated_singlet_seurat.rds"))

DefaultAssay(seu) <- "RNA"

# -----------------------------
# 1. Keep original annotations if available
# -----------------------------
if ("CellType_Category" %in% colnames(seu@meta.data)) {
  seu$celltype_major_original <- seu$CellType_Category
}

if ("Subclass_Cell_Identity" %in% colnames(seu@meta.data)) {
  seu$celltype_fine_original <- seu$Subclass_Cell_Identity
}

# -----------------------------
# 2. Marker list
# -----------------------------
marker_list <- list(
  Epithelial = c("EPCAM", "KRT8", "KRT18"),
  AT1 = c("AGER", "PDPN", "CAV1"),
  AT2 = c("SFTPC", "SFTPA1", "SFTPA2", "ABCA3"),
  Basal = c("KRT5", "KRT17", "TP63"),
  Aberrant_basaloid = c("KRT17", "KRT5", "COL1A1", "FN1", "ITGA6"),
  Fibroblast = c("COL1A1", "COL1A2", "DCN", "LUM"),
  Myofibroblast = c("ACTA2", "TAGLN", "POSTN", "COL3A1"),
  Macrophage = c("CD68", "MARCO", "C1QA", "C1QB"),
  Monocyte = c("LYZ", "S100A8", "S100A9", "FCN1"),
  Dendritic = c("FCER1A", "CLEC10A", "LILRA4", "ITGAX"),
  T_cell = c("CD3D", "CD3E", "TRAC"),
  CD8_T = c("CD8A", "CD8B", "GZMK", "NKG7"),
  CD4_T = c("CD4", "IL7R", "CCR7"),
  NK = c("NKG7", "GNLY", "PRF1", "KLRD1"),
  B_cell = c("MS4A1", "CD79A", "CD79B"),
  Plasma = c("MZB1", "JCHAIN", "XBP1"),
  Endothelial = c("PECAM1", "VWF", "KDR", "CLDN5"),
  Smooth_muscle = c("MYH11", "ACTA2", "TAGLN"),
  Mast = c("TPSAB1", "TPSB2", "KIT")
)

# -----------------------------
# 3. Module scores
# -----------------------------
for (nm in names(marker_list)) {
  genes <- intersect(marker_list[[nm]], rownames(seu))
  
  if (length(genes) >= 2) {
    seu <- AddModuleScore(
      object = seu,
      features = list(genes),
      name = paste0(nm, "_score")
    )
  }
}

score_cols <- grep("_score1$", colnames(seu@meta.data), value = TRUE)

cluster_col <- "cluster_default"
if (!cluster_col %in% colnames(seu@meta.data)) {
  cluster_col <- "seurat_clusters"
}

cluster_scores <- seu@meta.data %>%
  group_by(.data[[cluster_col]]) %>%
  summarise(across(all_of(score_cols), mean), .groups = "drop")

dir.create(file.path(project_dir, "results", "markers"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "metadata"), recursive = TRUE, showWarnings = FALSE)

write.csv(
  cluster_scores,
  file.path(project_dir, "results", "markers", "singlet_cluster_marker_module_scores.csv"),
  row.names = FALSE
)

score_mat <- as.matrix(cluster_scores[, score_cols, drop = FALSE])
best <- score_cols[max.col(score_mat, ties.method = "first")]
best <- gsub("_score1$", "", best)

cluster_to_type <- data.frame(
  cluster = cluster_scores[[cluster_col]],
  predicted_celltype_marker = best
)

write.csv(
  cluster_to_type,
  file.path(project_dir, "results", "markers", "singlet_cluster_to_celltype_marker_based.csv"),
  row.names = FALSE
)

map_vec <- setNames(
  cluster_to_type$predicted_celltype_marker,
  cluster_to_type$cluster
)

seu$celltype_marker_based <- unname(
  map_vec[as.character(seu@meta.data[[cluster_col]])]
)

# -----------------------------
# 4. SingleR
# -----------------------------

library(Seurat)
library(SingleR)
library(celldex)
library(SingleCellExperiment)
library(dplyr)

DefaultAssay(seu) <- "RNA"
Idents(seu) <- "cluster_default"

set.seed(123)

cluster_ids <- levels(Idents(seu))

cells_use <- unlist(lapply(cluster_ids, function(cl) {
  cells_cl <- WhichCells(seu, idents = cl)
  sample(cells_cl, min(300, length(cells_cl)))
}))

seu_singleR <- subset(
  seu,
  cells = cells_use
)

seu_singleR <- JoinLayers(
  object = seu_singleR,
  assay = "RNA"
)
sce <- as.SingleCellExperiment(seu_singleR)

ref <- celldex::HumanPrimaryCellAtlasData()

singleR_pred <- SingleR(
  test = sce,
  ref = ref,
  labels = ref$label.main
)


seu_singleR$SingleR_label <- singleR_pred$labels

singleR_cluster_map <- seu_singleR@meta.data %>%
  dplyr::group_by(cluster_default, SingleR_label) %>%
  dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
  dplyr::group_by(cluster_default) %>%
  dplyr::slice_max(order_by = n, n = 1, with_ties = FALSE) %>%
  dplyr::ungroup()
map_vec <- setNames(
  singleR_cluster_map$SingleR_label,
  singleR_cluster_map$cluster_default
)

seu$SingleR_label_cluster_level <- unname(
  map_vec[as.character(seu$cluster_default)]
)

write.csv(
  singleR_cluster_map,
  file.path(project_dir, "results", "markers", "singlet_cluster_to_SingleR_label.csv"),
  row.names = FALSE
)

saveRDS(
  seu,
  file.path(project_dir, "ipf_annotated_singlet_seurat.rds")
)

# -----------------------------
# 5. UMAP plots
# -----------------------------
p1 <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "celltype_marker_based",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(project_dir, "results", "figures", "singlet_umap_celltype_marker_based.pdf"),
  p1,
  width = 10,
  height = 8,
  limitsize = FALSE
)

p2 <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "SingleR_label_cluster_level",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(project_dir, "results", "figures", "singlet_umap_singleR_label_cluster_level.pdf"),
  p2,
  width = 10,
  height = 8,
  limitsize = FALSE
)

if ("celltype_fine_original" %in% colnames(seu@meta.data)) {
  p3 <- DimPlot(
    seu,
    reduction = "umap",
    group.by = "celltype_fine_original",
    label = FALSE,
    raster = TRUE
  )
  
  ggsave(
    file.path(project_dir, "results", "figures", "singlet_umap_original_subclass_nolabel.pdf"),
    p3,
    width = 14,
    height = 10,
    limitsize = FALSE
  )
}

# DotPlot
marker_genes <- unique(unlist(marker_list))
marker_genes <- intersect(marker_genes, rownames(seu))

p_dot <- DotPlot(
  seu,
  features = marker_genes,
  group.by = "celltype_marker_based"
) + RotatedAxis()

ggsave(
  file.path(project_dir, "results", "figures", "singlet_dotplot_celltype_markers.pdf"),
  p_dot,
  width = 16,
  height = 8,
  limitsize = FALSE
)

# -----------------------------
# 6. Save
# -----------------------------
saveRDS(
  seu,
  file.path(project_dir, "ipf_annotated_singlet_seurat.rds")
)

write.csv(
  seu@meta.data,
  file.path(project_dir, "metadata", "cell_metadata_
            annotated_singlet.csv")
)

message("Singlet annotated object saved: ipf_annotated_singlet_seurat.rds")