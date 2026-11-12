# 01_preprocess_GSE159354_validation.R
# GSE159354 independent validation preprocessing
# Input: GSE159354 RAW 10x h5 files
# Output: cleaned / integrated Seurat object + metadata + UMAP plots

suppressPackageStartupMessages({
  library(Seurat)
  library(harmony)
  library(dplyr)
  library(ggplot2)
  library(Matrix)
  library (hdf5r) 
  library(future)
})

plan("sequential")
options(future.globals.maxSize = 30 * 1024^3)
set.seed(123)

project_dir <- getwd()

raw_dir <- file.path(project_dir, "GSE159354_RAW")
out_dir <- file.path(project_dir, "results_GSE159354")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
meta_dir <- file.path(out_dir, "metadata")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(meta_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Locate 10x h5 files
# ============================================================

h5_files <- list.files(
  raw_dir,
  pattern = "\\.h5$",
  full.names = TRUE,
  recursive = TRUE
)

if (length(h5_files) == 0) {
  stop("No .h5 files found in: ", raw_dir)
}

message("Found h5 files:")
print(h5_files)

# ============================================================
# 2. Read each sample
# ============================================================

seu_list <- list()

for (f in h5_files) {
  sample_id <- tools::file_path_sans_ext(basename(f))
  sample_id <- gsub("_filtered_feature_bc_matrix|_raw_feature_bc_matrix", "", sample_id)
  
  message("Reading sample: ", sample_id)
  
  counts <- Read10X_h5(f)
  
  # If Read10X_h5 returns multi-assay list, use Gene Expression
  if (is.list(counts)) {
    if ("Gene Expression" %in% names(counts)) {
      counts <- counts[["Gene Expression"]]
    } else {
      counts <- counts[[1]]
    }
  }
  
  seu <- CreateSeuratObject(
    counts = counts,
    project = sample_id,
    min.cells = 3,
    min.features = 200
  )
  
  seu$sample_id <- sample_id
  seu$dataset <- "GSE159354"
  
  seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^MT-")
  seu[["percent.ribo"]] <- PercentageFeatureSet(seu, pattern = "^RP[SL]")
  
  seu_list[[sample_id]] <- seu
}

# ============================================================
# 3. Merge samples
# ============================================================

if (length(seu_list) == 1) {
  seu <- seu_list[[1]]
} else {
  seu <- merge(
    x = seu_list[[1]],
    y = seu_list[-1],
    add.cell.ids = names(seu_list),
    project = "GSE159354_validation"
  )
}

rm(seu_list)
gc()

message("Merged object:")
print(dim(seu))

saveRDS(
  seu,
  file.path(out_dir, "GSE159354_raw_merged_seurat.rds")
)

# ============================================================
# 4. Basic QC summary before filtering
# ============================================================

qc_before <- seu@meta.data %>%
  group_by(sample_id) %>%
  summarise(
    n_cells = n(),
    median_nFeature_RNA = median(nFeature_RNA),
    median_nCount_RNA = median(nCount_RNA),
    median_percent_mt = median(percent.mt),
    .groups = "drop"
  )

write.csv(
  qc_before,
  file.path(tab_dir, "01_QC_summary_before_filtering.csv"),
  row.names = FALSE
)

p <- VlnPlot(
  seu,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "sample_id",
  pt.size = 0,
  ncol = 3
)

ggsave(
  file.path(fig_dir, "01_QC_violin_before_filtering.pdf"),
  p,
  width = 14,
  height = 6,
  limitsize = FALSE
)

# ============================================================
# 5. QC filtering
#    Thresholds can be adjusted after viewing QC violin plot
# ============================================================

seu <- subset(
  seu,
  subset =
    nFeature_RNA >= 300 &
    nFeature_RNA <= 6000 &
    nCount_RNA >= 500 &
    percent.mt <= 20
)

message("After QC filtering:")
print(dim(seu))

qc_after <- seu@meta.data %>%
  group_by(sample_id) %>%
  summarise(
    n_cells = n(),
    median_nFeature_RNA = median(nFeature_RNA),
    median_nCount_RNA = median(nCount_RNA),
    median_percent_mt = median(percent.mt),
    .groups = "drop"
  )

write.csv(
  qc_after,
  file.path(tab_dir, "02_QC_summary_after_filtering.csv"),
  row.names = FALSE
)

p <- VlnPlot(
  seu,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "sample_id",
  pt.size = 0,
  ncol = 3
)

ggsave(
  file.path(fig_dir, "02_QC_violin_after_filtering.pdf"),
  p,
  width = 14,
  height = 6,
  limitsize = FALSE
)

saveRDS(
  seu,
  file.path(out_dir, "GSE159354_qc_filtered_seurat.rds")
)

# ============================================================
# 6. Normalize sample-by-sample
# ============================================================

DefaultAssay(seu) <- "RNA"

seu_list <- SplitObject(seu, split.by = "sample_id")

seu_list <- lapply(seu_list, function(x) {
  DefaultAssay(x) <- "RNA"
  
  x <- NormalizeData(
    x,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    verbose = TRUE
  )
  
  x <- FindVariableFeatures(
    x,
    selection.method = "vst",
    nfeatures = 3000,
    verbose = TRUE
  )
  
  return(x)
})

features_use <- SelectIntegrationFeatures(
  object.list = seu_list,
  nfeatures = 3000
)

seu <- merge(
  x = seu_list[[1]],
  y = seu_list[-1],
  project = "GSE159354_validation_normalized",
  merge.data = TRUE
)

rm(seu_list)
gc()

DefaultAssay(seu) <- "RNA"
VariableFeatures(seu) <- features_use

# ============================================================
# 7. Scale, PCA, Harmony, UMAP, clustering
# ============================================================

seu <- ScaleData(
  seu,
  features = features_use,
  vars.to.regress = "percent.mt",
  block.size = 500,
  verbose = TRUE
)

seu <- RunPCA(
  seu,
  features = features_use,
  npcs = 50,
  verbose = TRUE
)

p <- ElbowPlot(seu, ndims = 50)

ggsave(
  file.path(fig_dir, "03_elbowplot_pca.pdf"),
  p,
  width = 7,
  height = 5
)

if (length(unique(seu$sample_id)) >= 2) {
  seu <- RunHarmony(
    object = seu,
    group.by.vars = "sample_id",
    reduction.use = "pca",
    dims.use = 1:30,
    reduction.save = "harmony"
  )
  
  reduction_use <- "harmony"
} else {
  reduction_use <- "pca"
}

seu <- RunUMAP(
  seu,
  reduction = reduction_use,
  dims = 1:30,
  verbose = TRUE
)

seu <- FindNeighbors(
  seu,
  reduction = reduction_use,
  dims = 1:30,
  verbose = TRUE
)

for (res in c(0.2, 0.4, 0.6, 0.8, 1.0)) {
  seu <- FindClusters(
    seu,
    resolution = res,
    algorithm = 1,
    verbose = TRUE
  )
}

seu$cluster_default <- seu$RNA_snn_res.0.6
Idents(seu) <- "cluster_default"

# ============================================================
# 8. UMAP plots
# ============================================================

p <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "cluster_default",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(fig_dir, "04_umap_clusters_res06.pdf"),
  p,
  width = 8,
  height = 6,
  limitsize = FALSE
)

p <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "sample_id",
  raster = TRUE
)

ggsave(
  file.path(fig_dir, "05_umap_sample_id.pdf"),
  p,
  width = 12,
  height = 8,
  limitsize = FALSE
)

# ============================================================
# 9. Preliminary marker-based major annotation
# ============================================================

DefaultAssay(seu) <- "RNA"

message("RNA layers before JoinLayers:")
print(Layers(seu[["RNA"]]))

seu <- JoinLayers(
  object = seu,
  assay = "RNA"
)

message("RNA layers after JoinLayers:")
print(Layers(seu[["RNA"]]))

# Safety check: make sure normalized data exists
if (!"data" %in% Layers(seu[["RNA"]])) {
  message("No RNA data layer found after JoinLayers. Running NormalizeData again.")
  seu <- NormalizeData(
    seu,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    verbose = TRUE
  )
}

marker_list <- list(
  Epithelial = c("EPCAM", "KRT8", "KRT18", "KRT19"),
  AT1 = c("AGER", "PDPN", "CAV1", "CLIC5"),
  AT2 = c("SFTPC", "SFTPA1", "SFTPA2", "ABCA3", "LAMP3"),
  Basal = c("KRT5", "KRT17", "TP63", "KRT15"),
  Ciliated = c("FOXJ1", "PIFO", "TUBB4B", "CAPS"),
  Secretory = c("SCGB1A1", "SCGB3A1", "MUC5B", "MUC5AC"),
  Fibroblast = c("COL1A1", "COL1A2", "DCN", "LUM"),
  Myofibroblast = c("ACTA2", "TAGLN", "POSTN", "COL3A1"),
  Endothelial = c("PECAM1", "VWF", "KDR", "CLDN5"),
  Macrophage = c("CD68", "MARCO", "C1QA", "C1QB", "APOE"),
  Monocyte = c("LYZ", "S100A8", "S100A9", "FCN1"),
  Dendritic = c("FCER1A", "CLEC10A", "LILRA4", "ITGAX"),
  T_cell = c("CD3D", "CD3E", "TRAC"),
  NK = c("NKG7", "GNLY", "PRF1", "KLRD1"),
  B_cell = c("MS4A1", "CD79A", "CD79B"),
  Plasma = c("MZB1", "JCHAIN", "XBP1"),
  Mast = c("TPSAB1", "TPSB2", "KIT"),
  Smooth_muscle = c("MYH11", "ACTA2", "TAGLN")
)

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

cluster_scores <- seu@meta.data %>%
  group_by(cluster_default) %>%
  summarise(
    across(all_of(score_cols), mean),
    .groups = "drop"
  )

write.csv(
  cluster_scores,
  file.path(tab_dir, "03_cluster_marker_module_scores.csv"),
  row.names = FALSE
)

score_mat <- as.matrix(cluster_scores[, score_cols, drop = FALSE])
best <- score_cols[max.col(score_mat, ties.method = "first")]
best <- gsub("_score1$", "", best)

cluster_to_type <- data.frame(
  cluster_default = cluster_scores$cluster_default,
  predicted_celltype_marker = best
)

write.csv(
  cluster_to_type,
  file.path(tab_dir, "04_cluster_to_marker_based_celltype.csv"),
  row.names = FALSE
)

map_vec <- setNames(
  cluster_to_type$predicted_celltype_marker,
  cluster_to_type$cluster_default
)

seu$celltype_marker_based <- unname(
  map_vec[as.character(seu$cluster_default)]
)

p <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "celltype_marker_based",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(fig_dir, "06_umap_marker_based_celltype.pdf"),
  p,
  width = 10,
  height = 8,
  limitsize = FALSE
)

marker_genes <- unique(unlist(marker_list))
marker_genes <- intersect(marker_genes, rownames(seu))

p <- DotPlot(
  seu,
  features = marker_genes,
  group.by = "celltype_marker_based"
) + RotatedAxis()

ggsave(
  file.path(fig_dir, "07_dotplot_marker_based_celltype.pdf"),
  p,
  width = 18,
  height = 8,
  limitsize = FALSE
)

# ============================================================
# 10. Export metadata, embeddings, and object
# ============================================================

write.csv(
  seu@meta.data,
  file.path(meta_dir, "GSE159354_cell_metadata_preprocessed.csv")
)

umap <- Embeddings(seu, "umap")

write.csv(
  umap,
  file.path(tab_dir, "05_umap_embedding.csv")
)

if ("harmony" %in% Reductions(seu)) {
  harmony_emb <- Embeddings(seu, "harmony")
  
  write.csv(
    harmony_emb,
    file.path(tab_dir, "06_harmony_embedding.csv")
  )
}

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$celltype_marker_based)),
  file.path(tab_dir, "07_cluster_vs_marker_based_celltype.csv")
)

write.csv(
  as.data.frame.matrix(table(seu$cluster_default, seu$sample_id)),
  file.path(tab_dir, "08_cluster_vs_sample_id.csv")
)

saveRDS(
  seu,
  file.path(out_dir, "GSE159354_preprocessed_integrated_seurat.rds")
)

message("Done.")
message("Main output:")
message(file.path(out_dir, "GSE159354_preprocessed_integrated_seurat.rds"))