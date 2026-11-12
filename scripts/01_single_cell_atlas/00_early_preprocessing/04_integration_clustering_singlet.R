# 04_integration_clustering_singlet.R
# PCA, Harmony, UMAP, clustering and low-memory marker detection for singlet object

library(Seurat)
library(harmony)
library(ggplot2)
library(patchwork)
library(future)

plan("sequential")
options(future.globals.maxSize = 20 * 1024^3)

project_dir <- getwd()

seu <- readRDS(file.path(project_dir, "ipf_qc_singlet_normalized_seurat.rds"))

DefaultAssay(seu) <- "RNA"

if ("Library_Identity" %in% colnames(seu@meta.data)) {
  seu$sample_id <- seu$Library_Identity
}

batch_var <- "sample_id"

message("Using batch variable: ", batch_var)
print(table(seu@meta.data[[batch_var]]))
message("Number of batch levels: ", length(unique(seu@meta.data[[batch_var]])))

dir.create(file.path(project_dir, "results", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "markers"), recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 1. PCA
# -----------------------------
seu <- RunPCA(
  seu,
  features = VariableFeatures(seu),
  npcs = 50,
  verbose = TRUE
)

p_elbow <- ElbowPlot(seu, ndims = 50)

ggsave(
  file.path(project_dir, "results", "figures", "singlet_elbowplot_pca.pdf"),
  p_elbow,
  width = 7,
  height = 5
)

# -----------------------------
# 2. Harmony
# -----------------------------
n_batch <- length(unique(seu@meta.data[[batch_var]]))

if (n_batch >= 2) {
  seu <- harmony::RunHarmony(
    object = seu,
    group.by.vars = batch_var,
    reduction.use = "pca",
    dims.use = 1:30
  )
  reduction_use <- "harmony"
} else {
  message("Only one batch level detected. Skipping Harmony.")
  reduction_use <- "pca"
}

# -----------------------------
# 3. UMAP and clustering
# -----------------------------
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

# -----------------------------
# 4. UMAP plots
# -----------------------------
p_cluster <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "cluster_default",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(project_dir, "results", "figures", "singlet_umap_clusters_res06.pdf"),
  p_cluster,
  width = 8,
  height = 6,
  limitsize = FALSE
)

p_sample <- DimPlot(
  seu,
  reduction = "umap",
  group.by = "sample_id",
  raster = TRUE
)

ggsave(
  file.path(project_dir, "results", "figures", "singlet_umap_sample_id.pdf"),
  p_sample,
  width = 12,
  height = 8,
  limitsize = FALSE
)

if ("Disease_Identity" %in% colnames(seu@meta.data)) {
  p_disease <- DimPlot(
    seu,
    reduction = "umap",
    group.by = "Disease_Identity",
    raster = TRUE
  )
  
  ggsave(
    file.path(project_dir, "results", "figures", "singlet_umap_disease_identity.pdf"),
    p_disease,
    width = 8,
    height = 6,
    limitsize = FALSE
  )
}

if ("CellType_Category" %in% colnames(seu@meta.data)) {
  p_original_major <- DimPlot(
    seu,
    reduction = "umap",
    group.by = "CellType_Category",
    label = TRUE,
    repel = TRUE,
    raster = TRUE
  )
  
  ggsave(
    file.path(project_dir, "results", "figures", "singlet_umap_original_celltype_category.pdf"),
    p_original_major,
    width = 9,
    height = 7,
    limitsize = FALSE
  )
}

# -----------------------------
# 5. Low-memory markers
# -----------------------------
DefaultAssay(seu) <- "RNA"
Idents(seu) <- "cluster_default"

set.seed(123)

seu_marker <- subset(
  seu,
  downsample = 1000
)

seu_marker <- JoinLayers(
  object = seu_marker,
  assay = "RNA"
)

DefaultAssay(seu_marker) <- "RNA"
Idents(seu_marker) <- "cluster_default"

features_use <- VariableFeatures(seu)

markers <- FindAllMarkers(
  seu_marker,
  features = features_use,
  only.pos = TRUE,
  min.pct = 0.25,
  logfc.threshold = 0.25,
  test.use = "wilcox",
  verbose = TRUE
)

write.csv(
  markers,
  file.path(project_dir, "results", "markers", "singlet_cluster_markers_res06_downsample1000.csv"),
  row.names = FALSE
)

rm(seu_marker)
gc()
# -----------------------------
# 6. Save
# -----------------------------
saveRDS(
  seu,
  file.path(project_dir, "ipf_integrated_singlet_seurat.rds")
)

message("Singlet integrated object saved: ipf_integrated_singlet_seurat.rds")