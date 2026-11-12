library(Seurat)
library(dplyr)
library(future)

plan("sequential")
options(future.globals.maxSize = 20 * 1024^3)

project_dir <- getwd()

seu <- readRDS(
  file.path(project_dir, "ipf_integrated_singlet_seurat_nomarkers.rds")
)

DefaultAssay(seu) <- "RNA"
Idents(seu) <- "cluster_default"

dir.create(
  file.path(project_dir, "results", "markers"),
  recursive = TRUE,
  showWarnings = FALSE
)

# -----------------------------
# 1. 每个 cluster 抽样细胞名，不直接 subset
# -----------------------------
set.seed(123)

cluster_ids <- levels(Idents(seu))

cells_use <- unlist(lapply(cluster_ids, function(cl) {
  cells_cl <- WhichCells(seu, idents = cl)
  sample(cells_cl, min(500, length(cells_cl)))
}))

length(cells_use)

# -----------------------------
# 2. 用 cells 子集，而不是 downsample
# -----------------------------
seu_marker <- subset(
  seu,
  cells = cells_use
)

rm(seu)
gc()

# -----------------------------
# 3. Seurat v5 合并 layers
# -----------------------------
DefaultAssay(seu_marker) <- "RNA"

seu_marker <- JoinLayers(
  object = seu_marker,
  assay = "RNA"
)

Idents(seu_marker) <- "cluster_default"

features_use <- VariableFeatures(seu_marker)

if (length(features_use) == 0) {
  features_use <- rownames(seu_marker)
}

# 为进一步省内存，只用前 2000 个 HVG
features_use <- head(features_use, 2000)

# -----------------------------
# 4. FindAllMarkers
# -----------------------------
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
  file.path(project_dir, "results", "markers", "singlet_cluster_markers_res06_downsample500.csv"),
  row.names = FALSE
)

saveRDS(
  seu_marker,
  file.path(project_dir, "ipf_marker_downsample_object.rds")
)

message("Low-memory marker detection finished.")