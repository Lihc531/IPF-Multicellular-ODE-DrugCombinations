# 03c_normalize_after_doublet.R
# Normalize singlet-only object after doublet filtering

library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)
library(future)

plan("sequential")
options(future.globals.maxSize = 20 * 1024^3)

project_dir <- getwd()

# -----------------------------
# 1. Read singlet object
# -----------------------------
seu_qc <- readRDS(file.path(project_dir, "ipf_qc_singlet_seurat.rds"))

DefaultAssay(seu_qc) <- "RNA"

if ("Library_Identity" %in% colnames(seu_qc@meta.data)) {
  seu_qc$sample_id <- seu_qc$Library_Identity
} else if ("Subject_Identity" %in% colnames(seu_qc@meta.data)) {
  seu_qc$sample_id <- seu_qc$Subject_Identity
}

print(table(seu_qc$sample_id))

# -----------------------------
# 2. QC plots after doublet filtering
# -----------------------------
dir.create(file.path(project_dir, "results", "qc"), recursive = TRUE, showWarnings = FALSE)

p1 <- VlnPlot(
  seu_qc,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "sample_id",
  pt.size = 0,
  ncol = 3
)

ggsave(
  file.path(project_dir, "results", "qc", "qc_violin_after_doublet_filter.pdf"),
  p1,
  width = 14,
  height = 6,
  limitsize = FALSE
)

# -----------------------------
# 3. Split by sample for low-memory normalization
# -----------------------------
seu_list <- SplitObject(seu_qc, split.by = "sample_id")

seu_list <- lapply(seu_list, function(x) {
  message("Normalizing sample: ", unique(x$sample_id))
  
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

# -----------------------------
# 4. Select global variable features from sample-level HVGs
#    Do NOT run FindVariableFeatures again after merge
# -----------------------------
features_use <- SelectIntegrationFeatures(
  object.list = seu_list,
  nfeatures = 3000
)

# 如果内存仍紧张，可改成 2000：
# features_use <- SelectIntegrationFeatures(object.list = seu_list, nfeatures = 2000)

# -----------------------------
# 5. Merge back
# -----------------------------
seu_norm <- merge(
  x = seu_list[[1]],
  y = seu_list[-1],
  project = "IPF_GSE136831_singlet"
)

rm(seu_list)
gc()

DefaultAssay(seu_norm) <- "RNA"

VariableFeatures(seu_norm) <- features_use

# -----------------------------
# 6. Scale only selected features
# -----------------------------
seu_norm <- ScaleData(
  seu_norm,
  features = features_use,
  vars.to.regress = "percent.mt",
  block.size = 500,
  verbose = TRUE
)

# -----------------------------
# 7. Save
# -----------------------------
saveRDS(
  seu_norm,
  file.path(project_dir, "ipf_qc_singlet_normalized_seurat.rds")
)

write.csv(
  seu_norm@meta.data,
  file.path(project_dir, "cell_metadata_qc_singlet_normalized.csv")
)


message("Singlet normalized object saved: ipf_qc_singlet_normalized_seurat.rds")

message("Singlet normalized object saved: ipf_qc_singlet_normalized_seurat.rds")