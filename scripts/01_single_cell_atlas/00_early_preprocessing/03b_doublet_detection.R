# 03b_doublet_detection.R
# QC filtering + scDblFinder doublet detection by sample/library

library(Seurat)
library(SingleCellExperiment)
library(scDblFinder)
library(BiocParallel)
library(dplyr)
library(future)

plan("sequential")
project_dir <- getwd()

# -----------------------------
# 1. Read raw Seurat object
# -----------------------------
seu <- readRDS(file.path(project_dir, "ipf_raw_seurat.rds"))

# -----------------------------
# 2. QC metrics
# -----------------------------
seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^MT-")
seu[["percent.ribo"]] <- PercentageFeatureSet(seu, pattern = "^RPL|^RPS")
seu[["percent.hb"]] <- PercentageFeatureSet(seu, pattern = "^HB[AB]")

# -----------------------------
# 3. Use real sample/library identity
# -----------------------------
if ("Library_Identity" %in% colnames(seu@meta.data)) {
  seu$sample_id <- seu$Library_Identity
} else if ("Subject_Identity" %in% colnames(seu@meta.data)) {
  seu$sample_id <- seu$Subject_Identity
} else if ("orig.ident" %in% colnames(seu@meta.data)) {
  seu$sample_id <- seu$orig.ident
} else {
  stop("No suitable sample identity column found.")
}

print(table(seu$sample_id))

# -----------------------------
# 4. QC filtering
# -----------------------------
seu_qc <- subset(
  seu,
  subset =
    nFeature_RNA >= 200 &
    nFeature_RNA <= 6000 &
    nCount_RNA >= 500 &
    percent.mt <= 20
)

message("Cells after QC filtering: ", ncol(seu_qc))

# -----------------------------
# 5. Run scDblFinder by sample
# -----------------------------
sce <- as.SingleCellExperiment(seu_qc)

sample_ids <- unique(as.character(sce$sample_id))

doublet_df_list <- lapply(sample_ids, function(sid) {
  message("Running scDblFinder for sample: ", sid)
  
  sce_sub <- sce[, as.character(sce$sample_id) == sid]
  
  if (ncol(sce_sub) < 500) {
    message("Sample ", sid, " has fewer than 500 cells. Marking all as singlet.")
    
    return(data.frame(
      cell = colnames(sce_sub),
      sample_id = sid,
      doublet_class = "singlet",
      doublet_score = 0,
      stringsAsFactors = FALSE
    ))
  }
  
  sce_sub <- scDblFinder(
    sce_sub,
    BPPARAM = BiocParallel::SerialParam()
  )
  
  data.frame(
    cell = colnames(sce_sub),
    sample_id = sid,
    doublet_class = as.character(sce_sub$scDblFinder.class),
    doublet_score = as.numeric(sce_sub$scDblFinder.score),
    stringsAsFactors = FALSE
  )
})

doublet_df <- do.call(rbind, doublet_df_list)

# -----------------------------
# 6. Match doublet results back to Seurat object
# -----------------------------
idx <- match(colnames(seu_qc), doublet_df$cell)

if (any(is.na(idx))) {
  stop("Some Seurat cells were not found in doublet_df. Please check cell barcodes.")
}

seu_qc$doublet_class <- doublet_df$doublet_class[idx]
seu_qc$doublet_score <- doublet_df$doublet_score[idx]

# Check
print(table(seu_qc$doublet_class))

meta_df <- as.data.frame(seu_qc@meta.data)

doublet_summary <- meta_df %>%
  dplyr::group_by(sample_id, doublet_class) %>%
  dplyr::summarise(n_cells = dplyr::n(), .groups = "drop") %>%
  dplyr::group_by(sample_id) %>%
  dplyr::mutate(fraction = n_cells / sum(n_cells)) %>%
  dplyr::ungroup()

write.csv(
  doublet_summary,
  file.path(project_dir, "results", "qc", "doublet_summary_by_sample.csv"),
  row.names = FALSE
)

# -----------------------------
# 7. Save scored and singlet-only objects
# -----------------------------
saveRDS(
  seu_qc,
  file.path(project_dir, "ipf_qc_with_doublet_score.rds")
)

seu_singlet <- subset(
  seu_qc,
  subset = doublet_class == "singlet"
)

saveRDS(
  seu_singlet,
  file.path(project_dir, "ipf_qc_singlet_seurat.rds")
)

write.csv(
  seu_singlet@meta.data,
  file.path(project_dir, "cell_metadata_qc_singlet.csv")
)

message("Doublet detection finished.")
message("Saved: ipf_qc_with_doublet_score.rds")
message("Saved: ipf_qc_singlet_seurat.rds")

doublet_summary <- read.csv("results/qc/doublet_summary_by_sample.csv")
head(doublet_summary)
table(doublet_summary$doublet_class)
