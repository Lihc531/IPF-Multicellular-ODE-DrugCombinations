# 02_build_seurat_object.R
# Build Seurat object from downloaded expression matrix and metadata.
# This script assumes sparse Matrix Market format.
# You may need to modify file patterns depending on the downloaded GEO files.

library(Seurat)
library(Matrix)
library(readr)
library(dplyr)
library(stringr)

project_dir <- getwd()
data_dir <- file.path(project_dir, "GSE136831")

find_first <- function(pattern, folder = data_dir) {
  hits <- list.files(folder, pattern = pattern, recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  if (length(hits) == 0) {
    stop("No file found for pattern: ", pattern)
  }
  hits[1]
}

# Expected files may have names similar to:
# RawCounts_Sparse.mtx.gz
# RawCounts_Barcodes.tsv.gz
# RawCounts_Genes.tsv.gz
# metadata.tsv.gz / metadata.csv.gz
mtx_file <- find_first("mtx(\\.gz)?$")
barcode_file <- find_first("barcode|barcodes")
gene_file <- find_first("gene|genes|features")
meta_candidates <- list.files(data_dir, pattern = "metadata|meta|annotation", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)

message("Matrix file: ", mtx_file)
message("Barcode file: ", barcode_file)
message("Gene file: ", gene_file)

counts <- Matrix::readMM(mtx_file)

barcodes <- read.delim(
  barcode_file,
  header = FALSE,
  stringsAsFactors = FALSE
)[, 1]

genes_table <- read.delim(
  gene_file,
  header = TRUE,
  stringsAsFactors = FALSE
)

gene_symbols <- genes_table[, ncol(genes_table)]
gene_symbols <- make.unique(as.character(gene_symbols))

# 检查长度
length(gene_symbols)
nrow(counts)

rownames(counts) <- gene_symbols
colnames(counts) <- barcodes

seu <- CreateSeuratObject(
  counts = counts,
  project = "IPF_GSE136831",
  min.cells = 3,
  min.features = 200
)

if (length(meta_candidates) > 0) {
  meta_file <- meta_candidates[1]
  message("Metadata file: ", meta_file)

  if (grepl("\\.csv(\\.gz)?$", meta_file, ignore.case = TRUE)) {
    meta <- read.csv(meta_file, stringsAsFactors = FALSE)
  } else {
    meta <- read.delim(meta_file, stringsAsFactors = FALSE)
  }

  # Try to infer cell barcode column
  barcode_cols <- grep("barcode|cell|Cell|CELL", colnames(meta), value = TRUE)
  if (length(barcode_cols) > 0) {
    bc_col <- barcode_cols[1]
    rownames(meta) <- meta[[bc_col]]
    common_cells <- intersect(colnames(seu), rownames(meta))
    seu <- subset(seu, cells = common_cells)
    seu <- AddMetaData(seu, metadata = meta[common_cells, , drop = FALSE])
  } else {
    warning("Could not infer barcode column in metadata. Metadata not added.")
  }
} else {
  warning("No metadata file detected. Please add metadata manually.")
}

saveRDS(seu, file = file.path(project_dir,"ipf_raw_seurat.rds"))

write.csv(
  seu@meta.data,
  file = file.path(project_dir, "cell_metadata_raw.csv")
)

message("Seurat object saved: objects/ipf_raw_seurat.rds")
