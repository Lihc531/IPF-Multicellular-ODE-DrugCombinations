# 10_add_fibrosis_module_pathway_scores_to_axis.R
# Add fibrosis/module/pathway scores to multi-cellular disease progression axis
# Low-memory version:
# - does not subset full Seurat object
# - does not merge large Seurat objects
# - calculates module scores gene-set by gene-set
# - outputs axis v2 metadata and validation summaries

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(tibble)
})

set.seed(123)

project_dir <- getwd()

# ============================================================
# 0. Input / output paths
# ============================================================

seurat_rds <- file.path(project_dir, "ipf_annotated_singlet_seurat.rds")

cells_keep_file <- file.path(
  project_dir,
  "metadata",
  "cells_keep_nomultiplet.txt"
)

axis_meta_rds <- file.path(
  project_dir,
  "results",
  "progression_axis",
  "tables",
  "11_cell_metadata_with_multicellular_progression_axis.rds"
)

out_dir <- file.path(project_dir, "results", "progression_axis_v2")
tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Parameters
# ============================================================

assay_use <- "RNA"

# In Seurat v5, use "data" layer.
# In Seurat v4, this corresponds to slot = "data".
layer_use <- "data"

min_genes_per_module <- 2

# Axis v2 weights
# v1 score = composition-informed multi-cellular axis
# fibrosis composite = fibrosis / ECM / collagen / TGF / myofibroblast signals
# injury inflammatory composite = epithelial injury / inflammation / macrophage / hypoxia signals
w_axis_v1 <- 0.50
w_fibrosis <- 0.30
w_injury_inflammation <- 0.20

# ============================================================
# 2. Helper functions
# ============================================================

rescale01 <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || diff(rng) == 0) {
    return(rep(0.5, length(x)))
  }
  (x - rng[1]) / (rng[2] - rng[1])
}

safe_zscore <- function(x) {
  x <- as.numeric(x)
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) {
    return(rep(0, length(x)))
  }
  (x - m) / s
}

make_stage <- function(score) {
  cut(
    score,
    breaks = c(-Inf, 0.25, 0.50, 0.75, Inf),
    labels = c(
      "Stage_1_low",
      "Stage_2_intermediate_low",
      "Stage_3_intermediate_high",
      "Stage_4_high"
    ),
    right = TRUE
  )
}

safe_cor <- function(x, y, method = "spearman") {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 10) {
    return(NA_real_)
  }
  suppressWarnings(cor(x[ok], y[ok], method = method))
}

get_assay_matrix <- function(seu, assay = "RNA", layer = "data") {
  DefaultAssay(seu) <- assay
  
  mat <- NULL
  
  # Seurat v5
  mat <- tryCatch(
    {
      LayerData(seu, assay = assay, layer = layer)
    },
    error = function(e) NULL
  )
  
  # Seurat v4 fallback
  if (is.null(mat)) {
    mat <- tryCatch(
      {
        GetAssayData(seu, assay = assay, slot = layer)
      },
      error = function(e) NULL
    )
  }
  
  if (is.null(mat)) {
    stop("Cannot extract assay matrix. Please check assay/layer: ", assay, " / ", layer)
  }
  
  mat
}

calculate_module_score <- function(expr_mat, genes, cells_use, module_name, min_genes = 2) {
  genes_present <- intersect(genes, rownames(expr_mat))
  genes_missing <- setdiff(genes, rownames(expr_mat))
  
  message("Module: ", module_name)
  message("  genes present: ", length(genes_present))
  message("  genes missing: ", length(genes_missing))
  
  if (length(genes_present) < min_genes) {
    warning("Module ", module_name, " has fewer than ", min_genes, " genes present. Returning NA.")
    return(list(
      score = rep(NA_real_, length(cells_use)),
      genes_present = genes_present,
      genes_missing = genes_missing
    ))
  }
  
  calculate_module_score <- function(expr_mat, genes, cells_use, module_name, min_genes = 2) {
    genes_present <- intersect(genes, rownames(expr_mat))
    genes_missing <- setdiff(genes, rownames(expr_mat))
    
    cells_present <- intersect(cells_use, colnames(expr_mat))
    cells_missing <- setdiff(cells_use, colnames(expr_mat))
    
    message("Module: ", module_name)
    message("  genes present: ", length(genes_present))
    message("  genes missing: ", length(genes_missing))
    message("  cells requested: ", length(cells_use))
    message("  cells present in expr_mat: ", length(cells_present))
    message("  cells missing from expr_mat: ", length(cells_missing))
    
    if (length(genes_present) < min_genes) {
      warning("Module ", module_name, " has fewer than ", min_genes, " genes present. Returning NA.")
      return(list(
        score = rep(NA_real_, length(cells_use)),
        genes_present = genes_present,
        genes_missing = genes_missing,
        cells_present = cells_present,
        cells_missing = cells_missing
      ))
    }
    
    if (length(cells_present) < 1000) {
      stop(
        "Too few cells from cells_use are present in expr_mat for module ",
        module_name,
        ". Please check cell ID formats."
      )
    }
    
    sub_mat <- expr_mat[genes_present, cells_present, drop = FALSE]
    score_present <- Matrix::colMeans(sub_mat)
    
    score <- rep(NA_real_, length(cells_use))
    names(score) <- cells_use
    score[names(score_present)] <- as.numeric(score_present)
    
    rm(sub_mat)
    gc()
    
    list(
      score = as.numeric(score),
      genes_present = genes_present,
      genes_missing = genes_missing,
      cells_present = cells_present,
      cells_missing = cells_missing
    )
  }
  
  score <- Matrix::colMeans(sub_mat)
  
  rm(sub_mat)
  gc()
  
  list(
    score = as.numeric(score),
    genes_present = genes_present,
    genes_missing = genes_missing
  )
}

# ============================================================
# 3. Gene sets
# ============================================================

gene_sets <- list(
  fibrosis_core = c(
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
    "FN1", "POSTN", "THBS1", "SPARC", "CTGF", "SERPINE1"
  ),
  
  ECM_remodeling = c(
    "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2",
    "FN1", "LUM", "DCN", "BGN", "MMP2", "MMP7", "MMP9",
    "MMP14", "TIMP1", "TIMP2", "LOX", "LOXL1", "LOXL2"
  ),
  
  collagen_formation = c(
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
    "COL6A1", "COL6A2", "COL6A3", "COL14A1", "COL15A1"
  ),
  
  TGF_beta_signaling = c(
    "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
    "SMAD4", "SMAD7", "SERPINE1", "CTGF", "THBS1", "PMEPA1",
    "INHBA", "ITGB6"
  ),
  
  myofibroblast_activation = c(
    "ACTA2", "TAGLN", "MYH11", "CNN1", "POSTN", "COL1A1",
    "COL3A1", "FN1", "CTGF", "PDGFRB", "MCAM", "RGS5"
  ),
  
  fibroblast_homeostatic = c(
    "DCN", "LUM", "COL1A2", "COL1A1", "PDGFRA", "DPT",
    "PI16", "MFAP5", "CFD", "APOD"
  ),
  
  epithelial_injury = c(
    "KRT8", "KRT18", "KRT19", "CLDN4", "MUC1", "S100A2",
    "ANXA2", "LGALS3", "SERPINE1", "LCN2", "CXCL8"
  ),
  
  aberrant_basaloid = c(
    "KRT5", "KRT17", "TP63", "KRT14", "ITGA6", "COL1A1",
    "FN1", "LAMC2", "S100A2", "MMP7"
  ),
  
  AT2_stress_regeneration = c(
    "SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "ABCA3",
    "NAPSA", "ETV5", "KRT8", "CLDN4", "MUC1"
  ),
  
  EMT_like = c(
    "VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
    "TWIST1", "CDH2", "ITGA5", "MMP2", "COL1A1"
  ),
  
  profibrotic_macrophage = c(
    "SPP1", "TREM2", "APOE", "MERTK", "MARCO", "CD68",
    "C1QA", "C1QB", "C1QC", "LGALS3", "CTSL", "CHI3L1"
  ),
  
  inflammatory_monocyte = c(
    "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "IL1B",
    "CXCL8", "CCL2", "CCL3", "CCL4", "NFKBIA"
  ),
  
  antigen_presentation = c(
    "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
    "HLA-DQA1", "HLA-DQB1", "CD74", "B2M", "TAP1", "TAP2"
  ),
  
  chemokine_inflammatory = c(
    "CCL2", "CCL3", "CCL4", "CCL5", "CXCL8", "CXCL9",
    "CXCL10", "CXCL11", "CXCL12", "IL1B", "TNF"
  ),
  
  angiogenesis = c(
    "PECAM1", "VWF", "KDR", "FLT1", "ENG", "ESAM",
    "ANGPT2", "TEK", "RAMP2", "ACKR1", "PLVAP"
  ),
  
  endothelial_activation = c(
    "ICAM1", "VCAM1", "SELE", "VWF", "ANGPT2", "ACKR1",
    "CXCL12", "KDR", "FLT1", "ENG"
  ),
  
  hypoxia = c(
    "HIF1A", "VEGFA", "SLC2A1", "LDHA", "ENO1",
    "PGK1", "CA9", "BNIP3", "NDRG1", "ADM"
  ),
  
  oxidative_stress = c(
    "NFE2L2", "HMOX1", "NQO1", "SOD1", "SOD2",
    "GPX1", "GPX3", "PRDX1", "PRDX2", "TXN", "GCLC"
  ),
  
  interferon_response = c(
    "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
    "OAS1", "OAS2", "OAS3", "STAT1", "IRF7", "IFI6"
  ),
  
  senescence_SASP = c(
    "CDKN1A", "CDKN2A", "IL6", "CXCL8", "CCL2",
    "MMP1", "MMP3", "SERPINE1", "IGFBP3", "GDF15"
  ),
  
  apoptosis_stress = c(
    "BAX", "BAK1", "CASP3", "CASP7", "CASP8",
    "FAS", "FASLG", "BCL2L11", "BBC3", "PMAIP1"
  ),
  
  cytotoxicity = c(
    "NKG7", "GNLY", "PRF1", "GZMB", "GZMA",
    "KLRD1", "KLRK1", "IFNG", "CCL5"
  ),
  
  T_cell_activation = c(
    "CD3D", "CD3E", "TRAC", "IL7R", "CCR7",
    "CD69", "ICOS", "TNFRSF4", "TNFRSF9"
  ),
  
  T_cell_exhaustion = c(
    "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    "TOX", "ENTPD1", "CXCL13"
  ),
  
  B_plasma_activation = c(
    "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN",
    "XBP1", "IGHG1", "IGHG3", "SDC1"
  )
)

module_names <- names(gene_sets)

# ============================================================
# 4. Load data
# ============================================================

if (!file.exists(seurat_rds)) {
  stop("Seurat object not found: ", seurat_rds)
}

if (!file.exists(cells_keep_file)) {
  stop("cells_keep_nomultiplet.txt not found: ", cells_keep_file)
}

if (!file.exists(axis_meta_rds)) {
  stop("Axis metadata RDS not found: ", axis_meta_rds)
}

message("Reading Seurat object...")
seu <- readRDS(seurat_rds)

message("Seurat object dimension:")
print(dim(seu))

message("Reading cells_keep...")
cells_keep <- readLines(cells_keep_file)

message("Reading axis metadata...")
axis_meta <- readRDS(axis_meta_rds)

if ("CellBarcode_Identity" %in% colnames(axis_meta)) {
  axis_meta$cell_id <- as.character(axis_meta$CellBarcode_Identity)
  rownames(axis_meta) <- axis_meta$cell_id
} else if ("cell_id" %in% colnames(axis_meta)) {
  rownames(axis_meta) <- as.character(axis_meta$cell_id)
} else {
  stop("No valid cell ID column found in axis_meta.")
}

message("Axis metadata dimension:")
print(dim(axis_meta))

# Make sure rownames are cell IDs
message("Checking cell ID compatibility...")

message("Example colnames(seu):")
print(head(colnames(seu)))

message("Example cells_keep:")
print(head(cells_keep))

message("Example rownames(axis_meta):")
print(head(rownames(axis_meta)))

message("axis_meta columns:")
print(colnames(axis_meta))

# Case 1: rownames(axis_meta) are valid cell IDs
overlap_axis_rownames <- length(intersect(cells_keep, rownames(axis_meta)))

# Case 2: axis_meta has explicit cell ID column
candidate_cell_id_cols <- c(
  "cell_id",
  "cell_id_join",
  "barcode",
  "Cell",
  "cell",
  "orig.ident"
)

candidate_cell_id_cols <- intersect(candidate_cell_id_cols, colnames(axis_meta))

message("Overlap with axis_meta rownames: ", overlap_axis_rownames)
message("Candidate cell ID columns: ", paste(candidate_cell_id_cols, collapse = ", "))

if (overlap_axis_rownames >= 1000) {
  
  message("Using rownames(axis_meta) as cell IDs.")
  axis_meta$cell_id_recovered <- rownames(axis_meta)
  
} else if (length(candidate_cell_id_cols) > 0) {
  
  best_col <- NULL
  best_overlap <- 0
  
  for (cc in candidate_cell_id_cols) {
    ov <- length(intersect(cells_keep, as.character(axis_meta[[cc]])))
    message("Overlap using column ", cc, ": ", ov)
    
    if (ov > best_overlap) {
      best_overlap <- ov
      best_col <- cc
    }
  }
  
  if (!is.null(best_col) && best_overlap >= 1000) {
    message("Using axis_meta column as cell IDs: ", best_col)
    axis_meta$cell_id_recovered <- as.character(axis_meta[[best_col]])
    rownames(axis_meta) <- axis_meta$cell_id_recovered
  } else {
    stop("Could not recover cell IDs from axis_meta columns.")
  }
  
} else {
  
  stop(
    "axis_meta rownames are not cell IDs and no usable cell ID column was found. ",
    "Please regenerate 11_cell_metadata_with_multicellular_progression_axis.rds with explicit cell_id column."
  )
}

cells_use <- Reduce(
  intersect,
  list(
    cells_keep,
    colnames(seu),
    rownames(axis_meta)
  )
)

message("Cells in cells_keep: ", length(cells_keep))
message("Cells in Seurat object: ", ncol(seu))
message("Cells in axis_meta: ", nrow(axis_meta))
message("Overlapping cells: ", length(cells_use))

if (length(cells_use) < 1000) {
  stop("Too few overlapping cells after recovery. Please check cell ID formats.")
}

axis_meta <- axis_meta[cells_use, , drop = FALSE]

message("Cells in cells_keep: ", length(cells_keep))
message("Cells available in Seurat + axis metadata: ", length(cells_use))

if (length(cells_use) < 1000) {
  stop("Too few overlapping cells. Please check cell IDs.")
}

# Reorder axis metadata
axis_meta <- axis_meta[cells_use, , drop = FALSE]

# ============================================================
# 5. Calculate module scores from Seurat v5 multi-layers
# ============================================================

# 说明：
# 你的 RNA assay 里有 data.1, data.2, ..., data.107 多个 layer。
# 不能直接 LayerData(seu, assay = "RNA", layer = "data")，
# 否则 Seurat 只会使用第一个 data layer。
# 这里按 layer 分块计算 module score，避免 JoinLayers 和内存爆炸。

get_layer_names <- function(seu, assay = "RNA", layer_prefix = "data") {
  all_layers <- Layers(seu[[assay]])
  
  layer_names <- all_layers[
    all_layers == layer_prefix |
      grepl(paste0("^", layer_prefix, "\\."), all_layers)
  ]
  
  if (length(layer_names) == 0) {
    stop("No layers found with prefix: ", layer_prefix)
  }
  
  return(layer_names)
}

calculate_module_scores_by_layers <- function(
    seu,
    assay = "RNA",
    layer_prefix = "data",
    cells_use,
    gene_sets,
    min_genes = 2
) {
  DefaultAssay(seu) <- assay
  
  layer_names <- get_layer_names(
    seu = seu,
    assay = assay,
    layer_prefix = layer_prefix
  )
  
  message("Detected expression layers:")
  print(layer_names)
  
  all_score_list <- list()
  gene_coverage_list <- list()
  
  for (layer_name in layer_names) {
    message("Processing layer: ", layer_name)
    
    mat <- LayerData(
      object = seu,
      assay = assay,
      layer = layer_name
    )
    
    layer_cells <- intersect(cells_use, colnames(mat))
    
    message("  layer dim: ", paste(dim(mat), collapse = " x "))
    message("  overlapping cells: ", length(layer_cells))
    
    if (length(layer_cells) == 0) {
      rm(mat)
      gc()
      next
    }
    
    layer_score_df <- data.frame(
      cell_id = layer_cells,
      stringsAsFactors = FALSE
    )
    
    rownames(layer_score_df) <- layer_cells
    
    for (module_name in names(gene_sets)) {
      genes <- gene_sets[[module_name]]
      
      genes_present <- intersect(genes, rownames(mat))
      genes_missing <- setdiff(genes, rownames(mat))
      
      message("    Module: ", module_name)
      message("      genes present: ", length(genes_present))
      message("      genes missing: ", length(genes_missing))
      
      raw_col <- paste0(module_name, "_score_raw")
      
      if (length(genes_present) < min_genes) {
        layer_score_df[[raw_col]] <- NA_real_
      } else {
        layer_score_df[[raw_col]] <- as.numeric(
          Matrix::colMeans(
            mat[genes_present, layer_cells, drop = FALSE]
          )
        )
      }
      
      coverage_key <- paste(layer_name, module_name, sep = "__")
      
      gene_coverage_list[[coverage_key]] <- data.frame(
        layer = layer_name,
        module = module_name,
        gene = c(genes_present, genes_missing),
        status = c(
          rep("present", length(genes_present)),
          rep("missing", length(genes_missing))
        ),
        stringsAsFactors = FALSE
      )
    }
    
    all_score_list[[layer_name]] <- layer_score_df
    
    rm(mat, layer_score_df)
    gc()
  }
  
  score_df <- dplyr::bind_rows(all_score_list)
  
  if (nrow(score_df) == 0) {
    stop("No module scores were calculated. Please check layer names and cell IDs.")
  }
  
  if (anyDuplicated(score_df$cell_id) > 0) {
    warning("Duplicated cell_id detected after layer-wise scoring. Keeping first occurrence.")
    score_df <- score_df %>%
      dplyr::distinct(cell_id, .keep_all = TRUE)
  }
  
  rownames(score_df) <- score_df$cell_id
  
  raw_cols <- grep("_score_raw$", colnames(score_df), value = TRUE)
  
  for (raw_col in raw_cols) {
    module_name <- sub("_score_raw$", "", raw_col)
    
    z_col <- paste0(module_name, "_score_z")
    scaled_col <- paste0(module_name, "_score_0_1")
    
    score_df[[z_col]] <- safe_zscore(score_df[[raw_col]])
    score_df[[scaled_col]] <- rescale01(score_df[[raw_col]])
  }
  
  gene_coverage <- dplyr::bind_rows(gene_coverage_list)
  
  return(
    list(
      score_df = score_df,
      gene_coverage = gene_coverage
    )
  )
}

# ------------------------------------------------------------
# 5.1 Fix axis_meta cell IDs before scoring
# ------------------------------------------------------------

message("Fixing axis_meta cell IDs...")

if ("CellBarcode_Identity" %in% colnames(axis_meta)) {
  axis_meta$cell_id <- as.character(axis_meta$CellBarcode_Identity)
  rownames(axis_meta) <- axis_meta$cell_id
} else if ("cell_id" %in% colnames(axis_meta)) {
  axis_meta$cell_id <- as.character(axis_meta$cell_id)
  rownames(axis_meta) <- axis_meta$cell_id
} else {
  stop("No valid cell ID column found in axis_meta. Need CellBarcode_Identity or cell_id.")
}

message("Checking cell ID overlap before module scoring:")

message("Cells in cells_keep: ", length(cells_keep))
message("Cells in Seurat object: ", ncol(seu))
message("Cells in axis_meta: ", nrow(axis_meta))

cells_use <- Reduce(
  intersect,
  list(
    cells_keep,
    colnames(seu),
    rownames(axis_meta)
  )
)

message("Overlapping cells among cells_keep, Seurat, and axis_meta: ", length(cells_use))

if (length(cells_use) < 1000) {
  stop("Too few overlapping cells before layer-wise module scoring. Please check cell IDs.")
}

axis_meta <- axis_meta[cells_use, , drop = FALSE]

# ------------------------------------------------------------
# 5.2 Calculate module scores layer by layer
# ------------------------------------------------------------

message("Calculating module scores layer by layer...")

score_res <- calculate_module_scores_by_layers(
  seu = seu,
  assay = assay_use,
  layer_prefix = layer_use,
  cells_use = cells_use,
  gene_sets = gene_sets,
  min_genes = min_genes_per_module
)

score_df <- score_res$score_df
gene_coverage <- score_res$gene_coverage

message("Module score dataframe dimension:")
print(dim(score_df))

message("Cells with module scores:")
print(length(score_df$cell_id))

message("Overlap between axis_meta and score_df:")
print(length(intersect(rownames(axis_meta), score_df$cell_id)))

cells_use <- intersect(rownames(axis_meta), score_df$cell_id)

if (length(cells_use) < 1000) {
  stop("Too few overlapping cells after layer-wise module scoring.")
}

axis_meta <- axis_meta[cells_use, , drop = FALSE]
score_df <- score_df[cells_use, , drop = FALSE]

message("Final cells used for axis v2:")
message(length(cells_use))

# ------------------------------------------------------------
# 5.3 Save module score outputs
# ------------------------------------------------------------

write.csv(
  gene_coverage,
  file.path(tab_dir, "01_module_gene_coverage.csv"),
  row.names = FALSE
)

write.csv(
  score_df,
  file.path(tab_dir, "02_cell_module_scores_raw_z_scaled.csv"),
  row.names = FALSE
)

saveRDS(
  score_df,
  file.path(tab_dir, "02_cell_module_scores_raw_z_scaled.rds")
)

rm(seu)
gc()
# ============================================================
# 7. Merge module scores with axis metadata
# ============================================================

score_df2 <- score_df

if (!"cell_id" %in% colnames(score_df2)) {
  score_df2$cell_id <- rownames(score_df2)
}

axis_meta2 <- axis_meta

if ("CellBarcode_Identity" %in% colnames(axis_meta2)) {
  axis_meta2$cell_id <- as.character(axis_meta2$CellBarcode_Identity)
} else if ("cell_id" %in% colnames(axis_meta2)) {
  axis_meta2$cell_id <- as.character(axis_meta2$cell_id)
} else {
  axis_meta2$cell_id <- rownames(axis_meta2)
}

score_df2$cell_id <- as.character(score_df2$cell_id)
axis_meta2$cell_id <- as.character(axis_meta2$cell_id)

meta_scores <- axis_meta2 %>%
  dplyr::left_join(score_df2, by = "cell_id")

rownames(meta_scores) <- meta_scores$cell_id

message("Merged metadata + module scores:")
print(dim(meta_scores))

message("Number of cells with axis_v1 score:")
print(sum(!is.na(meta_scores$cell_multicellular_progression_score)))

message("Number of cells with fibrosis_core score:")
print(sum(!is.na(meta_scores$fibrosis_core_score_raw)))

# ============================================================
# 8. Build composite biological scores
# ============================================================

get_scaled <- function(module) {
  col <- paste0(module, "_score_0_1")
  if (!col %in% colnames(meta_scores)) {
    return(rep(NA_real_, nrow(meta_scores)))
  }
  meta_scores[[col]]
}

meta_scores$fibrosis_composite_score_raw <- rowMeans(
  data.frame(
    fibrosis_core = get_scaled("fibrosis_core"),
    ECM_remodeling = get_scaled("ECM_remodeling"),
    collagen_formation = get_scaled("collagen_formation"),
    TGF_beta_signaling = get_scaled("TGF_beta_signaling"),
    myofibroblast_activation = get_scaled("myofibroblast_activation")
  ),
  na.rm = TRUE
)

meta_scores$epithelial_injury_composite_score_raw <- rowMeans(
  data.frame(
    epithelial_injury = get_scaled("epithelial_injury"),
    aberrant_basaloid = get_scaled("aberrant_basaloid"),
    AT2_stress_regeneration = get_scaled("AT2_stress_regeneration"),
    EMT_like = get_scaled("EMT_like")
  ),
  na.rm = TRUE
)

meta_scores$myeloid_inflammatory_composite_score_raw <- rowMeans(
  data.frame(
    profibrotic_macrophage = get_scaled("profibrotic_macrophage"),
    inflammatory_monocyte = get_scaled("inflammatory_monocyte"),
    antigen_presentation = get_scaled("antigen_presentation"),
    chemokine_inflammatory = get_scaled("chemokine_inflammatory")
  ),
  na.rm = TRUE
)

meta_scores$vascular_remodeling_composite_score_raw <- rowMeans(
  data.frame(
    angiogenesis = get_scaled("angiogenesis"),
    endothelial_activation = get_scaled("endothelial_activation"),
    hypoxia = get_scaled("hypoxia")
  ),
  na.rm = TRUE
)

meta_scores$stress_inflammation_composite_score_raw <- rowMeans(
  data.frame(
    hypoxia = get_scaled("hypoxia"),
    oxidative_stress = get_scaled("oxidative_stress"),
    interferon_response = get_scaled("interferon_response"),
    senescence_SASP = get_scaled("senescence_SASP"),
    apoptosis_stress = get_scaled("apoptosis_stress")
  ),
  na.rm = TRUE
)

meta_scores$immune_activation_composite_score_raw <- rowMeans(
  data.frame(
    cytotoxicity = get_scaled("cytotoxicity"),
    T_cell_activation = get_scaled("T_cell_activation"),
    T_cell_exhaustion = get_scaled("T_cell_exhaustion"),
    B_plasma_activation = get_scaled("B_plasma_activation")
  ),
  na.rm = TRUE
)

# Scale composites
composite_cols_raw <- grep("_composite_score_raw$", colnames(meta_scores), value = TRUE)

for (cc in composite_cols_raw) {
  new_col <- sub("_raw$", "", cc)
  meta_scores[[new_col]] <- rescale01(meta_scores[[cc]])
}

# ============================================================
# 9. Build progression axis v2
# ============================================================

if (!"cell_multicellular_progression_score" %in% colnames(meta_scores)) {
  stop("cell_multicellular_progression_score not found in axis metadata.")
}

meta_scores$injury_inflammation_composite_score <- rowMeans(
  data.frame(
    epithelial = meta_scores$epithelial_injury_composite_score,
    myeloid = meta_scores$myeloid_inflammatory_composite_score,
    stress = meta_scores$stress_inflammation_composite_score,
    vascular = meta_scores$vascular_remodeling_composite_score
  ),
  na.rm = TRUE
)

meta_scores$axis_v2_raw <- (
  w_axis_v1 * rescale01(meta_scores$cell_multicellular_progression_score) +
    w_fibrosis * rescale01(meta_scores$fibrosis_composite_score) +
    w_injury_inflammation * rescale01(meta_scores$injury_inflammation_composite_score)
)

meta_scores$axis_v2_score <- rescale01(meta_scores$axis_v2_raw)
meta_scores$axis_v2_stage <- make_stage(meta_scores$axis_v2_score)

# Orient axis v2: IPF should be higher than Control
if ("Disease_Identity" %in% colnames(meta_scores)) {
  mean_control <- mean(
    meta_scores$axis_v2_score[meta_scores$Disease_Identity == "Control"],
    na.rm = TRUE
  )
  
  mean_ipf <- mean(
    meta_scores$axis_v2_score[meta_scores$Disease_Identity == "IPF"],
    na.rm = TRUE
  )
  
  if (is.finite(mean_control) && is.finite(mean_ipf) && mean_ipf < mean_control) {
    meta_scores$axis_v2_score <- 1 - meta_scores$axis_v2_score
    meta_scores$axis_v2_raw <- -meta_scores$axis_v2_raw
    meta_scores$axis_v2_stage <- make_stage(meta_scores$axis_v2_score)
  }
}

# ============================================================
# 10. Output full metadata with scores
# ============================================================

write.csv(
  meta_scores,
  file.path(tab_dir, "03_cell_metadata_with_axis_v2_module_scores.csv"),
  row.names = TRUE
)

saveRDS(
  meta_scores,
  file.path(tab_dir, "03_cell_metadata_with_axis_v2_module_scores.rds")
)

# ============================================================
# 11. Correlation summaries
# ============================================================

score_cols_for_cor <- c(
  grep("_score_0_1$", colnames(meta_scores), value = TRUE),
  grep("_composite_score$", colnames(meta_scores), value = TRUE),
  "injury_inflammation_composite_score",
  "axis_v2_score"
)

score_cols_for_cor <- unique(score_cols_for_cor)
score_cols_for_cor <- score_cols_for_cor[score_cols_for_cor %in% colnames(meta_scores)]

axis_cor_overall <- data.frame(
  score_name = score_cols_for_cor,
  spearman_with_axis_v1 = sapply(
    score_cols_for_cor,
    function(cc) safe_cor(
      meta_scores$cell_multicellular_progression_score,
      meta_scores[[cc]],
      method = "spearman"
    )
  ),
  spearman_with_axis_v2 = sapply(
    score_cols_for_cor,
    function(cc) safe_cor(
      meta_scores$axis_v2_score,
      meta_scores[[cc]],
      method = "spearman"
    )
  )
) %>%
  arrange(desc(spearman_with_axis_v1))

write.csv(
  axis_cor_overall,
  file.path(tab_dir, "04_module_score_correlation_with_axis_overall.csv"),
  row.names = FALSE
)

# By cell type
if ("celltype_marker_based" %in% colnames(meta_scores)) {
  axis_cor_by_celltype <- meta_scores %>%
    group_by(celltype_marker_based) %>%
    summarise(
      n_cells = n(),
      across(
        all_of(score_cols_for_cor),
        ~ safe_cor(cell_multicellular_progression_score, .x, method = "spearman"),
        .names = "cor_axis_v1_{.col}"
      ),
      .groups = "drop"
    )
  
  write.csv(
    axis_cor_by_celltype,
    file.path(tab_dir, "05_module_score_correlation_with_axis_by_celltype.csv"),
    row.names = FALSE
  )
}

# By disease
if ("Disease_Identity" %in% colnames(meta_scores)) {
  axis_cor_by_disease <- meta_scores %>%
    group_by(Disease_Identity) %>%
    summarise(
      n_cells = n(),
      across(
        all_of(score_cols_for_cor),
        ~ safe_cor(cell_multicellular_progression_score, .x, method = "spearman"),
        .names = "cor_axis_v1_{.col}"
      ),
      .groups = "drop"
    )
  
  write.csv(
    axis_cor_by_disease,
    file.path(tab_dir, "06_module_score_correlation_with_axis_by_disease.csv"),
    row.names = FALSE
  )
}

# ============================================================
# 12. Summaries by disease / stage / cell type
# ============================================================

summary_by_disease <- meta_scores %>%
  group_by(Disease_Identity) %>%
  summarise(
    n_cells = n(),
    axis_v1_mean = mean(cell_multicellular_progression_score, na.rm = TRUE),
    axis_v2_mean = mean(axis_v2_score, na.rm = TRUE),
    fibrosis_mean = mean(fibrosis_composite_score, na.rm = TRUE),
    epithelial_injury_mean = mean(epithelial_injury_composite_score, na.rm = TRUE),
    myeloid_inflammatory_mean = mean(myeloid_inflammatory_composite_score, na.rm = TRUE),
    vascular_remodeling_mean = mean(vascular_remodeling_composite_score, na.rm = TRUE),
    stress_inflammation_mean = mean(stress_inflammation_composite_score, na.rm = TRUE),
    immune_activation_mean = mean(immune_activation_composite_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  summary_by_disease,
  file.path(tab_dir, "07_axis_v2_summary_by_disease.csv"),
  row.names = FALSE
)

summary_by_stage <- meta_scores %>%
  group_by(axis_v2_stage) %>%
  summarise(
    n_cells = n(),
    axis_v1_mean = mean(cell_multicellular_progression_score, na.rm = TRUE),
    axis_v2_mean = mean(axis_v2_score, na.rm = TRUE),
    fibrosis_mean = mean(fibrosis_composite_score, na.rm = TRUE),
    epithelial_injury_mean = mean(epithelial_injury_composite_score, na.rm = TRUE),
    myeloid_inflammatory_mean = mean(myeloid_inflammatory_composite_score, na.rm = TRUE),
    vascular_remodeling_mean = mean(vascular_remodeling_composite_score, na.rm = TRUE),
    stress_inflammation_mean = mean(stress_inflammation_composite_score, na.rm = TRUE),
    immune_activation_mean = mean(immune_activation_composite_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  summary_by_stage,
  file.path(tab_dir, "08_axis_v2_summary_by_stage.csv"),
  row.names = FALSE
)

summary_by_celltype <- meta_scores %>%
  group_by(celltype_marker_based, Disease_Identity) %>%
  summarise(
    n_cells = n(),
    axis_v1_mean = mean(cell_multicellular_progression_score, na.rm = TRUE),
    axis_v2_mean = mean(axis_v2_score, na.rm = TRUE),
    fibrosis_mean = mean(fibrosis_composite_score, na.rm = TRUE),
    epithelial_injury_mean = mean(epithelial_injury_composite_score, na.rm = TRUE),
    myeloid_inflammatory_mean = mean(myeloid_inflammatory_composite_score, na.rm = TRUE),
    vascular_remodeling_mean = mean(vascular_remodeling_composite_score, na.rm = TRUE),
    stress_inflammation_mean = mean(stress_inflammation_composite_score, na.rm = TRUE),
    immune_activation_mean = mean(immune_activation_composite_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  summary_by_celltype,
  file.path(tab_dir, "09_axis_v2_summary_by_celltype_disease.csv"),
  row.names = FALSE
)

# ============================================================
# 13. Plots
# ============================================================

# Axis v1 vs v2
p <- ggplot(
  meta_scores,
  aes(
    x = cell_multicellular_progression_score,
    y = axis_v2_score,
    color = Disease_Identity
  )
) +
  geom_point(size = 0.05, alpha = 0.4) +
  theme_classic(base_size = 12) +
  labs(
    title = "Axis v1 vs axis v2",
    x = "Axis v1: composition-informed score",
    y = "Axis v2: module-informed score",
    color = "Disease"
  )

ggsave(
  file.path(fig_dir, "01_axis_v1_vs_axis_v2.pdf"),
  p,
  width = 6,
  height = 5,
  limitsize = FALSE
)

# Axis v2 by disease
p <- ggplot(
  meta_scores,
  aes(x = Disease_Identity, y = axis_v2_score, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.05, alpha = 0.2) +
  theme_classic(base_size = 12) +
  labs(
    title = "Axis v2 by disease group",
    x = NULL,
    y = "Axis v2 score"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "02_axis_v2_by_disease.pdf"),
  p,
  width = 5,
  height = 4,
  limitsize = FALSE
)

# Composite score by axis v2 stage
plot_long_stage <- meta_scores %>%
  select(
    axis_v2_stage,
    fibrosis_composite_score,
    epithelial_injury_composite_score,
    myeloid_inflammatory_composite_score,
    vascular_remodeling_composite_score,
    stress_inflammation_composite_score,
    immune_activation_composite_score
  ) %>%
  pivot_longer(
    cols = -axis_v2_stage,
    names_to = "score",
    values_to = "value"
  )

p <- ggplot(
  plot_long_stage,
  aes(x = axis_v2_stage, y = value, fill = axis_v2_stage)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  facet_wrap(~ score, scales = "free_y") +
  theme_classic(base_size = 11) +
  labs(
    title = "Module composite scores across axis v2 stages",
    x = NULL,
    y = "Score"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  file.path(fig_dir, "03_composite_scores_by_axis_v2_stage.pdf"),
  p,
  width = 12,
  height = 7,
  limitsize = FALSE
)

# Heatmap-like summary by disease
summary_heat <- summary_by_disease %>%
  select(
    Disease_Identity,
    fibrosis_mean,
    epithelial_injury_mean,
    myeloid_inflammatory_mean,
    vascular_remodeling_mean,
    stress_inflammation_mean,
    immune_activation_mean
  ) %>%
  pivot_longer(
    cols = -Disease_Identity,
    names_to = "score",
    values_to = "mean_score"
  )

p <- ggplot(
  summary_heat,
  aes(x = Disease_Identity, y = score, fill = mean_score)
) +
  geom_tile() +
  theme_classic(base_size = 12) +
  labs(
    title = "Mean module composite scores by disease",
    x = NULL,
    y = NULL,
    fill = "Mean score"
  )

ggsave(
  file.path(fig_dir, "04_mean_composite_scores_by_disease_heatmap.pdf"),
  p,
  width = 6,
  height = 4
)

# Top correlations
top_cor <- axis_cor_overall %>%
  filter(!is.na(spearman_with_axis_v1)) %>%
  arrange(desc(abs(spearman_with_axis_v1))) %>%
  head(25)

p <- ggplot(
  top_cor,
  aes(
    x = reorder(score_name, spearman_with_axis_v1),
    y = spearman_with_axis_v1,
    fill = spearman_with_axis_v1 > 0
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 11) +
  labs(
    title = "Top module correlations with axis v1",
    x = NULL,
    y = "Spearman correlation"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "05_top_module_correlations_with_axis_v1.pdf"),
  p,
  width = 8,
  height = 7
)

# ============================================================
# 14. Save compact object
# ============================================================

axis_v2_object <- list(
  metadata_scores = meta_scores,
  gene_coverage = gene_coverage,
  axis_cor_overall = axis_cor_overall,
  summary_by_disease = summary_by_disease,
  summary_by_stage = summary_by_stage,
  summary_by_celltype = summary_by_celltype,
  parameters = list(
    w_axis_v1 = w_axis_v1,
    w_fibrosis = w_fibrosis,
    w_injury_inflammation = w_injury_inflammation,
    assay_use = assay_use,
    layer_use = layer_use,
    min_genes_per_module = min_genes_per_module
  )
)

saveRDS(
  axis_v2_object,
  file.path(out_dir, "axis_v2_module_fibrosis_pathway_object.rds")
)

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "03_cell_metadata_with_axis_v2_module_scores.rds"))
message("  ", file.path(tab_dir, "04_module_score_correlation_with_axis_overall.csv"))
message("  ", file.path(tab_dir, "07_axis_v2_summary_by_disease.csv"))
message("  ", file.path(tab_dir, "08_axis_v2_summary_by_stage.csv"))
message("  ", file.path(out_dir, "axis_v2_module_fibrosis_pathway_object.rds"))