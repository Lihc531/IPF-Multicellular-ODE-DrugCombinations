# 11_celltype_dynamic_gene_modules_axis_v2.R
# Cell-type-specific dynamic gene module identification along axis_v2_score
#
# Low-memory strategy:
# - does not JoinLayers
# - does not subset the full Seurat object
# - processes Seurat v5 data layers one by one
# - bins cells by axis_v2_score within each cell type
# - computes gene trends across progression bins
# - clusters dynamic genes into modules per cell type

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
# 0. Input / output
# ============================================================

seurat_rds <- file.path(project_dir, "ipf_annotated_singlet_seurat.rds")

axis_v2_rds <- file.path(
  project_dir,
  "results",
  "progression_axis_v2",
  "tables",
  "03_cell_metadata_with_axis_v2_module_scores.rds"
)

out_dir <- file.path(project_dir, "results", "b_dynamic_gene_modules_axis_v2")
tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")
rds_dir <- file.path(out_dir, "rds")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rds_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Parameters
# ============================================================

assay_use <- "RNA"
layer_prefix <- "data"

axis_col <- "axis_v2_score"
celltype_col <- "celltype_marker_based"
disease_col <- "Disease_Identity"
sample_col <- "sample_id"

# Main cell types to analyze.
# Keep names exactly as in your celltype_marker_based column.
target_celltypes <- c(
  "Fibroblast",
  "Epithelial",
  "AT2",
  "Macrophage",
  "Monocyte",
  "Dendritic",
  "Endothelial",
  "T_cell",
  "NK",
  "B_cell",
  "Plasma",
  "Mast",
  "Smooth_muscle"
)

# Number of axis bins within each cell type.
n_bins <- 20

# Max cells per cell type for trend analysis.
# This controls memory and speed.
max_cells_per_celltype <- 30000

# Gene filtering
min_cells_per_celltype <- 500
min_cells_per_bin <- 30
min_pct_expressed <- 0.03
min_bins_expressed <- 3

# Gene universe size.
# If VariableFeatures exist, use them; otherwise use expressed genes after basic filtering.
max_genes_use <- 4000

# Dynamic gene thresholds
min_abs_spearman <- 0.35
max_fdr <- 0.05
min_dynamic_range <- 0.10

# Minimum cells required for a gene-state row in the exported v1B dynamic_gene_csv.
min_cells_state_dynamic_gene_csv <- 10

# Module clustering
max_dynamic_genes_for_clustering <- 1500
min_dynamic_genes_for_clustering <- 30
k_modules_default <- 6

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

safe_zscore_vec <- function(x) {
  x <- as.numeric(x)
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) {
    return(rep(0, length(x)))
  }
  (x - m) / s
}


make_state_id <- function(df, cols) {
  missing_cols <- setdiff(cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(
      "Cannot make cell_state_id because columns are missing: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  apply(
    df[, cols, drop = FALSE],
    1,
    function(z) paste(as.character(z), collapse = "__")
  )
}


safe_mean <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

safe_median <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) {
    return(NA_real_)
  }
  median(x, na.rm = TRUE)
}

safe_fraction <- function(x) {
  mean(x, na.rm = TRUE)
}

safe_spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 5) {
    return(NA_real_)
  }
  suppressWarnings(cor(x[ok], y[ok], method = "spearman"))
}

safe_cor_test_p <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 5) {
    return(NA_real_)
  }
  suppressWarnings(cor.test(x[ok], y[ok], method = "spearman")$p.value)
}

get_layer_names <- function(seu, assay = "RNA", layer_prefix = "data") {
  all_layers <- Layers(seu[[assay]])
  
  layer_names <- all_layers[
    all_layers == layer_prefix |
      grepl(paste0("^", layer_prefix, "\\."), all_layers)
  ]
  
  if (length(layer_names) == 0) {
    stop("No layers found with prefix: ", layer_prefix)
  }
  
  layer_names
}

make_axis_bins <- function(meta_ct, axis_col, n_bins = 20) {
  score <- meta_ct[[axis_col]]
  
  # Quantile bins are more stable when score distribution is uneven.
  probs <- seq(0, 1, length.out = n_bins + 1)
  breaks <- unique(as.numeric(quantile(score, probs = probs, na.rm = TRUE)))
  
  if (length(breaks) < 4) {
    return(NULL)
  }
  
  meta_ct$axis_bin <- cut(
    score,
    breaks = breaks,
    include.lowest = TRUE,
    labels = FALSE
  )
  
  meta_ct <- meta_ct %>%
    filter(!is.na(axis_bin))
  
  bin_info <- meta_ct %>%
    group_by(axis_bin) %>%
    summarise(
      n_cells = n(),
      axis_mean = mean(.data[[axis_col]], na.rm = TRUE),
      axis_median = median(.data[[axis_col]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(axis_bin)
  
  list(
    meta_ct = meta_ct,
    bin_info = bin_info
  )
}

choose_gene_universe <- function(seu, assay = "RNA", max_genes = 4000) {
  DefaultAssay(seu) <- assay
  
  vf <- tryCatch(
    {
      VariableFeatures(seu)
    },
    error = function(e) character(0)
  )
  
  vf <- unique(vf)
  vf <- vf[vf %in% rownames(seu[[assay]])]
  
  if (length(vf) >= 500) {
    genes <- vf
  } else {
    genes <- rownames(seu[[assay]])
  }
  
  genes <- genes[
    !grepl("^MT-", genes) &
      !grepl("^RPL", genes) &
      !grepl("^RPS", genes) &
      !grepl("^MTRNR", genes)
  ]
  
  genes <- unique(genes)
  
  if (length(genes) > max_genes) {
    genes <- genes[seq_len(max_genes)]
  }
  
  genes
}

# Curated genes to force include even if not in variable features.
curated_dynamic_genes <- unique(c(
  "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
  "FN1", "POSTN", "THBS1", "SPARC", "CTGF", "SERPINE1",
  "ACTA2", "TAGLN", "MYH11", "CNN1", "PDGFRB",
  "DCN", "LUM", "DPT", "PI16",
  "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD7",
  "KRT8", "KRT18", "KRT19", "CLDN4", "MUC1", "KRT5", "KRT17", "TP63",
  "SFTPC", "SFTPA1", "SFTPA2", "ABCA3", "AGER", "PDPN", "CAV1",
  "SPP1", "TREM2", "APOE", "MERTK", "MARCO", "CD68",
  "C1QA", "C1QB", "C1QC", "LGALS3", "CHI3L1",
  "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "IL1B", "CXCL8",
  "CCL2", "CCL3", "CCL4", "CXCL9", "CXCL10", "CXCL11",
  "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74",
  "PECAM1", "VWF", "KDR", "FLT1", "ENG", "ICAM1", "VCAM1", "ACKR1",
  "NKG7", "GNLY", "PRF1", "GZMB", "CD3D", "CD3E", "TRAC",
  "MS4A1", "CD79A", "MZB1", "JCHAIN", "XBP1"
))

classify_trend <- function(rho, fdr, dyn_range, z_trend) {
  if (is.na(rho) || is.na(fdr) || is.na(dyn_range)) {
    return("unclassified")
  }
  
  if (fdr <= max_fdr && rho >= min_abs_spearman && dyn_range >= min_dynamic_range) {
    return("increasing")
  }
  
  if (fdr <= max_fdr && rho <= -min_abs_spearman && dyn_range >= min_dynamic_range) {
    return("decreasing")
  }
  
  # transient: peak in middle bins and sufficient range
  if (dyn_range >= min_dynamic_range && length(z_trend) >= 6) {
    peak_bin <- which.max(z_trend)
    if (peak_bin > 2 && peak_bin < (length(z_trend) - 1)) {
      return("transient_peak")
    }
  }
  
  return("weak_or_flat")
}

# ============================================================
# 3. Load data
# ============================================================

message("Reading Seurat object...")
seu <- readRDS(seurat_rds)

message("Reading axis v2 metadata...")
meta <- readRDS(axis_v2_rds)

if (!"cell_id" %in% colnames(meta)) {
  if ("CellBarcode_Identity" %in% colnames(meta)) {
    meta$cell_id <- as.character(meta$CellBarcode_Identity)
  } else {
    meta$cell_id <- rownames(meta)
  }
}

rownames(meta) <- meta$cell_id

required_cols <- c("cell_id", axis_col, celltype_col, disease_col, sample_col)
missing_cols <- setdiff(required_cols, colnames(meta))

if (length(missing_cols) > 0) {
  stop("Missing required columns in axis v2 metadata: ", paste(missing_cols, collapse = ", "))
}

meta <- meta %>%
  filter(
    !is.na(.data[[axis_col]]),
    !is.na(.data[[celltype_col]])
  )

meta$cell_id <- as.character(meta$cell_id)
rownames(meta) <- meta$cell_id

message("Metadata dimension:")
print(dim(meta))

message("Cell type distribution:")
print(table(meta[[celltype_col]], useNA = "ifany"))

# Use only cells present in Seurat
cells_overlap <- intersect(meta$cell_id, colnames(seu))

message("Cells in metadata: ", nrow(meta))
message("Cells in Seurat: ", ncol(seu))
message("Overlapping cells: ", length(cells_overlap))

if (length(cells_overlap) < 1000) {
  stop("Too few overlapping cells between axis metadata and Seurat object.")
}

meta <- meta[cells_overlap, , drop = FALSE]

# Gene universe
message("Choosing gene universe...")
gene_universe <- choose_gene_universe(
  seu = seu,
  assay = assay_use,
  max_genes = max_genes_use
)

gene_universe <- unique(c(gene_universe, curated_dynamic_genes))
gene_universe <- gene_universe[gene_universe %in% rownames(seu[[assay_use]])]

message("Genes used for dynamic analysis: ", length(gene_universe))

writeLines(
  gene_universe,
  con = file.path(tab_dir, "00_gene_universe_dynamic_module_analysis.txt")
)

layer_names <- get_layer_names(
  seu = seu,
  assay = assay_use,
  layer_prefix = layer_prefix
)

message("Detected layers:")
print(layer_names)

# ============================================================
# 4. Main function for one cell type
# ============================================================

analyze_one_celltype <- function(
    celltype_name,
    meta,
    seu,
    gene_universe,
    layer_names,
    assay = "RNA",
    axis_col = "axis_v2_score",
    n_bins = 20,
    max_cells = 30000,
    min_cells_per_celltype = 500,
    min_cells_per_bin = 30
) {
  message("============================================================")
  message("Analyzing cell type: ", celltype_name)
  message("============================================================")
  
  meta_ct <- meta %>%
    filter(.data[[celltype_col]] == celltype_name)
  
  if (nrow(meta_ct) < min_cells_per_celltype) {
    warning("Skipping ", celltype_name, ": too few cells: ", nrow(meta_ct))
    return(NULL)
  }
  
  # Downsample evenly across disease groups if too many cells.
  if (nrow(meta_ct) > max_cells) {
    message("Downsampling ", celltype_name, " from ", nrow(meta_ct), " to ", max_cells, " cells.")
    
    meta_ct <- meta_ct %>%
      group_by(.data[[disease_col]]) %>%
      mutate(group_n = n()) %>%
      ungroup()
    
    disease_levels <- unique(as.character(meta_ct[[disease_col]]))
    n_per_group <- ceiling(max_cells / length(disease_levels))
    
    meta_ct <- meta_ct %>%
      group_by(.data[[disease_col]]) %>%
      group_modify(function(.x, .y) {
        n_take <- min(n_per_group, nrow(.x))
        dplyr::slice_sample(.x, n = n_take, replace = FALSE)
      }) %>%
      ungroup()
    
    if (nrow(meta_ct) > max_cells) {
      meta_ct <- meta_ct %>%
        slice_sample(n = max_cells, replace = FALSE)
    }
  }
  
  bin_res <- make_axis_bins(meta_ct, axis_col = axis_col, n_bins = n_bins)
  
  if (is.null(bin_res)) {
    warning("Skipping ", celltype_name, ": unable to create bins.")
    return(NULL)
  }
  
  meta_ct <- bin_res$meta_ct
  bin_info <- bin_res$bin_info
  
  good_bins <- bin_info %>%
    filter(n_cells >= min_cells_per_bin) %>%
    pull(axis_bin)
  
  meta_ct <- meta_ct %>%
    filter(axis_bin %in% good_bins)
  
  bin_info <- bin_info %>%
    filter(axis_bin %in% good_bins)
  
  if (nrow(bin_info) < 5) {
    warning("Skipping ", celltype_name, ": fewer than 5 valid bins.")
    return(NULL)
  }
  
  bins <- sort(unique(meta_ct$axis_bin))
  bin_labels <- paste0("bin_", bins)
  
  cells_ct <- meta_ct$cell_id
  cell_to_bin <- setNames(meta_ct$axis_bin, meta_ct$cell_id)
  
  genes <- gene_universe
  
  expr_sum <- matrix(
    0,
    nrow = length(genes),
    ncol = length(bins),
    dimnames = list(genes, bin_labels)
  )
  
  detect_sum <- matrix(
    0,
    nrow = length(genes),
    ncol = length(bins),
    dimnames = list(genes, bin_labels)
  )
  
  n_cells_bin <- setNames(rep(0, length(bins)), bin_labels)
  
  # Layer-wise accumulation
  for (layer_name in layer_names) {
    message("  Layer: ", layer_name)
    
    mat <- LayerData(
      object = seu,
      assay = assay,
      layer = layer_name
    )
    
    layer_cells <- intersect(cells_ct, colnames(mat))
    
    message("    overlapping cells: ", length(layer_cells))
    
    if (length(layer_cells) == 0) {
      rm(mat)
      gc()
      next
    }
    
    genes_present <- intersect(genes, rownames(mat))
    
    if (length(genes_present) == 0) {
      rm(mat)
      gc()
      next
    }
    
    sub <- mat[genes_present, layer_cells, drop = FALSE]
    
    layer_bins <- cell_to_bin[layer_cells]
    
    for (bb in bins) {
      bb_cells <- names(layer_bins)[layer_bins == bb]
      
      if (length(bb_cells) == 0) {
        next
      }
      
      bb_label <- paste0("bin_", bb)
      
      expr_sum[genes_present, bb_label] <-
        expr_sum[genes_present, bb_label] +
        Matrix::rowSums(sub[, bb_cells, drop = FALSE])
      
      detect_sum[genes_present, bb_label] <-
        detect_sum[genes_present, bb_label] +
        Matrix::rowSums(sub[, bb_cells, drop = FALSE] > 0)
      
      n_cells_bin[bb_label] <- n_cells_bin[bb_label] + length(bb_cells)
    }
    
    rm(mat, sub)
    gc()
  }
  
  # Remove bins with no cells
  valid_bin_labels <- names(n_cells_bin)[n_cells_bin > 0]
  
  expr_sum <- expr_sum[, valid_bin_labels, drop = FALSE]
  detect_sum <- detect_sum[, valid_bin_labels, drop = FALSE]
  n_cells_bin <- n_cells_bin[valid_bin_labels]
  
  mean_expr <- sweep(expr_sum, 2, n_cells_bin, "/")
  pct_expr <- sweep(detect_sum, 2, n_cells_bin, "/")
  
  # Filter expressed genes
  keep_gene <- rowSums(pct_expr >= min_pct_expressed, na.rm = TRUE) >= min_bins_expressed
  mean_expr <- mean_expr[keep_gene, , drop = FALSE]
  pct_expr <- pct_expr[keep_gene, , drop = FALSE]
  
  if (nrow(mean_expr) < 50) {
    warning("Skipping ", celltype_name, ": too few expressed genes after filtering.")
    return(NULL)
  }
  
  # Bin axis values
  bin_axis <- bin_info %>%
    mutate(bin_label = paste0("bin_", axis_bin)) %>%
    filter(bin_label %in% colnames(mean_expr)) %>%
    arrange(match(bin_label, colnames(mean_expr)))
  
  axis_values <- bin_axis$axis_mean
  
  # Gene trend statistics
  gene_stats <- data.frame(
    gene = rownames(mean_expr),
    stringsAsFactors = FALSE
  )
  
  gene_stats$mean_expression <- rowMeans(mean_expr, na.rm = TRUE)
  gene_stats$max_pct_expressed <- apply(pct_expr, 1, max, na.rm = TRUE)
  gene_stats$dynamic_range <- apply(mean_expr, 1, function(x) max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  
  gene_stats$spearman_axis <- apply(
    mean_expr,
    1,
    function(x) safe_spearman(axis_values, x)
  )
  
  gene_stats$p_value <- apply(
    mean_expr,
    1,
    function(x) safe_cor_test_p(axis_values, x)
  )
  
  gene_stats$fdr <- p.adjust(gene_stats$p_value, method = "BH")
  
  # Linear slope for reference
  gene_stats$linear_slope <- apply(
    mean_expr,
    1,
    function(x) {
      ok <- is.finite(axis_values) & is.finite(x)
      if (sum(ok) < 5) {
        return(NA_real_)
      }
      coef(lm(x[ok] ~ axis_values[ok]))[2]
    }
  )
  
  # Z-scaled trend matrix
  trend_z <- t(apply(mean_expr, 1, safe_zscore_vec))
  rownames(trend_z) <- rownames(mean_expr)
  colnames(trend_z) <- colnames(mean_expr)
  
  gene_stats$trend_class <- sapply(
    seq_len(nrow(gene_stats)),
    function(i) {
      gene <- gene_stats$gene[i]
      classify_trend(
        rho = gene_stats$spearman_axis[i],
        fdr = gene_stats$fdr[i],
        dyn_range = gene_stats$dynamic_range[i],
        z_trend = trend_z[gene, ]
      )
    }
  )
  
  gene_stats <- gene_stats %>%
    mutate(
      abs_spearman_axis = abs(spearman_axis),
      dynamic_score = abs_spearman_axis * rescale01(dynamic_range)
    ) %>%
    arrange(desc(dynamic_score))
  
  dynamic_genes <- gene_stats %>%
    filter(
      trend_class %in% c("increasing", "decreasing", "transient_peak"),
      fdr <= max_fdr,
      abs_spearman_axis >= min_abs_spearman,
      dynamic_range >= min_dynamic_range
    )
  
  # If too strict, keep top genes for exploratory modules
  if (nrow(dynamic_genes) < min_dynamic_genes_for_clustering) {
    gene_stats_valid <- gene_stats %>%
      filter(!is.na(spearman_axis)) %>%
      arrange(desc(dynamic_score))
    
    top_n_dynamic <- min(200, nrow(gene_stats_valid))
    
    dynamic_genes <- gene_stats_valid %>%
      head(top_n_dynamic)
  }
  
  # Limit genes for clustering
  dynamic_genes_for_cluster_all <- dynamic_genes %>%
    arrange(desc(dynamic_score))
  
  top_n_cluster <- min(
    max_dynamic_genes_for_clustering,
    nrow(dynamic_genes_for_cluster_all)
  )
  
  dynamic_genes_for_cluster <- dynamic_genes_for_cluster_all %>%
    head(top_n_cluster)
  
  module_assignments <- data.frame()
  module_trends <- data.frame()
  
  if (nrow(dynamic_genes_for_cluster) >= min_dynamic_genes_for_clustering) {
    genes_cluster <- dynamic_genes_for_cluster$gene
    
    mat_cluster <- trend_z[genes_cluster, , drop = FALSE]
    
    k_use <- min(k_modules_default, max(2, floor(nrow(mat_cluster) / 20)))
    
    set.seed(123)
    km <- kmeans(mat_cluster, centers = k_use, nstart = 25)
    
    module_assignments <- data.frame(
      celltype = celltype_name,
      gene = rownames(mat_cluster),
      dynamic_module = paste0(celltype_name, "_M", km$cluster),
      stringsAsFactors = FALSE
    ) %>%
      left_join(gene_stats, by = "gene")
    
    module_trends <- module_assignments %>%
      select(gene, dynamic_module) %>%
      left_join(
        as.data.frame(trend_z) %>%
          rownames_to_column("gene"),
        by = "gene"
      ) %>%
      pivot_longer(
        cols = starts_with("bin_"),
        names_to = "axis_bin",
        values_to = "z_expression"
      ) %>%
      group_by(dynamic_module, axis_bin) %>%
      summarise(
        mean_z_expression = mean(z_expression, na.rm = TRUE),
        n_genes = n(),
        .groups = "drop"
      ) %>%
      mutate(celltype = celltype_name)
  }
  
  # Output prefixes
  safe_name <- gsub("[^A-Za-z0-9_]+", "_", celltype_name)
  
  write.csv(
    bin_info,
    file.path(tab_dir, paste0(safe_name, "_01_axis_bin_info.csv")),
    row.names = FALSE
  )
  
  write.csv(
    gene_stats,
    file.path(tab_dir, paste0(safe_name, "_02_gene_axis_trend_statistics.csv")),
    row.names = FALSE
  )
  
  write.csv(
    dynamic_genes,
    file.path(tab_dir, paste0(safe_name, "_03_dynamic_genes.csv")),
    row.names = FALSE
  )
  
  write.csv(
    as.data.frame(mean_expr) %>% rownames_to_column("gene"),
    file.path(tab_dir, paste0(safe_name, "_04_gene_mean_expression_by_axis_bin.csv")),
    row.names = FALSE
  )
  
  write.csv(
    as.data.frame(trend_z) %>% rownames_to_column("gene"),
    file.path(tab_dir, paste0(safe_name, "_05_gene_ztrend_by_axis_bin.csv")),
    row.names = FALSE
  )
  
  if (nrow(module_assignments) > 0) {
    write.csv(
      module_assignments,
      file.path(tab_dir, paste0(safe_name, "_06_dynamic_module_gene_assignments.csv")),
      row.names = FALSE
    )
    
    write.csv(
      module_trends,
      file.path(tab_dir, paste0(safe_name, "_07_dynamic_module_trends.csv")),
      row.names = FALSE
    )
  }
  
  # Plots
  
  # Top increasing/decreasing genes
  top_plot_genes_all <- gene_stats %>%
    filter(trend_class %in% c("increasing", "decreasing")) %>%
    arrange(desc(dynamic_score))
  
  top_n_plot <- min(30, nrow(top_plot_genes_all))
  
  top_plot_genes <- top_plot_genes_all %>%
    head(top_n_plot)
  
  if (nrow(top_plot_genes) > 0) {
    plot_trend <- as.data.frame(trend_z[top_plot_genes$gene, , drop = FALSE]) %>%
      rownames_to_column("gene") %>%
      pivot_longer(
        cols = starts_with("bin_"),
        names_to = "axis_bin",
        values_to = "z_expression"
      ) %>%
      left_join(
        gene_stats %>% select(gene, trend_class),
        by = "gene"
      )
    
    p <- ggplot(
      plot_trend,
      aes(x = axis_bin, y = z_expression, group = gene, color = trend_class)
    ) +
      geom_line(alpha = 0.6) +
      theme_classic(base_size = 11) +
      labs(
        title = paste0(celltype_name, ": top dynamic gene trends"),
        x = "Axis v2 bin",
        y = "Z-scored mean expression",
        color = "Trend"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave(
      file.path(fig_dir, paste0(safe_name, "_01_top_dynamic_gene_trends.pdf")),
      p,
      width = 8,
      height = 5,
      limitsize = FALSE
    )
  }
  
  # Module trends
  if (nrow(module_trends) > 0) {
    p <- ggplot(
      module_trends,
      aes(x = axis_bin, y = mean_z_expression, group = dynamic_module, color = dynamic_module)
    ) +
      geom_line(linewidth = 1) +
      geom_point(size = 1.5) +
      theme_classic(base_size = 11) +
      labs(
        title = paste0(celltype_name, ": dynamic module trends"),
        x = "Axis v2 bin",
        y = "Mean module z-expression",
        color = "Module"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave(
      file.path(fig_dir, paste0(safe_name, "_02_dynamic_module_trends.pdf")),
      p,
      width = 8,
      height = 5,
      limitsize = FALSE
    )
  }
  
  result <- list(
    celltype = celltype_name,
    bin_info = bin_info,
    gene_stats = gene_stats,
    dynamic_genes = dynamic_genes,
    mean_expr_by_bin = mean_expr,
    pct_expr_by_bin = pct_expr,
    trend_z = trend_z,
    module_assignments = module_assignments,
    module_trends = module_trends
  )
  
  saveRDS(
    result,
    file.path(rds_dir, paste0(safe_name, "_dynamic_module_result.rds"))
  )
  
  message("Finished cell type: ", celltype_name)
  
  return(result)
}

# ============================================================
# 5. Run analysis for target cell types
# ============================================================

available_celltypes <- sort(unique(as.character(meta[[celltype_col]])))
target_celltypes <- intersect(target_celltypes, available_celltypes)

message("Target cell types:")
print(target_celltypes)

all_results <- list()
summary_list <- list()

for (ct in target_celltypes) {
  res <- analyze_one_celltype(
    celltype_name = ct,
    meta = meta,
    seu = seu,
    gene_universe = gene_universe,
    layer_names = layer_names,
    assay = assay_use,
    axis_col = axis_col,
    n_bins = n_bins,
    max_cells = max_cells_per_celltype,
    min_cells_per_celltype = min_cells_per_celltype,
    min_cells_per_bin = min_cells_per_bin
  )
  
  if (!is.null(res)) {
    all_results[[ct]] <- res
    
    summary_list[[ct]] <- data.frame(
      celltype = ct,
      n_bins = nrow(res$bin_info),
      n_genes_tested = nrow(res$gene_stats),
      n_dynamic_genes = nrow(res$dynamic_genes),
      n_increasing = sum(res$gene_stats$trend_class == "increasing", na.rm = TRUE),
      n_decreasing = sum(res$gene_stats$trend_class == "decreasing", na.rm = TRUE),
      n_transient_peak = sum(res$gene_stats$trend_class == "transient_peak", na.rm = TRUE),
      n_modules = ifelse(nrow(res$module_assignments) > 0,
                         length(unique(res$module_assignments$dynamic_module)),
                         0),
      stringsAsFactors = FALSE
    )
  }
  
  gc()
}

summary_df <- bind_rows(summary_list)

write.csv(
  summary_df,
  file.path(tab_dir, "00_dynamic_module_summary_by_celltype.csv"),
  row.names = FALSE
)

saveRDS(
  all_results,
  file.path(out_dir, "all_celltype_dynamic_module_results.rds")
)

# ============================================================
# 6. Combined dynamic gene table
# ============================================================

combined_dynamic_genes <- bind_rows(
  lapply(names(all_results), function(ct) {
    df <- all_results[[ct]]$dynamic_genes
    if (is.null(df) || nrow(df) == 0) {
      return(NULL)
    }
    df$celltype <- ct
    df
  })
)

if (!is.null(combined_dynamic_genes) && nrow(combined_dynamic_genes) > 0) {
  combined_dynamic_genes <- combined_dynamic_genes %>%
    relocate(celltype, gene)
  
  write.csv(
    combined_dynamic_genes,
    file.path(tab_dir, "00_combined_dynamic_genes_all_celltypes.csv"),
    row.names = FALSE
  )
}

combined_module_assignments <- bind_rows(
  lapply(names(all_results), function(ct) {
    df <- all_results[[ct]]$module_assignments
    if (is.null(df) || nrow(df) == 0) {
      return(NULL)
    }
    df
  })
)

if (!is.null(combined_module_assignments) && nrow(combined_module_assignments) > 0) {
  write.csv(
    combined_module_assignments,
    file.path(tab_dir, "00_combined_dynamic_module_gene_assignments.csv"),
    row.names = FALSE
  )
}


# ============================================================
# 6B. Build true v1B ODE-matched dynamic_gene_csv
#     gene x sample x celltype x cluster x axis_v2_stage
# ============================================================

message("Building true v1B ODE-matched dynamic_gene_csv...")

# This output is designed for the Python PPI-informed driver refinement pipeline.
# It uses the same cell_state_id structure as the v1B state velocity matrix:
#   sample_id__celltype_marker_based__cluster_default__axis_v2_stage
#
# Unlike a simple gene x celltype table, this section recalculates avg_expr and
# pct_expr directly from the Seurat expression layers for each exact v1B state.

min_cells_state_dynamic_gene_csv <- 10

required_dynamic_state_cols <- c(
  "cell_id",
  sample_col,
  disease_col,
  "Subject_Identity",
  "Library_Identity",
  celltype_col,
  "CellType_Category",
  "cluster_default",
  "axis_v2_stage"
)

missing_dynamic_state_cols <- setdiff(required_dynamic_state_cols, colnames(meta))

if (length(missing_dynamic_state_cols) > 0) {
  warning(
    "Skipping true dynamic_gene_csv because required columns are missing: ",
    paste(missing_dynamic_state_cols, collapse = ", ")
  )
} else if (is.null(combined_dynamic_genes) || nrow(combined_dynamic_genes) == 0) {
  warning("Skipping true dynamic_gene_csv because combined_dynamic_genes is empty.")
} else {

  stage_to_numeric <- function(x) {
    x <- as.character(x)
    out <- suppressWarnings(as.numeric(gsub(".*Stage[_ ]*([0-9]+).*", "\\1", x)))
    if (all(is.na(out))) {
      ux <- sort(unique(x))
      out <- match(x, ux)
    }
    out
  }

  disease_to_numeric <- function(x) {
    x <- as.character(x)
    preferred <- c("Control", "COPD", "IPF")
    if (all(x %in% preferred)) {
      return(match(x, preferred) - 1)
    }
    ux <- sort(unique(x))
    match(x, ux) - 1
  }

  compute_module_membership <- function(all_results) {
    bind_rows(
      lapply(names(all_results), function(ct) {
        res <- all_results[[ct]]

        ma <- res$module_assignments
        tz <- res$trend_z
        mt <- res$module_trends

        if (is.null(ma) || nrow(ma) == 0 || is.null(tz) || nrow(tz) == 0 ||
            is.null(mt) || nrow(mt) == 0) {
          return(NULL)
        }

        mt_wide <- mt %>%
          select(dynamic_module, axis_bin, mean_z_expression) %>%
          pivot_wider(
            names_from = axis_bin,
            values_from = mean_z_expression
          )

        out <- ma %>%
          select(celltype, gene, dynamic_module) %>%
          distinct() %>%
          rowwise() %>%
          mutate(
            module_membership_score = {
              gene_use <- gene
              mod_use <- dynamic_module
              if (!gene_use %in% rownames(tz) || !mod_use %in% mt_wide$dynamic_module) {
                NA_real_
              } else {
                mod_vec <- mt_wide %>%
                  filter(dynamic_module == mod_use) %>%
                  select(starts_with("bin_")) %>%
                  as.numeric()

                gene_vec <- as.numeric(tz[gene_use, names(mt_wide)[grepl("^bin_", names(mt_wide))], drop = TRUE])

                ok <- is.finite(gene_vec) & is.finite(mod_vec)
                if (sum(ok) < 3) {
                  NA_real_
                } else {
                  # Convert correlation from [-1, 1] to [0, 1].
                  (suppressWarnings(cor(gene_vec[ok], mod_vec[ok], method = "spearman")) + 1) / 2
                }
              }
            }
          ) %>%
          ungroup()

        out
      })
    )
  }

  module_membership_tbl <- compute_module_membership(all_results)

  dynamic_gene_seed <- combined_dynamic_genes %>%
    mutate(
      celltype = as.character(celltype),
      gene = as.character(gene)
    )

  if (!is.null(combined_module_assignments) && nrow(combined_module_assignments) > 0) {
    module_seed <- combined_module_assignments %>%
      mutate(
        celltype = as.character(celltype),
        gene = as.character(gene)
      ) %>%
      select(celltype, gene, dynamic_module) %>%
      distinct()

    dynamic_gene_seed <- dynamic_gene_seed %>%
      left_join(module_seed, by = c("celltype", "gene"))
  } else {
    dynamic_gene_seed$dynamic_module <- NA_character_
  }

  if (!is.null(module_membership_tbl) && nrow(module_membership_tbl) > 0) {
    dynamic_gene_seed <- dynamic_gene_seed %>%
      left_join(
        module_membership_tbl %>%
          select(celltype, gene, dynamic_module, module_membership_score),
        by = c("celltype", "gene", "dynamic_module")
      )
  } else {
    dynamic_gene_seed$module_membership_score <- NA_real_
  }

  dynamic_gene_seed <- dynamic_gene_seed %>%
    mutate(
      module_membership_score = ifelse(
        is.na(module_membership_score),
        rescale01(dynamic_score),
        module_membership_score
      ),
      dynamic_trend_score = rescale01(dynamic_score)
    ) %>%
    select(
      celltype,
      gene,
      dynamic_module,
      dynamic_trend_score,
      module_membership_score,
      mean_expression,
      max_pct_expressed,
      spearman_axis,
      abs_spearman_axis,
      fdr,
      dynamic_range,
      trend_class,
      everything()
    ) %>%
    distinct(celltype, gene, .keep_all = TRUE)

  genes_export <- sort(unique(dynamic_gene_seed$gene))

  meta_dynamic <- meta %>%
    filter(.data[[celltype_col]] %in% unique(dynamic_gene_seed$celltype)) %>%
    mutate(
      cell_state_id = make_state_id(
        .,
        c(sample_col, celltype_col, "cluster_default", "axis_v2_stage")
      ),
      axis_stage_numeric = stage_to_numeric(axis_v2_stage),
      disease_numeric = disease_to_numeric(.data[[disease_col]])
    )

  state_info <- meta_dynamic %>%
    group_by(
      cell_state_id,
      across(all_of(c(
        sample_col,
        disease_col,
        "Subject_Identity",
        "Library_Identity",
        "CellType_Category",
        celltype_col,
        "cluster_default",
        "axis_v2_stage"
      )))
    ) %>%
    summarise(
      n_cells_state = n(),
      axis_stage_numeric = safe_mean(axis_stage_numeric),
      disease_numeric = safe_mean(disease_numeric),
      mean_axis_v2_score = safe_mean(.data[[axis_col]]),
      .groups = "drop"
    ) %>%
    filter(n_cells_state >= min_cells_state_dynamic_gene_csv) %>%
    arrange(
      .data[[disease_col]],
      .data[[sample_col]],
      .data[[celltype_col]],
      cluster_default,
      axis_v2_stage
    )

  if (nrow(state_info) == 0) {
    warning("No v1B states pass min_cells_state_dynamic_gene_csv.")
  } else {

    state_ids <- state_info$cell_state_id
    cell_to_state <- setNames(meta_dynamic$cell_state_id, meta_dynamic$cell_id)
    cells_keep <- names(cell_to_state)[cell_to_state %in% state_ids]

    expr_sum_state <- matrix(
      0,
      nrow = length(genes_export),
      ncol = length(state_ids),
      dimnames = list(genes_export, state_ids)
    )

    detect_sum_state <- matrix(
      0,
      nrow = length(genes_export),
      ncol = length(state_ids),
      dimnames = list(genes_export, state_ids)
    )

    for (layer_name in layer_names) {
      message("  dynamic_gene_csv layer: ", layer_name)

      mat <- LayerData(
        object = seu,
        assay = assay_use,
        layer = layer_name
      )

      layer_cells <- intersect(cells_keep, colnames(mat))
      genes_present <- intersect(genes_export, rownames(mat))

      if (length(layer_cells) == 0 || length(genes_present) == 0) {
        rm(mat)
        gc()
        next
      }

      sub <- mat[genes_present, layer_cells, drop = FALSE]
      layer_states <- cell_to_state[layer_cells]
      states_present <- intersect(unique(layer_states), state_ids)

      for (sid in states_present) {
        sid_cells <- names(layer_states)[layer_states == sid]
        if (length(sid_cells) == 0) {
          next
        }

        expr_sum_state[genes_present, sid] <-
          expr_sum_state[genes_present, sid] +
          Matrix::rowSums(sub[, sid_cells, drop = FALSE])

        detect_sum_state[genes_present, sid] <-
          detect_sum_state[genes_present, sid] +
          Matrix::rowSums(sub[, sid_cells, drop = FALSE] > 0)
      }

      rm(mat, sub)
      gc()
    }

    n_cells_named <- setNames(state_info$n_cells_state, state_info$cell_state_id)

    avg_expr_state <- sweep(expr_sum_state, 2, n_cells_named[colnames(expr_sum_state)], "/")
    pct_expr_state <- sweep(detect_sum_state, 2, n_cells_named[colnames(detect_sum_state)], "/")

    dynamic_gene_state_list <- lapply(unique(dynamic_gene_seed$celltype), function(ct) {
      genes_ct <- dynamic_gene_seed %>%
        filter(celltype == ct) %>%
        pull(gene) %>%
        unique()

      states_ct <- state_info %>%
        filter(.data[[celltype_col]] == ct) %>%
        pull(cell_state_id)

      genes_ct <- intersect(genes_ct, rownames(avg_expr_state))
      states_ct <- intersect(states_ct, colnames(avg_expr_state))

      if (length(genes_ct) == 0 || length(states_ct) == 0) {
        return(NULL)
      }

      expand.grid(
        gene = genes_ct,
        cell_state_id = states_ct,
        stringsAsFactors = FALSE
      ) %>%
        mutate(
          avg_expr = avg_expr_state[cbind(gene, cell_state_id)],
          pct_expr = pct_expr_state[cbind(gene, cell_state_id)]
        )
    })

    dynamic_gene_state <- bind_rows(dynamic_gene_state_list)

    dynamic_gene_state <- dynamic_gene_state %>%
      left_join(state_info, by = "cell_state_id") %>%
      rename(
        celltype_marker_based = all_of(celltype_col),
        sample_id = all_of(sample_col),
        Disease_Identity = all_of(disease_col)
      ) %>%
      left_join(
        dynamic_gene_seed,
        by = c("gene", "celltype_marker_based" = "celltype")
      )

    gene_state_scores <- dynamic_gene_state %>%
      group_by(celltype_marker_based, gene) %>%
      summarise(
        disease_stage_association_raw = safe_spearman(disease_numeric, avg_expr),
        axis_stage_association_raw = safe_spearman(axis_stage_numeric, avg_expr),
        patient_consistency_score = {
          rho_global <- safe_spearman(axis_stage_numeric, avg_expr)
          subj_rho <- tapply(
            seq_along(avg_expr),
            Subject_Identity,
            function(ii) {
              safe_spearman(axis_stage_numeric[ii], avg_expr[ii])
            }
          )
          subj_rho <- as.numeric(subj_rho)
          subj_rho <- subj_rho[is.finite(subj_rho)]
          if (!is.finite(rho_global) || length(subj_rho) == 0) {
            NA_real_
          } else if (rho_global >= 0) {
            mean(subj_rho >= 0)
          } else {
            mean(subj_rho <= 0)
          }
        },
        n_states_for_gene = n_distinct(cell_state_id),
        n_subjects_for_gene = n_distinct(Subject_Identity),
        .groups = "drop"
      ) %>%
      mutate(
        disease_stage_association_score = abs(disease_stage_association_raw),
        neural_ode_direction_score = abs(axis_stage_association_raw)
      )

    dynamic_gene_csv <- dynamic_gene_state %>%
      left_join(
        gene_state_scores,
        by = c("celltype_marker_based", "gene")
      ) %>%
      mutate(
        patient_consistency_score = ifelse(
          is.na(patient_consistency_score),
          0,
          patient_consistency_score
        ),
        disease_stage_association_score = ifelse(
          is.na(disease_stage_association_score),
          0,
          disease_stage_association_score
        ),
        neural_ode_direction_score = ifelse(
          is.na(neural_ode_direction_score),
          0,
          neural_ode_direction_score
        ),
        pct_expr = ifelse(is.na(pct_expr), 0, pct_expr),
        avg_expr = ifelse(is.na(avg_expr), 0, avg_expr)
      ) %>%
      select(
        gene,
        cell_state_id,
        celltype_marker_based,
        cluster_default,
        axis_v2_stage,
        sample_id,
        Disease_Identity,
        Subject_Identity,
        Library_Identity,
        CellType_Category,
        dynamic_module,
        trend_class,
        dynamic_trend_score,
        module_membership_score,
        avg_expr,
        pct_expr,
        disease_stage_association_score,
        disease_stage_association_raw,
        neural_ode_direction_score,
        axis_stage_association_raw,
        patient_consistency_score,
        n_cells_state,
        n_states_for_gene,
        n_subjects_for_gene,
        spearman_axis,
        abs_spearman_axis,
        fdr,
        dynamic_range,
        mean_expression,
        max_pct_expressed,
        mean_axis_v2_score
      ) %>%
      arrange(
        celltype_marker_based,
        gene,
        sample_id,
        cluster_default,
        axis_v2_stage
      )

    write.csv(
      dynamic_gene_csv,
      file.path(tab_dir, "12_dynamic_gene_csv_v1B_ode_matched_true.csv"),
      row.names = FALSE
    )

    dynamic_gene_csv_qc <- data.frame(
      n_rows = nrow(dynamic_gene_csv),
      n_unique_genes = length(unique(dynamic_gene_csv$gene)),
      n_unique_cell_states = length(unique(dynamic_gene_csv$cell_state_id)),
      n_unique_celltypes = length(unique(dynamic_gene_csv$celltype_marker_based)),
      median_states_per_gene = median(gene_state_scores$n_states_for_gene, na.rm = TRUE),
      median_subjects_per_gene = median(gene_state_scores$n_subjects_for_gene, na.rm = TRUE),
      pct_rows_avg_expr_positive = mean(dynamic_gene_csv$avg_expr > 0, na.rm = TRUE),
      pct_rows_pct_expr_positive = mean(dynamic_gene_csv$pct_expr > 0, na.rm = TRUE),
      min_cells_state_dynamic_gene_csv = min_cells_state_dynamic_gene_csv,
      stringsAsFactors = FALSE
    )

    write.csv(
      dynamic_gene_csv_qc,
      file.path(tab_dir, "13_dynamic_gene_csv_v1B_ode_matched_true_qc.csv"),
      row.names = FALSE
    )

    message("Wrote true dynamic_gene_csv:")
    message("  ", file.path(tab_dir, "12_dynamic_gene_csv_v1B_ode_matched_true.csv"))
    message("  ", file.path(tab_dir, "13_dynamic_gene_csv_v1B_ode_matched_true_qc.csv"))
  }
}


# ============================================================
# 7. Summary plots
# ============================================================

if (nrow(summary_df) > 0) {
  p <- ggplot(
    summary_df,
    aes(x = reorder(celltype, n_dynamic_genes), y = n_dynamic_genes)
  ) +
    geom_col(width = 0.75) +
    coord_flip() +
    theme_classic(base_size = 12) +
    labs(
      title = "Number of dynamic genes by cell type",
      x = NULL,
      y = "Dynamic genes"
    )
  
  ggsave(
    file.path(fig_dir, "00_dynamic_genes_count_by_celltype.pdf"),
    p,
    width = 7,
    height = 5,
    limitsize = FALSE
  )
  
  summary_long <- summary_df %>%
    select(celltype, n_increasing, n_decreasing, n_transient_peak) %>%
    pivot_longer(
      cols = starts_with("n_"),
      names_to = "trend_type",
      values_to = "n_genes"
    )
  
  p <- ggplot(
    summary_long,
    aes(x = reorder(celltype, n_genes), y = n_genes, fill = trend_type)
  ) +
    geom_col(width = 0.75) +
    coord_flip() +
    theme_classic(base_size = 12) +
    labs(
      title = "Dynamic trend classes by cell type",
      x = NULL,
      y = "Number of genes",
      fill = "Trend class"
    )
  
  ggsave(
    file.path(fig_dir, "00_dynamic_trend_classes_by_celltype.pdf"),
    p,
    width = 8,
    height = 5,
    limitsize = FALSE
  )
}

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "00_dynamic_module_summary_by_celltype.csv"))
message("  ", file.path(tab_dir, "00_combined_dynamic_genes_all_celltypes.csv"))
message("  ", file.path(tab_dir, "00_combined_dynamic_module_gene_assignments.csv"))
message("  ", file.path(out_dir, "all_celltype_dynamic_module_results.rds"))