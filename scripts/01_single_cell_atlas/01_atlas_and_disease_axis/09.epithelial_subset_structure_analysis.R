# 09_epithelial_subset_structure_analysis.R
# Epithelial compartment fine-structure analysis for IPF / COPD / Control
# Low-memory version:
#   1. Use light metadata to define epithelial cells
#   2. Subset full Seurat object only to epithelial cells
#   3. Keep limited features: HVGs + epithelial marker genes
#   4. Re-run PCA / UMAP / clustering within epithelial compartment
#   5. Score epithelial injury / AT1 / AT2 / basal / aberrant basaloid / EMT-like modules
#   6. Export disease composition, sample-level fractions, module scores, and markers

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(readr)
  library(forcats)
  library(future)
})

plan("sequential")
options(future.globals.maxSize = 30 * 1024^3)

set.seed(123)

project_dir <- getwd()

out_dir <- file.path(project_dir, "results", "epithelial_subset")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
obj_dir <- file.path(out_dir, "objects")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(obj_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Load lightweight nomultiplet metadata
# ============================================================

light <- readRDS(file.path(project_dir, "ipf_nomultiplet_light_metadata_embeddings.rds"))

meta <- light$metadata
umap_global <- light$umap

message("Nomultiplet metadata:")
print(dim(meta))
print(table(meta$CellType_Category, useNA = "ifany"))
print(table(meta$celltype_marker_based, useNA = "ifany"))

required_cols <- c(
  "sample_id",
  "Disease_Identity",
  "CellType_Category",
  "celltype_marker_based",
  "cluster_default"
)

missing_cols <- setdiff(required_cols, colnames(meta))

if (length(missing_cols) > 0) {
  stop("Missing required metadata columns: ", paste(missing_cols, collapse = ", "))
}

meta$Disease_Identity <- factor(
  meta$Disease_Identity,
  levels = c("Control", "COPD", "IPF")
)

# ============================================================
# 2. Define epithelial cells
# ============================================================
# Use both original major class and marker-based annotation.
# This captures AT1, AT2, generic Epithelial, and possible epithelial-related clusters.

epi_cells <- rownames(meta)[
  meta$CellType_Category == "Epithelial" |
    meta$celltype_marker_based %in% c("AT1", "AT2", "Epithelial")
]

epi_cells <- unique(epi_cells)

message("Number of epithelial-related cells from light metadata:")
print(length(epi_cells))

meta_epi_global <- meta[epi_cells, , drop = FALSE]

write.csv(
  meta_epi_global,
  file.path(tab_dir, "01_epithelial_cells_metadata_from_global_annotation.csv")
)

writeLines(
  epi_cells,
  con = file.path(tab_dir, "01_epithelial_cells_keep.txt")
)

message("Epithelial cells by disease:")
print(table(meta_epi_global$Disease_Identity, useNA = "ifany"))

message("Epithelial cells by marker-based annotation:")
print(table(meta_epi_global$celltype_marker_based, useNA = "ifany"))

# ============================================================
# 3. Plot epithelial cells on global UMAP
# ============================================================

if (!is.null(umap_global)) {
  global_df <- as.data.frame(umap_global)
  colnames(global_df)[1:2] <- c("UMAP_1", "UMAP_2")
  global_df$cell_id <- rownames(global_df)
  
  global_df$is_epithelial_subset <- ifelse(global_df$cell_id %in% epi_cells, "Epithelial_subset", "Other")
  global_df$Disease_Identity <- meta[global_df$cell_id, "Disease_Identity"]
  global_df$celltype_marker_based <- meta[global_df$cell_id, "celltype_marker_based"]
  
  p <- ggplot(global_df, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(color = "grey85", size = 0.03, alpha = 0.25) +
    geom_point(
      data = global_df %>% filter(is_epithelial_subset == "Epithelial_subset"),
      aes(color = Disease_Identity),
      size = 0.08,
      alpha = 0.75
    ) +
    theme_classic(base_size = 12) +
    labs(
      title = "Epithelial-related cells highlighted on global UMAP",
      color = "Disease"
    )
  
  ggsave(
    file.path(fig_dir, "01_epithelial_cells_on_global_umap_by_disease.pdf"),
    p,
    width = 7,
    height = 6,
    limitsize = FALSE
  )
  
  p <- ggplot(global_df, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(color = "grey85", size = 0.03, alpha = 0.25) +
    geom_point(
      data = global_df %>% filter(is_epithelial_subset == "Epithelial_subset"),
      aes(color = celltype_marker_based),
      size = 0.08,
      alpha = 0.75
    ) +
    theme_classic(base_size = 12) +
    labs(
      title = "Epithelial-related cells highlighted on global UMAP",
      color = "Global annotation"
    )
  
  ggsave(
    file.path(fig_dir, "02_epithelial_cells_on_global_umap_by_global_annotation.pdf"),
    p,
    width = 7,
    height = 6,
    limitsize = FALSE
  )
  
  rm(global_df)
  gc()
}

# ============================================================
# 4. Load full Seurat object and subset epithelial cells
# ============================================================
# Important:
#   We restrict features during subset to reduce memory.
#   If this still fails, reduce max_hvg from 3000 to 2000 or 1000.

message("Reading full Seurat object...")
seu <- readRDS(file.path(project_dir, "ipf_annotated_singlet_seurat.rds"))

DefaultAssay(seu) <- "RNA"

epi_cells <- intersect(epi_cells, colnames(seu))

message("Epithelial cells found in Seurat object:")
print(length(epi_cells))

if (length(epi_cells) < 500) {
  stop("Too few epithelial cells found in Seurat object. Please check cell names.")
}

# Epithelial and fibrosis-related marker genes
epithelial_marker_genes <- unique(c(
  # pan epithelial
  "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CLDN4",
  
  # AT1
  "AGER", "PDPN", "CAV1", "CAV2", "AQP5", "HOPX", "RTKN2",
  
  # AT2
  "SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "SFTPD", "ABCA3", "NAPSA", "LAMP3",
  
  # basal / airway
  "KRT5", "KRT14", "KRT15", "KRT17", "TP63", "NGFR",
  
  # ciliated / secretory
  "FOXJ1", "PIFO", "TPPP3", "SCGB1A1", "SCGB3A2", "MUC5B", "MUC5AC",
  
  # aberrant basaloid / transitional / injury
  "KRT17", "KRT8", "KRT18", "COL1A1", "COL1A2", "FN1", "ITGA6",
  "VIM", "LGALS3", "CLDN4", "CDKN1A", "SOX4", "TPM1",
  
  # EMT-like / stress / fibrosis-related epithelial program
  "TGFB1", "TGFBR1", "TGFBR2", "SERPINE1", "CTGF", "POSTN",
  "MMP7", "MMP14", "SPP1", "CXCL8", "CXCL2", "IL6",
  
  # proliferation
  "MKI67", "TOP2A", "STMN1", "UBE2C"
))

hvg <- VariableFeatures(seu)

if (length(hvg) == 0) {
  warning("VariableFeatures(seu) is empty. Using epithelial markers only plus first 2000 genes.")
  hvg <- rownames(seu)[seq_len(min(2000, nrow(seu)))]
}

max_hvg <- 3000

features_use <- unique(c(
  head(hvg, max_hvg),
  epithelial_marker_genes
))

features_use <- intersect(features_use, rownames(seu))

message("Number of features used for epithelial subset:")
print(length(features_use))

message("Subsetting epithelial cells with limited features...")
seu_epi <- subset(
  seu,
  cells = epi_cells,
  features = features_use
)

rm(seu)
gc()

DefaultAssay(seu_epi) <- "RNA"

# Make sure disease order is consistent
seu_epi$Disease_Identity <- factor(
  seu_epi$Disease_Identity,
  levels = c("Control", "COPD", "IPF")
)

message("Epithelial subset object:")
print(dim(seu_epi))
print(table(seu_epi$Disease_Identity, useNA = "ifany"))
print(table(seu_epi$celltype_marker_based, useNA = "ifany"))

saveRDS(
  seu_epi,
  file.path(obj_dir, "epithelial_subset_limited_features_initial.rds")
)

# ============================================================
# 5. Normalize / variable features / scale within epithelial subset
# ============================================================
# If data are already normalized, NormalizeData is still acceptable here because
# the subset is much smaller. This gives a self-contained epithelial analysis.

message("Normalizing epithelial subset...")

seu_epi <- NormalizeData(
  seu_epi,
  normalization.method = "LogNormalize",
  scale.factor = 10000,
  verbose = TRUE
)

seu_epi <- FindVariableFeatures(
  seu_epi,
  selection.method = "dispersion",
  nfeatures = min(2000, length(features_use)),
  verbose = TRUE
)

features_epi <- unique(c(
  VariableFeatures(seu_epi),
  epithelial_marker_genes
))

features_epi <- intersect(features_epi, rownames(seu_epi))

DefaultAssay(seu_epi) <- "RNA"

# Remove old scale.data layer if present
if ("scale.data" %in% Layers(seu_epi[["RNA"]])) {
  seu_epi[["RNA"]]$scale.data <- NULL
}

seu_epi <- ScaleData(
  seu_epi,
  features = features_epi,
  vars.to.regress = "percent.mt",
  block.size = 500,
  verbose = TRUE
)

# ============================================================
# 6. PCA / UMAP / clustering within epithelial subset
# ============================================================

seu_epi <- RunPCA(
  seu_epi,
  features = features_epi,
  npcs = 30,
  verbose = TRUE
)

p_elbow <- ElbowPlot(seu_epi, ndims = 30)

ggsave(
  file.path(fig_dir, "03_epithelial_elbowplot_pca.pdf"),
  p_elbow,
  width = 6,
  height = 4
)

dims_use <- 1:20

seu_epi <- RunUMAP(
  seu_epi,
  reduction = "pca",
  dims = dims_use,
  verbose = TRUE
)

seu_epi <- FindNeighbors(
  seu_epi,
  reduction = "pca",
  dims = dims_use,
  verbose = TRUE
)

for (res in c(0.2, 0.4, 0.6, 0.8, 1.0)) {
  seu_epi <- FindClusters(
    seu_epi,
    resolution = res,
    algorithm = 1,
    verbose = TRUE
  )
}

seu_epi$epi_cluster_res06 <- seu_epi$RNA_snn_res.0.6
Idents(seu_epi) <- "epi_cluster_res06"

saveRDS(
  seu_epi,
  file.path(obj_dir, "epithelial_subset_clustered.rds")
)

# ============================================================
# 7. Epithelial module scoring
# ============================================================

suppressPackageStartupMessages({
  library(Matrix)
})

DefaultAssay(seu_epi) <- "RNA"

module_list <- list(
  Pan_epithelial = c("EPCAM", "KRT8", "KRT18", "KRT19"),
  AT1 = c("AGER", "PDPN", "CAV1", "CAV2", "AQP5", "HOPX"),
  AT2 = c("SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "SFTPD", "ABCA3", "NAPSA"),
  Basal = c("KRT5", "KRT14", "KRT17", "TP63", "NGFR"),
  Ciliated = c("FOXJ1", "PIFO", "TPPP3"),
  Secretory = c("SCGB1A1", "SCGB3A2", "MUC5B", "MUC5AC"),
  Aberrant_basaloid = c("KRT17", "KRT5", "COL1A1", "FN1", "ITGA6", "VIM", "LGALS3"),
  Transitional_KRT8 = c("KRT8", "KRT18", "CLDN4", "CDKN1A", "SOX4"),
  EMT_like = c("VIM", "FN1", "COL1A1", "COL1A2", "SERPINE1", "CTGF", "TGFBR1", "TGFBR2"),
  Injury_inflammation = c("MMP7", "SPP1", "CXCL8", "CXCL2", "IL6", "LGALS3"),
  Proliferation = c("MKI67", "TOP2A", "STMN1", "UBE2C")
)

rna_layers <- Layers(seu_epi[["RNA"]])

message("RNA layers:")
print(rna_layers)

# Prefer normalized data layers. If unavailable, use counts layers.
data_layers <- grep("^data", rna_layers, value = TRUE)
counts_layers <- grep("^counts", rna_layers, value = TRUE)

if (length(data_layers) > 0) {
  expr_layers <- data_layers
  message("Using RNA data layers:")
  print(expr_layers)
} else if (length(counts_layers) > 0) {
  expr_layers <- counts_layers
  warning("No data layers found. Using counts layers for module scores.")
  print(expr_layers)
} else {
  stop("No data.* or counts.* layers found. Please print Layers(seu_epi[['RNA']]) and inspect.")
}

score_cols <- c()

# Initialize score columns
for (nm in names(module_list)) {
  score_name <- paste0(nm, "_score")
  seu_epi@meta.data[[score_name]] <- NA_real_
  score_cols <- c(score_cols, score_name)
}

# Score each layer separately, then write scores to matching cells
for (ly in expr_layers) {
  message("Processing layer: ", ly)
  
  mat <- LayerData(
    object = seu_epi,
    assay = "RNA",
    layer = ly
  )
  
  cells_now <- colnames(mat)
  
  for (nm in names(module_list)) {
    genes <- intersect(module_list[[nm]], rownames(mat))
    score_name <- paste0(nm, "_score")
    
    message("  Module: ", nm, " | genes used: ", length(genes))
    
    if (length(genes) >= 2) {
      seu_epi@meta.data[cells_now, score_name] <- Matrix::colMeans(
        mat[genes, , drop = FALSE]
      )
    }
  }
  
  rm(mat)
  gc()
}

# Remove score columns that are entirely NA
score_cols <- score_cols[
  sapply(score_cols, function(x) any(!is.na(seu_epi@meta.data[[x]])))
]

message("Final module score columns:")
print(score_cols)

# Save metadata after module scoring
write.csv(
  seu_epi@meta.data,
  file.path(tab_dir, "02_epithelial_subset_metadata_with_module_scores.csv")
)

# ============================================================
# 8. UMAP plots
# ============================================================

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "epi_cluster_res06",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) + NoLegend()

ggsave(
  file.path(fig_dir, "04_epithelial_umap_epi_clusters_res06.pdf"),
  p,
  width = 7,
  height = 6,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "Disease_Identity",
  raster = TRUE
)

ggsave(
  file.path(fig_dir, "05_epithelial_umap_by_disease.pdf"),
  p,
  width = 7,
  height = 6,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "celltype_marker_based",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
)

ggsave(
  file.path(fig_dir, "06_epithelial_umap_by_global_marker_annotation.pdf"),
  p,
  width = 8,
  height = 6,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "sample_id",
  raster = TRUE
)

ggsave(
  file.path(fig_dir, "07_epithelial_umap_by_sample_id.pdf"),
  p,
  width = 12,
  height = 8,
  limitsize = FALSE
)

# Feature plots for key epithelial markers
marker_plot_genes <- c(
  "EPCAM", "AGER", "PDPN", "CAV1",
  "SFTPC", "SFTPA1", "ABCA3",
  "KRT5", "KRT17", "TP63",
  "MMP7", "SPP1", "COL1A1", "FN1",
  "MKI67"
)

marker_plot_genes <- intersect(marker_plot_genes, rownames(seu_epi))

if (length(marker_plot_genes) > 0) {
  p <- FeaturePlot(
    seu_epi,
    features = marker_plot_genes,
    reduction = "umap",
    ncol = 4,
    raster = TRUE
  )
  
  ggsave(
    file.path(fig_dir, "08_epithelial_featureplots_key_markers.pdf"),
    p,
    width = 14,
    height = ceiling(length(marker_plot_genes) / 4) * 3,
    limitsize = FALSE
  )
}

# Violin plots for module scores by disease
if (length(score_cols) > 0) {
  score_df <- seu_epi@meta.data %>%
    select(Disease_Identity, all_of(score_cols)) %>%
    pivot_longer(
      cols = all_of(score_cols),
      names_to = "module",
      values_to = "score"
    ) %>%
    mutate(module = gsub("_score1$", "", module))
  
  p <- ggplot(score_df, aes(x = Disease_Identity, y = score, fill = Disease_Identity)) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.15, size = 0.1, alpha = 0.25) +
    facet_wrap(~ module, scales = "free_y", ncol = 4) +
    theme_classic(base_size = 11) +
    labs(
      title = "Epithelial module scores by disease",
      x = NULL,
      y = "Module score"
    ) +
    theme(legend.position = "none")
  
  ggsave(
    file.path(fig_dir, "09_epithelial_module_scores_by_disease_boxplot.pdf"),
    p,
    width = 13,
    height = 10,
    limitsize = FALSE
  )
}

# ============================================================
# 9. Cluster composition and disease enrichment
# ============================================================

meta_epi <- seu_epi@meta.data

epi_cluster_count <- meta_epi %>%
  count(epi_cluster_res06, name = "n_cells") %>%
  arrange(desc(n_cells))

write.csv(
  epi_cluster_count,
  file.path(tab_dir, "03_epithelial_cluster_cell_counts.csv"),
  row.names = FALSE
)

epi_cluster_disease <- meta_epi %>%
  count(epi_cluster_res06, Disease_Identity, name = "n_cells") %>%
  group_by(epi_cluster_res06) %>%
  mutate(fraction_in_cluster = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  epi_cluster_disease,
  file.path(tab_dir, "04_epithelial_cluster_disease_composition.csv"),
  row.names = FALSE
)

epi_cluster_sample <- meta_epi %>%
  count(sample_id, Disease_Identity, epi_cluster_res06, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  epi_cluster_sample,
  file.path(tab_dir, "05_epithelial_cluster_fraction_by_sample.csv"),
  row.names = FALSE
)

# Observed / expected enrichment
compute_enrichment <- function(df, group_col, type_col) {
  tab <- df %>%
    count(.data[[group_col]], .data[[type_col]], name = "observed") %>%
    mutate(observed = as.numeric(observed))
  
  group_total <- df %>%
    count(.data[[group_col]], name = "group_total") %>%
    mutate(group_total = as.numeric(group_total))
  
  type_total <- df %>%
    count(.data[[type_col]], name = "type_total") %>%
    mutate(type_total = as.numeric(type_total))
  
  total_n <- as.numeric(nrow(df))
  
  enriched <- tab %>%
    left_join(group_total, by = group_col) %>%
    left_join(type_total, by = type_col) %>%
    mutate(
      expected = group_total * type_total / total_n,
      observed_expected_ratio = observed / expected,
      log2_observed_expected = log2(observed_expected_ratio)
    ) %>%
    arrange(.data[[type_col]], .data[[group_col]])
  
  return(enriched)
}

epi_cluster_enrichment <- compute_enrichment(
  df = meta_epi,
  group_col = "Disease_Identity",
  type_col = "epi_cluster_res06"
)

write.csv(
  epi_cluster_enrichment,
  file.path(tab_dir, "06_epithelial_cluster_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

p <- ggplot(
  epi_cluster_enrichment,
  aes(x = Disease_Identity, y = epi_cluster_res06, fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 12) +
  labs(
    title = "Disease enrichment of epithelial subclusters",
    x = NULL,
    y = "Epithelial subcluster",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "10_epithelial_cluster_disease_enrichment_heatmap.pdf"),
  p,
  width = 6,
  height = 6
)

# Sample-level cluster fraction boxplot
p <- ggplot(epi_cluster_sample, aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.6, alpha = 0.6) +
  facet_wrap(~ epi_cluster_res06, scales = "free_y", ncol = 4) +
  theme_classic(base_size = 10) +
  labs(
    title = "Sample-level epithelial subcluster fractions",
    x = NULL,
    y = "Fraction within epithelial compartment"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "11_epithelial_cluster_fraction_by_sample_boxplot.pdf"),
  p,
  width = 12,
  height = 8,
  limitsize = FALSE
)

# ============================================================
# 10. Pairwise Wilcoxon tests for epithelial subcluster fractions
# ============================================================

pairwise_wilcox_by_type <- function(frac_df, type_col) {
  disease_levels <- c("Control", "COPD", "IPF")
  pairs <- list(
    c("IPF", "Control"),
    c("IPF", "COPD"),
    c("COPD", "Control")
  )
  
  all_types <- sort(unique(frac_df[[type_col]]))
  
  res <- list()
  idx <- 1
  
  for (tp in all_types) {
    df_tp <- frac_df %>%
      filter(.data[[type_col]] == tp) %>%
      mutate(Disease_Identity = factor(Disease_Identity, levels = disease_levels))
    
    for (pr in pairs) {
      g1 <- pr[1]
      g2 <- pr[2]
      
      x <- df_tp %>% filter(Disease_Identity == g1) %>% pull(fraction)
      y <- df_tp %>% filter(Disease_Identity == g2) %>% pull(fraction)
      
      if (length(x) >= 2 && length(y) >= 2) {
        wt <- suppressWarnings(wilcox.test(x, y, exact = FALSE))
        pval <- wt$p.value
        median_g1 <- median(x, na.rm = TRUE)
        median_g2 <- median(y, na.rm = TRUE)
        delta <- median_g1 - median_g2
      } else {
        pval <- NA
        median_g1 <- ifelse(length(x) > 0, median(x, na.rm = TRUE), NA)
        median_g2 <- ifelse(length(y) > 0, median(y, na.rm = TRUE), NA)
        delta <- NA
      }
      
      res[[idx]] <- data.frame(
        subcluster = tp,
        comparison = paste0(g1, "_vs_", g2),
        group_1 = g1,
        group_2 = g2,
        median_group_1 = median_g1,
        median_group_2 = median_g2,
        median_delta = delta,
        p_value = pval
      )
      
      idx <- idx + 1
    }
  }
  
  res_df <- bind_rows(res) %>%
    group_by(comparison) %>%
    mutate(p_adj_BH = p.adjust(p_value, method = "BH")) %>%
    ungroup() %>%
    arrange(comparison, p_adj_BH)
  
  return(res_df)
}

epi_cluster_stats <- pairwise_wilcox_by_type(
  frac_df = epi_cluster_sample,
  type_col = "epi_cluster_res06"
)

write.csv(
  epi_cluster_stats,
  file.path(tab_dir, "07_epithelial_subcluster_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

ipf_enriched_epi_clusters <- epi_cluster_stats %>%
  filter(comparison %in% c("IPF_vs_Control", "IPF_vs_COPD")) %>%
  arrange(comparison, p_adj_BH, desc(median_delta))

write.csv(
  ipf_enriched_epi_clusters,
  file.path(tab_dir, "08_candidate_IPF_enriched_epithelial_subclusters.csv"),
  row.names = FALSE
)

# ============================================================
# 11. Module score summaries by disease and epithelial subcluster
# ============================================================

if (length(score_cols) > 0) {
  module_by_disease <- meta_epi %>%
    group_by(Disease_Identity) %>%
    summarise(
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_disease,
    file.path(tab_dir, "09_epithelial_module_scores_mean_by_disease.csv"),
    row.names = FALSE
  )
  
  module_by_cluster <- meta_epi %>%
    group_by(epi_cluster_res06) %>%
    summarise(
      n_cells = n(),
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_cluster,
    file.path(tab_dir, "10_epithelial_module_scores_mean_by_subcluster.csv"),
    row.names = FALSE
  )
  
  module_by_cluster_disease <- meta_epi %>%
    group_by(epi_cluster_res06, Disease_Identity) %>%
    summarise(
      n_cells = n(),
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_cluster_disease,
    file.path(tab_dir, "11_epithelial_module_scores_mean_by_subcluster_and_disease.csv"),
    row.names = FALSE
  )
}

# ============================================================
# 12. DotPlot for key epithelial markers
# ============================================================

dot_genes <- unique(c(
  "EPCAM", "KRT8", "KRT18", "KRT19",
  "AGER", "PDPN", "CAV1", "AQP5",
  "SFTPC", "SFTPA1", "SFTPA2", "ABCA3",
  "KRT5", "KRT17", "TP63",
  "FOXJ1", "SCGB1A1", "MUC5B",
  "MMP7", "SPP1", "COL1A1", "FN1", "VIM",
  "SERPINE1", "CTGF", "MKI67", "TOP2A"
))

dot_genes <- intersect(dot_genes, rownames(seu_epi))

if (length(dot_genes) > 0) {
  p <- DotPlot(
    seu_epi,
    features = dot_genes,
    group.by = "epi_cluster_res06"
  ) + RotatedAxis()
  
  ggsave(
    file.path(fig_dir, "12_epithelial_dotplot_key_markers_by_subcluster.pdf"),
    p,
    width = 14,
    height = 6,
    limitsize = FALSE
  )
}

# ============================================================
# 13. Low-memory marker detection for epithelial subclusters
# ============================================================

DefaultAssay(seu_epi) <- "RNA"
Idents(seu_epi) <- "epi_cluster_res06"

set.seed(123)

cluster_ids <- levels(Idents(seu_epi))

cells_marker <- unlist(lapply(cluster_ids, function(cl) {
  cells_cl <- WhichCells(seu_epi, idents = cl)
  sample(cells_cl, min(800, length(cells_cl)))
}))

seu_marker <- subset(
  seu_epi,
  cells = cells_marker
)

DefaultAssay(seu_marker) <- "RNA"
Idents(seu_marker) <- "epi_cluster_res06"

# Critical for Seurat v5 multi-layer assay
message("Layers before JoinLayers:")
print(Layers(seu_marker[["RNA"]]))

seu_marker <- JoinLayers(
  object = seu_marker,
  assay = "RNA"
)

message("Layers after JoinLayers:")
print(Layers(seu_marker[["RNA"]]))

# Make sure identities are still correct
Idents(seu_marker) <- "epi_cluster_res06"
print(table(Idents(seu_marker)))

features_marker <- unique(c(
  VariableFeatures(seu_epi),
  epithelial_marker_genes
))

features_marker <- intersect(features_marker, rownames(seu_marker))

message("Number of marker features used:")
print(length(features_marker))

# Optional safety check
if (length(features_marker) < 50) {
  warning("Very few marker features available. Consider increasing max_hvg or adding marker genes.")
}

markers_epi <- FindAllMarkers(
  seu_marker,
  features = features_marker,
  only.pos = TRUE,
  min.pct = 0.20,
  logfc.threshold = 0.25,
  test.use = "wilcox",
  verbose = TRUE
)

write.csv(
  markers_epi,
  file.path(tab_dir, "12_epithelial_subcluster_markers_downsample800.csv"),
  row.names = FALSE
)

if (nrow(markers_epi) > 0) {
  top_markers_epi <- markers_epi %>%
    group_by(cluster) %>%
    slice_max(order_by = avg_log2FC, n = 20, with_ties = FALSE) %>%
    ungroup()
  
  write.csv(
    top_markers_epi,
    file.path(tab_dir, "13_epithelial_subcluster_top20_markers.csv"),
    row.names = FALSE
  )
} else {
  warning("markers_epi is empty after JoinLayers. Please check features_marker and cluster identities.")
}

rm(seu_marker)
gc()

# ============================================================
# 14. Generate concise epithelial subcluster interpretation table
# ============================================================

cluster_top_global_anno <- meta_epi %>%
  count(epi_cluster_res06, celltype_marker_based, name = "n_cells") %>%
  group_by(epi_cluster_res06) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  slice_max(order_by = fraction, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(
    epi_cluster_res06,
    top_global_marker_annotation = celltype_marker_based,
    top_global_marker_annotation_fraction = fraction
  )

cluster_disease_wide <- epi_cluster_disease %>%
  select(epi_cluster_res06, Disease_Identity, fraction_in_cluster) %>%
  tidyr::pivot_wider(
    names_from = Disease_Identity,
    values_from = fraction_in_cluster,
    names_prefix = "fraction_"
  )

cluster_summary <- epi_cluster_count %>%
  left_join(cluster_top_global_anno, by = "epi_cluster_res06") %>%
  left_join(cluster_disease_wide, by = "epi_cluster_res06")

if (length(score_cols) > 0) {
  module_by_cluster <- meta_epi %>%
    group_by(epi_cluster_res06) %>%
    summarise(
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  cluster_summary <- cluster_summary %>%
    left_join(module_by_cluster, by = "epi_cluster_res06")
}

write.csv(
  cluster_summary,
  file.path(tab_dir, "14_epithelial_subcluster_summary_for_interpretation.csv"),
  row.names = FALSE
)

# ============================================================
# 15. Save final epithelial object
# ============================================================

saveRDS(
  seu_epi,
  file.path(obj_dir, "epithelial_subset_final.rds")
)

message("Done.")
message("Epithelial subset analysis finished.")
message("Main output directory:")
message(out_dir)

message("Key tables:")
message(file.path(tab_dir, "04_epithelial_cluster_disease_composition.csv"))
message(file.path(tab_dir, "07_epithelial_subcluster_fraction_pairwise_wilcox_sample_level.csv"))
message(file.path(tab_dir, "08_candidate_IPF_enriched_epithelial_subclusters.csv"))
message(file.path(tab_dir, "12_epithelial_subcluster_markers_downsample800.csv"))
message(file.path(tab_dir, "14_epithelial_subcluster_summary_for_interpretation.csv"))

message("Key figures:")
message(file.path(fig_dir, "04_epithelial_umap_epi_clusters_res06.pdf"))
message(file.path(fig_dir, "05_epithelial_umap_by_disease.pdf"))
message(file.path(fig_dir, "09_epithelial_module_scores_by_disease_boxplot.pdf"))
message(file.path(fig_dir, "10_epithelial_cluster_disease_enrichment_heatmap.pdf"))
message(file.path(fig_dir, "11_epithelial_cluster_fraction_by_sample_boxplot.pdf"))
message(file.path(fig_dir, "12_epithelial_dotplot_key_markers_by_subcluster.pdf"))