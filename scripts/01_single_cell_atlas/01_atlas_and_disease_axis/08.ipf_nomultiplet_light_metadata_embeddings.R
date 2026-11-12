# 08_multicellular_population_structure_nomultiplet.R
# Multi-cellular population structure analysis for IPF / COPD / Control
# Input: ipf_nomultiplet_light_metadata_embeddings.rds
# This script uses metadata + UMAP only. No Seurat subset is performed.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(readr)
  library(forcats)
})

project_dir <- getwd()

dir.create(file.path(project_dir, "results", "multicellular_structure"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "multicellular_structure", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(project_dir, "results", "multicellular_structure", "tables"), recursive = TRUE, showWarnings = FALSE)

out_dir <- file.path(project_dir, "results", "multicellular_structure")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")

# ============================================================
# 1. Load lightweight nomultiplet object
# ============================================================

light <- readRDS(file.path(project_dir, "ipf_nomultiplet_light_metadata_embeddings.rds"))

meta <- light$metadata
umap <- light$umap

message("Metadata dimension:")
print(dim(meta))

message("CellType_Category:")
print(table(meta$CellType_Category, useNA = "ifany"))

message("Disease_Identity:")
print(table(meta$Disease_Identity, useNA = "ifany"))

message("celltype_marker_based:")
print(table(meta$celltype_marker_based, useNA = "ifany"))

# Ensure key columns exist
required_cols <- c(
  "sample_id",
  "Disease_Identity",
  "CellType_Category",
  "celltype_marker_based",
  "cluster_default"
)

missing_cols <- setdiff(required_cols, colnames(meta))

if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Set disease order
meta$Disease_Identity <- factor(
  meta$Disease_Identity,
  levels = c("Control", "COPD", "IPF")
)

# Ensure cluster as factor
meta$cluster_default <- as.factor(meta$cluster_default)

# ============================================================
# 2. Global cell counts
# ============================================================

cell_count_by_disease <- meta %>%
  count(Disease_Identity, name = "n_cells") %>%
  mutate(fraction = n_cells / sum(n_cells))

write.csv(
  cell_count_by_disease,
  file.path(tab_dir, "01_cell_count_by_disease.csv"),
  row.names = FALSE
)

cell_count_by_sample <- meta %>%
  count(sample_id, Disease_Identity, name = "n_cells") %>%
  arrange(Disease_Identity, sample_id)

write.csv(
  cell_count_by_sample,
  file.path(tab_dir, "02_cell_count_by_sample.csv"),
  row.names = FALSE
)

# Plot cell counts by disease
p <- ggplot(cell_count_by_disease, aes(x = Disease_Identity, y = n_cells, fill = Disease_Identity)) +
  geom_col(width = 0.7) +
  theme_classic(base_size = 13) +
  labs(
    title = "Cell count by disease group",
    x = NULL,
    y = "Number of cells"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "01_cell_count_by_disease.pdf"),
  p,
  width = 5,
  height = 4
)

# ============================================================
# 3. Major cell class composition
# ============================================================

major_count <- meta %>%
  count(Disease_Identity, CellType_Category, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  major_count,
  file.path(tab_dir, "03_major_celltype_fraction_by_disease.csv"),
  row.names = FALSE
)

major_sample_frac <- meta %>%
  count(sample_id, Disease_Identity, CellType_Category, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  major_sample_frac,
  file.path(tab_dir, "04_major_celltype_fraction_by_sample.csv"),
  row.names = FALSE
)

# Stacked bar by disease
p <- ggplot(major_count, aes(x = Disease_Identity, y = fraction, fill = CellType_Category)) +
  geom_col(width = 0.75) +
  theme_classic(base_size = 13) +
  labs(
    title = "Major cell class composition by disease",
    x = NULL,
    y = "Fraction",
    fill = "Major cell class"
  )

ggsave(
  file.path(fig_dir, "02_major_celltype_fraction_by_disease_stacked.pdf"),
  p,
  width = 7,
  height = 5
)

# Sample-level stacked bar
p <- ggplot(
  major_sample_frac,
  aes(x = reorder(sample_id, as.numeric(Disease_Identity)), y = fraction, fill = CellType_Category)
) +
  geom_col(width = 0.85) +
  facet_grid(. ~ Disease_Identity, scales = "free_x", space = "free_x") +
  theme_classic(base_size = 11) +
  labs(
    title = "Major cell class composition by sample",
    x = "Sample",
    y = "Fraction",
    fill = "Major cell class"
  ) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.background = element_blank()
  )

ggsave(
  file.path(fig_dir, "03_major_celltype_fraction_by_sample_stacked.pdf"),
  p,
  width = 18,
  height = 6,
  limitsize = FALSE
)

# Boxplot sample-level major class fraction
p <- ggplot(major_sample_frac, aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 1, alpha = 0.7) +
  facet_wrap(~ CellType_Category, scales = "free_y") +
  theme_classic(base_size = 12) +
  labs(
    title = "Sample-level major cell class fractions",
    x = NULL,
    y = "Fraction"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "04_major_celltype_fraction_by_sample_boxplot.pdf"),
  p,
  width = 10,
  height = 7,
  limitsize = FALSE
)

# ============================================================
# 4. Marker-based cell type composition
# ============================================================

marker_count <- meta %>%
  count(Disease_Identity, celltype_marker_based, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  marker_count,
  file.path(tab_dir, "05_marker_celltype_fraction_by_disease.csv"),
  row.names = FALSE
)

marker_sample_frac <- meta %>%
  count(sample_id, Disease_Identity, celltype_marker_based, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  marker_sample_frac,
  file.path(tab_dir, "06_marker_celltype_fraction_by_sample.csv"),
  row.names = FALSE
)

# Stacked bar by disease
p <- ggplot(marker_count, aes(x = Disease_Identity, y = fraction, fill = celltype_marker_based)) +
  geom_col(width = 0.75) +
  theme_classic(base_size = 13) +
  labs(
    title = "Marker-based cell type composition by disease",
    x = NULL,
    y = "Fraction",
    fill = "Cell type"
  )

ggsave(
  file.path(fig_dir, "05_marker_celltype_fraction_by_disease_stacked.pdf"),
  p,
  width = 8,
  height = 5
)

# Sample-level stacked bar
p <- ggplot(
  marker_sample_frac,
  aes(x = reorder(sample_id, as.numeric(Disease_Identity)), y = fraction, fill = celltype_marker_based)
) +
  geom_col(width = 0.85) +
  facet_grid(. ~ Disease_Identity, scales = "free_x", space = "free_x") +
  theme_classic(base_size = 10) +
  labs(
    title = "Marker-based cell type composition by sample",
    x = "Sample",
    y = "Fraction",
    fill = "Cell type"
  ) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.background = element_blank()
  )

ggsave(
  file.path(fig_dir, "06_marker_celltype_fraction_by_sample_stacked.pdf"),
  p,
  width = 20,
  height = 7,
  limitsize = FALSE
)

# Boxplot sample-level marker cell type fraction
p <- ggplot(marker_sample_frac, aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.7) +
  facet_wrap(~ celltype_marker_based, scales = "free_y", ncol = 4) +
  theme_classic(base_size = 11) +
  labs(
    title = "Sample-level marker-based cell type fractions",
    x = NULL,
    y = "Fraction"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "07_marker_celltype_fraction_by_sample_boxplot.pdf"),
  p,
  width = 13,
  height = 10,
  limitsize = FALSE
)

# ============================================================
# 5. Disease enrichment by cell type
#    Use observed / expected ratio and log2 enrichment
# ============================================================

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

major_enrichment <- compute_enrichment(
  df = meta,
  group_col = "Disease_Identity",
  type_col = "CellType_Category"
)

write.csv(
  major_enrichment,
  file.path(tab_dir, "07_major_celltype_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

marker_enrichment <- compute_enrichment(
  df = meta,
  group_col = "Disease_Identity",
  type_col = "celltype_marker_based"
)

write.csv(
  marker_enrichment,
  file.path(tab_dir, "08_marker_celltype_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

# Heatmap-like tile plot for marker enrichment
p <- ggplot(
  marker_enrichment,
  aes(x = Disease_Identity, y = fct_reorder(celltype_marker_based, log2_observed_expected), fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 12) +
  labs(
    title = "Disease enrichment of marker-based cell types",
    x = NULL,
    y = "Cell type",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "08_marker_celltype_disease_enrichment_heatmap.pdf"),
  p,
  width = 6,
  height = 7
)

# Heatmap-like tile plot for major enrichment
p <- ggplot(
  major_enrichment,
  aes(x = Disease_Identity, y = fct_reorder(CellType_Category, log2_observed_expected), fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 12) +
  labs(
    title = "Disease enrichment of major cell classes",
    x = NULL,
    y = "Major cell class",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "09_major_celltype_disease_enrichment_heatmap.pdf"),
  p,
  width = 6,
  height = 5
)

# ============================================================
# 6. Statistical tests for sample-level fractions
#    Use sample as the statistical unit
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
        cell_type = tp,
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

major_stats <- pairwise_wilcox_by_type(
  frac_df = major_sample_frac,
  type_col = "CellType_Category"
)

write.csv(
  major_stats,
  file.path(tab_dir, "09_major_celltype_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

marker_stats <- pairwise_wilcox_by_type(
  frac_df = marker_sample_frac,
  type_col = "celltype_marker_based"
)

write.csv(
  marker_stats,
  file.path(tab_dir, "10_marker_celltype_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

# Candidate IPF-enriched cell types
ipf_enriched_marker <- marker_stats %>%
  filter(comparison %in% c("IPF_vs_Control", "IPF_vs_COPD")) %>%
  filter(!is.na(p_adj_BH)) %>%
  arrange(comparison, p_adj_BH, desc(median_delta))

write.csv(
  ipf_enriched_marker,
  file.path(tab_dir, "11_candidate_IPF_enriched_marker_celltypes.csv"),
  row.names = FALSE
)

# ============================================================
# 7. Cluster composition by disease
# ============================================================

cluster_count <- meta %>%
  count(Disease_Identity, cluster_default, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction_within_disease = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  cluster_count,
  file.path(tab_dir, "12_cluster_fraction_by_disease.csv"),
  row.names = FALSE
)

cluster_sample_frac <- meta %>%
  count(sample_id, Disease_Identity, cluster_default, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  cluster_sample_frac,
  file.path(tab_dir, "13_cluster_fraction_by_sample.csv"),
  row.names = FALSE
)

cluster_enrichment <- compute_enrichment(
  df = meta,
  group_col = "Disease_Identity",
  type_col = "cluster_default"
)

write.csv(
  cluster_enrichment,
  file.path(tab_dir, "14_cluster_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

cluster_stats <- pairwise_wilcox_by_type(
  frac_df = cluster_sample_frac,
  type_col = "cluster_default"
)

write.csv(
  cluster_stats,
  file.path(tab_dir, "15_cluster_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

# Cluster enrichment heatmap
p <- ggplot(
  cluster_enrichment,
  aes(x = Disease_Identity, y = cluster_default, fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 12) +
  labs(
    title = "Disease enrichment of clusters",
    x = NULL,
    y = "Cluster",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "10_cluster_disease_enrichment_heatmap.pdf"),
  p,
  width = 6,
  height = 8
)

# Cluster fraction boxplot
p <- ggplot(cluster_sample_frac, aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.7, alpha = 0.7) +
  facet_wrap(~ cluster_default, scales = "free_y", ncol = 5) +
  theme_classic(base_size = 10) +
  labs(
    title = "Sample-level cluster fractions",
    x = NULL,
    y = "Fraction"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "11_cluster_fraction_by_sample_boxplot.pdf"),
  p,
  width = 14,
  height = 10,
  limitsize = FALSE
)

# ============================================================
# 8. Cluster annotation summary
# ============================================================

cluster_celltype_summary <- meta %>%
  count(cluster_default, celltype_marker_based, name = "n_cells") %>%
  group_by(cluster_default) %>%
  mutate(fraction_in_cluster = n_cells / sum(n_cells)) %>%
  arrange(cluster_default, desc(fraction_in_cluster)) %>%
  ungroup()

write.csv(
  cluster_celltype_summary,
  file.path(tab_dir, "16_cluster_marker_celltype_composition.csv"),
  row.names = FALSE
)

cluster_major_summary <- meta %>%
  count(cluster_default, CellType_Category, name = "n_cells") %>%
  group_by(cluster_default) %>%
  mutate(fraction_in_cluster = n_cells / sum(n_cells)) %>%
  arrange(cluster_default, desc(fraction_in_cluster)) %>%
  ungroup()

write.csv(
  cluster_major_summary,
  file.path(tab_dir, "17_cluster_major_celltype_composition.csv"),
  row.names = FALSE
)

cluster_disease_summary <- meta %>%
  count(cluster_default, Disease_Identity, name = "n_cells") %>%
  group_by(cluster_default) %>%
  mutate(fraction_in_cluster = n_cells / sum(n_cells)) %>%
  arrange(cluster_default, Disease_Identity) %>%
  ungroup()

write.csv(
  cluster_disease_summary,
  file.path(tab_dir, "18_cluster_disease_composition.csv"),
  row.names = FALSE
)

# Generate one concise cluster summary table
cluster_top_celltype <- cluster_celltype_summary %>%
  group_by(cluster_default) %>%
  slice_max(order_by = fraction_in_cluster, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(
    cluster_default,
    top_marker_celltype = celltype_marker_based,
    top_marker_celltype_fraction = fraction_in_cluster
  )

cluster_top_major <- cluster_major_summary %>%
  group_by(cluster_default) %>%
  slice_max(order_by = fraction_in_cluster, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(
    cluster_default,
    top_major_celltype = CellType_Category,
    top_major_celltype_fraction = fraction_in_cluster
  )

cluster_total <- meta %>%
  count(cluster_default, name = "n_cells")

cluster_ipf_fraction <- cluster_disease_summary %>%
  filter(Disease_Identity == "IPF") %>%
  select(cluster_default, IPF_fraction_in_cluster = fraction_in_cluster)

cluster_control_fraction <- cluster_disease_summary %>%
  filter(Disease_Identity == "Control") %>%
  select(cluster_default, Control_fraction_in_cluster = fraction_in_cluster)

cluster_copd_fraction <- cluster_disease_summary %>%
  filter(Disease_Identity == "COPD") %>%
  select(cluster_default, COPD_fraction_in_cluster = fraction_in_cluster)

cluster_summary <- cluster_total %>%
  left_join(cluster_top_major, by = "cluster_default") %>%
  left_join(cluster_top_celltype, by = "cluster_default") %>%
  left_join(cluster_control_fraction, by = "cluster_default") %>%
  left_join(cluster_copd_fraction, by = "cluster_default") %>%
  left_join(cluster_ipf_fraction, by = "cluster_default") %>%
  arrange(desc(IPF_fraction_in_cluster), desc(n_cells))

write.csv(
  cluster_summary,
  file.path(tab_dir, "19_cluster_summary_for_interpretation.csv"),
  row.names = FALSE
)

# ============================================================
# 9. UMAP visualization using existing embedding
# ============================================================

if (!is.null(umap)) {
  plot_df <- as.data.frame(umap)
  colnames(plot_df)[1:2] <- c("UMAP_1", "UMAP_2")
  plot_df$cell_id <- rownames(plot_df)
  
  plot_df <- cbind(
    plot_df,
    meta[plot_df$cell_id, , drop = FALSE]
  )
  
  # Disease UMAP
  p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = Disease_Identity)) +
    geom_point(size = 0.05, alpha = 0.6) +
    theme_classic(base_size = 12) +
    labs(
      title = "UMAP by disease group",
      color = "Disease"
    )
  
  ggsave(
    file.path(fig_dir, "12_umap_by_disease.pdf"),
    p,
    width = 7,
    height = 6,
    limitsize = FALSE
  )
  
  # Major class UMAP
  p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = CellType_Category)) +
    geom_point(size = 0.05, alpha = 0.6) +
    theme_classic(base_size = 12) +
    labs(
      title = "UMAP by major cell class",
      color = "Major class"
    )
  
  ggsave(
    file.path(fig_dir, "13_umap_by_major_cellclass.pdf"),
    p,
    width = 8,
    height = 6,
    limitsize = FALSE
  )
  
  # Marker cell type UMAP
  p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = celltype_marker_based)) +
    geom_point(size = 0.05, alpha = 0.6) +
    theme_classic(base_size = 12) +
    labs(
      title = "UMAP by marker-based cell type",
      color = "Cell type"
    )
  
  ggsave(
    file.path(fig_dir, "14_umap_by_marker_celltype.pdf"),
    p,
    width = 9,
    height = 7,
    limitsize = FALSE
  )
  
  # Cluster UMAP
  p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = cluster_default)) +
    geom_point(size = 0.05, alpha = 0.6) +
    theme_classic(base_size = 12) +
    labs(
      title = "UMAP by cluster",
      color = "Cluster"
    )
  
  ggsave(
    file.path(fig_dir, "15_umap_by_cluster.pdf"),
    p,
    width = 8,
    height = 6,
    limitsize = FALSE
  )
  
  # Disease faceted UMAP
  p <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(color = "grey80", size = 0.03, alpha = 0.3) +
    geom_point(aes(color = Disease_Identity), size = 0.05, alpha = 0.6) +
    facet_wrap(~ Disease_Identity) +
    theme_classic(base_size = 12) +
    labs(
      title = "Disease-specific UMAP distribution",
      color = "Disease"
    )
  
  ggsave(
    file.path(fig_dir, "16_umap_disease_faceted.pdf"),
    p,
    width = 12,
    height = 4,
    limitsize = FALSE
  )
  
  rm(plot_df)
  gc()
}

# ============================================================
# 10. Compartment-level summaries
# ============================================================

for (comp in sort(unique(meta$CellType_Category))) {
  meta_comp <- meta %>%
    filter(CellType_Category == comp)
  
  comp_dir <- file.path(out_dir, "tables", paste0("compartment_", comp))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)
  
  comp_celltype <- meta_comp %>%
    count(Disease_Identity, celltype_marker_based, name = "n_cells") %>%
    group_by(Disease_Identity) %>%
    mutate(fraction = n_cells / sum(n_cells)) %>%
    ungroup()
  
  write.csv(
    comp_celltype,
    file.path(comp_dir, paste0("celltype_fraction_by_disease_", comp, ".csv")),
    row.names = FALSE
  )
  
  comp_cluster <- meta_comp %>%
    count(Disease_Identity, cluster_default, name = "n_cells") %>%
    group_by(Disease_Identity) %>%
    mutate(fraction = n_cells / sum(n_cells)) %>%
    ungroup()
  
  write.csv(
    comp_cluster,
    file.path(comp_dir, paste0("cluster_fraction_by_disease_", comp, ".csv")),
    row.names = FALSE
  )
}

# ============================================================
# 11. Save analysis object
# ============================================================

analysis_outputs <- list(
  cell_count_by_disease = cell_count_by_disease,
  cell_count_by_sample = cell_count_by_sample,
  major_count = major_count,
  major_sample_frac = major_sample_frac,
  marker_count = marker_count,
  marker_sample_frac = marker_sample_frac,
  major_enrichment = major_enrichment,
  marker_enrichment = marker_enrichment,
  major_stats = major_stats,
  marker_stats = marker_stats,
  cluster_count = cluster_count,
  cluster_sample_frac = cluster_sample_frac,
  cluster_enrichment = cluster_enrichment,
  cluster_stats = cluster_stats,
  cluster_summary = cluster_summary,
  cluster_celltype_summary = cluster_celltype_summary,
  cluster_disease_summary = cluster_disease_summary
)

saveRDS(
  analysis_outputs,
  file.path(out_dir, "multicellular_structure_analysis_outputs.rds")
)

message("Done.")
message("Results saved to:")
message(out_dir)
