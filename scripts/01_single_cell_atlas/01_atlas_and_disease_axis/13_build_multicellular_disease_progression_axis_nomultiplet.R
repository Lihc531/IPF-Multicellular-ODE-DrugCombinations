# 13_build_multicellular_disease_progression_axis_nomultiplet.R
# Build multi-cellular IPF disease progression axis
# Input: ipf_nomultiplet_light_metadata_embeddings.rds
# Output: sample-level, cell-level, cluster-level, and celltype-level progression scores
#
# This script is low-memory:
# - does not read full Seurat object
# - does not subset Seurat object
# - does not use expression matrix
# - uses metadata + UMAP only

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(tibble)
  library(viridis)
})

set.seed(123)

project_dir <- getwd()

out_dir <- file.path(project_dir, "results", "progression_axis")
tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 0. Parameters
# ============================================================

input_rds <- file.path(project_dir, "ipf_nomultiplet_light_metadata_embeddings.rds")

min_cells_per_sample <- 100

disease_order <- c("Control", "COPD", "IPF")

# Disease numeric coding used only for orientation / sanity check
disease_numeric_map <- c(
  "Control" = 0,
  "COPD"    = 0.5,
  "IPF"     = 1
)

# Weighting of different axis components at cell level
# sample_score: multi-cellular composition score
# celltype_score: disease enrichment of cell type
# cluster_score: disease enrichment of cluster
w_sample   <- 0.50
w_celltype <- 0.25
w_cluster  <- 0.25

pseudocount <- 1e-6

# ============================================================
# 1. Helper functions
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
  s <- sd(x, na.rm = TRUE)
  m <- mean(x, na.rm = TRUE)
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

# Centered log-ratio transform for compositional data
clr_transform <- function(mat, eps = 1e-6) {
  mat <- as.matrix(mat)
  mat <- mat + eps
  log_mat <- log(mat)
  gm <- rowMeans(log_mat)
  sweep(log_mat, 1, gm, "-")
}

# ============================================================
# 2. Load lightweight object
# ============================================================

if (!file.exists(input_rds)) {
  stop("Input file not found: ", input_rds)
}

light <- readRDS(input_rds)

meta <- light$metadata
umap <- light$umap

message("Loaded metadata:")
print(dim(meta))

message("Loaded UMAP:")
print(dim(umap))

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

meta$Disease_Identity <- as.character(meta$Disease_Identity)
meta$Disease_Identity <- factor(meta$Disease_Identity, levels = disease_order)

meta$cluster_default <- as.factor(meta$cluster_default)
meta$celltype_marker_based <- as.factor(meta$celltype_marker_based)
meta$CellType_Category <- as.factor(meta$CellType_Category)

message("Disease distribution:")
print(table(meta$Disease_Identity, useNA = "ifany"))

message("Major cell class distribution:")
print(table(meta$CellType_Category, useNA = "ifany"))

message("Marker-based cell type distribution:")
print(table(meta$celltype_marker_based, useNA = "ifany"))

# ============================================================
# 3. Sample-level quality table
# ============================================================

sample_info <- meta %>%
  count(sample_id, Disease_Identity, name = "n_cells") %>%
  arrange(Disease_Identity, sample_id) %>%
  mutate(
    sample_pass_min_cells = n_cells >= min_cells_per_sample,
    disease_numeric = unname(disease_numeric_map[as.character(Disease_Identity)])
  )

write.csv(
  sample_info,
  file.path(tab_dir, "01_sample_info_for_progression_axis.csv"),
  row.names = FALSE
)

message("Samples included:")
print(table(sample_info$Disease_Identity, sample_info$sample_pass_min_cells))

samples_use <- sample_info %>%
  filter(sample_pass_min_cells) %>%
  pull(sample_id)

meta_use <- meta %>%
  filter(sample_id %in% samples_use)

message("Cells used for axis construction:")
print(dim(meta_use))

# ============================================================
# 4. Build sample x cell type fraction matrix
# ============================================================

celltype_fraction <- meta_use %>%
  count(sample_id, Disease_Identity, celltype_marker_based, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(
    total_cells_sample = sum(n_cells),
    fraction = n_cells / total_cells_sample
  ) %>%
  ungroup()

write.csv(
  celltype_fraction,
  file.path(tab_dir, "02_sample_celltype_fraction_long.csv"),
  row.names = FALSE
)

celltype_fraction_wide <- celltype_fraction %>%
  select(sample_id, Disease_Identity, celltype_marker_based, fraction) %>%
  pivot_wider(
    names_from = celltype_marker_based,
    values_from = fraction,
    values_fill = 0
  ) %>%
  arrange(Disease_Identity, sample_id)

write.csv(
  celltype_fraction_wide,
  file.path(tab_dir, "03_sample_celltype_fraction_wide.csv"),
  row.names = FALSE
)

feature_cols <- setdiff(
  colnames(celltype_fraction_wide),
  c("sample_id", "Disease_Identity")
)

frac_mat <- as.matrix(celltype_fraction_wide[, feature_cols, drop = FALSE])
rownames(frac_mat) <- celltype_fraction_wide$sample_id

clr_mat <- clr_transform(frac_mat, eps = pseudocount)

clr_df <- as.data.frame(clr_mat) %>%
  rownames_to_column("sample_id") %>%
  left_join(
    celltype_fraction_wide %>% select(sample_id, Disease_Identity),
    by = "sample_id"
  ) %>%
  relocate(sample_id, Disease_Identity)

write.csv(
  clr_df,
  file.path(tab_dir, "04_sample_celltype_fraction_clr.csv"),
  row.names = FALSE
)

# ============================================================
# 5. Supervised multi-cellular composition axis
#    weight = mean(IPF) - mean(Control) in CLR space
# ============================================================

group_vec <- celltype_fraction_wide$Disease_Identity

if (!all(c("Control", "IPF") %in% as.character(unique(group_vec)))) {
  stop("Both Control and IPF samples are required to build the supervised progression axis.")
}

mean_control <- colMeans(clr_mat[group_vec == "Control", , drop = FALSE], na.rm = TRUE)
mean_ipf <- colMeans(clr_mat[group_vec == "IPF", , drop = FALSE], na.rm = TRUE)

axis_weight <- mean_ipf - mean_control

axis_weight_df <- data.frame(
  celltype_marker_based = names(axis_weight),
  axis_weight_raw = as.numeric(axis_weight),
  axis_weight_z = safe_zscore(axis_weight),
  axis_weight_scaled_0_1 = rescale01(axis_weight)
) %>%
  arrange(desc(axis_weight_raw))

write.csv(
  axis_weight_df,
  file.path(tab_dir, "05_celltype_axis_weights_IPF_minus_Control.csv"),
  row.names = FALSE
)

raw_supervised_score <- as.numeric(clr_mat %*% axis_weight)

sample_axis <- celltype_fraction_wide %>%
  select(sample_id, Disease_Identity) %>%
  mutate(
    disease_numeric = unname(disease_numeric_map[as.character(Disease_Identity)]),
    multicellular_axis_raw = raw_supervised_score,
    multicellular_axis_score = rescale01(raw_supervised_score)
  )

# Orient axis: Control should be low, IPF should be high
mean_control_score <- mean(
  sample_axis$multicellular_axis_score[sample_axis$Disease_Identity == "Control"],
  na.rm = TRUE
)

mean_ipf_score <- mean(
  sample_axis$multicellular_axis_score[sample_axis$Disease_Identity == "IPF"],
  na.rm = TRUE
)

if (mean_ipf_score < mean_control_score) {
  sample_axis$multicellular_axis_score <- 1 - sample_axis$multicellular_axis_score
  sample_axis$multicellular_axis_raw <- -sample_axis$multicellular_axis_raw
  axis_weight <- -axis_weight
  axis_weight_df$axis_weight_raw <- -axis_weight_df$axis_weight_raw
  axis_weight_df <- axis_weight_df %>%
    mutate(
      axis_weight_z = safe_zscore(axis_weight_raw),
      axis_weight_scaled_0_1 = rescale01(axis_weight_raw)
    ) %>%
    arrange(desc(axis_weight_raw))
  
  write.csv(
    axis_weight_df,
    file.path(tab_dir, "05_celltype_axis_weights_IPF_minus_Control.csv"),
    row.names = FALSE
  )
}

sample_axis <- sample_axis %>%
  mutate(
    multicellular_axis_stage = make_stage(multicellular_axis_score)
  ) %>%
  arrange(multicellular_axis_score)

write.csv(
  sample_axis,
  file.path(tab_dir, "06_sample_multicellular_progression_axis.csv"),
  row.names = FALSE
)

message("Sample-level multi-cellular axis summary:")
print(
  sample_axis %>%
    group_by(Disease_Identity) %>%
    summarise(
      n_samples = n(),
      mean_axis = mean(multicellular_axis_score, na.rm = TRUE),
      median_axis = median(multicellular_axis_score, na.rm = TRUE),
      sd_axis = sd(multicellular_axis_score, na.rm = TRUE),
      .groups = "drop"
    )
)

# ============================================================
# 6. Unsupervised composition PC1 axis for comparison
# ============================================================

pca <- prcomp(clr_mat, center = TRUE, scale. = TRUE)

pc1 <- pca$x[, 1]

pc1_score <- rescale01(pc1)

pc1_df <- celltype_fraction_wide %>%
  select(sample_id, Disease_Identity) %>%
  mutate(
    composition_PC1_raw = pc1,
    composition_PC1_score = pc1_score
  )

mean_control_pc1 <- mean(
  pc1_df$composition_PC1_score[pc1_df$Disease_Identity == "Control"],
  na.rm = TRUE
)

mean_ipf_pc1 <- mean(
  pc1_df$composition_PC1_score[pc1_df$Disease_Identity == "IPF"],
  na.rm = TRUE
)

if (mean_ipf_pc1 < mean_control_pc1) {
  pc1_df$composition_PC1_raw <- -pc1_df$composition_PC1_raw
  pc1_df$composition_PC1_score <- 1 - pc1_df$composition_PC1_score
}

pc_loadings_df <- data.frame(
  celltype_marker_based = rownames(pca$rotation),
  PC1_loading = pca$rotation[, 1]
) %>%
  arrange(desc(abs(PC1_loading)))

write.csv(
  pc1_df,
  file.path(tab_dir, "07_sample_unsupervised_composition_PC1_axis.csv"),
  row.names = FALSE
)

write.csv(
  pc_loadings_df,
  file.path(tab_dir, "08_unsupervised_composition_PC1_loadings.csv"),
  row.names = FALSE
)

# ============================================================
# 7. Cell type disease enrichment scores
# ============================================================

celltype_enrichment <- meta_use %>%
  count(Disease_Identity, celltype_marker_based, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(
    total_cells_disease = sum(n_cells),
    fraction = n_cells / total_cells_disease
  ) %>%
  ungroup() %>%
  select(Disease_Identity, celltype_marker_based, fraction, n_cells) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = c(fraction, n_cells),
    values_fill = 0
  )

# Add missing columns safely
for (col in c("fraction_Control", "fraction_COPD", "fraction_IPF")) {
  if (!col %in% colnames(celltype_enrichment)) {
    celltype_enrichment[[col]] <- 0
  }
}

celltype_enrichment <- celltype_enrichment %>%
  mutate(
    log2_IPF_vs_Control = log2((fraction_IPF + pseudocount) / (fraction_Control + pseudocount)),
    log2_IPF_vs_COPD = log2((fraction_IPF + pseudocount) / (fraction_COPD + pseudocount)),
    log2_COPD_vs_Control = log2((fraction_COPD + pseudocount) / (fraction_Control + pseudocount)),
    celltype_progression_score_raw = log2_IPF_vs_Control,
    celltype_progression_score = rescale01(celltype_progression_score_raw)
  ) %>%
  arrange(desc(celltype_progression_score_raw))

write.csv(
  celltype_enrichment,
  file.path(tab_dir, "09_celltype_disease_enrichment_scores.csv"),
  row.names = FALSE
)

# ============================================================
# 8. Cluster disease enrichment scores
# ============================================================

cluster_enrichment <- meta_use %>%
  count(Disease_Identity, cluster_default, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(
    total_cells_disease = sum(n_cells),
    fraction = n_cells / total_cells_disease
  ) %>%
  ungroup() %>%
  select(Disease_Identity, cluster_default, fraction, n_cells) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = c(fraction, n_cells),
    values_fill = 0
  )

for (col in c("fraction_Control", "fraction_COPD", "fraction_IPF")) {
  if (!col %in% colnames(cluster_enrichment)) {
    cluster_enrichment[[col]] <- 0
  }
}

cluster_enrichment <- cluster_enrichment %>%
  mutate(
    log2_IPF_vs_Control = log2((fraction_IPF + pseudocount) / (fraction_Control + pseudocount)),
    log2_IPF_vs_COPD = log2((fraction_IPF + pseudocount) / (fraction_COPD + pseudocount)),
    log2_COPD_vs_Control = log2((fraction_COPD + pseudocount) / (fraction_Control + pseudocount)),
    cluster_progression_score_raw = log2_IPF_vs_Control,
    cluster_progression_score = rescale01(cluster_progression_score_raw)
  ) %>%
  arrange(desc(cluster_progression_score_raw))

write.csv(
  cluster_enrichment,
  file.path(tab_dir, "10_cluster_disease_enrichment_scores.csv"),
  row.names = FALSE
)

# ============================================================
# 9. Map sample / cell type / cluster scores back to cells
# ============================================================

sample_score_map <- sample_axis %>%
  select(sample_id, multicellular_axis_score, multicellular_axis_stage)

celltype_score_map <- celltype_enrichment %>%
  select(
    celltype_marker_based,
    celltype_progression_score,
    celltype_progression_score_raw,
    log2_IPF_vs_Control,
    log2_IPF_vs_COPD
  )

cluster_score_map <- cluster_enrichment %>%
  select(
    cluster_default,
    cluster_progression_score,
    cluster_progression_score_raw,
    log2_IPF_vs_Control,
    log2_IPF_vs_COPD
  ) %>%
  rename(
    cluster_log2_IPF_vs_Control = log2_IPF_vs_Control,
    cluster_log2_IPF_vs_COPD = log2_IPF_vs_COPD
  )

meta_for_join <- meta

if ("CellBarcode_Identity" %in% colnames(meta_for_join)) {
  meta_for_join$cell_id <- as.character(meta_for_join$CellBarcode_Identity)
} else {
  meta_for_join$cell_id <- rownames(meta_for_join)
}

if (anyDuplicated(meta_for_join$cell_id) > 0) {
  stop("Duplicated cell_id detected before joining. Please check CellBarcode_Identity.")
}

meta_axis <- meta_for_join %>%
  left_join(sample_score_map, by = "sample_id") %>%
  left_join(celltype_score_map, by = "celltype_marker_based") %>%
  left_join(cluster_score_map, by = "cluster_default")

rownames(meta_axis) <- meta_axis$cell_id

# For samples excluded by min_cells, sample-level axis may be NA.
# Fill missing sample score with disease-level median where possible.
disease_median_score <- sample_axis %>%
  group_by(Disease_Identity) %>%
  summarise(
    disease_median_axis_score = median(multicellular_axis_score, na.rm = TRUE),
    .groups = "drop"
  )

meta_axis <- meta_axis %>%
  left_join(disease_median_score, by = "Disease_Identity") %>%
  mutate(
    multicellular_axis_score_filled = ifelse(
      is.na(multicellular_axis_score),
      disease_median_axis_score,
      multicellular_axis_score
    ),
    celltype_progression_score = ifelse(
      is.na(celltype_progression_score),
      0.5,
      celltype_progression_score
    ),
    cluster_progression_score = ifelse(
      is.na(cluster_progression_score),
      0.5,
      cluster_progression_score
    )
  )

meta_axis <- meta_axis %>%
  mutate(
    cell_multicellular_progression_score_raw =
      w_sample   * multicellular_axis_score_filled +
      w_celltype * celltype_progression_score +
      w_cluster  * cluster_progression_score,
    
    cell_multicellular_progression_score =
      rescale01(cell_multicellular_progression_score_raw),
    
    cell_multicellular_progression_stage =
      make_stage(cell_multicellular_progression_score)
  )

if (!"cell_id" %in% colnames(meta_axis)) {
  if ("CellBarcode_Identity" %in% colnames(meta_axis)) {
    meta_axis$cell_id <- as.character(meta_axis$CellBarcode_Identity)
  } else {
    stop("No valid cell_id or CellBarcode_Identity found in meta_axis.")
  }
}

rownames(meta_axis) <- meta_axis$cell_id

write.csv(
  meta_axis,
  file.path(tab_dir, "11_cell_metadata_with_multicellular_progression_axis.csv"),
  row.names = TRUE
)

saveRDS(
  meta_axis,
  file.path(tab_dir, "11_cell_metadata_with_multicellular_progression_axis.rds")
)

# ============================================================
# 10. Summaries by disease, cell type, major class, cluster
# ============================================================

disease_axis_summary <- meta_axis %>%
  group_by(Disease_Identity) %>%
  summarise(
    n_cells = n(),
    mean_cell_axis = mean(cell_multicellular_progression_score, na.rm = TRUE),
    median_cell_axis = median(cell_multicellular_progression_score, na.rm = TRUE),
    sd_cell_axis = sd(cell_multicellular_progression_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  disease_axis_summary,
  file.path(tab_dir, "12_disease_axis_summary.csv"),
  row.names = FALSE
)

celltype_axis_summary <- meta_axis %>%
  group_by(celltype_marker_based, Disease_Identity) %>%
  summarise(
    n_cells = n(),
    mean_cell_axis = mean(cell_multicellular_progression_score, na.rm = TRUE),
    median_cell_axis = median(cell_multicellular_progression_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  celltype_axis_summary,
  file.path(tab_dir, "13_celltype_axis_summary_by_disease.csv"),
  row.names = FALSE
)

major_axis_summary <- meta_axis %>%
  group_by(CellType_Category, Disease_Identity) %>%
  summarise(
    n_cells = n(),
    mean_cell_axis = mean(cell_multicellular_progression_score, na.rm = TRUE),
    median_cell_axis = median(cell_multicellular_progression_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  major_axis_summary,
  file.path(tab_dir, "14_major_class_axis_summary_by_disease.csv"),
  row.names = FALSE
)

cluster_axis_summary <- meta_axis %>%
  group_by(cluster_default, Disease_Identity) %>%
  summarise(
    n_cells = n(),
    mean_cell_axis = mean(cell_multicellular_progression_score, na.rm = TRUE),
    median_cell_axis = median(cell_multicellular_progression_score, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(
  cluster_axis_summary,
  file.path(tab_dir, "15_cluster_axis_summary_by_disease.csv"),
  row.names = FALSE
)

# Overall cluster annotation summary
cluster_annotation_summary <- meta_axis %>%
  group_by(cluster_default, celltype_marker_based, CellType_Category) %>%
  summarise(
    n_cells = n(),
    mean_cell_axis = mean(cell_multicellular_progression_score, na.rm = TRUE),
    median_cell_axis = median(cell_multicellular_progression_score, na.rm = TRUE),
    fraction_IPF = mean(Disease_Identity == "IPF", na.rm = TRUE),
    fraction_Control = mean(Disease_Identity == "Control", na.rm = TRUE),
    fraction_COPD = mean(Disease_Identity == "COPD", na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_cell_axis))

write.csv(
  cluster_annotation_summary,
  file.path(tab_dir, "16_cluster_annotation_progression_summary.csv"),
  row.names = FALSE
)

# ============================================================
# 11. Plots: sample-level axis
# ============================================================

p <- ggplot(
  sample_axis,
  aes(x = Disease_Identity, y = multicellular_axis_score, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.8) +
  theme_classic(base_size = 13) +
  labs(
    title = "Sample-level multi-cellular disease progression axis",
    x = NULL,
    y = "Multi-cellular progression score"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "01_sample_multicellular_axis_by_disease.pdf"),
  p,
  width = 5,
  height = 4
)

p <- ggplot(
  sample_axis,
  aes(
    x = reorder(sample_id, multicellular_axis_score),
    y = multicellular_axis_score,
    fill = Disease_Identity
  )
) +
  geom_col(width = 0.8) +
  theme_classic(base_size = 11) +
  labs(
    title = "Ranked samples along multi-cellular progression axis",
    x = "Sample",
    y = "Multi-cellular progression score"
  ) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )

ggsave(
  file.path(fig_dir, "02_ranked_samples_multicellular_axis.pdf"),
  p,
  width = 16,
  height = 5,
  limitsize = FALSE
)

# ============================================================
# 12. Plots: cell type weights and enrichment
# ============================================================

top_n_weight <- min(20, nrow(axis_weight_df))

top_weight_df <- axis_weight_df %>%
  arrange(desc(abs(axis_weight_raw))) %>%
  slice_head(n = top_n_weight)

p <- ggplot(
  top_weight_df,
  aes(
    x = reorder(celltype_marker_based, axis_weight_raw),
    y = axis_weight_raw,
    fill = axis_weight_raw > 0
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "Top cell type weights for multi-cellular IPF axis",
    x = NULL,
    y = "Axis weight: IPF - Control"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "03_top_celltype_axis_weights.pdf"),
  p,
  width = 7,
  height = 6
)

p <- ggplot(
  celltype_enrichment,
  aes(
    x = reorder(celltype_marker_based, log2_IPF_vs_Control),
    y = log2_IPF_vs_Control,
    fill = log2_IPF_vs_Control > 0
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "Cell type enrichment along IPF progression",
    x = NULL,
    y = "log2(IPF fraction / Control fraction)"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "04_celltype_IPF_vs_Control_enrichment.pdf"),
  p,
  width = 7,
  height = 6
)

# ============================================================
# 13. Plots: cluster enrichment
# ============================================================

p <- ggplot(
  cluster_enrichment,
  aes(
    x = reorder(as.character(cluster_default), log2_IPF_vs_Control),
    y = log2_IPF_vs_Control,
    fill = log2_IPF_vs_Control > 0
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "Cluster enrichment along IPF progression",
    x = "Cluster",
    y = "log2(IPF fraction / Control fraction)"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "05_cluster_IPF_vs_Control_enrichment.pdf"),
  p,
  width = 6,
  height = 7
)

# ============================================================
# 14. UMAP plots using metadata + UMAP
# ============================================================

if (!is.null(umap)) {
  umap_df <- as.data.frame(umap)
  colnames(umap_df)[1:2] <- c("UMAP_1", "UMAP_2")
  umap_df$cell_id <- rownames(umap_df)
  
  meta_axis_plot <- meta_axis
  
  if (!"cell_id" %in% colnames(meta_axis_plot)) {
    meta_axis_plot$cell_id <- rownames(meta_axis_plot)
  }
  
  meta_axis_plot <- meta_axis_plot %>%
    select(
      cell_id,
      Disease_Identity,
      CellType_Category,
      celltype_marker_based,
      cluster_default,
      cell_multicellular_progression_score,
      cell_multicellular_progression_stage
    )
  
  plot_df <- umap_df %>%
    left_join(
      meta_axis_plot,
      by = "cell_id"
    )
  
  p <- ggplot(
    plot_df,
    aes(
      x = UMAP_1,
      y = UMAP_2,
      color = cell_multicellular_progression_score
    )
  ) +
    geom_point(size = 0.05, alpha = 0.8) +
    scale_color_viridis_c(option = "magma") +
    theme_classic(base_size = 12) +
    labs(
      title = "Cell-level multi-cellular progression score",
      color = "Progression\nscore"
    )
  
  ggsave(
    file.path(fig_dir, "06_umap_cell_multicellular_progression_score_viridis.pdf"),
    p,
    width = 7,
    height = 6,
    limitsize = FALSE
  )
  
  p <- ggplot(
    plot_df,
    aes(
      x = UMAP_1,
      y = UMAP_2,
      color = cell_multicellular_progression_stage
    )
  ) +
    geom_point(size = 0.05, alpha = 0.7) +
    theme_classic(base_size = 12) +
    labs(
      title = "Cell-level multi-cellular progression stages",
      color = "Stage"
    )
  
  ggsave(
    file.path(fig_dir, "07_umap_cell_multicellular_progression_stage.pdf"),
    p,
    width = 8,
    height = 6,
    limitsize = FALSE
  )
}

# ============================================================
# 15. Save compact axis object
# ============================================================

axis_object <- list(
  sample_axis = sample_axis,
  sample_info = sample_info,
  celltype_axis_weights = axis_weight_df,
  celltype_enrichment = celltype_enrichment,
  cluster_enrichment = cluster_enrichment,
  disease_axis_summary = disease_axis_summary,
  celltype_axis_summary = celltype_axis_summary,
  major_axis_summary = major_axis_summary,
  cluster_axis_summary = cluster_axis_summary,
  cluster_annotation_summary = cluster_annotation_summary,
  metadata_axis = meta_axis
)

saveRDS(
  axis_object,
  file.path(out_dir, "multicellular_disease_progression_axis_object.rds")
)

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "06_sample_multicellular_progression_axis.csv"))
message("  ", file.path(tab_dir, "11_cell_metadata_with_multicellular_progression_axis.csv"))
message("  ", file.path(tab_dir, "09_celltype_disease_enrichment_scores.csv"))
message("  ", file.path(tab_dir, "10_cluster_disease_enrichment_scores.csv"))
message("  ", file.path(out_dir, "multicellular_disease_progression_axis_object.rds"))