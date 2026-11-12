# 09b_epithelial_manual_annotation_and_disease_state.R
# Manual annotation and disease-state fraction analysis for epithelial subset

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(readr)
  library(forcats)
})

set.seed(123)

project_dir <- getwd()

in_obj <- file.path(
  project_dir,
  "results",
  "epithelial_subset",
  "objects",
  "epithelial_subset_final.rds"
)

out_dir <- file.path(project_dir, "results", "epithelial_disease_state")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
obj_dir <- file.path(out_dir, "objects")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(obj_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Load epithelial subset object
# ============================================================

seu_epi <- readRDS(in_obj)

DefaultAssay(seu_epi) <- "RNA"

if (!"epi_cluster_res06" %in% colnames(seu_epi@meta.data)) {
  stop("epi_cluster_res06 not found. Please run epithelial subset clustering first.")
}

seu_epi$Disease_Identity <- factor(
  seu_epi$Disease_Identity,
  levels = c("Control", "COPD", "IPF")
)

seu_epi$epi_cluster_res06 <- as.character(seu_epi$epi_cluster_res06)

message("Epithelial subset:")
print(dim(seu_epi))
print(table(seu_epi$epi_cluster_res06, useNA = "ifany"))
print(table(seu_epi$Disease_Identity, useNA = "ifany"))

# ============================================================
# 2. Manual epithelial subcluster annotation
# ============================================================
# Based on epithelial UMAP, DotPlot, markers, module scores, and disease enrichment.
# Cluster 9 and 13 are marked as boundary/contaminant and can be excluded from
# core epithelial disease-state analysis.

manual_map <- c(
  "0"  = "IPF_enriched_ciliated_epithelial",
  "1"  = "Ciliated_epithelial",
  "2"  = "Ciliated_epithelial",
  "3"  = "Secretory_AT2_adjacent_epithelial",
  "4"  = "Mature_AT2",
  "5"  = "AT2_like_activated_AT2",
  "6"  = "IPF_enriched_basal_like_epithelial",
  "7"  = "IPF_enriched_secretory_mucous_epithelial",
  "8"  = "AT1_like_epithelial",
  "9"  = "Stromal_like_boundary_or_contaminant",
  "10" = "IPF_enriched_aberrant_basaloid_fibrotic_epithelial",
  "11" = "Rare_AT2_like",
  "12" = "Rare_secretory_endocrine_like_epithelial",
  "13" = "Myeloid_like_contaminant"
)

seu_epi$epi_cellstate_manual <- unname(
  manual_map[as.character(seu_epi$epi_cluster_res06)]
)

if (any(is.na(seu_epi$epi_cellstate_manual))) {
  warning("Some epithelial clusters were not annotated:")
  print(table(seu_epi$epi_cluster_res06[is.na(seu_epi$epi_cellstate_manual)]))
}

# Broad epithelial disease-state grouping
broad_map <- c(
  "IPF_enriched_ciliated_epithelial" = "IPF_remodeled_airway_like",
  "Ciliated_epithelial" = "Ciliated_homeostatic_or_remodeled",
  "Secretory_AT2_adjacent_epithelial" = "Secretory_AT2_adjacent",
  "Mature_AT2" = "Homeostatic_AT2",
  "AT2_like_activated_AT2" = "AT2_like_activated",
  "IPF_enriched_basal_like_epithelial" = "IPF_basal_like",
  "IPF_enriched_secretory_mucous_epithelial" = "IPF_secretory_mucous",
  "AT1_like_epithelial" = "Homeostatic_AT1",
  "Stromal_like_boundary_or_contaminant" = "Boundary_or_contaminant",
  "IPF_enriched_aberrant_basaloid_fibrotic_epithelial" = "IPF_aberrant_fibrotic_epithelial",
  "Rare_AT2_like" = "Rare_AT2_like",
  "Rare_secretory_endocrine_like_epithelial" = "Rare_secretory_endocrine_like",
  "Myeloid_like_contaminant" = "Boundary_or_contaminant"
)

seu_epi$epi_state_broad <- unname(
  broad_map[seu_epi$epi_cellstate_manual]
)

# Analysis inclusion flag
seu_epi$epi_core_analysis_flag <- ifelse(
  seu_epi$epi_state_broad %in% c("Boundary_or_contaminant"),
  "exclude_boundary_or_contaminant",
  "include_core_epithelial"
)

# IPF pathology grouping
seu_epi$epi_pathology_group <- dplyr::case_when(
  seu_epi$epi_state_broad %in% c(
    "IPF_basal_like",
    "IPF_aberrant_fibrotic_epithelial",
    "IPF_secretory_mucous",
    "IPF_remodeled_airway_like"
  ) ~ "IPF_associated_remodeled_epithelial",
  
  seu_epi$epi_state_broad %in% c(
    "Homeostatic_AT1",
    "Homeostatic_AT2",
    "AT2_like_activated"
  ) ~ "alveolar_homeostatic_or_AT2_like",
  
  seu_epi$epi_state_broad %in% c(
    "Ciliated_homeostatic_or_remodeled",
    "Secretory_AT2_adjacent",
    "Rare_AT2_like",
    "Rare_secretory_endocrine_like"
  ) ~ "other_epithelial_states",
  
  seu_epi$epi_state_broad == "Boundary_or_contaminant" ~ "boundary_or_contaminant",
  
  TRUE ~ "unassigned"
)

message("Manual epithelial states:")
print(table(seu_epi$epi_cellstate_manual, useNA = "ifany"))

message("Broad epithelial states:")
print(table(seu_epi$epi_state_broad, useNA = "ifany"))

message("Core analysis flag:")
print(table(seu_epi$epi_core_analysis_flag, useNA = "ifany"))

message("Pathology group:")
print(table(seu_epi$epi_pathology_group, useNA = "ifany"))

# Save metadata with manual annotations
write.csv(
  seu_epi@meta.data,
  file.path(tab_dir, "01_epithelial_metadata_manual_annotation.csv")
)

# ============================================================
# 3. Manual annotation UMAP plots
# ============================================================

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "epi_cellstate_manual",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) +
  NoLegend() +
  ggtitle("Manual epithelial cell-state annotation")

ggsave(
  file.path(fig_dir, "01_umap_epi_cellstate_manual.pdf"),
  p,
  width = 10,
  height = 8,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "epi_state_broad",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) +
  NoLegend() +
  ggtitle("Broad epithelial state annotation")

ggsave(
  file.path(fig_dir, "02_umap_epi_state_broad.pdf"),
  p,
  width = 10,
  height = 8,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "epi_pathology_group",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) +
  ggtitle("Epithelial pathology group")

ggsave(
  file.path(fig_dir, "03_umap_epi_pathology_group.pdf"),
  p,
  width = 9,
  height = 7,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "Disease_Identity",
  raster = TRUE
) +
  ggtitle("Epithelial subset by disease")

ggsave(
  file.path(fig_dir, "04_umap_disease_identity.pdf"),
  p,
  width = 8,
  height = 6,
  limitsize = FALSE
)

# ============================================================
# 4. Core epithelial object metadata
# ============================================================

meta_all <- seu_epi@meta.data

meta_core <- meta_all %>%
  filter(epi_core_analysis_flag == "include_core_epithelial")

write.csv(
  meta_core,
  file.path(tab_dir, "02_epithelial_core_metadata_excluding_boundary_contaminants.csv")
)

message("Core epithelial cells by disease:")
print(table(meta_core$Disease_Identity, useNA = "ifany"))

message("Excluded boundary/contaminant cells:")
print(table(meta_all$epi_cellstate_manual[meta_all$epi_core_analysis_flag != "include_core_epithelial"]))

# ============================================================
# 5. Disease composition of epithelial states
# ============================================================

state_disease_count <- meta_core %>%
  count(Disease_Identity, epi_cellstate_manual, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction_within_disease = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  state_disease_count,
  file.path(tab_dir, "03_epi_cellstate_fraction_by_disease.csv"),
  row.names = FALSE
)

broad_state_disease_count <- meta_core %>%
  count(Disease_Identity, epi_state_broad, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction_within_disease = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  broad_state_disease_count,
  file.path(tab_dir, "04_epi_broad_state_fraction_by_disease.csv"),
  row.names = FALSE
)

pathology_group_disease_count <- meta_core %>%
  count(Disease_Identity, epi_pathology_group, name = "n_cells") %>%
  group_by(Disease_Identity) %>%
  mutate(fraction_within_disease = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  pathology_group_disease_count,
  file.path(tab_dir, "05_epi_pathology_group_fraction_by_disease.csv"),
  row.names = FALSE
)

# Stacked bar plots
p <- ggplot(
  state_disease_count,
  aes(x = Disease_Identity, y = fraction_within_disease, fill = epi_cellstate_manual)
) +
  geom_col(width = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "Epithelial manual cell-state composition by disease",
    x = NULL,
    y = "Fraction within epithelial compartment",
    fill = "Manual epithelial state"
  )

ggsave(
  file.path(fig_dir, "05_epi_cellstate_fraction_by_disease_stacked.pdf"),
  p,
  width = 10,
  height = 6,
  limitsize = FALSE
)

p <- ggplot(
  broad_state_disease_count,
  aes(x = Disease_Identity, y = fraction_within_disease, fill = epi_state_broad)
) +
  geom_col(width = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "Broad epithelial state composition by disease",
    x = NULL,
    y = "Fraction within epithelial compartment",
    fill = "Broad epithelial state"
  )

ggsave(
  file.path(fig_dir, "06_epi_broad_state_fraction_by_disease_stacked.pdf"),
  p,
  width = 9,
  height = 5,
  limitsize = FALSE
)

p <- ggplot(
  pathology_group_disease_count,
  aes(x = Disease_Identity, y = fraction_within_disease, fill = epi_pathology_group)
) +
  geom_col(width = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "Epithelial pathology group composition by disease",
    x = NULL,
    y = "Fraction within epithelial compartment",
    fill = "Pathology group"
  )

ggsave(
  file.path(fig_dir, "07_epi_pathology_group_fraction_by_disease_stacked.pdf"),
  p,
  width = 8,
  height = 5,
  limitsize = FALSE
)

# ============================================================
# 6. Sample-level epithelial state fractions
# ============================================================

state_sample_frac <- meta_core %>%
  count(sample_id, Disease_Identity, epi_cellstate_manual, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  state_sample_frac,
  file.path(tab_dir, "06_epi_cellstate_fraction_by_sample.csv"),
  row.names = FALSE
)

broad_state_sample_frac <- meta_core %>%
  count(sample_id, Disease_Identity, epi_state_broad, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  broad_state_sample_frac,
  file.path(tab_dir, "07_epi_broad_state_fraction_by_sample.csv"),
  row.names = FALSE
)

pathology_group_sample_frac <- meta_core %>%
  count(sample_id, Disease_Identity, epi_pathology_group, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  pathology_group_sample_frac,
  file.path(tab_dir, "08_epi_pathology_group_fraction_by_sample.csv"),
  row.names = FALSE
)

# Sample-level boxplots
p <- ggplot(
  state_sample_frac,
  aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.7, alpha = 0.65) +
  facet_wrap(~ epi_cellstate_manual, scales = "free_y", ncol = 4) +
  theme_classic(base_size = 10) +
  labs(
    title = "Sample-level epithelial manual cell-state fractions",
    x = NULL,
    y = "Fraction within epithelial compartment"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "08_epi_cellstate_fraction_by_sample_boxplot.pdf"),
  p,
  width = 14,
  height = 10,
  limitsize = FALSE
)

p <- ggplot(
  broad_state_sample_frac,
  aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.9, alpha = 0.7) +
  facet_wrap(~ epi_state_broad, scales = "free_y", ncol = 3) +
  theme_classic(base_size = 11) +
  labs(
    title = "Sample-level broad epithelial state fractions",
    x = NULL,
    y = "Fraction within epithelial compartment"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "09_epi_broad_state_fraction_by_sample_boxplot.pdf"),
  p,
  width = 12,
  height = 8,
  limitsize = FALSE
)

p <- ggplot(
  pathology_group_sample_frac,
  aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 1.0, alpha = 0.75) +
  facet_wrap(~ epi_pathology_group, scales = "free_y") +
  theme_classic(base_size = 12) +
  labs(
    title = "Sample-level epithelial pathology group fractions",
    x = NULL,
    y = "Fraction within epithelial compartment"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "10_epi_pathology_group_fraction_by_sample_boxplot.pdf"),
  p,
  width = 10,
  height = 6,
  limitsize = FALSE
)

# ============================================================
# 7. Pairwise Wilcoxon tests using sample as unit
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
        mean_g1 <- mean(x, na.rm = TRUE)
        mean_g2 <- mean(y, na.rm = TRUE)
        delta_median <- median_g1 - median_g2
        delta_mean <- mean_g1 - mean_g2
      } else {
        pval <- NA
        median_g1 <- ifelse(length(x) > 0, median(x, na.rm = TRUE), NA)
        median_g2 <- ifelse(length(y) > 0, median(y, na.rm = TRUE), NA)
        mean_g1 <- ifelse(length(x) > 0, mean(x, na.rm = TRUE), NA)
        mean_g2 <- ifelse(length(y) > 0, mean(y, na.rm = TRUE), NA)
        delta_median <- NA
        delta_mean <- NA
      }
      
      res[[idx]] <- data.frame(
        state = tp,
        comparison = paste0(g1, "_vs_", g2),
        group_1 = g1,
        group_2 = g2,
        median_group_1 = median_g1,
        median_group_2 = median_g2,
        mean_group_1 = mean_g1,
        mean_group_2 = mean_g2,
        median_delta = delta_median,
        mean_delta = delta_mean,
        p_value = pval
      )
      
      idx <- idx + 1
    }
  }
  
  res_df <- bind_rows(res) %>%
    group_by(comparison) %>%
    mutate(p_adj_BH = p.adjust(p_value, method = "BH")) %>%
    ungroup() %>%
    arrange(comparison, p_adj_BH, desc(median_delta))
  
  return(res_df)
}

state_stats <- pairwise_wilcox_by_type(
  frac_df = state_sample_frac,
  type_col = "epi_cellstate_manual"
)

write.csv(
  state_stats,
  file.path(tab_dir, "09_epi_cellstate_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

broad_state_stats <- pairwise_wilcox_by_type(
  frac_df = broad_state_sample_frac,
  type_col = "epi_state_broad"
)

write.csv(
  broad_state_stats,
  file.path(tab_dir, "10_epi_broad_state_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

pathology_group_stats <- pairwise_wilcox_by_type(
  frac_df = pathology_group_sample_frac,
  type_col = "epi_pathology_group"
)

write.csv(
  pathology_group_stats,
  file.path(tab_dir, "11_epi_pathology_group_fraction_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

# Candidate IPF-enriched epithelial disease states
candidate_ipf_states <- state_stats %>%
  filter(comparison %in% c("IPF_vs_Control", "IPF_vs_COPD")) %>%
  arrange(comparison, p_adj_BH, desc(median_delta))

write.csv(
  candidate_ipf_states,
  file.path(tab_dir, "12_candidate_IPF_enriched_manual_epithelial_states.csv"),
  row.names = FALSE
)

candidate_ipf_broad_states <- broad_state_stats %>%
  filter(comparison %in% c("IPF_vs_Control", "IPF_vs_COPD")) %>%
  arrange(comparison, p_adj_BH, desc(median_delta))

write.csv(
  candidate_ipf_broad_states,
  file.path(tab_dir, "13_candidate_IPF_enriched_broad_epithelial_states.csv"),
  row.names = FALSE
)

candidate_ipf_pathology_groups <- pathology_group_stats %>%
  filter(comparison %in% c("IPF_vs_Control", "IPF_vs_COPD")) %>%
  arrange(comparison, p_adj_BH, desc(median_delta))

write.csv(
  candidate_ipf_pathology_groups,
  file.path(tab_dir, "14_candidate_IPF_enriched_epithelial_pathology_groups.csv"),
  row.names = FALSE
)

# ============================================================
# 8. Observed / expected enrichment for disease-state relation
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

state_enrichment <- compute_enrichment(
  df = meta_core,
  group_col = "Disease_Identity",
  type_col = "epi_cellstate_manual"
)

write.csv(
  state_enrichment,
  file.path(tab_dir, "15_epi_cellstate_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

broad_state_enrichment <- compute_enrichment(
  df = meta_core,
  group_col = "Disease_Identity",
  type_col = "epi_state_broad"
)

write.csv(
  broad_state_enrichment,
  file.path(tab_dir, "16_epi_broad_state_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

pathology_group_enrichment <- compute_enrichment(
  df = meta_core,
  group_col = "Disease_Identity",
  type_col = "epi_pathology_group"
)

write.csv(
  pathology_group_enrichment,
  file.path(tab_dir, "17_epi_pathology_group_disease_enrichment_observed_expected.csv"),
  row.names = FALSE
)

# Heatmaps
p <- ggplot(
  state_enrichment,
  aes(x = Disease_Identity, y = fct_reorder(epi_cellstate_manual, log2_observed_expected), fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 11) +
  labs(
    title = "Disease enrichment of manual epithelial states",
    x = NULL,
    y = "Manual epithelial state",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "11_epi_cellstate_disease_enrichment_heatmap.pdf"),
  p,
  width = 7,
  height = 8,
  limitsize = FALSE
)

p <- ggplot(
  broad_state_enrichment,
  aes(x = Disease_Identity, y = fct_reorder(epi_state_broad, log2_observed_expected), fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 11) +
  labs(
    title = "Disease enrichment of broad epithelial states",
    x = NULL,
    y = "Broad epithelial state",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "12_epi_broad_state_disease_enrichment_heatmap.pdf"),
  p,
  width = 7,
  height = 6,
  limitsize = FALSE
)

p <- ggplot(
  pathology_group_enrichment,
  aes(x = Disease_Identity, y = fct_reorder(epi_pathology_group, log2_observed_expected), fill = log2_observed_expected)
) +
  geom_tile(color = "white") +
  theme_classic(base_size = 12) +
  labs(
    title = "Disease enrichment of epithelial pathology groups",
    x = NULL,
    y = "Pathology group",
    fill = "log2(O/E)"
  )

ggsave(
  file.path(fig_dir, "13_epi_pathology_group_disease_enrichment_heatmap.pdf"),
  p,
  width = 7,
  height = 5,
  limitsize = FALSE
)

# ============================================================
# 9. Module score summaries by manual states and disease
# ============================================================

score_cols <- grep("_score$", colnames(meta_all), value = TRUE)

if (length(score_cols) > 0) {
  module_by_state <- meta_core %>%
    group_by(epi_cellstate_manual) %>%
    summarise(
      n_cells = n(),
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_state,
    file.path(tab_dir, "18_module_scores_mean_by_epi_cellstate_manual.csv"),
    row.names = FALSE
  )
  
  module_by_broad_state <- meta_core %>%
    group_by(epi_state_broad) %>%
    summarise(
      n_cells = n(),
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_broad_state,
    file.path(tab_dir, "19_module_scores_mean_by_epi_state_broad.csv"),
    row.names = FALSE
  )
  
  module_by_state_disease <- meta_core %>%
    group_by(epi_cellstate_manual, Disease_Identity) %>%
    summarise(
      n_cells = n(),
      across(all_of(score_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  write.csv(
    module_by_state_disease,
    file.path(tab_dir, "20_module_scores_mean_by_epi_cellstate_and_disease.csv"),
    row.names = FALSE
  )
  
  module_long <- meta_core %>%
    select(Disease_Identity, epi_pathology_group, all_of(score_cols)) %>%
    pivot_longer(
      cols = all_of(score_cols),
      names_to = "module",
      values_to = "score"
    ) %>%
    mutate(module = gsub("_score$", "", module))
  
  p <- ggplot(
    module_long,
    aes(x = epi_pathology_group, y = score, fill = epi_pathology_group)
  ) +
    geom_boxplot(outlier.shape = NA, width = 0.65) +
    facet_wrap(~ module, scales = "free_y", ncol = 4) +
    theme_classic(base_size = 10) +
    labs(
      title = "Module scores across epithelial pathology groups",
      x = NULL,
      y = "Module score"
    ) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  ggsave(
    file.path(fig_dir, "14_module_scores_by_epithelial_pathology_group.pdf"),
    p,
    width = 14,
    height = 10,
    limitsize = FALSE
  )
  
  p <- ggplot(
    module_long,
    aes(x = Disease_Identity, y = score, fill = Disease_Identity)
  ) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    facet_grid(module ~ epi_pathology_group, scales = "free_y") +
    theme_classic(base_size = 8) +
    labs(
      title = "Module scores by disease within epithelial pathology groups",
      x = NULL,
      y = "Module score"
    ) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text.y = element_text(angle = 0)
    )
  
  ggsave(
    file.path(fig_dir, "15_module_scores_by_disease_within_pathology_group.pdf"),
    p,
    width = 16,
    height = 18,
    limitsize = FALSE
  )
}

# ============================================================
# 10. Disease-state score per sample
# ============================================================
# This creates compact epithelial scores that can be used later for
# disease progression axis construction.

pathology_wide <- pathology_group_sample_frac %>%
  select(sample_id, Disease_Identity, epi_pathology_group, fraction) %>%
  pivot_wider(
    names_from = epi_pathology_group,
    values_from = fraction,
    values_fill = 0
  )

# Make sure columns exist
needed_cols <- c(
  "IPF_associated_remodeled_epithelial",
  "alveolar_homeostatic_or_AT2_like",
  "other_epithelial_states"
)

for (cc in needed_cols) {
  if (!cc %in% colnames(pathology_wide)) {
    pathology_wide[[cc]] <- 0
  }
}

pathology_wide <- pathology_wide %>%
  mutate(
    epithelial_remodeling_fraction = IPF_associated_remodeled_epithelial,
    epithelial_homeostatic_fraction = alveolar_homeostatic_or_AT2_like,
    epithelial_remodeling_index = epithelial_remodeling_fraction - epithelial_homeostatic_fraction,
    epithelial_remodeling_ratio = (epithelial_remodeling_fraction + 1e-6) /
      (epithelial_homeostatic_fraction + 1e-6),
    epithelial_remodeling_log2_ratio = log2(epithelial_remodeling_ratio)
  )

write.csv(
  pathology_wide,
  file.path(tab_dir, "21_sample_level_epithelial_disease_state_scores.csv"),
  row.names = FALSE
)

p <- ggplot(
  pathology_wide,
  aes(x = Disease_Identity, y = epithelial_remodeling_index, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 1.1, alpha = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "Sample-level epithelial remodeling index",
    x = NULL,
    y = "Remodeled fraction - homeostatic fraction"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "16_sample_level_epithelial_remodeling_index.pdf"),
  p,
  width = 6,
  height = 5,
  limitsize = FALSE
)

p <- ggplot(
  pathology_wide,
  aes(x = Disease_Identity, y = epithelial_remodeling_log2_ratio, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 1.1, alpha = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "Sample-level epithelial remodeling log2 ratio",
    x = NULL,
    y = "log2((remodeled + eps) / (homeostatic + eps))"
  ) +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "17_sample_level_epithelial_remodeling_log2_ratio.pdf"),
  p,
  width = 6,
  height = 5,
  limitsize = FALSE
)

# Pairwise test for remodeling index
pairwise_numeric_test <- function(df, value_col) {
  pairs <- list(
    c("IPF", "Control"),
    c("IPF", "COPD"),
    c("COPD", "Control")
  )
  
  res <- list()
  
  for (i in seq_along(pairs)) {
    g1 <- pairs[[i]][1]
    g2 <- pairs[[i]][2]
    
    x <- df %>% filter(Disease_Identity == g1) %>% pull(.data[[value_col]])
    y <- df %>% filter(Disease_Identity == g2) %>% pull(.data[[value_col]])
    
    if (length(x) >= 2 && length(y) >= 2) {
      wt <- suppressWarnings(wilcox.test(x, y, exact = FALSE))
      pval <- wt$p.value
    } else {
      pval <- NA
    }
    
    res[[i]] <- data.frame(
      metric = value_col,
      comparison = paste0(g1, "_vs_", g2),
      group_1 = g1,
      group_2 = g2,
      median_group_1 = ifelse(length(x) > 0, median(x, na.rm = TRUE), NA),
      median_group_2 = ifelse(length(y) > 0, median(y, na.rm = TRUE), NA),
      mean_group_1 = ifelse(length(x) > 0, mean(x, na.rm = TRUE), NA),
      mean_group_2 = ifelse(length(y) > 0, mean(y, na.rm = TRUE), NA),
      p_value = pval
    )
  }
  
  bind_rows(res) %>%
    mutate(p_adj_BH = p.adjust(p_value, method = "BH"))
}

remodeling_index_stats <- bind_rows(
  pairwise_numeric_test(pathology_wide, "epithelial_remodeling_index"),
  pairwise_numeric_test(pathology_wide, "epithelial_remodeling_log2_ratio"),
  pairwise_numeric_test(pathology_wide, "epithelial_remodeling_fraction"),
  pairwise_numeric_test(pathology_wide, "epithelial_homeostatic_fraction")
)

write.csv(
  remodeling_index_stats,
  file.path(tab_dir, "22_epithelial_remodeling_score_pairwise_wilcox_sample_level.csv"),
  row.names = FALSE
)

# ============================================================
# 11. Final summary tables
# ============================================================

state_count_total <- meta_core %>%
  count(epi_cellstate_manual, name = "n_cells_total")

state_disease_wide <- state_disease_count %>%
  select(Disease_Identity, epi_cellstate_manual, fraction_within_disease) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = fraction_within_disease,
    names_prefix = "fraction_"
  )

state_enrichment_wide <- state_enrichment %>%
  select(Disease_Identity, epi_cellstate_manual, log2_observed_expected) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = log2_observed_expected,
    names_prefix = "log2OE_"
  )

state_summary <- state_count_total %>%
  left_join(state_disease_wide, by = "epi_cellstate_manual") %>%
  left_join(state_enrichment_wide, by = "epi_cellstate_manual") %>%
  arrange(desc(fraction_IPF))

write.csv(
  state_summary,
  file.path(tab_dir, "23_epithelial_manual_state_summary_for_interpretation.csv"),
  row.names = FALSE
)

broad_state_count_total <- meta_core %>%
  count(epi_state_broad, name = "n_cells_total")

broad_state_disease_wide <- broad_state_disease_count %>%
  select(Disease_Identity, epi_state_broad, fraction_within_disease) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = fraction_within_disease,
    names_prefix = "fraction_"
  )

broad_state_enrichment_wide <- broad_state_enrichment %>%
  select(Disease_Identity, epi_state_broad, log2_observed_expected) %>%
  pivot_wider(
    names_from = Disease_Identity,
    values_from = log2_observed_expected,
    names_prefix = "log2OE_"
  )

broad_state_summary <- broad_state_count_total %>%
  left_join(broad_state_disease_wide, by = "epi_state_broad") %>%
  left_join(broad_state_enrichment_wide, by = "epi_state_broad") %>%
  arrange(desc(fraction_IPF))

write.csv(
  broad_state_summary,
  file.path(tab_dir, "24_epithelial_broad_state_summary_for_interpretation.csv"),
  row.names = FALSE
)

# ============================================================
# 12. Save final object and compact analysis outputs
# ============================================================

saveRDS(
  seu_epi,
  file.path(obj_dir, "epithelial_subset_manual_annotated.rds")
)

analysis_outputs <- list(
  state_disease_count = state_disease_count,
  broad_state_disease_count = broad_state_disease_count,
  pathology_group_disease_count = pathology_group_disease_count,
  state_sample_frac = state_sample_frac,
  broad_state_sample_frac = broad_state_sample_frac,
  pathology_group_sample_frac = pathology_group_sample_frac,
  state_stats = state_stats,
  broad_state_stats = broad_state_stats,
  pathology_group_stats = pathology_group_stats,
  state_enrichment = state_enrichment,
  broad_state_enrichment = broad_state_enrichment,
  pathology_group_enrichment = pathology_group_enrichment,
  pathology_wide = pathology_wide,
  remodeling_index_stats = remodeling_index_stats,
  state_summary = state_summary,
  broad_state_summary = broad_state_summary
)

saveRDS(
  analysis_outputs,
  file.path(out_dir, "epithelial_disease_state_analysis_outputs.rds")
)

message("Done.")
message("Manual epithelial annotation and disease-state fraction analysis finished.")
message("Main output directory:")
message(out_dir)

message("Key tables:")
message(file.path(tab_dir, "01_epithelial_metadata_manual_annotation.csv"))
message(file.path(tab_dir, "06_epi_cellstate_fraction_by_sample.csv"))
message(file.path(tab_dir, "09_epi_cellstate_fraction_pairwise_wilcox_sample_level.csv"))
message(file.path(tab_dir, "12_candidate_IPF_enriched_manual_epithelial_states.csv"))
message(file.path(tab_dir, "21_sample_level_epithelial_disease_state_scores.csv"))
message(file.path(tab_dir, "22_epithelial_remodeling_score_pairwise_wilcox_sample_level.csv"))
message(file.path(tab_dir, "23_epithelial_manual_state_summary_for_interpretation.csv"))

message("Key figures:")
message(file.path(fig_dir, "01_umap_epi_cellstate_manual.pdf"))
message(file.path(fig_dir, "08_epi_cellstate_fraction_by_sample_boxplot.pdf"))
message(file.path(fig_dir, "11_epi_cellstate_disease_enrichment_heatmap.pdf"))
message(file.path(fig_dir, "16_sample_level_epithelial_remodeling_index.pdf"))