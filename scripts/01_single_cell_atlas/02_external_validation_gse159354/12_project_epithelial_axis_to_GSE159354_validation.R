# 11_project_epithelial_axis_to_GSE159354_validation.R
# Project epithelial injury-remodeling axis to GSE159354 validation dataset
#
# Input:
#   results_GSE159354/GSE159354_preprocessed_integrated_seurat.rds
#
# Optional input:
#   results_GSE159354/metadata/GSE159354_sample_metadata_for_validation.csv
#   Required columns if available:
#     sample_id, Disease_Identity
#
# Output:
#   results_GSE159354_epithelial_axis_validation/

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(Matrix)
  library(readr)
  library(forcats)
})

set.seed(123)

project_dir <- getwd()

in_obj <- file.path(
  project_dir,
  "results_GSE159354",
  "GSE159354_preprocessed_integrated_seurat.rds"
)

sample_meta_file <- file.path(
  project_dir,
  "results_GSE159354",
  "metadata",
  "GSE159354_sample_metadata_for_validation.csv"
)

out_dir <- file.path(project_dir, "results_GSE159354_epithelial_axis_validation")
fig_dir <- file.path(out_dir, "figures")
tab_dir <- file.path(out_dir, "tables")
obj_dir <- file.path(out_dir, "objects")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(obj_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Load GSE159354 preprocessed object
# ============================================================

if (!file.exists(in_obj)) {
  stop("Cannot find input object: ", in_obj)
}

seu <- readRDS(in_obj)

DefaultAssay(seu) <- "RNA"

message("Loaded GSE159354 object:")
print(dim(seu))
print(colnames(seu@meta.data))

if (!"sample_id" %in% colnames(seu@meta.data)) {
  stop("sample_id column not found in metadata.")
}

if (!"celltype_marker_based" %in% colnames(seu@meta.data)) {
  stop("celltype_marker_based column not found. Please run marker-based annotation first.")
}

# Optional disease metadata
if (file.exists(sample_meta_file)) {
  message("Found sample metadata:")
  message(sample_meta_file)
  
  # 注意：这里不要让 read.csv 自动把字符串 "NA" 转成真正 NA
  sample_meta <- read.csv(
    sample_meta_file,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA_real_missing_value")
  )
  
  if (!all(c("sample_id", "Disease_Identity") %in% colnames(sample_meta))) {
    stop("Sample metadata must contain columns: sample_id, Disease_Identity")
  }
  
  sample_meta <- sample_meta %>%
    transmute(
      sample_id = trimws(as.character(sample_id)),
      Disease_Identity = trimws(as.character(Disease_Identity))
    ) %>%
    mutate(
      Disease_Identity = dplyr::case_when(
        Disease_Identity %in% c("Control", "control", "CTRL", "Ctrl", "Healthy", "Normal") ~ "Control",
        Disease_Identity %in% c("IPF", "ipf", "Idiopathic pulmonary fibrosis") ~ "IPF",
        Disease_Identity %in% c("COPD", "copd") ~ "COPD",
        Disease_Identity %in% c("", "Unknown", "unknown", "NA", "NaN", "nan") ~ NA_character_,
        TRUE ~ Disease_Identity
      )
    ) %>%
    distinct()
  
  # 检查 metadata 自身是否还有未补齐标签
  if (any(is.na(sample_meta$Disease_Identity))) {
    message("Samples with missing Disease_Identity in sample metadata:")
    print(sample_meta %>% filter(is.na(Disease_Identity)))
    stop("Please fix missing Disease_Identity in sample metadata before continuing.")
  }
  
  # 检查 sample_id 是否重复且对应多个疾病标签
  duplicated_meta <- sample_meta %>%
    count(sample_id) %>%
    filter(n > 1)
  
  if (nrow(duplicated_meta) > 0) {
    message("Duplicated sample_id in sample metadata:")
    print(duplicated_meta)
    stop("Duplicated sample_id found in sample metadata.")
  }
  
  # 清理 Seurat metadata 中的 sample_id
  seu@meta.data$sample_id <- trimws(as.character(seu@meta.data$sample_id))
  
  # 关键：先删除旧的 Disease_Identity，避免 left_join 生成 Disease_Identity.x / Disease_Identity.y
  seu@meta.data <- seu@meta.data %>%
    select(-any_of(c("Disease_Identity", "Disease_Identity.x", "Disease_Identity.y")))
  
  # 检查 Seurat 对象中哪些 sample_id 没有在 sample_meta 里匹配到
  seurat_samples <- data.frame(
    sample_id = sort(unique(seu@meta.data$sample_id)),
    stringsAsFactors = FALSE
  )
  
  unmatched_samples <- anti_join(
    seurat_samples,
    sample_meta,
    by = "sample_id"
  )
  
  if (nrow(unmatched_samples) > 0) {
    message("These sample_id values are present in Seurat object but missing from sample metadata:")
    print(unmatched_samples)
    stop("Sample metadata does not cover all sample_id values in Seurat object.")
  }
  
  # 合并疾病标签
  seu@meta.data <- seu@meta.data %>%
    tibble::rownames_to_column("cell_id") %>%
    left_join(sample_meta, by = "sample_id") %>%
    tibble::column_to_rownames("cell_id")
  
  if (any(is.na(seu@meta.data$Disease_Identity))) {
    message("Cells with missing Disease_Identity after merge:")
    print(head(seu@meta.data[is.na(seu@meta.data$Disease_Identity), c("sample_id", "Disease_Identity")]))
    stop("Disease_Identity still contains NA after merge.")
  }
  
} else {
  stop("External sample metadata not found: ", sample_meta_file)
}

seu$Disease_Identity <- trimws(as.character(seu$Disease_Identity))

disease_levels <- c("Control", "IPF", "SSc-ILD", "COPD")

disease_levels <- c(
  disease_levels[disease_levels %in% unique(seu$Disease_Identity)],
  setdiff(sort(unique(seu$Disease_Identity)), disease_levels)
)

seu$Disease_Identity <- factor(
  seu$Disease_Identity,
  levels = disease_levels
)

message("Disease identity table:")
print(table(seu$Disease_Identity, useNA = "ifany"))

# ============================================================
# 2. Extract epithelial compartment from validation dataset
# ============================================================

epi_celltypes <- c(
  "AT2",
  "AT1",
  "Basal",
  "Ciliated",
  "Secretory",
  "Epithelial"
)

available_epi_types <- intersect(
  epi_celltypes,
  unique(as.character(seu$celltype_marker_based))
)

message("Available epithelial-related marker-based cell types:")
print(available_epi_types)

if (length(available_epi_types) == 0) {
  stop("No epithelial-related cell types found in celltype_marker_based.")
}

cells_epi <- rownames(seu@meta.data)[
  as.character(seu$celltype_marker_based) %in% available_epi_types
]

message("Number of validation epithelial cells:")
print(length(cells_epi))

if (length(cells_epi) < 100) {
  warning("Very few epithelial cells extracted. Please check marker-based annotation.")
}

seu_epi <- subset(seu, cells = cells_epi)

DefaultAssay(seu_epi) <- "RNA"

message("Validation epithelial object:")
print(dim(seu_epi))
print(table(seu_epi$celltype_marker_based, useNA = "ifany"))
print(table(seu_epi$sample_id, useNA = "ifany"))

# ============================================================
# 3. Join layers and ensure normalized data
# ============================================================

message("RNA layers before JoinLayers:")
print(Layers(seu_epi[["RNA"]]))

seu_epi <- JoinLayers(
  object = seu_epi,
  assay = "RNA"
)

message("RNA layers after JoinLayers:")
print(Layers(seu_epi[["RNA"]]))

if (!"data" %in% Layers(seu_epi[["RNA"]])) {
  message("No data layer found. Running NormalizeData.")
  seu_epi <- NormalizeData(
    seu_epi,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    verbose = TRUE
  )
}

expr_mat <- LayerData(
  object = seu_epi,
  assay = "RNA",
  layer = "data"
)

# ============================================================
# 4. Define validation epithelial state gene programs
# ============================================================
# Positive direction:
#   IPF-associated remodeled / injury epithelial states
#
# Negative direction:
#   homeostatic alveolar epithelial states

state_modules <- list(
  Homeostatic_AT2 = c(
    "SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "SFTPD",
    "ABCA3", "NAPSA", "LAMP3"
  ),
  
  Homeostatic_AT1 = c(
    "AGER", "PDPN", "CAV1", "CAV2", "AQP5", "HOPX", "CLIC5"
  ),
  
  AT2_like_activated = c(
    "LAMP3", "SFTPD", "NAPSA", "ABCA3", "MUC1", "CLDN18",
    "CXCL17", "SLC34A2"
  ),
  
  IPF_basal_like = c(
    "KRT5", "KRT17", "TP63", "KRT15", "S100A2",
    "SERPINB5", "IL33", "TNS4"
  ),
  
  IPF_ciliated = c(
    "FOXJ1", "PIFO", "TUBB4B", "CAPS",
    "C1orf194", "C20orf85", "SNTN", "C9orf24"
  ),
  
  IPF_secretory_mucous = c(
    "SCGB1A1", "SCGB3A1", "SCGB3A2",
    "MUC5B", "MUC5AC", "BPIFB1", "LTF", "LCN2"
  ),
  
  IPF_aberrant_basaloid_fibrotic = c(
    "KRT17", "KRT5", "TP63", "COL1A1", "FN1", "VIM",
    "MMP7", "SPP1", "TGFBI", "VCAN", "LGALS3",
    "ITGA6", "CLDN4", "CDKN1A"
  )
)

positive_states <- c(
  "IPF_basal_like",
  "IPF_ciliated",
  "IPF_secretory_mucous",
  "IPF_aberrant_basaloid_fibrotic"
)

negative_states <- c(
  "Homeostatic_AT2",
  "Homeostatic_AT1",
  "AT2_like_activated"
)

# ============================================================
# 5. Check gene availability
# ============================================================

gene_presence <- lapply(names(state_modules), function(nm) {
  genes_present <- intersect(state_modules[[nm]], rownames(expr_mat))
  
  data.frame(
    module = nm,
    n_input = length(state_modules[[nm]]),
    n_present = length(genes_present),
    present_genes = paste(genes_present, collapse = ","),
    missing_genes = paste(setdiff(state_modules[[nm]], rownames(expr_mat)), collapse = ",")
  )
}) %>%
  bind_rows()

write.csv(
  gene_presence,
  file.path(tab_dir, "01_validation_epithelial_axis_gene_presence.csv"),
  row.names = FALSE
)

message("Gene presence:")
print(gene_presence)

if (max(gene_presence$n_present, na.rm = TRUE) < 2) {
  stop("Too few axis genes are present in validation epithelial object.")
}

# ============================================================
# 6. Compute per-cell epithelial state module scores
# ============================================================
# Score = average normalized expression of available genes.
# Then z-score each module across validation epithelial cells.

score_cols <- c()

for (nm in names(state_modules)) {
  genes <- intersect(state_modules[[nm]], rownames(expr_mat))
  score_name <- paste0(nm, "_score")
  
  message("Scoring module: ", nm, " | genes used: ", paste(genes, collapse = ", "))
  
  if (length(genes) >= 2) {
    seu_epi@meta.data[[score_name]] <- Matrix::colMeans(
      expr_mat[genes, , drop = FALSE]
    )
    
    score_cols <- c(score_cols, score_name)
  } else {
    warning("Skipping module due to insufficient genes: ", nm)
    seu_epi@meta.data[[score_name]] <- NA_real_
  }
}

rm(expr_mat)
gc()

zscore <- function(x) {
  if (all(is.na(x)) || sd(x, na.rm = TRUE) == 0) {
    return(rep(0, length(x)))
  }
  as.numeric(scale(x))
}

z_cols <- c()

for (sc in score_cols) {
  zc <- paste0(sc, "_z")
  seu_epi@meta.data[[zc]] <- zscore(seu_epi@meta.data[[sc]])
  z_cols <- c(z_cols, zc)
}

# ============================================================
# 7. Assign each validation epithelial cell to projected state
# ============================================================

state_score_cols <- paste0(names(state_modules), "_score_z")
state_score_cols <- intersect(state_score_cols, colnames(seu_epi@meta.data))

state_score_mat <- as.matrix(seu_epi@meta.data[, state_score_cols, drop = FALSE])
state_score_mat[is.na(state_score_mat)] <- -Inf

max_idx <- max.col(state_score_mat, ties.method = "first")
max_score <- state_score_mat[cbind(seq_len(nrow(state_score_mat)), max_idx)]

projected_state <- gsub("_score_z$", "", state_score_cols[max_idx])

# Low-confidence cells are assigned to Other_epithelial
min_z_for_assignment <- 0

projected_state[max_score < min_z_for_assignment] <- "Other_epithelial"

seu_epi$projected_epi_state <- projected_state
seu_epi$projected_epi_state_max_z <- max_score

seu_epi$projected_epi_axis_group <- dplyr::case_when(
  seu_epi$projected_epi_state %in% positive_states ~ "IPF_remodeled_epithelial",
  seu_epi$projected_epi_state %in% negative_states ~ "Homeostatic_alveolar_epithelial",
  TRUE ~ "Other_epithelial"
)

# Per-cell continuous axis score
positive_z_cols <- paste0(positive_states, "_score_z")
negative_z_cols <- paste0(negative_states, "_score_z")

positive_z_cols <- intersect(positive_z_cols, colnames(seu_epi@meta.data))
negative_z_cols <- intersect(negative_z_cols, colnames(seu_epi@meta.data))

seu_epi$per_cell_positive_remodeling_z <- rowMeans(
  seu_epi@meta.data[, positive_z_cols, drop = FALSE],
  na.rm = TRUE
)

seu_epi$per_cell_negative_homeostatic_z <- rowMeans(
  seu_epi@meta.data[, negative_z_cols, drop = FALSE],
  na.rm = TRUE
)

seu_epi$per_cell_epithelial_injury_remodeling_axis <- 
  seu_epi$per_cell_positive_remodeling_z -
  seu_epi$per_cell_negative_homeostatic_z

message("Projected epithelial states:")
print(table(seu_epi$projected_epi_state, useNA = "ifany"))

message("Projected epithelial axis groups:")
print(table(seu_epi$projected_epi_axis_group, useNA = "ifany"))

# ============================================================
# 8. Save validation epithelial metadata
# ============================================================

write.csv(
  seu_epi@meta.data,
  file.path(tab_dir, "02_validation_epithelial_cell_metadata_with_projected_axis.csv")
)

saveRDS(
  seu_epi,
  file.path(obj_dir, "GSE159354_epithelial_projected_axis_seurat.rds")
)

# ============================================================
# 9. UMAP visualization in validation epithelial cells
# ============================================================

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "projected_epi_state",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) +
  NoLegend() +
  ggtitle("GSE159354 projected epithelial states")

ggsave(
  file.path(fig_dir, "01_umap_projected_epithelial_states.pdf"),
  p,
  width = 10,
  height = 8,
  limitsize = FALSE
)

p <- DimPlot(
  seu_epi,
  reduction = "umap",
  group.by = "projected_epi_axis_group",
  label = TRUE,
  repel = TRUE,
  raster = TRUE
) +
  ggtitle("GSE159354 projected epithelial axis groups")

ggsave(
  file.path(fig_dir, "02_umap_projected_epithelial_axis_groups.pdf"),
  p,
  width = 9,
  height = 7,
  limitsize = FALSE
)

p <- FeaturePlot(
  seu_epi,
  reduction = "umap",
  features = "per_cell_epithelial_injury_remodeling_axis",
  raster = TRUE
) +
  ggtitle("Per-cell epithelial injury-remodeling axis")

ggsave(
  file.path(fig_dir, "03_umap_per_cell_epithelial_injury_remodeling_axis.pdf"),
  p,
  width = 8,
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
) +
  NoLegend() +
  ggtitle("Original marker-based epithelial labels")

ggsave(
  file.path(fig_dir, "04_umap_original_marker_based_epithelial_labels.pdf"),
  p,
  width = 9,
  height = 7,
  limitsize = FALSE
)

# ============================================================
# 10. Sample-level projected epithelial state fractions
# ============================================================

sample_state_fraction <- seu_epi@meta.data %>%
  count(sample_id, Disease_Identity, projected_epi_state, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  sample_state_fraction,
  file.path(tab_dir, "03_validation_sample_level_projected_epi_state_fractions.csv"),
  row.names = FALSE
)

sample_group_fraction <- seu_epi@meta.data %>%
  count(sample_id, Disease_Identity, projected_epi_axis_group, name = "n_cells") %>%
  group_by(sample_id) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()

write.csv(
  sample_group_fraction,
  file.path(tab_dir, "04_validation_sample_level_projected_axis_group_fractions.csv"),
  row.names = FALSE
)

# Make wide table
group_wide <- sample_group_fraction %>%
  select(sample_id, Disease_Identity, projected_epi_axis_group, fraction) %>%
  pivot_wider(
    names_from = projected_epi_axis_group,
    values_from = fraction,
    values_fill = 0
  )

required_group_cols <- c(
  "IPF_remodeled_epithelial",
  "Homeostatic_alveolar_epithelial",
  "Other_epithelial"
)

for (cc in required_group_cols) {
  if (!cc %in% colnames(group_wide)) {
    group_wide[[cc]] <- 0
  }
}

# ============================================================
# 11. Construct sample-level validation axis
# ============================================================

sample_axis <- group_wide %>%
  mutate(
    validation_epithelial_remodeling_fraction = IPF_remodeled_epithelial,
    validation_epithelial_homeostatic_fraction = Homeostatic_alveolar_epithelial,
    validation_epithelial_other_fraction = Other_epithelial,
    validation_epithelial_injury_remodeling_axis =
      validation_epithelial_remodeling_fraction -
      validation_epithelial_homeostatic_fraction,
    validation_epithelial_injury_remodeling_log2ratio =
      log2(
        (validation_epithelial_remodeling_fraction + 1e-6) /
          (validation_epithelial_homeostatic_fraction + 1e-6)
      )
  )

# Add mean per-cell continuous axis per sample
per_cell_axis_sample <- seu_epi@meta.data %>%
  group_by(sample_id) %>%
  summarise(
    mean_per_cell_epithelial_axis = mean(
      per_cell_epithelial_injury_remodeling_axis,
      na.rm = TRUE
    ),
    median_per_cell_epithelial_axis = median(
      per_cell_epithelial_injury_remodeling_axis,
      na.rm = TRUE
    ),
    n_epithelial_cells = n(),
    .groups = "drop"
  )

sample_axis <- sample_axis %>%
  left_join(per_cell_axis_sample, by = "sample_id") %>%
  mutate(
    Disease_Identity = trimws(as.character(Disease_Identity))
  )

disease_levels <- c("Control", "IPF", "SSc-ILD", "COPD")

disease_levels <- c(
  disease_levels[disease_levels %in% unique(sample_axis$Disease_Identity)],
  setdiff(sort(unique(sample_axis$Disease_Identity)), disease_levels)
)

sample_axis$Disease_Identity <- factor(
  sample_axis$Disease_Identity,
  levels = disease_levels
)

sample_axis <- sample_axis %>%
  arrange(validation_epithelial_injury_remodeling_axis) %>%
  mutate(
    validation_axis_rank = row_number(),
    validation_axis_percentile =
      (validation_axis_rank - 1) / pmax(n() - 1, 1)
  )

write.csv(
  sample_axis,
  file.path(tab_dir, "05_validation_sample_level_epithelial_injury_remodeling_axis.csv"),
  row.names = FALSE
)

message("Sample-level validation axis:")
print(head(sample_axis))

# ============================================================
# 12. Disease-level summaries and statistical tests if labels exist
# ============================================================

known_disease_levels <- sample_axis %>%
  filter(!is.na(Disease_Identity)) %>%
  pull(Disease_Identity) %>%
  as.character() %>%
  setdiff(c("Unknown", "")) %>%
  unique()

has_known_disease <- length(known_disease_levels) >= 2

if (has_known_disease) {
  
  disease_summary <- sample_axis %>%
    group_by(Disease_Identity) %>%
    summarise(
      n_samples = n(),
      median_remodeling_fraction = median(
        validation_epithelial_remodeling_fraction,
        na.rm = TRUE
      ),
      median_homeostatic_fraction = median(
        validation_epithelial_homeostatic_fraction,
        na.rm = TRUE
      ),
      median_axis = median(
        validation_epithelial_injury_remodeling_axis,
        na.rm = TRUE
      ),
      mean_axis = mean(
        validation_epithelial_injury_remodeling_axis,
        na.rm = TRUE
      ),
      median_log2ratio = median(
        validation_epithelial_injury_remodeling_log2ratio,
        na.rm = TRUE
      ),
      .groups = "drop"
    )
  
  write.csv(
    disease_summary,
    file.path(tab_dir, "06_validation_epithelial_axis_disease_summary.csv"),
    row.names = FALSE
  )
  
  pairwise_numeric_test <- function(df, value_col) {
    disease_levels <- unique(as.character(df$Disease_Identity))
    disease_levels <- disease_levels[!is.na(disease_levels)]
    
    if (length(disease_levels) < 2) {
      return(data.frame())
    }
    
    pairs <- combn(disease_levels, 2, simplify = FALSE)
    
    res <- list()
    
    for (i in seq_along(pairs)) {
      g1 <- pairs[[i]][1]
      g2 <- pairs[[i]][2]
      
      x <- df %>%
        filter(Disease_Identity == g1) %>%
        pull(.data[[value_col]])
      
      y <- df %>%
        filter(Disease_Identity == g2) %>%
        pull(.data[[value_col]])
      
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
        n_group_1 = length(x),
        n_group_2 = length(y),
        median_group_1 = ifelse(length(x) > 0, median(x, na.rm = TRUE), NA),
        median_group_2 = ifelse(length(y) > 0, median(y, na.rm = TRUE), NA),
        mean_group_1 = ifelse(length(x) > 0, mean(x, na.rm = TRUE), NA),
        mean_group_2 = ifelse(length(y) > 0, mean(y, na.rm = TRUE), NA),
        median_delta = ifelse(
          length(x) > 0 && length(y) > 0,
          median(x, na.rm = TRUE) - median(y, na.rm = TRUE),
          NA
        ),
        p_value = pval
      )
    }
    
    bind_rows(res)
  }
  
  validation_stats <- bind_rows(
    pairwise_numeric_test(sample_axis, "validation_epithelial_remodeling_fraction"),
    pairwise_numeric_test(sample_axis, "validation_epithelial_homeostatic_fraction"),
    pairwise_numeric_test(sample_axis, "validation_epithelial_injury_remodeling_axis"),
    pairwise_numeric_test(sample_axis, "validation_epithelial_injury_remodeling_log2ratio"),
    pairwise_numeric_test(sample_axis, "mean_per_cell_epithelial_axis")
  ) %>%
    group_by(metric) %>%
    mutate(p_adj_BH = p.adjust(p_value, method = "BH")) %>%
    ungroup()
  
  write.csv(
    validation_stats,
    file.path(tab_dir, "07_validation_epithelial_axis_pairwise_wilcox.csv"),
    row.names = FALSE
  )
}

# ============================================================
# 13. Plots: sample-level validation axis
# ============================================================

p <- ggplot(
  sample_axis,
  aes(
    x = validation_epithelial_homeostatic_fraction,
    y = validation_epithelial_remodeling_fraction,
    color = Disease_Identity
  )
) +
  geom_point(size = 2.2, alpha = 0.85) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_classic(base_size = 12) +
  labs(
    title = "GSE159354 epithelial remodeling shift",
    x = "Homeostatic epithelial fraction",
    y = "Remodeled epithelial fraction",
    color = "Disease"
  )

ggsave(
  file.path(fig_dir, "05_validation_remodeled_vs_homeostatic_fraction.pdf"),
  p,
  width = 6,
  height = 5,
  limitsize = FALSE
)

sample_axis <- sample_axis %>%
  arrange(validation_epithelial_injury_remodeling_axis) %>%
  mutate(
    validation_axis_rank = row_number(),
    validation_axis_percentile =
      (validation_axis_rank - 1) / pmax(n() - 1, 1)
  )

p <- ggplot(
  sample_axis,
  aes(
    x = validation_axis_rank,
    y = validation_epithelial_injury_remodeling_axis,
    color = Disease_Identity
  )
) +
  geom_point(size = 2.3, alpha = 0.85) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_classic(base_size = 12) +
  labs(
    title = "GSE159354 ranked epithelial injury-remodeling axis",
    x = "Samples ranked along validation axis",
    y = "Validation epithelial injury-remodeling axis",
    color = "Disease"
  )

ggsave(
  file.path(fig_dir, "06_validation_ranked_epithelial_axis.pdf"),
  p,
  width = 8,
  height = 5,
  limitsize = FALSE
)

if (has_known_disease) {
  
  sample_axis_plot <- sample_axis %>%
    filter(!is.na(Disease_Identity)) %>%
    mutate(
      Disease_Identity = factor(
        as.character(Disease_Identity),
        levels = disease_levels
      )
    )
  
  p <- ggplot(
    sample_axis_plot,
    aes(
      x = Disease_Identity,
      y = validation_epithelial_injury_remodeling_axis,
      fill = Disease_Identity
    )
  ) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.15, size = 1.2, alpha = 0.75) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_classic(base_size = 12) +
    labs(
      title = "GSE159354 validation epithelial injury-remodeling axis",
      x = NULL,
      y = "Remodeled fraction - homeostatic fraction"
    ) +
    theme(legend.position = "none")
  
  ggsave(
    file.path(fig_dir, "07_validation_epithelial_axis_by_disease.pdf"),
    p,
    width = 6,
    height = 5,
    limitsize = FALSE
  )
  
  p <- ggplot(
    sample_axis_plot,
    aes(
      x = Disease_Identity,
      y = mean_per_cell_epithelial_axis,
      fill = Disease_Identity
    )
  ) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.15, size = 1.2, alpha = 0.75) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_classic(base_size = 12) +
    labs(
      title = "GSE159354 mean per-cell epithelial axis by disease",
      x = NULL,
      y = "Mean per-cell axis"
    ) +
    theme(legend.position = "none")
  
  ggsave(
    file.path(fig_dir, "08_validation_mean_per_cell_axis_by_disease.pdf"),
    p,
    width = 6,
    height = 5,
    limitsize = FALSE
  )
}

# ============================================================
# 14. Projected state fraction plots
# ============================================================

p <- ggplot(
  sample_state_fraction,
  aes(x = Disease_Identity, y = fraction, fill = Disease_Identity)
) +
  geom_boxplot(outlier.shape = NA, width = 0.6) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.65) +
  facet_wrap(~ projected_epi_state, scales = "free_y", ncol = 4) +
  theme_classic(base_size = 10) +
  labs(
    title = "GSE159354 projected epithelial state fractions",
    x = NULL,
    y = "Fraction within validation epithelial compartment"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 8)
  )

ggsave(
  file.path(fig_dir, "09_validation_projected_epi_state_fractions_by_disease.pdf"),
  p,
  width = 14,
  height = 9,
  limitsize = FALSE
)

p <- ggplot(
  sample_group_fraction,
  aes(x = Disease_Identity, y = fraction, fill = projected_epi_axis_group)
) +
  geom_col(position = "fill", width = 0.75) +
  theme_classic(base_size = 12) +
  labs(
    title = "GSE159354 projected epithelial axis group composition",
    x = NULL,
    y = "Fraction within validation epithelial compartment",
    fill = "Projected group"
  )

ggsave(
  file.path(fig_dir, "10_validation_axis_group_composition_by_disease.pdf"),
  p,
  width = 7,
  height = 5,
  limitsize = FALSE
)

# ============================================================
# 15. Final compact summary
# ============================================================

validation_summary <- list(
  n_total_cells = ncol(seu),
  n_validation_epithelial_cells = ncol(seu_epi),
  epithelial_marker_types_used = available_epi_types,
  projected_state_table = table(seu_epi$projected_epi_state, useNA = "ifany"),
  projected_axis_group_table = table(seu_epi$projected_epi_axis_group, useNA = "ifany")
)

saveRDS(
  validation_summary,
  file.path(out_dir, "validation_epithelial_axis_summary.rds")
)

message("Done.")
message("Validation epithelial axis projection finished.")
message("Main output directory:")
message(out_dir)

message("Key tables:")
message(file.path(tab_dir, "01_validation_epithelial_axis_gene_presence.csv"))
message(file.path(tab_dir, "02_validation_epithelial_cell_metadata_with_projected_axis.csv"))
message(file.path(tab_dir, "03_validation_sample_level_projected_epi_state_fractions.csv"))
message(file.path(tab_dir, "05_validation_sample_level_epithelial_injury_remodeling_axis.csv"))

message("Key figures:")
message(file.path(fig_dir, "01_umap_projected_epithelial_states.pdf"))
message(file.path(fig_dir, "03_umap_per_cell_epithelial_injury_remodeling_axis.pdf"))
message(file.path(fig_dir, "05_validation_remodeled_vs_homeostatic_fraction.pdf"))
message(file.path(fig_dir, "06_validation_ranked_epithelial_axis.pdf"))
message(file.path(fig_dir, "07_validation_epithelial_axis_by_disease.pdf"))