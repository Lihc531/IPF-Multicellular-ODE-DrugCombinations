# 18_build_cell_state_feature_matrix_axis_v2.R
# Build Cell-state feature matrix based on axis v2
#
# Outputs:
#   v1A: sample x celltype x axis_v2_stage
#   v1B: sample x celltype x cluster x axis_v2_stage
#
# Input:
#   results/progression_axis_v2/tables/03_cell_metadata_with_axis_v2_module_scores.rds
#
# Main outputs:
#   results/cell_state_feature_matrix/tables/
#     01_cell_state_matrix_v1A_sample_celltype_stage.csv
#     02_cell_state_matrix_v1B_sample_celltype_cluster_stage.csv
#     03_cell_state_matrix_v1A_filtered.csv
#     04_cell_state_matrix_v1B_filtered.csv
#     05_cell_state_qc_summary.csv
#     06_feature_columns_used.csv

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tibble)
})

set.seed(123)

# ============================================================
# 0. Paths
# ============================================================

project_dir <- getwd()

axis_v2_rds <- file.path(
  project_dir,
  "results",
  "progression_axis_v2",
  "tables",
  "03_cell_metadata_with_axis_v2_module_scores.rds"
)

# Optional: gene-level dynamic evidence table.
# If present, this script will export a Python Layer 1/2 compatible
# dynamic_gene_csv whose cell_state_id values exactly match the v1B
# sample x celltype x cluster x axis_v2_stage states used for ODE velocity.
#
# Required columns:
#   gene
#   celltype_marker_based
#   dynamic_trend_score
#   module_membership_score
#   avg_expr
#   pct_expr
#   disease_stage_association_score
#   patient_consistency_score
#
# You can point this path to your previous gene-cell-state feature table.
dynamic_gene_seed_file <- file.path(
  project_dir,
  "results",
  "gene_cell_state_modeling",
  "tables",
  "13_gene_cell_state_modeling_feature_table.csv"
)

out_dir <- file.path(
  project_dir,
  "results",
  "cell_state_feature_matrix2"
)

tab_dir <- file.path(out_dir, "tables")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Parameters
# ============================================================

min_cells_state_v1A <- 20
min_cells_state_v1B <- 10

# Whether to calculate quantile bins from continuous axis_v2_score
make_axis_quantile_bins <- TRUE
n_axis_quantiles <- 5

# Feature summarization suffixes
# We mainly use normalized 0_1 module scores and composite scores.
module_score_pattern <- "_score_0_1$"
composite_score_pattern <- "_composite_score$"

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

safe_mean <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

safe_median <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  median(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (sum(is.finite(x)) < 2) {
    return(NA_real_)
  }
  sd(x, na.rm = TRUE)
}

safe_max_abs <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  max(abs(x), na.rm = TRUE)
}

safe_fraction <- function(x) {
  mean(x, na.rm = TRUE)
}

make_state_id <- function(df, cols) {
  apply(
    df[, cols, drop = FALSE],
    1,
    function(z) paste(z, collapse = "__")
  )
}

check_required_cols <- function(df, required_cols) {
  missing_cols <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
}

summarise_state_matrix <- function(meta, group_cols, feature_cols) {
  sample_total <- meta %>%
    count(sample_id, name = "n_cells_sample")
  
  state_base <- meta %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      n_cells = n(),
      n_subjects = n_distinct(Subject_Identity),
      n_libraries = n_distinct(Library_Identity),
      mean_nCount_RNA = safe_mean(nCount_RNA),
      mean_nFeature_RNA = safe_mean(nFeature_RNA),
      mean_percent_mt = safe_mean(percent.mt),
      mean_percent_ribo = safe_mean(percent.ribo),
      mean_doublet_score = safe_mean(doublet_score),
      .groups = "drop"
    ) %>%
    left_join(sample_total, by = "sample_id") %>%
    mutate(
      fraction_in_sample = n_cells / n_cells_sample
    )
  
  state_features_mean <- meta %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      across(
        all_of(feature_cols),
        ~ safe_mean(.x),
        .names = "mean_{.col}"
      ),
      .groups = "drop"
    )
  
  state_features_median <- meta %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      across(
        all_of(feature_cols),
        ~ safe_median(.x),
        .names = "median_{.col}"
      ),
      .groups = "drop"
    )
  
  state_features_sd <- meta %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      across(
        all_of(feature_cols),
        ~ safe_sd(.x),
        .names = "sd_{.col}"
      ),
      .groups = "drop"
    )
  
  out <- state_base %>%
    left_join(state_features_mean, by = group_cols) %>%
    left_join(state_features_median, by = group_cols) %>%
    left_join(state_features_sd, by = group_cols)
  
  out
}

add_matrix_qc_flags <- function(df, min_cells_state) {
  df %>%
    mutate(
      pass_min_cells = n_cells >= min_cells_state,
      pass_fraction_nonzero = fraction_in_sample > 0,
      matrix_qc_status = case_when(
        pass_min_cells & pass_fraction_nonzero ~ "pass",
        !pass_min_cells ~ "low_cells",
        TRUE ~ "check"
      )
    )
}

# ============================================================
# 3. Load axis v2 metadata
# ============================================================

if (!file.exists(axis_v2_rds)) {
  stop("Input axis v2 metadata not found: ", axis_v2_rds)
}

message("Reading axis v2 metadata...")
meta <- readRDS(axis_v2_rds)

message("Metadata dimension:")
print(dim(meta))

if (!"cell_id" %in% colnames(meta)) {
  if ("CellBarcode_Identity" %in% colnames(meta)) {
    meta$cell_id <- as.character(meta$CellBarcode_Identity)
  } else {
    meta$cell_id <- rownames(meta)
  }
}

rownames(meta) <- meta$cell_id

# ============================================================
# 4. Required columns
# ============================================================

required_cols <- c(
  "cell_id",
  "sample_id",
  "Disease_Identity",
  "Subject_Identity",
  "Library_Identity",
  "CellType_Category",
  "celltype_marker_based",
  "cluster_default",
  "axis_v2_score",
  "axis_v2_stage",
  "cell_multicellular_progression_score",
  "cell_multicellular_progression_stage"
)

check_required_cols(meta, required_cols)

# Some QC columns may not exist in all objects.
# Add safe NA columns if absent.
optional_qc_cols <- c(
  "nCount_RNA",
  "nFeature_RNA",
  "percent.mt",
  "percent.ribo",
  "doublet_score"
)

for (cc in optional_qc_cols) {
  if (!cc %in% colnames(meta)) {
    meta[[cc]] <- NA_real_
  }
}

# ============================================================
# 5. Add axis quantile bins
# ============================================================

if (make_axis_quantile_bins) {
  message("Adding axis_v2 quantile bins...")
  
  meta$axis_v2_quantile <- cut(
    meta$axis_v2_score,
    breaks = unique(
      quantile(
        meta$axis_v2_score,
        probs = seq(0, 1, length.out = n_axis_quantiles + 1),
        na.rm = TRUE
      )
    ),
    include.lowest = TRUE,
    labels = paste0("Q", seq_len(n_axis_quantiles))
  )
  
  meta$axis_v2_quantile <- as.character(meta$axis_v2_quantile)
} else {
  meta$axis_v2_quantile <- NA_character_
}

# ============================================================
# 6. Select feature columns
# ============================================================

axis_cols <- c(
  "cell_multicellular_progression_score",
  "axis_v2_score",
  "fibrosis_composite_score",
  "epithelial_injury_composite_score",
  "myeloid_inflammatory_composite_score",
  "vascular_remodeling_composite_score",
  "stress_inflammation_composite_score",
  "immune_activation_composite_score",
  "injury_inflammation_composite_score"
)

axis_cols <- intersect(axis_cols, colnames(meta))

module_cols <- grep(
  module_score_pattern,
  colnames(meta),
  value = TRUE
)

composite_cols <- grep(
  composite_score_pattern,
  colnames(meta),
  value = TRUE
)

feature_cols <- unique(c(axis_cols, module_cols, composite_cols))
feature_cols <- feature_cols[feature_cols %in% colnames(meta)]

if (length(feature_cols) == 0) {
  stop("No feature columns found. Please check module score columns.")
}

feature_info <- data.frame(
  feature_col = feature_cols,
  feature_type = case_when(
    feature_cols %in% axis_cols ~ "axis_or_composite_core",
    grepl(module_score_pattern, feature_cols) ~ "module_score_0_1",
    grepl(composite_score_pattern, feature_cols) ~ "composite_score",
    TRUE ~ "other"
  ),
  stringsAsFactors = FALSE
)

write.csv(
  feature_info,
  file.path(tab_dir, "06_feature_columns_used.csv"),
  row.names = FALSE
)

message("Number of feature columns used: ", length(feature_cols))

# ============================================================
# 7. Construct v1A matrix
#    sample x celltype x axis_v2_stage
# ============================================================

message("Building v1A: sample x celltype x axis_v2_stage...")

group_cols_v1A <- c(
  "sample_id",
  "Disease_Identity",
  "Subject_Identity",
  "Library_Identity",
  "CellType_Category",
  "celltype_marker_based",
  "axis_v2_stage"
)

matrix_v1A <- summarise_state_matrix(
  meta = meta,
  group_cols = group_cols_v1A,
  feature_cols = feature_cols
)

matrix_v1A$cell_state_id <- make_state_id(
  matrix_v1A,
  c("sample_id", "celltype_marker_based", "axis_v2_stage")
)

matrix_v1A <- matrix_v1A %>%
  relocate(cell_state_id, .before = 1) %>%
  add_matrix_qc_flags(min_cells_state = min_cells_state_v1A) %>%
  arrange(
    Disease_Identity,
    sample_id,
    CellType_Category,
    celltype_marker_based,
    axis_v2_stage
  )

write.csv(
  matrix_v1A,
  file.path(tab_dir, "01_cell_state_matrix_v1A_sample_celltype_stage.csv"),
  row.names = FALSE
)

matrix_v1A_filtered <- matrix_v1A %>%
  filter(pass_min_cells)

write.csv(
  matrix_v1A_filtered,
  file.path(tab_dir, "03_cell_state_matrix_v1A_filtered.csv"),
  row.names = FALSE
)

saveRDS(
  matrix_v1A,
  file.path(tab_dir, "01_cell_state_matrix_v1A_sample_celltype_stage.rds")
)

saveRDS(
  matrix_v1A_filtered,
  file.path(tab_dir, "03_cell_state_matrix_v1A_filtered.rds")
)

# ============================================================
# 8. Construct v1B matrix
#    sample x celltype x cluster x axis_v2_stage
# ============================================================

message("Building v1B: sample x celltype x cluster x axis_v2_stage...")

group_cols_v1B <- c(
  "sample_id",
  "Disease_Identity",
  "Subject_Identity",
  "Library_Identity",
  "CellType_Category",
  "celltype_marker_based",
  "cluster_default",
  "axis_v2_stage"
)

matrix_v1B <- summarise_state_matrix(
  meta = meta,
  group_cols = group_cols_v1B,
  feature_cols = feature_cols
)

matrix_v1B$cell_state_id <- make_state_id(
  matrix_v1B,
  c("sample_id", "celltype_marker_based", "cluster_default", "axis_v2_stage")
)

matrix_v1B <- matrix_v1B %>%
  relocate(cell_state_id, .before = 1) %>%
  add_matrix_qc_flags(min_cells_state = min_cells_state_v1B) %>%
  arrange(
    Disease_Identity,
    sample_id,
    CellType_Category,
    celltype_marker_based,
    cluster_default,
    axis_v2_stage
  )

write.csv(
  matrix_v1B,
  file.path(tab_dir, "02_cell_state_matrix_v1B_sample_celltype_cluster_stage.csv"),
  row.names = FALSE
)

matrix_v1B_filtered <- matrix_v1B %>%
  filter(pass_min_cells)

write.csv(
  matrix_v1B_filtered,
  file.path(tab_dir, "04_cell_state_matrix_v1B_filtered.csv"),
  row.names = FALSE
)

saveRDS(
  matrix_v1B,
  file.path(tab_dir, "02_cell_state_matrix_v1B_sample_celltype_cluster_stage.rds")
)

saveRDS(
  matrix_v1B_filtered,
  file.path(tab_dir, "04_cell_state_matrix_v1B_filtered.rds")
)

# ============================================================
# 9. Optional quantile-based matrices
#    These are useful for trend / dynamic analysis
# ============================================================

if (make_axis_quantile_bins) {
  message("Building optional quantile-based cell-state matrices...")
  
  group_cols_v1A_q <- c(
    "sample_id",
    "Disease_Identity",
    "Subject_Identity",
    "Library_Identity",
    "CellType_Category",
    "celltype_marker_based",
    "axis_v2_quantile"
  )
  
  matrix_v1A_q <- summarise_state_matrix(
    meta = meta,
    group_cols = group_cols_v1A_q,
    feature_cols = feature_cols
  )
  
  matrix_v1A_q$cell_state_id <- make_state_id(
    matrix_v1A_q,
    c("sample_id", "celltype_marker_based", "axis_v2_quantile")
  )
  
  matrix_v1A_q <- matrix_v1A_q %>%
    relocate(cell_state_id, .before = 1) %>%
    add_matrix_qc_flags(min_cells_state = min_cells_state_v1A)
  
  write.csv(
    matrix_v1A_q,
    file.path(tab_dir, "07_cell_state_matrix_v1A_sample_celltype_axis_quantile.csv"),
    row.names = FALSE
  )
  
  group_cols_v1B_q <- c(
    "sample_id",
    "Disease_Identity",
    "Subject_Identity",
    "Library_Identity",
    "CellType_Category",
    "celltype_marker_based",
    "cluster_default",
    "axis_v2_quantile"
  )
  
  matrix_v1B_q <- summarise_state_matrix(
    meta = meta,
    group_cols = group_cols_v1B_q,
    feature_cols = feature_cols
  )
  
  matrix_v1B_q$cell_state_id <- make_state_id(
    matrix_v1B_q,
    c("sample_id", "celltype_marker_based", "cluster_default", "axis_v2_quantile")
  )
  
  matrix_v1B_q <- matrix_v1B_q %>%
    relocate(cell_state_id, .before = 1) %>%
    add_matrix_qc_flags(min_cells_state = min_cells_state_v1B)
  
  write.csv(
    matrix_v1B_q,
    file.path(tab_dir, "08_cell_state_matrix_v1B_sample_celltype_cluster_axis_quantile.csv"),
    row.names = FALSE
  )
}

# ============================================================
# 9B. Optional: export dynamic_gene_csv matched to v1B ODE states
#     This fixes the common problem where a previous gene-state table
#     used module-label cell_state_id values that do not match the ODE
#     cell_state_id values.
# ============================================================

dynamic_gene_csv <- NULL
dynamic_gene_qc <- NULL

if (file.exists(dynamic_gene_seed_file)) {
  message("Building dynamic_gene_csv matched to v1B ODE cell_state_id...")
  message("Reading gene-level seed table: ", dynamic_gene_seed_file)

  dynamic_seed <- read.csv(dynamic_gene_seed_file, stringsAsFactors = FALSE)

  required_dynamic_seed_cols <- c(
    "gene",
    "celltype_marker_based",
    "dynamic_trend_score",
    "module_membership_score",
    "avg_expr",
    "pct_expr",
    "disease_stage_association_score",
    "patient_consistency_score"
  )

  check_required_cols(dynamic_seed, required_dynamic_seed_cols)

  dynamic_seed <- dynamic_seed %>%
    mutate(
      gene = as.character(gene),
      celltype_marker_based = as.character(celltype_marker_based)
    ) %>%
    filter(
      !is.na(gene),
      gene != "",
      !is.na(celltype_marker_based),
      celltype_marker_based != ""
    )

  # Collapse previous state/module rows to gene x cell type evidence.
  # This keeps the gene-level evidence but avoids carrying over old
  # module-label cell_state_id values that cannot join to ODE states.
  gene_celltype_evidence <- dynamic_seed %>%
    group_by(gene, celltype_marker_based) %>%
    summarise(
      dynamic_trend_score = safe_max_abs(dynamic_trend_score),
      module_membership_score = safe_mean(module_membership_score),
      avg_expr = safe_mean(avg_expr),
      pct_expr = safe_mean(pct_expr),
      disease_stage_association_score = safe_max_abs(disease_stage_association_score),
      patient_consistency_score = safe_mean(patient_consistency_score),
      n_seed_rows = n(),
      n_seed_cell_states = if ("cell_state_id" %in% colnames(dynamic_seed)) {
        n_distinct(cell_state_id)
      } else {
        NA_integer_
      },
      .groups = "drop"
    )

  # These are the exact v1B states used as ODE input.
  # Therefore the exported dynamic_gene_csv will join to
  # 12_direction_constrained_state_velocity_scores.csv by exact cell_state_id.
  ode_states_v1B <- matrix_v1B_filtered %>%
    select(
      cell_state_id,
      sample_id,
      Disease_Identity,
      Subject_Identity,
      Library_Identity,
      CellType_Category,
      celltype_marker_based,
      cluster_default,
      axis_v2_stage,
      n_cells,
      fraction_in_sample,
      mean_axis_v2_score
    ) %>%
    distinct()

  dynamic_gene_csv <- ode_states_v1B %>%
    inner_join(gene_celltype_evidence, by = "celltype_marker_based") %>%
    transmute(
      gene = gene,
      cell_state_id = cell_state_id,
      celltype_marker_based = celltype_marker_based,
      cluster_default = cluster_default,
      axis_v2_stage = axis_v2_stage,
      dynamic_trend_score = dynamic_trend_score,
      module_membership_score = module_membership_score,
      avg_expr = avg_expr,
      pct_expr = pct_expr,
      disease_stage_association_score = disease_stage_association_score,
      patient_consistency_score = patient_consistency_score,
      sample_id = sample_id,
      Disease_Identity = Disease_Identity,
      Subject_Identity = Subject_Identity,
      Library_Identity = Library_Identity,
      CellType_Category = CellType_Category,
      n_cells_state = n_cells,
      fraction_in_sample = fraction_in_sample,
      mean_axis_v2_score = mean_axis_v2_score,
      n_seed_rows = n_seed_rows,
      n_seed_cell_states = n_seed_cell_states
    ) %>%
    arrange(
      celltype_marker_based,
      cluster_default,
      axis_v2_stage,
      desc(dynamic_trend_score),
      desc(module_membership_score),
      gene
    )

  dynamic_gene_qc <- data.frame(
    n_dynamic_seed_rows = nrow(dynamic_seed),
    n_dynamic_seed_genes = n_distinct(dynamic_seed$gene),
    n_gene_celltype_pairs = nrow(gene_celltype_evidence),
    n_v1B_ode_states = nrow(ode_states_v1B),
    n_dynamic_gene_csv_rows = nrow(dynamic_gene_csv),
    n_dynamic_gene_csv_genes = n_distinct(dynamic_gene_csv$gene),
    n_dynamic_gene_csv_cell_states = n_distinct(dynamic_gene_csv$cell_state_id),
    n_celltypes_in_dynamic_seed = n_distinct(dynamic_seed$celltype_marker_based),
    n_celltypes_in_dynamic_gene_csv = n_distinct(dynamic_gene_csv$celltype_marker_based),
    stringsAsFactors = FALSE
  )

  write.csv(
    dynamic_gene_csv,
    file.path(tab_dir, "12_dynamic_gene_csv_v1B_ode_matched.csv"),
    row.names = FALSE
  )

  write.csv(
    dynamic_gene_qc,
    file.path(tab_dir, "13_dynamic_gene_csv_v1B_ode_matched_qc.csv"),
    row.names = FALSE
  )

  message("dynamic_gene_csv output:")
  message("  ", file.path(tab_dir, "12_dynamic_gene_csv_v1B_ode_matched.csv"))
  message("dynamic_gene_csv QC:")
  message("  ", file.path(tab_dir, "13_dynamic_gene_csv_v1B_ode_matched_qc.csv"))

} else {
  message("Optional dynamic_gene_seed_file not found; skipping dynamic_gene_csv export:")
  message("  ", dynamic_gene_seed_file)
  message("To enable this output, point dynamic_gene_seed_file to a gene-level evidence table.")
}

# ============================================================
# 10. QC summaries
# ============================================================

message("Generating QC summaries...")

qc_summary <- bind_rows(
  data.frame(
    matrix_version = "v1A_sample_celltype_stage",
    n_states_total = nrow(matrix_v1A),
    n_states_pass = sum(matrix_v1A$pass_min_cells),
    n_states_low_cells = sum(!matrix_v1A$pass_min_cells),
    min_cells_threshold = min_cells_state_v1A,
    median_cells_per_state = median(matrix_v1A$n_cells, na.rm = TRUE),
    mean_cells_per_state = mean(matrix_v1A$n_cells, na.rm = TRUE),
    max_cells_per_state = max(matrix_v1A$n_cells, na.rm = TRUE)
  ),
  data.frame(
    matrix_version = "v1B_sample_celltype_cluster_stage",
    n_states_total = nrow(matrix_v1B),
    n_states_pass = sum(matrix_v1B$pass_min_cells),
    n_states_low_cells = sum(!matrix_v1B$pass_min_cells),
    min_cells_threshold = min_cells_state_v1B,
    median_cells_per_state = median(matrix_v1B$n_cells, na.rm = TRUE),
    mean_cells_per_state = mean(matrix_v1B$n_cells, na.rm = TRUE),
    max_cells_per_state = max(matrix_v1B$n_cells, na.rm = TRUE)
  )
)

write.csv(
  qc_summary,
  file.path(tab_dir, "05_cell_state_qc_summary.csv"),
  row.names = FALSE
)

qc_by_disease_v1A <- matrix_v1A %>%
  group_by(Disease_Identity) %>%
  summarise(
    n_states = n(),
    n_states_pass = sum(pass_min_cells),
    total_cells = sum(n_cells),
    mean_axis_v2 = safe_mean(mean_axis_v2_score),
    mean_fibrosis = safe_mean(mean_fibrosis_composite_score),
    mean_epithelial_injury = safe_mean(mean_epithelial_injury_composite_score),
    .groups = "drop"
  )

write.csv(
  qc_by_disease_v1A,
  file.path(tab_dir, "09_qc_v1A_by_disease.csv"),
  row.names = FALSE
)

qc_by_celltype_v1A <- matrix_v1A %>%
  group_by(CellType_Category, celltype_marker_based) %>%
  summarise(
    n_states = n(),
    n_states_pass = sum(pass_min_cells),
    total_cells = sum(n_cells),
    mean_axis_v2 = safe_mean(mean_axis_v2_score),
    mean_fibrosis = safe_mean(mean_fibrosis_composite_score),
    mean_epithelial_injury = safe_mean(mean_epithelial_injury_composite_score),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_axis_v2))

write.csv(
  qc_by_celltype_v1A,
  file.path(tab_dir, "10_qc_v1A_by_celltype.csv"),
  row.names = FALSE
)

qc_by_celltype_cluster_v1B <- matrix_v1B %>%
  group_by(CellType_Category, celltype_marker_based, cluster_default) %>%
  summarise(
    n_states = n(),
    n_states_pass = sum(pass_min_cells),
    total_cells = sum(n_cells),
    mean_axis_v2 = safe_mean(mean_axis_v2_score),
    mean_fibrosis = safe_mean(mean_fibrosis_composite_score),
    mean_epithelial_injury = safe_mean(mean_epithelial_injury_composite_score),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_axis_v2))

write.csv(
  qc_by_celltype_cluster_v1B,
  file.path(tab_dir, "11_qc_v1B_by_celltype_cluster.csv"),
  row.names = FALSE
)

# ============================================================
# 11. Save compact object
# ============================================================

cell_state_matrix_object <- list(
  matrix_v1A = matrix_v1A,
  matrix_v1A_filtered = matrix_v1A_filtered,
  matrix_v1B = matrix_v1B,
  matrix_v1B_filtered = matrix_v1B_filtered,
  qc_summary = qc_summary,
  qc_by_disease_v1A = qc_by_disease_v1A,
  qc_by_celltype_v1A = qc_by_celltype_v1A,
  qc_by_celltype_cluster_v1B = qc_by_celltype_cluster_v1B,
  feature_info = feature_info,
  dynamic_gene_csv = dynamic_gene_csv,
  dynamic_gene_qc = dynamic_gene_qc,
  parameters = list(
    min_cells_state_v1A = min_cells_state_v1A,
    min_cells_state_v1B = min_cells_state_v1B,
    make_axis_quantile_bins = make_axis_quantile_bins,
    n_axis_quantiles = n_axis_quantiles
  )
)

saveRDS(
  cell_state_matrix_object,
  file.path(out_dir, "cell_state_feature_matrix_axis_v2_object.rds")
)

# ============================================================
# 12. Final messages
# ============================================================

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "01_cell_state_matrix_v1A_sample_celltype_stage.csv"))
message("  ", file.path(tab_dir, "02_cell_state_matrix_v1B_sample_celltype_cluster_stage.csv"))
message("  ", file.path(tab_dir, "03_cell_state_matrix_v1A_filtered.csv"))
message("  ", file.path(tab_dir, "04_cell_state_matrix_v1B_filtered.csv"))
message("  ", file.path(tab_dir, "05_cell_state_qc_summary.csv"))
if (!is.null(dynamic_gene_csv)) {
  message("  ", file.path(tab_dir, "12_dynamic_gene_csv_v1B_ode_matched.csv"))
  message("  ", file.path(tab_dir, "13_dynamic_gene_csv_v1B_ode_matched_qc.csv"))
}
message("  ", file.path(out_dir, "cell_state_feature_matrix_axis_v2_object.rds"))