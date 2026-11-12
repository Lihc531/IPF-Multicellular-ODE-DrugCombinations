# ============================================================
# config.R
# Virtual perturbation pipeline configuration
# Validation-completeness revision
# ============================================================

INPUT_ROOT <- normalizePath(
  Sys.getenv("VP_INPUT_ROOT", unset = file.path(getwd(), "virtual_perturbation_inputs")),
  winslash = "/", mustWork = FALSE
)
OUTPUT_ROOT <- normalizePath(
  Sys.getenv("VP_OUTPUT_ROOT", unset = file.path(getwd(), "virtual_perturbation_results")),
  winslash = "/", mustWork = FALSE
)

MAIN_DIR <- file.path(INPUT_ROOT, "main_by_celltype")
VALIDATION_DIR <- file.path(INPUT_ROOT, "validation_by_celltype")
SUPPORT_DIR <- file.path(INPUT_ROOT, "supporting_tables")

as_int_env <- function(name, default, min_value = NULL) {
  x <- suppressWarnings(as.integer(Sys.getenv(name, unset = as.character(default))))
  if (!is.finite(x) || (!is.null(min_value) && x < min_value)) x <- as.integer(default)
  x
}
as_num_env <- function(name, default, min_value = NULL) {
  x <- suppressWarnings(as.numeric(Sys.getenv(name, unset = as.character(default))))
  if (!is.finite(x) || (!is.null(min_value) && x < min_value)) x <- as.numeric(default)
  x
}
as_bool_env <- function(name, default = FALSE) {
  x <- tolower(trimws(Sys.getenv(name, unset = ifelse(default, "true", "false"))))
  x %in% c("1", "true", "t", "yes", "y")
}

# "pilot" controls defaults only. Every major parameter can be overridden
# independently, so increasing repeats does not force selection of more anchors.
ANALYSIS_MODE <- tolower(Sys.getenv("VP_ANALYSIS_MODE", unset = "pilot"))
if (!ANALYSIS_MODE %in% c("pilot", "full")) ANALYSIS_MODE <- "pilot"

COMMON_CELLTYPES <- c(
  "Fibroblast", "AT2", "Macrophage", "Monocyte",
  "Dendritic", "Endothelial", "T_cell", "B_cell"
)
TARGET_DISEASES <- c("Control", "IPF")

if (ANALYSIS_MODE == "pilot") {
  default_top_targets <- 1L
  default_network_repeats <- 1L
  default_network_gene_limit <- 1000L
  default_network_cell_limit <- 600L
  default_network_n_net <- 5L
  default_network_n_cells <- 400L
  default_celloracle_gene_limit <- 1800L
  default_celloracle_cell_limit <- 1500L
  default_celloracle_top_tfs <- 3L
} else {
  default_top_targets <- 5L
  default_network_repeats <- 3L
  default_network_gene_limit <- 2000L
  default_network_cell_limit <- 1500L
  default_network_n_net <- 10L
  default_network_n_cells <- 800L
  default_celloracle_gene_limit <- 2500L
  default_celloracle_cell_limit <- 3000L
  default_celloracle_top_tfs <- 5L
}

TOP_TARGETS_PER_CELLTYPE <- as_int_env("VP_TOP_TARGETS_PER_CELLTYPE", default_top_targets, 1L)
NETWORK_REPEATS <- as_int_env("VP_NETWORK_REPEATS", default_network_repeats, 1L)
NETWORK_GENE_LIMIT <- as_int_env("VP_NETWORK_GENE_LIMIT", default_network_gene_limit, 100L)
NETWORK_CELL_LIMIT <- as_int_env("VP_NETWORK_CELL_LIMIT", default_network_cell_limit, 50L)
NETWORK_N_NET <- as_int_env("VP_NETWORK_N_NET", default_network_n_net, 2L)
NETWORK_N_CELLS <- as_int_env("VP_NETWORK_N_CELLS", default_network_n_cells, 50L)
CELLORACLE_GENE_LIMIT <- as_int_env("VP_CELLORACLE_GENE_LIMIT", default_celloracle_gene_limit, 100L)
CELLORACLE_CELL_LIMIT_PER_CONDITION <- as_int_env(
  "VP_CELLORACLE_CELL_LIMIT_PER_CONDITION", default_celloracle_cell_limit, 50L
)
CELLORACLE_TOP_TFS_PER_ANCHOR <- as_int_env(
  "VP_CELLORACLE_TOP_TFS_PER_ANCHOR", default_celloracle_top_tfs, 1L
)

MIN_GROUP_CELLS <- as_int_env("VP_MIN_GROUP_CELLS", 200L, 1L)
MIN_GROUP_SAMPLES <- as_int_env("VP_MIN_GROUP_SAMPLES", 2L, 1L)
MIN_TARGET_EXPRESSED_CELLS <- as_int_env("VP_MIN_TARGET_EXPRESSED_CELLS", 25L, 1L)
MIN_GENE_EXPRESSED_CELLS <- as_int_env("VP_MIN_GENE_EXPRESSED_CELLS", 25L, 1L)

# Strict by default: partial inputs/results must not silently enter final ranking.
ALLOW_PARTIAL_INPUTS <- as_bool_env("VP_ALLOW_PARTIAL_INPUTS", FALSE)
ALLOW_PARTIAL_VALIDATION <- as_bool_env("VP_ALLOW_PARTIAL_VALIDATION", FALSE)
ALLOW_MISSING_TF_UNIVERSE <- as_bool_env("VP_ALLOW_MISSING_TF_UNIVERSE", FALSE)

NETWORK_N_COMP <- as_int_env("VP_NETWORK_N_COMP", 3L, 2L)
NETWORK_EDGE_QUANTILE <- as_num_env("VP_NETWORK_EDGE_QUANTILE", 0.90, 0)
NETWORK_TENSOR_K <- as_int_env("VP_NETWORK_TENSOR_K", 3L, 2L)
NETWORK_MANIFOLD_DIM <- as_int_env("VP_NETWORK_MANIFOLD_DIM", 2L, 1L)
NETWORK_OE_EDGE_MULTIPLIER <- as_num_env("VP_NETWORK_OE_EDGE_MULTIPLIER", 1.5, 0)
NETWORK_N_CORES <- as_int_env(
  "VP_NETWORK_N_CORES", max(1L, parallel::detectCores(logical = FALSE) - 1L), 1L
)

CELLORACLE_N_JOBS <- as_int_env(
  "VP_CELLORACLE_N_JOBS", max(1L, parallel::detectCores(logical = FALSE) - 1L), 1L
)
CELLORACLE_OE_FOLD <- as_num_env("VP_CELLORACLE_OE_FOLD", 1.5, 0)
CELLORACLE_N_PROPAGATION <- as_int_env("VP_CELLORACLE_N_PROPAGATION", 3L, 1L)
CELLORACLE_RUN_VECTOR_FIELD <- as_bool_env("VP_CELLORACLE_RUN_VECTOR_FIELD", FALSE)
CELLORACLE_REQUIRE_TF_MANIFEST <- TRUE
CELLORACLE_MIN_TF_EXPRESSED_CELLS <- MIN_TARGET_EXPRESSED_CELLS

set.seed(20260723)
dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)
