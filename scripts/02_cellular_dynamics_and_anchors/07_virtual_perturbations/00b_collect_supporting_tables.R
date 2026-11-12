#!/usr/bin/env Rscript
# ============================================================
# Collect supporting CSV tables required by the virtual
# perturbation pipeline. This script is intentionally separate
# from 00_reexport_main_seurat_v5_layers.R, so the expensive
# Seurat export does not need to be repeated.
# ============================================================

# ---- User settings -------------------------------------------------------
SOURCE_PROJECT_DIR <- "E:/Rproject"
OUTPUT_ROOT <- file.path(SOURCE_PROJECT_DIR, "virtual_perturbation_inputs")

# Optional: set exact source paths here. Leave as "" to auto-search.
CANDIDATE_TABLE_MANUAL <- ""
DYNAMIC_TABLE_MANUAL <- ""

# Candidate-table priority. The first existing usable table is selected.
CANDIDATE_BASENAMES <- c(
  "07_final_ppi_informed_key_driver_candidates.csv",
  "16_tierA_primary_driver_candidates.csv",
  "15_final_driver_shortlist_top50.csv",
  "12_global_top_key_drivers.csv"
)
DYNAMIC_BASENAME <- "00_combined_dynamic_genes_all_celltypes.csv"

SUPPORT_DIR <- file.path(OUTPUT_ROOT, "supporting_tables")
dir.create(SUPPORT_DIR, recursive = TRUE, showWarnings = FALSE)

normalize_path_safe <- function(x) {
  normalizePath(x, winslash = "/", mustWork = FALSE)
}

find_by_basename <- function(root, basename_target) {
  if (!dir.exists(root)) return(character())
  hits <- list.files(
    root,
    pattern = glob2rx(basename_target),
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  hits <- hits[file.exists(hits)]
  # Do not select a file already located in the destination directory.
  support_norm <- normalize_path_safe(SUPPORT_DIR)
  hits_norm <- vapply(hits, normalize_path_safe, character(1))
  hits[!startsWith(tolower(hits_norm), tolower(paste0(support_norm, "/")))]
}

first_existing <- function(paths) {
  paths <- unique(paths[nzchar(paths)])
  hit <- paths[file.exists(paths)]
  if (length(hit)) hit[[1]] else ""
}

pick_column <- function(nms, aliases) {
  idx <- match(tolower(aliases), tolower(nms), nomatch = 0L)
  idx <- idx[idx > 0L]
  if (length(idx)) nms[idx[[1]]] else ""
}

standardize_candidate_table <- function(path) {
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!nrow(x)) stop("Candidate table has zero rows: ", path)

  celltype_col <- pick_column(colnames(x), c(
    "celltype", "cell_type", "cell type", "celltype_marker_based",
    "major_celltype", "major_cell_type"
  ))
  gene_col <- pick_column(colnames(x), c(
    "gene", "gene_symbol", "symbol", "target_gene", "driver_gene",
    "key_driver", "gene_name"
  ))
  if (!nzchar(celltype_col) || !nzchar(gene_col)) {
    stop(
      "Selected candidate table does not contain recognizable cell-type and gene columns: ",
      path, "\nColumns found: ", paste(colnames(x), collapse = ", ")
    )
  }

  if (celltype_col != "celltype") x$celltype <- x[[celltype_col]]
  if (gene_col != "gene") x$gene <- x[[gene_col]]
  x$celltype <- trimws(as.character(x$celltype))
  x$gene <- trimws(as.character(x$gene))
  x <- x[nzchar(x$celltype) & nzchar(x$gene), , drop = FALSE]
  x <- x[!duplicated(paste(x$celltype, x$gene, sep = "||")), , drop = FALSE]
  if (!nrow(x)) stop("No valid celltype-gene rows remain after cleaning: ", path)
  x
}

# ---- Locate candidate table ---------------------------------------------
candidate_source <- first_existing(CANDIDATE_TABLE_MANUAL)
if (!nzchar(candidate_source)) {
  for (nm in CANDIDATE_BASENAMES) {
    hits <- find_by_basename(SOURCE_PROJECT_DIR, nm)
    if (length(hits)) {
      candidate_source <- hits[[1]]
      break
    }
  }
}

status <- list()
if (nzchar(candidate_source)) {
  candidate_data <- standardize_candidate_table(candidate_source)
  candidate_dest <- file.path(SUPPORT_DIR, "07_final_ppi_informed_key_driver_candidates.csv")
  write.csv(candidate_data, candidate_dest, row.names = FALSE, na = "")
  status[["candidate"]] <- data.frame(
    table_role = "required_candidate_table",
    status = "copied_and_standardized",
    source = normalize_path_safe(candidate_source),
    destination = normalize_path_safe(candidate_dest),
    n_rows = nrow(candidate_data),
    stringsAsFactors = FALSE
  )
  message("Candidate table prepared: ", candidate_dest)
} else {
  status[["candidate"]] <- data.frame(
    table_role = "required_candidate_table",
    status = "missing",
    source = "",
    destination = normalize_path_safe(file.path(SUPPORT_DIR, "07_final_ppi_informed_key_driver_candidates.csv")),
    n_rows = 0L,
    stringsAsFactors = FALSE
  )
  warning(
    "No candidate table was found. Set CANDIDATE_TABLE_MANUAL to the exact CSV path. ",
    "The pipeline cannot run 01_validate_inputs.R or 02_build_target_manifest.R without it."
  )
}

# ---- Locate dynamic-gene table ------------------------------------------
dynamic_source <- first_existing(DYNAMIC_TABLE_MANUAL)
if (!nzchar(dynamic_source)) {
  dynamic_hits <- find_by_basename(SOURCE_PROJECT_DIR, DYNAMIC_BASENAME)
  if (length(dynamic_hits)) dynamic_source <- dynamic_hits[[1]]
}

if (nzchar(dynamic_source)) {
  dynamic_data <- read.csv(dynamic_source, stringsAsFactors = FALSE, check.names = FALSE)
  dynamic_dest <- file.path(SUPPORT_DIR, DYNAMIC_BASENAME)
  write.csv(dynamic_data, dynamic_dest, row.names = FALSE, na = "")
  status[["dynamic"]] <- data.frame(
    table_role = "recommended_dynamic_gene_table",
    status = "copied",
    source = normalize_path_safe(dynamic_source),
    destination = normalize_path_safe(dynamic_dest),
    n_rows = nrow(dynamic_data),
    stringsAsFactors = FALSE
  )
  message("Dynamic-gene table copied: ", dynamic_dest)
} else {
  status[["dynamic"]] <- data.frame(
    table_role = "recommended_dynamic_gene_table",
    status = "missing_optional",
    source = "",
    destination = normalize_path_safe(file.path(SUPPORT_DIR, DYNAMIC_BASENAME)),
    n_rows = 0L,
    stringsAsFactors = FALSE
  )
  warning(
    "Dynamic-gene table was not found. Network perturbation can still run, ",
    "but dynamic-gene prioritization and CellOracle disease-axis scoring will be unavailable."
  )
}

status_df <- do.call(rbind, status)
status_path <- file.path(SUPPORT_DIR, "00_supporting_tables_status.csv")
write.csv(status_df, status_path, row.names = FALSE, na = "")
message("Supporting-table status written to: ", status_path)

candidate_ready <- file.exists(file.path(SUPPORT_DIR, "07_final_ppi_informed_key_driver_candidates.csv"))
if (!candidate_ready) {
  stop(
    "Required candidate table is still missing. Edit CANDIDATE_TABLE_MANUAL in ",
    "00b_collect_supporting_tables.R and run this script again."
  )
}
message("Supporting tables are ready: ", SUPPORT_DIR)
