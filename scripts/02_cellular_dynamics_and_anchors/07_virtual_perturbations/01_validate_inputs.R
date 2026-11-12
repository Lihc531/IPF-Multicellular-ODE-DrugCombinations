#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
})
source("config.R")

OUT <- file.path(OUTPUT_ROOT, "00_input_validation")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

inspect_bundle <- function(path) {
  x <- readRDS(path)
  required <- c("dataset", "celltype", "disease", "counts", "metadata")
  missing <- setdiff(required, names(x))
  if (length(missing)) stop("Bundle missing fields in ", path, ": ", paste(missing, collapse = ", "))
  if (!inherits(x$counts, "sparseMatrix")) stop("counts is not sparse in ", path)
  if (is.null(rownames(x$counts)) || is.null(colnames(x$counts))) stop("counts lacks dimnames in ", path)
  if (anyDuplicated(rownames(x$counts))) stop("Duplicated gene names in ", path)
  if (anyDuplicated(colnames(x$counts))) stop("Duplicated cell IDs in ", path)
  if (!all(colnames(x$counts) %in% rownames(x$metadata))) stop("Metadata cell IDs do not match counts in ", path)
  sample_col <- if ("sample_for_perturbation" %in% colnames(x$metadata)) {
    "sample_for_perturbation"
  } else if (!is.null(x$source_sample_column) && x$source_sample_column %in% colnames(x$metadata)) {
    x$source_sample_column
  } else if ("sample_id" %in% colnames(x$metadata)) {
    "sample_id"
  } else {
    NA_character_
  }
  n_samples <- if (is.na(sample_col)) 1L else length(unique(as.character(x$metadata[[sample_col]])))
  data.frame(
    dataset = as.character(x$dataset),
    celltype = as.character(x$celltype),
    disease = as.character(x$disease),
    n_genes = nrow(x$counts),
    n_cells = ncol(x$counts),
    n_samples = n_samples,
    sample_column = ifelse(is.na(sample_col), "<missing>", sample_col),
    file = normalizePath(path, winslash = "/", mustWork = TRUE),
    stringsAsFactors = FALSE
  )
}

files <- c(
  list.files(MAIN_DIR, pattern = "\\.rds$", full.names = TRUE),
  list.files(VALIDATION_DIR, pattern = "\\.rds$", full.names = TRUE)
)
if (!length(files)) stop("No grouped RDS files found under INPUT_ROOT: ", INPUT_ROOT)

availability <- do.call(rbind, lapply(files, inspect_bundle))
availability <- availability[order(availability$celltype, availability$dataset, availability$disease), ]
availability$enough_cells <- availability$n_cells >= MIN_GROUP_CELLS
availability$enough_samples <- availability$n_samples >= MIN_GROUP_SAMPLES
availability$group_eligible <- availability$enough_cells & availability$enough_samples
write.csv(availability, file.path(OUT, "group_availability.csv"), row.names = FALSE)

required_grid <- expand.grid(
  dataset = c("main", "validation"),
  celltype = COMMON_CELLTYPES,
  disease = TARGET_DISEASES,
  stringsAsFactors = FALSE
)
required_grid <- merge(
  required_grid,
  availability[, c("dataset", "celltype", "disease", "n_cells", "n_samples", "file", "group_eligible")],
  by = c("dataset", "celltype", "disease"), all.x = TRUE, sort = FALSE
)
required_grid$available <- !is.na(required_grid$file)
required_grid$group_eligible[is.na(required_grid$group_eligible)] <- FALSE
required_grid$status <- ifelse(
  !required_grid$available, "missing",
  ifelse(required_grid$n_cells < MIN_GROUP_CELLS, "too_few_cells",
    ifelse(required_grid$n_samples < MIN_GROUP_SAMPLES, "too_few_samples", "ok")
  )
)
required_grid <- required_grid[order(required_grid$celltype, required_grid$dataset, required_grid$disease), ]
write.csv(required_grid, file.path(OUT, "required_group_grid.csv"), row.names = FALSE)

bad <- required_grid[required_grid$status != "ok", , drop = FALSE]
if (nrow(bad)) {
  write.csv(bad, file.path(OUT, "blocking_input_groups.csv"), row.names = FALSE)
  msg <- paste(
    paste(bad$dataset, bad$celltype, bad$disease, bad$status, sep = "/"),
    collapse = "; "
  )
  if (!ALLOW_PARTIAL_INPUTS) {
    stop(
      "Required input groups are incomplete or underpowered: ", msg,
      ". See ", file.path(OUT, "blocking_input_groups.csv"),
      ". Set VP_ALLOW_PARTIAL_INPUTS=true only for exploratory runs."
    )
  }
  warning("Proceeding with partial inputs because VP_ALLOW_PARTIAL_INPUTS=true: ", msg)
}

candidate_path <- file.path(SUPPORT_DIR, "07_final_ppi_informed_key_driver_candidates.csv")
if (!file.exists(candidate_path)) stop(
  "Candidate table is missing: ", candidate_path,
  ". Run 00b_collect_supporting_tables.R, or manually copy the candidate CSV to supporting_tables."
)
message("Input validation completed. Report: ", OUT)
