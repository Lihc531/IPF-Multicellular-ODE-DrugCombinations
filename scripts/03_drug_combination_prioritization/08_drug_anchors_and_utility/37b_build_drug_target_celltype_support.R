#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(Matrix))

# Build drug-target-cell-type support from the same main/validation expression
# bundles used by the virtual perturbation pipeline.
#
# Example:
# Rscript 37b_build_drug_target_celltype_support.R \
#   --target-table E:/PythonProject/results/37_SIGCOM/02e_targets_main_consolidated.csv \
#   --pipeline-root E:/Rproject/virtual_perturbation_pipeline

parse_args <- function(x) {
  out <- list()
  i <- 1L
  while (i <= length(x)) {
    key <- sub("^--", "", x[[i]])
    if (i == length(x) || startsWith(x[[i + 1L]], "--")) {
      out[[key]] <- TRUE; i <- i + 1L
    } else {
      out[[key]] <- x[[i + 1L]]; i <- i + 2L
    }
  }
  out
}
args <- parse_args(commandArgs(trailingOnly = TRUE))
root <- normalizePath(if (!is.null(args[["pipeline-root"]])) args[["pipeline-root"]] else getwd(), winslash = "/", mustWork = FALSE)
source(file.path(root, "config.R"))
target_path <- if (!is.null(args[["target-table"]])) args[["target-table"]] else file.path("E:/PythonProject/results/37a_DRUG_PAIRING_RESULTS", "02e_targets_main_consolidated.csv")
out_dir <- if (!is.null(args[["output-dir"]])) args[["output-dir"]] else file.path("E:/PythonProject/results", "37b_DRUG_TARGET_CELLTYPE_SUPPORT")
min_expr_fraction <- if (!is.null(args[["min-expr-fraction"]])) as.numeric(args[["min-expr-fraction"]]) else 0.10
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(target_path)) stop("Target table not found: ", target_path)

x <- read.csv(target_path, stringsAsFactors = FALSE, check.names = FALSE)
pick <- function(nms, aliases) {
  idx <- match(tolower(aliases), tolower(nms), nomatch = 0L); idx <- idx[idx > 0L]
  if (length(idx)) nms[idx[[1L]]] else ""
}
drug_col <- pick(colnames(x), c("drug_id", "drug", "drug_name", "pert_id"))
drug_name_col <- pick(colnames(x), c("drug_name", "drug", "pert_name"))
target_col <- pick(colnames(x), c("target_gene", "gene_symbol", "gene", "target"))
action_col <- pick(colnames(x), c("action", "action_type", "mechanism_of_action", "moa", "target_action"))
if (!nzchar(drug_col) || !nzchar(target_col)) stop("Could not detect drug and target columns.")
targets <- data.frame(
  drug_id = trimws(as.character(x[[drug_col]])),
  drug = if (nzchar(drug_name_col)) trimws(as.character(x[[drug_name_col]])) else trimws(as.character(x[[drug_col]])),
  target_gene = toupper(trimws(as.character(x[[target_col]]))),
  action = if (nzchar(action_col)) trimws(as.character(x[[action_col]])) else "unknown",
  stringsAsFactors = FALSE
)
targets <- unique(targets[nzchar(targets$drug_id) & nzchar(targets$target_gene), , drop = FALSE])

bundle_path <- function(dataset, celltype, disease) {
  base <- if (dataset == "main") MAIN_DIR else VALIDATION_DIR
  file.path(base, paste0(dataset, "__", celltype, "__", disease, ".rds"))
}
all_files <- c(list.files(MAIN_DIR, pattern = "^main__.*__(Control|IPF)\\.rds$", full.names = FALSE),
               list.files(VALIDATION_DIR, pattern = "^validation__.*__(Control|IPF)\\.rds$", full.names = FALSE))
celltypes <- unique(sub("^(main|validation)__(.*)__(Control|IPF)\\.rds$", "\\2", all_files))
celltypes <- celltypes[nzchar(celltypes)]
if (!length(celltypes)) stop("No expression bundles found in MAIN_DIR/VALIDATION_DIR.")

stats_one <- function(bundle, gene) {
  idx <- match(gene, toupper(rownames(bundle$counts)))
  n <- ncol(bundle$counts)
  if (is.na(idx) || n == 0) return(c(present = 0, n_cells = n, n_expr = 0, expr_fraction = 0, mean_log1p = NA_real_))
  v <- as.numeric(bundle$counts[idx, ])
  c(present = 1, n_cells = n, n_expr = sum(v > 0), expr_fraction = mean(v > 0), mean_log1p = mean(log1p(v)))
}

rows <- list(); k <- 0L
for (ct in celltypes) {
  bundles <- list()
  complete <- TRUE
  for (ds in c("main", "validation")) for (dis in c("Control", "IPF")) {
    p <- bundle_path(ds, ct, dis)
    if (!file.exists(p)) { complete <- FALSE; next }
    bundles[[paste(ds, dis, sep = "__")]] <- readRDS(p)
  }
  if (!complete) next
  for (i in seq_len(nrow(targets))) {
    t <- targets[i, , drop = FALSE]
    st <- lapply(bundles, stats_one, gene = t$target_gene)
    main_lfc <- st[["main__IPF"]]["mean_log1p"] - st[["main__Control"]]["mean_log1p"]
    val_lfc <- st[["validation__IPF"]]["mean_log1p"] - st[["validation__Control"]]["mean_log1p"]
    direction_concordant <- is.finite(main_lfc) && is.finite(val_lfc) && (sign(main_lfc) == sign(val_lfc) || abs(main_lfc) < 1e-8 || abs(val_lfc) < 1e-8)
    min_frac <- min(vapply(st, function(z) z["expr_fraction"], numeric(1)), na.rm = TRUE)
    avg_frac <- mean(vapply(st, function(z) z["expr_fraction"], numeric(1)), na.rm = TRUE)
    cross_present <- all(vapply(st, function(z) z["present"] > 0, logical(1)))
    score <- 0.50 * pmin(1, avg_frac / 0.50) + 0.25 * as.numeric(cross_present) + 0.25 * as.numeric(direction_concordant)
    k <- k + 1L
    rows[[k]] <- data.frame(
      drug_id = t$drug_id, drug = t$drug, target_gene = t$target_gene, action = t$action, celltype = ct,
      main_control_expr_fraction = st[["main__Control"]]["expr_fraction"], main_ipf_expr_fraction = st[["main__IPF"]]["expr_fraction"],
      validation_control_expr_fraction = st[["validation__Control"]]["expr_fraction"], validation_ipf_expr_fraction = st[["validation__IPF"]]["expr_fraction"],
      main_log1p_mean_diff = main_lfc, validation_log1p_mean_diff = val_lfc,
      cross_dataset_present = cross_present, disease_direction_concordant = direction_concordant,
      min_expr_fraction = min_frac, celltype_support_score = score,
      celltype_supported = cross_present && min_frac >= min_expr_fraction,
      evidence_type = "single_cell_expression_support_not_target_engagement",
      stringsAsFactors = FALSE
    )
  }
}
out <- if (length(rows)) do.call(rbind, rows) else data.frame()
if (!nrow(out)) stop("No drug-target-celltype rows were generated.")
out <- out[order(out$drug_id, -out$celltype_support_score), , drop = FALSE]
write.csv(out, file.path(out_dir, "drug_target_celltype_support.csv"), row.names = FALSE)
best <- do.call(rbind, lapply(split(out, out$drug_id), function(z) z[order(-z$celltype_support_score), , drop = FALSE][1, ]))
write.csv(best, file.path(out_dir, "drug_target_best_celltype.csv"), row.names = FALSE)
writeLines(c(
  paste0("target_table=", normalizePath(target_path, winslash = "/", mustWork = FALSE)),
  paste0("n_drugs=", length(unique(out$drug_id))),
  paste0("n_rows=", nrow(out)),
  paste0("min_expr_fraction=", min_expr_fraction)
), file.path(out_dir, "run_metadata.txt"))
message("Drug-target-celltype support written to: ", out_dir)
