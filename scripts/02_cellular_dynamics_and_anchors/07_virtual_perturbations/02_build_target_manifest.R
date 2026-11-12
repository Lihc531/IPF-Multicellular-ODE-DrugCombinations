#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
})
source("config.R")

OUT <- file.path(OUTPUT_ROOT, "01_target_manifest")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

candidate_path <- file.path(SUPPORT_DIR, "07_final_ppi_informed_key_driver_candidates.csv")
cand <- read.csv(candidate_path, stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("celltype", "gene") %in% colnames(cand))) stop("Candidate table needs celltype and gene columns.")
cand$celltype <- trimws(as.character(cand$celltype))
cand$gene <- trimws(as.character(cand$gene))
cand <- cand[nzchar(cand$celltype) & nzchar(cand$gene), , drop = FALSE]

score_col <- if ("key_driver_score" %in% colnames(cand)) {
  "key_driver_score"
} else if ("dynamic_score" %in% colnames(cand)) {
  "dynamic_score"
} else {
  NULL
}
if (is.null(score_col)) cand$selection_score <- 0 else cand$selection_score <- suppressWarnings(as.numeric(cand[[score_col]]))
cand$selection_score[!is.finite(cand$selection_score)] <- 0

if ("non_protein_like_gene" %in% colnames(cand)) {
  cand <- cand[!tolower(as.character(cand$non_protein_like_gene)) %in% c("true", "t", "1", "yes"), ]
}
if ("keep_for_ppi_final" %in% colnames(cand)) {
  keep <- tolower(trimws(as.character(cand$keep_for_ppi_final)))
  cand <- cand[is.na(keep) | keep == "" | keep %in% c("yes", "true", "1"), ]
}
cand <- cand[cand$celltype %in% COMMON_CELLTYPES, , drop = FALSE]

bundle_path <- function(dataset, celltype, disease) {
  root <- if (dataset == "main") MAIN_DIR else VALIDATION_DIR
  file.path(root, paste0(dataset, "__", celltype, "__", disease, ".rds"))
}

expression_eligibility <- function(celltype, genes) {
  out <- data.frame(gene = genes, stringsAsFactors = FALSE)
  for (dataset in c("main", "validation")) {
    for (disease in TARGET_DISEASES) {
      path <- bundle_path(dataset, celltype, disease)
      col_nm <- paste(dataset, disease, "n_expr_cells", sep = "_")
      out[[col_nm]] <- NA_integer_
      if (!file.exists(path)) next
      x <- readRDS(path)
      out[[col_nm]] <- 0L
      hit <- intersect(genes, rownames(x$counts))
      if (length(hit)) {
        vals <- Matrix::rowSums(x$counts[hit, , drop = FALSE] > 0)
        out[[col_nm]][match(hit, out$gene)] <- as.integer(vals)
      }
      rm(x); gc(verbose = FALSE)
    }
  }
  expr_cols <- grep("_n_expr_cells$", colnames(out), value = TRUE)
  out$n_groups_present <- rowSums(!is.na(out[, expr_cols, drop = FALSE]))
  out$n_groups_passing_expression <- rowSums(
    !is.na(out[, expr_cols, drop = FALSE]) & out[, expr_cols, drop = FALSE] >= MIN_TARGET_EXPRESSED_CELLS
  )
  out$eligible_all_groups <- out$n_groups_passing_expression == length(expr_cols)
  out
}

eligibility_rows <- list()
selected <- list()
status_rows <- list()
for (ct in COMMON_CELLTYPES) {
  sub <- cand[cand$celltype == ct, , drop = FALSE]
  if (!nrow(sub)) {
    status_rows[[ct]] <- data.frame(
      celltype = ct, n_candidates_input = 0L, n_candidates_eligible = 0L,
      n_targets_selected = 0L, status = "no_candidate_rows", stringsAsFactors = FALSE
    )
    next
  }
  sub <- sub[order(sub$selection_score, decreasing = TRUE), ]
  elig <- expression_eligibility(ct, unique(sub$gene))
  elig$celltype <- ct
  eligibility_rows[[ct]] <- elig
  sub <- merge(sub, elig, by = c("celltype", "gene"), all.x = TRUE, sort = FALSE)
  eligible_sub <- sub[!is.na(sub$eligible_all_groups) & sub$eligible_all_groups, , drop = FALSE]
  if (nrow(eligible_sub)) {
    eligible_sub <- eligible_sub[order(eligible_sub$selection_score, decreasing = TRUE), ]
    eligible_sub <- eligible_sub[!duplicated(eligible_sub$gene), ]
    eligible_sub <- head(eligible_sub, TOP_TARGETS_PER_CELLTYPE)
    selected[[ct]] <- eligible_sub
  }
  status_rows[[ct]] <- data.frame(
    celltype = ct,
    n_candidates_input = length(unique(sub$gene)),
    n_candidates_eligible = length(unique(sub$gene[!is.na(sub$eligible_all_groups) & sub$eligible_all_groups])),
    n_targets_selected = if (ct %in% names(selected)) nrow(selected[[ct]]) else 0L,
    status = if (ct %in% names(selected)) "ok" else "no_expression_eligible_candidate",
    stringsAsFactors = FALSE
  )
}

if (length(eligibility_rows)) {
  eligibility_report <- do.call(rbind, eligibility_rows)
  write.csv(
    eligibility_report,
    file.path(OUT, "candidate_expression_eligibility.csv"),
    row.names = FALSE
  )
}
selection_status <- do.call(rbind, status_rows)
write.csv(selection_status, file.path(OUT, "celltype_target_selection_status.csv"), row.names = FALSE)

missing_ct <- selection_status$celltype[selection_status$n_targets_selected < 1L]
if (length(missing_ct) && !ALLOW_PARTIAL_INPUTS) {
  stop(
    "No target was selected for required cell types: ", paste(missing_ct, collapse = ", "),
    ". Review candidate_expression_eligibility.csv and input group coverage."
  )
}
if (!length(selected)) stop("No candidates were expressed sufficiently in all main/validation Control/IPF groups.")

manifest <- do.call(rbind, selected)
rownames(manifest) <- NULL

axis_direction <- rep(NA_real_, nrow(manifest))
if ("spearman_axis" %in% colnames(manifest)) axis_direction <- suppressWarnings(as.numeric(manifest$spearman_axis))
if (all(!is.finite(axis_direction)) && "linear_slope" %in% colnames(manifest)) axis_direction <- suppressWarnings(as.numeric(manifest$linear_slope))
axis_direction[!is.finite(axis_direction)] <- 0
manifest$axis_direction <- axis_direction
manifest$biological_role <- ifelse(axis_direction >= 0, "disease_increasing", "disease_decreasing")
manifest$ko_condition <- ifelse(axis_direction >= 0, "IPF", "Control")
manifest$oe_condition <- ifelse(axis_direction >= 0, "Control", "IPF")
manifest$primary_test <- ifelse(axis_direction >= 0, "KO_in_IPF_and_OE_in_Control", "OE_in_IPF_and_KO_in_Control")
manifest$anchor_gene <- manifest$gene
manifest$analysis_role <- "original_key_driver_or_effect_anchor"
manifest$target_id <- paste(manifest$celltype, manifest$gene, sep = "__")

keep_cols <- unique(c(
  "target_id", "celltype", "gene", "anchor_gene", "analysis_role", "selection_score", "key_driver_score",
  "dynamic_module", "trend_class", "spearman_axis", "linear_slope",
  "axis_direction", "biological_role", "ko_condition", "oe_condition", "primary_test",
  grep("_n_expr_cells$", colnames(manifest), value = TRUE)
))
keep_cols <- keep_cols[keep_cols %in% colnames(manifest)]
manifest <- manifest[, keep_cols, drop = FALSE]
write.csv(manifest, file.path(OUT, "target_manifest.csv"), row.names = FALSE)

manual <- manifest[, c("celltype", "gene", "biological_role", "ko_condition", "oe_condition"), drop = FALSE]
manual$include <- "yes"
manual$manual_notes <- ""
write.csv(manual, file.path(OUT, "manual_target_review.csv"), row.names = FALSE)
message("Target manifest created: ", file.path(OUT, "target_manifest.csv"))
