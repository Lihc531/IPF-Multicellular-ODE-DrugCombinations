#!/usr/bin/env Rscript
source("config.R")

NET_SUMMARY <- file.path(
  OUTPUT_ROOT, "02_network_perturbation", "summary",
  "network_perturbation_gene_summary.csv"
)
CELLORACLE_ROOT <- file.path(OUTPUT_ROOT, "04_celloracle")
TARGET_MANIFEST_PATH <- file.path(OUTPUT_ROOT, "01_target_manifest", "target_manifest.csv")
TF_TARGET_MANIFEST_PATH <- file.path(OUTPUT_ROOT, "01_target_manifest", "celloracle_tf_target_manifest.csv")
OUT <- file.path(OUTPUT_ROOT, "05_cross_dataset_validation")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

ZERO_TOL <- suppressWarnings(as.numeric(Sys.getenv("VP_ZERO_TOL", "1e-8")))
if (!is.finite(ZERO_TOL) || ZERO_TOL < 0) ZERO_TOL <- 1e-8
MIN_COMMON_GENES <- suppressWarnings(as.integer(Sys.getenv("VP_MIN_COMMON_GENES", "10")))
if (!is.finite(MIN_COMMON_GENES) || MIN_COMMON_GENES < 3L) MIN_COMMON_GENES <- 10L
TOP_N <- suppressWarnings(as.integer(Sys.getenv("VP_TOP_N", "100")))
if (!is.finite(TOP_N) || TOP_N < 1L) TOP_N <- 100L

# Parse the partial-validation switch locally so this script remains explicit
# even when config.R does not define ALLOW_PARTIAL_VALIDATION. Environment
# variable values override any fallback value supplied by config.R.
parse_bool_env <- function(name, fallback = FALSE) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(trimws(raw))) return(isTRUE(fallback))
  tolower(trimws(raw)) %in% c("1", "true", "t", "yes", "y", "on")
}
ALLOW_PARTIAL_VALIDATION <- parse_bool_env(
  "VP_ALLOW_PARTIAL_VALIDATION",
  if (exists("ALLOW_PARTIAL_VALIDATION", inherits = TRUE)) {
    get("ALLOW_PARTIAL_VALIDATION", inherits = TRUE)
  } else {
    FALSE
  }
)

# Remove stale files that could otherwise be mistaken for current-run outputs.
stale <- c(
  "network_cross_dataset_metrics.csv",
  "celloracle_tf_cross_dataset_metrics.csv",
  "celloracle_tf_ranking_within_anchor.csv",
  "celloracle_anchor_evidence_summary.csv",
  "final_cross_dataset_candidate_ranking.csv",
  "cross_dataset_validation_diagnostics.csv",
  "network_validation_coverage.csv",
  "celloracle_validation_coverage.csv",
  "celloracle_primary_direction_summary.csv",
  "celloracle_missing_tests.csv",
  "celloracle_missing_jobs_long.csv",
  "celloracle_missing_dataset_jobs.csv",
  "celloracle_missing_job_summary.csv",
  "celloracle_non_evaluable_tests.csv",
  "celloracle_non_evaluable_pairs.csv",
  "celloracle_hard_ineligible_pairs.csv",
  "celloracle_formal_completeness_summary.csv"
)
unlink(file.path(OUT, stale), force = TRUE)

diagnostics <- data.frame(
  stage = character(), status = character(), n_rows = integer(), detail = character(),
  stringsAsFactors = FALSE
)
validation_blockers <- character()
coverage_audit <- list()
add_blocker <- function(x) validation_blockers <<- unique(c(validation_blockers, as.character(x)))

add_diag <- function(stage, status, n_rows = NA_integer_, detail = "") {
  diagnostics <<- rbind(
    diagnostics,
    data.frame(
      stage = as.character(stage), status = as.character(status),
      n_rows = as.integer(n_rows), detail = as.character(detail),
      stringsAsFactors = FALSE
    )
  )
}

mean_finite <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (!length(x)) NA_real_ else mean(x)
}

safe_rbind <- function(rows) {
  rows <- Filter(function(x) !is.null(x) && nrow(x), rows)
  if (!length(rows)) data.frame() else do.call(rbind, rows)
}

safe_cor <- function(x, y, min_n = MIN_COMMON_GENES) {
  x <- suppressWarnings(as.numeric(x))
  y <- suppressWarnings(as.numeric(y))
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < min_n || stats::sd(x[ok]) == 0 || stats::sd(y[ok]) == 0) {
    return(NA_real_)
  }
  suppressWarnings(stats::cor(x[ok], y[ok], method = "spearman"))
}

# Only positive correlation is counted as reproducible evidence. This puts the
# correlation component on the same 0-1 direction as Jaccard/concordance scores.
positive_correlation_evidence <- function(rho) {
  if (!is.finite(rho)) return(NA_real_)
  max(0, min(1, rho))
}

jaccard <- function(a, b) {
  a <- unique(as.character(a[!is.na(a) & nzchar(as.character(a))]))
  b <- unique(as.character(b[!is.na(b) & nzchar(as.character(b))]))
  u <- union(a, b)
  if (!length(u)) return(NA_real_)
  length(intersect(a, b)) / length(u)
}

safe_sign_concordance <- function(x, y, tol = ZERO_TOL) {
  x <- suppressWarnings(as.numeric(x))
  y <- suppressWarnings(as.numeric(y))
  ok <- is.finite(x) & is.finite(y) & abs(x) > tol & abs(y) > tol
  if (!any(ok)) return(NA_real_)
  mean(sign(x[ok]) == sign(y[ok]))
}

safe_direction_concordance <- function(a, b, tol = ZERO_TOL) {
  if (!is.finite(a) || !is.finite(b)) return(NA_real_)
  # A zero/near-zero effect has no interpretable direction.
  if (abs(a) <= tol || abs(b) <= tol) return(NA_real_)
  as.numeric(sign(a) == sign(b))
}

safe_magnitude_similarity <- function(a, b, tol = ZERO_TOL) {
  if (!is.finite(a) || !is.finite(b)) return(NA_real_)
  # Two null effects do not constitute positive evidence of replicated magnitude.
  if (abs(a) <= tol && abs(b) <= tol) return(NA_real_)
  if (abs(a) <= tol || abs(b) <= tol) return(0)
  max(0, min(1, 1 - abs(a - b) / (abs(a) + abs(b) + 1e-12)))
}

sign_match_score <- function(value, expected_sign, tol = ZERO_TOL) {
  value <- suppressWarnings(as.numeric(value))
  expected_sign <- suppressWarnings(as.numeric(expected_sign))
  if (!is.finite(value) || !is.finite(expected_sign) || expected_sign == 0 || abs(value) <= tol) return(NA_real_)
  as.numeric(sign(value) == sign(expected_sign))
}

weighted_mean_finite <- function(x, w) {
  x <- suppressWarnings(as.numeric(x)); w <- suppressWarnings(as.numeric(w))
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(NA_real_)
  sum(x[ok] * w[ok]) / sum(w[ok])
}

top_genes <- function(d, value_col, n = TOP_N) {
  if (!nrow(d) || !all(c("gene", value_col) %in% colnames(d))) return(character())
  value <- suppressWarnings(as.numeric(d[[value_col]]))
  ok <- is.finite(value) & !is.na(d$gene) & nzchar(as.character(d$gene))
  if (!any(ok)) return(character())
  gene <- as.character(d$gene[ok])
  value <- value[ok]
  ordered_gene <- gene[order(abs(value), decreasing = TRUE)]
  unique(ordered_gene[seq_len(min(n, length(ordered_gene)))])
}

# -----------------------------------------------------------------------------
# General network perturbation: original anchor/key-driver genes.
# -----------------------------------------------------------------------------
network_metrics <- data.frame()
if (!file.exists(NET_SUMMARY)) {
  add_diag("network_input", "missing", 0L, NET_SUMMARY)
  warning("Network summary not found: ", NET_SUMMARY)
} else {
  net <- tryCatch(
    read.csv(NET_SUMMARY, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) {
      add_diag("network_input", "read_error", 0L, conditionMessage(e))
      NULL
    }
  )
  if (!is.null(net)) {
    required <- c(
      "dataset", "celltype", "disease", "target_gene", "perturbation",
      "gene", "median_FC", "median_p_adj"
    )
    missing_cols <- setdiff(required, colnames(net))
    if (length(missing_cols)) {
      add_diag(
        "network_input", "missing_columns", nrow(net),
        paste(missing_cols, collapse = ", ")
      )
      warning("Network summary missing columns: ", paste(missing_cols, collapse = ", "))
    } else {
      add_diag("network_input", "ok", nrow(net), NET_SUMMARY)
      key <- unique(net[, c("celltype", "disease", "target_gene", "perturbation"), drop = FALSE])
      rows <- lapply(seq_len(nrow(key)), function(i) {
        k <- key[i, , drop = FALSE]
        m <- subset(
          net,
          dataset == "main" & celltype == k$celltype & disease == k$disease &
            target_gene == k$target_gene & perturbation == k$perturbation
        )
        v <- subset(
          net,
          dataset == "validation" & celltype == k$celltype & disease == k$disease &
            target_gene == k$target_gene & perturbation == k$perturbation
        )
        if (!nrow(m) || !nrow(v)) return(NULL)
        z <- merge(
          m[, c("gene", "median_FC", "median_p_adj"), drop = FALSE],
          v[, c("gene", "median_FC", "median_p_adj"), drop = FALSE],
          by = "gene", suffixes = c("_main", "_validation")
        )
        if (!nrow(z)) return(NULL)
        rho <- safe_cor(z$median_FC_main, z$median_FC_validation)
        top_m <- top_genes(m, "median_FC")
        top_v <- top_genes(v, "median_FC")
        p_m <- suppressWarnings(as.numeric(m$median_p_adj))
        p_v <- suppressWarnings(as.numeric(v$median_p_adj))
        sig_m <- as.character(m$gene[is.finite(p_m) & p_m < 0.05])
        sig_v <- as.character(v$gene[is.finite(p_v) & p_v < 0.05])
        data.frame(
          celltype = k$celltype,
          disease = k$disease,
          anchor_gene = k$target_gene,
          target_gene = k$target_gene,
          perturbation = k$perturbation,
          n_common_genes = nrow(z),
          spearman_FC = rho,
          spearman_FC_evidence = positive_correlation_evidence(rho),
          # median_FC is a non-negative perturbation-distance statistic, so a
          # sign-concordance score is not defined for this branch.
          sign_concordance = NA_real_,
          top100_jaccard = jaccard(top_m, top_v),
          significant_jaccard = jaccard(sig_m, sig_v),
          n_sig_main = length(unique(sig_m)),
          n_sig_validation = length(unique(sig_v)),
          stringsAsFactors = FALSE
        )
      })
      network_metrics <- safe_rbind(rows)
      if (nrow(network_metrics)) {
        network_component_cols <- c(
          "spearman_FC_evidence", "top100_jaccard", "significant_jaccard"
        )
        network_metrics$network_metric_count <- rowSums(
          is.finite(as.matrix(network_metrics[, network_component_cols, drop = FALSE]))
        )
        network_metrics$network_metric_coverage <-
          network_metrics$network_metric_count / length(network_component_cols)
        network_metrics$network_evidence_score <- apply(
          network_metrics[, network_component_cols, drop = FALSE],
          1, mean_finite
        )
        write.csv(
          network_metrics,
          file.path(OUT, "network_cross_dataset_metrics.csv"),
          row.names = FALSE
        )
        add_diag("network_metrics", "ok", nrow(network_metrics), "paired main-validation rows")
      } else {
        add_diag("network_metrics", "zero_matches", 0L, "No matched main-validation perturbation groups")
        warning("No matched main-validation network perturbation groups were found.")
      }
    }
  }
}

# -----------------------------------------------------------------------------
# CellOracle: upstream TFs mapped to original anchor genes.
# -----------------------------------------------------------------------------
celloracle_dirs <- if (dir.exists(CELLORACLE_ROOT)) {
  list.dirs(CELLORACLE_ROOT, recursive = FALSE, full.names = TRUE)
} else {
  character()
}

gene_files <- file.path(celloracle_dirs, "celloracle_gene_deltas.csv.gz")
gene_files <- gene_files[file.exists(gene_files)]
summary_files <- file.path(celloracle_dirs, "celloracle_perturbation_summary.csv")
summary_files <- summary_files[file.exists(summary_files)]
add_diag("celloracle_gene_files", if (length(gene_files)) "ok" else "missing", length(gene_files), CELLORACLE_ROOT)
add_diag("celloracle_summary_files", if (length(summary_files)) "ok" else "missing", length(summary_files), CELLORACLE_ROOT)

co_all <- data.frame()
celloracle_gene_metrics <- data.frame()
if (length(gene_files)) {
  co_all <- safe_rbind(lapply(gene_files, function(f) {
    tryCatch(
      read.csv(gzfile(f), stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) {
        warning("Could not read ", f, ": ", conditionMessage(e))
        NULL
      }
    )
  }))
  if (nrow(co_all)) {
    if (!"tf_gene" %in% colnames(co_all) && "target_gene" %in% colnames(co_all)) {
      co_all$tf_gene <- co_all$target_gene
    }
    if (!"anchor_gene" %in% colnames(co_all)) co_all$anchor_gene <- co_all$tf_gene
    required <- c(
      "dataset", "celltype", "condition", "anchor_gene", "tf_gene",
      "perturbation", "gene", "mean_delta"
    )
    missing_cols <- setdiff(required, colnames(co_all))
    if (!length(missing_cols)) {
      keys <- unique(co_all[, c(
        "celltype", "condition", "anchor_gene", "tf_gene", "perturbation"
      ), drop = FALSE])
      rows <- lapply(seq_len(nrow(keys)), function(i) {
        k <- keys[i, , drop = FALSE]
        m <- subset(
          co_all,
          dataset == "main" & celltype == k$celltype & condition == k$condition &
            anchor_gene == k$anchor_gene & tf_gene == k$tf_gene &
            perturbation == k$perturbation
        )
        v <- subset(
          co_all,
          dataset == "validation" & celltype == k$celltype & condition == k$condition &
            anchor_gene == k$anchor_gene & tf_gene == k$tf_gene &
            perturbation == k$perturbation
        )
        if (!nrow(m) || !nrow(v)) return(NULL)
        z <- merge(
          m[, c("gene", "mean_delta"), drop = FALSE],
          v[, c("gene", "mean_delta"), drop = FALSE],
          by = "gene", suffixes = c("_main", "_validation")
        )
        if (!nrow(z)) return(NULL)
        rho <- safe_cor(z$mean_delta_main, z$mean_delta_validation)
        data.frame(
          celltype = k$celltype,
          condition = k$condition,
          anchor_gene = k$anchor_gene,
          tf_gene = k$tf_gene,
          perturbation = k$perturbation,
          n_common_genes = nrow(z),
          signed_delta_spearman = rho,
          signed_delta_spearman_evidence = positive_correlation_evidence(rho),
          sign_concordance = safe_sign_concordance(z$mean_delta_main, z$mean_delta_validation),
          top100_jaccard = jaccard(
            top_genes(m, "mean_delta"), top_genes(v, "mean_delta")
          ),
          stringsAsFactors = FALSE
        )
      })
      celloracle_gene_metrics <- safe_rbind(rows)
      add_diag("celloracle_gene_metrics", if (nrow(celloracle_gene_metrics)) "ok" else "zero_matches", nrow(celloracle_gene_metrics), "paired main-validation rows")
    } else {
      add_diag("celloracle_gene_metrics", "missing_columns", nrow(co_all), paste(missing_cols, collapse = ", "))
      warning("CellOracle gene-delta files missing columns: ", paste(missing_cols, collapse = ", "))
    }
  }
}

co_summary <- data.frame()
celloracle_state_metrics <- data.frame()
if (length(summary_files)) {
  co_summary <- safe_rbind(lapply(summary_files, function(f) {
    tryCatch(
      read.csv(f, stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) {
        warning("Could not read ", f, ": ", conditionMessage(e))
        NULL
      }
    )
  }))
  if (nrow(co_summary)) {
    if (!"tf_gene" %in% colnames(co_summary) && "target_gene" %in% colnames(co_summary)) {
      co_summary$tf_gene <- co_summary$target_gene
    }
    if (!"anchor_gene" %in% colnames(co_summary)) co_summary$anchor_gene <- co_summary$tf_gene
    required <- c(
      "dataset", "celltype", "condition", "anchor_gene", "tf_gene",
      "perturbation", "axis_shift_score", "anchor_mean_delta"
    )
    missing_cols <- setdiff(required, colnames(co_summary))
    if (!length(missing_cols)) {
      keys <- unique(co_summary[, c(
        "celltype", "condition", "anchor_gene", "tf_gene", "perturbation"
      ), drop = FALSE])
      rows <- lapply(seq_len(nrow(keys)), function(i) {
        k <- keys[i, , drop = FALSE]
        m <- subset(
          co_summary,
          dataset == "main" & celltype == k$celltype & condition == k$condition &
            anchor_gene == k$anchor_gene & tf_gene == k$tf_gene &
            perturbation == k$perturbation
        )
        v <- subset(
          co_summary,
          dataset == "validation" & celltype == k$celltype & condition == k$condition &
            anchor_gene == k$anchor_gene & tf_gene == k$tf_gene &
            perturbation == k$perturbation
        )
        if (!nrow(m) || !nrow(v)) return(NULL)
        axis_main <- mean_finite(m$axis_shift_score)
        axis_validation <- mean_finite(v$axis_shift_score)
        anchor_main <- mean_finite(m$anchor_mean_delta)
        anchor_validation <- mean_finite(v$anchor_mean_delta)
        data.frame(
          celltype = k$celltype,
          condition = k$condition,
          anchor_gene = k$anchor_gene,
          tf_gene = k$tf_gene,
          perturbation = k$perturbation,
          axis_shift_main = axis_main,
          axis_shift_validation = axis_validation,
          axis_direction_concordance = safe_direction_concordance(axis_main, axis_validation),
          axis_magnitude_similarity = safe_magnitude_similarity(axis_main, axis_validation),
          anchor_delta_main = anchor_main,
          anchor_delta_validation = anchor_validation,
          anchor_direction_concordance = safe_direction_concordance(anchor_main, anchor_validation),
          anchor_magnitude_similarity = safe_magnitude_similarity(anchor_main, anchor_validation),
          stringsAsFactors = FALSE
        )
      })
      celloracle_state_metrics <- safe_rbind(rows)
      add_diag("celloracle_state_metrics", if (nrow(celloracle_state_metrics)) "ok" else "zero_matches", nrow(celloracle_state_metrics), "paired main-validation rows")
    } else {
      add_diag("celloracle_state_metrics", "missing_columns", nrow(co_summary), paste(missing_cols, collapse = ", "))
      warning("CellOracle summary files missing columns: ", paste(missing_cols, collapse = ", "))
    }
  }
}

# -----------------------------------------------------------------------------
# Completeness audit. Final ranking is blocked by default when any expected
# main-validation pair is missing. Set VP_ALLOW_PARTIAL_VALIDATION=true only
# for explicitly exploratory output.
# -----------------------------------------------------------------------------
if (file.exists(TARGET_MANIFEST_PATH)) {
  expected_anchor <- read.csv(TARGET_MANIFEST_PATH, stringsAsFactors = FALSE, check.names = FALSE)
  if (all(c("celltype", "gene") %in% colnames(expected_anchor))) {
    expected_network <- merge(
      expected_anchor[, c("celltype", "gene"), drop = FALSE],
      expand.grid(disease = TARGET_DISEASES, perturbation = c("KO", "OE_GAIN"), stringsAsFactors = FALSE)
    )
    names(expected_network)[names(expected_network) == "gene"] <- "anchor_gene"
    if (exists("net") && is.data.frame(net) && nrow(net) &&
        all(c("dataset", "celltype", "disease", "target_gene", "perturbation") %in% colnames(net))) {
      main_keys <- unique(paste(net$celltype[net$dataset == "main"], net$disease[net$dataset == "main"], net$target_gene[net$dataset == "main"], net$perturbation[net$dataset == "main"], sep = "||"))
      validation_keys <- unique(paste(net$celltype[net$dataset == "validation"], net$disease[net$dataset == "validation"], net$target_gene[net$dataset == "validation"], net$perturbation[net$dataset == "validation"], sep = "||"))
      keys <- paste(expected_network$celltype, expected_network$disease, expected_network$anchor_gene, expected_network$perturbation, sep = "||")
      expected_network$main_present <- keys %in% main_keys
      expected_network$validation_present <- keys %in% validation_keys
    } else {
      expected_network$main_present <- FALSE
      expected_network$validation_present <- FALSE
    }
    expected_network$paired_complete <- expected_network$main_present & expected_network$validation_present
    expected_network$stage <- "network"
    coverage_audit[["network"]] <- expected_network
    if (any(!expected_network$paired_complete)) {
      add_blocker(paste0("network missing ", sum(!expected_network$paired_complete), "/", nrow(expected_network), " expected main-validation pairs"))
    }
  }
} else {
  add_blocker("target_manifest.csv is missing")
}

if (file.exists(TF_TARGET_MANIFEST_PATH)) {
  expected_tf <- read.csv(TF_TARGET_MANIFEST_PATH, stringsAsFactors = FALSE, check.names = FALSE)
  if (all(c("celltype", "anchor_gene", "tf_gene") %in% colnames(expected_tf))) {
    expected_tf <- unique(expected_tf[, c("celltype", "anchor_gene", "tf_gene"), drop = FALSE])
    expected_celloracle <- merge(
      expected_tf,
      expand.grid(condition = TARGET_DISEASES, perturbation = c("KO", "OE"), stringsAsFactors = FALSE)
    )
    make_key <- function(d) paste(d$celltype, d$condition, d$anchor_gene, d$tf_gene, d$perturbation, sep = "||")
    make_dataset_pair_key <- function(dataset, d) {
      paste(dataset, d$celltype, d$anchor_gene, d$tf_gene, sep = "||")
    }
    expected_keys <- make_key(expected_celloracle)

    # Read per-job eligibility exclusions. Only low expression is accepted as a
    # formal "non-evaluable" state. TF-space, missing-gene, or missing-anchor
    # failures remain hard blockers.
    skipped_all <- safe_rbind(lapply(celloracle_dirs, function(d) {
      f <- file.path(d, "skipped_targets.csv")
      if (!file.exists(f)) return(NULL)
      x <- tryCatch(
        read.csv(f, stringsAsFactors = FALSE, check.names = FALSE),
        error = function(e) {
          warning("Could not read ", f, ": ", conditionMessage(e))
          NULL
        }
      )
      if (is.null(x) || !nrow(x)) return(NULL)
      if (!"tf_gene" %in% colnames(x) && "gene" %in% colnames(x)) x$tf_gene <- x$gene
      if (!all(c("anchor_gene", "tf_gene") %in% colnames(x))) return(NULL)
      if (!"reason" %in% colnames(x)) x$reason <- "unknown_ineligible_reason"
      job <- basename(d)
      split_at <- regexpr("__", job, fixed = TRUE)[1]
      if (split_at < 1L) return(NULL)
      dataset <- substr(job, 1L, split_at - 1L)
      celltype <- substr(job, split_at + 2L, nchar(job))
      data.frame(
        dataset = dataset,
        celltype = celltype,
        anchor_gene = trimws(as.character(x$anchor_gene)),
        tf_gene = trimws(as.character(x$tf_gene)),
        reason = trimws(as.character(x$reason)),
        source_file = f,
        stringsAsFactors = FALSE
      )
    }))

    expected_dataset_pair_keys <- c(
      make_dataset_pair_key("main", expected_tf),
      make_dataset_pair_key("validation", expected_tf)
    )
    if (nrow(skipped_all)) {
      skipped_all$dataset_pair_key <- paste(
        skipped_all$dataset, skipped_all$celltype,
        skipped_all$anchor_gene, skipped_all$tf_gene, sep = "||"
      )
      skipped_all <- skipped_all[
        skipped_all$dataset_pair_key %in% expected_dataset_pair_keys,
        , drop = FALSE
      ]
    }

    non_evaluable_pairs <- if (nrow(skipped_all)) {
      unique(skipped_all[
        skipped_all$reason == "TF_below_expression_threshold",
        c("dataset", "celltype", "anchor_gene", "tf_gene", "reason", "source_file"),
        drop = FALSE
      ])
    } else {
      data.frame()
    }
    hard_ineligible_pairs <- if (nrow(skipped_all)) {
      unique(skipped_all[
        skipped_all$reason != "TF_below_expression_threshold",
        c("dataset", "celltype", "anchor_gene", "tf_gene", "reason", "source_file"),
        drop = FALSE
      ])
    } else {
      data.frame()
    }

    if (nrow(non_evaluable_pairs)) {
      write.csv(
        non_evaluable_pairs,
        file.path(OUT, "celloracle_non_evaluable_pairs.csv"),
        row.names = FALSE
      )
    }
    if (nrow(hard_ineligible_pairs)) {
      write.csv(
        hard_ineligible_pairs,
        file.path(OUT, "celloracle_hard_ineligible_pairs.csv"),
        row.names = FALSE
      )
      add_blocker(paste0(
        "CellOracle has ", nrow(hard_ineligible_pairs),
        " hard-ineligible dataset-specific anchor-TF pairs; see celloracle_hard_ineligible_pairs.csv"
      ))
    }

    non_eval_key <- function(dataset) {
      if (!nrow(non_evaluable_pairs)) return(character())
      z <- non_evaluable_pairs[non_evaluable_pairs$dataset == dataset, , drop = FALSE]
      if (!nrow(z)) return(character())
      paste(z$dataset, z$celltype, z$anchor_gene, z$tf_gene, sep = "||")
    }
    main_non_eval_keys <- non_eval_key("main")
    validation_non_eval_keys <- non_eval_key("validation")
    main_pair_keys <- make_dataset_pair_key("main", expected_celloracle)
    validation_pair_keys <- make_dataset_pair_key("validation", expected_celloracle)
    expected_celloracle$main_non_evaluable <- main_pair_keys %in% main_non_eval_keys
    expected_celloracle$validation_non_evaluable <- validation_pair_keys %in% validation_non_eval_keys
    expected_celloracle$evaluable <-
      !expected_celloracle$main_non_evaluable &
      !expected_celloracle$validation_non_evaluable

    reason_lookup <- function(keys, dataset) {
      if (!nrow(non_evaluable_pairs)) return(rep("", length(keys)))
      z <- non_evaluable_pairs[non_evaluable_pairs$dataset == dataset, , drop = FALSE]
      if (!nrow(z)) return(rep("", length(keys)))
      zkeys <- paste(z$dataset, z$celltype, z$anchor_gene, z$tf_gene, sep = "||")
      out <- z$reason[match(keys, zkeys)]
      out[is.na(out)] <- ""
      out
    }
    expected_celloracle$main_non_evaluable_reason <-
      reason_lookup(main_pair_keys, "main")
    expected_celloracle$validation_non_evaluable_reason <-
      reason_lookup(validation_pair_keys, "validation")

    if (nrow(co_all) && all(c("dataset", "celltype", "condition", "anchor_gene", "tf_gene", "perturbation") %in% colnames(co_all))) {
      gene_main <- unique(make_key(co_all[co_all$dataset == "main", , drop = FALSE]))
      gene_validation <- unique(make_key(co_all[co_all$dataset == "validation", , drop = FALSE]))
    } else {
      gene_main <- gene_validation <- character()
    }
    if (nrow(co_summary) && all(c("dataset", "celltype", "condition", "anchor_gene", "tf_gene", "perturbation") %in% colnames(co_summary))) {
      summary_main <- unique(make_key(co_summary[co_summary$dataset == "main", , drop = FALSE]))
      summary_validation <- unique(make_key(co_summary[co_summary$dataset == "validation", , drop = FALSE]))
    } else {
      summary_main <- summary_validation <- character()
    }

    expected_celloracle$main_gene_delta_present <- expected_keys %in% gene_main
    expected_celloracle$validation_gene_delta_present <- expected_keys %in% gene_validation
    expected_celloracle$main_summary_present <- expected_keys %in% summary_main
    expected_celloracle$validation_summary_present <- expected_keys %in% summary_validation
    expected_celloracle$missing_main_gene_delta <- !expected_celloracle$main_gene_delta_present
    expected_celloracle$missing_validation_gene_delta <- !expected_celloracle$validation_gene_delta_present
    expected_celloracle$missing_main_summary <- !expected_celloracle$main_summary_present
    expected_celloracle$missing_validation_summary <- !expected_celloracle$validation_summary_present
    expected_celloracle$missing_component_count <- rowSums(expected_celloracle[, c(
      "missing_main_gene_delta", "missing_validation_gene_delta",
      "missing_main_summary", "missing_validation_summary"
    ), drop = FALSE])
    expected_celloracle$missing_components <- apply(
      expected_celloracle[, c(
        "missing_main_gene_delta", "missing_validation_gene_delta",
        "missing_main_summary", "missing_validation_summary"
      ), drop = FALSE],
      1,
      function(x) {
        labels <- c(
          "main_gene_delta", "validation_gene_delta",
          "main_summary", "validation_summary"
        )
        paste(labels[as.logical(x)], collapse = ";")
      }
    )
    expected_celloracle$paired_complete <- expected_celloracle$missing_component_count == 0L
    expected_celloracle$formal_requirement_satisfied <-
      !expected_celloracle$evaluable | expected_celloracle$paired_complete
    expected_celloracle$validation_status <- ifelse(
      expected_celloracle$evaluable,
      ifelse(expected_celloracle$paired_complete, "complete", "missing_evaluable_output"),
      ifelse(
        expected_celloracle$main_non_evaluable & expected_celloracle$validation_non_evaluable,
        "non_evaluable_both_datasets",
        ifelse(
          expected_celloracle$main_non_evaluable,
          "non_evaluable_main",
          "non_evaluable_validation"
        )
      )
    )
    expected_celloracle$stage <- "celloracle"
    coverage_audit[["celloracle"]] <- expected_celloracle

    non_evaluable_tests <- expected_celloracle[!expected_celloracle$evaluable, , drop = FALSE]
    write.csv(
      non_evaluable_tests,
      file.path(OUT, "celloracle_non_evaluable_tests.csv"),
      row.names = FALSE
    )

    missing_tests <- expected_celloracle[
      expected_celloracle$evaluable & !expected_celloracle$paired_complete,
      , drop = FALSE
    ]
    write.csv(
      missing_tests,
      file.path(OUT, "celloracle_missing_tests.csv"),
      row.names = FALSE
    )

    missing_jobs <- data.frame(
      job_key = character(), dataset = character(), celltype = character(),
      condition = character(), anchor_gene = character(), tf_gene = character(),
      perturbation = character(), need_gene_delta = logical(),
      need_summary = logical(), missing_outputs = character(),
      stringsAsFactors = FALSE
    )
    if (nrow(missing_tests)) {
      # One row per dataset-specific evaluable job. Non-evaluable rows are never
      # sent back to the runner as missing work.
      missing_jobs <- safe_rbind(lapply(seq_len(nrow(missing_tests)), function(i) {
        z <- missing_tests[i, , drop = FALSE]
        rows <- list()
        if (isTRUE(z$missing_main_gene_delta) || isTRUE(z$missing_main_summary)) {
          rows[[length(rows) + 1L]] <- data.frame(
            dataset = "main",
            celltype = z$celltype, condition = z$condition,
            anchor_gene = z$anchor_gene, tf_gene = z$tf_gene,
            perturbation = z$perturbation,
            need_gene_delta = isTRUE(z$missing_main_gene_delta),
            need_summary = isTRUE(z$missing_main_summary),
            missing_outputs = paste(
              c(
                if (isTRUE(z$missing_main_gene_delta)) "celloracle_gene_deltas.csv.gz",
                if (isTRUE(z$missing_main_summary)) "celloracle_perturbation_summary.csv"
              ),
              collapse = ";"
            ),
            stringsAsFactors = FALSE
          )
        }
        if (isTRUE(z$missing_validation_gene_delta) || isTRUE(z$missing_validation_summary)) {
          rows[[length(rows) + 1L]] <- data.frame(
            dataset = "validation",
            celltype = z$celltype, condition = z$condition,
            anchor_gene = z$anchor_gene, tf_gene = z$tf_gene,
            perturbation = z$perturbation,
            need_gene_delta = isTRUE(z$missing_validation_gene_delta),
            need_summary = isTRUE(z$missing_validation_summary),
            missing_outputs = paste(
              c(
                if (isTRUE(z$missing_validation_gene_delta)) "celloracle_gene_deltas.csv.gz",
                if (isTRUE(z$missing_validation_summary)) "celloracle_perturbation_summary.csv"
              ),
              collapse = ";"
            ),
            stringsAsFactors = FALSE
          )
        }
        safe_rbind(rows)
      }))
      if (nrow(missing_jobs)) {
        missing_jobs$job_key <- paste(
          missing_jobs$dataset, missing_jobs$celltype, missing_jobs$condition,
          missing_jobs$anchor_gene, missing_jobs$tf_gene,
          missing_jobs$perturbation, sep = "||"
        )
        missing_jobs <- missing_jobs[, c(
          "job_key", "dataset", "celltype", "condition", "anchor_gene",
          "tf_gene", "perturbation", "need_gene_delta", "need_summary",
          "missing_outputs"
        ), drop = FALSE]
      }
    }
    write.csv(
      missing_jobs,
      file.path(OUT, "celloracle_missing_jobs_long.csv"),
      row.names = FALSE
    )

    dataset_jobs <- if (nrow(missing_jobs)) {
      unique(missing_jobs[, c("dataset", "celltype"), drop = FALSE])
    } else {
      data.frame(dataset = character(), celltype = character(), stringsAsFactors = FALSE)
    }
    dataset_jobs$celloracle_job <- paste(dataset_jobs$dataset, dataset_jobs$celltype, sep = "__")
    dataset_jobs <- dataset_jobs[order(dataset_jobs$celloracle_job), , drop = FALSE]
    write.csv(
      dataset_jobs,
      file.path(OUT, "celloracle_missing_dataset_jobs.csv"),
      row.names = FALSE
    )

    job_summary <- if (nrow(missing_jobs)) {
      z <- aggregate(
        rep(1L, nrow(missing_jobs)),
        by = list(
          dataset = missing_jobs$dataset,
          need_gene_delta = missing_jobs$need_gene_delta,
          need_summary = missing_jobs$need_summary
        ),
        FUN = sum
      )
      names(z)[names(z) == "x"] <- "n_jobs"
      z
    } else {
      data.frame(
        dataset = character(), need_gene_delta = logical(),
        need_summary = logical(), n_jobs = integer(),
        stringsAsFactors = FALSE
      )
    }
    write.csv(
      job_summary,
      file.path(OUT, "celloracle_missing_job_summary.csv"),
      row.names = FALSE
    )

    n_canonical_tests <- nrow(expected_celloracle)
    n_non_evaluable_tests <- nrow(non_evaluable_tests)
    n_evaluable_tests <- sum(expected_celloracle$evaluable)
    n_complete_evaluable_tests <- sum(
      expected_celloracle$evaluable & expected_celloracle$paired_complete
    )
    n_missing_evaluable_tests <- nrow(missing_tests)
    formal_summary <- data.frame(
      n_canonical_tests = n_canonical_tests,
      n_non_evaluable_tests = n_non_evaluable_tests,
      n_evaluable_tests = n_evaluable_tests,
      n_complete_evaluable_tests = n_complete_evaluable_tests,
      n_missing_evaluable_tests = n_missing_evaluable_tests,
      n_non_evaluable_dataset_pairs = nrow(non_evaluable_pairs),
      n_hard_ineligible_dataset_pairs = nrow(hard_ineligible_pairs),
      formal_complete = (
        n_missing_evaluable_tests == 0L &&
        nrow(hard_ineligible_pairs) == 0L
      ),
      stringsAsFactors = FALSE
    )
    write.csv(
      formal_summary,
      file.path(OUT, "celloracle_formal_completeness_summary.csv"),
      row.names = FALSE
    )

    add_diag(
      "celloracle_evaluability",
      if (n_non_evaluable_tests) "complete_with_non_evaluable" else "all_evaluable",
      n_evaluable_tests,
      paste0(
        "canonical=", n_canonical_tests,
        "; evaluable=", n_evaluable_tests,
        "; non_evaluable=", n_non_evaluable_tests,
        "; hard_ineligible_dataset_pairs=", nrow(hard_ineligible_pairs)
      )
    )

    if (n_missing_evaluable_tests) {
      add_diag(
        "celloracle_missing_jobs", "incomplete", n_missing_evaluable_tests,
        paste0(
          "missing evaluable paired tests=", n_missing_evaluable_tests,
          "; dataset-specific jobs=", nrow(missing_jobs),
          "; see celloracle_missing_jobs_long.csv"
        )
      )
      add_blocker(paste0(
        "CellOracle missing ", n_missing_evaluable_tests, "/",
        n_evaluable_tests, " evaluable main-validation TF tests",
        " (", n_non_evaluable_tests, " additional canonical tests are non-evaluable)"
      ))
    } else {
      add_diag(
        "celloracle_missing_jobs", "complete", n_complete_evaluable_tests,
        paste0(
          "complete evaluable tests=", n_complete_evaluable_tests, "/",
          n_evaluable_tests,
          "; non-evaluable canonical tests=", n_non_evaluable_tests
        )
      )
    }
  } else {
    add_blocker("celloracle_tf_target_manifest.csv lacks required columns")
  }
} else {
  add_blocker("celloracle_tf_target_manifest.csv is missing")
}

if (length(coverage_audit)) {
  # Write separate files because network and CellOracle have different columns.
  if (!is.null(coverage_audit$network)) write.csv(coverage_audit$network, file.path(OUT, "network_validation_coverage.csv"), row.names = FALSE)
  if (!is.null(coverage_audit$celloracle)) write.csv(coverage_audit$celloracle, file.path(OUT, "celloracle_validation_coverage.csv"), row.names = FALSE)
}
if (length(validation_blockers)) {
  add_diag("completeness_gate", if (ALLOW_PARTIAL_VALIDATION) "partial_allowed" else "blocked", length(validation_blockers), paste(validation_blockers, collapse = "; "))
  if (!ALLOW_PARTIAL_VALIDATION) {
    write.csv(diagnostics, file.path(OUT, "cross_dataset_validation_diagnostics.csv"), row.names = FALSE)
    stop(
      "Cross-dataset validation is incomplete: ", paste(validation_blockers, collapse = "; "),
      ". Complete the missing jobs first. Coverage and rerun manifests were written to ", OUT,
      ". Start with celloracle_missing_jobs_long.csv. ",
      "Set VP_ALLOW_PARTIAL_VALIDATION=true only for exploratory summaries."
    )
  }
  warning("Partial validation explicitly allowed: ", paste(validation_blockers, collapse = "; "))
} else {
  add_diag("completeness_gate", "complete", 0L, "All evaluable main-validation pairs are present; non-evaluable tests are documented separately")
}

celloracle_metrics <- data.frame()
merge_keys <- c("celltype", "condition", "anchor_gene", "tf_gene", "perturbation")
if (nrow(celloracle_gene_metrics) && nrow(celloracle_state_metrics)) {
  celloracle_metrics <- merge(
    celloracle_gene_metrics, celloracle_state_metrics,
    by = merge_keys, all = TRUE
  )
} else if (nrow(celloracle_gene_metrics)) {
  celloracle_metrics <- celloracle_gene_metrics
} else if (nrow(celloracle_state_metrics)) {
  celloracle_metrics <- celloracle_state_metrics
}

celloracle_anchor_summary <- data.frame()
if (nrow(celloracle_metrics)) {
  # Attach the frozen MAIN-derived biological interpretation. Validation is used
  # only for replication, never for TF selection.
  tf_info <- data.frame()
  if (file.exists(TF_TARGET_MANIFEST_PATH)) {
    tf_info <- read.csv(TF_TARGET_MANIFEST_PATH, stringsAsFactors = FALSE, check.names = FALSE)
    keep <- intersect(
      c("celltype", "anchor_gene", "tf_gene", "biological_role", "axis_direction",
        "ko_condition", "oe_condition", "primary_test", "selection_mode",
        "selection_dataset", "validation_used_for_selection"),
      colnames(tf_info)
    )
    tf_info <- unique(tf_info[, keep, drop = FALSE])
    if (all(c("celltype", "anchor_gene", "tf_gene") %in% colnames(tf_info))) {
      celloracle_metrics <- merge(
        celloracle_metrics, tf_info,
        by = c("celltype", "anchor_gene", "tf_gene"), all.x = TRUE
      )
    }
  }

  if (!"biological_role" %in% colnames(celloracle_metrics)) celloracle_metrics$biological_role <- "disease_increasing"
  if (!"axis_direction" %in% colnames(celloracle_metrics)) celloracle_metrics$axis_direction <- NA_real_
  for (nm in c("axis_shift_main", "axis_shift_validation", "anchor_delta_main", "anchor_delta_validation")) {
    if (!nm %in% colnames(celloracle_metrics)) celloracle_metrics[[nm]] <- NA_real_
  }
  role_text <- tolower(as.character(celloracle_metrics$biological_role))
  axis_dir_num <- suppressWarnings(as.numeric(celloracle_metrics$axis_direction))
  disease_role_sign <- ifelse(
    grepl("decreas|protect|healthy|negative", role_text) | (is.finite(axis_dir_num) & axis_dir_num < 0),
    -1, 1
  )
  celloracle_metrics$disease_role_sign <- disease_role_sign
  celloracle_metrics$test_class <- ifelse(
    celloracle_metrics$condition == "IPF" & celloracle_metrics$perturbation == "KO",
    "primary_reversal",
    ifelse(
      celloracle_metrics$condition == "Control" & celloracle_metrics$perturbation == "OE",
      "disease_induction",
      "supportive_specificity"
    )
  )
  celloracle_metrics$expected_axis_sign <- ifelse(
    celloracle_metrics$test_class == "primary_reversal", -disease_role_sign,
    ifelse(celloracle_metrics$test_class == "disease_induction", disease_role_sign, NA_real_)
  )
  celloracle_metrics$expected_anchor_sign <- celloracle_metrics$expected_axis_sign
  celloracle_metrics$axis_expected_direction_main <- mapply(
    sign_match_score, celloracle_metrics$axis_shift_main, celloracle_metrics$expected_axis_sign
  )
  celloracle_metrics$axis_expected_direction_validation <- mapply(
    sign_match_score, celloracle_metrics$axis_shift_validation, celloracle_metrics$expected_axis_sign
  )
  celloracle_metrics$anchor_expected_direction_main <- mapply(
    sign_match_score, celloracle_metrics$anchor_delta_main, celloracle_metrics$expected_anchor_sign
  )
  celloracle_metrics$anchor_expected_direction_validation <- mapply(
    sign_match_score, celloracle_metrics$anchor_delta_validation, celloracle_metrics$expected_anchor_sign
  )
  celloracle_metrics$expected_direction_score_main <- mapply(
    function(a, b) mean_finite(c(a, b)),
    celloracle_metrics$axis_expected_direction_main,
    celloracle_metrics$anchor_expected_direction_main
  )
  celloracle_metrics$expected_direction_score_validation <- mapply(
    function(a, b) mean_finite(c(a, b)),
    celloracle_metrics$axis_expected_direction_validation,
    celloracle_metrics$anchor_expected_direction_validation
  )
  celloracle_metrics$expected_direction_reproducibility <- mapply(
    function(a, b) mean_finite(c(a, b)),
    celloracle_metrics$expected_direction_score_main,
    celloracle_metrics$expected_direction_score_validation
  )
  celloracle_metrics$direction_pass_main <- is.finite(celloracle_metrics$expected_direction_score_main) & celloracle_metrics$expected_direction_score_main >= 1
  celloracle_metrics$direction_pass_validation <- is.finite(celloracle_metrics$expected_direction_score_validation) & celloracle_metrics$expected_direction_score_validation >= 1
  celloracle_metrics$direction_pass_both <- celloracle_metrics$direction_pass_main & celloracle_metrics$direction_pass_validation
  celloracle_metrics$test_class_weight <- ifelse(
    celloracle_metrics$test_class == "primary_reversal", 0.55,
    ifelse(celloracle_metrics$test_class == "disease_induction", 0.30, 0.15)
  )
  metric_cols <- c(
    "signed_delta_spearman_evidence", "sign_concordance", "top100_jaccard",
    "axis_direction_concordance", "axis_magnitude_similarity",
    "anchor_direction_concordance", "anchor_magnitude_similarity",
    "expected_direction_reproducibility"
  )
  for (nm in setdiff(metric_cols, colnames(celloracle_metrics))) {
    celloracle_metrics[[nm]] <- NA_real_
  }
  celloracle_metrics$tf_evidence_metric_count <- rowSums(
    is.finite(as.matrix(celloracle_metrics[, metric_cols, drop = FALSE]))
  )
  celloracle_metrics$tf_evidence_metric_coverage <-
    celloracle_metrics$tf_evidence_metric_count / length(metric_cols)
  celloracle_metrics$tf_cross_dataset_evidence_score_raw <- apply(
    celloracle_metrics[, setdiff(metric_cols, "expected_direction_reproducibility"), drop = FALSE],
    1, mean_finite
  )
  celloracle_metrics$tf_cross_dataset_evidence_score <- mapply(
    function(base, direction, cls) {
      if (!is.finite(base)) return(NA_real_)
      if (cls %in% c("primary_reversal", "disease_induction") && is.finite(direction)) {
        return(0.75 * base + 0.25 * direction)
      }
      base
    },
    celloracle_metrics$tf_cross_dataset_evidence_score_raw,
    celloracle_metrics$expected_direction_reproducibility,
    celloracle_metrics$test_class
  )
  write.csv(
    celloracle_metrics,
    file.path(OUT, "celloracle_tf_cross_dataset_metrics.csv"),
    row.names = FALSE
  )
  add_diag("celloracle_combined_metrics", "ok", nrow(celloracle_metrics), paste(metric_cols, collapse = ", "))

  group_cols <- c("celltype", "anchor_gene", "tf_gene")
  tf_keys <- unique(celloracle_metrics[, group_cols, drop = FALSE])
  tf_rows <- lapply(seq_len(nrow(tf_keys)), function(i) {
    k <- tf_keys[i, , drop = FALSE]
    z <- subset(
      celloracle_metrics,
      celltype == k$celltype & anchor_gene == k$anchor_gene & tf_gene == k$tf_gene
    )
    cls_score <- function(cls) {
      zz <- z[z$test_class == cls, , drop = FALSE]
      if (!nrow(zz)) return(NA_real_)
      mean_finite(zz$tf_cross_dataset_evidence_score)
    }
    cls_dir <- function(cls) {
      zz <- z[z$test_class == cls, , drop = FALSE]
      if (!nrow(zz)) return(NA_real_)
      mean_finite(zz$expected_direction_reproducibility)
    }
    data.frame(
      celltype = k$celltype, anchor_gene = k$anchor_gene, tf_gene = k$tf_gene,
      tf_cross_dataset_evidence_score = weighted_mean_finite(
        z$tf_cross_dataset_evidence_score, z$test_class_weight
      ),
      tf_evidence_metric_coverage = mean_finite(z$tf_evidence_metric_coverage),
      n_condition_perturbation_tests = sum(is.finite(z$tf_cross_dataset_evidence_score)),
      primary_reversal_score = cls_score("primary_reversal"),
      disease_induction_score = cls_score("disease_induction"),
      supportive_test_score = cls_score("supportive_specificity"),
      primary_direction_score = cls_dir("primary_reversal"),
      induction_direction_score = cls_dir("disease_induction"),
      primary_direction_pass_main = any(z$test_class == "primary_reversal" & z$direction_pass_main, na.rm = TRUE),
      primary_direction_pass_validation = any(z$test_class == "primary_reversal" & z$direction_pass_validation, na.rm = TRUE),
      primary_direction_pass_both = any(z$test_class == "primary_reversal" & z$direction_pass_both, na.rm = TRUE),
      primary_metric_coverage = mean_finite(z$tf_evidence_metric_coverage[z$test_class == "primary_reversal"]),
      stringsAsFactors = FALSE
    )
  })
  tf_summary <- safe_rbind(tf_rows)
  tf_summary <- tf_summary[order(
    tf_summary$celltype, tf_summary$anchor_gene,
    -tf_summary$primary_reversal_score,
    -tf_summary$tf_cross_dataset_evidence_score
  ), , drop = FALSE]
  write.csv(
    tf_summary,
    file.path(OUT, "celloracle_tf_ranking_within_anchor.csv"),
    row.names = FALSE
  )
  write.csv(
    tf_summary[, intersect(c(
      "celltype", "anchor_gene", "tf_gene",
      "primary_reversal_score", "disease_induction_score", "supportive_test_score",
      "primary_direction_score", "induction_direction_score",
      "primary_direction_pass_main", "primary_direction_pass_validation",
      "primary_direction_pass_both", "primary_metric_coverage"
    ), colnames(tf_summary)), drop = FALSE],
    file.path(OUT, "celloracle_primary_direction_summary.csv"),
    row.names = FALSE
  )

  key <- unique(tf_summary[, c("celltype", "anchor_gene"), drop = FALSE])
  rows <- lapply(seq_len(nrow(key)), function(i) {
    k <- key[i, , drop = FALSE]
    z <- subset(
      tf_summary,
      celltype == k$celltype & anchor_gene == k$anchor_gene &
        is.finite(tf_cross_dataset_evidence_score)
    )
    if (!nrow(z)) return(NULL)
    z <- z[order(z$tf_cross_dataset_evidence_score, decreasing = TRUE), , drop = FALSE]
    data.frame(
      celltype = k$celltype,
      anchor_gene = k$anchor_gene,
      n_celloracle_tfs_validated = nrow(z),
      best_celloracle_tf = z$tf_gene[1],
      best_celloracle_tf_score = z$tf_cross_dataset_evidence_score[1],
      best_primary_reversal_score = z$primary_reversal_score[1],
      best_primary_direction_pass_both = z$primary_direction_pass_both[1],
      mean_celloracle_tf_score = mean_finite(z$tf_cross_dataset_evidence_score),
      celloracle_anchor_evidence_score = mean_finite(
        head(z$tf_cross_dataset_evidence_score, min(2L, nrow(z)))
      ),
      mean_celloracle_metric_coverage = mean_finite(z$tf_evidence_metric_coverage),
      stringsAsFactors = FALSE
    )
  })
  celloracle_anchor_summary <- safe_rbind(rows)
  if (nrow(celloracle_anchor_summary)) {
    write.csv(
      celloracle_anchor_summary,
      file.path(OUT, "celloracle_anchor_evidence_summary.csv"),
      row.names = FALSE
    )
  }
}

# -----------------------------------------------------------------------------
# Final ranking remains anchored to the original PPI/dynamic candidate genes.
# CellOracle TF evidence is aggregated back to its anchor_gene.
# -----------------------------------------------------------------------------
manifest_path <- TARGET_MANIFEST_PATH
if (!file.exists(manifest_path)) {
  add_diag("target_manifest", "missing", 0L, manifest_path)
  write.csv(diagnostics, file.path(OUT, "cross_dataset_validation_diagnostics.csv"), row.names = FALSE)
  stop("Missing target_manifest.csv; run the target-manifest step first.")
}
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE, check.names = FALSE)
add_diag("target_manifest", "ok", nrow(manifest), manifest_path)
final <- manifest

if (nrow(network_metrics)) {
  net_keys <- unique(network_metrics[, c("celltype", "anchor_gene"), drop = FALSE])
  net_rows <- lapply(seq_len(nrow(net_keys)), function(i) {
    k <- net_keys[i, , drop = FALSE]
    z <- subset(
      network_metrics,
      celltype == k$celltype & anchor_gene == k$anchor_gene
    )
    data.frame(
      celltype = k$celltype,
      anchor_gene = k$anchor_gene,
      network_spearman_FC = mean_finite(z$spearman_FC),
      network_spearman_evidence = mean_finite(z$spearman_FC_evidence),
      network_sign_concordance = mean_finite(z$sign_concordance),
      network_top100_jaccard = mean_finite(z$top100_jaccard),
      network_significant_jaccard = mean_finite(z$significant_jaccard),
      network_metric_coverage = mean_finite(z$network_metric_coverage),
      network_evidence_score = mean_finite(z$network_evidence_score),
      stringsAsFactors = FALSE
    )
  })
  net_anchor <- safe_rbind(net_rows)
  final <- merge(
    final, net_anchor,
    by.x = c("celltype", "gene"),
    by.y = c("celltype", "anchor_gene"),
    all.x = TRUE
  )
}

if (nrow(celloracle_anchor_summary)) {
  final <- merge(
    final, celloracle_anchor_summary,
    by.x = c("celltype", "gene"),
    by.y = c("celltype", "anchor_gene"),
    all.x = TRUE
  )
}

if (!"network_evidence_score" %in% colnames(final)) final$network_evidence_score <- NA_real_
if (!"best_celloracle_tf_score" %in% colnames(final)) final$best_celloracle_tf_score <- NA_real_
if (!"celloracle_anchor_evidence_score" %in% colnames(final)) final$celloracle_anchor_evidence_score <- NA_real_

source_matrix <- as.matrix(
  final[, c("network_evidence_score", "celloracle_anchor_evidence_score"), drop = FALSE]
)
final$n_cross_dataset_evidence_sources <- rowSums(is.finite(source_matrix))
final$cross_dataset_evidence_coverage <- final$n_cross_dataset_evidence_sources / 2
final$cross_dataset_evidence_score_raw <- apply(source_matrix, 1, mean_finite)
# Coverage-adjusted score prevents a one-source result from looking equivalent to
# a result supported by both network and CellOracle evidence.
final$cross_dataset_evidence_score <-
  final$cross_dataset_evidence_score_raw * final$cross_dataset_evidence_coverage
final$cross_dataset_available_sources <- ifelse(
  is.finite(final$network_evidence_score) & is.finite(final$celloracle_anchor_evidence_score),
  "network+celloracle",
  ifelse(
    is.finite(final$network_evidence_score), "network_only",
    ifelse(is.finite(final$celloracle_anchor_evidence_score), "celloracle_only", "none")
  )
)
final$cross_dataset_support_status <- ifelse(
  final$n_cross_dataset_evidence_sources == 2L, "both_sources",
  ifelse(final$n_cross_dataset_evidence_sources == 1L, "single_source", "not_evaluated")
)

secondary_order <- if ("selection_score" %in% colnames(final)) {
  -suppressWarnings(as.numeric(final$selection_score))
} else {
  seq_len(nrow(final))
}
final <- final[order(
  -final$cross_dataset_evidence_score,
  secondary_order,
  na.last = TRUE
), , drop = FALSE]
write.csv(
  final,
  file.path(OUT, "final_cross_dataset_candidate_ranking.csv"),
  row.names = FALSE
)
add_diag(
  "final_ranking", "ok", nrow(final),
  paste0(
    "both_sources=", sum(final$n_cross_dataset_evidence_sources == 2L),
    "; single_source=", sum(final$n_cross_dataset_evidence_sources == 1L),
    "; not_evaluated=", sum(final$n_cross_dataset_evidence_sources == 0L)
  )
)
write.csv(
  diagnostics,
  file.path(OUT, "cross_dataset_validation_diagnostics.csv"),
  row.names = FALSE
)

message("Cross-dataset validation completed: ", OUT)
message("Review diagnostics first: ", file.path(OUT, "cross_dataset_validation_diagnostics.csv"))
