#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
})
source("config.R")

TARGET_MANIFEST <- file.path(OUTPUT_ROOT, "01_target_manifest", "target_manifest.csv")
TF_TARGET_MANIFEST <- file.path(OUTPUT_ROOT, "01_target_manifest", "celloracle_tf_target_manifest.csv")
TF_UNIVERSE <- file.path(SUPPORT_DIR, "CellOracle_human_promoter_TF_universe.csv")
OUT <- file.path(OUTPUT_ROOT, "03_celloracle_inputs")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(TARGET_MANIFEST)) stop("Run 02_build_target_manifest.R first.")
if (!file.exists(TF_UNIVERSE) && !ALLOW_MISSING_TF_UNIVERSE) {
  stop(
    "Missing CellOracle TF universe: ", TF_UNIVERSE,
    ". Run python 04a_prepare_celloracle_tf_targets.py --tf-universe-only before exporting inputs."
  )
}

targets <- read.csv(TARGET_MANIFEST, stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("celltype", "gene") %in% colnames(targets))) stop("target_manifest.csv must contain celltype and gene columns.")
if (!"anchor_gene" %in% colnames(targets)) targets$anchor_gene <- targets$gene

tf_targets <- if (file.exists(TF_TARGET_MANIFEST)) {
  read.csv(TF_TARGET_MANIFEST, stringsAsFactors = FALSE, check.names = FALSE)
} else data.frame()
if (nrow(tf_targets) && !all(c("celltype", "anchor_gene", "tf_gene") %in% colnames(tf_targets))) {
  stop("celloracle_tf_target_manifest.csv must contain celltype, anchor_gene and tf_gene.")
}
allow_validation_tf_selection <- tolower(Sys.getenv("VP_ALLOW_VALIDATION_TF_SELECTION", "false")) %in% c("1", "true", "yes", "y")
if (nrow(tf_targets) && "validation_used_for_selection" %in% colnames(tf_targets)) {
  leaked <- as.logical(tf_targets$validation_used_for_selection)
  leaked[is.na(leaked)] <- FALSE
  if (any(leaked) && !allow_validation_tf_selection) {
    stop(
      "TF manifest indicates validation data were used for TF selection. ",
      "Re-run 04a_prepare_celloracle_tf_targets.py with --selection-mode main_discovery_only. ",
      "Set VP_ALLOW_VALIDATION_TF_SELECTION=true only to reproduce a legacy exploratory run."
    )
  }
}

tf_universe <- if (file.exists(TF_UNIVERSE)) {
  z <- read.csv(TF_UNIVERSE, stringsAsFactors = FALSE, check.names = FALSE)
  if ("gene" %in% colnames(z)) unique(trimws(as.character(z$gene))) else character()
} else character()
tf_universe <- tf_universe[nzchar(tf_universe)]

dynamic_path <- file.path(SUPPORT_DIR, "00_combined_dynamic_genes_all_celltypes.csv")
dynamic <- if (file.exists(dynamic_path)) read.csv(dynamic_path, stringsAsFactors = FALSE, check.names = FALSE) else data.frame()

bundle_path <- function(dataset, celltype, disease) {
  root <- if (dataset == "main") MAIN_DIR else VALIDATION_DIR
  file.path(root, paste0(dataset, "__", celltype, "__", disease, ".rds"))
}

balanced_cells <- function(meta, n_max) {
  sample_col <- if ("sample_for_perturbation" %in% colnames(meta)) {
    "sample_for_perturbation"
  } else if ("sample_id" %in% colnames(meta)) {
    "sample_id"
  } else NA_character_
  cells <- rownames(meta)
  if (length(cells) <= n_max) return(cells)
  if (is.na(sample_col)) return(sample(cells, n_max))
  parts <- split(cells, as.character(meta[[sample_col]]))
  quota <- max(1L, floor(n_max / length(parts)))
  chosen <- unlist(lapply(parts, function(v) if (length(v) <= quota) v else sample(v, quota)), use.names = FALSE)
  if (length(chosen) < n_max) {
    rest <- setdiff(cells, chosen)
    chosen <- c(chosen, sample(rest, min(length(rest), n_max - length(chosen))))
  }
  chosen[seq_len(min(length(chosen), n_max))]
}

select_shared_genes <- function(counts, mandatory_genes, preferred_genes, n_limit) {
  expressed <- Matrix::rowSums(counts > 0) >= MIN_GENE_EXPRESSED_CELLS
  counts <- counts[expressed, , drop = FALSE]
  if (!nrow(counts)) stop("No genes passed MIN_GENE_EXPRESSED_CELLS.")
  lib <- Matrix::colSums(counts)
  lib[lib <= 0] <- 1
  norm <- counts %*% Diagonal(x = 1e4 / lib)
  norm@x <- log1p(norm@x)
  mu <- Matrix::rowMeans(norm)
  second <- Matrix::rowMeans(norm^2)
  variance <- pmax(0, second - mu^2)
  names(variance) <- rownames(norm)

  mandatory <- intersect(unique(mandatory_genes[nzchar(mandatory_genes)]), rownames(norm))
  preferred <- setdiff(intersect(unique(preferred_genes[nzchar(preferred_genes)]), rownames(norm)), mandatory)
  remaining <- setdiff(names(sort(variance, decreasing = TRUE)), c(mandatory, preferred))
  # Mandatory genes are never truncated. The configured limit applies to the
  # total only when it is large enough to contain all mandatory genes.
  effective_limit <- max(as.integer(n_limit), length(mandatory))
  genes <- unique(c(mandatory, preferred, remaining))
  genes <- head(genes, min(effective_limit, length(genes)))
  list(
    genes = genes,
    n_mandatory = length(mandatory),
    n_preferred_retained = sum(preferred %in% genes),
    n_expressed_pool = nrow(norm)
  )
}

safe_dynamic_weights <- function(dynamic, ct) {
  required <- c("celltype", "gene", "spearman_axis")
  if (!nrow(dynamic) || !all(required %in% colnames(dynamic))) return(data.frame())
  optional <- intersect(c("trend_class", "dynamic_module"), colnames(dynamic))
  cols <- unique(c(required, optional))
  w <- dynamic[dynamic$celltype == ct, cols, drop = FALSE]
  if (!"trend_class" %in% colnames(w)) w$trend_class <- NA_character_
  if (!"dynamic_module" %in% colnames(w)) w$dynamic_module <- NA_character_
  w <- w[, c("gene", "spearman_axis", "trend_class", "dynamic_module"), drop = FALSE]
  w[!duplicated(w$gene), , drop = FALSE]
}

allow_noncount_input <- tolower(Sys.getenv("VP_ALLOW_NONCOUNT_CELLORACLE_INPUT", "false")) %in% c("1", "true", "yes", "y")
scale_audit_rows <- list()

matrix_scale_audit <- function(counts, dataset, celltype, disease, stage) {
  nr <- nrow(counts); nc <- ncol(counts)
  total <- as.double(nr) * as.double(nc)
  values <- if (inherits(counts, "sparseMatrix")) counts@x else as.numeric(counts)
  values <- values[is.finite(values)]
  nonzero <- values[abs(values) > 0]
  min_value <- if (length(values)) min(values) else 0
  max_value <- if (length(values)) max(values) else 0
  integer_fraction <- if (length(nonzero)) mean(abs(nonzero - round(nonzero)) < 1e-8) else 1
  nnz <- if (inherits(counts, "sparseMatrix")) length(counts@x) else sum(counts != 0)
  zero_fraction <- if (total > 0) 1 - nnz / total else NA_real_
  suspected_raw <- isTRUE(min_value >= 0) && isTRUE(integer_fraction >= 0.995)
  suspected_log <- isTRUE(min_value >= 0) && isTRUE(integer_fraction < 0.95) && isTRUE(max_value < 50)
  status <- if (suspected_raw) "raw_count_like" else if (suspected_log) "log_normalized_like" else "scale_uncertain"
  data.frame(
    dataset = dataset, celltype = celltype, disease = disease, stage = stage,
    n_genes = nr, n_cells = nc, min_value = min_value, max_value = max_value,
    zero_fraction = zero_fraction, integer_fraction_nonzero = integer_fraction,
    suspected_raw_counts = suspected_raw, suspected_log_normalized = suspected_log,
    scale_status = status, stringsAsFactors = FALSE
  )
}

assert_count_like <- function(audit_row) {
  if (isTRUE(audit_row$suspected_log_normalized[1]) && !allow_noncount_input) {
    stop(
      "CellOracle input appears log-normalized rather than raw count-like: ",
      audit_row$dataset[1], "/", audit_row$celltype[1], "/", audit_row$disease[1],
      ". Inspect input_scale_audit.csv or set VP_ALLOW_NONCOUNT_CELLORACLE_INPUT=true only for an intentional exploratory run."
    )
  }
}

manifest <- list()
shared_gene_rows <- list()
for (ct in unique(targets$celltype)) {
  bundle_grid <- expand.grid(
    dataset = c("main", "validation"), disease = TARGET_DISEASES,
    stringsAsFactors = FALSE
  )
  bundle_grid$path <- mapply(function(ds, dis) bundle_path(ds, ct, dis), bundle_grid$dataset, bundle_grid$disease)
  if (!all(file.exists(bundle_grid$path))) {
    missing <- bundle_grid$path[!file.exists(bundle_grid$path)]
    msg <- paste(missing, collapse = "; ")
    if (!ALLOW_PARTIAL_INPUTS) stop("Missing CellOracle bundles for ", ct, ": ", msg)
    warning("Skipping ", ct, " because bundles are missing: ", msg)
    next
  }

  bundles <- setNames(lapply(bundle_grid$path, readRDS), paste(bundle_grid$dataset, bundle_grid$disease, sep = "__"))
  for (ii in seq_len(nrow(bundle_grid))) {
    nm <- paste(bundle_grid$dataset[ii], bundle_grid$disease[ii], sep = "__")
    a <- matrix_scale_audit(
      bundles[[nm]]$counts, bundle_grid$dataset[ii], ct, bundle_grid$disease[ii], "input_bundle"
    )
    scale_audit_rows[[paste(ct, nm, "input", sep = "__")]] <- a
    assert_count_like(a)
  }
  for (nm in names(bundles)) {
    use <- balanced_cells(bundles[[nm]]$metadata, CELLORACLE_CELL_LIMIT_PER_CONDITION)
    use <- intersect(use, colnames(bundles[[nm]]$counts))
    bundles[[nm]]$counts <- bundles[[nm]]$counts[, use, drop = FALSE]
    bundles[[nm]]$metadata <- bundles[[nm]]$metadata[use, , drop = FALSE]
  }

  # One common gene universe is selected across all four dataset-condition groups.
  # This prevents main and validation from being compared on different feature sets.
  common_genes <- Reduce(intersect, lapply(bundles, function(x) rownames(x$counts)))
  if (!length(common_genes)) stop("No common genes across the four groups for ", ct)
  # Select variable/background genes from the main dataset only, restricted to
  # genes present in all four groups. This keeps the feature space identical
  # without using validation expression patterns to rank features.
  main_bundle_names <- paste("main", TARGET_DISEASES, sep = "__")
  main_selection_counts <- do.call(cbind, lapply(
    bundles[main_bundle_names], function(x) x$counts[common_genes, , drop = FALSE]
  ))

  ct_targets <- unique(targets$anchor_gene[targets$celltype == ct])
  ct_selected_tfs <- if (nrow(tf_targets)) unique(tf_targets$tf_gene[tf_targets$celltype == ct]) else character()
  ct_dynamic <- if (nrow(dynamic) && all(c("celltype", "gene") %in% colnames(dynamic))) {
    unique(dynamic$gene[dynamic$celltype == ct])
  } else character()

  mandatory_genes <- c(ct_targets, ct_selected_tfs, tf_universe)
  selected <- select_shared_genes(
    main_selection_counts,
    mandatory_genes = mandatory_genes,
    preferred_genes = ct_dynamic,
    n_limit = CELLORACLE_GENE_LIMIT
  )
  genes <- selected$genes
  missing_anchor <- setdiff(ct_targets, genes)
  if (length(missing_anchor)) stop(
    "Anchors were lost from the shared CellOracle gene universe for ", ct, ": ",
    paste(missing_anchor, collapse = ", ")
  )

  shared_gene_rows[[ct]] <- data.frame(
    celltype = ct,
    n_common_genes_all_four_groups = length(common_genes),
    n_expressed_main_selection = selected$n_expressed_pool,
    requested_gene_limit = as.integer(CELLORACLE_GENE_LIMIT),
    actual_gene_count = length(genes),
    gene_limit_expanded_for_mandatory = length(genes) > as.integer(CELLORACLE_GENE_LIMIT),
    n_shared_genes_selected = length(genes),
    n_mandatory_genes_retained = selected$n_mandatory,
    n_tf_universe_retained = sum(tf_universe %in% genes),
    n_tf_universe_total = length(tf_universe),
    identical_main_validation_gene_space = TRUE,
    stringsAsFactors = FALSE
  )

  for (dataset in c("main", "validation")) {
    dataset_names <- paste(dataset, TARGET_DISEASES, sep = "__")
    dataset_bundles <- bundles[dataset_names]
    counts <- do.call(cbind, lapply(dataset_bundles, function(x) x$counts[genes, , drop = FALSE]))
    metadata <- do.call(rbind, lapply(TARGET_DISEASES, function(d) {
      m <- dataset_bundles[[paste(dataset, d, sep = "__")]]$metadata
      m$disease_for_perturbation <- d
      m$cell_id <- rownames(m)
      m
    }))
    metadata <- metadata[colnames(counts), , drop = FALSE]
    exported_audit <- matrix_scale_audit(counts, dataset, ct, "Control+IPF", "exported_celloracle_matrix")
    scale_audit_rows[[paste(ct, dataset, "exported", sep = "__")]] <- exported_audit
    assert_count_like(exported_audit)

    stub <- paste(dataset, ct, sep = "__")
    out_dir <- file.path(OUT, stub)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    Matrix::writeMM(counts, file.path(out_dir, "counts_genes_by_cells.mtx"))
    write.table(data.frame(gene = rownames(counts)), file.path(out_dir, "genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
    write.table(data.frame(cell_id = colnames(counts)), file.path(out_dir, "barcodes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
    write.csv(metadata, file.path(out_dir, "cell_metadata.csv"), row.names = FALSE)

    original_rows <- targets[targets$celltype == ct, , drop = FALSE]
    original_rows$dataset <- dataset
    original_rows$anchor_gene <- original_rows$gene
    write.csv(original_rows, file.path(out_dir, "targets.csv"), row.names = FALSE)

    if (nrow(tf_targets)) {
      tf_rows <- tf_targets[tf_targets$celltype == ct, , drop = FALSE]
      if (nrow(tf_rows)) {
        tf_rows$dataset <- dataset
        tf_rows$gene <- tf_rows$tf_gene
        write.csv(tf_rows, file.path(out_dir, "targets_celloracle.csv"), row.names = FALSE)
      }
    }

    w <- safe_dynamic_weights(dynamic, ct)
    if (nrow(w)) write.csv(w, file.path(out_dir, "axis_gene_weights.csv"), row.names = FALSE)

    manifest[[stub]] <- data.frame(
      dataset = dataset, celltype = ct,
      requested_n_genes = as.integer(CELLORACLE_GENE_LIMIT),
      actual_n_genes = nrow(counts), n_genes = nrow(counts), n_cells = ncol(counts),
      gene_limit_expanded_for_mandatory = nrow(counts) > as.integer(CELLORACLE_GENE_LIMIT),
      n_control = sum(metadata$disease_for_perturbation == "Control"),
      n_ipf = sum(metadata$disease_for_perturbation == "IPF"),
      n_anchor_genes_retained = sum(ct_targets %in% rownames(counts)),
      n_selected_tfs_retained = sum(ct_selected_tfs %in% rownames(counts)),
      n_tf_universe_retained = sum(tf_universe %in% rownames(counts)),
      shared_gene_universe = TRUE,
      has_targets_celloracle = file.exists(file.path(out_dir, "targets_celloracle.csv")),
      directory = out_dir, stringsAsFactors = FALSE
    )
  }
  rm(bundles, main_selection_counts)
  gc(verbose = FALSE)
}

if (!length(manifest)) stop("No CellOracle input directories were exported.")
write.csv(do.call(rbind, manifest), file.path(OUT, "celloracle_input_manifest.csv"), row.names = FALSE)
write.csv(do.call(rbind, shared_gene_rows), file.path(OUT, "celloracle_shared_gene_universe.csv"), row.names = FALSE)
if (length(scale_audit_rows)) {
  write.csv(do.call(rbind, scale_audit_rows), file.path(OUT, "input_scale_audit.csv"), row.names = FALSE)
}
message("CellOracle inputs exported with identical main/validation gene spaces: ", OUT)
message("Input-scale audit: ", file.path(OUT, "input_scale_audit.csv"))
