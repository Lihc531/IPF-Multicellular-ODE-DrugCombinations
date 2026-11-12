#!/usr/bin/env Rscript
# Builds each baseline network once, then simulates all selected target KOs
# and outgoing-edge gain perturbations on the same network.

suppressPackageStartupMessages({
  library(Matrix)
  library(scTenifoldNet)
  library(MASS)
})
source("config.R")

TARGET_MANIFEST <- file.path(OUTPUT_ROOT, "01_target_manifest", "target_manifest.csv")
OUT <- file.path(OUTPUT_ROOT, "02_network_perturbation")
NET_DIR <- file.path(OUT, "baseline_networks")
RES_DIR <- file.path(OUT, "gene_results")
dir.create(NET_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RES_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(TARGET_MANIFEST)) stop("Run 02_build_target_manifest.R first.")
targets <- read.csv(TARGET_MANIFEST, stringsAsFactors = FALSE, check.names = FALSE)

strict_direction <- function(X, lambda = 0) {
  S <- sign(X)
  S[abs(S) < abs(t(S))] <- 0
  O <- t(((1 - lambda) * X) + (lambda * S))
  Matrix::Matrix(O)
}

d_regulation <- function(manifold_output, target_gene) {
  gene_list <- rownames(manifold_output)
  gene_list <- gene_list[grepl("^X_", gene_list)]
  gene_list <- sub("^X_", "", gene_list)
  n_genes <- length(gene_list)
  y_genes <- rownames(manifold_output)
  y_genes <- y_genes[grepl("^Y_", y_genes)]
  y_genes <- sub("^Y_", "", y_genes)
  if (length(y_genes) != n_genes || !all(y_genes == gene_list)) stop("Unexpected manifold row ordering.")

  d_metric <- vapply(seq_len(n_genes), function(i) {
    sqrt(sum((as.numeric(manifold_output[i, ]) - as.numeric(manifold_output[i + n_genes, ]))^2))
  }, numeric(1))

  lambda_values <- seq(-2, 2, length.out = 1000)
  lambda_values <- lambda_values[lambda_values != 0]
  positive <- d_metric > 0
  nD <- d_metric
  if (sum(positive) >= 3L) {
    bc <- try(MASS::boxcox(d_metric[positive] ~ 1, plotit = FALSE, lambda = lambda_values), silent = TRUE)
    if (!inherits(bc, "try-error")) {
      lam <- bc$x[which.max(bc$y)]
      nD <- if (lam < 0) 1 / (pmax(d_metric, .Machine$double.eps)^lam) else d_metric^lam
    }
  }
  z <- as.numeric(scale(nD))
  target_idx <- match(target_gene, gene_list)
  reference <- d_metric^2
  if (!is.na(target_idx)) reference <- reference[-target_idx]
  denom <- mean(reference[is.finite(reference)], na.rm = TRUE)
  if (!is.finite(denom) || denom <= 0) denom <- 1
  fc <- (d_metric^2) / denom
  p <- pchisq(fc, df = 1, lower.tail = FALSE)
  data.frame(
    gene = gene_list, distance = d_metric, Z = z, FC = fc,
    p.value = p, p.adj = p.adjust(p, method = "fdr"),
    stringsAsFactors = FALSE
  )[order(fc, decreasing = TRUE), ]
}

balanced_cells <- function(meta, n_max) {
  sample_col <- if ("sample_for_perturbation" %in% colnames(meta)) "sample_for_perturbation" else "sample_id"
  cells <- rownames(meta)
  if (length(cells) <= n_max) return(cells)
  parts <- split(cells, as.character(meta[[sample_col]]))
  quota <- max(1L, floor(n_max / length(parts)))
  chosen <- unlist(lapply(parts, function(v) if (length(v) <= quota) v else sample(v, quota)), use.names = FALSE)
  if (length(chosen) < n_max) {
    rest <- setdiff(cells, chosen)
    chosen <- c(chosen, sample(rest, min(length(rest), n_max - length(chosen))))
  }
  chosen[seq_len(min(length(chosen), n_max))]
}

select_network_genes <- function(counts, candidate_genes, dynamic_genes, n_limit) {
  expressed <- Matrix::rowSums(counts > 0) >= MIN_GENE_EXPRESSED_CELLS
  counts <- counts[expressed, , drop = FALSE]
  lib <- Matrix::colSums(counts)
  lib[lib <= 0] <- 1
  norm <- counts %*% Diagonal(x = 1e4 / lib)
  norm@x <- log1p(norm@x)
  mu <- Matrix::rowMeans(norm)
  second <- Matrix::rowMeans(norm^2)
  variance <- pmax(0, second - mu^2)
  names(variance) <- rownames(norm)
  priority <- unique(c(candidate_genes, dynamic_genes))
  priority <- intersect(priority, rownames(norm))
  remaining <- setdiff(names(sort(variance, decreasing = TRUE)), priority)
  genes <- unique(c(priority, remaining))
  head(genes, n_limit)
}

bundle_path <- function(dataset, celltype, disease) {
  root <- if (dataset == "main") MAIN_DIR else VALIDATION_DIR
  file.path(root, paste0(dataset, "__", celltype, "__", disease, ".rds"))
}

dynamic_path <- file.path(SUPPORT_DIR, "00_combined_dynamic_genes_all_celltypes.csv")
dynamic <- if (file.exists(dynamic_path)) read.csv(dynamic_path, stringsAsFactors = FALSE) else data.frame()

jobs <- expand.grid(
  dataset = c("main", "validation"),
  celltype = unique(targets$celltype),
  disease = TARGET_DISEASES,
  repeat_id = seq_len(NETWORK_REPEATS),
  stringsAsFactors = FALSE
)
jobs$input_file <- mapply(bundle_path, jobs$dataset, jobs$celltype, jobs$disease)
jobs <- jobs[file.exists(jobs$input_file), ]
jobs$job_id <- sprintf("%04d", seq_len(nrow(jobs)))
write.csv(jobs, file.path(OUT, "network_job_manifest.csv"), row.names = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if ("--manifest-only" %in% args) {
  message("Network job manifest written: ", file.path(OUT, "network_job_manifest.csv"))
  quit(save = "no", status = 0)
}
job_index <- if (length(args)) as.integer(args[[1]]) else as.integer(Sys.getenv("VP_JOB_INDEX", unset = "1"))
if (!is.finite(job_index) || job_index < 1 || job_index > nrow(jobs)) stop("Invalid job index. Valid range: 1-", nrow(jobs))
job <- jobs[job_index, ]
message("Running job ", job_index, "/", nrow(jobs), ": ", paste(job[1, c("dataset", "celltype", "disease", "repeat_id")], collapse = " | "))
set.seed(20260723 + job_index + 1000L * job$repeat_id)

bundle <- readRDS(job$input_file)
use_cells <- balanced_cells(bundle$metadata, NETWORK_CELL_LIMIT)
counts <- bundle$counts[, intersect(use_cells, colnames(bundle$counts)), drop = FALSE]
ct_targets <- unique(targets$gene[targets$celltype == job$celltype])
ct_dynamic <- if (nrow(dynamic) && all(c("celltype", "gene") %in% colnames(dynamic))) unique(dynamic$gene[dynamic$celltype == job$celltype]) else character()
genes <- select_network_genes(counts, ct_targets, ct_dynamic, NETWORK_GENE_LIMIT)
counts <- counts[genes, , drop = FALSE]
ct_targets <- intersect(ct_targets, rownames(counts))
if (!length(ct_targets)) stop("No selected target is present after gene filtering.")

n_cells_net <- min(NETWORK_N_CELLS, ncol(counts))
if (n_cells_net < 50) stop("Too few cells for network construction.")

message("Building baseline network: ", nrow(counts), " genes x ", ncol(counts), " cells")
nets <- scTenifoldNet::makeNetworks(
  X = counts,
  q = NETWORK_EDGE_QUANTILE,
  nNet = NETWORK_N_NET,
  nCells = n_cells_net,
  scaleScores = TRUE,
  symmetric = FALSE,
  nComp = NETWORK_N_COMP,
  nCores = NETWORK_N_CORES
)
td <- scTenifoldNet::tensorDecomposition(
  xList = nets, K = NETWORK_TENSOR_K,
  maxError = 1e-5, maxIter = 1000, nDecimal = 3
)
WT <- td$X
WT <- strict_direction(WT, lambda = 0)
WT <- as.matrix(WT)
diag(WT) <- 0
WT <- t(WT)

base_stub <- paste(job$dataset, job$celltype, job$disease, sprintf("rep%02d", job$repeat_id), sep = "__")
network_file <- file.path(NET_DIR, paste0(base_stub, "__WT_network.rds"))
saveRDS(
  list(WT = Matrix(WT, sparse = TRUE), genes = rownames(WT), cells = colnames(counts), job = job),
  network_file, compress = "xz"
)

for (target in ct_targets) {
  for (perturbation in c("KO", "OE_GAIN")) {
    perturbed <- WT
    if (perturbation == "KO") {
      perturbed[target, ] <- 0
    } else {
      perturbed[target, ] <- perturbed[target, ] * NETWORK_OE_EDGE_MULTIPLIER
    }
    ma <- scTenifoldNet::manifoldAlignment(WT, perturbed, d = NETWORK_MANIFOLD_DIM, nCores = NETWORK_N_CORES)
    dr <- d_regulation(ma, target)
    dr$dataset <- job$dataset
    dr$celltype <- job$celltype
    dr$disease <- job$disease
    dr$repeat_id <- job$repeat_id
    dr$target_gene <- target
    dr$perturbation <- perturbation
    dr$oe_edge_multiplier <- ifelse(perturbation == "OE_GAIN", NETWORK_OE_EDGE_MULTIPLIER, NA_real_)
    out_file <- file.path(RES_DIR, paste0(base_stub, "__", perturbation, "__", target, ".csv.gz"))
    con <- gzfile(out_file, open = "wt")
    write.csv(dr, con, row.names = FALSE)
    close(con)
  }
}
message("Completed job ", job_index, ".")
