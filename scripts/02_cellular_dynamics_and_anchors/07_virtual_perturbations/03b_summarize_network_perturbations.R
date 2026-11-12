#!/usr/bin/env Rscript
source("config.R")

ROOT <- file.path(OUTPUT_ROOT, "02_network_perturbation")
IN_DIR <- file.path(ROOT, "gene_results")
OUT <- file.path(ROOT, "summary")
JOB_MANIFEST <- file.path(ROOT, "network_job_manifest.csv")
TARGET_MANIFEST <- file.path(OUTPUT_ROOT, "01_target_manifest", "target_manifest.csv")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(JOB_MANIFEST)) stop("Missing network_job_manifest.csv. Run 03_run_network_perturbations.R first.")
if (!file.exists(TARGET_MANIFEST)) stop("Missing target_manifest.csv. Run 02_build_target_manifest.R first.")
jobs <- read.csv(JOB_MANIFEST, stringsAsFactors = FALSE, check.names = FALSE)
targets <- read.csv(TARGET_MANIFEST, stringsAsFactors = FALSE, check.names = FALSE)

expected_rows <- list()
k <- 0L
for (i in seq_len(nrow(jobs))) {
  job <- jobs[i, , drop = FALSE]
  ct_targets <- unique(targets$gene[targets$celltype == job$celltype])
  base_stub <- paste(
    job$dataset, job$celltype, job$disease,
    sprintf("rep%02d", as.integer(job$repeat_id)), sep = "__"
  )
  for (target in ct_targets) {
    for (perturbation in c("KO", "OE_GAIN")) {
      k <- k + 1L
      path <- file.path(IN_DIR, paste0(base_stub, "__", perturbation, "__", target, ".csv.gz"))
      expected_rows[[k]] <- data.frame(
        dataset = job$dataset, celltype = job$celltype, disease = job$disease,
        repeat_id = job$repeat_id, target_gene = target, perturbation = perturbation,
        expected_file = path, exists = file.exists(path), stringsAsFactors = FALSE
      )
    }
  }
}
completion <- if (length(expected_rows)) do.call(rbind, expected_rows) else data.frame()
write.csv(completion, file.path(OUT, "network_completion_status.csv"), row.names = FALSE)
missing <- completion[!completion$exists, , drop = FALSE]
if (nrow(missing)) {
  write.csv(missing, file.path(OUT, "missing_network_results.csv"), row.names = FALSE)
  msg <- paste0(nrow(missing), "/", nrow(completion), " expected network result files are missing")
  if (!ALLOW_PARTIAL_VALIDATION) {
    stop(msg, ". Complete the missing jobs before summarization. See missing_network_results.csv.")
  }
  warning(msg, "; proceeding because VP_ALLOW_PARTIAL_VALIDATION=true.")
}

files <- completion$expected_file[completion$exists]
if (!length(files)) stop("No network perturbation result files found.")
all <- do.call(rbind, lapply(files, function(f) read.csv(gzfile(f), stringsAsFactors = FALSE)))
key_cols <- c("dataset", "celltype", "disease", "target_gene", "perturbation", "gene")
keys <- interaction(all[, key_cols], drop = TRUE, lex.order = TRUE)
idx <- split(seq_len(nrow(all)), keys)
summary_rows <- lapply(idx, function(ii) {
  x <- all[ii, , drop = FALSE]
  data.frame(
    dataset = x$dataset[1], celltype = x$celltype[1], disease = x$disease[1],
    target_gene = x$target_gene[1], perturbation = x$perturbation[1], gene = x$gene[1],
    median_distance = median(x$distance, na.rm = TRUE),
    median_FC = median(x$FC, na.rm = TRUE),
    median_Z = median(x$Z, na.rm = TRUE),
    median_p_adj = median(x$p.adj, na.rm = TRUE),
    significant_fraction = mean(x$p.adj < 0.05, na.rm = TRUE),
    n_repeats = length(unique(x$repeat_id)), stringsAsFactors = FALSE
  )
})
agg <- do.call(rbind, summary_rows)
agg <- agg[order(agg$dataset, agg$celltype, agg$disease, agg$target_gene, agg$perturbation, -agg$median_FC), ]
write.csv(agg, file.path(OUT, "network_perturbation_gene_summary.csv"), row.names = FALSE)

agg$n_significant <- as.integer(agg$median_p_adj < 0.05)
candidate_keys <- c("dataset", "celltype", "disease", "target_gene", "perturbation")
key_factor <- interaction(agg[, candidate_keys], drop = TRUE, lex.order = TRUE)
candidate_summary <- do.call(rbind, lapply(split(seq_len(nrow(agg)), key_factor), function(ii) {
  x <- agg[ii, , drop = FALSE]
  data.frame(
    dataset = x$dataset[1], celltype = x$celltype[1], disease = x$disease[1],
    target_gene = x$target_gene[1], perturbation = x$perturbation[1],
    n_significant = sum(x$n_significant, na.rm = TRUE),
    median_FC_across_genes = median(x$median_FC, na.rm = TRUE),
    n_repeats = min(x$n_repeats, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
write.csv(candidate_summary, file.path(OUT, "network_perturbation_candidate_summary.csv"), row.names = FALSE)
message("Network summaries written to: ", OUT)
