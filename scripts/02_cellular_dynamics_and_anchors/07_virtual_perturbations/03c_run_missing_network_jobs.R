#!/usr/bin/env Rscript
# Resume-safe runner for all network perturbation jobs.
source("config.R")

script <- normalizePath("03_run_network_perturbations.R", winslash = "/", mustWork = TRUE)
rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
status <- system2(rscript, c(script, "--manifest-only"))
if (status != 0L) stop("Could not create network_job_manifest.csv.")

root <- file.path(OUTPUT_ROOT, "02_network_perturbation")
jobs <- read.csv(file.path(root, "network_job_manifest.csv"), stringsAsFactors = FALSE)
targets <- read.csv(file.path(OUTPUT_ROOT, "01_target_manifest", "target_manifest.csv"), stringsAsFactors = FALSE)
res_dir <- file.path(root, "gene_results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

job_complete <- function(job) {
  ct_targets <- unique(targets$gene[targets$celltype == job$celltype])
  stub <- paste(
    job$dataset, job$celltype, job$disease,
    sprintf("rep%02d", as.integer(job$repeat_id)), sep = "__"
  )
  expected <- unlist(lapply(ct_targets, function(target) {
    file.path(res_dir, paste0(stub, "__", c("KO", "OE_GAIN"), "__", target, ".csv.gz"))
  }), use.names = FALSE)
  length(expected) > 0L && all(file.exists(expected))
}

for (i in seq_len(nrow(jobs))) {
  if (job_complete(jobs[i, , drop = FALSE])) {
    message("Skipping completed network job ", i, "/", nrow(jobs))
    next
  }
  message("Running network job ", i, "/", nrow(jobs))
  status <- system2(rscript, c(script, as.character(i)))
  if (status != 0L) stop("Network job failed at index ", i, ". Fix the error and rerun this resume-safe script.")
}
message("All network jobs are complete. Run 03b_summarize_network_perturbations.R.")
