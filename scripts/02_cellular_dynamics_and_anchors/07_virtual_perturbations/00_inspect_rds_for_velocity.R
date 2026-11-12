#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript 00_inspect_rds_for_velocity.R path/to/GSE136831_RAW.tar")
}

tar_path <- args[1]

if (!file.exists(tar_path)) {
  stop("TAR file not found: ", tar_path)
}

message("Inspecting TAR file: ", tar_path)

tar_files <- utils::untar(tar_path, list = TRUE)
rds_files <- tar_files[grepl("\\.rds$", tar_files, ignore.case = TRUE)]

message("Number of RDS files found: ", length(rds_files))

if (length(rds_files) == 0) {
  message("No RDS files found inside TAR.")
  quit(status = 0)
}

velocity_keywords <- c("spliced", "unspliced", "ambiguous", "velocity", "velocyto")

inspect_one_rds <- function(tar_path, rds_member) {
  tmpdir <- tempfile("rds_check_")
  dir.create(tmpdir, recursive = TRUE, showWarnings = FALSE)
  
  message("\n--------------------------------------------")
  message("Extracting: ", rds_member)
  
  utils::untar(tar_path, files = rds_member, exdir = tmpdir)
  
  extracted_path <- file.path(tmpdir, rds_member)
  
  if (!file.exists(extracted_path)) {
    candidates <- list.files(tmpdir, recursive = TRUE, full.names = TRUE)
    extracted_path <- candidates[basename(candidates) == basename(rds_member)][1]
  }
  
  if (is.na(extracted_path) || !file.exists(extracted_path)) {
    message("Could not find extracted RDS file.")
    return(NULL)
  }
  
  obj <- readRDS(extracted_path)
  
  message("Object class:")
  print(class(obj))
  
  found <- character()
  
  # General names
  obj_names <- tryCatch(names(obj), error = function(e) NULL)
  if (!is.null(obj_names)) {
    message("Top-level names:")
    print(head(obj_names, 50))
    found <- c(found, obj_names)
  }
  
  # Seurat-like object
  if ("Seurat" %in% class(obj)) {
    message("Detected Seurat object.")
    
    assays <- tryCatch(names(obj@assays), error = function(e) NULL)
    message("Assays:")
    print(assays)
    found <- c(found, assays)
    
    for (assay in assays) {
      message("Checking assay: ", assay)
      
      assay_obj <- obj@assays[[assay]]
      
      slot_names <- tryCatch(slotNames(assay_obj), error = function(e) NULL)
      message("Slot names:")
      print(slot_names)
      found <- c(found, slot_names)
      
      # Seurat v5 layers
      layer_names <- tryCatch({
        if (requireNamespace("SeuratObject", quietly = TRUE)) {
          SeuratObject::Layers(assay_obj)
        } else {
          NULL
        }
      }, error = function(e) NULL)
      
      if (!is.null(layer_names)) {
        message("Layer names:")
        print(layer_names)
        found <- c(found, layer_names)
      }
    }
  }
  
  # SingleCellExperiment-like object
  if ("SingleCellExperiment" %in% class(obj) || "SummarizedExperiment" %in% class(obj)) {
    message("Detected SingleCellExperiment / SummarizedExperiment-like object.")
    
    assay_names <- tryCatch(SummarizedExperiment::assayNames(obj), error = function(e) NULL)
    message("Assay names:")
    print(assay_names)
    found <- c(found, assay_names)
    
    alt_names <- tryCatch(SingleCellExperiment::altExpNames(obj), error = function(e) NULL)
    message("AltExp names:")
    print(alt_names)
    found <- c(found, alt_names)
  }
  
  found_lower <- tolower(unique(found))
  velocity_hits <- found_lower[
    sapply(found_lower, function(x) any(grepl(paste(velocity_keywords, collapse = "|"), x)))
  ]
  
  message("Velocity-related internal names:")
  print(velocity_hits)
  
  unlink(tmpdir, recursive = TRUE)
  
  return(data.frame(
    rds_member = rds_member,
    object_class = paste(class(obj), collapse = ";"),
    velocity_internal_hits = paste(velocity_hits, collapse = ";"),
    likely_contains_velocity_layers = length(velocity_hits) > 0,
    stringsAsFactors = FALSE
  ))
}

# 为了避免一次性读太多，只抽查前 5 个 RDS
n_check <- min(5, length(rds_files))

message("Will inspect first ", n_check, " RDS files.")
message("If these are representative and no spliced/unspliced appears, direct RNA velocity is unlikely.")

res_list <- list()

for (i in seq_len(n_check)) {
  res_list[[i]] <- tryCatch(
    inspect_one_rds(tar_path, rds_files[i]),
    error = function(e) {
      message("Error inspecting ", rds_files[i], ": ", conditionMessage(e))
      data.frame(
        rds_member = rds_files[i],
        object_class = NA_character_,
        velocity_internal_hits = paste("ERROR:", conditionMessage(e)),
        likely_contains_velocity_layers = FALSE,
        stringsAsFactors = FALSE
      )
    }
  )
}

res <- do.call(rbind, res_list)

out_csv <- file.path(dirname(tar_path), "rds_velocity_internal_check_report.csv")
write.csv(res, out_csv, row.names = FALSE)

message("\n========== Final RDS inspection summary ==========")
print(res)

if (any(res$likely_contains_velocity_layers, na.rm = TRUE)) {
  message("\n[POSSIBLY YES] Some RDS objects contain velocity-related assays/layers/names.")
  message("Please inspect the report and confirm whether spliced/unspliced matrices are real count layers.")
} else {
  message("\n[LIKELY NO DIRECT RNA VELOCITY FROM RDS] No spliced/unspliced/velocity-related internal names found in inspected RDS objects.")
  message("If no FASTQ/BAM/loom/h5ad exists either, use pseudotime-based CellRank instead.")
}

message("Report saved to: ", out_csv)