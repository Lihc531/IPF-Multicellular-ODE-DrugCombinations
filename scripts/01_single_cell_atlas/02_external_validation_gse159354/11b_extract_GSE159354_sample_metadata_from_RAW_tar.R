# 13_extract_GSE159354_sample_metadata_from_RAW_tar.R
# Extract sample-level disease metadata for GSE159354 validation
#
# Main output:
#   results_GSE159354/metadata/GSE159354_sample_metadata_for_validation.csv
#
# This script first inspects GSE159354_RAW.tar.
# If disease labels are not encoded inside the tar archive, it optionally uses GEOquery
# to retrieve sample phenotype metadata from public GEO.

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(readr)
  library(tidyr)
})

project_dir <- getwd()

raw_tar <- file.path(project_dir, "GSE159354_RAW.tar")

out_dir <- file.path(project_dir, "results_GSE159354", "metadata")
tab_dir <- file.path(project_dir, "results_GSE159354", "tables")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

out_meta <- file.path(
  out_dir,
  "GSE159354_sample_metadata_for_validation.csv"
)

# Set TRUE if you want the script to query public GEO metadata when tar does not contain disease labels
USE_GEOQUERY_FALLBACK <- TRUE

# ============================================================
# 1. Inspect RAW tar
# ============================================================

if (!file.exists(raw_tar)) {
  stop("Cannot find RAW tar file: ", raw_tar)
}

tar_files <- untar(raw_tar, list = TRUE)

tar_manifest <- data.frame(
  tar_path = tar_files,
  basename = basename(tar_files),
  stringsAsFactors = FALSE
) %>%
  mutate(
    basename_no_gz = str_remove(basename, "\\.gz$"),
    basename_no_ext = basename_no_gz %>%
      str_remove("\\.h5$") %>%
      str_remove("\\.hdf5$") %>%
      str_remove("\\.mtx$") %>%
      str_remove("\\.tsv$") %>%
      str_remove("\\.txt$") %>%
      str_remove("\\.csv$"),
    GSM = str_extract(basename, "GSM[0-9]+"),
    SAM = str_extract(basename, "SAM[0-9]+"),
    sample_id_candidate = case_when(
      !is.na(GSM) & !is.na(SAM) ~ paste0(GSM, "_", SAM),
      !is.na(GSM) ~ GSM,
      TRUE ~ NA_character_
    )
  )

write.csv(
  tar_manifest,
  file.path(tab_dir, "GSE159354_RAW_tar_manifest.csv"),
  row.names = FALSE
)

message("Number of files in tar:")
print(nrow(tar_manifest))

message("Example files:")
print(head(tar_manifest$tar_path, 20))

sample_from_tar <- tar_manifest %>%
  filter(!is.na(sample_id_candidate)) %>%
  distinct(sample_id = sample_id_candidate, GSM, SAM) %>%
  arrange(sample_id)

if (nrow(sample_from_tar) == 0) {
  warning("No GSM/SAM sample IDs could be parsed from tar file names.")
} else {
  message("Samples parsed from tar:")
  print(sample_from_tar)
}

write.csv(
  sample_from_tar,
  file.path(tab_dir, "GSE159354_samples_parsed_from_RAW_tar.csv"),
  row.names = FALSE
)

# ============================================================
# 2. Try to find metadata-like files inside tar
# ============================================================

metadata_like <- tar_manifest %>%
  filter(
    str_detect(
      tolower(basename),
      "meta|metadata|phenotype|pheno|sample|annotation|clinical|series_matrix"
    ) |
      str_detect(tolower(basename), "\\.csv$|\\.tsv$|\\.txt$|\\.xlsx$")
  )

write.csv(
  metadata_like,
  file.path(tab_dir, "GSE159354_metadata_like_files_in_RAW_tar.csv"),
  row.names = FALSE
)

message("Metadata-like files found inside tar:")
print(metadata_like$tar_path)

# Extract metadata-like files to a temporary folder
extract_dir <- file.path(project_dir, "GSE159354_RAW_metadata_extracted")
dir.create(extract_dir, recursive = TRUE, showWarnings = FALSE)

if (nrow(metadata_like) > 0) {
  untar(
    raw_tar,
    files = metadata_like$tar_path,
    exdir = extract_dir
  )
}

# ============================================================
# 3. Helper functions for disease inference
# ============================================================

infer_disease <- function(x) {
  x0 <- tolower(paste(x, collapse = " "))
  
  case_when(
    str_detect(x0, "copd|chronic obstructive") ~ "COPD",
    str_detect(x0, "ipf|idiopathic pulmonary fibrosis|pulmonary fibrosis|fibrotic") ~ "IPF",
    str_detect(x0, "control|normal|healthy|donor|non[-_ ]?fibrotic|unused donor") ~ "Control",
    TRUE ~ NA_character_
  )
}

standardize_disease <- function(x) {
  x0 <- tolower(as.character(x))
  
  case_when(
    str_detect(x0, "copd|chronic obstructive") ~ "COPD",
    str_detect(x0, "ipf|idiopathic pulmonary fibrosis|pulmonary fibrosis|fibrotic") ~ "IPF",
    str_detect(x0, "control|normal|healthy|donor|non[-_ ]?fibrotic|unused donor") ~ "Control",
    TRUE ~ NA_character_
  )
}

safe_read_delim <- function(path) {
  out <- NULL
  
  try({
    out <- readr::read_csv(path, show_col_types = FALSE)
  }, silent = TRUE)
  
  if (!is.null(out) && ncol(out) > 1) {
    return(out)
  }
  
  try({
    out <- readr::read_tsv(path, show_col_types = FALSE)
  }, silent = TRUE)
  
  if (!is.null(out) && ncol(out) > 1) {
    return(out)
  }
  
  try({
    out <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  }, silent = TRUE)
  
  return(out)
}

# ============================================================
# 4. Search extracted metadata files for disease labels
# ============================================================

extracted_files <- list.files(
  extract_dir,
  recursive = TRUE,
  full.names = TRUE
)

candidate_meta_list <- list()

if (length(extracted_files) > 0) {
  for (f in extracted_files) {
    if (!str_detect(tolower(f), "\\.csv$|\\.tsv$|\\.txt$")) {
      next
    }
    
    message("Trying to read metadata-like file: ", f)
    
    df <- safe_read_delim(f)
    
    if (is.null(df) || nrow(df) == 0) {
      next
    }
    
    df_chr <- df %>%
      mutate(across(everything(), as.character))
    
    # Try to find GSM/SAM/sample_id columns
    all_text_by_row <- apply(df_chr, 1, paste, collapse = " ")
    
    tmp <- data.frame(
      source_file = basename(f),
      row_index = seq_len(nrow(df_chr)),
      row_text = all_text_by_row,
      GSM = str_extract(all_text_by_row, "GSM[0-9]+"),
      SAM = str_extract(all_text_by_row, "SAM[0-9]+"),
      Disease_Identity_candidate = sapply(all_text_by_row, infer_disease),
      stringsAsFactors = FALSE
    ) %>%
      mutate(
        sample_id = case_when(
          !is.na(GSM) & !is.na(SAM) ~ paste0(GSM, "_", SAM),
          !is.na(GSM) ~ GSM,
          TRUE ~ NA_character_
        )
      ) %>%
      filter(!is.na(sample_id) | !is.na(Disease_Identity_candidate))
    
    if (nrow(tmp) > 0) {
      candidate_meta_list[[f]] <- tmp
    }
  }
}

candidate_from_tar_metadata <- bind_rows(candidate_meta_list)

if (nrow(candidate_from_tar_metadata) > 0) {
  write.csv(
    candidate_from_tar_metadata,
    file.path(tab_dir, "GSE159354_disease_candidates_from_tar_metadata_files.csv"),
    row.names = FALSE
  )
} else {
  message("No usable disease metadata found inside tar metadata-like files.")
}

# ============================================================
# 5. GEOquery fallback
# ============================================================

geo_sample_meta <- data.frame()

if (USE_GEOQUERY_FALLBACK) {
  message("Attempting GEOquery fallback for GSE159354 metadata...")
  
  if (!requireNamespace("GEOquery", quietly = TRUE)) {
    message("Package GEOquery not installed. Installing from Bioconductor may be needed:")
    message("if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager')")
    message("BiocManager::install('GEOquery')")
  } else {
    suppressPackageStartupMessages(library(GEOquery))
    
    gse <- GEOquery::getGEO(
      "GSE159354",
      GSEMatrix = TRUE,
      getGPL = FALSE
    )
    
    if (length(gse) > 1) {
      gse_obj <- gse[[1]]
    } else {
      gse_obj <- gse[[1]]
    }
    
    pdata <- Biobase::pData(gse_obj)
    
    write.csv(
      pdata,
      file.path(tab_dir, "GSE159354_GEOquery_pData_raw.csv"),
      row.names = FALSE
    )
    
    pdata_chr <- pdata %>%
      tibble::rownames_to_column("geo_accession_row") %>%
      mutate(across(everything(), as.character))
    
    # GEO sample accession
    if ("geo_accession" %in% colnames(pdata_chr)) {
      pdata_chr$GSM <- pdata_chr$geo_accession
    } else {
      pdata_chr$GSM <- pdata_chr$geo_accession_row
    }
    
    # Combine all columns per sample for disease inference
    pdata_chr$all_metadata_text <- apply(
      pdata_chr,
      1,
      function(x) paste(x, collapse = " | ")
    )
    
    geo_sample_meta <- pdata_chr %>%
      transmute(
        GSM = GSM,
        title = if ("title" %in% colnames(pdata_chr)) title else NA_character_,
        source_name_ch1 = if ("source_name_ch1" %in% colnames(pdata_chr)) source_name_ch1 else NA_character_,
        characteristics_ch1 = if ("characteristics_ch1" %in% colnames(pdata_chr)) characteristics_ch1 else NA_character_,
        all_metadata_text = all_metadata_text,
        Disease_Identity = sapply(all_metadata_text, infer_disease)
      )
    
    # If disease is not inferred from all text, try common columns explicitly
    possible_disease_cols <- grep(
      "disease|diagnosis|condition|phenotype|group|status",
      colnames(pdata_chr),
      ignore.case = TRUE,
      value = TRUE
    )
    
    if (length(possible_disease_cols) > 0) {
      disease_explicit <- pdata_chr %>%
        select(GSM, all_of(possible_disease_cols)) %>%
        pivot_longer(
          cols = all_of(possible_disease_cols),
          names_to = "metadata_column",
          values_to = "metadata_value"
        ) %>%
        mutate(Disease_Identity_explicit = standardize_disease(metadata_value)) %>%
        filter(!is.na(Disease_Identity_explicit)) %>%
        group_by(GSM) %>%
        summarise(
          Disease_Identity_explicit = first(Disease_Identity_explicit),
          disease_source_column = first(metadata_column),
          disease_source_value = first(metadata_value),
          .groups = "drop"
        )
      
      geo_sample_meta <- geo_sample_meta %>%
        left_join(disease_explicit, by = "GSM") %>%
        mutate(
          Disease_Identity = ifelse(
            is.na(Disease_Identity) & !is.na(Disease_Identity_explicit),
            Disease_Identity_explicit,
            Disease_Identity
          )
        )
    }
    
    write.csv(
      geo_sample_meta,
      file.path(tab_dir, "GSE159354_GEOquery_sample_metadata_inferred.csv"),
      row.names = FALSE
    )
  }
}

# ============================================================
# 6. Build final sample metadata table
# ============================================================

if (nrow(sample_from_tar) == 0) {
  # If tar sample parsing failed, fall back to GEO GSMs
  if (nrow(geo_sample_meta) > 0) {
    final_meta <- geo_sample_meta %>%
      transmute(
        sample_id = GSM,
        GSM = GSM,
        SAM = NA_character_,
        Disease_Identity = Disease_Identity,
        disease_metadata_source = "GEOquery"
      )
  } else {
    stop("Could not build sample metadata: no sample IDs from tar and no GEO metadata.")
  }
} else {
  final_meta <- sample_from_tar %>%
    mutate(
      Disease_Identity = NA_character_,
      disease_metadata_source = NA_character_
    )
  
  # Merge GEO metadata by GSM
  if (nrow(geo_sample_meta) > 0) {
    final_meta <- final_meta %>%
      left_join(
        geo_sample_meta %>%
          select(GSM, Disease_Identity_geo = Disease_Identity),
        by = "GSM"
      ) %>%
      mutate(
        Disease_Identity = ifelse(
          is.na(Disease_Identity) & !is.na(Disease_Identity_geo),
          Disease_Identity_geo,
          Disease_Identity
        ),
        disease_metadata_source = ifelse(
          !is.na(Disease_Identity_geo),
          "GEOquery",
          disease_metadata_source
        )
      ) %>%
      select(-Disease_Identity_geo)
  }
  
  # Merge tar metadata candidates by sample_id if available
  if (exists("candidate_from_tar_metadata") && nrow(candidate_from_tar_metadata) > 0) {
    tar_disease <- candidate_from_tar_metadata %>%
      filter(!is.na(sample_id), !is.na(Disease_Identity_candidate)) %>%
      group_by(sample_id) %>%
      summarise(
        Disease_Identity_tar = first(Disease_Identity_candidate),
        tar_metadata_source_file = first(source_file),
        .groups = "drop"
      )
    
    final_meta <- final_meta %>%
      left_join(tar_disease, by = "sample_id") %>%
      mutate(
        Disease_Identity = ifelse(
          is.na(Disease_Identity) & !is.na(Disease_Identity_tar),
          Disease_Identity_tar,
          Disease_Identity
        ),
        disease_metadata_source = ifelse(
          is.na(disease_metadata_source) & !is.na(Disease_Identity_tar),
          paste0("tar:", tar_metadata_source_file),
          disease_metadata_source
        )
      ) %>%
      select(-Disease_Identity_tar, -tar_metadata_source_file)
  }
}

final_meta <- final_meta %>%
  mutate(
    Disease_Identity = ifelse(is.na(Disease_Identity), "Unknown", Disease_Identity),
    disease_metadata_source = ifelse(
      is.na(disease_metadata_source),
      "not_found",
      disease_metadata_source
    )
  ) %>%
  arrange(sample_id)

write.csv(
  final_meta,
  out_meta,
  row.names = FALSE
)

message("Final sample metadata written to:")
message(out_meta)

message("Disease identity summary:")
print(table(final_meta$Disease_Identity, useNA = "ifany"))

message("Metadata source summary:")
print(table(final_meta$disease_metadata_source, useNA = "ifany"))

# ============================================================
# 7. Stop with useful message if disease labels remain unknown
# ============================================================

if (all(final_meta$Disease_Identity == "Unknown")) {
  warning(
    "All Disease_Identity values are Unknown. ",
    "This usually means the RAW tar does not contain disease labels, ",
    "and GEOquery did not infer them automatically. ",
    "Please inspect: ",
    file.path(tab_dir, "GSE159354_GEOquery_pData_raw.csv"),
    " and manually fill ",
    out_meta
  )
}

if (any(final_meta$Disease_Identity == "Unknown")) {
  warning(
    "Some samples still have Unknown Disease_Identity. ",
    "Please inspect the output CSV before running by-disease validation plots."
  )
}