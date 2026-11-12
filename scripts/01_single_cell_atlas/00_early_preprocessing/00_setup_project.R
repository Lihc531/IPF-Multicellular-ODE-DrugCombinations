# 00_setup_project.R
# Create project folders and install/load required packages.

project_dir <- getwd()

dirs <- c(
  "data_raw",
  "data_raw/GSE136831",
  "data_raw/GSE159354",
  "metadata",
  "objects",
  "results",
  "results/qc",
  "results/figures",
  "results/markers",
  "results/tables",
  "scripts"
)

for (d in dirs) {
  dir.create(file.path(project_dir, d), recursive = TRUE, showWarnings = FALSE)
}

required_pkgs <- c(
  "Seurat",
  "SeuratObject",
  "Matrix",
  "dplyr",
  "ggplot2",
  "patchwork",
  "readr",
  "stringr",
  "tibble",
  "GEOquery",
  "harmony",
  "SingleR",
  "celldex",
  "scDblFinder",
  "SingleCellExperiment"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing package: ", pkg)
    if (pkg %in% c("SingleR", "celldex", "scDblFinder", "SingleCellExperiment")) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      install.packages(pkg)
    }
  }
}

invisible(lapply(required_pkgs, install_if_missing))
invisible(lapply(required_pkgs, library, character.only = TRUE))

sample_template <- data.frame(
  sample_id = character(),
  dataset = character(),
  patient_id = character(),
  condition = character(),
  disease_group = character(),
  control_type = character(),
  sex = character(),
  age = character(),
  smoking_status = character(),
  tissue_region = character(),
  fibrosis_severity = character(),
  platform = character(),
  chemistry = character(),
  batch = character(),
  raw_file = character(),
  metadata_file = character(),
  use_for_main_analysis = logical(),
  notes = character()
)

write.csv(
  sample_template,
  file = file.path(project_dir, "metadata", "sample_metadata_template.csv"),
  row.names = FALSE
)

message("Project folders created.")
message("Sample metadata template written to metadata/sample_metadata_template.csv")
