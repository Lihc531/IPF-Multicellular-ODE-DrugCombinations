# 01_download_geo_data.R
# Download GEO supplementary files for IPF scRNA-seq datasets.

library(GEOquery)

project_dir <- getwd()

download_geo_supp <- function(accession, dest_dir) {
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  message("Downloading GEO supplementary files for: ", accession)
  GEOquery::getGEOSuppFiles(
    GEO = accession,
    baseDir = dirname(dest_dir),
    makeDirectory = TRUE
  )
  message("Downloaded: ", accession)
}

# Main IPF atlas dataset
download_geo_supp(
  accession = "GSE136831",
  dest_dir = file.path(project_dir, "data_raw", "GSE136831")
)

# Optional external validation dataset
# Uncomment when needed.
# download_geo_supp(
#   accession = "GSE159354",
#   dest_dir = file.path(project_dir, "data_raw", "GSE159354")
# )

# Unpack common compressed files
unpack_all_gz <- function(folder) {
  gz_files <- list.files(folder, pattern = "\\.gz$", recursive = TRUE, full.names = TRUE)
  for (f in gz_files) {
    out <- sub("\\.gz$", "", f)
    if (!file.exists(out)) {
      message("Unzipping: ", f)
      R.utils::gunzip(f, overwrite = FALSE, remove = FALSE)
    }
  }
}

if (!requireNamespace("R.utils", quietly = TRUE)) {
  install.packages("R.utils")
}

# Do not automatically unzip very large files unless needed.
# unpack_all_gz(file.path(project_dir, "data_raw", "GSE136831"))

message("Download step finished.")
message("Please inspect data_raw/GSE136831 and confirm file names before running 02_build_seurat_object.R")
