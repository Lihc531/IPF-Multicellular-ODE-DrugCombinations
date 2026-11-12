#!/usr/bin/env Rscript
# ============================================================
# Repair exporter for a Seurat v5 object with multiple RNA layers.
# It scans every counts.* layer once and exports complete cell-type
# and disease groups without calling JoinLayers on the full object.
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
})

SOURCE_PROJECT_DIR <- "E:/Rproject"
MAIN_SEURAT_RDS <- file.path(SOURCE_PROJECT_DIR, "ipf_annotated_singlet_seurat.rds")
AXIS_METADATA_RDS <- file.path(
  SOURCE_PROJECT_DIR, "results", "progression_axis_v2", "tables",
  "03_cell_metadata_with_axis_v2_module_scores.rds"
)

# Point this at the input root used by config.R.
OUTPUT_ROOT <- file.path(SOURCE_PROJECT_DIR, "virtual_perturbation_inputs")
OUT_DIR <- file.path(OUTPUT_ROOT, "main_by_celltype")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUTPUT_ROOT, "supporting_tables"), recursive = TRUE, showWarnings = FALSE)

TARGET_CELLTYPES <- c(
  "Fibroblast", "Epithelial", "AT2", "Macrophage", "Monocyte",
  "Dendritic", "Endothelial", "T_cell", "NK", "B_cell",
  "Plasma", "Mast", "Smooth_muscle"
)
TARGET_DISEASES <- c("Control", "IPF")
MAX_CELLS_PER_GROUP <- 5000L
MIN_CELLS_PER_GROUP <- 200L
EXPORT_NORMALIZED_DATA <- FALSE
set.seed(20260723)

normalize_disease <- function(x) {
  x <- trimws(as.character(x))
  x[tolower(x) %in% c("control", "ctrl", "healthy", "normal")] <- "Control"
  x[tolower(x) %in% c("ipf", "idiopathic pulmonary fibrosis")] <- "IPF"
  x
}

load_axis_metadata <- function(path) {
  if (!file.exists(path)) return(NULL)
  x <- readRDS(path)
  if (inherits(x, "Seurat")) return(x@meta.data)
  if (is.data.frame(x)) return(x)
  if (is.list(x)) {
    for (nm in c("metadata", "meta", "cell_metadata", "meta_scores")) {
      if (!is.null(x[[nm]]) && is.data.frame(x[[nm]])) return(x[[nm]])
    }
  }
  warning("Axis RDS exists but its metadata structure was not recognized: ", path)
  NULL
}

merge_axis_metadata <- function(obj, axis_meta) {
  if (is.null(axis_meta)) return(obj)
  common <- intersect(colnames(obj), rownames(axis_meta))
  if (!length(common)) {
    warning("No matching cell IDs between the main object and axis metadata.")
    return(obj)
  }
  useful <- setdiff(colnames(axis_meta), colnames(obj@meta.data))
  if (length(useful)) {
    obj@meta.data[common, useful] <- axis_meta[common, useful, drop = FALSE]
  }
  obj
}

balanced_sample <- function(cells, sample_ids, n_max) {
  if (length(cells) <= n_max) return(cells)
  split_cells <- split(cells, sample_ids[cells])
  quota <- max(1L, floor(n_max / length(split_cells)))
  selected <- unlist(lapply(split_cells, function(v) {
    if (length(v) <= quota) v else sample(v, quota)
  }), use.names = FALSE)
  if (length(selected) < n_max) {
    remaining <- setdiff(cells, selected)
    n_add <- min(length(remaining), n_max - length(selected))
    if (n_add > 0) selected <- c(selected, sample(remaining, n_add))
  }
  selected[seq_len(min(length(selected), n_max))]
}

bind_sparse_pieces <- function(pieces) {
  pieces <- Filter(Negate(is.null), pieces)
  if (!length(pieces)) return(NULL)
  if (length(pieces) == 1L) return(pieces[[1]])
  common_genes <- Reduce(intersect, lapply(pieces, rownames))
  if (!length(common_genes)) stop("RNA layers do not share gene names.")
  pieces <- lapply(pieces, function(x) x[common_genes, , drop = FALSE])
  out <- do.call(cbind, pieces)
  as(out, "dgCMatrix")
}

if (!file.exists(MAIN_SEURAT_RDS)) stop("Missing main Seurat object: ", MAIN_SEURAT_RDS)
message("Loading main Seurat object...")
obj <- readRDS(MAIN_SEURAT_RDS)
obj <- merge_axis_metadata(obj, load_axis_metadata(AXIS_METADATA_RDS))
DefaultAssay(obj) <- "RNA"

meta <- obj@meta.data
required_cols <- c("celltype_marker_based", "Disease_Identity", "sample_id")
missing_cols <- setdiff(required_cols, colnames(meta))
if (length(missing_cols)) stop("Missing metadata columns: ", paste(missing_cols, collapse = ", "))

meta$cell_id <- rownames(meta)
meta$celltype_for_perturbation <- as.character(meta$celltype_marker_based)
meta$disease_for_perturbation <- normalize_disease(meta$Disease_Identity)
meta$sample_for_perturbation <- as.character(meta$sample_id)
sample_map <- setNames(meta$sample_for_perturbation, meta$cell_id)

selected_by_group <- list()
manifest_rows <- list()
for (ct in TARGET_CELLTYPES) {
  for (dis in TARGET_DISEASES) {
    cells <- meta$cell_id[
      meta$celltype_for_perturbation == ct &
        meta$disease_for_perturbation == dis
    ]
    n_before <- length(cells)
    if (n_before < MIN_CELLS_PER_GROUP) next
    cells <- balanced_sample(cells, sample_map, MAX_CELLS_PER_GROUP)
    gid <- paste(ct, dis, sep = "||")
    selected_by_group[[gid]] <- cells
    manifest_rows[[gid]] <- data.frame(
      dataset = "main", celltype = ct, disease = dis,
      n_cells_before = n_before, n_cells_selected = length(cells),
      stringsAsFactors = FALSE
    )
  }
}
if (!length(selected_by_group)) stop("No groups passed the minimum-cell threshold.")

all_selected_cells <- unlist(selected_by_group, use.names = FALSE)
all_selected_groups <- rep(names(selected_by_group), lengths(selected_by_group))
cell_to_group <- setNames(all_selected_groups, all_selected_cells)

extract_all_layers <- function(layer_pattern) {
  layer_names <- Layers(obj[["RNA"]], search = layer_pattern)
  if (!length(layer_names)) return(setNames(vector("list", length(selected_by_group)), names(selected_by_group)))
  message("Found layers: ", paste(layer_names, collapse = ", "))
  pieces <- setNames(vector("list", length(selected_by_group)), names(selected_by_group))
  for (ln in layer_names) {
    message("Reading RNA layer: ", ln)
    m <- LayerData(obj, assay = "RNA", layer = ln, fast = FALSE)
    if (!inherits(m, "sparseMatrix")) m <- as(m, "dgCMatrix")
    hit <- intersect(colnames(m), names(cell_to_group))
    if (!length(hit)) next
    gids <- unique(unname(cell_to_group[hit]))
    for (gid in gids) {
      use <- hit[unname(cell_to_group[hit]) == gid]
      pieces[[gid]][[length(pieces[[gid]]) + 1L]] <- m[, use, drop = FALSE]
    }
    rm(m); gc(verbose = FALSE)
  }
  lapply(pieces, bind_sparse_pieces)
}

message("Extracting all counts.* layers...")
counts_by_group <- extract_all_layers("^counts")

data_by_group <- NULL
if (EXPORT_NORMALIZED_DATA) {
  message("Extracting all data.* layers...")
  data_by_group <- extract_all_layers("^data")
}

metadata_cols <- unique(c(
  "cell_id", "celltype_for_perturbation", "disease_for_perturbation",
  "sample_for_perturbation", "sample_id", "Disease_Identity",
  "celltype_marker_based", "Subclass_Cell_Identity", "cluster_default",
  "seurat_clusters", "axis_v2_score", "axis_v2_stage",
  "fibrosis_composite_score", "epithelial_injury_composite_score",
  "injury_inflammation_composite_score", "percent.mt", "nCount_RNA",
  "nFeature_RNA"
))
metadata_cols <- metadata_cols[metadata_cols %in% colnames(meta)]

export_rows <- list()
for (gid in names(selected_by_group)) {
  counts <- counts_by_group[[gid]]
  if (is.null(counts)) {
    warning("No counts found for group: ", gid)
    next
  }
  desired <- selected_by_group[[gid]]
  found <- intersect(desired, colnames(counts))
  counts <- counts[, found, drop = FALSE]
  parts <- strsplit(gid, "\\|\\|", fixed = FALSE)[[1]]
  ct <- parts[[1]]; dis <- parts[[2]]

  norm <- NULL
  if (!is.null(data_by_group) && !is.null(data_by_group[[gid]])) {
    norm <- data_by_group[[gid]][, intersect(found, colnames(data_by_group[[gid]])), drop = FALSE]
  }

  bundle <- list(
    dataset = "main",
    celltype = ct,
    disease = dis,
    counts = counts,
    normalized_data = norm,
    metadata = meta[found, metadata_cols, drop = FALSE],
    source_celltype_column = "celltype_marker_based",
    source_disease_column = "Disease_Identity",
    source_sample_column = "sample_id",
    n_cells_before_downsampling = manifest_rows[[gid]]$n_cells_before,
    n_cells_exported = length(found),
    source_rna_layers = Layers(obj[["RNA"]], search = "^counts")
  )
  out_file <- file.path(OUT_DIR, paste0("main__", ct, "__", dis, ".rds"))
  saveRDS(bundle, out_file, compress = "xz")
  export_rows[[gid]] <- data.frame(
    dataset = "main", celltype = ct, disease = dis,
    n_cells_before = manifest_rows[[gid]]$n_cells_before,
    n_cells_selected = length(desired), n_cells_exported = length(found),
    n_genes = nrow(counts), file = out_file,
    stringsAsFactors = FALSE
  )
  message("Saved: ", out_file, " [", nrow(counts), " genes x ", ncol(counts), " cells]")
}

manifest <- do.call(rbind, export_rows)
write.csv(manifest, file.path(OUTPUT_ROOT, "main_export_manifest_v5_layers.csv"), row.names = FALSE)

if (any(manifest$n_cells_exported < pmin(manifest$n_cells_before, MAX_CELLS_PER_GROUP))) {
  warning("At least one group exported fewer cells than selected. Inspect the manifest.")
}
message("Main-data re-export completed: ", OUT_DIR)
message("Note: this script only creates supporting_tables; it does not populate it.")
message("Next run 00b_collect_supporting_tables.R before 01_validate_inputs.R.")
