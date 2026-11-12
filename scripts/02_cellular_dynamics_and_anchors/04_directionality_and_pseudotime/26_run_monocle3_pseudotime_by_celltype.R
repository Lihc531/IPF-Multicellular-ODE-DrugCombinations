suppressPackageStartupMessages({
  library(Seurat)
  library(monocle3)
  library(dplyr)
  library(ggplot2)
  library(Matrix)
})

# ============================================================
# Fibroblast-lineage progression pseudotime analysis using Monocle3
# Robust version for:
#   1) Seurat v5 multi-layer assays
#   2) preserving original cell type / disease annotations
#   3) Fibroblast + Myofibroblast lineage selection
#   4) Control / COPD / IPF disease labels
#   5) avoiding Inf pseudotime caused by disconnected partitions
#   6) finite-only statistical summaries
# ============================================================

# =========================
# 1. 参数区
# =========================

project_dir <- getwd()

seurat_rds <- file.path(project_dir, "ipf_annotated_singlet_seurat.rds")

axis_v2_rds <- file.path(
  project_dir,
  "results",
  "progression_axis_v2",
  "tables",
  "03_cell_metadata_with_axis_v2_module_scores.rds"
)

target_celltype_name <- "Fibroblast_lineage"

# 当前对象里真正带文字细胞类型注释的列
celltype_col <- "Subclass_Cell_Identity"

# 推荐用于 fibrosis progression 的 lineage
target_celltypes <- c(
  "Fibroblast",
  "Myofibroblast"
)

# 当前对象里该列已经是 Control / COPD / IPF
disease_col <- "Disease_Identity"
disease_order <- c("Control", "COPD", "IPF")

axis_score_col <- "axis_v2_score"
axis_stage_col <- "axis_v2_stage"
fibrosis_col <- "fibrosis_composite_score"

# Seurat assay
assay_use <- NULL   # NULL = use DefaultAssay(obj)

# Monocle3 parameters
num_dim <- 50

# 关键：FALSE 会让 Monocle3 学一个整体 graph，通常可避免 root 只在一个 partition 内导致其他 partition 为 Inf pseudotime
learn_graph_use_partition <- FALSE

# root cell 选择：axis_v2_score 最低的 5%
root_quantile <- 0.05
min_root_cells <- 20

# 是否抽样；默认不抽样
max_cells <- Inf
random_seed <- 123

set.seed(random_seed)

outdir <- file.path(project_dir, "results", "pseudotime_results", target_celltype_name)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("Project dir: ", project_dir)
message("Output dir: ", outdir)

# =========================
# 2. helper functions
# =========================

finite_or_na <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA_real_
  x
}

mean_finite <- function(x) {
  x <- finite_or_na(x)
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

median_finite <- function(x) {
  x <- finite_or_na(x)
  if (all(is.na(x))) return(NA_real_)
  median(x, na.rm = TRUE)
}

quantile_finite <- function(x, prob) {
  x <- finite_or_na(x)
  if (all(is.na(x))) return(NA_real_)
  as.numeric(quantile(x, probs = prob, na.rm = TRUE, names = FALSE))
}

safe_table <- function(x, useNA = "ifany") {
  sort(table(x, useNA = useNA), decreasing = TRUE)
}

add_metadata_from_axis <- function(obj, meta_axis, cols, common_cells, overwrite = FALSE) {
  for (cc in cols) {
    if (!cc %in% colnames(meta_axis)) {
      warning("Column not found in axis metadata, skipped: ", cc)
      next
    }
    
    if (!cc %in% colnames(obj@meta.data)) {
      obj@meta.data[[cc]] <- NA
      obj@meta.data[common_cells, cc] <- meta_axis[common_cells, cc]
      message("Added metadata column from axis metadata: ", cc)
    } else if (overwrite) {
      obj@meta.data[common_cells, cc] <- meta_axis[common_cells, cc]
      message("Updated metadata column from axis metadata: ", cc)
    } else {
      message("Preserved existing Seurat metadata column: ", cc)
    }
  }
  obj
}

make_factor_if_matching <- function(x, preferred_levels) {
  x_chr <- as.character(x)
  observed <- unique(x_chr[!is.na(x_chr)])
  
  if (length(observed) > 0 && all(observed %in% preferred_levels)) {
    factor(x_chr, levels = preferred_levels)
  } else {
    x
  }
}

# =========================
# 3. 读取 Seurat 对象和 axis_v2 metadata
# =========================

message("Loading Seurat object...")
obj <- readRDS(seurat_rds)

message("Loading axis_v2 metadata...")
meta_axis <- readRDS(axis_v2_rds)

if (!is.data.frame(meta_axis)) {
  meta_axis <- as.data.frame(meta_axis)
}

if (!"cell_id" %in% colnames(meta_axis)) {
  if ("CellBarcode_Identity" %in% colnames(meta_axis)) {
    meta_axis$cell_id <- as.character(meta_axis$CellBarcode_Identity)
  } else {
    meta_axis$cell_id <- rownames(meta_axis)
  }
}

meta_axis$cell_id <- as.character(meta_axis$cell_id)
rownames(meta_axis) <- meta_axis$cell_id

if (anyDuplicated(rownames(meta_axis)) > 0) {
  warning("Duplicated cell_id detected in axis metadata. Keeping first occurrence.")
  meta_axis <- meta_axis[!duplicated(rownames(meta_axis)), , drop = FALSE]
}

common_cells <- intersect(colnames(obj), rownames(meta_axis))

message("Cells in Seurat: ", ncol(obj))
message("Cells in axis metadata: ", nrow(meta_axis))
message("Overlapping cells: ", length(common_cells))

if (length(common_cells) < 1000) {
  stop("Too few overlapping cells between Seurat object and axis_v2 metadata.")
}

# =========================
# 4. 合并 metadata
# =========================
# 核心修复：
#   - axis / fibrosis 分数来自 axis metadata，可以更新
#   - cell type / disease 必须优先保留 Seurat 原始注释，避免被 axis metadata 覆盖成数字或不完整标签
# =========================

axis_meta_cols <- c(axis_score_col, axis_stage_col, fibrosis_col)
available_axis_meta_cols <- intersect(axis_meta_cols, colnames(meta_axis))

message("Axis metadata columns available:")
print(available_axis_meta_cols)

if (length(available_axis_meta_cols) == 0) {
  stop("None of the required axis/fibrosis metadata columns were found in axis_v2 metadata.")
}

# axis/fibrosis 列允许从 axis metadata 更新
obj <- add_metadata_from_axis(
  obj = obj,
  meta_axis = meta_axis,
  cols = available_axis_meta_cols,
  common_cells = common_cells,
  overwrite = TRUE
)

# celltype / disease 只在 Seurat 对象缺失时才从 axis metadata 补充，不覆盖已有原始注释
fallback_cols <- c(celltype_col, disease_col, "sample_id")
fallback_cols <- fallback_cols[
  !fallback_cols %in% colnames(obj@meta.data) &
    fallback_cols %in% colnames(meta_axis)
]

if (length(fallback_cols) > 0) {
  message("Adding missing annotation columns from axis metadata as fallback:")
  print(fallback_cols)
  obj <- add_metadata_from_axis(
    obj = obj,
    meta_axis = meta_axis,
    cols = fallback_cols,
    common_cells = common_cells,
    overwrite = FALSE
  )
}

rm(meta_axis)
gc()

needed_cols <- c(
  celltype_col,
  disease_col,
  axis_score_col,
  axis_stage_col,
  fibrosis_col
)

missing_cols <- setdiff(needed_cols, colnames(obj@meta.data))

if (length(missing_cols) > 0) {
  stop("Missing required metadata columns: ", paste(missing_cols, collapse = ", "))
}

message("Required metadata columns are present:")
print(needed_cols)

message("Disease distribution:")
print(safe_table(obj@meta.data[[disease_col]], useNA = "ifany"))

message("Cell type distribution in selected column:")
print(safe_table(obj@meta.data[[celltype_col]], useNA = "ifany"))

message("Fibroblast-like labels in selected column:")
print(grep(
  "fibro|fib|stromal|myofibro",
  unique(as.character(obj@meta.data[[celltype_col]])),
  ignore.case = TRUE,
  value = TRUE
))

# =========================
# 5. 提取目标细胞
# =========================

message("Using celltype column: ", celltype_col)
message("Target cell types:")
print(target_celltypes)

cells_celltype <- rownames(obj@meta.data)[
  as.character(obj@meta.data[[celltype_col]]) %in% target_celltypes &
    !is.na(obj@meta.data[[celltype_col]])
]

message("Cells matching target cell types only: ", length(cells_celltype))

cells_with_axis <- rownames(obj@meta.data)[
  !is.na(obj@meta.data[[axis_score_col]]) &
    !is.na(obj@meta.data[[axis_stage_col]]) &
    !is.na(obj@meta.data[[fibrosis_col]])
]

message("Cells with complete axis/fibrosis metadata: ", length(cells_with_axis))

cells_use <- intersect(cells_celltype, cells_with_axis)
cells_use <- intersect(cells_use, colnames(obj))

message("Final cells before sampling: ", length(cells_use))

message("Selected cell type composition:")
print(table(obj@meta.data[cells_use, celltype_col], useNA = "ifany"))

message("Selected disease composition:")
print(table(obj@meta.data[cells_use, disease_col], useNA = "ifany"))

if (length(cells_use) < 100) {
  stop(
    "Too few cells for target cell types: ",
    paste(target_celltypes, collapse = ", "),
    "\nCheck celltype_col and exact target_celltypes labels."
  )
}

if (is.finite(max_cells) && length(cells_use) > max_cells) {
  cells_use <- sample(cells_use, max_cells)
  message("Final cells after sampling: ", length(cells_use))
}

# =========================
# 6. 从 Seurat assay 中提取 counts
# =========================

if (is.null(assay_use)) {
  assay_use <- DefaultAssay(obj)
}

message("Assay used: ", assay_use)

if (!assay_use %in% names(obj@assays)) {
  stop("Assay not found in Seurat object: ", assay_use)
}

expr_list <- list()

# Seurat v5 multi-layer path
layer_names <- tryCatch(
  Layers(obj[[assay_use]]),
  error = function(e) character(0)
)

if (length(layer_names) > 0) {
  message("Detected layers:")
  print(layer_names)
  
  count_layers <- layer_names[
    layer_names == "counts" | grepl("^counts\\.", layer_names)
  ]
  
  message("Detected count layers:")
  print(count_layers)
  
  if (length(count_layers) == 0) {
    stop("No counts layers found in assay: ", assay_use)
  }
  
  for (layer_name in count_layers) {
    message("Reading layer: ", layer_name)
    
    mat <- LayerData(
      object = obj,
      assay = assay_use,
      layer = layer_name
    )
    
    layer_cells <- intersect(cells_use, colnames(mat))
    message("  target cells in this layer: ", length(layer_cells))
    
    if (length(layer_cells) == 0) {
      rm(mat)
      gc()
      next
    }
    
    expr_list[[layer_name]] <- mat[, layer_cells, drop = FALSE]
    
    rm(mat)
    gc()
  }
} else {
  # Seurat v4 fallback
  message("No assay layers detected. Falling back to GetAssayData(slot = 'counts').")
  
  mat <- GetAssayData(
    object = obj,
    assay = assay_use,
    slot = "counts"
  )
  
  layer_cells <- intersect(cells_use, colnames(mat))
  
  if (length(layer_cells) == 0) {
    stop("No target cells found in counts matrix.")
  }
  
  expr_list[["counts"]] <- mat[, layer_cells, drop = FALSE]
  
  rm(mat)
  gc()
}

if (length(expr_list) == 0) {
  stop("No target cells found in any counts layer.")
}

# 如果不同 layer 的基因集不完全一致，取交集以保证 cbind 安全
gene_sets <- lapply(expr_list, rownames)
common_genes <- Reduce(intersect, gene_sets)

if (length(common_genes) == 0) {
  stop("No common genes across selected count layers.")
}

if (!all(vapply(gene_sets, function(x) identical(x, common_genes), logical(1)))) {
  warning("Count layers have non-identical gene sets/order. Using common genes: ", length(common_genes))
  expr_list <- lapply(expr_list, function(x) x[common_genes, , drop = FALSE])
} else {
  # 保持原顺序
  common_genes <- rownames(expr_list[[1]])
}

expr_mat <- Reduce(Matrix::cbind2, expr_list)

rm(expr_list)
gc()

message("Expression matrix dimensions:")
print(dim(expr_mat))

# 去掉重复细胞名
if (anyDuplicated(colnames(expr_mat)) > 0) {
  duplicated_cells <- colnames(expr_mat)[duplicated(colnames(expr_mat))]
  warning("Duplicated cells detected in expr_mat. Removing duplicates: ", length(duplicated_cells))
  expr_mat <- expr_mat[, !duplicated(colnames(expr_mat)), drop = FALSE]
}

# 按 cells_use 顺序重排
valid_cells <- intersect(cells_use, colnames(expr_mat))
expr_mat <- expr_mat[, valid_cells, drop = FALSE]

cell_metadata <- obj@meta.data[colnames(expr_mat), , drop = FALSE]

# 固定 disease 顺序：Control -> COPD -> IPF
cell_metadata[[disease_col]] <- make_factor_if_matching(
  cell_metadata[[disease_col]],
  disease_order
)

gene_metadata <- data.frame(
  gene_short_name = rownames(expr_mat),
  row.names = rownames(expr_mat)
)

stopifnot(identical(colnames(expr_mat), rownames(cell_metadata)))
stopifnot(identical(rownames(expr_mat), rownames(gene_metadata)))

message("Cells used in Monocle3: ", ncol(expr_mat))
message("Genes used in Monocle3: ", nrow(expr_mat))

if (ncol(expr_mat) < 100) {
  stop("Too few cells left after count extraction.")
}

# =========================
# 7. 构建 Monocle3 cds
# =========================

cds <- monocle3::new_cell_data_set(
  expression_data = expr_mat,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)

rm(expr_mat)
gc()

# =========================
# 8. Monocle3 预处理和学习 graph
# =========================

message("Running preprocess_cds...")
cds <- preprocess_cds(cds, num_dim = num_dim)

message("Running reduce_dimension...")
cds <- reduce_dimension(cds, reduction_method = "UMAP")

message("Running cluster_cells...")
cds <- cluster_cells(cds, reduction_method = "UMAP")

message("Running learn_graph...")
message("learn_graph_use_partition = ", learn_graph_use_partition)
cds <- learn_graph(cds, use_partition = learn_graph_use_partition)

# =========================
# 9. 自动选择 root cells
# =========================
# 原则：
#   1) root 来自 axis_v2_score 最低的一批细胞
#   2) 如果过少，则取 axis_v2_score 最低的前 min_root_cells 个细胞
# =========================

meta <- as.data.frame(colData(cds))

if (!axis_score_col %in% colnames(meta)) {
  stop("Missing axis score column in cds colData: ", axis_score_col)
}

axis_score <- as.numeric(meta[[axis_score_col]])

if (all(is.na(axis_score))) {
  stop("All axis scores are NA; cannot select root cells.")
}

root_cutoff <- as.numeric(quantile(axis_score, root_quantile, na.rm = TRUE))

root_candidates <- rownames(meta)[
  axis_score <= root_cutoff &
    !is.na(axis_score)
]

if (length(root_candidates) < min_root_cells) {
  ordered_cells <- rownames(meta)[order(axis_score, decreasing = FALSE, na.last = NA)]
  root_candidates <- ordered_cells[seq_len(min(min_root_cells, length(ordered_cells)))]
}

root_candidates <- intersect(root_candidates, colnames(cds))

message("Root quantile: ", root_quantile)
message("Root axis score cutoff: ", root_cutoff)
message("Number of root cells: ", length(root_candidates))

if (length(root_candidates) == 0) {
  stop("No root candidates found.")
}

message("Root cell type composition:")
print(table(meta[root_candidates, celltype_col], useNA = "ifany"))

message("Root disease composition:")
print(table(meta[root_candidates, disease_col], useNA = "ifany"))

cds <- order_cells(cds, root_cells = root_candidates)

# =========================
# 10. 提取 pseudotime
# =========================

pt <- pseudotime(cds)

result_df <- as.data.frame(colData(cds))
result_df$cell_id <- rownames(result_df)
result_df$pseudotime <- as.numeric(pt[result_df$cell_id])

result_df$pseudotime_finite <- finite_or_na(result_df$pseudotime)

# 再次固定 disease 顺序，保证输出和图一致
result_df[[disease_col]] <- make_factor_if_matching(
  result_df[[disease_col]],
  disease_order
)

message("Pseudotime summary:")
print(summary(result_df$pseudotime))

n_finite <- sum(is.finite(result_df$pseudotime))
n_inf <- sum(is.infinite(result_df$pseudotime))
n_na <- sum(is.na(result_df$pseudotime))

message("Finite pseudotime cells: ", n_finite)
message("Inf pseudotime cells: ", n_inf)
message("NA pseudotime cells: ", n_na)

if (n_inf > 0) {
  warning(
    "Inf pseudotime still detected. Statistical summaries will use pseudotime_finite only. ",
    "Check graph connectivity or set learn_graph_use_partition = FALSE."
  )
}

# =========================
# 11. 输出 cell-level 结果和诊断
# =========================

cell_level_out <- file.path(
  outdir,
  paste0("pseudotime_cell_level_", target_celltype_name, ".csv")
)

write.csv(result_df, cell_level_out, row.names = FALSE)
message("Saved: ", cell_level_out)

diagnostic_df <- data.frame(
  metric = c(
    "target_celltype_name",
    "celltype_col",
    "target_celltypes",
    "disease_col",
    "assay_use",
    "learn_graph_use_partition",
    "n_cells",
    "n_finite_pseudotime",
    "n_inf_pseudotime",
    "n_na_pseudotime",
    "fraction_inf_pseudotime"
  ),
  value = c(
    target_celltype_name,
    celltype_col,
    paste(target_celltypes, collapse = ";"),
    disease_col,
    assay_use,
    as.character(learn_graph_use_partition),
    as.character(nrow(result_df)),
    as.character(n_finite),
    as.character(n_inf),
    as.character(n_na),
    as.character(n_inf / nrow(result_df))
  )
)

diagnostic_out <- file.path(
  outdir,
  paste0("pseudotime_run_diagnostics_", target_celltype_name, ".csv")
)

write.csv(diagnostic_df, diagnostic_out, row.names = FALSE)
message("Saved: ", diagnostic_out)

# =========================
# 12. 输出 Inf pseudotime 统计
# =========================

inf_summary_stage <- result_df |>
  group_by(.data[[axis_stage_col]]) |>
  summarise(
    n_cells = n(),
    n_finite_pseudotime = sum(is.finite(pseudotime)),
    n_inf_pseudotime = sum(is.infinite(pseudotime)),
    n_na_pseudotime = sum(is.na(pseudotime)),
    fraction_inf_pseudotime = n_inf_pseudotime / n_cells,
    .groups = "drop"
  )

inf_stage_out <- file.path(
  outdir,
  paste0("pseudotime_inf_summary_by_axis_stage_", target_celltype_name, ".csv")
)

write.csv(inf_summary_stage, inf_stage_out, row.names = FALSE)
message("Saved: ", inf_stage_out)

inf_summary_disease <- result_df |>
  group_by(.data[[disease_col]]) |>
  summarise(
    n_cells = n(),
    n_finite_pseudotime = sum(is.finite(pseudotime)),
    n_inf_pseudotime = sum(is.infinite(pseudotime)),
    n_na_pseudotime = sum(is.na(pseudotime)),
    fraction_inf_pseudotime = n_inf_pseudotime / n_cells,
    .groups = "drop"
  )

inf_disease_out <- file.path(
  outdir,
  paste0("pseudotime_inf_summary_by_disease_", target_celltype_name, ".csv")
)

write.csv(inf_summary_disease, inf_disease_out, row.names = FALSE)
message("Saved: ", inf_disease_out)

# =========================
# 13. 统计 pseudotime 与 axis / fibrosis 的相关性
# =========================

result_df_finite <- result_df |>
  filter(
    is.finite(pseudotime),
    !is.na(.data[[axis_score_col]]),
    !is.na(.data[[fibrosis_col]])
  )

message("Cells used for correlation: ", nrow(result_df_finite))

if (nrow(result_df_finite) < 100) {
  stop("Too few cells with finite pseudotime for correlation analysis.")
}

cor_axis <- suppressWarnings(cor.test(
  result_df_finite$pseudotime,
  as.numeric(result_df_finite[[axis_score_col]]),
  method = "spearman",
  use = "complete.obs"
))

cor_fibrosis <- suppressWarnings(cor.test(
  result_df_finite$pseudotime,
  as.numeric(result_df_finite[[fibrosis_col]]),
  method = "spearman",
  use = "complete.obs"
))

cor_df <- data.frame(
  celltype = target_celltype_name,
  variable = c(axis_score_col, fibrosis_col),
  n_cells = c(nrow(result_df_finite), nrow(result_df_finite)),
  spearman_rho = c(
    unname(cor_axis$estimate),
    unname(cor_fibrosis$estimate)
  ),
  p_value = c(
    cor_axis$p.value,
    cor_fibrosis$p.value
  )
)

cor_out <- file.path(
  outdir,
  paste0("pseudotime_axis_correlation_", target_celltype_name, ".csv")
)

write.csv(cor_df, cor_out, row.names = FALSE)
message("Saved: ", cor_out)

# =========================
# 14. 按 axis stage 汇总
# =========================

summary_stage <- result_df |>
  group_by(.data[[axis_stage_col]]) |>
  summarise(
    n_cells = n(),
    n_finite_pseudotime = sum(is.finite(pseudotime)),
    n_inf_pseudotime = sum(is.infinite(pseudotime)),
    n_na_pseudotime = sum(is.na(pseudotime)),
    fraction_inf_pseudotime = n_inf_pseudotime / n_cells,
    mean_pseudotime = mean_finite(pseudotime),
    median_pseudotime = median_finite(pseudotime),
    q25_pseudotime = quantile_finite(pseudotime, 0.25),
    q75_pseudotime = quantile_finite(pseudotime, 0.75),
    mean_axis_v2_score = mean_finite(.data[[axis_score_col]]),
    median_axis_v2_score = median_finite(.data[[axis_score_col]]),
    mean_fibrosis_score = mean_finite(.data[[fibrosis_col]]),
    median_fibrosis_score = median_finite(.data[[fibrosis_col]]),
    .groups = "drop"
  )

stage_out <- file.path(
  outdir,
  paste0("pseudotime_summary_by_axis_stage_", target_celltype_name, ".csv")
)

write.csv(summary_stage, stage_out, row.names = FALSE)
message("Saved: ", stage_out)

# =========================
# 15. 按 disease 汇总
# =========================

summary_disease <- result_df |>
  group_by(.data[[disease_col]]) |>
  summarise(
    n_cells = n(),
    n_finite_pseudotime = sum(is.finite(pseudotime)),
    n_inf_pseudotime = sum(is.infinite(pseudotime)),
    n_na_pseudotime = sum(is.na(pseudotime)),
    fraction_inf_pseudotime = n_inf_pseudotime / n_cells,
    mean_pseudotime = mean_finite(pseudotime),
    median_pseudotime = median_finite(pseudotime),
    q25_pseudotime = quantile_finite(pseudotime, 0.25),
    q75_pseudotime = quantile_finite(pseudotime, 0.75),
    mean_axis_v2_score = mean_finite(.data[[axis_score_col]]),
    median_axis_v2_score = median_finite(.data[[axis_score_col]]),
    mean_fibrosis_score = mean_finite(.data[[fibrosis_col]]),
    median_fibrosis_score = median_finite(.data[[fibrosis_col]]),
    .groups = "drop"
  )

disease_out <- file.path(
  outdir,
  paste0("pseudotime_summary_by_disease_", target_celltype_name, ".csv")
)

write.csv(summary_disease, disease_out, row.names = FALSE)
message("Saved: ", disease_out)

# =========================
# 16. 按 celltype 汇总
# =========================

summary_celltype <- result_df |>
  group_by(.data[[celltype_col]]) |>
  summarise(
    n_cells = n(),
    n_finite_pseudotime = sum(is.finite(pseudotime)),
    n_inf_pseudotime = sum(is.infinite(pseudotime)),
    n_na_pseudotime = sum(is.na(pseudotime)),
    mean_pseudotime = mean_finite(pseudotime),
    median_pseudotime = median_finite(pseudotime),
    mean_axis_v2_score = mean_finite(.data[[axis_score_col]]),
    mean_fibrosis_score = mean_finite(.data[[fibrosis_col]]),
    .groups = "drop"
  )

celltype_out <- file.path(
  outdir,
  paste0("pseudotime_summary_by_celltype_", target_celltype_name, ".csv")
)

write.csv(summary_celltype, celltype_out, row.names = FALSE)
message("Saved: ", celltype_out)

# =========================
# 17. 作图：Monocle3 pseudotime UMAP
# =========================

p1 <- plot_cells(
  cds,
  color_cells_by = "pseudotime",
  label_groups_by_cluster = FALSE,
  label_leaves = FALSE,
  label_branch_points = FALSE
)

p1_out <- file.path(
  outdir,
  paste0("pseudotime_umap_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p1_out,
  plot = p1,
  width = 7,
  height = 6
)

message("Saved: ", p1_out)

# =========================
# 18. 作图：axis stage boxplot
# =========================

p2 <- ggplot(
  result_df,
  aes(
    x = .data[[axis_stage_col]],
    y = pseudotime_finite
  )
) +
  geom_boxplot(outlier.size = 0.3, na.rm = TRUE) +
  theme_bw() +
  labs(
    title = paste0(target_celltype_name, " pseudotime by axis_v2_stage"),
    x = "axis_v2_stage",
    y = "Monocle3 pseudotime"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p2_out <- file.path(
  outdir,
  paste0("pseudotime_axis_stage_boxplot_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p2_out,
  plot = p2,
  width = 8,
  height = 5
)

message("Saved: ", p2_out)

# =========================
# 19. 作图：disease boxplot
# =========================

p3 <- ggplot(
  result_df,
  aes(
    x = .data[[disease_col]],
    y = pseudotime_finite
  )
) +
  geom_boxplot(outlier.size = 0.3, na.rm = TRUE) +
  theme_bw() +
  labs(
    title = paste0(target_celltype_name, " pseudotime by disease"),
    x = "Disease",
    y = "Monocle3 pseudotime"
  )

p3_out <- file.path(
  outdir,
  paste0("pseudotime_disease_boxplot_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p3_out,
  plot = p3,
  width = 6,
  height = 5
)

message("Saved: ", p3_out)

# =========================
# 20. 作图：celltype boxplot
# =========================

p_celltype <- ggplot(
  result_df,
  aes(
    x = .data[[celltype_col]],
    y = pseudotime_finite
  )
) +
  geom_boxplot(outlier.size = 0.3, na.rm = TRUE) +
  theme_bw() +
  labs(
    title = paste0(target_celltype_name, " pseudotime by cell type"),
    x = celltype_col,
    y = "Monocle3 pseudotime"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_celltype_out <- file.path(
  outdir,
  paste0("pseudotime_celltype_boxplot_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p_celltype_out,
  plot = p_celltype,
  width = 7,
  height = 5
)

message("Saved: ", p_celltype_out)

# =========================
# 21. 作图：pseudotime vs axis_v2_score
# =========================

p4 <- ggplot(
  result_df,
  aes(
    x = .data[[axis_score_col]],
    y = pseudotime_finite
  )
) +
  geom_point(size = 0.3, alpha = 0.4, na.rm = TRUE) +
  geom_smooth(method = "loess", se = TRUE, na.rm = TRUE) +
  theme_bw() +
  labs(
    title = paste0(target_celltype_name, " pseudotime vs axis_v2_score"),
    x = axis_score_col,
    y = "Monocle3 pseudotime"
  )

p4_out <- file.path(
  outdir,
  paste0("pseudotime_vs_axis_v2_score_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p4_out,
  plot = p4,
  width = 6,
  height = 5
)

message("Saved: ", p4_out)

# =========================
# 22. 作图：pseudotime vs fibrosis score
# =========================

p5 <- ggplot(
  result_df,
  aes(
    x = .data[[fibrosis_col]],
    y = pseudotime_finite
  )
) +
  geom_point(size = 0.3, alpha = 0.4, na.rm = TRUE) +
  geom_smooth(method = "loess", se = TRUE, na.rm = TRUE) +
  theme_bw() +
  labs(
    title = paste0(target_celltype_name, " pseudotime vs fibrosis score"),
    x = fibrosis_col,
    y = "Monocle3 pseudotime"
  )

p5_out <- file.path(
  outdir,
  paste0("pseudotime_vs_fibrosis_score_", target_celltype_name, ".pdf")
)

ggsave(
  filename = p5_out,
  plot = p5,
  width = 6,
  height = 5
)

message("Saved: ", p5_out)

# =========================
# 23. 保存 cds 对象
# =========================

cds_out <- file.path(
  outdir,
  paste0("monocle3_cds_", target_celltype_name, ".rds")
)

saveRDS(cds, cds_out)
message("Saved: ", cds_out)

# =========================
# 24. 保存 session info
# =========================

session_out <- file.path(
  outdir,
  paste0("sessionInfo_", target_celltype_name, ".txt")
)

sink(session_out)
print(sessionInfo())
sink()

message("Saved: ", session_out)

message("Done.")
