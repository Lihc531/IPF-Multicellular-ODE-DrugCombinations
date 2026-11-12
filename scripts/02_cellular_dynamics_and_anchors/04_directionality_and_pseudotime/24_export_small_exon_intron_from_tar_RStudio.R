# -*- coding: utf-8 -*-
# ============================================================
# 01_export_small_exon_intron_test_from_tar_RStudio_v2.R
#
# 修正版 v2：
#   解决 obj$umicount$exon / obj$umicount$intron 不是 matrix，而是 list 的问题。
#
# 核心变化：
#   - 自动检查 umicount$exon / umicount$intron 内部结构；
#   - 如果 exon/intron 是 list，会在其中自动寻找第一个可用矩阵；
#   - 输出 matrix_path，方便判断实际取到的是哪一层；
#   - 避免直接 as(list, "dgCMatrix") 报错。
#
# 运行方式：
#   RStudio 中打开本脚本，修改 CONFIG 区域，点击 Source。
# ============================================================


# =========================
# CONFIG：本地测试时主要改这里
# =========================

TAR_PATH <- "E:/Rproject/GSE136831/GSE136831_RAW.tar"

OUT_DIR <- "E:/Rproject/GSE136831/local_test_export"

N_RDS_TO_USE <- 2

MAX_CELLS_PER_RDS <- 1000

RANDOM_SAMPLE_CELLS <- TRUE

SEED <- 123

# 是否打印每个 RDS 的内部结构
PRINT_STRUCTURE <- TRUE

# 如果 exon/intron 内部有多个矩阵，优先选择名字中包含这些关键词的矩阵
PREFERRED_MATRIX_NAME_KEYWORDS <- c("matrix", "counts", "count", "dge", "umi")


# =========================
# CONFIG：正式运行时重点修改这里
# =========================
# 1. TAR_PATH：改为正式机器上的 GSE136831_RAW.tar 路径。
# 2. OUT_DIR：改为正式输出目录。
# 3. N_RDS_TO_USE：
#      本地测试：2
#      正式运行：Inf 或按批次运行。
# 4. MAX_CELLS_PER_RDS：
#      本地测试：1000
#      正式运行：Inf 或按批次设定。
# 5. 正式运行不建议 all cells 混跑，应结合 metadata 按 cell type 分批导出。
# =========================


suppressPackageStartupMessages({
  library(Matrix)
})


safe_message <- function(...) {
  message(paste0(...))
}


resolve_tar_path <- function(tar_path) {
  if (file.exists(tar_path)) {
    return(normalizePath(tar_path, winslash = "/", mustWork = TRUE))
  }
  stop("TAR_PATH does not exist: ", tar_path)
}


read_rds_or_rds_gz <- function(path) {
  if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    ungz_path <- sub("\\.gz$", "", path, ignore.case = TRUE)
    
    con_in <- gzfile(path, "rb")
    con_out <- file(ungz_path, "wb")
    
    repeat {
      buf <- readBin(con_in, what = "raw", n = 1024 * 1024)
      if (length(buf) == 0) break
      writeBin(buf, con_out)
    }
    
    close(con_in)
    close(con_out)
    
    return(readRDS(ungz_path))
  } else {
    return(readRDS(path))
  }
}


extract_member_to_temp <- function(tar_path, member_name) {
  tmpdir <- tempfile("gse136831_extract_")
  dir.create(tmpdir, recursive = TRUE, showWarnings = FALSE)
  
  utils::untar(tar_path, files = member_name, exdir = tmpdir)
  
  expected <- file.path(tmpdir, member_name)
  if (file.exists(expected)) {
    return(list(path = expected, tmpdir = tmpdir))
  }
  
  candidates <- list.files(tmpdir, recursive = TRUE, full.names = TRUE)
  hit <- candidates[basename(candidates) == basename(member_name)]
  
  if (length(hit) == 0) {
    unlink(tmpdir, recursive = TRUE)
    stop("Cannot locate extracted file: ", member_name)
  }
  
  list(path = hit[1], tmpdir = tmpdir)
}


is_matrix_like <- function(x) {
  inherits(x, "Matrix") || is.matrix(x) || inherits(x, "sparseMatrix")
}


object_summary <- function(x) {
  cls <- paste(class(x), collapse = "/")
  dm <- tryCatch(dim(x), error = function(e) NULL)
  nm <- tryCatch(names(x), error = function(e) NULL)
  
  dm_txt <- if (is.null(dm)) "dim=NULL" else paste0("dim=", paste(dm, collapse = "x"))
  nm_txt <- if (is.null(nm)) "names=NULL" else paste0("names=", paste(head(nm, 20), collapse = ","))
  
  paste(cls, dm_txt, nm_txt, sep = " | ")
}


print_nested_structure <- function(x, prefix = "obj", max_depth = 3, depth = 0) {
  indent <- paste(rep("  ", depth), collapse = "")
  safe_message(indent, prefix, ": ", object_summary(x))
  
  if (depth >= max_depth) return(invisible(NULL))
  
  if (is.list(x) && !is.null(names(x))) {
    for (nm in head(names(x), 20)) {
      print_nested_structure(x[[nm]], paste0(prefix, "$", nm), max_depth, depth + 1)
    }
  }
  
  invisible(NULL)
}


find_matrix_candidates <- function(x, path = "x", max_depth = 5, depth = 0) {
  candidates <- list()
  
  if (is_matrix_like(x)) {
    candidates[[length(candidates) + 1]] <- list(path = path, obj = x)
    return(candidates)
  }
  
  if (depth >= max_depth) {
    return(candidates)
  }
  
  if (is.list(x)) {
    nm <- names(x)
    if (is.null(nm)) {
      nm <- as.character(seq_along(x))
    }
    
    for (i in seq_along(x)) {
      child_path <- paste0(path, "$", nm[i])
      child_candidates <- find_matrix_candidates(x[[i]], child_path, max_depth, depth + 1)
      candidates <- c(candidates, child_candidates)
    }
  }
  
  candidates
}


choose_best_matrix <- function(x, label, preferred_keywords = PREFERRED_MATRIX_NAME_KEYWORDS) {
  candidates <- find_matrix_candidates(x, path = label, max_depth = 6)
  
  if (length(candidates) == 0) {
    stop("No matrix-like object found inside: ", label)
  }
  
  candidate_paths <- sapply(candidates, function(z) z$path)
  candidate_dims <- sapply(candidates, function(z) {
    d <- dim(z$obj)
    if (is.null(d)) "NULL" else paste(d, collapse = "x")
  })
  
  safe_message("Matrix candidates for ", label, ":")
  for (i in seq_along(candidates)) {
    safe_message("  ", i, ". ", candidate_paths[i], " | dim=", candidate_dims[i], " | class=", paste(class(candidates[[i]]$obj), collapse = "/"))
  }
  
  lower_paths <- tolower(candidate_paths)
  for (kw in preferred_keywords) {
    hit <- which(grepl(tolower(kw), lower_paths))
    if (length(hit) > 0) {
      idx <- hit[1]
      safe_message("Selected by preferred keyword '", kw, "': ", candidate_paths[idx])
      return(list(path = candidate_paths[idx], mat = candidates[[idx]]$obj))
    }
  }
  
  scores <- sapply(candidates, function(z) {
    m <- z$obj
    if (inherits(m, "Matrix")) {
      return(length(m@x))
    }
    d <- dim(m)
    if (is.null(d)) return(0)
    prod(d)
  })
  
  idx <- which.max(scores)
  safe_message("Selected largest/most-nonzero candidate: ", candidate_paths[idx])
  list(path = candidate_paths[idx], mat = candidates[[idx]]$obj)
}


to_dgCMatrix <- function(x, label) {
  if (inherits(x, "dgCMatrix")) {
    return(x)
  }
  
  if (inherits(x, "Matrix") || inherits(x, "sparseMatrix")) {
    return(as(x, "dgCMatrix"))
  }
  
  if (is.matrix(x)) {
    return(as(Matrix(x, sparse = TRUE), "dgCMatrix"))
  }
  
  stop("Object cannot be converted to dgCMatrix: ", label, " class=", paste(class(x), collapse = "/"))
}


get_matrix_dim_string <- function(x) {
  d <- dim(x)
  if (is.null(d)) return("")
  paste(d, collapse = " x ")
}


make_unique_cell_ids <- function(mat, sample_id) {
  cn <- colnames(mat)
  if (is.null(cn)) {
    cn <- paste0("cell_", seq_len(ncol(mat)))
  }
  paste0(sample_id, "__", cn)
}


standardize_gene_names <- function(mat) {
  rn <- rownames(mat)
  if (is.null(rn)) {
    rn <- paste0("gene_", seq_len(nrow(mat)))
    rownames(mat) <- rn
  }
  mat
}


sample_columns <- function(n, max_n, random_sample = TRUE) {
  if (is.infinite(max_n) || max_n >= n) {
    return(seq_len(n))
  }
  if (random_sample) {
    return(sort(sample(seq_len(n), max_n)))
  }
  seq_len(max_n)
}


combine_sparse_list_by_cbind <- function(x_list) {
  if (length(x_list) == 1) {
    return(x_list[[1]])
  }
  
  out <- x_list[[1]]
  for (i in 2:length(x_list)) {
    out <- Matrix::cbind2(out, x_list[[i]])
  }
  out
}


message("========== Export small exon/intron test matrices v2 ==========")

set.seed(SEED)

tar_path <- resolve_tar_path(TAR_PATH)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

safe_message("Using TAR file: ", tar_path)
safe_message("Output dir: ", OUT_DIR)

tar_files <- utils::untar(tar_path, list = TRUE)
rds_files <- tar_files[grepl("\\.rds(\\.gz)?$", tar_files, ignore.case = TRUE)]

if (length(rds_files) == 0) {
  stop("No .rds or .rds.gz files found inside TAR.")
}

n_use <- if (is.infinite(N_RDS_TO_USE)) length(rds_files) else min(N_RDS_TO_USE, length(rds_files))
rds_files_use <- rds_files[seq_len(n_use)]

safe_message("Total RDS/RDS.GZ files in TAR: ", length(rds_files))
safe_message("RDS/RDS.GZ files to use now: ", length(rds_files_use))

exon_list <- list()
intron_list <- list()
total_list <- list()
cell_info_list <- list()
report_list <- list()
common_genes <- NULL

for (i in seq_along(rds_files_use)) {
  member <- rds_files_use[i]
  sample_id <- sub("\\.dgecounts\\.rds\\.gz$", "", basename(member), ignore.case = TRUE)
  sample_id <- sub("\\.rds\\.gz$", "", sample_id, ignore.case = TRUE)
  sample_id <- sub("\\.rds$", "", sample_id, ignore.case = TRUE)
  
  safe_message("\n--------------------------------------------")
  safe_message("Reading ", i, "/", length(rds_files_use), ": ", member)
  safe_message("Sample id: ", sample_id)
  
  extracted <- extract_member_to_temp(tar_path, member)
  
  tryCatch({
    obj <- read_rds_or_rds_gz(extracted$path)
    
    if (PRINT_STRUCTURE) {
      safe_message("Object nested structure preview:")
      print_nested_structure(obj, prefix = "obj", max_depth = 3)
    }
    
    if (!is.list(obj) || is.null(obj$umicount)) {
      warning("Skipped because object is not expected list with umicount: ", member)
      next
    }
    
    if (is.null(obj$umicount$exon) || is.null(obj$umicount$intron)) {
      warning("Skipped because umicount$exon or umicount$intron is missing: ", member)
      next
    }
    
    exon_pick <- choose_best_matrix(obj$umicount$exon, label = "umicount$exon")
    intron_pick <- choose_best_matrix(obj$umicount$intron, label = "umicount$intron")
    
    exon <- to_dgCMatrix(exon_pick$mat, exon_pick$path)
    intron <- to_dgCMatrix(intron_pick$mat, intron_pick$path)
    
    exon <- standardize_gene_names(exon)
    intron <- standardize_gene_names(intron)
    
    if (!identical(rownames(exon), rownames(intron))) {
      genes_intersect <- intersect(rownames(exon), rownames(intron))
      if (length(genes_intersect) == 0) {
        stop("No overlapping genes between exon and intron matrices.")
      }
      exon <- exon[genes_intersect, , drop = FALSE]
      intron <- intron[genes_intersect, , drop = FALSE]
    }
    
    if (!identical(colnames(exon), colnames(intron))) {
      cells_intersect <- intersect(colnames(exon), colnames(intron))
      if (length(cells_intersect) == 0) {
        stop("No overlapping cells between exon and intron matrices.")
      }
      exon <- exon[, cells_intersect, drop = FALSE]
      intron <- intron[, cells_intersect, drop = FALSE]
    }
    
    keep_cols <- sample_columns(ncol(exon), MAX_CELLS_PER_RDS, RANDOM_SAMPLE_CELLS)
    exon <- exon[, keep_cols, drop = FALSE]
    intron <- intron[, keep_cols, drop = FALSE]
    
    original_barcodes <- colnames(exon)
    unique_cell_ids <- make_unique_cell_ids(exon, sample_id)
    
    colnames(exon) <- unique_cell_ids
    colnames(intron) <- unique_cell_ids
    
    genes_now <- rownames(exon)
    if (is.null(common_genes)) {
      common_genes <- genes_now
    } else {
      common_genes <- intersect(common_genes, genes_now)
    }
    
    exon_list[[sample_id]] <- exon
    intron_list[[sample_id]] <- intron
    total_list[[sample_id]] <- exon + intron
    
    cell_info_list[[sample_id]] <- data.frame(
      cell_id = unique_cell_ids,
      sample_id = sample_id,
      original_barcode = original_barcodes,
      source_rds = member,
      stringsAsFactors = FALSE
    )
    
    report_list[[sample_id]] <- data.frame(
      sample_id = sample_id,
      source_rds = member,
      exon_matrix_path = exon_pick$path,
      intron_matrix_path = intron_pick$path,
      exon_dim = get_matrix_dim_string(exon),
      intron_dim = get_matrix_dim_string(intron),
      n_genes = nrow(exon),
      n_cells_exported = ncol(exon),
      exon_nnz = length(exon@x),
      intron_nnz = length(intron@x),
      stringsAsFactors = FALSE
    )
    
  }, error = function(e) {
    warning("Failed sample ", sample_id, ": ", conditionMessage(e))
  })
  
  unlink(extracted$tmpdir, recursive = TRUE)
}

if (length(exon_list) == 0) {
  stop("No usable exon/intron matrices were exported. Please inspect printed nested structure.")
}

safe_message("\nCommon genes across exported samples: ", length(common_genes))

exon_list <- lapply(exon_list, function(x) x[common_genes, , drop = FALSE])
intron_list <- lapply(intron_list, function(x) x[common_genes, , drop = FALSE])
total_list <- lapply(total_list, function(x) x[common_genes, , drop = FALSE])

spliced_exon <- combine_sparse_list_by_cbind(exon_list)
unspliced_intron <- combine_sparse_list_by_cbind(intron_list)
total_counts <- combine_sparse_list_by_cbind(total_list)

cells <- do.call(rbind, cell_info_list)
report <- do.call(rbind, report_list)

genes <- data.frame(
  gene_id = common_genes,
  gene_name = common_genes,
  stringsAsFactors = FALSE
)

Matrix::writeMM(spliced_exon, file.path(OUT_DIR, "spliced_exon.mtx"))
Matrix::writeMM(unspliced_intron, file.path(OUT_DIR, "unspliced_intron.mtx"))
Matrix::writeMM(total_counts, file.path(OUT_DIR, "total_counts.mtx"))

write.csv(genes, file.path(OUT_DIR, "genes.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(cells, file.path(OUT_DIR, "cells.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(report, file.path(OUT_DIR, "export_report.csv"), row.names = FALSE, fileEncoding = "UTF-8")

safe_message("\n========== Export finished ==========")
safe_message("spliced_exon.mtx: ", file.path(OUT_DIR, "spliced_exon.mtx"))
safe_message("unspliced_intron.mtx: ", file.path(OUT_DIR, "unspliced_intron.mtx"))
safe_message("total_counts.mtx: ", file.path(OUT_DIR, "total_counts.mtx"))
safe_message("genes.csv: ", file.path(OUT_DIR, "genes.csv"))
safe_message("cells.csv: ", file.path(OUT_DIR, "cells.csv"))
safe_message("export_report.csv: ", file.path(OUT_DIR, "export_report.csv"))

safe_message("\nFinal matrix dimensions:")
safe_message("spliced_exon genes x cells: ", paste(dim(spliced_exon), collapse = " x "))
safe_message("unspliced_intron genes x cells: ", paste(dim(unspliced_intron), collapse = " x "))
safe_message("total_counts genes x cells: ", paste(dim(total_counts), collapse = " x "))

safe_message("\nPlease inspect export_report.csv, especially exon_matrix_path and intron_matrix_path.")
safe_message("Next step: run 02_run_local_cellrank_test_pycharm.py.")
