# 12_prepare_dynamic_modules_for_ppi_driver_analysis.R
# Prepare cell-type-specific dynamic modules for PPI-informed key driver identification
#
# Input:
#   results/dynamic_gene_modules_axis_v2/tables/00_combined_dynamic_genes_all_celltypes.csv
#   results/dynamic_gene_modules_axis_v2/tables/00_combined_dynamic_module_gene_assignments.csv
#
# Optional input:
#   data/ppi_edges.tsv
#
# Expected PPI edge file format:
#   geneA    geneB
#   TGFB1    TGFBR1
#   COL1A1   ITGB1
#
# Output:
#   high-confidence dynamic genes
#   module summaries
#   auto biological labels
#   PPI input gene lists
#   optional PPI centrality and driver candidate tables

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tibble)
  library(ggplot2)
  library(stringr)
})

set.seed(123)

project_dir <- getwd()

# ============================================================
# 0. Input / output paths
# ============================================================

dynamic_gene_file <- file.path(
  project_dir,
  "results",
  "dynamic_gene_modules_axis_v2",
  "tables",
  "00_combined_dynamic_genes_all_celltypes.csv"
)

module_assignment_file <- file.path(
  project_dir,
  "results",
  "dynamic_gene_modules_axis_v2",
  "tables",
  "00_combined_dynamic_module_gene_assignments.csv"
)

# Optional PPI edge file
ppi_edge_file <- file.path(
  project_dir,
  "string_ppi_edges_700.tsv"
)

out_dir <- file.path(
  project_dir,
  "results",
  "ppi_driver_preparation"
)

tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")
gene_list_dir <- file.path(out_dir, "gene_lists")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(gene_list_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Parameters
# ============================================================

# High-confidence dynamic gene filters
max_fdr <- 0.05
min_abs_spearman <- 0.50
min_dynamic_score <- 0.15
min_dynamic_range <- 0.10

# Per cell type cap to keep PPI manageable
top_n_genes_per_celltype <- 300

# Per module cap
top_n_genes_per_module <- 100

# Driver score weights
w_dynamic <- 0.35
w_trend <- 0.20
w_module_priority <- 0.20
w_ppi_centrality <- 0.25

# ============================================================
# 2. Helper functions
# ============================================================

rescale01 <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || diff(rng) == 0) {
    return(rep(0.5, length(x)))
  }
  (x - rng[1]) / (rng[2] - rng[1])
}

safe_mean <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

collapse_top_genes <- function(genes, n = 15) {
  genes <- unique(genes)
  genes <- genes[!is.na(genes)]
  paste(head(genes, n), collapse = ";")
}

label_module_by_markers <- function(genes) {
  genes <- unique(as.character(genes))
  
  marker_sets <- list(
    collagen_ECM_deposition = c(
      "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
      "COL6A1", "COL6A2", "COL6A3", "FN1", "SPARC", "LUM", "DCN", "BGN"
    ),
    activated_profibrotic_fibroblast = c(
      "POSTN", "CTHRC1", "CTGF", "TNC", "ASPN", "THBS1",
      "SERPINE1", "COL1A1", "COL3A1", "FN1"
    ),
    myofibroblast_contractile = c(
      "ACTA2", "TAGLN", "MYH11", "CNN1", "TPM2", "MYL9", "RGS5"
    ),
    aberrant_epithelial_injury_remodeling = c(
      "KRT8", "KRT17", "KRT19", "KRT7", "CLDN4", "MUC1",
      "TACSTD2", "S100A2", "MMP7", "ITGA2", "LAMC2"
    ),
    AT2_surfactant_identity = c(
      "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "ABCA3", "NAPSA", "ETV5"
    ),
    AT1_identity = c(
      "AGER", "PDPN", "CAV1", "AQP5", "RTKN2"
    ),
    profibrotic_macrophage = c(
      "SPP1", "TREM2", "APOE", "MERTK", "MARCO",
      "LGALS3", "CHI3L1", "CTSL", "FN1", "CCL18"
    ),
    antigen_presentation_complement = c(
      "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
      "CD74", "B2M", "C1QA", "C1QB", "C1QC"
    ),
    inflammatory_monocyte_chemokine = c(
      "LYZ", "S100A8", "S100A9", "FCN1", "VCAN",
      "IL1B", "CXCL8", "CCL2", "CCL3", "CCL4"
    ),
    endothelial_activation_vascular_remodeling = c(
      "VWF", "PECAM1", "KDR", "FLT1", "ENG", "ACKR1",
      "ICAM1", "VCAM1", "SELE", "ANGPT2", "PLVAP"
    ),
    lymphatic_endothelial = c(
      "CCL21", "PROX1", "LYVE1", "PDPN", "MMRN1", "FLT4"
    ),
    cytotoxic_lymphocyte = c(
      "NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "KLRD1", "KLRK1", "IFNG"
    ),
    T_cell_activation_exhaustion = c(
      "CD3D", "CD3E", "TRAC", "IL7R", "CCR7",
      "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TOX"
    ),
    B_plasma_activation = c(
      "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "XBP1", "IGHG1", "SDC1"
    ),
    mast_activation = c(
      "TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC"
    ),
    hypoxia_oxidative_stress = c(
      "HIF1A", "VEGFA", "SLC2A1", "LDHA", "ENO1", "PGK1",
      "HMOX1", "NQO1", "SOD2", "TXN"
    ),
    interferon_response = c(
      "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
      "OAS1", "OAS2", "STAT1", "IRF7"
    )
  )
  
  scores <- sapply(marker_sets, function(ms) {
    length(intersect(genes, ms))
  })
  
  best <- names(scores)[which.max(scores)]
  
  if (max(scores) == 0) {
    return("unlabeled_dynamic_module")
  }
  
  best
}

infer_module_direction <- function(trend_class_vec, spearman_vec) {
  trend_class_vec <- as.character(trend_class_vec)
  
  n_inc <- sum(trend_class_vec == "increasing", na.rm = TRUE)
  n_dec <- sum(trend_class_vec == "decreasing", na.rm = TRUE)
  n_tr <- sum(trend_class_vec == "transient_peak", na.rm = TRUE)
  
  if (n_inc >= n_dec && n_inc >= n_tr) {
    return("increasing")
  }
  
  if (n_dec >= n_inc && n_dec >= n_tr) {
    return("decreasing")
  }
  
  if (n_tr > 0) {
    return("transient_peak")
  }
  
  if (safe_mean(spearman_vec) > 0) {
    return("weak_increasing")
  }
  
  if (safe_mean(spearman_vec) < 0) {
    return("weak_decreasing")
  }
  
  "mixed"
}

# ============================================================
# 3. Load dynamic gene/module results
# ============================================================

if (!file.exists(dynamic_gene_file)) {
  stop("Dynamic gene file not found: ", dynamic_gene_file)
}

if (!file.exists(module_assignment_file)) {
  stop("Module assignment file not found: ", module_assignment_file)
}

dynamic_genes <- read.csv(dynamic_gene_file, stringsAsFactors = FALSE)
module_assignments <- read.csv(module_assignment_file, stringsAsFactors = FALSE)

message("Dynamic genes:")
print(dim(dynamic_genes))

message("Module assignments:")
print(dim(module_assignments))

required_dynamic_cols <- c(
  "celltype",
  "gene",
  "spearman_axis",
  "fdr",
  "dynamic_score",
  "dynamic_range",
  "trend_class"
)

missing_dynamic_cols <- setdiff(required_dynamic_cols, colnames(dynamic_genes))

if (length(missing_dynamic_cols) > 0) {
  stop("Missing columns in dynamic gene table: ", paste(missing_dynamic_cols, collapse = ", "))
}

required_module_cols <- c(
  "celltype",
  "gene",
  "dynamic_module"
)

missing_module_cols <- setdiff(required_module_cols, colnames(module_assignments))

if (length(missing_module_cols) > 0) {
  stop("Missing columns in module assignment table: ", paste(missing_module_cols, collapse = ", "))
}

# ============================================================
# 4. Clean and merge
# ============================================================

dynamic_genes <- dynamic_genes %>%
  mutate(
    gene = as.character(gene),
    celltype = as.character(celltype),
    trend_class = as.character(trend_class),
    fdr = as.numeric(fdr),
    spearman_axis = as.numeric(spearman_axis),
    dynamic_score = as.numeric(dynamic_score),
    dynamic_range = as.numeric(dynamic_range),
    abs_spearman_axis = abs(spearman_axis),
    direction_score = ifelse(
      trend_class == "increasing",
      1,
      ifelse(trend_class == "decreasing", -1, 0)
    )
  )

module_assignments <- module_assignments %>%
  mutate(
    gene = as.character(gene),
    celltype = as.character(celltype),
    dynamic_module = as.character(dynamic_module)
  )

merged <- dynamic_genes %>%
  left_join(
    module_assignments %>%
      select(celltype, gene, dynamic_module),
    by = c("celltype", "gene")
  )

# Some genes may not be in clustering assignment if module clustering was skipped.
merged$dynamic_module[is.na(merged$dynamic_module)] <- paste0(
  merged$celltype[is.na(merged$dynamic_module)],
  "_unassigned_dynamic"
)

write.csv(
  merged,
  file.path(tab_dir, "01_all_dynamic_genes_with_module_assignment.csv"),
  row.names = FALSE
)

# ============================================================
# 5. High-confidence dynamic genes
# ============================================================

high_conf <- merged %>%
  filter(
    !is.na(fdr),
    !is.na(spearman_axis),
    !is.na(dynamic_score),
    fdr <= max_fdr,
    abs_spearman_axis >= min_abs_spearman,
    dynamic_score >= min_dynamic_score,
    dynamic_range >= min_dynamic_range,
    trend_class %in% c("increasing", "decreasing", "transient_peak")
  ) %>%
  arrange(celltype, desc(dynamic_score))

# Cap per cell type
high_conf_capped <- high_conf %>%
  group_by(celltype) %>%
  arrange(desc(dynamic_score), .by_group = TRUE) %>%
  slice_head(n = top_n_genes_per_celltype) %>%
  ungroup()

write.csv(
  high_conf,
  file.path(tab_dir, "02_high_confidence_dynamic_genes_all.csv"),
  row.names = FALSE
)

write.csv(
  high_conf_capped,
  file.path(tab_dir, "03_high_confidence_dynamic_genes_capped_by_celltype.csv"),
  row.names = FALSE
)

# ============================================================
# 6. Module-level summary and auto labels
# ============================================================

module_summary <- merged %>%
  group_by(celltype, dynamic_module) %>%
  arrange(desc(dynamic_score), .by_group = TRUE) %>%
  summarise(
    n_genes_total = n(),
    n_high_conf = sum(
      fdr <= max_fdr &
        abs_spearman_axis >= min_abs_spearman &
        dynamic_score >= min_dynamic_score &
        dynamic_range >= min_dynamic_range,
      na.rm = TRUE
    ),
    mean_spearman = mean(spearman_axis, na.rm = TRUE),
    median_spearman = median(spearman_axis, na.rm = TRUE),
    mean_dynamic_score = mean(dynamic_score, na.rm = TRUE),
    max_dynamic_score = max(dynamic_score, na.rm = TRUE),
    mean_dynamic_range = mean(dynamic_range, na.rm = TRUE),
    n_increasing = sum(trend_class == "increasing", na.rm = TRUE),
    n_decreasing = sum(trend_class == "decreasing", na.rm = TRUE),
    n_transient_peak = sum(trend_class == "transient_peak", na.rm = TRUE),
    top_genes = collapse_top_genes(gene[order(-dynamic_score)], n = 20),
    .groups = "drop"
  )

module_label_df <- merged %>%
  group_by(celltype, dynamic_module) %>%
  arrange(desc(dynamic_score), .by_group = TRUE) %>%
  summarise(
    auto_biological_label = label_module_by_markers(gene[1:min(100, n())]),
    module_direction = infer_module_direction(trend_class, spearman_axis),
    .groups = "drop"
  )

module_summary <- module_summary %>%
  left_join(
    module_label_df,
    by = c("celltype", "dynamic_module")
  ) %>%
  mutate(
    module_priority_score =
      rescale01(n_high_conf) * 0.30 +
      rescale01(mean_dynamic_score) * 0.30 +
      rescale01(abs(mean_spearman)) * 0.20 +
      rescale01(mean_dynamic_range) * 0.20
  ) %>%
  arrange(desc(module_priority_score))

write.csv(
  module_summary,
  file.path(tab_dir, "04_dynamic_module_summary_with_auto_labels.csv"),
  row.names = FALSE
)

# Also write a manual annotation template
manual_template <- module_summary %>%
  select(
    celltype,
    dynamic_module,
    auto_biological_label,
    module_direction,
    n_genes_total,
    n_high_conf,
    top_genes,
    module_priority_score
  ) %>%
  mutate(
    manual_biological_label = auto_biological_label,
    keep_for_ppi = ifelse(n_high_conf >= 10, "yes", "review"),
    notes = ""
  )

write.csv(
  manual_template,
  file.path(tab_dir, "05_manual_module_label_template.csv"),
  row.names = FALSE
)

# ============================================================
# 7. Gene-level module priority
# ============================================================

gene_priority <- high_conf_capped %>%
  left_join(
    module_summary %>%
      select(celltype, dynamic_module, auto_biological_label, module_direction, module_priority_score),
    by = c("celltype", "dynamic_module")
  ) %>%
  mutate(
    dynamic_score_scaled = rescale01(dynamic_score),
    abs_spearman_scaled = rescale01(abs_spearman_axis),
    module_priority_scaled = rescale01(module_priority_score),
    trend_priority = case_when(
      trend_class == "increasing" ~ 1.0,
      trend_class == "transient_peak" ~ 0.8,
      trend_class == "decreasing" ~ 0.6,
      TRUE ~ 0.4
    ),
    ppi_pre_driver_score =
      0.40 * dynamic_score_scaled +
      0.25 * abs_spearman_scaled +
      0.20 * module_priority_scaled +
      0.15 * trend_priority
  ) %>%
  arrange(celltype, desc(ppi_pre_driver_score))

write.csv(
  gene_priority,
  file.path(tab_dir, "06_gene_priority_before_ppi.csv"),
  row.names = FALSE
)

# ============================================================
# 8. Export PPI input gene lists
# ============================================================

# Global high-confidence genes
global_genes <- sort(unique(gene_priority$gene))

writeLines(
  global_genes,
  con = file.path(gene_list_dir, "all_high_confidence_dynamic_genes_for_ppi.txt")
)

# Cell-type-specific gene lists
for (ct in sort(unique(gene_priority$celltype))) {
  genes_ct <- gene_priority %>%
    filter(celltype == ct) %>%
    arrange(desc(ppi_pre_driver_score)) %>%
    pull(gene) %>%
    unique()
  
  safe_ct <- gsub("[^A-Za-z0-9_]+", "_", ct)
  
  writeLines(
    genes_ct,
    con = file.path(gene_list_dir, paste0(safe_ct, "_high_confidence_dynamic_genes_for_ppi.txt"))
  )
}

# Module-specific gene lists
for (mod in sort(unique(gene_priority$dynamic_module))) {
  genes_mod <- gene_priority %>%
    filter(dynamic_module == mod) %>%
    arrange(desc(ppi_pre_driver_score)) %>%
    pull(gene) %>%
    unique()
  
  safe_mod <- gsub("[^A-Za-z0-9_]+", "_", mod)
  
  writeLines(
    genes_mod,
    con = file.path(gene_list_dir, paste0(safe_mod, "_genes_for_ppi.txt"))
  )
}

# Cytoscape node table
cytoscape_nodes <- gene_priority %>%
  select(
    gene,
    celltype,
    dynamic_module,
    trend_class,
    spearman_axis,
    fdr,
    dynamic_score,
    dynamic_range,
    auto_biological_label,
    module_direction,
    module_priority_score,
    ppi_pre_driver_score
  ) %>%
  arrange(desc(ppi_pre_driver_score))

write.csv(
  cytoscape_nodes,
  file.path(tab_dir, "07_cytoscape_node_table_dynamic_genes.csv"),
  row.names = FALSE
)

# Long table for STRING API or manual upload
string_input <- gene_priority %>%
  select(gene, celltype, dynamic_module, auto_biological_label, ppi_pre_driver_score) %>%
  arrange(celltype, dynamic_module, desc(ppi_pre_driver_score))

write.csv(
  string_input,
  file.path(tab_dir, "08_string_input_gene_table.csv"),
  row.names = FALSE
)

# ============================================================
# 9. Optional PPI network analysis
# ============================================================

ppi_available <- file.exists(ppi_edge_file)

if (ppi_available) {
  message("PPI edge file found: ", ppi_edge_file)
  
  suppressPackageStartupMessages({
    library(igraph)
  })
  
  ppi_edges <- read.delim(ppi_edge_file, stringsAsFactors = FALSE)
  
  if (!all(c("geneA", "geneB") %in% colnames(ppi_edges))) {
    stop("PPI edge file must contain columns: geneA, geneB")
  }
  
  ppi_edges <- ppi_edges %>%
    mutate(
      geneA = as.character(geneA),
      geneB = as.character(geneB)
    ) %>%
    filter(
      geneA %in% global_genes,
      geneB %in% global_genes,
      geneA != geneB
    ) %>%
    distinct(geneA, geneB)
  
  write.csv(
    ppi_edges,
    file.path(tab_dir, "09_ppi_edges_among_high_confidence_dynamic_genes.csv"),
    row.names = FALSE
  )
  
  if (nrow(ppi_edges) > 0) {
    g <- graph_from_data_frame(
      ppi_edges,
      directed = FALSE,
      vertices = data.frame(name = global_genes)
    )
    
    ppi_metrics <- data.frame(
      gene = V(g)$name,
      ppi_degree = degree(g),
      ppi_betweenness = betweenness(g, normalized = TRUE),
      ppi_closeness = closeness(g, normalized = TRUE),
      ppi_eigenvector = eigen_centrality(g)$vector,
      stringsAsFactors = FALSE
    )
    
    ppi_metrics <- ppi_metrics %>%
      mutate(
        ppi_degree_scaled = rescale01(ppi_degree),
        ppi_betweenness_scaled = rescale01(ppi_betweenness),
        ppi_closeness_scaled = rescale01(ppi_closeness),
        ppi_eigenvector_scaled = rescale01(ppi_eigenvector),
        ppi_centrality_score =
          0.35 * ppi_degree_scaled +
          0.25 * ppi_betweenness_scaled +
          0.20 * ppi_closeness_scaled +
          0.20 * ppi_eigenvector_scaled
      )
    
    driver_candidates <- gene_priority %>%
      left_join(ppi_metrics, by = "gene") %>%
      mutate(
        ppi_centrality_score = ifelse(is.na(ppi_centrality_score), 0, ppi_centrality_score),
        key_driver_score =
          w_dynamic * rescale01(dynamic_score) +
          w_trend * rescale01(abs_spearman_axis) +
          w_module_priority * rescale01(module_priority_score) +
          w_ppi_centrality * rescale01(ppi_centrality_score)
      ) %>%
      arrange(desc(key_driver_score))
    
    write.csv(
      ppi_metrics,
      file.path(tab_dir, "10_ppi_centrality_metrics.csv"),
      row.names = FALSE
    )
    
    write.csv(
      driver_candidates,
      file.path(tab_dir, "11_ppi_informed_key_driver_candidates.csv"),
      row.names = FALSE
    )
    
    # Cell-type-specific driver candidates
    driver_by_celltype <- driver_candidates %>%
      group_by(celltype) %>%
      arrange(desc(key_driver_score), .by_group = TRUE) %>%
      slice_head(n = 50) %>%
      ungroup()
    
    write.csv(
      driver_by_celltype,
      file.path(tab_dir, "12_top_key_driver_candidates_by_celltype.csv"),
      row.names = FALSE
    )
    
  } else {
    warning("No PPI edges found among high-confidence dynamic genes.")
  }
  
} else {
  message("No local PPI edge file found.")
  message("Skipping PPI centrality calculation.")
  message("Expected optional file: ", ppi_edge_file)
}

# ============================================================
# 10. Summary plots
# ============================================================

# High confidence genes by cell type
plot_df <- gene_priority %>%
  count(celltype, name = "n_high_conf_genes") %>%
  arrange(desc(n_high_conf_genes))

p <- ggplot(
  plot_df,
  aes(x = reorder(celltype, n_high_conf_genes), y = n_high_conf_genes)
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "High-confidence dynamic genes prepared for PPI",
    x = NULL,
    y = "Number of genes"
  )

ggsave(
  file.path(fig_dir, "01_high_confidence_dynamic_genes_by_celltype.pdf"),
  p,
  width = 7,
  height = 5,
  limitsize = FALSE
)

# Top modules
top_modules <- module_summary %>%
  arrange(desc(module_priority_score)) %>%
  head(30)

p <- ggplot(
  top_modules,
  aes(
    x = reorder(dynamic_module, module_priority_score),
    y = module_priority_score,
    fill = celltype
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 11) +
  labs(
    title = "Top dynamic modules prioritized for PPI",
    x = NULL,
    y = "Module priority score",
    fill = "Cell type"
  )

ggsave(
  file.path(fig_dir, "02_top_dynamic_modules_for_ppi.pdf"),
  p,
  width = 9,
  height = 7,
  limitsize = FALSE
)

# Pre-PPI driver genes
top_genes <- gene_priority %>%
  arrange(desc(ppi_pre_driver_score)) %>%
  head(50)

p <- ggplot(
  top_genes,
  aes(
    x = reorder(gene, ppi_pre_driver_score),
    y = ppi_pre_driver_score,
    fill = celltype
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 10) +
  labs(
    title = "Top pre-PPI driver candidates",
    x = NULL,
    y = "Pre-PPI driver score",
    fill = "Cell type"
  )

ggsave(
  file.path(fig_dir, "03_top_pre_ppi_driver_candidates.pdf"),
  p,
  width = 8,
  height = 10,
  limitsize = FALSE
)

# If PPI driver table exists, plot top key drivers
driver_file <- file.path(tab_dir, "11_ppi_informed_key_driver_candidates.csv")

if (file.exists(driver_file)) {
  driver_candidates <- read.csv(driver_file, stringsAsFactors = FALSE)
  
  top_drivers <- driver_candidates %>%
    arrange(desc(key_driver_score)) %>%
    head(50)
  
  p <- ggplot(
    top_drivers,
    aes(
      x = reorder(gene, key_driver_score),
      y = key_driver_score,
      fill = celltype
    )
  ) +
    geom_col(width = 0.75) +
    coord_flip() +
    theme_classic(base_size = 10) +
    labs(
      title = "Top PPI-informed key driver candidates",
      x = NULL,
      y = "Key driver score",
      fill = "Cell type"
    )
  
  ggsave(
    file.path(fig_dir, "04_top_ppi_informed_key_driver_candidates.pdf"),
    p,
    width = 8,
    height = 10,
    limitsize = FALSE
  )
}

# ============================================================
# 11. Save compact object
# ============================================================

prep_object <- list(
  dynamic_genes_all = merged,
  high_confidence_dynamic_genes = high_conf,
  high_confidence_dynamic_genes_capped = high_conf_capped,
  module_summary = module_summary,
  gene_priority_before_ppi = gene_priority,
  ppi_edge_file = ppi_edge_file,
  ppi_available = ppi_available,
  parameters = list(
    max_fdr = max_fdr,
    min_abs_spearman = min_abs_spearman,
    min_dynamic_score = min_dynamic_score,
    min_dynamic_range = min_dynamic_range,
    top_n_genes_per_celltype = top_n_genes_per_celltype,
    top_n_genes_per_module = top_n_genes_per_module,
    w_dynamic = w_dynamic,
    w_trend = w_trend,
    w_module_priority = w_module_priority,
    w_ppi_centrality = w_ppi_centrality
  )
)

saveRDS(
  prep_object,
  file.path(out_dir, "ppi_driver_preparation_object.rds")
)

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "04_dynamic_module_summary_with_auto_labels.csv"))
message("  ", file.path(tab_dir, "05_manual_module_label_template.csv"))
message("  ", file.path(tab_dir, "06_gene_priority_before_ppi.csv"))
message("  ", file.path(tab_dir, "07_cytoscape_node_table_dynamic_genes.csv"))
message("  ", file.path(gene_list_dir, "all_high_confidence_dynamic_genes_for_ppi.txt"))

if (ppi_available) {
  message("  ", file.path(tab_dir, "11_ppi_informed_key_driver_candidates.csv"))
} else {
  message("No PPI edge file was found, so PPI centrality was skipped.")
  message("Add a PPI file at: ", ppi_edge_file)
  message("Required columns: geneA, geneB")
}