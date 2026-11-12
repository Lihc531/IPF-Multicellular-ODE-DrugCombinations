# 17_run_ppi_informed_key_driver_with_manual_labels.R
# Run formal PPI-informed key driver identification using manually reviewed module labels
#
# Required input:
#   results/ppi_driver_preparation/tables/05_manual_module_label_template.csv
#   results/ppi_driver_preparation/tables/06_gene_priority_before_ppi.csv
#   results/ppi_driver_preparation/tables/07_cytoscape_node_table_dynamic_genes.csv
#   data/string_ppi_edges_700.tsv
#
# Required columns in manual label file:
#   celltype
#   dynamic_module
#   suggested_manual_biological_label
#   suggested_keep_for_ppi
#   suggested_notes

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tibble)
  library(ggplot2)
  library(igraph)
  library(stringr)
})

set.seed(123)

project_dir <- getwd()

# ============================================================
# 0. Input / output paths
# ============================================================

manual_label_file <- file.path(
  project_dir,
  "results",
  "ppi_driver_preparation",
  "tables",
  "05_manual_module_label_template.csv"
)

gene_priority_file <- file.path(
  project_dir,
  "results",
  "ppi_driver_preparation",
  "tables",
  "06_gene_priority_before_ppi.csv"
)

cytoscape_node_file <- file.path(
  project_dir,
  "results",
  "ppi_driver_preparation",
  "tables",
  "07_cytoscape_node_table_dynamic_genes.csv"
)

ppi_edge_file <- file.path(
  project_dir,
  "string_ppi_edges_700.tsv"
)

out_dir <- file.path(
  project_dir,
  "results",
  "ppi_key_driver_final"
)

tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")
cyto_dir <- file.path(out_dir, "cytoscape")
rds_dir <- file.path(out_dir, "rds")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cyto_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rds_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Parameters
# ============================================================

# PPI network threshold is already encoded in your edge file name.
ppi_network_label <- "STRING_combined_score_700"

# Remove genes unlikely to be usable in STRING PPI.
remove_non_protein_like_genes <- TRUE

# Minimum PPI degree for "network-supported" flag.
min_ppi_degree_supported <- 1

# Top output limits
top_n_drivers_per_celltype <- 50
top_n_drivers_per_module <- 30
top_n_global_drivers <- 100

# Driver score weights
w_dynamic <- 0.30
w_trend <- 0.15
w_module_priority <- 0.15
w_ppi_centrality <- 0.30
w_module_retention <- 0.10

# Centrality score weights
w_degree <- 0.35
w_betweenness <- 0.25
w_closeness <- 0.15
w_eigenvector <- 0.25

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

clean_yes_no <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x[x %in% c("y", "yes", "true", "1", "keep")] <- "yes"
  x[x %in% c("n", "no", "false", "0", "remove")] <- "no"
  x[is.na(x) | x == ""] <- "review"
  x
}

is_non_protein_like_gene <- function(gene) {
  grepl("^ENSG", gene) |
    grepl("^LINC", gene) |
    grepl("MIR.*HG", gene) |
    grepl("-AS[0-9]*$", gene) |
    grepl("^TTTY", gene) |
    grepl("P[0-9]+$", gene) & grepl("P$", gene)
}

safe_col <- function(df, col, default = NA) {
  if (col %in% colnames(df)) {
    df[[col]]
  } else {
    rep(default, nrow(df))
  }
}

# ============================================================
# 3. Load files
# ============================================================

if (!file.exists(manual_label_file)) {
  stop("Manual label file not found: ", manual_label_file)
}

if (!file.exists(gene_priority_file)) {
  stop("Gene priority file not found: ", gene_priority_file)
}

if (!file.exists(cytoscape_node_file)) {
  stop("Cytoscape node file not found: ", cytoscape_node_file)
}

if (!file.exists(ppi_edge_file)) {
  stop("PPI edge file not found: ", ppi_edge_file)
}

manual_labels <- read.csv(manual_label_file, stringsAsFactors = FALSE)
gene_priority <- read.csv(gene_priority_file, stringsAsFactors = FALSE)
cyto_nodes_pre <- read.csv(cytoscape_node_file, stringsAsFactors = FALSE)
ppi_edges_raw <- read.delim(ppi_edge_file, stringsAsFactors = FALSE)

message("Manual labels:")
print(dim(manual_labels))

message("Gene priority:")
print(dim(gene_priority))

message("PPI edges raw:")
print(dim(ppi_edges_raw))

# ============================================================
# 4. Check / recover manual label columns
# ============================================================

message("Manual label columns:")
print(colnames(manual_labels))

required_base_cols <- c("celltype", "dynamic_module")
missing_base <- setdiff(required_base_cols, colnames(manual_labels))

if (length(missing_base) > 0) {
  stop(
    "Missing required base columns in manual label file: ",
    paste(missing_base, collapse = ", ")
  )
}

pick_first_existing_col <- function(df, candidates) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) == 0) {
    return(NULL)
  }
  hits[1]
}

label_col <- pick_first_existing_col(
  manual_labels,
  c(
    "suggested_manual_biological_label",
    "manual_biological_label",
    "manual_biological_label.y",
    "manual_biological_label.x",
    "suggested_manual_biological_label.y",
    "suggested_manual_biological_label.x",
    "auto_biological_label",
    "auto_biological_label.y",
    "auto_biological_label.x"
  )
)

keep_col <- pick_first_existing_col(
  manual_labels,
  c(
    "suggested_keep_for_ppi",
    "keep_for_ppi",
    "keep_for_ppi.y",
    "keep_for_ppi.x",
    "suggested_keep_for_ppi.y",
    "suggested_keep_for_ppi.x"
  )
)

notes_col <- pick_first_existing_col(
  manual_labels,
  c(
    "suggested_notes",
    "notes",
    "notes.y",
    "notes.x",
    "suggested_notes.y",
    "suggested_notes.x"
  )
)

if (is.null(label_col)) {
  stop(
    "Cannot find a usable biological label column. ",
    "Expected one of: suggested_manual_biological_label, manual_biological_label, auto_biological_label."
  )
}

message("Using label column: ", label_col)
message("Using keep column: ", ifelse(is.null(keep_col), "created from n_high_conf", keep_col))
message("Using notes column: ", ifelse(is.null(notes_col), "created as empty", notes_col))

# Create standardized columns explicitly
manual_labels$suggested_manual_biological_label <- as.character(manual_labels[[label_col]])

if (is.null(keep_col)) {
  if ("n_high_conf" %in% colnames(manual_labels)) {
    manual_labels$suggested_keep_for_ppi <- ifelse(manual_labels$n_high_conf >= 10, "yes", "review")
  } else {
    manual_labels$suggested_keep_for_ppi <- "review"
  }
} else {
  manual_labels$suggested_keep_for_ppi <- as.character(manual_labels[[keep_col]])
}

if (is.null(notes_col)) {
  manual_labels$suggested_notes <- ""
} else {
  manual_labels$suggested_notes <- as.character(manual_labels[[notes_col]])
}

# ============================================================
# 5. Clean manual labels
# ============================================================

manual_labels2 <- manual_labels %>%
  mutate(
    celltype = as.character(celltype),
    dynamic_module = as.character(dynamic_module),
    suggested_manual_biological_label = as.character(suggested_manual_biological_label),
    suggested_keep_for_ppi = clean_yes_no(suggested_keep_for_ppi),
    suggested_notes = as.character(suggested_notes)
  ) %>%
  select(
    celltype,
    dynamic_module,
    suggested_manual_biological_label,
    suggested_keep_for_ppi,
    suggested_notes,
    everything()
  )

write.csv(
  manual_labels2,
  file.path(tab_dir, "01_manual_module_labels_cleaned.csv"),
  row.names = FALSE
)

modules_keep <- manual_labels2 %>%
  filter(suggested_keep_for_ppi == "yes") %>%
  select(
    celltype,
    dynamic_module,
    manual_biological_label = suggested_manual_biological_label,
    keep_for_ppi_final = suggested_keep_for_ppi,
    manual_notes = suggested_notes
  )

message("Modules kept for PPI:")
print(nrow(modules_keep))

write.csv(
  modules_keep,
  file.path(tab_dir, "02_modules_kept_for_ppi.csv"),
  row.names = FALSE
)
# ============================================================
# 6. Merge manual labels into gene table
# ============================================================

gene_priority2 <- gene_priority %>%
  mutate(
    gene = as.character(gene),
    celltype = as.character(celltype),
    dynamic_module = as.character(dynamic_module)
  ) %>%
  left_join(
    manual_labels2 %>%
      select(
        celltype,
        dynamic_module,
        manual_biological_label = suggested_manual_biological_label,
        keep_for_ppi_final = suggested_keep_for_ppi,
        manual_notes = suggested_notes
      ),
    by = c("celltype", "dynamic_module")
  ) %>%
  mutate(
    keep_for_ppi_final = clean_yes_no(keep_for_ppi_final),
    manual_biological_label = ifelse(
      is.na(manual_biological_label) | manual_biological_label == "",
      as.character(auto_biological_label),
      manual_biological_label
    ),
    non_protein_like_gene = is_non_protein_like_gene(gene)
  )

gene_for_ppi <- gene_priority2 %>%
  filter(keep_for_ppi_final == "yes")

if (remove_non_protein_like_genes) {
  gene_for_ppi <- gene_for_ppi %>%
    filter(!non_protein_like_gene)
}

gene_for_ppi <- gene_for_ppi %>%
  distinct(celltype, dynamic_module, gene, .keep_all = TRUE)

message("Gene entries kept for PPI:")
print(nrow(gene_for_ppi))

message("Unique genes kept for PPI:")
print(length(unique(gene_for_ppi$gene)))

write.csv(
  gene_priority2,
  file.path(tab_dir, "03_gene_priority_with_manual_labels_all.csv"),
  row.names = FALSE
)

write.csv(
  gene_for_ppi,
  file.path(tab_dir, "04_gene_priority_with_manual_labels_kept_for_ppi.csv"),
  row.names = FALSE
)

writeLines(
  sort(unique(gene_for_ppi$gene)),
  con = file.path(tab_dir, "04_unique_genes_kept_for_ppi.txt")
)

# ============================================================
# 7. Build PPI subnetwork
# ============================================================

ppi_edges <- ppi_edges_raw %>%
  mutate(
    geneA = as.character(geneA),
    geneB = as.character(geneB),
    gene_low = pmin(geneA, geneB),
    gene_high = pmax(geneA, geneB)
  ) %>%
  filter(gene_low != gene_high)

if ("score" %in% colnames(ppi_edges)) {
  ppi_edges$ppi_score <- as.numeric(ppi_edges$score)
} else if ("combined_score" %in% colnames(ppi_edges)) {
  ppi_edges$ppi_score <- as.numeric(ppi_edges$combined_score)
} else {
  ppi_edges$ppi_score <- 1
}

ppi_edges <- ppi_edges %>%
  arrange(desc(ppi_score)) %>%
  distinct(gene_low, gene_high, .keep_all = TRUE) %>%
  transmute(
    geneA = gene_low,
    geneB = gene_high,
    ppi_score = ppi_score
  )

genes_ppi <- sort(unique(gene_for_ppi$gene))

ppi_sub_edges <- ppi_edges %>%
  filter(
    geneA %in% genes_ppi,
    geneB %in% genes_ppi
  )

message("PPI subnetwork edges:")
print(nrow(ppi_sub_edges))

write.csv(
  ppi_sub_edges,
  file.path(tab_dir, "05_ppi_edges_among_manual_kept_dynamic_genes.csv"),
  row.names = FALSE
)

# ============================================================
# 8. Compute PPI centrality
# ============================================================

if (nrow(ppi_sub_edges) == 0) {
  stop("No PPI edges among kept dynamic genes. Consider using score 400 or relaxing module filters.")
}

g <- graph_from_data_frame(
  ppi_sub_edges,
  directed = FALSE,
  vertices = data.frame(name = genes_ppi)
)

E(g)$weight <- ppi_sub_edges$ppi_score

ppi_metrics <- data.frame(
  gene = V(g)$name,
  ppi_degree = degree(g),
  ppi_strength = strength(g, weights = E(g)$weight),
  ppi_betweenness = betweenness(g, normalized = TRUE),
  ppi_closeness = closeness(g, normalized = TRUE),
  ppi_eigenvector = eigen_centrality(g, weights = E(g)$weight)$vector,
  stringsAsFactors = FALSE
)

ppi_metrics <- ppi_metrics %>%
  mutate(
    ppi_degree_scaled = rescale01(ppi_degree),
    ppi_strength_scaled = rescale01(ppi_strength),
    ppi_betweenness_scaled = rescale01(ppi_betweenness),
    ppi_closeness_scaled = rescale01(ppi_closeness),
    ppi_eigenvector_scaled = rescale01(ppi_eigenvector),
    ppi_centrality_score =
      w_degree * ppi_degree_scaled +
      w_betweenness * ppi_betweenness_scaled +
      w_closeness * ppi_closeness_scaled +
      w_eigenvector * ppi_eigenvector_scaled,
    ppi_network_supported = ppi_degree >= min_ppi_degree_supported
  ) %>%
  arrange(desc(ppi_centrality_score))

write.csv(
  ppi_metrics,
  file.path(tab_dir, "06_ppi_centrality_metrics_manual_kept_genes.csv"),
  row.names = FALSE
)

# ============================================================
# 9. Compute final key driver score
# ============================================================

driver_candidates <- gene_for_ppi %>%
  left_join(ppi_metrics, by = "gene") %>%
  mutate(
    ppi_degree = ifelse(is.na(ppi_degree), 0, ppi_degree),
    ppi_strength = ifelse(is.na(ppi_strength), 0, ppi_strength),
    ppi_centrality_score = ifelse(is.na(ppi_centrality_score), 0, ppi_centrality_score),
    ppi_network_supported = ifelse(is.na(ppi_network_supported), FALSE, ppi_network_supported),
    
    dynamic_score_scaled = rescale01(dynamic_score),
    abs_spearman_scaled = rescale01(abs(spearman_axis)),
    module_priority_scaled = rescale01(module_priority_score),
    ppi_centrality_scaled = rescale01(ppi_centrality_score),
    module_retention_score = ifelse(keep_for_ppi_final == "yes", 1, 0),
    
    key_driver_score =
      w_dynamic * dynamic_score_scaled +
      w_trend * abs_spearman_scaled +
      w_module_priority * module_priority_scaled +
      w_ppi_centrality * ppi_centrality_scaled +
      w_module_retention * module_retention_score
  ) %>%
  arrange(desc(key_driver_score))

write.csv(
  driver_candidates,
  file.path(tab_dir, "07_final_ppi_informed_key_driver_candidates.csv"),
  row.names = FALSE
)

# ============================================================
# Extra output:
# Gene-level feature table for downstream modeling
# ============================================================

pick_col <- function(df, candidates, default = NA_real_) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) == 0) {
    return(rep(default, nrow(df)))
  }
  df[[hits[1]]]
}

pick_chr_col <- function(df, candidates, default = NA_character_) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) == 0) {
    return(rep(default, nrow(df)))
  }
  as.character(df[[hits[1]]])
}

# ============================================================
# Extra output:
# Gene-level feature table for downstream modeling
# ============================================================

pick_col <- function(df, candidates, default = NA_real_) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) == 0) {
    return(rep(default, nrow(df)))
  }
  as.numeric(df[[hits[1]]])
}

pick_chr_col <- function(df, candidates, default = NA_character_) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) == 0) {
    return(rep(default, nrow(df)))
  }
  as.character(df[[hits[1]]])
}

dynamic_trend_vec <- pick_col(
  driver_candidates,
  c("dynamic_score", "abs_spearman_axis", "spearman_abs", "trend_score"),
  default = NA_real_
)

if (all(is.na(dynamic_trend_vec)) && "spearman_axis" %in% colnames(driver_candidates)) {
  dynamic_trend_vec <- abs(as.numeric(driver_candidates$spearman_axis))
}

disease_stage_vec <- pick_col(
  driver_candidates,
  c(
    "disease_stage_association_score",
    "stage_association_score",
    "disease_stage_score",
    "stage_score"
  ),
  default = NA_real_
)

if (all(is.na(disease_stage_vec)) && "spearman_axis" %in% colnames(driver_candidates)) {
  disease_stage_vec <- abs(as.numeric(driver_candidates$spearman_axis))
}

if (all(is.na(disease_stage_vec)) && "dynamic_score" %in% colnames(driver_candidates)) {
  disease_stage_vec <- as.numeric(driver_candidates$dynamic_score)
}

ppi_feature_table <- driver_candidates %>%
  mutate(
    gene = as.character(gene),
    
    cell_state_id = ifelse(
      !is.na(manual_biological_label) & manual_biological_label != "",
      as.character(manual_biological_label),
      as.character(dynamic_module)
    ),
    
    celltype_marker_based = as.character(celltype),
    
    cluster_default = pick_chr_col(
      driver_candidates,
      c(
        "cluster_default",
        "cluster",
        "seurat_cluster",
        "cluster_id",
        "default_cluster",
        "dynamic_module"
      ),
      default = NA_character_
    ),
    
    dynamic_trend_score = dynamic_trend_vec,
    
    module_membership_score = pick_col(
      driver_candidates,
      c(
        "module_membership_score",
        "module_score",
        "module_priority_score",
        "ppi_pre_driver_score"
      ),
      default = NA_real_
    ),
    
    avg_expr = pick_col(
      driver_candidates,
      c(
        "avg_expr",
        "avg_expression",
        "mean_expr",
        "mean_expression",
        "avg_log2FC_expr",
        "average_expression"
      ),
      default = NA_real_
    ),
    
    pct_expr = pick_col(
      driver_candidates,
      c(
        "pct_expr",
        "pct_expression",
        "pct.1",
        "pct1",
        "percent_expr",
        "percent_expression"
      ),
      default = NA_real_
    ),
    
    disease_stage_association_score = disease_stage_vec,
    
    patient_consistency_score = pick_col(
      driver_candidates,
      c(
        "patient_consistency_score",
        "patient_consistency",
        "donor_consistency_score",
        "sample_consistency_score",
        "patient_reproducibility_score"
      ),
      default = NA_real_
    )
  ) %>%
  select(
    gene,
    cell_state_id,
    celltype_marker_based,
    cluster_default,
    dynamic_trend_score,
    module_membership_score,
    avg_expr,
    pct_expr,
    disease_stage_association_score,
    patient_consistency_score
  ) %>%
  distinct()

write.csv(
  ppi_feature_table,
  file.path(tab_dir, "13_gene_cell_state_modeling_feature_table.csv"),
  row.names = FALSE
)

message("Extra feature table written:")
message("  ", file.path(tab_dir, "13_gene_cell_state_modeling_feature_table.csv"))

# ============================================================
# 10. Summaries by cell type and module
# ============================================================

top_drivers_by_celltype <- driver_candidates %>%
  group_by(celltype) %>%
  arrange(desc(key_driver_score), .by_group = TRUE) %>%
  slice_head(n = top_n_drivers_per_celltype) %>%
  ungroup()

write.csv(
  top_drivers_by_celltype,
  file.path(tab_dir, "08_top_key_drivers_by_celltype.csv"),
  row.names = FALSE
)

top_drivers_by_module <- driver_candidates %>%
  group_by(celltype, dynamic_module, manual_biological_label) %>%
  arrange(desc(key_driver_score), .by_group = TRUE) %>%
  slice_head(n = top_n_drivers_per_module) %>%
  ungroup()

write.csv(
  top_drivers_by_module,
  file.path(tab_dir, "09_top_key_drivers_by_module.csv"),
  row.names = FALSE
)

module_driver_summary <- driver_candidates %>%
  group_by(celltype, dynamic_module, manual_biological_label) %>%
  summarise(
    n_driver_genes = n(),
    n_ppi_supported = sum(ppi_network_supported, na.rm = TRUE),
    mean_key_driver_score = mean(key_driver_score, na.rm = TRUE),
    max_key_driver_score = max(key_driver_score, na.rm = TRUE),
    mean_ppi_centrality = mean(ppi_centrality_score, na.rm = TRUE),
    top_driver_genes = paste(head(gene[order(-key_driver_score)], 20), collapse = ";"),
    top_ppi_genes = paste(head(gene[order(-ppi_centrality_score)], 20), collapse = ";"),
    .groups = "drop"
  ) %>%
  arrange(desc(max_key_driver_score))

write.csv(
  module_driver_summary,
  file.path(tab_dir, "10_module_level_key_driver_summary.csv"),
  row.names = FALSE
)

celltype_driver_summary <- driver_candidates %>%
  group_by(celltype) %>%
  summarise(
    n_driver_genes = n(),
    n_ppi_supported = sum(ppi_network_supported, na.rm = TRUE),
    mean_key_driver_score = mean(key_driver_score, na.rm = TRUE),
    max_key_driver_score = max(key_driver_score, na.rm = TRUE),
    mean_ppi_centrality = mean(ppi_centrality_score, na.rm = TRUE),
    top_driver_genes = paste(head(gene[order(-key_driver_score)], 25), collapse = ";"),
    .groups = "drop"
  ) %>%
  arrange(desc(max_key_driver_score))

write.csv(
  celltype_driver_summary,
  file.path(tab_dir, "11_celltype_level_key_driver_summary.csv"),
  row.names = FALSE
)

global_top_drivers <- driver_candidates %>%
  arrange(desc(key_driver_score)) %>%
  head(top_n_global_drivers)

write.csv(
  global_top_drivers,
  file.path(tab_dir, "12_global_top_key_drivers.csv"),
  row.names = FALSE
)

# ============================================================
# 11. Cytoscape tables
# ============================================================

cytoscape_nodes <- driver_candidates %>%
  select(
    gene,
    celltype,
    dynamic_module,
    manual_biological_label,
    trend_class,
    spearman_axis,
    fdr,
    dynamic_score,
    dynamic_range,
    module_priority_score,
    ppi_degree,
    ppi_strength,
    ppi_betweenness,
    ppi_closeness,
    ppi_eigenvector,
    ppi_centrality_score,
    key_driver_score,
    ppi_network_supported,
    manual_notes
  ) %>%
  arrange(desc(key_driver_score))

write.csv(
  cytoscape_nodes,
  file.path(cyto_dir, "cytoscape_nodes_final_key_drivers.csv"),
  row.names = FALSE
)

cytoscape_edges <- ppi_sub_edges %>%
  filter(
    geneA %in% cytoscape_nodes$gene,
    geneB %in% cytoscape_nodes$gene
  ) %>%
  rename(
    source = geneA,
    target = geneB,
    interaction_score = ppi_score
  )

write.csv(
  cytoscape_edges,
  file.path(cyto_dir, "cytoscape_edges_final_ppi_subnetwork.csv"),
  row.names = FALSE
)

# Module-specific Cytoscape edge tables
for (mod in sort(unique(driver_candidates$dynamic_module))) {
  genes_mod <- driver_candidates %>%
    filter(dynamic_module == mod) %>%
    pull(gene) %>%
    unique()
  
  edges_mod <- cytoscape_edges %>%
    filter(source %in% genes_mod, target %in% genes_mod)
  
  if (nrow(edges_mod) > 0) {
    safe_mod <- gsub("[^A-Za-z0-9_]+", "_", mod)
    
    write.csv(
      edges_mod,
      file.path(cyto_dir, paste0("module_", safe_mod, "_edges.csv")),
      row.names = FALSE
    )
  }
}

# ============================================================
# 12. Plots
# ============================================================

p <- ggplot(
  celltype_driver_summary,
  aes(
    x = reorder(celltype, max_key_driver_score),
    y = max_key_driver_score
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(
    title = "Maximum PPI-informed key driver score by cell type",
    x = NULL,
    y = "Max key driver score"
  )

ggsave(
  file.path(fig_dir, "01_max_key_driver_score_by_celltype.pdf"),
  p,
  width = 7,
  height = 5,
  limitsize = FALSE
)

top_plot <- global_top_drivers %>%
  head(50)

p <- ggplot(
  top_plot,
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
    title = "Global top PPI-informed key drivers",
    x = NULL,
    y = "Key driver score",
    fill = "Cell type"
  )

ggsave(
  file.path(fig_dir, "02_global_top_key_drivers.pdf"),
  p,
  width = 8,
  height = 10,
  limitsize = FALSE
)

top_module_plot <- module_driver_summary %>%
  head(30)

p <- ggplot(
  top_module_plot,
  aes(
    x = reorder(dynamic_module, max_key_driver_score),
    y = max_key_driver_score,
    fill = celltype
  )
) +
  geom_col(width = 0.75) +
  coord_flip() +
  theme_classic(base_size = 10) +
  labs(
    title = "Top modules by max key driver score",
    x = NULL,
    y = "Max key driver score",
    fill = "Cell type"
  )

ggsave(
  file.path(fig_dir, "03_top_modules_by_key_driver_score.pdf"),
  p,
  width = 9,
  height = 8,
  limitsize = FALSE
)

# Relationship between dynamic score and PPI centrality
p <- ggplot(
  driver_candidates,
  aes(
    x = dynamic_score,
    y = ppi_centrality_score,
    color = celltype
  )
) +
  geom_point(size = 1, alpha = 0.7) +
  theme_classic(base_size = 12) +
  labs(
    title = "Dynamic score vs PPI centrality",
    x = "Dynamic score",
    y = "PPI centrality score",
    color = "Cell type"
  )

ggsave(
  file.path(fig_dir, "04_dynamic_score_vs_ppi_centrality.pdf"),
  p,
  width = 7,
  height = 5,
  limitsize = FALSE
)

# ============================================================
# 13. Save compact object
# ============================================================

final_object <- list(
  manual_labels = manual_labels2,
  modules_keep = modules_keep,
  gene_for_ppi = gene_for_ppi,
  ppi_edges = ppi_sub_edges,
  ppi_metrics = ppi_metrics,
  driver_candidates = driver_candidates,
  top_drivers_by_celltype = top_drivers_by_celltype,
  top_drivers_by_module = top_drivers_by_module,
  module_driver_summary = module_driver_summary,
  celltype_driver_summary = celltype_driver_summary,
  global_top_drivers = global_top_drivers,
  parameters = list(
    ppi_network_label = ppi_network_label,
    ppi_edge_file = ppi_edge_file,
    remove_non_protein_like_genes = remove_non_protein_like_genes,
    w_dynamic = w_dynamic,
    w_trend = w_trend,
    w_module_priority = w_module_priority,
    w_ppi_centrality = w_ppi_centrality,
    w_module_retention = w_module_retention,
    centrality_weights = c(
      degree = w_degree,
      betweenness = w_betweenness,
      closeness = w_closeness,
      eigenvector = w_eigenvector
    )
  )
)

saveRDS(
  final_object,
  file.path(rds_dir, "ppi_informed_key_driver_final_object.rds")
)

message("Done.")
message("Main outputs:")
message("  ", file.path(tab_dir, "07_final_ppi_informed_key_driver_candidates.csv"))
message("  ", file.path(tab_dir, "08_top_key_drivers_by_celltype.csv"))
message("  ", file.path(tab_dir, "09_top_key_drivers_by_module.csv"))
message("  ", file.path(tab_dir, "10_module_level_key_driver_summary.csv"))
message("  ", file.path(tab_dir, "11_celltype_level_key_driver_summary.csv"))
message("  ", file.path(tab_dir, "12_global_top_key_drivers.csv"))
message("  ", file.path(cyto_dir, "cytoscape_nodes_final_key_drivers.csv"))
message("  ", file.path(cyto_dir, "cytoscape_edges_final_ppi_subnetwork.csv"))

driver <- read.csv(
  "results/ppi_key_driver_final/tables/07_final_ppi_informed_key_driver_candidates.csv"
)

head(driver, 30)

celltype_summary <- read.csv(
  "results/ppi_key_driver_final/tables/11_celltype_level_key_driver_summary.csv"
)

celltype_summary

module_summary <- read.csv(
  "results/ppi_key_driver_final/tables/10_module_level_key_driver_summary.csv"
)

head(module_summary, 30)