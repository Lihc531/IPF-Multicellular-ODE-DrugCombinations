#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
})
source("config.R")

IN_DIR <- file.path(OUTPUT_ROOT, "05_cross_dataset_validation")
IN_FILE <- file.path(IN_DIR, "final_cross_dataset_candidate_ranking.csv")
OUT <- file.path(OUTPUT_ROOT, "06_figures")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(IN_FILE)) stop("Run 06_cross_dataset_validation.R first.")

save_plot <- function(filename, plot, width, height) {
  ggsave(file.path(OUT, paste0(filename, ".pdf")), plot, width = width, height = height)
  ggsave(
    file.path(OUT, paste0(filename, ".png")), plot,
    width = width, height = height, dpi = 300
  )
}

# -----------------------------------------------------------------------------
# 1. Candidate evidence and coverage.
# -----------------------------------------------------------------------------
x <- read.csv(IN_FILE, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "celltype", "gene", "cross_dataset_evidence_score",
  "cross_dataset_evidence_coverage", "cross_dataset_support_status"
)
missing_cols <- setdiff(required, colnames(x))
if (length(missing_cols)) {
  stop("Ranking file lacks revised columns: ", paste(missing_cols, collapse = ", "))
}

x$label <- paste(x$celltype, x$gene, sep = " | ")
x$evaluated <- is.finite(x$cross_dataset_evidence_score)
x$plot_score <- ifelse(x$evaluated, x$cross_dataset_evidence_score, 0)
x <- x[order(x$evaluated, x$plot_score), , drop = FALSE]
x$label <- factor(x$label, levels = x$label)

p <- ggplot(x, aes(y = label)) +
  geom_segment(
    data = subset(x, evaluated),
    aes(x = 0, xend = plot_score, yend = label),
    linewidth = 0.5
  ) +
  geom_point(
    data = subset(x, evaluated),
    aes(x = plot_score, shape = cross_dataset_support_status),
    size = 2.8
  ) +
  geom_text(
    data = subset(x, evaluated),
    aes(
      x = pmin(plot_score + 0.02, 0.96),
      label = sprintf("%.3f; coverage %.0f%%", plot_score, 100 * cross_dataset_evidence_coverage)
    ),
    hjust = 0, size = 3.2
  ) +
  geom_text(
    data = subset(x, !evaluated),
    aes(x = 0.02, label = "Not evaluated"),
    hjust = 0, size = 3.2
  ) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Coverage-adjusted cross-dataset evidence score",
    y = NULL,
    shape = "Evidence coverage",
    title = "Candidate anchors: score and cross-dataset evidence coverage",
    subtitle = "Missing evidence is shown as 'Not evaluated', not as a zero score"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
save_plot("01_candidate_evidence_coverage", p, 9, max(5, 0.42 * nrow(x)))

# -----------------------------------------------------------------------------
# 2. General network perturbation validation, when available.
# -----------------------------------------------------------------------------
net_file <- file.path(IN_DIR, "network_cross_dataset_metrics.csv")
if (file.exists(net_file)) {
  n <- read.csv(net_file, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c(
    "spearman_FC", "top100_jaccard", "perturbation",
    "celltype", "anchor_gene"
  )
  if (nrow(n) && all(required %in% colnames(n))) {
    n$point_label <- paste(n$anchor_gene, n$perturbation, sep = " | ")
    p2 <- ggplot(
      n,
      aes(x = spearman_FC, y = top100_jaccard, shape = perturbation)
    ) +
      geom_vline(xintercept = 0, linetype = 2, linewidth = 0.4) +
      geom_point(size = 2.7) +
      geom_text(
        aes(label = point_label),
        nudge_x = 0.015, nudge_y = 0.005,
        size = 2.6, check_overlap = TRUE
      ) +
      facet_wrap(~celltype, scales = "free") +
      theme_bw(base_size = 10) +
      theme(legend.position = "bottom") +
      labs(
        x = "Main-validation response-magnitude Spearman correlation",
        y = "Top-100 Jaccard",
        shape = "Perturbation",
        title = "General network perturbation: anchor-gene validation"
      )
    save_plot("02_network_validation_concordance", p2, 11, 8)
  }
} else {
  message("Network validation figure skipped: ", net_file, " does not exist.")
}

# -----------------------------------------------------------------------------
# 3. CellOracle TF ranking. Use facet-specific factor levels so that TF ordering
#    in one anchor does not interfere with another anchor.
# -----------------------------------------------------------------------------
tf_file <- file.path(IN_DIR, "celloracle_tf_ranking_within_anchor.csv")
if (file.exists(tf_file)) {
  t <- read.csv(tf_file, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c(
    "celltype", "anchor_gene", "tf_gene",
    "tf_cross_dataset_evidence_score"
  )
  if (nrow(t) && all(required %in% colnames(t))) {
    t$anchor_label <- paste(t$celltype, t$anchor_gene, sep = " | ")
    t <- t[order(
      t$anchor_label, t$tf_cross_dataset_evidence_score,
      decreasing = FALSE
    ), , drop = FALSE]
    t$tf_key <- paste(t$anchor_label, t$tf_gene, sep = "___")
    t$tf_key <- factor(t$tf_key, levels = unique(t$tf_key))
    p3 <- ggplot(
      t,
      aes(x = tf_cross_dataset_evidence_score, y = tf_key)
    ) +
      geom_segment(
        aes(x = 0, xend = tf_cross_dataset_evidence_score, yend = tf_key),
        linewidth = 0.5
      ) +
      geom_point(size = 2.7) +
      geom_text(
        aes(
          x = pmin(tf_cross_dataset_evidence_score + 0.02, 0.96),
          label = sprintf("%.3f", tf_cross_dataset_evidence_score)
        ),
        hjust = 0, size = 3
      ) +
      facet_wrap(~anchor_label, scales = "free_y") +
      scale_y_discrete(labels = function(v) sub("^.*___", "", v)) +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      theme_bw(base_size = 10) +
      labs(
        x = "CellOracle cross-dataset evidence score",
        y = "Upstream TF",
        title = "Upstream TFs mapped to original disease anchors"
      )
    save_plot(
      "03_celloracle_tf_ranking_by_anchor", p3,
      11, max(6, 0.40 * nrow(t))
    )
  }
}

# -----------------------------------------------------------------------------
# 4. CellOracle perturbation concordance.
# -----------------------------------------------------------------------------
co_file <- file.path(IN_DIR, "celloracle_tf_cross_dataset_metrics.csv")
if (file.exists(co_file)) {
  z <- read.csv(co_file, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c(
    "signed_delta_spearman", "top100_jaccard", "condition",
    "perturbation", "tf_gene"
  )
  if (nrow(z) && all(required %in% colnames(z))) {
    p4 <- ggplot(
      z,
      aes(
        x = signed_delta_spearman,
        y = top100_jaccard,
        shape = perturbation
      )
    ) +
      geom_vline(xintercept = 0, linetype = 2, linewidth = 0.4) +
      geom_point(size = 2.7) +
      geom_text(
        aes(label = tf_gene),
        nudge_x = 0.012, nudge_y = 0.003,
        size = 2.8, check_overlap = TRUE
      ) +
      facet_wrap(~condition, scales = "free") +
      theme_bw(base_size = 10) +
      theme(legend.position = "bottom") +
      labs(
        x = "Main-validation signed-delta Spearman correlation",
        y = "Top-100 Jaccard",
        shape = "Perturbation",
        title = "CellOracle TF perturbation concordance"
      )
    save_plot("04_celloracle_validation_concordance", p4, 10, 6)
  }
}

message("Figures written to: ", OUT)
