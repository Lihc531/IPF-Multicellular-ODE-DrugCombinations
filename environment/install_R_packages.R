# R package installation script
pkgs <- c('Seurat', 'Harmony', 'scDblFinder', 'monocle3', 'scTenifoldNet', 
          'OmnipathR', 'clusterProfiler', 'tidyverse')
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = 'https://cloud.r-project.org')
  }
}
