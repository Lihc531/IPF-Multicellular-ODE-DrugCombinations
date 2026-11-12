# Integrative Multicellular Molecular State Modeling for IPF Drug Combinations

This repository hosts the end-to-end computational pipeline for multicellular state trajectory modeling, in silico perturbation, and mechanism-guided candidate drug combination prioritization in idiopathic pulmonary fibrosis (IPF).

## Workflow Modules
- **`scripts/01_single_cell_atlas/`**: Preprocessing of GSE136831 and GSE159354, cell-type annotation, and centered log-ratio (CLR) progression axis construction.
- **`scripts/02_cellular_dynamics_and_anchors/`**: Direction-constrained Neural ODE modeling, CellRank/Monocle directional validation, PPI integration, Sinkhorn Optimal Transport state reversal, and scTenifoldNet/CellOracle in silico perturbations.
- **`scripts/03_drug_combination_prioritization/`**: Mechanistic bridging through OmniPath and DGIdb, and deterministic utility increment prioritization.
- **`scripts/04_robustness_and_docking/`**: Neural ODE ablation audits, Conditional Flow Matching benchmark screening, and AutoDock Vina molecular docking.

## System Requirements
- OS: Linux, macOS, or Windows
- Python >= 3.10
- R >= 4.5.2
- Hardware: >= 32 GB RAM recommended

## Quick Start
```bash
# 1. Create and activate Conda environment
conda env create -f environment/environment.yml
conda activate ipf_multicellular_modeling

# 2. Install Python dependencies
pip install -r environment/requirements.txt

# 3. Install R dependencies
Rscript environment/install_R_packages.R
```

## Reproducibility
All stochastic steps utilize deterministic random seed `20260723`. Detailed checksums and manifests are available in `docs/`.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
