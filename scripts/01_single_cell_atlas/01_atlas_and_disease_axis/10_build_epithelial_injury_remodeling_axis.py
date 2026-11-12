#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_DIR = os.getcwd()

IN_SCORE = "21_sample_level_epithelial_disease_state_scores.csv"
IN_STATE_SUMMARY = "23_epithelial_manual_state_summary_for_interpretation.csv"
IN_STATE_STATS = "12_candidate_IPF_enriched_manual_epithelial_states.csv"
IN_WILCOX = "22_epithelial_remodeling_score_pairwise_wilcox_sample_level.csv"

OUT_DIR = os.path.join(PROJECT_DIR, "results", "epithelial_axis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ============================================================
# 1. Load sample-level epithelial scores
# ============================================================

score = pd.read_csv(IN_SCORE)

required_cols = [
    "sample_id",
    "Disease_Identity",
    "epithelial_remodeling_fraction",
    "epithelial_homeostatic_fraction",
    "epithelial_remodeling_index",
    "epithelial_remodeling_ratio",
    "epithelial_remodeling_log2_ratio",
]

missing = [c for c in required_cols if c not in score.columns]
if missing:
    raise ValueError(f"Missing required columns in {IN_SCORE}: {missing}")

score["Disease_Identity"] = pd.Categorical(
    score["Disease_Identity"],
    categories=["Control", "COPD", "IPF"],
    ordered=True,
)

# ============================================================
# 2. Construct epithelial injury-remodeling axis
# ============================================================

# Core axis metric:
# remodeled fraction - homeostatic fraction
score["epithelial_injury_remodeling_axis_raw"] = score["epithelial_remodeling_index"]

# Ratio-based alternative axis:
score["epithelial_injury_remodeling_axis_log2ratio"] = score[
    "epithelial_remodeling_log2_ratio"
]

# Control-centered z score
control_mean = score.loc[
    score["Disease_Identity"] == "Control",
    "epithelial_injury_remodeling_axis_raw",
].mean()

control_sd = score.loc[
    score["Disease_Identity"] == "Control",
    "epithelial_injury_remodeling_axis_raw",
].std(ddof=1)

score["epithelial_injury_remodeling_axis_z_control"] = (
    score["epithelial_injury_remodeling_axis_raw"] - control_mean
) / control_sd

# Global z score
global_scaler = StandardScaler()
score["epithelial_injury_remodeling_axis_z_global"] = global_scaler.fit_transform(
    score[["epithelial_injury_remodeling_axis_raw"]]
)

# Rank samples along the axis
score = score.sort_values("epithelial_injury_remodeling_axis_raw").reset_index(drop=True)
score["epithelial_axis_rank"] = np.arange(1, len(score) + 1)
score["epithelial_axis_percentile"] = (
    score["epithelial_axis_rank"] - 1
) / (len(score) - 1)

# Tertile-based state assignment
score["epithelial_axis_stage"] = pd.qcut(
    score["epithelial_injury_remodeling_axis_raw"],
    q=3,
    labels=[
        "Axis_low_homeostatic",
        "Axis_intermediate",
        "Axis_high_injury_remodeling",
    ],
)

# Control-centered biological stage
score["epithelial_axis_stage_control_z"] = pd.cut(
    score["epithelial_injury_remodeling_axis_z_control"],
    bins=[-np.inf, 1, 2, np.inf],
    labels=[
        "Control_like_or_low",
        "Moderate_remodeling",
        "High_injury_remodeling",
    ],
)

# Save main axis table
axis_table_path = os.path.join(
    TAB_DIR,
    "24_sample_level_epithelial_injury_remodeling_axis.csv",
)

score.to_csv(axis_table_path, index=False)

# ============================================================
# 3. Disease-level summary
# ============================================================

summary = (
    score.groupby("Disease_Identity", observed=False)
    .agg(
        n_samples=("sample_id", "count"),
        median_axis_raw=("epithelial_injury_remodeling_axis_raw", "median"),
        mean_axis_raw=("epithelial_injury_remodeling_axis_raw", "mean"),
        sd_axis_raw=("epithelial_injury_remodeling_axis_raw", "std"),
        median_axis_log2ratio=(
            "epithelial_injury_remodeling_axis_log2ratio",
            "median",
        ),
        median_axis_z_control=(
            "epithelial_injury_remodeling_axis_z_control",
            "median",
        ),
    )
    .reset_index()
)

summary.to_csv(
    os.path.join(TAB_DIR, "25_epithelial_axis_disease_group_summary.csv"),
    index=False,
)

# ============================================================
# 4. Statistical tests
# ============================================================

groups = {
    g: score.loc[score["Disease_Identity"] == g, "epithelial_injury_remodeling_axis_raw"]
    .dropna()
    .values
    for g in ["Control", "COPD", "IPF"]
}

kw_stat, kw_p = kruskal(groups["Control"], groups["COPD"], groups["IPF"])

pairs = [
    ("IPF", "Control"),
    ("IPF", "COPD"),
    ("COPD", "Control"),
]

pairwise_rows = []

for g1, g2 in pairs:
    x = groups[g1]
    y = groups[g2]

    stat, p = mannwhitneyu(x, y, alternative="two-sided")

    pairwise_rows.append(
        {
            "metric": "epithelial_injury_remodeling_axis_raw",
            "comparison": f"{g1}_vs_{g2}",
            "group_1": g1,
            "group_2": g2,
            "median_group_1": np.median(x),
            "median_group_2": np.median(y),
            "mean_group_1": np.mean(x),
            "mean_group_2": np.mean(y),
            "median_delta": np.median(x) - np.median(y),
            "mean_delta": np.mean(x) - np.mean(y),
            "p_value": p,
        }
    )

pairwise = pd.DataFrame(pairwise_rows)
pairwise["p_adj_BH"] = pairwise["p_value"] * len(pairwise)
pairwise["p_adj_BH"] = pairwise["p_adj_BH"].clip(upper=1)

pairwise.to_csv(
    os.path.join(TAB_DIR, "26_epithelial_axis_pairwise_mannwhitney.csv"),
    index=False,
)

pd.DataFrame(
    [
        {
            "metric": "epithelial_injury_remodeling_axis_raw",
            "test": "Kruskal_Wallis",
            "statistic": kw_stat,
            "p_value": kw_p,
        }
    ]
).to_csv(
    os.path.join(TAB_DIR, "27_epithelial_axis_global_kruskal.csv"),
    index=False,
)

# ============================================================
# 5. Optional PCA axis using remodeled/homeostatic/other fractions
# ============================================================

pca_features = [
    "epithelial_remodeling_fraction",
    "epithelial_homeostatic_fraction",
    "other_epithelial_states",
]

if all(c in score.columns for c in pca_features):
    x = score[pca_features].fillna(0).values
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x_scaled)

    score["epithelial_axis_PC1"] = pcs[:, 0]
    score["epithelial_axis_PC2"] = pcs[:, 1]

    # Orient PC1 so that higher PC1 means more remodeling
    corr, _ = spearmanr(
        score["epithelial_axis_PC1"],
        score["epithelial_injury_remodeling_axis_raw"],
    )

    if corr < 0:
        score["epithelial_axis_PC1"] = -score["epithelial_axis_PC1"]

    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_features,
        columns=["PC1", "PC2"],
    ).reset_index(names="feature")

    pca_loadings.to_csv(
        os.path.join(TAB_DIR, "28_epithelial_axis_pca_loadings.csv"),
        index=False,
    )

    score.to_csv(axis_table_path, index=False)

# ============================================================
# 6. Plots
# ============================================================

colors = {
    "Control": "#F8766D",
    "COPD": "#00BA38",
    "IPF": "#619CFF",
}

# Boxplot: raw axis
plt.figure(figsize=(5.2, 4.2))
data = [
    score.loc[score["Disease_Identity"] == g, "epithelial_injury_remodeling_axis_raw"]
    for g in ["Control", "COPD", "IPF"]
]
plt.boxplot(data, tick_labels=["Control", "COPD", "IPF"], showfliers=False)

for i, g in enumerate(["Control", "COPD", "IPF"], start=1):
    y = score.loc[
        score["Disease_Identity"] == g,
        "epithelial_injury_remodeling_axis_raw",
    ].values
    x = np.random.normal(i, 0.06, size=len(y))
    plt.scatter(x, y, s=12, alpha=0.7)

plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Remodeled fraction - homeostatic fraction")
plt.title("Epithelial injury-remodeling axis")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "24_epithelial_injury_remodeling_axis_boxplot.pdf")
)
plt.close()

# Ranked axis plot
plt.figure(figsize=(8.5, 4.2))

for g in ["Control", "COPD", "IPF"]:
    df = score[score["Disease_Identity"] == g]
    plt.scatter(
        df["epithelial_axis_rank"],
        df["epithelial_injury_remodeling_axis_raw"],
        label=g,
        s=28,
        alpha=0.85,
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Samples ranked along epithelial axis")
plt.ylabel("Epithelial injury-remodeling axis")
plt.title("Ranked epithelial injury-remodeling axis")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "25_ranked_epithelial_injury_remodeling_axis.pdf")
)
plt.close()

# Z-score plot
plt.figure(figsize=(5.2, 4.2))
data = [
    score.loc[score["Disease_Identity"] == g, "epithelial_injury_remodeling_axis_z_control"]
    for g in ["Control", "COPD", "IPF"]
]
plt.boxplot(data, labels=["Control", "COPD", "IPF"], showfliers=False)

for i, g in enumerate(["Control", "COPD", "IPF"], start=1):
    y = score.loc[
        score["Disease_Identity"] == g,
        "epithelial_injury_remodeling_axis_z_control",
    ].values
    x = np.random.normal(i, 0.06, size=len(y))
    plt.scatter(x, y, s=12, alpha=0.7)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(1, linestyle=":", linewidth=1)
plt.axhline(2, linestyle=":", linewidth=1)
plt.ylabel("Control-centered z score")
plt.title("Control-centered epithelial axis")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "26_control_centered_epithelial_axis_zscore.pdf")
)
plt.close()

# Remodeling vs homeostatic scatter
plt.figure(figsize=(5.2, 4.8))

for g in ["Control", "COPD", "IPF"]:
    df = score[score["Disease_Identity"] == g]
    plt.scatter(
        df["epithelial_homeostatic_fraction"],
        df["epithelial_remodeling_fraction"],
        label=g,
        s=30,
        alpha=0.8,
    )

max_lim = max(
    score["epithelial_homeostatic_fraction"].max(),
    score["epithelial_remodeling_fraction"].max(),
)
plt.plot([0, max_lim], [0, max_lim], linestyle="--", linewidth=1)

plt.xlabel("Homeostatic epithelial fraction")
plt.ylabel("Remodeled epithelial fraction")
plt.title("Epithelial remodeling shift")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "27_remodeled_vs_homeostatic_epithelial_fraction.pdf")
)
plt.close()

# PCA plot if available
if "epithelial_axis_PC1" in score.columns:
    plt.figure(figsize=(5.2, 4.8))

    for g in ["Control", "COPD", "IPF"]:
        df = score[score["Disease_Identity"] == g]
        plt.scatter(
            df["epithelial_axis_PC1"],
            df["epithelial_axis_PC2"],
            label=g,
            s=30,
            alpha=0.8,
        )

    plt.xlabel("PC1 epithelial axis")
    plt.ylabel("PC2")
    plt.title("PCA-based epithelial state axis")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "28_pca_based_epithelial_axis.pdf")
    )
    plt.close()

print("Done.")
print(f"Main axis table: {axis_table_path}")
print(f"Figures saved to: {FIG_DIR}")