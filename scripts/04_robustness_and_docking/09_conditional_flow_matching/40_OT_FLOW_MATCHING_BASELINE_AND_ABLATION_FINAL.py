#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_OT_FLOW_MATCHING_BASELINE_AND_ABLATION_FINAL.py
=================================================

用途
----
在脚本29主体实现之后，用一个脚本完成：
1. 规则、距离、Sinkhorn OT和传统机器学习基线；
2. OT耦合、随机配对、最近邻配对的Flow Matching消融；
3. 机制特征组、网络容量和ODE积分消融；
4. 多随机种子重复、均值/标准差/95%CI和Top-K排名稳定性；
5. 自动冻结最佳总体模型与最佳Flow配置，供后续最终筛选脚本读取。

重要边界
--------
所有“阳性”目标仍来自脚本28的非冗余计算优先级A，并非真实组合协同标签。
本脚本评价的是内部计算目标恢复能力与排名稳定性，不能宣称外部生物学性能。

依赖
----
本脚本应与以下文件放在同一目录：
29_OT_COMBINATION_BASELINE_AND_FLOW_MATCHING_FINAL.py
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_VERSION = "30.2.0-FINAL-BASELINES-ABLATION-STABILITY"

# =============================================================================
# PyCharm用户主要修改区
# =============================================================================
INPUT_ROOT = r"E:\PythonProject\results\38_COMBINATION_CANDIDATE_SPACE"
OUTPUT_ROOT = r"E:\PythonProject\results\40_OT_FLOW_MATCHING_ABLATION"
SCRIPT29_PATH = Path(__file__).with_name(
    "39_OT_COMBINATION_BASELINE_AND_FLOW_MATCHING_FINAL.py"
)
EXISTING_OUTPUT_POLICY = "overwrite"  # overwrite | timestamp | error
DEVICE = "auto"
RUN_PROFILE = "standard"              # smoke | standard | full
PRIMARY_SELECTION_METRIC = "average_precision"
TOP_K_VALUES = (10, 20)
# =============================================================================


PROFILE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "seeds": [20260719, 20260720],
        "train_steps": 40,
        "validate_every": 10,
        "early_stopping_patience": 4,
        "train_cache_batches": 12,
        "validation_cache_batches": 4,
        "random_forest_trees": 80,
        "feature_groups": ["query_evidence", "target_complementarity"],
        "structure_experiments": ["hidden64"],
        "ode_experiments": ["euler50"],
    },
    "standard": {
        "seeds": [20260719, 20260720, 20260721, 20260722, 20260723],
        "train_steps": 700,
        "validate_every": 50,
        "early_stopping_patience": 10,
        "train_cache_batches": 100,
        "validation_cache_batches": 20,
        "random_forest_trees": 400,
        "feature_groups": [
            "member_quality",
            "query_evidence",
            "mechanism_union",
            "mechanism_gain",
            "source_robustness",
            "target_complementarity",
            "context_and_conflict",
        ],
        "structure_experiments": ["hidden64", "blocks1", "blocks5"],
        "ode_experiments": ["euler50", "rk4_20", "rk4_100"],
    },
    "full": {
        "seeds": [20260719, 20260720, 20260721, 20260722, 20260723],
        "train_steps": 1500,
        "validate_every": 50,
        "early_stopping_patience": 12,
        "train_cache_batches": 160,
        "validation_cache_batches": 24,
        "random_forest_trees": 800,
        "feature_groups": [
            "member_quality",
            "query_evidence",
            "mechanism_union",
            "mechanism_gain",
            "source_robustness",
            "target_complementarity",
            "context_and_conflict",
        ],
        "structure_experiments": ["hidden64", "hidden256", "blocks1", "blocks5"],
        "ode_experiments": ["euler50", "rk4_20", "rk4_100"],
    },
}


FLOW_EXPERIMENT_DESCRIPTIONS: Dict[str, str] = {
    "cfm_ot_full": "完整特征、minibatch OT耦合、残差向量场",
    "cfm_random_pair": "去除OT耦合，随机源-目标配对",
    "cfm_nearest_pair": "以批次最近邻替代OT一一配对",
    "cfm_hidden64": "完整特征，隐藏维度64",
    "cfm_hidden256": "完整特征，隐藏维度256",
    "cfm_blocks1": "完整特征，1个残差块",
    "cfm_blocks5": "完整特征，5个残差块",
    "cfm_ode_euler50": "完整OT-CFM模型，Euler 50步推断",
    "cfm_ode_rk4_20": "完整OT-CFM模型，RK4 20步推断",
    "cfm_ode_rk4_100": "完整OT-CFM模型，RK4 100步推断",
}


def load_script29(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到脚本29：{path}")
    spec = importlib.util.spec_from_file_location("ot_cfm_script29", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本29：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_output_dir(path: Path, policy: str) -> Path:
    policy = str(policy).strip().casefold()
    if path.exists() and any(path.iterdir()):
        if policy == "overwrite":
            shutil.rmtree(path)
        elif policy == "timestamp":
            path = path.parent / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif policy == "error":
            raise FileExistsError(f"输出目录非空：{path}")
        else:
            raise ValueError(f"未知输出策略：{policy}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rank_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    top_k_values: Sequence[int] = TOP_K_VALUES,
) -> Dict[str, Optional[float]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    output: Dict[str, Optional[float]] = {}
    if len(np.unique(labels)) < 2:
        output["roc_auc"] = None
        output["average_precision"] = None
    else:
        output["roc_auc"] = float(roc_auc_score(labels, scores))
        output["average_precision"] = float(average_precision_score(labels, scores))

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    positives = int(labels.sum())
    positive_positions = np.flatnonzero(ranked_labels == 1)
    output["mrr"] = (
        float(1.0 / (positive_positions[0] + 1))
        if positive_positions.size
        else 0.0
    )

    for k in top_k_values:
        effective_k = min(int(k), len(ranked_labels))
        top = ranked_labels[:effective_k]
        hits = int(top.sum())
        output[f"precision_at_{k}"] = float(hits / effective_k) if effective_k else None
        output[f"recall_at_{k}"] = float(hits / positives) if positives else None
        gains = top / np.log2(np.arange(2, effective_k + 2))
        dcg = float(gains.sum())
        ideal = np.sort(labels)[::-1][:effective_k]
        idcg = float((ideal / np.log2(np.arange(2, effective_k + 2))).sum())
        output[f"ndcg_at_{k}"] = float(dcg / idcg) if idcg > 0 else None
    return output


def append_metric_row(
    rows: List[Dict[str, Any]],
    experiment: str,
    category: str,
    seed: int,
    labels: np.ndarray,
    scores: np.ndarray,
    description: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    row: Dict[str, Any] = {
        "experiment": experiment,
        "category": category,
        "seed": int(seed),
        "description": description,
    }
    row.update(rank_metrics(labels, scores))
    if extra:
        row.update(dict(extra))
    rows.append(row)


def add_pair_scores(
    rows: List[pd.DataFrame],
    valid: pd.DataFrame,
    experiment: str,
    category: str,
    seed: int,
    scores: np.ndarray,
) -> None:
    frame = valid[
        [
            "pair_id", "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name",
            "pair_tier", "is_target_reference", "is_source_distribution",
        ]
    ].copy()
    frame["experiment"] = experiment
    frame["category"] = category
    frame["seed"] = int(seed)
    frame["model_score"] = np.asarray(scores, dtype=float)
    frame["model_rank"] = pd.Series(scores).rank(method="first", ascending=False).astype(int).to_numpy()
    rows.append(frame)


def supervised_baseline_scores(
    model_name: str,
    x_all: np.ndarray,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
    random_forest_trees: int,
) -> np.ndarray:
    if model_name == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=seed,
        )
        model.fit(x_all[train_indices], train_labels)
        return model.predict_proba(x_all)[:, 1]

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(random_forest_trees),
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_all[train_indices], train_labels)
        return model.predict_proba(x_all)[:, 1]

    if model_name == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        )
        positive = max(int(train_labels.sum()), 1)
        negative = max(int((train_labels == 0).sum()), 1)
        weights = np.where(train_labels == 1, negative / positive, 1.0)
        model.fit(x_all[train_indices], train_labels, sample_weight=weights)
        return model.predict_proba(x_all)[:, 1]

    raise KeyError(f"未知监督基线：{model_name}")


def flow_experiment_specs(m29, settings: Mapping[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "name": "cfm_ot_full",
            "category": "flow_primary",
            "exclude_groups": [],
            "config_updates": {"coupling_mode": "ot"},
        },
        {
            "name": "cfm_random_pair",
            "category": "coupling_ablation",
            "exclude_groups": [],
            "config_updates": {"coupling_mode": "random"},
        },
        {
            "name": "cfm_nearest_pair",
            "category": "coupling_ablation",
            "exclude_groups": [],
            "config_updates": {"coupling_mode": "nearest"},
        },
    ]
    for group in settings["feature_groups"]:
        specs.append(
            {
                "name": f"cfm_drop_{group}",
                "category": "feature_ablation",
                "exclude_groups": [group],
                "config_updates": {"coupling_mode": "ot"},
            }
        )
    structure_map = {
        "hidden64": {"hidden_dim": 64},
        "hidden256": {"hidden_dim": 256},
        "blocks1": {"n_hidden_blocks": 1},
        "blocks5": {"n_hidden_blocks": 5},
    }
    for name in settings["structure_experiments"]:
        specs.append(
            {
                "name": f"cfm_{name}",
                "category": "structure_ablation",
                "exclude_groups": [],
                "config_updates": {"coupling_mode": "ot", **structure_map[name]},
            }
        )
    return specs


def flow_config_for_spec(m29, settings: Mapping[str, Any], updates: Mapping[str, Any]):
    config = m29.FlowConfig(
        train_steps=int(settings["train_steps"]),
        validate_every=int(settings["validate_every"]),
        early_stopping_patience=int(settings["early_stopping_patience"]),
        train_cache_batches=int(settings["train_cache_batches"]),
        validation_cache_batches=int(settings["validation_cache_batches"]),
    )
    for key, value in updates.items():
        if not hasattr(config, key):
            raise KeyError(f"FlowConfig没有参数：{key}")
        setattr(config, key, value)
    return config


def score_ode_variant(m29, full_result: Mapping[str, Any], device, config) -> np.ndarray:
    forward_endpoints, forward_length, forward_energy, forward_straightness = m29.integrate_all(
        full_result["model"], full_result["x_all"], device, config, reverse=False
    )
    reverse_endpoints, reverse_length, reverse_energy, reverse_straightness = m29.integrate_all(
        full_result["model"], full_result["x_all"], device, config, reverse=True
    )
    x_all = full_result["x_all"]
    source_train = x_all[full_result["source_train_indices"]]
    target_train = x_all[full_result["target_train_indices"]]
    scores = m29.compute_flow_scores(
        x_all,
        forward_endpoints,
        reverse_endpoints,
        target_train,
        source_train,
        forward_length,
        forward_energy,
        forward_straightness,
        reverse_length,
        reverse_energy,
        reverse_straightness,
    )
    return scores["flow_matching_score"].to_numpy(float)


def summarize_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["experiment", "category", "description"]
    metric_columns = [
        column
        for column in seed_metrics.columns
        if column not in id_columns + ["seed"]
        and pd.api.types.is_numeric_dtype(seed_metrics[column])
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in seed_metrics.groupby(id_columns, dropna=False):
        row = dict(zip(id_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
                row[f"{column}_ci95_low"] = np.nan
                row[f"{column}_ci95_high"] = np.nan
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95_low"] = mean - half
            row[f"{column}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_stability(pair_scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frequency_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    for experiment, group in pair_scores.groupby("experiment"):
        n_seeds = int(group["seed"].nunique())
        pair_group = group.groupby("pair_id", as_index=False).agg(
            drug_a_name=("drug_a_name", "first"),
            drug_b_name=("drug_b_name", "first"),
            pair_tier=("pair_tier", "first"),
            mean_score=("model_score", "mean"),
            score_std=("model_score", "std"),
            mean_rank=("model_rank", "mean"),
            rank_std=("model_rank", "std"),
            top10_count=("model_rank", lambda values: int((values <= 10).sum())),
            top20_count=("model_rank", lambda values: int((values <= 20).sum())),
            top50_count=("model_rank", lambda values: int((values <= 50).sum())),
        )
        pair_group.insert(0, "experiment", experiment)
        pair_group["n_seeds"] = n_seeds
        pair_group["top10_frequency"] = pair_group["top10_count"] / n_seeds
        pair_group["top20_frequency"] = pair_group["top20_count"] / n_seeds
        pair_group["top50_frequency"] = pair_group["top50_count"] / n_seeds
        frequency_rows.extend(pair_group.to_dict("records"))

        pivot = group.pivot_table(
            index="pair_id", columns="seed", values="model_rank", aggfunc="first"
        )
        correlations: List[float] = []
        for seed_a, seed_b in itertools.combinations(pivot.columns.tolist(), 2):
            subset = pivot[[seed_a, seed_b]].dropna()
            if len(subset) > 2:
                rho = spearmanr(subset[seed_a], subset[seed_b]).statistic
                if np.isfinite(rho):
                    correlations.append(float(rho))
        stability_rows.append(
            {
                "experiment": experiment,
                "n_seeds": n_seeds,
                "n_seed_pairs": len(correlations),
                "mean_pairwise_spearman": float(np.mean(correlations)) if correlations else np.nan,
                "std_pairwise_spearman": float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0,
            }
        )
    return pd.DataFrame(frequency_rows), pd.DataFrame(stability_rows)


def select_best_models(summary: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    metric = f"{PRIMARY_SELECTION_METRIC}_mean"
    if metric not in summary.columns:
        raise KeyError(f"汇总结果缺少选择指标：{metric}")
    sortable = summary.copy()
    sortable[metric] = pd.to_numeric(sortable[metric], errors="coerce")
    sortable["ndcg_at_20_mean"] = pd.to_numeric(
        sortable.get("ndcg_at_20_mean", np.nan), errors="coerce"
    )
    sortable["average_precision_std"] = pd.to_numeric(
        sortable.get("average_precision_std", np.nan), errors="coerce"
    )
    sortable = sortable.sort_values(
        [metric, "ndcg_at_20_mean", "average_precision_std"],
        ascending=[False, False, True],
        na_position="last",
    )
    best_overall = sortable.iloc[0]
    # “最佳Flow模型”只在真正重新训练、可保存检查点的CFM实验中选择。
    # ODE推断变体和规则/OT融合分数保留为诊断或决策层结果，不冒充Flow模型。
    flow_categories = {
        "flow_primary", "coupling_ablation", "feature_ablation", "structure_ablation",
    }
    flow = sortable[sortable["category"].isin(flow_categories)]
    best_flow = flow.iloc[0] if not flow.empty else best_overall
    return best_overall, best_flow


def save_plots(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    output_root: Path,
) -> None:
    plot = summary.sort_values("average_precision_mean", ascending=True).tail(20)
    plt.figure(figsize=(10, 8))
    plt.barh(plot["experiment"], plot["average_precision_mean"])
    plt.xlabel("Mean average precision")
    plt.ylabel("Experiment")
    plt.title("Baseline and ablation comparison")
    plt.tight_layout()
    plt.savefig(output_root / "12_baseline_and_ablation_comparison.png", dpi=250, bbox_inches="tight")
    plt.close()

    full_row = summary[summary["experiment"].eq("cfm_ot_full")]
    ablation = summary[summary["category"].isin(
        ["coupling_ablation", "feature_ablation", "structure_ablation", "ode_ablation"]
    )].copy()
    if not full_row.empty and not ablation.empty:
        full_ap = float(full_row.iloc[0]["average_precision_mean"])
        ablation["delta_ap_vs_full"] = ablation["average_precision_mean"] - full_ap
        ablation = ablation.sort_values("delta_ap_vs_full")
        plt.figure(figsize=(10, 7))
        plt.barh(ablation["experiment"], ablation["delta_ap_vs_full"])
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("Delta average precision vs full OT-CFM")
        plt.ylabel("Ablation")
        plt.title("Ablation effects")
        plt.tight_layout()
        plt.savefig(output_root / "13_ablation_effects.png", dpi=250, bbox_inches="tight")
        plt.close()

    if not stability.empty:
        stable_plot = stability.sort_values("mean_pairwise_spearman", ascending=True).tail(20)
        plt.figure(figsize=(10, 8))
        plt.barh(stable_plot["experiment"], stable_plot["mean_pairwise_spearman"])
        plt.xlabel("Mean pairwise Spearman rank correlation")
        plt.ylabel("Experiment")
        plt.title("Ranking stability across seeds")
        plt.tight_layout()
        plt.savefig(output_root / "14_ranking_stability.png", dpi=250, bbox_inches="tight")
        plt.close()


def write_methods(
    path: Path,
    profile: str,
    settings: Mapping[str, Any],
    experiments: pd.DataFrame,
) -> None:
    text = f"""脚本版本：{SCRIPT_VERSION}

评价目标
========
阳性：脚本28的非冗余计算优先级A参考组合。
阴性/源分布：计算优先级B与探索性组合。
所有指标仅代表内部计算目标恢复能力，不代表实验协同预测性能。

重复设计
========
运行配置：{profile}
随机种子：{settings['seeds']}
每个种子保持同一源/目标训练-留出划分用于全部基线和消融。
主要选择指标：{PRIMARY_SELECTION_METRIC}

指标
====
ROC-AUC、Average Precision、MRR、Precision@10/20、Recall@10/20、NDCG@10/20。
跨种子输出均值、标准差和正态近似95%置信区间。
排名稳定性使用各随机种子全组合排名的两两Spearman相关，以及Top10/20/50进入频率。

模型和消融
==========
""" + "\n".join(
        f"- {row.experiment} [{row.category}]：{row.description}"
        for row in experiments.itertuples(index=False)
    ) + """

最佳配置冻结
============
脚本输出09_best_model_configuration.json和10_best_flow_model_checkpoint.pt。
后续最终筛选脚本应直接读取冻结配置，不应根据最终候选名单再次调参。

解释边界
========
1. 监督基线也只是在恢复脚本28的计算标签。
2. 不应把内部AUC/AP写成对真实协同的外部验证。
3. 最终候选仍须进行机制非冗余、安全、暴露、剂量和药物相互作用复核。
"""
    path.write_text(text, encoding="utf-8")


def run_ablation(
    input_root: Path,
    output_root: Path,
    script29_path: Path,
    profile: str,
    device_name: str,
    output_policy: str,
) -> Path:
    if profile not in PROFILE_SETTINGS:
        raise KeyError(f"未知profile：{profile}")
    settings = copy.deepcopy(PROFILE_SETTINGS[profile])
    output_root = choose_output_dir(output_root, output_policy)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    m29 = load_script29(script29_path)
    device = m29.resolve_device(device_name)
    pairs, target_reference = m29.load_pair_inputs(input_root)
    valid = m29.prepare_model_frame(pairs, target_reference)
    all_features = m29.selected_feature_columns()
    ot_config = m29.OTConfig()
    flow_specs = flow_experiment_specs(m29, settings)

    seed_metric_rows: List[Dict[str, Any]] = []
    pair_score_frames: List[pd.DataFrame] = []
    experiment_registry: Dict[str, Dict[str, str]] = {}
    best_artifact_by_experiment: Dict[str, Dict[str, Any]] = {}

    for seed in settings["seeds"]:
        print(f"\n========== seed {seed} ==========")
        full_result: Optional[Dict[str, Any]] = None
        full_valid = valid

        for spec in flow_specs:
            name = spec["name"]
            category = spec["category"]
            description = FLOW_EXPERIMENT_DESCRIPTIONS.get(
                name,
                f"删除特征组：{','.join(spec['exclude_groups'])}",
            )
            experiment_registry[name] = {
                "experiment": name,
                "category": category,
                "description": description,
            }
            feature_columns = m29.selected_feature_columns(
                exclude_groups=spec["exclude_groups"]
            )
            flow_config = flow_config_for_spec(m29, settings, spec["config_updates"])
            print(f"[RUN] {name}: features={len(feature_columns)}")
            result = m29.train_and_score_once(
                valid,
                feature_columns,
                seed,
                device,
                ot_config,
                flow_config,
                verbose=(name == "cfm_ot_full"),
            )
            if name == "cfm_ot_full":
                full_result = result

            target_holdout = result["target_holdout_indices"]
            source_holdout = result["source_holdout_indices"]
            eval_indices = np.concatenate([target_holdout, source_holdout])
            labels = np.concatenate(
                [
                    np.ones(len(target_holdout), dtype=int),
                    np.zeros(len(source_holdout), dtype=int),
                ]
            )
            flow_scores = result["scores"]["flow_matching_score"].to_numpy(float)
            append_metric_row(
                seed_metric_rows,
                name,
                category,
                seed,
                labels,
                flow_scores[eval_indices],
                description,
                {
                    "n_features": len(result["feature_columns"]),
                    "best_validation_loss": result["training_diagnostics"].get("best_validation_loss"),
                },
            )
            add_pair_scores(
                pair_score_frames, valid, name, category, seed, flow_scores
            )

            current_ap = seed_metric_rows[-1].get("average_precision")
            existing = best_artifact_by_experiment.get(name)
            if current_ap is not None and (
                existing is None or float(current_ap) > float(existing["average_precision"])
            ):
                best_artifact_by_experiment[name] = {
                    "average_precision": float(current_ap),
                    "seed": int(seed),
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in result["model"].state_dict().items()
                    },
                    "feature_columns": result["feature_columns"],
                    "imputation_values": result["imputation_values"],
                    "robust_transform": result["transform"].to_jsonable(),
                    "flow_config": asdict(result["flow_config"]),
                    "ot_config": asdict(result["ot_config"]),
                }

        if full_result is None:
            raise RuntimeError("未运行完整OT-CFM实验。")

        # 同一完整模型上的ODE积分消融，无需重复训练。
        ode_map = {
            "euler50": {"ode_method": "euler", "ode_steps": 50},
            "rk4_20": {"ode_method": "rk4", "ode_steps": 20},
            "rk4_100": {"ode_method": "rk4", "ode_steps": 100},
        }
        for ode_name in settings["ode_experiments"]:
            experiment = f"cfm_ode_{ode_name}"
            description = FLOW_EXPERIMENT_DESCRIPTIONS[experiment]
            category = "ode_ablation"
            experiment_registry[experiment] = {
                "experiment": experiment,
                "category": category,
                "description": description,
            }
            ode_config = copy.deepcopy(full_result["flow_config"])
            for key, value in ode_map[ode_name].items():
                setattr(ode_config, key, value)
            scores = score_ode_variant(m29, full_result, device, ode_config)
            eval_indices = np.concatenate(
                [full_result["target_holdout_indices"], full_result["source_holdout_indices"]]
            )
            labels = np.concatenate(
                [
                    np.ones(len(full_result["target_holdout_indices"]), dtype=int),
                    np.zeros(len(full_result["source_holdout_indices"]), dtype=int),
                ]
            )
            append_metric_row(
                seed_metric_rows,
                experiment,
                category,
                seed,
                labels,
                scores[eval_indices],
                description,
                {"n_features": len(full_result["feature_columns"])},
            )
            add_pair_scores(pair_score_frames, valid, experiment, category, seed, scores)

        # 基线和评分融合：使用完整模型的相同划分、相同预处理。
        result_scores = full_result["scores"]
        target_holdout = full_result["target_holdout_indices"]
        source_holdout = full_result["source_holdout_indices"]
        eval_indices = np.concatenate([target_holdout, source_holdout])
        labels = np.concatenate(
            [
                np.ones(len(target_holdout), dtype=int),
                np.zeros(len(source_holdout), dtype=int),
            ]
        )

        baseline_defs = [
            ("original_rule_score", "rule_baseline", "脚本28原始组合评分", result_scores["original_pair_candidate_score"].to_numpy(float)),
            ("nearest_target", "distance_baseline", "到训练目标的最近距离基线", result_scores["nearest_target_score"].to_numpy(float)),
            ("target_centroid", "distance_baseline", "到训练目标质心的距离基线", result_scores["target_centroid_score"].to_numpy(float)),
            ("sinkhorn_ot", "ot_baseline", "熵正则Sinkhorn OT基线", result_scores["ot_baseline_score"].to_numpy(float)),
            ("ot_fm_fusion", "score_fusion", "0.45 OT + 0.55 Flow Matching", result_scores["ot_fm_model_score"].to_numpy(float)),
            ("ot_fm_rule_consensus", "score_fusion", "0.70 OT-FM + 0.30脚本28原始评分", result_scores["ot_fm_consensus_score"].to_numpy(float)),
        ]
        for experiment, category, description, scores in baseline_defs:
            experiment_registry[experiment] = {
                "experiment": experiment,
                "category": category,
                "description": description,
            }
            append_metric_row(
                seed_metric_rows,
                experiment,
                category,
                seed,
                labels,
                scores[eval_indices],
                description,
                {"n_features": len(full_result["feature_columns"])},
            )
            add_pair_scores(pair_score_frames, valid, experiment, category, seed, scores)

        train_indices = np.concatenate(
            [full_result["target_train_indices"], full_result["source_train_indices"]]
        )
        train_labels = np.concatenate(
            [
                np.ones(len(full_result["target_train_indices"]), dtype=int),
                np.zeros(len(full_result["source_train_indices"]), dtype=int),
            ]
        )
        for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            description = {
                "logistic_regression": "平衡类权重逻辑回归基线",
                "random_forest": "平衡随机森林基线",
                "gradient_boosting": "样本加权梯度提升基线",
            }[model_name]
            scores = supervised_baseline_scores(
                model_name,
                full_result["x_all"],
                train_indices,
                train_labels,
                seed,
                int(settings["random_forest_trees"]),
            )
            category = "supervised_baseline"
            experiment_registry[model_name] = {
                "experiment": model_name,
                "category": category,
                "description": description,
            }
            append_metric_row(
                seed_metric_rows,
                model_name,
                category,
                seed,
                labels,
                scores[eval_indices],
                description,
                {"n_features": len(full_result["feature_columns"])},
            )
            add_pair_scores(pair_score_frames, valid, model_name, category, seed, scores)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    pair_scores = pd.concat(pair_score_frames, ignore_index=True)
    summary = summarize_metrics(seed_metrics)
    frequency, stability = ranking_stability(pair_scores)
    summary = summary.merge(stability, on="experiment", how="left")
    best_overall, best_flow = select_best_models(summary)

    experiment_table = pd.DataFrame(experiment_registry.values()).sort_values(
        ["category", "experiment"]
    )
    baseline_categories = {
        "rule_baseline", "distance_baseline", "ot_baseline", "supervised_baseline",
        "flow_primary", "score_fusion",
    }
    baseline_summary = summary[summary["category"].isin(baseline_categories)].copy()
    ablation_summary = summary[~summary["category"].isin(
        {"rule_baseline", "distance_baseline", "ot_baseline", "supervised_baseline"}
    )].copy()

    baseline_summary.to_csv(output_root / "01_baseline_performance.csv", index=False)
    ablation_summary.to_csv(output_root / "02_ablation_performance.csv", index=False)
    seed_metrics.to_csv(output_root / "03_seed_level_metrics.csv", index=False)
    summary[summary["category"].eq("feature_ablation")].to_csv(
        output_root / "04_feature_group_ablation.csv", index=False
    )
    summary[summary["category"].eq("structure_ablation")].to_csv(
        output_root / "05_model_structure_ablation.csv", index=False
    )
    stability.to_csv(output_root / "06_ranking_stability.csv", index=False)
    frequency.to_csv(output_root / "07_topk_selection_frequency.csv", index=False)
    pair_scores.to_csv(output_root / "08_pair_scores_all_models_long.csv", index=False)

    best_flow_name = str(best_flow["experiment"])
    best_checkpoint = best_artifact_by_experiment.get(best_flow_name)
    checkpoint_path: Optional[Path] = None
    if best_checkpoint is not None:
        checkpoint_path = output_root / "10_best_flow_model_checkpoint.pt"
        import torch
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "source_script29_version": m29.SCRIPT_VERSION,
                "experiment": best_flow_name,
                **best_checkpoint,
            },
            checkpoint_path,
        )

    config_payload = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "source_script29_version": m29.SCRIPT_VERSION,
        "run_time": datetime.now().isoformat(),
        "profile": profile,
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "best_overall_model": {
            key: (None if pd.isna(value) else value)
            for key, value in best_overall.to_dict().items()
        },
        "best_flow_model": {
            key: (None if pd.isna(value) else value)
            for key, value in best_flow.to_dict().items()
        },
        "best_flow_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "seeds": settings["seeds"],
        "input_files": {
            "pair_space": m29.PAIR_SPACE_FILE,
            "target_reference": m29.TARGET_REFERENCE_FILE,
        },
        "interpretation_boundary": (
            "internal_computational_target_recovery_not_experimental_synergy_validation"
        ),
    }
    (output_root / "09_best_model_configuration.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    qc = {
        "status": "ok",
        "script_version": SCRIPT_VERSION,
        "profile": profile,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "device": str(device),
        "n_pair_rows_input": int(len(pairs)),
        "n_valid_pairs": int(len(valid)),
        "n_target_reference": int(valid["is_target_reference"].sum()),
        "n_source_distribution": int(valid["is_source_distribution"].sum()),
        "n_experiments": int(summary["experiment"].nunique()),
        "n_seeds": int(len(settings["seeds"])),
        "settings": settings,
        "best_overall_experiment": str(best_overall["experiment"]),
        "best_flow_experiment": best_flow_name,
        "interpretation_boundary": config_payload["interpretation_boundary"],
    }
    (output_root / "11_ablation_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_methods(
        output_root / "15_methods_and_ablation_definition.txt",
        profile,
        settings,
        experiment_table,
    )
    experiment_table.to_csv(output_root / "16_experiment_registry.csv", index=False)
    save_plots(summary, stability, output_root)

    print("\n完成。主要结果：")
    print(output_root / "01_baseline_performance.csv")
    print(output_root / "02_ablation_performance.csv")
    print(output_root / "09_best_model_configuration.json")
    print(f"最佳总体模型：{best_overall['experiment']}")
    print(f"最佳Flow模型：{best_flow_name}")
    print("重要：指标只反映脚本28计算目标恢复，不是实验协同外部验证。")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OT-CFM基线、消融和稳定性评估")
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--script29", default=str(SCRIPT29_PATH))
    parser.add_argument("--profile", default=RUN_PROFILE, choices=list(PROFILE_SETTINGS))
    parser.add_argument("--device", default=DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-policy", default=EXISTING_OUTPUT_POLICY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation(
        Path(os.path.expanduser(os.path.expandvars(args.input_root))),
        Path(os.path.expanduser(os.path.expandvars(args.output_root))),
        Path(os.path.expanduser(os.path.expandvars(args.script29))),
        args.profile,
        args.device,
        args.output_policy,
    )


if __name__ == "__main__":
    main()
