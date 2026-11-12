#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_integrate_cellrank_directionality_all_celltypes_v4.py

功能
----
1. 汇总此前已经完成的 Macrophage、Plasma、T_cell、B_cell、
   Smooth_muscle、NK 等 CellRank/DPT 验证结论。
2. 自动读取 Fibroblast 和 Endothelial CellRank 结果，并进行保守验证。
3. 同时兼容带 `_oriented` 后缀和不带 `_oriented` 后缀的文件名。
4. Endothelial 目录名可包含 Endothelial、Endo 或 vascular。
5. 如果标准文件直接位于 CELLRANK_RESULTS_ROOT 下，只要该目录中存在
   Endothelial/Fibroblast 相关文件名，也可以自动识别。
6. AT1/AT2 保留在统一验证表中，但明确标记为不用于最终方向性证据。
7. 输出完整的 driver 方向性支持、机制解释、结果叙述和全细胞类型验证总表。

运行前可通过环境变量指定路径
----------------------------
DRIVER_TABLE_BASE_DIR
DRIVER_TABLE_OUT_DIR
CELLRANK_RESULTS_ROOT
FIBROBLAST_CELLRANK_DIR
ENDOTHELIAL_CELLRANK_DIR
REQUIRE_FINAL_CELLTYPE_RESULTS

示例
----
PowerShell:

$env:DRIVER_TABLE_BASE_DIR="E:\\PythonProject\\results\\ppi_informed_driver_refinement_layers1_2_v1B1\\tables"
$env:DRIVER_TABLE_OUT_DIR="E:\\PythonProject\\results\\ppi_informed_driver_refinement_layers1_2_v1B1\\tables"
$env:CELLRANK_RESULTS_ROOT="G:\\package\\result\\Pythonresult"

# 如果自动搜索失败，直接指定真实结果目录：
$env:ENDOTHELIAL_CELLRANK_DIR="G:\\package\\result\\Pythonresult\\Endothelial_result"
$env:FIBROBLAST_CELLRANK_DIR="G:\\package\\result\\Pythonresult\\Fibroblast_result"

python E:\\PythonProject\\25_integrate_cellrank_directionality_all_celltypes_v4.py
"""

from pathlib import Path
import os
import re
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 0. CONFIG
# ============================================================

DEFAULT_BASE_DIR = (
    r"E:\PythonProject\results"
    r"\ppi_informed_driver_refinement_layers1_2_v1B1\tables"
)
DEFAULT_CELLRANK_ROOT = r"G:\package\result\Pythonresult"

BASE_DIR = Path(
    os.environ.get("DRIVER_TABLE_BASE_DIR", DEFAULT_BASE_DIR)
).expanduser()

OUT_DIR = Path(
    os.environ.get("DRIVER_TABLE_OUT_DIR", str(BASE_DIR))
).expanduser()

CELLRANK_RESULTS_ROOT = Path(
    os.environ.get("CELLRANK_RESULTS_ROOT", DEFAULT_CELLRANK_ROOT)
).expanduser()

EXPLICIT_CELLRANK_DIRS = {
    "Fibroblast": os.environ.get("FIBROBLAST_CELLRANK_DIR", "").strip(),
    "Endothelial": os.environ.get("ENDOTHELIAL_CELLRANK_DIR", "").strip(),
}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {
        "1", "true", "t", "yes", "y", "on"
    }


REQUIRE_FINAL_CELLTYPE_RESULTS = env_bool(
    "REQUIRE_FINAL_CELLTYPE_RESULTS",
    True,
)

POSITIVE_RHO_THRESHOLD = float(
    os.environ.get("POSITIVE_RHO_THRESHOLD", "0.10")
)
WEAK_POSITIVE_RHO_THRESHOLD = float(
    os.environ.get("WEAK_POSITIVE_RHO_THRESHOLD", "0.00")
)
TERMINAL_LATE_STAGE_MIN = float(
    os.environ.get("TERMINAL_LATE_STAGE_MIN", "0.50")
)
TERMINAL_IPF_MIN = float(
    os.environ.get("TERMINAL_IPF_MIN", "0.50")
)
TERMINAL_TARGET_SUBTYPE_MIN = float(
    os.environ.get("TERMINAL_TARGET_SUBTYPE_MIN", "0.30")
)

FINAL_VALIDATED_CELLTYPES = {
    "Fibroblast",
    "Endothelial",
}

EXCLUDED_CELLRANK_CELLTYPES = {
    "AT1",
    "AT2",
    "ATI",
    "ATII",
    "AT1_AT2",
    "Epithelial",
}


# ============================================================
# 1. Flexible CellRank result filenames
# ============================================================

# 每一类只需要命中其中一个文件名。
# 优先选择带 `_oriented` 的文件；不存在时回退普通文件名。
RESULT_FILE_CANDIDATES = {
    "root": [
        "root_cell_report.csv",
    ],
    "correlation": [
        "pseudotime_spearman_correlations_oriented.csv",
        "pseudotime_spearman_correlations.csv",
    ],
    "disease_composition": [
        "terminal_composition_by_Disease_Identity_oriented.csv",
        "terminal_composition_by_Disease_Identity.csv",
    ],
    "stage_composition": [
        "terminal_composition_by_axis_v2_stage_oriented.csv",
        "terminal_composition_by_axis_v2_stage.csv",
    ],
}

OPTIONAL_SUBTYPE_FILE_CANDIDATES = [
    [
        "terminal_composition_by_Subclass_Cell_Identity_oriented.csv",
        "terminal_composition_by_Subclass_Cell_Identity.csv",
    ],
    [
        "terminal_composition_by_Manuscript_Identity_oriented.csv",
        "terminal_composition_by_Manuscript_Identity.csv",
    ],
    [
        "terminal_composition_by_celltype_marker_based_oriented.csv",
        "terminal_composition_by_celltype_marker_based.csv",
    ],
    [
        "terminal_composition_by_Celltype_Identity_oriented.csv",
        "terminal_composition_by_Celltype_Identity.csv",
    ],
]

CELLTYPE_DIRECTORY_ALIASES = {
    "Fibroblast": [
        "fibroblast",
        "fibro",
    ],
    "Endothelial": [
        "endothelial",
        "endo",
        "vascular",
    ],
}

CELLTYPE_RULES = {
    "Fibroblast": {
        "expected_terminal_terms": [
            "myofibroblast",
            "myofibro",
            "pathological fibroblast",
            "fibrotic fibroblast",
            "cthrc1",
            "col1a1",
            "col3a1",
            "acta2",
        ],
        "forbidden_root_terms": [
            "myofibroblast",
            "myofibro",
            "pathological",
            "fibrotic",
            "cthrc1",
            "acta2",
        ],
        "mechanism_label": (
            "fibroblast_fibrosis_aligned_mesenchymal_progression"
        ),
    },
    "Endothelial": {
        "expected_terminal_terms": [
            "disease-associated",
            "vascular remodeling",
            "activated endothelial",
            "fibrotic endothelial",
            "endothelial",
            "vascular",
            "capillary",
            "arterial",
            "venous",
        ],
        "forbidden_root_terms": [
            "disease-associated",
            "fibrotic",
            "activated",
            "remodeling",
        ],
        "mechanism_label": (
            "endothelial_disease_associated_vascular_remodeling"
        ),
    },
}


# ============================================================
# 2. Previously completed qualitative evidence
# ============================================================

BASE_CELLRANK_EVIDENCE = {
    "Macrophage": {
        "usable": True,
        "support_type": (
            "IPF-associated macrophage/myeloid terminal-like transition"
        ),
        "strength": "moderate",
        "supports_axis": "weak",
        "supports_fibrosis": "no",
        "interpretation": (
            "Macrophage DPT/CellRank identified terminal-like states enriched "
            "in IPF, but not aligned with the fibrosis composite score; this "
            "is best interpreted as an IPF-associated inflammatory/myeloid "
            "transition rather than a direct fibrosis-terminal trajectory."
        ),
    },
    "Plasma": {
        "usable": True,
        "support_type": (
            "IPF-associated plasma/humoral immune remodeling"
        ),
        "strength": "moderate",
        "supports_axis": "weak_positive",
        "supports_fibrosis": "weak_positive",
        "interpretation": (
            "Plasma-cell CellRank/DPT provides supplementary support for an "
            "IPF-associated humoral immune state, with weak positive alignment "
            "to disease-remodeling and fibrosis-related axes."
        ),
    },
    "T_cell": {
        "usable": True,
        "support_type": (
            "IPF-associated T-cell terminal-like immune remodeling"
        ),
        "strength": "moderate",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "T-cell terminal-like states were enriched in IPF, supporting "
            "disease-associated lymphoid immune remodeling rather than a "
            "fibrosis-aligned continuous trajectory."
        ),
    },
    "B_cell": {
        "usable": True,
        "support_type": (
            "B-cell immune remodeling context"
        ),
        "strength": "weak_to_moderate",
        "supports_axis": "weak",
        "supports_fibrosis": "weak",
        "interpretation": (
            "B-cell CellRank/DPT supplies weak-to-moderate contextual support "
            "for antigen-presentation and humoral immune remodeling."
        ),
    },
    "Smooth_muscle": {
        "usable": True,
        "support_type": (
            "smooth-muscle/pericyte mesenchymal context"
        ),
        "strength": "weak",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "Smooth-muscle CellRank support is weak and should be treated as "
            "supplementary mesenchymal context rather than direct trajectory "
            "evidence."
        ),
    },
    "NK": {
        "usable": True,
        "support_type": (
            "NK-cell immune-remodeling context"
        ),
        "strength": "weak",
        "supports_axis": "no",
        "supports_fibrosis": "no",
        "interpretation": (
            "NK-cell directionality evidence is weak and is retained only as "
            "supplementary lymphoid immune-remodeling context."
        ),
    },
    "AT1_AT2": {
        "usable": False,
        "support_type": (
            "AT1/AT2 CellRank trajectory not accepted as disease progression "
            "evidence"
        ),
        "strength": "not_used",
        "supports_axis": "no",
        "supports_fibrosis": "positive",
        "interpretation": (
            "AT1/AT2 oriented DPT was not accepted as disease-progression "
            "evidence: the root was biologically plausible and fibrosis "
            "correlation became positive, but axis_v2 correlation was absent "
            "and the terminal region remained dominated by Control, early-stage "
            "ATI cells. Epithelial interpretation should instead rely on the "
            "injury-remodeling axis and external projection validation."
        ),
    },
}

GENE_DIRECTIONALITY_MAP = {
    "CCL18": ["Macrophage"],
    "SPP1": ["Macrophage"],
    "CD74": ["Macrophage", "T_cell", "Plasma", "B_cell"],
    "LPL": ["Macrophage"],
    "S100A10": ["Macrophage"],
    "FOS": ["Macrophage"],
    "APOE": ["Macrophage"],
    "TREM2": ["Macrophage"],
    "C1QA": ["Macrophage"],
    "C1QB": ["Macrophage"],
    "C1QC": ["Macrophage"],
    "LGALS3": ["Macrophage"],
    "CCL5": ["T_cell"],
    "CXCL13": ["B_cell", "Plasma"],
    "CXCL10": ["T_cell"],
    "IL32": ["T_cell"],
    "HLA-DRA": ["Macrophage", "B_cell", "Plasma"],
    "HLA-DRB1": ["Macrophage", "B_cell", "Plasma"],
    "MMP7": ["Epithelial_axis_validation"],
    "KRT17": ["Epithelial_axis_validation"],
    "CDKN2A": ["Epithelial_axis_validation"],
    "ACTA2": ["Smooth_muscle"],
    "TAGLN": ["Smooth_muscle"],
    "MYH11": ["Smooth_muscle"],
    "CNN1": ["Smooth_muscle"],
}

MECHANISM_LABELS = {
    "Macrophage": (
        "macrophage_IPF_associated_myeloid_transition"
    ),
    "T_cell": (
        "T_cell_IPF_terminal_like_immune_remodeling"
    ),
    "Plasma": (
        "plasma_humoral_immune_remodeling"
    ),
    "B_cell": (
        "B_cell_immune_remodeling"
    ),
    "Smooth_muscle": (
        "smooth_muscle_weak_context_support"
    ),
    "NK": (
        "NK_weak_context_support"
    ),
    "Epithelial_axis_validation": (
        "epithelial_injury_axis_validation"
    ),
    "Fibroblast": CELLTYPE_RULES["Fibroblast"]["mechanism_label"],
    "Endothelial": CELLTYPE_RULES["Endothelial"]["mechanism_label"],
}


# ============================================================
# 3. Generic utilities
# ============================================================

def clean_gene(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "gene" in df.columns:
        df["gene"] = df["gene"].map(clean_gene)

    return df


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[optional missing] {path.name}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "gene" in df.columns:
        df["gene"] = df["gene"].map(clean_gene)

    return df


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(
            s,
            errors="coerce",
        ).fillna(0) > 0

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "t", "1", "yes", "y"])
    )


def ensure_bool(df: pd.DataFrame, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = False
        df[col] = as_bool(df[col])

    return df


def ensure_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return df


def contains_any(value, terms) -> bool:
    value_text = text(value).lower()
    return any(
        str(term).lower() in value_text
        for term in terms
    )


def split_tokens(value):
    if pd.isna(value):
        return []

    return [
        token.strip()
        for token in re.split(r"[;,|/]+", str(value))
        if token.strip()
    ]


def normalize_celltype(value):
    value_text = (
        text(value)
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if not value_text:
        return ""

    if "fibro" in value_text:
        return "Fibroblast"

    if (
        "endothelial" in value_text
        or value_text in {"endo", "vascular", "ec"}
    ):
        return "Endothelial"

    if value_text in {
        "at1",
        "at2",
        "ati",
        "atii",
        "at1_at2",
        "epithelial",
    }:
        return "AT1_AT2"

    if (
        "macrophage" in value_text
        or value_text == "myeloid"
    ):
        return "Macrophage"

    if value_text in {"t_cell", "tcell", "t"}:
        return "T_cell"

    if value_text in {"b_cell", "bcell", "b"}:
        return "B_cell"

    if "plasma" in value_text:
        return "Plasma"

    if (
        value_text == "nk"
        or "natural_killer" in value_text
    ):
        return "NK"

    if "smooth" in value_text:
        return "Smooth_muscle"

    return text(value)


def classify_rho(rho):
    if not np.isfinite(rho):
        return "missing"

    if rho >= POSITIVE_RHO_THRESHOLD:
        return "positive"

    if rho > WEAK_POSITIVE_RHO_THRESHOLD:
        return "weak_positive"

    return "no"


def strength_rank(strength):
    order = {
        "strong": 1,
        "moderate": 2,
        "weak_to_moderate": 3,
        "weak": 4,
        "context": 5,
        "none": 6,
        "not_used": 7,
    }
    return order.get(text(strength), 99)


def format_float(value, digits=3):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"

    if not np.isfinite(numeric):
        return "NA"

    return f"{numeric:.{digits}f}"


# ============================================================
# 4. Discover and read CellRank result directories
# ============================================================

def resolve_result_file(result_dir: Path, file_type: str) -> Path:
    """
    优先读取带 `_oriented` 后缀的文件；
    不存在时回退到普通文件名。
    """
    if file_type not in RESULT_FILE_CANDIDATES:
        raise KeyError(
            f"Unknown CellRank result file type: {file_type}. "
            f"Available types: {list(RESULT_FILE_CANDIDATES)}"
        )

    candidates = RESULT_FILE_CANDIDATES[file_type]

    for filename in candidates:
        path = result_dir / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing required {file_type} file in: {result_dir}\n"
        f"Expected one of: {candidates}"
    )


def resolve_optional_result_file(
    result_dir: Path,
    candidates,
):
    for filename in candidates:
        path = result_dir / filename
        if path.exists():
            return path

    return None


def has_required_cellrank_files(path: Path) -> bool:
    if not path.is_dir():
        return False

    for candidates in RESULT_FILE_CANDIDATES.values():
        if not any(
            (path / filename).exists()
            for filename in candidates
        ):
            return False

    return True


def missing_required_cellrank_files(path: Path):
    missing = []

    for file_type, candidates in RESULT_FILE_CANDIDATES.items():
        if not any(
            (path / filename).exists()
            for filename in candidates
        ):
            missing.append(
                f"{file_type}: expected one of {candidates}"
            )

    return missing


def directory_matches_celltype(
    path: Path,
    celltype: str,
) -> bool:
    """
    检查目录名及其直接子文件名是否含有细胞类型线索。

    这样即使结果直接位于 Pythonresult 根目录，而根目录中存在
    Endothelial.zip 或 Endothelial_CellRank_*.py，也可以识别。
    """
    aliases = CELLTYPE_DIRECTORY_ALIASES[celltype]

    searchable_text = path.name.lower()

    try:
        immediate_names = [
            child.name.lower()
            for child in path.iterdir()
        ]
    except (PermissionError, OSError):
        immediate_names = []

    searchable_text += " " + " ".join(immediate_names)

    return any(
        alias.lower() in searchable_text
        for alias in aliases
    )


def version_number(path: Path) -> int:
    values = [
        int(value)
        for value in re.findall(
            r"(?:^|[_-])v(\d+)(?:$|[_-])",
            path.name,
            flags=re.I,
        )
    ]

    return max(values) if values else -1


def resolve_cellrank_result_dir(celltype: str) -> Path:
    explicit = EXPLICIT_CELLRANK_DIRS.get(
        celltype,
        "",
    )

    if explicit:
        path = Path(explicit).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"{celltype} explicit result directory does not exist: "
                f"{path}"
            )

        if not has_required_cellrank_files(path):
            missing = missing_required_cellrank_files(path)
            raise FileNotFoundError(
                f"{celltype} explicit result directory is incomplete: "
                f"{path}\nMissing: {missing}"
            )

        print(
            f"[{celltype}] using explicit result directory: "
            f"{path}"
        )
        return path

    if not CELLRANK_RESULTS_ROOT.exists():
        raise FileNotFoundError(
            f"CELLRANK_RESULTS_ROOT does not exist: "
            f"{CELLRANK_RESULTS_ROOT}\n"
            f"Set {celltype.upper()}_CELLRANK_DIR or "
            f"CELLRANK_RESULTS_ROOT."
        )

    all_directories = [CELLRANK_RESULTS_ROOT]

    try:
        all_directories.extend(
            path
            for path in CELLRANK_RESULTS_ROOT.rglob("*")
            if path.is_dir()
        )
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot recursively search: "
            f"{CELLRANK_RESULTS_ROOT}"
        ) from exc

    candidates = []

    for path in all_directories:
        if not has_required_cellrank_files(path):
            continue

        if directory_matches_celltype(path, celltype):
            candidates.append(path)

    if not candidates:
        complete_directories = [
            path
            for path in all_directories
            if has_required_cellrank_files(path)
        ]

        details = "\n".join(
            f"  - {path}"
            for path in complete_directories[:10]
        )

        if details:
            details = (
                "\nComplete CellRank-like directories found, "
                "but none matched this cell type:\n"
                + details
            )
        else:
            details = (
                "\nNo directory containing all required result "
                "file types was found."
            )

        raise FileNotFoundError(
            f"No complete {celltype} CellRank result directory "
            f"found under: {CELLRANK_RESULTS_ROOT}"
            f"{details}\n"
            f"You can set "
            f"{celltype.upper()}_CELLRANK_DIR explicitly."
        )

    candidates = sorted(
        set(candidates),
        key=lambda path: (
            version_number(path),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    chosen = candidates[0]

    print(
        f"[{celltype}] auto-selected result directory: "
        f"{chosen}"
    )

    if len(candidates) > 1:
        print(
            f"[{celltype}] other complete candidates:"
        )
        for path in candidates[1:6]:
            print("   ", path)

    return chosen


def composition_fraction_table(path: Path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    fraction_col = next(
        (
            column
            for column in df.columns
            if "fraction" in column.lower()
        ),
        None,
    )

    count_col = next(
        (
            column
            for column in df.columns
            if (
                "n_terminal" in column.lower()
                or column.lower() == "count"
            )
        ),
        None,
    )

    category_candidates = [
        column
        for column in df.columns
        if column not in {fraction_col, count_col}
        and "cutoff" not in column.lower()
    ]

    if not category_candidates:
        raise ValueError(
            f"Cannot identify category column in: {path}"
        )

    category_col = category_candidates[0]

    if fraction_col is None:
        if count_col is None:
            raise ValueError(
                f"No fraction/count column in: {path}"
            )

        counts = pd.to_numeric(
            df[count_col],
            errors="coerce",
        ).fillna(0.0)

        denominator = counts.sum()

        if denominator > 0:
            df["_fraction"] = counts / denominator
        else:
            df["_fraction"] = 0.0

        fraction_col = "_fraction"

    else:
        df[fraction_col] = pd.to_numeric(
            df[fraction_col],
            errors="coerce",
        ).fillna(0.0)

        # 如果比例以百分数形式保存，自动转换为 0-1。
        max_fraction = df[fraction_col].max()
        total_fraction = df[fraction_col].sum()

        if (
            max_fraction > 1.0
            and total_fraction > 1.5
        ):
            df[fraction_col] = df[fraction_col] / 100.0

    return df, category_col, fraction_col


def fraction_matching(path: Path, terms):
    df, category_col, fraction_col = (
        composition_fraction_table(path)
    )

    mask = pd.Series(
        False,
        index=df.index,
    )

    values = df[category_col].astype(str)

    for term in terms:
        mask = mask | values.str.contains(
            str(term),
            case=False,
            na=False,
            regex=False,
        )

    return (
        float(df.loc[mask, fraction_col].sum()),
        category_col,
    )


def extract_oriented_rho(
    corr_df: pd.DataFrame,
    reference_name: str,
):
    ref_col = next(
        (
            column
            for column in [
                "y",
                "reference_col",
                "variable",
                "feature",
                "reference",
            ]
            if column in corr_df.columns
        ),
        None,
    )

    # 优先读取明确标记为 orientation 后的相关性。
    rho_col = next(
        (
            column
            for column in [
                "spearman_rho_after_orientation",
                "oriented_spearman_rho",
                "spearman_rho_oriented",
                "spearman_rho",
                "rho",
            ]
            if column in corr_df.columns
        ),
        None,
    )

    if ref_col is None or rho_col is None:
        return np.nan

    exact_mask = (
        corr_df[ref_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == reference_name.lower()
    )

    mask = exact_mask

    if not mask.any():
        mask = (
            corr_df[ref_col]
            .astype(str)
            .str.contains(
                reference_name,
                case=False,
                na=False,
                regex=False,
            )
        )

    values = pd.to_numeric(
        corr_df.loc[mask, rho_col],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[0])


def read_root_summary(result_dir: Path):
    root_path = resolve_result_file(
        result_dir,
        "root",
    )

    root = pd.read_csv(
        root_path,
        low_memory=False,
    )

    root.columns = [
        str(column).strip()
        for column in root.columns
    ]

    if root.empty:
        raise ValueError(
            f"Empty root_cell_report.csv: {result_dir}"
        )

    row = root.iloc[0]

    subtype_cols = [
        column
        for column in [
            "celltype_marker_based",
            "Subclass_Cell_Identity",
            "Manuscript_Identity",
            "Celltype_Identity",
            "celltype",
            "cell_type",
        ]
        if column in root.columns
    ]

    subtype_text = ";".join(
        text(row.get(column, ""))
        for column in subtype_cols
        if text(row.get(column, ""))
    )

    root_axis_score = pd.to_numeric(
        pd.Series([
            row.get("axis_v2_score", np.nan)
        ]),
        errors="coerce",
    ).iloc[0]

    root_fibrosis_score = pd.to_numeric(
        pd.Series([
            row.get("fibrosis_composite_score", np.nan)
        ]),
        errors="coerce",
    ).iloc[0]

    return {
        "root_cell": text(
            row.get("root_cell", "")
        ),
        "root_disease": text(
            row.get("Disease_Identity", "")
        ),
        "root_stage": text(
            row.get(
                "axis_v2_stage",
                row.get(
                    "multicellular_axis_stage",
                    "",
                ),
            )
        ),
        "root_subtype": subtype_text,
        "root_axis_score": root_axis_score,
        "root_fibrosis_score": (
            root_fibrosis_score
        ),
        "root_chosen_tier": text(
            row.get("chosen_tier", "")
        ),
    }


# ============================================================
# 5. Conservative validation of Fibroblast / Endothelial
# ============================================================

def validate_cellrank_result(
    celltype: str,
    result_dir: Path,
):
    rules = CELLTYPE_RULES[celltype]

    corr_path = resolve_result_file(
        result_dir,
        "correlation",
    )
    disease_path = resolve_result_file(
        result_dir,
        "disease_composition",
    )
    stage_path = resolve_result_file(
        result_dir,
        "stage_composition",
    )
    root_path = resolve_result_file(
        result_dir,
        "root",
    )

    corr = pd.read_csv(
        corr_path,
        low_memory=False,
    )
    corr.columns = [
        str(column).strip()
        for column in corr.columns
    ]

    rho_axis = extract_oriented_rho(
        corr,
        "axis_v2_score",
    )
    rho_fib = extract_oriented_rho(
        corr,
        "fibrosis_composite_score",
    )

    ipf_fraction, _ = fraction_matching(
        disease_path,
        ["IPF"],
    )
    control_fraction, _ = fraction_matching(
        disease_path,
        ["Control"],
    )
    copd_fraction, _ = fraction_matching(
        disease_path,
        ["COPD"],
    )

    late_fraction, _ = fraction_matching(
        stage_path,
        [
            "Stage_3",
            "Stage_4",
            "Stage 3",
            "Stage 4",
        ],
    )
    early_fraction, _ = fraction_matching(
        stage_path,
        [
            "Stage_1",
            "Stage_2",
            "Stage 1",
            "Stage 2",
        ],
    )

    subtype_fraction = np.nan
    subtype_source = ""
    subtype_parts = []

    for candidate_group in OPTIONAL_SUBTYPE_FILE_CANDIDATES:
        subtype_path = resolve_optional_result_file(
            result_dir,
            candidate_group,
        )

        if subtype_path is None:
            continue

        fraction, category_col = fraction_matching(
            subtype_path,
            rules["expected_terminal_terms"],
        )

        subtype_parts.append(
            (
                fraction,
                subtype_path.name,
                category_col,
            )
        )

    if subtype_parts:
        subtype_fraction, subtype_source, _ = max(
            subtype_parts,
            key=lambda item: item[0],
        )

    root = read_root_summary(result_dir)

    root_early = contains_any(
        root["root_stage"],
        [
            "Stage_1",
            "Stage_2",
            "Stage 1",
            "Stage 2",
            "early",
        ],
    )

    root_control = contains_any(
        root["root_disease"],
        ["Control"],
    )

    root_forbidden = contains_any(
        (
            root["root_subtype"]
            + ";"
            + root["root_chosen_tier"]
        ),
        rules["forbidden_root_terms"],
    )

    root_plausible = bool(
        root_early
        and not root_forbidden
    )

    axis_positive = bool(
        np.isfinite(rho_axis)
        and rho_axis >= POSITIVE_RHO_THRESHOLD
    )
    fib_positive = bool(
        np.isfinite(rho_fib)
        and rho_fib >= POSITIVE_RHO_THRESHOLD
    )

    positive_corr_count = (
        int(axis_positive)
        + int(fib_positive)
    )

    late_dominant = bool(
        late_fraction
        >= TERMINAL_LATE_STAGE_MIN
    )

    ipf_enriched = bool(
        ipf_fraction >= TERMINAL_IPF_MIN
        and ipf_fraction > control_fraction
    )

    subtype_aligned = bool(
        np.isfinite(subtype_fraction)
        and subtype_fraction
        >= TERMINAL_TARGET_SUBTYPE_MIN
    )

    terminal_signal_count = (
        int(late_dominant)
        + int(ipf_enriched)
        + int(subtype_aligned)
    )

    # root、连续方向和 terminal 至少各有一条证据才纳入。
    usable = bool(
        root_plausible
        and positive_corr_count >= 1
        and terminal_signal_count >= 1
    )

    if (
        usable
        and root_control
        and positive_corr_count == 2
        and terminal_signal_count >= 2
    ):
        strength = "strong"

    elif usable:
        strength = "moderate"

    elif (
        root_plausible
        and (
            positive_corr_count >= 1
            or terminal_signal_count >= 1
        )
    ):
        strength = "weak_to_moderate"

    else:
        strength = "not_used"

    if celltype == "Fibroblast":
        if subtype_aligned:
            support_type = (
                "fibrosis-aligned fibroblast-to-myofibroblast "
                "progression trajectory"
            )
        else:
            support_type = (
                "fibroblast late-stage remodeling trajectory"
            )
    else:
        support_type = (
            "disease-associated endothelial vascular "
            "remodeling trajectory"
        )

    subtype_text = ""

    if np.isfinite(subtype_fraction):
        subtype_text = (
            ", target-terminal subtype fraction="
            f"{subtype_fraction:.3f}"
        )

    used_oriented_filename = any(
        "_oriented" in path.name.lower()
        for path in [
            corr_path,
            disease_path,
            stage_path,
        ]
    )

    if used_oriented_filename:
        filename_status = (
            "At least one explicitly named oriented result file "
            "was used."
        )
    else:
        filename_status = (
            "Plain result filenames were used. The script does not "
            "rename them or claim that filename alone proves "
            "orientation; values are interpreted as stored in the "
            "provided result files."
        )

    interpretation = (
        f"{celltype} DPT/CellRank: "
        f"rho(axis_v2)={format_float(rho_axis)}, "
        f"rho(fibrosis)={format_float(rho_fib)}; "
        f"terminal IPF={ipf_fraction:.3f}, "
        f"Control={control_fraction:.3f}, "
        f"Stage3/4={late_fraction:.3f}"
        f"{subtype_text}. "
        f"Root: disease={root['root_disease']}, "
        f"stage={root['root_stage']}, "
        f"subtype={root['root_subtype']}. "
        f"Conservative decision: usable={usable}, "
        f"strength={strength}. "
        f"{filename_status}"
    )

    return {
        "celltype": celltype,
        "result_dir": str(result_dir),
        "root_file": root_path.name,
        "correlation_file": corr_path.name,
        "disease_composition_file": disease_path.name,
        "stage_composition_file": stage_path.name,
        "used_oriented_filename": (
            used_oriented_filename
        ),
        "usable": usable,
        "strength": strength,
        "support_type": support_type,
        "supports_axis": classify_rho(rho_axis),
        "supports_fibrosis": classify_rho(rho_fib),
        "interpretation": interpretation,
        "rho_axis_v2": rho_axis,
        "rho_fibrosis": rho_fib,
        "terminal_ipf_fraction": ipf_fraction,
        "terminal_control_fraction": control_fraction,
        "terminal_copd_fraction": copd_fraction,
        "terminal_late_stage_fraction": late_fraction,
        "terminal_early_stage_fraction": early_fraction,
        "terminal_target_subtype_fraction": (
            subtype_fraction
        ),
        "terminal_target_subtype_source": (
            subtype_source
        ),
        "root_plausible": root_plausible,
        "root_control": root_control,
        "root_early_stage": root_early,
        "root_forbidden_subtype": root_forbidden,
        "root_cell": root["root_cell"],
        "root_disease": root["root_disease"],
        "root_stage": root["root_stage"],
        "root_subtype": root["root_subtype"],
        "root_axis_score": root["root_axis_score"],
        "root_fibrosis_score": (
            root["root_fibrosis_score"]
        ),
        "root_chosen_tier": (
            root["root_chosen_tier"]
        ),
        "axis_positive": axis_positive,
        "fibrosis_positive": fib_positive,
        "late_stage_dominant": late_dominant,
        "ipf_enriched": ipf_enriched,
        "target_subtype_aligned": subtype_aligned,
    }


# ============================================================
# 6. Unified validation table
# ============================================================

AT_NEGATIVE_VALIDATION = {
    "validation_source": (
        "completed_AT_v5_negative_result"
    ),
    "metrics_available": True,
    "rho_axis_v2": -0.010,
    "rho_fibrosis": 0.220,
    "terminal_ipf_fraction": 0.024,
    "terminal_control_fraction": 0.879,
    "terminal_copd_fraction": 0.097,
    "terminal_late_stage_fraction": 0.105,
    "terminal_early_stage_fraction": 0.895,
    "terminal_target_subtype_fraction": np.nan,
    "terminal_target_subtype_source": "",
    "root_plausible": True,
    "root_control": True,
    "root_early_stage": True,
    "root_forbidden_subtype": False,
    "root_cell": (
        "GSM4058900_133C-a__CAGAGAGAGGTAGCCA"
    ),
    "root_disease": "Control",
    "root_stage": "Stage_1_low",
    "root_subtype": "AT1;ATI",
    "root_axis_score": 0.0096,
    "root_fibrosis_score": 0.0,
    "root_chosen_tier": (
        "Control+Stage1+ATI/lowATII+nonHighSurf+"
        "axis/fib<=q0.1"
    ),
    "axis_positive": False,
    "fibrosis_positive": True,
    "late_stage_dominant": False,
    "ipf_enriched": False,
    "target_subtype_aligned": False,
    "decision_basis": (
        "Excluded because oriented pseudotime was not aligned "
        "with axis_v2 and the terminal region was dominated by "
        "Control/Stage1-2/ATI rather than IPF/Stage3-4 states."
    ),
}

VALIDATION_OUTPUT_COLUMNS = [
    "celltype",
    "validation_source",
    "metrics_available",
    "result_dir",
    "root_file",
    "correlation_file",
    "disease_composition_file",
    "stage_composition_file",
    "used_oriented_filename",
    "usable",
    "included_in_final_directionality",
    "strength",
    "support_type",
    "supports_axis",
    "supports_fibrosis",
    "rho_axis_v2",
    "rho_fibrosis",
    "terminal_ipf_fraction",
    "terminal_control_fraction",
    "terminal_copd_fraction",
    "terminal_late_stage_fraction",
    "terminal_early_stage_fraction",
    "terminal_target_subtype_fraction",
    "terminal_target_subtype_source",
    "root_plausible",
    "root_control",
    "root_early_stage",
    "root_forbidden_subtype",
    "root_cell",
    "root_disease",
    "root_stage",
    "root_subtype",
    "root_axis_score",
    "root_fibrosis_score",
    "root_chosen_tier",
    "axis_positive",
    "fibrosis_positive",
    "late_stage_dominant",
    "ipf_enriched",
    "target_subtype_aligned",
    "decision_basis",
    "interpretation",
]

VALIDATION_CELLTYPE_ORDER = {
    "Macrophage": 1,
    "Plasma": 2,
    "T_cell": 3,
    "B_cell": 4,
    "Smooth_muscle": 5,
    "NK": 6,
    "Fibroblast": 7,
    "Endothelial": 8,
    "AT1_AT2": 9,
}


def blank_validation_row():
    return {
        column: np.nan
        for column in VALIDATION_OUTPUT_COLUMNS
    }


def build_previous_validation_rows():
    rows = []

    for celltype, evidence in (
        BASE_CELLRANK_EVIDENCE.items()
    ):
        row = blank_validation_row()

        row.update({
            "celltype": celltype,
            "validation_source": (
                "previous_completed_validation_summary"
            ),
            "metrics_available": False,
            "result_dir": "",
            "root_file": "",
            "correlation_file": "",
            "disease_composition_file": "",
            "stage_composition_file": "",
            "used_oriented_filename": np.nan,
            "usable": bool(
                evidence.get("usable", False)
            ),
            "included_in_final_directionality": bool(
                evidence.get("usable", False)
            ),
            "strength": evidence.get(
                "strength",
                "not_used",
            ),
            "support_type": evidence.get(
                "support_type",
                "",
            ),
            "supports_axis": evidence.get(
                "supports_axis",
                "not_available",
            ),
            "supports_fibrosis": evidence.get(
                "supports_fibrosis",
                "not_available",
            ),
            "decision_basis": (
                "Imported from the previously completed "
                "CellRank/DPT validation summary. The current "
                "inputs retain the validated interpretation, "
                "but do not include the original quantitative "
                "diagnostic files for this cell type."
            ),
            "interpretation": evidence.get(
                "interpretation",
                "",
            ),
        })

        if celltype == "AT1_AT2":
            row.update(AT_NEGATIVE_VALIDATION)
            row["usable"] = False
            row[
                "included_in_final_directionality"
            ] = False
            row["strength"] = "not_used"
            row["support_type"] = evidence.get(
                "support_type",
                "",
            )
            row["supports_axis"] = evidence.get(
                "supports_axis",
                "no",
            )
            row["supports_fibrosis"] = evidence.get(
                "supports_fibrosis",
                "positive",
            )
            row["interpretation"] = evidence.get(
                "interpretation",
                "",
            )

        rows.append(row)

    return rows


def standardize_new_validation_row(result):
    row = blank_validation_row()
    row.update(result)

    row.update({
        "validation_source": (
            "current_standard_cellrank_output_files"
        ),
        "metrics_available": True,
        "included_in_final_directionality": bool(
            result.get("usable", False)
        ),
        "decision_basis": (
            "Automatically evaluated from the selected root, "
            "Spearman-correlation and terminal-composition "
            "files using the conservative thresholds defined "
            "in this script."
        ),
    })

    return row


def build_final_evidence():
    evidence = {
        key: dict(value)
        for key, value in (
            BASE_CELLRANK_EVIDENCE.items()
        )
    }

    validation_rows = build_previous_validation_rows()

    for celltype in [
        "Fibroblast",
        "Endothelial",
    ]:
        try:
            result_dir = resolve_cellrank_result_dir(
                celltype
            )

            result = validate_cellrank_result(
                celltype,
                result_dir,
            )

            validation_rows.append(
                standardize_new_validation_row(result)
            )

            evidence[celltype] = {
                "usable": bool(result["usable"]),
                "support_type": result["support_type"],
                "strength": result["strength"],
                "supports_axis": (
                    result["supports_axis"]
                ),
                "supports_fibrosis": (
                    result["supports_fibrosis"]
                ),
                "interpretation": (
                    result["interpretation"]
                ),
            }

            print(
                f"[{celltype}] "
                f"usable={result['usable']}, "
                f"strength={result['strength']}, "
                f"rho_axis={format_float(result['rho_axis_v2'])}, "
                f"rho_fib={format_float(result['rho_fibrosis'])}, "
                f"terminal_late="
                f"{result['terminal_late_stage_fraction']:.3f}, "
                f"terminal_IPF="
                f"{result['terminal_ipf_fraction']:.3f}"
            )

            print(
                f"[{celltype}] selected files: "
                f"{result['correlation_file']}; "
                f"{result['disease_composition_file']}; "
                f"{result['stage_composition_file']}"
            )

        except Exception as exc:
            if REQUIRE_FINAL_CELLTYPE_RESULTS:
                raise

            print(
                f"[{celltype}] validation skipped: "
                f"{exc}"
            )

            evidence[celltype] = {
                "usable": False,
                "support_type": (
                    "result_missing_or_invalid"
                ),
                "strength": "not_used",
                "supports_axis": "not_used",
                "supports_fibrosis": "not_used",
                "interpretation": str(exc),
            }

            missing_row = blank_validation_row()

            missing_row.update({
                "celltype": celltype,
                "validation_source": (
                    "current_standard_cellrank_output_files"
                ),
                "metrics_available": False,
                "result_dir": "",
                "root_file": "",
                "correlation_file": "",
                "disease_composition_file": "",
                "stage_composition_file": "",
                "used_oriented_filename": np.nan,
                "usable": False,
                "included_in_final_directionality": False,
                "strength": "not_used",
                "support_type": (
                    "result_missing_or_invalid"
                ),
                "supports_axis": "not_used",
                "supports_fibrosis": "not_used",
                "decision_basis": (
                    "Result directory missing or invalid."
                ),
                "interpretation": str(exc),
            })

            validation_rows.append(missing_row)

    return (
        evidence,
        pd.DataFrame(validation_rows),
    )


# ============================================================
# 7. Row-level directionality integration
# ============================================================

def row_final_celltypes(row: pd.Series):
    """
    优先使用 top_celltype。
    仅当 top_celltype 为空时，才从其他字段回退识别。
    """
    top = normalize_celltype(
        row.get("top_celltype", "")
    )

    if top in FINAL_VALIDATED_CELLTYPES:
        return [top]

    if top:
        return []

    found = set()

    for column in [
        "celltypes_detected",
        "celltype",
        "cell_type",
    ]:
        if column not in row.index:
            continue

        for token in split_tokens(
            row.get(column, "")
        ):
            celltype = normalize_celltype(token)

            if celltype in FINAL_VALIDATED_CELLTYPES:
                found.add(celltype)

    return sorted(found)


def directionality_support_for_row(
    row: pd.Series,
    evidence,
):
    gene = clean_gene(
        row.get("gene", "")
    )

    supports = list(
        GENE_DIRECTIONALITY_MAP.get(
            gene,
            [],
        )
    )

    supports.extend(
        row_final_celltypes(row)
    )

    cleaned = []

    for support in supports:
        if support in EXCLUDED_CELLRANK_CELLTYPES:
            continue

        if support not in cleaned:
            cleaned.append(support)

    labels = []
    strengths = []
    notes = []
    supporting_celltypes = []

    for support in cleaned:
        if support == "Epithelial_axis_validation":
            labels.append(
                MECHANISM_LABELS[support]
            )
            strengths.append("moderate")
            supporting_celltypes.append(support)
            notes.append(
                "Supported by the epithelial "
                "injury-remodeling axis/external "
                "validation, not by AT CellRank."
            )
            continue

        celltype_evidence = evidence.get(
            support
        )

        if (
            celltype_evidence is None
            or not celltype_evidence.get(
                "usable",
                False,
            )
        ):
            continue

        labels.append(
            MECHANISM_LABELS.get(
                support,
                support,
            )
        )
        strengths.append(
            celltype_evidence.get(
                "strength",
                "context",
            )
        )
        supporting_celltypes.append(support)
        notes.append(
            celltype_evidence.get(
                "interpretation",
                "",
            )
        )

    if not labels:
        return {
            "current_directionality_support": (
                "no_direct_support_in_current_"
                "validated_celltypes"
            ),
            "current_directionality_strength": (
                "none"
            ),
            "directionality_supporting_celltypes": "",
            "directionality_interpretation_note": "",
        }

    final_strength = sorted(
        strengths,
        key=strength_rank,
    )[0]

    return {
        "current_directionality_support": ";".join(
            sorted(set(labels))
        ),
        "current_directionality_strength": (
            final_strength
        ),
        "directionality_supporting_celltypes": ";".join(
            sorted(set(supporting_celltypes))
        ),
        "directionality_interpretation_note": " | ".join(
            sorted(
                set(
                    note
                    for note in notes
                    if note
                )
            )
        ),
    }


def mechanism_interpretation(
    row: pd.Series,
    evidence,
) -> str:
    gene = clean_gene(
        row.get("gene", "")
    )

    supporting = set(
        split_tokens(
            row.get(
                "directionality_supporting_celltypes",
                "",
            )
        )
    )

    statements = []

    if "Fibroblast" in supporting:
        statements.append(
            f"{gene} is linked to the validated "
            "fibroblast remodeling trajectory. "
            + evidence["Fibroblast"]["interpretation"]
        )

    if "Endothelial" in supporting:
        statements.append(
            f"{gene} is linked to the validated "
            "endothelial vascular-remodeling trajectory. "
            + evidence["Endothelial"]["interpretation"]
        )

    if statements:
        return " ".join(statements)

    if gene in {
        "CCL18",
        "SPP1",
        "CD74",
        "LPL",
        "S100A10",
        "FOS",
        "APOE",
        "TREM2",
        "LGALS3",
        "C1QA",
        "C1QB",
        "C1QC",
    }:
        return (
            f"{gene} links the PPI-informed driver network "
            "to IPF-associated macrophage/myeloid remodeling. "
            "Current CellRank/DPT evidence supports an "
            "IPF-enriched myeloid terminal-like state, but "
            "this should be interpreted as inflammatory/"
            "myeloid progression rather than a direct "
            "fibrosis-terminal trajectory."
        )

    if gene in {
        "CCL5",
        "CXCL10",
        "IL32",
    }:
        return (
            f"{gene} is best interpreted as a lymphoid "
            "immune-remodeling driver. Current T-cell/NK "
            "directionality support is supplementary."
        )

    if gene in {
        "CXCL13",
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
    }:
        return (
            f"{gene} supports B/plasma/antigen-presentation "
            "immune remodeling. Plasma and B-cell CellRank "
            "results provide supplementary support for "
            "IPF-associated humoral immune states."
        )

    if gene in {
        "MMP7",
        "KRT17",
        "CDKN2A",
    }:
        return (
            f"{gene} should be interpreted through the "
            "epithelial injury-remodeling axis and external "
            "epithelial validation. AT1/AT2 CellRank is not "
            "used as primary directionality evidence."
        )

    if gene in {
        "ACTA2",
        "TAGLN",
        "MYH11",
        "CNN1",
    }:
        return (
            f"{gene} is a smooth-muscle/pericyte or "
            "mesenchymal context candidate. Current "
            "Smooth_muscle CellRank support is weak and "
            "supplementary."
        )

    return (
        f"{gene} remains a PPI/ODE-informed driver candidate, "
        "but the currently validated CellRank trajectories do "
        "not provide direct strong directionality support. "
        "Interpret using ODE, dynamic-module, PPI and "
        "druggability evidence."
    )


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "gene",
        "driver_score_rank",
        "ppi_informed_driver_score",
        "refined_priority_tier",
        "refined_driver_category",
        "mechanism_group",
        "current_directionality_support",
        "current_directionality_strength",
        "directionality_supporting_celltypes",
        "directionality_interpretation_note",
        "mechanism_interpretation",
        "top_celltype",
        "n_celltypes",
        "n_states",
        "active_progression_driver",
        "terminal_state_maintenance_driver",
        "shared_profibrotic_hub",
        "celltype_specific_driver",
        "high_priority_druggable_driver",
        "approved_druggable_driver",
        "anti_neoplastic_druggable_driver",
        "progression_driver_subscore",
        "terminal_maintenance_subscore",
        "network_hub_subscore",
        "neural_ode_direction_score_norm",
        "ppi_centrality_score_norm",
        "druggability_score_norm",
        "drug_names",
    ]

    existing = [
        column
        for column in preferred
        if column in df.columns
    ]

    remaining = [
        column
        for column in df.columns
        if column not in existing
    ]

    return df[existing + remaining]


# ============================================================
# 8. Main
# ============================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("BASE_DIR:", BASE_DIR)
    print("OUT_DIR:", OUT_DIR)
    print(
        "CELLRANK_RESULTS_ROOT:",
        CELLRANK_RESULTS_ROOT,
    )
    print(
        "AT CellRank excluded:",
        sorted(EXCLUDED_CELLRANK_CELLTYPES),
    )
    print(
        "Accepted correlation filenames:",
        RESULT_FILE_CANDIDATES["correlation"],
    )
    print(
        "Accepted disease-composition filenames:",
        RESULT_FILE_CANDIDATES[
            "disease_composition"
        ],
    )
    print(
        "Accepted stage-composition filenames:",
        RESULT_FILE_CANDIDATES[
            "stage_composition"
        ],
    )

    evidence, validation_df = (
        build_final_evidence()
    )

    for column in VALIDATION_OUTPUT_COLUMNS:
        if column not in validation_df.columns:
            validation_df[column] = np.nan

    validation_df["_celltype_order"] = (
        validation_df["celltype"]
        .map(VALIDATION_CELLTYPE_ORDER)
        .fillna(99)
    )

    validation_df = (
        validation_df
        .sort_values([
            "_celltype_order",
            "celltype",
        ])
        .drop(columns=["_celltype_order"])
    )

    validation_df = validation_df[
        VALIDATION_OUTPUT_COLUMNS
    ]

    validation_path = (
        OUT_DIR
        / "22_cellrank_directionality_validation_all_celltypes.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    top50 = read_csv_required(
        BASE_DIR
        / "15_final_driver_shortlist_top50.csv"
    )
    tier_a = read_csv_required(
        BASE_DIR
        / "16_tierA_primary_driver_candidates.csv"
    )
    mechanism = read_csv_required(
        BASE_DIR
        / "17_mechanism_grouped_driver_table.csv"
    )
    watchlist = read_csv_optional(
        BASE_DIR
        / "18_supplementary_watchlist_candidates.csv"
    )
    druggable = read_csv_optional(
        BASE_DIR
        / "11d_high_priority_druggable_driver_candidates.csv"
    )

    bool_cols = [
        "active_progression_driver",
        "terminal_state_maintenance_driver",
        "shared_profibrotic_hub",
        "celltype_specific_driver",
        "high_priority_druggable_driver",
        "approved_druggable_driver",
        "anti_neoplastic_druggable_driver",
    ]

    numeric_cols = [
        "ppi_informed_driver_score",
        "progression_driver_subscore",
        "terminal_maintenance_subscore",
        "network_hub_subscore",
        "neural_ode_direction_score_norm",
        "ppi_centrality_score_norm",
        "druggability_score_norm",
        "n_celltypes",
        "n_states",
    ]

    frames = []

    for label, dataframe in [
        ("top50", top50),
        ("tierA", tier_a),
        ("mechanism", mechanism),
        ("watchlist", watchlist),
        ("druggable", druggable),
    ]:
        if dataframe.empty:
            continue

        dataframe = dataframe.copy()
        dataframe["source_table"] = label
        frames.append(dataframe)

    if not frames:
        raise ValueError(
            "No driver input table contains data."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if "gene" not in combined.columns:
        raise ValueError(
            "Input driver tables do not contain "
            "a 'gene' column."
        )

    combined["gene"] = combined["gene"].map(
        clean_gene
    )

    combined = ensure_numeric(
        combined,
        ["ppi_informed_driver_score"],
    )

    combined["_source_priority"] = (
        combined["source_table"]
        .map({
            "tierA": 1,
            "top50": 2,
            "mechanism": 3,
            "druggable": 4,
            "watchlist": 5,
        })
        .fillna(9)
    )

    combined = (
        combined
        .sort_values(
            [
                "gene",
                "_source_priority",
                "ppi_informed_driver_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            "gene",
            keep="first",
        )
        .reset_index(drop=True)
    )

    combined = ensure_bool(
        combined,
        bool_cols,
    )
    combined = ensure_numeric(
        combined,
        numeric_cols,
    )

    support_rows = [
        directionality_support_for_row(
            row,
            evidence,
        )
        for _, row in combined.iterrows()
    ]

    support_df = pd.DataFrame(
        support_rows
    )

    combined = pd.concat(
        [
            combined.reset_index(drop=True),
            support_df,
        ],
        axis=1,
    )

    combined["mechanism_interpretation"] = (
        combined.apply(
            lambda row: mechanism_interpretation(
                row,
                evidence,
            ),
            axis=1,
        )
    )

    # --------------------------------------------------------
    # 19. Full final directionality support table
    # --------------------------------------------------------
    out19 = combined.copy()

    out19["_strength_order"] = (
        out19[
            "current_directionality_strength"
        ].map(strength_rank)
    )

    out19 = (
        out19
        .sort_values(
            [
                "_strength_order",
                "ppi_informed_driver_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(columns=["_strength_order"])
    )

    out19 = reorder(out19)

    out19_path = (
        OUT_DIR
        / "19_driver_directionality_support_all_validated_celltypes.csv"
    )

    out19.to_csv(
        out19_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 20. Mechanism interpretation table
    # --------------------------------------------------------
    tier_a_genes = set(
        tier_a["gene"].map(clean_gene)
    )

    out20 = combined[
        (
            combined[
                "current_directionality_strength"
            ]
            != "none"
        )
        | (
            combined["gene"].isin(
                tier_a_genes
            )
        )
    ].copy()

    out20["_strength_order"] = (
        out20[
            "current_directionality_strength"
        ].map(strength_rank)
    )

    out20 = (
        out20
        .sort_values(
            [
                "_strength_order",
                "ppi_informed_driver_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(columns=["_strength_order"])
    )

    out20 = reorder(out20)

    out20_path = (
        OUT_DIR
        / "20_driver_mechanism_interpretation_all_validated_celltypes.csv"
    )

    out20.to_csv(
        out20_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 21. Result narrative table
    # --------------------------------------------------------
    narrative_rows = []

    for _, row in out20.iterrows():
        gene = row["gene"]

        support = row[
            "current_directionality_support"
        ]

        strength = row[
            "current_directionality_strength"
        ]

        category = text(
            row.get(
                "refined_driver_category",
                "",
            )
        )

        tier = text(
            row.get(
                "refined_priority_tier",
                "",
            )
        )

        score = pd.to_numeric(
            pd.Series([
                row.get(
                    "ppi_informed_driver_score",
                    np.nan,
                )
            ]),
            errors="coerce",
        ).iloc[0]

        interpretation = text(
            row.get(
                "mechanism_interpretation",
                "",
            )
        )

        score_text = (
            f"{score:.3f}"
            if np.isfinite(score)
            else "NA"
        )

        narrative = (
            f"{gene} was prioritized as a {tier} "
            f"candidate (PPI-informed score={score_text}) "
            f"with category evidence [{category}]. "
            f"Current directionality support: "
            f"{support} ({strength}). "
            f"{interpretation}"
        )

        narrative_rows.append({
            "gene": gene,
            "ppi_informed_driver_score": score,
            "refined_priority_tier": tier,
            "refined_driver_category": category,
            "current_directionality_support": support,
            "current_directionality_strength": strength,
            "directionality_supporting_celltypes": text(
                row.get(
                    "directionality_supporting_celltypes",
                    "",
                )
            ),
            "result_sentence": narrative,
        })

    out21 = pd.DataFrame(
        narrative_rows
    )

    out21_path = (
        OUT_DIR
        / "21_driver_result_narrative_all_validated_celltypes.csv"
    )

    out21.to_csv(
        out21_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 23. Run configuration
    # --------------------------------------------------------
    run_meta = {
        "script": (
            "25_integrate_cellrank_directionality_"
            "all_celltypes_v4.py"
        ),
        "base_dir": str(BASE_DIR),
        "out_dir": str(OUT_DIR),
        "cellrank_results_root": str(
            CELLRANK_RESULTS_ROOT
        ),
        "explicit_cellrank_dirs": (
            EXPLICIT_CELLRANK_DIRS
        ),
        "require_final_celltype_results": (
            REQUIRE_FINAL_CELLTYPE_RESULTS
        ),
        "accepted_result_file_candidates": (
            RESULT_FILE_CANDIDATES
        ),
        "optional_subtype_file_candidates": (
            OPTIONAL_SUBTYPE_FILE_CANDIDATES
        ),
        "excluded_cellrank_celltypes": sorted(
            EXCLUDED_CELLRANK_CELLTYPES
        ),
        "positive_rho_threshold": (
            POSITIVE_RHO_THRESHOLD
        ),
        "weak_positive_rho_threshold": (
            WEAK_POSITIVE_RHO_THRESHOLD
        ),
        "terminal_late_stage_min": (
            TERMINAL_LATE_STAGE_MIN
        ),
        "terminal_ipf_min": TERMINAL_IPF_MIN,
        "terminal_target_subtype_min": (
            TERMINAL_TARGET_SUBTYPE_MIN
        ),
        "complete_validation_table": str(
            validation_path
        ),
        "final_evidence": evidence,
    }

    config_path = (
        OUT_DIR
        / "23_cellrank_integration_all_celltypes_run_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_meta,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\nDone. Generated:")

    for path in [
        out19_path,
        out20_path,
        out21_path,
        validation_path,
        config_path,
    ]:
        print(" ", path)

    print("\nCounts:")
    print(
        "  combined genes:",
        combined.shape[0],
    )
    print(
        "  supported genes:",
        int(
            (
                combined[
                    "current_directionality_strength"
                ]
                != "none"
            ).sum()
        ),
    )

    for celltype in [
        "Fibroblast",
        "Endothelial",
    ]:
        count = int(
            combined[
                "directionality_supporting_celltypes"
            ]
            .fillna("")
            .str.contains(
                celltype,
                regex=False,
            )
            .sum()
        )

        print(
            f"  genes supported by "
            f"{celltype}: {count}"
        )

    print(
        "  table 19:",
        out19.shape,
    )
    print(
        "  table 20:",
        out20.shape,
    )
    print(
        "  table 21:",
        out21.shape,
    )
    print(
        "  complete validation table:",
        validation_df.shape,
    )
    print(
        "  cell types included in final directionality:",
        int(
            validation_df[
                "included_in_final_directionality"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )
    main()
