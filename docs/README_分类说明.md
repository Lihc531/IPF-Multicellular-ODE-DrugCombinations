# 脚本分类说明（v0.5：加入主队列早期预处理）

本压缩包按稿件四个大模块整理用于正式分析/复现的脚本。脚本内容均按原文件字节级复制，不修改脚本本身。

## 本次新增

模块01新增 `00_主队列上游预处理`，用于补齐GSE136831从早期raw object到正式singlet/no-multiplet对象的上游溯源：

- `01_数据获取与raw_object_溯源`：项目设置、GEO下载、raw-object辅助构建；其中 `02_build_seurat_object.R` 因明确假定通用Matrix Market文件结构，只作为部分固定的上游溯源脚本。
- `02_QC与doublet_正式主线`：QC阈值与逐sample/library的scDblFinder。
- `03_标准化整合聚类_正式主线`：LogNormalize、HVG、Harmony、UMAP/邻接图、Louvain和marker。
- `04_注释与no_multiplet_正式主线`：marker-module/SingleR注释、建模导出及残余Multiplet清除。

正式主队列分支为 **LogNormalize + Harmony**。SCTransform支线和跳过doublet的旧版流程未混入正式主线，详见 `00_分类说明与脚本清单/nonfinal_or_uncertain_versions.md`。

## 四大模块

1. `01_单细胞数据获取_预处理与多细胞图谱构建`
2. `02_多细胞疾病进展建模与可干预疾病锚点识别`
3. `03_机制约束的候选药物组合计算优先级`
4. `04_稳健性_消融与正交验证分析`

## 核对文件

- `script_manifest.csv`：目标路径、原始来源、SHA256和归类理由。
- `source_hashes.sha256`：归档文件哈希，用于确认复制后脚本未被改写。
- `nonfinal_or_uncertain_versions.md`：未纳入正式主线的重复、旧版、可选支线及原因。

## 既有版本边界仍保持不变

- 动态基因模块采用 `15_celltype_dynamic_gene_modules_axis_v2_with_true_dynamic_gene_csv_FIXED2.R`。
- `B! 21a` 与 `21a2` 哈希相同，仅保留 `21a2`。
- 正式药物候选身份以Module 08冻结的 `pair_vs_best_single.csv` 为结果源；Flow Matching属于补充排序/敏感性分析。
- Vina采用V3.1.1正式流程；Bortezomib/PSMB5为共价机制，不以标准非共价Vina替代正式共价验证。
