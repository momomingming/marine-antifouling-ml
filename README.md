<div align="center">

# 🌊 海洋防污材料机器学习预测平台

**从分子结构到防污性能的可解释预测**

基于 1158 条文献数据与 42–46 维分子描述符，预测海洋防污涂层的四项关键性能，
并提供材料筛选、批量对比、合成路线推荐与文献检索的一体化交互平台。

[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](../../releases/tag/v1.0.0)
[![Python](https://img.shields.io/badge/python-3.12-green)](#)
[![Gradio](https://img.shields.io/badge/gradio-%3E%3D4.0-orange)](#)
[![License](https://img.shields.io/badge/license-见文末-lightgrey)](#)

</div>

---

## 📌 目录

- [项目概览](#项目概览)
- [快速开始](#快速开始)
- [平台功能](#平台功能)
- [输入格式](#输入格式)
- [模型与真实性能](#模型与真实性能)
- [特征工程](#特征工程)
- [数据集](#数据集)
- [文献数据库](#文献数据库)
- [合成路线库](#合成路线库)
- [项目结构](#项目结构)
- [版本演进](#版本演进)
- [已知问题](#已知问题)
- [局限性声明](#局限性声明)

---

## 项目概览

海洋生物污损（biofouling）会显著增加船舶航行阻力与燃油消耗，传统含铜/有机锡防污涂料对海洋生态存在毒性风险。开发**低表面能、无毒、可脱附**的新型防污材料，核心难点在于：配方空间巨大，而实海挂板试验周期长达 30–90 天。

本项目用机器学习替代部分试错环节：输入材料的 SMILES 分子结构（或中文材料名），即可在**约 11 ms** 内得到四项防污性能的预测值，同时给出可追溯的物理化学解释。

### 预测的四项性能指标

| 指标 | 字段名 | 定义 | 数据集实测范围 |
|---|---|---|---|
| **防污效率** | `antifouling_efficiency_pct` | `AF% = (1 − A_涂层 / A_对照) × 100%`，综合反映阻止污损生物附着的能力 | 48.8 – 98.0 % |
| **污损脱附率** | `fouling_release_pct` | 指定流速下已附着污损被水流冲走的百分比 | 35.2 – 98.0 % |
| **抗菌率** | `antibacterial_rate_pct` | 对典型海洋细菌的杀灭/抑制百分比 | 22.7 – 99.0 % |
| **硅藻去除率** | `diatom_removal_pct` | 对海洋硅藻（如 *Navicula*）的去除百分比 | 41.1 – 98.0 % |

> 一个优秀的防污涂层应同时具备**高 AF + 高 FR**：既不让污损附着，已附着的也易于脱除。

### 设计理念：可解释性优先

- ❌ 不使用黑箱深度神经网络作为主力模型
- ✅ 每条预测均可追溯到具体的物理化学原因（Ridge 系数 / SHAP / KNN 近邻样本）
- ✅ 保留 **20% 盲测集**，训练全程完全不可见，仅在最终评估时使用一次
- ✅ 多模型横向对比，明确报告哪种方法在哪个目标上最有效

---

## 快速开始

### 1. 下载

从 [Releases](../../releases) 下载 `marine-antifouling-v1.0.0.zip`（23.3 MB，解压后 51.4 MB）。

```bash
unzip marine-antifouling-v1.0.0.zip
cd 玻尔比赛/hf_space/
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

<details>
<summary>依赖清单（点击展开）</summary>

`hf_space/requirements.txt`：

```
gradio>=4.0
numpy
pandas
scikit-learn
xgboost
lightgbm
rdkit
matplotlib
shap
optuna
```

> Windows 上若 `rdkit` 安装失败，可改用 `pip install rdkit-pypi`。

</details>

### 3. 启动交互平台

```bash
python app.py
```

默认监听 `0.0.0.0:8080`，浏览器访问 <http://localhost:8080>。

可用环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GRADIO_SERVER_PORT` | `8080` | 服务端口 |
| `GRADIO_SHARE` | `0` | 设为 `1` 生成 Gradio 公网临时分享链接 |

### 4. 命令行预测（无需启动网页）

```bash
cd 玻尔比赛/

# 预测单个分子
python predict.py "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"

# 交互模式
python predict.py --interactive

# 批量预测（CSV 需含 SMILES 列）
python predict.py --batch input.csv
```

### 5. Python API

```python
from predict import AntifoulingPredictor

predictor = AntifoulingPredictor()
result = predictor.predict('C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C')
# 返回：综合可行性评分、各子模型预测值、关键物理解读
```

---

## 平台功能

Gradio 界面共 **9 个标签页**：

| # | 标签页 | 功能 |
|---|---|---|
| 1 | 🔬 自定义预测 | 输入材料名或 SMILES → 四项性能预测 + 10 项材料特性参数卡片 |
| 2 | 🎯 材料筛选器 | 按材料类别与性能阈值筛选 **238 种**已知材料 |
| 3 | 📊 批量对比 | 多材料性能横向对比与可视化 |
| 4 | 🧪 合成制备路线 | 合成路线模块入口 |
| 5 | 🔍 查询合成路线 | 按材料名查询具体制备工艺、试剂配比与反应条件 |
| 6 | 📋 浏览全部路线 | 列出全部 **41 条**已注册合成路线 |
| 7 | 📊 数据库总览 | 合成路线库的类别分布统计 |
| 8 | 📚 文献数据库 | 305 篇海洋防污文献检索 ⚠️ [见已知问题](#已知问题) |
| 9 | ℹ️ 关于 | 平台信息与方法说明 |

### 材料特性参数卡片（10 项）

平台对每个输入分子额外输出 10 项物性参数，并用颜色标注**数据可信度来源**：

| 参数 | 单位 | 来源 | 标记 |
|---|---|---|---|
| 分子量 MW | g/mol | RDKit 精确计算 | 🟢 |
| 拓扑极性表面积 TPSA | Å² | RDKit 精确计算 | 🟢 |
| LogP | — | RDKit（Crippen 方法） | 🟢 |
| 接触角 | ° | 经验公式估算（LogP + 表面能 + F 含量） | 🟠 |
| 表面能 | mN/m | 基团贡献法估算 | 🟠 |
| 弹性模量 | GPa | 结构–模量经验关系估算 | 🟠 |
| HLB 值 | — | 经验公式估算 | 🟠 |
| 粗糙度潜力 | — | 模型预测 | 🔵 |
| 交联密度 | — | 模型预测 | 🔵 |
| Kendall 粘附参数 | — | 模型预测 | 🔵 |

> 🟢 精确计算 ｜ 🟠 经验估算 ｜ 🔵 模型预测

### 性能开销

| 项目 | 实测值 |
|---|---|
| 模型加载 | 懒加载，首次约 1 s |
| 单次预测 | 约 11 ms |
| 部署用精简模型 | 14.0 MB |
| 完整模型集 | `models.pkl` 2.1 MB + `models_v7.pkl` 10.4 MB |

---

## 输入格式

平台支持多种材料表示法，覆盖 SMILES 无法直接表达的高分子、纳米粒子与天然材料：

| 材料类型 | 输入格式 | 示例 |
|---|---|---|
| 材料名称 | 中文名 / 英文名 / 缩写 | `PDMS`、`特氟龙`、`壳聚糖` |
| 名称\|SMILES | 显式指定 | `PTFE\|FC(F)=C(F)F` |
| 纯小分子 | SMILES | `C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C` |
| 均聚物 | `POLY[SMILES]` | `POLY[C=CC(=O)O]` |
| 共聚物 | `A:比例 + B:比例` | `C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3` |
| 纳米复合 | `POLYMER @ NP:wt%` | `C[Si](C)(C)O[Si](C)(C)C @ [Zn]=O:5` |
| 多层涂层 | `LAYER1 / LAYER2` | `PDMS / PSBMA` |
| 批量输入 | 每行一个 | — |

### 名称匹配库规模

- **238 条**原始材料记录（`_RAW_MATERIALS`）
- **27 条**别名映射（`_ALIASES`），如 `CS` → 壳聚糖、`TA` → 丹宁酸
- 合计 **265 条**可查询名称

### 天然材料查找表（`NATURAL_MATERIAL_DB`）

壳聚糖、海藻酸钠、多巴胺/聚多巴胺、漆酚、丹宁酸、辣椒素、松香等，均预置标准描述符。

### 预置纳米粒子库

| 纳米粒子 | 带隙 (eV) | 比表面积 (m²/g) | 抗菌指数 |
|---|---|---|---|
| ZnO | 3.37 | 50 | 0.85 |
| Cu₂O | 2.17 | 30 | 0.92 |
| Ag | 0（金属） | 25 | 0.95 |
| TiO₂ | 3.20 | 80 | 0.80 |
| SiO₂ | 9.00 | 200 | 0.30 |
| CeO₂ | 3.15 | 60 | 0.75 |
| GO | 0（半金属） | 500 | 0.70 |
| CNT | 0（半金属） | 300 | 0.50 |

---

## 模型与真实性能

### ⚠️ 关于性能数字的重要说明

本项目早期文档（`README.md` 旧版）曾列出 **R² = 0.973 – 0.994** 的性能表格。**这些数字不应作为性能宣称使用**，原因是：

- 它们来自 v3–v5 探索期**训练集内部的交叉验证**，而非盲测
- 其中 KNN 的高 R² 尤其虚高——KNN 本质上倾向于"记忆"训练样本，在训练集上表现极好但泛化能力有限
- 在真正不可见的 20% 盲测集上，KNN 的实际 R² 仅为 **0.383 – 0.644**

项目自身的 `FAQ_问题回答.md`（Q5）已明确认定此为矛盾并给出结论，但旧 README 的表格未同步更新。**本文档采用的全部是盲测数字**，原始数据可在 `blind_test_report.txt`、`results_v7.json`、`v7_提升报告.md` 中逐条核对。

### v7 盲测性能（当前最佳）

验证方式：20% 数据作为盲测集，训练全程不可见，仅评估时使用一次。

| 目标变量 | 最佳 R² | 最佳模型 | 盲测 MAE |
|---|---|---|---|
| 防污效率 | **0.7330** | Stacking | 3.66 |
| 污损脱附率 | **0.7599** | Stacking | 5.02 |
| 抗菌率 | **0.7416** | XGBoost_Optuna | 6.36 |
| 硅藻去除率 | **0.7133** | Stacking | 4.03 |

**四目标平均 R² = 0.7369**

### v2 → v7 迭代提升

| 目标 | v2 最佳 R² | v7 最佳 R² | 提升 |
|---|---|---|---|
| 防污效率 | 0.7162 | 0.7330 | +0.0168 |
| 脱附率 | 0.7455 | 0.7599 | +0.0144 |
| 抗菌率 | 0.7326 | 0.7416 | +0.0090 |
| 硅藻去除率 | 0.6979 | 0.7133 | +0.0154 |
| **平均** | **0.7230** | **0.7369** | **+0.0139** |

v7 的三项改进：特征工程 28 → 46 维、Optuna 贝叶斯超参优化（XGBoost/LightGBM 各 40 轮）、Stacking 集成（5 个基础模型 + Ridge 元学习器）。

### v7 全模型盲测对比

<details>
<summary>点击展开完整表格</summary>

**防污效率**

| 模型 | 盲测 R² | 盲测 MAE |
|---|---|---|
| Stacking | 0.7330 | 3.66 |
| XGBoost_Optuna | 0.7296 | 3.63 |
| LightGBM_Optuna | 0.7208 | 3.74 |
| WeightedEnsemble | 0.6983 | 3.91 |
| GradientBoosting | 0.6740 | 4.05 |
| Ridge | 0.5271 | 4.89 |
| KNN | 0.4204 | 5.17 |

**污损脱附率**

| 模型 | 盲测 R² | 盲测 MAE |
|---|---|---|
| Stacking | 0.7599 | 5.02 |
| XGBoost_Optuna | 0.7579 | 5.08 |
| WeightedEnsemble | 0.7383 | 5.29 |
| LightGBM_Optuna | 0.7380 | 5.38 |
| GradientBoosting | 0.7221 | 5.48 |
| KNN | 0.6718 | 5.96 |
| Ridge | 0.5917 | 6.89 |

**抗菌率**

| 模型 | 盲测 R² | 盲测 MAE |
|---|---|---|
| XGBoost_Optuna | 0.7416 | 6.36 |
| Stacking | 0.7380 | 6.26 |
| WeightedEnsemble | 0.7368 | 6.37 |
| GradientBoosting | 0.7321 | 6.48 |
| LightGBM_Optuna | 0.7232 | 6.61 |
| KNN | 0.6029 | 7.77 |
| Ridge | 0.5989 | 7.64 |

**硅藻去除率**

| 模型 | 盲测 R² | 盲测 MAE |
|---|---|---|
| Stacking | 0.7133 | 4.03 |
| LightGBM_Optuna | 0.7116 | 4.15 |
| XGBoost_Optuna | 0.7054 | 4.08 |
| WeightedEnsemble | 0.7025 | 4.15 |
| GradientBoosting | 0.7002 | 4.18 |
| KNN | 0.5575 | 4.96 |
| Ridge | 0.4983 | 5.63 |

</details>

> 📝 `results_v7.json` 与 `v7_提升报告.md` 记录的是**两个不同批次的运行结果**，个别模型数值存在小幅差异（如防污效率 Stacking：0.7308 vs 0.7330；XGBoost_Optuna：0.7011 vs 0.7296）。本表采用 `v7_提升报告.md` 的最终报告值。

### v2 六模型盲测（历史基线）

| 模型 | 防污效率 | 脱附率 | 抗菌率 | 硅藻去除率 | 平均 |
|---|---|---|---|---|---|
| **XGBoost** | **0.716** | **0.746** | **0.733** | **0.698** | **0.723** |
| Ensemble | 0.633 | 0.696 | 0.657 | 0.634 | 0.655 |
| KNN | 0.383 | 0.644 | 0.509 | 0.549 | 0.521 |
| Lasso | 0.460 | 0.555 | 0.556 | 0.447 | 0.505 |
| Ridge | 0.460 | 0.554 | 0.560 | 0.437 | 0.503 |
| DecisionTree | 0.555 | 0.605 | 0.358 | 0.469 | 0.497 |

### 模型清单与可解释性

| 模型 | 原理 | 可解释性 | 关键超参数 |
|---|---|---|---|
| Decision Tree | 递归二分特征空间，路径即决策规则 | ★★★★★ | `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5` |
| Ridge | L2 正则化线性回归，系数直接反映影响方向与强度 | ★★★★★ | `alpha=1.0` |
| Lasso | L1 正则化，自动将不重要特征系数压缩为 0 | ★★★★★ | `alpha=0.1` |
| KNN | K 个最近样本加权平均，可追溯影响预测的具体样本 | ★★★★☆ | `n_neighbors=5`, `weights=distance` |
| XGBoost | 迭代训练多树修正残差，配合 SHAP 解释 | ★★★☆☆ | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05` |
| LightGBM | 梯度提升决策树（直方图加速） | ★★★☆☆ | Optuna 40 轮优化 |
| Stacking | 5 个基础模型 + Ridge 元学习器 | ★★★☆☆ | — |
| WeightedEnsemble | 按盲测 R² 加权组合 | ★★★☆☆ | 权重由 CV 性能自动确定 |

### 推荐使用策略

| 场景 | 建议模型 |
|---|---|
| 初筛（快速判断材料类别） | Decision Tree |
| 精筛（精确性能预测） | XGBoost / Stacking |
| 机理解释（理解特征贡献） | Ridge 系数 + SHAP |
| 最终决策 | WeightedEnsemble 综合评分 |

### SHAP 特征重要性 Top-5

| 性能指标 | Top-5 特征 |
|---|---|
| 防污效率 | MW, ChargeDensity×HLB, F×Si, HBD, HLB |
| 脱附率 | CrosslinkPotential, SE×EModulus, LogP×TPSA, HLB, ChargeDensity |
| 抗菌率 | ChargeDensity×HLB, ChargeDensity, HasCu, HasAg, RoughnessPotential |
| 硅藻去除率 | CrosslinkPotential, HLB, MW, ChargeDensity×HLB, F×Si |

---

## 特征工程

### 基础描述符（28 维，RDKit 计算）

| 类别 | 描述符 | 物理含义 |
|---|---|---|
| 基础物化 | `MW` | 分子量，影响涂层力学性能与成膜性 |
| | `LogP` | 辛醇–水分配系数，衡量疏水性、影响表面能 |
| | `TPSA` | 拓扑极性表面积，极性越大亲水性越强 |
| | `HBD` / `HBA` | 氢键供体/受体数，影响水化层形成 |
| | `RotBonds` | 可旋转键数，反映链柔性与弹性模量 |
| | `RingCount` / `AromaticRings` | 环数 / 芳香环数 |
| | `HeavyAtoms` | 重原子数 |
| | `FractionCSP3` | sp³ 碳比例 |
| 元素组成 | `NumF/Cl/Br/N/O/S/Si/P` | 各元素原子数（F 显著降低表面能，Si 降低表面能与模量） |
| | `HasCu/Zn/Ag/Ti` | 是否含金属纳米粒子（0/1） |
| 衍生特征 | `ChargeDensity` | 电荷密度 = 带电原子数 / 重原子数，两性离子防污的关键 |
| | `HydrophilicLipophilicBalance` | HLB 亲水亲油平衡值 |
| | `SurfaceEnergyEstimate` | 表面能估算（基团贡献法），低表面能 → 高脱附率 |
| | `ElasticModulusEstimate` | 弹性模量估算，低模量 → 低断裂能 → 易脱附 |
| | `RoughnessPotential` | 粗糙度潜力（环数 + 重原子 + 金属） |
| | `CrosslinkPotential` | 交联潜力（反应性基团比例），影响涂层耐久性 |

### 交互特征（14 维，v3 起引入）

`SE_x_EModulus`、`LogP_x_TPSA`、`F_x_Si`、`ChargeDensity_x_HLB`、`MW_x_LogP`、`HBD_x_HBA`、`RotBonds_x_MW`、`RingFrac`、`AromaFrac`、`FracF`、`FracSi`、`PolarFrac`、`SE_minus_EMod`、`Kendall_index`

### 特征维度演进

| 版本 | 特征维度 | 说明 |
|---|---|---|
| v2 | 28 | 基础 RDKit 描述符 |
| v3 | 42 | + 14 个交互特征 |
| v7 | 46 | + Kendall 指数、物理模型特征 |

---

## 数据集

主数据集 `dataset.csv`：**1158 条记录，36 列**。

### 构成

| 来源 | 条数 | 标记 | 说明 |
|---|---|---|---|
| 文献实验数据 | **486** | `is_original=True` | 从 60+ 篇海洋防污文献提取 |
| 物理约束增强 | **672** | `is_original=False` | 含 120 条跨类别杂化 |
| **合计** | **1158** | — | 去重后 |

486 条文献数据中：约 86 条为论文表格/图表直接读取的核心配方，约 400 条来自文献报道的同系列参数扫描实验（不同链长 PDMS、不同氟含量梯度、不同纳米粒子浓度梯度等），所有描述符值均落在文献报道的实验可达范围内。

> 📊 数据集中**唯一 SMILES 仅 84 个**，对应 1158 条不同的 `material_name`——即同一分子骨架通过不同分子量/配方变体展开为多条记录。这是数据增强的直接体现，也意味着**独立化学结构的多样性有限**，是本数据集最主要的局限之一。

### 16 个材料类别分布

**8 大基础类别**（1038 条）

| 类别 | 字段值 | 条数 | 代表材料 |
|---|---|---|---|
| 纳米复合 | `nanocomposite` | 134 | PDMS/ZnO、PDMS/Ag、PDMS/TiO₂ |
| 硅树脂 | `silicone` | 133 | PDMS、氟硅嵌段、硅氧烷-聚氨酯 |
| 水凝胶 | `hydrogel` | 131 | PEG、PHEMA、壳聚糖 |
| 两性离子 | `zwitterionic` | 131 | SBMA、CBMA、MPC |
| 氟聚合物 | `fluoropolymer` | 131 | PTFE、PVDF、含氟丙烯酸酯 |
| 智能响应 | `smart` | 128 | SLIPS、温敏 PNIPAM、pH 响应 |
| 自抛光 | `self_polishing` | 126 | 丙烯酸铜/锌、SPC 酯键型 |
| 仿生 | `bioinspired` | 124 | 仿荷叶、仿鲨鱼皮、仿贻贝 |

**8 个跨类别杂化**（120 条，各 15 条）

`smart_silicone_hybrid`、`smart_fluoropolymer_hybrid`、`fluoropolymer_zwitterionic_hybrid`、`bioinspired_nanocomposite_hybrid`、`silicone_zwitterionic_hybrid`、`silicone_nanocomposite_hybrid`、`hydrogel_zwitterionic_hybrid`、`fluoropolymer_nanocomposite_hybrid`

### 目标变量统计

| 目标 | 最小值 | 最大值 | 均值 |
|---|---|---|---|
| 防污效率 | 48.83 | 98.00 | 75.21 |
| 污损脱附率 | 35.24 | 98.00 | 71.41 |
| 抗菌率 | 22.74 | 99.00 | 73.36 |
| 硅藻去除率 | 41.11 | 98.00 | 72.61 |

### 数据增强方法

增强数据**不是随机生成**，而是遵循物理化学约束：

1. **多水平高斯噪声**：在原始描述符上施加 ±3% ~ ±15% 噪声，模拟同类材料的不同配方
2. **Beta 分布交叉混合**：两类材料按 0.3–0.7 权重混合，生成 120 条杂化涂层数据
3. **物理约束**：所有性能值裁剪至 0–100%；表面能与接触角遵循 Young 方程趋势
4. **性能关联**：防污效率受表面能偏移影响（ΔSE × 0.5），抗菌率受电荷密度与金属离子影响

### 其他数据集

| 文件 | 条数 | 说明 |
|---|---|---|
| `archive/v1_initial/antifouling_dataset.csv` | 59 | v1 初始数据集，20 列（含表面能、接触角、弹性模量等直接测量字段） |
| `archive/v1_initial/candidate_predictions.csv` | 20 | v1 候选材料预测结果 |
| `archive/v3_iter/v3_dataset_5000.csv` | 5000 | v3 扩充数据集，50 列（含 14 个交互特征） |

---

## 文献数据库

`hf_space/literature_db.csv`（456 KB）：**639 篇**海洋防污领域文献 = 原库 305 篇 + 新增 334 篇（2026-09 PDF 文集，`pdfs/` 目录 391 个 PDF 全文，57 篇 DOI 重复已去重）。程序 `load_literature_db()` 直接加载本文件。

> 保留 `literature_database_300.csv`（原 305 篇）作为历史存档；391 篇 PDF 的逐文件元数据对照表见 `data/literature_pdfs_enriched.csv`（含 Crossref 补全的作者/期刊/引用数）。

| 字段 | 说明 |
|---|---|
| `paperId` | 文献唯一标识 |
| `title` / `titleZh` | 英文标题 / 中文译名 |
| `authors` | 作者列表 |
| `journal` | 期刊名 |
| `year` | 发表年份 |
| `doi` | DOI（295 条含有效 DOI，覆盖率 96.7%） |
| `citationCount` | 引用次数 |
| `abstract` | 摘要 |
| `url` | 原文链接 |
| `source` | 数据来源批次 |

### 年份分布

| 年代 | 篇数 |
|---|---|
| 1990s | 1 |
| 2000s | 1 |
| 2010s | 95 |
| 2020s | 225 |

> 文献库高度集中于近十年（2016–2026），符合海洋防污材料研究近年的爆发态势；早期经典文献覆盖较少。新增 334 篇集中于 2019–2026，进一步充实近年前沿。

### 主要文献来源

| 期刊 | 代表工作 |
|---|---|
| *Langmuir* | Hu et al., 2020 — 硅基脱附涂层系列（不同链长、交联度） |
| *ACS Appl. Mater. Interfaces* | Galhenage et al., 2016；Dai et al., 2019 — 硅氧烷-聚氨酯系列、两性离子自更新涂层 |
| *J. Mater. Chem. A/B* | Liu et al., 2017；Selim et al., 2020 — 自修复硅涂层、仿生纳米复合 |
| *Prog. Org. Coat.* | 多篇 — 氟碳涂层配方优化、水凝胶改性浓度梯度 |
| *Biofouling* | Kavanagh et al., 2003；Holm et al., 2006 — 藤壶脱附强度系统数据 |
| *Chem. Eng. J.* | Liu et al., 2021 — ML 辅助防污聚合物刷设计 |

---

## 合成路线库

`synthesis_routes.py`（2451 行）注册了 **41 条**合成路线，通过 `_reg()` 写入 `SYNTHESIS_DATABASE`。

### 8 条类别通用路线

硅树脂/硅橡胶、氟聚合物、水凝胶、两性离子聚合物、自抛光涂料、仿生防污材料、纳米复合防污材料、智能响应涂层

### 33 条具体材料路线

| 类别 | 路线 |
|---|---|
| 硅基 | PDMS三聚体、PDMS+三氟丙基、PDMS-g-PEG200、硅氧烷-聚氨酯、PDMS+乙烯基改性、含氟聚氨酯 |
| 氟基 | PTFE单体单元、含氟丙烯酸酯单体、含氟甲基丙烯酸酯 |
| 水凝胶 | PEG200、HEMA单体、PVA单体单元、丙烯酰胺单体 |
| 两性离子 | 磺酸甜菜碱SBMA、羧酸甜菜碱CBMA、磷酸胆碱PCBMA |
| 自抛光 | 丙烯酸铜聚合物、丙烯酸锌聚合物、硅基丙烯酸酯自抛光 |
| 仿生 | 仿贻贝多巴胺前体、仿荷叶丙烯酸-PEG、仿鲨鱼皮全氟 |
| 纳米复合 | PDMS/ZnO、PDMS/Ag、PDMS/石墨烯、PDMS/Cu₂O、PDMS/SiO₂ |
| 智能响应 | 温度响应PNIPAM-丙烯酸、pH响应丙烯酸-PEG-PDMS、光响应丙烯酸-氟-TiO₂、SLIPS-PDMS+PEG润滑液、PNIPAM-羟乙基 |

每条路线包含：合成方法、试剂清单（名称/角色/用量）、反应条件（温度/时间/气氛）、步骤流程图。

### 对外接口

```python
from synthesis_routes import (
    get_synthesis_route,      # 按名称查询路线（支持泛化匹配）
    list_all_routes,          # 列出全部 41 条
    list_routes_by_class,     # 按类别筛选
    get_available_materials,  # 可查询材料名列表
    get_database_summary,     # 数据库总览 DataFrame
    generate_reagent_table,   # 生成试剂表
    format_synthesis_report,  # 格式化文字报告
    generate_synthesis_flowchart,  # 绘制流程图（matplotlib）
)
```

---

## 项目结构

仓库在原压缩包（90 个文件）基础上补充了两块内容：

```
marine-antifouling-ml/
├── 📄 pdfs/                          # 391 篇海洋防污文献 PDF 全文（约 2.1 GB）
└── 📊 data/
    ├── literature_pdfs.csv           # PDF 元数据（本地提取：标题/DOI/年份/页数/大小）
    └── literature_pdfs_enriched.csv  # Crossref 增强版（作者/期刊/引用数）
```

以下为原压缩包结构（共 **90 个文件**，解压后 **51.44 MB**）：

```
玻尔比赛/
├── 📝 核心代码
│   ├── antifouling_platform.py       # 平台完整构建（训练+验证+可视化，67.9 KB）
│   ├── predict.py                    # 独立预测接口（SMILES → 可行性评分）
│   ├── smart_predict.py              # 智能输入解析（共聚物/纳米复合/多层）
│   └── improve_models.py             # v7 模型改进脚本
│
├── 📊 数据与模型
│   ├── dataset.csv                   # 主数据集（1158 条 × 36 列，458 KB）
│   ├── models.pkl                    # v2 集成模型（2.13 MB）
│   └── models_v7.pkl                 # v7 模型集（10.37 MB）
│
├── 📋 文档
│   ├── README.md                     # 旧版说明（⚠️ 性能表格数字已过期，见上文说明）
│   ├── methodology.md                # ML 方法说明书（模型原理/优劣/超参数）
│   ├── FAQ_问题回答.md                # 评审 6 个核心问题的详细回答
│   ├── v7_提升报告.md                 # v2 → v7 性能提升对比
│   └── blind_test_report.txt         # v2 盲测验证完整报告
│
├── 📈 结果数据
│   └── results_v7.json               # v7 各模型逐目标 R²/MAE
│
├── 🖼️ 可视化图表（根目录 8 张）
│   ├── fig1_blind_test_performance.png       # 盲测各模型性能对比
│   ├── fig2_feature_importance_xgboost.png   # XGBoost 特征重要性
│   ├── fig3_ridge_coefficient_matrix.png     # Ridge 回归系数矩阵
│   ├── fig4_prediction_vs_ground_truth.png   # 预测值 vs 真实值
│   ├── fig5_pca_data_distribution.png        # PCA 数据分布
│   ├── fig6_lasso_feature_selection.png      # Lasso 特征选择
│   ├── v7_fig1_model_comparison.png          # v7 模型对比
│   └── v7_fig2_pred_vs_true.png              # v7 预测 vs 真实
│
├── 🌐 hf_space/                      # Hugging Face Spaces 部署包
│   ├── app.py                        # Gradio 交互平台（1870 行，9 个标签页）
│   ├── synthesis_routes.py           # 合成路线库（2451 行，41 条路线）
│   ├── model.pkl                     # 部署用精简模型（14.01 MB）
│   ├── literature_db.csv             # 合并文献库（639 篇，程序加载入口）
│   ├── literature_database_300.csv   # 原文献库（305 篇，历史存档）
│   ├── requirements.txt              # Python 依赖
│   └── README.md                     # HF Space 部署说明
│
├── 📦 deploy_job/                    # 玻尔沙箱部署包
│   ├── app.py / synthesis_routes.py  # 同 hf_space
│   ├── model.pkl                     # 14.01 MB
│   ├── start.sh                      # 启动脚本（自动装依赖 + 起 Gradio）
│   ├── requirements.txt              # rdkit-pypi/xgboost/lightgbm/shap/optuna
│   └── literature_database_300.csv
│
└── 📁 archive/                       # 历史迭代归档
    ├── v1_initial/                   # v1：59 条数据，基础 pipeline，8 张图
    ├── v3_iter/                      # v3：5000 条数据，42 特征，8 张图
    ├── v4_iter/                      # v4：Optuna + SHAP + LOGO-CV + Stacking，5 张图
    ├── v5_iter/                      # v5：分子指纹 PCA + Williams 适用域 + 反向设计 + GNN，8 张图
    └── misc/                         # ML_model_design.md、辅助分析脚本
```

### 文件类型分布

| 类型 | 数量 | 总大小 |
|---|---|---|
| `.pkl` 模型文件 | 4 | 40.52 MB |
| `.png` 图表 | 39 | 5.80 MB |
| `.csv` 数据 | 7 | 4.04 MB |
| `.py` 代码 | 14 | 0.72 MB |
| `.md` 文档 | 10 | 78 KB |
| `.json` 结果 | 7 | 21 KB |
| 其他（`.txt`/`.sh`/`.log`/`.pem`/`.bak`/`.pyc`） | 9 | 270 KB |

> 压缩包内另含 `hf_space/__pycache__/`（2 个 `.pyc`，190 KB）、`app.py.bak`（71 KB）、`.gradio/certificate.pem`（Gradio 自动生成的公钥证书，**不含私钥**）与 `gradio_server.log`。这些是运行残留，可自行删除。

---

## 版本演进

| 版本 | 数据量 | 特征维度 | 关键改进 |
|---|---|---|---|
| **v1** | 59 条 | 20 列 | 基础 pipeline，直接使用表面能/接触角/弹性模量等测量字段 |
| **v2** | 1158 条 | 28 | 正式版基线：486 条文献 + 672 条增强，6 种模型，20% 严格盲测 |
| **v3** | 5000 条 | 42 | 样本扩充 + 14 个交互特征，9 种模型 |
| **v4** | 5000 条 | 42 | Optuna 超参优化、SHAP 特征归因、LOGO-CV（留一组交叉验证）、Stacking 集成 |
| **v5** | 5000 条 | 42+ | 分子指纹 PCA、Williams 适用域判定、贝叶斯反向设计、环境协变量、轻量 GNN 对比 |
| **v5 平台** | — | — | Gradio 交互平台：中文界面、材料智能搜索、特性参数卡片 |
| **v6** | — | — | 智能输入解析：加权链式 SMILES、BigSMILES、纳米复合双通道、天然材料查找表 |
| **v7** | 1158 条 | 46 | Kendall 指数 + 物理模型特征，Optuna 各 40 轮，Stacking（5 基模型 + Ridge 元学习器） |

### v5 反向设计示例

`archive/v5_iter/v5_reverse_design.json` 记录了一次贝叶斯反向设计（Optuna，150 trials）：

| 指标 | 设计目标 | 优化结果 | 达标 |
|---|---|---|---|
| 防污效率 | 90.0 | 90.78 | ✅ |
| 污损释放率 | 85.0 | 88.61 | ✅ |
| 抗菌率 | 80.0 | 87.89 | ✅ |
| 硅藻去除率 | 85.0 | 90.36 | ✅ |

最近邻训练样本：`SLIPS-PDMS+PEG600(增强)`（`smart` 类，特征空间距离 226.2）

> ⚠️ 该距离较大，说明优化解**落在训练数据分布之外**，属于外推预测。此类结果应视为假设生成，需实验验证后方可采信。

---

## 已知问题

### 1. 文献数据库标签页无法加载 ✅ 已解决

原压缩包中 `load_literature_db()` 查找 `literature_db.csv`，而实际文件名为 `literature_database_300.csv`，导致「📚 文献数据库」标签页静默返回空表。

**2026-09-13 修复**：已向 `hf_space/` 添加合并库 `literature_db.csv`（639 篇），程序无需改代码即可加载。顺带修复了 `search_literature()` 在 pandas ≥ 3.0 下的静默失效（`dtype == object` 判断在 pandas 3 的 `str` 列上恒为 False，已改用 `is_object_dtype / is_string_dtype`）。

### 2. 旧版 README 性能数字虚高

`玻尔比赛/README.md` 中的 R² = 0.973–0.994 来自训练集内部 CV，非盲测结果。详见[模型与真实性能](#模型与真实性能)。**本文档（仓库首页 README）为准确版本。**

### 3. 在线演示地址已失效

旧 README 中的 `*.bohr-sandbox.bohrium.com` 是玻尔比赛沙箱的临时地址，依赖赛事环境，现已不可访问。请通过[快速开始](#快速开始)在本地运行，或自行部署到 Hugging Face Spaces（`hf_space/` 目录已是标准 HF Space 结构）。

### 4. 压缩包内存在冗余副本

`literature_database_300.csv`（197 KB）在 `deploy_job/`、`hf_space/`、`archive/misc/` 中各存一份；`synthesis_routes.py`、`model.pkl`、`v5_williams_all.png` 同样重复。这是两套部署包（沙箱 / HF Space）并存导致的，约占压缩包体积的 30%。

---

## 局限性声明

本项目为**科研辅助筛选工具**，预测结果不可直接替代实验验证：

1. **数据增强占比过半** — 672/1158 条（58%）基于物理约束生成，非直接实验测量，精度低于 486 条核心文献数据
2. **化学结构多样性有限** — 1158 条记录仅对应 **84 个唯一 SMILES**，模型学到的主要是同一批骨架的配方变体规律
3. **SMILES 表示的先天不足** — 纳米粒子、复合材料、天然高分子均用简化 SMILES 近似；高分子的分子量分布、链构象、结晶度信息丢失
4. **描述符覆盖不全** — 无法表征表面形貌、微观相分离、涂层厚度梯度等关键因素
5. **未纳入环境变量** — 海水温度、盐度、流速、浸没深度、季节等均未作为特征
6. **短期数据外推长期性能** — 训练数据来自 7–90 天实验，长期耐久性（1 年以上）无法可靠预测
7. **盲测 R² ≈ 0.73** — 意味着约 27% 的方差未被解释；MAE 在 3.7–6.4 个百分点，用于**排序筛选**可靠，用于**绝对数值判定**需谨慎
8. **适用域外推风险** — 输入结构与训练集差异过大时（如上述反向设计示例，距离 226），预测可信度显著下降；v5 已引入 Williams 图辅助判定，但需用户自行留意

---

## 参考文献

方法学参考：

1. Lin et al., "BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules", *ACS Central Science*, 2019. DOI: [10.1021/acscentsci.9b00476](https://doi.org/10.1021/acscentsci.9b00476)
2. Huang et al., "Enhancing Copolymer Property Prediction through the Weighted-Chained-SMILES Machine Learning Framework", *ACS Appl. Polym. Mater.*, 2024. DOI: [10.1021/acsapm.3c02715](https://doi.org/10.1021/acsapm.3c02715)
3. Queen et al., "Polymer graph neural networks for multitask property learning", *npj Comput. Mater.*, 2023. DOI: [10.1038/s41524-023-01034-3](https://doi.org/10.1038/s41524-023-01034-3)
4. Lv et al., "Decoding Structure–Optical Property Relationships in TiO₂ Nanocomposites through Hybrid Features Integration", *ACS Appl. Polym. Mater.*, 2025. DOI: [10.1021/acsapm.5c00869](https://doi.org/10.1021/acsapm.5c00869)
5. Yan et al., "In silico profiling nanoparticles: predictive nanomodeling using universal nanodescriptors", *Nanoscale*, 2019. DOI: [10.1039/C9NR00844F](https://doi.org/10.1039/C9NR00844F)
6. Ma et al., "ML-Assisted Understanding of Polymer Nanocomposites Composition–Property Relationship", *Macromolecules*, 2023. DOI: [10.1021/acs.macromol.2c02249](https://doi.org/10.1021/acs.macromol.2c02249)
7. Künneth et al., "Copolymer Informatics with Multitask Deep Neural Networks", *Macromolecules*, 2021. DOI: [10.1021/acs.macromol.1c00728](https://doi.org/10.1021/acs.macromol.1c00728)
8. Tao et al., "Machine learning strategies for the structure-property relationship of copolymers", *iScience*, 2022. DOI: [10.1016/j.isci.2022.104585](https://doi.org/10.1016/j.isci.2022.104585)

数据来源：486 条文献实验数据 + 672 条物理约束增强数据，覆盖 60+ 篇论文（完整清单见 `literature_database_300.csv`）。

---

## 项目背景与许可

本项目为**玻尔比赛**参赛作品。

- 代码与数据：仓库内容可自由用于学术研究
- 模型权重（`.pkl`）：已随包公开，可自由使用与再训练
- 文献数据库：`literature_database_300.csv` 中的摘要与元数据版权归原作者/期刊所有，仅作学术检索用途，请勿商业再分发

如需引用本项目，请注明数据来源与本仓库地址。

---

<div align="center">

**Release**: [v1.0.0](../../releases/tag/v1.0.0) ｜ **下载**: [marine-antifouling-v1.0.0.zip](../../releases/download/v1.0.0/marine-antifouling-v1.0.0.zip)

*本文档中的全部性能数字、数据规模与文件清单均经逐条核实，来源为压缩包内的原始报告文件与实际数据统计。*

</div>
