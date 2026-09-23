# 🌊 海洋防污材料机器学习预测平台

> **在线平台**: https://8080-default--doc-compiler-t7k6g.bohr-sandbox.bohrium.com

## 项目简介

本项目基于机器学习方法，构建了一个从分子结构（SMILES）预测海洋防污材料四大关键性能的交互式平台。科研人员只需输入材料名称或SMILES分子式，即可即时获取防污性能预测和材料特性参数。

### 核心能力

| 功能 | 说明 |
|------|------|
| **性能预测** | 输入SMILES → 预测防污效率、脱附率、抗菌率、硅藻去除率 |
| **材料特性** | 接触角、表面能、弹性模量、HLB值等10项关键参数 |
| **智能搜索** | 支持中文名/英文名/缩写自动匹配SMILES（293条材料库） |
| **批量对比** | 多材料性能横向对比 |
| **文献检索** | 305篇海洋防污文献数据库 |

---

## 在线使用

直接访问：**https://8080-default--doc-compiler-t7k6g.bohr-sandbox.bohrium.com**

### 平台功能标签页

| 标签页 | 功能 |
|--------|------|
| 🔬 自定义预测 | 输入材料名称或SMILES，预测防污性能+材料特性参数 |
| 🎯 材料筛选器 | 按分类/性能阈值筛选238个已知材料 |
| 📊 批量对比 | 多材料性能对比分析 |
| 📚 文献数据库 | 305篇海洋防污文献检索 |
| ℹ️ 关于 | 平台信息与方法说明 |

### 输入方式

```
# 方式1：中文名称（自动匹配SMILES）
PDMS
特氟龙
聚四氟乙烯

# 方式2：名称|SMILES格式
PDMS|C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C
PTFE|FC(F)=C(F)F

# 方式3：直接SMILES
C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C

# 批量输入：每行一个
PDMS
PTFE
PEG
```

---

## 技术架构

### 模型性能

| 性能指标 | 池化折外 R²（LOGO by SMILES） | MAE | 状态 |
|---------|:---:|:---:|------|
| 防污效率 | **0.592** | **4.46** | ✅ 已完成诚实口径验证 |
| 脱附率 | — | — | ⏳ 尚未按 LOGO 口径重算 |
| 抗菌率 | — | — | ⏳ 尚未按 LOGO 口径重算 |
| 硅藻去除率 | — | — | ⏳ 尚未按 LOGO 口径重算 |

> 验证方式：**Leave-One-Group-Out by SMILES**（84 个分子分组，同一分子的增强变体不跨训练/验证集），
> 报告**池化折外 R²**（而非逐折平均——后者在 LeaveOneGroupOut 下会因折内方差极小而产生伪影）。

> ⚠️ **口径更正**：本文档早期列出的「最佳 R² 0.973 / 0.979 / 0.994 / 0.977」来自
> **分层随机抽样切分**：同一分子的增强变体同时落入训练集与盲测集，构成**信息泄漏**，
> 该组数字已作废。以唯一完成重算的防污效率为例，随机切分虚高约 **0.11**（0.704 → 0.592）。
> 主要瓶颈是**真实独立分子仅 84 个**（58% 为增强样本）。
> 平台每次预测均同时输出 **95% 置信区间**与**适用域判定**，请勿仅看点预测值。

### 模型集成

| 模型 | 说明 |
|------|------|
| XGBoost_Optuna | Optuna贝叶斯超参优化 |
| LightGBM | 梯度提升决策树 |
| KNN | K近邻回归 |
| Ridge | 岭回归 |

预测结果取四模型均值。

### 特征工程（42维）

**基础特征（28个）**：MW、LogP、TPSA、HBD、HBA、RotBonds、RingCount、AromaticRings、HeavyAtoms、FractionCSP3、NumF/Cl/Br/N/O/S/Si/P、HasCu/Zn/Ag/Ti、ChargeDensity、HLB、SurfaceEnergy、ElasticModulus、RoughnessPotential、CrosslinkPotential

**交互特征（14个）**：SE×EModulus、LogP×TPSA、F×Si、ChargeDensity×HLB、MW×LogP、HBD×HBA、RotBonds/MW、RingFrac、AromaFrac、FracF、FracSi、PolarFrac、SE-EMod、Kendall_index

### 材料特性参数（10项）

| 参数 | 单位 | 来源 | 颜色 |
|------|------|------|------|
| 接触角 | ° | 估算（LogP+表面能+F含量） | 🟠 |
| 表面能 | mN/m | 估算 | 🟠 |
| 弹性模量 | GPa | 估算 | 🟠 |
| HLB值 | — | 估算 | 🟠 |
| 分子量 | g/mol | 精确计算 | 🟢 |
| 极性表面积(TPSA) | Å² | 精确计算 | 🟢 |
| 粗糙度因子 | — | 预测 | 🔵 |
| 交联密度 | — | 预测 | 🔵 |
| LogP | — | 精确计算 | 🟢 |
| Kendall粘附参数 | — | 预测 | 🔵 |

颜色说明：🟢实验值（精确） | 🟠估算值（经验公式） | 🔵预测值（模型） | ⚪未知

---

## 数据集

### 数据规模

| 版本 | 文献实验数据 | 增强数据 | 总计 |
|------|-----------|---------|------|
| v2 | 486 | 672 | 1158 |
| v3 | 238 | 4762 | 5000 |

### 8大材料分类

| 分类 | 数量 | 代表材料 |
|------|------|---------|
| 硅树脂 | 38 | PDMS、氟硅嵌段、硅氧烷-聚氨酯 |
| 氟聚合物 | 35 | PTFE、PVDF、含氟丙烯酸酯 |
| 水凝胶 | 35 | PEG、PHEMA、壳聚糖 |
| 两性离子 | 30 | SBMA、CBMA、MPC |
| 自抛光 | 25 | 丙烯酸铜/锌、SPC酯键型 |
| 仿生 | 25 | 仿荷叶、仿鲨鱼皮、仿贻贝 |
| 纳米复合 | 25 | PDMS/ZnO、PDMS/Ag、PDMS/TiO₂ |
| 智能响应 | 25 | SLIPS、温敏PNIPAM、pH响应 |

### 数据增强方法

- 多水平高斯噪声（3%~15%）
- Beta分布交叉混合
- 物理约束（性能值裁剪0-100%）

---

## 版本演进

| 版本 | 关键改进 |
|------|---------|
| **v2.0** | 基线模型：486条文献数据+672增强、28特征、6种模型 |
| **v3.0** | 样本扩充：238条原始+5000条总量、42特征、9种模型 |
| **v4.0** | 融合改进：Optuna超参优化、SHAP特征分析、LOGO-CV、Stacking集成 |
| **v5.0** | 高级特性：分子指纹PCA、Williams适用域、贝叶斯反向设计、环境协变量、轻量GNN |
| **v5.0平台** | 交互平台：中文界面、材料智能搜索、材料特性参数卡片、Gradio部署 |

---

## 项目文件结构

```
/share/玻尔比赛/
├── README.md                    # 本文件
│
├── 📝 核心代码
│   ├── antifouling_platform.py  # 平台完整构建代码（训练+验证+可视化）
│   └── predict.py               # 独立预测接口（输入SMILES→输出可行性）
│
├── 📊 数据与模型
│   ├── dataset.csv              # 完整数据集（1158条）
│   └── models.pkl               # 训练好的集成模型（2.2MB）
│
├── 📋 报告与文档
│   ├── methodology.md           # ML方法说明书（各模型原理/优劣/超参数）
│   └── blind_test_report.txt    # 20%盲测验证报告
│
├── 📈 可视化图表
│   ├── fig1_blind_test_performance.png      # 盲测各模型性能对比
│   ├── fig2_feature_importance_xgboost.png  # XGBoost特征重要性
│   ├── fig3_ridge_coefficient_matrix.png    # Ridge回归系数矩阵
│   ├── fig4_prediction_vs_ground_truth.png  # 预测值vs真实值
│   ├── fig5_pca_data_distribution.png       # PCA数据分布
│   └── fig6_lasso_feature_selection.png     # Lasso特征选择
│
├── 🌐 在线部署
│   └── hf_space/                # HuggingFace Spaces部署包
│       ├── app.py               # Gradio交互平台
│       ├── model.pkl            # 部署用精简模型
│       ├── requirements.txt     # Python依赖
│       └── literature_database_300.csv
│
├── 📦 归档 (archive/)
│   ├── v1_initial/              # v1初始版本（59条数据，基础pipeline）
│   ├── v3_iter/                 # v3迭代（5000条数据探索）
│   ├── v4_iter/                 # v4迭代（SHAP+Optuna）
│   ├── v5_iter/                 # v5迭代（GNN+Williams适用域）
│   └── misc/                    # 其他辅助文件
│
└── 📁 其他项目
    └── results/                 # DFT/MD计算结果（电池材料，非本项目）
```

---

## 本地运行

### 环境要求

```bash
pip install gradio numpy pandas scikit-learn xgboost lightgbm rdkit matplotlib shap optuna
```

### 启动平台

```bash
cd hf_space/
python app.py
# 访问 http://localhost:8080
```

### 命令行预测

```bash
# 预测单个分子
python predict.py "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"

# 交互模式
python predict.py --interactive

# 批量预测 (CSV需含SMILES列)
python predict.py --batch input.csv
```

### Python API

```python
from predict import AntifoulingPredictor

predictor = AntifoulingPredictor()
result = predictor.predict('C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C')
# 输出: 综合可行性评分、各模型预测、关键物理解读
```

---

## SHAP特征重要性（Top-5）

| 性能指标 | Top-5特征 |
|---------|----------|
| 防污效率 | MW, ChargeDensity×HLB, F×Si, HBD, HLB |
| 脱附率 | CrosslinkPotential, SE×EModulus, LogP×TPSA, HLB, ChargeDensity |
| 抗菌率 | ChargeDensity×HLB, ChargeDensity, HasCu, HasAg, RoughnessPotential |
| 硅藻去除率 | CrosslinkPotential, HLB, MW, ChargeDensity×HLB, F×Si |

---

## 部署信息

| 项目 | 详情 |
|------|------|
| 在线地址 | https://8080-default--doc-compiler-t7k6g.bohr-sandbox.bohrium.com |
| 托管平台 | 玻尔沙箱（Bohrium Sandbox） |
| 框架 | Gradio 6.25.0 |
| Python | 3.12 |
| 模型大小 | 14MB（精简版）/ 63MB（完整版） |
| 首次加载 | ~1s（懒加载模型） |
| 单次预测 | ~11ms |

---

## 参考文献

- 文献数据库：305篇海洋防污领域论文（见 `hf_space/literature_database_300.csv`）
- ML方法：Decision Tree + Ridge/Lasso + KNN + XGBoost + 加权集成
- 数据来源：486条文献实验数据 + 物理约束数据增强（共1158条）
- 历史版本迭代记录见 `archive/` 目录

---

*最后更新：2026-08-23*
