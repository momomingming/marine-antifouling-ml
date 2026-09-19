# 代码审查报告：仓库不足与改进点

> 通读 `src/`（antifouling_platform.py / improve_models.py / predict.py / smart_predict.py）、
> `deployments/`（app.py / search_engine.py / synthesis_routes.py）后整理。
> 按影响排序：**P0 必须修 / P1 建议修 / P2 可优化**。

## P0 — 影响结论可信度（最关键）

1. **盲测 R² 虚高（信息泄漏）**
   `dataset.csv` 1158 条中仅 ~86 条为严格一致 SMILES，其余是「模板 + 高斯噪声」增强样本
   （`is_original` 列标记）。`antifouling_platform.py` 与 `improve_models.py` 都用**随机**
   `train_test_split`/`KFold`，同一模板的多个变体同时落入训练集与测试集 → 模型记住模板，
   盲测 R² 好看但无泛化意义。
   → 严谨评估必须按**模板/来源分组**（Leave-One-Group-Out，参考 `experiments/archive/v4_iter/v4_logo_results.json`）。

2. **训练目标被增强数据污染**
   增强样本的性能值由启发式公式 + 噪声生成，模型学到的是「模板→属性」映射而非真实构效关系；
   真正支撑泛化的只有 ~86 条原始数据。需补**真实文献值外部验证集**。

3. **描述符含拍脑袋启发式且共线**
   `SurfaceEnergyEstimate / ElasticModulusEstimate / HydrophilicLipophilicBalance` 是人工经验公式，
   又作为特征输入模型 → 多重共线性，特征重要性解释存疑。

## P1 — 工程质量（本轮已修部分，其余待办）

4. **写死绝对路径 `/share/玻尔比赛`** → Windows 本地无法运行。✅ 本轮已改为相对路径（env 可覆盖）。
5. **仓库根缺 requirements.txt** → 依赖不透明、难复现。✅ 本轮已补根级 `requirements.txt`。
6. **目录混乱** → 训练脚本/归档/图/pkl/重复 app 散落。✅ 本轮已重组成 `src/ data/ models/ docs/ experiments/ deployments/`（见 `docs/CODE_STRUCTURE.md`）。
7. **重复代码**：`deploy_job/app.py` 与 `hf_space/app.py` 近重复（差 ~43 行）；`synthesis_routes.py` 两份。
   → 建议抽公共模块，部署时仅保留差异配置。
8. **`is_original` 未被用于防泄漏**：CV 时未排除增强样本。
9. **无单元测试 / CI**；同一套描述符计算在 `predict.py / smart_predict.py / app.py` 重复实现三遍。
10. **pickle 模型无版本/依赖锁定**：xgboost 跨大版本加载会静默出错（已知 ≥3 才对）。
11. **文献库 pkl 可能缺失**：`load_literature_db` 读 `literature_db.pkl`，但仓库只有 `.csv`，
    部署环境若无构建步骤会报错 → 需确认/补构建逻辑。

## P2 — 科学严谨与功能扩展

12. 描述符未覆盖表面形貌、微观结构、海水温度/盐度/流速等环境因素（方法说明书已自陈）。
13. 纳米粒子/复合材料用简化 SMILES（`[Zn]`、`[Cu]O[Cu]`）近似，描述符代表力有限。
14. `MODEL_DOCS` 提到 SHAP 局部解释，但代码未真正落地 SHAP。
15. 缺不确定性量化 / 主动学习（现有 consensus 仅看多模型 std）。
16. 架构为单体脚本集合，缺「数据→特征→CV→解释→部署」模块化流水线（任务 5 学优时可补）。

## 本轮已交付

- 目录重组成型（src/data/models/docs/experiments/deployments），全部 `.py` 编译通过。
- 修复两个写死路径；新增根级 `requirements.txt` 与 `.gitignore`。
- 新增 `docs/CODE_STRUCTURE.md`（结构 + 运行 + 服务器路径变更提醒）。

## 建议后续轮次优先级

1. **任务④交叉验证**：用 LOGO 重算诚实 R²，量化虚高幅度（直接回应 P0-1/2）。
2. **任务②文献库清洗**：合并两份文献 CSV，补 `literature_db.pkl` 构建（回应 P1-11）。
3. **任务⑤学优并入**：抽公共模块 + 加 SHAP/配置驱动流水线（回应 P2-14/16）。
