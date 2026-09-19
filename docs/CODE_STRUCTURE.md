# 代码结构与维护指南（整理归类版）

> 本文件说明重组后的目录约定与运行方式，供后续更新参考。
> 仓库根已有的 `README.md`（740 行）是竞赛申报主文档，`docs/competition_readme.md` 是原 `玻尔比赛/README.md`。

## 1. 目录约定

```
marine-antifouling-ml/
├── src/                       # 单一真实来源：训练 / 预测 / 分析库(非部署)
│   ├── antifouling_platform.py  主训练+分析流程(原 玻尔比赛/antifouling_platform.py)
│   ├── improve_models.py        v7 模型提升(含 LightGBM)
│   ├── predict.py                CLI 预测接口
│   └── smart_predict.py          智能材料解析+多组分预测引擎 v6
├── deployments/               # 可运行的部署单元(自包含，改动请同步两份)
│   ├── deploy_job/              nssm/服务器部署版(app.py + model.pkl + search_engine.py + synthesis_routes.py)
│   └── hf_space/                HuggingFace Space 变体(与 deploy_job 仅差 ~43 行)
├── data/
│   ├── raw/dataset.csv          主训练集(1158 条，含数据坑，见 §3)
│   └── literature/              literature_pdfs.csv / literature_pdfs_enriched.csv(文献元数据)
├── models/                    # 模型与结果(*.pkl 已被 .gitignore 忽略)
│   ├── models.pkl / models_v7.pkl
│   └── results/results_v7.json
├── experiments/archive/       # 历史迭代归档 v1_initial / v3_iter / v4_iter / v5_iter / misc
├── docs/                      # 文档与图(blind_test_report / methodology / figures / competition_readme)
├── pdfs/                      # 391 篇文献 PDF(不入库)
├── requirements.txt           # 根级依赖清单(新增)
└── .gitignore                 # 忽略 pkl/pyc/log/pdf(新增)
```

## 2. 运行方式

```bash
pip install -r requirements.txt      # rdkit 建议用 conda 装
python src/antifouling_platform.py   # 重新生成数据集+训练+盲测报告+图(输出到 models/results)
python src/improve_models.py          # 跑 v7 模型提升(读取 data/raw/dataset.csv)
python src/predict.py "C[Si](C)(C)O[Si](C)(C)C"   # 单分子预测
```

部署单元直接运行：
```bash
python deployments/deploy_job/app.py   # Gradio 8502；model.pkl / literature_db 与 app.py 同目录
```

## 3. ⚠️ 数据坑（务必带）

- `data/raw/dataset.csv` 共 1158 条，但仅约 **86 条**是严格一致的 SMILES 描述符，
  其余为「模板 + 高斯噪声」增强样本（`is_original` 列标记真假）。
- 因此随机切分的交叉验证 / 盲测 R² 会**虚高**（同一模板变体同时落在训练/测试集 → 信息泄漏）。
- `antifouling_platform.py` 与 `improve_models.py` 当前仍用随机 `train_test_split`；
  严谨评估需按模板/来源做分组验证（Leave-One-Group-Out，参见 `experiments/archive/v4_iter/v4_logo_results.json`）。
- 详见后续轮次的「交叉验证更新预测准确度」任务。

## 4. 服务器部署路径变更提醒

原部署路径 `玻尔比赛/deploy_job/app.py` 已移至 **`deployments/deploy_job/app.py`**。
下一次在 Lighthouse 服务器 `git pull` 并更新 nssm 启动项时，需把启动命令中的
脚本路径同步改为 `deployments/deploy_job/app.py`（或重新打包 Release）。

## 5. 已知重复 / 待清理

- `deployments/deploy_job/app.py` 与 `deployments/hf_space/app.py` 高度重复，
  仅部署目标差异。后续可抽公共模块，仅保留差异配置。
- `synthesis_routes.py`(2450 行) 在两部署单元各一份，建议抽为 `src/` 单一副本后部署时软链/复制。
