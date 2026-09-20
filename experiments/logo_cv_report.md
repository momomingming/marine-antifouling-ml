# 诚实交叉验证报告 — LOGO by SMILES

> ⚠️ **口径更正（2026-09-19 复核，务必先读本节）**
>
> 本报告第 3 节给出的 **LOGO R² = -0.594 是被聚合方式扭曲的数字，不能作为主指标**。
> 复核脚本 `experiments/verify_cv_protocol.py` 用同一模型配置（300 树）算清了两种口径：
>
> | 切分方式 | 逐折平均 R² | 池化 R²（标准） | 池化 MAE |
> |---|---|---|---|
> | 随机 5 折（含泄漏） | 0.703 | **0.704** | 3.728 |
> | LOGO by SMILES | **-0.594** | **+0.592** | 4.460 |
>
> - **逐折平均**：LeaveOneGroupOut 每一折 = 一个分子的约 14 条增强变体，折内 y 方差极小，
>   模型无法预测"分子内部的增强噪声"，每折 R² 必然崩 → 平均为负。这是**伪影**。
> - **池化**：把所有折外预测拼起来算一个 R²（sklearn `cross_val_predict` + `r2_score` 的标准做法），
>   衡量跨分子泛化能力 → **0.592**。
>
> ✅ **对外应报告：LOGO 池化 R² = 0.592，MAE = 4.46。**
> ❌ 不应再引用 -0.594 / "虚高 185%" / "模型比猜均值还差" 等说法。
>
> 修正后的真实结论：信息泄漏**确实存在但幅度有限**（R² 0.704 → 0.592，虚高仅 +0.112；
> MAE 3.73 → 4.46）。模型**具备真实的跨分子泛化能力**，主要瓶颈是 84 个真实分子太少。
> 以下正文保留原始记录供追溯，但请以本节口径为准。

> 生成脚本: `experiments/logo_cv.py`（原始）/ `experiments/verify_cv_protocol.py`（口径复核）
> 数据: `data/raw/dataset.csv` (1158 行 × 36 列)
> 标签: `antifouling_efficiency_pct` (防污效率 %)

## 1. 背景 / 数据坑

`dataset.csv` 共 1158 行，但 **唯一 SMILES 仅 84 个**。
- `is_original=1` 标记 486 行（原始样本）
- 增强样本 672 行，占总样本 **58.0%**（同一 SMILES 的「模板+噪声」变体）

旧流程（`src/improve_models.py` 各版本 / v7）用**随机 KFold** 切分，
同一 SMILES 的增强变体同时落在训练集与测试集 → 模型"背住"了这些样本 →
报告的 R² 好看但**不可信**（信息泄漏 / data leakage）。

## 2. 方法

- 分组键: `SMILES` 字符串 → **84 组**，同一分子的所有变体同进同出
- 对比:
  - **A) 随机 5-fold KFold** — 旧做法，含泄漏（虚高基准）
  - **B) LOGO (Leave-One-Group-Out) by SMILES** — 诚实验证
- 模型: Ridge / RandomForest / XGBoost（统一 `random_state=42`）
- 指标: R² / MAE / RMSE

## 3. 结果

| 验证方案 | 模型 | R² | MAE | RMSE |
|---|---|---|---|---|
| 随机 KFold（旧/虚高） | Ridge | 0.500 | 4.992 | 6.365 |
| 随机 KFold（旧/虚高） | RandomForest | 0.693 | 3.810 | 4.980 |
| 随机 KFold（旧/虚高） | XGBoost | **0.703** | 3.728 | 4.909 |
| LOGO by SMILES（诚实） | Ridge | -1.518 | 5.373 | 6.345 |
| LOGO by SMILES（诚实） | RandomForest | -0.588 | 4.576 | 5.401 |
| LOGO by SMILES（诚实） | XGBoost | **-0.594** | 4.480 | 5.354 |

### 结论
- XGBoost: 随机 KFold R²=**0.703** vs LOGO R²=**-0.594**
- **信息泄漏导致的 R² 虚高幅度 ≈ 1.297（约 +185%）**
- 诚实 R² 为负 → **模型在未见过的独立分子上无法泛化，预测不可信**
- 旧报告（含 `docs/blind_test_report.txt`、`docs/v7_提升报告.md`）的"高准确度"结论**作废**

## 4. 根因

1. **训练目标被增强数据污染**: 58% 样本是启发式公式+噪声生成的性能值，非真实测量
2. **随机切分信息泄漏**: 同分子变体跨训练/测试，模型记忆而非学习构效关系
3. **描述符含启发式共线项**: `SurfaceEnergyEstimate`/`ElasticModulusEstimate`/`HLB` 为拍脑袋公式且彼此共线，特征解释存疑

## 5. 行动建议（按优先级）

1. **立即将验证协议改为 LOGO by SMILES**（或 LOCO by scaffold），禁用随机切分作为准确度宣称依据
2. **扩充真实独立样本**（不同 SMILES 的实测数据）— 这是任务③扩库（CrossRef / ScienceDirect 抓文献元数据）的核心价值，当前卡网络
3. **重训模型**: 仅用 `is_original=1` 的真实样本 + 新扩充独立数据；增强样本仅作辅助
4. **平台前端**: 隐藏/标注旧虚高 R²，改为展示 LOGO 诚实指标（待代码推送后改）

## 6. 复现

```bash
python experiments/logo_cv.py
# 依赖: pandas, scikit-learn, xgboost (本机 venv: C:/Users/HJ/.workbuddy/binaries/python/envs/default)
```
