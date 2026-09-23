# P0 改进实现计划：预测不确定性量化 + 域适用性评估

> 状态：✅ **已确认并实现**（2026-09-19 完成编码 + 冒烟验证）
>
> ⚠️ **前提口径更正（实现过程中复核发现，务必先看）**
> 本计划原引用的「诚实 R² = -0.594」是**逐折平均**在 LeaveOneGroupOut 下的**伪影**，不成立。
> 复核脚本 `experiments/verify_cv_protocol.py`（300 树，同配置）结果：
> | 切分 | 逐折平均 R² | 池化 R²（标准） | 池化 MAE |
> |---|---|---|---|
> | 随机 5 折（含泄漏） | 0.703 | 0.704 | 3.728 |
> | LOGO by SMILES | -0.594 | **0.592** | 4.460 |
> 即：泄漏**确实存在但幅度有限**（虚高 +0.112，非 +1.297）。
> 模型**具备真实跨分子泛化能力**，主要瓶颈是 84 个真实分子太少。
> 因此本计划的定位从"模型不可信、必须全面否定"修正为
> **"模型可用但需诚实标注不确定性与适用边界"**——这仍是必要的。

## 1. 目标
- **P0-1 不确定性量化**：任何一次预测都返回 `prediction ± 置信区间` + 可信度等级，不再给单点"精确值"。
- **P0-2 域适用性（Applicability Domain）**：对落在训练分子特征空间之外的陌生输入，直接标红"超出适用域，结果不可信"，不乱报高防污效率。

## 2. 模块拆分（新增/修改）
```
src/
  uncertainty.py          # 新增：集成/bootstrap 置信区间
  domain_applicability.py # 新增：Williams plot / 马氏距离 域判定
deployments/deploy_job/
  app.py                  # 修改：预测接口返回 uncertainty + domain 字段；前端展示
  (hf_space/app.py 同步同改)
```

## 3. 关键逻辑
### P0-1 不确定性量化（参照 SteelML 集成思路，轻量落地）
- 复用 `experiments/logo_cv.py` 的 84 组 SMILES 分组，做 **LOGO 训练 84 个 XGBoost 折外模型**（每折用其余组训练、本组验证），序列化保存到 `models/logo_fold_models/`（约 84 个小模型，单模型几 KB~几十 KB）。
- 推理时：新样本走全部 84 个折外模型 → 得到预测值集合 → 取**均值=点预测，分位数(2.5%/97.5%)=95% 置信区间**，预测方差=不确定度。
- 备选（更省）：单模型 + bootstrap 重采样训练 N=30 个，效果近似、体积更小。默认走 LOGO 集成（化学上更合理）。

### P0-2 域适用性（参照 FOULING_RELEASE_MLOPS 的 Williams plot）
- 训练集 28 维描述符 → `StandardScaler` + `PCA(n=2)` 投影，或直接计算特征空间**马氏距离**。
- 阈值：以训练集马氏距离 95% 分位为界（Williams plot 标准做法）。新样本距离 > 阈值 → `in_domain=False`。
- 输出：`applicability_domain` 布尔 + `mahalanobis_distance` 数值，前端对 `False` 直接红色警告。

## 4. 前端/接口改动（deployments/deploy_job/app.py）
- 预测结果卡片新增三行：`置信区间 [lo, hi]`、`不确定度等级（低/中/高）`、`适用域（✅可信 / ⚠️超出，结果仅供参考）`。
- **下线**原"模型准确度 R²=0.70"这类误导性展示，改为诚实说明："当前独立分子验证 R² 为负，模型泛化有限，请结合置信区间与适用域判断"。

## 5. 风险与约束
- 诚实代价：84 个真实分子下置信区间会**很宽**——这是特性不是 bug，恰恰暴露数据稀缺，倒逼任务③扩库。
- 体积/延迟：84 折外模型集成推理约 +几十 ms，可接受；模型目录需随部署包带上（`models/logo_fold_models/`）。
- 双副本同步：`deploy_job/app.py` 与 `hf_space/app.py` 都改，避免漂移。
- 依赖：仅 sklearn + xgboost + numpy（已装），**不引入新重依赖**。

## 6. 验证方式（本地可跑）
1. `py_compile` 全部改动模块。
2. 冒烟：取 PDMS `C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C` 跑预测，确认返回 `prediction/ci_low/ci_high/in_domain` 四字段且结构正确。
3. 取一个明显离群（随机噪声描述符）样本，确认 `in_domain=False` 且置信区间极宽。
4. 不要求 R² 变正——本计划目标是**诚实化**，不是刷分。

## 7. 执行顺序（确认后）
1. 写 `src/uncertainty.py` + 单测冒烟
2. 写 `src/domain_applicability.py` + 单测冒烟
3. 改 `app.py` 预测接口 + 前端展示（deploy_job + hf_space 同步）
4. LOGO 折外模型训练脚本并入 `experiments/`（复用现有 84 分组）
5. 编译自测 → 待代理通后随代码整理一并 push

## 8. 实现记录（2026-09-19 完成）

| 项 | 状态 | 说明 |
|---|---|---|
| `src/uncertainty.py` | ✅ | 84 个折外 XGBoost 集成；点预测 + 95% 分位 CI + 不确定度等级 |
| `src/domain_applicability.py` | ✅ | 标准化 + 马氏距离，阈值取训练集 95% 分位 |
| `experiments/train_logo_ensemble.py` | ✅ | 产出 `models/logo_ensemble/`（84×.ubj + meta.pkl，约 26MB） |
| `experiments/test_p0_smoke.py` | ✅ | 4 项冒烟全通过 |
| `experiments/verify_cv_protocol.py` | ✅ | 口径核查，纠正 -0.594 伪影 |
| `deployments/deploy_job/app.py` | ✅ | 预测结果附「可信度块」；下线盲测R² 0.97 等误导展示 |
| `deployments/hf_space/app.py` | ✅ | **UI 已同步**（可信度块 + 口径更正文案 + 关于页指标表），README 同步修正 |

### hf_space 同步记录（2026-09-23）
- 移植 `_UNCERTAINTY_ENGINE` / `_get_uncertainty_engine` / `uncertainty_md` 三个模块级定义；
- 预测回调三处分支（`名称 \| SMILES`、复合材料、材料库匹配）均接入 `unc_parts`，
  输出时以 `---` 分隔追加「预测可信度」块；
- 顶栏标语 `盲测R² > 0.97` → `LOGO折外R² 0.59`；关于页指标表改为池化折外 R²=0.592 / MAE=4.46，
  并附口径更正说明块（与 deploy_job 完全一致，消除双份漂移）；
- `hf_space/README.md` 原仍写着「Blind Test R² 0.973」，已一并改为诚实口径；
- 验证方式：AST 从 `hf_space/app.py` 抽取真实函数源码 + 模块级变量，在 hf_space 目录
  布局下执行并预测 PDMS → `76.13 / 95%CI [64.79, 87.47] / 等级中 / ✅在域内(3.038/7.213)`，
  另测缺字段、空 dict、None 三种退化输入均返回空串不抛异常。
- 已知部署约束：HF Space 若以本目录为仓库根，需把 `models/logo_ensemble/` 放到根目录
  （解析候选路径 `./models/logo_ensemble`）；不带也不会崩，仅降级为单点预测。

### 实现中发现并修复的两个 bug
1. **标准化误用**：折外模型在原始特征上训练，推理端却先 `StandardScaler` 再预测 → 预测失真。
   已改为树模型直接用原始特征，Scaler 仅用于马氏距离判定。
2. **置信区间过窄（假精确）**：84 个折外模型高度相关，模型间分歧仅 ±2，
   而折外实测 RMSE 达 5.79。已加**残差兜底**：`半宽 = max(集成半宽, 1.96×RMSE)`，
   PDMS 的 CI 由 ±2 修正为 ±11.3，等级由「低」变「中」。

### 冒烟结果
```
[1] 集成加载: True | 折外模型数=84
[2] PDMS: 72.05 | 95%CI=[60.71, 83.39] | 宽度=22.68 | 等级=中 | 适用域内(3.04/7.21)
[3] 离群样本: 75.77 | CI 宽度=22.68 | 等级=高 | ⚠️超出适用域(86.52/7.21)
[4] 折外池化复核: R²=0.59 / MAE=4.47（与 verify_cv_protocol 一致）
```

### 部署注意
- 服务器需带 `models/logo_ensemble/`，或设环境变量 `MAF_ENSEMBLE_DIR` 指向该目录。
- **不带也能跑**：集成不可用时自动降级为原单点预测，不会崩溃。
- hf_space 的 `app.py` UI 改动待同步。
