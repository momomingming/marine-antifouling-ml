# 成熟材料预测平台 Benchmark — 可借鉴点与并入方案（任务⑤）

> 目的: 从 GitHub / 文献中筛选成熟材料/分子预测平台, 对照 `docs/CODE_REVIEW.md` 的不足,
> 提炼能直接并入 `marine-antifouling-ml` 的优秀实践。
> 检索日期: 2026-09-19 (WebSearch, 平台通道)

## 1. 筛选平台清单

| 平台 | 链接 | 类型 | 核心亮点 |
|---|---|---|---|
| **FOULING_RELEASE_MLOPS** | github.com/Rahel-mahini/FOULING_RELEASE_MLOPS | **防污释放 ML** | 组合特征生成(combinatorixPy)、并行超参优化、**Williams plot(域适用性)**、**ALE 图(可解释)**、QSAR |
| **MLMD** | github.com/Jiaxuan-Ma/MLMD | 材料设计平台 | 完整闭环(数据→特征→建模→优化→发现→验证)、**主动学习(EI/PI/UCB)**、**多目标优化(NSGA-II)**、特征重要性排序 |
| **MaterialGen** | github.com/mwasifanwar/MaterialGen | 深度学习平台 | **模块化微服务 + REST API**、晶体图神经网络(CGNN)、条件生成、**注意力可解释** |
| **SteelML** | github.com/vamsi-op/steel-design-predictor | 钢铁性能预测 | **物理约束损失(PCNN)**、逆向设计(遗传算法)、**集成不确定性量化(置信区间)** |
| **OpenPoly** | WangGroupFDU/Openpoly_benchmark | 聚合物数据库 | 3985 条/26 属性、**数据稀缺时 XGBoost 优于深度学习(R²0.65–0.87)**、多任务 benchmark |
| molecular-viz / Melty-Molecules | github (Streamlit) | 分子性质 | RDKit 描述符标准化、**图神经网络(MPNN/PNA/GIN)**、3D 分子可视化(3Dmol.js) |

## 2. 借鉴点 × 本平台不足 映射

| 本平台不足 (CODE_REVIEW) | 借鉴源 | 具体做法 |
|---|---|---|
| P0 盲测 R² 虚高(信息泄漏) | 所有平台 | 独立/域外验证 + Williams plot 域适用性(已用 LOGO 修) |
| P0 训练目标被增强污染 | OpenPoly | 只用真实实验样本建模; 数据稀缺时 XGBoost 足够, 勿盲目上深度学习 |
| P0 描述符含启发式共线 | MLMD / FOULING_RELEASE | 相关性分析 + 特征重要性排序 + 共线剔除; 组合特征生成 |
| P1 无不确定性量化 | **SteelML** | 集成/bootstrap 输出置信区间 — R² 为负时**必须**告知用户可信度 |
| P1 单体架构缺 API | **MaterialGen** | 抽 REST API 层, 前端(Gradio)与模型解耦 |
| P1 无域适用性评估 | **FOULING_RELEASE** | Williams plot / 训练集凸包, 拒绝域外瞎预测 |
| P1 无逆向设计 | MLMD / SteelML | 输入目标效率→反推分子/配方(遗传/贝叶斯优化) |
| P1 无主动学习 | MLMD | 选"最该测"的样本建议用户补实验 |
| P2 模型单一(XGBoost) | OpenPoly / SteelML | 多模型对比面板(Ridge/RF/XGB/集成) |
| P2 缺 3D 可视化 | molecular-viz | RDKit + 3Dmol.js 分子结构渲染 |
| P2 特征不可解释 | FOULING_RELEASE / MaterialGen | SHAP / ALE / 注意力可视化 |

## 3. 可并入的改进清单（按性价比排序）

### 🔴 P0 — 立刻做（关乎"别给假结果"的底线）
1. **不确定性量化** (借鉴 SteelML)
   - 做法: 训练 N 个 bootstrap/XGBoost 集成, 预测时输出均值 ± 标准差(或分位数)
   - 并入: `src/predict.py` / `deployments/deploy_job/app.py` 预测接口返回置信区间
   - 价值: 诚实 R² 为负时, 必须让用户知道"这个结果不可信"
2. **域适用性评估** (借鉴 FOULING_RELEASE)
   - 做法: Williams plot(预测值 vs 杠杆 h) / 训练集凸包距离, 域外分子标红拒绝
   - 并入: `src/antifouling_platform.py` 新增 `applicability_domain()`
   - 价值: 防止对陌生分子瞎报高防污效率

### 🟡 P1 — 第二轮做（提升质量与科研价值）
3. **特征工程规范化** (借鉴 MLMD / FOULING_RELEASE)
   - 做法: 计算特征相关性矩阵, 剔除共线项(`SurfaceEnergyEstimate`等启发式), 做特征重要性排序
   - 并入: `src/antifouling_platform.py` 特征管线
4. **多模型对比面板** (借鉴 OpenPoly)
   - 做法: 同界面并排 Ridge/RF/XGBoost/集成 的 R²/MAE, 默认推荐 XGBoost
   - 并入: `deployments/deploy_job/app.py` 新增模型选择 Tab
5. **逆向设计** (借鉴 MLMD / SteelML)
   - 做法: 输入目标防污效率 → 遗传算法反推配方/侧链占比
   - 价值: 从"预测"升级到"设计", 比赛材料加分项
6. **主动学习建议** (借鉴 MLMD)
   - 做法: 用模型不确定性选"最该补实验"的分子, 反馈给用户扩库

### 🟢 P2 — 数据够了再做（体验与扩展）
7. **3D 分子可视化** (借鉴 molecular-viz) — RDKit + 3Dmol.js
8. **REST API 解耦** (借鉴 MaterialGen) — 抽 `/api/predict`, 便于接比赛演示/前端
9. **SHAP/ALE 可解释** (借鉴 FOULING_RELEASE) — 解释哪些描述符驱动预测

## 4. 暂不照搬的理由
- **图神经网络(CGNN/MPNN)**: OpenPoly 实测"数据稀缺时 XGBoost 优于深度学习"。
  本平台仅 84 个真实分子, 图网络必过拟合, **不划算**。
- **微服务/Docker 化**: 当前单体 Gradio 部署在 8502 已够用, 微服务是过度工程, 待用户量/比赛演示需要再抽 API(见 P2-8)。
- **完整主动学习闭环**: 依赖实验反馈回路, 当前无自动补数据通道, 先做"建议"不接"自动"。

## 5. 下一步
- 最贴领域: **FOULING_RELEASE_MLOPS** — 下一步深读其 `main.py` 的 Williams plot / 组合特征源码, 直接移植域适用性+特征生成逻辑
- P0-1/P0-2 不确定性+域适用性与 `experiments/logo_cv.py` 的 LOGO 验证一并构成"可信预测"底座
- 并入改动走"先计划→确认→写码"流程(见 MEMORY.md Part A 约定精神), 不动已部署服务直到验证通过
