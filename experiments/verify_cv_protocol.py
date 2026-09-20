"""
交叉验证协议口径核查 (关键!)
============================
问题: 同一个 LOGO 验证, 两种聚合方式给出截然相反的结论
      逐折平均 R² ≈ -0.594  vs  折外池化 R² ≈ +0.59

原因:
  - 逐折平均: 每个折 = 一个分子的增强变体(约14条), 折内 y 方差极小,
    模型无法预测"分子内部的增强噪声" → 每折 R² 极差 → 平均为负。
  - 池化: 把所有折外预测拼起来算一个 R², 衡量"跨分子"的预测能力,
    这也是 sklearn cross_val_predict + r2_score 的标准做法。

本脚本用同一模型配置, 把 2 种切分 × 2 种聚合 四种组合全部算出来,
并给出"信息泄漏虚高幅度"在每种口径下的真实值。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "raw", "dataset.csv")
LABEL = "antifouling_efficiency_pct"
EXCLUDE = {
    "SMILES", "material_class", "material_name", "is_original",
    "antifouling_efficiency_pct", "fouling_release_pct",
    "antibacterial_rate_pct", "diatom_removal_pct",
}
N_EST, DEPTH, LR, SEED = 300, 4, 0.05, 42   # 与 logo_cv.py 保持一致


def make_model():
    return xgb.XGBRegressor(n_estimators=N_EST, max_depth=DEPTH,
                            learning_rate=LR, random_state=SEED)


def evaluate(splits, X, y):
    """splits: 可迭代的 (tr, te)。返回 (逐折平均R², 池化R², 池化MAE, 折数)"""
    per_fold, oof = [], np.full(len(y), np.nan)
    n = 0
    for tr, te in splits:
        m = make_model()
        m.fit(X[tr], y[tr])
        p = np.clip(m.predict(X[te]), 0, 100)
        oof[te] = p
        per_fold.append(r2_score(y[te], p))
        n += 1
    pooled = r2_score(y, oof)
    return float(np.mean(per_fold)), float(pooled), float(mean_absolute_error(y, oof)), n


def main():
    df = pd.read_csv(DATA)
    feats = [c for c in df.columns if c not in EXCLUDE]
    X = df[feats].astype(float).values
    y = df[LABEL].astype(float).values
    groups = df["SMILES"].astype(str).values
    print(f"样本={len(df)} 特征={len(feats)} 唯一SMILES={len(set(groups))} 配置={N_EST}树\n")

    kf = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    logo = list(LeaveOneGroupOut().split(X, y, groups))

    kf_per, kf_pool, kf_mae, kf_n = evaluate(kf, X, y)
    lg_per, lg_pool, lg_mae, lg_n = evaluate(logo, X, y)

    print(f"{'切分方式':<18}{'折数':>6}{'逐折平均R²':>14}{'池化R²':>12}{'池化MAE':>10}")
    print("-" * 62)
    print(f"{'随机5折(泄漏)':<18}{kf_n:>6}{kf_per:>14.3f}{kf_pool:>12.3f}{kf_mae:>10.3f}")
    print(f"{'LOGO-by-SMILES':<18}{lg_n:>6}{lg_per:>14.3f}{lg_pool:>12.3f}{lg_mae:>10.3f}")

    print("\n=== 信息泄漏造成的虚高幅度 ===")
    print(f"  按逐折平均口径: {kf_per:.3f} → {lg_per:.3f}   虚高 {kf_per - lg_per:+.3f}")
    print(f"  按池化口径    : {kf_pool:.3f} → {lg_pool:.3f}   虚高 {kf_pool - lg_pool:+.3f}")

    print("\n=== 结论 ===")
    print("  池化 R² 是 sklearn cross_val_predict 的标准做法, 衡量跨分子泛化能力;")
    print("  逐折平均在 LeaveOneGroupOut 下会被'折内方差过小'严重扭曲, 不宜作为主指标。")
    print(f"  对外应报告: LOGO 池化 R² = {lg_pool:.3f}, MAE = {lg_mae:.3f}")


if __name__ == "__main__":
    main()
