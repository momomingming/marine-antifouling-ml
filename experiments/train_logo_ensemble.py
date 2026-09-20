"""
训练 LOGO 折外集成模型 + 域适用性参数 (P0 基础设施)
====================================================
产出 (写入 models/logo_ensemble/):
  fold_000.ubj ... fold_083.ubj   84 个折外 XGBoost 模型 (按 SMILES 分组留一)
  meta.pkl                        feature_names / scaler / 适用域参数 / 训练元信息

与 experiments/logo_cv.py 保持同一口径:
  特征 = dataset.csv 除 EXCLUDE 外的 28 列; 标签 = antifouling_efficiency_pct; 分组 = SMILES。

用法:
  python experiments/train_logo_ensemble.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from domain_applicability import ApplicabilityDomain  # noqa: E402
from uncertainty import BASE_FEATURE_NAMES            # noqa: E402

DATA = os.path.join(REPO, "data", "raw", "dataset.csv")
OUT = os.path.join(REPO, "models", "logo_ensemble")
LABEL = "antifouling_efficiency_pct"
EXCLUDE = {
    "SMILES", "material_class", "material_name", "is_original",
    "antifouling_efficiency_pct", "fouling_release_pct",
    "antibacterial_rate_pct", "diatom_removal_pct",
}
N_ESTIMATORS = 200
RANDOM = 42


def main():
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c not in EXCLUDE]
    # 与 BASE_FEATURE_NAMES 对齐, 保证推理端取数顺序一致
    if set(features) == set(BASE_FEATURE_NAMES):
        features = list(BASE_FEATURE_NAMES)
    X = df[features].astype(float).values
    y = df[LABEL].astype(float).values
    groups = df["SMILES"].astype(str).values
    uniq = sorted(set(groups))
    print(f"样本={len(df)} 特征={len(features)} 唯一SMILES组={len(uniq)}")

    os.makedirs(OUT, exist_ok=True)

    # 1) LOGO 折外模型 (同时收集折外预测, 用于估计真实残差尺度)
    logo = LeaveOneGroupOut()
    n_fold = 0
    oof = np.full(len(df), np.nan)
    for i, (tr, te) in enumerate(logo.split(X, y, groups)):
        m = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=4,
                             learning_rate=0.05, random_state=RANDOM)
        m.fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0.0, 100.0)
        m.save_model(os.path.join(OUT, f"fold_{i:03d}.ubj"))
        n_fold += 1
        if (i + 1) % 20 == 0 or i + 1 == len(uniq):
            print(f"  已训练折外模型 {i+1}/{len(uniq)}")
    print(f"折外模型训练完成: {n_fold} 个")

    # 折外残差尺度: 集成模型间分歧会严重低估真实误差(84 个模型高度相关),
    # 必须用实测折外 RMSE 给置信区间兜底, 否则会给出"假精确"的窄区间。
    resid = y - oof
    logo_rmse = float(np.sqrt(np.mean(resid ** 2)))
    logo_mae = float(np.mean(np.abs(resid)))
    print(f"折外残差: RMSE={logo_rmse:.3f} MAE={logo_mae:.3f}")

    # 2) 标准化 + 适用域 (在全量训练集上拟合, 用于推理端判定)
    scaler = StandardScaler().fit(X)
    ad = ApplicabilityDomain(q=95.0).fit(X)

    meta = {
        "feature_names": features,
        "label": LABEL,
        "n_models": n_fold,
        "n_groups": len(uniq),
        "n_samples": len(df),
        "n_estimators": N_ESTIMATORS,
        "scaler": scaler,
        "ad": ad.to_dict(),
        # 诚实性护栏: 折外实测残差尺度, 用于给置信区间兜底
        "logo_rmse": logo_rmse,
        "logo_mae": logo_mae,
        "logo_r2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
    }
    with open(os.path.join(OUT, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    size = sum(os.path.getsize(os.path.join(OUT, f_))
               for f_ in os.listdir(OUT)) / 1024 / 1024
    print(f"\n已保存: {OUT}")
    print(f"  折外模型 {n_fold} 个 | 特征 {len(features)} | 总大小 {size:.1f} MB")
    print(f"  适用域阈值(马氏距离95%分位) = {ad.threshold_:.3f}")


if __name__ == "__main__":
    main()
