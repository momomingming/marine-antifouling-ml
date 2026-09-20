"""
海洋防污材料预测 — 诚实交叉验证 (LOGO by SMILES)
====================================================
目的: 量化「随机切分 → 信息泄漏 → R² 虚高」的真实幅度。

背景(数据坑):
  dataset.csv 共 1158 行, 但唯一 SMILES 仅 84 个。
  其余行是同一分子的「模板+噪声」增强变体。
  旧流程 (improve_models.py / v7) 用随机 KFold 切分,
  同一 SMILES 的增强变体同时落在训练/测试集 → 模型背住样本 → R² 虚高。

本脚本对比两种验证:
  A) 随机 5-fold KFold  (旧做法, 含泄漏, 虚高)
  B) LOGO by SMILES     (按分子分组, 同分子变体不跨集, 诚实)

输出: R² / MAE / RMSE 对比 + 结论。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

RANDOM = 42
DATA = r"D:\锂硫电池\marine-antifouling-ml\data\raw\dataset.csv"
LABEL = "antifouling_efficiency_pct"
EXCLUDE = {
    "SMILES", "material_class", "material_name", "is_original",
    "antifouling_efficiency_pct", "fouling_release_pct",
    "antibacterial_rate_pct", "diatom_removal_pct",
}


def main():
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c not in EXCLUDE]
    X = df[features].astype(float).values
    y = df[LABEL].astype(float).values
    groups = df["SMILES"].astype(str).values
    n_groups = len(set(groups))
    n_orig = int((df["is_original"] == 1).sum()) if "is_original" in df else -1
    print(f"样本={len(df)} 特征={len(features)} 唯一SMILES组={n_groups} 原始样本(is_original=1)={n_orig}")
    print(f"增强样本占比 = {(1 - n_orig/len(df))*100:.1f}%")

    def make_models():
        return {
            "Ridge": Ridge(alpha=1.0, random_state=RANDOM),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, random_state=RANDOM),
        }

    def metrics(yt, yp):
        return (r2_score(yt, yp), mean_absolute_error(yt, yp),
                mean_squared_error(yt, yp) ** 0.5)

    results = []

    # A) 随机 KFold (旧做法, 含泄漏)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM)
    for name, m in make_models().items():
        r2s, maes, rmses = [], [], []
        for tr, te in kf.split(X):
            m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r, ma, rm = metrics(y[te], p)
            r2s.append(r); maes.append(ma); rmses.append(rm)
        results.append(("随机KFold(旧/虚高)", name, np.mean(r2s), np.mean(maes), np.mean(rmses)))

    # B) LOGO by SMILES (诚实)
    logo = LeaveOneGroupOut()
    for name, m in make_models().items():
        r2s, maes, rmses = [], [], []
        for tr, te in logo.split(X, y, groups):
            m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r, ma, rm = metrics(y[te], p)
            r2s.append(r); maes.append(ma); rmses.append(rm)
        results.append(("LOGO-SMILES(诚实)", name, np.mean(r2s), np.mean(maes), np.mean(rmses)))

    print(f"\n{'验证方案':20}{'模型':14}{'R2':>9}{'MAE':>9}{'RMSE':>9}")
    print("-" * 61)
    for r in results:
        print(f"{r[0]:20}{r[1]:14}{r[2]:9.3f}{r[3]:9.3f}{r[4]:9.3f}")

    # 结论: 取 XGBoost 对比
    kf_xgb = [r for r in results if r[1] == "XGBoost" and "随机" in r[0]][0]
    logo_xgb = [r for r in results if r[1] == "XGBoost" and "LOGO" in r[0]][0]
    drop = kf_xgb[2] - logo_xgb[2]
    print("\n=== 结论 ===")
    print(f"XGBoost: 随机KFold R²={kf_xgb[2]:.3f}  vs  LOGO R²={logo_xgb[2]:.3f}")
    print(f"信息泄漏导致的 R² 虚高幅度 = {drop:.3f} ({(drop/abs(kf_xgb[2])*100 if kf_xgb[2] else 0):.0f}%)")
    if logo_xgb[2] < 0.3:
        print("⚠️ LOGO 诚实 R² 偏低: 真实构效关系信号弱, 旧报告的虚高不可信, 需扩充真实独立样本。")


if __name__ == "__main__":
    main()
