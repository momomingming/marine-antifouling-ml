"""
P0 冒烟测试 (对应 docs/P0_UNCERTAINTY_DOMAIN_PLAN.md 第 6 节验证方式)
====================================================================
1. 集成模型能否加载
2. PDMS 样本 → 返回 prediction/ci_low/ci_high/in_domain
3. 明显离群样本 → in_domain=False 且置信区间更宽
4. 折外预测复核诚实 R² (应≈ logo_cv.py 的 -0.594)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from uncertainty import LOGOEnsemble, BASE_FEATURE_NAMES  # noqa: E402

DATA = os.path.join(REPO, "data", "raw", "dataset.csv")
LABEL = "antifouling_efficiency_pct"
PDMS = "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"


def main():
    ens = LOGOEnsemble()
    ok = ens.load()
    print(f"[1] 集成加载: {ok} | 折外模型数={len(ens.models)} | err={ens.load_error}")
    if not ok:
        return 1

    df = pd.read_csv(DATA)
    feats = ens.feature_names

    # [2] PDMS 真实描述符
    row = df[df["SMILES"].astype(str) == PDMS]
    if row.empty:
        print("[2] dataset 中无 PDMS 行, 跳过")
        return 1
    desc = {f: float(row.iloc[0][f]) for f in feats}
    res = ens.predict(desc)
    print(f"[2] PDMS 预测: {res['prediction']} | 95%CI=[{res['ci_low']}, {res['ci_high']}] "
          f"| 宽度={res['ci_width']} | 等级={res['uncertainty_level']} "
          f"| 适用域={res.get('in_domain')} 距离={res.get('distance')}/{res.get('threshold')}")

    # [3] 离群样本: 在训练特征范围外大幅偏移
    rng = np.random.default_rng(7)
    X = df[feats].astype(float).values
    lo, hi = X.min(0), X.max(0)
    outlier = lo + (hi - lo) * rng.uniform(-3.0, 4.0, size=len(feats))
    desc_o = {f: float(v) for f, v in zip(feats, outlier)}
    res_o = ens.predict(desc_o)
    print(f"[3] 离群样本: {res_o['prediction']} | 95%CI=[{res_o['ci_low']}, {res_o['ci_high']}] "
          f"| 宽度={res_o['ci_width']} | 等级={res_o['uncertainty_level']} "
          f"| 适用域={res_o.get('in_domain')} 距离={res_o.get('distance')}/{res_o.get('threshold')}")
    assert res_o.get("in_domain") is False, "离群样本应被判为超出适用域"
    assert res_o["ci_width"] >= res["ci_width"], "离群样本置信区间应更宽"

    # [4] 折外预测复核诚实 R²
    groups = df["SMILES"].astype(str).values
    y = df[LABEL].astype(float).values
    yp = np.empty_like(y)
    uniq = sorted(set(groups))
    for i, g in enumerate(uniq):
        mask = groups == g
        # 折外模型在原始特征上训练, 此处必须用原始 X(不可用标准化后的)
        yp[mask] = ens.models[i].predict(X[mask])
    yp = np.clip(yp, 0, 100)
    print(f"[4] 折外集成复核: 诚实 R²={r2_score(y, yp):.3f} MAE={mean_absolute_error(y, yp):.3f} "
          f"(参照 logo_cv LOGO R²=-0.594)")

    print("\n✅ P0 冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
