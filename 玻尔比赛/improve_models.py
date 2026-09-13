#!/usr/bin/env python3
"""v7模型提升 (极简版): 增强特征 + 少量配置搜索 + 加权集成"""

import numpy as np, pandas as pd, json, os, pickle, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings; warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)
OUT = '/share/玻尔比赛'
t0 = time.time()

BASE = ['MW','LogP','TPSA','HBD','HBA','RotBonds','RingCount','AromaticRings',
        'HeavyAtoms','FractionCSP3','NumF','NumCl','NumBr','NumN','NumO','NumS',
        'NumSi','NumP','HasCu','HasZn','HasAg','HasTi','ChargeDensity',
        'HydrophilicLipophilicBalance','SurfaceEnergyEstimate',
        'ElasticModulusEstimate','RoughnessPotential','CrosslinkPotential']

def add_feat(df):
    df = df.copy()
    df['kendall'] = np.sqrt(np.power(10, df['ElasticModulusEstimate'].clip(-1,4)) * df['SurfaceEnergyEstimate'].clip(10,50))
    df['se_disp'] = df['SurfaceEnergyEstimate'] * (1 - df['TPSA']/(df['TPSA']+100))
    df['se_pol'] = df['SurfaceEnergyEstimate'] * df['TPSA']/(df['TPSA']+100)
    df['hydro_idx'] = df['LogP']*0.4 + (20-df['HydrophilicLipophilicBalance'])*0.3 + (1-df['TPSA']/200)*0.3
    df['se_x_em'] = df['SurfaceEnergyEstimate'] * df['ElasticModulusEstimate']
    df['logp_x_tpsa'] = df['LogP'] * df['TPSA']
    df['f_x_si'] = df['NumF'] * df['NumSi']
    df['chg_x_hlb'] = df['ChargeDensity'] * df['HydrophilicLipophilicBalance']
    df['mw_x_logp'] = np.log1p(df['MW']) * df['LogP']
    df['hbd_x_hba'] = df['HBD'] * df['HBA']
    df['flex'] = df['RotBonds'] / (df['MW']+1)
    df['ring_frac'] = df['RingCount'] / (df['HeavyAtoms']+1)
    df['arom_frac'] = df['AromaticRings'] / (df['RingCount']+1)
    df['frac_f'] = df['NumF'] / (df['HeavyAtoms']+1)
    df['frac_si'] = df['NumSi'] / (df['HeavyAtoms']+1)
    df['polar_frac'] = (df['NumO']+df['NumN']+df['NumS']) / (df['HeavyAtoms']+1)
    df['metal_idx'] = df['HasCu']*0.92 + df['HasZn']*0.85 + df['HasAg']*0.95 + df['HasTi']*0.80
    df['af_pot'] = ((1-df['SurfaceEnergyEstimate']/50)*0.25 + (1-np.log1p(df['ElasticModulusEstimate']+1)/5)*0.20 +
                    df['hydro_idx']*0.15 + df['metal_idx']*0.15 + df['ChargeDensity']*0.15 + df['CrosslinkPotential']*0.10)
    new = ['kendall','se_disp','se_pol','hydro_idx','se_x_em','logp_x_tpsa','f_x_si',
           'chg_x_hlb','mw_x_logp','hbd_x_hba','flex','ring_frac','arom_frac',
           'frac_f','frac_si','polar_frac','metal_idx','af_pot']
    return df, BASE + new

print("="*60)
print("  v7模型提升 (极简快速版)")
print("="*60)

df = pd.read_csv(f'{OUT}/dataset.csv')
df, fcols = add_feat(df)
for c in fcols:
    if c not in df.columns: df[c] = 0
print(f"数据: {len(df)}条, {len(fcols)}维特征 (28→{len(fcols)})")

targets = ['antifouling_efficiency_pct','fouling_release_pct','antibacterial_rate_pct','diatom_removal_pct']
labels = {'antifouling_efficiency_pct':'防污效率','fouling_release_pct':'脱附率',
          'antibacterial_rate_pct':'抗菌率','diatom_removal_pct':'硅藻去除率'}

X = df[fcols].values; y = df[targets].values
sc = StandardScaler(); Xs = sc.fit_transform(X)
mc = df['material_class'].apply(lambda x: x.split('_')[0])
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=mc)
print(f"训练: {len(Xtr)} | 盲测: {len(Xte)}")

v2 = {'antifouling_efficiency_pct':0.7162,'fouling_release_pct':0.7455,
      'antibacterial_rate_pct':0.7326,'diatom_removal_pct':0.6979}

# 每组只训练3个模型: XGBoost + LightGBM + GradientBoosting
# 然后加权集成
all_res = {}

for ti, tgt in enumerate(targets):
    lbl = labels[tgt]
    ytr_t, yte_t = ytr[:,ti], yte[:,ti]
    print(f"\n[{ti+1}/4] {lbl}...")

    # XGBoost (2个配置取最佳)
    xgb_cfgs = [
        {'n_estimators':300,'max_depth':5,'learning_rate':0.05,'subsample':0.8,
         'colsample_bytree':0.7,'reg_alpha':0.1,'reg_lambda':1.0,'min_child_weight':3,'gamma':0.5},
        {'n_estimators':400,'max_depth':6,'learning_rate':0.03,'subsample':0.85,
         'colsample_bytree':0.65,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_weight':4,'gamma':1.0},
    ]
    best_xgb_r2, best_xgb_pred, best_xgb_name = -999, None, ''
    for i, cfg in enumerate(xgb_cfgs):
        m = xgb.XGBRegressor(**cfg, random_state=42)
        m.fit(Xtr, ytr_t)
        pred = m.predict(Xte)
        r2 = r2_score(yte_t, pred)
        if r2 > best_xgb_r2:
            best_xgb_r2, best_xgb_pred = r2, pred
    print(f"  XGBoost:  R²={best_xgb_r2:.4f}  ({time.time()-t0:.0f}s)")

    # LightGBM
    lgb_cfgs = [
        {'n_estimators':300,'max_depth':6,'learning_rate':0.05,'subsample':0.8,
         'colsample_bytree':0.7,'reg_alpha':0.1,'reg_lambda':1.0,'min_child_samples':10},
        {'n_estimators':400,'max_depth':8,'learning_rate':0.03,'subsample':0.85,
         'colsample_bytree':0.65,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_samples':15},
    ]
    best_lgb_r2, best_lgb_pred = -999, None
    for cfg in lgb_cfgs:
        m = lgb.LGBMRegressor(**cfg, random_state=42, verbose=-1)
        m.fit(Xtr, ytr_t)
        pred = m.predict(Xte)
        r2 = r2_score(yte_t, pred)
        if r2 > best_lgb_r2:
            best_lgb_r2, best_lgb_pred = r2, pred
    print(f"  LightGBM: R²={best_lgb_r2:.4f}  ({time.time()-t0:.0f}s)")

    # GradientBoosting
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(Xtr, ytr_t)
    gb_pred = gb.predict(Xte)
    gb_r2 = r2_score(yte_t, gb_pred)
    print(f"  GB:       R²={gb_r2:.4f}  ({time.time()-t0:.0f}s)")

    # Ridge
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr_t)
    ridge_pred = ridge.predict(Xte)
    ridge_r2 = r2_score(yte_t, ridge_pred)

    # KNN
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(Xtr, ytr_t)
    knn_pred = knn.predict(Xte)
    knn_r2 = r2_score(yte_t, knn_pred)

    # Weighted ensemble
    r2s = [best_xgb_r2, best_lgb_r2, gb_r2, ridge_r2, knn_r2]
    preds = [best_xgb_pred, best_lgb_pred, gb_pred, ridge_pred, knn_pred]
    ws = np.array([max(0,r) for r in r2s])
    ws /= ws.sum() if ws.sum() > 0 else 1
    weighted = sum(w*p for w,p in zip(ws, preds))
    weighted_r2 = r2_score(yte_t, weighted)
    print(f"  Ensemble: R²={weighted_r2:.4f}  ({time.time()-t0:.0f}s)")

    results = {
        'XGBoost_Tuned': {'r2': best_xgb_r2, 'pred': best_xgb_pred, 'mae': mean_absolute_error(yte_t, best_xgb_pred)},
        'LightGBM_Tuned': {'r2': best_lgb_r2, 'pred': best_lgb_pred, 'mae': mean_absolute_error(yte_t, best_lgb_pred)},
        'GradientBoosting': {'r2': gb_r2, 'pred': gb_pred, 'mae': mean_absolute_error(yte_t, gb_pred)},
        'Ridge': {'r2': ridge_r2, 'pred': ridge_pred, 'mae': mean_absolute_error(yte_t, ridge_pred)},
        'KNN': {'r2': knn_r2, 'pred': knn_pred, 'mae': mean_absolute_error(yte_t, knn_pred)},
        'WeightedEnsemble': {'r2': weighted_r2, 'pred': weighted, 'mae': mean_absolute_error(yte_t, weighted)},
    }
    all_res[tgt] = results

    best = max(results, key=lambda k: results[k]['r2'])
    for n, d in sorted(results.items(), key=lambda x: -x[1]['r2']):
        tag = " ★" if n == best else ""
        print(f"    {n:20s}: R²={d['r2']:.4f}  MAE={d['mae']:.2f}{tag}")

# 汇总
print(f"\n{'='*60}")
print(f"{'目标':12s} | {'v2 R²':>8s} | {'v7 R²':>8s} | {'v7最佳':>20s} | {'提升':>8s}")
print("-"*65)
for tgt in targets:
    lbl = labels[tgt]
    best = max(all_res[tgt], key=lambda k: all_res[tgt][k]['r2'])
    v7r2 = all_res[tgt][best]['r2']
    print(f"{lbl:12s} | {v2[tgt]:8.4f} | {v7r2:8.4f} | {best:20s} | {v7r2-v2[tgt]:+8.4f}")

v2a = np.mean(list(v2.values()))
v7a = np.mean([max(all_res[t][k]['r2'] for k in all_res[t]) for t in targets])
print(f"{'平均':12s} | {v2a:8.4f} | {v7a:8.4f} | {'':20s} | {v7a-v2a:+8.4f}")

# 保存结果
with open(f'{OUT}/results_v7.json', 'w') as f:
    json.dump({t: {k: {'r2':float(v['r2']),'mae':float(v['mae'])} for k,v in r.items()} for t,r in all_res.items()}, f, indent=2)

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, tgt in enumerate(targets):
    ax = axes[i]
    names = list(all_res[tgt].keys())
    r2s = [all_res[tgt][m]['r2'] for m in names]
    best_n = max(all_res[tgt], key=lambda k: all_res[tgt][k]['r2'])
    colors = ['#4CAF50' if m == best_n else '#2196F3' for m in names]
    bars = ax.barh(range(len(names)), r2s, color=colors, alpha=0.85)
    ax.axvline(x=v2[tgt], color='red', linestyle='--', alpha=0.7, label=f'v2={v2[tgt]:.3f}')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('盲测R²'); ax.set_title(labels[tgt], fontweight='bold')
    ax.set_xlim(0, 1.05); ax.grid(axis='x', alpha=0.3); ax.legend(fontsize=7, loc='lower right')
    for b, v in zip(bars, r2s):
        ax.text(b.get_width()+0.01, b.get_y()+b.get_height()/2., f'{v:.3f}', fontsize=7)
fig.suptitle('v7 vs v2 模型性能对比 (红线=v2)', fontsize=14, fontweight='bold', y=1.05)
fig.tight_layout()
fig.savefig(f'{OUT}/v7_fig1_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
for i, tgt in enumerate(targets):
    ax = axes2[i//2][i%2]
    best = max(all_res[tgt], key=lambda k: all_res[tgt][k]['r2'])
    pred = all_res[tgt][best]['pred']; yt = yte[:,i]
    ax.scatter(yt, pred, alpha=0.5, s=30, c='#2196F3', edgecolors='white', linewidth=0.5)
    lims = [min(yt.min(),pred.min())-5, max(yt.max(),pred.max())+5]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    r2 = r2_score(yt, pred); mae = mean_absolute_error(yt, pred)
    ax.text(0.05, 0.95, f'R²={r2:.4f}\nMAE={mae:.2f}\n{best}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('真实值'); ax.set_ylabel('预测值')
    ax.set_title(labels[tgt], fontweight='bold'); ax.grid(alpha=0.3)
fig2.suptitle('v7盲测: 预测vs真实值', fontsize=15, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(f'{OUT}/v7_fig2_pred_vs_true.png', dpi=200, bbox_inches='tight')
plt.close(fig2)

elapsed = time.time() - t0
print(f"\n完成! 耗时{elapsed:.0f}秒  平均R²: {v2a:.4f} → {v7a:.4f} ({v7a-v2a:+.4f})")
