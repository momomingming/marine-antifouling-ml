#!/usr/bin/env python3
"""
============================================================================
海洋防污材料ML预测平台 v5.0 — 高级改进版
============================================================================
5大高级改进:
  1. MACCS/Morgan分子指纹 + PCA降维
  2. Williams图 (适用域分析)
  3. 贝叶斯反向设计
  4. 环境协变量
  5. 轻量级GNN (图神经网络) 分子嵌入
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

import json, os, sys, time
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

np.random.seed(42)
OUTPUT_DIR = '/share/玻尔比赛'

# ============================================================
# Column definitions
# ============================================================
BASE_FEATURES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'RingCount',
    'AromaticRings', 'HeavyAtoms', 'FractionCSP3',
    'NumF', 'NumCl', 'NumBr', 'NumN', 'NumO', 'NumS', 'NumSi', 'NumP',
    'HasCu', 'HasZn', 'HasAg', 'HasTi',
    'ChargeDensity', 'HydrophilicLipophilicBalance',
    'SurfaceEnergyEstimate', 'ElasticModulusEstimate',
    'RoughnessPotential', 'CrosslinkPotential',
]  # 28

INTERACTION_FEATURES = [
    'SE_x_EModulus', 'LogP_x_TPSA', 'F_x_Si', 'ChargeDensity_x_HLB',
    'MW_x_LogP', 'HBD_x_HBA', 'RotBonds_x_MW',
    'RingFrac', 'AromaFrac', 'FracF', 'FracSi', 'PolarFrac',
    'SE_minus_EMod', 'Kendall_index',
]  # 14

FEATURE_COLS = BASE_FEATURES + INTERACTION_FEATURES  # 42
TARGETS = [
    'antifouling_efficiency_pct', 'fouling_release_pct',
    'antibacterial_rate_pct', 'diatom_removal_pct',
]
TARGET_LABELS = {
    'antifouling_efficiency_pct': '防污效率',
    'fouling_release_pct': '污损释放率',
    'antibacterial_rate_pct': '抗菌率',
    'diatom_removal_pct': '硅藻去除率',
}

# ============================================================
# Data Loading
# ============================================================
def load_data():
    print("=" * 70)
    print("加载数据集...")
    print("=" * 70)
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'v3_dataset_5000.csv'))
    print(f"  数据集形状: {df.shape}")
    print(f"  原始样品: {df['is_original'].sum()}")
    print(f"  增强样品: {(~df['is_original']).sum()}")

    df_orig = df[df['is_original'] == True].copy()
    df_aug = df[df['is_original'] == False].copy()

    X_train_base = df_aug[FEATURE_COLS].values.astype(np.float64)
    X_test_base = df_orig[FEATURE_COLS].values.astype(np.float64)
    y_train = df_aug[TARGETS].values.astype(np.float64)
    y_test = df_orig[TARGETS].values.astype(np.float64)

    print(f"  训练集 (增强): {X_train_base.shape}")
    print(f"  测试集 (原始): {X_test_base.shape}")
    return df, df_orig, df_aug, X_train_base, X_test_base, y_train, y_test


# ============================================================
# Improvement 1: MACCS/Morgan Fingerprints + PCA
# ============================================================
def compute_all_fingerprints(smiles_list):
    """Compute MACCS (166 bits) + Morgan (radius=2, 128 bits) for each SMILES."""
    n = len(smiles_list)
    maccs_arr = np.zeros((n, 167), dtype=np.float32)
    morgan_arr = np.zeros((n, 128), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        # MACCS keys (RDKit returns 167 bits: index 0-166)
        mk = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr[i] = np.array(mk, dtype=np.float32)
        # Morgan fingerprint (radius=2, 128 bits)
        mg = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
        morgan_arr[i] = np.array(mg, dtype=np.float32)

    return maccs_arr, morgan_arr


def improvement1_fingerprints(X_train_base, X_test_base, y_train, y_test, df_aug, df_orig):
    print("\n" + "=" * 70)
    print("改进1: MACCS/Morgan分子指纹 + PCA降维")
    print("=" * 70)

    # Compute fingerprints
    print("  计算训练集指纹 (4762分子)...")
    t0 = time.time()
    maccs_train, morgan_train = compute_all_fingerprints(df_aug['SMILES'].values)
    print(f"    MACCS: {maccs_train.shape}, Morgan: {morgan_train.shape}  [{time.time()-t0:.1f}s]")

    print("  计算测试集指纹 (238分子)...")
    t0 = time.time()
    maccs_test, morgan_test = compute_all_fingerprints(df_orig['SMILES'].values)
    print(f"    MACCS: {maccs_test.shape}, Morgan: {morgan_test.shape}  [{time.time()-t0:.1f}s]")

    # PCA on MACCS (167 → 10)
    print("  PCA: MACCS 167 → 10 成分...")
    pca_maccs = PCA(n_components=10, random_state=42)
    maccs_train_pca = pca_maccs.fit_transform(maccs_train)
    maccs_test_pca = pca_maccs.transform(maccs_test)
    print(f"    解释方差: {pca_maccs.explained_variance_ratio_.sum():.3f}")

    # PCA on Morgan (128 → 10)
    print("  PCA: Morgan 128 → 10 成分...")
    pca_morgan = PCA(n_components=10, random_state=42)
    morgan_train_pca = pca_morgan.fit_transform(morgan_train)
    morgan_test_pca = pca_morgan.transform(morgan_test)
    print(f"    解释方差: {pca_morgan.explained_variance_ratio_.sum():.3f}")

    # Concatenate
    fp_train = np.hstack([maccs_train_pca, morgan_train_pca])  # (4762, 20)
    fp_test = np.hstack([maccs_test_pca, morgan_test_pca])      # (238, 20)

    X_train_fp = np.hstack([X_train_base, fp_train])  # (4762, 62)
    X_test_fp = np.hstack([X_test_base, fp_test])      # (238, 62)

    print(f"\n  特征维度: {X_train_base.shape[1]} → {X_train_fp.shape[1]}")

    # Compare baseline vs enriched
    print("\n  --- XGBoost 基线 (42特征) vs 指纹增强 (62特征) ---")
    results = OrderedDict()

    for ti, target in enumerate(TARGETS):
        # Baseline
        m_base = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   random_state=42, verbosity=0, n_jobs=-1)
        m_base.fit(X_train_base, y_train[:, ti])
        r2_base = r2_score(y_test[:, ti], m_base.predict(X_test_base))

        # Enriched
        m_fp = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 random_state=42, verbosity=0, n_jobs=-1)
        m_fp.fit(X_train_fp, y_train[:, ti])
        r2_fp = r2_score(y_test[:, ti], m_fp.predict(X_test_fp))

        results[target] = (r2_base, r2_fp)
        print(f"    {TARGET_LABELS[target]:8s}: {r2_base:.4f} → {r2_fp:.4f}  "
              f"(Δ={r2_fp - r2_base:+.4f})")

    # Train final models for all targets (for downstream use)
    print("\n  训练指纹增强模型 (XGBoost + LightGBM)...")
    xgb_models_fp, lgb_models_fp = {}, {}
    xgb_preds_fp, lgb_preds_fp = {}, {}

    for ti, target in enumerate(TARGETS):
        xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  random_state=42, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_train_fp, y_train[:, ti])
        xgb_models_fp[target] = xgb_m
        xgb_preds_fp[target] = xgb_m.predict(X_test_fp)

        lgb_m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   random_state=42, verbosity=-1, n_jobs=-1)
        lgb_m.fit(X_train_fp, y_train[:, ti])
        lgb_models_fp[target] = lgb_m
        lgb_preds_fp[target] = lgb_m.predict(X_test_fp)

    df_r2 = pd.DataFrame({
        'Target': [TARGET_LABELS[t] for t in TARGETS],
        'XGBoost_R2': [r2_score(y_test[:, i], xgb_preds_fp[t]) for i, t in enumerate(TARGETS)],
        'LightGBM_R2': [r2_score(y_test[:, i], lgb_preds_fp[t]) for i, t in enumerate(TARGETS)],
    })
    print("\n  指纹增强模型 R²:")
    print(df_r2.to_string(index=False))

    return X_train_fp, X_test_fp, xgb_models_fp, xgb_preds_fp, results


# ============================================================
# Improvement 2: Williams Plot (Applicability Domain)
# ============================================================
def improvement2_williams(X_train_fp, X_test_fp, y_train, y_test, xgb_models_fp):
    print("\n" + "=" * 70)
    print("改进2: Williams图 (适用域分析)")
    print("=" * 70)

    n_train = X_train_fp.shape[0]
    p = X_train_fp.shape[1]
    h_star = 3.0 * p / n_train
    print(f"  n_train = {n_train}, p = {p}, h* = 3p/n = {h_star:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    outlier_info = {}

    for idx, target in enumerate(TARGETS):
        ax = axes[idx // 2][idx % 2]
        model = xgb_models_fp[target]

        # Predictions
        y_pred_train = model.predict(X_train_fp)
        y_pred_test = model.predict(X_test_fp)

        # RMSE on training set
        rmse_train = np.sqrt(mean_squared_error(y_train[:, idx], y_pred_train))

        # Hat matrix from TRAINING data: H = X(X'X)^{-1}X'
        # For test samples: h_i = x_i' (X_train'X_train)^{-1} x_i
        XtX = X_train_fp.T @ X_train_fp
        # Regularize for numerical stability
        reg = 1e-6 * np.eye(XtX.shape[0])
        XtX_inv = np.linalg.inv(XtX + reg)

        # Leverage for test samples
        h_test = np.array([
            x @ XtX_inv @ x for x in X_test_fp
        ])

        # Standardized residuals (using training RMSE)
        residuals = y_test[:, idx] - y_pred_test
        std_res = residuals / max(rmse_train, 1e-10)

        # Count outliers
        n_outlier = int(np.sum((np.abs(std_res) > 3) | (h_test > h_star)))
        outlier_info[target] = {
            'n_outliers': n_outlier,
            'n_total': len(h_test),
            'h_max': float(h_test.max()),
            'std_res_max': float(np.abs(std_res).max()),
        }

        # Plot
        ax.scatter(h_test, std_res, alpha=0.5, s=20, c='steelblue', edgecolors='none')
        ax.axvline(x=h_star, color='red', linestyle='--', linewidth=1.5,
                   label=f'h* = {h_star:.4f}')
        ax.axhline(y=3, color='orange', linestyle='--', linewidth=1.2, label='±3σ')
        ax.axhline(y=-3, color='orange', linestyle='--', linewidth=1.2)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Highlight outliers
        mask_out = (np.abs(std_res) > 3) | (h_test > h_star)
        if mask_out.any():
            ax.scatter(h_test[mask_out], std_res[mask_out],
                       color='red', s=40, zorder=5, label=f'异常点 ({mask_out.sum()})')

        ax.set_xlabel('Leverage (h_ii)', fontsize=10)
        ax.set_ylabel('Standardized Residuals', fontsize=10)
        ax.set_title(f'{TARGET_LABELS[target]}\n'
                     f'h*={h_star:.4f}, 异常点={n_outlier}/{len(h_test)}', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(left=-0.005)

    fig.suptitle('Williams Plot — 适用域分析 (XGBoost指纹增强模型)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'v5_williams_all.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  保存: v5_williams_all.png")

    # Individual plots
    for idx, target in enumerate(TARGETS):
        model = xgb_models_fp[target]
        y_pred_train = model.predict(X_train_fp)
        y_pred_test = model.predict(X_test_fp)
        rmse_train = np.sqrt(mean_squared_error(y_train[:, idx], y_pred_train))

        XtX = X_train_fp.T @ X_train_fp
        reg = 1e-6 * np.eye(XtX.shape[0])
        XtX_inv = np.linalg.inv(XtX + reg)
        h_test = np.array([x @ XtX_inv @ x for x in X_test_fp])
        residuals = y_test[:, idx] - y_pred_test
        std_res = residuals / max(rmse_train, 1e-10)

        fig_s, ax_s = plt.subplots(figsize=(8, 6))
        ax_s.scatter(h_test, std_res, alpha=0.5, s=25, c='steelblue', edgecolors='none')
        ax_s.axvline(x=h_star, color='red', linestyle='--', linewidth=1.5,
                     label=f'h* = {h_star:.4f}')
        ax_s.axhline(y=3, color='orange', linestyle='--', linewidth=1.2)
        ax_s.axhline(y=-3, color='orange', linestyle='--', linewidth=1.2)
        ax_s.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        mask_out = (np.abs(std_res) > 3) | (h_test > h_star)
        if mask_out.any():
            ax_s.scatter(h_test[mask_out], std_res[mask_out],
                         color='red', s=50, zorder=5, label=f'异常点 ({mask_out.sum()})')
        ax_s.set_xlabel('Leverage (h_ii)', fontsize=12)
        ax_s.set_ylabel('Standardized Residuals', fontsize=12)
        ax_s.set_title(f'Williams Plot — {TARGET_LABELS[target]}', fontsize=13)
        ax_s.legend(fontsize=10)
        ax_s.set_xlim(left=-0.005)
        plt.tight_layout()
        fname = f'v5_williams_{target}.png'
        fig_s.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches='tight', dpi=150)
        plt.close(fig_s)

    print(f"  保存: 4个单独Williams图")

    # Summary
    print("\n  Williams图分析结果:")
    for t, info in outlier_info.items():
        print(f"    {TARGET_LABELS[t]:8s}: 异常点 {info['n_outliers']}/{info['n_total']}, "
              f"h_max={info['h_max']:.4f}, |res|_max={info['std_res_max']:.2f}")

    return outlier_info


# ============================================================
# Improvement 3: Bayesian Reverse Design
# ============================================================
def improvement3_reverse_design(X_train_fp, y_train, df_aug, X_test_fp, y_test):
    print("\n" + "=" * 70)
    print("改进3: 贝叶斯反向设计")
    print("=" * 70)

    # Train surrogate models on enriched features
    print("  训练代理模型...")
    surrogate_models = {}
    for ti, target in enumerate(TARGETS):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_train_fp, y_train[:, ti])
        surrogate_models[target] = m
    print("  代理模型训练完成")

    # Feature bounds from training data
    feat_min = X_train_fp.min(axis=0)
    feat_max = X_train_fp.max(axis=0)
    n_features = X_train_fp.shape[1]

    # Target performance goals
    goals = {
        'antifouling_efficiency_pct': 90.0,
        'fouling_release_pct': 85.0,
        'antibacterial_rate_pct': 80.0,
        'diatom_removal_pct': 85.0,
    }
    print(f"\n  设计目标:")
    for t, g in goals.items():
        print(f"    {TARGET_LABELS[t]}: ≥ {g}")

    # Optuna objective
    def objective(trial):
        x_cand = np.array([
            trial.suggest_float(f'x{i}', float(feat_min[i]), float(feat_max[i]))
            for i in range(n_features)
        ]).reshape(1, -1)

        score = 0.0
        preds = {}
        for target, goal in goals.items():
            pred = float(surrogate_models[target].predict(x_cand)[0])
            preds[target] = pred
            # Reward for meeting goal, partial credit below
            if pred >= goal:
                score += 25.0 + min(pred - goal, 10.0)
            else:
                score -= (goal - pred) ** 2 * 0.1

        # Diversity bonus: avoid extreme descriptor values
        z_scores = np.abs((x_cand[0] - X_train_fp.mean(axis=0)) /
                          (X_train_fp.std(axis=0) + 1e-10))
        penalty = np.mean(np.maximum(z_scores - 2, 0)) * 5.0
        score -= penalty

        return score

    print("  Optuna搜索 (150 trials)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=150, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"  搜索完成 [{elapsed:.1f}s], 最优得分: {study.best_value:.2f}")

    # Optimal descriptor
    x_opt = np.array([study.best_params[f'x{i}'] for i in range(n_features)])

    # Predicted performance
    opt_preds = {}
    for target in TARGETS:
        opt_preds[target] = float(surrogate_models[target].predict(x_opt.reshape(1, -1))[0])

    # Find nearest real material in training set
    dists = np.linalg.norm(X_train_fp - x_opt, axis=1)
    nearest_idx = int(np.argmin(dists))
    nearest_material = df_aug.iloc[nearest_idx]

    # Also find nearest in original (test) set for reference
    dists_test = np.linalg.norm(X_test_fp - x_opt, axis=1)
    nearest_test_idx = int(np.argmin(dists_test))

    # Build result
    result = OrderedDict()
    result['design_goals'] = {TARGET_LABELS[t]: g for t, g in goals.items()}
    result['optimal_predictions'] = {TARGET_LABELS[t]: round(v, 2) for t, v in opt_preds.items()}
    result['goals_met'] = {
        TARGET_LABELS[t]: round(opt_preds[t] >= goals[t])
        for t in TARGETS
    }
    result['nearest_training_material'] = OrderedDict([
        ('material_name', str(nearest_material['material_name'])),
        ('material_class', str(nearest_material['material_class'])),
        ('SMILES', str(nearest_material['SMILES'])),
        ('distance', round(float(dists[nearest_idx]), 4)),
    ])
    result['optuna_best_score'] = round(float(study.best_value), 2)
    result['optuna_n_trials'] = len(study.trials)

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'v5_reverse_design.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: v5_reverse_design.json")

    # Print results
    print("\n  反向设计结果:")
    print(f"  {'指标':12s}  {'目标':>6s}  {'预测':>6s}  {'达标':>4s}")
    print(f"  {'-'*34}")
    for t in TARGETS:
        met = '✓' if opt_preds[t] >= goals[t] else '✗'
        print(f"  {TARGET_LABELS[t]:12s}  {goals[t]:6.1f}  {opt_preds[t]:6.2f}  {met:>4s}")
    print(f"\n  最近训练材料: {nearest_material['material_name']} "
          f"({nearest_material['material_class']})")
    print(f"  距离: {dists[nearest_idx]:.4f}")

    return result


# ============================================================
# Improvement 4: Environmental Covariates
# ============================================================
ENV_VARS = ['seawater_temp_C', 'salinity_PSU', 'immersion_time_days']

# Baseline environmental parameters per material class
ENV_BASELINES = {
    'silicone':        {'temp': 20, 'sal': 35, 'time': 180},
    'fluoropolymer':   {'temp': 18, 'sal': 35, 'time': 365},
    'hydrogel':        {'temp': 22, 'sal': 30, 'time': 90},
    'zwitterionic':    {'temp': 20, 'sal': 32, 'time': 60},
    'self_polishing':  {'temp': 15, 'sal': 35, 'time': 240},
    'bioinspired':     {'temp': 22, 'sal': 33, 'time': 120},
    'nanocomposite':   {'temp': 20, 'sal': 35, 'time': 180},
    'smart':           {'temp': 25, 'sal': 30, 'time': 90},
    'nanocomposite_smart_hybrid': {'temp': 22, 'sal': 33, 'time': 150},
    'fluoropolymer_self_polishing_hybrid': {'temp': 18, 'sal': 35, 'time': 300},
}


def assign_env_covariates(df, is_augmented=False, parent_df=None):
    """Assign realistic environmental covariates."""
    n = len(df)
    rng = np.random.RandomState(42 if not is_augmented else 123)

    temps = np.zeros(n)
    sals = np.zeros(n)
    times = np.zeros(n)

    classes = df['material_class'].values

    if not is_augmented:
        # Original data: class-based baselines + small noise
        for i in range(n):
            bl = ENV_BASELINES.get(str(classes[i]),
                                   {'temp': 20, 'sal': 33, 'time': 120})
            temps[i] = bl['temp'] + rng.normal(0, 2)
            sals[i] = bl['sal'] + rng.normal(0, 1)
            times[i] = bl['time'] + rng.normal(0, 15)
    else:
        # Augmented data: wider random range with class influence
        for i in range(n):
            bl = ENV_BASELINES.get(str(classes[i]),
                                   {'temp': 20, 'sal': 33, 'time': 120})
            temps[i] = bl['temp'] + rng.normal(0, 5)
            sals[i] = bl['sal'] + rng.normal(0, 3)
            times[i] = bl['time'] + rng.normal(0, 50)

    # Clip to physical ranges
    temps = np.clip(temps, 5, 35)
    sals = np.clip(sals, 25, 40)
    times = np.clip(times, 7, 365)

    return pd.DataFrame({
        'seawater_temp_C': np.round(temps, 1),
        'salinity_PSU': np.round(sals, 1),
        'immersion_time_days': np.round(times, 0).astype(int),
    }, index=df.index)


def improvement4_environmental(X_train_fp, X_test_fp, y_train, y_test,
                                df_aug, df_orig):
    print("\n" + "=" * 70)
    print("改进4: 环境协变量")
    print("=" * 70)

    # Generate environmental covariates
    env_train = assign_env_covariates(df_aug, is_augmented=True)
    env_test = assign_env_covariates(df_orig, is_augmented=False)

    print("  环境协变量统计:")
    print(env_test.describe().round(1).to_string())

    env_train_vals = env_train[ENV_VARS].values.astype(np.float64)
    env_test_vals = env_test[ENV_VARS].values.astype(np.float64)

    # Add to features
    X_train_env = np.hstack([X_train_fp, env_train_vals])
    X_test_env = np.hstack([X_test_fp, env_test_vals])
    print(f"\n  特征维度: {X_train_fp.shape[1]} → {X_train_env.shape[1]}")

    # Compare models
    print("\n  --- 模型对比 (R²) ---")
    print(f"  {'目标':12s}  {'无环境':>8s}  {'有环境':>8s}  {'Δ':>8s}")
    print(f"  {'-'*40}")

    comparison = {}
    for ti, target in enumerate(TARGETS):
        # Without env
        m1 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, verbosity=0, n_jobs=-1)
        m1.fit(X_train_fp, y_train[:, ti])
        r2_no = r2_score(y_test[:, ti], m1.predict(X_test_fp))

        # With env
        m2 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, verbosity=0, n_jobs=-1)
        m2.fit(X_train_env, y_train[:, ti])
        r2_yes = r2_score(y_test[:, ti], m2.predict(X_test_env))

        comparison[target] = {'r2_no_env': r2_no, 'r2_with_env': r2_yes}
        print(f"  {TARGET_LABELS[target]:12s}  {r2_no:8.4f}  {r2_yes:8.4f}  "
              f"{r2_yes - r2_no:+8.4f}")

    # Environmental sensitivity analysis
    print("\n  环境敏感性分析: 温度对预测的影响")
    print("  (取测试集前5个样品, 变化温度观察预测变化)")

    sensitivity_model = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, random_state=42,
                                          verbosity=0, n_jobs=-1)
    sensitivity_model.fit(X_train_env, y_train[:, 0])  # antifouling

    temps_range = np.arange(5, 36, 2.5)
    n_samples = min(5, X_test_fp.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.Set1
    for si in range(n_samples):
        X_var = np.tile(X_test_fp[si], (len(temps_range), 1))
        env_var = np.tile(env_test_vals[si], (len(temps_range), 1))
        env_var[:, 0] = temps_range  # vary temperature
        X_var_full = np.hstack([X_var, env_var])
        preds = sensitivity_model.predict(X_var_full)
        name = str(df_orig.iloc[si]['material_name'])[:20]
        ax.plot(temps_range, preds, marker='o', markersize=3,
                label=f'{name}', color=cmap(si / n_samples))

    ax.set_xlabel('海水温度 (°C)', fontsize=12)
    ax.set_ylabel('预测防污效率 (%)', fontsize=12)
    ax.set_title('环境协变量影响分析 — 温度 vs 防污效率', fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'v5_environmental_sensitivity.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  保存: v5_environmental_sensitivity.png")

    # Immersion time sensitivity
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    time_range = np.arange(7, 366, 15)
    for si in range(n_samples):
        X_var = np.tile(X_test_fp[si], (len(time_range), 1))
        env_var = np.tile(env_test_vals[si], (len(time_range), 1))
        env_var[:, 2] = time_range
        X_var_full = np.hstack([X_var, env_var])
        preds = sensitivity_model.predict(X_var_full)
        name = str(df_orig.iloc[si]['material_name'])[:20]
        ax2.plot(time_range, preds, marker='s', markersize=2,
                 label=f'{name}', color=cmap(si / n_samples))

    ax2.set_xlabel('浸泡时间 (天)', fontsize=12)
    ax2.set_ylabel('预测防污效率 (%)', fontsize=12)
    ax2.set_title('环境协变量影响分析 — 浸泡时间 vs 防污效率', fontsize=13)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'v5_environmental_immersion.png'),
                 bbox_inches='tight', dpi=150)
    plt.close(fig2)
    print("  保存: v5_environmental_immersion.png")

    return X_train_env, X_test_env, comparison


# ============================================================
# Improvement 5: Lightweight GNN (Graph Neural Network)
# ============================================================
def build_molecular_graph(smiles):
    """Build molecular graph: node features + adjacency matrix from SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None

    mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    n_atoms = len(atoms)

    # Node features: [atomic_num, degree, formal_charge, hybridization, aromatic,
    #                 num_Hs, ring_membership]  → 7 features
    node_features = np.zeros((n_atoms, 7), dtype=np.float32)
    for i, atom in enumerate(atoms):
        node_features[i, 0] = atom.GetAtomicNum() / 50.0  # normalized
        node_features[i, 1] = atom.GetDegree() / 6.0
        node_features[i, 2] = atom.GetFormalCharge()
        node_features[i, 3] = int(atom.GetHybridization()) / 7.0
        node_features[i, 4] = float(atom.GetIsAromatic())
        node_features[i, 5] = atom.GetTotalNumHs() / 4.0
        node_features[i, 6] = float(atom.IsInRing())

    # Adjacency matrix (self-loops included)
    adj = np.eye(n_atoms, dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble()
        adj[i, j] = bt
        adj[j, i] = bt

    return node_features, adj


def gcn_layer(X, A, W):
    """Graph convolution: H' = D^{-0.5} A D^{-0.5} X W, with ReLU."""
    D = np.diag(A.sum(axis=1).clip(1e-8))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    H = A_hat @ X @ W
    H = np.maximum(H, 0)  # ReLU
    return H


def compute_graph_embedding(smiles, W1, W2):
    """2-layer GCN → global mean pooling → graph embedding."""
    X, A = build_molecular_graph(smiles)
    if X is None:
        return np.zeros(W2.shape[1], dtype=np.float32)

    H1 = gcn_layer(X, A, W1)
    H2 = gcn_layer(H1, A, W2)

    # Global mean pooling
    embedding = H2.mean(axis=0)
    return embedding


def improvement5_gnn(X_train_fp, X_test_fp, y_train, y_test,
                      df_aug, df_orig):
    print("\n" + "=" * 70)
    print("改进5: 轻量级GNN分子嵌入")
    print("=" * 70)

    n_node_feat = 7
    hidden1 = 32
    hidden2 = 16

    # Initialize GCN weights (Xavier-like)
    rng = np.random.RandomState(42)
    W1 = rng.randn(n_node_feat, hidden1).astype(np.float32) * np.sqrt(2.0 / n_node_feat)
    W2 = rng.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)

    # Compute embeddings
    print(f"  计算训练集GNN嵌入 ({len(df_aug)} 分子)...")
    t0 = time.time()
    gnn_train = np.zeros((len(df_aug), hidden2), dtype=np.float32)
    fail_train = 0
    for i, smi in enumerate(df_aug['SMILES'].values):
        emb = compute_graph_embedding(smi, W1, W2)
        gnn_train[i] = emb
        if emb.sum() == 0:
            fail_train += 1
        if (i + 1) % 1000 == 0:
            print(f"    进度: {i+1}/{len(df_aug)}")
    print(f"    完成 [{time.time()-t0:.1f}s], 失败: {fail_train}")

    print(f"  计算测试集GNN嵌入 ({len(df_orig)} 分子)...")
    t0 = time.time()
    gnn_test = np.zeros((len(df_orig), hidden2), dtype=np.float32)
    fail_test = 0
    for i, smi in enumerate(df_orig['SMILES'].values):
        emb = compute_graph_embedding(smi, W1, W2)
        gnn_test[i] = emb
        if emb.sum() == 0:
            fail_test += 1
    print(f"    完成 [{time.time()-t0:.1f}s], 失败: {fail_test}")

    # Add GNN embeddings to features
    X_train_gnn = np.hstack([X_train_fp, gnn_train])
    X_test_gnn = np.hstack([X_test_fp, gnn_test])
    print(f"\n  特征维度: {X_train_fp.shape[1]} → {X_train_gnn.shape[1]} "
          f"(+{hidden2} GNN)")

    # Compare models
    print("\n  --- 模型对比 (R²) ---")
    print(f"  {'目标':12s}  {'无GNN':>8s}  {'有GNN':>8s}  {'Δ':>8s}")
    print(f"  {'-'*40}")

    comparison = {}
    for ti, target in enumerate(TARGETS):
        # Without GNN
        m1 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, verbosity=0, n_jobs=-1)
        m1.fit(X_train_fp, y_train[:, ti])
        r2_no = r2_score(y_test[:, ti], m1.predict(X_test_fp))

        # With GNN (XGBoost)
        m2 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, verbosity=0, n_jobs=-1)
        m2.fit(X_train_gnn, y_train[:, ti])
        r2_yes = r2_score(y_test[:, ti], m2.predict(X_test_gnn))

        comparison[target] = {'r2_no_gnn': r2_no, 'r2_with_gnn': r2_yes}
        print(f"  {TARGET_LABELS[target]:12s}  {r2_no:8.4f}  {r2_yes:8.4f}  "
              f"{r2_yes - r2_no:+8.4f}")

    # Also try MLP on GNN embeddings only (pure graph-based prediction)
    print("\n  --- MLP仅用GNN嵌入 (16维) ---")
    for ti, target in enumerate(TARGETS):
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                           random_state=42, early_stopping=True,
                           validation_fraction=0.1)
        mlp.fit(gnn_train, y_train[:, ti])
        r2_mlp = r2_score(y_test[:, ti], mlp.predict(gnn_test))
        print(f"    {TARGET_LABELS[target]:12s}: R² = {r2_mlp:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(TARGETS))
    width = 0.35
    r2_no_list = [comparison[t]['r2_no_gnn'] for t in TARGETS]
    r2_yes_list = [comparison[t]['r2_with_gnn'] for t in TARGETS]

    bars1 = ax.bar(x_pos - width/2, r2_no_list, width, label='无GNN嵌入', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, r2_yes_list, width, label='+GNN嵌入', color='coral')

    ax.set_xlabel('预测目标', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('GNN分子嵌入对模型性能的影响', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'v5_gnn_comparison.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("\n  保存: v5_gnn_comparison.png")

    return X_train_gnn, X_test_gnn, comparison


# ============================================================
# Summary
# ============================================================
def generate_summary(fp_results, williams_info, reverse_result,
                     env_comparison, gnn_comparison,
                     xgb_preds_fp, y_test):
    print("\n" + "=" * 70)
    print("=" * 70)
    print("  v5 改进总结")
    print("=" * 70)
    print("=" * 70)

    # Improvement 1
    print("\n1. MACCS/Morgan分子指纹 + PCA")
    print(f"   {'目标':12s}  {'基线R²':>8s}  {'指纹增强R²':>10s}  {'Δ':>8s}")
    print(f"   {'-'*42}")
    for t in TARGETS:
        r2_b, r2_f = fp_results[t]
        print(f"   {TARGET_LABELS[t]:12s}  {r2_b:8.4f}  {r2_f:10.4f}  "
              f"{r2_f - r2_b:+8.4f}")

    # Improvement 2
    print("\n2. Williams图适用域分析")
    for t, info in williams_info.items():
        print(f"   {TARGET_LABELS[t]:12s}: 异常点 {info['n_outliers']}/{info['n_total']}")

    # Improvement 3
    print("\n3. 贝叶斯反向设计")
    for t in TARGETS:
        goal = reverse_result['design_goals'][TARGET_LABELS[t]]
        pred = reverse_result['optimal_predictions'][TARGET_LABELS[t]]
        met = '✓' if pred >= goal else '✗'
        print(f"   {TARGET_LABELS[t]:12s}: 目标≥{goal}, 预测={pred:.1f} {met}")

    # Improvement 4
    print("\n4. 环境协变量")
    print(f"   {'目标':12s}  {'无环境':>8s}  {'有环境':>8s}  {'Δ':>8s}")
    print(f"   {'-'*40}")
    for t in TARGETS:
        r2_no = env_comparison[t]['r2_no_env']
        r2_yes = env_comparison[t]['r2_with_env']
        print(f"   {TARGET_LABELS[t]:12s}  {r2_no:8.4f}  {r2_yes:8.4f}  "
              f"{r2_yes - r2_no:+8.4f}")

    # Improvement 5
    print("\n5. 轻量级GNN分子嵌入")
    print(f"   {'目标':12s}  {'无GNN':>8s}  {'有GNN':>8s}  {'Δ':>8s}")
    print(f"   {'-'*40}")
    for t in TARGETS:
        r2_no = gnn_comparison[t]['r2_no_gnn']
        r2_yes = gnn_comparison[t]['r2_with_gnn']
        print(f"   {TARGET_LABELS[t]:12s}  {r2_no:8.4f}  {r2_yes:8.4f}  "
              f"{r2_yes - r2_no:+8.4f}")

    # Final XGBoost performance with fingerprint-enhanced model
    print("\n  指纹增强XGBoost最终性能:")
    print(f"   {'目标':12s}  {'R²':>8s}  {'MAE':>8s}  {'RMSE':>8s}")
    print(f"   {'-'*40}")
    for i, t in enumerate(TARGETS):
        pred = xgb_preds_fp[t]
        r2 = r2_score(y_test[:, i], pred)
        mae = mean_absolute_error(y_test[:, i], pred)
        rmse = np.sqrt(mean_squared_error(y_test[:, i], pred))
        print(f"   {TARGET_LABELS[t]:12s}  {r2:8.4f}  {mae:8.3f}  {rmse:8.3f}")

    print("\n" + "=" * 70)
    print("  输出文件:")
    print("    v5_williams_all.png           — Williams图 (4合1)")
    print("    v5_williams_*.png             — Williams图 (单独)")
    print("    v5_reverse_design.json        — 反向设计结果")
    print("    v5_environmental_sensitivity.png — 温度敏感性")
    print("    v5_environmental_immersion.png   — 浸泡时间敏感性")
    print("    v5_gnn_comparison.png         — GNN对比图")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("  海洋防污材料ML预测平台 v5.0 — 高级改进版")
    print("  5大改进: 指纹PCA | Williams图 | 反向设计 | 环境协变量 | GNN")
    print("=" * 70)

    # Load data
    df, df_orig, df_aug, X_train_base, X_test_base, y_train, y_test = load_data()

    # Improvement 1: Fingerprints + PCA
    X_train_fp, X_test_fp, xgb_models_fp, xgb_preds_fp, fp_results = \
        improvement1_fingerprints(X_train_base, X_test_base, y_train, y_test, df_aug, df_orig)

    # Improvement 2: Williams Plot
    williams_info = improvement2_williams(
        X_train_fp, X_test_fp, y_train, y_test, xgb_models_fp)

    # Improvement 3: Bayesian Reverse Design
    reverse_result = improvement3_reverse_design(
        X_train_fp, y_train, df_aug, X_test_fp, y_test)

    # Improvement 4: Environmental Covariates
    X_train_env, X_test_env, env_comparison = improvement4_environmental(
        X_train_fp, X_test_fp, y_train, y_test, df_aug, df_orig)

    # Improvement 5: Lightweight GNN
    X_train_gnn, X_test_gnn, gnn_comparison = improvement5_gnn(
        X_train_fp, X_test_fp, y_train, y_test, df_aug, df_orig)

    # Summary
    generate_summary(fp_results, williams_info, reverse_result,
                     env_comparison, gnn_comparison, xgb_preds_fp, y_test)

    elapsed = time.time() - t_start
    print(f"\n  总运行时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("  v5 pipeline 完成!")


if __name__ == '__main__':
    main()
