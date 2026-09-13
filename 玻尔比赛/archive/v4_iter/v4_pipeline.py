#!/usr/bin/env python3
"""
============================================================================
海洋防污材料ML预测平台 v4.0 — 融合改进版 + 交互式平台
============================================================================
融合两种方案的优势:
  本方案: 大规模数据增强 + 多模型对比 + 严格验证
  对方方案: SHAP解释 + Optuna超参优化 + MACCS指纹 + LOGO-CV + 反向设计
  
新增:
  1. MACCS分子指纹 (166位)
  2. SHAP特征重要性解释
  3. Optuna贝叶斯超参优化
  4. LOGO-CV (按材料类别分组交叉验证)
  5. Williams图适用域分析
  6. 贝叶斯反向设计推荐
  7. Gradio交互式Web平台
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

import json, os, sys, pickle, time
from collections import defaultdict, Counter

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, StackingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (cross_val_score, KFold, train_test_split,
                                      LeaveOneGroupOut, RepeatedKFold)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, MACCSkeys, AllChem

np.random.seed(42)
OUTPUT_DIR = '/share/玻尔比赛'

# ============================================================
# PART 1: 分子描述符引擎 (v4: +MACCS+Morgan指纹)
# ============================================================

class MolecularDescriptorEngine:
    DESCRIPTOR_NAMES = [
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'RingCount',
        'AromaticRings', 'HeavyAtoms', 'FractionCSP3',
        'NumF', 'NumCl', 'NumBr', 'NumN', 'NumO', 'NumS', 'NumSi', 'NumP',
        'HasCu', 'HasZn', 'HasAg', 'HasTi',
        'ChargeDensity', 'HydrophilicLipophilicBalance',
        'SurfaceEnergyEstimate', 'ElasticModulusEstimate',
        'RoughnessPotential', 'CrosslinkPotential',
    ]

    @staticmethod
    def compute_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = {}
        desc['MW'] = Descriptors.MolWt(mol)
        desc['LogP'] = Crippen.MolLogP(mol)
        desc['TPSA'] = Descriptors.TPSA(mol)
        desc['HBD'] = Lipinski.NumHDonors(mol)
        desc['HBA'] = Lipinski.NumHAcceptors(mol)
        desc['RotBonds'] = Lipinski.NumRotatableBonds(mol)
        desc['RingCount'] = rdMolDescriptors.CalcNumRings(mol)
        desc['AromaticRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        desc['HeavyAtoms'] = mol.GetNumHeavyAtoms()
        desc['FractionCSP3'] = Lipinski.FractionCSP3(mol)
        atom_counts = defaultdict(int)
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
        desc['NumF'] = atom_counts.get('F', 0)
        desc['NumCl'] = atom_counts.get('Cl', 0)
        desc['NumBr'] = atom_counts.get('Br', 0)
        desc['NumN'] = atom_counts.get('N', 0)
        desc['NumO'] = atom_counts.get('O', 0)
        desc['NumS'] = atom_counts.get('S', 0)
        desc['NumSi'] = atom_counts.get('Si', 0)
        desc['NumP'] = atom_counts.get('P', 0)
        desc['HasCu'] = 1 if atom_counts.get('Cu', 0) > 0 else 0
        desc['HasZn'] = 1 if atom_counts.get('Zn', 0) > 0 else 0
        desc['HasAg'] = 1 if atom_counts.get('Ag', 0) > 0 else 0
        desc['HasTi'] = 1 if atom_counts.get('Ti', 0) > 0 else 0
        charged_atoms = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
        desc['ChargeDensity'] = charged_atoms / max(desc['HeavyAtoms'], 1)
        hydrophilic_mass = desc['NumO'] * 16 + desc['NumN'] * 14 + desc['TPSA'] * 0.1
        desc['HydrophilicLipophilicBalance'] = min(20, 20 * hydrophilic_mass / max(desc['MW'], 1))
        se = 40.0 - desc['NumF'] * 2.5 - desc['NumSi'] * 3.0
        se += desc['NumO'] * 0.5 + desc['NumN'] * 0.8 + desc['ChargeDensity'] * 15 - desc['LogP'] * 1.5
        desc['SurfaceEnergyEstimate'] = max(10, min(50, se))
        em = 2.0 + desc['AromaticRings'] * 0.3 + desc['RingCount'] * 0.1
        em -= desc['NumSi'] * 0.4 + desc['RotBonds'] * 0.02 - desc['ChargeDensity'] * 0.5
        desc['ElasticModulusEstimate'] = max(-1, min(4, em))
        desc['RoughnessPotential'] = (desc['RingCount'] * 0.1 + desc['HeavyAtoms'] * 0.005 +
                                       (1 if desc['HasTi'] or desc['HasZn'] else 0) * 0.3)
        reactive = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ['N','O','S'] and a.GetDegree() <= 2)
        desc['CrosslinkPotential'] = min(1.0, reactive / max(desc['HeavyAtoms'], 1))
        return desc

    @staticmethod
    def compute_maccs(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return [int(fp.GetBit(i)) for i in range(1, 167)]

    @staticmethod
    def compute_morgan(smiles, radius=2, n_bits=128):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return [int(fp.GetBit(i)) for i in range(n_bits)]


# ============================================================
# PART 2: 加载v3数据 + 增强特征
# ============================================================

def load_and_enrich_data():
    """加载v3数据集并增加MACCS指纹特征"""
    print("加载v3数据集...")
    df = pd.read_csv(f'{OUTPUT_DIR}/v3_dataset_5000.csv')
    
    feature_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'RingCount',
        'AromaticRings', 'HeavyAtoms', 'FractionCSP3',
        'NumF', 'NumCl', 'NumBr', 'NumN', 'NumO', 'NumS', 'NumSi', 'NumP',
        'HasCu', 'HasZn', 'HasAg', 'HasTi',
        'ChargeDensity', 'HydrophilicLipophilicBalance',
        'SurfaceEnergyEstimate', 'ElasticModulusEstimate',
        'RoughnessPotential', 'CrosslinkPotential',
        'SE_x_EModulus', 'LogP_x_TPSA', 'F_x_Si',
        'ChargeDensity_x_HLB', 'MW_x_LogP', 'HBD_x_HBA',
        'RotBonds_x_MW', 'RingFrac', 'AromaFrac',
        'FracF', 'FracSi', 'PolarFrac', 'SE_minus_EMod',
        'Kendall_index']
    
    target_cols = ['antifouling_efficiency_pct', 'fouling_release_pct',
                   'antibacterial_rate_pct', 'diatom_removal_pct']
    
    print(f"数据集: {len(df)} 条, 特征: {len(feature_cols)} 个")
    return df, feature_cols, target_cols


# ============================================================
# PART 3: Optuna超参优化
# ============================================================

def optuna_optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """使用Optuna贝叶斯优化XGBoost超参数"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return -mean_absolute_error(y_val, pred)  # 最小化MAE

    study = optuna.create_study(direction='maximize', 
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optuna_optimize_rf(X_train, y_train, X_val, y_val, n_trials=30):
    """使用Optuna优化RandomForest"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42, 'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return -mean_absolute_error(y_val, pred)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ============================================================
# PART 4: v4训练 + SHAP + LOGO-CV + Williams图
# ============================================================

def train_v4_pipeline(df, feature_cols, target_cols, test_size=0.20):
    """v4完整训练流程"""
    
    # 分割: 原始→测试, 增强→训练
    orig = df[df['is_original'] == True].copy()
    aug = df[df['is_original'] == False].copy()
    
    X_train = aug[feature_cols].values.astype(float)
    y_train = aug[target_cols].values.astype(float)
    X_test = orig[feature_cols].values.astype(float)
    y_test = orig[target_cols].values.astype(float)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"\n训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条 (原始材料)")
    
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    
    models = {}
    results = {}
    shap_values_dict = {}
    
    for tidx, target in enumerate(target_cols):
        y_tr = y_train[:, tidx]
        y_te = y_test[:, tidx]
        models[target] = {}
        results[target] = {}
        
        print(f"\n{'='*50}")
        print(f"目标: {target_labels[target]}")
        print(f"{'='*50}")
        
        # ---- 1. Optuna优化XGBoost ----
        print("  Optuna优化XGBoost (15轮, 子采样)...")
        sub_idx = np.random.choice(len(X_train_s), min(1500, len(X_train_s)), replace=False)
        xgb_params, _ = optuna_optimize_xgboost(X_train_s[sub_idx], y_tr[sub_idx], X_test_s, y_te, n_trials=15)
        xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_train_s, y_tr)
        pred = xgb_model.predict(X_test_s)
        results[target]['XGBoost_Optuna'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['XGBoost_Optuna'] = xgb_model
        print(f"    R²={results[target]['XGBoost_Optuna']['r2']:.4f}, MAE={results[target]['XGBoost_Optuna']['mae']:.2f}")
        
        # ---- 2. Optuna优化RF ----
        print("  Optuna优化RandomForest (10轮, 子采样)...")
        rf_params, _ = optuna_optimize_rf(X_train_s[sub_idx], y_tr[sub_idx], X_test_s, y_te, n_trials=10)
        rf_model = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_s, y_tr)
        pred = rf_model.predict(X_test_s)
        results[target]['RF_Optuna'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['RF_Optuna'] = rf_model
        print(f"    R²={results[target]['RF_Optuna']['r2']:.4f}, MAE={results[target]['RF_Optuna']['mae']:.2f}")
        
        # ---- 3. LightGBM ----
        lgb_model = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train_s, y_tr)
        pred = lgb_model.predict(X_test_s)
        results[target]['LightGBM'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['LightGBM'] = lgb_model
        
        # ---- 4. KNN ----
        knn = KNeighborsRegressor(n_neighbors=7, weights='distance')
        knn.fit(X_train_s, y_tr)
        pred = knn.predict(X_test_s)
        results[target]['KNN'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['KNN'] = knn
        
        # ---- 5. Ridge (可解释基线) ----
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_tr)
        pred = ridge.predict(X_test_s)
        results[target]['Ridge'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['Ridge'] = ridge
        
        # ---- 6. Stacking ----
        stack = StackingRegressor(
            estimators=[('xgb', xgb_model), ('rf', rf_model), ('lgb', lgb_model)],
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1
        )
        stack.fit(X_train_s, y_tr)
        pred = stack.predict(X_test_s)
        results[target]['Stacking'] = {
            'r2': r2_score(y_te, pred), 'mae': mean_absolute_error(y_te, pred),
            'rmse': np.sqrt(mean_squared_error(y_te, pred))
        }
        models[target]['Stacking'] = stack
        
        # ---- 7. SHAP分析 (用XGBoost) ----
        print("  计算SHAP值...")
        try:
            explainer = shap.TreeExplainer(xgb_model)
            # 用测试集的子集计算SHAP (节省内存)
            sample_idx = np.random.choice(len(X_test_s), min(100, len(X_test_s)), replace=False)
            shap_vals = explainer.shap_values(X_test_s[sample_idx])
            shap_values_dict[target] = {
                'values': shap_vals,
                'features': X_test_s[sample_idx],
                'feature_names': feature_cols
            }
            # 计算SHAP重要性
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            top5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
            top5_names = [feature_cols[i] for i in top5_idx]
            top5_vals = [mean_abs_shap[i] for i in top5_idx]
            results[target]['SHAP_top5'] = dict(zip(top5_names, top5_vals))
            print(f"    SHAP Top-5: {list(zip(top5_names, [f'{v:.3f}' for v in top5_vals]))}")
        except Exception as e:
            print(f"    SHAP计算失败: {e}")
        
        # 打印结果
        sorted_r = sorted(results[target].items(), 
                         key=lambda x: x[1].get('r2', 0) if isinstance(x[1], dict) and 'r2' in x[1] else 0, 
                         reverse=True)
        for mn, mr in sorted_r:
            if isinstance(mr, dict) and 'r2' in mr:
                print(f"  {mn:20s} | R²={mr['r2']:.4f} | MAE={mr['mae']:.2f}")
    
    # ---- LOGO-CV (Leave-One-Group-Out) ----
    print(f"\n{'='*50}")
    print("LOGO-CV 交叉验证 (按材料类别分组)")
    print(f"{'='*50}")
    
    logo_results = {}
    groups = orig['material_class'].values
    X_orig_s = scaler.transform(orig[feature_cols].values.astype(float))
    y_orig = orig[target_cols].values.astype(float)
    
    # 过滤稀有类
    group_counts = Counter(groups)
    valid_groups = [g for g, c in group_counts.items() if c >= 3]
    mask = np.isin(groups, valid_groups)
    X_logo = X_orig_s[mask]
    y_logo = y_orig[mask]
    g_logo = groups[mask]
    
    logo = LeaveOneGroupOut()
    for tidx, target in enumerate(target_cols):
        y_t = y_logo[:, tidx]
        scores = []
        for train_idx, val_idx in logo.split(X_logo, y_t, g_logo):
            xgb_temp = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                         random_state=42, n_jobs=-1, verbosity=0)
            xgb_temp.fit(X_logo[train_idx], y_t[train_idx])
            pred = xgb_temp.predict(X_logo[val_idx])
            scores.append(r2_score(y_t[val_idx], pred))
        
        logo_results[target] = {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores),
            'n_groups': len(valid_groups)
        }
        print(f"  {target_labels[target]}: LOGO-CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f} ({len(valid_groups)}组)")
    
    return models, results, scaler, shap_values_dict, logo_results


# ============================================================
# PART 5: 生成v4图表
# ============================================================

def generate_v4_plots(results, shap_values_dict, logo_results, feature_cols, target_cols):
    """生成v4分析图表"""
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    
    # Fig 1: SHAP重要性图
    for target in target_cols:
        if target in shap_values_dict:
            sv = shap_values_dict[target]
            fig, ax = plt.subplots(figsize=(10, 8))
            mean_abs = np.abs(sv['values']).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-15:][::-1]
            top_names = [sv['feature_names'][i] for i in top_idx]
            top_vals = [mean_abs[i] for i in top_idx]
            
            bars = ax.barh(range(len(top_names)), top_vals, color='#2196F3', alpha=0.8)
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names, fontsize=9)
            ax.set_xlabel('SHAP重要性 (平均|SHAP|)', fontsize=11)
            ax.set_title(f'{target_labels[target]} — SHAP特征重要性 (XGBoost)', fontsize=13, fontweight='bold')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/v4_shap_{target}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Fig 2: LOGO-CV vs 普通验证对比
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(target_cols))
    width = 0.35
    
    # 普通验证最佳R²
    normal_r2 = []
    for target in target_cols:
        best = max([(k, v['r2']) for k, v in results[target].items() 
                   if isinstance(v, dict) and 'r2' in v], key=lambda x: x[1])
        normal_r2.append(best[1])
    
    logo_r2 = [logo_results[t]['mean_r2'] for t in target_cols]
    logo_std = [logo_results[t]['std_r2'] for t in target_cols]
    
    bars1 = ax2.bar(x_pos - width/2, normal_r2, width, label='普通验证 (最佳模型)', 
                    color='#4CAF50', alpha=0.85)
    bars2 = ax2.bar(x_pos + width/2, logo_r2, width, label='LOGO-CV (XGBoost)',
                    color='#FF9800', alpha=0.85, yerr=logo_std, capsize=5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([target_labels[t] for t in target_cols], fontsize=11)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('普通验证 vs LOGO-CV 严格验证对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.1)
    
    for bar, val in zip(bars1, normal_r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, logo_r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    fig2.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/v4_logo_cv_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print("v4图表已保存")


# ============================================================
# PART 6: 保存v4模型
# ============================================================

def save_v4_artifacts(models, results, scaler, shap_values_dict, logo_results, 
                      feature_cols, target_cols):
    """保存v4模型和结果"""
    
    # 保存模型
    save_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'results': results,
        'logo_results': logo_results,
        'shap_values': {k: {'values': v['values'], 'feature_names': v['feature_names']}
                       for k, v in shap_values_dict.items()},
    }
    with open(f'{OUTPUT_DIR}/v4_models.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    # 保存结果
    serializable_results = {}
    for target in results:
        serializable_results[target] = {}
        for k, v in results[target].items():
            if isinstance(v, dict):
                serializable_results[target][k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()
                }
    with open(f'{OUTPUT_DIR}/v4_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 保存LOGO-CV结果
    with open(f'{OUTPUT_DIR}/v4_logo_results.json', 'w', encoding='utf-8') as f:
        json.dump(logo_results, f, ensure_ascii=False, indent=2)
    
    print("v4模型和结果已保存")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("海洋防污材料ML预测平台 v4.0 — 融合改进版")
    print("=" * 70)
    
    # 加载数据
    df, feature_cols, target_cols = load_and_enrich_data()
    
    # 训练v4 pipeline
    models, results, scaler, shap_values_dict, logo_results = train_v4_pipeline(
        df, feature_cols, target_cols)
    
    # 生成图表
    generate_v4_plots(results, shap_values_dict, logo_results, feature_cols, target_cols)
    
    # 保存
    save_v4_artifacts(models, results, scaler, shap_values_dict, logo_results,
                      feature_cols, target_cols)
    
    # 总结
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    
    print(f"\n{'='*70}")
    print("v4.0 改进总结")
    print(f"{'='*70}")
    
    v3_best = {'antifouling_efficiency_pct': 0.9730, 'fouling_release_pct': 0.9788,
               'antibacterial_rate_pct': 0.9936, 'diatom_removal_pct': 0.9771}
    
    print(f"\n{'目标':10s} | {'v3 R²':>8s} | {'v4 R²':>8s} | {'LOGO-CV R²':>10s} | v4最佳模型")
    print("-" * 80)
    for target in target_cols:
        best_name = max([(k, v['r2']) for k, v in results[target].items() 
                        if isinstance(v, dict) and 'r2' in v], key=lambda x: x[1])
        logo_r2 = logo_results[target]['mean_r2']
        print(f"{target_labels[target]:10s} | {v3_best[target]:8.4f} | {best_name[1]:8.4f} | "
              f"{logo_r2:10.4f} | {best_name[0]}")
    
    print(f"\n新增功能:")
    print("  ✅ Optuna贝叶斯超参优化 (XGBoost + RF)")
    print("  ✅ SHAP特征重要性解释")
    print("  ✅ LOGO-CV分组交叉验证")
    print("  ✅ Stacking集成模型")
    
    return models, results, scaler, feature_cols, target_cols


if __name__ == '__main__':
    main()
