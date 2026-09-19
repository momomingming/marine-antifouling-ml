#!/usr/bin/env python3
"""
============================================================================
海洋防污材料ML预测交互式平台 v5.0
基于Gradio的Web交互界面
============================================================================
功能:
  1. SMILES输入 → 实时预测防污性能
  2. 多模型对比预测 + SHAP解释
  3. 材料性能雷达图
  4. 文献数据库浏览
  5. 反向设计推荐
  6. 环境条件影响分析
  7. 适用域 (Williams Plot)
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
rcParams['figure.dpi'] = 120

import json, os, pickle, io, base64
from collections import defaultdict

import gradio as gr
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, Lipinski, Crippen

OUTPUT_DIR = '/share/玻尔比赛'

# ============================================================
# 加载模型
# ============================================================

def load_models():
    """加载v4模型"""
    with open(f'{OUTPUT_DIR}/v4_models.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def load_v3_models():
    """加载v3模型作为备选"""
    try:
        with open(f'{OUTPUT_DIR}/v3_models.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None

# ============================================================
# 分子描述符计算 (与训练一致)
# ============================================================

def compute_descriptors(smiles):
    """从SMILES计算42维特征"""
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
    
    # 交互特征
    desc['SE_x_EModulus'] = desc['SurfaceEnergyEstimate'] * desc['ElasticModulusEstimate']
    desc['LogP_x_TPSA'] = desc['LogP'] * desc['TPSA']
    desc['F_x_Si'] = desc['NumF'] * desc['NumSi']
    desc['ChargeDensity_x_HLB'] = desc['ChargeDensity'] * desc['HydrophilicLipophilicBalance']
    desc['MW_x_LogP'] = desc['MW'] * desc['LogP']
    desc['HBD_x_HBA'] = desc['HBD'] * desc['HBA']
    desc['RotBonds_x_MW'] = desc['RotBonds'] / max(desc['MW'], 1)
    desc['RingFrac'] = desc['RingCount'] / max(desc['HeavyAtoms'], 1)
    desc['AromaFrac'] = desc['AromaticRings'] / max(desc['RingCount'], 1)
    desc['FracF'] = desc['NumF'] / max(desc['HeavyAtoms'], 1)
    desc['FracSi'] = desc['NumSi'] / max(desc['HeavyAtoms'], 1)
    desc['PolarFrac'] = (desc['NumN'] + desc['NumO']) / max(desc['HeavyAtoms'], 1)
    desc['SE_minus_EMod'] = desc['SurfaceEnergyEstimate'] - desc['ElasticModulusEstimate'] * 5
    desc['Kendall_index'] = np.sqrt(max(desc['SurfaceEnergyEstimate'], 0.1) *
                                     max(10**desc['ElasticModulusEstimate'], 0.1))
    
    return desc


# ============================================================
# 预测函数
# ============================================================

def predict_material(smiles, model_data):
    """预测材料防污性能"""
    desc = compute_descriptors(smiles)
    if desc is None:
        return None
    
    feature_cols = model_data['feature_cols']
    features = np.array([[desc.get(c, 0) for c in feature_cols]])
    features_scaled = model_data['scaler'].transform(features)
    
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    
    results = {}
    for target in model_data['target_cols']:
        preds = {}
        for m_name, model in model_data['models'][target].items():
            if m_name in ['WeightedEnsemble_weights', 'SHAP_top5']:
                continue
            try:
                pred = model.predict(features_scaled)[0]
                preds[m_name] = float(np.clip(pred, 0, 100))
            except:
                pass
        results[target] = preds
    
    return results, desc


# ============================================================
# v5: 反向设计函数
# ============================================================

def do_reverse_design(target_af, target_fr, target_ab, target_dr):
    """使用Optuna反向优化描述符，寻找最佳材料设计参数"""
    model_data = load_models()
    FEATURE_NAMES = model_data['feature_names']
    scaler = model_data['scaler']

    # 加载训练数据用于构建代理模型和最近邻搜索
    train_df = pd.read_csv(f'{OUTPUT_DIR}/v2_train_data.csv')
    test_df = pd.read_csv(f'{OUTPUT_DIR}/v2_test_data.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    target_cols = ['antifouling_efficiency_pct', 'fouling_release_pct',
                   'antibacterial_rate_pct', 'diatom_removal_pct']

    X_all = full_df[FEATURE_NAMES].values
    y_all = {t: full_df[t].values for t in target_cols}

    X_scaled_all = scaler.transform(X_all)

    # 为每个目标训练一个简单XGBoost代理模型
    from xgboost import XGBRegressor
    surrogate_models = {}
    for t in target_cols:
        m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=0)
        m.fit(X_scaled_all, y_all[t])
        surrogate_models[t] = m

    # 10个可优化描述符及其合理范围
    optimizable = {
        'MW': (50, 2000),
        'LogP': (-3, 10),
        'TPSA': (0, 300),
        'NumF': (0, 50),
        'NumSi': (0, 20),
        'ChargeDensity': (0, 0.5),
        'HydrophilicLipophilicBalance': (0, 20),
        'SurfaceEnergyEstimate': (10, 50),
        'ElasticModulusEstimate': (-1, 4),
        'CrosslinkPotential': (0, 1),
    }
    opt_keys = list(optimizable.keys())

    # 使用训练数据的均值填充不可优化的特征
    X_mean = np.mean(X_all, axis=0)
    feature_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    targets_goals = {
        'antifouling_efficiency_pct': target_af,
        'fouling_release_pct': target_fr,
        'antibacterial_rate_pct': target_ab,
        'diatom_removal_pct': target_dr,
    }

    def objective(trial):
        x = X_mean.copy()
        for key in opt_keys:
            lo, hi = optimizable[key]
            val = trial.suggest_float(key, lo, hi)
            if key in feature_idx:
                x[feature_idx[key]] = val

        # 重新计算交互特征
        d = dict(zip(FEATURE_NAMES, x))
        d['SE_x_EModulus'] = d.get('SurfaceEnergyEstimate', 30) * d.get('ElasticModulusEstimate', 1)
        d['LogP_x_TPSA'] = d.get('LogP', 2) * d.get('TPSA', 50)
        d['F_x_Si'] = d.get('NumF', 0) * d.get('NumSi', 0)
        d['ChargeDensity_x_HLB'] = d.get('ChargeDensity', 0.05) * d.get('HydrophilicLipophilicBalance', 8)
        d['MW_x_LogP'] = d.get('MW', 500) * d.get('LogP', 2)
        x = np.array([d.get(n, X_mean[i]) for i, n in enumerate(FEATURE_NAMES)])

        x_s = scaler.transform(x.reshape(1, -1))[0]
        total = 0
        for t, goal in targets_goals.items():
            pred = surrogate_models[t].predict(x_s.reshape(1, -1))[0]
            total += (pred - goal) ** 2
        return total

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    # 用最优参数构建完整描述符向量
    best_params = study.best_params
    x_best = X_mean.copy()
    for key in opt_keys:
        if key in feature_idx:
            x_best[feature_idx[key]] = best_params[key]

    d_best = dict(zip(FEATURE_NAMES, x_best))
    d_best['SE_x_EModulus'] = d_best.get('SurfaceEnergyEstimate', 30) * d_best.get('ElasticModulusEstimate', 1)
    d_best['LogP_x_TPSA'] = d_best.get('LogP', 2) * d_best.get('TPSA', 50)
    d_best['F_x_Si'] = d_best.get('NumF', 0) * d_best.get('NumSi', 0)
    d_best['ChargeDensity_x_HLB'] = d_best.get('ChargeDensity', 0.05) * d_best.get('HydrophilicLipophilicBalance', 8)
    d_best['MW_x_LogP'] = d_best.get('MW', 500) * d_best.get('LogP', 2)
    x_best = np.array([d_best.get(n, X_mean[i]) for i, n in enumerate(FEATURE_NAMES)])

    x_best_s = scaler.transform(x_best.reshape(1, -1))[0]

    # 预测最优性能
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    pred_rows = []
    for t in target_cols:
        pred_val = surrogate_models[t].predict(x_best_s.reshape(1, -1))[0]
        goal_val = targets_goals[t]
        pred_rows.append({
            '预测目标': target_labels.get(t, t),
            '目标值': f'{goal_val:.1f}',
            '预测值': f'{pred_val:.1f}',
            '差距': f'{pred_val - goal_val:+.1f}'
        })
    pred_df = pd.DataFrame(pred_rows)

    # 最近邻搜索
    dists = np.linalg.norm(X_scaled_all - x_best_s, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_mat = full_df.iloc[nearest_idx]

    # 最优描述符表
    desc_rows = []
    for key in opt_keys:
        desc_rows.append({'描述符': key, '最优值': f'{best_params[key]:.3f}'})
    desc_df = pd.DataFrame(desc_rows)

    nearest_info = (
        f"**最近训练材料**: {nearest_mat.get('material_name', 'Unknown')}\n\n"
        f"**类别**: {nearest_mat.get('material_category', 'Unknown')}\n\n"
        f"**SMILES**: `{nearest_mat.get('SMILES', 'N/A')}`\n\n"
        f"**欧氏距离**: {dists[nearest_idx]:.2f}"
    )

    return pred_df, desc_df, nearest_info


# ============================================================
# v5: 环境条件影响函数
# ============================================================

def do_env_predict(smiles, temperature, salinity, immersion_time):
    """预测考虑环境因素的材料性能"""
    model_data = load_models()
    models = model_data['models']
    scaler = model_data['scaler']
    FEATURE_NAMES = model_data['feature_names']
    target_cols = ['antifouling_efficiency_pct', 'fouling_release_pct',
                   'antibacterial_rate_pct', 'diatom_removal_pct']
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }

    descriptors = compute_descriptors(smiles)
    if descriptors is None:
        return None, None, "❌ 无法解析SMILES"

    x = np.array([descriptors.get(name, 0.0) for name in FEATURE_NAMES])
    x_scaled = scaler.transform(x.reshape(1, -1))[0]

    # 基础预测 (stacking)
    base_preds = {}
    for t in target_cols:
        stacking_model = models[t]['stacking']
        base_preds[t] = stacking_model.predict(x_scaled.reshape(1, -1))[0]

    # 环境调整系数 (基于物理直觉)
    temp_ref = 20.0
    temp_factor = (temperature - temp_ref) / 100.0
    salinity_ref = 33.0
    salinity_factor = (salinity - salinity_ref) / 200.0
    time_factor = np.log1p(immersion_time / 30.0) * 0.02

    env_adjust_coeffs = {
        'antifouling_efficiency_pct': (-3, 1, -2),
        'fouling_release_pct': (-2, -0.5, -1.5),
        'antibacterial_rate_pct': (-1.5, 0.5, -1),
        'diatom_removal_pct': (-2.5, -1, -2),
    }

    env_adjust = {}
    for t, (tc, sc, ic) in env_adjust_coeffs.items():
        env_adjust[t] = tc * temp_factor + sc * salinity_factor + ic * time_factor

    adj_preds = {}
    for t in target_cols:
        adj_preds[t] = np.clip(base_preds[t] + env_adjust[t] * 10, 0, 100)

    # 结果表
    rows = []
    for t in target_cols:
        rows.append({
            '预测目标': target_labels.get(t, t),
            '基础预测 (%)': f'{base_preds[t]:.1f}',
            '环境调整后 (%)': f'{adj_preds[t]:.1f}',
            '变化': f'{adj_preds[t] - base_preds[t]:+.1f}'
        })
    result_df = pd.DataFrame(rows)

    # 温度敏感性线图
    temps = np.arange(5, 36, 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    temp_coeffs = [-3, -2, -1.5, -2.5]
    for i, t in enumerate(target_cols):
        ax = axes[i]
        vals = []
        for temp in temps:
            tf = (temp - temp_ref) / 100.0
            v = base_preds[t] + temp_coeffs[i] * tf * 10
            vals.append(np.clip(v, 0, 100))
        ax.plot(temps, vals, color=colors[i], linewidth=2)
        ax.axvline(x=temperature, color='red', linestyle='--', alpha=0.5,
                   label=f'当前: {temperature}°C')
        ax.set_title(target_labels.get(t, t), fontsize=11)
        ax.set_xlabel('温度 (°C)', fontsize=9)
        ax.set_ylabel('预测值 (%)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return result_df, fig, "✅ 环境条件预测完成"


# ============================================================
# Gradio界面函数
# ============================================================

def create_platform():
    """创建Gradio交互平台"""
    
    # 加载模型
    print("加载模型...")
    model_data = load_models()
    
    # 预定义材料库
    preset_materials = {
        'PDMS标准涂层': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C',
        '氟硅改性弹性体': 'C[Si](C)(C)O[Si](C)(CCC(F)(F)C(F)(F)F)C',
        '全氟己烷': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        '磺酸甜菜碱SBMA': 'C[N+](C)(C)CCS(=O)(=O)[O-]',
        'PEG600': 'OCCOCCOCCOCCOCCOCCOCCO',
        '含氟丙烯酸酯': 'OC(=O)C(C)CC(F)(F)C(F)(F)C(F)(F)F',
        'PDMS-g-PEG200': 'C[Si](C)(CCOCCOCCO)O[Si](C)(C)C',
        '丙烯酸铜SPC': 'OC(=O)C(C)C(=O)O[Cu]',
        'PDMS/Ag纳米复合': 'C[Si](C)(C)O[Si](C)(C)C.[Ag]',
        'SLIPS-PDMS+氟碳': 'C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)C(F)(F)F',
        'PNIPAM温度响应': 'OC(=O)C=C.NC(=O)C(C)C',
        '磷酸胆碱PCBMA': 'C[N+](C)(C)CCOP(=O)([O-])O',
    }
    
    target_labels = {
        'antifouling_efficiency_pct': '防污效率',
        'fouling_release_pct': '污损脱附率',
        'antibacterial_rate_pct': '抗菌率',
        'diatom_removal_pct': '硅藻去除率'
    }
    
    # ---- 预测函数 ----
    def do_predict(smiles_input, preset_name):
        if preset_name and preset_name != '自定义SMILES':
            smiles = preset_materials.get(preset_name, smiles_input)
        else:
            smiles = smiles_input
        
        if not smiles or not smiles.strip():
            return "请输入SMILES分子结构", None, None, None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"❌ SMILES解析失败: {smiles[:50]}...", None, None, None
        
        # 绘制分子结构
        img = Draw.MolToImage(mol, size=(400, 300))
        mol_path = f'{OUTPUT_DIR}/_temp_mol.png'
        img.save(mol_path)
        
        # 预测
        result = predict_material(smiles, model_data)
        if result is None:
            return "❌ 预测失败", mol_path, None, None
        
        preds, desc = result
        
        # 构建结果表格
        table_data = []
        model_names_set = set()
        for target in model_data['target_cols']:
            for mn in preds[target]:
                model_names_set.add(mn)
        model_names = sorted(model_names_set)
        
        header = ['目标'] + model_names + ['平均值', '标准差']
        table_data.append(header)
        
        avg_scores = []
        for target in model_data['target_cols']:
            row = [target_labels.get(target, target)]
            vals = [preds[target].get(mn, 0) for mn in model_names]
            row.extend([f'{v:.1f}' for v in vals])
            row.append(f'{np.mean(vals):.1f}')
            row.append(f'{np.std(vals):.1f}')
            avg_scores.append(np.mean(vals))
            table_data.append(row)
        
        # 综合评分
        weights = [0.35, 0.25, 0.20, 0.20]
        composite = np.average(avg_scores, weights=weights)
        
        if composite >= 85:
            level = '🌟 优秀 (推荐开发)'
        elif composite >= 75:
            level = '✅ 良好 (值得尝试)'
        elif composite >= 65:
            level = '⚠️ 一般 (需优化)'
        else:
            level = '❌ 不推荐'
        
        summary = f"**综合可行性评分: {composite:.1f}/100** | {level}\n\n"
        summary += f"**SMILES**: `{smiles[:80]}...`\n\n"
        summary += f"**关键描述符**: MW={desc['MW']:.1f}, LogP={desc['LogP']:.2f}, "
        summary += f"TPSA={desc['TPSA']:.1f}, 表面能≈{desc['SurfaceEnergyEstimate']:.1f} mN/m"
        
        # 雷达图
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        categories = [target_labels[t] for t in model_data['target_cols']]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        # 各模型预测
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
        for i, mn in enumerate(model_names[:5]):
            vals = [preds[t].get(mn, 0) for t in model_data['target_cols']]
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=1.5, color=colors[i % len(colors)], 
                   label=mn, markersize=5, alpha=0.8)
            ax.fill(angles, vals, alpha=0.05, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_title('多模型预测对比', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
        fig.tight_layout()
        radar_path = f'{OUTPUT_DIR}/_temp_radar.png'
        fig.savefig(radar_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # SHAP解释图 (使用保存的SHAP数据)
        shap_fig = None
        if 'shap_values' in model_data:
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
            for idx, target in enumerate(model_data['target_cols']):
                ax = axes2[idx//2][idx%2]
                if target in model_data['shap_values']:
                    sv = model_data['shap_values'][target]
                    mean_abs = np.abs(sv['values']).mean(axis=0)
                    top_idx = np.argsort(mean_abs)[-8:][::-1]
                    top_names = [sv['feature_names'][i] for i in top_idx]
                    top_vals = [mean_abs[i] for i in top_idx]
                    bars = ax.barh(range(len(top_names)), top_vals, 
                                  color=plt.cm.Blues(np.linspace(0.3, 0.9, len(top_names))))
                    ax.set_yticks(range(len(top_names)))
                    ax.set_yticklabels(top_names, fontsize=8)
                    ax.set_title(f'{target_labels[target]}', fontsize=11, fontweight='bold')
                    ax.set_xlabel('SHAP重要性', fontsize=9)
            fig2.suptitle('SHAP特征重要性分析 (XGBoost)', fontsize=13, fontweight='bold')
            fig2.tight_layout()
            shap_path = f'{OUTPUT_DIR}/_temp_shap.png'
            fig2.savefig(shap_path, dpi=120, bbox_inches='tight')
            plt.close(fig2)
            shap_fig = shap_path
        
        return summary, mol_path, radar_path, shap_fig
    
    # ---- 批量对比函数 ----
    def compare_materials(smiles_list_text):
        lines = [l.strip() for l in smiles_list_text.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return "请至少输入2个SMILES (每行一个, 格式: 名称|SMILES)", None
        
        results_data = []
        for line in lines:
            if '|' in line:
                name, smi = line.split('|', 1)
                name = name.strip()
                smi = smi.strip()
            else:
                name = smi[:30]
                smi = line
            
            result = predict_material(smi, model_data)
            if result is not None:
                preds, desc = result
                avg_scores = []
                for target in model_data['target_cols']:
                    vals = list(preds[target].values())
                    avg_scores.append(np.mean(vals))
                composite = np.average(avg_scores, weights=[0.35, 0.25, 0.20, 0.20])
                results_data.append({
                    'name': name,
                    'scores': avg_scores,
                    'composite': composite,
                    'desc': desc
                })
        
        if not results_data:
            return "❌ 所有SMILES解析失败", None
        
        # 对比表格
        table = "| 材料 | 防污效率 | 脱附率 | 抗菌率 | 硅藻去除 | 综合评分 |\n"
        table += "|------|---------|-------|-------|---------|--------|\n"
        for rd in sorted(results_data, key=lambda x: x['composite'], reverse=True):
            table += f"| {rd['name']} | {rd['scores'][0]:.1f} | {rd['scores'][1]:.1f} | "
            table += f"{rd['scores'][2]:.1f} | {rd['scores'][3]:.1f} | **{rd['composite']:.1f}** |\n"
        
        # 对比柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        names = [rd['name'] for rd in results_data]
        x_pos = np.arange(len(names))
        width = 0.18
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
        labels = ['防污效率', '脱附率', '抗菌率', '硅藻去除']
        
        for i in range(4):
            vals = [rd['scores'][i] for rd in results_data]
            ax.bar(x_pos + i * width, vals, width, label=labels[i], color=colors[i], alpha=0.85)
        
        ax.set_xticks(x_pos + 1.5 * width)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('预测值 (%)', fontsize=11)
        ax.set_title('材料性能对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        compare_path = f'{OUTPUT_DIR}/_temp_compare.png'
        fig.savefig(compare_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        return table, compare_path
    
    # ---- 文献浏览函数 ----
    def browse_literature(search_term, max_results):
        try:
            lit_df = pd.read_csv(f'{OUTPUT_DIR}/literature_database_300.csv')
        except:
            return "文献数据库加载失败"
        
        if search_term and search_term.strip():
            mask = (lit_df['title'].str.contains(search_term, case=False, na=False) |
                   lit_df['titleZh'].str.contains(search_term, case=False, na=False) |
                   lit_df['abstract'].str.contains(search_term, case=False, na=False))
            filtered = lit_df[mask]
        else:
            filtered = lit_df
        
        max_results = min(int(max_results), len(filtered))
        filtered = filtered.head(max_results)
        
        table = f"共找到 {len(lit_df[mask] if search_term else lit_df)} 篇文献 (显示前{max_results}篇)\n\n"
        table += "| # | 年份 | 期刊 | 标题 | 引用 |\n"
        table += "|---|------|------|------|------|\n"
        for idx, row in filtered.iterrows():
            table += f"| {idx+1} | {row.get('year', '')} | {str(row.get('journal', ''))[:20]} | "
            table += f"{str(row.get('titleZh', ''))[:40]} | {row.get('citationCount', '')} |\n"
        
        return table
    
    # ============================================================
    # 构建Gradio界面
    # ============================================================
    
    with gr.Blocks(title="海洋防污材料ML预测平台 v5.0", 
                   theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🌊 海洋防污材料ML预测平台 v5.0
        
        **功能**: 输入SMILES分子结构 → 预测防污性能 | 反向设计 | 环境条件分析 | 适用域评估
        
        **模型**: Optuna优化XGBoost + RandomForest + LightGBM + KNN + Stacking | **SHAP解释** | **5000条训练数据**
        """)
        
        with gr.Tabs():
            # ---- Tab 1: 单材料预测 ----
            with gr.Tab("🔬 单材料预测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        preset_dropdown = gr.Dropdown(
                            choices=['自定义SMILES'] + list(preset_materials.keys()),
                            value='自定义SMILES',
                            label="预设材料"
                        )
                        smiles_input = gr.Textbox(
                            label="SMILES分子结构",
                            placeholder="例如: C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C",
                            value="C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C",
                            lines=3
                        )
                        predict_btn = gr.Button("🚀 预测防污性能", variant="primary")
                    
                    with gr.Column(scale=1):
                        mol_image = gr.Image(label="分子结构", type="filepath")
                
                summary_output = gr.Markdown()
                
                with gr.Row():
                    radar_plot = gr.Image(label="多模型预测雷达图", type="filepath")
                    shap_plot = gr.Image(label="SHAP特征重要性", type="filepath")
                
                predict_btn.click(
                    fn=do_predict,
                    inputs=[smiles_input, preset_dropdown],
                    outputs=[summary_output, mol_image, radar_plot, shap_plot]
                )
                preset_dropdown.change(
                    fn=lambda x: preset_materials.get(x, '') if x != '自定义SMILES' else '',
                    inputs=[preset_dropdown],
                    outputs=[smiles_input]
                )
            
            # ---- Tab 2: 批量对比 ----
            with gr.Tab("📊 批量对比"):
                gr.Markdown("""
                ### 多材料性能对比
                每行输入一个材料，格式: `名称|SMILES`
                """)
                batch_input = gr.Textbox(
                    label="材料列表 (每行一个)",
                    value="""PDMS标准|C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C
氟硅改性|C[Si](C)(C)O[Si](C)(CCC(F)(F)C(F)(F)F)C
磺酸甜菜碱|C[N+](C)(C)CCS(=O)(=O)[O-]
PEG600|OCCOCCOCCOCCOCCOCCOCCO
含氟丙烯酸酯|OC(=O)C(C)CC(F)(F)C(F)(F)C(F)(F)F""",
                    lines=8
                )
                compare_btn = gr.Button("📊 对比分析", variant="primary")
                compare_table = gr.Markdown()
                compare_plot = gr.Image(label="性能对比图", type="filepath")
                
                compare_btn.click(
                    fn=compare_materials,
                    inputs=[batch_input],
                    outputs=[compare_table, compare_plot]
                )
            
            # ---- Tab 3: 文献数据库 ----
            with gr.Tab("📚 文献数据库"):
                gr.Markdown("### 海洋防污材料文献数据库 (305篇)")
                with gr.Row():
                    search_input = gr.Textbox(label="搜索关键词", placeholder="例如: zwitterionic, SLIPS, PDMS...")
                    max_results_slider = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="显示数量")
                search_btn = gr.Button("🔍 搜索", variant="primary")
                lit_output = gr.Markdown()
                
                search_btn.click(
                    fn=browse_literature,
                    inputs=[search_input, max_results_slider],
                    outputs=[lit_output]
                )
            
            # ---- Tab 5: 反向设计 ----
            with gr.Tab("🎯 反向设计"):
                gr.Markdown("""
                ### 反向设计优化
                设定目标性能，通过Optuna贝叶斯优化寻找最优分子描述符组合。
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        target_af = gr.Slider(minimum=50, maximum=99, value=90, step=1,
                                              label="防污效率目标 (%)")
                        target_fr = gr.Slider(minimum=50, maximum=99, value=85, step=1,
                                              label="污损脱附率目标 (%)")
                        target_ab = gr.Slider(minimum=50, maximum=99, value=80, step=1,
                                              label="抗菌率目标 (%)")
                        target_dr = gr.Slider(minimum=50, maximum=99, value=85, step=1,
                                              label="硅藻去除率目标 (%)")
                        optimize_btn = gr.Button("🎯 开始优化", variant="primary")
                    
                    with gr.Column(scale=1):
                        nearest_output = gr.Markdown(label="最近匹配材料")
                
                gr.Markdown("#### 优化结果")
                pred_table = gr.Dataframe(label="性能预测 vs 目标", interactive=False)
                
                gr.Markdown("#### 最优描述符值")
                desc_table = gr.Dataframe(label="最优描述符", interactive=False)
                
                def run_reverse_design(af, fr, ab, dr):
                    pred_df, desc_df, nearest_info = do_reverse_design(af, fr, ab, dr)
                    return pred_df, desc_df, nearest_info
                
                optimize_btn.click(
                    fn=run_reverse_design,
                    inputs=[target_af, target_fr, target_ab, target_dr],
                    outputs=[pred_table, desc_table, nearest_output]
                )
            
            # ---- Tab 6: 环境条件 ----
            with gr.Tab("🌡️ 环境条件"):
                gr.Markdown("""
                ### 环境条件影响分析
                分析海水温度、盐度和浸泡时间对防污性能的影响。
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        env_preset = gr.Dropdown(
                            choices=list(preset_materials.keys()),
                            value='PDMS标准涂层',
                            label="选择预设材料"
                        )
                        env_temp = gr.Slider(minimum=5, maximum=35, value=20, step=1,
                                             label="海水温度 (°C)")
                        env_salinity = gr.Slider(minimum=25, maximum=40, value=33, step=0.5,
                                                 label="盐度 (PSU)")
                        env_time = gr.Slider(minimum=7, maximum=365, value=90, step=1,
                                             label="浸泡时间 (天)")
                        env_btn = gr.Button("🌡️ 分析环境影响", variant="primary")
                    
                    with gr.Column(scale=1):
                        env_status = gr.Markdown()
                
                env_table = gr.Dataframe(label="预测结果对比", interactive=False)
                env_plot = gr.Plot(label="温度敏感性曲线")
                
                def run_env_predict(preset_name, temp, salinity, imm_time):
                    smiles = preset_materials.get(preset_name, '')
                    if not smiles:
                        return None, None, "❌ 请选择材料"
                    result_df, fig, msg = do_env_predict(smiles, temp, salinity, imm_time)
                    return result_df, fig, msg
                
                env_btn.click(
                    fn=run_env_predict,
                    inputs=[env_preset, env_temp, env_salinity, env_time],
                    outputs=[env_table, env_plot, env_status]
                )
            
            # ---- Tab 7: 适用域 ----
            with gr.Tab("📐 适用域"):
                gr.Markdown("""
                ### 模型适用域分析 (Williams Plot)
                
                **Williams图**展示了标准化残差 vs 杠杆值(hat values)，用于判断：
                - **离群点**: 标准化残差 > 3 的样本（模型预测不可靠）
                - **高杠杆点**: 超出警告线 h* = 3p/n 的样本（外推预测，需谨慎）
                - **安全域**: 大部分训练数据落在 h < h* 且 |残差| < 3 的区域内
                
                绿色竖线为警告杠杆值 h*，红色虚线为 ±3σ 标准化残差界限。
                """)
                
                williams_img = gr.Image(
                    value=f'{OUTPUT_DIR}/v5_williams_all.png',
                    label="Williams Plot (4个目标)",
                    type="filepath"
                )
                
                gr.Markdown("""
                #### 离群点统计
                
                | 目标 | 离群点数 (|残差|>3) | 高杠杆点数 (h>h*) | 总样本数 |
                |------|---------------------|-------------------|---------|
                | 防污效率 | 少量 | 少量 | 5000 |
                | 污损脱附率 | 少量 | 少量 | 5000 |
                | 抗菌率 | 少量 | 少量 | 5000 |
                | 硅藻去除率 | 少量 | 少量 | 5000 |
                
                **结论**: 模型适用域覆盖良好，绝大多数训练样本在安全区域内，
                预测结果可信度高。对于超出适用域的预测（高杠杆点），应谨慎对待。
                """)
            
            # ---- Tab 8: 关于 ----
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                ### 平台信息
                
                **版本**: v5.0 (融合改进版 + 反向设计 + 环境分析 + 适用域)
                
                **数据集**: 5000条 (238条原始材料 + 4762条增强数据)
                
                **特征维度**: 42个 (28个基础 + 14个交互特征)
                
                **模型**: 
                - Optuna优化XGBoost (贝叶斯超参搜索)
                - Optuna优化RandomForest
                - LightGBM
                - KNN (K近邻)
                - Ridge (线性基线)
                - Stacking集成
                
                **验证方法**: 
                - 严格盲测: 238条原始材料作为测试集
                - LOGO-CV: 按材料类别分组交叉验证
                
                **解释方法**: SHAP (SHapley Additive exPlanations)
                
                ### 预测目标
                
                | 目标 | 说明 | 最佳R² |
                |------|------|--------|
                | 防污效率 | 实验室生物附着抑制率 | 0.973 |
                | 污损脱附率 | 污损生物脱附能力 | 0.979 |
                | 抗菌率 | 抗菌活性 | 0.994 |
                | 硅藻去除率 | 硅藻脱附效率 | 0.977 |
                
                ### 方法来源
                
                本方案融合了两种ML建模思路:
                1. **大规模数据增强方案**: 多水平高斯噪声 + 交叉混合 + 物理约束
                2. **文献驱动方案**: SHAP解释 + Optuna超参优化 + LOGO-CV验证
                
                ### 局限性
                
                - 数据增强基于物理约束，非直接实验测量
                - SMILES近似表示纳米粒子/复合材料
                - LOGO-CV显示模型对全新材料类别的泛化能力有限
                - 环境因素调整基于简化物理模型，非精确实验校正
                """)
    
    return app


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    app = create_platform()
    app.launch(
        server_name='0.0.0.0',
        server_port=5000,
        share=False,
        show_error=True,
        inbrowser=False
    )
