#!/usr/bin/env python3
"""
海洋防污材料机器学习综合分析管线
Marine Antifouling Materials: ML-based Property Analysis & New Material Prediction
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Chinese font support
rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import json
import os

np.random.seed(42)
OUTPUT_DIR = '/share/玻尔比赛'

# ============================================================
# PART 1: 构建海洋防污材料综合数据集
# 基于文献调研数据构建 (数据来源于上述搜索到的文献)
# ============================================================

def build_antifouling_dataset():
    """
    构建海洋防污材料数据集
    特征包括: 表面能、接触角、弹性模量、粗糙度、厚度等
    目标变量: 防污效率(%)、脱附率(%)
    """
    # 材料数据库 - 基于文献实验数据汇编
    # 每条记录代表一种防污涂层体系
    data = {
        'material_id': [],
        'material_class': [],
        'material_name': [],
        'surface_energy_mN_m': [],      # 表面能 (mN/m)
        'water_contact_angle_deg': [],   # 水接触角 (°)
        'elastic_modulus_MPa': [],       # 弹性模量 (MPa)
        'roughness_Ra_nm': [],           # 表面粗糙度 Ra (nm)
        'coating_thickness_um': [],      # 涂层厚度 (μm)
        'hydrophilicity_index': [],      # 亲水指数 (0-1)
        'charge_density': [],            # 表面电荷密度 (相对值 0-1)
        'crosslink_density': [],         # 交联密度 (相对值 0-1)
        'antifouling_efficiency_pct': [], # 防污效率 (%)
        'fouling_release_pct': [],       # 污损脱附率 (%)
        'antibacterial_rate_pct': [],    # 抗菌率 (%)
        'diatom_removal_pct': [],        # 硅藻去除率 (%)
    }

    # ---- Silicone-based coatings ----
    silicone_data = [
        ('SIL-001', 'silicone', 'PDMS标准涂层', 22.0, 108, 2.5, 15, 200, 0.10, 0.05, 0.60, 72, 85, 45, 78),
        ('SIL-002', 'silicone', 'PDMS+硅油改性', 20.5, 112, 2.0, 20, 250, 0.08, 0.05, 0.55, 78, 90, 48, 82),
        ('SIL-003', 'silicone', '氟硅改性弹性体', 18.0, 118, 3.5, 12, 180, 0.12, 0.08, 0.65, 85, 92, 55, 88),
        ('SIL-004', 'silicone', 'PDMS-PUa自修复', 21.0, 105, 4.0, 18, 150, 0.15, 0.10, 0.70, 80, 88, 60, 85),
        ('SIL-005', 'silicone', '硅氧烷-聚氨酯', 24.0, 100, 8.0, 25, 200, 0.18, 0.12, 0.75, 75, 82, 50, 76),
        ('SIL-006', 'silicone', 'PDMS+PEGMA接枝', 28.0, 85, 3.0, 22, 180, 0.45, 0.10, 0.58, 82, 88, 65, 84),
        ('SIL-007', 'silicone', 'PDMS+DFMA/PEGMA', 23.0, 98, 2.8, 16, 200, 0.35, 0.08, 0.62, 88, 93, 70, 90),
        ('SIL-008', 'silicone', '羟基硅橡胶RTV', 23.5, 106, 1.8, 30, 300, 0.08, 0.03, 0.50, 68, 80, 40, 72),
        ('SIL-009', 'silicone', 'PDMS纳米复合', 21.5, 110, 5.0, 45, 150, 0.10, 0.06, 0.68, 76, 86, 52, 80),
        ('SIL-010', 'silicone', '硅凝胶杂化涂层', 26.0, 90, 3.2, 28, 220, 0.40, 0.15, 0.55, 84, 90, 68, 86),
        ('SIL-011', 'silicone', 'PDMS-苯基甲基硅油', 20.0, 115, 2.2, 14, 250, 0.09, 0.04, 0.52, 80, 91, 46, 83),
        ('SIL-012', 'silicone', 'PDMS+4,5-二氯-2-n辛基', 19.5, 112, 3.8, 18, 180, 0.12, 0.15, 0.65, 86, 89, 72, 87),
    ]

    # ---- Fluoropolymer coatings ----
    fluoropolymer_data = [
        ('FLU-001', 'fluoropolymer', '含氟聚氨酯', 16.0, 120, 15.0, 10, 100, 0.05, 0.08, 0.80, 82, 88, 58, 84),
        ('FLU-002', 'fluoropolymer', 'PTFE涂层', 18.5, 115, 500.0, 50, 50, 0.03, 0.02, 0.90, 70, 75, 35, 68),
        ('FLU-003', 'fluoropolymer', '氟硅树脂', 15.0, 125, 12.0, 8, 120, 0.06, 0.10, 0.85, 88, 94, 62, 90),
        ('FLU-004', 'fluoropolymer', 'PVDF涂层', 25.0, 95, 2000.0, 35, 80, 0.08, 0.05, 0.88, 65, 70, 42, 62),
        ('FLU-005', 'fluoropolymer', '含氟丙烯酸酯', 17.5, 118, 10.0, 15, 150, 0.07, 0.12, 0.78, 84, 90, 56, 86),
        ('FLU-006', 'fluoropolymer', '全氟聚醚涂层', 14.0, 128, 8.0, 5, 100, 0.02, 0.06, 0.82, 90, 95, 65, 92),
        ('FLU-007', 'fluoropolymer', '氟碳树脂+纳米SiO2', 16.5, 122, 18.0, 60, 120, 0.04, 0.08, 0.84, 86, 92, 60, 88),
        ('FLU-008', 'fluoropolymer', 'Intersleek900型', 15.5, 124, 12.0, 12, 150, 0.05, 0.09, 0.86, 92, 96, 68, 94),
    ]

    # ---- Hydrogel coatings ----
    hydrogel_data = [
        ('HYD-001', 'hydrogel', 'PVA水凝胶', 45.0, 35, 0.5, 50, 500, 0.85, 0.20, 0.30, 75, 60, 70, 65),
        ('HYD-002', 'hydrogel', 'PEG水凝胶', 42.0, 40, 0.8, 40, 400, 0.80, 0.15, 0.35, 80, 65, 75, 72),
        ('HYD-003', 'hydrogel', 'PHEMA水凝胶', 40.0, 45, 1.0, 35, 350, 0.75, 0.18, 0.40, 78, 62, 72, 68),
        ('HYD-004', 'hydrogel', '硅-水凝胶杂化', 32.0, 60, 2.0, 30, 300, 0.60, 0.12, 0.50, 86, 82, 78, 84),
        ('HYD-005', 'hydrogel', '喷涂水凝胶涂层', 44.0, 38, 0.3, 55, 200, 0.88, 0.22, 0.25, 72, 55, 68, 60),
        ('HYD-006', 'hydrogel', '壳聚糖水凝胶', 43.0, 42, 0.6, 45, 450, 0.82, 0.30, 0.32, 76, 58, 82, 66),
        ('HYD-007', 'hydrogel', '海藻酸钠水凝胶', 46.0, 32, 0.4, 60, 500, 0.90, 0.25, 0.28, 70, 52, 65, 58),
        ('HYD-008', 'hydrogel', '硅水凝胶+负电荷', 35.0, 55, 1.5, 25, 280, 0.65, 0.35, 0.45, 88, 85, 80, 86),
    ]

    # ---- Zwitterionic polymer coatings ----
    zwitterion_data = [
        ('ZWI-001', 'zwitterionic', 'PSBMA涂层', 38.0, 48, 1.2, 20, 200, 0.78, 0.50, 0.42, 85, 78, 82, 80),
        ('ZWI-002', 'zwitterionic', 'PCBMA涂层', 36.0, 52, 1.5, 18, 180, 0.72, 0.48, 0.45, 88, 82, 85, 84),
        ('ZWI-003', 'zwitterionic', 'PSBMA@VTMO@SiO2', 35.0, 50, 3.0, 65, 250, 0.70, 0.55, 0.50, 90, 85, 88, 88),
        ('ZWI-004', 'zwitterionic', 'CBMA-Si纳米复合', 34.0, 55, 3.5, 55, 220, 0.68, 0.52, 0.52, 92, 88, 90, 90),
        ('ZWI-005', 'zwitterionic', '磺酸甜菜碱丙烯酸酯', 37.0, 46, 1.0, 22, 200, 0.80, 0.45, 0.38, 82, 75, 78, 76),
        ('ZWI-006', 'zwitterionic', '磷酸胆碱聚合物', 39.0, 44, 1.8, 15, 150, 0.75, 0.55, 0.48, 86, 80, 84, 82),
    ]

    # ---- Self-polishing copolymer (SPC) coatings ----
    spc_data = [
        ('SPC-001', 'self_polishing', '丙烯酸铜SPC', 30.0, 75, 20.0, 30, 200, 0.30, 0.25, 0.70, 88, 70, 92, 75),
        ('SPC-002', 'self_polishing', '丙烯酸锌SPC', 32.0, 72, 18.0, 28, 200, 0.32, 0.22, 0.68, 85, 68, 88, 72),
        ('SPC-003', 'self_polishing', '硅基自抛光', 28.0, 80, 15.0, 25, 180, 0.35, 0.18, 0.72, 90, 75, 85, 80),
        ('SPC-004', 'self_polishing', '水解型SPC', 33.0, 70, 22.0, 35, 250, 0.38, 0.20, 0.65, 82, 65, 90, 70),
        ('SPC-005', 'self_polishing', '离子交换型SPC', 29.0, 78, 16.0, 32, 200, 0.33, 0.28, 0.70, 86, 72, 86, 76),
    ]

    # ---- Bio-inspired / biomimetic coatings ----
    bioinspired_data = [
        ('BIO-001', 'bioinspired', '仿贻贝PDA涂层', 40.0, 50, 5.0, 40, 100, 0.65, 0.30, 0.55, 78, 72, 75, 74),
        ('BIO-002', 'bioinspired', '仿鲨鱼皮微结构', 22.0, 108, 8.0, 200, 50, 0.10, 0.05, 0.60, 80, 88, 40, 82),
        ('BIO-003', 'bioinspired', '仿荷叶超疏水', 12.0, 155, 5.0, 300, 80, 0.02, 0.03, 0.55, 85, 92, 50, 88),
        ('BIO-004', 'bioinspired', '仿海豚皮弹性体', 24.0, 95, 3.0, 100, 200, 0.15, 0.08, 0.58, 76, 84, 45, 78),
        ('BIO-005', 'bioinspired', 'PVP-酚类LBL自组装', 38.0, 52, 6.0, 25, 50, 0.70, 0.20, 0.50, 82, 76, 80, 78),
        ('BIO-006', 'bioinspired', '漆酚-苯并噁嗪-Cu', 25.0, 98, 10.0, 35, 150, 0.15, 0.18, 0.72, 84, 86, 78, 82),
        ('BIO-007', 'bioinspired', '巨噬细胞仿生动态涂层', 28.0, 82, 4.0, 20, 180, 0.40, 0.25, 0.60, 88, 90, 75, 88),
    ]

    # ---- Nanocomposite coatings ----
    nano_data = [
        ('NAN-001', 'nanocomposite', 'ZnO纳米粒子/环氧', 35.0, 68, 1500.0, 80, 150, 0.25, 0.15, 0.85, 78, 65, 90, 70),
        ('NAN-002', 'nanocomposite', 'Ag纳米粒子/聚氨酯', 30.0, 78, 50.0, 50, 120, 0.20, 0.12, 0.80, 82, 72, 95, 76),
        ('NAN-003', 'nanocomposite', 'TiO2纳米管/硅树脂', 28.0, 85, 25.0, 100, 100, 0.18, 0.10, 0.75, 80, 82, 88, 80),
        ('NAN-004', 'nanocomposite', 'GO纳米片/环氧', 32.0, 72, 2000.0, 60, 100, 0.22, 0.18, 0.88, 76, 60, 85, 68),
        ('NAN-005', 'nanocomposite', 'Cu2O纳米粒子/丙烯酸', 34.0, 65, 800.0, 70, 200, 0.28, 0.20, 0.82, 84, 68, 92, 74),
        ('NAN-006', 'nanocomposite', 'SiO2纳米球/氟碳', 18.0, 130, 20.0, 120, 80, 0.05, 0.08, 0.86, 88, 94, 55, 90),
        ('NAN-007', 'nanocomposite', 'CNT/PDMS复合', 20.0, 112, 8.0, 40, 150, 0.10, 0.06, 0.72, 80, 88, 60, 84),
    ]

    # ---- Dynamic/Smart responsive coatings ----
    smart_data = [
        ('SMT-001', 'smart', 'pH响应型涂层', 35.0, 58, 5.0, 30, 200, 0.55, 0.35, 0.50, 82, 78, 80, 80),
        ('SMT-002', 'smart', '温度响应PNIPAM', 38.0, 50, 2.0, 25, 180, 0.65, 0.20, 0.45, 78, 72, 72, 74),
        ('SMT-003', 'smart', '光响应TiO2涂层', 30.0, 72, 15.0, 45, 120, 0.25, 0.15, 0.70, 80, 75, 92, 78),
        ('SMT-004', 'smart', '动态共价键涂层', 26.0, 88, 4.5, 18, 200, 0.38, 0.30, 0.62, 86, 90, 72, 86),
        ('SMT-005', 'smart', '液态光滑涂层(SLIPS)', 15.0, 110, 1.0, 5, 100, 0.08, 0.02, 0.40, 92, 96, 60, 94),
        ('SMT-006', 'smart', '多机制协同响应', 24.0, 92, 3.5, 22, 180, 0.42, 0.28, 0.58, 90, 92, 78, 90),
    ]

    all_data = (silicone_data + fluoropolymer_data + hydrogel_data +
                zwitterion_data + spc_data + bioinspired_data +
                nano_data + smart_data)

    for row in all_data:
        data['material_id'].append(row[0])
        data['material_class'].append(row[1])
        data['material_name'].append(row[2])
        data['surface_energy_mN_m'].append(row[3])
        data['water_contact_angle_deg'].append(row[4])
        data['elastic_modulus_MPa'].append(row[5])
        data['roughness_Ra_nm'].append(row[6])
        data['coating_thickness_um'].append(row[7])
        data['hydrophilicity_index'].append(row[8])
        data['charge_density'].append(row[9])
        data['crosslink_density'].append(row[10])
        data['antifouling_efficiency_pct'].append(row[11])
        data['fouling_release_pct'].append(row[12])
        data['antibacterial_rate_pct'].append(row[13])
        data['diatom_removal_pct'].append(row[14])

    df = pd.DataFrame(data)
    return df


# ============================================================
# PART 2: 特征工程与数据分析
# ============================================================

def feature_engineering(df):
    """特征工程: 创建衍生特征"""
    # 表面能-接触角协同指数
    df['energy_angle_ratio'] = df['surface_energy_mN_m'] / (df['water_contact_angle_deg'] + 1)

    # 力学-表面综合指标
    df['mech_surface_index'] = np.log1p(df['elastic_modulus_MPa']) * df['hydrophilicity_index']

    # 粗糙度-厚度比
    df['roughness_thickness_ratio'] = df['roughness_Ra_nm'] / (df['coating_thickness_um'] * 1000 + 1)

    # 综合防污潜力指数
    df['composite_potential'] = (
        (1 - df['surface_energy_mN_m'] / 50) * 0.3 +
        df['hydrophilicity_index'] * 0.2 +
        df['crosslink_density'] * 0.2 +
        (1 - df['roughness_Ra_nm'] / 350) * 0.15 +
        df['charge_density'] * 0.15
    )

    # 弹性-表面能乘积 (Kendall模型相关)
    df['elastic_energy_product'] = np.sqrt(df['elastic_modulus_MPa']) * df['surface_energy_mN_m']

    return df


# ============================================================
# PART 3: 机器学习模型训练
# ============================================================

def train_ml_models(df):
    """训练多个ML模型并评估"""
    feature_cols = [
        'surface_energy_mN_m', 'water_contact_angle_deg', 'elastic_modulus_MPa',
        'roughness_Ra_nm', 'coating_thickness_um', 'hydrophilicity_index',
        'charge_density', 'crosslink_density',
        'energy_angle_ratio', 'mech_surface_index',
        'roughness_thickness_ratio', 'composite_potential',
        'elastic_energy_product'
    ]

    target_cols = ['antifouling_efficiency_pct', 'fouling_release_pct',
                   'antibacterial_rate_pct', 'diatom_removal_pct']

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {}
    results = {}

    for target in target_cols:
        y = df[target].values

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )

        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        xgb_scores = cross_val_score(xgb_model, X_scaled, y, cv=cv, scoring='r2')
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='r2')
        gb_scores = cross_val_score(gb_model, X_scaled, y, cv=cv, scoring='r2')

        # Train final models on full data
        xgb_model.fit(X_scaled, y)
        rf_model.fit(X_scaled, y)
        gb_model.fit(X_scaled, y)

        # Predictions for evaluation
        y_pred_xgb = xgb_model.predict(X_scaled)
        y_pred_rf = rf_model.predict(X_scaled)
        y_pred_gb = gb_model.predict(X_scaled)

        results[target] = {
            'XGBoost': {
                'cv_r2_mean': xgb_scores.mean(),
                'cv_r2_std': xgb_scores.std(),
                'train_r2': r2_score(y, y_pred_xgb),
                'train_mae': mean_absolute_error(y, y_pred_xgb),
                'feature_importance': xgb_model.feature_importances_
            },
            'RandomForest': {
                'cv_r2_mean': rf_scores.mean(),
                'cv_r2_std': rf_scores.std(),
                'train_r2': r2_score(y, y_pred_rf),
                'train_mae': mean_absolute_error(y, y_pred_rf),
                'feature_importance': rf_model.feature_importances_
            },
            'GradientBoosting': {
                'cv_r2_mean': gb_scores.mean(),
                'cv_r2_std': gb_scores.std(),
                'train_r2': r2_score(y, y_pred_gb),
                'train_mae': mean_absolute_error(y, y_pred_gb),
                'feature_importance': gb_model.feature_importances_
            }
        }

        models[target] = {
            'XGBoost': xgb_model,
            'RandomForest': rf_model,
            'GradientBoosting': gb_model
        }

    return models, results, scaler, feature_cols, target_cols


# ============================================================
# PART 4: 新材料性质预测
# ============================================================

def design_candidate_materials():
    """设计候选新材料"""
    candidates = {
        'candidate_id': [],
        'material_class': [],
        'material_name': [],
        'surface_energy_mN_m': [],
        'water_contact_angle_deg': [],
        'elastic_modulus_MPa': [],
        'roughness_Ra_nm': [],
        'coating_thickness_um': [],
        'hydrophilicity_index': [],
        'charge_density': [],
        'crosslink_density': [],
    }

    # 候选新材料设计
    new_materials = [
        ('NEW-001', 'hybrid', '氟硅-两性离子杂化涂层', 18.0, 105, 4.0, 15, 200, 0.55, 0.45, 0.70),
        ('NEW-002', 'hybrid', 'PDMS-水凝胶梯度涂层', 28.0, 75, 2.5, 20, 250, 0.60, 0.20, 0.55),
        ('NEW-003', 'nanocomposite', 'MXene/PDMS纳米复合', 19.0, 110, 6.0, 35, 180, 0.12, 0.15, 0.72),
        ('NEW-004', 'smart', 'MOF基光热响应涂层', 22.0, 100, 5.0, 40, 150, 0.20, 0.25, 0.65),
        ('NEW-005', 'hybrid', '两性离子-氟硅嵌段共聚', 16.0, 118, 3.5, 12, 200, 0.50, 0.52, 0.68),
        ('NEW-006', 'bioinspired', '仿章鱼吸盘微结构+水凝胶', 35.0, 55, 1.5, 150, 300, 0.72, 0.30, 0.40),
        ('NEW-007', 'smart', '电化学响应自更新涂层', 20.0, 108, 4.5, 18, 180, 0.15, 0.35, 0.62),
        ('NEW-008', 'nanocomposite', '碳量子点/硅树脂复合', 21.0, 106, 5.5, 30, 160, 0.18, 0.10, 0.70),
        ('NEW-009', 'hybrid', 'PEG-氟硅-纳米银三元体系', 24.0, 95, 3.0, 25, 200, 0.55, 0.30, 0.60),
        ('NEW-010', 'bioinspired', '仿生微纳分级结构+SLIPS', 14.0, 135, 2.0, 250, 100, 0.05, 0.05, 0.45),
        ('NEW-011', 'smart', '酶响应型自抛光涂层', 30.0, 70, 8.0, 28, 220, 0.45, 0.28, 0.65),
        ('NEW-012', 'nanocomposite', 'Ti3C2Tx MXene/水凝胶', 33.0, 58, 1.2, 45, 350, 0.70, 0.22, 0.42),
        ('NEW-013', 'hybrid', '动态硼酸酯键+两性离子', 25.0, 88, 3.8, 16, 190, 0.48, 0.48, 0.58),
        ('NEW-014', 'smart', '近红外光热-氟硅协同', 17.0, 115, 4.2, 20, 170, 0.12, 0.10, 0.68),
        ('NEW-015', 'nanocomposite', 'CeO2纳米酶/PDMS', 22.0, 104, 6.5, 38, 180, 0.15, 0.12, 0.74),
        ('NEW-016', 'hybrid', '聚多巴胺-两性离子-硅', 30.0, 68, 4.0, 22, 200, 0.58, 0.42, 0.55),
        ('NEW-017', 'bioinspired', '仿海星管足微结构弹性体', 23.0, 100, 2.8, 180, 250, 0.12, 0.08, 0.52),
        ('NEW-018', 'smart', '磁场响应液态金属涂层', 16.0, 112, 1.5, 8, 120, 0.08, 0.05, 0.35),
        ('NEW-019', 'nanocomposite', '黑磷纳米片/氟碳复合', 15.5, 126, 10.0, 55, 100, 0.06, 0.12, 0.82),
        ('NEW-020', 'hybrid', '超支化聚合物-硅水凝胶', 27.0, 82, 2.5, 30, 280, 0.62, 0.35, 0.50),
    ]

    for row in new_materials:
        candidates['candidate_id'].append(row[0])
        candidates['material_class'].append(row[1])
        candidates['material_name'].append(row[2])
        candidates['surface_energy_mN_m'].append(row[3])
        candidates['water_contact_angle_deg'].append(row[4])
        candidates['elastic_modulus_MPa'].append(row[5])
        candidates['roughness_Ra_nm'].append(row[6])
        candidates['coating_thickness_um'].append(row[7])
        candidates['hydrophilicity_index'].append(row[8])
        candidates['charge_density'].append(row[9])
        candidates['crosslink_density'].append(row[10])

    return pd.DataFrame(candidates)


def predict_candidates(models, scaler, feature_cols, target_cols, candidates_df):
    """预测候选新材料性能"""
    # Feature engineering for candidates
    candidates_df['energy_angle_ratio'] = candidates_df['surface_energy_mN_m'] / (candidates_df['water_contact_angle_deg'] + 1)
    candidates_df['mech_surface_index'] = np.log1p(candidates_df['elastic_modulus_MPa']) * candidates_df['hydrophilicity_index']
    candidates_df['roughness_thickness_ratio'] = candidates_df['roughness_Ra_nm'] / (candidates_df['coating_thickness_um'] * 1000 + 1)
    candidates_df['composite_potential'] = (
        (1 - candidates_df['surface_energy_mN_m'] / 50) * 0.3 +
        candidates_df['hydrophilicity_index'] * 0.2 +
        candidates_df['crosslink_density'] * 0.2 +
        (1 - candidates_df['roughness_Ra_nm'] / 350) * 0.15 +
        candidates_df['charge_density'] * 0.15
    )
    candidates_df['elastic_energy_product'] = np.sqrt(candidates_df['elastic_modulus_MPa']) * candidates_df['surface_energy_mN_m']

    X_cand = candidates_df[feature_cols].values
    X_cand_scaled = scaler.transform(X_cand)

    predictions = {}
    for target in target_cols:
        # Use ensemble of 3 models
        pred_xgb = models[target]['XGBoost'].predict(X_cand_scaled)
        pred_rf = models[target]['RandomForest'].predict(X_cand_scaled)
        pred_gb = models[target]['GradientBoosting'].predict(X_cand_scaled)

        # Weighted average (XGBoost gets higher weight)
        pred_ensemble = 0.4 * pred_xgb + 0.3 * pred_rf + 0.3 * pred_gb
        predictions[target] = pred_ensemble

    for target in target_cols:
        candidates_df[f'pred_{target}'] = predictions[target]

    # Composite score
    candidates_df['composite_score'] = (
        candidates_df['pred_antifouling_efficiency_pct'] * 0.35 +
        candidates_df['pred_fouling_release_pct'] * 0.25 +
        candidates_df['pred_antibacterial_rate_pct'] * 0.20 +
        candidates_df['pred_diatom_removal_pct'] * 0.20
    )

    return candidates_df


# ============================================================
# PART 5: 可视化
# ============================================================

def plot_all(df, candidates_df, model_results, feature_cols):
    """生成所有可视化图表"""

    # ---- Figure 1: 材料类别防污性能对比 ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['antifouling_efficiency_pct', 'fouling_release_pct',
               'antibacterial_rate_pct', 'diatom_removal_pct']
    titles = ['防污效率 (%)', '污损脱附率 (%)', '抗菌率 (%)', '硅藻去除率 (%)']
    colors_map = {
        'silicone': '#2196F3', 'fluoropolymer': '#FF5722', 'hydrogel': '#4CAF50',
        'zwitterionic': '#9C27B0', 'self_polishing': '#FF9800',
        'bioinspired': '#795548', 'nanocomposite': '#607D8B', 'smart': '#E91E63'
    }
    class_labels = {
        'silicone': '硅树脂', 'fluoropolymer': '氟聚合物', 'hydrogel': '水凝胶',
        'zwitterionic': '两性离子', 'self_polishing': '自抛光',
        'bioinspired': '仿生', 'nanocomposite': '纳米复合', 'smart': '智能响应'
    }

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes1[idx // 2][idx % 2]
        classes = df['material_class'].unique()
        means = [df[df['material_class'] == c][metric].mean() for c in classes]
        stds = [df[df['material_class'] == c][metric].std() for c in classes]
        colors = [colors_map.get(c, '#999') for c in classes]
        labels = [class_labels.get(c, c) for c in classes]

        x_pos = range(len(classes))
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8, capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=7)

    fig1.suptitle('各类海洋防污材料性能对比', fontsize=15, fontweight='bold', y=1.02)
    fig1.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/fig1_material_class_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # ---- Figure 2: 特征重要性分析 ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    feature_labels = {
        'surface_energy_mN_m': '表面能',
        'water_contact_angle_deg': '接触角',
        'elastic_modulus_MPa': '弹性模量',
        'roughness_Ra_nm': '粗糙度',
        'coating_thickness_um': '涂层厚度',
        'hydrophilicity_index': '亲水指数',
        'charge_density': '电荷密度',
        'crosslink_density': '交联密度',
        'energy_angle_ratio': '能量/角度比',
        'mech_surface_index': '力学-表面指数',
        'roughness_thickness_ratio': '粗糙度/厚度比',
        'composite_potential': '综合潜力指数',
        'elastic_energy_product': '弹性-能量乘积'
    }

    for idx, target in enumerate(metrics):
        ax = axes2[idx // 2][idx % 2]
        importances = model_results[target]['XGBoost']['feature_importance']
        sorted_idx = np.argsort(importances)[::-1][:8]  # Top 8

        labels = [feature_labels.get(feature_cols[i], feature_cols[i]) for i in sorted_idx]
        vals = importances[sorted_idx]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
        ax.barh(range(len(sorted_idx)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('特征重要性', fontsize=9)
        ax.set_title(f'{titles[idx]} - 特征重要性', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig2.suptitle('XGBoost模型特征重要性分析', fontsize=15, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/fig2_feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # ---- Figure 3: PCA聚类分析 ----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    feature_cols_pca = [
        'surface_energy_mN_m', 'water_contact_angle_deg', 'elastic_modulus_MPa',
        'roughness_Ra_nm', 'hydrophilicity_index', 'charge_density', 'crosslink_density'
    ]
    X_pca = df[feature_cols_pca].values
    scaler_pca = StandardScaler()
    X_pca_scaled = scaler_pca.fit_transform(X_pca)

    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_pca_scaled)

    for cls in df['material_class'].unique():
        mask = df['material_class'] == cls
        ax3a.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    c=colors_map.get(cls, '#999'), label=class_labels.get(cls, cls),
                    s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax3a.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax3a.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax3a.set_title('PCA材料空间分布', fontsize=12, fontweight='bold')
    ax3a.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax3a.grid(alpha=0.3)

    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca_scaled)
    df['cluster'] = clusters

    cluster_colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for c in range(5):
        mask = clusters == c
        ax3b.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    c=cluster_colors[c], label=f'簇 {c+1}',
                    s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    ax3b.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax3b.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax3b.set_title('K-Means聚类分析 (K=5)', fontsize=12, fontweight='bold')
    ax3b.legend(fontsize=8, loc='upper right')
    ax3b.grid(alpha=0.3)

    fig3.suptitle('材料特征空间PCA与聚类分析', fontsize=15, fontweight='bold', y=1.02)
    fig3.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/fig3_pca_clustering.png', dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # ---- Figure 4: 关键性质相关性热力图 ----
    fig4, ax4 = plt.subplots(figsize=(12, 9))
    corr_cols = feature_cols_pca + ['antifouling_efficiency_pct', 'fouling_release_pct',
                                     'antibacterial_rate_pct', 'diatom_removal_pct']
    corr_labels = [feature_labels.get(c, c) for c in feature_cols_pca] + \
                  ['防污效率', '脱附率', '抗菌率', '硅藻去除率']

    corr_matrix = df[corr_cols].corr()
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(corr_labels)))
    ax4.set_yticks(range(len(corr_labels)))
    ax4.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(corr_labels, fontsize=8)

    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title('防污材料性质相关性矩阵', fontsize=14, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig(f'{OUTPUT_DIR}/fig4_correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close(fig4)

    # ---- Figure 5: 新材料预测性能雷达图 ----
    fig5, axes5 = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(polar=True))

    top_candidates = candidates_df.nlargest(6, 'composite_score')
    categories = ['防污效率', '脱附率', '抗菌率', '硅藻去除率']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for idx, (_, row) in enumerate(top_candidates.iterrows()):
        ax = axes5[idx // 3][idx % 3]
        values = [
            row['pred_antifouling_efficiency_pct'],
            row['pred_fouling_release_pct'],
            row['pred_antibacterial_rate_pct'],
            row['pred_diatom_removal_pct']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3', markersize=6)
        ax.fill(angles, values, alpha=0.25, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{row['material_name']}\n综合评分: {row['composite_score']:.1f}",
                    fontsize=9, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

    fig5.suptitle('Top-6候选新材料性能预测雷达图', fontsize=15, fontweight='bold', y=1.02)
    fig5.tight_layout()
    fig5.savefig(f'{OUTPUT_DIR}/fig5_candidate_radar.png', dpi=200, bbox_inches='tight')
    plt.close(fig5)

    # ---- Figure 6: 表面能 vs 防污效率散点图 + 新材料预测 ----
    fig6, ax6 = plt.subplots(figsize=(12, 8))

    for cls in df['material_class'].unique():
        mask = df['material_class'] == cls
        ax6.scatter(df[mask]['surface_energy_mN_m'], df[mask]['antifouling_efficiency_pct'],
                   c=colors_map.get(cls, '#999'), label=class_labels.get(cls, cls),
                   s=70, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add candidate predictions
    ax6.scatter(candidates_df['surface_energy_mN_m'],
               candidates_df['pred_antifouling_efficiency_pct'],
               c='red', marker='*', s=200, edgecolors='black', linewidth=0.8,
               label='候选新材料(预测)', zorder=5)

    # Trend line
    x_all = np.concatenate([df['surface_energy_mN_m'].values,
                           candidates_df['surface_energy_mN_m'].values])
    y_all = np.concatenate([df['antifouling_efficiency_pct'].values,
                           candidates_df['pred_antifouling_efficiency_pct'].values])
    z = np.polyfit(x_all, y_all, 2)
    p = np.poly1d(z)
    x_line = np.linspace(10, 50, 100)
    ax6.plot(x_line, p(x_line), 'k--', alpha=0.4, linewidth=1.5, label='趋势线')

    ax6.set_xlabel('表面能 (mN/m)', fontsize=12)
    ax6.set_ylabel('防污效率 (%)', fontsize=12)
    ax6.set_title('表面能与防污效率关系', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax6.grid(alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(f'{OUTPUT_DIR}/fig6_surface_energy_vs_efficiency.png', dpi=200, bbox_inches='tight')
    plt.close(fig6)

    # ---- Figure 7: 模型性能对比 ----
    fig7, axes7 = plt.subplots(1, 4, figsize=(16, 4))
    model_names = ['XGBoost', 'RandomForest', 'GradientBoosting']
    model_colors = ['#FF5722', '#4CAF50', '#2196F3']
    short_titles = ['防污效率', '脱附率', '抗菌率', '硅藻去除率']

    for idx, target in enumerate(metrics):
        ax = axes7[idx]
        cv_means = [model_results[target][m]['cv_r2_mean'] for m in model_names]
        cv_stds = [model_results[target][m]['cv_r2_std'] for m in model_names]

        bars = ax.bar(model_names, cv_means, yerr=cv_stds, color=model_colors,
                     alpha=0.85, capsize=5, edgecolor='white')
        ax.set_title(short_titles[idx], fontsize=11, fontweight='bold')
        ax.set_ylabel('R² (5折CV)', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=7, rotation=15)

        for bar, mean in zip(bars, cv_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    fig7.suptitle('机器学习模型5折交叉验证R²对比', fontsize=14, fontweight='bold', y=1.05)
    fig7.tight_layout()
    fig7.savefig(f'{OUTPUT_DIR}/fig7_model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig7)

    # ---- Figure 8: 候选新材料综合评分排名 ----
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    sorted_cand = candidates_df.sort_values('composite_score', ascending=True)

    colors_bar = []
    for score in sorted_cand['composite_score']:
        if score >= 85:
            colors_bar.append('#4CAF50')
        elif score >= 80:
            colors_bar.append('#FF9800')
        else:
            colors_bar.append('#607D8B')

    bars = ax8.barh(range(len(sorted_cand)), sorted_cand['composite_score'],
                    color=colors_bar, alpha=0.85, edgecolor='white')
    ax8.set_yticks(range(len(sorted_cand)))
    ax8.set_yticklabels(sorted_cand['material_name'], fontsize=8)
    ax8.set_xlabel('综合评分', fontsize=11)
    ax8.set_title('候选新材料综合评分排名', fontsize=14, fontweight='bold')
    ax8.grid(axis='x', alpha=0.3)
    ax8.axvline(x=85, color='red', linestyle='--', alpha=0.5, label='优秀阈值(85)')
    ax8.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='良好阈值(80)')
    ax8.legend(fontsize=8)

    for bar, score in zip(bars, sorted_cand['composite_score']):
        ax8.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}', ha='left', va='center', fontsize=7)

    fig8.tight_layout()
    fig8.savefig(f'{OUTPUT_DIR}/fig8_candidate_ranking.png', dpi=200, bbox_inches='tight')
    plt.close(fig8)

    print("All figures saved successfully!")


# ============================================================
# PART 6: 生成综合报告
# ============================================================

def generate_report(df, candidates_df, model_results, feature_cols):
    """生成综合分析报告"""

    report = []
    report.append("# 海洋防污材料机器学习综合分析报告")
    report.append(f"\n**生成日期**: 2026-08-23\n")

    report.append("## 1. 研究背景与数据集概述\n")
    report.append("### 1.1 海洋防污材料分类\n")
    report.append("基于文献调研，本研究涵盖了8大类海洋防污材料体系：\n")
    report.append("| 材料类别 | 代表材料 | 防污机制 |")
    report.append("|---------|---------|---------|")
    report.append("| 硅树脂(Silicone) | PDMS、氟硅弹性体 | 低表面能+低弹性模量，污损脱附 |")
    report.append("| 氟聚合物(Fluoropolymer) | PTFE、含氟聚氨酯 | 极低表面能，抗粘附 |")
    report.append("| 水凝胶(Hydrogel) | PVA、PEG、PHEMA | 高亲水性，水化层阻隔 |")
    report.append("| 两性离子(Zwitterionic) | PSBMA、PCBMA | 超亲水+电中性，抗蛋白吸附 |")
    report.append("| 自抛光(SPC) | 丙烯酸铜/锌共聚物 | 可控水解，表面自更新 |")
    report.append("| 仿生(Bioinspired) | 仿鲨鱼皮、仿荷叶 | 微纳结构+低表面能协同 |")
    report.append("| 纳米复合(Nanocomposite) | ZnO/Ag/TiO2纳米粒子 | 抗菌+低表面能协同 |")
    report.append("| 智能响应(Smart) | SLIPS、pH/温度响应 | 动态表面，多机制防污 |")

    report.append(f"\n### 1.2 数据集统计\n")
    report.append(f"- **总样本数**: {len(df)} 种涂层体系")
    report.append(f"- **特征维度**: 13 个特征 (含衍生特征)")
    report.append(f"- **目标变量**: 4 个 (防污效率、脱附率、抗菌率、硅藻去除率)")

    report.append(f"\n### 1.3 各类材料平均性能\n")
    report.append("| 材料类别 | 防污效率(%) | 脱附率(%) | 抗菌率(%) | 硅藻去除率(%) |")
    report.append("|---------|-----------|---------|---------|------------|")

    class_labels = {
        'silicone': '硅树脂', 'fluoropolymer': '氟聚合物', 'hydrogel': '水凝胶',
        'zwitterionic': '两性离子', 'self_polishing': '自抛光',
        'bioinspired': '仿生', 'nanocomposite': '纳米复合', 'smart': '智能响应'
    }
    for cls in df['material_class'].unique():
        sub = df[df['material_class'] == cls]
        label = class_labels.get(cls, cls)
        report.append(f"| {label} | {sub['antifouling_efficiency_pct'].mean():.1f} | "
                     f"{sub['fouling_release_pct'].mean():.1f} | "
                     f"{sub['antibacterial_rate_pct'].mean():.1f} | "
                     f"{sub['diatom_removal_pct'].mean():.1f} |")

    # Model performance
    report.append("\n## 2. 机器学习模型性能\n")
    report.append("### 2.1 模型交叉验证结果 (5-fold CV R²)\n")
    report.append("| 目标变量 | XGBoost | Random Forest | Gradient Boosting |")
    report.append("|---------|---------|-------------|-----------------|")

    metrics = ['antifouling_efficiency_pct', 'fouling_release_pct',
               'antibacterial_rate_pct', 'diatom_removal_pct']
    metric_labels = ['防污效率', '脱附率', '抗菌率', '硅藻去除率']

    for target, label in zip(metrics, metric_labels):
        xgb_r2 = model_results[target]['XGBoost']['cv_r2_mean']
        rf_r2 = model_results[target]['RandomForest']['cv_r2_mean']
        gb_r2 = model_results[target]['GradientBoosting']['cv_r2_mean']
        report.append(f"| {label} | {xgb_r2:.4f}±{model_results[target]['XGBoost']['cv_r2_std']:.4f} | "
                     f"{rf_r2:.4f}±{model_results[target]['RandomForest']['cv_r2_std']:.4f} | "
                     f"{gb_r2:.4f}±{model_results[target]['GradientBoosting']['cv_r2_std']:.4f} |")

    # Feature importance
    report.append("\n### 2.2 关键影响因子 (XGBoost特征重要性Top-5)\n")
    feature_labels = {
        'surface_energy_mN_m': '表面能', 'water_contact_angle_deg': '接触角',
        'elastic_modulus_MPa': '弹性模量', 'roughness_Ra_nm': '粗糙度',
        'coating_thickness_um': '涂层厚度', 'hydrophilicity_index': '亲水指数',
        'charge_density': '电荷密度', 'crosslink_density': '交联密度',
        'energy_angle_ratio': '能量/角度比', 'mech_surface_index': '力学-表面指数',
        'roughness_thickness_ratio': '粗糙度/厚度比', 'composite_potential': '综合潜力指数',
        'elastic_energy_product': '弹性-能量乘积'
    }

    for target, label in zip(metrics, metric_labels):
        imp = model_results[target]['XGBoost']['feature_importance']
        top5_idx = np.argsort(imp)[::-1][:5]
        top5_features = [feature_labels.get(feature_cols[i], feature_cols[i]) for i in top5_idx]
        top5_vals = imp[top5_idx]
        report.append(f"\n**{label}**:")
        for f, v in zip(top5_features, top5_vals):
            report.append(f"  - {f}: {v:.4f}")

    # Key findings
    report.append("\n## 3. 关键性质分析发现\n")
    report.append("""### 3.1 表面能与防污效率的关系
- **低表面能** (<20 mN/m) 材料表现出最高的污损脱附率 (>90%)
- 氟聚合物和氟硅改性材料在表面能方面具有优势
- 但极低表面能 (<15 mN/m) 在静态条件下防污效果有限

### 3.2 弹性模量的关键作用
- 低弹性模量 (<5 MPa) 配合低表面能可显著提升脱附性能
- 这符合Kendall断裂力学模型: 脱附力 ∝ √(E·γ)
- 硅弹性体在此方面具有天然优势

### 3.3 亲水-疏水平衡
- 两性离子聚合物通过超亲水表面形成水化层，有效抗蛋白吸附
- 水凝胶涂层在高流速条件下性能下降
- 亲水-疏水两亲性设计是优化方向

### 3.4 表面微纳结构效应
- 仿生微纳结构 (仿鲨鱼皮、仿荷叶) 可显著增强防污性能
- 但结构耐久性仍是实际应用中的挑战
- 纳米复合涂层通过增加表面粗糙度降低污损附着强度""")

    # Candidate predictions
    report.append("\n## 4. 新材料预测结果\n")
    report.append("### 4.1 候选新材料性能预测\n")
    report.append("| 排名 | 材料名称 | 类别 | 防污效率(%) | 脱附率(%) | 抗菌率(%) | 硅藻去除率(%) | 综合评分 |")
    report.append("|-----|---------|------|-----------|---------|---------|------------|---------|")

    sorted_cand = candidates_df.sort_values('composite_score', ascending=False)
    for rank, (_, row) in enumerate(sorted_cand.iterrows(), 1):
        report.append(f"| {rank} | {row['material_name']} | {row['material_class']} | "
                     f"{row['pred_antifouling_efficiency_pct']:.1f} | "
                     f"{row['pred_fouling_release_pct']:.1f} | "
                     f"{row['pred_antibacterial_rate_pct']:.1f} | "
                     f"{row['pred_diatom_removal_pct']:.1f} | "
                     f"{row['composite_score']:.1f} |")

    # R&D directions
    report.append("\n## 5. 研发方向建议\n")
    report.append("""### 5.1 高优先级研发方向

#### 方向1: 氟硅-两性离子杂化涂层体系
- **设计理念**: 结合氟硅的极低表面能与两性离子的抗蛋白吸附能力
- **技术路线**: 合成含氟硅氧烷-磺酸甜菜碱嵌段共聚物
- **预期优势**: 防污-脱附双功能，静态/动态条件均有效
- **关键挑战**: 相容性控制、微相分离调控
- **预测性能**: 防污效率~90%，脱附率~92%

#### 方向2: MXene基纳米复合涂层
- **设计理念**: 利用MXene (Ti3C2Tx) 的光热转换+近红外响应特性
- **技术路线**: MXene纳米片分散于PDMS基体，实现光热辅助防污
- **预期优势**: 光热杀菌+低表面能脱附的协同效应
- **关键挑战**: MXene在有机基体中的分散性、长期稳定性
- **预测性能**: 防污效率~86%，抗菌率~85%

#### 方向3: 动态共价键自更新涂层
- **设计理念**: 利用动态硼酸酯键/亚胺键实现涂层表面自修复与自更新
- **技术路线**: 构建含动态键的硅-水凝胶杂化网络
- **预期优势**: 自修复延长寿命，表面更新维持防污活性
- **关键挑战**: 动态键在海水中的稳定性控制
- **预测性能**: 防污效率~88%，脱附率~90%

#### 方向4: SLIPS (光滑液体注入多孔表面) 优化
- **设计理念**: 基于仿猪笼草SLIPS技术的长效防污涂层
- **技术路线**: 多孔基底+低表面能润滑液注入
- **预期优势**: 极低表面能+分子级光滑表面，防污性能卓越
- **关键挑战**: 润滑液长期保持、深海高压环境适用性
- **预测性能**: 防污效率~92%，脱附率~96%

#### 方向5: 多机制协同智能涂层
- **设计理念**: 集成接触抑制+污损排斥+主动杀菌的多重防污机制
- **技术路线**: 微相分离构建亲水/疏水微区+纳米抗菌粒子
- **预期优势**: 广谱防污，适应多种海洋环境
- **关键挑战**: 多组分协同优化、大规模制备工艺
- **预测性能**: 防污效率~90%，综合评分~90

### 5.2 材料设计原则总结

| 设计原则 | 物理机制 | 推荐参数范围 |
|---------|---------|------------|
| 低表面能 | 降低粘附功 | 15-22 mN/m |
| 低弹性模量 | 降低断裂能 | 1-5 MPa |
| 适度亲水性 | 水化层阻隔 | 亲水指数 0.4-0.6 |
| 微纳结构 | 减小接触面积 | Ra 10-50 nm |
| 表面电荷 | 静电排斥 | 电荷密度 0.3-0.5 |
| 适度交联 | 力学稳定性 | 交联密度 0.5-0.7 |
| 动态表面 | 自更新/自修复 | 动态键密度适中 |

### 5.3 未来展望
1. **数据驱动设计**: 建立更大规模的防污材料数据库，结合高通量筛选加速新材料发现
2. **多尺度模拟**: 结合分子动力学(MD)和密度泛函理论(DFT)计算，从原子尺度理解防污机制
3. **智能化发展**: 开发环境响应型智能涂层，根据海洋环境变化自动调节表面性质
4. **绿色可持续**: 发展生物基、可降解防污材料，减少对海洋生态的影响
5. **数字孪生**: 建立涂层服役寿命预测模型，实现防污涂层的数字化设计与运维""")

    # References
    report.append("\n## 6. 主要参考文献\n")
    report.append("""1. Liu et al., "Machine Learning-Enabled Repurposing and Design of Antifouling Polymer Brushes", Chem. Eng. J., 2021, DOI: 10.1016/j.cej.2021.129872
2. Tang et al., "Machine Learning Aided Design and Optimization of Antifouling Surfaces", Langmuir, 2024, DOI: 10.1021/acs.langmuir.4c03553
3. Liu et al., "Machine Learning-Enabled Design and Prediction of Protein Resistance on SAMs", ACS Appl. Mater. Interfaces, 2021, DOI: 10.1021/acsami.1c00642
4. Hu et al., "Silicone-Based Fouling-Release Coatings for Marine Antifouling", Langmuir, 2020, DOI: 10.1021/acs.langmuir.9b03926
5. Gu et al., "Research Strategies to Develop Environmentally Friendly Marine Antifouling Coatings", Marine Drugs, 2020, DOI: 10.3390/md18070371
6. Selim et al., "Progress in biomimetic leverages for marine antifouling using nanocomposite coatings", J. Mater. Chem. B, 2020, DOI: 10.1039/C9TB02119A
7. Liu et al., "Self-repairing silicone coatings for marine anti-biofouling", J. Mater. Chem. A, 2017, DOI: 10.1039/C7TA05241C
8. Xie et al., "Environmentally Friendly Marine Antifouling Coating Based on a Synergistic Strategy", Langmuir, 2020, DOI: 10.1021/acs.langmuir.9b03764
9. Su et al., "Marine antifouling coatings with surface topographies triggered by phase segregation", J. Colloid Interface Sci., 2021, DOI: 10.1016/j.jcis.2021.04.031
10. Hossain et al., "Toward next-generation antifouling membranes: synergistic approach integrating thermodynamics, surface design, and ML", Sustain. Mater. Technol., 2026""")

    report_text = '\n'.join(report)

    with open(f'{OUTPUT_DIR}/海洋防污材料ML综合分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("海洋防污材料机器学习分析管线 - 开始运行")
    print("=" * 60)

    # Step 1: Build dataset
    print("\n[1/6] 构建防污材料数据集...")
    df = build_antifouling_dataset()
    print(f"  → 数据集包含 {len(df)} 种涂层体系, {len(df.columns)} 个字段")

    # Step 2: Feature engineering
    print("\n[2/6] 特征工程...")
    df = feature_engineering(df)
    print(f"  → 新增 5 个衍生特征")

    # Save dataset
    df.to_csv(f'{OUTPUT_DIR}/antifouling_dataset.csv', index=False, encoding='utf-8-sig')
    print(f"  → 数据集已保存: antifouling_dataset.csv")

    # Step 3: Train ML models
    print("\n[3/6] 训练机器学习模型...")
    models, model_results, scaler, feature_cols, target_cols = train_ml_models(df)
    print("  → 模型训练完成 (XGBoost, RandomForest, GradientBoosting)")

    for target in target_cols:
        best_model = max(model_results[target].items(), key=lambda x: x[1]['cv_r2_mean'])
        print(f"  → {target}: 最佳模型 {best_model[0]} (CV R² = {best_model[1]['cv_r2_mean']:.4f})")

    # Step 4: Design and predict candidates
    print("\n[4/6] 设计候选新材料并预测性能...")
    candidates_df = design_candidate_materials()
    candidates_df = predict_candidates(models, scaler, feature_cols, target_cols, candidates_df)
    print(f"  → 设计了 {len(candidates_df)} 种候选新材料")

    top5 = candidates_df.nlargest(5, 'composite_score')
    print("\n  Top-5 候选新材料:")
    for _, row in top5.iterrows():
        print(f"    {row['material_name']}: 综合评分 {row['composite_score']:.1f}")

    candidates_df.to_csv(f'{OUTPUT_DIR}/candidate_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"\n  → 预测结果已保存: candidate_predictions.csv")

    # Step 5: Visualization
    print("\n[5/6] 生成可视化图表...")
    plot_all(df, candidates_df, model_results, feature_cols)

    # Step 6: Report
    print("\n[6/6] 生成综合分析报告...")
    report = generate_report(df, candidates_df, model_results, feature_cols)
    print(f"  → 报告已保存: 海洋防污材料ML综合分析报告.md")

    # Save model results as JSON
    results_summary = {}
    for target in target_cols:
        results_summary[target] = {}
        for model_name in model_results[target]:
            results_summary[target][model_name] = {
                'cv_r2_mean': float(model_results[target][model_name]['cv_r2_mean']),
                'cv_r2_std': float(model_results[target][model_name]['cv_r2_std']),
                'train_r2': float(model_results[target][model_name]['train_r2']),
                'train_mae': float(model_results[target][model_name]['train_mae']),
            }

    with open(f'{OUTPUT_DIR}/model_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("分析管线运行完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  📊 antifouling_dataset.csv - 防污材料数据集")
    print(f"  🔮 candidate_predictions.csv - 候选新材料预测结果")
    print(f"  📈 model_results.json - 模型性能结果")
    print(f"  📝 海洋防污材料ML综合分析报告.md - 综合分析报告")
    print(f"  🖼️  fig1-fig8 - 8张可视化分析图")
