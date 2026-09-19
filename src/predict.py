#!/usr/bin/env python3
"""
============================================================================
海洋防污材料分子结构→可行性预测接口
用法: python predict.py "SMILES_STRING"
      python predict.py --interactive  (交互模式)
      python predict.py --batch input.csv  (批量预测)
============================================================================
"""

import sys
import os
import pickle
import numpy as np
import json
from collections import defaultdict

# 确保路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen

OUTPUT_DIR = SCRIPT_DIR


class MolecularDescriptorEngine:
    """分子描述符计算引擎 (与训练时一致)"""

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

        charged_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
        desc['ChargeDensity'] = charged_atoms / max(desc['HeavyAtoms'], 1)

        hydrophilic_mass = desc['NumO'] * 16 + desc['NumN'] * 14 + desc['TPSA'] * 0.1
        desc['HydrophilicLipophilicBalance'] = min(20, 20 * hydrophilic_mass / max(desc['MW'], 1))

        se = 40.0
        se -= desc['NumF'] * 2.5
        se -= desc['NumSi'] * 3.0
        se += desc['NumO'] * 0.5
        se += desc['NumN'] * 0.8
        se += desc['ChargeDensity'] * 15
        se -= desc['LogP'] * 1.5
        desc['SurfaceEnergyEstimate'] = max(10, min(50, se))

        em = 2.0
        em += desc['AromaticRings'] * 0.3
        em += desc['RingCount'] * 0.1
        em -= desc['NumSi'] * 0.4
        em -= desc['RotBonds'] * 0.02
        em += desc['ChargeDensity'] * 0.5
        desc['ElasticModulusEstimate'] = max(-1, min(4, em))

        desc['RoughnessPotential'] = (desc['RingCount'] * 0.1 +
                                       desc['HeavyAtoms'] * 0.005 +
                                       (1 if desc['HasTi'] or desc['HasZn'] else 0) * 0.3)

        reactive_groups = sum(1 for atom in mol.GetAtoms()
                            if atom.GetSymbol() in ['N', 'O', 'S'] and atom.GetDegree() <= 2)
        desc['CrosslinkPotential'] = min(1.0, reactive_groups / max(desc['HeavyAtoms'], 1))

        return desc


class AntifoulingPredictor:
    """
    海洋防污材料可行性预测器
    加载预训练模型，输入SMILES即可获得预测结果
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(OUTPUT_DIR, 'models.pkl')

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        self.target_cols = data['target_cols']
        self.cv_results = data['cv_results']

        self.target_labels = {
            'antifouling_efficiency_pct': '防污效率',
            'fouling_release_pct': '污损脱附率',
            'antibacterial_rate_pct': '抗菌率',
            'diatom_removal_pct': '硅藻去除率'
        }

    def predict(self, smiles, verbose=True):
        """
        预测分子结构的防污可行性

        参数:
            smiles: SMILES分子结构字符串
            verbose: 是否打印详细结果

        返回:
            dict: 包含各模型预测结果和综合评分
        """
        desc = MolecularDescriptorEngine.compute_descriptors(smiles)
        if desc is None:
            return {'error': f'SMILES解析失败: {smiles}'}

        # 构建特征向量
        features = np.array([[desc.get(col, 0) for col in self.feature_cols]])
        features_scaled = self.scaler.transform(features)

        results = {
            'SMILES': smiles,
            'descriptors': desc,
            'predictions': {}
        }

        for target in self.target_cols:
            predictions = {}
            for m_name in ['DecisionTree', 'Ridge', 'Lasso', 'KNN', 'XGBoost']:
                pred = self.models[target][m_name].predict(features_scaled)[0]
                predictions[m_name] = float(np.clip(pred, 0, 100))

            # Ensemble
            weights = self.models[target]['Ensemble_weights']
            ens_pred = sum(w * predictions[m] for m, w in weights.items())
            predictions['Ensemble'] = float(np.clip(ens_pred, 0, 100))

            pred_values = list(predictions.values())[:-1]  # 不含ensemble
            predictions['model_std'] = float(np.std(pred_values))
            predictions['consensus'] = ('高一致性' if np.std(pred_values) < 5
                                       else '中等一致性' if np.std(pred_values) < 10
                                       else '低一致性(需实验验证)')

            results['predictions'][target] = predictions

        # 综合可行性评分
        ensemble_scores = [results['predictions'][t]['Ensemble'] for t in self.target_cols]
        results['feasibility_score'] = float(np.average(ensemble_scores, weights=[0.35, 0.25, 0.20, 0.20]))
        results['feasibility_level'] = (
            '🟢 优秀 (强烈推荐开发)' if results['feasibility_score'] >= 85 else
            '🔵 良好 (值得尝试)' if results['feasibility_score'] >= 75 else
            '🟡 一般 (需要优化设计)' if results['feasibility_score'] >= 65 else
            '🔴 较差 (不推荐此方案)'
        )

        # 关键描述符分析
        results['key_insights'] = self._analyze_key_insights(desc)

        if verbose:
            self._print_results(results)

        return results

    def _analyze_key_insights(self, desc):
        """分析关键描述符，给出物理解释"""
        insights = []

        se = desc.get('SurfaceEnergyEstimate', 25)
        if se < 18:
            insights.append(f"✅ 表面能极低 ({se:.1f} mN/m) → 有利于污损脱附")
        elif se < 25:
            insights.append(f"🔵 表面能较低 ({se:.1f} mN/m) → 中等脱附能力")
        elif se > 35:
            insights.append(f"⚠️ 表面能较高 ({se:.1f} mN/m) → 脱附能力有限")

        logp = desc.get('LogP', 2)
        if logp > 3:
            insights.append(f"✅ 高疏水性 (LogP={logp:.1f}) → 有利于降低表面能")
        elif logp < 0:
            insights.append(f"🔵 亲水性强 (LogP={logp:.1f}) → 可能形成水化层防污")

        if desc.get('NumF', 0) > 0:
            insights.append(f"✅ 含{desc['NumF']}个氟原子 → 显著降低表面能")

        if desc.get('NumSi', 0) > 0:
            insights.append(f"✅ 含{desc['NumSi']}个硅原子 → 降低表面能和模量")

        if desc.get('ChargeDensity', 0) > 0.1:
            insights.append(f"✅ 较高电荷密度 ({desc['ChargeDensity']:.2f}) → 有利于两性离子防污机制")

        if desc.get('HasCu', 0) or desc.get('HasZn', 0) or desc.get('HasAg', 0):
            metals = []
            if desc.get('HasCu', 0): metals.append('Cu')
            if desc.get('HasZn', 0): metals.append('Zn')
            if desc.get('HasAg', 0): metals.append('Ag')
            insights.append(f"✅ 含金属纳米粒子 ({', '.join(metals)}) → 增强抗菌性能")

        em = desc.get('ElasticModulusEstimate', 2)
        if em < 1:
            insights.append(f"✅ 低弹性模量 (10^{em:.1f} MPa) → 有利于断裂力学脱附")

        return insights

    def _print_results(self, results):
        """打印预测结果"""
        print("\n" + "=" * 60)
        print(f"  SMILES: {results['SMILES'][:60]}{'...' if len(results['SMILES']) > 60 else ''}")
        print(f"  综合可行性评分: {results['feasibility_score']:.1f}/100")
        print(f"  可行性等级: {results['feasibility_level']}")
        print("=" * 60)

        print("\n📊 各模型预测结果:")
        print(f"{'目标':12s} | {'DecisionTree':>12s} | {'Ridge':>8s} | {'Lasso':>8s} | {'KNN':>8s} | {'XGBoost':>8s} | {'Ensemble':>8s} | 一致性")
        print("-" * 100)

        for target in self.target_cols:
            label = self.target_labels.get(target, target)
            preds = results['predictions'][target]
            print(f"{label:12s} | {preds['DecisionTree']:12.1f} | {preds['Ridge']:8.1f} | "
                  f"{preds['Lasso']:8.1f} | {preds['KNN']:8.1f} | {preds['XGBoost']:8.1f} | "
                  f"{preds['Ensemble']:8.1f} | {preds['consensus']}")

        print("\n🔍 关键物理解读:")
        for insight in results.get('key_insights', []):
            print(f"  {insight}")
        print()


def interactive_mode():
    """交互式预测模式"""
    print("=" * 60)
    print("  海洋防污材料分子结构→可行性预测平台 v2.0")
    print("  输入SMILES分子结构，获取防污性能预测")
    print("  输入 'quit' 或 'q' 退出")
    print("=" * 60)

    predictor = AntifoulingPredictor()

    while True:
        print("\n请输入SMILES分子结构:")
        smiles = input("> ").strip()

        if smiles.lower() in ['quit', 'q', 'exit']:
            print("再见！")
            break

        if not smiles:
            continue

        result = predictor.predict(smiles)
        if 'error' in result:
            print(f"❌ {result['error']}")


def batch_predict(input_file, output_file=None):
    """批量预测"""
    import pandas as pd

    predictor = AntifoulingPredictor()
    df = pd.read_csv(input_file)

    if 'SMILES' not in df.columns:
        print("错误: 输入CSV必须包含 'SMILES' 列")
        return

    results = []
    for idx, row in df.iterrows():
        smi = row['SMILES']
        result = predictor.predict(smi, verbose=False)
        if 'error' not in result:
            rec = {'SMILES': smi}
            if 'material_name' in row:
                rec['material_name'] = row['material_name']
            for target in predictor.target_cols:
                rec[f'{target}_ensemble'] = result['predictions'][target]['Ensemble']
                rec[f'{target}_std'] = result['predictions'][target]['model_std']
            rec['feasibility_score'] = result['feasibility_score']
            rec['feasibility_level'] = result['feasibility_level']
            results.append(rec)
        print(f"  [{idx+1}/{len(df)}] {smi[:40]}... → {result.get('feasibility_score', 'N/A')}")

    result_df = pd.DataFrame(results)
    if output_file is None:
        output_file = input_file.replace('.csv', '_predictions.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n批量预测完成! 结果保存至: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_mode()
        elif sys.argv[1] == '--batch':
            batch_predict(sys.argv[2])
        else:
            # 直接预测SMILES
            predictor = AntifoulingPredictor()
            smiles = sys.argv[1]
            result = predictor.predict(smiles)
    else:
        print("用法:")
        print('  python predict.py "SMILES_STRING"     # 预测单个分子')
        print("  python predict.py --interactive       # 交互模式")
        print("  python predict.py --batch input.csv   # 批量预测")
        print()
        print("示例:")
        print('  python predict.py "C[Si](C)(C)O[Si](C)(C)C"')
        print('  python predict.py "FC(F)(F)C(F)(F)C(F)(F)F"')
