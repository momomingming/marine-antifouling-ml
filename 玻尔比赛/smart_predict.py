#!/usr/bin/env python3
"""
============================================================================
海洋防污材料智能预测引擎 v6.0
Smart Material Parser + Multi-Component Prediction
============================================================================

核心升级:
  1. 智能材料解析器: 自动识别材料类型(纯高分子/共聚物/纳米复合/混合物/天然材料)
  2. 多组分特征计算: 加权描述符 + 交互特征 + 纳米粒子特征
  3. 向后兼容: 原有SMILES输入仍然有效

输入语法:
  纯小分子:    C[Si](C)(C)O[Si](C)(C)C
  均聚物:      POLY[C=CC(=O)O]
  共聚物:      C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3
  纳米复合:    C[Si](C)(C)O[Si](C)(C)C @ [Zn]=O:5
  多层涂层:    PDMS / PSBMA
  天然材料:    壳聚糖
  混合材料:    PVA:0.6 + 壳聚糖:0.4

作者: MatMaster AI Platform
日期: 2026-08-24
============================================================================
"""

import re
import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 纳米粒子数据库
# ============================================================

NANOPARTICLE_DB = {
    'ZnO':  {'name': '氧化锌', 'bandgap_eV': 3.37, 'surface_area_m2g': 50,
             'antibacterial_index': 0.85, 'size_default_nm': 30, 'density_gcm3': 5.61,
             'zeta_mV': -25, 'crystal': 'wurtzite'},
    'Cu2O': {'name': '氧化亚铜', 'bandgap_eV': 2.17, 'surface_area_m2g': 30,
             'antibacterial_index': 0.92, 'size_default_nm': 50, 'density_gcm3': 6.0,
             'zeta_mV': -15, 'crystal': 'cubic'},
    'CuO':  {'name': '氧化铜', 'bandgap_eV': 1.2, 'surface_area_m2g': 40,
             'antibacterial_index': 0.88, 'size_default_nm': 40, 'density_gcm3': 6.31,
             'zeta_mV': -20, 'crystal': 'monoclinic'},
    'Ag':   {'name': '银纳米粒子', 'bandgap_eV': 0.0, 'surface_area_m2g': 25,
             'antibacterial_index': 0.95, 'size_default_nm': 20, 'density_gcm3': 10.49,
             'zeta_mV': -30, 'crystal': 'fcc'},
    'TiO2': {'name': '二氧化钛', 'bandgap_eV': 3.20, 'surface_area_m2g': 80,
             'antibacterial_index': 0.80, 'size_default_nm': 25, 'density_gcm3': 4.23,
             'zeta_mV': -35, 'crystal': 'anatase'},
    'SiO2': {'name': '二氧化硅', 'bandgap_eV': 9.00, 'surface_area_m2g': 200,
             'antibacterial_index': 0.30, 'size_default_nm': 20, 'density_gcm3': 2.65,
             'zeta_mV': -40, 'crystal': 'amorphous'},
    'CeO2': {'name': '二氧化铈', 'bandgap_eV': 3.15, 'surface_area_m2g': 60,
             'antibacterial_index': 0.75, 'size_default_nm': 15, 'density_gcm3': 7.22,
             'zeta_mV': -20, 'crystal': 'fluorite'},
    'GO':   {'name': '氧化石墨烯', 'bandgap_eV': 0.5, 'surface_area_m2g': 500,
             'antibacterial_index': 0.70, 'size_default_nm': 500, 'density_gcm3': 1.8,
             'zeta_mV': -45, 'crystal': '2D'},
    'CNT':  {'name': '碳纳米管', 'bandgap_eV': 0.0, 'surface_area_m2g': 300,
             'antibacterial_index': 0.50, 'size_default_nm': 1000, 'density_gcm3': 2.1,
             'zeta_mV': -25, 'crystal': '1D'},
    'Cu':   {'name': '铜纳米粒子', 'bandgap_eV': 0.0, 'surface_area_m2g': 20,
             'antibacterial_index': 0.90, 'size_default_nm': 30, 'density_gcm3': 8.96,
             'zeta_mV': -10, 'crystal': 'fcc'},
}

# SMILES近似映射 (用于纳米粒子的SMILES表示)
NP_SMILES_MAP = {
    'ZnO': '[Zn]=O', 'Cu2O': '[Cu]O[Cu]', 'CuO': 'O=[Cu]',
    'Ag': '[Ag]', 'TiO2': 'O=[Ti]=O', 'SiO2': '[Si](=O)(=O)',
    'CeO2': '[Ce]=O', 'GO': 'C1=CC=C2C(=C1)C(=O)C(=C2O)O',
    'CNT': 'C1=CC=CC=C1', 'Cu': '[Cu]',
}

# ============================================================
# 天然/复杂材料查找表
# ============================================================

NATURAL_MATERIAL_DB = {
    # 中文名 → (英文名, SMILES/描述符, material_class)
    '壳聚糖': ('chitosan', 'OC[C@H]1OC(O)[C@H](N)[C@@H](O)[C@@H]1O', 'hydrogel',
               {'MW': 161.16, 'LogP': -2.5, 'TPSA': 116, 'HBD': 5, 'HBA': 6}),
    'chitosan': ('chitosan', 'OC[C@H]1OC(O)[C@H](N)[C@@H](O)[C@@H]1O', 'hydrogel',
                 {'MW': 161.16, 'LogP': -2.5, 'TPSA': 116, 'HBD': 5, 'HBA': 6}),
    'CS': ('chitosan', 'OC[C@H]1OC(O)[C@H](N)[C@@H](O)[C@@H]1O', 'hydrogel',
           {'MW': 161.16, 'LogP': -2.5, 'TPSA': 116, 'HBD': 5, 'HBA': 6}),
    '海藻酸钠': ('sodium alginate', 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 'hydrogel',
                {'MW': 198.14, 'LogP': -3.8, 'TPSA': 130, 'HBD': 5, 'HBA': 8}),
    'alginate': ('sodium alginate', 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 'hydrogel',
                 {'MW': 198.14, 'LogP': -3.8, 'TPSA': 130, 'HBD': 5, 'HBA': 8}),
    '多巴胺': ('dopamine', 'NCc1ccc(O)c(O)c1', 'bioinspired',
              {'MW': 153.18, 'LogP': -0.9, 'TPSA': 73, 'HBD': 4, 'HBA': 3}),
    'dopamine': ('dopamine', 'NCc1ccc(O)c(O)c1', 'bioinspired',
                 {'MW': 153.18, 'LogP': -0.9, 'TPSA': 73, 'HBD': 4, 'HBA': 3}),
    'PDA': ('polydopamine', 'NCc1ccc(O)c(O)c1', 'bioinspired',
            {'MW': 153.18, 'LogP': -0.9, 'TPSA': 73, 'HBD': 4, 'HBA': 3}),
    '漆酚': ('urushiol', 'CCCCCCCCCCCCCCCCc1ccc(O)c(O)c1', 'bioinspired',
            {'MW': 316.5, 'LogP': 7.5, 'TPSA': 40, 'HBD': 2, 'HBA': 2}),
    'urushiol': ('urushiol', 'CCCCCCCCCCCCCCCCc1ccc(O)c(O)c1', 'bioinspired',
                 {'MW': 316.5, 'LogP': 7.5, 'TPSA': 40, 'HBD': 2, 'HBA': 2}),
    '丹宁酸': ('tannic acid', 'OC(=O)c1cc(O)c(O)c(O)c1', 'bioinspired',
              {'MW': 170.12, 'LogP': 1.2, 'TPSA': 130, 'HBD': 5, 'HBA': 8}),
    'tannic acid': ('tannic acid', 'OC(=O)c1cc(O)c(O)c(O)c1', 'bioinspired',
                    {'MW': 170.12, 'LogP': 1.2, 'TPSA': 130, 'HBD': 5, 'HBA': 8}),
    'TA': ('tannic acid', 'OC(=O)c1cc(O)c(O)c(O)c1', 'bioinspired',
           {'MW': 170.12, 'LogP': 1.2, 'TPSA': 130, 'HBD': 5, 'HBA': 8}),
    '辣椒素': ('capsaicin', 'COc1ccc(CCN=Cc2ccc(O)c(OC)c2)cc1O', 'bioinspired',
              {'MW': 305.4, 'LogP': 3.6, 'TPSA': 59, 'HBD': 2, 'HBA': 4}),
    '松香': ('rosin', 'OC(=O)C1CC2CCC(C(C)C)C2CC1', 'self_polishing',
            {'MW': 302.5, 'LogP': 6.5, 'TPSA': 37, 'HBD': 1, 'HBA': 2}),
    # 常见聚合物缩写
    'PDMS': ('PDMS', 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'silicone', None),
    'PTFE': ('PTFE', 'FC(F)=C(F)F', 'fluoropolymer', None),
    'PVA': ('PVA', 'OC(C)C', 'hydrogel', None),
    'PEG': ('PEG', 'OCCOCCOCCO', 'hydrogel', None),
    'PHEMA': ('PHEMA', 'CC(C)(C(=O)OCCO)C(=O)O', 'hydrogel', None),
    'PNIPAM': ('PNIPAM', 'CC(C)NC(=O)C=C', 'smart', None),
    'PVDF': ('PVDF', 'FC(F)CC(F)(F)', 'fluoropolymer', None),
    'PMMA': ('PMMA', 'CC(C)(C(=O)OC)C(=O)OC', 'fluoropolymer', None),
    'PSBMA': ('PSBMA', 'C[N+](C)(C)CCCS([O-])(=O)=O', 'zwitterionic', None),
    'PCBMA': ('PCBMA', 'C[N+](C)(C)CC(=O)[O-]', 'zwitterionic', None),
    'MPC': ('MPC', 'C[N+](C)(C)CCOP([O-])(=O)O', 'zwitterionic', None),
    'SBMA': ('SBMA', 'C[N+](C)(C)CCCS([O-])(=O)=O', 'zwitterionic', None),
    'CBMA': ('CBMA', 'C[N+](C)(C)CC(=O)[O-]', 'zwitterionic', None),
    'PVP': ('PVP', 'O=C1CCCN1', 'bioinspired', None),
    'NIPAM': ('NIPAM', 'CC(C)NC(=O)C=C', 'smart', None),
    'PEGMA': ('PEGMA', 'C=CC(=O)OCCOCCOCCO', 'hydrogel', None),
    'MMA': ('MMA', 'C=CC(=O)OC(C)C', 'fluoropolymer', None),
    'DFMA': ('DFMA', 'C=CC(=O)OCCC(F)(F)C(F)(F)F', 'fluoropolymer', None),
    'GMA': ('GMA', 'C=CC(=O)OCC1CO1', 'fluoropolymer', None),
    'BA': ('BA', 'C=CC(=O)OCCCC', 'fluoropolymer', None),
    '特氟龙': ('PTFE', 'FC(F)=C(F)F', 'fluoropolymer', None),
    '聚四氟乙烯': ('PTFE', 'FC(F)=C(F)F', 'fluoropolymer', None),
    '聚二甲基硅氧烷': ('PDMS', 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'silicone', None),
    '聚乙烯醇': ('PVA', 'OC(C)C', 'hydrogel', None),
    '聚乙二醇': ('PEG', 'OCCOCCOCCO', 'hydrogel', None),
}


# ============================================================
# 分子描述符计算引擎
# ============================================================

DESCRIPTOR_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'RingCount',
    'AromaticRings', 'HeavyAtoms', 'FractionCSP3',
    'NumF', 'NumCl', 'NumBr', 'NumN', 'NumO', 'NumS', 'NumSi', 'NumP',
    'HasCu', 'HasZn', 'HasAg', 'HasTi',
    'ChargeDensity', 'HydrophilicLipophilicBalance',
    'SurfaceEnergyEstimate', 'ElasticModulusEstimate',
    'RoughnessPotential', 'CrosslinkPotential',
]


def compute_descriptors(smiles):
    """从SMILES计算分子描述符"""
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


# ============================================================
# 智能材料解析器
# ============================================================

class SmartMaterialParser:
    """
    智能材料解析器
    自动识别输入材料的类型并选择合适的特征计算策略
    """

    # 材料类型枚举
    TYPE_SIMPLE = 'simple_molecule'       # 纯小分子
    TYPE_HOMOPOLYMER = 'homopolymer'      # 均聚物
    TYPE_COPOLYMER = 'copolymer'          # 共聚物
    TYPE_NANOCOMPOSITE = 'nanocomposite'  # 纳米复合材料
    TYPE_MULTILAYER = 'multilayer'        # 多层涂层
    TYPE_NATURAL = 'natural_material'     # 天然/复杂材料
    TYPE_BLEND = 'blend'                  # 混合材料

    @classmethod
    def parse(cls, input_str):
        """
        解析输入字符串，返回材料类型和组分信息

        返回:
            dict: {
                'type': 材料类型,
                'components': 组分列表,
                'raw_input': 原始输入,
                'description': 人类可读描述
            }
        """
        input_str = input_str.strip()

        # 1. 检查是否为天然材料/常见缩写
        if input_str in NATURAL_MATERIAL_DB:
            entry = NATURAL_MATERIAL_DB[input_str]
            return {
                'type': cls.TYPE_NATURAL,
                'components': [{'name': entry[0], 'smiles': entry[1],
                               'class': entry[2], 'override_desc': entry[3],
                               'ratio': 1.0}],
                'raw_input': input_str,
                'description': f'天然/已知材料: {entry[0]}'
            }

        # 2. 检查多层涂层 (用 / 分隔)
        if '/' in input_str and '+' not in input_str and '@' not in input_str:
            layers = [l.strip() for l in input_str.split('/')]
            if len(layers) >= 2:
                parsed_layers = []
                for i, layer in enumerate(layers):
                    sub = cls.parse(layer)
                    parsed_layers.append({
                        'layer_index': i,
                        'parsed': sub
                    })
                return {
                    'type': cls.TYPE_MULTILAYER,
                    'components': parsed_layers,
                    'raw_input': input_str,
                    'description': f'{len(layers)}层涂层: ' + ' / '.join(
                        [l['parsed']['description'] for l in parsed_layers])
                }

        # 3. 检查纳米复合 (用 @ 分隔)
        if '@' in input_str:
            parts = input_str.split('@')
            if len(parts) == 2:
                polymer_part = parts[0].strip()
                np_part = parts[1].strip()

                # 解析纳米粒子
                np_match = re.match(r'(\w+)(?::(\d+\.?\d*))?', np_part)
                if np_match:
                    np_name = np_match.group(1)
                    np_wt = float(np_match.group(2)) if np_match.group(2) else 5.0

                    # 解析聚合物基体
                    polymer_parsed = cls.parse(polymer_part)

                    return {
                        'type': cls.TYPE_NANOCOMPOSITE,
                        'components': [{
                            'polymer': polymer_parsed,
                            'nanoparticle': np_name,
                            'np_wt_pct': np_wt,
                            'np_info': NANOPARTICLE_DB.get(np_name, None)
                        }],
                        'raw_input': input_str,
                        'description': f'纳米复合: {polymer_parsed["description"]} + {np_wt}% {np_name}'
                    }

        # 4. 检查共聚物/混合物 (用 " + " 分隔, 避免匹配SMILES中的[N+]等)
        # 只在 " + " (两侧有空格) 处分割
        copolymer_parts = re.split(r'\s+\+\s+', input_str)
        if len(copolymer_parts) >= 2:
            parts = [p.strip() for p in copolymer_parts if p.strip()]
            components = []
            for part in parts:
                # 解析 SMILES:ratio 格式
                ratio_match = re.match(r'(.+?):(\d+\.?\d*)', part)
                if ratio_match:
                    smi = ratio_match.group(1).strip()
                    ratio = float(ratio_match.group(2))
                else:
                    smi = part.strip()
                    ratio = 1.0 / len(parts)  # 默认等比

                # 尝试解析每个组分
                comp_parsed = cls.parse(smi)
                components.append({
                    'smiles': smi,
                    'ratio': ratio,
                    'parsed': comp_parsed
                })

            # 归一化比例
            total_ratio = sum(c['ratio'] for c in components)
            for c in components:
                c['ratio'] /= total_ratio

            return {
                'type': cls.TYPE_COPOLYMER,
                'components': components,
                'raw_input': input_str,
                'description': f'共聚物/混合物: ' + ' + '.join(
                    [f"{c['parsed']['description']}({c['ratio']:.0%})" for c in components])
            }

        # 5. 检查均聚物 POLY[...] 格式
        poly_match = re.match(r'POLY\[(.+?)\]', input_str)
        if poly_match:
            monomer_smi = poly_match.group(1)
            return {
                'type': cls.TYPE_HOMOPOLYMER,
                'components': [{'smiles': monomer_smi, 'ratio': 1.0}],
                'raw_input': input_str,
                'description': f'均聚物: POLY[{monomer_smi}]'
            }

        # 6. 尝试作为SMILES解析
        mol = Chem.MolFromSmiles(input_str)
        if mol is not None:
            return {
                'type': cls.TYPE_SIMPLE,
                'components': [{'smiles': input_str, 'ratio': 1.0}],
                'raw_input': input_str,
                'description': f'小分子: {input_str[:50]}'
            }

        # 7. 无法解析
        return {
            'type': 'unknown',
            'components': [],
            'raw_input': input_str,
            'description': f'无法识别: {input_str[:50]}'
        }

    @classmethod
    def compute_features(cls, parsed_material):
        """
        根据材料类型计算特征向量
        返回与模型兼容的描述符字典
        """
        mat_type = parsed_material['type']

        if mat_type == cls.TYPE_SIMPLE:
            return cls._features_simple(parsed_material)
        elif mat_type == cls.TYPE_HOMOPOLYMER:
            return cls._features_homopolymer(parsed_material)
        elif mat_type == cls.TYPE_COPOLYMER:
            return cls._features_copolymer(parsed_material)
        elif mat_type == cls.TYPE_NANOCOMPOSITE:
            return cls._features_nanocomposite(parsed_material)
        elif mat_type == cls.TYPE_MULTILAYER:
            return cls._features_multilayer(parsed_material)
        elif mat_type == cls.TYPE_NATURAL:
            return cls._features_natural(parsed_material)
        else:
            return None

    @classmethod
    def _features_simple(cls, parsed):
        """纯小分子: 直接计算SMILES描述符"""
        smi = parsed['components'][0]['smiles']
        return compute_descriptors(smi)

    @classmethod
    def _features_homopolymer(cls, parsed):
        """均聚物: 基于单体SMILES计算，调整分子量相关特征"""
        smi = parsed['components'][0]['smiles']
        desc = compute_descriptors(smi)
        if desc is None:
            return None
        # 聚合后MW增大，RotBonds减少(链刚性增加)
        desc['MW'] *= 50  # 近似聚合度
        desc['RotBonds'] = max(1, desc['RotBonds'] - 2)
        desc['CrosslinkPotential'] = min(1.0, desc['CrosslinkPotential'] * 1.5)
        return desc

    @classmethod
    def _features_copolymer(cls, parsed):
        """
        共聚物: 加权链式SMILES方法
        参考: Huang et al., ACS Appl. Polym. Mater., 2024
        """
        components = parsed['components']

        # 计算各组分的描述符
        comp_descs = []
        ratios = []
        for comp in components:
            sub_parsed = comp['parsed']
            desc = None

            if sub_parsed['type'] == cls.TYPE_NATURAL:
                # 天然材料使用查找表
                if sub_parsed['components']:
                    override = sub_parsed['components'][0].get('override_desc')
                    if override:
                        desc = dict(override)
                    else:
                        desc = compute_descriptors(sub_parsed['components'][0].get('smiles', ''))
            elif sub_parsed['type'] == 'unknown':
                # 尝试直接用comp中的smiles
                smi = comp.get('smiles', '')
                if smi:
                    desc = compute_descriptors(smi)
            else:
                # 其他类型: 优先用comp中的smiles，否则从parsed获取
                smi = comp.get('smiles', '')
                if not smi and sub_parsed.get('components'):
                    smi = sub_parsed['components'][0].get('smiles', '')
                if smi:
                    desc = compute_descriptors(smi)

            if desc is not None:
                comp_descs.append(desc)
                ratios.append(comp['ratio'])

        if not comp_descs:
            return None

        # 归一化比例
        total = sum(ratios)
        ratios = [r / total for r in ratios]

        # 加权平均
        result = {}
        for key in DESCRIPTOR_NAMES:
            vals = [d.get(key, 0) for d in comp_descs]
            result[key] = sum(v * r for v, r in zip(vals, ratios))

        # 添加交互特征 (捕捉共聚效应)
        if len(comp_descs) >= 2:
            d1, d2 = comp_descs[0], comp_descs[1]
            # LogP差异 → 微相分离倾向
            logp_diff = abs(d1.get('LogP', 0) - d2.get('LogP', 0))
            # 表面能差异 → 表面偏析
            se_diff = abs(d1.get('SurfaceEnergyEstimate', 25) - d2.get('SurfaceEnergyEstimate', 25))
            # 调整: 差异越大，越容易形成微相分离，有利于防污
            result['SurfaceEnergyEstimate'] -= se_diff * 0.1  # 微相分离降低有效表面能
            result['RoughnessPotential'] += logp_diff * 0.05  # 增加粗糙度潜力

        # 聚合后调整
        result['MW'] *= 30
        result['CrosslinkPotential'] = min(1.0, result['CrosslinkPotential'] * 1.3)

        return result

    @classmethod
    def _features_nanocomposite(cls, parsed):
        """
        纳米复合材料: 聚合物基体 + 纳米粒子特征
        参考: Lv et al., ACS Appl. Polym. Mater., 2025
        """
        comp = parsed['components'][0]
        polymer_parsed = comp['polymer']
        np_name = comp['nanoparticle']
        np_wt = comp['np_wt_pct']
        np_info = comp.get('np_info')

        # 聚合物基体描述符
        desc = cls.compute_features(polymer_parsed)
        if desc is None:
            return None

        # 纳米粒子效应
        if np_info:
            # 抗菌率增强
            antibac_boost = np_info['antibacterial_index'] * np_wt * 0.8
            # 表面能微调
            se_effect = -np_wt * 0.2  # 纳米粒子通常略微降低表面能
            # 粗糙度增加
            roughness_boost = np_wt * 0.02
            # 模量增加
            modulus_boost = np_wt * 0.01

            desc['SurfaceEnergyEstimate'] = max(10, desc['SurfaceEnergyEstimate'] + se_effect)
            desc['RoughnessPotential'] += roughness_boost
            desc['ElasticModulusEstimate'] += modulus_boost

            # 金属纳米粒子标记
            if np_name in ['Cu', 'Cu2O', 'CuO']:
                desc['HasCu'] = 1
            elif np_name == 'ZnO':
                desc['HasZn'] = 1
            elif np_name == 'Ag':
                desc['HasAg'] = 1
            elif np_name == 'TiO2':
                desc['HasTi'] = 1

            # 交联潜力 (纳米粒子可作为物理交联点)
            desc['CrosslinkPotential'] = min(1.0, desc['CrosslinkPotential'] + np_wt * 0.005)

        return desc

    @classmethod
    def _features_multilayer(cls, parsed):
        """
        多层涂层: 取各层描述符的加权 (外层权重更高)
        """
        layers = parsed['components']
        n_layers = len(layers)

        # 外层权重更高 (防污性能主要由最外层决定)
        weights = [(i + 1) / sum(range(1, n_layers + 1)) for i in range(n_layers)]

        layer_descs = []
        for layer in layers:
            desc = cls.compute_features(layer['parsed'])
            if desc is not None:
                layer_descs.append(desc)

        if not layer_descs:
            return None

        # 归一化权重
        total_w = sum(weights[:len(layer_descs)])
        weights = [w / total_w for w in weights[:len(layer_descs)]]

        result = {}
        for key in DESCRIPTOR_NAMES:
            vals = [d.get(key, 0) for d in layer_descs]
            result[key] = sum(v * w for v, w in zip(vals, weights))

        return result

    @classmethod
    def _features_natural(cls, parsed):
        """天然/已知材料: 使用查找表"""
        comp = parsed['components'][0]
        override = comp.get('override_desc')
        if override:
            # 使用预定义描述符
            desc = dict(override)
            # 补充衍生特征
            se = 40.0
            se -= desc.get('NumF', 0) * 2.5
            se -= desc.get('NumSi', 0) * 3.0
            se += desc.get('NumO', 0) * 0.5
            se += desc.get('NumN', 0) * 0.8
            se -= desc.get('LogP', 0) * 1.5
            desc['SurfaceEnergyEstimate'] = max(10, min(50, se))

            em = 2.0
            em -= desc.get('LogP', 0) * 0.1
            desc['ElasticModulusEstimate'] = max(-1, min(4, em))
            desc['RoughnessPotential'] = 0.2
            desc['CrosslinkPotential'] = 0.4
            desc['HasCu'] = 0
            desc['HasZn'] = 0
            desc['HasAg'] = 0
            desc['HasTi'] = 0
            desc['ChargeDensity'] = 0.05
            desc['HydrophilicLipophilicBalance'] = min(20, 20 * (desc.get('NumO', 0) * 16 + desc.get('NumN', 0) * 14) / max(desc.get('MW', 100), 1))
            desc['NumCl'] = 0
            desc['NumBr'] = 0
            desc['NumS'] = 0
            desc['NumP'] = 0
            desc['FractionCSP3'] = 0.5
            desc['RotBonds'] = 3
            desc['RingCount'] = 1
            desc['AromaticRings'] = 0
            desc['HeavyAtoms'] = int(desc.get('MW', 100) / 12)
            return desc
        else:
            smi = comp.get('smiles', '')
            return compute_descriptors(smi)


# ============================================================
# 预测引擎
# ============================================================

class SmartAntifoulingPredictor:
    """
    智能防污材料预测器 v6.0
    支持多种材料输入格式，自动识别材料类型
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'models.pkl')

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

    def predict(self, input_str, verbose=True):
        """
        智能预测: 自动解析输入 → 计算特征 → 预测性能
        """
        # Step 1: 解析材料
        parsed = SmartMaterialParser.parse(input_str)

        if parsed['type'] == 'unknown':
            result = {'error': f'无法识别输入: {input_str}', 'parsed': parsed}
            if verbose:
                print(f"❌ {result['error']}")
            return result

        # Step 2: 计算特征
        desc = SmartMaterialParser.compute_features(parsed)
        if desc is None:
            result = {'error': f'特征计算失败: {input_str}', 'parsed': parsed}
            if verbose:
                print(f"❌ {result['error']}")
            return result

        # Step 3: 构建特征向量
        features = np.array([[desc.get(col, 0) for col in self.feature_cols]])
        features_scaled = self.scaler.transform(features)

        # Step 4: 多模型预测
        results = {
            'input': input_str,
            'material_type': parsed['type'],
            'material_description': parsed['description'],
            'descriptors': desc,
            'predictions': {}
        }

        for target in self.target_cols:
            predictions = {}
            for m_name in ['DecisionTree', 'Ridge', 'Lasso', 'KNN', 'XGBoost']:
                pred = self.models[target][m_name].predict(features_scaled)[0]
                predictions[m_name] = float(np.clip(pred, 0, 100))

            weights = self.models[target]['Ensemble_weights']
            ens_pred = sum(w * predictions[m] for m, w in weights.items())
            predictions['Ensemble'] = float(np.clip(ens_pred, 0, 100))

            pred_values = [predictions[m] for m in ['DecisionTree', 'Ridge', 'Lasso', 'KNN', 'XGBoost']]
            predictions['model_std'] = float(np.std(pred_values))
            predictions['consensus'] = ('高一致性' if np.std(pred_values) < 5
                                       else '中等一致性' if np.std(pred_values) < 10
                                       else '低一致性(需实验验证)')

            results['predictions'][target] = predictions

        # Step 5: 综合评分
        ensemble_scores = [results['predictions'][t]['Ensemble'] for t in self.target_cols]
        results['feasibility_score'] = float(np.average(ensemble_scores, weights=[0.35, 0.25, 0.20, 0.20]))
        results['feasibility_level'] = (
            '🟢 优秀 (强烈推荐开发)' if results['feasibility_score'] >= 85 else
            '🔵 良好 (值得尝试)' if results['feasibility_score'] >= 75 else
            '🟡 一般 (需要优化设计)' if results['feasibility_score'] >= 65 else
            '🔴 较差 (不推荐此方案)'
        )

        results['key_insights'] = self._analyze_insights(parsed, desc)

        if verbose:
            self._print_results(results)

        return results

    def _analyze_insights(self, parsed, desc):
        """根据材料类型生成解读"""
        insights = []
        mat_type = parsed['type']

        # 材料类型信息
        type_names = {
            'simple_molecule': '小分子',
            'homopolymer': '均聚物',
            'copolymer': '共聚物/混合物',
            'nanocomposite': '纳米复合材料',
            'multilayer': '多层涂层',
            'natural_material': '天然/已知材料',
        }
        insights.append(f"📋 材料类型: {type_names.get(mat_type, mat_type)}")

        se = desc.get('SurfaceEnergyEstimate', 25)
        if se < 18:
            insights.append(f"✅ 表面能极低 ({se:.1f} mN/m) → 有利于污损脱附")
        elif se < 25:
            insights.append(f"🔵 表面能较低 ({se:.1f} mN/m) → 中等脱附能力")
        elif se > 35:
            insights.append(f"⚠️ 表面能较高 ({se:.1f} mN/m) → 脱附能力有限")

        if desc.get('NumF', 0) > 0:
            insights.append(f"✅ 含{desc['NumF']:.0f}个氟原子 → 显著降低表面能")
        if desc.get('NumSi', 0) > 0:
            insights.append(f"✅ 含{desc['NumSi']:.0f}个硅原子 → 降低表面能和模量")
        if desc.get('ChargeDensity', 0) > 0.1:
            insights.append(f"✅ 较高电荷密度 ({desc['ChargeDensity']:.2f}) → 有利于两性离子防污")

        if desc.get('HasCu', 0) or desc.get('HasZn', 0) or desc.get('HasAg', 0):
            metals = []
            if desc.get('HasCu', 0): metals.append('Cu')
            if desc.get('HasZn', 0): metals.append('Zn')
            if desc.get('HasAg', 0): metals.append('Ag')
            insights.append(f"✅ 含金属纳米粒子 ({', '.join(metals)}) → 增强抗菌性能")

        if mat_type == 'copolymer':
            insights.append("💡 共聚物设计: 单体比例和序列分布影响最终性能，建议实验验证")
        elif mat_type == 'nanocomposite':
            insights.append("💡 纳米复合设计: 纳米粒子分散性和界面相容性是关键因素")
        elif mat_type == 'multilayer':
            insights.append("💡 多层设计: 外层性能对防污效果影响最大")

        return insights

    def _print_results(self, results):
        """打印预测结果"""
        print("\n" + "=" * 70)
        print(f"  输入: {results['input'][:60]}")
        print(f"  类型: {results['material_description']}")
        print(f"  综合可行性评分: {results['feasibility_score']:.1f}/100")
        print(f"  可行性等级: {results['feasibility_level']}")
        print("=" * 70)

        print("\n📊 各模型预测结果:")
        print(f"{'目标':12s} | {'DT':>6s} | {'Ridge':>6s} | {'Lasso':>6s} | {'KNN':>6s} | {'XGB':>6s} | {'集成':>6s} | 一致性")
        print("-" * 85)

        for target in self.target_cols:
            label = self.target_labels.get(target, target)
            preds = results['predictions'][target]
            print(f"{label:12s} | {preds['DecisionTree']:6.1f} | {preds['Ridge']:6.1f} | "
                  f"{preds['Lasso']:6.1f} | {preds['KNN']:6.1f} | {preds['XGBoost']:6.1f} | "
                  f"{preds['Ensemble']:6.1f} | {preds['consensus']}")

        print("\n🔍 解读:")
        for insight in results.get('key_insights', []):
            print(f"  {insight}")
        print()


# ============================================================
# 命令行接口
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            print("=" * 70)
            print("  海洋防污材料智能预测平台 v6.0")
            print("  支持: SMILES | 共聚物 | 纳米复合 | 天然材料 | 多层涂层")
            print("  输入 'quit' 退出")
            print("=" * 70)
            print("\n输入示例:")
            print("  C[Si](C)(C)O[Si](C)(C)C              (PDMS小分子)")
            print("  C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3    (共聚物)")
            print("  C[Si](C)(C)O[Si](C)(C)C @ ZnO:5      (纳米复合)")
            print("  壳聚糖                                (天然材料)")
            print("  PDMS / PSBMA                          (多层涂层)")
            print()

            predictor = SmartAntifoulingPredictor()
            while True:
                try:
                    user_input = input("\n🔬 请输入材料 > ").strip()
                    if user_input.lower() in ['quit', 'q', 'exit']:
                        break
                    if not user_input:
                        continue
                    predictor.predict(user_input)
                except KeyboardInterrupt:
                    break
            print("再见！")

        elif sys.argv[1] == '--demo':
            # 演示各种输入格式
            predictor = SmartAntifoulingPredictor()
            demos = [
                'C[Si](C)(C)O[Si](C)(C)C',                          # PDMS
                'C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3',                 # 共聚物
                'C[Si](C)(C)O[Si](C)(C)C @ ZnO:5',                   # 纳米复合
                '壳聚糖',                                              # 天然材料
                'PDMS / PSBMA',                                       # 多层涂层
                'C[N+](C)(C)CCCS([O-])(=O)=O:0.5 + FC(F)(F)C(F)(F)F:0.5',  # 两性离子-氟硅共聚
            ]
            for d in demos:
                predictor.predict(d)
                print("-" * 70)
        else:
            predictor = SmartAntifoulingPredictor()
            predictor.predict(' '.join(sys.argv[1:]))
    else:
        print("海洋防污材料智能预测平台 v6.0")
        print()
        print("用法:")
        print('  python smart_predict.py "SMILES或材料描述"')
        print("  python smart_predict.py --interactive")
        print("  python smart_predict.py --demo")
        print()
        print("支持的输入格式:")
        print("  纯小分子:    C[Si](C)(C)O[Si](C)(C)C")
        print("  均聚物:      POLY[C=CC(=O)O]")
        print("  共聚物:      SMILES_A:0.7 + SMILES_B:0.3")
        print("  纳米复合:    POLYMER_SMILES @ NP_NAME:wt%")
        print("  多层涂层:    LAYER1 / LAYER2")
        print("  天然材料:    壳聚糖 / chitosan / PDMS")
        print()
        print("支持的纳米粒子:", ', '.join(NANOPARTICLE_DB.keys()))
        print("支持的材料名称:", ', '.join(sorted(set(
            [k for k in NATURAL_MATERIAL_DB.keys() if not k.isupper() or len(k) <= 4]
        ))[:20]))
