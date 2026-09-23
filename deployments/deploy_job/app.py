#!/usr/bin/env python3
"""
海洋防污材料ML预测平台 v6.0 — 多格式复合材料预测版
速度优化 + 全中文界面 + 智能材料解析器 + 6种输入格式 + 合成制备路线

输入语法:
  纯小分子:    C[Si](C)(C)O[Si](C)(C)C
  均聚物:      POLY[C=CC(=O)O]
  共聚物:      C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3
  纳米复合:    C[Si](C)(C)O[Si](C)(C)C @ ZnO:5
  多层涂层:    PDMS / PSBMA
  天然材料:    壳聚糖
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 中文字体
for font in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']:
    try:
        rcParams['font.sans-serif'] = [font]
        break
    except:
        pass
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100

import json, os, pickle, io, time
from collections import defaultdict
from functools import lru_cache

import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, Lipinski, Crippen

# 合成路线模块
from synthesis_routes import (
    get_synthesis_route, generate_synthesis_flowchart,
    generate_reagent_table, format_synthesis_report,
    list_all_routes, list_routes_by_class, get_database_summary,
    SYNTHESIS_DATABASE, get_available_materials
)

# ============================================================
# 纳米粒子数据库 (from smart_predict.py v6.0)
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

# ============================================================
# 天然/复杂材料查找表 (from smart_predict.py v6.0)
# ============================================================

NATURAL_MATERIAL_DB = {
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

# Base 28 descriptor names (used by SmartMaterialParser for composite features)
BASE_DESCRIPTOR_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'RingCount',
    'AromaticRings', 'HeavyAtoms', 'FractionCSP3',
    'NumF', 'NumCl', 'NumBr', 'NumN', 'NumO', 'NumS', 'NumSi', 'NumP',
    'HasCu', 'HasZn', 'HasAg', 'HasTi',
    'ChargeDensity', 'HydrophilicLipophilicBalance',
    'SurfaceEnergyEstimate', 'ElasticModulusEstimate',
    'RoughnessPotential', 'CrosslinkPotential',
]


# ============================================================
# 智能材料解析器 (ported from smart_predict.py v6.0)
# ============================================================

class SmartMaterialParser:
    """
    智能材料解析器 — 自动识别输入材料类型并选择合适的特征计算策略
    支持: 纯小分子 / 均聚物 / 共聚物 / 纳米复合 / 多层涂层 / 天然材料
    """
    TYPE_SIMPLE = 'simple_molecule'
    TYPE_HOMOPOLYMER = 'homopolymer'
    TYPE_COPOLYMER = 'copolymer'
    TYPE_NANOCOMPOSITE = 'nanocomposite'
    TYPE_MULTILAYER = 'multilayer'
    TYPE_NATURAL = 'natural_material'

    TYPE_LABELS = {
        'simple_molecule': '小分子/单体',
        'homopolymer': '均聚物',
        'copolymer': '共聚物/混合物',
        'nanocomposite': '纳米复合材料',
        'multilayer': '多层涂层',
        'natural_material': '天然/已知材料',
        'unknown': '未知',
    }

    @classmethod
    def parse(cls, input_str):
        """
        解析输入字符串，返回材料类型和组分信息。
        返回: dict with keys: type, components, raw_input, description
        """
        input_str = input_str.strip()

        # 1. 天然材料/常见缩写
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

        # 2. 多层涂层 (用 / 分隔, 但不含 + 和 @)
        if '/' in input_str and '+' not in input_str and '@' not in input_str:
            layers = [l.strip() for l in input_str.split('/')]
            if len(layers) >= 2:
                parsed_layers = []
                for i, layer in enumerate(layers):
                    sub = cls.parse(layer)
                    parsed_layers.append({'layer_index': i, 'parsed': sub})
                return {
                    'type': cls.TYPE_MULTILAYER,
                    'components': parsed_layers,
                    'raw_input': input_str,
                    'description': f'{len(layers)}层涂层: ' + ' / '.join(
                        [l['parsed']['description'] for l in parsed_layers])
                }

        # 3. 纳米复合 (用 @ 分隔)
        if '@' in input_str:
            parts = input_str.split('@')
            if len(parts) == 2:
                polymer_part = parts[0].strip()
                np_part = parts[1].strip()
                np_match = re.match(r'(\w+)(?::(\d+\.?\d*))?', np_part)
                if np_match:
                    np_name = np_match.group(1)
                    np_wt = float(np_match.group(2)) if np_match.group(2) else 5.0
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

        # 4. 共聚物/混合物 (用 " + " 分隔 — 避免匹配SMILES中的[N+])
        copolymer_parts = re.split(r'\s+\+\s+', input_str)
        if len(copolymer_parts) >= 2:
            parts = [p.strip() for p in copolymer_parts if p.strip()]
            components = []
            for part in parts:
                ratio_match = re.match(r'(.+?):(\d+\.?\d*)', part)
                if ratio_match:
                    smi = ratio_match.group(1).strip()
                    ratio = float(ratio_match.group(2))
                else:
                    smi = part.strip()
                    ratio = 1.0 / len(parts)
                comp_parsed = cls.parse(smi)
                components.append({'smiles': smi, 'ratio': ratio, 'parsed': comp_parsed})
            # 归一化比例
            total_ratio = sum(c['ratio'] for c in components)
            for c in components:
                c['ratio'] /= total_ratio
            return {
                'type': cls.TYPE_COPOLYMER,
                'components': components,
                'raw_input': input_str,
                'description': '共聚物/混合物: ' + ' + '.join(
                    [f"{c['parsed']['description']}({c['ratio']:.0%})" for c in components])
            }

        # 5. 均聚物 POLY[...] 格式
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
        根据解析结果计算28个基础描述符（不含交互特征）。
        交互特征由 add_interaction_features() 在最终预测时追加。
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
        return None

    @classmethod
    def _features_simple(cls, parsed):
        smi = parsed['components'][0]['smiles']
        return compute_base_descriptors(smi)

    @classmethod
    def _features_homopolymer(cls, parsed):
        smi = parsed['components'][0]['smiles']
        desc = compute_base_descriptors(smi)
        if desc is None:
            return None
        desc['MW'] *= 50
        desc['RotBonds'] = max(1, desc['RotBonds'] - 2)
        desc['CrosslinkPotential'] = min(1.0, desc['CrosslinkPotential'] * 1.5)
        return desc

    @classmethod
    def _features_copolymer(cls, parsed):
        components = parsed['components']
        comp_descs = []
        ratios = []
        for comp in components:
            sub_parsed = comp['parsed']
            desc = None
            if sub_parsed['type'] == cls.TYPE_NATURAL:
                if sub_parsed['components']:
                    override = sub_parsed['components'][0].get('override_desc')
                    if override:
                        desc = dict(override)
                    else:
                        desc = compute_base_descriptors(
                            sub_parsed['components'][0].get('smiles', ''))
            elif sub_parsed['type'] == 'unknown':
                smi = comp.get('smiles', '')
                if smi:
                    desc = compute_base_descriptors(smi)
            else:
                smi = comp.get('smiles', '')
                if not smi and sub_parsed.get('components'):
                    smi = sub_parsed['components'][0].get('smiles', '')
                if smi:
                    desc = compute_base_descriptors(smi)
            if desc is not None:
                comp_descs.append(desc)
                ratios.append(comp['ratio'])
        if not comp_descs:
            return None
        total = sum(ratios)
        ratios = [r / total for r in ratios]
        result = {}
        for key in BASE_DESCRIPTOR_NAMES:
            vals = [d.get(key, 0) for d in comp_descs]
            result[key] = sum(v * r for v, r in zip(vals, ratios))
        if len(comp_descs) >= 2:
            d1, d2 = comp_descs[0], comp_descs[1]
            logp_diff = abs(d1.get('LogP', 0) - d2.get('LogP', 0))
            se_diff = abs(d1.get('SurfaceEnergyEstimate', 25) -
                          d2.get('SurfaceEnergyEstimate', 25))
            result['SurfaceEnergyEstimate'] -= se_diff * 0.1
            result['RoughnessPotential'] += logp_diff * 0.05
        result['MW'] *= 30
        result['CrosslinkPotential'] = min(1.0, result['CrosslinkPotential'] * 1.3)
        return result

    @classmethod
    def _features_nanocomposite(cls, parsed):
        comp = parsed['components'][0]
        polymer_parsed = comp['polymer']
        np_name = comp['nanoparticle']
        np_wt = comp['np_wt_pct']
        np_info = comp.get('np_info')
        desc = cls.compute_features(polymer_parsed)
        if desc is None:
            return None
        if np_info:
            se_effect = -np_wt * 0.2
            roughness_boost = np_wt * 0.02
            modulus_boost = np_wt * 0.01
            desc['SurfaceEnergyEstimate'] = max(10, desc.get('SurfaceEnergyEstimate', 25) + se_effect)
            desc['RoughnessPotential'] = desc.get('RoughnessPotential', 0) + roughness_boost
            desc['ElasticModulusEstimate'] = desc.get('ElasticModulusEstimate', 2) + modulus_boost
            if np_name in ['Cu', 'Cu2O', 'CuO']:
                desc['HasCu'] = 1
            elif np_name == 'ZnO':
                desc['HasZn'] = 1
            elif np_name == 'Ag':
                desc['HasAg'] = 1
            elif np_name == 'TiO2':
                desc['HasTi'] = 1
            desc['CrosslinkPotential'] = min(1.0, desc.get('CrosslinkPotential', 0) + np_wt * 0.005)
        return desc

    @classmethod
    def _features_multilayer(cls, parsed):
        layers = parsed['components']
        n_layers = len(layers)
        weights = [(i + 1) / sum(range(1, n_layers + 1)) for i in range(n_layers)]
        layer_descs = []
        for layer in layers:
            desc = cls.compute_features(layer['parsed'])
            if desc is not None:
                layer_descs.append(desc)
        if not layer_descs:
            return None
        total_w = sum(weights[:len(layer_descs)])
        weights = [w / total_w for w in weights[:len(layer_descs)]]
        result = {}
        for key in BASE_DESCRIPTOR_NAMES:
            vals = [d.get(key, 0) for d in layer_descs]
            result[key] = sum(v * w for v, w in zip(vals, weights))
        return result

    @classmethod
    def _features_natural(cls, parsed):
        comp = parsed['components'][0]
        override = comp.get('override_desc')
        if override:
            desc = dict(override)
            se = 40.0
            se -= desc.get('NumF', 0) * 2.5
            se -= desc.get('NumSi', 0) * 3.0
            se += desc.get('NumO', 0) * 0.5
            se += desc.get('NumN', 0) * 0.8
            se -= desc.get('LogP', 0) * 1.5
            desc['SurfaceEnergyEstimate'] = max(10, min(50, se))
            em = 2.0 - desc.get('LogP', 0) * 0.1
            desc['ElasticModulusEstimate'] = max(-1, min(4, em))
            desc['RoughnessPotential'] = 0.2
            desc['CrosslinkPotential'] = 0.4
            desc['HasCu'] = 0
            desc['HasZn'] = 0
            desc['HasAg'] = 0
            desc['HasTi'] = 0
            desc['ChargeDensity'] = 0.05
            desc['HydrophilicLipophilicBalance'] = min(
                20, 20 * (desc.get('NumO', 0) * 16 + desc.get('NumN', 0) * 14)
                / max(desc.get('MW', 100), 1))
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
            return compute_base_descriptors(smi)


def format_parsed_info(parsed):
    """将解析结果格式化为用户可读的Markdown字符串"""
    mat_type = parsed['type']
    label = SmartMaterialParser.TYPE_LABELS.get(mat_type, mat_type)
    lines = [f"**识别类型**: {label}",
             f"**描述**: {parsed['description']}"]

    if mat_type == SmartMaterialParser.TYPE_COPOLYMER:
        lines.append("**组分**:")
        for i, c in enumerate(parsed['components'], 1):
            lines.append(f"  - 组分{i}: {c['parsed']['description']} — 比例 {c['ratio']:.1%}")
    elif mat_type == SmartMaterialParser.TYPE_NANOCOMPOSITE:
        comp = parsed['components'][0]
        lines.append(f"**聚合物基体**: {comp['polymer']['description']}")
        np_name = comp['nanoparticle']
        np_info = comp.get('np_info')
        np_label = f"{np_name}"
        if np_info:
            np_label += f" ({np_info['name']})"
        lines.append(f"**纳米粒子**: {np_label}, 含量 {comp['np_wt_pct']}wt%")
        if np_info:
            lines.append(f"  - 带隙: {np_info['bandgap_eV']} eV, "
                         f"比表面积: {np_info['surface_area_m2g']} m²/g, "
                         f"抗菌指数: {np_info['antibacterial_index']}")
    elif mat_type == SmartMaterialParser.TYPE_MULTILAYER:
        lines.append("**各层**:")
        for layer in parsed['components']:
            idx = layer['layer_index'] + 1
            lines.append(f"  - 第{idx}层: {layer['parsed']['description']}")
    elif mat_type == SmartMaterialParser.TYPE_HOMOPOLYMER:
        smi = parsed['components'][0]['smiles']
        lines.append(f"**单体SMILES**: `{smi}`")
    elif mat_type == SmartMaterialParser.TYPE_NATURAL:
        comp = parsed['components'][0]
        lines.append(f"**材料名**: {comp['name']}, **分类**: {comp.get('class', '—')}")
    elif mat_type == SmartMaterialParser.TYPE_SIMPLE:
        smi = parsed['components'][0]['smiles']
        lines.append(f"**SMILES**: `{smi}`")

    return '\n'.join(lines)


# ============================================================
# 材料数据库（从训练数据集提取 + 常用别名）— 保持原有
# ============================================================

_MATERIAL_DB = None

_ALIASES = {
    'PDMS': 'PDMS三聚体', '聚二甲基硅氧烷': 'PDMS三聚体', '硅橡胶': 'PDMS三聚体',
    '硅树脂': 'PDMS三聚体', 'silicone': 'PDMS三聚体', 'dimethicone': 'PDMS三聚体',
    '硅油': 'PDMS+甲基硅油', '苯基硅油': 'PDMS+苯基硅油', '氨基硅油': 'PDMS+氨基改性',
    'PTFE': 'PTFE单体单元', '聚四氟乙烯': 'PTFE单体单元', '特氟龙': 'PTFE单体单元',
    'Teflon': 'PTFE单体单元', 'PVDF': '含氟丙烯酸酯单体', '聚偏氟乙烯': '含氟丙烯酸酯单体',
    '氟碳树脂': '含氟甲基丙烯酸酯', '氟硅': '氟硅嵌段共聚物',
    'PEG': 'PEG200', '聚乙二醇': 'PEG200', 'PHEMA': 'HEMA单体',
    '聚丙烯酸': '丙烯酸单体', 'PAA': '丙烯酸单体', '壳聚糖': '壳聚糖单体',
    'chitosan': '壳聚糖单体', '海藻酸钠': '海藻酸单体', '透明质酸': '透明质酸单体',
    '玻尿酸': '透明质酸单体', '聚乙烯醇': 'PVA单体单元', 'PVA': 'PVA单体单元',
    '明胶': '明胶单体', 'gelatin': '明胶单体', '琼脂': '琼脂糖单体',
    'SBMA': '磺酸甜菜碱SBMA', '磺酸甜菜碱': '磺酸甜菜碱SBMA',
    'CBMA': '羧酸甜菜碱CBMA', '羧酸甜菜碱': '羧酸甜菜碱CBMA',
    'MPC': '磷酸胆碱PCBMA', '磷酰胆碱': '磷酸胆碱PCBMA',
    'DOPA': '仿贻贝多巴胺前体', '多巴胺': '仿贻贝多巴胺前体',
    'dopamine': '仿贻贝多巴胺前体',
    'SLIPS': 'SLIPS-PDMS+PEG润滑液', '超滑表面': 'SLIPS-PDMS+PEG润滑液',
    '荷叶效应': '仿荷叶丙烯酸-PEG', '鲨鱼皮': '仿鲨鱼皮全氟',
    '贻贝': '仿贻贝多巴胺前体', 'mussel': '仿贻贝多巴胺前体',
    '纳米氧化锌': 'PDMS/ZnO纳米复合', 'ZnO': 'PDMS/ZnO纳米复合',
    '纳米二氧化硅': 'PDMS/SiO2纳米复合', 'SiO2': 'PDMS/SiO2纳米复合',
    '纳米二氧化钛': 'PDMS/TiO2纳米复合', 'TiO2': 'PDMS/TiO2纳米复合',
    '纳米银': 'PDMS/Ag纳米复合', 'Ag': 'PDMS/Ag纳米复合',
    '银纳米粒子': 'PDMS/Ag纳米复合', '石墨烯': 'PDMS/石墨烯复合',
    '温敏': '温度响应PNIPAM-丙烯酸', 'PNIPAM': 'PNIPAM-羟乙基',
    'pH响应': 'pH响应丙烯酸-PEG-PDMS', '光响应': '光响应丙烯酸-氟-TiO2',
    '自修复': 'PDMS-PUa自修复', '氧化亚铜': 'PDMS/Cu2O', 'Cu2O': 'PDMS/Cu2O',
}

_RAW_MATERIALS = [
    {'name': 'PDMS三聚体', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS五聚体', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS+甲基硅油', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CCCC)C', 'cls': 'silicone'},
    {'name': 'PDMS+辛基硅油', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CCCCCCCC)C', 'cls': 'silicone'},
    {'name': 'PDMS+苯基硅油', 'smiles': 'C[Si](C)(c1ccccc1)C', 'cls': 'silicone'},
    {'name': 'PDMS+三氟丙基', 'smiles': 'C[Si](C)(C)O[Si](C)(CCC(F)(F)F)C', 'cls': 'silicone'},
    {'name': 'PDMS+全氟己基', 'smiles': 'C[Si](C)(C)O[Si](C)(CCC(F)(F)C(F)(F)F)C', 'cls': 'silicone'},
    {'name': '氟硅嵌段共聚物', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)[Si](C)(C)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS-g-PEG200', 'smiles': 'C[Si](C)(CCOCCOCCO)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS-g-PEG400', 'smiles': 'C[Si](C)(CCOCCOCCOCCOCCO)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS-PUa自修复', 'smiles': 'C[Si](C)(C)NCCCCCCNC(=O)NCCCCCCN', 'cls': 'silicone'},
    {'name': '硅氧烷-聚氨酯', 'smiles': 'C[Si](C)(C)O[Si](C)(C)OC(=O)NCCCCCCNC(=O)O', 'cls': 'silicone'},
    {'name': 'PDMS/SiO2纳米复合', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Si](=O)(=O)', 'cls': 'silicone'},
    {'name': 'PDMS/ZnO纳米复合', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Zn]', 'cls': 'silicone'},
    {'name': 'PDMS七聚体MW1200', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS+三氟甲基侧链', 'smiles': 'C[Si](C)(C)O[Si](C)(CC(F)(F)F)C', 'cls': 'silicone'},
    {'name': '羟基封端PDMS', 'smiles': 'C[Si](C)(O)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS+癸基硅油', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CCCCCCCCCC)C', 'cls': 'silicone'},
    {'name': 'PDMS+甲苯基硅油', 'smiles': 'C[Si](C)(C)O[Si](C)(c1ccc(C)cc1)C', 'cls': 'silicone'},
    {'name': 'PDMS+乙烯基改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCCC=C)C', 'cls': 'silicone'},
    {'name': 'PDMS+氨基改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCCN)C', 'cls': 'silicone'},
    {'name': 'PDMS+羧基改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCC(=O)O)C', 'cls': 'silicone'},
    {'name': 'PDMS+羧基PEG改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCOCC(=O)O)C', 'cls': 'silicone'},
    {'name': 'PDMS+全氟辛基', 'smiles': 'C[Si](C)(C)O[Si](C)(CCC(F)(F)C(F)(F)C(F)(F)F)C', 'cls': 'silicone'},
    {'name': 'PDMS+全氟丁基', 'smiles': 'C[Si](C)(C)O[Si](C)(CC(F)(F)C(F)(F)F)C', 'cls': 'silicone'},
    {'name': '双氟硅改性PDMS', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CCC(F)(F)F)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS-g-PEG600', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CCOCCOCCOCCO)C', 'cls': 'silicone'},
    {'name': 'PDMS-PU-PEG', 'smiles': 'C[Si](C)(C)O[Si](C)(C)NCCCCCCNC(=O)OCCO', 'cls': 'silicone'},
    {'name': 'PDMS/TiO2纳米复合', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ti]', 'cls': 'silicone'},
    {'name': 'PDMS/Ag纳米复合', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ag]', 'cls': 'silicone'},
    {'name': 'PDMS+醋酸封端', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C.CC(=O)O', 'cls': 'silicone'},
    {'name': '苯基甲基硅氧烷', 'smiles': 'C[Si](C)(c1ccccc1)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS四聚体', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C', 'cls': 'silicone'},
    {'name': 'PDMS+甘油改性', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(CC(O)CO)C', 'cls': 'silicone'},
    {'name': 'PDMS+磺酸基改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCS(=O)(=O)O)C', 'cls': 'silicone'},
    {'name': 'PDMS+季铵盐改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CC[N+](C)(C)C)C', 'cls': 'silicone'},
    {'name': 'PDMS+磷酸基改性', 'smiles': 'C[Si](C)(C)O[Si](C)(CCOP(=O)(O)O)C', 'cls': 'silicone'},
    {'name': 'PDMS+羟乙基酰胺', 'smiles': 'C[Si](C)(C)O[Si](C)(CCC(=O)NCCO)C', 'cls': 'silicone'},
    {'name': '全氟戊烷', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': 'PCTFE类', 'smiles': 'FC(F)(F)C(F)(Cl)C(F)(F)C(F)(Cl)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': 'PTFE单体单元', 'smiles': 'FC(F)=C(F)F', 'cls': 'fluoropolymer'},
    {'name': '含氟丙烯酸酯单体', 'smiles': 'OC(=O)CCC(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '全氟丙烯酸酯', 'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '含氟甲基丙烯酸酯', 'smiles': 'OC(=O)C(C)CC(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '含氟芳香酸', 'smiles': 'FC(F)(F)C(F)(F)c1ccc(CC(=O)O)cc1', 'cls': 'fluoropolymer'},
    {'name': '氟硅共聚物', 'smiles': 'OC(=O)CC(F)(F)C(F)(F)F.[Si](C)(C)O[Si](C)(C)C', 'cls': 'fluoropolymer'},
    {'name': '全氟辛基链', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '对氟苯甲酸酯', 'smiles': 'OC(=O)c1ccc(F)cc1', 'cls': 'fluoropolymer'},
    {'name': '含氟聚氨酯', 'smiles': 'FC(F)(F)C(F)(F)CC(=O)NCCCCCCNC(=O)', 'cls': 'fluoropolymer'},
    {'name': '含氟甘油酯', 'smiles': 'FC(F)(F)C(F)(F)C(=O)OCC(O)CO', 'cls': 'fluoropolymer'},
    {'name': '全氟丁烷', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '全氟己烷', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '甲基丙烯酸全氟丁酯', 'smiles': 'OC(=O)C(C)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '甲基丙烯酸全氟辛酯', 'smiles': 'OC(=O)C(C)CC(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '含氟苯酚', 'smiles': 'FC(F)(F)C(F)(F)c1ccc(O)cc1', 'cls': 'fluoropolymer'},
    {'name': '全氟丁基乙醇', 'smiles': 'FC(F)(F)C(F)(F)CCO', 'cls': 'fluoropolymer'},
    {'name': '全氟己基乙醇', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)CCO', 'cls': 'fluoropolymer'},
    {'name': '全氟辛基乙醇', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)CCO', 'cls': 'fluoropolymer'},
    {'name': '3,3,3-三氟丙烯酸', 'smiles': 'OC(=O)CC(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '三氟甲基丙烯酸羟乙酯', 'smiles': 'C(C(F)(F)F)C(=O)OCCO', 'cls': 'fluoropolymer'},
    {'name': '全氟丁基磺酸', 'smiles': 'FC(F)(F)C(F)(F)S(=O)(=O)O', 'cls': 'fluoropolymer'},
    {'name': '全氟己基磺酸', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O', 'cls': 'fluoropolymer'},
    {'name': '含氟酰胺醇', 'smiles': 'FC(F)(F)C(F)(F)C(=O)NCCO', 'cls': 'fluoropolymer'},
    {'name': '全氟丁酸', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(=O)O', 'cls': 'fluoropolymer'},
    {'name': '二氟丙烯酸', 'smiles': 'OC(=O)C=C(F)F', 'cls': 'fluoropolymer'},
    {'name': '五氟苯', 'smiles': 'FC(F)(F)c1ccccc1', 'cls': 'fluoropolymer'},
    {'name': '六氟苯', 'smiles': 'FC(F)(F)c1ccc(F)c(F)c1F', 'cls': 'fluoropolymer'},
    {'name': '全氟戊烯酸', 'smiles': 'OC(=O)CC(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '含氟甘油单酯', 'smiles': 'FC(F)(F)C(F)(F)CC(=O)OCC(O)CO', 'cls': 'fluoropolymer'},
    {'name': '全氟辛酸', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)CC(=O)O', 'cls': 'fluoropolymer'},
    {'name': '五氟丙酸乙烯酯', 'smiles': 'OC(=O)C(F)(F)C(F)(F)F', 'cls': 'fluoropolymer'},
    {'name': '全氟癸基酰胺', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC(=O)NCCCCC', 'cls': 'fluoropolymer'},
    {'name': '含氟苯二甲酸', 'smiles': 'FC(F)(F)C(F)(F)c1ccc(C(=O)O)cc1F', 'cls': 'fluoropolymer'},
    {'name': 'PVA单体单元', 'smiles': 'OC(=O)C(O)CO', 'cls': 'hydrogel'},
    {'name': 'PEG200', 'smiles': 'OCCOCCOCCO', 'cls': 'hydrogel'},
    {'name': 'PEG600', 'smiles': 'OCCOCCOCCOCCOCCOCCOCCO', 'cls': 'hydrogel'},
    {'name': 'PEG2000', 'smiles': 'OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'cls': 'hydrogel'},
    {'name': '丙烯酸单体', 'smiles': 'OC(=O)C=C', 'cls': 'hydrogel'},
    {'name': 'HEMA单体', 'smiles': 'OC(=O)C(C)C(=O)NCCO', 'cls': 'hydrogel'},
    {'name': '丙烯酰胺单体', 'smiles': 'NC(=O)C=C', 'cls': 'hydrogel'},
    {'name': '苹果酸', 'smiles': 'OC(=O)CC(O)C(=O)O', 'cls': 'hydrogel'},
    {'name': '酒石酸', 'smiles': 'OC(=O)C(O)C(O)C(=O)O', 'cls': 'hydrogel'},
    {'name': '甘油', 'smiles': 'OCC(O)CO', 'cls': 'hydrogel'},
    {'name': '柠檬酸', 'smiles': 'OC(=O)CC(O)(CC(=O)O)C(=O)O', 'cls': 'hydrogel'},
    {'name': '马来酸', 'smiles': 'OC(=O)C=CC(=O)O', 'cls': 'hydrogel'},
    {'name': '缩水甘油', 'smiles': 'OCC1OC(O1)CO', 'cls': 'hydrogel'},
    {'name': 'HEMA-PEG共聚', 'smiles': 'OC(=O)C(C)C(=O)OCCOCCO', 'cls': 'hydrogel'},
    {'name': 'N-羟乙基丙烯酰胺', 'smiles': 'NC(=O)C(C)C(=O)NCCO', 'cls': 'hydrogel'},
    {'name': '天冬氨酸', 'smiles': 'OC(=O)CC(N)C(=O)O', 'cls': 'hydrogel'},
    {'name': '谷氨酸', 'smiles': 'OC(=O)CCC(N)C(=O)O', 'cls': 'hydrogel'},
    {'name': '天冬酰胺', 'smiles': 'OC(=O)C(N)CC(=O)N', 'cls': 'hydrogel'},
    {'name': '山梨醇', 'smiles': 'OCC(O)C(O)C(O)C(O)CO', 'cls': 'hydrogel'},
    {'name': '乙醇酸二聚体', 'smiles': 'OC(=O)C(O)C(=O)O', 'cls': 'hydrogel'},
    {'name': 'PEG4000', 'smiles': 'OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO', 'cls': 'hydrogel'},
    {'name': '丙烯酸-醋酸乙烯共聚', 'smiles': 'OC(=O)C=C.CC(=O)OCC', 'cls': 'hydrogel'},
    {'name': '丙二酰胺', 'smiles': 'NC(=O)CC(=O)N', 'cls': 'hydrogel'},
    {'name': '羟基丁二酸', 'smiles': 'OC(=O)C(O)CC(=O)O', 'cls': 'hydrogel'},
    {'name': '衣康酸酐', 'smiles': 'OC(=O)C1CC(=O)OC1=O', 'cls': 'hydrogel'},
    {'name': 'PEG-丙烯酸酯', 'smiles': 'OCCOCCNC(=O)C=C', 'cls': 'hydrogel'},
    {'name': '甲基丙烯酸-PEG酯', 'smiles': 'OC(=O)C(C)C(=O)NCCOCCO', 'cls': 'hydrogel'},
    {'name': 'N-丁基丙烯酰胺', 'smiles': 'NC(=O)C(C)C(=O)NCCCCN', 'cls': 'hydrogel'},
    {'name': '丙烯酸-水体系', 'smiles': 'OC(=O)C=C.O', 'cls': 'hydrogel'},
    {'name': '甲基丙烯酸环己酯', 'smiles': 'OC(=O)C(C)C(=O)NC1CCCCC1', 'cls': 'hydrogel'},
    {'name': '丙烯酸-乙二醇共聚', 'smiles': 'OC(=O)C=C.OCCO', 'cls': 'hydrogel'},
    {'name': 'HEMA-甘油酯共聚', 'smiles': 'OC(=O)C(C)C(=O)OCC(O)CO', 'cls': 'hydrogel'},
    {'name': 'PEG-酰胺嵌段', 'smiles': 'OCCOCCOCC(=O)NCCOCCO', 'cls': 'hydrogel'},
    {'name': '丙烯酸-甲基丙烯酸共聚', 'smiles': 'OC(=O)C=C.C(=C)C(=O)O', 'cls': 'hydrogel'},
    {'name': 'PVA-PEG共混', 'smiles': 'OC(=O)C(O)COCCO', 'cls': 'hydrogel'},
    {'name': '磺酸甜菜碱SBMA', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '磷酸胆碱PCBMA', 'smiles': 'C[N+](C)(C)CCOP(=O)([O-])O', 'cls': 'zwitterionic'},
    {'name': '羧酸甜菜碱CBMA', 'smiles': 'C[N+](C)(C)CCC(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '羟基磺酸甜菜碱', 'smiles': 'OCC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '酰胺磺酸甜菜碱', 'smiles': 'C[N+](C)(C)CC(=O)NCCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '双磷酸胆碱', 'smiles': 'C[N+](C)(C)CCOP(=O)([O-])OCC[N+](C)(C)C', 'cls': 'zwitterionic'},
    {'name': '甲基丙烯酸磺酸甜菜碱', 'smiles': 'OC(=O)C(C)[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': 'SBMA-甲基丙烯酸酯', 'smiles': 'OC(=O)C(C)C(=O)OCC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '双羧酸甜菜碱', 'smiles': 'C[N+](C)(C)CCC(=O)OCC[N+](C)(C)CCC(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '磺酸-羧酸混合甜菜碱', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].C[N+](C)(C)CCC(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': 'PCBMA-甲基丙烯酸酯', 'smiles': 'OC(=O)C(C)C(=O)OCC[N+](C)(C)CCOP(=O)([O-])O', 'cls': 'zwitterionic'},
    {'name': 'SBMA-丙烯酸共聚', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].OC(=O)C=C', 'cls': 'zwitterionic'},
    {'name': '扩展链磺酸甜菜碱', 'smiles': 'C[N+](C)(C)CC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '长链磺酸甜菜碱', 'smiles': 'C[N+](C)(C)CC(=O)NCCCCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '酰胺型磺酸甜菜碱酯', 'smiles': 'OC(=O)C(C)C(=O)NCC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': 'SBMA-PEG共混', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].OCCOCCO', 'cls': 'zwitterionic'},
    {'name': 'CBMA-丙烯酸共聚', 'smiles': 'C[N+](C)(C)CCC(=O)[O-].OC(=O)C=C', 'cls': 'zwitterionic'},
    {'name': '羟基磷酸胆碱', 'smiles': 'C[N+](C)(C)CCOP(=O)([O-])OCCO', 'cls': 'zwitterionic'},
    {'name': '丙基磺酸甜菜碱酯', 'smiles': 'OC(=O)C(C)C(=O)OCCC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '磺酸-羧酸双甜菜碱', 'smiles': 'C[N+](C)(C)CC(=O)NCC[N+](C)(C)CCC(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': 'SBMA-氟碳混合', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].FC(F)(F)C(F)(F)F', 'cls': 'zwitterionic'},
    {'name': 'SBMA-PDMS杂化', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].[Si](C)(C)O[Si](C)(C)C', 'cls': 'zwitterionic'},
    {'name': 'PCBMA-PDMS杂化', 'smiles': 'C[N+](C)(C)CCOP(=O)([O-])O.[Si](C)(C)O[Si](C)(C)C', 'cls': 'zwitterionic'},
    {'name': 'SBMA-丙烯酸-PEG三元', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].OC(=O)C=C.OCCOCCO', 'cls': 'zwitterionic'},
    {'name': '甜菜碱-甘油酯', 'smiles': 'C[N+](C)(C)CC(=O)NCC(O)CO', 'cls': 'zwitterionic'},
    {'name': '甲基丙烯酸羧酸甜菜碱酯', 'smiles': 'OC(=O)C(C)C(=O)OCC[N+](C)(C)CCC(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': 'SBMA-甲基丙烯酸共聚', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].OC(=O)C(C)C(=O)O', 'cls': 'zwitterionic'},
    {'name': 'SBMA-丙烯酰胺共聚', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].NC(=O)C=C', 'cls': 'zwitterionic'},
    {'name': 'SBMA-丙烯酸-醋酸共聚', 'smiles': 'C[N+](C)(C)CCS(=O)(=O)[O-].OC(=O)C=C.CC(=O)O', 'cls': 'zwitterionic'},
    {'name': '磷酸胆碱-磺酸双功能', 'smiles': 'C[N+](C)(C)CCOP(=O)([O-])OCC[N+](C)(C)CCS(=O)(=O)[O-]', 'cls': 'zwitterionic'},
    {'name': '丙烯酸铜聚合物', 'smiles': 'OC(=O)C(C)C(=O)O[Cu]', 'cls': 'self_polishing'},
    {'name': '丙烯酸锌聚合物', 'smiles': 'OC(=O)C(C)C(=O)O[Zn]', 'cls': 'self_polishing'},
    {'name': '丙烯酸叔丁酯', 'smiles': 'OC(=O)C(C)C(=O)OC(C)(C)C', 'cls': 'self_polishing'},
    {'name': '丙烯酸-PEG水解型', 'smiles': 'OC(=O)C=C.OCCOCCO', 'cls': 'self_polishing'},
    {'name': '甲基丙烯酸-乙醇酸共聚', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)O', 'cls': 'self_polishing'},
    {'name': '甲基丙烯酸-丙酸共聚', 'smiles': 'OC(=O)C(C)C(=O)OCCC(=O)O', 'cls': 'self_polishing'},
    {'name': 'SPC酯键型', 'smiles': 'OC(=O)C(C)C(=O)OCCOC(=O)C', 'cls': 'self_polishing'},
    {'name': '铜锌复合SPC', 'smiles': 'OC(=O)C(C)C(=O)O[Cu].[Zn]', 'cls': 'self_polishing'},
    {'name': '丙烯酸-乙醇酸铜', 'smiles': 'OC(=O)C=C.OCC(=O)O[Cu]', 'cls': 'self_polishing'},
    {'name': '丙烯酸-乙醇酸锌', 'smiles': 'OC(=O)C=C.OCC(=O)O[Zn]', 'cls': 'self_polishing'},
    {'name': 'SPC铜基丙烯酸酯', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)O[Cu]', 'cls': 'self_polishing'},
    {'name': 'SPC锌基丙烯酸酯', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)O[Zn]', 'cls': 'self_polishing'},
    {'name': 'SPC-PEG水解型', 'smiles': 'OC(=O)C(C)C(=O)OCCOCC(=O)O', 'cls': 'self_polishing'},
    {'name': 'SPC-琥珀酸酯', 'smiles': 'OC(=O)C(C)C(=O)OCCOC(=O)CC(=O)O', 'cls': 'self_polishing'},
    {'name': '丙烯酸-醋酸铜SPC', 'smiles': 'OC(=O)C=C.CC(=O)O[Cu]', 'cls': 'self_polishing'},
    {'name': '丙烯酸-醋酸锌SPC', 'smiles': 'OC(=O)C=C.CC(=O)O[Zn]', 'cls': 'self_polishing'},
    {'name': 'SPC-酰胺水解型', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)NCCO', 'cls': 'self_polishing'},
    {'name': '甲基丙烯酸环己酯SPC', 'smiles': 'OC(=O)C(C)C(=O)OC1CCCCC1', 'cls': 'self_polishing'},
    {'name': 'SPC-乳酸型', 'smiles': 'OC(=O)C(C)C(=O)OCC(C)C(=O)O', 'cls': 'self_polishing'},
    {'name': '丙烯酸-丙二酸铜', 'smiles': 'OC(=O)C=C.OC(=O)CC(=O)O[Cu]', 'cls': 'self_polishing'},
    {'name': 'SPC-硫醚键型', 'smiles': 'OC(=O)C(C)C(=O)OCCSCC(=O)O', 'cls': 'self_polishing'},
    {'name': '双酯键SPC', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)OCC(=O)O', 'cls': 'self_polishing'},
    {'name': '丙烯酸-铜锌SPC', 'smiles': 'OC(=O)C=C.CC(=O)O[Cu].[Zn]', 'cls': 'self_polishing'},
    {'name': 'SPC-PEG单酯', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)OCCO', 'cls': 'self_polishing'},
    {'name': 'SPC-环己胺酯', 'smiles': 'OC(=O)C(C)C(=O)OCC(=O)NC1CCCCC1', 'cls': 'self_polishing'},
    {'name': '仿荷叶丙烯酸-PEG', 'smiles': 'OC(=O)C=C.OCCO', 'cls': 'bioinspired'},
    {'name': '仿荷叶氟碳-PEG', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F.OCCOCCO', 'cls': 'bioinspired'},
    {'name': '仿生丙烯酸-氟碳', 'smiles': 'OC(=O)C=C.FC(F)(F)C(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿贻贝多巴胺前体', 'smiles': 'OC1CC(O)C(O)C(O)C1O', 'cls': 'bioinspired'},
    {'name': '仿贻贝-丙烯酰胺', 'smiles': 'OC(=O)C=C.NCC(=O)NCCO', 'cls': 'bioinspired'},
    {'name': '仿鲨鱼皮全氟', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生硅-氟微结构', 'smiles': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生丙烯酸-甘油', 'smiles': 'OC(=O)C=C.OCC(O)CO', 'cls': 'bioinspired'},
    {'name': '仿生氟硅微纳结构', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)F.[Si](C)(C)O[Si](C)(C)C', 'cls': 'bioinspired'},
    {'name': '仿荷叶HEMA-氟碳', 'smiles': 'OC(=O)C(C)C(=O)NCCO.FC(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生丙烯酸-PDMS', 'smiles': 'OC(=O)C=C.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'bioinspired'},
    {'name': '仿贻贝-PDMS复合', 'smiles': 'OC1CC(O)C(O)C(O)C1O.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'bioinspired'},
    {'name': '仿荷叶全氟-甘油', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F.OCC(O)CO', 'cls': 'bioinspired'},
    {'name': '仿生丙烯酸-含氟酯', 'smiles': 'OC(=O)C=C.OC(=O)CCC(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生丙烯酰胺-PEG', 'smiles': 'NC(=O)C=C.OCCOCCO', 'cls': 'bioinspired'},
    {'name': '三元仿生丙烯酸-硅-氟', 'smiles': 'OC(=O)C=C.C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿贻贝-氟碳复合', 'smiles': 'OC1CC(O)C(O)C(O)C1O.FC(F)(F)C(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生HEMA-PDMS', 'smiles': 'OC(=O)C(C)C(=O)OCCO.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'bioinspired'},
    {'name': '仿荷叶氟碳-PEG400', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)F.OCCOCCOCCO', 'cls': 'bioinspired'},
    {'name': '丙烯酸-丙烯酰胺仿生', 'smiles': 'OC(=O)C=C.NC(=O)C=C', 'cls': 'bioinspired'},
    {'name': '仿贻贝-丙烯酸涂层', 'smiles': 'OC1CC(O)C(O)C(O)C1O.OC(=O)C=C', 'cls': 'bioinspired'},
    {'name': '仿生PDMS-山梨醇', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCC(O)C(O)CO', 'cls': 'bioinspired'},
    {'name': '仿生HEMA-氟碳酯', 'smiles': 'OC(=O)C(C)C(=O)OCCO.FC(F)(F)C(F)(F)F', 'cls': 'bioinspired'},
    {'name': '仿生氟碳-丙烯酰胺', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)F.NC(=O)C=C', 'cls': 'bioinspired'},
    {'name': '三元仿生丙烯酸-PEG-氟', 'smiles': 'OC(=O)C=C.OCCOCCO.FC(F)(F)F', 'cls': 'bioinspired'},
    {'name': 'PDMS/ZnO', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Zn]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/Ag', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ag]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/TiO2', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ti]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/SiO2', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Si](=O)(=O)', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/ZnO', 'smiles': 'OC(=O)C=C.[Zn]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/Ag', 'smiles': 'OC(=O)C=C.[Ag]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/TiO2', 'smiles': 'OC(=O)C=C.[Ti]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/Cu2O', 'smiles': 'OC(=O)C=C.[Cu]', 'cls': 'nanocomposite'},
    {'name': '氟碳/ZnO', 'smiles': 'FC(F)(F)C(F)(F)F.[Zn]', 'cls': 'nanocomposite'},
    {'name': '氟碳/Ag', 'smiles': 'FC(F)(F)C(F)(F)F.[Ag]', 'cls': 'nanocomposite'},
    {'name': '氟碳/TiO2', 'smiles': 'FC(F)(F)C(F)(F)F.[Ti]', 'cls': 'nanocomposite'},
    {'name': '氟碳/Cu2O', 'smiles': 'FC(F)(F)C(F)(F)F.[Cu]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/ZnO+Ag', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Zn].[Ag]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/TiO2+ZnO', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ti].[Zn]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/ZnO+Ag', 'smiles': 'OC(=O)C=C.[Zn].[Ag]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸-氟碳/ZnO', 'smiles': 'OC(=O)C=C.FC(F)(F)F.[Zn]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸-氟碳/Ag', 'smiles': 'OC(=O)C=C.FC(F)(F)F.[Ag]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/Cu2O', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Cu]', 'cls': 'nanocomposite'},
    {'name': '氟碳/ZnO+Ag', 'smiles': 'FC(F)(F)C(F)(F)F.[Zn].[Ag]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/CeO2', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ce]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸/CeO2', 'smiles': 'OC(=O)C=C.[Ce]', 'cls': 'nanocomposite'},
    {'name': 'PDMS-氟碳/ZnO', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F.[Zn]', 'cls': 'nanocomposite'},
    {'name': 'PDMS-氟碳/Ag', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F.[Ag]', 'cls': 'nanocomposite'},
    {'name': '丙烯酸-氟碳/Cu2O', 'smiles': 'OC(=O)C=C.FC(F)(F)F.[Cu]', 'cls': 'nanocomposite'},
    {'name': 'PDMS/ZnO+TiO2', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Zn].[Ti]', 'cls': 'nanocomposite'},
    {'name': 'SLIPS-PDMS+PEG润滑液', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCCOCCO', 'cls': 'smart'},
    {'name': 'SLIPS-PDMS+氟碳润滑液', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)C(F)(F)F', 'cls': 'smart'},
    {'name': '温度响应PNIPAM-丙烯酸', 'smiles': 'OC(=O)C=C.NC(=O)C(C)C', 'cls': 'smart'},
    {'name': 'PNIPAM-羟乙基', 'smiles': 'NC(=O)C(C)C(=O)NCCO', 'cls': 'smart'},
    {'name': 'pH响应丙烯酸-PEG-PDMS', 'smiles': 'OC(=O)C=C.OCCOCCO.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'smart'},
    {'name': '三重SLIPS硅氟PEG', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F.OCCOCCO', 'cls': 'smart'},
    {'name': '温度响应丙烯酸-NIPAM', 'smiles': 'OC(=O)C=C.NC(=O)C=C', 'cls': 'smart'},
    {'name': '响应型HEMA-PDMS', 'smiles': 'OC(=O)C(C)C(=O)NCCO.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'smart'},
    {'name': 'SLIPS氟碳-PEG-PDMS', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)F.OCCOCCO.C[Si](C)(C)O[Si](C)(C)C', 'cls': 'smart'},
    {'name': 'pH响应丙烯酸-氟-PEG', 'smiles': 'OC(=O)C=C.FC(F)(F)F.OCCOCCO', 'cls': 'smart'},
    {'name': 'SLIPS-PDMS+PEG400', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCCOCCOCCOCCO', 'cls': 'smart'},
    {'name': 'NIPAM-PEG共聚', 'smiles': 'NC(=O)C(C)C(=O)NCCOCCO', 'cls': 'smart'},
    {'name': '光响应丙烯酸-氟-TiO2', 'smiles': 'OC(=O)C=C.FC(F)(F)F.[Ti]', 'cls': 'smart'},
    {'name': '光响应PDMS-TiO2-氟', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.[Ti].FC(F)(F)F', 'cls': 'smart'},
    {'name': '温度响应三元共聚', 'smiles': 'OC(=O)C=C.NC(=O)C(C)C.OCCO', 'cls': 'smart'},
    {'name': 'SLIPS-PDMS+PEG600', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCCOCCOCCOCCOCCO', 'cls': 'smart'},
    {'name': '温度响应丙烯酸-PDMS-NIPAM', 'smiles': 'OC(=O)C=C.C[Si](C)(C)O[Si](C)(C)C.NC(=O)C(C)C', 'cls': 'smart'},
    {'name': 'SLIPS氟碳+PEG400', 'smiles': 'FC(F)(F)C(F)(F)C(F)(F)F.OCCOCCOCCOCCO', 'cls': 'smart'},
    {'name': '响应型丙烯酸-PEG-Ag', 'smiles': 'OC(=O)C=C.OCCOCCO.[Ag]', 'cls': 'smart'},
    {'name': 'SLIPS-PDMS-PEG-Ag', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCCOCCO.[Ag]', 'cls': 'smart'},
    {'name': '温度响应NIPAM-氟碳', 'smiles': 'NC(=O)C(C)C(=O)NCCO.FC(F)(F)F', 'cls': 'smart'},
    {'name': '四元智能响应涂层', 'smiles': 'OC(=O)C=C.OCCOCCO.FC(F)(F)F.[Zn]', 'cls': 'smart'},
    {'name': 'PDMS-NIPAM-PEG三元', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.NC(=O)C(C)C.OCCOCCO', 'cls': 'smart'},
    {'name': '丙烯酸-PDMS-氟碳响应', 'smiles': 'OC(=O)C=C.C[Si](C)(C)O[Si](C)(C)C.FC(F)(F)F', 'cls': 'smart'},
    {'name': 'SLIPS-PDMS+PEG1000', 'smiles': 'C[Si](C)(C)O[Si](C)(C)C.OCCOCCOCCOCCOCCOCCO', 'cls': 'smart'},
]


def get_material_db():
    """懒加载材料数据库（含别名展开）"""
    global _MATERIAL_DB
    if _MATERIAL_DB is not None:
        return _MATERIAL_DB
    db = list(_RAW_MATERIALS)
    existing_names = {m['name'] for m in db}
    for alias, target_name in _ALIASES.items():
        target = None
        for m in db:
            if m['name'] == target_name:
                target = m
                break
        if target and alias not in existing_names:
            db.append({
                'name': alias, 'smiles': target['smiles'],
                'cls': target['cls'], 'alias_of': target_name,
            })
            existing_names.add(alias)
    _MATERIAL_DB = db
    return _MATERIAL_DB


from search_engine import get_searcher, MaterialSearcher


def search_materials(query, max_results=20):
    db = get_material_db()
    searcher = get_searcher(db)
    results = searcher.search(query, max_results=max_results)
    return [searcher.to_choice(r) for r in results]


def search_results_md_handler(query):
    db = get_material_db()
    searcher = get_searcher(db)
    results = searcher.search(query, max_results=20)
    return searcher.format_results_md(results, query)


def resolve_input(user_input):
    """智能解析用户输入：中文名→SMILES，或直接SMILES。返回 (smiles, name) 或 (None, None)"""
    db = get_material_db()
    text = user_input.strip()
    if not text:
        return None, None
    for m in db:
        if m['name'].lower() == text.lower():
            return m['smiles'], m['name']
    q = text.lower()
    for m in db:
        if q in m['name'].lower() and len(q) >= 2:
            return m['smiles'], m['name']
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return text, text
    return None, None


def _parse_search_label(label):
    if '|' in label:
        name_part = label.split('|')[0].strip()
        smi_part = label.split('|')[1].strip().rstrip('.')
        if ' (即 ' in name_part:
            name_part = name_part.split(' (即 ')[0]
        return name_part, smi_part
    return label, None


def add_to_input(search_box_value, current_input):
    if not search_box_value or not search_box_value.strip():
        return current_input or ""
    name, smi = _parse_search_label(search_box_value)
    if smi and smi != '...':
        line = f"{name} | {smi}"
    else:
        resolved_smi, resolved_name = resolve_input(search_box_value)
        if resolved_smi:
            line = f"{resolved_name} | {resolved_smi}"
        else:
            return current_input or ""
    if current_input and current_input.strip():
        return current_input.rstrip('\n') + '\n' + line
    return line


# ============================================================
# 全局模型缓存
# ============================================================
_MODEL_CACHE = None

def get_models():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    t0 = time.time()
    pkl_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(pkl_path, 'rb') as f:
        _MODEL_CACHE = pickle.load(f)
    print(f"模型加载完成，耗时 {time.time()-t0:.1f}s")
    return _MODEL_CACHE


# ============================================================
# 分子描述符计算 — 基础28维 + 14维交互特征
# ============================================================

def compute_base_descriptors(smiles):
    """从SMILES计算28个基础分子描述符（不含交互特征）"""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
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

    reactive = sum(1 for a in mol.GetAtoms()
                   if a.GetSymbol() in ['N', 'O', 'S'] and a.GetDegree() <= 2)
    desc['CrosslinkPotential'] = min(1.0, reactive / max(desc['HeavyAtoms'], 1))

    return desc


def add_interaction_features(desc):
    """给28维基础描述符追加14个交互特征，返回完整42维字典"""
    if desc is None:
        return None
    d = dict(desc)  # copy
    d['SE_x_EModulus'] = d.get('SurfaceEnergyEstimate', 0) * d.get('ElasticModulusEstimate', 0)
    d['LogP_x_TPSA'] = d.get('LogP', 0) * d.get('TPSA', 0)
    d['F_x_Si'] = d.get('NumF', 0) * d.get('NumSi', 0)
    d['ChargeDensity_x_HLB'] = d.get('ChargeDensity', 0) * d.get('HydrophilicLipophilicBalance', 0)
    d['MW_x_LogP'] = d.get('MW', 0) * d.get('LogP', 0)
    d['HBD_x_HBA'] = d.get('HBD', 0) * d.get('HBA', 0)
    d['RotBonds_x_MW'] = d.get('RotBonds', 0) / max(d.get('MW', 1), 1)
    d['RingFrac'] = d.get('RingCount', 0) / max(d.get('HeavyAtoms', 1), 1)
    d['AromaFrac'] = d.get('AromaticRings', 0) / max(d.get('RingCount', 1), 1)
    d['FracF'] = d.get('NumF', 0) / max(d.get('HeavyAtoms', 1), 1)
    d['FracSi'] = d.get('NumSi', 0) / max(d.get('HeavyAtoms', 1), 1)
    d['PolarFrac'] = (d.get('NumN', 0) + d.get('NumO', 0)) / max(d.get('HeavyAtoms', 1), 1)
    d['SE_minus_EMod'] = d.get('SurfaceEnergyEstimate', 0) - d.get('ElasticModulusEstimate', 0) * 5
    d['Kendall_index'] = np.sqrt(max(d.get('SurfaceEnergyEstimate', 0.1), 0.1) *
                                  max(10**d.get('ElasticModulusEstimate', 0), 0.1))
    return d


def compute_descriptors(smiles):
    """从SMILES计算完整42维描述符（28基础+14交互），兼容原有调用"""
    base = compute_base_descriptors(smiles)
    return add_interaction_features(base)


# ============================================================
# 核心预测
# ============================================================

TARGET_CN = {
    'antifouling_efficiency_pct': '防污效率',
    'fouling_release_pct': '脱附率',
    'antibacterial_rate_pct': '抗菌率',
    'diatom_removal_pct': '硅藻去除率'
}

def predict_from_descriptors(desc):
    """从描述符字典直接预测（支持复合材料）"""
    if desc is None:
        return None
    # 确保交互特征已追加
    full_desc = add_interaction_features(desc)
    models = get_models()
    feature_names = models['feature_cols']
    X = np.array([[full_desc.get(f, 0.0) for f in feature_names]])
    X_scaled = models['scaler'].transform(X)
    results = {}
    for target in models['target_cols']:
        preds = {}
        for m_name, model in models['models'][target].items():
            try:
                pred = float(np.clip(model.predict(X_scaled)[0], 0, 100))
                preds[m_name] = pred
            except:
                pass
        avg = np.mean(list(preds.values())) if preds else 0
        cn_name = TARGET_CN.get(target, target)
        results[cn_name] = avg
    results['综合评分'] = (0.35*results.get('防污效率', 0) + 0.25*results.get('脱附率', 0) +
                         0.25*results.get('抗菌率', 0) + 0.15*results.get('硅藻去除率', 0))
    return results


def predict_single(smiles):
    """原有接口：从SMILES预测"""
    desc = compute_descriptors(smiles)
    if desc is None:
        return None
    results = predict_from_descriptors(desc)
    if results:
        results['SMILES'] = smiles
    return results


def predict_composite(input_text):
    """
    多格式复合材料预测。
    返回 (results_dict, parsed_info_dict, descriptors_dict) 或 (None, parsed, None)
    """
    parsed = SmartMaterialParser.parse(input_text)
    if parsed['type'] == 'unknown':
        return None, parsed, None

    # 计算基础描述符（28维，由SmartMaterialParser根据类型组合）
    base_desc = SmartMaterialParser.compute_features(parsed)
    if base_desc is None:
        return None, parsed, None

    # 追加交互特征并预测
    results = predict_from_descriptors(base_desc)
    if results:
        results['输入'] = input_text
        results['类型'] = SmartMaterialParser.TYPE_LABELS.get(parsed['type'], parsed['type'])
    return results, parsed, add_interaction_features(base_desc)


# ============================================================
# P0: 不确定性量化 + 域适用性
# 可选能力 —— 折外集成模型缺失/加载失败时自动降级返回空, 绝不打断原有预测流程。
# ============================================================

_UNCERTAINTY_ENGINE = None


def _get_uncertainty_engine():
    """惰性加载 LOGO 折外集成; 不可用时返回 None。"""
    global _UNCERTAINTY_ENGINE
    if _UNCERTAINTY_ENGINE is None:
        try:
            import uncertainty as _u
            eng = _u.LOGOEnsemble()
            eng.load()
            _UNCERTAINTY_ENGINE = eng if eng.available else False
        except Exception:
            _UNCERTAINTY_ENGINE = False
    return _UNCERTAINTY_ENGINE or None


def uncertainty_md(desc):
    """
    给定描述符字典(28维基础 或 42维含交互), 返回不确定性/适用域 Markdown 块。
    不可用时返回空字符串。
    """
    eng = _get_uncertainty_engine()
    if eng is None or not desc:
        return ""
    try:
        res = eng.predict(desc)
    except Exception:
        return ""
    if not res.get("available"):
        return ""

    lines = [
        f"\n**预测可信度** (LOGO 折外集成 · {res['n_models']} 个模型)",
        f"- 点预测 **{res['prediction']}**, 95% 置信区间 "
        f"**[{res['ci_low']}, {res['ci_high']}]** (宽度 {res['ci_width']}, 口径: {res['ci_basis']})",
        f"- 不确定度等级: **{res['uncertainty_level']}**",
    ]
    if res.get("in_domain") is False:
        lines.append(
            f"- ⚠️ **超出适用域**: 马氏距离 {res.get('distance')} > 阈值 {res.get('threshold')}"
            " — 该分子与训练集差异过大, 结果仅供参考")
    else:
        lines.append(
            f"- 适用域: ✅ 在训练特征空间内 (马氏距离 {res.get('distance')} / 阈值 {res.get('threshold')})")
    return "\n".join(lines)



def compute_material_properties(desc):
    """从分子描述符计算重要材料特性，返回 (值, 置信度, 单位) 元组"""
    props = {}
    logp = desc.get('LogP', 0)
    se = desc.get('SurfaceEnergyEstimate', 30)
    frac_f = desc.get('FracF', 0)
    frac_si = desc.get('FracSi', 0)
    ca = 60 + logp * 8 + frac_f * 80 + frac_si * 40 - se * 0.5
    ca = max(10, min(170, ca))
    conf = 'estimated' if frac_f > 0.1 or frac_si > 0.05 else 'predicted'
    props['接触角'] = (round(ca, 1), conf, '°')
    se_val = desc.get('SurfaceEnergyEstimate', None)
    props['表面能'] = (round(se_val, 1), 'estimated', 'mN/m') if se_val is not None else (None, 'unknown', 'mN/m')
    em = desc.get('ElasticModulusEstimate', None)
    if em is not None:
        props['弹性模量'] = (round(10 ** em, 2), 'estimated', 'GPa')
    else:
        props['弹性模量'] = (None, 'unknown', 'GPa')
    hlb = desc.get('HydrophilicLipophilicBalance', None)
    props['HLB值'] = (round(hlb, 1), 'estimated', '') if hlb is not None else (None, 'unknown', '')
    mw = desc.get('MW', None)
    props['分子量'] = (round(mw, 1), 'measured', 'g/mol') if mw is not None else (None, 'unknown', 'g/mol')
    tpsa = desc.get('TPSA', None)
    props['极性表面积'] = (round(tpsa, 1), 'measured', 'Å²') if tpsa is not None else (None, 'unknown', 'Å²')
    rough = desc.get('RoughnessPotential', None)
    props['粗糙度因子'] = (round(rough, 3), 'predicted', '') if rough is not None else (None, 'unknown', '')
    cross = desc.get('CrosslinkPotential', None)
    props['交联密度'] = (round(cross, 3), 'predicted', '') if cross is not None else (None, 'unknown', '')
    props['LogP'] = (round(logp, 2), 'measured', '') if logp is not None else (None, 'unknown', '')
    kendall = desc.get('Kendall_index', None)
    props['Kendall粘附参数'] = (round(kendall, 2), 'predicted', '') if kendall is not None else (None, 'unknown', '')
    return props


def plot_properties_card(properties, material_name=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(5, 9.5, f'🔬 {material_name} — 材料特性参数', ha='center', va='top',
            fontsize=14, fontweight='bold')
    conf_colors = {'measured': '#2ecc71', 'estimated': '#f39c12', 'predicted': '#3498db', 'unknown': '#95a5a6'}
    conf_labels = {'measured': '实验值', 'estimated': '估算值', 'predicted': '预测值', 'unknown': '未知'}
    y = 8.5; row_h = 0.7
    for name, (val, conf, unit) in properties.items():
        color = conf_colors.get(conf, '#95a5a6')
        ax.text(0.5, y, name, ha='left', va='center', fontsize=11, fontweight='bold')
        val_str = f'{val} {unit}' if val is not None else f'— {unit}'
        ax.text(5.5, y, val_str, ha='left', va='center', fontsize=11, color=color, fontweight='bold')
        ax.add_patch(plt.Rectangle((8.2, y-0.2), 1.5, 0.4, facecolor=color, alpha=0.2, edgecolor=color))
        ax.text(8.95, y, conf_labels.get(conf, conf), ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')
        y -= row_h
    ax.text(0.5, y-0.3, '图例:', fontsize=9, fontweight='bold')
    x_leg = 1.5
    for conf_key, label in conf_labels.items():
        c = conf_colors[conf_key]
        ax.add_patch(plt.Circle((x_leg, y-0.3), 0.12, color=c))
        ax.text(x_leg+0.3, y-0.3, label, fontsize=8, va='center')
        x_leg += 2.0
    plt.tight_layout()
    return fig


def predict_batch(smiles_list):
    results = []
    for smi in smiles_list:
        r = predict_single(smi)
        if r is not None:
            results.append(r)
    return pd.DataFrame(results) if results else pd.DataFrame()


# ============================================================
# 分类与筛选
# ============================================================

_CLASS_NAMES = {
    'silicone': '硅树脂', 'fluoropolymer': '氟聚合物', 'hydrogel': '水凝胶',
    'zwitterionic': '两性离子', 'self_polishing': '自抛光',
    'bioinspired': '仿生', 'nanocomposite': '纳米复合', 'smart': '智能响应'
}
_CLASS_COLORS = {
    '硅树脂':'#e41a1c','氟聚合物':'#377eb8','水凝胶':'#4daf4a','两性离子':'#984ea3',
    '自抛光':'#ff7f00','仿生':'#a65628','纳米复合':'#f781bf','智能响应':'#999999'
}

def classify_material(smiles):
    desc = compute_descriptors(smiles)
    if desc is None:
        return '未知', 0.0
    scores = {
        '硅树脂': desc.get('NumSi',0)*2.0 + desc.get('MW',0)*0.005 + (1 if desc.get('LogP',0)>2 else 0),
        '氟聚合物': desc.get('NumF',0)*1.5 + desc.get('MW',0)*0.003 + (1 if desc.get('LogP',0)>3 else 0),
        '水凝胶': desc.get('NumO',0)*0.8 + desc.get('HBD',0)*1.5 + desc.get('TPSA',0)*0.02 - desc.get('LogP',0)*0.5,
        '两性离子': desc.get('ChargeDensity',0)*8.0 + desc.get('NumN',0)*1.0 + desc.get('NumS',0)*1.0,
        '自抛光': desc.get('HasCu',0)*3.0 + desc.get('HasZn',0)*3.0 + desc.get('RotBonds',0)*0.3,
        '仿生': desc.get('RingCount',0)*0.5 + desc.get('HBD',0)*1.0 + desc.get('NumO',0)*0.5,
        '纳米复合': (desc.get('HasCu',0)+desc.get('HasZn',0)+desc.get('HasAg',0)+desc.get('HasTi',0))*4.0,
        '智能响应': desc.get('RotBonds',0)*0.3 + desc.get('NumN',0)*0.8 + desc.get('TPSA',0)*0.01,
    }
    best = max(scores, key=scores.get)
    return best, scores[best] / (sum(scores.values()) + 1e-6)


# ============================================================
# 可视化
# ============================================================

def plot_single(results):
    if results is None:
        return None
    props = ['防污效率','脱附率','抗菌率','硅藻去除率']
    vals = [results[p] for p in props]
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for i, (prop, val, color) in enumerate(zip(props, vals, colors)):
        ax = axes[i]
        gauge = np.linspace(0, 100, 100).reshape(1, -1)
        ax.imshow(gauge, aspect='auto', cmap='RdYlGn', extent=[0, 1, 0, 1])
        ax.axvline(val/100, color='black', linewidth=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(prop, fontsize=12, fontweight='bold')
        ax.text(0.5, -0.15, f'{val:.1f}%', ha='center', va='top', fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[4]
    gauge = np.linspace(0, 100, 100).reshape(1, -1)
    ax.imshow(gauge, aspect='auto', cmap='RdYlBu_r', extent=[0, 1, 0, 1])
    ax.axvline(results['综合评分']/100, color='black', linewidth=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('综合评分', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, f'{results["综合评分"]:.1f}', ha='center', va='top', fontsize=14, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_batch(df):
    if df.empty:
        return None
    props = ['防污效率','脱附率','抗菌率','硅藻去除率','综合评分']
    n = len(df)
    fig, axes = plt.subplots(1, 5, figsize=(20, max(4, n*0.6)))
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
    for i, (prop, color) in enumerate(zip(props, colors)):
        ax = axes[i]
        y_pos = np.arange(n)
        bars = ax.barh(y_pos, df[prop].values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(x)[:25] for x in df['材料名称'].values], fontsize=8)
        ax.set_xlabel('数值', fontsize=9)
        ax.set_title(prop, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100)
        for bar, val in zip(bars, df[prop].values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=8)
    plt.tight_layout()
    return fig


def plot_radar(results):
    if results is None:
        return None
    props = ['防污效率','脱附率','抗菌率','硅藻去除率']
    vals = [results[p] for p in props]
    angles = np.linspace(0, 2*np.pi, len(props), endpoint=False).tolist()
    vals_plot = vals + vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_plot, 'o-', linewidth=2, color='#e41a1c')
    ax.fill(angles, vals_plot, alpha=0.25, color='#e41a1c')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(props, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('性能雷达图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# ============================================================
# 筛选器
# ============================================================

def filter_materials(df, class_filter=None, min_af=0, min_fr=0, min_ab=0, min_dr=0, min_score=0):
    mask = pd.Series([True]*len(df))
    if class_filter and class_filter != '全部':
        mask &= df['分类'] == class_filter
    mask &= df['防污效率'] >= min_af
    mask &= df['脱附率'] >= min_fr
    mask &= df['抗菌率'] >= min_ab
    mask &= df['硅藻去除率'] >= min_dr
    mask &= df['综合评分'] >= min_score
    return df[mask]


# ============================================================
# 文献数据库
# ============================================================

def load_literature_db():
    pkl_path = os.path.join(os.path.dirname(__file__), 'literature_db.pkl')
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    csv_path = os.path.join(os.path.dirname(__file__), 'literature_db.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def lit_count():
    """文献库条目数（供 UI 文案动态显示，避免硬编码过期）。"""
    try:
        df = load_literature_db()
        return len(df) if df is not None and not df.empty else 0
    except Exception:
        return 0


def search_literature(query, max_results=20):
    df = load_literature_db()
    if df.empty:
        return pd.DataFrame()
    if not query or not query.strip():
        return df.head(max_results)
    q = query.lower()
    mask = pd.Series([False]*len(df))
    text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        mask |= df[col].astype(str).str.lower().str.contains(q, na=False)
    return df[mask].head(max_results)



# ============================================================
# 平台构建
# ============================================================

def create_platform():
    print("正在初始化平台 v6.0 ...")
    t0 = time.time()
    get_models()
    print(f"模型加载耗时: {time.time()-t0:.1f}s")

    # 预计算238个材料
    t1 = time.time()
    db = get_material_db()
    orig_materials = [m for m in db if not m.get('alias_of')]
    all_results = []
    for m in orig_materials:
        r = predict_single(m['smiles'])
        if r is not None:
            r['材料名称'] = m['name']
            r['分类'] = _CLASS_NAMES.get(m['cls'], m['cls'])
            all_results.append(r)
    PRECOMPUTED = pd.DataFrame(all_results)
    print(f"预计算{len(orig_materials)}个材料耗时: {time.time()-t1:.1f}s")

    class_counts = PRECOMPUTED['分类'].value_counts().to_dict()

    def quick_search(query):
        if not query or not query.strip():
            return PRECOMPUTED.head(20)
        q = query.lower()
        mask = (PRECOMPUTED['材料名称'].str.lower().str.contains(q, na=False) |
                PRECOMPUTED['分类'].str.lower().str.contains(q, na=False))
        return PRECOMPUTED[mask]

    # ── 回调: Tab 1 自定义预测（升级版，支持6种输入格式）──
    def on_predict_custom(user_input):
        if not user_input or not user_input.strip():
            return "请输入材料（支持SMILES、中文名、复合材料语法）", None, None, pd.DataFrame(), None, ""

        lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
        if len(lines) > 20:
            return "最多支持20个材料", None, None, pd.DataFrame(), None, ""

        results_list = []
        errors = []
        all_props = {}
        parsed_info_parts = []
        unc_parts = []   # P0: 不确定性/适用域 说明块

        for line in lines:
            # 支持 "名称 | SMILES" 格式（仍兼容旧输入方式）
            parts = line.split('|')
            if len(parts) == 2:
                name_hint, smi = parts[0].strip(), parts[1].strip()
                # 旧模式：直接用SMILES预测
                desc = compute_descriptors(smi)
                r = predict_single(smi)
                if r is None:
                    errors.append(f"❌ {smi}: 无法解析SMILES结构")
                    continue
                r['材料名称'] = name_hint or smi
                cls_name, _ = classify_material(smi)
                r['分类'] = cls_name
                results_list.append(r)
                if desc:
                    all_props[name_hint or smi] = compute_material_properties(desc)
                    _um = uncertainty_md(desc)
                    if _um:
                        unc_parts.append(f"**{name_hint or smi}**{_um}")
                parsed_info_parts.append(f"**{name_hint}**: 直接SMILES输入\n- SMILES: `{smi}`")
                continue

            # 新模式：尝试SmartMaterialParser多格式解析
            result, parsed, full_desc = predict_composite(line)

            if result is not None:
                # 复合材料预测成功
                name = line[:40]
                result['材料名称'] = name
                # 分类: 尝试从解析信息推断
                mat_type = parsed['type']
                if mat_type == SmartMaterialParser.TYPE_NANOCOMPOSITE:
                    result['分类'] = '纳米复合'
                elif mat_type == SmartMaterialParser.TYPE_NATURAL:
                    comp = parsed['components'][0]
                    result['分类'] = _CLASS_NAMES.get(comp.get('class', ''), '仿生')
                else:
                    # 尝试获取SMILES并分类
                    smi = None
                    if parsed['components']:
                        comp = parsed['components'][0]
                        smi = comp.get('smiles', '')
                        if not smi and comp.get('parsed', {}).get('components'):
                            smi = comp['parsed']['components'][0].get('smiles', '')
                    if smi:
                        cls_name, _ = classify_material(smi)
                        result['分类'] = cls_name
                    else:
                        result['分类'] = SmartMaterialParser.TYPE_LABELS.get(mat_type, '未知')

                results_list.append(result)
                if full_desc:
                    all_props[name] = compute_material_properties(full_desc)
                    _um = uncertainty_md(full_desc)
                    if _um:
                        unc_parts.append(f"**{name}**{_um}")
                parsed_info_parts.append(format_parsed_info(parsed))
                continue

            # 如果SmartMaterialParser也无法解析，尝试旧方式（材料名→SMILES）
            smi, resolved_name = resolve_input(line)
            if smi is not None:
                r = predict_single(smi)
                if r is not None:
                    r['材料名称'] = resolved_name or line
                    cls_name, _ = classify_material(smi)
                    r['分类'] = cls_name
                    results_list.append(r)
                    desc = compute_descriptors(smi)
                    if desc:
                        all_props[resolved_name or line] = compute_material_properties(desc)
                        _um = uncertainty_md(desc)
                        if _um:
                            unc_parts.append(f"**{resolved_name or line}**{_um}")
                    parsed_info_parts.append(
                        f"**{resolved_name or line}**: 材料数据库匹配\n- SMILES: `{smi}`")
                    continue

            errors.append(f"❌ {line}: 无法识别为材料名称、SMILES或复合材料表达式")

        if not results_list:
            err_msg = '\n'.join(errors) if errors else "无有效输入"
            return err_msg, None, None, pd.DataFrame(), None, ""

        df = pd.DataFrame(results_list)
        msg_parts = [f"✅ 成功预测 **{len(df)}** 个材料"]
        if errors:
            msg_parts.append('\n' + '\n'.join(errors))
        if unc_parts:
            msg_parts.append('\n---\n' + '\n\n'.join(unc_parts))
        msg = '\n'.join(msg_parts)

        if len(df) == 1:
            r = df.iloc[0].to_dict()
            fig1 = plot_single(r)
            fig2 = plot_radar(r)
        else:
            fig1 = plot_batch(df)
            fig2 = None

        props_fig = None
        if all_props:
            first_name = list(all_props.keys())[0]
            props_fig = plot_properties_card(all_props[first_name], first_name)

        display_cols = ['材料名称', '分类', '防污效率', '脱附率', '抗菌率', '硅藻去除率', '综合评分']
        avail_cols = [c for c in display_cols if c in df.columns]

        parsed_md = '\n\n---\n\n'.join(parsed_info_parts) if parsed_info_parts else ""

        return msg, fig1, fig2, df[avail_cols], props_fig, parsed_md

    # ── 回调: Tab 2 筛选 ──
    def on_screen(class_f, min_af, min_fr, min_ab, min_dr, min_score, sort_by, ascending):
        filtered = filter_materials(PRECOMPUTED, class_f if class_f != '全部' else None,
                                     min_af, min_fr, min_ab, min_dr, min_score)
        if filtered.empty:
            return "无匹配材料", None, pd.DataFrame()
        filtered = filtered.sort_values(sort_by, ascending=ascending)
        fig = plt.figure(figsize=(10, 6))
        top_n = filtered.head(15)
        x = np.arange(len(top_n))
        w = 0.25
        plt.bar(x - w, top_n['防污效率'], w, label='防污效率', color='#e41a1c')
        plt.bar(x, top_n['脱附率'], w, label='脱附率', color='#377eb8')
        plt.bar(x + w, top_n['抗菌率'], w, label='抗菌率', color='#4daf4a')
        plt.xticks(x, [str(n)[:12] for n in top_n['材料名称']], rotation=45, ha='right')
        plt.legend(); plt.ylabel('性能值'); plt.title(f'筛选结果 ({len(filtered)}个材料)')
        plt.tight_layout()
        display_cols = ['材料名称','分类','防污效率','脱附率','抗菌率','硅藻去除率','综合评分']
        return f"找到 {len(filtered)} 个匹配材料", fig, filtered[display_cols]

    # ── 回调: Tab 3 对比 ──
    def on_compare(batch_input):
        if not batch_input or not batch_input.strip():
            return "请输入材料", pd.DataFrame(), None
        names = [n.strip() for n in batch_input.split(',') if n.strip()]
        rows = []
        for name in names:
            match = PRECOMPUTED[PRECOMPUTED['材料名称'].str.contains(name, case=False, na=False)]
            if not match.empty:
                rows.append(match.iloc[0])
        if not rows:
            return "未找到匹配材料", pd.DataFrame(), None
        cmp_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        props = ['防污效率','脱附率','抗菌率','硅藻去除率']
        for i, (prop, ax) in enumerate(zip(props, axes.flat)):
            ax.barh([str(n)[:15] for n in cmp_df['材料名称']], cmp_df[prop].values,
                    color=['#e41a1c','#377eb8','#4daf4a','#984ea3'][i])
            ax.set_title(prop, fontweight='bold'); ax.set_xlim(0, 100)
        plt.tight_layout()
        display_cols = ['材料名称','分类','防污效率','脱附率','抗菌率','硅藻去除率','综合评分']
        return f"对比 {len(cmp_df)} 个材料", cmp_df[display_cols], fig

    # ── 回调: 文献 ──
    def on_search_lit(query, max_results):
        results = search_literature(query, int(max_results))
        if results.empty:
            return "未找到相关文献"
        md = f"### 找到 {len(results)} 篇文献\n\n"
        for _, row in results.iterrows():
            md += f"**{row.get('title','N/A')}**\n"
            md += f"- 作者: {row.get('authors','N/A')}\n"
            md += f"- 期刊: {row.get('journal','N/A')} ({row.get('year','N/A')})\n"
            if 'doi' in row and pd.notna(row['doi']):
                md += f"- DOI: [{row['doi']}](https://doi.org/{row['doi']})\n"
            md += '\n'
        return md

    # ── 回调: 合成路线 ──
    _CLASS_MAP_REV = {v: k for k, v in _CLASS_NAMES.items()}

    def on_synthesis_query(material_name, material_class):
        if not material_name or not material_name.strip():
            return "⚠️ 请输入材料名称", None, pd.DataFrame(), ""
        name = material_name.strip()
        if name in _ALIASES:
            name = _ALIASES[name]
        cls = None
        if material_class and material_class != '全部':
            cls = _CLASS_MAP_REV.get(material_class, material_class)
        if cls is None:
            match = PRECOMPUTED[PRECOMPUTED['材料名称'].str.contains(name[:4], case=False, na=False)]
            if not match.empty:
                cls_zh = match.iloc[0].get('分类', '')
                cls = _CLASS_MAP_REV.get(cls_zh, None)
        route = get_synthesis_route(name, cls)
        if route is None:
            return f"⚠️ 未找到 **{name}** 的合成路线，请尝试其他关键词或选择材料类别", None, pd.DataFrame(), ""
        report_md = format_synthesis_report(route)
        flowchart_fig = generate_synthesis_flowchart(route)
        reagent_df = generate_reagent_table(route)
        summary = f"### ✅ 找到合成路线：**{route['name']}**\n"
        summary += f"- 🏷️ 类别：{route['class']}  |  📐 方法：{route['method']}  |  "
        summary += f"⚡ 难度：{route.get('difficulty','—')}  |  📏 规模：{route.get('scalability','—')}  |  💰 成本：{route.get('cost_level','—')}\n"
        return summary, flowchart_fig, reagent_df, report_md

    def on_synthesis_browse(cls_filter):
        cls_key = _CLASS_MAP_REV.get(cls_filter, cls_filter) if cls_filter and cls_filter != '全部' else None
        routes = list_routes_by_class(cls_key) if cls_key else list_all_routes()
        if not routes:
            return "未找到合成路线", pd.DataFrame()
        rows = []
        for r in routes:
            rows.append({
                '材料名称': r['name'], '合成方法': r['method'], '类别': r['class'],
                '难度': r.get('difficulty', '—'),
                '试剂数': len(r.get('reagents', [])),
            })
        df = pd.DataFrame(rows)
        md = f"### 共收录 **{len(routes)}** 条合成路线"
        if cls_key:
            md += f"（类别: {cls_filter}）"
        return md, df

    def on_synthesis_overview():
        return get_database_summary()


    # ============================================================
    # 构建Gradio界面
    # ============================================================
    with gr.Blocks(title="海洋防污材料ML预测平台 v6.0",
                   theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green")) as app:

        gr.Markdown("""
        # 🌊 海洋防污材料ML预测平台 v6.0
        
        **智能材料解析器** | **6种输入格式** | **复合材料预测** | **42维分子描述符** | **LOGO折外R² 0.59** | **合成制备路线**
        
        基于238个原始材料 + 4762个增强样本训练的Optuna-XGBoost集成模型。v6.0新增 **智能材料解析器**，
        支持纯小分子、均聚物、共聚物、纳米复合材料、多层涂层和天然材料6种输入格式。
        """)

        with gr.Tabs():
            # ====== Tab 1: 自定义预测（升级版）======
            with gr.Tab("🔬 自定义预测"):
                gr.Markdown("### 🆕 v6.0 — 支持多格式复合材料智能解析与预测")

                with gr.Accordion("📖 输入格式说明（点击展开）", open=False):
                    gr.Markdown("""
| 类型 | 语法格式 | 示例 |
|------|---------|------|
| **纯小分子** | 直接输入SMILES | `C[Si](C)(C)O[Si](C)(C)C` |
| **均聚物** | `POLY[单体SMILES]` | `POLY[C=CC(=O)O]` |
| **共聚物** | `SMILES_A:比例 + SMILES_B:比例` | `C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3` |
| **纳米复合** | `聚合物SMILES @ 纳米粒子:wt%` | `C[Si](C)(C)O[Si](C)(C)C @ ZnO:5` |
| **多层涂层** | `材料1 / 材料2` | `PDMS / PSBMA` |
| **天然材料** | 中文名/英文名/缩写 | `壳聚糖`、`PDMS`、`chitosan` |

#### ⚠️ 注意
- 共聚物分隔符 ` + ` 两侧需要**空格**（避免与SMILES中的`[N+]`冲突）
- 纳米粒子支持: """ + ', '.join(NANOPARTICLE_DB.keys()) + """
- 天然材料支持: 壳聚糖, 海藻酸钠, 多巴胺, 漆酚, 丹宁酸, 辣椒素, 松香, PDMS, PTFE, PVA, PEG, PSBMA, PCBMA, MPC 等
- 每行一个材料，最多20个，也支持旧格式 `名称 | SMILES`
- 也可以直接搜索已有材料数据库（下方搜索框）
                    """)

                # 快捷示例按钮
                gr.Markdown("#### 💡 快速输入示例")
                with gr.Row():
                    example_btns = {}
                    examples = {
                        "PDMS小分子": "C[Si](C)(C)O[Si](C)(C)C",
                        "共聚物": "C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3",
                        "纳米复合": "C[Si](C)(C)O[Si](C)(C)C @ ZnO:5",
                        "多层涂层": "PDMS / PSBMA",
                        "天然材料": "壳聚糖",
                        "均聚物": "POLY[C=CC(=O)O]",
                    }
                    for label, val in examples.items():
                        example_btns[label] = gr.Button(label, size="sm")

                # 材料搜索
                with gr.Row():
                    search_box = gr.Dropdown(
                        label="🔍 搜索已有材料（输入中文/英文名/缩写）",
                        choices=[], allow_custom_value=True, interactive=True, scale=3
                    )
                    add_btn = gr.Button("➕ 添加到预测列表", scale=1)

                smiles_input = gr.Textbox(
                    label="材料输入（每行一个，支持6种格式）",
                    placeholder="示例：\nC[Si](C)(C)O[Si](C)(C)C\nC=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3\nPDMS / PSBMA\n壳聚糖",
                    lines=5
                )
                predict_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")

                pred_msg = gr.Markdown()
                parsed_info = gr.Markdown(label="🧩 材料解析结果")

                with gr.Row():
                    pred_plot = gr.Plot(label="性能可视化")
                    radar_plot = gr.Plot(label="雷达图")
                props_plot = gr.Plot(label="📋 材料特性参数")
                pred_table = gr.Dataframe(label="预测结果", interactive=False)

                # 绑定事件
                search_box.input(fn=search_materials, inputs=[search_box], outputs=[search_box])
                add_btn.click(fn=add_to_input, inputs=[search_box, smiles_input], outputs=[smiles_input])
                search_results_md = gr.Markdown("输入物质名称 / 别名 / SMILES / 类别，结果按相关度排序展示。")
                search_box.input(fn=search_results_md_handler, inputs=[search_box], outputs=[search_results_md])

                predict_btn.click(
                    fn=on_predict_custom,
                    inputs=[smiles_input],
                    outputs=[pred_msg, pred_plot, radar_plot, pred_table, props_plot, parsed_info]
                )

                # 快捷示例绑定
                for label, val in examples.items():
                    example_btns[label].click(
                        fn=lambda v=val: v,
                        outputs=[smiles_input]
                    )

            # ====== Tab 2: 材料筛选器 ======
            with gr.Tab("🎯 材料筛选器"):
                gr.Markdown("### 筛选238个已知材料")
                with gr.Row():
                    class_filter = gr.Dropdown(
                        label="材料分类",
                        choices=['全部'] + list(class_counts.keys()),
                        value='全部'
                    )
                    sort_by = gr.Dropdown(
                        label="排序字段",
                        choices=['综合评分','防污效率','脱附率','抗菌率','硅藻去除率'],
                        value='综合评分'
                    )
                    ascending = gr.Checkbox(label="升序", value=False)
                with gr.Row():
                    min_af = gr.Slider(0, 100, 0, 5, label="最低防污效率")
                    min_fr = gr.Slider(0, 100, 0, 5, label="最低脱附率")
                    min_ab = gr.Slider(0, 100, 0, 5, label="最低抗菌率")
                    min_dr = gr.Slider(0, 100, 0, 5, label="最低硅藻去除率")
                    min_score = gr.Slider(0, 100, 0, 5, label="最低综合评分")
                screen_btn = gr.Button("🔍 筛选", variant="primary")
                screen_msg = gr.Markdown()
                screen_plot = gr.Plot(label="筛选结果可视化")
                screen_table = gr.Dataframe(label="筛选结果", interactive=False)
                screen_btn.click(
                    fn=on_screen,
                    inputs=[class_filter, min_af, min_fr, min_ab, min_dr, min_score, sort_by, ascending],
                    outputs=[screen_msg, screen_plot, screen_table]
                )

            # ====== Tab 3: 批量对比 ======
            with gr.Tab("📊 批量对比"):
                gr.Markdown("### 输入材料名称（逗号分隔）进行对比")
                batch_in = gr.Textbox(
                    label="材料名称",
                    placeholder="PDMS三聚体, PTFE单体单元, 磺酸甜菜碱SBMA",
                    lines=2
                )
                cmp_btn = gr.Button("📊 对比分析", variant="primary")
                cmp_msg = gr.Markdown()
                cmp_table = gr.Dataframe(label="对比结果", interactive=False)
                cmp_plot = gr.Plot(label="对比图")
                cmp_btn.click(
                    fn=on_compare,
                    inputs=[batch_in],
                    outputs=[cmp_msg, cmp_table, cmp_plot]
                )

            # ====== Tab 4: 合成制备路线 ======
            with gr.Tab("🧪 合成制备路线"):
                gr.Markdown("""### 🧪 材料合成制备路线数据库
                
本模块收录 **8 大类 41 种** 防污材料的完整合成路线，包括试剂配方、反应条件、
分步操作流程、表征方法和文献出处。所有路线均源自同行评审文献，
证明平台中的材料具有真实可行的制备方案。
                """)

                with gr.Tabs():
                    with gr.Tab("🔍 查询合成路线"):
                        gr.Markdown("输入材料名称（中/英文均可），获取完整合成路线与流程图")
                        with gr.Row():
                            synth_name_input = gr.Textbox(
                                label="材料名称",
                                placeholder="例：PDMS, 壳聚糖, SBMA, 纳米氧化锌, SLIPS, PNIPAM...",
                                scale=3
                            )
                            synth_cls_filter = gr.Dropdown(
                                label="材料类别（可选，辅助匹配）",
                                choices=['全部'] + list(class_counts.keys()),
                                value='全部', scale=1
                            )
                        synth_btn = gr.Button("🧪 查询合成路线", variant="primary", size="lg")
                        synth_summary = gr.Markdown()
                        with gr.Row():
                            synth_flowchart = gr.Plot(label="🔬 合成流程图")
                        synth_reagent_table = gr.Dataframe(label="📋 试剂清单", interactive=False)
                        synth_report = gr.Markdown(label="📄 完整合成报告")
                        synth_btn.click(
                            fn=on_synthesis_query,
                            inputs=[synth_name_input, synth_cls_filter],
                            outputs=[synth_summary, synth_flowchart, synth_reagent_table, synth_report]
                        )
                        gr.Markdown("#### 💡 快捷示例")
                        with gr.Row():
                            for example_name in ["PDMS", "磺酸甜菜碱SBMA", "壳聚糖", "纳米氧化锌复合", "SLIPS仿生润滑", "PNIPAM温敏"]:
                                gr.Button(example_name, size="sm").click(
                                    fn=lambda n=example_name: on_synthesis_query(n, '全部'),
                                    outputs=[synth_summary, synth_flowchart, synth_reagent_table, synth_report]
                                )

                    with gr.Tab("📋 浏览全部路线"):
                        gr.Markdown("按类别浏览全部合成路线")
                        with gr.Row():
                            browse_cls = gr.Dropdown(
                                label="筛选类别",
                                choices=['全部'] + list(class_counts.keys()),
                                value='全部', scale=2
                            )
                            browse_btn = gr.Button("📋 显示路线列表", variant="primary", scale=1)
                        browse_msg = gr.Markdown()
                        browse_table = gr.Dataframe(label="合成路线列表", interactive=False)
                        browse_btn.click(
                            fn=on_synthesis_browse, inputs=[browse_cls],
                            outputs=[browse_msg, browse_table]
                        )

                    with gr.Tab("📊 数据库总览"):
                        gr.Markdown("合成路线数据库统计总览")
                        overview_btn = gr.Button("📊 刷新总览", variant="primary")
                        overview_table = gr.Dataframe(label="数据库总览", interactive=False)
                        overview_btn.click(fn=on_synthesis_overview, outputs=[overview_table])
                        gr.Markdown("""
                        #### 📖 数据来源说明
                        
                        | 来源类型 | 说明 |
                        |---------|------|
                        | 学术文献 | 来自 *Chem. Rev.*, *Science*, *Nature*, *Langmuir*, *ACS AMI* 等顶级期刊 |
                        | 合成方法 | 涵盖缩合聚合、自由基聚合、ATRP/RAFT、溶胶-凝胶、氧化自聚合等主流路线 |
                        | 质量保证 | 每条路线均标注原始文献DOI、反应条件和关键参数 |
                        """)

            # ====== Tab 5: 文献数据库 ======
            with gr.Tab("📚 文献数据库"):
                gr.Markdown(f"### 检索{lit_count()}篇海洋防污文献")
                with gr.Row():
                    lit_query = gr.Textbox(label="搜索关键词", placeholder="PDMS, antifouling, zwitterionic...", scale=3)
                    lit_max = gr.Slider(5, 50, 20, 5, label="最大返回数", scale=1)
                lit_btn = gr.Button("🔍 搜索文献", variant="primary")
                lit_results = gr.Markdown()
                lit_btn.click(fn=on_search_lit, inputs=[lit_query, lit_max], outputs=[lit_results])

            # ====== Tab 6: 关于 ======
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                ### 🌊 海洋防污材料ML预测平台 v6.0
                
                #### 🆕 v6.0 更新：智能材料解析器
                
                | 新功能 | 说明 |
                |--------|------|
                | 🧩 SmartMaterialParser | 自动识别6种材料输入格式 |
                | 🔗 共聚物预测 | 加权描述符 + 交互特征（微相分离、表面偏析） |
                | 🧬 纳米复合预测 | 聚合物基体 + 纳米粒子特征增强（抗菌、粗糙度、模量） |
                | 🎭 多层涂层预测 | 外层加权描述符平均（外层权重更高） |
                | 🌿 天然材料支持 | 内建壳聚糖、海藻酸钠、多巴胺等14种天然材料SMILES映射 |
                | 🏭 纳米粒子数据库 | ZnO, Cu2O, CuO, Ag, TiO2, SiO2, CeO2, GO, CNT, Cu |
                | 🧪 均聚物模式 | POLY[单体]语法，自动调整分子量和交联参数 |
                
                #### 📊 模型参数
                
                | 参数 | 值 |
                |------|-----|
                | 训练数据 | 238条原始材料 + 4762条增强数据 |
                | 特征维度 | 42个（28个基础 + 14个交互特征） |
                | 模型 | Optuna-XGBoost, LightGBM, KNN, Ridge 集成 |
                | 验证方式 | LOGO by SMILES（84分子分组，池化折外 R²；已修正随机切分泄漏） |
                | 合成路线 | 8大类 41种材料，附文献DOI |
                | 新功能 | 🆕 智能材料解析器 + 复合材料预测（v6.0） |
                
                #### 📈 模型验证指标（已按诚实口径修正）
                
                | 性能指标 | 池化折外 R² | MAE |
                |---------|-----------|-----|
                | 防污效率 | **0.592** | **4.46** |
                
                > ⚠️ **口径更正**：早期展示的「盲测 R² 0.97」来自**含信息泄漏**的随机切分验证
                > （同一分子的增强变体同时落在训练/测试集），已下线。
                > 现采用 **Leave-One-Group-Out by SMILES**（84 个分子分组，同分子变体不跨集）
                > + 池化折外 R²；随机切分相比虚高约 0.11（0.704 → 0.592）。
                > 主要瓶颈：**真实独立分子仅 84 个**，58% 为增强样本。
                > 因此每次预测均同时给出 **95% 置信区间** 与 **适用域判定**，请勿仅看点预测值。
                
                #### 🔬 使用方法
                
                1. **自定义预测**：支持6种格式输入（SMILES、共聚物、纳米复合、多层涂层、均聚物、天然材料）
                2. **材料筛选器**：按防污性能阈值筛选238种已知材料
                3. **批量对比**：多材料性能横向对比
                4. **合成制备路线**：查看41种材料的完整合成方案（试剂、条件、步骤、文献出处）
                5. **文献数据库**：检索海洋防污文献库（元数据 + 摘要 + 被引数）
                
                #### 📖 输入格式速查
                
                | 类型 | 格式 | 示例 |
                |------|------|------|
                | 纯小分子 | SMILES | `C[Si](C)(C)O[Si](C)(C)C` |
                | 均聚物 | `POLY[单体]` | `POLY[C=CC(=O)O]` |
                | 共聚物 | `A:比例 + B:比例` | `C=CC(=O)OCCO:0.7 + C=CC(=O)OC:0.3` |
                | 纳米复合 | `聚合物 @ NP:wt%` | `PDMS @ ZnO:5` |
                | 多层涂层 | `层1 / 层2` | `PDMS / PSBMA` |
                | 天然材料 | 中文/英文名 | `壳聚糖`, `PDMS`, `chitosan` |
                
                #### 🧪 合成路线覆盖
                
                | 材料类别 | 代表方法 | 路线数 |
                |---------|---------|:------:|
                | 硅树脂/硅橡胶 | 铂催化加成固化、缩合聚合 | 6 |
                | 氟聚合物 | 自由基聚合、乳液聚合 | 5 |
                | 水凝胶 | UV光引发、冻融循环 | 5 |
                | 两性离子 | RAFT/ATRP、自由基聚合 | 4 |
                | 自抛光 | 共聚酯化、锌盐交联 | 4 |
                | 仿生 | 氧化自聚合、SLIPS、光刻 | 4 |
                | 纳米复合 | 溶胶-凝胶、原位还原 | 7 |
                | 智能响应 | RAFT、共聚、溶胶-凝胶 | 6 |
                
                #### 📝 引用
                
                本平台基于机器学习方法，结合分子描述符特征工程，对海洋防污材料的四大关键性能进行预测。
                合成路线数据库的参考文献均来自同行评审期刊（Chem. Rev., Science, Nature, Langmuir, ACS AMI等）。
                """)

    return app


if __name__ == '__main__':
    import os
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    app = create_platform()
    app.launch(server_name='0.0.0.0', server_port=port, share=False)
