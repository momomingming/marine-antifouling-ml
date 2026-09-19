#!/usr/bin/env python3
"""
海洋防污材料合成制备路线模块
Synthesis / preparation route database and visualisation helpers
for the antifouling-materials ML prediction platform.

Covers 8 material classes × representative specific materials (30+).
Each entry carries: reagents, conditions, step-by-step procedure,
characterisation, safety, literature references, and metadata.
"""

from __future__ import annotations

import io
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
import pandas as pd

# ── Chinese font setup ──────────────────────────────────────────
def _setup_cjk_font():
    """Find the best available CJK font on this system."""
    import matplotlib.font_manager as fm
    candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
                  'Noto Sans CJK KR', 'WenQuanYi Micro Hei', 'SimHei']
    for name in candidates:
        try:
            fp = fm.FontProperties(family=name)
            resolved = fm.findfont(fp, fallback_to_default=False)
            if resolved and 'DejaVu' not in resolved:
                return name
        except Exception:
            continue
    for f in fm.findSystemFonts():
        if 'NotoSansCJK' in f:
            try:
                prop = fm.FontProperties(fname=f)
                return prop.get_name()
            except Exception:
                pass
    return 'DejaVu Sans'

_best_cjk = _setup_cjk_font()
rcParams['font.sans-serif'] = [_best_cjk, 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ================================================================
#  SYNTHESIS_DATABASE
# ================================================================

SYNTHESIS_DATABASE: Dict[str, dict] = {}

def _reg(entry: dict):
    """Register one synthesis entry by its name."""
    SYNTHESIS_DATABASE[entry['name']] = entry

# ────────────────────────────────────────────────────────────────
#  CLASS-LEVEL GENERIC ROUTES  (8 classes)
# ────────────────────────────────────────────────────────────────

# ---------- silicone ----------
_reg({
    'name': '硅树脂/硅橡胶 (通用)',
    'class': 'silicone',
    'method': '铂催化硅氢加成交联',
    'reagents': [
        {'name': '乙烯基封端PDMS (Base)', 'role': '主体预聚物', 'amount': '10份'},
        {'name': '含氢硅油 (Crosslinker)', 'role': '交联剂', 'amount': '1份'},
        {'name': 'Karstedt铂催化剂', 'role': '催化剂', 'amount': '10-20 ppm Pt'},
        {'name': '1-乙炔基环己醇', 'role': '抑制剂', 'amount': '0.01-0.05份'},
        {'name': '气相法SiO₂', 'role': '补强填料', 'amount': '5-30 phr'},
    ],
    'conditions': {
        'temperature': '室温混合, 80-150 °C固化',
        'time': '室温24 h或80 °C/2 h + 150 °C/1 h',
        'atmosphere': '空气或N₂',
        'solvent': '无溶剂(本体)',
        'catalyst': 'Karstedt Pt(0) 络合物',
        'other': 'Si-H/Si-Vi比 1.5-2.5:1',
    },
    'steps': [
        '称取乙烯基封端PDMS于洁净容器, 加入气相法SiO₂, 行星式搅拌脱泡10 min',
        '加入Karstedt铂催化剂(10 ppm Pt), 低速搅拌均匀, 加入抑制剂调节适用期',
        '加入含氢硅油交联剂, 快速搅拌混匀, 真空脱泡5 min',
        '将混合物浇注于预处理模具或基材上, 刮涂至目标厚度(100-500 μm)',
        '室温放置消泡30 min, 然后送入烘箱: 80 °C/2 h初步固化',
        '升温至150 °C/1 h完全交联, 自然降温脱模',
        '用FTIR确认Si-H峰(~2160 cm⁻¹)消失, 证明交联完全',
    ],
    'characterization': [
        'FTIR: Si-H (2160 cm⁻¹) 消失确认完全交联',
        '接触角测量: 水接触角 100-110°',
        'Shore A硬度 (20-60)',
        'TGA热重: 起始分解>350 °C',
        'AFM表面粗糙度 Ra < 5 nm',
    ],
    'safety': [
        'Karstedt催化剂含Pt(0)有机络合物, 避免皮肤接触',
        '含氢硅油遇酸/碱可释放H₂, 远离明火',
        '固化过程放热轻微, 厚层浇注注意散热',
        '操作区域保持通风',
    ],
    'references': [
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
        'Yilgör, E.; Yilgör, I., Prog. Polym. Sci. 2014, 39, 1165-1195. DOI: 10.1016/j.progpolymsci.2013.11.003',
        'Chaudhury, M. K.; Finlay, J. A.; et al., Biofouling 2005, 21, 41-48. DOI: 10.1080/08927010500044377',
    ],
    'difficulty': '基础',
    'scalability': '工业',
    'cost_level': '中',
})

# ---------- fluoropolymer ----------
_reg({
    'name': '氟聚合物 (通用)',
    'class': 'fluoropolymer',
    'method': '含氟丙烯酸酯自由基共聚',
    'reagents': [
        {'name': '全氟己基乙基甲基丙烯酸酯', 'role': '含氟单体', 'amount': '60 mol%'},
        {'name': '甲基丙烯酸甲酯 (MMA)', 'role': '共聚单体', 'amount': '30 mol%'},
        {'name': '甲基丙烯酸羟乙酯 (HEMA)', 'role': '功能单体', 'amount': '10 mol%'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '1 wt% (基于单体)'},
        {'name': '三氟甲苯', 'role': '溶剂', 'amount': '使固含量50%'},
    ],
    'conditions': {
        'temperature': '65-70 °C',
        'time': '12-18 h',
        'atmosphere': 'N₂保护',
        'solvent': '三氟甲苯 (BTF)',
        'catalyst': 'AIBN热引发',
        'other': '需反复冻融脱气3次',
    },
    'steps': [
        '将含氟单体、MMA、HEMA按比例加入Schlenk瓶, 加入三氟甲苯溶剂',
        '加入AIBN引发剂, 冻融脱气3次 (液氮/室温循环), 充N₂封管',
        '油浴加热至65 °C, 磁力搅拌反应12-18 h',
        '反应液冷却后, 滴入大量甲醇中沉淀纯化, 过滤收集白色沉淀',
        '重复溶解-沉淀2次, 40 °C真空干燥24 h得氟碳共聚物',
        '将共聚物溶于三氟甲苯 (5-10 wt%), 旋涂或喷涂于基材, 80 °C烘干',
        '对涂层做静态水接触角、XPS表面氟含量和FTIR结构表征',
    ],
    'characterization': [
        'FTIR: C-F伸缩 (1100-1250 cm⁻¹), C=O酯基 (1730 cm⁻¹)',
        '¹H-NMR / ¹⁹F-NMR: 确认共聚组成',
        'XPS: 表面F/C原子比',
        'GPC: Mn, Mw, PDI',
        '静态水接触角 ≥ 105°',
    ],
    'safety': [
        'AIBN为偶氮引发剂, 不可受热>80 °C, 避免摩擦撞击',
        '三氟甲苯低毒但仍需通风橱操作',
        '含氟单体成本高, 注意减少浪费',
        '液氮操作注意防冻伤',
    ],
    'references': [
        'Wang, J.; Mao, G.; et al., Macromolecules 1997, 30, 1906-1914. DOI: 10.1021/ma9607759',
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
        'Hu, Z.; Finlay, J. A.; et al., ACS Appl. Mater. Interfaces 2009, 1, 1029-1037. DOI: 10.1021/am900027r',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '高',
})

# ---------- hydrogel ----------
_reg({
    'name': '水凝胶 (通用)',
    'class': 'hydrogel',
    'method': 'UV引发自由基光聚合',
    'reagents': [
        {'name': 'PEG二丙烯酸酯 (PEG-DA, Mn 700)', 'role': '双官能交联单体', 'amount': '20 wt%水溶液'},
        {'name': '丙烯酸羟乙酯 (HEA)', 'role': '共聚亲水单体', 'amount': '10 wt%'},
        {'name': 'Irgacure 2959 (I-2959)', 'role': '光引发剂', 'amount': '0.5 wt%'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '余量至100%'},
    ],
    'conditions': {
        'temperature': '室温 (20-25 °C)',
        'time': 'UV照射 5-15 min',
        'atmosphere': 'N₂吹扫 (消除O₂阻聚)',
        'solvent': '去离子水',
        'catalyst': 'UV 365 nm光引发',
        'other': 'UV光强 10-20 mW/cm², 模具厚度1-2 mm',
    },
    'steps': [
        '配制PEG-DA水溶液 (20 wt%), 加入HEA共聚单体, 搅拌溶解',
        '避光条件下加入光引发剂I-2959 (0.5 wt%), 搅拌至完全溶解',
        '用N₂鼓泡30 min除去溶液中溶解氧',
        '将预聚液注入PDMS模具 (厚度1 mm), 盖上石英玻璃片',
        'UV灯 (365 nm, 10 mW/cm²) 照射10 min引发聚合交联',
        '脱模后将凝胶浸入去离子水中溶胀平衡48 h, 每12 h换水',
        '测量溶胀率、压缩模量、水接触角 (~20°), FTIR确认交联',
    ],
    'characterization': [
        'FTIR: C=C (1620 cm⁻¹) 消失确认聚合; C-O-C (1100 cm⁻¹) PEG特征',
        '溶胀率测量: (W_wet - W_dry)/W_dry',
        '压缩力学: 弹性模量 10-200 kPa',
        '水接触角 < 30°',
        'SEM: 冻干后观察孔结构',
    ],
    'safety': [
        '丙烯酸酯类单体具有皮肤致敏性, 戴手套操作',
        'UV灯操作佩戴UV防护眼镜',
        '光引发剂避光保存',
        'N₂使用注意通风, 防止局部缺氧',
    ],
    'references': [
        'Lin, C.-C.; Anseth, K. S., Pharm. Res. 2009, 26, 631-643. DOI: 10.1007/s11095-008-9801-2',
        'Ekblad, T. et al., Biomacromolecules 2008, 9, 2775-2783. DOI: 10.1021/bm800547s',
        'Peppas, N. A. et al., Adv. Mater. 2006, 18, 1345-1360. DOI: 10.1002/adma.200501612',
    ],
    'difficulty': '基础',
    'scalability': '实验室',
    'cost_level': '低',
})

# ---------- zwitterionic ----------
_reg({
    'name': '两性离子聚合物 (通用)',
    'class': 'zwitterionic',
    'method': 'SI-ATRP表面引发原子转移自由基聚合',
    'reagents': [
        {'name': 'SBMA (磺酸甜菜碱甲基丙烯酸酯)', 'role': '两性离子单体', 'amount': '0.5 M'},
        {'name': 'CuBr', 'role': 'ATRP催化剂', 'amount': '1 equiv.'},
        {'name': '2,2\'-联吡啶 (bpy)', 'role': '配体', 'amount': '2 equiv.'},
        {'name': '溴代异丁酸酯硅烷偶联剂', 'role': '表面引发剂', 'amount': '涂覆基材'},
        {'name': '甲醇/水 (1:1 v/v)', 'role': '溶剂', 'amount': '使单体浓度0.5 M'},
    ],
    'conditions': {
        'temperature': '室温 (25 °C)',
        'time': '6-24 h (控制膜厚)',
        'atmosphere': 'N₂保护 (严格无氧)',
        'solvent': '甲醇/水 1:1',
        'catalyst': 'CuBr / 2,2\'-联吡啶',
        'other': '表面先修饰ATRP引发剂; 膜厚可通过时间调控',
    },
    'steps': [
        '基材(玻璃/硅片)经Piranha溶液清洗, 大量去离子水冲洗, N₂吹干',
        '将基材浸入溴代异丁酸酯硅烷的甲苯溶液(1 mM)中, 室温24 h固定引发剂',
        '甲苯、乙醇依次超声清洗去除未反应硅烷, N₂吹干',
        '手套箱中: 将SBMA溶于脱气的甲醇/水(1:1), 加入CuBr和bpy配体, 搅拌至溶解呈棕色',
        '将引发剂修饰基材浸入聚合液, N₂保护下室温反应12 h',
        '取出基材, 依次用甲醇、水、EDTA溶液(除Cu)、水清洗, N₂吹干',
        'XPS确认表面S、N元素, 椭偏仪测膜厚(20-100 nm), 水接触角<15°',
    ],
    'characterization': [
        'XPS: S 2p, N 1s 峰确认两性离子涂层',
        '椭偏仪: 干膜厚度 20-100 nm',
        '水接触角 < 15° (超亲水)',
        'SPR/QCM: 蛋白质吸附 < 5 ng/cm²',
        'AFM: 表面粗糙度 Ra < 2 nm',
    ],
    'safety': [
        'Piranha溶液 (H₂SO₄/H₂O₂) 强腐蚀强氧化, 严禁接触有机物',
        'CuBr对皮肤有刺激, 在手套箱中操作',
        'EDTA清洗液含Cu离子, 按重金属废液收集',
        '甲醇有毒, 通风橱操作',
    ],
    'references': [
        'Zhang, Z. et al., Biomaterials 2009, 30, 4975-4982. DOI: 10.1016/j.biomaterials.2009.05.068',
        'Jiang, S.; Cao, Z., Adv. Mater. 2010, 22, 920-932. DOI: 10.1002/adma.200901407',
        'Yang, W. et al., Langmuir 2008, 24, 9211-9214. DOI: 10.1021/la801487f',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})

# ---------- self_polishing ----------
_reg({
    'name': '自抛光涂料 (通用)',
    'class': 'self_polishing',
    'method': '可水解丙烯酸酯共聚',
    'reagents': [
        {'name': '甲基丙烯酸三丁基锡酯 或 硅基丙烯酸酯', 'role': '可水解单体', 'amount': '40-60 mol%'},
        {'name': '甲基丙烯酸甲酯 (MMA)', 'role': '硬段共聚单体', 'amount': '30-40 mol%'},
        {'name': '丙烯酸丁酯 (BA)', 'role': '软段共聚单体', 'amount': '10-20 mol%'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '1-2 wt%'},
        {'name': '二甲苯', 'role': '溶剂', 'amount': '使固含量40-50%'},
        {'name': 'Cu₂O', 'role': '防污颜料(可选)', 'amount': '20-40 wt%'},
    ],
    'conditions': {
        'temperature': '75-80 °C',
        'time': '8-12 h',
        'atmosphere': 'N₂保护',
        'solvent': '二甲苯',
        'catalyst': 'AIBN热引发',
        'other': '共聚物Mn目标 2-5万, 可后期配漆加入Cu₂O',
    },
    'steps': [
        '在三口瓶中加入溶剂(二甲苯), N₂鼓泡30 min除氧, 升温至75 °C',
        '将可水解单体、MMA、BA和AIBN溶于二甲苯配成单体进料液',
        '用恒压滴液漏斗将单体液在3 h内匀速滴入反应瓶',
        '滴加完毕后保温反应5 h, 补加AIBN (0.5 wt%) 后再反应2 h',
        '冷却至室温, GPC检测分子量; 必要时减压浓缩至固含量50%',
        '配漆: 将共聚物树脂与Cu₂O粉末、流变助剂、溶剂高速分散研磨',
        '喷涂或刷涂于底漆处理过的基材, 室温干燥24 h + 40 °C烘烤2 h',
        'FTIR和海水浸泡实验验证自抛光性能',
    ],
    'characterization': [
        'GPC: Mn, PDI (目标Mn 2-5×10⁴, PDI < 2.0)',
        'FTIR: 酯键 (1730 cm⁻¹), 确认可水解基团',
        '海水浸泡失重法: 测量抛光速率 (μm/月)',
        '旋转圆盘: 动态海水中附着物评估',
        '接触角: 初始及浸泡后变化',
    ],
    'safety': [
        '有机锡化合物(TBT系)已被IMO禁用, 现用硅基/锌基替代',
        'Cu₂O粉末有毒, 佩戴防尘口罩和手套',
        '二甲苯易燃有毒, 通风橱操作',
        'AIBN在干燥状态下有爆炸风险, 湿润保存',
    ],
    'references': [
        'Yebra, D. M. et al., Prog. Org. Coat. 2004, 50, 75-104. DOI: 10.1016/j.porgcoat.2003.06.001',
        'Kiil, S. et al., Ind. Eng. Chem. Res. 2002, 41, 3956-3966. DOI: 10.1021/ie020021e',
        'Bressy, C. et al., J. Coat. Technol. Res. 2014, 11, 383-392. DOI: 10.1007/s11998-014-9576-9',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '中',
})

# ---------- bioinspired ----------
_reg({
    'name': '仿生防污材料 (通用)',
    'class': 'bioinspired',
    'method': '多巴胺氧化自聚合 + 功能化',
    'reagents': [
        {'name': '盐酸多巴胺', 'role': '单体', 'amount': '2 mg/mL'},
        {'name': 'Tris-HCl缓冲液 (pH 8.5, 10 mM)', 'role': '碱性缓冲体系', 'amount': '余量'},
        {'name': 'PEG-NH₂ (Mn 5000) 或mPEG-SH', 'role': '亲水修饰剂', 'amount': '1 mg/mL'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '配制缓冲液'},
    ],
    'conditions': {
        'temperature': '室温 (25 °C)',
        'time': '第一步: 12-24 h; 第二步: 4-12 h',
        'atmosphere': '空气 (需氧气氧化)',
        'solvent': 'Tris-HCl缓冲液 (pH 8.5)',
        'catalyst': 'O₂氧化自催化',
        'other': '溶液在沉积过程中逐渐变黑',
    },
    'steps': [
        '基材 (玻璃/钢片/聚合物) 用乙醇和去离子水超声清洗各10 min',
        '配制10 mM Tris-HCl缓冲液, 调pH至8.5',
        '将盐酸多巴胺溶于缓冲液 (2 mg/mL), 快速搅拌溶解',
        '将基材浸入多巴胺溶液, 室温开放体系搅拌24 h, 溶液变为深棕/黑色',
        '取出基材, 大量去离子水冲洗, N₂吹干, 获得PDA修饰表面 (~50 nm)',
        '将PDA修饰基材浸入PEG-NH₂的Tris-HCl溶液 (1 mg/mL, pH 8.5), 反应12 h',
        '去离子水充分清洗, N₂吹干, XPS验证N、S(如mPEG-SH)信号, 接触角测量',
    ],
    'characterization': [
        'XPS: N 1s (PDA), C-O/C-N变化 (PEG接枝确认)',
        'UV-Vis: PDA吸收特征 (~280 nm)',
        '椭偏仪: PDA层厚 ~50 nm, PEG层 ~5-10 nm',
        '水接触角: PDA修饰后~50°, PEG接枝后<30°',
        '蛋白质吸附测试: BSA吸附降低>80%',
    ],
    'safety': [
        '多巴胺溶液会使皮肤、衣物染色, 难以去除',
        'Tris缓冲液基本无毒, 但pH 8.5有轻微腐蚀',
        'PDA沉积过程中会黏附容器壁, 用Piranha清洗',
        '废液含有机物, 按有机废液收集',
    ],
    'references': [
        'Lee, H. et al., Science 2007, 318, 426-430. DOI: 10.1126/science.1147241',
        'Xu, L. Q. et al., Biomacromolecules 2012, 13, 2681-2688. DOI: 10.1021/bm300899j',
        'Jiang, J. et al., Langmuir 2011, 27, 14180-14187. DOI: 10.1021/la202877k',
    ],
    'difficulty': '基础',
    'scalability': '中试',
    'cost_level': '低',
})

# ---------- nanocomposite ----------
_reg({
    'name': '纳米复合防污材料 (通用)',
    'class': 'nanocomposite',
    'method': '纳米粒子原位生长/共混法',
    'reagents': [
        {'name': 'PDMS预聚物 (Sylgard 184 A)', 'role': '基体', 'amount': '10份'},
        {'name': 'Sylgard 184 固化剂', 'role': '交联剂', 'amount': '1份'},
        {'name': 'ZnO纳米粒子 (30-50 nm)', 'role': '抗菌纳米填料', 'amount': '1-5 wt%'},
        {'name': '硅烷偶联剂 (KH-570)', 'role': '纳米粒子表面改性', 'amount': '2 wt% (基于NP)'},
        {'name': '无水乙醇', 'role': '分散介质', 'amount': '适量'},
    ],
    'conditions': {
        'temperature': '60 °C表面改性; 80-150 °C固化',
        'time': '表面改性 4 h; 固化 2-4 h',
        'atmosphere': '空气',
        'solvent': '无水乙醇 (表面改性) + 无溶剂 (固化)',
        'catalyst': 'Pt催化 (PDMS固化)',
        'other': 'ZnO需先超声分散, 硅烷偶联改性后再加入PDMS',
    },
    'steps': [
        '将ZnO纳米粒子分散于无水乙醇, 超声30 min',
        '加入KH-570硅烷偶联剂 (2 wt%), 60 °C回流搅拌4 h表面改性',
        '离心收集改性ZnO, 乙醇洗涤3次, 60 °C真空干燥12 h',
        '将改性ZnO按比例加入PDMS预聚物A中, 超声+行星搅拌混合均匀',
        '加入固化剂B (A:B = 10:1), 真空脱泡10 min',
        '浇注于模具或刮涂于基材, 80 °C/2 h + 150 °C/1 h固化',
        'SEM观察ZnO分散状态, 抗菌测试 (E. coli/S. aureus), 接触角测量',
    ],
    'characterization': [
        'SEM+EDS: ZnO分散均匀性和元素分布',
        'XRD: ZnO晶体结构确认',
        'TGA: 纳米填料实际含量',
        '抗菌测试: 抑菌圈法或菌落计数法',
        '水接触角: 100-115°',
        '力学性能: 拉伸强度、断裂伸长率',
    ],
    'safety': [
        'ZnO纳米粒子吸入有害, 佩戴N95口罩操作',
        '硅烷偶联剂(KH-570)易水解, 密封保存',
        '乙醇易燃, 远离明火',
        '离心操作平衡配重, 防止甩脱',
    ],
    'references': [
        'Beigbeder, A. et al., Biofouling 2008, 24, 291-302. DOI: 10.1080/08927010802162885',
        'Al-Naamani, L. et al., Chemosphere 2017, 168, 408-417. DOI: 10.1016/j.chemosphere.2016.10.066',
        'Carl, C. et al., Biofouling 2012, 28, 175-186. DOI: 10.1080/08927014.2012.659244',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

# ---------- smart ----------
_reg({
    'name': '智能响应涂层 (通用)',
    'class': 'smart',
    'method': 'RAFT聚合制备温度响应聚合物',
    'reagents': [
        {'name': 'N-异丙基丙烯酰胺 (NIPAM)', 'role': '温度响应单体', 'amount': '2.0 g'},
        {'name': '2-(十二烷基三硫代碳酸酯基)-2-甲基丙酸 (DDMAT)', 'role': 'RAFT试剂', 'amount': '0.05 equiv.'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.2 equiv. (相对CTA)'},
        {'name': '1,4-二氧六环', 'role': '溶剂', 'amount': '使单体浓度1 M'},
        {'name': 'N,N\'-亚甲基双丙烯酰胺 (BIS)', 'role': '交联剂 (可选)', 'amount': '2-5 mol%'},
    ],
    'conditions': {
        'temperature': '70 °C',
        'time': '12-24 h',
        'atmosphere': 'N₂保护',
        'solvent': '1,4-二氧六环',
        'catalyst': 'AIBN / RAFT CTA',
        'other': 'LCST ≈ 32 °C; 可共聚亲水单体调节LCST',
    },
    'steps': [
        '将NIPAM单体用正己烷重结晶提纯, 真空干燥',
        'Schlenk瓶中加入NIPAM、DDMAT (CTA)、AIBN, 溶于脱气的二氧六环',
        '冻融脱气3次, 充N₂, 油浴加热至70 °C搅拌反应',
        '定时取样用¹H-NMR监控转化率, 反应至目标转化率(60-80%)',
        '冷却, 透析 (MWCO 3500) 除去残余单体和CTA, 冻干得PNIPAM',
        '将PNIPAM水溶液旋涂于基材, 或UV交联制备涂层',
        'DSC测LCST, DLS测温度响应粒径变化, 接触角随温度变化曲线',
    ],
    'characterization': [
        '¹H-NMR: 确认PNIPAM结构和转化率',
        'GPC: Mn和PDI (RAFT控制PDI < 1.3)',
        'DSC: LCST测定 (~32 °C)',
        'DLS: 温度响应粒径变化',
        '接触角: 25 °C 亲水 (~40°) → 40 °C 疏水 (~80°)',
    ],
    'safety': [
        '1,4-二氧六环有致癌性, 严格通风橱操作',
        'NIPAM单体有皮肤致敏和潜在毒性, 戴手套',
        'AIBN注意避免过量加热',
        'RAFT试剂含硫醇基团, 有臭味, 通风操作',
    ],
    'references': [
        'Schild, H. G., Prog. Polym. Sci. 1992, 17, 163-249. DOI: 10.1016/0079-6700(92)90023-R',
        'Moad, G. et al., Aust. J. Chem. 2005, 58, 379-410. DOI: 10.1071/CH05072',
        'Gao, C. et al., Langmuir 2013, 29, 14573-14581. DOI: 10.1021/la4028987',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})


# ────────────────────────────────────────────────────────────────
#  SPECIFIC MATERIALS  (30+ entries from _RAW_MATERIALS)
# ────────────────────────────────────────────────────────────────

# ======== silicone 类 ========

_reg({
    'name': 'PDMS三聚体',
    'class': 'silicone',
    'method': '铂催化硅氢加成 (Sylgard 184型)',
    'reagents': [
        {'name': 'Sylgard 184 A组分 (乙烯基PDMS)', 'role': '预聚物', 'amount': '10 g'},
        {'name': 'Sylgard 184 B组分 (含氢硅油+Pt)', 'role': '交联剂/催化剂', 'amount': '1 g'},
    ],
    'conditions': {
        'temperature': '室温混合; 80 °C固化',
        'time': '80 °C/2 h, 或室温/48 h',
        'atmosphere': '空气',
        'solvent': '无溶剂(本体)',
        'catalyst': 'Pt(0) (B组分中内含)',
        'other': 'A:B = 10:1 (质量比), 真空脱泡必要',
    },
    'steps': [
        '称取Sylgard 184 A组分10 g于洁净聚丙烯杯中',
        '加入B组分1 g (A:B = 10:1 w/w), 玻璃棒搅拌3 min混匀',
        '将混合物放入真空干燥箱, 抽真空至无气泡 (约15-20 min)',
        '将脱泡后混合物倒入培养皿或PTFE模具, 刮平',
        '送入80 °C烘箱, 固化2 h (或室温放置48 h)',
        '冷却后脱模, 得到透明弹性PDMS片材',
    ],
    'characterization': [
        'FTIR: Si-CH₃ (1260 cm⁻¹), Si-O-Si (1000-1100 cm⁻¹)',
        '水接触角 105-110°',
        'Shore A硬度 ~44',
        '拉伸强度 ~6 MPa, 断裂伸长率 >100%',
    ],
    'safety': [
        'A/B组分本身毒性低, 但避免皮肤长期接触',
        '未固化PDMS黏稠, 注意防止溅到衣物',
        '80 °C烘箱操作防烫',
    ],
    'references': [
        'Dow Corning, "Sylgard 184 Silicone Elastomer Kit Technical Data Sheet", 2017',
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
    ],
    'difficulty': '基础',
    'scalability': '工业',
    'cost_level': '低',
})

_reg({
    'name': 'PDMS+三氟丙基',
    'class': 'silicone',
    'method': '氟硅烷水解缩合 + 硅氢加成',
    'reagents': [
        {'name': '3,3,3-三氟丙基三甲氧基硅烷', 'role': '含氟硅烷前体', 'amount': '5 g'},
        {'name': '乙烯基封端PDMS (Mn~28000)', 'role': '主链预聚物', 'amount': '10 g'},
        {'name': '含氢硅油 (Si-H 值 0.5%)', 'role': '交联剂', 'amount': '1.5 g'},
        {'name': 'Karstedt催化剂', 'role': '铂催化剂', 'amount': '15 ppm Pt'},
        {'name': '异丙醇', 'role': '水解介质', 'amount': '20 mL'},
        {'name': '稀盐酸 (0.1 M)', 'role': '水解催化剂', 'amount': '0.5 mL'},
    ],
    'conditions': {
        'temperature': '室温水解; 120 °C固化',
        'time': '水解4 h; 固化4 h',
        'atmosphere': '空气',
        'solvent': '异丙醇 (水解步骤)',
        'catalyst': 'HCl (水解) + Karstedt Pt (交联)',
        'other': '氟含量可通过硅烷/PDMS比调控',
    },
    'steps': [
        '将三氟丙基三甲氧基硅烷溶于异丙醇, 加入稀盐酸, 室温水解4 h',
        '旋蒸除去溶剂, 得到氟硅预水解物',
        '将氟硅预水解物与乙烯基封端PDMS在高速分散机中混合15 min',
        '加入含氢硅油交联剂和Karstedt催化剂, 搅拌混匀',
        '真空脱泡10 min, 倾倒于模具或涂覆基材',
        '120 °C固化4 h, 自然降温脱模',
        '测量水接触角 (目标>110°), ¹⁹F-NMR确认氟含量',
    ],
    'characterization': [
        'FTIR: C-F (1100-1250 cm⁻¹), Si-O-Si (1000-1100 cm⁻¹)',
        '¹⁹F-NMR: CF₃基团化学位移',
        'XPS: 表面F含量',
        '水接触角 110-120°, 二碘甲烷接触角 >80°',
        '表面能 < 15 mN/m',
    ],
    'safety': [
        '三氟丙基硅烷对呼吸道有刺激, 通风橱操作',
        '盐酸具腐蚀性',
        '异丙醇易燃',
    ],
    'references': [
        'Martinelli, E. et al., Macromol. Chem. Phys. 2012, 213, 1448-1456. DOI: 10.1002/macp.201200087',
        'Uilk, J. M. et al., Macromolecules 2000, 33, 8791-8801. DOI: 10.1021/ma001013e',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': 'PDMS-g-PEG200',
    'class': 'silicone',
    'method': '硅氢加成接枝PEG大分子单体',
    'reagents': [
        {'name': '含氢PDMS (PHMS, Si-H 值 1.0%)', 'role': '主链', 'amount': '10 g'},
        {'name': '烯丙基PEG (PEG-200-allyl)', 'role': '亲水大分子单体', 'amount': '5 g'},
        {'name': 'Karstedt催化剂', 'role': '铂催化剂', 'amount': '20 ppm Pt'},
        {'name': '甲苯', 'role': '溶剂', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '80 °C',
        'time': '8 h',
        'atmosphere': 'N₂保护',
        'solvent': '甲苯',
        'catalyst': 'Karstedt Pt催化剂',
        'other': 'Si-H/C=C = 1:1 (摩尔比), 过量烯丙基PEG不利',
    },
    'steps': [
        '将含氢PDMS溶于甲苯(30 mL), N₂气氛搅拌',
        '加入Karstedt催化剂, 升温至80 °C',
        '将烯丙基PEG溶于少量甲苯, 在1 h内缓慢滴入反应瓶',
        '80 °C持续反应8 h, 用FTIR监测Si-H峰 (2160 cm⁻¹) 下降',
        '冷却, 旋蒸除甲苯, 得到PDMS-g-PEG接枝共聚物',
        '将产物溶于正己烷:乙醇=1:1中萃取除去未反应PEG',
        '真空干燥, ¹H-NMR测PEG接枝率, 接触角测量',
    ],
    'characterization': [
        'FTIR: Si-H消失 (2160 cm⁻¹), C-O-C出现 (1100 cm⁻¹)',
        '¹H-NMR: PEG -OCH₂- (3.6 ppm) / Si-CH₃ (0.1 ppm) 积分比',
        '水接触角: 70-90° (依PEG接枝率而变)',
        'GPC: 分子量分布',
        '蛋白质吸附: 比纯PDMS降低 50-70%',
    ],
    'safety': [
        '甲苯有毒易燃, 通风橱操作',
        'Karstedt催化剂含铂, 避免接触皮肤',
        '烯丙基PEG低毒但避免吸入',
    ],
    'references': [
        'Rufin, M. A. et al., Acta Biomater. 2016, 41, 247-252. DOI: 10.1016/j.actbio.2016.04.048',
        'Martinelli, E. et al., J. Polym. Sci. A Polym. Chem. 2015, 53, 1213-1225. DOI: 10.1002/pola.27556',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': '硅氧烷-聚氨酯',
    'class': 'silicone',
    'method': '异氰酸酯-羟基缩聚',
    'reagents': [
        {'name': '异佛尔酮二异氰酸酯 (IPDI)', 'role': '硬段', 'amount': '3.5 g'},
        {'name': '羟基封端PDMS (Mn~1000)', 'role': '软段/硅氧烷', 'amount': '10 g'},
        {'name': '1,4-丁二醇 (BDO)', 'role': '扩链剂', 'amount': '0.9 g'},
        {'name': '二月桂酸二丁基锡 (DBTDL)', 'role': '催化剂', 'amount': '0.05 wt%'},
        {'name': 'N,N-二甲基甲酰胺 (DMF)', 'role': '溶剂', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '60-80 °C',
        'time': '6-10 h',
        'atmosphere': 'N₂保护',
        'solvent': 'DMF',
        'catalyst': 'DBTDL',
        'other': 'NCO/OH = 1.05-1.1:1 (略过量NCO)',
    },
    'steps': [
        '羟基封端PDMS在80 °C真空干燥4 h除水',
        '在N₂保护三口瓶中, 将PDMS溶于DMF, 升温至60 °C',
        '缓慢加入IPDI, 加入DBTDL催化剂, 60 °C反应3 h形成预聚体',
        '用FT-IR监测NCO峰 (2270 cm⁻¹) 变化',
        '降温至40 °C, 加入BDO扩链, 再升温至80 °C反应4 h',
        '反应结束(NCO峰消失), 降温, 倒入去离子水中沉淀, 过滤洗涤干燥',
        '将SiPU溶于THF (15 wt%), 流延成膜或喷涂于底漆基材',
    ],
    'characterization': [
        'FTIR: 氨基甲酸酯C=O (1700-1720 cm⁻¹), N-H (3300 cm⁻¹), 无残余NCO',
        'GPC: Mn 5-10 万',
        'DSC: 硅氧烷软段Tg ~-120 °C, 硬段Tm',
        '水接触角 95-105°',
        '拉伸强度 15-30 MPa',
    ],
    'safety': [
        'IPDI异氰酸酯对呼吸道高度敏感, 严格通风橱/手套箱操作',
        'DBTDL有机锡有毒, 微量使用',
        'DMF有生殖毒性, 佩戴手套和防毒面具',
    ],
    'references': [
        'Bodkhe, R. B. et al., Prog. Org. Coat. 2012, 75, 38-48. DOI: 10.1016/j.porgcoat.2012.03.006',
        'Sommer, S. et al., Biofouling 2010, 26, 961-972. DOI: 10.1080/08927014.2010.531272',
    ],
    'difficulty': '高级',
    'scalability': '中试',
    'cost_level': '中',
})

# ======== fluoropolymer 类 ========

_reg({
    'name': 'PTFE单体单元',
    'class': 'fluoropolymer',
    'method': 'PTFE分散液涂覆 + 烧结',
    'reagents': [
        {'name': 'PTFE水性分散液 (60 wt%, ~0.2 μm粒径)', 'role': '含氟聚合物', 'amount': '50 g'},
        {'name': 'PAI底漆', 'role': '底漆(增强附着力)', 'amount': '适量'},
        {'name': '表面活性剂 (Triton X-100)', 'role': '分散稳定', 'amount': '0.5 wt%'},
    ],
    'conditions': {
        'temperature': '干燥 100 °C; 烧结 380 °C',
        'time': '干燥30 min; 烧结15-20 min',
        'atmosphere': '空气 (低温) → N₂ (高温)',
        'solvent': '水 (分散介质)',
        'catalyst': '无',
        'other': '基材需喷砂粗化 + 底漆处理',
    },
    'steps': [
        '金属基材 (不锈钢/铝) 喷砂粗化至Ra 3-5 μm, 丙酮超声脱脂',
        '喷涂PAI底漆 (~10 μm), 80 °C干燥15 min, 250 °C烘烤10 min',
        '将PTFE分散液稀释至40 wt%, 加入表面活性剂搅拌均匀',
        '喷枪喷涂PTFE分散液于底漆面 (每层10-15 μm), 可喷2-3层',
        '每层喷涂后100 °C干燥30 min蒸发水分',
        '最终380 °C (N₂气氛) 烧结15-20 min, PTFE颗粒熔融成连续膜',
        '自然冷却, 测量膜厚 (30-50 μm)、接触角 (>110°) 和附着力',
    ],
    'characterization': [
        'FTIR-ATR: C-F₂伸缩 (1150, 1210 cm⁻¹)',
        '水接触角 110-120°',
        '铅笔硬度 H-2H',
        '附着力: 百格法 ≥ 4B',
        'SEM: 膜层致密性',
    ],
    'safety': [
        '380 °C烧结温度, PTFE超过400 °C会分解释放有毒气体(PFIB), 严控温度',
        '喷涂操作佩戴防毒面具, 避免吸入PTFE粉尘',
        '高温操作佩戴隔热手套',
    ],
    'references': [
        'Drobny, J. G., "Technology of Fluoropolymers", 2nd Ed., CRC Press, 2009',
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '中',
})

_reg({
    'name': '含氟丙烯酸酯单体',
    'class': 'fluoropolymer',
    'method': '自由基溶液聚合',
    'reagents': [
        {'name': '2,2,2-三氟乙基甲基丙烯酸酯', 'role': '含氟单体', 'amount': '8 g'},
        {'name': '甲基丙烯酸甲酯 (MMA)', 'role': '共聚单体', 'amount': '4 g'},
        {'name': '甲基丙烯酸丁酯 (BMA)', 'role': '内增塑单体', 'amount': '2 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.14 g (1 wt%)'},
        {'name': '三氟甲苯 (BTF)', 'role': '溶剂', 'amount': '14 mL'},
    ],
    'conditions': {
        'temperature': '65 °C',
        'time': '15 h',
        'atmosphere': 'N₂保护',
        'solvent': '三氟甲苯',
        'catalyst': 'AIBN热引发',
        'other': '冻融脱气; 含氟溶剂利于氟单体溶解',
    },
    'steps': [
        '将三种单体和AIBN加入干燥Schlenk瓶, 加入BTF溶剂',
        '冻融脱气3次 (液氮冻结→真空→N₂解冻), 确保无氧',
        'N₂保护下, 油浴加热至65 °C, 磁力搅拌反应15 h',
        '反应液冷却, 缓慢滴入10倍体积甲醇中沉淀聚合物',
        '过滤, 再溶于BTF, 重新沉淀于甲醇中 (重复2次)',
        '40 °C真空干燥24 h, 得到白色粉末状氟丙烯酸酯共聚物',
        '配制BTF溶液 (8 wt%), 旋涂于硅片, 60 °C退火12 h',
    ],
    'characterization': [
        'FTIR: C-F (1100-1250 cm⁻¹), C=O (1730 cm⁻¹)',
        '¹H-NMR: 各单体比例 (CF₃CH₂-: 4.3 ppm)',
        '¹⁹F-NMR: CF₃ (-75 ppm)',
        'GPC: Mn ~3×10⁴, PDI ~1.6',
        '水接触角 105-115°',
    ],
    'safety': [
        'BTF低毒但仍需通风操作',
        'AIBN不可受撞击, 冰箱保存',
        '液氮操作注意低温防护',
    ],
    'references': [
        'Wang, J.; Mao, G.; et al., Macromolecules 1997, 30, 1906-1914. DOI: 10.1021/ma9607759',
        'Hu, Z.; Finlay, J. A.; et al., ACS Appl. Mater. Interfaces 2009, 1, 1029-1037. DOI: 10.1021/am900027r',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '高',
})

_reg({
    'name': '含氟甲基丙烯酸酯',
    'class': 'fluoropolymer',
    'method': '乳液聚合',
    'reagents': [
        {'name': '全氟己基乙基甲基丙烯酸酯 (C6FMA)', 'role': '含氟单体', 'amount': '10 g'},
        {'name': 'MMA', 'role': '共聚单体', 'amount': '5 g'},
        {'name': '十二烷基硫酸钠 (SDS)', 'role': '乳化剂', 'amount': '0.3 g'},
        {'name': '过硫酸钾 (KPS)', 'role': '水溶性引发剂', 'amount': '0.15 g'},
        {'name': '去离子水', 'role': '连续相', 'amount': '80 mL'},
    ],
    'conditions': {
        'temperature': '70 °C',
        'time': '8 h',
        'atmosphere': 'N₂保护',
        'solvent': '水 (乳液体系)',
        'catalyst': 'KPS热分解引发',
        'other': '搅拌速率800-1000 rpm保持乳液稳定',
    },
    'steps': [
        '去离子水80 mL中溶解SDS (0.3 g), 搅拌至起泡',
        '加入MMA, 高速搅拌乳化; 再加入C6FMA, 继续乳化15 min',
        'N₂鼓泡30 min除氧, 升温至70 °C',
        'KPS溶于5 mL水, 注入反应体系引发聚合',
        '70 °C / 800 rpm搅拌反应8 h, 乳液逐渐变白',
        '冷却, 破乳: 加入CaCl₂或冻融, 过滤收集聚合物, 水洗3次',
        '60 °C真空干燥24 h, 研磨过筛得产物',
    ],
    'characterization': [
        'FTIR: C-F (1150 cm⁻¹), C=O (1730 cm⁻¹)',
        'DLS: 乳液粒径 100-300 nm',
        'TGA: 热分解温度 >300 °C',
        'DSC: Tg ~50-70 °C',
        '涂膜水接触角 >105°',
    ],
    'safety': [
        'C6FMA成本高, 计量准确减少浪费',
        'KPS强氧化剂, 远离有机物和还原剂',
        'SDS对皮肤有刺激',
    ],
    'references': [
        'Tsibouklis, J. et al., Biomaterials 1999, 20, 1229-1235. DOI: 10.1016/S0142-9612(99)00023-4',
        'Schmidt, D. L. et al., Langmuir 1994, 10, 2442-2447. DOI: 10.1021/la00019a066',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '高',
})


# ======== hydrogel 类 ========

_reg({
    'name': 'PEG200',
    'class': 'hydrogel',
    'method': 'PEG-DA紫外光交联水凝胶',
    'reagents': [
        {'name': 'PEG二丙烯酸酯 (PEG-DA Mn 200)', 'role': '交联单体', 'amount': '2 g'},
        {'name': '2-羟基-4\'-(2-羟乙氧基)-2-甲基苯丙酮 (Irgacure 2959)', 'role': '光引发剂', 'amount': '20 mg (1 wt%)'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '8 mL'},
    ],
    'conditions': {
        'temperature': '室温',
        'time': 'UV照射10 min',
        'atmosphere': 'N₂吹扫除O₂',
        'solvent': '去离子水',
        'catalyst': 'UV 365 nm光引发',
        'other': '高交联密度(低Mn PEG-DA), 凝胶较硬',
    },
    'steps': [
        '称取PEG-DA (Mn 200) 2 g, 溶于8 mL去离子水 (20 wt%)',
        '避光称取Irgacure 2959, 加入PEG-DA溶液, 涡旋溶解',
        'N₂鼓泡溶液15 min, 除去溶解氧',
        '注入PDMS模具 (1 mm厚), 覆盖石英玻璃',
        'UV灯照射 (365 nm, 10 mW/cm²) 10 min',
        '脱模, 浸泡于过量去离子水48 h溶胀平衡, 每12 h换水',
    ],
    'characterization': [
        'FTIR: C=C (810 cm⁻¹) 消失, C-O-C (1100 cm⁻¹) 保留',
        '溶胀率 ~200-400%',
        '压缩模量 ~100-500 kPa',
        '水接触角 < 25°',
    ],
    'safety': [
        '丙烯酸酯致敏, 戴手套',
        'UV灯佩戴防护眼镜',
        '光引发剂避光储存',
    ],
    'references': [
        'Lin, C.-C.; Anseth, K. S., Pharm. Res. 2009, 26, 631-643. DOI: 10.1007/s11095-008-9801-2',
        'Cruise, G. M. et al., Biomaterials 1998, 19, 1287-1294. DOI: 10.1016/S0142-9612(98)00025-8',
    ],
    'difficulty': '基础',
    'scalability': '实验室',
    'cost_level': '低',
})

_reg({
    'name': 'HEMA单体',
    'class': 'hydrogel',
    'method': 'AIBN引发本体/溶液聚合',
    'reagents': [
        {'name': '甲基丙烯酸羟乙酯 (HEMA)', 'role': '主单体', 'amount': '5 g'},
        {'name': '乙二醇二甲基丙烯酸酯 (EGDMA)', 'role': '交联剂', 'amount': '0.1 g (2 wt%)'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.05 g (1 wt%)'},
        {'name': '去离子水', 'role': '致孔剂/溶剂', 'amount': '5 mL'},
    ],
    'conditions': {
        'temperature': '60 °C',
        'time': '12 h',
        'atmosphere': 'N₂',
        'solvent': '水 (40-50 vol%)',
        'catalyst': 'AIBN热引发',
        'other': '水含量影响孔径和溶胀率',
    },
    'steps': [
        '将HEMA、EGDMA、AIBN溶解于去离子水中, 混合均匀',
        'N₂鼓泡15 min除氧',
        '注入密封玻璃模具 (两片玻璃+PTFE垫片)',
        '60 °C水浴加热12 h引发聚合',
        '90 °C后处理2 h确保完全反应',
        '脱模, 在去离子水中浸泡72 h (每24 h换水), 除去残余单体',
        '测量溶胀比、透光率和接触角',
    ],
    'characterization': [
        'FTIR: -OH (3400 cm⁻¹), C=O (1720 cm⁻¹)',
        '溶胀率 40-70%',
        '透光率 >90% (含水态)',
        '水接触角 ~50-60°',
        '压缩模量 0.5-2 MPa',
    ],
    'safety': [
        'HEMA有皮肤致敏性, 戴丁腈手套',
        'AIBN避免加热至>80 °C',
        '残余单体需充分浸泡除去(生物应用)',
    ],
    'references': [
        'Wichterle, O.; Lim, D., Nature 1960, 185, 117-118. DOI: 10.1038/185117a0',
        'Peppas, N. A. et al., Adv. Mater. 2006, 18, 1345-1360. DOI: 10.1002/adma.200501612',
    ],
    'difficulty': '基础',
    'scalability': '工业',
    'cost_level': '低',
})

_reg({
    'name': 'PVA单体单元',
    'class': 'hydrogel',
    'method': '冻融循环物理交联',
    'reagents': [
        {'name': '聚乙烯醇 (PVA, Mw 89000-98000, 99+%水解)', 'role': '凝胶基体', 'amount': '5 g'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '45 mL'},
    ],
    'conditions': {
        'temperature': '-20 °C冷冻 / 25 °C融化',
        'time': '每周期: 冻16 h / 融8 h, 3-5个周期',
        'atmosphere': '空气',
        'solvent': '去离子水',
        'catalyst': '无 (物理交联)',
        'other': 'PVA浓度和冻融次数控制力学性能',
    },
    'steps': [
        'PVA粉末加入去离子水, 90 °C水浴搅拌溶解4 h (10 wt% PVA)',
        '冷却至室温, 脱泡 (静置或离心)',
        '将PVA溶液注入模具',
        '置于-20 °C冰箱冷冻16 h',
        '取出, 室温融化8 h, 完成第1个冻融循环',
        '重复冻融3-5次 (次数越多, 结晶度越高, 凝胶越硬)',
        '最终得到白色半透明PVA水凝胶, 测量力学性能',
    ],
    'characterization': [
        'XRD: PVA结晶峰 (2θ ~19.5°)',
        '拉伸: 弹性模量随冻融次数增大',
        '溶胀率 300-800%',
        '水接触角 ~30-40°',
        'DSC: 结晶度',
    ],
    'safety': [
        'PVA基本无毒',
        '90 °C热溶解注意防烫',
        '-20 °C冷冻操作注意防冻伤',
    ],
    'references': [
        'Hassan, C. M.; Peppas, N. A., Adv. Polym. Sci. 2000, 153, 37-65. DOI: 10.1007/3-540-46414-X_2',
        'Stauffer, S. R.; Peppas, N. A., Polymer 1992, 33, 3932-3936. DOI: 10.1016/0032-3861(92)90385-A',
    ],
    'difficulty': '基础',
    'scalability': '工业',
    'cost_level': '低',
})

_reg({
    'name': '丙烯酰胺单体',
    'class': 'hydrogel',
    'method': '过硫酸铵/TEMED引发自由基聚合',
    'reagents': [
        {'name': '丙烯酰胺 (AAm)', 'role': '主单体', 'amount': '3 g'},
        {'name': 'N,N\'-亚甲基双丙烯酰胺 (BIS)', 'role': '交联剂', 'amount': '0.03 g (1 mol%)'},
        {'name': '过硫酸铵 (APS)', 'role': '引发剂', 'amount': '0.03 g'},
        {'name': 'TEMED', 'role': '加速剂', 'amount': '15 μL'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '10 mL'},
    ],
    'conditions': {
        'temperature': '室温 (APS/TEMED体系)',
        'time': '1-2 h凝胶化',
        'atmosphere': 'N₂吹扫',
        'solvent': '去离子水',
        'catalyst': 'APS/TEMED氧化还原引发',
        'other': '经典PAGE凝胶化学',
    },
    'steps': [
        '将AAm和BIS溶于去离子水 (30 wt% 单体), 搅拌溶解',
        'N₂鼓泡15 min除氧',
        '加入APS水溶液 (10 wt%, 0.3 mL), 搅匀',
        '加入TEMED 15 μL, 快速搅匀后迅速注入模具',
        '室温静置1-2 h完成凝胶化',
        '脱模后浸泡于去离子水48 h除去残余单体',
    ],
    'characterization': [
        'FTIR: 酰胺 I (1650 cm⁻¹), 酰胺 II (1610 cm⁻¹)',
        '溶胀率 800-2000% (高含水)',
        '压缩应力 5-50 kPa',
        '水接触角 < 20°',
    ],
    'safety': [
        '丙烯酰胺为神经毒素和潜在致癌物, 严格戴手套通风操作',
        'APS强氧化剂',
        'TEMED有刺激气味, 通风操作',
    ],
    'references': [
        'Sun, J.-Y. et al., Nature 2012, 489, 133-136. DOI: 10.1038/nature11409',
        'Calvert, P., Adv. Mater. 2009, 21, 743-756. DOI: 10.1002/adma.200800534',
    ],
    'difficulty': '基础',
    'scalability': '实验室',
    'cost_level': '低',
})

# ======== zwitterionic 类 ========

_reg({
    'name': '磺酸甜菜碱SBMA',
    'class': 'zwitterionic',
    'method': 'ATRP表面接枝聚合',
    'reagents': [
        {'name': 'SBMA (N-(3-磺丙基)-N-甲基丙烯酰氧乙基-N,N-二甲基铵内盐)', 'role': '两性离子单体', 'amount': '2.8 g (10 mmol)'},
        {'name': 'CuBr', 'role': 'ATRP催化剂', 'amount': '14 mg (0.1 mmol)'},
        {'name': 'CuBr₂', 'role': '钝化剂', 'amount': '2.2 mg (0.01 mmol)'},
        {'name': '2,2\'-联吡啶 (bpy)', 'role': '配体', 'amount': '34 mg (0.22 mmol)'},
        {'name': 'ω-巯基十一烷基溴代异丁酸酯 (Au引发剂层)', 'role': '表面ATRP引发剂', 'amount': '1 mM乙醇溶液'},
        {'name': '甲醇/水 (4:1)', 'role': '混合溶剂', 'amount': '10 mL'},
    ],
    'conditions': {
        'temperature': '室温 (25 °C)',
        'time': '1-24 h (控制厚度)',
        'atmosphere': 'N₂ (手套箱)',
        'solvent': '甲醇/水 4:1',
        'catalyst': 'CuBr/bpy',
        'other': '膜厚与时间近线性; CuBr₂抑制自由基终止',
    },
    'steps': [
        '金基材经Piranha清洗, 大量水冲洗, N₂吹干',
        '浸入引发剂硫醇乙醇溶液 (1 mM) 24 h形成自组装单层(SAM)',
        '乙醇/水清洗去除物理吸附, N₂吹干',
        '手套箱中: SBMA溶于脱气的甲醇/水, 加入CuBr、CuBr₂和bpy',
        '将SAM修饰基材浸入, N₂保护下聚合6-12 h',
        '取出, 甲醇洗、Na₂EDTA溶液 (50 mM) 清洗30 min除Cu',
        '大量水洗, N₂吹干, XPS和椭偏表征',
    ],
    'characterization': [
        'XPS: S 2p (168 eV, -SO₃⁻), N 1s (402 eV, -N⁺-)',
        '椭偏仪: 膜厚 10-80 nm',
        '水接触角 < 10°',
        'SPR: 纤维蛋白原吸附 < 2 ng/cm²',
    ],
    'safety': [
        'Piranha溶液强腐蚀, 戴护目镜和耐酸手套',
        'CuBr在空气中氧化, 手套箱操作',
        'EDTA废液含Cu, 按重金属废液收集',
    ],
    'references': [
        'Zhang, Z. et al., Biomaterials 2009, 30, 4975-4982. DOI: 10.1016/j.biomaterials.2009.05.068',
        'Yang, W. et al., Langmuir 2008, 24, 9211-9214. DOI: 10.1021/la801487f',
        'Jiang, S.; Cao, Z., Adv. Mater. 2010, 22, 920-932. DOI: 10.1002/adma.200901407',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})

_reg({
    'name': '羧酸甜菜碱CBMA',
    'class': 'zwitterionic',
    'method': 'SI-ATRP制备pCBMA刷',
    'reagents': [
        {'name': 'CBMA (羧酸甜菜碱甲基丙烯酸酯)', 'role': '两性离子单体', 'amount': '2.3 g (10 mmol)'},
        {'name': 'CuBr', 'role': '催化剂', 'amount': '14 mg'},
        {'name': 'bpy', 'role': '配体', 'amount': '34 mg'},
        {'name': '(3-(2-溴异丁酰胺基)丙基)三乙氧基硅烷', 'role': '硅烷引发剂', 'amount': '1 mM甲苯溶液'},
        {'name': '水', 'role': '溶剂', 'amount': '10 mL'},
    ],
    'conditions': {
        'temperature': '室温',
        'time': '0.5-6 h',
        'atmosphere': 'N₂',
        'solvent': '水 (CBMA水溶性好)',
        'catalyst': 'CuBr/bpy',
        'other': 'pCBMA可后续EDC/NHS偶联功能化',
    },
    'steps': [
        'SiO₂/玻璃基材经O₂等离子体处理5 min活化表面羟基',
        '浸入硅烷引发剂的甲苯溶液 (1 mM), 室温12 h, 再100 °C退火1 h',
        '甲苯/乙醇/水依次超声清洗, N₂吹干',
        '手套箱: CBMA溶于脱气水 (1 M), 加CuBr和bpy搅拌溶解',
        '引发剂修饰基材浸入, N₂下聚合2-4 h',
        '取出, EDTA溶液清洗30 min, 去离子水洗涤',
        'XPS确认羧基 (289 eV) 和季铵 (402 eV) 信号',
    ],
    'characterization': [
        'XPS: C 1s中COO⁻ (289 eV), N 1s (402 eV)',
        '椭偏: 膜厚 5-50 nm',
        '水接触角 < 12°',
        'BSA/纤维蛋白原吸附 < 5 ng/cm²',
    ],
    'safety': [
        '等离子体设备操作注意真空安全',
        '硅烷引发剂对水敏感, 干燥保存',
        'CuBr有毒, 手套箱内操作',
    ],
    'references': [
        'Zhang, Z. et al., J. Phys. Chem. B 2006, 110, 10799-10804. DOI: 10.1021/jp057266i',
        'Vaisocherová, H. et al., Anal. Chem. 2008, 80, 7894-7901. DOI: 10.1021/ac8015888',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})

_reg({
    'name': '磷酸胆碱PCBMA',
    'class': 'zwitterionic',
    'method': '自由基聚合制备MPC聚合物',
    'reagents': [
        {'name': '2-甲基丙烯酰氧乙基磷酸胆碱 (MPC)', 'role': '两性离子单体', 'amount': '3 g'},
        {'name': '甲基丙烯酸丁酯 (BMA)', 'role': '疏水共聚单体', 'amount': '1.5 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.05 g'},
        {'name': '乙醇', 'role': '溶剂', 'amount': '15 mL'},
    ],
    'conditions': {
        'temperature': '60 °C',
        'time': '18 h',
        'atmosphere': 'N₂',
        'solvent': '乙醇',
        'catalyst': 'AIBN热引发',
        'other': 'MPC:BMA比例决定亲疏水平衡',
    },
    'steps': [
        '将MPC和BMA溶于乙醇, 加入AIBN',
        'N₂鼓泡20 min除氧',
        '油浴加热至60 °C, 反应18 h',
        '反应液浓缩后滴入大量乙醚中沉淀',
        '过滤, 真空干燥24 h得到白色粉末 (PMB共聚物)',
        '溶于乙醇 (5 wt%), 浸涂法涂覆基材, 室温干燥',
    ],
    'characterization': [
        'FTIR: P=O (1230 cm⁻¹), P-O-C (1070 cm⁻¹), C=O (1720 cm⁻¹)',
        '¹H-NMR: -N⁺(CH₃)₃ (3.2 ppm), -OCH₂- (4.0 ppm)',
        'GPC: Mn ~5×10⁴',
        '水接触角 < 20° (MPC含量>30 mol%)',
        '蛋白质吸附降低>90%',
    ],
    'safety': [
        'MPC单体价格昂贵, 称量准确',
        '乙醚高度易燃, 远离明火',
        'AIBN常规安全操作',
    ],
    'references': [
        'Ishihara, K. et al., J. Biomed. Mater. Res. 1998, 39, 323-330. DOI: 10.1002/(SICI)1097-4636(199802)39:2<323::AID-JBM21>3.0.CO;2-C',
        'Lewis, A. L., Colloids Surf. B 2000, 18, 261-275. DOI: 10.1016/S0927-7765(99)00152-6',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '高',
})


# ======== self_polishing 类 ========

_reg({
    'name': '丙烯酸铜聚合物',
    'class': 'self_polishing',
    'method': '铜盐交换法制备铜丙烯酸酯树脂',
    'reagents': [
        {'name': '甲基丙烯酸 (MAA)', 'role': '功能单体', 'amount': '5 g'},
        {'name': 'MMA', 'role': '硬段单体', 'amount': '10 g'},
        {'name': '丙烯酸丁酯 (BA)', 'role': '软段单体', 'amount': '3 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.18 g'},
        {'name': '碱式碳酸铜 Cu₂(OH)₂CO₃', 'role': '铜源', 'amount': '按-COOH当量'},
        {'name': '二甲苯', 'role': '溶剂', 'amount': '20 mL'},
    ],
    'conditions': {
        'temperature': '共聚 75 °C; 铜交换 80 °C',
        'time': '共聚 10 h; 铜交换 4 h',
        'atmosphere': 'N₂ (共聚); 空气 (铜交换)',
        'solvent': '二甲苯',
        'catalyst': 'AIBN',
        'other': 'MAA含量决定自抛光速率',
    },
    'steps': [
        '将MMA、BA和MAA混合, 加入二甲苯和AIBN, N₂保护',
        '75 °C滴加法聚合10 h, 得到含-COOH的丙烯酸共聚物',
        '冷却, 取样测GPC确认Mn 2-4万',
        '加入碱式碳酸铜 (按-COOH摩尔数计算), 80 °C搅拌4 h铜交换',
        '过滤去除未反应碳酸铜, 得到蓝绿色铜丙烯酸酯树脂溶液',
        '与Cu₂O防污颜料、填料和溶剂高速分散配漆',
        '喷涂于环氧底漆基材, 室温干燥 + 40 °C烘烤',
    ],
    'characterization': [
        'FTIR: -COO-Cu (1540-1560 cm⁻¹), -COOH消失 (1700 cm⁻¹)',
        'ICP-OES: Cu含量',
        '海水浸泡失重: 抛光速率 5-15 μm/月',
        '涂层厚度 200-350 μm',
        '附着力 ≥ 3B',
    ],
    'safety': [
        '铜化合物有水生生物毒性, 按环保要求处理废液',
        '二甲苯易燃有毒, 通风橱操作',
        'Cu₂O粉末佩戴口罩',
    ],
    'references': [
        'Yebra, D. M. et al., Prog. Org. Coat. 2004, 50, 75-104. DOI: 10.1016/j.porgcoat.2003.06.001',
        'Bressy, C. et al., Prog. Org. Coat. 2017, 104, 32-37. DOI: 10.1016/j.porgcoat.2016.11.002',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '中',
})

_reg({
    'name': '丙烯酸锌聚合物',
    'class': 'self_polishing',
    'method': '锌盐交联型自抛光共聚物',
    'reagents': [
        {'name': '甲基丙烯酸 (MAA)', 'role': '羧酸单体', 'amount': '4 g'},
        {'name': 'MMA', 'role': '共聚单体', 'amount': '8 g'},
        {'name': '丙烯酸丁酯 (BA)', 'role': '增柔单体', 'amount': '3 g'},
        {'name': '氧化锌 (ZnO)', 'role': '锌交联源', 'amount': '1.5 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.15 g'},
        {'name': '二甲苯', 'role': '溶剂', 'amount': '18 mL'},
    ],
    'conditions': {
        'temperature': '75 °C聚合; 90 °C锌交换',
        'time': '聚合10 h; 锌交换3 h',
        'atmosphere': 'N₂ (聚合)',
        'solvent': '二甲苯',
        'catalyst': 'AIBN',
        'other': '锌丙烯酸酯水解速率介于铜和硅之间',
    },
    'steps': [
        '将MMA、BA、MAA和AIBN溶于二甲苯, N₂鼓泡30 min',
        '75 °C滴加聚合10 h, 得到含-COOH共聚物溶液',
        '加入研磨后ZnO粉末, 升温至90 °C搅拌3 h进行锌交换',
        '过滤, 产品为浅黄色透明树脂液',
        '加入松香、增塑剂和防沉剂调配成防污漆',
        '刷涂或喷涂于底漆处理基材, 干膜厚度250-300 μm',
    ],
    'characterization': [
        'FTIR: -COO-Zn (1560 cm⁻¹)',
        'TGA: 锌含量 (灰分)',
        '海水抛光实验: 失重 + 厚度变化',
        '防污板挂测试 (静态海试)',
    ],
    'safety': [
        'ZnO粉末避免吸入',
        '锌化合物对水生生物有毒性, 但低于铜',
        '二甲苯通风操作',
    ],
    'references': [
        'Kiil, S. et al., Ind. Eng. Chem. Res. 2002, 41, 3956-3966. DOI: 10.1021/ie020021e',
        'Yebra, D. M. et al., Prog. Org. Coat. 2004, 50, 75-104. DOI: 10.1016/j.porgcoat.2003.06.001',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '中',
})

_reg({
    'name': '硅基丙烯酸酯自抛光',
    'class': 'self_polishing',
    'method': '硅酯基可水解共聚物',
    'reagents': [
        {'name': '三异丙基硅基甲基丙烯酸酯 (TIPSMA)', 'role': '可水解硅酯单体', 'amount': '8 g'},
        {'name': 'MMA', 'role': '硬段单体', 'amount': '5 g'},
        {'name': 'BA', 'role': '软段单体', 'amount': '2 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.15 g'},
        {'name': '甲苯', 'role': '溶剂', 'amount': '15 mL'},
    ],
    'conditions': {
        'temperature': '70 °C',
        'time': '14 h',
        'atmosphere': 'N₂',
        'solvent': '甲苯',
        'catalyst': 'AIBN',
        'other': '无金属环保型自抛光; 水解产生醇和硅醇',
    },
    'steps': [
        '单体、AIBN溶于甲苯, 冻融脱气2次',
        'N₂保护, 70 °C聚合14 h',
        '冷却, 甲醇沉淀纯化, 真空干燥',
        'GPC测Mn (目标3-5万)',
        '溶于甲苯 (30 wt%), 刮涂于底漆基材, 干燥成膜',
        '人工海水 (pH 8.2) 浸泡测试抛光速率',
    ],
    'characterization': [
        'FTIR: Si-O-C (1080 cm⁻¹), 浸泡后Si-OH出现',
        'GPC: Mn 3-5×10⁴',
        '失重法: 抛光速率',
        '水接触角: 初始 ~95°, 浸泡6月后 ~70°',
    ],
    'safety': [
        '无重金属, 环保性好',
        '甲苯易燃, 通风操作',
        'AIBN常规安全',
    ],
    'references': [
        'Bressy, C. et al., J. Coat. Technol. Res. 2014, 11, 383-392. DOI: 10.1007/s11998-014-9576-9',
        'Lejars, M. et al., Polym. Chem. 2013, 4, 3282-3292. DOI: 10.1039/c3py00154g',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

# ======== bioinspired 类 ========

_reg({
    'name': '仿贻贝多巴胺前体',
    'class': 'bioinspired',
    'method': '多巴胺碱性自聚合沉积(PDA)',
    'reagents': [
        {'name': '盐酸多巴胺', 'role': '单体', 'amount': '100 mg'},
        {'name': 'Tris碱', 'role': '缓冲剂', 'amount': '121 mg (10 mM)'},
        {'name': '盐酸 (1 M)', 'role': '调pH', 'amount': '适量至pH 8.5'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '50 mL'},
    ],
    'conditions': {
        'temperature': '25 °C',
        'time': '24 h',
        'atmosphere': '空气(需O₂)',
        'solvent': 'Tris缓冲液 pH 8.5',
        'catalyst': 'O₂氧化',
        'other': '溶液逐渐变黑; 多次沉积可增加膜厚',
    },
    'steps': [
        '配制10 mM Tris-HCl缓冲液 (pH 8.5): Tris碱溶于水, HCl调pH',
        '基材 (玻璃/钢/塑料) 用乙醇和水超声清洗各15 min',
        '将盐酸多巴胺快速溶于缓冲液 (2 mg/mL), 立刻浸入基材',
        '室温开放搅拌, 24 h后溶液变深棕至黑色',
        '取出基材, 去离子水洗涤3次, N₂吹干',
        '重复沉积可叠加厚度: 每次 ~5 nm/h',
    ],
    'characterization': [
        'XPS: N 1s (399-400 eV, 胺/亚胺)',
        'UV-Vis: 广谱吸收, ~280 nm邻苯二酚特征',
        '椭偏: 24 h膜厚 ~40-50 nm',
        '水接触角: ~50° (PDA)',
    ],
    'safety': [
        '多巴胺溶液使皮肤和衣物永久染色, 佩戴手套和防护服',
        'Tris缓冲液本身基本无毒',
        '废液中含聚合物颗粒, 过滤后排放',
    ],
    'references': [
        'Lee, H. et al., Science 2007, 318, 426-430. DOI: 10.1126/science.1147241',
        'Liu, Y. et al., Langmuir 2014, 30, 14189-14197. DOI: 10.1021/la503915z',
    ],
    'difficulty': '基础',
    'scalability': '中试',
    'cost_level': '低',
})

_reg({
    'name': '仿荷叶丙烯酸-PEG',
    'class': 'bioinspired',
    'method': '微纳复合结构超疏水表面',
    'reagents': [
        {'name': '含氟丙烯酸酯共聚物 (FMA/MMA)', 'role': '低表面能涂层', 'amount': '5 wt%溶液'},
        {'name': 'SiO₂纳米粒子 (7 nm, 疏水改性)', 'role': '纳米粗糙结构', 'amount': '2 wt%'},
        {'name': 'PDMS微球模板', 'role': '微米级结构', 'amount': '适量'},
        {'name': '1H,1H,2H,2H-全氟癸基三乙氧基硅烷', 'role': '表面修饰', 'amount': '1 vol%'},
        {'name': '乙醇', 'role': '溶剂', 'amount': '50 mL'},
    ],
    'conditions': {
        'temperature': '室温喷涂; 100 °C固化',
        'time': '固化1 h',
        'atmosphere': '空气',
        'solvent': '乙醇',
        'catalyst': '无',
        'other': '需构建微纳二级结构实现超疏水',
    },
    'steps': [
        '将SiO₂纳米粒子分散于乙醇, 超声30 min',
        '加入氟硅烷 (1 vol%), 室温搅拌2 h进行表面改性',
        '离心收集改性SiO₂, 重新分散于含氟丙烯酸酯溶液',
        '喷涂于基材表面 (可选先用软光刻制微米柱阵列)',
        '100 °C固化1 h',
        '测量: 水接触角 >150°, 滚动角 <10°, 确认超疏水',
    ],
    'characterization': [
        '水接触角 >150° (超疏水)',
        '滚动角 <10°',
        'SEM: 微纳二级结构',
        'AFM: 粗糙度 Ra 50-200 nm',
        '耐磨性测试: 砂纸摩擦后接触角变化',
    ],
    'safety': [
        '含氟硅烷对皮肤有刺激',
        '纳米SiO₂粉尘佩戴口罩',
        '乙醇易燃',
    ],
    'references': [
        'Bhushan, B.; Jung, Y. C., Prog. Mater. Sci. 2011, 56, 1-108. DOI: 10.1016/j.pmatsci.2010.04.003',
        'Zhang, X. et al., J. Mater. Chem. 2008, 18, 621-633. DOI: 10.1039/B711226B',
    ],
    'difficulty': '中等',
    'scalability': '实验室',
    'cost_level': '中',
})

_reg({
    'name': '仿鲨鱼皮全氟',
    'class': 'bioinspired',
    'method': '光刻 + PDMS翻模 + 氟涂层',
    'reagents': [
        {'name': 'SU-8负性光刻胶', 'role': '母版制作', 'amount': '适量'},
        {'name': 'Sylgard 184 PDMS', 'role': '翻模材料', 'amount': '10:1'},
        {'name': 'UV固化氟碳树脂', 'role': '最终涂层', 'amount': '适量'},
        {'name': '全氟聚醚脱模剂', 'role': '脱模', 'amount': '少量'},
    ],
    'conditions': {
        'temperature': 'SU-8 95 °C烘烤; PDMS 80 °C固化',
        'time': 'PDMS固化 2 h; UV固化 5 min',
        'atmosphere': '净化间操作',
        'solvent': '无',
        'catalyst': 'UV光固化',
        'other': '鲨鱼皮肋条间距 50-100 μm, 高度 10-50 μm',
    },
    'steps': [
        '设计鲨鱼皮肋条 (riblet) 图案, 间距50-100 μm, 高宽比2:1',
        'SU-8光刻制作硅片母版: 旋涂→前烘→UV曝光→后烘→显影',
        '母版表面涂全氟聚醚脱模剂, PDMS浇注翻模, 80 °C/2 h固化',
        '脱模得到PDMS阴模, 再次浇注UV固化氟碳树脂',
        'UV曝光固化5 min, 脱模得到具有鲨鱼皮微结构的氟碳表面',
        'SEM验证肋条尺寸, 接触角和水下阻力测试',
    ],
    'characterization': [
        'SEM: 肋条高度/间距/形貌',
        '水接触角: 结构+化学协同 >120°',
        '水下摩擦阻力: 减阻5-10%',
        '防污板测试: 藤壶附着降低>50%',
    ],
    'safety': [
        'SU-8光刻胶含环氧基, 避免皮肤接触',
        'UV灯佩戴防护镜',
        '有机溶剂(显影液)通风操作',
    ],
    'references': [
        'Schumacher, J. F. et al., Biofouling 2007, 23, 55-62. DOI: 10.1080/08927010601136957',
        'Bixler, G. D.; Bhushan, B., Adv. Funct. Mater. 2013, 23, 4507-4528. DOI: 10.1002/adfm.201203683',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})

# ======== nanocomposite 类 ========

_reg({
    'name': 'PDMS/ZnO纳米复合',
    'class': 'nanocomposite',
    'method': '硅烷改性ZnO纳米粒子/PDMS共混',
    'reagents': [
        {'name': 'ZnO纳米粒子 (30-50 nm)', 'role': '抗菌纳米填料', 'amount': '0.5 g (5 wt%)'},
        {'name': 'KH-570 (γ-甲基丙烯酰氧丙基三甲氧基硅烷)', 'role': '偶联剂', 'amount': '0.025 g (5 wt%/NP)'},
        {'name': 'Sylgard 184 A (PDMS预聚物)', 'role': '基体', 'amount': '9.5 g'},
        {'name': 'Sylgard 184 B (固化剂)', 'role': '交联剂', 'amount': '0.95 g'},
        {'name': '无水乙醇', 'role': '改性溶剂', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '60 °C改性; 80 °C固化',
        'time': '改性4 h; 固化2 h + 后固化150 °C/1 h',
        'atmosphere': '空气',
        'solvent': '无水乙醇 (改性步骤)',
        'catalyst': 'Pt (B组分中)',
        'other': 'ZnO含量1-5 wt%范围可调',
    },
    'steps': [
        'ZnO纳米粒子分散于乙醇 (30 mL), 超声30 min',
        '加入KH-570, 60 °C回流搅拌4 h进行表面改性',
        '离心 (8000 rpm, 10 min), 乙醇洗3次, 60 °C真空干燥过夜',
        '将改性ZnO加入PDMS A组分, 超声分散30 min',
        '加入B组分 (A:B = 10:1), 搅拌混匀, 真空脱泡15 min',
        '倾倒入模具, 80 °C/2 h + 150 °C/1 h梯度固化',
        '脱模, 进行抗菌测试和接触角测量',
    ],
    'characterization': [
        'SEM+EDS: ZnO分散均匀性, Zn元素mapping',
        'XRD: ZnO (100), (002), (101) 衍射峰',
        '水接触角 105-115°',
        '抗菌率: E. coli >99%, S. aureus >95% (5 wt% ZnO)',
        'TGA: 实际ZnO含量确认',
    ],
    'safety': [
        'ZnO纳米粒子吸入有害, 佩戴N95口罩',
        '离心操作配重平衡',
        'KH-570遇水水解, 密封保存',
    ],
    'references': [
        'Al-Naamani, L. et al., Chemosphere 2017, 168, 408-417. DOI: 10.1016/j.chemosphere.2016.10.066',
        'Beigbeder, A. et al., Biofouling 2008, 24, 291-302. DOI: 10.1080/08927010802162885',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': 'PDMS/Ag纳米复合',
    'class': 'nanocomposite',
    'method': '原位还原法制备Ag NP/PDMS',
    'reagents': [
        {'name': '硝酸银 (AgNO₃)', 'role': '银源', 'amount': '0.17 g (1 mmol)'},
        {'name': '柠檬酸三钠', 'role': '还原剂+稳定剂', 'amount': '0.29 g (1 mmol)'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '50 mL'},
        {'name': 'Sylgard 184 A+B', 'role': 'PDMS基体', 'amount': '10+1 g'},
        {'name': '正己烷', 'role': 'PDMS溶胀剂', 'amount': '20 mL'},
    ],
    'conditions': {
        'temperature': 'Ag NP合成 100 °C; PDMS固化 80 °C',
        'time': 'Ag NP 1 h; 溶胀吸附 24 h; 固化 2 h',
        'atmosphere': '空气',
        'solvent': '水 (Ag NP合成) + 正己烷 (PDMS溶胀)',
        'catalyst': '柠檬酸钠 (还原)',
        'other': 'Ag NP粒径 20-40 nm (柠檬酸钠法)',
    },
    'steps': [
        '柠檬酸三钠法合成Ag NP: AgNO₃水溶液煮沸, 加入柠檬酸钠, 1 h',
        '溶液变为黄色确认Ag NP形成, 冷却',
        '先固化PDMS片材 (A:B = 10:1, 80 °C/2 h), 脱模',
        '将PDMS片浸入正己烷24 h溶胀 (体积膨胀~100%)',
        '将溶胀PDMS浸入Ag NP胶体溶液 (加少量乙醇助混) 24 h',
        '取出, 60 °C干燥12 h, 正己烷挥发PDMS收缩包裹Ag NP',
        '测量: SEM/TEM观察Ag分布, 抗菌性能, 银离子释放',
    ],
    'characterization': [
        'UV-Vis: Ag SPR峰 ~420 nm',
        'TEM: Ag NP粒径 20-40 nm',
        'XRD: Ag (111), (200), (220)',
        'ICP-MS: Ag含量和释放速率',
        '抗菌率: E. coli >99%',
    ],
    'safety': [
        'AgNO₃强氧化性, 使皮肤变黑, 戴手套',
        '正己烷易燃, 神经毒性, 通风橱',
        '纳米银对环境有潜在生态毒性',
    ],
    'references': [
        'Kvitek, L. et al., J. Phys. Chem. C 2008, 112, 5825-5834. DOI: 10.1021/jp711616v',
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': 'PDMS/石墨烯复合',
    'class': 'nanocomposite',
    'method': 'Hummers法制GO + PDMS共混',
    'reagents': [
        {'name': '石墨粉 (天然, 325目)', 'role': '碳源', 'amount': '2 g'},
        {'name': 'KMnO₄', 'role': '氧化剂', 'amount': '6 g'},
        {'name': '浓H₂SO₄', 'role': '氧化介质', 'amount': '46 mL'},
        {'name': 'H₂O₂ (30%)', 'role': '终止/还原残余KMnO₄', 'amount': '10 mL'},
        {'name': 'Sylgard 184', 'role': 'PDMS基体', 'amount': 'A:B = 10:1'},
        {'name': '甲苯', 'role': 'GO分散/PDMS溶胀', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '0→35→98 °C (分段控温)',
        'time': 'GO合成 ~6 h; PDMS固化 2 h',
        'atmosphere': '空气',
        'solvent': '甲苯',
        'catalyst': '无',
        'other': '改良Hummers法; GO表面含含氧基团有利于分散',
    },
    'steps': [
        '冰浴下将石墨粉加入浓H₂SO₄, 搅拌, 缓慢加入KMnO₄ (温度<20 °C)',
        '35 °C搅拌2 h, 缓慢加入去离子水, 升温至98 °C搅拌15 min',
        '加入H₂O₂至溶液变金黄色, 趁热过滤',
        '10% HCl洗涤3次去除金属离子, 去离子水透析至pH~7',
        '超声剥离1 h得到GO悬浮液, 冻干得GO粉末',
        '将GO分散于甲苯 (超声), 加入PDMS预聚物混合, 真空除溶剂后加固化剂',
        '80 °C/2 h固化, 得到PDMS/GO复合材料',
    ],
    'characterization': [
        'Raman: G (~1580 cm⁻¹) 和 D (~1350 cm⁻¹) 峰',
        'XRD: GO (001) ~10°',
        'SEM: GO在PDMS中的分散',
        '拉伸强度提升 30-50%',
        '水接触角 100-110°',
    ],
    'safety': [
        '改良Hummers法强放热, 必须冰浴并缓慢加KMnO₄',
        '浓H₂SO₄强腐蚀, 佩戴护目镜和耐酸手套',
        'KMnO₄强氧化剂',
        'H₂O₂与KMnO₄反应剧烈, 缓慢添加',
    ],
    'references': [
        'Hummers, W. S.; Offeman, R. E., J. Am. Chem. Soc. 1958, 80, 1339. DOI: 10.1021/ja01539a017',
        'Stankovich, S. et al., Carbon 2007, 45, 1558-1565. DOI: 10.1016/j.carbon.2007.02.034',
    ],
    'difficulty': '高级',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': 'PDMS/Cu2O',
    'class': 'nanocomposite',
    'method': '化学沉淀法制Cu₂O + PDMS复合',
    'reagents': [
        {'name': '硫酸铜 (CuSO₄·5H₂O)', 'role': '铜源', 'amount': '2.5 g'},
        {'name': '氢氧化钠 (NaOH)', 'role': '沉淀剂', 'amount': '2 g'},
        {'name': '葡萄糖', 'role': '还原剂', 'amount': '1.5 g'},
        {'name': 'Sylgard 184', 'role': 'PDMS基体', 'amount': '10+1 g'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '100 mL'},
    ],
    'conditions': {
        'temperature': 'Cu₂O合成 60 °C; PDMS固化 80 °C',
        'time': 'Cu₂O 30 min; PDMS 2 h',
        'atmosphere': '空气',
        'solvent': '水 (Cu₂O) / 本体 (PDMS)',
        'catalyst': '无',
        'other': 'Cu₂O为传统防污活性成分',
    },
    'steps': [
        'CuSO₄溶于50 mL水, 加入NaOH溶液生成Cu(OH)₂蓝色沉淀',
        '加入葡萄糖溶液, 60 °C水浴搅拌30 min, 沉淀变为砖红色Cu₂O',
        '抽滤, 去离子水洗3次, 乙醇洗1次, 60 °C干燥12 h',
        'XRD确认Cu₂O纯度',
        '将Cu₂O粉末 (10-30 wt%) 加入PDMS A组分, 超声+搅拌分散',
        '加B组分, 脱泡, 浇注, 80 °C/2 h固化',
    ],
    'characterization': [
        'XRD: Cu₂O (110), (111), (200) 确认物相',
        'SEM: 粒子形貌 (立方/八面体, ~200 nm-1 μm)',
        'Cu²⁺释放速率 (ICP)',
        '防污板海试',
    ],
    'safety': [
        'CuSO₄对皮肤有刺激, Cu₂O有水生毒性',
        'NaOH强碱腐蚀',
        'Cu₂O粉末佩戴口罩',
    ],
    'references': [
        'Yebra, D. M. et al., Prog. Org. Coat. 2004, 50, 75-104. DOI: 10.1016/j.porgcoat.2003.06.001',
        'Devi, A. B. et al., J. Mater. Sci. 2014, 49, 5887-5898. DOI: 10.1007/s10853-014-8306-x',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '低',
})


# ======== smart 类 ========

_reg({
    'name': '温度响应PNIPAM-丙烯酸',
    'class': 'smart',
    'method': 'RAFT聚合制备PNIPAM-co-AA',
    'reagents': [
        {'name': 'N-异丙基丙烯酰胺 (NIPAM)', 'role': '温度响应单体', 'amount': '2.0 g'},
        {'name': '丙烯酸 (AA)', 'role': 'pH响应共聚单体', 'amount': '0.2 g'},
        {'name': '4-氰基-4-(十二基三硫代碳酸酯基)戊酸 (CDTPA)', 'role': 'RAFT链转移剂', 'amount': '0.08 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '5 mg'},
        {'name': '1,4-二氧六环', 'role': '溶剂', 'amount': '10 mL'},
    ],
    'conditions': {
        'temperature': '70 °C',
        'time': '18 h',
        'atmosphere': 'N₂',
        'solvent': '1,4-二氧六环',
        'catalyst': 'RAFT CTA + AIBN',
        'other': 'CTA:AIBN = 5:1; AA含量调节LCST',
    },
    'steps': [
        'NIPAM用正己烷重结晶, 干燥; AA减压蒸馏纯化',
        '称取NIPAM、AA、CDTPA、AIBN于Schlenk瓶, 加入二氧六环',
        '冻融脱气3次, 充N₂密封',
        '70 °C油浴搅拌反应18 h',
        '冷却, 打开后在冰乙醚中沉淀3次',
        '透析 (MWCO 3500, 去离子水) 48 h, 冻干得粉红色粉末',
        'GPC测分子量, DSC测LCST, 水溶液turbidimetry测云点',
    ],
    'characterization': [
        '¹H-NMR: NIPAM酰胺 (6.5 ppm), 异丙基 (4.0, 1.1 ppm)',
        'GPC: Mn ~2-4×10⁴, PDI < 1.25',
        'DSC: LCST 32-38 °C (AA含量提高LCST)',
        'UV-Vis turbidimetry: 云点温度',
        '水接触角: 25 °C ~40° → 40 °C ~80°',
    ],
    'safety': [
        '1,4-二氧六环可能致癌, 严格通风橱',
        'NIPAM有潜在毒性, 戴手套',
        'RAFT试剂含硫, 有臭味',
    ],
    'references': [
        'Moad, G. et al., Aust. J. Chem. 2005, 58, 379-410. DOI: 10.1071/CH05072',
        'Schild, H. G., Prog. Polym. Sci. 1992, 17, 163-249. DOI: 10.1016/0079-6700(92)90023-R',
        'Gao, C. et al., Langmuir 2013, 29, 14573-14581. DOI: 10.1021/la4028987',
    ],
    'difficulty': '高级',
    'scalability': '实验室',
    'cost_level': '高',
})

_reg({
    'name': 'pH响应丙烯酸-PEG-PDMS',
    'class': 'smart',
    'method': '多嵌段共聚 + 层层自组装',
    'reagents': [
        {'name': '丙烯酸 (AA)', 'role': 'pH响应单体', 'amount': '2 g'},
        {'name': 'PEG甲基丙烯酸酯 (PEG-MA, Mn 360)', 'role': '亲水嵌段', 'amount': '3 g'},
        {'name': 'PDMS甲基丙烯酸酯 (PDMS-MA, Mn 1000)', 'role': '疏水嵌段', 'amount': '5 g'},
        {'name': 'AIBN', 'role': '引发剂', 'amount': '0.1 g'},
        {'name': 'THF', 'role': '溶剂', 'amount': '20 mL'},
    ],
    'conditions': {
        'temperature': '65 °C',
        'time': '16 h',
        'atmosphere': 'N₂',
        'solvent': 'THF',
        'catalyst': 'AIBN',
        'other': 'AA:PEG-MA:PDMS-MA ≈ 4:2:1 (mol)',
    },
    'steps': [
        '单体和AIBN溶于THF, N₂鼓泡脱气30 min',
        '65 °C反应16 h',
        '冷却, 旋蒸除THF, 正己烷沉淀3次',
        '真空干燥得到两亲性三元共聚物',
        '将共聚物溶于乙醇 (5 wt%), 旋涂于基材',
        'pH 4→pH 8缓冲液浸泡测试接触角变化',
    ],
    'characterization': [
        'FTIR: -COOH (1710 cm⁻¹), PEG (1100 cm⁻¹), Si-O (1020 cm⁻¹)',
        'pH 4: 接触角 ~85° (COOH质子化, PDMS朝表面)',
        'pH 8: 接触角 ~45° (COO⁻, PEG朝表面)',
        'QCM: pH切换时蛋白质吸脱附行为',
    ],
    'safety': [
        '丙烯酸有皮肤腐蚀性, 戴手套护目镜',
        'THF可能含过氧化物, 使用前检测',
        '正己烷神经毒性, 通风操作',
    ],
    'references': [
        'Nath, N.; Chilkoti, A., Adv. Mater. 2002, 14, 1243-1246. DOI: 10.1002/1521-4095(20020903)14:17<1243::AID-ADMA1243>3.0.CO;2-M',
        'Chen, S. et al., Polymer 2010, 51, 5283-5293. DOI: 10.1016/j.polymer.2010.08.072',
    ],
    'difficulty': '中等',
    'scalability': '实验室',
    'cost_level': '中',
})

_reg({
    'name': '光响应丙烯酸-氟-TiO2',
    'class': 'smart',
    'method': 'TiO₂溶胶-凝胶法 + 含氟丙烯酸酯涂层',
    'reagents': [
        {'name': '钛酸四丁酯 (TBOT)', 'role': 'TiO₂前驱体', 'amount': '5 mL'},
        {'name': '无水乙醇', 'role': '溶剂', 'amount': '20 mL'},
        {'name': '冰醋酸', 'role': '螯合剂/抑制水解', 'amount': '1 mL'},
        {'name': '含氟丙烯酸酯共聚物 (FMA/MMA)', 'role': '基体涂层', 'amount': '5 g (BTF溶液)'},
        {'name': '去离子水', 'role': '水解剂', 'amount': '2 mL'},
    ],
    'conditions': {
        'temperature': '溶胶-凝胶 60 °C; 煅烧 450 °C; 涂层固化 80 °C',
        'time': '溶胶陈化 24 h; 煅烧 2 h',
        'atmosphere': '空气',
        'solvent': '乙醇',
        'catalyst': '醋酸 (抑制)',
        'other': 'UV照射使TiO₂光催化降解有机污损物',
    },
    'steps': [
        'TBOT溶于乙醇, 加入冰醋酸螯合, 搅拌30 min',
        '缓慢滴加去离子水, 室温搅拌, 形成透明溶胶',
        '溶胶室温陈化24 h, 旋涂于玻璃基材, 60 °C干燥',
        '450 °C煅烧2 h得到锐钛矿TiO₂薄膜',
        '在TiO₂膜上旋涂一薄层含氟丙烯酸酯溶液, 80 °C固化',
        'UV照射下测接触角变化 (超亲水化), 暗处恢复',
    ],
    'characterization': [
        'XRD: 锐钛矿TiO₂ (101) 2θ=25.3°',
        'UV-Vis: 吸收边 ~385 nm',
        '水接触角: 暗态 >100° → UV照2 h < 10° → 暗恢复',
        'SEM: TiO₂颗粒 ~10-20 nm',
    ],
    'safety': [
        'TBOT遇水剧烈水解, 密封保存',
        '450 °C煅烧注意高温安全',
        'TiO₂纳米粒子吸入有害',
    ],
    'references': [
        'Wang, R. et al., Nature 1997, 388, 431-432. DOI: 10.1038/41233',
        'Banerjee, S. et al., J. Phys. Chem. Lett. 2011, 2, 2726-2737. DOI: 10.1021/jz201279j',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})

_reg({
    'name': 'SLIPS-PDMS+PEG润滑液',
    'class': 'smart',
    'method': 'SLIPS超滑表面制备 (液体灌注)',
    'reagents': [
        {'name': '多孔PTFE膜 (0.2 μm孔径)', 'role': '多孔基底', 'amount': '5 cm × 5 cm'},
        {'name': '全氟聚醚润滑油 (Krytox 100)', 'role': '灌注润滑液', 'amount': '5 mL'},
        {'name': 'PDMS油 (100 cSt) + PEG 200 混合', 'role': '替代润滑液', 'amount': '5 mL (1:1 v/v)'},
        {'name': '全氟癸基三氯硅烷', 'role': '基底疏氟化处理', 'amount': '0.1 mL'},
    ],
    'conditions': {
        'temperature': '室温',
        'time': '灌注浸泡 2-4 h',
        'atmosphere': '空气',
        'solvent': '无',
        'catalyst': '无',
        'other': '润滑液需与基底化学亲和; 须完全覆盖微孔',
    },
    'steps': [
        '多孔PTFE膜经O₂等离子处理提高表面能 (可选)',
        '气相沉积全氟硅烷修饰表面 (真空干燥器中放硅烷, 室温12 h)',
        '将润滑液 (PDMS/PEG混合或Krytox) 滴加于多孔膜表面',
        '让润滑液在毛细作用下完全灌注微孔 (2-4 h)',
        '轻轻倾斜排去表面过量润滑液',
        '测试: 水滴在SLIPS上滑动角 <5°, 无接触角滞后',
    ],
    'characterization': [
        '滑动角 < 5° (10 μL水滴)',
        '接触角滞后 < 3°',
        '显微镜观察液膜完整性',
        '长期稳定性: 流动海水冲刷测试',
        '防污: 藻类和细菌附着降低>95%',
    ],
    'safety': [
        '全氟硅烷有腐蚀性, 通风橱气相沉积',
        'Krytox油本身低毒但价格高',
        '等离子设备操作注意真空和电气安全',
    ],
    'references': [
        'Wong, T.-S. et al., Nature 2011, 477, 443-447. DOI: 10.1038/nature10447',
        'Epstein, A. K. et al., Proc. Natl. Acad. Sci. U.S.A. 2012, 109, 13182-13187. DOI: 10.1073/pnas.1210770109',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '高',
})

_reg({
    'name': 'PNIPAM-羟乙基',
    'class': 'smart',
    'method': '自由基共聚合 (NIPAM + HEMA)',
    'reagents': [
        {'name': 'NIPAM', 'role': '温度响应单体', 'amount': '2.5 g'},
        {'name': 'HEMA', 'role': '羟基功能单体', 'amount': '0.5 g'},
        {'name': 'BIS', 'role': '交联剂', 'amount': '0.03 g'},
        {'name': 'APS', 'role': '引发剂', 'amount': '0.05 g'},
        {'name': 'TEMED', 'role': '加速剂', 'amount': '20 μL'},
        {'name': '去离子水', 'role': '溶剂', 'amount': '20 mL'},
    ],
    'conditions': {
        'temperature': '室温引发 → 25 °C聚合',
        'time': '6 h',
        'atmosphere': 'N₂',
        'solvent': '水',
        'catalyst': 'APS/TEMED氧化还原引发',
        'other': 'HEMA提供羟基锚固位点',
    },
    'steps': [
        'NIPAM和HEMA溶于去离子水, 加入BIS交联剂',
        'N₂鼓泡20 min除氧',
        '加入APS溶液, 搅匀后加入TEMED',
        '迅速注入模具, 25 °C静置6 h完成凝胶化',
        '脱模后在去离子水中平衡48 h',
        '测试LCST和温度响应溶胀比变化',
    ],
    'characterization': [
        'FTIR: 酰胺I/II (NIPAM), -OH (HEMA)',
        'DSC: LCST ~33-35 °C',
        '溶胀比: 25 °C/40 °C体积变化 >3倍',
        '水接触角: 25 °C 亲水 → 40 °C 疏水',
    ],
    'safety': [
        'NIPAM有毒性, 戴手套',
        'APS强氧化剂',
        'TEMED腐蚀性, 通风操作',
    ],
    'references': [
        'Schild, H. G., Prog. Polym. Sci. 1992, 17, 163-249. DOI: 10.1016/0079-6700(92)90023-R',
        'Xia, Y. et al., Macromolecules 2005, 38, 5937-5943. DOI: 10.1021/ma050564g',
    ],
    'difficulty': '中等',
    'scalability': '实验室',
    'cost_level': '中',
})

# ── Additional specific materials to reach 30+ ──

_reg({
    'name': 'PDMS+乙烯基改性',
    'class': 'silicone',
    'method': '乙烯基硅氧烷与含氢硅油硅氢加成',
    'reagents': [
        {'name': '乙烯基封端PDMS (Vi-PDMS, Mn 28000)', 'role': '预聚物', 'amount': '10 g'},
        {'name': '含氢硅油 (PMHS)', 'role': '交联剂', 'amount': '0.8 g'},
        {'name': 'Karstedt催化剂 (Pt 2%)', 'role': '催化剂', 'amount': '50 μL'},
        {'name': '1-乙炔基环己醇', 'role': '阻聚剂', 'amount': '10 μL'},
    ],
    'conditions': {
        'temperature': '100 °C',
        'time': '1-2 h',
        'atmosphere': '空气',
        'solvent': '无溶剂',
        'catalyst': 'Karstedt Pt',
        'other': 'Si-H:Vi = 1.5:1',
    },
    'steps': [
        '将Vi-PDMS和阻聚剂混合, 加入Karstedt催化剂',
        '加入含氢硅油, 高速搅拌3 min混匀',
        '真空脱泡10 min',
        '浇注于铝模具, 100 °C固化1.5 h',
        '脱模, FTIR确认Vi (1600 cm⁻¹) 和Si-H (2160 cm⁻¹) 消失',
    ],
    'characterization': [
        'FTIR: 乙烯基和Si-H消失确认完全交联',
        'Shore A 30-50',
        '水接触角 105-112°',
        '拉伸强度 5-8 MPa',
    ],
    'safety': [
        'Pt催化剂避免皮肤接触',
        '阻聚剂含炔基, 少量使用',
    ],
    'references': [
        'Yilgör, E.; Yilgör, I., Prog. Polym. Sci. 2014, 39, 1165-1195. DOI: 10.1016/j.progpolymsci.2013.11.003',
    ],
    'difficulty': '基础',
    'scalability': '工业',
    'cost_level': '低',
})

_reg({
    'name': '含氟聚氨酯',
    'class': 'fluoropolymer',
    'method': '含氟二醇扩链聚氨酯合成',
    'reagents': [
        {'name': 'IPDI', 'role': '二异氰酸酯', 'amount': '4.4 g'},
        {'name': '聚四亚甲基醚二醇 (PTMG, Mn 1000)', 'role': '软段多元醇', 'amount': '10 g'},
        {'name': '全氟己基乙醇 (C6F-OH)', 'role': '含氟扩链/封端', 'amount': '2 g'},
        {'name': 'DBTDL', 'role': '催化剂', 'amount': '0.02 wt%'},
        {'name': 'DMF', 'role': '溶剂', 'amount': '25 mL'},
    ],
    'conditions': {
        'temperature': '60-80 °C',
        'time': '8 h',
        'atmosphere': 'N₂',
        'solvent': 'DMF',
        'catalyst': 'DBTDL',
        'other': 'NCO/OH略过量; C6F-OH封端引入氟链段',
    },
    'steps': [
        'PTMG真空干燥除水, 溶于DMF, 加入IPDI和DBTDL',
        '60 °C反应3 h生成NCO封端预聚体',
        '加入C6F-OH扩链/封端, 80 °C反应5 h',
        'FTIR监测NCO消失',
        '反应液倒入水中沉淀, 洗涤, 真空干燥',
        '溶于DMF (10 wt%), 流延成膜, 60 °C/24 h',
    ],
    'characterization': [
        'FTIR: C-F (1100-1200 cm⁻¹), 氨酯C=O (1710 cm⁻¹), N-H (3300 cm⁻¹)',
        'XPS: 表面F含量',
        '水接触角 100-115°',
        '拉伸强度 20-35 MPa',
    ],
    'safety': [
        'IPDI异氰酸酯, 严格通风橱操作',
        'DMF有生殖毒性',
        'DBTDL有机锡有毒',
    ],
    'references': [
        'Wang, L.; Nelson, R. C.; et al., J. Polym. Sci. A Polym. Chem. 2000, 38, 3270-3279. DOI: 10.1002/1099-0518(20000901)38:17<3270::AID-POLA210>3.0.CO;2-Z',
        'Lejars, M. et al., Chem. Rev. 2012, 112, 4347-4390. DOI: 10.1021/cr200350u',
    ],
    'difficulty': '高级',
    'scalability': '中试',
    'cost_level': '高',
})

_reg({
    'name': 'PDMS/SiO2纳米复合',
    'class': 'nanocomposite',
    'method': '气相法SiO₂表面改性 + PDMS共混',
    'reagents': [
        {'name': '气相法SiO₂ (Aerosil 200, 12 nm)', 'role': '补强纳米填料', 'amount': '1-3 g (10-30 phr)'},
        {'name': '六甲基二硅氮烷 (HMDS)', 'role': 'SiO₂疏水化改性', 'amount': '3 mL'},
        {'name': 'Sylgard 184 A+B', 'role': 'PDMS基体', 'amount': '10+1 g'},
        {'name': '甲苯', 'role': '分散介质', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '改性 110 °C; 固化 80-150 °C',
        'time': '改性 3 h; 固化 2 h',
        'atmosphere': '空气',
        'solvent': '甲苯 (改性步骤)',
        'catalyst': 'Pt (PDMS固化)',
        'other': 'HMDS使SiO₂由亲水变疏水, 改善PDMS中分散',
    },
    'steps': [
        'SiO₂分散于甲苯, 超声30 min',
        '加入HMDS, 110 °C回流3 h表面甲基化',
        '离心, 甲苯洗涤2次, 真空干燥',
        '将改性SiO₂加入PDMS A组分, 行星式搅拌30 min',
        '加B组分, 搅拌脱泡, 浇注固化 80 °C/2 h',
        '测量拉伸强度和撕裂强度提升',
    ],
    'characterization': [
        'TEM: SiO₂分散状态',
        'FTIR: Si-CH₃ (HMDS改性确认)',
        '拉伸强度提升 50-200%',
        '水接触角 105-115°',
    ],
    'safety': [
        '纳米SiO₂粉尘佩戴口罩',
        'HMDS有刺激性, 通风操作',
        '甲苯易燃有毒',
    ],
    'references': [
        'Beigbeder, A. et al., Biofouling 2008, 24, 291-302. DOI: 10.1080/08927010802162885',
        'Mark, J. E., Acc. Chem. Res. 2004, 37, 946-953. DOI: 10.1021/ar030279z',
    ],
    'difficulty': '中等',
    'scalability': '工业',
    'cost_level': '中',
})

_reg({
    'name': 'PDMS/TiO2纳米复合',
    'class': 'nanocomposite',
    'method': 'TiO₂纳米粒子/PDMS光催化防污复合材料',
    'reagents': [
        {'name': 'TiO₂纳米粒子 (P25, 21 nm, 锐钛矿/金红石混合)', 'role': '光催化纳米填料', 'amount': '0.5 g (5 wt%)'},
        {'name': 'KH-570', 'role': '偶联剂', 'amount': '0.025 g'},
        {'name': 'Sylgard 184 A+B', 'role': 'PDMS基体', 'amount': '10+1 g'},
        {'name': '无水乙醇', 'role': '分散介质', 'amount': '30 mL'},
    ],
    'conditions': {
        'temperature': '改性 60 °C; 固化 80 °C',
        'time': '改性 4 h; 固化 2 h',
        'atmosphere': '空气',
        'solvent': '乙醇 (改性)',
        'catalyst': 'Pt (PDMS固化)',
        'other': 'UV照射激活TiO₂光催化杀菌',
    },
    'steps': [
        'P25 TiO₂分散于乙醇, 超声30 min',
        '加入KH-570, 60 °C搅拌4 h表面改性',
        '离心收集, 乙醇洗3次, 干燥',
        '加入PDMS A组分, 超声分散, 加B组分',
        '脱泡, 浇注, 80 °C/2 h固化',
        'UV照射下测光催化降解亚甲基蓝速率',
    ],
    'characterization': [
        'XRD: 锐钛矿 (101) + 金红石 (110)',
        'SEM: TiO₂分布',
        '水接触角 100-110°',
        'UV-Vis: 亚甲基蓝降解速率',
        '抗菌测试: UV下杀菌率 >99%',
    ],
    'safety': [
        'TiO₂纳米粉尘佩戴口罩',
        'UV操作佩戴防护镜',
    ],
    'references': [
        'Banerjee, S. et al., J. Phys. Chem. Lett. 2011, 2, 2726-2737. DOI: 10.1021/jz201279j',
        'Carl, C. et al., Biofouling 2012, 28, 175-186. DOI: 10.1080/08927014.2012.659244',
    ],
    'difficulty': '中等',
    'scalability': '中试',
    'cost_level': '中',
})


# ================================================================
#  CLASS-LEVEL LOOKUP TABLE
# ================================================================

_CLASS_GENERIC_NAMES = {
    'silicone':      '硅树脂/硅橡胶 (通用)',
    'fluoropolymer': '氟聚合物 (通用)',
    'hydrogel':      '水凝胶 (通用)',
    'zwitterionic':  '两性离子聚合物 (通用)',
    'self_polishing': '自抛光涂料 (通用)',
    'bioinspired':   '仿生防污材料 (通用)',
    'nanocomposite': '纳米复合防污材料 (通用)',
    'smart':         '智能响应涂层 (通用)',
}

_CLASS_CN = {
    'silicone':      '硅树脂/硅橡胶',
    'fluoropolymer': '氟聚合物',
    'hydrogel':      '水凝胶',
    'zwitterionic':  '两性离子',
    'self_polishing': '自抛光',
    'bioinspired':   '仿生',
    'nanocomposite': '纳米复合',
    'smart':         '智能响应',
}


# ================================================================
#  PUBLIC API FUNCTIONS
# ================================================================

def get_synthesis_route(
    material_name: Optional[str] = None,
    material_class: Optional[str] = None,
) -> Optional[dict]:
    """Return the best-matching synthesis route.

    Lookup priority:
    1. Exact name match in SYNTHESIS_DATABASE
    2. Substring/fuzzy match on name
    3. Generic class-level route
    """
    if material_name:
        name = material_name.strip()
        # 1. exact
        if name in SYNTHESIS_DATABASE:
            return SYNTHESIS_DATABASE[name]
        # 2. substring
        name_lower = name.lower()
        for key, entry in SYNTHESIS_DATABASE.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return entry

    # 3. class fallback
    if material_class:
        cls_key = material_class.strip().lower()
        generic = _CLASS_GENERIC_NAMES.get(cls_key)
        if generic and generic in SYNTHESIS_DATABASE:
            return SYNTHESIS_DATABASE[generic]

    # 4. If nothing matched but material_name given, try to infer class
    if material_name:
        name_lower = material_name.lower()
        class_hints = {
            'silicone': ['pdms', '硅', 'silicone', 'siloxane'],
            'fluoropolymer': ['氟', 'ptfe', 'fluor', 'pvdf', 'perfluor', '全氟'],
            'hydrogel': ['peg', 'pva', 'hema', '水凝胶', 'hydrogel', '丙烯酰胺', '丙烯酸', '壳聚糖', '海藻'],
            'zwitterionic': ['sbma', 'cbma', 'mpc', '甜菜碱', '两性', 'zwitterion', '磷酸胆碱'],
            'self_polishing': ['自抛光', 'self_polish', '丙烯酸铜', '丙烯酸锌', '水解', '硅酯'],
            'bioinspired': ['仿', '多巴胺', 'dopa', 'pda', 'slips', '鲨鱼', '荷叶', 'mussel', 'bioinspir'],
            'nanocomposite': ['nano', '纳米', 'zno', 'ag', 'sio2', 'tio2', '石墨烯', 'graphene', 'cu2o', '复合'],
            'smart': ['nipam', 'pnipam', '响应', '温度', 'ph响应', '光响应', '智能', 'slips'],
        }
        for cls, hints in class_hints.items():
            for h in hints:
                if h in name_lower:
                    generic = _CLASS_GENERIC_NAMES.get(cls)
                    if generic and generic in SYNTHESIS_DATABASE:
                        return SYNTHESIS_DATABASE[generic]

    return None


def generate_reagent_table(route: dict) -> pd.DataFrame:
    """Return a reagent DataFrame from a route dict."""
    rows = []
    for i, r in enumerate(route.get('reagents', []), 1):
        rows.append({
            '序号': i,
            '试剂名称': r.get('name', ''),
            '角色/用途': r.get('role', ''),
            '用量': r.get('amount', ''),
        })
    return pd.DataFrame(rows)


def format_synthesis_report(route: dict) -> str:
    """Return a comprehensive Markdown report for one synthesis route."""
    lines: List[str] = []
    name = route.get('name', '未知材料')
    cls_cn = _CLASS_CN.get(route.get('class', ''), route.get('class', ''))

    lines.append(f"# 🧪 {name} — 合成制备路线")
    lines.append("")
    lines.append(f"**材料类别：** {cls_cn}　|　"
                 f"**合成方法：** {route.get('method', '')}　|　"
                 f"**难度：** {route.get('difficulty', '')}　|　"
                 f"**可放大性：** {route.get('scalability', '')}　|　"
                 f"**成本：** {route.get('cost_level', '')}")
    lines.append("")

    # Reagents table
    lines.append("## 📋 试剂清单")
    lines.append("")
    lines.append("| 序号 | 试剂名称 | 角色/用途 | 用量 |")
    lines.append("|:----:|----------|----------|------|")
    for i, r in enumerate(route.get('reagents', []), 1):
        lines.append(f"| {i} | {r['name']} | {r['role']} | {r['amount']} |")
    lines.append("")

    # Conditions
    cond = route.get('conditions', {})
    lines.append("## ⚗️ 反应条件")
    lines.append("")
    lines.append(f"| 参数 | 条件 |")
    lines.append(f"|------|------|")
    labels = [('temperature', '温度'), ('time', '时间'), ('atmosphere', '气氛'),
              ('solvent', '溶剂'), ('catalyst', '催化剂'), ('other', '其他')]
    for key, label in labels:
        val = cond.get(key, '-')
        if val:
            lines.append(f"| {label} | {val} |")
    lines.append("")

    # Steps
    lines.append("## 🔬 操作步骤")
    lines.append("")
    for i, step in enumerate(route.get('steps', []), 1):
        lines.append(f"**{i}.** {step}")
        lines.append("")

    # Characterization
    lines.append("## 📊 表征与验证")
    lines.append("")
    for item in route.get('characterization', []):
        lines.append(f"- {item}")
    lines.append("")

    # Safety
    lines.append("## ⚠️ 安全注意事项")
    lines.append("")
    for item in route.get('safety', []):
        lines.append(f"- 🛡️ {item}")
    lines.append("")

    # References
    lines.append("## 📚 参考文献")
    lines.append("")
    for i, ref in enumerate(route.get('references', []), 1):
        lines.append(f"{i}. {ref}")
    lines.append("")

    return "\n".join(lines)


# ================================================================
#  FLOWCHART GENERATION
# ================================================================

# Color scheme for step types
_STEP_COLORS = {
    'prep':   '#4FC3F7',   # light blue – preparation
    'react':  '#FF8A65',   # orange     – reaction
    'post':   '#81C784',   # green      – post-processing
    'char':   '#BA68C8',   # purple     – characterization
}

_STEP_LABELS = {
    'prep':  '准备',
    'react': '反应',
    'post':  '后处理',
    'char':  '表征',
}


def _classify_step(text: str) -> str:
    """Heuristic classification of a synthesis step."""
    t = text.lower()
    char_kw = ['ftir', 'xps', 'nmr', 'gpc', 'sem', 'tem', 'xrd', 'dsc',
               'tga', 'afm', 'uv-vis', '接触角', '表征', '椭偏', 'spr',
               'qcm', '测量', '测试', '验证', '确认', '抗菌测试', '失重',
               '力学', '拉伸', '硬度']
    for kw in char_kw:
        if kw in t:
            return 'char'
    post_kw = ['冷却', '沉淀', '过滤', '洗涤', '干燥', '透析', '冻干',
               '纯化', '脱模', '浸泡', '旋蒸', '离心', '萃取', '蒸发',
               '涂覆', '旋涂', '喷涂', '刷涂', '成膜', '流延']
    for kw in post_kw:
        if kw in t:
            return 'post'
    react_kw = ['反应', '加热', '聚合', '交联', '搅拌', '固化', '回流',
                '水解', '油浴', '紫外', 'uv', '照射', '冻融脱气', '陈化',
                '引发', '滴加', '注入', '煮沸', '曝光', '退火', '烧结',
                '煅烧']
    for kw in react_kw:
        if kw in t:
            return 'react'
    return 'prep'


def _safe_plot_text(text: str) -> str:
    """Replace Unicode subscript/superscript characters with ASCII
    equivalents so matplotlib doesn't warn about missing glyphs."""
    _sub_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    _sup_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    return text.translate(_sub_map).translate(_sup_map)


def _wrap_text(text: str, width: int = 18) -> str:
    """Wrap long Chinese/mixed text for display in flowchart boxes."""
    text = _safe_plot_text(text)
    lines = []
    line = ''
    for ch in text:
        line += ch
        # Approximate width: CJK chars ~2 cols, ASCII ~1
        w = sum(2 if ord(c) > 0x7F else 1 for c in line)
        if w >= width:
            lines.append(line)
            line = ''
    if line:
        lines.append(line)
    return '\n'.join(lines)


def generate_synthesis_flowchart(route: dict) -> plt.Figure:
    """Create a vertical flowchart figure for the synthesis procedure.

    Steps are drawn as rounded-rectangle boxes connected by arrows.
    Box colour encodes step category (preparation / reaction /
    post-processing / characterisation).

    Returns a matplotlib Figure ready for ``fig.savefig()`` or display
    in Gradio.
    """
    steps = route.get('steps', [])
    if not steps:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, '暂无步骤数据', ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig

    n = len(steps)
    box_h = 0.9           # box height in data coords
    gap   = 0.55          # vertical gap between boxes
    total_h = n * box_h + (n - 1) * gap + 1.5  # extra for title + legend
    fig_h = max(6, total_h * 0.75)
    fig_w = 8.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-total_h - 0.3, 1.8)
    ax.set_axis_off()

    # Title
    name = _safe_plot_text(route.get('name', ''))
    method = _safe_plot_text(route.get('method', ''))
    ax.text(5, 1.2, f'{name}', fontsize=15, fontweight='bold',
            ha='center', va='center')
    ax.text(5, 0.5, f'合成方法: {method}', fontsize=10,
            ha='center', va='center', color='#555')

    x_center = 5.0
    box_w = 8.0  # box width

    for i, step_text in enumerate(steps):
        cat = _classify_step(step_text)
        color = _STEP_COLORS[cat]

        y_top = -(i * (box_h + gap))
        y_center = y_top - box_h / 2

        # Rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, y_top - box_h),
            box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='#333',
            linewidth=1.2, alpha=0.88,
        )
        ax.add_patch(rect)

        # Step number badge
        badge_x = x_center - box_w / 2 + 0.45
        badge_y = y_center
        ax.add_patch(plt.Circle((badge_x, badge_y), 0.28,
                                facecolor='white', edgecolor='#333',
                                linewidth=1, zorder=5))
        ax.text(badge_x, badge_y, str(i + 1), fontsize=10,
                fontweight='bold', ha='center', va='center', zorder=6)

        # Step text (wrapped)
        wrapped = _wrap_text(step_text, width=32)
        ax.text(x_center + 0.3, y_center, wrapped,
                fontsize=8.5, ha='center', va='center',
                linespacing=1.35)

        # Arrow to next box
        if i < n - 1:
            arr_y_start = y_top - box_h
            arr_y_end   = y_top - box_h - gap
            ax.annotate('', xy=(x_center, arr_y_end),
                        xytext=(x_center, arr_y_start),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                        lw=1.8, connectionstyle='arc3'))

    # Legend
    legend_y = -(n * (box_h + gap)) - 0.2
    legend_items = [('prep', '准备'), ('react', '反应'),
                    ('post', '后处理'), ('char', '表征')]
    for j, (cat, label) in enumerate(legend_items):
        lx = 1.5 + j * 2.2
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx, legend_y - 0.25), 0.5, 0.35,
            boxstyle="round,pad=0.05",
            facecolor=_STEP_COLORS[cat], edgecolor='#333',
            linewidth=0.8, alpha=0.85))
        ax.text(lx + 0.7, legend_y - 0.07, label, fontsize=9,
                va='center')

    fig.tight_layout()
    return fig


# ================================================================
#  CONVENIENCE: list all entries
# ================================================================

def list_all_routes() -> List[dict]:
    """Return a list of all registered synthesis routes."""
    return list(SYNTHESIS_DATABASE.values())


def list_routes_by_class(cls: str) -> List[dict]:
    """Return routes belonging to a material class."""
    cls_lower = cls.strip().lower()
    return [e for e in SYNTHESIS_DATABASE.values()
            if e.get('class', '').lower() == cls_lower]


def get_available_materials() -> List[str]:
    """Return sorted list of all material names in the database."""
    return sorted(SYNTHESIS_DATABASE.keys())


def get_database_summary() -> pd.DataFrame:
    """Return a summary DataFrame of the entire database."""
    rows = []
    for entry in SYNTHESIS_DATABASE.values():
        rows.append({
            '材料名称': entry['name'],
            '类别': _CLASS_CN.get(entry.get('class', ''), entry.get('class', '')),
            '合成方法': entry.get('method', ''),
            '难度': entry.get('difficulty', ''),
            '可放大性': entry.get('scalability', ''),
            '成本': entry.get('cost_level', ''),
            '步骤数': len(entry.get('steps', [])),
            '试剂数': len(entry.get('reagents', [])),
        })
    return pd.DataFrame(rows)
