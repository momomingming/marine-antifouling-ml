"""
P0-1 预测不确定性量化 (LOGO 折外集成)
=====================================
背景:
  experiments/logo_cv.py 已证实——随机 KFold R²=0.703 是信息泄漏虚高,
  LOGO-by-SMILES 诚实 R²=-0.594。模型在独立分子上泛化能力很差。
  此时若继续返回单点"精确值"(如 68.24), 是**假精确**, 会误导使用者。

本模块:
  用 84 组 SMILES 做 Leave-One-Group-Out, 训练 84 个折外 XGBoost 模型。
  推理时新样本走全部折外模型 → 预测值集合:
    均值    = 点预测
    2.5/97.5 分位 = 95% 置信区间
    标准差  = 不确定度
  同时调用 domain_applicability 判定输入是否落在训练分子特征空间内。

设计约束:
  - 若集成模型文件缺失(如服务器尚未部署), available=False, 调用方应优雅降级, 绝不崩溃。
  - 不引入新重依赖: numpy + xgboost + sklearn(StandardScaler)。
"""
import os
import sys
import glob
import pickle
import numpy as np

# 28 个基础描述符 (与 data/raw/dataset.csv 特征列、app.py BASE_DESCRIPTOR_NAMES 一致)
BASE_FEATURE_NAMES = [
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "RingCount",
    "AromaticRings", "HeavyAtoms", "FractionCSP3", "NumF", "NumCl",
    "NumBr", "NumN", "NumO", "NumS", "NumSi", "NumP", "HasCu",
    "HasZn", "HasAg", "HasTi", "ChargeDensity",
    "HydrophilicLipophilicBalance", "SurfaceEnergyEstimate",
    "ElasticModulusEstimate", "RoughnessPotential", "CrosslinkPotential",
]

# 仓库根 = src/ 的上一级
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENSEMBLE_DIR = os.path.join(_REPO_ROOT, "models", "logo_ensemble")


def _resolve_ensemble_dir():
    """
    按候选路径定位集成目录, 使本模块既能放在 src/ 也能放在 deployments/deploy_job/。
    优先级: 环境变量 MAF_ENSEMBLE_DIR > 存在 meta.pkl 的候选路径 > 默认路径。
    """
    env = os.environ.get("MAF_ENSEMBLE_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(_REPO_ROOT, "models", "logo_ensemble"),   # src/ 布局
        os.path.join(here, "..", "models", "logo_ensemble"),   # deployments/deploy_job/ → deployments/models
        os.path.join(here, "..", "..", "models", "logo_ensemble"),  # deployments/deploy_job/ → 仓库根/models
        os.path.join(here, "models", "logo_ensemble"),         # 同目录自带
    ]
    for c in cands:
        p = os.path.normpath(c)
        if os.path.exists(os.path.join(p, "meta.pkl")):
            return p
    return os.path.normpath(cands[0])

# 置信区间宽度 → 不确定度等级 (效率为 0~100 的百分比)
_W_LOW, _W_MID = 10.0, 25.0
_Z = 1.96  # 95% 置信区间对应的正态分位数


class LOGOEnsemble:
    """84 个 LOGO 折外 XGBoost 模型的集成预测器。"""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or _resolve_ensemble_dir()
        self.models = []
        self.meta = {}
        self.feature_names = list(BASE_FEATURE_NAMES)
        self._ad = None
        self._loaded = False
        self.load_error = None

    # ---------- 加载 ----------
    def load(self):
        """惰性加载折外模型与元数据。失败不抛异常, 置 available=False。"""
        if self._loaded:
            return self.available
        meta_path = os.path.join(self.model_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            self.load_error = f"集成元数据缺失: {meta_path}"
            self._loaded = True
            return False
        try:
            import xgboost as xgb
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
            self.feature_names = self.meta.get("feature_names", list(BASE_FEATURE_NAMES))
            paths = sorted(glob.glob(os.path.join(self.model_dir, "fold_*.ubj")))
            if not paths:
                paths = sorted(glob.glob(os.path.join(self.model_dir, "fold_*.json")))
            if not paths:
                self.load_error = f"折外模型缺失: {self.model_dir}/fold_*.ubj"
                self._loaded = True
                return False
            self.models = []
            for p in paths:
                m = xgb.XGBRegressor()
                m.load_model(p)
                self.models.append(m)
            # 域适用性
            try:
                from domain_applicability import ApplicabilityDomain
            except ImportError:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from domain_applicability import ApplicabilityDomain
            self._ad = ApplicabilityDomain.from_dict(self.meta.get("ad", {}))
        except Exception as e:  # 任何加载失败都降级, 不拖垮主流程
            self.load_error = f"{type(e).__name__}: {e}"
            self.models = []
        self._loaded = True
        return self.available

    @property
    def available(self):
        return len(self.models) > 0

    # ---------- 推理 ----------
    def _vector(self, desc):
        """从描述符字典构造 (1, n_features) 向量, 缺失特征报错由调用方捕获。"""
        missing = [f for f in self.feature_names if f not in desc]
        if missing:
            raise KeyError(f"描述符缺少字段: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return np.array([[float(desc[f]) for f in self.feature_names]], dtype=float)

    def predict(self, desc):
        """
        输入: desc —— 28 个基础描述符字典 (app.py compute_base_descriptors 的输出)
        返回: dict, 含 point/ci_low/ci_high/std/level/n_models + 域适用性字段
        """
        if not self.load() or not self.available:
            return {"available": False, "reason": self.load_error or "集成模型未加载"}

        x = self._vector(desc)
        # 折外模型是在「原始」特征上训练的: 树模型无需标准化, 且标准化会导致预测失真。
        # StandardScaler 仅用于适用域(马氏距离)判定, 不参与模型推理。
        preds = np.array([float(m.predict(x)[0]) for m in self.models], dtype=float)
        preds = np.clip(preds, 0.0, 100.0)

        point = float(np.mean(preds))
        q_lo = float(np.percentile(preds, 2.5))
        q_hi = float(np.percentile(preds, 97.5))
        std = float(np.std(preds))

        # 诚实性护栏: 84 个折外模型高度相关, 模型间分歧会严重低估真实误差
        # (实测: 集成 CI 仅 ±2, 而折外 RMSE 达 ±10 量级)。
        # 因此用折外实测 RMSE 给区间兜底, 避免输出"假精确"的窄区间。
        rmse = float(self.meta.get("logo_rmse") or 0.0)
        half_ens = (q_hi - q_lo) / 2.0
        half = max(half_ens, _Z * rmse) if rmse > 0 else half_ens
        lo = max(0.0, point - half)
        hi = min(100.0, point + half)
        width = hi - lo

        if width <= _W_LOW:
            level = "低"
        elif width <= _W_MID:
            level = "中"
        else:
            level = "高"

        out = {
            "available": True,
            "prediction": round(point, 2),
            "ci_low": round(lo, 2),
            "ci_high": round(hi, 2),
            "ci_width": round(width, 2),
            "std": round(std, 2),
            "n_models": len(preds),
            "uncertainty_level": level,
            "ensemble_ci": [round(q_lo, 2), round(q_hi, 2)],
            "ci_basis": ("折外残差兜底" if rmse > 0 and half > half_ens
                         else "模型间分歧"),
            "logo_rmse": round(rmse, 3) if rmse else None,
        }

        # 域适用性
        if self._ad is not None and self._ad.fitted:
            dom = self._ad.check(x)
            out.update(dom)
            if not dom["in_domain"]:
                out["uncertainty_level"] = "高"
                out["warning"] = "输入分子超出训练特征空间(不适用域), 预测结果仅供参考"
        return out


# 模块级单例, 避免重复加载 84 个模型
_ENSEMBLE = None


def get_ensemble(model_dir=None):
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = LOGOEnsemble(model_dir)
    return _ENSEMBLE


def predict_with_uncertainty(desc, model_dir=None):
    """便捷入口: 返回 (ok, result)。ok=False 时调用方应走原有单点预测逻辑。"""
    ens = get_ensemble(model_dir)
    res = ens.predict(desc)
    return bool(res.get("available")), res


if __name__ == "__main__":
    ens = LOGOEnsemble()
    print("模型目录:", ens.model_dir)
    print("可用:", ens.load(), "| 折外模型数:", len(ens.models), "| err:", ens.load_error)
