"""
P0-2 域适用性评估 (Applicability Domain, Williams plot 思路)
============================================================
背景:
  训练集只有 84 个真实分子(其余为增强变体)。模型对落在训练特征空间之外的
  陌生输入会"外推", 给出毫无依据的高防污效率——这正是假精确的另一来源。

方法:
  训练集 28 维描述符 → 标准化 → 马氏距离 (Mahalanobis distance)。
  以训练集马氏距离的 95% 分位为阈值 (Williams plot 标准做法)。
  新样本距离 > 阈值 → in_domain=False。

参考: FOULING_RELEASE_MLOPS 的 Williams plot 域适用性判定。
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


class ApplicabilityDomain:
    """基于马氏距离的适用域判定。"""

    def __init__(self, q=95.0):
        self.q = q
        self.scaler = None
        self.mean_ = None
        self.cov_pinv_ = None
        self.threshold_ = None
        self.n_train_ = 0
        self.fitted = False

    def fit(self, X):
        """X: (n_samples, n_features) 训练集特征矩阵。"""
        X = np.asarray(X, dtype=float)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.mean_ = Xs.mean(axis=0)
        cov = np.cov(Xs, rowvar=False)
        # 描述符间存在共线(见 CODE_REVIEW), 用伪逆保证数值稳定
        self.cov_pinv_ = np.linalg.pinv(cov)
        d = self._distances(Xs)
        self.threshold_ = float(np.percentile(d, self.q))
        self.n_train_ = X.shape[0]
        self.fitted = True
        return self

    def _distances(self, Xs):
        diff = Xs - self.mean_
        m2 = np.einsum("ij,jk,ik->i", diff, self.cov_pinv_, diff)
        return np.sqrt(np.clip(m2, 0.0, None))

    def check(self, x):
        """
        x: (1, n_features) 或 (n_features,)
        返回 dict: in_domain / distance / threshold / ratio
        """
        if not self.fitted:
            return {"in_domain": True, "distance": None, "threshold": None,
                    "ratio": None, "ad_fitted": False}
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        Xs = self.scaler.transform(x)
        d = float(self._distances(Xs)[0])
        thr = float(self.threshold_)
        return {
            "in_domain": bool(d <= thr),
            "distance": round(d, 3),
            "threshold": round(thr, 3),
            "ratio": round(d / thr, 3) if thr else None,
            "ad_fitted": True,
        }

    # ---------- 序列化 ----------
    def to_dict(self):
        if not self.fitted:
            return {}
        return {
            "q": self.q,
            "mean_": self.mean_,
            "cov_pinv_": self.cov_pinv_,
            "threshold_": self.threshold_,
            "n_train_": self.n_train_,
            "scaler_mean_": self.scaler.mean_,
            "scaler_scale_": self.scaler.scale_,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(q=float(d.get("q", 95.0)))
        if not d or "mean_" not in d:
            return obj
        obj.mean_ = np.asarray(d["mean_"], dtype=float)
        obj.cov_pinv_ = np.asarray(d["cov_pinv_"], dtype=float)
        obj.threshold_ = float(d["threshold_"])
        obj.n_train_ = int(d.get("n_train_", 0))
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.asarray(d["scaler_mean_"], dtype=float)
        obj.scaler.scale_ = np.asarray(d["scaler_scale_"], dtype=float)
        obj.scaler.n_features_in_ = obj.mean_.shape[0]
        obj.fitted = True
        return obj


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 28))
    ad = ApplicabilityDomain().fit(X)
    print("阈值(95%分位):", round(ad.threshold_, 3))
    print("训练集内样本:", ad.check(X[0]))
    print("离群样本:", ad.check(rng.normal(size=28) * 8))
