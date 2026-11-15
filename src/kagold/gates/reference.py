"""参考度量实现 —— 门禁用它来"独立复算"，不采信 agent 自己报的分数。

为什么必须独立实现：
- 竞赛 metric 的细节（clip、log1p、多分类平均方式）是第一名和第二名的主要差距；
- 如果门禁直接用 agent 写的 score 函数打分，那 agent 写个返回 0.99 的函数就能"通过"。

这里每个实现都对应 Kaggle 常见口径，名字与方向(GREATER_IS_BETTER)显式登记。
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-15


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim > 2:
        raise ValueError("metric 只接受 1D/2D 数组")
    return a


def _binary_p(y_pred: np.ndarray) -> np.ndarray:
    p = y_pred
    if p.ndim == 2 and p.shape[1] == 2:
        p = p[:, 1]
    p = np.clip(p.ravel(), EPS, 1 - EPS)
    return p


# ---------------- 回归 ----------------
def rmse(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mse(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    return float(np.mean((t - p) ** 2))


def mae(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    return float(np.mean(np.abs(t - p)))


def rmsle(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    t = np.log1p(np.clip(t, -1 + 1e-9, None))
    p = np.log1p(np.clip(p, -1 + 1e-9, None))
    return float(np.sqrt(np.mean((t - p) ** 2)))


def brier(y_true, y_pred) -> float:
    """Brier score = mean((p - y)^2)，越小越好。"""
    t, p = _arr(y_true).ravel().astype(float), _arr(y_pred)
    p = (p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()).astype(float)
    return float(np.mean((p - t) ** 2))


def mape(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    denom = np.where(np.abs(t) < 1e-9, 1e-9, np.abs(t))
    return float(np.mean(np.abs(t - p) / denom))


def smape(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    denom = np.abs(t) + np.abs(p)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return float(np.mean(2 * np.abs(t - p) / denom))


def r2(y_true, y_pred) -> float:
    t, p = _arr(y_true).ravel(), _arr(y_pred).ravel()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / (ss_tot if ss_tot > 0 else EPS))


# ---------------- 分类 ----------------
def logloss(y_true, y_pred) -> float:
    t = _arr(y_true)
    p = _arr(y_pred)
    if p.ndim == 2 and p.shape[1] > 2:  # 多分类概率矩阵
        pi = np.clip(p, EPS, 1 - EPS)
        rows = np.arange(t.shape[0])
        return float(-np.mean(np.log(pi[rows, t.astype(int).ravel()])))
    p = _binary_p(p)
    t = t.ravel()
    p = np.where(t > 0.5, p, 1 - p)
    return float(-np.mean(np.log(np.clip(p, EPS, 1 - EPS))))


def _binary_pred(y_pred) -> np.ndarray:
    p = _arr(y_pred)
    if p.ndim == 2:
        return p.argmax(axis=1)
    return (p.ravel() >= 0.5).astype(int)


def accuracy(y_true, y_pred) -> float:
    t = _arr(y_true).ravel()
    p = _binary_pred(y_pred)
    return float(np.mean(t.astype(int) == p))


def f1(y_true, y_pred) -> float:
    t = _arr(y_true).ravel().astype(int)
    p = _binary_pred(y_pred)
    tp = int(np.sum((t == 1) & (p == 1)))
    fp = int(np.sum((t == 0) & (p == 1)))
    fn = int(np.sum((t == 1) & (p == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


def auc(y_true, y_pred) -> float:
    t = _arr(y_true).ravel().astype(int)
    s = _arr(y_pred)
    s = (s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()).astype(float)
    n_pos = int(np.sum(t == 1))
    n_neg = int(np.sum(t == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：取平均秩
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mcc(y_true, y_pred) -> float:
    t = _arr(y_true).ravel().astype(int)
    p = _binary_pred(y_pred)
    tp = np.sum((t == 1) & (p == 1)); tn = np.sum((t == 0) & (p == 0))
    fp = np.sum((t == 0) & (p == 1)); fn = np.sum((t == 1) & (p == 0))
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(((tp * tn) - (fp * fn)) / den) if den else 0.0


def qwk(y_true, y_pred) -> float:
    t = _arr(y_true).ravel().astype(int)
    p = _binary_pred(y_pred) if _arr(y_pred).ndim < 2 else _arr(y_pred).ravel().astype(int)
    labels = np.unique(np.concatenate([t, p]))
    idx = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(t, p):
        cm[idx[a], idx[b]] += 1
    hist_t = cm.sum(axis=1)
    hist_p = cm.sum(axis=0)
    expected = np.outer(hist_t, hist_p) / cm.sum()
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / max(1, (k - 1) ** 2)
    num = np.sum(w * cm)
    den = np.sum(w * expected)
    return float(1 - num / den) if den else 0.0


# ---------------- 注册表 ----------------
REGISTRY: dict[str, dict] = {
    "rmse": {"fn": rmse, "greater_is_better": False, "kind": "regression"},
    "mse": {"fn": mse, "greater_is_better": False, "kind": "regression"},
    "mae": {"fn": mae, "greater_is_better": False, "kind": "regression"},
    "rmsle": {"fn": rmsle, "greater_is_better": False, "kind": "regression"},
    "mape": {"fn": mape, "greater_is_better": False, "kind": "regression"},
    "smape": {"fn": smape, "greater_is_better": False, "kind": "regression"},
    "r2": {"fn": r2, "greater_is_better": True, "kind": "regression"},
    "brier": {"fn": brier, "greater_is_better": False, "kind": "binary"},
    "logloss": {"fn": logloss, "greater_is_better": False, "kind": "binary"},
    "auc": {"fn": auc, "greater_is_better": True, "kind": "binary"},
    "roc_auc": {"fn": auc, "greater_is_better": True, "kind": "binary"},
    "accuracy": {"fn": accuracy, "greater_is_better": True, "kind": "binary"},
    "f1": {"fn": f1, "greater_is_better": True, "kind": "binary"},
    "mcc": {"fn": mcc, "greater_is_better": True, "kind": "binary"},
    "qwk": {"fn": qwk, "greater_is_better": True, "kind": "ordinal"},
}

ALIASES = {
    "rootmeansquarederror": "rmse", "rootmeansquaredlogerror": "rmsle",
    "meanabsoluteerror": "mae", "meansquarederror": "mse",
    "meanabsolutepercent误差": "mape", "symmetricmeanabsolutepercentageerror": "smape",
    "rsquared": "r2", "r^2": "r2", "logloss": "logloss", "multiclasslogloss": "logloss",
    "brierscore": "brier", "brierskillscore": "brier",
    "binarylogloss": "logloss", "roc": "auc", "rocauc": "auc", "areaunderroccurve": "auc",
    "categoraccuracy": "accuracy", "f1score": "f1", "fscore": "f1",
    "quadraticweightedkappa": "qwk", "kappa": "qwk", "matthewscorrelationcoefficient": "mcc",
}


def canon(name: str) -> str | None:
    if not name:
        return None
    key = "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch in "^")
    if key in REGISTRY:
        return key
    if key in ALIASES:
        return ALIASES[key]
    for k in REGISTRY:
        if key.endswith(k) or key.startswith(k):
            return k
    return None


def score(name: str, y_true, y_pred) -> float:
    c = canon(name)
    if not c:
        raise KeyError(f"未登记的 metric：{name}")
    return REGISTRY[c]["fn"](y_true, y_pred)


def greater_is_better(name: str) -> bool:
    c = canon(name)
    return bool(REGISTRY[c]["greater_is_better"]) if c else True


def synthetic(kind: str, n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """生成一组"有确定性"的合成数据，用于比对 agent 的 metric 实现。"""
    rng = np.random.default_rng(seed)
    if kind == "regression":
        t = rng.normal(5, 2, n)
        return t, t + rng.normal(0, 0.7, n)
    if kind == "binary":
        t = rng.integers(0, 2, n)
        return t, np.clip(rng.beta(2 + 2 * t, 2 + 2 * (1 - t)), 0.001, 0.999)
    if kind == "ordinal":
        t = rng.integers(0, 4, n)
        return t, np.clip(t + rng.normal(0, 0.9, n), 0, 3)
    raise ValueError(kind)
