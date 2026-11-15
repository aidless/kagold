"""集成工具：用 OOF 做加权融合。

判据很朴素但很关键：
- 只用 OOF 做权重（任何用 LB 反馈调权重的做法，本质是在 public LB 上过拟合）；
- 权重搜索在**参考 metric** 上做，和门禁同一套口径；
- 输出每折分数，融合后的 std 不能比单模型差太多。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .base import ToolError, obj, tool
from ..gates import reference as ref
from ..runtime import RunContext


def _score(canon: str, y, p) -> float:
    return float(ref.score(canon, y, p))


@tool("ensemble",
      "用 OOF 预测做加权融合（mean / greedy / rank_mean），权重在参考 metric 上搜索。"
      "输出 oof/blend.csv + oof/ensemble.json。融合后必须再跑 gate_oof 验证覆盖与分数。",
      obj({"metric": {"type": "string"},
           "oof_csv": {"type": "string", "description": "默认 oof/oof.csv"},
           "train_csv": {"type": "string", "description": "可选，默认自动找"},
           "method": {"type": "string", "enum": ["greedy", "mean", "rank_mean"], "description": "默认 greedy"},
           "models": {"type": "array", "items": {"type": "string"},
                      "description": "只用这些 pred 列；默认全部"},
           "id_col": {"type": "string"}, "target_col": {"type": "string"},
           "fold_col": {"type": "string"}}, ["metric"]))
def ensemble(ctx: RunContext, a: dict) -> dict:
    canon = ref.canon(a["metric"])
    if not canon:
        raise ToolError(f"metric {a['metric']!r} 未登记")
    gib = ref.greater_is_better(canon)
    oof_p = ctx.resolve(a.get("oof_csv") or "oof/oof.csv")
    if not oof_p.exists():
        raise ToolError(f"缺少 {oof_p}")
    train_p = ctx.resolve(a["train_csv"]) if a.get("train_csv") else None
    if train_p is None:
        cands = list(ctx.input_dir.rglob("train*.csv"))
        if not cands:
            raise ToolError("找不到训练表，请传 train_csv")
        train_p = cands[0]
    oof = pd.read_csv(oof_p, low_memory=False)
    tr = pd.read_csv(train_p, low_memory=False)
    idc = a.get("id_col") or next((c for c in ("id", "row_id", "ID", "Id") if c in oof.columns and c in tr.columns), None)
    tc = a.get("target_col") or next((c for c in ("target", "label", "y", "class") if c in tr.columns), None)
    fc = a.get("fold_col") or "fold"
    if not idc or not tc or fc not in tr.columns:
        raise ToolError("无法确定 id/target/fold 列，请显式指定")
    cols = a.get("models") or [c for c in oof.columns if c.lower().startswith("pred")]
    cols = [c for c in cols if c in oof.columns]
    if len(cols) < 2:
        raise ToolError(f"至少需要 2 个 pred 列才能融合，当前 {cols}")
    m = tr[[idc, tc, fc]].merge(oof[[idc] + cols], on=idc, how="inner")
    y = m[tc].values.astype(float)
    folds = m[fc].values
    uniq = sorted(pd.unique(folds).tolist())

    P = m[cols].values.astype(float)
    singles = {c: _score(canon, y, P[:, i]) for i, c in enumerate(cols)}
    best_single = max(singles, key=lambda c: singles[c] if gib else -singles[c])

    method = a.get("method") or "greedy"
    if method == "mean":
        w = np.ones(len(cols)) / len(cols)
    elif method == "rank_mean":
        if canon not in ("auc", "accuracy", "f1", "mcc", "qwk"):
            raise ToolError(f"rank_mean 会破坏数值尺度，只适用于排名型指标；当前 {canon} 请用 greedy/mean")
        R = np.column_stack([pd.Series(P[:, i]).rank(pct=True).values for i in range(len(cols))])
        w = np.ones(len(cols)) / len(cols)
        P = R
    else:
        w = _greedy(canon, y, P, gib)

    blend = P @ w
    total = _score(canon, y, blend)
    per_fold = [_score(canon, y[folds == f], blend[folds == f]) for f in uniq]
    out = pd.DataFrame({idc: m[idc].values, "pred_blend": blend})
    bp = ctx.oof_dir / "blend.csv"
    out.to_csv(bp, index=False)
    meta: dict[str, Any] = {
        "method": method, "metric": canon, "models": cols,
        "singles": {k: round(v, 8) for k, v in singles.items()},
        "weights": {c: round(float(w), 4) for c, w in zip(cols, w)},
        "blend_score": round(total, 8),
        "blend_fold_scores": [round(v, 8) for v in per_fold],
        "blend_std": round(float(np.std(per_fold, ddof=1)), 8),
        "best_single": {"model": best_single, "score": round(singles[best_single], 8)},
        "gain_vs_best_single": round(float(
            (total - singles[best_single]) if gib else (singles[best_single] - total)), 8),
        "output": ctx.rel(bp),
    }
    (ctx.oof_dir / "ensemble.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    from .gates import _auto_advance
    phase = _auto_advance(ctx, "ENSEMBLE")
    meta["phase"] = phase
    return {"ok": True, "data": meta,
            "next": f"用 gate_oof(metric={canon}, oof_csv='{ctx.rel(bp)}', pred 列名视为 pred_blend) 验收；"
                    f"注意 gate_oof 需要 id 列，已带上。"}


def _greedy(canon: str, y: np.ndarray, P: np.ndarray, gib: bool, iters: int = 6) -> np.ndarray:
    """坐标上升：从一个等权起点出发，逐个模型在其权重上做一维搜索。"""
    n, k = P.shape
    w = np.ones(k) / k
    best = _score(canon, y, P @ w)
    for _ in range(iters):
        improved = False
        for i in range(k):
            for cand in np.linspace(0, 1, 21):
                w2 = w.copy()
                rest = 1.0 - cand
                others = np.delete(w, i)
                total_others = others.sum()
                if total_others > 0:
                    w2[:] = 0
                    w2[i] = cand
                    others_scaled = others / total_others * rest
                    j = 0
                    for idx in range(k):
                        if idx != i:
                            w2[idx] = others_scaled[j]
                            j += 1
                else:
                    w2[i] = cand
                    w2[(i + 1) % k] = 1 - cand
                s = _score(canon, y, P @ w2)
                if (s > best) if gib else (s < best):
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w
