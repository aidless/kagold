"""门禁 3：OOF 评分（out-of-fold gate）—— 集成的唯一货币。

判据：
1. 覆盖：train 的每一行在 OOF 里恰好出现一次（不漏、不重、不串行）；
2. 有限：没有 NaN / inf；
3. 可复算：用**参考实现**（不是 agent 的 metric.py）按 fold 列重算每折分数与总体分数；
4. 有离散度：记录每折分数的 mean/std（std 太大说明划分不稳，不能拿去选模型）；
5. 多模型：`pred` 或 `pred_<name>` 每列都算一遍，产出相关系数与最优者。

落盘：oof/score.json（供集成与提交门禁读取）
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import RunContext
from . import reference as ref


def _find(paths: list, name: str) -> Any:
    for p in paths:
        if p and p.name.lower() == name.lower():
            return p
    return None


def run_oof_gate(ctx: RunContext, metric_name: str, oof_csv: str = "oof/oof.csv",
                 train_csv: str | None = None, id_col: str | None = None,
                 target_col: str | None = None, fold_col: str = "fold") -> dict:
    ev: dict[str, Any] = {"gate": "oof", "oof_csv": oof_csv, "metric": metric_name,
                          "problems": [], "models": {}}

    def finish(passed: bool, reason: str) -> dict:
        ev["passed"] = passed
        ev["reason"] = reason
        ctx.store.record_gate(ctx.run_id, "oof", passed, ev)
        _dump(ctx, "oof", "gate_oof.json", ev)
        return {"passed": passed, "reason": reason, "evidence": ev}

    canonical = ref.canon(metric_name)
    if not canonical:
        return finish(False, f"metric {metric_name!r} 未登记，无法复算")
    ev["canonical"] = canonical
    gib = ref.greater_is_better(canonical)

    oof_path = ctx.resolve(oof_csv)
    if not oof_path.exists():
        return finish(False, f"缺少 {oof_csv}；OOF 文件至少要包含 id、pred（或 pred_<model>）列")
    train_path = ctx.resolve(train_csv) if train_csv else _find(list(ctx.input_dir.rglob("train*.csv")), "train.csv")
    if train_path is None or not train_path.exists():
        return finish(False, "找不到训练表；请传 train_csv 或把 train.csv 放进 input/")

    try:
        oof = pd.read_csv(oof_path, low_memory=False)
        tr = pd.read_csv(train_path, low_memory=False)
    except Exception as e:  # noqa: BLE001
        return finish(False, f"读取失败：{e}")

    idc = id_col or next((c for c in ("id", "ID", "Id", "row_id", "Id") if c in oof.columns and c in tr.columns), None)
    if not idc:
        return finish(False, f"无法确定 id 列；oof={list(oof.columns)[:8]} train={list(tr.columns)[:8]}")
    tc = target_col or next((c for c in ("target", "label", "y", "Target", "class") if c in tr.columns), None)
    if not tc:
        return finish(False, "无法确定 target 列，请用 target_col 指定")
    if fold_col not in tr.columns:
        return finish(False, f"训练表缺少 {fold_col!r} 列（先过 fold 门禁）")
    ev.update({"id_col": idc, "target_col": tc, "fold_col": fold_col,
               "train_rows": int(len(tr)), "oof_rows": int(len(oof))})

    # ---- 覆盖检查 ----
    if oof[idc].duplicated().any():
        ev["problems"].append(f"OOF 中 id 重复 {int(oof[idc].duplicated().sum())} 行")
    tr_ids, oof_ids = set(tr[idc].tolist()), set(oof[idc].tolist())
    missing, extra = tr_ids - oof_ids, oof_ids - tr_ids
    if missing:
        ev["problems"].append(f"OOF 缺少 {len(missing)} 个训练行 id（例如 {list(missing)[:3]}）")
    if extra:
        ev["problems"].append(f"OOF 多出 {len(extra)} 个不在 train 里的 id（例如 {list(extra)[:3]}）")

    merged = tr[[idc, tc, fold_col]].merge(oof, on=idc, how="inner")
    ev["merged_rows"] = int(len(merged))
    if len(merged) < 0.999 * len(tr):
        ev["problems"].append(f"合并后只剩 {len(merged)}/{len(tr)} 行，覆盖率不足 99.9%")

    pred_cols = [c for c in oof.columns if c.lower().startswith("pred")]
    if not pred_cols:
        return finish(False, "OOF 里没有 pred / pred_<model> 列")
    ev["pred_columns"] = pred_cols

    y = merged[tc].values
    folds = merged[fold_col].values
    uniq = sorted(pd.unique(folds).tolist())
    ev["folds"] = [str(u) for u in uniq]

    for col in pred_cols:
        p = merged[col].values.astype(float)
        info: dict[str, Any] = {"nan": int(np.isnan(p).sum()), "inf": int(np.isinf(p).sum())}
        if info["nan"] or info["inf"]:
            ev["problems"].append(f"{col}: 含 {info['nan']} 个 NaN / {info['inf']} 个 inf")
            ev["models"][col] = info
            continue
        per_fold = []
        for f in uniq:
            m = folds == f
            if m.sum() < 2:
                continue
            try:
                per_fold.append(float(ref.score(canonical, y[m], p[m])))
            except Exception as e:  # noqa: BLE001
                ev["problems"].append(f"{col} 折 {f} 打分失败：{e}")
        if len(per_fold) < 2:
            ev["problems"].append(f"{col}: 有效折数不足 2")
            continue
        try:
            overall = float(ref.score(canonical, y, p))
        except Exception as e:  # noqa: BLE001
            ev["problems"].append(f"{col} 总体打分失败：{e}")
            continue
        info.update({
            "overall": round(overall, 8),
            "fold_scores": [round(v, 8) for v in per_fold],
            "mean": round(float(np.mean(per_fold)), 8),
            "std": round(float(np.std(per_fold, ddof=1)), 8),
            "n_folds": len(per_fold),
        })
        info["fold_spread_flag"] = bool(abs(np.mean(per_fold) - overall) > 3 * (np.std(per_fold, ddof=1) or 1e-12) + 1e-9)
        ev["models"][col] = info

    scored = {k: v for k, v in ev["models"].items() if "overall" in v}
    if len(scored) >= 2:
        try:
            cols = list(scored)
            corr = merged[cols].astype(float).corr().round(4)
            ev["correlation"] = {c: {k2: float(corr.loc[c, k2]) for k2 in cols} for c in cols}
        except Exception:
            pass
    if scored:
        best = max(scored.items(), key=lambda kv: kv[1]["overall"] if gib else -kv[1]["overall"])
        ev["best"] = {"column": best[0], **{k: best[1][k] for k in ("overall", "mean", "std")}}
        ctx.store.kv_set(ctx.run_id, "oof_best", ev["best"])
        prev = ctx.store.kv_get(ctx.run_id, "baseline_score")
        if prev is None:
            ctx.store.kv_set(ctx.run_id, "baseline_score", ev["best"]["overall"])
            ev["baseline_set"] = True
        else:
            delta = (ev["best"]["overall"] - prev) if gib else (prev - ev["best"]["overall"])
            ev["vs_baseline"] = round(float(delta), 8)

    passed = not ev["problems"] and bool(scored)
    reason = (f"OOF 通过：{len(scored)} 个模型，最优 {ev.get('best', {}).get('overall')} "
              f"(±{ev.get('best', {}).get('std')})" if passed
              else f"OOF 不合格：{len(ev['problems'])} 个问题")
    return finish(passed, reason)


def _dump(ctx: RunContext, stem: str, fname: str, ev: dict) -> None:
    try:
        ctx.oof_dir.mkdir(parents=True, exist_ok=True)
        (ctx.oof_dir / fname).write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
        if stem == "oof" and "best" in ev:
            (ctx.oof_dir / "score.json").write_text(
                json.dumps({"best": ev["best"], "models": ev["models"],
                            "correlation": ev.get("correlation"), "metric": ev.get("canonical")},
                           ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
