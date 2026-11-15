"""门禁 2：交叉验证划分（fold gate）。

判据（全部可失败）：
1. 训练表里必须存在 fold 列（单一真相源——OOF、集成、提交都从这一列推）；
2. 折数 k ∈ [3, 10]，每折样本数 ≥ max(20, 0.5% 总行数)；
3. 目标分布：分类看各类比例偏差 ≤ 0.03；回归看 |折均值-全局均值| ≤ 0.10×全局标准差；
4. 若给了 group 列：同一 group 不得跨折（这是最常见的泄漏）；
5. 若给了 id 列且重复：告警（近重复样本应同行或剔除）。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import RunContext


def _infer(df: pd.DataFrame, hint: str | None, candidates: list[str]) -> str | None:
    if hint and hint in df.columns:
        return hint
    for c in candidates:
        if c in df.columns:
            return c
    return None


def run_fold_gate(ctx: RunContext, train_csv: str, fold_col: str = "fold",
                  target_col: str | None = None, id_col: str | None = None,
                  group_col: str | None = None) -> dict:
    ev: dict[str, Any] = {"gate": "folds", "train_csv": train_csv, "fold_col": fold_col,
                          "problems": [], "warnings": []}

    def finish(passed: bool, reason: str) -> dict:
        ev["passed"] = passed
        ev["reason"] = reason
        ctx.store.record_gate(ctx.run_id, "folds", passed, ev)
        _dump(ctx, ev)
        return {"passed": passed, "reason": reason, "evidence": ev}

    path = ctx.resolve(train_csv)
    if not path.exists():
        return finish(False, f"找不到训练表 {train_csv}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # noqa: BLE001
        return finish(False, f"读取 {train_csv} 失败（大文件可先转 parquet 再传 target 列）：{e}")

    n = len(df)
    ev["n_rows"] = n
    ev["columns"] = list(df.columns)[:60]
    if fold_col not in df.columns:
        return finish(False, f"训练表缺少 {fold_col!r} 列。正确做法：划分一次并写回数据文件"
                             f"（如 train_folds.csv），让 OOF/集成/提交全部引用同一列，"
                             f"禁止每次运行时重新划分")
    if df[fold_col].isna().any():
        return finish(False, f"{fold_col} 列存在空值（{int(df[fold_col].isna().sum())} 行）")

    folds = df[fold_col].values
    uniq = sorted(pd.unique(folds).tolist())
    ev["folds"] = [str(u) for u in uniq]
    k = len(uniq)
    ev["k"] = k
    if not (3 <= k <= 10):
        return finish(False, f"折数 k={k} 不在 [3,10]；太少方差大、太多训练集太小")

    counts = df[fold_col].value_counts().reindex(uniq).fillna(0).astype(int)
    ev["fold_sizes"] = {str(i): int(c) for i, c in counts.items()}
    min_need = max(20, int(0.005 * n))
    if counts.min() < min_need:
        ev["problems"].append(f"最小折只有 {int(counts.min())} 行，低于下限 {min_need}")

    tgt = _infer(df, target_col, ["target", "label", "y", "Target", "class"])
    if tgt:
        ev["target_col"] = tgt
        s = df[tgt]
        if s.dtype.kind in "ifu" and s.nunique() > 20:
            g_mean, g_std = float(s.mean()), float(s.std(ddof=0))
            dev = {str(f): float(s[folds == f].mean()) for f in uniq}
            worst = max(abs(v - g_mean) for v in dev.values())
            ev["target"] = {"kind": "regression", "global_mean": round(g_mean, 6),
                            "global_std": round(g_std, 6),
                            "fold_means": {kk: round(vv, 6) for kk, vv in dev.items()},
                            "worst_abs_dev": round(worst, 6),
                            "allowed": round(0.10 * (g_std or 1.0), 6)}
            if g_std > 0 and worst > 0.10 * g_std:
                ev["problems"].append(
                    f"目标分布不均：最差折均值偏离全局 {worst:.4f} > 0.10×std={0.10*g_std:.4f}")
        else:
            vc = s.astype(str).value_counts(normalize=True)
            top = vc.index[:5]
            rates = {}
            for cls in top:
                global_rate = float((s.astype(str) == cls).mean())
                rates[cls] = {str(f): float((s[folds == f].astype(str) == cls).mean()) for f in uniq}
                worst = max(abs(v - global_rate) for v in rates[cls].values())
                if global_rate >= 0.01 and worst > 0.03:
                    ev["problems"].append(
                        f"类别 {cls!r} 折间比例偏差 {worst:.3f} > 0.03（全局 {global_rate:.3f}）")
            ev["target"] = {"kind": "classification", "top_classes": [str(c) for c in top],
                            "fold_rates": {str(k2): {kk: round(vv, 4) for kk, vv in v.items()}
                                           for k2, v in rates.items()}}
    else:
        ev["warnings"].append("未识别目标列（可用 target_col 指定），跳过分布检验")

    gcol = _infer(df, group_col, ["group", "user_id", "entity", "patient_id", "series_id"])
    if gcol and gcol != fold_col:
        ev["group_col"] = gcol
        leak = df.groupby(gcol)[fold_col].nunique()
        bad = int((leak > 1).sum())
        ev["groups"] = int(len(leak))
        ev["groups_crossing_folds"] = bad
        if bad:
            ev["problems"].append(f"{bad} 个 group 跨折（共 {len(leak)} 个）——必须用 GroupKFold")
    else:
        ic = _infer(df, id_col, ["id", "ID", "Id", "row_id"])
        if ic:
            dup = int(df[ic].duplicated().sum())
            ev["id_col"] = ic
            ev["duplicate_ids"] = dup
            if dup:
                crossing = df.groupby(ic)[fold_col].nunique()
                bad_ids = int((crossing > 1).sum())
                ev["ids_crossing_folds"] = bad_ids
                if bad_ids:
                    ev["problems"].append(
                        f"{bad_ids} 个 id 同时出现在多个折（共 {dup} 行重复）——同一实体的副本跨折等于泄漏，"
                        f"必须用 GroupKFold 按 id 分组")
                else:
                    ev["warnings"].append(
                        f"{dup} 行重复 id，但都落在同一折内（无跨折泄漏）；建议核实是否为合法的重复记录")

    passed = not ev["problems"]
    reason = (f"fold 划分通过：k={k}，行数 {n}，无泄漏/分布问题" if passed
              else f"fold 划分不合格：{len(ev['problems'])} 个问题")
    return finish(passed, reason)


def _dump(ctx: RunContext, ev: dict) -> None:
    try:
        p = ctx.oof_dir / "gate_folds.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
