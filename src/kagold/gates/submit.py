"""门禁 4：提交（submit gate）—— 最后一道，也是最硬的一道。

它同时检查「过程」和「产物」：
过程  metric / folds / oof 三道门必须已通过（读 SQLite，不读对话上下文）；
产物  提交文件必须与 sample_submission 的列、行集合、顺序完全一致，无 NaN/inf；
配额  Kaggle 侧 allowed_now > 0（拿不到就降级为警告，不假装通过）。

通过后写出 sub/gate_submit.json，内含提交文件的 sha256。
`submit` 工具在真正上传前会再核对一遍门禁状态 + 文件哈希，
所以「模型说自己过了门」无法伪造提交。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import RunContext

REQUIRED = ("metric", "folds", "oof")


def find_sample(ctx: RunContext) -> Path | None:
    for p in ctx.input_dir.rglob("sample_submission*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".parquet", ".txt"):
            return p
    return None


def run_submit_gate(ctx: RunContext, sub_path: str = "sub/submission.csv",
                    sample_path: str | None = None,
                    required: tuple[str, ...] = REQUIRED) -> dict:
    ev: dict[str, Any] = {"gate": "submit", "sub_path": sub_path,
                          "required_gates": list(required), "problems": [], "warnings": []}

    def finish(passed: bool, reason: str) -> dict:
        ev["passed"] = passed
        ev["reason"] = reason
        ctx.store.record_gate(ctx.run_id, "submit", passed, ev)
        _dump(ctx, ev)
        return {"passed": passed, "reason": reason, "evidence": ev}

    # ---- 过程门禁 ----
    states = ctx.store.gate_state(ctx.run_id)
    missing = [g for g in required if not states.get(g, {}).get("passed")]
    ev["gate_states"] = {g: bool(states.get(g, {}).get("passed")) for g in states}
    if missing:
        detail = []
        for g in missing:
            st = states.get(g)
            detail.append(f"{g}: {'未执行' if not st else '未通过 — ' + str(st.get('evidence', {}).get('reason', ''))[:120]}")
        return finish(False, f"前置门禁未通过：{missing}。{'; '.join(detail)}")

    # ---- 产物 ----
    p = ctx.resolve(sub_path)
    if not p.exists():
        return finish(False, f"提交文件不存在：{sub_path}")
    sample = ctx.resolve(sample_path) if sample_path else find_sample(ctx)
    if sample is None:
        return finish(False, "找不到 sample_submission（请下载数据或显式传 sample_path）")
    ev["sample"] = ctx.rel(sample)
    # 注意：样例文件可能比 20 万行大得多（本项目实测 299844 行）。
    # 绝不能用 nrows 截断后再比行数——那会让门禁对任何大样本竞赛永久失败。
    # 策略：列名用 nrows=0 读，行数与 id 只读第一列（流式、内存友好），
    #       数值量纲对照用前 20 万行（只是统计，不影响行数判定）。
    if sample.suffix.lower() == ".parquet":
        samp_all = pd.read_parquet(sample)
        samp_cols = list(samp_all.columns)
        samp_ids = samp_all[samp_cols[0]].astype(str) if samp_cols else pd.Series(dtype=str)
        samp = samp_all.head(200000)
    else:
        samp_cols = [str(c) for c in pd.read_csv(sample, nrows=0).columns]
        if not samp_cols:
            return finish(False, "样例文件没有列名，无法校验")
        samp_ids = pd.read_csv(sample, usecols=[samp_cols[0]], low_memory=False)[samp_cols[0]].astype(str)
        samp = pd.read_csv(sample, nrows=200000)
    sub = pd.read_csv(p, low_memory=False) if p.suffix.lower() != ".parquet" else pd.read_parquet(p)

    ev["rows"] = int(len(sub))
    ev["sample_rows"] = int(len(samp_ids))
    ev["cols"] = list(sub.columns)
    if list(sub.columns) != samp_cols:
        ev["problems"].append(f"列不一致：提交={list(sub.columns)} 样例={samp_cols}")
    if len(sub) != len(samp_ids):
        ev["problems"].append(f"行数不一致：提交 {len(sub)} vs 样例 {len(samp_ids)}")
    if list(sub.columns) == samp_cols and len(sub) == len(samp_ids):
        key = samp_cols[0]
        sub_ids = sub[key].astype(str)
        same_order = bool((sub_ids.values == samp_ids.values).all())
        ev["id_order_matches"] = same_order
        if not same_order:
            same_set = set(sub_ids) == set(samp_ids)
            ev["id_set_matches"] = same_set
            if not same_set:
                ev["problems"].append("id 集合与样例不一致")
            else:
                ev["warnings"].append("id 顺序与样例不同，部分竞赛会因此判负分")
    if len(samp_ids) > 200000:
        ev["warnings"].append(f"样例共 {len(samp_ids)} 行（>20 万）；数值量纲对照只取前 20 万行")

    # ---- 数值健康 ----
    num_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
    ev["numeric_cols"] = num_cols
    stats = {}
    for c in num_cols:
        s = pd.to_numeric(sub[c], errors="coerce")
        bad = int(s.isna().sum() - sub[c].isna().sum())
        inf = int(np.isinf(s.fillna(0).values).sum())
        stats[c] = {"min": float(s.min()) if s.notna().any() else None,
                    "max": float(s.max()) if s.notna().any() else None,
                    "mean": float(s.mean()) if s.notna().any() else None,
                    "nan": int(sub[c].isna().sum()), "non_numeric": bad, "inf": inf}
        if stats[c]["nan"] or stats[c]["inf"]:
            ev["problems"].append(f"{c}: NaN={stats[c]['nan']} inf={stats[c]['inf']}")
    ev["stats"] = stats
    # 与样例的量纲对照（样例若是 0/1 概率而提交是 0/100 的 logit 分数，这里能看出来）
    for c in num_cols:
        if c not in samp.columns:
            continue
        s0 = pd.to_numeric(samp[c], errors="coerce")
        s1 = pd.to_numeric(sub[c], errors="coerce")
        if not s0.notna().any() or not s1.notna().any():
            continue
        try:
            rng0 = float(s0.max() - s0.min())
            rng1 = float(s1.max() - s1.min())
            if rng0 <= 1.01 and rng1 > 2.0:
                ev["warnings"].append(
                    f"{c}: 样例取值范围 [{s0.min():.3f},{s0.max():.3f}]，提交为 "
                    f"[{s1.min():.3f},{s1.max():.3f}] —— 检查是否需要 sigmoid/归一化")
        except Exception:
            pass

    # ---- 配额 ----
    try:
        lim = ctx.kaggle().competition_get_submission_limits(ctx.comp)
        allowed = int(getattr(lim, "num_allowed_now", 0))
        used_today = int(getattr(lim, "num_today", 0))
        ev["quota"] = {"allowed_now": allowed, "used_today": used_today}
        if allowed <= 0:
            ev["problems"].append(f"今日提交配额已用尽（used_today={used_today}）")
    except Exception as e:  # noqa: BLE001
        ev["warnings"].append(f"配额查询失败（不阻断，但请确认）：{type(e).__name__}: {str(e)[:120]}")

    ev["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    ev["bytes"] = p.stat().st_size
    passed = not ev["problems"]
    reason = ("提交门禁通过：前置门禁齐备，格式与样例一致，无 NaN/inf"
              if passed else f"提交门禁失败：{len(ev['problems'])} 个问题")
    return finish(passed, reason)


def _dump(ctx: RunContext, ev: dict) -> None:
    try:
        ctx.sub_dir.mkdir(parents=True, exist_ok=True)
        (ctx.sub_dir / "gate_submit.json").write_text(
            json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
