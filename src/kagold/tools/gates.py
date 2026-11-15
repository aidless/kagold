"""门禁与提交工具（给 agent 用的入口）。

关键一点：`submit` 不是"转发给 Kaggle"，而是先自己查一遍台账：
metric/folds/oof/submit 四道门必须都是 passed，且提交文件的 sha256 必须与提交门禁记录的一致。
任何一条不满足 → 直接拒绝并告诉模型缺什么。模型无法用文字绕过。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .base import ToolError, obj, tool
from ..gates import run_fold_gate, run_metric_gate, run_oof_gate, run_submit_gate
from ..gates.submit import find_sample
from ..runtime import RunContext

PHASES = ["SELECT", "REPRO", "IMPROVE", "ENSEMBLE", "SUBMIT", "DONE"]


def _auto_advance(ctx: RunContext, target: str) -> str:
    """阶段自动前进一格（只前进一名，且只从紧邻的前一阶段）。

    模型经常忘记调 set_phase，导致状态卡一直显示 SELECT；与其靠提示词纪律，
    不如让"做完事"本身推动状态机。手工调 set_phase 仍然有效。
    """
    cur = (ctx.store.run(ctx.run_id) or {}).get("phase", "SELECT")
    if cur in PHASES and target in PHASES and PHASES.index(target) == PHASES.index(cur) + 1:
        ctx.store.update_run(ctx.run_id, phase=target)
        return target
    return cur


def _objdump(o: Any) -> dict:
    for attr in ("model_dump", "dict"):
        if hasattr(o, attr):
            try:
                d = getattr(o, attr)()
                return {str(k).lstrip("_"): v for k, v in d.items()}
            except Exception:
                pass
    if hasattr(o, "__dict__"):
        return {str(k).lstrip("_"): v for k, v in vars(o).items()}
    return {"raw": str(o)}


# ---------------------------------------------------------------- gates
@tool("gate_metric",
      "【门禁1/4】验证你写的 src/metric.py: score(y_true, y_pred) 与官方口径一致。"
      "会用独立参考实现在 5 组合成数据上逐例比对，并检查方向（越大越好/越小越好）。"
      "不通过不能提交，也不能进入改进阶段。",
      obj({"metric": {"type": "string", "description": "竞赛 metric，如 rmse / auc / logloss / qwk"},
           "module": {"type": "string", "description": "默认 src/metric.py"}}, ["metric"]))
def gate_metric(ctx: RunContext, a: dict) -> dict:
    r = run_metric_gate(ctx, a["metric"], a.get("module") or "src/metric.py")
    return {"ok": True, "passed": r["passed"], "reason": r["reason"], "evidence": r["evidence"],
            "next": "" if r["passed"] else "按 evidence.cases 逐个修正 score 实现后重跑本门禁"}


@tool("gate_folds",
      "【门禁2/4】检查交叉验证划分：fold 列是否存在、折数/每折样本量、目标分布均衡、group 是否跨折泄漏。"
      "先划分并写回数据文件，再跑本门禁。",
      obj({"train_csv": {"type": "string", "description": "如 input/train_folds.csv"},
           "fold_col": {"type": "string", "description": "默认 fold"},
           "target_col": {"type": "string", "description": "可选，自动识别"},
           "id_col": {"type": "string", "description": "可选"},
           "group_col": {"type": "string", "description": "可选，有实体重复时必须给"}}, ["train_csv"]))
def gate_folds(ctx: RunContext, a: dict) -> dict:
    r = run_fold_gate(ctx, a["train_csv"], a.get("fold_col") or "fold",
                      a.get("target_col"), a.get("id_col"), a.get("group_col"))
    return {"ok": True, "passed": r["passed"], "reason": r["reason"], "evidence": r["evidence"]}


@tool("gate_oof",
      "【门禁3/4】用参考实现复算 OOF：覆盖每一行、无 NaN、每折分数+均值+标准差、多模型相关系数。"
      "OOF 文件需含 id 列与 pred（或 pred_<model>）列；fold 从训练表按 id 关联。",
      obj({"metric": {"type": "string"},
           "oof_csv": {"type": "string", "description": "默认 oof/oof.csv"},
           "train_csv": {"type": "string", "description": "可选，默认自动找 input/train*.csv"},
           "id_col": {"type": "string"}, "target_col": {"type": "string"},
           "fold_col": {"type": "string"}}, ["metric"]))
def gate_oof(ctx: RunContext, a: dict) -> dict:
    r = run_oof_gate(ctx, a["metric"], a.get("oof_csv") or "oof/oof.csv",
                     a.get("train_csv"), a.get("id_col"), a.get("target_col"),
                     a.get("fold_col") or "fold")
    phase = _auto_advance(ctx, "IMPROVE") if r["passed"] else None
    return {"ok": True, "passed": r["passed"], "reason": r["reason"], "evidence": r["evidence"],
            "phase": phase}


@tool("gate_submit",
      "【门禁4/4】检查提交产物：前置三道门是否通过、列/行数与 sample_submission 是否一致、"
      "有无 NaN/inf、量纲是否可疑、今日配额是否还有。通过后才允许 submit。",
      obj({"sub_path": {"type": "string", "description": "默认 sub/submission.csv"},
           "sample_path": {"type": "string", "description": "可选，默认自动找 input/ 下 sample_submission"}}, []))
def gate_submit(ctx: RunContext, a: dict) -> dict:
    r = run_submit_gate(ctx, a.get("sub_path") or "sub/submission.csv", a.get("sample_path"))
    return {"ok": True, "passed": r["passed"], "reason": r["reason"], "evidence": r["evidence"]}


# ---------------------------------------------------------------- submit
@tool("submit",
      "把 sub/submission.csv 提交到 Kaggle（唯一提交入口）。"
      "会在上传前复核四道门禁与文件哈希；缺门禁会被拒绝并告诉你缺哪道。"
      "每次提交都会记进台账。代码类竞赛（kernels only）需要走 notebook 版本提交，本工具会给出提示。",
      obj({"message": {"type": "string", "description": "提交说明，例如 'lgbm v3 + cat v2 blend 0.72/0.28'"},
           "sub_path": {"type": "string", "description": "默认 sub/submission.csv"},
           "dry_run": {"type": "boolean", "description": "true 时只做检查不上传"}}, ["message"]))
def submit(ctx: RunContext, a: dict) -> dict:
    sub_path = a.get("sub_path") or "sub/submission.csv"
    # 1) 四道门
    states = ctx.store.gate_state(ctx.run_id)
    need = ["metric", "folds", "oof", "submit"]
    missing = [g for g in need if not states.get(g, {}).get("passed")]
    if missing:
        raise ToolError(
            "拒绝提交：门禁未通过 " + json.dumps(missing, ensure_ascii=False)
            + "。当前状态：" + json.dumps(
                {k: bool(v.get("passed")) for k, v in states.items()}, ensure_ascii=False)
            + "。请先跑 gate_metric / gate_folds / gate_oof / gate_submit。"
        )
    # 2) 文件哈希必须与提交门禁记录一致（防止过了门禁再换文件）
    p = ctx.resolve(sub_path)
    if not p.exists():
        raise ToolError(f"提交文件不存在：{sub_path}")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    recorded = states["submit"].get("evidence", {}).get("sha256")
    if recorded and recorded != digest:
        raise ToolError(f"提交文件已被修改（sha256 不符），请重跑 gate_submit。"
                        f"记录={recorded[:12]} 当前={digest[:12]}")
    # 3) 预算
    run = ctx.store.run(ctx.run_id) or {}
    if int(run.get("submissions") or 0) >= int(ctx.settings.max_submissions):
        raise ToolError(f"本次 run 提交次数已达上限 {ctx.settings.max_submissions}，"
                        f"如确需继续请显式提高 --max-submissions")
    if a.get("dry_run"):
        return {"ok": True, "dry_run": True, "would_submit": sub_path,
                "sha256": digest[:16], "gates": {g: True for g in need}}

    api = ctx.kaggle()
    try:
        resp = api.competition_submit(str(p), str(a["message"])[:200], ctx.comp, quiet=True)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:400]}"
        hint = ""
        if "400" in msg or "kernel" in msg.lower():
            hint = ("；如果是代码竞赛（Kernels-only），文件上传会被拒："
                    "需要用 kaggle kernels push 推 notebook 再用 `kaggle competitions submit -k <kernel> -v <version>`")
        raise ToolError(f"Kaggle 提交失败：{msg}{hint}")
    info = _objdump(resp)
    ctx.store.add_submission(ctx.run_id, ctx.comp, str(p), str(a["message"]),
                             str(info.get("status") or "submitted"),
                             _num(info.get("public_score")), info)
    ctx.store.bump(ctx.run_id, submissions=1)
    ctx.store.log_event(ctx.run_id, "submit", {"message": a["message"], "sha256": digest, "resp": info})
    _auto_advance(ctx, "SUBMIT")
    return {"ok": True, "submitted": True, "response": info,
            "message": "已提交。用 my_submissions 看当前分数；public LB 与 CV 差距大时先审 split/metric/提交格式，别急着调参。"}


def _num(v: Any) -> float | None:
    try:
        return float(str(v))
    except Exception:
        return None


# ---------------------------------------------------------------- state
@tool("record_competition",
      "登记竞赛元信息（进入 REPRO 阶段的前提）：metric、队数、金牌门槛、截止时间、我的目标名次。",
      obj({"ref": {"type": "string"},
           "metric": {"type": "string"},
           "teams": {"type": "integer"},
           "gold_rank_cutoff": {"type": "integer"},
           "deadline": {"type": "string"},
           "public_best": {"type": "string", "description": "当前公开榜第一名分数（字符串，便于带小数）"},
           "target": {"type": "string", "description": "你的目标，如 '进前 10'"}}, ["ref", "metric"]))
def record_competition(ctx: RunContext, a: dict) -> dict:
    meta = {"ref": a["ref"], "metric": a["metric"], "teams": a.get("teams"),
            "gold_rank_cutoff": a.get("gold_rank_cutoff"), "deadline": a.get("deadline"),
            "public_best": a.get("public_best"), "target": a.get("target"),
            "recorded_at": time.time()}
    ctx.store.kv_set(ctx.run_id, "competition", meta)
    phase = _auto_advance(ctx, "REPRO")
    return {"ok": True, "competition": meta, "phase": phase}


@tool("set_phase", "推进阶段（只能前进一格，且需要满足该阶段的前置条件）。阶段：SELECT→REPRO→IMPROVE→ENSEMBLE→SUBMIT→DONE。",
      obj({"phase": {"type": "string", "enum": PHASES}}, ["phase"]))
def set_phase(ctx: RunContext, a: dict) -> dict:
    target = a["phase"]
    cur = (ctx.store.run(ctx.run_id) or {}).get("phase", "SELECT")
    if target not in PHASES:
        raise ToolError(f"未知阶段 {target}")
    if PHASES.index(target) > PHASES.index(cur) + 1:
        raise ToolError(f"不能跳阶段：{cur} → {target}（一次只能前进一格）")
    need = []
    if target in ("REPRO", "IMPROVE", "ENSEMBLE", "SUBMIT", "DONE"):
        if not ctx.store.kv_get(ctx.run_id, "competition"):
            need.append("record_competition（登记 metric/队数/门槛）")
    if target in ("IMPROVE", "ENSEMBLE", "SUBMIT", "DONE"):
        for g in ("metric", "folds"):
            if not ctx.store.gate_pass(ctx.run_id, g):
                need.append(f"gate_{g} 通过")
    if target in ("ENSEMBLE", "SUBMIT", "DONE") and not ctx.store.gate_pass(ctx.run_id, "oof"):
        need.append("gate_oof 通过")
    if target == "DONE" and not ctx.store.gate_pass(ctx.run_id, "submit"):
        need.append("gate_submit 通过 + 至少一次 submit")
    if need:
        raise ToolError(f"不能进入 {target}：还缺 {need}")
    ctx.store.update_run(ctx.run_id, phase=target)
    return {"ok": True, "phase": target, "previous": cur}


@tool("note", "记一条工作笔记到 logs/notes.md（保存关键结论、失败原因、下一步计划）。",
      obj({"text": {"type": "string"}}, ["text"]))
def note(ctx: RunContext, a: dict) -> dict:
    line = f"- [{time.strftime('%m-%d %H:%M')}] {a['text']}\n"
    p = ctx.log_dir / "notes.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True, "appended": line.strip()}


@tool("finish", "结束本次 run 并写明原因。只有真的完成目标或确认无法继续时才用。",
      obj({"reason": {"type": "string"},
           "status": {"type": "string", "enum": ["done", "paused", "failed"]}}, ["reason"]))
def finish(ctx: RunContext, a: dict) -> dict:
    status = a.get("status") or "done"
    ctx.store.update_run(ctx.run_id, status=status, note=a["reason"])
    ctx.store.log_event(ctx.run_id, "finish", {"reason": a["reason"], "status": status})
    return {"ok": True, "finished": True, "status": status, "reason": a["reason"]}


@tool("run_state", "查看本次 run 的完整状态：阶段、门禁、OOF 最优分、提交记录、预算剩余。",
      obj({}, []))
def run_state(ctx: RunContext, a: dict) -> dict:
    run = ctx.store.run(ctx.run_id) or {}
    gates = {k: bool(v.get("passed")) for k, v in ctx.store.gate_state(ctx.run_id).items()}
    return {"ok": True, "data": {
        "run_id": ctx.run_id, "comp": ctx.comp, "phase": run.get("phase"),
        "status": run.get("status"), "steps": run.get("steps"),
        "tokens_in": run.get("tokens_in"), "tokens_out": run.get("tokens_out"),
        "submissions": run.get("submissions"), "gates": gates,
        "oof_best": ctx.store.kv_get(ctx.run_id, "oof_best"),
        "baseline_score": ctx.store.kv_get(ctx.run_id, "baseline_score"),
        "competition": ctx.store.kv_get(ctx.run_id, "competition"),
    }}


GATE_TOOLS = [gate_metric, gate_folds, gate_oof, gate_submit, submit,
              record_competition, set_phase, note, finish, run_state]
