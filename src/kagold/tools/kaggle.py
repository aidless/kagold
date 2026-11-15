"""Kaggle 平台工具：列赛、看榜、下载数据、提交、查配额。

所有返回值都做了瘦身（只保留决策需要的字段），避免把巨大 JSON 灌进上下文。
提交走 `submit` 工具，它在代码层强制查门禁（见 tools/gates.py）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .base import ToolError, obj, tool
from .. import config
from ..runtime import RunContext


def _s(v: Any, n: int = 200) -> str:
    return "" if v is None else str(v)[:n]


def medal_slots(teams: int) -> dict:
    """按 Kaggle 公开规则估算奖牌名次门槛（启发式，输出里会标注）。

    1000 队以上：金 = 前 10 队 + 0.2% 队数；银 = 前 5%；铜 = 前 10%。
    1000 队以下：金 10% / 银 20% / 铜 40%（最少 1/2/3）。
    """
    t = max(0, int(teams or 0))
    if t == 0:
        return {"gold": 0, "silver": 0, "bronze": 0, "note": "队数未知"}
    if t < 1000:
        g, s, b = max(1, int(t * 0.10)), max(2, int(t * 0.20)), max(3, int(t * 0.40))
    else:
        g = 10 + int(round(0.002 * t))
        s = max(g + 1, int(round(0.05 * t)))
        b = max(s + 1, int(round(0.10 * t)))
    return {"gold": g, "silver": s, "bronze": b,
            "note": "启发式估算，以 kaggle.com/progression 官方口径为准"}


@tool("competitions_list",
      "列出 Kaggle 竞赛（可按官方排序 / 关键词过滤）。返回 ref、队数、截止、奖励、金牌名次门槛，"
      "用于选题：优先『队数少 + 有奖牌 + 剩余时间足够』的竞赛。",
      obj({"sort_by": {"type": "string",
                       "enum": ["recentlyCreated", "earliestDeadline", "latestDeadline",
                                "numberOfTeams", "prize", "best", "grouped", "relevance"]},
           "search": {"type": "string", "description": "关键词过滤"},
           "page": {"type": "integer", "description": "默认 1"},
           "limit": {"type": "integer", "description": "默认 20，最多 60"}}, []))
def competitions_list(ctx: RunContext, a: dict) -> dict:
    api = ctx.kaggle()
    page = max(1, int(a.get("page") or 1))
    limit = min(int(a.get("limit") or 20), 60)
    show = a.get("sort_by") or "recentlyCreated"
    resp = api.competitions_list(sort_by=show, search=a.get("search"), page=page)
    comps = getattr(resp, "competitions", None) or []
    rows = []
    for c in comps[:limit]:
        teams = _int(getattr(c, "team_count", 0))
        slots = medal_slots(teams)
        rows.append({
            "ref": config.slug_of(_s(getattr(c, "ref", ""))),
            "title": _s(getattr(c, "title", ""), 90),
            "teams": teams,
            "gold_rank_cutoff": slots["gold"],
            "deadline": _s(getattr(c, "deadline", ""), 19),
            "reward": _s(getattr(c, "reward", ""), 40),
            "metric": _s(getattr(c, "evaluation_metric", ""), 60),
            "category": _s(getattr(c, "category", ""), 20),
            "awards_points": bool(getattr(c, "awards_points", False)),
            "entered": bool(getattr(c, "user_has_entered", False)),
            "my_rank": _s(getattr(c, "user_rank", ""), 12),
            "max_daily_submissions": _int(getattr(c, "max_daily_submissions", 0)),
            "max_team_size": _int(getattr(c, "max_team_size", 0)),
        })
    return {"ok": True, "page": page, "count": len(rows), "competitions": rows}


@tool("competition_info", "查看单个竞赛的详情：数据文件清单、提交限制、当前榜单门槛附近的队伍。",
      obj({"ref": {"type": "string", "description": "竞赛 slug"},
           "top_n": {"type": "integer", "description": "榜单取前 N 名，默认 30"}}, ["ref"]))
def competition_info(ctx: RunContext, a: dict) -> dict:
    api = ctx.kaggle()
    slug = config.slug_of(a["ref"])
    info: dict[str, Any] = {"ref": slug}
    try:
        files, token, all_files = [], None, []
        for _ in range(12):
            resp = api.competition_list_files(slug, page_token=token, page_size=100)
            fs = getattr(resp, "files", None) or []
            all_files.extend(fs)
            token = getattr(resp, "next_page_token", None)
            if not token or not fs:
                break
        total = 0
        for f in all_files:
            sz = _int(getattr(f, "total_bytes", 0))
            total += sz
            files.append({"name": _s(getattr(f, "name", ""), 120), "bytes": sz})
        info["files"] = files[:80]
        info["file_count"] = len(files)
        info["total_bytes"] = total
        info["total_gb"] = round(total / (1024 ** 3), 2)
    except Exception as e:  # 未接受规则时这里常报 403
        info["files_error"] = f"{type(e).__name__}: {_s(e, 300)}"
    try:
        lim = api.competition_get_submission_limits(slug)
        info["limits"] = {
            "today": _int(getattr(lim, "num_today", 0)),
            "total": _int(getattr(lim, "num_total", 0)),
            "allowed_now": _int(getattr(lim, "num_allowed_now", 0)),
            "limited_by_total": bool(getattr(lim, "limited_by_total", False)),
        }
    except Exception as e:
        info["limits_error"] = f"{type(e).__name__}: {_s(e, 200)}"
    try:
        top = api.competition_leaderboard_view(slug, page_size=max(5, int(a.get("top_n") or 30)))
        rows = [{"team": _s(getattr(e, "team_name", ""), 40),
                 "score": _s(getattr(e, "score", ""), 20),
                 "date": _s(getattr(e, "submission_date", ""), 19)} for e in (top or [])]
        info["leaderboard_top"] = rows
        if rows:
            info["public_best"] = rows[0]["score"]
    except Exception as e:
        info["leaderboard_error"] = f"{type(e).__name__}: {_s(e, 200)}"
    return {"ok": True, "data": info}


@tool("download_data",
      "下载竞赛数据到 input/（先做磁盘守卫；未接受规则会失败并提示）。"
      "大数据集请先看 competition_info 的 total_gb。",
      obj({"ref": {"type": "string"},
           "need_gb": {"type": "number", "description": "预估所需空间，默认 2"}}, ["ref"]))
def download_data(ctx: RunContext, a: dict) -> dict:
    slug = config.slug_of(a["ref"])
    need = float(a.get("need_gb") or 2.0)
    config.disk_guard(ctx.settings.min_free_gb, need_gb=need)
    api = ctx.kaggle()
    dest = ctx.input_dir
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(slug, path=str(dest), force=True, quiet=True)
    except Exception as e:
        msg = _s(e, 400)
        hint = ""
        if "403" in msg or "forbidden" in msg.lower() or "rules" in msg.lower():
            hint = ("需要先在网页上接受竞赛规则（https://www.kaggle.com/competitions/"
                    f"{slug}/rules）；本 agent 默认不代你接受法律条款。")
        raise ToolError(f"下载失败：{msg} {hint}")
    files = sorted(p.name for p in dest.iterdir())[:50]
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    return {"ok": True, "path": str(dest), "files": files,
            "total_gb": round(size / (1024 ** 3), 2)}


@tool("leaderboard",
      "拉取公开榜。给 gold_rank 时返回该名次的分数——这就是『进金牌区需要的分』。",
      obj({"ref": {"type": "string"},
           "top_n": {"type": "integer", "description": "默认 50，最多 200"},
           "gold_rank": {"type": "integer", "description": "金牌名次门槛（来自竞赛元信息）"}},
          ["ref"]))
def leaderboard(ctx: RunContext, a: dict) -> dict:
    api = ctx.kaggle()
    slug = config.slug_of(a["ref"])
    n = min(int(a.get("top_n") or 50), 200)
    entries = api.competition_leaderboard_view(slug, page_size=n) or []
    rows = []
    for i, e in enumerate(entries, 1):
        rows.append({"rank": i, "team": _s(getattr(e, "team_name", ""), 40),
                     "score": _s(getattr(e, "score", ""), 24),
                     "date": _s(getattr(e, "submission_date", ""), 19)})
    gr = int(a.get("gold_rank") or 0)
    return {"ok": True, "rows": rows,
            "best": rows[0]["score"] if rows else None,
            "cutoff_10": rows[9]["score"] if len(rows) >= 10 else None,
            "gold_rank": gr or None,
            "gold_cutoff_score": (rows[gr - 1]["score"] if gr and len(rows) >= gr else None),
            "hint": ("gold_cutoff_score 是当前公开榜第 {} 名的分数；你的 OOF 要能压过它才有戏。"
                     .format(gr) if gr else "")}


@tool("my_submissions", "查看我在该竞赛的历史提交（含公开/私榜分数与状态）。",
      obj({"ref": {"type": "string"}, "limit": {"type": "integer"}}, ["ref"]))
def my_submissions(ctx: RunContext, a: dict) -> dict:
    api = ctx.kaggle()
    slug = config.slug_of(a["ref"])
    subs = api.competition_submissions(slug, page_size=min(int(a.get("limit") or 20), 50)) or []
    rows = []
    for s in subs:
        rows.append({
            "ref": _s(getattr(s, "ref", ""), 20),
            "date": _s(getattr(s, "date", ""), 19),
            "score": _num(getattr(s, "public_score", None)),
            "private": _num(getattr(s, "private_score", None)),
            "status": _s(getattr(s, "status", ""), 40),
            "description": _s(getattr(s, "description", ""), 80),
            "file": _s(getattr(s, "file_name", ""), 60),
        })
    return {"ok": True, "count": len(rows), "submissions": rows}


def _int(v: Any) -> int:
    try:
        return int(float(str(v)))
    except Exception:
        return 0


def _num(v: Any) -> Any:
    try:
        f = float(str(v))
        return None if math.isnan(f) else f
    except Exception:
        return None
