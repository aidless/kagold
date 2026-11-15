"""阻塞等待工具：让 agent 有办法「训练跑完了再继续」而不必 sleep 干等。

设计取舍：用轮询而不是 inotify/fanotify —— 轮询 10s 一次的成本可以忽略，
但不需要额外依赖，也不会因为编辑器/日志轮转造成事件丢失。
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..runtime import RunContext
from .base import ToolError, obj, tool


@tool("wait_log",
      "等日志文件里出现某个模式再返回（轮询，命中即返回；超时返回 matched=false）。"
      "训练/下载跑在后台时用它代替 sleep，命中就继续下一步，不浪费墙钟与步数。",
      obj({"path": {"type": "string", "description": "日志文件（相对 run 根）"},
           "pattern": {"type": "string", "description": "正则；默认匹配 'done|完成|Traceback'"},
           "timeout": {"type": "integer", "description": "最长等待秒数，默认 900，上限 1800"},
           "interval": {"type": "integer", "description": "轮询间隔秒，默认 10"},
           "tail_lines": {"type": "integer", "description": "命中时返回最后多少行，默认 20"}},
          ["path"]))
def wait_log(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    if not p.exists():
        raise ToolError(f"{a['path']} 不存在；先确认后台任务是否真的启动了")
    pat = re.compile(a.get("pattern") or r"(?i)\b(done|finished|完成|Traceback)\b")
    timeout = max(5, min(int(a.get("timeout") or 900), 1800))
    interval = max(2, min(int(a.get("interval") or 10), 60))
    tail_n = max(3, min(int(a.get("tail_lines") or 20), 200))
    deadline = time.time() + timeout
    hit = None
    while True:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:  # 文件被轮转/half-written
            text = ""
            _ = e
        for line in text.splitlines():
            if pat.search(line):
                hit = line.strip()[:300]
                break
        if hit or time.time() >= deadline:
            break
        time.sleep(interval)
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    tail = "\n".join(text.splitlines()[-tail_n:])
    return {"ok": True, "matched": bool(hit), "hit": hit, "waited_s": round(timeout - (deadline - time.time()), 1),
            "tail": tail,
            "next": "" if hit else "超时了：看一眼 tail 里到哪一步，再决定是继续等还是换策略。"}


@tool("wait_file",
      "等某个产物文件出现且大小稳定（写完）再返回。适合等 sub/submission.csv、oof/oof.csv。",
      obj({"path": {"type": "string"},
           "timeout": {"type": "integer", "description": "默认 900，上限 1800"},
           "min_bytes": {"type": "integer", "description": "至少多大才算写完，默认 16"}},
          ["path"]))
def wait_file(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    timeout = max(5, min(int(a.get("timeout") or 900), 1800))
    min_bytes = int(a.get("min_bytes") or 16)
    deadline = time.time() + timeout
    last = -1
    stable = 0
    while time.time() < deadline:
        if p.exists():
            n = p.stat().st_size
            if n >= min_bytes and n == last:
                stable += 1
                if stable >= 1:
                    return {"ok": True, "exists": True, "bytes": n,
                            "waited_s": round(timeout - (deadline - time.time()), 1)}
            else:
                stable = 0
                last = n
        time.sleep(5)
    return {"ok": True, "exists": p.exists(), "bytes": (p.stat().st_size if p.exists() else 0),
            "timeout": True, "next": "文件没出现/没写完：先确认后台任务的 stderr。"}
