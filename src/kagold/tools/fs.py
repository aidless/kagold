"""文件工具：所有路径都被限制在本次 run 的工作目录内。"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from .base import ToolError, obj, tool
from ..runtime import RunContext

MAX_READ = 400_000


@tool("read_file", "读取 run 目录内的文本文件，返回带行号内容。大文件请用 offset/limit 分页。",
      obj({"path": {"type": "string", "description": "相对 run 根的路径，如 src/baseline.py"},
           "offset": {"type": "integer", "description": "起始行（0 基），默认 0"},
           "limit": {"type": "integer", "description": "最多行数，默认 400"}},
          ["path"]))
def read_file(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    if not p.exists():
        raise ToolError(f"文件不存在：{a['path']}（run 根目录：{ctx.root}）")
    if p.is_dir():
        raise ToolError(f"{a['path']} 是目录，请用 list_dir")
    if p.stat().st_size > MAX_READ:
        raise ToolError(f"文件 {p.stat().st_size/1e6:.1f}MB 过大，请用 offset/limit 分页读取")
    text = p.read_text(errors="replace")
    lines = text.splitlines()
    off = max(0, int(a.get("offset") or 0))
    lim = min(int(a.get("limit") or 400), 4000)
    chunk = lines[off:off + lim]
    body = "\n".join(f"{i + off + 1:5d}| {ln}" for i, ln in enumerate(chunk))
    tail = "" if off + lim >= len(lines) else f"\n... 还有 {len(lines) - off - lim} 行（用 offset={off+lim} 继续）"
    return {"ok": True, "path": ctx.rel(p), "total_lines": len(lines), "content": body + tail}


@tool("write_file", "写入（覆盖）run 目录内的文本文件，父目录自动创建。",
      obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]))
def write_file(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    content = a["content"]
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": ctx.rel(p), "bytes": len(content.encode()), "lines": content.count("\n") + 1}


@tool("append_file", "在 run 目录内文件末尾追加文本（用于日志/笔记）。",
      obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]))
def append_file(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(a["content"])
    return {"ok": True, "path": ctx.rel(p), "bytes": p.stat().st_size}


@tool("list_dir", "列出 run 目录内容（含大小），用于了解工作区和数据文件。",
      obj({"path": {"type": "string", "description": "默认 ."},
           "depth": {"type": "integer", "description": "默认 1，最多 3"}}, []))
def list_dir(ctx: RunContext, a: dict) -> dict:
    base = ctx.resolve(a.get("path") or ".")
    depth = min(int(a.get("depth") or 1), 3)
    out: list[str] = []
    base_real = base.resolve()
    for root, dirs, files in os.walk(base):
        rel = Path(root).resolve().relative_to(base_real)
        level = 0 if str(rel) == "." else len(rel.parts)
        if level >= depth:
            dirs[:] = []
            continue
        dirs.sort()
        for d in dirs[:40]:
            out.append(f"{'  ' * level}[dir] {rel / d if str(rel) != '.' else d}/")
        for f in sorted(files)[:60]:
            fp = Path(root) / f
            try:
                sz = fp.stat().st_size
            except OSError:
                sz = -1
            out.append(f"{'  ' * level}{rel / f if str(rel) != '.' else f}  ({_human(sz)})")
    return {"ok": True, "root": str(base_real), "entries": out[:400], "truncated": len(out) > 400}


@tool("grep", "在 run 目录内按正则搜索文本（用于查代码/数据头）。",
      obj({"pattern": {"type": "string"},
           "path": {"type": "string", "description": "默认 ."},
           "glob": {"type": "string", "description": "默认 *（只搜文本文件）"},
           "max_hits": {"type": "integer", "description": "默认 60"}}, ["pattern"]))
def grep(ctx: RunContext, a: dict) -> dict:
    import re
    try:
        rx = re.compile(a["pattern"])
    except re.error as e:
        raise ToolError(f"正则不合法：{e}")
    base = ctx.resolve(a.get("path") or ".")
    pat = a.get("glob") or "*"
    hits: list[str] = []
    max_hits = min(int(a.get("max_hits") or 60), 300)
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not fnmatch.fnmatch(f, pat):
                continue
            fp = Path(root) / f
            try:
                if fp.stat().st_size > 5_000_000:
                    continue
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{ctx.rel(fp)}:{i}: {line[:300]}")
                    if len(hits) >= max_hits:
                        return {"ok": True, "hits": hits, "truncated": True}
    return {"ok": True, "hits": hits, "truncated": False}


@tool("delete_file", "删除 run 目录内的单个文件（不能删目录，防误伤）。",
      obj({"path": {"type": "string"}}, ["path"]))
def delete_file(ctx: RunContext, a: dict) -> dict:
    p = ctx.resolve(a["path"])
    if p.is_dir():
        raise ToolError("拒绝删除目录；只允许删单个文件")
    if not p.exists():
        raise ToolError(f"不存在：{a['path']}")
    p.unlink()
    return {"ok": True, "deleted": ctx.rel(p)}


def _human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
