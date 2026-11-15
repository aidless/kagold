"""受限 shell：cwd 固定在 run 目录，带危险命令黑名单、超时与输出截断。

这不是安全沙箱（本机 danger-full-access），而是「防手滑」+「防 agent 越界」的护栏：
- 黑名单命中直接拒绝并记日志；
- 所有命令都会被写进台账，事后可审计。
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from .base import ToolError, obj, tool
from ..runtime import RunContext

OUT_HEAD = 1500
OUT_TAIL = 6000

FORBIDDEN = [
    (r"rm\s+(-[a-zA-Z]*\s+)*-?r?f?[a-zA-Z]*\s+/(\s|$)", "禁止递归删除根目录"),
    (r"\bmkfs\b", "禁止格式化磁盘"),
    (r"\bdd\b[^\n]*of=/dev/", "禁止直接写块设备"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b", "禁止关机/重启"),
    (r":\(\)\s*\{.*\};\s*:", "禁止 fork 炸弹"),
    (r"\bchmod\s+-R\s+777\s+/(\s|$)", "禁止全盘放开权限"),
    (r"\bgit\s+push\b", "禁止 push 到远端（研究代码不自动发布）"),
    (r"\bkaggle\s+datasets\s+(create|version)\b", "禁止发布数据集"),
    (r"\bsystemctl\s+(restart|stop|disable|enable)\s+dsh", "禁止操控 dsh 服务"),
    (r"\b(sudo|su)\b", "禁止提权"),
    (r"(curl|wget)[^\n|]*\|\s*(ba)?sh", "禁止 curl|sh 一键安装"),
    (r">\s*/etc/", "禁止写 /etc"),
    (r"~?/?\.ssh/", "禁止触碰 SSH 凭据"),
]


def _guard(cmd: str) -> None:
    for pat, why in FORBIDDEN:
        if re.search(pat, cmd):
            raise ToolError(f"命令被黑名单拦截（{why}）：{cmd[:200]}")


def _pypath(p: Path) -> str:
    """返回可安全放进 PYTHONPATH 的路径。

    踩过的坑：某些工作区的根目录名里字面含反斜杠，而 python 在解析
    PYTHONPATH 时会把 `\\d` 之类当转义，导致子进程 `import` 直接找不到模块。
    这里对有反斜杠的路径建一个无歧义的符号链接（按内容哈希，不同 run 互不干扰）。
    """
    s = str(p)
    if "\\" not in s:
        return s
    import hashlib
    tag = hashlib.sha1(s.encode()).hexdigest()[:10]
    link = Path.home() / ".cache" / "kagold" / "pypath" / tag
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == p.resolve():
            return str(link)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.resolve(), target_is_directory=True)
        return str(link)
    except Exception:
        return s


def _env(ctx: RunContext) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _pypath(ctx.src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.pop("KAGOLD_API_KEY", None)  # 不要把密钥泄漏给子进程输出
    return env


@tool("sh",
      "在 run 目录执行 bash 命令（非交互、有超时）。用于跑脚本、统计文件、调用 kaggle 等。"
      "长任务请在命令里用 nohup + 日志文件，或分小步跑，避免超时。",
      obj({"cmd": {"type": "string"},
           "timeout": {"type": "integer", "description": "秒，默认 600，上限由配置决定"}}, ["cmd"]))
def sh(ctx: RunContext, a: dict) -> dict:
    cmd = a["cmd"]
    _guard(cmd)
    cap = int(getattr(ctx.settings, "max_shell_seconds", 1800))
    timeout = min(int(a.get("timeout") or 600), cap)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(ctx.root), env=_env(ctx),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        code = proc.returncode
        out, err = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        code = 124
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = f"[timeout after {timeout}s]"
    ms = int((time.time() - t0) * 1000)
    ctx.store.bump(ctx.run_id, shell_calls=1)
    hint = "exit_code!=0 通常是脚本报错，先读 stderr 尾部再改。" if code else ""
    if code and "No such file or directory" in err and ("\\" in cmd or "/home/" in cmd):
        hint = ("路径被 bash 的转义吃掉了：工作区根目录名里含反斜杠，绝对路径必须加引号或用单引号。"
                "更省事的做法：本工具 cwd 已经是 run 根目录，直接用相对路径（python3 src/train.py）。")
    return {
        "ok": True,
        "exit_code": code,
        "duration_ms": ms,
        "stdout": _trim(out),
        "stderr": _trim(err),
        "hint": hint,
    }


@tool("python",
      "在 run 目录执行一段 Python（新进程，cwd=run 根，PYTHONPATH 含 src/）。"
      "适合读数据、算 OOF、验证 metric；不适合长训练（用 sh 跑脚本）。",
      obj({"code": {"type": "string"},
           "timeout": {"type": "integer", "description": "秒，默认 600"}}, ["code"]))
def python(ctx: RunContext, a: dict) -> dict:
    code = a["code"]
    snippet = ctx.log_dir / f"_snippet_{int(time.time()*1000)}.py"
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_text(code, encoding="utf-8")
    cap = int(getattr(ctx.settings, "max_shell_seconds", 1800))
    timeout = min(int(a.get("timeout") or 600), cap)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python3", str(snippet)], cwd=str(ctx.root), env=_env(ctx),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        code_rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        code_rc, out, err = 124, "", f"[timeout after {timeout}s]"
    ms = int((time.time() - t0) * 1000)
    return {"ok": True, "exit_code": code_rc, "duration_ms": ms,
            "stdout": _trim(out), "stderr": _trim(err), "script": ctx.rel(snippet)}


def _trim(text: str) -> str:
    if not text:
        return ""
    if len(text) <= OUT_HEAD + OUT_TAIL:
        return text
    return text[:OUT_HEAD] + f"\n...[截断 {len(text) - OUT_HEAD - OUT_TAIL} 字符]...\n" + text[-OUT_TAIL:]
