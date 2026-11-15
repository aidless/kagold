"""主循环：模型 → 工具 → 结果 → 模型，直到完成或预算耗尽。

设计要点（都是踩过坑才这么写的）：
- 每一条消息落库，断点续跑靠 DB 而不是内存；
- 上下文按"完整工具调用组"裁剪，绝不把 assistant.tool_calls 和它的结果拆开；
- 预算是硬停止（步数/token/时长/磁盘），不是提示词里的建议；
- 连续空转、重复同一调用都会被判定并纠正。
"""
from __future__ import annotations

import json
import time
import types
import traceback
from collections import deque
from typing import Any, Callable

from ..config import Settings, free_gb
from ..llm import ContextOverflow, LLM, LLMError
from ..runtime import RunContext
from ..store import Store
from ..tools.base import ToolBox
from . import prompts

# 上下文预算（字符数，粗略 4 字符 ≈ 1 token）
MAX_CONTEXT_CHARS = 900_000
TRIM_TARGET_CHARS = 420_000
KEEP_RECENT_GROUPS = 8


class Agent:
    def __init__(self, ctx: RunContext, llm: LLM, toolbox: ToolBox,
                 settings: Settings | None = None, max_steps: int | None = None,
                 log: Callable[[str], None] | None = None):
        self.ctx = ctx
        self.llm = llm
        self.tb = toolbox
        self.s = settings or ctx.settings
        self.max_steps = int(max_steps or self.s.max_steps)
        self.log = log or (lambda m: print(m, flush=True))
        self.finished = False
        self.stop_reason: str | None = None
        self.status = "running"
        self._no_tool = 0
        self._sigs: deque[str] = deque(maxlen=8)
        self._next_idx = 1
        self.session_steps = 0
        self.t0 = time.time()

    # ---------------- 主循环 ----------------
    def run(self) -> dict:
        msgs = self._load_or_bootstrap()
        self.log(f"[kagold] run {self.ctx.run_id} · {self.ctx.comp} · 已加载 {len(msgs)} 条消息")
        try:
            while True:
                reason = self._budget_stop()
                if reason:
                    self.stop_reason = reason
                    break
                self.ctx.store.bump(self.ctx.run_id, steps=1)
                self.session_steps += 1
                run = self.ctx.store.run(self.ctx.run_id) or {}
                step_no = int(run.get("steps") or 0)
                self.log(f"\n[kagold] ── step {self.session_steps}/{self.max_steps}"
                         f"（累计 {step_no}）phase={run.get('phase')} "
                         f"tokens_out={run.get('tokens_out')}")

                try:
                    reply = self.llm.chat(msgs, self.tb.specs(),
                                          max_tokens=4096, temperature=self.s.temperature)
                except ContextOverflow as e:
                    self.log(f"[kagold] 上下文溢出，强制裁剪后重试：{e}")
                    msgs = self._trim(msgs, force=True)
                    self._persist("user", {"role": "user", "content":
                                           "（上下文已裁剪，请基于当前状态继续推进。）"})
                    reply = self.llm.chat(msgs, self.tb.specs(), max_tokens=4096,
                                          temperature=self.s.temperature)

                self.ctx.store.bump(self.ctx.run_id,
                                    tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)
                if reply.content.strip():
                    self.log(f"[kagold] {reply.content.strip()[:400]}")

                a_msg = {"role": "assistant", "content": reply.content or ""}
                if reply.tool_calls:
                    a_msg["tool_calls"] = [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name,
                                     "arguments": tc.raw_args or json.dumps(tc.args, ensure_ascii=False)},
                    } for tc in reply.tool_calls]
                self._append(msgs, a_msg, tin=reply.tokens_in, tout=reply.tokens_out)

                if reply.tool_calls:
                    self._no_tool = 0
                    if self._exec_calls(reply.tool_calls, msgs, step_no):
                        self.finished = True
                        self.status = "finished"
                        self.stop_reason = "agent 主动结束"
                        break
                else:
                    self._nudge(msgs)

                msgs = self._trim(msgs)
        except KeyboardInterrupt:
            self.status = "paused"
            self.stop_reason = "用户中断"
        except StallStop as e:
            self.status = "stalled"
            self.stop_reason = str(e)
        except LLMError as e:
            self.status = "error"
            self.stop_reason = f"LLM 错误：{e}"
        except Exception as e:  # noqa: BLE001
            self.status = "error"
            self.stop_reason = f"{type(e).__name__}: {e}"
            self.log("[kagold] 未捕获异常：\n" + traceback.format_exc()[-3000:])
        finally:
            report = self._close()
        return report

    # ---------------- 工具执行 ----------------
    def _exec_calls(self, calls, msgs: list[dict], step_no: int) -> bool:
        sig = "|".join(f"{c.name}:{json.dumps(c.args, sort_keys=True, ensure_ascii=False)}" for c in calls)
        self._sigs.append(sig)
        for tc in calls:
            t0 = time.time()
            result = self.tb.call(self.ctx, tc.name, tc.args)
            ms = int((time.time() - t0) * 1000)
            ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
            self.ctx.store.log_tool(self.ctx.run_id, step_no, tc.name, tc.args, ok, result, ms)
            text = json.dumps(result, ensure_ascii=False, default=str)
            if len(text) > self.s.max_tool_output:
                text = text[: self.s.max_tool_output] + f"…[输出被截断，原长 {len(text)}]"
            self.log(f"[tool] {tc.name} {'ok' if ok else 'FAIL'} {ms}ms "
                     f"{'' if ok else text[:300]}")
            self._append(msgs, {"role": "tool", "tool_call_id": tc.id,
                                "name": tc.name, "content": text})
            if isinstance(result, dict) and result.get("finished"):
                return True
        return False

    def _nudge(self, msgs: list[dict]) -> None:
        self._no_tool += 1
        if self._no_tool == 1:
            text = ("你这一轮没有调用任何工具。请立刻用工具推进下一步："
                    "如果还没登记比赛就 record_competition；如果任务已完成就调用 finish。")
        elif self._no_tool == 2:
            text = "再次提醒：不要只在文字里描述计划，直接调用工具。"
        else:
            text = ("你已连续多轮空转。请二选一：① 调用 finish 说明原因；"
                    "② 调用一个具体工具（read_file / python / sh / gate_* 等）继续。")
        self._append(msgs, {"role": "user", "content": text})
        if self._no_tool >= 4:
            self.status = "stalled"
            self.stop_reason = "连续 4 轮未调用工具"
            raise StallStop(self.stop_reason)

    # ---------------- 消息管理 ----------------
    def _load_or_bootstrap(self) -> list[dict]:
        msgs = self.ctx.store.messages(self.ctx.run_id)
        if msgs:
            self._next_idx = 1
            return self._trim(msgs, force=len(msgs) > 60)
        msgs = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.task_message(self.ctx, self.ctx.comp)},
        ]
        # 逐条落库（bootstrap 不参与裁剪）
        self._next_idx = 1
        self.session_steps = 0
        for m in list(msgs):
            self.ctx.store.append_message(self.ctx.run_id, self._next_idx, m)
            self._next_idx += 1
        return msgs

    def _append(self, msgs: list[dict], msg: dict, tin: int = 0, tout: int = 0) -> None:
        msgs.append(msg)
        self.ctx.store.append_message(self.ctx.run_id, self._next_idx, msg, tin, tout)
        self._next_idx += 1

    def _persist(self, role: str, msg: dict) -> None:
        self.ctx.store.append_message(self.ctx.run_id, self._next_idx, msg)
        self._next_idx += 1

    # ---------------- 裁剪 ----------------
    @staticmethod
    def _size(msgs: list[dict]) -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)

    def _trim(self, msgs: list[dict], force: bool = False) -> list[dict]:
        size = self._size(msgs)
        if not force and size <= MAX_CONTEXT_CHARS:
            return msgs
        head = [m for m in msgs[:2]]
        rest = msgs[2:]
        groups = _group(rest)
        kept: list[list[dict]] = []
        total = sum(len(json.dumps(m, ensure_ascii=False)) for m in head)
        for g in reversed(groups):
            gs = sum(len(json.dumps(m, ensure_ascii=False)) for m in g)
            if len(kept) >= KEEP_RECENT_GROUPS and total + gs > TRIM_TARGET_CHARS:
                break
            kept.append(g)
            total += gs
        kept.reverse()
        dropped = len(groups) - len(kept)
        out = list(head)
        if dropped > 0:
            out.append({"role": "user", "content":
                        f"（上下文已裁剪：省略了较早的 {dropped} 组工具交互。"
                        f"需要回顾时用 read_file 看 logs/ 或 oof/ 里的结果文件。）"})
        for g in kept:
            out.extend(g)
        self.log(f"[kagold] 上下文裁剪 {size}→{self._size(out)} 字符，省略 {dropped} 组")
        return out

    # ---------------- 停止条件 ----------------
    def _budget_stop(self) -> str | None:
        run = self.ctx.store.run(self.ctx.run_id) or {}
        if run.get("status") not in (None, "running"):
            return f"台账状态为 {run.get('status')}，停止"
        # --max-steps 的语义是「本次调用最多再跑几步」，不是整个 run 的总预算；
        # 否则续跑一个已经 30 步的 run 时，传任何小值都会立刻退出。
        if self.session_steps >= self.max_steps:
            return (f"本次步数上限 {self.max_steps}（累计 {run.get('steps')}）；"
                    f"续跑用 --resume {self.ctx.run_id} 再加步数")
        total_cap = int(getattr(self.s, "max_total_steps", 600) or 600)
        if int(run.get("steps") or 0) >= total_cap:
            return f"累计步数达到硬上限 {total_cap}"
        if int(run.get("tokens_out") or 0) >= self.s.max_tokens_out:
            return f"达到输出 token 上限 {self.s.max_tokens_out}"
        if int(run.get("tokens_in") or 0) >= self.s.max_tokens_in:
            return f"达到输入 token 上限 {self.s.max_tokens_in}"
        # 时长上限同样是「本次调用」的墙钟；否则续跑旧 run 会立刻超时。
        session_h = (time.time() - self.t0) / 3600
        if session_h >= self.s.max_hours:
            return f"本次时长上限 {self.s.max_hours}h（已跑 {session_h:.2f}h）"
        total_h = (time.time() - float(run.get("created_at") or time.time())) / 3600
        cap_h = float(getattr(self.s, "max_total_hours", 24.0) or 24.0)
        if total_h >= cap_h:
            return f"累计时长达到硬上限 {cap_h}h"
        gb = free_gb()
        if gb < self.s.min_free_gb:
            return f"磁盘剩余 {gb:.1f}GB < 下限 {self.s.min_free_gb}GB"
        return None

    # ---------------- 收尾 ----------------
    def _close(self) -> dict:
        run = self.ctx.store.run(self.ctx.run_id) or {}
        status = self.status if self.status != "running" else "paused"
        self.ctx.store.update_run(self.ctx.run_id, status=status,
                                  note=(self.stop_reason or "")[:2000])
        try:
            path = self._write_report(run, status)
        except Exception as e:  # noqa: BLE001
            self.log(f"[kagold] 报告写入失败：{e}")
            path = None
        summary = {
            "run_id": self.ctx.run_id, "comp": self.ctx.comp, "status": status,
            "stop_reason": self.stop_reason,
            "steps": run.get("steps"), "tokens_in": self.llm.total_in,
            "tokens_out": self.llm.total_out, "phase": run.get("phase"),
            "gates": {k: v.get("passed") for k, v in self.ctx.store.gate_state(self.ctx.run_id).items()},
            "report": str(path) if path else None,
        }
        self.log(f"[kagold] 结束：{json.dumps(summary, ensure_ascii=False)}")
        return summary

    def _write_report(self, run: dict, status: str) -> str:
        st = self.ctx.store
        gates = st.gate_state(self.ctx.run_id)
        comp = st.kv_get(self.ctx.run_id, "competition") or {}
        best = st.kv_get(self.ctx.run_id, "oof_best") or {}
        subs = st.submissions(self.ctx.run_id)
        notes = ""
        np_ = self.ctx.log_dir / "notes.md"
        if np_.exists():
            notes = np_.read_text(errors="ignore").strip()[-4000:]
        lines = [
            f"# kagold 运行报告 · {self.ctx.comp}",
            "",
            f"- run_id: {self.ctx.run_id}",
            f"- 状态: {status}（{self.stop_reason or '—'}）",
            f"- 阶段: {run.get('phase')} · 步数 {run.get('steps')} · "
            f"tokens in/out {run.get('tokens_in')}/{run.get('tokens_out')}",
            f"- 用时: {(time.time() - float(run.get('created_at') or time.time())) / 3600:.2f}h",
            "",
            "## 比赛信息",
            "```json",
            json.dumps(comp, ensure_ascii=False, indent=1)[:2000],
            "```",
            "",
            "## 门禁",
        ]
        for name in ("metric", "folds", "oof", "submit"):
            g = gates.get(name) or {}
            lines.append(f"- `{name}`: {'PASS' if g.get('passed') else 'FAIL'}")
        lines += ["", "## 当前最优", "```json",
                  json.dumps(best, ensure_ascii=False, indent=1)[:1500], "```", "", "## 提交"]
        if subs:
            live = _live_scores(self.ctx)      # 榜分可能滞后于提交，报告里现取一次
            for s in subs:
                ref = str(s.get("ref") or "")
                score = live.get(ref)
                pub = s.get("public_score")
                priv = s.get("private_score")
                if score:
                    pub, priv = score
                at = time.strftime("%m-%d %H:%M", time.localtime(float(s.get("created_at") or 0)))
                pub = "—" if pub in (None, "") else pub
                priv = "—" if priv in (None, "") else priv
                lines.append(f"- {at}: `{s.get('file')}` → public={pub} private={priv}"
                             + ("" if score else "（榜分未取到，可 my_submissions 复查）"))
        else:
            lines.append("- （无）")
        lines += ["", "## 笔记", notes or "（无）", ""]
        p = self.ctx.root / "report.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        return str(p)


def _live_scores(ctx) -> dict[str, tuple]:
    """拉一次本竞赛的提交列表，把 ref → (public, private) 映射出来。失败就返回空。"""
    try:
        api = ctx.kaggle()
        rows = api.competition_submissions(ctx.comp, page_size=50) or []
        out = {}
        for r in rows:
            ref = str(getattr(r, "ref", "") or "")
            if ref:
                out[ref] = (getattr(r, "public_score", None), getattr(r, "private_score", None))
        return out
    except Exception:  # noqa: BLE001
        return {}


def render_report(ctx, run: dict, status: str | None = None) -> str:
    """不依赖 Agent 实例生成/刷新 report.md（CLI 的 `report` 子命令用）。"""
    shim = types.SimpleNamespace(ctx=ctx, stop_reason=run.get("stop_reason"))
    return Agent._write_report(shim, run, status or run.get("status") or "unknown")


class StallStop(Exception):
    pass


def _group(msgs: list[dict]) -> list[list[dict]]:
    """把消息切成"完整交互组"：assistant(tool_calls) + 其后所有 tool 结果；其它各成一组。"""
    groups: list[list[dict]] = []
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            g = [m]
            i += 1
            while i < n and msgs[i].get("role") == "tool":
                g.append(msgs[i])
                i += 1
            groups.append(g)
        else:
            groups.append([m])
            i += 1
    # 丢掉末尾不完整的组（assistant 有 tool_calls 但没有结果）
    if groups and groups[-1] and groups[-1][0].get("role") == "assistant" \
            and groups[-1][0].get("tool_calls") and len(groups[-1]) == 1:
        groups.pop()
    return groups
