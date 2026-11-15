"""LLM 客户端：OpenAI 兼容协议 + 重试 + token 记账。

不引入任何 agent 框架——循环、工具调度、预算都由 kagold 自己控制，
这样门禁可以真的挡住提交，而不是靠模型自觉。
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


class LLMError(RuntimeError):
    pass


class ContextOverflow(LLMError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    raw_args: str = ""


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str = ""
    raw: dict | None = None


class LLM:
    def __init__(self, *, base_url: str, api_key: str, model: str,
                 timeout: float = 300.0, max_retries: int = 4, temperature: float = 0.25):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.total_in = 0
        self.total_out = 0
        self.calls = 0

    # ---- helpers ----
    @staticmethod
    def _to_messages(messages: list[dict], tools: list[dict] | None) -> tuple[list[dict], list[dict] | None]:
        """把内部消息格式转成 OpenAI wire 格式（已是同一套，只做浅拷贝防御）。"""
        wire = []
        for m in messages:
            mm = {k: v for k, v in m.items() if k in (
                "role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content")}
            if mm.get("role") == "assistant" and "tool_calls" in mm and not mm["tool_calls"]:
                mm.pop("tool_calls")
            if mm.get("content") is None and mm.get("role") != "assistant":
                mm["content"] = ""
            wire.append(mm)
        return wire, tools

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 4096, temperature: float | None = None) -> Reply:
        wire, tspec = self._to_messages(messages, tools)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=wire,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens,
                )
                if tspec:
                    kwargs["tools"] = tspec
                    kwargs["tool_choice"] = "auto"
                resp = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                msg = resp.choices[0].message
                usage = resp.usage
                tin = int(getattr(usage, "prompt_tokens", 0) or 0)
                tout = int(getattr(usage, "completion_tokens", 0) or 0)
                self.total_in += tin
                self.total_out += tout
                calls: list[ToolCall] = []
                for tc in (msg.tool_calls or []):
                    raw = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw) if raw.strip() else {}
                    except json.JSONDecodeError:
                        args = {"__bad_json__": raw}
                    calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args, raw_args=raw))
                return Reply(
                    content=msg.content or "",
                    tool_calls=calls,
                    tokens_in=tin,
                    tokens_out=tout,
                    finish_reason=resp.choices[0].finish_reason or "",
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                text = f"{type(e).__name__}: {e}"
                low = text.lower()
                if "context" in low and ("length" in low or "too long" in low or "maximum" in low):
                    raise ContextOverflow(text) from e
                retryable = any(k in low for k in (
                    "rate limit", "429", "timeout", "timed out", "connection",
                    "502", "503", "504", "500", "overloaded", "internal", "temporarily"))
                if attempt >= self.max_retries or not retryable:
                    raise LLMError(f"LLM 调用失败（{self.model}）：{text}") from e
                sleep = min(60.0, (2 ** attempt) + random.random())
                time.sleep(sleep)
        raise LLMError(f"LLM 调用失败：{last_err}")
