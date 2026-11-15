"""工具注册表：把 python 函数暴露成 OpenAI function-calling 工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..runtime import RunContext


class ToolError(Exception):
    """可预期的失败（参数错、文件不存在、门禁未过）。会原样回灌给模型。"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[RunContext, dict], Any]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


def tool(name: str, description: str, parameters: dict | None = None):
    def deco(fn: Callable[[RunContext, dict], Any]) -> Tool:
        return Tool(name=name, description=description,
                    parameters=parameters or {"type": "object", "properties": {}}, fn=fn)
    return deco


class ToolBox:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"重复工具名：{t.name}")
        self._tools[t.name] = t

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def call(self, ctx: RunContext, name: str, args: dict) -> dict:
        t = self._tools.get(name)
        if not t:
            return {"ok": False, "error": f"未知工具 {name!r}；可用：{self.names()}"}
        if not isinstance(args, dict):
            return {"ok": False, "error": f"参数必须是 JSON 对象，收到 {type(args).__name__}"}
        if "__bad_json__" in args:
            return {"ok": False, "error": f"工具参数不是合法 JSON：{args['__bad_json__'][:400]}"}
        # 必填校验（JSON schema 的最小实现）
        required = (t.parameters or {}).get("required") or []
        missing = [k for k in required if k not in args]
        if missing:
            return {"ok": False, "error": f"{name} 缺少必填参数：{missing}"}
        try:
            out = t.fn(ctx, args)
            if isinstance(out, dict) and "ok" in out:
                return out
            return {"ok": True, "data": out}
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except PermissionError as e:
            return {"ok": False, "error": f"权限/路径错误：{e}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}
