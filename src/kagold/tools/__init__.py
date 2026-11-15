"""工具装配：一次性把全部工具注册进 ToolBox。"""
from __future__ import annotations

from .base import Tool, ToolBox, ToolError, obj, tool
from . import ensemble as _ensemble
from . import fs as _fs
from . import gates as _gates
from . import kaggle as _kaggle
from . import shell as _shell
from . import wait as _wait


def build_toolbox() -> ToolBox:
    tb = ToolBox()
    for mod in (_fs, _shell, _kaggle, _gates, _ensemble, _wait):
        for name in dir(mod):
            obj_ = getattr(mod, name)
            if isinstance(obj_, Tool) and not name.startswith("_"):
                tb.add(obj_)
    return tb


__all__ = ["build_toolbox", "ToolBox", "Tool", "ToolError", "obj", "tool"]
