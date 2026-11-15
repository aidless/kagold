"""运行上下文：把一次 run 需要的路径、台账、配置、Kaggle 客户端绑在一起。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .store import Store


@dataclass
class RunContext:
    run_id: int
    comp: str
    root: Path
    settings: config.Settings
    store: Store
    _api: Any = None

    # ---------- kaggle ----------
    def kaggle(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    # ---------- paths ----------
    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def oof_dir(self) -> Path:
        return self.root / "oof"

    @property
    def sub_dir(self) -> Path:
        return self.root / "sub"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def resolve(self, rel: str) -> Path:
        """相对 run 根目录解析；绝对路径必须落在 run 根内。"""
        p = Path(rel)
        if p.is_absolute():
            target = p.resolve()
            root = self.root.resolve()
            if root not in target.parents and target != root:
                raise PermissionError(f"路径逃逸：{rel} 不在 run 目录 {root} 内")
            return target
        target = (self.root / p).resolve()
        root = self.root.resolve()
        if root not in target.parents and target != root:
            raise PermissionError(f"路径逃逸：{rel}")
        return target

    def rel(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(p)

    # ---------- gates ----------
    def gate(self, name: str) -> bool:
        return self.store.gate_pass(self.run_id, name)

    def gates(self) -> dict:
        return self.store.gate_state(self.run_id)

    def phase(self) -> str:
        r = self.store.run(self.run_id) or {}
        return r.get("phase") or "SELECT"
