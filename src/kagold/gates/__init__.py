"""四道硬门禁：metric / folds / oof / submit。

设计原则（来自本工作区 AGENTS.md 第 8 条「门必须能失败」）：
- 每道门都有明确的判据，且判据写在代码里，不写在 prompt 里；
- 每道门都能被坏输入打穿（tests/selftest_gates.py 有反证用例）；
- 每道门的结果写进 SQLite，后续门禁与提交只认台账。
"""
from .metric import run_metric_gate
from .folds import run_fold_gate
from .oof import run_oof_gate
from .submit import run_submit_gate

__all__ = ["run_metric_gate", "run_fold_gate", "run_oof_gate", "run_submit_gate"]
