"""门禁 1：度量复现（metric gate）。

为什么这是第一道门：整个竞赛里所有"改进"都是由 metric 定义的。
如果 metric 实现错了 1%（例如 logloss 没 clip、RMSLE 忘了 log1p、AUC 用了错误的正类），
后面所有 OOF、集成权重、提交选择全都在优化一个错的东西。这是 Kaggle 最贵的错误，且几乎从不可见。

门禁做法：加载 agent 写的 `src/metric.py: score(y_true, y_pred)`，
在 5 组确定性合成数据上与**独立参考实现**逐例比对（相对误差 ≤ 1e-6），
再检查"更好/更差"的方向是否与参考一致（防止符号搞反）。
任何一例不一致 → FAIL，并把每例的 ref/agent 差值写进证据。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import RunContext
from . import reference as ref


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"kagold_metric_{path.stem}_{abs(hash(str(path))) % 10**6}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases(kind: str, canonical: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(13)
    hard = canonical in ("accuracy", "f1", "mcc", "qwk")
    t, p = ref.synthetic(kind, n=200, seed=7)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    if kind == "regression":
        out.append(("perfect", t, t.copy()))
        out.append(("noisy", t, p))
        out.append(("constant", t, np.full_like(t, float(t.mean()))))
        out.append(("small_n10", t[:10], p[:10]))
        out.append(("sign_flip", t, -t))
    else:
        if kind == "binary":
            labels = t.astype(int)
            if hard:
                out.append(("perfect_labels", labels, labels.copy()))
                out.append(("noisy_labels", labels, (p >= 0.5).astype(int)))
                out.append(("constant_labels", labels, np.zeros_like(labels)))
                out.append(("small_n40", labels[:40], (p[:40] >= 0.5).astype(int)))
                out.append(("inverted_labels", labels, 1 - labels))
            else:
                oh = np.eye(2)[labels]
                out.append(("perfect_probs", labels, oh))
                out.append(("noisy_probs", labels, p))
                out.append(("constant_probs", labels, np.full_like(p, 0.5)))
                out.append(("small_n40", labels[:40], p[:40]))
                out.append(("inverted_probs", labels, 1 - p))
        else:  # ordinal
            labels = t.astype(int)
            out.append(("perfect", labels, labels.copy()))
            out.append(("noisy", labels, p.astype(int)))
            out.append(("constant", labels, np.full_like(labels, 1)))
            out.append(("small_n40", labels[:40], p[:40].astype(int)))
            out.append(("shifted", labels, (labels + 1) % 4))
    return out


def run_metric_gate(ctx: RunContext, metric_name: str, module_rel: str = "src/metric.py",
                    tol: float = 1e-6) -> dict:
    canonical = ref.canon(metric_name)
    evidence: dict[str, Any] = {
        "gate": "metric",
        "metric_requested": metric_name,
        "canonical": canonical,
        "module": module_rel,
        "tolerance": tol,
        "cases": [],
        "problems": [],
    }

    def finish(passed: bool, reason: str) -> dict:
        evidence["passed"] = passed
        evidence["reason"] = reason
        ctx.store.record_gate(ctx.run_id, "metric", passed, evidence)
        _dump(ctx, "metric", evidence)
        return {"passed": passed, "reason": reason, "evidence": evidence}

    if not canonical:
        return finish(False, f"metric {metric_name!r} 未在参考实现中登记；"
                             f"可选：{sorted(ref.REGISTRY)}。如属竞赛自定义 metric，请先在 "
                             f"gates/reference.py 添加参考实现（这一步必须人工确认）")
    meta = ref.REGISTRY[canonical]
    path = ctx.resolve(module_rel)
    if not path.exists():
        return finish(False, f"缺少 {module_rel}；请实现 def score(y_true, y_pred) -> float")
    try:
        mod = _load_module(path)
    except Exception as e:  # noqa: BLE001
        return finish(False, f"无法导入 {module_rel}：{type(e).__name__}: {e}")
    fn = getattr(mod, "score", None)
    if not callable(fn):
        return finish(False, f"{module_rel} 里没有可调用的 score(y_true, y_pred)")
    evidence["module_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:32]

    worse = 0
    for name, yt, yp in _cases(meta["kind"], canonical):
        try:
            expected = float(meta["fn"](yt, yp))
        except Exception as e:  # noqa: BLE001
            evidence["problems"].append(f"参考实现自检失败（{name}）：{e}")
            continue
        try:
            got = float(fn(yt.copy(), yp.copy()))
        except Exception as e:  # noqa: BLE001
            evidence["cases"].append({"case": name, "expected": expected,
                                      "error": f"{type(e).__name__}: {e}"})
            evidence["problems"].append(f"case {name} 抛异常：{type(e).__name__}: {e}")
            continue
        diff = abs(got - expected)
        rel = diff / max(1.0, abs(expected))
        ok = (math.isfinite(got) and (rel <= tol or diff <= 1e-9))
        evidence["cases"].append({"case": name, "expected": round(expected, 10),
                                  "got": round(got, 10), "abs_diff": float(f"{diff:.3e}"),
                                  "rel_diff": float(f"{rel:.3e}"), "ok": ok})
        if not ok:
            evidence["problems"].append(
                f"case {name}: 期望 {expected:.6f}，你的实现给出 {got:.6f}")
        else:
            worse += 1

    # 方向一致性：参考认为"越好"的 metric，perfect 必须优于 noisy
    case_map = {c["case"]: c.get("got") for c in evidence["cases"] if "got" in c}
    perfect_key = next((k for k in ("perfect", "perfect_probs", "perfect_labels") if k in case_map), None)
    noisy_key = next((k for k in ("noisy", "noisy_probs", "noisy_labels") if k in case_map), None)
    if perfect_key and noisy_key and case_map[perfect_key] is not None and case_map[noisy_key] is not None:
        better = (case_map[perfect_key] > case_map[noisy_key]) if meta["greater_is_better"] \
            else (case_map[perfect_key] < case_map[noisy_key])
        evidence["direction_ok"] = bool(better)
        if not better:
            evidence["problems"].append(
                f"方向错误：{canonical} 应当 {'越大越好' if meta['greater_is_better'] else '越小越好'}，"
                f"但 perfect={case_map[perfect_key]:.6f} 反而劣于 noisy={case_map[noisy_key]:.6f}")

    passed = bool(not evidence["problems"] and evidence["cases"]
                 and worse == len(evidence["cases"]))
    reason = ("度量复现通过：5 组用例与参考实现一致，方向正确"
              if passed else f"度量复现失败，{len(evidence['problems'])} 个问题")
    return finish(passed, reason)


def _dump(ctx: RunContext, name: str, evidence: dict) -> None:
    try:
        p = ctx.oof_dir / f"gate_{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
