#!/usr/bin/env python3
"""门禁自检：每道门都要有「能通过」和「能拦住」两类用例。

这条纪律来自工作区 AGENTS.md 第 8 条：门必须能失败，否则只是装饰。
用法：python3 tests/selftest.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kagold.config import Settings                      # noqa: E402
from kagold.gates import run_fold_gate, run_metric_gate, run_oof_gate, run_submit_gate  # noqa: E402
from kagold.gates import reference as ref              # noqa: E402
from kagold.runtime import RunContext                  # noqa: E402
from kagold.store import Store                         # noqa: E402
from kagold.tools import build_toolbox                 # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- fixtures
def make_ctx(base: Path, comp: str, store: Store, settings: Settings | None = None) -> RunContext:
    root = base / comp
    for sub in ("input", "src", "oof", "sub", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    run_id = store.new_run(comp, goal="selftest", config={"test": True})
    return RunContext(run_id=run_id, comp=comp, root=root,
                      settings=settings or Settings(), store=store)


def build_data(ctx: RunContext, n: int = 600, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    target = rng.normal(50, 15, n)
    order = np.argsort(target)
    fold = np.empty(n, dtype=int)
    fold[order] = np.arange(n) % 3
    ids = [f"r{i:04d}" for i in range(n)]
    pd.DataFrame({"id": ids, "target": target, "fold": fold}).to_csv(ctx.input_dir / "train.csv", index=False)
    pd.DataFrame({"id": ids}).to_csv(ctx.input_dir / "test.csv", index=False)
    pd.DataFrame({"id": ids, "target": np.full(n, 50.0)}).to_csv(
        ctx.input_dir / "sample_submission.csv", index=False)
    pred = target + rng.normal(0, 3, n)
    pd.DataFrame({"id": ids, "pred": pred}).to_csv(ctx.oof_dir / "oof.csv", index=False)
    pd.DataFrame({"id": ids, "target": pred}).to_csv(ctx.sub_dir / "submission.csv", index=False)


GOOD_METRIC = """\
import numpy as np

def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
"""

BAD_METRIC_MSE = """\
import numpy as np

def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))   # 故意写错：MSE 当成 RMSE
"""

BAD_METRIC_SIGN = """\
import numpy as np

def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return -float(np.sqrt(np.mean((y_true - y_pred) ** 2)))   # 方向反了
"""


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="kagold-selftest-"))
    store = Store(base / "test.db")
    try:
        # ---------- 1. 参考度量本身 ----------
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.0, 2.0, 3.0, 5.0])
        check("参考 rmse 手算一致", abs(ref.score("rmse", y, p) - np.sqrt(0.25)) < 1e-12)
        check("metric 别名归一", ref.canon("Root Mean Squared Error") == "rmse"
              and ref.canon("AUC") == "auc" and ref.canon("quadratic_weighted_kappa") == "qwk")
        check("方向标注正确", ref.greater_is_better("auc") and not ref.greater_is_better("rmse"))

        # ---------- 2. metric 门 ----------
        ctx = make_ctx(base, "comp-a", store)
        build_data(ctx)
        (ctx.src_dir / "metric.py").write_text(GOOD_METRIC, encoding="utf-8")
        r = run_metric_gate(ctx, "rmse")
        check("metric 门：正确实现应通过", r["passed"], r["reason"])
        check("metric 门：passed 必须是 bool（不是用例列表）",
              isinstance(r["passed"], bool), f"{type(r['passed'])}")

        (ctx.src_dir / "metric.py").write_text(BAD_METRIC_MSE, encoding="utf-8")
        r = run_metric_gate(ctx, "rmse")
        check("metric 门：把 MSE 当 RMSE 必须被拦", not r["passed"], r["reason"])

        (ctx.src_dir / "metric.py").write_text(BAD_METRIC_SIGN, encoding="utf-8")
        r = run_metric_gate(ctx, "rmse")
        check("metric 门：方向写反必须被拦", not r["passed"], r["reason"])

        (ctx.src_dir / "metric.py").unlink()
        r = run_metric_gate(ctx, "rmse")
        check("metric 门：缺文件必须被拦", not r["passed"], r["reason"])

        r = run_metric_gate(ctx, "definitely_not_a_metric")
        check("metric 门：未登记指标必须被拦", not r["passed"], r["reason"])

        (ctx.src_dir / "metric.py").write_text(GOOD_METRIC, encoding="utf-8")
        run_metric_gate(ctx, "rmse")

        # ---------- 3. folds 门 ----------
        tr = pd.read_csv(ctx.input_dir / "train.csv")
        r = run_fold_gate(ctx, "input/train.csv", "fold", "target", "id")
        check("folds 门：均衡三分应通过", r["passed"], r["reason"])
        check("folds 门：passed 必须是 bool", isinstance(r["passed"], bool),
              f"{type(r['passed'])}")

        tr.drop(columns=["fold"]).to_csv(ctx.input_dir / "train_nofold.csv", index=False)
        r = run_fold_gate(ctx, "input/train_nofold.csv", "fold", "target", "id")
        check("folds 门：缺 fold 列必须被拦", not r["passed"], r["reason"])

        leak = tr.copy()
        leak.loc[leak.index[:5], "id"] = "dup_entity"      # 同一实体跨折
        leak.to_csv(ctx.input_dir / "train_leak.csv", index=False)
        r = run_fold_gate(ctx, "input/train_leak.csv", "fold", "target", "id")
        check("folds 门：同一 id 跨折必须被拦", not r["passed"], r["reason"])

        bad_k = tr[tr["fold"] < 2].copy()                  # 只剩 2 折
        bad_k.to_csv(ctx.input_dir / "train_2f.json.csv", index=False)
        r = run_fold_gate(ctx, "input/train_2f.json.csv", "fold", "target", "id")
        check("folds 门：折数不足必须被拦", not r["passed"], r["reason"])

        # ---------- 4. oof 门 ----------
        r = run_oof_gate(ctx, "rmse", "oof/oof.csv", "input/train.csv", "id", "target", "fold")
        check("oof 门：完整 OOF 应通过", r["passed"], r["reason"])
        check("oof 门：passed 必须是 bool", isinstance(r["passed"], bool),
              f"{type(r['passed'])}")
        check("oof 门：给出每折分数与 std",
              bool(r["evidence"].get("models", {}).get("pred", {}).get("fold_scores"))
              and r["evidence"].get("best", {}).get("std") is not None,
              str(r["evidence"].get("models")))

        oof = pd.read_csv(ctx.oof_dir / "oof.csv")
        oof.loc[3, "pred"] = np.nan
        oof.to_csv(ctx.oof_dir / "oof_nan.csv", index=False)
        r = run_oof_gate(ctx, "rmse", "oof/oof_nan.csv", "input/train.csv", "id", "target", "fold")
        check("oof 门：NaN 必须被拦", not r["passed"], r["reason"])

        oof.iloc[:50].to_csv(ctx.oof_dir / "oof_short.csv", index=False)
        r = run_oof_gate(ctx, "rmse", "oof/oof_short.csv", "input/train.csv", "id", "target", "fold")
        check("oof 门：覆盖不全必须被拦", not r["passed"], r["reason"])

        # ---------- 5. submit 门：前置门禁的硬约束 ----------
        ctx_b = make_ctx(base, "comp-b", store)            # 全新 run，没有任何门禁记录
        build_data(ctx_b)
        r = run_submit_gate(ctx_b, "sub/submission.csv")
        check("submit 门：无前置门禁必须被拦", not r["passed"], r["reason"])

        tb = build_toolbox()
        res = tb.call(ctx_b, "submit", {"message": "should be blocked"})
        check("submit 工具：门禁未过时拒绝上传",
              (not res.get("ok", True)) and "门禁" in json.dumps(res, ensure_ascii=False),
              json.dumps(res, ensure_ascii=False)[:200])

        # ---------- 6. submit 门：产物校验 ----------
        for name in ("metric", "folds", "oof"):
            store.record_gate(ctx.run_id, name, True, {"reason": "selftest"})
        r = run_submit_gate(ctx, "sub/submission.csv")
        check("submit 门：格式正确应通过", r["passed"], r["reason"])

        bad = pd.read_csv(ctx.sub_dir / "submission.csv").rename(columns={"target": "prediction"})
        bad.to_csv(ctx.sub_dir / "sub_bad.csv", index=False)
        r = run_submit_gate(ctx, "sub/sub_bad.csv")
        check("submit 门：列名不符必须被拦", not r["passed"], r["reason"])

        mis = pd.read_csv(ctx.sub_dir / "submission.csv").iloc[:-5]
        mis.to_csv(ctx.sub_dir / "sub_short.csv", index=False)
        r = run_submit_gate(ctx, "sub/sub_short.csv")
        check("submit 门：行数不符必须被拦", not r["passed"], r["reason"])

        # 回归：样例文件超过 20 万行时，行数判定不能被 nrows 截断（真实竞赛踩过）
        big_n = 205_000
        big_ids = np.arange(big_n, dtype=np.int64) + 7_000_000
        pd.DataFrame({"id": big_ids, "target": np.linspace(40, 60, big_n)}).to_csv(
            ctx.input_dir / "sample_submission.csv", index=False)
        pd.DataFrame({"id": big_ids, "target": np.linspace(41, 61, big_n)}).to_csv(
            ctx.sub_dir / "sub_big.csv", index=False)
        r = run_submit_gate(ctx, "sub/sub_big.csv")
        check("submit 门：样例 >20 万行时不应误报行数不符", r["passed"],
              f"{r['reason']} | sample_rows={r['evidence'].get('sample_rows')}")

        # 恢复小样例，供后续用例使用
        _tr = pd.read_csv(ctx.input_dir / "train.csv")
        pd.DataFrame({"id": _tr["id"].astype(str), "target": _tr["target"]}).to_csv(
            ctx.input_dir / "sample_submission.csv", index=False)
        run_submit_gate(ctx, "sub/submission.csv")         # 恢复通过状态

        # ---------- 7. 集成工具 ----------
        ids = pd.read_csv(ctx.input_dir / "train.csv")["id"].astype(str).tolist()
        tg = pd.read_csv(ctx.input_dir / "train.csv")["target"].values
        rng = np.random.default_rng(1)
        pa = tg + rng.normal(0, 3, len(tg))
        pb = tg + rng.normal(0, 6, len(tg))
        pd.DataFrame({"id": ids, "pred_a": pa, "pred_b": pb}).to_csv(
            ctx.oof_dir / "oof_multi.csv", index=False)
        res = tb.call(ctx, "ensemble", {"metric": "rmse", "oof_csv": "oof/oof_multi.csv",
                                        "train_csv": "input/train.csv", "method": "greedy",
                                        "id_col": "id", "target_col": "target", "fold_col": "fold"})
        d = res.get("data") or {}
        single = d.get("singles") or {}
        best_single = min(single.values()) if single else 999
        check("ensemble：融合不劣于最优单模型",
              res.get("ok") and float(d.get("blend_score", 999)) <= best_single + 1e-9,
              json.dumps(d, ensure_ascii=False)[:300])
        check("ensemble：写出 blend 文件", (ctx.oof_dir / "blend.csv").exists())

        # ---------- 8. 阶段机 ----------
        res = tb.call(ctx_b, "set_phase", {"phase": "DONE"})
        check("set_phase：不能跳阶段", not res.get("ok", True),
              json.dumps(res, ensure_ascii=False)[:160])
        res = tb.call(ctx_b, "set_phase", {"phase": "REPRO"})
        check("set_phase：未登记竞赛不能进 REPRO", not res.get("ok", True),
              json.dumps(res, ensure_ascii=False)[:160])
        tb.call(ctx_b, "record_competition", {"ref": "comp-b", "metric": "rmse",
                                              "teams": 600, "gold_rank_cutoff": 60})
        res = tb.call(ctx_b, "set_phase", {"phase": "REPRO"})
        check("set_phase：登记后可进 REPRO", res.get("ok") and res.get("phase") == "REPRO",
              json.dumps(res, ensure_ascii=False)[:160])
        res = tb.call(ctx_b, "set_phase", {"phase": "IMPROVE"})
        check("set_phase：无门禁不能进 IMPROVE", not res.get("ok", True),
              json.dumps(res, ensure_ascii=False)[:160])

        # ---------- 9. 等待工具 ----------
        lp = ctx_b.resolve("logs/wait_test.log")
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("fold0 ok\ncv done\n", encoding="utf-8")
        res = tb.call(ctx_b, "wait_log", {"path": "logs/wait_test.log", "pattern": "done", "timeout": 30})
        check("wait_log：命中模式立即返回", res.get("matched") is True, json.dumps(res, ensure_ascii=False)[:160])
        res = tb.call(ctx_b, "wait_log", {"path": "logs/missing.log", "timeout": 5})
        check("wait_log：文件不存在要报错", not res.get("ok", True), json.dumps(res, ensure_ascii=False)[:120])
        res = tb.call(ctx_b, "wait_file", {"path": "logs/wait_test.log", "timeout": 20})
        check("wait_file：文件存在且稳定即返回", res.get("exists") is True and res.get("bytes", 0) > 0,
              json.dumps(res, ensure_ascii=False)[:160])

        # ---------- 10. 工具沙箱 ----------
        res = tb.call(ctx, "read_file", {"path": "../../etc/passwd"})
        check("fs 工具：拒绝目录逃逸", not res.get("ok", True),
              json.dumps(res, ensure_ascii=False)[:150])
        res = tb.call(ctx, "sh", {"cmd": "git push --force"})
        check("shell 工具：拦截危险命令", not res.get("ok", True),
              json.dumps(res, ensure_ascii=False)[:150])
    finally:
        store.close()
        shutil.rmtree(base, ignore_errors=True)

    bad = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 60)
    print(f"共 {len(RESULTS)} 项，通过 {len(RESULTS) - len(bad)}，失败 {len(bad)}")
    if bad:
        for name, _, detail in bad:
            print(f"  ✗ {name}: {detail[:300]}")
        return 1
    print("全部门禁自检通过（含反证：每道门都能拦住对应的违规输入）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
