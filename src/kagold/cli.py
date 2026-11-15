"""命令行入口：kagold {scan,run,status,report,init,selftest}"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from . import __version__, config
from .agent import Agent
from .config import ensure_run_dirs, free_gb, resolve_provider, run_root, settings, slug_of
from .llm import LLM
from .runtime import RunContext
from .store import Store
from .tools import build_toolbox

DB_PATH = config.RUNS_DIR / "kagold.db"

BANNER = f"""kagold {__version__} — 自主 Kaggle agent
"""


# ---------------------------------------------------------------- helpers
def _store() -> Store:
    return Store(DB_PATH)


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _dummy_ctx(api=None):
    api = api or _api()
    return SimpleNamespace(kaggle=lambda: api)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}GB"


# ---------------------------------------------------------------- commands
def cmd_scan(args) -> int:
    from .tools.kaggle import competitions_list
    res = competitions_list.fn(_dummy_ctx(), {
        "sort_by": args.sort, "search": args.query, "page": args.page, "limit": args.limit})
    comps = res.get("competitions") or []
    if not comps:
        print("没有找到竞赛。")
        return 1
    print(f"{'竞赛':38} {'队数':>6} {'金牌':>5} {'截止':12} {'已进':>4} {'积分':>4}  指标")
    print("-" * 110)
    for c in comps:
        teams = c["teams"]
        if args.min_teams and teams < args.min_teams:
            continue
        if args.max_teams and teams > args.max_teams:
            continue
        diff = c["gold_rank_cutoff"] / max(teams, 1) * 100
        print(f"{c['ref'][:38]:38} {teams:>6} {c['gold_rank_cutoff']:>5} "
              f"{(c['deadline'] or '')[:10]:12} {'是' if c['entered'] else '-':>4} "
              f"{'1' if c['awards_points'] else '-':>4}  {(c['metric'] or '')[:40]}"
              f"  [金=前{diff:.1f}%]")
    print("\n提示：金牌≈前 10 名（大竞赛略增）。队数少 + 剩余时间长的比赛更容易进入奖牌区。")
    return 0



def cmd_shortlist(args) -> int:
    """每周扫一遍公开竞赛，按「金牌赔率」排序出候选清单。"""
    import datetime as _dt
    from .tools.kaggle import competitions_list

    dummy = _dummy_ctx()
    seen: dict[str, dict] = {}
    for sort_by in ("recentlyCreated", "numberOfTeams", "earliestDeadline"):
        for page in (1, 2):
            try:
                res = competitions_list.fn(dummy, {"sort_by": sort_by, "page": page, "limit": 60})
            except Exception as e:  # noqa: BLE001
                print(f"  扫描 {sort_by} p{page} 失败：{type(e).__name__}: {e}", file=sys.stderr)
                continue
            for c in res.get("competitions") or []:
                seen.setdefault(c["ref"], c)

    now = _dt.datetime.now(_dt.timezone.utc)
    rows = []
    for c in seen.values():
        try:
            dt = _dt.datetime.fromisoformat((c.get("deadline") or "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            continue
        days = (dt - now).total_seconds() / 86400
        if days < args.min_days or days > args.max_days:
            continue
        teams = int(c.get("teams") or 0)
        if teams < args.min_teams:
            continue
        gold = int(c.get("gold_rank_cutoff") or 0)
        rows.append({**c, "days_left": round(days, 1),
                     "gold_pct": round(100.0 * gold / max(teams, 1), 2)})

    rows.sort(key=lambda r: (0 if r.get("awards_points") else 1, -r["gold_pct"], -r["days_left"]))
    top = rows[: args.top_n]
    stamp = _dt.datetime.now().strftime("%Y-%m-%d")
    out_dir = config.PROJECT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shortlist-{stamp}.md"

    lines = [f"# 候选竞赛清单 · {stamp}", "",
             f"扫描 {len(seen)} 场，符合条件 {len(rows)} 场（剩余 {args.min_days:g}–{args.max_days:g} 天、"
             f"队数 ≥{args.min_teams}）。按「金牌赔率」排序：", "",
             "| 竞赛 | 队数 | 金牌名次 | 金=前% | 剩余天 | 指标 | 已进 | 积分 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in top:
        lines.append(f"| [{r['ref']}](https://www.kaggle.com/competitions/{r['ref']}) | {r['teams']} | "
                     f"{r['gold_rank_cutoff']} | {r['gold_pct']}% | {r['days_left']} | "
                     f"{(r['metric'] or '—')[:38]} | {'是' if r['entered'] else ''} | "
                     f"{'1' if r['awards_points'] else ''} |")
    lines += ["", "## 判读方式", "",
              "- **金=前%**：这一列越高，进金牌区所需的相对名次越宽松（小规模竞赛的优势所在）。",
              "- 大竞赛（3000+ 队）金牌≈前 0.5%，除非有明确方法优势，不要先打。",
              "- `已进=是` 的场次优先——已有提交可以对照 CV/LB 差距，改进增量更容易验证。",
              "- 选定后：`kagold run <slug>`；想先摸底可 `kagold run <slug> --max-steps 40`。"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"扫描 {len(seen)} 场 → 候选 {len(rows)} → 报告 {out}")
    print(f"{'竞赛':40} {'队数':>6} {'金牌':>5} {'金=前':>7} {'剩余':>6}  指标")
    for r in top[: args.show]:
        print(f"{r['ref'][:40]:40} {r['teams']:>6} {r['gold_rank_cutoff']:>5} "
              f"{r['gold_pct']:>6}% {r['days_left']:>6}  {(r['metric'] or '')[:34]}")
    return 0


def cmd_run(args) -> int:
    st = _store()
    if args.resume:
        run = st.run(int(args.resume))
        if not run:
            print(f"找不到 run {args.resume}", file=sys.stderr)
            return 2
        run_id, comp = int(args.resume), run["comp"]
        if run.get("status") == "finished":
            print(f"run {run_id} 已结束（finished），如需继续请用 --new 开新 run。")
            return 0
        # 续跑必须先把台账状态改回 running，否则 _budget_stop 一进来就停
        st.update_run(run_id, status="running")
        print(f"续跑 run {run_id} · {comp}（状态 paused/error → running）")
    elif args.dry_run:
        prov = resolve_provider(args.provider, args.model)
        comp = slug_of(args.comp)
        print(f"[dry-run] 会创建 run：comp={comp} goal={args.goal or f'冲刺 {comp} 奖牌'}")
        print(f"[dry-run] provider={prov.name} model={prov.model} base={prov.base_url} key={prov.source}")
        print(f"[dry-run] 目录 {run_root(comp)}  磁盘剩余 {free_gb():.1f}GB")
        print(f"[dry-run] max_steps={args.max_steps or settings().max_steps} "
              f"max_hours={args.max_hours or settings().max_hours} "
              f"max_submissions={args.max_submissions or settings().max_submissions}")
        return 0
    else:
        comp = slug_of(args.comp)
        root0 = ensure_run_dirs(comp)
        run_id = st.new_run(comp, goal=args.goal or f"冲刺 {comp} 奖牌",
                            config={"provider": args.provider or "", "model": args.model or ""})
        print(f"新 run {run_id} · {comp} · 目录 {root0}")

    run = st.run(run_id) or {}
    comp = run.get("comp") or comp
    root = ensure_run_dirs(comp)
    s = settings(
        max_steps=args.max_steps or None,
        max_hours=args.max_hours or None,
        max_submissions=args.max_submissions or None,
        llm_provider=args.provider,
        llm_model=args.model,
    )
    ctx = RunContext(run_id=run_id, comp=comp, root=root, settings=s, store=st)

    prov = resolve_provider(s.llm_provider, s.llm_model)
    if not prov.api_key:
        print(f"provider {prov.name} 没有可用密钥", file=sys.stderr)
        return 2
    print(f"LLM: {prov.name}/{prov.model}  磁盘剩余 {free_gb():.1f}GB")
    llm = LLM(base_url=prov.base_url, api_key=prov.api_key, model=prov.model,
              temperature=s.temperature)
    tb = build_toolbox()
    agent = Agent(ctx, llm, tb, settings=s, max_steps=args.max_steps)
    summary = agent.run()
    print("\n=== 运行结束 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in ("finished", "paused") else 1


def cmd_status(args) -> int:
    st = _store()
    if args.run_id:
        run = st.run(int(args.run_id))
        if not run:
            print(f"找不到 run {args.run_id}", file=sys.stderr)
            return 2
        gates = st.gate_state(int(args.run_id))
        print(json.dumps({**run, "gates": gates}, ensure_ascii=False, indent=2)[:6000])
        print("\n最近工具调用：")
        for t in st.tool_calls(int(args.run_id), limit=15)[::-1]:
            flag = "ok" if t["ok"] else "FAIL"
            print(f"  [{t['step_idx']:>3}] {t['name']:<16} {flag:<4} {t['ms']:>6}ms "
                  f"{json.dumps(t['args'], ensure_ascii=False)[:90]}")
        return 0
    rows = st.runs(limit=args.limit)
    if not rows:
        print("还没有 run。先 `kagold scan` 找比赛，再 `kagold run <slug>`。")
        return 0
    print(f"{'id':>4} {'阶段':9} {'状态':10} {'步数':>4} {'提交':>4} {'创建':18}  竞赛")
    for r in rows:
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"{r['id']:>4} {r['phase']:9} {r['status']:10} {r['steps']:>4} "
              f"{r['submissions']:>4} {ts}  {r['comp']}")
    return 0


def cmd_report(args) -> int:
    st = _store()
    run = st.run(int(args.run_id))
    if not run:
        print(f"找不到 run {args.run_id}", file=sys.stderr)
        return 2
    from .agent.loop import render_report
    from .runtime import RunContext
    ctx = RunContext(run_id=int(args.run_id), comp=run["comp"],
                     root=run_root(run["comp"]), settings=settings(), store=st)
    path = render_report(ctx, run)
    print(Path(path).read_text(encoding="utf-8"))
    return 0


def cmd_init(args) -> int:
    root = config.PROJECT
    root.mkdir(parents=True, exist_ok=True)
    yml = root / "kagold.yaml"
    if yml.exists() and not args.force:
        print(f"{yml} 已存在（--force 覆盖）")
        return 0
    yml.write_text(TEMPLATE_YAML, encoding="utf-8")
    (root / "runs").mkdir(exist_ok=True)
    print(f"写入 {yml}；数据/台账在 {root / 'runs'}")
    return 0


def cmd_selftest(args) -> int:
    script = config.PROJECT / "tests" / "selftest.py"
    if not script.exists():
        print(f"缺少 {script}", file=sys.stderr)
        return 2
    env = {"PYTHONPATH": str(config.SRC_DIR)}
    import os
    e = {**os.environ, **env}
    p = subprocess.run([sys.executable, str(script)], env=e, cwd=str(config.PROJECT))
    return p.returncode


TEMPLATE_YAML = """\
# kagold 配置（命令行参数优先于这里的值）
llm:
  provider: huo          # huo / commandcode / open / open2 ...
  model: deepseek-v4.1-flash

max_steps: 140           # 单次 run 的最大模型轮数
max_hours: 8             # 单次 run 的墙钟上限
max_submissions: 6       # 单次 run 允许的真实提交次数
max_tokens_in: 4000000
max_tokens_out: 500000
max_shell_seconds: 1800  # 单条 shell 命令超时
min_free_gb: 3           # 磁盘守卫：低于此值停止下载/停止运行
temperature: 0.25
allow_accept_rules: false   # 是否允许 agent 自动接受竞赛规则
allow_public_notebooks: false
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kagold", description=BANNER)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="列出候选竞赛并估算金牌门槛")
    s.add_argument("--sort", default="recentlyCreated",
                   choices=["recentlyCreated", "earliestDeadline", "latestDeadline",
                            "numberOfTeams", "prize", "best", "grouped", "relevance"])
    s.add_argument("--query", default=None, help="关键词")
    s.add_argument("--page", type=int, default=1)
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--min-teams", type=int, default=0)
    s.add_argument("--max-teams", type=int, default=0)
    s.set_defaults(fn=cmd_scan)

    sl = sub.add_parser("shortlist", help="扫公开竞赛，按金牌赔率出候选清单（可配 cron 每周跑）")
    sl.add_argument("--top-n", type=int, default=25)
    sl.add_argument("--show", type=int, default=15)
    sl.add_argument("--min-days", type=float, default=3)
    sl.add_argument("--max-days", type=float, default=200)
    sl.add_argument("--min-teams", type=int, default=50)
    sl.set_defaults(fn=cmd_shortlist)

    r = sub.add_parser("run", help="跑一场竞赛（自主循环）")
    r.add_argument("comp", nargs="?", help="竞赛 slug 或 URL")
    r.add_argument("--resume", type=int, help="续跑已有 run id")
    r.add_argument("--goal", default=None)
    r.add_argument("--max-steps", type=int, default=0)
    r.add_argument("--max-hours", type=float, default=0)
    r.add_argument("--max-submissions", type=int, default=0)
    r.add_argument("--provider", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--dry-run", action="store_true", help="只检查配置与凭据")
    r.set_defaults(fn=cmd_run)

    st = sub.add_parser("status", help="查看 run 列表 / 单个 run 详情")
    st.add_argument("run_id", nargs="?", type=int)
    st.add_argument("--limit", type=int, default=15)
    st.set_defaults(fn=cmd_status)

    rp = sub.add_parser("report", help="打印某次 run 的报告")
    rp.add_argument("run_id", type=int)
    rp.set_defaults(fn=cmd_report)

    i = sub.add_parser("init", help="生成 kagold.yaml")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)

    t = sub.add_parser("selftest", help="跑门禁自检（含反证）")
    t.set_defaults(fn=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except KeyboardInterrupt:
        print("\n中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
