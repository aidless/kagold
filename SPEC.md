# SPEC — kagold 自主 Kaggle agent

本文件是**实现契约**：改代码前先读这里，改完必须让 `bin/kagold selftest` 全绿。

---

## 1. 目标与非目标

**目标**：给一场 Kaggle 竞赛，agent 能自主走完
`SELECT → REPRO → IMPROVE → ENSEMBLE → SUBMIT`，
产出一个**可复现、可信、且真的提交过**的结果。

**非目标**：
- 不追求「自动拿金牌」。奖牌取决于选题、算力与对手，agent 只保证过程不出低级错误。
- 不做多机调度/分布式训练。本机无 GPU，重训练走 Kaggle Notebook。
- 不做自动化推理服务。

---

## 2. 核心不变量（invariants）

这些是硬约束，任何改动不得破坏：

| # | 不变量 | 实现位置 | 违反后果 |
|---|---|---|---|
| I1 | 没有通过 `metric` 门，不能进入 `IMPROVE`/上传 | `tools/gates.py:set_phase`、`submit` | 度量错 = 后面全部白做 |
| I2 | 没有通过 `folds` 门，不能进 `IMPROVE` | 同上 | 口径不一致 = CV 自欺 |
| I3 | 没有 OOF，模型不许进集成 | `tools/ensemble.py`（要求 pred 列） | 权重无法无偏估计 |
| I4 | 上传前必须四门齐备 + 文件 sha256 与门禁记录一致 | `tools/gates.py:submit` | 防「过门禁后换文件」 |
| I5 | 分数只来自工具输出（参考实现复算），不接受模型自报 | `gates/*` | 防幻觉数字 |
| I6 | fold 列只划分一次并落盘，后续全部引用同一列 | `gates/folds.py` + prompt 铁律 4 | 防选择性划分 |
| I7 | 预算（步数/token/时长/提交数/磁盘）是硬停止 | `agent/loop.py:_budget_stop` | 防失控烧钱 |
| I8 | agent 写不出 `runs/<comp>/` 之外、跑不了黑名单命令 | `runtime.resolve`、`tools/shell.py` | 防越界 |
| I9 | 上下文裁剪只能裁「完整交互组」 | `agent/loop.py:_group/_trim` | 拆散 tool_calls 会导致 API 报错 |
| I10 | 每条消息、每次工具调用都落 SQLite | `store.py` | 断点续跑与事后审计的前提 |

---

## 3. 模块与职责

```
config.py     provider 解析（复用 ~/.dsh 凭据）、预算、路径、磁盘守卫、host_info
store.py      SQLite(WAL) 台账：runs/steps/tool_calls/gates/submissions/kv/events
runtime.py    RunContext：路径沙箱 resolve()、目录属性、懒加载 KaggleApi、gate 查询
llm.py        OpenAI 兼容客户端：重试退避、ContextOverflow 识别、token 记账
agent/
  prompts.py  SYSTEM（方法论+铁律+六阶段）、task_message、state_card
  loop.py     主循环、工具调度、预算停止、上下文裁剪、断点续跑、report.md
gates/
  reference.py  独立参考度量 + 别名归一 + 合成数据生成（**不采信 agent 实现**）
  metric.py     门1：agent 的 score() vs 参考实现 ×5 用例 ×方向
  folds.py      门2：fold 列/折数/样本量/分布均衡/跨折泄漏
  oof.py        门3：覆盖/NaN/每折分/相关性/写 baseline_score 与 oof_best
  submit.py     门4：前置门禁 + 产物格式 + 量纲 + 配额 + sha256
tools/
  base.py     Tool/ToolBox/装饰器/schema
  fs.py       read_file/write_file/append_file/list_dir/grep/delete_file（沙箱）
  shell.py    sh/python（黑名单、超时、cwd=run root、PYTHONPATH 安全化）
  kaggle.py   competitions_list/competition_info/download_data/leaderboard/my_submissions
  gates.py    四门工具封装 + submit（唯一上传入口）+ record_competition/set_phase/note/finish/run_state
  ensemble.py OOF 加权融合（greedy/mean/rank_mean）
cli.py        scan/run/status/report/init/selftest
tests/selftest.py  25 项自检，每道门都有反证
```

---

## 4. 运行时协议

### 4.1 一轮循环

```
budget_check → LLM.chat(messages, tool_specs)
  ├─ 有 tool_calls → 逐个 ToolBox.call → 结果作为 role=tool 回灌 → 落库
  │                  （任一结果含 finished=true → 结束）
  └─ 无 tool_calls → nudge（空转计数，≥4 判定 stalled）
```

### 4.2 消息落库

`steps(run_id, idx, role, payload, tokens_in, tokens_out)`，`payload` 是完整 JSON。
续跑时 `SELECT payload ... ORDER BY idx` 重建，末尾不完整的 tool_calls 组会被丢弃（`_group`）。

### 4.3 上下文裁剪

超过 900k 字符触发；保留 system + 首条 task + 最近 8 组完整交互，其余折叠成一条说明。
估算用 JSON 字符数（≈4 字符/token）。

### 4.4 阶段自动推进

模型经常忘记调 `set_phase`，所以状态机由"做完事"驱动（`tools/gates.py:_auto_advance`，只前进一格）：

| 触发 | 结果阶段 |
|---|---|
| `record_competition` 成功 | SELECT → REPRO |
| `gate_oof` 通过 | REPRO → IMPROVE |
| `ensemble` 成功 | IMPROVE → ENSEMBLE |
| `submit` 成功 | ENSEMBLE → SUBMIT |

`set_phase` 仍可手动推一格，且带前置条件校验（缺门禁/缺登记直接拒绝）。跳阶段永远不允许。

### 4.5 停止条件

步数 ≥ max_steps / tokens_in ≥ 上限 / tokens_out ≥ 上限 / 墙钟 ≥ max_hours /
磁盘 < min_free_gb / agent 调 `finish` / 连续 4 轮无工具调用 / LLMError / KeyboardInterrupt。

结束时写 `report.md` 并把 status 落库（finished|paused|stalled|error）。

---

## 5. 门禁细节与扩展方式

### 5.1 加一个新 metric

1. 在 `gates/reference.py` 的 `REGISTRY` 加实现（`fn / greater_is_better / kind`），
   `kind` 决定用哪种合成数据（regression/binary/ordinal）。
2. 在 `ALIASES` 加竞赛里的官方写法（大小写/空格/缩写）。
3. 在 `tests/selftest.py` 加一条「该指标的参考实现 vs 手算」用例。

**这一步必须人工确认**：宁可停下来让人核对官方定义，也不让 agent 自己发明近似指标。

### 5.2 加一个新工具

```python
@tool("name", "描述（模型据此决定何时调用）", obj({"arg": {"type": "string"}}, ["arg"]))
def name(ctx: RunContext, a: dict) -> dict:
    ...
    return {"ok": True, "data": ...}
```

- 放 `tools/<module>.py`，`tools/__init__.py` 的 `build_toolbox()` 会自动收集。
- 抛 `ToolError` 表示业务性拒绝（会被包装成 `{"ok": False, "error": ...}`）。
- 需要终止 run 的返回 `{"ok": True, "finished": True, ...}`。

### 5.3 加一道门

1. `gates/<name>.py` 写 `run_<name>_gate(ctx, ...) -> {"passed", "reason", "evidence"}`，
   内部必须 `ctx.store.record_gate(...)` 且把证据落到 `oof/gate_<name>.json`。
2. 在 `gates/__init__.py`、`tools/gates.py` 暴露工具，`submit`/`set_phase` 里加前置检查。
3. `tests/selftest.py` 补**正例 + 反例**（门必须能失败）。

---

## 6. 性能与成本预算

| 项 | 实测 | 备注 |
|---|---|---|
| 系统提示 + 26 个工具 schema | ≈3.1k input tokens/轮 | huo provider 实测 |
| 一轮 2~3 次工具调用 | 30~90s | 取决于工具 |
| 30 步 smoke run | ≈15 分钟、<10k output tokens | playground 数据 |

成本优化手段（已实现）：工具结果截断到 6000 字符、大文件禁止 `read_file`、
上下文按组折叠、`max_steps` 硬停。

---

## 7. 故障与恢复

| 症状 | 处理 |
|---|---|
| `ContextOverflow` | 强制裁剪 + 单次重试，仍失败则 run 停止 |
| LLM 429/5xx/timeout | llm.py 指数退避（≤4 次），仍失败则停止 |
| 进程被杀 | `kagold run --resume <id>`，从 SQLite 恢复 |
| 上下文里出现「重复同一调用」 | loop 记录连续签名，由 nudge 纠正 |
| 提交被 Kaggle 400 拒 | 大概率 Kernels-only，提示走 notebook 版本提交 |
| 磁盘涨满 | `_budget_stop` 直接停；`download_data` 前置 `disk_guard` |

---

## 8. 已知取舍

- **不用 agent 框架**：循环/调度/预算全自研，因为门禁必须能真的挡住提交——框架的工具层不提供这种硬约束。
- **参考度量不自动扩展**：自动推断官方 metric 实现容易错得离谱，宁可 fail-closed。
- **greedy 集成而非 stacking**：OOF 少时 stacking 极易过拟合；权重搜索更稳。
- **一次只改一个变量**：牺牲探索速度换可归因性；在预算有限时这是更划算的取舍。
