# kagold — 自主 Kaggle agent

一个**能自己跑完一场 Kaggle 比赛**的 agent：选赛 → 复现基线 → 单变量改进 → 集成 → 提交，
每一步都被**代码门禁**约束（不是靠模型自觉）。

它解决的是同一个问题：Kaggle 拿牌失败通常不是「模型不够深」，而是四个执行漏洞——
度量写错、验证口径与榜不一致、没有基线就调参、多变量一起改导致无法归因。
kagold 把这四件事变成四道**硬门**，任何一道不过，提交入口直接拒绝。

```
你的目标：进奖牌区（1000 队以上 ≈ 前 10 名）
kagold 的做法：先保证每一步都「可信」，再谈「更快」
```

---

## 快速开始

```bash
cd kagold

./bin/kagold init                     # 生成 kagold.yaml
./bin/kagold scan                     # 列候选竞赛（带金牌名次门槛）
./bin/kagold scan --sort numberOfTeams --max-teams 800   # 找小规模套利机会

./bin/kagold run <comp-slug>          # 全自动跑一场（默认 ≤140 步 / 8 小时 / 6 次提交）
./bin/kagold run <slug> --max-steps 40 --max-hours 2      # 控制预算
./bin/kagold run --resume 3 --max-steps 20 --max-hours 1  # 续跑：步数/时长都是「本次再加多少」

./bin/kagold status                   # 所有 run 一览
./bin/kagold status 3                 # 单个 run 详情 + 最近工具调用
./bin/kagold report 3                 # 生成/打印运行报告
./bin/kagold selftest                 # 门禁自检（36 项，含反证）
```

## 实测记录（2026-10-26，真实提交）

| 项 | 值 |
|---|---|
| 竞赛 | `playground-series-s6e10`（299,844 测试行，ROC AUC，3,179 队） |
| agent 产出 | LightGBM 5 折，OOF AUC **0.958923**（std 0.00061） |
| 真实提交 | ref `57687756`，public LB **0.95840**，status COMPLETE |
| CV↔LB 差 | 0.0005（口径一致，说明 fold/metric/提交格式都对） |
| 名次 | **283 / 3,179**（前 8.9%，铜牌线内侧；金牌线 rank 16 = 0.96214） |
| 差距 | 距金牌 **0.0037 AUC**（约 0.39% 相对提升） |
| 步数/耗时 | 40 步 / 1.04h / tokens in 747k, out 20k |

**每周自动出候选清单**（已装 cron：每周一 09:17）：

```bash
./bin/kagold shortlist            # 扫 106 场 → 按「金=前%」排序，写 reports/shortlist-<date>.md
```

`shortlist` 会过滤掉已截止/太远的赛题，按金牌赔率排序。2026-10-26 实测：小型赛题
（`nfl-big-data-bowl-2027` 416 队 → 金牌前 41 名、`the-pokemon-company` 540 队 → 前 54 名）
的金牌门槛比 3000+ 队的大赛（≈前 0.5%）宽 20 倍。

`bin/kagold` 直接跑源码，不需要 pip 安装。

---

## 四道硬门禁

| 门 | 检查什么 | 不过会怎样 |
|---|---|---|
| `gate_metric` | 你写的 `src/metric.py: score(y_true, y_pred)` 是否与**独立参考实现**在 5 组合成数据上逐例一致（相对误差 ≤1e-6），方向是否正确 | 不能进改进/提交阶段 |
| `gate_folds` | fold 列存在且无空值、3≤k≤10、最小折样本量、目标分布均衡、同一 id/group 是否跨折泄漏 | 同上 |
| `gate_oof` | OOF 覆盖每一行、无 NaN/inf、按 fold 用**参考实现**复算每折分与 mean±std、多模型相关性 | 同上 |
| `gate_submit` | 前三门通过 + 列/行/id 与 `sample_submission` 一致 + 无量纲异常 + 配额 | 上传被拒 |

关键设计：**`submit` 工具在真正上传前会重新查一遍 SQLite 台账**，并核对提交文件的 sha256
与门禁记录是否一致。模型无法通过「嘴上说自己过了」来绕过——它连伪造状态的机会都没有。

`selftest`（32 项）对每道门都配了**反证用例**（把 MSE 当 RMSE、fold 缺失、id 跨折、NaN、列名不符…），
证明门真的能拦住，而不是摆设。

---

## 工作目录结构

每次 run 在 `runs/<comp>/` 下：

```
runs/<comp>/
├── input/          比赛数据（download_data 落这里）
├── src/            agent 写的代码：metric.py、基线脚本、特征工程
├── oof/            oof.csv、blend.csv、gate_*.json、score.json
├── sub/            submission.csv、gate_submit.json
├── logs/           notes.md（agent 的工作笔记）、_snippet_*.py
└── report.md       运行结束时自动生成
```

台账在 `runs/kagold.db`（SQLite）：runs / steps / tool_calls / gates / submissions / kv。
**断点续跑靠它**，不靠内存。

---

## 配置（`kagold.yaml`）

```yaml
llm:
  provider: huo                     # 复用 ~/.dsh 的 provider 与密钥
  model: deepseek-v4.1-flash
max_steps: 140                      # 模型轮数上限
max_hours: 8                        # 墙钟上限
max_submissions: 6                  # 真实提交次数上限
max_tokens_in: 4000000
max_tokens_out: 500000
max_shell_seconds: 1800             # 单条命令超时
min_free_gb: 3                      # 磁盘守卫
temperature: 0.25
allow_accept_rules: false           # 自动接受竞赛规则（法律动作，默认关）
allow_public_notebooks: false
```

命令行参数优先于 yaml。provider 凭据按 `~/.dsh/.env` → 环境变量 → `.credentials.yaml`
的顺序解析，也可以用 `KAGOLD_PROVIDER/KAGOLD_MODEL/KAGOLD_BASE_URL/KAGOLD_API_KEY` 覆盖。

---

## 安全护栏

- **shell 黑名单**：`rm -rf /`、`mkfs`、`dd of=/dev/`、关机、fork 炸弹、`git push`、
  发布 dataset、`sudo`、`curl|sh`、碰 `.ssh` —— 命中即拒绝并记账。
- **路径沙箱**：所有文件工具强制落在 `runs/<comp>/` 内，`../../etc/passwd` 这类逃逸直接报错。
- **磁盘守卫**：剩余空间低于 `min_free_gb` 时停止下载并终止运行（本机磁盘长期紧张）。
- **不发布**：不 push、不发布 notebook/dataset、不自动接受规则。
- 只读 `~/.kaggle/kaggle.json`，不写不改。

---

## 为什么这是「拿金牌」的路径

奖牌看**最终榜绝对名次**。kagold 不自称能保证奖牌，但它把「拿牌的必要条件」做成了默认流程：

1. **度量可信**：metric 门用独立实现交叉验证，杜绝「CV 0.94 / LB 0.88」这种事故。
2. **口径一致**：fold 只划分一次并写回文件，OOF 与提交用同一套划分。
3. **有基线再优化**：先复现一个能提交的基线拿到 CV↔LB 锚点，再谈改进。
4. **可归因**：一次只改一个变量，每次改动都有 OOF delta 记账。
5. **集成为王**：权重在 OOF 上搜，绝不用 LB 反馈调权重。

剩下的靠选题（队数少的比赛金=前 10%）和算力（本机无 GPU，重训练走 Kaggle Notebook）。

---

## 已知限制

- **本机无 GPU**：深度学习类竞赛只能靠 CPU 或 Kaggle Notebook 算力，训练规模受限。
- 参考度量只覆盖常见指标（rmse/mse/mae/rmsle/mape/smape/r2/logloss/auc/accuracy/f1/mcc/qwk）；
  竞赛自定义指标需要先在 `src/kagold/gates/reference.py` 里人工确认并加实现——这是**故意的**，
  宁可停下来让人确认，也不让 agent 自己发明一个"差不多"的度量。
- 代码类竞赛（Kernels-only）的文件提交会被 Kaggle 拒绝，需要走 notebook 版本提交（agent 会收到提示）。
- 单场 run 默认 140 步，深度有限；真正冲榜需要多轮 resume + 人工定方向。

---

## 目录

```
kagold/
├── bin/kagold                 启动器
├── src/kagold/
│   ├── config.py              provider/预算/路径/磁盘守卫
│   ├── store.py               SQLite 台账
│   ├── runtime.py             run 上下文（路径沙箱 + Kaggle 客户端）
│   ├── llm.py                 OpenAI 兼容客户端（重试/token 记账）
│   ├── agent/{loop,prompts}.py 主循环 + 方法论提示词
│   ├── gates/{reference,metric,folds,oof,submit}.py   四道门 + 参考实现
│   ├── tools/{base,fs,shell,kaggle,gates,ensemble}.py  工具集
│   └── cli.py                 命令行
└── tests/selftest.py          门禁自检（含反证）
```
