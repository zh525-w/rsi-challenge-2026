# 校赛工程：A/B/C 三赛道（CPU 续训，Harness SDK 1.0.2）

本目录把三个官方训练包原样接进来，共用一套适配层与模型代码：
`cpu-a-guide-20261110-v1`（气体识别，分类）、`cpu-b-rul-20261110-v1`（XJTU-SY 轴承 RUL，回归）、
`cpu-c-guide-20261110-v1`（C-MAPSS FD001 RUL，回归）。
用途对应校赛提交要求里的三项：可复现最小原型、项目源代码、系统运行说明。

## 〇之前、三赛道实测（2026-11-17，本地协议替身，未消耗平台试运行次数）

| 赛道 | 任务 | 特征维度 | 平台基线 | 本工程 10 轮后 | 变化 |
| --- | --- | --- | --- | --- | --- |
| A | 6 类气体分类 | 128 | Macro-F1 0.714030 | Macro-F1 0.724650（第 3 轮） | +0.0106 |
| B | 轴承 RUL（分钟） | 34 | RMSE 567.15（官方公布）/ 681.16（镜像内检查点实测） | **RMSE 528.91** | −7.0% / −22.3% |
| C | FD001 RUL（周期） | 24 | RMSE 53.31 / 镜像内 52.87 | **RMSE 51.85** | −2.7% / −1.9% |

口径说明：

- 打分函数与官方 `baselines/train.py` 完全一致——把官方 B 检查点喂进本工程的抽取与打分，
  得到 `RMSE 567.1475036182951`，与该包 `history.jsonl` 记录逐位相同，说明特征路径与指标口径没有偏差。
- 回归的父权重在 B 上已经过拟合（在同一验证集上检查点为 681，而它训练日志是 567），
  因此回归轮做“双路探针”：继承父权重续训与从零初始化各训练一版，按验证集择优提交；
  C 上继承更优、B 上从零更优，两条路径都由验证分决定，不写死。
- 每轮训练都在公开 `train.jsonl` 上做，评估只用公开 `validation.jsonl`，不碰任何隐藏标签；
  连续回滚时自动衰减学习率，另外叠加逐轮步长阶梯，避免同一配方重复空转。

## 〇、平台入口（Harness SDK 1.0.4 契约）

平台用 `contest.harness.workspace.v2` 协议驱动镜像，入口固定为 `/app/harness.py`（SDK 官方模板），
真正的选手代码在 `/app/contest_submission.py`，只对外暴露两个函数：

```text
python3 /app/harness.py describe     # 一行 JSON：protocol / model_access / tasks
python3 /app/harness.py train        # 平台从 stdin 写一帧 start，SDK 解帧后调用 train(rnd)
python3 /app/harness.py predict      # 平台逐行写样本，本工程逐行回预测
```

- `train(rnd)`：读 `/data/*.jsonl` 公开投影与 `/previous` 父检查点，续训一小步，
  **每轮至少一次真实模型调用**（`rnd.model()`，平台持有密钥与端点），
  `rnd.submit_candidate()` 提交候选、`rnd.save_state()` 留状态、`rnd.finish()` 结束本轮。
  平台共跑 10 轮，每轮一个独立进程；每轮只做一次“读历史—做实验—依结果决策”的完整动作。
- `predict(sample, model_dir)`：检查点真实路径取 `CONTEST_ARTIFACT_PATH`，`/model` 是根目录；
  分类返回 `[{"label","score"}]`（score∈[0,1]、标签唯一、最高分唯一），回归返回 `{"value": 非负数}`；
  必须读完一行就写一行并 flush，stdout 只放协议 JSONL，日志一律 stderr（SDK 已做 stdout 隔离）。
- 数据集 A/B/C 用同一套代码：特征按“路径”描述（如 `[["key","features"],["index",0]]`），
  任意嵌套的 `inputs` 都能压平成向量；A 赛道直接沿用官方父检查点的 128 条路径与权重，B/C 按当前
  `inputs` 结构生成路径。任务类型以 `rnd.task["task"]` 为准（classification / regression），
  不拿数据集代号当任务类型。

镜像内的对应位置：

| 路径 | 内容 |
| --- | --- |
| `/app/harness.py` | SDK 入口模板 |
| `/app/contest_submission.py` | `train(rnd)` / `predict(sample, model_dir)` |
| `/app/contest_harness/` | SDK 1.0.2 本体（vendored，import 即可，无需 pip） |
| `/app/rsi/core.py` | 路径化特征抽取 + MLP 分类/回归 |

本地离线自查（不需要真实模型，用 JSONL 宿主替身）：

```sh
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

> 说明：下面第十四节记录的是 SDK 到手**之前**自研适配层的形态（`scripts/run_harness.py` 三模式）。
> 它已在 2026-11-17 被官方 SDK 入口取代，保留仅为过程留痕，平台侧不再使用。

## 一、环境

已验证通过的组合：

- Python 3.10.11（官方锁定 3.10–3.12）
- NumPy 1.26.4
- 纯 CPU，单 BLAS 线程（`BASELINE_CPU_THREADS=1`）
- 不需要 GPU，也不需要 PyTRiO

安装：

```sh
python -m pip install -r requirements.txt
```

## 二、跑一轮

```sh
python scripts/run_round.py --round 1
python scripts/run_round.py --round 2
python scripts/run_round.py --round 3 --lr 0.004 --epochs 6
```

每次运行会依次做四件事：

1. 从 `configs/rounds.json` 读出本轮的假设、改动和失败判据并打印；
2. 自动定位父检查点（第 1 轮用 `checkpoint.npz`，之后用 `runs/round-{N-1}/checkpoint.npz`）；
3. 调用官方训练器写入 `runs/round-{N}/`，并核对 `training.json` 里的父检查点摘要；
4. 把本轮指标追加到 `logs/experiments.csv`。

跑之前想先看命令，加 `--dry-run`。

看全部轮次：

```sh
python scripts/summarize.py
```

## 三、官方单轮命令（不经封装时）

```sh
python -m baselines.train --profile A --train public/train.jsonl --validation public/validation.jsonl --from-baseline checkpoint.npz --epochs 5 --output runs/round-1 --stage-id round-1
```

预测协议自检：

```sh
python -m baselines.predict --checkpoint runs/round-1/checkpoint.npz
```

## 四、轮次编排（规划十轮，实际执行八轮）

`configs/rounds.json` 预置了十轮规划。**本地台账的 8 轮是按 `max_rounds=8` 跑的；提交镜像内已改为 `max_rounds=10`，与规则「每个数据集 10 轮」一致**；已执行的 8 轮改动与指标见第九节与 `logs/experiments.csv`，本节的第 5–10 轮是当初的规划设想，其中部分改动（类别权重、label smoothing、温度校准等）并未落地。
每轮只允许一个变量，`change` 字段就是本轮的改动声明。

| 轮次 | 改动 | 目的 |
|---|---|---|
| 1 | 无 | 确认链路与基线数字 |
| 2 | 学习率 0.003 → 0.0015 | 后半程步长是否偏大 |
| 3 | 学习率 0.003 → 0.006 | 摸学习率上界 |
| 4 | 轮内步数翻倍 | 每轮 20 步是否偏少 |
| 5 | 轮内按验证保留最优 | 避免固定步数浪费峰值 epoch |
| 6 | 标签平滑 | 抑制过度自信 |
| 7 | 类别加权 | 少数类拖累 Macro-F1 |
| 8 | 正则加强 | 跨批次漂移下防过拟合 |
| 9 | 温度校准 | 边界样本 |
| 10 | 无 | 复验冻结版本 |

**表中第 6–10 行是规划项，均未执行。**实际第 6–8 轮做的是轮内步数微调（再走 2/3/2 epoch），标签平滑、类别加权、正则加强、温度校准都没有落地——写报告时不要把这几项当作已完成的消融实验。

十轮规划的总预算只有 10×5×4 ≈ 200 步，所以每轮都必须能在 20 步内看出方向。

## 五、三条硬边界（写进报告时要声明）

官方包把下面三件事明确禁掉了，踩了就不是"续训"而是"另起模型"：

1. **不能重估标准化参数。** 输入均值、标准差只在初始训练集拟合，续训沿用检查点状态；验证集和隐藏集不得参与拟合。所以"按目标批次重做归一化"这条路不通，要缓解漂移只能改模型内部，例如加入可学习仿射或加强正则。
2. **不能用验证集打伪标签或做自训练。** 验证集只用于衡量改动，合并进训练集即违反隔离要求。
3. **不能随机混合批次再切分。** 训练与验证保持批次与时间顺序隔离，这也是本题的漂移来源。

另外，改网络结构不是不能做，但必须显式实现权重与优化器状态的迁移并保留来源证据，不能把新模型声明成原模型续训。

## 六、目录

```
configs/rounds.json      轮次编排与单轮改动声明（原始规划十轮；configs/harness.json 现取 max_rounds=10，与规则一致；已提交台账为 8 轮）
scripts/run_round.py     轮次调度：定位父检查点、跑训练、写记录
scripts/summarize.py     汇总日志，指出当前最佳
scripts/run_harness.py   Harness 循环入口
scripts/selfcheck.py     环境自检（版本、检查点、官方测试）
harness/decide.py        决策层：preset / llm 两种实现
harness/monitor.py       漂移监测：特征偏移与类先验偏移
harness/diagnose.py      诊断归因：把漂移归到可处理的几类原因
harness/swan.py          SwanLab 接入：云端实验记录
harness/loop.py          循环主体：五环串联与受控门限
scripts/dataset_profile.py  数据画像：批次数与类别分布
configs/harness.json     Harness 配置：决策器、预算、阈值、单价、预设方案
results/final.json       跑完一轮循环后的最终产物摘要与门槛判定
docker/Dockerfile        镜像定义，构建期执行自检
docker/build.ps1         构建 + 验证 + 导出（Windows）
docker/build.sh          构建 + 验证 + 导出（Linux / macOS）
state.json               当前被采纳的轮次与检查点
logs/experiments.csv     逐轮指标（本轮首次运行后生成）
runs/                    每轮输出：checkpoint.npz / training.json / history.jsonl
baselines/               官方模型、数据、训练、预测代码与配置
public/                  公开训练集与验证集
checkpoint.npz           官方起始检查点
TASK.md                  任务与标签说明（官方）
TRAINING_GUIDE.md        续训指南（官方）
```

## 七、基线事实（来自官方包，勿改）

- 参数量 4,326；128 维特征；32 个 tanh 隐藏单元；6 类 softmax
- 训练集 445 行，验证集 1,244 行（按批次隔离，训练集比验证集小）
- 基线：20 epoch / 80 步，学习率 0.003，批大小 128
- 基线验证指标：Macro-F1 **0.714029**，Accuracy 0.878617
- 基线训练耗时 0.24 秒

`logs/experiments.csv` 的 `macro_f1_after` 必须与 `runs/round-N/training.json` 一致，平台会独立复跑预测，自报数字不能替代平台评测。

## 八、已知风险

- 本包只覆盖 A 赛道。B/C 需要各自的训练包，且官方 README 把 B 描述为三分类（Normal / Degrading / SevereFault），与赛事页面"按分钟回归、RMSE/MAE"不一致，需向组委会确认后再定 B 的路线。
- `baseline-info.json` 里 `label_status` 为 `draft`，正式口径以组委会发布为准。
- 官方测试命令 `python -m unittest discover -s baselines/tests -v` 在当前 Python 3.10 下会因 `baselines/tests` 不是包而报 `Start directory is not importable`；改用下面这条即可，9 项测试全部通过：

```sh
PYTHONPATH=. python baselines/tests/test_baselines.py -v
```

## 九、八轮实测结果与结论（2026-11-14，本地运行，未消耗平台次数）

| 轮次 | 改动 | 改前 | 改后 | 变化 | 决策 |
|---|---|---|---|---|---|
| 1 | 无 | 0.714030 | 0.725782 | +0.011752 | 采纳 |
| 2 | 学习率 → 0.0015 | 0.725782 | 0.717688 | -0.008094 | 回滚 |
| 3 | 学习率 → 0.006 | 0.725782 | 0.722551 | -0.003231 | 回滚 |
| 4 | 轮内步数 5→10 | 0.725782 | 0.718114 | -0.007668 | 回滚 |
| 5 | 轮内步数 5→3 | 0.725782 | 0.726366 | +0.000584 | 采纳 |
| 6 | 轮内步数 3→2 | 0.726366 | 0.714093 | -0.012274 | 回滚 |
| 7 | 从被采纳轮再走 3 epoch（112→124 步） | 0.726366 | 0.720465 | -0.005901 | 回滚 |
| 8 | 再走 2 epoch（112→120 步） | 0.726366 | 0.714093 | -0.012274 | 回滚 |

四条结论：

1. **学习率两侧都会掉。** 0.0015 和 0.006 分别掉 0.0081 和 0.0032，说明 0.003 已经贴近这个结构的最优点位，继续在这一维上调不划算。
2. **步数存在峰值，区间很窄。** 第 4 轮把步数从 100 加到 140，指标降 0.0077；第 5 轮只加 12 步（100→112），反而升 0.0006。峰值大致落在 110 步附近，再往前就是在拟合 445 行的训练批次，跨批次指标随之回落。
3. **梯度下降这条路接近用尽。** 八轮里六轮回滚、两轮小幅提升，累计比基线高 0.0123，后面必须换到损失函数、正则与结构迁移上，继续调学习率和步数只会重复消耗轮次。
4. **这六次回滚就是"最差阶段与恢复速度"的证据。** 每轮都有父检查点摘要、失败的改动与回滚决策，写报告时直接引用 `runs/round-2~4`、`runs/round-6~8` 与 `logs/experiments.csv` 即可，不需要另造材料。

备选做法（已放弃）：曾考虑在第 5 轮引入"轮内按验证指标保留最优权重"。这属于用验证集选模型，在"过拟合"这一评分维度上有风险，最终没有采用——第 5 轮只调了轮内 epoch 数（3 epoch / 112 步），验证集全程未参与训练。

## 十、Docker 镜像

`docker/Dockerfile` 把本工程打成可离线运行的镜像，基础镜像是 `python:3.10.11-slim`，与官方锁定的 3.10–3.12 和 NumPy 1.26.4 对齐。构建期会执行 `scripts/selfcheck.py`，9 项官方测试不通过则构建直接失败，所以镜像能出来就代表环境是干净的。

构建与验证：

```sh
sh docker/build.sh          # Linux / macOS
powershell -ExecutionPolicy Bypass -File docker/build.ps1   # Windows
```

脚本按四步走：构建 → 镜像内自检 → 入口 dry-run → 可选导出 tar（加 `--save` 或 `-Save`）。

跑完整闭环（镜像入口就是 Harness；`runs-docker` 用空目录，避免与本地已有的 `runs/` 冲突）：

```sh
docker run --rm \
  -v $PWD/runs-docker:/app/runs \
  -v $PWD/logs:/app/logs \
  -v $PWD/results:/app/results \
  -v $PWD/state.json:/app/state.json \
  rsi-a-cpu:1.0 --unattended
```

不挂 `state.json` 时，回滚状态只在容器内有效，容器退出即丢，多轮连续运行必须挂上。

提交用的镜像 tar：

```sh
sh docker/build.sh --save     # 产出 rsi-a-cpu-1.0.tar
```

镜像内已固定 `PYTHONHASHSEED=42`、`BASELINE_CPU_THREADS=1` 与 `OMP/OPENBLAS/MKL_NUM_THREADS=1`，同一检查点重复构建与运行的可复现性由此保证。

### 无 Docker 引擎下的构建结果与运行验证（2026-11-14）

本机装不上 Docker Desktop（无 WSL2，安装器最后一步 TLS 网络中断），改在一台无 docker、无 root、无 sudo 的 Linux 主机上按 docker-archive 规范逐字节拼装，产出可直接 `docker load` 的归档。同一镜像有两个归档版本，用途不同：

| 归档 | 体积 | 内层 layer.tar | sha256 | 用途 |
| --- | --- | --- | --- | --- |
| `rsi-a-cpu-1.0-raw-8rounds.tar` | 198,891,520 字节（189.7 MiB） | 未压缩 plain tar（6 层） | `3caa429ffaf0c6e3bf4f17ef7a4aff214c19198235eb6da0cf6fedef785ff89b` | **平台提交件**：符合"docker save 导出的未压缩 tar"要求 |
| `rsi-a-cpu-1.0.tar` | 64,880,640 字节（61.9 MiB） | gzip 字节（6 层） | `4d856174c448525deed5d755b6c02af26d351527ef9dd605eaf9b7be95b243d5` | 仅本地 / udocker 装载使用；**不作为平台提交件**（平台明示拒收 gzip 压缩包，实测该件上传返回 502） |

两者镜像一致：config 摘要 `sha256:e820a301dc86…`（load 后即镜像 ID），linux/amd64，6 层（基础镜像 5 层 + 应用层 1 层），基础镜像 `library/python:3.10.11-slim`（amd64，5 层 diff_id 原样沿用），Entrypoint `python scripts/run_harness.py`，Cmd `--unattended`，WorkingDir `/app`。

已完成的两级验证：

- 结构：6 层 `layer.tar`（raw 归档为未压缩原件，gz 归档解压后）的 sha256 与 config 的 `rootfs.diff_ids` 逐条一致；manifest 指向的 config 摘要与文件名一致；条目名不带 `./` 前缀。
- 运行：干净状态（清空 `runs/`、`logs/`、`results/` 与 `state.json`）下在 Linux 主机执行 `--unattended`，ENTRYPOINT、WorkingDir=/app、numpy 1.26.4 全部正常，五环跑完八轮，数值与 `logs/experiments.csv` 逐位一致：0.725782 / 0.717688 / 0.722551 / 0.718114 / 0.726366 / 0.714093 / 0.720465 / 0.714093，最终 0.726366，静态基线 0.714030 满足，相对起点 +0.012337。更早一轮还在 udocker（免 root 容器运行器，PRoot 引擎）内装载同版归档跑通过前六轮。

如实标注：本镜像没有在 Docker Engine / Docker Desktop 上跑过（构建主机没有 docker），对外只说“结构已验证 + 容器内运行已验证”，不说“docker build 通过”；真实 Docker 环境验收以平台试运行结果为准。

冷启动复现：镜像内不带 `logs/experiments.csv` 与 `runs/`，因此从第 1 轮起跑。本机目录里若已有八轮台账，想重跑冷启动，先把 `logs/experiments.csv` 与 `runs/` 移出去。

`configs/harness.json` 的 `stop_after_rollback_streak` 早期取 4（预设脚本第 2–4 轮连续三次回滚，上限取 3 会在第 4 轮后停住、拿不到第 5 轮的峰值 0.726366）。按十轮口径改为 10：预设第 2–4 轮、第 6–10 轮都在回滚，若仍取 4，第 9 轮前就熔断，等于没跑满十轮；取 10 让十轮规划全部走完，由轮次上限收尾。

## 十一、Harness 自循环

分工：`scripts/run_round.py` 只负责执行一轮并维护回滚状态；`harness/loop.py` 负责把多轮串成一次迭代循环。

循环严格按规则的五环组织，每一环都有独立模块与日志字段：

| 环节 | 模块 | 做什么 |
|---|---|---|
| 漂移监测 | `harness/monitor.py` | 以初始训练集为参照，算特征标准化偏移与类先验总变差，按批次定位最差批次 |
| 诊断归因 | `harness/diagnose.py` | 把偏移归到特征漂移／类先验漂移／过拟合训练批次／批次局部漂移，给出对应动作 |
| 实验优化 | `harness/decide.py` | preset（离线）或 llm（调平台模型）产生本轮唯一改动 |
| 评估上线 | `scripts/run_round.py` | 跑一轮、比对指标、达标则更新 `state.json` 为上线版本 |
| 回滚 | `scripts/run_round.py` | 未超过参照值即回滚，父检查点回退到被采纳轮次，失败目录保留 |

受控机制（对应规则里"防止无限制恶意迭代"）：轮次上限 `max_rounds`、累计时长上限 `max_wall_seconds`、Token 上限 `token_budget`、连续回滚上限 `stop_after_rollback_streak`，任一触发即收尾。

### 采纳的三道门槛（contest_submission.py）

单点比较（候选分比父模型好一点就采纳）在回归任务上会被噪声带偏：验证集整体微涨，但某个阶段可能已经崩了，而平台计分里"最差阶段"占 20%。因此 `adopt_decision()` 设三道门槛，任何一道不过就保留父模型：

| 门槛 | 判据 | 常量 |
|---|---|---|
| 更优 | 候选必须严格优于父模型 | — |
| 有实质提升 | 相对提升 ≥ 0.2%，避免采纳数值噪声 | `ADOPT_MARGIN = 0.002` |
| 不许塌方 | 把验证集按目标量切成 4 段（回归按 RUL 从低到高，分类按标签分层），任一段相对回退 > 10% 即拒绝 | `WORST_BLOCK_TOL = 0.10`、`BLOCK_COUNT = 4` |

拒绝采纳时父检查点原样保留，并在 `round.json` 写入 `adopt_reason` 与各分段落差；连续两轮保留父模型后，学习率按 `0.5^(streak-1)` 自动回落，避免在同一步长上反复撞墙。分段切片按目标量排序，是"时间先后"的近似，不依赖平台未公开的阶段划分。

### 候选锚定与平台口径择优（v10）

平台正式评测 50.68 的复盘暴露出两件事：数据集 A 的八个阶段里早段 Macro-F1 只有 0.60 上下、后段 0.74–0.79，十轮续训把权重持续往后段分布推；数据集 C 四段 RMSE 全部低于基线，第三段从 48.07 崩到 78.80。而计分是"平均提升 80% + 最差阶段 20%"，被牺牲的那一段正好是最贵的地方。

v10 不动网络、不动起点，只改"提交哪份权重"：

| 机制 | 实现 | 作用 |
|---|---|---|
| 锚定候选 | `anchor(parent, trained, r)` 逐字段算 `(1-r)·θ训后 + r·θ父`，r 取 0 / 0.25 / 0.5 / 0.75 四档 | 把权重拉住，没训到的阶段不退化 |
| 平台口径择优 | `platform_objective()` 按 `0.8·平均相对提升 + 0.2·最差分段相对提升` 打分，四档里取最高 | 直接对齐计分公式，而不是只看单点验证分 |
| 逐 epoch 早停 | 配方里的 `epochs` 不再一次走完：每个 epoch 结束都评估四档候选，取全局最优快照，最优 epoch 写入 `epochs_used` | 走过头再回头混合，等于先过拟合再补救；早停把补救提前到每一步 |
| 阶段数对齐 | `BLOCK_COUNT_BY = {"A": 8, "B": 3, "C": 4}`，按平台公布的阶段数切段 | 护栏粒度与计分粒度一致，A 不再用 4 段近似它实际的 8 段 |
| C 冻结 | `FREEZE_DATASETS = ("C",)`，该数据集只提交候选、不推进权重 | C 阶段跨度大，推进只会放大负提升 |

父权重取训练前快照（`copy.deepcopy`），混合比例写进检查点 `anchor_ratio` 与轮次配置，每一轮都能查到"这轮交的是哪一档"。学习率配方表本轮不动：收缩交给锚定择优自动完成，同一轮只调一个旋钮，收益才可归因。

注意一个局限：公开数据只有 b01（训练）/ b02（验证）两个批次，切不出平台那 8 个独立测试时段的结构，所以本地只能验证"有没有退化"，验证不了"能不能涨分"——涨分与否必须由平台隐藏集给出。

### v12：把 v11 的锚定回退，只留 C 冻结

两次正式评测给了对照：v8（单点比较、不锚定）50.68；v11（四档锚定 + 逐 epoch 早停 + 8/3/4 段护栏）49.88。拆开看，v11 的 C 冻结确实把 C 从 23.33 拉回基线附近，但 A、B 从 64.42 / 64.29 掉到 50 上下——锚定与 8 段护栏把均值提升一起掐死，净亏 0.8。

教训很直接：公开验证集只有两个批次，切出来的“段”不是平台的 8 个独立时段，用它做代理去否决候选，会把真提升也一起否决；逐 epoch 早停则把“batch 2 上最好的一刻”当成最优，进一步拟合单批次。

v12 做减法：

| 开关 | 值 | 含义 |
|---|---|---|
| `ANCHOR_RATIOS` | `(0.0,)` | 不锚定 |
| `EARLY_STOP` | `False` | 按配方一次训完，seed=42 |
| `USE_BLOCK_GUARD` | `False` | 回到单点比较 |
| `ADOPT_MARGIN` | `0.0` | 优于父模型即采纳（v8 行为） |
| `DECAY_ON_ROLLBACK` | `False` | v8 未接回滚降学习率，一并关掉 |
| `FREEZE_DATASETS` | `("C",)` | 唯一保留的新增：C 只提交候选，不推进权重 |

本地复算：第 1、2 轮公开验证分 0.718735 / 0.726830，与 v8 记录逐位一致，说明训练轨迹已回到 v8。预期总分 `(64.42 + 64.29 + 50) / 3 ≈ 59.6`。

锚定、早停、分段护栏的代码留在开关后面，默认关闭，方便复现对照——机制本身没错，错在用错了代理集。

### v13：冻结必须按数据集名生效

v11 的 C 冻结实际上没生效。原因是 C 数据集的父检查点与当前数据维度不兼容（`inherited=False`），而冻结条件写成了 `and inherited`，于是条件恒假——报告里 C 的十轮全是“采用本轮模型”，还从零重训花掉 8,686 秒，把资源效率打到 0/5。这解释了 v11 为什么反而低 0.8 分。

v13 把冻结改成按数据集名判断，不再依赖继承状态：

- 冻结轮次完全不训练，不建新模型；
- 候选 = 父检查点**逐字节复制**，`improved=False` 恒定，最终产物就是 baseline 本身；
- 单轮耗时从数千秒降到秒级，资源效率那 5 分同时捡回来。

其余行为保持 v8 原样：一次训完、单点比较、不降学习率。

决策层有两种实现：

| 决策器 | 用途 | 依赖 |
|---|---|---|
| `preset` | 读 `configs/harness.json` 里的预设脚本，离线把链路跑通 | 无 |
| `llm` | 调平台模型产生决策，带 JSON 校验与成本累计 | 标准库 |

`llm` 需要三个环境变量，不要把 key 写进代码：

```sh
PA_MODEL_BASE_URL=https://xxx/v1
PA_MODEL_API_KEY=xxx
PA_MODEL_NAME=xxx
python scripts/run_harness.py --decider llm
```

运行方式：

```sh
python scripts/run_harness.py --max-rounds 2 --dry-run   # 只走链路，不训练
python scripts/run_harness.py --max-rounds 1             # 用预设方案真跑一轮
```

每轮写一条 `logs/harness.jsonl`，字段为 `ts / round / decider / status / action / wall_seconds / macro_f1_after / decision / prompt_tokens / completion_tokens / cost_cny`。Token 与成本现在是 0，接上模型后自动累计；单价回填 `configs/harness.json` 的 `price`。

SwanLab 记录（已接通云端）：`swanlab.mode` 为 `local` 时只写本地 `swanlog/`，不需要登录；设为 `cloud` 并配置 `SWANLAB_API_KEY` 即写云端。本工程默认 `cloud`；未安装 swanlab、没有 key 或云端失败时只打印提示，不阻断本地闭环。

云端记录实测（2026-11-14，Python 3.10 + swanlab 0.10.4，真实执行第 7、8 轮）：

```text
项目地址：https://swanlab.cn/@slience917/rsi-a-cpu
本次 run：https://swanlab.cn/@slience917/rsi-a-cpu/runs/ek5f1ezu
写入内容：58 条记录，含每轮 macro_f1 / seconds / deployed / prompt_tokens / cost_cny，收尾写 final_macro_f1、meets_baseline、total_wall_seconds
```

第 7 轮（从 112 步再走 3 epoch）得 0.720465，回滚；第 8 轮（再走 2 epoch）得 0.714093，回滚。两轮都低于 0.726366，与“峰值在 112 步附近”一致；且第 8 轮与第 6 轮数字完全相同，证明整条链路可复现。

接入千问（把决策层从预设脚本换成平台模型）：

```powershell
# 1) 配三个变量（只对当前窗口生效；写死到用户环境用 [Environment]::SetEnvironmentVariable(...,"User")）
$env:PA_MODEL_BASE_URL = "https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:PA_MODEL_API_KEY  = "sk-xxxxxxxx"
$env:PA_MODEL_NAME     = "qwen3.8-max"

# 2) 先探针，不跑训练、只花一次极小请求
python scripts/probe_llm.py              # 列模型 + 发一次请求，校验输出是不是合法 JSON
python scripts/probe_llm.py --list-only  # 只列模型，不消耗 token

# 3) 探针通过后再跑真实一轮
python scripts/run_harness.py --decider llm --max-rounds 1
```

`PA_MODEL_BASE_URL` 填到 `/v1` 为止，代码会自动拼 `/chat/completions`；模型名以 `GET /models` 返回值里的真实 id 为准，不要猜。单价回填 `configs/harness.json` 的 `price.prompt_per_1k` / `completion_per_1k`，日志里的 `cost_cny` 才会变成真实数字。

框架选型说明：循环用标准库实现，没有引入外部框架，四个环节与 LangGraph 的 Node、条件边一一对应。评委若要求"自选开源 Agent 框架"，把 `decide` / `execute` / `evaluate` / `record` 注册成四个节点、把 `采纳|回滚` 作为条件边即可，循环逻辑不变，也不会因为换框架而改变实验结论。

已验证：dry-run 走通全链路；真实执行第 5 轮（epochs 3，采纳）、第 6 轮（epochs 2，回滚）、第 7 轮（epochs 3，回滚）、第 8 轮（epochs 2，回滚）。八轮记录全部写入 `logs/experiments.csv` 与 `logs/harness.jsonl`，收尾产出 `results/final.json`。

无人值守运行（对应封闭评测的入口）：

```sh
python scripts/run_harness.py --unattended
```

结束时 `results/final.json` 会写明停止原因、最终检查点、最终 Macro-F1、是否不低于静态基线、累计时长与 Token 成本，评测方据此读取结果，不需要人工介入。

## 十二、与官方规则的逐条对照

依据《"智训引擎"—工业制造场景下的模型训练与超参数优化》赛题规则（第二届北京市大学生"人工智能+"创新大赛）。

| 规则要求 | 本工程对应 | 状态 |
|---|---|---|
| 漂移监测→诊断归因→实验优化→评估上线→回滚 闭环 | `harness/` 五个模块，每环有独立日志字段 | 已实现 |
| 系统无人工干预运行并输出结果 | `--unattended` + `results/final.json` | 已实现 |
| 系统性能不低于组委会静态基线 | 回滚机制 + `meets_static_baseline` 判定 | 满足（0.7264 vs 0.7140） |
| 全部实验按《实验记录规范》留完整记录 | `logs/experiments.csv` + `logs/harness.jsonl` + SwanLab 云端 | 满足：8 轮本地台账 + 云端 run 已产出 |
| 单位增益的算力消耗可核算 | 每轮 `wall_seconds`、累计时长、Token 与成本字段 | 字段就位，成本待回填单价 |
| 驱动 Harness 的大模型须为 PyTRiO 模型或 Qwen3.8-27B | `harness/gateway.py` 双 provider：正式评测走平台注入的 `model()` 网关，本地调试可退 OpenAI 兼容端点；Token 与成本逐轮入账 | 网关探测与记账已实现；模块名与签名待契约确认 |
| 受控机制，防止无限制迭代 | 轮次／时长／Token／连续回滚四道上限 | 已实现 |
| 提交模型配置与 Harness 系统镜像 | `docker/Dockerfile`（入口即 Harness），提交归档 `rsi-a-cpu-1.0-raw-8rounds.tar`（未压缩 docker save tar，189.7 MiB） | 已构建并通过结构 + 干净状态八轮冒烟；平台上传曾返回 502，待重传与真实 Docker 试运行验收 |
| 提交系统运行说明文档 | 本 README | 已实现 |

两条从数据里读出来的事实，写报告漂移章节时可直接用：

1. **训练集与验证集各自只有一个批次**（训练集批 1 共 445 行，验证集批 2 共 1,244 行），所以这里的漂移是"批次之间的漂移"，不是"流式到达的漂移"，报告里要写清口径，别把它说成在线流。
2. **两批的类别先验差得很远**：Toluene 在训练集占 16.6%（74 行），在验证集只占 0.4%（5 行）；Acetone 从 15.7% 涨到 42.8%。类先验总变差 0.339。这意味着 Macro-F1 里 Toluene 那一项只由 5 个样本决定，评分波动天然就大——这既解释了为什么指标卡在 0.72 附近，也说明下一步最值得做的是类别加权，而不是继续调步数（第 6 轮 +8 步直接从 0.7264 掉到 0.7141 就是证据）。

## 十三、来源与许可证声明

| 项 | 来源 | 许可证 / 使用方式 |
| --- | --- | --- |
| 数据 | 组委会下发的 A 赛道公开包 `public/train.jsonl`（445 行）、`public/validation.jsonl`（1,244 行）；原始数据出自 UCI Machine Learning Repository, *Gas Sensor Array Drift Dataset* (#224)，https://archive.ics.uci.edu/dataset/224/gas | 只使用官方下发的 JSONL，**不重新分发原始 `.dat` 与隐藏测试集**；两份 JSONL 的 sha256 记录在 `runs/round-5/training.json`（`train_sha256` / `validation_sha256`） |
| 官方基线代码 | `baselines/` 目录为组委会随包提供的参考实现（`model.py` / `data.py` / `train.py` / `predict.py` 及配置、测试） | 按官方包自带说明使用；本队未改动模型定义与训练逻辑，改动痕迹全部落在 `runs/` 与 `logs/`，可逐轮比对 |
| 本队原创 | `harness/`（`loop.py` / `decide.py` / `monitor.py` / `diagnose.py` / `swan.py`）、`scripts/`（`run_round.py` / `run_harness.py` / `summarize.py` / `selfcheck.py` / `dataset_profile.py` / `probe_llm.py`）、`docker/` 构建脚本、`configs/` 配置 | 本队成员编写 |
| 第三方依赖 | numpy 1.26.4、Python 3.10.11、基础镜像 `library/python:3.10.11-slim`（Debian + CPython） | numpy 为 BSD-3-Clause；CPython 为 PSF 许可；基础镜像内 Debian 组件按其各自许可。依赖清单见 `requirements.txt` |
| 外部模型 | 无 | 校赛阶段 Harness 决策层为 `preset` 离线脚本，未调用任何大模型；`llm` 决策器只是 OpenAI 兼容接口的实现，需自备密钥，**仓库内不含任何模型权重、API Key 或凭据**（SwanLab key 通过环境变量注入） |
| 本队代码许可证 | — | 暂无指定开源许可证，默认保留所有权利；如需公开复用，可在此处补充 MIT / Apache-2.0 等条款 |

图片、图标与第三方字体：本工程不含自制或引入的图形素材，产物只有数值检查点、JSON 与 Markdown。

## 十四、工作区三模式与挂载点（历史：SDK 到手前的自研适配层）

平台在工作区契约 `contest.harness.workspace.v2` 下要求入口按 `describe / train / predict` 分派，并读写固定挂载点。本工程据此加了适配层，原有的本地用法完全保留。

### 三个模式

| 模式 | 命令 | 作用 | 写盘 |
| --- | --- | --- | --- |
| describe | `python scripts/run_harness.py describe` | 输出能力清单（任务/指标/类别/挂载点/轮次来源/入口/adapter_source_ref），供平台校验 | 只写 stdout，不落盘 |
| train | `python scripts/run_harness.py train --unattended` | 五环闭环：漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚 | `runs/`、`logs/`、`results/`、`state.json` |
| predict | `python scripts/run_harness.py predict` | 独立预测入口，逐行输出 `[[label, score], ...]` | 默认 `predictions.jsonl` |

不带子命令时等价于 `train`，因此镜像里原有的 `CMD ["--unattended"]` 仍然可用。

### 四个挂载点

适配层只认挂载点，不再写死仓库内路径；目录不存在时回落到仓库根，所以同一份代码本地也能跑：

| 挂载点 | 用途 | 本地回退 |
| --- | --- | --- |
| `/data` | `train.jsonl`、`validation.jsonl`、`test.jsonl` | `public/` |
| `/previous` | 上一轮检查点与状态 | 仓库根 |
| `/model` | 初始模型 / 基线权重 | `checkpoint.npz` |
| `/output` | 本轮产物（`runs/`、`logs/`、`results/`、`state.json`） | 仓库根 |

四者都可用环境变量覆盖（`HARNESS_DATA_DIR` 等），便于契约未下发前先做联调。

### 轮次

轮次**不写死**：优先级为 平台 `workspace_suite.rounds` ＞ 环境变量 `HARNESS_ROUNDS` ＞ 命令行 `--max-rounds` ＞ `configs/harness.json` ＞ 缺省 10。提交镜像内配置取 10，与规则「每个数据集 10 轮」一致；平台若下发 `workspace_suite.rounds`，以平台为准。

### 大模型

`harness/gateway.py` 提供两个 provider：

- `PlatformGateway`：按 `HARNESS_GATEWAY_MODULES`（默认探测 `harness_runtime` / `workspace_runtime` / `platform_runtime` / `model_gateway` / `harness_gateway` / `rsi_runtime` 六个模块的 `model` / `chat` / `complete` / `invoke` / `generate` 属性）找到平台注入的 `model()` 网关；
- `OpenAICompat`：`PA_MODEL_BASE_URL` + `PA_MODEL_API_KEY` + `PA_MODEL_NAME`，**仅本地调试**。

两者都返回 Token 用量，`harness/loop.py` 逐轮累加并写进 `logs/harness.jsonl`（`prompt_tokens` / `completion_tokens` / `cost_cny`）与 `results/final.json`（`total_tokens` / `total_cost_cny`），不再固定为 0。

找不到任何 provider 时**直接报错退出**，错误信息里列出试过的全部候选项，不静默退回、不伪造调用。

### 本地自查

```sh
python scripts/run_harness.py describe           # 看能力清单
python scripts/run_harness.py predict --limit 3  # 有 test.jsonl 时可用
python scripts/run_harness.py train --unattended # 十轮闭环（preset）
python scripts/selfcheck.py                      # 镜像自检 + 官方测试
```

适配层源码位置记录在 `adapter_manifest.json` 的 `adapter_source_ref` 字段（当前指向 `scripts/run_harness.py`）。该字段的取值形式待契约确认后可能调整。

### 镜像内实测（2026-11-16）

用 udocker（PRoot 引擎，免 root）装载 `rsi-a-cpu-1.0.tar` 后逐项执行，全部通过：

```sh
udocker load -i rsi-a-cpu-1.0.tar
udocker run --rm --entrypoint=python rsi-a-cpu:1.0 scripts/selfcheck.py            # Python 3.10.11 / NumPy 1.26.4 / 15 数组 13235 元素 / 官方测试 OK
udocker run --rm --entrypoint=python rsi-a-cpu:1.0 scripts/run_harness.py describe  # schema=contest.harness.workspace.v2，6 类，4326 参数，rounds=10
udocker run --rm --entrypoint=python rsi-a-cpu:1.0 scripts/run_harness.py predict --input /app/public/validation.jsonl --limit 3 --output /tmp/p.jsonl
udocker run --rm --entrypoint=python rsi-a-cpu:1.0 scripts/run_harness.py train --max-rounds 1 --dry-run
```

归档本身为 `docker save` 未压缩 tar 格式，198,922,240 B（约 190 MiB），6 层（5 层基础 + 1 层应用 68 MB），sha256 以 `949588b104aa0ac9` 开头。四个挂载点未挂载时回落到 `/app` 内路径，`describe` 的 `mounts` 字段会如实输出 `null`。

### 预测协议（对齐官方基线包）

官方基线包 `baselines/README.md` 写明：分类输出 `outputs:[{label,score}]`、回归输出 `outputs:{value}`，「与平台预测协议一致」。本工程 `predict` 默认输出该协议：

```sh
# 文件模式：读 /data/test.jsonl，写 /output/predictions.jsonl
python scripts/run_harness.py predict

# 管道模式：stdin 逐行读、stdout 逐行写（与 baselines/predict.py 的调用方式一致）
echo '{"task":"classification","inputs":{"sensors":[...]}}' | python scripts/run_harness.py predict --input -
```

`--format pairs` 保留旧形状 `[[label, score], ...]` 仅供本地比对，正式评测不要用。

### v14：C 站上基线再微步，四段护栏保最差阶段

v13 的冻结解决了"不再从零重训"，但 C 等于交给平台一个 baseline，最差阶段那 5 分拿不到。
v14 从两条线改：

1. **先让 C 能继承**。C 的父检查点 paths 与当前阶段字段不完全一致，`usable()` 抽查失败会
   触发 `union_paths()` 重建，`can_inherit()` 随即因 `net.paths != paths` 返回 False。现在改为：
   父路径存在但部分不可用时，只要 `path_coverage()` 覆盖度 ≥ 50%，就沿用父路径、缺失通道由
   `core.extract` 的 `fill=0.0` 补齐；覆盖度不足才回退重建。继承一旦成立，C 就从 baseline 权重起步。

2. **微步 + 强锚定**。C 固定 1 个 epoch、lr 5e-4、batch 64；训完用 `anchor(parent, model, 0.9)`
   得到锚定候选，与未锚定候选比验证分取更优者，只有优于父模型才采纳。若继承仍失败，
   该数据集自动降级为 v13 的冻结语义（不训练、提交父检查点原样）。

3. **Token 压缩**。上下文只留"数据集/轮次/指标/父分"，历史从最近 5 轮减到 2 轮，
   输出上限 96 → 48 token，system 提示词合并进一行 JSON 模板。

开关现状：`FREEZE_DATASETS=()`、`ANCHOR_RATIOS=(0.0,)`、`EARLY_STOP=False`、
`ADOPT_MARGIN=0.0`、`BLOCK_COUNT_BY={}`，锚定/早停/护栏代码保留在开关后，默认关闭。
本地复算 A 前两轮公开验证分 0.718735 / 0.726830，与 v8 逐位一致。

### 本地干跑（2026-11-25，公开验证集）

`dryrun_any.py A|B|C` 用假 Round 按平台节奏跑十轮，模型调用打桩，只看公开集：

- A：0.714030 → 0.718735 → 0.726830，后八轮候选均低于父模型，全部退回，终版 +1.79%；
- B：RMSE 681.16 → 656.25（首轮采纳）、655.77（第 9 轮再采纳），终版 +3.72%；
- C：RMSE 52.8686 → 51.9075（+1.82%），只采纳第 1 轮；四段 FD001–FD004 增益
  +3.77% / +0.24% / +4.66% / +1.20%，四段均非负。其余九轮因 FD002 回退被零容差护栏拒绝。

C 的阶段切分按 `meta.source_subset`（FD001–FD004），容差 0：任一阶段回退就整轮不采纳。
A、B 不启用该护栏，仍是 v8 的单点比较。十轮耗时：A 2.0 s、B 2.1 s、C 12.4 s。

C 的配方在本地扫过 36 组（lr 5e-4/1e-3/2e-3 × epochs 1/2/3 × 锚定 None/0.5/0.75/0.9）：
锚定同时压低总增益与最差段，1 epoch 的最差段只有 +0.24%，2 epoch 不锚定可到 +0.75%；
再叠加“每轮换种子”，十轮里第 4 轮命中最优候选，终版 +2.04%、四段 +4.25/+0.82/+5.66/+1.03。

镜像 description 已改为「智训引擎：数据集 A/B/C 续训原型（CPU）」，避免出现“赛道”字样。
