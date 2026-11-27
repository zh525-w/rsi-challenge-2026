# 智训引擎：A/B/C 三数据集续训工程（Harness SDK 1.0.5）

本目录把三个官方训练包原样接入，共用一套适配层与 NumPy 模型：

- A：UCI Gas Sensor Array Drift，六类气体分类，Macro-F1；
- B：XJTU-SY 轴承 RUL，回归，RMSE；
- C：C-MAPSS FD001 RUL，回归，RMSE。

平台入口固定为 `/app/harness.py`，真正的选手代码是 `/app/contest_submission.py`。

```text
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

## 训练路径（v19）

每一轮都从平台给出的父检查点继续，不再有快速冻结、预置最佳权重或 400 行探针：

1. 读取公开 `train.jsonl` 与 `validation.jsonl`。C 的 113,089 行训练样本全量读入，验证集 23,049 行全量保留。
2. 读取父检查点的权重、标准化参数、Adam 一阶/二阶矩和累计步数；网络宽度不一致时直接报错，不会偷偷从零重训。
3. 调用平台模型一次，让模型根据“当前数据集、训练前验证分、最近两轮结果”返回本轮
   `learning_rate / epochs / batch_size`；这三个值直接用于本轮训练，只做安全范围裁剪。
4. 在公开训练集上执行真实梯度更新，Adam `step` 随父检查点继续累加；候选检查点保存训练后的权重和优化器状态。
5. 在公开验证集上比较训练前后的指标，并做分段护栏：平均分变好且没有阶段崩掉才采纳候选，否则 `finish` 仍选父模型，但候选本身是真实训练出来的。
6. `round.json` 与 `finish.configuration` 同时记录 `validation_before`、`validation_after`、`model_reply`、`model_call_id`，便于平台复核。

v19 在三套数据上分别做逐样本归一化（A）、逐行标准化（B）与原始尺度（C），隐层由父检查点的 32/24/32 扩到 64；新单元权重置零，扩宽后的第一步输出与父模型完全一致。没有“把父检查点原样复制一份当候选”的分支；也不长期冻结任何数据集。C 每轮照常训练，只有分段护栏在候选某段明显变差时阻止它被选为最终模型。

## 目录

```text
contest_submission.py   训练、决策、采纳护栏、预测
harness.py              SDK 入口
rsi/core.py             NumPy MLP、特征抽取、打分
contest_harness/        SDK 1.0.5 协议层
public/                 公开训练/验证投影
baselines/              官方基线代码与说明
TASK.md                 三个数据集的任务与标签说明
TRAINING_GUIDE.md       续训命令与隔离要求
```

## 数据边界

梯度更新只用 `public/train.jsonl`；`public/validation.jsonl` 只用于比较训练前后和分段护栏，不参与梯度更新。隐藏测试集不随包分发，代码不读取、不推断隐藏标签。

## 本地干跑

```sh
python3 dryrun_any.py A
python3 dryrun_any.py B
python3 dryrun_any.py C
```

干跑使用本地协议替身，只验证协议、继承、训练、提交和状态链路，不消耗平台试运行或正式评测次数。
