# 智训引擎 v19 实验记录

本记录对应镜像内的 `contest_submission.py` 与 `rsi/core.py`。每轮都从平台父检查点继续，继承权重、标准化参数和 Adam 状态；模型返回的 `learning_rate / epochs / batch_size` 参与本轮训练。

## 本地协议干跑（公开 train/validation）

| 数据集 | 官方 baseline | v19 干跑 | 关键设置 |
| --- | --- | --- | --- |
| A | Macro-F1 0.714030 | 0.751813 | l2 逐样本归一化，隐层 32→64，lr 0.003，3 epoch，batch 64 |
| B | RMSE 681.159833 | 517.570716 | rowstd 逐行标准化，隐层 24→64，lr 0.003，batch 64，B 段不做硬护栏 |
| C | RMSE 52.868573 | 51.528548 | raw，隐层 32→64，lr 0.002，batch 256，10 轮 |

A 首轮第 3 个 epoch 采纳；B 首轮第 1 个 epoch 从 681.16 降到 520.80，后续第 8、10 轮继续小幅采纳；C 第 2、3、5 轮采纳，最终 51.53。

## 与 v17 的区别

- 每轮 `load_parent()` 读父检查点，不再读取 `best_ckpt`。
- 每轮真实执行 Adam 更新，不冻结、不复制父检查点当候选。
- 模型提示词要求输出 JSON 超参，解析结果直接进入 `net.fit`，只做范围裁剪。
- 训练用公开 `train.jsonl`；`validation.jsonl` 只用于训练前后比较和分段护栏。
- 旧 `FAST_FREEZE`、`BEST_DIR`、`probe_round`、写死 schedule 已删除。

## 运行边界

A 用 445 行训练、1,244 行验证；B 用 7,032 行训练、1,679 行验证；C 用 113,089 行训练、23,049 行验证。隐藏测试标签不读取、不推断。
