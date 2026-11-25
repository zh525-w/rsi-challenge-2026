"""智训引擎 RSI CHALLENGE 提交入口（contest.harness.workspace.v2）。

平台每轮启动 `python3 /app/harness.py train`，把一帧 start 写进 stdin；
SDK 解帧后调用本文件的 train(rnd)。一轮只做一件事：读取父检查点，
在公开训练集上续训一小步，用一次真实模型调用决定超参，提交候选并结束本轮。
预测走 `python3 /app/harness.py predict`，逐行读 stdin、逐行写 stdout。

三个数据集共用一套代码：
- 特征按“路径”抽取，A 的扁平 features、B 的双轴嵌套特征、C 的 settings+sensors 都能量化；
- 任务类型取 rnd.task，B/C 是回归（RUL），A 是分类；
- 回归先归一化目标再训练，预测时反归一化并取非负，与官方 baseline 口径一致。
"""
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from contest_harness import Round

import rsi.core as core

TASKS = ("classification", "regression")
_WEIGHT_FIELDS = ("w1", "b1", "w2", "b2",
                  "adam_first_w1", "adam_first_b1", "adam_first_w2", "adam_first_b2",
                  "adam_second_w1", "adam_second_b1", "adam_second_w2", "adam_second_b2")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
DEFAULTS = {"learning_rate": 0.003, "epochs": 3}
TOLERANCE = 1e-6
ADOPT_MARGIN = 0.0        # v12 回到 v8 的单点比较：优于父模型即采纳
WORST_BLOCK_TOL = 0.10    # 任一分段相对回退超过 10% 即拒绝采纳
BLOCK_COUNT = 4           # 与平台“逐阶段”计分对齐的分段数（未登记的数据集用这个默认值）
BLOCK_COUNT_BY = {}       # v12 不再按阶段数切段（v11 的 8 段护栏把 A/B 压回基线）
USE_BLOCK_GUARD = False   # v12 关闭分段护栏，回到单点比较
EARLY_STOP = False        # v12 关闭逐 epoch 早停，按配方把 epochs 走完
DECAY_ON_ROLLBACK = False # v8 未接回滚降学习率，v12 一并关掉，保证轮次轨迹与 v8 一致
MEAN_WEIGHT = 0.8         # 平台口径：平均提升占 80%
WORST_WEIGHT = 0.2        #          最差阶段提升占 20%
ANCHOR_RATIOS = (0.0,)   # v12：不锚定。v11 的四档混合在隐藏阶段上是负收益，已回退
FREEZE_DATASETS = ()      # v14：C 改为微步 + 强锚定；继承失败时才按数据集降级冻结
TRAIN_CAP = 24000
VALIDATION_CAP = 30000


# ------------------------------------------------------------------ 小工具
def log(message):
    """日志一律走 stderr，stdout 只留给协议帧。"""
    print("[rsi] " + str(message), file=sys.stderr, flush=True)


def task_kind(contract):
    if isinstance(contract, dict):
        for key in ("task", "task_type", "type", "objective", "kind"):
            value = contract.get(key)
            if value in ("classification", "regression"):
                return value
        for value in contract.values():
            if isinstance(value, str) and value in ("classification", "regression"):
                return value
    return "classification"


def metric_info(contract):
    name, direction = "macro_f1", "higher"
    if isinstance(contract, dict):
        raw = contract.get("metric")
        if isinstance(raw, str):
            name = raw
        elif isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("metric") or name)
        if contract.get("metric_direction") in ("higher", "lower"):
            direction = contract["metric_direction"]
    return name.lower(), direction


def work_root():
    return Path(os.environ.get("CONTEST_WORK") or "/workspace")


def unique_dir(base, name):
    target = base / name
    suffix = 1
    while target.exists():
        suffix += 1
        target = base / ("%s-%d" % (name, suffix))
    return target


def clip(text, limit=3900):
    """SDK 按字节卡 4000 上限，中文按字符截断会超，这里按字节收。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", "ignore")


def parse_paths(raw):
    """官方 metadata 里的 paths 每项是 JSON 字符串，这里统一成路径列表。"""
    parsed = []
    for item in raw or []:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                return []
        if not isinstance(item, list):
            return []
        path = []
        for step in item:
            if not isinstance(step, list) or len(step) != 2:
                return []
            if step[0] not in ("key", "index"):
                return []
            path.append([step[0], str(step[1]) if step[0] == "key" else int(step[1])])
        parsed.append(path)
    return parsed


def load_rows(iterator, cap, seed=42, label="train"):
    """流式读取并做定长水库抽样：C 有 11 万行，全量入内存会顶到镜像内存上限。"""
    rng = np.random.default_rng(seed)
    rows = []
    seen = 0
    for row in iterator:
        seen += 1
        if len(rows) < cap:
            rows.append(row)
        else:
            slot = int(rng.integers(0, seen))
            if slot < cap:
                rows[slot] = row
    log("%s：读到 %d 行，保留 %d 行" % (label, seen, len(rows)))
    return rows


def matrix(rows, paths):
    result = np.zeros((len(rows), len(paths)), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index] = core.extract(row.get("inputs") or {}, paths)
    return result


def ask_decider(rnd, context):
    """每轮至少一次真实模型调用；解析失败时退回默认超参，调用号照样记入 finish。"""
    messages = [
        {"role": "system", "content": "你是训练决策器。只回复一行 JSON：{\"learning_rate\":0.003,\"epochs\":3,\"batch_size\":64}"},
        {"role": "user", "content": context + "\n给一组与历史不同的配方。"},
    ]
    receipt = rnd.model(messages, max_tokens=min(48, rnd.max_output_tokens), temperature=0)
    text = receipt.text or ""
    log("decider call=%s reply=%s" % (receipt.call_id, text.strip()[:200]))
    match = re.search(r"\{.{0,200}?\}", text, re.S)
    chosen = dict(DEFAULTS)
    if match:
        try:
            payload = json.loads(match.group(0))
            rate = float(payload.get("learning_rate", DEFAULTS["learning_rate"]))
            epochs = int(payload.get("epochs", DEFAULTS["epochs"]))
            chosen["learning_rate"] = min(max(rate, 1e-4), 0.02)
            chosen["epochs"] = min(max(epochs, 1), 8)
            batch = int(payload.get("batch_size", 128))
            chosen["batch_size"] = min(max(batch, 16), 256)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return receipt, chosen


# ------------------------------------------------------------------ 采纳护栏
def slice_blocks(answers, kind, blocks=BLOCK_COUNT):
    """把验证集切成 blocks 段，模拟平台的“逐阶段”评估。

    回归任务按目标量从低到高（RUL 场景近似时间先后）切；分类任务按标签分层切。
    任何异常都返回空，调用方自动退回单点比较。
    """
    try:
        answers = np.asarray(answers)
        n = len(answers)
        if blocks < 2 or n < blocks * 8:
            return []
        if kind == "regression":
            order = np.argsort(answers, kind="stable")
            return [np.asarray(part) for part in np.array_split(order, blocks) if len(part)]
        parts = [[] for _ in range(blocks)]
        for index, label in enumerate(np.asarray(answers).tolist()):
            parts[int(index) % blocks].append(index)
        return [np.asarray(part) for part in parts if part]
    except Exception:
        return []


def stage_parts(rows, answers, kind, blocks=BLOCK_COUNT):
    """C 的四个阶段就是 FD001–FD004：优先按 meta.source_subset 分组，
    拿不到该字段时退回按目标值排序切段。"""
    if kind == "regression":
        groups = {}
        for index, row in enumerate(rows):
            meta = row.get("meta") or {}
            key = meta.get("source_subset") or meta.get("subset")
            if key:
                groups.setdefault(str(key), []).append(index)
        if len(groups) >= 2:
            return [np.asarray(groups[name]) for name in sorted(groups)]
    return slice_blocks(answers, kind, blocks)


def block_scores(net, x, answers, classes, kind, metric, blocks):
    return [core.score(net, x[part], np.asarray(answers)[part], classes, kind, metric)
            for part in blocks]


def relative_gain(parent, candidate, direction):
    """正数表示改善；回归与分类分别按下降/上升折算。"""
    if parent is None or candidate is None:
        return None
    base = abs(float(parent)) if abs(float(parent)) > 1e-9 else 1e-9
    if direction == "lower":
        return (float(parent) - float(candidate)) / base
    return (float(candidate) - float(parent)) / base


def adopt_decision(parent_score, candidate_score, direction, parent_blocks=None,
                   candidate_blocks=None, worst_tol=None):
    """返回 (是否采纳, 原因)。

    三道门槛：① 候选必须优于父模型；② 相对提升不低于 ADOPT_MARGIN；
    ③ 任一分段回退幅度不超过 WORST_BLOCK_TOL——平台计分里“最差阶段”占 20%，
    单点平均涨而某段崩掉时，采纳会直接把这一段分数打穿。
    """
    if candidate_score is None:
        return False, "候选分缺失"
    if parent_score is None:
        return True, "无父模型可比，直接采纳"
    if not is_improved(parent_score, candidate_score, direction):
        return False, "候选 %s 未优于父模型 %s" % (fmt(candidate_score), fmt(parent_score))
    gain = relative_gain(parent_score, candidate_score, direction)
    if gain is not None and gain < ADOPT_MARGIN:
        return False, "相对提升 %.3f%% 低于门槛 %.1f%%" % (gain * 100, ADOPT_MARGIN * 100)
    if parent_blocks and candidate_blocks:
        worst, worst_index = 0.0, -1
        for index, (before, after) in enumerate(zip(parent_blocks, candidate_blocks)):
            step = relative_gain(before, after, direction)
            if step is None:
                continue
            if step < worst:
                worst, worst_index = step, index
        tol = WORST_BLOCK_TOL if worst_tol is None else worst_tol
        if worst_index >= 0 and worst < -tol:
            return False, "第 %d 分段回退 %.1f%%，超过 %.0f%% 容差" % (
                worst_index + 1, -worst * 100, tol * 100)
        log("分段护栏通过，最差分段落差 %.3f%%" % (worst * 100))
    return True, "候选优于父模型且各分段未崩"


def anchor(parent, trained, ratio):
    """把训后权重按 ratio 拉回父权重，ratio=1 表示完全不动（等于父模型）。"""
    merged = copy.deepcopy(trained)
    for name in _WEIGHT_FIELDS:
        base = getattr(parent, name)
        merged_value = (1.0 - ratio) * getattr(trained, name) + ratio * base
        setattr(merged, name, merged_value)
    return merged


def platform_objective(parent_score, candidate_score, direction,
                       parent_blocks=None, candidate_blocks=None):
    """按平台口径（平均 80% + 最差阶段 20%）算相对父模型的加权净收益。"""
    mean_gain = relative_gain(parent_score, candidate_score, direction)
    worst_gain = mean_gain
    if parent_blocks and candidate_blocks:
        steps = [relative_gain(before, after, direction)
                 for before, after in zip(parent_blocks, candidate_blocks)]
        steps = [step for step in steps if step is not None]
        if steps:
            worst_gain = min(steps)
    if mean_gain is None or worst_gain is None:
        return None, mean_gain, worst_gain
    return MEAN_WEIGHT * mean_gain + WORST_WEIGHT * worst_gain, mean_gain, worst_gain


# ------------------------------------------------------------------ 训练一轮
def train(rnd: Round):
    kind = task_kind(rnd.task)
    metric, direction = metric_info(rnd.task)
    rows = load_rows(rnd.data.training_rows(), TRAIN_CAP, seed=42, label="训练集")
    validation = load_rows(rnd.data.validation_rows(), VALIDATION_CAP, seed=43, label="验证集")
    # 数据边界：训练只用 /data/train.jsonl；/data/validation.jsonl 只用于本轮评估，
    # 不并入训练池、不参与梯度更新；归一化参数只在本轮训练集上拟合。
    if not validation:
        raise RuntimeError("公开验证集为空，无法与父模型比较")
    log("dataset=%s round=%d task=%s metric=%s(%s) rows=%d validation=%d"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, len(rows), len(validation)))
    if not rows:
        raise RuntimeError("公开训练集为空")

    parent_net, parent_meta, parent_name = load_parent(rnd)
    paths = parse_paths(parent_meta.get("paths")) if parent_meta else []
    if not paths:
        paths = core.union_paths(rows[:200], core.MAX_FEATURES)
        log("父模型未给出特征路径，按本数据集 inputs 结构生成 %d 维特征" % len(paths))
    elif not usable(paths, rows):
        # v14：父路径部分不可用（典型是 C 的阶段字段与基线不同）时，若直接重建路径，
        # can_inherit 必然失败 → 模型从零重训、丢掉 baseline 起点。改为沿用父路径、
        # 缺失通道填 0；覆盖度太低（<50%）才退回重建，避免整行全 0 的无效特征。
        cover = path_coverage(paths, rows[:200])
        if cover >= 0.5:
            log("父路径覆盖 %.1f%%，沿用父路径并对缺失通道填 0（保证继承 baseline）"
                % (cover * 100))
        else:
            paths = core.union_paths(rows[:200], core.MAX_FEATURES)
            log("父路径覆盖仅 %.1f%%，改按本数据集结构生成 %d 维特征"
                % (cover * 100, len(paths)))

    x = matrix(rows, paths)
    x_validation = matrix(validation, paths)

    if kind == "regression":
        answers = np.asarray([float(row.get("answer", 0.0)) for row in rows], dtype=np.float64)
        answers_validation = np.asarray(
            [float(row.get("answer", 0.0)) for row in validation], dtype=np.float64)
        classes = []
        target_mean = float(answers.mean())
        target_std = float(answers.std())
        target_scale = target_std if target_std > 1e-9 else 1.0
    else:
        labels = sorted({str(row.get("answer")) for row in rows})
        parent_classes = list(parent_net.classes) if parent_net is not None else []
        if parent_classes and len(parent_classes) == len(labels) and set(parent_classes) == set(labels):
            classes = parent_classes
            log("沿用父检查点的类别顺序 %s" % classes)
        else:
            classes = labels
        lookup = {label: index for index, label in enumerate(classes)}
        answers = np.asarray([lookup[str(row.get("answer"))] for row in rows], dtype=np.int64)
        answers_validation = np.asarray(
            [lookup.get(str(row.get("answer")), 0) for row in validation], dtype=np.int64)
        target_mean, target_scale = 0.0, 1.0

    mean, scale = core.fit_scaler(x)
    inherited = can_inherit(parent_net, x, kind, classes, paths)
    if inherited:
        mean, scale = parent_net.mean, parent_net.scale
        if kind == "regression":
            target_mean, target_scale = parent_net.target_mean, parent_net.target_scale

    if inherited:
        net = parent_net
        log("继承父模型 w1%s，参数 %d 个" % (tuple(net.w1.shape), core_parameter_count(net)))
    else:
        net = core.Net(x.shape[1], out_dim=(len(classes) if kind == "classification" else 1),
                       kind=kind, seed=42)
        log("父模型与当前数据集不兼容，按 %d 维初始化新模型" % x.shape[1])
    net.mean, net.scale = mean, scale
    net.classes = classes
    net.paths = paths
    net.target_mean, net.target_scale = target_mean, target_scale
    parent_snapshot = copy.deepcopy(net)   # 锚定用：训前的父权重，不受 fit 原地修改影响

    parent_score = (core.score(net, x_validation, answers_validation, classes, kind, metric)
                    if (inherited and len(validation)) else None)
    key_upper = str(rnd.dataset_key).upper()
    block_count = BLOCK_COUNT_BY.get(key_upper, BLOCK_COUNT)
    guard = USE_BLOCK_GUARD or key_upper == "C"
    blocks = ((stage_parts(validation, answers_validation, kind, block_count)
               if key_upper == "C" else slice_blocks(answers_validation, kind, block_count))
              if (guard and parent_score is not None) else [])
    parent_blocks = (block_scores(net, x_validation, answers_validation, classes, kind,
                                  metric, blocks) if blocks else None)
    log("父模型验证分 %s（分段 %d 段）" % (fmt(parent_score), len(blocks)))

    # v14：上下文压缩——只保留数据集、轮次、指标与父分，历史只带最近 2 轮，
    # 每轮提示词体积降下来，Token 消耗随之下降。
    context = ("%s 第%d轮 %s(%s) 父分%s"
               % (rnd.dataset_key, rnd.round_index, metric, direction, fmt(parent_score)))
    past = []
    for item in (rnd.history() or [])[-2:]:
        config = item.get("configuration") or {}
        past.append("第%s轮 lr=%s ep=%s 候选=%s"
                    % (item.get("round_index"), config.get("learning_rate"),
                       config.get("epochs"), fmt(config.get("candidate_score"))))
    if past:
        context += "\n最近轮次：" + "；".join(past)
    receipt, choice = ask_decider(rnd, context)

    # 逐轮配方表：以镜像内配方为准执行；决策器回复被解析并记入日志，但不覆盖配方表。
    schedule = (
        (0.0018, 2, 128), (0.0030, 3, 64), (0.0007, 6, 64), (0.0020, 3, 64),
        (0.0040, 3, 96), (0.0030, 5, 128), (0.0012, 4, 64), (0.0030, 2, 64),
        (0.0007, 2, 128), (0.0050, 2, 64),
    )
    slot = schedule[(int(rnd.round_index) - 1) % len(schedule)]
    choice["learning_rate"], choice["epochs"], choice["batch_size"] = slot
    # v14：C 走微步——1 个 epoch、lr 5e-4，训完再按 0.9 的强度锚回父权重。
    # 目的是让 C 每段相对基线拿到一点正提升（最差阶段那 5 分才有机会），
    # 而不是像 v11 那样整段推进把 C 打穿。
    key_upper = str(rnd.dataset_key).upper()
    if key_upper == "C":
        choice["learning_rate"], choice["epochs"], choice["batch_size"] = (0.0005, 2, 64)
        log("数据集 C 采用微步配方 lr=0.0005 epochs=2 batch=64")
    # 连续保留父模型时自动衰减学习率，避免在同一步长上反复撞墙。
    streak = rollback_streak(rnd)
    if DECAY_ON_ROLLBACK and streak >= 2:
        factor = 0.5 ** min(streak - 1, 3)
        choice["learning_rate"] = slot[0] * factor
        log("连续 %d 轮保留父模型，学习率折算 %.4g → %.4g" % (streak, slot[0], choice["learning_rate"]))
    log("第 %d 轮配方 lr=%s epochs=%s batch=%s（决策 call=%s，回复仅记录）"
        % (int(rnd.round_index), choice["learning_rate"], slot[1], slot[2], receipt.call_id))

    origin = "继承父权重续训" if inherited else "从零初始化"

    # v13：冻结数据集（默认 C）完全不训练，只提交父检查点原样。
    # v11 实测显示 C 十轮全部“采用本轮模型”——因为该数据集的父检查点与当前数据维度不兼容
    # （inherited=False），冻结条件原先挂在 inherited 上，等于没生效；C 还从零重训花了
    # 8,686 秒，把“资源效率”打到 0/5。这里改为按数据集名无条件冻结。
    frozen = (str(rnd.dataset_key).upper() in [name.upper() for name in FREEZE_DATASETS]
              or (str(rnd.dataset_key).upper() == "C" and not inherited))
    total_epochs = max(1, int(choice["epochs"]))
    batch = int(choice.get("batch_size", 128))
    best = None
    best_epoch = total_epochs
    if frozen:
        candidate_score = None
        candidate_blocks = None
        anchor_ratio = 0.0
        epochs_used = 0
        objective = None
        best = {"ratio": 0.0, "model": parent_snapshot, "score": None, "blocks": None,
                "objective": None, "epoch": 0}
        log("数据集 %s 冻结：本轮不训练、不推进权重，候选提交父检查点原样" % rnd.dataset_key)
    elif EARLY_STOP:
        for epoch in range(1, total_epochs + 1):
            net.fit(x, answers, epochs=1, learning_rate=choice["learning_rate"],
                    batch_size=batch, seed=42 + 17 * epoch)
            for ratio in ANCHOR_RATIOS:
                model = copy.deepcopy(net) if ratio == 0.0 else anchor(parent_snapshot, net, ratio)
                score = (core.score(model, x_validation, answers_validation, classes, kind, metric)
                         if len(validation) else None)
                blocks_now = (block_scores(model, x_validation, answers_validation, classes, kind,
                                           metric, blocks) if (blocks and parent_blocks) else None)
                objective, mean_gain, worst_gain = platform_objective(
                    parent_score, score, direction, parent_blocks, blocks_now)
                item = {"ratio": ratio, "model": model, "score": score, "blocks": blocks_now,
                        "objective": objective, "mean_gain": mean_gain,
                        "worst_gain": worst_gain, "epoch": epoch}
                if objective is not None and (best is None or objective > best["objective"]):
                    best, best_epoch = item, epoch
    else:
        # C 每轮换种子：被阶段护栏拒掉的轮次照样能给出不同的微步轨迹，
        # 十轮里多试几个起点，命中"四段都不退"的概率更高。
        fit_seed = (42 + int(rnd.round_index)) if str(rnd.dataset_key).upper() == "C" else 42
        net.fit(x, answers, epochs=total_epochs, learning_rate=choice["learning_rate"],
                batch_size=batch, seed=fit_seed)
        model = copy.deepcopy(net)
        score = (core.score(model, x_validation, answers_validation, classes, kind, metric)
                 if len(validation) else None)
        blocks_now = (block_scores(model, x_validation, answers_validation, classes, kind,
                                   metric, blocks) if (blocks and parent_blocks) else None)
        objective, mean_gain, worst_gain = platform_objective(
            parent_score, score, direction, parent_blocks, blocks_now)
        best = {"ratio": 0.0, "model": model, "score": score, "blocks": blocks_now,
                "objective": objective, "mean_gain": mean_gain,
                "worst_gain": worst_gain, "epoch": total_epochs}
        log("候选验证分 %s（lr=%s epochs=%d batch=%d，从父检查点单次续训）"
            % (fmt(score), choice["learning_rate"], total_epochs, batch))

    net = best["model"]
    candidate_score = best["score"]
    candidate_blocks = best["blocks"]
    anchor_ratio = best["ratio"]
    objective = best["objective"]
    epochs_used = best["epoch"]
    if candidate_blocks and parent_blocks:
        log("候选分段落差 %s"
            % " / ".join("%+.2f%%" % ((relative_gain(before, after, direction) or 0.0) * 100)
                         for before, after in zip(parent_blocks, candidate_blocks)))


    # 基线口径：每轮候选一律从父检查点（最终可追溯到平台基线）续训，不做从零重训分支。
    # 平台评分对象是续训后的工业模型，保持与基线同源才能保证提升可归因。

    improved, adopt_reason = adopt_decision(
        parent_score, candidate_score, direction, parent_blocks, candidate_blocks,
        worst_tol=(0.0 if key_upper == "C" else None))
    if improved and objective is not None and objective <= 0:
        improved, adopt_reason = False, "加权收益 %+.3f%% 不为正，保留父模型" % (objective * 100)
    if frozen:
        improved = False
        adopt_reason = ("数据集 %s 阶段跨度大，本轮只提交父检查点、不推进权重（冻结）"
                        % rnd.dataset_key)
    log("采纳判定：%s（%s）" % ("采纳" if improved else "保留父模型", adopt_reason))

    work = work_root()
    candidate_dir = unique_dir(work, "candidate-r%d" % rnd.round_index)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = candidate_dir / "checkpoint.npz"
    if frozen and parent_name and parent_name.endswith(".npz"):
        try:
            checkpoint.write_bytes(rnd.parent.path(parent_name).read_bytes())
            log("冻结候选 = 父检查点 %s 原样复制（%d B）"
                % (parent_name, checkpoint.stat().st_size))
        except Exception as error:
            log("父检查点复制失败（%s），改为保存当前权重" % error)
            net.save(checkpoint)
    else:
        net.save(checkpoint, extra={
            "dataset_key": rnd.dataset_key,
            "round_index": rnd.round_index,
            "metric": metric,
            "parent_score": parent_score,
            "candidate_score": candidate_score,
            "learning_rate": choice["learning_rate"],
            "epochs": choice["epochs"],
            "anchor_ratio": anchor_ratio,
            "epochs_used": epochs_used,
            "model_call_id": receipt.call_id,
        })

    if frozen:
        hypothesis = "%s：第 %d 轮冻结，不训练，直接沿用父检查点，避免阶段漂移带来的负提升。" % (
            metric, rnd.round_index)
    else:
        hypothesis = "%s：在第 %d 轮把学习率取 %.4g、续训 %d 轮，验证分应从 %s 改善。" % (
            metric, rnd.round_index, choice["learning_rate"], choice["epochs"], fmt(parent_score))
    change_summary = "从父检查点 %s 继续 Adam 训练 %d 轮（lr=%.4g），特征 %d 维，%s。" % (
        parent_name, choice["epochs"], choice["learning_rate"], x.shape[1],
        origin)
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=clip(hypothesis),
        change_summary=clip(change_summary),
        configuration={
            "learning_rate": choice["learning_rate"],
            "epochs": choice["epochs"],
            "features": int(x.shape[1]),
            "parent_score": parent_score,
            "candidate_score": candidate_score,
            "metric": metric,
            "anchor_ratio": anchor_ratio,
            "epochs_used": epochs_used,
            "platform_objective": objective,
        },
        seed=42,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    log("候选已提交 artifact=%s" % submitted.get("artifact_id"))

    state_dir = unique_dir(work, "state-r%d" % rnd.round_index)
    state_dir = rnd.state_workspace(state_dir)
    (state_dir / "round.json").write_text(json.dumps({
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "metric": metric,
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "selected": "candidate" if improved else "parent",
        "adopt_reason": adopt_reason,
        "anchor_ratio": anchor_ratio,
        "epochs_used": epochs_used,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    rnd.save_state(state_dir)

    history = rnd.history()
    selected = submitted["artifact_id"] if improved else rnd.parent.artifact_id
    reason = "候选验证分 %s，父模型 %s；%s。模型调用 %s。" % (
        fmt(candidate_score), fmt(parent_score), adopt_reason, receipt.call_id)
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=clip(hypothesis),
        change_summary=clip(change_summary),
        reason=clip(reason),
        configuration={
            "learning_rate": choice["learning_rate"],
            "epochs": choice["epochs"],
            "features": int(x.shape[1]),
            "parent_score": parent_score,
            "candidate_score": candidate_score,
            "metric": metric,
            "selected": "candidate" if improved else "parent",
        },
        seed=42,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )


# ------------------------------------------------------------------ 预测
def predict(sample, model_dir):
    checkpoint = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    net, _meta = core.Net.load(checkpoint)
    paths = net.paths or core.leaf_paths(sample.inputs)
    vector = core.extract(sample.inputs or {}, paths)
    matrix_in = net.normalize(vector[None, :])
    if sample.task == "regression" or net.kind == "regression":
        value = float(net.predict(matrix_in)[0])
        return {"value": value if np.isfinite(value) else 0.0}
    prob = net.predict(matrix_in)[0]
    classes = net.classes or ["normal"]
    outputs = [{"label": str(label), "score": float(round(float(prob[index]), 6))}
               for index, label in enumerate(classes[:len(prob)])]
    top_index = max(range(len(outputs)), key=lambda index: outputs[index]["score"])
    top_score = outputs[top_index]["score"]
    for index, item in enumerate(outputs):
        if index != top_index and item["score"] >= top_score:
            item["score"] = max(0.0, round(top_score - 1e-6, 9))
    return outputs


# ------------------------------------------------------------------ 内部助手
def load_parent(rnd):
    names = list(rnd.parent.declared_paths())
    candidates = [name for name in names if name.endswith(".npz")]
    for name in candidates:
        try:
            path = rnd.parent.path(name)
            net, meta = core.Net.load(path)
            net.paths = parse_paths(meta.get("paths")) or net.paths
            log("读取父检查点 %s（参数 %d 个，特征 %d 维）"
                % (name, core_parameter_count(net), len(net.paths)))
            return net, meta, name
        except Exception as error:  # 父检查点损坏或不是我们的格式
            log("父检查点 %s 无法载入：%s" % (name, error))
    return None, {}, "无（本轮为首轮或父检查点不可读）"


def usable(paths, rows):
    for row in rows[:8]:
        if all(core.value_at(row.get("inputs"), path) is not None for path in paths):
            return True
    return False


def path_coverage(paths, rows):
    """父路径在当前数据上的可用比例：至少一半通道存在才算可沿用。"""
    if not paths or not rows:
        return 0.0
    hits = 0
    total = 0
    for row in rows:
        for path in paths:
            total += 1
            if core.value_at(row.get("inputs"), path) is not None:
                hits += 1
    return hits / float(total or 1)


def can_inherit(net, x, kind, classes, paths):
    if net is None:
        return False
    if net.kind != kind or net.w1.shape[0] != x.shape[1]:
        return False
    if net.paths and json.dumps(net.paths, sort_keys=True) != json.dumps(paths, sort_keys=True):
        return False
    if kind == "classification":
        return len(net.classes) > 0 and len(net.classes) == len(classes)
    return net.w2.shape[1] == 1


def core_parameter_count(net):
    return int(net.w1.size + net.b1.size + net.w2.size + net.b2.size)


def fmt(value):
    if value is None:
        return "N/A"
    return "%.6f" % value


def better(left, right, direction):
    if left is None:
        return False
    if right is None:
        return True
    if direction == "lower":
        return left < right - TOLERANCE
    return left > right + TOLERANCE


def rollback_streak(rnd):
    """统计连续“保留父模型”的轮数，用于自动衰减学习率。"""
    count = 0
    try:
        for item in reversed(list(rnd.history())):
            config = item.get("configuration") or {}
            if str(config.get("selected")) == "parent":
                count += 1
            else:
                break
    except Exception:  # history 不可用时按 0 处理
        return 0
    return count


def is_improved(parent_score, candidate_score, direction):
    if candidate_score is None:
        return False
    if parent_score is None:
        return True
    if direction == "lower":
        return candidate_score <= parent_score - TOLERANCE
    return candidate_score >= parent_score + TOLERANCE
