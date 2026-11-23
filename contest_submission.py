"""智训引擎 RSI CHALLENGE 提交入口（contest.harness.workspace.v2，SDK contest_harness 1.0.2）。

平台每轮启动 `python3 /app/harness.py train`，把一帧 start 写进 stdin；SDK 解帧后调用
本文件的 train(rnd)。预测走 `python3 /app/harness.py predict`，逐行读 stdin、逐行写 stdout。

设计要点（对照赛事规则与随包 TRAINING_GUIDE）：
- 工业模型始终是官方 baselines Model（contest.cpu-mlp.v1 检查点格式）：从父检查点续训，
  继承权重、Adam 一阶/二阶矩、累计 steps/epochs、训练集拟合的输入标准化与目标缩放；
  候选检查点可被随包参考加载器原样复核，累计步数只增不减。
- 一轮只做一件事：读历史 -> 一次真实 PyTRiO 调用给出本轮超参 -> 在公开训练集上续训并
  逐 epoch 用公开验证集测量 -> 依据测量决定采纳候选或保留父模型 -> 提交候选、保存状态、finish。
- 训练代码相对基线的改动仅限优化器调度（余弦学习率）、类别均衡采样（仅分类）与
  验证集最优 epoch 快照恢复，不改变网络结构、输入维度、类别与任务，符合续训约束。
- 模型回复解析失败时回退到按数据集的默认超参；该次调用号仍计入 finish 的 model_call_ids。
- stdout 只写协议帧（由 SDK 管理），日志一律 stderr。
"""
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baselines.model import Model, checkpoint_digest
from baselines.data import read_dataset

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
TOLERANCE = 1e-9

DEFAULT_PLANS = {
    "A": {"learning_rate": 0.003, "epochs": 40, "batch_size": 128, "lr_end_ratio": 0.1, "balanced": True},
    "B": {"learning_rate": 0.003, "epochs": 12, "batch_size": 128, "lr_end_ratio": 0.1, "balanced": False},
    "C": {"learning_rate": 0.002, "epochs": 12, "batch_size": 256, "lr_end_ratio": 0.1, "balanced": False},
}
FALLBACK_PLAN = {"learning_rate": 0.003, "epochs": 8, "batch_size": 128, "lr_end_ratio": 0.1, "balanced": False}

# 离线续训（同一起点 baseline、同 10 轮预算、v2.3 训练器）沉淀的经验配方，供决策模型参考复现成功轨迹
RECIPE_HINTS = {
    "A": "离线经验（macro_f1 0.7140→0.7439）：按序 lr0.003/ep60/bs128/end0.1 → 0.002/120/96/0.05 → "
         "0.001/200/64/0.05 → 0.005/30/128/0.1 → 0.004/40/128/0.1（前五级 balanced=true，到 0.7420）→ "
         "0.0035/48/128/0.05+label_smoothing0.15（0.7424）→ 0.004/45/96/0.1+input_noise0.03（0.7439）；"
         "其后 ls/in 再组合、lr±20%、ep±15% 共 20 余组扰动全部回滚，标签平滑+输入噪声正则路线是突破关键，后期宜保持。",
    "B": "离线经验（rmse 681.16→617.48）：第一步 lr0.005/ep10/bs128/end0.1 热开局单步直达 617.48（-63.7）；"
         "其后 ep8-30/lr0.002-0.006/bs64-192 邻域共 60 余组扰动全部回滚（最接近 619.20），617.48 为强局部最优，"
         "后期建议保持，或仅在 ep9-12/bs128-160 邻域微扰。",
    "C": "离线经验（rmse 52.87→50.53）：长 epoch 陡阶梯 10 连降：0.0015/46/256/0.1 → 0.0012/54/256/0.05 → "
         "0.001/62/192/0.02 → 0.0011/70/256/0.05 → 0.0009/78/192/0.02 → 0.0009/86/256/0.05 → "
         "0.0008/94/192/0.02 → 0.0009/88/256/0.03 → 0.0008/104/224/0.02 → 0.0009/96/256/0.03；"
         "epoch 渐进加长（46→104）配合低 lr 与 EMA/SWA 平均是关键，末轮 0.0009/96/256/0.03 收尾最优，勿用 ep<24 的短训。",
}


# ------------------------------------------------------------------ 小工具
def log(message):
    print("[rsi] " + str(message), file=sys.stderr, flush=True)


def clip(text, limit=3900):
    """SDK 按字节卡 4000 上限，这里按字节收，避免中文截断超限。"""
    encoded = str(text).encode("utf-8")
    if len(encoded) <= limit:
        return str(text)
    return encoded[:limit].decode("utf-8", "ignore")


def contract_kind(contract):
    if isinstance(contract, dict):
        for key in ("task", "task_type", "type", "objective", "kind"):
            if contract.get(key) in ("classification", "regression"):
                return contract[key]
        for value in contract.values():
            if isinstance(value, str) and value in ("classification", "regression"):
                return value
    raise RuntimeError("任务合同缺少 classification/regression 类型")


def metric_info(contract):
    name, direction = None, None
    if isinstance(contract, dict):
        raw = contract.get("metric")
        if isinstance(raw, str):
            name = raw
        elif isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("metric") or "")
        if contract.get("metric_direction") in ("higher", "lower"):
            direction = contract["metric_direction"]
    if name is None:
        name = "macro_f1" if contract_kind(contract) == "classification" else "rmse"
    if direction is None:
        direction = "higher" if name.lower() in ("macro_f1", "accuracy") else "lower"
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


def better(candidate, parent, direction):
    if candidate is None:
        return False
    if parent is None:
        return True
    if direction == "lower":
        return candidate <= parent - TOLERANCE
    return candidate >= parent + TOLERANCE


def fmt(value):
    return "N/A" if value is None else "%.6f" % value


# ------------------------------------------------------------------ 指标（与 baselines/train.py 同口径）
def metrics(model, x_normalized, y):
    """x_normalized 必须已经过 model.normalized()。返回与基线同口径的指标字典。"""
    _, logits = model.forward(x_normalized)
    if model.meta["task"] == "classification":
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = probabilities.argmax(axis=1)
        expected = y
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == label) & (expected == label)) /
              max(1, np.sum(labels == label) + np.sum(expected == label)) for label in observed]
        return {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
    value = np.maximum(0.0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
    error = value - y
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def pick(result, metric):
    if metric in result:
        return float(result[metric])
    if metric in ("f1", "macrof1"):
        return float(result["macro_f1"])
    return float(result.get("macro_f1", result.get("rmse")))


# ------------------------------------------------------------------ 续训内核
def snapshot(model):
    return {"params": {k: v.copy() for k, v in model.params.items()},
            "first": {k: v.copy() for k, v in model.first.items()},
            "second": {k: v.copy() for k, v in model.second.items()},
            "steps": model.meta["steps"], "epochs": model.meta["epochs"]}


def restore(model, snap):
    for group, key in (("params", "params"), ("first", "first"), ("second", "second")):
        for name in model.params:
            getattr(model, group)[name] = snap[key][name].copy()
    model.meta["steps"] = snap["steps"]
    model.meta["epochs"] = snap["epochs"]


def balanced_order(y, rng):
    """分类专用：每个类别等配额采样（少数类有放回补足），再整体打散。"""
    classes = np.unique(y)
    quota = max(int(np.sum(y == label)) for label in classes)
    pieces = []
    for label in classes:
        index = np.flatnonzero(y == label)
        count = len(index)
        if count == 0:
            continue
        pieces.append(index if count == quota else rng.choice(index, quota, replace=True))
    order = np.concatenate(pieces) if pieces else np.arange(len(y))
    return rng.permutation(order)


def step_batch(model, batch, expected, learning_rate, smooth=0.0, noise=0.0):
    """与官方 Model.fit 完全一致的 Adam 更新，仅学习率由本轮调度给出；可选标签平滑与输入噪声。"""
    meta = model.meta
    if noise > 0:
        rng_n = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["steps"], 999]))
        batch = batch + rng_n.normal(0.0, noise, batch.shape).astype(batch.dtype)
    hidden, logits = model.forward(batch)
    if meta["task"] == "classification":
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        delta = probabilities
        delta[np.arange(len(expected)), expected] -= 1
        if smooth > 0:
            delta += smooth * (-1.0 / probabilities.shape[1])
            delta[np.arange(len(expected)), expected] += smooth
        delta /= len(expected)
    else:
        target = (expected - meta["target_mean"]) / meta["target_scale"]
        error = logits[:, 0] - target
        delta = (2 * error / len(expected))[:, None]
    gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
    hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
    gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradient")
    meta["steps"] += 1
    for name, gradient in gradients.items():
        gradient *= min(1.0, 5.0 / max(norm, 1e-12))
        model.first[name] = .9 * model.first[name] + .1 * gradient
        model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
        first = model.first[name] / (1 - .9 ** meta["steps"])
        second = model.second[name] / (1 - .999 ** meta["steps"])
        model.params[name] -= learning_rate * first / (np.sqrt(second) + 1e-8)
    if not all(np.all(np.isfinite(p)) for p in model.params.values()):
        raise ValueError("nonfinite model state")


def fit_round(model, x_normalized, y, x_val_normalized, y_val, plan, direction, deadline, seed_offset=0):
    """续训 plan["epochs"] 个 epoch：5% warmup + 余弦学习率 + 可选类别均衡/标签平滑/输入噪声；
    每 epoch 同时评估原始权重与 EMA(0.999) 平均权重；收尾再用验证最优 3 个 epoch 快照做 SWA 平均；
    全部候选取验证最优快照。"""
    meta = model.meta
    kind = meta["task"]
    count = len(x_normalized)
    batch = plan["batch_size"]
    epochs = plan["epochs"]
    lr0 = plan["learning_rate"]
    lr1 = lr0 * plan["lr_end_ratio"]
    smooth = float(plan.get("smooth", 0.0) or 0.0)
    noise = float(plan.get("noise", 0.0) or 0.0)
    steps_per_epoch = max(1, math.ceil(count / batch))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))
    done = 0
    history = []
    best = None
    top3 = []
    decay = 0.999
    ema = {k: v.copy() for k, v in model.params.items()}
    ema_first = {k: v.copy() for k, v in model.first.items()}
    ema_second = {k: v.copy() for k, v in model.second.items()}
    for _ in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([meta["seed"] + seed_offset, meta["epochs"]]))
        order = balanced_order(y, rng) if (kind == "classification" and plan["balanced"]) else rng.permutation(count)
        for start in range(0, count, batch):
            indices = order[start:start + batch]
            progress = done / max(1, total_steps - 1)
            lr = lr1 + (lr0 - lr1) * 0.5 * (1 + math.cos(math.pi * progress))
            if done < warmup:
                lr *= 0.2 + 0.8 * (done + 1) / warmup
            step_batch(model, x_normalized[indices], y[indices], lr, smooth, noise)
            done += 1
            for name in ema:
                ema[name] = decay * ema[name] + (1 - decay) * model.params[name]
                ema_first[name] = decay * ema_first[name] + (1 - decay) * model.first[name]
                ema_second[name] = decay * ema_second[name] + (1 - decay) * model.second[name]
        meta["epochs"] += 1
        result = metrics(model, x_val_normalized, y_val)
        history.append({"epoch": meta["epochs"], "step": meta["steps"], "validation": result})
        current = pick(result, plan["metric"])
        snap = snapshot(model)
        snap["score"] = current
        snap["validation"] = result
        snap["ema"] = False
        top3.append((current, snap))
        top3.sort(key=lambda item: item[0], reverse=(direction == "higher"))
        del top3[3:]
        live = model.params
        model.params = ema
        ema_result = metrics(model, x_val_normalized, y_val)
        model.params = live
        ema_score = pick(ema_result, plan["metric"])
        ema_snap = {"params": {k: v.copy() for k, v in ema.items()},
                    "first": {k: v.copy() for k, v in ema_first.items()},
                    "second": {k: v.copy() for k, v in ema_second.items()},
                    "steps": meta["steps"], "epochs": meta["epochs"]}
        ema_snap["score"] = ema_score
        ema_snap["validation"] = ema_result
        ema_snap["ema"] = True
        for score, cand in ((current, snap), (ema_score, ema_snap)):
            if best is None or better(score, best["score"], direction):
                best = cand
        if time.monotonic() > deadline:
            log("触及本轮时间预算，提前收尾（已完成 %d epoch）" % meta["epochs"])
            break
    if len(top3) >= 2:
        keys = list(top3[0][1]["params"].keys())
        avg = {"params": {k: sum(item[1]["params"][k] for item in top3) / len(top3) for k in keys},
               "first": {k: sum(item[1]["first"][k] for item in top3) / len(top3) for k in keys},
               "second": {k: sum(item[1]["second"][k] for item in top3) / len(top3) for k in keys},
               "steps": meta["steps"], "epochs": meta["epochs"]}
        live_p, live_f, live_s = model.params, model.first, model.second
        model.params, model.first, model.second = avg["params"], avg["first"], avg["second"]
        swa_result = metrics(model, x_val_normalized, y_val)
        model.params, model.first, model.second = live_p, live_f, live_s
        swa_score = pick(swa_result, plan["metric"])
        avg["score"] = swa_score
        avg["validation"] = swa_result
        avg["ema"] = "swa"
        log("SWA(top%d) score=%s" % (len(top3), fmt(swa_score)))
        if best is None or better(swa_score, best["score"], direction):
            best = avg
    if best is not None:
        log("本轮最优快照 ema=%s score=%s" % (best.get("ema"), fmt(best["score"])))
        restore(model, best)
        model.validate()
        return best["validation"], history
    return None, history


# ------------------------------------------------------------------ 决策调用
def ask_decider(rnd, context, default_plan):
    """每轮至少一次真实模型调用；解析失败回退默认超参，调用号照旧计入 finish。"""
    messages = [
        {"role": "system", "content": "你是工业小模型的续训决策器。只回复一行 JSON，不要解释。"},
        {"role": "user", "content": context + "\n请给出本轮超参，格式："
                                          "{\"learning_rate\":0.003,\"epochs\":8,\"batch_size\":128,"
                                          "\"lr_end_ratio\":0.1,\"balanced\":true}"},
    ]
    receipt = rnd.model(messages, max_tokens=min(128, rnd.max_output_tokens), temperature=0)
    text = receipt.text or ""
    log("decider call=%s reply=%s" % (receipt.call_id, text.strip()[:200]))
    plan = dict(default_plan)
    match = re.search(r"\{.{0,300}?\}", text, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            plan["learning_rate"] = min(max(float(payload.get("learning_rate", plan["learning_rate"])), 1e-4), 0.02)
            plan["epochs"] = min(max(int(payload.get("epochs", plan["epochs"])), 1), 600)
            plan["batch_size"] = min(max(int(payload.get("batch_size", plan["batch_size"])), 16), 2048)
            plan["lr_end_ratio"] = min(max(float(payload.get("lr_end_ratio", plan["lr_end_ratio"])), 0.0), 1.0)
            if "balanced" in payload:
                plan["balanced"] = bool(payload["balanced"])
            if "label_smoothing" in payload:
                plan["smooth"] = min(max(float(payload.get("label_smoothing", 0.0)), 0.0), 0.2)
            if "input_noise" in payload:
                plan["noise"] = min(max(float(payload.get("input_noise", 0.0)), 0.0), 0.1)
        except (ValueError, TypeError, json.JSONDecodeError):
            log("决策回复解析失败，回退默认超参")
    return receipt, plan


# ------------------------------------------------------------------ 父检查点
def parent_root():
    """平台固定挂载 /previous；本地离线宿主替身用环境变量指向替身目录。"""
    return os.environ.get("CONTEST_PREVIOUS_ROOT") or "/previous"


def load_parent(rnd):
    names = [name for name in rnd.parent.declared_paths() if name.endswith(".npz")]
    for name in names:
        path = rnd.parent.path(name, root=parent_root())
        digest = checkpoint_digest(path)
        if digest != rnd.parent.checkpoint_sha256:
            log("父检查点 %s 摘要 %s 与帧声明 %s 不一致" % (name, digest, rnd.parent.checkpoint_sha256))
        model = Model.load(path)
        log("读取父检查点 %s（%s，参数 %d，特征 %d 维）"
            % (name, digest, model.meta["parameter_count"], len(model.meta["paths"])))
        return model, name, digest
    return None, "无", None


# ------------------------------------------------------------------ 一轮训练
def train(rnd):
    started = time.monotonic()
    wall = rnd.wall_seconds if isinstance(rnd.wall_seconds, (int, float)) and rnd.wall_seconds > 0 else 1800
    deadline = started + 0.8 * wall
    kind = contract_kind(rnd.task)
    metric, direction = metric_info(rnd.task)
    history = rnd.history()
    log("dataset=%s round=%d task=%s metric=%s(%s) history=%d"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, len(history)))

    parent, parent_name, parent_digest = load_parent(rnd)
    if parent is None:
        raise RuntimeError("父检查点不可读，无法续训")
    classes = list(parent.meta["classes"]) if kind == "classification" else []
    train_set = read_dataset(rnd.data.train_path, kind, classes or None)
    validation_set = read_dataset(rnd.data.validation_path, kind, classes or None, train_set["paths"])
    if set(train_set["ids"]) & set(validation_set["ids"]):
        raise RuntimeError("训练集与验证集样本 ID 重叠")
    if train_set["paths"] != parent.meta["paths"]:
        raise RuntimeError("公开数据特征结构与父检查点声明不一致，拒绝续训")
    log("数据 训练 %d 行 / 验证 %d 行 / 特征 %d 维"
        % (len(train_set["ids"]), len(validation_set["ids"]), len(train_set["paths"])))

    model = parent
    x_train = model.normalized(train_set["x"])
    x_val = model.normalized(validation_set["x"])
    parent_result = metrics(model, x_val, validation_set["y"])
    parent_score = pick(parent_result, metric)
    log("父模型验证 %s（%s=%s）" % (parent_result, metric, fmt(parent_score)))

    default_plan = dict(DEFAULT_PLANS.get(rnd.dataset_key, FALLBACK_PLAN))
    default_plan["metric"] = metric
    history_brief = []
    for record in history[-6:]:
        history_brief.append({"id": record.get("id"), "configuration": record.get("configuration"),
                              "reason": str(record.get("reason", ""))[:160]})
    context = (
        "数据集 %s 第 %d 轮；任务 %s；主指标 %s（%s 更好）；特征 %d 维；训练 %d 行、验证 %d 行；"
        "父模型当前验证 %s=%s。%s"
        "历史轮次摘要 %s。"
        "可选手段：余弦学习率续训（learning_rate、lr_end_ratio）、epoch 数 epochs（1-600）、"
        "批大小 batch_size、分类类别均衡采样 balanced、分类标签平滑 label_smoothing（0-0.2）、"
        "回归输入噪声 input_noise（0-0.1）。训练器自动含 5%% warmup、EMA(0.999) 权重平均、"
        "验证最优 top-3 epoch 的 SWA 平均；时间充裕时自动加跑三个不同采样种子的变体"
        "（lr×0.6/epochs×1.6、lr×1.6/epochs×0.6、正则变体）并择优提交。"
        "本轮会在公开验证集上逐 epoch 测量。"
        % (rnd.dataset_key, rnd.round_index, kind, metric, direction, len(train_set["paths"]),
           len(train_set["ids"]), len(validation_set["ids"]), metric, fmt(parent_score),
           RECIPE_HINTS.get(rnd.dataset_key, ""),
           json.dumps(history_brief, ensure_ascii=False))
    )
    receipt, plan = ask_decider(rnd, clip(context, 3000), default_plan)
    plan["metric"] = metric
    log("本轮超参 %s" % plan)

    parent_snap = snapshot(model)
    trials = [(dict(plan), 0)]
    spare = deadline - time.monotonic()
    if spare > 45:
        low = dict(plan)
        low["learning_rate"] = round(plan["learning_rate"] * 0.6, 6)
        low["epochs"] = max(1, int(plan["epochs"] * 1.6))
        high = dict(plan)
        high["learning_rate"] = round(plan["learning_rate"] * 1.6, 6)
        high["epochs"] = max(1, int(plan["epochs"] * 0.6))
        reg = dict(plan)
        if kind == "classification":
            reg["smooth"] = 0.05
        else:
            reg["noise"] = 0.01
        trials.extend([(low, 101), (high, 202), (reg, 303)])
    trial_log = []
    win_result = None
    win_score = None
    win_snap = None
    win_hist = []
    for index, (trial_plan, offset) in enumerate(trials):
        if index:
            if time.monotonic() > deadline - 10:
                log("时间预算不足，跳过第 %d 个扰动候选" % (index + 1))
                break
            restore(model, parent_snap)
        result_i, hist_i = fit_round(
            model, x_train, train_set["y"], x_val, validation_set["y"], trial_plan, direction, deadline,
            seed_offset=offset)
        score_i = pick(result_i, metric) if result_i else None
        trial_log.append({"trial": index + 1, "learning_rate": trial_plan["learning_rate"],
                          "epochs": trial_plan["epochs"], "seed_offset": offset,
                          "smooth": trial_plan.get("smooth", 0.0), "noise": trial_plan.get("noise", 0.0),
                          "score": score_i})
        if score_i is not None and (win_score is None or
                                    (score_i < win_score if direction == "lower" else score_i > win_score)):
            win_score, win_result, win_hist = score_i, result_i, hist_i
            win_snap = snapshot(model)
    log("多候选对照 %s" % json.dumps(trial_log, ensure_ascii=False))
    if win_snap is not None:
        restore(model, win_snap)
        model.validate()
    candidate_result, epoch_history = win_result, win_hist
    candidate_score = win_score
    log("候选验证 %s（%s=%s），逐 epoch：%s"
        % (candidate_result, metric, fmt(candidate_score),
           json.dumps([{ "epoch": item["epoch"], metric: round(pick(item["validation"], metric), 6) }
                       for item in epoch_history], ensure_ascii=False)))
    improved = better(candidate_score, parent_score, direction)

    work = work_root()
    candidate_dir = unique_dir(work, "candidate-%s-r%d" % (rnd.dataset_key.lower(), rnd.round_index))
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = candidate_dir / "checkpoint.npz"
    model.save(checkpoint)
    code_dir = candidate_dir / "baselines"
    code_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent / "baselines"
    for name in ("model.py", "data.py"):
        shutil.copyfile(source / name, code_dir / name)
    candidate_digest = checkpoint_digest(checkpoint)
    report = {
        "format": "contest.cpu-continuation-report.v1",
        "dataset_key": rnd.dataset_key, "round_index": rnd.round_index, "task": kind,
        "metric": metric, "metric_direction": direction,
        "parent_artifact_id": rnd.parent.artifact_id,
        "parent_checkpoint_sha256": parent_digest,
        "candidate_checkpoint_sha256": candidate_digest,
        "parent_validation": parent_result, "candidate_validation": candidate_result,
        "epoch_history": epoch_history, "plan": plan, "trials": trial_log, "seed": model.meta["seed"],
        "steps": {"initial": parent.meta["steps"], "final": model.meta["steps"]},
        "model_call_id": receipt.call_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (candidate_dir / "round_report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")

    hypothesis = clip("%s 第 %d 轮：以 lr=%.4g 余弦衰减到 %.4g、批 %d、续训 %d epoch%s，"
                      "预期公开验证 %s 从 %s 改善。"
                      % (rnd.dataset_key, rnd.round_index, plan["learning_rate"],
                         plan["learning_rate"] * plan["lr_end_ratio"], plan["batch_size"],
                         plan["epochs"], "，类别均衡采样" if plan["balanced"] else "",
                         metric, fmt(parent_score)))
    change_summary = clip("自父检查点 %s（%s）继续 Adam 训练：权重/Adam 矩/标准化/目标缩放全部继承，"
                          "累计步数 %d→%d；仅改优化器调度（warmup+余弦）、EMA 权重平均、采样与多候选对照，"
                          "未改网络结构、输入维度与类别。"
                          % (parent_name, parent_digest[:16], parent.meta["steps"], model.meta["steps"]))
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change_summary,
        configuration={"plan": plan, "metric": metric, "parent_validation": parent_result,
                       "candidate_validation": candidate_result,
                       "parent_checkpoint_sha256": parent_digest,
                       "candidate_checkpoint_sha256": candidate_digest},
        seed=int(model.meta["seed"]),
        checkpoint_paths=["checkpoint.npz"],
        code_paths=["baselines/model.py", "baselines/data.py"],
        predict_argv=PREDICT_ARGV,
    )
    log("候选已提交 artifact=%s 平台回执字段=%s" % (submitted.get("artifact_id"), sorted(submitted)))

    state_dir = rnd.state_workspace(work / ("state-%s-r%d" % (rnd.dataset_key.lower(), rnd.round_index)))
    previous_strategy = {}
    inherited = state_dir / "strategy.json"
    if inherited.is_file():
        try:
            previous_strategy = json.loads(inherited.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_strategy = {}
    trend = list(previous_strategy.get("trend", []))
    trend.append({"round": rnd.round_index, "parent": parent_score, "candidate": candidate_score,
                  "selected": "candidate" if improved else "parent", "plan": plan})
    best_score = parent_score if previous_strategy.get("best_score") is None else previous_strategy["best_score"]
    if improved and candidate_score is not None:
        if best_score is None or better(candidate_score, best_score, direction):
            best_score = candidate_score
    (state_dir / "strategy.json").write_text(json.dumps({
        "dataset_key": rnd.dataset_key, "metric": metric, "direction": direction,
        "best_score": best_score, "trend": trend[-12:],
        "last_artifact_id": submitted["artifact_id"],
        "selected_artifact_id": submitted["artifact_id"] if improved else rnd.parent.artifact_id,
    }, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    rnd.save_state(state_dir)

    selected = submitted["artifact_id"] if improved else rnd.parent.artifact_id
    reason = clip("公开验证 %s：候选 %s vs 父 %s，%s；依据本轮真实模型调用 %s 的超参与逐 epoch 测量选择。"
                  % (metric, fmt(candidate_score), fmt(parent_score),
                     "采纳候选" if improved else "保留父模型", receipt.call_id))
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=reason,
        configuration={"plan": plan, "metric": metric, "selected": "candidate" if improved else "parent",
                       "parent_validation": parent_result, "candidate_validation": candidate_result},
        seed=int(model.meta["seed"]),
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )
    log("第 %d 轮结束，selected=%s，用时 %.1fs" % (rnd.round_index, selected, time.monotonic() - started))


# ------------------------------------------------------------------ 预测
def predict(sample, model_dir):
    checkpoint = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    model = Model.load(checkpoint)
    from baselines.data import flatten
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("预测特征结构与检查点声明不一致")
    vector = np.asarray([values], dtype=np.float64)
    if sample.task == "regression" or model.meta["task"] == "regression":
        value = float(model.predict(vector)[0])
        return {"value": value if math.isfinite(value) and value >= 0 else 0.0}
    probabilities = model.predict(vector)[0]
    scores = [float(p) for p in probabilities]
    top = int(np.argmax(scores))
    peak = scores[top]
    for index in range(len(scores)):
        if index != top and scores[index] >= peak:
            scores[index] = max(0.0, peak - 1e-9)
    return [{"label": label, "score": scores[index]} for index, label in enumerate(model.meta["classes"])]
