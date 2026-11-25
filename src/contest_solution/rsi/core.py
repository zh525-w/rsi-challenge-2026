"""通用小模型：路径化特征抽取 + MLP 分类/回归。

口径对齐官方 baseline（contest.cpu-mlp.v1）：
- 特征按路径描述，任意嵌套 inputs 都能压平，A/B/C 共用一套代码；
- 输入标准化 + 裁剪到 [-10, 10]；
- 回归先归一化目标，输出层线性，预测时反归一化并取非负；
- 分类 tanh 隐层 + softmax，回归 tanh 隐层 + 单输出；
- 检查点沿用 w1/b1/w2/b2/mean/scale/adam_*/metadata 命名，A 赛道可直接继承官方权重。
"""
import json
import math

import numpy as np

FORMAT = "contest.cpu-mlp.v1"
DEFAULT_HIDDEN = 32
MAX_FEATURES = 4096
CLIP = 10.0


def _walk(node, prefix, out):
    if isinstance(node, dict):
        for key in sorted(node, key=lambda item: str(item)):
            _walk(node[key], prefix + [["key", str(key)]], out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, prefix + [["index", index]], out)
    elif isinstance(node, bool):
        out.append(prefix)
    elif isinstance(node, (int, float)):
        out.append(prefix)


def leaf_paths(inputs, limit=MAX_FEATURES):
    found = []
    _walk(inputs, [], found)
    return found[:limit]


def union_paths(rows, limit=MAX_FEATURES):
    merged = {}
    for row in rows:
        for path in leaf_paths(row.get("inputs"), limit):
            merged.setdefault(json.dumps(path, sort_keys=True), path)
            if len(merged) >= limit:
                return [merged[key] for key in sorted(merged)]
    return [merged[key] for key in sorted(merged)]


def value_at(inputs, path):
    current = inputs
    for kind, key in path:
        if isinstance(current, dict):
            if kind != "key" or key not in current:
                return None
            current = current[key]
        elif isinstance(current, (list, tuple)):
            if kind != "index":
                return None
            index = int(key)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    number = float(current)
    return number if math.isfinite(number) else None


def extract(inputs, paths, fill=0.0):
    vector = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        value = value_at(inputs, path)
        vector[index] = fill if value is None else value
    return vector


def parse_paths(raw):
    """官方 metadata 的 paths 是 JSON 字符串数组，统一解析成 [(kind, key)] 列表。"""
    result = []
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
            if not isinstance(step, list) or len(step) != 2 or step[0] not in ("key", "index"):
                return []
            path.append([step[0], str(step[1]) if step[0] == "key" else int(step[1])])
        result.append(path)
    return result


def _softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Net(object):
    def __init__(self, n_features, hidden=DEFAULT_HIDDEN, out_dim=6, kind="classification", seed=0):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.w1 = rng.normal(0.0, 0.1, (n_features, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0.0, 0.1, (self.hidden, self.out_dim))
        self.b2 = np.zeros(self.out_dim)
        self.adam_first_w1 = np.zeros_like(self.w1)
        self.adam_first_b1 = np.zeros_like(self.b1)
        self.adam_first_w2 = np.zeros_like(self.w2)
        self.adam_first_b2 = np.zeros_like(self.b2)
        self.adam_second_w1 = np.zeros_like(self.w1)
        self.adam_second_b1 = np.zeros_like(self.b1)
        self.adam_second_w2 = np.zeros_like(self.w2)
        self.adam_second_b2 = np.zeros_like(self.b2)
        self.mean = np.zeros(n_features)
        self.scale = np.ones(n_features)
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.classes = []
        self.paths = []
        self.step = 0

    def normalize(self, x):
        safe = np.where(self.scale < 1e-12, 1.0, self.scale)
        return np.clip((x - self.mean) / safe, -CLIP, CLIP)

    def _hidden(self, x):
        return np.tanh(x.dot(self.w1) + self.b1)

    def logits(self, x):
        return self._hidden(x).dot(self.w2) + self.b2

    def predict(self, x):
        z = self.logits(x)
        if self.kind == "regression":
            return np.maximum(0.0, z[:, 0] * self.target_scale + self.target_mean)
        return _softmax(z)

    def _backward(self, x, grad_z):
        h = self._hidden(x)
        grad_w2 = h.T.dot(grad_z)
        grad_b2 = grad_z.sum(axis=0)
        grad_h = grad_z.dot(self.w2.T) * (1.0 - h * h)
        grad_w1 = x.T.dot(grad_h)
        grad_b1 = grad_h.sum(axis=0)
        return grad_w1, grad_b1, grad_w2, grad_b2

    def fit(self, x, y, epochs, learning_rate, batch_size=128, seed=0, clip=5.0):
        """y：分类为整数下标，回归为原始目标（内部按 target_mean/scale 归一化）。"""
        rng = np.random.default_rng(seed)
        x = self.normalize(np.asarray(x, dtype=np.float64))
        total = x.shape[0]
        if total == 0:
            return self
        for _ in range(int(epochs)):
            order = rng.permutation(total)
            for start in range(0, total, batch_size):
                index = order[start:start + batch_size]
                xb, yb = x[index], y[index]
                z = self.logits(xb)
                if self.kind == "regression":
                    target = ((yb - self.target_mean) / self.target_scale).reshape(-1, 1)
                    grad_z = 2.0 * (z - target) / max(len(yb), 1)
                else:
                    prob = _softmax(z)
                    prob[np.arange(len(yb)), yb] -= 1.0
                    grad_z = prob / max(len(yb), 1)
                self._adam(self._backward(xb, grad_z), learning_rate, clip)
        return self

    def _adam(self, grads, learning_rate, clip, beta1=0.9, beta2=0.999, eps=1e-8):
        """与官方 baseline 一致：先按全局 L2 范数裁剪到 clip，再做 Adam。"""
        self.step += 1
        norm = float(np.sqrt(sum(float(np.sum(grad * grad)) for grad in grads)))
        scale = min(1.0, clip / max(norm, 1e-12)) if clip and clip > 0 else 1.0
        for name, grad in zip(("w1", "b1", "w2", "b2"), grads):
            grad = grad * scale
            first = getattr(self, "adam_first_" + name)
            second = getattr(self, "adam_second_" + name)
            first *= beta1
            first += (1.0 - beta1) * grad
            second *= beta2
            second += (1.0 - beta2) * grad * grad
            corrected_first = first / (1.0 - beta1 ** self.step)
            corrected_second = second / (1.0 - beta2 ** self.step)
            getattr(self, name)[...] -= learning_rate * corrected_first / (np.sqrt(corrected_second) + eps)

    def save(self, path, extra=None):
        meta = {
            "format": FORMAT,
            "kind": self.kind,
            "hidden": self.hidden,
            "classes": list(self.classes),
            "paths": self.paths,
            "target_mean": float(self.target_mean),
            "target_scale": float(self.target_scale),
            "feature_clip": [-CLIP, CLIP],
            "parameter_count": int(self.w1.size + self.b1.size + self.w2.size + self.b2.size),
            "feature_count": int(self.w1.shape[0]),
            "adam_steps": self.step,
        }
        if extra:
            meta.update(extra)
        np.savez_compressed(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            mean=self.mean, scale=self.scale,
            adam_first_w1=self.adam_first_w1, adam_first_b1=self.adam_first_b1,
            adam_first_w2=self.adam_first_w2, adam_first_b2=self.adam_first_b2,
            adam_second_w1=self.adam_second_w1, adam_second_b1=self.adam_second_b1,
            adam_second_w2=self.adam_second_w2, adam_second_b2=self.adam_second_b2,
            metadata=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )

    @classmethod
    def load(cls, path):
        with np.load(str(path), allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            net = cls(data["w1"].shape[0],
                      hidden=int(meta.get("hidden", data["w1"].shape[1])),
                      out_dim=data["w2"].shape[1],
                      kind=(meta.get("kind") or meta.get("task") or "classification"))
            for name in ("w1", "b1", "w2", "b2", "mean", "scale",
                         "adam_first_w1", "adam_first_b1", "adam_first_w2", "adam_first_b2",
                         "adam_second_w1", "adam_second_b1", "adam_second_w2", "adam_second_b2"):
                if name in data:
                    setattr(net, name, np.array(data[name], dtype=np.float64))
            net.classes = list(meta.get("classes") or [])
            net.paths = parse_paths(meta.get("paths")) or list(meta.get("paths") or [])
            net.target_mean = float(meta.get("target_mean", 0.0))
            net.target_scale = float(meta.get("target_scale", 1.0)) or 1.0
            net.step = int(meta.get("adam_steps", 0))
            return net, meta


def fit_scaler(x):
    if x.size:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
    else:
        mean = np.zeros(x.shape[1])
        scale = np.ones(x.shape[1])
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def macro_f1(labels, predicted, classes):
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    scores = []
    for index in range(len(classes)):
        true_positive = int(((labels == index) & (predicted == index)).sum())
        false_positive = int(((labels != index) & (predicted == index)).sum())
        false_negative = int(((labels == index) & (predicted != index)).sum())
        if true_positive == 0:
            scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def mean_absolute_error(target, predicted):
    if len(target) == 0:
        return 0.0
    return float(np.abs(np.asarray(target) - np.asarray(predicted)).mean())


def root_mean_squared_error(target, predicted):
    if len(target) == 0:
        return 0.0
    return float(np.sqrt(((np.asarray(target) - np.asarray(predicted)) ** 2).mean()))


def score(net, x, answers, classes, task, metric="macro_f1"):
    predicted = net.predict(net.normalize(x))
    if task == "regression":
        if str(metric).lower().startswith("mae"):
            return mean_absolute_error(answers, predicted)
        return root_mean_squared_error(answers, predicted)
    return macro_f1(answers, predicted.argmax(axis=1), classes)
