"""本地干跑任意数据集：python3 dryrun_any.py A|B|C，假 Round 跑十轮，打印采纳链与最差段。"""
import json, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import contest_submission as cs
import rsi.core as core

DS = sys.argv[1].upper()
SRC = {"A": ".", "B": "/home/user/build/data_bc/b", "C": "/home/user/build/data_bc/c"}[DS]
TASK = {"A": {"task": "classification", "metric": "macro_f1", "direction": "maximize"},
        "B": {"task": "regression", "metric": "rmse", "metric_direction": "lower"},
        "C": {"task": "regression", "metric": "rmse", "metric_direction": "lower"}}[DS]

raw_tr = [json.loads(l) for l in open(os.path.join(SRC, "public/train.jsonl"), encoding="utf-8")]
raw_va = [json.loads(l) for l in open(os.path.join(SRC, "public/validation.jsonl"), encoding="utf-8")]

ROOT = Path(tempfile.mkdtemp(prefix="dry-%s-" % DS)); STORE = ROOT / "store"; STORE.mkdir()
os.environ["CONTEST_WORK"] = str(ROOT / "work"); (ROOT / "work").mkdir()
shutil.copy(os.path.join(SRC, "checkpoint.npz"), STORE / "baseline.npz")

class Data:
    def training_rows(self): return raw_tr
    def validation_rows(self): return raw_va
class Parent:
    def __init__(self, name, aid): self._name, self.artifact_id = name, aid
    def declared_paths(self): return [self._name]
    def path(self, name): return STORE / name
class Receipt:
    def __init__(self, text, cid): self.text, self.call_id = text, cid
class Round:
    def __init__(self, idx, parent, hist):
        self.round_index, self.parent, self._hist = idx, parent, hist
        self.dataset_key = DS; self.task = TASK; self.data = Data()
        self.max_output_tokens = 512; self.call_ids = []; self.cfg = {}
    def history(self): return list(self._hist)
    def model(self, messages, max_tokens=48, temperature=0):
        self._chars = sum(len(m["content"]) for m in messages)
        return Receipt('{"learning_rate":0.003,"epochs":3,"batch_size":64}', "stub-%d" % self.round_index)
    def state_workspace(self, p):
        p = Path(p); p.mkdir(parents=True, exist_ok=True); return p
    def save_state(self, p): pass
    def submit_candidate(self, candidate_dir, **kw):
        aid = "cand-%d" % self.round_index; self._cand = aid + ".npz"
        shutil.copy(Path(candidate_dir) / "checkpoint.npz", STORE / self._cand)
        return {"artifact_id": aid}
    def finish(self, selected_artifact_id, **kw):
        self.selected = selected_artifact_id; self.cfg = kw.get("configuration") or {}

def block_gain(parent_net, cand_net, paths, va, kind, metric, blocks=4):
    rows = cs.load_rows(va, cs.VALIDATION_CAP, seed=43, label="验证集")
    x = cs.matrix(rows, paths)
    if kind == "regression":
        ans = np.asarray([float(r.get("answer", 0.0)) for r in rows]); classes = []
    else:
        classes = sorted({str(r.get("answer")) for r in cs.load_rows(raw_tr, cs.TRAIN_CAP, seed=42, label="x")})
        lk = {c: i for i, c in enumerate(classes)}
        ans = np.asarray([lk.get(str(r.get("answer")), 0) for r in rows])
    groups = {}
    if rows and isinstance(rows[0].get("meta"), dict) and "source_subset" in rows[0]["meta"]:
        for i, r in enumerate(rows):
            groups.setdefault(str(r["meta"]["source_subset"]), []).append(i)
        parts = [np.asarray(v) for v in sorted(groups.values(), key=lambda v: len(v))] and [np.asarray(groups[k]) for k in sorted(groups)]
    else:
        parts = cs.slice_blocks(ans, kind, blocks)
    out = []
    for part in parts:
        p = core.score(parent_net, x[part], ans[part], classes, kind, metric)
        c = core.score(cand_net, x[part], ans[part], classes, kind, metric)
        out.append(cs.relative_gain(p, c, "lower" if metric in ("mae", "rmse") else "higher"))
    return out

parent = Parent("baseline.npz", "baseline"); hist = []; t0 = time.time()
for i in range(1, 11):
    r = Round(i, parent, hist); cs.train(r)
    cfg = r.cfg; aid = "cand-%d" % i
    adopted = (r.selected == aid)
    hist.append({"id": "stub-%d" % i, "round_index": i, "configuration": cfg})
    pn, pmeta = core.Net.load(STORE / parent._name)
    pn.paths = cs.parse_paths(pmeta.get("paths")) or pn.paths
    cand_name = getattr(r, "_cand", parent._name)
    try:
        cnet = core.Net.load(STORE / cand_name)[0]
        cnet.paths = pn.paths
        gains = block_gain(pn, cnet, pn.paths, raw_va, cs.task_kind(TASK), TASK["metric"])
        gtxt = " ".join("%+.2f%%" % (g * 100) if g is not None else "—" for g in gains)
    except Exception as e:
        gtxt = "读不出（%s）" % e
    print("round %2d 父 %s 候选 %s %s 分段 %s 用时 %.1fs"
          % (i, cs.fmt(cfg.get("parent_score")), cs.fmt(cfg.get("candidate_score")),
             "采纳" if adopted else "未采用", gtxt, time.time() - t0), flush=True)
    if adopted:
        parent = Parent(cand_name, aid)
print("总用时 %.1fs" % (time.time() - t0))
