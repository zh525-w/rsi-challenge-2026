"""本地干跑：假 Round 跑 A 十轮，验证 v14 代码路径、采纳链与提示词体积。"""
import json, os, shutil, tempfile, time
from pathlib import Path
import contest_submission as cs

ROOT = Path(tempfile.mkdtemp(prefix="dryrun-")); STORE = ROOT / "store"; STORE.mkdir()
os.environ["CONTEST_WORK"] = str(ROOT / "work"); (ROOT / "work").mkdir()
shutil.copy("checkpoint.npz", STORE / "baseline.npz")
rows = [json.loads(l) for l in open("public/train.jsonl", encoding="utf-8")]
val = [json.loads(l) for l in open("public/validation.jsonl", encoding="utf-8")]
logfile = ROOT / "prompt.jsonl"

class Data:
    def training_rows(self): return list(rows)
    def validation_rows(self): return list(val)
class Parent:
    def __init__(self, name, aid): self._name, self.artifact_id = name, aid
    def declared_paths(self): return [self._name]
    def path(self, name): return STORE / name
class Receipt:
    def __init__(self, text, cid): self.text, self.call_id = text, cid
class Round:
    def __init__(self, idx, parent, hist):
        self.round_index, self.parent, self._hist = idx, parent, hist
        self.dataset_key = "A"
        self.task = {"task": "classification", "metric": "macro_f1", "direction": "maximize"}
        self.data = Data(); self.max_output_tokens = 512; self.call_ids = []
        self.cfg = {}
    def history(self): return list(self._hist)
    def model(self, messages, max_tokens=48, temperature=0):
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps({"round": self.round_index,
                                "prompt_chars": sum(len(m["content"]) for m in messages),
                                "max_tokens": max_tokens}, ensure_ascii=False) + "\n")
        self.call_ids.append("stub-%d" % self.round_index)
        return Receipt('{"learning_rate":0.003,"epochs":3,"batch_size":64}', "stub-%d" % self.round_index)
    def state_workspace(self, p):
        p = Path(p); p.mkdir(parents=True, exist_ok=True); return p
    def save_state(self, p): pass
    def submit_candidate(self, candidate_dir, **kw):
        aid = "cand-%d" % self.round_index
        self._cand_name = aid + ".npz"
        shutil.copy(Path(candidate_dir) / "checkpoint.npz", STORE / self._cand_name)
        return {"artifact_id": aid}
    def finish(self, selected_artifact_id, **kw):
        self.selected = selected_artifact_id
        self.cfg = kw.get("configuration") or {}

parent = Parent("baseline.npz", "baseline"); hist = []; t0 = time.time()
for i in range(1, 11):
    r = Round(i, parent, hist); cs.train(r)
    cfg = r.cfg
    hist.append({"id": "stub-%d" % i, "round_index": i, "configuration": cfg})
    aid = "cand-%d" % i
    adopted = (r.selected == aid)
    chosen = getattr(r, "_cand_name", parent._name) if adopted else parent._name
    print("round %2d 父分 %.6f 候选 %.6f %s 用时 %.1fs"
          % (i, cfg.get("parent_score") or float("nan"), cfg.get("candidate_score") or float("nan"),
             "采纳" if adopted else "未采用", time.time() - t0), flush=True)
    parent = Parent(chosen, r.selected)
print("总用时 %.1fs" % (time.time() - t0))
print("提示词：", open(logfile, encoding="utf-8").read())
