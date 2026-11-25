"""C：2 epoch、无锚定、按轮换种子，模拟十轮严格四段护栏后的累计结果。"""
import json, copy, numpy as np, rsi.core as core, contest_submission as cs
rows=[json.loads(l) for l in open("/home/user/build/data_bc/c/public/train.jsonl")]
val=[json.loads(l) for l in open("/home/user/build/data_bc/c/public/validation.jsonl")]
paths=core.Net.load("/home/user/build/data_bc/c/checkpoint.npz")[0].paths
X=cs.matrix(cs.load_rows(rows,cs.TRAIN_CAP,seed=42,label="t"),paths)
y=np.array([float(r["answer"]) for r in cs.load_rows(rows,cs.TRAIN_CAP,seed=42,label="t")])
XV=cs.matrix(cs.load_rows(val,cs.VALIDATION_CAP,seed=43,label="v"),paths)
yv=np.array([float(r["answer"]) for r in cs.load_rows(val,cs.VALIDATION_CAP,seed=43,label="v")])
subs=("FD001","FD002","FD003","FD004")
idx={s:np.array([i for i,r in enumerate(val) if r["meta"]["source_subset"]==s]) for s in subs}
def gains(net):
    out=[]
    for s in subs:
        out.append(cs.relative_gain(core.score(BASE,XV[idx[s]],yv[idx[s]],[],"regression","rmse"),
                                    core.score(net,XV[idx[s]],yv[idx[s]],[],"regression","rmse"),"lower"))
    return out
BASE=core.Net.load("/home/user/build/data_bc/c/checkpoint.npz")[0]; BASE.paths=paths
cur=copy.deepcopy(BASE); cur.paths=paths
print("round  总增益   四段")
for r in range(1,11):
    cand=copy.deepcopy(cur); cand.paths=paths
    cand.fit(X,y,epochs=2,learning_rate=0.0005,batch_size=64,seed=42+r)
    g=gains(cand)
    ok=(cs.relative_gain(core.score(cur,XV,yv,[],"regression","rmse"),core.score(cand,XV,yv,[],"regression","rmse"),"lower") or 0)>0 and min(g)>0
    if ok: cur=cand
    print("%2d %s %6.2f%%  %s"%(r,"采纳" if ok else "未采用",
          (cs.relative_gain(core.score(BASE,XV,yv,[],"regression","rmse"),core.score(cur,XV,yv,[],"regression","rmse"),"lower"))*100,
          "/".join("%+.2f%%"%(x*100) for x in gains(cur))),flush=True)
