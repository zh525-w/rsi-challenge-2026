"""C 微步配方扫描：从 baseline 权重出发单次续训，评估 FD001–FD004 四段增益。"""
import json, numpy as np, rsi.core as core, contest_submission as cs
rows=[json.loads(l) for l in open("/home/user/build/data_bc/c/public/train.jsonl")]
val=[json.loads(l) for l in open("/home/user/build/data_bc/c/public/validation.jsonl")]
base,meta=core.Net.load("/home/user/build/data_bc/c/checkpoint.npz"); paths=base.paths
X=cs.matrix(cs.load_rows(rows,cs.TRAIN_CAP,seed=42,label="t"),paths)
XV=cs.matrix(cs.load_rows(val,cs.VALIDATION_CAP,seed=43,label="v"),paths)
y=np.array([float(r["answer"]) for r in cs.load_rows(rows,cs.TRAIN_CAP,seed=42,label="t")])
yv=np.array([float(r["answer"]) for r in cs.load_rows(val,cs.VALIDATION_CAP,seed=43,label="v")])
print("baseline %.3f"%core.score(base,XV,yv,[],"regression","rmse"))
for lr in (0.0005,0.001,0.002):
    for ep in (1,2,3):
        for ar in (None,0.5,0.75,0.9):
            net=core.Net.load("/home/user/build/data_bc/c/checkpoint.npz")[0]
            net.paths=paths
            net.fit(X,y,epochs=ep,learning_rate=lr,batch_size=64,seed=42)
            cand=cs.anchor(base,net,ar) if ar else net
            tot=core.score(cand,XV,yv,[],"regression","rmse")
            gains=[]
            for sub in ("FD001","FD002","FD003","FD004"):
                idx=np.array([i for i,r in enumerate(val) if r["meta"]["source_subset"]==sub])
                b=core.score(base,XV[idx],yv[idx],[],"regression","rmse")
                c=core.score(cand,XV[idx],yv[idx],[],"regression","rmse")
                gains.append(cs.relative_gain(b,c,"lower"))
            print("lr=%.4g ep=%d anchor=%s 总 %.2f%% 四段 %s 最差 %+.2f%%"%(lr,ep,ar,(cs.relative_gain(52.8686,tot,"lower"))*100,
                  "/".join("%+.2f%%"%(g*100) for g in gains),min(gains)*100),flush=True)
