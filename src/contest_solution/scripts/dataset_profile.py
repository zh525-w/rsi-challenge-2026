"""数据画像：批次数、类别分布，用于漂移章节的事实基础。"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def profile(name, path):
    batches, labels = Counter(), Counter()
    rows = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            batches[row.get("meta", {}).get("source_batch")] += 1
            labels[row["answer"]] += 1
    print("%s：%d 行，%d 个批次" % (name, rows, len(batches)))
    batch_text = "，".join("b%s=%d" % (k, v) for k, v in sorted(batches.items(), key=lambda x: str(x[0])))
    print("  批次分布：" + batch_text)
    print("  类别分布：" + "，".join("%s=%d" % (k, v) for k, v in labels.most_common()))


if __name__ == "__main__":
    profile("训练集", ROOT / "public" / "train.jsonl")
    profile("验证集", ROOT / "public" / "validation.jsonl")
