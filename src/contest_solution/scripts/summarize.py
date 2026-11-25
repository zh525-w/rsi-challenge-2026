"""汇总 logs/experiments.csv，打印逐轮指标、决策，并指出当前被采纳的轮次。

用法：
    python scripts/summarize.py
"""

import csv
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "experiments.csv"
STATE_PATH = ROOT / "state.json"
BASELINE_F1 = 0.7140297220788709


def width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad(text, target):
    text = str(text)
    return text + " " * max(0, target - width(text))


def main():
    if not LOG_PATH.exists():
        raise SystemExit("还没有实验记录，请先运行 scripts/run_round.py --round 1")
    rows = list(csv.DictReader(open(LOG_PATH, encoding="utf-8")))
    if not rows:
        raise SystemExit("实验记录为空")

    labels = ["轮次", "父轮次", "校验", "改前", "改后", "变化", "步数", "学习率", "耗时", "决策"]
    widths = [5, 7, 6, 9, 9, 10, 12, 8, 7, 5]

    header = "".join(pad(l, w) for l, w in zip(labels, widths))
    print(header)
    print("-" * width(header))

    for row in rows:
        cells = [
            row["round"], row["parent_round"],
            "通过" if row["parent_digest_ok"] == "True" else "不一致",
            row["macro_f1_before"], row["macro_f1_after"], row["delta"],
            "%s→%s" % (row["steps_before"], row["steps_after"]),
            row["learning_rate"], row["training_seconds"], row["decision"],
        ]
        print("".join(pad(c, w) for c, w in zip(cells, widths)))

    print("")
    best = max(rows, key=lambda r: float(r["macro_f1_after"]))
    adopted = [r for r in rows if r["decision"] == "采纳"]
    print("已被采纳的轮次：%s" % ("、".join(r["round"] for r in adopted) if adopted else "无"))
    print("当前最佳：第 %s 轮，Macro-F1 = %s" % (best["round"], best["macro_f1_after"]))
    print("平台静态基线：%.6f" % BASELINE_F1)
    print("已用轮次：%d / 10" % len(rows))

    if STATE_PATH.exists():
        import json
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print("续训起点：%s" % state["selected_checkpoint"])
        if state.get("selected_f1"):
            print("相对基线：%+.6f" % (state["selected_f1"] - BASELINE_F1))


if __name__ == "__main__":
    main()
