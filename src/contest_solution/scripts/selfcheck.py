# 镜像内自检：版本、检查点完整性、官方测试。
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("Python  :", sys.version.split()[0])
    print("NumPy   :", np.__version__)
    print("平台    :", sys.platform)
    print("线程设置:", os.environ.get("BASELINE_CPU_THREADS", "未设置"))

    checkpoint = ROOT / "checkpoint.npz"
    if not checkpoint.exists():
        raise SystemExit("缺少 checkpoint.npz")
    with np.load(checkpoint, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        elements = sum(int(archive[key].size) for key in keys)
    print("检查点  : %d 个数组，共 %d 个元素" % (len(keys), elements))

    info = json.loads((ROOT / "baseline-info.json").read_text(encoding="utf-8"))
    print("声明的参数量:", info.get("parameter_count"))
    if int(info.get("parameter_count", 0)) != 4326:
        raise SystemExit("参数量与预期 4,326 不一致")

    env = dict(os.environ)
    # 用相对路径，避免中文绝对路径在 PYTHONPATH 里失效
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [sys.executable, str(ROOT / "baselines" / "tests" / "test_baselines.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="strict")
    tail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
    print("官方测试:", tail or completed.stdout.strip()[-80:])
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit("官方测试未通过，镜像不可用")
    print("自检通过")


if __name__ == "__main__":
    main()
