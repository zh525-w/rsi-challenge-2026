# A 赛道 CPU 续训镜像
# 用法见 docker/build.ps1、docker/build.sh 与 README 第十节。

FROM python:3.10.11-slim

LABEL org.opencontainers.image.title="rsi-a-cpu" \
      org.opencontainers.image.description="智训引擎 RSI CHALLENGE A 赛道：气体识别 CPU 续训最小原型" \
      org.opencontainers.image.version="1.0" \
      org.opencontainers.image.source="cpu-a-guide-20261110-v1"

# 确定性相关的环境变量：单 BLAS 线程、固定哈希种子、不写 pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    PYTHONPATH=/app \
    BASELINE_CPU_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# 依赖单独一层，避免改代码就重装
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY baselines ./baselines
COPY public ./public
COPY configs ./configs
COPY scripts ./scripts
COPY harness ./harness
COPY checkpoint.npz ./checkpoint.npz
COPY baseline-info.json ./baseline-info.json
# 适配层声明：describe 模式从这里读 schema / modes / adapter_source_ref
COPY adapter_manifest.json ./adapter_manifest.json
COPY TASK.md ./TASK.md
COPY TRAINING_GUIDE.md ./TRAINING_GUIDE.md
COPY README.md ./README.md

RUN mkdir -p /app/runs /app/logs /app/results

# 构建期自检：9 项官方测试必须全部通过，否则镜像构建失败
RUN python scripts/selfcheck.py

# 入口是 Harness 本体：封闭评测时无人值守跑完五环并输出 results/final.json
ENTRYPOINT ["python", "scripts/run_harness.py"]
CMD ["--unattended"]
