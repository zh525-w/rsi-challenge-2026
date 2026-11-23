# 智训引擎 RSI CHALLENGE 续训镜像（A/B/C 三数据集，contest.harness.workspace.v2）
# 平台以固定命令调用：python3 /app/harness.py describe|train|predict
# 镜像内不含密钥、不含公开数据与父检查点（由平台挂载 /data、/previous、/model）。

FROM python:3.10.11-alpine

LABEL org.opencontainers.image.title="rsi-cpu-abc" \
      org.opencontainers.image.description="智训引擎 RSI CHALLENGE：气体识别/轴承RUL/发动机RUL CPU 续训提交镜像" \
      org.opencontainers.image.version="2.8" \
      org.opencontainers.image.source="cpu-a/b/c-guide-20261110-v1 + contest-harness-sdk-1.0.4"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    PYTHONPATH=/app \
    BASELINE_CPU_THREADS=4 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# 依赖单独一层：仅 numpy（与基线报告同版本）；构建期走阿里云 PyPI 镜像
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check \
      -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
      -r requirements.txt

# Harness SDK（contest_harness 1.0.2）与固定入口
COPY contest_harness ./contest_harness
COPY harness.py ./harness.py

# 选手模块与工业模型代码（官方 contest.cpu-mlp.v1 检查点格式）
COPY contest_submission.py ./contest_submission.py
COPY baselines ./baselines

# 构建期自检：describe 必须通过，且不得在镜像内留下密钥或数据
RUN python3 /app/harness.py describe && \
    test ! -e /app/data && test ! -e /app/checkpoint.npz

# 平台以 `python3 /app/harness.py <describe|train|predict>` 启动，不设 ENTRYPOINT 覆盖
CMD ["python3", "/app/harness.py", "describe"]
