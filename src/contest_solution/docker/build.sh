#!/usr/bin/env sh
# 构建并验证 A 赛道镜像；给 --save 时导出 tar 供提交。
# 用法： sh docker/build.sh [--save]
set -eu

NAME="rsi-a-cpu:1.0"

echo "[1/4] 构建镜像 $NAME"
docker build -t "$NAME" -f docker/Dockerfile .

echo "[2/4] 镜像内自检（版本、检查点、9 项官方测试）"
docker run --rm --entrypoint python "$NAME" scripts/selfcheck.py

echo "[3/4] 入口检查（不落盘）"
docker run --rm "$NAME" --help

if [ "${1:-}" = "--save" ]; then
  echo "[4/4] 导出镜像 tar"
  docker save -o rsi-a-cpu-1.0.tar "$NAME"
  echo "已导出 rsi-a-cpu-1.0.tar"
else
  echo "[4/4] 跳过导出（加 --save 可导出 tar）"
fi

echo "完成。实际跑一轮："
echo "  docker run --rm -v \$PWD/runs-docker:/app/runs -v \$PWD/logs:/app/logs $NAME --round 1"
