"""联调千问（OpenAI 兼容接口）的最小探针：不用跑训练就能确认接口是否可用。

用法：
    set PA_MODEL_BASE_URL / PA_MODEL_API_KEY / PA_MODEL_NAME 后
    python scripts/probe_llm.py             # 列模型 + 发一次极小请求
    python scripts/probe_llm.py --list-only # 只列模型，不花 token

通过标准：打印出的 JSON 能解析，且只含 epochs / lr / batch / accept / note。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("PA_MODEL_BASE_URL", "").rstrip("/")
KEY = os.environ.get("PA_MODEL_API_KEY", "")
NAME = os.environ.get("PA_MODEL_NAME", "")

PROBE_PROMPT = (
    "你是训练实验的决策器。当前第 1 轮，历史为空，父检查点验证集 Macro-F1=0.714030。"
    "请只输出一个 JSON 对象，字段仅限 epochs / lr / batch / accept / note，未改的字段留空。"
)


def call(method, path, payload=None):
    url = BASE + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    list_only = "--list-only" in sys.argv
    missing = [name for name, value in (("PA_MODEL_BASE_URL", BASE),
                                        ("PA_MODEL_API_KEY", KEY),
                                        ("PA_MODEL_NAME", NAME)) if not value]
    if missing:
        print("缺少环境变量：" + "、".join(missing))
        print(r"""
PowerShell 里这样设（只对当前窗口生效）：
  $env:PA_MODEL_BASE_URL = "https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
  $env:PA_MODEL_API_KEY  = "sk-xxxxxxxx"
  $env:PA_MODEL_NAME     = "qwen3.8-max"    # 以 GET /models 返回的真实 id 为准

想写进系统环境（重开窗口仍有效）：
  [Environment]::SetEnvironmentVariable("PA_MODEL_API_KEY", "sk-xxx", "User")
""")
        return 2

    print("BASE_URL = %s" % BASE)
    print("MODEL    = %s" % NAME)

    try:
        models = call("GET", "/models")
    except urllib.error.HTTPError as error:
        print("拉取模型列表失败 %d：%s" % (error.code, error.read()[:300]))
        models = None
    except urllib.error.URLError as error:
        print("接口不可达：%s（检查网络或代理）" % error.reason)
        return 3

    if models:
        ids = [item.get("id") for item in models.get("data", [])]
        print("可用模型 %d 个；是否包含目标模型：%s" % (len(ids), NAME in ids))
        for item in ids[:15]:
            print("  - %s" % item)
        if NAME not in ids:
            print("提示：目标模型名不在列表里，请从上面挑一个真实 id 回填 PA_MODEL_NAME")
    if list_only:
        return 0

    payload = {
        "model": NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "只输出 JSON，不要解释。"},
            {"role": "user", "content": PROBE_PROMPT},
        ],
    }
    started = time.time()
    try:
        body = call("POST", "/chat/completions", payload)
    except urllib.error.HTTPError as error:
        print("请求失败 %d：%s" % (error.code, error.read()[:300]))
        return 4
    except urllib.error.URLError as error:
        print("请求不可达：%s" % error.reason)
        return 3
    elapsed = round(time.time() - started, 2)

    content = body["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        content = content[4:] if content.startswith("json") else content
    usage = body.get("usage", {})
    print("耗时 %.2f 秒，prompt_tokens=%s，completion_tokens=%s" % (
        elapsed, usage.get("prompt_tokens"), usage.get("completion_tokens")))
    print("原始输出：" + content[:300])

    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        print("结论：输出不是合法 JSON，决策层会拒绝；换模型或改用更强的模型 id")
        return 5

    allowed = {"epochs", "lr", "batch", "accept", "note"}
    extra = set(action) - allowed
    if extra:
        print("结论：JSON 合法但含多余键 %s，validate() 会拒绝" % "、".join(sorted(extra)))
        return 6
    print("结论：通过。可执行 python scripts/run_harness.py --decider llm --max-rounds 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
