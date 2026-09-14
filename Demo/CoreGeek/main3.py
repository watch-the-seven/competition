#!/usr/bin/env python3
"""程序入口：判题器执行 `bash run.sh <port>`（或 `python3 main3.py <port>`）时被拉起。

只做三件事：校验端口参数 -> 把工作目录切到本文件所在目录 -> 启动 HTTP 服务并阻塞。
之后判题器会不停地往这个端口 POST 每个回合的地图状态。
"""

import logging
import os
import sys
from pathlib import Path


def main() -> None:
    # 接口文档第 1 章：端口由判题器通过命令行传入，必须且只能有这一个参数。
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main3.py <port>")
    port = int(sys.argv[1])

    # 不管从哪个目录被拉起，都把工作目录固定到工程根目录，
    # 这样 `src/agent` 的导入不会受调用方当前目录影响。
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    # 日志输出到 stdout：判题器一般会收集选手进程的输出，便于赛后排查。
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    # 延迟导入：保证上面的 sys.path 已经生效后再导入 agent 包。
    from agent.server import serve

    logging.info("listening on 0.0.0.0:%d", port)
    serve(port)          # 内部是 serve_forever()，会一直阻塞到进程被杀


if __name__ == "__main__":
    main()
