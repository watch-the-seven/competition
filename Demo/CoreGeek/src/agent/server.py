"""HTTP 接入层：判题器 POST 当前回合状态，本程序返回本回合指令。

接口文档第 1/2 章：
  - 监听地址固定 0.0.0.0，端口由启动参数传入；
  - 请求体是一个 Request JSON，响应体是一个 Response JSON，
    形如 {"roleCommandMap": {...}, "prompt": "...", "executeCmd": "..."}。
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide
from .protocol import empty_response

LOGGER = logging.getLogger(__name__)

# 任务书 4.2：一天 130 回合，前 70 回合是白天。仅用于日志可读性。
ROUNDS_PER_DAY = 130
DAY_ROUNDS = 70


class Handler(BaseHTTPRequestHandler):
    """每来一次 POST 就决策一个回合。"""

    def do_POST(self) -> None:
        # 按 Content-Length 读完整个请求体（请求体可能几千字节）。
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            response = decide(payload)
            # 打一行摘要日志：回合数、金币、昼夜、本回合下发的指令。
            LOGGER.info(
                "round %s gold=%s day=%s cmd=%s",
                payload.get("roundNo"),
                (payload.get("teamOur") or {}).get("goldNum"),
                (payload.get("roundNo", 0) - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
                response["roleCommandMap"],
            )
        except Exception:
            # 关键容错：无论报文多畸形、策略里出什么异常，都必须回一个合法响应，
            # 绝不能让进程崩掉或卡住，否则后面所有回合都拿不到指令。
            LOGGER.exception("decision failed")
            response = empty_response()

        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        # 接口文档要求返回 JSON；显式给 Content-Length，客户端才能正确读完。
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # 屏蔽 BaseHTTPRequestHandler 默认打到 stderr 的每请求访问日志，
        # 避免和自己的 LOGGER 重复、刷屏。
        return


def serve(port: int) -> None:
    # ThreadingHTTPServer：每个连接一个线程，判题器并发/Keep-Alive 都不会互相阻塞。
    # 接口文档未规定路径，do_POST 不检查 self.path，任意路径都能收到决策请求。
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
