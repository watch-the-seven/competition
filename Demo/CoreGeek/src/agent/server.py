"""HTTP 接入层：判题器 POST 当前回合状态，本程序返回本回合指令。

接口文档第 1/2 章：
  - 监听地址固定 0.0.0.0，端口由启动参数传入；
  - 请求体是一个 Request JSON，响应体是一个 Response JSON，
    形如 {"roleCommandMap": {...}, "prompt": "...", "executeCmd": "..."}。

日志是**机械**的：读到的原始字节原样写进去，准备发出去的字节原样写进去，
中间用分界线隔开。不做美化、不重新序列化，所以日志里看到的就是链路上真实的内容。

    ──────────────────── round 41 ────────────────────
    >>> REQUEST
    {"roundNo":41,"mapInfo":{...}}          ← 判题器发来的原文
    <<< RESPONSE
    {"roleCommandMap":{...},"prompt":"","executeCmd":""}   ← 即将发出的原文
    ──────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide
from .protocol import as_int, empty_response

LOGGER = logging.getLogger(__name__)

# 任务书 4.2：一天 130 回合，前 70 回合是白天。仅用于日志可读性。
ROUNDS_PER_DAY = 130
DAY_ROUNDS = 70

# 日志详细程度（可用环境变量 AGENT_LOG_VERBOSITY 覆盖，不必改代码）：
#   "full"   原样写入读到的 Request 和即将发出的 Response。
#            实测约 4.3KB/回合 -> 一天(130 回合)约 562KB，十天约 5.5MB。
#   "digest" 只挑选关键字段——注意这一档**不是**原样，仅供日志太大时使用。
#   "off"    完全不写报文。
LOG_VERBOSITY = os.environ.get("AGENT_LOG_VERBOSITY", "full").strip().lower()

RULE_WIDTH = 96          # 分界线宽度
RAW_LOG_LIMIT = 500      # 报文不是合法 JSON 时，最多记录多少原始字节


def _rule(title: str = "") -> str:
    """生成一条分界线；带标题时把标题居中，便于在长日志里定位回合。"""
    if not title:
        return "─" * RULE_WIDTH
    text = f" {title} "
    left = max(0, (RULE_WIDTH - len(text)) // 2)
    right = max(0, RULE_WIDTH - left - len(text))
    return "─" * left + text + "─" * right


def _round_title(payload: dict[str, Any]) -> str:
    """分界线上的标题，只放回合号，方便在长日志里定位。"""
    return f"round {as_int(payload.get('roundNo'), 1)}"


def _request_digest(payload: dict[str, Any]) -> str:
    """LOG_VERBOSITY="digest" 专用：只挑出驱动决策的字段（非原样）。"""
    team = payload.get("teamOur") or {}
    digest = {
        "roundNo": payload.get("roundNo"),
        "goldNum": team.get("goldNum"),
        "phaseTask": payload.get("phaseTask"),
        "llmResp": payload.get("llmResp"),
        "lastCmdResult": payload.get("lastCmdResult"),
        "lastRoundRoleActionResults": payload.get("lastRoundRoleActionResults"),
        "roles": [
            "{id}/{kind}@{x},{y}/hp{hp}/bp{n}".format(
                id=role.get("id"), kind=role.get("roleType"),
                x=(role.get("pos") or {}).get("x"),
                y=(role.get("pos") or {}).get("y"),
                hp=role.get("health"), n=len(role.get("backpack") or []),
            )
            for role in team.get("roles") or ()
        ],
        "robotNum": len((payload.get("robot") or {}).get("roles") or ()),
        "errors": payload.get("errors"),
    }
    return json.dumps(digest, ensure_ascii=False)


def _log_round(payload: dict[str, Any], raw_text: str, body: bytes) -> None:
    """把一个回合的请求与响应写进日志。

    参数就是链路上的原文：raw_text 是收到的正文，body 是马上要发出去的字节。
    除 digest 档外不做任何加工。
    """
    if LOG_VERBOSITY == "digest":
        request_text = _request_digest(payload)
        response_text = body.decode("utf-8", errors="replace")
    else:
        request_text = raw_text
        response_text = body.decode("utf-8", errors="replace")
    # 一次 info 写完整个区块，保证 REQUEST / RESPONSE 不会被别的日志插在中间。
    LOGGER.info(
        "%s\n>>> REQUEST\n%s\n<<< RESPONSE\n%s\n%s",
        _rule(_round_title(payload)), request_text, response_text, _rule(),
    )


class Handler(BaseHTTPRequestHandler):
    """每来一次 POST 就决策一个回合。"""

    def do_POST(self) -> None:
        # 按 Content-Length 读完整个请求体（请求体可能几千字节）。
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)

        try:
            text = raw.decode("utf-8")
            payload = json.loads(text)
        except Exception:
            # 报文连 JSON 都不是：正常日志记不了，就把原始内容前几百字节记下来。
            LOGGER.exception("请求体不是合法 JSON：%r", raw[:RAW_LOG_LIMIT])
            self._send(json.dumps(empty_response(), ensure_ascii=False).encode("utf-8"))
            return

        try:
            response = decide(payload)
        except Exception:
            # 关键容错：无论报文多畸形、策略里出什么异常，都必须回一个合法响应，
            # 绝不能让进程崩掉或卡住，否则后面所有回合都拿不到指令。
            LOGGER.exception("decision failed")
            response = empty_response()

        # 只序列化这一次：记进日志的就是马上要发出去的那串字节。
        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        if LOG_VERBOSITY != "off":
            _log_round(payload, text, body)
        self._send(body)

    def _send(self, body: bytes) -> None:
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
