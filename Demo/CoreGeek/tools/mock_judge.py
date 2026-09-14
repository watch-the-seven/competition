"""HTTP 级联调：像判题器一样，用真实 POST 逐回合驱动选手程序。

与 smoke.py 的区别：smoke.py 直接在进程内调用 decide()，
本脚本把选手程序当成独立进程拉起来（默认 `bash run.sh <port>`），
走完整的 socket / HTTP / JSON 链路，并逐条校验响应是否符合《接口文档》。

    python3 tools/mock_judge.py

校验内容：
  1. 传输层：连接建立耗时、请求->响应耗时（接口文档第 8 章：建连 10s、响应 5s）
  2. 报文层：顶层三字段、动作码合法、必填字段齐全、targetPos 形状、
     controllerId 为字符串、num 为整数、多目标武器的目标数 == 等级
  3. 语义层：指令指向的角色/单位必须存在；prompt / executeCmd 只在任务期间使用
  4. 业务层：第 70 回合 3 火箭 + 10 墙 + 三人就位；夜晚开火不超射程
  5. 健壮性：畸形报文不让进程崩溃；整场结束后进程仍在

为什么区分「协议错误」和「语义告警」：接口文档第 8 章规定，累计 5 次异常
（超时 / 格式错误 / 指令错误）就会被停止调度，所以报文层面的错误是致命的；
而「操控者离武器太远」这类属于「指令执行失败」，只判该条无效、不计异常。
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

# 复用 smoke 里的世界模拟与目标校验，避免两套测试各自维护一份地图。
from smoke import World, check_goals, cheb            # noqa: E402

# ---------------------------------------------------------------- 协议常量
TOP_KEYS = ("roleCommandMap", "prompt", "executeCmd")

# 接口文档 2.3 动作码全集 + 2.2 各动作的必填字段。
# 少写一个必填字段就是「指令错误」，会计入队伍异常次数，所以这里必须严格。
ACTION_REQUIRED: dict[str, tuple[str, ...]] = {
    "move": ("targetPos",),
    "attack": ("targetPos", "controllerId"),
    "sell": ("name",),
    "buy": ("name",),
    "build": ("name", "targetPos"),
    "remove": ("targetPos",),
    "acceptTask": (),
    "submitAnswer": ("taskAnswer",),
    "summonTreasure": ("targetPos", "item"),
    "use": ("name",),
    "drop": ("name",),
    "collect": ("targetPos",),
}
# 任务书 4.5.4：加特林/火箭按等级传多个落点，电磁狙击炮恒为 1 个
MULTI_TARGET_TOWERS = ("gatling", "rocket")
BUILD_NAMES = ("gatling", "railgun", "rocket", "wall")
CONNECT_LIMIT_S = 10.0      # 接口文档第 8 章
RESPONSE_LIMIT_S = 5.0

# 用高矿价跑主场景，保证「攒够 200 金 -> 买 BOSS 令」这条链路一定会被走到。
PRICES = {"stone": 1, "iron": 3, "copper": 25}


# ------------------------------------------------------------------ 进程
def wait_ready(port: int, timeout: float = 10.0) -> float | None:
    """轮询直到端口可连接，返回建连耗时（秒）；超时返回 None。

    这对应接口文档里「建立连接过程超过 10 秒判超时」那条限制。
    """
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return time.perf_counter() - started
        except OSError:
            time.sleep(0.05)          # 进程还在启动，稍后再试
    return None


def launch(command: list[str], log_path: Path) -> subprocess.Popen:
    """按判题器的方式拉起选手程序，stdout/stderr 落到临时日志文件。"""
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
    )


# ------------------------------------------------------------------ 收发
def post_json(port: int, payload: Any, path: str = "/") -> tuple[int, Any, bytes, float]:
    """POST 一个回合，返回 (HTTP 状态码, 解析后的 JSON, 原始字节, 耗时秒)。

    payload 传 bytes 就会原样发送（用来测畸形报文），
    否则按 JSON 序列化。4xx/5xx 也要把 body 读出来，方便看原因。
    """
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=RESPONSE_LIMIT_S + 5) as reply:
            raw = reply.read()
            status = reply.status
    except urllib.error.HTTPError as error:                 # 4xx/5xx 也要拿到 body
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None                  # 不是合法 JSON，交给调用方报错
    return status, parsed, raw, elapsed


# ------------------------------------------------------------------ 校验
def validate(response: Any, world: World, payload: dict) -> tuple[list[str], list[str]]:
    """校验一份响应，返回 (协议错误, 语义告警)。

    协议错误 = 接口文档第 8 章里的「异常」，累计 5 次就会被停止调度；
    语义告警 = 指令本身合法但大概率不会生效，只提示不判失败。
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(response, dict):
        return ["响应不是合法 JSON 对象"], warnings

    # ---- 顶层结构（接口文档 2.1）----
    for key in TOP_KEYS:
        if key not in response:
            errors.append(f"缺少顶层字段 {key}")
    commands = response.get("roleCommandMap")
    if not isinstance(commands, dict):
        return errors + ["roleCommandMap 不是对象"], warnings
    for key in ("prompt", "executeCmd"):
        if key in response and not isinstance(response[key], str):
            errors.append(f"{key} 不是字符串")

    # 攻击指令的 key 是武器 id，需要拿它的等级来判断落点个数。
    towers = {
        unit_id: unit for unit_id, unit in world.units.items()
        if unit["roleType"] in MULTI_TARGET_TOWERS or unit["roleType"] == "railgun"
    }

    for key, command in commands.items():
        if not isinstance(command, dict):
            errors.append(f"{key}: 指令不是对象")
            continue
        action = command.get("action")
        if action not in ACTION_REQUIRED:
            errors.append(f"{key}: 未知动作码 {action!r}")
            continue
        # 必填字段一张表查完（接口文档 2.2）
        for field in ACTION_REQUIRED[action]:
            if field not in command:
                errors.append(f"{key}: {action} 缺少必填字段 {field}")

        # targetPos 必须是非空的 [{x:int, y:int}, ...]
        targets = command.get("targetPos")
        if targets is not None:
            if not isinstance(targets, list) or not targets:
                errors.append(f"{key}: targetPos 必须是非空数组")
            else:
                for item in targets:
                    if not isinstance(item, dict) or "x" not in item or "y" not in item:
                        errors.append(f"{key}: targetPos 元素不是 {{x,y}}")
                        break
                    if not isinstance(item["x"], int) or not isinstance(item["y"], int):
                        errors.append(f"{key}: targetPos 坐标不是整数")
                        break

        # 若干字段的类型约定（接口文档 2.2）
        if "controllerId" in command and not isinstance(command["controllerId"], str):
            errors.append(f"{key}: controllerId 必须是字符串")
        if "num" in command and not isinstance(command["num"], int):
            errors.append(f"{key}: num 必须是整数")
        if "item" in command and not isinstance(command["item"], list):
            errors.append(f"{key}: item 必须是数组")
        if action == "build" and command.get("name") not in BUILD_NAMES:
            errors.append(f"{key}: build name={command.get('name')!r} 不是合法建筑名")

        # ---- 语义检查：指令得指向一个真实存在的单位 ----
        try:
            unit_id = int(key)
        except (TypeError, ValueError):
            errors.append(f"{key}: 角色 key 不是整数")
            continue
        if unit_id not in world.units:
            errors.append(f"{key}: 指令指向不存在/已阵亡的单位")
            continue

        if action == "attack":
            tower = towers.get(unit_id)
            # 多目标武器的落点数必须等于当前等级（火箭 level1 -> 1 个落点）
            if tower is not None and isinstance(targets, list):
                expected = 1 if tower["roleType"] == "railgun" else max(1, tower["level"])
                if len(targets) != expected:
                    errors.append(
                        f"{key}: {tower['roleType']} level{tower['level']} "
                        f"需要 {expected} 个落点，实际 {len(targets)}"
                    )
            # 操控者必须真实存在、且紧贴武器（任务书 4.4）
            controller = command.get("controllerId")
            if isinstance(controller, str) and tower is not None:
                try:
                    cid = int(controller)
                except ValueError:
                    cid = -1
                actor = world.units.get(cid)
                if actor is None:
                    errors.append(f"{key}: controllerId={controller} 不存在")
                elif cheb(actor["pos"], tower["pos"]) > 1:
                    # 这类属于「指令执行失败」，只判无效、不计队伍异常，所以只告警。
                    warnings.append(
                        f"{key}: 操控者 {cid} 在 {actor['pos']}，"
                        f"离武器 {tower['pos']} 超过 1 格（该指令会被判无效）"
                    )

    # prompt / executeCmd 只在任务期间可用：
    # executeCmd 越界属于协议错误；prompt 越界会白占每天 3 次的 LLM 额度，只告警。
    if not world.phase_task:
        if response.get("executeCmd"):
            errors.append("非任务期间却下发了 executeCmd")
        if response.get("prompt"):
            warnings.append("非任务期间调用了 LLM（占用每日 3 次额度）")
    return errors, warnings


# ------------------------------------------------------------------ 场景
def run_match(port: int, launch_cmd: list[str], rounds: int = 130,
              seed: int = 7) -> dict:
    """场景 1：拉起进程，用真实 POST 打满 130 回合，返回一份体检报告。"""
    log_path = Path(tempfile.mkstemp(prefix="judge_agent_", suffix=".log")[1])
    process = launch(launch_cmd, log_path)
    report: dict[str, Any] = {
        "port": port, "launch": " ".join(launch_cmd),
        "connect_s": None, "latencies": [], "errors": [], "warnings": [],
        "rounds": 0, "alive": False, "log": log_path,
    }
    try:
        connect = wait_ready(port)
        report["connect_s"] = connect
        if connect is None:
            report["errors"].append("10 秒内无法建立连接")
            return report

        world = World(seed=seed, vendor_prices=PRICES)
        for round_no in range(1, rounds + 1):
            world.round_no = round_no
            world.last_ok = {unit_id: True for unit_id in world.units}
            world.tick_start()
            payload = world.payload()
            # 关键区别：这里是真正的网络往返，不是进程内函数调用。
            status, response, _raw, elapsed = post_json(port, payload)
            report["latencies"].append(elapsed)
            report["rounds"] += 1
            if status != 200:
                report["errors"].append(f"r{round_no}: HTTP {status}")
                break
            errors, warnings = validate(response, world, payload)
            report["errors"].extend(f"r{round_no}: {item}" for item in errors)
            report["warnings"].extend(f"r{round_no}: {item}" for item in warnings)
            if errors:
                break                    # 报文都错了，后面继续跑没意义
            world.apply(response)
            world.tick_end()

        # 跑完之后做业务层断言（复用 smoke 的第 70 回合快照校验）。
        rocket_count, wall_count, goal_problems = check_goals(world)
        report["rockets"] = rocket_count
        report["walls"] = wall_count
        report["errors"].extend(goal_problems)
        report["events"] = dict(world.events)
        report["final_gold"] = world.gold
        report["alive"] = process.poll() is None
        if not report["alive"]:
            report["errors"].append("进程在比赛结束前退出")
    finally:
        process.terminate()              # 无论成败都要收掉子进程
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return report


def probe_keepalive(port: int, launch_cmd: list[str], rounds: int = 6) -> dict:
    """场景 2：用同一个 TCP 连接连发多回合，验证服务端支持长连接。

    如果判题器用连接池复用连接，而服务端每次都关连接，就会表现为随机超时。
    """
    log_path = Path(tempfile.mkstemp(prefix="judge_ka_", suffix=".log")[1])
    process = launch(launch_cmd, log_path)
    result: dict[str, Any] = {"ok": False, "detail": "", "rounds": 0}
    try:
        if wait_ready(port) is None:
            result["detail"] = "无法建立连接"
            return result
        world = World(seed=3, vendor_prices=PRICES)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        for round_no in range(1, rounds + 1):
            world.round_no = round_no
            world.last_ok = {unit_id: True for unit_id in world.units}
            world.tick_start()
            body = json.dumps(world.payload()).encode("utf-8")
            # 复用同一个 connection，并显式声明 keep-alive。
            connection.request(
                "POST", "/", body=body,
                headers={"Content-Type": "application/json",
                         "Content-Length": str(len(body)),
                         "Connection": "keep-alive"},
            )
            reply = connection.getresponse()
            raw = reply.read()
            response = json.loads(raw.decode("utf-8"))
            world.apply(response)
            world.tick_end()
            result["rounds"] += 1
        connection.close()
        result["ok"] = True
    except Exception as error:                      # noqa: BLE001
        # 拿不到第二回合的响应（例如服务端已关连接）就会走到这里。
        result["detail"] = f"{type(error).__name__}: {error}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return result


def probe_robustness(port: int, launch_cmd: list[str]) -> list[tuple[str, str]]:
    """场景 3：畸形输入不应让进程崩溃（判题器可能补发/重发报文）。"""
    log_path = Path(tempfile.mkstemp(prefix="judge_rb_", suffix=".log")[1])
    process = launch(launch_cmd, log_path)
    outcomes: list[tuple[str, str]] = []
    try:
        if wait_ready(port) is None:
            return [("启动", "无法建立连接")]
        world = World(seed=3, vendor_prices=PRICES)
        world.round_no = 1
        world.last_ok = {}
        good = json.dumps(world.payload()).encode("utf-8")

        # 1) 非法 JSON：应当回空指令而不是 500/崩溃
        status, parsed, _raw, _t = post_json(port, b"{not-json", path="/")
        outcomes.append((
            "非法 JSON",
            f"HTTP {status}，响应={parsed}",
        ))
        # 2) 空对象：所有字段缺失
        status, parsed, _raw, _t = post_json(port, {}, path="/")
        outcomes.append(("空对象", f"HTTP {status}，响应={parsed}"))
        # 3) 非根路径：接口文档没规定路径，任何路径都应当能处理
        status, parsed, _raw, _t = post_json(port, good, path="/some/other/path")
        outcomes.append((
            "非根路径 POST",
            f"HTTP {status}，roleCommandMap 键数="
            f"{len(parsed.get('roleCommandMap', {})) if isinstance(parsed, dict) else 'N/A'}",
        ))
        # 4) 经历上面三次异常输入后进程是否还活着
        outcomes.append(("进程存活", "是" if process.poll() is None else "否（已崩溃）"))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return outcomes


def raw_exchange(port: int, launch_cmd: list[str]) -> list[str]:
    """打印一次完整的原始 HTTP 往返（相当于 Postman 的 Raw 视图）。"""
    log_path = Path(tempfile.mkstemp(prefix="judge_raw_", suffix=".log")[1])
    process = launch(launch_cmd, log_path)
    lines: list[str] = []
    try:
        if wait_ready(port) is None:
            return ["无法建立连接"]
        world = World(seed=7, vendor_prices=PRICES)
        world.round_no = 1
        world.last_ok = {}
        body = json.dumps(world.payload(), ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST", "/", body=body,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body))},
        )
        reply = connection.getresponse()
        payload = reply.read()
        # 手工拼出请求行/请求头/响应行/响应头，方便肉眼核对。
        lines.append(">>> POST / HTTP/1.1")
        lines.append(f">>> Content-Type: application/json")
        lines.append(f">>> Content-Length: {len(body)}")
        lines.append(f">>> (请求体 {len(body)} 字节，节选)")
        lines.append(json.dumps(json.loads(body.decode("utf-8")), ensure_ascii=False)[:220])
        lines.append("")
        lines.append(f"<<< HTTP {reply.status} {reply.reason}")
        for name, value in reply.getheaders():
            lines.append(f"<<< {name}: {value}")
        lines.append(f"<<< (响应体 {len(payload)} 字节)")
        lines.append(json.dumps(json.loads(payload.decode("utf-8")), ensure_ascii=False))
        connection.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return lines


# ------------------------------------------------------------------ 主流程
def main() -> int:
    """依次跑三个场景 + 打印一次原始往返，返回进程退出码。"""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    launch_cmd = ["bash", "run.sh", str(port)]   # 与接口文档的样例启动方式一致
    failures: list[str] = []

    # ---- 场景 1：完整一天 ----
    print("=" * 74)
    print(f"场景 1：完整一天 130 回合（启动命令：{' '.join(launch_cmd)}）")
    print("=" * 74)
    report = run_match(port, launch_cmd)
    connect = report["connect_s"]
    latencies = report["latencies"]
    print(f"建连耗时      : {connect:.3f}s" if connect else "建连耗时      : 失败")
    if latencies:
        ordered = sorted(latencies)
        print(f"完成回合数    : {report['rounds']}/130")
        print(f"单回合往返    : 平均 {sum(latencies)/len(latencies)*1000:.1f}ms / "
              f"最大 {max(latencies)*1000:.1f}ms / "
              f"P95 {ordered[int(len(ordered)*0.95)-1]*1000:.1f}ms")
        print(f"限值对照      : 建连 {CONNECT_LIMIT_S:.0f}s、响应 {RESPONSE_LIMIT_S:.0f}s "
              f"-> {'全部满足' if max(latencies) < RESPONSE_LIMIT_S else '存在超时'}")
    print(f"第 70 回合    : 火箭 {report.get('rockets')} 座 / 城墙 {report.get('walls')} 面")
    events = report.get("events", {})
    print(f"任务/金币     : 提交 {events.get('submitted')} 个任务，"
          f"金币首次>=200 于第 {events.get('gold_200')} 回合，"
          f"BOSS 令买于第 {events.get('boss_bought')} 回合、用于第 {events.get('boss_used')} 回合")
    print(f"夜晚          : 开火 {events.get('attacks')} 次 / 伤害 {events.get('damage')} / "
          f"超射程 {events.get('bad_shots')}")
    print(f"进程存活      : {'是' if report['alive'] else '否'}")
    if report["errors"]:
        failures.extend(report["errors"])
        print(f"协议错误 {len(report['errors'])} 条（前 8 条）：")
        for item in report["errors"][:8]:
            print("   -", item)
    else:
        print("协议错误      : 0 条（响应完全符合接口文档）")
    if report["warnings"]:
        print(f"语义告警 {len(report['warnings'])} 条（前 5 条）：")
        for item in report["warnings"][:5]:
            print("   -", item)

    # ---- 场景 2：长连接 ----
    print()
    print("=" * 74)
    print("场景 2：长连接（同一 TCP 连接连发 6 回合）")
    print("=" * 74)
    keepalive = probe_keepalive(port + 1, ["bash", "run.sh", str(port + 1)])
    print(f"结果          : {'支持 keep-alive' if keepalive['ok'] else '不支持'}"
          f"（完成 {keepalive['rounds']} 回合）")
    if not keepalive["ok"]:
        print(f"细节          : {keepalive['detail']}")
        failures.append("长连接不可用：" + keepalive["detail"])

    # ---- 场景 3：健壮性 ----
    print()
    print("=" * 74)
    print("场景 3：健壮性（畸形/异常报文）")
    print("=" * 74)
    for title, detail in probe_robustness(port + 2, ["bash", "run.sh", str(port + 2)]):
        print(f"{title:<14}: {detail}")

    # ---- 附：原始 HTTP 往返 ----
    print()
    print("=" * 74)
    print("原始 HTTP 往返（Postman Raw 等价视图）")
    print("=" * 74)
    for line in raw_exchange(port + 3, ["bash", "run.sh", str(port + 3)]):
        print(line)

    print()
    if failures:
        print("==== 失败 ====")
        for item in failures:
            print(" -", item)
        return 1
    print("==== HTTP 级联调全部通过 ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
