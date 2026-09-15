"""单回合诊断：拿一份判题器真实的 Request 报文，讲清楚「为什么某个角色没动作」。

用法：
    python3 tools/diagnose.py <request.json> [更多 request.json ...]
    python3 tools/diagnose.py ../../docs/request.txt

它会打印：
  1. 报文概况：回合、昼夜、金币、阵营、地图尺寸；
  2. 我方单位逐条列出（id / roleType / health / 坐标 / 背包），
     并明确指出哪些单位**不会被调度**以及原因；
  3. 解析出的任务点、基地布局（C 形护罩选址 / 火箭位 / 站位）；
  4. 每个可调度角色本回合拿到的岗位与最终指令；
  5. 针对「开拓者一直不动」这类问题给出结论性提示。

只读不改，可以安全地对比赛日志里存下来的报文反复运行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import brain                                    # noqa: E402
from agent import memory as memory_module                  # noqa: E402
from agent.protocol import (                               # noqa: E402
    CONTROLLABLE_TYPES,
    KNOWN_KINDS,
    Turn,
)


def report(path: Path) -> None:
    print("=" * 74)
    print(f"报文：{path}")
    print("=" * 74)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"  读取失败：{type(error).__name__}: {error}")
        return

    turn = Turn.load(payload)
    team = payload.get("teamOur") or {}
    print(f"回合 {turn.round_no}（{'白天' if turn.is_day else '夜晚'}） "
          f"金币 {turn.gold} 阵营 {turn.our_type or '(缺 type)'} "
          f"地图 {turn.width}x{turn.height}")

    # ---- 1. 我方单位逐条列出来，并标出谁不会被调度 ----
    controllable = {unit.unit_id for unit in turn.controllable()}
    # 记下报文里哪些 role 根本没带 health 字段（会被按「存活」兜底处理）。
    missing_health = {
        int(role.get("id") or 0)
        for role in (team.get("roles") or ())
        if "health" not in role
    }
    print(f"\n我方单位 {len(turn.ours)} 个：")
    print(f"  {'id':>6}  {'roleType':<10} {'health':>7}  {'坐标':<14} 背包  可调度?")
    for unit in turn.ours:
        why = ""
        if unit.unit_id not in controllable:
            if unit.kind not in KNOWN_KINDS:
                why = "→ roleType 未识别，不会被调度"
            elif unit.kind not in CONTROLLABLE_TYPES:
                why = "→ 是建筑/武器，本来就不下指令"
            elif unit.health <= 0:
                why = "→ health<=0，被当成阵亡"
            else:
                why = "→ 未被调度，原因未知"
        elif unit.unit_id in missing_health:
            why = "← 报文缺 health，已按存活兜底"
        print(f"  {unit.unit_id:>6}  {unit.kind:<10} {unit.health:>7}  "
              f"{str(unit.pos):<14} {len(unit.backpack):>3}   "
              f"{'是' if unit.unit_id in controllable else '否'} {why}")

    unknown = sorted({unit.kind for unit in turn.ours} - KNOWN_KINDS)
    if unknown:
        print(f"  ⚠ 未识别的 roleType：{unknown}")

    # ---- 2. 任务点 ----
    print(f"\n任务点 {len(turn.player_tasks)} 个：")
    if not turn.player_tasks:
        print("  ⚠ 报文里没有 playerTasks —— 开拓者将无任务可做，只会走回站位待命")
    for task in turn.player_tasks:
        print(f"  {task.task_type:<12} 坐标 {task.pos} 可接={task.is_valid} "
              f"冷却={task.cooldown} 金币奖励={task.gold_reward}")

    # ---- 3. 实际跑一遍决策，看每个角色拿到什么 ----
    memory_module.MEMORY.__init__()          # 单回合诊断：用干净的记忆
    response = brain.decide(payload)
    plan = memory_module.MEMORY.plan
    print("\n基地布局：")
    if plan is None:
        print("  ⚠ 没有基地(station)，无法规划布局 -> 所有归位/站位都会失效")
    else:
        print(f"  方位={plan.corner} 镜像={plan.mirrored}")
        print(f"  城墙位({len(plan.wall_sites)})={[str(p) for p in plan.wall_sites]}")
        # 站位按 unit_id 末两位绑定，这里直接摊开成「谁站哪、操哪座」方便核对。
        for slot in ("10", "11", "12"):
            index = plan.slot_index.get(slot)
            if index is None:
                continue
            print(f"  槽 {slot}: 站位 {plan.stands[index]} -> 炮台 {plan.tower_sites[index]}")

    print("\n本回合每个可调度角色的决定：")
    jobs = memory_module.MEMORY.last_jobs
    for unit in turn.controllable():
        command = response["roleCommandMap"].get(
            unit.unit_id, response["roleCommandMap"].get(str(unit.unit_id))
        )
        job = jobs.get(unit.unit_id, "(夜晚无岗位)" if not turn.is_day else "?")
        print(f"  {unit.unit_id} kind={unit.kind:<8} 岗位={job:<8} "
              f"指令={json.dumps(command, ensure_ascii=False) if command else '（无）'}")
    if response.get("prompt"):
        print(f"  顶层 prompt：{response['prompt'][:80]}...")
    if response.get("executeCmd"):
        print(f"  顶层 executeCmd：{response['executeCmd']}")

    # ---- 4. 结论性提示 ----
    print("\n结论：")
    idle = [u for u in turn.controllable()
            if response["roleCommandMap"].get(u.unit_id) is None
            and response["roleCommandMap"].get(str(u.unit_id)) is None]
    if not turn.controllable():
        print("  ✗ 没有任何可调度角色：所有角色都不会行动，先看上面的 health / roleType")
    elif idle:
        print(f"  本回合无指令的角色：{[u.unit_id for u in idle]}")
        print("  （若该角色连续多回合都无指令，且岗位是 task/solve，请把多份报文一起跑）")
    else:
        print("  ✓ 所有可调度角色本回合都有指令")

    pioneer = turn.pioneer()
    if pioneer is None:
        print("  ✗ 报文里找不到开拓者：它要么不在 roles 里，要么 roleType/health 不对")
    elif turn.is_day:
        task_state = memory_module.MEMORY.task
        print(f"  开拓者 {pioneer.unit_id}：任务阶段={task_state.phase} "
              f"第{task_state.index}个任务点 已完成{task_state.done}个")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for name in sys.argv[1:]:
        report(Path(name))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
