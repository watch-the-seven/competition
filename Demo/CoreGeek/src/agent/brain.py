"""第一天（前 130 回合）完整解题策略。

白天目标（需求）
  1. 工人 A：采 10 个石头 -> 按 C 形护罩砌 10 面墙 -> 去采最近的铜/铁矿并卖给小贩；
  2. 工人 B：开局建 3 座火箭发射台 -> 去采铜/铁矿并卖给小贩；
  3. 开拓者：完成 2 个自进化任务（任务点接取 + 沙盒执行 + 提交答案）；
  4. 全队金币攒到 200 后到武器商店买 BOSS 召唤令并立刻使用；
  5. 第 70 回合（白天最后一回合）三个角色全部站在火箭发射台旁边。

夜晚目标（第 71~130 回合）
  操控 3 台火箭发射台射击机器人，优先打「离我方基地最近」的敌人，
  并按火箭的落点伤害模型（中心 20 / 周围 8 格 10）求单回合最大伤害。

代码结构（自上而下）：
  调参常量 -> decide 入口 -> 白天分工 _day -> 各岗位实现 -> 通用工具 -> 夜晚逻辑
"""

from __future__ import annotations

import logging
from typing import Any

from . import grid
from .memory import MEMORY
from .protocol import (
    BOSS_ORDER,
    CONTROLLABLE_TYPES,
    COPPER,
    DAY_ONE_LAST_ROUND,
    IRON,
    KNOWN_KINDS,
    PIONEER,
    ROCKET,
    ROCKET_CENTER_DAMAGE,
    ROCKET_SPLASH_DAMAGE,
    STONE,
    TRADE_ORES,
    WALL,
    WALL_MATERIAL,
    Pos,
    Turn,
    Unit,
    accept_task_command,
    attack_command,
    build_command,
    buy_command,
    collect_command,
    distance,
    empty_response,
    move_command,
    neighbours,
    sell_command,
    submit_answer_command,
    use_command,
)

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------- 调参
# 这些数字是策略的「手感开关」，改这里就能调整行为，不用动逻辑。
SELL_BATCH = 20            # 一趟最多背 20 块（= 需求里说的「采两个矿」）再去卖
MINE_YIELD = 10            # 任务书 4.1：每个矿可采 10 次
RETURN_SLACK = 1           # 归位留 1 回合余量，防止最后一回合被碰撞卡住
BOSS_USE_DEADLINE = 68     # 必须在这回合前完成购买+使用，夜晚才吃得到效果
ORE_DEADLINE = DAY_ONE_LAST_ROUND - RETURN_SLACK - 1   # 一趟采矿循环最晚收尾回合
MIN_YIELD = 3              # 少于 3 块就不值得专门跑一趟
MAX_CHAIN_LEG = 8          # 背着货时，只顺路再采一次近距离的矿
TASK_COUNT = 2             # 第一天要完成的自进化任务数
TASK_BUDGET = 14           # 预估单个任务占用的回合数（含往返与 LLM 往返）
# 需求指定的挖矿分工：尾号 10 的工人挖铁、尾号 12 的工人挖铜。
# 分开矿种还有个附带好处：两个工人天然不会抢同一个矿。
SLOT_ORE: dict[str, str] = {"10": IRON, "12": COPPER}
TASK_LLM_MAX = 6           # 单个任务最多几轮 LLM 往返
TASK_CMD_MAX = 4           # 单个任务最多几次沙盒执行
# 下面三个是「防僵死」超时：判题器/LLM/沙盒任何一环不回包时，
# 都必须在有限回合内放弃或兜底，绝不能让开拓者整局僵在原地。
LLM_WAIT_LIMIT = 3         # 发出 prompt 后最多等几回合 llmResp
ACCEPT_WAIT_LIMIT = 3      # 发出 acceptTask 后最多等几回合任务原文
TASK_STALL_LIMIT = 5       # 任何阶段连续几回合毫无进展就放弃该任务点
TASK_WAIT_ROUNDS = 10      # 任务点迟迟不下发时，最多等几个回合再放开拓者去干别的
FIRE_MIN_VALUE = 40.0      # 一轮齐射至少打出「2 个机器人份」的伤害才开火
URGENT_BASE_DISTANCE = 9   # 敌人已经逼近基地到这个距离就无条件开火


# --------------------------------------------------------------------- 入口
def _log_roster(turn: Turn) -> None:
    """第一次收到报文时，把「解析出了哪些我方单位」打进日志。

    排查「某个角色一直不动」时第一件事就是看它有没有被解析出来、kind 对不对、
    health 是不是 0——这三样任一出问题，该角色就会被排除在调度之外。
    """
    LOGGER.info(
        "我方单位(id/kind/health): %s",
        [(unit.unit_id, unit.kind, unit.health) for unit in turn.ours],
    )
    LOGGER.info(
        "可调度角色: %s | 任务点: %s",
        [unit.unit_id for unit in turn.controllable()],
        [str(task.pos) for task in turn.player_tasks],
    )
    # 下面几条是最容易导致「角色不动」的报文问题，直接给出告警。
    unknown = sorted({unit.kind for unit in turn.ours} - KNOWN_KINDS)
    if unknown:
        LOGGER.warning("报文里出现未识别的 roleType: %s（这些单位不会被调度）", unknown)
    if turn.station() is None:
        LOGGER.warning("报文里没有基地(station)：布局与归位都会失效")
    if turn.pioneer() is None:
        LOGGER.warning("报文里没有可用的开拓者(pioneer)")
    if not turn.player_tasks:
        LOGGER.warning("报文里没有 playerTasks：开拓者将无任务可做")


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    """判题器每回合调用一次：吃 Request，吐 Response。

    固定流程：解析报文 -> 消化上一回合的结果 -> 按昼夜分派 -> 记录本回合指令。
    整个函数在 MEMORY.lock 内执行，避免并发请求把跨回合状态改乱。
    """
    turn = Turn.load(payload)
    with MEMORY.lock:
        if MEMORY.last_round == 0:
            _log_roster(turn)               # 整局只打一次，便于赛后排查
        MEMORY.begin(turn)                      # 先解释上一回合哪些动作失败了
        response = empty_response()             # 默认不下任何指令
        if turn.is_day:
            _day(turn, response)
        else:
            _night(turn, response)
        MEMORY.remember(response["roleCommandMap"], turn)   # 供下一回合复盘
        return response


# ===================================================================== 白天
def _day(turn: Turn, response: dict[str, Any]) -> None:
    """白天主循环：给每个角色分配一个岗位，再交给对应的处理函数。"""
    commands = response["roleCommandMap"]
    claimed: set[Pos] = set()      # 本回合已被别的角色预定要去的格子，避免撞车
    jobs = _assign_jobs(turn)
    MEMORY.last_jobs = dict(jobs)  # 存一份便于排查（mock judge 会用到）

    for role in turn.controllable():
        job = jobs.get(role.unit_id, "hold")
        # 一个岗位一个处理函数；hold 不匹配任何分支，等于原地待命。
        if job == "task":
            _task_step(turn, role, claimed, commands, response)
        elif job == "boss":
            _boss_step(turn, role, claimed, commands)
        elif job == "return":
            _return_step(turn, role, claimed, commands)
        elif job == "sell":
            _sell_step(turn, role, claimed, commands)
        elif job == "towers":
            _tower_step(turn, role, claimed, commands)
        elif job == "stone":
            _stone_step(turn, role, claimed, commands)
        elif job == "walls":
            _wall_step(turn, role, claimed, commands)
        elif job == "ore":
            _ore_step(turn, role, claimed, commands)


def _assign_jobs(turn: Turn) -> dict[int, str]:
    """决定每个角色本回合干什么，优先级：归位 > BOSS 采购 > 常规分工。

    「归位」排最前，是因为需求硬性要求第 70 回合三人都在火箭旁；
    一旦时间不够就必须立刻往回赶，其它事都可以放弃。
    """
    jobs: dict[int, str] = {}
    owner = _boss_owner(turn)      # 可能为 None；有金主时他会一直占用 boss 岗位

    # 第一轮：先判定谁必须回防。
    for role in turn.controllable():
        if role.unit_id == owner:
            continue               # 采购员例外，见下面
        if _must_return(turn, role):
            # 回防路上顺手把矿卖掉（前提是卖完还赶得上）。
            jobs[role.unit_id] = "sell" if _can_still_sell(turn, role) else "return"

    if owner is not None:
        jobs[owner] = "boss"

    # 第二轮：没被归位/采购占用的角色，按自己的阶段任务安排。
    for role in turn.controllable():
        if role.unit_id in jobs:
            continue
        if role.kind == PIONEER:
            jobs[role.unit_id] = "task" if _pioneer_busy(turn) else "return"
        elif role.unit_id == MEMORY.tower_worker:
            # 先造满 3 座火箭，再去挣金币。
            jobs[role.unit_id] = (
                "towers" if not MEMORY.rockets_ready(turn) else _roam_job(turn)
            )
        elif role.unit_id == MEMORY.stone_worker:
            # 先采够石头，再砌墙，最后去挣金币。
            if not _stone_ready(turn, role):
                jobs[role.unit_id] = "stone"
            elif not MEMORY.walls_ready(turn):
                jobs[role.unit_id] = "walls"
            else:
                jobs[role.unit_id] = _roam_job(turn)
        else:
            jobs[role.unit_id] = _roam_job(turn)
    return jobs


def _pioneer_busy(turn: Turn) -> bool:
    """开拓者手头是否还有任务要做（没任务时就该放它去干别的，比如采购）。"""
    state = MEMORY.task
    if state.done >= TASK_COUNT or state.phase == "halt":
        return False
    if not MEMORY.task_points:
        # 任务点还没下发（判题器可能晚几回合才给）：再等等，
        # 否则会一上来就把开拓者抓去商店，任务下发后它已经走远了。
        return turn.round_no <= TASK_WAIT_ROUNDS
    return state.index < len(MEMORY.task_points)


def _roam_job(turn: Turn) -> str:
    """本回合该不该去采矿：金币够了就收工回防。"""
    if MEMORY.boss["phase"] == "done":
        return "return"                       # BOSS 令已到手，再挖也没用
    if turn.gold >= turn.boss_price():
        return "return"                       # 已经买得起，等采购流程走完即可
    return "ore"


# --------------------------------------------------------------------- 工人 B：3 火箭
def _tower_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """造火箭：挑一个还没建的武器位，走过去建。"""
    plan = MEMORY.plan
    if plan is None:
        return
    standing = {unit.pos for unit in MEMORY.rockets(turn)}
    pending = [
        pos
        for pos in plan.tower_sites
        if pos not in standing and pos not in MEMORY.bad_build
    ]
    if not pending:
        _return_step(turn, role, claimed, commands)   # 建完了就去干别的
        return
    # 先建最近的那个，省走路回合。
    target = min(pending, key=lambda pos: (distance(role.pos, pos), pos.x, pos.y))
    _build_at(turn, role, target, ROCKET, claimed, commands)


# --------------------------------------------------------------------- 工人 A：10 石 + 10 墙
def _stone_ready(turn: Turn, role: Unit) -> bool:
    """石头够不够砌完剩下的墙。

    注意是「够砌剩下的」，不是「永远要 10 块」——否则砌完第一面墙
    石头降到 9 块，工人就会跑回矿里补采，反而耽误砌墙。
    """
    return role.count(WALL_MATERIAL) >= len(MEMORY.outstanding_walls(turn))


def _stone_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """采石：石头够了就转去砌墙，否则走到石矿旁边一直采。"""
    if _stone_ready(turn, role):
        _wall_step(turn, role, claimed, commands)
        return
    if role.backpack_full:
        # 背包满：先丢掉矿石腾格子，保证石头能装进来。
        for item in TRADE_ORES:
            if role.count(item):
                commands[role.unit_id] = {"action": "drop", "name": item}
                return
    adjacent = _adjacent_mine(turn, role, STONE)
    if adjacent is not None:
        commands[role.unit_id] = collect_command(adjacent)   # 已就位，直接采
        return
    # 没就位：把「所有石矿的相邻空格」当作目标集合，走向最近的一个。
    mines = [
        pos
        for pos in turn.ore_mines(STONE)
        if pos not in MEMORY.bad_build
    ]
    if not mines:
        return
    blocked = set(turn.blocked(role)) | MEMORY.bad_build
    goals: set[Pos] = set()
    for mine in mines:
        goals |= _adjacent_free(turn, mine, blocked)
    _advance(turn, role, goals, claimed, commands)


def _wall_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """砌墙：按「离自己最近」的顺序把剩余城墙逐个建完。"""
    pending = MEMORY.order_candidates(turn, role)
    if not pending:
        _ore_step(turn, role, claimed, commands)     # 墙砌完了，转去采矿
        return
    if role.count(WALL_MATERIAL) <= 0:
        # 石头不够（例如有墙被拆、或建造被拒后重来），回去补采。
        _stone_step(turn, role, claimed, commands)
        return
    _build_at(turn, role, pending[0], WALL, claimed, commands)


def _build_at(
    turn: Turn,
    role: Unit,
    target: Pos,
    name: str,
    claimed: set[Pos],
    commands: dict[str, Any],
) -> None:
    """走到 target 旁边并建造。

    任务书 4.4：建造要求目标与自身距离一格内，所以目标是「站到它周围」，
    而不是「站到它上面」。非法格会被记进 bad_build，以后不再重试。
    """
    if not turn.land(target):
        MEMORY.bad_build.add(target)     # 目标本身不是合法空地，直接拉黑
        return
    blocked = set(turn.blocked(role)) | MEMORY.bad_build
    if role.pos == target:
        # 站在目标格上是建不了的，先挪到旁边一格。
        goals = _adjacent_free(turn, target, blocked) - claimed
        if goals:
            _advance(turn, role, goals, claimed, commands)
        return
    if distance(role.pos, target) <= 1:
        commands[role.unit_id] = build_command(target, name)     # 就位，建造
        claimed.add(target)
        return
    # 还没到位：走向目标周围的空格（优先没被别人预定的）。
    goals = _adjacent_free(turn, target, blocked) - claimed
    if not goals:
        goals = _adjacent_free(turn, target, blocked)
    _advance(turn, role, goals, claimed, commands, extra=blocked)


def _adjacent_mine(turn: Turn, role: Unit, ore: str) -> Pos | None:
    """找出角色旁边（距离 1）的某种矿；没有返回 None。

    注意排除「站在矿上」的情况——矿本身是障碍，站不上去。
    """
    mines = [
        pos
        for pos in turn.ore_mines(ore)
        if role.pos != pos and distance(role.pos, pos) <= 1
    ]
    if not mines:
        return None
    return min(mines, key=lambda pos: (distance(role.pos, pos), pos.x, pos.y))


# --------------------------------------------------------------------- 采矿卖钱
def _ore_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """采矿循环：去矿点 -> 采够 -> 去小贩卖掉 -> 再挑下一个矿。"""
    vendor = turn.vendor_pos()
    carried = {ore: role.count(ore) for ore in TRADE_ORES if role.count(ore)}
    total = sum(carried.values())

    target = MEMORY.ore_target.get(role.unit_id)
    if target is not None and target[0] not in turn.zones:
        target = None                      # 矿被采空后从地图上消失了
        MEMORY.ore_target.pop(role.unit_id, None)
    want = target[2] if target is not None else SELL_BATCH   # 这趟计划背到多少块

    # 该去卖了吗？三种情况：背够计划量 / 背包快满 / 这个矿已经采空了。
    # 「矿采空就立刻去卖」很重要：金币越早到账，开拓者越来得及在
    # 归位截止前把 BOSS 令买下来，而不是为了多凑几块矿把卖矿拖到入夜。
    exhausted = target is None
    if carried and vendor is not None and (
        exhausted or total >= want or role.backpack_full
    ):
        if _go_sell(turn, role, carried, claimed, commands):
            return

    if target is None:
        target = _pick_mine(turn, role)    # 身上没货了，挑一个来得及跑完的矿
        if target is not None:
            MEMORY.ore_target[role.unit_id] = target
    if target is None:
        # 没有来得及采的矿了：回防。
        _return_step(turn, role, claimed, commands)
        return

    mine, _ore, _want = target
    if role.pos != mine and distance(role.pos, mine) <= 1:
        commands[role.unit_id] = collect_command(mine)       # 已就位，采一块
        return
    blocked = set(turn.blocked(role)) | MEMORY.bad_build
    goals = _adjacent_free(turn, mine, blocked)
    if not _advance(turn, role, goals, claimed, commands, extra=blocked):
        # 走不到这个矿（被堵死/临时不可达），放弃它换下一个。
        MEMORY.ore_target.pop(role.unit_id, None)


def _go_sell(
    turn: Turn,
    role: Unit,
    carried: dict[str, int],
    claimed: set[Pos],
    commands: dict[str, Any],
) -> bool:
    """朝小贩走；已经在小贩旁边就卖掉最值钱的那种矿。返回是否下发了指令。"""
    vendor = turn.vendor_pos()
    if vendor is None or not carried:
        return False
    if distance(role.pos, vendor) <= 1:
        # 一次只卖一种，优先卖单价高的（铜 > 铁）。
        ore = max(carried, key=lambda item: (turn.vendor_prices.get(item, 0), item))
        commands[role.unit_id] = sell_command(ore, carried[ore])
        return True
    goals = _adjacent_free(turn, vendor, set(turn.blocked(role)))
    return _advance(turn, role, goals, claimed, commands)


def _sell_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """归位前把背包里的矿石卖掉；卖不掉就直接回防。"""
    carried = {ore: role.count(ore) for ore in TRADE_ORES if role.count(ore)}
    if not carried or not _go_sell(turn, role, carried, claimed, commands):
        _return_step(turn, role, claimed, commands)


def _preferred_ores(role: Unit) -> tuple[str, ...]:
    """这个角色该挖的矿种顺序：需求指定的那种优先，采不到再退而求其次。

    需求：尾号 10 的工人挖铁、尾号 12 的工人挖铜。这样两人不会抢同一个矿；
    万一地图上该矿种全都不合适（太远/来不及），才退让到另一种矿，避免干站着。
    """
    first = SLOT_ORE.get(str(role.unit_id)[-2:])
    if first is None:
        return TRADE_ORES
    return (first,) + tuple(ore for ore in TRADE_ORES if ore != first)


def _best_mine_of(
    turn: Turn,
    role: Unit,
    ore: str,
    carried: int,
    home_from_vendor: int,
    taken: set[Pos],
) -> tuple[Pos, str, int] | None:
    """在指定矿种里挑「金币收益 / 花费回合」最高的一个矿。"""
    vendor = turn.vendor_pos()
    assert vendor is not None
    price = turn.vendor_prices.get(ore, 0)
    if price <= 0:
        return None
    best: tuple[Pos, str, int] | None = None
    best_score = 0.0
    for mine in turn.ore_mines(ore):
        if mine in MEMORY.bad_build:
            continue
        leg_in = _rounds_to(turn, role, mine)
        if leg_in is None:
            continue
        if carried and leg_in > MAX_CHAIN_LEG:
            continue                       # 背着货别再跑远矿，先把货换成钱
        leg_out = _rounds_between(turn, mine, vendor)
        if leg_out is None:
            continue
        # 固定开销（去程 + 回程 + 交易 + 卖完回家）之后还剩多少回合可以采。
        slack = ORE_DEADLINE - (
            turn.round_no + leg_in + leg_out + 1 + home_from_vendor
        )
        room = SELL_BATCH - carried        # 批量上限的剩余空间
        want = min(MINE_YIELD, room, slack)
        if want < MIN_YIELD:
            continue                       # 采不了几块，不值得专门跑一趟
        score = price * want / (leg_in + want + leg_out + 1)
        if mine in taken:
            score *= 0.5                   # 另一个工人已经盯上的矿就别抢
        if score > best_score:
            best, best_score = (mine, ore, carried + want), score
    return best


def _pick_mine(
    turn: Turn, role: Unit
) -> tuple[Pos, str, int] | None:
    """挑一个「还来得及跑完」的矿，返回 (矿点, 矿种, 这趟准备背到多少块)。

    一次完整的采矿循环 = 走到矿 -> 采 N 块 -> 走到小贩 -> 卖 -> 走回站位，
    必须能在归位截止前收尾；否则宁可少采几块（把 N 调小）也不要半路丢货。
    先在自己负责的矿种里找；实在没有可行的，才退让到另一种矿。
    """
    vendor = turn.vendor_pos()
    stand = MEMORY.stand_for(role.unit_id)
    if vendor is None:
        return None
    # 万一没分到站位（例如武器位没配齐），按「回家 0 回合」乐观估算，
    # 不要因此彻底放弃采矿——否则整个白天一分钱都挣不到。
    home_from_vendor = _rounds_between(turn, vendor, stand) if stand else 0
    if home_from_vendor is None:
        return None
    # 另一个角色已经盯上的矿，评分打对折，避免两人抢同一个。
    taken = {
        pos
        for other, entry in MEMORY.ore_target.items()
        if other != role.unit_id
        for pos in (entry[0],)
    }
    carried = sum(role.count(ore) for ore in TRADE_ORES)

    for ore in _preferred_ores(role):
        found = _best_mine_of(turn, role, ore, carried, home_from_vendor, taken)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------- BOSS 召唤令
def _boss_owner(turn: Turn) -> int | None:
    """决定谁负责买 BOSS 召唤令；返回角色 id（None 表示暂时没人负责）。

    采购状态机（存在 MEMORY.boss）：
        phase: wait -> travel -> use_sent -> done
        owner: 当前采购员
        blocked: 曾经因为「再不走就赶不上就位」而放弃过采购

    选人口径（需求）：**开拓者做完两个任务后就去武器商店门口等**，
    金币一到 200 立刻买、下一回合立刻用，把「等钱」的时间压成 0。
    开拓者不可用（还在做任务 / 阵亡 / 已经放弃）时，才在金够 200 后
    派一个「来回都赶得上第 71 回合就位」的角色去补买。
    """
    state = MEMORY.boss
    phase = state["phase"]
    alive = {unit.unit_id for unit in turn.controllable()}

    if phase == "done":
        return None            # 已经买过且用过了，采购流程结束（否则会重复买第二张）

    # ---- 已有采购员：继续由他负责，直到用完 / 失联 / 必须回防 ----
    if phase != "wait":
        owner = state.get("owner")
        holder = next((u for u in turn.controllable() if u.unit_id == owner), None)
        if holder is None:
            state["phase"], state["owner"] = "wait", None        # 采购员失联，重选
        elif phase == "use_sent" and BOSS_ORDER not in holder.backpack:
            state["phase"] = "done"                              # 已用掉，收工
            return None
        elif _must_return(turn, holder):
            # 再不走就赶不上第 71 回合就位。宁可这次不买也不能缺席夜晚，
            # 并且记下 blocked，避免下回合又把同一个人派出去来回横跳。
            state["phase"], state["owner"], state["blocked"] = "wait", None, True
            LOGGER.info("放弃 BOSS 采购：角色 %s 必须回防", owner)
            return None
        else:
            return owner

    shop = turn.weapon_shop_pos()
    if shop is None:
        return None

    # ---- 首选：开拓者做完任务了，就让它先去门口等（钱不够也去） ----
    pioneer = turn.pioneer()
    if pioneer is not None and not _pioneer_busy(turn) and not state.get("blocked"):
        state["phase"], state["owner"] = "travel", pioneer.unit_id
        return pioneer.unit_id

    # ---- 次选：金已经够了，派个来回都赶得上的角色 ----
    if turn.gold < turn.boss_price() or state.get("blocked"):
        return None

    best: int | None = None
    best_cost: tuple[int, int] | None = None
    for role in turn.controllable():
        if role.kind == PIONEER and _pioneer_busy(turn):
            continue                       # 不打断开拓者跑任务
        blocked_cells = set(turn.blocked(role))
        goals = _adjacent_free(turn, shop, blocked_cells)
        if not goals:
            continue
        there = grid.travel_rounds(turn, role.pos, goals, frozenset(blocked_cells))
        if there is None:
            continue
        # 买(1) + 用(1) 之后还得赶回自己的站位，整体必须收在归位截止前。
        back = _rounds_home_from(turn, role, shop)
        if back is None:
            continue
        if turn.round_no + there + 2 + back > DAY_ONE_LAST_ROUND - RETURN_SLACK:
            continue
        cost = (there, role.unit_id)
        if best_cost is None or cost < best_cost:
            best, best_cost = role.unit_id, cost
    if best is not None:
        state["phase"], state["owner"] = "travel", best
    return best


def _boss_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """采购员每回合的动作：走到武器商店门口 -> 等钱 -> 买 -> 下一回合立刻用。"""
    if BOSS_ORDER in role.backpack:
        commands[role.unit_id] = use_command(BOSS_ORDER)     # 到手了就马上用
        MEMORY.boss["phase"] = "use_sent"
        return
    shop = turn.weapon_shop_pos()
    if shop is None:
        return
    blocked = set(turn.blocked(role))
    goals = _adjacent_free(turn, shop, blocked)
    if not goals:
        return
    if distance(role.pos, shop) <= 1 or role.pos in goals:
        # 已经到门口。钱够了才买——钱不够就原地等，绝不发无效的 buy
        # （买到不存在的东西属于指令非法，会计入队伍异常次数）。
        if turn.gold >= turn.boss_price():
            commands[role.unit_id] = buy_command(BOSS_ORDER, 1)
        return
    _advance(turn, role, goals, claimed, commands, extra=blocked)


# --------------------------------------------------------------------- 自进化任务
def _task_step(
    turn: Turn,
    role: Unit,
    claimed: set[Pos],
    commands: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """开拓者任务状态机：赶路 -> 接取 -> 求解 -> 等待结束，逐个做完全部任务点。"""
    state = MEMORY.task
    if state.index >= len(MEMORY.task_points):
        _return_step(turn, role, claimed, commands)     # 任务都做完了
        return
    point = MEMORY.task_points[state.index]

    # 防僵死兜底：任何阶段连续多回合没有进展（走不动、等不到回包），
    # 就放弃这个任务点转下一个，绝不让开拓者整天零指令。
    if state.phase in ("travel", "accept", "solve") \
            and turn.round_no - state.progress_round > TASK_STALL_LIMIT:
        _abandon_task(state, turn)
        _return_step(turn, role, claimed, commands)
        return

    if state.phase == "travel":
        # 出发前先算总账：来回路程 + 任务预算 + 归位，超了就跳过这个任务点。
        # 注意是「跳过」而不是「整体放弃」——另一个任务点可能离得更近、来得及做。
        if not _task_fits(turn, role, point):
            _abandon_task(state, turn)
            _return_step(turn, role, claimed, commands)
            return
        # 任务领取要求站在任务点周围一格内；注意不能站到点上。
        if role.pos != point and distance(role.pos, point) <= 1:
            state.phase = "accept"
            _touch(state, turn)
        else:
            blocked = set(turn.blocked(role))
            goals = _adjacent_free(turn, point, blocked)
            if role.pos in goals:
                state.phase = "accept"      # 本回合刚好走到位，直接进入领取
                _touch(state, turn)
            else:
                if _advance(turn, role, goals, claimed, commands):
                    _touch(state, turn)
                return

    if state.phase == "accept":
        commands[role.unit_id] = accept_task_command()
        state.phase = "solve"
        state.pending = None
        state.accept_retried = False
        _touch(state, turn)
        return

    if state.phase == "solve":
        _task_solve(turn, role, commands, response)
        return

    if state.phase == "verify":
        # 任务结束的标志：任务原文消失，或提交后过了几回合还没新消息。
        if not turn.phase_task:
            _finish_task(state, turn)
        elif state.submitted_round is not None \
                and turn.round_no - state.submitted_round >= 4:
            _finish_task(state, turn)
        return


def _task_solve(
    turn: Turn,
    role: Unit,
    commands: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """求解阶段：和 LLM / 沙盒来回，直到拿到答案。

    注意 prompt、executeCmd、submitAnswer 都是「一发一回」的：
    本回合发出去，下一回合才会在 llmResp / lastCmdResult 里看到结果。
    所以用 state.pending 记住「上回合发了什么、这回合在等什么」。
    """
    state = MEMORY.task
    waited = turn.round_no - state.progress_round

    if not turn.phase_task:
        # acceptTask 发出后任务原文一直没下发：补发一次，再不来就放弃这个任务点。
        if state.submitted_round is not None:
            _finish_task(state, turn)      # 已经提交过，说明任务确实结束了
            return
        if waited >= ACCEPT_WAIT_LIMIT:
            if state.accept_retried:
                _abandon_task(state, turn)
            else:
                commands[role.unit_id] = accept_task_command()
                state.accept_retried = True
                _touch(state, turn)
        return

    if state.pending == "llm":
        if turn.llm_resp:
            state.pending = None
            _touch(state, turn)
            _handle_llm(turn, role, commands, response)
            return
        if waited >= LLM_WAIT_LIMIT:
            # LLM 迟迟不回：把沙盒最后的输出当答案兜底提交，别把任务耗到超时。
            _submit_fallback(turn, role, commands, state)
        return

    if state.pending == "cmd":
        state.last_output = turn.last_cmd_result     # 沙盒输出到手（可能为空）
        state.pending = None
        _touch(state, turn)
        _ask_llm(turn, role, commands, response)     # 把输出回灌给 LLM
        return

    _ask_llm(turn, role, commands, response)         # 空闲状态：发起第一轮提问


def _ask_llm(
    turn: Turn,
    role: Unit,
    commands: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """发出一次 prompt；如果轮次已经用尽，就兜底提交，别让任务空转到超时。"""
    state = MEMORY.task
    if state.iterations >= TASK_LLM_MAX:
        _submit_fallback(turn, role, commands, state)
        return
    response["prompt"] = _build_prompt(turn.phase_task, state)
    state.pending = "llm"
    state.iterations += 1
    _touch(state, turn)


def _handle_llm(
    turn: Turn,
    role: Unit,
    commands: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """解析 LLM 回复：要么给最终答案，要么让我们去沙盒跑一条命令。"""
    state = MEMORY.task
    text = turn.llm_resp.strip()
    answer = _extract(text, "ANSWER:")
    command = _extract(text, "CMD:")

    if answer:
        commands[role.unit_id] = submit_answer_command(answer)
        state.answer = answer
        state.submitted_round = turn.round_no
        state.phase = "verify"
        _touch(state, turn)
        return
    if command and state.command_sent < TASK_CMD_MAX:
        response["executeCmd"] = command     # 仅在任务期间可用（接口文档 2.1）
        state.last_command = command
        state.command_sent += 1
        state.pending = "cmd"
        _touch(state, turn)
        return
    # 既没给答案也没给（可用的）命令：再问一次，轮次用尽后会自动兜底提交。
    _ask_llm(turn, role, commands, response)


def _build_prompt(task_text: str, state: Any) -> str:
    """拼出给 LLM 的提示词。

    约定两种回复格式，方便 _handle_llm 用前缀直接解析：
        CMD: <命令>      去沙盒里执行
        ANSWER: <答案>   直接提交
    """
    lines = [
        "你在《未来战争》编程大赛的沙盒里求解「自进化类」任务。",
        "沙盒可以执行 shell / python 命令，但没有外网访问。",
        "",
        "任务原文：",
        task_text.strip(),
        "",
    ]
    if state.last_command:
        # 有历史命令时，把「上一轮命令 + 它的输出」一并带上，让模型能接着推理。
        lines += [
            f"你上一轮提交的命令：{state.last_command}",
            "该命令的输出：",
            (state.last_output or "(无输出)")[:4000],   # 截断，防止 prompt 过长
            "",
        ]
    lines += [
        "请只输出下面两种格式之一，不要有多余内容：",
        "1) 还需要探索时，第一行写：CMD: <一条可直接执行的命令>",
        "2) 已经能给出最终答案时，第一行写：ANSWER: <最终答案>",
        "答案要覆盖任务要求的全部字段，字段间用英文逗号分隔。",
    ]
    return "\n".join(lines)


def _extract(text: str, marker: str) -> str | None:
    """找出以 marker 开头的那一行，返回其后的内容；没有则 None。"""
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith(marker):
            return stripped[len(marker):].strip()
    return None


def _touch(state: Any, turn: Turn) -> None:
    """记录「本回合有进展」，供防僵死超时判断使用。"""
    state.progress_round = turn.round_no


def _reset_task(state: Any) -> None:
    """清空当前任务的临时状态（不改 index/done）。"""
    state.phase = "travel"
    state.pending = None
    state.iterations = 0
    state.command_sent = 0
    state.submitted_round = None
    state.last_command = ""
    state.last_output = ""
    state.answer = ""
    state.accept_retried = False


def _finish_task(state: Any, turn: Turn) -> None:
    """任务正常收尾（拿到并通过验证），去做下一个任务点。"""
    state.done += 1
    state.index += 1
    _reset_task(state)
    _touch(state, turn)


def _abandon_task(state: Any, turn: Turn) -> None:
    """放弃当前任务点（超时/卡死），跳过它去做下一个。

    不计入 done，所以不会虚报任务数；如果所有任务点都被跳过，
    _task_step 会因为 index 越界而转成回防。
    """
    state.index += 1
    _reset_task(state)
    _touch(state, turn)


def _submit_fallback(
    turn: Turn,
    role: Unit,
    commands: dict[str, Any],
    state: Any,
) -> None:
    """兜底提交：把沙盒最后的输出当作答案交上去（有分总比弃权强）。"""
    lines = (state.last_output or "").strip().splitlines()
    answer = lines[0][:200] if lines else ""
    commands[role.unit_id] = submit_answer_command(answer)
    state.answer = answer
    state.phase = "verify"
    state.submitted_round = turn.round_no
    _touch(state, turn)


def _task_fits(turn: Turn, role: Unit, point: Pos) -> bool:
    """现在开始做这个任务，是否还赶得上第 70 回合归位。"""
    leg_in = _rounds_to(turn, role, point)
    if leg_in is None:
        return False
    return (
        turn.round_no + leg_in + TASK_BUDGET + _rounds_home(turn, role)
        <= DAY_ONE_LAST_ROUND - RETURN_SLACK
    )


# --------------------------------------------------------------------- 归位
def _must_return(turn: Turn, role: Unit) -> bool:
    """现在不回去就赶不上第 70 回合到位了吗。"""
    return turn.round_no + _rounds_home(turn, role) >= DAY_ONE_LAST_ROUND - RETURN_SLACK


def _can_still_sell(turn: Turn, role: Unit) -> bool:
    """回防途中绕去小贩卖货，是否还赶得上第 70 回合到位。

    注意要从「小贩那里」估算回家路程：卖完之后人是在小贩旁边，
    不是在他现在的位置——否则会误判成来不及而丢掉一整背矿石。
    """
    if not any(role.count(ore) for ore in TRADE_ORES):
        return False
    vendor = turn.vendor_pos()
    if vendor is None:
        return False
    to_vendor = _rounds_to(turn, role, vendor)
    home = _rounds_home_from(turn, role, vendor)
    if to_vendor is None or home is None:
        return False
    return (
        turn.round_no + to_vendor + 1 + home      # +1 是卖掉本身要花一回合
        <= DAY_ONE_LAST_ROUND - RETURN_SLACK
    )


def _rounds_home_from(turn: Turn, role: Unit, start: Pos) -> int | None:
    """从 start 走回自己站位需要几回合。"""
    stand = MEMORY.stand_for(role.unit_id)
    if stand is None:
        return None
    return _rounds_between(turn, start, stand)


def _rounds_home(turn: Turn, role: Unit) -> int:
    """从当前位置走回站位需要几回合；算不出来就用切比雪夫距离兜底。"""
    stand = MEMORY.stand_for(role.unit_id)
    if stand is None:
        return 0
    goals = _park_goals(turn, role, stand)
    # 估算不考虑队友（他们会走开），否则容易被误判成完全走不回去。
    blocked = turn.blocked(role, characters=False)
    rounds = grid.travel_rounds(turn, role.pos, goals, blocked)
    if rounds is None:
        return distance(role.pos, stand)
    return rounds


def _park_goals(turn: Turn, role: Unit, stand: Pos) -> set[Pos]:
    """归位的目标格：优先站位本身；被占了就退而求其次站它旁边。"""
    blocked = turn.blocked(role, characters=False)
    if turn.land(stand) and stand not in blocked:
        return {stand}
    options = {pos for pos in neighbours(stand) if turn.land(pos) and pos not in blocked}
    return options or {stand}


def _return_step(
    turn: Turn, role: Unit, claimed: set[Pos], commands: dict[str, Any]
) -> None:
    """回防：已经站在站位上就什么都不做，否则朝站位走一步。"""
    stand = MEMORY.stand_for(role.unit_id)
    if stand is None:
        return
    goals = _park_goals(turn, role, stand)
    if role.pos in goals:
        return
    _advance(turn, role, goals, claimed, commands)


# --------------------------------------------------------------------- 通用移动
def _advance(
    turn: Turn,
    role: Unit,
    goals: set[Pos],
    claimed: set[Pos],
    commands: dict[str, Any],
    extra: set[Pos] | frozenset[Pos] = frozenset(),
) -> bool:
    """朝最近的目标格走一步。返回是否真的下发了移动。

    寻路时对障碍采取「逐级放宽」策略，因为队友是会走开的：
      1) 避开固定建筑 + 所有队友 + 上次撞过的格子（最保险）
      2) 避开固定建筑 + 所有队友
      3) 只避开固定建筑（此时允许穿过队友所在格，下回合他会挪开）
    这样既能尽量避免碰撞，又不会因为队友一时挡路就彻底走不动。
    """
    # 别人本回合已经预定的格子不选，避免两个角色撞在同一格。
    candidates = {pos for pos in goals if pos == role.pos or pos not in claimed}
    if not candidates:
        return False
    static = set(turn.blocked(role, characters=False)) | set(extra) | MEMORY.bad_build
    others = {
        unit.pos
        for unit in turn.ours
        if unit.unit_id != role.unit_id and unit.kind in CONTROLLABLE_TYPES
    }
    stale = MEMORY.blocked_step.get(role.unit_id)     # 上回合被撞的落点
    attempts = [static | others | ({stale} if stale else set()), static | others, static]
    for blocked in attempts:
        step = grid.first_step(
            turn, role.pos, frozenset(candidates), frozenset(blocked)
        )
        if step is None or step in claimed:
            continue
        commands[role.unit_id] = move_command(step)
        claimed.add(step)
        return True
    return False


def _adjacent_free(turn: Turn, target: Pos, blocked: set[Pos]) -> set[Pos]:
    """target 周围可以站人的格子（排除 target 自身）。"""
    return {
        pos
        for pos in neighbours(target)
        if pos != target and turn.land(pos) and pos not in blocked
    }


def _rounds_to(turn: Turn, role: Unit, target: Pos) -> int | None:
    """从角色当前位置走到 target 旁边需要几回合。"""
    blocked = set(turn.blocked(role, characters=False))
    goals = _adjacent_free(turn, target, blocked)
    if not goals:
        return None
    return grid.travel_rounds(turn, role.pos, goals, frozenset(blocked))


def _rounds_between(turn: Turn, start: Pos, target: Pos) -> int | None:
    """从 start（可能是矿区这种阻挡格）走到 target 旁边需要几回合。"""
    blocked = set(turn.blocked(None, characters=False))
    goals = _adjacent_free(turn, target, blocked)
    if not goals:
        return None
    return grid.travel_rounds(turn, start, goals, frozenset(blocked))


# ===================================================================== 夜晚
def _night(turn: Turn, response: dict[str, Any]) -> None:
    """夜晚三段式：配对角色与武器 -> 能开火的先开火 -> 其余角色跑位。"""
    commands = response["roleCommandMap"]
    rockets = list(MEMORY.rockets(turn))
    if not rockets:
        return                              # 一座武器都没有，只能挨打
    characters = list(turn.controllable())
    priority = _priority_robot(turn)        # 本回合最该打的那台机器人
    # 虚拟血量池：多个火箭打同一批敌人时，用来扣减已分配伤害、避免重复计算溢出。
    virtual_hp = {robot.robot_id: robot.health for robot in turn.robots if robot.health > 0}

    # ---- 第一步：给每个角色配一座武器 ----
    pairs: list[tuple[Unit, Unit]] = []      # (操控角色, 武器)
    used: set[int] = set()                   # 已被占用的武器 id
    # 1a) 先照顾已经贴着武器、且武器不在冷却的角色，让他们直接开火；
    #     同等条件下优先回到白天分配的专属武器位。
    for role in characters:
        assigned = MEMORY.tower_for(role.unit_id)
        adjacent = [
            unit
            for unit in rockets
            if unit.unit_id not in used and distance(role.pos, unit.pos) <= 1
        ]
        ready = [unit for unit in adjacent if unit.cooldown <= 0]
        pool = ready or adjacent
        if not pool:
            continue
        tower = min(
            pool,
            key=lambda unit: (0 if unit.pos == assigned else 1, unit.unit_id),
        )
        used.add(tower.unit_id)
        pairs.append((role, tower))
    # 1b) 其余角色分配到剩下的武器，负责跑位过去。
    for role in characters:
        if any(paired.unit_id == role.unit_id for paired, _ in pairs):
            continue
        assigned = MEMORY.tower_for(role.unit_id)
        free = [unit for unit in rockets if unit.unit_id not in used]
        if not free:
            break
        tower = min(
            free,
            key=lambda unit: (
                0 if unit.pos == assigned else 1,
                distance(role.pos, unit.pos),
                unit.unit_id,
            ),
        )
        used.add(tower.unit_id)
        pairs.append((role, tower))

    # ---- 第二步：满足条件的武器开火 ----
    # attack 指令挂在「武器 id」下，controllerId 才是操控者（接口文档 2.2）。
    fired: set[int] = set()                  # 本回合已开火的角色 id
    for role, tower in pairs:
        if distance(role.pos, tower.pos) > 1 or tower.cooldown > 0:
            continue                         # 够不着或还在冷却
        cell, value = _best_blast(turn, tower, virtual_hp, priority)
        if cell is None or not _should_fire(turn, cell, value, priority):
            continue                         # 没有值得打的落点，留着冷却
        commands[tower.unit_id] = attack_command(role.unit_id, cell)
        fired.add(role.unit_id)
        _apply_blast(turn, cell, virtual_hp)  # 把这发伤害记进虚拟血量池

    # ---- 第三步：没开火的角色继续朝自己的武器靠拢 ----
    claimed: set[Pos] = set()
    for role, tower in pairs:
        if role.unit_id in fired or distance(role.pos, tower.pos) <= 1:
            continue
        goals = {
            pos
            for pos in neighbours(tower.pos)
            if turn.land(pos) and pos not in turn.blocked(role)
        }
        stand = MEMORY.stand_for(role.unit_id)
        if stand is not None:
            goals.add(stand)                 # 站位也是可接受的落点
        _advance(turn, role, goals, claimed, commands)


def _priority_robot(turn: Turn) -> Any:
    """本回合的头号目标：离我方基地最近的那台机器人。

    需求口径明确是「离我们基地最近」，而不是「离敌方基地最近」。
    若报文给了 targetTeam，则同距离下优先打「正在打我们」的那台。
    """
    robots = [robot for robot in turn.robots if robot.health > 0]
    if not robots:
        return None

    def key(robot: Any) -> tuple[int, int, int]:
        # 未提供 targetTeam 时按「威胁我方」处理，避免把目标误判成友军。
        threat = 0 if (not robot.target_team or robot.target_team == turn.our_type) else 1
        return (turn.distance_to_base(robot.pos), threat, robot.robot_id)

    return min(robots, key=key)


def _best_blast(
    turn: Turn,
    tower: Unit,
    virtual_hp: dict[int, int],
    priority: Any,
) -> tuple[Pos | None, float]:
    """在射程内枚举落点，返回单发期望伤害最大的坐标。

    候选落点 = 每台机器人的位置 + 它们周围 8 格，
    因为「打中心」和「打溅射边缘」是两种不同的覆盖方式，都要试。
    """
    reach = tower.range_of_attack()
    # 只考虑射程内、且虚拟血量还没被打空的机器人。
    robots = [
        robot
        for robot in turn.robots
        if virtual_hp.get(robot.robot_id, 0) > 0
        and distance(tower.pos, robot.pos) <= reach
    ]
    if not robots:
        return None, 0.0
    cells: set[Pos] = set()
    for robot in robots:
        cells.add(robot.pos)
        cells.update(neighbours(robot.pos))
    best: Pos | None = None
    best_value = 0.0
    for cell in sorted(cells):
        if not turn.land(cell) or distance(tower.pos, cell) > reach:
            continue                         # 落点必须在射程内
        value = _blast_value(cell, robots, virtual_hp, priority)
        if value > best_value:
            best, best_value = cell, value
    return best, best_value


def _blast_value(
    cell: Pos,
    robots: list[Any],
    virtual_hp: dict[int, int],
    priority: Any,
) -> float:
    """给一个落点打分。

    伤害模型（任务书 4.5.4）：中心 20，周围 8 格各 10。
    两个细节：
      - 用 min(伤害, 剩余血量) 截断，避免把「打不死还溢出」的伤害算成收益；
      - 头号目标（离基地最近那台）加权 2 倍，兼顾「最大化伤害」和「优先解围」。
    """
    total = 0.0
    for robot in robots:
        span = distance(robot.pos, cell)
        if span == 0:
            damage = ROCKET_CENTER_DAMAGE
        elif span == 1:
            damage = ROCKET_SPLASH_DAMAGE
        else:
            continue                         # 超出溅射范围，打不到
        effective = min(damage, virtual_hp.get(robot.robot_id, 0))
        if effective <= 0:
            continue
        weight = 2.0 if priority is not None and robot.robot_id == priority.robot_id else 1.0
        total += effective * weight
    return total


def _apply_blast(turn: Turn, cell: Pos, virtual_hp: dict[int, int]) -> None:
    """把一次已决定的爆炸伤害从虚拟血量池里扣掉，供后续火箭参考。"""
    for robot in turn.robots:
        span = distance(robot.pos, cell)
        if span == 0:
            damage = ROCKET_CENTER_DAMAGE
        elif span == 1:
            damage = ROCKET_SPLASH_DAMAGE
        else:
            continue
        if robot.robot_id in virtual_hp:
            virtual_hp[robot.robot_id] = max(0, virtual_hp[robot.robot_id] - damage)


def _should_fire(turn: Turn, cell: Pos, value: float, priority: Any) -> bool:
    """该不该开这一炮。

    火箭有 3 回合冷却，打一只远处的散兵很亏（占了冷却却只换 20 伤害），
    所以要满足其一才开火：一轮能打出 2 个机器人份的伤害，或敌人已经逼近基地。
    """
    if value <= 0:
        return False
    if value >= FIRE_MIN_VALUE:
        return True
    if priority is not None and turn.distance_to_base(priority.pos) <= URGENT_BASE_DISTANCE:
        return True
    return turn.distance_to_base(cell) <= URGENT_BASE_DISTANCE
