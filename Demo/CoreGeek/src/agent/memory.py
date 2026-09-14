"""跨回合记忆与基地几何规划。

HTTP 服务在整个比赛期间常驻（判题器只在一局开始时拉起一次），
所以可以用模块级状态记住「上一回合下发了什么」，据此解释
lastRoundRoleActionResults，并保存任务状态机、BOSS 采购状态、采矿目标等。

本文件分两部分：
  1. Plan 及其私有构造函数——开局算一次、整局不变的静态布局；
  2. Memory——每回合读写一次的动态状态。

建造区规则（本项目采用的口径）：
  基地是 2x2；与基地外形切比雪夫距离 = 1 的那一圈是「武器可建造区」，
  距离 = 2 的那一圈是「城墙可建造区」。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .protocol import (
    PIONEER,
    ROCKET,
    STONE,
    TEAM_DEFENDER,
    WALL_MATERIAL,
    Pos,
    Turn,
    Unit,
    distance,
    neighbours,
    station_footprint,
)

WALL_COUNT = 10                 # 需求：10 块石头砌 10 面墙
TOWER_COUNT = 3                 # 任务书 4.5.1：武器工事全局最多 3 座


# --------------------------------------------------------------------- 几何规划
@dataclass(frozen=True, slots=True)
class Plan:
    """第一天开局就确定下来的静态布局（基地位置整局不变）。"""

    corner: str                                  # "tl"=基地在左上角, "br"=右下角
    footprint: tuple[Pos, ...]                   # 基地占据的 4 格
    ring_weapon: tuple[Pos, ...]                 # 基地外一圈：武器可建造区
    ring_wall: tuple[Pos, ...]                   # 再外一圈：城墙可建造区
    tower_sites: tuple[Pos, ...]                 # 3 座火箭发射台选址
    wall_sites: tuple[Pos, ...]                  # 10 面城墙选址（C 形护罩）
    stands: tuple[Pos, ...]                      # 与 tower_sites 一一对应的站位格

    @classmethod
    def build(cls, turn: Turn) -> "Plan | None":
        """按当前回合的地图算一遍布局；没有基地时返回 None。"""
        station = turn.station()
        if station is None:
            return None
        footprint = station_footprint(station.pos)
        corner = _corner_of(turn, station)
        ring_weapon = _ring(turn, footprint, 1)
        ring_wall = _ring(turn, footprint, 2)
        wall_sites = _wall_sites(turn, footprint, ring_wall, corner)
        tower_sites, stands = _tower_sites(turn, footprint, ring_weapon, wall_sites)
        return cls(
            corner=corner,
            footprint=footprint,
            ring_weapon=tuple(sorted(ring_weapon)),
            ring_wall=tuple(sorted(ring_wall)),
            tower_sites=tower_sites,
            wall_sites=wall_sites,
            stands=stands,
        )

    def wall_set(self) -> frozenset[Pos]:
        """城墙选址集合（按坐标去重后便于 in 判断）。"""
        return frozenset(self.wall_sites)


def _corner_of(turn: Turn, station: Unit) -> str:
    """判断基地在左上角还是右下角（敌方在斜对角）。"""
    if turn.our_type == TEAM_DEFENDER:
        return "br"
    if turn.our_type:
        return "tl"
    # 报文缺 type 时退化为几何判断：X 偏左且 Y 偏上就是左上角。
    left = station.pos.x < turn.width / 2
    top = station.pos.y > turn.height / 2
    return "tl" if left and top else "br"


def _ring(turn: Turn, footprint: tuple[Pos, ...], radius: int) -> list[Pos]:
    """与基地 2*2 外形的切比雪夫距离恰为 radius 的所有合法空地。"""
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cells: list[Pos] = []
    # 只需在基地外扩 radius 的方框内枚举。
    for x in range(xmin - radius, xmax + radius + 1):
        for y in range(ymin - radius, ymax + radius + 1):
            pos = Pos(x, y)
            if pos in footprint:
                continue                                   # 基地自身不算
            if min(distance(pos, cell) for cell in footprint) != radius:
                continue                                   # 只要正好这一圈
            if turn.land(pos):                             # 排除矿区/商店等中立元素
                cells.append(pos)
    return cells


def _wall_shape(footprint: tuple[Pos, ...], corner: str) -> list[Pos]:
    """C 形护罩的 10 个理想格：朝敌一侧竖排 6 格，顶行/底行各再向内延伸 2 格。"""
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if corner == "tl":
        # 基地在左上角 -> 敌人在右下 -> 封右侧，开口留在左（背向敌人）。
        col_x, arm = xmax + 2, -1
    else:
        # 基地在右下角 -> 敌人在左上 -> 封左侧，开口留在右。
        col_x, arm = xmin - 2, 1
    # 竖排 6 格：从基地下沿再往下 2 格，到上沿再往上 2 格。
    cells = [Pos(col_x, y) for y in range(ymin - 2, ymax + 3)]
    # 顶行/底行各向内（朝基地方向）延伸 2 格，形成 "C" 的两条短臂。
    cells.append(Pos(col_x + arm, ymax + 2))
    cells.append(Pos(col_x + 2 * arm, ymax + 2))
    cells.append(Pos(col_x + arm, ymin - 2))
    cells.append(Pos(col_x + 2 * arm, ymin - 2))
    return cells


def _wall_sites(
    turn: Turn,
    footprint: tuple[Pos, ...],
    ring_wall: list[Pos],
    corner: str,
) -> tuple[Pos, ...]:
    """把理想的 C 形落到合法空地上，凑不够 10 格就用外圈剩余格子补齐。"""
    ideal = [pos for pos in _wall_shape(footprint, corner) if turn.land(pos)]
    sites = list(ideal)
    if len(sites) < WALL_COUNT:
        # 基地贴边导致理想形状被裁掉时，用外圈其它空格补齐（优先离原缺口最近的）。
        spare = sorted(
            (pos for pos in ring_wall if pos not in sites),
            key=lambda pos: (
                min((distance(pos, q) for q in ideal), default=0),
                pos.x,
                pos.y,
            ),
        )
        sites.extend(spare[: WALL_COUNT - len(sites)])
    return tuple(sites)


def _pick_stand(
    turn: Turn,
    tower: Pos,
    footprint: tuple[Pos, ...],
    wall_set: frozenset[Pos],
    towers: set[Pos],
    stands: set[Pos],
    ring_weapon: list[Pos],
    enemy: Pos,
) -> Pos | None:
    """给一座武器挑一个操控站位：合法、未被占用、尽量靠向基地后方。

    额外要求站位至少有 2 个可通行邻格：否则队友停在自己的站位上时，
    就会把只有一个入口的站位彻底堵死（实战踩过这个坑）。
    """
    # 把「所有已定/待定的建筑」都当作障碍，用来估算某格还剩几个出口。
    static = set(wall_set) | set(towers) | set(stands)
    for unit in turn.ours:
        static.update(turn.footprint_of(unit))

    def degree(pos: Pos) -> int:
        """该格周围还有几个可通行、且不会变成建筑的格子。"""
        return sum(
            1 for step in neighbours(pos) if turn.land(step) and step not in static
        )

    options = [
        pos
        for pos in neighbours(tower)          # 必须紧贴武器才能操控
        if pos not in footprint
        and pos not in wall_set
        and pos not in towers
        and pos not in stands
        and turn.land(pos)
        and degree(pos) >= 2                  # 不能选死胡同
    ]
    if not options:
        return None
    options.sort(
        key=lambda pos: (
            0 if pos in ring_weapon else 1,      # 优先基地外一圈
            -distance(pos, enemy),               # 再优先远离敌人（更安全）
            pos.x,
            pos.y,
        )
    )
    return options[0]


def _tower_sites(
    turn: Turn,
    footprint: tuple[Pos, ...],
    ring_weapon: list[Pos],
    wall_sites: tuple[Pos, ...],
) -> tuple[tuple[Pos, ...], tuple[Pos, ...]]:
    """先定 3 个朝向敌人的武器位（彼此留间隔），再给它们配站位。

    必须分两步：某个站位会不会被堵死，取决于「所有」武器位最终落在哪，
    如果先配一个再配下一个，就会算出过于乐观的出口数。
    """
    enemy = turn.enemy_station_pos() or Pos(turn.width // 2, turn.height // 2)
    wall_set = frozenset(wall_sites)
    # 离敌人越近越优先，这样武器尽量顶在来袭方向上。
    ordered = sorted(
        ring_weapon, key=lambda pos: (distance(pos, enemy), pos.x, pos.y)
    )

    # 第一步：选 3 个武器位，先要求彼此不相邻（spacing=2），凑不齐再放宽到相邻也行。
    chosen: list[Pos] = []
    for spacing in (2, 1):
        for cand in ordered:
            if len(chosen) >= TOWER_COUNT:
                break
            if cand in chosen:
                continue
            if any(distance(cand, other) < spacing for other in chosen):
                continue
            chosen.append(cand)
        if len(chosen) >= TOWER_COUNT:
            break

    # 第二步：给每个武器位配站位；配不到站位的武器位就放弃，顺延到下一个候选。
    planned = set(chosen)
    towers: list[Pos] = []
    stands: list[Pos] = []
    for cand in list(chosen) + [pos for pos in ordered if pos not in planned]:
        if len(towers) >= TOWER_COUNT:
            break
        if cand in stands:
            continue
        stand = _pick_stand(
            turn, cand, footprint, wall_set, planned, set(stands),
            ring_weapon, enemy,
        )
        if stand is None:
            continue
        towers.append(cand)
        stands.append(stand)
    return tuple(towers), tuple(stands)


# --------------------------------------------------------------------- 记忆
@dataclass
class TaskState:
    """开拓者的自进化任务状态机。

    phase 流转：travel（赶路）-> accept（已发 acceptTask）
                -> solve（和 LLM / 沙盒往返）-> verify（已提交，等任务结束）
    任何阶段时间不够都会进入 halt（放弃任务，回防）。
    """

    index: int = 0                       # 当前做第几个任务点
    phase: str = "travel"                # travel | accept | solve | verify | halt
    pending: str | None = None           # "llm" | "cmd"，等待上一回合的哪个回包
    last_command: str = ""               # 最近一次发给沙盒的命令
    last_output: str = ""                # 该命令的输出
    command_sent: int = 0                # 本任务已执行的沙盒命令数
    iterations: int = 0                  # 本任务已用的 LLM 往返数
    submitted_round: int | None = None   # 提交答案的回合
    answer: str = ""                     # 已提交的答案
    done: int = 0                        # 已完成任务数


class Memory:
    """整局共享的可变状态。所有读写都在 MEMORY.lock 保护下进行。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.plan: Plan | None = None
        self.stone_worker: int | None = None     # 负责采石 + 砌墙的工人
        self.tower_worker: int | None = None     # 负责造 3 座火箭的工人
        self.stand_of: dict[int, Pos] = {}       # role_id -> 白天归位/夜晚站位
        self.tower_of: dict[int, Pos] = {}       # role_id -> 负责操控的武器位
        self.issued: dict[int, dict] = {}        # 上一回合每个角色下发的指令
        self.issued_round: int = 0
        self.bad_build: set[Pos] = set()         # 试探失败的建造点（黑名单）
        self.blocked_step: dict[int, Pos] = {}   # 上一回合被碰撞挡住的落点
        self.boss: dict = {"phase": "wait", "owner": None}   # BOSS 采购状态
        self.task = TaskState()
        self.task_points: list[Pos] = []         # 第一天要做的自进化任务点
        self.ore_target: dict[int, tuple[Pos, str, int]] = {}   # role_id -> (矿点, 矿种, 目标块数)
        self.last_jobs: dict[int, str] = {}      # 最近一次白天分工（排查用）
        self.last_round: int = 0

    # ------------------------------------------------------------ 每回合开始
    def begin(self, turn: Turn) -> None:
        """每回合第一件事：开局初始化 + 解释上一回合的动作合法性。"""
        # 布局只算一次；基地不会动，所以后面回合直接复用。
        if self.plan is None:
            self.plan = Plan.build(turn)
        if self.plan is not None and self.stone_worker is None:
            self._assign_roles(turn)

        # 用「上一回合下发的指令」+「本回合返回的合法性」推断失败原因。
        for role_id, ok in turn.action_results.items():
            if ok:
                self.blocked_step.pop(role_id, None)     # 成功了就清掉旧记录
                continue
            command = self.issued.get(role_id)
            if not command:
                continue
            action = command.get("action")
            target = _first_target(command)
            if action == "build" and target is not None:
                # 建造非法：多半是可建造区判定不符，把这格拉黑以后别再试。
                self.bad_build.add(target)
            elif action == "move" and target is not None:
                # 移动非法：多半是碰撞。记下这个落点，下次优先绕开。
                self.blocked_step[role_id] = target
        self.last_round = turn.round_no

    def remember(self, commands: dict[str, dict], turn: Turn) -> None:
        """记录本回合下发的指令，供下一回合解释合法性。"""
        # JSON 序列化后 key 会变成字符串，这里统一转回 int。
        self.issued = {int(key): value for key, value in commands.items()}
        self.issued_round = turn.round_no

    def _assign_roles(self, turn: Turn) -> None:
        """开局分配分工与站位：谁采石砌墙、谁造火箭、每个人守哪座武器。"""
        plan = self.plan
        if plan is None:
            return
        workers = list(turn.workers())
        stones = [pos for pos, kind in turn.zones.items() if kind == STONE]

        def stone_cost(unit: Unit) -> int:
            """到最近石矿的距离；没有石矿就给一个很大的值。"""
            if not stones:
                return 10 ** 6
            return min(distance(unit.pos, mine) for mine in stones)

        # 离石矿近的那个工人去采石砌墙，另一个去造火箭。
        workers.sort(key=lambda unit: (stone_cost(unit), unit.unit_id))
        if workers:
            self.stone_worker = workers[0].unit_id
        if len(workers) > 1:
            self.tower_worker = workers[1].unit_id
        elif workers:
            # 只有一个工人时仍然优先保证火箭，石墙退居其次。
            self.tower_worker = self.stone_worker

        # 站位分配：把「离敌人最远（最安全）」的站位给开拓者（血量最低），
        # 其余按角色 id 顺序补齐。
        order = sorted(
            range(len(plan.stands)),
            key=lambda index: -distance(plan.stands[index], _enemy_of(turn)),
        )
        holders = list(turn.controllable())
        pioneer = turn.pioneer()
        if pioneer is not None:
            holders = [pioneer] + [u for u in holders if u.unit_id != pioneer.unit_id]
        for slot, unit in zip(order, holders):
            self.stand_of[unit.unit_id] = plan.stands[slot]
            self.tower_of[unit.unit_id] = plan.tower_sites[slot]

        # 任务点按报文给定顺序排列，策略层依次去做。
        self.task_points = [task.pos for task in turn.player_tasks]

    # ---------------------------------------------------------------- 便捷查询
    def stand_for(self, role_id: int) -> Pos | None:
        """该角色白天归位 / 夜晚操控的站位格。"""
        return self.stand_of.get(role_id)

    def tower_for(self, role_id: int) -> Pos | None:
        """该角色专属的武器位（夜晚配对时优先回到这里）。"""
        return self.tower_of.get(role_id)

    def rockets(self, turn: Turn) -> tuple[Unit, ...]:
        """我方存活的火箭发射台。"""
        return tuple(unit for unit in turn.weapons() if unit.kind == ROCKET)

    def rockets_ready(self, turn: Turn) -> bool:
        """计划中的 3 个武器位是否都已经站上火箭。"""
        plan = self.plan
        if plan is None:
            return False
        standing = {unit.pos for unit in self.rockets(turn)}
        return all(site in standing for site in plan.tower_sites)

    def outstanding_walls(self, turn: Turn) -> list[Pos]:
        """还差哪些城墙没砌（已排除建造失败被拉黑的格子）。"""
        plan = self.plan
        if plan is None:
            return []
        standing = {unit.pos for unit in turn.walls()}
        return [
            pos
            for pos in plan.wall_sites
            if pos not in standing
            and pos not in self.bad_build
        ]

    def walls_ready(self, turn: Turn) -> bool:
        """10 面墙是否都砌好了。"""
        return not self.outstanding_walls(turn)

    def order_candidates(self, turn: Turn, role: Unit) -> list[Pos]:
        """待建城墙按「离工人当前位置」排序——就近施工，少走路。"""
        pending = self.outstanding_walls(turn)
        if not pending:
            return []
        return sorted(
            pending,
            key=lambda pos: (distance(role.pos, pos), pos.x, pos.y),
        )


def _first_target(command: dict) -> Pos | None:
    """取出指令里的第一个目标坐标（用于分析失败原因）。"""
    targets = command.get("targetPos") or []
    if not targets:
        return None
    raw = targets[0]
    try:
        return Pos(int(raw["x"]), int(raw["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _enemy_of(turn: Turn) -> Pos:
    """敌方基地坐标；看不到就用地图中心近似。"""
    return turn.enemy_station_pos() or Pos(turn.width // 2, turn.height // 2)


# 全局单例：整局比赛共用一个 Memory。
MEMORY = Memory()
