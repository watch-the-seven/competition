"""跨回合记忆与基地几何规划。

HTTP 服务在整个比赛期间常驻（判题器只在一局开始时拉起一次），
所以可以用模块级状态记住「上一回合下发了什么」，据此解释
lastRoundRoleActionResults，并保存任务状态机、BOSS 采购状态、采矿目标等。

本文件分两部分：
  1. Plan——开局算一次、整局不变的静态布局（武器位 / 站位 / 城墙位）；
  2. Memory——每回合读写一次的动态状态。

布局规则（需求给定，全是固定坐标，不再做启发式选址）
------------------------------------------------------
记基地「左下角」为 (x, y)，则基地占 4 格：

    (x,y) (x+1,y) (x,y+1) (x+1,y+1)

3 座火箭发射台：      (x+2,y+2)   (x+2,y-1)   (x-1,y)
3 个站位（按 unit_id 末两位绑定，站位必须紧贴自己要操控的炮台）：

    末两位 10 的工人   -> 站 (x+1,y+2)，操控 (x+2,y+2)
    末两位 11 的开拓者 -> 站 (x-2,y)  ，操控 (x-1,y)
    末两位 12 的工人   -> 站 (x+1,y-1)，操控 (x+2,y-1)

10 面城墙（C 形：下沿 3 格 + 右列 6 格 + 上沿 3 格，两个角格共用）：

    (x+1,y-2) (x+2,y-2) (x+3,y-2)
    (x+3,y-1) (x+3,y) (x+3,y+1) (x+3,y+2) (x+3,y+3)
    (x+2,y+3) (x+1,y+3)

基地 x > 10 时，整套坐标关于点 (20, 15.5) 中心对称，也就是右下角那一方的镜像布局。

坐标口径提醒（很容易差一格）
---------------------------
需求里的 (x,y) 是基地「左下角」，而报文里 station 的 pos 是「左上角」
（接口文档 1.3.1），所以 x = pos.x、y = pos.y - 1。
样例报文里 station=(10,24)，已有的三座塔位于 (9,24)/(10,25)/(9,25)——
只有按「pos 是左上角」理解，这三格才都落在基地外一圈；若把 pos 当左下角，
(10,25) 会落进基地里，说明口径必须是左上角。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .protocol import (
    PIONEER,
    ROCKET,
    STONE,
    TEAM_DEFENDER,
    Pos,
    Turn,
    Unit,
    base_lower_left,
    distance,
    neighbours,
    station_footprint,
)

LOGGER = logging.getLogger(__name__)

WALL_COUNT = 10                 # 需求：10 块石头砌 10 面墙
TOWER_COUNT = 3                 # 需求：3 座火箭发射台（任务书 4.5.1 上限也是 3）

# --------------------------------------------------------------- 固定布局表
# 偏移量一律以「基地左下角」为原点，和需求里的写法逐条对应，方便核对。
# unit_id 末两位 -> (站位偏移, 该站位负责操控的炮台偏移)
# 三座炮台分别是 (x+2,y+2) / (x+2,y-1) / (x-1,y)，与需求逐条对应。
SLOT_LAYOUT: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "10": ((+1, +2), (+2, +2)),     # 工人1
    "11": ((-2, 0), (-1, 0)),       # 开拓者
    "12": ((+1, -1), (+2, -1)),     # 工人2
}
SLOT_ORDER: tuple[str, ...] = ("10", "11", "12")
WALL_OFFSETS: tuple[tuple[int, int], ...] = (
    (+1, -2), (+2, -2), (+3, -2),
    (+3, -1), (+3, 0), (+3, +1), (+3, +2), (+3, +3),
    (+2, +3), (+1, +3),
)
# 需求：基地 x>10 时关于该点中心对称（41x32 地图的中心）
MIRROR_PIVOT_X = 20.0
MIRROR_PIVOT_Y = 15.5
MIRROR_THRESHOLD_X = 10


# --------------------------------------------------------------------- 几何规划
@dataclass(frozen=True, slots=True)
class Plan:
    """开局算一次、整局不变的静态布局。"""

    corner: str                                  # "tl"=原布局, "br"=镜像布局
    mirrored: bool                               # 是否用了镜像坐标
    footprint: tuple[Pos, ...]                   # 基地占据的 4 格
    tower_sites: tuple[Pos, ...]                 # 3 座火箭发射台（按 SLOT_ORDER 排列）
    stands: tuple[Pos, ...]                      # 与 tower_sites 一一对应的站位
    wall_sites: tuple[Pos, ...]                  # 10 面城墙
    slot_index: dict[str, int]                   # 末两位 -> 上面两个数组的下标

    @classmethod
    def build(cls, turn: Turn) -> "Plan | None":
        """按当前回合的基地位置算出整套固定布局。"""
        station = turn.station()
        if station is None:
            LOGGER.warning("报文里没有基地(station)，无法规划布局")
            return None
        footprint = station_footprint(station.pos)
        # 需求里的 (x,y) 是左下角，报文 pos 是左上角；换算集中在 protocol 一处。
        origin = base_lower_left(station.pos)
        mirrored = _is_mirrored(turn, station)

        def place(offset: tuple[int, int]) -> Pos:
            """把「相对基地左下角」的偏移换成真实坐标。

            镜像时偏移取 (1-dx, 1-dy)：镜像会把基地的左下角映射到对面基地的
            右上角，所以相对新基地左下角的偏移正好是 (1-dx, 1-dy)。
            """
            dx, dy = offset
            if mirrored:
                dx, dy = 1 - dx, 1 - dy
            return Pos(origin.x + dx, origin.y + dy)

        tower_sites: list[Pos] = []
        stands: list[Pos] = []
        slot_index: dict[str, int] = {}
        for slot in SLOT_ORDER:
            stand_off, rocket_off = SLOT_LAYOUT[slot]
            slot_index[slot] = len(tower_sites)
            tower_sites.append(place(rocket_off))
            stands.append(place(stand_off))

        walls = [place(off) for off in WALL_OFFSETS]
        # 布局是写死的：落点并非合法空地时只剔除并告警，不悄悄换到别处。
        bad_walls = [pos for pos in walls if not turn.land(pos)]
        if bad_walls:
            LOGGER.warning("有 %d 个城墙位不是合法空地，已剔除：%s",
                           len(bad_walls), [str(pos) for pos in bad_walls])
            walls = [pos for pos in walls if turn.land(pos)]
        bad_fixed = [
            str(pos) for pos in (*tower_sites, *stands) if not turn.land(pos)
        ]
        if bad_fixed:
            LOGGER.warning("固定布局里有 %d 格不是合法空地：%s",
                           len(bad_fixed), bad_fixed)
        if len(tower_sites) != TOWER_COUNT or len(walls) != WALL_COUNT:
            LOGGER.warning("布局数量异常：炮台 %d/%d，城墙 %d/%d",
                           len(tower_sites), TOWER_COUNT,
                           len(walls), WALL_COUNT)

        return cls(
            corner="br" if mirrored else "tl",
            mirrored=mirrored,
            footprint=footprint,
            tower_sites=tuple(tower_sites),
            stands=tuple(stands),
            wall_sites=tuple(walls),
            slot_index=slot_index,
        )

    def slot_for(self, unit: Unit, used: set[int]) -> int | None:
        """给角色分配专属槽位：优先按 unit_id 末两位，认不出再按兵种兜底。"""
        suffix = str(unit.unit_id)[-2:]
        index = self.slot_index.get(suffix)
        if index is not None and index not in used:
            return index
        prefer = ("11",) if unit.kind == PIONEER else ("10", "12")
        for key in prefer:
            index = self.slot_index.get(key)
            if index is not None and index not in used:
                return index
        return None

def _is_mirrored(turn: Turn, station: Unit) -> bool:
    """需求口径：基地 x<10 用原布局，x>10 用镜像；正好 10 时按阵营判断。"""
    if station.pos.x > MIRROR_THRESHOLD_X:
        return True
    if station.pos.x < MIRROR_THRESHOLD_X:
        return False
    return turn.our_type == TEAM_DEFENDER


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
    progress_round: int = 0              # 最近一次「有进展」的回合，用于识别卡死
    accept_retried: bool = False         # acceptTask 是否已经补发过一次


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
        self.boss: dict = {"phase": "wait", "owner": None, "blocked": False}   # BOSS 采购状态
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

        # 判题器可能第 1 回合还没下发任务点（或当时开拓者不在场）：
        # 只要还没拿到就每回合补一次，否则开拓者会一整天没任务可做。
        if not self.task_points and turn.player_tasks:
            self.task_points = [task.pos for task in turn.player_tasks]

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

        # 站位 / 炮台绑定：需求指定按 unit_id 末两位（10/11/12）一一对应，
        # 每个人固定守自己的那一格、操控自己那一座炮台。
        used: set[int] = set()
        for unit in turn.controllable():
            index = plan.slot_for(unit, used)
            if index is None:
                LOGGER.warning("角色 %s(%s) 没分到站位槽", unit.unit_id, unit.kind)
                continue
            used.add(index)
            self.stand_of[unit.unit_id] = plan.stands[index]
            self.tower_of[unit.unit_id] = plan.tower_sites[index]

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


# 全局单例：整局比赛共用一个 Memory。
MEMORY = Memory()
