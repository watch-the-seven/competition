"""协议层：解析判题器下发的 Request，并构造 Response 中的动作指令。

字段命名严格对齐 docs/接口文档.md，取值形状参考 docs/request.txt 与 docs/response.txt。
本文件只做「数据 <-> 结构」的转换，不含任何策略——策略在 brain.py。

阅读顺序建议：Pos/neighbours/distance -> Unit/Robot/PlayerTask -> Turn -> 末尾的命令构造函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# ---------------------------------------------------------------- 时间单位
# 任务书 4.2：白天 70 回合、晚上 60 回合，一天 130 回合。
DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS
# 本解题只负责第一天，第 70 回合是白天最后一个回合，第 71 回合入夜。
DAY_ONE_LAST_ROUND = DAY_ROUNDS

# ---------------------------------------------------------------- 规则常量
WEAPON_BUILD_COST = 25          # 任务书 4.5.1：三种武器工事建造代价均为 25 金币
WALL_MATERIAL = "stone"         # 围墙 level1 建造代价：石头 * 1
LAND = "land"                   # 地图上没有任何中立元素的空地

# 单位类型（对应报文里的 roleType 字段）
STATION = "station"             # 基地
WALL = "wall"                   # 围墙
WORKER = "worker"               # 工人
PIONEER = "pioneer"             # 开拓者
GATLING = "gatling"             # 加特林炮台
RAILGUN = "railgun"             # 电磁狙击炮
ROCKET = "rocket"               # 火箭发射台
TOWER_TYPES = (GATLING, RAILGUN, ROCKET)
CONTROLLABLE_TYPES = (WORKER, PIONEER)   # 能被我们下指令的角色（相对于建筑）

# 中立元素类型（对应报文 mapInfo.zones[].neutralType）
STONE = "stone"                 # 石矿
IRON = "iron"                   # 铁矿
COPPER = "copper"               # 铜矿
ORE_TYPES = (STONE, IRON, COPPER)
# 只卖铁/铜：石头要留给围墙（需求是工人采 10 石砌墙）。
TRADE_ORES = (IRON, COPPER)
VENDOR = "vendor"               # 小贩（收购矿石）
WEAPON_SHOP = "weaponShop"      # 武器商店（卖升级券/消耗品/召唤令）

BOSS_ORDER = "BossRobotSummonOrder"     # 任务书 4.6.3：BOSS 召唤令，200 金币

TEAM_CHALLENGER = "challenger"          # 左上角基地
TEAM_DEFENDER = "defender"              # 右下角基地

# 任务书 4.5.1：各武器 level1..level3 的攻击距离；火箭 level3 为全图（用大数代替）。
TOWER_RANGE_BY_LEVEL: dict[str, tuple[int, int, int]] = {
    GATLING: (3, 5, 7),
    RAILGUN: (6, 8, 10),
    ROCKET: (10, 15, 10 ** 9),
}
# 任务书 4.5.4：火箭弹中心伤害 20，落点周围 8 格溅射为中心伤害的一半。
ROCKET_CENTER_DAMAGE = 20
ROCKET_SPLASH_DAMAGE = 10

# 八方向偏移量（上下左右 + 四个斜角）
STEPS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# 所有已知的单位类型，用于排查「报文里出现了没见过的 roleType」。
KNOWN_KINDS = frozenset(
    (STATION, WALL, WORKER, PIONEER, GATLING, RAILGUN, ROCKET)
)


def as_int(value: Any, default: int = 0) -> int:
    """尽量把报文里的值转成整数；转不了就用默认值，绝不抛异常。

    报文是外部输入，字段偶尔缺类型（null、字符串、甚至整个字段没有），
    解析层必须扛住——抛异常会导致整个回合拿不到指令。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# 报文里的枚举值到「规范写法」的映射（键统一小写）。
# 注意不能简单粗暴地转小写：像 weaponShop 这种驼峰值转小写后就再也匹配不上了，
# 必须映回文档里的规范拼写。
_CANONICAL = {
    name.lower(): name
    for name in (
        STONE, IRON, COPPER, VENDOR, WEAPON_SHOP,
        "challengerTaskPoint1", "challengerTaskPoint2",
        "defenderTaskPoint1", "defenderTaskPoint2",
        STATION, WALL, WORKER, PIONEER, GATLING, RAILGUN, ROCKET,
        TEAM_CHALLENGER, TEAM_DEFENDER,
    )
}


def as_kind(value: Any) -> str:
    """规范化 roleType / neutralType 这类枚举字符串。

    去掉首尾空白、容忍大小写差异，但仍映回文档里的规范拼写；
    没见过的取值原样返回，方便日志里看出「报文给了什么奇怪的值」。
    """
    raw = str(value or "").strip()
    return _CANONICAL.get(raw.lower(), raw)


# ------------------------------------------------------------------- 基础结构
@dataclass(frozen=True, slots=True, order=True)
class Pos:
    """地图坐标。frozen 让它可哈希，能直接当 dict/set 的键。"""

    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        """从报文的 {"x":..,"y":..} 构造。"""
        return cls(as_int(raw.get("x")), as_int(raw.get("y")))

    def dump(self) -> dict[str, int]:
        """转回报文格式。"""
        return {"x": self.x, "y": self.y}

    def offset(self, dx: int, dy: int) -> "Pos":
        """偏移后的新坐标（不改原对象）。"""
        return Pos(self.x + dx, self.y + dy)


def neighbours(pos: Pos) -> tuple[Pos, ...]:
    """周围 8 个格子（可能越界，调用方需自行判断）。"""
    return tuple(Pos(pos.x + dx, pos.y + dy) for dx, dy in STEPS_8)


def distance(first: Pos, second: Pos) -> int:
    """任务书 4.5.4：距离采用切比雪夫距离 max(|dx|, |dy|)。

    即「走 8 个方向、斜走也算 1 步」时的实际步数，所以它天然等于走过去的回合数。
    """
    return max(abs(first.x - second.x), abs(first.y - second.y))


# 报文里 station 的 pos 是基地 2x2 的「左上角」（接口文档 1.3.1），
# 而需求描述布局时说的 (x,y) 是基地「左下角」，两者相差一格。
# 全工程只在这里做换算：口径若变，改这一个常量即可，
# 基地占格与布局锚点会一起跟着变，不会出现一处改一处没改的错位。
POS_IS_TOP_LEFT = True


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    """基地 2x2 占据的 4 格（按 POS_IS_TOP_LEFT 解释 pos）。

    用于判断障碍、以及算「某格离基地几圈」。
    """
    if POS_IS_TOP_LEFT:
        return (
            pos,
            Pos(pos.x + 1, pos.y),
            Pos(pos.x, pos.y - 1),
            Pos(pos.x + 1, pos.y - 1),
        )
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y + 1),
        Pos(pos.x + 1, pos.y + 1),
    )


def base_lower_left(pos: Pos) -> Pos:
    """把报文的 station pos 换算成需求口径的基地左下角 (x, y)。

    布局公式（火箭 / 站位 / 城墙）全部以这个左下角为原点。
    """
    return Pos(pos.x, pos.y - 1) if POS_IS_TOP_LEFT else Pos(pos.x, pos.y)


# ------------------------------------------------------------------- Request
@dataclass(frozen=True, slots=True)
class Unit:
    """我方/敌方的一个单位（角色或建筑）。对应接口文档 1.3.1 的 Role。"""

    unit_id: int
    pos: Pos
    kind: str                 # roleType：worker/pioneer/station/wall/gatling/...
    health: int
    level: int                # 仅建筑有，角色为 0
    cooldown: int             # 武器冷却剩余回合（只有火箭有）
    attack_range: int         # 报文给的距离，0 表示角色没有攻击能力
    capacity: int | None      # 背包上限，建筑为 None
    backpack: tuple[str, ...] # 背包内物品名列表

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        # 报文里建筑没有 backPackCapability，用 None 表示「不适用」。
        raw_capacity = raw.get("backPackCapability")
        # health 缺失时按「存活」处理：把缺失当成 0 会把整个队伍判成阵亡，
        # 结果所有角色都不再被调度。只有显式给 0 才视为阵亡。
        raw_health = raw.get("health")
        return cls(
            as_int(raw.get("id")),
            Pos.load(raw["pos"]),
            as_kind(raw.get("roleType")),
            as_int(raw_health, 1) if raw_health is not None else 1,
            as_int(raw.get("level")),
            as_int(raw.get("cooldown")),
            as_int(raw.get("attackRange")),
            as_int(raw_capacity) if raw_capacity is not None else None,
            tuple(str(item) for item in raw.get("backpack") or ()),
        )

    @property
    def backpack_full(self) -> bool:
        """背包是否已满（建筑没有背包，恒为 False）。"""
        if self.capacity is None:
            return False
        return len(self.backpack) >= self.capacity

    def count(self, item: str) -> int:
        """背包里某种物品的数量。"""
        return self.backpack.count(item)

    def range_of_attack(self) -> int:
        """武器攻击距离：优先信任报文里的 attackRange，缺失时按等级表回退。"""
        if self.attack_range > 0:
            return self.attack_range
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        if table is None:
            return 0
        level = min(max(self.level, 1), len(table))   # 等级夹到 1..3，防止越界
        return table[level - 1]


@dataclass(frozen=True, slots=True)
class Robot:
    """场上的一台机器人。对应接口文档 1.5.1 的 RobotRole。"""

    robot_id: int
    pos: Pos
    kind: str                 # smallRobot/middleRobot/largeRobot/bossRobot
    health: int
    target_team: str          # 它要打哪一方；报文未下发时为空串
    abnormal_state: str       # 被眩晕时为 "dizzy"，否则空串

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(
            as_int(raw.get("id")),
            Pos.load(raw["pos"]),
            as_kind(raw.get("roleType")),
            as_int(raw.get("health")),
            as_kind(raw.get("targetTeam")),
            str(raw.get("abnormalState") or ""),
        )


@dataclass(frozen=True, slots=True)
class PlayerTask:
    """一个可接取的自进化任务点。对应接口文档 1.3.2 的 PlayerTask。"""

    task_type: str
    pos: Pos
    cooldown: int             # 任务刷新冷却剩余回合，0 表示可接
    score_reward: int
    gold_reward: int
    is_valid: bool            # 冷却中或任务做完则为 False
    timeout_rounds: int       # 超过这么多回合任务强制结束

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "PlayerTask":
        return cls(
            as_kind(raw.get("taskType")),
            Pos.load(raw["taskPosition"]),
            as_int(raw.get("coldDownRounds")),
            as_int(raw.get("scoreReward")),
            as_int(raw.get("goldReward")),
            bool(raw.get("isValid")),
            as_int(raw.get("timeoutRounds")),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    """一个回合的完整输入快照——策略层只需要和这个对象打交道。"""

    round_no: int
    is_day: bool
    gold: int                 # 队伍金币（买东西花的就是它）
    total_score: int
    width: int
    height: int
    our_type: str             # challenger / defender
    zones: dict[Pos, str]     # 中立元素：矿区、小贩、武器商店、任务点
    ours: tuple[Unit, ...]    # 我方全部单位
    enemy_roles: tuple[Unit, ...]      # 敌方可见单位（基地与围墙是全局可见的）
    robots: tuple[Robot, ...]
    player_tasks: tuple[PlayerTask, ...]
    phase_task: str           # 当前已领取任务的原文，空串表示没有在做的任务
    llm_resp: str             # 上一回合 prompt 的回复
    last_cmd_result: str      # 上一回合 executeCmd 的输出
    action_results: dict[int, bool]    # 上一回合各角色动作是否合法
    summon_result: int        # 上一回合召唤宝藏的结果码
    vendor_prices: dict[str, int]      # 小贩收购价：矿种 -> 单价
    weapon_prices: dict[str, int]      # 武器商店售价：商品名 -> 单价
    errors: tuple[tuple[int, str], ...]  # (errorCode, 描述)

    # -------------------------------------------------------------- 反序列化
    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        """把判题器的 Request JSON 转成 Turn。

        对所有可选字段都做了兜底（`or {}` / `or 0`），
        这样即使某个字段缺失也只是少一点信息，不会让整场异常退出。
        """
        round_no = as_int(payload.get("roundNo"), 1)
        info = payload.get("mapInfo") or {}
        team = payload.get("teamOur") or {}
        robot_block = payload.get("robot") or {}
        enemy_block = payload.get("teamEnemy") or {}

        # 商店/小贩列表是数组，转成「名字 -> 价格」的字典方便查询。
        prices = {
            str(item.get("name")): int(item.get("price") or 0)
            for item in payload.get("vendorShopList") or ()
        }
        shop = {
            str(item.get("name")): int(item.get("price") or 0)
            for item in payload.get("weaponShopList") or ()
        }
        # 动作合法性结果的 key 在 JSON 里是字符串，这里统一转成 int。
        actions = {
            int(key): bool(value)
            for key, value in (payload.get("lastRoundRoleActionResults") or {}).items()
        }

        return cls(
            round_no=round_no,
            # 任务书 4.2：白天 = 每天的前 70 回合。
            is_day=(round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
            gold=as_int(team.get("goldNum")),
            total_score=as_int(team.get("totalScore")),
            width=as_int(info.get("width"), 41),
            height=as_int(info.get("height"), 32),
            our_type=as_kind(team.get("type")),
            zones={
                Pos.load(zone["pos"]): as_kind(zone.get("neutralType"))
                for zone in info.get("zones") or ()
            },
            ours=tuple(Unit.load(role) for role in team.get("roles") or ()),
            enemy_roles=tuple(
                Unit.load(role) for role in enemy_block.get("roles") or ()
            ),
            robots=tuple(
                Robot.load(role) for role in robot_block.get("roles") or ()
            ),
            player_tasks=tuple(
                PlayerTask.load(task) for task in team.get("playerTasks") or ()
            ),
            phase_task=str(payload.get("phaseTask") or ""),
            llm_resp=str(payload.get("llmResp") or ""),
            last_cmd_result=str(payload.get("lastCmdResult") or ""),
            action_results=actions,
            summon_result=as_int(payload.get("lastSummonTreasureResult")),
            vendor_prices=prices,
            weapon_prices=shop,
            errors=tuple(
                (as_int(err.get("errorCode")), str(err.get("description") or ""))
                for err in payload.get("errors") or ()
            ),
        )

    # ------------------------------------------------------------------ 查询
    def alive(self, kinds: Iterable[str]) -> tuple[Unit, ...]:
        """按类型筛选存活（health > 0）的单位。"""
        wanted = tuple(kinds)
        return tuple(
            unit for unit in self.ours if unit.kind in wanted and unit.health > 0
        )

    def station(self) -> Unit | None:
        """我方基地；理论上一定有。"""
        for unit in self.ours:
            if unit.kind == STATION:
                return unit
        return None

    def footprint(self) -> tuple[Pos, ...]:
        """基地占据的格子（没有基地时返回空）。"""
        station = self.station()
        return station_footprint(station.pos) if station else ()

    def footprint_of(self, unit: Unit) -> tuple[Pos, ...]:
        """任意单位占据的格子：基地 4 格，其它 1 格。"""
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def controllable(self) -> tuple[Unit, ...]:
        """可被调度的角色：开拓者 + 工人，按 id 升序保证每回合顺序稳定。"""
        return tuple(
            sorted(self.alive(CONTROLLABLE_TYPES), key=lambda unit: unit.unit_id)
        )

    def workers(self) -> tuple[Unit, ...]:
        """存活的工人，按 id 升序。"""
        return tuple(sorted(self.alive((WORKER,)), key=lambda unit: unit.unit_id))

    def pioneer(self) -> Unit | None:
        """存活的开拓者（没有则 None，比如阵亡等待复活）。"""
        found = self.alive((PIONEER,))
        return found[0] if found else None

    def weapons(self) -> tuple[Unit, ...]:
        """存活的武器工事，按坐标排序（夜晚配对时顺序稳定）。"""
        return tuple(
            sorted(
                self.alive(TOWER_TYPES),
                key=lambda unit: (unit.pos.x, unit.pos.y, unit.unit_id),
            )
        )

    def walls(self) -> tuple[Unit, ...]:
        """存活的围墙。"""
        return self.alive((WALL,))

    def has_structure_at(self, pos: Pos) -> bool:
        """某个格子上是否已经站着（我方）建筑。"""
        return any(pos in self.footprint_of(unit) for unit in self.ours)

    def enemy_station_pos(self) -> Pos | None:
        """敌方基地坐标（全局可见），用来判断「哪一侧朝向敌人」。"""
        for unit in self.enemy_roles:
            if unit.kind == STATION:
                return unit.pos
        return None

    def zone_of(self, kind: str) -> tuple[Pos, ...]:
        """某种中立元素的全部坐标，按坐标排序。"""
        return tuple(
            sorted(pos for pos, value in self.zones.items() if value == kind)
        )

    def ore_mines(self, ore: str) -> tuple[Pos, ...]:
        """某种矿的全部矿点。"""
        return self.zone_of(ore)

    def vendor_pos(self) -> Pos | None:
        """小贩坐标：卖矿石要站到它周围一格内。"""
        found = self.zone_of(VENDOR)
        return found[0] if found else None

    def weapon_shop_pos(self) -> Pos | None:
        """武器商店坐标：买东西要站到它周围一格内。"""
        found = self.zone_of(WEAPON_SHOP)
        return found[0] if found else None

    def boss_price(self) -> int:
        """BOSS 召唤令价格，从报文动态读取，读不到就按任务书的 200。"""
        return int(self.weapon_prices.get(BOSS_ORDER) or 200)

    def valid_tasks(self) -> tuple[PlayerTask, ...]:
        """当前可接取的任务点。"""
        return tuple(task for task in self.player_tasks if task.is_valid)

    # ------------------------------------------------------------------ 地形
    def land(self, pos: Pos) -> bool:
        """可通行/可建造的空地：在图内、且没有中立元素。

        注意：建筑和角色不在这里判断，它们由 blocked() 负责。
        """
        if not 0 <= pos.x < self.width or not 0 <= pos.y < self.height:
            return False
        return self.zones.get(pos, LAND) == LAND

    def occupied_cells(self) -> frozenset[Pos]:
        """我方全部单位占据的格子。"""
        cells: set[Pos] = set()
        for unit in self.ours:
            cells.update(self.footprint_of(unit))
        return frozenset(cells)

    def structure_cells(self) -> frozenset[Pos]:
        """只包含固定障碍：基地、武器工事、围墙（角色会移动，不算）。"""
        cells: set[Pos] = set()
        for unit in self.ours:
            if unit.kind == STATION or unit.kind == WALL or unit.kind in TOWER_TYPES:
                cells.update(self.footprint_of(unit))
        return frozenset(cells)

    def blocked(
        self,
        moving: Unit | None = None,
        extra: Iterable[Pos] = (),
        characters: bool = True,
    ) -> frozenset[Pos]:
        """任务书 4.1：中立元素、以及所有单位所在格均阻挡移动。

        characters=False 时忽略己方角色——他们每回合都在动，长距离寻路时
        把他们当死障碍会导致「队友挡住唯一入口 -> 永远走不回去」。
        extra 用来临时排除一些格子（例如已经决定要建墙的位置）。
        """
        cells = {pos for pos, kind in self.zones.items() if kind != LAND}
        if characters:
            cells.update(self.occupied_cells())
        else:
            cells.update(self.structure_cells())
        cells.update(extra)
        for robot in self.robots:
            cells.add(robot.pos)
        if moving is not None:
            # 自己所在的格不算障碍，否则无法规划。
            cells.difference_update(self.footprint_of(moving))
        return frozenset(cells)

    def distance_to_base(self, pos: Pos) -> int:
        """到「我方基地最近一格」的距离——判断敌人逼近程度用。"""
        footprint = self.footprint()
        if not footprint:
            return 10 ** 6
        return min(distance(pos, cell) for cell in footprint)


# ------------------------------------------------------------------ Response
# 下面每个函数对应一个动作码（接口文档 2.3），只负责拼出该动作需要的字段。
# 字段名必须与接口文档完全一致，写错会被判「指令错误」并计入队伍异常次数。
def move_command(pos: Pos) -> dict[str, Any]:
    """移动一格到 pos。"""
    return {"action": "move", "targetPos": [pos.dump()]}


def collect_command(pos: Pos) -> dict[str, Any]:
    """采集 pos 处的矿（需站在矿周围一格内，仅工人可用）。"""
    return {"action": "collect", "targetPos": [pos.dump()]}


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    """在 pos 建造 name（"rocket"/"wall" 等），需站在 pos 周围一格内。"""
    return {"action": "build", "targetPos": [pos.dump()], "name": name}


def remove_command(pos: Pos) -> dict[str, Any]:
    """拆除 pos 处的围墙（仅工人可用）。"""
    return {"action": "remove", "targetPos": [pos.dump()]}


def attack_command(controller_id: int, pos: Pos) -> dict[str, Any]:
    """由 controller_id 操控武器攻击 pos（仅夜晚可用）。

    controllerId 必须是字符串，这是接口文档 2.2 的约定。
    """
    return {
        "action": "attack",
        "targetPos": [pos.dump()],
        "controllerId": str(controller_id),
    }


def sell_command(name: str, num: int) -> dict[str, Any]:
    """把小贩处的 num 个 name 卖掉换金币。"""
    return {"action": "sell", "name": name, "num": int(num)}


def buy_command(name: str, num: int = 1) -> dict[str, Any]:
    """在武器商店购买 num 个 name（花的是队伍金币）。"""
    return {"action": "buy", "name": name, "num": int(num)}


def use_command(name: str, pos: Pos | None = None) -> dict[str, Any]:
    """使用背包里的物品。

    消耗品/升级券有的需要指定坐标（如眩晕法宝、围墙修复包），
    召唤令这类不需要，所以 targetPos 是可选的。
    """
    command: dict[str, Any] = {"action": "use", "name": name}
    if pos is not None:
        command["targetPos"] = [pos.dump()]
    return command


def drop_command(name: str) -> dict[str, Any]:
    """丢弃背包里的一件物品（背包满了腾位置用）。"""
    return {"action": "drop", "name": name}


def accept_task_command() -> dict[str, Any]:
    """开拓者在任务点接取任务（需站在任务点周围一格内）。"""
    return {"action": "acceptTask"}


def submit_answer_command(answer: str) -> dict[str, Any]:
    """提交任务答案。"""
    return {"action": "submitAnswer", "taskAnswer": answer}


def empty_response() -> dict[str, Any]:
    """空响应：本回合不下任何指令。异常兜底和「原地待命」都用它。"""
    return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
