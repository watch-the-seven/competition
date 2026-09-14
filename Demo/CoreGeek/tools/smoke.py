"""离线迷你裁判器：在进程内直接调用策略，跑满第一天 130 回合并断言全部需求。

它不连判题器，只用 docs/request.txt 里的地图数据搭一个简化沙盒：
    python3 tools/smoke.py
会打印每回合的金币/动作轨迹，并跑 4 个场景，断言：
  1. 白天建成 3 座火箭发射台（且都落在「基地外一圈」的武器可建造区）
  2. 白天建成 10 面城墙（且都落在「再外一圈」的城墙可建造区）
  3. 开拓者完成 2 个自进化任务（acceptTask + submitAnswer 各 2 次）
  4. 第 70 回合三个角色都站在火箭发射台旁边
  5. 夜晚开火不超射程、且确实造成伤害
  6. （高矿价/任务奖励充足时）金币达到 200 并买入 + 使用 BOSS 召唤令

想验证真实 HTTP 链路请用 tools/mock_judge.py。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# 把 src 加进搜索路径，才能 import agent.*（本脚本在 tools/ 下运行）。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import memory as memory_module          # noqa: E402
from agent.brain import decide                      # noqa: E402
from agent.protocol import (                        # noqa: E402
    BOSS_ORDER,
    ROCKET,
    ROCKET_CENTER_DAMAGE,
    ROCKET_SPLASH_DAMAGE,
    WALL,
    Pos,
)

# ---------------------------------------------------------------- 固定地图
# 全部取自 docs/request.txt 的样例，保证测试场景和官方样例一致。
WIDTH, HEIGHT = 41, 32
STATION_POS = Pos(10, 24)                    # 基地左上角
FOOTPRINT = (STATION_POS, Pos(11, 24), Pos(10, 23), Pos(11, 23))
ENEMY_STATION = Pos(30, 10)

BASE_ZONES = {
    Pos(14, 14): "challengerTaskPoint1",
    Pos(17, 17): "challengerTaskPoint2",
    Pos(16, 17): "challengerTaskPoint2",
    Pos(23, 14): "defenderTaskPoint1",
    Pos(26, 17): "defenderTaskPoint2",
    Pos(27, 17): "defenderTaskPoint2",
    Pos(20, 16): "vendor",                   # 小贩
    Pos(25, 20): "weaponShop",               # 武器商店
    Pos(4, 24): "stone",
    Pos(14, 3): "stone",
    Pos(25, 10): "iron",
    Pos(8, 28): "iron",
    Pos(22, 26): "copper",
    Pos(7, 2): "copper",
}
VENDOR_PRICES = {"stone": 1, "iron": 3, "copper": 5}   # 样例里的小贩收购价
WEAPON_PRICES = {
    "WeaponUpgradeVoucher1": 100, "WeaponUpgradeVoucher2": 150,
    "WallUpgradeVoucher1": 20, "WallUpgradeVoucher2": 30,
    "StationUpgradeVoucher1": 100, "StationUpgradeVoucher2": 150,
    "WallFixer": 10, "Medicine": 10, "DizzyWeapon": 100, "Bomb": 100,
    "SmallRobotSummonOrder": 20, "MiddleRobotSummonOrder": 30,
    "LargeRobotSummonOrder": 100, "BossRobotSummonOrder": 200,
    "AcientTablet": 15, "StarSand": 15, "FlameBreath": 15,
    "FrostPotion": 15, "ThornAmulet": 15, "IronWhistle": 15,
}
MINE_CAPACITY = 10          # 任务书 4.1：每个矿可采 10 次


# ---------------------------------------------------------------- 小工具
def cheb(a: Pos, b: Pos) -> int:
    """切比雪夫距离（和游戏内规则一致）。"""
    return max(abs(a.x - b.x), abs(a.y - b.y))


def footprint_distance(pos: Pos) -> int:
    """该格到基地 2x2 外形的距离：1 = 武器区，2 = 城墙区。"""
    return min(cheb(pos, cell) for cell in FOOTPRINT)


def neighbours(pos: Pos) -> list[Pos]:
    """周围 8 格。"""
    return [
        Pos(pos.x + dx, pos.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx or dy)
    ]


class World:
    """简化版沙盒：只实现验证需求所必需的规则。

    它不是完整判题器，但足以暴露「建造区不对、卖不出矿、归位迟到、
    开火超射程」这类真实问题。
    """

    def __init__(self, seed: int = 7, vendor_prices: dict | None = None,
                 llm_mode: str = "answer", task_gold: int = 80) -> None:
        self.rng = random.Random(seed)
        self.vendor_prices = dict(vendor_prices or VENDOR_PRICES)
        self.llm_mode = llm_mode      # "answer"=正常作答, "cmd_only"=只会要命令（触发兜底提交）
        self.task_gold = task_gold    # 单个自进化任务的金币奖励（两个合计 2*task_gold）
        self.round_no = 0
        self.gold = 75                # 任务书 4.5.3：初始 75 金，刚好够 3 座火箭
        self.score = 0
        self.zones: dict[Pos, str] = dict(BASE_ZONES)
        self.mine_left: dict[Pos, int] = {          # 每个矿还剩几次可采
            pos: MINE_CAPACITY for pos, kind in BASE_ZONES.items()
            if kind in ("stone", "iron", "copper")
        }
        # 我方单位：两个工人 + 开拓者 + 基地，坐标取自样例报文。
        self.units: dict[int, dict] = {}
        for unit_id, pos, kind in (
            (10010, Pos(5, 23), "worker"),
            (10011, Pos(10, 12), "pioneer"),
            (10012, Pos(10, 16), "worker"),
        ):
            self.units[unit_id] = {
                "id": unit_id, "pos": pos, "roleType": kind, "health": 220,
                "level": 0, "cooldown": 0, "attackRange": 0,
                "backPackCapability": 100,
                "backpack": [],
            }
        self.units[10013] = {
            "id": 10013, "pos": STATION_POS, "roleType": "station",
            "health": 1500, "level": 1, "cooldown": 0, "attackRange": 0,
            "backPackCapability": 0, "backpack": [],
        }
        self.gatling_seq = 0          # 新建武器的自增编号
        self.wall_seq = 0             # 新建围墙的自增编号
        self.robots: dict[int, dict] = {}
        self.robot_seq = 0
        # 任务 / LLM / 沙盒这三条通道的模拟状态
        self.phase_task = ""
        self.llm_resp = ""
        self.last_cmd_result = ""
        self.pending_llm = False
        self.task_stage = 0
        # 供断言使用的统计
        self.events: dict = {
            "accepted": 0, "submitted": 0, "answers": [],
            "boss_bought": None, "boss_used": None, "gold_200": None,
            "attacks": 0, "damage": 0, "bad_shots": 0,
        }
        self.last_response: dict = {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
        self.last_actions: dict[int, str] = {}

    # ------------------------------------------------------------- 序列化
    def roles(self) -> list[dict]:
        """把内部单位格式转成报文里的 Role 数组。"""
        return [
            {
                "id": unit["id"], "pos": {"x": unit["pos"].x, "y": unit["pos"].y},
                "roleType": unit["roleType"], "health": unit["health"],
                "attackPower": 0, "attackRange": unit["attackRange"],
                "level": unit["level"], "cooldown": unit["cooldown"],
                "backPackCapability": unit["backPackCapability"],
                "backpack": list(unit["backpack"]),
            }
            for unit in self.units.values()
        ]

    def payload(self) -> dict:
        """拼出本回合的 Request，字段与 docs/request.txt 完全同构。"""
        return {
            "roundNo": self.round_no,
            "mapInfo": {
                "width": WIDTH, "height": HEIGHT,
                "zones": [
                    {"neutralType": kind, "pos": {"x": pos.x, "y": pos.y}}
                    for pos, kind in sorted(self.zones.items())
                ],
            },
            "teamOur": {
                "type": "challenger", "teamId": "1", "teamName": "smoke",
                "goldNum": self.gold, "totalScore": self.score,
                "playerTasks": [
                    {"taskType": "自进化类1", "taskPosition": {"x": 14, "y": 14},
                     "coldDownRounds": 0, "scoreReward": 50,
                     "goldReward": self.task_gold,
                     "isValid": True, "timeoutRounds": 30},
                    {"taskType": "自进化类2", "taskPosition": {"x": 17, "y": 17},
                     "coldDownRounds": 0, "scoreReward": 50,
                     "goldReward": self.task_gold,
                     "isValid": True, "timeoutRounds": 30},
                ],
                "roles": self.roles(),
            },
            "teamEnemy": {"roles": [
                {"id": 20013, "pos": {"x": ENEMY_STATION.x, "y": ENEMY_STATION.y},
                 "roleType": "station", "health": 1500, "level": 1},
            ]},
            "robot": {"roles": [
                {"id": r["id"], "pos": {"x": r["pos"].x, "y": r["pos"].y},
                 "roleType": r["kind"], "health": r["health"],
                 "abnormalState": "", "targetTeam": "challenger"}
                for r in self.robots.values()
            ]},
            "phaseTask": self.phase_task,
            # 上一回合各角色动作是否合法：由 apply() 逐条写回
            "lastRoundRoleActionResults": {
                str(unit_id): self.last_ok.get(unit_id, True)
                for unit_id in self.units
            },
            "lastSummonTreasureResult": 0,
            "llmResp": self.llm_resp,
            "worldNews": {"officialNews": "今日无重大新闻", "folkLegends": "无事发生"},
            "lastCmdResult": self.last_cmd_result,
            "vendorShopList": [
                {"name": name, "price": price}
                for name, price in self.vendor_prices.items()
            ],
            "weaponShopList": [
                {"name": name, "price": price} for name, price in WEAPON_PRICES.items()
            ],
            "errors": [],
        }

    # ------------------------------------------------------------- 规则
    def occupied(self) -> set[Pos]:
        """当前被占用的格子：中立元素 + 我方单位 + 机器人。"""
        cells: set[Pos] = {pos for pos in self.zones}
        for unit in self.units.values():
            if unit["roleType"] == "station":
                cells.update(FOOTPRINT)          # 基地占 4 格
            else:
                cells.add(unit["pos"])
        cells.update(robot["pos"] for robot in self.robots.values())
        return cells

    def spawn_robots(self) -> None:
        """夜晚第一回合在敌方侧生成 4 台小机器人。"""
        count = 4
        for index in range(count):
            self.robot_seq += 1
            self.robots[self.robot_seq] = {
                "id": 30000 + self.robot_seq,
                "pos": Pos(38 - index % 2, 3 + index // 2),
                "kind": "smallRobot",
                "health": 40,
            }

    def respawn_mine(self) -> None:
        """矿采空后在随机空地重新生成一个（不会落在可建造区里）。"""
        for _ in range(200):
            pos = Pos(self.rng.randint(0, WIDTH - 1), self.rng.randint(0, HEIGHT - 1))
            band = footprint_distance(pos)
            if band in (1, 2):          # 矿区不会生成在可建造区域内
                continue
            if pos in self.zones or pos in self.occupied():
                continue
            self.zones[pos] = self.rng.choice(["stone", "iron", "copper"])
            self.mine_left[pos] = MINE_CAPACITY
            return

    def apply(self, response: dict) -> None:
        """把选手的一整份指令落到世界状态上，并记录每条指令是否合法。"""
        # 默认全部合法，逐条覆盖。
        self.last_ok: dict[int, bool] = {unit_id: True for unit_id in self.units}
        for key, command in response["roleCommandMap"].items():
            unit_id = int(key)
            unit = self.units.get(unit_id)
            if unit is None:
                continue
            action = command.get("action")
            targets = command.get("targetPos") or []
            target = Pos(targets[0]["x"], targets[0]["y"]) if targets else None
            if action == "move" and target is not None:
                self.last_ok[unit_id] = self._move(unit, target)
            elif action == "build" and target is not None:
                self.last_ok[unit_id] = self._build(unit, target, command.get("name"))
            elif action == "collect" and target is not None:
                self.last_ok[unit_id] = self._collect(unit, target)
            elif action == "sell":
                self.last_ok[unit_id] = self._sell(unit, command)
            elif action == "buy":
                self.last_ok[unit_id] = self._buy(unit, command)
            elif action == "use":
                self.last_ok[unit_id] = self._use(unit, command)
            elif action == "drop":
                name = command.get("name")
                if name in unit["backpack"]:
                    unit["backpack"].remove(name)
            elif action == "acceptTask":
                # 简化：一接任务就立刻下发任务原文。
                self.events["accepted"] += 1
                self.phase_task = "示例任务：请在沙盒中计算 1+1 并给出答案。"
            elif action == "submitAnswer":
                self.events["submitted"] += 1
                answer = str(command.get("taskAnswer") or "").strip()
                self.events["answers"].append(answer)
                # 任务书第六章：金币 = 任务金币奖励 * 通过率
                rate = 1.0 if answer == "2" else 0.2
                self.gold += int(self.task_gold * rate)
                self.phase_task = ""
            elif action == "attack":
                self._attack(unit_id, command, target)
        self.last_response = response

    def _move(self, unit: dict, target: Pos) -> bool:
        """只能走到相邻的空格上。"""
        if unit["pos"] != target and target in neighbours(unit["pos"]) \
                and 0 <= target.x < WIDTH and 0 <= target.y < HEIGHT \
                and target not in self.occupied():
            unit["pos"] = target
            return True
        return False

    def _build(self, unit: dict, target: Pos, name: str | None) -> bool:
        """按「外一圈=武器、再外一圈=城墙」校验建造区，并扣资源。"""
        if cheb(unit["pos"], target) > 1 or target in self.occupied():
            return False
        band = footprint_distance(target)
        if name == ROCKET:
            if band != 1:
                return False                 # 火箭必须建在武器可建造区
            if self.gold < 25:
                return False
            self.gold -= 25
            self.gatling_seq += 1
            new_id = 10040 + self.gatling_seq - 1
            self.units[new_id] = {
                "id": new_id, "pos": target, "roleType": ROCKET, "health": 1000,
                "level": 1, "cooldown": 0, "attackRange": 10,
                "backPackCapability": 0, "backpack": [],
            }
            return True
        if name == WALL:
            if band != 2:
                return False                 # 城墙必须建在城墙可建造区
            if "stone" not in unit["backpack"]:
                return False
            unit["backpack"].remove("stone")
            self.wall_seq += 1
            new_id = 40000 + self.wall_seq - 1
            self.units[new_id] = {
                "id": new_id, "pos": target, "roleType": WALL, "health": 1000,
                "level": 1, "cooldown": 0, "attackRange": 0,
                "backPackCapability": 0, "backpack": [],
            }
            return True
        return False

    def _collect(self, unit: dict, target: Pos) -> bool:
        """采一块矿进背包；矿被采空就移除并异地重生。"""
        kind = self.zones.get(target)
        if kind not in ("stone", "iron", "copper"):
            return False
        if cheb(unit["pos"], target) > 1 or unit["pos"] == target:
            return False
        if len(unit["backpack"]) >= unit["backPackCapability"]:
            return False
        unit["backpack"].append(kind)
        self.mine_left[target] = self.mine_left.get(target, MINE_CAPACITY) - 1
        if self.mine_left[target] <= 0:
            self.zones.pop(target, None)
            self.mine_left.pop(target, None)
            self.respawn_mine()
        return True

    def _sell(self, unit: dict, command: dict) -> bool:
        """在小贩旁边卖矿换金币。"""
        vendor = next((p for p, k in self.zones.items() if k == "vendor"), None)
        if vendor is None or cheb(unit["pos"], vendor) > 1:
            return False
        name = command.get("name")
        num = int(command.get("num") or 1)
        num = min(num, unit["backpack"].count(name))   # 背包不够就按实际数量卖
        if num <= 0:
            return False
        for _ in range(num):
            unit["backpack"].remove(name)
        self.gold += self.vendor_prices.get(name, 0) * num
        return True

    def _buy(self, unit: dict, command: dict) -> bool:
        """在武器商店买东西；钱不够或背包放不下就失败。"""
        shop = next((p for p, k in self.zones.items() if k == "weaponShop"), None)
        if shop is None or cheb(unit["pos"], shop) > 1:
            return False
        name = command.get("name")
        num = int(command.get("num") or 1)
        price = WEAPON_PRICES.get(name, 0)
        if self.gold < price * num or len(unit["backpack"]) + num > unit["backPackCapability"]:
            return False
        self.gold -= price * num
        unit["backpack"].extend([name] * num)
        if name == BOSS_ORDER:
            self.events["boss_bought"] = self.round_no
        return True

    def _use(self, unit: dict, command: dict) -> bool:
        """使用背包里的道具（这里只关心 BOSS 召唤令的时点）。"""
        name = command.get("name")
        if name not in unit["backpack"]:
            return False
        unit["backpack"].remove(name)
        if name == BOSS_ORDER:
            self.events["boss_used"] = self.round_no
        return True

    def _attack(self, tower_id: int, command: dict, target: Pos | None) -> None:
        """结算一次火箭攻击：中心 20、周围 8 格 10，并让武器进入冷却。"""
        if target is None:
            return
        tower = self.units.get(tower_id)
        self.events["attacks"] += 1
        # 超出射程算「无效射击」，用来抓策略层的低级错误。
        if tower is not None and cheb(tower["pos"], target) > tower["attackRange"]:
            self.events["bad_shots"] += 1
        for robot in self.robots.values():
            span = cheb(robot["pos"], target)
            if span == 0:
                robot["health"] -= ROCKET_CENTER_DAMAGE
                self.events["damage"] += ROCKET_CENTER_DAMAGE
            elif span == 1:
                robot["health"] -= ROCKET_SPLASH_DAMAGE
                self.events["damage"] += ROCKET_SPLASH_DAMAGE
        if tower is not None:
            tower["cooldown"] = 3            # 任务书：火箭发射后冷却 3 回合

    def tick_start(self) -> None:
        """回合开始：冷却递减，并把上一回合的 prompt/executeCmd 结果准备好。"""
        for unit in self.units.values():
            if unit["cooldown"] > 0:
                unit["cooldown"] -= 1
        self.llm_resp = ""
        self.last_cmd_result = ""
        # 模拟 LLM：上一回合发过 prompt，这回合就给回复。
        if self.pending_llm:
            self.pending_llm = False
            self.task_stage += 1
            if self.llm_mode == "cmd_only":
                self.llm_resp = "CMD: echo 1+1"        # 永远只给命令，逼出兜底分支
            else:
                self.llm_resp = "CMD: echo 1+1" if self.task_stage % 2 else "ANSWER: 2"
        # 模拟沙盒：上一回合发过 executeCmd，这回合就给输出。
        if self.last_response.get("executeCmd"):
            self.last_cmd_result = "[exitCode:0]\n2"
        if self.last_response.get("prompt"):
            self.pending_llm = True

    def move_robots(self) -> None:
        """夜晚机器人朝我方基地推进（简化：贪心走一格），并清理已阵亡的。"""
        for key, robot in list(self.robots.items()):
            if robot["health"] <= 0:
                del self.robots[key]
                continue
            occupied = self.occupied()
            best: tuple[int, Pos] | None = None
            for step in neighbours(robot["pos"]):
                if not (0 <= step.x < WIDTH and 0 <= step.y < HEIGHT):
                    continue
                if step in occupied:
                    continue
                candidate = (cheb(step, STATION_POS), step)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is not None and best[0] < cheb(robot["pos"], STATION_POS):
                robot["pos"] = best[1]

    def tick_end(self) -> None:
        """回合结束：打点统计、拍第 70 回合快照、生成/推进夜晚机器人。"""
        if self.gold >= 200 and self.events["gold_200"] is None:
            self.events["gold_200"] = self.round_no
        if self.round_no == 70:
            # 需求只要求第 70 回合三人就位，所以在这一回合留一份快照供断言。
            self.snapshot_70 = {
                "rockets": [
                    u for u in self.units.values() if u["roleType"] == ROCKET
                ],
                "walls": [u for u in self.units.values() if u["roleType"] == WALL],
                "units": dict(self.units),
            }
        if (self.round_no - 1) % 130 < 70:
            if self.round_no % 130 == 70:
                self.spawn_robots()          # 白天最后一回合末生成当晚的机器人
        else:
            self.move_robots()               # 夜晚期间机器人持续逼近


def check_goals(world: "World") -> tuple[int, int, list[str]]:
    """校验第 70 回合的结构目标，返回 (火箭数, 城墙数, 失败原因)。

    同时校验「建造区是否用对」——火箭必须在外一圈、城墙必须在再外一圈，
    这样如果布局算错了，测试会直接报出来而不是默默通过。
    """
    snapshot = getattr(world, "snapshot_70", None)
    if snapshot is None:
        return 0, 0, ["没有第 70 回合快照"]
    rockets = snapshot["rockets"]
    walls = snapshot["walls"]
    problems: list[str] = []
    if len(rockets) != 3:
        problems.append(f"火箭数量 {len(rockets)} != 3")
    if len(walls) != 10:
        problems.append(f"城墙数量 {len(walls)} != 10")
    for rocket in rockets:
        if footprint_distance(rocket["pos"]) != 1:
            problems.append(f"火箭 {rocket['pos']} 不在武器可建造区")
    for wall in walls:
        if footprint_distance(wall["pos"]) != 2:
            problems.append(f"城墙 {wall['pos']} 不在城墙可建造区")
    for unit_id in (10010, 10011, 10012):
        unit = snapshot["units"].get(unit_id)
        if unit is None:
            problems.append(f"角色 {unit_id} 不见了")
            continue
        if not any(cheb(unit["pos"], r["pos"]) <= 1 for r in rockets):
            problems.append(
                f"第 70 回合角色 {unit_id} 在 {unit['pos']}，不在任何火箭旁"
            )
    return len(rockets), len(walls), problems


def run(prices: dict, seed: int = 7, expect_boss: bool = True,
        min_gold: int = 0, llm_mode: str = "answer",
        task_gold: int = 80) -> list[str]:
    """跑完第一天 130 回合，返回失败原因列表（空表示全部通过）。

    参数都是为了让同一个函数能跑不同场景：矿价、任务奖励、LLM 行为等。
    """
    # 策略的记忆是模块级单例，跑新场景前必须清空，否则会带着上一局的进度。
    memory_module.MEMORY.__init__()
    world = World(seed=seed, vendor_prices=prices, llm_mode=llm_mode,
                  task_gold=task_gold)
    trace = []
    for round_no in range(1, 131):
        world.round_no = round_no
        world.last_ok = {unit_id: True for unit_id in world.units}
        world.tick_start()                       # 准备 llmResp / lastCmdResult
        response = decide(world.payload())       # 进程内直接调策略
        world.apply(response)                    # 落盘并记录指令合法性
        world.tick_end()
        # 只在若干关键回合取样打印，避免刷屏。
        if round_no in (1, 10, 20, 30, 40, 50, 60, 65, 70, 71, 75, 100, 130):
            trace.append((round_no, world.gold, len(world.units), len(world.robots),
                          len(response["roleCommandMap"])))

    print("round  gold  units  robots  cmds")
    for row in trace:
        print(f"{row[0]:>5}  {row[1]:>4}  {row[2]:>5}  {row[3]:>6}  {row[4]:>4}")

    snapshot = getattr(world, "snapshot_70", None)
    failures: list[str] = []
    if snapshot is None:
        failures.append("没有第 70 回合快照")
    else:
        rocket_count, wall_count, structure_failures = check_goals(world)
        print(f"\n第 70 回合：火箭 {rocket_count} 座 / 城墙 {wall_count} 面")
        failures.extend(structure_failures)

    events = world.events
    print(f"任务：接取 {events['accepted']} 次 / 提交 {events['submitted']} 次 "
          f"答案={events['answers']}")
    print(f"金币首次 >=200 的回合：{events['gold_200']}")
    print(f"BOSS 令购买回合：{events['boss_bought']} / 使用回合：{events['boss_used']}")
    print(f"夜晚开火 {events['attacks']} 次，累计伤害 {events['damage']}，"
          f"超出射程的射击 {events['bad_shots']} 次")

    # ---- 断言 ----
    if events["submitted"] < 2:
        failures.append(f"完成的自进化任务数 {events['submitted']} < 2")
    if events["attacks"] == 0:
        failures.append("夜晚一次都没开火")
    if events["damage"] == 0:
        failures.append("夜晚开火但没打出任何伤害")
    if events["bad_shots"]:
        failures.append(f"有 {events['bad_shots']} 次射击超出武器射程")
    if world.gold < min_gold:
        failures.append(f"第 130 回合金币 {world.gold} < 期望下限 {min_gold}")
    if expect_boss:
        if events["boss_bought"] is None:
            failures.append("没有买到 BOSS 召唤令（金币没攒够 200？）")
        if events["boss_used"] is None:
            failures.append("没有使用 BOSS 召唤令")
        if events["boss_used"] is not None and events["boss_used"] > 70:
            failures.append("BOSS 召唤令在入夜后才使用，赶不上当晚")
    return failures


def main() -> int:
    # 每个场景 = (标题, 小贩收购价, run() 的额外参数)
    scenarios = [
        # 主场景：样例收购价 + 两个自进化任务合计 160 金（实际 request 的量级）。
        # 75(初始) - 75(3 火箭) + 160(任务) + 一轮卖矿 >= 200，应当买得起 BOSS 令。
        ("样例矿价 + 任务 160 金",
         {"stone": 1, "iron": 3, "copper": 5}, {"expect_boss": True}),
        # 高矿价：矿价波动时同样要能走通整条链路。
        ("高矿价 copper=25",
         {"stone": 1, "iron": 3, "copper": 25}, {"expect_boss": True}),
        # 任务奖励极低：验证结构目标仍然全达成（只是买不起 BOSS 令）。
        ("任务奖励低 (30/个)",
         {"stone": 1, "iron": 3, "copper": 5},
         {"expect_boss": False, "min_gold": 100, "task_gold": 30}),
        # LLM 一直只给命令不给答案：验证轮次用尽时会兜底提交，任务不会白做。
        ("LLM 不作答（兜底提交）",
         {"stone": 1, "iron": 3, "copper": 5},
         {"expect_boss": False, "llm_mode": "cmd_only"}),
    ]
    failures: list[str] = []
    for title, prices, options in scenarios:
        print(f"\n########## {title} ##########")
        problems = run(prices, **options)
        if problems:
            failures.extend(f"[{title}] {item}" for item in problems)
        else:
            print(f"---- {title}: 通过 ----")

    if failures:
        print("\n==== 失败 ====")
        for item in failures:
            print(" -", item)
        return 1
    print("\n==== 全部场景通过 ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
