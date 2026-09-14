"""八方向栅格寻路。

任务书 4.5.4：角色可朝 8 个方向移动一格，距离用切比雪夫距离
（即斜着走和直着走代价一样，都是 1 回合 1 格）。

对外只暴露两个函数：
    first_step()    朝最近目标走的第一步（下层再调一次就是完整路径）
    travel_rounds() 到最近目标需要几回合

两者都接受「目标格集合」，因为实际需求几乎都是「走到某建筑旁边」，
而不是「走到某个确定的格子」——建筑本身是障碍，只能站它周围。
"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count

from .protocol import Pos, Turn, distance, neighbours


def _search(
    turn: Turn,
    start: Pos,
    goals: frozenset[Pos],
    blocked: frozenset[Pos],
) -> tuple[dict[Pos, Pos | None], int, Pos] | None:
    """A* 搜索：在 8 连通栅格上找「最近的合法目标格」。

    返回 (每个格子的前驱, 总步数, 命中的目标格)；无解返回 None。

    启发函数用切比雪夫距离——在 8 方向、单步代价恒为 1 的栅格上，
    它不会高估真实步数，因此 A* 求出的就是最短路。
    """
    if not goals:
        return None

    def heuristic(pos: Pos) -> int:
        # 到「最近的那个目标」的估计步数。
        return min(distance(pos, goal) for goal in goals)

    order = count()          # 用于给同优先级节点排个稳定顺序，避免比较 Pos
    frontier: list[tuple[int, int, int, Pos]] = [
        (heuristic(start), 0, next(order), start)
    ]
    came: dict[Pos, Pos | None] = {start: None}   # 前驱表，用来回溯路径
    best: dict[Pos, int] = {start: 0}             # 已确认的最优步数

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in goals:
            return came, cost, current            # 出堆时才是最优，直接返回
        if cost > best.get(current, cost):
            continue                              # 过期条目，跳过
        for step in neighbours(current):
            # 任务书 4.1：中立元素与所有单位所在格都阻挡移动；出界也不算合法格。
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue                          # 已经有更优路径到这一格
            best[step] = new_cost
            came[step] = current
            heappush(
                frontier,
                (new_cost + heuristic(step), new_cost, next(order), step),
            )
    return None


def first_step(
    turn: Turn,
    start: Pos,
    goals: frozenset[Pos] | set[Pos],
    blocked: frozenset[Pos] | set[Pos],
) -> Pos | None:
    """返回从 start 走向最近目标的第一步。

    已经在目标上、没有目标、或完全不可达时都返回 None（调用方按「原地不动」处理）。
    """
    result = _search(turn, start, frozenset(goals), frozenset(blocked))
    if result is None:
        return None
    came, _, goal = result
    if goal == start:
        return None
    # 从目标沿前驱表回溯到起点的下一格。
    current = goal
    previous = came[current]
    while previous is not None and previous != start:
        current = previous
        previous = came[current]
    return current


def travel_rounds(
    turn: Turn,
    start: Pos,
    goals: frozenset[Pos] | set[Pos],
    blocked: frozenset[Pos] | set[Pos],
) -> int | None:
    """走到最近目标需要几回合（每回合只能走一格），不可达返回 None。

    策略层用它来算「还来不来得及赶回火箭旁」，所以必须是最短路长度而不是估计值。
    """
    result = _search(turn, start, frozenset(goals), frozenset(blocked))
    if result is None:
        return None
    return result[1]
