"""
Multiple Choice Knapsack Problem (MCKP) solver via dynamic programming.

Given N groups of items (one group per quantizable component, items are
bit-width choices), select exactly one item from each group to maximise
total value while keeping total cost within the budget.

This module is intentionally self-contained: the fragility / quantization
code only *prepares* the DP inputs; it never calls into this solver
directly during sensitivity measurement.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DPEntry:
    """One (component, bit-width) choice for the DP table."""
    component_id: int
    bit: int
    cost: int          # param_count * bit
    value: float       # Omega(c, min_bit) - Omega(c, bit)


def _auto_cost_scale(budget: int, max_dp_width: int = 50_000) -> int:
    """Pick a cost-scaling factor that keeps the DP table under *max_dp_width*."""
    if budget <= max_dp_width:
        return 1
    return 2 ** math.ceil(math.log2(budget / max_dp_width))


def _min_cost_suffix(dp_table: List[List[DPEntry]]) -> List[int]:
    """Return suffix sums of the minimum per-group costs."""
    suffix = [0] * (len(dp_table) + 1)
    for i in range(len(dp_table) - 1, -1, -1):
        if not dp_table[i]:
            raise ValueError(f"MCKP: empty group at index {i}")
        suffix[i] = suffix[i + 1] + min(entry.cost for entry in dp_table[i])
    return suffix


def solve_random_allocation(
    dp_table: List[List[DPEntry]],
    budget: int,
) -> Dict[int, int]:
    """Randomly pick one feasible bit-width per component under the budget."""
    min_cost_suffix = _min_cost_suffix(dp_table)
    if min_cost_suffix[0] > budget:
        raise ValueError("Random allocation: no feasible solution within the given budget")

    assignments: Dict[int, int] = {}
    spent = 0

    for i, group in enumerate(dp_table):
        min_future_cost = min_cost_suffix[i + 1]
        feasible = [
            entry for entry in group
            if spent + entry.cost + min_future_cost <= budget
        ]
        if not feasible:
            raise ValueError(
                f"Random allocation: no feasible assignment for group {i} under the budget"
            )

        choice = random.choice(feasible)
        assignments[choice.component_id] = choice.bit
        spent += choice.cost

    return assignments


def solve_greedy_allocation(
    dp_table: List[List[DPEntry]],
    budget: int,
) -> Dict[int, int]:
    """Greedy bit allocation via repeated best marginal value-per-cost upgrades."""
    groups: List[List[DPEntry]] = []
    selected: List[int] = []
    spent = 0

    for i, group in enumerate(dp_table):
        if not group:
            raise ValueError(f"Greedy allocation: empty group at index {i}")

        ordered = sorted(group, key=lambda entry: (entry.cost, entry.value))
        cheapest_cost = ordered[0].cost
        cheapest = max(
            (entry for entry in ordered if entry.cost == cheapest_cost),
            key=lambda entry: entry.value,
        )
        ordered = sorted(group, key=lambda entry: (entry.cost, entry.value))
        groups.append(ordered)
        selected_idx = ordered.index(cheapest)
        selected.append(selected_idx)
        spent += cheapest.cost

    if spent > budget:
        raise ValueError("Greedy allocation: no feasible solution within the given budget")

    remaining = budget - spent

    while remaining > 0:
        best_upgrade = None

        for group_idx, group in enumerate(groups):
            current = group[selected[group_idx]]
            for cand_idx, candidate in enumerate(group):
                if cand_idx == selected[group_idx]:
                    continue
                delta_cost = candidate.cost - current.cost
                if delta_cost <= 0 or delta_cost > remaining:
                    continue

                delta_value = candidate.value - current.value
                if delta_value <= 0:
                    continue
                score = delta_value / delta_cost
                rank = (score, delta_value, -delta_cost)
                if best_upgrade is None or rank > best_upgrade[0]:
                    best_upgrade = (rank, group_idx, cand_idx, delta_cost)

        if best_upgrade is None:
            break

        _, group_idx, cand_idx, delta_cost = best_upgrade
        selected[group_idx] = cand_idx
        remaining -= delta_cost

    assignments: Dict[int, int] = {}
    for group_idx, group in enumerate(groups):
        choice = group[selected[group_idx]]
        assignments[choice.component_id] = choice.bit

    return assignments


def solve_mckp(
    dp_table: List[List[DPEntry]],
    budget: int,
    cost_scale: Optional[int] = None,
) -> Dict[int, int]:
    """Solve the Multiple Choice Knapsack Problem.

    Picks exactly one bit-width per component (group), maximising total
    drift-reduction value while keeping total cost <= *budget*.

    Parameters
    ----------
    dp_table : list of list of DPEntry
        ``dp_table[i]`` is the group for component *i*; each element is a
        ``DPEntry`` for one candidate bit-width.
    budget : int
        Total cost capacity  (typically ``total_params * target_avg_bit``).
    cost_scale : int or None
        Divide all costs by this factor before running DP so the table
        stays in memory.  Auto-detected when ``None``.

    Returns
    -------
    dict[int, int]
        ``{component_id: assigned_bit}``

    Raises
    ------
    ValueError
        If no feasible solution exists within the budget.
    """
    if cost_scale is None:
        cost_scale = _auto_cost_scale(budget)

    n_groups = len(dp_table)
    W = budget // cost_scale
    NEG_INF = float('-inf')

    # dp[w] = best total value reachable with scaled cost exactly w
    dp = [NEG_INF] * (W + 1)
    dp[0] = 0.0

    # choices[i][w] = (item_index_in_group, previous_w)
    choices: List[list] = []

    for i in range(n_groups):
        new_dp = [NEG_INF] * (W + 1)
        bt: list = [None] * (W + 1)

        for w in range(W + 1):
            if dp[w] <= NEG_INF:
                continue
            for j, entry in enumerate(dp_table[i]):
                sc = math.ceil(entry.cost / cost_scale)
                nw = w + sc
                if nw <= W and dp[w] + entry.value > new_dp[nw]:
                    new_dp[nw] = dp[w] + entry.value
                    bt[nw] = (j, w)

        dp = new_dp
        choices.append(bt)

    # --- find the best reachable total cost ---
    best_w = max(range(W + 1), key=lambda w: dp[w])
    if dp[best_w] <= NEG_INF:
        raise ValueError("MCKP: no feasible solution within the given budget")

    # --- backtrack to recover per-component assignments ---
    assignments: Dict[int, int] = {}
    w = best_w
    for i in range(n_groups - 1, -1, -1):
        if choices[i][w] is None:
            raise ValueError(f"MCKP: backtracking failed at group {i}")
        j, w = choices[i][w]
        assignments[dp_table[i][j].component_id] = dp_table[i][j].bit

    return assignments


def solve_bit_allocation(
    dp_table: List[List[DPEntry]],
    budget: int,
    strategy: str = "mckp",
    cost_scale: Optional[int] = None,
) -> Dict[int, int]:
    """Dispatch to the requested bit-allocation strategy."""
    strategy = strategy.lower()

    if strategy == "mckp":
        return solve_mckp(dp_table, budget, cost_scale)
    if strategy == "greedy":
        return solve_greedy_allocation(dp_table, budget)
    if strategy == "random":
        return solve_random_allocation(dp_table, budget)

    raise ValueError(
        f"Unknown allocation strategy '{strategy}'. "
        "Expected one of: mckp, greedy, random."
    )
