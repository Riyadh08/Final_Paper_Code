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
