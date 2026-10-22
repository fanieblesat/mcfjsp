"""Heuristic pricers: cheap constructions of cell schedules (columns).

A pricer receives the cell problem and the weights (w, v) and returns a list
of feasible columns. Column generation keeps those with negative reduced
cost. Pricers never affect the certificate (Lemma 'Anytime certificate').

Included:
  RulePricer      list scheduling inside the cell with priority (w-v)/p_min
                  (descending), 'as early as possible', plus a right-shift
                  pass that delays operations with negative net weight.
  LPRoundPricer   priorities = fractional start times of the time-indexed LP
                  solution (requires the LP oracle's frac_start).
The learned PPO+GNN pricer plugs into the same interface (agents/ package).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .oracles import CellProblem, Column


def cell_list_schedule(cp: CellProblem, priority: Dict[int, float], w: Dict[int, float],
                       v: Dict[int, float], right_shift: bool = True) -> Optional[Column]:
    """Serial schedule generation inside one cell. Returns None if the
    construction violates a deadline (column infeasible)."""
    inst = cp.inst
    ops = cp.ops
    pred_in_cell = {a: inst.op_pred[a] if (inst.op_pred[a] >= 0 and inst.op_cell[inst.op_pred[a]] == cp.cell) else -1
                    for a in ops}
    mfree = {i: 0 for i in cp.machines}
    start: Dict[int, int] = {}; assign: Dict[int, int] = {}
    remaining = set(ops)
    while remaining:
        elig = [a for a in remaining if pred_in_cell[a] < 0 or pred_in_cell[a] in start]
        a = min(elig, key=lambda a: (priority[a], a))
        es = cp.heads[a]
        if pred_in_cell[a] >= 0:
            es = max(es, start[pred_in_cell[a]] + inst.op_proc[pred_in_cell[a]][assign[pred_in_cell[a]]])
        best = None
        for i, p in inst.op_proc[a].items():
            s = max(es, mfree[i])
            cost = w[a] * (s + p) - v[a] * s
            key = (cost, s + p, i)
            if best is None or key < best[0]:
                best = (key, i, s)
        _, i, s = best
        if s + inst.op_proc[a][i] + cp.tails[a] > cp.T:
            return None
        start[a] = s; assign[a] = i; mfree[i] = s + inst.op_proc[a][i]
        remaining.discard(a)
    if right_shift:
        _right_shift(cp, start, assign, w, v)
    return cp.column_from(start, assign)


def _right_shift(cp: CellProblem, start: Dict[int, int], assign: Dict[int, int], w, v) -> None:
    """Delay operations with w_a < v_a as late as feasible (reverse pass)."""
    inst = cp.inst
    by_m: Dict[int, List[int]] = {}
    for a in start:
        by_m.setdefault(assign[a], []).append(a)
    for i in by_m:
        by_m[i].sort(key=lambda a: start[a])
    next_on_m = {}
    for i, lst in by_m.items():
        for k, a in enumerate(lst):
            next_on_m[a] = lst[k + 1] if k + 1 < len(lst) else -1
    succ_in_cell = {a: inst.op_succ[a] if (inst.op_succ[a] >= 0 and inst.op_succ[a] in start) else -1 for a in start}
    for a in sorted(start, key=lambda a: -start[a]):
        if w[a] >= v[a]:
            continue
        p = inst.op_proc[a][assign[a]]
        latest = cp.T - cp.tails[a] - p
        if next_on_m[a] >= 0:
            latest = min(latest, start[next_on_m[a]] - p)
        if succ_in_cell[a] >= 0:
            latest = min(latest, start[succ_in_cell[a]] - p)
        if latest > start[a]:
            start[a] = latest


class RulePricer:
    name = "rule"

    def __call__(self, cp: CellProblem, w, v, context: dict) -> List[Column]:
        pmin = cp.inst.p_min
        pri1 = {a: -(w[a] - v[a]) / pmin[a] for a in cp.ops}
        pri2 = {a: -(w[a] - v[a]) for a in cp.ops}
        pri3 = {a: (cp.heads[a], -(w[a] - v[a]) / pmin[a]) for a in cp.ops}
        cols = []
        for pri in (pri1, pri2, pri3):
            for rs in (True, False):
                col = cell_list_schedule(cp, pri, w, v, right_shift=rs)
                if col is not None:
                    cols.append(col)
        return cols


class LPRoundPricer:
    name = "lp-round"

    def __call__(self, cp: CellProblem, w, v, context: dict) -> List[Column]:
        fs = context.get("frac_start")
        if not fs:
            return []
        pri = {a: fs[a] for a in cp.ops}
        cols = []
        for rs in (True, False):
            col = cell_list_schedule(cp, pri, w, v, right_shift=rs)
            if col is not None:
                cols.append(col)
        return cols
