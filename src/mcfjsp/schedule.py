"""Plant schedules, feasibility checking and semi-active evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .instances import Instance


@dataclass
class Schedule:
    """assign[a] = machine, start[a] = start time, comp[a] = completion."""
    assign: List[int]
    start: List[int]
    comp: List[int]

    @property
    def makespan(self) -> int:
        return max(self.comp) if self.comp else 0

    def machine_orders(self, inst: Instance) -> List[List[int]]:
        orders: List[List[int]] = [[] for _ in range(inst.n_machines)]
        for a in range(inst.n_ops):
            orders[self.assign[a]].append(a)
        for i in range(inst.n_machines):
            orders[i].sort(key=lambda a: (self.start[a], a))
        return orders


def check_feasible(inst: Instance, sch: Schedule, durations: Optional[Dict[int, Dict[int, int]]] = None,
                   release: Optional[List[int]] = None) -> None:
    """Raise AssertionError if the schedule violates any constraint."""
    rel = release if release is not None else inst.release
    for a in range(inst.n_ops):
        i = sch.assign[a]
        assert i in inst.op_proc[a], f"op {a} on ineligible machine {i}"
        p = durations[a][i] if durations is not None else inst.op_proc[a][i]
        assert sch.comp[a] == sch.start[a] + p, f"op {a}: completion inconsistent"
        if inst.op_pos[a] == 0:
            assert sch.start[a] >= rel[inst.op_job[a]], f"op {a} before release"
        else:
            assert sch.start[a] >= sch.comp[inst.op_pred[a]], f"op {a} before predecessor"
    for i, order in enumerate(sch.machine_orders(inst)):
        for x in range(len(order) - 1):
            assert sch.comp[order[x]] <= sch.start[order[x + 1]], f"overlap on machine {i}"


def semi_active(inst: Instance, assign: List[int], orders: List[List[int]],
                durations: Optional[Dict[int, Dict[int, int]]] = None,
                release: Optional[List[int]] = None) -> Schedule:
    """Earliest-start schedule for fixed assignment and machine orders
    (longest paths in the disjunctive graph). Raises on a cycle."""
    rel = release if release is not None else inst.release
    n = inst.n_ops
    mpred = [-1] * n
    for order in orders:
        for x in range(1, len(order)):
            mpred[order[x]] = order[x - 1]
    start = [None] * n; comp = [None] * n
    state = [0] * n

    def dur(a):
        return durations[a][assign[a]] if durations is not None else inst.op_proc[a][assign[a]]

    import sys
    sys.setrecursionlimit(max(10000, 4 * n + 100))

    def visit(a):
        if state[a] == 2:
            return
        if state[a] == 1:
            raise ValueError("cycle in disjunctive graph")
        state[a] = 1
        s = rel[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        if inst.op_pred[a] >= 0:
            visit(inst.op_pred[a]); s = max(s, comp[inst.op_pred[a]])
        if mpred[a] >= 0:
            visit(mpred[a]); s = max(s, comp[mpred[a]])
        start[a] = s; comp[a] = s + dur(a); state[a] = 2

    for a in range(n):
        visit(a)
    return Schedule(list(assign), start, comp)
