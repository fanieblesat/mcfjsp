"""Serial schedule generation (Lemma 'Repair') and dispatching rules.

`serial_sgs(inst, priority)` implements the repair procedure of the paper:
repeatedly take the eligible operation (job predecessor scheduled) with the
smallest priority value, assign it to the machine of its cell on which it
completes earliest, and start it as early as feasible.

Dispatching rules are priority functions evaluated dynamically by
`list_schedule`, which keeps the same mechanics but recomputes priorities
from the current state (so that rules like MWR or earliest-start work).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .instances import Instance
from .schedule import Schedule


def _durations(inst: Instance, durations):
    return durations if durations is not None else inst.op_proc


def serial_sgs(inst: Instance, priority: Sequence[float],
               durations: Optional[Dict[int, Dict[int, int]]] = None,
               release: Optional[List[int]] = None) -> Schedule:
    return list_schedule(inst, lambda a, s, mfree, rel: (priority[a], a), durations, release)


def list_schedule(inst: Instance, rule: Callable, durations=None, release=None) -> Schedule:
    """rule(a, earliest_start_of_a, machine_free_times, release) -> sort key (smaller first)."""
    D = _durations(inst, durations)
    rel = release if release is not None else inst.release
    n = inst.n_ops
    assign = [-1] * n; start = [0] * n; comp = [0] * n
    mfree = [0] * inst.n_machines
    done = [False] * n
    eligible = set(a for a in range(n) if inst.op_pos[a] == 0)
    for _ in range(n):
        best = None
        for a in eligible:
            es = rel[inst.op_job[a]] if inst.op_pos[a] == 0 else comp[inst.op_pred[a]]
            key = rule(a, es, mfree, rel)
            if best is None or key < best[0]:
                best = (key, a, es)
        _, a, es = best
        # machine on which it completes earliest (ties: lowest index)
        bi, bc = None, None
        for i, p in sorted(D[a].items()):
            c = max(es, mfree[i]) + p
            if bc is None or c < bc:
                bi, bc = i, c
        assign[a] = bi; start[a] = bc - D[a][bi]; comp[a] = bc; mfree[bi] = bc; done[a] = True
        eligible.discard(a)
        if inst.op_succ[a] >= 0:
            eligible.add(inst.op_succ[a])
    return Schedule(assign, start, comp)


# ---- rules -----------------------------------------------------------
def rules(inst: Instance, tails: Optional[List[int]] = None) -> Dict[str, Callable]:
    pmin = inst.p_min
    q = tails if tails is not None else [0] * inst.n_ops
    return {
        "EST": lambda a, es, mf, rel: (es, a),
        "SPT": lambda a, es, mf, rel: (pmin[a], a),
        "LPT": lambda a, es, mf, rel: (-pmin[a], a),
        "MWR": lambda a, es, mf, rel: (-(pmin[a] + q[a]), a),      # most work remaining
        "EST-MWR": lambda a, es, mf, rel: (es, -(pmin[a] + q[a]), a),
        "FIFO": lambda a, es, mf, rel: (inst.op_job[a], inst.op_pos[a]),
    }


def best_dispatch(inst: Instance, tails: Optional[List[int]] = None, durations=None, release=None):
    """Run all rules, return (best schedule, name, dict of makespans)."""
    res = {}
    best = None
    for name, rule in rules(inst, tails).items():
        sch = list_schedule(inst, rule, durations, release)
        res[name] = sch.makespan
        if best is None or sch.makespan < best[0].makespan:
            best = (sch, name)
    return best[0], best[1], res
