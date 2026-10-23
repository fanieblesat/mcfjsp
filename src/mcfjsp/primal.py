"""Primal improvement by large neighbourhood search with CP-SAT subproblems.

A neighbourhood frees a set of operations: freed operations may change
machine and order; every other operation keeps its machine and its machine
order (kept as precedence constraints), while all start times may shift.
CP-SAT re-optimizes the plant makespan over the freed set; only improvements
are accepted. Neighbourhoods, chosen in rotation:
  cell       all operations of one cell (decomposition-consistent)
  cell-pair  all operations of two cells
  jobs       all operations of a random subset of jobs
Schedules it produces are feasible plant schedules whose cell restrictions
are added to the master as columns.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .instances import Instance
from .schedule import Schedule, check_feasible


def reoptimize(inst: Instance, sch: Schedule, free: set, time_limit: float = 5.0,
               threads: int = 1) -> Optional[Schedule]:
    from ortools.sat.python import cp_model
    horizon = sch.makespan
    orders = sch.machine_orders(inst)
    mdl = cp_model.CpModel()
    start, end, pres = {}, {}, {}
    ivs = {i: [] for i in range(inst.n_machines)}
    cmax = mdl.NewIntVar(0, horizon, "cmax")
    for a in range(inst.n_ops):
        lo = inst.release[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        s = mdl.NewIntVar(lo, horizon, f"s{a}"); e = mdl.NewIntVar(0, horizon, f"e{a}")
        start[a] = s; end[a] = e
        if a in free:
            lits = []
            for i, p in inst.op_proc[a].items():
                x = mdl.NewBoolVar(f"x{a}_{i}")
                ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}"))
                lits.append(x); pres[(a, i)] = x
                mdl.AddHint(x, 1 if i == sch.assign[a] else 0)     # warm start = incumbent
            mdl.AddExactlyOne(lits)
            mdl.AddHint(s, sch.start[a])
        else:
            mdl.AddHint(s, sch.start[a])
            i = sch.assign[a]
            mdl.Add(e == s + inst.op_proc[a][i])
            ivs[i].append(mdl.NewIntervalVar(s, inst.op_proc[a][i], e, f"fiv{a}"))
        if inst.op_succ[a] < 0:
            mdl.Add(cmax >= e)
        if inst.op_pred[a] >= 0:
            mdl.Add(s >= end[inst.op_pred[a]])
    for i in range(inst.n_machines):
        fixed = [a for a in orders[i] if a not in free]
        for k in range(len(fixed) - 1):
            mdl.Add(start[fixed[k + 1]] >= end[fixed[k]])
        if ivs[i]:
            mdl.AddNoOverlap(ivs[i])
    mdl.Add(cmax <= horizon - 1)  # only improvements
    mdl.Minimize(cmax)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    st = solver.Solve(mdl)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    assign = list(sch.assign); s_ = [0] * inst.n_ops
    for a in range(inst.n_ops):
        if a in free:
            assign[a] = next(i for i in inst.op_proc[a] if solver.Value(pres[(a, i)]))
        s_[a] = int(solver.Value(start[a]))
    new = Schedule(assign, s_, [s_[a] + inst.op_proc[a][assign[a]] for a in range(inst.n_ops)])
    check_feasible(inst, new)
    return new if new.makespan < sch.makespan else None


def critical_ops(inst: Instance, sch: Schedule) -> List[int]:
    """Operations on a longest path of the schedule's disjunctive graph (job
    precedence + machine order): H_a + p_a + Q_a == Cmax, where H (Q) is the
    longest path from the source (to the sink)."""
    orders = sch.machine_orders(inst)
    mpred = {}; msucc = {}
    for i in range(inst.n_machines):
        seq = orders[i]
        for k in range(len(seq) - 1):
            msucc[seq[k]] = seq[k + 1]; mpred[seq[k + 1]] = seq[k]
    # topological order = order of start times (ties by job position) in a feasible schedule
    topo = sorted(range(inst.n_ops), key=lambda a: (sch.start[a], inst.op_pos[a]))
    H = [0] * inst.n_ops; Q = [0] * inst.n_ops
    for a in topo:
        lo = inst.release[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        H[a] = max(lo, H[inst.op_pred[a]] + sch.comp[inst.op_pred[a]] - sch.start[inst.op_pred[a]] if inst.op_pred[a] >= 0 else 0,
                   H[mpred[a]] + sch.comp[mpred[a]] - sch.start[mpred[a]] if a in mpred else 0)
    for a in reversed(topo):
        p = sch.comp[a] - sch.start[a]
        Q[a] = p + max(Q[inst.op_succ[a]] if inst.op_succ[a] >= 0 else 0, Q[msucc[a]] if a in msucc else 0)
    cmax = max(H[a] + Q[a] for a in range(inst.n_ops))
    return [a for a in range(inst.n_ops) if H[a] + Q[a] == cmax]


def neighbourhoods(inst: Instance, sch_ref, rng, size: Optional[dict] = None):
    """Infinite generator of (name, freed-operation set). `sch_ref()` returns
    the current incumbent (neighbourhoods 'critical' and 'window' depend on it).
      cell       all operations of one cell (decomposition-consistent)
      critical   operations on a longest path of the incumbent, plus their
                 machine neighbours (coupling-dominated instances)
      window     all operations starting inside a random time window
      jobs       all operations of a random subset of jobs
      cell-pair  all operations of two cells"""
    import itertools
    cells = list(range(inst.n_cells))
    pairs = list(itertools.combinations(cells, 2))
    size = size if size is not None else {"job_frac": 0.3, "window_frac": 0.25}   # mutable: adapted by cell_lns

    def jobs_nb():
        k = max(1, int(size["job_frac"] * inst.n_jobs))
        js = rng.choice(inst.n_jobs, size=k, replace=False)
        return set(a for j in js for a in inst.job_ops[j])

    def critical_nb():
        sch = sch_ref()
        crit = set(critical_ops(inst, sch))
        orders = sch.machine_orders(inst)
        ext = set(crit)
        for i in range(inst.n_machines):
            seq = orders[i]
            for idx, a in enumerate(seq):
                if a in crit:
                    ext.update(seq[max(0, idx - 1): idx + 2])
        return ext

    def window_nb():
        sch = sch_ref()
        W = max(1, int(size["window_frac"] * sch.makespan))
        t = int(rng.integers(0, max(1, sch.makespan - W + 1)))
        return set(a for a in range(inst.n_ops) if t <= sch.start[a] < t + W)

    while True:
        for c in cells:
            yield f"cell{c}", set(inst.ops_of_cell[c])
            yield ("critical", critical_nb()) if c % 2 == 0 else ("window", window_nb())
        yield "jobs", jobs_nb()
        if pairs:
            c1, c2 = pairs[int(rng.integers(len(pairs)))]
            yield f"cells{c1}-{c2}", set(inst.ops_of_cell[c1]) | set(inst.ops_of_cell[c2])
        yield "critical", critical_nb()
        yield "window", window_nb()
        yield "jobs", jobs_nb()


def cell_lns(inst: Instance, sch: Schedule, time_budget: float = 30.0, cell_time: float = 5.0,
             threads: int = 1, log=None, seed: int = 0, max_fail: int = 50) -> Schedule:
    import numpy as np
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = sch
    fails = 0
    size = {"job_frac": 0.3, "window_frac": 0.25}
    base = dict(size)
    for name, free in neighbourhoods(inst, lambda: best, rng, size):
        if time.time() - t0 >= time_budget or fails >= max_fail:
            break
        tl = min(cell_time, max(0.5, time_budget - (time.time() - t0)))
        new = reoptimize(inst, best, free, tl, threads)
        if new is not None:
            if log:
                log.info("  LNS %s (%d ops): %d -> %d", name, len(free), best.makespan, new.makespan)
            best = new; fails = 0
            size.update(base)
        else:
            fails += 1
            # adaptive neighbourhood size: grow on failure (up to 70 % of the horizon / jobs)
            size["window_frac"] = min(0.7, size["window_frac"] * 1.15)
            size["job_frac"] = min(0.7, size["job_frac"] * 1.15)
    return best
