"""Monolithic baselines on the plant problem.

cpsat_plant : CP-SAT interval model (optional intervals per eligible machine,
              NoOverlap per machine, job precedences, release dates, makespan).
              Returns incumbent, proven lower bound, status.
gurobi_plant: MILP of Appendix EC.1 (disjunctive big-M with M = T), through
              gurobipy if installed (academic licence). Untested in the
              development sandbox; the formulation follows EC.1 line by line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .instances import Instance
from .schedule import Schedule, check_feasible


@dataclass
class BaselineResult:
    ub: Optional[int]
    lb: int
    status: str
    time: float
    schedule: Optional[Schedule] = None


def cpsat_plant(inst: Instance, horizon: int, time_limit: float = 60.0, threads: int = 8,
                hint: Optional[Schedule] = None, log: bool = False) -> BaselineResult:
    import time as _t
    from ortools.sat.python import cp_model
    t0 = _t.time()
    mdl = cp_model.CpModel()
    start, end, pres = {}, {}, {}
    ivs = {i: [] for i in range(inst.n_machines)}
    cmax = mdl.NewIntVar(0, horizon, "cmax")
    for a in range(inst.n_ops):
        lo = inst.release[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        s = mdl.NewIntVar(lo, horizon, f"s{a}"); e = mdl.NewIntVar(0, horizon, f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in inst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}"))
            lits.append(x); pres[(a, i)] = x
        mdl.AddExactlyOne(lits)
        if inst.op_succ[a] < 0:
            mdl.Add(cmax >= e)
        if inst.op_pred[a] >= 0:
            mdl.Add(s >= end[inst.op_pred[a]])
    for i, lst in ivs.items():
        if lst:
            mdl.AddNoOverlap(lst)
    mdl.Minimize(cmax)
    if hint is not None:
        for a in range(inst.n_ops):
            mdl.AddHint(start[a], hint.start[a])
            for i in inst.op_proc[a]:
                mdl.AddHint(pres[(a, i)], 1 if hint.assign[a] == i else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    solver.parameters.log_search_progress = log
    st = solver.Solve(mdl)
    name = solver.StatusName(st)
    sch = None; ub = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assign = [next(i for i in inst.op_proc[a] if solver.Value(pres[(a, i)])) for a in range(inst.n_ops)]
        s_ = [int(solver.Value(start[a])) for a in range(inst.n_ops)]
        sch = Schedule(assign, s_, [s_[a] + inst.op_proc[a][assign[a]] for a in range(inst.n_ops)])
        check_feasible(inst, sch)
        ub = sch.makespan
    lb = int(math.ceil(solver.BestObjectiveBound() - 1e-9)) if st != cp_model.UNKNOWN else 0
    return BaselineResult(ub, lb, name, _t.time() - t0, sch)


def gurobi_plant(inst: Instance, horizon: int, time_limit: float = 60.0, threads: int = 8,
                 hint: Optional[Schedule] = None) -> BaselineResult:
    import time as _t
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("gurobipy not available") from e
    t0 = _t.time()
    T = horizon
    m = gp.Model("mcfjsp_ec1")
    m.Params.TimeLimit = time_limit; m.Params.Threads = threads; m.Params.OutputFlag = 0
    x = {(a, i): m.addVar(vtype=GRB.BINARY, name=f"x{a}_{i}") for a in range(inst.n_ops) for i in inst.op_proc[a]}
    s = {a: m.addVar(lb=0, ub=T, name=f"s{a}") for a in range(inst.n_ops)}
    C = {a: m.addVar(lb=0, ub=T, name=f"C{a}") for a in range(inst.n_ops)}
    cmax = m.addVar(lb=0, ub=T, name="Cmax")
    m.setObjective(cmax, GRB.MINIMIZE)
    for a in range(inst.n_ops):
        m.addConstr(gp.quicksum(x[a, i] for i in inst.op_proc[a]) == 1)
        m.addConstr(C[a] == s[a] + gp.quicksum(p * x[a, i] for i, p in inst.op_proc[a].items()))
        if inst.op_pred[a] >= 0:
            m.addConstr(C[inst.op_pred[a]] <= s[a])
        else:
            m.addConstr(s[a] >= inst.release[inst.op_job[a]])
        if inst.op_succ[a] < 0:
            m.addConstr(cmax >= C[a])
    # disjunctions for pairs sharing a machine, a < b
    for c in range(inst.n_cells):
        ops = inst.ops_of_cell[c]
        for ia in range(len(ops)):
            for ib in range(ia + 1, len(ops)):
                a, b = ops[ia], ops[ib]
                common = set(inst.op_proc[a]) & set(inst.op_proc[b])
                if not common:
                    continue
                y = m.addVar(vtype=GRB.BINARY, name=f"y{a}_{b}")
                for i in common:
                    m.addConstr(s[b] >= C[a] - T * (3 - x[a, i] - x[b, i] - y))
                    m.addConstr(s[a] >= C[b] - T * (2 - x[a, i] - x[b, i] + y))
    if hint is not None:
        for a in range(inst.n_ops):
            s[a].Start = hint.start[a]
            for i in inst.op_proc[a]:
                x[a, i].Start = 1 if hint.assign[a] == i else 0
    m.optimize()
    sch = None; ub = None
    if m.SolCount > 0:
        assign = [next(i for i in inst.op_proc[a] if x[a, i].X > 0.5) for a in range(inst.n_ops)]
        st_ = [int(round(s[a].X)) for a in range(inst.n_ops)]
        sch = Schedule(assign, st_, [st_[a] + inst.op_proc[a][assign[a]] for a in range(inst.n_ops)])
        check_feasible(inst, sch)
        ub = sch.makespan
    lb = int(math.ceil(m.ObjBound - 1e-9)) if m.ObjBound > -GRB.INFINITY else 0
    return BaselineResult(ub, lb, str(m.Status), _t.time() - t0, sch)
