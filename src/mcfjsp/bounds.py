"""Heads, tails and head--tail lower bounds per cell (Lemma 'Windows',
Proposition 'Head--tail bound').

Three lower bounds on Gamma_c = min max_a (C_a + q_a) over cell c:
  (a) energetic / threshold bound          O(|O_c|^2) closed form
  (b) preemptive relaxation (Lawler--Labetoulle LP) with binary search
  (c) CP-SAT proven bound within a time limit
All are valid lower bounds on the plant makespan C*_max.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instances import Instance


def heads_tails(inst: Instance) -> Tuple[List[int], List[int]]:
    h = [0] * inst.n_ops
    q = [0] * inst.n_ops
    for j, ops in enumerate(inst.job_ops):
        h[ops[0]] = inst.release[j]
        for k in range(1, len(ops)):
            h[ops[k]] = h[ops[k - 1]] + inst.p_min[ops[k - 1]]
        for k in range(len(ops) - 2, -1, -1):
            q[ops[k]] = q[ops[k + 1]] + inst.p_min[ops[k + 1]]
    return h, q


# ---- (a) energetic bound ---------------------------------------------
def energetic_bound_cell(inst: Instance, c: int, h: List[int], q: List[int]) -> int:
    ops = inst.ops_of_cell[c]
    if not ops:
        return 0
    m = len(inst.machines_of_cell[c])
    best = max(h[a] + inst.p_min[a] + q[a] for a in ops)
    hs = sorted(set(h[a] for a in ops))
    qs = sorted(set(q[a] for a in ops))
    for ht in hs:
        for qt in qs:
            A = [a for a in ops if h[a] >= ht and q[a] >= qt]
            if not A:
                continue
            work = sum(inst.p_min[a] for a in A)
            val = min(h[a] for a in A) + math.ceil(work / m) + min(q[a] for a in A)
            if val > best:
                best = val
    return best


# ---- (b) preemptive relaxation ---------------------------------------
def preemptive_feasible(inst: Instance, c: int, h: List[int], q: List[int], theta: int) -> bool:
    """Lawler--Labetoulle LP: can all operations of cell c be preemptively
    processed on the unrelated machines of the cell within windows
    [h_a, theta - q_a]? Intra-cell precedences are relaxed (still valid)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    ops = inst.ops_of_cell[c]
    ms = inst.machines_of_cell[c]
    for a in ops:
        if theta - q[a] - h[a] < inst.p_min[a]:
            return False
    pts = sorted(set([h[a] for a in ops] + [theta - q[a] for a in ops]))
    ivs = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    # variables x[a,i,k] = fraction of a's work done on i in interval k.
    # Constraint rows are collected in one pass over the variables (grouped by
    # operation, by (machine, interval) and by (operation, interval)) instead of
    # re-scanning all variables for every row, which was quadratic in |var|.
    rows_eq, cols_eq = [], []
    cap = {}     # (i, k) -> (list of var idx, list of coeffs)
    par = {}     # (a, k) -> (list of var idx, list of coeffs)
    nv = 0
    for a in ops:
        lo_a, hi_a = h[a], theta - q[a]
        for i, p_ai in inst.op_proc[a].items():
            for k, (lo, hi) in enumerate(ivs):
                if lo >= lo_a and hi <= hi_a:
                    rows_eq.append(a); cols_eq.append(nv)
                    cap.setdefault((i, k), ([], []))[0].append(nv); cap[(i, k)][1].append(float(p_ai))
                    par.setdefault((a, k), ([], []))[0].append(nv); par[(a, k)][1].append(float(p_ai))
                    nv += 1
    if nv == 0:
        return False
    op_row = {a: r for r, a in enumerate(ops)}
    A_eq = coo_matrix(([1.0] * nv, ([op_row[a] for a in rows_eq], cols_eq)), shape=(len(ops), nv))
    b_eq = [1.0] * len(ops)
    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    for (i, k), (idx, coef) in cap.items():        # machine capacity per interval
        rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coef)
        b_ub.append(float(ivs[k][1] - ivs[k][0])); r += 1
    for (a, k), (idx, coef) in par.items():        # no parallel processing of one operation
        rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coef)
        b_ub.append(float(ivs[k][1] - ivs[k][0])); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)) if r else None
    res = linprog(np.zeros(nv), A_ub=A_ub, b_ub=np.array(b_ub) if r else None, A_eq=A_eq,
                  b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    return res.status == 0


def preemptive_bound_cell(inst: Instance, c: int, h: List[int], q: List[int],
                          lo: int, hi: int) -> int:
    """Largest theta in [lo, hi] such that theta-1 is infeasible: binary search.
    Returns a valid lower bound >= lo (assumes hi is feasible)."""
    if not inst.ops_of_cell[c]:
        return lo
    if preemptive_feasible(inst, c, h, q, lo):
        return lo
    a, b = lo, hi  # infeasible at a, feasible at b
    while b - a > 1:
        mid = (a + b) // 2
        if preemptive_feasible(inst, c, h, q, mid):
            b = mid
        else:
            a = mid
    return b


# ---- (c) CP-SAT cell bound -------------------------------------------
def cpsat_bound_cell(inst: Instance, c: int, h: List[int], q: List[int], ub: int,
                     time_limit: float = 10.0, threads: int = 1) -> Tuple[int, int]:
    """Returns (proven lower bound, best objective found) for
    min max_a (C_a + q_a) over cell c with heads, intra-cell precedences."""
    from ortools.sat.python import cp_model
    ops = inst.ops_of_cell[c]
    if not ops:
        return 0, 0
    mdl = cp_model.CpModel()
    start = {}; end = {}; pres = {}
    ivs_by_m = {i: [] for i in inst.machines_of_cell[c]}
    obj = mdl.NewIntVar(0, ub, "obj")
    for a in ops:
        s = mdl.NewIntVar(h[a], ub, f"s{a}"); e = mdl.NewIntVar(0, ub, f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in inst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            iv = mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}")
            ivs_by_m[i].append(iv); lits.append(x); pres[(a, i)] = x
        mdl.AddExactlyOne(lits)
        mdl.Add(e + q[a] <= obj)
    for a in ops:
        p_ = inst.op_pred[a]
        if p_ >= 0 and inst.op_cell[p_] == c:
            mdl.Add(start[a] >= end[p_])
    for i, ivs in ivs_by_m.items():
        mdl.AddNoOverlap(ivs)
    mdl.Minimize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    st = solver.Solve(mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return int(math.ceil(solver.BestObjectiveBound() - 1e-9)), int(solver.ObjectiveValue())
    if st == cp_model.INFEASIBLE:
        return ub + 1, ub + 1
    return int(math.ceil(solver.BestObjectiveBound() - 1e-9)), ub


def head_tail_bounds(inst: Instance, ub: int, cp_time: float = 5.0, threads: int = 1,
                     use_preemptive: bool = True, use_cpsat: bool = True,
                     time_budget: Optional[float] = None) -> Dict:
    """`time_budget`: total wall-clock budget for the phase; the CP-SAT time
    per cell is reduced to fit, and the preemptive LP is skipped for the
    remaining cells once the budget is exhausted (bounds stay valid)."""
    import time as _t
    t0 = _t.time()
    h, q = heads_tails(inst)
    # gamma_ht is the MAX over cells, so a cell whose preemptive relaxation is
    # feasible at the current running maximum cannot raise it: one LP decides
    # that, instead of a full binary search.  Cells are visited in decreasing
    # order of their energetic bound so that the bottleneck cell comes first.
    # The returned gamma_ht is identical to the exhaustive computation; only
    # the per-cell "preemptive" entries of dominated cells are reported as
    # their energetic value (still a valid lower bound for that cell).
    energetic = [energetic_bound_cell(inst, c, h, q) for c in range(inst.n_cells)]
    order = sorted(range(inst.n_cells), key=lambda c: -energetic[c])
    per_cell = [None] * inst.n_cells
    gamma = max(energetic) if energetic else 0
    for k, c in enumerate(order):
        e = energetic[c]
        left = (time_budget - (_t.time() - t0)) if time_budget else None
        cells_left = inst.n_cells - k
        pr = e
        if use_preemptive and inst.ops_of_cell[c] and (left is None or left > 0.5):
            lo = max(e, gamma)
            if lo >= ub:
                pr = e
            elif preemptive_feasible(inst, c, h, q, lo):
                pr = e                       # cannot exceed the running maximum
            else:
                pr = preemptive_bound_cell(inst, c, h, q, lo, ub)
        left = (time_budget - (_t.time() - t0)) if time_budget else None
        cpt = cp_time if left is None else min(cp_time, max(0.0, left / cells_left))
        cp = cpsat_bound_cell(inst, c, h, q, ub, cpt, threads)[0] if (use_cpsat and cpt >= 0.2) else e
        best = max(e, pr, cp); gamma = max(gamma, best)
        per_cell[c] = {"energetic": e, "preemptive": pr, "cpsat": cp, "best": best}
    return {"heads": h, "tails": q, "cells": per_cell,
            "gamma_ht": max(pc["best"] for pc in per_cell) if per_cell else 0,
            "gamma_energetic": max(pc["energetic"] for pc in per_cell) if per_cell else 0}
