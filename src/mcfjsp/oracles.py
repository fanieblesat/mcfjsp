"""Cell bounding oracles (Section 5.1) and the pricing problem.

Cell problem for cell c, weights (w_a, v_a), windows [h_a, T - q_a]:
    min  sum_{a in O_c} (w_a C_a - v_a s_a)
    s.t. assignment to an eligible machine of the cell, no overlap,
         intra-cell job precedences, h_a <= s_a, C_a <= T - q_a.

Oracle (i): CP-SAT on the integer cell problem. Returns a *proven* lower
bound (valid even at a time limit) and the best schedule found, which is a
feasible column whatever the solver status.
Oracle (ii): LP relaxation of the time-indexed formulation (Appendix EC.2),
solved with HiGHS through OR-Tools' model_builder. Returns the LP value (a lower bound by Lemma EC.1) and the
fractional start/completion expressions used by the supergradient variant.

Weights are real (LP duals). CP-SAT needs integer coefficients, so weights
are scaled by WSCALE and rounded; the bound is corrected by the worst-case
rounding error  sum_a (|w_a - w'_a| + |v_a - v'_a|) * T  so that it remains
a valid lower bound for the *exact* weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instances import Instance

WSCALE = 10 ** 6


@dataclass
class Column:
    cell: int
    start: Dict[int, int]       # a -> s_a
    comp: Dict[int, int]        # a -> C_a
    assign: Dict[int, int]      # a -> machine

    def cost(self, w: Dict[int, float], v: Dict[int, float]) -> float:
        return sum(w[a] * self.comp[a] - v[a] * self.start[a] for a in self.start)

    def key(self) -> tuple:
        return tuple(sorted((a, self.assign[a], self.start[a]) for a in self.start))


@dataclass
class CellProblem:
    inst: Instance
    cell: int
    heads: List[int]
    tails: List[int]
    T: int

    @property
    def ops(self) -> List[int]:
        return self.inst.ops_of_cell[self.cell]

    @property
    def machines(self) -> List[int]:
        return self.inst.machines_of_cell[self.cell]

    def internal_pairs(self) -> List[Tuple[int, int]]:
        inst = self.inst
        return [(inst.op_pred[a], a) for a in self.ops
                if inst.op_pred[a] >= 0 and inst.op_cell[inst.op_pred[a]] == self.cell]

    def latest_start(self, a: int, i: int) -> int:
        return self.T - self.tails[a] - self.inst.op_proc[a][i]

    def column_from(self, start: Dict[int, int], assign: Dict[int, int]) -> Column:
        comp = {a: start[a] + self.inst.op_proc[a][assign[a]] for a in start}
        return Column(self.cell, dict(start), comp, dict(assign))

    def check_column(self, col: Column) -> None:
        inst = self.inst
        for a in self.ops:
            i = col.assign[a]
            assert i in inst.op_proc[a]
            assert col.start[a] >= self.heads[a], (a, col.start[a], self.heads[a])
            assert col.comp[a] == col.start[a] + inst.op_proc[a][i]
            assert col.comp[a] + self.tails[a] <= self.T, (a, col.comp[a], self.tails[a], self.T)
        for (b, a) in self.internal_pairs():
            assert col.start[a] >= col.comp[b]
        by_m: Dict[int, List[int]] = {}
        for a in self.ops:
            by_m.setdefault(col.assign[a], []).append(a)
        for i, ops in by_m.items():
            ops.sort(key=lambda a: col.start[a])
            for x in range(len(ops) - 1):
                assert col.comp[ops[x]] <= col.start[ops[x + 1]], f"overlap on {i}"


# ---------------------------------------------------------------------
# Oracle (i): CP-SAT
# ---------------------------------------------------------------------
@dataclass
class OracleResult:
    bound: float                 # valid lower bound on the cell problem value
    value: Optional[float]       # exact cost of `column` under the true weights (None if no column)
    column: Optional[Column]
    status: str
    proven: bool                 # bound == optimum proven


def cpsat_cell(cp: CellProblem, w: Dict[int, float], v: Dict[int, float], time_limit: float = 10.0,
               threads: int = 1, hint: Optional[Column] = None,
               lower_bound: Optional[float] = None) -> OracleResult:
    """`lower_bound`: a known valid lower bound on the cell value (e.g. the
    TI-LP value); added as a cut, which lets CP-SAT prove optimality as soon
    as it finds a matching solution. The returned bound is never below it."""
    from ortools.sat.python import cp_model
    inst = cp.inst
    ops = cp.ops
    if not ops:
        return OracleResult(0.0, 0.0, Column(cp.cell, {}, {}, {}), "EMPTY", True)
    wi = {a: int(round(w[a] * WSCALE)) for a in ops}
    vi = {a: int(round(v[a] * WSCALE)) for a in ops}
    err = sum(abs(w[a] - wi[a] / WSCALE) + abs(v[a] - vi[a] / WSCALE) for a in ops) * cp.T
    mdl = cp_model.CpModel()
    start, end, pres = {}, {}, {}
    ivs = {i: [] for i in cp.machines}
    obj_terms = []
    for a in ops:
        s = mdl.NewIntVar(cp.heads[a], cp.T, f"s{a}")
        e = mdl.NewIntVar(0, cp.T - cp.tails[a], f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in inst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}"))
            lits.append(x); pres[(a, i)] = x
        mdl.AddExactlyOne(lits)
        if wi[a]:
            obj_terms.append(wi[a] * e)
        if vi[a]:
            obj_terms.append(-vi[a] * s)
    for (b, a) in cp.internal_pairs():
        mdl.Add(start[a] >= end[b])
    for i, lst in ivs.items():
        if lst:
            mdl.AddNoOverlap(lst)
    mdl.Minimize(sum(obj_terms) if obj_terms else 0)
    if lower_bound is not None and obj_terms:
        # rounded objective >= true objective - err  >= lower_bound - err
        mdl.Add(sum(obj_terms) >= int(math.floor((lower_bound - err) * WSCALE)) - 1)
    if hint is not None:
        for a in ops:
            if a in hint.start:
                mdl.AddHint(start[a], hint.start[a])
                for i in inst.op_proc[a]:
                    mdl.AddHint(pres[(a, i)], 1 if hint.assign[a] == i else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    st = solver.Solve(mdl)
    name = solver.StatusName(st)
    col = None; val = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        s_ = {a: int(solver.Value(start[a])) for a in ops}
        as_ = {a: next(i for i in inst.op_proc[a] if solver.Value(pres[(a, i)])) for a in ops}
        col = cp.column_from(s_, as_)
        val = col.cost(w, v)
        bound = solver.BestObjectiveBound() / WSCALE - err
        proven = st == cp_model.OPTIMAL
    elif st == cp_model.INFEASIBLE:
        raise RuntimeError(f"cell {cp.cell} infeasible within horizon {cp.T}")
    else:  # UNKNOWN: no solution; bound still valid
        bound = solver.BestObjectiveBound() / WSCALE - err
        proven = False
    if not obj_terms:
        bound = 0.0
    if lower_bound is not None:
        bound = max(bound, lower_bound)
        if val is not None and val <= lower_bound + 1e-9:
            proven = True
    return OracleResult(bound, val, col, name, proven)


# ---------------------------------------------------------------------
# Oracle (ii): time-indexed LP (HiGHS)
# ---------------------------------------------------------------------
class TimeIndexedLP:
    """Builds the LP once per (cell, windows, T); re-solves for new weights.

    Variables: z[a,i,t] for t in [h_a, T - q_a - p_ia];
               u[a,i,t] = sum_{tau<=t} z[a,i,tau]   (cumulative, per machine)
    Constraints: assignment, machine capacity, precedence (disaggregated via u).
    """

    def __init__(self, cp: CellProblem):
        self.cp = cp
        inst = cp.inst
        self.ops = cp.ops
        self.T = cp.T
        self.z = {}   # (a,i,t) -> var index
        self.u = {}   # (a,i,t) -> var index
        ncol = 0
        for a in self.ops:
            for i, p in inst.op_proc[a].items():
                lo, hi = cp.heads[a], cp.latest_start(a, i)
                for t in range(lo, hi + 1):
                    self.z[(a, i, t)] = ncol; ncol += 1
        self.nz = ncol
        pairs = cp.internal_pairs()
        need_u = set()
        for (b, a) in pairs:
            need_u.add(b)
        self.need_u = need_u
        for a in self.ops:
            if a in need_u:
                for i, p in inst.op_proc[a].items():
                    lo, hi = cp.heads[a], cp.latest_start(a, i)
                    for t in range(lo, self.T + 1):
                        self.u[(a, i, t)] = ncol; ncol += 1
        self.ncol = ncol
        rows_lo, rows_hi = [], []
        ri, ci, vals = [], [], []
        r = 0
        # assignment
        for a in self.ops:
            for (aa, i, t), c in self.z.items():
                if aa == a:
                    ri.append(r); ci.append(c); vals.append(1.0)
            rows_lo.append(1.0); rows_hi.append(1.0); r += 1
        # capacity: for machine i and time t: sum_a sum_{tau=t-p+1..t} z <= 1
        zz_by_m: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for (a, i, t), c in self.z.items():
            zz_by_m.setdefault(i, []).append((a, t, inst.op_proc[a][i], c))
        for i, lst in zz_by_m.items():
            cover: Dict[int, List[int]] = {}
            for (a, t, p, c) in lst:
                for tau in range(t, t + p):
                    cover.setdefault(tau, []).append(c)
            for tau, cs in sorted(cover.items()):
                if len(cs) <= 1:
                    continue
                for c in cs:
                    ri.append(r); ci.append(c); vals.append(1.0)
                rows_lo.append(-math.inf); rows_hi.append(1.0); r += 1
        # cumulative definitions: u[a,i,t] - u[a,i,t-1] - z[a,i,t] = 0
        for (a, i, t), c in self.u.items():
            ri.append(r); ci.append(c); vals.append(1.0)
            if (a, i, t - 1) in self.u:
                ri.append(r); ci.append(self.u[(a, i, t - 1)]); vals.append(-1.0)
            if (a, i, t) in self.z:
                ri.append(r); ci.append(self.z[(a, i, t)]); vals.append(-1.0)
            rows_lo.append(0.0); rows_hi.append(0.0); r += 1
        # precedence (b before a): sum_i sum_{tau<=t} z[a,i,tau] <= sum_i u[b,i,t-p_ib]
        for (b, a) in pairs:
            for t in range(cp.heads[a], self.T + 1):
                nzr = 0
                for i in inst.op_proc[a]:
                    for tau in range(cp.heads[a], min(t, cp.latest_start(a, i)) + 1):
                        ri.append(r); ci.append(self.z[(a, i, tau)]); vals.append(1.0); nzr += 1
                if nzr == 0:
                    continue
                for i, p in inst.op_proc[b].items():
                    tt = t - p
                    if tt >= cp.heads[b]:
                        key = (b, i, min(tt, self.T))
                        ri.append(r); ci.append(self.u[key]); vals.append(-1.0)
                rows_lo.append(-math.inf); rows_hi.append(0.0); r += 1
        self.nrow = r
        from .lp import LP
        lp = LP("glop")
        for c in range(ncol):
            lp.add_var(0.0, 1.0 if c < self.nz else math.inf, 0.0)
        # group terms by row
        by_row: Dict[int, List[Tuple[int, float]]] = {}
        for rr, cc, vv in zip(ri, ci, vals):
            by_row.setdefault(rr, []).append((cc, vv))
        for rr in range(r):
            lp.add_row(by_row.get(rr, []), rows_lo[rr], rows_hi[rr])
        self.lp = lp

    def solve(self, w: Dict[int, float], v: Dict[int, float]) -> Tuple[float, Dict[int, float], Dict[int, float], str]:
        inst = self.cp.inst
        cost = {}
        for (a, i, t), c in self.z.items():
            cost[c] = w[a] * (t + inst.op_proc[a][i]) - v[a] * t
        scale = max(1e-12, max(abs(x) for x in cost.values())) if cost else 1.0
        for c, x in cost.items():
            self.lp.set_obj(c, x / scale)
        status = self.lp.solve(fallback="highs")
        if status != "OPTIMAL":
            raise RuntimeError(f"TI-LP status {status}")
        x = self.lp.values()
        s = {a: 0.0 for a in self.ops}; C = {a: 0.0 for a in self.ops}
        for (a, i, t), c in self.z.items():
            s[a] += t * x[c]; C[a] += (t + inst.op_proc[a][i]) * x[c]
        return float(self.lp.objective) * scale, s, C, status


def ti_lp_cell(cp: CellProblem, w, v, cache: Optional[dict] = None) -> OracleResult:
    key = (cp.cell, cp.T)
    if cache is not None and key in cache:
        lp = cache[key]
    else:
        lp = TimeIndexedLP(cp)
        if cache is not None:
            cache[key] = lp
    if not cp.ops:
        return OracleResult(0.0, 0.0, Column(cp.cell, {}, {}, {}), "EMPTY", True)
    val, s, C, st = lp.solve(w, v)
    res = OracleResult(val - 1e-7 * (1 + abs(val)), None, None, "LP_" + st, False)
    res.frac_start = s; res.frac_comp = C
    return res


# ---------------------------------------------------------------------
# Compact time-indexed LP of the plant problem (z_TI, Proposition 'Hierarchy')
# ---------------------------------------------------------------------
def compact_ti_lp(inst: Instance, heads: List[int], tails: List[int], T: int,
                  gamma_ht: Optional[int] = None, time_limit: Optional[float] = None,
                  backend: str = "highs", max_vars: int = 2_000_000) -> Tuple[float, str]:
    """LP relaxation of: all cells' time-indexed blocks (EC.2) + epigraph
    Cmax >= C_a + q_a + inter-cell precedences s_{a+} >= C_a (+ optional
    head-tail cut). Its value z_TI <= z_LD (Proposition 'Hierarchy')."""
    from .lp import LP
    est = sum(len(inst.op_proc[a]) * max(0, T - tails[a] - heads[a]) for a in range(inst.n_ops))
    if est > max_vars:
        return -math.inf, f"too_large({est} vars)"
    lp = LP(backend)
    cmax = lp.add_var(0.0, float(T), 1.0, "Cmax")
    sub = {}
    for c in range(inst.n_cells):
        cp = CellProblem(inst, c, heads, tails, T)
        if not cp.ops:
            continue
        sub[c] = _ti_block(lp, cp)
    # linear expressions of s_a, C_a per operation
    def s_expr(a):
        return [(idx, float(t)) for (aa, i, t), idx in sub[inst.op_cell[a]].items() if aa == a]
    def C_expr(a):
        return [(idx, float(t + inst.op_proc[a][i])) for (aa, i, t), idx in sub[inst.op_cell[a]].items() if aa == a]
    for a in range(inst.n_ops):
        # Cmax - C_a >= q_a
        lp.add_row([(cmax, 1.0)] + [(idx, -coef) for idx, coef in C_expr(a)], float(tails[a]), math.inf)
    for a in inst.transitions:
        b = inst.op_succ[a]
        lp.add_row(s_expr(b) + [(idx, -coef) for idx, coef in C_expr(a)], 0.0, math.inf)
    if gamma_ht is not None:
        lp.add_row([(cmax, 1.0)], float(gamma_ht), math.inf)
    st = lp.solve(time_limit=time_limit)
    return (float(lp.objective) if st == "OPTIMAL" else -math.inf), st


def _ti_block(lp, cp: CellProblem) -> Dict[Tuple[int, int, int], int]:
    """Add the time-indexed block of one cell to `lp` (LP-relaxed, z in [0,1]).
    Returns the map (a, i, t) -> variable index."""
    inst = cp.inst
    z: Dict[Tuple[int, int, int], int] = {}
    for a in cp.ops:
        for i, p in inst.op_proc[a].items():
            for t in range(cp.heads[a], cp.latest_start(a, i) + 1):
                z[(a, i, t)] = lp.add_var(0.0, 1.0, 0.0)
    for a in cp.ops:
        lp.add_row([(idx, 1.0) for (aa, i, t), idx in z.items() if aa == a], 1.0, 1.0)
    by_m: Dict[int, Dict[int, List[int]]] = {}
    for (a, i, t), idx in z.items():
        for tau in range(t, t + inst.op_proc[a][i]):
            by_m.setdefault(i, {}).setdefault(tau, []).append(idx)
    for i, cov in by_m.items():
        for tau, idxs in cov.items():
            if len(idxs) > 1:
                lp.add_row([(idx, 1.0) for idx in idxs], -math.inf, 1.0)
    # cumulative u[b,i,t] for predecessors in intra-cell pairs
    pairs = cp.internal_pairs()
    u: Dict[Tuple[int, int, int], int] = {}
    for (b, a) in pairs:
        for i, p in inst.op_proc[b].items():
            prev = None
            for t in range(cp.heads[b], cp.T + 1):
                if (b, i, t) in u:
                    prev = u[(b, i, t)]; continue
                idx = lp.add_var(0.0, math.inf, 0.0)
                terms = [(idx, 1.0)]
                if prev is not None:
                    terms.append((prev, -1.0))
                if (b, i, t) in z:
                    terms.append((z[(b, i, t)], -1.0))
                lp.add_row(terms, 0.0, 0.0)
                u[(b, i, t)] = idx; prev = idx
    for (b, a) in pairs:
        for t in range(cp.heads[a], cp.T + 1):
            terms = [(z[(a, i, tau)], 1.0) for i in inst.op_proc[a]
                     for tau in range(cp.heads[a], min(t, cp.latest_start(a, i)) + 1)]
            if not terms:
                continue
            for i, p in inst.op_proc[b].items():
                tt = t - p
                if tt >= cp.heads[b]:
                    terms.append((u[(b, i, min(tt, cp.T))], -1.0))
            lp.add_row(terms, -math.inf, 0.0)
    return z


def compact_ti_lp_coarse(inst: Instance, T: int, kappas=(1.0, 0.5, 0.25, 0.125), time_limit: float = 60.0,
                         backend: str = "highs", max_vars: int = 300_000):
    """z_TI on a coarsened time grid: durations and release dates are scaled by
    kappa and rounded down (Lemma EC.3), the LP value is divided by kappa. The
    first kappa whose LP solves within the limit is used. Returns
    (bound, kappa, status)."""
    import math as _m
    from .bounds import heads_tails
    for kappa in kappas:
        if kappa == 1.0:
            sinst = inst
        else:
            dur = {a: {i: max(1, int(_m.floor(kappa * p))) for i, p in inst.op_proc[a].items()} for a in range(inst.n_ops)}
            rel = [int(_m.floor(kappa * r)) for r in inst.release]
            sinst = inst.with_durations(dur, rel, name_suffix=f"_k{kappa}")
        h, q = heads_tails(sinst)
        Tk = int(_m.ceil(kappa * T)) + 1
        z, st = compact_ti_lp(sinst, h, q, Tk, time_limit=time_limit, backend=backend, max_vars=max_vars)
        if st == "OPTIMAL":
            return z / kappa, kappa, st
    return -_m.inf, None, "unsolved"
