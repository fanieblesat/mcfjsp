"""Window propagation, shaving and the destructive (propagation-strengthened)
bound.

Windows are on START times: for a target makespan T every operation a has
[es_a, ls_a] (earliest / latest start).  Initially es = longest-path heads and
ls_a = T - q_a - p_min_a with q the longest-path tails (Lemma 'Windows').  The
windows are tightened to a fixed point with

  (P)  job precedence:   es_{a+} >= es_a + p_min_a,   ls_{a-} <= ls_{a+} - p_min_{a-}
  (E)  energetic (cumulative) reasoning per cell with m machines: the first
       p_min_a units of every operation lie in [es_a, ls_a + p_min_a], so for
       thresholds (H, D): sum{p_min_a : es_a >= H, ls_a + p_min_a <= D} <= m (D - H)
  (EF) edge-finding on each machine for the operations whose only eligible
       machine it is (Carlier & Pinson 1989; Baptiste, Le Pape & Nuijten
       2001, O(n^2) form), on heads and, mirrored, on latest starts.

Shaving (Martin & Shmoys 1996; Carlier & Pinson 1994): the hypothesis
's_a <= t' (resp. 's_a >= t') is propagated; if it is refuted, es_a >= t + 1
(resp. ls_a <= t - 1).  Binary search over t.

After the fixed point, each cell's exact feasibility (CP-SAT with the windows
and the true tails e_a + q_a <= T) is checked; UNKNOWN counts as feasible.
If T is proven infeasible, T + 1 is a valid lower bound (destructive bound).

Validity: every rule only removes start times that no feasible schedule with
makespan <= T can use, for any machine choice (durations enter only through
p_min, except on machines where the operation has no alternative).  (P) is
the only cross-cell step, so the scheme is a message-passing iteration
between cell agents that propagate locally and export their windows.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from .instances import Instance


class Infeasible(Exception):
    pass


def _edge_finding(ops: List[int], es: Dict[int, int], lct: Dict[int, int], p: Dict[int, int]) -> Dict[int, int]:
    """O(n^2) edge-finding on one machine: ops fixed to it, es = earliest
    starts, lct = latest completions, p = durations.  Returns increased
    earliest starts.  Raises Infeasible on a proven overload."""
    n = len(ops)
    if n <= 1:
        return {}
    by_d = sorted(ops, key=lambda a: lct[a])
    by_h = sorted(ops, key=lambda a: es[a])
    new = {}
    for k_idx, k in enumerate(by_d):
        D = lct[k]
        omega = set(by_d[: k_idx + 1])
        suffix = 0; tail_sum = {}
        for a in reversed(by_h):
            if a in omega:
                suffix += p[a]; tail_sum[a] = suffix
        C = -math.inf; est = math.inf; P = 0
        for a in by_h:
            if a in omega:
                C = max(C, es[a] + tail_sum[a]); est = min(est, es[a]); P += p[a]
        if C > D:
            raise Infeasible(f"machine overload est+p={C} > lct={D}")
        for i in by_d[k_idx + 1:]:
            if min(es[i], est) + P + p[i] > D and C > es[i]:
                new[i] = max(new.get(i, es[i]), C)
    return new


class Windows:
    """Start-time windows for target T with worklist propagation to a fixed
    point (compiled kernels for edge-finding and energetic reasoning)."""

    def __init__(self, inst: Instance, T: int, h: List[int], q: List[int]):
        import numpy as np
        self.inst = inst; self.T = T; self.q0 = list(q)
        self.pmin = np.array(inst.p_min, dtype=np.int64)
        self.es = np.array(h, dtype=np.int64)
        self.ls = np.array([T - q[a] - inst.p_min[a] for a in range(inst.n_ops)], dtype=np.int64)
        self.cell_ops = [np.array(inst.ops_of_cell[c], dtype=np.int64) for c in range(inst.n_cells)]
        self.cell_m = [len(inst.machines_of_cell[c]) for c in range(inst.n_cells)]
        self.fixed_on = {i: [] for i in range(inst.n_machines)}
        for a in range(inst.n_ops):
            if len(inst.op_proc[a]) == 1:
                self.fixed_on[next(iter(inst.op_proc[a]))].append(a)
        self.mach_ops = {i: np.array(v, dtype=np.int64) for i, v in self.fixed_on.items() if len(v) > 1}
        self.mach_p = {i: np.array([inst.op_proc[a][i] for a in v], dtype=np.int64) for i, v in self.mach_ops.items()}
        self.cell_machines = [[i for i in inst.machines_of_cell[c] if i in self.mach_ops] for c in range(inst.n_cells)]
        self.op_cell = np.array(inst.op_cell, dtype=np.int64)
        self.succ = np.array(inst.op_succ, dtype=np.int64); self.pred = np.array(inst.op_pred, dtype=np.int64)

    def copy(self) -> "Windows":
        w = Windows.__new__(Windows)
        w.__dict__.update(self.__dict__)
        w.es = self.es.copy(); w.ls = self.ls.copy()
        return w

    # ---- precedence push along the job; returns cells touched
    def _push(self, a: int, touched: set) -> None:
        es, ls, pmin = self.es, self.ls, self.pmin
        b = a
        while self.succ[b] >= 0:
            n_ = self.succ[b]; v = es[b] + pmin[b]
            if v > es[n_]:
                es[n_] = v; touched.add(int(self.op_cell[n_])); b = n_
            else:
                break
        b = a
        while self.pred[b] >= 0:
            p_ = self.pred[b]; v = ls[b] - pmin[p_]
            if v < ls[p_]:
                ls[p_] = v; touched.add(int(self.op_cell[p_])); b = p_
            else:
                break

    def _cell_rules(self, c: int, touched: set) -> None:
        from .kernels import edge_finding, energetic_ok
        es, ls, pmin = self.es, self.ls, self.pmin
        ops = self.cell_ops[c]
        if ops.shape[0] == 0:
            return
        if (es[ops] > ls[ops]).any():
            raise Infeasible(f"cell {c}: empty window")
        if not energetic_ok(es[ops], ls[ops] + pmin[ops], pmin[ops], self.cell_m[c]):
            raise Infeasible(f"cell {c}: energy")
        for i in self.cell_machines[c]:
            mo = self.mach_ops[i]; p = self.mach_p[i]
            ok, new = edge_finding(es[mo], ls[mo] + p, p)
            if not ok:
                raise Infeasible(f"machine {i}: overload")
            for k in range(mo.shape[0]):
                if new[k] > es[mo[k]]:
                    es[mo[k]] = new[k]; touched.add(c); self._push(int(mo[k]), touched)
            ok, new = edge_finding(self.T - ls[mo] - p, self.T - es[mo], p)       # mirror
            if not ok:
                raise Infeasible(f"machine {i}: overload (mirror)")
            for k in range(mo.shape[0]):
                nl = self.T - new[k] - p[k]
                if nl < ls[mo[k]]:
                    ls[mo[k]] = nl; touched.add(c); self._push(int(mo[k]), touched)

    def propagate(self, seeds: Optional[List[int]] = None, max_rounds: int = 10_000) -> None:
        """Worklist propagation. `seeds`: operations whose windows changed
        (None = everything)."""
        touched = set()
        if seeds is None:
            for a in range(self.inst.n_ops):
                self._push(a, touched)
            queue = list(range(self.inst.n_cells))
        else:
            for a in seeds:
                touched.add(int(self.op_cell[a])); self._push(int(a), touched)
            queue = list(touched)
        inq = set(queue); touched = set()
        rounds = 0
        while queue:
            c = queue.pop(); inq.discard(c)
            touched.clear()
            self._cell_rules(c, touched)
            for c2 in touched:
                if c2 not in inq:
                    queue.append(c2); inq.add(c2)
            rounds += 1
            if rounds > max_rounds * max(1, self.inst.n_cells):
                break
        if (self.es > self.ls).any():
            raise Infeasible("empty window")

    def refutes(self, a: int, side: str, t: int) -> bool:
        """True if the hypothesis (s_a <= t if side == 'head' else s_a >= t) is refuted."""
        w = self.copy()
        if side == "head":
            if t < w.ls[a]:
                w.ls[a] = t
        else:
            if t > w.es[a]:
                w.es[a] = t
        try:
            w.propagate(seeds=[a])
            return False
        except Infeasible:
            return True

    def shave(self, sweeps: int = 2, time_budget: Optional[float] = None, ops: Optional[List[int]] = None) -> None:
        """Shaving on heads and latest starts (binary search), to a fixed point
        or `sweeps` passes.  Raises Infeasible if some window closes."""
        t0 = time.time()
        self.propagate()
        rng = list(range(self.inst.n_ops)) if ops is None else list(ops)
        for _ in range(sweeps):
            changed = False
            for side in ("head", "tail"):
                for a in rng:
                    if time_budget and time.time() - t0 > time_budget:
                        return
                    lo, hi = int(self.es[a]), int(self.ls[a])
                    if side == "head":
                        if not self.refutes(a, "head", lo):
                            continue
                        best = lo; L, R = lo + 1, hi
                        while L <= R:
                            t = (L + R) // 2
                            if self.refutes(a, "head", t):
                                best = t; L = t + 1
                            else:
                                R = t - 1
                        if best + 1 > hi:
                            raise Infeasible(f"op {a}: no start time (shaving)")
                        self.es[a] = best + 1
                    else:
                        if not self.refutes(a, "tail", hi):
                            continue
                        best = hi; L, R = lo, hi - 1
                        while L <= R:
                            t = (L + R) // 2
                            if self.refutes(a, "tail", t):
                                best = t; R = t - 1
                            else:
                                L = t + 1
                        if best - 1 < lo:
                            raise Infeasible(f"op {a}: no start time (shaving)")
                        self.ls[a] = best - 1
                    changed = True
                    self.propagate(seeds=[a])
            if not changed:
                break


def cell_feasible(inst: Instance, c: int, T: int, es: List[int], ls: List[int], q0: List[int],
                  time_limit: float = 2.0, threads: int = 1) -> Optional[bool]:
    """Exact feasibility of cell c alone: s in [es, ls], e + q0 <= T, intra-cell
    precedences, machine disjunctions.  True/False when decided, None on time limit."""
    from ortools.sat.python import cp_model
    ops = inst.ops_of_cell[c]
    if not ops:
        return True
    mdl = cp_model.CpModel()
    start = {}; end = {}
    ivs = {i: [] for i in inst.machines_of_cell[c]}
    for a in ops:
        s = mdl.NewIntVar(int(es[a]), int(ls[a]), f"s{a}"); e = mdl.NewIntVar(0, int(T - q0[a]), f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in inst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}")); lits.append(x)
        mdl.AddExactlyOne(lits)
    for a in ops:
        p_ = inst.op_pred[a]
        if p_ >= 0 and inst.op_cell[p_] == c:
            mdl.Add(start[a] >= end[p_])
    for i, iv in ivs.items():
        if iv:
            mdl.AddNoOverlap(iv)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    st = solver.Solve(mdl)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return True
    if st == cp_model.INFEASIBLE:
        return False
    return None


def infeasible_T(inst: Instance, T: int, h: List[int], q: List[int], shaving: bool = True, sweeps: int = 2,
                 cell_time: float = 2.0, threads: int = 1, shave_budget: Optional[float] = None) -> Tuple[bool, str]:
    """True if makespan <= T is proven impossible."""
    w = Windows(inst, T, h, q)
    try:
        if shaving:
            w.shave(sweeps=sweeps, time_budget=shave_budget)
        else:
            w.propagate()
    except Infeasible as e:
        return True, f"propagation: {e}"
    for c in range(inst.n_cells):
        if cell_feasible(inst, c, T, w.es, w.ls, w.q0, cell_time, threads) is False:
            return True, f"cell {c} infeasible with propagated windows"
    return False, "not proven"


def destructive_bound(inst: Instance, lb: int, ub: int, shaving: bool = True, sweeps: int = 2, cell_time: float = 2.0,
                      threads: int = 1, time_budget: Optional[float] = None, log=None) -> Dict:
    """Binary search for the largest proven-infeasible T in [lb, ub - 1]; returns
    T + 1 (each proven-infeasible T is a valid bound by itself)."""
    from .bounds import heads_tails
    t0 = time.time()
    h, q = heads_tails(inst)
    best = lb; lo = lb; hi = ub - 1; tests = 0; reason = ""
    while lo <= hi:
        if time_budget and time.time() - t0 > time_budget:
            break
        T = (lo + hi) // 2
        left = None if time_budget is None else max(1.0, time_budget - (time.time() - t0))
        inf, why = infeasible_T(inst, T, h, q, shaving, sweeps, cell_time, threads, shave_budget=left)
        tests += 1
        if log:
            log.info("  destructive: T=%d -> %s (%s) [%.1fs]", T, "infeasible" if inf else "open", why, time.time() - t0)
        if inf:
            best = max(best, T + 1); reason = why; lo = T + 1
        else:
            hi = T - 1
    return {"lb": best, "tests": tests, "time": time.time() - t0, "reason": reason}


# ---- cell-local shaving with window exchange (message passing) ---------------
def _cell_propagate(inst: Instance, c: int, T: int, es: Dict[int, int], ls: Dict[int, int], fixed_on) -> bool:
    """Rules (E) and (EF) of cell c alone on local windows (dicts over the cell's
    ops) plus intra-cell job precedence. Returns True if anything changed."""
    pmin = inst.p_min
    ops = inst.ops_of_cell[c]
    m = len(inst.machines_of_cell[c])
    changed = False
    for _ in range(50):
        ch = False
        for a in ops:                                               # intra-cell precedence
            p_ = inst.op_pred[a]
            if p_ >= 0 and inst.op_cell[p_] == c:
                v = es[p_] + pmin[p_]
                if v > es[a]:
                    es[a] = v; ch = True
                v = ls[a] - pmin[p_]
                if v < ls[p_]:
                    ls[p_] = v; ch = True
        for a in ops:
            if es[a] > ls[a]:
                raise Infeasible(f"cell {c}: window of op {a} empty")
        for H in sorted(set(es[a] for a in ops)):                   # (E)
            sel = sorted((ls[a] + pmin[a], pmin[a]) for a in ops if es[a] >= H)
            e = 0
            for D, pm in sel:
                e += pm
                if e > m * (D - H):
                    raise Infeasible(f"cell {c} energy")
        for i in inst.machines_of_cell[c]:                          # (EF)
            fo = fixed_on[i]
            if len(fo) <= 1:
                continue
            p = {a: inst.op_proc[a][i] for a in fo}
            new = _edge_finding(fo, {a: es[a] for a in fo}, {a: ls[a] + p[a] for a in fo}, p)
            for a, v in new.items():
                if v > es[a]:
                    es[a] = v; ch = True
            new = _edge_finding(fo, {a: T - ls[a] - p[a] for a in fo}, {a: T - es[a] for a in fo}, p)
            for a, v in new.items():
                nl = T - v - p[a]
                if nl < ls[a]:
                    ls[a] = nl; ch = True
        changed |= ch
        if not ch:
            break
    return changed


def shave_local_cell(inst: Instance, c: int, T: int, es_in: Dict[int, int], ls_in: Dict[int, int], fixed_on,
                     sweeps: int = 2, exact: bool = False, q0: Optional[List[int]] = None,
                     cp_time: float = 1.0) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Shaving of cell c's operations using only the cell's own rules
    (`exact=True`: the cell's exact feasibility problem, CP-SAT, as refuter;
    a time-out counts as not refuted). Returns tightened local windows;
    raises Infeasible."""
    es = dict(es_in); ls = dict(ls_in)
    ops = inst.ops_of_cell[c]
    _cell_propagate(inst, c, T, es, ls, fixed_on)
    if exact:
        full_es = [0] * inst.n_ops; full_ls = [0] * inst.n_ops

    def refutes(a, side, t):
        e2, l2 = dict(es), dict(ls)
        if side == "head":
            l2[a] = min(l2[a], t)
        else:
            e2[a] = max(e2[a], t)
        try:
            _cell_propagate(inst, c, T, e2, l2, fixed_on)
        except Infeasible:
            return True
        if not exact:
            return False
        for b in ops:
            full_es[b] = e2[b]; full_ls[b] = l2[b]
        return cell_feasible(inst, c, T, full_es, full_ls, q0, cp_time, 1) is False

    for _ in range(sweeps):
        changed = False
        for side in ("head", "tail"):
            for a in ops:
                lo, hi = es[a], ls[a]
                if side == "head":
                    if not refutes(a, "head", lo):
                        continue
                    best = lo; L, R = lo + 1, hi
                    while L <= R:
                        t = (L + R) // 2
                        if refutes(a, "head", t):
                            best = t; L = t + 1
                        else:
                            R = t - 1
                    if best + 1 > hi:
                        raise Infeasible(f"cell {c}: op {a} no start time")
                    es[a] = best + 1
                else:
                    if not refutes(a, "tail", hi):
                        continue
                    best = hi; L, R = lo, hi - 1
                    while L <= R:
                        t = (L + R) // 2
                        if refutes(a, "tail", t):
                            best = t; R = t - 1
                        else:
                            L = t + 1
                    if best - 1 < lo:
                        raise Infeasible(f"cell {c}: op {a} no start time")
                    ls[a] = best - 1
                changed = True
                _cell_propagate(inst, c, T, es, ls, fixed_on)
        if not changed:
            break
    return es, ls


def shave_local(w: "Windows", rounds: int = 20, sweeps: int = 2, time_budget: Optional[float] = None,
                pool=None, log=None, exact: bool = False) -> int:
    """Message passing: every cell shaves locally (in parallel if `pool` is a
    multiprocessing pool), windows are exchanged along the jobs by (P), until no
    cell changes. Raises Infeasible. Returns the number of rounds."""
    inst = w.inst; t0 = time.time()
    w.propagate()
    for r in range(rounds):
        if time_budget and time.time() - t0 > time_budget:
            return r
        changed = False
        args = [(c, {a: w.es[a] for a in inst.ops_of_cell[c]}, {a: w.ls[a] for a in inst.ops_of_cell[c]})
                for c in range(inst.n_cells) if inst.ops_of_cell[c]]
        if pool is None:
            results = [(c, shave_local_cell(inst, c, w.T, e, l, w.fixed_on, sweeps, exact, w.q0)) for c, e, l in args]
        else:
            results = pool.starmap(_shave_worker, [(c, w.T, e, l, sweeps, exact, w.q0) for c, e, l in args])
            for c, res in results:
                if isinstance(res, str):
                    raise Infeasible(res)
        for c, (es, ls) in results:
            for a in es:
                if es[a] > w.es[a]:
                    w.es[a] = es[a]; changed = True
                if ls[a] < w.ls[a]:
                    w.ls[a] = ls[a]; changed = True
        w.propagate()          # (P) exchange + global rules
        if log:
            log.info("    local shaving round %d: %s [%.0fs]", r + 1, "changed" if changed else "fixed point", time.time() - t0)
        if not changed:
            return r + 1
    return rounds


_W_INST = None


def _shave_init(inst_json: str):
    global _W_INST
    _W_INST = Instance.from_json(inst_json)
    _W_INST._fixed_on = {i: [] for i in range(_W_INST.n_machines)}
    for a in range(_W_INST.n_ops):
        if len(_W_INST.op_proc[a]) == 1:
            _W_INST._fixed_on[next(iter(_W_INST.op_proc[a]))].append(a)


def _shave_worker(c, T, es, ls, sweeps, exact=False, q0=None):
    try:
        return c, shave_local_cell(_W_INST, c, T, es, ls, _W_INST._fixed_on, sweeps, exact, q0)
    except Infeasible as e:
        return c, str(e)


def infeasible_T_local(inst: Instance, T: int, h, q, sweeps: int = 2, rounds: int = 20, cell_time: float = 2.0,
                       threads: int = 1, pool=None, time_budget: Optional[float] = None, log=None,
                       exact: bool = False) -> Tuple[bool, str]:
    w = Windows(inst, T, h, q)
    try:
        shave_local(w, rounds, sweeps, time_budget, pool, log, exact)
    except Infeasible as e:
        return True, f"local shaving: {e}"
    for c in range(inst.n_cells):
        if cell_feasible(inst, c, T, w.es, w.ls, w.q0, cell_time, threads) is False:
            return True, f"cell {c} infeasible with shaved windows"
    return False, "not proven"
