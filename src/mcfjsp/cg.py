"""Certified column generation with heuristic pricing (Algorithm 1).

Master (RMP), min Cmax over convex combinations of cell schedules:
    sum_k lambda_ck = 1                               [gamma_c]   c in cells
    Cmax - sum_k C^k_a lambda_{c_a k} >= q_a          [rho_a]     a in ops
    sum_k s^k_{a+} lambda - sum_k C^k_a lambda >= 0   [pi_a]      a in transitions
    Cmax >= gamma_HT                                  [eta]       (head-tail cut)
Reduced cost of column k of cell c:  kappa_k(pi, rho) - gamma_c, with
    kappa_k = sum_a (rho_a + pi_a 1[a in T]) C^k_a - sum_a pi_{a-} 1[a- in T] s^k_a.

Certificate at duals (pi, rho, eta) with cell bounds beta_c (Theorem 1 +
head-tail cut):   Phi = sum_a rho_a q_a + eta * gamma_HT + sum_c beta_c,
where beta_c is any valid lower bound on min_k kappa_k (TI-LP value and/or
CP-SAT proven bound). Valid at every iteration, for any duals in the dual
domain, in particular at Wentges-smoothed duals.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bounds import head_tail_bounds, heads_tails
from .dispatch import best_dispatch, serial_sgs
from .instances import Instance
from .lp import LP
from .oracles import CellProblem, Column, cpsat_cell, ti_lp_cell
from .pricers import LPRoundPricer, RulePricer
from .primal import cell_lns
from .schedule import Schedule, check_feasible

log = logging.getLogger("mcfjsp.cg")

# ---- per-cell pricing (runs sequentially or in a worker process) -------
_W = {"inst": None, "cache": {}, "pricers": None}


def _worker_init(inst, use_heuristics):
    _W["inst"] = inst; _W["cache"] = {}
    _W["pricers"] = [RulePricer(), LPRoundPricer()] if use_heuristics else []


def price_cell(task):
    """task: dict(c, heads, tails, T, w, v, gamma_c, use_lp, need_exact, cp_time, threads, eps)
    Returns dict(beta, proven, columns=[Column...], exact_called)."""
    inst = _W["inst"]
    cp = CellProblem(inst, task["c"], task["heads"], task["tails"], task["T"])
    if not cp.ops:
        return dict(beta=0.0, proven=True, columns=[], exact_called=False)
    w, v, gamma_c, eps = task["w"], task["v"], task["gamma_c"], task["eps"]
    beta = -math.inf; context = {}; cols = []; seen = set()
    if task["use_lp"]:
        r = ti_lp_cell(cp, w, v, _W["cache"])
        beta = max(beta, r.bound); context["frac_start"] = r.frac_start
    for pr in _W["pricers"]:
        for col in pr(cp, w, v, context):
            if col.cost(w, v) - gamma_c < -eps and col.key() not in seen:
                cp.check_column(col); cols.append(col); seen.add(col.key())
    need_exact = task["need_exact"] or not cols
    proven = False; exact_called = False
    if need_exact and task["cp_time"] > 0:
        exact_called = True
        r = cpsat_cell(cp, w, v, task["cp_time"], task["threads"], lower_bound=beta if beta > -math.inf else None)
        beta = max(beta, r.bound); proven = r.proven
        if r.column is not None and r.value - gamma_c < -eps and r.column.key() not in seen:
            cols.append(r.column)
    if beta == -math.inf:
        beta = 0.0
    return dict(beta=beta, proven=proven, columns=cols, exact_called=exact_called)


@dataclass
class CGParams:
    time_limit: float = 600.0
    max_iter: int = 10_000
    eps_rc: float = 1e-6          # reduced-cost tolerance
    gap_tol: float = 0.0          # stop when (UB-LB)/LB <= gap_tol
    alpha: float = 0.7            # Wentges smoothing weight on the stability center
    alpha_min: float = 0.0
    alpha_step: float = 0.1
    exact_every: int = 5          # exact (CP-SAT) pricing at least every m iterations
    cp_time: float = 5.0          # CP-SAT time limit per cell pricing call
    cp_threads: int = 1
    ht_cp_time: float = 5.0       # CP-SAT time for head-tail cell bounds
    use_ht_cut: bool = True
    use_lp_oracle: bool = True
    pb_every: int = 10            # price-and-branch every k iterations
    pb_time: float = 10.0
    lns_time: float = 20.0        # cell-wise LNS budget per primal round (0 = off)
    lns_cell_time: float = 5.0
    use_heuristic_pricers: bool = True
    seed: int = 0
    preemptive_max_ops: int = 400
    n_workers: int = 1            # process pool over cells (>1 = parallel pricing)
    stop_on_gap: bool = True      # False: continue to LP optimality for bound studies


@dataclass
class CGResult:
    instance: str
    lb: float
    ub: int
    schedule: Optional[Schedule]
    gamma_ht: int
    z_rmp: float
    lb_cg: float                  # best Lagrangian/CG certificate (without HT max)
    iterations: int
    n_columns: int
    exact_calls: int
    time: float
    history: List[dict] = field(default_factory=list)
    status: str = ""

    @property
    def gap(self) -> float:
        return (self.ub - self.lb) / self.lb if self.lb > 0 else math.inf


class ColumnGeneration:
    def __init__(self, inst: Instance, params: CGParams = CGParams(), pricers=None):
        self.inst = inst
        self.P = params
        self.pricers = pricers if pricers is not None else ([RulePricer(), LPRoundPricer()] if params.use_heuristic_pricers else [])
        self.heads, self.tails = heads_tails(inst)
        self.columns: List[Column] = []
        self.col_var: List[int] = []
        self.col_keys = set()
        self.lp_cache: Dict = {}
        self.history: List[dict] = []
        self.exact_calls = 0

    # ---- RMP construction -------------------------------------------
    def _build_rmp(self, init_sched: Schedule, gamma_ht: int) -> None:
        inst = self.inst
        lp = LP("glop")
        self.v_cmax = lp.add_var(0.0, math.inf, 1.0, "Cmax")
        self.row_conv = [lp.add_row([], 1.0, 1.0, f"conv{c}") for c in range(inst.n_cells)]
        self.row_epi = [lp.add_row([(self.v_cmax, 1.0)], float(self.tails[a]), math.inf, f"epi{a}")
                        for a in range(inst.n_ops)]
        self.row_link = {a: lp.add_row([], 0.0, math.inf, f"link{a}") for a in inst.transitions}
        self.row_ht = lp.add_row([(self.v_cmax, 1.0)], float(gamma_ht), math.inf, "ht") if self.P.use_ht_cut else None
        self.lp = lp
        self.columns = []; self.col_var = []; self.col_keys = set()
        for col in self.schedule_to_columns(init_sched):
            self.add_column(col)

    def schedule_to_columns(self, sch: Schedule) -> List[Column]:
        cols = []
        for c in range(self.inst.n_cells):
            ops = self.inst.ops_of_cell[c]
            cols.append(Column(c, {a: sch.start[a] for a in ops}, {a: sch.comp[a] for a in ops},
                               {a: sch.assign[a] for a in ops}))
        return cols

    def add_column(self, col: Column) -> bool:
        key = (col.cell, col.key())
        if key in self.col_keys:
            return False
        inst = self.inst
        v = self.lp.add_var(0.0, math.inf, 0.0)
        self.lp.set_coef(self.row_conv[col.cell], v, 1.0)
        for a in col.start:
            self.lp.set_coef(self.row_epi[a], v, -float(col.comp[a]))
            if inst.is_transition[a]:
                self.lp.set_coef(self.row_link[a], v, -float(col.comp[a]))
            p_ = inst.op_pred[a]
            if p_ >= 0 and inst.is_transition[p_]:
                self.lp.set_coef(self.row_link[p_], v, float(col.start[a]))
        self.columns.append(col); self.col_var.append(v); self.col_keys.add(key)
        return True

    # ---- duals and weights ------------------------------------------
    def _duals(self):
        lp = self.lp
        gamma = [lp.dual(r) for r in self.row_conv]
        rho = [max(0.0, lp.dual(r)) for r in self.row_epi]
        pi = {a: max(0.0, lp.dual(r)) for a, r in self.row_link.items()}
        eta = max(0.0, lp.dual(self.row_ht)) if self.row_ht is not None else 0.0
        tot = sum(rho) + eta
        if abs(tot - 1.0) > 1e-6:  # numerical safety: renormalize onto the simplex
            if tot <= 0:
                rho = [1.0 / len(rho)] * len(rho); eta = 0.0
            else:
                rho = [r / tot for r in rho]; eta = eta / tot
        return gamma, rho, pi, eta

    def weights(self, rho, pi):
        inst = self.inst
        w = {}; v = {}
        for a in range(inst.n_ops):
            w[a] = rho[a] + (pi[a] if inst.is_transition[a] else 0.0)
            p_ = inst.op_pred[a]
            v[a] = pi[p_] if (p_ >= 0 and inst.is_transition[p_]) else 0.0
        return w, v

    def certificate(self, rho, eta, betas) -> float:
        return sum(rho[a] * self.tails[a] for a in range(self.inst.n_ops)) + eta * self.gamma_ht + sum(betas)

    # ---- primal side --------------------------------------------------
    def _set_horizon(self, T: int) -> None:
        """Reduce windows to the new horizon: C_a + q_a <= T. Columns that
        violate it are fixed to zero; oracle caches keyed by T are rebuilt."""
        self.T = T
        for k, col in enumerate(self.columns):
            if any(col.comp[a] + self.tails[a] > T for a in col.start):
                self.lp.set_var_bounds(self.col_var[k], 0.0, 0.0)
        self.lp_cache = {}

    def price_and_branch(self, time_limit: float) -> Optional[Schedule]:
        from ortools.sat.python import cp_model
        inst = self.inst
        active = [k for k, col in enumerate(self.columns)
                  if not any(col.comp[a] + self.tails[a] > self.T for a in col.start)]
        mdl = cp_model.CpModel()
        y = {k: mdl.NewBoolVar(f"y{k}") for k in active}
        cmax = mdl.NewIntVar(0, self.T, "cmax")
        by_cell: Dict[int, List[int]] = {}
        for k in active:
            by_cell.setdefault(self.columns[k].cell, []).append(k)
        for c in range(inst.n_cells):
            mdl.AddExactlyOne([y[k] for k in by_cell.get(c, [])])
        for a in range(inst.n_ops):
            c = inst.op_cell[a]
            mdl.Add(cmax >= sum(self.columns[k].comp[a] * y[k] for k in by_cell[c]) + self.tails[a])
        for a in inst.transitions:
            b = inst.op_succ[a]
            mdl.Add(sum(self.columns[k].start[b] * y[k] for k in by_cell[inst.op_cell[b]])
                    >= sum(self.columns[k].comp[a] * y[k] for k in by_cell[inst.op_cell[a]]))
        mdl.Minimize(cmax)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = self.P.cp_threads
        st = solver.Solve(mdl)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        assign = [0] * inst.n_ops; start = [0] * inst.n_ops; comp = [0] * inst.n_ops
        for k in active:
            if solver.Value(y[k]):
                col = self.columns[k]
                for a in col.start:
                    assign[a] = col.assign[a]; start[a] = col.start[a]; comp[a] = col.comp[a]
        sch = Schedule(assign, start, comp)
        check_feasible(inst, sch)
        return sch

    def repair_from_lp(self) -> Schedule:
        """Serial SGS with priorities = LP-weighted start times of the columns."""
        x = self.lp.values()
        pri = [0.0] * self.inst.n_ops
        for k, col in enumerate(self.columns):
            lam = x[self.col_var[k]]
            if lam > 1e-9:
                for a in col.start:
                    pri[a] += lam * col.start[a]
        sch = serial_sgs(self.inst, pri)
        check_feasible(self.inst, sch)
        return sch

    # ---- main loop ------------------------------------------------------
    def run(self) -> CGResult:
        P = self.P; inst = self.inst
        t0 = time.time()
        sch, rule, _ = best_dispatch(inst, self.tails)
        UB = sch.makespan; best_sched = sch
        ht = head_tail_bounds(inst, UB, cp_time=P.ht_cp_time, threads=P.cp_threads,
                              use_preemptive=inst.n_ops <= P.preemptive_max_ops)
        self.gamma_ht = ht["gamma_ht"]
        LB = float(self.gamma_ht); lb_cg = 0.0
        log.info("init: dispatch %s UB=%d  gamma_HT=%d  (energetic %d)", rule, UB, self.gamma_ht, ht["gamma_energetic"])
        self.T = UB
        self._build_rmp(sch, self.gamma_ht)
        center = None            # (rho, pi, eta) with the best certificate
        alpha = P.alpha
        it = 0; status = "iter_limit"; z = math.inf
        base_cp_time = P.cp_time
        while it < P.max_iter:
            if time.time() - t0 > P.time_limit:
                status = "time_limit"; break
            st = self.lp.solve()
            if st != "OPTIMAL":
                log.warning("RMP status %s; stopping column generation (bounds so far remain valid)", st)
                status = "rmp_" + st.lower(); break
            z = self.lp.objective
            gamma, rho, pi, eta = self._duals()
            # ---- smoothing
            if center is not None and alpha > 0:
                c_rho, c_pi, c_eta = center
                s_rho = [alpha * c_rho[a] + (1 - alpha) * rho[a] for a in range(inst.n_ops)]
                s_pi = {a: alpha * c_pi[a] + (1 - alpha) * pi[a] for a in pi}
                s_eta = alpha * c_eta + (1 - alpha) * eta
            else:
                s_rho, s_pi, s_eta = rho, pi, eta
            w, v = self.weights(s_rho, s_pi)
            # ---- pricing at smoothed duals
            added, betas, exact_all = self._price(w, v, gamma, it)
            phi = self.certificate(s_rho, s_eta, betas)
            if phi > lb_cg + 1e-9:
                lb_cg = phi; center = (s_rho, s_pi, s_eta)
            LB = max(LB, lb_cg)
            misprice = False
            if added == 0 and alpha > 0 and center is not None and (s_rho is not rho):
                # mis-price: no column at the smoothed point; price at the true duals
                misprice = True
                w, v = self.weights(rho, pi)
                added, betas2, exact_all = self._price(w, v, gamma, it, force_exact=True)
                phi2 = self.certificate(rho, eta, betas2)
                if phi2 > lb_cg + 1e-9:
                    lb_cg = phi2; center = (rho, pi, eta)
                LB = max(LB, lb_cg)
                alpha = max(P.alpha_min, alpha - P.alpha_step)
            # ---- primal
            if it % P.pb_every == 0 or added == 0:
                pb = self.price_and_branch(P.pb_time)
                cands = [pb] if pb is not None else []
                try:
                    cands.append(self.repair_from_lp())
                except Exception as e:  # pragma: no cover
                    log.warning("repair failed: %s", e)
                if P.lns_time > 0:
                    seed_s = min(cands + [best_sched], key=lambda s_: s_.makespan)
                    cands.append(cell_lns(inst, seed_s, P.lns_time, P.lns_cell_time, P.cp_threads, log, seed=P.seed + it))
                for s_ in cands:
                    if s_.makespan < UB:
                        UB = s_.makespan; best_sched = s_
                        for col in self.schedule_to_columns(s_):
                            self.add_column(col)
                        self._set_horizon(UB)
                        log.info("  UB improved to %d", UB)
            it += 1
            rec = dict(it=it, t=time.time() - t0, z_rmp=z, phi=phi, lb=LB, ub=UB, cols=len(self.columns),
                       added=added, alpha=alpha, misprice=misprice, exact_all=exact_all)
            self.history.append(rec)
            log.info("it %3d  t=%6.1fs  z_RMP=%.3f  Phi=%.3f  LB=%.3f  UB=%d  gap=%.2f%%  cols=%d  +%d  a=%.2f%s",
                     it, rec["t"], z, phi, LB, UB, 100 * (UB - LB) / max(LB, 1e-9), len(self.columns), added, alpha,
                     " (misprice)" if misprice else "")
            if P.stop_on_gap and LB > 0 and (UB - LB) / LB <= P.gap_tol + 1e-12:
                status = "gap_closed"; break
            if added == 0 and exact_all:
                status = "lp_optimal"; break
            if added == 0 and not exact_all:
                # exact pricing did not prove optimality within its time limit; keep iterating
                # with a longer exact time limit, but stop if nothing changes
                P.cp_time *= 2
                if P.cp_time > 8 * base_cp_time:
                    status = "stalled"; break
        # final primal round with the remaining budget
        if P.lns_time > 0:
            remaining = max(0.0, P.time_limit - (time.time() - t0))
            fin = cell_lns(inst, best_sched, min(P.lns_time * 2, remaining) if remaining > 0 else 1.0,
                           P.lns_cell_time, P.cp_threads, log, seed=P.seed + 1)
            if fin.makespan < UB:
                UB = fin.makespan; best_sched = fin
                log.info("  final UB %d", UB)
            LB = max(LB, lb_cg)
        P.cp_time = base_cp_time
        self.close()
        return CGResult(inst.name, LB, UB, best_sched, self.gamma_ht, z, lb_cg, it, len(self.columns),
                        self.exact_calls, time.time() - t0, self.history, status)

    def _pool(self):
        if self.P.n_workers <= 1:
            if _W["inst"] is not self.inst:
                _worker_init(self.inst, self.P.use_heuristic_pricers)
            return None
        if getattr(self, "_executor", None) is None:
            from concurrent.futures import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor(self.P.n_workers, initializer=_worker_init,
                                                 initargs=(self.inst, self.P.use_heuristic_pricers))
        return self._executor

    def close(self):
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown(); self._executor = None

    def _price(self, w, v, gamma, it, force_exact: bool = False):
        """Price every cell (in parallel when n_workers > 1). Returns (#added, betas, exact_all)."""
        P = self.P; inst = self.inst
        tasks = []
        for c in range(inst.n_cells):
            ops = inst.ops_of_cell[c]
            need_exact = P.exact_every > 0 and (force_exact or it % P.exact_every == 0)
            tasks.append(dict(c=c, heads=self.heads, tails=self.tails, T=self.T,
                              w={a: w[a] for a in ops}, v={a: v[a] for a in ops}, gamma_c=gamma[c],
                              use_lp=P.use_lp_oracle, need_exact=need_exact,
                              cp_time=P.cp_time if P.exact_every > 0 else 0.0,
                              threads=max(1, P.cp_threads // max(1, min(P.n_workers, inst.n_cells))) if P.n_workers > 1 else P.cp_threads,
                              eps=P.eps_rc))
        pool = self._pool()
        results = list(pool.map(price_cell, tasks)) if pool is not None else [price_cell(t) for t in tasks]
        added = 0; betas = []; exact_all = True
        for res in results:
            n_new = sum(1 for col in res["columns"] if self.add_column(col))
            added += n_new
            if res["exact_called"]:
                self.exact_calls += 1
            has_ops = bool(inst.ops_of_cell[tasks[len(betas)]["c"]])
            if has_ops and not (res["exact_called"] and res["proven"] and n_new == 0):
                exact_all = False
            betas.append(res["beta"])
        return added, betas, exact_all
