"""Penalized information relaxation (Section 5.4): own-duration penalties.

Penalty (Definition 'penalty', with the start time clipped at a horizon
T_bar so that the inner problem is bounded):

    zeta_theta(S, omega) = sum_a (theta_{a,i} + eta_{a,i} min(s_S(a), T_bar)) (D_{a,i}(omega) - ptilde_{a,i}),
    i = i_S(a),   ptilde_{a,i} = p_{a,i} (1 + lam_i / mu_i) = E[D_{a,i}].

Zero mean (Lemma 'Zero mean') holds for every non-anticipative policy because
(i_a, s_a) is determined by the history, which is independent of (D_{a,i})_i;
clipping keeps this property.  Penalized certificate (Theorem 'Penalized
certificate'):

    Psi_theta(omega) = min_S { Cmax(S) - zeta_theta(S, omega) }   <=  Cmax^sigma(omega) - zeta_theta(S^sigma(omega), omega)

for every sigma in P_NA and every omega, hence z^NA >= E[Psi_theta] >= E[Psi_hat_theta].

Computing a lower bound Psi_hat_theta(omega) (valid by construction):
  * `cpsat`:  CP-SAT on the rounded scenario instance (Lemma EC.3; start
    times in the integer model are mapped so that the integer objective is
    at most the real one, see `_coefficients`), objective scaled by W and
    rounded down; proven bound / (W kappa).  Schedules with a start beyond
    T_bar are covered by the term T_bar - K(omega) (see `penalty_magnitude`).
  * `windows`: propagation-based bound for large instances: for every T in
    [LB, T_bar], T minus the largest penalty attainable within the windows
    at T.

Choice of theta (paragraph 'Choosing the penalty'): supergradient ascent of
the sample average of Psi_hat over TRAINING scenarios (gradient at the
optimal schedule of the penalized problem: -x_{a,i}(D - ptilde) for theta
and -x_{a,i} s_a (D - ptilde) for eta), projected onto boxes; evaluation on
INDEPENDENT scenarios.  Parameterizations: 'scalar' (one theta, one eta),
'cell' (per cell), 'op' (per operation and machine).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .instances import Instance
from .simulator import Scenario, Realized, round_scenario, expected_durations, sample_scenario


@dataclass
class Penalty:
    inst: Instance
    theta: Dict[Tuple[int, int], float]
    eta: Dict[Tuple[int, int], float]
    t_bar: float
    ptilde: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.ptilde:
            self.ptilde = expected_durations(self.inst)

    @staticmethod
    def zero(inst: Instance, t_bar: float) -> "Penalty":
        return Penalty(inst, {}, {}, t_bar)

    def coef(self, a: int, i: int) -> Tuple[float, float]:
        return self.theta.get((a, i), 0.0), self.eta.get((a, i), 0.0)

    # ---- the penalty of a realized schedule
    def zeta(self, R: Realized, D: Scenario) -> float:
        z = 0.0
        for a in range(self.inst.n_ops):
            i = R.assign[a]
            th, et = self.coef(a, i)
            if th == 0.0 and et == 0.0:
                continue
            s = min(R.start[a], self.t_bar)
            z += (th + et * s) * (D[a][i] - self.ptilde[a][i])
        return z

    def magnitude(self, D: Scenario) -> float:
        """K(omega) = sum_a max_i max(|theta|, |theta + eta T_bar|) |D - ptilde|:
        an upper bound on |zeta| over all schedules."""
        K = 0.0
        for a in range(self.inst.n_ops):
            best = 0.0
            for i in self.inst.op_proc[a]:
                th, et = self.coef(a, i)
                m = max(abs(th), abs(th + et * self.t_bar)) * abs(D[a][i] - self.ptilde[a][i])
                best = max(best, m)
            K += best
        return K


# ---- CP-SAT bound on the penalized scenario problem ------------------------
@dataclass
class PenalizedBound:
    psi_hat: float            # valid lower bound on Psi_theta(omega), original time units
    psi_model: float          # CP-SAT proven bound (schedules within T_bar)
    tail_term: float          # T_bar - K(omega)
    status: str
    value: float              # objective of the best schedule found (upper bound on Psi, info only)
    realized: Optional[Realized] = None   # best schedule found (for supergradients), rounded units / kappa
    time: float = 0.0


def penalized_bound_cpsat(inst: Instance, D: Scenario, pen: Penalty, kappa: float = 1.0, time_limit: float = 10.0,
                          threads: int = 1, W: int = 1000, hint: Optional[Realized] = None) -> PenalizedBound:
    from ortools.sat.python import cp_model
    t0 = time.time()
    sinst = round_scenario(inst, D, kappa)
    Tb = int(math.ceil(kappa * pen.t_bar))
    mdl = cp_model.CpModel()
    start, end, pres = {}, {}, {}
    ivs = {i: [] for i in range(inst.n_machines)}
    cmax = mdl.NewIntVar(0, Tb + max(max(d.values()) for d in sinst.op_proc), "cmax")
    obj_terms = [W * cmax]
    for a in range(inst.n_ops):
        lo = sinst.release[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        s = mdl.NewIntVar(lo, Tb, f"s{a}"); e = mdl.NewIntVar(0, Tb + 10 ** 6, f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in sinst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}")); lits.append(x); pres[(a, i)] = x
            th, et = pen.coef(a, i)
            if th == 0.0 and et == 0.0:
                continue
            dev = D[a][i] - pen.ptilde[a][i]
            alpha = -th * dev; beta = -et * dev          # real cost term alpha + beta * s_clip (s_clip <= T_bar)
            A = int(math.floor(W * kappa * alpha)); B = int(math.floor(W * beta))
            # u = s if beta >= 0 (u/kappa <= s_real), u = s + 1 if beta < 0 ((s+1)/kappa >= s_real): see module doc
            u = mdl.NewIntVar(0, Tb + 1, f"u{a}_{i}")
            if B >= 0:
                mdl.Add(u >= s).OnlyEnforceIf(x)
            else:
                mdl.Add(u <= s + 1).OnlyEnforceIf(x)
            mdl.Add(u == 0).OnlyEnforceIf(x.Not())
            if A != 0:
                obj_terms.append(A * x)
            if B != 0:
                obj_terms.append(B * u)
        mdl.AddExactlyOne(lits)
        if inst.op_succ[a] < 0:
            mdl.Add(cmax >= e)
        if inst.op_pred[a] >= 0:
            mdl.Add(s >= end[inst.op_pred[a]])
    for i in range(inst.n_machines):
        if ivs[i]:
            mdl.AddNoOverlap(ivs[i])
    mdl.Minimize(sum(obj_terms))
    if hint is not None:
        for a in range(inst.n_ops):
            mdl.AddHint(start[a], min(Tb, int(kappa * hint.start[a])))
            for i in sinst.op_proc[a]:
                mdl.AddHint(pres[(a, i)], 1 if hint.assign[a] == i else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = threads
    st = solver.Solve(mdl)
    name = solver.StatusName(st)
    K = pen.magnitude(D)
    tail = pen.t_bar - K
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bound = solver.BestObjectiveBound() / (W * kappa)
        value = solver.ObjectiveValue() / (W * kappa)
        assign = [next(i for i in sinst.op_proc[a] if solver.Value(pres[(a, i)])) for a in range(inst.n_ops)]
        s_ = [solver.Value(start[a]) / kappa for a in range(inst.n_ops)]
        R = Realized(assign, s_, [s_[a] + D[a][assign[a]] for a in range(inst.n_ops)])
    elif st == cp_model.INFEASIBLE:
        bound = math.inf; value = math.inf; R = None      # no schedule within T_bar: only the tail term remains
    else:
        bound = -math.inf; value = math.inf; R = None
    psi = min(bound, tail) if bound < math.inf else tail
    return PenalizedBound(psi, bound, tail, name, value, R, time.time() - t0)


# ---- window-based bound for large instances ---------------------------------
def penalized_bound_windows(inst: Instance, D: Scenario, pen: Penalty, lb: int, kappa: float = 1.0,
                            step: Optional[int] = None) -> PenalizedBound:
    """Psi >= min( min_{T in [lb, T_bar]} [ T/kappa - max_{S: Cmax<=T} zeta ],  T_bar - K ).
    The inner max is relaxed to independent choices per operation within the
    propagated windows at T (valid, cheap, usually weak)."""
    from .bounds import heads_tails
    from .propagation import Windows, Infeasible
    t0 = time.time()
    sinst = round_scenario(inst, D, kappa)
    h, q = heads_tails(sinst)
    Tb = int(math.ceil(kappa * pen.t_bar))
    step = step or max(1, (Tb - lb) // 40)
    best = math.inf
    T = lb
    while T <= Tb:
        w = Windows(sinst, T, h, q)
        try:
            w.propagate()
        except Infeasible:
            T += step; continue
        zmax = 0.0
        for a in range(inst.n_ops):
            m = -math.inf
            for i in inst.op_proc[a]:
                th, et = pen.coef(a, i)
                dev = D[a][i] - pen.ptilde[a][i]
                es, ls = w.es[a] / kappa, min(w.ls[a] / kappa + 1.0 / kappa, pen.t_bar)
                v = max((th + et * es) * dev, (th + et * ls) * dev)
                m = max(m, v)
            zmax += m
        best = min(best, T / kappa - zmax)
        T += step
    tail = pen.t_bar - pen.magnitude(D)
    return PenalizedBound(min(best, tail), best, tail, "windows", math.inf, None, time.time() - t0)


# ---- fitting theta on training scenarios ------------------------------------
def _keys(inst: Instance, param: str):
    if param == "scalar":
        return lambda a, i: 0
    if param == "cell":
        return lambda a, i: inst.op_cell[a]
    if param == "op":
        return lambda a, i: (a, i)
    raise ValueError(param)


def fit_penalty(inst: Instance, train_seeds: List[Tuple[int, int]], t_bar: float, param: str = "cell",
                iters: int = 20, step0: float = 0.05, theta_max: float = 2.0, eta_max: Optional[float] = None,
                kappa: float = 1.0, cp_time: float = 10.0, threads: int = 1, log=None,
                use_eta: bool = True) -> Penalty:
    """Projected supergradient ascent of (1/N) sum_n Psi_hat_theta(omega_n) over
    the training scenarios (base_seed, index) in `train_seeds`."""
    key = _keys(inst, param)
    pt = expected_durations(inst)
    eta_max = eta_max if eta_max is not None else theta_max / t_bar
    th: Dict = {}; et: Dict = {}
    Ds = [sample_scenario(inst, b, n) for b, n in train_seeds]
    hints = [None] * len(Ds)
    best_val = -math.inf; best = (dict(th), dict(et))

    def build():
        theta = {}; eta = {}
        for a in range(inst.n_ops):
            for i in inst.op_proc[a]:
                k = key(a, i)
                theta[(a, i)] = th.get(k, 0.0); eta[(a, i)] = et.get(k, 0.0)
        return Penalty(inst, theta, eta, t_bar, pt)

    def objective_and_grad(th_, et_):
        nonlocal th, et
        th, et = th_, et_
        pen = build()
        g_th: Dict = {}; g_et: Dict = {}; total = 0.0
        for n, D in enumerate(Ds):
            r = penalized_bound_cpsat(inst, D, pen, kappa, cp_time, threads, hint=hints[n])
            total += r.psi_hat
            if r.realized is None:
                continue
            hints[n] = r.realized
            R = r.realized
            for a in range(inst.n_ops):
                i = R.assign[a]; k = key(a, i)
                dev = D[a][i] - pt[a][i]; s_ = min(R.start[a], t_bar)
                g_th[k] = g_th.get(k, 0.0) - dev / len(Ds)
                g_et[k] = g_et.get(k, 0.0) - s_ * dev / len(Ds)
        return total / len(Ds), g_th, g_et

    cur, g_th, g_et = objective_and_grad({}, {})
    best_val = cur; best = ({}, {})
    if log:
        log.info("  fit it 0: mean Psi_hat = %.3f (theta = 0)", cur)
    step = step0
    for it in range(1, iters + 1):
        gn = math.sqrt(sum(v * v for v in g_th.values()) + sum(v * v for v in g_et.values()))
        if gn < 1e-12:
            break
        improved = False
        for _ in range(4):                                   # backtracking line search on the sample average
            th_new = {k: min(theta_max, max(-theta_max, th.get(k, 0.0) + step * theta_max * v / gn)) for k, v in g_th.items()}
            et_new = dict(et)
            if use_eta:
                et_new = {k: min(eta_max, max(-eta_max, et.get(k, 0.0) + step * eta_max * v / gn)) for k, v in g_et.items()}
            th_old, et_old = dict(th), dict(et)
            val, g_th2, g_et2 = objective_and_grad(th_new, et_new)
            if val > cur + 1e-9:
                cur, g_th, g_et = val, g_th2, g_et2; improved = True
                break
            th, et = th_old, et_old
            step *= 0.5
        if log:
            log.info("  fit it %d: mean Psi_hat = %.3f  step %.4f %s", it, cur, step, "" if improved else "(no improvement)")
        if cur > best_val:
            best_val = cur; best = (dict(th), dict(et))
        if not improved:
            # non-monotone move (subgradient methods are not monotone): take the step anyway with a diminishing size
            th = {k: min(theta_max, max(-theta_max, th.get(k, 0.0) + step * theta_max * v / gn)) for k, v in g_th.items()}
            if use_eta:
                et = {k: min(eta_max, max(-eta_max, et.get(k, 0.0) + step * eta_max * v / gn)) for k, v in g_et.items()}
            cur, g_th, g_et = objective_and_grad(th, et)
            step = max(step, step0 / (1 + it))
        else:
            step *= 1.2
    th, et = best
    return build()


# ---- evaluation on independent scenarios ------------------------------------
@dataclass
class PenalizedReport:
    n: int
    psi_mean: float
    psi_lcb: float
    pi_mean: float             # unpenalized (theta = 0) PI bound on the same scenarios, same method
    policy_mean: Dict[str, float]
    gap_mean: Dict[str, float]       # E[Cmax^sigma] - E[Psi_hat]  (paired, penalty as control variate)
    gap_ucb: Dict[str, float]
    min_pathwise: Dict[str, float]   # min_n Cmax - zeta - Psi_hat  (>= 0 by Theorem 'Penalized certificate')
    zeta_mean: Dict[str, float]      # sample mean of the penalty under each policy (should be ~0, Lemma 'Zero mean')


def evaluate_penalized(inst: Instance, pen: Penalty, policies: Dict[str, Callable[[Scenario], Realized]],
                       n_scen: int, base_seed: int, kappa: float = 1.0, cp_time: float = 10.0, threads: int = 1,
                       alpha: float = 0.05, log=None, method: str = "cpsat", lb_for_windows: int = 0) -> PenalizedReport:
    from scipy import stats
    zero = Penalty.zero(inst, pen.t_bar)
    psis, pis = [], []
    vals = {k: [] for k in policies}; zetas = {k: [] for k in policies}
    for n in range(n_scen):
        D = sample_scenario(inst, base_seed, n)
        if method == "cpsat":
            r = penalized_bound_cpsat(inst, D, pen, kappa, cp_time, threads)
            r0 = penalized_bound_cpsat(inst, D, zero, kappa, cp_time, threads)
        else:
            r = penalized_bound_windows(inst, D, pen, lb_for_windows, kappa)
            r0 = penalized_bound_windows(inst, D, zero, lb_for_windows, kappa)
        psis.append(r.psi_hat); pis.append(r0.psi_hat)
        for k, pol in policies.items():
            R = pol(D)
            vals[k].append(R.makespan); zetas[k].append(pen.zeta(R, D))
        if log and (n + 1) % 10 == 0:
            log.info("  scenario %d/%d  Psi_hat=%.2f  PI=%.2f", n + 1, n_scen, r.psi_hat, r0.psi_hat)
    psis = np.array(psis); pis = np.array(pis)
    tq = stats.t.ppf(1 - alpha, n_scen - 1)
    rep = PenalizedReport(n_scen, float(psis.mean()), float(psis.mean() - tq * psis.std(ddof=1) / math.sqrt(n_scen)),
                          float(pis.mean()), {}, {}, {}, {}, {})
    for k in policies:
        x = np.array(vals[k]); z = np.array(zetas[k]); d = x - z - psis
        rep.policy_mean[k] = float(x.mean()); rep.zeta_mean[k] = float(z.mean())
        rep.min_pathwise[k] = float(d.min())
        rep.gap_mean[k] = float(d.mean())
        rep.gap_ucb[k] = float(d.mean() + tq * d.std(ddof=1) / math.sqrt(n_scen))
    return rep
