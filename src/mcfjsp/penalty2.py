"""Revealed-duration penalties (extension of Section 5.4).

For ordered pairs (a, b) of operations in the same cell and a reference
machine r_b of b (its fastest eligible machine), let X_b = D_{b, r_b}(omega).
The penalty is

    zeta_c(S, omega) = sum_{(a,b)} c_{ab} ( X_b - E[ X_b | F_{s_S(a)} ] ),

where F_t is the event history (starts, completions, failures, repairs) up
to t.  Along a realized schedule, E[X_b | F_{s_a}] equals
    X_b                      if b ran on r_b and completed before a started,
    g_b(s_a - s_b)           if b is running on r_b at s_a (conditional_expectation of the path),
    ptilde_b = E[X_b]        otherwise (b not started, or processed on another machine:
                             D_{b,i} and D_{b,r_b} are independent).
Zero mean: for any non-anticipative policy the start time tau_a is a stopping
time, so E[E[X_b | F_{tau_a}]] = E[X_b] by the tower property; the
coefficients are constants.  Hence (Theorem 'Penalized certificate', same
proof) z^NA >= E[Psi_c], Psi_c(omega) = min_S {Cmax(S) - zeta_c(S, omega)},
and Cmax^sigma(omega) - zeta_c(S^sigma(omega), omega) >= Psi_c(omega) pathwise.
Own-duration penalties (penalty.py) are the case b = a with a start-time
multiplier; this family charges the clairvoyant for sequencing a with
knowledge of OTHER operations' durations (the mechanism of Proposition
'counterexample').

Lower bound Psi_hat_c(omega) by CP-SAT on the rounded instance (grid 1/kappa):
the per-pair term is replaced by an upper envelope over all real schedules
that map to the same integer relation between s_a and s_b (see
`_envelope_table`), so the integer objective is at most kappa W times the
real one, and the proven bound / (kappa W) is valid.  Schedules with a start
beyond the horizon T_bar are covered by T_bar - K(omega), K = sum of the
maximal absolute pair terms.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .instances import Instance
from .simulator import (Scenario, Paths, Realized, round_scenario, expected_durations, sample_scenario_paths,
                        conditional_expectation)


def reference_machines(inst: Instance) -> List[int]:
    return [min(inst.op_proc[b], key=lambda i: (inst.op_proc[b][i], i)) for b in range(inst.n_ops)]


@dataclass
class RevealedPenalty:
    """mode 'revealed': c_ab (X_b - E[X_b | F_{s_a}]) for all pairs;
    mode 'prestart':  c_ab 1[b not started at s_a] (X_b - E[X_b]) -- the
    multiplier is F_{s_a}-measurable and on that event E[X_b | F_{s_a}] = E[X_b],
    so the term has zero mean; it isolates the sequencing information and
    drops the in-progress martingale terms that cancel the charge."""
    inst: Instance
    c: Dict[Tuple[int, int], float]          # (a, b) -> coefficient
    t_bar: float
    ref: List[int] = field(default_factory=list)
    ptilde: Dict[int, Dict[int, float]] = field(default_factory=dict)
    mode: str = "prestart"
    theta: Dict[Tuple[int, int], float] = field(default_factory=dict)   # own-duration (machine-choice) component

    def __post_init__(self):
        if not self.ref:
            self.ref = reference_machines(self.inst)
        if not self.ptilde:
            self.ptilde = expected_durations(self.inst)

    @staticmethod
    def zero(inst: Instance, t_bar: float) -> "RevealedPenalty":
        return RevealedPenalty(inst, {}, t_bar)


    def g(self, b: int, D: Scenario, P: Paths, e: float) -> float:
        i = self.ref[b]
        return conditional_expectation(self.inst.op_proc[b][i], self.inst.lam[i], self.inst.mu[i], P[b][i], e)

    def cond_exp_along(self, b: int, R: Realized, D: Scenario, P: Paths, tau: float) -> float:
        """E[X_b | F_tau] along the realized schedule R."""
        i = self.ref[b]
        if R.assign[b] == i and R.start[b] <= tau:
            if R.comp[b] <= tau:
                return D[b][i]
            return self.g(b, D, P, tau - R.start[b])
        return self.ptilde[b][i]

    def zeta(self, R: Realized, D: Scenario, P: Paths) -> float:
        z = 0.0
        for (a, i), th in self.theta.items():
            if th != 0.0 and R.assign[a] == i:
                z += th * (D[a][i] - self.ptilde[a][i])
        for (a, b), cab in self.c.items():
            if cab == 0.0:
                continue
            if self.mode == "prestart":
                if R.start[b] >= R.start[a]:        # b not started when a starts (ties: simultaneous starts count as not started)
                    ib = R.assign[b]                # duration on the machine b actually uses (decided at s_b >= s_a)
                    z += cab * (D[b][ib] - self.ptilde[b][ib])
            else:
                z += cab * (D[b][self.ref[b]] - self.cond_exp_along(b, R, D, P, R.start[a]))
        return z

    def magnitude(self, D: Scenario, P: Paths) -> float:
        K = 0.0
        for a in range(self.inst.n_ops):
            K += max((abs(self.theta.get((a, i), 0.0)) * abs(D[a][i] - self.ptilde[a][i]) for i in self.inst.op_proc[a]), default=0.0)
        for (a, b), cab in self.c.items():
            if cab == 0.0:
                continue
            X = D[b][self.ref[b]]
            if self.mode == "prestart":
                K += abs(cab) * max(abs(D[b][i] - self.ptilde[b][i]) for i in self.inst.op_proc[b])
                continue
            lo, hi = self._g_range(b, D, P, 0.0, X)
            K += abs(cab) * max(abs(X - lo), abs(X - hi))
        return K

    def _g_range(self, b: int, D: Scenario, P: Paths, e0: float, e1: float) -> Tuple[float, float]:
        """min and max of g_b over [e0, e1] (g is piecewise linear with jumps at
        failure and repair epochs; evaluate at endpoints and around breakpoints)."""
        i = self.ref[b]
        p = self.inst.op_proc[b][i]
        epochs, reps = P[b][i]
        pts = [e0, e1]
        t = 0.0; w = 0.0
        for wk, rk in zip(epochs, reps):
            t += wk - w; w = wk
            for x in (t - 1e-9, t, t + 1e-9, t + rk - 1e-9, t + rk, t + rk + 1e-9):
                if e0 <= x <= e1:
                    pts.append(x)
            t += rk
        vals = [self.g(b, D, P, max(0.0, x)) for x in pts]
        return min(vals), max(vals)


# ---- CP-SAT bound on the penalized scenario problem --------------------------
@dataclass
class PenalizedBound2:
    psi_hat: float
    psi_model: float
    tail_term: float
    status: str
    value: float
    realized: Optional[Realized] = None
    time: float = 0.0


def _envelope_table(pen: RevealedPenalty, a: int, b: int, cab: float, D: Scenario, P: Paths, kappa: float,
                    K_int: int, W: int) -> Tuple[List[int], int]:
    """Integer table over k = s_a_int - s_b_int: entry 0 for k < 0 (b not
    started), entry j+1 for k = j in [0, K_int] with K_int = floor(kappa X);
    each entry is ceil(W kappa * max{ c (X - g(e)) : e in the real interval
    consistent with k }).  Also returns the 'b on another machine' constant."""
    X = D[b][pen.ref[b]]
    pt = pen.ptilde[b][pen.ref[b]]
    const = int(math.ceil(W * kappa * cab * (X - pt)))
    tab = [const]                                   # k < 0: not started
    if pen.mode == "prestart":
        # k = 0: real relation may be 'not started' (term const) or 'started' (term 0); k >= 1: started
        tab.append(max(const, 0))
        tab += [0] * K_int
        return tab, const
    for j in range(K_int + 1):
        e0 = max(0.0, (j - 1) / kappa); e1 = min(X, (j + 1) / kappa)
        if j >= K_int:
            e0 = max(0.0, X - 2.0 / kappa); e1 = X
        lo, hi = pen._g_range(b, D, P, e0, e1)
        v = cab * (X - lo) if cab >= 0 else cab * (X - hi)
        if j == 0:
            v = max(v, cab * (X - pt))
        if j >= K_int:
            v = max(v, 0.0)                         # real case may be 'completed' (zero term)
        tab.append(int(math.ceil(W * kappa * v)))
    return tab, const


def penalized_bound_cpsat2(inst: Instance, D: Scenario, P: Paths, pen: RevealedPenalty, kappa: float = 1.0,
                           time_limit: float = 10.0, threads: int = 1, W: int = 100,
                           hint: Optional[Realized] = None) -> PenalizedBound2:
    from ortools.sat.python import cp_model
    t0 = time.time()
    sinst = round_scenario(inst, D, kappa)
    Tb = int(math.ceil(kappa * pen.t_bar))
    mdl = cp_model.CpModel()
    start, end, pres = {}, {}, {}
    ivs = {i: [] for i in range(inst.n_machines)}
    maxp = max(max(d.values()) for d in sinst.op_proc)
    cmax = mdl.NewIntVar(0, Tb + maxp, "cmax")
    obj = [W * cmax]
    for a in range(inst.n_ops):
        lo = sinst.release[inst.op_job[a]] if inst.op_pos[a] == 0 else 0
        s = mdl.NewIntVar(lo, Tb, f"s{a}"); e = mdl.NewIntVar(0, Tb + maxp, f"e{a}")
        start[a] = s; end[a] = e
        lits = []
        for i, p in sinst.op_proc[a].items():
            x = mdl.NewBoolVar(f"x{a}_{i}")
            ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}")); lits.append(x); pres[(a, i)] = x
            th = pen.theta.get((a, i), 0.0)
            if th != 0.0:
                A = int(math.floor(W * kappa * (-th * (D[a][i] - pen.ptilde[a][i]))))
                if A != 0:
                    obj.append(A * x)
        mdl.AddExactlyOne(lits)
        if inst.op_succ[a] < 0:
            mdl.Add(cmax >= e)
        if inst.op_pred[a] >= 0:
            mdl.Add(s >= end[inst.op_pred[a]])
    for i in range(inst.n_machines):
        if ivs[i]:
            mdl.AddNoOverlap(ivs[i])
    for (a, b), cab in pen.c.items():
        if cab == 0.0:
            continue
        if pen.mode == "prestart":
            # term c (D_{b,i_b} - ptilde_{b,i_b}) if b not started when a starts.  k = s_a - s_b (integer grid):
            # real 'not started' => k <= 0; real 'started' => k >= 0; so k < 0: term, k = 0: max(term, 0), k > 0: 0.
            consts = {i: int(math.ceil(W * kappa * cab * (D[b][i] - pen.ptilde[b][i]))) for i in sinst.op_proc[b]}
            lo_c, hi_c = min(consts.values()), max(consts.values())
            v = mdl.NewIntVar(min(lo_c, 0), max(hi_c, 0), f"v{a}_{b}")
            mdl.Add(v == sum(consts[i] * pres[(b, i)] for i in sinst.op_proc[b]))
            m = mdl.NewIntVar(0, max(hi_c, 0), f"m{a}_{b}")
            mdl.AddMaxEquality(m, [v, 0])
            k = mdl.NewIntVar(-Tb - maxp, Tb + maxp, f"k{a}_{b}")
            mdl.Add(k == start[a] - start[b])
            y_neg = mdl.NewBoolVar(f"yn{a}_{b}"); y_zero = mdl.NewBoolVar(f"yz{a}_{b}"); y_pos = mdl.NewBoolVar(f"yp{a}_{b}")
            mdl.AddExactlyOne([y_neg, y_zero, y_pos])
            mdl.Add(k <= -1).OnlyEnforceIf(y_neg); mdl.Add(k == 0).OnlyEnforceIf(y_zero); mdl.Add(k >= 1).OnlyEnforceIf(y_pos)
            z = mdl.NewIntVar(min(lo_c, 0), max(hi_c, 0), f"z{a}_{b}")
            mdl.Add(z == v).OnlyEnforceIf(y_neg); mdl.Add(z == m).OnlyEnforceIf(y_zero); mdl.Add(z == 0).OnlyEnforceIf(y_pos)
            obj.append(-z)
            continue
        r = pen.ref[b]
        K_int = sinst.op_proc[b][r]
        tab, const = _envelope_table(pen, a, b, cab, D, P, kappa, K_int, W)
        k = mdl.NewIntVar(-Tb - maxp, Tb + maxp, f"k{a}_{b}")
        mdl.Add(k == start[a] - start[b])
        m1 = mdl.NewIntVar(-Tb - maxp, K_int, f"m{a}_{b}")
        mdl.AddMinEquality(m1, [k, K_int])
        idx = mdl.NewIntVar(-1, K_int, f"i{a}_{b}")
        mdl.AddMaxEquality(idx, [m1, -1])
        idx1 = mdl.NewIntVar(0, K_int + 1, f"j{a}_{b}")
        mdl.Add(idx1 == idx + 1)
        v = mdl.NewIntVar(min(tab), max(tab), f"v{a}_{b}")
        mdl.AddElement(idx1, tab, v)
        z = mdl.NewIntVar(min(min(tab), const), max(max(tab), const), f"z{a}_{b}")
        xb = pres[(b, r)]
        mdl.Add(z == v).OnlyEnforceIf(xb)
        mdl.Add(z == const).OnlyEnforceIf(xb.Not())
        obj.append(-z)
    mdl.Minimize(sum(obj))
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
    tail = pen.t_bar - pen.magnitude(D, P)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bound = solver.BestObjectiveBound() / (W * kappa)
        value = solver.ObjectiveValue() / (W * kappa)
        assign = [next(i for i in sinst.op_proc[a] if solver.Value(pres[(a, i)])) for a in range(inst.n_ops)]
        s_ = [solver.Value(start[a]) / kappa for a in range(inst.n_ops)]
        R = Realized(assign, s_, [s_[a] + D[a][assign[a]] for a in range(inst.n_ops)])
    elif st == cp_model.INFEASIBLE:
        bound = math.inf; value = math.inf; R = None
    else:
        bound = -math.inf; value = math.inf; R = None
    psi = min(bound, tail) if bound < math.inf else tail
    return PenalizedBound2(psi, bound, tail, name, value, R, time.time() - t0)


# ---- pairs and parameterizations --------------------------------------------
def candidate_pairs(inst: Instance, scope: str = "cell") -> List[Tuple[int, int]]:
    pairs = []
    if scope == "cell":
        for c in range(inst.n_cells):
            ops = inst.ops_of_cell[c]
            pairs += [(a, b) for a in ops for b in ops if a != b]
    elif scope == "machine":   # same reference machine
        ref = reference_machines(inst)
        by = {}
        for a in range(inst.n_ops):
            by.setdefault(ref[a], []).append(a)
        for ops in by.values():
            pairs += [(a, b) for a in ops for b in ops if a != b]
    else:
        raise ValueError(scope)
    return pairs


def _pair_key(inst: Instance, param: str):
    if param == "scalar":
        return lambda a, b: 0
    if param == "cell":
        return lambda a, b: inst.op_cell[b]
    if param == "b":
        return lambda a, b: b
    if param == "pair":
        return lambda a, b: (a, b)
    raise ValueError(param)


def _theta_key(inst: Instance, param: str):
    if param == "none":
        return None
    if param == "scalar":
        return lambda a, i: ("t", 0)
    if param == "machine":
        return lambda a, i: ("t", i)
    if param == "op":
        return lambda a, i: ("t", a, i)
    raise ValueError(param)


def fit_revealed(inst: Instance, train_seeds: List[Tuple[int, int]], t_bar: float, scope: str = "machine",
                 param: str = "b", iters: int = 15, step0: float = 0.05, c_max: float = 1.0, kappa: float = 1.0,
                 cp_time: float = 10.0, threads: int = 1, log=None, mode: str = "prestart",
                 theta_param: str = "none", init: Optional[Dict] = None) -> RevealedPenalty:
    """Projected supergradient ascent of the training sample average of
    Psi_hat_c (supergradient at the optimal schedule: -(X_b - E[X_b | F_{s_a}])
    per pair), with a backtracking line search and non-monotone fallback."""
    pairs = candidate_pairs(inst, scope) if scope != "none" else []
    key = _pair_key(inst, param)
    tkey = _theta_key(inst, theta_param)
    scen = [sample_scenario_paths(inst, b, n) for b, n in train_seeds]
    hints = [None] * len(scen)
    coef: Dict = dict(init) if init else {}
    pt = expected_durations(inst)

    def build(cf):
        theta = {}
        if tkey is not None:
            theta = {(a, i): cf.get(tkey(a, i), 0.0) for a in range(inst.n_ops) for i in inst.op_proc[a]}
        return RevealedPenalty(inst, {(a, b): cf.get(key(a, b), 0.0) for a, b in pairs}, t_bar, mode=mode, theta=theta)

    def objective_and_grad(cf):
        pen = build(cf)
        g: Dict = {}; total = 0.0
        for n, (D, P) in enumerate(scen):
            r = penalized_bound_cpsat2(inst, D, P, pen, kappa, cp_time, threads, hint=hints[n])
            total += r.psi_hat
            if r.realized is None:
                continue
            hints[n] = r.realized; R = r.realized
            if tkey is not None:
                for a in range(inst.n_ops):
                    i = R.assign[a]; k = tkey(a, i)
                    g[k] = g.get(k, 0.0) - (D[a][i] - pt[a][i]) / len(scen)
                    for i2 in inst.op_proc[a]:
                        g.setdefault(tkey(a, i2), 0.0)
            for a, b in pairs:
                k = key(a, b)
                if pen.mode == "prestart":
                    if R.start[b] >= R.start[a]:
                        ib = R.assign[b]
                        g[k] = g.get(k, 0.0) - (D[b][ib] - pt[b][ib]) / len(scen)
                    else:
                        g.setdefault(k, 0.0)
                else:
                    g[k] = g.get(k, 0.0) - (D[b][pen.ref[b]] - pen.cond_exp_along(b, R, D, P, R.start[a])) / len(scen)
        return total / len(scen), g

    cur, grad = objective_and_grad(coef)
    best_val, best = cur, dict(coef)
    if log:
        log.info("  fit2 it 0: mean Psi_hat = %.3f (%s)", cur, "warm start" if init else "c = 0")
    step = step0
    for it in range(1, iters + 1):
        gn = math.sqrt(sum(v * v for v in grad.values()))
        if gn < 1e-12:
            break
        improved = False
        for _ in range(4):
            new = {k: min(c_max, max(-c_max, coef.get(k, 0.0) + step * c_max * v / gn)) for k, v in grad.items()}
            val, g2 = objective_and_grad(new)
            if val > cur + 1e-9:
                coef, cur, grad = new, val, g2; improved = True
                break
            step *= 0.5
        if not improved:
            coef = {k: min(c_max, max(-c_max, coef.get(k, 0.0) + step * c_max * v / gn)) for k, v in grad.items()}
            cur, grad = objective_and_grad(coef)
            step = max(step, step0 / (1 + it))
        else:
            step *= 1.2
        if log:
            log.info("  fit2 it %d: mean Psi_hat = %.3f  step %.4f%s", it, cur, step, "" if improved else " (non-monotone)")
        if cur > best_val:
            best_val, best = cur, dict(coef)
    pen = build(best)
    pen.coef = best
    return pen


@dataclass
class Report2:
    n: int
    psi_mean: float
    psi_lcb: float
    pi_mean: float
    pi_lcb: float
    policy_mean: Dict[str, float]
    gap_mean: Dict[str, float]
    gap_ucb: Dict[str, float]
    min_pathwise: Dict[str, float]
    zeta_mean: Dict[str, float]


def evaluate_revealed(inst: Instance, pen: RevealedPenalty, policies: Dict[str, Callable[[Scenario], Realized]],
                      n_scen: int, base_seed: int, kappa: float = 1.0, cp_time: float = 10.0, threads: int = 1,
                      alpha: float = 0.05, log=None) -> Report2:
    from scipy import stats
    zero = RevealedPenalty.zero(inst, pen.t_bar)
    psis, pis = [], []
    vals = {k: [] for k in policies}; zetas = {k: [] for k in policies}
    for n in range(n_scen):
        D, P = sample_scenario_paths(inst, base_seed, n)
        r = penalized_bound_cpsat2(inst, D, P, pen, kappa, cp_time, threads)
        r0 = penalized_bound_cpsat2(inst, D, P, zero, kappa, cp_time, threads)
        psis.append(r.psi_hat); pis.append(r0.psi_hat)
        for k, pol in policies.items():
            R = pol(D)
            vals[k].append(R.makespan); zetas[k].append(pen.zeta(R, D, P))
        if log and (n + 1) % 10 == 0:
            log.info("  scenario %d/%d  Psi_hat=%.2f  PI=%.2f", n + 1, n_scen, r.psi_hat, r0.psi_hat)
    psis = np.array(psis); pis = np.array(pis)
    tq = stats.t.ppf(1 - alpha, n_scen - 1)
    rep = Report2(n_scen, float(psis.mean()), float(psis.mean() - tq * psis.std(ddof=1) / math.sqrt(n_scen)),
                  float(pis.mean()), float(pis.mean() - tq * pis.std(ddof=1) / math.sqrt(n_scen)), {}, {}, {}, {}, {})
    for k in policies:
        x = np.array(vals[k]); z = np.array(zetas[k]); d = x - z - psis
        rep.policy_mean[k] = float(x.mean()); rep.zeta_mean[k] = float(z.mean())
        rep.min_pathwise[k] = float(d.min()); rep.gap_mean[k] = float(d.mean())
        rep.gap_ucb[k] = float(d.mean() + tq * d.std(ddof=1) / math.sqrt(n_scen))
    return rep
