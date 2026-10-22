"""Stochastic certificates (Section 5.2 / 5.4).

mean_value_bound     Proposition 'Mean-value bound': certificate for static
                     policies, computed on the instance floor(p (1+lam/mu)).
pi_certificate       Theorem 'Perfect-information certificate': per scenario
                     omega, a lower bound Phi_omega on C*_max(D(omega)) from
                     the head-tail bound and/or the Lagrangian certificate at
                     given duals (rounded instance, Lemma EC.3); averages give
                     an unbiased estimate of E[Phi_omega] <= z^NA.
paired_gap           pathwise differences C^sigma(omega_n) - Phi_{omega_n} >= 0
                     and their confidence bounds.
Penalized relaxation (Section 5.4) is in penalty.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .bounds import head_tail_bounds, heads_tails
from .dispatch import best_dispatch
from .instances import Instance
from .oracles import CellProblem, ti_lp_cell, cpsat_cell
from .schedule import Schedule
from .simulator import Scenario, round_scenario, sample_scenario, static_execute, reactive_execute


def mean_value_bound(inst: Instance, cg_runner: Callable[[Instance], float]) -> float:
    """Certificate for z^stat: any lower bound on the instance with p~ rounded down."""
    tilde = inst.failure_inflated_rounded(round_down=True)
    return cg_runner(tilde)


@dataclass
class ScenarioBound:
    phi: float          # certified lower bound on C*_max(D(omega)), original time scale
    gamma_ht: float
    phi_lagr: float
    ub_ref: float       # realized makespan of the reference dispatching policy


def scenario_bound(inst: Instance, D: Scenario, kappa: float = 1.0, duals: Optional[dict] = None,
                   cp_time: float = 0.0, lp_cache: Optional[dict] = None, threads: int = 1,
                   ht_cp_time: float = 0.0) -> ScenarioBound:
    """Lower bound on the perfect-information optimum for scenario D.

    Steps: round to integers with factor kappa (Lemma EC.3); head-tail bound
    (energetic + preemptive, optional CP-SAT); Lagrangian certificate at the
    supplied duals (rho over ops, pi over transitions, eta) using the TI-LP
    oracle and optional short CP-SAT; return max / kappa."""
    sinst = round_scenario(inst, D, kappa)
    h, q = heads_tails(sinst)
    ref, _, _ = best_dispatch(sinst, q)
    T = ref.makespan
    ht = head_tail_bounds(sinst, T, cp_time=ht_cp_time, threads=threads, use_cpsat=ht_cp_time > 0)
    gamma = ht["gamma_ht"]
    phi_l = -math.inf
    if duals is not None:
        rho, pi, eta = duals["rho"], duals["pi"], duals.get("eta", 0.0)
        tot = sum(rho) + eta
        rho = [r / tot for r in rho]; eta = eta / tot
        w = {a: rho[a] + (pi.get(a, 0.0) if sinst.is_transition[a] else 0.0) for a in range(sinst.n_ops)}
        v = {a: (pi.get(sinst.op_pred[a], 0.0) if (sinst.op_pred[a] >= 0 and sinst.is_transition[sinst.op_pred[a]]) else 0.0)
             for a in range(sinst.n_ops)}
        phi_l = sum(rho[a] * q[a] for a in range(sinst.n_ops)) + eta * gamma
        for c in range(sinst.n_cells):
            cp = CellProblem(sinst, c, h, q, T)
            if not cp.ops:
                continue
            wc = {a: w[a] for a in cp.ops}; vc = {a: v[a] for a in cp.ops}
            beta = ti_lp_cell(cp, wc, vc, None).bound
            if cp_time > 0:
                r = cpsat_cell(cp, wc, vc, cp_time, threads, lower_bound=beta)
                beta = max(beta, r.bound)
            phi_l += beta
    phi = max(float(gamma), phi_l)
    return ScenarioBound(phi / kappa, gamma / kappa, (phi_l / kappa) if phi_l > -math.inf else -math.inf,
                         T / kappa)


@dataclass
class StochasticReport:
    n: int
    phi_mean: float
    phi_lcb: float            # lower confidence bound on E[Phi] (<= z^NA)
    policy_mean: Dict[str, float]
    policy_cvar95: Dict[str, float]
    gap_mean: Dict[str, float]
    gap_ucb: Dict[str, float]  # upper confidence bound on E[C^sigma] - z^NA
    min_pathwise_gap: Dict[str, float]


def evaluate(inst: Instance, policies: Dict[str, Callable[[Scenario], float]], n_scen: int, base_seed: int,
             kappa: float = 1.0, duals: Optional[dict] = None, alpha: float = 0.05, cp_time: float = 0.0,
             threads: int = 1) -> StochasticReport:
    from scipy import stats
    phis = []; vals = {k: [] for k in policies}
    for n in range(n_scen):
        D = sample_scenario(inst, base_seed, n)
        sb = scenario_bound(inst, D, kappa, duals, cp_time=cp_time, threads=threads)
        phis.append(sb.phi)
        for k, pol in policies.items():
            vals[k].append(pol(D))
    phis = np.array(phis)
    tq = stats.t.ppf(1 - alpha, n_scen - 1)
    rep = StochasticReport(n_scen, float(phis.mean()), float(phis.mean() - tq * phis.std(ddof=1) / math.sqrt(n_scen)),
                           {}, {}, {}, {}, {})
    for k in policies:
        x = np.array(vals[k]); d = x - phis
        assert d.min() >= -1e-6, f"pathwise gap negative for {k}: {d.min()} (bound invalid?)"
        rep.policy_mean[k] = float(x.mean())
        srt = np.sort(x); m = max(1, int(math.ceil(0.05 * n_scen)))
        rep.policy_cvar95[k] = float(srt[-m:].mean())
        rep.gap_mean[k] = float(d.mean())
        rep.gap_ucb[k] = float(d.mean() + tq * d.std(ddof=1) / math.sqrt(n_scen))
        rep.min_pathwise_gap[k] = float(d.min())
    return rep
