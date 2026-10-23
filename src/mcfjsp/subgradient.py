"""Projected supergradient baseline for the Lagrangian dual (Appendix EC.4).

Multipliers x = (pi, rho) in R_+^{|T|} x Delta (simplex over operations).
Step alpha_t = a / (b + t). Certificate at every iterate by Theorem 1:
    Phi = sum_a rho_a q_a + sum_c beta_c(pi, rho).
The oracle solutions (CP-SAT exact or TI-LP fractional) provide the
supergradient g_pi,a = C_a - s_{a+}, g_rho,a = C_a + q_a.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .bounds import head_tail_bounds, heads_tails
from .dispatch import best_dispatch
from .instances import Instance
from .oracles import CellProblem, cpsat_cell, ti_lp_cell

log = logging.getLogger("mcfjsp.subgradient")


def project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    tau = (css[k] - 1) / (k + 1)
    return np.maximum(v - tau, 0.0)


@dataclass
class SGResult:
    lb: float
    lb_lagr: float
    gamma_ht: int
    ub: int
    iterations: int
    time: float
    history: List[dict]


def subgradient(inst: Instance, time_limit: float = 300.0, max_iter: int = 500, a: float = 1.0, b: float = 10.0,
                oracle: str = "lp", cp_time: float = 3.0, threads: int = 1, ht_cp_time: float = 5.0,
                rule: str = "divergent", lp_max_vars: int = 30_000) -> SGResult:
    t0 = time.time()
    deadline = t0 + time_limit
    h, q = heads_tails(inst)
    sch, _, _ = best_dispatch(inst, q)
    UB = sch.makespan; T = UB
    ht = head_tail_bounds(inst, UB, cp_time=ht_cp_time, threads=threads, use_preemptive=inst.n_ops <= 400,
                          time_budget=0.15 * time_limit)
    gamma_ht = ht["gamma_ht"]
    n = inst.n_ops; trans = inst.transitions; tidx = {a: k for k, a in enumerate(trans)}
    pi = np.zeros(len(trans)); rho = np.ones(n) / n
    best = -math.inf; history = []
    cache: Dict = {}
    cps = [CellProblem(inst, c, h, q, T) for c in range(inst.n_cells)]
    step_param = a; nonimprove = 0
    for t in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        w = {aa: rho[aa] + (pi[tidx[aa]] if inst.is_transition[aa] else 0.0) for aa in range(n)}
        v = {aa: (pi[tidx[inst.op_pred[aa]]] if (inst.op_pred[aa] >= 0 and inst.is_transition[inst.op_pred[aa]]) else 0.0)
             for aa in range(n)}
        phi = float(np.dot(rho, q)); C = {}; S = {}
        aborted = False
        for cp in cps:
            if not cp.ops:
                continue
            if time.time() > deadline:
                aborted = True; break
            wc = {aa: w[aa] for aa in cp.ops}; vc = {aa: v[aa] for aa in cp.ops}
            lp_time = max(1.0, deadline - time.time())
            if oracle == "lp":
                try:
                    r = ti_lp_cell(cp, wc, vc, cache, max_vars=lp_max_vars, time_limit=lp_time)
                except Exception:
                    aborted = True; break
                phi += r.bound
                C.update(r.frac_comp); S.update(r.frac_start)
            else:
                rl = ti_lp_cell(cp, wc, vc, cache, max_vars=lp_max_vars, time_limit=lp_time)
                r = cpsat_cell(cp, wc, vc, cp_time, threads, lower_bound=rl.bound)
                phi += max(r.bound, rl.bound)
                if r.column is not None:
                    C.update(r.column.comp); S.update(r.column.start)
                else:
                    C.update(rl.frac_comp); S.update(rl.frac_start)
        if aborted:
            break
        if phi > best + 1e-9:
            best = phi; nonimprove = 0
        else:
            nonimprove += 1
        g_pi = np.array([C[aa] - S[inst.op_succ[aa]] for aa in trans])
        g_rho = np.array([C[aa] + q[aa] for aa in range(n)])
        if rule == "divergent":
            # a/(b+t) scaled by 1/||g||: still a divergent, square-summable-free series since ||g|| is bounded
            gn = math.sqrt(float(np.dot(g_pi, g_pi) + np.dot(g_rho, g_rho)))
            alpha = (a / (b + t)) * max(UB - max(best, gamma_ht), 1.0) / max(gn, 1e-9)
        else:  # Held-Wolfe-Crowder style: step = param * (UB - phi) / ||g||^2, halve after 5 non-improving
            if nonimprove >= 5:
                step_param /= 2; nonimprove = 0
            gn = float(np.dot(g_pi, g_pi) + np.dot(g_rho, g_rho))
            alpha = step_param * max(UB - phi, 1e-3) / max(gn, 1e-9)
        pi = np.maximum(pi + alpha * g_pi, 0.0)
        rho = project_simplex(rho + alpha * g_rho)
        history.append(dict(it=t, t=time.time() - t0, phi=phi, best=best, alpha=alpha))
        if t % 10 == 0:
            log.info("sg it %3d  t=%6.1fs  Phi=%.3f  best=%.3f  LB=%.3f", t, time.time() - t0, phi, best, max(best, gamma_ht))
    return SGResult(max(best, gamma_ht), best, gamma_ht, UB, len(history), time.time() - t0, history)
