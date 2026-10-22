"""Scenario sampling and execution under machine failures (Assumption 1).

Failures occur only while a machine processes an operation, at rate lam_i per
unit of busy time; repairs are Exp(mu_i); preempt-resume on the same machine.
Hence the occupation time of operation a on machine i is
    D_{a,i} = p_{ia} + sum_{l=1}^{N} R_l,  N ~ Poisson(lam_i p_{ia}),  R_l ~ Exp(mu_i),
independent across (a, i) pairs, and independent of start times.

A *scenario* fixes D_{a,i} for every operation and every eligible machine.
Each (a, i) pair has its own random stream, seeded by (base_seed, a, i), so
that common random numbers hold across policies at the level of individual
operations.

Execution:
  static_execute   fixed assignment + machine orders, semi-active (P_stat)
  reactive_execute event-driven list scheduling with a priority policy that
                   sees only the history (P_NA): durations are revealed at
                   completion.
"""
from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional

import numpy as np

from .instances import Instance
from .schedule import Schedule, semi_active


Scenario = Dict[int, Dict[int, float]]   # a -> {i -> D_{a,i}}


def sample_scenario(inst: Instance, base_seed: int, scen_index: int) -> Scenario:
    D: Scenario = {}
    for a in range(inst.n_ops):
        D[a] = {}
        for i, p in inst.op_proc[a].items():
            lam, mu = inst.lam[i], inst.mu[i]
            if lam <= 0:
                D[a][i] = float(p)
                continue
            rng = np.random.default_rng([base_seed, scen_index, a, i])
            n = rng.poisson(lam * p)
            rep = float(rng.gamma(n, 1.0 / mu)) if n > 0 else 0.0
            D[a][i] = p + rep
    return D


def expected_durations(inst: Instance) -> Dict[int, Dict[int, float]]:
    return {a: {i: p * (1.0 + inst.lam[i] / inst.mu[i]) for i, p in inst.op_proc[a].items()}
            for a in range(inst.n_ops)}


def round_scenario(inst: Instance, D: Scenario, kappa: float = 1.0) -> Instance:
    """Deterministic integer instance with durations floor(kappa D) and
    release floor(kappa r) (Lemma EC.3): any bound on it, divided by kappa,
    is a lower bound on C*_max(D)."""
    dur = {a: {i: max(1, int(math.floor(kappa * d))) for i, d in D[a].items()} for a in D}
    rel = [int(math.floor(kappa * r)) for r in inst.release]
    return inst.with_durations(dur, rel, name_suffix=f"_k{kappa}")


# ---- static execution --------------------------------------------------
def static_execute(inst: Instance, plan: Schedule, D: Scenario) -> float:
    """Semi-active execution of a plan's assignment and machine orders with
    realized durations. Returns the realized makespan."""
    orders = plan.machine_orders(inst)
    sch = semi_active(inst, plan.assign, orders, durations=D)  # floats are fine here
    return float(max(sch.comp))


# ---- reactive execution ------------------------------------------------
class ReactivePolicy:
    """priority(a, state) -> sort key; smaller is scheduled first.
    `state` exposes: t (time), ready ops, machine free flags, and plan info."""

    def __init__(self, priority: Callable[[int, "ReactiveState"], tuple]):
        self.priority = priority


class ReactiveState:
    def __init__(self, inst: Instance, t: float):
        self.inst = inst; self.t = t
        self.comp: Dict[int, float] = {}
        self.machine_busy_until: List[float] = [0.0] * inst.n_machines


def reactive_execute(inst: Instance, policy: ReactivePolicy, D: Scenario,
                     assign_rule: str = "earliest_nominal") -> float:
    """Event-driven list scheduling. At each decision epoch, ready operations
    are considered in priority order; each is started on a *free* eligible
    machine of its cell (the one with the smallest nominal processing time,
    or the plan's machine if free and assign_rule == 'plan'). Decisions use
    only revealed information (completions), so the policy is in P_NA."""
    n = inst.n_ops
    state = ReactiveState(inst, 0.0)
    free = [True] * inst.n_machines
    ready_time = {}  # op -> earliest time it can start (release / predecessor completion)
    for a in range(n):
        if inst.op_pos[a] == 0:
            ready_time[a] = float(inst.release[inst.op_job[a]])
    events = []  # (time, kind, payload); kind 0=release/ready, 1=completion
    for a, t in ready_time.items():
        heapq.heappush(events, (t, 0, a, -1))
    done = 0; makespan = 0.0
    ready = set()
    while done < n:
        if not events:
            raise RuntimeError("deadlock in reactive execution")
        t = events[0][0]
        state.t = t
        while events and events[0][0] <= t + 1e-12:
            _, kind, a, i = heapq.heappop(events)
            if kind == 0:
                ready.add(a)
            else:
                state.comp[a] = t; free[i] = True; done += 1; makespan = max(makespan, t)
                s = inst.op_succ[a]
                if s >= 0:
                    ready.add(s)
        # dispatch
        progress = True
        while progress:
            progress = False
            for a in sorted(ready, key=lambda a: policy.priority(a, state)):
                cands = [i for i in inst.op_proc[a] if free[i]]
                if not cands:
                    continue
                if assign_rule == "plan" and getattr(policy, "plan_assign", None) is not None \
                        and free[policy.plan_assign[a]]:
                    i = policy.plan_assign[a]
                else:
                    i = min(cands, key=lambda i: (inst.op_proc[a][i], i))
                free[i] = False
                ready.discard(a)
                heapq.heappush(events, (t + D[a][i], 1, a, i))
                progress = True
                break
    return makespan


def plan_order_policy(inst: Instance, plan: Schedule) -> ReactivePolicy:
    """Follow the plan's start-time order, reassigning to free machines."""
    pol = ReactivePolicy(lambda a, st: (plan.start[a], a))
    pol.plan_assign = plan.assign
    return pol


def rule_policy(inst: Instance, tails: List[int], name: str = "MWR") -> ReactivePolicy:
    pmin = inst.p_min
    if name == "MWR":
        return ReactivePolicy(lambda a, st: (-(pmin[a] + tails[a]), a))
    if name == "SPT":
        return ReactivePolicy(lambda a, st: (pmin[a], a))
    if name == "FIFO":
        return ReactivePolicy(lambda a, st: (inst.op_job[a], inst.op_pos[a]))
    raise ValueError(name)
