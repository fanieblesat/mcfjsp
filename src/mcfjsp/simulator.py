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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .instances import Instance
from .schedule import Schedule, semi_active


Scenario = Dict[int, Dict[int, float]]   # a -> {i -> D_{a,i}}


Path = Tuple[Tuple[float, ...], Tuple[float, ...]]   # (failure epochs in work time, sorted; repair durations)
Paths = Dict[int, Dict[int, Path]]


def sample_scenario_paths(inst: Instance, base_seed: int, scen_index: int) -> Tuple[Scenario, Paths]:
    """Scenario with the failure path of every (a, i): N ~ Poisson(lam p)
    failures at work-time epochs uniform on [0, p), repairs Exp(mu).
    D_{a,i} = p + sum of repairs.  The path is what a policy can observe while
    a runs on i (failure and repair events), and what the conditional
    expectations of the revealed-duration penalties need."""
    D: Scenario = {}; P: Paths = {}
    for a in range(inst.n_ops):
        D[a] = {}; P[a] = {}
        for i, p in inst.op_proc[a].items():
            lam, mu = inst.lam[i], inst.mu[i]
            if lam <= 0:
                D[a][i] = float(p); P[a][i] = ((), ())
                continue
            rng = np.random.default_rng([base_seed, scen_index, a, i])
            n = int(rng.poisson(lam * p))
            if n == 0:
                D[a][i] = float(p); P[a][i] = ((), ())
                continue
            epochs = tuple(sorted(float(x) for x in rng.uniform(0.0, p, n)))
            reps = tuple(float(x) for x in rng.exponential(1.0 / mu, n))
            D[a][i] = p + sum(reps); P[a][i] = (epochs, reps)
    return D, P


def sample_scenario(inst: Instance, base_seed: int, scen_index: int) -> Scenario:
    return sample_scenario_paths(inst, base_seed, scen_index)[0]


def conditional_expectation(p: float, lam: float, mu: float, path: Path, e: float) -> float:
    """E[D | F_e]: expected total occupation time of an operation with nominal
    duration p on a machine with failure rate lam and repair rate mu, given
    the failure/repair path observed during the first e time units of its
    processing (Assumption 1: busy-time failures, Exp repairs, memoryless).
    For e >= D the value is D itself."""
    epochs, reps = path
    inflate = 1.0 + (lam / mu if mu > 0 else 0.0)
    t = 0.0; w = 0.0           # elapsed time, work done
    for wk, rk in zip(epochs, reps):
        run = wk - w            # running segment until the next failure
        if e < t + run:
            w_e = w + (e - t)
            return e + (p - w_e) * inflate
        t += run; w = wk
        if e < t + rk:          # under repair at time e
            return e + (1.0 / mu) + (p - w) * inflate
        t += rk
    run = p - w
    if e < t + run:
        w_e = w + (e - t)
        return e + (p - w_e) * inflate
    return t + run              # completed: D


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
class Realized:
    """Realized execution of a policy under a scenario: machine, start and
    completion of every operation (floats), and the makespan.  This is the
    schedule S^sigma(omega) of Theorem 'PI' / Lemma 'Zero mean'."""
    __slots__ = ("assign", "start", "comp", "makespan")

    def __init__(self, assign, start, comp):
        self.assign = list(assign); self.start = list(start); self.comp = list(comp)
        self.makespan = float(max(comp)) if comp else 0.0


def static_execute_realized(inst: Instance, plan: Schedule, D: Scenario) -> Realized:
    orders = plan.machine_orders(inst)
    sch = semi_active(inst, plan.assign, orders, durations=D)  # floats are fine here
    return Realized(sch.assign, sch.start, sch.comp)


def static_execute(inst: Instance, plan: Schedule, D: Scenario) -> float:
    """Semi-active execution of a plan's assignment and machine orders with
    realized durations. Returns the realized makespan."""
    return static_execute_realized(inst, plan, D).makespan


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
    return reactive_execute_realized(inst, policy, D, assign_rule).makespan


def reactive_execute_realized(inst: Instance, policy: ReactivePolicy, D: Scenario,
                              assign_rule: str = "earliest_nominal") -> Realized:
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
    assign = [-1] * n; start = [0.0] * n; comp = [0.0] * n
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
                state.comp[a] = t; free[i] = True; done += 1; makespan = max(makespan, t); comp[a] = t
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
                assign[a] = i; start[a] = t
                heapq.heappush(events, (t + D[a][i], 1, a, i))
                progress = True
                break
    return Realized(assign, start, comp)


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
