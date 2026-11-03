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
    """Revealed history available to a policy: completions, in-progress
    operations (machine, start time) and the scenario paths of started
    operations (failure/repair events observed so far)."""
    def __init__(self, inst: Instance, t: float):
        self.inst = inst; self.t = t
        self.comp: Dict[int, float] = {}
        self.in_progress: Dict[int, Tuple[int, float]] = {}
        self.machine_busy_until: List[float] = [0.0] * inst.n_machines
        self.paths: Optional[Paths] = None
        self.cache: Dict[int, tuple] = {}

    def expected_remaining(self, a: int) -> float:
        """E[C_a | F_t] - t for an in-progress operation (Assumption 1)."""
        i, s0 = self.in_progress[a]
        e = self.t - s0
        p = self.inst.op_proc[a][i]; lam, mu = self.inst.lam[i], self.inst.mu[i]
        if self.paths is not None and lam > 0:
            return max(0.0, conditional_expectation(p, lam, mu, self.paths[a][i], e) - e)
        return max(0.0, p * (1.0 + (lam / mu if mu > 0 else 0.0)) - e)


def reactive_execute(inst: Instance, policy: ReactivePolicy, D: Scenario,
                     assign_rule: str = "earliest_nominal") -> float:
    return reactive_execute_realized(inst, policy, D, assign_rule).makespan


def reactive_execute_realized(inst: Instance, policy: ReactivePolicy, D: Scenario,
                              assign_rule: str = "earliest_nominal", paths: Optional[Paths] = None) -> Realized:
    """Event-driven list scheduling. At each decision epoch, ready operations
    are considered in priority order; each is started on a *free* eligible
    machine of its cell (the one with the smallest nominal processing time,
    or the plan's machine if free and assign_rule == 'plan'). Decisions use
    only revealed information (completions), so the policy is in P_NA."""
    n = inst.n_ops
    state = ReactiveState(inst, 0.0); state.paths = paths
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
                state.in_progress.pop(a, None)
                if not hasattr(state, "last_assign"):
                    state.last_assign = {}
                state.last_assign[a] = i
                s = inst.op_succ[a]
                if s >= 0:
                    ready.add(s)
        # dispatch
        progress = True
        epoch_hook = getattr(policy, "on_epoch", None)
        while progress:
            progress = False
            if epoch_hook is not None:
                epoch_hook(state, ready, free)
            for a in sorted(ready, key=lambda a: policy.priority(a, state)):
                cands = [i for i in inst.op_proc[a] if free[i]]
                if not cands:
                    continue
                pa = getattr(policy, "plan_assign", None)
                pi_ = (pa[a] if (pa is not None and (isinstance(pa, dict) and a in pa or isinstance(pa, list))) else None)
                if assign_rule == "plan" and pi_ is not None and free[pi_]:
                    i = pi_
                else:
                    i = min(cands, key=lambda i: (inst.op_proc[a][i], i))
                free[i] = False
                ready.discard(a)
                assign[a] = i; start[a] = t
                state.in_progress[a] = (i, t)
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


# ---- re-planning policy (cell agents re-invoked during execution) ----------
def replan_policy(inst: Instance, tails: List[int], rule: str = "EST-MWR", replan_every: int = 1) -> ReactivePolicy:
    """At every decision epoch, re-plan the residual problem on the revealed
    state: completed operations are fixed, in-progress operations occupy
    their machines for their expected remaining time given the observed
    failure path, all other operations use expected durations ptilde.  The
    residual plan (serial list scheduling with heads/tails) gives the
    dispatch order (planned start, ties by remaining work) and the preferred
    machine (the plan's machine if free).  Decisions use only F_t, so the
    policy is in P_NA."""
    pt = expected_durations(inst)
    pmin_t = [min(pt[a].values()) for a in range(inst.n_ops)]
    # expected tails along the job (static)
    tail_e = [0.0] * inst.n_ops
    for ops in inst.job_ops:
        for k in range(len(ops) - 2, -1, -1):
            tail_e[ops[k]] = tail_e[ops[k + 1]] + pmin_t[ops[k + 1]]

    pol = ReactivePolicy(lambda a, st: st.cache.get(a, (float("inf"), 0.0, a)))
    pol.plan_assign = {}
    counter = [0]

    def on_epoch(state: ReactiveState, ready: set, free: List[bool]):
        t = state.t
        counter[0] += 1
        if replan_every > 1 and (counter[0] - 1) % replan_every != 0 and all(a in state.cache for a in ready):
            return
        # machine availability and job availability from the revealed state
        mfree = [t] * inst.n_machines
        job_avail = {}
        for a, (i, s0) in state.in_progress.items():
            fin = t + state.expected_remaining(a)
            mfree[i] = max(mfree[i], fin)
            s = inst.op_succ[a]
            if s >= 0:
                job_avail[s] = fin
        remaining = [a for a in range(inst.n_ops) if a not in state.comp and a not in state.in_progress]
        est = {}
        for a in remaining:
            if inst.op_pos[a] == 0:
                est[a] = max(t, float(inst.release[inst.op_job[a]]))
            else:
                p_ = inst.op_pred[a]
                est[a] = job_avail.get(a, t) if p_ in state.in_progress else (max(t, state.comp.get(p_, t)) if p_ in state.comp else None)
        # serial list scheduling over the residual problem
        planned = {}
        pending = set(remaining)
        while pending:
            cands = [a for a in pending if est[a] is not None]
            if not cands:
                break
            # earliest start - most work remaining
            def key(a):
                opts = [(max(est[a], mfree[i]) + pt[a][i], max(est[a], mfree[i]), i) for i in inst.op_proc[a]]
                fin, s_, i = min(opts)
                return (s_, -(pmin_t[a] + tail_e[a]), a, i, fin)
            s_, _, a, i, fin = min(key(a) for a in cands)
            planned[a] = (s_, i); mfree[i] = fin
            pending.discard(a)
            succ = inst.op_succ[a]
            if succ >= 0 and succ in pending:
                est[succ] = fin
        state.cache = {a: (s_, -(pmin_t[a] + tail_e[a]), a) for a, (s_, i) in planned.items()}
        pol.plan_assign = {a: i for a, (s_, i) in planned.items()}

    pol.on_epoch = on_epoch
    return pol


# ---- re-optimizing re-planning policy (planner's machinery during execution) ----
def replan_opt_policy(inst: Instance, plan: Schedule, cp_time: float = 1.0, threads: int = 1,
                      disruption_factor: float = 1.5, every: int = 0, kappa: float = 1.0) -> ReactivePolicy:
    """P_NA policy that re-optimizes the residual problem with CP-SAT.
    Triggers: t = 0; a completion whose realized duration exceeded its
    expectation by the factor `disruption_factor`; every `every`-th
    completion if every > 0.  Residual model: in-progress operations occupy
    their machine for E[remaining | observed path] (rounded up), remaining
    operations are free (expected durations ptilde rounded up, grid 1/kappa),
    job precedences and release dates; objective makespan; warm start from
    the current plan; time limit cp_time.  Between triggers the current
    plan's order and machines are followed (plan-order dispatch)."""
    from ortools.sat.python import cp_model
    pt = expected_durations(inst)
    pol = ReactivePolicy(lambda a, st: st.cache.get(a, (float("inf"), a)))
    pol.plan_assign = {a: plan.assign[a] for a in range(inst.n_ops)}
    cur_start = {a: float(plan.start[a]) for a in range(inst.n_ops)}
    state_box = {"completions": 0, "last_disrupted": True, "stats": {"replans": 0, "time": 0.0}}
    pol.stats = state_box["stats"]

    def reoptimize(state: ReactiveState):
        import time as _t
        t0 = _t.time(); t = state.t
        remaining = [a for a in range(inst.n_ops) if a not in state.comp and a not in state.in_progress]
        if not remaining:
            return
        mdl = cp_model.CpModel()
        H = int(math.ceil(kappa * (t + sum(max(pt[a].values()) for a in remaining) + 1)))
        for a, (i, s0) in state.in_progress.items():
            H += int(math.ceil(kappa * state.expected_remaining(a)))
        start, end, pres = {}, {}, {}
        ivs = {i: [] for i in range(inst.n_machines)}
        t_int = int(math.floor(kappa * t))
        cmax = mdl.NewIntVar(0, H, "cmax")
        fin_pred = {}
        for a, (i, s0) in state.in_progress.items():        # fixed occupation until expected completion
            rem = int(math.ceil(kappa * state.expected_remaining(a)))
            if rem > 0:
                ivs[i].append(mdl.NewIntervalVar(t_int, rem, t_int + rem, f"busy{a}"))
            succ = inst.op_succ[a]
            if succ >= 0:
                fin_pred[succ] = t_int + rem
        for a in remaining:
            lo = t_int
            if inst.op_pos[a] == 0:
                lo = max(lo, int(math.ceil(kappa * inst.release[inst.op_job[a]])))
            if a in fin_pred:
                lo = max(lo, fin_pred[a])
            s = mdl.NewIntVar(lo, H, f"s{a}"); e = mdl.NewIntVar(0, H, f"e{a}")
            start[a] = s; end[a] = e
            lits = []
            for i in inst.op_proc[a]:
                p = max(1, int(math.ceil(kappa * pt[a][i])))
                x = mdl.NewBoolVar(f"x{a}_{i}")
                ivs[i].append(mdl.NewOptionalIntervalVar(s, p, e, x, f"iv{a}_{i}")); lits.append(x); pres[(a, i)] = x
                mdl.AddHint(x, 1 if pol.plan_assign.get(a) == i else 0)
            mdl.AddExactlyOne(lits)
            mdl.AddHint(s, max(lo, int(kappa * cur_start.get(a, t))))
            if inst.op_succ[a] < 0:
                mdl.Add(cmax >= e)
        for a in remaining:
            p_ = inst.op_pred[a]
            if p_ >= 0 and p_ in start:
                mdl.Add(start[a] >= end[p_])
        for i in range(inst.n_machines):
            if len(ivs[i]) > 1:
                mdl.AddNoOverlap(ivs[i])
        mdl.Minimize(cmax)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = cp_time
        solver.parameters.num_workers = threads
        st = solver.Solve(mdl)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for a in remaining:
                cur_start[a] = solver.Value(start[a]) / kappa
                pol.plan_assign[a] = next(i for i in inst.op_proc[a] if solver.Value(pres[(a, i)]))
        state_box["stats"]["replans"] += 1
        state_box["stats"]["time"] += _t.time() - t0

    def on_epoch(state: ReactiveState, ready: set, free: List[bool]):
        # detect disruption since the last epoch: completions with D >= factor * ptilde
        n_comp = len(state.comp)
        trigger = state.t == 0.0 and state_box["last_disrupted"]
        if n_comp > state_box["completions"]:
            new = n_comp - state_box["completions"]
            state_box["completions"] = n_comp
            # realized durations of the newly completed operations are in state.comp - start; we approximate the
            # disruption test by comparing completion time with the planned completion (plan start + ptilde)
            for a, c in state.comp.items():
                if a in state_box.setdefault("seen", set()):
                    continue
                state_box["seen"].add(a)
                i = state.last_assign.get(a) if hasattr(state, "last_assign") else None
                planned_c = cur_start.get(a, 0.0) + (pt[a][i] if i is not None else min(pt[a].values()))
                if c > planned_c + (disruption_factor - 1.0) * min(pt[a].values()):
                    trigger = True
            if every > 0 and (n_comp // every) != ((n_comp - new) // every):
                trigger = True
        if trigger:
            reoptimize(state)
            state_box["last_disrupted"] = False
        state.cache = {a: (cur_start.get(a, float("inf")), a) for a in ready}

    pol.on_epoch = on_epoch
    return pol
