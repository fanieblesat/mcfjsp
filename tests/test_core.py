"""Correctness tests: every bound is checked against CP-SAT optima on small
instances; every schedule is checked for feasibility; stochastic certificates
are checked pathwise."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcfjsp.instances import Instance, read_fjsp_text  # noqa: E402
from mcfjsp.partition import multicell_variant  # noqa: E402
from mcfjsp.generator import generate  # noqa: E402
from mcfjsp.bounds import heads_tails, head_tail_bounds  # noqa: E402
from mcfjsp.dispatch import best_dispatch, serial_sgs  # noqa: E402
from mcfjsp.schedule import check_feasible  # noqa: E402
from mcfjsp.oracles import CellProblem, cpsat_cell, ti_lp_cell, compact_ti_lp  # noqa: E402
from mcfjsp.cg import ColumnGeneration, CGParams  # noqa: E402
from mcfjsp.baselines import cpsat_plant  # noqa: E402
from mcfjsp.simulator import sample_scenario, static_execute, reactive_execute, plan_order_policy, rule_policy, round_scenario  # noqa: E402
from mcfjsp.stochastic import scenario_bound, evaluate  # noqa: E402
from mcfjsp.subgradient import subgradient  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks")


def small_instances():
    out = []
    for seed in range(4):
        out.append(generate(n_cells=3, machines_per_cell=(1, 2), n_jobs=5, ops_per_job=(2, 4), proc_range=(1, 9),
                            flexibility=0.6, revisit_prob=0.3, release_spread=0.3, seed=seed))
    base = read_fjsp_text(os.path.join(DATA, "brandimarte", "mk01.txt"), "mk01")
    out.append(multicell_variant(base, 3))
    return out


@pytest.mark.parametrize("inst", small_instances(), ids=lambda i: i.name)
def test_bounds_below_optimum_and_schedules_feasible(inst):
    h, q = heads_tails(inst)
    sch, _, _ = best_dispatch(inst, q)
    check_feasible(inst, sch)
    opt = cpsat_plant(inst, sch.makespan, time_limit=30, threads=2)
    assert opt.status == "OPTIMAL"
    z = opt.ub
    # windows lemma: every feasible schedule satisfies s >= h and C + q <= Cmax
    for a in range(inst.n_ops):
        assert sch.start[a] >= h[a] and sch.comp[a] + q[a] <= sch.makespan
    ht = head_tail_bounds(inst, sch.makespan, cp_time=5, threads=2)
    for pc in ht["cells"]:
        assert pc["energetic"] <= pc["preemptive"] <= z and pc["cpsat"] <= z
    z_ti, st = compact_ti_lp(inst, h, q, sch.makespan)
    assert st == "OPTIMAL" and z_ti <= z + 1e-6
    cg = ColumnGeneration(inst, CGParams(time_limit=120, cp_time=5, cp_threads=2, lns_time=5, exact_every=2,
                                         stop_on_gap=False))
    r = cg.run()
    check_feasible(inst, r.schedule)
    assert r.ub >= z and r.lb <= z + 1e-6
    for rec in r.history:   # anytime certificate valid at every iteration
        assert rec["phi"] <= z + 1e-6 and rec["lb"] <= z + 1e-6
    assert z_ti <= r.lb_cg + 1e-6 or r.status != "lp_optimal"   # hierarchy z_TI <= z_LD when CG converged
    sg = subgradient(inst, time_limit=10, max_iter=30, oracle="lp", threads=2, ht_cp_time=1)
    assert sg.lb <= z + 1e-6


def test_cell_oracles_consistent():
    inst = small_instances()[0]
    h, q = heads_tails(inst)
    sch, _, _ = best_dispatch(inst, q)
    rng = np.random.default_rng(0)
    for c in range(inst.n_cells):
        cp = CellProblem(inst, c, h, q, sch.makespan)
        if not cp.ops:
            continue
        w = {a: float(rng.random()) for a in cp.ops}
        v = {a: float(rng.random()) * 0.5 for a in cp.ops}
        r_lp = ti_lp_cell(cp, w, v, {})
        r_cp = cpsat_cell(cp, w, v, time_limit=20, threads=2, lower_bound=r_lp.bound)
        cp.check_column(r_cp.column)
        assert r_cp.proven, r_cp.status
        assert r_lp.bound <= r_cp.value + 1e-6
        assert r_cp.bound <= r_cp.value + 1e-6
        # bound correction makes the CP-SAT bound valid for exact weights: value of its own column
        assert abs(r_cp.value - r_cp.bound) < 1e-3


def test_repair_always_feasible():
    inst = small_instances()[1]
    rng = np.random.default_rng(1)
    for _ in range(20):
        pri = rng.random(inst.n_ops)
        check_feasible(inst, serial_sgs(inst, pri))


def test_simulator_and_pi_certificate():
    inst = generate(n_cells=3, machines_per_cell=(1, 2), n_jobs=6, ops_per_job=(2, 4), availability=0.85,
                    mttr_factor=1.5, seed=7)
    h, q = heads_tails(inst)
    plan, _, _ = best_dispatch(inst, q)
    # deterministic reproduction and CRN
    assert sample_scenario(inst, 3, 1) == sample_scenario(inst, 3, 1)
    inst0 = generate(n_cells=3, machines_per_cell=(1, 2), n_jobs=6, ops_per_job=(2, 4), seed=7)
    assert static_execute(inst0, plan, sample_scenario(inst0, 3, 1)) == plan.makespan
    pols = {"static": lambda D: static_execute(inst, plan, D),
            "reactive-plan": lambda D: reactive_execute(inst, plan_order_policy(inst, plan), D, "plan"),
            "reactive-MWR": lambda D: reactive_execute(inst, rule_policy(inst, q, "MWR"), D)}
    rep = evaluate(inst, pols, n_scen=30, base_seed=5, kappa=4.0)
    for k in pols:
        assert rep.min_pathwise_gap[k] >= -1e-6     # Theorem: C^sigma(omega) >= Phi_omega
    # exact check of one scenario: PI optimum via CP-SAT on the rounded instance
    D = sample_scenario(inst, 5, 0)
    sinst = round_scenario(inst, D, 4.0)
    hh, qq = heads_tails(sinst)
    ref, _, _ = best_dispatch(sinst, qq)
    opt = cpsat_plant(sinst, ref.makespan, time_limit=30, threads=2)
    sb = scenario_bound(inst, D, 4.0)
    assert sb.phi <= opt.ub / 4.0 + 1e-9


def test_counterexample_prop3():
    """Prop. 'mean-value bound fails': 2 identical machines, 3 unit jobs,
    lam=mu=1; reactive policy E[Cmax] ~ 3.79 < inflated-time optimum 4."""
    rng = np.random.default_rng(0)
    N = 200_000
    def D():
        n = rng.poisson(1.0, N)
        return 1.0 + np.where(n > 0, rng.gamma(np.maximum(n, 1), 1.0), 0.0)
    d1, d2, d3 = D(), D(), D()
    adaptive = np.maximum(np.maximum(d1, d2), np.minimum(d1, d2) + d3)
    assert adaptive.mean() < 4.0 - 0.1
    static = np.maximum(d1 + d3, d2)
    assert static.mean() >= 4.0
