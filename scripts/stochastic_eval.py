#!/usr/bin/env python3
"""Stochastic evaluation: policies vs. perfect-information certificate.

For an instance with failures: plan with CG on the rounded mean-value
instance (p~), then evaluate on N scenarios (common random numbers):
  static            semi-active execution of the plan           (P_stat)
  reactive-plan     plan order, greedy reassignment on failures (P_NA)
  reactive-MWR      most-work-remaining dispatching             (P_NA)
Certificate per scenario: max(head-tail, Lagrangian at the CG's final duals)
on the rounded scenario instance (kappa). Reports E[C], CVaR95, E[Phi],
lower confidence bound on z^NA, paired gap with upper confidence bound.

Usage: python scripts/stochastic_eval.py --gen n_cells=5,n_jobs=40,availability=0.9,mttr_factor=2,seed=1 \
         --scenarios 100 --kappa 2 --time 120 --threads 2 2>&1 | tee logs/stoch_<date>.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mcfjsp.generator import generate  # noqa: E402
from mcfjsp.instances import Instance  # noqa: E402
from mcfjsp.bounds import heads_tails  # noqa: E402
from mcfjsp.cg import CGParams, ColumnGeneration  # noqa: E402
from mcfjsp.simulator import static_execute, reactive_execute, plan_order_policy, rule_policy  # noqa: E402
from mcfjsp.stochastic import evaluate  # noqa: E402


def parse_kv(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="n_cells=5,n_jobs=40,availability=0.9,mttr_factor=2,seed=1")
    ap.add_argument("--instance", default=None, help="JSON instance file (overrides --gen)")
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cp-time", type=float, default=0.0, help="CP-SAT seconds per cell per scenario (0 = LP only)")
    ap.add_argument("--out", default="results/stochastic")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    log = logging.getLogger("stoch")
    inst = Instance.load(args.instance) if args.instance else generate(**parse_kv(args.gen))
    assert max(inst.lam) > 0, "instance has no failures"
    log.info("instance %s: ops %d cells %d, availability %.3f", inst.name, inst.n_ops, inst.n_cells,
             inst.mu[0] / (inst.lam[0] + inst.mu[0]))
    tilde = inst.failure_inflated_rounded(round_down=False, kappa=1)   # planning instance (heuristic)
    cg = ColumnGeneration(tilde, CGParams(time_limit=args.time, cp_threads=args.threads, cp_time=5, lns_time=30))
    r = cg.run()
    plan = r.schedule
    log.info("plan on inflated instance: makespan %d, certificate %.2f", plan.makespan, r.lb)
    # duals from the CG run (for the Lagrangian part of the scenario certificate)
    gamma, rho, pi, eta = cg._duals() if cg.lp.status == "OPTIMAL" else (None, None, None, None)
    duals = {"rho": rho, "pi": pi, "eta": eta} if rho is not None else None
    h, q = heads_tails(inst)
    pols = {"static": lambda D: static_execute(inst, plan, D),
            "reactive-plan": lambda D: reactive_execute(inst, plan_order_policy(inst, plan), D, "plan"),
            "reactive-MWR": lambda D: reactive_execute(inst, rule_policy(inst, q, "MWR"), D)}
    t = time.time()
    rep = evaluate(inst, pols, args.scenarios, args.seed, kappa=args.kappa, duals=duals, cp_time=args.cp_time,
                   threads=args.threads)
    log.info("N=%d scenarios in %.1fs: E[Phi]=%.2f  LCB(z^NA)=%.2f", rep.n, time.time() - t, rep.phi_mean, rep.phi_lcb)
    for k in pols:
        log.info("  %-14s E[C]=%.2f  CVaR95=%.2f  gap=%.2f (UCB %.2f)  min pathwise gap %.2f", k, rep.policy_mean[k],
                 rep.policy_cvar95[k], rep.gap_mean[k], rep.gap_ucb[k], rep.min_pathwise_gap[k])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(dict(instance=inst.name, plan_makespan=plan.makespan, plan_certificate=r.lb, **rep.__dict__), f, indent=1)


if __name__ == "__main__":
    main()
