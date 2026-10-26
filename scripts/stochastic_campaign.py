#!/usr/bin/env python3
"""Stochastic campaign (Section 5.2 / 5.4): certified gaps of policies under
machine failures.

Per instance:
  plan          CG + LNS on the inflated instance p~ (the planning heuristic)
  z^stat bound  mean-value certificate for static policies: head-tail +
                destructive bound on the instance floor(p~)  (Prop. 'MV')
  policies      static (P_stat); reactive-plan, reactive-MWR, reactive-SPT (P_NA)
  per scenario  PI bound by head-tail (always), by CP-SAT (<= --cp-max-ops),
                penalized bound with own-duration penalties fitted on separate
                training scenarios (<= --cp-max-ops)                (Thm 'PI', Thm 'penalized')
  report        E[C], CVaR95, E[Phi] and its LCB for each bound, paired gaps
                with UCB, value of information proxies

Usage:
  python scripts/stochastic_campaign.py --gen n_cells=5,n_jobs=20,flexibility=0.8,availability=0.9,mttr_factor=2,seed=1 \
      --scenarios 200 --train 40 --kappa 2 --time 120 --threads 6 --cp-time 10 --out results/stoch_X 2>&1 | tee logs/stoch_X.log
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mcfjsp.generator import generate, generate_coupled  # noqa: E402
from mcfjsp.instances import Instance  # noqa: E402
from mcfjsp.bounds import heads_tails, head_tail_bounds  # noqa: E402
from mcfjsp.dispatch import best_dispatch  # noqa: E402
from mcfjsp.cg import CGParams, ColumnGeneration  # noqa: E402
from mcfjsp.propagation import destructive_bound  # noqa: E402
from mcfjsp.simulator import (static_execute_realized, reactive_execute_realized, plan_order_policy, rule_policy,  # noqa: E402
                              sample_scenario, round_scenario)
from mcfjsp.stochastic import scenario_bound  # noqa: E402
from mcfjsp.penalty import Penalty, fit_penalty, penalized_bound_cpsat  # noqa: E402
from mcfjsp.baselines import cpsat_plant  # noqa: E402

RANGE_KEYS = {"machines_per_cell", "ops_per_job", "visits_per_job", "ops_per_visit", "proc_range"}


def parse_kv(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        if k in RANGE_KEYS:
            parts = [int(x) for x in v.split("-")]
            out[k] = (parts[0], parts[-1]); continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def lcb(x, alpha=0.05):
    from scipy import stats
    x = np.asarray(x); n = len(x)
    return float(x.mean() - stats.t.ppf(1 - alpha, n - 1) * x.std(ddof=1) / math.sqrt(n))


def ucb(x, alpha=0.05):
    return -lcb(-np.asarray(x), alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--coupled", nargs="*", default=[])
    ap.add_argument("--instances", nargs="*", default=[])
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--train", type=int, default=40, help="training scenarios for the penalty (separate seed)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--time", type=float, default=120.0, help="planning time (CG + LNS)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cp-time", type=float, default=10.0, help="CP-SAT seconds per scenario (PI and penalized)")
    ap.add_argument("--cp-max-ops", type=int, default=400, help="CP-SAT scenario bounds only up to this size")
    ap.add_argument("--penalty-param", default="op", choices=["scalar", "cell", "op"])
    ap.add_argument("--penalty-iters", type=int, default=12)
    ap.add_argument("--shave-time", type=float, default=120.0)
    ap.add_argument("--out", default="results/stochastic_campaign")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("stoch")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    insts = [generate(**parse_kv(g)) for g in args.gen] + [generate_coupled(**parse_kv(g)) for g in args.coupled]
    insts += [Instance.load(f) for f in args.instances]
    rows = []
    for inst in insts:
        assert max(inst.lam) > 0, f"{inst.name}: no failures"
        avail = inst.mu[0] / (inst.lam[0] + inst.mu[0])
        log.info("=== %s: ops %d cells %d machines %d  availability %.3f  flexibility %.2f", inst.name, inst.n_ops,
                 inst.n_cells, inst.n_machines, avail, sum(len(d) for d in inst.op_proc) / inst.n_ops)
        os.makedirs("data/generated", exist_ok=True); inst.save(os.path.join("data", "generated", inst.name + ".json"))
        row = dict(instance=inst.name, ops=inst.n_ops, cells=inst.n_cells, machines=inst.n_machines, availability=round(avail, 3),
                   flexibility=round(sum(len(d) for d in inst.op_proc) / inst.n_ops, 2), scenarios=args.scenarios, kappa=args.kappa)
        # ---- planning on the inflated instance
        tilde = inst.failure_inflated_rounded(round_down=False, kappa=1)
        cg = ColumnGeneration(tilde, CGParams(time_limit=args.time, cp_threads=args.threads, cp_time=5, lns_time=30,
                                              n_workers=max(1, args.threads // 2)))
        r = cg.run(); plan = r.schedule; cg.close()
        row.update(plan_makespan=plan.makespan, plan_certificate=round(r.lb, 2))
        log.info("plan (inflated instance): makespan %d, certificate %.2f", plan.makespan, r.lb)
        # ---- mean-value certificate for static policies: floor(p~) instance, head-tail + destructive bound
        fl = inst.failure_inflated_rounded(round_down=True, kappa=1)
        hf, qf = heads_tails(fl); sf, _, _ = best_dispatch(fl, qf)
        htf = head_tail_bounds(fl, sf.makespan, cp_time=5, threads=args.threads)["gamma_ht"]
        ub_fl = min(sf.makespan, plan.makespan)
        rd = destructive_bound(fl, htf, ub_fl, cell_time=5, threads=args.threads, time_budget=args.shave_time)
        mv_bound = max(htf, rd["lb"])
        row.update(mv_bound=mv_bound)
        log.info("mean-value certificate (static policies): %d (head-tail %d, destructive %d)", mv_bound, htf, rd["lb"])
        # ---- policies
        h, q = heads_tails(inst)
        pols = {"static": lambda D: static_execute_realized(inst, plan, D),
                "reactive-plan": lambda D: reactive_execute_realized(inst, plan_order_policy(inst, plan), D, "plan"),
                "reactive-MWR": lambda D: reactive_execute_realized(inst, rule_policy(inst, q, "MWR"), D),
                "reactive-SPT": lambda D: reactive_execute_realized(inst, rule_policy(inst, q, "SPT"), D)}
        use_cp = inst.n_ops <= args.cp_max_ops
        t_bar = 3.0 * plan.makespan
        pen = None
        if use_cp:
            t = time.time()
            pen = fit_penalty(inst, [(args.seed + 1, n) for n in range(args.train)], t_bar, param=args.penalty_param,
                              iters=args.penalty_iters, kappa=args.kappa, cp_time=args.cp_time, threads=args.threads, log=log)
            row.update(t_penalty_fit=round(time.time() - t, 1),
                       penalty_max_theta=round(max((abs(v) for v in pen.theta.values()), default=0.0), 4))
            log.info("penalty fitted in %.0fs (param %s)", time.time() - t, args.penalty_param)
        # ---- scenarios
        phi_ht, phi_cp, psi, cp_closed = [], [], [], 0
        vals = {k: [] for k in pols}; zeta = {k: [] for k in pols}
        t0 = time.time()
        for n in range(args.scenarios):
            D = sample_scenario(inst, args.seed, n)
            sb = scenario_bound(inst, D, args.kappa, use_lp=False)
            phi_ht.append(sb.phi)
            if use_cp:
                r0 = penalized_bound_cpsat(inst, D, Penalty.zero(inst, t_bar), args.kappa, args.cp_time, args.threads)
                phi_cp.append(max(sb.phi, r0.psi_hat)); cp_closed += int(r0.status == "OPTIMAL")
                rp = penalized_bound_cpsat(inst, D, pen, args.kappa, args.cp_time, args.threads)
                psi.append(rp.psi_hat)
            for k, pol in pols.items():
                R = pol(D)
                vals[k].append(R.makespan)
                if pen is not None:
                    zeta[k].append(pen.zeta(R, D))
            if (n + 1) % 20 == 0:
                log.info("  scenario %d/%d  [%.0fs]  Phi_HT=%.1f%s", n + 1, args.scenarios, time.time() - t0, sb.phi,
                         f"  Phi_CP={phi_cp[-1]:.1f}  Psi={psi[-1]:.1f}" if use_cp else "")
        phi_ht = np.array(phi_ht)
        row.update(t_scenarios=round(time.time() - t0, 1), pi_ht_mean=round(phi_ht.mean(), 2), pi_ht_lcb=round(lcb(phi_ht), 2))
        best_pi = phi_ht
        if use_cp:
            phi_cp = np.array(phi_cp); psi = np.array(psi)
            row.update(pi_cp_mean=round(phi_cp.mean(), 2), pi_cp_lcb=round(lcb(phi_cp), 2), pi_cp_optimal_frac=round(cp_closed / args.scenarios, 3),
                       pen_mean=round(psi.mean(), 2), pen_lcb=round(lcb(psi), 2))
            best_pi = phi_cp
        log.info("certificates: PI(head-tail) %.2f (LCB %.2f)%s", phi_ht.mean(), lcb(phi_ht),
                 f"; PI(CP-SAT) {phi_cp.mean():.2f} (LCB {lcb(phi_cp):.2f}, optimal in {100*cp_closed/args.scenarios:.0f}%); penalized {psi.mean():.2f} (LCB {lcb(psi):.2f})" if use_cp else "")
        for k in pols:
            x = np.array(vals[k]); srt = np.sort(x); m = max(1, int(math.ceil(0.05 * len(x))))
            cvar = float(srt[-m:].mean())
            d = x - best_pi
            assert d.min() >= -1e-6, (k, d.min())
            cert = lcb(best_pi)
            if k == "static":
                cert = max(cert, mv_bound)
            rec = dict(mean=round(x.mean(), 2), cvar95=round(cvar, 2), gap_pi_mean=round(d.mean(), 2), gap_pi_ucb=round(ucb(d), 2),
                       cert=round(cert, 2), gap_cert_pct=round(100 * (x.mean() - cert) / cert, 2))
            if pen is not None:
                dz = x - np.array(zeta[k]) - psi
                assert dz.min() >= -1e-6, (k, dz.min())
                rec.update(zeta_mean=round(float(np.mean(zeta[k])), 3), gap_pen_mean=round(dz.mean(), 2), gap_pen_ucb=round(ucb(dz), 2))
            for kk, v in rec.items():
                row[f"{k}_{kk}"] = v
            log.info("  %-14s E[C]=%.2f CVaR95=%.2f | gap to PI %.2f (UCB %.2f) | certificate %.2f -> gap %.1f%%%s", k, x.mean(), cvar,
                     d.mean(), ucb(d), cert, 100 * (x.mean() - cert) / cert,
                     f" | gap to penalized {dz.mean():.2f} (UCB {ucb(dz):.2f})" if pen is not None else "")
        rows.append(row)
        with open(args.out + ".csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=sorted(set(k for r_ in rows for k in r_))); wr.writeheader(); wr.writerows(rows)
        with open(args.out + "_" + inst.name + ".json", "w") as f:
            json.dump(dict(row=row, phi_ht=phi_ht.tolist(), phi_cp=(phi_cp.tolist() if use_cp else None),
                           psi=(psi.tolist() if use_cp else None), makespans=vals), f)
        log.info("row: %s", json.dumps(row))
    log.info("done: %d instances -> %s.csv", len(rows), args.out)


if __name__ == "__main__":
    main()
