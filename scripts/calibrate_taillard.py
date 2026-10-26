#!/usr/bin/env python3
"""Calibration of the deterministic bound engine on classical job-shop
instances (one cell per machine): head-tail bound, destructive bound
(propagation + shaving), CP-SAT at equal resources, against published bounds.

  python scripts/calibrate_taillard.py ta41 ta43 ta45 ta47 ta50 --threads 6 --shave-time 1500 \
         --cpsat-time 600 --out results/calibration_taillard 2>&1 | tee logs/calibration_taillard.log
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mcfjsp.instances import read_jsp_taillard  # noqa: E402
from mcfjsp.bounds import heads_tails, head_tail_bounds  # noqa: E402
from mcfjsp.dispatch import best_dispatch  # noqa: E402
from mcfjsp.generator import diagnose  # noqa: E402
from mcfjsp.propagation import destructive_bound  # noqa: E402
from mcfjsp.baselines import cpsat_plant  # noqa: E402

# best known bounds (LB, UB) as of April 2025, arXiv:2504.16106; JSPLIB values in comments
PUBLISHED = {"ta41": (1915, 2006), "ta42": (1888, 1939), "ta43": (1809, 1846), "ta44": (1952, 1979),
             "ta45": (1997, 1997), "ta46": (1967, 2006), "ta47": (1819, 1889), "ta48": (1915, 1937),
             "ta49": (1934, 1960), "ta50": (1837, 1923)}   # Table 14 of arXiv:2504.16106 (April 2025)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--shave-time", type=float, default=1500.0)
    ap.add_argument("--cpsat-time", type=float, default=600.0)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmarks", "taillard"))
    ap.add_argument("--out", default="results/calibration_taillard")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger("calib")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for nm in args.names:
        inst = read_jsp_taillard(os.path.join(args.data, nm), nm)
        h, q = heads_tails(inst); sch, _, _ = best_dispatch(inst, q)
        dg = diagnose(inst)
        t = time.time(); ht = head_tail_bounds(inst, sch.makespan, cp_time=10, threads=args.threads); t_ht = time.time() - t
        cp = cpsat_plant(inst, sch.makespan, time_limit=args.cpsat_time, threads=args.threads)
        ub = min(sch.makespan, cp.ub)
        r = destructive_bound(inst, ht["gamma_ht"], ub, sweeps=2, cell_time=5, threads=args.threads,
                              time_budget=args.shave_time, log=log)
        lbp, ubp = PUBLISHED.get(nm, (None, None))
        row = dict(instance=nm, jobs=inst.n_jobs, machines=inst.n_machines, ops=inst.n_ops,
                   ratio_workload=round(dg["ratio_workload"], 3), ratio_path=round(dg["ratio_path"], 3),
                   lb_ht=ht["gamma_ht"], t_ht=round(t_ht, 1), lb_destructive=r["lb"], t_destructive=round(r["time"], 1),
                   destructive_tests=r["tests"], lb_cpsat=cp.lb, ub_cpsat=cp.ub, t_cpsat=round(cp.time, 1),
                   published_lb=lbp, published_ub=ubp,
                   dev_vs_published_lb_pct=(round(100 * (r["lb"] - lbp) / lbp, 2) if lbp else None),
                   threads=args.threads)
        rows.append(row)
        log.info("row: %s", json.dumps(row))
        with open(args.out + ".csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    log.info("done -> %s.csv", args.out)


if __name__ == "__main__":
    main()
