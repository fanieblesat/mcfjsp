#!/usr/bin/env python3
"""Go/no-go pilot (plan v4): bounds and certified gaps at equal time budgets.

For each instance:
  dispatch UB, head-tail bound (energetic / preemptive / CP-SAT), compact
  TI-LP bound z_TI (CG with LP oracle only, no HT cut), CG bound with exact
  pricing + HT cut, subgradient bound, CP-SAT plant (UB, LB) at the same time
  limit and thread count, optional Gurobi MILP. Writes one CSV row per
  instance and a JSON with the full histories.

Usage:
  python scripts/pilot.py --bench mk01 mk02 --cells 3 --time 120 --threads 2 \
         --gen 'n_cells=5,n_jobs=100,seed=1' --out results/pilot
Always run as:  python scripts/pilot.py ... 2>&1 | tee logs/pilot_<date>.log
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

from mcfjsp.instances import Instance, read_fjsp_text  # noqa: E402
from mcfjsp.partition import multicell_variant  # noqa: E402
from mcfjsp.generator import generate, generate_coupled, diagnose  # noqa: E402
from mcfjsp.bounds import head_tail_bounds, heads_tails  # noqa: E402
from mcfjsp.dispatch import best_dispatch  # noqa: E402
from mcfjsp.cg import CGParams, ColumnGeneration  # noqa: E402
from mcfjsp.subgradient import subgradient  # noqa: E402
from mcfjsp.baselines import cpsat_plant  # noqa: E402

BENCH_DIRS = {"mk": "brandimarte", "edata": "hurink/edata", "rdata": "hurink/rdata", "vdata": "hurink/vdata"}


def load_bench(name: str, root: str) -> Instance:
    if name.startswith("mk"):
        path = os.path.join(root, "brandimarte", name + ".txt")
    elif "/" in name:
        path = os.path.join(root, name + ".txt")
    else:
        path = os.path.join(root, "dauzere", name + ".txt")
    return read_fjsp_text(path, name)


RANGE_KEYS = {"machines_per_cell", "ops_per_job", "visits_per_job", "ops_per_visit", "proc_range"}


def parse_kv(s: str) -> dict:
    """k=v pairs separated by commas; range parameters accept 'a-b' or a single value."""
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        if k in RANGE_KEYS:
            parts = [int(x) for x in v.split("-")]
            out[k] = (parts[0], parts[-1])
            continue
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
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--gen", nargs="*", default=[], help="generator v1 kwargs, e.g. n_cells=5,n_jobs=100,seed=1")
    ap.add_argument("--coupled", nargs="*", default=[], help="generator v2 (coupling-dominated) kwargs, e.g. n_cells=8,n_jobs=20,seed=1")
    ap.add_argument("--instances", nargs="*", default=[], help="JSON instance files")
    ap.add_argument("--cells", type=int, nargs="*", default=[3])
    ap.add_argument("--time", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cp-time", type=float, default=5.0)
    ap.add_argument("--lns-time", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=3, help="process pool over cells for pricing")
    ap.add_argument("--preemptive-max-ops", type=int, default=400, help="skip the preemptive HT bound above this size")
    ap.add_argument("--ti-max-ops", type=int, default=1000, help="skip the compact TI-LP diagnostic above this size")
    ap.add_argument("--shave-time", type=float, default=0.0,
                    help="time budget (s) for the destructive bound with shaving (propagation.py); 0 = off")
    ap.add_argument("--skip-sg", action="store_true")
    ap.add_argument("--skip-cpsat", action="store_true")
    ap.add_argument("--gurobi", action="store_true")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmarks"))
    ap.add_argument("--out", default="results/pilot")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    log = logging.getLogger("pilot")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    instances = []
    for b in args.bench:
        base = load_bench(b, args.data)
        for C in args.cells:
            instances.append(multicell_variant(base, C) if C > 1 else base)
    for g in args.gen:
        instances.append(generate(**parse_kv(g)))
    for g in args.coupled:
        instances.append(generate_coupled(**parse_kv(g)))
    for f in args.instances:
        instances.append(Instance.load(f))

    rows = []; hist = {}
    for inst in instances:
        log.info("=== %s: jobs %d machines %d ops %d cells %d transitions %d", inst.name, inst.n_jobs,
                 inst.n_machines, inst.n_ops, inst.n_cells, len(inst.transitions))
        row = dict(instance=inst.name, jobs=inst.n_jobs, machines=inst.n_machines, ops=inst.n_ops,
                   cells=inst.n_cells, transitions=len(inst.transitions), time_limit=args.time, threads=args.threads)
        dg = diagnose(inst)
        row.update(ratio_workload=dg["ratio_workload"], ratio_path=dg["ratio_path"])
        log.info("structure: workload/UB %.2f, path/UB %.2f", dg["ratio_workload"], dg["ratio_path"])
        os.makedirs("data/generated", exist_ok=True)
        inst.save(os.path.join("data", "generated", inst.name + ".json"))
        h, q = heads_tails(inst)
        sch, rule, _ = best_dispatch(inst, q)
        row["ub_dispatch"] = sch.makespan
        t = time.time(); ht = head_tail_bounds(inst, sch.makespan, cp_time=args.cp_time, threads=args.threads,
                                               use_preemptive=inst.n_ops <= args.preemptive_max_ops)
        row.update(lb_ht_energetic=ht["gamma_energetic"], lb_ht=ht["gamma_ht"], t_ht=round(time.time() - t, 1))
        log.info("dispatch UB %d (%s); HT energetic %d, HT best %d (%.1fs)", sch.makespan, rule,
                 ht["gamma_energetic"], ht["gamma_ht"], time.time() - t)
        # z_TI: compact time-indexed LP of the plant (Proposition 'Hierarchy'), coarsened if needed
        # (diagnostic only, not a certificate; skipped above --ti-max-ops where the LP does not solve in time/4)
        if inst.n_ops <= args.ti_max_ops:
            from mcfjsp.oracles import compact_ti_lp_coarse
            t = time.time()
            z_ti, kappa_ti, st_ti = compact_ti_lp_coarse(inst, sch.makespan, time_limit=args.time / 4)
            row.update(z_ti=round(z_ti, 3), z_ti_kappa=kappa_ti, t_ti=round(time.time() - t, 1), z_ti_status=st_ti)
            log.info("compact TI-LP bound z_TI: %.3f (grid kappa=%s) [%s, %.1fs]", z_ti, kappa_ti, st_ti, time.time() - t)
        else:
            row.update(z_ti=None, z_ti_kappa=None, t_ti=0.0, z_ti_status="skipped")
        # CG with exact pricing and HT cut
        p_cg = CGParams(time_limit=args.time, cp_time=args.cp_time, cp_threads=args.threads, lns_time=args.lns_time,
                        exact_every=3, stop_on_gap=False, n_workers=args.workers)
        cg = ColumnGeneration(inst, p_cg)
        r_cg = cg.run()
        t_close = next((rec["t"] for rec in r_cg.history if rec["lb"] > 0 and (rec["ub"] - rec["lb"]) / rec["lb"] <= 1e-12), None)
        # which bound carries the certificate. With the head-tail cut in the master the CG bound is >= gamma_HT by
        # construction, so 'lagr' is claimed only when it is strictly above the head-tail bound.
        if r_cg.lb_cg > r_cg.gamma_ht + 1e-6:
            active = "lagr"
        elif r_cg.gamma_ht > row["lb_ht_energetic"]:
            active = "ht_cp_or_preemptive"
        else:
            active = "ht_energetic"
        row.update(lb_cg=round(r_cg.lb, 3), lb_cg_lagr=round(r_cg.lb_cg, 3), ub_cg=r_cg.ub, gap_cg=round(100 * r_cg.gap, 2),
                   t_gap_closed=t_close, active_bound=active,
                   cg_iters=r_cg.iterations, cg_cols=r_cg.n_columns, cg_exact=r_cg.exact_calls, t_cg=round(r_cg.time, 1),
                   cg_status=r_cg.status)
        hist[inst.name] = {"cg": r_cg.history}
        if args.shave_time > 0:
            from mcfjsp.propagation import destructive_bound
            r_d = destructive_bound(inst, int(r_cg.lb), r_cg.ub, shaving=True, sweeps=2, cell_time=args.cp_time,
                                    threads=args.threads, time_budget=args.shave_time, log=log)
            row.update(lb_destructive=r_d["lb"], t_destructive=round(r_d["time"], 1), destructive_tests=r_d["tests"])
            log.info("destructive bound (shaving): %d [%d tests, %.1fs]", r_d["lb"], r_d["tests"], r_d["time"])
        log.info("CG: LB %.3f (Lagr %.3f, HT %d) UB %d gap %.2f%% [%s, %.1fs, %d it, %d cols]", r_cg.lb, r_cg.lb_cg,
                 r_cg.gamma_ht, r_cg.ub, 100 * r_cg.gap, r_cg.status, r_cg.time, r_cg.iterations, r_cg.n_columns)
        if not args.skip_sg:
            r_sg = subgradient(inst, time_limit=args.time / 2, oracle="lp", threads=args.threads, ht_cp_time=args.cp_time)
            row.update(lb_sg=round(r_sg.lb_lagr, 3), sg_iters=r_sg.iterations, t_sg=round(r_sg.time, 1))
            hist[inst.name]["sg"] = r_sg.history
            log.info("subgradient (LP oracle): Lagr %.3f [%d it, %.1fs]", r_sg.lb_lagr, r_sg.iterations, r_sg.time)
        if not args.skip_cpsat:
            r_cp = cpsat_plant(inst, sch.makespan, time_limit=args.time, threads=args.threads)
            row.update(ub_cpsat=r_cp.ub, lb_cpsat=r_cp.lb, cpsat_status=r_cp.status, t_cpsat=round(r_cp.time, 1))
            log.info("CP-SAT plant: UB %s LB %d [%s, %.1fs]", r_cp.ub, r_cp.lb, r_cp.status, r_cp.time)
        if args.gurobi:
            from mcfjsp.baselines import gurobi_plant
            try:
                r_g = gurobi_plant(inst, sch.makespan, time_limit=args.time, threads=args.threads)
                row.update(ub_gurobi=r_g.ub, lb_gurobi=r_g.lb, gurobi_status=r_g.status, t_gurobi=round(r_g.time, 1))
                log.info("Gurobi: UB %s LB %d [%s, %.1fs]", r_g.ub, r_g.lb, r_g.status, r_g.time)
            except Exception as e:
                log.warning("Gurobi baseline skipped: %s", e)
                row.update(gurobi_status=f"skipped: {e}")
        best_ub = min(x for x in [row.get("ub_cg"), row.get("ub_cpsat"), row.get("ub_gurobi"), row["ub_dispatch"]] if x)
        best_lb_ours = max(row["lb_cg"], row.get("lb_destructive", 0))
        row["best_ub"] = best_ub
        row["cert_gap_ours"] = round(100 * (best_ub - best_lb_ours) / best_lb_ours, 2) if best_lb_ours > 0 else None
        if "lb_cpsat" in row and row["lb_cpsat"] > 0:
            row["cert_gap_cpsat"] = round(100 * (best_ub - row["lb_cpsat"]) / row["lb_cpsat"], 2)
        rows.append(row)
        with open(args.out + ".csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=sorted(set(k for r in rows for k in r)))
            wr.writeheader(); wr.writerows(rows)
        with open(args.out + "_history.json", "w") as f:
            json.dump(hist, f)
        log.info("row: %s", json.dumps(row))
    log.info("done: %d instances -> %s.csv", len(rows), args.out)


if __name__ == "__main__":
    main()
