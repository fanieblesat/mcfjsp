"""Build anonymized multi-cell FJSP instances from the plant's order export.

    python scripts/export_industrial.py --raw /path/to/21s_export_2025 --out data/industrial \
        --windows 1:4 3:4 7:4 --scale 0.9137 --seed 7

The raw export (one JSON line per closed production order, with the planned
step list `extra.planning` and the routing) never enters the repository.
What is written:

  * `ind_<days>d_<k>.json`  - Instance files in the package's JSON format.
  * `industrial_summary.csv` - one row per instance (jobs, ops, cells, ...).
  * `industrial_cells.csv`   - the cell structure actually used.

Anonymization (irreversible from the published files):
  * cells and machines are renamed c1..c8 / m1..m48, in a fixed order that is
    not the plant's naming order; article numbers, order ids, product names,
    zone ids and dates are dropped;
  * every duration is multiplied by a fixed, unpublished factor (`--scale`) and
    rounded to whole minutes; release times are relative to the window start;
  * the calendar position of each window is not recorded (only its length).

Construction:
  * one cell per process stage with more than one interchangeable station,
    plus the single-station stages that appear in routings (see STAGES);
  * one job per order released (`start`) inside the window; its operations are
    the planned steps in order of planned start, restricted to the cells kept;
  * eligible machines of an operation = stations of that stage that processed
    the same article anywhere in the year (observed flexibility); if the
    article was only ever seen on one station, the operation is dedicated;
  * processing time on a station = median planned duration of (article, stage,
    station) over the year when observed, otherwise the article-stage median
    scaled by the station's speed factor (station median minutes per unit
    divided by the stage median), times the order quantity;
  * failure parameters are NOT estimated from this export (it records zone
    dwell, which mixes queueing and processing); the instances carry lam=0 and
    the campaign sets availability levels explicitly, as for the synthetic
    instances.
"""
import argparse, collections, csv, glob, json, os, random, statistics
from datetime import datetime, timedelta

# stage name in the export -> cell index; stages absent here are ignored
STAGES = {
    "Bestücken": 0,       # SMT placement lines
    "ICT": 1,             # in-circuit test
    "Nutzentrennen": 2,   # depaneling
    "Funktionstest": 3,   # functional test
    "Zwischenprüfen": 4,  # intermediate test
    "Lackieren": 5,       # conformal coating
    "Combi-1": 6, "THT": 6, "Selektivlöten": 6, "AOI": 6,   # THT / selective soldering / AOI group
    "Underfill": 7,
}
SKIP = {"Bahnhof", "Nacharbeit"}


def P(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_orders(raw):
    for f in sorted(glob.glob(os.path.join(raw, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                steps = []
                for name, v in (d["extra"].get("planning") or {}).items():
                    if name in SKIP or name not in STAGES:
                        continue
                    if not (v.get("targetStart") and v.get("targetEnd") and v.get("targetZoneName")):
                        continue
                    dur = (P(v["targetEnd"]) - P(v["targetStart"])).total_seconds() / 60
                    if dur <= 0:
                        continue
                    steps.append((P(v["targetStart"]), STAGES[name], v["targetZoneName"], dur))
                if len(steps) < 2:
                    continue
                steps.sort()
                yield {"start": P(d["start"]), "due": P(d["due"]) if d.get("due") else None,
                       "article": d["articleNumber"], "qty": max(1, int(d["quantity"])), "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/industrial")
    ap.add_argument("--windows", nargs="+", default=["1:4", "3:4", "7:4"], help="days:count")
    ap.add_argument("--scale", type=float, required=True, help="unpublished duration factor")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-release-year", type=int, default=2025)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    orders = list(load_orders(a.raw))
    # ---- station universe and anonymous names -------------------------
    stations = collections.defaultdict(set)
    for o in orders:
        for _, c, z, _ in o["steps"]:
            stations[c].add(z)
    cells = sorted(stations)
    m_name, machine_cell, machine_id = {}, [], {}
    for c in cells:
        zs = sorted(stations[c]); rng.shuffle(zs)
        for z in zs:
            machine_id[(c, z)] = len(machine_cell); machine_cell.append(c)
    # ---- durations: (article, stage, station) medians, speed factors ----
    asz = collections.defaultdict(list); as_ = collections.defaultdict(list)
    sz_unit = collections.defaultdict(list); s_unit = collections.defaultdict(list)
    eligible = collections.defaultdict(set)
    for o in orders:
        for _, c, z, dur in o["steps"]:
            asz[(o["article"], c, z)].append(dur); as_[(o["article"], c)].append(dur)
            sz_unit[(c, z)].append(dur / o["qty"]); s_unit[c].append(dur / o["qty"])
            eligible[(o["article"], c)].add(z)
    med = lambda v: statistics.median(v)
    speed = {k: med(v) / med(s_unit[k[0]]) for k, v in sz_unit.items()}

    def proc_time(article, c, z, qty_ref_dur):
        if (article, c, z) in asz:
            d = med(asz[(article, c, z)])
        else:
            d = med(as_[(article, c)]) * speed[(c, z)]
        return max(1, int(round(d * a.scale)))

    # ---- windows --------------------------------------------------------
    starts = sorted(o["start"] for o in orders if o["start"].year >= a.min_release_year)
    t0, t1 = starts[0], starts[-1]
    rows = []
    for spec in a.windows:
        days, count = map(int, spec.split(":"))
        for k in range(count):
            # window start chosen uniformly in the year, avoiding the last `days`
            ws = t0 + timedelta(seconds=rng.random() * ((t1 - t0).total_seconds() - days * 86400))
            ws = ws.replace(hour=0, minute=0, second=0, microsecond=0)
            # start weekly windows on a Monday and shorter ones on a weekday
            while ws.weekday() >= 5 or (days >= 7 and ws.weekday() != 0):
                ws += timedelta(days=1)
            we = ws + timedelta(days=days)
            jobs = [o for o in orders if ws <= o["start"] < we]
            if not jobs:
                continue
            job_ops, op_job, op_pos, op_cell, op_proc, release = [], [], [], [], [], []
            for j, o in enumerate(jobs):
                ops = []
                for pos, (_, c, z, dur) in enumerate(o["steps"]):
                    elig = eligible[(o["article"], c)] or {z}
                    proc = {machine_id[(c, zz)]: proc_time(o["article"], c, zz, dur) for zz in sorted(elig)}
                    ops.append(len(op_job)); op_job.append(j); op_pos.append(pos); op_cell.append(c); op_proc.append(proc)
                job_ops.append(ops)
                release.append(int(round((o["start"] - ws).total_seconds() / 60 * a.scale)))
            name = f"ind_{days}d_{k+1}"
            flex = round(sum(len(d) for d in op_proc) / len(op_proc), 3)
            inst = {"name": name, "n_machines": len(machine_cell), "cell_of_machine": machine_cell,
                    "lam": [0.0] * len(machine_cell), "mu": [1.0] * len(machine_cell),
                    "jobs": [{"release": release[j],
                              "ops": [{"cell": op_cell[x], "proc": {str(m): p for m, p in op_proc[x].items()}}
                                      for x in ops]} for j, ops in enumerate(job_ops)],
                    "meta": {"source": "anonymized plant export", "window_days": days, "flexibility": flex,
                             "note": "durations rescaled by an unpublished factor; failures set by the campaign"}}
            with open(os.path.join(a.out, name + ".json"), "w") as f:
                json.dump(inst, f)
            n_ops = len(op_job)
            rows.append({"instance": name, "window_days": days, "jobs": len(jobs), "ops": n_ops,
                         "cells": len(cells), "machines": len(machine_cell),
                         "flexibility": inst["meta"]["flexibility"],
                         "ops_per_job": round(n_ops / len(jobs), 2),
                         "total_work_min": sum(min(d.values()) for d in op_proc),
                         "share_dedicated": round(sum(len(d) == 1 for d in op_proc) / n_ops, 3)})
            print(f"{name}: jobs {len(jobs)} ops {n_ops} flex {inst['meta']['flexibility']}")
    with open(os.path.join(a.out, "industrial_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "industrial_cells.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cell", "stations", "median_min_per_unit_scaled"])
        for c in cells:
            w.writerow([f"c{c+1}", len(stations[c]), round(med(s_unit[c]) * a.scale, 3)])


if __name__ == "__main__":
    main()
