"""Frozen stochastic campaign of the paper (Table: policies under failures).

Runs scripts/stochastic_campaign.py with the fixed design below.  Nothing is
tunable from the command line except the output prefix and the thread count,
so that every reported number comes from one configuration.  Expected wall
time on a 6-core laptop: ~42 h (the 438-op synthetic and the 529-op
industrial instance dominate).  Design rows 0-8 are synthetic; index 9 in
--only stands for the two industrial windows (data/industrial).

    python scripts/final_stochastic.py --out results/final_stoch --threads 6
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (n_cells, machines_per_cell, n_jobs, ops_per_job, flexibility, availability, mttr_factor, seed)
DESIGN = [
    (4, "2-3", 12, "2-4", 0.8, 0.95, 2, 1),
    (4, "2-3", 12, "2-4", 0.8, 0.85, 2, 1),
    (4, "2-3", 12, "2-4", 0.3, 0.85, 2, 1),
    (5, "2-3", 40, "3-5", 0.6, 0.90, 2, 2),
    (5, "2-3", 40, "3-5", 0.6, 0.90, 2, 12),
    (5, "2-3", 40, "3-5", 0.6, 0.90, 2, 22),
    (8, "2-4", 100, "3-6", 0.5, 0.90, 2, 3),
    (8, "2-4", 100, "3-6", 0.5, 0.90, 2, 13),
    (8, "2-4", 100, "3-6", 0.5, 0.90, 2, 23),
]
INDUSTRIAL = ["data/industrial/ind_1d_1.json", "data/industrial/ind_3d_1.json"]  # availability set by --availability below
FIXED = ["--scenarios", "200", "--kappa", "2", "--time", "300", "--cp-time", "10", "--cp-max-ops", "400",
         "--opt-cp-time", "2", "--opt-every", "5", "--shave-time", "120", "--seed", "12345",
         "--inst-availability", "0.9", "--inst-mttr-factor", "2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final_stoch")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="design row indices (0-based) to run, e.g. for resuming")
    args = ap.parse_args()
    rows = [DESIGN[i] for i in args.only if i < len(DESIGN)] if args.only else DESIGN
    gens = [f"n_cells={c},machines_per_cell={m},n_jobs={j},ops_per_job={o},flexibility={f},availability={a},mttr_factor={r},seed={s}"
            for c, m, j, o, f, a, r, s in rows]
    inst = [] if args.only else INDUSTRIAL
    if args.only is not None and any(i >= len(DESIGN) for i in args.only):
        inst = INDUSTRIAL
    cmd = [sys.executable, os.path.join(HERE, "stochastic_campaign.py"), "--gen", *gens, *FIXED,
           *(["--instances", *[os.path.join(HERE, "..", f) for f in inst]] if inst else []),
           "--threads", str(args.threads), "--out", args.out]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
