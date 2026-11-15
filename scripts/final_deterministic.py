"""Frozen deterministic campaign of the paper (Table: certified gaps by structure
and size, plus the industrial windows).  Runs scripts/pilot.py in groups with
fixed time limits; only --out, --threads and --groups (to resume) are accepted.

    python scripts/final_deterministic.py --out results/final_det --threads 6

Groups (approximate wall time on a 6-core laptop, total ~13 h):
  bench      Brandimarte mk01-mk10 as 4-cell plants, 600 s            (~2.5 h)
  balanced   balanced generated plants, 265-1300 ops, 600 s           (~1 h)
  coupled    coupling- and workload-dominated plants, 225-4500 ops,
             600 s (<=600 ops) / 900 s (>600 ops)                     (~3.5 h)
  square     40x40 and 60x60 square plants, 3600 s                    (~3 h)
  industrial twelve anonymized plant windows, 300 s                   (~2 h)
Each group writes <out>_<group>.csv/.log; scripts/make_tables.py --det merges them.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = ["--cp-time", "60", "--lns-time", "30", "--shave-time", "120", "--ti-max-ops", "600", "--preemptive-max-ops", "400"]

GROUPS = {
    "bench": [["--bench", *[f"mk{i:02d}" for i in range(1, 11)], "--cells", "4", "--time", "600"]],
    "balanced": [["--gen", "n_cells=5,machines_per_cell=2-4,n_jobs=60,ops_per_job=3-6,flexibility=0.6,seed=1",
                  "n_cells=8,machines_per_cell=2-4,n_jobs=120,ops_per_job=3-6,flexibility=0.5,seed=2",
                  "n_cells=8,machines_per_cell=2-4,n_jobs=200,ops_per_job=4-8,flexibility=0.5,seed=3",
                  "--time", "600"]],
    "coupled": [["--coupled", "n_cells=15,n_jobs=15,visits_per_job=15-15,ops_per_visit=1-1,seed=1", "n_cells=20,n_jobs=20,visits_per_job=20-20,ops_per_visit=1-1,seed=1", "n_cells=20,n_jobs=30,visits_per_job=20-20,ops_per_visit=1-1,seed=1",
                 "--time", "600"],
                ["--coupled", "n_cells=20,n_jobs=60,visits_per_job=20-20,ops_per_visit=1-1,seed=1", "n_cells=30,n_jobs=100,visits_per_job=30-30,ops_per_visit=1-1,seed=1", "n_cells=30,n_jobs=150,visits_per_job=30-30,ops_per_visit=1-1,seed=1",
                 "n_cells=50,n_jobs=80,visits_per_job=50-50,ops_per_visit=1-1,seed=1", "--time", "900"]],
    "square": [["--coupled", "n_cells=40,machines_per_cell=1-1,n_jobs=40,visits_per_job=40-40,ops_per_visit=1-1,seed=1",
                "n_cells=60,machines_per_cell=1-1,n_jobs=60,visits_per_job=60-60,ops_per_visit=1-1,seed=1",
                "--time", "3600"]],
    "industrial": [["--instances", *[os.path.join(HERE, "..", "data", "industrial", f"ind_{d}d_{k}.json")
                                     for d in (1, 3, 7) for k in (1, 2, 3, 4)], "--time", "300"]],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final_det")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--groups", nargs="*", default=list(GROUPS), help="subset of groups to run (resume)")
    args = ap.parse_args()
    for g in args.groups:
        for i, spec in enumerate(GROUPS[g]):
            out = f"{args.out}_{g}" + (f"{i+1}" if len(GROUPS[g]) > 1 else "")
            cmd = [sys.executable, os.path.join(HERE, "pilot.py"), *spec, *COMMON, "--threads", str(args.threads), "--out", out]
            print(" ".join(cmd), flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"group {g} part {i+1} failed with code {rc}; rerun with --groups {g}", file=sys.stderr)
                sys.exit(rc)


if __name__ == "__main__":
    main()
