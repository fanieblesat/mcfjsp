# Quick start (laptops)

## 1. Set up (once per laptop)

```powershell
# Windows PowerShell (py launcher) — or use bash on WSL/Linux with python3
cd mcfjsp
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py -m pytest tests -q          # expect: 12 passed (~15 s)
```

Gurobi (university/personal laptop only): `pip install gurobipy`, licence
already installed → add `--gurobi` to the pilot.

## 2. Pilot runs (November weeks 1–2)

Each run logs to a file; keep the logs, they are part of the repository.

```powershell
# A. Brandimarte variants, 4 cells, 600 s, 8 threads, with Gurobi
py scripts/pilot.py --bench mk01 mk02 mk03 mk04 mk05 mk06 mk07 mk08 mk09 mk10 --cells 4 `
   --time 600 --threads 6 --cp-time 10 --lns-time 60 --gurobi --out results/pilot_A 2>&1 | Tee-Object logs/pilot_A.log

# B. Large generated instances (n >= 200 jobs), 8 cells
py scripts/pilot.py --gen n_cells=8,n_jobs=200,seed=1 n_cells=8,n_jobs=200,seed=2 n_cells=8,n_jobs=300,seed=3 `
   --time 600 --threads 6 --cp-time 10 --lns-time 60 --out results/pilot_B 2>&1 | Tee-Object logs/pilot_B.log

# C. Stochastic smoke test (certificate vs. policies)
py scripts/stochastic_eval.py --gen n_cells=5,n_jobs=40,availability=0.9,mttr_factor=2,seed=1 `
   --scenarios 100 --kappa 2 --time 120 --threads 6 --out results/stoch_C 2>&1 | Tee-Object logs/stoch_C.log
```

On bash replace the backtick continuation by `\` and `Tee-Object` by `tee`.

## 3. What to send back

`results/*.csv`, `results/*_history.json`, `logs/*.log`. The go/no-go
decision reads the columns `lb_cg`, `lb_cpsat`, `lb_gurobi`, `best_ub`,
`cert_gap_ours`, `cert_gap_cpsat` for the n ≥ 200 instances.

## 4. Coupling-dominated instances (v0.2)

```powershell
py scripts/pilot.py --coupled n_cells=15,machines_per_cell=1,n_jobs=15,visits_per_job=15,ops_per_visit=1,flexibility=0,seed=1 `
   --time 600 --threads 6 --workers 3 --cp-time 10 --lns-time 60 --gurobi --out results/pilot_D 2>&1 | Tee-Object logs/pilot_D.log
```
(`machines_per_cell=1` means the range (1,1); pass `machines_per_cell=1,2`-style ranges as `machines_per_cell=1-2`.)

## 5. Large instances (pilots E/F)

```powershell
py scripts/pilot.py --coupled n_cells=40,machines_per_cell=1,n_jobs=40,visits_per_job=40,ops_per_visit=1,flexibility=0,seed=1 `
   n_cells=60,machines_per_cell=1,n_jobs=60,visits_per_job=60,ops_per_visit=1,flexibility=0,seed=2 `
   n_cells=50,machines_per_cell=1-2,n_jobs=80,visits_per_job=50,ops_per_visit=1,flexibility=0.2,seed=3 `
   --time 900 --threads 6 --workers 3 --cp-time 15 --lns-time 60 --skip-sg --out results/pilot_F 2>&1 | Tee-Object logs/pilot_F.log
```
Check the `structure:` line: coupling-dominated means workload/UB and path/UB both well below 1.

## 6. Parallel pricing

`--workers K` (pilot flag to add; `CGParams(n_workers=K)`) prices the cells
in K processes. Use K = number of cells (≤ cores) and `--threads` = cores so
that each CP-SAT call gets cores/K threads.
