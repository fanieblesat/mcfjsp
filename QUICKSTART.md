# Quick start (laptops)

## 1. Set up (once per laptop)

```powershell
# Windows PowerShell (py launcher) — or use bash on WSL/Linux with python3
cd mcfjsp
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py -m pytest tests -q          # expect: 16 passed (~60 s)
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

## 6. Long upper-bound runs and the destructive bound (v0.2.2)

```powershell
py scripts/pilot.py --instances data/generated/cpl_c40_n40_s1.json data/generated/cpl_c60_n60_s2.json `
   --time 3600 --threads 6 --workers 3 --cp-time 15 --lns-time 600 --skip-sg --ti-max-ops 0 --shave-time 1800 `
   --out results/ub_long 2>&1 | Tee-Object logs/ub_long.log
```

## 7. Taillard calibration (v0.2.3)

```powershell
python scripts/calibrate_taillard.py ta41 ta42 ta43 ta44 ta45 ta46 ta47 ta48 ta49 ta50 --threads 6 --shave-time 1500 `
   --cpsat-time 600 --out results/calibration_taillard 2>&1 | Tee-Object logs/calibration_taillard.log
```

## 8. Stochastic campaign (v0.3.0)

Smoke test (~15 min), then the grid:

```powershell
python scripts/stochastic_campaign.py --gen n_cells=4,machines_per_cell=2-3,n_jobs=12,ops_per_job=2-4,flexibility=0.8,availability=0.9,mttr_factor=2,seed=1 `
   --scenarios 200 --train 40 --kappa 2 --time 120 --threads 6 --cp-time 10 --out results/stoch_smoke 2>&1 | Tee-Object logs/stoch_smoke.log

python scripts/stochastic_campaign.py `
   --gen n_cells=4,machines_per_cell=2-3,n_jobs=12,ops_per_job=2-4,flexibility=0.8,availability=0.95,mttr_factor=2,seed=1 `
         n_cells=4,machines_per_cell=2-3,n_jobs=12,ops_per_job=2-4,flexibility=0.8,availability=0.85,mttr_factor=2,seed=1 `
         n_cells=4,machines_per_cell=2-3,n_jobs=12,ops_per_job=2-4,flexibility=0.3,availability=0.85,mttr_factor=2,seed=1 `
         n_cells=5,machines_per_cell=2-3,n_jobs=40,ops_per_job=3-5,flexibility=0.6,availability=0.9,mttr_factor=2,seed=2 `
         n_cells=8,machines_per_cell=2-4,n_jobs=100,ops_per_job=3-6,flexibility=0.5,availability=0.9,mttr_factor=2,seed=3 `
   --scenarios 200 --train 40 --kappa 2 --time 300 --threads 6 --cp-time 10 --cp-max-ops 400 --out results/stoch_grid 2>&1 | Tee-Object logs/stoch_grid.log
```
Columns: `pi_ht_*`, `pi_cp_*`, `pen_*` (bounds, mean and 95 % LCB), per policy
`<policy>_mean/_cvar95/_gap_pi_*/_cert/_gap_cert_pct/_gap_pen_*`.

## 9. Parallel pricing

`--workers K` (pilot flag to add; `CGParams(n_workers=K)`) prices the cells
in K processes. Use K = number of cells (≤ cores) and `--threads` = cores so
that each CP-SAT call gets cores/K threads.
