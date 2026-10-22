# mcfjsp — certified column generation for multi-cell flexible job shops under machine failures

Code base for the manuscript *Certified Primal–Dual Coordination of Cell
Agents for Multi-Cell Flexible Job Shop Scheduling under Machine Failures*
(target: INFORMS Journal on Computing). Everything is standalone: Python 3.11+,
OR-Tools (CP-SAT, GLOP/HiGHS through `model_builder`), NumPy, SciPy. Gurobi is
optional (MILP baseline of Appendix EC.1). No dependency on any proprietary
simulator.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                                 # ~10 s
```

## Layout

```
src/mcfjsp/
  instances.py    data model, FJSP text reader, JSON (de)serialization
  partition.py    cell partition of benchmark instances (documented rule)
  generator.py    seeded generator of multi-cell instances with failures
  bounds.py       heads/tails, head–tail bounds (energetic, preemptive LP, CP-SAT)
  schedule.py     schedules, feasibility check, semi-active evaluation
  dispatch.py     serial schedule generation (repair) and dispatching rules
  oracles.py      cell oracles: CP-SAT exact pricing (with bound correction for
                  rounded weights), time-indexed LP (EC.2), compact plant TI-LP (z_TI)
  lp.py           LP wrapper (GLOP for duals; HiGHS fallback for primal-only solves)
  pricers.py      heuristic pricers (rule-based, LP-rounding); agent interface
  cg.py           Algorithm 1: stabilized column generation, anytime certificate,
                  head–tail cut, price-and-branch, horizon reduction
  primal.py       LNS with CP-SAT subproblems (cell / cell-pair / job neighbourhoods)
  subgradient.py  projected supergradient baseline (EC.4)
  baselines.py    monolithic CP-SAT; Gurobi MILP (EC.1)
  simulator.py    scenarios with per-(operation, machine) streams; static and
                  reactive execution (Assumption 1, Definition 1)
  stochastic.py   perfect-information certificate, paired gaps, CVaR
scripts/
  pilot.py            go/no-go pilot: all bounds at equal time and threads → CSV + JSON
  prepare_benchmarks.py  multi-cell variants of Brandimarte/Hurink/Dauzère-Pérès → data/
  counterexample.py   EC.5 (quadrature + Monte Carlo for Proposition 3)
tests/test_core.py    every bound checked against CP-SAT optima; feasibility; pathwise gaps
data/benchmarks/      instances (MIT, SchedulingLab/fjsp-instances) + bounds metadata
```

## Running

Always log: `python scripts/pilot.py ... 2>&1 | tee logs/pilot_$(date +%F).log`

```bash
# pilot on Brandimarte variants with 4 cells and two generated instances, 600 s, 8 threads
python scripts/pilot.py --bench mk01 mk02 mk03 mk04 mk05 mk06 mk07 mk08 mk09 mk10 \
    --cells 4 --gen n_cells=8,n_jobs=200,seed=1 n_cells=8,n_jobs=300,seed=2 \
    --time 600 --threads 8 --cp-time 10 --lns-time 60 --out results/pilot_$(date +%F) \
    2>&1 | tee logs/pilot_$(date +%F).log
# with Gurobi (academic licence on the university/personal laptop only)
python scripts/pilot.py ... --gurobi
```

Columns of the CSV: `ub_dispatch, lb_ht_energetic, lb_ht, z_ti, lb_cg (certificate =
max{CG, HT}), lb_cg_lagr (CG bound alone), ub_cg, gap_cg, t_gap_closed, lb_sg,
ub_cpsat, lb_cpsat, [ub_gurobi, lb_gurobi], best_ub, cert_gap_ours, cert_gap_cpsat`.
Pilot thresholds (plan v4): go if on n ≥ 200 the best of our bounds beats the
CP-SAT bound and the median certified gap is ≤ 10–15 %; strengthen at 15–25 %;
no-go above 25 % even with the head–tail bound.

## Correctness guarantees implemented

* Every certificate is a valid lower bound by construction: Theorem 1 at
  any duals in the dual domain (also at smoothed duals), with cell bounds
  `beta_c = max(TI-LP value, CP-SAT proven bound - rounding correction)`.
* CP-SAT pricing receives the TI-LP value as a cut (valid by Lemma EC.1) and
  returns a bound corrected for the rounding of real weights to integers.
* Horizon reduction after every UB improvement; columns violating the new
  windows are fixed to zero; incumbent columns always present (RMP feasible).
* Stochastic: scenarios are rounded down after scaling (Lemma EC.3); the
  perfect-information certificate is checked pathwise against every policy.

## Known limitations / to do

* Learned PPO+GNN pricer: interface in `pricers.py`; implementation pending
  (December). Rule-based and LP-rounding pricers are in place.
* Penalized information relaxation (Section 5.4): `penalty.py` pending.
* OR-Tools 9.15: `model_builder` with the HiGHS backend returns row
  activities from `dual_value()`; duals are therefore read with GLOP.
* Gurobi baseline is untested in the development sandbox.
