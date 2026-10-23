"""Generator of multi-cell instances with native inter-cell routings.

Parameters (all documented in the paper's experimental design):
  n_cells            number of cells
  machines_per_cell  (lo, hi) uniform per cell
  n_jobs             number of jobs
  ops_per_job        (lo, hi) uniform per job
  proc_range         (lo, hi) uniform integer processing times (per machine)
  flexibility        probability that an additional machine of the cell is
                     eligible (at least one always is)
  revisit_prob       probability that a job's next cell equals a cell already
                     visited (otherwise a new cell is drawn); consecutive
                     repeats are allowed (same-cell precedences stay internal)
  release_spread     release dates uniform in [0, spread * total_mean_work/ (m)]
  availability       long-run machine availability mu/(lam+mu); None = no failures
  mttr_factor        mean time to repair as a multiple of the mean processing time
Seeded by `seed` (numpy Generator), fully reproducible.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .instances import Instance


def generate(n_cells: int = 3, machines_per_cell: Tuple[int, int] = (2, 4), n_jobs: int = 20,
             ops_per_job: Tuple[int, int] = (3, 6), proc_range: Tuple[int, int] = (1, 20),
             flexibility: float = 0.5, revisit_prob: float = 0.2, release_spread: float = 0.0,
             availability: Optional[float] = None, mttr_factor: float = 1.0,
             seed: int = 0, name: Optional[str] = None) -> Instance:
    rng = np.random.default_rng(seed)
    cell_of_machine = []
    for c in range(n_cells):
        k = int(rng.integers(machines_per_cell[0], machines_per_cell[1] + 1))
        cell_of_machine += [c] * k
    m = len(cell_of_machine)
    machines_of_cell = [[i for i in range(m) if cell_of_machine[i] == c] for c in range(n_cells)]
    release, job_ops, op_job, op_pos, op_cell, op_proc = [], [], [], [], [], []
    for j in range(n_jobs):
        n_ops = int(rng.integers(ops_per_job[0], ops_per_job[1] + 1))
        visited = []
        ops = []
        for k in range(n_ops):
            if visited and rng.random() < revisit_prob:
                c = int(rng.choice(visited))
            else:
                c = int(rng.integers(0, n_cells))
            visited.append(c)
            ms = machines_of_cell[c]
            elig = [int(rng.choice(ms))]
            for i in ms:
                if i not in elig and rng.random() < flexibility:
                    elig.append(i)
            proc = {i: int(rng.integers(proc_range[0], proc_range[1] + 1)) for i in sorted(elig)}
            a = len(op_job)
            ops.append(a)
            op_job.append(j); op_pos.append(k); op_cell.append(c); op_proc.append(proc)
        job_ops.append(ops)
    mean_p = (proc_range[0] + proc_range[1]) / 2
    if release_spread > 0:
        horizon = release_spread * len(op_job) * mean_p / m
        release = [int(rng.integers(0, max(1, int(horizon)) + 1)) for _ in range(n_jobs)]
    else:
        release = [0] * n_jobs
    lam = [0.0] * m; mu = [1.0] * m
    if availability is not None and availability < 1.0:
        mttr = mttr_factor * mean_p
        mu = [1.0 / mttr] * m
        # availability A = mu/(lam+mu)  =>  lam = mu (1-A)/A
        lam = [mu[0] * (1.0 - availability) / availability] * m
    meta = dict(generator="mcfjsp.generator.generate", seed=seed, n_cells=n_cells,
                machines_per_cell=list(machines_per_cell), n_jobs=n_jobs, ops_per_job=list(ops_per_job),
                proc_range=list(proc_range), flexibility=flexibility, revisit_prob=revisit_prob,
                release_spread=release_spread, availability=availability, mttr_factor=mttr_factor)
    return Instance(name or f"gen_c{n_cells}_n{n_jobs}_s{seed}", m, cell_of_machine, release, job_ops,
                    op_job, op_pos, op_cell, op_proc, lam, mu, meta)


# ---------------------------------------------------------------------
# Generator v2: coupling-dominated ("job-shop-like") multi-cell instances
# ---------------------------------------------------------------------
def generate_coupled(n_cells: int = 8, machines_per_cell: Tuple[int, int] = (2, 3), n_jobs: int = 30,
                     visits_per_job: Tuple[int, int] = (6, 8), ops_per_visit: Tuple[int, int] = (1, 2),
                     proc_range: Tuple[int, int] = (1, 99), flexibility: float = 0.5,
                     release_spread: float = 0.0, availability: Optional[float] = None,
                     mttr_factor: float = 1.0, seed: int = 0, name: Optional[str] = None) -> Instance:
    """Instances whose makespan is governed by inter-cell coupling rather than
    by the workload of one bottleneck cell: few jobs relative to machines
    (n_jobs ~ number of machines), each job visits most cells in a random
    order (as in a classical job shop), Taillard-type processing times
    U(1, 99). The head-tail bound is typically not tight on this class and
    monolithic solvers struggle to prove bounds (compare Taillard 30x20)."""
    rng = np.random.default_rng(seed)
    cell_of_machine = []
    for c in range(n_cells):
        k = int(rng.integers(machines_per_cell[0], machines_per_cell[1] + 1))
        cell_of_machine += [c] * k
    m = len(cell_of_machine)
    machines_of_cell = [[i for i in range(m) if cell_of_machine[i] == c] for c in range(n_cells)]
    release, job_ops, op_job, op_pos, op_cell, op_proc = [], [], [], [], [], []
    for j in range(n_jobs):
        nv = int(rng.integers(visits_per_job[0], min(visits_per_job[1], n_cells) + 1))
        route = list(rng.permutation(n_cells)[:nv])
        ops = []
        for c in route:
            c = int(c)
            for _ in range(int(rng.integers(ops_per_visit[0], ops_per_visit[1] + 1))):
                ms = machines_of_cell[c]
                elig = [int(rng.choice(ms))]
                for i in ms:
                    if i not in elig and rng.random() < flexibility:
                        elig.append(i)
                base = int(rng.integers(proc_range[0], proc_range[1] + 1))
                proc = {}
                for i in sorted(elig):
                    proc[i] = max(1, int(round(base * float(rng.uniform(0.8, 1.25)))))
                a = len(op_job)
                ops.append(a)
                op_job.append(j); op_pos.append(len(ops) - 1); op_cell.append(c); op_proc.append(proc)
        job_ops.append(ops)
    mean_p = (proc_range[0] + proc_range[1]) / 2
    if release_spread > 0:
        horizon = release_spread * len(op_job) * mean_p / m
        release = [int(rng.integers(0, max(1, int(horizon)) + 1)) for _ in range(n_jobs)]
    else:
        release = [0] * n_jobs
    lam = [0.0] * m; mu = [1.0] * m
    if availability is not None and availability < 1.0:
        mttr = mttr_factor * mean_p
        mu = [1.0 / mttr] * m
        lam = [mu[0] * (1.0 - availability) / availability] * m
    meta = dict(generator="mcfjsp.generator.generate_coupled", seed=seed, n_cells=n_cells,
                machines_per_cell=list(machines_per_cell), n_jobs=n_jobs, visits_per_job=list(visits_per_job),
                ops_per_visit=list(ops_per_visit), proc_range=list(proc_range), flexibility=flexibility,
                release_spread=release_spread, availability=availability, mttr_factor=mttr_factor)
    return Instance(name or f"cpl_c{n_cells}_n{n_jobs}_s{seed}", m, cell_of_machine, release, job_ops,
                    op_job, op_pos, op_cell, op_proc, lam, mu, meta)


def diagnose(inst: Instance) -> dict:
    """Cheap structure diagnostics: workload bound (busiest cell), critical-path
    bound (longest job), dispatch UB, and their ratios. An instance is
    'bottleneck-dominated' when workload/UB is close to 1 and
    'coupling-dominated' when both ratios are well below 1."""
    from .bounds import heads_tails
    from .dispatch import best_dispatch
    h, q = heads_tails(inst)
    sch, rule, _ = best_dispatch(inst, q)
    ub = sch.makespan
    workload = max((sum(inst.p_min[a] for a in inst.ops_of_cell[c]) + len(inst.machines_of_cell[c]) - 1)
                   // max(1, len(inst.machines_of_cell[c])) for c in range(inst.n_cells))
    path = max(inst.release[j] + sum(inst.p_min[a] for a in ops) for j, ops in enumerate(inst.job_ops))
    return dict(ops=inst.n_ops, cells=inst.n_cells, machines=inst.n_machines, transitions=len(inst.transitions),
                ub_dispatch=ub, workload_lb=workload, path_lb=path,
                ratio_workload=round(workload / ub, 3), ratio_path=round(path / ub, 3))
