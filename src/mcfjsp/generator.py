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
