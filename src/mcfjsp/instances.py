"""Instance model for the multi-cell flexible job shop with machine failures.

Operations are indexed globally, a = 0..n_ops-1, in job order. For each
operation `a`:
  op_job[a]   job index
  op_pos[a]   position within the job (0-based)
  op_cell[a]  cell that must process it
  op_proc[a]  dict {machine: processing time} over eligible machines (ints)
Jobs: release[j] (int), job_ops[j] = list of operation ids in order.
Machines: cell_of_machine[i]; machines_of_cell[c] = list of machine ids.
Failures: lam[i] >= 0 (rate per unit of busy time), mu[i] > 0 (repair rate).

All processing times, release dates and horizons are integers (Lemma EC.1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Instance:
    name: str
    n_machines: int
    cell_of_machine: List[int]
    release: List[int]
    job_ops: List[List[int]]
    op_job: List[int]
    op_pos: List[int]
    op_cell: List[int]
    op_proc: List[Dict[int, int]]
    lam: List[float] = field(default_factory=list)
    mu: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- derived -----------------------------------------------------
    def __post_init__(self) -> None:
        if not self.lam:
            self.lam = [0.0] * self.n_machines
        if not self.mu:
            self.mu = [1.0] * self.n_machines
        self.n_cells = max(self.cell_of_machine) + 1 if self.n_machines else 0
        self.machines_of_cell: List[List[int]] = [[] for _ in range(self.n_cells)]
        for i, c in enumerate(self.cell_of_machine):
            self.machines_of_cell[c].append(i)
        self.n_ops = len(self.op_job)
        self.n_jobs = len(self.job_ops)
        self.ops_of_cell: List[List[int]] = [[] for _ in range(self.n_cells)]
        for a in range(self.n_ops):
            self.ops_of_cell[self.op_cell[a]].append(a)
        # successor / predecessor within job (or -1)
        self.op_succ = [-1] * self.n_ops
        self.op_pred = [-1] * self.n_ops
        for ops in self.job_ops:
            for k in range(len(ops) - 1):
                self.op_succ[ops[k]] = ops[k + 1]
                self.op_pred[ops[k + 1]] = ops[k]
        # inter-cell transitions: a in T iff succ exists and in another cell
        self.transitions = [a for a in range(self.n_ops)
                            if self.op_succ[a] >= 0 and self.op_cell[self.op_succ[a]] != self.op_cell[a]]
        self.is_transition = [False] * self.n_ops
        for a in self.transitions:
            self.is_transition[a] = True
        self.p_min = [min(pr.values()) for pr in self.op_proc]
        self.validate()

    def validate(self) -> None:
        for a in range(self.n_ops):
            c = self.op_cell[a]
            assert self.op_proc[a], f"operation {a} has no eligible machine"
            for i, p in self.op_proc[a].items():
                assert self.cell_of_machine[i] == c, f"op {a}: machine {i} not in cell {c}"
                assert isinstance(p, int) and p > 0, f"op {a}: processing time must be positive int"
        for j, ops in enumerate(self.job_ops):
            assert isinstance(self.release[j], int) and self.release[j] >= 0
            for k, a in enumerate(ops):
                assert self.op_job[a] == j and self.op_pos[a] == k

    # ---- convenience -------------------------------------------------
    def total_work_lb(self) -> int:
        return sum(self.p_min)

    def failure_inflated(self) -> "Instance":
        """Instance with p~ = p (1 + lam/mu), rounded UP to integers.

        Rounding up keeps p~ >= E[D]; the mean-value bound (Prop. 2) requires
        p~ <= E[D] per path to remain a lower bound, so this variant is used
        only as a *heuristic* planning instance. For a valid mean-value bound
        use `failure_inflated(round_down=True)`.
        """
        return self.failure_inflated_rounded(round_down=False)

    def failure_inflated_rounded(self, round_down: bool = True, kappa: int = 1) -> "Instance":
        import math
        proc = []
        for a in range(self.n_ops):
            d = {}
            for i, p in self.op_proc[a].items():
                v = kappa * p * (1.0 + self.lam[i] / self.mu[i])
                d[i] = max(1, int(math.floor(v)) if round_down else int(math.ceil(v)))
            proc.append(d)
        rel = [int(math.floor(kappa * r)) if round_down else int(math.ceil(kappa * r)) for r in self.release]
        return Instance(self.name + "_inflated", self.n_machines, list(self.cell_of_machine), rel,
                        [list(o) for o in self.job_ops], list(self.op_job), list(self.op_pos),
                        list(self.op_cell), proc, [0.0] * self.n_machines, list(self.mu),
                        dict(self.meta, kappa=kappa, inflated=True))

    def with_durations(self, dur: Dict[int, Dict[int, int]], release: Optional[List[int]] = None,
                       name_suffix: str = "_scen") -> "Instance":
        """Copy with processing times replaced (e.g. a rounded scenario)."""
        proc = [dict(dur[a]) for a in range(self.n_ops)]
        rel = list(release) if release is not None else list(self.release)
        return Instance(self.name + name_suffix, self.n_machines, list(self.cell_of_machine), rel,
                        [list(o) for o in self.job_ops], list(self.op_job), list(self.op_pos),
                        list(self.op_cell), proc, [0.0] * self.n_machines, list(self.mu), dict(self.meta))

    # ---- (de)serialization ------------------------------------------
    def to_json(self) -> dict:
        return {
            "name": self.name,
            "n_machines": self.n_machines,
            "cell_of_machine": self.cell_of_machine,
            "lam": self.lam,
            "mu": self.mu,
            "jobs": [{"release": self.release[j],
                      "ops": [{"cell": self.op_cell[a],
                               "proc": {str(i): p for i, p in self.op_proc[a].items()}}
                              for a in ops]} for j, ops in enumerate(self.job_ops)],
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "Instance":
        release, job_ops, op_job, op_pos, op_cell, op_proc = [], [], [], [], [], []
        for j, job in enumerate(d["jobs"]):
            release.append(int(job["release"]))
            ops = []
            for k, op in enumerate(job["ops"]):
                a = len(op_job)
                ops.append(a)
                op_job.append(j); op_pos.append(k); op_cell.append(int(op["cell"]))
                op_proc.append({int(i): int(p) for i, p in op["proc"].items()})
            job_ops.append(ops)
        return Instance(d["name"], int(d["n_machines"]), list(map(int, d["cell_of_machine"])), release,
                        job_ops, op_job, op_pos, op_cell, op_proc,
                        list(map(float, d.get("lam", []))), list(map(float, d.get("mu", []))),
                        d.get("meta", {}))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path) as f:
            return Instance.from_json(json.load(f))


# ---------------------------------------------------------------------
# Standard FJSP text format (Brandimarte / Hurink / Dauzère-Pérès as in
# SchedulingLab/fjsp-instances): first line "<jobs> <machines>", then one
# line per job: n_ops, then for each op: n_machines, then pairs (machine, p).
# Machine indices start at 0 in that repository.
# ---------------------------------------------------------------------
def read_fjsp_text(path: str, name: Optional[str] = None, machine_base: int = 0) -> Instance:
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)
    n_jobs = int(next(it)); n_machines = int(next(it))
    release, job_ops, op_job, op_pos, op_cell, op_proc = [], [], [], [], [], []
    for j in range(n_jobs):
        n_ops = int(next(it))
        ops = []
        for k in range(n_ops):
            nm = int(next(it))
            proc = {}
            for _ in range(nm):
                i = int(next(it)) - machine_base; p = int(next(it))
                proc[i] = p
            a = len(op_job)
            ops.append(a)
            op_job.append(j); op_pos.append(k); op_cell.append(0); op_proc.append(proc)
        job_ops.append(ops); release.append(0)
    # single cell (the original FJSP); use partition.apply_partition to split.
    return Instance(name or path, n_machines, [0] * n_machines, release, job_ops,
                    op_job, op_pos, op_cell, op_proc, meta={"source": path, "cells": 1})


def read_jsp_taillard(path: str, name: Optional[str] = None) -> Instance:
    """Classical job shop in Taillard/JSPLIB format (first line: n_jobs
    n_machines; then one line per job with (machine, duration) pairs).  One
    cell per machine: the plant-level problem is exactly J||Cmax, so published
    bounds calibrate the cell-decomposed bound engine."""
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)
    n_jobs = int(next(it)); n_machines = int(next(it))
    release, job_ops, op_job, op_pos, op_cell, op_proc = [], [], [], [], [], []
    for j in range(n_jobs):
        ops = []
        for k in range(n_machines):
            i = int(next(it)); p = int(next(it))
            a = len(op_job); ops.append(a)
            op_job.append(j); op_pos.append(k); op_cell.append(i); op_proc.append({i: p})
        job_ops.append(ops); release.append(0)
    return Instance(name or path, n_machines, list(range(n_machines)), release, job_ops,
                    op_job, op_pos, op_cell, op_proc, meta={"source": path, "cells": n_machines, "format": "taillard"})
