"""Cell partition of benchmark FJSP instances (Section 6, 'Instances (a)').

Rule (deterministic, documented in the paper):
  1. Build the machine co-eligibility graph: weight(i, i') = number of
     operations eligible on both i and i'.
  2. Greedy agglomerative merging of machine groups, always merging the pair
     of groups with the largest total inter-group weight divided by the
     product of group sizes, subject to a size cap ceil(m/C) (relaxed by one
     only when no merge is possible), until C
     groups remain. Ties are broken by lowest machine index.
  3. Operations are assigned to cells by a load-balancing rule: operations
     are taken in non-increasing order of p_min (ties by index); each is
     assigned, among the cells that contain at least one of its eligible
     machines, to the cell with the smallest load per machine after adding
     p_min (ties: most eligible machines, then lowest cell index). Its
     eligible set is restricted to that cell.

Restricting eligibility can only increase the optimum, so literature lower
bounds for the original instance remain valid lower bounds for the variant.
"""
from __future__ import annotations

import math
from typing import List

from .instances import Instance


def partition_machines(inst: Instance, n_cells: int) -> List[int]:
    m = inst.n_machines
    if n_cells >= m:
        return list(range(m))
    cap = math.ceil(m / n_cells)
    w = [[0] * m for _ in range(m)]
    for pr in inst.op_proc:
        ms = sorted(pr.keys())
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                w[ms[x]][ms[y]] += 1
                w[ms[y]][ms[x]] += 1
    groups = [[i] for i in range(m)]
    while len(groups) > n_cells:
        best = None
        for g1 in range(len(groups)):
            for g2 in range(g1 + 1, len(groups)):
                if len(groups[g1]) + len(groups[g2]) > cap:
                    continue
                tot = sum(w[i][k] for i in groups[g1] for k in groups[g2])
                score = tot / (len(groups[g1]) * len(groups[g2]))
                key = (score, -min(groups[g1] + groups[g2]))
                if best is None or key > best[0]:
                    best = (key, g1, g2)
        if best is None:  # cap prevents all merges: relax the cap
            cap += 1
            continue
        _, g1, g2 = best
        groups[g1] = sorted(groups[g1] + groups[g2])
        del groups[g2]
    groups.sort(key=lambda g: g[0])
    cell_of = [0] * m
    for c, g in enumerate(groups):
        for i in g:
            cell_of[i] = c
    return cell_of


def apply_partition(inst: Instance, cell_of_machine: List[int], name: str | None = None) -> Instance:
    n_cells = max(cell_of_machine) + 1
    n_mach = [0] * n_cells
    for c in cell_of_machine:
        n_mach[c] += 1
    load = [0.0] * n_cells
    op_cell = [0] * inst.n_ops
    op_proc: List[dict] = [dict() for _ in range(inst.n_ops)]
    order = sorted(range(inst.n_ops), key=lambda a: (-inst.p_min[a], a))
    for a in order:
        count = [0] * n_cells
        for i in inst.op_proc[a]:
            count[cell_of_machine[i]] += 1
        best = None
        for c in range(n_cells):
            if count[c] == 0:
                continue
            key = ((load[c] + inst.p_min[a]) / n_mach[c], -count[c], c)
            if best is None or key < best[0]:
                best = (key, c)
        c = best[1]
        op_cell[a] = c
        op_proc[a] = {i: p for i, p in inst.op_proc[a].items() if cell_of_machine[i] == c}
        load[c] += min(op_proc[a].values())
    meta = dict(inst.meta, cells=n_cells, partition="coelig-greedy/load-balance", base=inst.name)
    return Instance(name or f"{inst.name}_c{n_cells}", inst.n_machines, list(cell_of_machine),
                    list(inst.release), [list(o) for o in inst.job_ops], list(inst.op_job),
                    list(inst.op_pos), op_cell, op_proc, list(inst.lam), list(inst.mu), meta)


def multicell_variant(inst: Instance, n_cells: int) -> Instance:
    return apply_partition(inst, partition_machines(inst, n_cells))
