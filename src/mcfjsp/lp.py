"""Thin LP wrapper over OR-Tools model_builder (HiGHS backend, GLOP fallback).

Provides what column generation needs: add variables/rows incrementally,
change objective coefficients, re-solve, read primal values and row duals.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ortools.linear_solver.python import model_builder as mb


class LP:
    def __init__(self, backend: str = "glop", minimize: bool = True):
        self.model = mb.Model()
        self.h = self.model.helper
        self.backend = backend
        self.minimize = minimize
        self.h.set_maximize(not minimize)
        self._solver = None
        self.status = None

    # ---- building ----
    def add_var(self, lb: float = 0.0, ub: float = math.inf, obj: float = 0.0, name: str = "") -> int:
        idx = self.h.add_var()
        self.h.set_var_lower_bound(idx, lb); self.h.set_var_upper_bound(idx, ub)
        self.h.set_var_objective_coefficient(idx, obj)
        if name:
            self.h.set_var_name(idx, name)
        return idx

    def add_row(self, terms: Sequence[Tuple[int, float]], lb: float = -math.inf, ub: float = math.inf,
                name: str = "") -> int:
        r = self.h.add_linear_constraint()
        self.h.set_constraint_lower_bound(r, lb); self.h.set_constraint_upper_bound(r, ub)
        for var, val in terms:
            self.h.add_term_to_constraint(r, var, float(val))
        if name:
            self.h.set_constraint_name(r, name)
        return r

    def set_coef(self, row: int, var: int, val: float) -> None:
        self.h.set_constraint_coefficient(row, var, val)

    def set_obj(self, var: int, val: float) -> None:
        self.h.set_var_objective_coefficient(var, val)

    def set_row_bounds(self, row: int, lb: float, ub: float) -> None:
        self.h.set_constraint_lower_bound(row, lb); self.h.set_constraint_upper_bound(row, ub)

    def set_var_bounds(self, var: int, lb: float, ub: float) -> None:
        self.h.set_var_lower_bound(var, lb); self.h.set_var_upper_bound(var, ub)

    @property
    def num_vars(self) -> int:
        return self.h.num_variables()

    @property
    def num_rows(self) -> int:
        return self.h.num_constraints()

    # ---- solving ----
    def solve(self, time_limit: Optional[float] = None, fallback: Optional[str] = None) -> str:
        """Solve with the configured backend; on a non-optimal status, retry
        with `fallback` (e.g. 'highs'). NOTE: in OR-Tools 9.15 the HiGHS
        wrapper's dual_value() returns row activities, so duals must be read
        with GLOP only; primal values and objective are fine with both."""
        for backend in [self.backend] + ([fallback] if fallback else []):
            s = mb.Solver(backend)
            if not s.solver_is_supported():
                continue
            s.enable_output(False)
            if time_limit:
                s.set_time_limit_in_seconds(time_limit)
            st = s.solve(self.model)
            self._solver = s
            self.status = st.name if hasattr(st, "name") else str(st)
            self.used_backend = backend
            if self.status == "OPTIMAL":
                break
        return self.status

    @property
    def objective(self) -> float:
        return float(self._solver.objective_value)

    def values(self) -> np.ndarray:
        try:
            return np.array(self._solver.values(self.model.get_variables()) if False else
                            [self._solver.value(mb.Variable(self.h, i)) for i in range(self.num_vars)])
        except Exception:
            return np.array([self._solver.value(mb.Variable(self.h, i)) for i in range(self.num_vars)])

    def value(self, var: int) -> float:
        return float(self._solver.value(mb.Variable(self.h, var)))

    def dual(self, row: int) -> float:
        return float(self._solver.dual_value(mb.LinearConstraint(self.h, index=row)))

    def duals(self, rows: Iterable[int]) -> List[float]:
        return [self.dual(r) for r in rows]
