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
        # shadow copy of the LP for the SciPy/HiGHS fallback (true duals)
        self._rows: List[dict] = []          # row -> {var: coef}
        self._rbnd: List[Tuple[float, float]] = []
        self._vbnd: List[Tuple[float, float]] = []
        self._obj: List[float] = []
        self._sp = None                      # last SciPy result (fallback)

    # ---- building ----
    def add_var(self, lb: float = 0.0, ub: float = math.inf, obj: float = 0.0, name: str = "") -> int:
        idx = self.h.add_var()
        self.h.set_var_lower_bound(idx, lb); self.h.set_var_upper_bound(idx, ub)
        self.h.set_var_objective_coefficient(idx, obj)
        if name:
            self.h.set_var_name(idx, name)
        self._vbnd.append((lb, ub)); self._obj.append(float(obj))
        return idx

    def add_row(self, terms: Sequence[Tuple[int, float]], lb: float = -math.inf, ub: float = math.inf,
                name: str = "") -> int:
        r = self.h.add_linear_constraint()
        self.h.set_constraint_lower_bound(r, lb); self.h.set_constraint_upper_bound(r, ub)
        d = {}
        for var, val in terms:
            self.h.add_term_to_constraint(r, var, float(val))
            d[var] = d.get(var, 0.0) + float(val)
        if name:
            self.h.set_constraint_name(r, name)
        self._rows.append(d); self._rbnd.append((lb, ub))
        return r

    def set_coef(self, row: int, var: int, val: float) -> None:
        self.h.set_constraint_coefficient(row, var, val)
        self._rows[row][var] = float(val)

    def set_obj(self, var: int, val: float) -> None:
        self.h.set_var_objective_coefficient(var, val)
        self._obj[var] = float(val)

    def set_row_bounds(self, row: int, lb: float, ub: float) -> None:
        self.h.set_constraint_lower_bound(row, lb); self.h.set_constraint_upper_bound(row, ub)
        self._rbnd[row] = (lb, ub)

    def set_var_bounds(self, var: int, lb: float, ub: float) -> None:
        self.h.set_var_lower_bound(var, lb); self.h.set_var_upper_bound(var, ub)
        self._vbnd[var] = (lb, ub)

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
        self._sp = None
        for backend in [self.backend] + ([fallback] if fallback else []):
            if backend == "scipy":
                self.status = self._solve_scipy(time_limit)
                self.used_backend = backend
                if self.status == "OPTIMAL":
                    break
                continue
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

    def _solve_scipy(self, time_limit: Optional[float]) -> str:
        """Dual-capable fallback: SciPy's HiGHS interface (dual simplex) on the
        shadow copy. Duals follow the OR-Tools convention (minimisation:
        >= rows have dual >= 0, <= rows dual <= 0, equality rows free)."""
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        n = len(self._obj); c = np.array(self._obj) * (1.0 if self.minimize else -1.0)
        ei, ej, ev, be = [], [], [], []          # equality rows
        ui, uj, uv, bu = [], [], [], []          # <= rows
        self._sp_map = {}                        # row -> list of (kind, index, sign)
        for r, (d, (lb, ub)) in enumerate(zip(self._rows, self._rbnd)):
            refs = []
            if lb == ub:
                k = len(be)
                for j, v in d.items():
                    ei.append(k); ej.append(j); ev.append(v)
                be.append(lb); refs.append(("eq", k, 1.0))
            else:
                if ub < math.inf:
                    k = len(bu)
                    for j, v in d.items():
                        ui.append(k); uj.append(j); uv.append(v)
                    bu.append(ub); refs.append(("ub", k, 1.0))
                if lb > -math.inf:
                    k = len(bu)
                    for j, v in d.items():
                        ui.append(k); uj.append(j); uv.append(-v)
                    bu.append(-lb); refs.append(("ub", k, -1.0))
            self._sp_map[r] = refs
        A_eq = coo_matrix((ev, (ei, ej)), shape=(len(be), n)).tocsr() if be else None
        A_ub = coo_matrix((uv, (ui, uj)), shape=(len(bu), n)).tocsr() if bu else None
        bounds = [(lb if lb > -math.inf else None, ub if ub < math.inf else None) for lb, ub in self._vbnd]
        opts = {"presolve": True}
        if time_limit:
            opts["time_limit"] = float(time_limit)
        res = linprog(c, A_ub=A_ub, b_ub=np.array(bu) if bu else None, A_eq=A_eq,
                      b_eq=np.array(be) if be else None, bounds=bounds, method="highs-ds", options=opts)
        self._sp = res
        if res.status == 0:
            return "OPTIMAL"
        return {1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, "ABNORMAL")

    @property
    def objective(self) -> float:
        if self._sp is not None:
            return float(self._sp.fun) * (1.0 if self.minimize else -1.0)
        return float(self._solver.objective_value)

    def values(self) -> np.ndarray:
        if self._sp is not None:
            return np.array(self._sp.x)
        return np.array([self._solver.value(mb.Variable(self.h, i)) for i in range(self.num_vars)])

    def value(self, var: int) -> float:
        if self._sp is not None:
            return float(self._sp.x[var])
        return float(self._solver.value(mb.Variable(self.h, var)))

    def dual(self, row: int) -> float:
        if self._sp is not None:
            y = 0.0
            for kind, k, sgn in self._sp_map[row]:
                m = self._sp.eqlin.marginals[k] if kind == "eq" else self._sp.ineqlin.marginals[k]
                # marginal = d obj / d rhs; a >= row was negated, so its dual is -marginal
                y += sgn * float(m)
            return y * (1.0 if self.minimize else -1.0)
        return float(self._solver.dual_value(mb.LinearConstraint(self.h, index=row)))

    def duals(self, rows: Iterable[int]) -> List[float]:
        return [self.dual(r) for r in rows]
