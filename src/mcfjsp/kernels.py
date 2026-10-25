"""Compiled propagation kernels (numba). Pure-Python fallbacks keep the
package usable without numba (slower)."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def edge_finding(es, lct, p):
    """O(n^2) edge-finding on one machine (Carlier & Pinson; Baptiste et al.).
    es, lct, p: int64 arrays. Returns (ok, new_es): ok == False means a proven
    overload; new_es >= es are the tightened earliest starts."""
    n = es.shape[0]
    new_es = es.copy()
    if n <= 1:
        return True, new_es
    by_d = np.argsort(lct)
    by_h = np.argsort(es)
    in_omega = np.zeros(n, dtype=np.bool_)
    tail_sum = np.zeros(n, dtype=np.int64)
    for k_idx in range(n):
        k = by_d[k_idx]
        in_omega[k] = True
        D = lct[k]
        suffix = 0
        for idx in range(n - 1, -1, -1):
            a = by_h[idx]
            if in_omega[a]:
                suffix += p[a]
                tail_sum[a] = suffix
        C = -(1 << 60); est = (1 << 60); P = 0
        for idx in range(n):
            a = by_h[idx]
            if in_omega[a]:
                v = es[a] + tail_sum[a]
                if v > C:
                    C = v
                if es[a] < est:
                    est = es[a]
                P += p[a]
        if C > D:
            return False, new_es
        for j in range(k_idx + 1, n):
            i = by_d[j]
            m = es[i] if es[i] < est else est
            if m + P + p[i] > D and C > new_es[i]:
                new_es[i] = C
    return True, new_es


@njit(cache=True)
def energetic_ok(es, lct, pmin, m):
    """Cumulative energetic check: for all thresholds (H, D),
    sum{pmin_a : es_a >= H, lct_a <= D} <= m (D - H). lct here = ls + pmin."""
    n = es.shape[0]
    order_h = np.argsort(es)
    order_d = np.argsort(lct)
    # for each distinct H (descending), accumulate ops with es >= H in deadline order
    sel = np.zeros(n, dtype=np.bool_)
    idx_h = n - 1
    H_vals = np.unique(es)
    for hi in range(H_vals.shape[0] - 1, -1, -1):
        H = H_vals[hi]
        while idx_h >= 0 and es[order_h[idx_h]] >= H:
            sel[order_h[idx_h]] = True
            idx_h -= 1
        e = 0
        for j in range(n):
            a = order_d[j]
            if sel[a]:
                e += pmin[a]
                if e > m * (lct[a] - H):
                    return False
    return True
