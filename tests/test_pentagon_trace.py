"""Test the rho^2-twisted trace on the pentagon algebra.

The trace is determined uniquely by:
  * cyclicity Tr(uv) = Tr(rho^2(v) u),
  * Z[[q]]-valuedness (orthonormality at q^0 implies it),
  * normalization Tr(1) = 1.

Verifies that the resulting Tr(L_0)(q) matches the cluster Schur-index
value computed via bps_quiver_tools.CoulombAlgebra.schur_index, namely

    Tr(L_0)(q) = -q + q^5 - q^9 - q^11 + q^13 + 2 q^15 + ...

Approach: use truncation in basis-degree D and Taylor order K_T.  Build
the linear system in {Tr[e]_l} variables, solve, read off Tr[(0,1,0)]_l.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction
from pentagon_algebra import PentagonAlgebra as PA


D_BASIS = 3
K_T = 6


def _basis_keys():
    keys = [(0, 0, 0)]
    for i in range(5):
        for a in range(1, D_BASIS + 1):
            keys.append((i, a, 0))
    for i in range(5):
        for a in range(1, D_BASIS):
            for b in range(1, D_BASIS - a + 1):
                keys.append((i, a, b))
    return keys


def _basis_deg(k):
    return k[1] + k[2]


KEYS = _basis_keys()
KEY_IDX = {k: i for i, k in enumerate(KEYS)}
N_BASIS = len(KEYS)
N_VARS = N_BASIS * (K_T + 1)


def _var(e_idx, l):
    return e_idx * (K_T + 1) + l


def _cyclicity_equations():
    eqs = []
    for u_key in KEYS:
        u = PA.basis(*u_key)
        for v_key in KEYS:
            if _basis_deg(u_key) + _basis_deg(v_key) > D_BASIS:
                continue
            v = PA.basis(*v_key)
            diff = u * v - v.rho().rho() * u
            if diff.is_zero():
                continue
            d_terms = list(diff._terms.items())
            qs = []
            for _, lp in d_terms:
                qs.extend(lp._coeffs.keys())
            if not qs:
                continue
            n_lo = min(qs)
            for n in range(n_lo, K_T + 1):
                eq = {}
                skip = False
                for e, lp in d_terms:
                    if skip:
                        break
                    if e not in KEY_IDX:
                        skip = True
                        break
                    ei = KEY_IDX[e]
                    for kpow, c in lp._coeffs.items():
                        if kpow > n:
                            continue
                        l = n - kpow
                        if l > K_T:
                            skip = True
                            break
                        v_idx = _var(ei, l)
                        eq[v_idx] = eq.get(v_idx, Fraction(0)) + Fraction(c)
                if skip:
                    continue
                eq = {k: v for k, v in eq.items() if v != 0}
                if eq:
                    eqs.append((eq, Fraction(0)))
    return eqs


def _normalization_equations():
    eqs = []
    e_id = KEY_IDX[(0, 0, 0)]
    eqs.append(({_var(e_id, 0): Fraction(1)}, Fraction(1)))
    for l in range(1, K_T + 1):
        eqs.append(({_var(e_id, l): Fraction(1)}, Fraction(0)))
    return eqs


def _gauss(equations):
    nrows = len(equations)
    M = [[Fraction(0)] * (N_VARS + 1) for _ in range(nrows)]
    for r, (eq, rhs) in enumerate(equations):
        for c, v in eq.items():
            M[r][c] = v
        M[r][N_VARS] = rhs
    pivot_cols = []
    rr = 0
    for c in range(N_VARS):
        piv = None
        for r in range(rr, nrows):
            if M[r][c] != 0:
                piv = r
                break
        if piv is None:
            continue
        M[rr], M[piv] = M[piv], M[rr]
        pv = M[rr][c]
        if pv != 1:
            M[rr] = [x / pv for x in M[rr]]
        for r in range(nrows):
            if r == rr or M[r][c] == 0:
                continue
            v = M[r][c]
            M[r] = [a - v * b for a, b in zip(M[r], M[rr])]
        pivot_cols.append(c)
        rr += 1
        if rr == nrows:
            break
    return pivot_cols, M


def _solved_value(pivot_cols, M, var):
    if var not in pivot_cols:
        return None
    r = pivot_cols.index(var)
    pivot_set = set(pivot_cols)
    for c in range(N_VARS):
        if c in pivot_set:
            continue
        if M[r][c] != 0:
            return None
    return M[r][N_VARS]


def test_trace_L0_matches_cluster():
    """Tr(L_0)(q) at q^0..q^K_T should equal cluster value."""
    cy = _cyclicity_equations()
    norm = _normalization_equations()
    pivot_cols, M = _gauss(cy + norm)

    L0_idx = KEY_IDX[(0, 1, 0)]
    expected = {0: 0, 1: -1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}
    for l in range(K_T + 1):
        v = _solved_value(pivot_cols, M, _var(L0_idx, l))
        if v is None:
            raise AssertionError(f"Tr(L_0)_{l} not pinned uniquely")
        assert v == Fraction(expected[l]), (
            f"Tr(L_0)_{l} = {v}, expected {expected[l]}"
        )


def test_trace_rho_invariant():
    """Tr(L_i) is i-independent (rho-invariance from cyclicity + Tr ρ²-twisted)."""
    cy = _cyclicity_equations()
    norm = _normalization_equations()
    pivot_cols, M = _gauss(cy + norm)
    for i in range(5):
        for l in range(K_T + 1):
            v_i = _solved_value(pivot_cols, M, _var(KEY_IDX[(i, 1, 0)], l))
            v_0 = _solved_value(pivot_cols, M, _var(KEY_IDX[(0, 1, 0)], l))
            assert v_i is not None
            assert v_0 is not None
            assert v_i == v_0, f"Tr(L_{i})_{l}={v_i} != Tr(L_0)_{l}={v_0}"


if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception:
            failures += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} failure(s).")
        sys.exit(1)
    print("\nAll trace tests passed.")
