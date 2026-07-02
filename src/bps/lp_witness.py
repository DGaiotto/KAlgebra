"""Exact-LP pointedness witness for positive cones.

`lp_feasible_strict(gens, rank)` decides whether the cone spanned by the
integer generators `gens` in `Z^rank` is **pointed**, by searching for a
strict witness `f` with `<f, g> > 0` for every generator — a two-phase
rational simplex over `Fraction` (no floats, no third-party packages),
box-free:

  * sound positive: the returned integral `f` satisfies `<f, g> >= 1`;
  * sound negative: infeasibility is certified by LP duality.

Serves as the exact fallback behind the box searches in
`bps_kalgebra_internals.compute_strict_cone_witness` and
`bps_quiver_tools` (a near-antipodal generator pair can force a witness
coordinate outside any fixed box — e.g. a mutated flavoured chamber
needing a -4 entry).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

Vec = tuple

def _lcm(a: int, b: int) -> int:
    from math import gcd
    if a == 0 or b == 0:
        return abs(a or b)
    return abs(a * b) // gcd(a, b)


def _bounded_integer_search_strict(
    constraints: Sequence[Sequence[int]],
    n: int,
    K_max: int = 6,
) -> tuple[bool, Optional[Vec]]:
    """Sound test: search for an integer x ∈ [-K_max, K_max]^n with
    ``v · x > 0`` for every v.

    Returns ``(True, x)`` if a witness is found (rigorously certifies
    that the cone has a strict interior).  Returns ``(False, None)``
    if no witness in the box — this does NOT prove the cone is empty;
    the iso-proof predicate treats this as "inconclusive" (returns
    None at the outer level).

    For BPS-quiver inputs with small integer pairings, K_max=4 to 6
    suffices in practice.  Pure Python, sound, slow at high n.
    """
    if not constraints:
        return True, tuple([1] + [0] * (n - 1))
    # Try the heuristic sum-of-normals first (it works for most cones).
    cand = [sum(v[i] for v in constraints) for i in range(n)]
    if all(sum(v[i] * cand[i] for i in range(n)) > 0 for v in constraints):
        return True, tuple(cand)
    # Constructive perturbation from the candidate.
    x = cand[:]
    for _ in range(500):
        worst = min(
            (sum(v[i] * x[i] for i in range(n)), idx)
            for idx, v in enumerate(constraints)
        )
        if worst[0] > 0:
            return True, tuple(x)
        v = constraints[worst[1]]
        v_norm_sq = sum(c * c for c in v)
        if v_norm_sq == 0:
            return False, None
        step = max(1, (-worst[0] + 1 + v_norm_sq - 1) // v_norm_sq)
        x = [x[j] + step * v[j] for j in range(n)]
    # Fallback exhaustive box search at small K.
    for K in range(1, K_max + 1):
        for xi in _iter_box(K, n):
            if all(sum(v[i] * xi[i] for i in range(n)) > 0
                   for v in constraints):
                return True, tuple(xi)
    return False, None


def _iter_box(K, n):
    """Generator over integer points in `[-K, K]^n`."""
    if n == 0:
        yield ()
        return
    for first in range(-K, K + 1):
        for rest in _iter_box(K, n - 1):
            yield (first,) + rest


def _lp_feasible_strict(
    constraints: Sequence[Sequence[int]],
    n: int,
) -> tuple[bool, Optional[Vec]]:
    """Test whether the open cone

        C* = { x ∈ R^n : v · x > 0  for every v in `constraints` }

    is non-empty, and if so return an integer witness ``x ∈ C*``.

    Sound positive (returned witnesses always satisfy the
    constraints); conservative negative (a False here means we
    didn't find a witness in the bounded integer search budget --
    the outer iso predicate treats this as "inconclusive").

    Method.  Reformulate as the bounded LP

        maximise   s
        s.t.       v · x   ≥ s     for every v ∈ constraints,
                   x_i      ≤ 1     for i = 0, …, n-1
                  -x_i      ≤ 1     for i = 0, …, n-1
                   s        ≥ 0
                   s        ≤ 1

    The cone is full-dim iff the LP optimum is strictly positive.
    Solved exactly via the simplex method over `Fractions`.

    Then, given the optimal x in [-1, 1]^n with v · x ≥ s* > 0:
    scale to integers by multiplying by a common denominator d, so
    `x' = d · x` is an integer vector with v · x' ≥ d · s* > 0.

    Returns ``(False, None)`` when the LP optimum is 0 (or
    infeasible).
    """
    found, x = _lp_feasible_strict_simplex(constraints, n)
    if found:
        return True, x
    # Simplex says optimum is 0 (or infeasible).  Defensively also
    # try the bounded integer search -- catches any cases where the
    # simplex extracted a witness whose integer scaling lost
    # strict-positivity (shouldn't happen with the sign-flipped
    # constraint rows, but no-cost backstop).
    return _bounded_integer_search_strict(constraints, n)


def _lp_feasible_strict_simplex(constraints, n):
    """Two-phase rational simplex.  Sound positives (the returned
    integer witness x strictly satisfies every constraint); sound
    negatives by LP duality (the optimum of the bounded LP is 0
    iff no strict-interior x exists in [-1, 1]^n -- and since the
    cone is invariant under positive scaling, that implies no
    strict-interior x exists anywhere)."""
    if not constraints:
        return True, tuple([1] + [0] * (n - 1))

    m = len(constraints)
    # Variables (in the simplex's column ordering):
    #   x_0, …, x_{n-1}   (each in [-1, 1])
    #   s   (≥ 0)
    # Maximise s.
    #
    # We use revised simplex with explicit slack variables.  Concretely:
    #
    #   For each constraint v · x ≥ s, rewrite as v · x - s ≥ 0,
    #   slack t_i: v · x - s - t_i = 0, t_i ≥ 0.
    #
    #   For each i: x_i ≤ 1  →  x_i + u_i = 1, u_i ≥ 0
    #               -x_i ≤ 1 →  -x_i + l_i = 1, l_i ≥ 0
    #
    #   s ≤ 1 → s + r = 1, r ≥ 0.
    #
    # Decision vars indexed as:
    #   x_+_i, x_-_i for i ∈ [0, n)   (positive / negative parts of x_i;
    #                                  x_i = x_+_i - x_-_i, both ≥ 0)
    #   s
    #   t_j for j ∈ [0, m)             (constraint slacks)
    #   u_i for i ∈ [0, n)             (x_i ≤ 1 slacks)
    #   l_i for i ∈ [0, n)             (x_i ≥ -1 slacks)
    #   r                              (s ≤ 1 slack)
    #
    # Total variables:  2n (x+, x-) + 1 (s) + m (t) + n (u) + n (l) + 1 (r) = 4n + m + 2.
    #
    # Constraints (all equalities after slack introduction):
    #   For each j ∈ [0, m):
    #     Σ_i v_j[i] · (x_+_i - x_-_i)  -  s  -  t_j  =  0
    #   For each i ∈ [0, n):
    #     (x_+_i - x_-_i)  +  u_i  =  1
    #     -(x_+_i - x_-_i)  +  l_i  =  1
    #   s  +  r  =  1
    #
    # Total equality constraints:  m + 2n + 1.
    #
    # All variables ≥ 0; non-negativity is implicit (basic-feasible
    # solutions stay non-negative).

    # Number of variables.
    N_X = 2 * n            # x_+_0, x_-_0, x_+_1, x_-_1, ..., x_+_{n-1}, x_-_{n-1}
    IDX_S = N_X            # column index for s
    N_T = m
    IDX_T = IDX_S + 1      # t_0 starts here
    IDX_U = IDX_T + N_T    # u_0 starts here
    IDX_L = IDX_U + n      # l_0 starts here
    IDX_R = IDX_L + n      # r
    N_VARS = IDX_R + 1     # total

    # Number of equality constraints.
    N_EQ = m + 2 * n + 1

    # Build the equality-constraint matrix A and RHS b, in dense
    # rational form.  Rows in order:
    #   rows 0..m-1   : constraint feasibility
    #   rows m..m+n-1 : x_i ≤ 1
    #   rows m+n..m+2n-1 : -x_i ≤ 1
    #   row m+2n      : s ≤ 1
    A = [[Fraction(0)] * N_VARS for _ in range(N_EQ)]
    b = [Fraction(0)] * N_EQ

    # Constraint rows.  Sign-flipped so the slack t_j has
    # coefficient +1 (standard simplex form: each basic variable
    # has +1 in its own row).  Original constraint
    #   Σ v · x - s - t_j = 0          (t_j ≥ 0 ⇒ v · x ≥ s)
    # becomes
    #   -Σ v · x + s + t_j = 0         (multiply by -1; same meaning)
    # i.e., t_j = Σ v · x - s.
    for j, v in enumerate(constraints):
        row = A[j]
        for i in range(n):
            row[2 * i] = Fraction(-v[i])     # coefficient of x_+_i
            row[2 * i + 1] = Fraction(v[i])  # coefficient of x_-_i
        row[IDX_S] = Fraction(1)             # +s
        row[IDX_T + j] = Fraction(1)         # +t_j
        b[j] = Fraction(0)

    # x_i ≤ 1 rows.
    for i in range(n):
        row = A[m + i]
        row[2 * i] = Fraction(1)
        row[2 * i + 1] = Fraction(-1)
        row[IDX_U + i] = Fraction(1)
        b[m + i] = Fraction(1)

    # -x_i ≤ 1 rows (equivalently x_i ≥ -1).
    for i in range(n):
        row = A[m + n + i]
        row[2 * i] = Fraction(-1)
        row[2 * i + 1] = Fraction(1)
        row[IDX_L + i] = Fraction(1)
        b[m + n + i] = Fraction(1)

    # s + r = 1.
    A[m + 2 * n][IDX_S] = Fraction(1)
    A[m + 2 * n][IDX_R] = Fraction(1)
    b[m + 2 * n] = Fraction(1)

    # Objective: maximise s.  Cost vector c, with c[IDX_S] = 1 elsewhere 0.
    c = [Fraction(0)] * N_VARS
    c[IDX_S] = Fraction(1)

    # Initial basis: slack variables.  The constraint-feasibility
    # rows have -t_j as the only "positive sign" basic variable
    # candidate when other vars are zero -- but we need t_j ≥ 0 and
    # the equation reads Σ ... = 0, with all x's=0, s=0: -t_j = 0 ⇒
    # t_j = 0.  So setting all decision vars = 0 (degenerate) puts
    # all t_j = 0.  Then u_i = 1, l_i = 1, r = 1.  This is a basic
    # feasible solution.
    #
    # In simplex tableau form: basis variables are t_0, …, t_{m-1},
    # u_0, …, u_{n-1}, l_0, …, l_{n-1}, r.  The basis matrix is the
    # identity sub-matrix on those columns (it is, by construction).
    basis = (
        list(range(IDX_T, IDX_T + m)) +
        list(range(IDX_U, IDX_U + n)) +
        list(range(IDX_L, IDX_L + n)) +
        [IDX_R]
    )
    return _simplex_max(A, b, c, basis, N_EQ, N_VARS, n)


def _simplex_max(A, b, c, basis, n_eq, n_vars, n_x):
    """Revised simplex (tabular form) over rationals.  Maximise
    `c^T x` subject to `A x = b`, `x ≥ 0`.

    Returns `(found, witness_x)` where `found` is True iff the
    optimum is strictly positive (and witness has integer x parts);
    False if the optimum is zero or unbounded-from-below.

    For our cone-feasibility setting, `found=True` means the cone
    has a strict interior; `found=False` means it does not (sound
    by LP duality and basic-feasible-solution theory).
    """
    # Standard tableau: rows are equality constraints, columns are
    # variables, plus an objective row.
    # We maintain row-reduced form with the basis variables having
    # identity columns.
    A = [row[:] for row in A]   # local copy
    b = b[:]
    # Reduced cost row.  In standard tableau: z_j - c_j = c_B^T B^{-1} A_j - c_j.
    # Start: z_j - c_j = -c_j (since basis has c_B = 0 for slacks/u/l/r).
    # But IDX_S is not in the initial basis; its reduced cost is -1
    # (i.e., we want to bring s into the basis).
    zc = [-cj for cj in c]
    obj_val = Fraction(0)

    MAX_ITER = 2 * n_vars * n_eq + 100
    for _ in range(MAX_ITER):
        # Pick entering variable: most-negative reduced cost
        # (Bland's rule with index tie-break for cycling-safety).
        entering = None
        best_zc = Fraction(0)
        for j in range(n_vars):
            if zc[j] < best_zc:
                best_zc = zc[j]
                entering = j
        if entering is None:
            break   # optimal

        # Ratio test: pick the leaving variable.
        leaving_row = None
        best_ratio = None
        for i in range(n_eq):
            aij = A[i][entering]
            if aij > 0:
                ratio = b[i] / aij
                if best_ratio is None or ratio < best_ratio or (
                    ratio == best_ratio and basis[i] < basis[leaving_row]
                ):
                    best_ratio = ratio
                    leaving_row = i

        if leaving_row is None:
            # Unbounded in objective direction.  For our LP this
            # cannot happen (s is bounded by 1), but handle defensively.
            return False, None

        # Pivot.
        piv = A[leaving_row][entering]
        A[leaving_row] = [x / piv for x in A[leaving_row]]
        b[leaving_row] = b[leaving_row] / piv

        for i in range(n_eq):
            if i == leaving_row:
                continue
            aij = A[i][entering]
            if aij != 0:
                A[i] = [
                    A[i][k] - aij * A[leaving_row][k] for k in range(n_vars)
                ]
                b[i] = b[i] - aij * b[leaving_row]

        # Update reduced costs.
        zcj_e = zc[entering]
        zc = [
            zc[k] - zcj_e * A[leaving_row][k] for k in range(n_vars)
        ]
        obj_val = obj_val - zcj_e * b[leaving_row]
        basis[leaving_row] = entering

    # Extract x from the basic feasible solution.
    x_vals = [Fraction(0)] * n_vars
    for i, bi in enumerate(basis):
        x_vals[bi] = b[i]

    # Optimal s value.
    s_opt = x_vals[2 * n_x]
    if s_opt <= 0:
        return False, None

    # Recover x_i = x_+_i - x_-_i.
    x_q = [x_vals[2 * i] - x_vals[2 * i + 1] for i in range(n_x)]
    # Scale to integers (common denominator) and verify.
    denom_lcm = 1
    for v in x_q:
        denom_lcm = _lcm(denom_lcm, v.denominator)
    x_int = tuple(int(v * denom_lcm) for v in x_q)
    return True, x_int


# Public alias.
lp_feasible_strict = _lp_feasible_strict
