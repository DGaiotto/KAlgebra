"""
cone_algebra.py
================

Closed-form cone-monomial algebra for `U1HexagonKAlg` — bar-invariant canonical
basis with explicit multiplication table.  **No BPSKAlgebra dependency.**

Mathematical setup
------------------
The u(1)-gauged hexagon has 11 multiplicative generators:
  L_{1, 0..5}, L_{2, 0..2}, E, E^{-1}    (E = electric quantum-torus, charge (1,0,1,0))

There are 14 maximal subsets of pairwise-fq-commuting generators ("cones"),
each of the form {E, E^{-1}, three L-letters}.  Every charge γ ∈ Z⁴ admits a
non-negative-integer decomposition in some cone (verified on `[-3,3]⁴`).

A **cone monomial** at charge γ is the canonical basis element at γ.  Encoded
as `(factors, e_E)` where `factors` is a sorted tuple `((a, i, exp), ...)` of
L-generators all belonging to one cone, and `e_E ∈ Z`.  Implicitly carries an
`fq^{T_half}` prefactor (with T_half computed from the cone-internal
commutators + E-letter pairings) that makes the cone monomial bar-invariant
under `q → 1/q, factor-reverse`.

Multiplication table
--------------------
`MULT_TABLE_LL[(L_a, L_b)] = [(fq_pow, sign, factors, e_E), ...]` gives the
cone-monomial expansion of the algebra product `L_a · L_b`.  Derived once by
exhaustively multiplying via BPS, then frozen as a Python literal.

Recursive multiply
------------------
`multiply_cone_monomials(M_a, M_b)` reduces an algebra word obtained by
flattening both inputs:
  - apply MULT_TABLE_LL on any non-canonical adjacent pair (unsorted, or
    multi-term Plücker);
  - bubble together non-adjacent non-fq-commuting pairs via fq-commute swaps;
  - collect when the word is in cone-canonical form (all adjacent pairs are
    sorted, single-term, canonical-shape).

Verified against BPS: 81/81 single-letter, 729/729 + 729/729 left/right-assoc
3-letter, 300/300 4-letter sample.
"""
from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from collections import Counter
from laurent_poly import LaurentPoly


# ----------------------------------------------------------------------
# Static data: charges, cones, generator-level tables.
# ----------------------------------------------------------------------

MU_CHARGE: tuple[int, int, int, int] = (1, 0, 1, 0)  # E charge

L1_CHARGES: dict[int, tuple[int, int, int, int]] = {
    0: (0,  1,  0,  1),
    1: (0, -1,  0, -1),
    2: (-1, 0, -1,  1),
    3: (1,  0,  1, -1),
    4: (-1, 0, -2,  1),
    5: (1, -1,  2, -1),
}

L2_CHARGES: dict[int, tuple[int, int, int, int]] = {
    0: (1,  0,  0,  0),
    1: (-1,-1, -1,  0),
    2: (0,  0,  1,  0),
    3: (0,  0, -1,  0),
    4: (0, -1,  0,  0),
    5: (-1, 0,  0,  0),
}


def charge(letter: tuple[int, int]) -> tuple[int, int, int, int]:
    """Tropical charge of a canonical-range L-letter (a, i)."""
    a, i = letter
    if a == 1: return L1_CHARGES[i % 6]
    if a == 2: return L2_CHARGES[i % 6]
    raise ValueError(f"a must be 1 or 2, got {a}")


# B(L_l, E) = fq-power when L_l · E (single-term q-commute)
MU_LETTER_QPOWER: dict[tuple[int, int], int] = {
    (1, 0): -1, (1, 1): +1, (1, 2): -1,
    (1, 3): +1, (1, 4): -1, (1, 5): +1,
    (2, 0):  0, (2, 1):  0, (2, 2):  0,
}


# 14 maximal q-commuting subsets (each: 3 L-letters; E and E^{-1} implicit).
CONES: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = [
    ((1,1),(1,3),(1,5)), ((1,1),(1,3),(2,0)), ((1,3),(1,5),(2,2)),
    ((1,0),(1,3),(2,2)), ((1,0),(1,3),(2,0)), ((1,1),(1,5),(2,1)),
    ((1,1),(1,4),(2,1)), ((1,1),(1,4),(2,0)), ((1,2),(1,5),(2,1)),
    ((1,2),(1,4),(2,1)), ((1,2),(1,5),(2,2)), ((1,0),(1,2),(2,2)),
    ((1,0),(1,2),(1,4)), ((1,0),(1,4),(2,0)),
]


# ----------------------------------------------------------------------
# Frozen MULT_TABLE_LL: 81 entries L_a · L_b → Σ fq^c · cone_monomial.
# Each entry: (fq_pow, sign_coef, factors, e_E).  Derived one-time from BPS
# on the gauged-hexagon quiver, then frozen here.  No runtime BPS dep.
# ----------------------------------------------------------------------

MULT_TABLE_LL: dict = {
    ((1, 0), (1, 0)): [(0, 1, ((1, 0, 2),), 0)],
    ((1, 0), (1, 1)): [(0, 1, (), 0), (-1, 1, ((2, 0, 1),), 0)],
    ((1, 0), (1, 2)): [(1, 1, ((1, 0, 1), (1, 2, 1)), 0)],
    ((1, 0), (1, 3)): [(-1, 1, ((1, 0, 1), (1, 3, 1)), 0)],
    ((1, 0), (1, 4)): [(1, 1, ((1, 0, 1), (1, 4, 1)), 0)],
    ((1, 0), (1, 5)): [(-1, 1, ((2, 2, 1),), 1), (-2, 1, (), 2)],
    ((1, 0), (2, 0)): [(-1, 1, ((1, 0, 1), (2, 0, 1)), 0)],
    ((1, 0), (2, 1)): [(1, 1, ((1, 2, 1),), 0), (0, 1, ((1, 4, 1),), 1)],
    ((1, 0), (2, 2)): [(0, 1, ((1, 0, 1), (2, 2, 1)), 0)],
    ((1, 1), (1, 0)): [(0, 1, (), 0), (1, 1, ((2, 0, 1),), 0)],
    ((1, 1), (1, 1)): [(0, 1, ((1, 1, 2),), 0)],
    ((1, 1), (1, 2)): [(-1, 1, ((2, 1, 1),), 0), (0, 1, (), 0)],
    ((1, 1), (1, 3)): [(1, 1, ((1, 1, 1), (1, 3, 1)), 0)],
    ((1, 1), (1, 4)): [(-1, 1, ((1, 1, 1), (1, 4, 1)), 0)],
    ((1, 1), (1, 5)): [(1, 1, ((1, 1, 1), (1, 5, 1)), 0)],
    ((1, 1), (2, 0)): [(1, 1, ((1, 1, 1), (2, 0, 1)), 0)],
    ((1, 1), (2, 1)): [(-1, 1, ((1, 1, 1), (2, 1, 1)), 0)],
    ((1, 1), (2, 2)): [(0, 1, ((1, 5, 1),), -1), (1, 1, ((1, 3, 1),), 0)],
    ((1, 2), (1, 0)): [(-1, 1, ((1, 0, 1), (1, 2, 1)), 0)],
    ((1, 2), (1, 1)): [(1, 1, ((2, 1, 1),), 0), (0, 1, (), 0)],
    ((1, 2), (1, 2)): [(0, 1, ((1, 2, 2),), 0)],
    ((1, 2), (1, 3)): [(0, 1, (), 0), (-1, 1, ((2, 2, 1),), 0)],
    ((1, 2), (1, 4)): [(1, 1, ((1, 2, 1), (1, 4, 1)), 0)],
    ((1, 2), (1, 5)): [(-1, 1, ((1, 2, 1), (1, 5, 1)), 0)],
    ((1, 2), (2, 0)): [(0, 1, ((1, 4, 1),), 1), (-1, 1, ((1, 0, 1),), 0)],
    ((1, 2), (2, 1)): [(1, 1, ((1, 2, 1), (2, 1, 1)), 0)],
    ((1, 2), (2, 2)): [(-1, 1, ((1, 2, 1), (2, 2, 1)), 0)],
    ((1, 3), (1, 0)): [(1, 1, ((1, 0, 1), (1, 3, 1)), 0)],
    ((1, 3), (1, 1)): [(-1, 1, ((1, 1, 1), (1, 3, 1)), 0)],
    ((1, 3), (1, 2)): [(0, 1, (), 0), (1, 1, ((2, 2, 1),), 0)],
    ((1, 3), (1, 3)): [(0, 1, ((1, 3, 2),), 0)],
    ((1, 3), (1, 4)): [(-1, 1, ((2, 0, 1),), -1), (0, 1, (), 0)],
    ((1, 3), (1, 5)): [(1, 1, ((1, 3, 1), (1, 5, 1)), 0)],
    ((1, 3), (2, 0)): [(0, 1, ((1, 3, 1), (2, 0, 1)), 0)],
    ((1, 3), (2, 1)): [(-1, 1, ((1, 1, 1),), 0), (0, 1, ((1, 5, 1),), -1)],
    ((1, 3), (2, 2)): [(1, 1, ((1, 3, 1), (2, 2, 1)), 0)],
    ((1, 4), (1, 0)): [(-1, 1, ((1, 0, 1), (1, 4, 1)), 0)],
    ((1, 4), (1, 1)): [(1, 1, ((1, 1, 1), (1, 4, 1)), 0)],
    ((1, 4), (1, 2)): [(-1, 1, ((1, 2, 1), (1, 4, 1)), 0)],
    ((1, 4), (1, 3)): [(1, 1, ((2, 0, 1),), -1), (0, 1, (), 0)],
    ((1, 4), (1, 4)): [(0, 1, ((1, 4, 2),), 0)],
    ((1, 4), (1, 5)): [(-1, 1, ((2, 1, 1),), 1), (0, 1, (), 0)],
    ((1, 4), (2, 0)): [(0, 1, ((1, 4, 1), (2, 0, 1)), 0)],
    ((1, 4), (2, 1)): [(0, 1, ((1, 4, 1), (2, 1, 1)), 0)],
    ((1, 4), (2, 2)): [(-1, 1, ((1, 2, 1),), 0), (0, 1, ((1, 0, 1),), -1)],
    ((1, 5), (1, 0)): [(1, 1, ((2, 2, 1),), 1), (2, 1, (), 2)],
    ((1, 5), (1, 1)): [(-1, 1, ((1, 1, 1), (1, 5, 1)), 0)],
    ((1, 5), (1, 2)): [(1, 1, ((1, 2, 1), (1, 5, 1)), 0)],
    ((1, 5), (1, 3)): [(-1, 1, ((1, 3, 1), (1, 5, 1)), 0)],
    ((1, 5), (1, 4)): [(1, 1, ((2, 1, 1),), 1), (0, 1, (), 0)],
    ((1, 5), (1, 5)): [(0, 1, ((1, 5, 2),), 0)],
    ((1, 5), (2, 0)): [(1, 1, ((1, 1, 1),), 2), (0, 1, ((1, 3, 1),), 1)],
    ((1, 5), (2, 1)): [(0, 1, ((1, 5, 1), (2, 1, 1)), 0)],
    ((1, 5), (2, 2)): [(0, 1, ((1, 5, 1), (2, 2, 1)), 0)],
    ((2, 0), (1, 0)): [(1, 1, ((1, 0, 1), (2, 0, 1)), 0)],
    ((2, 0), (1, 1)): [(-1, 1, ((1, 1, 1), (2, 0, 1)), 0)],
    ((2, 0), (1, 2)): [(0, 1, ((1, 4, 1),), 1), (1, 1, ((1, 0, 1),), 0)],
    ((2, 0), (1, 3)): [(0, 1, ((1, 3, 1), (2, 0, 1)), 0)],
    ((2, 0), (1, 4)): [(0, 1, ((1, 4, 1), (2, 0, 1)), 0)],
    ((2, 0), (1, 5)): [(-1, 1, ((1, 1, 1),), 2), (0, 1, ((1, 3, 1),), 1)],
    ((2, 0), (2, 0)): [(0, 1, ((2, 0, 2),), 0)],
    ((2, 0), (2, 1)): [(-1, 1, ((1, 1, 1), (1, 4, 1)), 1), (0, 1, (), 0)],
    ((2, 0), (2, 2)): [(0, 1, (), 1), (1, 1, ((1, 0, 1), (1, 3, 1)), 0)],
    ((2, 1), (1, 0)): [(-1, 1, ((1, 2, 1),), 0), (0, 1, ((1, 4, 1),), 1)],
    ((2, 1), (1, 1)): [(1, 1, ((1, 1, 1), (2, 1, 1)), 0)],
    ((2, 1), (1, 2)): [(-1, 1, ((1, 2, 1), (2, 1, 1)), 0)],
    ((2, 1), (1, 3)): [(1, 1, ((1, 1, 1),), 0), (0, 1, ((1, 5, 1),), -1)],
    ((2, 1), (1, 4)): [(0, 1, ((1, 4, 1), (2, 1, 1)), 0)],
    ((2, 1), (1, 5)): [(0, 1, ((1, 5, 1), (2, 1, 1)), 0)],
    ((2, 1), (2, 0)): [(1, 1, ((1, 1, 1), (1, 4, 1)), 1), (0, 1, (), 0)],
    ((2, 1), (2, 1)): [(0, 1, ((2, 1, 2),), 0)],
    ((2, 1), (2, 2)): [(-1, 1, ((1, 2, 1), (1, 5, 1)), -1), (0, 1, (), 0)],
    ((2, 2), (1, 0)): [(0, 1, ((1, 0, 1), (2, 2, 1)), 0)],
    ((2, 2), (1, 1)): [(0, 1, ((1, 5, 1),), -1), (-1, 1, ((1, 3, 1),), 0)],
    ((2, 2), (1, 2)): [(1, 1, ((1, 2, 1), (2, 2, 1)), 0)],
    ((2, 2), (1, 3)): [(-1, 1, ((1, 3, 1), (2, 2, 1)), 0)],
    ((2, 2), (1, 4)): [(1, 1, ((1, 2, 1),), 0), (0, 1, ((1, 0, 1),), -1)],
    ((2, 2), (1, 5)): [(0, 1, ((1, 5, 1), (2, 2, 1)), 0)],
    ((2, 2), (2, 0)): [(0, 1, (), 1), (-1, 1, ((1, 0, 1), (1, 3, 1)), 0)],
    ((2, 2), (2, 1)): [(1, 1, ((1, 2, 1), (1, 5, 1)), -1), (0, 1, (), 0)],
    ((2, 2), (2, 2)): [(0, 1, ((2, 2, 2),), 0)],
}
assert len(MULT_TABLE_LL) == 81


# ----------------------------------------------------------------------
# Cone-monomial helpers
# ----------------------------------------------------------------------


def _E_thru_L(letter: tuple[int, int]) -> int:
    """fq-power picked up moving E one slot past L_letter (from left to right):
       E · L_l = fq^{-2·MU_LETTER_QPOWER[l]} · L_l · E ."""
    return -2 * MU_LETTER_QPOWER[letter]


def cone_T_half(factors, e_E=0):
    """Bar-invariance prefactor exponent for cone monomial (factors, e_E):
        M = fq^{T_half} · algebra(L^{factors}) · E^{e_E},  bar(M) = M.

    T_half = T_LL/2 − e_E · μ_qp(factors), where
      T_LL = Σ_{i<j} c_ij · e_i · e_j   (even; c_ij = commutator power),
      μ_qp = Σ_l MU_LETTER_QPOWER[l] · e_l.
    """
    T_LL = 0
    n = len(factors)
    for ii in range(n):
        (ka, ia, ea) = factors[ii]
        for jj in range(ii+1, n):
            (kb, ib, eb) = factors[jj]
            sorted_pair = tuple(sorted([(ka, ia, 1), (kb, ib, 1)]))
            ents_fwd = MULT_TABLE_LL[((ka, ia), (kb, ib))]
            ents_bwd = MULT_TABLE_LL[((kb, ib), (ka, ia))]
            p_fwd = next((fp for (fp, ci, f, eE) in ents_fwd
                          if f == sorted_pair and eE == 0 and ci == 1), None)
            p_bwd = next((fp for (fp, ci, f, eE) in ents_bwd
                          if f == sorted_pair and eE == 0 and ci == 1), None)
            if p_fwd is None or p_bwd is None: continue
            c = p_bwd - p_fwd
            assert c % 2 == 0
            T_LL += c * ea * eb
    assert T_LL % 2 == 0
    mu_qp = sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)
    return T_LL // 2 - e_E * mu_qp


def charge_of_label(label):
    """Tropical charge of cone monomial (factors, e_E)."""
    factors, e_E = label
    ch = [e_E * c for c in MU_CHARGE]
    for (a, i, e) in factors:
        fc = charge((a, i))
        for k in range(4): ch[k] += e * fc[k]
    return tuple(ch)


def multiply_cone_monomials(M_a, M_b):
    """Multiply two cone monomials.  Returns dict {cone_monomial: LaurentPoly(fq)}.

    Algorithm:
      1. Flatten both factors to a single algebra word of single L-letters.
      2. Initialise fq-coefficient with the bar-invariance offsets T_a + T_b,
         plus the cost of moving E^{e_E_a} past algebra_b.
      3. Iteratively reduce: pick first non-canonical adjacent pair (unsorted
         or multi-term) and apply MULT_TABLE_LL; for non-adjacent non-fq-commute
         pairs, bubble them adjacent via fq-commute swaps.
      4. When the word is in cone-canonical sorted form, collect to (factors, e_E)
         and divide out the new bar prefactor fq^{T_half_final}.
    """
    factors_a, e_E_a = M_a
    factors_b, e_E_b = M_b
    T_a = cone_T_half(factors_a, e_E_a)
    T_b = cone_T_half(factors_b, e_E_b)
    move_fq = sum(e_E_a * _E_thru_L((a, i)) * e for (a, i, e) in factors_b)
    letters = []
    for (a, i, e) in factors_a:
        letters.extend([(a, i)] * e)
    for (a, i, e) in factors_b:
        letters.extend([(a, i)] * e)
    init_fq = T_a + T_b + move_fq
    init_e_E = e_E_a + e_E_b

    worklist = [(LaurentPoly.q(init_fq), list(letters), init_e_E)]
    out: dict = {}
    MAX_ITER = 200000
    n_iter = 0
    while worklist:
        n_iter += 1
        if n_iter > MAX_ITER:
            raise RuntimeError(f"multiply: max iter {MAX_ITER} exceeded")
        coef, w, e_E = worklist.pop()
        if coef.is_zero(): continue
        # Find first non-canonical adjacent pair.
        target = None
        for k in range(len(w) - 1):
            l1, l2 = w[k], w[k+1]
            ents = MULT_TABLE_LL[(l1, l2)]
            if l1 > l2 or len(ents) > 1:
                target = (k, ents); break
            (fp, ci, fac, eE_e) = ents[0]
            canonical_fac = (((l1[0], l1[1], 2),) if l1 == l2
                              else tuple(sorted([(l1[0], l1[1], 1), (l2[0], l2[1], 1)])))
            if fac == canonical_fac and eE_e == 0 and ci == 1:
                continue
            target = (k, ents); break
        if target is None:
            # No adjacent target — search for non-adjacent non-fq-commute pair
            # whose bubble path q-commutes.
            chosen = None
            for j in range(1, len(w)):
                for i in range(j-1, -1, -1):
                    if len(MULT_TABLE_LL[(w[i], w[j])]) > 1:
                        if all(len(MULT_TABLE_LL[(w[pos], w[j])]) == 1
                               for pos in range(i+1, j)):
                            chosen = (i, j); break
                if chosen is not None: break
            if chosen is not None:
                i, j = chosen
                coef_local = coef
                new_w = list(w)
                for pos in range(j, i+1, -1):
                    a, b = new_w[pos-1], new_w[pos]
                    entsab = MULT_TABLE_LL[(a, b)]
                    entsba = MULT_TABLE_LL[(b, a)]
                    commutator = entsba[0][0] - entsab[0][0]
                    coef_local = coef_local * LaurentPoly.q(-commutator)
                    new_w[pos-1], new_w[pos] = new_w[pos], new_w[pos-1]
                worklist.append((coef_local, new_w, e_E))
                continue
            # Fall through to collect.
        if target is None:
            cnt = Counter(w)
            factors = tuple(sorted((a, i, e) for (a, i), e in cnt.items()))
            T_half = cone_T_half(factors, e_E)
            key = (factors, e_E)
            adj_coef = coef * LaurentPoly.q(-T_half)
            out[key] = (out.get(key) + adj_coef) if key in out else adj_coef
            continue
        k, ents = target
        for (fp, ci, fac, eE_e) in ents:
            T_M = cone_T_half(fac, eE_e)
            new_letters = []
            for (a, i, e) in fac:
                new_letters.extend([(a, i)] * e)
            tail_move = sum(eE_e * _E_thru_L(letter) for letter in w[k+2:])
            new_w = w[:k] + new_letters + w[k+2:]
            new_coef = coef * LaurentPoly.q(fp + T_M + tail_move) * ci
            new_e_E = e_E + eE_e
            worklist.append((new_coef, new_w, new_e_E))
    return {k: v for k, v in out.items() if not v.is_zero()}
