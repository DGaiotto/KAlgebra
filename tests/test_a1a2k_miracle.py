"""The A1A2k miracle test at k=3 (nonagon = `A_𝖖([A_1, A_6])`).

  Generalises the Heptagon miracle (k=2) to k=3.  For k=3 there are
  four elementary traces (T_0, T_1, T_2, T_3) — the M(2, 9) Virasoro
  minimal-model primaries — and we form a **3×4 stabilisation matrix**
  whose three 3×3 minors reproduce the four elementary traces up to
  the standard cofactor sign convention.

  Decoration recipe (nonagon)
  ---------------------------

  For k=3 we use the q-commuting triple (1, L_1, L_3) — i.e., the
  *distinct-orbit* decorations.  Squared powers like L_1^2 produce
  *linearly dependent* rows (empirically verified: with H = Tr(L_2^a
  L_1^2), the H-row equals the F-row up to a q-monomial scalar, so the
  matrix is rank-2 and the 3×3 minors vanish).

  The three rows:

      F[*](q)  =  q^{a^2 - 1}  · Layer1( Tr(L_2^a)             )
      G[*](q)  =  q^{-emin_G}  · Layer1( Tr(L_2^a · L_1)       )
      H[*](q)  =  q^{-emin_H}  · Layer1( Tr(L_2^a · L_3)       )

  where * ranges over the four canonical ρ²-orbit seeds:
      ()             → T_0
      ((1, 0, 1),)   → T_1
      ((2, 0, 1),)   → T_2
      ((3, 0, 1),)   → T_3
  emin_G = emin_F - (2k+3),   emin_H = emin_F - 1   (empirical at a=12)

  The miracle
  -----------

  At a=12 (= a²-1 = 143 q-shift, F·T = 0 to q^{a+a²-1} = q^155):

      F_L·G_2·H_3 ± ... permutations  =  − T_0   (sign +/- by cofactor pattern)
      ...                              =  + T_1
      ...                              =  − T_2
      ...                              =  + T_3

  Verified to q^23 (M_omit_0 vs T_0), q^26 (M_omit_1 vs T_1),
  q^24 (M_omit_2 vs T_2), q^25 (M_omit_3 vs T_3).

  Layer-1 algorithm is fast (~7s total for a=12); higher a extends the
  match depth but at growing compute cost (Layer 1 scales ~3x per +2 in a).

  This makes the construction "discover Layer 2 from Layer 1 alone"
  reproducible at k=3, providing in-principle a route to Andrews-Gordon
  characters of the M(2, 2k+3) minimal models without analytic input.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from a1a2k_kalg import A1A2kKAlg
from laurent_poly import LaurentPoly


def _to_int_dict(rps):
    out = {}
    for e, rc in rps.coeffs.items():
        for _, v in rc.terms.items():
            out[e] = v
    return out


def _layer1(A, cd, label):
    """Layer 1 of `Tr(L_label)` in A_𝖖([A_1, A_{2k}]) → 4-tuple of
    (c_0, c_1, c_2, c_3) for k=3."""
    s = cd.simplify_trace_via_cone_data(A, label)
    return tuple(
        dict(s.terms.get(seed, LaurentPoly.zero())._coeffs)
        for seed in ((), ((1, 0, 1),), ((2, 0, 1),), ((3, 0, 1),))
    )


def _shift_layer(coeffs):
    all_e = [e for c in coeffs for e in c]
    if not all_e:
        return coeffs, 0
    emin = min(all_e)
    return tuple({e - emin: v for e, v in c.items()} for c in coeffs), emin


def _mul(p, q, q_max=40):
    out = {}
    for ep, cp in p.items():
        for eq, cq in q.items():
            e = ep + eq
            if e > q_max:
                continue
            out[e] = out.get(e, 0) + cp * cq
    return {e: c for e, c in out.items() if c != 0}


def _sub(p, q):
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) - c
    return {e: c for e, c in out.items() if c != 0}


def _add(*polys):
    out = {}
    for p in polys:
        for e, c in p.items():
            out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c != 0}


def _det3x3(rows, q_max=40):
    a0, a1, a2 = rows[0]
    b0, b1, b2 = rows[1]
    c0, c1, c2 = rows[2]
    return _sub(
        _add(_mul(a0, _sub(_mul(b1, c2, q_max), _mul(b2, c1, q_max)), q_max),
             _mul(a2, _sub(_mul(b0, c1, q_max), _mul(b1, c0, q_max)), q_max)),
        _mul(a1, _sub(_mul(b0, c2, q_max), _mul(b2, c0, q_max)), q_max),
    )


def _matches_up_to(actual: dict, expected: dict, q_max: int) -> int:
    """Largest q-power ≤ q_max where actual and expected agree."""
    for q in range(0, q_max + 1):
        if actual.get(q, 0) != expected.get(q, 0):
            return q - 1
    return q_max


def test_a1a2k_k3_distinct_orbits_give_rank_3_matrix():
    """Sanity: the decoration recipe (1, L_1, L_3) gives F, G, H rows
    that are linearly independent (i.e., H ≠ F up to q-monomial)."""
    A = A1A2kKAlg(k=3)
    cd = A.cone_data()
    a = 8
    F_s, _ = _shift_layer(_layer1(A, cd, ((2, 0, a),)))
    G_s, _ = _shift_layer(_layer1(A, cd, ((1, 0, 1), (2, 0, a))))
    H_s, _ = _shift_layer(_layer1(A, cd, ((2, 0, a), (3, 0, 1))))
    # H == F would mean rank-2 (degenerate).  Check at least one column differs.
    assert any(F_s[i] != H_s[i] for i in range(4)), (
        "Decoration (1, L_1, L_3) gave H == F (rank-2 degeneracy)"
    )


def test_a1a2k_k3_squared_power_is_rank_degenerate():
    """Documentation test: the decoration (1, L_1, L_1^2) gives a
    RANK-2 matrix (H proportional to F as a q-shift).  This is why we
    use distinct orbits, not powers."""
    A = A1A2kKAlg(k=3)
    cd = A.cone_data()
    a = 6
    F_s, _ = _shift_layer(_layer1(A, cd, ((2, 0, a),)))
    H_s, _ = _shift_layer(_layer1(A, cd, ((1, 0, 2), (2, 0, a))))
    # H == F (modulo q-shift, already absorbed by shifting to start at q^0).
    for i in range(4):
        for q in range(0, 10):
            assert F_s[i].get(q, 0) == H_s[i].get(q, 0), (
                f"Tr(L_2^{a} · L_1^2): expected H_s[{i}][q^{q}] = "
                f"F_s[{i}][q^{q}] = {F_s[i].get(q, 0)}, got {H_s[i].get(q, 0)} "
                f"(rank degeneracy of L_1^2 decoration)"
            )


def test_a1a2k_k3_miracle_3x3_minors_match_elementary_traces():
    """THE MIRACLE at k=3.

    With F, G, H as described in the module docstring, the four 3×3
    minors of [F; G; H] reproduce the four M(2, 9) elementary traces:

        M_omit_0  =  -T_0
        M_omit_1  =  +T_1
        M_omit_2  =  -T_2
        M_omit_3  =  +T_3

    Sign pattern (-, +, -, +) = (-1)^{i+1}, the standard Cramer cofactor
    convention for null-vector extraction.

    Verified to q^23 (the shallowest of the four minor depths) at a=12.
    """
    A = A1A2kKAlg(k=3)
    cd = A.cone_data()
    a = 12
    F_s, _ = _shift_layer(_layer1(A, cd, ((2, 0, a),)))
    G_s, _ = _shift_layer(_layer1(A, cd, ((1, 0, 1), (2, 0, a))))
    H_s, _ = _shift_layer(_layer1(A, cd, ((2, 0, a), (3, 0, 1))))

    # Truth: M(2, 9) elementary traces.
    K_T = 30
    T_series = A._compute_T_series(K_T)
    T_0 = _to_int_dict(T_series[0])
    T_1 = _to_int_dict(T_series[1])
    T_2 = _to_int_dict(T_series[2])
    T_3 = _to_int_dict(T_series[3])

    F0, F1, F2, F3 = F_s
    G0, G1, G2, G3 = G_s
    H0, H1, H2, H3 = H_s

    q_max = 23
    M_omit_0 = _det3x3([(F1, F2, F3), (G1, G2, G3), (H1, H2, H3)], q_max)
    M_omit_1 = _det3x3([(F0, F2, F3), (G0, G2, G3), (H0, H2, H3)], q_max)
    M_omit_2 = _det3x3([(F0, F1, F3), (G0, G1, G3), (H0, H1, H3)], q_max)
    M_omit_3 = _det3x3([(F0, F1, F2), (G0, G1, G2), (H0, H1, H2)], q_max)

    neg_T_0 = {e: -c for e, c in T_0.items()}
    neg_T_2 = {e: -c for e, c in T_2.items()}

    d0 = _matches_up_to(M_omit_0, neg_T_0, q_max)
    d1 = _matches_up_to(M_omit_1, T_1, q_max)
    d2 = _matches_up_to(M_omit_2, neg_T_2, q_max)
    d3 = _matches_up_to(M_omit_3, T_3, q_max)

    assert d0 >= q_max, (
        f"M_omit_0 = -T_0 matches only to q^{d0} (expected q^{q_max})"
    )
    assert d1 >= q_max, (
        f"M_omit_1 = +T_1 matches only to q^{d1}"
    )
    assert d2 >= q_max, (
        f"M_omit_2 = -T_2 matches only to q^{d2}"
    )
    assert d3 >= q_max, (
        f"M_omit_3 = +T_3 matches only to q^{d3}"
    )


if __name__ == "__main__":
    tests = [
        test_a1a2k_k3_distinct_orbits_give_rank_3_matrix,
        test_a1a2k_k3_squared_power_is_rank_degenerate,
        test_a1a2k_k3_miracle_3x3_minors_match_elementary_traces,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
            fails += 1
    print()
    if fails == 0:
        print(f"All {len(tests)} A1A2k(k=3) miracle tests passed.")
    else:
        print(f"{fails} failure(s).")
        sys.exit(1)
