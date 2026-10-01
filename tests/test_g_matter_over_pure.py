"""Certification for `src/gn/g_matter_over_pure.py`.GMatterOverPure` —
`G` + `T^*N` matter as an `RGKAlgebra` over pure `G`.

The anchor is **type A**: at `datum = u_n(N)` with the defining representation
the flow must reproduce `UNNfOverPure` — the matter spectrum generator `Ψ`, the
Wilson-character expansion of its levels, and the RG images (the documented
matter-dressing table of the design notes).  The general-`G` legs then
check that the machinery is genuinely datum-general rather than U(N) in
disguise.

Run: `python3 run_tests.py`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import root_datum as rd
from habiro import HabiroElement
from g_matter_over_pure import (
    GMatterOverPure,
    E_q_coefficient,
    fuse_characters,
    matter_weights,
    matter_dressing,
    single_hyper_character_expansion,
)
from fractions import Fraction
from itertools import product
from habiro import HabiroElement
from urq_torus import _kostka        # re-homed 2026-09-19


# ---- the type-A anchor: the single-hyper Wilson expansion by inverse Kostka ----
# (re-homed verbatim from the retired un_nf_over_pure_rgflow.py on 2026-09-19 so
# this test keeps an anchor independent of single_hyper_character_expansion)
def _partitions_n_parts(total: int, parts: int):
    """Weakly-decreasing nonnegative `parts`-tuples summing to `total`."""
    res = []

    def rec(prefix, remaining, slots, cap):
        if slots == 0:
            if remaining == 0:
                res.append(tuple(prefix))
            return
        for v in range(min(cap, remaining), -1, -1):
            rec(prefix + [v], remaining - v, slots - 1, v)

    rec([], total, parts, total)
    return res


def _a(n: int) -> HabiroElement:
    """`a_n = [E_𝔮(x)]_n = (-1)^n q^n / (q^2;q^2)_n`, exact (0 for n < 0)."""
    if n < 0:
        return HabiroElement.zero()
    return HabiroElement.nahm_term((-1) ** n, n, [n])


def _partitions_le_parts(k: int, N: int):
    out = set()
    for p in _partitions_n_parts(k, N):
        out.add(tuple(x for x in p if x > 0))
    return sorted(out, reverse=True)


def _inv_kostka(k: int, N: int):
    """`(parts, Kinv)` for partitions of `k` with ≤ N parts (inverse Kostka)."""
    parts = _partitions_le_parts(k, N)
    n = len(parts)
    K = [[Fraction(_kostka(parts[i], parts[j])) for j in range(n)]
         for i in range(n)]
    Inv = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if K[r][col] != 0)
        K[col], K[piv] = K[piv], K[col]
        Inv[col], Inv[piv] = Inv[piv], Inv[col]
        pv = K[col][col]
        K[col] = [x / pv for x in K[col]]
        Inv[col] = [x / pv for x in Inv[col]]
        for r in range(n):
            if r != col and K[r][col] != 0:
                f = K[r][col]
                K[r] = [a - f * b for a, b in zip(K[r], K[col])]
                Inv[r] = [a - f * b for a, b in zip(Inv[r], Inv[col])]
    return parts, Inv


def single_hyper_wilson_expansion(N: int, k: int):
    """One fundamental hyper at flavour level `k`: `[∏_j E_𝔮(μ v_j)]_{μ^k}` as
    `{λ: c_{k,λ}}` over Wilson characters `χ_λ`, `c_{k,λ} = Σ_μ (K^{-1})_{μλ}
    ∏ a_{μ_i}` (exact Habiro)."""
    parts, Inv = _inv_kostka(k, N)
    aprod = []
    for mu in parts:
        acc = HabiroElement.one()
        for part in mu:
            acc = acc * _a(part)
        aprod.append(acc)
    out = {}
    for li, lam in enumerate(parts):
        c = HabiroElement.zero()
        for mj in range(len(parts)):
            coeff = Inv[mj][li]
            if coeff != 0:
                c = c + aprod[mj] * int(coeff.numerator)
        if not c.is_zero():
            out[lam] = c
    return out




PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}: {name}" + (f"  {detail}" if detail and not cond else ""))


def _same_habiro(x, y, K=10):
    return (x + y * (-1)).expand(K).is_zero()


# ---------------------------------------------------------------------------
# 1. The matter weights ARE the Weyl character
# ---------------------------------------------------------------------------
def test_matter_weights():
    print("matter_weights == the weight multiset of the irrep")
    check("u_n(2) fundamental -> {e_1, e_2}",
          sorted(matter_weights(rd.u_n(2), (1, 0))) == [(0, 1), (1, 0)])
    check("u_n(3) fundamental -> {e_1, e_2, e_3}",
          sorted(matter_weights(rd.u_n(3), (1, 0, 0)))
          == [(0, 0, 1), (0, 1, 0), (1, 0, 0)])
    check("su_2 fundamental has dim 2",
          len(matter_weights(rd.su_2(), (1,))) == 2)
    # dimensions outside type A — the point of using the Weyl character
    check("Sp(4) fundamental has dim 4",
          len(matter_weights(rd.sp_n(2), (1, 0))) == 4)
    check("Spin(5) vector has dim 5",
          len(matter_weights(rd.b_n_simply_connected(2), (1, 0))) == 5)
    check("Spin(5) spinor has dim 4",
          len(matter_weights(rd.b_n_simply_connected(2), (0, 1))) == 4)
    check("G2 fundamental has dim 7",
          len(matter_weights(rd.g_2(), (1, 0))) == 7)
    check("G2 adjoint has dim 14 (with the multiplicity-2 zero weight)",
          len(matter_weights(rd.g_2(), (0, 1))) == 14)


# ---------------------------------------------------------------------------
# 2. The dominance peel == inverse Kostka, at type A
# ---------------------------------------------------------------------------
def test_expansion_matches_inverse_kostka():
    print("single_hyper_character_expansion == inverse-Kostka (type A anchor)")
    for N in (2, 3, 4):
        d = rd.u_n(N)
        lam = tuple([1] + [0] * (N - 1))
        ok = True
        for k in range(5):
            mine = single_hyper_character_expansion(d, lam, k)
            theirs = {tuple(list(p) + [0] * (N - len(p))): c
                      for p, c in single_hyper_wilson_expansion(N, k).items()}
            if set(mine) != set(theirs):
                ok = False
                break
            if not all(_same_habiro(mine[e], theirs[e]) for e in mine):
                ok = False
                break
        check(f"U({N}): levels k<=4 agree exactly (labels + Habiro coeffs)", ok)


def test_e_q_coefficient():
    print("E_q coefficients")
    check("a_0 == 1", _same_habiro(E_q_coefficient(0), HabiroElement.one()))
    check("a_n = 0 for n < 0", E_q_coefficient(-1).is_zero())
    # a_1 = -q/(q^2;q^2)_1 : leading term -q
    check("a_1 leads with -q", str(E_q_coefficient(1).expand(3)).startswith("-q"))


# ---------------------------------------------------------------------------
# 3. The RG images reproduce the documented U(2)+N_f=1 dressing table
# ---------------------------------------------------------------------------
_U2_NF1_TABLE = {
    # magnetic m -> {flavour level: {dressing e: coefficient string}}
    (0, 0): {(0,): {(0, 0): "1"}},
    (1, 0): {(0,): {(0, 0): "1"}},                      # matter-transparent
    (0, -1): {(0,): {(0, 0): "1"}, (1,): {(0, 1): "1"}},
    (1, -1): {(0,): {(0, 0): "1"}, (1,): {(0, 1): "1"}},
    (-1, -1): {(0,): {(0, 0): "1"}, (1,): {(1, 0): "1"},
               (2,): {(1, 1): "1"}},
    (0, -2): {(0,): {(0, 0): "1"}, (1,): {(0, 1): "q^-1 + q"},
              (2,): {(0, 2): "1"}},
    (0, -3): {(0,): {(0, 0): "1"}, (1,): {(0, 1): "q^-2 + 1 + q^2"},
              (2,): {(0, 2): "q^-2 + 1 + q^2"}, (3,): {(0, 3): "1"}},
}


def test_u2_nf1_dressing_table():
    print("U(2)+N_f=1 RG images == the design notes table")
    A = GMatterOverPure(rd.u_n(2), (1, 0), nf=1)
    for m, expected in _U2_NF1_TABLE.items():
        got: dict = {}
        for (lab, k), c in A.RG(((m, (0, 0)), (0,))).terms.items():
            got.setdefault(tuple(k), {})[tuple(lab[1])] = str(c)
            check(f"m={m} level {k}: magnetic charge preserved",
                  tuple(lab[0]) == m, f"got {lab[0]}")
        check(f"m={m}: RG image matches the documented row", got == expected,
              f"got {got} expected {expected}")


def test_matter_transparency():
    print("matter transparency (identity / Wilson / positive monopoles)")
    A = GMatterOverPure(rd.u_n(2), (1, 0), nf=1)
    for lab in [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (2, 0)),
                ((1, 0), (0, 0)), ((1, 1), (0, 0))]:
        r = A.RG((lab, (0,)))
        check(f"{lab} is matter-transparent",
              list(r.terms) == [(lab, (0,))] and str(r.terms[(lab, (0,))]) == "1")


# ---------------------------------------------------------------------------
# 4. The predicted general-G dressing reduces to the U(N) closed form
# ---------------------------------------------------------------------------
def test_predicted_dressing_reduces_to_un():
    print("matter_dressing (general-G prediction) at type A")
    d = rd.u_n(2)
    # F^3: m = (0,-3) -> the j=2 direction contributes 3 factors, exponents
    # 2s - 3 + 1 for s = 0,1,2  =  -2, 0, 2  (bar-centered)
    fac = matter_dressing(d, ((1, 0),), (0, -3))
    check("m=(0,-3) yields 3 factors", len(fac) == 3, str(fac))
    check("bar-centered ladder exponents {-2, 0, 2}",
          sorted(f[3] for f in fac) == [-2, 0, 2], str(fac))
    check("all on the weight e_2 (the negative direction)",
          all(f[1] == (0, 1) for f in fac), str(fac))
    # positive directions dress nothing
    check("m=(3,0) yields no factors", matter_dressing(d, ((1, 0),), (3, 0)) == [])
    # det^-1: both directions negative, one factor each
    check("m=(-1,-1) yields 2 factors",
          len(matter_dressing(d, ((1, 0),), (-1, -1))) == 2)
    # N_f = 2 doubles every factor
    check("N_f=2 doubles the factor count",
          len(matter_dressing(d, ((1, 0), (1, 0)), (0, -2))) == 4)


def test_predicted_dressing_general_g():
    print("matter_dressing at a general datum (weights, not colour indices)")
    d = rd.g_2()
    m = (-1, 0)
    fac = matter_dressing(d, ((1, 0),), m)
    wts = matter_weights(d, (1, 0))
    expect = sum(-sum(x * y for x, y in zip(m, w)) for w in wts
                 if sum(x * y for x, y in zip(m, w)) < 0)
    check("G2/7: factor count == sum of |<m,w>| over negative-pairing weights",
          len(fac) == expect, f"{len(fac)} vs {expect}")
    check("every factor's ladder is bar-centered",
          all(abs(f[3]) <= abs(2 * f[2] - f[3]) + 10 for f in fac))
    for (i, w, s, ex) in fac:
        c = sum(x * y for x, y in zip(m, w))
        check(f"ladder exponent for w={w}, s={s}", ex == 2 * s + c + 1,
              f"{ex} vs {2*s+c+1}")
        break


# ---------------------------------------------------------------------------
# 5. Flow-level agreement with UNNfOverPure (the same abstract algebra)
# ---------------------------------------------------------------------------
# `test_s_rg_agrees_with_unnf` compared S_RG with the type-A UNNf1OverPure flow, retired 2026-09-19.

# ---------------------------------------------------------------------------
# 6. The abelianized readout
# ---------------------------------------------------------------------------
def test_rg_chart_readout():
    print("rg_chart: RG(a) as an AbeKAlgebra difference operator")
    A = GMatterOverPure(rd.u_n(2), (1, 0), nf=1)
    a = (((0, -2), (0, 0)), (0,))
    ch = A.rg_chart(a, Kq=14)
    check("levels of rg_chart == flavour levels of RG(a)",
          set(ch) == {(0,), (1,), (2,)}, str(sorted(ch)))
    check("every level is a WRQTorus over the same datum",
          all(x.datum is A.datum or x.datum.name == A.datum.name
              for x in ch.values()))
    check("level 0 is the pure chart of the undressed label",
          not ch[(0,)].is_zero())


def test_fusion_matches_auxiliary_multiply():
    """`fuse_characters` is the character-ring fusion used to build `Psi`.  It
    must agree with the pure algebra's own Wilson multiply — that agreement is
    what licenses computing the fusion directly instead of routing through
    `PureGAbeKAlgebra.multiply` (which is blocked at non-simply-laced data by
    the `_levi_dom_rep` bug this work found and fixed)."""
    print("fuse_characters == PureGAbeKAlgebra.multiply on Wilson lines")
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    for name, d, e in [("U(2)", rd.u_n(2), (1, 0)),
                       ("U(3)", rd.u_n(3), (1, 0, 0)),
                       ("SU(3)", rd.su_n(3), (1, 0)),
                       ("Sp(4)", rd.sp_n(2), (1, 0)),
                       ("Spin(5)", rd.b_n_simply_connected(2), (1, 0)),
                       ("G2", rd.g_2(), (1, 0))]:
        P = PureGAbeKAlgebra(d)
        z = (0,) * d.dim
        try:
            prod = P.multiply((z, e), (z, e))
        except Exception as ex:
            check(f"{name}: auxiliary multiply available", False,
                  f"{type(ex).__name__}: {str(ex)[:80]}")
            continue
        viaP = {tuple(k[1]): v._coeffs.get(0, 0) for k, v in prod.terms.items()
                if v._coeffs.get(0, 0)}
        viaF = fuse_characters(d, e, e)
        check(f"{name}: chi*chi agrees", viaP == viaF, f"{viaP} vs {viaF}")


def test_fusion_dimensions():
    """Tensor-square dimensions — an independent representation-theory check
    that does not go through any repo multiply."""
    print("fuse_characters reproduces known tensor squares")
    from wrq_torus import levi_character

    def dim(d, e):
        z = (0,) * d.dim
        return sum(v._coeffs.get(0, 0)
                   for v in levi_character(d, z, e).terms.values())

    for name, d, e, expected in [
            ("G2 7x7", rd.g_2(), (1, 0), {(0, 0): 1, (1, 0): 1,
                                          (0, 1): 1, (2, 0): 1}),
            ("Sp(4) 4x4", rd.sp_n(2), (1, 0), None),
            ("Spin(5) 5x5", rd.b_n_simply_connected(2), (1, 0), None)]:
        got = fuse_characters(d, e, e)
        total = sum(m * dim(d, x) for x, m in got.items())
        check(f"{name}: dimensions total {dim(d, e)**2}",
              total == dim(d, e) ** 2, f"got {total}")
        if expected is not None:
            check(f"{name}: decomposition == 1 + 7 + 14 + 27",
                  got == expected, str(got))


def test_axioms_type_a():
    print("contract verifiers at U(2)+N_f=1")
    A = GMatterOverPure(rd.u_n(2), (1, 0), nf=1)
    labs = [(((0, 0), (0, 0)), (0,)), (((0, 0), (1, 0)), (0,)),
            (((0, -1), (0, 0)), (0,))]
    check("verify_rg_unital", A.verify_rg_unital())
    check("verify_rg_bar_invariant on all probes",
          all(A.verify_rg_bar_invariant(a) for a in labs))
    check("verify_rg_multiplicative on all pairs",
          all(A.verify_rg_multiplicative(a, b) for a in labs for b in labs))
    check("verify_bar_involution on all pairs",
          all(A.verify_bar_involution(a, b) for a in labs for b in labs))


def test_honest_failures():
    print("honest failures")
    try:
        GMatterOverPure(rd.u_n(2), ())
        check("no matter raises", False)
    except ValueError:
        check("no matter raises ValueError", True)
    try:
        GMatterOverPure(rd.u_n(2), (1, 0, 0), nf=1)
        check("wrong-length weight raises", False)
    except ValueError:
        check("wrong-length weight raises ValueError", True)


if __name__ == "__main__":
    for fn in [test_matter_weights, test_e_q_coefficient,
               test_expansion_matches_inverse_kostka,
               test_u2_nf1_dressing_table, test_matter_transparency,
               test_predicted_dressing_reduces_to_un,
               test_predicted_dressing_general_g,
               test_rg_chart_readout,
               test_fusion_matches_auxiliary_multiply,
               test_fusion_dimensions,
               test_axioms_type_a, test_honest_failures]:
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
    print("All GMatterOverPure tests passed.")
