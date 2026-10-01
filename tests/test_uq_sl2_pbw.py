"""Tests for `uq_sl2_pbw` — standalone `U_𝖖(sl_2)` (central quotient) on the PBW
basis, as a `KAlgebra`.  Verifies the paper relations + the KAlgebra axioms
(bar-involution, ρ-automorphism) through the universal contract surface.

Run: `python3 run_tests.py`.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "implementations"))

from laurent_poly import LaurentPoly
from uq_sl2_pbw import UqSL2PBW, E_LAB, F_LAB, K_LAB, C_LAB

_A = UqSL2PBW()
_E, _F = E_LAB, F_LAB
_q = lambda n: LaurentPoly({n: 1})


def _terms(elt):
    return {l: c for l, c in elt.terms.items()}


def test_paper_relations():
    """KE=q^{-2}EK, KF=q^{2}FK, EF=C+qK+q^{-1}K^{-1}, FE=C+q^{-1}K+qK^{-1},
    [E,F]=(q-q^{-1})(K-K^{-1}) — the central-quotient U_𝖖(sl_2) relations."""
    EF = _terms(_A.multiply(_E, _F))
    FE = _terms(_A.multiply(_F, _E))
    # C = ('K',0,1); K = ('K',1,0); K^{-1} = ('K',-1,0)
    assert EF == {('K', 0, 1): LaurentPoly.one(), ('K', 1, 0): _q(1), ('K', -1, 0): _q(-1)}
    assert FE == {('K', 0, 1): LaurentPoly.one(), ('K', 1, 0): _q(-1), ('K', -1, 0): _q(1)}
    # KE = q^{-2} EK
    KE = _A.multiply(K_LAB(1), _E); EK = _A.multiply(_E, K_LAB(1))
    (lab,) = KE.terms.keys()
    assert KE.terms[lab] == _q(-2) * EK.terms[lab]
    # KF = q^{2} FK
    KF = _A.multiply(K_LAB(1), _F); FK = _A.multiply(_F, K_LAB(1))
    (lab,) = KF.terms.keys()
    assert KF.terms[lab] == _q(2) * FK.terms[lab]
    # [E,F] = (q-q^{-1})(K - K^{-1})
    comm = {}
    for l, c in EF.items():
        comm[l] = comm.get(l, LaurentPoly.zero()) + c
    for l, c in FE.items():
        comm[l] = comm.get(l, LaurentPoly.zero()) - c
    comm = {l: c for l, c in comm.items() if not c.is_zero()}
    assert comm == {('K', 1, 0): _q(1) - _q(-1), ('K', -1, 0): _q(-1) - _q(1)}


def test_bar_involution_axiom():
    """C^c_{ab}(𝖖^{-1}) = C^c_{ba}(𝖖): multiply(a,b).bar() == multiply(b,a)."""
    labs = [_E, _F, K_LAB(1), K_LAB(-1), C_LAB(1),
            ('E', 2, 1, 0), ('F', 1, -1, 0), ('E', 1, 2, 0), ('F', 2, 0, 1)]
    for a in labs:
        for b in labs:
            assert _A.verify_bar_involution(a, b), (a, b)


def test_rho_is_lusztig_braid():
    """ρ(E)=q^{-1}FK^{-1}=F_{1,-1}, ρ(F)=qKE=E_{1,1}, ρ(K)=K^{-1}; ρ is infinite
    order (ρ²: b↦b+2a) — Lusztig's braid, not a finite permutation."""
    assert _A.rho(_E) == ('F', 1, -1, 0)
    assert _A.rho(_F) == ('E', 1, 1, 0)
    assert _A.rho(K_LAB(1)) == ('K', -1, 0)
    assert _A.rho(_A.rho(_E)) == ('E', 1, 2, 0)        # ρ²(E)=E_{1,2}
    assert _A.rho(_A.rho(_A.rho(_A.rho(_E)))) == ('E', 1, 4, 0)   # infinite order
    assert _A.verify_rho_inverse(_E) and _A.verify_rho_inverse(_F)
    assert _A.verify_rho_fixes_identity()


def test_rho_automorphism_axiom():
    """ρ(L_a L_b) == ρ(L_a) ρ(L_b)."""
    labs = [_E, _F, K_LAB(1), K_LAB(-1), C_LAB(1),
            ('E', 2, 1, 0), ('F', 1, -1, 0), ('E', 1, 2, 0), ('F', 2, 0, 1), ('K', 3, 0)]
    for a in labs:
        for b in labs:
            assert _A.verify_rho_is_automorphism(a, b), (a, b)


def test_central_casimir():
    """C (= ('K',0,1)) is central: C·x == x·C."""
    C = C_LAB(1)
    for x in [_E, _F, K_LAB(2), ('E', 2, 1, 0), ('F', 1, -1, 0)]:
        assert _terms(_A.multiply(C, x)) == _terms(_A.multiply(x, C)), x


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, t in tests:
        t()
        print(f"  PASS: {name}")
    print(f"\nAll {len(tests)} UqSL2PBW tests passed.")
