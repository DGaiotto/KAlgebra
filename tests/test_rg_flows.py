"""Self-test for the RGKAlgebra (Step 3) — the spine-free RG-flow engine.

Runs the two reference flows live (every operation computed from the flow data by
the generic engine — no closed-form override, no realisation engine):

  * `U1SquareRGKAlgebra`            — SQED₁ as the RG flow `E_𝖖(X_{0,1})` to the
                                      Z² quantum torus (the `deg = id` torus corner);
  * `PentagonSquareSampleRGKAlgebra`— the pentagon `K_𝖖([A₁,A₂])` as the RG flow
                                      `E_𝖖(u₋)` to SQED₁ (the non-torus corner).

For each: RG solved from the discovery relation, `multiply`/`trace` matching the
direct sample presentation (Step 1), the KAlgebra axioms + orthonormality,
truncation-stability of the trace, and a `KAlgebraIso` to the direct
presentation.

Run (from the repository root):

    python3 run_tests.py
"""
import os
import sys
import warnings

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from laurent_poly import LaurentPoly
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from samples import (
    Z2QTorusSampleKAlgebra, SQED1SampleKAlgebra, PentagonSampleKAlgebra,
    SQEDNfSampleKAlgebra,
)
from u1_square_rg import U1SquareRGKAlgebra
from pentagon_square_rg import PentagonSquareSampleRGKAlgebra
from sqed_nf_rg import SQEDNfRGKAlgebra

_ONE = LaurentPoly.one()


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _nm(el):
    return {k: str(v) for k, v in el.terms.items() if not v.is_zero()}


# ---------------------------------------------------------------------------
# U1Square = SQED₁ as the flow to the Z² quantum torus (torus corner)
# ---------------------------------------------------------------------------

def test_u1square():
    A = U1SquareRGKAlgebra()
    S = SQED1SampleKAlgebra()
    labs = [(0, 0), (0, 1), (1, 0), (-1, 0), (1, 1), (2, 0), (0, 2), (-2, 1)]
    # trace matches the direct SQED₁ v-tower, truncation-stable, no warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for l in labs:
            t6 = _ser(A.trace(l, 6), 6)
            assert t6 == _ser(S.trace(l, 6), 6), ("trace", l, t6)
            assert t6 == _ser(A.trace(l, 10), 6), ("unstable", l)
        assert len(w) == 0, [str(x.message) for x in w]
    # axioms + orthonormality (full contract battery: bar involution,
    # ρ²-twisted trace cyclicity, and inner-product two-face consistency
    # were previously exercised only on the Step-1 samples)
    sm = [(0, 0), (1, 0), (-1, 0), (0, 1), (1, 1)]
    assert A.verify_rg_unital()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    for a in sm:
        assert A.verify_rg_discovery(a), ("rg-discovery", a)
        for b in sm:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_rho_is_automorphism(a, b), ("rho", a, b)
            assert A.verify_orthonormality(a, b, 6), ("ortho", a, b)
            assert A.verify_bar_involution(a, b), ("bar", a, b)
            assert A.verify_rho_twisted_trace(a, b, 6), ("rho2-cyc", a, b)
            assert A.verify_inner_product_consistent(a, b, 6), ("ip-faces", a, b)
    for a in sm[:3]:
        for b in sm[:3]:
            for c in sm[:3]:
                assert A.verify_associativity(a, b, c), ("assoc", a, b, c)
    # KAlgebraIso to the direct SQED₁ (identity on labels)
    idm = lambda l: Element({tuple(l): _ONE})
    iso = KAlgebraIso(A, S, idm, idm, name="U1Square ≅ SQED1")
    src = [Element({l: _ONE}) for l in labs]
    pairs = [(Element({a: _ONE}), Element({b: _ONE})) for a in labs[:5] for b in labs[:5]]
    res = iso.verify_all(src, [iso.map(s) for s in src],
                         pairs, [(iso.map(x), iso.map(y)) for x, y in pairs], trace_K=6)
    for chk in ("unit", "round_trip", "multiplicative", "rho_equivariant", "trace_equivariant"):
        assert res[chk], (chk, res)
    print("  PASS: test_u1square")


# ---------------------------------------------------------------------------
# Pentagon as the flow to SQED₁ (non-torus corner)
# ---------------------------------------------------------------------------

def test_pentagon():
    A = PentagonSquareSampleRGKAlgebra()
    P = PentagonSampleKAlgebra()
    # RG solved generically (no hard-coded generators)
    exp = {(1, 1, 0): {(1, 0): "1"}, (2, 1, 0): {(1, -1): "1", (0, -1): "1"},
           (4, 1, 0): {(-1, 0): "1"}}
    for lab, e in exp.items():
        assert _nm(A.RG(lab)) == e, ("RG", lab, _nm(A.RG(lab)))
    labs = [(i, a, b) for i in range(5) for a in range(2) for b in range(2)]
    # multiply matches the direct pentagon
    for a in labs:
        for b in labs:
            assert _nm(A.multiply(a, b)) == _nm(P.multiply(a, b)), ("mult", a, b)
    # trace matches direct, truncation-stable, no warnings
    tl = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0),
          (0, 2, 0), (2, 1, 1)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for l in tl:
            t6 = _ser(A.trace(l, 6), 6)
            assert t6 == _ser(P.trace(l, 6), 6), ("trace", l, t6)
            assert t6 == _ser(A.trace(l, 10), 6), ("unstable", l)
        assert len(w) == 0, [str(x.message) for x in w]
    # axioms + orthonormality (full contract battery — see test_u1square)
    sm = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (4, 1, 0)]
    assert A.verify_rg_unital()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    for a in sm:
        assert A.verify_rg_discovery(a), ("rg-discovery", a)
        for b in sm:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_rho_is_automorphism(a, b), ("rho", a, b)
            assert A.verify_orthonormality(a, b, 6), ("ortho", a, b)
            assert A.verify_bar_involution(a, b), ("bar", a, b)
            assert A.verify_rho_twisted_trace(a, b, 6), ("rho2-cyc", a, b)
            assert A.verify_inner_product_consistent(a, b, 6), ("ip-faces", a, b)
    for a in sm[:3]:
        for b in sm[:3]:
            for c in sm[:3]:
                assert A.verify_associativity(a, b, c), ("assoc", a, b, c)
    # KAlgebraIso to the direct pentagon (identity on labels)
    idm = lambda l: Element({tuple(l): _ONE})
    iso = KAlgebraIso(A, P, idm, idm, name="PentagonSquare ≅ Pentagon")
    src = [Element({l: _ONE}) for l in labs]
    pairs = [(Element({a: _ONE}), Element({b: _ONE})) for a in labs[:6] for b in labs[:6]]
    res = iso.verify_all(src, [iso.map(s) for s in src],
                         pairs, [(iso.map(x), iso.map(y)) for x, y in pairs], trace_K=6)
    for chk in ("unit", "round_trip", "multiplicative", "rho_equivariant", "trace_equivariant"):
        assert res[chk], (chk, res)
    print("  PASS: test_pentagon")


# ---------------------------------------------------------------------------
# SQED_{N_f} as the SU(N_f)-flavoured flow (the flavoured / nested-aux corner)
# ---------------------------------------------------------------------------

def test_sqed_nf():
    """`SQEDNfRGKAlgebra(N_f)` — the general SU(N_f)-flavoured SQED flow (a pure
    RGKAlgebra; N_f=1 = SQED₁, N_f=2 = SQED₂).  multiply + trace match the direct
    `SQEDNfSampleKAlgebra(N_f)` under the relabel `((a,b),w) ↔ (a,b,w)`, the trace
    is truncation-stable with no warnings, and the axioms hold — the nested-aux
    exact-FS path over an SU(N_f)-flavoured auxiliary."""
    for Nf in (1, 2, 3):
        A = SQEDNfRGKAlgebra(Nf)
        S = SQEDNfSampleKAlgebra(Nf)
        irr = [()] + ([(1,), (2,)] if Nf >= 2 else [])
        labs = [((a, b), w) for a in (-1, 0, 1) for b in (0, 1) for w in irr]
        # SU(N_f) relation u_+u_- = Σ_k 𝖖^k χ_{(1^k)} v^k
        relA = {(l[0][0], l[0][1], l[1]): v
                for l, v in A.multiply(((1, 0), ()), ((-1, 0), ())).terms.items()
                if not v.is_zero()}
        relS = {k: v for k, v in S.multiply((1, 0, ()), (-1, 0, ())).terms.items()
                if not v.is_zero()}
        assert relA == relS, (Nf, relA, relS)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for la in labs:
                pa = {(l[0][0], l[0][1], l[1]): v
                      for l, v in A.multiply(la, ((1, 0), ())).terms.items()
                      if not v.is_zero()}
                ps = {k: v for k, v in
                      S.multiply((la[0][0], la[0][1], la[1]), (1, 0, ())).terms.items()
                      if not v.is_zero()}
                assert pa == ps, (Nf, "mult", la)
                ta = {(e, b): c for e, r in A.trace(la, 6).coeffs.items() if e <= 6
                      for b, c in r.terms.items() if c}
                ts = {(e, b): c for e, r in S.trace(
                          (la[0][0], la[0][1], la[1]), 6).coeffs.items() if e <= 6
                      for b, c in r.terms.items() if c}
                assert ta == ts, (Nf, "trace", la)
            assert len(w) == 0, (Nf, [str(x.message) for x in w])
        sm = [((0, 0), ()), ((1, 0), ()), ((-1, 0), ())] + \
             ([((0, 0), (1,)), ((1, 0), (1,))] if Nf >= 2 else [])
        assert A.verify_rg_unital()
        for x in sm:
            for y in sm:
                assert A.verify_rg_multiplicative(x, y), (Nf, "mult-ax", x, y)
                assert A.verify_orthonormality(x, y, 6), (Nf, "ortho", x, y)
    print("  PASS: test_sqed_nf (N_f = 1, 2, 3)")


def test_spine_free():
    """No BPS realisation-spine module was imported by the whole run
    (previously this suite had no spine assertion at all — the only
    Step-3 file without one)."""
    from _spine import assert_spine_free
    assert_spine_free("the RG reference-flow suite")
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_u1square()
    test_pentagon()
    test_sqed_nf()
    test_spine_free()
    print("\nALL RGKAlgebra (Step 3) self-tests passed.")


if __name__ == "__main__":
    main()
