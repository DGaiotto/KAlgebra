"""`lower_flavour(φ)` — flavour reduction along an augmentation-
compatible ring hom.  Transport: **identify sections, push the R-elements
through `φ`**; a canonical *splits* where `φ` grows `Λ`.

Validations:
  * `φ = identity`  → iso to the source (multiply reproduces source `to_R_form`);
  * `φ = augmentation` → *merge* (== `forget()`);
  * `φ = SU(2)→U(1)` restriction → *split* (fundamental `χ_1 ↦ μ+μ⁻¹`), valid
    KAlgebra on dressed labels;
  * `trace` transports (`Tr` is R-linear over the centre) and the lowered
    algebra is orthonormal.

Run:  `python3 run_tests.py`
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalgebra import Element
from laurent_poly import LaurentPoly
from su3_ad_kalg import SU3ADKAlg
from finite_a1d3_zform import FiniteA1D3ZKAlgebra
from zplus_ring import identity_hom, AbelianZPlusRing, RingHom


def _basis(x):
    return Element({x: LaurentPoly({0: 1})})


def _low(A, lbl):
    """A source canonical ↦ its `(section, R'-irrep)` lowered label (φ=id)."""
    s, r = A._label_section_decompose(lbl)
    (w, n), = [(b, nn) for b, nn in r.terms.items() if nn]   # single irrep
    return (s, w)


def test_section_single_irrep_verifier():
    """The sharpened contract is explicit + holds on the contractual flavoured
    realisations (here SU(3)-flavoured su3_ad)."""
    A = SU3ADKAlg()
    for lbl in [A.identity(), A.T(0), A.T(1), A.D(0), A.chi(1, 0), A.chi(0, 1)]:
        assert A.verify_section_is_single_irrep(lbl), lbl


def test_identity_lowering_is_iso_to_source():
    A = SU3ADKAlg()
    L = A.lower_flavour(identity_hom(A.coefficient_ring()))
    ll = [L.identity()] + [_low(A, g)
                           for g in (A.T(0), A.T(1), A.D(0), A.chi(1, 0), A.chi(0, 1))]
    o = L.identity()
    assert all(L.multiply(o, x) == _basis(x) and L.multiply(x, o) == _basis(x) for x in ll)
    me = L.multiply_elements
    assert all(
        me(me(_basis(a), _basis(b)), _basis(c)) == me(_basis(a), me(_basis(b), _basis(c)))
        for a in ll[:4] for b in ll[:4] for c in ll[:4])
    assert all(L.verify_rho_is_automorphism(a, b) for a in ll[:4] for b in ll[:4])
    assert all(L.verify_bar_involution(a, b) for a in ll[:4] for b in ll[:4])
    # iso: lowered multiply == source to_R_form
    a, b = A.T(1), A.chi(1, 0)
    low_prod = L.multiply(_low(A, a), _low(A, b))
    src = A.to_R_form(A.multiply(a, b))
    reb = {}
    for (s, w), lp in low_prod.terms.items():
        for q, n in lp._coeffs.items():
            reb.setdefault(s, {}).setdefault(q, {})[w] = reb.get(s, {}).get(q, {}).get(w, 0) + n
    assert bool(low_prod.terms) and all(
        reb.get(s, {}).get(q, {}).get(w, 0) == n
        for s, rl in src.terms.items() for q, re in rl.coeffs.items() for w, n in re.terms.items())


def test_augmentation_lowering_merges_like_forget():
    A = SU3ADKAlg()
    Lm = A.lower_flavour(A.coefficient_ring().augmentation())
    F = A.forget()
    ob = Lm.coefficient_ring().one_basis()
    sec = lambda l: A._label_section_decompose(l)[0]
    lm = Lm.multiply((sec(A.T(0)), ob), (sec(A.T(1)), ob))
    fm = F.multiply(A.T(0), A.T(1))
    assert {s: lp for (s, w), lp in lm.terms.items()} == dict(fm.terms)


def test_su2_to_u1_split_is_valid_kalgebra():
    B = FiniteA1D3ZKAlgebra()
    su2 = B.coefficient_ring()
    u1 = AbelianZPlusRing(1)
    phi = RingHom(su2, u1, lambda b: su2.to_abelian(su2.basis_element(b), u1))
    assert sorted(phi.apply_basis(1).terms.keys()) == [(-1,), (1,)]   # the split
    L = B.lower_flavour(phi)
    CD = B._native.cone_data()
    gsecs = [(0, CD.from_cone_label(frozenset({g}), {g: 1}))
             for g in list(CD.mult_gens())[:3]]
    labs = [L.identity()] + [(s, (k,)) for s in gsecs for k in (0, 1, -1)]
    me = L.multiply_elements
    assert all(L.multiply(L.identity(), x) == _basis(x) for x in labs)
    assert all(
        me(me(_basis(a), _basis(b)), _basis(c)) == me(_basis(a), me(_basis(b), _basis(c)))
        for a in labs[:7] for b in labs[:7] for c in labs[:7])
    assert all(L.verify_rho_is_automorphism(a, b) for a in labs for b in labs)
    assert all(L.verify_bar_involution(a, b) for a in labs for b in labs)
    assert all(L.verify_section_is_single_irrep(x) for x in labs)


def test_lowered_trace_transports_and_is_orthonormal():
    """`Tr` is R-linear over the centre — `Tr((s,χ)) = χ·Tr(M_s)` — so it
    transports pointwise: `Tr'((s,w)) = w·φ(Tr(M_s))`.  Under `φ=id` it
    reproduces `source.trace`, and the lowered `inner_product`
    (multiply-then-trace) is orthonormal."""
    from zplus_ring import RPowerSeries
    A = SU3ADKAlg(); R = A.coefficient_ring()
    # R-linearity anchor on the source: Tr(χ_(1,0)) == χ_(1,0)·Tr(id)
    tr_chi, tr_id = A.trace(A.chi(1, 0), 5), A.trace(A.identity(), 5)
    chi10 = R.basis_element((1, 0))
    exp = RPowerSeries(R, {q: chi10 * c for q, c in tr_id.coeffs.items()
                          if not (chi10 * c).is_zero()}, tr_id.K)
    assert tr_chi == exp
    # lowered trace == source under φ=id
    L = A.lower_flavour(identity_hom(R))
    assert L.trace(_low(A, A.chi(1, 0)), 5) == A.trace(A.chi(1, 0), 5)
    # orthonormality on the lowered algebra (inner_product via multiply-then-trace)
    labs = [L.identity(), _low(A, A.T(0)), _low(A, A.chi(1, 0)), _low(A, A.chi(0, 1))]
    assert all(L.verify_orthonormality(x, x, 5) for x in labs)
    assert all(L.verify_orthonormality(a, b, 5)
               for i, a in enumerate(labs) for b in labs[i + 1:])


if __name__ == "__main__":
    import traceback
    fails = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS: {name}")
            except Exception:
                fails += 1; print(f"  FAIL: {name}"); traceback.print_exc()
    print("\nAll lower_flavour tests passed." if not fails else f"\n{fails} failure(s).")
    sys.exit(1 if fails else 0)
