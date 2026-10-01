"""Tests for `src/bps/uq_su2_bps_iso.py` — the certified
`KAlgebraIso` between `UqSU2KAlgebra` (the `U_𝖖(sl_2)` `K_𝖖`-algebra, direct
from the draft's definition) and `gf_bps_sqed_nf(2)` (the SU(2)-flavoured BPS
chart of SQED_2), connections to other algebraic frameworks / the design record.

The evidence is the iso battery in BOTH directions (`KAlgebraIso.verify_all`:
unit, round trip, multiplicativity, ρ-equivariance, trace-equivariance) on a
basket that covers the generators, dressed and Casimir-dressed labels in every
sector, plus the section ↦ 1-dim-rep check of the flavour layer.  The chart
side computes with the BPS F-solver and Nahm-sum traces, the quantum-group
side with PBW straightening and the generating function `G(x, μ)`, so an
agreement is a genuine cross-presentation certificate, not a tautology.

Run: `python3 run_tests.py`.
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "implementations"))
warnings.simplefilter("ignore")

from kalgebra import Element
from laurent_poly import LaurentPoly
from uq_su2_bps_iso import (chart_label_to_uq, iso_samples, uq_label_to_chart,
                            uq_su2_bps_iso)

_ISO = uq_su2_bps_iso()
_A, _G = _ISO.source, _ISO.target
_ONE = LaurentPoly.one()
_q = lambda n: LaurentPoly({n: 1})


def _terms(elt):
    return {l: c for l, c in elt.terms.items()}


def test_generator_dictionary_on_the_chart():
    """The draft's dictionary in the chart's `(m, e, k)` labels: K = v ↔ (0,1,0),
    E = u₊ ↔ (1,0,0), F = u₋ = D_{−1,−1} ↔ (−1,−1,0), χ₁ ↔ (0,0,1); the chart's
    own products reproduce `EF = χ₁ + 𝖖K + 𝖖⁻¹K⁻¹` (the draft's
    `u₊u₋ = 𝖖⁻¹v⁻¹ + μ + μ⁻¹ + 𝖖v`), `FE`, and `KE = 𝖖⁻²EK`."""
    assert uq_label_to_chart(_A, _A.K(1)) == (0, 1, 0)
    assert uq_label_to_chart(_A, _A.E()) == (1, 0, 0)
    assert uq_label_to_chart(_A, _A.F()) == (-1, -1, 0)
    assert uq_label_to_chart(_A, _A.K(0, 1)) == (0, 0, 1)
    assert uq_label_to_chart(_A, _A.E(2, 3)) == (2, 3, 0)          # E_{a,b} = D_{a,b}
    assert _terms(_G.multiply((1, 0, 0), (-1, -1, 0))) == {
        (0, 0, 1): _ONE, (0, 1, 0): _q(1), (0, -1, 0): _q(-1)}
    assert _terms(_G.multiply((-1, -1, 0), (1, 0, 0))) == {
        (0, 0, 1): _ONE, (0, 1, 0): _q(-1), (0, -1, 0): _q(1)}
    KE, EK = _G.multiply((0, 1, 0), (1, 0, 0)), _G.multiply((1, 0, 0), (0, 1, 0))
    (lab,) = KE.terms
    assert lab == (1, 1, 0) and KE.terms[lab] == _q(-2) * EK.terms[lab]
    for lab in iso_samples(_A):
        assert chart_label_to_uq(_A, uq_label_to_chart(_A, lab)) == lab, lab


def test_full_iso_battery():
    """`verify_all` in both directions on the basket (a ≤ 2, |b| ≤ 1, χ₀/χ₁)
    and on generator pairs; traces compared through 𝖖^8."""
    labs = iso_samples(_A)
    src = [Element({l: _ONE}) for l in labs]
    tgt = [_ISO.map(s) for s in src]
    gens = [_A.K(1), _A.K(-1), _A.K(0, 1), _A.E(), _A.F(), _A.E(1, 1), _A.F(1, -1),
            _A.E(2, 0), _A.F(2, 0)]
    pairs = [(Element({a: _ONE}), Element({b: _ONE})) for a in gens for b in gens]
    tpairs = [(_ISO.map(a), _ISO.map(b)) for a, b in pairs]
    res = _ISO.verify_all(src, tgt, pairs, tpairs, trace_K=8)
    assert all(res.values()), res


def test_sections_map_to_one_dim_rep_dressed_canonicals():
    labs = [l for l in iso_samples(_A) if l[1] == _A.coefficient_ring().one_basis()]
    assert _ISO.verify_maps_section_to_section_1drep(
        labs, [uq_label_to_chart(_A, l) for l in labs])


def test_rho_is_the_chart_half_monodromy():
    """ρ on the quantum-group side (Lusztig's braid) IS the chart's `ρ` under the
    iso, including its infinite order: ρ⁴(E) = E_{1,4} ↔ (1, 4, 0); and
    ρ(E) ↔ (−1, −2, 0), the lower charge of the draft's `ρ(u₊) = 𝖖v⁻¹u₋`."""
    assert _G.rho((1, 0, 0)) == uq_label_to_chart(_A, _A.rho(_A.E())) == (-1, -2, 0)
    x = (1, 0, 0)
    for _ in range(4):
        x = _G.rho(x)
    assert x == uq_label_to_chart(_A, _A.E(1, 4)) == (1, 4, 0)
    for lab in [_A.E(2, 1), _A.F(2, -1, 1), _A.K(2, 1), _A.F(1, 2)]:
        assert _G.rho(uq_label_to_chart(_A, lab)) == uq_label_to_chart(_A, _A.rho(lab)), lab
        assert _G.rho_inverse(uq_label_to_chart(_A, lab)) == \
            uq_label_to_chart(_A, _A.rho_inverse(lab)), lab


def test_deeper_products_and_dressed_traces():
    """Products of length-2 labels, and Casimir-dressed Cartan traces, agree
    across the iso beyond the battery's basket."""
    for a, b in [(_A.E(2, 1), _A.F(2, -1)), (_A.F(1, 1), _A.E(1, -1, 1)),
                 (_A.E(3, 0), _A.F(1, 2)), (_A.K(2, 1), _A.F(2, 1, 1))]:
        lhs = _ISO.map(_A.multiply(a, b))
        rhs = _G.multiply(uq_label_to_chart(_A, a), uq_label_to_chart(_A, b))
        assert _terms(lhs) == _terms(rhs), (a, b)
    for n in range(-2, 3):
        for k in range(3):
            assert (_A.trace(_A.K(n, k), K=7) - _G.trace((0, n, k), K=7)).is_zero(), (n, k)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, t in tests:
        t()
        print(f"  PASS: {name}")
    print(f"\nAll {len(tests)} UqSU2KAlgebra ≅ SQED_2-chart iso tests passed.")
