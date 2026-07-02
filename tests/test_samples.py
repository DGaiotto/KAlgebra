"""Self-contained contract tests for the sample K_𝖖-algebras.

Run (from the repository root):

    python3 run_tests.py

Exercises, on each sample, the KAlgebra axiom verifiers (unit law,
associativity, ρ-automorphism, ρ⁻¹, ρ²-twisted trace cyclicity) and the
canonical-basis orthonormality relation `Tr(ρ(a)·b) = δ_{a,b} + O(𝖖)`.
"""
import os
import sys

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from samples import (
    Z2QTorusSampleKAlgebra,
    PentagonSampleKAlgebra,
    SQED1SampleKAlgebra,
    SQED2SampleKAlgebra,
    SQEDNfSampleKAlgebra,
)
from kalgebra import Element
from laurent_poly import LaurentPoly
from quantum_torus_kalgebra import QuantumTorusKAlg
from kalgebra_iso import KAlgebraIso


def _check(name, A, labels, K=5, ortho_pairs=None):
    """Run the standard contract battery on algebra `A` over `labels`."""
    for a in labels:
        assert A.verify_unit_law(a), (name, "unit_law", a)
        assert A.verify_rho_inverse(a), (name, "rho_inverse", a)
    for a in labels:
        for b in labels:
            assert A.verify_rho_is_automorphism(a, b), (name, "rho_aut", a, b)
    for a in labels[:4]:
        for b in labels[:4]:
            for c in labels[:3]:
                assert A.verify_associativity(a, b, c), (name, "assoc", a, b, c)
    pairs = ortho_pairs if ortho_pairs is not None else \
        [(a, b) for a in labels for b in labels]
    onf = 0
    for a, b in pairs:
        if not A.verify_orthonormality(a, b, K=K):
            onf += 1
    assert onf == 0, (name, f"orthonormality {onf}/{len(pairs)} failed")
    for a in labels[:5]:
        for b in labels[:5]:
            assert A.verify_trace_pairing_faces(a, b, K=K), (name, "trace_faces", a, b)
    print(f"  OK  {name}: unit/assoc/ρ/ρ²-cyclicity + orthonormality {len(pairs)} pairs")


def test_z2_qtorus():
    A = Z2QTorusSampleKAlgebra()
    labs = [(a, b) for a in range(-1, 2) for b in range(-1, 2)]
    _check("Z2QTorus", A, labs)


def test_pentagon():
    A = PentagonSampleKAlgebra()
    # Canonical labels only: the unit (0,0,0), pure powers L_i^a = (i,a,0), and
    # the mixed L_i L_{i+1} = (i,1,1).  Labels with a=0, b>0 (e.g. (0,0,1)=L_1)
    # are NON-canonical (they reduce, here to (1,1,0)) and are excluded.
    labs = ([(0, 0, 0)]
            + [(i, a, 0) for i in range(5) for a in (1, 2)]
            + [(i, 1, 1) for i in range(5)])
    _check("Pentagon", A, labs)


def test_sqed1():
    A = SQED1SampleKAlgebra()
    labs = [(m, n) for m in range(-1, 2) for n in range(-1, 2)]
    _check("SQED1", A, labs)


def test_sqed2():
    A = SQED2SampleKAlgebra()
    labs = [(("K", n), k) for n in range(-1, 2) for k in range(2)]
    _check("SQED2", A, labs)


def test_sqednf():
    for Nf in (1, 2, 3):
        A = SQEDNfSampleKAlgebra(Nf)
        R = A.coefficient_ring()
        irs = [R.one_basis()] + ([(1,)] if Nf >= 2 else [])
        labs = [(m, n, w) for m in range(-1, 2) for n in range(-1, 2) for w in irs]
        _check(f"SQEDNf({Nf})", A, labs, K=4)


def test_quantum_torus_gamma():
    """`QuantumTorusKAlg(B)` — the quantum torus as a function of the lattice
    `Γ = Z^n` with antisymmetric form `B`.  Two regimes:
      * non-degenerate `B` (`Γ_f = 0`): the plain symplectic Z² torus;
      * degenerate `B` (`Γ_f ≠ 0`): a flavour direction lands in the
        coefficient ring, exercised through the flavoured trace."""
    # non-degenerate: Z² symplectic ⟨(a,b),(c,d)⟩ = ad − bc
    Q2 = QuantumTorusKAlg([[0, 1], [-1, 0]])
    assert Q2.flavour_rank == 0 and Q2.gauge_rank == 2
    labs2 = [(a, b) for a in range(-1, 2) for b in range(-1, 2)]
    _check("QuantumTorusKAlg(Z²,symplectic)", Q2, labs2)
    # degenerate: Z³ with a rank-1 kernel (3rd direction = flavour)
    Q3 = QuantumTorusKAlg([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    assert Q3.flavour_rank == 1 and Q3.gauge_rank == 2
    labs3 = [(a, b, f) for a in range(2) for b in range(2) for f in range(-1, 2)]
    _check("QuantumTorusKAlg(Z³,1 flavour)", Q3, labs3, K=4)


def test_kalgebra_iso():
    """`KAlgebraIso` — a structure-preserving correspondence between two
    presentations.  Witness: the B-preserving frame change `M = [[1,1],[0,1]]`
    (det 1 ⇒ `MᵀBM = B`) as an *automorphism* of the symplectic Z² torus,
    `(a,b) ↦ (a+b, b)`.  Verified in both directions: unit, round-trip,
    multiplicativity, and ρ-equivariance."""
    Q = QuantumTorusKAlg([[0, 1], [-1, 0]])
    one = LaurentPoly.one()

    def fwd(lbl):                       # M·(a,b) = (a+b, b)
        a, b = lbl
        return Element({(a + b, b): one})

    def inv(lbl):                       # M⁻¹·(a,b) = (a−b, b)
        a, b = lbl
        return Element({(a - b, b): one})

    iso = KAlgebraIso(Q, Q, fwd, inv, name="QT frame change M=[[1,1],[0,1]]")
    samples = [Element({g: one}) for g in [(1, 0), (0, 1), (1, 1), (2, -1)]]
    pairs = [(Element({(1, 0): one}), Element({(0, 1): one})),
             (Element({(1, 1): one}), Element({(1, 0): one}))]
    assert iso.verify_unit(), "iso unit"
    assert iso.verify_round_trip(samples, samples), "iso round-trip"
    assert iso.verify_multiplicative(pairs, pairs), "iso multiplicative"
    assert iso.verify_rho_equivariant(samples, samples), "iso ρ-equivariance"
    print("  OK  KAlgebraIso: unit / round-trip / multiplicative / ρ-equivariant")


def test_trace_element_widening_and_bilinear_pairing():
    """Formal-sum assembly of the trace pairing.

    (1) `trace_element` must widen each per-label trace request by the
    negative q-valuation of that label's Laurent coefficient: `q^{-n}·Tr(L_a)`
    is exact through `q^K` only if `Tr(L_a)` is known through `q^{K+n}`.
    Without the widening every order above `q^{K-n}` of the sum is silently
    corrupted — on deep products (Laurent coefficients down to `q^{-10}` occur
    already in the `[A_1,D_3]` mixed tiles of the cone layer) this produced
    spurious orthonormality violations.

    (2) `inner_product_element` is the bilinear extension of the pairing
    `I_{a,b}` over formal sums — safer than assembling `Tr(ρ(x)·y)` from a
    product Element, and it routes through a realisation's sharp per-pair
    `inner_product` override where one exists."""
    from zplus_ring import RPowerSeries
    A = PentagonSampleKAlgebra()
    R = A.coefficient_ring()
    lab = (0, 1, 0)
    K, shift = 4, -3
    x = Element({lab: LaurentPoly({shift: 1})})
    got = A.trace_element(x, K)
    tr_wide = A.trace(lab, K - shift)
    expected = RPowerSeries(
        R,
        {e + shift: c for e, c in tr_wide.coeffs.items() if e + shift <= K},
        K,
    )
    assert got == expected, "trace_element did not widen for q^-3 coefficient"
    assert any(K < e <= K - shift for e in tr_wide.coeffs), \
        "sensitivity lost: no trace tail in the widened window"
    # bilinear pairing: q^{-2}·L_a + L_e against L_a.
    e_id = A.identity()
    y = Element({lab: LaurentPoly({-2: 1}), e_id: LaurentPoly({0: 1})})
    got2 = A.inner_product_element(y, Element.basis(lab), K)
    iaa_wide = A.inner_product(lab, lab, K + 2)
    expected2 = RPowerSeries(
        R,
        {e - 2: v for e, v in iaa_wide.coeffs.items() if e - 2 <= K},
        K,
    ) + A.inner_product(e_id, lab, K)
    assert got2 == expected2, "inner_product_element bilinearity mismatch"
    assert A.inner_product_element(
        Element.basis(lab), Element.basis(lab), K) == A.inner_product(lab, lab, K)
    print("  OK  trace_element widening + bilinear inner_product_element")


if __name__ == "__main__":
    print("running sample contract tests...")
    test_z2_qtorus()
    test_pentagon()
    test_sqed1()
    test_sqed2()
    test_sqednf()
    test_quantum_torus_gamma()
    test_kalgebra_iso()
    test_trace_element_widening_and_bilinear_pairing()
    print("ALL SAMPLE CONTRACT TESTS PASSED")
