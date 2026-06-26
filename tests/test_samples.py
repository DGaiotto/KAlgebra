"""Self-contained contract tests for the sample K_𝖖-algebras.

Run from inside the package folder:

    python test_samples.py

Exercises, on each sample, the KAlgebra axiom verifiers (unit law,
associativity, ρ-automorphism, ρ⁻¹, ρ²-twisted trace cyclicity) and the
canonical-basis orthonormality relation `Tr(ρ(a)·b) = δ_{a,b} + O(𝖖)`.
"""
import os
import sys

# The package modules live in this same (flat) folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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
        coefficient ring, exercised at the trace boundary."""
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


if __name__ == "__main__":
    print("running sample contract tests...")
    test_z2_qtorus()
    test_pentagon()
    test_sqed1()
    test_sqed2()
    test_sqednf()
    test_quantum_torus_gamma()
    test_kalgebra_iso()
    print("ALL SAMPLE CONTRACT TESTS PASSED")
