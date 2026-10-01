"""Tests for `A1A2kRGKAlgebra` — `[A_1, A_{2k}]` as a new-contract
`RGKAlgebra` over `A1A2kKAlg(k-1) ⊗ QT(Z₂)` (drop the second node γ₂),
Γ_RG = Z² (the QT charge), `S_RG = E_𝖖(X_{γ₁})·E_𝖖(X_{γ₂}·L)` with `L`
the short dressing chord.

The full `KAlgebra` API (`RG` via the graded solver, `multiply`, `rho`,
`trace`) is the generic RGKAlgebra default — no UV BPS F-finder.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "implementations"))

from kalgebra import Element
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from a1a2k_rgkalgebra import A1A2kRGKAlgebra

ONE = LaurentPoly.one()


def _aux_node_gens(A):
    """Auxiliary node generators: each survivor chord generator at qt=(0,0),
    plus the two QT generators γ₁=(1,0), γ₂=(0,1)."""
    surv = A.survivor
    sid = surv.identity()
    gens = []
    if A.k >= 2:
        for a in range(1, surv.k + 1):
            gens.append((surv.L((a, 0)), (0, 0)))
    gens += [(sid, (1, 0)), (sid, (0, 1))]
    return gens


def _mul_elem(A, x, y):
    out = Element({})
    for la, ca in x.terms.items():
        for lb, cb in y.terms.items():
            p = A.multiply(la, lb)
            out = out + Element({l: ca * cb * c for l, c in p.terms.items()})
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction():
    from tensor_kalgebra import TensorKAlgebra
    from quantum_torus_kalgebra import QuantumTorusKAlg
    from a1a2k_kalg import A1A2kKAlg
    for k in (1, 2, 3):
        A = A1A2kRGKAlgebra(k)
        aux = A.auxiliary()
        assert isinstance(aux, TensorKAlgebra), k
        assert isinstance(aux.factor_B, QuantumTorusKAlg), k
        if k >= 2:
            assert isinstance(aux.factor_A, A1A2kKAlg), k
            assert aux.factor_A.k == k - 1, k
    # grading is Z^2 (the qt charge); identity is the aux identity.
    A2 = A1A2kRGKAlgebra(2)
    g = A2.grading()
    assert g.rank == 2 and g.height == (1, 1)
    assert g.charge((A2.survivor.L((1, 0)), (3, -2))) == (3, -2)
    assert A2.identity() == A2.auxiliary().identity()


def test_dressing_is_short_chord():
    """The S_RG dressing L is a *short* chord (letter 1) for all k — the
    second-node-drop result."""
    for k in (2, 3, 4):
        A = A1A2kRGKAlgebra(k)
        assert A.dressing_chord_index is not None, k
        # S_RG dressed γ₂-tower lives on short chords ((1, i0, m),).
        srg = A.rg_generator(4)
        for (chord, qt), _ in srg.items():
            if qt[1] > 0:                       # dressed γ₂-tower term
                assert all(c[0] == 1 for c in chord), (k, chord)


def test_s_rg_unit_and_positive_cone():
    """`[S_RG]_0 = 1_aux` and the whole S_RG support has positive height
    (lies in the obvious cone Z²_{≥0})."""
    for k in (1, 2, 3):
        A = A1A2kRGKAlgebra(k)
        g = A.grading()
        srg = A.rg_generator(A._rg_cutoff())
        aux_id = A.auxiliary().identity()
        assert srg[aux_id].expand(2) == LaurentPoly.one(), k
        for lbl, h in srg.items():
            if lbl == aux_id:
                continue
            c = g.charge(lbl)
            assert g.h(c) > 0, (k, lbl, c)
            assert c[0] >= 0 and c[1] >= 0, (k, lbl, c)   # obvious cone


# ---------------------------------------------------------------------------
# k = 1 : reproduce BPS(A_2) (pentagon) exactly
# ---------------------------------------------------------------------------

def test_k1_matches_pentagon():
    A = A1A2kRGKAlgebra(1)
    PENT = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                       node_charges=[(1, 0), (0, 1)],
                       spec=[(1, 0), (0, 1)], verify="off")
    # UV label ((), (c1,c2)) <-> pentagon charge (c1,c2).
    triv_id = A.survivor.identity()

    def lift(c):
        return (triv_id, tuple(c))

    gens = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    for a in gens:
        for b in gens:
            ours = {l[1]: v
                    for l, v in A.multiply(lift(a), lift(b)).terms.items()}
            ref = dict(PENT.multiply(a, b).terms)
            assert ours == ref, (a, b, ours, ref)
    # rho / rho_inverse
    for a in gens:
        assert A.rho(lift(a))[1] == tuple(PENT.rho(a)), a
        assert A.rho_inverse(lift(a))[1] == tuple(PENT.rho_inverse(a)), a
    # trace (Schur index)
    for a in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        assert str(A.trace(lift(a), K=6)) == str(PENT.trace(a, K=6)), a


def test_k1_rg_is_multiterm():
    """RG is the genuine (S_RG-mediated) homomorphism, not the Darboux
    relabel: RG(γ₂) has the extra γ₁+γ₂ term."""
    A = A1A2kRGKAlgebra(1)
    tid = A.survivor.identity()
    rg = A.RG((tid, (0, 1)))
    assert (tid, (1, 1)) in rg.terms and (tid, (0, 1)) in rg.terms


# ---------------------------------------------------------------------------
# k >= 2 : intrinsic axioms (RG is the genuine solve)
# ---------------------------------------------------------------------------

def test_rg_unital():
    for k in (1, 2, 3):
        assert A1A2kRGKAlgebra(k).verify_rg_unital(), k


def test_rg_multiplicative():
    for k in (1, 2):
        A = A1A2kRGKAlgebra(k)
        gens = _aux_node_gens(A)
        for a in gens:
            for b in gens:
                assert A.verify_rg_multiplicative(a, b), (k, a, b)


def test_associativity():
    """Associativity on a few representative triples.  (Full associativity
    is implied by `verify_rg_multiplicative` — RG is a homomorphism and the
    auxiliary is associative — so this is a light spot-check; deep γ₂-power
    towers are avoided to keep the high-charge RG solves cheap.)"""
    one = ONE
    for k in (1, 2):
        A = A1A2kRGKAlgebra(k)
        gens = _aux_node_gens(A)
        triples = [(gens[0], gens[-2], gens[-1]),     # (chord/γ₁, γ₁, γ₂)
                   (gens[-1], gens[0], gens[-2]),
                   (gens[-2], gens[-1], gens[0])]
        for a, b, c in triples:
            ea = Element({a: one}); eb = Element({b: one}); ec = Element({c: one})
            left = _mul_elem(A, _mul_elem(A, ea, eb), ec)
            right = _mul_elem(A, ea, _mul_elem(A, eb, ec))
            assert left == right, (k, a, b, c)


if __name__ == "__main__":
    test_construction()
    test_dressing_is_short_chord()
    test_s_rg_unit_and_positive_cone()
    test_k1_matches_pentagon()
    test_k1_rg_is_multiterm()
    test_rg_unital()
    test_rg_multiplicative()
    test_associativity()
    print("\nAll A1A2kRGKAlgebra tests passed.")
