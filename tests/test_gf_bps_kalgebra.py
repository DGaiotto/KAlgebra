"""Tests for `GfBPSKAlgebra` -- the G-flavoured BPS realisation (the
Weyl-invariant subalgebra of a `BPSKAlgebra` for an `S_n`-symmetric quiver).

  * SU(2) (S_2 = Z_2): reproduces `SU2BPSKAlgebra` on [A_1, D_3];
  * SU(3) (S_3): central multiply == R(SU(3)) fusion -- the fix for the
    `Z_3`-based `SU3BPSKAlgebra`, which returns the wrong `3 x 3 = 6 + 2.3bar`.

Run: `python3 run_tests.py`
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from gf_bps_kalgebra import GfBPSKAlgebra
from su2_bps_kalgebra import SU2BPSKAlgebra
from zplus_ring import SU2ZPlusRing, SU3ZPlusRing


# --- builders ---------------------------------------------------------------

_SU2_P = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
_SU2_SPEC = [(1, 0, 0), (0, 1, 1), (0, 1, -1)]
_SU2_W = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]

_SU3_P = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
_SU3_NODES = [(1, 0, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 1, -1, -1)]
_SU3_TA = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
_SU3_TB = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, -1]]


def _su2():
    return GfBPSKAlgebra(_SU2_P, _SU2_SPEC, [_SU2_W], SU2ZPlusRing(),
                         mu_basis=[(0, 0, 1)])


def _su3():
    return GfBPSKAlgebra(_SU3_P, _SU3_NODES, [_SU3_TA, _SU3_TB], SU3ZPlusRing(),
                         mu_basis=[(0, 0, 1, 0), (0, 0, 0, 1)])


def _fexp(alg, e):
    _, x = alg.to_u1(e)
    return {k: dict(v._coeffs) for k, v in x.terms.items()}


# --- construction & validation ---------------------------------------------

def test_W_orders():
    assert len(_su2()._W) == 2     # S_2
    assert len(_su3()._W) == 6     # S_3


def test_reflection_must_permute_spec():
    # negate the gauge direction e_1: a Poisson involution that does NOT
    # permute the spec
    bad = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    try:
        GfBPSKAlgebra(_SU2_P, _SU2_SPEC, [bad], SU2ZPlusRing(), mu_basis=[(0, 0, 1)])
    except ValueError as e:
        assert "permute the spec" in str(e) or "not in ker B" in str(e) \
            or "Poisson" in str(e)
    else:
        raise AssertionError("expected ValueError")


# --- SU(2): reproduces SU2BPSKAlgebra ---------------------------------------

def test_su2_reproduces_su2bps():
    G = _su2()
    S = SU2BPSKAlgebra(_SU2_P, _SU2_SPEC, w_su2=_SU2_W)
    sample = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (0, 1, 1)]
    for a in sample:
        for b in sample:
            ga = _fexp(G, G.multiply(G.canonicalise(a), G.canonicalise(b)))
            sa = _fexp(S, S.multiply(S.canonicalise(a), S.canonicalise(b)))
            assert ga == sa, (a, b)


def test_su2_clebsch_gordan():
    G = _su2()
    assert sorted(G.ring_label_of(l)
                  for l in G.multiply(G.character_label(1),
                                      G.character_label(1)).terms) == [0, 2]
    assert sorted(G.ring_label_of(l)
                  for l in G.multiply(G.character_label(1),
                                      G.character_label(2)).terms) == [1, 3]


# --- SU(3): correct fusion (the Z_3 -> S_3 fix) -----------------------------

def test_su3_fundamental_squared_is_6_plus_3bar():
    """3 x 3 = 6 + 3bar, each multiplicity 1 -- the full-S_3 fix for the
    retired Z_3-based SU3BPSKAlgebra (which returned 6 + 2.3bar)."""
    A = _su3()
    prod = A.multiply(A.character_label((1, 0)), A.character_label((1, 0)))
    got = {A.ring_label_of(l): dict(c._coeffs) for l, c in prod.terms.items()}
    assert got == {(2, 0): {0: 1}, (0, 1): {0: 1}}, got


def test_su3_central_multiply_is_fusion():
    A = _su3()
    R = A.coefficient_ring()
    for p1, q1 in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]:
        for p2, q2 in [(1, 0), (0, 1), (1, 1)]:
            prod = A.multiply(A.character_label((p1, q1)),
                              A.character_label((p2, q2)))
            got = {A.ring_label_of(l): dict(c._coeffs) for l, c in prod.terms.items()}
            exp = {k: {0: m} for k, m in R.multiply_basis((p1, q1), (p2, q2)).items()}
            assert got == exp, ((p1, q1), (p2, q2), got, exp)


def test_su3_rho_is_dual():
    A = _su3()
    for (p, q) in [(1, 0), (2, 1), (1, 2)]:
        assert A.ring_label_of(A.rho(A.character_label((p, q)))) == (q, p)


def test_su3_trace_central_characters():
    A = _su3()
    assert A.trace(A.identity(), K=4).coeffs[0].terms == {(0, 0): 1}        # chi_0
    assert A.trace(A.character_label((1, 0)), K=4).coeffs[0].terms == {(1, 0): 1}


def test_su3_orthonormality_central():
    A = _su3()
    cl = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for a in cl:
        for b in cl:
            assert A.verify_orthonormality(A.character_label(a),
                                           A.character_label(b), K=4)


def test_contract_verifiers():
    A = _su3()
    cl = [(0, 0), (1, 0), (0, 1), (1, 1)]
    labs = [A.character_label(x) for x in cl]
    assert all(A.verify_bar_involution(a, b) for a in labs for b in labs)
    assert all(A.verify_rho_is_automorphism(a, b) for a in labs for b in labs)
    assert all(A.verify_rho_twisted_trace(a, b, 4) for a in labs[:3] for b in labs[:3])


# --- product flavour group  prod_a SU(N_a) --------
#
# "Extend GfBPSKAlgebra if that was the most recent attempt to flavoured
# BPSKAlgebras" -- it was (SU3BPSKAlgebra's standalone was retired for it
# 2026-06-13, the flavour_enhancement / sun_flavour_enhancement / su2n_flavour
# wrappers were deprecated in its favour 2026-06-19, and gf_bps_sqed_nf extended
# it 2026-06-20; nothing in the tree supersedes it).
#
# The extension is the BLOCK generalisation: a product's weight coordinates are
# the concatenation of the factors', so dominance, the orbit-ordering key and the
# weight<->label maps act factor by factor.  It is shape-preserving -- every test
# above is a single simple factor and none of them changed.


def _flavoured(factors, pairing=((0, 1), (-1, 0))):
    """`GfBPSKAlgebra` from a `FlavouredQuiver`, via `gf_chart_data()`."""
    from flavoured_factor_spectrum import FlavouredQuiver
    fq = FlavouredQuiver([list(r) for r in pairing], list(factors))
    P, C, Rf, ring, mu = fq.gf_chart_data()
    return fq, GfBPSKAlgebra(P, C, Rf, ring, mu_basis=mu)


def test_generated_chart_data_is_the_hand_written_convention():
    """`gf_chart_data()` reproduces THIS module's own hand-written matrices.

    The positive control for the generated data: if the derived reflections and
    `mu_basis` were a parallel convention rather than the established one, every
    product-group result below would be measuring the wrong object.  (This caught
    a real error while it was written -- the reflections were generated at the
    weight block's offset within `P` instead of within the full lattice, so they
    acted on the gauge block.)
    """
    _fq, _A = _flavoured([1, 2])
    from flavoured_factor_spectrum import FlavouredQuiver
    P, C, Rf, ring, mu = FlavouredQuiver([[0, 1], [-1, 0]], [1, 2]).gf_chart_data()
    assert P == _SU2_P, P
    assert sorted(C) == sorted(_SU2_SPEC), C
    assert Rf == [_SU2_W], Rf
    assert mu == [(0, 0, 1)], mu

    P, C, Rf, ring, mu = FlavouredQuiver([[0, 1], [-1, 0]], [1, 3]).gf_chart_data()
    assert P == _SU3_P, P
    assert sorted(C) == sorted(_SU3_NODES), C
    assert sorted(map(str, Rf)) == sorted(map(str, [_SU3_TA, _SU3_TB])), Rf
    assert mu == [(0, 0, 1, 0), (0, 0, 0, 1)], mu


def test_product_group_builds_with_the_product_weyl_group():
    """`W = prod_a S_{N_a}`, and the ring is the product `Z_+`-ring."""
    from zplus_ring import TensorZPlusRing

    for factors, order in ([2, 2], 4), ([2, 3], 12), ([3, 3], 36):
        _fq, A = _flavoured(factors)
        assert len(A._W) == order, (factors, len(A._W))
        assert isinstance(A.coefficient_ring(), TensorZPlusRing)
        assert len(A.coefficient_ring().factors) == len(factors)
        assert A.flavour_blocks() == tuple(("sun", n - 1) for n in factors)


def test_product_group_fuses_in_the_RIGHT_factor():
    """Clebsch-Gordan acts factor-wise, and the discriminator is WHICH factor.

    `f_0 (x) f_0` must land in factor 0's adjoint and leave factor 1 trivial, and
    conversely -- a chart that fused into a single lumped group would put both in
    the same slot.  Checked against the ring's own `multiply_basis`, so the
    algebra and the coefficient ring have to agree independently.
    """
    _fq, A = _flavoured([2, 2])
    R = A.coefficient_ring()
    f0 = A.canonicalise((1, 0, 1, 0))
    f1 = A.canonicalise((0, 1, 0, 1))
    assert A.ring_label_of(f0) == ((1,), ()), A.ring_label_of(f0)
    assert A.ring_label_of(f1) == ((), (1,)), A.ring_label_of(f1)

    got0 = {A.ring_label_of(k) for k in A.multiply(f0, f0).terms}
    assert got0 == set(R.multiply_basis(((1,), ()), ((1,), ()))), got0
    assert got0 == {((), ()), ((2,), ())}, got0

    got1 = {A.ring_label_of(k) for k in A.multiply(f1, f1).terms}
    assert got1 == {((), ()), ((), (2,))}, got1

    cross = {A.ring_label_of(k) for k in A.multiply(f0, f1).terms}
    assert cross == set(R.multiply_basis(((1,), ()), ((), (1,)))), cross
    assert cross == {((1,), (1,))}, cross


def test_product_group_trace_and_axioms():
    """The trace lands in the PRODUCT ring, and the contract verifiers hold."""
    from zplus_ring import TensorZPlusRing

    _fq, A = _flavoured([2, 2])
    labs = [A.identity(), A.canonicalise((1, 0, 1, 0)), A.canonicalise((0, 1, 0, 1))]
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    for a in labs:
        assert isinstance(A.trace(a, 4).ring, TensorZPlusRing)
        assert A.trace(a, 4).ring == A.coefficient_ring()
        assert A.verify_rho_inverse(a), a
        assert A.verify_orthonormality(a, a, K=4), a
    for a in labs:
        for b in labs:
            assert A.verify_bar_involution(a, b), (a, b)
            assert A.verify_rho_is_automorphism(a, b), (a, b)


def test_agrees_with_the_independent_gbps_construction():
    """Cross-certification against `gbps_kalgebra.py::GBPSKAlgebra`.

    Two genuinely independent routes to the same algebra: this class lifts a
    label to its Weyl-character diagram in the F-basis, multiplies pairwise and
    peels the most-dominant character back out; `GBPSKAlgebra` expands onto the
    `Gamma_gauge (+) P` chart, multiplies there, and recovers the irreps through
    `FlavourSpace.decompose` (i.e. the certified `SUNZPlusRing.from_abelian`).
    Neither shares code with the other.
    """
    import itertools
    from gbps_kalgebra import GBPSKAlgebra

    for factors in ([1, 2], [1, 3]):
        fq, A = _flavoured(factors)
        G = GBPSKAlgebra(fq, max_irrep=4)
        F, g = fq.flavour, fq.rank

        def to_gf(lab):
            return A.r_label_compose(tuple(lab[0]) + tuple([0] * F.dim),
                                     F.irrep_to_ring_key(lab[1]))

        def from_gf(lab):
            sec, key = A.r_label_decompose(lab)
            return (tuple(sec[:g]), F.ring_key_to_irrep(key))

        assert to_gf(G.identity()) == A.identity()      # positive control

        labels = [(c, r) for c in itertools.product(range(3), repeat=g)
                  for r in G.admissible_irreps(c)][:10]
        checked = 0
        for a in labels:
            for b in labels:
                want = {from_gf(k): str(v)
                        for k, v in A.multiply(to_gf(a), to_gf(b)).terms.items()}
                got = {k: str(v) for k, v in G.multiply(a, b).terms.items()}
                assert got == want, (factors, a, b, got, want)
                checked += 1
        assert checked == 100, checked


def test_canonicalise_matches_the_naive_orbit_loop():
    """The `O(|W|)` canonicalisation returns exactly the naive `O(|W|²)` labels.

    `_weight` itself averages over `W`, so recomputing it once per group element
    squares the cost — invisible for one simple factor, fatal for a product where
    `|W| = ∏_a N_a!` (measured 4.5 s per call at `|W| = 144`, ~70 s at
    `SU(4)×SU(4)`).  Hoisting it relies on `weight(w·γ) = M_w · weight(γ)`, which
    holds because `_proj_kerB` is the `W`-average and so `W`-equivariant.  This
    test is the guard on that: the labels must not move.
    """
    import itertools
    from gf_quantum_torus_kalgebra import _matvec

    def naive(A, gamma):
        gamma = A._check(gamma)
        best, best_key = None, None
        for w in A._W:
            g = _matvec(w, gamma)
            wt = A._weight(g)                 # recomputed per w, as before
            if not A._is_dominant(wt):
                continue
            key = A._dom_key(wt)
            if best is None or key > best_key:
                best, best_key = g, key
        if best is None:
            raise ValueError("no dominant rep")
        return best

    checked = 0
    for factors in ([1, 2], [1, 3], [2, 2], [2, 3]):
        _fq, A = _flavoured(factors)
        pts = set()
        for combo in itertools.product(range(-2, 3), repeat=min(A._rank, 4)):
            pts.add(tuple(list(combo) + [0] * (A._rank - len(combo))))
        for g in sorted(pts)[:60]:
            try:
                want = naive(A, g)
            except ValueError:
                continue
            assert A.canonicalise(g) == want, (factors, g)
            checked += 1
    assert checked > 200, checked


def test_the_weyl_cap_is_a_budget_not_a_finiteness_verdict():
    """`|W| = ∏_a N_a!` MULTIPLIES, so the old cap rejected finite groups.

    `SU(5)×SU(5)` has `|W| = 14400`, which the previous `cap=4096` refused with
    `"Weyl group not finite"` — a false statement about a finite group, not a
    budget message.  Asserted here on a product the suite can afford, plus the
    wording of the refusal when the budget really is exceeded.
    """
    _fq, A = _flavoured([4, 4])
    assert len(A._W) == 576, len(A._W)          # 4! x 4!, comfortably finite

    # a deliberately tiny budget must say BUDGET, not "not finite"
    A2 = GfBPSKAlgebra.__new__(GfBPSKAlgebra)
    A2._rank = 2
    try:
        A2._generate_group([((0, 1), (1, 0))], cap=0)
    except ValueError as exc:
        assert "cap" in str(exc) and "finiteness verdict" in str(exc), str(exc)
    else:
        raise AssertionError("expected the budget refusal")


def test_mu_basis_block_order_is_checked():
    """A `mu_basis` permuted across factors is caught, not silently mis-read.

    `TensorZPlusRing.to_abelian` emits CONCATENATED per-factor coordinates and
    every block helper slices by offset, so a `mu_basis` whose order does not
    match `coefficient_ring().factors` reads one factor's weights in another's
    convention — wrong labels, no exception.  `W = ∏_a W_a` acts factor-wise, so
    the discriminator is that each `w` is block-diagonal on weight coordinates.
    """
    for factors in ([1, 2], [1, 3], [2, 2], [2, 3], [3, 3]):
        _fq, A = _flavoured(factors)
        assert A.verify_mu_basis_blocks(), factors

    # NEGATIVE CONTROL: swap an SU(2) slot with an SU(3) slot
    from flavoured_factor_spectrum import FlavouredQuiver
    fq = FlavouredQuiver([[0, 1], [-1, 0]], [2, 3])
    P, C, Rf, ring, mu = fq.gf_chart_data()
    permuted = [mu[1], mu[0], mu[2]]
    B = GfBPSKAlgebra(P, C, Rf, ring, mu_basis=permuted)
    assert not B.verify_mu_basis_blocks()


# --- runner -----------------------------------------------------------------

def main():
    tests = [
        test_W_orders,
        test_reflection_must_permute_spec,
        test_su2_reproduces_su2bps,
        test_su2_clebsch_gordan,
        test_su3_fundamental_squared_is_6_plus_3bar,
        test_su3_central_multiply_is_fusion,
        test_su3_rho_is_dual,
        test_su3_trace_central_characters,
        test_su3_orthonormality_central,
        test_contract_verifiers,
        test_generated_chart_data_is_the_hand_written_convention,
        test_product_group_builds_with_the_product_weyl_group,
        test_product_group_fuses_in_the_RIGHT_factor,
        test_product_group_trace_and_axioms,
        test_agrees_with_the_independent_gbps_construction,
        test_canonicalise_matches_the_naive_orbit_loop,
        test_the_weyl_cap_is_a_budget_not_a_finiteness_verdict,
        test_mu_basis_block_order_is_checked,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [ok]   {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n  Results: {passed} passed, {failed} failed of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
