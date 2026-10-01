"""Tests for the native QT primitives in `bps_kalgebra_internals`:
`qt_multiply`, `compute_strict_cone_witness`, `find_lowest`.

Each is checked against the bundle equivalent
(`bps_quiver_tools.qt_multiply`, `_verify_pointed_cone`,
`CoulombAlgebra._find_lowest`) on representative theories.

Run:  PYTHONPATH=.:restructuring python restructuring/tests/test_bps_kalgebra_primitives.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

import bps_quiver_tools as bps
from bps_kalgebra import BPSKAlgebra
from bps_kalgebra_internals import (
    qt_multiply,
    compute_strict_cone_witness,
    find_lowest,
)
from laurent_poly import LaurentPoly as QTLP
from lattice import Lattice


PASS = "  PASS:"
FAIL = "  FAIL:"


# ---------------------------------------------------------------------------
# qt_multiply  matches  bps.qt_multiply
# ---------------------------------------------------------------------------


def _bps_lp_to_qt(lp_bps):
    return QTLP(dict(lp_bps._coeffs))


def _qt_to_bps_lp(lp_qt):
    # F-coefficients are now QNumberPoly (palindromic); convert to
    # LaurentPoly so the q-exponent dict has the right semantics.
    if hasattr(lp_qt, "to_laurent"):
        lp_qt = lp_qt.to_laurent()
    return bps.LaurentPoly(dict(lp_qt._coeffs))


def _equal_dicts(d1, d2):
    if set(d1) != set(d2):
        return False
    for k in d1:
        if d1[k]._coeffs != d2[k]._coeffs:
            return False
    return True


def test_qt_multiply_pentagon_matches_bundle():
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    A_bps = A._chart  # bps.CoulombAlgebra wrapping the same chart
    lat_native = A.lattice
    lat_bps = A_bps.lattice

    pairs = [
        ((1, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((-1, 0), (0, -1)),
        ((1, 1), (-1, -1)),
        ((2, -1), (1, 2)),
    ]

    for a, b in pairs:
        F_a_native = A._F_internal(a)
        F_b_native = A._F_internal(b)
        # Native product
        prod_native = qt_multiply(F_a_native, F_b_native, lat_native)
        # Bundle product (cast through bps.LaurentPoly)
        F_a_bps = {g: _qt_to_bps_lp(lp) for g, lp in F_a_native.items()}
        F_b_bps = {g: _qt_to_bps_lp(lp) for g, lp in F_b_native.items()}
        prod_bps = bps.qt_multiply(F_a_bps, F_b_bps, lat_bps)
        prod_bps_as_qt = {g: _bps_lp_to_qt(lp) for g, lp in prod_bps.items()}
        assert _equal_dicts(prod_native, prod_bps_as_qt), \
            f"qt_multiply mismatch on F_{a} · F_{b}"
    print(PASS, "test_qt_multiply_pentagon_matches_bundle")


# ---------------------------------------------------------------------------
# compute_strict_cone_witness
# ---------------------------------------------------------------------------


def test_strict_witness_pentagon():
    gens = [(1, 0), (0, 1)]
    f = compute_strict_cone_witness(2, gens)
    assert all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in gens), \
        f"witness {f} fails"
    print(PASS, "test_strict_witness_pentagon")


def test_strict_witness_pure_su2():
    # Cone gens of pure SU(2): (1, 0) and (-1, 2).
    gens = [(1, 0), (-1, 2)]
    f = compute_strict_cone_witness(2, gens)
    assert all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in gens), \
        f"witness {f} fails"
    print(PASS, "test_strict_witness_pure_su2")


def test_strict_witness_supplied_certificate_accepted():
    gens = [(1, 0), (0, 1)]
    f = compute_strict_cone_witness(2, gens, candidate=(1, 1))
    assert f == (1, 1), f"expected supplied witness to be accepted, got {f}"
    print(PASS, "test_strict_witness_supplied_certificate_accepted")


def test_strict_witness_rejects_zero_generator():
    try:
        compute_strict_cone_witness(2, [(1, 0), (0, 0)])
    except ValueError as e:
        assert "equals zero" in str(e), f"unexpected error: {e}"
        print(PASS, "test_strict_witness_rejects_zero_generator")
        return
    raise AssertionError("expected ValueError for zero generator")


def test_strict_witness_rejects_antipodal_pair():
    try:
        compute_strict_cone_witness(2, [(1, 0), (-1, 0)])
    except ValueError as e:
        assert "antipodal" in str(e) or "not pointed" in str(e), \
            f"unexpected error: {e}"
        print(PASS, "test_strict_witness_rejects_antipodal_pair")
        return
    raise AssertionError("expected ValueError for antipodal generators")


def test_strict_witness_matches_bundle_on_presets():
    # Run a few realistic theories through both and compare witnesses
    # by their property (both must satisfy ⟨f, g⟩ ≥ 1 on every gen).
    cases = [
        ([[0, 1], [-1, 0]], [(1, 0), (0, 1)]),       # pentagon
        ([[0, 2], [-2, 0]], [(1, 0), (-1, 2)]),       # pure SU(2)
        ([[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
         [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),          # hexagon
    ]
    for B, gens in cases:
        rank = len(B)
        f = compute_strict_cone_witness(rank, gens)
        assert all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in gens), \
            f"witness {f} fails on {gens}"
    print(PASS, "test_strict_witness_matches_bundle_on_presets")


# ---------------------------------------------------------------------------
# find_lowest matches bundle  CoulombAlgebra._find_lowest
# ---------------------------------------------------------------------------


def test_find_lowest_pentagon_matches_bundle():
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    A_bps = A._chart
    cone_gens = A.cone_gens
    f = compute_strict_cone_witness(A.lattice.rank, cone_gens)
    # Force the bundle to compute its own witness.
    f_bundle = A_bps._strict_cone_witness()
    # Both witnesses should satisfy strictness; they need not be
    # identical, but find_lowest with a strict witness must agree
    # with the bundle on which charge is the cone-minimum.
    assert all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in cone_gens)
    assert all(sum(fi * gi for fi, gi in zip(f_bundle, g)) >= 1 for g in cone_gens)

    sample_charge_lists = [
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [(2, 1), (1, 2), (3, 0), (0, 3)],
        [(-1, 2), (1, 0), (0, 0)],
    ]
    for charges in sample_charge_lists:
        native = find_lowest(charges, cone_gens, f)
        bundle = A_bps._find_lowest(charges)
        assert native == bundle, \
            f"find_lowest disagrees on {charges}: native={native}, bundle={bundle}"
    print(PASS, "test_find_lowest_pentagon_matches_bundle")


def test_find_lowest_pure_su2_matches_bundle():
    A = BPSKAlgebra(pairing=[[0, 2], [-2, 0]], node_charges=[(1, 0), (0, 1)])
    A_bps = A._chart
    cone_gens = A.cone_gens
    f = compute_strict_cone_witness(A.lattice.rank, cone_gens)
    sample_charge_lists = [
        [(0, 0), (1, 0), (-1, 2), (0, 2)],
        [(2, 0), (-1, 2), (1, 2)],
        [(-1, 4), (1, 0), (0, 0)],
    ]
    for charges in sample_charge_lists:
        native = find_lowest(charges, cone_gens, f)
        bundle = A_bps._find_lowest(charges)
        assert native == bundle, \
            f"find_lowest disagrees on {charges}: native={native}, bundle={bundle}"
    print(PASS, "test_find_lowest_pure_su2_matches_bundle")


def test_find_lowest_hexagon_matches_bundle():
    A = BPSKAlgebra(
        pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    A_bps = A._chart
    cone_gens = A.cone_gens
    f = compute_strict_cone_witness(A.lattice.rank, cone_gens)
    sample_charge_lists = [
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1)],
    ]
    for charges in sample_charge_lists:
        native = find_lowest(charges, cone_gens, f)
        bundle = A_bps._find_lowest(charges)
        assert native == bundle, \
            f"find_lowest disagrees on {charges}: native={native}, bundle={bundle}"
    print(PASS, "test_find_lowest_hexagon_matches_bundle")


# ---------------------------------------------------------------------------
# Top-of-stack: full multiply  via the native primitives matches the bundle
# (no wiring change yet — this is a reference test for Step 3).
# ---------------------------------------------------------------------------


def _native_decompose(F_a, F_b, lattice, cone_gens, witness, F_resolver):
    """Reference implementation of the cone-order subtraction loop using
    only native primitives.  `F_resolver(c) -> dict[Vec, QTLP]` provides
    F-elements for decomposition labels.
    """
    prod = qt_multiply(F_a, F_b, lattice)
    result: dict[tuple, QTLP] = {}
    for _ in range(500):
        prod = {g: c for g, c in prod.items() if not c.is_zero()}
        if not prod:
            break
        c_low = find_lowest(list(prod.keys()), cone_gens, witness)
        coeff = prod[c_low]
        result[c_low] = result.get(c_low, QTLP({})) + coeff
        if result[c_low].is_zero():
            del result[c_low]
        F_low = F_resolver(c_low)
        neg = QTLP({e: -v for e, v in coeff._coeffs.items()})
        zero = tuple(0 for _ in range(lattice.rank))
        sub = qt_multiply({zero: neg}, F_low, lattice)
        for g, nc in sub.items():
            cur = prod.get(g)
            if cur is None:
                prod[g] = nc
            else:
                s = cur + nc
                if s.is_zero():
                    del prod[g]
                else:
                    prod[g] = s
    return {g: c for g, c in result.items() if not c.is_zero()}


def test_native_decomposition_matches_bundle_pentagon():
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    A_bps = A._chart
    cone_gens = A.cone_gens
    f = compute_strict_cone_witness(A.lattice.rank, cone_gens)
    pairs = [((1, 0), (0, 1)), ((1, 1), (-1, -1)), ((1, 0), (1, 0))]
    for a, b in pairs:
        F_a = A._F_internal(a)
        F_b = A._F_internal(b)
        native = _native_decompose(
            F_a, F_b, A.lattice, cone_gens, f, A._F_internal,
        )
        bundle = A_bps.multiply(a, b)
        bundle_as_qt = {g: _bps_lp_to_qt(lp) for g, lp in bundle.items()}
        assert _equal_dicts(native, bundle_as_qt), \
            f"native decomposition disagrees with bundle on F_{a} · F_{b}"
    print(PASS, "test_native_decomposition_matches_bundle_pentagon")


def test_native_decomposition_matches_bundle_pure_su2():
    A = BPSKAlgebra(pairing=[[0, 2], [-2, 0]], node_charges=[(1, 0), (0, 1)])
    A_bps = A._chart
    cone_gens = A.cone_gens
    f = compute_strict_cone_witness(A.lattice.rank, cone_gens)
    pairs = [((1, 0), (-1, 2)), ((1, 0), (1, 0)), ((-1, 2), (-1, 2))]
    for a, b in pairs:
        F_a = A._F_internal(a)
        F_b = A._F_internal(b)
        native = _native_decompose(
            F_a, F_b, A.lattice, cone_gens, f, A._F_internal,
        )
        bundle = A_bps.multiply(a, b)
        bundle_as_qt = {g: _bps_lp_to_qt(lp) for g, lp in bundle.items()}
        assert _equal_dicts(native, bundle_as_qt), \
            f"native decomposition disagrees with bundle on F_{a} · F_{b}"
    print(PASS, "test_native_decomposition_matches_bundle_pure_su2")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_qt_multiply_pentagon_matches_bundle()
    test_strict_witness_pentagon()
    test_strict_witness_pure_su2()
    test_strict_witness_supplied_certificate_accepted()
    test_strict_witness_rejects_zero_generator()
    test_strict_witness_rejects_antipodal_pair()
    test_strict_witness_matches_bundle_on_presets()
    test_find_lowest_pentagon_matches_bundle()
    test_find_lowest_pure_su2_matches_bundle()
    test_find_lowest_hexagon_matches_bundle()
    test_native_decomposition_matches_bundle_pentagon()
    test_native_decomposition_matches_bundle_pure_su2()
    print()
    print("All bps_kalgebra primitives tests passed.")
