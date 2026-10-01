"""Tests for PentagonConeData — `PentagonKAlg`'s cone-filtration sidecar.

Three pillars:

  (1) Round-trip: `from_cone_label(*to_cone_label(x)) == x` on every
      canonical-basis label `(i, a, b)` with `i ∈ ℤ/5`, `a, b ∈ [0..3]`.

  (2) `derived_multiply == multiply` on the full 31² grid of canonical
      basis labels with `a, b ∈ [0..2]` (= 961 ordered pairs).  This
      replaces `scripts/test_pentagon_from_cone_data.py` as a proper
      regression test.

  (3) ρ²-cycle-out trace simplification preserves the trace.
"""

import os
import sys
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from kalgebra import Element
from kalgebra_samples import PentagonKAlg, _pent_canon_key
from laurent_poly import LaurentPoly
from pentagon_cone_data import PentagonConeData, PENTAGON_CONE_DATA


# ----------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------

def _all_labels(amax=3):
    labels = {(0, 0, 0)}
    for i in range(5):
        for a, b in product(range(amax + 1), range(amax + 1)):
            if a == 0 and b == 0:
                continue
            labels.add(_pent_canon_key(i, a, b))
    return sorted(labels)


# ----------------------------------------------------------------------
# (1) Bijection round-trip
# ----------------------------------------------------------------------

def test_to_cone_label_roundtrip():
    cd = PENTAGON_CONE_DATA
    for label in _all_labels(amax=3):
        gens, powers = cd.to_cone_label(label)
        # Powers strictly positive.
        assert all(p > 0 for p in powers.values()), \
            f"powers must be > 0, got {powers} for {label}"
        # Power-keys == gens.
        assert set(powers.keys()) == set(gens), \
            f"powers keys {set(powers.keys())} != gens {set(gens)} " \
            f"for {label}"
        # Gens pairwise q-commute.
        gs = list(gens)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                assert cd.q_commute(gs[i], gs[j]), \
                    f"gens {gens} from {label} are not pairwise q-commuting"
        # Round-trip.
        recovered = cd.from_cone_label(gens, powers)
        assert recovered == label, \
            f"roundtrip failed: {label} → {(gens, powers)} → {recovered}"


def test_verify_roundtrip_method():
    cd = PENTAGON_CONE_DATA
    for label in _all_labels(amax=3):
        assert cd.verify_roundtrip(label), f"verify_roundtrip({label}) False"


# ----------------------------------------------------------------------
# (2) derived_multiply == multiply on the 961-pair grid
# ----------------------------------------------------------------------

def test_derived_multiply_matches_pentagon_multiply():
    A = PentagonKAlg()
    cd = A.cone_data()
    assert isinstance(cd, PentagonConeData)
    assert cd is PENTAGON_CONE_DATA

    labels = _all_labels(amax=2)
    n_ok = 0
    n_fail = 0
    fails = []
    for a in labels:
        for b in labels:
            got = cd.derived_multiply(a, b)
            # `A.multiply` now also routes through derived_multiply via
            # `_multiply_via_cone_data`, so the comparison below is
            # tautological — but it locks in that the public `multiply`
            # method continues to return what we expect.  We additionally
            # cross-check against the legacy reducer for safety.
            expected = A.multiply(a, b)
            if got == expected:
                n_ok += 1
            else:
                n_fail += 1
                if len(fails) < 5:
                    fails.append((a, b, got, expected))
    assert n_fail == 0, (
        f"{n_fail} mismatches; first few: " + repr(fails)
    )
    assert n_ok == len(labels) ** 2


def test_derived_multiply_matches_legacy_reducer():
    """Independent cross-check against the original `_pent_reduce` /
    `_pent_step` implementation, recreated here from the source."""
    from kalgebra_samples import (
        _pent_normalize_mono, _pent_reduce
    )
    cd = PENTAGON_CONE_DATA

    def legacy_multiply(a, b):
        i1, a1, b1 = a
        i2, a2, b2 = b
        shift = LaurentPoly.q(a1 * b1 + a2 * b2)
        m = _pent_normalize_mono([
            (i1, a1), (i1 + 1, b1),
            (i2, a2), (i2 + 1, b2),
        ])
        return Element(_pent_reduce(shift, m))

    labels = _all_labels(amax=2)
    n_fail = 0
    fails = []
    for a in labels:
        for b in labels:
            got = cd.derived_multiply(a, b)
            expected = legacy_multiply(a, b)
            if got != expected:
                n_fail += 1
                if len(fails) < 5:
                    fails.append((a, b, got, expected))
    assert n_fail == 0, (
        f"{n_fail} mismatches vs legacy reducer; first: " + repr(fails)
    )


# ----------------------------------------------------------------------
# (3) ρ²-cycle-out trace preserves the trace
# ----------------------------------------------------------------------

def test_simplify_trace_preserves_trace():
    """`alg.trace(L_x) == alg.trace_element(simplify_trace_via_cone_data(x))`
    on a sample of pentagon labels."""
    A = PentagonKAlg()
    cd = A.cone_data()
    K = 24

    labels = _all_labels(amax=2)
    for label in labels:
        simplified = cd.simplify_trace_via_cone_data(A, label)
        # Widen K so `trace_element` accurately computes the series
        # despite negative q-powers in `simplified`'s coefficients
        # (which shift trace coefficients down by that amount).
        neg = 0
        for coef in simplified.terms.values():
            if coef._coeffs:
                emin = min(coef._coeffs.keys())
                if emin < neg:
                    neg = emin
        inner_K = K - neg
        direct = A.trace(label, K=inner_K)
        via_simplified = A.trace_element(simplified, K=inner_K)
        d1 = {e: c for e, c in direct.coeffs.items() if e <= K}
        d2 = {e: c for e, c in via_simplified.coeffs.items() if e <= K}
        # Also: simplified must be supported only on trace seeds — that
        # is, the identity native label and single-mult-gen labels.
        # This catches no-op behaviour (which we previously had).
        for lbl in simplified.terms:
            i, x, y = lbl
            assert (x, y) in {(0, 0), (1, 0)}, (
                f"simplify_trace produced non-seed label {lbl} for input "
                f"{label}; expected only identity or single-mult-gen seeds"
            )
        assert d1 == d2, (
            f"trace mismatch for {label}: "
            f"direct={direct} vs via_simplified={via_simplified}"
        )


def test_pentagon_trace_layer1_matches_schur_recursion():
    """`PentagonKAlg.trace` is now wired through
    `simplify_trace_via_cone_data` for Layer 1.  Cross-check against
    the legacy `_pent_tr_power_coeffs`-based Layer 1 on `Tr(L_1^a)`
    for `a = 0..8` at `K = 30`.  This locks in that the tagged-
    cyclicity recursion reproduces the Schur recursion result
    bit-for-bit on the pure-power family."""
    from kalgebra_samples import (
        _pent_tr_power_coeffs, _pent_tr_1_rps, _pent_tr_L_rps,
    )
    from zplus_ring import RLaurent, RPowerSeries

    A = PentagonKAlg()

    def trace_via_schur(a, K):
        """The legacy Layer 1 path, reconstructed inline so this test
        remains valid even after `PentagonKAlg.trace` is fully migrated."""
        _, x, y = a
        c1, cL = _pent_tr_power_coeffs(x + y)
        q_xy = LaurentPoly.q(x * y)
        c1 = c1 * q_xy
        cL = cL * q_xy
        emin = 0
        for lp in (c1, cL):
            for e in lp._coeffs.keys():
                if e < emin:
                    emin = e
        inner_K = K - emin
        R = A.coefficient_ring()
        Tr1 = _pent_tr_1_rps(R, inner_K)
        TrL = _pent_tr_L_rps(R, inner_K)
        c1_rl = RLaurent(R, dict(c1._coeffs))
        cL_rl = RLaurent(R, dict(cL._coeffs))
        result = Tr1 * c1_rl + TrL * cL_rl
        return RPowerSeries(
            R, {e: c for e, c in result.coeffs.items() if e <= K}, K,
        )

    K = 30
    for a in range(0, 9):
        label = (1, a, 0)
        new_tr = A.trace(label, K=K)
        old_tr = trace_via_schur(label, K=K)
        d_new = {e: c for e, c in new_tr.coeffs.items() if e <= K}
        d_old = {e: c for e, c in old_tr.coeffs.items() if e <= K}
        assert d_new == d_old, (
            f"Tr(L_1^{a}) mismatch at K={K}: "
            f"tagged-cyclicity={d_new} vs Schur={d_old}"
        )


if __name__ == "__main__":
    test_to_cone_label_roundtrip()
    print("test_to_cone_label_roundtrip: OK")
    test_verify_roundtrip_method()
    print("test_verify_roundtrip_method: OK")
    test_derived_multiply_matches_pentagon_multiply()
    print("test_derived_multiply_matches_pentagon_multiply: OK")
    test_derived_multiply_matches_legacy_reducer()
    print("test_derived_multiply_matches_legacy_reducer: OK")
    test_simplify_trace_preserves_trace()
    print("test_simplify_trace_preserves_trace: OK")
    test_pentagon_trace_layer1_matches_schur_recursion()
    print("test_pentagon_trace_layer1_matches_schur_recursion: OK")
    print("\nAll pentagon_cone_data tests passed.")
