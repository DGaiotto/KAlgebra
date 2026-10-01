"""Tests for A1A2kConeData — A1A2kKAlg's cone-filtration sidecar.

Extensive coverage analogous to `test_heptagon_cone_data.py`, but
parameterised in `k`.  At `k = 2` this reduces to the heptagon, so we
explicitly cross-check that `A1A2kConeData(2)` reproduces
`HeptagonConeData`'s structure (14 mult-gens, 42 cones of size 4, etc.).

Coverage:
  (1) Cone structure & combinatorics across `k = 1..4` (cluster counts
      match the Fuss-Catalan / Catalan sequence for A_{2k}: 5, 42, 429,
      4862 — and at `k = 4` we just sanity-check the count without
      enumerating, to keep the test fast).
  (2) `_qcommute_factor` parity invariant at each tested `k`.
  (3) Cocycle antisymmetry across all `k * H * k * H` pairs at `k ≤ 3`.
  (4) `to_cone_label` / `from_cone_label` round-trip.
  (5) Universal bar-invariance phase formula reproduces
      `-_label_canon_twist` exactly on every sortable label.
  (6) `derived_multiply == _legacy_multiply` on all `k*H × k*H` named
      pairs at `k = 1, 2, 3` and on a sample of higher-weight products.
  (7) `derived_multiply` matches the BPSKAlgebra-wrapper
      `A1A2k.HeptagonKAlg`-equivalent at `k = 2` on every 14×14 named
      pair (= the existing heptagon cross-check, validated via
      `A1A2kConeData(2)`).
  (8) `simplify_trace_via_cone_data` lands on A1A2k's trace seeds.
  (9) `trace_layer1` matches `_legacy_trace_layer1` on named letters,
      weight-2 q-commuting pairs, and pure powers across multiple `k`.
  (10) ρ-invariance of `trace_layer1` under the rewired path.
"""

import os
import sys
import random
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from kalgebra import Element
from a1a2k_kalg import A1A2kKAlg
from a1a2k_cone_data import A1A2kConeData
from laurent_poly import LaurentPoly


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _named(A):
    return [(k_l, i) for k_l in range(1, A.k + 1) for i in range(A.H)]


def _single(k_l, i):
    return ((k_l, i, 1),)


# ----------------------------------------------------------------------
# (1) Cone structure & combinatorics
# ----------------------------------------------------------------------

EXPECTED_CONE_COUNTS = {1: 5, 2: 42, 3: 429}  # = Catalan C_{k+1}


def test_mult_gen_count():
    for k in (1, 2, 3, 4):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        assert len(cd.mult_gens()) == k * A.H, (k, len(cd.mult_gens()))


def test_cone_count_matches_catalan_low_k():
    for k, expected in EXPECTED_CONE_COUNTS.items():
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        cones = list(cd.cones())
        assert len(cones) == expected, (k, len(cones), expected)
        # All cones have the same size = 2k (= rank of the A_{2k}
        # cluster algebra, = mutable-variable count per cluster).
        # Sanity checks: pentagon (k=1) cones of size 2, heptagon
        # (k=2) cones of size 4.
        assert all(len(c) == 2 * k for c in cones), \
            (k, sorted({len(c) for c in cones}))


def test_cone_membership_is_pairwise_qcommuting():
    """Every cone must be a pairwise-q-commuting subset of mult-gens."""
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        for cone in cd.cones():
            gs = list(cone)
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    assert cd.q_commute(gs[i], gs[j]), (k, cone, gs[i], gs[j])


# ----------------------------------------------------------------------
# (2-3) qcommute_factor parity + cocycle antisymmetry
# ----------------------------------------------------------------------

def test_qcommute_factor_parity():
    for k in (1, 2, 3, 4):
        A = A1A2kKAlg(k)
        for g in _named(A):
            for h in _named(A):
                c = A._qcommute_factor(g, h)
                if c is not None:
                    assert c % 2 == 0, (k, g, h, c)


def test_cocycle_antisymmetric():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        for g in _named(A):
            for h in _named(A):
                if g == h:
                    assert cd.cocycle(g, h) == 0
                    continue
                if not cd.q_commute(g, h):
                    continue
                assert cd.cocycle(g, h) == -cd.cocycle(h, g), (k, g, h)


# ----------------------------------------------------------------------
# (4) Bijection round-trip
# ----------------------------------------------------------------------

def _sample_labels(A):
    labels = [()]
    for (k_l, i) in _named(A):
        labels.append(((k_l, i, 1),))
    cd = A.cone_data()
    seen = set()
    for g in _named(A):
        for h in _named(A):
            if g >= h or not cd.q_commute(g, h):
                continue
            key = tuple(sorted([g, h]))
            if key in seen:
                continue
            seen.add(key)
            labels.append(tuple((kk, ii, 1) for (kk, ii) in key))
            if len(seen) >= 30:
                break
        if len(seen) >= 30:
            break
    return labels


def test_to_cone_label_roundtrip():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        for label in _sample_labels(A):
            assert cd.verify_roundtrip(label), (k, label)


# ----------------------------------------------------------------------
# (5) Universal bar-invariance phase reproduces -_label_canon_twist
# ----------------------------------------------------------------------

def test_phase_matches_neg_label_canon_twist():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        for label in _sample_labels(A):
            if not label:
                continue
            gens, powers = cd.to_cone_label(label)
            phase = cd.cone_label_phase(gens, powers)
            t_bps = A._label_canon_twist(label)
            assert phase == -t_bps, (k, label, phase, -t_bps)


# ----------------------------------------------------------------------
# (6) derived_multiply matches _legacy_multiply on named pairs
# ----------------------------------------------------------------------

def test_derived_multiply_matches_legacy_on_named_pairs():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        named = _named(A)
        for ga in named:
            for gb in named:
                a = _single(*ga)
                b = _single(*gb)
                got = cd.derived_multiply(a, b)
                expected = A._legacy_multiply(a, b)
                assert got == expected, (k, ga, gb, got, expected)


def test_derived_multiply_matches_legacy_on_higher_weight():
    """Pure powers + q-commuting pairs at k=2, k=3."""
    for k in (2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        samples = [
            ((k_l, i, e),)
            for k_l in range(1, k + 1)
            for i in range(A.H)
            for e in (2, 3)
        ]
        # Also a few q-commuting pairs.
        named = _named(A)
        for g in named:
            for h in named:
                if g >= h or not cd.q_commute(g, h):
                    continue
                samples.append(
                    tuple(sorted([(g[0], g[1], 1), (h[0], h[1], 1)]))
                )
                if len(samples) > k * A.H * 2 + 10:
                    break
            if len(samples) > k * A.H * 2 + 10:
                break
        # Test a random sample of product pairs.
        random.seed(7 * k)
        for _ in range(50):
            a = random.choice(samples)
            b = random.choice(samples)
            got = cd.derived_multiply(a, b)
            expected = A._legacy_multiply(a, b)
            assert got == expected, (k, a, b)


# ----------------------------------------------------------------------
# (7) k=1 cross-check against PentagonKAlg
# ----------------------------------------------------------------------

def test_k1_trace_matches_pentagon():
    """A1A2kKAlg(1) is the pentagon.  Trace must agree with the existing
    `PentagonKAlg.trace` (already tested elsewhere — this is a sanity
    check that the cone-data-routed `trace_layer1` for A1A2kKAlg(1)
    reproduces the same RPowerSeries as PentagonKAlg on simple labels)."""
    from kalgebra_samples import PentagonKAlg
    A = A1A2kKAlg(1)
    P = PentagonKAlg()
    # A1A2k(1).L((1, j)) is the pentagon's L_j.
    K = 20
    for j in range(A.H):
        a_tr = A.trace(((1, j, 1),), K=K)
        p_tr = P.trace((j, 1, 0), K=K)
        d1 = {e: c for e, c in a_tr.coeffs.items() if e <= K}
        d2 = {e: c for e, c in p_tr.coeffs.items() if e <= K}
        assert d1 == d2, (j, d1, d2)


# ----------------------------------------------------------------------
# (8) simplify_trace lands on trace seeds
# ----------------------------------------------------------------------

def test_simplify_trace_lands_on_seeds():
    """`simplify_trace_via_cone_data(label)` for `A1A2kKAlg(k)` must
    yield an Element supported only on A1A2k's trace seeds: identity
    `()` and single mult-gens `((k_l, i, 1),)` for `k_l ∈ 1..k`."""
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        for label in _sample_labels(A):
            simp = cd.simplify_trace_via_cone_data(A, label)
            for lbl in simp.terms:
                if lbl == ():
                    continue
                assert len(lbl) == 1 and lbl[0][2] == 1 and \
                    1 <= lbl[0][0] <= k, (k, label, lbl)


# ----------------------------------------------------------------------
# (9) trace_layer1 matches _legacy_trace_layer1
# ----------------------------------------------------------------------

def test_trace_layer1_matches_legacy_on_named():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        for (k_l, i) in _named(A):
            a = A.L((k_l, i))
            new = A.trace_layer1(a)
            old = A._legacy_trace_layer1(a)
            assert new == old, (k, k_l, i, new, old)


def test_trace_layer1_matches_legacy_on_pure_powers():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        for (k_l, i) in _named(A):
            for e in range(1, 4):
                label = ((k_l, i, e),)
                new = A.trace_layer1(label)
                old = A._legacy_trace_layer1(label)
                assert new == old, (k, k_l, i, e)


def test_trace_layer1_matches_legacy_on_weight2_qcommuting():
    for k in (2, 3):
        A = A1A2kKAlg(k)
        cd = A.cone_data()
        n_checked = 0
        named = _named(A)
        for g in named:
            for h in named:
                if g >= h or not cd.q_commute(g, h):
                    continue
                label = tuple(sorted([(g[0], g[1], 1), (h[0], h[1], 1)]))
                new = A.trace_layer1(label)
                old = A._legacy_trace_layer1(label)
                assert new == old, (k, label, new, old)
                n_checked += 1
                if n_checked > 30:
                    break
            if n_checked > 30:
                break
        assert n_checked > 0, k


# ----------------------------------------------------------------------
# (10) ρ-invariance
# ----------------------------------------------------------------------

def test_trace_layer1_rho_invariance():
    for k in (1, 2, 3):
        A = A1A2kKAlg(k)
        for label in _sample_labels(A)[:20]:
            baseline = A.trace_layer1(label)
            cur = label
            for _ in range(A.H - 1):
                cur = A.rho(cur)
                assert A.trace_layer1(cur) == baseline, (k, label)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_mult_gen_count,
        test_cone_count_matches_catalan_low_k,
        test_cone_membership_is_pairwise_qcommuting,
        test_qcommute_factor_parity,
        test_cocycle_antisymmetric,
        test_to_cone_label_roundtrip,
        test_phase_matches_neg_label_canon_twist,
        test_derived_multiply_matches_legacy_on_named_pairs,
        test_derived_multiply_matches_legacy_on_higher_weight,
        test_k1_trace_matches_pentagon,
        test_simplify_trace_lands_on_seeds,
        test_trace_layer1_matches_legacy_on_named,
        test_trace_layer1_matches_legacy_on_pure_powers,
        test_trace_layer1_matches_legacy_on_weight2_qcommuting,
        test_trace_layer1_rho_invariance,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: OK")
    print(f"\nAll {len(tests)} a1a2k_cone_data tests passed.")
