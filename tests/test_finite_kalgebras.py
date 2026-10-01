"""Tests for the ``finite_kalgebras`` package: registry, naming
aliases, ray labels, and chord classifiers."""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import finite_kalgebras as fk
from kalgebra import Element
from zplus_ring import RLaurent
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing


def test_registry_complete():
    """All 17 entries import cleanly + match the FINITE_KALGEBRAS dict."""
    expected = {
        "pentagon", "hexagon", "heptagon", "octagon", "decagon",
        "a3", "a5", "a7",
        "a1d3", "a1d4", "a1d5", "a1d6", "a1d7", "a1d8",
        "e6", "e7", "e8",
    }
    assert set(fk.FINITE_KALGEBRAS) == expected
    for short_id, cls in fk.FINITE_KALGEBRAS.items():
        A = cls()
        assert A.coefficient_ring() is not None


def test_naming_aliases():
    assert fk.FiniteA1A2KAlgebra is fk.FinitePentagonKAlgebra
    assert fk.FiniteA1A3KAlgebra is fk.FiniteHexagonKAlgebra
    assert fk.FiniteA1A4KAlgebra is fk.FiniteHeptagonKAlgebra
    assert fk.FiniteA1A5KAlgebra is fk.FiniteOctagonKAlgebra
    assert fk.FiniteA1A7KAlgebra is fk.FiniteDecagonKAlgebra
    assert fk.FinitePuncturedTriangleKAlgebra is fk.FiniteA1D3KAlgebra
    assert fk.FinitePuncturedOctagonKAlgebra is fk.FiniteA1D8KAlgebra
    assert fk.FiniteA1E7KAlgebra is fk.FiniteE7KAlgebra


def test_ray_labels_round_trip():
    """Every mg gets a label; pretty_ray_name produces a valid string."""
    for short_id, cls in fk.FINITE_KALGEBRAS.items():
        labels = fk.ray_labels(cls)
        A = cls()
        n_mg = len(A.cone_data().mult_gens())
        assert set(labels) == set(range(n_mg)), short_id
        for i in range(n_mg):
            name = fk.pretty_ray_name(cls, i)
            assert name.startswith("L_")


def test_chord_classifiers_polygons():
    """Polygon-cluster-algebra classifiers (A_1[A_odd]) produce
    expected counts.
    """
    cases = [
        ("hexagon",  {"skip3_diameter": 3, "doubled_skip2_skip2": 3}),
        ("octagon",  {"skip3_single": 8, "doubled_skip2_skip2": 8,
                      "doubled_skip2_skip4": 8}),
    ]
    for k, expected_counts in cases:
        cls = fk.FINITE_KALGEBRAS[k]
        labels = fk.identify_skip_classes(cls)
        from collections import Counter
        seen = Counter()
        for mg, lab in labels.items():
            parts = lab.rsplit("_", 1)
            base = parts[0] if parts[-1].isdigit() else lab
            seen[base] += 1
        for typ, n in expected_counts.items():
            assert seen[typ] == n, (k, typ, dict(seen))


def test_associativity_random(trials=50):
    """Spot-check: every algebra is associative on random triples."""
    for short_id, cls in fk.FINITE_KALGEBRAS.items():
        A = cls()
        cd = A.cone_data()
        n = len(cd.mult_gens())
        R = A.coefficient_ring()
        if isinstance(R, TrivialZPlusRing):
            one_RL = LaurentPoly({0: 1})
        else:
            one_RL = RLaurent(R, {0: R.one()})

        def mul(e1, e2):
            out = Element({})
            for la, ca in e1.terms.items():
                for lb, cb in e2.terms.items():
                    prod = A.multiply(la, lb)
                    for lo, co in prod.terms.items():
                        prev = out.terms.get(lo)
                        cur = ca * cb * co
                        out.terms[lo] = cur if prev is None else prev + cur
            return out

        random.seed(0)
        fails = 0
        for _ in range(trials):
            i, j, k = (random.randrange(n) for _ in range(3))
            a = Element({((i, 1),): one_RL})
            b = Element({((j, 1),): one_RL})
            c = Element({((k, 1),): one_RL})
            L = {kk: str(v) for kk, v in mul(mul(a, b), c).terms.items()
                 if not v.is_zero()}
            R_ = {kk: str(v) for kk, v in mul(a, mul(b, c)).terms.items()
                  if not v.is_zero()}
            if L != R_:
                fails += 1
        assert fails == 0, f"{short_id} assoc failures: {fails}/{trials}"


def test_rho_is_automorphism_all_entries():
    """Every native zoo entry's ρ is an honest algebra automorphism on its
    generators.  Regression guard for the A12 class: the a1d4 truncation
    and su2_nf1 *crashed* `verify_rho_is_automorphism` (and the flavoured
    siblings returned False before) — uncaught because no test ran
    this verifier on the native entries.  ρ uses only multiply + rho_element
    (no trace), so it applies even to entries whose su2u1 trace is pending.
    """
    for short_id, cls in fk.FINITE_KALGEBRAS.items():
        A = cls()
        n = len(A.cone_data().mult_gens())
        gens = [((i, 1),) for i in range(min(n, 6))]
        fails = []
        for a in gens:
            for b in gens:
                # A crash here is itself the regression (a1d4/su2_nf1 class).
                if not A.verify_rho_is_automorphism(a, b):
                    fails.append((a, b))
        assert not fails, f"{short_id} ρ-automorphism fails: {fails[:3]}"


def test_chord_intersection_graph():
    """Graph is symmetric, q_commute negation, no self-loops."""
    for k in ("pentagon", "hexagon", "heptagon", "octagon", "decagon"):
        cls = fk.FINITE_KALGEBRAS[k]
        g = fk.chord_intersection_graph(cls)
        A = cls()
        cd = A.cone_data()
        for i, nbrs in g.items():
            assert i not in nbrs, (k, i, "self-loop")
            for j in nbrs:
                # Symmetry
                assert i in g[j], (k, i, j)
                # q_commute is the inverse
                assert not cd.q_commute(i, j), (k, i, j)


def test_predict_cross_product_term_count():
    """Function returns nonneg int matching len(cross_product(i, j))."""
    cls = fk.FINITE_KALGEBRAS["octagon"]
    A = cls()
    cd = A.cone_data()
    n = len(cd.mult_gens())
    for i in range(min(n, 10)):
        for j in range(min(n, 10)):
            if i == j or cd.q_commute(i, j):
                continue
            k = fk.predict_cross_product_term_count(cls, i, j)
            assert k == len(cd.cross_product(i, j))


def test_a1dn_chord_classes_n_odd():
    """A_1[D_n] for n odd gets geometric chord-type labels."""
    from collections import Counter
    expected = {
        "a1d3": {"a1dn_skip2_even": 3, "a1dn_skip2_odd": 3},
        "a1d5": {"a1dn_skip2_even": 5, "a1dn_skip2_odd": 5,
                 "a1dn_skip4_odd": 10},
        "a1d7": {"a1dn_skip2_even": 7, "a1dn_skip2_odd": 7,
                 "a1dn_skip4_odd": 14, "a1dn_skip6_odd": 14},
    }
    for k, want in expected.items():
        cls = fk.FINITE_KALGEBRAS[k]
        labels = fk.identify_a1dn_chord_classes(cls)
        seen = Counter()
        for mg, lab in labels.items():
            parts = lab.rsplit("_", 1)
            base = parts[0] if parts[-1].isdigit() else lab
            seen[base] += 1
        for typ, n in want.items():
            assert seen[typ] == n, (k, typ, dict(seen))


def test_chord_labels_dispatcher():
    """Top-level chord_labels(cls) routes to correct classifier."""
    for k in ("hexagon", "octagon", "decagon"):
        labels = fk.chord_labels(fk.FINITE_KALGEBRAS[k])
        # Polygon classifier produces 'skip' or 'doubled' labels.
        assert any("skip" in l or "doubled" in l for l in labels.values()), k
    for k in ("a1d3", "a1d5", "a1d7"):
        labels = fk.chord_labels(fk.FINITE_KALGEBRAS[k])
        assert any("a1dn_" in l for l in labels.values()), k
    for k in ("pentagon", "heptagon", "e7"):
        labels = fk.chord_labels(fk.FINITE_KALGEBRAS[k])
        assert all(l.startswith("L_") for l in labels.values()), k


def test_a1d_parity_labels_dont_crash():
    """Parity classifier returns labels for A_1[D_n] without error."""
    for k in ("a1d3", "a1d4", "a1d5", "a1d6", "a1d7", "a1d8"):
        cls = fk.FINITE_KALGEBRAS[k]
        labels = fk.chord_parity_label(cls)
        A = cls()
        n_mg = len(A.cone_data().mult_gens())
        assert set(labels) == set(range(n_mg))
        assert all(v in ("mixed", "doubled") for v in labels.values())


if __name__ == "__main__":
    test_registry_complete()
    print("test_registry_complete PASSED")
    test_naming_aliases()
    print("test_naming_aliases PASSED")
    test_ray_labels_round_trip()
    print("test_ray_labels_round_trip PASSED")
    test_chord_classifiers_polygons()
    print("test_chord_classifiers_polygons PASSED")
    test_chord_intersection_graph()
    print("test_chord_intersection_graph PASSED")
    test_predict_cross_product_term_count()
    print("test_predict_cross_product_term_count PASSED")
    test_a1dn_chord_classes_n_odd()
    print("test_a1dn_chord_classes_n_odd PASSED")
    test_chord_labels_dispatcher()
    print("test_chord_labels_dispatcher PASSED")
    test_a1d_parity_labels_dont_crash()
    print("test_a1d_parity_labels_dont_crash PASSED")
    test_associativity_random()
    print("test_associativity_random PASSED")
    test_rho_is_automorphism_all_entries()
    print("test_rho_is_automorphism_all_entries PASSED")
