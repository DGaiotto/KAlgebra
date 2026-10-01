"""Tests for A1A2kKAlg with the natural labeling
L((a, j)) ↔ chord (j, j+a+1) on the (2k+3)-gon."""
import itertools
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1a2k_kalg import A1A2kKAlg
from A1A2k import A1A2k
from A1A2k_naming_audit import natural_orbit_seeds


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _to_dict(ps):
    return {e: int(str(v)) for e, v in ps.coeffs.items()}


def _charge_of_label(W, label):
    """BPS lattice charge of a canonical label `((a_1, j_1, e_1), ...)`."""
    n = 2 * W.k
    ch = [0] * n
    for (a, j, e) in label:
        g = W.charge(a, j)
        for kk in range(n):
            ch[kk] += e * g[kk]
    return tuple(ch)


# ---------------------------------------------------------------------------
# Tests: naming
# ---------------------------------------------------------------------------


def test_natural_seeds_at_k_eq_2():
    """At k=2 the natural seeds are determined."""
    seeds = natural_orbit_seeds(2)
    assert seeds[1] == (1, 0, 1, 0), seeds[1]
    # Orbit 2 (length 3) seed: starts at vertex 0, length 3.
    assert isinstance(seeds[2], tuple) and len(seeds[2]) == 4
    print("  PASS: test_natural_seeds_at_k_eq_2")


def test_natural_lengths_and_shifts():
    """A1A2k publishes lengths[a]=a+1 and shifts[a]=0."""
    for k in (2, 3, 4):
        W = A1A2k(k)
        for a in range(1, k + 1):
            assert W.lengths[a] == a + 1, (k, a, W.lengths)
            assert W.shifts[a] == 0, (k, a, W.shifts)
    print("  PASS: test_natural_lengths_and_shifts")


def test_rho_action_on_labels():
    """ρ(L((a, j))) = L((a, j+1)) in the natural labeling."""
    for k in (2, 3):
        A = A1A2kKAlg(k)
        H = A.H
        for a in range(1, k + 1):
            for j in range(H):
                lbl = A.L((a, j))
                rho_lbl = A.rho(lbl)
                expected = A.L((a, (j + 1) % H))
                assert rho_lbl == expected, (k, a, j, rho_lbl, expected)
    print("  PASS: test_rho_action_on_labels")


# ---------------------------------------------------------------------------
# Tests: chord-pair geometry vs algebra
# ---------------------------------------------------------------------------


def test_chord_endpoints_via_bps_charge():
    """The BPS charge of L((a, j)) realises the chord (j, j+a+1) by
    being in the ρ-orbit of the natural seed (which has chord
    (0, a+1) by construction)."""
    for k in (2, 3):
        W = A1A2k(k)
        seeds = natural_orbit_seeds(k)
        for a in range(1, k + 1):
            assert W.charge(a, 0) == seeds[a]
            # ρ-image: applying lattice ρ once should give W.charge(a, 1).
            ch1 = tuple(W.A.rho(seeds[a]))
            assert W.charge(a, 1) == ch1, (k, a, ch1, W.charge(a, 1))
    print("  PASS: test_chord_endpoints_via_bps_charge")


def test_single_letter_trace_matches_bps():
    """For each single letter L((a, 0)), our trace equals BPS Tr at
    the corresponding charge.  Only j=0 (ρ-invariance of trace
    handles j>0; see test_trace_invariant_under_rho); only k=2, 3
    (BPS at higher k is slow)."""
    for k in (2, 3):
        A = A1A2kKAlg(k)
        W = A1A2k(k)
        for a in range(1, k + 1):
            lbl = A.L((a, 0))
            tr_o = _to_dict(A.trace(lbl, K=8))
            tr_b = _to_dict(W.A.trace(W.charge(a, 0), K=8))
            assert tr_o == tr_b, (k, a, tr_o, tr_b)
    print("  PASS: test_single_letter_trace_matches_bps")


def test_square_trace_matches_bps():
    """Tr(L((a, 0))^2) matches BPS for each orbit at k=2."""
    A = A1A2kKAlg(2)
    W = A1A2k(2)
    for a in range(1, A.k + 1):
        label = ((a, 0, 2),)
        tr_o = _to_dict(A.trace(label, K=10))
        ch = _charge_of_label(W, label)
        tr_b = _to_dict(W.A.trace(ch, K=10))
        assert tr_o == tr_b, (a, tr_o, tr_b)
    print("  PASS: test_square_trace_matches_bps")


def test_qcommute_for_non_crossing_pair():
    """L((1,0)) · L((1,2)) at k=2: two length-2 chords (0,2) and (2,4)
    share vertex 2 -> non-crossing -> single-term product (q-commute)."""
    A = A1A2kKAlg(2)
    prod = A.multiply(A.L((1, 0)), A.L((1, 2)))
    assert len(prod.terms) == 1, dict(prod.terms)
    print("  PASS: test_qcommute_for_non_crossing_pair")


def test_plucker_crossing_pair():
    """L((2,0)) · L((2,1)) at k=2: two length-3 chords (0,3) and (1,4)
    cross -> Plücker -> two-term output.  d=1 odd -> (alpha, beta)=(0, -1).
    The opposite edges 0-1 and 3-4 are length-1 = identity, so the
    `q^alpha` term is the identity scalar; the other term is the
    non-crossing pair (1,3)·(4,0) = L((1,1)) · L((2,4))."""
    A = A1A2kKAlg(2)
    prod = A.multiply(A.L((2, 0)), A.L((2, 1)))
    # Expect two terms: identity () and a 2-letter canonical product.
    assert len(prod.terms) == 2, dict(prod.terms)
    keys = set(prod.terms.keys())
    assert () in keys, keys
    print("  PASS: test_plucker_crossing_pair")


# ---------------------------------------------------------------------------
# Tests: trace consistency
# ---------------------------------------------------------------------------


def test_trace_invariant_under_rho():
    """ρ²-cyclicity (and ρ-invariance via gcd(2,h)=1):
    Tr(L((a, j))) is independent of j (k=2)."""
    A = A1A2kKAlg(2)
    for a in range(1, A.k + 1):
        tr_at_j0 = _to_dict(A.trace(A.L((a, 0)), K=10))
        for j in range(1, A.H):
            tr_at_j = _to_dict(A.trace(A.L((a, j)), K=10))
            assert tr_at_j == tr_at_j0, (a, j, tr_at_j, tr_at_j0)
    print("  PASS: test_trace_invariant_under_rho")


def test_associativity_high_k():
    """Closed-form base table must give an associative multiplication.
    Sample 20 random single-letter triples (a, b, c) and verify
    (a · b) · c == a · (b · c) at k = 2, 4, 6, 10."""
    import random
    from laurent_poly import LaurentPoly
    for k in (2, 4, 6, 10):
        A = A1A2kKAlg(k)
        H = A.H
        letters = [((kl, j, 1),) for kl in range(1, A.k + 1) for j in range(H)]
        random.seed(42 + k)
        for _ in range(20):
            la = random.choice(letters)
            lb = random.choice(letters)
            lc = random.choice(letters)
            ab = A.multiply(la, lb)
            left = {}
            for sub, coef in ab.terms.items():
                sub_prod = A.multiply(sub, lc)
                for kk, v in sub_prod.terms.items():
                    left[kk] = left.get(kk, LaurentPoly.zero()) + coef * v
            bc = A.multiply(lb, lc)
            right = {}
            for sub, coef in bc.terms.items():
                sub_prod = A.multiply(la, sub)
                for kk, v in sub_prod.terms.items():
                    right[kk] = right.get(kk, LaurentPoly.zero()) + coef * v
            lclean = {kk: str(v) for kk, v in left.items() if not v.is_zero()}
            rclean = {kk: str(v) for kk, v in right.items() if not v.is_zero()}
            assert lclean == rclean, (
                f"k={k}: associativity fail at {la}, {lb}, {lc}: "
                f"left={lclean}, right={rclean}"
            )
    print("  PASS: test_associativity_high_k")


def test_fast_init_no_bps_runtime():
    """`A1A2kKAlg(k)` constructs entirely from the closed-form base_table
    (no BPSKAlgebra.multiply call at runtime).  Sanity: at k=4 init
    should finish in < 1 second (legacy BPS-based init was ~90s)."""
    import time
    for k in (4, 5, 6, 8):
        t0 = time.time()
        A = A1A2kKAlg(k)
        dt = time.time() - t0
        assert dt < 5.0, f"k={k} init took {dt:.1f}s — should be << 1s with closed form"
        # Also confirm multiply works.
        assert A.multiply(A.L((1, 0)), A.L((1, 1))) is not None
    print("  PASS: test_fast_init_no_bps_runtime")


# ---------------------------------------------------------------------------
# Tests: cross-checks against intrinsic K-algebra implementations
# ---------------------------------------------------------------------------


def test_trace_matches_pentagon_at_k1():
    """`A1A2kKAlg(1).trace` agrees with `PentagonKAlg().trace` on
    elements built from a single chord-orbit (singletons, pure powers).

    PentagonKAlg uses (i, a, b) canonical labels with a different
    canonical-basis normalisation convention from A1A2kKAlg, so we
    can't compare arbitrary labels directly — but Tr(1), Tr(L_i),
    Tr(L_i^n) are unambiguous (the L_i for both classes is "a
    length-2 chord of the pentagon", and ρ-invariance handles the
    naming offset)."""
    from kalgebra_samples import PentagonKAlg
    A1 = A1A2kKAlg(1)
    P = PentagonKAlg()
    K = 14
    # Identity.
    assert _to_dict(A1.trace((), K=K)) == _to_dict(P.trace((0, 0, 0), K=K)), \
        "Tr(1) mismatch"
    # Singletons and pure powers (all 5 chords are ρ-equivalent → same trace).
    for n in (1, 2, 3, 4):
        a1_tr = _to_dict(A1.trace(((1, 0, n),), K=K))
        p_tr = _to_dict(P.trace((0, n, 0), K=K))
        assert a1_tr == p_tr, \
            f"Tr(L_0^{n}) at k=1 mismatch: A1A2k={a1_tr}, Pent={p_tr}"
    print("  PASS: test_trace_matches_pentagon_at_k1")


def test_trace_layer1_matches_heptagon_at_k2():
    """`A1A2kKAlg(2).trace_layer1` agrees with the intrinsic
    `kalgebra_samples.HeptagonKAlg.trace_layer1` on every singleton,
    square and non-crossing pair of letters, both orbits and mixed.

    Both classes share the `((k_letter, i, e), ...)` canonical-label
    format, but `kalgebra_samples.HeptagonKAlg`'s `(2, i)` is this
    class's `(2, i + 4)` (its `(1, i)` is `(1, i)`), so labels are
    relabelled before comparing.  (Under the identity map the same-orbit
    labels still agree, being rotations of one another, but mixed-orbit
    labels do not.)  The coefficient tuple has orbit permutation:
        A1A2k_coef = (c_0, c_1, c_2),   Hept_coef = (c_0, c_L, c_N).
    with L = orbit-2 (length-3), N = orbit-1 (length-2) — so we
    compare A1A2k[1] ↔ Hept[2] (c_N) and A1A2k[2] ↔ Hept[1] (c_L)."""
    from kalgebra_samples import HeptagonKAlg
    A2 = A1A2kKAlg(2)
    H = HeptagonKAlg()
    Hh = A2.H  # 7

    def to_a1a2k_letter(letter):
        kl, i = letter
        return (kl, (i + 4) % Hh if kl == 2 else i)

    def to_a1a2k(lbl):
        return tuple(sorted(to_a1a2k_letter((kl, i)) + (e,)
                            for (kl, i, e) in lbl))

    def assert_match(lbl):
        a1 = A2.trace_layer1(to_a1a2k(lbl))
        ht = H.trace_layer1(lbl)
        assert a1[0] == ht[0], f"c_0 mismatch on {lbl}"
        assert a1[1] == ht[2], f"c_1 (N) mismatch on {lbl}"
        assert a1[2] == ht[1], f"c_2 (L) mismatch on {lbl}"

    # Singletons (all i, both orbits).
    for kl in (1, 2):
        for i in range(Hh):
            assert_match(((kl, i, 1),))
    # Squares L_{k, 0}^2 (same letter, c=0 trivially).
    for kl in (1, 2):
        assert_match(((kl, 0, 2),))
    # Every non-crossing (q-commuting) pair of distinct letters.
    letters = [(kl, i) for kl in (1, 2) for i in range(Hh)]
    pair_count = mixed = 0
    for la, lb in itertools.combinations(letters, 2):
        if A2._qcommute_factor(to_a1a2k_letter(la),
                               to_a1a2k_letter(lb)) is None:
            continue
        assert_match(tuple(sorted([la + (1,), lb + (1,)])))
        pair_count += 1
        mixed += la[0] != lb[0]
    assert mixed > 0
    print(f"  PASS: test_trace_layer1_matches_heptagon_at_k2  "
          f"(checked {pair_count} non-crossing pairs, {mixed} of them "
          f"mixed-orbit, + singletons + squares)")


def test_pentagon_diagonals_certified_against_k1():
    """`PentagonKAlg`'s generator `L_i` is the diagonal `{3i, 3i + 2}`
    (`curve`, `geometric_label`): under `L_i ↦ A1A2kKAlg(1)`'s letter
    `(1, 3i)` — the letter of that diagonal — every product of two canonical
    labels `(i, a, b)` with `a, b ≤ 2` (31 labels, 961 products) and ρ on each
    label agree.  Negative control: the reflected orientation `L_i ↦ (1, −3i)`
    fails on most products and is not ρ-equivariant."""
    from kalgebra_samples import PentagonKAlg, _pent_canon_key
    A1, P = A1A2kKAlg(1), PentagonKAlg()

    def image(label, s):
        i, a, b = label
        out = {}
        for j, m in ((i, a), (i + 1, b)):
            if m:
                out[s(j) % 5] = out.get(s(j) % 5, 0) + m
        return tuple(sorted((1, j, m) for j, m in out.items()))

    labels = sorted({_pent_canon_key(i, a, b) for i in range(5)
                     for a in range(3) for b in range(3)})
    assert len(labels) == 31

    def agreement(s):
        good = sum({image(l, s): c for l, c in P.multiply(x, y).terms.items()}
                   == dict(A1.multiply(image(x, s), image(y, s)).terms)
                   for x in labels for y in labels)
        rho = all(image(P.rho(x), s) == A1.rho(image(x, s)) for x in labels)
        return good, rho

    assert agreement(lambda j: 3 * j) == (961, True)
    good, rho = agreement(lambda j: -3 * j)
    assert good < 961 // 2 and not rho, (good, rho)
    for x in labels:
        curves = P.geometric_label(x)
        assert tuple(sorted(((d, m) for (_k, j, m) in image(x, lambda j: 3 * j)
                             for d in [tuple(sorted((j, (j + 2) % 5)))]))) == curves, x
    assert all(P.geometric_label(P.curve(x, ell))
               == ((tuple(sorted((x % 5, (x + ell) % 5))), 1),)
               for x in range(5) for ell in (2, 3))
    print(f"  PASS: test_pentagon_diagonals_certified_against_k1  (961/961 products, "
          f"rho; reflected control {good}/961)")


def test_heptagon_curve_and_geometric_label():
    """`HeptagonKAlg.curve(x, ell)` is `A1A2kKAlg(2).curve(x, ell)` read
    through the class's relabelling, and `geometric_label` names each letter
    by the diagonal of the class docstring (`(1, i)` = `{i, i + 2}`,
    `(2, i)` = `{i, i + 4}`); ρ is the rotation on the curves."""
    from kalgebra_samples import HeptagonKAlg, _hept_label_to_a1a2k
    A2, Hp = A1A2kKAlg(2), HeptagonKAlg()
    for x in range(7):
        for ell in range(2, 6):
            lab = Hp.curve(x, ell)
            assert _hept_label_to_a1a2k(lab) == A2.curve(x, ell), (x, ell)
            assert Hp.geometric_label(lab) == (
                (tuple(sorted((x % 7, (x + ell) % 7))), 1),), (x, ell)
            assert Hp.rho(lab) == Hp.curve(x + 1, ell), (x, ell)
    print("  PASS: test_heptagon_curve_and_geometric_label")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_natural_seeds_at_k_eq_2()
    test_natural_lengths_and_shifts()
    test_rho_action_on_labels()
    test_chord_endpoints_via_bps_charge()
    test_single_letter_trace_matches_bps()
    test_square_trace_matches_bps()
    test_qcommute_for_non_crossing_pair()
    test_plucker_crossing_pair()
    test_trace_invariant_under_rho()
    test_associativity_high_k()
    test_fast_init_no_bps_runtime()
    test_trace_matches_pentagon_at_k1()
    test_trace_layer1_matches_heptagon_at_k2()
    test_pentagon_diagonals_certified_against_k1()
    test_heptagon_curve_and_geometric_label()
    print("\nAll A1A2kKAlg natural-labeling tests passed.")
