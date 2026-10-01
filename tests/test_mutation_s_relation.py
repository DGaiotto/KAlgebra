"""Tests for `mutation_s_relation`: the relation
`S_{μ_k Q} = E_𝖖(X_{γ_k})⁻¹ · S_Q · E_𝖖(X_{−γ_k})` compared exactly at finite
depth, with positive cases (the pentagon, `A_3`, the 3-cycle, a wild Kronecker
quiver, the Markov quiver — no negating sequence, and mapped to itself by every
mutation), NEGATIVE controls (the wrong identification for the side; a
perturbed generator), the frame conventions, and paths through heavier quivers.

Run:  `python3 run_tests.py`
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from habiro import HabiroElement  # noqa: E402
from mutation_s_relation import (  # noqa: E402
    VARIANTS, compare_edge, compare_ends, compare_steps, coordinate_change, generator,
    path_matrices, transport,
)
from quiver_enumeration import canonical_form, mutate, total_weight  # noqa: E402

A2 = ((0, 1), (-1, 0))
A3 = ((0, 1, 0), (-1, 0, 1), (0, -1, 0))
CYC3 = ((0, 1, -1), (-1, 0, 1), (1, -1, 0))
KR3 = ((0, 3), (-3, 0))
MARKOV = ((0, 2, -2), (-2, 0, 2), (2, -2, 0))


def test_the_relation_holds_on_every_edge_of_the_small_cases():
    for B in (A2, A3, CYC3, KR3, MARKOV):
        for k in range(len(B)):
            for variant in ("green", "reverse"):
                c = compare_edge(B, k, 5, variant=variant)
                assert c.ok, (B, k, variant, c.as_dict())
                assert c.n_content > 0, (B, k, variant)


def test_the_markov_quiver_satisfies_it_as_a_functional_equation():
    """Every mutation maps the Markov quiver to itself (up to relabelling), so
    the relation is an equation on its one generator — and it has no negating
    sequence, so this is the conjecture proper."""
    for k in range(3):
        assert canonical_form(mutate(MARKOV, k)).key == canonical_form(MARKOV).key
        c = compare_edge(MARKOV, k, 6)
        assert c.ok and c.n_content >= 40, c.as_dict()


def test_the_wrong_identification_for_the_side_fails():
    """The negative controls: stripping on one side needs the matching lattice
    identification; the other one must fail wherever the relation has
    content."""
    for B in (A2, A3, CYC3, KR3, MARKOV):
        for k in range(len(B)):
            for variant in ("mixed_green", "mixed_reverse"):
                c = compare_edge(B, k, 5, variant=variant)
                assert not c.ok, (B, k, variant)


def test_a_perturbed_generator_is_caught():
    """Changing one coefficient of `S_Q` beyond the leading data must break
    the comparison: the test sees content, not just the axioms' first order."""
    B = CYC3
    S = dict(generator(B, 5))
    g = (1, 1, 1)
    S[g] = S.get(g, HabiroElement.zero()) + HabiroElement.q_power(7)
    c = compare_ends(B, [0], 5, 5, S_start=S)
    assert not c.ok and any(sum(w) >= 3 for w, _a, _b in c.mismatches), c.as_dict()


def test_another_quivers_generator_does_not_pass():
    c = compare_ends(CYC3, [0], 5, 5, S_end=generator(mutate(A3, 1), 5))
    assert not c.ok


def test_the_coordinate_change_is_an_involution_and_gives_the_fz_matrix():
    rng = random.Random(3)
    for _ in range(40):
        n = rng.randint(2, 5)
        B = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x = rng.randint(-2, 2)
                B[i][j], B[j][i] = x, -x
        k = rng.randrange(n)
        Bk = mutate(B, k)
        for kind in ("green", "reverse"):
            phi = coordinate_change(B, k, kind)
            for _ in range(5):
                v = tuple(rng.randint(0, 3) for _ in range(n))
                assert phi(phi(v)) == v
            # the new basis in old coordinates, and its pairing under B
            basis = [phi(tuple(int(i == j) for i in range(n))) for j in range(n)]
            for i in range(n):
                for j in range(n):
                    p = sum(basis[i][a] * B[a][b] * basis[j][b] for a in range(n) for b in range(n))
                    assert p == Bk[i][j], (B, k, kind)


def test_a_path_through_a_heavier_quiver_agrees_end_to_end_and_step_by_step():
    """The author's extension: the ends of a path whose middle quiver is heavier
    are compared without the middle's generator, and agree with the steps."""
    # a witness of the W = 7 orbit build: weights 5 -> 9 -> 7, the middle above the cutoff
    B = ((0, -2, 0, -1), (2, 0, -1, -1), (0, 1, 0, 0), (1, 1, 0, 0))
    found = [1, 2]
    assert [total_weight(M) for M in path_matrices(B, found)] == [5, 9, 7]
    c = compare_ends(B, found, 5, 4)
    assert c.ok and c.n_content > 0, c.as_dict()
    assert all(s.ok for s in compare_steps(B, found, 4))


def test_the_transported_coefficients_are_the_generator_where_known():
    """Where `transport` reports a coefficient as known it equals the end
    quiver's own generator, including at charges the comparison would skip."""
    B = A3
    tr = transport(B, generator(B, 6), [1, 0], 6, 4)
    S_end = generator(tr.B, 4)
    assert tr.known
    for w in tr.known:
        if sum(w) <= 4:
            assert tr.S.get(w, HabiroElement.zero() if any(w) else HabiroElement.one()) == \
                S_end.get(w, HabiroElement.zero() if any(w) else HabiroElement.one()), w


def test_the_pentagon_compares_what_it_should():
    """A regression pin on the determinable region: the pentagon's edge at
    node 0 compares the whole target cone to depth 5 in the green frame."""
    c = compare_edge(A2, 0, 5)
    assert c.compared_by_depth == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, c.compared_by_depth
    assert path_matrices(A2, [0])[1] == mutate(A2, 0)
    assert set(VARIANTS) == {"green", "reverse", "mixed_green", "mixed_reverse"}


def test_the_reach_labels_say_what_a_comparison_tested():
    """A compared charge tests the subquiver where it is nonzero,
    and one avoiding the nodes that shift k's coordinate agrees whatever S is.
    The example has 6 arrows into node 0: at depth 6 the green
    comparison tests nothing, and at depth 8 = 2 + 6 it reaches every node."""
    from mutation_s_relation import depth_for_every_node, informative_nodes
    R = ((0, 2, -6), (-2, 0, 2), (6, -2, 0))
    assert informative_nodes(R, 0, "green") == {2} and informative_nodes(R, 0, "reverse") == {1}
    assert depth_for_every_node(R, 0, "green") == 8 and depth_for_every_node(R, 0, "reverse") == 4
    c6 = compare_edge(R, 0, 6)
    assert c6.ok and c6.n_informative == 0 and c6.reach == "nothing_informative"
    c8 = compare_edge(R, 0, 8)
    assert c8.ok and c8.n_every_node > 0 and c8.reach == "whole_quiver", c8.as_dict()
    # a source or a sink: every comparison agrees automatically (the peel on one
    # side, the reflection identity on the other)
    for B, k in ((KR3, 0), (KR3, 1), (A2, 0), (A2, 1), (A3, 0), (A3, 2)):
        for v in ("green", "reverse"):
            c = compare_edge(B, k, 6, variant=v)
            assert c.ok and c.reach == "nothing_informative" and c.n_informative == 0, (B, k, v)
    assert compare_edge(A3, 1, 5).reach == "whole_quiver"
    assert depth_for_every_node(KR3, 0, "green") == 1
    # Markov at depth 4 already reaches every node
    assert compare_edge(MARKOV, 0, 4).reach == "whole_quiver"


def test_the_closed_form_of_the_compared_charges():
    """`compared_charges_edge` equals the charges `transport` determines, so a
    run's counts can be recomputed from `(B, k, D)` alone."""
    from mutation_s_relation import compared_charges_edge
    rng = random.Random(11)
    for _ in range(25):
        n = rng.randint(2, 5)
        B = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x = rng.randint(-3, 3)
                B[i][j], B[j][i] = x, -x
        k, D = rng.randrange(n), rng.randint(3, 6)
        for kind in ("green", "reverse"):
            tr = transport(B, generator(B, D), [k], D, D, variant=kind, check_support=False)
            assert sorted(w for w in tr.known if sum(w) <= D) == sorted(compared_charges_edge(B, k, D, kind))


def test_crystalline_entries_of_the_tier_satisfy_it():
    """A few `crystalline_only` entries of the shipped tier (no negating
    sequence), one edge each, at depth 4."""
    from dictionary_loader import iter_enumerated_entries
    picked: dict[int, tuple] = {}
    for ent in iter_enumerated_entries():
        if str(ent.get("certification", "")).startswith("crystalline_only"):
            B = tuple(tuple(r) for r in ent["exchange"])
            picked.setdefault(len(B), B)
        if len(picked) >= 5:
            break
    assert picked
    for n, B in sorted(picked.items()):
        c = compare_edge(B, n - 1, 4)
        assert c.ok and c.n_content > 0, (B, c.as_dict())


if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception:
            failures += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} failure(s).")
        sys.exit(1)
    print("\nAll mutation-relation tests passed.")
