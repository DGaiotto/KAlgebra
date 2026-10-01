"""Tests for `quiver_enumeration`: the canonical form is an
isomorphism invariant with the right automorphism count, and the enumeration
of connected quivers by total arrow weight reproduces the independent census
(a probe in the source repository, networkx) cell by cell.

Run:  `python3 run_tests.py`
"""

from __future__ import annotations

import itertools
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from quiver_enumeration import (  # noqa: E402
    canonical_form, canonical_spec, cell_counts, connected_components, delete_node,
    enumerate_connected_quivers, is_acyclic, is_connected, mutate,
    mutation_components, relabel_spec, source_sink_order,
    strongly_connected_components, total_weight,
)

# The census (networkx, independent implementation) — the design notes of the design record.
CENSUS_CELLS = {
    (1, 0): (1, 1), (2, 1): (1, 1), (2, 2): (1, 1), (2, 3): (1, 1), (2, 4): (1, 1),
    (2, 5): (1, 1), (3, 2): (3, 3), (3, 3): (6, 5), (3, 4): (11, 10), (3, 5): (16, 14),
    (4, 3): (8, 8), (4, 4): (30, 27), (4, 5): (86, 73), (5, 4): (27, 27),
    (5, 5): (148, 134), (6, 5): (91, 91),
}
CENSUS_COMPONENTS = {1: 2, 2: 4, 3: 9, 4: 23, 5: 63}


def _random_quiver(rng, n):
    B = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.choice([0, 0, 0, 1, -1, 2, -2, 3])
            B[i][j] = v
            B[j][i] = -v
    return B


def _permuted(B, p):
    n = len(B)
    return [[B[p[i]][p[j]] for j in range(n)] for i in range(n)]


def test_canonical_form_is_permutation_invariant():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 6)
        B = _random_quiver(rng, n)
        p = list(range(n))
        rng.shuffle(p)
        cf, cf2 = canonical_form(B), canonical_form(_permuted(B, p))
        assert cf.key == cf2.key
        assert cf.aut_order == cf2.aut_order
        # perm semantics: key[i][j] = B[perm[i]][perm[j]]
        assert all(cf.key[i][j] == B[cf.perm[i]][cf.perm[j]]
                   for i in range(n) for j in range(n))
        for tau in cf.automorphisms:
            assert all(cf.key[tau[i]][tau[j]] == cf.key[i][j]
                       for i in range(n) for j in range(n))


def test_automorphism_count_matches_brute_force():
    rng = random.Random(11)
    for _ in range(150):
        n = rng.randint(1, 5)
        B = _random_quiver(rng, n)
        cf = canonical_form(B)
        brute = sum(1 for p in itertools.permutations(range(n))
                    if all(B[p[i]][p[j]] == B[i][j] for i in range(n) for j in range(n)))
        assert cf.aut_order == brute, (B, cf.aut_order, brute)
        assert len(cf.automorphisms) == brute


def test_non_isomorphic_quivers_get_distinct_keys():
    # the pentagon vs the Kronecker-2 quiver, the cyclic vs the acyclic triangle
    pairs = [
        ([[0, 1], [-1, 0]], [[0, 2], [-2, 0]]),
        ([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
    ]
    for A, B in pairs:
        assert canonical_form(A).key != canonical_form(B).key


def test_enumeration_matches_census():
    recs = enumerate_connected_quivers(5)
    assert cell_counts(recs.values()) == CENSUS_CELLS
    for E, want in CENSUS_COMPONENTS.items():
        sub = {k: r for k, r in recs.items() if r.e <= E}
        assert len(set(mutation_components(sub, E).values())) == want, E


def test_rank_two_is_one_kronecker_quiver_per_weight():
    recs = enumerate_connected_quivers(6, max_rank=2)
    by_e = {}
    for r in recs.values():
        if r.n == 2:
            by_e.setdefault(r.e, []).append(r.key)
    assert all(len(v) == 1 for v in by_e.values())
    assert sorted(by_e) == [1, 2, 3, 4, 5, 6]
    assert all(abs(by_e[e][0][0][1]) == e for e in by_e)


def test_set_is_closed_under_node_deletion():
    recs = enumerate_connected_quivers(4)
    keys = set(recs)
    for key, r in recs.items():
        for k in range(r.n):
            if r.n == 1:
                continue
            sub = delete_node(key, k)
            for comp in connected_components(sub):
                piece = tuple(tuple(sub[i][j] for j in comp) for i in comp)
                assert canonical_form(piece).key in keys


def test_exits_and_mutation_consistency():
    recs = enumerate_connected_quivers(4)
    for key, r in recs.items():
        for k in range(r.n):
            M = mutate(key, k)
            assert total_weight(M) == r.exits[k]
            assert mutate(M, k) == key          # involution
            assert is_connected(M)


def test_acyclic_and_strip_order():
    A3 = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
    assert is_acyclic(A3) and source_sink_order(A3) == [0, 1, 2]
    C3 = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    assert not is_acyclic(C3) and source_sink_order(C3) is None


def test_strongly_connected_components():
    """The strongly connected components in a source-first order.

    Checked against the definitions directly, not against another
    implementation: every node lies in exactly one component; the nodes of a
    component reach each other (arrow i → j iff B[i][j] > 0) and no node
    outside it reaches it both ways; every arrow between two components points
    forward.  The author's characterisation (2026-09-23) is pinned too: a
    quiver is strongly connected iff its nodes admit no split into two
    non-empty parts with every arrow between them pointing the same way.
    """
    def reach(B):
        n = len(B)
        R = [{i} for i in range(n)]
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if B[i][j] > 0 and not R[j] <= R[i]:
                        R[i] |= R[j]
                        changed = True
        return R

    rng = random.Random(20260923)
    for _ in range(300):
        n = rng.randint(1, 9)
        B = _random_quiver(rng, n)
        comps = strongly_connected_components(B)
        pos = {i: c for c, comp in enumerate(comps) for i in comp}
        assert sorted(pos) == list(range(n))
        assert all(comp == sorted(comp) for comp in comps)
        R = reach(B)
        for i in range(n):
            for j in range(n):
                both = j in R[i] and i in R[j]
                assert both == (pos[i] == pos[j]), (B, comps)
                if B[i][j] > 0 and pos[i] != pos[j]:
                    assert pos[i] < pos[j], ("backward arrow", B, comps)
        if is_acyclic(B):
            assert [c for (c,) in comps] == source_sink_order(B)
        # the two-part characterisation, by brute force over the splits
        splittable = any(
            all(B[a][b] >= 0 for a in part for b in range(n) if b not in part)
            for r in range(1, n) for part in map(set, itertools.combinations(range(n), r)))
        assert splittable == (len(comps) > 1), (B, comps)

    three_cycle = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    assert strongly_connected_components(three_cycle) == [[0, 1, 2]]
    assert strongly_connected_components([[0, 1], [-1, 0]]) == [[0], [1]]
    # two oriented 3-cycles joined by one arrow 2 → 3, listed backwards
    n = 6
    B = [[0] * n for _ in range(n)]
    for i, j in [(3, 4), (4, 5), (5, 3), (0, 1), (1, 2), (2, 0), (2, 3)]:
        B[i][j], B[j][i] = 1, -1
    assert strongly_connected_components(B) == [[0, 1, 2], [3, 4, 5]]
    B[2][3], B[3][2] = -1, 1          # reverse the joining arrow
    assert strongly_connected_components(B) == [[3, 4, 5], [0, 1, 2]]
    assert strongly_connected_components([]) == []


def test_spec_relabel_and_canonical_spec():
    P = [[0, 1], [-1, 0]]
    cf = canonical_form(P)
    spec = ((1, 0), (0, 1))
    rel = relabel_spec(spec, cf.perm)
    assert set(rel) == {(1, 0), (0, 1)}
    K2 = [[0, 2], [-2, 0]]
    cf2 = canonical_form(K2)
    assert cf2.aut_order == 1
    # the swap is NOT an automorphism of the Kronecker quiver, so the
    # canonical spec is the spec itself
    assert canonical_spec(((0, 1), (1, 0)), cf2.automorphisms) == ((0, 1), (1, 0))




def test_enumeration_workers_match_sequential():
    """`enumerate_connected_quivers(E, workers=2)` returns the same records in
    the same insertion order as the sequential walk, and the parallel class
    index the same components."""
    from quiver_enumeration import enumerate_connected_quivers, mutation_components
    seq = enumerate_connected_quivers(5)
    par = enumerate_connected_quivers(5, workers=2)
    assert list(par) == list(seq)
    for k, r in seq.items():
        q = par[k]
        assert (q.n, q.e, q.aut_order, q.acyclic, q.exits, q.automorphisms) == \
               (r.n, r.e, r.aut_order, r.acyclic, r.exits, r.automorphisms)
    assert mutation_components(par, 5, workers=2) == mutation_components(seq, 5)


def test_strongly_connected_components_come_source_first():
    """`strongly_connected_components` lists the strongly connected
    components so that every arrow between two of them points from the
    earlier to the later, and `is_strongly_connected` agrees with it."""
    from quiver_enumeration import is_strongly_connected, strongly_connected_components
    tri = ((0, 1, -1), (-1, 0, 1), (1, -1, 0))                 # the oriented 3-cycle
    assert strongly_connected_components(tri) == [[0, 1, 2]] and is_strongly_connected(tri)
    # two oriented 3-cycles joined by one arrow 2 -> 3: no source and no sink,
    # yet two components (the design's example, §1.1)
    B = [[0] * 6 for _ in range(6)]
    for a, b in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)):
        B[a][b], B[b][a] = 1, -1
    assert strongly_connected_components(B) == [[0, 1, 2], [3, 4, 5]]
    assert not is_strongly_connected(B)
    B[2][3], B[3][2] = -1, 1                                   # the joining arrow reversed
    assert strongly_connected_components(B) == [[3, 4, 5], [0, 1, 2]]
    assert strongly_connected_components(((0,),)) == [[0]] and is_strongly_connected(((0,),))
    assert not is_strongly_connected(())
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 7)
        M = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = rng.choice((0, 0, 1, -1, 2))
                M[i][j], M[j][i] = v, -v
        comps = strongly_connected_components(M)
        assert sorted(x for c in comps for x in c) == list(range(n))
        pos = {x: k for k, c in enumerate(comps) for x in c}
        for i in range(n):
            for j in range(n):
                if M[i][j] > 0 and pos[i] != pos[j]:
                    assert pos[i] < pos[j], (M, comps)
        for c in comps:
            assert is_strongly_connected([[M[i][j] for j in c] for i in c]), (M, c)
        assert is_strongly_connected(M) == (len(comps) == 1)


def test_strongly_connected_walk_is_the_strongly_connected_part_of_the_enumeration():
    """The walk of `strongly_connected_redesign.md` §4 finds exactly
    the strongly connected quivers of the full enumeration, matches the
    design's census through weight 9, and returns the same records in the
    same order with workers."""
    from quiver_enumeration import enumerate_strongly_connected_quivers, is_strongly_connected
    full = enumerate_connected_quivers(7)
    for E in range(0, 8):
        walk = enumerate_strongly_connected_quivers(E)
        assert set(walk) == {k for k, r in full.items() if r.e <= E and is_strongly_connected(k)}, E
    walk = enumerate_strongly_connected_quivers(9)
    counts: dict[int, int] = {}
    for r in walk.values():
        counts[r.e] = counts.get(r.e, 0) + 1
    assert counts == {0: 1, 3: 1, 4: 2, 5: 6, 6: 23, 7: 96, 8: 481, 9: 2586}, counts
    weights = [r.e for r in walk.values()]
    assert weights == sorted(weights), "the builder's stages follow this order: weight first"
    par = enumerate_strongly_connected_quivers(9, workers=2)
    assert list(par) == list(walk) and all(par[k] == walk[k] for k in walk)
    r = next(r for r in walk.values() if r.e == 9)
    assert r.n == len(r.key) and not r.acyclic and len(r.exits) == r.n


def test_compact_record_shares_rows_and_keeps_equality():
    """`compact_record` returns a record equal to the original (key as a
    value, automorphisms, exits) whose rows, automorphisms and exits are the
    shared objects of the intern table — the memory cut the enumeration and
    the builder rely on (3.1× on the weight-8 records)."""
    from quiver_enumeration import compact_record, enumerate_connected_quivers, intern_vec
    R = enumerate_connected_quivers(4)
    for k, r in R.items():
        c = compact_record(r)
        assert c.key == r.key == k
        assert (c.n, c.e, c.aut_order, c.acyclic, c.exits, c.automorphisms) == \
               (r.n, r.e, r.aut_order, r.acyclic, r.exits, r.automorphisms)
        for row in c.key:
            assert intern_vec(row) is row
        assert intern_vec(c.exits) is c.exits
        assert intern_vec(c.automorphisms) is c.automorphisms
    # the records the enumeration returns are already compact
    rows = {id(row) for r in R.values() for row in r.key}
    assert len(rows) < sum(len(r.key) for r in R.values())


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
    print("\nAll quiver-enumeration tests passed.")
