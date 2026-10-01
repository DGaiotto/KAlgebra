"""Tests for `a2ak_induction` — the `[A_2, A_k]` family and its tower.

Run:  `python3 run_tests.py`
"""
from __future__ import annotations

import random
import sys

from a2ak_induction import (
    KNOWN_NEGATING_SEQUENCES, a2a2_rung_iso, column_drop, flavour_rank,
    gauge_u1, member, square_product, tower,
)
from bps_quiver_tools import BPSQuiver
from kalgebra import Element
from laurent_poly import LaurentPoly

_ONE = LaurentPoly.one()


def _exchange(alg):
    P = [list(r) for r in alg.lattice.pairing]
    n = len(P)
    g = alg.node_charges
    return [[sum(P[a][b] * gi[a] * gj[b] for a in range(n) for b in range(n))
             for gj in g] for gi in g]


def _is_acyclic(B):
    n = len(B)
    adj = [[j for j in range(n) if B[i][j] > 0] for i in range(n)]
    colour = [0] * n

    def dfs(u):
        colour[u] = 1
        for v in adj[u]:
            if colour[v] == 1:
                return False
            if colour[v] == 0 and not dfs(v):
                return False
        colour[u] = 2
        return True

    return all(colour[u] != 0 or dfs(u) for u in range(n))


def _tree_canon(B):
    """AHU canonical form of the underlying graph (a tree, for Dynkin quivers)."""
    n = len(B)
    adj = {i: [j for j in range(n) if B[i][j] != 0] for i in range(n)}
    if sum(len(v) for v in adj.values()) // 2 != n - 1:
        return "not-a-tree"
    deg = {i: len(adj[i]) for i in range(n)}
    removed, cur, rem = set(), [i for i in range(n) if deg[i] <= 1], n
    while rem > 2:
        nxt = []
        for u in cur:
            removed.add(u)
            rem -= 1
            for v in adj[u]:
                if v not in removed:
                    deg[v] -= 1
                    if deg[v] == 1:
                        nxt.append(v)
        cur = nxt
    centres = [i for i in range(n) if i not in removed]

    def enc(u, p):
        return "(" + "".join(sorted(enc(v, u) for v in adj[u] if v != p)) + ")"

    return min(enc(c, -1) for c in centres)


def _dynkin(kind, n):
    if kind == "A":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif kind == "D":
        edges = [(i, i + 1) for i in range(n - 2)] + [(n - 3, n - 1)]
    else:
        edges = [(0, 1), (1, 3), (2, 3)] + [(i, i + 1) for i in range(3, n - 1)]
    B = [[0] * n for _ in range(n)]
    for a, b in edges:
        B[a][b] += 1
        B[b][a] -= 1
    return B


# ---------------------------------------------------------------------------
# The quiver
# ---------------------------------------------------------------------------


def test_square_product_shape_and_orientation():
    """Rank 2k, and every unit square cyclically oriented."""
    for k in range(1, 8):
        B = square_product(k)
        assert len(B) == 2 * k
        assert all(B[i][j] == -B[j][i] for i in range(2 * k) for j in range(2 * k))
        for j in range(k - 1):
            idx = [0 * k + j, 0 * k + j + 1, 1 * k + j + 1, 1 * k + j]
            signs = [B[idx[t]][idx[(t + 1) % 4]] for t in range(4)]
            assert all(s > 0 for s in signs) or all(s < 0 for s in signs), (k, j)


def test_induced_subquiver_is_the_previous_member():
    """`A_2 x A_{k-1}` is LITERALLY the induced subquiver on the first 2(k-1)
    nodes — equality of matrices, not merely isomorphism."""
    for k in range(2, 13):
        Bk = square_product(k)
        keep = [i * k + j for i in range(2) for j in range(k - 1)]
        sub = [[Bk[a][b] for b in keep] for a in keep]
        assert sub == square_product(k - 1), k


def test_cluster_type_of_the_finite_members():
    """k = 2, 3, 4 have acyclic representatives D_4, E_6, E_8 (NOT E_7)."""
    want = {2: ("D", 4), 3: ("E", 6), 4: ("E", 8)}
    for k, (kind, n) in want.items():
        target = _tree_canon(_dynkin(kind, n))
        found = None
        for seed in range(6):
            rng = random.Random(seed)
            q = BPSQuiver([tuple(1 if t == i else 0 for t in range(2 * k))
                           for i in range(2 * k)], None, square_product(k))
            for _ in range(4000):
                if _is_acyclic(q.exchange):
                    found = _tree_canon(q.exchange)
                    break
                q = q.mutate(rng.randrange(2 * k))
            if found is not None:
                break
        assert found == target, (k, kind, n, found)


def test_k5_is_not_of_finite_cluster_type():
    """Fomin-Zelevinsky: finite type iff |b_ij| <= 1 throughout the class."""
    k = 5
    rng = random.Random(0)
    q = BPSQuiver([tuple(1 if t == i else 0 for t in range(2 * k))
                   for i in range(2 * k)], None, square_product(k))
    worst = 1
    for _ in range(4000):
        q = q.mutate(rng.randrange(2 * k))
        worst = max(worst, max(abs(x) for row in q.exchange for x in row))
        if worst >= 2:
            break
    assert worst >= 2, "no |b_ij| >= 2 witness found for [A_2, A_5]"


def test_flavour_rank_law():
    """`dim ker B = 2` exactly when `3 | (k + 1)`, else 0."""
    for k in range(1, 10):
        assert flavour_rank(k) == (2 if (k + 1) % 3 == 0 else 0), k


# ---------------------------------------------------------------------------
# Gauging
# ---------------------------------------------------------------------------


def test_gauging_control_against_the_repo_hexagon():
    """The u(1)-gauged `[A_1, A_3]` recipe must reproduce the repo's own
    `u1_hexagon_kalg.B_GAUGED` up to a unimodular shift fixing the nodes."""
    from u1_hexagon_kalg import B_GAUGED
    a3 = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
    Bg, f = gauge_u1(a3)
    assert f == 1
    assert len(Bg) == len(B_GAUGED) == 4
    match = None
    for s0 in range(-4, 5):
        for s1 in range(-4, 5):
            for s2 in range(-4, 5):
                s = (s0, s1, s2)
                shifted = [row[:] for row in Bg]
                for a in range(3):
                    shifted[a][3] = Bg[a][3] + sum(a3[a][b] * s[b] for b in range(3))
                    shifted[3][a] = -shifted[a][3]
                if shifted == [list(r) for r in B_GAUGED]:
                    match = s
    assert match is not None, "gauging does not match the repo's B_GAUGED"


def test_gauging_kills_the_flavour_and_keeps_the_exchange_matrix():
    for k in (2, 5):
        assert flavour_rank(k) == 2
        Bg, f = gauge_u1(square_product(k))
        assert f == 2 and len(Bg) == 2 * k + 2
        n = 2 * k
        N = len(Bg)
        ch = [tuple(1 if t == i else 0 for t in range(N)) for i in range(n)]
        ex = [[sum(Bg[a][b] * gi[a] * gj[b] for a in range(N) for b in range(N))
               for gj in ch] for gi in ch]
        assert ex == square_product(k)


def test_members_are_unflavoured_when_gauged():
    from zplus_ring import TrivialZPlusRing
    for k in range(1, 6):
        assert member(k).coefficient_ring() == TrivialZPlusRing(), k
    # and the flavoured ones really are flavoured without gauging
    assert member(2, gauged=False).coefficient_ring() != TrivialZPlusRing()


def test_spec_length_is_3k():
    """A finite BPS chamber of 3k hypermultiplets at every k reached — including
    k = 5, which is NOT of finite cluster type."""
    for k, seq in sorted(KNOWN_NEGATING_SEQUENCES.items()):
        assert len(member(k).spec) == 3 * k, k


# ---------------------------------------------------------------------------
# The tower
# ---------------------------------------------------------------------------


def test_tower_rungs_land_on_the_previous_member():
    rungs = tower(5)
    assert len(rungs) == 4
    for idx, flow in enumerate(rungs):
        k = 5 - idx
        ir = flow.auxiliary()
        assert _exchange(ir) == square_product(k - 1), k
        assert len(ir.node_charges) == 2 * (k - 1)
        assert len(ir.spec) == 3 * (k - 1)


def test_tower_rungs_satisfy_the_rg_axioms():
    from zplus_ring import TrivialZPlusRing
    rungs = tower(5)
    for idx, flow in enumerate(rungs):
        labels = [tuple(c) for c in flow.starting_algebra().node_charges][:3]
        assert flow.verify_rg_unital()
        for a in labels:
            assert flow.verify_rg_bar_invariant(a)
            for b in labels:
                assert flow.verify_rg_multiplicative(a, b)
        # no rung carries a spectator flavour into its IR
        assert flow.auxiliary().coefficient_ring() == TrivialZPlusRing()


def test_tower_composes_to_the_pentagon():
    rungs = tower(5)
    composed = rungs[0]
    for flow in rungs[1:]:
        composed = composed.then(flow)
    final = composed.auxiliary()
    assert _exchange(final) == square_product(1)
    assert len(final.node_charges) == 2


def test_a2a2_rung_is_the_standalone_gauged_member():
    """The [A_2, A_3] drop hands you the u(1)^2-gauged [A_2, A_2] — certified."""
    iso = a2a2_rung_iso()
    src = [Element({tuple(c): _ONE}) for c in iso.source.node_charges]
    tgt = [Element({tuple(c): _ONE}) for c in iso.target.node_charges]
    assert iso.verify_unit()
    assert iso.verify_round_trip(src, tgt)
    assert iso.verify_multiplicative([(a, b) for a in src for b in src],
                                     [(c, d) for c in tgt for d in tgt])
    assert iso.verify_rho_equivariant(src, tgt)
    assert iso.verify_trace_equivariant(src, tgt, 4)


def test_a2a2_rung_and_standalone_share_the_schur_index():
    rung = column_drop(member(3), 3).auxiliary()
    standalone = member(2)
    assert (rung.trace(rung.identity(), K=6)
            == standalone.trace(standalone.identity(), K=6))


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
    print("\nAll [A_2, A_k] induction tests passed.")
