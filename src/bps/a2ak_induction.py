"""``a2ak_induction`` — the ``[A_2, A_k]`` family and its node-deletion tower.

The Argyres-Douglas family whose BPS quiver is the ``A_2 x A_k`` product the
quiver catalogue draws as the "rectangular form" (the ``2 x k`` bipartite grid,
every square cyclically oriented).  Rank ``2k``.

Cluster type, measured (``tests/test_a2ak_induction.py``)
--------------------------------------------------------

    k = 1   A_2  (pentagon)      k = 3   E_6
    k = 2   D_4                  k = 4   E_8
    k >= 5  NOT of finite cluster type

The mutation classes close for ``k <= 3`` (2, 50, 42840 labelled exchange
matrices); at ``k = 4`` an acyclic representative is the ``E_8`` Dynkin quiver;
at ``k = 5, 6`` a ``|b_ij| = 2`` witness appears within a few hundred mutations,
which is Fomin-Zelevinsky's criterion for infinite type.

⚠ The catalogue and ``tests/the source repository's archive` attribute ``[A_2, A_4]``
to ``E_7``.  That is wrong -- ``A_2 x A_4`` has 8 nodes and is ``E_8``; ``E_7``
is not an A-type product at all.  The *computed* content of those files is fine
(their exchange matrix really is the ``E_7`` tree); only the attribution is bad.

Flavour, and why the flavoured members are gauged
-------------------------------------------------

``dim ker B = 2`` exactly when ``3 | (k + 1)`` (measured k = 1..9), else 0 --
i.e. at ``k = 2, 5, 8, ...``; elsewhere the family is unflavoured.

Those members are built **u(1)^2-gauged**.  Gauging is the lattice extension the repo already uses for the
u(1)-gauged members of the sibling ``[A_1, A_odd]`` / ``[A_1, D_even]`` families:
adjoin one magnetic dual per flavour generator, leaving the node charges alone.
Controlled against ``u1_hexagon_kalg.B_GAUGED`` (the u(1)-gauged ``[A_1, A_3]``,
whose gauged pairing is the ``A_4`` linear one with 3 dynamical nodes): the
construction here reproduces it up to a unimodular shift fixing the node
sublattice.

The payoff is measured, not decorative:

* the coefficient ring drops from ``AbelianZPlusRing(rank=2)`` to
  ``TrivialZPlusRing()`` on *both* sides of every rung, so no rung carries a
  spectator flavour into its IR;
* ``Tr(1)`` for ``[A_2, A_2]`` costs 0.7 s gauged (a plain integer q-series)
  against 181.9 s ungauged (a flavour-character-valued series);
* gauging the top of the tower once suffices -- the enlarged ambient lattice
  supplies the magnetic duals for every flavoured member met on the way down.

The tower
---------

``A_2 x A_{k-1}`` is *literally* the induced subquiver of ``A_2 x A_k`` on the
first ``2(k-1)`` nodes (equality of matrices, checked k = 2..12), so deleting one
end column -- two nodes -- is well posed at every ``k``, finite type or not.
``SubquiverRG`` builds each rung and ``.then()`` composes them, walking

    [A_2, A_5] -> [A_2, A_4] -> [A_2, A_3] -> [A_2, A_2] -> [A_2, A_1]

with the IR exchange matrix exactly ``A_2 x A_{k-1}`` at every step.  This is the
``[A_2, A_k]`` counterpart of ``a1a2k_induction`` for ``[A_1, A_{2k}]`` and of
``g_matter_over_matter.matter_removal_tower`` for ``(G, N)`` -- new algebras by RG flow, with their algebra embeddings
in iterated form.  Note the contrast with ``a1a2k_induction``, whose two-node
drop leaves ``[A_1, A_{2k-2}] (x) QT(Z^2)``: here the IR is the smaller member
itself, with no spectator factor, once the flavoured members are gauged.

The rung out of ``[A_2, A_3]`` lands on the u(1)^2-gauged ``[A_2, A_2]``, and
that identification is certified by a ``KAlgebraIso`` (unit, round trip,
multiplicativity, rho-equivariance, trace-equivariance) in the suite.

A finite BPS chamber exists at every ``k`` reached so far, of length ``3k``
(measured k = 1..6) -- including ``k = 5, 6``, which are *not* of finite cluster
type.  Finite chamber and finite cluster type are independent properties; pure
SU(2) is the standing example of the first without the second.
"""
from __future__ import annotations

from typing import Sequence

from bps_kalgebra import BPSKAlgebra
from bps_quiver_tools import BPSQuiver
from rg_flow import SubquiverRG
from snf_kernel import integer_kernel_and_section

__all__ = [
    "square_product", "flavour_rank", "gauge_u1", "member",
    "column_drop", "tower", "KNOWN_NEGATING_SEQUENCES",
]

# Negating sequences for the 2 x k grid chart, found with
# `BPSQuiver.find_negating_sequence` (the cluster-side definition of the DT
# invariant).  Cached because the k >= 5 searches are minutes long: k = 5 took
# 46 s and k = 6 took 753 s.  Each yields a spec of length 3k.
KNOWN_NEGATING_SEQUENCES: dict[int, list[int]] = {
    1: [0, 1, 0],
    2: [1, 0, 2, 3, 0, 1],
    3: [0, 2, 4, 1, 3, 5, 2, 0, 4],
    4: [0, 2, 5, 1, 7, 3, 6, 2, 4, 0, 7, 5],
    5: [0, 2, 4, 6, 1, 5, 0, 8, 3, 9, 4, 7, 2, 8, 6],
}


def _alternating_a(n: int):
    """`A_n` with the alternating orientation; vertex `i` is a source iff even."""
    arrows = [(i, i + 1) if i % 2 == 0 else (i + 1, i) for i in range(n - 1)]
    return arrows, [i % 2 == 0 for i in range(n)]


def square_product(k: int, m: int = 2) -> list[list[int]]:
    """The exchange matrix of `A_m x A_k` — the `m x k` grid with every square
    cyclically oriented.  Vertex `(i, j)` is index `i * k + j`."""
    arrows_q, src_q = _alternating_a(m)
    arrows_r, src_r = _alternating_a(k)
    n = m * k
    B = [[0] * n for _ in range(n)]

    def add(a, b):
        B[a][b] += 1
        B[b][a] -= 1

    for (i, ip) in arrows_q:
        for j in range(k):
            add(*((ip * k + j, i * k + j) if src_r[j] else (i * k + j, ip * k + j)))
    for (j, jp) in arrows_r:
        for i in range(m):
            add(*((i * k + j, i * k + jp) if src_q[i] else (i * k + jp, i * k + j)))
    return B


def _rank(B: Sequence[Sequence[int]]) -> int:
    from fractions import Fraction
    M = [[Fraction(x) for x in row] for row in B]
    n, r = len(M), 0
    for c in range(n):
        p = next((i for i in range(r, n) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        for i in range(n):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def flavour_rank(k: int) -> int:
    """`dim ker B` for `[A_2, A_k]` — measured to be 2 iff `3 | (k + 1)`, else 0."""
    B = square_product(k)
    return len(B) - _rank(B)


def _dual_column(ker, j: int, n: int) -> list[int]:
    """Integer `x` with `ker[i] . x = delta_{ij}`; exists because `ker` is the
    (saturated) kernel of an integer matrix."""
    from fractions import Fraction
    f = len(ker)
    A = [[Fraction(v) for v in row] + [Fraction(int(i == j))]
         for i, row in enumerate(ker)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, f) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for i in range(f):
            if i != r and A[i][c] != 0:
                fac = A[i][c]
                A[i] = [a - fac * b for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == f:
            break
    sol = [Fraction(0)] * n
    for i, c in enumerate(piv):
        sol[c] = A[i][n]
    if any(v.denominator != 1 for v in sol):
        raise ValueError(f"non-integral magnetic dual: {sol}")
    return [int(v) for v in sol]


def gauge_u1(B: Sequence[Sequence[int]]) -> tuple[list[list[int]], int]:
    """u(1)^f-gauge a (possibly degenerate) pairing: extend `Z^n -> Z^{n+f}` with
    one magnetic dual per flavour generator, `<ker_i, m_j> = delta_ij`.  Node
    charges are unchanged, so the exchange matrix on the nodes is untouched and
    only `ker` is killed.  Returns `(B_gauged, f)`; `f = 0` leaves `B` alone.

    Controlled against `u1_hexagon_kalg.B_GAUGED` — see the module docstring."""
    n = len(B)
    ker, _ = integer_kernel_and_section([list(r) for r in B])
    f = len(ker)
    if f == 0:
        return [list(r) for r in B], 0
    N = n + f
    Bg = [[0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            Bg[i][j] = B[i][j]
    for j in range(f):
        col = _dual_column(ker, j, n)
        for a in range(n):
            Bg[a][n + j] = col[a]
            Bg[n + j][a] = -col[a]
    return Bg, f


def member(k: int, *, gauged: bool = True, negating_sequence=None,
           max_depth: int | None = None) -> BPSKAlgebra:
    """`[A_2, A_k]` as a `BPSKAlgebra` on the `2 x k` grid chart.

    `gauged=True` (default) builds the flavoured members — `3 | (k + 1)` — in
    their u(1)^2-gauged form, so every member of the family is unflavoured and
    the tower's rungs carry no spectator flavour.  `gauged=False` gives the plain
    flavoured presentation, kept for comparison.

    The negating sequence is taken from `KNOWN_NEGATING_SEQUENCES` when
    available and searched for otherwise (minutes for `k >= 6`)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    B = square_product(k)
    n = 2 * k
    if gauged:
        B, _ = gauge_u1(B)
    N = len(B)
    charges = [tuple(1 if t == i else 0 for t in range(N)) for i in range(n)]
    exchange = [[sum(B[a][b] * gi[a] * gj[b] for a in range(N) for b in range(N))
                 for gj in charges] for gi in charges]
    if exchange != square_product(k):
        raise AssertionError("gauging changed the exchange matrix")
    quiver = BPSQuiver(charges, None, exchange)
    seq = negating_sequence or KNOWN_NEGATING_SEQUENCES.get(k)
    if seq is None:
        seq = quiver.find_negating_sequence(max_depth=max_depth or (3 * k + 2))
        if seq is None:
            raise ValueError(f"no negating sequence found for k={k}")
    spec = [tuple(g) for g in quiver.build_spectrum_generator(seq)]
    return BPSKAlgebra(pairing=B, node_charges=charges, spec=spec)


def column_drop(algebra: BPSKAlgebra, k: int) -> SubquiverRG:
    """The `[A_2, A_k] -> [A_2, A_{k-1}]` flow: delete the grid's last column,
    i.e. the two nodes `(i, k-1)`.  The surviving induced subquiver is exactly
    `A_2 x A_{k-1}`."""
    if k < 2:
        raise ValueError(f"nothing to drop below k=2, got {k}")
    return SubquiverRG(algebra, [i * k + (k - 1) for i in range(2)])


def tower(k: int, *, gauged: bool = True) -> list[SubquiverRG]:
    """The rungs `[A_2, A_k] -> ... -> [A_2, A_1]`, outermost first.

    Each rung's UV is the previous rung's IR, so the list composes with
    `RGKAlgebra.then` (which requires `first.auxiliary() is second`)."""
    rungs, uv = [], member(k, gauged=gauged)
    for j in range(k, 1, -1):
        flow = column_drop(uv, j)
        rungs.append(flow)
        uv = flow.auxiliary()
    return rungs


# ---------------------------------------------------------------------------
# The [A_2, A_2] rung == the standalone u(1)^2-gauged member
# ---------------------------------------------------------------------------

# `A_2 x A_3`'s last-column drop lands on a 4-node algebra in the rank-6
# `[A_2, A_3]` lattice; the standalone `member(2)` is a 4-node algebra in its own
# rank-6 gauged lattice.  They are the same object: this unimodular `A` satisfies
# `A^T B_standalone A = B_rung` and carries node i to node i.  Frozen here after
# a search over the pairing-intertwining maps (100 of them, all agreeing on the
# canonical labels, which is forced: every label of the rung lies in the rank-4
# node sublattice, where `A` is already pinned by the node correspondence).
#
# `bpskalgebra_iso.find_isomorphism` does NOT find this witness -- it returns a
# silent `None` in both directions -- so the map is supplied rather than searched.
A2A2_RUNG_TO_GAUGED: tuple[tuple[int, ...], ...] = (
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, -1),
    (0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 1, 0),
    (0, 0, -1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1),
)


def _apply(M, v):
    return tuple(sum(M[a][b] * v[b] for b in range(len(v))) for a in range(len(M)))


def _int_inverse(A):
    from fractions import Fraction
    n = len(A)
    M = [[Fraction(A[i][j]) for j in range(n)]
         + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [[int(M[i][n + j]) for j in range(n)] for i in range(n)]


def a2a2_rung_iso(rung=None, standalone=None):
    """`KAlgebraIso` certifying that the `[A_2, A_3] -> [A_2, A_2]` rung's IR is
    the standalone u(1)^2-gauged `[A_2, A_2]` — the statement that the tower
    hands you the gauged member rather than it being an extra convention."""
    from kalgebra import Element
    from kalgebra_iso import KAlgebraIso
    from laurent_poly import LaurentPoly

    if rung is None:
        rung = column_drop(member(3), 3).auxiliary()
    if standalone is None:
        standalone = member(2)
    A = [list(r) for r in A2A2_RUNG_TO_GAUGED]
    Ainv = _int_inverse(A)
    one = LaurentPoly.one()
    return KAlgebraIso(
        rung, standalone,
        lambda lbl: Element({_apply(A, lbl): one}),
        lambda lbl: Element({_apply(Ainv, lbl): one}),
        name="[A_2, A_2] tower rung == standalone u(1)^2-gauged [A_2, A_2]",
    )
