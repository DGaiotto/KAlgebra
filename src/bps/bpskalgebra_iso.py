"""Isomorphism between two `BPSKAlgebra` instances.

A `BPSKAlgebra` is determined by `(pairing B, node_charges, spec)`,
NOT by `(B, node_charges)` alone -- the spec encodes which factorization
of the spectrum generator is used, which corresponds physically to a
choice of quiver potential / superpotential.  Different non-equivalent
specs on the same quiver may represent **inequivalent algebras**.

This module provides a *sufficient* certificate of isomorphism: two
`BPSKAlgebra`s are *certified isomorphic* by a witness `(A, chain)` if

1. There is a unimodular integer matrix `A : Γ_1 → Γ_2` such that
   `A^T B_2 A = B_1` (intertwines the pairings) and `A` maps the
   first algebra's node charges to the second's (as multisets).
2. Applying `A` to the first algebra's spec yields a spec
   reachable from the second's by a finite sequence of *known
   algebra-preserving local moves*: pentagon collapses + commute
   swaps + pentagon expansions.

When `find_isomorphism` returns a witness, the algebras are
isomorphic.  When it returns `None`, **we have NOT proven they are
non-isomorphic** -- the witness is one sufficient condition.

This module provides:

* `find_isomorphism(alg_1, alg_2)`  →  `Iso | None`
* `verify_isomorphism(alg_1, alg_2, iso)`  →  bool
* `find_weak_isomorphism(alg_1, alg_2, ...)` → `WeakIso | None`
  (matches at non-root charts of either algebra).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from typing import Optional, Sequence
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chart_graph import _enumerate_local_moves, _apply_local_move
from snf_kernel import int_det as _int_det


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclass
class Iso:
    """Witness that two `BPSKAlgebra`s are isomorphic.

    * `A`: rank × rank unimodular integer matrix with
      `A^T B_2 A = B_1` (pairing-intertwining) and
      `A · alg_1.node_charges = alg_2.node_charges` (as multisets).
    * `local_move_chain`: sequence of `(kind, position)` tuples such
      that replaying them on `[A · g for g in alg_1.spec]` yields
      `alg_2.spec`.

    Use `verify_isomorphism(alg_1, alg_2, iso)` to validate.
    """

    A: tuple[tuple[int, ...], ...]
    local_move_chain: list


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def _int_matmul(A, B):
    n = len(A)
    m = len(B[0]) if B else 0
    k = len(B)
    out = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = sum(A[i][r] * B[r][j] for r in range(k))
    return out


def _int_transpose(A):
    return [list(row) for row in zip(*A)]


def _solve_AM_eq_N(M, N):
    """Solve A · M = N over Q for square M.  Returns A or None
    if M is singular.  All inputs are int matrices; output is
    Fraction matrix.
    """
    n = len(M)
    if any(len(row) != n for row in M):
        return None
    # Augment [M^T | N^T] then row-reduce columnwise.
    # Easier: A^T = M^T \ N^T  (i.e., A^T columns satisfy M^T x = N^T col).
    # Solve via Gauss-Jordan on [M^T | N^T].
    rank = n
    aug = [
        [Fraction(M[j][i]) for j in range(rank)]   # row i = column i of M = M^T row i
        + [Fraction(N[j][i]) for j in range(rank)]
        for i in range(rank)
    ]
    # Gauss-Jordan
    for c in range(rank):
        piv = None
        for r in range(c, rank):
            if aug[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None
        aug[c], aug[piv] = aug[piv], aug[c]
        pv = aug[c][c]
        aug[c] = [x / pv for x in aug[c]]
        for r in range(rank):
            if r != c and aug[r][c] != 0:
                fac = aug[r][c]
                aug[r] = [aug[r][k] - fac * aug[c][k] for k in range(2 * rank)]
    # A^T columns are aug[i][rank:].  So A is transpose of that.
    AT = [row[rank:] for row in aug]
    A = [[AT[j][i] for j in range(rank)] for i in range(rank)]
    return A


def _all_integer(M) -> bool:
    return all(
        all(x.denominator == 1 if isinstance(x, Fraction) else True for x in row)
        for row in M
    )


def _to_int_matrix(M):
    return tuple(
        tuple(int(x) if isinstance(x, Fraction) else x for x in row)
        for row in M
    )


# ---------------------------------------------------------------------------
# Cross-lattice automorphism finder
# ---------------------------------------------------------------------------


def _is_standard_basis(nodes, rank):
    """True iff `nodes` is exactly the rank-`rank` standard basis
    `(e_1, ..., e_rank)` in some order (each `e_i` appearing once).
    This is the common case from `entry_to_bpskalgebra`, where iso
    auts must be permutation matrices."""
    if len(nodes) != rank:
        return False
    seen = set()
    for g in nodes:
        if len(g) != rank:
            return False
        nz = [k for k in range(rank) if g[k] != 0]
        if len(nz) != 1 or g[nz[0]] != 1:
            return False
        if nz[0] in seen:
            return False
        seen.add(nz[0])
    return True


def _find_pairing_intertwining_perm(B_1, B_2, nodes_1, nodes_2):
    """Permutation fast-path for `_find_pairing_intertwining_aut`.

    When both node lists are the standard basis (the common case
    from `entry_to_bpskalgebra`), the only valid auts are
    permutation matrices: enumerate `π ∈ S_n` with
    `B_2[π(i)][π(j)] = B_1[i][j]` for all `i, j`, and return the
    induced permutation matrix.

    Avoids the rational Gauss-Jordan solve used by the general
    path, which is `~30 s / 200 same-quiver pairs at n=6` of
    `fractions.py` arithmetic that this routine sidesteps entirely.
    """
    rank = len(B_1)
    if not _is_standard_basis(nodes_1, rank) \
            or not _is_standard_basis(nodes_2, rank):
        return None
    # Map each standard basis element to its non-zero index, so we
    # can rebuild `A · nodes_1` after finding π.
    nz_1 = tuple(
        next(k for k in range(rank) if g[k] != 0) for g in nodes_1
    )
    nz_2 = tuple(
        next(k for k in range(rank) if g[k] != 0) for g in nodes_2
    )
    if sorted(nz_1) != sorted(nz_2):
        # Different basis-element multisets => no permutation aut.
        return None
    for perm in permutations(range(rank)):
        ok = True
        for i in range(rank):
            pi = perm[i]
            row_B2 = B_2[pi]
            row_B1 = B_1[i]
            for j in range(rank):
                if row_B2[perm[j]] != row_B1[j]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # A is the permutation matrix with A · e_j = e_{π(j)}.
        A = [[0] * rank for _ in range(rank)]
        for j in range(rank):
            A[perm[j]][j] = 1
        # Multiset check: π applied to nz_1 must equal nz_2 as a
        # multiset.  Since both sides are permutations of
        # {0, ..., rank-1} (under _is_standard_basis), this is
        # automatically satisfied; assert for safety.
        return tuple(tuple(row) for row in A)
    return None


def _find_pairing_intertwining_aut(
    B_1, B_2, nodes_1, nodes_2,
):
    """Return a unimodular integer matrix `A` with `A^T B_2 A = B_1`
    AND `A · nodes_1 = nodes_2` as multisets, or `None` if no such
    `A` exists.

    Strategy: pick a rank-subset of `nodes_1` as a basis of the
    rank-`n` lattice, pick a corresponding rank-subset of `nodes_2`
    in some order, solve `A · (basis_1 columns) = (basis_2 columns)`
    over `Q`, and verify integer + unimodular + pairing-intertwining
    + multiset-matching.

    For small ranks (≤ 6) the enumeration is `O((n choose r)² · r!)`
    in the worst case but usually terminates quickly because most
    combinations fail at the integer-A check.

    Fast path: when both node lists are the standard basis (the
    common case from the dictionary builder), the answer must be
    a permutation matrix, so we enumerate `π ∈ S_n` directly and
    skip the rational Gauss-Jordan.
    """
    rank = len(B_1)
    if len(B_2) != rank or len(nodes_1) != len(nodes_2):
        return None
    # When both node lists are standard-basis (the common case from
    # `entry_to_bpskalgebra`), any pairing-intertwining aut is forced
    # to be a permutation matrix: `A · nodes_1 = nodes_2` as multisets
    # of standard-basis vectors means `A` permutes them.  So the perm
    # enumeration is *complete*: a None return is definitive, and the
    # rational Gauss-Jordan fallback would only re-derive the same
    # answer at much higher cost (Fraction arithmetic on each candidate).
    # Skip the fallback entirely for this case.  This is what saves the
    # WL-signature false-collision pairs from blowing up: WL over-buckets
    # by design, putting non-iso quivers in the same bucket; the perm
    # path proves "no aut" in microseconds, but the rational fallback
    # used to grind for ~80 ms per pair before reaching the same verdict.
    nodes_1_std = _is_standard_basis(nodes_1, rank)
    nodes_2_std = _is_standard_basis(nodes_2, rank)
    if nodes_1_std and nodes_2_std:
        return _find_pairing_intertwining_perm(
            B_1, B_2, nodes_1, nodes_2,
        )
    A_perm = _find_pairing_intertwining_perm(
        B_1, B_2, nodes_1, nodes_2,
    )
    if A_perm is not None:
        return A_perm
    nodes_1_t = [tuple(g) for g in nodes_1]
    nodes_2_t = [tuple(g) for g in nodes_2]
    multiset_2 = sorted(nodes_2_t)
    n_nodes = len(nodes_1)

    for idx_1 in combinations(range(n_nodes), rank):
        # M_1 is rank×rank with column i = nodes_1[idx_1[i]]
        M_1 = [
            [nodes_1_t[idx_1[i]][r] for i in range(rank)]
            for r in range(rank)
        ]
        d1 = _int_det(M_1)
        if abs(d1) != 1:
            continue
        # Try every rank-subset and permutation of nodes_2 as targets.
        for idx_2 in combinations(range(n_nodes), rank):
            for perm in permutations(idx_2):
                M_2 = [
                    [nodes_2_t[perm[i]][r] for i in range(rank)]
                    for r in range(rank)
                ]
                A_q = _solve_AM_eq_N(M_1, M_2)
                if A_q is None:
                    continue
                if not _all_integer(A_q):
                    continue
                A = _to_int_matrix(A_q)
                # Unimodular?
                if abs(_int_det([list(row) for row in A])) != 1:
                    continue
                # Pairing-intertwining: A^T B_2 A = B_1
                AT_B2 = _int_matmul(_int_transpose(A),
                                    [list(r) for r in B_2])
                AT_B2_A = _int_matmul(AT_B2, [list(r) for r in A])
                if AT_B2_A != [list(row) for row in B_1]:
                    continue
                # Multiset-matching: A · nodes_1 = nodes_2 as multisets.
                mapped = sorted(
                    tuple(
                        sum(A[i][j] * g[j] for j in range(rank))
                        for i in range(rank)
                    )
                    for g in nodes_1_t
                )
                if mapped == multiset_2:
                    return A
    return None


# ---------------------------------------------------------------------------
# Local-move chain search between two specs (same chart)
# ---------------------------------------------------------------------------


# Each local move is reversed by another legal local move at the
# resulting state:
#   * commute(i)        is self-inverse (⟨a,b⟩=0 ⟹ ⟨b,a⟩=0);
#   * pent_expand(i)    is undone by pent_collapse(i) at the post-
#                       expand state — the new triple [b, a+b, a]
#                       matches the collapse pattern, and ⟨a,b⟩=1
#                       is preserved as ⟨Y[i+2], Y[i]⟩=1;
#   * pent_collapse(i)  is undone by pent_expand(i) at [a, b], where
#                       ⟨a,b⟩=1 was the precondition for the original
#                       collapse.
# So the move-edge relation on specs is symmetric, and bidirectional
# (meet-in-the-middle) BFS finds the same minimal-length chains as
# plain forward BFS using a single shared `_enumerate_local_moves`.
_INVERSE_MOVE_KIND = {
    "commute": "commute",
    "pent_expand": "pent_collapse",
    "pent_collapse": "pent_expand",
}


def _inverse_move(move):
    """Inverse of a local move (applied at the move's destination)."""
    return (_INVERSE_MOVE_KIND[move[0]], move[1])


def _reconstruct_chain(parents, state):
    """Walk parent pointers back to the root; return moves root→state.

    `parents[state]` is `(parent_state, move_from_parent_to_state)`
    or `None` if `state` is the root of its BFS tree.
    """
    moves = []
    cur = state
    while parents[cur] is not None:
        parent, move = parents[cur]
        moves.append(move)
        cur = parent
    moves.reverse()
    return moves


class BFSExplorer:
    """Lazy forward-BFS expansion of the local-move graph rooted at
    a given spec, designed to be reused across many bidirectional
    iso checks where the same `(pairing, spec)` recurs (e.g. a
    pairwise sweep through a bucket of charts sharing one quiver).

    `parents` maps each visited canonical spec to `(parent_spec, move)`
    or `None` for the root.  `frontier` is the list of states whose
    neighbours have not yet been enumerated.  `bg_cache` memoises
    `B @ b` per unique charge tuple (see `chart_graph._bg_compute`)
    and survives across resumed expansions.

    Mutated in-place when passed to
    `_find_local_moves_chain_bidirectional` as `src_explorer` /
    `dst_explorer`.  All explorers used together must share the same
    `pairing_l` and `max_extra_length`.
    """

    __slots__ = ("pairing_l", "max_extra_length",
                 "parents", "frontier", "base_len", "bg_cache")

    def __init__(self, spec, pairing_l, max_extra_length):
        self.pairing_l = pairing_l
        self.max_extra_length = max_extra_length
        self.parents = {spec: None}
        self.frontier = [spec]
        self.base_len = len(spec)
        self.bg_cache: dict = {}


def _expand_explorer(explorer, other_parents, max_visits):
    """Expand `explorer.frontier` until either (a) a state appears in
    `other_parents` (meet), (b) `max_visits` frontier states have
    been processed, or (c) the frontier is exhausted.

    Returns `(meet_state_or_None, visits_used)`.  Mutates the
    explorer in place: `parents` grows with newly discovered states
    and `frontier` is updated to the next layer (or the unprocessed
    suffix if the budget cap was reached).
    """
    parents = explorer.parents
    pairing_l = explorer.pairing_l
    bg_cache = explorer.bg_cache
    base_len = explorer.base_len
    max_extra = explorer.max_extra_length
    frontier = explorer.frontier
    new_frontier: list = []
    visits = 0
    for idx, s in enumerate(frontier):
        if visits >= max_visits:
            new_frontier.extend(frontier[idx:])
            explorer.frontier = new_frontier
            return None, visits
        visits += 1
        s_list = [list(g) for g in s]
        allow_pent = (len(s) - base_len) < max_extra
        for move in _enumerate_local_moves(
            s_list, pairing_l, allow_pent, bg_cache=bg_cache,
        ):
            s_new = _apply_local_move(s_list, move)
            key = tuple(map(tuple, s_new))
            if key in parents:
                continue
            parents[key] = (s, move)
            if key in other_parents:
                explorer.frontier = new_frontier
                return key, visits
            new_frontier.append(key)
    explorer.frontier = new_frontier
    return None, visits


def _find_local_moves_chain_unidirectional(
    src, dst, pairing_l, max_states, max_extra_length,
):
    base_len = len(src)
    parents = {src: None}
    frontier = [src]
    visited = 0
    bg_cache: dict = {}
    while frontier and visited < max_states:
        new_frontier = []
        for s in frontier:
            visited += 1
            if visited > max_states:
                break
            s_list = [list(g) for g in s]
            allow_pentagon = (len(s) - base_len) < max_extra_length
            for move in _enumerate_local_moves(
                s_list, pairing_l, allow_pentagon, bg_cache=bg_cache,
            ):
                s_new = _apply_local_move(s_list, move)
                key = tuple(map(tuple, s_new))
                if key in parents:
                    continue
                parents[key] = (s, move)
                if key == dst:
                    return _reconstruct_chain(parents, dst)
                new_frontier.append(key)
        frontier = new_frontier
    return None


def _find_local_moves_chain_bidirectional(
    src, dst, pairing_l, max_states, max_extra_length,
    *, src_explorer=None, dst_explorer=None,
):
    """Meet-in-the-middle BFS.

    Two BFS trees grow simultaneously: a forward tree rooted at `src`
    and a backward tree rooted at `dst`.  Because the local-move edge
    relation is symmetric (see `_inverse_move`), the backward tree
    uses the *same* `_enumerate_local_moves`; an edge `X → Y` in the
    backward tree means there is a forward move `Y → X` (its inverse).
    The pentagon-length budget is applied per side: forward allows
    `pent_expand` while `len(s) - len(src) < max_extra_length`,
    backward while `len(s) - len(dst) < max_extra_length`.

    On meet at state `m`, the combined forward chain is
    ``fwd[src→m] + [inverse(b) for b in reversed(bwd[dst→m])]``.

    `src_explorer` / `dst_explorer`, if provided, are reused (and
    extended) instead of starting fresh.  This is bucket-level
    BFS state caching: the same `(B, spec)` combination recurs across
    a pairwise sweep through a same-quiver bucket, and lazy
    persistent expansion lets each pair re-cross only newly needed
    states.  Both explorers must share `pairing_l` and
    `max_extra_length` with the call (the caller is responsible for
    keying the cache accordingly).
    """
    if src_explorer is None:
        src_explorer = BFSExplorer(src, pairing_l, max_extra_length)
    if dst_explorer is None:
        dst_explorer = BFSExplorer(dst, pairing_l, max_extra_length)

    # Quick-hit short-circuits: when one explorer's cached parents
    # already contains the other explorer's root, no new BFS work
    # is needed.
    if src in dst_explorer.parents:
        fwd = _reconstruct_chain(src_explorer.parents, src)  # empty
        bwd = _reconstruct_chain(dst_explorer.parents, src)
        return fwd + [_inverse_move(m) for m in reversed(bwd)]
    if dst in src_explorer.parents:
        fwd = _reconstruct_chain(src_explorer.parents, dst)
        bwd = _reconstruct_chain(dst_explorer.parents, dst)  # empty
        return fwd + [_inverse_move(m) for m in reversed(bwd)]

    visits_remaining = max_states
    while visits_remaining > 0:
        # Pick the explorer to expand: the one with the smaller
        # non-empty frontier; if one frontier is empty but the other
        # is not, expand the non-empty one (the empty side's cached
        # `parents` is still useful as a meet target).  When both are
        # empty, nothing more can be done.
        if src_explorer.frontier and dst_explorer.frontier:
            if len(src_explorer.frontier) <= len(dst_explorer.frontier):
                target, other = src_explorer, dst_explorer
            else:
                target, other = dst_explorer, src_explorer
        elif src_explorer.frontier:
            target, other = src_explorer, dst_explorer
        elif dst_explorer.frontier:
            target, other = dst_explorer, src_explorer
        else:
            return None
        meet, used = _expand_explorer(
            target, other.parents, visits_remaining,
        )
        visits_remaining -= used
        if meet is not None:
            fwd = _reconstruct_chain(src_explorer.parents, meet)
            bwd = _reconstruct_chain(dst_explorer.parents, meet)
            return fwd + [_inverse_move(m) for m in reversed(bwd)]
        # If we used 0 visits and didn't find meet, the chosen target's
        # frontier was already empty — but we filtered above so this
        # shouldn't happen.  Defensive break just in case.
        if used == 0:
            return None
    return None


def find_local_moves_chain(
    spec_src: Sequence[Sequence[int]],
    spec_dst: Sequence[Sequence[int]],
    pairing: Sequence[Sequence[int]],
    *,
    max_states: int = 4096,
    max_extra_length: int = 4,
    bidirectional: bool = True,
    src_explorer: Optional[BFSExplorer] = None,
    dst_explorer: Optional[BFSExplorer] = None,
) -> Optional[list]:
    """Search for a local-moves chain from `spec_src` to `spec_dst`.

    Local moves: commute swap (`⟨a, b⟩ = 0`), pentagon expand (`= +1`,
    grows by 1), pentagon collapse (matching pattern, shrinks by 1).

    Returns the sequence of `(kind, position)` tuples that transforms
    `spec_src` into `spec_dst`, or `None` if not reachable within
    budget.

    When `bidirectional=True` (default), uses meet-in-the-middle BFS:
    two BFS trees grow from `spec_src` and `spec_dst`, terminating
    when they meet on a common state.  This typically explores
    `O(b^(d/2))` states instead of `O(b^d)` for branching factor `b`
    and chain length `d`, which dominates the cost of
    `find_isomorphism` on same-quiver buckets (e.g. the n=6
    dictionary).  Pass `bidirectional=False` to fall back to the
    plain forward BFS.
    """
    src = tuple(tuple(g) for g in spec_src)
    dst = tuple(tuple(g) for g in spec_dst)
    if src == dst:
        return []
    pairing_l = [list(row) for row in pairing]
    if bidirectional:
        return _find_local_moves_chain_bidirectional(
            src, dst, pairing_l, max_states, max_extra_length,
            src_explorer=src_explorer, dst_explorer=dst_explorer,
        )
    return _find_local_moves_chain_unidirectional(
        src, dst, pairing_l, max_states, max_extra_length,
    )


# ---------------------------------------------------------------------------
# Isomorphism finder
# ---------------------------------------------------------------------------


def _apply_aut(A: Sequence[Sequence[int]], v: Sequence[int]) -> Vec:
    rank = len(A)
    return tuple(
        sum(A[i][j] * v[j] for j in range(rank)) for i in range(rank)
    )


def find_isomorphism_raw(
    pairing_1: Sequence[Sequence[int]],
    nodes_1: Sequence[Sequence[int]],
    spec_1: Sequence[Sequence[int]],
    pairing_2: Sequence[Sequence[int]],
    nodes_2: Sequence[Sequence[int]],
    spec_2: Sequence[Sequence[int]],
    *,
    max_states: int = 4096,
    max_extra_length: int = 4,
    bidirectional: bool = True,
    cache: Optional[dict] = None,
) -> Optional[Iso]:
    """Plain-data variant of `find_isomorphism` operating directly on
    `(pairing, node_charges, spec)` tuples.

    Skips both `BPSKAlgebra` constructions that the wrapper variant
    pays via `entry_to_bpskalgebra`.  Same recipe and same return type
    (`Iso | None`).  Caller is responsible for ensuring the implicit
    coefficient-ring consistency holds (this raw API does **not**
    invoke `coefficient_ring()`); for the dictionary builder's
    standard-basis entries the coefficient rings always agree, so the
    skip is sound.

    See `find_isomorphism` for the algorithmic details.
    """
    rank_1 = len(pairing_1)
    rank_2 = len(pairing_2)
    if rank_1 != rank_2:
        return None
    B_1 = [list(row) for row in pairing_1]
    B_2 = [list(row) for row in pairing_2]
    A = _find_pairing_intertwining_aut(
        B_1, B_2, nodes_1, nodes_2,
    )
    if A is None:
        return None
    spec_1_mapped = tuple(_apply_aut(A, g) for g in spec_1)
    spec_2_t = tuple(tuple(g) for g in spec_2)
    src_explorer = None
    dst_explorer = None
    if cache is not None and bidirectional:
        # Cache key: `(B_2 as tuple of tuples, spec as tuple of tuples)`.
        # All explorers in this cache must share the same
        # `max_extra_length` (the caller is responsible for keeping the
        # cache scoped to one budget configuration).
        B_2_t = tuple(tuple(r) for r in B_2)
        src_key = (B_2_t, spec_1_mapped)
        dst_key = (B_2_t, spec_2_t)
        src_explorer = cache.get(src_key)
        if src_explorer is None:
            src_explorer = BFSExplorer(
                spec_1_mapped, B_2, max_extra_length,
            )
            cache[src_key] = src_explorer
        dst_explorer = cache.get(dst_key)
        if dst_explorer is None:
            dst_explorer = BFSExplorer(
                spec_2_t, B_2, max_extra_length,
            )
            cache[dst_key] = dst_explorer
    chain = find_local_moves_chain(
        spec_1_mapped, spec_2, B_2,
        max_states=max_states, max_extra_length=max_extra_length,
        bidirectional=bidirectional,
        src_explorer=src_explorer, dst_explorer=dst_explorer,
    )
    if chain is None:
        return None
    return Iso(A=A, local_move_chain=chain)


def find_isomorphism(
    alg_1, alg_2, *,
    max_states: int = 4096,
    max_extra_length: int = 4,
    bidirectional: bool = True,
    cache: Optional[dict] = None,
) -> Optional[Iso]:
    """Search for an isomorphism witness between two `BPSKAlgebra`s.

    Now handles **cross-lattice** cases: `B_1` and `B_2` may differ as
    long as there is a unimodular `A` with `A^T B_2 A = B_1`.  The
    pairing-intertwining condition automatically maps `ker(B_1)` to
    `ker(B_2)`, so the abelian-flavour structure is preserved by any
    such `A`.  We also require the algebras' coefficient rings to
    match — different `R` means different algebra signatures (cf.
    base-ring change as a separate operation, not iso).

    Steps:
    1. `coefficient_ring()` consistency check.
    2. `_find_pairing_intertwining_aut(B_1, B_2, nodes_1, nodes_2)`
       enumerates unimodular `A` with `A^T B_2 A = B_1` and
       `A · nodes_1 = nodes_2` as multisets.
    3. Apply `A` to `alg_1.spec` factor-wise.
    4. BFS over local moves on the pairing `B_2` to find a chain
       transforming `A · spec_1` into `alg_2.spec`.

    Hot-path callers that already work in `(pairing, nodes, spec)`
    form (notably the dictionary builder's iso check) should call
    `find_isomorphism_raw` to skip the `BPSKAlgebra`-required
    `coefficient_ring()` check and avoid the wrapper construction
    overhead.
    """
    if alg_1.lattice.rank != alg_2.lattice.rank:
        return None
    # Coefficient-ring consistency: iso requires the same Z₊-ring R.
    # Different R means the algebras live over different coefficient
    # rings; relating them needs a base-ring change (cf.
    # `KAlgebra.base_change`), not an iso witness.  This check is
    # what `find_isomorphism_raw` skips -- callers that know their
    # entries share a coefficient ring (the build-pass case) can
    # bypass it via the raw variant.
    if alg_1.coefficient_ring() != alg_2.coefficient_ring():
        return None
    return find_isomorphism_raw(
        alg_1.lattice.pairing, alg_1.node_charges, alg_1.spec,
        alg_2.lattice.pairing, alg_2.node_charges, alg_2.spec,
        max_states=max_states,
        max_extra_length=max_extra_length,
        bidirectional=bidirectional,
        cache=cache,
    )


def verify_isomorphism(alg_1, alg_2, iso: Iso) -> bool:
    """Verify that `iso` is a valid isomorphism witness."""
    if alg_1.lattice.rank != alg_2.lattice.rank:
        return False
    if alg_1.coefficient_ring() != alg_2.coefficient_ring():
        return False
    rank = alg_1.lattice.rank
    A = iso.A
    B_1 = [list(row) for row in alg_1.lattice.pairing]
    B_2 = [list(row) for row in alg_2.lattice.pairing]
    # 1. A^T B_2 A = B_1  (pairing-intertwining)
    AT_B2_A = [
        [
            sum(A[k][i] * B_2[k][l] * A[l][j]
                for k in range(rank) for l in range(rank))
            for j in range(rank)
        ]
        for i in range(rank)
    ]
    if AT_B2_A != B_1:
        return False
    # 2. A · alg_1.nodes maps onto alg_2.nodes (as multisets)
    nodes_mapped = [_apply_aut(A, g) for g in alg_1.node_charges]
    if sorted(nodes_mapped) != sorted(tuple(g) for g in alg_2.node_charges):
        return False
    # 3. Replaying chain on A · spec_1 gives alg_2.spec.
    spec_mapped = [_apply_aut(A, g) for g in alg_1.spec]
    cur = list(spec_mapped)
    for move in iso.local_move_chain:
        cur = _apply_local_move(cur, move)
    return cur == [tuple(g) for g in alg_2.spec]


# ---------------------------------------------------------------------------
# Weak isomorphism: match at non-root charts of either algebra
# ---------------------------------------------------------------------------


@dataclass
class WeakIso:
    """Witness that two `BPSKAlgebra`s are isomorphic *at some pair of
    charts in their mutation orbits*.

    * `chain_1`: list of `(node_index, direction)` pairs taken on
      `alg_1`'s chart graph, ending at `chart_1`.
    * `chain_2`: same on `alg_2`'s chart graph, ending at `chart_2`.
    * `inner`: a strong-`Iso` witness between `chart_1` and `chart_2`
      treated as standalone BPSKAlgebras with `(B, chart.nodes,
      chart.spec)`.
    """

    chain_1: list
    chain_2: list
    inner: Iso


def _enumerate_charts(alg, max_depth: int, max_local_moves: int = 64):
    """BFS over `alg`'s chart graph up to `max_depth` mutation steps,
    yielding `(chain, chart)` pairs (chain is a list of `(k, direction)`).
    Includes the root with an empty chain.
    """
    graph = alg._chart_graph
    if graph is None:
        return
    # frontier: list of (chart_id, chain)
    frontier = [(graph.root_id, [])]
    seen_chart_ids = {graph.root_id}
    yield ([], graph.chart(graph.root_id))
    for depth in range(max_depth):
        new_frontier = []
        for cid, chain in frontier:
            cur_chart = graph.chart(cid)
            for k in range(len(cur_chart.nodes)):
                for direction in ("fwd", "inv"):
                    dst_id = graph.mutate(
                        cid, k, direction=direction,
                        max_local_moves=max_local_moves,
                    )
                    if dst_id is None:
                        continue
                    if dst_id in seen_chart_ids:
                        continue
                    seen_chart_ids.add(dst_id)
                    new_chain = chain + [(k, direction)]
                    new_frontier.append((dst_id, new_chain))
                    yield (new_chain, graph.chart(dst_id))
        frontier = new_frontier


def find_weak_isomorphism(
    alg_1, alg_2,
    *,
    max_depth_1: int = 1,
    max_depth_2: int = 1,
    max_states: int = 1024,
    max_extra_length: int = 4,
    bidirectional: bool = True,
) -> Optional[WeakIso]:
    """Search for an isomorphism between any pair of charts `(C_1,
    C_2)` reachable from `alg_1.root` and `alg_2.root` within the
    given depths.

    Returns the first witness found.  Like `find_isomorphism`, the
    witness is a *sufficient* certificate -- absence of a witness
    does not prove non-isomorphism.

    Both algebras must have the same lattice rank.  Pairings need
    not match -- the inner strong-iso step finds a unimodular A
    with `A^T B_2 A = B_1`.
    """
    if alg_1.lattice.rank != alg_2.lattice.rank:
        return None
    pairing_1 = [list(row) for row in alg_1.lattice.pairing]
    pairing_2 = [list(row) for row in alg_2.lattice.pairing]
    # Avoid circular import
    from bps_kalgebra import BPSKAlgebra as _BPSKA

    charts_1 = list(_enumerate_charts(alg_1, max_depth_1))
    charts_2 = list(_enumerate_charts(alg_2, max_depth_2))

    for chain_1, chart_1 in charts_1:
        sub_1 = _BPSKA(
            pairing=pairing_1,
            node_charges=chart_1.nodes,
            spec=chart_1.spec,
            verify="off",
        )
        for chain_2, chart_2 in charts_2:
            sub_2 = _BPSKA(
                pairing=pairing_2,
                node_charges=chart_2.nodes,
                spec=chart_2.spec,
                verify="off",
            )
            inner = find_isomorphism(
                sub_1, sub_2,
                max_states=max_states,
                max_extra_length=max_extra_length,
                bidirectional=bidirectional,
            )
            if inner is not None:
                return WeakIso(
                    chain_1=chain_1, chain_2=chain_2, inner=inner,
                )
    return None
