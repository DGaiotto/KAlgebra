"""Lazy chart graph for BPSKAlgebra.

A `BPSKAlgebra` represents `A_𝖖[T]` via a *root chart*, but several
algorithms (chart-local F-finding, chart-search multiplication, …)
benefit from working in a *neighbouring chart* and translating back.

This module provides the lazy-graph machinery -- private state on
`BPSKAlgebra`, never user-facing.

## Concepts

* **Chart**: one BPS-quiver presentation of `A_𝖖[T]`.  Carries
  `(nodes, spec)` plus per-chart caches.
* **Mutation**: an edge in the graph.  Goes from a source chart to a
  destination chart via a single necklacing step (forward or inverse)
  at one node, possibly preceded by a local-moves rearrangement of
  the source spec to bring the chosen node-charge to the head (forward)
  or to the tail (inverse).  The local-moves sequence is recorded as
  part of the mutation.
* **ChartGraph**: holds the root chart, a lazy `dict[ChartId, Chart]`,
  and a registry of `Mutation` edges.  Provides `mutate(src, k, dir)`
  which materializes the destination chart on demand.

## Local moves

The only allowed local moves on a spec (within the same chart) are:

* `commute(i)`: swap `spec[i]` and `spec[i+1]` when `⟨spec[i],
  spec[i+1]⟩ = 0`.
* `pent_expand(i)`: replace `(a, b)` at positions `(i, i+1)` with
  `(b, a+b, a)` when `⟨a, b⟩ = +1`.  Spec lengthens by one.
* `pent_collapse(i)`: replace `(b, a+b, a)` at positions `(i, i+1, i+2)`
  with `(a, b)` when `⟨a, b⟩ = +1`.  Spec shortens by one.

These are the pentagon and commute identities on `S` as a quantum-torus
element; they preserve `S` and so don't affect any algebra-level
transport.

## Mutation rule (cluster necklacing)

Given a source chart with spec `s` and a desired node `g`:

* **Forward** (`'fwd'`): require `g` at the head of (a local-move
  rearrangement of) `s`.  Then `necklace`:
  * new spec = `(μ_g(s[1]), …, μ_g(s[N-1]), -g)` where
    `μ_g(α) = α + max(⟨α, g⟩, 0) · g`.
  * new nodes = FZ mutation of source.nodes at the index where
    `nodes[k] == g`.
  * `(l', u') = (μ_g(l), μ_g(u))`.
* **Inverse** (`'inv'`): require `g` at the *tail* of a rearrangement
  of `s`.  Then prepend `-g` and inverse-mutate everything else by
  `-g`.

Validity gate: in either direction, the destination chart's `(u' -
l')` must lie in the destination's positive cone.  If not, the
mutation is rejected.

If the local-moves search to bring `g` to head/tail exhausts its
budget without success, the mutation is rejected (returns `None`).

## Algebra-level transport

When transporting a canonical-basis element `F^{src}` across the
mutation to get `F^{dst}`:

* `'fwd'`: `F^{dst} = _lm_solve(F^{src}, charge)`.
* `'inv'`: `F^{dst} = _lm_solve_inverse(F^{src}, -charge)`.

The local-moves portion of the mutation is purely chart-bookkeeping
and does NOT show up at the algebra level (local moves preserve `S`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from spec_shortening import (
    _mutate_node_charges,
    doubly_tropical_l1_in_cone,
)


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Local-move primitives
# ---------------------------------------------------------------------------

def _bracket(a, b, B):
    n = len(B)
    return sum(a[i] * B[i][j] * b[j] for i in range(n) for j in range(n))


def _bg_compute(b, pairing, rank):
    """`pairing @ b` as a tuple.  Cached by `_enumerate_local_moves`
    via the optional `bg_cache` kwarg -- many adjacent (a, b) pairs
    across a BFS share the same `b`, so this turns the inner
    bracket cost from O(rank^2) into O(rank) per (a, b) pair after
    a one-time O(rank^2) per unique charge."""
    return tuple(
        sum(pairing[i][j] * b[j] for j in range(rank))
        for i in range(rank)
    )


def _apply_local_move(spec, move):
    """Apply one local move to `spec`, returning a new spec list."""
    kind = move[0]
    s = list(spec)
    if kind == "commute":
        i = move[1]
        s[i], s[i + 1] = s[i + 1], s[i]
    elif kind == "pent_expand":
        i = move[1]
        a, b = s[i], s[i + 1]
        ab = tuple(a_ + b_ for a_, b_ in zip(a, b))
        s = s[:i] + [tuple(b), ab, tuple(a)] + s[i + 2:]
    elif kind == "pent_collapse":
        i = move[1]
        b, ab, a = s[i], s[i + 1], s[i + 2]
        s = s[:i] + [tuple(a), tuple(b)] + s[i + 3:]
    else:
        raise ValueError(f"unknown local move {move!r}")
    return s


def _enumerate_local_moves(spec, pairing, allow_pentagon=True, *,
                           bg_cache=None):
    """Generate the legal local moves on `spec`.  Yields tuples like
    `('commute', i)`, `('pent_expand', i)`, `('pent_collapse', i)`.

    When `bg_cache` (a `dict[tuple, tuple]`) is provided, the
    pairing-vector products `B @ g` are memoised across calls, and
    the bracket `⟨a, b⟩` is computed as the O(rank) dot product
    `a · (B @ b)` instead of the O(rank^2) double sum.  This is the
    hot path of the spec-iso BFS, and the cache hit rate is high
    because the same charges appear repeatedly across adjacent
    states in a BFS frontier.
    """
    n = len(spec)
    rank = len(pairing)
    if bg_cache is None:
        for i in range(n - 1):
            a, b = spec[i], spec[i + 1]
            br = _bracket(tuple(a), tuple(b), pairing)
            if br == 0:
                yield ("commute", i)
            elif allow_pentagon and br == 1:
                yield ("pent_expand", i)
        if allow_pentagon:
            for i in range(n - 2):
                b, ab, a = spec[i], spec[i + 1], spec[i + 2]
                if all(ab[k] == a[k] + b[k] for k in range(len(ab))) \
                        and _bracket(tuple(a), tuple(b), pairing) == 1:
                    yield ("pent_collapse", i)
        return
    # Fast path: precompute `B @ b` once per unique charge.
    for i in range(n - 1):
        a, b = spec[i], spec[i + 1]
        b_t = tuple(b)
        Bb = bg_cache.get(b_t)
        if Bb is None:
            Bb = _bg_compute(b_t, pairing, rank)
            bg_cache[b_t] = Bb
        br = 0
        for k in range(rank):
            br += a[k] * Bb[k]
        if br == 0:
            yield ("commute", i)
        elif allow_pentagon and br == 1:
            yield ("pent_expand", i)
    if allow_pentagon:
        for i in range(n - 2):
            b, ab, a = spec[i], spec[i + 1], spec[i + 2]
            ok = True
            for k in range(rank):
                if ab[k] != a[k] + b[k]:
                    ok = False
                    break
            if not ok:
                continue
            b_t = tuple(b)
            Bb = bg_cache.get(b_t)
            if Bb is None:
                Bb = _bg_compute(b_t, pairing, rank)
                bg_cache[b_t] = Bb
            br = 0
            for k in range(rank):
                br += a[k] * Bb[k]
            if br == 1:
                yield ("pent_collapse", i)


def _reconstruct_chain_from_parents(parents, state):
    """Walk parent pointers from `state` back to the BFS root, returning
    the list of moves applied (root -> state)."""
    moves = []
    cur = state
    while parents[cur] is not None:
        parent, move = parents[cur]
        moves.append(move)
        cur = parent
    moves.reverse()
    return moves


def find_local_moves_for_targets(
    spec: Sequence[Sequence[int]],
    head_targets: Sequence[Sequence[int]],
    tail_targets: Sequence[Sequence[int]],
    pairing: Sequence[Sequence[int]],
    *,
    max_states: int = 64,
    max_extra_length: int = 4,
) -> tuple[dict, dict]:
    """Bulk amortised version of `find_local_moves_to_head` /
    `find_local_moves_to_tail`.

    Runs a SINGLE BFS over the local-move orbit of `spec`; at each
    visited state, checks whether `state[0]` matches any pending
    `head_targets` or `state[-1]` matches any pending `tail_targets`,
    and records the BFS-shortest chain to that state.  Terminates
    early when every requested target has been satisfied.

    Returns `(head_chains, tail_chains)`: two dicts mapping
    `target_tuple -> chain_or_None`.  A target absent from the
    returned dict was never requested.  A target mapped to `None`
    means the BFS finished without finding a state with that head /
    tail value within the budget.

    This is the right primitive for "mutation-completing" a chart:
    callers like `ChartGraph.mutate` that loop over `range(n_nodes) x
    {fwd, inv}` previously paid `2n` independent BFSes from the same
    spec; the bulk BFS pays a single shared one with the same per-call
    `max_states` budget.

    Pure local moves: every visited state is reached from `spec` by a
    finite chain of valid moves.  Uses the `bg_cache=` fast path of
    `_enumerate_local_moves` and parent-pointer chain reconstruction
    (vs the per-frontier-entry list carried by the legacy single-
    target finders), so per-state overhead is much lower at large
    `max_states`.
    """
    head_targets_set = {tuple(t) for t in head_targets}
    tail_targets_set = {tuple(t) for t in tail_targets}
    head_chains: dict = {t: None for t in head_targets_set}
    tail_chains: dict = {t: None for t in tail_targets_set}
    pending_head = set(head_targets_set)
    pending_tail = set(tail_targets_set)

    start = tuple(map(tuple, spec))
    base_len = len(start)

    # Initial-state check: spec[0] / spec[-1] may already match.
    if pending_head and start[0] in pending_head:
        head_chains[start[0]] = []
        pending_head.discard(start[0])
    if pending_tail and start[-1] in pending_tail:
        tail_chains[start[-1]] = []
        pending_tail.discard(start[-1])
    if not pending_head and not pending_tail:
        return head_chains, tail_chains

    pairing_l = [list(row) for row in pairing]
    bg_cache: dict = {}
    parents: dict = {start: None}
    frontier = [start]
    visited = 0
    while frontier and visited < max_states:
        new_frontier = []
        done = False
        for s in frontier:
            visited += 1
            if visited > max_states:
                break
            s_list = [list(g) for g in s]
            allow_pent = (len(s) - base_len) < max_extra_length
            for move in _enumerate_local_moves(
                s_list, pairing_l, allow_pent, bg_cache=bg_cache,
            ):
                s_new = _apply_local_move(s_list, move)
                key = tuple(map(tuple, s_new))
                if key in parents:
                    continue
                parents[key] = (s, move)
                head = key[0]
                tail = key[-1]
                hit = False
                if head in pending_head:
                    head_chains[head] = _reconstruct_chain_from_parents(
                        parents, key,
                    )
                    pending_head.discard(head)
                    hit = True
                if tail in pending_tail:
                    tail_chains[tail] = _reconstruct_chain_from_parents(
                        parents, key,
                    )
                    pending_tail.discard(tail)
                    hit = True
                if hit and not pending_head and not pending_tail:
                    done = True
                    break
                new_frontier.append(key)
            if done:
                break
        if done:
            break
        frontier = new_frontier
    return head_chains, tail_chains


def find_local_moves_to_head(
    spec: Sequence[Sequence[int]],
    target: Sequence[int],
    pairing: Sequence[Sequence[int]],
    *,
    max_states: int = 64,
    max_extra_length: int = 4,
) -> list | None:
    """BFS over local-moves orbit of `spec` to find a sequence ending
    in a state with `spec[0] == target`.

    Returns the move sequence (as list of `(kind, position)` tuples), or
    `None` if not reached within `max_states` visited states or
    `max_extra_length` excess pentagon expansions.
    """
    target_t = tuple(target)
    if tuple(spec[0]) == target_t:
        return []
    pairing_l = [list(row) for row in pairing]
    base_len = len(spec)
    seen = {tuple(map(tuple, spec))}
    # frontier: list of (current_spec, moves_so_far)
    frontier = [(list(map(tuple, spec)), [])]
    visited = 0
    while frontier and visited < max_states:
        new_frontier = []
        for s, moves in frontier:
            visited += 1
            if visited > max_states:
                break
            allow_pentagon = (len(s) - base_len) < max_extra_length
            for move in _enumerate_local_moves(s, pairing_l, allow_pentagon):
                s_new = _apply_local_move(s, move)
                key = tuple(map(tuple, s_new))
                if key in seen:
                    continue
                seen.add(key)
                if tuple(s_new[0]) == target_t:
                    return moves + [move]
                new_frontier.append((s_new, moves + [move]))
        frontier = new_frontier
    return None


def find_local_moves_to_tail(
    spec: Sequence[Sequence[int]],
    target: Sequence[int],
    pairing: Sequence[Sequence[int]],
    *,
    max_states: int = 64,
    max_extra_length: int = 4,
) -> list | None:
    """Like `find_local_moves_to_head` but ends with `spec[-1] == target`."""
    target_t = tuple(target)
    if tuple(spec[-1]) == target_t:
        return []
    pairing_l = [list(row) for row in pairing]
    base_len = len(spec)
    seen = {tuple(map(tuple, spec))}
    frontier = [(list(map(tuple, spec)), [])]
    visited = 0
    while frontier and visited < max_states:
        new_frontier = []
        for s, moves in frontier:
            visited += 1
            if visited > max_states:
                break
            allow_pentagon = (len(s) - base_len) < max_extra_length
            for move in _enumerate_local_moves(s, pairing_l, allow_pentagon):
                s_new = _apply_local_move(s, move)
                key = tuple(map(tuple, s_new))
                if key in seen:
                    continue
                seen.add(key)
                if tuple(s_new[-1]) == target_t:
                    return moves + [move]
                new_frontier.append((s_new, moves + [move]))
        frontier = new_frontier
    return None


def replay_local_moves(spec, moves):
    """Apply a sequence of local moves to `spec`, returning the final spec."""
    cur = list(map(tuple, spec))
    for m in moves:
        cur = _apply_local_move(cur, m)
    return cur


# ---------------------------------------------------------------------------
# Necklacing primitives (single-step mutation post-rearrangement)
# ---------------------------------------------------------------------------

def _mu_g(alpha, g, pairing):
    """μ_g(α) = α + max(⟨α, g⟩, 0) · g."""
    m = _bracket(alpha, g, pairing)
    if m > 0:
        return tuple(c + m * gi for c, gi in zip(alpha, g))
    return tuple(alpha)


def _mu_inv_g(alpha, g, pairing):
    """μ⁻¹_g(α) = α − max(⟨α, g⟩, 0) · g."""
    m = _bracket(alpha, g, pairing)
    if m > 0:
        return tuple(c - m * gi for c, gi in zip(alpha, g))
    return tuple(alpha)


def _nu_g(alpha, g, pairing):
    """ν_g(α) = α + max(⟨g, α⟩, 0) · g = α + max(−⟨α, g⟩, 0) · g.

    The "upper-tropical" mutation rule (dual of `_mu_g`).  Node
    charges, lower-tropical charges, and upper-tropical charges have
    *different* mutation rules under cluster mutation:
    * **Node charges** mutate by FZ (one negates, others by `μ_g`).
    * **Lower tropical** charges mutate by `μ_g`.
    * **Upper tropical** charges mutate by `ν_g`.
    """
    n = len(pairing)
    m = -sum(alpha[a] * pairing[a][b] * g[b]
             for a in range(n) for b in range(n))
    if m > 0:
        return tuple(c + m * gi for c, gi in zip(alpha, g))
    return tuple(alpha)


def _nu_inv_g(alpha, g, pairing):
    """ν⁻¹_g(α) = α − max(⟨g, α⟩, 0) · g.  Inverse of `_nu_g`."""
    n = len(pairing)
    m = -sum(alpha[a] * pairing[a][b] * g[b]
             for a in range(n) for b in range(n))
    if m > 0:
        return tuple(c - m * gi for c, gi in zip(alpha, g))
    return tuple(alpha)


def _necklace_forward(spec, nodes, l, u, pairing):
    """Forward necklacing at the head `g = spec[0]`.

    The cluster necklace operation is *just rotation* on the spec:
    remove the head, append `-g` at the tail.  Intermediate spec
    factors are NOT mutated by `μ_g`.

    Other data uses its own rule:
    * `nodes`        → FZ mutation at the node `g`
    * lower trop `l` → `μ_g(l)`
    * upper trop `u` → `ν_g(u)`

    Precondition: caller has arranged spec so that `spec[0]` equals
    one of `nodes`.

    Returns `(new_spec, new_nodes, new_l, new_u, g)`.
    """
    g = tuple(spec[0])
    new_spec = [tuple(s) for s in spec[1:]] + [tuple(-x for x in g)]
    new_nodes = _mutate_node_charges(list(nodes), g, pairing)
    new_l = _mu_g(tuple(l), g, pairing)
    new_u = _nu_g(tuple(u), g, pairing)
    return new_spec, new_nodes, new_l, new_u, g


def _necklace_inverse(spec, nodes, l, u, pairing):
    """Inverse necklacing at the tail `h = spec[-1]`.

    Inverse of `_necklace_forward`: remove the tail, prepend `-h` at
    the head.  Spec rotation only -- no `μ` on the rest.

    Other data:
    * `nodes`        → inverse FZ mutation by `-h`
    * lower trop `l` → `μ⁻¹_{-h}(l)`
    * upper trop `u` → `ν⁻¹_{-h}(u)`

    Precondition: caller has arranged spec so `spec[-1]` is one of
    the nodes.

    Returns `(new_spec, new_nodes, new_l, new_u, h)`.
    """
    h = tuple(spec[-1])
    new_head = tuple(-x for x in h)
    new_spec = [new_head] + [tuple(s) for s in spec[:-1]]
    new_nodes = _inverse_mutate_node_charges(list(nodes), new_head, pairing)
    new_l = _mu_inv_g(tuple(l), new_head, pairing)
    new_u = _nu_inv_g(tuple(u), new_head, pairing)
    return new_spec, new_nodes, new_l, new_u, h


def _inverse_mutate_node_charges(nodes, charge, pairing):
    """Inverse FZ mutation by `charge`: find the node equal to
    `-charge`, flip it to `charge`, inverse-mutate the rest."""
    n = len(pairing)
    rank = len(charge)
    target = tuple(-x for x in charge)
    k = None
    for idx, c in enumerate(nodes):
        if c == target:
            k = idx
            break
    if k is None:
        # `-charge` not in nodes -- caller should not have called this.
        return nodes
    new = []
    for j, gj in enumerate(nodes):
        if j == k:
            new.append(tuple(charge))
        else:
            m = _bracket(gj, charge, pairing)
            shift = max(m, 0)
            new.append(tuple(gj[a] - shift * charge[a] for a in range(rank)))
    return new


# ---------------------------------------------------------------------------
# Data structures: Chart, Mutation, ChartGraph
# ---------------------------------------------------------------------------

ChartId = int


@dataclass
class Mutation:
    """One edge in the chart graph: a single necklacing step at a quiver
    node, optionally preceded by local-moves rearrangement of the
    source spec to bring the desired node-charge to the head (forward)
    or tail (inverse).

    The `local_moves` sequence is part of the mutation's identity: two
    mutations with the same `(direction, charge)` but different
    rearrangements of the source spec yield different destination
    charts in general.

    Algebra-level transport across this edge uses only `(direction,
    charge)`; the local-moves part is chart-bookkeeping (preserving S
    by pentagon / commute identities)."""

    src_id: ChartId
    dst_id: ChartId
    direction: str          # 'fwd' or 'inv'
    charge: Vec             # the consumed head (fwd) or tail (inv) of src spec
    local_moves: list       # list of (kind, position) tuples on src.spec


@dataclass
class Chart:
    """One BPS-quiver presentation of `A_𝖖[T]`.  Identified by
    `chart_id` within a `ChartGraph`; carries the chart-local
    `(nodes, spec)` plus per-chart caches for F (canonical basis at
    this chart's labels) and F·S (`HabiroElement` coefficients).

    `parent_edge` is `None` for the root chart; non-root charts
    record the mutation edge that produced them.

    Strategy-(b) availability tracking:

    * `s1_chain`: ordered list of charges consumed from the root
      spec's head into "S_1" along the chain to this chart.  At root
      `s1_chain == []`.  Each forward-necklace step appends the
      consumed head; each inverse-necklace step pops the last one
      (when un-consuming a previously-pushed head).
    * `boundary_preserved`: True iff every step from root to here
      kept the spec in the form `S_2 ++ [-g for g in s1_chain]`.
      Forward necklacing the actual spec head (no local moves)
      preserves it; local-move rearrangements that bridge the
      `s2_length`-th boundary destroy it.
    * `s2_length`: `len(spec) - len(s1_chain)`, the index of the
      first `-g_i` factor in the chart's spec (when boundary is
      preserved).
    """

    chart_id: ChartId
    nodes: list[Vec]
    spec: list[Vec]
    parent_edge: Optional[Mutation]
    F_cache: dict = field(default_factory=dict)
    FS_cache: dict = field(default_factory=dict)
    s1_chain: list = field(default_factory=list)
    boundary_preserved: bool = True

    @property
    def s2_length(self) -> int:
        return len(self.spec) - len(self.s1_chain)


class ChartGraph:
    """Lazy graph of chart presentations of one `A_𝖖[T]`.

    Built monotonically: charts and edges accumulate as the user (or
    internal algorithms) mutate, never garbage-collected.  Two
    different paths to "the same" chart produce two distinct
    `ChartId`s in this implementation -- canonical hashing is a future
    optimization.
    """

    def __init__(
        self,
        nodes: Sequence[Sequence[int]],
        spec: Sequence[Sequence[int]],
        pairing: Sequence[Sequence[int]],
    ):
        self.pairing: list[list[int]] = [list(row) for row in pairing]
        self._charts: dict[ChartId, Chart] = {}
        self._edges: dict[tuple[ChartId, ChartId], Mutation] = {}
        self._next_id: ChartId = 0
        # Per-source bulk-mutation cache.  Key is
        # `(src_id, max_local_moves)`; value is
        # `dict[(node_index, direction), ChartId | None]` populated
        # eagerly by `_compute_all_mutations` on first call.  Amortises
        # the local-moves BFS across all 2n (node, side) requests from
        # the same source -- the "mutation-completion" pattern.
        self._mutate_cache: dict[tuple, dict[tuple, Optional[ChartId]]] = {}
        self.root_id = self._add_chart(
            nodes=[tuple(n) for n in nodes],
            spec=[tuple(g) for g in spec],
            parent_edge=None,
        )

    # -------------- chart access --------------

    def chart(self, chart_id: ChartId) -> Chart:
        return self._charts[chart_id]

    def root(self) -> Chart:
        return self._charts[self.root_id]

    def __len__(self):
        return len(self._charts)

    # -------------- mutation --------------

    def mutate(
        self,
        src_id: ChartId,
        node_index: int,
        direction: str = "fwd",
        *,
        max_local_moves: int = 64,
    ) -> Optional[ChartId]:
        """Lazily produce the chart reached by mutating the source's
        BPS quiver at `node_index` in `direction`.

        The mutation is admissible iff a local-moves sequence can be
        found within `max_local_moves` BFS states that brings the
        chosen `nodes[node_index]` to the head (forward) or tail
        (inverse) of the spec.  When admissible, the destination chart
        is materialized (or fetched from cache) and a `Mutation` edge
        is recorded.  Returns the destination `ChartId`.

        Returns `None` if the mutation is inadmissible (no local-move
        sequence within budget).

        On the *first* call for a given `(src_id, max_local_moves)`,
        the underlying local-moves BFS is run **once** to populate
        the cache for *all* `2n` `(node_index, direction)` queries
        from the same source -- the "mutation-completion" pattern.
        Subsequent calls with the same source return the cached
        destination directly.
        """
        src = self._charts[src_id]
        if not (0 <= node_index < len(src.nodes)):
            raise IndexError(
                f"node_index {node_index} out of range "
                f"[0, {len(src.nodes)})"
            )
        if direction not in ("fwd", "inv"):
            raise ValueError(
                f"direction must be 'fwd' or 'inv'; got {direction!r}"
            )
        cache_key = (src_id, max_local_moves)
        cache = self._mutate_cache.get(cache_key)
        if cache is None:
            cache = self._compute_all_mutations(src_id, max_local_moves)
            self._mutate_cache[cache_key] = cache
        return cache.get((node_index, direction))

    def _compute_all_mutations(
        self, src_id: ChartId, max_local_moves: int,
    ) -> dict[tuple, Optional[ChartId]]:
        """Run a single bulk-BFS for every `(node_index, direction)`
        from `src_id` and materialise the destination charts.

        Returns `{(node_index, direction): dst_id_or_None}`.  Used
        by `mutate` as a per-source cache fill.
        """
        src = self._charts[src_id]
        # All node charges are candidates for both head and tail.
        head_chains, tail_chains = find_local_moves_for_targets(
            src.spec, src.nodes, src.nodes, self.pairing,
            max_states=max_local_moves,
        )
        result: dict[tuple, Optional[ChartId]] = {}
        for k, target in enumerate(src.nodes):
            target_t = tuple(target)
            for direction, chains in (
                ("fwd", head_chains), ("inv", tail_chains),
            ):
                moves = chains.get(target_t)
                if moves is None:
                    result[(k, direction)] = None
                    continue
                result[(k, direction)] = self._materialise_mutation(
                    src_id, k, direction, moves,
                )
        return result

    def _materialise_mutation(
        self, src_id: ChartId, node_index: int, direction: str,
        moves: list,
    ) -> ChartId:
        """Apply `moves` to `src.spec`, then necklace forward/inverse,
        and create the destination chart with the appropriate edge.

        Factored out of the legacy `mutate` so the bulk-BFS path can
        share the materialisation logic.
        """
        src = self._charts[src_id]
        # Apply local moves, then necklacing.
        rearranged = replay_local_moves(src.spec, moves)
        # We don't track (l, u) per chart -- those are query-specific.
        # Just compute the new spec/nodes for the destination.
        if direction == "fwd":
            new_spec, new_nodes, _, _, charge = _necklace_forward(
                rearranged, src.nodes, (0,) * len(src.nodes[0]),
                (0,) * len(src.nodes[0]), self.pairing,
            )
        else:
            new_spec, new_nodes, _, _, charge = _necklace_inverse(
                rearranged, src.nodes, (0,) * len(src.nodes[0]),
                (0,) * len(src.nodes[0]), self.pairing,
            )

        # Create mutation edge + dst chart
        edge = Mutation(
            src_id=src_id,
            dst_id=-1,  # filled below
            direction=direction,
            charge=tuple(charge),
            local_moves=list(moves),
        )
        # Boundary-preservation tracking.  The "preserved" structure
        # is: chart spec = `S_2 ++ [-g for g in s1_chain]`.
        #
        # * forward necklace at head (no local moves) with `src`
        #   preserved + non-empty S_2: new chart preserves, with
        #   s1_chain extended by `charge`.
        # * inverse necklace at tail (no local moves) with `src`
        #   preserved + non-empty s1_chain + tail == `-s1_chain[-1]`:
        #   new chart preserves, with s1_chain shrunk by one.
        # * any local moves OR non-aligned mutation: preservation is
        #   destroyed (s1_chain frozen at src's value).
        new_s1_chain = list(src.s1_chain)
        new_boundary_preserved = False
        if not moves and src.boundary_preserved:
            if direction == "fwd" and src.s2_length >= 1:
                new_s1_chain = list(src.s1_chain) + [tuple(charge)]
                new_boundary_preserved = True
            elif direction == "inv" and src.s1_chain:
                expected_tail = tuple(-x for x in src.s1_chain[-1])
                if tuple(charge) == expected_tail:
                    new_s1_chain = list(src.s1_chain)[:-1]
                    new_boundary_preserved = True

        dst_id = self._add_chart(
            nodes=new_nodes,
            spec=new_spec,
            parent_edge=edge,
            s1_chain=new_s1_chain,
            boundary_preserved=new_boundary_preserved,
        )
        edge.dst_id = dst_id
        return dst_id

    def mutate_all(
        self,
        src_id: ChartId,
        *,
        max_local_moves: int = 64,
    ) -> dict[tuple, Optional[ChartId]]:
        """Compute every admissible mutation from `src_id` in one shot,
        amortising the local-moves BFS across the `2 · n_nodes`
        `(node_index, direction)` queries.

        Returns `{(node_index, direction): dst_id_or_None}`.  Identical
        to invoking `mutate(src_id, k, direction)` for every
        `(k, direction)` -- shares the cache; subsequent
        `mutate(src_id, k, direction)` calls hit the populated cache.
        """
        cache_key = (src_id, max_local_moves)
        cache = self._mutate_cache.get(cache_key)
        if cache is None:
            cache = self._compute_all_mutations(src_id, max_local_moves)
            self._mutate_cache[cache_key] = cache
        return dict(cache)

    # -------------- internal: chart creation --------------

    def _add_chart(self, nodes, spec, parent_edge,
                   s1_chain=None, boundary_preserved=True):
        chart_id = self._next_id
        self._next_id += 1
        chart = Chart(
            chart_id=chart_id,
            nodes=list(nodes),
            spec=list(spec),
            parent_edge=parent_edge,
            s1_chain=list(s1_chain) if s1_chain is not None else [],
            boundary_preserved=boundary_preserved,
        )
        self._charts[chart_id] = chart
        if parent_edge is not None:
            self._edges[(parent_edge.src_id, chart_id)] = parent_edge
        return chart_id
