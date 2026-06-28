"""Spec shortening via local moves on the BPS quiver.

Uses only the local-move primitives (pentagon expand / pentagon collapse
/ commute swap) directly on the ordered list of E_q-factor charges.
This is the "alt spec" machinery of `bps_quiver_tools.alt_spec_with_head`
adapted for spec shortening rather than head-bubbling.

Local moves
-----------

Given adjacent factors `E_q(X_a) E_q(X_b)` in the spec, with pairing
`p = <a, b>` in the lattice:

* `p == 0`  : commute swap        — `[a, b] → [b, a]`     (length unchanged)
* `p == +1` : pentagon expand     — `[a, b] → [b, a+b, a]` (length +1)
* `p == -1` : pentagon collapse   — when the spec contains the triple
              `[b, a+b, a]` with `<a, b> = +1`, replace by `[a, b]`
              (length −1).

`|p| ≥ 2` has no length-bounded local rewrite (the corresponding spec
identity expands to an infinite product). These are "blocking" pairs.

Shortening strategy
-------------------

Pure greedy on pentagon collapses: scan adjacent triples for the
`[b, a+b, a]` pattern (with `<a, b> = +1`); apply the collapse if
found. Each accepted collapse shrinks the spec by 1, so the iteration
terminates. Optionally interleaved with commute swaps to expose new
collapses (currently we use the commute moves implicitly via the scan
order; a more aggressive search would explore the commute orbit).

This is O(|spec| · #collapses) per pass, no BFS — strictly local
moves. Compare to the previous global approach which used
`find_mutation_path` BFS and is now superseded.
"""

from __future__ import annotations

from typing import Sequence

from snf_kernel import int_det as _int_det


def _bracket(g1: Sequence[int], g2: Sequence[int],
             B: Sequence[Sequence[int]]) -> int:
    """Lattice pairing `<g1, g2> = g1^T · B · g2`."""
    n = len(B)
    return sum(g1[a] * B[a][b] * g2[b] for a in range(n) for b in range(n))


def _bracket_via_cache(g1, g2, B, bg_cache, n):
    """O(rank) lattice pairing via a precomputed `B @ b` cache.

    Mirrors the `bg_cache` trick from #115's `_enumerate_local_moves`
    speedup: each unique `g2` charge is mapped to its `B @ g2`
    column-product once (`O(rank^2)`), and the pairing is then
    a single dot product (`O(rank)`).

    At high `n` (and large specs) the saving is dramatic: spec
    shortening at n=12 spends 86% of per-entry time in the
    no-cache `_bracket` pattern.
    """
    Bg2 = bg_cache.get(g2)
    if Bg2 is None:
        Bg2 = tuple(
            sum(B[i][j] * g2[j] for j in range(n)) for i in range(n)
        )
        bg_cache[g2] = Bg2
    return sum(g1[i] * Bg2[i] for i in range(n))


def _find_pentagon_collapse(
    spec: list[tuple], B: Sequence[Sequence[int]],
    bg_cache: dict | None = None,
) -> tuple[int, tuple, tuple] | None:
    """Find a triple `spec[i:i+3] = [b, a+b, a]` with `<a, b> = +1`.

    Returns `(i, a, b)` or `None`.  Optional `bg_cache` (a
    `dict[charge_tuple, B@g_tuple]`) is propagated to the inner
    bracket computation; supply one to amortise `B @ b` across many
    calls (the build pass scales badly at n=12 without this).
    """
    n = len(B)
    if bg_cache is None:
        bracket = lambda a, b: _bracket(a, b, B)
    else:
        bracket = lambda a, b: _bracket_via_cache(a, b, B, bg_cache, n)
    for i in range(len(spec) - 2):
        b = spec[i]
        ab_candidate = spec[i + 1]
        a = spec[i + 2]
        # is spec[i+1] the sum a + b?
        if any(ab_candidate[k] != a[k] + b[k] for k in range(n)):
            continue
        if bracket(a, b) != 1:
            continue
        return (i, a, b)
    return None


def _apply_collapse(spec: list[tuple], i: int,
                    a: tuple, b: tuple) -> list[tuple]:
    """Replace `spec[i:i+3] = [b, a+b, a]` with `[a, b]`."""
    return spec[:i] + [a, b] + spec[i + 3:]


def _nu_forward(
    spec_suffix: Sequence[Sequence[int]],
    gamma: Sequence[int],
    pairing: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """`ν_{S}(α) = α + max(⟨g, α⟩, 0) · g`, iterated for g in
    *REVERSED* spec_suffix.

    The "upper-tropical-charge" mutation. By identity
    `ν_S(γ) = -σ⁻¹(γ)` when S is the full spec; since `σ⁻¹` walks the
    spec in reversed order, the matching forward-style ν must also
    iterate in reverse.

    Bracket sign: `⟨g, α⟩ = -⟨α, g⟩` (antisymmetric pairing), so this
    is `max(-⟨α, g⟩, 0)` -- the dual of the forward `μ_g`.
    """
    n = len(pairing)
    cur = tuple(int(x) for x in gamma)
    for g in reversed(list(spec_suffix)):
        gt = tuple(int(x) for x in g)
        # m = ⟨g, cur⟩ = -⟨cur, g⟩
        m = -sum(cur[a] * pairing[a][b] * gt[b]
                 for a in range(n) for b in range(n))
        if m > 0:
            cur = tuple(c + m * gi for c, gi in zip(cur, gt))
    return cur


def _mutate_node_charges(
    nodes: list[tuple], spec_charge: tuple, pairing: Sequence[Sequence[int]],
) -> list[tuple]:
    """Mutate the node-charges list at the index whose charge equals
    `spec_charge`. FZ tropical mutation:

        γ_k → −γ_k
        γ_j → γ_j + max(⟨γ_j, γ_k⟩, 0) · γ_k    for j ≠ k

    Returns the new charges list, or the original if `spec_charge`
    isn't found (defensive).
    """
    n = len(pairing)
    rank = len(spec_charge)

    def bracket(a, b):
        return sum(a[i] * pairing[i][j] * b[j]
                   for i in range(n) for j in range(n))

    k = None
    for idx, c in enumerate(nodes):
        if c == spec_charge:
            k = idx
            break
    if k is None:
        return nodes
    gk = nodes[k]
    new = []
    for j, gj in enumerate(nodes):
        if j == k:
            new.append(tuple(-x for x in gk))
        else:
            m = bracket(gj, gk)
            shift = max(m, 0)
            new.append(tuple(gj[a] + shift * gk[a] for a in range(rank)))
    return new


def _make_in_cone_predicate(cone_t: list, rank: int):
    """Build a `v -> bool` callable testing whether `v` is a non-negative
    rational combination of `cone_t`.

    For a full-rank cone (`|cone_t| == rank` and `det != 0`) the test
    uses Cramer's rule.  Otherwise (degenerate or overcomplete) it falls
    back to the positive-orthant test, which is the natural semantics
    when cone generators ARE the standard basis.
    """
    from fractions import Fraction
    if len(cone_t) != rank:
        return lambda v: all(x >= 0 for x in v)
    M = [list(row) for row in zip(*cone_t)]   # columns are cone gens
    det_M = _int_det(M)
    if det_M == 0:
        return lambda v: all(x >= 0 for x in v)
    def in_cone(v):
        for i in range(rank):
            Mi = [row[:] for row in M]
            for r in range(rank):
                Mi[r][i] = v[r]
            if Fraction(_int_det(Mi), det_M) < 0:
                return False
        return True
    return in_cone


def _doubly_tropical_bfs(
    lower: Sequence[int],
    upper: Sequence[int],
    cone_gens: Sequence[Sequence[int]],
) -> list[tuple]:
    """BFS-ordered list of cone points γ with `lower ≤_cone γ ≤_cone upper`.

    Shared core of `doubly_tropical_interval_set` and
    `doubly_tropical_interval_size_with_cone`.
    """
    from collections import deque
    rank = len(lower)
    lower_t = tuple(int(x) for x in lower)
    upper_t = tuple(int(x) for x in upper)
    cone_t = [tuple(int(x) for x in g) for g in cone_gens]
    in_cone = _make_in_cone_predicate(cone_t, rank)

    diff = tuple(u - l for u, l in zip(upper_t, lower_t))
    if not in_cone(diff):
        return [lower_t] if lower_t == upper_t else []
    visited: set = {lower_t}
    order: list = [lower_t]
    queue: deque = deque([lower_t])
    while queue:
        cur = queue.popleft()
        for g in cone_t:
            nxt = tuple(c + gi for c, gi in zip(cur, g))
            if nxt in visited:
                continue
            d = tuple(u - n_ for u, n_ in zip(upper_t, nxt))
            if in_cone(d):
                visited.add(nxt)
                order.append(nxt)
                queue.append(nxt)
    return order


def doubly_tropical_interval_set(
    lower: Sequence[int],
    upper: Sequence[int],
    cone_gens: Sequence[Sequence[int]],
) -> list[tuple]:
    """Enumerate the doubly-tropical interval `[lower, upper]` in `cone_gens`
    via cone-BFS from `lower`, returning the cone-BFS-ordered list.

    Useful as the F'-support to drive `solve_F_modified` over.
    """
    return _doubly_tropical_bfs(lower, upper, cone_gens)


def doubly_tropical_l1_in_cone(
    lower: Sequence[int],
    upper: Sequence[int],
    cone_gens: Sequence[Sequence[int]],
):
    """Cheap proxy for interval size: `Σ_i k_i` where
    `upper − lower = Σ_i k_i · c_i` over Q, with `c_i = cone_gens[i]`.

    For a unimodular cone (`|det| = 1`) the `k_i` are non-negative
    integers and this is the L1 distance from `lower` to `upper`
    measured in cone steps; for a general full-rank cone it's a
    rational L1 distance. Strictly monotonic in the size of the
    integer doubly-tropical interval (more cone-steps = bigger
    interval), so it's a sound search-cost proxy that avoids the
    BFS-over-interval blow-up.

    If `(upper − lower)` is not in the cone (i.e., F'-support empty),
    returns +∞. If the cone is degenerate / not full rank, falls back
    to L1 of (upper − lower) coords.
    """
    from fractions import Fraction
    rank = len(lower)
    diff = tuple(int(u) - int(l) for u, l in zip(upper, lower))
    cone_t = [tuple(int(x) for x in g) for g in cone_gens]
    if len(cone_t) != rank:
        return sum(abs(d) for d in diff)
    M = [list(row) for row in zip(*cone_t)]   # columns are cone gens
    det_M = _int_det(M)
    if det_M == 0:
        return sum(abs(d) for d in diff)
    total = Fraction(0)
    for i in range(rank):
        Mi = [row[:] for row in M]
        for r in range(rank):
            Mi[r][i] = diff[r]
        ci = Fraction(_int_det(Mi), det_M)
        if ci < 0:
            return float("inf")
        total += ci
    # +1 for the lower endpoint itself, so a monomial F' (lower==upper)
    # has cost 1 -- consistent with the BFS-based cost.
    return total + 1


def doubly_tropical_interval_size_with_cone(
    lower: Sequence[int],
    upper: Sequence[int],
    cone_gens: Sequence[Sequence[int]],
) -> int:
    """|{γ : lower ≤_cone γ ≤_cone upper}|.  Uses the same cone-BFS
    as `doubly_tropical_interval_set`; just returns the count."""
    return len(_doubly_tropical_bfs(lower, upper, cone_gens))




def _commute_orbit_to_collapse(
    spec: list[tuple],
    B: Sequence[Sequence[int]],
    max_states: int,
    bg_cache: dict | None = None,
) -> list[tuple] | None:
    """BFS over the commute-only orbit of `spec` looking for an
    adjacent state that admits a pentagon collapse.

    Pure local moves: every visited state is reached from `spec` by a
    finite chain of commute swaps (`⟨a, b⟩ = 0` adjacent
    transpositions).  Commutes preserve length, so the final state has
    the same length as `spec`.  When a state with an admissible
    pentagon collapse is found, we apply the collapse and return the
    resulting (length-1) spec.  Returns `None` if no commute-reachable
    state admits a collapse within `max_states` budget.

    This finds collapses that the pure greedy `_find_pentagon_collapse`
    misses because they are blocked by single (or multiple) commute
    swaps -- e.g. `[a, c, b, ...]` with `c = a + b`, `⟨a, b⟩ = 1`,
    `⟨a, c⟩ = 0`: a single swap exposes the collapse pattern that the
    greedy never sees.
    """
    n = len(B)
    if bg_cache is None:
        bg_cache = {}
    start_key = tuple(map(tuple, spec))
    visited = {start_key}
    frontier: list[list[tuple]] = [list(spec)]
    explored = 0
    while frontier and explored < max_states:
        new_frontier: list[list[tuple]] = []
        for s in frontier:
            explored += 1
            if explored > max_states:
                return None
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                # Commutable iff bracket is zero -- cached O(rank)
                # dot product instead of O(rank^2) double sum.
                br = _bracket_via_cache(a, b, B, bg_cache, n)
                if br != 0:
                    continue
                s_swapped = s[:i] + [s[i + 1], s[i]] + s[i + 2:]
                key = tuple(map(tuple, s_swapped))
                if key in visited:
                    continue
                visited.add(key)
                # Does the swapped state admit a pentagon collapse?
                coll = _find_pentagon_collapse(s_swapped, B, bg_cache)
                if coll is not None:
                    ci, ca, cb = coll
                    return _apply_collapse(s_swapped, ci, ca, cb)
                new_frontier.append(s_swapped)
        frontier = new_frontier
    return None


def shorten_spec(
    spec: Sequence[Sequence[int]],
    exchange: Sequence[Sequence[int]],
    *,
    frozen: Sequence[bool] | None = None,
    max_iter: int = 200,
    commute_orbit_states: int = 64,
) -> list[tuple]:
    """Greedily shorten a spec by pentagon-collapsing adjacent triples.

    Local moves only -- no global BFS, no from-scratch generation.
    Each accepted collapse strictly shrinks the spec, so the iteration
    terminates in at most `|spec|` steps.  Two-tier strategy:

    1. **Direct pentagon collapse.**  Scan adjacent triples for the
       `[b, a+b, a]` pattern with `⟨a, b⟩ = 1`; if found, collapse.
    2. **Commute-orbit search** (when no direct collapse exists).
       BFS over commute-only neighbours up to
       `commute_orbit_states` distinct states, looking for one that
       admits a collapse.  This finds collapses behind commute walls
       that the pure greedy misses.  Pure local moves: every state
       in the BFS is reachable from the input by commute swaps.  Set
       `commute_orbit_states=0` to disable (legacy greedy-only path).

    Parameters
    ----------
    spec
        The starting spec, as a list of charge tuples.
    exchange
        The lattice pairing matrix `B` (`<g1, g2> = g1^T B g2`).  Must
        be the same B that defines E_q commutation in the chart.
    frozen
        Boolean flags per node -- accepted for API symmetry; unused
        (frozen nodes affect the BPS quiver, not local-move shortening).
    max_iter
        Safety cap on collapse iterations; collapses always shrink so
        termination is bounded by `|spec|` anyway.
    commute_orbit_states
        Per-iteration commute-BFS budget when no direct collapse is
        available.  64 is a good default for n ≤ 10 (commute orbits
        are small for BPS quivers since most adjacent pairs have
        non-zero bracket).  Set to 0 to disable the commute-search
        tier entirely (recovers the legacy greedy-only behaviour).

    Returns
    -------
    list[tuple]
        A possibly-shorter spec.  Returns a copy of the input if no
        collapse (direct or commute-reachable) is found.
    """
    cur = [tuple(g) for g in spec]
    B = [list(row) for row in exchange]
    # `bg_cache` (charge -> `B @ charge` tuple) is amortised across
    # ALL iterations of the shortening loop (and both inner search
    # passes); each unique charge pays `B @ g` once.  At n=12 this
    # is the difference between 86 % and a few percent of per-entry
    # build time.
    bg_cache: dict = {}
    for _ in range(max_iter):
        coll = _find_pentagon_collapse(cur, B, bg_cache)
        if coll is not None:
            i, a, b = coll
            cur = _apply_collapse(cur, i, a, b)
            continue
        if commute_orbit_states <= 0:
            break
        nxt = _commute_orbit_to_collapse(
            cur, B, commute_orbit_states, bg_cache,
        )
        if nxt is None:
            break
        cur = nxt
    return cur
