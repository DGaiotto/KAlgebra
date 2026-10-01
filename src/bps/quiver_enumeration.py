"""Enumeration and canonical forms of connected BPS quivers.

A BPS quiver here is its exchange matrix `B`: an antisymmetric integer matrix
on the active nodes (standard basis), `B[i][j] > 0` meaning `B[i][j]` arrows
`i → j`.  Two quivers are the same entry of the enumerated dictionary iff they
differ by a permutation of the nodes — the exchange matrix alone is the
identity of an entry, because the crystalline spectrum generator `S` is a
function of it (`bps_factor_spectrum`; `pronilpotent_group_conjecture.md`).

Two primitives, both pure Python and dependency-free:

* :func:`canonical_form` — an isomorphism-invariant canonical exchange
  matrix, found by colour refinement + individualisation: the lex-min matrix
  over the node orderings compatible with the refined colouring (the leaves
  of that search tree with the minimal key are one orbit of `Aut(B)`, which
  is how `aut_order` and the automorphisms come out for free).  It is NOT the
  lex-min over all of `S_n` (that is `spec_chart_space.canonical_Q`, brute
  force to `n ≤ 9`); the two keys differ on most quivers and are both
  canonical.  What the test pins is the property that matters: equal keys
  iff isomorphic, and `aut_order` equal to the brute-force count.
* :func:`enumerate_connected_quivers` — every connected quiver with **total
  arrow weight** `e = Σ_{i<j} |B[i][j]| ≤ E` (arrows counted WITH
  multiplicity; counting distinct edges leaves the cells infinite), up to
  isomorphism, level by level in `e`.  Completeness rests on one observation:
  every connected quiver with `e ≥ 1` arrows is obtained from a connected
  quiver with `e − 1` arrows by one of three moves — raise the multiplicity
  of an existing arrow, add a new arrow between two existing nodes, or attach
  a new leaf node by a single arrow.  (Remove one arrow: a multiple arrow
  decrements; a non-bridge single arrow is the second move in reverse; and if
  every single arrow is a bridge the underlying multigraph is a tree of
  multi-edges, which has a leaf whose arrow is single or a multi-edge to
  decrement.)  Hence a breadth-first walk from the one-node quiver through
  those three moves, canonicalising at each level, reaches every connected
  quiver with `e ≤ E` exactly once.

The set `{connected B : e ≤ E}` is closed under node deletion (`delete_node`),
so RG edges never leave it; mutation (`mutate`) changes `e`, and the per-node
exit weights are exposed for the class index of the dictionary builder.

Cross-check: a probe in the source repository counts the same cells with
an independent implementation (networkx, Weisfeiler–Lehman hash + exact
isomorphism); the two agree cell by cell through `E = 7`
(`tests/test_quiver_enumeration.py`; the census table is in
the design record).
"""

from __future__ import annotations

import heapq
import signal
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Vec = tuple[int, ...]
Matrix = tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# Elementary operations on exchange matrices
# ---------------------------------------------------------------------------


def as_matrix(B: Sequence[Sequence[int]]) -> Matrix:
    """Freeze `B` as a tuple of tuples of ints (validated antisymmetric)."""
    M = tuple(tuple(int(x) for x in row) for row in B)
    n = len(M)
    for i in range(n):
        if len(M[i]) != n:
            raise ValueError("exchange matrix must be square")
        if M[i][i] != 0:
            raise ValueError("exchange matrix must have zero diagonal")
        for j in range(i + 1, n):
            if M[i][j] != -M[j][i]:
                raise ValueError("exchange matrix must be antisymmetric")
    return M


def total_weight(B: Sequence[Sequence[int]]) -> int:
    """`e = Σ_{i<j} |B[i][j]|` — arrows counted with multiplicity."""
    n = len(B)
    return sum(abs(B[i][j]) for i in range(n) for j in range(i + 1, n))


def is_connected(B: Sequence[Sequence[int]]) -> bool:
    n = len(B)
    if n == 0:
        return False
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if B[i][j] != 0 and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == n


def connected_components(B: Sequence[Sequence[int]]) -> list[list[int]]:
    """Node index lists of the connected components, each sorted."""
    n = len(B)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for s in range(n):
        if s in seen:
            continue
        comp = [s]
        seen.add(s)
        stack = [s]
        while stack:
            i = stack.pop()
            for j in range(n):
                if B[i][j] != 0 and j not in seen:
                    seen.add(j)
                    comp.append(j)
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def is_strongly_connected(B: Sequence[Sequence[int]]) -> bool:
    """Every node reaches every other along arrows (arrow `i → j` iff
    `B[i][j] > 0`).  The one-node quiver is strongly connected, the empty one
    is not."""
    n = len(B)
    if n == 0:
        return False
    for forward in (True, False):
        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j not in seen and (B[i][j] > 0 if forward else B[j][i] > 0):
                    seen.add(j)
                    stack.append(j)
        if len(seen) != n:
            return False
    return True


def strongly_connected_components(B: Sequence[Sequence[int]]) -> list[list[int]]:
    """Node index lists of the strongly connected components, each sorted,
    in a SOURCE-FIRST order: every arrow between two components (arrow
    `i → j` iff `B[i][j] > 0`) points from an earlier component to a later one.

    A strongly connected component is a maximal set of nodes each of which
    reaches every other along arrows; the components form a graph with no
    directed cycle, and a source-first order of it is what the composition
    theorems of the design record §1 are stated in:
    the specs of the components, concatenated in this order, form a spec of
    `B`, and `S(B)` is the ordered product of the components' `S`.

    Deterministic: among the components whose predecessors are all placed,
    the one with the smallest node index comes first.  On an acyclic quiver
    every component is a single node and this is `source_sink_order`'s order
    exactly, so `strip_spec` is the all-single-nodes case.  The components are
    found by Tarjan's algorithm (iterative, so no recursion limit), linear in
    the size of `B`.
    """
    n = len(B)
    succ = [[j for j in range(n) if B[i][j] > 0] for i in range(n)]
    index: list[int | None] = [None] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    found: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] is not None:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, 0)]
        while work:
            v, pos = work[-1]
            if pos < len(succ[v]):
                work[-1] = (v, pos + 1)
                w = succ[v][pos]
                if index[w] is None:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                found.append(sorted(comp))
    # The component graph, placed sources first (Kahn), smallest node first.
    comp_of = [0] * n
    for c, comp in enumerate(found):
        for i in comp:
            comp_of[i] = c
    m = len(found)
    later: list[set[int]] = [set() for _ in range(m)]
    indeg = [0] * m
    for i in range(n):
        for j in succ[i]:
            a, b = comp_of[i], comp_of[j]
            if a != b and b not in later[a]:
                later[a].add(b)
                indeg[b] += 1
    avail = [(found[c][0], c) for c in range(m) if indeg[c] == 0]
    heapq.heapify(avail)
    order: list[list[int]] = []
    while avail:
        _, a = heapq.heappop(avail)
        order.append(found[a])
        for b in later[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(avail, (found[b][0], b))
    return order


def is_acyclic(B: Sequence[Sequence[int]]) -> bool:
    """No directed cycle (arrows `i → j` for `B[i][j] > 0`)."""
    n = len(B)
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if B[i][j] > 0:
                indeg[j] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    removed = 0
    while stack:
        i = stack.pop()
        removed += 1
        for j in range(n):
            if B[i][j] > 0:
                indeg[j] -= 1
                if indeg[j] == 0:
                    stack.append(j)
    return removed == n


def source_sink_order(B: Sequence[Sequence[int]]) -> list[int] | None:
    """A topological order of an acyclic quiver, sources first (arrows
    `i → j` for `B[i][j] > 0`), deterministic (smallest index first among
    the available sources); `None` if `B` has a directed cycle."""
    n = len(B)
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if B[i][j] > 0:
                indeg[j] += 1
    avail = sorted(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while avail:
        i = avail.pop(0)
        order.append(i)
        for j in range(n):
            if B[i][j] > 0:
                indeg[j] -= 1
                if indeg[j] == 0:
                    avail.append(j)
        avail.sort()
    return order if len(order) == n else None


def mutate(B: Sequence[Sequence[int]], k: int) -> Matrix:
    """Fomin–Zelevinsky mutation of the exchange matrix at node `k`."""
    n = len(B)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == k or j == k:
                out[i][j] = -B[i][j]
            else:
                bik, bkj = B[i][k], B[k][j]
                out[i][j] = B[i][j] + (abs(bik) * bkj + bik * abs(bkj)) // 2
    return tuple(tuple(r) for r in out)


def delete_node(B: Sequence[Sequence[int]], k: int) -> Matrix:
    """The induced sub-quiver on all nodes but `k` (RG node deletion)."""
    keep = [i for i in range(len(B)) if i != k]
    return induced(B, keep)


def induced(B: Sequence[Sequence[int]], nodes: Sequence[int]) -> Matrix:
    """The induced sub-quiver on `nodes`, in that order."""
    return tuple(tuple(B[i][j] for j in nodes) for i in nodes)


def standard_nodes(n: int) -> list[Vec]:
    return [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalForm:
    """`key[i][j] = B[perm[i]][perm[j]]` is the canonical exchange matrix
    (lex-min over the colour-refinement-compatible orderings — see the
    module doc); `perm` is one witness (canonical node `i`
    is the original node `perm[i]`); `aut_order = |Aut(B)|`; and
    `automorphisms` are the permutations `τ` of the CANONICAL matrix with
    `key[τ[i]][τ[j]] = key[i][j]` (all of them when `aut_order ≤ aut_cap`,
    otherwise the identity alone)."""

    key: Matrix
    perm: Vec
    aut_order: int
    automorphisms: tuple[Vec, ...]
    #: The initial colours in canonical node order, or `None` when the quiver
    #: was canonicalised uncoloured.  Part of the identity of a coloured
    #: entry: two quivers with the same exchange matrix and different colours
    #: are different entries, so a key must carry both.
    colour_key: Vec | None = None


def _refine(B: Sequence[Sequence[int]], colours: list[int]) -> list[int]:
    """Colour refinement to a stable, canonically named colouring.

    A node's signature is its colour plus the sorted multiset of
    `(B[i][j], colour_j)` over the other nodes; signatures are renamed to
    their rank among the distinct signatures, which is isomorphism-invariant,
    so two isomorphic inputs with isomorphic colourings refine to isomorphic
    outputs with the SAME colour names."""
    n = len(B)
    cur = list(colours)
    while True:
        sigs = []
        for i in range(n):
            neigh = sorted((B[i][j], cur[j]) for j in range(n) if j != i)
            sigs.append((cur[i], tuple(neigh)))
        distinct = sorted(set(sigs))
        rank = {s: r for r, s in enumerate(distinct)}
        new = [rank[s] for s in sigs]
        if len(distinct) == len(set(cur)):
            return new
        cur = new


def canonical_form(B: Sequence[Sequence[int]], *, colours: Sequence | None = None,
                   aut_cap: int = 20000,
                   leaf_cap: int = 8_000_000) -> CanonicalForm:
    """Canonical form by individualisation–refinement (see the class doc).

    The search tree individualises, at each internal node, every vertex of
    the first non-singleton colour class; every leaf is a discrete colouring,
    i.e. a permutation, and the leaves whose matrix is the lex-min one form
    one `Aut(B)`-orbit.  No pruning, so the leaf count is `|Aut|` times the
    number of inequivalent leaves; `leaf_cap` guards against pathological
    inputs (raise rather than hang).

    `colours` gives each node a label that an isomorphism must preserve; the
    refinement is seeded with it instead of with the constant colouring, and
    `Aut` is then the colour-preserving automorphism group.  Any sortable
    labels will do — they are ranked, so only their order matters, which keeps
    the naming isomorphism-invariant.  This is what carries a **flavoured**
    quiver (`flavoured_factor_spectrum.FlavouredQuiver`), whose node reps are
    exactly such a label; `colour_key` returns them in canonical order, and an
    entry's identity is the pair `(key, colour_key)` since two quivers may
    share an exchange matrix and differ in their colours.

    **`leaf_cap` is a time/memory guard, not a correctness parameter** — the
    canonical form it computes does not depend on it — and the star quivers
    set its scale, since the leaf count runs with `|Aut|` and a star on `n`
    nodes has `(n−1)!`:

    | star | weight | `\|Aut\|` | time |
    |---|---|---|---|
    | `n = 9` | 8 | 40,320 | 1.1 s |
    | `n = 10` | 9 | 362,880 | 12 s |
    | `n = 11` | 10 | 3,628,800 | 147 s, 1.2 GB |
    | `n = 12` | 11 | 39,916,800 | ≈ 25 min, ≈ 10 GB |

    The default was 2,000,000 until 2026-09-13, which enumerated weight 9 but
    stopped the weight-10 walk dead on the 11-node star (`RuntimeError: leaf
    cap exceeded` after 79 s in a worker).  8,000,000 clears weight 10.
    **Weight 11 needs a different algorithm, not a bigger number**: the
    12-node star's leaves are factorial and the minimal-permutation list that
    collects them is ≈ 10 GB inside one worker.  The fix is the standard
    automorphism pruning — individualise one vertex per orbit of the pointwise
    stabiliser of the path so far, rather than every vertex of the target cell
    — which is why this docstring records the wall rather than the number.
    """
    M = as_matrix(B)
    n = len(M)
    if n == 0:
        return CanonicalForm(key=(), perm=(), aut_order=1, automorphisms=((),),
                             colour_key=() if colours is not None else None)
    if colours is not None:
        if len(colours) != n:
            raise ValueError(
                f"canonical_form: {len(colours)} colours for {n} nodes")
        ranked = {c: i for i, c in enumerate(sorted(set(colours)))}
        init = [ranked[c] for c in colours]
    else:
        init = [0] * n
    best: list | None = [None]  # [key]
    min_perms: list[Vec] = []
    leaves = [0]

    def leaf(colours: list[int]) -> None:
        perm = tuple(sorted(range(n), key=lambda i: colours[i]))
        key = tuple(tuple(M[perm[i]][perm[j]] for j in range(n)) for i in range(n))
        leaves[0] += 1
        if leaves[0] > leaf_cap:
            raise RuntimeError("canonical_form: leaf cap exceeded")
        if best[0] is None or key < best[0]:
            best[0] = key
            min_perms.clear()
            min_perms.append(perm)
        elif key == best[0]:
            min_perms.append(perm)

    def search(colours: list[int]) -> None:
        colours = _refine(M, colours)
        if len(set(colours)) == n:
            leaf(colours)
            return
        # first non-singleton cell, by colour value (canonical choice)
        counts: dict[int, int] = {}
        for c in colours:
            counts[c] = counts.get(c, 0) + 1
        cell_colour = min(c for c, m in counts.items() if m > 1)
        cell = [i for i in range(n) if colours[i] == cell_colour]
        for v in cell:
            # individualise v: v keeps its colour, the rest of its cell moves
            # up by one and every colour above the cell shifts by one
            new = []
            for i in range(n):
                c = colours[i]
                if c > cell_colour:
                    new.append(c + 1)
                elif c == cell_colour and i != v:
                    new.append(c + 1)
                else:
                    new.append(c)
            search(new)

    search(list(init))
    key = best[0]
    p0 = min_perms[0]
    inv0 = [0] * n
    for i, v in enumerate(p0):
        inv0[v] = i
    auts: list[Vec] = []
    seen: set[Vec] = set()
    for q in min_perms:
        tau = tuple(inv0[q[i]] for i in range(n))
        if tau not in seen:
            seen.add(tau)
            auts.append(tau)
    aut_order = len(auts)
    if aut_order > aut_cap:
        auts = [tuple(range(n))]
    return CanonicalForm(
        key=key, perm=p0, aut_order=aut_order,
        automorphisms=tuple(sorted(auts)),
        colour_key=(tuple(init[p0[i]] for i in range(n))
                    if colours is not None else None))


def canonical_key(B: Sequence[Sequence[int]]) -> Matrix:
    return canonical_form(B).key


def relabel_charge(v: Sequence[int], perm: Sequence[int]) -> Vec:
    """A charge in original node coordinates → canonical coordinates, for
    `perm` from :func:`canonical_form` (canonical node `i` = original
    `perm[i]`, so the canonical `i`-th coordinate is the original
    `perm[i]`-th)."""
    return tuple(int(v[p]) for p in perm)


def relabel_spec(spec: Iterable[Sequence[int]], perm: Sequence[int]) -> tuple[Vec, ...]:
    return tuple(relabel_charge(g, perm) for g in spec)


def apply_perm(v: Sequence[int], tau: Sequence[int]) -> Vec:
    """`w[tau[i]] = v[i]` — push a charge through a node permutation."""
    w = [0] * len(v)
    for i, t in enumerate(tau):
        w[t] = int(v[i])
    return tuple(w)


def canonical_spec(spec: Sequence[Sequence[int]],
                   automorphisms: Sequence[Sequence[int]]) -> tuple[Vec, ...]:
    """Lex-min image of a (canonical-coordinate) spec under `Aut(key)`, so
    that specs of one entry are compared up to the entry's own symmetry."""
    spec_t = tuple(tuple(int(x) for x in g) for g in spec)
    best = spec_t
    for tau in automorphisms:
        img = tuple(apply_perm(g, tau) for g in spec_t)
        if img < best:
            best = img
    return best


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuiverRecord:
    """One enumerated connected quiver, in canonical form."""

    key: Matrix
    n: int
    e: int
    aut_order: int
    automorphisms: tuple[Vec, ...]
    acyclic: bool
    exits: Vec          # per node k: the total weight after mutation at k


def _children(M: Matrix, *, max_rank: int | None) -> Iterable[Matrix]:
    n = len(M)
    rows = [list(r) for r in M]
    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j] != 0:
                s = 1 if M[i][j] > 0 else -1
                rows[i][j] += s
                rows[j][i] -= s
                yield tuple(tuple(r) for r in rows)
                rows[i][j] -= s
                rows[j][i] += s
            else:
                for s in (1, -1):
                    rows[i][j] = s
                    rows[j][i] = -s
                    yield tuple(tuple(r) for r in rows)
                rows[i][j] = 0
                rows[j][i] = 0
    if max_rank is None or n < max_rank:
        for i in range(n):
            for s in (1, -1):
                ext = [r + [0] for r in rows] + [[0] * (n + 1)]
                ext[i][n] = s
                ext[n][i] = -s
                yield tuple(tuple(r) for r in ext)


_INTERN: dict[Vec, Vec] = {}


def intern_vec(t: Vec) -> Vec:
    """One shared object per distinct small integer tuple (a matrix row, an
    automorphism, an exit vector).  Rows repeat heavily across the
    enumeration, so sharing them cuts the memory of the records by a
    multiple without changing the key type; tuples compare by value, so
    the interned copy is interchangeable with the original everywhere."""
    r = _INTERN.get(t)
    if r is None:
        r = _INTERN[t] = t
    return r


def compact_record(rec: QuiverRecord) -> QuiverRecord:
    """`rec` with its key rows, automorphisms and exits interned (see
    :func:`intern_vec`); the key is equal to the original as a value."""
    return QuiverRecord(
        key=tuple(intern_vec(row) for row in rec.key), n=rec.n, e=rec.e,
        aut_order=rec.aut_order,
        automorphisms=intern_vec(tuple(intern_vec(a) for a in rec.automorphisms)),
        acyclic=rec.acyclic, exits=intern_vec(rec.exits),
    )


def _record(cf: CanonicalForm) -> QuiverRecord:
    key = cf.key
    n = len(key)
    return QuiverRecord(
        key=key, n=n, e=total_weight(key), aut_order=cf.aut_order,
        automorphisms=cf.automorphisms, acyclic=is_acyclic(key),
        exits=tuple(total_weight(mutate(key, k)) for k in range(n)),
    )


def _level_chunk(task) -> dict:
    """The children of a chunk of one level, canonicalised and recorded, with
    the duplicates inside the chunk removed — the unit of work of the
    parallel enumeration (runs in a forked worker)."""
    chunk, max_rank = task
    found: dict[Matrix, QuiverRecord] = {}
    for M in chunk:
        for C in _children(M, max_rank=max_rank):
            cf = canonical_form(C)
            if cf.key not in found:
                found[cf.key] = _record(cf)
    return found


def enumerate_connected_quivers(
    E: int, *, max_rank: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> dict[Matrix, QuiverRecord]:
    """All connected quivers with total arrow weight `e ≤ E` (and `n ≤
    max_rank` if given), keyed by canonical exchange matrix.  `progress(e,
    count)` is called after each level.  With `workers > 1` each level's
    children are canonicalised in forked workers (`_level_chunk`) and merged
    here in a fixed order, so the result — keys, records and their
    insertion order — is the same as the sequential walk."""
    if E < 0:
        return {}
    one = ((0,),)
    out: dict[Matrix, QuiverRecord] = {one: compact_record(_record(canonical_form(one)))}
    level: list[Matrix] = [one]
    pool = None
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(workers)
    try:
        for e in range(1, E + 1):
            nxt: list[Matrix] = []
            if pool is None or len(level) < 4 * workers:
                for M in level:
                    for C in _children(M, max_rank=max_rank):
                        cf = canonical_form(C)
                        if cf.key in out:
                            continue
                        rec = compact_record(_record(cf))
                        out[rec.key] = rec
                        nxt.append(rec.key)
            else:
                n_chunks = 8 * workers
                size = (len(level) + n_chunks - 1) // n_chunks
                chunks = [(level[i:i + size], max_rank) for i in range(0, len(level), size)]
                for found in pool.map(_level_chunk, chunks):       # in chunk order
                    for key, rec in found.items():
                        if key in out:
                            continue
                        rec = compact_record(rec)
                        out[rec.key] = rec
                        nxt.append(rec.key)
            level = nxt
            if progress is not None:
                progress(e, len(nxt))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def _oriented_cycle(m: int) -> Matrix:
    rows = [[0] * m for _ in range(m)]
    for i in range(m):
        rows[i][(i + 1) % m] = 1
        rows[(i + 1) % m][i] = -1
    return tuple(tuple(r) for r in rows)


def _sc_children(M: Matrix, *, max_weight: int, max_rank: int | None) -> Iterable[Matrix]:
    """The moves of the strongly connected walk on `M`, each child strongly
    connected with total weight `≤ max_weight`: raise a multiplicity; add an
    arrow, either way, between two nodes with none (weight + 1); add an EAR, a
    directed path `u → w_1 → … → w_k → v` through `k ≥ 1` new nodes, `k ≥ 2`
    when `u = v` (weight + k + 1).  Adding arrows cannot break strong
    connectivity, and every node of an ear lies on a path from `u` to `v`."""
    n = len(M)
    e = total_weight(M)
    rows = [list(r) for r in M]
    if e + 1 <= max_weight:
        for i in range(n):
            for j in range(i + 1, n):
                if M[i][j] != 0:
                    s = 1 if M[i][j] > 0 else -1
                    rows[i][j] += s
                    rows[j][i] -= s
                    yield tuple(tuple(r) for r in rows)
                    rows[i][j] -= s
                    rows[j][i] += s
                else:
                    for s in (1, -1):
                        rows[i][j] = s
                        rows[j][i] = -s
                        yield tuple(tuple(r) for r in rows)
                    rows[i][j] = 0
                    rows[j][i] = 0
    k_max = max_weight - e - 1
    if max_rank is not None:
        k_max = min(k_max, max_rank - n)
    for k in range(1, k_max + 1):
        size = n + k
        for u in range(n):
            for v in range(n):
                if u == v and k < 2:
                    continue
                ext = [r + [0] * k for r in rows] + [[0] * size for _ in range(k)]
                path = [u, *range(n, size), v]
                for a, b in zip(path, path[1:]):
                    ext[a][b] = 1
                    ext[b][a] = -1
                yield tuple(tuple(r) for r in ext)


def _sc_level_chunk(task) -> dict:
    """`{key: record}` of the walk's children of a chunk of strongly connected
    quivers, duplicates inside the chunk removed (runs in a forked worker)."""
    chunk, max_weight, max_rank = task
    found: dict[Matrix, QuiverRecord] = {}
    for M in chunk:
        for C in _sc_children(M, max_weight=max_weight, max_rank=max_rank):
            cf = canonical_form(C)
            if cf.key not in found:
                found[cf.key] = _record(cf)
    return found


def enumerate_strongly_connected_quivers(
    E: int, *, max_rank: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> dict[Matrix, QuiverRecord]:
    """Every STRONGLY CONNECTED quiver with total arrow weight `e ≤ E` (and at
    most `max_rank` nodes), up to node permutation, as `{canonical key:
    record}` — the one-node quiver, then weight by weight in the order found.
    The entries of the re-designed dictionary; the walk of
    `strongly_connected_redesign.md` §4, which reaches weights the enumeration
    of all connected quivers cannot.

    Base: the directed cycles of length `≥ 3`.  Moves (`_sc_children`): raise a
    multiplicity, add an arrow, add an ear.  Complete: lowering multiplicities
    turns a strongly connected quiver into its underlying oriented graph,
    which is still strongly connected and has a directed ear decomposition
    starting from a directed cycle; every step of that decomposition, then of
    raising the multiplicities back, is a move, and weights only grow along
    it.  Layers are expanded in weight order, so a layer is complete before
    it is expanded.  With `workers > 1` the children are canonicalised in
    forked workers (`_sc_level_chunk`) and merged in chunk order, so the
    result is the same as the sequential walk's, order included."""
    if E < 0:
        return {}
    one = ((0,),)
    out: dict[Matrix, QuiverRecord] = {one: compact_record(_record(canonical_form(one)))}
    layers: dict[int, list[Matrix]] = {e: [] for e in range(E + 1)}
    for m in range(3, E + 1):
        if max_rank is not None and m > max_rank:
            break
        rec = compact_record(_record(canonical_form(_oriented_cycle(m))))
        out[rec.key] = rec
        layers[m].append(rec.key)
    if progress is not None:
        for e in range(1, min(E, 3) + 1):
            progress(e, len(layers[e]))
    pool = fork_pool(workers) if workers > 1 else None
    try:
        for e in range(3, E):
            level = layers[e]
            if pool is None or len(level) < 4 * workers:
                for M in level:
                    for C in _sc_children(M, max_weight=E, max_rank=max_rank):
                        cf = canonical_form(C)
                        if cf.key in out:
                            continue
                        rec = compact_record(_record(cf))
                        out[rec.key] = rec
                        layers[rec.e].append(rec.key)
            else:
                n_chunks = 8 * workers
                size = (len(level) + n_chunks - 1) // n_chunks
                chunks = [(level[i:i + size], E, max_rank) for i in range(0, len(level), size)]
                for found in pool.map(_sc_level_chunk, chunks):       # in chunk order
                    for key, rec in found.items():
                        if key in out:
                            continue
                        rec = compact_record(rec)
                        out[key] = rec
                        layers[rec.e].append(key)
            if progress is not None:
                progress(e + 1, len(layers[e + 1]))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    # in weight order, as the enumeration of every connected quiver is: the
    # builder's per-entry stages follow this order, so a stage that a deadline
    # interrupts has finished the lower weights first
    return {one: out[one], **{k: out[k] for e in range(1, E + 1) for k in layers[e]}}


def cell_counts(records: Iterable[QuiverRecord]) -> dict[tuple[int, int], tuple[int, int]]:
    """`{(n, e): (quivers, acyclic quivers)}`."""
    out: dict[tuple[int, int], list[int]] = {}
    for r in records:
        c = out.setdefault((r.n, r.e), [0, 0])
        c[0] += 1
        c[1] += int(r.acyclic)
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Forked worker pools that `Pool.terminate()` can always stop
# ---------------------------------------------------------------------------

STOP_SIGNALS = tuple(getattr(signal, _name) for _name in ("SIGTERM", "SIGINT")
                     if hasattr(signal, _name))


def default_stop_signals() -> None:
    """Pool `initializer`: SIGTERM and SIGINT back to their DEFAULT action in a
    forked worker, then unblocked (`fork_pool` forks with them blocked).

    Why this is required, not tidiness.  `Pool.terminate()` — which
    `with Pool(...)` runs on exit — stops the workers by sending them SIGTERM,
    and from then on keeps the task queue's read lock (`_help_stuff_finish`).
    A forked worker inherits the parent's handlers, and a build's parent has
    one that catches SIGTERM (the cooperative stop,
    `EnumeratedBuild.install_stop_handlers`).  A worker that catches the
    signal instead of dying waits on that lock forever, and the parent blocks
    in `wait4` joining it.  Measured 2026-09-21: a SIGTERM to a 61-hour
    E10 build deadlocked exactly this way, parent and all five workers idle at
    0% CPU and nothing written."""
    for sig in STOP_SIGNALS:
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def fork_pool(workers: int, maxtasksperchild: int | None = None):
    """A fork-context `multiprocessing.Pool` whose workers die on SIGTERM and
    SIGINT, however early the signal comes.

    The initializer alone leaves a window: a signal that reaches a worker
    between its fork and the initializer is caught by the inherited handler.
    `terminate()` hits that window when it closes a pool milliseconds after
    forking it — a stage with a few tasks, all done by one worker before the
    scheduler has started another.  Measured 2026-09-23: the builder suite
    hung twice in a row in `test_workers_build_matches_sequential`, the
    parent in `wait4` and one worker, at 0 s CPU, in `sem_wait`.  So the stop
    signals stay BLOCKED across the fork: one that arrives early is held
    pending and delivered, to the default action, the moment the initializer
    unblocks it.  The pool's handler threads are created inside the block and
    keep the mask, so a replacement worker they fork (`maxtasksperchild`)
    starts blocked too.  The parent's own mask is restored at once, and a
    signal it received meanwhile is delivered then (a fork never inherits
    pending signals)."""
    import multiprocessing as mp
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        return mp.get_context("fork").Pool(workers, maxtasksperchild=maxtasksperchild,
                                           initializer=default_stop_signals)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def _mutation_targets_chunk(task) -> list:
    """`[(key, [target key per node inside e ≤ E]), …]` for a chunk of
    records — the unit of work of the parallel class index (forked worker)."""
    chunk, E = task
    out = []
    for key, n, exits in chunk:
        targets = []
        for node in range(n):
            if exits[node] > E:
                continue
            targets.append(canonical_form(mutate(key, node)).key)
        out.append((key, targets))
    return out


def mutation_components(records: dict[Matrix, QuiverRecord], E: int, *,
                        workers: int = 1) -> dict[Matrix, Matrix]:
    """Union-find over the mutation graph restricted to the quivers with
    `e ≤ E` present in `records`.  Returns `{key: representative key}` where
    the representative is the member with minimal `(e, key)` — the
    minimal-weight member with the canonical form as tie-break, "minimal
    within `e ≤ E`"."""
    keys = [k for k, r in records.items() if r.e <= E]
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def edges():
        if workers > 1 and len(keys) >= 4 * workers:
            items = [(k, records[k].n, records[k].exits) for k in keys]
            n_chunks = 8 * workers
            size = (len(items) + n_chunks - 1) // n_chunks
            chunks = [(items[i:i + size], E) for i in range(0, len(items), size)]
            # `with` exits through terminate(): its workers must die on SIGTERM
            with fork_pool(workers) as pool:
                for part in pool.map(_mutation_targets_chunk, chunks):
                    for k, targets in part:
                        for t in targets:
                            yield k, t
        else:
            for k in keys:
                r = records[k]
                for node in range(r.n):
                    if r.exits[node] > E:
                        continue
                    yield k, canonical_form(mutate(k, node)).key

    for k, t in edges():
        if t not in parent:
            raise RuntimeError("mutation target inside e <= E is missing from the enumeration")
        a, b = find(k), find(t)
        if a != b:
            # keep the (e, key)-minimal representative
            if (records[b].e, b) < (records[a].e, a):
                a, b = b, a
            parent[b] = a
    return {k: find(k) for k in keys}


# ---------------------------------------------------------------------------
# The compact entry encoding of the permanent tier (the design record, shard format 2,
# 2026-09-13): a quiver as its ARROW STRING, a green spec as its NODE-INDEX
# STRING.  Node indices are single characters of `ALPHABET` (base 36, so
# n <= 36); an arrow i -> j of multiplicity m is the pair `ALPHABET[i] +
# ALPHABET[j]` repeated m times, pairs listed in the row-major order of the
# upper triangle of the exchange matrix (direction by the sign of b_ij), so
# the string's length is 2e for total arrow weight e and the matrix is
# recovered exactly.  Measured on the shipped weight-8 set: 5.7 gzipped bytes
# per cyclic entry against 11.9 for the matrix + charge-vector form
# ---------------------------------------------------------------------------

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def arrows_string(B: Sequence[Sequence[int]]) -> str:
    """The arrow string of an exchange matrix (see the module note)."""
    n = len(B)
    if n > len(ALPHABET):
        raise ValueError(f"arrows_string: n = {n} exceeds the alphabet ({len(ALPHABET)})")
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            b = int(B[i][j])
            if b > 0:
                out.append((ALPHABET[i] + ALPHABET[j]) * b)
            elif b < 0:
                out.append((ALPHABET[j] + ALPHABET[i]) * (-b))
    return "".join(out)


def matrix_from_arrows(n: int, s: str) -> Matrix:
    """Inverse of `arrows_string` given the node count."""
    if len(s) % 2:
        raise ValueError("matrix_from_arrows: odd-length arrow string")
    B = [[0] * n for _ in range(n)]
    for k in range(0, len(s), 2):
        i, j = _INDEX[s[k]], _INDEX[s[k + 1]]
        if i >= n or j >= n or i == j:
            raise ValueError(f"matrix_from_arrows: bad arrow {s[k:k + 2]!r} for n = {n}")
        B[i][j] += 1
        B[j][i] -= 1
    return tuple(tuple(r) for r in B)


def sequence_string(seq: Sequence[int]) -> str:
    """A node-index sequence as a string of `ALPHABET` characters."""
    return "".join(ALPHABET[int(k)] for k in seq)


def sequence_from_string(s: str) -> list[int]:
    return [_INDEX[c] for c in s]
