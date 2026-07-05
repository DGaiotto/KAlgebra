"""Cluster cone graph builder.

Self-contained: implements the BFS, the tropical-mutation rules, and
the integer-kernel helpers used.  No imports from other algorithm
modules in the repo.

Mathematical content
--------------------

A "cluster" is identified by a cone in the root chart, computed as
follows.  Each chart C carries:
  * `n` node charges `g_1, ..., g_n in Z^rank` (the BPS-quiver
    decoration of the chart);
  * a `rank × rank` antisymmetric integer pairing `B` on the
    ambient lattice Γ.

The chart's local "monomial cone" is the dual cone
    `D(C) = { γ in Γ : <γ, g_i> ≥ 0 for all i }`.
Its primitive integer rays (one per facet) are the chart's
"monomial tropical charges" in C-local coordinates.

To compare cones across charts we transport each ray back to the
root chart by applying the inverse lower-tropical mutation
    `μ_g^{-1}(α) = α - max(<α, g>, 0) · g`
once for each edge of the path root → C, taken in reverse order
(so the most recent edge's charge is applied first).  Two charts
represent the same cluster iff their root-cone ray-sets agree
(mod flavour lineality, when present).

Mutation
--------

We apply the standard BPS-quiver "principal" tropical mutation at
each node: negate the mutated node charge, shift the others by
`max(0, B_{jk}) · charge_k`, and update B by the FZ rule on
indices.  This is NOT involutive on node charges, but the
root-cone-rays it produces ARE invariant under the BFS dedup
(repeated mutation lands on a root cone we've already seen).

Lineality / flavour
-------------------

When `B` restricted to `span(node_charges)` has a non-trivial
kernel, that's the flavour direction `Γ_f`.  Cones genuinely live
in the gauge quotient `Γ_g = Γ / Γ_f`; we pick a section
`sec_basis ⊕ L_basis = Z^rank` and project the root cone via
`decompose_in_basis` to obtain a canonical Γ_g fingerprint.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Graph data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterNode:
    cluster_id: int
    node_charges: list
    base_monomial_rays: tuple        # rays of the cone AT ROOT
    path_from_root: list = field(default_factory=list)  # consumed charges
    spec: Optional[list] = None      # ordered necklace, None if not yet derived
    F_cache: dict = field(default_factory=dict)
    cone_structure_cache: dict = field(default_factory=dict)


@dataclass
class ClusterEdge:
    src_cluster: int
    src_k: int
    charge: tuple
    dst_cluster: int
    dst_k: Optional[int]
    shared_facet: tuple = ()    # (n-1) rays in common with the dst's cone


def _compute_shared_facets(clusters, edges):
    """For each edge, the shared facet between src's and dst's cones
    is the set of rays common to both (in their cone-at-root form).
    Each tropical mutation flips exactly one ray; the other (n-1)
    survive and form the facet.
    """
    rays_by_id = {c.cluster_id: set(c.base_monomial_rays) for c in clusters}
    for e in edges:
        common = rays_by_id[e.src_cluster] & rays_by_id[e.dst_cluster]
        e.shared_facet = tuple(sorted(common))


# ---------------------------------------------------------------------------
# Linear algebra: integer kernel of an (n-1) × n matrix, section split
# ---------------------------------------------------------------------------

def _kernel_1d(rows, n):
    """Primitive integer kernel vector of an (n-1) × n integer matrix.
    Returns None if rank < n - 1.
    """
    if len(rows) != n - 1:
        return None
    A = [[Fraction(r[k]) for k in range(n)] for r in rows]
    pivot_col_of_row: list[int] = []
    col_is_pivot = [False] * n
    r = 0
    for c in range(n):
        if r >= n - 1:
            break
        piv = next((rr for rr in range(r, n - 1) if A[rr][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pivot = A[r][c]
        A[r] = [x / pivot for x in A[r]]
        for rr in range(n - 1):
            if rr != r and A[rr][c] != 0:
                f = A[rr][c]
                A[rr] = [A[rr][k] - f * A[r][k] for k in range(n)]
        pivot_col_of_row.append(c)
        col_is_pivot[c] = True
        r += 1
    if r != n - 1:
        return None
    free_col = next((c for c in range(n) if not col_is_pivot[c]), None)
    if free_col is None:
        return None
    ker = [Fraction(0)] * n
    ker[free_col] = Fraction(1)
    for i, pc in enumerate(pivot_col_of_row):
        ker[pc] = -A[i][free_col]
    # Clear denominators and reduce by gcd
    from functools import reduce
    from math import gcd
    denom_lcm = reduce(lambda a, b: a * b // gcd(a, b),
                       (c.denominator for c in ker), 1)
    scaled = [int(c * denom_lcm) for c in ker]
    g = 0
    for x in scaled:
        if x:
            g = abs(x) if g == 0 else gcd(g, abs(x))
    if g == 0:
        return None
    return tuple(x // g for x in scaled)


def _integer_lattice_split(rows):
    """Given integer `rows` (each of length n), compute an integer
    basis of `L = ker(rows^T)` and a Z-complement `sec_basis` so
    `sec_basis ⊕ L_basis = Z^n`.  Returns `(L_basis, sec_basis)`.

    Implementation: HNF via column operations on the matrix whose
    rows are the `rows`.  Equivalent to a Smith-normal-form split,
    sufficient for the integer kernel.
    """
    # We solve: find integer v in Z^n with sum_j rows[i][j] · v[j] = 0
    # for all rows. Build the columns of `rows`-matrix and reduce.
    if not rows:
        n = 0
    else:
        n = len(rows[0])
    if n == 0:
        return [], []
    # Use HNF (Hermite Normal Form) of the row-matrix to find ker.
    m = len(rows)
    A = [[int(rows[i][j]) for j in range(n)] for i in range(m)]
    # Augment with identity to track column operations
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def col_swap(c1, c2):
        for r in range(m):
            A[r][c1], A[r][c2] = A[r][c2], A[r][c1]
        for r in range(n):
            U[r][c1], U[r][c2] = U[r][c2], U[r][c1]

    def col_add(c_target, c_source, k):
        # c_target += k * c_source
        for r in range(m):
            A[r][c_target] += k * A[r][c_source]
        for r in range(n):
            U[r][c_target] += k * U[r][c_source]

    def col_negate(c):
        for r in range(m):
            A[r][c] = -A[r][c]
        for r in range(n):
            U[r][c] = -U[r][c]

    # HNF via column reduction
    pivot_row = 0
    pivot_col = 0
    while pivot_row < m and pivot_col < n:
        # Find any non-zero in row pivot_row, columns pivot_col..n-1
        nonzero = [c for c in range(pivot_col, n) if A[pivot_row][c] != 0]
        if not nonzero:
            pivot_row += 1
            continue
        # Bring smallest |value| to pivot_col, then reduce
        while True:
            # Find smallest non-zero in row
            cs = [c for c in range(pivot_col, n) if A[pivot_row][c] != 0]
            if not cs:
                break
            cs.sort(key=lambda c: abs(A[pivot_row][c]))
            best = cs[0]
            if best != pivot_col:
                col_swap(pivot_col, best)
            if A[pivot_row][pivot_col] < 0:
                col_negate(pivot_col)
            pivot_val = A[pivot_row][pivot_col]
            # Reduce others
            reduced = False
            for c in range(pivot_col + 1, n):
                if A[pivot_row][c] != 0:
                    q = A[pivot_row][c] // pivot_val
                    if q != 0:
                        col_add(c, pivot_col, -q)
                        reduced = True
            if not reduced:
                break
        pivot_row += 1
        pivot_col += 1

    # After HNF, columns with all-zero in A correspond to kernel
    ker_cols = []
    sec_cols = []
    for c in range(n):
        if all(A[r][c] == 0 for r in range(m)):
            ker_cols.append(c)
        else:
            sec_cols.append(c)

    L_basis = [tuple(U[r][c] for r in range(n)) for c in ker_cols]
    sec_basis = [tuple(U[r][c] for r in range(n)) for c in sec_cols]
    return L_basis, sec_basis


def _decompose(gamma, sec_basis, L_basis):
    """Decompose `gamma ∈ Z^n` into `(s, l)` with `gamma = Σ s_i ·
    sec_basis[i] + Σ l_j · L_basis[j]`.  Both coordinate tuples are
    integer (raises if not).
    """
    n = len(gamma)
    g = len(sec_basis)
    f = len(L_basis)
    if g + f != n:
        raise ValueError("section + kernel dimensions don't sum to rank")
    # Build V whose columns are sec_basis ++ L_basis; solve V x = gamma
    V = [[0] * n for _ in range(n)]
    for j, v in enumerate(sec_basis):
        for i in range(n):
            V[i][j] = v[i]
    for j, v in enumerate(L_basis):
        for i in range(n):
            V[i][g + j] = v[i]
    # Solve V x = gamma via Gauss-Jordan over Fraction
    A = [[Fraction(V[i][j]) for j in range(n)] + [Fraction(gamma[i])]
         for i in range(n)]
    r = 0
    for c in range(n):
        piv = next((rr for rr in range(r, n) if A[rr][c] != 0), None)
        if piv is None:
            raise ValueError("V not invertible -- bad section/kernel basis")
        A[r], A[piv] = A[piv], A[r]
        pivot = A[r][c]
        A[r] = [x / pivot for x in A[r]]
        for rr in range(n):
            if rr != r and A[rr][c] != 0:
                f0 = A[rr][c]
                A[rr] = [A[rr][k] - f0 * A[r][k] for k in range(n + 1)]
        r += 1
    coords = [A[i][n] for i in range(n)]
    if any(c.denominator != 1 for c in coords):
        raise ValueError("decomposition isn't integer")
    return (tuple(int(coords[i]) for i in range(g)),
            tuple(int(coords[g + j]) for j in range(f)))


# ---------------------------------------------------------------------------
# BPS quiver: minimal node-charges + exchange matrix + mutation
# ---------------------------------------------------------------------------

@dataclass
class _Quiver:
    charges: list   # list of length-rank int tuples
    exchange: list  # n×n int matrix  B[i][j] = <g_i, g_j>


def _quiver_from_pairing(charges, pairing):
    rank = len(pairing)
    n = len(charges)
    B = [[sum(pairing[a][b] * charges[i][a] * charges[j][b]
              for a in range(rank) for b in range(rank))
          for j in range(n)] for i in range(n)]
    return _Quiver(charges=[tuple(c) for c in charges], exchange=B)


def _mutate(q, k):
    """Tropical mutation at node k: negate g_k, shift others by
    [B_{jk}]_+ g_k, update exchange by the standard FZ rule on indices.
    """
    n = len(q.charges)
    rank = len(q.charges[0])
    ck = q.charges[k]
    new_charges = []
    for j in range(n):
        if j == k:
            new_charges.append(tuple(-x for x in ck))
        else:
            shift = max(0, q.exchange[j][k])
            new_charges.append(tuple(
                q.charges[j][a] + shift * ck[a] for a in range(rank)
            ))
    new_exchange = [row[:] for row in q.exchange]
    for i in range(n):
        for j in range(n):
            if i == k or j == k:
                new_exchange[i][j] = -q.exchange[i][j] if i != j else 0
            else:
                new_exchange[i][j] = (
                    q.exchange[i][j]
                    + max(0, q.exchange[i][k]) * max(0, q.exchange[k][j])
                    - max(0, -q.exchange[i][k]) * max(0, -q.exchange[k][j])
                )
    return _Quiver(charges=new_charges, exchange=new_exchange)


# ---------------------------------------------------------------------------
# Dual cone (lineality-aware) and tropical pullback
# ---------------------------------------------------------------------------

def _dual_cone_rays(nodes, pairing, sec_basis, n_q):
    """Primitive rays of `{γ : <γ, g> ≥ 0 ∀ g ∈ nodes}`, computed in
    the gauge quotient `Γ_g` and lifted back to `Z^rank` via `sec_basis`.

    When there is no lineality, `sec_basis` is the standard basis and
    `n_q == rank`, so the lift is the identity.
    """
    rank = len(pairing)
    # Project each "wall" Bg_j onto Γ_g coords (sec_basis is dual basis).
    proj_walls = []
    for g in nodes:
        wall = tuple(
            sum(pairing[a][b] * g[b] for b in range(rank))
            for a in range(rank)
        )
        proj_walls.append(tuple(
            sum(wall[a] * sec_basis[i][a] for a in range(rank))
            for i in range(n_q)
        ))
    from itertools import combinations
    if n_q < 1:
        return ()
    rays_set: set = set()
    for sub in combinations(range(len(proj_walls)), n_q - 1):
        ker = _kernel_1d([proj_walls[j] for j in sub], n_q) \
              if n_q >= 2 else (1,)
        if ker is None:
            continue
        lifted = tuple(
            sum(ker[i] * sec_basis[i][a] for i in range(n_q))
            for a in range(rank)
        )
        for sign in (+1, -1):
            cand = tuple(sign * x for x in lifted)
            if not any(x != 0 for x in cand):
                continue
            ok = all(
                sum(cand[a] * pairing[a][b] * g[b]
                    for a in range(rank) for b in range(rank)) >= 0
                for g in nodes
            )
            if ok:
                rays_set.add(cand)
    return tuple(sorted(rays_set))


def _mu_inv(alpha, g, pairing):
    """Lower-tropical inverse mutation: α - max(<α, g>, 0) · g."""
    rank = len(pairing)
    m = sum(alpha[a] * pairing[a][b] * g[b]
            for a in range(rank) for b in range(rank))
    if m > 0:
        return tuple(alpha[a] - m * g[a] for a in range(rank))
    return tuple(alpha)


def _pullback_path(rays, path_charges, pairing):
    """Pull each ray back to root via _mu_inv along the reversed path."""
    out = []
    for r in rays:
        cur = tuple(r)
        for g in reversed(path_charges):
            cur = _mu_inv(cur, g, pairing)
        out.append(cur)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Spec local moves: commute + pentagon
# ---------------------------------------------------------------------------

def _bracket(a, b, pairing):
    rank = len(pairing)
    return sum(a[i] * pairing[i][j] * b[j]
               for i in range(rank) for j in range(rank))


def _enum_local_moves(spec, pairing):
    """Yield legal local moves on `spec`:
       ('commute', i)        : swap spec[i], spec[i+1] when <a, b> = 0.
       ('pent_expand', i)    : [a, b] -> [b, a+b, a] when <a, b> = 1.
       ('pent_collapse', i)  : [b, a+b, a] -> [a, b] (the inverse).
    """
    n = len(spec)
    for i in range(n - 1):
        a, b = spec[i], spec[i + 1]
        br = _bracket(a, b, pairing)
        if br == 0:
            yield ("commute", i)
        elif br == 1:
            yield ("pent_expand", i)
    for i in range(n - 2):
        b, ab, a = spec[i], spec[i + 1], spec[i + 2]
        if all(ab[k] == a[k] + b[k] for k in range(len(ab))):
            if _bracket(a, b, pairing) == 1:
                yield ("pent_collapse", i)


def _apply_local_move(spec, move):
    """Apply one local move; returns a new list."""
    kind, i = move
    s = list(spec)
    if kind == "commute":
        s[i], s[i + 1] = s[i + 1], s[i]
    elif kind == "pent_expand":
        a, b = s[i], s[i + 1]
        ab = tuple(a[k] + b[k] for k in range(len(a)))
        s = s[:i] + [tuple(b), ab, tuple(a)] + s[i + 2:]
    elif kind == "pent_collapse":
        b, ab, a = s[i], s[i + 1], s[i + 2]
        s = s[:i] + [tuple(a), tuple(b)] + s[i + 3:]
    return s


def _local_moves_to_head(start_spec, target_head, pairing, max_states=256):
    """BFS through local moves to find a spec with `target_head` at
    position 0.  Returns the rearranged spec, or None if not reachable
    within `max_states` enumerated specs.
    """
    target = tuple(target_head)
    start_key = tuple(tuple(c) for c in start_spec)
    if start_spec[0] == target:
        return list(start_spec)
    seen = {start_key}
    queue = deque([list(start_spec)])
    while queue:
        if len(seen) > max_states:
            return None
        spec = queue.popleft()
        for move in _enum_local_moves(spec, pairing):
            new_spec = _apply_local_move(spec, move)
            new_key = tuple(tuple(c) for c in new_spec)
            if new_key in seen:
                continue
            seen.add(new_key)
            if new_spec[0] == target:
                return new_spec
            queue.append(new_spec)
    return None


def _head_mutate(spec, charge):
    """Spec head mutation: pop head g, append -g to tail.
    Requires spec[0] == charge.
    """
    assert spec[0] == tuple(charge), \
        f"head_mutate expects head {charge}, got {spec[0]}"
    neg = tuple(-x for x in spec[0])
    return list(spec[1:]) + [neg]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_cluster_graph(
    pairing,
    node_charges,
    *,
    max_charts: int = 5000,
    verbose: bool = False,
    populate_specs: bool = False,
):
    """Build the cluster mutation graph.

    Returns ``(clusters, edges)``:
      ``clusters``: list[ClusterNode] -- one per distinct cone-at-root.
      ``edges``: list[ClusterEdge] -- n outgoing edges per cluster,
        n incoming per cluster (i.e. n-regular for cluster-finite quivers).

    Raises RuntimeError if the BFS exceeds ``max_charts`` clusters.
    """
    pairing = [list(row) for row in pairing]
    rank = len(pairing)
    originals = [tuple(g) for g in node_charges]
    n = len(originals)

    # Lineality of the chart family: ker( <·, span(originals)> ).
    rows = []
    for g in originals:
        rows.append(tuple(
            sum(pairing[i][j] * g[j] for j in range(rank))
            for i in range(rank)
        ))
    while len(rows) < rank:
        rows.append(tuple(0 for _ in range(rank)))
    L_basis, sec_basis = _integer_lattice_split(rows[:rank])
    n_q = rank - len(L_basis)
    if verbose and L_basis:
        print(f"  Lineality: dim L = {len(L_basis)}, gauge rank n_q = {n_q}")

    def fingerprint(rays_in_Z_rank):
        """Canonical Γ_g representative of a cone-ray set."""
        if not L_basis:
            return tuple(sorted(tuple(r) for r in rays_in_Z_rank))
        return tuple(sorted(
            _decompose(r, sec_basis, L_basis)[0] for r in rays_in_Z_rank
        ))

    q0 = _quiver_from_pairing(originals, pairing)
    root_cone = _dual_cone_rays(q0.charges, pairing, sec_basis, n_q)
    root_fp = fingerprint(root_cone)

    seen: dict = {root_fp: 0}
    quivers: dict = {0: q0}
    paths: dict = {0: []}
    clusters: list[ClusterNode] = [ClusterNode(
        cluster_id=0,
        node_charges=list(q0.charges),
        base_monomial_rays=root_cone,
        path_from_root=[],
        spec=list(originals),
    )]
    # Per-cluster spec ordering, computed lazily as we discover edges.
    # If a consumed charge is at the head of the parent's spec, we head-
    # mutate (rotate + negate the head into the tail).  Otherwise we
    # leave spec as None for now (local-move search deferred).
    specs_by_cluster: dict = {0: list(originals)}
    edges: list[ClusterEdge] = []
    next_id = 1
    queue = deque([0])

    while queue:
        if len(clusters) > max_charts:
            raise RuntimeError(
                f"BFS exceeded {max_charts} clusters; mutation orbit "
                f"likely infinite (found {len(clusters)} so far)."
            )
        cid = queue.popleft()
        q = quivers[cid]
        path = paths[cid]
        for k in range(n):
            charge_k = tuple(q.charges[k])
            q_new = _mutate(q, k)
            new_path = path + [charge_k]
            local_rays = _dual_cone_rays(
                q_new.charges, pairing, sec_basis, n_q,
            )
            root_rays = _pullback_path(local_rays, new_path, pairing)
            fp = fingerprint(root_rays)
            if fp in seen:
                dst_cluster = seen[fp]
            else:
                dst_cluster = next_id
                next_id += 1
                seen[fp] = dst_cluster
                quivers[dst_cluster] = q_new
                paths[dst_cluster] = new_path
                # Don't track spec during BFS -- propagate in a separate
                # fixpoint pass after the graph is built (head mutations
                # propagate cheaply; local-move searches only when needed).
                new_spec: Optional[list] = None
                clusters.append(ClusterNode(
                    cluster_id=dst_cluster,
                    node_charges=list(q_new.charges),
                    base_monomial_rays=root_rays,
                    path_from_root=list(new_path),
                    spec=new_spec,
                ))
                queue.append(dst_cluster)
                if verbose:
                    print(f"  cluster {dst_cluster}: NEW rays={root_rays}")
            # Resolve dst_k via -charge if present in dst's nodes
            neg = tuple(-x for x in charge_k)
            dst_q = quivers[dst_cluster]
            dst_k = next(
                (i for i, c in enumerate(dst_q.charges)
                 if tuple(c) == neg),
                None,
            )
            edges.append(ClusterEdge(
                src_cluster=cid, src_k=k, charge=charge_k,
                dst_cluster=dst_cluster, dst_k=dst_k,
            ))
    _compute_shared_facets(clusters, edges)
    if populate_specs:
        _propagate_specs(clusters, edges, pairing)
    return clusters, edges


def build_cluster_graph_truncated(
    pairing,
    node_charges,
    *,
    max_charts: int = 50,
    verbose: bool = False,
):
    """Build the cluster mutation graph, **truncating** at `max_charts`
    instead of raising.  For BPS theories of infinite mutation class
    (Kronecker quivers like pure SU(2), affine A_n^{(1)}, etc.) where
    `build_cluster_graph` would raise, this returns the finite portion
    of the cluster fan reached by BFS within `max_charts` clusters.

    Edges originating from clusters that the BFS dequeued are
    populated; edges originating from clusters still in the queue at
    truncation time are NOT emitted (so the returned (clusters, edges)
    represents a self-consistent finite sub-graph).

    Returns ``(clusters, edges, truncated)`` where ``truncated`` is
    True iff the BFS hit the cap, False if the orbit closed naturally
    within `max_charts`.
    """
    pairing = [list(row) for row in pairing]
    rank = len(pairing)
    originals = [tuple(g) for g in node_charges]
    n = len(originals)
    rows = []
    for g in originals:
        rows.append(tuple(
            sum(pairing[i][j] * g[j] for j in range(rank))
            for i in range(rank)
        ))
    while len(rows) < rank:
        rows.append(tuple(0 for _ in range(rank)))
    L_basis, sec_basis = _integer_lattice_split(rows[:rank])
    n_q = rank - len(L_basis)

    def fingerprint(rays_in_Z_rank):
        if not L_basis:
            return tuple(sorted(tuple(r) for r in rays_in_Z_rank))
        return tuple(sorted(
            _decompose(r, sec_basis, L_basis)[0] for r in rays_in_Z_rank
        ))

    q0 = _quiver_from_pairing(originals, pairing)
    root_cone = _dual_cone_rays(q0.charges, pairing, sec_basis, n_q)
    root_fp = fingerprint(root_cone)
    seen: dict = {root_fp: 0}
    quivers: dict = {0: q0}
    paths: dict = {0: []}
    clusters: list = [ClusterNode(
        cluster_id=0,
        node_charges=list(q0.charges),
        base_monomial_rays=root_cone,
        path_from_root=[],
        spec=list(originals),
    )]
    edges: list = []
    next_id = 1
    queue = deque([0])
    truncated = False

    while queue:
        if len(clusters) >= max_charts:
            truncated = True
            break
        cid = queue.popleft()
        q = quivers[cid]
        path = paths[cid]
        for k in range(n):
            charge_k = tuple(q.charges[k])
            q_new = _mutate(q, k)
            new_path = path + [charge_k]
            local_rays = _dual_cone_rays(
                q_new.charges, pairing, sec_basis, n_q,
            )
            root_rays = _pullback_path(local_rays, new_path, pairing)
            fp = fingerprint(root_rays)
            if fp in seen:
                dst_cluster = seen[fp]
            else:
                if len(clusters) >= max_charts:
                    truncated = True
                    break
                dst_cluster = next_id
                next_id += 1
                seen[fp] = dst_cluster
                quivers[dst_cluster] = q_new
                paths[dst_cluster] = new_path
                clusters.append(ClusterNode(
                    cluster_id=dst_cluster,
                    node_charges=list(q_new.charges),
                    base_monomial_rays=root_rays,
                    path_from_root=list(new_path),
                    spec=None,
                ))
                queue.append(dst_cluster)
                if verbose:
                    print(f"  cluster {dst_cluster}: NEW rays={root_rays}")
            neg = tuple(-x for x in charge_k)
            dst_q = quivers[dst_cluster]
            dst_k = next(
                (i for i, c in enumerate(dst_q.charges)
                 if tuple(c) == neg),
                None,
            )
            edges.append(ClusterEdge(
                src_cluster=cid, src_k=k, charge=charge_k,
                dst_cluster=dst_cluster, dst_k=dst_k,
            ))
        if truncated:
            break
    _compute_shared_facets(clusters, edges)
    return clusters, edges, truncated


def _propagate_specs(clusters, edges, pairing, *, max_local=256, max_passes=10):
    """Fixpoint propagation of specs along cluster mutation edges.

    Starts from root's spec = initial node_charges.  Each pass:
      A. Push head mutations greedily (cheap, exhaustive).
      B. For neighbours still without spec, search for local moves
         (commute + pentagon) to bring the consumed charge to head,
         then head-mutate.
    Repeats until no new specs are found.
    """
    out_by_src: dict = {}
    for e in edges:
        out_by_src.setdefault(e.src_cluster, []).append(e)
    specs: dict = {clusters[0].cluster_id: list(clusters[0].node_charges)}
    for _ in range(max_passes):
        changed = False
        # Pass A: head-mutation propagation (greedy, cheap).
        frontier = list(specs.keys())
        while frontier:
            cid = frontier.pop()
            src_spec = specs[cid]
            for e in out_by_src.get(cid, []):
                if e.dst_cluster in specs:
                    continue
                if tuple(src_spec[0]) == tuple(e.charge):
                    specs[e.dst_cluster] = _head_mutate(src_spec, e.charge)
                    frontier.append(e.dst_cluster)
                    changed = True
        # Pass B: local-move search for unsolved neighbours.
        for cid in list(specs.keys()):
            src_spec = specs[cid]
            for e in out_by_src.get(cid, []):
                if e.dst_cluster in specs:
                    continue
                rearranged = _local_moves_to_head(
                    src_spec, e.charge, pairing, max_states=max_local,
                )
                if rearranged is not None:
                    specs[e.dst_cluster] = _head_mutate(rearranged, e.charge)
                    changed = True
        if not changed:
            break
    for c in clusters:
        c.spec = specs.get(c.cluster_id)


# ---------------------------------------------------------------------------
# Cone finder: given γ, locate the cluster whose cone contains it
# ---------------------------------------------------------------------------

_edge_by_facet_cache: dict = {}


def find_cone(clusters, edges, gamma, *, start_id: int = 0,
              max_steps: int = 1000, sec_basis=None, L_basis=None):
    """Walk the cluster graph to find a cluster whose Γ_g cone contains γ.

    When the algebra has flavour lineality, both γ and the cluster cone
    rays are projected to Γ_g via ``sec_basis`` (with ``L_basis`` as
    the flavour kernel) before the membership test; otherwise the test
    is in Γ directly.
    """
    from snf_kernel import decompose_in_basis

    # Algorithm: at each cluster, decompose γ in the basis of cone rays.
    # If all coefficients ≥ 0, γ is in this cone -- return.  Otherwise
    # pick the ray with the most negative coefficient and walk across the
    # facet opposite that ray.  When flavour lineality is present, the
    # decomposition is done in Γ_g (after sec_basis projection) so a γ
    # that's in the cone modulo lineality is found correctly.

    # Edge index by (src_cluster, sorted shared_facet) -- cached by
    # id(edges) so repeated callers reuse the index instead of rebuilding
    # (E_8 has 200k edges; rebuild was the dominant cost).
    edge_by_facet = _edge_by_facet_cache.get(id(edges))
    if edge_by_facet is None:
        edge_by_facet = {
            (e.src_cluster, tuple(sorted(e.shared_facet))): e
            for e in edges
        }
        _edge_by_facet_cache[id(edges)] = edge_by_facet

    # If there is lineality, work in Γ_g.  The cone's rays are
    # represented in Γ-coords, but γ-membership is only well-defined
    # modulo lineality.  Project both γ and rays to Γ_g via sec_basis.
    has_flav = bool(L_basis)
    if has_flav:
        def project(v):
            sec, _ = decompose_in_basis(v, sec_basis, L_basis)
            return tuple(sec)
        gamma_proj = project(gamma)
        n_g = len(gamma_proj)

    def _scan_flav():
        """Linear scan: find a cluster whose projected cone contains γ
        as a non-negative integer combination of any rank-n_g subset of
        its rays.  Fallback for the flavoured case where the BFS walk
        can stall at partial-rank or empty-rays clusters."""
        from itertools import combinations
        for cid_s, c_s in enumerate(clusters):
            rays_s = c_s.base_monomial_rays
            if not rays_s:
                continue
            rays_p = [project(r) for r in rays_s]
            if len(rays_p) < n_g:
                continue
            for sub in combinations(range(len(rays_p)), n_g):
                basis = [rays_p[i] for i in sub]
                cs = _solve_in_basis(gamma_proj, basis)
                if cs is not None and all(c >= 0 for c in cs):
                    return cid_s
        return None

    cid = start_id
    for _ in range(max_steps):
        cluster = clusters[cid]
        rays = cluster.base_monomial_rays
        if not rays:
            if has_flav:
                # Empty cone: jump to neighbor (any edge).
                edge = next(
                    (e for e in edges if e.src_cluster == cid), None,
                )
                if edge is None:
                    raise RuntimeError(
                        f"cluster {cid} has empty cone and no out-edges"
                    )
                cid = edge.dst_cluster
                continue
            raise RuntimeError(
                f"γ = {gamma} not in span of cluster {cid}'s rays"
            )
        if has_flav:
            rays_proj = [project(r) for r in rays]
            if len(rays_proj) < n_g:
                # Partial-rank cluster: BFS can't decide membership;
                # fall back to a linear scan.
                cid_s = _scan_flav()
                if cid_s is not None:
                    return cid_s
                raise RuntimeError(
                    f"γ = {gamma} not in any cluster's cone (partial-rank scan)"
                )
            coeffs = _solve_in_basis(gamma_proj, rays_proj)
        else:
            coeffs = _solve_in_basis(gamma, list(rays))
        if coeffs is None:
            if has_flav:
                cid_s = _scan_flav()
                if cid_s is not None:
                    return cid_s
            raise RuntimeError(
                f"γ = {gamma} not in span of cluster {cid}'s rays"
            )
        if all(c >= 0 for c in coeffs):
            return cid
        # Pick ray with most-negative coefficient
        bad_idx = min(range(len(coeffs)), key=lambda i: coeffs[i])
        bad_ray = rays[bad_idx]
        wanted_facet = tuple(sorted(r for r in rays if r != bad_ray))
        edge = edge_by_facet.get((cid, wanted_facet))
        if edge is None:
            if has_flav:
                cid_s = _scan_flav()
                if cid_s is not None:
                    return cid_s
            raise RuntimeError(
                f"no edge from cluster {cid} across facet {wanted_facet}"
            )
        cid = edge.dst_cluster
    raise RuntimeError(f"find_cone did not converge in {max_steps} steps")


_inverse_cache: dict = {}


def _compute_inverse_or_pseudo(rays):
    """Compute (inverse_columns, is_square_unimodular) for the basis ``rays``.

    Returns ``(inv, det)`` where ``inv`` is a list of m row-tuples (the
    inverse matrix as rows) such that for square unimodular ``rays``,
    ``coeffs[i] = sum(inv[i][k] * gamma[k] for k in range(n))`` solves
    γ = Σ coeffs[i] · rays[i].  ``det`` is the determinant (Fraction).
    Returns ``None`` when rays do not form a square invertible matrix
    (caller should fall back to the generic Fraction path).
    """
    n = len(rays[0])
    m = len(rays)
    if n != m:
        return None
    # Build augmented (rays-as-columns | I) over Fraction; reduce.
    A = [[Fraction(rays[j][i]) for j in range(m)]
         + [Fraction(1 if k == i else 0) for k in range(n)]
         for i in range(n)]
    det = Fraction(1)
    for col in range(m):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            return None
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det = -det
        pivot = A[col][col]
        det *= pivot
        inv_p = Fraction(1) / pivot
        A[col] = [x * inv_p for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[col][k] for k in range(2 * n)]
    inv_rows = tuple(tuple(A[i][n:]) for i in range(n))
    return (inv_rows, det)


def _solve_in_basis(gamma, rays):
    """Solve γ = Σ a_i · rays[i] over Q.  Returns the rational
    coefficient list (length = len(rays)), or None if γ is not in the
    Q-span of rays (i.e. inconsistent or rays don't span).

    Fast path: square ``rays`` (the common case from cluster cones) —
    cache an inverse matrix per ``rays`` tuple, then each solve is a
    single matrix-vector product over Fraction.
    """
    n = len(gamma)
    m = len(rays)
    if n == m:
        key = tuple(tuple(r) for r in rays)
        cached = _inverse_cache.get(key)
        if cached is None:
            cached = _compute_inverse_or_pseudo(rays)
            _inverse_cache[key] = cached  # may be None (singular)
        if cached is not None:
            inv_rows, det = cached
            g = [Fraction(x) for x in gamma]
            return [sum(inv_rows[i][k] * g[k] for k in range(n))
                    for i in range(m)]
        # singular — fall through
    # Build n x m matrix M with rays as columns, RHS γ.
    A = [[Fraction(rays[j][i]) for j in range(m)] + [Fraction(gamma[i])]
         for i in range(n)]
    row = 0
    pivot_col_of_row: list[int] = []
    for col in range(m):
        piv = next((r for r in range(row, n) if A[r][col] != 0), None)
        if piv is None:
            continue
        A[row], A[piv] = A[piv], A[row]
        pivot = A[row][col]
        A[row] = [x / pivot for x in A[row]]
        for r in range(n):
            if r != row and A[r][col] != 0:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[row][k] for k in range(m + 1)]
        pivot_col_of_row.append(col)
        row += 1
    # Inconsistency check: any non-pivot row with RHS ≠ 0
    for r in range(row, n):
        if A[r][m] != 0:
            return None
    if len(pivot_col_of_row) != m:
        return None  # under-determined; not a simplicial-basis solve
    coeffs = [Fraction(0)] * m
    for r, col in enumerate(pivot_col_of_row):
        coeffs[col] = A[r][m]
    return coeffs


# ---------------------------------------------------------------------------
# F-finding via cones + expand-contract transport
# ---------------------------------------------------------------------------

def compute_F(clusters, edges, pairing, gamma, *,
              sec_basis=None, L_basis=None):
    """Compute the canonical F-polynomial F_γ at the root chart for the
    label γ ∈ Γ.

    Algorithm:
      1. Find a cluster C whose cone-at-root contains γ via `find_cone`.
      2. Transport γ to C's local frame via lower-tropical mutation
         (`_mu_g`) along the consumed-charges path root → C.
      3. At C, F is the single monomial X^{γ_at_C}  (LatticeTorus with
         one entry, coefficient 1).
      4. Transport back to root by applying ``lattice_mutation.solve_inverse``
         for each edge charge in REVERSE path order -- this is the
         expand-contract algorithm.

    Returns a dict {γ_label: LaurentPoly(q)}.
    """
    from lattice import Lattice, LatticeTorus
    from laurent_poly import LaurentPoly
    from lattice_mutation import solve_inverse as _lm_solve_inverse

    cid = find_cone(
        clusters, edges, gamma,
        sec_basis=sec_basis, L_basis=L_basis,
    )
    path = clusters[cid].path_from_root
    gamma_at_C = tuple(gamma)
    for g in path:
        gamma_at_C = _mu_g_step(gamma_at_C, g, pairing)
    lattice = Lattice(pairing)
    F = LatticeTorus(lattice, {gamma_at_C: LaurentPoly({0: 1})})
    for g in reversed(path):
        F = _lm_solve_inverse(F, g)
    return dict(F._terms)


def _mu_g_step(alpha, g, pairing):
    """Lower-tropical forward mutation: α + max(<α, g>, 0) · g."""
    rank = len(pairing)
    m = sum(alpha[a] * pairing[a][b] * g[b]
            for a in range(rank) for b in range(rank))
    if m > 0:
        return tuple(alpha[a] + m * g[a] for a in range(rank))
    return tuple(alpha)


# ---------------------------------------------------------------------------
# Multiplication via cluster routing
# ---------------------------------------------------------------------------

def multiply_in_cone(clusters, edges, pairing, gamma_1, gamma_2):
    """Cluster-monomial product:  L_{γ₁} · L_{γ₂} = q^a · L_{γ₁+γ₂}.

    Algorithm:
      1. γ = γ₁ + γ₂.
      2. Find a cluster C whose cone contains γ via `find_cone`.
      3. Verify γ₁ and γ₂ also live in C's cone (necessary for the
         product to collapse to a single monomial -- the q-skew case).
      4. Decompose each in the basis of C's rays:
            γ₁ = Σ b_i r_i,  γ₂ = Σ c_i r_i,  γ = Σ (b_i + c_i) r_i.
         The q-power from reordering L_{r_i}^{b_i} L_{r_j}^{c_j} into
         canonical position is
            a = Σ_{i<j} b_i · c_j · ⟨r_i, r_j⟩
         (using L_{r_i} · L_{r_j} = q^{⟨r_i, r_j⟩} L_{r_j} · L_{r_i}).

    Returns ``(q_power, gamma_sum, cluster_id)``.  Raises ``ValueError``
    if γ₁ or γ₂ is not in the located cluster's cone (Plücker case --
    requires sum-of-monomials handling, not implemented here).
    """
    rank = len(pairing)
    gamma = tuple(gamma_1[a] + gamma_2[a] for a in range(rank))
    cid = find_cone(clusters, edges, gamma)
    cluster = clusters[cid]
    rays = list(cluster.base_monomial_rays)
    b = _solve_in_basis(gamma_1, rays)
    c = _solve_in_basis(gamma_2, rays)
    if b is None or c is None or any(x < 0 for x in b) or any(x < 0 for x in c):
        raise ValueError(
            f"γ₁={gamma_1} or γ₂={gamma_2} not in cone of cluster {cid} "
            f"(rays={cluster.base_monomial_rays}); Plücker case not handled"
        )
    n = len(rays)
    # a = Σ_{i<j} b_i c_j <r_i, r_j>  (the swap-cost from reordering)
    a = Fraction(0)
    for i in range(n):
        for j in range(i + 1, n):
            pij = sum(
                rays[i][p] * pairing[p][q] * rays[j][q]
                for p in range(rank) for q in range(rank)
            )
            a += b[i] * c[j] * pij
    if a.denominator != 1:
        raise ValueError(f"non-integer q-power {a}; pairing convention mismatch")
    return int(a), gamma, cid


def f_inverse_decompose(clusters, edges, pairing, torus, *,
                         max_iterations: int = 100):
    """Decompose a `LatticeTorus` (F-cache form) into canonical-basis
    `Σ c_γ · L_γ` form, returned as a dict `{γ → LaurentPoly c_γ}`.

    Peel-off algorithm:
      1. Find γ_canon = the lex-min chart γ in `torus._terms` (= min
         first coordinate, then min second).  For BPS K-algebras where
         every canonical F_γ has leading monomial `X^γ` with coef 1,
         this γ_canon is the canonical label of one component.
      2. c = `torus._terms[γ_canon]` (LaurentPoly coefficient).
      3. Subtract `c · F_{γ_canon}` from `torus`.
      4. Repeat until `torus` is empty.

    Raises `RuntimeError` if a γ_canon found by min-lex isn't in the
    cluster fan (= `compute_F` fails on it).  For pure SU(2) this
    happens when the decomposition needs Wilson F's (γ on the
    imaginary root direction), which aren't reachable by cluster
    transport — caller should fall back to a separate Wilson handler.
    """
    from lattice import LatticeTorus, Lattice
    from laurent_poly import LaurentPoly
    lat = torus.lattice
    result: dict = {}
    T = LatticeTorus(lat)
    T._terms = dict(torus._terms)
    iters = 0
    while T._terms:
        iters += 1
        if iters > max_iterations:
            raise RuntimeError(
                f"f_inverse_decompose: exceeded {max_iterations} iterations"
            )
        # lex-min by first coord (then second).
        gamma_canon = min(T._terms.keys())
        coef = T._terms[gamma_canon]
        try:
            F_canon = compute_F(clusters, edges, pairing, gamma_canon)
        except Exception as e:
            raise RuntimeError(
                f"f_inverse_decompose: compute_F failed for γ={gamma_canon}: {e}"
            )
        # Build F_canon as a LatticeTorus.
        F_torus = LatticeTorus(lat)
        for g, c in F_canon.items():
            F_torus._terms[g] = (c if isinstance(c, LaurentPoly)
                                  else LaurentPoly({0: int(c)}))
        # Multiply by coef (scalar LaurentPoly) and subtract.
        scaled = F_torus * coef  # LatticeTorus · LaurentPoly via scalar __mul__
        new_terms = dict(T._terms)
        for g, c in scaled._terms.items():
            cur = new_terms.get(g, LaurentPoly.zero())
            d = cur - c
            if d.is_zero():
                if g in new_terms:
                    del new_terms[g]
            else:
                new_terms[g] = d
        T._terms = new_terms
        result[gamma_canon] = coef
    return result


def multiply_via_transport(clusters, edges, pairing, gamma_1, gamma_2):
    """`L_{γ₁} · L_{γ₂}` via F-via-transport + QT q-skew multiplication.

    Handles BOTH the within-cluster q-commute case AND the cross-cluster
    Plücker case uniformly:
      1. Compute `F_{γ₁}`, `F_{γ₂}` via `compute_F` (lazy transport).
      2. Multiply them as QT Laurent polynomials (q-skew via the ambient
         lattice pairing).
      3. Return the resulting `LatticeTorus` (= F-cache representation
         of the BPS K-algebra product).

    Within-cluster cases reduce to a single QT monomial (matching
    `multiply_in_cone` up to the `cone_label_phase` convention shift).
    Cross-cluster Plücker products come out as multi-term sums in QT,
    encoding the BPS K-algebra Plücker relation.

    This is the cone-based, BPS-free-at-runtime multiplication route
    that works for **any** BPS theory once `compute_F` succeeds — i.e.,
    once the cluster-cone graph contains both `γ₁`'s and `γ₂`'s home
    clusters.  For infinite-mutation theories (pure SU(2), affine
    A_n^{(1)}, etc.), use `build_cluster_graph_truncated` to materialise
    a large-enough portion of the cluster fan.
    """
    from lattice import LatticeTorus, Lattice
    from laurent_poly import LaurentPoly
    F1 = compute_F(clusters, edges, pairing, gamma_1)
    F2 = compute_F(clusters, edges, pairing, gamma_2)
    lat = Lattice(pairing)

    def _as_torus(F):
        t = LatticeTorus(lat)
        for g, c in F.items():
            if isinstance(c, LaurentPoly):
                t._terms[g] = c
            else:
                t._terms[g] = LaurentPoly({0: int(c)})
        return t

    return _as_torus(F1) * _as_torus(F2)


# ---------------------------------------------------------------------------
# FiniteBPSKAlgebra: BPSKAlgebra subclass with cone-routed F-finder
# ---------------------------------------------------------------------------

def _trace_via_monomial_nahm(A_at_C, gamma_at_C, K, *, cone_cutoff=12):
    """Trace at the monomial chart C: direct (n, ñ) enumeration.

    Each pair (n, ñ) with Σ(n−ñ)α = γ_at_C contributes one
    HabiroElement.nahm_term carrying its own (q²;q²)_{n_a} denominators.
    Sum, multiply by (q²;q²)_∞^g, expand.

    Handles pent-expanded specs (where N > rank, so the η→n map is
    many-to-one) automatically -- the (n, ñ) sweep covers every valid
    combination including kernel parametrisations.

    Falls back to the older s_gamma_habiro-based assembly if anything
    goes wrong (e.g. flavoured cases not yet handled).
    """
    spec = list(A_at_C.spec)
    if not spec:
        return None
    # Two implementations available:
    #   * _trace_direct_n_nt: direct (n, ñ) enumeration, conceptually
    #     cleanest; O(K^{2N}) so depends on spec being short.  Needs
    #     spec-shortening BFS (spec_shortening.py) during spec
    #     population to be competitive on pent-expanded specs.
    #   * _trace_via_s_assembly: Schur-index sum over η via
    #     s_gamma_habiro; cluster-cached.
    # For now use the s-assembly: validated 2-3x faster than reference
    # on deep γ, correct on all tested cases.
    return _trace_via_s_assembly(A_at_C, gamma_at_C, K, cone_cutoff)


def _trace_direct_n_nt(A_at_C, gamma_at_C, K):
    """Direct (n, ñ) enumeration trace.

    For each pair (n, ñ) ∈ Z^N_{≥0}² satisfying Σ(n−ñ)α = γ_at_C
    with total q-shift ≤ K, build one
        HabiroElement.nahm_term(
            sign = (−1)^{|n|+|ñ|},
            shift = shift(n) + shift(ñ) + ⟨γ_at_C, η(n)⟩,
            ns = n ++ ñ,
        )
    -- this term carries the elementary (q²;q²)_{n_a} · (q²;q²)_{ñ_a}
    denominator automatically.  Sum all terms, multiply by
    (q²;q²)_∞^g, expand to PowerSeries.

    Works for any spec length N (including pent-expanded N > rank).
    """
    from habiro import HabiroElement
    from qpoch import qpoch_infty
    from bps_kalgebra_internals import _habiro_to_ps
    from zplus_ring import RPowerSeries

    spec = [tuple(s) for s in A_at_C.spec]
    N = len(spec)
    rank = A_at_C.lattice.rank
    pairing = A_at_C.lattice.pairing
    g = A_at_C._gauge_rank
    R = A_at_C._R

    def bracket(a, b):
        return sum(
            a[i] * pairing[i][j] * b[j]
            for i in range(rank) for j in range(rank)
        )

    kmat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            kmat[i][j] = bracket(spec[i], spec[j])

    # Build a list of all n with shift(n) ≤ K, bucketed by η(n).
    # This is enumerate_nahm_buckets but with shift bound on n directly.
    n_by_eta: dict = {}

    def recurse(level, n_partial, shift_partial, eta_partial):
        if shift_partial > K:
            return
        if level == N:
            n_by_eta.setdefault(eta_partial, []).append(
                (tuple(n_partial), shift_partial)
            )
            return
        ai = spec[level]
        v = 0
        while True:
            new_partial = n_partial + [v]
            new_shift = v
            for j in range(level):
                new_shift += v * new_partial[j] * kmat[j][level]
            total_shift = shift_partial + new_shift
            if total_shift > K:
                break
            new_eta = tuple(
                eta_partial[i] + v * ai[i] for i in range(rank)
            )
            recurse(level + 1, new_partial, total_shift, new_eta)
            v += 1

    zero_eta = tuple(0 for _ in range(rank))
    recurse(0, [], 0, zero_eta)

    if not n_by_eta:
        return RPowerSeries(R, {}, K)

    terms = []
    for eta_n, n_list in n_by_eta.items():
        eta_minus = tuple(eta_n[i] - gamma_at_C[i] for i in range(rank))
        nt_list = n_by_eta.get(eta_minus)
        if not nt_list:
            continue
        twist = bracket(gamma_at_C, eta_n)
        for (n, sh_n) in n_list:
            for (nt, sh_nt) in nt_list:
                total = sh_n + sh_nt + twist
                if total > K:
                    continue
                sign = 1 if (sum(n) + sum(nt)) % 2 == 0 else -1
                terms.append(
                    HabiroElement.nahm_term(
                        sign, total, list(n) + list(nt),
                    )
                )

    if not terms:
        return RPowerSeries(R, {}, K)

    result_h = HabiroElement.sum(terms)
    ps = _habiro_to_ps(result_h, K)
    pf = qpoch_infty(K)
    pf_g = pf
    for _ in range(g - 1):
        pf_g = pf_g * pf
    scaled = pf_g * ps

    one_basis = R.basis_element(())
    final = {
        q_exp: q_coeff * one_basis
        for q_exp, q_coeff in scaled._c.items()
        if q_coeff != 0
    }
    return RPowerSeries(R, final, K)


def _trace_via_s_assembly(A_at_C, gamma_at_C, K, cone_cutoff=12):
    """Schur-index sum over η using s_gamma_habiro (via _s_coefficient).
    Bypasses _schur_index but still relies on s_γ caching.  Used as
    fallback when direct enumeration would be too slow.
    """
    from laurent_poly import LaurentPoly
    from habiro import HabiroElement
    from bps_kalgebra_internals import (
        _enumerate_output_charges, _habiro_to_ps,
    )
    from qpoch import qpoch_infty
    from zplus_ring import RPowerSeries

    pairing = A_at_C.lattice.pairing
    rank = A_at_C.lattice.rank
    g = A_at_C._gauge_rank
    R = A_at_C._R
    sec_basis = A_at_C._sec_basis
    ker_basis = A_at_C._ker_basis
    from snf_kernel import decompose_in_basis

    # The "F" for the trace is the monomial X^{gamma_at_C}.
    F_b = {gamma_at_C: LaurentPoly({0: 1})}
    output_charges = _enumerate_output_charges(
        [None, F_b], A_at_C.cone_gens, rank,
        A_at_C._effective_cone_cutoff(None, F_b, K, cone_cutoff),
        cone_witness=A_at_C._cone_witness,
    )
    if not output_charges:
        return RPowerSeries(R, {}, K)

    s_fn = A_at_C._s_coefficient
    overlap_terms: dict = {}  # (sec_c, mu_exp) -> list[HabiroElement]

    for eta in output_charges:
        sec_eta, flav_eta = decompose_in_basis(eta, sec_basis, ker_basis)
        # c_a(η) for F_a = identity is just s_η.
        s_eta = s_fn(eta)
        if s_eta.is_zero():
            continue
        # c_b(η) for F_b = {γ_at_C: 1} = q^{<γ_at_C, η - γ_at_C>} · s_{η - γ_at_C}
        eta_minus = tuple(eta[a] - gamma_at_C[a] for a in range(rank))
        s_eta_minus = s_fn(eta_minus)
        if s_eta_minus.is_zero():
            continue
        twist = A_at_C.lattice.bracket(gamma_at_C, eta_minus)
        # c_b as Habiro: q^{twist} · s_{η - γ_at_C}
        c_b = HabiroElement(
            (LaurentPoly({twist: 1})) * s_eta_minus.numerator,
            dict(s_eta_minus.denom),
        )
        # c_a and c_b are both evaluated at the same lattice point η,
        # so they share sec/flav classes by construction.
        mu_exp = tuple(0 for _ in range(len(flav_eta)))
        prod = s_eta * c_b
        if prod.is_zero():
            continue
        overlap_terms.setdefault(mu_exp, []).append(prod)

    overlap_per_mu = {
        mu: HabiroElement.sum(lst) for mu, lst in overlap_terms.items()
    }
    pf = qpoch_infty(K)
    pf_g = pf
    for _ in range(g - 1):
        pf_g = pf_g * pf
    final_coeffs: dict = {}
    for mu_exp, h in overlap_per_mu.items():
        ps = _habiro_to_ps(h, K)
        scaled = pf_g * ps
        mu_basis_elem = R.basis_element(mu_exp)
        for q_exp, q_coeff in scaled._c.items():
            if q_coeff == 0:
                continue
            term = q_coeff * mu_basis_elem
            if q_exp in final_coeffs:
                final_coeffs[q_exp] = final_coeffs[q_exp] + term
            else:
                final_coeffs[q_exp] = term
    return RPowerSeries(
        R,
        {q: c for q, c in final_coeffs.items() if not c.is_zero()},
        K,
    )


def _build_cone_kalgebra_from_finite(A):
    """Export a FiniteBPSKAlgebra as a self-contained ConeKAlgebra
    structurally analogous to PentagonKAlgebra.

    Returns a ``ConeKAlgebra`` subclass instance with:
      * cone_data() -> FiniteConeData (mult_gens, cones, cocycle,
        cross_product, ray-trace data).
      * coefficient_ring(), identity(), rho(), rho_inverse(),
        _label_section_decompose() -- the KAlgebra contract.
      * _trace_residual(seed_label, K) -- Layer 2 trace.
      * multiply, trace inherited from ConeKAlgebra.
    """
    from cone_data import FiniteConeData
    from cone_kalgebra import ConeKAlgebra
    from zplus_ring import TrivialZPlusRing, AbelianZPlusRing
    from laurent_poly import LaurentPoly
    from snf_kernel import decompose_in_basis

    import time as _time
    _t = _time.time()
    def _log(msg):
        nonlocal _t
        print(f"    [{_time.time()-_t:6.1f}s] {msg}", flush=True)
        _t = _time.time()

    _log("starting wrapper")
    # 1. Collect all mult-gens.  For flavoured quivers (L_basis non-empty),
    # we identify rays modulo the flavour lineality: two Γ-rays whose
    # Γ_g projections coincide represent the same canonical-basis class
    # up to a μ-character.  We pick the SHORTEST (= "canonical") lift per
    # Γ_g class; the per-cluster lift information is kept in ray_to_idx
    # so cones know which mult-gen index each ray maps to.
    # For flavored quivers, identify mult-gens at Γ_g (gauge sublattice)
    # level: two Γ-rays differing by ker B are the same canonical-basis
    # class up to a μ-character.  Pick the smallest Γ-lift as canonical.
    # ρ-propagation tracks the section-offset δ via _rho_propagate_term.
    _has_flav_id = bool(A._fbk_L_basis)
    if _has_flav_id:
        # σ-anti axis detection: if there is a w-pair, identify which
        # L-basis vector is the σ-anti direction (= a unit vector e_k
        # negated by the node-swap permutation).  Prefer canonical-lift
        # reps with that gauge coord = 0 so the basis is σ-invariant.
        _sigma_anti_axes: list = []
        if A._fbk_w_pairs:
            for v in A._fbk_L_basis:
                nz = [i for i, x in enumerate(v) if x != 0]
                if len(nz) == 1 and abs(v[nz[0]]) == 1:
                    _sigma_anti_axes.append(nz[0])

        def _canon_rank(rt):
            sigma_score = sum(abs(rt[ax]) for ax in _sigma_anti_axes)
            return (sigma_score, sum(abs(x) for x in rt))

        _seen_proj: dict = {}
        for c in A._fbk_clusters:
            for r in c.base_monomial_rays:
                r_t = tuple(r)
                proj, _flav = decompose_in_basis(
                    r_t, A._fbk_sec_basis, A._fbk_L_basis,
                )
                key = tuple(proj)
                cur = _seen_proj.get(key)
                if cur is None or _canon_rank(r_t) < _canon_rank(cur):
                    _seen_proj[key] = r_t
        mult_gens_list = sorted(_seen_proj.values())
    else:
        all_rays_set: set = set()
        for c in A._fbk_clusters:
            for r in c.base_monomial_rays:
                all_rays_set.add(tuple(r))
        mult_gens_list = sorted(all_rays_set)
    n_mg = len(mult_gens_list)

    # --- ρ-orbit completion of the mult-gens (smallest-member fix) -------
    # The cluster BFS yields the *gauge* cone rays.  For the family minimum
    # (a1d4 / D_4, gauge rank 2) these are not ρ-closed: ρ maps a gauge ray
    # to its σ-orbit partners, which carry flavour and lie outside the gauge
    # cone, so ρ cannot be a generator permutation and the construction
    # truncates.  Larger members (D_6/D_8/…) self-close, so this loop adds
    # nothing for them.  ρ is an algebra automorphism, so completing the
    # generators (and the cones + cross-products, below) under ρ yields the
    # honest ρ-invariant presentation.  Identification is at the Γ_g (gauge)
    # level, matching the BFS rays.
    def _mg_gauge_key(r):
        if _has_flav_id:
            return tuple(decompose_in_basis(
                r, A._fbk_sec_basis, A._fbk_L_basis)[0])
        return tuple(r)
    _n_before = len(mult_gens_list)
    _present_keys = {_mg_gauge_key(g) for g in mult_gens_list}
    _grew = True
    while _grew:
        _grew = False
        for g in list(mult_gens_list):
            for img in (tuple(A.rho(g)), tuple(A.rho_inverse(g))):
                k = _mg_gauge_key(img)
                if k not in _present_keys:
                    _present_keys.add(k)
                    mult_gens_list.append(img)
                    _grew = True
    mult_gens_list = sorted(set(mult_gens_list))
    n_mg = len(mult_gens_list)
    _did_complete = (n_mg > _n_before)

    # ray_to_idx: any Γ-ray seen in a cluster maps to its Γ_g class's
    # canonical-lift index.  For non-flavoured case this is just the
    # identity-on-the-set.
    ray_to_idx: dict = {r: i for i, r in enumerate(mult_gens_list)}
    if _has_flav_id:
        _canon_lift = {tuple(decompose_in_basis(r, A._fbk_sec_basis, A._fbk_L_basis)[0]): r
                       for r in mult_gens_list}
        for c in A._fbk_clusters:
            for r in c.base_monomial_rays:
                r_t = tuple(r)
                if r_t not in ray_to_idx:
                    proj, _ = decompose_in_basis(
                        r_t, A._fbk_sec_basis, A._fbk_L_basis,
                    )
                    canon = _canon_lift[tuple(proj)]
                    ray_to_idx[r_t] = ray_to_idx[canon]

    # 2. Cones (= cluster ray-sets, as frozensets of mult-gen indices).
    cones_list: list = []
    seen_cones: set = set()
    for c in A._fbk_clusters:
        if not c.base_monomial_rays:
            continue
        cone_idx_set = frozenset(
            ray_to_idx[tuple(r)] for r in c.base_monomial_rays
        )
        if cone_idx_set in seen_cones:
            continue
        seen_cones.add(cone_idx_set)
        cones_list.append(cone_idx_set)

    # ρ-complete the cones (see the ρ-orbit completion of the mult-gens
    # above).  Since ρ is an algebra automorphism, the ρ-image of a cone is
    # a cone; close cones_list under the gen-index ρ-permutation.  No-op
    # when nothing was completed.
    if _did_complete:
        _gkey_to_idx: dict = {}
        for _i, _g in enumerate(mult_gens_list):
            _gkey_to_idx.setdefault(_mg_gauge_key(_g), _i)

        def _gen_rho_idx(i):
            return _gkey_to_idx.get(
                _mg_gauge_key(tuple(A.rho(mult_gens_list[i]))))

        _cone_set = set(cones_list)
        _grew2 = True
        while _grew2:
            _grew2 = False
            for cone in list(_cone_set):
                img_ids = [_gen_rho_idx(i) for i in cone]
                if any(x is None for x in img_ids):
                    continue
                img = frozenset(img_ids)
                if img not in _cone_set:
                    _cone_set.add(img)
                    _grew2 = True
        cones_list = sorted(_cone_set, key=lambda s: (len(s), sorted(s)))

    _log(f"step 1+2: {n_mg} mult-gens, {len(cones_list)} cones")
    pairing = A._fbk_pairing
    rank = len(pairing)

    def bracket(a, b):
        return sum(
            a[i] * pairing[i][j] * b[j]
            for i in range(rank) for j in range(rank)
        )

    # 3. Cocycle table -- direct from the lattice bracket on cone rays.
    # Cone rays are honest lattice vectors, so for g, h sharing a cone
    # the q-commutation is L_g · L_h = q^{⟨g, h⟩} L_{g+h}.  No need to
    # run A.multiply (which would force F-finder calls at E_8 scale).
    cocycle_table: dict = {}
    for cone_fs in cones_list:
        cone_ids = sorted(cone_fs)
        for i_g in cone_ids:
            for i_h in cone_ids:
                if i_g == i_h:
                    continue
                if (i_g, i_h) in cocycle_table:
                    continue
                cocycle_table[(i_g, i_h)] = bracket(
                    mult_gens_list[i_g], mult_gens_list[i_h],
                )

    _log(f"step 3: cocycle table {len(cocycle_table)} entries")
    # 4. ρ-permutation on mult-gens (via parent.rho on the underlying γ).
    # For flavoured quivers, sections are not ρ-covariant: parent.rho
    # of a canonical lift may land on a different lift of the Γ_g
    # class.  We extend ρ-perm via the canon-lift map and track the
    # per-index flavor offset δ_ρ(i) so cross-product propagation can
    # compensate with μ-shifts.
    rho_perm: dict = {}
    rho_inv_perm: dict = {}
    rho_delta: dict = {}     # i → flavor offset (tuple in L-basis coords)
    for i, ray in enumerate(mult_gens_list):
        rho_image = tuple(A.rho(ray))
        delta = None
        j = None
        if rho_image in ray_to_idx:
            j = ray_to_idx[rho_image]
            if _has_flav_id:
                # ray_to_idx may have been extended to map non-canonical
                # lifts to their canonical Γ_g class.  Compute δ as the
                # offset of rho_image from canonical_lift[j].
                canon_of_j = mult_gens_list[j]
                _, flav_image = decompose_in_basis(
                    rho_image, A._fbk_sec_basis, A._fbk_L_basis,
                )
                _, flav_canon = decompose_in_basis(
                    canon_of_j, A._fbk_sec_basis, A._fbk_L_basis,
                )
                delta = tuple(int(a) - int(b)
                              for a, b in zip(flav_image, flav_canon))
            else:
                delta = ()
        elif _has_flav_id:
            try:
                proj, flav = decompose_in_basis(
                    rho_image, A._fbk_sec_basis, A._fbk_L_basis,
                )
                proj_t = tuple(proj)
                if proj_t in _canon_lift:
                    canon = _canon_lift[proj_t]
                    j = ray_to_idx[canon]
                    _, flav_canon = decompose_in_basis(
                        canon, A._fbk_sec_basis, A._fbk_L_basis,
                    )
                    delta = tuple(
                        int(a) - int(b) for a, b in zip(flav, flav_canon)
                    )
                else:
                    continue
            except Exception:
                continue
        else:
            continue
        rho_perm[i] = j
        rho_inv_perm[j] = i
        if delta is not None:
            rho_delta[i] = delta

    _log(f"step 4: rho perm {len(rho_perm)} entries")
    # 5. Coefficient ring: flavour kernel dim -> AbelianZPlusRing.
    L_basis = A._fbk_L_basis
    if L_basis:
        R = AbelianZPlusRing(rank=len(L_basis))
    else:
        R = TrivialZPlusRing()

    # Coefficient-type helpers.  In the flavoured case, coefficients are
    # RLaurent over the AbelianZPlusRing (q-Laurent with μ-monomials);
    # otherwise they are LaurentPoly (q-Laurent over Z).
    from zplus_ring import RElement, RLaurent
    _has_flav_coeffs = bool(L_basis)

    def _lp_to_R(lp_coeffs):
        """Convert a LaurentPoly-style dict {q_exp: int_coeff} to the
        coefficient_ring's poly type.  In trivial case returns a
        LaurentPoly; in flavoured case returns an RLaurent with the
        same q-exponents and trivial μ-character."""
        if _has_flav_coeffs:
            return RLaurent(R, {e: c for e, c in lp_coeffs.items() if c != 0})
        from laurent_poly import LaurentPoly as _LP_
        return _LP_(dict(lp_coeffs))

    def _mu_monomial(flav_exp_tuple, q_exp=0):
        """μ^{flav_exp_tuple} · q^{q_exp} as an RLaurent."""
        if not _has_flav_coeffs:
            from laurent_poly import LaurentPoly as _LP_
            return _LP_({q_exp: 1})
        char = tuple(int(x) for x in flav_exp_tuple)
        return RLaurent(R, {q_exp: RElement(R, {char: 1})})

    def _coeff_one():
        if _has_flav_coeffs:
            return RLaurent(R, {0: 1})
        from laurent_poly import LaurentPoly as _LP_
        return _LP_({0: 1})

    # 6. Cross-product table -- orbit-reduced under ρ.
    # For E_6 (Z_14 on 42² = 1764 pairs): only 126 orbit reps to
    # compute via parent.multiply; the rest fill in by ρ-applying
    # the result words.
    cross_cache: dict = {}

    # Pre-build edge index for fast find_cone (used per delta below).
    _edge_by_facet = {
        (e.src_cluster, tuple(sorted(e.shared_facet))): e
        for e in A._fbk_edges
    }
    _clusters_list = A._fbk_clusters
    _sec_basis = A._fbk_sec_basis
    _L_basis = A._fbk_L_basis
    _has_flav = bool(_L_basis)

    def _project(v):
        sec, _ = decompose_in_basis(v, _sec_basis, _L_basis)
        return tuple(sec)

    # Pre-pick a non-empty-rays cluster as default start (flavored
    # quivers can have cluster 0 with empty rays at the root chart).
    _nonempty_start = next(
        (cid for cid, c in enumerate(_clusters_list)
         if c.base_monomial_rays), 0,
    )

    # Precompute projected rays per cluster (avoids re-projecting in
    # every scan).  Also cache γ → containing-cluster results.
    _cluster_rays_proj_cache: list = []
    _scan_cache: dict = {}

    def _scan_for_containing_cluster(gamma):
        """Linear scan: find a cluster whose Γ_g-projected cone contains
        γ.  Used as fallback for flavored partial-rank clusters where
        _solve_in_basis can't conclude."""
        key = tuple(int(x) for x in gamma)
        if key in _scan_cache:
            return _scan_cache[key]
        if _has_flav:
            gamma_proj = _project(gamma)
            n_g = len(gamma_proj)
        else:
            n_g = len(gamma)
        # Build projected-rays cache on first call.
        if not _cluster_rays_proj_cache:
            for c in _clusters_list:
                rays = c.base_monomial_rays
                if not rays:
                    _cluster_rays_proj_cache.append(None)
                    continue
                if _has_flav:
                    _cluster_rays_proj_cache.append(
                        [_project(r) for r in rays],
                    )
                else:
                    _cluster_rays_proj_cache.append(list(rays))
        from itertools import combinations
        g_use = gamma_proj if _has_flav else gamma
        for cid, rays_use in enumerate(_cluster_rays_proj_cache):
            if rays_use is None or len(rays_use) < n_g:
                continue
            for sub in combinations(range(len(rays_use)), n_g):
                basis = [rays_use[i] for i in sub]
                sub_coeffs = _solve_in_basis(g_use, basis)
                if sub_coeffs is None:
                    continue
                if all(c >= 0 for c in sub_coeffs):
                    _scan_cache[key] = cid
                    return cid
        _scan_cache[key] = None
        return None

    def _find_one_containing_cluster(gamma, start_id=None, max_steps=2000):
        """Walk cluster graph to one cluster whose cone contains γ."""
        if start_id is None:
            start_id = _nonempty_start
        if _has_flav:
            gamma_proj = _project(gamma)
        cid = start_id
        for _ in range(max_steps):
            cluster = _clusters_list[cid]
            rays = cluster.base_monomial_rays
            if not rays:
                # Empty-cone cluster (root in flavored cyclic case).
                # Jump to any neighbor.
                edge = next(
                    (e for e in A._fbk_edges if e.src_cluster == cid),
                    None,
                )
                if edge is None:
                    return None
                cid = edge.dst_cluster
                continue
            if _has_flav:
                rays_proj = [_project(r) for r in rays]
                # For partial-rank clusters (fewer rays than rank(Γ_g)),
                # _solve_in_basis returns None even for points actually
                # in the cone.  Fall back to a full scan.
                if len(rays_proj) < len(gamma_proj):
                    return _scan_for_containing_cluster(gamma)
                coeffs = _solve_in_basis(gamma_proj, rays_proj)
            else:
                coeffs = _solve_in_basis(gamma, list(rays))
            if coeffs is None:
                if _has_flav:
                    return _scan_for_containing_cluster(gamma)
                return None
            if all(c >= 0 for c in coeffs):
                return cid
            bad_idx = min(range(len(coeffs)), key=lambda i: coeffs[i])
            bad_ray = rays[bad_idx]
            wanted_facet = tuple(sorted(r for r in rays if r != bad_ray))
            edge = _edge_by_facet.get((cid, wanted_facet))
            if edge is None:
                if _has_flav:
                    return _scan_for_containing_cluster(gamma)
                return None
            cid = edge.dst_cluster
        if _has_flav:
            return _scan_for_containing_cluster(gamma)
        return None

    def _convert_lattice_term_to_word(delta, coeff):
        """(lattice point, q-coeff) → (R[q]-coeff, word) for the cone algebra.

        Locates δ via cluster-graph BFS (find_cone).  For δ with non-zero
        flavour part f (in flavoured case), the cone-label word covers the
        gauge part δ_gauge; the flavour part f becomes a μ^f factor on
        the coefficient, returned as an RLaurent.
        """
        # Coerce incoming coeff to the K-algebra's coefficient ring poly type.
        # (compute_cross_terms feeds us LaurentPoly intermediates from the
        # at-root product; for flavoured K-algebras we need RLaurent.)
        if _has_flav_coeffs:
            if hasattr(coeff, "_coeffs"):
                coeff = _lp_to_R(coeff._coeffs)
            elif isinstance(coeff, RLaurent):
                pass
            else:
                coeff = _lp_to_R({0: int(coeff)})

        if all(d == 0 for d in delta):
            return (coeff, ())

        # Flavoured path: split δ into gauge + flavour, encode flavour
        # as a μ-monomial on the coefficient.
        if _has_flav:
            sec_d, flav_d = decompose_in_basis(delta, _sec_basis, _L_basis)
            sec_d = tuple(int(x) for x in sec_d)
            flav_d = tuple(int(x) for x in flav_d)
            # If gauge part is zero, the lattice point is purely flavour:
            # cone-label identity () with coefficient · μ^{flav_d}.
            if all(d == 0 for d in sec_d) and any(d != 0 for d in flav_d):
                mu_factor = _mu_monomial(flav_d, q_exp=0)
                return (coeff * mu_factor, ())
        else:
            sec_d, flav_d = None, None

        if _did_complete:
            # ρ-completed entries (a1d4): the cluster rays cover only the
            # original (gauge) cone, so a product landing on a completed
            # generator can't be expressed via a cluster.  Decompose the
            # gauge part over the ρ-completed cones directly.
            _t = sec_d if _has_flav else delta
            ray_indices = None
            coeffs = None
            for _cone in cones_list:
                _ids = sorted(_cone)
                _bas = ([_project(mult_gens_list[i]) for i in _ids]
                        if _has_flav else [mult_gens_list[i] for i in _ids])
                _sol = _solve_in_basis(_t, _bas)
                if _sol is not None and all(c >= 0 for c in _sol):
                    ray_indices = _ids
                    coeffs = list(_sol)
                    break
            if ray_indices is None:
                return None
        else:
            cid = _find_one_containing_cluster(delta)
            if cid is None:
                return None
            cluster = _clusters_list[cid]
            rays_in_cluster = list(cluster.base_monomial_rays)
            ray_indices = [ray_to_idx[tuple(r)] for r in rays_in_cluster]
            if _has_flav:
                rays_sec = [_project(r) for r in rays_in_cluster]
                coeffs = _solve_in_basis(sec_d, rays_sec)
            else:
                coeffs = _solve_in_basis(delta, rays_in_cluster)
            if coeffs is None or any(c < 0 for c in coeffs):
                # Under-determined case (more rays than gauge-rank): the
                # simplicial solver bails.  Fall back to subset enumeration —
                # find a size-n_g subset whose square solve is non-negative.
                if _has_flav:
                    target = sec_d
                    basis = rays_sec
                    n_g = len(target)
                else:
                    target = delta
                    basis = rays_in_cluster
                    n_g = len(target)
                if len(basis) > n_g:
                    from itertools import combinations
                    found = False
                    for sub in combinations(range(len(basis)), n_g):
                        sub_basis = [basis[i] for i in sub]
                        sub_coeffs = _solve_in_basis(target, sub_basis)
                        if sub_coeffs is None:
                            continue
                        if all(c >= 0 for c in sub_coeffs):
                            # Expand back to length-len(basis) coeff vector.
                            full = [0] * len(basis)
                            for k, idx in enumerate(sub):
                                full[idx] = sub_coeffs[k]
                            coeffs = full
                            found = True
                            break
                    if not found:
                        return None
                else:
                    return None
        face_ids = sorted(
            ray_indices[k] for k in range(len(coeffs)) if coeffs[k] > 0
        )
        face_rays = [mult_gens_list[i] for i in face_ids]
        if _has_flav:
            rays_sec_face = [_project(r) for r in face_rays]
            face_coeffs = _solve_in_basis(sec_d, rays_sec_face)
        else:
            face_coeffs = _solve_in_basis(delta, face_rays)
        if face_coeffs is None:
            return None
        powers_tuple = tuple(int(c) for c in face_coeffs)
        word = tuple(
            face_ids[k]
            for k in range(len(powers_tuple))
            for _ in range(powers_tuple[k])
        )
        # cone_label_phase (heptagon convention): coefficient gets a
        # q^phase factor where phase = -Σ_{i<j} cocycle(g_i, g_j) · p_i · p_j.
        phase = 0
        for a in range(len(face_ids)):
            for b in range(a + 1, len(face_ids)):
                pa, pb = powers_tuple[a], powers_tuple[b]
                if pa and pb:
                    phase -= cocycle_table.get(
                        (face_ids[a], face_ids[b]), 0,
                    ) * pa * pb
        # In the flavored case, L_{word} = Π_k L_{canonical_lift[face_id_k]}^{p_k}
        # has flavor part Σ_k p_k · canonical_lift[face_id_k].flav.  The lattice
        # point we want is δ = sec_d + flav_d, so the μ-shift to apply is
        # flav_d - Σ_k p_k · canonical_lift_flav_k.
        effective_flav = flav_d if _has_flav_coeffs else None
        if _has_flav_coeffs:
            from snf_kernel import decompose_in_basis as _dib
            sum_lift_flav = [0] * len(_L_basis)
            for fi, p in zip(face_ids, powers_tuple):
                if p == 0:
                    continue
                _, lift_flav = _dib(
                    mult_gens_list[fi], _sec_basis, _L_basis,
                )
                for k in range(len(_L_basis)):
                    sum_lift_flav[k] += int(lift_flav[k]) * p
            effective_flav = tuple(
                int(flav_d[k]) - sum_lift_flav[k]
                for k in range(len(_L_basis))
            )
        # Apply phase and effective μ^{flav_d - Σ lift_flav}.
        if _has_flav_coeffs and any(d != 0 for d in effective_flav):
            coeff = coeff * _mu_monomial(effective_flav, q_exp=phase)
        elif phase != 0:
            from laurent_poly import LaurentPoly as _LP
            if _has_flav_coeffs:
                coeff = coeff * RLaurent(R, {phase: 1})
            else:
                coeff = coeff * _LP({phase: 1})
        return (coeff, word)

    def _rho_apply_word(word):
        return tuple(rho_perm.get(i, i) for i in word)

    def _rho_propagate_term(coeff, word, i_g, i_h):
        """ρ-propagate a (coeff, word) entry of cross_product(i_g, i_h)
        to cross_product(ρ(i_g), ρ(i_h)).  In the flavored case:

            ρ(L_i · L_j) = ρ(L_i) · ρ(L_j)
                         = μ^{δ(i)+δ(j)} L_{ρ(i)} · L_{ρ(j)}

        and ρ acts on the result by negating its μ-character.  So if
        cross_product(i,j) contains (coeff · μ^a, word), then
        cross_product(ρ(i),ρ(j)) contains
        (star(coeff) · μ^{-δ(i)-δ(j) + Σ_w δ(w)}, ρ(word))."""
        new_word = _rho_apply_word(word)
        if not _has_flav_coeffs:
            return (coeff, new_word)
        # Total flavor offset to apply.
        d_i = rho_delta.get(i_g, (0,) * len(L_basis))
        d_j = rho_delta.get(i_h, (0,) * len(L_basis))
        d_word_sum = [0] * len(L_basis)
        for w in word:
            dw = rho_delta.get(w, None)
            if dw is None:
                continue
            for k in range(len(L_basis)):
                d_word_sum[k] += dw[k]
        # Net μ-shift applied to the coeff.
        net = tuple(d_word_sum[k] - d_i[k] - d_j[k] for k in range(len(L_basis)))
        # Star the existing coeff (μ → μ^{-1}), then multiply by μ^net.
        new_coeff_dict: dict = {}
        for q_exp, r_el in coeff.coeffs.items():
            starred = r_el.star()
            # Shift by μ^net.
            shifted_basis: dict = {}
            for char, c in starred.terms.items():
                new_char = tuple(char[k] + net[k] for k in range(len(L_basis)))
                shifted_basis[new_char] = shifted_basis.get(new_char, 0) + c
            shifted_basis = {k: v for k, v in shifted_basis.items() if v != 0}
            if shifted_basis:
                new_coeff_dict[q_exp] = RElement(R, shifted_basis)
        from zplus_ring import RLaurent as _RL
        return (_RL(R, new_coeff_dict), new_word)

    # Chart-multiply: compute L_g · L_h via the cluster cone graph,
    # bypassing the slow qt_multiply-at-root inside BPSKAlgebra (which
    # explodes at E_7+ scale).  Goes to a chart C containing γ_a where
    # F_a is a single Laurent monomial, multiplies there (monomial ×
    # small polynomial F_b-at-C), then pulls back to root and runs the
    # standard F-basis peel-off.
    from lattice import Lattice as _Lattice, LatticeTorus as _LT
    from laurent_poly import LaurentPoly as _LP
    from lattice_mutation import solve as _lm_solve, solve_inverse as _lm_sinv
    from bps_kalgebra_internals import qt_multiply as _qt_mul
    from bps_kalgebra_internals import find_lowest as _find_lowest
    _lattice = _Lattice(A._fbk_pairing)
    # LRU cache for F-internal-at-root: bounded by max_F_cache to avoid OOM
    # at E_8 scale (deep mult-gens have F-polys of ~100MB each).
    from collections import OrderedDict
    _F_internal_lp_cache: "OrderedDict[tuple, dict]" = OrderedDict()
    _max_F_cache = 32  # ~3 GB worst case for E_8

    def _F_lp_root(gamma):
        """F(γ) at root as dict[Vec, LaurentPoly], LRU-cached."""
        cached = _F_internal_lp_cache.get(gamma)
        if cached is not None:
            _F_internal_lp_cache.move_to_end(gamma)
            return cached
        F_qn = A._F_internal(gamma)
        F_lp = {
            d: (qn.to_laurent() if hasattr(qn, "to_laurent") else qn)
            for d, qn in F_qn.items()
        }
        _F_internal_lp_cache[gamma] = F_lp
        while len(_F_internal_lp_cache) > _max_F_cache:
            _F_internal_lp_cache.popitem(last=False)
        return F_lp

    _zero_vec = tuple(0 for _ in range(len(A._fbk_pairing)))

    # Precompute: for each mult-gen-as-tuple, find a cluster that has it
    # as a ray (used as a non-empty start for find_cone, since flavored
    # quivers can have cluster 0 with empty rays at the root chart).
    _ray_to_cid: dict = {}
    for _cid, _c in enumerate(A._fbk_clusters):
        for _r in _c.base_monomial_rays:
            _rt = tuple(_r)
            if _rt not in _ray_to_cid:
                _ray_to_cid[_rt] = _cid

    def _chart_multiply_via_parent(g_a, g_b):
        """Flavoured fallback: use parent A.multiply (correct for any
        case; the chart-multiply trick can lose terms with mixed q/μ
        structure)."""
        prod = A.multiply(g_a, g_b)
        result: dict = {}
        for delta, c in prod.terms.items():
            # delta is a Γ-lattice point, c is a LaurentPoly (q-only).
            d_tuple = tuple(int(x) for x in delta)
            # Promote to LaurentPoly→RLaurent conversion path: pass to
            # _convert_lattice_term_to_word with the right typed coeff.
            result[d_tuple] = c
        return result

    def _chart_multiply(g_a, g_b):
        """L_{γ_a} · L_{γ_b}.  Goes via chart route: F_b computed at root
        via compute_F, forward-walked to chart C (where γ_a is monomial)
        via solve, multiplied with monomial F_a there, walked back to
        root for the F-basis peel-off.  Works for both non-flavoured and
        flavoured cases."""
        # For mult-gens we already know a containing cluster; for general
        # γ we walk from a non-empty start.
        start = _ray_to_cid.get(tuple(g_a), 0)
        try:
            cid = find_cone(
                A._fbk_clusters, A._fbk_edges, tuple(g_a),
                start_id=start,
                sec_basis=A._fbk_sec_basis, L_basis=A._fbk_L_basis,
            )
        except RuntimeError:
            # g_a is a ρ-completed mult-gen lying OUTSIDE the gauge cluster
            # cones — a1d4: the D₄ gauge cone is not ρ-closed (the S₃⊃Z₂
            # family minimum), so the ρ-orbit completion adds generators that
            # are not rays of any cluster, and the chart route (which needs
            # g_a monomial in some cluster) cannot locate them.  Fall back to
            # the parent BPS multiply, correct for any charge (the chart trick
            # is only a speed optimisation of it).
            return _chart_multiply_via_parent(g_a, g_b)
        path = A._fbk_clusters[cid].path_from_root
        g_a_at_C = tuple(g_a)
        for ch in path:
            g_a_at_C = _mu_g_step(g_a_at_C, ch, A._fbk_pairing)
        F_b_root = dict(_F_lp_root(tuple(g_b)))
        F_b_lt = _LT(_lattice, F_b_root)
        for ch in path:
            F_b_lt = _lm_solve(F_b_lt, ch)
        F_b_at_C = dict(F_b_lt._terms)
        F_a_at_C = {g_a_at_C: _LP({0: 1})}
        prod_at_C = _qt_mul(F_a_at_C, F_b_at_C, _lattice)
        prod_lt = _LT(_lattice, prod_at_C)
        for ch in reversed(path):
            prod_lt = _lm_sinv(prod_lt, ch)
        prod_root = dict(prod_lt._terms)
        # 5. F-basis peel-off.
        cone_gens_list = list(A.cone_gens)
        cone_witness = A._cone_witness
        F_low_cache: dict = {}
        result: dict = {}
        prod = dict(prod_root)
        for _ in range(2000):
            prod = {g: c for g, c in prod.items() if not c.is_zero()}
            if not prod:
                break
            c_low = _find_lowest(list(prod.keys()), cone_gens_list, cone_witness)
            coeff = prod[c_low]
            prev = result.get(c_low)
            new_c = coeff if prev is None else (prev + coeff)
            if new_c.is_zero():
                result.pop(c_low, None)
            else:
                result[c_low] = new_c
            F_low_lp = F_low_cache.get(c_low)
            if F_low_lp is None:
                F_low_lp = _F_lp_root(c_low)
                F_low_cache[c_low] = F_low_lp
            neg_coeff = _LP({k: -v for k, v in coeff._coeffs.items()})
            sub = _qt_mul({_zero_vec: neg_coeff}, F_low_lp, _lattice)
            for g, nc in sub.items():
                cur = prod.get(g)
                if cur is None:
                    prod[g] = nc
                else:
                    s = cur + nc
                    if s.is_zero():
                        prod.pop(g, None)
                    else:
                        prod[g] = s
        prod = {g: c for g, c in prod.items() if not c.is_zero()}
        if prod:
            raise RuntimeError(
                "cluster_cone_builder F-basis peel: cap (2000 iterations) "
                f"exhausted with {len(prod)} residual charges; the "
                "decomposition is incomplete and the multiplication table "
                "would be silently truncated -- raise the cap."
            )
        return {g: c for g, c in result.items() if not c.is_zero()}

    def _build_full_mult_table():
        if not rho_perm:
            return
        # Checkpointing: load existing cross_cache from disk if present,
        # save periodically so a container interrupt doesn't lose work.
        import os, pickle
        ckpt_path = os.environ.get("CROSS_CACHE_CKPT")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "rb") as _f:
                    loaded = pickle.load(_f)
                cross_cache.update(loaded)
                print(f"      [resumed from {ckpt_path}: "
                      f"{len(cross_cache)} entries]", flush=True)
            except Exception as _e:
                print(f"      [checkpoint load failed: {_e}]", flush=True)
        seen: set = set(cross_cache.keys())
        _n_mult = 0
        _t_mult = _time.time()
        _last_ckpt = _time.time()
        # Iterate shallow mult-gens first (highest tuple values =
        # close to root chart).  Their F-polys are smaller, so we
        # warm up the LRU cache without the giant deep F-polys.
        for i_g in range(n_mg - 1, -1, -1):
            for i_h in range(n_mg - 1, -1, -1):
                if (i_g, i_h) in seen:
                    continue
                g = mult_gens_list[i_g]
                h = mult_gens_list[i_h]
                _t_one = _time.time()
                prod_terms = _chart_multiply(g, h)
                _dt_mul = _time.time() - _t_one
                _n_mult += 1
                if _n_mult <= 10 or _n_mult % 50 == 0:
                    print(f"      [mult {_n_mult}: chart {_dt_mul*1000:.0f}ms, "
                          f"{len(prod_terms)} terms, total {_time.time()-_t_mult:.1f}s]",
                          flush=True)
                rep_terms_by_word: dict = {}
                for delta, coeff in prod_terms.items():
                    t = _convert_lattice_term_to_word(delta, coeff)
                    if t is not None:
                        c, w = t
                        if w in rep_terms_by_word:
                            rep_terms_by_word[w] = rep_terms_by_word[w] + c
                        else:
                            rep_terms_by_word[w] = c
                rep_terms: list = [
                    (c, w) for w, c in rep_terms_by_word.items()
                    if not c.is_zero()
                ]
                cross_cache[(i_g, i_h)] = rep_terms
                seen.add((i_g, i_h))
                # Free intermediate polynomial state between mults.
                import gc
                gc.collect()
                # Checkpoint after every mult (orbit propagation just below
                # may add many more entries — save before/after both).
                if ckpt_path:
                    tmp = ckpt_path + ".tmp"
                    with open(tmp, "wb") as _f:
                        pickle.dump(cross_cache, _f)
                    os.replace(tmp, ckpt_path)
                # Propagate via ρ.  Skip propagation if ρ isn't defined
                # on BOTH indices (otherwise we'd alias an unrelated pair
                # to the current result, polluting the cross_table).
                cur_pair = (i_g, i_h)
                cur_terms = rep_terms
                for _ in range(n_mg):
                    if (cur_pair[0] not in rho_perm
                            or cur_pair[1] not in rho_perm):
                        break
                    new_pair = (
                        rho_perm[cur_pair[0]],
                        rho_perm[cur_pair[1]],
                    )
                    if new_pair in seen:
                        break
                    # ρ-equivariance with section-offset tracking.
                    new_terms = [
                        _rho_propagate_term(coeff, word,
                                            cur_pair[0], cur_pair[1])
                        for coeff, word in cur_terms
                    ]
                    cross_cache[new_pair] = new_terms
                    seen.add(new_pair)
                    cur_pair = new_pair
                    cur_terms = new_terms

    _log(f"step 5+6 setup: ready for mult table")
    _build_full_mult_table()
    _log(f"step 6: mult table {len(cross_cache)} entries")

    def compute_cross_terms(i_g, i_h):
        if (i_g, i_h) in cross_cache:
            return cross_cache[(i_g, i_h)]
        g = mult_gens_list[i_g]
        h = mult_gens_list[i_h]
        prod = A.multiply(g, h)
        result = []
        for delta, coeff in prod.terms.items():
            t = _convert_lattice_term_to_word(delta, coeff)
            if t is not None:
                result.append(t)
        cross_cache[(i_g, i_h)] = result
        return result

    # 7. ConeData class.
    class _ExportedConeData(FiniteConeData):
        def coefficient_ring(self):
            return R

        def mult_gens(self):
            return tuple(range(n_mg))

        def cones(self):
            return tuple(cones_list)

        def q_commute(self, g, h):
            return g == h or (g, h) in cocycle_table

        def cocycle(self, g, h):
            if g == h:
                return 0
            return cocycle_table[(g, h)]

        def cross_product(self, g, h):
            return compute_cross_terms(g, h)

        def to_cone_label(self, native_label):
            # Native label = sorted tuple of (idx, power) pairs.
            # Cone label = (frozenset(idx_with_power), {idx: power}).
            gens = frozenset(i for i, p in native_label if p != 0)
            powers = {i: p for i, p in native_label if p != 0}
            return (gens, powers)

        def from_cone_label(self, gens, powers):
            return tuple(sorted((i, powers[i]) for i in gens))

        def cycle_period_bound(self):
            return n_mg

    cone_data_instance = _ExportedConeData()

    # 8. ConeKAlgebra subclass.
    class _ExportedConeKAlgebra(ConeKAlgebra):
        """Auto-exported ConeKAlgebra from a FiniteBPSKAlgebra."""
        _R = R
        _cone_data = cone_data_instance
        _rho_perm = rho_perm
        _rho_inv_perm = rho_inv_perm
        _rho_delta = rho_delta
        _L_basis_rank = len(L_basis) if L_basis else 0
        _ray_idx_to_gamma = mult_gens_list  # for trace dispatch
        _algebra_ref = A  # captured for trace evaluation (closure)

        def coefficient_ring(self):
            return self._R

        def identity(self):
            # Native label = sorted tuple of (idx, power) pairs.
            return ()

        def cone_data(self):
            return self._cone_data

        def rho(self, label):
            return tuple(sorted(
                (self._rho_perm.get(i, i), p) for (i, p) in label
            ))

        def rho_inverse(self, label):
            return tuple(sorted(
                (self._rho_inv_perm.get(i, i), p) for (i, p) in label
            ))

        def rho_R_element(self, elem):
            """Full ρ on an R-coefficient Element: permute labels by
            ``rho_perm``, then for each term apply

                c · L_a  →  star(c) · μ^{Σ δ(i_k)·p_k} · L_{ρ(a)}

            where star is the R-module μ ↔ μ^{-1} conjugation and δ(i)
            is ``rho_delta[i]``.  For trivial-flavor algebras this
            reduces to the Z-form ``rho_element``.
            """
            from kalgebra import Element as _El
            if self._L_basis_rank == 0:
                return self.rho_element(elem)
            d_rank = self._L_basis_rank
            out: dict = {}
            for label, coeff in elem.terms.items():
                new_label = self.rho(label)
                # Sum δ(i)·p over (i, p) in label.
                net = [0] * d_rank
                for i, p in label:
                    di = self._rho_delta.get(i, None)
                    if di is None:
                        continue
                    for k in range(d_rank):
                        net[k] += di[k] * p
                net_t = tuple(net)
                # Apply star + μ^net to coeff (RLaurent).
                new_coeff_dict: dict = {}
                for q_exp, r_el in coeff.coeffs.items():
                    starred = r_el.star()
                    shifted: dict = {}
                    for char, c in starred.terms.items():
                        new_char = tuple(char[k] + net_t[k] for k in range(d_rank))
                        shifted[new_char] = shifted.get(new_char, 0) + c
                    shifted = {k: v for k, v in shifted.items() if v != 0}
                    if shifted:
                        new_coeff_dict[q_exp] = RElement(self._R, shifted)
                new_coeff = RLaurent(self._R, new_coeff_dict)
                if new_label in out:
                    out[new_label] = out[new_label] + new_coeff
                else:
                    out[new_label] = new_coeff
            return _El({l: c for l, c in out.items() if not c.is_zero()})

        def r_label_decompose(self, label):
            # Flavour-in-coefficients (free over R(G_f)): the canonical basis
            # is flavour-neutral, so the lift coordinate is trivial — section =
            # label, single irrep = χ₀ (R.one_basis()).  Implemented DIRECTLY
            # (not via the to-be-retired _label_section_decompose, which now
            # derives from this through KAlgebra's forward bridge) so that
            # method can be obsoleted.
            return label, self.coefficient_ring().one_basis()

        def r_label_compose(self, section, r_basis_label):
            # Trivial lift: the irrep is central/neutral, so the label is the
            # section (no embed_R round-trip — keeps it independent of the
            # to-be-retired embed_R too).
            return section

        def _trace_residual(self, seed_label, K):
            # seed_label = native = sorted tuple of (idx, power) pairs.
            if not seed_label:
                return self._algebra_ref.trace(
                    tuple(0 for _ in range(rank)), K=K,
                )
            if len(seed_label) == 1 and seed_label[0][1] == 1:
                ray_idx = seed_label[0][0]
                gamma = self._ray_idx_to_gamma[ray_idx]
                return self._algebra_ref.trace_at_chart(gamma, K=K)
            raise ValueError(
                f"_trace_residual: unexpected seed {seed_label!r}"
            )

    return _ExportedConeKAlgebra()


def _mu_relement_to_chi(r_el, su2_ring):
    """RElement over AbelianZPlusRing(rank=1) → RElement over SU2ZPlusRing.

    Each σ-symmetric μ-polynomial p(μ) = Σ a_k (μ^k + μ^{-k}) (k ≥ 0)
    converts via χ_k - χ_{k-2} = μ^k + μ^{-k} as:
        p(μ) = Σ_k (p_k - p_{k+2}) χ_k    where p_k = p_{-k} for k ≥ 0.
    Raises ValueError on non-σ-symmetric input.
    """
    from zplus_ring import RElement
    p: dict = {}
    for char, c in r_el.terms.items():
        p[char[0]] = c
    for k in list(p.keys()):
        if p.get(k, 0) != p.get(-k, 0):
            raise ValueError(
                f"non-σ-symmetric μ-polynomial: {dict(p)}"
            )
    max_k = max(abs(k) for k in p) if p else 0
    chi_terms: dict = {}
    for k in range(max_k + 1):
        c_k = p.get(k, 0) - p.get(k + 2, 0)
        if c_k != 0:
            chi_terms[k] = c_k
    return RElement(su2_ring, chi_terms)


def recognize_su2_flavor(cone_kalg):
    """Convert a U(1)-flavored ConeKAlgebra (over AbelianZPlusRing(rank=1))
    to an SU(2)-flavored ConeKAlgebra (over SU2ZPlusRing) by repackaging
    σ-symmetric μ-Laurent coefficients as SU(2) χ-character polynomials.

    Precondition: every aggregated cross_product entry's μ-Laurent
    coefficient is σ-symmetric (p(μ^{-1}) = p(μ)).  This holds when the
    underlying BPS quiver was specified in a w-diagonal frame (the
    σ-anti-fixed direction is along a single basis vector = ker B).

    For rank-2 (e.g. A_1D_{2k} with k ≥ 2: U(1) × SU(2) flavor), use
    ``recognize_su2_u1_flavor`` which keeps one direction as U(1) and
    promotes the other to SU(2).
    """
    from zplus_ring import (
        AbelianZPlusRing, SU2ZPlusRing, RElement, RLaurent,
    )
    from cone_data import FiniteConeData
    from cone_kalgebra import ConeKAlgebra
    R_old = cone_kalg.coefficient_ring()
    if not (isinstance(R_old, AbelianZPlusRing) and R_old.rank == 1):
        raise ValueError(
            f"recognize_su2_flavor: expected AbelianZPlusRing(rank=1), "
            f"got {R_old!r}"
        )
    su2 = SU2ZPlusRing()
    old_cd = cone_kalg.cone_data()
    n_mg = len(cone_kalg._ray_idx_to_gamma)

    # Aggregate + convert each cross_product entry.
    new_cross: dict = {}
    for i in range(n_mg):
        for j in range(n_mg):
            agg: dict = {}
            for c, w in old_cd.cross_product(i, j):
                if not isinstance(c, RLaurent):
                    continue
                if w in agg:
                    agg[w] = agg[w] + c
                else:
                    agg[w] = c
            new_terms: list = []
            for w, c in agg.items():
                new_qcoeffs: dict = {}
                for q_exp, r_el in c.coeffs.items():
                    chi_el = _mu_relement_to_chi(r_el, su2)
                    if not chi_el.is_zero():
                        new_qcoeffs[q_exp] = chi_el
                if new_qcoeffs:
                    new_terms.append((RLaurent(su2, new_qcoeffs), w))
            new_cross[(i, j)] = new_terms

    class _SU2ExportedConeData(FiniteConeData):
        def coefficient_ring(self):
            return su2
        def mult_gens(self):
            return old_cd.mult_gens()
        def cones(self):
            return old_cd.cones()
        def q_commute(self, g, h):
            return old_cd.q_commute(g, h)
        def cocycle(self, g, h):
            return old_cd.cocycle(g, h)
        def cross_product(self, g, h):
            return tuple(new_cross.get((g, h), []))
        def to_cone_label(self, native_label):
            return old_cd.to_cone_label(native_label)
        def from_cone_label(self, gens, powers):
            return old_cd.from_cone_label(gens, powers)
        def cycle_period_bound(self):
            return old_cd.cycle_period_bound()

    _new_cd = _SU2ExportedConeData()

    class _SU2ExportedConeKAlgebra(ConeKAlgebra):
        _R = su2
        _cone_data = _new_cd
        _rho_perm = cone_kalg._rho_perm
        _rho_inv_perm = cone_kalg._rho_inv_perm
        _ray_idx_to_gamma = cone_kalg._ray_idx_to_gamma
        def coefficient_ring(self): return self._R
        def identity(self): return ()
        def cone_data(self): return self._cone_data
        def rho(self, label):
            return tuple(sorted(
                (self._rho_perm.get(i, i), p) for (i, p) in label
            ))
        def rho_inverse(self, label):
            return tuple(sorted(
                (self._rho_inv_perm.get(i, i), p) for (i, p) in label
            ))
        def rho_R_element(self, elem):
            # SU(2) χ-characters are Weyl-invariant: star is identity
            # and the U(1) rho_delta vanishes after χ-recognition.  So
            # the full ρ on R-Elements reduces to the Z-form linear
            # extension of the label permutation.
            return self.rho_element(elem)
        def r_label_decompose(self, label):
            # Flavour-in-coefficients (free over R(G_f)): the canonical basis
            # is flavour-neutral, so the lift coordinate is trivial — section =
            # label, single irrep = χ₀ (R.one_basis()).  Implemented DIRECTLY
            # (not via the to-be-retired _label_section_decompose, which now
            # derives from this through KAlgebra's forward bridge) so that
            # method can be obsoleted.
            return label, self.coefficient_ring().one_basis()

        def r_label_compose(self, section, r_basis_label):
            # Trivial lift: the irrep is central/neutral, so the label is the
            # section (no embed_R round-trip — keeps it independent of the
            # to-be-retired embed_R too).
            return section
        def _trace_residual(self, seed_label, K):
            raise NotImplementedError(
                "SU(2)-flavored trace_residual: not implemented"
            )

    return _SU2ExportedConeKAlgebra()


def recognize_su2_u1_flavor(cone_kalg, su2_axis=-1):
    """Convert a U(1)×U(1)-flavored ConeKAlgebra (over
    AbelianZPlusRing(rank=2)) to SU(2)×U(1)-flavored (over
    SU2xU1ZPlusRing) by promoting one μ-axis to SU(2) χ-characters
    and keeping the other as U(1).

    ``su2_axis`` selects which of the two μ-axes is the σ-anti-fixed
    (SU(2) Cartan) direction.  Default ``-1`` = last axis (matches the
    A_1D_{2k} w-diagonal convention where the last lattice coord is
    σ-anti-fixed = ker B direction).
    """
    from zplus_ring import (
        AbelianZPlusRing, SU2xU1ZPlusRing, RElement, RLaurent,
    )
    from cone_data import FiniteConeData
    from cone_kalgebra import ConeKAlgebra
    R_old = cone_kalg.coefficient_ring()
    if not (isinstance(R_old, AbelianZPlusRing) and R_old.rank == 2):
        raise ValueError(
            f"recognize_su2_u1_flavor: expected AbelianZPlusRing(rank=2), "
            f"got {R_old!r}"
        )
    if su2_axis == -1:
        su2_axis = 1
    if su2_axis not in (0, 1):
        raise ValueError(f"su2_axis must be 0 or 1, got {su2_axis}")
    other_axis = 1 - su2_axis
    new_ring = SU2xU1ZPlusRing()

    def _convert_relement(r_el):
        """RElement over AbelianZPlusRing(rank=2) → RElement over
        SU2xU1ZPlusRing.  Groups by U(1) charge, then converts the
        σ-anti μ-polynomial to χ via μ^k + μ^{-k} → χ_k - χ_{k-2}."""
        # Group by other_axis (= U(1) charge m).
        by_m: dict = {}
        for char, c in r_el.terms.items():
            m_other = char[other_axis]
            m_su2 = char[su2_axis]
            by_m.setdefault(m_other, {})[m_su2] = c
        out: dict = {}
        for m_other, p in by_m.items():
            # Check σ-symmetry: p[k] == p[-k].
            for k in list(p.keys()):
                if p.get(k, 0) != p.get(-k, 0):
                    raise ValueError(
                        f"non-σ-symmetric μ-polynomial in σ-anti axis "
                        f"(σ-fixed charge m={m_other}): {p}"
                    )
            max_k = max(abs(k) for k in p) if p else 0
            for k in range(max_k + 1):
                c_k = p.get(k, 0) - p.get(k + 2, 0)
                if c_k != 0:
                    out[(k, m_other)] = c_k
        return RElement(new_ring, out)

    old_cd = cone_kalg.cone_data()
    n_mg = len(cone_kalg._ray_idx_to_gamma)
    new_cross: dict = {}
    for i in range(n_mg):
        for j in range(n_mg):
            agg: dict = {}
            for c, w in old_cd.cross_product(i, j):
                if not isinstance(c, RLaurent):
                    continue
                if w in agg:
                    agg[w] = agg[w] + c
                else:
                    agg[w] = c
            new_terms: list = []
            for w, c in agg.items():
                new_qcoeffs: dict = {}
                for q_exp, r_el in c.coeffs.items():
                    converted = _convert_relement(r_el)
                    if not converted.is_zero():
                        new_qcoeffs[q_exp] = converted
                if new_qcoeffs:
                    new_terms.append((RLaurent(new_ring, new_qcoeffs), w))
            new_cross[(i, j)] = new_terms

    class _SU2U1ExportedConeData(FiniteConeData):
        def coefficient_ring(self): return new_ring
        def mult_gens(self): return old_cd.mult_gens()
        def cones(self): return old_cd.cones()
        def q_commute(self, g, h): return old_cd.q_commute(g, h)
        def cocycle(self, g, h): return old_cd.cocycle(g, h)
        def cross_product(self, g, h):
            return tuple(new_cross.get((g, h), []))
        def to_cone_label(self, native_label):
            return old_cd.to_cone_label(native_label)
        def from_cone_label(self, gens, powers):
            return old_cd.from_cone_label(gens, powers)
        def cycle_period_bound(self):
            return old_cd.cycle_period_bound()

    _new_cd = _SU2U1ExportedConeData()

    # Project rho_delta to the U(1) axis only (after SU(2) recognition,
    # the SU(2)-side δ vanishes because χ_n is Weyl-invariant).
    rho_delta_old = getattr(cone_kalg, "_rho_delta", None) or {}
    rho_delta_u1: dict = {
        i: (d[other_axis],) for i, d in rho_delta_old.items()
    }

    class _SU2U1ExportedConeKAlgebra(ConeKAlgebra):
        _R = new_ring
        _cone_data = _new_cd
        _rho_perm = cone_kalg._rho_perm
        _rho_inv_perm = cone_kalg._rho_inv_perm
        _rho_delta = rho_delta_u1
        _ray_idx_to_gamma = cone_kalg._ray_idx_to_gamma
        def coefficient_ring(self): return self._R
        def identity(self): return ()
        def cone_data(self): return self._cone_data
        def rho(self, label):
            return tuple(sorted(
                (self._rho_perm.get(i, i), p) for (i, p) in label
            ))
        def rho_inverse(self, label):
            return tuple(sorted(
                (self._rho_inv_perm.get(i, i), p) for (i, p) in label
            ))
        def rho_R_element(self, elem):
            """Full ρ on SU(2)×U(1)-coefficient Element:
            label perm + μ^{δ_U1} shift on U(1)-charge + star (which
            flips U(1) charge sign, leaves SU(2) χ-index alone).
            """
            from kalgebra import Element as _El
            out: dict = {}
            for label, coeff in elem.terms.items():
                new_label = self.rho(label)
                net = 0
                for i, p in label:
                    di = self._rho_delta.get(i, None)
                    if di is None:
                        continue
                    net += di[0] * p
                new_coeff_dict: dict = {}
                for q_exp, r_el in coeff.coeffs.items():
                    starred = r_el.star()
                    shifted: dict = {}
                    for (k, m), c in starred.terms.items():
                        new_char = (k, m + net)
                        shifted[new_char] = shifted.get(new_char, 0) + c
                    shifted = {kc: v for kc, v in shifted.items() if v != 0}
                    if shifted:
                        new_coeff_dict[q_exp] = RElement(self._R, shifted)
                new_coeff = RLaurent(self._R, new_coeff_dict)
                if new_label in out:
                    out[new_label] = out[new_label] + new_coeff
                else:
                    out[new_label] = new_coeff
            return _El({l: c for l, c in out.items() if not c.is_zero()})
        def r_label_decompose(self, label):
            # Flavour-in-coefficients (free over R(G_f)): the canonical basis
            # is flavour-neutral, so the lift coordinate is trivial — section =
            # label, single irrep = χ₀ (R.one_basis()).  Implemented DIRECTLY
            # (not via the to-be-retired _label_section_decompose, which now
            # derives from this through KAlgebra's forward bridge) so that
            # method can be obsoleted.
            return label, self.coefficient_ring().one_basis()

        def r_label_compose(self, section, r_basis_label):
            # Trivial lift: the irrep is central/neutral, so the label is the
            # section (no embed_R round-trip — keeps it independent of the
            # to-be-retired embed_R too).
            return section
        def _trace_residual(self, seed_label, K):
            raise NotImplementedError(
                "SU(2)×U(1) flavored trace_residual: not implemented"
            )

    return _SU2U1ExportedConeKAlgebra()


def _make_finite_bps_kalgebra_class():
    """Builds FiniteBPSKAlgebra lazily so the bps_kalgebra import is
    deferred until the class is first requested (avoids an unconditional
    import-time dependency)."""
    from bps_kalgebra import BPSKAlgebra
    from q_number_poly import QNumberPoly

    class FiniteBPSKAlgebra(BPSKAlgebra):
        """A :class:`BPSKAlgebra` whose canonical-basis F-finder is
        accelerated by a precomputed cluster-cone graph.

        The cluster mutation graph (cones at root + edges + paths) is
        built eagerly at construction.  ``F(γ)`` then uses
        :func:`compute_F` to locate γ in the cluster fan and apply the
        expand-contract algorithm exactly once per edge of the path --
        substantially faster than the default chart-graph search that
        the parent uses for large charges.

        The downstream cone-K-algebra view (lattice-free presentation,
        PentagonKAlgebra-style) will be exposed via
        :meth:`to_cone_kalgebra` (TBD: absorbs the older
        finite_bps_kalgebra machinery).

        Flavour-lift coordinate (Plan 32).  As a ``BPSKAlgebra`` subclass,
        ``FiniteBPSKAlgebra`` inherits ``r_label_decompose`` /
        ``r_label_compose`` — implemented *directly* on ``BPSKAlgebra`` from the
        ``ker(B)`` section/kernel bases, independent of the to-be-retired
        ``_label_section_decompose`` / ``embed_R`` — so ``forget()`` and ring-hom
        flavour reduction / promotion (``base_change`` through ``augmentation`` /
        ``restriction_hom`` / ``unit_hom``) work out of the box.  The cone-K-algebra
        views from :meth:`to_cone_kalgebra` carry the flavour in their
        *coefficient ring* (free over ``R(G_f)``); their canonical basis is
        flavour-neutral, so they implement the **trivial** lift coordinate
        directly (``(label, χ₀)``) — likewise independent of the old methods.
        """

        def __init__(self, pairing, node_charges, *,
                     populate_specs: bool = False,
                     max_charts: int = 5000,
                     w_pairs: "list | None" = None, **kwargs):
            # `w_pairs`: optional explicit specification of the node-swap
            # (flavour-Weyl) symmetry pairs.  When given, it OVERRIDES the
            # auto-detection below.  Auto-detection finds the largest
            # *abelian* (Z_2 node-swap) symmetry; for theories whose flavour
            # symmetry is a non-abelian extension of it (e.g. A_1D_4, whose
            # S_3 triality extends the Z_2), the detector sees only the Z_2
            # and silently undercounts.  Specifying the swaps explicitly is
            # how a caller hard-codes the intended symmetry and avoids that
            # class of bug when the machinery is reused.
            self._fbk_w_pairs_override = (
                None if w_pairs is None else [tuple(p) for p in w_pairs]
            )
            super().__init__(
                pairing=pairing, node_charges=node_charges, **kwargs,
            )
            pairing_list = [list(row) for row in self.lattice.pairing]
            self._fbk_pairing = pairing_list
            self._fbk_clusters, self._fbk_edges = build_cluster_graph(
                pairing_list, list(self.node_charges),
                populate_specs=populate_specs,
                max_charts=max_charts,
            )
            # Cache (L_basis, sec_basis, n_q) for cone-finder use.
            rank = len(pairing_list)
            rows = []
            for g in self.node_charges:
                rows.append(tuple(
                    sum(pairing_list[i][j] * g[j] for j in range(rank))
                    for i in range(rank)
                ))
            while len(rows) < rank:
                rows.append(tuple(0 for _ in range(rank)))
            from cluster_cone_builder import _integer_lattice_split
            self._fbk_L_basis, self._fbk_sec_basis = _integer_lattice_split(
                rows[:rank],
            )
            # Detect node-swap symmetry pairs (a, b) from BPS quiver:
            # ⟨N_a, N_b⟩ = 0 and ⟨N_a, N_c⟩ = ⟨N_b, N_c⟩ for all c ∉ {a, b}.
            # Operates on the node-charge brackets (not the raw pairing-
            # matrix entries) so it works in arbitrary lattice frames
            # (e.g., SU(3)-flavored A_1D_4 in SU3ADKAlgebra's setup).
            # Skipped entirely when an explicit `w_pairs` override is given.
            if self._fbk_w_pairs_override is not None:
                self._fbk_w_pairs = self._fbk_w_pairs_override
                self._fbk_specs_populated = populate_specs
                return
            _N_list = list(self.node_charges)
            _B_int = [[int(x) for x in row] for row in pairing_list]
            _n_nodes = len(_N_list)
            _rank = len(pairing_list)

            def _bracket(u, v):
                s = 0
                for i in range(_rank):
                    if u[i] == 0:
                        continue
                    for j in range(_rank):
                        if v[j]:
                            s += u[i] * _B_int[i][j] * v[j]
                return s
            _brackets = [
                [_bracket(_N_list[a], _N_list[b]) for b in range(_n_nodes)]
                for a in range(_n_nodes)
            ]
            _w_pairs: list = []
            for a in range(_n_nodes):
                for b in range(a + 1, _n_nodes):
                    if _brackets[a][b] != 0:
                        continue
                    ok = True
                    for c in range(_n_nodes):
                        if c == a or c == b:
                            continue
                        if (_brackets[a][c] != _brackets[b][c]
                                or _brackets[c][a] != _brackets[c][b]):
                            ok = False
                            break
                    if ok:
                        _w_pairs.append((a, b))
            self._fbk_w_pairs = _w_pairs
            self._fbk_specs_populated = populate_specs

        def populate_specs(self):
            """Populate cluster specs (head-mutation + commute/pentagon
            local-move fixpoint).  Required for ``trace_at_chart``;
            opt-in because it's an extra cost on top of cone-graph
            construction.
            """
            if self._fbk_specs_populated:
                return
            _propagate_specs(
                self._fbk_clusters, self._fbk_edges, self._fbk_pairing,
            )
            self._fbk_specs_populated = True

        def _F_internal(self, gamma):
            try:
                F_lp = compute_F(
                    self._fbk_clusters, self._fbk_edges,
                    self._fbk_pairing, tuple(gamma),
                    sec_basis=self._fbk_sec_basis,
                    L_basis=self._fbk_L_basis,
                )
                return {
                    tuple(d): QNumberPoly.from_palindromic_laurent(c)
                    for d, c in F_lp.items()
                }
            except RuntimeError:
                # γ outside the cluster cone surface; defer to parent.
                return super()._F_internal(gamma)

        @property
        def cluster_graph(self):
            """Tuple ``(clusters, edges)`` of the cluster cone graph."""
            return self._fbk_clusters, self._fbk_edges

        def find_cluster_of(self, gamma):
            """Return the ``cluster_id`` whose Γ_g cone contains γ."""
            return find_cone(
                self._fbk_clusters, self._fbk_edges, tuple(gamma),
                sec_basis=self._fbk_sec_basis,
                L_basis=self._fbk_L_basis,
            )

        def trace_at_chart(self, gamma, K=20, **kwargs):
            """Experimental: trace of L_γ via the monomial-chart Nahm sum.

            Requires spec population (calls :meth:`populate_specs` if
            not done).

            Locate cluster C whose cone contains γ.  At C, the canonical
            basis element L_γ is a single monomial X^{γ_at_C}, so the
            Schur-index Nahm sum collapses with massive cancellations
            (per the user).  Delegates to a temp BPSKAlgebra constructed
            with C's spec (POC -- per-call construction; cache later).
            """
            from bps_kalgebra import BPSKAlgebra
            self.populate_specs()
            cid = self.find_cluster_of(gamma)
            cluster = self._fbk_clusters[cid]
            spec = cluster.spec
            if spec is None:
                return super().trace(gamma, K, **kwargs)
            gamma_at_C = tuple(gamma)
            for g in cluster.path_from_root:
                gamma_at_C = _mu_g_step(
                    gamma_at_C, g, self._fbk_pairing,
                )
            # Cache per-cluster BPSKAlgebra so the second trace on the
            # same cluster pays only the Nahm-sum work, not the
            # constructor.  (The hand-coded Nahm sum -- bypassing
            # _schur_index / c_gamma_via_s entirely -- is the next step.)
            cache = getattr(self, "_fbk_chart_algebras", None)
            if cache is None:
                cache = {}
                self._fbk_chart_algebras = cache
            A_at_C = cache.get(cid)
            if A_at_C is None:
                A_at_C = BPSKAlgebra(
                    pairing=self._fbk_pairing,
                    node_charges=list(cluster.node_charges),
                    spec=[tuple(s) for s in spec],
                    verify="off",
                )
                cache[cid] = A_at_C
            return _trace_via_monomial_nahm(
                A_at_C, gamma_at_C, K, **kwargs,
            )

        def to_cone_kalgebra(self):
            """Export this algebra as a self-contained
            :class:`ConeKAlgebra` (lattice-free presentation).

            Builds a ``FiniteConeData`` from the cluster cone graph:

              * mult-gens = all distinct cone-rays across clusters.
              * cones    = each cluster's ray-set (as frozenset of
                           mult-gen indices).
              * cocycle  = ``⟨g, h⟩`` from the pairing (integer; the
                           half-bracket convention follows the parent).
              * cross_product = computed via the parent's ``multiply``
                           on first access (lazy, cached).

            The resulting ConeKAlgebra is independent of ``self`` once
            built -- safe to pickle / share / use without re-running the
            BPS-quiver machinery.

            Works for ANY FiniteBPSKAlgebra (Pentagon, Heptagon, A_{2k}
            linear, A_{2k+1} linear, [A_n, A_m] in finite-Stokes charts,
            etc.).
            """
            return _build_cone_kalgebra_from_finite(self)

    return FiniteBPSKAlgebra


def __getattr__(name):
    if name == "FiniteBPSKAlgebra":
        cls = _make_finite_bps_kalgebra_class()
        globals()["FiniteBPSKAlgebra"] = cls
        return cls
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def summarize(clusters, edges, *, max_print=None):
    from collections import Counter
    print(f"Clusters: {len(clusters)}")
    for c in clusters[:max_print]:
        print(f"  cluster {c.cluster_id}:  "
              f"nodes={c.node_charges}  rays={c.base_monomial_rays}")
    if max_print and len(clusters) > max_print:
        print(f"  ... and {len(clusters) - max_print} more")
    out = Counter(e.src_cluster for e in edges)
    inc = Counter(e.dst_cluster for e in edges)
    print(f"Edges: {len(edges)}")
    print(f"  out-degree distribution: "
          f"{sorted(Counter(out.values()).items())}")
    print(f"  in-degree distribution:  "
          f"{sorted(Counter(inc.values()).items())}")
    for e in edges[:max_print]:
        print(f"  cluster {e.src_cluster}[{e.src_k}] "
              f"--charge={e.charge}--> cluster {e.dst_cluster}[{e.dst_k}]")
    if max_print and len(edges) > max_print:
        print(f"  ... and {len(edges) - max_print} more")


# ---------------------------------------------------------------------------
# A1A2k = A_{2k} linear quiver (n=2k, non-degenerate alternating B)
# ---------------------------------------------------------------------------

def _A2k_quiver(k):
    n = 2 * k
    B = [[0] * n for _ in range(n)]
    for a in range(n - 1):
        B[a][a + 1] = 1
        B[a + 1][a] = -1
    nodes = [tuple(1 if p == a else 0 for p in range(n)) for a in range(n)]
    return B, nodes


if __name__ == "__main__":
    import time
    catalan = {1: 5, 2: 42, 3: 429, 4: 4862}
    print(f"{'k':>3} {'rank':>5} {'clusters':>10} {'expected':>10} "
          f"{'edges':>8} {'n*clusters':>11} {'time':>7}")
    for k in [1, 2, 3]:
        B, nodes = _A2k_quiver(k)
        n_nodes = len(nodes)
        t = time.time()
        try:
            clusters, edges = build_cluster_graph(B, nodes, max_charts=2000)
            dt = time.time() - t
            print(f"{k:>3} {2*k:>5} {len(clusters):>10} {catalan[k]:>10} "
                  f"{len(edges):>8} {n_nodes*len(clusters):>11} {dt:>6.1f}s")
        except RuntimeError as e:
            print(f"{k:>3} ABORT: {e}")
