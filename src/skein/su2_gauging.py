"""SU(2) flavour gauging at the spectrum-generator level.

Implements the gauging rule for the SU(2) x SU(2) quiver gauge theory
and the trinion SU(2)^3 theory: given a finite-chamber
spectrum generator of a theory with an SU(2) flavour symmetry whose
Cartan fugacity zeta dresses the spec in DOUBLET PAIRS

    ... E(zeta X_w) E(zeta^{-1} X_w) ...        (adjacent entries w+f, w-f)

promote the flavour symmetry to a new SU(2) gauge node (Kronecker pair
g, g' with <g, g'> = 2): each doublet pair is replaced by the gauged
node's WILSON TROPICAL TRIPLE and the node's pure factors are appended,

    E(zeta X_w) E(X_w/zeta)  ->  E(X_{w + d}) E(X_{w + d + g'}) E(X_{w - d}),
    append E(X_g) E(X_{g'}),            d := -(g + g')/2,

(the three monomials of the pure-SU(2) fundamental Wilson line F(W),
including the middle bubbling term).  Singlet entries pass through.

Lattice bookkeeping (all integral -- the trinion node-basis trick):
the eaten flavour direction `f` is dropped; two new coordinates
`(d, g')` are appended, with `g = -2d - g'`.  The new pairing extends
the old one by

    <d, g'> = -1   (equivalently <g, g'> = 2,  <d, g> = +1),
    <d, old> = <g', old> = 0,

which reproduces the quiver connectors (e.g. the SU(2)^2
bifund node pairing <gamma_5, gamma_3> = 1, and the certified trinion
pairing).

Every output should be certified by
`BPSQuiver.verify_spectrum_generator`; `gauge_su2_flavour` does this
by default and raises on failure.

Iterating the rule assembles the SU(2)^{n-3} sphere chambers (and any
pants decomposition reachable by repeated flavour gauging) WITHOUT
spectrum-generator auto-find.
"""

from __future__ import annotations

from typing import Sequence

Vec = tuple


def gauge_su2_flavour(
    pairing: Sequence[Sequence[int]],
    spec: Sequence[Sequence[int]],
    f_index: int,
    nodes: Sequence[Sequence[int]] | None = None,
    *,
    verify: bool = True,
) -> tuple[list[list[int]], list[Vec], list[Vec] | None]:
    """Gauge the SU(2) flavour whose Cartan is coordinate `f_index`.

    Requirements on the input:
      * `pairing[f_index]` is identically zero (a kernel direction);
      * spec entries have f-charge in {-1, 0, +1};
      * the (+1)-charged entries come in ADJACENT doublet pairs with a
        (-1)-charged partner at the same underlying charge
        (`w + f` immediately followed by `w - f`, or vice versa).

    `nodes`, when given, is the input theory's simple (node) charge
    set; the output node set is derived (an f-charged node pair w+-f
    becomes the single node w+d; neutral nodes pass through; g, g'
    appended) and used for the certification.

    Returns `(new_pairing, new_spec, new_nodes)` over the lattice with
    `f` dropped and `(d, g')` appended (so rank -> rank + 1).
    """
    n = len(pairing)
    if any(pairing[f_index][j] != 0 for j in range(n)) or any(
        pairing[j][f_index] != 0 for j in range(n)
    ):
        raise ValueError(
            f"coordinate {f_index} is not a kernel (flavour) direction"
        )
    keep = [j for j in range(n) if j != f_index]

    def strip(e):
        return tuple(e[j] for j in keep)

    # --- new pairing: old block (+) the (d, g') block -----------------
    m = len(keep)
    B = [[0] * (m + 2) for _ in range(m + 2)]
    for i2, i in enumerate(keep):
        for j2, j in enumerate(keep):
            B[i2][j2] = pairing[i][j]
    D, GP = m, m + 1            # indices of d and g'
    B[D][GP] = -1
    B[GP][D] = 1

    def emb(w, dd=0, gg=0):
        return strip(w) + (dd, gg)

    # --- transform the spec -------------------------------------------
    out: list[Vec] = []
    i = 0
    while i < len(spec):
        e = tuple(spec[i])
        c = e[f_index]
        if c == 0:
            out.append(emb(e))
            i += 1
            continue
        if i + 1 >= len(spec):
            raise ValueError(f"unpaired flavour-charged spec entry {e}")
        e2 = tuple(spec[i + 1])
        w = tuple(x - c * (1 if j == f_index else 0)
                  for j, x in enumerate(e))
        w2 = tuple(x + c * (1 if j == f_index else 0)
                   for j, x in enumerate(e2))
        if w != w2:
            raise ValueError(
                f"entries {e}, {e2} are not an SU(2) doublet pair"
            )
        # the Wilson tropical triple: w+d, w+d+g', w-d
        out.append(emb(w, 1, 0))
        out.append(emb(w, 1, 1))
        out.append(emb(w, -1, 0))
        i += 2

    # append the new node's pure factors: g = -2d - g', then g'
    zero = tuple(0 for _ in range(m))
    g_vec = zero + (-2, -1)
    gp_vec = zero + (0, 1)
    out.append(g_vec)
    out.append(gp_vec)

    new_nodes = None
    if nodes is not None:
        new_nodes = []
        seen_pairs = set()
        for nd in nodes:
            nd = tuple(nd)
            c = nd[f_index]
            if c == 0:
                new_nodes.append(emb(nd))
                continue
            w = tuple(x - c * (1 if j == f_index else 0)
                      for j, x in enumerate(nd))
            if w in seen_pairs:
                continue            # partner already handled
            seen_pairs.add(w)
            new_nodes.append(emb(w, 1, 0))   # w + d
        new_nodes += [g_vec, gp_vec]

    if verify:
        if new_nodes is None:
            raise ValueError(
                "verification requires the input node set (pass nodes=...)"
            )
        verify_gauged_spec(B, out, new_nodes)
    return B, out, new_nodes


def verify_gauged_spec(B, spec, basis) -> None:
    """Certify a spec against the given node basis: re-express in node
    coordinates and run `BPSQuiver.verify_spectrum_generator`.
    Raises on failure."""
    from bps_quiver_tools import BPSQuiver

    rank = len(B)
    basis = [tuple(b) for b in basis]
    if len(basis) != rank:
        raise ValueError(
            f"node basis has {len(basis)} entries, expected {rank}"
        )

    from directional_subquiver_rg import _node_basis_solver
    solver = _node_basis_solver([tuple(b) for b in basis])
    spec_nb = [list(solver(tuple(v))) for v in spec]
    if any(x != int(x) for row in spec_nb for x in row):
        raise ValueError("spec entries not integral over the chosen node basis")
    spec_nb = [[int(x) for x in row] for row in spec_nb]

    def br(a, b):
        return sum(a[i] * B[i][j] * b[j] for i in range(rank) for j in range(rank))

    Bn = [[br(basis[i], basis[j]) for j in range(rank)] for i in range(rank)]
    nodes = [tuple(1 if j == i else 0 for j in range(rank)) for i in range(rank)]
    Q = BPSQuiver(nodes, exchange_matrix=Bn)
    ok, info = Q.verify_spectrum_generator(spec_nb)
    if not ok:
        raise ValueError(f"gauged spec failed verification: {info}")


# ---------------------------------------------------------------------------
# Pair-basis rotation and doublet arrangement
# ---------------------------------------------------------------------------


def rotate_pair_basis(
    pairing: Sequence[Sequence[int]],
    vectors: Sequence[Sequence[int]],
    i: int,
    j: int,
) -> tuple[list[list[int]], list[Vec]]:
    """Change coordinates so that columns (i, j) become (w, f) with
    `w = (gamma_i + gamma_j)/2`, `f = (gamma_i - gamma_j)/2` -- the
    SU(2)-pair Cartan `f` becomes an honest integral coordinate (the
    lattice is refined by the spinor-class vector `w`).

    A vector `a*gamma_i + b*gamma_j + rest` gets coordinates
    `(w, f) = (a + b, a - b)`.  Requires the rotated pairing to stay
    integral (true when gamma_i, gamma_j pair identically with
    everything, e.g. an SU(2)-flavour pair of an SO(2Nf)-symmetric
    matter sector).
    """
    n = len(pairing)

    def rot(v):
        out = list(v)
        a, b = v[i], v[j]
        out[i], out[j] = a + b, a - b
        return tuple(out)

    # new pairing: <e_i', e_j'> with e_i' = w = (e_i+e_j)/2 etc.
    B = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            def basis_vec(k, which):
                # returns the old-coordinate expansion of new basis k
                from fractions import Fraction
                v = [Fraction(0)] * n
                if k == i:
                    v[i], v[j] = Fraction(1, 2), Fraction(1, 2)
                elif k == j:
                    v[i], v[j] = Fraction(1, 2), Fraction(-1, 2)
                else:
                    v[k] = Fraction(1)
                return v
            vr = basis_vec(r, None)
            vc = basis_vec(c, None)
            val = sum(vr[x] * pairing[x][y] * vc[y]
                      for x in range(n) for y in range(n))
            if val != int(val):
                raise ValueError(
                    f"pair rotation ({i},{j}) gives non-integral pairing"
                )
            B[r][c] = int(val)
    return B, [rot(v) for v in vectors]


def arrange_doublet_pairs(
    pairing: Sequence[Sequence[int]],
    spec: Sequence[Vec],
    f_index: int,
    *,
    max_states: int = 60000,
    max_len: int | None = None,
):
    """BFS over S-preserving local moves (commutations + pentagons,
    via directional_subquiver_rg._local_move_neighbours) to a spec
    where every f-charged entry sits in an adjacent doublet pair
    (w+f immediately next to w-f) with all f-charges in {-1, 0, +1}.
    Returns the arranged spec or None."""
    from directional_subquiver_rg import _local_move_neighbours

    start = tuple(tuple(g) for g in spec)
    if max_len is None:
        max_len = len(start) + 8

    def ok(s):
        k = 0
        while k < len(s):
            c = s[k][f_index]
            if abs(c) > 1:
                return False
            if c == 0:
                k += 1
                continue
            if k + 1 >= len(s):
                return False
            e2 = s[k + 1]
            if e2[f_index] != -c:
                return False
            if any(s[k][m] - c * (1 if m == f_index else 0)
                   != e2[m] + c * (1 if m == f_index else 0)
                   for m in range(len(e2))):
                return False
            k += 2
        return True

    if ok(start):
        return list(start)
    seen = {start}
    frontier = [start]
    while frontier and len(seen) < max_states:
        nxt = []
        for s in frontier:
            for t in _local_move_neighbours(s, pairing, max_len):
                if t in seen:
                    continue
                if ok(t):
                    return list(t)
                seen.add(t)
                nxt.append(t)
                if len(seen) >= max_states:
                    break
            if len(seen) >= max_states:
                break
        frontier = nxt
    return None


def glue_trinion(
    pairing: Sequence[Sequence[int]],
    spec: Sequence[Vec],
    nodes: Sequence[Vec],
    f_index: int,
    *,
    arrange_budget: int = 60000,
) -> tuple[list[list[int]], list[Vec], list[Vec]]:
    """Glue a trinion onto the SU(2) flavour at coordinate `f_index`:
    couple a (2,2,2) half-hypermultiplet between that SU(2) (gauged)
    and two FRESH flavour SU(2)'s, growing the pants decomposition by
    one (n -> n+1 punctures on the sphere).

    Implementation: extend the lattice by two fresh flavour Cartans
    (mu1, mu2); insert the four free-half-hyper factors at the central
    charges (the zeta-symmetric CPT half, mu2-positive)

        +-f + m1 + m2,   +-f - m1 + m2,

    which commute with everything (pure-flavour charges), arranged as
    two adjacent zeta-doublet pairs; arrange the rest of the spec into
    doublet-adjacent form; gauge via `gauge_su2_flavour` (certified).

    Returns `(new_pairing, new_spec, new_nodes)`.
    """
    n = len(pairing)
    # extend by two flavour columns (kernel directions)
    B = [list(row) + [0, 0] for row in pairing] + [[0] * (n + 2), [0] * (n + 2)]
    M1, M2 = n, n + 1

    def ext(v, m1=0, m2=0):
        return tuple(v) + (m1, m2)

    f = tuple(1 if j == f_index else 0 for j in range(n))
    matter = [
        ext(f, 1, 1), ext(tuple(-x for x in f), 1, 1),
        ext(f, -1, 1), ext(tuple(-x for x in f), -1, 1),
    ]
    spec_ext = matter + [ext(v) for v in spec]
    nodes_ext = [ext(v) for v in nodes] + [ext(f, 1, 1), ext(f, -1, 1)]

    arranged = arrange_doublet_pairs_windowed(
        B, spec_ext, f_index, max_states=arrange_budget,
    )
    if arranged is None:
        raise ValueError(
            "glue_trinion: no doublet-adjacent arrangement found in budget"
        )
    return gauge_su2_flavour(B, arranged, f_index, nodes_ext)


# Nf=4 seed (the Kronecker-2 + 4-fundamental chart, before the (2,3)
# pair-rotation) -- the base of the sphere trinion-gluing recursion.
_NF4_B6 = [
    [0, 2, -1, -1, -1, -1], [-2, 0, 1, 1, 1, 1],
    [1, -1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0],
    [1, -1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0],
]
_NF4_SPEC = [
    (0, 0, 0, 1, 0, 0), (0, 1, 0, 1, 0, 0), (1, 1, 0, 1, 0, 0),
    (0, 0, 1, 0, 0, 0), (0, 1, 1, 0, 0, 0), (1, 1, 1, 0, 0, 0),
    (0, 0, 0, 0, 1, 0), (0, 1, 0, 0, 1, 0), (1, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 1), (1, 1, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
]


def sphere_chart(n: int) -> tuple[list[list[int]], list[Vec], list[Vec]]:
    """The S^2_{0,n} BPS chart `(pairing, spec, nodes)` via the SU(2)-
    gauging recursion (`n >= 4`), with **no spectrum-generator auto-find**
    -- the systematic n>5 route.

    Starts from the rotated Nf=4 chart (`S^2_{0,4}` = SU(2) N_f=4) and glues
    `n-4` trinions (each `glue_trinion` grows the pants decomposition by one
    puncture, gauging a fresh SU(2)).  Ranks: `3(n-2) = 2(n-3) + n`
    (gauge `2(n-3)`, flavour `n`); certified at every step by
    `gauge_su2_flavour`/`BPSQuiver.verify_spectrum_generator`.  This chart
    is NOT a skein-nice FST triangulation (it has Kronecker pairs); it is
    the canonical/spec side -- the skein surface needs the nice-chart
    transport or the multi-edge engine upgrade.
    """
    if n < 4:
        raise ValueError("sphere_chart needs n >= 4")
    B_r, spec_r = rotate_pair_basis(_NF4_B6, _NF4_SPEC, 2, 3)
    nodes4 = [tuple(1 if j == i else 0 for j in range(6)) for i in range(6)]
    _, nodes_r = rotate_pair_basis(_NF4_B6, nodes4, 2, 3)
    Bc, sc, nc = B_r, list(spec_r), list(nodes_r)
    f_next = 3
    for _ in range(n - 4):
        prev = len(Bc)
        Bc, sc, nc = glue_trinion(Bc, sc, nc, f_next)
        f_next = prev - 1
    return Bc, sc, nc


def sphere_kalgebra(n: int, *, verify: str = "off"):
    """The canonical `S^2_{0,n}` closed-sphere KAlgebra (a `BPSKAlgebra`)
    via the SU(2)-gauging recursion, in NODE-BASIS form -- builds
    INSTANTLY (no auto-find): the systematic n>5 route.

    `sphere_chart(n)` gives the chart in ambient coordinates; this
    re-expresses the pairing + spec in the simple-node basis (the same
    transform `verify_gauged_spec` certifies through) so `BPSKAlgebra`
    gets unit-vector `node_charges` -- which is what makes the build
    instant.  (Passing the ambient nodes/pairing/spec instead is the bug
    that made the n=6 build hang.)  The returned algebra's
    `multiply`/`trace` still carry the inherent BPS F-solve / Schur-Nahm
    cost; only the *construction* is free.
    """
    from bps_kalgebra import BPSKAlgebra
    from directional_subquiver_rg import _node_basis_solver

    B, spec, nodes = sphere_chart(n)
    rank = len(B)
    solver = _node_basis_solver([tuple(b) for b in nodes])
    spec_nb = [tuple(int(x) for x in solver(tuple(v))) for v in spec]

    def br(a, b):
        return sum(a[i] * B[i][j] * b[j]
                   for i in range(rank) for j in range(rank))

    Bn = [[br(nodes[i], nodes[j]) for j in range(rank)] for i in range(rank)]
    units = [tuple(1 if j == i else 0 for j in range(rank)) for i in range(rank)]
    return BPSKAlgebra(pairing=Bn, node_charges=units, spec=spec_nb, verify=verify)


def arrange_doublet_pairs_windowed(
    pairing: Sequence[Sequence[int]],
    spec: Sequence[Vec],
    f_index: int,
    *,
    max_states: int = 60000,
    margin: int = 2,
):
    """`arrange_doublet_pairs`, restricted to the contiguous window
    spanning the f-charged entries (plus a small neutral margin).
    Local moves inside a contiguous slice preserve the total ordered
    product, so the splice is S-exact.  Falls back to the global
    arranger when the window is the whole spec."""
    spec = [tuple(g) for g in spec]
    charged = [k for k, v in enumerate(spec) if v[f_index] != 0]
    if not charged:
        return list(spec)
    lo = max(0, min(charged) - margin)
    hi = min(len(spec), max(charged) + 1 + margin)
    if hi - lo >= len(spec):
        return arrange_doublet_pairs(pairing, spec, f_index,
                                     max_states=max_states)
    window = spec[lo:hi]
    arr = arrange_doublet_pairs(pairing, window, f_index,
                                max_states=max_states)
    if arr is None:
        # widen once to the full spec before giving up
        return arrange_doublet_pairs(pairing, spec, f_index,
                                     max_states=max_states)
    return spec[:lo] + list(arr) + spec[hi:]
