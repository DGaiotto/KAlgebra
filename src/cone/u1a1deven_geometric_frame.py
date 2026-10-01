"""
u1a1deven_geometric_frame.py
============================

The curve frame of the U(1)-gauged `[A_1, D_{2k+2}]`: the combinatorics, the
closed forms and the multiplication rule behind the public class
`u1a1deven_cone_kalgebra.U1A1DevenConeKAlgebra` (the frame became the public
class on 2026-09-24).
This module holds module-level functions and the class's cone data
(`_CurveConeData`); the labels, the trace, the bridges to the flow and the
convention that fixes the power of `E` are documented with the class.  The
provisional private class that used to live here was folded into
`U1A1DevenConeKAlgebra` at the switch.  Certified by the source
repository's tests.

(A) Combinatorics of curves, for any `n ≥ 3`
--------------------------------------------
The curves are those of `a1dn_kalg.A1DnKAlg` (its labels): a curve
`(x, ℓ)`, `x ∈ Z/n`, `2 ≤ ℓ ≤ n`, is the simple curve from marked point `x` to
`x + ℓ` with `ℓ` boundary edges on the side not containing the puncture; `ℓ = n`
is the loop at `x` around the puncture.  There are `n(n − 1)` of them.
(Dictionary: the draft notation `arc(g, u)` of the probes this frame was
measured with is the curve `(x, ℓ) = (u, g)`.)  `a1dodd_cone_data` computes
the same curves for odd `n = 2k + 3` as centrally symmetric chord pairs of the
`2n`-gon, keyed by `(a, p, i)`; the functions here take `n` and a curve instead:
`_curve_chords` (the chord pair; one diameter for `ℓ = n`), `_curve_crossings`
(0, 1 or 2 — two crossings are the ones that carry `χ₁` at odd `n`),
`_rotate_curve`, and `_maximal_cones` (maximal sets of pairwise non-crossing
curves).  At odd `n` they reproduce `arcs_cross` / `arc_puncture_crossing`
through `A1DnKAlg`'s dictionary, and the crossing count equals
`a1dn_kalg._arc_crossings` (which counts in the universal cover) at every `n`.

For even `n` a curve also has a magnetic charge (`_magnetic_charge`): a curve
of odd `ℓ` has endpoints of different parity and charge 0; a curve of even `ℓ`
has endpoints of one parity (`n` is even, so parity is defined mod `n`), and
charge `−1` if that is the parity of the merged vertex (below), `+1` if not.
It is the `X_{1,0}` charge `c0` of the curve's flow label, and the charge
`UngaugedKAlgebra.mag` reads off the `E`-commutator is `−2·c0` (measured on
every curve at k = 1).  `U1A1AoddKAlg` has the same endpoint-parity rule
relative to its merged vertex `H − 1` (checked on its `_phi_letter_data` and
`_letter_mag`, k = 1..3: `c0` as above, and its magnetic charge `−2·c0`).

`_rho_equivariant_isomorphisms` is a pure-Python search (no `networkx`) for
the bijections between two graphs that carry adjacency to adjacency and
intertwine a given permutation on each side.  With it, the curves of the
`(2k+2)`-gon — adjacency = not crossing, permutation = `x ↦ x + 1` — were
found isomorphic to the rays of the table presentation this frame replaced
(retired 2026-09-24) with the
rays that are products of two rays (times a power of `X_{0,1}`) removed —
adjacency = `𝖖`-commuting, permutation = the tables' ρ on rays — at k = 1, 2,
3 and, held out, at k = 4; every removed ray is the product of a ray of
magnetic charge `+1` and one of `−1` (4 / 21 / 64 / 145 of 16 / 51 / 120 /
235 rays at k = 1..4).  That comparison is now a regression test of the
source repository.  The maximal cones are
simplicial: `C(4k+2, 2k+1)` cones of `2k + 1` curves each.

(B) The flow's RG image in closed form
--------------------------------------
The flow `U1A1DevenViaDoddRG(k)` maps into `A1DoddConeKAlg(k − 1) ⊗ QT(Z²)`,
whose `A1Dodd` factor is labelled by the curves of the `(2k+1)`-gon with one
interior puncture (`A1DnKAlg(2k+1)`'s dictionary `_arc_to_ap`).  Its RG image
`Φ` of each curve is the **vertex merge** of `U1A1AoddKAlg`'s embedding, one
polygon smaller: the marked point `_MERGED_VERTEX = 2` of the `(2k+2)`-gon is
removed, its two boundary edges becoming one edge of the `(2k+1)`-gon, and the
other marked points are relabelled by

    π(v) = (v if v < 2 else v − 1) + S  (mod 2k + 1),

where `S` is the start point of the flow's dressing letter `L = (k, 1, 0)` read
as a curve: `L = (S, 2)` with `S = _ap_base(k, 1, k − 1) mod (2k + 1)`
(`1, 0, 5, 7, 8` at `k = 1..5`, read from `a1dodd_cone_data._ap_base`, not
fitted).  So `L = Φ((0, 3))`, the image of the curve that cuts off the marked
points 1 and 2.  With `c0` the curve's magnetic charge (A):

  * a curve with no endpoint at the merged vertex: ONE term, the curve with
    the same endpoints (one boundary edge fewer if the merged vertex lies on
    its puncture-free side), at `X_{(c0, 0)}`;
  * a curve with one endpoint at the merged vertex (the analogue of
    `U1A1AoddKAlg`'s seam chords): TWO terms, that endpoint moved back to
    `π(1)` at `X_{(c0, 0)}` and forward to `π(3)` at `X_{(c0, 1)}`;
  * the loop at the merged vertex: FOUR terms, the loop at `π(1)` at
    `X_{(c0, 0)}`, `(𝖖⁻¹ + 𝖖)·(π(3), 2k)` and `χ₁` at `X_{(c0, 1)}`, the loop
    at `π(3)` at `X_{(c0, 2)}`;

every coefficient 1 but the `𝖖⁻¹ + 𝖖`, and a curve of the `(2k+1)`-gon with
one boundary edge (`ℓ = 1`) is the unit.  The term of lowest `E`-power has
`E`-power 0 and coefficient 1 — the normalisation of `E` chosen for this frame
(the position it singles out is the public convention, see
`U1A1DevenConeKAlgebra`); it is the curve's flow label.  ρ is

    ρ((x, ℓ)) = E^d · (x + 1, ℓ),   ρ(E) = E⁻¹,
    d = −#{endpoints of (x, ℓ) at marked point 1}  (the loop at 1 counts two).

Both are exact against the flow on every curve at `k = 1..5`, and fail with
`S + 1`, with the two terms of a merged-vertex curve at swapped `E`-powers, and
with the drift at marked point 3.

(C) The class — see `u1a1deven_cone_kalgebra.U1A1DevenConeKAlgebra`
--------------------------------------------------------------------
Labels `(curves, e, κ)`, `Φ` of a label, the flow-free trace and pairing
through `u1a1deven_trace_transport.DevenTraceTransport`, and the bridges to
the flow's labels.

(D) Multiplication in the curve frame
-------------------------------------
`U1A1DevenConeKAlgebra.multiply` runs the generic cone-monomial reducer
(`ConeData.derived_multiply`) over `_CurveConeData`, on the `χ`-stripped
labels `(curves, e)` with the `SU(2)` character in the coefficient (the
class then moves it into the label, as `A1DoddConeKAlg` does): letters the
curves and `E^{±1}` (`E` a torus letter), cones the maximal non-crossing sets
of curves with `E^{±1}`.  Its two tables (`E^{±1}` `𝖖`-commutes with every
curve, so it enters only the first):

  * **cocycle** (`_frame_cocycle`, closed form): `L_g·L_h = 𝖖^c·L_{g+h}` for
    non-crossing `g, h`, with `c` the `A1Dodd` arc cocycle of the two curves'
    lowest terms (0 when one of them is the unit), and `c(curve, E^{±1}) =
    ±c0(curve)`.  Derived from (B): `Φ` is multiplicative and its lowest term
    is a single label with coefficient 1, so the cocycle of two labels is the
    cocycle of their lowest terms, and `X_{(c0,0)}·X_{(0,1)} = 𝖖^{c0}·X_{(c0,1)}`.
  * **cross products of two crossing curves**, two routes that agree:
    - DERIVED (`U1A1DevenConeKAlgebra._cross_derived`): `Φ(g)·Φ(h)` in the
      auxiliary algebra, peeled (`U1A1DevenConeKAlgebra._peel`) by its
      lowest-`E` terms — each such term is the lowest term of exactly one
      label, whose curves must be a skein resolution of `g, h` (below; the
      candidates), its `E`-power the term's `c1` and its coefficient (with
      its `χ`) the term's; that label's image is subtracted, and the peel
      repeats until the residual is exactly zero.  A lowest term naming no
      candidate, or a residual that does not vanish, raises `ValueError`;
      nothing is fitted.
    - ANALYTIC (`_analytic_product`): the `a1dodd_skein` model at even `n`.
      The resolutions (`_resolutions`) are those of that model on the
      `2n`-gon with central symmetry: for two curves that are not loops, the
      centrally symmetric states of `a1dodd_skein._skein_states` (two states
      for one crossing, the ordinary Ptolemy resolutions; four for two
      crossings); for a loop and a curve, the FZ double resolution (the two
      pure daughters, `a1dodd_cone_data._resolve_crossing` of the diameter with
      one chord of the curve, completed centrally) and the χ₁ daughter of the
      doubled-diameter puncture loop, which is the pair of arcs from the
      curve's endpoints to the loop's marked point on the curve's
      puncture-free side (`_loop_fork_arcs`: at odd `n` it is the model's
      `χ₁` word on every diameter × non-diameter crossing, k = 0..3); for two
      loops, `P² + χ₁·PQ + Q²` with `P`, `Q` the two
      reconnections of the diameters.  A state is folded into a label: its
      arcs become curves, its boundary edges are units except the edge from
      marked point 1 to the merged vertex, each copy of which is one `E`, and
      a loop around the puncture is `χ₁`.  The `𝖖`-power of each term is the
      model's: the bulk rule `Σ_f c(g, f)` over the factors `f` of the daughter
      (`E` included) for one crossing and for two loops; `ε·Σ_f c(loop, f)`
      for a loop and a curve (`ε = +1` when the loop is the left factor, `−1`
      when it is the right one); and `A + 1 − #B` for two crossings of two
      curves that are not loops, `#B` the model's count of crossing classes
      smoothed the `B` way and `A` the bulk value of the state that carries
      `χ₁`.  (In the model at odd `n` that anchor is `arc_cocycle(g, h)`, and
      it equals the bulk value of the `χ₁` state on every such crossing, 20 /
      140 / 504 at `k = 1, 2, 3`; the fold formula of `arc_cocycle` has no
      even-`n` analogue.)  Among rules that give each boundary edge `{x, x+1}`
      of a state a weight `E^{a_x}`, commuting with ρ (drift at marked point
      1, `ρ(E) = E⁻¹`; the resolution keeps the multiset of endpoints)
      requires `a_{x+1} = −a_x + [x ∈ {0, 1}]`, whose solutions are `a_0 = t`,
      `a_1 = 1 − t`, `a_x = (−1)^x·t` for `x ≥ 2`; the measured `E`-powers are
      the solution `t = 0`.
    `_CurveConeData.cross_product` uses the analytic route wherever it
    applies (`_analytic_product` returns `None` where it does not: a state
    that does not fold, a bulk rule whose factor crosses its curve, or two
    crossings without exactly one `χ₁` state; this happens on no crossing
    pair at k = 1..5) and the derived route otherwise;
    `U1A1DevenConeKAlgebra(k, route="derived")` uses the derived route
    throughout.

`U1A1DevenConeKAlgebra._multiply_via_phi(a, b)` is the derived peel of
`Φ(a)·Φ(b)` for any two labels, with no reducer: the independent cross-check
of the reducer on composite labels (for two single crossing curves its
candidates are the skein resolutions; otherwise the lowest terms name their
labels through the inverse label map).

The letters `E^{±1}` of `_CurveConeData` are the integer pairs `(−1, ±1)`
(marked point −1 does not exist, so they cannot collide with a curve), so a
list of letters sorts.
"""
from __future__ import annotations

import os
import sys

# APPEND, never insert(0) (the hazard recorded in u1a1aodd_kalg.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import functools

from cone_data import FiniteConeData, Cone
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent
from a1dodd_cone_data import _chord_cross, _ap_base, arc_cocycle, _resolve_crossing
from a1dn_kalg import _arc_to_ap


# ===========================================================================
# (A) curves of the n-gon with one interior puncture, any n >= 3
# ===========================================================================

def _curves(n: int):
    """The `n(n − 1)` curves `(x, ℓ)`, `x ∈ Z/n`, `2 ≤ ℓ ≤ n`."""
    return [(x, l) for x in range(n) for l in range(2, n + 1)]


def _curve_chords(c, n: int):
    """The centrally symmetric chord set of the curve `c = (x, ℓ)` on the
    `2n`-gon: the chord `{x, x + ℓ}` and its image under the half-turn
    `v ↦ v + n`, or the single diameter `{x, x + n}` when `ℓ = n` — the
    `a1dodd_cone_data._ap_chords` model at any `n`."""
    x, l = c
    N = 2 * n
    u, v = x % N, (x + l) % N
    if l == n:
        return (tuple(sorted((u, v))),)
    return (tuple(sorted((u, v))), tuple(sorted(((u + n) % N, (v + n) % N))))


def _curve_crossings(c, d, n: int) -> int:
    """How many times the curves `c`, `d` cross (0, 1 or 2; `c == d` gives
    0), from their chords on the `2n`-gon (`_curve_chords`): two chord pairs
    have 0, 2 or 4 strictly interleaving chord pairs (the half-turn doubles
    each crossing), counted as 0, 1, 2; a loop and a chord pair have 0 or 2;
    two distinct loops always cross, at the puncture, which counts 2.  Two is
    `a1dodd_cone_data.arc_puncture_crossing` (one chord crossing every chord
    of the other), the crossing that carries `χ₁` at odd `n`."""
    if tuple(c) == tuple(d):
        return 0
    N = 2 * n
    cc, dd = _curve_chords(c, n), _curve_chords(d, n)
    k = sum(1 for p in cc for q in dd if _chord_cross(p, q, N))
    if len(cc) == 1 and len(dd) == 1:
        return 2 if k else 0
    return k if (len(cc) == 1 or len(dd) == 1) else k // 2


def _curves_cross(c, d, n: int) -> bool:
    """Do the curves `c`, `d` cross?  (Non-crossing ⟺ `𝖖`-commuting.)"""
    return _curve_crossings(c, d, n) > 0


def _rotate_curve(c, n: int, s: int = 1):
    """The rotation `x ↦ x + s` of a curve."""
    x, l = c
    return ((x + s) % n, l)


def _compatibility(n: int) -> dict:
    """`{curve: frozenset of the curves that do not cross it}`."""
    cs = _curves(n)
    return {c: frozenset(d for d in cs if d != c and not _curves_cross(c, d, n))
            for c in cs}


def _maximal_cliques(vertices, nb) -> list:
    """Maximal cliques of the graph `nb` (Bron–Kerbosch with pivoting)."""
    out = []

    def bk(R, P, X):
        if not P and not X:
            out.append(R)
            return
        pivot = max(P | X, key=lambda u: len(P & nb[u]))
        for v in list(P - nb[pivot]):
            bk(R | {v}, P & nb[v], X & nb[v])
            P = P - {v}
            X = X | {v}

    bk(frozenset(), frozenset(vertices), frozenset())
    return out


def _maximal_cones(n: int) -> tuple:
    """The maximal sets of pairwise non-crossing curves of the `n`-gon."""
    return tuple(_maximal_cliques(_curves(n), _compatibility(n)))


def _magnetic_charge(c, n: int, merged: int) -> int:
    """For even `n`: 0 if `ℓ` is odd; else `−1` if the endpoints have the
    parity of the marked point `merged`, `+1` if not (module docstring, A)."""
    if n % 2:
        raise ValueError(f"_magnetic_charge: n = {n} is odd (parity is not "
                         f"defined mod n)")
    x, l = c
    if l % 2:
        return 0
    return -1 if (x - merged) % 2 == 0 else 1


def _perm_orbits(vertices, perm) -> list:
    """The cycles of the permutation `perm` of `vertices`, each listed from
    its first vertex in `vertices` order along `perm`."""
    seen, orbits = set(), []
    for v in vertices:
        if v in seen:
            continue
        orb = [v]
        seen.add(v)
        w = perm[v]
        while w != v:
            if w in seen:
                raise ValueError("_perm_orbits: not a permutation")
            orb.append(w)
            seen.add(w)
            w = perm[w]
        orbits.append(orb)
    return orbits


def _rho_equivariant_isomorphisms(V1, adj1, rho1, V2, adj2, rho2, limit=None):
    """All bijections `f: V1 → V2` with `f∘rho1 = rho2∘f` and `u ~ w ⟺
    f(u) ~ f(w)` (adjacency `adj*`: vertex → set of neighbours, symmetric),
    up to `limit` of them.  Pure Python: an equivariant `f` is fixed on each
    `rho1`-cycle by the image of one vertex, so the search assigns cycles to
    `rho2`-cycles of the same length, one phase at a time, and checks
    adjacency against everything assigned so far."""
    orb1 = _perm_orbits(list(V1), rho1)
    orb2 = _perm_orbits(list(V2), rho2)
    if sorted(map(len, orb1)) != sorted(map(len, orb2)):
        return []
    if len(V1) != len(V2):
        return []
    # most-constrained cycles first
    orb1.sort(key=lambda o: -len(adj1[o[0]]))
    sols = []
    f = {}

    def consistent(new):
        for u, fu in new:
            a1, a2 = adj1[u], adj2[fu]
            for w, fw in f.items():
                if (w in a1) != (fw in a2):
                    return False
        return True

    def rec(i, used):
        if limit is not None and len(sols) >= limit:
            return
        if i == len(orb1):
            sols.append(dict(f))
            return
        O = orb1[i]
        L = len(O)
        for j, P in enumerate(orb2):
            if j in used or len(P) != L:
                continue
            for s in range(L):
                new = [(O[t], P[(t + s) % L]) for t in range(L)]
                if any(len(adj1[u]) != len(adj2[fu]) for u, fu in new):
                    continue
                # adjacency inside the cycle, then against the assigned part
                ok = all((O[a] in adj1[O[b]]) == (P[(a + s) % L] in adj2[P[(b + s) % L]])
                         for a in range(L) for b in range(a + 1, L))
                if not ok or not consistent(new):
                    continue
                for u, fu in new:
                    f[u] = fu
                rec(i + 1, used | {j})
                for u, _ in new:
                    del f[u]

    rec(0, frozenset())
    return sols


# ===========================================================================
# (B) the flow's RG image and rho, in closed form
# ===========================================================================

# The marked point of the (2k+2)-gon that the flow merges (module docstring,
# B).  A labelling choice on the (2k+2)-gon side; S below is then read from
# `_ap_base`, not fitted.
_MERGED_VERTEX = 2

_ONE = LaurentPoly.one()
_Q_PLUS_QINV = LaurentPoly({-1: 1, 1: 1})


def _dressing_start(k: int) -> int:
    """`S`: the start point of the flow's dressing letter `L = (k, 1, 0)` of
    `A1DoddConeKAlg(k − 1)` as a curve `(S, 2)` of the `(2k+1)`-gon
    (`A1DnKAlg`'s dictionary: `S = _ap_base(k, 1, k − 1) mod (2k + 1)`)."""
    return _ap_base(k, 1, k - 1) % (2 * k + 1)


def _vertex_map(v: int, k: int) -> int:
    """`π`: marked point `v ≠ 2` of the `(2k+2)`-gon ↦ marked point of the
    `(2k+1)`-gon (module docstring, B)."""
    n = 2 * k + 2
    m = _MERGED_VERTEX
    v %= n
    if v == m:
        raise ValueError("_vertex_map: the merged vertex has no image")
    return ((v if v < m else v - 1) + _dressing_start(k)) % (n - 1)


def _charge(c, k: int) -> int:
    """The magnetic charge `c0` of the curve `c` of the `(2k+2)`-gon."""
    return _magnetic_charge(c, 2 * k + 2, _MERGED_VERTEX)


def _phi_curve_terms(c, k: int):
    """`Φ(c)` as a list of `(curve of the (2k+1)-gon or None, κ, (c0, c1),
    coefficient)`; `None` is the empty word (a boundary edge of the
    `(2k+1)`-gon, or the `χ₁` term, `κ = 1`)."""
    n = 2 * k + 2
    nd = n - 1
    m = _MERGED_VERTEX
    x, l = c
    x %= n
    if not (2 <= l <= n):
        raise ValueError(f"curve {c!r}: need 2 ≤ ℓ ≤ {n}")
    pi = lambda v: _vertex_map(v, k)

    def dc(y, g):
        return None if g == 1 else (y, g)

    c0 = _charge((x, l), k)
    v = (x + l) % n
    if l == n and x == m:                       # the loop at the merged vertex
        return [(dc(pi(m - 1), nd), 0, (c0, 0), _ONE),
                (dc(pi(m + 1), nd - 1), 0, (c0, 1), _Q_PLUS_QINV),
                (None, 1, (c0, 1), _ONE),
                (dc(pi(m + 1), nd), 0, (c0, 2), _ONE)]
    if x == m:                                  # starts at the merged vertex
        return [(dc(pi(m - 1), l), 0, (c0, 0), _ONE),
                (dc(pi(m + 1), l - 1), 0, (c0, 1), _ONE)]
    if v == m and l < n:                        # ends at the merged vertex
        return [(dc(pi(x), l - 1), 0, (c0, 0), _ONE),
                (dc(pi(x), l), 0, (c0, 1), _ONE)]
    inside = l == n or any((x + s) % n == m for s in range(1, l))
    return [(dc(pi(x), l - (1 if inside else 0)), 0, (c0, 0), _ONE)]


def _dodd_word(d, k: int):
    """A curve of the `(2k+1)`-gon (or None) as an `A1DoddConeKAlg(k − 1)`
    word."""
    return () if d is None else ((_arc_to_ap(d, k - 1), 1),)


def _phi_curve(c, k: int) -> dict:
    """`Φ(c)`: `{auxiliary label ((word, κ), (c0, c1)): LaurentPoly}`."""
    out = {}
    for d, kap, qt, co in _phi_curve_terms(c, k):
        lab = ((_dodd_word(d, k), kap), qt)
        out[lab] = out.get(lab, LaurentPoly({})) + co
    return {l: co for l, co in out.items() if not co.is_zero()}


def _lowest_term(c, k: int):
    """The term of lowest `E`-power of `Φ(c)`: `(curve of the (2k+1)-gon or
    None, c0)` — `E`-power 0, `κ = 0`, coefficient 1."""
    (d, kap, (c0, c1), co), = [t for t in _phi_curve_terms(c, k) if t[2][1] == 0]
    assert kap == 0 and co == _ONE
    return d, c0


def _rho_curve(c, k: int):
    """`ρ(c) = E^d·(x + 1, ℓ)`: returns `((x + 1, ℓ), d)`."""
    n = 2 * k + 2
    x, l = c
    d = -sum(1 for e in (x % n, (x + l) % n) if e == (_MERGED_VERTEX - 1) % n)
    return ((x + 1) % n, l), d


def _compositions(m: int, parts: int):
    """All tuples of `parts` non-negative ints summing to `m`."""
    if parts == 0:
        if m == 0:
            yield ()
        return
    if parts == 1:
        yield (m,)
        return
    for first in range(m + 1):
        for rest in _compositions(m - first, parts - 1):
            yield (first,) + rest


# ===========================================================================
# (D) multiplication: cocycle, skein resolutions, the analytic rule, cone data
# ===========================================================================

# The gauge letter E = X_{0,1} and its inverse as letters of `_CurveConeData`:
# the integer pairs (−1, ±1) — marked point −1 does not exist, so they cannot
# collide with a curve (x, ℓ), 0 ≤ x < n, and a list of letters sorts.
_E_POS = (-1, 1)
_E_NEG = (-1, -1)
_E_LETTERS = (_E_POS, _E_NEG)

# The boundary edge {1, 2} of the (2k+2)-gon — from marked point 1 to the
# merged vertex — named by its first marked point: each copy of it in a
# resolved state is one E (module docstring, D).
_E_EDGE = _MERGED_VERTEX - 1


def _su2_fuse(a: int, b: int):
    """Clebsch–Gordan: the highest weights in `χ_a·χ_b`."""
    return range(abs(a - b), a + b + 1, 2)


def _letter_key(g):
    """Order of the letters in a cone: curves (sorted), then `E`, `E⁻¹`."""
    return (1, -g[1]) if g in _E_LETTERS else (0, g)


@functools.lru_cache(maxsize=None)
def _frame_cocycle(g, h, k: int) -> int:
    """`c(g, h)` with `L_g·L_h = 𝖖^c·L_{g+h}`, for two `𝖖`-commuting letters of
    the curve frame (curves and `E^{±1}`; module docstring, D): 0 for `g = h`
    or two `E` letters; `c(curve, E^{±1}) = ±c0(curve)`; for two non-crossing
    curves the `A1Dodd` arc cocycle of their lowest terms (0 when either is the
    unit or they coincide).  `ValueError` on two crossing curves."""
    if g == h:
        return 0
    ge, he = g in _E_LETTERS, h in _E_LETTERS
    if ge and he:
        return 0
    if he:
        return h[1] * _charge(g, k)
    if ge:
        return -g[1] * _charge(h, k)
    if _curves_cross(g, h, 2 * k + 2):
        raise ValueError(f"_frame_cocycle: the curves {g} and {h} cross")
    dg, _ = _lowest_term(g, k)
    dh, _ = _lowest_term(h, k)
    if dg is None or dh is None or dg == dh:
        return 0
    return arc_cocycle(_arc_to_ap(dg, k - 1), _arc_to_ap(dh, k - 1), k - 1)


def _fold_arcs(arcs, n: int):
    """A centrally symmetric multiset of arcs of the `2n`-gon → `(curves, e)`:
    a curve `(x, ℓ)` is two arcs (its chord and the chord's half-turn image; a
    loop, `ℓ = n`, the two lifts of its diameter), a boundary edge two arcs;
    `e` counts the copies of the edge `{_E_EDGE, _E_EDGE + 1}`, and the other
    edges are units.  `None` if an arc joins a marked point to itself or a
    count is odd."""
    N = 2 * n
    cnt: dict = {}
    e2 = 0
    for arc in arcs:
        u, v = sorted(arc)
        if u == v:
            return None
        d = (v - u) % N
        if d in (1, N - 1):
            e2 += ((u if d == 1 else v) % n == _E_EDGE)
            continue
        key = (u % n, d) if d <= n else (v % n, N - d)
        cnt[key] = cnt.get(key, 0) + 1
    if e2 % 2 or any(m % 2 for m in cnt.values()):
        return None
    return tuple(sorted((c, m // 2) for c, m in cnt.items())), e2 // 2


def _resolutions(g, h, k: int):
    """The skein resolution of two crossing curves on the `a1dodd_skein`
    model at `n = 2k + 2` (module docstring, D): `(kind, states)`, each state
    `(curves, e, chi, nB)` — the folded label (`_fold_arcs`), `chi = 1` on the
    state that carries `χ₁`, and `nB` the model's count of crossing classes
    smoothed the `B` way (`None` where the model's rule does not use it).
    `kind` is `"one crossing"`, `"two crossings"` (two curves that are not
    loops), `"loop and curve"`, `"curve and loop"` (by operand order) or
    `"two loops"`; it is `None`, with no states, when a state does not fold.
    `ValueError` if the curves do not cross."""
    from a1dodd_skein import _skein_states
    n = 2 * k + 2
    N = 2 * n
    cr = _curve_crossings(g, h, n)
    if cr == 0:
        raise ValueError(f"_resolutions: the curves {g} and {h} do not cross")
    gl, hl = g[1] == n, h[1] == n
    states = []
    if not gl and not hl:
        cg, ch = _curve_chords(g, n), _curve_chords(h, n)
        for arcs, npunct, ncontr, nB in _skein_states(list(cg) + list(ch), len(cg), N, n):
            f = None if ncontr else _fold_arcs(arcs, n)
            if f is None:
                return None, []
            states.append((f[0], f[1], 1 if npunct else 0, nB))
        return ("one crossing" if cr == 1 else "two crossings"), states

    def central(c):
        return ((c[0] + n) % N, (c[1] + n) % N)

    if gl and hl:
        # the two reconnections P, Q of the diameters {u, u+n}, {v, v+n}
        u, v = g[0] % n, h[0] % n
        P = [(u, v), central((u, v))]
        Q = [(u, v + n), central((u, v + n))]
        for arcs, chi in ((P + P, 0), (P + Q, 1), (Q + Q, 0)):
            f = _fold_arcs(arcs, n)
            if f is None:
                return None, []
            states.append((f[0], f[1], chi, None))
        return "two loops", states
    loop, c = (g, h) if gl else (h, g)
    # the two pure daughters: the FZ double resolution of the diameter with one
    # chord of the curve, completed by the half-turn
    (Dc,) = _curve_chords(loop, n)
    dc = next(x for x in _curve_chords(c, n) if _chord_cross(Dc, x, N))
    for reso in _resolve_crossing(Dc, dc, N):
        f = _fold_arcs(list(reso) + [central(x) for x in reso], n)
        if f is None:
            return None, []
        states.append((f[0], f[1], 0, None))
    f = _fold_arcs(_loop_fork_arcs(loop, c, n), n)
    if f is None:
        return None, []
    states.append((f[0], f[1], 1, None))
    return ("loop and curve" if gl else "curve and loop"), states


def _loop_fork_arcs(loop, c, n: int):
    """The `χ₁` daughter of a loop `(x, n)` crossing a curve `c = (a, ℓ)` of the
    `n`-gon (any `n`), as arcs of the `2n`-gon: the arcs from the curve's
    endpoints `a`, `a + ℓ` to the loop's marked point `X`, which lies on the
    curve's puncture-free side (`a < X < a + ℓ`, lifted), with their half-turn
    images.  At odd `n` this is the `χ₁` word of `a1dodd_skein`'s
    doubled-diameter puncture loop (`_doubled_diam_chi_words`), checked on
    every diameter × non-diameter crossing at k = 0..3 of `a1dodd_cone_data`."""
    N = 2 * n
    a, l = c
    X = next(x for x in (loop[0] % n + t * n for t in range(-1, 3)) if a < x < a + l)
    arcs = []
    for s, t in ((a, X), (X, a + l)):
        arcs += [(s % N, t % N), ((s + n) % N, (t + n) % N)]
    return arcs


def _bulk(u, curves, e: int, k: int):
    """The bulk rule `Σ_f c(u, f)` over the factors of the label `(curves, e)`
    (`E` included), or `None` if `u` crosses one of the curves."""
    n = 2 * k + 2
    s = 0
    for c, m in curves:
        if c != u and _curves_cross(u, c, n):
            return None
        s += m * _frame_cocycle(u, c, k)
    return s + e * _frame_cocycle(u, _E_POS, k)


def _analytic_product(g, h, k: int):
    """The analytic rule (module docstring, D): `L_g·L_h` for two crossing
    curves as `{(curves, e): {κ: LaurentPoly}}`, from `_resolutions` and the
    `a1dodd_skein` model's `𝖖`-powers — the bulk rule (one crossing, two
    loops), `ε·` the loop's bulk rule (a loop and a curve), `A + 1 − #B` with
    `A` the bulk value of the `χ₁` state (two crossings).  `None` where it
    does not apply: a state that does not fold, a bulk rule whose factor
    crosses its curve, or not exactly one `χ₁` state."""
    kind, states = _resolutions(g, h, k)
    if kind is None:
        return None
    terms = []
    if kind in ("one crossing", "two loops"):
        for curves, e, chi, _nb in states:
            q = _bulk(g, curves, e, k)
            if q is None:
                return None
            terms.append((curves, e, chi, q))
    elif kind == "two crossings":
        anchor = [s for s in states if s[2]]
        if len(anchor) != 1:
            return None
        A = _bulk(g, anchor[0][0], anchor[0][1], k)
        if A is None:
            return None
        for curves, e, chi, nB in states:
            terms.append((curves, e, chi, A + 1 - nB))
    else:
        loop, eps = (g, 1) if kind == "loop and curve" else (h, -1)
        for curves, e, chi, _nb in states:
            q = _bulk(loop, curves, e, k)
            if q is None:
                return None
            terms.append((curves, e, chi, eps * q))
    out: dict = {}
    for curves, e, chi, q in terms:
        byk = out.setdefault((curves, e), {})
        byk[chi] = byk.get(chi, LaurentPoly({})) + LaurentPoly({q: 1})
    return out


class _CurveConeData(FiniteConeData):
    """Cone data of `U1A1DevenConeKAlgebra(k)` (module docstring D): letters
    the curves and `E^{±1}` (torus letters), cones the maximal non-crossing
    sets of curves with `E^{±1}`, native labels the `χ`-stripped
    `(curves, e)` (the class's labels are `(curves, e, κ)`; like
    `A1DoddConeKAlg`'s word-level cone data, this one carries `χ_κ` in
    `RLaurent` coefficients over `R(SU(2))`, and the class moves it into the
    label), cocycle `_frame_cocycle`, cross products
    `U1A1DevenConeKAlgebra._cross` in the literal convention (canonical
    `𝖖`-power plus the daughter's `cone_label_phase`).  `frame` is the
    algebra (it supplies `k`, `n`, the curves, the ring and `_cross`)."""

    def __init__(self, frame):
        self._F = frame
        self.k = frame.k
        self._n = frame.n
        self._R = frame.coefficient_ring()
        self._gens = tuple(sorted(frame._curve_set)) + _E_LETTERS
        self._cross_cache: dict = {}
        self._cones = None

    def coefficient_ring(self):
        return self._R

    def mult_gens(self):
        return self._gens

    def cones(self):
        if self._cones is None:
            E = frozenset(_E_LETTERS)
            self._cones = tuple(frozenset(C) | E for C in _maximal_cones(self._n))
        return self._cones

    def iter_cones(self):
        E = frozenset(_E_LETTERS)
        for C in self.cones():
            yield Cone(self, C, torus_gens=E)

    def q_commute(self, g, h):
        if g == h or g in _E_LETTERS or h in _E_LETTERS:
            return True
        return not _curves_cross(g, h, self._n)

    def cocycle(self, g, h):
        return _frame_cocycle(g, h, self.k)

    def cross_product(self, g, h):
        hit = self._cross_cache.get((g, h))
        if hit is not None:
            return hit
        if self.q_commute(g, h):
            raise ValueError(f"cross_product on the 𝖖-commuting letters {g}, {h}")
        R = self._R
        out = []
        for (curves, e), byk in sorted(self._F._cross(g, h).items()):
            gens, powers = self.to_cone_label((curves, e))
            phase = self.cone_label_phase(gens, powers)
            co: dict = {}
            for kap, lp in byk.items():
                for q, v in lp._coeffs.items():
                    if v:
                        t = R.basis_element(kap) * v
                        co[q + phase] = co[q + phase] + t if q + phase in co else t
            out.append((RLaurent(R, co), self._cone_label_to_word(gens, powers)))
        out = tuple(out)
        self._cross_cache[(g, h)] = out
        return out

    def canonical_cone_order(self, gens):
        return tuple(sorted(gens, key=_letter_key))

    def to_cone_label(self, native_label):
        curves, e = native_label
        powers = {c: m for c, m in curves if m}
        if e > 0:
            powers[_E_POS] = e
        elif e < 0:
            powers[_E_NEG] = -e
        return frozenset(powers), powers

    def from_cone_label(self, gens, powers):
        curves = tuple(sorted((g, powers[g]) for g in gens
                              if g not in _E_LETTERS and powers.get(g, 0) > 0))
        return (curves, powers.get(_E_POS, 0) - powers.get(_E_NEG, 0))

    def _torus_inverse_letter(self, g):
        if g == _E_POS:
            return _E_NEG
        if g == _E_NEG:
            return _E_POS
        return None

    def cycle_period_bound(self) -> int:
        return 2 * self._n
