"""[A₁,A₂ₖ]: the zoo entries `pentagon` / `heptagon` through the closed-form
geometric class `a1a2k_kalg.A1A2kKAlg(k)`, k = 1, 2.

The ADE finite K-algebras are served here by
self-contained realizations, with no frozen trace data and no runtime oracle on
the serving path, fully functional (multiply and trace on every label to any
order).  `A1A2kKAlg(k)` is `[A₁,A₂ₖ]` on the diagonals of the `(2k+3)`-gon
(`curve(x, ℓ)`; its letter `(a, j)` is the diagonal from `j` to `j + a + 1`):
closed-form products (the quantum Ptolemy relations, from the per-k base table
that `A1A2k_plucker_closed_form.base_table_predict` computes from chord
geometry) and closed-form traces (Layer 1, then the M(2,2k+3) Andrews–Gordon
characters on the product side, `minimal_model_characters.char_product`), with
no bootstrap.  This module identifies the zoo standalone's canonical basis with
it and serves the zoo's seed traces from it (`elem_traces._seed_series`).
Until 2026-09-24 the zoo served the pentagon / heptagon seeds from the trivial-R
orthonormality bootstrap `elem_traces._generate_bootstrap`, with `Tr(1)` the
M(2,5) / M(2,7) vacuum character (`ad_characters.m2_2np3_character`); that
route is self-contained too (exact, from the class's own data and the
orthonormality axiom), and
it stays as the witness `elem_traces.generate` runs.  The closed forms are
preferred where they are cheap, and here they are.

The generator map, found at runtime (nothing is stored)
-------------------------------------------------------
The zoo's generators carry charges (`<PRE>_MULT_GENS_LATTICE`) and
`A1A2kKAlg`'s do not, so the map is found by structure, as in `a1d3_seeds`:
both classes have `k(2k+3)` generators in `k` ρ-orbits of length `2k+3` (the
zoo's `<PRE>_RHO_PERM`; the rotation `(a, j) ↦ (a, j + 1)`, one orbit per
diagonal length `a + 1`).  The candidates are the ρ-equivariant bijections
(which orbit goes where, and one cyclic shift per orbit): 5 for the pentagon,
2·7·7 = 98 for the heptagon.  A zoo word `((mg, p), …)` goes to the
`A1A2kKAlg` label `((a, j, p), …)` letter by letter (sorted), its coefficient
unchanged: the flavour is trivial, so there is no coefficient to move and no
offset to solve for.  A candidate is certified when every ordered pair of zoo
generators has its product carried over term for term onto `A1A2kKAlg`'s
product of the images, and the zoo's `rho_element` on every generator is
carried onto `A1A2kKAlg`'s ρ of the image (`product_agreement`,
`rho_agreement`).

Measured (2026-09-24): exactly one ρ-class of maps is certified — the map
composed with the `2k+3` powers of ρ.  Pentagon: all 5 candidates (they form a
single class); among all 120 bijections of the 5 generators exactly these 5
reproduce the 25 products, every other one at most 11.  Heptagon: 7 of the 98,
the zoo's orbit of generator 0 going onto the long diagonals `(2, j)`; every
other candidate reproduces at most 84 of the 196 products.  The served map is
the first in the fixed enumeration order.  The seed traces do not depend on the
member of the class: a seed trace is constant on its ρ-orbit, since ρ² generates
the same cyclic group as ρ on orbits of odd length `2k+3`, and the trivial-R
entries fold along ρ²-orbits (`elem_traces.fold_policy`).  If the certified set
is anything else — empty, or containing a map outside that class — the
construction raises.  The negative controls (every transposition of two
generators, the reflection of the polygon, the other candidates) are in
the suite in the source repository.

Depth and cost
--------------
Exact to any order: `A1A2kKAlg`'s trace of a generator is one Layer-1 step to
the ρ²-orbit seed and the Andrews–Gordon product, `O(K²)` integer operations.
Each series is cached at the largest order asked and truncated on smaller
requests (a truncation of an exact series is exact).

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from itertools import permutations, product

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent


__all__ = [
    "K_OF",
    "a1a2k_algebra",
    "generator_map",
    "product_agreement",
    "rho_agreement",
    "to_a1a2k_label",
    "to_a1a2k",
    "seeds",
]

#: zoo id -> `k` of `A1A2kKAlg(k)` (the `(2k+3)`-gon, `[A₁,A₂ₖ]`)
K_OF = {"pentagon": 1, "heptagon": 2}

_STATE: dict = {}         # k -> per-k state dict


def _k(short_id: str) -> int:
    if short_id not in K_OF:
        raise KeyError(
            f"aeven_seeds: {short_id!r} is not an [A1,A_2k] zoo entry "
            f"(expected one of {sorted(K_OF)})")
    return K_OF[short_id]


def _st(short_id: str) -> dict:
    k = _k(short_id)
    st = _STATE.get(k)
    if st is None:
        st = _STATE[k] = {"k": k, "sid": short_id}
    return st


def a1a2k_algebra(short_id: str):
    """The (shared, per-process) `A1A2kKAlg(k)` for this entry."""
    st = _st(short_id)
    if "A" not in st:
        from a1a2k_kalg import A1A2kKAlg
        st["A"] = A1A2kKAlg(st["k"])
    return st["A"]


def _zoo(short_id: str):
    st = _st(short_id)
    if "Z" not in st:
        from elem_traces import _standalone_algebra
        st["Z"] = _standalone_algebra(short_id)
    return st["Z"]


# ---------------------------------------------------------------------------
# term dictionaries
# ---------------------------------------------------------------------------

def _coeff_dict(c) -> dict:
    """`{𝖖-power: int}` of a coefficient over the trivial ring (`LaurentPoly`,
    or `RLaurent` whose only basis key is the unit)."""
    if isinstance(c, RLaurent):
        out: dict = {}
        for q, r in c.coeffs.items():
            for key, v in r.terms.items():
                if not v:
                    continue
                if key != c.ring.one_basis():
                    raise ValueError(f"aeven_seeds: a flavoured coefficient "
                                     f"{c!r} in a trivial-R entry")
                out[q] = out.get(q, 0) + v
        return {q: v for q, v in out.items() if v}
    return {q: v for q, v in c._coeffs.items() if v}


def _as_dict(x: Element) -> dict:
    """`{label: {𝖖-power: int}}` of an `Element`."""
    out = {}
    for lab, c in x.terms.items():
        d = _coeff_dict(c)
        if d:
            out[lab] = d
    return out


def _translate(gm: dict, x: Element) -> dict:
    """A zoo `Element` carried to `A1A2kKAlg`'s `{label: {𝖖-power: int}}`
    through the generator map `gm = {mg: (a, j)}`, letter by letter."""
    out: dict = {}
    for word, d in _as_dict(x).items():
        lab = tuple(sorted((gm[mg][0], gm[mg][1], p) for mg, p in word))
        tgt = out.setdefault(lab, {})
        for q, v in d.items():
            tgt[q] = tgt.get(q, 0) + v
    return {lab: {q: v for q, v in d.items() if v}
            for lab, d in out.items() if any(d.values())}


def _letter(g) -> tuple:
    """`A1A2kKAlg`'s label of its generator `g = (a, j)`."""
    return ((g[0], g[1], 1),)


def _orbits(gens, rho):
    seen, out = set(), []
    for g in gens:
        if g in seen:
            continue
        orbit = [g]
        seen.add(g)
        x = rho(g)
        while x != g:
            if x in seen or len(orbit) > len(gens):
                raise AssertionError(f"aeven_seeds: ρ is not a permutation "
                                     f"of the generators at {g!r}")
            orbit.append(x)
            seen.add(x)
            x = rho(x)
        out.append(orbit)
    return out


def _a1a2k_rho_gen(A, g) -> tuple:
    """`A1A2kKAlg`'s ρ on a generator `(a, j)`, read off its `rho` on labels."""
    ((a, j, _e),) = A.rho(_letter(g))
    return (a, j)


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------

def _products(short_id: str):
    st = _st(short_id)
    if "products" not in st:
        Z, A = _zoo(short_id), a1a2k_algebra(short_id)
        n = len(Z.cone_data().mult_gens())
        gens = list(A.cone_data().mult_gens())
        zp = {(a, b): Z.multiply(((a, 1),), ((b, 1),))
              for a in range(n) for b in range(n)}
        ap = {(x, y): _as_dict(A.multiply(_letter(x), _letter(y)))
              for x in gens for y in gens}
        st["products"] = (n, gens, zp, ap)
    return st["products"]


def product_agreement(short_id: str, gm: dict, *, stop_early=False) -> tuple:
    """`(agreeing, total)` over the ordered pairs of zoo generators: the zoo
    product, carried over by `gm = {mg: (a, j)}`, against `A1A2kKAlg`'s
    product of the images, term for term."""
    n, _gens, zp, ap = _products(short_id)
    ok = 0
    for a in range(n):
        for b in range(n):
            if _translate(gm, zp[(a, b)]) == ap[(gm[a], gm[b])]:
                ok += 1
            elif stop_early:
                return ok, n * n
    return ok, n * n


def rho_agreement(short_id: str, gm: dict) -> tuple:
    """`(agreeing, total)` over the zoo generators: the zoo's `rho_element`,
    carried over by `gm`, against `A1A2kKAlg`'s ρ of the image."""
    Z, A = _zoo(short_id), a1a2k_algebra(short_id)
    n = len(Z.cone_data().mult_gens())
    ok = 0
    for g in range(n):
        got = _translate(gm, Z.rho_element(Element({((g, 1),): LaurentPoly.one()})))
        ok += got == {A.rho(_letter(gm[g])): {0: 1}}
    return ok, n


def _rho_power(A, g, j: int):
    for _ in range(j):
        g = _a1a2k_rho_gen(A, g)
    return g


def generator_map(short_id: str) -> dict:
    """`{mg: (a, j)}`: the zoo generator `mg` is `A1A2kKAlg(k)`'s generator
    `(a, j)`, the diagonal `curve(j, a + 1)` from `j` to `j + a + 1` (module
    docstring).  Found at first use and cached; raises `AssertionError` unless
    the certified maps (every generator product term for term, ρ on every
    generator) are exactly one map composed with the powers of ρ."""
    st = _st(short_id)
    if "gm" in st:
        return st["gm"]
    Z, A = _zoo(short_id), a1a2k_algebra(short_id)
    n, gens, _zp, _ap = _products(short_id)
    H = A.H
    zo = _orbits(range(n), lambda g: Z._rho_perm.get(g, g))
    ao = _orbits(gens, lambda g: _a1a2k_rho_gen(A, g))
    if sorted(map(len, zo)) != sorted(map(len, ao)):
        raise AssertionError(
            f"aeven_seeds({short_id}): ρ-orbit profiles differ "
            f"({sorted(map(len, zo))} vs {sorted(map(len, ao))})")
    candidates = 0
    passing = []
    for pairing in permutations(ao):
        if [len(o) for o in pairing] != [len(o) for o in zo]:
            continue
        for shifts in product(*[range(len(o)) for o in zo]):
            candidates += 1
            gm = {g: aorb[(j + s) % len(aorb)]
                  for zorb, aorb, s in zip(zo, pairing, shifts)
                  for j, g in enumerate(zorb)}
            if (product_agreement(short_id, gm, stop_early=True)[0] == n * n
                    and rho_agreement(short_id, gm)[0] == n):
                passing.append(gm)
    if not passing:
        raise AssertionError(
            f"aeven_seeds({short_id}): none of the {candidates} ρ-equivariant "
            f"generator maps reproduces the generator products and ρ")
    first = passing[0]
    rho_class = [{g: _rho_power(A, h, j) for g, h in first.items()}
                 for j in range(H)]
    distinct = {tuple(sorted(m.items())) for m in rho_class}
    if (any(gm not in rho_class for gm in passing)
            or len(passing) != len(distinct)):
        raise AssertionError(
            f"aeven_seeds({short_id}): the certified generator maps are not "
            f"one map composed with the powers of ρ ({len(passing)} certified "
            f"of {candidates}); the seed traces would be ambiguous")
    st["class"] = passing
    st["candidates"] = candidates
    st["gm"] = first
    return first


# ---------------------------------------------------------------------------
# labels and elements
# ---------------------------------------------------------------------------

def to_a1a2k_label(short_id: str, word) -> tuple:
    """`A1A2kKAlg(k)`'s label of the zoo label `word = ((mg, p), …)`: the
    multiset of diagonals `((a, j, p), …)`, sorted."""
    gm = generator_map(short_id)
    return tuple(sorted((gm[mg][0], gm[mg][1], p) for mg, p in word))


def to_a1a2k(short_id: str, x: Element) -> Element:
    """A zoo `Element` (coefficients `LaurentPoly`, or `RLaurent` over the
    trivial ring) as an `A1A2kKAlg(k)` `Element`."""
    d = _translate(generator_map(short_id), x)
    return Element({lab: LaurentPoly(c) for lab, c in d.items()})


# ---------------------------------------------------------------------------
# the seed server
# ---------------------------------------------------------------------------

class _AevenSeeds:
    """The zoo's `Tr(1)` and seed traces for one entry, read off
    `A1A2kKAlg(k)`'s trace through the map, in the zoo's trivial-R data format
    `{𝖖-power: int}`.  Each series is cached at the largest order asked and
    truncated on smaller requests."""

    def __init__(self, short_id: str):
        self.short_id = short_id
        self.k = _k(short_id)
        self._cache: dict = {}

    def _series(self, key, label, K: int) -> dict:
        hit = self._cache.get(key)
        if hit is None or hit[0] < K:
            A = a1a2k_algebra(self.short_id)
            R = A.coefficient_ring()
            tr = A.trace(label, K)
            if tr.K < K:
                raise AssertionError(
                    f"{self.short_id}: A1A2kKAlg({self.k}) returned the trace "
                    f"of {label!r} through q^{tr.K} < q^{K}")
            data: dict = {}
            for q, c in tr.coeffs.items():
                if q > K:
                    continue
                if hasattr(c, "terms"):
                    bad = [kk for kk, v in c.terms.items()
                           if v and kk != R.one_basis()]
                    if bad:
                        raise ValueError(
                            f"{self.short_id}: flavoured trace coefficient "
                            f"{c!r} at q^{q}")
                    v = int(c.terms.get(R.one_basis(), 0))
                else:
                    v = int(c)
                if v:
                    data[q] = v
            hit = (K, data)
            self._cache[key] = hit
        return {q: v for q, v in hit[1].items() if q <= K}

    def vacuum_trace(self, K: int) -> dict:
        """`Tr(1)` through `𝖖^K`."""
        return self._series("identity", (), K)

    def seed_trace(self, i: int, K: int) -> dict:
        """`Tr(L_{((i, 1),)})` of the zoo generator `i` through `𝖖^K`."""
        gm = generator_map(self.short_id)
        if i not in gm:
            raise KeyError(f"{self.short_id}: no multiplicative generator {i!r}")
        return self._series(i, to_a1a2k_label(self.short_id, ((i, 1),)), K)


def seeds(short_id: str) -> _AevenSeeds:
    """The seed server for this entry: `.vacuum_trace(K)` and
    `.seed_trace(i, K)`, as `elem_traces._seed_series` reads them."""
    st = _st(short_id)
    if "seeds" not in st:
        st["seeds"] = _AevenSeeds(short_id)
    return st["seeds"]
