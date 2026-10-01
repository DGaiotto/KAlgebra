"""[A₁,D₃]: the zoo entry `a1d3` through the closed-form cone class
`a1dodd_kalg.A1DoddConeKAlg(0)`.

The ADE finite K-algebras are served here by
self-contained realizations, with no frozen trace data and no runtime oracle on
the serving path.  `A1DoddConeKAlg(0)` is `[A₁,D₃]` in the `(a, p, i)` frame of
the once-punctured triangle: closed-form products (`a1dodd_cone_data(0)`, the
arc rules) and closed-form traces (`a1dodd_layer2`: `Tr(1)` the sl(2) vacuum
character, a single generator `seed_trace_ap(0, a, p)`), over `R(SU(2))` with
the SU(2) highest weight κ in the label.  Until 2026-09-23 the zoo served a1d3
from the SU(2) orthonormality bootstrap `su2_bootstrap.generate_su2`, which
reads a never-solved lower-order value as 0 and at
`K = 30` stops "inconsistent at k=23"; it stays as a witness.

Why `A1DoddConeKAlg(0)` and not `A1DnKAlg(3)`: `A1DnKAlg(3)` computes its
products, ρ and trace BY DELEGATION to `A1DoddConeKAlg(0)`, through the
bijection `(a, p, i) ↦ (x, ℓ)` onto the curves of the once-punctured triangle.
A map onto it would be this map composed with that relabelling.  Mapping onto
`A1DoddConeKAlg(0)` directly keeps one dictionary: its labels are cone-monomial
words shaped like the zoo's, and its ρ is a letter permutation as the zoo's is.

The generator map, built at runtime (nothing is stored).  Neither class carries
charges for its generators, so the map is found by structure: both have six
generators in two ρ-orbits of length 3 (the zoo's `A1D3_RHO_PERM`; `i ↦ i+1` on
`(1, p, i)`, p = 0, 1).  Of the 2·3·3 = 18 ρ-equivariant bijections (which orbit
goes where, and one cyclic shift per orbit), exactly three reproduce all 36
generator products, with the zoo's SU(2) character coefficients `χ_κ` moved into
the label's κ slot (`to_a1dodd`); they differ by a simultaneous shift of both
orbits, i.e. by the algebra's own automorphism ρ.  The served map is the first
of them in a fixed enumeration order; the seed traces do not depend on the
choice, since a seed trace is constant on its ρ-orbit here (ρ² generates the
same cyclic group as ρ on orbits of length 3, and the su2 entries fold along
ρ²-orbits, `elem_traces.fold_policy`).  If the passing set is anything else —
empty, or containing two maps not related by ρ — the construction raises.
The orbit-swapped pairing reproduces at most 18 of the 36 products, and a shift
of one orbit alone 6 (the negative controls in the suite in the source repository).

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from itertools import permutations, product

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent


__all__ = ["cone_algebra", "generator_map", "product_agreement",
           "to_a1dodd_label", "to_a1dodd", "vacuum_trace", "seed_trace"]

_STATE: dict = {}


def cone_algebra():
    """The (shared, per-process) `A1DoddConeKAlg(0)`."""
    if "D" not in _STATE:
        from a1dodd_kalg import A1DoddConeKAlg
        _STATE["D"] = A1DoddConeKAlg(0)
    return _STATE["D"]


def _zoo():
    if "Z" not in _STATE:
        from finite_a1d3_kalg import FiniteA1D3KAlgebra
        _STATE["Z"] = FiniteA1D3KAlgebra()
    return _STATE["Z"]


def _orbits(gens, rho):
    seen, out = set(), []
    for g in gens:
        if g in seen:
            continue
        orbit = [g]
        seen.add(g)
        x = rho(g)
        while x != g:
            orbit.append(x)
            seen.add(x)
            x = rho(x)
        out.append(orbit)
    return out


def _as_dict(x: Element) -> dict:
    return {lab: {q: v for q, v in c._coeffs.items() if v}
            for lab, c in x.terms.items() if not c.is_zero()}


def _to_a1dodd_with(gm: dict, x: Element) -> dict:
    out: dict = {}
    for word, c in x.terms.items():
        w = tuple(sorted((gm[mg], p) for mg, p in word))
        if isinstance(c, RLaurent):
            items = [(q, n, v) for q, r in c.coeffs.items()
                     for n, v in r.terms.items() if v]
        else:
            items = [(q, 0, v) for q, v in c._coeffs.items() if v]
        for q, n, v in items:
            d = out.setdefault((w, n), {})
            d[q] = d.get(q, 0) + v
    return {lab: {q: v for q, v in d.items() if v}
            for lab, d in out.items() if any(d.values())}


def product_agreement(gm: dict) -> tuple:
    """`(agreeing, total)` over the ordered pairs of zoo generators: the zoo
    product, carried over by `gm`, against `A1DoddConeKAlg(0)`'s product of the
    images."""
    Z, D = _zoo(), cone_algebra()
    gens = sorted(gm)
    ok = 0
    for a in gens:
        for b in gens:
            lhs = _to_a1dodd_with(gm, Z.multiply(((a, 1),), ((b, 1),)))
            rhs = _as_dict(D.multiply((((gm[a], 1),), 0), (((gm[b], 1),), 0)))
            ok += lhs == rhs
    return ok, len(gens) ** 2


def generator_map() -> dict:
    """`{mg: (a, p, i)}`: the zoo generator `mg` is `A1DoddConeKAlg(0)`'s
    generator `(a, p, i)` (module docstring).  Found at first use and cached;
    raises `AssertionError` unless the passing maps are exactly one class
    under ρ."""
    if "gm" in _STATE:
        return _STATE["gm"]
    Z, D = _zoo(), cone_algebra()
    zperm = Z._rho_perm
    zgens = sorted(range(len(Z.cone_data().mult_gens())))
    cd = D.cone_data()
    zo = _orbits(zgens, lambda g: zperm.get(g, g))
    do = _orbits(sorted(cd.mult_gens()), cd.rho_label)
    if sorted(map(len, zo)) != sorted(map(len, do)):
        raise AssertionError(
            f"a1d3_seeds: ρ-orbit profiles differ ({[len(o) for o in zo]} vs "
            f"{[len(o) for o in do]})")
    passing = []
    for pairing in permutations(do):
        if [len(o) for o in pairing] != [len(o) for o in zo]:
            continue
        for shifts in product(*[range(len(o)) for o in zo]):
            gm = {}
            for zorb, dorb, s in zip(zo, pairing, shifts):
                for j, g in enumerate(zorb):
                    gm[g] = dorb[(j + s) % len(dorb)]
            ok, tot = product_agreement(gm)
            if ok == tot:
                passing.append(gm)
    if not passing:
        raise AssertionError("a1d3_seeds: no ρ-equivariant generator map "
                             "reproduces the generator products")
    first = passing[0]
    rho_class = []
    for n in range(len(zgens)):
        shifted = {}
        for g, h in first.items():
            for _ in range(n):
                h = cd.rho_label(h)
            shifted[g] = h
        rho_class.append(shifted)
    if any(gm not in rho_class for gm in passing):
        raise AssertionError(
            "a1d3_seeds: two generator maps not related by ρ reproduce the "
            "products; the seed traces would be ambiguous")
    _STATE["gm"] = first
    return first


def to_a1dodd_label(word, kappa: int = 0) -> tuple:
    """`A1DoddConeKAlg(0)`'s label of the zoo label `word = ((mg, p), …)`
    dressed by `χ_κ`."""
    gm = generator_map()
    return (tuple(sorted((gm[mg], p) for mg, p in word)), kappa)


def to_a1dodd(x: Element) -> Element:
    """A zoo `Element` (coefficients `LaurentPoly`, or `RLaurent` over
    `SU2ZPlusRing`) as an `A1DoddConeKAlg(0)` `Element`: each coefficient
    `𝖖^j·χ_κ` moved into the label's κ slot."""
    d = _to_a1dodd_with(generator_map(), x)
    return Element({lab: LaurentPoly(c) for lab, c in d.items()})


def _series(label, K: int) -> dict:
    cache = _STATE.setdefault("series", {})
    hit = cache.get(label)
    if hit is None or hit[0] < K:
        tr = cone_algebra().trace(label, K)
        data = {}
        for q, r in tr.coeffs.items():
            row = {int(n): int(v) for n, v in r.terms.items() if v}
            if row and q <= K:
                data[q] = row
        hit = (K, data)
        cache[label] = hit
    return {q: dict(row) for q, row in hit[1].items() if q <= K}


def vacuum_trace(K: int) -> dict:
    """`Tr(1)` through `𝖖^K` as `{𝖖-power: {SU(2) highest weight: int}}`."""
    return _series(cone_algebra().identity(), K)


def seed_trace(i: int, K: int) -> dict:
    """`Tr(L_{((i, 1),)})` of the zoo generator `i` through `𝖖^K`, in the same
    format."""
    if i not in generator_map():
        raise KeyError(f"a1d3: no multiplicative generator {i!r}")
    return _series(to_a1dodd_label(((i, 1),)), K)
