"""[A₁,D₄]: the zoo entry `a1d4` through `SU3ADKAlg`, restricted to SU(2)×U(1).

The ADE finite K-algebras are served here by
self-contained realizations, with no frozen trace data and no runtime oracle on
the serving path.  `su3_ad_kalg.SU3ADKAlg` is `[A₁,D₄]` with its SU(3)
flavour: eight generators `T_i`, `D_i` (`i ∈ Z/4`), closed-form relations, and a
trace without BPS (`sl3_su3_traces`: `Tr(1)` and the `T` / `D` seeds since
2026-09-24 from the closed forms of the even-D family at k = 1 through
`SU3ADKAlg`'s curve map onto `A1DevenKAlg(1)`, `Tr(1)` from Creutzig's gauge
tower — before, `Tr(1)` the Kac–Wakimoto vacuum character of `sl(3)_{−3/2}`
and the seeds by a forward orthonormality pass, both now witnesses).  So zoo
a1d4's `Tr(1)` below is that gauge tower too, summed over the gauge charge
and branched.  Its restriction `SU3ADKAlg().base_change(su3_to_su2u1_hom())`
has the zoo's flavour group.  Earlier releases served a1d4 through an su2u1
route that had the flavour slots the wrong way round (no SU(2) triplet at 𝖖²).

The U(1) normalisation (the factor 3).  `zplus_ring.su3_to_su2u1_hom` branches
along `SU(2)×U(1) ⊂ SU(3)`, `U(1) = diag(e^{iθ}, e^{iθ}, e^{−2iθ})`, so
`3 ↦ 2_{+1} ⊕ 1_{−2}`: the U(1) charge `Y` of a branched SU(3) character takes
every integer value.  The zoo's `SU2xU1ZPlusRing` charge `m` is a third of it:
the zoo's U(1) fugacity is `μ = e^{3iθ}`.  A zoo generator is an `SU3ADKAlg`
generator carrying the extra branching charge `c_g`: `c_g = 0` on the T-orbit
and `c_g = −2, +2, −2, +2` along the D-orbit.  The sign alternates along each
ρ-orbit because ρ acts by `⋆` (`Y ↦ −Y`) and the zoo's ρ carries no μ-shift.
A zoo label `w = ((g, p), …)` is therefore the `SU3ADKAlg` monomial `L_sec` of
the same letters with branching charge `c(w) = Σ p·c_g` added; a term of
branching charge `Y` on `L_sec` in an `SU3ADKAlg` product is the zoo term
`μ^{(Y − c(w))/3}·L_w`, and the zoo trace of `L_w` is the branched
`SU3ADKAlg` trace with every charge `Y` shifted by `c(w)` and divided by 3.
Every such charge is an integer (asserted): the triality of an SU(3)
character is tied to the gauge charge it sits at.  The BPS chart of the zoo's
own quiver agrees independently: its Cartan fugacities are
`(m, SU(2) weight)`, and the SU(3) adjoint at `𝖖²` of its `Tr(1)` has its
doublets at `m = ±1`, i.e. `Y = ±3` (the suite in the source repository).

The generator map, built at runtime (nothing is stored).  Both classes have
eight generators in two ρ-orbits of length 4.  Of the 2·4·4 ρ-equivariant
bijections, each combined with an offset pair `(c_A, c_B) ∈ [−6, 6]²` for the
two zoo orbits (alternating along each orbit), the ones that reproduce all 64
generator products are found at first use.  Exactly two are found: the zoo
orbit `(0, 2, 6, 4)` goes to the T-orbit with `c = 0`, `(1, 5, 3, 7)` to the
D-orbit with `c = −2, +2, −2, +2`, and the two differ by ρ², which fixes every
trace here (`δ = 0` and `⋆² = 1`).  The served map is the first of them; any
other outcome raises.

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from itertools import permutations, product

from kalgebra import Element
from laurent_poly import LaurentPoly


__all__ = ["cone_algebra", "generator_map", "product_agreement", "to_su3ad",
           "vacuum_trace", "seed_trace"]

_STATE: dict = {}
_OFFSETS = range(-6, 7)


def cone_algebra():
    """The (shared, per-process) `SU3ADKAlg`."""
    if "A" not in _STATE:
        from su3_ad_kalg import SU3ADKAlg
        _STATE["A"] = SU3ADKAlg()
    return _STATE["A"]


def _branch():
    if "br" not in _STATE:
        from zplus_ring import su3_to_su2u1_hom
        _STATE["br"] = su3_to_su2u1_hom()
    return _STATE["br"]


def _zoo():
    if "Z" not in _STATE:
        from finite_a1d4_kalg import FiniteA1D4KAlgebra
        _STATE["Z"] = FiniteA1D4KAlgebra()
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


def _letter(label):
    """The letter `('T', i)` / `('D', i)` of an `SU3ADKAlg` generator label."""
    from su3_ad_kalg import _TILE_LETTERS
    tile, a, b, _p, _q = label
    l_t, l_d = _TILE_LETTERS[tile]
    return l_t if a else l_d


def _section_letters(tile, a, b):
    from su3_ad_kalg import _TILE_LETTERS
    l_t, l_d = _TILE_LETTERS[tile]
    return [(L, n) for L, n in ((l_t, a), (l_d, b)) if n]


def _zoo_terms(x: Element) -> dict:
    """`{(word, 𝖖-power, k, m): c}` of a zoo `Element`."""
    out: dict = {}
    for w, c in x.terms.items():
        for q, r in c.coeffs.items():
            for (k, m), v in r.terms.items():
                if v:
                    out[(w, q, k, m)] = out.get((w, q, k, m), 0) + v
    return {key: v for key, v in out.items() if v}


def _su3_terms(x: Element, letter_to_zoo: dict, offsets: dict, c_in: int):
    """An `SU3ADKAlg` `Element`, its branching charges shifted by `c_in`, in
    the zoo's terms `{(word, 𝖖-power, k, m): c}`, or `None` if a zoo charge is
    not integral."""
    br = _branch()
    out: dict = {}
    for (tile, a, b, p, q), lp in x.terms.items():
        letters = _section_letters(tile, a, b)
        word = tuple(sorted((letter_to_zoo[L], n) for L, n in letters))
        c_out = sum(offsets[letter_to_zoo[L]] * n for L, n in letters)
        branched = br.apply_basis((p, q))
        for e, v in lp._coeffs.items():
            if not v:
                continue
            for (k, Y), c in branched.terms.items():
                num = c_in + Y - c_out
                if num % 3:
                    return None
                key = (word, e, k, num // 3)
                out[key] = out.get(key, 0) + v * c
    return {key: v for key, v in out.items() if v}


def _products():
    if "products" not in _STATE:
        Z, A = _zoo(), cone_algebra()
        sgen = [A.T(i) for i in range(4)] + [A.D(i) for i in range(4)]
        _STATE["products"] = (
            {(a, b): _zoo_terms(Z.multiply(((a, 1),), ((b, 1),)))
             for a in range(8) for b in range(8)},
            {(a, b): A.multiply(a, b) for a in sgen for b in sgen})
    return _STATE["products"]


def product_agreement(z2s: dict, offsets: dict, *, stop_early=False) -> tuple:
    """`(agreeing, total)` over the ordered pairs of zoo generators, for the
    generator map `z2s` (zoo index ↦ `SU3ADKAlg` generator label) and the
    offsets `c_g`."""
    zp, sp = _products()
    letter_to_zoo = {_letter(s): g for g, s in z2s.items()}
    ok = tot = 0
    for a in range(8):
        for b in range(8):
            tot += 1
            got = _su3_terms(sp[(z2s[a], z2s[b])], letter_to_zoo, offsets,
                             offsets[a] + offsets[b])
            if got is not None and got == zp[(a, b)]:
                ok += 1
            elif stop_early:
                return ok, 64
    return ok, tot


def generator_map() -> tuple:
    """`(z2s, offsets)`: zoo generator `g` is the `SU3ADKAlg` generator
    `z2s[g]` carrying the extra branching charge `offsets[g]` (module
    docstring).  Found at first use and
    cached; raises `AssertionError` unless the certified maps are exactly one
    class under ρ²."""
    if "gm" in _STATE:
        return _STATE["gm"]
    Z, A = _zoo(), cone_algebra()
    zo = _orbits(range(8), lambda g: Z._rho_perm.get(g, g))
    so = _orbits([A.T(i) for i in range(4)] + [A.D(i) for i in range(4)],
                 A.rho)
    if sorted(map(len, zo)) != sorted(map(len, so)):
        raise AssertionError("a1d4_seeds: ρ-orbit profiles differ")
    passing = []
    for pairing in permutations(so):
        if [len(o) for o in pairing] != [len(o) for o in zo]:
            continue
        for shifts in product(*[range(len(o)) for o in zo]):
            z2s = {g: sorb[(j + s) % len(sorb)]
                   for zorb, sorb, s in zip(zo, pairing, shifts)
                   for j, g in enumerate(zorb)}
            for cs in product(_OFFSETS, repeat=len(zo)):
                offsets = {g: (c0 if j % 2 == 0 else -c0)
                           for zorb, c0 in zip(zo, cs)
                           for j, g in enumerate(zorb)}
                if product_agreement(z2s, offsets, stop_early=True)[0] == 64:
                    passing.append((z2s, offsets))
    if not passing:
        raise AssertionError("a1d4_seeds: no ρ-equivariant generator map with "
                             "U(1) offsets reproduces the generator products")
    z2s0, off0 = passing[0]
    rho2 = {g: A.rho(A.rho(s)) for g, s in z2s0.items()}
    for z2s, offsets in passing:
        if offsets != off0 or z2s not in (z2s0, rho2):
            raise AssertionError(
                "a1d4_seeds: two certified generator maps not related by ρ²; "
                "the seed traces would be ambiguous")
    _STATE["gm"] = (z2s0, off0)
    return _STATE["gm"]


def to_su3ad(word) -> tuple:
    """`(c, label)`: the zoo label `word = ((g, p), …)` is the `SU3ADKAlg`
    label `label` (flavour-trivial) carrying the extra branching charge `c`."""
    z2s, offsets = generator_map()
    A = cone_algebra()
    lab = A.identity()
    for (g, p) in word:
        for _ in range(p):
            prod = A.multiply(lab, z2s[g])
            if len(prod.terms) != 1:
                raise ValueError(f"a1d4: {word!r} is not a cone monomial")
            (lab, coef), = prod.terms.items()
    c = sum(offsets[g] * p for g, p in word)
    return c, lab


def _series(key, label, c: int, K: int) -> dict:
    cache = _STATE.setdefault("series", {})
    hit = cache.get(key)
    if hit is None or hit[0] < K:
        br = _branch()
        tr = cone_algebra().trace(label, K)
        data: dict = {}
        for q, r in tr.coeffs.items():
            if q > K:
                continue
            row: dict = {}
            for (k, Y), v in br.apply_RElement(r).terms.items():
                if not v:
                    continue
                if (Y + c) % 3:
                    raise ArithmeticError(
                        f"a1d4: branching charge {Y + c} at 𝖖^{q} is not a "
                        f"multiple of 3, the zoo's U(1) unit")
                row[(k, (Y + c) // 3)] = row.get((k, (Y + c) // 3), 0) + int(v)
            row = {kk: v for kk, v in row.items() if v}
            if row:
                data[q] = row
        hit = (K, data)
        cache[key] = hit
    return {q: dict(row) for q, row in hit[1].items() if q <= K}


def vacuum_trace(K: int) -> dict:
    """`Tr(1)` through `𝖖^K` as `{𝖖-power: {(k, m): int}}` over the zoo's
    `SU2xU1ZPlusRing`."""
    return _series("identity", cone_algebra().identity(), 0, K)


def seed_trace(i: int, K: int) -> dict:
    """`Tr(L_{((i, 1),)})` of the zoo generator `i` through `𝖖^K`, in the same
    format."""
    z2s, offsets = generator_map()
    if i not in z2s:
        raise KeyError(f"a1d4: no multiplicative generator {i!r}")
    return _series(i, z2s[i], offsets[i], K)
