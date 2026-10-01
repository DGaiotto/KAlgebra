"""[A₁,D₂ₖ₊₃]: the zoo entries `a1d3` / `a1d5` / `a1d7` identified with the
closed-form cone class `a1dodd_kalg.A1DoddConeKAlg(k)`, k = 0, 1, 2.

The ADE finite K-algebras are served by self-contained realizations and, for
the `A` and `D` families, carry geometric labels of the canonical basis.
`A1DoddConeKAlg(k)` carries them: its `geometric_label` reads an `(a, p, i)`
word as curves of the once-punctured `(2k+3)`-gon, `A1DnKAlg(2k+3)`'s labels.
This module identifies the zoo standalone's canonical basis with
`A1DoddConeKAlg(k)`'s, so that `zoo_geometry` can read the zoo's labels as
curves.

What it serves.  For `a1d5` / `a1d7`, the map behind their geometric labels.
Their seed traces stay on the `a1d5_layer2` / `a1d7_layer2` closed forms
(`elem_traces`), and `A1DoddConeKAlg(k)`'s traces of the images, the
`a1dodd_layer2` closed forms, are their witness (the suite in the source
repository).
For `a1d3` the map is a positive control of the method.  `a1d3_seeds`
serves that entry through a map found by an exhaustive search, and the two
maps are the same map (measured 2026-09-24).  The same pattern appears in `a1deven_seeds`, which carries `a1d4`
as a witness while `a1d4_seeds` serves it.

The map, found at runtime (nothing is stored)
---------------------------------------------
Zoo generator `g` goes to `A1DoddConeKAlg(k)`'s generator `σ(g) = (a, p, i)`,
and a zoo coefficient `𝖖^j·χ_κ` goes to `𝖖^j` on the κ slot of the label, since
`L_{(w, κ)} = χ_κ·L_{(w, 0)}` there.  A zoo cone monomial `((g, n), …)` goes to
the word `((σ(g), n), …)`.  The flavour is SU(2), whose only one-dimensional
representation is the trivial one, so there are no offsets to solve for; the
U(1) of `a1deven_seeds` needs them.  `σ` is found by a backtracking search
over the ρ-orbits (which orbit goes where, and one cyclic shift per orbit).
The search is pruned by a signature of each generator product: for each term
label, the multiset of (𝖖-power, SU(2) weight, coefficient).  Every candidate
is then certified term for term on every generator product and on ρ.

Measured (2026-09-24):

* the ρ-orbits are `[3, 3]` / `[5⁴]` / `[7⁶]` on both sides at k = 0 / 1 / 2;
* the passing maps are exactly one map composed with the powers of ρ, 3 / 5 /
  7 of them, each reproducing all 36 / 400 / 1,764 generator products and ρ;
* they are found in well under a second.

The first certified map in the search order is served, and the others give
the same labels up to the rotation.  Any other outcome raises.

The KAlgebraIso
---------------
Whether the seeds maps are `KAlgebraIso`s was an open question; they were
not, and `kalgebra_iso(short_id, native=None)` makes
this one a `KAlgebraIso`.  The zoo keeps its SU(2) flavour in its coefficients
and `A1DoddConeKAlg(k)` in its labels, and a `KAlgebraIso` multiplies
coefficients through.  So the source is the zoo's Z-form wrapper
`finite_su2_zform.FiniteSU2ZKAlgebra(native)`, the generalisation of
`FiniteA1D3ZKAlgebra` to every SU(2)-flavoured standalone, written for this;
its label `(κ, word)` is `χ_κ·L_word`.  Forward is `(κ, word) ↦ (σ(word), κ)`,
the coefficient move of `to_a1dodd` made on labels.  The inverse goes back
letter by letter through `σ⁻¹`, and the word is made canonical in the zoo's
cone (the identity here, where every cone is simplicial).  The two rings are
the same `R(SU(2))`, which has no nontrivial one-dimensional representation,
so no normalisation enters.

Certified by the suite in the source repository, measured 2026-09-26 (shared machine).
`verify_all` passes all five checks — unit, round trip, multiplicative,
ρ-equivariant, trace-equivariant through `𝖖¹²` — on the identity, every
generator and `χ₁`, and on every ordered pair of them each way:

| entry | pairs each way | time | round trip on product labels |
|---|---|---|---|
| a1d3 | 7² = 49 | 0.1 s | 27 + 27 |
| a1d5 | 21² = 441 | 0.2 s | 163 + 163 |
| a1d7 | 43² = 1,849 | 2.9 s | 619 + 619 |

The trace check at a1d5 / a1d7 compares two independent routes: the zoo's own
`a1d5_layer2` / `a1d7_layer2` closed forms against `A1DoddConeKAlg(k)`'s
`a1dodd_layer2`.  `verify_maps_section_to_section_1drep` holds both ways.  At
a1d3 the same maps pass from the existing wrapper `FiniteA1D3ZKAlgebra()`.
Negative controls at a1d5 and a1d7: with two ρ-orbits swapped the battery
fails the multiplicative and trace checks, and with one orbit shifted alone it
fails the multiplicative check.  `finite_kalgebra_objects.kalgebra_object`
registers the pair as 'z-form' → 'a1dodd'.

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent

from regen import _canonical_zoo_word


__all__ = ["K_OF", "cone_algebra", "generator_map", "product_agreement",
           "rho_agreement", "to_a1dodd_label", "to_a1dodd", "kalgebra_iso"]

K_OF = {"a1d3": 0, "a1d5": 1, "a1d7": 2}

_STATE: dict = {}


def _st(short_id: str) -> dict:
    if short_id not in K_OF:
        raise KeyError(f"a1dodd_seeds: {short_id!r} is not one of "
                       f"{sorted(K_OF)}")
    return _STATE.setdefault(short_id, {"k": K_OF[short_id]})


def cone_algebra(short_id: str):
    """The (shared, per-process) `A1DoddConeKAlg(k)` for this entry."""
    st = _st(short_id)
    if "D" not in st:
        from a1dodd_kalg import A1DoddConeKAlg
        st["D"] = A1DoddConeKAlg(st["k"])
    return st["D"]


def _zoo(short_id: str):
    st = _st(short_id)
    if "Z" not in st:
        from elem_traces import _standalone_algebra
        st["Z"] = _standalone_algebra(short_id)
    return st["Z"]


# ---------------------------------------------------------------------------
# term dictionaries
# ---------------------------------------------------------------------------

def _zoo_terms(x: Element) -> dict:
    """`{(word, 𝖖-power, κ): c}` of a zoo `Element` (coefficients `RLaurent`
    over `SU2ZPlusRing`, keyed by the SU(2) weight κ, or `LaurentPoly`)."""
    out: dict = {}
    for w, c in x.terms.items():
        if isinstance(c, RLaurent):
            items = [(q, int(kap), v) for q, r in c.coeffs.items()
                     for kap, v in r.terms.items() if v]
        else:
            items = [(q, 0, v) for q, v in c._coeffs.items() if v]
        for q, kap, v in items:
            out[(w, q, kap)] = out.get((w, q, kap), 0) + v
    return {t: v for t, v in out.items() if v}


def _dodd_terms(x: Element) -> dict:
    """`{(word, 𝖖-power, κ): c}` of an `A1DoddConeKAlg` `Element` (labels
    `(word, κ)`, coefficients `LaurentPoly`)."""
    out: dict = {}
    for (w, kap), c in x.terms.items():
        for q, v in c._coeffs.items():
            if v:
                out[(w, q, kap)] = out.get((w, q, kap), 0) + v
    return {t: v for t, v in out.items() if v}


def _image(sigma: dict, terms: dict) -> dict:
    """Zoo terms carried to `A1DoddConeKAlg` terms by the generator map."""
    out: dict = {}
    for (w, q, kap), v in terms.items():
        t = (tuple(sorted((sigma[g], p) for g, p in w)), q, kap)
        out[t] = out.get(t, 0) + v
    return {t: v for t, v in out.items() if v}


def _signature(terms: dict) -> tuple:
    """A relabelling-blind invariant of a product: the multiset over its term
    labels of the multisets of (𝖖-power, SU(2) weight, coefficient)."""
    per: dict = {}
    for (w, q, kap), v in terms.items():
        per.setdefault(w, []).append((q, kap, v))
    return tuple(sorted(tuple(sorted(x)) for x in per.values()))


def _orbits(gens, rho) -> list:
    seen, out = set(), []
    for g in gens:
        if g in seen:
            continue
        orbit = [g]
        seen.add(g)
        x = rho(g)
        while x != g:
            if x in seen or len(orbit) > len(gens):
                raise AssertionError(f"a1dodd_seeds: ρ is not a permutation "
                                     f"of the generators at {g!r}")
            orbit.append(x)
            seen.add(x)
            x = rho(x)
        out.append(orbit)
    return out


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _zoo_rho(short_id: str, g: int) -> int:
    (h, _p), = _zoo(short_id).rho(((g, 1),))
    return h


def _dodd_rho(short_id: str, g: tuple) -> tuple:
    D = cone_algebra(short_id)
    ((h, _p),), _kappa = D.rho(D.gen(*g))
    return h


def _products(short_id: str):
    st = _st(short_id)
    if "products" not in st:
        Z, D = _zoo(short_id), cone_algebra(short_id)
        n = len(Z.cone_data().mult_gens())
        dg = sorted(D.cone_data().mult_gens())
        zp = {(a, b): _zoo_terms(Z.multiply(((a, 1),), ((b, 1),)))
              for a in range(n) for b in range(n)}
        dp = {(x, y): _dodd_terms(D.multiply(D.gen(*x), D.gen(*y)))
              for x in dg for y in dg}
        st["products"] = (n, dg, zp, dp)
    return st["products"]


def _candidates(short_id: str) -> list:
    """The ρ-equivariant bijections `σ` whose generator products agree with
    the zoo's in `_signature`, in the search order."""
    n, dg, zp, dp = _products(short_id)
    zo = _orbits(range(n), lambda g: _zoo_rho(short_id, g))
    do = _orbits(dg, lambda g: _dodd_rho(short_id, g))
    if sorted(map(len, zo)) != sorted(map(len, do)):
        raise AssertionError(
            f"a1dodd_seeds({short_id}): ρ-orbit profiles differ "
            f"({sorted(map(len, zo))} vs {sorted(map(len, do))})")
    zs = {key: _signature(t) for key, t in zp.items()}
    ds = {key: _signature(t) for key, t in dp.items()}
    order = sorted(range(len(zo)), key=lambda i: (-len(zo[i]), i))
    found: list = []

    def placed(i, j, t):
        return {g: do[j][(p + t) % len(do[j])] for p, g in enumerate(zo[i])}

    def fits(assign, m):
        for a in m:
            for b in m:
                if zs[(a, b)] != ds[(m[a], m[b])]:
                    return False
        for m2 in assign.values():
            for a in m:
                for b in m2:
                    if (zs[(a, b)] != ds[(m[a], m2[b])]
                            or zs[(b, a)] != ds[(m2[b], m[a])]):
                        return False
        return True

    def walk(idx, assign, used):
        if idx == len(order):
            sigma: dict = {}
            for m in assign.values():
                sigma.update(m)
            found.append(sigma)
            return
        i = order[idx]
        for j in range(len(do)):
            if j in used or len(do[j]) != len(zo[i]):
                continue
            for t in range(len(do[j])):
                m = placed(i, j, t)
                if fits(assign, m):
                    assign[i] = m
                    used.add(j)
                    walk(idx + 1, assign, used)
                    del assign[i]
                    used.discard(j)

    walk(0, {}, set())
    return found


def product_agreement(short_id: str, sigma: dict) -> tuple:
    """`(agreeing, total)` over the ordered pairs of zoo generators: the zoo
    product carried over by `sigma`, term for term (𝖖-power and SU(2) weight
    included), against `A1DoddConeKAlg(k)`'s product of the images."""
    n, _dg, zp, dp = _products(short_id)
    ok = sum(_image(sigma, zp[(a, b)]) == dp[(sigma[a], sigma[b])]
             for a in range(n) for b in range(n))
    return ok, n * n


def rho_agreement(short_id: str, sigma: dict) -> tuple:
    """`(agreeing, total)` over the zoo generators: `σ(ρ(g)) == ρ(σ(g))`."""
    n = len(sigma)
    ok = sum(sigma[_zoo_rho(short_id, g)] == _dodd_rho(short_id, sigma[g])
             for g in range(n))
    return ok, n


def generator_map(short_id: str) -> dict:
    """`{g: (a, p, i)}`: the zoo generator `g` is `A1DoddConeKAlg(k)`'s
    generator `(a, p, i)` (module docstring).  Found at first use and cached;
    raises `AssertionError` unless the certified maps are exactly one map
    composed with the powers of ρ."""
    st = _st(short_id)
    if "gm" in st:
        return st["gm"]
    n = len(_products(short_id)[1])
    certified = [s for s in _candidates(short_id)
                 if product_agreement(short_id, s) == (n * n, n * n)
                 and rho_agreement(short_id, s) == (n, n)]
    if not certified:
        raise AssertionError(f"a1dodd_seeds({short_id}): no ρ-equivariant "
                             f"generator map reproduces the generator products")
    first = certified[0]
    orbit, cur = [], dict(first)
    for _ in range(2 * st["k"] + 3):
        orbit.append(frozenset(cur.items()))
        cur = {g: _dodd_rho(short_id, h) for g, h in cur.items()}
    if sorted(map(sorted, map(frozenset, (s.items() for s in certified)))) \
            != sorted(map(sorted, orbit)):
        raise AssertionError(
            f"a1dodd_seeds({short_id}): the certified maps are not one map "
            f"composed with the powers of ρ ({len(certified)} of them)")
    st["gm"] = first
    return first


def to_a1dodd_label(short_id: str, word, kappa: int = 0) -> tuple:
    """`A1DoddConeKAlg(k)`'s label of the zoo label `word = ((g, n), …)`
    dressed by `χ_κ`."""
    gm = generator_map(short_id)
    return (tuple(sorted((gm[g], n) for g, n in word)), kappa)


def to_a1dodd(short_id: str, x: Element) -> Element:
    """A zoo `Element` as an `A1DoddConeKAlg(k)` `Element`: each coefficient
    `𝖖^j·χ_κ` moved into the label's κ slot."""
    out: dict = {}
    for (w, q, kap), v in _image(generator_map(short_id), _zoo_terms(x)).items():
        c = out.setdefault((w, kap), {})
        c[q] = c.get(q, 0) + v
    return Element({lab: LaurentPoly(c) for lab, c in out.items()})


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

_ONE = LaurentPoly.one()


def kalgebra_iso(short_id: str, native=None):
    """The generator map as a `KAlgebraIso` from the zoo's Z-form wrapper
    `FiniteSU2ZKAlgebra(native)` onto `A1DoddConeKAlg(k)` (module docstring,
    "The KAlgebraIso"): `(κ, word) ↦ (σ(word), κ)`, and back letter by letter
    through `σ⁻¹`, the word made canonical in the zoo's cone.

    `native` is the zoo standalone to wrap, an instance of
    `FINITE_KALGEBRAS[short_id]` (default: a fresh one;
    `finite_kalgebra_objects.kalgebra_object` passes its `'cone-frozen'`
    realization); the target is the shared `cone_algebra(short_id)`.  The
    label maps build the generator map at first use."""
    import finite_kalgebras as fk
    from finite_su2_zform import FiniteSU2ZKAlgebra
    from kalgebra_iso import KAlgebraIso
    st = _st(short_id)
    cls = fk.FINITE_KALGEBRAS[short_id]
    if native is None:
        native = cls()
    elif not isinstance(native, cls):
        raise TypeError(f"a1dodd_seeds.kalgebra_iso({short_id!r}): native is "
                        f"a {type(native).__name__}, not a {cls.__name__}")
    Z = FiniteSU2ZKAlgebra(native)
    D = cone_algebra(short_id)

    def forward(label) -> Element:
        kappa, word = label
        return Element({to_a1dodd_label(short_id, word, kappa): _ONE})

    def inverse(label) -> Element:
        dword, kappa = D.canonicalise(label)
        if "gm_inverse" not in st:
            st["gm_inverse"] = {h: g for g, h in generator_map(short_id).items()}
        inv = st["gm_inverse"]
        word = _canonical_zoo_word(native, ((inv[h], n) for h, n in dword))
        return Element({(kappa, word): _ONE})

    return KAlgebraIso(Z, D, forward, inverse,
                       name=f"{short_id}[z-form→a1dodd]")
