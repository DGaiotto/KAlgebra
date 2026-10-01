"""[A₁,D₂ₖ₊₂]: the zoo entries `a1d6` / `a1d8` through the ungauged D-even
algebra `A1DevenKAlg(k)`, k = 2, 3 (and `a1d4` at k = 1, as a witness only).

The ADE finite K-algebras are served here by
self-contained realizations, with no frozen trace data and no runtime oracle on
the serving path.  `a1deven_kalg.A1DevenKAlg(k)` is the ungauged `[A₁,D₂ₖ₊₂]`:
the centralizer of the gauge letter `E = X₀₁` in `U1A1DevenConeKAlgebra(k)`,
with `E` promoted to a U(1) flavour fugacity — `E = z⁻¹`, so a label
`(F, e, κ)` (`F` a balanced multiset of curves of the once-punctured
`(2k+2)`-gon) is `z^{−e}·χ_κ·(F, 0, 0)` — over
`R(SU(2)) ⊗ R(U(1))` (keys `(b, (f,))`).  Since 2026-09-24 it is Z-form: its
`multiply` keeps both weights in the label with integral coefficients, and its
trace and flavour lift `(F, e, κ) ↦ ((F, 0, 0), (κ, (−e,)))` read the label
so; this module reads its elements through that lift (`_deven_terms`), which
gives the R-form view it was written for (`E`-free labels, `z` and `χ` in the
coefficient key).  Its products are the closed forms of the gauged curve
frame (`U1A1DevenConeKAlgebra`); its trace is the gauged trace summed over
the gauge charge with the restored `(𝖖²;𝖖²)²_∞` measure, the gauged matter
traces being the exact transport of the gauged class's closed-form RG image
into the A1Dodd closed forms (`u1a1deven_trace_transport`).  This module identifies the zoo standalone's
canonical basis with it and serves the zoo's seed traces from it
(`elem_traces._seed_series`).  Earlier releases served `a1d6` / `a1d8`
through an su2u1 route that peeled the flavour slots the wrong way round.
`a1d4` stays on `SU3ADKAlg` (`a1d4_seeds`); its map here is the positive
control of the method.

The map, found at runtime (nothing is stored)
---------------------------------------------
Zoo generator `g` goes to `z^{c_g}·L_{σ(g)}`, `L_{σ(g)}` an E-free generator
`(F, 0, 0)` of `A1DevenKAlg(k)` (`mult_generators()`), and a zoo coefficient
`χ_b·μ^m` to
`χ_b·z^{s·m}`.  A zoo cone monomial `w` goes to `z^{c(w)}·L_{ℓ(w)}`, with
`ℓ(w)` the single label of the product of the letters' images `L_{σ(g)}` and
`c(w) = Σ p·c_g + f(w)`, `f(w)` that product's own `z`-power (a 𝖖-commuting
product of E-free generators can carry a `z`).  Found at first use, per `k`:

  1. `σ`: the ρ-equivariant bijections (which ρ-orbit goes where, one cyclic
     shift per orbit; `A1DevenKAlg`'s ρ is the gauged ρ with the E-power
     dropped) whose generator products agree with the zoo's in a
     flavour-blind invariant: per product, the multiset over its terms of the
     multisets of (𝖖-power, integer coefficient).  A backtracking search over
     the orbits (ρ-orbit lengths [4, 4] / [3, 6⁶] / [8¹⁵] at k = 1 / 2 / 3);
  2. for each such `σ` and each slot order of the zoo's coefficient keys, the
     U(1) normalisation `s` and the offsets `c_g`, as the unique solution of the
     linear equations that the generator products (every term whose label,
     𝖖-power and SU(2) weight carry one U(1) charge on each side) and ρ
     (`s·δ_g + c_{ρ(g)} + c_g = −e_g`, `δ` the zoo's `_rho_delta`, `e_g` the
     E-power of the gauged ρ-image) impose, by exact elimination over `Q`.
     No window, no candidate list for `s`;
  3. the certificate: every generator product agrees term for term, and ρ on
     every generator (`rho_element` of the zoo against the gauged ρ read in
     the ungauged frame, `ρ(z) = z⁻¹`).

Measured (2026-09-24): the slot order is the ring's (`SU2xU1ZPlusRing`: slot 0
the SU(2) weight, slot 1 the U(1) charge; with the slots swapped not even the
product labels group alike); `s = ±1` — the zoo's U(1) unit is the ungauged
gauge charge itself, no factor 3 (the ×3 of `a1d4_seeds` belongs to the
SU(3) branching, and a ring isomorphism of `R(SU(2)×U(1))` permuting the basis
sends `μ` to `z^{±1}` anyway); the offsets are all determined.  The certified
maps are exactly one map composed with the powers of ρ, with `s`
alternating in sign (ρ conjugates the U(1)): 4 / 6 / 8 at k = 1 / 2 / 3,
found and certified in 0.5 s / 3.6 s / 28 s (on the table-frame labels; on
the curve-frame labels of the same day, 0.1 s / 1.9 s / 26 s with the two
algebras already built, the same classes, `s` and certificates).  Served is
the first with `s = +1` in the search order; the others serve the same seed
traces (the suite in the source repository).  Any other outcome raises.

Depth
-----
Every zoo seed is an `A1DevenKAlg(k)` generator, whose trace sums gauged
traces of seeds of `U1A1DevenConeKAlgebra(k)` — an odd curve, or a
non-crossing pair of a +1 and a −1 curve, times a power of `E` — and since
2026-09-24 the gauged class serves those from closed forms
(`u1a1deven_seed_characters`, MEASURED against the exact transport; recorded
in the design notes), to any order: all 39 / 120
seeds and `Tr 1` through `𝖖⁴⁰` in 1.1 s at a1d6 / 2.9 s at a1d8 (peak RSS
119 / 302 MB), after the map's discovery (2 s / 25 s; measured 2026-09-24).  Until then the gauged traces were the RG
transport's, whose limit on the length of an A1Dodd word
(`u1a1deven_trace_transport._MAX_WORD_DEGREE`) stopped the seeds at `𝖖¹²`
(k = 2) and `𝖖⁸` (k = 3).  If the transport is reached after all — the
closed forms switched off (`seed_closed_forms=False` on the gauged class) —
its `ValueError` still becomes a `NotImplementedError` naming the entry, the
seed, the order, the class and the limit (the served route never truncates
without notice).  `Tr 1` is Creutzig's closed form.

The KAlgebraIso
---------------
Whether the seeds maps are `KAlgebraIso`s was an open question; they were
not, and `kalgebra_iso(short_id, native=None)` makes
this one a `KAlgebraIso`.  The zoo keeps its SU(2)×U(1) flavour in its
coefficients and `A1DevenKAlg(k)` in its labels, and a `KAlgebraIso`
multiplies coefficients through.  So the source is the zoo's Z-form wrapper
`finite_su2u1_zform.FiniteSU2U1ZKAlgebra(native)`, whose label
`(b, (word, m))` is `χ_b·μ^m·L_word` and whose ring,
`TensorZPlusRing(R(SU(2)), R(U(1)))`, is `A1DevenKAlg`'s:

  * forward: `(b, (word, m)) ↦ (F, −c(word) − s·m, b)`, `(F, −c(word), 0) =
    to_ungauged_label(word)` — the image `χ_b·z^{s·m + c(word)}·L_{(F, 0, 0)}`;
  * inverse: `(F, e, κ) ↦ (κ, (word, s·(−c(word) − e)))`, where `word` is the
    zoo word whose generators' images make up `F`.  The charge-0 curves come
    one generator each; the ±1 curves are paired across the sign of their
    magnetic charge (`UngaugedKAlgebra.mag`).  The word is then made canonical
    in the zoo's cone.

The served map has `s = +1`, so the zoo's `μ` is `z` itself and the two rings
agree key for key; no base change is needed (contrast `aodd_seeds`).  The
wrapper gained the flavour-lift coordinate the same day, for the section
check.  The battery also found a defect in the wrapper, fixed the same day:
its `trace` dropped the SU(2) character of the label, returning `Tr(1)` for
`χ₁`; the trace check fails on `χ₁` before the fix.

Certified by the suite in the source repository, measured 2026-09-26 (shared
machine).  `verify_all` passes all five checks — unit, round trip,
multiplicative, ρ-equivariant, trace-equivariant through `𝖖¹²` — on the
identity, every generator, `χ₁` and `μ^{±1}`, and on every ordered pair of
them each way:

| entry | pairs each way | time | round trip on product labels |
|---|---|---|---|
| a1d4 | 11² = 121 | 0.3 s | 77 + 76 |
| a1d6 | 42² = 1,764 | 3.3 s | 871 + 819 |
| a1d8 (`--slow`) | 123² = 15,129 | 242 s | 6,512 + 6,099 |
| a1d8 (default) | 33² = 1,089 | 37 s | 1,295 + 1,388 |

a1d8's default run takes the pairs among every 4th generator and the three
flavour characters, with the other four checks on all 124 samples (37 s with
the map's discovery).  At a1d4 the iso is the method's positive control:
`SU3ADKAlg` serves that entry's traces, so the trace check compares two
independent routes there.  `verify_maps_section_to_section_1drep` holds both
ways.  Negative controls at a1d4 and a1d6, each failing the battery: the U(1)
normalisation flipped (`s → −s`), the offset of one generator moved by 1, and
two generators transposed.  `finite_kalgebra_objects.kalgebra_object`
registers the pair as 'z-form' → 'a1deven'.

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import RElement, RLaurent

from regen import _canonical_zoo_word, _zoo_word_of


__all__ = [
    "K_OF",
    "ungauged_algebra",
    "generator_map",
    "product_agreement",
    "rho_agreement",
    "to_ungauged_label",
    "to_ungauged",
    "seeds",
    "kalgebra_iso",
]

#: zoo id -> `k` of `A1DevenKAlg(k)` (`[A₁,D₂ₖ₊₂]`); a1d4 is a witness only
K_OF = {"a1d4": 1, "a1d6": 2, "a1d8": 3}

_STATE: dict = {}         # k -> per-k state dict


def _k(short_id: str) -> int:
    if short_id not in K_OF:
        raise KeyError(
            f"a1deven_seeds: {short_id!r} is not an [A1,D_(2k+2)] zoo entry "
            f"(expected one of {sorted(K_OF)})")
    return K_OF[short_id]


def _st(short_id: str) -> dict:
    k = _k(short_id)
    st = _STATE.get(k)
    if st is None:
        st = _STATE[k] = {"k": k, "sid": short_id}
    return st


def ungauged_algebra(short_id: str):
    """The (shared, per-process) `A1DevenKAlg(k)` for this entry."""
    st = _st(short_id)
    if "A" not in st:
        from a1deven_kalg import A1DevenKAlg
        st["A"] = A1DevenKAlg(st["k"])
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

def _zoo_terms(x: Element) -> dict:
    """`{(word, 𝖖-power, key): c}` of a zoo `Element` (`key` the zoo ring's
    basis key, as stored)."""
    out: dict = {}
    for w, c in x.terms.items():
        if isinstance(c, RLaurent):
            items = [(q, key, v) for q, r in c.coeffs.items()
                     for key, v in r.terms.items() if v]
        else:
            items = [(q, (0, 0), v) for q, v in c._coeffs.items() if v]
        for q, key, v in items:
            out[(w, q, key)] = out.get((w, q, key), 0) + v
    return {t: v for t, v in out.items() if v}


def _deven_terms(x: Element) -> dict:
    """`{(section, 𝖖-power, (b, f)): c}` of an `A1DevenKAlg` `Element` — the
    R-form view.  A Z-form term on the label `(F, e, κ)` is
    `z^{−e}·χ_κ·L_{(F, 0, 0)}` (the class's flavour lift,
    `A1DevenKAlg.r_label_decompose`), so it goes to the section `(F, 0, 0)`
    with key `(κ, −e)`; an element with `RLaurent` coefficients (keys
    `(b, (f,))` flattened to `(b, f)`) is read as it stands."""
    out: dict = {}
    for L, c in x.terms.items():
        if isinstance(c, RLaurent):
            items = [(L, q, (b, f), v) for q, r in c.coeffs.items()
                     for (b, (f,)), v in r.terms.items() if v]
        else:
            F, e, kap = L
            items = [((F, 0, 0), q, (kap, -e), v)
                     for q, v in c._coeffs.items() if v]
        for lab, q, key, v in items:
            out[(lab, q, key)] = out.get((lab, q, key), 0) + v
    return {t: v for t, v in out.items() if v}


def _signature(terms: dict) -> tuple:
    """Flavour-blind invariant of a product: the multiset over its terms of
    the multisets of (𝖖-power, integer coefficient)."""
    per: dict = {}
    for (lab, q, _key), v in terms.items():
        per.setdefault(lab, []).append((q, v))
    return tuple(sorted(tuple(sorted(x)) for x in per.values()))


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
                raise AssertionError(f"a1deven_seeds: ρ is not a permutation "
                                     f"of the generators at {g!r}")
            orbit.append(x)
            seen.add(x)
            x = rho(x)
        out.append(orbit)
    return out


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _products(short_id: str):
    st = _st(short_id)
    if "products" not in st:
        Z, A = _zoo(short_id), ungauged_algebra(short_id)
        n = len(Z.cone_data().mult_gens())
        gens = A.mult_generators()
        zp = {(a, b): _zoo_terms(Z.multiply(((a, 1),), ((b, 1),)))
              for a in range(n) for b in range(n)}
        ap = {(x, y): _deven_terms(A.multiply(x, y)) for x in gens for y in gens}
        st["products"] = (n, gens, zp, ap)
    return st["products"]


def _deven_rho_label(A, label):
    """`A1DevenKAlg`'s ρ on a label, E-power dropped: the gauged ρ-image's
    E-free label (the E-power is a `z`-unit in the ungauged frame)."""
    F, _e, _kap = A._G.rho(label)
    return (F, 0, 0)


def _candidate_bijections(short_id: str) -> list:
    """The ρ-equivariant bijections `σ` (zoo index -> `A1DevenKAlg` generator
    label) whose generator products agree with the zoo's in the flavour-blind
    signature, in the search order."""
    Z, A = _zoo(short_id), ungauged_algebra(short_id)
    n, gens, zp, ap = _products(short_id)
    zo = _orbits(range(n), lambda g: Z._rho_perm.get(g, g))
    ao = _orbits(gens, lambda L: _deven_rho_label(A, L))
    if sorted(map(len, zo)) != sorted(map(len, ao)):
        raise AssertionError(
            f"a1deven_seeds({short_id}): ρ-orbit profiles differ "
            f"({sorted(map(len, zo))} vs {sorted(map(len, ao))})")
    zs = {key: _signature(t) for key, t in zp.items()}
    asg = {key: _signature(t) for key, t in ap.items()}
    order = sorted(range(len(zo)), key=lambda i: (-len(zo[i]), i))

    def placed(i, j, t):
        return {g: ao[j][(p + t) % len(ao[j])] for p, g in enumerate(zo[i])}

    found = []

    def fits(assign, m):
        for a in m:
            for b in m:
                if zs[(a, b)] != asg[(m[a], m[b])]:
                    return False
        for m2 in assign.values():
            for a in m:
                for b in m2:
                    if (zs[(a, b)] != asg[(m[a], m2[b])]
                            or zs[(b, a)] != asg[(m2[b], m[a])]):
                        return False
        return True

    def walk(idx, assign, used):
        if idx == len(order):
            sigma = {}
            for m in assign.values():
                sigma.update(m)
            found.append(sigma)
            return
        i = order[idx]
        for j in range(len(ao)):
            if j in used or len(ao[j]) != len(zo[i]):
                continue
            for t in range(len(ao[j])):
                m = placed(i, j, t)
                if fits(assign, m):
                    assign[i] = m
                    used.add(j)
                    walk(idx + 1, assign, used)
                    del assign[i]
                    used.discard(j)

    walk(0, {}, set())
    return found


def _word_image(A, sigma: dict, word, cache: dict):
    """`(ℓ(w), f(w))`: the single label of the product of the letters' images
    and its `z`-power, or `None` if that product is not a single
    flavour-neutral term (then `σ` is not the map)."""
    hit = cache.get(word)
    if hit is not None or word in cache:
        return hit
    lab, f = A.identity(), 0
    for (g, p) in word:
        for _ in range(p):
            prod = A.multiply(lab, sigma[g])
            if len(prod.terms) != 1:
                cache[word] = None
                return None
            ((F, e, kap), c), = prod.terms.items()
            # one term, one 𝖖-power, SU(2)-neutral; its E-power is z^{−e}
            if kap != 0 or len([q for q, v in c._coeffs.items() if v]) != 1:
                cache[word] = None
                return None
            lab, f = (F, 0, 0), f - e
    cache[word] = (lab, f)
    return cache[word]


def _eliminate(equations, unknowns):
    """Exact elimination over `Q` of the equations `Σ co[u]·u = r`;
    `(solution, free, consistent)`, `solution` over the determined unknowns.
    Pivots are kept fully reduced (each in terms of non-pivot unknowns only),
    so an equation is reduced in one pass; once every unknown is determined the
    remaining equations are checked by evaluation."""
    piv: dict = {}
    rank = {u: i for i, u in enumerate(unknowns)}
    n = len(unknowns)
    for co, r in equations:
        r = Fraction(r)
        if len(piv) == n:                       # all determined: evaluate
            if sum(Fraction(v) * piv[u][1] for u, v in co.items()) != r:
                return None, [], False
            continue
        red: dict = {}
        for u, v in co.items():
            if not v:
                continue
            hit = piv.get(u)
            if hit is None:
                red[u] = red.get(u, 0) + Fraction(v)
            else:
                pco, pr = hit
                for x, w in pco.items():
                    red[x] = red.get(x, 0) + v * w
                r -= v * pr
        red = {x: w for x, w in red.items() if w}
        if not red:
            if r:
                return None, [], False
            continue
        u = min(red, key=rank.__getitem__)
        cu = red.pop(u)
        pco = {x: -w / cu for x, w in red.items()}
        pr = r / cu
        for p, (qco, qr) in list(piv.items()):
            f = qco.get(u)
            if f:
                nq = dict(qco)
                del nq[u]
                for x, w in pco.items():
                    nq[x] = nq.get(x, 0) + f * w
                piv[p] = ({x: w for x, w in nq.items() if w}, qr + f * pr)
        piv[u] = (pco, pr)
    sol = {u: pr for u, (pco, pr) in piv.items() if not pco}
    return sol, [u for u in unknowns if u not in sol], True


def _solve_u1(short_id: str, sigma: dict, slots: tuple, cache: dict):
    """The U(1) normalisation `s` and the offsets `c_g` for the bijection `σ` and the
    zoo key slot order `slots = (SU(2) slot, U(1) slot)`, as the unique
    solution of the product and ρ equations; `None` if there is none or it is
    not unique."""
    Z, A = _zoo(short_id), ungauged_algebra(short_id)
    n, gens, zp, ap = _products(short_id)
    su, uu = slots
    eqs = []
    delta = getattr(Z, "_rho_delta", None) or {}
    for g in range(n):                   # ρ first: it ties each orbit together
        F2, e2, _k2 = A._G.rho(sigma[g])
        rg = Z._rho_perm.get(g, g)
        if (F2, 0, 0) != sigma[rg]:
            return None
        co = {"s": delta.get(g, (0,))[0], rg: 1}
        co[g] = co.get(g, 0) + 1
        eqs.append((co, -e2))
    for (a, b), terms in zp.items():
        zg, ag = {}, {}
        for (w, q, key), v in terms.items():
            im = _word_image(A, sigma, w, cache)
            if im is None:
                return None
            zg.setdefault((im[0], q, key[su]), []).append((key[uu], v, w, im[1]))
        for (L, q, (bb, f)), v in ap[(sigma[a], sigma[b])].items():
            ag.setdefault((L, q, bb), []).append((f, v))
        if set(zg) != set(ag):
            return None
        for key, zl in zg.items():
            al = ag[key]
            if len(zl) != 1 or len(al) != 1:
                continue
            (m, v, w, fw), = zl
            (fa, va), = al
            if v != va:
                return None
            # fa = s·m + Σ_{g∈w} p·c_g + f(w) − c_a − c_b
            co: dict = {"s": m}
            for (g, p) in w:
                co[g] = co.get(g, 0) + p
            co[a] = co.get(a, 0) - 1
            co[b] = co.get(b, 0) - 1
            eqs.append((co, fa - fw))
    sol, free, ok = _eliminate(eqs, ["s"] + list(range(n)))
    if not ok or free:
        return None
    if any(v.denominator != 1 for v in sol.values()):
        return None
    return int(sol["s"]), {g: int(sol[g]) for g in range(n)}


# ---------------------------------------------------------------------------
# the map as data, and the certificate
# ---------------------------------------------------------------------------

def _image_element(short_id: str, gm: dict, x: Element, cache: dict) -> dict:
    """A zoo `Element` carried to `A1DevenKAlg`'s terms `{(label, q, (b, f)):
    c}` through the map `gm`; `None` if a word has no image."""
    A = ungauged_algebra(short_id)
    sigma, off, s = gm["sigma"], gm["offsets"], gm["scale"]
    su, uu = gm["slots"]
    out: dict = {}
    for (w, q, key), v in _zoo_terms(x).items():
        im = _word_image(A, sigma, w, cache)
        if im is None:
            return None
        lab, fw = im
        cw = sum(off[g] * p for g, p in w) + fw
        t = (lab, q, (key[su], s * key[uu] + cw))
        out[t] = out.get(t, 0) + v
    return {t: v for t, v in out.items() if v}


def _shift(terms: dict, c: int) -> dict:
    return {(L, q, (b, f + c)): v for (L, q, (b, f)), v in terms.items()}


def product_agreement(short_id: str, gm: dict, *, stop_early=False) -> tuple:
    """`(agreeing, total)` over the ordered pairs of zoo generators: the zoo
    product carried over by the map `gm` against `A1DevenKAlg`'s product of
    the images."""
    Z, A = _zoo(short_id), ungauged_algebra(short_id)
    n, gens, zp, ap = _products(short_id)
    sigma, off = gm["sigma"], gm["offsets"]
    cache = gm.setdefault("_words", {})
    ok = tot = 0
    for a in range(n):
        for b in range(n):
            tot += 1
            got = _image_element(short_id, gm,
                                 Z.multiply(((a, 1),), ((b, 1),)), cache)
            want = _shift(ap[(sigma[a], sigma[b])], off[a] + off[b])
            if got is not None and got == want:
                ok += 1
            elif stop_early:
                return ok, n * n
    return ok, tot


def _deven_rho_image(A, label, c: int) -> dict:
    """The honest ρ of `z^c·L_label` in `A1DevenKAlg`'s terms: the gauged ρ
    of the E-shifted label, its E-power read back as a `z`-unit
    (`ρ(E) = E⁻¹`, so `ρ(z) = z⁻¹`)."""
    F, e, kap = label
    F2, e2, _k2 = A._G.rho((F, e - c, kap))
    return {((F2, 0, 0), 0, (kap, -e2)): 1}


def rho_agreement(short_id: str, gm: dict) -> tuple:
    """`(agreeing, total)` over the zoo generators: the zoo's `rho_element`
    carried over by `gm` against the honest ρ of the image."""
    Z, A = _zoo(short_id), ungauged_algebra(short_id)
    n = len(Z.cone_data().mult_gens())
    R = Z.coefficient_ring()
    one = RLaurent(R, {0: RElement(R, {R.one_basis(): 1})})
    cache = gm.setdefault("_words", {})
    ok = 0
    for g in range(n):
        got = _image_element(short_id, gm,
                             Z.rho_element(Element({((g, 1),): one})), cache)
        want = _deven_rho_image(A, gm["sigma"][g], gm["offsets"][g])
        ok += got == want
    return ok, n


def _rho_power(A, label, j: int):
    for _ in range(j):
        label = _deven_rho_label(A, label)
    return label


def generator_map(short_id: str) -> dict:
    """The certified map for this entry: `{"sigma": {g: label}, "offsets":
    {g: c_g}, "scale": s, "slots": (SU(2) slot, U(1) slot)}` (`s` the U(1)
    normalisation) — zoo generator
    `g` is `z^{c_g}·L_{sigma[g]}` and a zoo coefficient `χ_b·μ^m` (b, m in
    `slots`) is `χ_b·z^{s·m}` (module docstring).  Found at first use and
    cached; raises `AssertionError` unless the certified maps are exactly one
    map composed with the powers of ρ."""
    st = _st(short_id)
    if "gm" in st:
        return st["gm"]
    Z, A = _zoo(short_id), ungauged_algebra(short_id)
    n = len(Z.cone_data().mult_gens())
    passing = []
    candidates = _candidate_bijections(short_id)
    for sigma in candidates:
        cache: dict = {}
        for slots in ((0, 1), (1, 0)):
            u1 = _solve_u1(short_id, sigma, slots, cache)
            if u1 is None:
                continue
            gm = {"sigma": sigma, "offsets": u1[1], "scale": u1[0],
                  "slots": slots, "_words": cache}
            if (product_agreement(short_id, gm, stop_early=True)[0] == n * n
                    and rho_agreement(short_id, gm)[0] == n):
                passing.append(gm)
    if not passing:
        raise AssertionError(
            f"a1deven_seeds({short_id}): no ρ-equivariant generator map with a "
            f"U(1) normalisation and offsets reproduces the generator products and ρ "
            f"({len(candidates)} bijections pass the flavour-blind signature)")
    first = passing[0]
    orbit_order = 1
    for g in range(n):
        j, x = 1, Z._rho_perm.get(g, g)
        while x != g:
            j, x = j + 1, Z._rho_perm.get(x, x)
        orbit_order = orbit_order * j // gcd(orbit_order, j)
    shifts = []
    for gm in passing:
        js = [j for j in range(orbit_order)
              if all(gm["sigma"][g] == _rho_power(A, first["sigma"][g], j)
                     for g in range(n))]
        if (not js or gm["slots"] != first["slots"]
                or gm["scale"] != first["scale"] * (-1) ** js[0]
                or abs(gm["scale"]) != 1):
            raise AssertionError(
                f"a1deven_seeds({short_id}): two certified generator maps not "
                f"related by ρ; the seed traces would be ambiguous")
        shifts.append(js[0])
    if sorted(shifts) != list(range(orbit_order)):
        raise AssertionError(
            f"a1deven_seeds({short_id}): the certified maps are not one map "
            f"composed with each of the {orbit_order} powers of ρ (powers "
            f"{sorted(shifts)})")
    served = next((gm for gm in passing if gm["scale"] == 1), first)
    st["class"] = passing
    st["gm"] = served
    return served


# ---------------------------------------------------------------------------
# labels and elements
# ---------------------------------------------------------------------------

def to_ungauged_label(short_id: str, word) -> tuple:
    """The `A1DevenKAlg` label of the zoo label `word = ((g, p), …)`: the
    single label `(F, 0, 0)` of the product of the letters' images, carrying
    the E-power `−c(w)` (so `(F, −c, 0) = z^{c}·(F, 0, 0)` in `A1DevenKAlg`,
    whose labels read an E-power as the `z`-unit `z^{−e}`)."""
    gm = generator_map(short_id)
    A = ungauged_algebra(short_id)
    im = _word_image(A, gm["sigma"], tuple(word), gm.setdefault("_words", {}))
    if im is None:
        raise ValueError(f"{short_id}: {word!r} is not a cone monomial")
    (F, _e, _kap), fw = im
    c = sum(gm["offsets"][g] * p for g, p in word) + fw
    return (F, -c, 0)


def to_ungauged(short_id: str, x: Element) -> Element:
    """A zoo `Element` as an `A1DevenKAlg` `Element`, Z-form: the term
    `χ_b·z^f·L_{(F, 0, 0)}` of the image is the label `(F, −f, b)` (the
    class's flavour lift) with an integral coefficient."""
    gm = generator_map(short_id)
    terms = _image_element(short_id, gm, x, gm.setdefault("_words", {}))
    if terms is None:
        raise ValueError(f"{short_id}: an element with a non-monomial label")
    out: dict = {}
    for ((F, _e, _k), q, (b, f)), v in terms.items():
        L = (F, -f, b)
        out[L] = out.get(L, LaurentPoly({})) + LaurentPoly({q: v})
    return Element({L: c for L, c in out.items() if not c.is_zero()})


# ---------------------------------------------------------------------------
# the seed server
# ---------------------------------------------------------------------------

class _DevenSeeds:
    """The zoo's `Tr(1)` and seed traces for one entry, read off
    `A1DevenKAlg(k)`'s trace through the map, in the zoo's data format
    `{𝖖-power: {(SU(2) weight, U(1) charge): int}}`.  Each series is cached at
    the largest order asked and truncated on smaller requests."""

    def __init__(self, short_id: str):
        self.short_id = short_id
        self.k = _k(short_id)
        self._cache: dict = {}

    def _deven_trace(self, label, what: str, K: int):
        A = ungauged_algebra(self.short_id)
        try:
            return A.trace(label, K)
        except ValueError as e:
            if not str(e).startswith("DevenTraceTransport("):
                raise
            from u1a1deven_trace_transport import _shared_transport
            limit = _shared_transport(self.k).max_word_degree
            raise NotImplementedError(
                f"{self.short_id}: the trace of {what} through q^{K} is "
                f"beyond what A1DevenKAlg({self.k}) serves: its trace "
                f"transport stopped at one of its limits — in practice the "
                f"length of an A1Dodd word, max_word_degree = {limit} "
                f"letters at k = {self.k} (u1a1deven_trace_transport."
                f"_MAX_WORD_DEGREE); the seeds are served by closed forms "
                f"unless those are switched off (seed_closed_forms=False on "
                f"the gauged class): {e}") from e

    def _series(self, key, label, c: int, what: str, K: int) -> dict:
        hit = self._cache.get(key)
        if hit is None or hit[0] < K:
            gm = generator_map(self.short_id)
            s = gm["scale"]
            su, uu = gm["slots"]
            tr = self._deven_trace(label, what, K)
            data: dict = {}
            for q, r in tr.coeffs.items():
                if q > K:
                    continue
                row: dict = {}
                for (b, (f,)), v in r.terms.items():
                    if not v:
                        continue
                    m = (f + c) * s              # s = ±1: z^f ↦ μ^{f/s}
                    zkey = [0, 0]
                    zkey[su], zkey[uu] = b, m
                    row[tuple(zkey)] = row.get(tuple(zkey), 0) + int(v)
                row = {kk: v for kk, v in row.items() if v}
                if row:
                    data[q] = row
            hit = (K, data)
            self._cache[key] = hit
        return {q: dict(row) for q, row in hit[1].items() if q <= K}

    def vacuum_trace(self, K: int) -> dict:
        """`Tr(1)` through `𝖖^K`."""
        A = ungauged_algebra(self.short_id)
        return self._series("identity", A.identity(), 0, "the identity", K)

    def seed_trace(self, i: int, K: int) -> dict:
        """`Tr(L_{((i, 1),)})` of the zoo generator `i` through `𝖖^K`."""
        gm = generator_map(self.short_id)
        if i not in gm["sigma"]:
            raise KeyError(f"{self.short_id}: no multiplicative generator {i!r}")
        return self._series(i, gm["sigma"][i], gm["offsets"][i],
                            f"seed {i}", K)


def seeds(short_id: str) -> _DevenSeeds:
    """The seed server for this entry: `.vacuum_trace(K)` and
    `.seed_trace(i, K)`, as `elem_traces._seed_series` reads them."""
    st = _st(short_id)
    if "seeds" not in st:
        st["seeds"] = _DevenSeeds(short_id)
    return st["seeds"]


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

_ONE = LaurentPoly.one()


def _word_tables(short_id: str) -> tuple:
    """`(singles, pairs, side)` for `_zoo_word_of`: the zoo generators whose
    image `σ(g)` is one curve, keyed by it; those whose image is a pair of
    curves, keyed by the pair; and each paired curve's magnetic charge `mag`
    in `A1DevenKAlg(k)` (nonzero, of opposite signs within a pair —
    checked)."""
    st = _st(short_id)
    if "words" not in st:
        A = ungauged_algebra(short_id)
        singles: dict = {}
        pairs: dict = {}
        for g, (F, _e, _kap) in generator_map(short_id)["sigma"].items():
            curves = tuple(c for (c, m) in F for _ in range(m))
            if len(curves) == 1:
                singles[curves[0]] = g
            elif len(curves) == 2:
                pairs[curves] = g
            else:
                raise AssertionError(f"a1deven_seeds({short_id}): generator "
                                     f"{g} has {len(curves)} curves")
        side = {c: A.mag((((c, 1),), 0, 0)) for pr in pairs for c in pr}
        if any(side[a] * side[b] >= 0 for a, b in pairs):
            raise AssertionError(f"a1deven_seeds({short_id}): a paired "
                                 f"generator whose curves do not carry "
                                 f"opposite magnetic charges")
        st["words"] = (singles, pairs, side)
    return st["words"]


def kalgebra_iso(short_id: str, native=None):
    """The generator map as a `KAlgebraIso` from the zoo's Z-form wrapper
    `FiniteSU2U1ZKAlgebra(native)` onto `A1DevenKAlg(k)` (module docstring,
    "The KAlgebraIso"): `(b, (word, m)) ↦ (F, −c(word) − s·m, b)`, and back
    through the generators' images, the word made canonical in the zoo's
    cone.

    `native` is the zoo standalone to wrap, an instance of
    `FINITE_KALGEBRAS[short_id]` (default: a fresh one;
    `finite_kalgebra_objects.kalgebra_object` passes its `'cone-frozen'`
    realization); the target is the shared `ungauged_algebra(short_id)`.  The
    label maps build the generator map (the search and its certificate: about
    2 s at a1d6 and 25 s at a1d8) and the inverse's tables at first use."""
    import finite_kalgebras as fk
    from finite_su2u1_zform import FiniteSU2U1ZKAlgebra
    from kalgebra_iso import KAlgebraIso
    _k(short_id)
    cls = fk.FINITE_KALGEBRAS[short_id]
    if native is None:
        native = cls()
    elif not isinstance(native, cls):
        raise TypeError(f"a1deven_seeds.kalgebra_iso({short_id!r}): native is "
                        f"a {type(native).__name__}, not a {cls.__name__}")
    Z = FiniteSU2U1ZKAlgebra(native)
    A = ungauged_algebra(short_id)

    def scale() -> int:
        gm = generator_map(short_id)
        if gm["slots"] != (0, 1):
            # FiniteSU2U1ZKAlgebra reads the zoo key (n, m) as SU(2) weight,
            # U(1) charge; the map's slot order must be the same.
            raise AssertionError(f"a1deven_seeds({short_id}): slot order "
                                 f"{gm['slots']}, the Z-form wrapper's is (0, 1)")
        return gm["scale"]

    def forward(label) -> Element:
        b, (word, m) = label
        F, e0, _kap = to_ungauged_label(short_id, word)
        return Element({(F, e0 - scale() * m, b): _ONE})

    def inverse(label) -> Element:
        F, e, kappa = label
        counts: dict = {}
        for (c, m) in F:
            counts[c] = counts.get(c, 0) + m
        word = _canonical_zoo_word(native,
                                   _zoo_word_of(counts, *_word_tables(short_id)))
        F2, e0, _kap = to_ungauged_label(short_id, word)
        if F2 != tuple(F):
            raise ValueError(f"{short_id}: the zoo word {word!r} of {label!r} "
                             f"maps to {F2!r}")
        return Element({(kappa, (word, scale() * (e0 - e))): _ONE})

    return KAlgebraIso(Z, A, forward, inverse,
                       name=f"{short_id}[z-form→a1deven]")
