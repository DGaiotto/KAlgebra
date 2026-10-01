"""`SU3ADKAlg` ([A₁,D₄] with its SU(3) flavour) on the curves of
`A1DevenKAlg(1)`: the map of its sections, its geometric labels, and its
`Tr_1`, `Tr_T` / `Tr_D` seeds from the even-D k = 1 closed forms (the design record; the "Geometric labels" block of `src/cone/su3_ad_kalg.py`
and the serving seeds `sl3_su3_traces._ClosedFormSeeds`).

Restricted to SU(2)×U(1) (`zplus_ring.su3_to_su2u1_hom`, U(1) charge `Y`)
`SU3ADKAlg` is `A1DevenKAlg(1)`; `A1DevenKAlg(1)`'s U(1) fugacity `z` has
`Y`-charge `3s`, `s = ±1`, and a generator `X` goes to a generator `F(X)` with
a `Y`-offset `Δ(X)`.  The restriction is injective on class functions, so a
product or a trace agreeing after it agrees in `R(SU(3))`.

Positive controls first; a failing control aborts the run:
  * the served `Tr_1` — the seed route on the identity: the gauge-charge sum
    of Creutzig's gauge tower with the measure restored, restricted to the
    SU(3) torus by the served weight map — equals its witness, the
    Kac–Wakimoto vacuum character of `sl(3)_{−3/2}` (the route until
    2026-09-24), through 𝖖⁶⁰ (𝖖¹⁰⁰ with --slow), with the cost of both;
  * the certificate on a known answer: the composite of the two certified zoo
    maps (zoo a1d4 → `SU3ADKAlg` restricted, `a1d4_seeds`;
    zoo a1d4 → `A1DevenKAlg(1)`, `a1deven_seeds`) reproduces
    all 64 generator products and ρ on all 8 generators, and it is the closed
    form of `su3_ad_kalg`.
Then:
  * the map, searched: every ρ-equivariant bijection of the eight generators
    (2 orbit pairings × 4 × 4 shifts), both signs `s`, the two base offsets
    `Δ(T_0)`, `Δ(D_0)` solved exactly from the uniquely matched terms of the
    64 generator products (the rest of each orbit follows from ρ); exactly 4
    maps reproduce every product and ρ — the closed form composed with the
    4 powers of ρ, `s` alternating.  Negative controls, each caught by the
    products: `Δ(D_0)` moved by 3, the offsets without ρ's power of `E`, `s`
    flipped, the `z`-charge read as `Y` (no factor 3);
  * the sections: on the window `2a + b ≤ 8` the images are distinct balanced
    non-crossing multisets and are all of them (145), and `A1DevenKAlg(1)`'s
    product of a section's letters' images is that one label with no power of
    `z`, so the offsets add; `geometric_label` is total and injective on the
    window's labels (four SU(3) weights), and ρ rotates the curves and
    conjugates the weight;
  * the seeds: `Tr_T`, `Tr_D` (both parities) equal the witness, the forward
    orthonormality pass `SU3ElemTraces`, through 𝖖⁴⁰ (𝖖⁸⁰ with --slow), with
    the cost of both; every `T_i` gives `Tr_T`, `D_{i+2}` gives `D_i`'s,
    `Tr(D_1) = ⋆Tr(D_0)`; the gauge charges contributing sit well inside the
    window `|n| ≤ K + 1`; negative controls: `D_0`'s offset moved by 6
    (the smallest move keeping the SU(3) weight lattice) and `D_1`'s image
    in place of `D_0`'s differ from the witness, a move by 3 is refused;
  * product traces (`trace_word`, `inner_product`) ask their seeds for
    exactly the depth their monomials' reductions read: from a fresh provider
    `(T_i²D_i)·(T_i²D_i)` equals its value from a deepened one (the former
    padded depth was one order short there and got the top order wrong), and
    a provider one order too shallow is refused, not truncated;
  * composite labels: on the 25 sections with `2a + b ≤ 3`, `SU3ADKAlg`'s
    trace, restricted, equals `A1DevenKAlg(1)`'s trace of the image shifted by
    the offset, through 𝖖⁸ (𝖖¹² with --slow) — taken on the transport route
    (`seed_closed_forms=False` on the gauged class), so that on the seeds too
    it is independent of the closed forms;
  * the serving path, in a fresh process: `Tr(1)` and the traces of `T_0`,
    `D_0`, `D_1` through 𝖖⁴⁰, and zoo a1d4's `Tr(1)` and eight seeds through
    𝖖⁴⁰, with no method of `SU3ElemTraces`, no function of the Kac–Wakimoto
    vacuum and no BPS function run, and no module of the even-D classes (the
    gauged and ungauged classes, their frame, the transport, the ungauger)
    and no bootstrap imported — after a control run of the same script, with
    the forward pass served and `a1deven_kalg` imported, is flagged.

Run:  `python3 run_tests.py` [--slow]
"""
import os
import subprocess
import sys
import time
from fractions import Fraction
from itertools import permutations, product

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

import sl3_su3_traces as T
from su3_ad_kalg import (SU3ADKAlg, _TILE_LETTERS, _deven_letter_images,
                         _deven_section)

SLOW = "--slow" in sys.argv[1:]
K_SEEDS = 80 if SLOW else 40
K_TR1 = 100 if SLOW else 60
K_COMPOSITE = 12 if SLOW else 8
PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# the two algebras and their generator products
# ---------------------------------------------------------------------------

_ST: dict = {}


def _alg():
    if "A" not in _ST:
        from a1deven_kalg import A1DevenKAlg
        from zplus_ring import su3_to_su2u1_hom
        _ST["A"] = SU3ADKAlg()
        _ST["B"] = A1DevenKAlg(1)
        _ST["br"] = su3_to_su2u1_hom()
    return _ST["A"], _ST["B"], _ST["br"]


def _su3_gens():
    A, _B, _br = _alg()
    return [A.T(i) for i in range(4)] + [A.D(i) for i in range(4)]


def _letter(label):
    """The letter `('T', i)` / `('D', i)` of an `SU3ADKAlg` generator."""
    tile, a, _b, _p, _q = label
    l_t, l_d = _TILE_LETTERS[tile]
    return l_t if a else l_d


def _deven_rho(label):
    """`A1DevenKAlg(1)`'s ρ on an E-free label: the E-free image and the power
    of `E` the gauged ρ produces (`E = z⁻¹`)."""
    _A, B, _br = _alg()
    F2, e2, k2 = B._G.rho(label)
    assert k2 == label[2]
    return (F2, 0, 0), e2


def _products():
    if "products" not in _ST:
        A, B, _br = _alg()
        sg = _su3_gens()
        dg = B.mult_generators()
        _ST["products"] = ({(x, y): A.multiply(x, y) for x in sg for y in sg},
                           {(u, v): B.multiply(u, v) for u in dg for v in dg})
    return _ST["products"]


def _union(parts):
    acc = {}
    for curves, m in parts:
        for c, k in curves:
            acc[c] = acc.get(c, 0) + k * m
    return tuple(sorted((c, k) for c, k in acc.items() if k))


def _section_image(section, img):
    """`(curves, Δ)` of a section under `img = {letter: (curves, Δ)}` (the
    offsets may be affine dicts `{unknown: coefficient}`, `'1'` the constant)."""
    tile, a, b = section
    parts, off = [], {}
    for L, m in zip(_TILE_LETTERS[tile], (a, b)):
        if m:
            curves, d = img[L]
            parts.append((curves, m))
            for u, c in (d.items() if isinstance(d, dict) else (("1", d),)):
                off[u] = off.get(u, 0) + m * c
    return _union(parts), off


def _su3_terms(x, img):
    """`{(curves, 𝖖-power, κ): [(Y, offset of the section, c)]}`: an
    `SU3ADKAlg` element, each SU(3) character branched to `(κ, Y)`, each
    section carried to its image."""
    _A, _B, br = _alg()
    out = {}
    for (tile, a, b, p, q), lp in x.terms.items():
        curves, off = _section_image((tile, a, b), img)
        for e, v in lp._coeffs.items():
            if not v:
                continue
            for (kap, Y), c in br.apply_basis((p, q)).terms.items():
                if c:
                    out.setdefault((curves, e, kap), []).append((Y, off, v * c))
    return out


def _dv_terms(x):
    """`{(curves, 𝖖-power, κ): [(f, c)]}` of an `A1DevenKAlg(1)` element
    (Z-form: the label `(F, e, κ)` is `z^{−e}·χ_κ·L_(F,0,0)`, so `f = −e`)."""
    out = {}
    for (F, e, kap), lp in x.terms.items():
        for q, v in lp._coeffs.items():
            if v:
                out.setdefault((F, q, kap), []).append((-e, v))
    return out


def _flat_su3(x, img):
    out = {}
    for key, lst in _su3_terms(x, img).items():
        for Y, off, c in lst:
            kk = key + (Y + off.get("1", 0),)
            out[kk] = out.get(kk, 0) + c
    return {k: v for k, v in out.items() if v}


def _flat_dv(x, s, shift, factor=3):
    out = {}
    for key, lst in _dv_terms(x).items():
        for f, c in lst:
            kk = key + (factor * s * f + shift,)
            out[kk] = out.get(kk, 0) + c
    return {k: v for k, v in out.items() if v}


def agreement(img, s, factor=3):
    """`(products agreeing, 64, ρ agreeing, 8)` for the map `img = {letter:
    (curves, Δ)}` (numeric offsets) with `z` of `Y`-charge `factor·s`."""
    A, _B, _br = _alg()
    sp, dp = _products()
    sg = _su3_gens()
    lab = {x: (img[_letter(x)][0], 0, 0) for x in sg}
    off = {x: img[_letter(x)][1] for x in sg}
    ok = 0
    for x in sg:
        for y in sg:
            ok += (_flat_su3(sp[(x, y)], img)
                   == _flat_dv(dp[(lab[x], lab[y])], s, off[x] + off[y], factor))
    rok = 0
    for x in sg:
        (F2, drift) = _deven_rho(lab[x])
        rx = A.rho(x)
        rok += F2 == lab[rx] and off[rx] == -off[x] - factor * s * drift
    return ok, 64, rok, 8


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _orbit(x, rho):
    out, y = [x], rho(x)
    while y != x:
        out.append(y)
        y = rho(y)
    return out


def _affine(*signed):
    out = {}
    for sgn, d in signed:
        for u, c in d.items():
            out[u] = out.get(u, 0) + sgn * c
    return {u: c for u, c in out.items() if c}


def _solve(eqs, unknowns):
    """Exact elimination over `Q` of `[(coefficients, rhs)]`; the solution
    or `None` (inconsistent, or an unknown left free)."""
    rows = [({u: Fraction(c) for u, c in co.items()}, Fraction(r))
            for co, r in eqs]
    piv = []
    for u in unknowns:
        i = next((i for i, (co, _r) in enumerate(rows) if co.get(u)), None)
        if i is None:
            return None
        co, r = rows.pop(i)
        cu = co[u]
        co = {k: v / cu for k, v in co.items()}
        r = r / cu
        new = []
        for co2, r2 in rows:
            f = co2.get(u, 0)
            if f:
                co2 = {k: co2.get(k, 0) - f * co.get(k, 0)
                       for k in set(co2) | set(co)}
                co2 = {k: v for k, v in co2.items() if v}
                r2 = r2 - f * r
            new.append((co2, r2))
        rows = new
        piv.append((u, co, r))
    if any(co or r for co, r in rows):
        return None
    sol = {}
    for u, co, r in reversed(piv):
        sol[u] = r - sum(v * sol[k] for k, v in co.items() if k != u)
    return sol


def search():
    """Every certified map: `[(img, s)]`, and a tally of why the other
    candidates failed."""
    A, B, _br = _alg()
    sp, dp = _products()
    so = [_orbit(A.T(0), A.rho), _orbit(A.D(0), A.rho)]
    do, seen = [], set()
    for g in B.mult_generators():
        if g not in seen:
            o = _orbit(g, lambda L: _deven_rho(L)[0])
            seen |= set(o)
            do.append(o)
    assert sorted(map(len, so)) == sorted(map(len, do)) == [4, 4]
    found, tally = [], {}
    for pairing in permutations(range(2)):
        for shifts in product(range(4), repeat=2):
            sigma = {x: do[j][(i + sh) % 4]
                     for sorb, j, sh in zip(so, pairing, shifts)
                     for i, x in enumerate(sorb)}
            for s in (1, -1):
                why = _try(so, sigma, s, sp, dp, found)
                tally[why] = tally.get(why, 0) + 1
    return found, tally


def _try(so, sigma, s, sp, dp, found):
    off = {}
    for sorb, u in zip(so, ("uT", "uD")):
        cur = {u: 1}
        for x in sorb:
            off[x] = cur
            _F2, drift = _deven_rho(sigma[x])
            cur = _affine((-1, cur), (1, {"1": -3 * s * drift}))
        if _affine((1, cur), (-1, off[sorb[0]])):
            return "the offsets do not close under ρ"
    img = {_letter(x): (sigma[x][0], off[x]) for x in sigma}
    eqs = []
    for (x, y), prod_s in sp.items():
        st = _su3_terms(prod_s, img)
        dt = _dv_terms(dp[(sigma[x], sigma[y])])
        if set(st) != set(dt):
            return "the product labels differ"
        for key, lst in st.items():
            if len(lst) == 1 and len(dt[key]) == 1:
                (Y, o, c), = lst
                (f, cd), = dt[key]
                if c != cd:
                    return "a coefficient differs"
                lhs = _affine((1, o), (-1, off[x]), (-1, off[y]))
                eqs.append(({u: v for u, v in lhs.items() if u != "1"},
                            3 * s * f - Y - lhs.get("1", 0)))
    sol = _solve(eqs, ["uT", "uD"])
    if sol is None or any(v.denominator != 1 for v in sol.values()):
        return "no integral solution for the offsets"
    num = {L: (curves, d.get("1", 0) + sum(int(sol[u]) * c
                                           for u, c in d.items() if u != "1"))
           for L, (curves, d) in img.items()}
    if agreement(num, s) != (64, 64, 8, 8):
        return "the certificate fails"
    found.append((num, s))
    return "certified"


def _rotate(curves, j):
    return tuple(sorted((((x + j) % 4, l), m) for (x, l), m in curves))


# ---------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------

def _kw_z(K):
    return {2 * k: T.char_to_zlaurent(c)
            for k, c in T.vacuum_character(K // 2 + 2).items() if 2 * k <= K}


def _upto(series, K):
    return {q: z for q, z in series.items() if q <= K and z}


def test_control_vacuum(K=K_TR1):
    """The served `Tr_1` — the gauge-charge sum of Creutzig's gauge tower with
    the measure restored, on the SU(3) torus (`_deven_seed_z((), 0, K)`, the
    identity's image) — against its witness, the Kac–Wakimoto vacuum
    character of `sl(3)_{−3/2}`, which served it until 2026-09-24.  Run first:
    it is also the positive control of the route's weight map and sum."""
    t0 = time.time()
    got = _upto(T._ClosedFormSeeds().ensure(K).series(("Tr_1",)), K)
    t1 = time.time()
    want = _upto(_kw_z(K), K)
    t2 = time.time()
    return check(f"control: the served Tr_1 (the gauge-charge sum of Creutzig's "
                 f"tower, measure restored, on the SU(3) torus) = the witness "
                 f"Kac–Wakimoto vacuum through q^{K} (served {t1 - t0:.2f} s "
                 f"with the T / D seeds, witness {t2 - t1:.2f} s)", got == want)


def _composite_map():
    """The composite of the two served zoo maps as `{letter: (curves, Δ)}`
    and `s`: zoo generator `g` is `SU3ADKAlg`'s `z2s[g]` with extra branching
    charge `c_g` (`a1d4_seeds`), and `z^{c'_g}·L_{σ(g)}` of `A1DevenKAlg(1)`
    with `μ^m ↦ z^{s·m}` (`a1deven_seeds`), `μ` the zoo's U(1) = the cube of
    the branching one; so `L_{z2s[g]} = μ_Y^{3s·c'_g − c_g}·L_{σ(g)}`."""
    import a1d4_seeds
    import a1deven_seeds
    z2s, c = a1d4_seeds.generator_map()
    gm = a1deven_seeds.generator_map("a1d4")
    s = gm["scale"]
    return ({_letter(z2s[g]): (gm["sigma"][g][0], 3 * s * gm["offsets"][g] - c[g])
             for g in z2s}, s)


def test_control_composite():
    img, s = _composite_map()
    ok = agreement(img, s)
    same = img == _deven_letter_images() and s == 1
    return check(f"control: the composite of the two certified zoo maps "
                 f"reproduces {ok[0]}/{ok[1]} generator products and ρ on "
                 f"{ok[2]}/{ok[3]}, and is su3_ad_kalg's closed form (s = +1)",
                 ok == (64, 64, 8, 8) and same)


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------

def test_search():
    t0 = time.time()
    found, tally = search()
    closed = _deven_letter_images()
    js = []
    for img, s in found:
        j = [j for j in range(4)
             if all(img[L][0] == _rotate(closed[L][0], j) for L in closed)]
        js.append((j[0] if len(j) == 1 else None, s))
    check(f"search: {sum(tally.values())} candidates, {tally}; the "
          f"{len(found)} certified maps are the closed form composed with "
          f"ρ^j, (j, s) = {sorted(js)} [{time.time() - t0:.1f} s]",
          len(found) == 4 and sorted(js) == [(0, 1), (1, -1), (2, 1), (3, -1)]
          and any(img == closed and s == 1 for img, s in found))


def test_negative_controls():
    closed = _deven_letter_images()
    moved = {L: (c, d + (3 if L[0] == "D" and L[1] % 2 == 0 else
                         -3 if L[0] == "D" else 0))
             for L, (c, d) in closed.items()}
    a = agreement(moved, 1)
    check(f"negative control: Δ(D_0) moved by 3 (ρ-consistently along the "
          f"orbit) — {a[0]}/64 products", a[0] < 64)
    nodrift = {}
    for kind, d0 in (("T", 0), ("D", 1)):
        for i in range(4):
            nodrift[(kind, i)] = (closed[(kind, i)][0], d0 * (-1) ** i)
    a = agreement(nodrift, 1)
    check(f"negative control: the offsets without ρ's power of E "
          f"(Δ(ρX) = −Δ(X)) — {a[0]}/64 products, ρ {a[2]}/8",
          a[0] < 64 and a[2] < 8)
    a = agreement(closed, -1)
    check(f"negative control: s flipped — {a[0]}/64 products, ρ {a[2]}/8",
          a[0] < 64)
    a = agreement(closed, 1, factor=1)
    check(f"negative control: the z-charge read as Y (no factor 3) — "
          f"{a[0]}/64 products", a[0] < 64)


# ---------------------------------------------------------------------------
# the sections and the geometric labels
# ---------------------------------------------------------------------------

def _window_sections(M=8):
    A, _B, _br = _alg()
    secs = set()
    for tile in range(8):
        for a in range(M + 1):
            for b in range(M + 1 - 2 * a):
                t, aa, bb, _p, _q = A.canonicalise((tile, a, b, 0, 0))
                secs.add((t, aa, bb))
    return sorted(secs)


def _balanced_multisets(M=8):
    from u1a1deven_geometric_frame import _maximal_cones, _charge
    out = set()
    for cone in _maximal_cones(4):
        cone = sorted(cone)
        for ms in product(range(M + 1), repeat=len(cone)):
            if sum(ms) <= M and sum(m * _charge(c, 1) for c, m in zip(cone, ms)) == 0:
                out.add(tuple(sorted((c, m) for c, m in zip(cone, ms) if m)))
    return out


def test_sections(M=8):
    A, B, _br = _alg()
    from u1a1deven_geometric_frame import _curves_cross, _charge
    secs = _window_sections(M)
    img = {sec: _deven_section(sec) for sec in secs}
    curves = {sec: c for sec, (c, _d) in img.items()}
    target = _balanced_multisets(M)
    check(f"the {len(secs)} sections with 2a + b <= {M} go to "
          f"{len(set(curves.values()))} distinct multisets, the "
          f"{len(target)} balanced non-crossing ones",
          len(set(curves.values())) == len(secs)
          and set(curves.values()) == target)
    letters = _deven_letter_images()

    def product_of_images(sec):
        """`(label, z-power)` of the product of the section's letters' images
        in `A1DevenKAlg(1)`, or `None` unless every step is one term with one
        𝖖-power and SU(2) weight 0."""
        tile, a, b = sec
        x, f = B.identity(), 0
        for L, m in zip(_TILE_LETTERS[tile], (a, b)):
            for _ in range(m):
                pr = B.multiply(x, (letters[L][0], 0, 0))
                if len(pr.terms) != 1:
                    return None
                ((F, e, kap), co), = pr.terms.items()
                if kap != 0 or len([v for v in co._coeffs.values() if v]) != 1:
                    return None
                x, f = (F, 0, 0), f - e
        return x, f

    bad = [sec for sec in secs
           if product_of_images(sec) != ((curves[sec], 0, 0), 0)]
    check(f"A1DevenKAlg(1)'s product of each section's letters' images is "
          f"the one label of its image, with no power of z (the offsets add): "
          f"{len(secs) - len(bad)}/{len(secs)}", not bad)
    weights = [(0, 0), (1, 0), (0, 1), (1, 1)]
    geo, rot_bad, shape_bad = {}, [], []
    for sec in secs:
        for w in weights:
            lab = sec + w
            g = A.geometric_label(lab)
            geo[lab] = g
            cs, wt = g
            if (wt != w or any(m < 1 for _c, m in cs)
                    or sum(m * _charge(c, 1) for c, m in cs) != 0
                    or any(_curves_cross(c1, c2, 4)
                           for (c1, _m1) in cs for (c2, _m2) in cs if c1 < c2)):
                shape_bad.append(lab)
            r = A.geometric_label(A.rho(lab))
            if r != (_rotate(cs, 1), (w[1], w[0])):
                rot_bad.append(lab)
    check(f"geometric_label on {len(geo)} labels (the window × 4 SU(3) "
          f"weights): balanced non-crossing curves with the weight, "
          f"injective, and ρ rotates the curves and conjugates the weight",
          not shape_bad and not rot_bad and len(set(geo.values())) == len(geo))
    # a non-canonical spelling of T_0 (tile 1, b = 0) names the same element
    check("geometric_label reads the canonical label: (1, 1, 0, 0, 0) and "
          "T_0 = (0, 1, 0, 0, 0) have one geometric label",
          A.geometric_label((1, 1, 0, 0, 0)) == A.geometric_label(A.T(0)))
    for lab in (A.T(0), A.D(0), (1, 2, 3, 1, 0), (3, 1, 1, 0, 2)):
        print(f"      geometric_label{lab} = {A.geometric_label(lab)}")


# ---------------------------------------------------------------------------
# the seeds
# ---------------------------------------------------------------------------

def test_seeds_against_witness(K=K_SEEDS):
    """The cost is reported in three parts: the served `Tr_T`, `Tr_D`
    (`_deven_seed_z` on `T_0`, `D_0`, `D_1`), the witness's forward pass
    (`SU3ElemTraces.ensure`) less its `Tr_1`, and that `Tr_1` — the
    Kac–Wakimoto vacuum, which the forward pass is seeded by."""
    img = _deven_letter_images()
    t0 = time.time()
    T.vacuum_character(K // 2 + 2)
    t1 = time.time()
    served = {("Tr_T",): T._deven_seed_z(*img[("T", 0)], K),
              ("Tr_D", 0): T._deven_seed_z(*img[("D", 0)], K),
              ("Tr_D", 1): T._deven_seed_z(*img[("D", 1)], K)}
    t2 = time.time()
    wit = T.SU3ElemTraces().ensure(K)
    t3 = time.time()
    same = all(_upto(z, K) == _upto(wit.series(k), K) for k, z in served.items())
    via = T._ClosedFormSeeds().ensure(K)
    same = same and all(_upto(via.series(k), K) == _upto(z, K)
                        for k, z in served.items())
    check(f"Tr_T and Tr_D (both parities) served = the witness SU3ElemTraces "
          f"through q^{K} (served T/D {t2 - t1:.2f} s; witness forward pass "
          f"{t3 - t2 - (t1 - t0):.2f} s beyond its Kac–Wakimoto Tr_1, "
          f"{t1 - t0:.2f} s)", same)


def test_seed_consistency(K=24):
    img = _deven_letter_images()
    z = {L: _upto(T._deven_seed_z(*img[L], K), K) for L in img}
    check(f"every T_i gives Tr_T and D_(i+2) gives D_i's trace, through q^{K}",
          all(z[("T", i)] == z[("T", 0)] for i in range(4))
          and z[("D", 2)] == z[("D", 0)] and z[("D", 3)] == z[("D", 1)])
    check("Tr(D_1) = ⋆Tr(D_0) (weights negated)",
          z[("D", 1)] == {q: T.zstar(w) for q, w in z[("D", 0)].items()})


def test_window_margin(K=K_SEEDS):
    from u1a1deven_seed_characters import seed_trace
    img = _deven_letter_images()
    top = {}
    for L in (("T", 0), ("D", 0), ("D", 1)):
        top[L] = max(abs(n) for n in range(-(K + 1), K + 2)
                     if seed_trace(1, img[L][0], n, K))
    check(f"through q^{K} the gauge charges contributing reach |n| = "
          f"{top} — inside the window |n| <= {K + 1}",
          all(v <= K // 2 + 2 for v in top.values()))


def test_seed_negative_controls(K=16):
    img = _deven_letter_images()
    wit = T.SU3ElemTraces().ensure(K)
    want = _upto(wit.series(("Tr_D", 0)), K)
    c, d = img[("D", 0)]
    check("negative control: D_0's offset moved by 6 differs from the witness",
          _upto(T._deven_seed_z(c, d + 6, K), K) != want)
    check("negative control: D_1's image and offset in place of D_0's differ "
          "from the witness",
          _upto(T._deven_seed_z(*img[("D", 1)], K), K) != want)
    try:
        T._deven_seed_z(c, d + 3, K)
        refused = False
    except ValueError:
        refused = True
    check("negative control: D_0's offset moved by 3 is refused (not an SU(3) "
          "weight)", refused)


# ---------------------------------------------------------------------------
# product traces: the seed depth is the exact one (2026-09-24)
# ---------------------------------------------------------------------------

def test_product_trace_depth(K=8):
    """`product_trace` (so `trace_word` and `inner_product`) asks its seeds for
    exactly the depth its monomials' reductions read.  Its former padded guess
    was one order short on `(T₀²D₀)·(T₀²D₀)` (monomial `(0, 4, 2)`) and its
    ρ-images, so from a fresh provider the top order came out wrong: the value
    from a fresh provider must equal the one from a provider already deepened
    far beyond; and `_single_label_trace_z` refuses a provider one order too
    shallow instead of truncating."""
    A, _B, _br = _alg()
    saved = T._PROVIDER
    try:
        bad = []
        for tile in (0, 2):
            f = [(tile, 2, 1, 0, 0)] * 2
            T._PROVIDER = T._ClosedFormSeeds()
            fresh = T.product_trace(A, f, K)
            T._PROVIDER = T._ClosedFormSeeds().ensure(K + 40)
            deep = T.product_trace(A, f, K)
            if any(str(fresh.coeffs.get(q)) != str(deep.coeffs.get(q))
                   for q in range(K + 1)):
                bad.append(tile)
        check(f"product_trace from a fresh provider = from a deepened one on "
              f"(T_i²D_i)·(T_i²D_i), i = 0, 1, through q^{K} (the former padded "
              f"depth got q^{K} wrong)", not bad)
        tab = (0, 4, 2)
        red = T.seed_z_fast(tab + (0, 0))
        need = K + T._seed_depth(*(d.keys() for d in red.values()))
        prov = T._ClosedFormSeeds()
        prov.ensure(need - 1)
        prov.K = need - 1                  # exactly one order short
        try:
            T._single_label_trace_z(prov, tab, K, red)
            refused = False
        except ValueError:
            refused = True
        check(f"_single_label_trace_z refuses seeds one order short (q^{need - 1} "
              f"for q^{need}) instead of truncating", refused)
    finally:
        T._PROVIDER = saved


# ---------------------------------------------------------------------------
# composite labels against A1DevenKAlg(1)'s own trace
# ---------------------------------------------------------------------------

def _transport_ungauged():
    """`A1DevenKAlg(1)` on the transport route: the ungauging of
    `U1A1DevenConeKAlgebra(1, seed_closed_forms=False)`, whose traces of seeds
    too go through the exact transport — independent of the closed forms,
    which serve `A1DevenKAlg(1)`'s own seeds since 2026-09-24 as they serve
    `SU3ADKAlg`'s."""
    if "Bt" not in _ST:
        from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
        from ungauge_kalgebra import UngaugedKAlgebra
        G = U1A1DevenConeKAlgebra(1, seed_closed_forms=False)
        _ST["Bt"] = UngaugedKAlgebra(G, ((), 1, 0), epow=lambda lbl: lbl[1],
                                     e_shift=lambda lbl, n: (lbl[0], lbl[1] + n, lbl[2]))
    return _ST["Bt"]


def test_composite_labels(K=K_COMPOSITE):
    A, _B, br = _alg()
    B = _transport_ungauged()
    secs = _window_sections(3)
    bad = []
    t0 = time.time()
    for sec in secs:
        curves, delta = _deven_section(sec)
        got = {}
        for q, r in A.trace(sec + (0, 0), K).coeffs.items():
            for (kap, Y), v in br.apply_RElement(r).terms.items():
                if v and q <= K:
                    got[(q, kap, Y)] = got.get((q, kap, Y), 0) + v
        want = {}
        for q, r in B.trace((curves, 0, 0), K).coeffs.items():
            for (kap, (f,)), v in r.terms.items():
                if v and q <= K:
                    key = (q, kap, 3 * f + delta)
                    want[key] = want.get(key, 0) + v
        if ({k: v for k, v in got.items() if v}
                != {k: v for k, v in want.items() if v}):
            bad.append(sec)
    check(f"SU3ADKAlg's trace, restricted, = A1DevenKAlg(1)'s trace of the "
          f"image (Y = 3f + Δ) on the transport route, on "
          f"{len(secs) - len(bad)}/{len(secs)} sections with 2a + b <= 3 "
          f"through q^{K} [{time.time() - t0:.1f} s]", not bad)


# ---------------------------------------------------------------------------
# the serving path, in a fresh process
# ---------------------------------------------------------------------------

_FRESH = r"""
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "implementations"))
preloaded = set(sys.modules)
WITNESS = {"_solve_TrT", "_solve_SD", "_smallest_DT", "_smallest_DS",
           "_eq_rest", "Dseed", "_seed_layer1_z",
           "vacuum_character", "theta", "antisym_to_char"}
ran = set()
def prof(frame, event, arg):
    if event != "call":
        return
    co = frame.f_code
    if co.co_filename.endswith("sl3_su3_traces.py") and (
            co.co_name in WITNESS
            or type(frame.f_locals.get("self")).__name__ == "SU3ElemTraces"):
        ran.add("sl3_su3_traces." + co.co_name)
    if co.co_filename.endswith(("bps_kalgebra.py", "bps_factor_spectrum.py")):
        ran.add(co.co_name)
CONTROL = os.environ.get("SU3AD_FRESH_CONTROL") == "1"
sys.setprofile(prof)
from su3_ad_kalg import SU3ADKAlg
A = SU3ADKAlg()
K = 40
if CONTROL:     # the positive control: the forward pass served, an even-D class imported
    import sl3_su3_traces, a1deven_kalg
    sl3_su3_traces._PROVIDER = sl3_su3_traces.SU3ElemTraces()
    K = 12
ts = [A.trace(x, K) for x in (A.identity(), A.T(0), A.D(0), A.D(1))]
zs = []
if not CONTROL:
    import finite_kalgebras as fk
    Z = fk.FINITE_KALGEBRAS["a1d4"]()
    zs = [Z.trace((), 40)] + [Z.trace(((i, 1),), 40) for i in range(8)]
sys.setprofile(None)
assert all(t.coeffs for t in ts + zs) and max(ts[0].coeffs) >= K - 2
EVEN_D = ("u1a1deven_cone_kalgebra", "a1deven_kalg", "u1a1deven_geometric_frame",
          "u1a1deven_trace_transport", "ungauge_kalgebra", "u1a1deven_via_dodd_rg",
          "bps_kalgebra", "bps_factor_spectrum")
def in_repo(m):
    f = getattr(sys.modules.get(m), "__file__", None) or ""
    return os.path.abspath(f).startswith(os.getcwd() + os.sep)
bad = sorted(m for m in set(sys.modules) - preloaded
             if in_repo(m) and (m.split(".")[-1] in EVEN_D or "bootstrap" in m))
print("RESULT", sorted(ran), bad, "u1a1deven_seed_characters" in sys.modules)
"""


def _fresh(control):
    env = dict(os.environ, PYTHONPATH=_ROOT,
               SU3AD_FRESH_CONTROL="1" if control else "0")
    r = subprocess.run([sys.executable, "-c", _FRESH], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=1200)
    return r, [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]


def test_fresh_process():
    r, line = _fresh(True)
    fired = (r.returncode == 0 and len(line) == 1
             and "sl3_su3_traces._solve_TrT" in line[0]
             and "sl3_su3_traces.vacuum_character" in line[0]
             and "'a1deven_kalg'" in line[0])
    check("control: the fresh-process scan flags the forward pass and the "
          "Kac–Wakimoto vacuum when they are served, and an even-D class "
          "module when one is imported", fired)
    if not fired:
        print(line, r.stderr[-2000:])
        return
    t0 = time.time()
    r, line = _fresh(False)
    check(f"fresh process: SU3ADKAlg Tr(1), T_0, D_0, D_1 and zoo a1d4's Tr(1) "
          f"and 8 seeds through q^40 with no SU3ElemTraces method, no "
          f"Kac–Wakimoto vacuum and no BPS function run, and no even-D class "
          f"module or bootstrap imported (the closed forms are) "
          f"[{time.time() - t0:.1f} s]",
          r.returncode == 0 and line == ["RESULT [] [] True"])
    if r.returncode or line != ["RESULT [] [] True"]:
        print(line, r.stderr[-2000:])


if __name__ == "__main__":
    if not (test_control_vacuum() and test_control_composite()):
        print("\npositive control failed — aborting")
        sys.exit(1)
    test_search()
    test_negative_controls()
    test_sections()
    test_seeds_against_witness()
    test_seed_consistency()
    test_window_margin()
    test_seed_negative_controls()
    test_product_trace_depth()
    test_composite_labels()
    test_fresh_process()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
