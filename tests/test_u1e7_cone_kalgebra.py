"""Tests for `U1E7ConeKAlgebra` — standalone QT-cone realisation of u(1)-gauged E7.

The tables (`u1e7_cone_tables.pkl`, `u1e7_rho_tables.pkl`) are built from the
flow `U1A1E7RGKAlgebra`, whose dressing chord is the central chord `(3, 0)`
(rebuilt 2026-09-23; the earlier tables came from the `(2, 2)` dressing, whose
flow is `[A₁,D₇]` with the Cartan `U(1)` of its `SU(2)` gauged, and the class
refuses them by sha256).  202 atoms (every chord atom at gauge charge `c1 = 0`;
`E^{±1}` in every cone) and 4160 cones.  The spine-free cone **multiply**
reproduces the flow on an atom-pair sample and on 120 random native label ×
atom products, and the cone closes (every product canonical of every ordered
atom pair factored when the tables were built); `to_cone_label` factors 300
random native labels (`|c0| ≤ 6`); ρ/ρ⁻¹ equal the flow on every tabulated
key.  Slow tier (`U1E7_RUN_SLOW=1`): a rebuild of both tables (about two hours)
compared by content with the shipped ones.

**Trace**: magnetic (`c0`) charge vanishes exactly; the gauge v-tower `E^n`
(incl. `Tr(1)`) comes from the lazy vacuum recipe (Nahm e7); a magnetically
neutral chord label (`c0 = 0`, non-empty chord) is served through the label
map `φ` onto the ungauged `[A₁,E₇]` algebra `FiniteE7KAlgebra` (`E ↦ μ`),
`Tr(L_ℓ) = [μ^{−r}]((𝖖²;𝖖²)_∞²·Tr_{[A₁,E₇]}(L_z; μ))` for `φ(L_ℓ) = μ^r L_z`.
`φ` is certified by products: all 8100 ordered generator products agree
under it, every zoo 6-cone (a sample here, all 2600 in the slow tier) gives a
single-term image product with the zoo's phase, ρ commutes with it, and four
perturbed maps fail.  The served traces equal the flow's on labels not used
to find `φ`, satisfy ρ²-twisted cyclicity (the pair that once failed
included), are orthonormal on atoms, and load no RG flow, BPS engine or
bootstrap.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "implementations"))

import u1e7_cone_kalgebra as _M
from u1e7_cone_kalgebra import U1E7ConeKAlgebra

_SLOW = bool(os.environ.get("U1E7_RUN_SLOW"))
_K = None
_T = None
_SKIPPED: set = set()        # tests that returned without checking (slow tier)


def _alg():
    global _K
    if _K is None:
        _K = U1E7ConeKAlgebra()
    return _K


def _flow():
    """A fresh corrected flow, the oracle (never on the class's serving path)."""
    global _T
    if _T is None:
        from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra
        _T = U1A1E7RGKAlgebra()
        assert (_T.DRESS_TYPE, _T.DRESS_INDEX) == (3, 0)
    return _T


def _norm(elt):
    return {l: tuple(sorted(c._coeffs.items()))
            for l, c in elt.terms.items() if not c.is_zero()}


def _qseries(rps):
    return {q: sum(int(x) for x in v.terms.values())
            for q, v in rps.coeffs.items() if any(v.terms.values())}


def _light_atoms(n):
    atoms = _alg().cone_data().mult_gens()
    return sorted(atoms, key=lambda a: (sum(e for (_x, _y, e) in a[0])
                                        + abs(a[1][0]), repr(a)))[:n]


def _random_labels(n, c0max, c1max, seed, maxlet=5):
    """Random native labels: greedy compatible A6 chord multisets of 1 to
    `maxlet` letters (letters of the atoms), multiplicities 1–2,
    `|c0| ≤ c0max`, `|c1| ≤ c1max`."""
    import random
    rng = random.Random(seed)
    a6 = _flow()._surv
    atoms = _alg().cone_data().mult_gens()
    letters = sorted({(k, i) for g in atoms for (k, i, e) in g[0]})
    memo = {}

    def comp(x, y):
        key = (min(x, y), max(x, y))
        if key not in memo:
            p = a6.multiply(((x[0], x[1], 1),), ((y[0], y[1], 1),))
            memo[key] = len([1 for c in p.terms.values() if not c.is_zero()]) == 1
        return memo[key]

    out = []
    for _ in range(n):
        s, target = [], rng.randint(1, maxlet)
        for c in rng.sample(letters, len(letters)):
            if len(s) < target and all(comp(c, d) for d in s):
                s.append(c)
        ch = tuple(sorted((k, i, rng.randint(1, 2)) for (k, i) in s))
        out.append((ch, (rng.randint(-c0max, c0max), rng.randint(-c1max, c1max))))
    return out, comp, letters, rng


def test_stale_tables_refused(tmp_path=None):
    """The sha256 guard: a table whose digest is listed as stale is refused
    with a pointer to the rebuild (positive control on a temp file registered
    as stale); the shipped tables are not listed, and load."""
    import hashlib
    import tempfile
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    fake = os.path.join(d, "u1e7_cone_tables.pkl")
    with open(fake, "wb") as f:
        f.write(b"not a table")
    digest = hashlib.sha256(b"not a table").hexdigest()
    saved = dict(_M._STALE_SHA256)
    try:
        _M._refuse_stale_tables(fake)               # not listed: accepted
        _M._STALE_SHA256[digest] = "u1e7_cone_tables.pkl"
        try:
            _M._refuse_stale_tables(fake)
        except RuntimeError as exc:
            assert "(3, 0)" in str(exc) and "u1e7_cone_kalgebra.py" in str(exc)
        else:
            raise AssertionError("a listed digest was not refused")
    finally:
        _M._STALE_SHA256.clear()
        _M._STALE_SHA256.update(saved)
    for path in (_M._frozen_path(), _M._rho_frozen_path()):
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() not in _M._STALE_SHA256, path
    assert _alg()._oracle is None                    # spine-free load


def test_construction():
    K = _alg()
    cd = K.cone_data()
    atoms = cd.mult_gens()
    assert len(atoms) == 202
    assert len(cd.cones()) == 4160
    assert K.identity() == ((), (0, 0))
    from collections import Counter
    assert dict(Counter(a[1][0] for a in atoms)) == {-3: 5, -2: 25, -1: 35, 0: 72, 1: 35, 2: 25, 3: 5}
    E, Ei = ((), (0, 1)), ((), (0, -1))
    assert all(E in c and Ei in c for c in cd.cones())
    assert all(a[1][1] == 0 for a in atoms if a not in (E, Ei))
    assert len(K._rhotab) == 201 and K._Hval == 9


def test_multiply_closes_vs_oracle():
    """POSITIVE CONTROL: the spine-free cone multiply reproduces the flow
    exactly on every pair of a 12-atom sample (the lightest atoms)."""
    K, T = _alg(), _flow()
    atoms = _light_atoms(12)
    n = 0
    for a in atoms:
        for b in atoms:
            assert _norm(K.multiply(a, b)) == _norm(T.multiply(a, b)), (a, b)
            n += 1
    assert n == 144


def test_random_products_vs_oracle():
    """120 random native label × atom products (labels of 1–3 letters,
    `|c0| ≤ 2`, `|c1| ≤ 3`) equal the flow's; NEGATIVE CONTROL: the flow with
    the earlier `(2, 2)` dressing disagrees on some of them."""
    from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra

    class _Dress22(U1A1E7RGKAlgebra):
        DRESS_TYPE, DRESS_INDEX = 2, 2

    K, T = _alg(), _flow()
    labs, _c, _l, rng = _random_labels(120, 2, 3, seed=7, maxlet=3)
    atoms = list(K.cone_data().mult_gens())
    pairs = [(l, rng.choice(atoms)) for l in labs]
    for l, b in pairs:
        assert _norm(K.multiply(l, b)) == _norm(T.multiply(l, b)), (l, b)
    B = _Dress22()
    assert any(_norm(K.multiply(l, b)) != _norm(B.multiply(l, b))
               for l, b in pairs[:20])


def test_to_cone_label_sweep():
    """300 random native labels (`|c0| ≤ 6`, `|c1| ≤ 4`) factor through the
    cones; NEGATIVE CONTROL: a label with two crossing chords is refused."""
    K = _alg()
    cd = K.cone_data()
    labs, comp, letters, _rng = _random_labels(300, 6, 4, seed=11)
    for l in labs:
        cd.to_cone_label(l)
    x, y = next((x, y) for x in letters for y in letters if x < y and not comp(x, y))
    bad = (((x[0], x[1], 1), (y[0], y[1], 1)), (0, 0))
    try:
        cd.to_cone_label(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"crossing label {bad} was factored")


def test_bar_involution():
    K = _alg()
    atoms = _light_atoms(16)
    for a in atoms:
        for b in atoms:
            assert K.verify_bar_involution(a, b), (a, b)


def test_rho_table_matches_flow():
    """ρ and ρ⁻¹ equal the flow's on every tabulated key at a random gauge
    charge `c1 ∈ [−2, 2]` (tests the table and the gauge reflection together);
    NEGATIVE CONTROL: the `(2, 2)`-dressed flow's ρ disagrees on some keys."""
    import random
    rng = random.Random(3)
    K, T = _alg(), _flow()
    keys = sorted(K._rhotab, key=repr)
    for (w, c0) in keys:
        c1 = rng.randint(-2, 2)
        assert K.rho((w, (c0, c1))) == T.rho((w, (c0, c1))), (w, c0, c1)
        assert K.rho_inverse((w, (c0, c1))) == T.rho_inverse((w, (c0, c1))), (w, c0, c1)
    from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra

    class _Dress22(U1A1E7RGKAlgebra):
        DRESS_TYPE, DRESS_INDEX = 2, 2

    B = _Dress22()
    assert any(K.rho((w, (c0, 0))) != B.rho((w, (c0, 0))) for (w, c0) in keys[:20])


def test_regenerated_tables_match_slow():
    """Slow tier (`U1E7_RUN_SLOW=1`, about two hours): rebuild both tables from
    the flow into a temporary directory and compare their content (not their
    bytes) with the shipped ones."""
    if not _SLOW:
        print("  SKIP: test_regenerated_tables_match_slow (set U1E7_RUN_SLOW=1)")
        _SKIPPED.add("test_regenerated_tables_match_slow")
        return
    import pickle
    import tempfile
    from u1e7_gauged_rg import U1E7GaugedRG
    d = tempfile.mkdtemp()
    saved = (_M._frozen_path, _M._rho_frozen_path)
    shipped = (_M._frozen_path(), _M._rho_frozen_path())
    try:
        _M._frozen_path = lambda: os.path.join(d, "u1e7_cone_tables.pkl")
        _M._rho_frozen_path = lambda: os.path.join(d, "u1e7_rho_tables.pkl")
        _M.U1E7ConeData(U1E7GaugedRG(), use_frozen=False).freeze()
        _M.U1E7ConeKAlgebra(use_frozen=True).freeze_rho()
        fresh = (_M._frozen_path(), _M._rho_frozen_path())
    finally:
        _M._frozen_path, _M._rho_frozen_path = saved
    old, new = (pickle.load(open(p, "rb")) for p in (shipped[0], fresh[0]))
    assert set(old["_atoms"]) == set(new["_atoms"])
    for k in ("_sig", "_mono", "_E", "_nb", "_qpow"):
        assert old[k] == new[k], k
    assert set(old["_cones"]) == set(new["_cones"])
    assert old["_torus_gens"] == new["_torus_gens"]

    def xp(d):
        return {k: tuple((tuple(sorted(lp._coeffs.items())), w) for lp, w in v)
                for k, v in d.items()}

    assert xp(old["_xprod"]) == xp(new["_xprod"])
    ro, rn = (pickle.load(open(p, "rb")) for p in (shipped[1], fresh[1]))
    assert ro == rn


def test_trace_magnetic_vanishes():
    """Non-zero magnetic charge c0 ⇒ Tr = 0 exactly (spine-free)."""
    K = _alg()
    mag = (((1, 0, 1),), (1, 0))
    tr = K._trace_residual(mag, 6)
    assert all(not any(c.terms.values()) for c in tr.coeffs.values())


def test_vacuum_and_wilson_trace():
    """`Tr(1)=1-q²+q⁶+…` and `Tr(E^n)` via the lazy vacuum recipe (spine-free);
    `Tr(E^n)=Tr(E^{-n})`."""
    K = _alg()
    assert _qseries(K.trace(((), (0, 0)), 8)) == {0: 1, 2: -1, 6: 1}
    assert _qseries(K.trace(((), (0, 1)), 8)) == {3: -1}
    assert _qseries(K.trace(((), (0, 2)), 8)) == {6: 1}
    assert _qseries(K.trace(((), (0, -1)), 8)) == _qseries(K.trace(((), (0, 1)), 8))


# ---- the neutral sector as FiniteE7KAlgebra: the label map and its traces --


def _e7():
    """`FiniteE7KAlgebra` and the preimages `{g: y_g}` of its generators
    (`φ(L_{y_g}) = L_g`), read from `_E7_GENERATOR_PREIMAGES`."""
    from finite_e7_kalg import FiniteE7KAlgebra, E7_MULT_GENS_LATTICE
    Y = {}
    for g, gamma in enumerate(E7_MULT_GENS_LATTICE):
        w, c1 = _M._E7_GENERATOR_PREIMAGES[tuple(gamma)]
        Y[g] = (w, (0, c1))
    return FiniteE7KAlgebra(), Y


def _mu_terms(c):
    """A zoo coefficient as `{(q, mu-power): int}`."""
    out = {}
    if hasattr(c, "coeffs"):
        for e, r in c.coeffs.items():
            for kb, v in r.terms.items():
                if int(v):
                    k = (e, kb[0] if kb else 0)
                    out[k] = out.get(k, 0) + int(v)
    else:
        for e, v in c._coeffs.items():
            if int(v):
                out[(e, 0)] = out.get((e, 0), 0) + int(v)
    return {k: v for k, v in out.items() if v}


def _phi_inv(elt, Y, sigma=1):
    """`φ⁻¹` of a `FiniteE7KAlgebra` element: the zoo label `((g, p), …)` goes
    to the native sum `Σ p·y_g` (chord multiset, gauge charge) and `μ^t` to
    `E^{σ t}`; as `{label: {q: int}}`."""
    out = {}
    for zl, c in elt.terms.items():
        chord, c1 = {}, 0
        for (g, p) in zl:
            w, (_c0, cg) = Y[g]
            for (a, i, e) in w:
                chord[(a, i)] = chord.get((a, i), 0) + e * p
            c1 += cg * p
        w = tuple(sorted((a, i, e) for (a, i), e in chord.items() if e))
        for (e, t), v in _mu_terms(c).items():
            d = out.setdefault((w, (0, c1 + sigma * t)), {})
            d[e] = d.get(e, 0) + v
    return {l: {e: v for e, v in d.items() if v} for l, d in out.items()
            if any(d.values())}


def _gauged(elt):
    return {l: {e: int(v) for e, v in c._coeffs.items() if int(v)}
            for l, c in elt.terms.items() if not c.is_zero()}


def _generator_products_agree(Z, Y, K, sigma=1, stop_at_first=False):
    bad = 0
    for g in range(90):
        for h in range(90):
            if _phi_inv(Z.multiply(((g, 1),), ((h, 1),)), Y, sigma) != \
                    _gauged(K.multiply(Y[g], Y[h])):
                bad += 1
                if stop_at_first:
                    return bad
    return bad


def test_e7_label_map_certified_by_products():
    """The label map `φ` of the neutral sector onto `FiniteE7KAlgebra` (`E ↦ μ`)
    is certified by products.  POSITIVE CONTROL first: the table is keyed by
    exactly the zoo's 90 generator lattice vectors, and `_e7_image` sends each
    preimage back to its generator.  Then: all 8100 ordered generator
    products agree under `φ`; ρ commutes with `φ` on every generator; random
    light products of cone monomials agree.  NEGATIVE CONTROLS, each of which
    must fail the generator products: two images swapped inside a
    ρ²-orbit, one μ-shift off by one, `E ↦ μ⁻¹`, and one ρ-orbit's images
    translated by ρ²."""
    import random
    from finite_e7_kalg import E7_MULT_GENS_LATTICE, E7_RHO_PERM, E7_CONES
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    K = _alg()
    Z, Y = _e7()
    assert set(_M._E7_GENERATOR_PREIMAGES) == {tuple(g) for g in E7_MULT_GENS_LATTICE}
    assert len(_M._E7_GENERATOR_PREIMAGES) == 90
    for g in range(90):
        assert K._e7_image(Y[g]) == (((g, 1),), 0), g
    assert _generator_products_agree(Z, Y, K) == 0
    one = LaurentPoly({0: 1})
    for g in range(90):
        img = _phi_inv(Z.rho_element(Element({((g, 1),): one})), Y)
        assert img == {K.rho(Y[g]): {0: 1}}, g
    rng = random.Random(5)
    six = [c for c in E7_CONES if len(c) == 6]

    def light():
        while True:
            cone = rng.choice(six)
            lab = tuple(sorted((g, rng.randint(1, 2))
                               for g in rng.sample(list(cone), rng.randint(1, 2))))
            if sum(p for _, p in lab) <= 3:
                return lab

    def native(zl):
        (w, c1), = [(l[0], l[1][1]) for l in _phi_inv(Element({zl: one}), Y)]
        return (w, (0, c1))

    for _ in range(60):
        z1, z2 = light(), light()
        assert _phi_inv(Z.multiply(z1, z2), Y) == \
            _gauged(K.multiply(native(z1), native(z2))), (z1, z2)
    # negative controls
    P = E7_RHO_PERM
    Ys = dict(Y)
    Ys[0], Ys[P[P[0]]] = Y[P[P[0]]], Y[0]
    assert _generator_products_agree(Z, Ys, K, stop_at_first=True)
    Yb = dict(Y)
    w, (z0, c1) = Y[5]
    Yb[5] = (w, (z0, c1 + 1))
    assert _generator_products_agree(Z, Yb, K, stop_at_first=True)
    assert _generator_products_agree(Z, Y, K, sigma=-1, stop_at_first=True)
    orbit, x = [0], P[0]
    while x != 0:
        orbit.append(x)
        x = P[x]
    Yt = dict(Y)
    for g in orbit:
        Yt[P[P[g]]] = Y[g]
    assert _generator_products_agree(Z, Yt, K, stop_at_first=True)


def test_e7_label_map_cones():
    """Every zoo 6-cone: the ordered product of its generators' images is ONE
    term, the image of the cone monomial with the zoo's phase (a 300-cone
    sample; all 2600 in the slow tier, about a minute)."""
    import random
    from finite_e7_kalg import E7_CONES
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    K = _alg()
    Z, Y = _e7()
    one = LaurentPoly({0: 1})
    six = [c for c in E7_CONES if len(c) == 6]
    if not _SLOW:
        six = random.Random(8).sample(six, 300)
    for cone in six:
        order = sorted(cone)
        acc = Element({K.identity(): one})
        zacc = None
        for g in order:
            acc = K.multiply_elements(acc, Element({Y[g]: one}))
            e = Element({((g, 1),): one})
            zacc = e if zacc is None else Z.multiply_elements(zacc, e)
        got = _gauged(acc)
        assert len(got) == 1 and got == _phi_inv(zacc, Y), cone


def _qs(rps):
    return {e: sum(int(c) for c in v.terms.values()) for e, v in rps.coeffs.items()
            if sum(int(c) for c in v.terms.values())}


def test_neutral_chord_traces_served():
    """A magnetically neutral label is traced through `FiniteE7KAlgebra`.
    POSITIVE CONTROL first: the E-tower, now the identity case of the route
    (the zoo's closed-form vacuum), equals its independent witness, the gauged
    Nahm e7 sum (`_vacuum_mu`), for `n ∈ [−3, 3]` through `𝖖¹⁰`.  Then, on
    labels NOT used to find `φ` (products of a c0 = ±1 atom pair, a two-atom
    neutral monomial, an E-shifted label), the traces equal the flow's through
    `𝖖⁶` (`𝖖⁸` in the slow tier); and the cover `_e7_image` inverts is unique
    on a sample."""
    import random
    K = _alg()
    Z, Y = _e7()
    P = K._vacuum_mu(10)
    for n in range(-3, 4):
        exp = {q: md[-n] for q, md in P.items() if md.get(-n, 0)}
        assert _qseries(K.trace(((), (0, n)), 10)) == exp, n
    held = [Y[7],                                   # a c0 = ±1 atom pair
            (((1, 3, 1), (2, 3, 1)), (0, -1)),      # a c0 = ±1 pair, E-shifted
            (((1, 8, 1), (3, 2, 1)), (0, 0)),       # two neutral atoms
            (((2, 3, 1), (3, 2, 1), (3, 7, 1)), (0, 1)),
            (((2, 5, 1),), (0, -1))]                # E-shifted, r = -1
    if _SLOW:
        held += [Y[46], (Y[86][0], (0, 2)), (((1, 3, 1), (2, 5, 1)), (0, 1)),
                 (((1, 2, 1), (1, 8, 1), (2, 2, 1)), (0, 0)),
                 (((1, 3, 2), (1, 8, 2), (3, 7, 1)), (0, -1))]
    Kq = 8 if _SLOW else 6
    T = _flow()
    nonzero = flipped_differs = 0
    for lab in held:
        served, flow = _qs(K.trace(lab, Kq)), _qs(T.trace(lab, Kq))
        assert served == flow, (lab, served, flow)
        nonzero += bool(served)
        # NEGATIVE CONTROL: the other reading of the gauge projection, [mu^{+r}]
        z, r = K._e7_image(lab)
        if r and _qs(K._gauged_zoo_trace(z, -r, Kq)) != flow:
            flipped_differs += 1
    assert nonzero >= 3, nonzero               # the comparison is not vacuous
    assert flipped_differs >= 1
    # uniqueness of the cover on random neutral labels
    rng = random.Random(3)
    labs, _c, _l, _r = _random_labels(40, 0, 3, seed=19, maxlet=4)
    gens, by_letter = K._e7_cover
    qc = Z.cone_data().q_commute
    for lab in labs:
        if lab[1][0] != 0 or not lab[0]:
            continue
        target = {}
        for (a, i, e) in lab[0]:
            target[(a, i)] = target.get((a, i), 0) + e
        covers = []

        def all_covers(rem, chosen):
            if not rem:
                covers.append(tuple(sorted(chosen)))
                return
            letter = min(rem)
            for g in by_letter.get(letter, ()):
                ct = gens[g][0]
                if any(rem.get(x, 0) < n for x, n in ct.items()):
                    continue
                if not all(qc(g, h) for h in chosen):
                    continue
                nxt = dict(rem)
                for x, n in ct.items():
                    nxt[x] -= n
                    if not nxt[x]:
                        del nxt[x]
                all_covers(nxt, chosen + [g])

        all_covers(target, [])
        assert len(set(covers)) == 1, (lab, set(covers))


def test_rho2_cyclicity_with_neutral_traces():
    """ρ²-twisted cyclicity `Tr(L_a·L_b) = Tr(ρ²(L_b)·L_a)` through `𝖖⁶` on
    the pair recorded as failing under the former bootstrap, and on random
    pairs of atoms with opposite magnetic charge (so that the products are
    neutral)."""
    import random
    K = _alg()
    a = (((1, 0, 1), (1, 4, 1)), (1, 4))
    b = (((1, 1, 1), (1, 6, 1)), (-1, -4))
    assert K.verify_rho_twisted_trace(a, b, 6)
    atoms = list(K.cone_data().mult_gens())
    rng = random.Random(11)
    for _ in range(12 if not _SLOW else 40):
        x = rng.choice(atoms)
        y = rng.choice([t for t in atoms if t[1][0] == -x[1][0]])
        x = (x[0], (x[1][0], rng.randint(-2, 2)))
        y = (y[0], (y[1][0], rng.randint(-2, 2)))
        assert K.verify_rho_twisted_trace(x, y, 6), (x, y)


def test_orthonormality_on_atoms():
    """`⟨a, b⟩ = δ_{ab} + O(𝖖)` (no negative powers) on neutral and magnetic
    atoms, in the label form: this class's coefficient ring is `Z` and `E` sits
    in the label, so the μ-shift split of the flavoured zoo does not arise."""
    import random
    K = _alg()
    atoms = list(K.cone_data().mult_gens())
    rng = random.Random(29)
    n = 6 if not _SLOW else 12
    sample = (rng.sample([a for a in atoms if a[1][0] == 0 and a[0]], n)
              + rng.sample([a for a in atoms if a[1][0] != 0], n))
    for a in sample:
        for b in sample:
            assert K.verify_orthonormality(a, b, 2), (a, b)


def test_neutral_trace_loads_no_engine():
    """Tracing a neutral chord label and an E-tower label at `K = 20` loads no
    RG flow, no BPS engine, no bootstrap module and no Nahm-sum engine (a
    fresh interpreter)."""
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys\n"
        "from u1e7_cone_kalgebra import U1E7ConeKAlgebra\n"
        "K = U1E7ConeKAlgebra()\n"
        "t = K.trace((((1, 3, 1),), (0, 2)), 20)\n"
        "assert t.coeffs, 'empty trace'\n"
        "assert K.trace(((), (0, 1)), 20).coeffs, 'empty E-tower trace'\n"
        "bad = sorted(m for m in sys.modules if m == 'bps_kalgebra'\n"
        "             or ('bootstrap' in m and not m.startswith('importlib'))\n"
        "             or m in ('e7_rgkalgebra', 'u1a1e7_rgkalgebra',\n"
        "                      'u1e7_gauged_rg', 'rgkalgebra', 'vacuum_nahm'))\n"
        "print('LOADED', bad)\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [here, os.path.join(here, "implementations"), env.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=here,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "LOADED []" in out.stdout, out.stdout


if __name__ == "__main__":
    # the guard and the counts first, then the positive controls against the
    # flow and the zoo; an assertion aborts the run
    for fn in (test_stale_tables_refused, test_construction,
               test_multiply_closes_vs_oracle, test_random_products_vs_oracle,
               test_to_cone_label_sweep, test_bar_involution,
               test_rho_table_matches_flow, test_trace_magnetic_vanishes,
               test_vacuum_and_wilson_trace,
               test_e7_label_map_certified_by_products, test_e7_label_map_cones,
               test_neutral_chord_traces_served,
               test_rho2_cyclicity_with_neutral_traces,
               test_orthonormality_on_atoms, test_neutral_trace_loads_no_engine,
               test_regenerated_tables_match_slow):
        fn()
        print("skipped" if fn.__name__ in _SKIPPED else "ok", fn.__name__)
    print("ALL U1E7ConeKAlgebra TESTS PASSED"
          + (f" ({len(_SKIPPED)} skipped: {', '.join(sorted(_SKIPPED))})"
             if _SKIPPED else ""))
