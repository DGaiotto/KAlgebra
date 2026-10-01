"""The zoo entries a1d6 / a1d8 served through `A1DevenKAlg(2)` / `A1DevenKAlg(3)`
(`a1deven_seeds`, the design record), with controls.

Positive controls first; a failing control aborts the run:
  * `A1DevenKAlg(2)`'s Tr(1) at 𝖖² is `1 + χ₂`: the U(1) current and the SU(2)
    triplet (the flavour adjoint the withdrawn su2u1 route lacked,
    the audit), and Tr(1) equals the independent flow
    `A1DevenRGKAlgebra(2)`'s through 𝖖⁸;
  * the same method at k = 1 reproduces zoo a1d4 as served by the independent
    `SU3ADKAlg` route (`a1d4_seeds`): its map is certified
    (64/64 products, 8/8 ρ) and Tr(1) and all 8 seeds agree through 𝖖⁸
    (𝖖¹² with --slow);
  * the a1d6 map reproduces all 39² = 1521 generator products and ρ on 39/39.
Then, for a1d6:
  * the map: the ring's slot order, U(1) normalisation s = +1 (the zoo's μ^m
    is z^{s·m}), the certified maps are one map composed with the 6 powers of
    ρ, s alternating in sign;
    negative controls, each
    caught by the products: the slots swapped (no solution at all, and the
    served offsets fail), s = 3 (a1d4's branching factor) or s = −1, and one
    offset moved by 1;
  * `to_ungauged` carries zoo products to `A1DevenKAlg` products (40 pairs);
  * the seeds do not depend on the member of the class used to serve them
    (through the map composed with ρ, s = −1, they are equal through 𝖖⁶);
  * the zoo's Tr(1) and every seed, through the zoo's own `trace`, equal
    `A1DevenKAlg(2)`'s through the map, through 𝖖¹² (𝖖²⁴ with --slow);
  * the witness: every `A1DevenKAlg(2)` generator trace — served since
    2026-09-24 by the seed closed forms (`u1a1deven_seed_characters`, wired
    into the gauged class) — equals the transport route (the gauged class
    with `seed_closed_forms=False`) through 𝖖⁶ (𝖖¹⁰ with --slow);
  * the zoo's whole trace — its Layer-1 reduction over the served seeds —
    equals `A1DevenKAlg(2)`'s own trace of the image on 20 composite labels
    through 𝖖⁴;
  * the depth limit is lifted: with the transport's word-length limit
    lowered to 3 letters (in a fresh process) a seed is still served, by its
    closed form; with the closed forms switched off the transport route still
    raises NotImplementedError naming the entry, the seed, the order, the
    class and the limit;
  * the routing: `generate` names the route; the object layer tags a1d6 / a1d8
    `trace-exact` and prefers the zoo class for it;
  * in a fresh process Tr(1) and every seed through 𝖖¹² (𝖖⁴⁰ with --slow), with
    neither `bps_kalgebra` nor `bps_factor_spectrum` nor any bootstrap
    imported — so no BPS function ran (a module never imported runs nothing;
    the per-call profiler spy checks the same, at low order, in
    tests/test_finite_zoo_traces.py).
a1d8 (k = 3): the map on all 120² = 14400 generator products and ρ on 120/120,
its normalisation and class (8 powers of ρ), Tr(1) at 𝖖² and against the flow through
𝖖⁶, the seeds through 𝖖⁸ (𝖖¹⁶ with --slow), the generator traces against the
transport route through 𝖖² (𝖖⁴ with --slow), and 16 composite labels through
𝖖² (with --slow).
The map as a `KAlgebraIso` (`S.kalgebra_iso`, the design record): from the zoo's
Z-form wrapper `FiniteSU2U1ZKAlgebra` onto `A1DevenKAlg(k)` — the five checks
of `verify_all` (unit, round trip, multiplicative, ρ-equivariant,
trace-equivariant through 𝖖¹²) on the identity, every generator, `χ₁` and
`μ^{±1}`, with every ordered pair of them (11² / 42² pairs each way at a1d4 /
a1d6; at a1d8 the pairs among every 4th generator and the three flavour
characters, 33² each way; all 123² with --slow, 242 s, 295 s with the round
trip),
`verify_maps_section_to_section_1drep` on the sections, and the round trip on
every label of those products; negative controls at a1d4 and a1d6: the U(1)
normalisation flipped (s → −s), one offset moved by 1, and two generators
transposed each fail the battery.

Run:  `python3 run_tests.py` [--slow]
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

import finite_kalgebras as fk
import a1deven_seeds as S
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

SLOW = "--slow" in sys.argv[1:]
PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}", flush=True)
    return ok


def _norm(series, K):
    """An `A1DevenKAlg` / `A1DevenRGKAlgebra` trace as `{q: {(b, f): c}}`."""
    out = {}
    for q, r in series.coeffs.items():
        if q > K:
            continue
        row = {}
        for key, v in r.terms.items():
            if not v:
                continue
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], tuple):
                kk = (key[0], key[1][0])
            elif isinstance(key, tuple) and len(key) == 2:
                kk = key
            else:
                kk = (key, 0)
            row[kk] = row.get(kk, 0) + int(v)
        row = {kk: v for kk, v in row.items() if v}
        if row:
            out[q] = row
    return out


def _zoo_flat(series, K):
    """A zoo trace (`RPowerSeries` over `SU2xU1ZPlusRing`) as `{q: {(k, m): c}}`."""
    return {q: {kk: int(v) for kk, v in r.terms.items() if v}
            for q, r in series.coeffs.items()
            if q <= K and any(v for v in r.terms.values())}


def _read_back(norm_series, gm, c=0):
    """`{q: {(b, f): c}}` of an `A1DevenKAlg` trace, times `z^c`, read in the
    zoo's ring through the map `gm` (an independent restatement of the
    served readback: `z^f ↦ μ^{f·s}`, `s = ±1`)."""
    su, uu = gm["slots"]
    out = {}
    for q, row in norm_series.items():
        new = {}
        for (b, f), v in row.items():
            key = [0, 0]
            key[su], key[uu] = b, (f + c) * gm["scale"]
            new[tuple(key)] = new.get(tuple(key), 0) + v
        new = {kk: v for kk, v in new.items() if v}
        if new:
            out[q] = new
    return out


# ---------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------

def test_control_triplet_and_flow(k=2, K=8):
    from a1deven_rgkalgebra import A1DevenRGKAlgebra
    A = S.ungauged_algebra({2: "a1d6", 3: "a1d8"}[k])
    ta = _norm(A.trace(A.identity(), K), K)
    ok1 = check(f"control: A1DevenKAlg({k}) Tr(1)[q^2] = 1 + χ₂ (the U(1) "
                f"current and the SU(2) triplet)",
                ta.get(2) == {(0, 0): 1, (2, 0): 1})
    Ref = A1DevenRGKAlgebra(k)
    tr = _norm(Ref.trace(Ref.identity(), K), K)
    ok2 = check(f"control: A1DevenKAlg({k}) Tr(1) == A1DevenRGKAlgebra({k}) "
                f"Tr(1) through q^{K}", ta == tr)
    return ok1 and ok2


def test_control_a1d4(K=None):
    """The method at k = 1 against the independent SU3ADKAlg route of a1d4."""
    K = K or (12 if SLOW else 8)
    import a1d4_seeds as D4
    gm = S.generator_map("a1d4")
    ok_p = S.product_agreement("a1d4", gm)
    ok_r = S.rho_agreement("a1d4", gm)
    sd = S.seeds("a1d4")
    same = (sd.vacuum_trace(K) == D4.vacuum_trace(K)
            and all(sd.seed_trace(i, K) == D4.seed_trace(i, K)
                    for i in range(8)))
    return check(f"control: at k = 1 the map is certified ({ok_p[0]}/{ok_p[1]} "
                 f"products, {ok_r[0]}/{ok_r[1]} ρ) and zoo a1d4 through "
                 f"A1DevenKAlg(1) equals zoo a1d4 through SU3ADKAlg "
                 f"(a1d4_seeds): Tr(1) and all 8 seeds through q^{K}",
                 ok_p == (64, 64) and ok_r == (8, 8) and same)


def test_control_products(sid="a1d6"):
    gm = S.generator_map(sid)
    n = len(gm["sigma"])
    ok_p = S.product_agreement(sid, gm)
    ok_r = S.rho_agreement(sid, gm)
    return check(f"control: the {sid} map reproduces {ok_p[0]}/{ok_p[1]} "
                 f"generator products and ρ on {ok_r[0]}/{ok_r[1]} generators",
                 ok_p == (n * n, n * n) and ok_r == (n, n))


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------

def test_map_and_negative_controls(sid="a1d6", n_class=6):
    gm = S.generator_map(sid)
    n = len(gm["sigma"])
    st = S._STATE[S.K_OF[sid]]
    cls = st["class"]
    check(f"{sid}: slot order (SU(2), U(1)) = the ring's, U(1) normalisation "
          f"s = +1, and "
          f"the certified maps are one map composed with the {len(cls)} "
          f"powers of ρ, s = "
          f"{[g['scale'] for g in cls]}",
          gm["slots"] == (0, 1) and gm["scale"] == 1 and len(cls) == n_class
          and [g["scale"] for g in cls]
          == [cls[0]["scale"] * (-1) ** j for j in range(n_class)])
    words = gm["_words"]
    none = S._solve_u1(sid, gm["sigma"], (1, 0), dict(words))
    swapped = S.product_agreement(sid, dict(gm, slots=(1, 0), _words=words))
    check(f"negative control: with the slots swapped the product and ρ "
          f"equations have no solution, and the served offsets give "
          f"{swapped[0]}/{swapped[1]} products", none is None
          and swapped[0] < n * n)
    for s in (3, -1):
        bad = S.product_agreement(sid, dict(gm, scale=s, _words=words))
        check(f"negative control: U(1) normalisation s = {s} gives "
              f"{bad[0]}/{bad[1]} "
              f"products", bad[0] < n * n)
    g0 = next(g for g in range(n) if gm["offsets"][g] == 0)
    off = dict(gm["offsets"])
    off[g0] += 1
    bad = S.product_agreement(sid, dict(gm, offsets=off, _words=words))
    check(f"negative control: offset of generator {g0} moved by 1 gives "
          f"{bad[0]}/{bad[1]} products", bad[0] < n * n)


def test_seeds_do_not_depend_on_class_member(sid="a1d6", K=6):
    gm = S.generator_map(sid)
    st = S._STATE[S.K_OF[sid]]
    other = next(g for g in st["class"] if g["scale"] == -1)
    A = S.ungauged_algebra(sid)
    sd = S.seeds(sid)
    n = len(gm["sigma"])
    vac = _norm(A.trace(A.identity(), K), K)
    same = _read_back(vac, other) == sd.vacuum_trace(K)
    bad = [i for i in range(n)
           if _read_back(_norm(A.trace(other["sigma"][i], K), K), other,
                         other["offsets"][i]) != sd.seed_trace(i, K)]
    check(f"{sid}: served through the map composed with ρ (s = −1), "
          f"Tr(1) and "
          f"the {n} seeds are the served ones through q^{K} "
          f"({n - len(bad)}/{n})", same and not bad)


def test_rho_element(sid="a1d6"):
    ok = S.rho_agreement(sid, S.generator_map(sid))
    check(f"{sid}: rho_element agrees on {ok[0]}/{ok[1]} generators",
          ok[0] == ok[1])


def test_to_ungauged(sid="a1d6", pairs=40):
    """`to_ungauged` carries a zoo product to `A1DevenKAlg`'s product of the
    images (each image `to_ungauged` of the generator itself), as Elements."""
    Z = fk.FINITE_KALGEBRAS[sid]()
    A = S.ungauged_algebra(sid)
    n = len(S.generator_map(sid)["sigma"])
    img = {g: S.to_ungauged(sid, Z.multiply(((g, 1),), Z.identity()))
           for g in range(n)}
    step = max(1, n * n // pairs)
    todo = [(a, b) for a in range(n) for b in range(n)][::step][:pairs]
    bad = [(a, b) for a, b in todo
           if S._deven_terms(S.to_ungauged(sid, Z.multiply(((a, 1),),
                                                           ((b, 1),))))
           != S._deven_terms(A.multiply_elements(img[a], img[b]))]
    check(f"{sid}: to_ungauged(zoo L_a·L_b) == A1DevenKAlg's product of "
          f"to_ungauged(L_a), to_ungauged(L_b) on {len(todo) - len(bad)}/"
          f"{len(todo)} generator pairs", not bad)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def test_zoo_seeds_equal_class(sid="a1d6", K=None):
    K = K or {"a1d6": 24 if SLOW else 12, "a1d8": 16 if SLOW else 8}[sid]
    gm = S.generator_map(sid)
    Z = fk.FINITE_KALGEBRAS[sid]()
    A = S.ungauged_algebra(sid)
    n = len(gm["sigma"])
    vac = _zoo_flat(Z.trace((), K), K) == _read_back(
        _norm(A.trace(A.identity(), K), K), gm)
    bad = [i for i in range(n)
           if _zoo_flat(Z.trace(((i, 1),), K), K)
           != _read_back(_norm(A.trace(gm["sigma"][i], K), K), gm,
                         gm["offsets"][i])]
    check(f"{sid}: the zoo's Tr(1) and its {n} seed traces equal "
          f"A1DevenKAlg({S.K_OF[sid]})'s through the map, through q^{K} "
          f"({n - len(bad)}/{n} seeds)", vac and not bad)


def _composite_labels(sid, count):
    Z = fk.FINITE_KALGEBRAS[sid]()
    cd = Z.cone_data()
    n = len(cd.mult_gens())
    pairs = [(a, b) for a in range(n) for b in range(a, n)
             if a == b or cd.q_commute(a, b)]
    step = max(1, len(pairs) // count)
    labels = []
    for a, b in pairs[::step][:count]:
        (lab,) = Z.multiply(((a, 1),), ((b, 1),)).terms
        labels.append(lab)
    return labels


def test_composite_labels(sid="a1d6", count=20, K=None):
    K = K or {"a1d6": 4, "a1d8": 2}[sid]
    gm = S.generator_map(sid)
    Z = fk.FINITE_KALGEBRAS[sid]()
    A = S.ungauged_algebra(sid)
    labels = _composite_labels(sid, count)
    bad = []
    for lab in labels:
        img = S.to_ungauged_label(sid, lab)
        want = _read_back(_norm(A.trace(img, K), K), gm)
        if _zoo_flat(Z.trace(lab, K), K) != want:
            bad.append(lab)
    check(f"{sid}: the zoo's Layer-1 trace equals A1DevenKAlg's own trace of "
          f"the image on {len(labels) - len(bad)}/{len(labels)} composite "
          f"labels through q^{K}", len(labels) >= min(count, 15) and not bad)


def test_closed_forms_equal_transport(sid="a1d6", K=None):
    """The witness: every generator trace of `A1DevenKAlg(k)` as served (the
    seed closed forms) equals the transport route (the gauged class with
    `seed_closed_forms=False`)."""
    K = K or {"a1d6": 10 if SLOW else 6, "a1d8": 4 if SLOW else 2}[sid]
    from a1deven_kalg import A1DevenKAlg
    A = S.ungauged_algebra(sid)
    W = A1DevenKAlg(A.k)
    W._G._seed_closed_forms = False
    gens = A.mult_generators()
    bad = [g for g in gens if _norm(A.trace(g, K), K) != _norm(W.trace(g, K), K)]
    check(f"{sid}: every A1DevenKAlg({A.k}) generator trace from the seed "
          f"closed forms equals the transport route through q^{K} "
          f"({len(gens) - len(bad)}/{len(gens)})", not bad)


_LIMIT = r"""
import sys, os
sys.path.insert(0, os.getcwd())
import finite_kalgebras as fk
import a1deven_seeds as S
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from u1a1deven_trace_transport import _shared_transport
_shared_transport(2).max_word_degree = 3
A = fk.FINITE_KALGEBRAS["a1d6"]()
A.trace(((0, 1),), 6)
print("RESULT1 served")
S.ungauged_algebra("a1d6")._G._seed_closed_forms = False
try:
    A.trace(((15, 1),), 6)
    print("RESULT2 served")
except NotImplementedError as e:
    print("RESULT2", str(e).replace("\n", " "))
"""


def test_depth_limit_lifted():
    # Seed 15 is its ρ²-orbit's representative (the orbit minimum), so either
    # Layer-1 route asks Layer 2 for seed 15 itself; a1d6 runs Layer 1 on
    # canonical labels since 2026-09-26, which folds a label to that
    # representative first (the orbit's seed 20 is served as seed 15).
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, "-c", _LIMIT], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    lines = r.stdout.splitlines()
    served = "RESULT1 served" in lines
    line = next((ln for ln in lines if ln.startswith("RESULT2")), "")
    need = ("a1d6:", "seed 15", "q^6", "A1DevenKAlg(2)",
            "max_word_degree = 3", "DevenTraceTransport(k=2)")
    check("with the transport's word-length limit lowered to 3 letters a seed "
          "is still served (its closed form); with the closed forms off the "
          "transport route raises NotImplementedError naming the entry, the "
          "seed, the order, the class and the limit",
          r.returncode == 0 and served and all(x in line for x in need))
    if r.returncode:
        print(r.stderr[-2000:])


def test_routes():
    import elem_traces as ET
    from finite_kalgebra_objects import kalgebra_object
    msgs = {}
    for sid in ("a1d6", "a1d8"):
        try:
            ET.generate(sid, 4)
            msgs[sid] = None
        except NotImplementedError as e:
            msgs[sid] = str(e)
    check("generate() for a1d6 / a1d8 raises naming A1DevenKAlg(k) and "
          "a1deven_seeds",
          all(m and f"A1DevenKAlg({k})" in m and "a1deven_seeds" in m
              for m, k in ((msgs["a1d6"], 2), (msgs["a1d8"], 3))))
    ok = True
    for sid in ("a1d6", "a1d8"):
        O = kalgebra_object(sid)
        ok &= ("trace-exact" in O.capabilities("cone-frozen")
               and O.preferred("trace-exact") is O.realization("cone-frozen"))
    check("the object layer tags a1d6 / a1d8 trace-exact and prefers the zoo "
          "class for it", ok)


_FRESH = r"""
import sys, os
sys.path.insert(0, os.getcwd())
K = int(sys.argv[1])
import finite_kalgebras as fk
A = fk.FINITE_KALGEBRAS["a1d6"]()
t1 = A.trace((), K)
ts = [A.trace(((i, 1),), K) for i in range(39)]
assert 0 in t1.coeffs and max(t1.coeffs) >= K - 1
assert all(t.coeffs for t in ts)
bad = [m for m in ("bps_kalgebra", "bps_factor_spectrum",
                   "finite_kalgebras.u1_bootstrap",
                   "finite_kalgebras.su2_bootstrap",
                   "finite_kalgebras.su2u1_bootstrap",
                   "finite_kalgebras.su2u1_trace_bootstrap") if m in sys.modules]
print("RESULT", bad)
"""


def test_fresh_process(K=None):
    K = K or (40 if SLOW else 12)
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, "-c", _FRESH, str(K)], cwd=_ROOT,
                       env=env, capture_output=True, text=True, timeout=7200)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    check(f"fresh process: a1d6 Tr(1) and every seed through q^{K} with "
          f"neither bps_kalgebra nor bps_factor_spectrum nor any bootstrap "
          f"imported (so no BPS function ran)",
          r.returncode == 0 and line == ["RESULT []"])
    if r.returncode:
        print(r.stderr[-2000:])


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

def _iso_samples(sid, iso):
    """The identity, every zoo generator, `χ₁` and `μ^{±1}` on the Z-form side;
    the identity, every `A1DevenKAlg` generator, `χ₁` and `z^{±1}` on the other
    (the three flavour characters last)."""
    n = len(S.generator_map(sid)["sigma"])
    src = ([iso.source.identity()] + [(0, (((g, 1),), 0)) for g in range(n)]
           + [(1, ((), 0)), (0, ((), 1)), (0, ((), -1))])
    tgt = ([iso.target.identity()] + list(iso.target.mult_generators())
           + [((), 0, 1), ((), -1, 0), ((), 1, 0)])
    return src, tgt


def _iso_battery(iso, src, tgt, K, step=1):
    """The five checks of `KAlgebraIso.verify_all`, both ways, traces through
    𝖖^K: unit, round trip, ρ and traces on every sample; multiplicativity on
    every ordered pair of the non-identity samples — with `step > 1` only of
    every `step`-th generator and the three flavour characters (the documented
    sample of a1d8's default run)."""
    one = LaurentPoly.one()
    se = [Element({l: one}) for l in src]
    te = [Element({l: one}) for l in tgt]
    ps = se[1:-3][::step] + se[-3:]
    pt = te[1:-3][::step] + te[-3:]
    return {
        "unit": iso.verify_unit(),
        "round_trip": iso.verify_round_trip(se, te),
        "multiplicative": iso.verify_multiplicative(
            [(a, b) for a in ps for b in ps], [(a, b) for a in pt for b in pt]),
        "rho_equivariant": iso.verify_rho_equivariant(se, te),
        "trace_equivariant": iso.verify_trace_equivariant(se, te, K=K),
    }, len(ps)


def _product_labels(alg, samples, step=1):
    labs = set(samples)
    ps = samples[1:-3][::step] + samples[-3:]
    for a in ps:
        for b in ps:
            labs.update(alg.multiply(a, b).terms)
    return labs


def _relabelled(iso, perm, perm_inv, name):
    """`iso` precomposed with a bijection `perm` of the source labels (inverse
    `perm_inv`): a bijection still, an algebra map only if `perm` is one."""
    one = LaurentPoly.one()
    return KAlgebraIso(
        iso.source, iso.target,
        lambda l: iso.map(Element({perm(l): one})),
        lambda l: Element({perm_inv(x): c for x, c in
                           iso.inverse(Element({l: one})).terms.items()}),
        name=name)


def test_kalgebra_iso(sid="a1d6", K=12, step=1):
    import time
    one = LaurentPoly.one()
    t0 = time.time()
    Z = fk.FINITE_KALGEBRAS[sid]()
    iso = S.kalgebra_iso(sid, native=Z)
    src, tgt = _iso_samples(sid, iso)
    res, n_pairs = _iso_battery(iso, src, tgt, K, step)
    n = len(src) - 4
    what = (f"every ordered pair of the {n} generators and the 3 flavour "
            f"characters ({n_pairs}² each way)" if step == 1 else
            f"the ordered pairs of every {step}th generator and the 3 flavour "
            f"characters ({n_pairs}² of the {n + 3}² each way)")
    check(f"{sid}: KAlgebraIso z-form → A1DevenKAlg({S.K_OF[sid]}) passes "
          f"{sorted(c for c, v in res.items() if v)}: unit, round trip, ρ and "
          f"traces through q^{K} on the identity, {n} generators, χ₁ and μ^±1; "
          f"multiplicative on {what} ({time.time() - t0:.1f} s)",
          all(res.values()) and iso.source._native is Z)
    check(f"{sid}: sections go to sections dressed by a one-dimensional rep, "
          f"both ways ({n + 1} + {n + 1})",
          iso.verify_maps_section_to_section_1drep(src[:-3], tgt[:-3]))
    ls = _product_labels(iso.source, src, step)
    lt = _product_labels(iso.target, tgt, step)
    bad = ([l for l in ls if iso.inverse(iso.map(Element({l: one})))
            != Element({l: one})]
           + [l for l in lt if iso.map(iso.inverse(Element({l: one})))
              != Element({l: one})])
    check(f"{sid}: the round trip holds on all {len(ls)} + {len(lt)} labels of "
          f"those products", not bad)


def test_kalgebra_iso_negative_controls(K=6):
    for sid in ("a1d4", "a1d6"):
        iso = S.kalgebra_iso(sid)
        src, tgt = _iso_samples(sid, iso)

        def flip(l):
            b, (word, m) = l
            return (b, (word, -m))

        def moved(sign):
            def perm(l):
                b, (word, m) = l
                return (b, (word, m + sign * dict(word).get(0, 0)))
            return perm

        def swap(l):
            b, (word, m) = l
            t = {0: 1, 1: 0}
            return (b, (tuple(sorted((t.get(g, g), p) for g, p in word)), m))

        for what, perm, perm_inv in (
                ("the U(1) normalisation flipped (s → −s)", flip, flip),
                ("the offset of generator 0 moved by 1", moved(1), moved(-1)),
                ("zoo generators 0 and 1 transposed", swap, swap)):
            res, _n = _iso_battery(_relabelled(iso, perm, perm_inv, what),
                                   src, tgt, K)
            check(f"{sid}: negative control — with {what} the battery fails "
                  f"{sorted(c for c, v in res.items() if not v)}",
                  not all(res.values()))


# ---------------------------------------------------------------------------
# a1d8
# ---------------------------------------------------------------------------

def test_a1d8():
    if not test_control_products("a1d8"):
        return
    gm = S.generator_map("a1d8")
    cls = S._STATE[3]["class"]
    check(f"a1d8: slot order the ring's, U(1) normalisation s = +1, one map "
          f"composed with the {len(cls)} powers of ρ, s alternating",
          gm["slots"] == (0, 1) and gm["scale"] == 1 and len(cls) == 8
          and [g["scale"] for g in cls]
          == [cls[0]["scale"] * (-1) ** j for j in range(8)])
    test_control_triplet_and_flow(k=3, K=6)
    test_zoo_seeds_equal_class("a1d8")
    test_closed_forms_equal_transport("a1d8")
    test_kalgebra_iso("a1d8", step=1 if SLOW else 4)
    if SLOW:
        test_composite_labels("a1d8", count=16)


if __name__ == "__main__":
    if not (test_control_triplet_and_flow() and test_control_a1d4()
            and test_control_products()):
        print("\npositive control failed — aborting")
        sys.exit(1)
    test_map_and_negative_controls()
    test_kalgebra_iso("a1d4")
    test_kalgebra_iso("a1d6")
    test_kalgebra_iso_negative_controls()
    test_rho_element()
    test_to_ungauged()
    test_seeds_do_not_depend_on_class_member()
    test_zoo_seeds_equal_class()
    test_closed_forms_equal_transport()
    test_composite_labels()
    test_depth_limit_lifted()
    test_routes()
    test_fresh_process()
    test_a1d8()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
