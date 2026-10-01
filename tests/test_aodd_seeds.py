"""The zoo's [A₁,A₂ₖ₊₁] entries a3 / a5 / a7 (aliases hexagon / octagon /
decagon) served through `ungauge_u1a1aodd(k)` (`aodd_seeds`,
the design record), with controls.

Positive controls first; a failing control aborts the run:
  * the map's three sign conventions — word ↦ E-power −Σp·f, zoo μ^f ↦ E^{+f},
    trace read back with z ↦ z⁻¹ — reproduce the a3 generator products (36/36)
    and the a3 closed-form characters (`ad_characters.a3_elem_entry`) through
    𝖖⁴⁸; each alternative sign fails (products 18/18/22 of 36; with z ↦ z,
    4 of the 6 seeds).
Then:
  * the generator map is a bijection of generators with vanishing magnetic
    charge (6, 24, 65);
  * generator products agree: 36/36, 576/576, 4225/4225;
  * `rho_element` agrees on every generator: 6/24/65;
  * the served seed traces equal the u(1) bootstrap `generate_u1` (a5 through
    𝖖¹⁰, a7 through 𝖖⁶; `--slow`: a5 through 𝖖²⁰, a7 through 𝖖¹⁴, where it
    finishes in about 100 s / 150 s);
  * the zoo's whole trace — its own Layer-1 reduction over the served seeds —
    equals the ungauged trace of the mapped label on composite labels;
  * the aliases serve the same series;
  * in a fresh process a5 and a7 serve Tr(1) and every seed through 𝖖²⁴
    (`--slow`: 𝖖⁴⁰) with neither `bps_kalgebra` nor the bootstraps imported;
  * negative control: `f + 1` on one generator is caught by the product check;
  * the map as a `KAlgebraIso` (`S.kalgebra_iso`, the design record): from the zoo's
    Z-form wrapper `FiniteU1ZKAlgebra`, base-changed along `μ ↦ z⁻¹`, onto
    `ungauge_u1a1aodd(k)` — `verify_all` (unit, round trip, multiplicative,
    ρ-equivariant, trace-equivariant through 𝖖¹²) on the identity, every
    generator and `μ^{±1}` and every ordered pair of them (8² / 26² / 67²
    pairs each way), `verify_maps_section_to_section_1drep` on the sections,
    and the round trip on every label of those products; negative controls:
    the flavour normalisation flipped (`μ ↦ z`) and two generators transposed
    each fail the battery.

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
import aodd_seeds as S
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent

PASS, FAIL = [], []
_IDS = {1: "a3", 2: "a5", 3: "a7"}


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _as_dict(x):
    return {lab: {q: v for q, v in c._coeffs.items() if v}
            for lab, c in x.terms.items() if not c.is_zero()}


def _mapped(sid, x, gm, s_word=-1, s_mu=1):
    """`S.to_ungauged` with the two label signs as parameters and the generator
    map `gm` as an argument (the controls vary them)."""
    out = {}
    for word, c in x.terms.items():
        letters, e = {}, 0
        for (mg, p) in word:
            F, f = gm[mg]
            for (t, i, m) in F:
                letters[(t, i)] = letters.get((t, i), 0) + p * m
            e += s_word * p * f
        F = tuple(sorted((t, i, m) for (t, i), m in letters.items()))
        if isinstance(c, RLaurent):
            items = [(q, key[0], v) for q, r in c.coeffs.items()
                     for key, v in r.terms.items() if v]
        else:
            items = [(q, 0, v) for q, v in c._coeffs.items() if v]
        for q, f, v in items:
            d = out.setdefault((F, e + s_mu * f), {})
            d[q] = d.get(q, 0) + v
    return {lab: {q: v for q, v in d.items() if v}
            for lab, d in out.items() if any(d.values())}


def _products(sid, gm, s_word=-1, s_mu=1):
    """(agreeing, total) over the ordered generator pairs."""
    Z = fk.FINITE_KALGEBRAS[sid]()
    U = S.ungauged_algebra(sid)
    ok = tot = 0
    for a in sorted(gm):
        for b in sorted(gm):
            tot += 1
            lhs = _mapped(sid, Z.multiply(((a, 1),), ((b, 1),)), gm, s_word, s_mu)
            la = next(iter(_mapped(sid, Element({((a, 1),): LaurentPoly.one()}),
                                   gm, s_word, s_mu)))
            lb = next(iter(_mapped(sid, Element({((b, 1),): LaurentPoly.one()}),
                                   gm, s_word, s_mu)))
            ok += lhs == _as_dict(U.multiply(la, lb))
    return ok, tot


def _readback(series, K, s_z=-1):
    out = {}
    for q, r in series.coeffs.items():
        row = {(s_z * z,): int(v) for (z,), v in r.terms.items() if v}
        if row and q <= K:
            out[q] = row
    return out


def _mu_series(d, K):
    """A record's series with zero rows dropped, keys normalised to `(μ,)`."""
    out = {}
    for q, row in d.items():
        r = {(k if isinstance(k, tuple) else (k,)): int(c)
             for k, c in row.items() if c}
        if r and q <= K:
            out[q] = r
    return out


# ---------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------

def test_control_signs_on_products():
    gm = S.generator_map("a3")
    counts = {sg: _products("a3", gm, *sg)
              for sg in ((-1, 1), (1, 1), (1, -1), (-1, -1))}
    ok = check("control: a3 generator products agree 36/36 under word ↦ −Σp·f, "
               "μ^f ↦ E^{+f}", counts[(-1, 1)] == (36, 36))
    check(f"control: each other sign choice fails on the a3 products "
          f"({[counts[sg][0] for sg in ((1, 1), (1, -1), (-1, -1))]} of 36)",
          all(counts[sg][0] < 36 for sg in ((1, 1), (1, -1), (-1, -1))))
    return ok


def test_control_a3_closed_form(K=48):
    from ad_characters import a3_elem_entry
    ref = a3_elem_entry(K)
    sd = S.seeds("a3")
    ok = check(f"control: a3 Tr(1) and all 6 seeds equal the closed-form "
               f"characters through q^{K}",
               sd.vacuum_trace(K) == ref["identity"]
               and all(sd.seed_trace(i, K) == ref["orbits"][i] for i in range(6)))
    U = S.ungauged_algebra("a3")
    wrong = sum(_readback(U.trace(S.to_ungauged_label("a3", ((i, 1),)), 12), 12,
                          s_z=+1) != _mu_series(a3_elem_entry(12)["orbits"][i], 12)
                for i in range(6))
    check(f"control: read back with z ↦ z instead, {wrong} of the 6 a3 seeds "
          f"differ at q^12", wrong == 4)
    return ok


# ---------------------------------------------------------------------------
# the map, products, rho
# ---------------------------------------------------------------------------

def test_map_is_a_bijection():
    for k, n in ((1, 6), (2, 24), (3, 65)):
        sid = _IDS[k]
        gm = S.generator_map(sid)
        U = S.ungauged_algebra(sid)
        cd = U._G.cone_data()
        mags = {sum(m * cd._letter_mag((t, i)) for (t, i, m) in F)
                for F, _f in gm.values()}
        check(f"{sid}: the generator map is a bijection onto the {n} ungauged "
              f"generators, all magnetically neutral",
              len(gm) == n and len({F for F, _f in gm.values()}) == n
              and {(F, 0) for F, _f in gm.values()} == set(U.mult_generators())
              and mags == {0})


def test_generator_products():
    for k, n in ((1, 36), (2, 576), (3, 4225)):
        sid = _IDS[k]
        ok, tot = _products(sid, S.generator_map(sid))
        check(f"{sid}: generator products agree {ok}/{tot}", (ok, tot) == (n, n))


def test_rho_element():
    one = LaurentPoly.one()
    for k in (1, 2, 3):
        sid = _IDS[k]
        Z = fk.FINITE_KALGEBRAS[sid]()
        U = S.ungauged_algebra(sid)
        gm = S.generator_map(sid)
        ok = sum(_as_dict(S.to_ungauged(sid, Z.rho_element(Element({((i, 1),): one}))))
                 == {U.rho(S.to_ungauged_label(sid, ((i, 1),))): {0: 1}}
                 for i in sorted(gm))
        check(f"{sid}: rho_element agrees on {ok}/{len(gm)} generators",
              ok == len(gm))


def test_negative_control_f_plus_one():
    """Shift one generator's `f` by one: the product check must see it."""
    for k in (1, 2, 3):
        sid = _IDS[k]
        gm = dict(S.generator_map(sid))
        F, f = gm[0]
        gm[0] = (F, f + 1)
        ok, tot = _products(sid, gm)
        check(f"{sid}: negative control — f + 1 on generator 0 breaks "
              f"{tot - ok} of {tot} products", ok < tot)


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

def _iso_samples(sid, iso):
    """The identity, every zoo generator and `μ^{±1}` on the Z-form side; the
    identity, every ungauged generator and `E^{±1}` on the ungauged side (the
    flavour characters last)."""
    n = len(S.generator_map(sid))
    src = ([iso.source.identity()]
           + [((), (((g, 1),), (0,))) for g in range(n)]
           + [((), ((), (m,))) for m in (1, -1)])
    tgt = ([iso.target.identity()] + list(iso.target.mult_generators())
           + [((), e) for e in (1, -1)])
    return src, tgt


def _iso_battery(iso, src, tgt, K):
    """`KAlgebraIso.verify_all` on the samples and every ordered pair of the
    non-identity ones, both ways, traces through 𝖖^K."""
    one = LaurentPoly.one()
    se = [Element({l: one}) for l in src]
    te = [Element({l: one}) for l in tgt]
    return iso.verify_all(se, te, [(a, b) for a in se[1:] for b in se[1:]],
                          [(a, b) for a in te[1:] for b in te[1:]], trace_K=K)


def _product_labels(alg, samples):
    labs = set(samples)
    for a in samples[1:]:
        for b in samples[1:]:
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


def test_kalgebra_iso(K=12):
    import time
    one = LaurentPoly.one()
    for k in (1, 2, 3):
        sid = _IDS[k]
        t0 = time.time()
        Z = fk.FINITE_KALGEBRAS[sid]()
        iso = S.kalgebra_iso(sid, native=Z)
        src, tgt = _iso_samples(sid, iso)
        res = _iso_battery(iso, src, tgt, K)
        n = len(src) - 1
        check(f"{sid}: KAlgebraIso z-form → ungauge_u1a1aodd({k}) passes "
              f"{sorted(c for c, v in res.items() if v)} on the identity, "
              f"{n - 2} generators and μ^±1, {n}² ordered pairs each way, "
              f"traces through q^{K} ({time.time() - t0:.1f} s)",
              all(res.values()) and len(res) == 5)
        check(f"{sid}: the source is FiniteU1ZKAlgebra over this very zoo "
              f"instance, base-changed onto the ungauged ring",
              iso.source._source_alg._native is Z
              and iso.source.coefficient_ring() == iso.target.coefficient_ring())
        check(f"{sid}: sections go to sections dressed by a one-dimensional "
              f"rep, both ways ({n - 1} + {n - 1})",
              iso.verify_maps_section_to_section_1drep(src[:-2], tgt[:-2]))
        ls = _product_labels(iso.source, src)
        lt = _product_labels(iso.target, tgt)
        bad = ([l for l in ls if iso.inverse(iso.map(Element({l: one})))
                != Element({l: one})]
               + [l for l in lt if iso.map(iso.inverse(Element({l: one})))
                  != Element({l: one})])
        check(f"{sid}: the round trip holds on all {len(ls)} + {len(lt)} labels "
              f"of those products", not bad)


def test_kalgebra_iso_negative_controls(K=6):
    for k in (1, 2):
        sid = _IDS[k]
        iso = S.kalgebra_iso(sid)
        src, tgt = _iso_samples(sid, iso)

        def flip(l):
            w, (word, (m,)) = l
            return (w, (word, (-m,)))

        res = _iso_battery(_relabelled(iso, flip, flip, "μ ↦ z"), src, tgt, K)
        check(f"{sid}: negative control — with the flavour normalisation "
              f"flipped (μ ↦ z) the battery fails "
              f"{sorted(c for c, v in res.items() if not v)}",
              not all(res.values()))

        def swap(l):
            w, (word, m) = l
            t = {0: 1, 1: 0}
            return (w, (tuple(sorted((t.get(g, g), p) for g, p in word)), m))

        res = _iso_battery(_relabelled(iso, swap, swap, "0 ↔ 1"), src, tgt, K)
        check(f"{sid}: negative control — with zoo generators 0 and 1 "
              f"transposed the battery fails "
              f"{sorted(c for c, v in res.items() if not v)}",
              not all(res.values()))


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def test_against_u1_bootstrap(Ks=((2, 10), (3, 6))):
    from u1_bootstrap import generate_u1
    for k, K in Ks:
        sid = _IDS[k]
        rec = generate_u1(sid, K)
        sd = S.seeds(sid)
        n = len(rec["orbits"])
        ok = sum(sd.seed_trace(i, K) == _mu_series(rec["orbits"][i], K)
                 for i in rec["orbits"])
        check(f"{sid}: Tr(1) and {ok}/{n} seeds equal generate_u1 through q^{K}",
              sd.vacuum_trace(K) == _mu_series(rec["identity"], K) and ok == n)


def test_composite_labels(K=12):
    """The zoo's own trace — Layer 1 over the zoo cones, then the served seeds —
    against the ungauged trace of the mapped label: squares of generators and
    products of two q-commuting generators."""
    for k in (1, 2, 3):
        sid = _IDS[k]
        Z = fk.FINITE_KALGEBRAS[sid]()
        U = S.ungauged_algebra(sid)
        cd = Z.cone_data()
        n = len(cd.mult_gens())
        pairs = [(a, a) for a in range(0, n, max(1, n // 6))]
        qc = [(a, b) for a in range(n) for b in range(a + 1, n)
              if cd.q_commute(a, b)]
        pairs += qc[::max(1, len(qc) // 6)]
        # the canonical label of each product (a single term: the letters
        # q-commute), so a non-simplicial cone's word is never passed raw
        labels = []
        for a, b in pairs:
            (lab,) = Z.multiply(((a, 1),), ((b, 1),)).terms
            labels.append(lab)
        bad = []
        for lab in labels:
            zt = {q: {key: int(v) for key, v in r.terms.items() if v}
                  for q, r in Z.trace(lab, K).coeffs.items() if q <= K}
            zt = {q: r for q, r in zt.items() if r}
            if zt != _readback(U.trace(S.to_ungauged_label(sid, lab), K), K):
                bad.append(lab)
        check(f"{sid}: zoo Tr == ungauged Tr on {len(labels) - len(bad)}/"
              f"{len(labels)} composite labels through q^{K}", not bad)


def test_aliases(K=10):
    from elem_traces import trace_residual
    for alias, sid in (("hexagon", "a3"), ("octagon", "a5"), ("decagon", "a7")):
        A = fk.FINITE_KALGEBRAS[alias]()
        ok = all(trace_residual(alias, A, lab, K).coeffs
                 == trace_residual(sid, A, lab, K).coeffs
                 for lab in ((), ((0, 1),), ((3, 1),)))
        check(f"{alias} is served as {sid}", ok)


_FRESH = r"""
import sys, os
sys.path.insert(0, os.getcwd())
import finite_kalgebras as fk
ran = set()
def prof(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.endswith(
            ("bps_kalgebra.py", "bps_factor_spectrum.py")):
        ran.add(frame.f_code.co_name)
sys.setprofile(prof)
for sid in ("a5", "a7"):
    A = fk.FINITE_KALGEBRAS[sid]()
    n = len(A.cone_data().mult_gens())
    t1 = A.trace((), K)
    ts = [A.trace(((i, 1),), K) for i in range(n)]
    assert 0 in t1.coeffs and max(t1.coeffs) > K // 2, sid
    assert sum(1 for t in ts if t.coeffs) >= n // 2, sid
sys.setprofile(None)
bad = [m for m in ("bps_kalgebra", "finite_kalgebras.u1_bootstrap",
                   "finite_kalgebras.su2_bootstrap") if m in sys.modules]
print("RESULT", sorted(ran), bad)
"""


def test_fresh_process(K=24):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, "-c", f"K = {K}\n" + _FRESH],
                       cwd=_ROOT, env=env, capture_output=True, text=True,
                       timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    check(f"fresh process: a5 and a7 serve Tr(1) and every seed through q^{K} "
          f"with no BPS function run and neither bps_kalgebra nor a bootstrap "
          f"imported", r.returncode == 0 and line == ["RESULT [] []"])
    if r.returncode:
        print(r.stderr[-2000:])


if __name__ == "__main__":
    slow = "--slow" in sys.argv
    if not (test_control_signs_on_products() and test_control_a3_closed_form()):
        print("\npositive control failed — aborting")
        sys.exit(1)
    test_map_is_a_bijection()
    test_generator_products()
    test_rho_element()
    test_negative_control_f_plus_one()
    test_kalgebra_iso()
    test_kalgebra_iso_negative_controls()
    test_composite_labels()
    test_aliases()
    if slow:
        test_against_u1_bootstrap(Ks=((2, 20), (3, 14)))
        test_fresh_process(K=40)
    else:
        test_against_u1_bootstrap()
        test_fresh_process()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
