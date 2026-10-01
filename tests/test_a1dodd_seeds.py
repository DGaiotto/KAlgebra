"""The zoo entries a1d3 / a1d5 / a1d7 identified with `A1DoddConeKAlg(k)`
(`a1dodd_seeds`, the design record), with controls.

Positive control first; a failing control aborts the run:
  * at a1d3 the map equals `a1d3_seeds`' (an exhaustive search over all 18
    ρ-equivariant bijections, which serves that entry).
Then, at k = 0, 1, 2:
  * the map reproduces every generator product term for term (𝖖-power and
    SU(2) weight included: 36 / 400 / 1,764) and ρ on every generator, and the
    certified maps are exactly one map composed with the 3 / 5 / 7 powers of
    ρ; negative controls at a1d5 and a1d7: two orbits swapped, and one orbit
    shifted alone, each fail products;
  * zoo products of generator pairs and of two-letter cone monomials, carried
    over by `to_a1dodd`, equal `A1DoddConeKAlg(k)`'s products of the images;
  * witness: the zoo's traces (its `a1d5_layer2` / `a1d7_layer2` closed forms)
    equal `A1DoddConeKAlg(k)`'s traces of the images (the general-k
    `a1dodd_layer2`) on `Tr(1)` and every generator; negative control: with
    two orbits of different traces swapped, they do not;
  * in a fresh process the a1d7 map is found with no BPS function run and
    neither `bps_kalgebra` nor a bootstrap imported;
  * the map as a `KAlgebraIso` (`S.kalgebra_iso`, the design record): from the zoo's
    Z-form wrapper `FiniteSU2ZKAlgebra` onto `A1DoddConeKAlg(k)` —
    `verify_all` (unit, round trip, multiplicative, ρ-equivariant,
    trace-equivariant through 𝖖¹²) on the identity, every generator and `χ₁`
    and every ordered pair of them (7² / 21² / 43² pairs each way),
    `verify_maps_section_to_section_1drep` on the sections, and the round trip
    on every label of those products; at a1d3 the same battery with the
    existing wrapper `FiniteA1D3ZKAlgebra()` as the source; negative controls
    at a1d5 and a1d7: two ρ-orbits swapped, and one orbit shifted alone, each
    fail the battery.

Run:  `python3 run_tests.py`
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

import finite_kalgebras as fk
import a1dodd_seeds as S
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

PASS, FAIL = [], []
N_OF = {"a1d3": 6, "a1d5": 20, "a1d7": 42}


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _flat(rps, K):
    out = {}
    for q, r in rps.coeffs.items():
        row = {int(n): int(v) for n, v in r.terms.items() if v}
        if row and q <= K:
            out[q] = row
    return out


def _orbits(short_id):
    return S._orbits(range(N_OF[short_id]), lambda g: S._zoo_rho(short_id, g))


def test_control_a1d3():
    import a1d3_seeds
    return check("a1d3: the map equals a1d3_seeds' (found exhaustively)",
                 S.generator_map("a1d3") == a1d3_seeds.generator_map())


def test_maps():
    for sid, n in N_OF.items():
        gm = S.generator_map(sid)
        cands = S._candidates(sid)
        certified = [s for s in cands if S.product_agreement(sid, s) == (n * n, n * n)
                     and S.rho_agreement(sid, s) == (n, n)]
        k = S.K_OF[sid]
        check(f"{sid}: all {n * n} generator products and ρ on all {n} generators; "
              f"{len(certified)} certified maps = the {2 * k + 3} powers of ρ",
              S.product_agreement(sid, gm) == (n * n, n * n)
              and S.rho_agreement(sid, gm) == (n, n)
              and len(certified) == len(cands) == 2 * k + 3)


def test_negative_controls():
    for sid in ("a1d5", "a1d7"):
        n = N_OF[sid]
        gm = S.generator_map(sid)
        zo = _orbits(sid)
        a, b = zo[0], zo[1]
        swapped = dict(gm)
        for g, h in zip(a, b):
            swapped[g], swapped[h] = gm[h], gm[g]
        shifted = dict(gm)
        for j, g in enumerate(a):
            shifted[g] = gm[a[(j + 1) % len(a)]]
        ok_sw, _ = S.product_agreement(sid, swapped)
        ok_sh, _ = S.product_agreement(sid, shifted)
        check(f"{sid}: negative controls — two orbits swapped {ok_sw}/{n * n}, one "
              f"orbit shifted alone {ok_sh}/{n * n}", ok_sw < n * n and ok_sh < n * n)


def test_composites():
    for sid in ("a1d5", "a1d7"):
        Z = fk.FINITE_KALGEBRAS[sid]()
        D = S.cone_algebra(sid)
        cd = Z.cone_data()
        n = N_OF[sid]
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if cd.q_commute(a, b)]
        monomials = []
        for a, b in pairs[:12]:
            (lab,) = Z.multiply(((a, 1),), ((b, 1),)).terms
            monomials.append(lab)
        labels = [((g, 1),) for g in range(0, n, 3)] + monomials
        bad = 0
        for x in labels:
            for y in labels[:8]:
                lhs = S.to_a1dodd(sid, Z.multiply(x, y))
                rhs = D.multiply(S.to_a1dodd_label(sid, x), S.to_a1dodd_label(sid, y))
                bad += dict(lhs.terms) != dict(rhs.terms)
        check(f"{sid}: {len(labels) * 8 - bad}/{len(labels) * 8} products of generators "
              f"and two-letter cone monomials carried over by to_a1dodd", bad == 0)


def test_traces_witness():
    for sid, K in (("a1d5", 16), ("a1d7", 12)):
        Z = fk.FINITE_KALGEBRAS[sid]()
        D = S.cone_algebra(sid)
        n = N_OF[sid]
        gm = S.generator_map(sid)

        def agree(sigma):
            ok = sum(_flat(Z.trace(((g, 1),), K), K)
                     == _flat(D.trace(((((sigma[g], 1),)), 0), K), K) for g in range(n))
            return ok

        vac = _flat(Z.trace((), K), K) == _flat(D.trace(D.identity(), K), K)
        good = agree(gm)
        zo = _orbits(sid)
        tr = {i: _flat(Z.trace(((zo[i][0], 1),), K), K) for i in range(len(zo))}
        i, j = next((i, j) for i in tr for j in tr if i < j and tr[i] != tr[j])
        swapped = dict(gm)
        for g, h in zip(zo[i], zo[j]):
            swapped[g], swapped[h] = gm[h], gm[g]
        bad = agree(swapped)
        check(f"{sid}: zoo Tr == A1DoddConeKAlg({S.K_OF[sid]}) Tr of the image on Tr(1) "
              f"and {good}/{n} generators through q^{K}; two orbits swapped: {bad}/{n}",
              vac and good == n and bad < n)


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

def _iso_samples(sid, iso):
    """The identity, every zoo generator and `χ₁` on the Z-form side; the
    identity, every `A1DoddConeKAlg` generator and `χ₁` on the other (the
    flavour character last)."""
    D = iso.target
    src = ([iso.source.identity()] + [(0, ((g, 1),)) for g in range(N_OF[sid])]
           + [(1, ())])
    tgt = ([D.identity()] + [(((g, 1),), 0)
                             for g in sorted(D.cone_data().mult_gens())]
           + [((), 1)])
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


def _relabelled(iso, perm, perm_inv, name, source=None):
    """`iso` precomposed with a bijection `perm` of the source labels (inverse
    `perm_inv`), optionally on another source algebra with the same labels."""
    one = LaurentPoly.one()
    return KAlgebraIso(
        source if source is not None else iso.source, iso.target,
        lambda l: iso.map(Element({perm(l): one})),
        lambda l: Element({perm_inv(x): c for x, c in
                           iso.inverse(Element({l: one})).terms.items()}),
        name=name)


def _on_letters(t):
    """The source-label bijection induced by the generator bijection `t`."""
    def perm(l):
        w, word = l
        return (w, tuple(sorted((t.get(g, g), p) for g, p in word)))
    return perm


def test_kalgebra_iso(K=12):
    import time
    one = LaurentPoly.one()
    for sid, n in N_OF.items():
        t0 = time.time()
        Z = fk.FINITE_KALGEBRAS[sid]()
        iso = S.kalgebra_iso(sid, native=Z)
        src, tgt = _iso_samples(sid, iso)
        res = _iso_battery(iso, src, tgt, K)
        check(f"{sid}: KAlgebraIso z-form → A1DoddConeKAlg({S.K_OF[sid]}) passes "
              f"{sorted(c for c, v in res.items() if v)} on the identity, {n} "
              f"generators and χ₁, {n + 1}² ordered pairs each way, traces "
              f"through q^{K} ({time.time() - t0:.1f} s)",
              all(res.values()) and len(res) == 5 and iso.source._native is Z)
        check(f"{sid}: sections go to sections dressed by a one-dimensional "
              f"rep, both ways ({n + 1} + {n + 1})",
              iso.verify_maps_section_to_section_1drep(src[:-1], tgt[:-1]))
        ls = _product_labels(iso.source, src)
        lt = _product_labels(iso.target, tgt)
        bad = ([l for l in ls if iso.inverse(iso.map(Element({l: one})))
                != Element({l: one})]
               + [l for l in lt if iso.map(iso.inverse(Element({l: one})))
                  != Element({l: one})])
        check(f"{sid}: the round trip holds on all {len(ls)} + {len(lt)} labels "
              f"of those products", not bad)
        if sid == "a1d3":
            from finite_a1d3_zform import FiniteA1D3ZKAlgebra
            old = _relabelled(iso, lambda l: l, lambda l: l,
                              "a1d3[FiniteA1D3ZKAlgebra→a1dodd]",
                              source=FiniteA1D3ZKAlgebra())
            res = _iso_battery(old, src, tgt, K)
            check(f"a1d3: the same maps from the existing wrapper "
                  f"FiniteA1D3ZKAlgebra() pass "
                  f"{sorted(c for c, v in res.items() if v)}",
                  all(res.values()) and len(res) == 5)


def test_kalgebra_iso_negative_controls(K=6):
    for sid in ("a1d5", "a1d7"):
        iso = S.kalgebra_iso(sid)
        src, tgt = _iso_samples(sid, iso)
        zo = _orbits(sid)
        a, b = zo[0], zo[1]
        swap = {}
        for g, h in zip(a, b):
            swap[g], swap[h] = h, g
        shift = {g: a[(j + 1) % len(a)] for j, g in enumerate(a)}
        back = {h: g for g, h in shift.items()}
        for what, t, t_inv in (("two ρ-orbits swapped", swap, swap),
                               ("one ρ-orbit shifted alone", shift, back)):
            res = _iso_battery(_relabelled(iso, _on_letters(t), _on_letters(t_inv),
                                           what), src, tgt, K)
            check(f"{sid}: negative control — with {what} the battery fails "
                  f"{sorted(c for c, v in res.items() if not v)}",
                  not all(res.values()))


_FRESH = r"""
import sys, os
sys.path.insert(0, os.getcwd())
ran = set()
def prof(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.endswith(
            ("bps_kalgebra.py", "bps_factor_spectrum.py")):
        ran.add(frame.f_code.co_name)
sys.setprofile(prof)
import a1dodd_seeds as S
gm = S.generator_map("a1d7")
sys.setprofile(None)
assert len(gm) == 42
bad = [m for m in ("bps_kalgebra", "finite_kalgebras.u1_bootstrap",
                   "finite_kalgebras.su2_bootstrap") if m in sys.modules]
print("RESULT", sorted(ran), bad)
"""


def test_fresh_process():
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, "-c", _FRESH], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    check("fresh process: the a1d7 map with no BPS function run and neither "
          "bps_kalgebra nor a bootstrap imported",
          r.returncode == 0 and line == ["RESULT [] []"])
    if r.returncode:
        print(r.stderr[-2000:])


if __name__ == "__main__":
    if not test_control_a1d3():
        print("\npositive control failed — aborting")
        sys.exit(1)
    test_maps()
    test_negative_controls()
    test_composites()
    test_traces_witness()
    test_kalgebra_iso()
    test_kalgebra_iso_negative_controls()
    test_fresh_process()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
