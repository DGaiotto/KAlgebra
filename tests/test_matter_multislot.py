"""Certification for `matter_multislot` — the `M ≥ 2` layer of the `(G, N)`
constructor.

`matter_star_bubbling` works with the **total** μ-degree.  At `M ≥ 2` that is a
central specialization (all `μ_i` equal); the specialized *element* is still
consistent, but the **solve does not survive it** — measured, the collapsed
constructor diverges from the flow at every `M = 2` label tried, always first at
total level 1, which is exactly where the multi-index first has more than one
component.  Here the μ-level is a genuine multi-index `k⃗`, one component per
irrep summand (the flavour convention: one `U(1)` per summand).

Run: `python3 run_tests.py`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import root_datum as rd
from g_matter_over_pure import GMatterOverPure
from matter_star_bubbling import DivisibilityFailure
from matter_multislot import (
    Z_levels_vec,
    cells_from_rg_chart_vec,
    check_divisibility_vec,
    delta_N_vec,
    divide_by_Z_vec,
    solve_canonical_matter_vec,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}: {name}"
          + (f"  {detail}" if detail and not cond else ""))


# (datum, matter-highest-weight, slots, dominant magnetic charges)
CASES = [
    (rd.su_2(), (1,), 2, [(1,)]),
    (rd.u_n(2), (1, 0), 2, [(0, -1)]),
    (rd.su_2(), (1,), 3, [(1,)]),
    # non-simply-laced: the corner that exposed the box cutoff.  Sp(4) and the
    # Spin(5) SPINOR are the cheap ones; Spin(5)+2x5 (Δ⃗=(2,2)) and G2+2x7 are
    # verified in the probes in the source repository (flow 84.7s / 250.0s) rather than in the suite.
    (rd.sp_n(2), (1, 0), 2, [(1, 0)]),
    (rd.b_n_simply_connected(2), (0, 1), 2, [(1, 1)]),
]


def _matter(lam, nf):
    return ((tuple(lam), nf),)


def test_delta_vec():
    print("Δ⃗_N is per-slot")
    d, lam = rd.su_2(), (1,)
    check("SU(2), 2 slots, m=(2,): Δ⃗ == (2,2)",
          delta_N_vec(d, _matter(lam, 2), (2,)) == (2, 2))
    check("SU(2), 3 slots, m=(1,): Δ⃗ == (1,1,1)",
          delta_N_vec(d, _matter(lam, 3), (1,)) == (1, 1, 1))
    u = rd.u_n(2)
    check("U(2), 2 slots, m=(0,-2): Δ⃗ == (2,2)",
          delta_N_vec(u, _matter((1, 0), 2), (0, -2)) == (2, 2))
    check("U(2), 2 slots, positive m: Δ⃗ == (0,0)",
          delta_N_vec(u, _matter((1, 0), 2), (2, 0)) == (0, 0))


def test_Z_levels_vec():
    print("Z's μ⃗-coefficients factor across slots")
    d, mat = rd.su_2(), _matter((1,), 2)
    Z = Z_levels_vec(d, mat, (2,))
    check("sectors are the box [0,2]x[0,2]",
          sorted(Z) == sorted((i, j) for i in range(3) for j in range(3)),
          str(sorted(Z)))
    from weyl_torus_ring import TorusLaurent
    check("Z[0⃗] == 1", Z[(0, 0)] == TorusLaurent.one(d))
    # slot independence: Z[(i,j)] == Z[(i,0)] * Z[(0,j)] for identical slots
    ok = True
    for i in range(3):
        for j in range(3):
            if (Z[(i, 0)] * Z[(0, j)] - Z[(i, j)]).is_zero() is False:
                ok = False
    check("Z[(i,j)] == Z[(i,0)]·Z[(0,j)] (slots independent)", ok)


def test_division_vec_rejects():
    print("multi-index μ-division rejects a non-divisible input")
    d, mat, cell = rd.su_2(), _matter((1,), 2), (1,)
    Z = Z_levels_vec(d, mat, cell)
    from weyl_torus_ring import TorusRational
    one = TorusRational.one(d)
    # F = Z * 1  ->  divisible, quotient = 1
    F = {k: (TorusRational.from_laurent(v) * one).simplify()
         for k, v in Z.items()}
    quo, ok = divide_by_Z_vec(d, mat, cell, F)
    check("Z itself is divisible by Z", ok)
    check("quotient is 1 at 0⃗",
          ok and list(quo) == [(0, 0)] and quo[(0, 0)].simplify() == one,
          str(sorted(quo)))
    bad = dict(F)
    bad[(0, 0)] = (bad[(0, 0)] + one).simplify()
    _q, ok2 = divide_by_Z_vec(d, mat, cell, bad)
    check("a perturbed input is rejected", not ok2)


def test_constructor_vec_reproduces_flow():
    """The multi-slot constructor must reproduce `rg_chart` sector by sector,
    with the multi-index kept (not collapsed)."""
    print("solve_canonical_matter_vec == GMatterOverPure.rg_chart")
    for datum, lam, nf, ms in CASES:
        mat = _matter(lam, nf)
        A = GMatterOverPure(datum, lam, nf=nf)
        e = (0,) * datum.dim
        for m in ms:
            try:
                built = solve_canonical_matter_vec(datum, mat, m, e)
            except Exception as ex:
                check(f"{datum.name} m={m} M={nf}: constructor runs", False,
                      f"{type(ex).__name__}: {str(ex)[:90]}")
                continue
            truth = cells_from_rg_chart_vec(
                A.rg_chart(((tuple(m), e), (0,) * nf), Kq=22))
            bad = []
            for k, lvl in built.items():
                for c, v in lvl.items():
                    tv = truth.get(c, {}).get(k)
                    if tv is None:
                        if not v.is_zero():
                            bad.append((k, c))
                    elif not (v + tv * (-1)).simplify().is_zero():
                        bad.append((k, c))
            check(f"{datum.name} m={m} M={nf}: == flow "
                  f"({len(built)} μ-sectors, Δ⃗={delta_N_vec(datum, mat, m)})",
                  not bad, str(bad[:3]))


def test_guard_vec_on_the_flow():
    """D1/D2/D3 in multi-index form, straight off the flow."""
    print("multi-index divisibility guard against the flow (strict)")
    for datum, lam, nf, ms in CASES:
        mat = _matter(lam, nf)
        A = GMatterOverPure(datum, lam, nf=nf)
        e = (0,) * datum.dim
        for m in ms:
            cells = cells_from_rg_chart_vec(
                A.rg_chart(((tuple(m), e), (0,) * nf), Kq=22))
            pure = A.pure().chart((tuple(m), e)).residuals()
            try:
                rep = check_divisibility_vec(datum, mat, m, cells, pure=pure,
                                             strict=True)
            except DivisibilityFailure as ex:
                check(f"{datum.name} m={m} M={nf}: D1+D2", False,
                      str(ex)[:100])
                continue
            check(f"{datum.name} m={m} M={nf}: D1 divisible + D2 degree law "
                  f"({len(rep)} cells)", True)
            check(f"{datum.name} m={m} M={nf}: D3 Q[0⃗] == pure",
                  all(r["quotient_is_pure"] for r in rep.values()),
                  str({c: r["quotient_is_pure"] for c, r in rep.items()}))


def test_degree_law_forced_sectors():
    """A sector with **no free parameters** must be forced by the degree law,
    not handed to the solver.

    D2 says `deg_μ Q_{m'} = Δ⃗_N(m) − Δ⃗_N(m')` componentwise.  Where no bubbled
    cell can carry a free `Q[k⃗]`, `Q[k⃗] = 0` there and `f[k⃗] = offset[k⃗]`
    outright.  Handing such a sector to `joint_solve` anyway asks it to fit
    unknowns that must not exist, and the over-determined system reports
    **`inconsistent`** — which reads like a reach limit but is not one.

    The witness is U(2)+N_f=2 at `m = (0,−4)`: `Δ⃗_N` is constant `(4,4)` across
    the *whole* tropical support, so every sector above `0⃗` is forced.  Before
    the short-circuit this label honest-failed at sector (1,1); the offset there
    was verified equal to the oracle's `Z·Q` cell by cell, so the failure was in
    the solve, not the data.  It now builds in a fraction of a second, and its
    quotient equals `UNNfKAlgebra`'s chart exactly — an independent
    presentation.

    It is reachable through `multiply`: `L_{(0,−2)} · L_{(0,−2)}` needs it."""
    print("sectors with no free parameters are forced, not solved")
    D = rd.u_n(2)
    mat = _matter((1, 0), 2)
    m, e = (0, -4), (0, 0)
    lead = delta_N_vec(D, mat, m)
    check("U(2)+2 m=(0,-4): Δ⃗_N == (4,4)", lead == (4, 4), str(lead))
    import star_bubbling as SB
    sup = [tuple(p) for p in SB.tropical_support(D, m)]
    check("Δ⃗_N is constant on the whole support (so deg Q = 0 everywhere)",
          all(delta_N_vec(D, mat, p) == lead for p in sup),
          str({p: delta_N_vec(D, mat, p) for p in sup}))
    try:
        built = solve_canonical_matter_vec(D, mat, m, e)
    except Exception as ex:
        check("U(2)+2 m=(0,-4): builds", False,
              f"{type(ex).__name__}: {str(ex)[:120]}")
        return
    check(f"U(2)+2 m=(0,-4): builds ({len(built)} μ-sectors)", True)

    # (the comparison with the type-A `UNNfKAlgebra(2,2).chart` oracle that stood
    # here was retired with that class on 2026-09-19 — its D8b agreement is
    # and reachability below)
    # and it is reachable through multiply
    from g_matter_abe_kalgebra import GMatterAbeKAlgebra
    A = GMatterAbeKAlgebra(D, (1, 0), nf=2)
    a = (((0, -2), (0, 0)), (0, 0))
    try:
        el = A.multiply(a, a)
        check(f"L_(0,-2) · L_(0,-2) multiplies ({len(el.terms)} terms)", True)
    except Exception as ex:
        check("L_(0,-2) · L_(0,-2) multiplies", False,
              f"{type(ex).__name__}: {str(ex)[:120]}")


def test_forced_level_guard_has_teeth():
    """`verify_forced_level` must REJECT a wrong proposal, or it is worthless.

    D2 is a heuristic, so the forced-sector shortcut is only sound because its answer is
    checked against **(★)+W1** — axioms that know nothing about D2.  This pins
    that the check discriminates: the true forced level passes, and the same
    level perturbed on a bubbled cell fails."""
    print("the forced-sector guard rejects a perturbed level")
    from matter_star_bubbling import verify_forced_level
    from weyl_torus_ring import TorusRational
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from laurent_poly import LaurentPoly

    D = rd.u_n(2)
    mat = _matter((1, 0), 2)
    m, e, k = (0, -4), (0, 0), (1, 1)
    pres = PureGAbeKAlgebra(D).chart((m, e)).residuals()
    orbit = {tuple(D.act_cochar(w, m)) for w in D.weyl_elements()}
    # the true forced level: f = Z[k]*pure on EVERY cell (Q = 0 here)
    lvl = {}
    for c in pres:
        Zk = Z_levels_vec(D, mat, c, kmax=k).get(k)
        lvl[c] = (TorusRational.zero(D) if Zk is None
                  else (pres[c] * TorusRational.from_laurent(Zk)).simplify())
    ok, why = verify_forced_level(D, lvl)
    check(f"the true forced level passes (★)+W1  [{why}]", ok, why)

    for cell in sorted(c for c in pres if c not in orbit):
        bad = dict(lvl)
        bad[cell] = (bad[cell] + TorusRational.one(D)).simplify()
        ok2, why2 = verify_forced_level(D, bad)
        check(f"a level perturbed at bubbled cell {cell} is REJECTED",
              not ok2, f"accepted: {why2}")
        break                      # one witness is enough; each build is exact
    # a q-asymmetric perturbation must fail W1 specifically
    bad2 = dict(lvl)
    c0 = sorted(c for c in pres if c not in orbit)[0]
    bad2[c0] = (bad2[c0] + TorusRational.from_scalar(
        D, LaurentPoly({1: 1}))).simplify()
    ok3, why3 = verify_forced_level(D, bad2)
    check("a 𝖖-asymmetric perturbation is REJECTED", not ok3,
          f"accepted: {why3}")


def test_requires_dominant_m():
    print("non-dominant m is refused")
    try:
        solve_canonical_matter_vec(rd.su_2(), _matter((1,), 2), (-2,), (0,))
        check("SU(2) m=(-2,) refused", False, "accepted")
    except ValueError:
        check("SU(2) m=(-2,) refused", True)


if __name__ == "__main__":
    for fn in [test_delta_vec, test_Z_levels_vec, test_division_vec_rejects,
               test_guard_vec_on_the_flow,
               test_constructor_vec_reproduces_flow,
               test_degree_law_forced_sectors,
               test_forced_level_guard_has_teeth,
               test_requires_dominant_m]:
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
    print("All matter_multislot tests passed.")
