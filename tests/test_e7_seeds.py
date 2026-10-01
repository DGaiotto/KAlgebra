"""The [A₁,E₇] seed traces from their closed forms (`e7_seeds`,
the design record) against independent routes, with
controls.

  * positive control of the series engine: the Jacobi triple product for
    θ(μ), and `D · D⁻¹ = 1` for the denominator `(q;q)_∞ θ(μ)`;
  * the vacuum (the Bershadsky–Polyakov product at level −12/5) against the
    Nahm sum on the BPS spectrum through 𝖖²⁰ (fixture), with the wrong levels
    u = 4, 7 as negative controls;
  * the eight distinct representatives against the certified orthonormality
    bootstrap (`u1_bootstrap.generate_u1`, under its recorded
    μ-support hypothesis `|μ| ≤ 4k`, ρ-fold), pinned through 𝖖²⁰ (fixture below,
    from the run of 2026-09-23: 569,185 rows, 17,777 determined values);
    seeds 0 and 1 are equal;
  * the ρ-rule: the recipes' nine ρ-orbits cover the 90 seeds, and the rule
    `T_{ρ(i)}(μ) = μ^{−δ_i} T_i(μ⁻¹)` closes around every orbit;
  * the recorded μ-support hypothesis holds for every closed form through 𝖖⁶⁰;
  * negative control: a perturbed recipe is caught by the fixture;
  * the zoo class `FiniteE7KAlgebra` serves these closed forms (`Tr 1` through
    𝖖³⁰, all 90 seeds), with neither the BPS engine nor the bootstrap loaded,
    with leading multiplicativity, and with orthonormality in the ELEMENT form
    `Tr(ρ(L_a)·L_b) = δ + O(𝖖)` (ρ the automorphism `rho_element`; the
    label-form `inner_product` is off by the unit character `μ^{δ(a)}` on this
    tier — the encoding split recorded in the audit);
  * slow tier (`--slow`): all 90 seeds against a live bootstrap with the
    ρ²-FOLD (which does not use the ρ-rule, so it checks the transported seeds
    independently).

Run:  `python3 run_tests.py` [--slow]
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

import e7_seeds as E

PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# the certified orthonormality bootstrap, pinned through 𝖖²⁰: {seed: {𝖖-power: {μ-power: c}}}
PINNED_20 = {
    0: {3: {-1: -1}, 7: {-1: -1}, 8: {-2: 1}, 9: {-1: -2}, 10: {-2: 1, 0: 1}, 11: {-1: -3}, 12: {-2: 2, 0: 1}, 13: {-3: -1, -1: -5}, 14: {-2: 4, 0: 2}, 15: {-3: -1, -1: -9}, 16: {-2: 7, 0: 4}, 17: {-3: -3, -1: -12, 1: -1}, 18: {-4: 1, -2: 11, 0: 7}, 19: {-3: -5, -1: -21, 1: -1}, 20: {-4: 1, -2: 19, 0: 11}},
    2: {2: {-1: 1}, 6: {-1: 2}, 7: {-2: -1, 0: -1}, 8: {-1: 2}, 9: {-2: -1, 0: -1}, 10: {-1: 5}, 11: {-2: -3, 0: -2}, 12: {-3: 1, -1: 7, 1: 1}, 13: {-2: -5, 0: -5}, 14: {-3: 1, -1: 13, 1: 1}, 15: {-2: -9, 0: -8}, 16: {-3: 4, -1: 19, 1: 3}, 17: {-4: -1, -2: -15, 0: -14, 2: -1}, 18: {-3: 6, -1: 34, 1: 5}, 19: {-4: -1, -2: -27, 0: -24, 2: -1}, 20: {-3: 13, -1: 49, 1: 11}},
    3: {2: {0: 1}, 6: {0: 2}, 7: {-1: -1, 1: -1}, 8: {0: 2}, 9: {-1: -1, 1: -1}, 10: {0: 4}, 11: {-1: -2, 1: -2}, 12: {-2: 1, 0: 7, 2: 1}, 13: {-1: -5, 1: -5}, 14: {-2: 1, 0: 12, 2: 1}, 15: {-1: -8, 1: -8}, 16: {-2: 3, 0: 17, 2: 3}, 17: {-3: -1, -1: -13, 1: -13, 3: -1}, 18: {-2: 5, 0: 31, 2: 5}, 19: {-3: -1, -1: -23, 1: -23, 3: -1}, 20: {-2: 11, 0: 45, 2: 11}},
    5: {1: {0: -1}, 3: {0: -1}, 4: {-1: 1, 1: 1}, 5: {0: -3}, 6: {-1: 2, 1: 2}, 7: {-2: -1, 0: -5, 2: -1}, 8: {-1: 4, 1: 4}, 9: {-2: -2, 0: -11, 2: -2}, 10: {-3: 1, -1: 9, 1: 9, 3: 1}, 11: {-2: -5, 0: -18, 2: -5}, 12: {-3: 2, -1: 17, 1: 17, 3: 2}, 13: {-4: -1, -2: -10, 0: -34, 2: -10, 4: -1}, 14: {-3: 5, -1: 31, 1: 31, 3: 5}, 15: {-4: -2, -2: -21, 0: -57, 2: -21, 4: -2}, 16: {-5: 1, -3: 11, -1: 56, 1: 56, 3: 11, 5: 1}, 17: {-4: -5, -2: -38, 0: -98, 2: -38, 4: -5}, 18: {-5: 2, -3: 22, -1: 96, 1: 96, 3: 22, 5: 2}, 19: {-6: -1, -4: -11, -2: -70, 0: -159, 2: -70, 4: -11, 6: -1}, 20: {-5: 5, -3: 42, -1: 161, 1: 161, 3: 42, 5: 5}},
    7: {1: {0: -1}, 5: {0: -2}, 6: {-1: 1, 1: 1}, 7: {0: -2}, 8: {-1: 1, 1: 1}, 9: {0: -6}, 10: {-1: 3, 1: 3}, 11: {-2: -1, 0: -7, 2: -1}, 12: {-1: 5, 1: 5}, 13: {-2: -1, 0: -14, 2: -1}, 14: {-1: 9, 1: 9}, 15: {-2: -4, 0: -21, 2: -4}, 16: {-3: 1, -1: 16, 1: 16, 3: 1}, 17: {-2: -6, 0: -37, 2: -6}, 18: {-3: 1, -1: 28, 1: 28, 3: 1}, 19: {-2: -13, 0: -53, 2: -13}, 20: {-3: 4, -1: 44, 1: 44, 3: 4}},
    12: {2: {0: 1}, 3: {-1: -1}, 4: {0: 1}, 5: {-1: -1, 1: -1}, 6: {-2: 1, 0: 3}, 7: {-1: -3, 1: -2}, 8: {-2: 2, 0: 5, 2: 1}, 9: {-3: -1, -1: -6, 1: -4}, 10: {-2: 4, 0: 11, 2: 2}, 11: {-3: -2, -1: -12, 1: -9, 3: -1}, 12: {-4: 1, -2: 9, 0: 19, 2: 5}, 13: {-3: -5, -1: -22, 1: -17, 3: -2}, 14: {-4: 2, -2: 17, 0: 35, 2: 10, 4: 1}, 15: {-5: -1, -3: -10, -1: -40, 1: -31, 3: -5}, 16: {-4: 5, -2: 32, 0: 59, 2: 21, 4: 2}, 17: {-5: -2, -3: -21, -1: -68, 1: -56, 3: -11, 5: -1}, 18: {-6: 1, -4: 11, -2: 57, 0: 102, 2: 38, 4: 5}, 19: {-5: -5, -3: -38, -1: -117, 1: -96, 3: -22, 5: -2}, 20: {-6: 2, -4: 22, -2: 100, 0: 166, 2: 70, 4: 11, 6: 1}},
    13: {1: {0: -1}, 2: {-1: 1}, 3: {0: -1}, 4: {-1: 1, 1: 1}, 5: {-2: -1, 0: -3}, 6: {-1: 4, 1: 2}, 7: {-2: -2, 0: -6, 2: -1}, 8: {-3: 1, -1: 7, 1: 5}, 9: {-2: -5, 0: -13, 2: -2}, 10: {-3: 2, -1: 15, 1: 10, 3: 1}, 11: {-4: -1, -2: -11, 0: -22, 2: -6}, 12: {-3: 6, -1: 27, 1: 20, 3: 2}, 13: {-4: -2, -2: -21, 0: -42, 2: -12, 4: -1}, 14: {-5: 1, -3: 12, -1: 50, 1: 37, 3: 6}, 15: {-4: -6, -2: -40, 0: -72, 2: -25, 4: -2}, 16: {-5: 2, -3: 26, -1: 86, 1: 68, 3: 13, 5: 1}, 17: {-6: -1, -4: -13, -2: -73, 0: -125, 2: -46, 4: -6}, 18: {-5: 6, -3: 47, -1: 150, 1: 117, 3: 27, 5: 2}, 19: {-6: -2, -4: -27, -2: -128, 0: -207, 2: -86, 4: -13, 6: -1}, 20: {-7: 1, -5: 13, -3: 89, -1: 247, 1: 201, 3: 51, 5: 6}},
    21: {1: {-1: -1}, 2: {0: 1}, 3: {-1: -1}, 4: {-2: 1, 0: 1}, 5: {-1: -4, 1: -1}, 6: {-2: 2, 0: 4}, 7: {-3: -1, -1: -6, 1: -2}, 8: {-2: 5, 0: 7, 2: 1}, 9: {-3: -2, -1: -14, 1: -5}, 10: {-4: 1, -2: 11, 0: 15, 2: 2}, 11: {-3: -6, -1: -24, 1: -11, 3: -1}, 12: {-4: 2, -2: 21, 0: 28, 2: 6}, 13: {-5: -1, -3: -12, -1: -45, 1: -21, 3: -2}, 14: {-4: 6, -2: 39, 0: 51, 2: 12, 4: 1}, 15: {-5: -2, -3: -26, -1: -77, 1: -40, 3: -6}, 16: {-6: 1, -4: 13, -2: 72, 0: 88, 2: 26, 4: 2}, 17: {-5: -6, -3: -47, -1: -134, 1: -73, 3: -13, 5: -1}, 18: {-6: 2, -4: 27, -2: 124, 0: 154, 2: 47, 4: 6}, 19: {-7: -1, -5: -13, -3: -89, -1: -219, 1: -129, 3: -27, 5: -2}, 20: {-6: 6, -4: 52, -2: 212, 0: 254, 2: 89, 4: 13, 6: 1}},
}

# the Nahm sum on the BPS spectrum (elem_traces._vacuum_rps), through 𝖖²⁰: {𝖖-power: {μ-power: c}}
NAHM_VACUUM_20 = {0: {0: 1}, 2: {0: 1}, 3: {-1: -1, 1: -1}, 4: {0: 3}, 5: {-1: -2, 1: -2}, 6: {-2: 1, 0: 6, 2: 1}, 7: {-1: -5, 1: -5}, 8: {-2: 2, 0: 12, 2: 2}, 9: {-3: -1, -1: -10, 1: -10, 3: -1}, 10: {-2: 6, 0: 21, 2: 6}, 11: {-3: -2, -1: -20, 1: -20, 3: -2}, 12: {-4: 1, -2: 12, 0: 40, 2: 12, 4: 1}, 13: {-3: -6, -1: -37, 1: -37, 3: -6}, 14: {-4: 2, -2: 25, 0: 67, 2: 25, 4: 2}, 15: {-5: -1, -3: -13, -1: -67, 1: -67, 3: -13, 5: -1}, 16: {-4: 6, -2: 46, 0: 117, 2: 46, 4: 6}, 17: {-5: -2, -3: -27, -1: -116, 1: -116, 3: -27, 5: -2}, 18: {-6: 1, -4: 13, -2: 86, 0: 193, 2: 86, 4: 13, 6: 1}, 19: {-5: -6, -3: -51, -1: -198, 1: -198, 3: -51, 5: -6}, 20: {-6: 2, -4: 28, -2: 149, 0: 319, 2: 149, 4: 28, 6: 2}}


def _flat(series):
    """`{q: {(m,): c}}` -> `{q: {m: c}}`."""
    return {q: {mu[0]: c for mu, c in d.items()} for q, d in series.items()}


def test_engine_control():
    K = 40
    th = E._theta_mu(K)
    jtp = E._prod_one_minus([2 * n for n in range(1, K // 2 + 1)], K)
    for n in range(1, K + 1, 2):                       # (1 + 𝖖ⁿ μ)(1 + 𝖖ⁿ μ⁻¹)
        jtp = E._mul(jtp, {(0, 0): 1, (n, 1): 1}, K)
        jtp = E._mul(jtp, {(0, 0): 1, (n, -1): 1}, K)
    check("engine: theta(mu) == the Jacobi triple product through q^40", th == jtp)
    D = E._mul(E._prod_one_minus([2 * n for n in range(1, K // 2 + 1)], K), th, K)
    check("engine: D * D^-1 == 1 for D = (q;q)_inf theta(mu), through q^40",
          E._mul(D, E._inverse(D, K), K) == {(0, 0): 1})


def test_vacuum():
    check("Tr 1 (BP product, u = 5) equals the Nahm sum on the BPS spectrum through q^20",
          _flat(E.vacuum_trace(20)) == NAHM_VACUUM_20)
    saved = E._U
    try:
        wrong = []
        for u in (4, 7):
            E._U = u
            wrong.append(_flat(E.vacuum_trace(20)) != NAHM_VACUUM_20)
    finally:
        E._U = saved
    check("negative control: the levels u = 4, 7 give a different vacuum", all(wrong))
    v = E.vacuum_trace(40)
    check("Tr 1 is mu -> 1/mu symmetric (through q^40)",
          all(d.get((-m[0],)) == c for d in v.values() for m, c in d.items()))


def test_certified_fixture():
    ok = all(_flat(E.seed_trace(r, 20)) == ser for r, ser in PINNED_20.items())
    check(f"the {len(PINNED_20)} distinct representatives equal the certified bootstrap through q^20", ok)
    check("seeds 0 and 1 have equal traces (through q^60)", E.seed_trace(0, 60) == E.seed_trace(1, 60))


def test_rho_rule():
    reps = {E.rho_orbit_representative(i) for i in range(90)}
    check("the recipes' rho-orbits cover all 90 seeds", reps == set(E.RECIPES))
    from finite_e7_kalg import E7_RHO_PERM, E7_RHO_DELTA
    ok = True
    for r in E.RECIPES:
        T = {(q, m[0]): c for q, d in E.seed_trace(r, 30).items() for m, c in d.items()}
        x, steps = r, []
        while True:
            steps.append(x)
            x = E7_RHO_PERM[x]
            if x == r:
                break
        back = E._transport(dict(T), steps)
        ok &= back == T
    check("the rho-rule closes around every orbit (transport by the whole orbit is the identity)", ok)


def test_mu_support():
    bad = []
    for i in range(90):
        for q, d in E.seed_trace(i, 60).items():
            if any(abs(m[0]) > 4 * q for m in d):
                bad.append((i, q))
    check("every closed form obeys the recorded mu-support hypothesis |mu| <= 4k through q^60", not bad)


def test_negative_control():
    saved = E.RECIPES[13]
    try:
        E.RECIPES[13] = E._merge((0, saved), (0, [(1, 12, 0, "th2", 0, 4)]))
        E._BLOCKS.clear()
        caught = _flat(E.seed_trace(13, 20)) != PINNED_20[13]
    finally:
        E.RECIPES[13] = saved
        E._BLOCKS.clear()
    check("negative control: one extra term deep in a recipe is caught by the fixture", caught)
    check("negative control restored", _flat(E.seed_trace(13, 20)) == PINNED_20[13])


def _flat_rps(ps, K):
    out = {}
    for e, c in ps.coeffs.items():
        d = {k: int(v) for k, v in c.terms.items() if v}
        if d and e <= K:
            out[e] = d
    return out


def test_class():
    from finite_e7_kalg import FiniteE7KAlgebra
    from kalgebra import Element
    from zplus_ring import RLaurent, RElement
    A = FiniteE7KAlgebra()
    check("FiniteE7KAlgebra serves Tr 1 = the BP product through q^30",
          _flat_rps(A.trace(A.identity(), 30), 30) == E.vacuum_trace(30))
    check("FiniteE7KAlgebra serves all 90 seed traces from the recipes through q^16",
          all(_flat_rps(A.trace(((i, 1),), 16), 16) == E.seed_trace(i, 16) for i in range(90)))
    check("no BPS engine and no bootstrap loaded by the served route",
          "bps_kalgebra" not in sys.modules and "finite_kalgebras.u1_bootstrap" not in sys.modules)

    def val(lab, K):
        t = _flat_rps(A.trace(lab, K), K)
        return min(t) if t else None
    ok = all(val(((i, m),), 16) == m * val(((i, 1),), 16) for i in (0, 5, 12) for m in (2, 3))
    check("leading multiplicativity val Tr(g^m) = m val Tr(g) (g = L_0, L_5, L_12; m = 2, 3)", ok)
    R = A.coefficient_ring()
    one = RLaurent(R, {0: RElement(R, {R.one_basis(): 1})})
    window = [A.identity()] + [((i, 1),) for i in (0, 3, 4, 5, 8, 12, 21, 22, 70)]
    bad = 0
    for a in window:
        x = A.rho_element(Element({a: one}))
        for b in window:
            I = A.trace_element(A.multiply_elements(x, Element({b: one})), 1)
            low = {e: d for e, d in _flat_rps(I, 1).items() if e <= 0}
            bad += low != ({0: {(0,): 1}} if a == b else {})
    check(f"element form Tr(rho(L_a) L_b) = delta + O(q) on a window of {len(window)} labels "
          f"(all pairs; includes delta = -1, -2 generators)", bad == 0)


def test_rho2_fold_bootstrap_slow(K=6):
    from u1_bootstrap import generate_u1
    rec = generate_u1("e7", K, fold="rho2")
    ok = all({q: {m: c for m, c in d.items() if c} for q, d in rec["orbits"][i].items()
              if q <= K and any(d.values())} == E.seed_trace(i, K) for i in range(90))
    check(f"all 90 seeds equal a live rho2-fold bootstrap through q^{K}", ok)


if __name__ == "__main__":
    test_engine_control()
    test_vacuum()
    test_certified_fixture()
    test_rho_rule()
    test_mu_support()
    test_negative_control()
    test_class()
    if "--slow" in sys.argv:
        test_rho2_fold_bootstrap_slow()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
