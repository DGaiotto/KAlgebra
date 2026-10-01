"""The [A₁,E₈] and [A₁,E₆] seed traces from W₃(3,8) / W₃(3,7) characters
(`w3_seeds`, 2026-09-23) against independent routes, with
controls.

  * positive control of the character engine: W₂(2,5) = M(2,5) gives the
    Rogers–Ramanujan sums;
  * the vacuum χ₀ against the K = 6 bootstrap table (a fixture below) and the
    `E8RGKAlgebra` flow vacuum (𝖖¹⁸, a flow into `U1A1AoddKAlg(3)`);
  * every seed against the K = 6 bootstrap table and against the ansatz-FREE
    solve of the orthonormality bootstrap, which pins every seed uniquely
    through 𝖖³⁶ (fixtures below, from the design record scouting run of 2026-09-23);
  * negative control: a perturbed recipe is caught by the fixtures;
  * the class `FiniteE8KAlgebra` now answers past its old 𝖖⁶ cap, with
    I(a, b) = δ + O(𝖖) on a window and leading multiplicativity;
  * E₆ the same way: the vacuum against the exact Nahm-sum vacuum (𝖖¹²)
    and every seed against the ansatz-free solve (𝖖³⁷–𝖖⁴⁰); the frozen e6
    table was removed on 2026-09-23.

Run:  `python3 run_tests.py`
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

from fractions import Fraction as Fr

import w3_seeds as W

E = W.E8

PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# The K = 6 e8 table that `elem_traces.freeze` generated through the
# orthonormality bootstrap (a probe in the source repository).  It was served
# from `finite_kalgebras/elem_trace_data.py` until 2026-09-23, when the W₃(3,8)
# recipes replaced it on the serving path; its values stay
# here, verbatim, as an independent witness.
E8_TABLE_K6 = {
    'K': 6,
    'identity': {0: 1, 4: 1, 6: 2},
    'orbits': {0: {4: 1},
               1: {1: -1, 5: -2},
               2: {1: -1, 5: -2},
               5: {2: 1, 6: 1},
               8: {1: -1, 5: -1},
               10: {2: 1, 6: 2},
               13: {2: 1, 6: 2},
               15: {1: -1, 5: -1},
               17: {1: -1, 5: -1},
               21: {2: 1, 6: 1},
               22: {2: 1, 6: 1},
               23: {2: 1, 6: 1},
               24: {4: 1},
               26: {3: -1},
               27: {1: -1, 5: -1},
               28: {3: -1}},
}
# ansatz-free orthonormality-bootstrap solve, every seed pinned through 𝖖³⁶
# (the eight distinct values; the other eight seeds equal these by ρ)
PINNED_36 = {
    0: {4: 1, 10: 1, 12: 1, 14: 2, 16: 3, 18: 3, 20: 5, 22: 7, 24: 9, 26: 11, 28: 17, 30: 21, 32: 28, 34: 36, 36: 47},
    1: {1: -1, 5: -2, 7: -1, 9: -4, 11: -4, 13: -7, 15: -10, 17: -14, 19: -19, 21: -29, 23: -36, 25: -51, 27: -66, 29: -93, 31: -116, 33: -157, 35: -200},
    5: {2: 1, 6: 1, 8: 1, 10: 3, 12: 3, 14: 4, 16: 7, 18: 10, 20: 12, 22: 19, 24: 24, 26: 34, 28: 42, 30: 59, 32: 76, 34: 100, 36: 126},
    8: {1: -1, 5: -1, 7: -1, 9: -3, 11: -3, 13: -6, 15: -7, 17: -10, 19: -14, 21: -21, 23: -26, 25: -38, 27: -48, 29: -67, 31: -84, 33: -114, 35: -144},
    10: {2: 1, 6: 2, 8: 1, 10: 3, 12: 4, 14: 6, 16: 9, 18: 13, 20: 16, 22: 25, 24: 32, 26: 45, 28: 56, 30: 80, 32: 101, 34: 135, 36: 171},
    17: {1: -1, 5: -1, 7: -2, 9: -3, 11: -3, 13: -7, 15: -8, 17: -12, 19: -17, 21: -24, 23: -31, 25: -45, 27: -57, 29: -78, 31: -101, 33: -135, 35: -172},
    21: {2: 1, 6: 1, 8: 1, 10: 3, 12: 3, 14: 5, 16: 7, 18: 10, 20: 13, 22: 20, 24: 25, 26: 36, 28: 45, 30: 63, 32: 80, 34: 107, 36: 135},
    26: {3: -1, 7: -1, 9: -1, 11: -2, 13: -3, 15: -4, 17: -6, 19: -9, 21: -11, 23: -16, 25: -21, 27: -30, 29: -37, 31: -51, 33: -66, 35: -87},
}
# E8RGKAlgebra vacuum through 𝖖¹⁸ (same scouting run)
FLOW_VACUUM_18 = {0: 1, 4: 1, 6: 2, 8: 3, 10: 4, 12: 7, 14: 8, 16: 14, 18: 18}
# E6: the ansatz-free solve, (window, series)
E6_PINNED = {
    0: (40, {1: -1, 5: -2, 7: -1, 9: -3, 11: -4, 13: -5, 15: -8, 17: -12, 19: -14, 21: -20, 23: -26, 25: -37, 27: -44, 29: -61, 31: -76, 33: -101, 35: -125, 37: -162, 39: -199}),
    2: (40, {1: -1, 5: -1, 7: -1, 9: -3, 11: -3, 13: -4, 15: -6, 17: -9, 19: -11, 21: -16, 23: -20, 25: -29, 27: -34, 29: -47, 31: -59, 33: -78, 35: -96, 37: -125, 39: -153}),
    4: (38, {3: -1, 9: -1, 11: -1, 13: -2, 15: -3, 17: -3, 19: -4, 21: -6, 23: -8, 25: -10, 27: -14, 29: -17, 31: -23, 33: -29, 35: -37, 37: -46}),
    5: (37, {2: 1, 6: 1, 8: 1, 10: 2, 12: 3, 14: 4, 16: 5, 18: 8, 20: 10, 22: 14, 24: 17, 26: 25, 28: 30, 30: 41, 32: 51, 34: 67, 36: 83}),
}
RHO_PAIRS = [(0, 24), (1, 2), (5, 22), (8, 15), (10, 13), (17, 27), (21, 23), (26, 28)]


def test_engine_control():
    """W₂(2,5) = M(2,5): the two normalised characters are the Rogers–Ramanujan
    sums Σ q^{n²}/(q)_n and Σ q^{n²+n}/(q)_n."""
    M = 40
    ch = W.distinct_characters(2, 2, 5, M)

    def rr(shift):
        out = [0] * (M + 1)
        n = 0
        while n * n + shift * n <= M:
            part = [0] * (M + 1)
            part[n * n + shift * n] = 1
            for j in range(1, n + 1):                  # divide by (1 - q^j)
                for m in range(j, M + 1):
                    part[m] += part[m - j]
            out = [a + b for a, b in zip(out, part)]
            n += 1
        return out

    got = sorted(tuple(v[:M + 1]) for v in ch.values())
    want = sorted([tuple(rr(0)), tuple(rr(1))])
    check("character engine: W2(2,5) characters == Rogers-Ramanujan sums through q^40", got == want)
    check("W3(3,8) has exactly the five weights 0, -1/2, -3/4, -7/8, -1",
          set(W.distinct_characters(3, 3, 8, 4)) == {Fr(0), Fr(-1, 2), Fr(-3, 4), Fr(-7, 8), Fr(-1)})


def test_vacuum():
    fz = E8_TABLE_K6
    v = E.vacuum_trace(fz["K"])
    check("Tr 1 = chi_0 equals the K = 6 bootstrap table (fixture) through q^%d" % fz["K"],
          v == {e: c for e, c in fz["identity"].items() if c})
    check("Tr 1 = chi_0 equals the E8RGKAlgebra flow vacuum through q^18",
          E.vacuum_trace(18) == FLOW_VACUUM_18)


def test_seeds():
    fz = E8_TABLE_K6
    ok_fz = all(E.seed_trace(s, fz["K"]) == {e: c for e, c in fz["orbits"][s].items() if c}
                for s in E.recipes)
    check(f"all {len(E.recipes)} seeds equal the K = 6 bootstrap table (fixture) through q^{fz['K']}",
          ok_fz and sorted(E.recipes) == sorted(fz["orbits"]))
    ok_deep = all(E.seed_trace(s, 36) == ser for s, ser in PINNED_36.items())
    check("the 8 distinct seeds equal the ansatz-free bootstrap solve through q^36", ok_deep)
    check("rho-related seeds have equal traces", all(E.seed_trace(a, 60) == E.seed_trace(b, 60) for a, b in RHO_PAIRS))


def test_negative_control():
    """Perturbing one recipe coefficient is caught by the deep fixtures."""
    saved = E.recipes[26]
    try:
        E.recipes[26] = saved + [("-1", 21, 1)]      # an extra term deep in the window
        caught = E.seed_trace(26, 36) != PINNED_36[26]
    finally:
        E.recipes[26] = saved
    check("negative control: a perturbed recipe differs from the pinned series", caught)


def test_class():
    from finite_kalgebras import FINITE_KALGEBRAS
    A = FINITE_KALGEBRAS["e8"]()
    r = A.trace(A.identity(), 20)
    check("FiniteE8KAlgebra answers Tr 1 through q^20 (was capped at q^6)",
          {e: int(str(c)) for e, c in r.coeffs.items() if int(str(c))} == E.vacuum_trace(20))

    def val(lab, K):
        t = {e: int(str(c)) for e, c in A.trace(lab, K).coeffs.items() if int(str(c))}
        return min(t) if t else None
    ok = all(val(((i, m),), 16) == m * val(((i, 1),), 16) for i in (0, 1, 5) for m in (2, 3))
    check("leading multiplicativity val Tr(g^m) = m val Tr(g) (g = L_0, L_1, L_5; m = 2, 3)", ok)
    window = [A.identity()] + [((i, 1),) for i in range(8)]
    bad = 0
    for a in window:
        for b in window:
            low = {e: int(str(c)) for e, c in A.inner_product(a, b, K=1).coeffs.items()
                   if e <= 0 and int(str(c))}
            bad += low != ({0: 1} if a == b else {})
    check(f"I(a, b) = delta + O(q) on a window of {len(window)} labels (all pairs)", bad == 0)


def test_e6():
    from elem_traces import _vacuum_rps, _series_to_data
    from finite_kalgebras import FINITE_KALGEBRAS
    E6 = W.E6
    check("E6: W3(3,7) has exactly the four weights 0, -3/7, -4/7, -5/7",
          set(W.distinct_characters(3, 3, 7, 4)) == {Fr(0), Fr(-3, 7), Fr(-4, 7), Fr(-5, 7)})
    # The frozen e6 table was removed on 2026-09-23; the vacuum's
    # table-free witness is the exact Nahm sum on the embedded spectrum.
    Kn = 12
    nahm = {int(e): int(c) for e, c in _series_to_data("e6", _vacuum_rps("e6", Kn)).items() if int(c)}
    check(f"E6: Tr 1 = chi_0 equals the exact Nahm-sum vacuum through q^{Kn}", E6.vacuum_trace(Kn) == nahm)
    check("E6: seeds equal the ansatz-free bootstrap solve through q^37-q^40",
          all(E6.seed_trace(s, K) == ser for s, (K, ser) in E6_PINNED.items()))
    check("E6: rho-related seeds equal (2 = 6, 5 = 7)",
          E6.seed_trace(2, 60) == E6.seed_trace(6, 60) and E6.seed_trace(5, 60) == E6.seed_trace(7, 60))
    # E6 IS served from these recipes (the design record, item 5); comp:e6/traces runs
    # the Nahm-sum + bootstrap route as the witness with the recipes absent.
    A = FINITE_KALGEBRAS["e6"]()
    served = {e: int(str(c)) for e, c in A.trace(((0, 1),), 12).coeffs.items() if int(str(c))}
    check("E6: the class's served Tr(L_0) equals the recipe through q^12", served == E6.seed_trace(0, 12))


if __name__ == "__main__":
    test_engine_control()
    test_vacuum()
    test_seeds()
    test_negative_control()
    test_class()
    test_e6()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
