"""Tests for `src/cone/u1a1deven_seed_characters.py` — closed forms for
the seed traces `Tr(seed . E^n)` of the u(1)-gauged `[A_1, D_{2k+2}]`: an odd
curve, or a non-crossing pair of a charge +1 and a charge -1 curve, in the
curve frame of `U1A1DevenConeKAlgebra` (`u1a1deven_geometric_frame`; the design record).

The independent witness is the exact RG transport (`DevenTraceTransport`,
through the frame's closed-form RG image `Phi`), which shares nothing with the
closed forms but the algebra.

Positive controls first: the same evaluator with `P = q^{-p/2}` is Creutzig's
gauge tower (`exact_characters.deven_gauged_xn_qn`).  Then the witness on
every seed orbit at k = 1, 2, 3, at the orbit representative and at rotated
labels (rho with its E-drift).  Negative controls: the curve rule of the wrong
curve type, the wrong U(1) flow, the nested rule with its gaps swapped, and
rho without its E-drift each disagree with the witness.

`slow` adds every seed orbit at k = 4 and a k = 5 sample:
`python3 run_tests.py` slow`.
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "implementations"))

import u1a1deven_seed_characters as M

_FRAMES = {}


def _frame(k):
    if k not in _FRAMES:
        from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
        F = U1A1DevenConeKAlgebra(k)
        F._transport.max_word_degree = 60
        _FRAMES[k] = F
    return _FRAMES[k]


def _witness(k, curves, n, K):
    """`Tr(L_curves E^n)` through q^K by the transport."""
    F = _frame(k)
    raw = F._transport.trace_aux(F._phi2((curves, n)), K)
    out = {q: {i: v for i, v in d.items() if v} for q, d in raw.items()}
    return {q: d for q, d in out.items() if d}


def _seed_orbits(k):
    """One representative per rho-orbit of seeds, from the frame's curve
    enumeration (independent of the module's classification)."""
    from u1a1deven_geometric_frame import _curves, _charge, _curves_cross
    n = 2 * k + 2
    cs = sorted(_curves(n))
    labs = [(((c, 1),),) for c in cs if _charge(c, k) == 0]
    pos = [c for c in cs if _charge(c, k) == 1]
    neg = [c for c in cs if _charge(c, k) == -1]
    labs += [(tuple(sorted(((a, 1), (b, 1)))),) for a in pos for b in neg
             if not _curves_cross(a, b, n)]
    seen, reps = set(), []
    for (lab,) in labs:
        if lab in seen:
            continue
        orbit, cur, e = [], lab, 0
        while cur not in seen:
            seen.add(cur)
            orbit.append(cur)
            cur, e = M.rho(cur, e, k)
        reps.append(min(orbit))
    return reps


def _agree(k, curves, ns, K):
    for n in ns:
        got = M.seed_trace(k, curves, n, K)
        want = _witness(k, curves, n, K)
        assert got == want, (k, curves, n, K, got, want)


# -- positive controls --------------------------------------------------------

def test_gauge_tower_is_creutzig():
    """POSITIVE CONTROL: the evaluator with P = q^{-p/2}, support corner (0, 1), equals
    Creutzig's closed form (k = 1..4, |n| <= 4, through q^40)."""
    from exact_characters import deven_gauged_xn_qn
    for k in range(1, 5):
        for n in range(-4, 5):
            want = {q: d for q, d in deven_gauged_xn_qn(k, n, 40).items() if d}
            assert M.gauge_trace(k, n, 40) == want, (k, n)


def test_curve_rule_two_forms():
    """The curve rule in the natural variables (`curve_F`) equals its
    form in the lattice variables (the polynomial `S_j`), k = 1..7."""
    for k in range(1, 8):
        p = k + 1
        for j in range(1, k + 1):
            A = k + 1 - j
            sg = -1 if (p + j) % 2 else 1
            stair = tuple(sorted((A + ex, -A + ey, 2 * ex, sg * c)
                                 for (ex, ey), c in M._curve_S(j).items()))
            assert M.curve_P(k, j) == ((1, 1), stair), (k, j)


def test_orbit_counts():
    """Seed orbits 2 / 7 / 15 at k = 1 / 2 / 3 (odd curves plus non-crossing
    +1/-1 pairs, up to rho), and the module covers every representative."""
    for k, count in ((1, 2), (2, 7), (3, 15)):
        reps = _seed_orbits(k)
        assert len(reps) == count, (k, len(reps))
        for rep in reps:
            assert M.orbit_rep(rep, 0, k) is not None, (k, rep)


def test_curves_match_transport():
    """Every odd curve type at k = 1, 2, 3 equals the transport (|n| <= 3)."""
    for k, K in ((1, 20), (2, 16), (3, 14)):
        for j in range(1, k + 1):
            _agree(k, (((0, 2 * j + 1), 1),), range(-3, 4), K)


def test_pairs_match_transport():
    """Every pair orbit at k = 1, 2, 3 equals the transport (|n| <= 2):
    nested and side-by-side configurations."""
    for k, K in ((1, 16), (2, 14), (3, 12)):
        for rep in _seed_orbits(k):
            if len(rep) == 2:
                _agree(k, rep, range(-2, 3), K)


def test_rotated_labels_match_transport():
    """rho with its E-drift: a seed rotated off its representative, at a
    shifted E-power, still equals the transport (k = 2, 3)."""
    for k, K in ((2, 12), (3, 10)):
        for rep in _seed_orbits(k):
            cur, e = rep, 0
            for step in range(1, 2 * k + 2):
                cur, e = M.rho(cur, e, k)
                if step in (1, k + 1):
                    _agree(k, cur, (e - 1, e + 2), K)


def test_not_a_seed_returns_none():
    """A squared curve, a crossing pair, a charged curve and the empty label
    are not seeds: `seed_trace` returns None (the caller keeps the
    transport)."""
    k = 2
    assert M.seed_trace(k, (((0, 3), 2),), 0, 6) is None
    assert M.seed_trace(k, (((0, 2), 1),), 0, 6) is None
    assert M.seed_trace(k, (), 0, 6) is None
    assert M.seed_trace(k, (((0, 2), 1), ((1, 2), 1)), 0, 6) is None      # cross


# -- negative controls --------------------------------------------------------

def _disagrees(k, curves, ns, K, patch):
    saved = {name: getattr(M, name) for name in patch}
    try:
        for name, fn in patch.items():
            setattr(M, name, fn)
        return any(M.seed_trace(k, curves, n, K) != _witness(k, curves, n, K)
                   for n in ns)
    finally:
        for name, fn in saved.items():
            setattr(M, name, fn)


def test_negative_controls():
    """Each perturbation disagrees with the transport: the curve rule of the
    other curve type, the U(1) flow A -> A + 1, the nested rule with its gaps
    swapped (on an asymmetric nested pair), rho without its E-drift (off the
    representative)."""
    k, K = 2, 12
    cp = M.curve_P
    assert _disagrees(k, (((0, 3), 1),), range(-2, 3), K,
                      {"curve_P": lambda kk, j: cp(kk, 2)})
    assert _disagrees(k, (((0, 5), 1),), range(-2, 3), K,
                      {"curve_P": lambda kk, j: cp(kk, 1)})

    def flow(kk, j):
        corner, P = cp(kk, j)
        return corner, tuple((i + 1, jj - 1, m2, c) for i, jj, m2, c in P)

    assert _disagrees(k, (((0, 3), 1),), range(-2, 3), K, {"curve_P": flow})
    nf = M.nested_F
    asym = (((0, 2), 1), ((3, 6), 1))                  # nested, gaps (2, 0)
    assert M.pair_configuration(k, asym) == ("nested", 2, 2, 0, -1)
    assert _disagrees(k, asym, range(-2, 3), K,
                      {"nested_F": lambda kk, l, g1, g2: nf(kk, l, g2, g1)})
    r = M.rho

    def nodrift(curves, e, kk):
        c, _ = r(curves, e, kk)
        return c, -e

    lab, e = r((((0, 3), 1),), 0, k)
    lab, e = r(lab, e, k)                              # the curve (2, 3)
    assert _disagrees(k, lab, range(-2, 3), K, {"rho": nodrift})


# -- slow ---------------------------------------------------------------------

def slow_k4_every_orbit():
    """SLOW: every seed orbit at k = 4 (30 of them), |n| <= 2, through q^20;
    nested and side-by-side pairs with every gap pattern, all held out from the
    measurements the rules were read from."""
    k = 4
    reps = _seed_orbits(k)
    assert len(reps) == 30, len(reps)
    for rep in reps:
        _agree(k, rep, range(-2, 3), 20)


def slow_k5_sample():
    """SLOW: at k = 5 the five curve types and the side-by-side pairs with a
    nonzero gap after the charge -1 curve, |n| <= 2, through q^18."""
    k = 5
    for rep in _seed_orbits(k):
        conf = M.pair_configuration(k, rep) if len(rep) == 2 else None
        if len(rep) == 1 or (conf and conf[0] == "side" and conf[3] > 0):
            _agree(k, rep, range(-2, 3), 18)


_TESTS = [test_gauge_tower_is_creutzig, test_curve_rule_two_forms, test_orbit_counts,
          test_curves_match_transport, test_pairs_match_transport,
          test_rotated_labels_match_transport, test_not_a_seed_returns_none,
          test_negative_controls]

if __name__ == "__main__":
    import time
    run = _TESTS + ([slow_k4_every_orbit, slow_k5_sample]
                    if "slow" in sys.argv[1:] else [])
    for t in run:
        t0 = time.time()
        t()
        print(f"  PASS: {t.__name__} [{time.time() - t0:.1f}s]", flush=True)
    print(f"\nAll {len(run)} u1a1deven_seed_characters tests passed.")
