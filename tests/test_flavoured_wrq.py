"""Testing flavoured AbeKAlgebras (U(N)+N_f) with the new WRQTorus machinery.

Findings validated here (the new substrate as the flavoured gauge backbone):

  1. **Matter adds no new poles.**  The matter rung `Z_m` (`quiver_urq_torus.
     _rung_levels`) is a *polynomial* per flavour level, and the flavoured gauge
     cocycle's denominators are *gauge roots* `1 − 𝖖^k v^{e_i−e_j}` only.  So the
     new `weyl_torus_ring` denominator structure (gauge roots `1 − 𝖖^k v^α`)
     already covers the flavoured U(N)+N_f torus — no ring redesign for flavour.

  2. **The new WRQTorus cocycle IS the flavoured gauge backbone.**  The pure-U(N)
     `wrq_torus.CC` reproduces the flavoured substrate's gauge cocycle
     `quiver_urq_torus._gauge_rtilde` exactly.

Consequence: the flavoured extension over WRQTorus is *additive* — a flavour-level
(μ) grading on residuals + the (polynomial) matter dressing `Z` + the matter Schur
factor `M` in the trace — built on the unchanged gauge foundation.  (N_f=1 has the
trivial flavour ring; the matter there enters mainly via the trace.)
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from root_datum import u_n
from wrq_torus import CC
from quiver_urq_torus import _rung_levels, _gauge_rtilde


def _eval_lp(lp, qv):
    return sum(Fraction(c) * Fraction(qv) ** e for e, c in lp._coeffs.items())


def _eval_TR(tr, qv, vv):
    num = Fraction(0)
    for lam, lp in tr._num._t.items():
        m = _eval_lp(lp, qv)
        for i, e in enumerate(lam):
            m *= Fraction(vv[i]) ** e
        num += m
    den = Fraction(1)
    for (a, k), mult in tr._den.items():
        f = Fraction(1)
        for i, e in enumerate(a):
            f *= Fraction(vv[i]) ** e
        den *= (Fraction(1) - Fraction(qv) ** k * f) ** mult
    return num / den


def _eval_VR(vr, qv, vv):
    num = Fraction(0)
    for e, lp in vr._num._terms.items():
        m = _eval_lp(lp, qv)
        for i, x in enumerate(e):
            m *= Fraction(vv[i]) ** x
        num += m
    den = Fraction(1)
    for (i, j, mm), mult in vr._den.items():
        den *= (Fraction(vv[i]) - Fraction(qv) ** mm * Fraction(vv[j])) ** mult
    return num / den


def test_matter_adds_no_new_poles():
    """The matter rung `Z_m` is polynomial (no denominators) for U(2)+N_f=1,2 —
    so the new ring's gauge-root denominators suffice for flavour."""
    for Nf in (1, 2):
        for m in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]:
            Z = _rung_levels(m, (2,), (Nf,))
            for lvl in Z.values():
                assert not lvl._den and not lvl._sq, \
                    f"matter rung not polynomial: N_f={Nf} m={m}"
    print("  PASS: matter rungs are polynomials (no new poles) — gauge roots suffice")


def test_gauge_cocycle_denominators_are_roots():
    """The flavoured gauge cocycle's denominators are gauge roots `(v_i−𝖖^m v_j)`
    — exactly the `1 − 𝖖^k v^α` form the new WeylTorusRing represents."""
    rt = _gauge_rtilde((1, 0), (0, 1), (2,))
    assert not rt._sq, "unexpected square factors"
    for (i, j, m) in rt._den:
        assert 0 <= i < 2 and 0 <= j < 2 and i != j, f"non-root denom {(i, j, m)}"
    print("  PASS: flavoured gauge-cocycle denominators are gauge roots")


def test_new_cocycle_is_flavoured_gauge_backbone():
    """The pure-U(N) WRQTorus cocycle reproduces the flavoured substrate's gauge
    cocycle exactly — the new machinery IS the flavoured gauge backbone."""
    d = u_n(2)
    pts = [(2, (3, 5)), (3, (2, 7)), (5, (2, 3))]
    for a, b in [((1, 0), (0, 1)), ((1, 0), (1, 0)), ((2, 0), (0, 1)),
                 ((1, 1), (1, 0)), ((2, 1), (0, 1))]:
        old = _gauge_rtilde(a, b, (2,))
        new = CC(d, a, b)
        for qv, vv in pts:
            assert _eval_VR(old, qv, vv) == _eval_TR(new, qv, vv), \
                f"gauge cocycle R̃_{a},{b} at {qv,vv}"
    print("  PASS: WRQTorus cocycle == flavoured gauge cocycle (U(2) backbone)")


def main():
    print("Flavoured AbeKAlgebra × WRQTorus tests")
    test_matter_adds_no_new_poles()
    test_gauge_cocycle_denominators_are_roots()
    test_new_cocycle_is_flavoured_gauge_backbone()
    print("All flavoured-WRQTorus tests passed.")


if __name__ == "__main__":
    main()
