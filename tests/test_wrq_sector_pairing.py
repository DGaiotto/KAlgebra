"""The per-sector-measure route to the Schur pairing on `WRQTorus`
(`sector_weight` / `pairing_residual` / `inner_by_sector`).

The pairing written as a sum over magnetic charge of contour integrals,

    I_{a,b}  =  Σ_m ∮ w_m(v) · f^a_m(arg) · f^b_m(arg),

with the per-sector measure `w_m = sector_weight(datum, m)` a first-class
object.  It is a genuinely **separate expression** from `inner`'s
`Tr(ρ(a)·b)`: `inner` builds `ρ(a)` and runs the full cocycle convolution,
this one never forms `ρ(a)` and touches only the shared magnetic support.
Their agreement is the total-charge-0 part of the convolution — a theorem,
certified here rather than assumed.

Four groups:

  1. `w_m` AS AN OBJECT: it exists for every sector tested, is independent of
     the states, and reproduces its own definition `T_{−m}(G̃_m)·R̃_{−m,m}`.
  2. AGREEMENT with the shipped `inner`, in BOTH `w_cutoff` modes, at pure
     SU(2) (supports to 9) and pure SU(3).
  3. THE NON-LINEARITY GUARD: `trace_residual` is non-linear in its argument
     when `w_cutoff=True`, so summing per-sector residues would be wrong;
     this pins that the shipped implementation sums residuals first.
  4. ORTHONORMALITY through the new route — `I_{a,a} = 1 + O(𝖖)` — so the
     route is checked against the contract, not only against its twin.

⚠ A zero from either route can mean "cut", not "orthogonal" — see the two
warnings in `wrq_torus.trace_residual`.  Comparisons here are made in raw
mode (`w_cutoff=False`) as well as cut mode for exactly that reason.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "implementations")

import root_datum as rd
from wrq_torus import sector_weight, trace_residual, CC, _rho_Gtilde

FAILURES = []


def check(name, ok):
    print(("  PASS  " if ok else "  FAIL  ") + name)
    if not ok:
        FAILURES.append(name)


def _charts(A, labels):
    out = {}
    for lab in labels:
        try:
            out[lab] = A.chart(lab)
        except Exception as ex:                                  # pragma: no cover
            print(f"    SKIP chart{lab}: {type(ex).__name__}: {str(ex)[:70]}")
    return out


def leg_weight_object():
    print("\n[1] w_m as an object")
    datum = rd.su_2()
    for m in [(0,), (1,), (2,), (3,)]:
        w = sector_weight(datum, m)
        neg = tuple(-x for x in m)
        expect = (_rho_Gtilde(datum, m, False).q_shift(neg)
                  * CC(datum, neg, m)).simplify()
        check(f"su_2  w_{m} == T_(-m)(G~_m)·R~_(-m,m)",
              (w - expect).simplify().is_zero())
    d3 = rd.su_n(3)
    for m in [(0, 0), (1, 0), (1, 1)]:
        check(f"su_3  w_{m} builds",
              not sector_weight(d3, m).is_zero() or m == (0, 0))


def leg_agreement():
    print("\n[2] agreement with the shipped `inner` (both w_cutoff modes)")
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    t = time.time()
    A2 = PureGAbeKAlgebra(rd.su_2())
    labs2 = [((0,), (0,)), ((1,), (0,)), ((2,), (0,)),
             ((3,), (0,)), ((4,), (0,)), ((2,), (2,))]
    ch = _charts(A2, labs2)
    ok = tot = 0
    for a in ch:
        for b in ch:
            for wc in (False, True):
                tot += 1
                lhs = ch[a].inner_by_sector(ch[b], K=10, w_cutoff=wc)
                rhs = ch[a].inner(ch[b], K=10, w_cutoff=wc)
                ok += (lhs - rhs).is_zero()
    check(f"pure SU(2): inner_by_sector == inner  ({ok}/{tot}, K=10)", ok == tot)
    print(f"    [{time.time()-t:.1f}s]")

    t = time.time()
    A3 = PureGAbeKAlgebra(rd.su_n(3))
    labs3 = [((0, 0), (0, 0)), ((0, 0), (1, 0)),
             ((1, 0), (0, 0)), ((1, 1), (0, 0))]
    ch3 = _charts(A3, labs3)
    ok = tot = 0
    for a in ch3:
        for b in ch3:
            for wc in (False, True):
                tot += 1
                lhs = ch3[a].inner_by_sector(ch3[b], K=8, w_cutoff=wc)
                rhs = ch3[a].inner(ch3[b], K=8, w_cutoff=wc)
                ok += (lhs - rhs).is_zero()
    check(f"pure SU(3): inner_by_sector == inner  ({ok}/{tot}, K=8)", ok == tot)
    print(f"    [{time.time()-t:.1f}s]")
    return A2, ch


def leg_nonlinearity_guard(A2, ch):
    print("\n[3] the non-linearity guard (residuals summed BEFORE the residue)")
    datum = rd.su_2()
    a = ((3,), (0,))
    b = ((3,), (0,))
    if a not in ch or b not in ch:                               # pragma: no cover
        print("    SKIP (charts unavailable)")
        return
    x, y = ch[a], ch[b]
    whole = trace_residual(datum, x.pairing_residual(y), 10, w_cutoff=True)
    # the WRONG way: per-sector residues, each cut independently, then summed
    per = None
    for m, fa in x.residuals().items():
        fb = y.residual(m)
        if fb.is_zero():
            continue
        neg = tuple(-t for t in m)
        g = (sector_weight(datum, m) * fa.vinv().q_shift(neg)
             * fb.q_shift(neg)).simplify()
        piece = trace_residual(datum, g, 10, w_cutoff=True)
        per = piece if per is None else per + piece
    check("summing cut per-sector residues differs from the correct order "
          "(so the guard is load-bearing, not decorative)",
          per is not None and not (whole - per).is_zero())
    check("the correct order agrees with `inner`",
          (whole - x.inner(y, K=10, w_cutoff=True)).is_zero())


def leg_orthonormality(ch):
    print("\n[4] orthonormality through the new route: I_{a,a} = 1 + O(𝖖)")
    for lab, c in ch.items():
        val = c.inner_by_sector(c, K=10, w_cutoff=True)
        check(f"I_{lab},{lab}[𝖖⁰] == 1", val._coeffs.get(0, 0) == 1)


if __name__ == "__main__":
    t0 = time.time()
    leg_weight_object()
    A2, ch = leg_agreement()
    leg_nonlinearity_guard(A2, ch)
    leg_orthonormality(ch)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print(f"All WRQ sector-pairing tests passed.   [{time.time()-t0:.1f}s]")
