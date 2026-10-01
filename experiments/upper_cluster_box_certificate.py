"""A finite test of the second half of conj:upper on a box of charges -- a research probe, in
the battery's conj:upper row.

The draft's conjecture says K_q(Q,Gamma) coincides with the "upper cluster algebra" of mutation-covariant Laurent polynomials
in Q_q(Gamma).  The first half (every RG_gamma Laurent in every chart) is checked in checks_rg.check_upper_sample.  This file
checks the reverse inclusion on a box B of charges, by two finite exact computations:

  1. every RG_gamma supported inside B stays a Laurent polynomial after one mutation at each node of the root chart
     (lattice_mutation.solve, which raises exactly when the conjugate is not Laurent);
  2. for the box points whose RG_gamma leaves B, the linear conditions for a combination of their monomials to stay Laurent
     after one mutation at each root node have full rank.  lattice_mutation's own criterion: a gamma_k-line with
     m = <gamma_k, beta> > 0 must be divisible by B_m(z) = sum_k [m,k]_q z^k, i.e. vanish at its m roots z = -q^s (found
     and checked exactly here).  The rank is taken at q = 2 in exact rationals; rank can only drop under specialisation, so
     full rank at q = 2 is full rank over Q(q).

Since RG_gamma = X_gamma + higher terms, the RG_gamma inside B and the monomials of the other box points form a basis of the
box.  So 1 and 2 give: the elements of the box that are Laurent in the root chart and its immediate neighbours are exactly
the span of the RG_gamma inside B -- hence the upper cluster algebra meets the box inside that span, and equals it there
given the first half.  No enumeration of charts is needed.

A chart with no finite spec (the Markov quiver) builds S to a finite depth; an F reaching that depth is truncated, and a
truncated F fails check 1 (measured: two Markov F's at depth 6, fine at depths 8 and 10).  So every F inside the box is
required to be unchanged when S is built two degrees deeper.

Controls: 1 = X_0 is Laurent in every chart, so adding it to the other box points must drop the rank; an RG_gamma with its
top-degree terms dropped must fail check 1.  (F's from the pentagon's S in the wrong order are NOT a control for check 1:
measured, they pass it -- the reversed product is still in the group, and the certificate holds for any triangular family
Laurent in the neighbours; that the family is the RG_gamma is check 1 on the chart's own F's, whose correctness is
sec:rg/F-unique's.)

Run:  PYTHONPATH=. python a probe in the source repository [--large]
"""
from __future__ import annotations

import itertools
import sys
import time
from fractions import Fraction

sys.path.insert(0, ".")
sys.path.insert(0, "implementations")
sys.path.insert(0, "battery")

from laurent_poly import LaurentPoly  # noqa: E402
from lattice_mutation import _build_from_dict, bezout_cofactor, solve  # noqa: E402
from mutation import _bpoly  # noqa: E402

_ROOTS: dict[int, list[int]] = {}


def bm_root_exponents(m: int) -> list[int]:
    """The s with B_m(-q^s) = 0 identically in q; B_m has exactly m of them."""
    if m not in _ROOTS:
        B = _bpoly(m)
        out = []
        for s in range(-2 * m - 2, 2 * m + 3):
            val = LaurentPoly.zero()
            for k, c in B.items():
                val = val + c * LaurentPoly({s * k: (-1) ** k})
            if val.is_zero():
                out.append(s)
        assert len(out) == m, (m, out)
        _ROOTS[m] = out
    return _ROOTS[m]


def rank(rows: list[list[Fraction]]) -> int:
    rows = [list(r) for r in rows]
    r = 0
    ncol = len(rows[0]) if rows else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c] / rows[r][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        r += 1
    return r


def neighbour_rank(lattice, nodes, points, q0=2) -> int:
    """Rank at q = q0 of the conditions for a combination of the monomials X_g (g in points) to stay Laurent after one
    mutation at each node."""
    rows = []
    x0 = Fraction(q0)
    for gk in nodes:
        u = bezout_cofactor(gk)
        lines: dict = {}
        for j, g in enumerate(points):
            k = sum(ui * gi for ui, gi in zip(u, g))
            beta = tuple(gi - k * gki for gi, gki in zip(g, gk))
            lines.setdefault(beta, []).append((j, k))
        for beta, members in lines.items():
            m = lattice.bracket(gk, beta)
            if m <= 0:
                continue
            for s in bm_root_exponents(m):
                row = [Fraction(0)] * len(points)
                for j, k in members:
                    row[j] += Fraction(-1) ** k * x0 ** (s * k)
                if any(row):
                    rows.append(row)
    return rank(rows) if rows and points else 0


def box_certificate(F, lattice, nodes, box):
    """F: label -> {charge: LaurentPoly}.  Returns (RG inside, other points, rank, RG inside that fail one mutation)."""
    bset = set(box)
    inside = [g for g in box if all(k in bset for k in F(g))]
    other = [g for g in box if g not in set(inside)]
    bad = []
    for g in inside:
        for gk in nodes:
            try:
                solve(_build_from_dict(lattice, F(g)), gk)
            except ValueError:
                bad.append((g, gk))
    return inside, other, neighbour_rank(lattice, nodes, other), bad


def _F_of(A):
    cache = {}

    def F(g):
        if g not in cache:
            cache[g] = {tuple(k): v for k, v in A.F(g).items() if not v.is_zero()}
        return cache[g]
    return F


def _box(nodes, R):
    n = len(nodes)
    return [tuple(sum(cc[i] * nodes[i][k] for i in range(n)) for k in range(len(nodes[0])))
            for cc in itertools.product(range(-R, R + 1), repeat=n)]


def main() -> int:
    import checks_rg as c
    large = "--large" in sys.argv
    ok_all = True
    # control 1: 1 = X_0 among the other points must drop the rank (pentagon, box [-1,1]^2)
    A = dict(c._wc_charts())["pentagon"]
    nodes = [tuple(v) for v in A.node_charges]
    box = _box(nodes, 1)
    inside, other, rk, bad = box_certificate(_F_of(A), A.lattice, nodes, box)
    rk0 = neighbour_rank(A.lattice, nodes, other + [(0, 0)])
    ctl1 = rk == len(other) and rk0 < len(other) + 1
    print(f"control: adding X_0 = 1 to the other points drops the rank ({rk0} < {len(other) + 1}): {ctl1}")
    # control 2: an RG_gamma with its top-degree terms dropped is no longer Laurent after one mutation (checked on every
    # multi-term RG_gamma inside the box; some must fail, or check 1 would pass anything triangular)
    F = _F_of(A)
    cut = fail = 0
    for g in inside:
        f = F(g)
        if len(f) < 2:
            continue
        degs = {k: sum(k) for k in f}
        top = max(degs.values())
        trunc = {k: v for k, v in f.items() if degs[k] < top}
        cut += 1
        for gk in nodes:
            try:
                solve(_build_from_dict(A.lattice, trunc), gk)
            except ValueError:
                fail += 1
                break
    ctl2 = fail > 0
    print(f"control: RG_gamma inside the box with the top-degree terms dropped fail one mutation on {fail} of {cut}: {ctl2}")
    ok_all &= ctl1 and ctl2
    charts = [(nm, A) for nm, A in c._wc_charts() if nm != "pentagon, 3-factor spec"] + [("Markov", None)]
    for nm, A in charts:
        for R in ((1, 2) if large else (1,)):
            t = time.time()
            if nm == "Markov":
                depth = 10 if R == 2 else 6
                A, A_deeper = c._chart_markov(depth), c._chart_markov(depth + 2)
            else:
                A_deeper = None
            nodes = [tuple(v) for v in A.node_charges]
            if R == 2 and len(nodes) > 3:
                continue
            box = _box(nodes, R)
            F = _F_of(A)
            inside, other, rk, bad = box_certificate(F, A.lattice, nodes, box)
            unstable = []
            if A_deeper is not None:
                Fd = _F_of(A_deeper)
                unstable = [g for g in inside if F(g) != Fd(g)]
            ok = rk == len(other) and not bad and not unstable
            ok_all &= ok
            print(f"{nm} [-{R},{R}]^{len(nodes)}: {len(box)} monomials, RG inside {len(inside)}, other points {len(other)}, "
                  f"rank at q=2 {rk}; RG inside failing one mutation {bad[:2]}; unstable under a deeper S {unstable[:2]} -> "
                  f"{'certified' if ok else 'NOT certified'} ({time.time() - t:.1f}s)", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
