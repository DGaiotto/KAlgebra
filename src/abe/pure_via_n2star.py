"""`N2StarBuilder` — finding pure U(N) / SU(N) canonicals `L_{m,e}` by
solving in N=2* and RG-flowing down (Plan 36 T13; user charter 2026-07-02:
"a fully fledged method for pure U(N) and SU(N) gauge groups: solve in
N=2^* and RG flow … for finding L_{m,e}").

Method
------
Work in N=2* U(N) — the self-link shape
`QuiverURQTorus(ranks=(N,), links=((0,0),))` — where the canonical basis
is SL(2,Z)-organized and constructible:

  * a label is the multiset of per-slot charge points `(m_i, e_i) ∈ Z²`;
  * **primitive single points** `E₁^{(p,q)}` come from the d=1 ladder
    `[E_u, E_v] = ±(q−q⁻¹)·E_{u+v}` along Farey (mediant) parents,
    seeded by `H_{±1}`, `w_{±1}` and the dressed minuscules (|p|,|q| ≤ 1);
    each rung is an exact Laurent division certified by W1+W2;
  * **aligned labels** (all points on one primitive slope `(p,q)` with
    multiplicities `t_i` ≥ 1, partition λ) are classical Schur polynomials
    in the slope's column dyons with the adjoint μ-weights:
        L_{λ·(p,q)} = s_λ(e_k ↦ μ^{C(k,2)}·E_k^{(p,q)})
    (dual Jacobi–Trudi expansion; structure constants classical LR × μ-powers);
  * **minuscule-magnetic labels** (all `m_i ∈ {0,s}`, s = ±1) never bubble:
    the bare joint Weyl-orbit closed form is the canonical (any dressing);
  * everything else honest-fails (`NotImplementedError`) — the tracked
    frontier is the middle columns `E_k^{(p,q)}`, 1 < k < N, |p| ≥ 2, and
    generic multi-slope labels (build by per-slope products + peel; the
    coupled same-magnetic fibers split via the dyonic isolation
    `X₂ = E₁^{(1,1)}·H₁ = μ·A + q⁻¹·E^{(2,1)}` — see
    `experiments/n2star_fiber_split.py`).

The **RG flow** to the pure theory is the large-mass μ-level-0 slice
(`slice₀`, an algebra homomorphism — `W[0] = 1`, levels non-negative on
natural-frame charts): `pure_chart(points) = slice₀(build(points))`.
Pure structure constants are the μ⁰-part of the N=2* ones.  The
homomorphism is **certified** (`tests/test_n2star_rg_flow.py`,
`experiments/n2star_rg_homomorphism.py`): `slice₀(Â·B̂) =
slice₀(Â)·slice₀(B̂)` and the flowed product lands on the certified pure
engine with `PureUNKAlgebra.multiply`'s structure constants, at N=2/3.

**SU(N)**: pure SU(N) consumes exactly the **trace-zero** pure U(N)
charts (`PureSUNKAlgebra`'s single oracle input — restrict + det-collapse
on `SUNRQTorus`); `sun_chart(points)` returns that oracle input for
`Σ m_i = 0` labels.

Every output is acceptance-checked (W1 + W2) and the tests certify the
slices against the certified `PureUNKAlgebra` registry via its
content-based `decompose` (single canonical, coefficient 1).

Validation: `tests/test_pure_via_n2star.py`.  Probes/derivations:
`experiments/n2star_{u2,u3,u4}_probe.py`, `n2star_slope_constructor.py`,
`n2star_gap_flow.py`, `n2star_fiber_split.py`.
"""
from __future__ import annotations

import os
import sys
from itertools import permutations
from math import gcd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from laurent_poly import LaurentPoly
from abelianized_torus import VRational, VLaurent
from quiver_urq_torus import QuiverURQTorus

__all__ = ["N2StarBuilder"]

_QMQ = LaurentPoly({1: 1, -1: -1})          # q − q⁻¹


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------
def _lp_div(a: LaurentPoly, b: LaurentPoly):
    """Exact Laurent division `a / b`; ``None`` if not divisible.
    Fast-fail: an exact quotient's top exponent is `max(a) − max(b)`, so
    any quotient term beyond that means non-divisibility (without the
    bound a failing division grinds out ever-higher remainder tails)."""
    if a.is_zero():
        return LaurentPoly.zero()
    rem = dict(a._coeffs)
    bexps = sorted(b._coeffs)
    b0e = bexps[0]
    b0c = b._coeffs[b0e]
    qtop = max(a._coeffs) - bexps[-1]
    out: dict = {}
    while rem:
        e0 = min(rem)
        c0 = rem[e0]
        if c0 % b0c:
            return None
        qe, qc = e0 - b0e, c0 // b0c
        if qe > qtop:
            return None
        out[qe] = out.get(qe, 0) + qc
        for be, bc in b._coeffs.items():
            k = qe + be
            v = rem.get(k, 0) - qc * bc
            if v:
                rem[k] = v
            elif k in rem:
                del rem[k]
    return LaurentPoly(out)


def _schur_in_e(lam):
    """Dual Jacobi–Trudi: `s_λ = det(e_{λ'_i + j − i})` expanded into
    e-monomials `{sorted-tuple-of-column-sizes: int}` (symbolic; `e_0 = 1`,
    `e_{<0} = 0`; the `e_{>N}` cut is applied by the caller)."""
    lam = tuple(x for x in lam if x)
    if not lam:
        return {(): 1}
    lam1 = lam[0]
    conj = tuple(sum(1 for x in lam if x > j) for j in range(lam1))
    n = len(conj)
    out: dict = {}
    for sig in permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if sig[i] > sig[j])
        cols = []
        ok = True
        for i in range(n):
            k = conj[i] + sig[i] - i
            if k < 0:
                ok = False
                break
            if k:
                cols.append(k)
        if not ok:
            continue
        key = tuple(sorted(cols, reverse=True))
        out[key] = out.get(key, 0) + (-1) ** inv
    return {k: c for k, c in out.items() if c}


def _primitive(p, q):
    g = gcd(abs(p), abs(q))
    return (p // g, q // g), g


def _farey_parents(p, q):
    """Mediant parents `u + v = (p,q)`, `|det(u,v)| = 1`, both shorter —
    the d=1 ladder recursion step for a primitive `(p,q)`."""
    # extended Euclid: c*q - d*p = ±1 with (c,d) minimal
    a, b = abs(p), abs(q)
    # find x,y: p*y - q*x = 1  (p,q primitive)
    def egcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x, y = egcd(b, a % b)
        return g, y, x - (a // b) * y
    g, x, y = egcd(p, q)          # p*x + q*y = ±1 (g = ±1)
    if g == -1:
        x, y = -x, -y
    # u = (-y, x) satisfies det((p,q),(-y,x)) = p*x + q*y = 1
    u = (-y, x)
    best = None
    for t in range(-3, 4):
        cand = (u[0] + t * p, u[1] + t * q)
        v = (p - cand[0], q - cand[1])
        norm = abs(cand[0]) + abs(cand[1]) + abs(v[0]) + abs(v[1])
        if abs(cand[0]) + abs(cand[1]) >= abs(p) + abs(q):
            continue
        if abs(v[0]) + abs(v[1]) >= abs(p) + abs(q):
            continue
        if best is None or norm < best[0]:
            best = (norm, cand, v)
    if best is None:
        raise NotImplementedError(f"no shrinking Farey parents for {(p, q)}")
    return best[1], best[2]


class N2StarBuilder:
    """The `L_{m,e}` finder for pure U(N)/SU(N): build the canonical in
    N=2* U(N) (self-link torus), RG-flow by the μ-level-0 slice.

    A label is `points` — a length-N tuple of per-slot `(m_i, e_i)` charge
    pairs (order irrelevant; the joint Weyl group acts by permuting slots).
    """

    def __init__(self, N: int, mu_cap: int | None = None):
        """`mu_cap`: the μ-truncated **flow mode** — all torus products are
        truncated above μ-level `mu_cap`.  Exact for the level-0 slice
        (products only raise levels; the triangular divisions depend
        downward) provided the cap exceeds the read/division depth
        (`N + 4` is comfortable); the W1+W2 acceptance on capped charts is
        weaker, so flow-mode outputs should be certified via the slice
        (registry `decompose`) or an uncapped rebuild.  `None` = exact."""
        self.N = int(N)
        self.mu_cap = mu_cap
        self._shape = ((self.N,), (0,), ((0, 0),))
        self._E1: dict = {}
        self._built: dict = {}
        self._stack: set = set()

    def _cap(self, x: QuiverURQTorus) -> QuiverURQTorus:
        if self.mu_cap is None:
            return x
        return QuiverURQTorus(
            {m: {k: vr for k, vr in row.items() if k[0] <= self.mu_cap}
             for m, row in x.residuals().items()}, *self._shape)

    def _mul(self, x: QuiverURQTorus, y: QuiverURQTorus) -> QuiverURQTorus:
        return self._cap(x * y)

    # ----- torus element helpers --------------------------------------
    def _g(self, m, e=None):
        e = e if e is not None else (0,) * self.N
        return QuiverURQTorus.minuscule(0, m, e, *self._shape)

    def _w(self, e):
        return QuiverURQTorus.wilson(0, e, *self._shape)

    def _scaled(self, x, c, d=0):
        return QuiverURQTorus(
            {m: {(k[0] + d,): vr * c for k, vr in row.items()}
             for m, row in x.residuals().items()}, *self._shape)

    def _comm(self, x, y):
        return self._mul(x, y) + self._scaled(self._mul(y, x),
                                              LaurentPoly({0: -1}))

    def _div_scalar(self, x, c):
        out = {}
        for m, row in x.residuals().items():
            dst = {}
            for k, vr in row.items():
                num = {}
                for ve, lp in vr.num._terms.items():
                    qq = _lp_div(lp, c)
                    if qq is None:
                        return None
                    num[ve] = qq
                dst[k] = VRational(VLaurent(num, n=self.N), dict(vr.den),
                                   n=self.N)
            out[m] = dst
        return QuiverURQTorus(out, *self._shape)

    # ----- generators ---------------------------------------------------
    def E1(self, p: int, q: int) -> QuiverURQTorus:
        """The primitive single-point dyon `E₁^{(p,q)}` (gcd(p,q) = 1):
        base charges directly, deeper ones by the d=1 Farey ladder."""
        key = (int(p), int(q))
        if gcd(abs(p), abs(q)) != 1:
            raise ValueError(f"E1{key}: not primitive")
        hit = self._E1.get(key)
        if hit is not None:
            return hit
        p, q = key
        pad = (0,) * (self.N - 1)
        if p == 0:                       # pure electric: Wilson box
            out = self._w((q,) + pad)
        elif abs(p) == 1:                # minuscule magnetic: no bubbling
            out = self._g((p,) + pad, (q,) + pad)
        else:
            u, v = _farey_parents(p, q)
            Eu, Ev = self.E1(*u), self.E1(*v)
            c = self._div_scalar(self._comm(Eu, Ev), _QMQ)
            if c is None:
                raise NotImplementedError(
                    f"E1{key}: ladder commutator not divisible by (q−q⁻¹)")
            out = None
            for sgn in (1, -1):
                cand = self._scaled(c, LaurentPoly({0: sgn}))
                if cand.well_formed():
                    out = cand
                    break
            if out is None:
                raise NotImplementedError(
                    f"E1{key}: ladder output not a single canonical "
                    f"(parents {u}, {v}) — d=1 tail; needs the peel")
        wf = out.well_formed()
        if not wf:
            raise NotImplementedError(f"E1{key}: not well-formed")
        self._E1[key] = out
        return out

    def E_col(self, k: int, p: int, q: int) -> QuiverURQTorus:
        """The k-column slope dyon `E_k^{(p,q)}` — the label with `k` slots
        at the point `(p,q)`.  Direct for k=1, minuscule |p| ≤ 1, and the
        central k=N; the middle columns at |p| ≥ 2 are the v1 frontier."""
        if k == 1:
            return self.E1(p, q)
        col = lambda t: (t,) * k + (0,) * (self.N - k)
        if abs(p) <= 1:
            return self._g(col(p), col(q)) if p else self._w(col(q))
        if k == self.N:                  # central magnetic: single atom
            return self._g(col(p), col(q))
        if k == 2:
            return self._col2(p, q)
        raise NotImplementedError(
            f"E_col(k={k}, ({p},{q})): middle columns k ≥ 3 at |p| ≥ 2 — "
            f"the tracked frontier (higher d=k relations)")

    def _col2(self, p: int, q: int) -> QuiverURQTorus:
        """`E_2^{(p,q)}` (and the single-slot double point `T = 2·(p,q)`)
        for a deep slope `|p| ≥ 2`: the in-span 2×2 inversion of

            A = E₁²                          = T + μ^a·E₂
            B = [E₁^{(p,q)−w}, E₁^{(p,q)+w}] = c₁·T + c₂·E₂   (d = 2)

        with symmetry-predicted unit frames (`a = 1`; `c₁ = ±(q²−q⁻²)μ^{s₁}`,
        `c₂ = ±(q−q⁻¹)(1+μ²)μ^{s₂}` — the U(2)-verified d=2 shape, small
        anomaly scan), solved by the exact triangular division
        `T = (B − c₂μ^{−a}A)/(c₁ − c₂μ^{−a})`; W1+W2 acceptance on BOTH
        `T` and `E₂` gates every frame (in-span: A, B are products /
        commutators of built generators — the D6 criterion (i))."""
        key = ("col2", p, q)
        hit = self._built.get(key)
        if hit is not None:
            return hit
        E1 = self.E1(p, q)
        A = self._mul(E1, E1)
        # a det-2 primitive pair u + v = 2·(p,q):  w with p·w_y − q·w_x = ±1
        def egcd(a, b):
            if b == 0:
                return a, 1, 0
            g, x, y = egcd(b, a % b)
            return g, y, x - (a // b) * y
        g, x0, y0 = egcd(p, q)
        if g == -1:
            x0, y0 = -x0, -y0
        B = None
        for t in range(-3, 4):
            w = (-y0 + t * p, x0 + t * q)
            u = (p - w[0], q - w[1])
            v = (p + w[0], q + w[1])
            if u == (0, 0) or v == (0, 0):
                continue
            if gcd(abs(u[0]), abs(u[1])) != 1 or gcd(abs(v[0]), abs(v[1])) != 1:
                continue
            try:
                B = self._comm(self.E1(u[0], u[1]), self.E1(v[0], v[1]))
                break
            except NotImplementedError:
                continue
        if B is None:
            raise NotImplementedError(
                f"E_col(2, ({p},{q})): no buildable det-2 pair found")
        # read-based inversion first (no grid): c₁ from B at T's leading
        # monomial; σ from (B − c₁A) at E₂'s leading; α from A likewise.
        got = self._col2_by_reads(p, q, A, B)
        if got is not None:
            E2, T = got
            self._built[key] = E2
            self._built[("tower2", p, q)] = T
            return E2
        qmq = LaurentPoly({1: 1, -1: -1})
        q2 = LaurentPoly({2: 1, -2: -1})
        a = 1
        for s1 in range(0, 3):
            for g1 in (1, -1):
                for s2 in range(-1, 4):
                    for g2 in (1, -1):
                        # c2·mu^{-a}·A with c2 = g2 (q−1/q)(1+mu^2) mu^{s2}
                        corr = self._scaled(A, qmq * LaurentPoly({0: -g2}),
                                            s2 - a) \
                            + self._scaled(A, qmq * LaurentPoly({0: -g2}),
                                           s2 - a + 2)
                        num = B + corr
                        # scalar c1 − c2 mu^{−a}: levels {s1: ±q2,
                        # s2−a: ∓qmq, s2−a+2: ∓qmq}
                        den = {}
                        den[s1] = den.get(s1, LaurentPoly.zero()) + \
                            q2 * LaurentPoly({0: g1})
                        for lv in (s2 - a, s2 - a + 2):
                            den[lv] = den.get(lv, LaurentPoly.zero()) + \
                                qmq * LaurentPoly({0: -g2})
                        den = {lv: c for lv, c in den.items()
                               if not c.is_zero()}
                        T = self._div_by_mu_scalar(num, den)
                        if T is None or not T.well_formed():
                            continue
                        E2 = self._scaled(
                            A + self._scaled(T, LaurentPoly({0: -1})),
                            LaurentPoly({0: 1}), -a)
                        if not E2.well_formed():
                            continue
                        self._built[key] = E2
                        self._built[("tower2", p, q)] = T
                        return E2
        raise NotImplementedError(
            f"E_col(2, ({p},{q})): no accepted frame in the d=2 inversion")

    def _read_coeff(self, x: QuiverURQTorus, atom, ve):
        """The μ-graded scalar multiplying the monomial `v^{ve}` at `atom`
        in `x` — den-free rows only (rational tail content is excluded by
        construction).  Returns `{level: LaurentPoly}` (possibly empty)."""
        out = {}
        for k, vr in x.residuals().get(tuple(atom), {}).items():
            vr = vr.simplify()
            if vr.den or getattr(vr, "_sq", None):
                continue
            lp = vr.num._terms.get(tuple(ve))
            if lp is not None and not lp.is_zero():
                out[k[0]] = lp
        return out

    def _sub_scaled_graded(self, x, y, ct: dict):
        """`x − (Σ_d ct[d]·μ^d)·y`."""
        out = x
        for d, c in ct.items():
            out = out + self._scaled(y, c * LaurentPoly({0: -1}), d)
        return out

    def _col2_by_reads(self, p, q, A, B):
        """The read-based d=2 inversion (no frame grid): all three scalars
        are read off leading monomials; W1+W2 acceptance on both outputs;
        ``None`` on any failure (the caller falls back to the grid)."""
        pad = (0,) * (self.N - 2)
        atomT = (2 * p, 0) + pad          # T's dominant atom
        veT = (2 * q, 0) + pad
        atomE = (p, p) + pad              # E₂'s dominant atom
        veE = (q, q) + pad
        c1 = self._read_coeff(B, atomT, veT)
        if not c1:
            return None
        D = self._sub_scaled_graded(B, A, c1)     # = (c₂ − c₁μ^a)·E₂
        sig = self._read_coeff(D, atomE, veE)
        if not sig:
            return None
        E2 = self._div_by_mu_scalar(D, sig)
        if E2 is None or not E2.well_formed():
            return None
        alpha = self._read_coeff(A, atomE, veE)   # = μ^a (E₂'s coeff in A)
        T = self._sub_scaled_graded(A, E2, alpha)
        if not T.well_formed():
            return None
        return E2, T

    def _div_by_mu_scalar(self, x: QuiverURQTorus, ct: dict):
        """Exact division of an element by a μ-graded scalar
        `Σ_d ct[d]·μ^d` (triangular over levels; ``None`` on failure)."""
        if not ct:
            return None
        d0 = min(ct)
        c0 = ct[d0]
        res = {m: dict(row) for m, row in x.residuals().items()}
        out: dict = {}
        levels = sorted({k[0] for row in res.values() for k in row})
        if not levels:
            return QuiverURQTorus.zero(*self._shape)
        for k in range(levels[0] - d0, levels[-1] - d0 + 1):
            acc: dict = {}
            for m, row in res.items():
                vr = row.get((k + d0,))
                if vr is not None:
                    acc[m] = vr
            for d, cd in ct.items():
                if d == d0:
                    continue
                kk = k + d0 - d
                for m, row in out.items():
                    vr = row.get((kk,))
                    if vr is not None:
                        sub = (vr * VRational.from_scalar(
                            cd * LaurentPoly({0: -1}), n=self.N)).simplify()
                        acc[m] = sub if m not in acc else (
                            acc[m] + sub).simplify()
            for m, vr in acc.items():
                vr = vr.simplify()
                if vr.is_zero():
                    continue
                num = {}
                for ve, lp in vr.num._terms.items():
                    qq = _lp_div(lp, c0)
                    if qq is None:
                        return None
                    num[ve] = qq
                out.setdefault(m, {})[(k,)] = VRational(
                    VLaurent(num, n=self.N), dict(vr.den), n=self.N)
        return QuiverURQTorus(out, *self._shape)

    def aligned(self, slope, partition) -> QuiverURQTorus:
        """`L_{λ·(p,q)} = s_λ(e_k ↦ μ^{C(k,2)}·E_k^{(p,q)})` — classical
        Schur in the slope's column dyons with the adjoint μ-weights."""
        p, q = slope
        lam = tuple(sorted((int(t) for t in partition if t), reverse=True))
        if len(lam) > self.N:
            raise ValueError("aligned: more parts than slots")
        surviving = [(cols, coeff) for cols, coeff in _schur_in_e(lam).items()
                     if not any(k > self.N for k in cols)]
        if not surviving:
            raise NotImplementedError(f"aligned({slope}, {lam}): empty e-expansion")
        wmin = min(sum(k * (k - 1) // 2 for k in cols) for cols, _ in surviving)
        out = None
        for cols, coeff in surviving:
            term = None
            for k in cols:
                Ek = self.E_col(k, p, q)
                term = Ek if term is None else self._mul(term, Ek)
            if term is None:
                term = QuiverURQTorus.one(*self._shape)
            w = sum(k * (k - 1) // 2 for k in cols) - wmin
            term = self._scaled(term, LaurentPoly({0: coeff}), w)
            out = term if out is None else out + term
        if out is None or not out.well_formed():
            raise NotImplementedError(
                f"aligned({slope}, {lam}): output not accepted (W1+W2)")
        return out

    # ----- the dispatcher ------------------------------------------------
    def build(self, points) -> QuiverURQTorus:
        """The N=2* canonical `L_{points}`; honest-fails off the covered
        classes.  `points` = length-N iterable of `(m_i, e_i)`."""
        pts = tuple(sorted((int(m), int(e)) for m, e in points))
        if len(pts) != self.N:
            raise ValueError(f"build: need {self.N} points")
        key = pts
        hit = self._built.get(key)
        if hit is not None:
            return hit
        if key in self._stack:
            raise NotImplementedError(
                f"build{pts}: cycle (frame-shifted read of a label already "
                f"being built — the Witten-frame canonicalization of peel "
                f"reads is the tracked fix)")
        self._stack.add(key)
        try:
            out = self._build_inner(pts)
        finally:
            self._stack.discard(key)
        self._built[key] = out
        return out

    def _build_inner(self, pts) -> QuiverURQTorus:
        nz = [pt for pt in pts if pt != (0, 0)]
        m_vec = tuple(m for m, _ in pts)
        e_vec = tuple(e for _, e in pts)
        mset = set(m_vec)
        out = None
        if not nz:
            out = QuiverURQTorus.one(*self._shape)
        elif mset == {0}:
            out = self._w(e_vec)                       # pure Wilson
        elif mset <= {0, 1} or mset <= {0, -1}:
            out = self._g(m_vec, e_vec)                # minuscule magnetic
        elif len(nz) == 2 and sorted(m for m, _ in nz) == [-1, 1]:
            out = self._adjoint_pair(nz)               # SU(N) minimal monopole
        else:
            # aligned: all points positive multiples of one primitive slope
            dirs = set()
            ts = []
            for m, e in nz:
                (pp, qq), t = _primitive(m, e)
                dirs.add((pp, qq))
                ts.append(t)
            if len(dirs) == 1:
                out = self.aligned(next(iter(dirs)), ts)
            else:
                out = self._multi_slope(pts, nz)
        if not out.well_formed():
            raise NotImplementedError(f"build{pts}: output not accepted")
        return out

    # ----- the generic multi-slope peel -----------------------------------
    def _extreme_reads(self, x: QuiverURQTorus):
        """Global q-extreme reads of an arbitrary element: returns
        `(qext, {(joint_label, level): int})` via the substrate's joint
        recognizer on the extreme slice, per μ-level."""
        from quiver_urq_torus import _joint_recognize_q_extreme
        vals = {}
        for m, row in x.residuals().items():
            for k, f in row.items():
                f = f.simplify()
                if f.num.is_zero():
                    continue
                vnum = min(min(lp._coeffs) for lp in f.num._terms.values())
                vden = sum(M * mult for (i, j, M), mult in f.den.items()
                           if M < 0)
                vals[(m, k)] = (vnum - vden, f)
        if not vals:
            return None, {}
        qext = min(v for v, _ in vals.values())
        reads: dict = {}
        levels = {k for (_, k), (v, _) in vals.items() if v == qext}
        for k in levels:
            slice_f = {m: f for (m, kk), (v, f) in vals.items()
                       if kk == k and v == qext}
            for lab, c in _joint_recognize_q_extreme(
                    slice_f, (self.N,)).items():
                reads[(lab, k)] = reads.get((lab, k), 0) + c
        return qext, reads

    def _multi_slope(self, pts, nz) -> QuiverURQTorus:
        """Generic mixed-slope label: the product of the per-slope aligned
        canonicals, peeled against recursively-built lower canonicals by
        the extreme read (the ABSOLUTE-RULE product-and-peel; every
        subtraction is a built, acceptance-checked chart).  Honest-fails
        when a needed lower canonical is off the covered classes."""
        groups: dict = {}
        for m, e in nz:
            (p, q), t = _primitive(m, e)
            groups.setdefault((p, q), []).append(t)
        P = None
        for slope in sorted(groups):
            f = self.aligned(slope, groups[slope])
            P = f if P is None else self._mul(P, f)
        target = tuple(sorted(nz))
        tkey = tuple(sorted((m for m, _ in pts), reverse=True))
        r = P
        for _ in range(128):
            # dominance-first: magnetic-merge content lives on orbits
            # STRICTLY more dominant than the target's (target tails live
            # strictly below) — peel the most-dominant foreign orbit
            # before reading at the global q-extreme.
            higher = sorted(
                (tuple(sorted(m, reverse=True)) for m in r.residuals()
                 if tuple(sorted(m, reverse=True)) > tkey),
                reverse=True)
            if higher:
                hkey = higher[0]
                sub = QuiverURQTorus(
                    {m: row for m, row in r.residuals().items()
                     if tuple(sorted(m, reverse=True)) == hkey},
                    *self._shape)
                qext, reads = self._extreme_reads(sub)
            else:
                qext, reads = self._extreme_reads(r)
            if qext is None:
                raise NotImplementedError(
                    f"multi-slope {pts}: peel emptied the product")
            todo = []
            only_target = True
            for (lab, k), c in reads.items():
                lab_pts = tuple(sorted(
                    (m, e) for m, e in zip(lab[0][0], lab[0][1])
                    if (m, e) != (0, 0)))
                if lab_pts == target:
                    continue
                only_target = False
                todo.append((lab, k, c))
            if only_target:
                # r = c_t(q,μ)·L_target (+ magnetic-merge content hiding
                # above the extreme): extract c_t from the leading monomial
                # of the dominant atom, divide exactly (triangular over
                # μ-levels); on acceptance failure peel merge-point
                # canonicals (the X₂ = μA + q⁻¹E^{(p+p',q+q')} pattern).
                out = self._normalize_target(r, pts)
                if out is not None and out.well_formed():
                    return out
                # peel magnetic-merge descendants (unit merges — the
                # X₂ = μA + q⁻¹E^{(p+p',q+q')} pattern), acceptance-gated
                descs = []
                for dpts in self._merge_descendants(nz):
                    try:
                        descs.append(self.build(tuple(sorted(
                            list(dpts) + [(0, 0)] * (self.N - len(dpts))))))
                    except (NotImplementedError, ValueError):
                        continue
                frames = [(a, b, s) for a in range(0, self.N + 1)
                          for b in range(-3, 4) for s in (1, -1)]
                for D in descs:
                    for a, b, s in frames:
                        r2 = r + self._scaled(D, LaurentPoly({b: -s}), a)
                        out = self._normalize_target(r2, pts)
                        if out is not None and out.well_formed():
                            return out
                for i in range(len(descs)):
                    for j in range(i, len(descs)):
                        for a, b, s in frames:
                            r2 = r + self._scaled(
                                descs[i], LaurentPoly({b: -s}), a)
                            for a2, b2, s2 in frames:
                                r3 = r2 + self._scaled(
                                    descs[j], LaurentPoly({b2: -s2}), a2)
                                out = self._normalize_target(r3, pts)
                                if out is not None and out.well_formed():
                                    return out
                raise NotImplementedError(
                    f"multi-slope {pts}: normalized remainder not "
                    f"accepted (W1+W2)")
            for lab, k, c in todo:
                sub_pts = tuple(sorted(zip(lab[0][0], lab[0][1])))
                ch = self.build(sub_pts)
                v0, r0 = self._extreme_reads(ch)
                ((lab0, k0), c0), = r0.items()
                if abs(c0) != 1:
                    raise NotImplementedError(
                        f"multi-slope {pts}: sub-chart {sub_pts} extreme "
                        f"coefficient {c0}")
                r = r + self._scaled(
                    ch, LaurentPoly({qext - v0: -c * c0}), k[0] - k0[0])
        raise NotImplementedError(f"multi-slope {pts}: peel did not converge")

    def _merge_descendants(self, nz):
        """Unit-merge closure of a point multiset: all point-multisets
        reachable by merging two charge units — the candidate lower labels
        of a multi-slope product (magnetic merges)."""
        from itertools import combinations
        units = []
        for m, e in nz:
            (p, q), t = _primitive(m, e)
            units += [(p, q)] * t
        # seed with both the unit decomposition and the target point
        # multiset — same-slope unit merges must not regenerate the target
        seen = {tuple(sorted(units)), tuple(sorted(nz))}
        out = []
        stack = [units]
        while stack:
            us = stack.pop()
            for i, j in combinations(range(len(us)), 2):
                merged = [u for k, u in enumerate(us) if k not in (i, j)]
                s = (us[i][0] + us[j][0], us[i][1] + us[j][1])
                if s != (0, 0):
                    merged.append(s)
                key = tuple(sorted(merged))
                if key in seen or len(merged) > self.N:
                    continue
                seen.add(key)
                out.append(key)
                stack.append(merged)
        return out

    def _normalize_target(self, r: QuiverURQTorus, pts):
        """Divide `r = c_t(q,μ)·L_target` by the extracted coefficient.
        `c_t` is read off the leading v-monomial of the dominant atom
        (den-free there, else give up); the division is exact and
        triangular over μ-levels; the caller acceptance-checks."""
        dom = [m for m in r.residuals()
               if m == tuple(sorted(m, reverse=True))]
        if not dom:
            return None
        best_atom = max(dom)
        row = r.residuals()[best_atom]
        # the leading v-monomial: global q-extreme among den-free content
        lead = None
        for k, vr in row.items():
            vr = vr.simplify()
            if vr.den or getattr(vr, "_sq", None):
                continue
            for ve, lp in vr.num._terms.items():
                v = min(lp._coeffs)
                if lead is None or v < lead[0]:
                    lead = (v, ve)
        if lead is None:
            return None
        _, ve = lead
        ct = {}
        for k, vr in row.items():
            vr = vr.simplify()
            if vr.den:
                continue
            lp = vr.num._terms.get(ve)
            if lp is not None and not lp.is_zero():
                ct[k[0]] = lp
        if not ct:
            return None
        d0 = min(ct)
        c0 = ct[d0]
        # triangular division over levels: L_k = (r_{k+d0} − Σ_{d>d0} c_d·L_{k+d0−d})/c0
        res = {m: dict(row) for m, row in r.residuals().items()}
        out: dict = {}
        levels = sorted({k[0] for row in res.values() for k in row})
        for k in range(levels[0] - d0, levels[-1] - d0 + 1):
            acc: dict = {}
            for m, row in res.items():
                vr = row.get((k + d0,))
                if vr is not None:
                    acc[m] = vr
            for d, cd in ct.items():
                if d == d0:
                    continue
                kk = k + d0 - d
                for m, row in out.items():
                    vr = row.get((kk,))
                    if vr is not None:
                        sub = (vr * VRational.from_scalar(
                            cd * LaurentPoly({0: -1}), n=self.N)).simplify()
                        acc[m] = sub if m not in acc else (
                            acc[m] + sub).simplify()
            for m, vr in acc.items():
                vr = vr.simplify()
                if vr.is_zero():
                    continue
                num = {}
                for ve2, lp in vr.num._terms.items():
                    q = _lp_div(lp, c0)
                    if q is None:
                        return None
                    num[ve2] = q
                nvr = VRational(VLaurent(num, n=self.N), dict(vr.den),
                                n=self.N)
                out.setdefault(m, {})[(k,)] = nvr
        return QuiverURQTorus(out, *self._shape)

    def _adjoint_pair(self, nz) -> QuiverURQTorus:
        """The SU(N)-in-U(N) minimal ('t Hooft adjoint) monopole with
        single-slot dressings: `L_{{(1,e₊),(−1,e₋)}}` from the product of
        the two single-point dyons minus its Wilson screening correction,

            E₁^{(1,e₊)}·E₁^{(−1,e₋)} = L + Σ q^b μ^a·w_λ,

        the correction scanned over the small dressing-contraction Wilson
        set with W1+W2 acceptance (bare pair: `μ^{N−1}·1` — adjoint ⊕
        singlet, the singlet worth the contraction's N−1 adjoint zero
        modes; matches ρ's `μ^{1−N}` frame).  Single-correction cases only
        (|e₊ + e₋| ≤ 1 tested); richer contractions honest-fail."""
        (m1, e1), (m2, e2) = sorted(nz)          # m1 = -1, m2 = +1
        X = self._mul(self.E1(m2, e2), self.E1(m1, e1))
        wf = X.well_formed()
        if wf:
            return X
        tot = e1 + e2
        pad = (0,) * (self.N - 1)
        cands = [QuiverURQTorus.one(*self._shape)] if tot == 0 else []
        if tot != 0:
            cands.append(self._w((tot,) + pad))
        # closed-form frame first (screening = the contraction's N−1
        # adjoint zero modes: μ^{N−1}, with a q-unit for dressed pairs),
        # then the fallback scan
        frames = [(self.N - 1, 0), (self.N - 1, 1), (self.N - 1, -1)]
        frames += [(a, b) for a in range(0, self.N + 2)
                   for b in range(-3, 4) if (a, b) not in frames]
        for corr in cands:
            for a, b in frames:
                r = X + self._scaled(corr, LaurentPoly({b: -1}), a)
                if r.well_formed_w1() and r.well_formed():
                    return r
        raise NotImplementedError(
            f"adjoint pair {nz}: no single-Wilson screening frame found "
            f"(richer contraction — needs the general peel)")

    # ----- the RG flow ----------------------------------------------------
    def slice0(self, x: QuiverURQTorus) -> QuiverURQTorus:
        """The large-mass flow N=2* → pure: the μ-level-0 slice (an algebra
        homomorphism onto the pure torus — certified multiply-compatible
        against `PureUNKAlgebra`, `tests/test_n2star_rg_flow.py`)."""
        out = {}
        for m, row in x.residuals().items():
            f = row.get((0,))
            if f is not None:
                out[m] = {(): f}
        return QuiverURQTorus(out, (self.N,), (0,))

    def pure_chart(self, points) -> QuiverURQTorus:
        """The pure-U(N) canonical chart of `L_{points}` via N=2*."""
        return self.slice0(self.build(points))

    def sun_chart(self, points) -> QuiverURQTorus:
        """The trace-zero pure chart — the SU(N) oracle input
        (`PureSUNKAlgebra` restricts + det-collapses exactly this)."""
        pts = tuple(points)
        if sum(m for m, _ in pts) != 0:
            raise ValueError("sun_chart: SU(N) labels need Σ m_i = 0")
        return self.pure_chart(pts)
