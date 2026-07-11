"""`SkeinA1D3KAlg` — `A_𝖖[[A₁,D₃]]` realized by the stated-SKEIN engine
on the ONCE-PUNCTURED TRIANGLE (the FST chart of the D₃
Argyres–Douglas theory) — the FIRST FLAVOURED SKEIN LEG:
coefficient ring `R(SU(2))`, and the flavour character χ₁
IS the Kauffman peripheral loop around the puncture.

Chart (Z₃-symmetric star triangulation; the bordered
`triangulation.py` engine): vertices 0,1,2 = boundary marked points,
3 = interior puncture; edges 0:r₀=(0,3), 1:r₁=(1,3), 2:r₂=(2,3)
internal (mutable), 3:s₀=(0,1), 4:s₁=(1,2), 5:s₂=(0,2) boundary
(frozen); triangles T₀=(3,1,0), T₁=(4,2,1), T₂=(5,0,2).  Measured:

* SURFACE: `marked_points == (0,1,2)`, one hole `(3,4,5)`, internal
  ids `[0,1,2]`, mutable block = the 3-cycle quiver
  `[[0,-1,1],[1,0,-1],[-1,1,0]]`, and the puncture flavour charge
  `f = r₀+r₁+r₂ = (1,1,1|0,0,0)` lies in `ker(σ)` (full 6×6) — the
  SU(2)-flavour direction of the frozen-extended torus.

* PRODUCT: the GLOBAL-σ ordered Y-product with the κ order reversal
  (`gmul(x, y) = ym(y, x)`, the `YDelta.__mul__` cocycle formula on
  the full 6×6 σ) — certified against the certified
  `PinnedPolygon.multiply` on the square (random monomials + all arc
  pairs) and by the R-intertwining + SQED1 exchange-pair anchor on
  the bordered square chart (gauges `x_T1 = x_V = −1`; the
  non-reversed convention admits NO gauge).  This differs from the
  per-triangle tensor product `tmul`:
  that conversion omits the charge↔key reorder corrections (see
  `project_triangle_monomial`'s docstring) — invisible to
  commutators, wrong on the symmetric part.

* GENERATORS (corner v; boundary ends ENDS = {0:(s₂,s₀), 1:(s₀,s₁),
  2:(s₁,s₂)}): the corner units G_v = long-arc_v(−,+) are UNIT
  MONOMIALS; the canonical generators are

      D_i = Weyl-1 normalization of  short_i(−,+) · G_i^{-1}
      T_i = long_{i+1}(+,−) · G_{i+1}          (ratio exactly 1)
      χ₁  = the bare peripheral LOOP (no dressing, no boundary states)

  with measured images (ordered form; boundary block 0 — the
  canonicals unpin completely, unlike the even-marked square)

      D₀ = lq^{1/2}·y^{(-1,-1,1)} + lq^{-3/2}·y^{(1,-1,1)}
      D₁ = lq^{1/2}·y^{(1,-1,-1)} + lq^{1/2}·y^{(1,1,-1)}
      D₂ = lq^{-3/2}·y^{(-1,1,-1)} + lq^{1/2}·y^{(-1,1,1)}
      χ₁ = lq^{1/2}·(y^{-f} + y^{f})

  and `T_i == lq^{-2}·(D_{i-1}·D_i − 1)` EXACTLY — the geometric and
  relation-derived T's coincide.  All 18 dressing variants collapse
  onto the 3 weyl-1 canonicals.

* RELATIONS: the full standalone `A1D3KAlg` relation set holds
  EXACTLY at q_chart = lq^{-2} in the cyclic frame
  (d₀,d₁,d₂) = (K₀,K₁,K₂) with ZERO lq-gauges — the 18 interaction
  identities (3 × {D_i·D_{i+1}, D_{i+1}·D_i, T_i·D_{i+1},
  D_{i+1}·T_i, T_i·T_{i+1}, T_{i+1}·T_i}), the 6 tile q-commutations
  (T_iD_i = q^{-2}D_iT_i, T_iD_{i-1} = q^{+2}D_{i-1}T_i), and χ₁
  exactly central; χ₁ = +LOOP.  (The −LOOP frames are the
  Z₂-centre / 1-dim-rep ambiguity of the canonical flavour lift —
  the lift is canonical only up to a 1-dimensional rep of `G_f`.)

* FORWARD MAP + STANDING GUARD: labels are the intrinsic `A1D3KAlg`
  4-tuples `(tile, a, b, k)`;

      forward(tile,a,b,k) = lq^{2ab·tw(tile)} · T_{i_T}^a · D_{i_D}^b · χ_k

  (`i_T`/`i_D`/`tw` from `a1d3_kalg._TILE_LETTERS`/`_TILE_TWIST`; χ_k
  by the Chebyshev recursion χ_{k+1} = χ₁·χ_k − χ_{k-1}).  `multiply`
  is computed GENUINELY on the skein side and the result ASSERTED
  verbatim equal — WHOLE-ELEMENT, normalization δ ≡ 0, no per-label
  table — to the intrinsic expansion under `q^k → lq^{-2k}`;
  8725 products were verified (the a,b ≤ 2 grid + the deeper-χ sweep), and
  the guard honest-fails on any mismatch.

`rho` / `rho_inverse` / `trace` are transported from the intrinsic
realization (ρ is geometrically the Z₃ chart rotation r_i → r_{i+1},
s_i → s_{i+1}; all arc images are Z₃-equivariant).
`r_label_decompose` mirrors the intrinsic's single-irrep flavour-lift
coordinate `((tile,a,b,0), k)` exactly.  `build_iso()` returns the
identity-on-labels `KAlgebraIso` to the intrinsic (verifying it tests
the SKEIN structure constants against the intrinsic's —
non-circular).  Registered as the ``'skein-pinned'`` realization of
the A₁D₃ object.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product as iproduct

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly

from a1d3_kalg import A1D3KAlg, _TILE_LETTERS, _TILE_TWIST

from half_laurent import HalfLaurent, ONE as HL_ONE
from phi_map import phi
from triangulation import Triangulation
from y_delta import project_triangle_monomial, _weyl_correction
from curve_realization import _ARC_TYPE_BY_POSITIONS, _CCW_LATER_FIRST

Q = HalfLaurent.monomial

# ---- the chart --------------------------------------------------------------

_EDGES = [(0, 3), (1, 3), (2, 3), (0, 1), (1, 2), (0, 2)]
_TRIANGLE_EDGES = [(3, 1, 0), (4, 2, 1), (5, 0, 2)]
N_E = 6

MUTABLE_BLOCK = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
PUNCTURE_FLAVOUR = (1, 1, 1, 0, 0, 0)   # f = r0 + r1 + r2 ∈ ker σ

# arc charges over (r0, r1, r2, s0, s1, s2)
LOOP_CHARGE = PUNCTURE_FLAVOUR           # peripheral loop around p
SHORT = {                                # short corner arc at vertex v
    0: (1, 0, 0, 1, 0, 1),               # ends s2, s0; crosses r0
    1: (0, 1, 0, 1, 1, 0),               # ends s0, s1; crosses r1
    2: (0, 0, 1, 0, 1, 1),               # ends s1, s2; crosses r2
}
LONG = {                                 # long corner arc at v (around p)
    0: (0, 1, 1, 1, 0, 1),               # ends s2, s0; crosses r1, r2
    1: (1, 0, 1, 1, 1, 0),               # ends s0, s1; crosses r2, r0
    2: (1, 1, 0, 0, 1, 1),               # ends s1, s2; crosses r0, r1
}
# boundary end ids per corner vertex (in (earlier, later) ccw order along
# the boundary: at vertex v the two sides are s_{v-1} and s_v)
ENDS = {0: (5, 3), 1: (3, 4), 2: (4, 5)}

# the exact measured generator images (ordered form; construction guards)
D_GUARD = {
    0: {(-1, -1, 1, 0, 0, 0): Q(Fraction(1, 2)),
        (1, -1, 1, 0, 0, 0): Q(Fraction(-3, 2))},
    1: {(1, -1, -1, 0, 0, 0): Q(Fraction(1, 2)),
        (1, 1, -1, 0, 0, 0): Q(Fraction(1, 2))},
    2: {(-1, 1, -1, 0, 0, 0): Q(Fraction(-3, 2)),
        (-1, 1, 1, 0, 0, 0): Q(Fraction(1, 2))},
}
LOOP_GUARD = {(-1, -1, -1, 0, 0, 0): Q(Fraction(1, 2)),
              (1, 1, 1, 0, 0, 0): Q(Fraction(1, 2))}


# ---- the global-σ ordered torus arithmetic ----------------------------------


def ordered_ymul(sig, a, b):
    """Ordered-form Y-product `a·b`: `y^u y^v = lq^{-Σ_{i<j} v_i u_j
    σ_ij} y^{u+v}` (the `YDelta.__mul__` formula on the full σ)."""
    n = len(sig)
    out = {}
    for ca, va in a.items():
        for cb, vb in b.items():
            lq_pow = 0
            for i in range(n):
                bi = cb[i]
                if bi == 0:
                    continue
                for j in range(i + 1, n):
                    aj = ca[j]
                    if aj == 0:
                        continue
                    lq_pow -= bi * aj * sig[i][j]
            cc = tuple(ca[k] + cb[k] for k in range(n))
            nv = (va * vb).shift(lq_pow)
            s = out.get(cc, HalfLaurent.zero()) + nv
            if s.is_zero():
                out.pop(cc, None)
            else:
                out[cc] = s
    return out


def skein_mul(sig, x, y):
    """The skein product via the κ anti-hom order reversal:
    `skein(a·b) = Y(b-img)·Y(a-img)` in ordered form (the convention
    matching `PinnedPolygon.multiply`; certified on the square)."""
    return ordered_ymul(sig, y, x)


def inverse_monomial(sig, x):
    """Inverse of a single-monomial torus element."""
    ((u, c),) = x.items()
    neg = tuple(-a for a in u)
    z = skein_mul(sig, {u: c}, {neg: HL_ONE})
    ((_zk, zc),) = z.items()
    return {neg: zc ** -1}


def torus_add(x, y):
    out = dict(x)
    for k, c in y.items():
        s = out.get(k, HalfLaurent.zero()) + c
        if s.is_zero():
            out.pop(k, None)
        else:
            out[k] = s
    return out


def torus_sub(x, y):
    return torus_add(x, {k: -v for k, v in y.items()})


def torus_scale(x, c):
    return {k: v * c for k, v in x.items() if not (v * c).is_zero()}


# ---- stated-arc quantum-trace images ----------------------------------------


def local_realize(tri, charge):
    """Per-triangle arc realization of an edge-charge vector (no global
    closure check; ≤ 1 arc per triangle — the scope of this chart)."""
    out = []
    for ti, ecc in enumerate(tri.triangle_edges):
        a = [abs(charge[e]) for e in ecc]
        if sum(a) % 2:
            raise ValueError(f"triangle {ti}: odd parity {a}")
        m = [0, 0, 0]
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            if a[k] > a[i] + a[j]:
                raise ValueError(f"triangle {ti}: triangle ineq fails {a}")
            m[k] = (a[i] + a[j] - a[k]) // 2
        if sum(m) > 1:
            raise ValueError(
                f"triangle {ti}: {sum(m)} arcs (multi-arc out of scope)")
        if sum(m) == 0:
            out.append(None)
            continue
        k = m.index(1)
        i, j = (k + 1) % 3, (k + 2) % 3
        at = _ARC_TYPE_BY_POSITIONS[frozenset({i, j})]
        lp, ep = _CCW_LATER_FIRST[at]
        out.append((at, ecc[lp], ecc[ep]))
    return out


def arc_image(tri, charge, eps_by_bnd):
    """Quantum trace of the (stated) arc/curve on `tri`: boundary end
    states `eps_by_bnd: boundary edge id -> ±1`; internal crossings
    state-summed.  Returns dict `charge -> ORDERED HalfLaurent` (the
    global ordered frame of `project_triangle_monomial`)."""
    arcs = local_realize(tri, charge)
    internal_states = sorted(
        e for e in tri.internal_edge_ids if charge[e])
    for e in tri.boundary_edge_ids:
        if charge[e] and e not in eps_by_bnd:
            raise ValueError(f"boundary edge {e} crossed but no state given")
    out = {}
    for bits in iproduct((1, -1), repeat=len(internal_states)):
        mu = dict(zip(internal_states, bits))

        def state_at(e):
            return eps_by_bnd[e] if e in eps_by_bnd else mu[e]

        triples, coef, dead = [], HL_ONE, False
        for ti, a in enumerate(arcs):
            if a is None:
                triples.append((0, 0, 0))
                continue
            at, e_eps, e_epsp = a
            f = phi(at, state_at(e_eps), state_at(e_epsp))
            if f.is_zero():
                dead = True
                break
            ((e, co),) = f.items()
            triples.append(e)
            coef = coef * co
        if dead:
            continue
        pr = project_triangle_monomial(tri, triples, coef)
        if pr is None:
            continue
        ch, oc = pr
        s = out.get(ch, HalfLaurent.zero()) + oc
        if s.is_zero():
            out.pop(ch, None)
        else:
            out[ch] = s
    return out


def weylize(tri, charge, oc):
    """The Weyl-symmetric normal form of an ordered coefficient: shift
    by −(Weyl correction) and return an integer-exponent LaurentPoly."""
    w = oc.shift(-_weyl_correction(tri, charge))
    d = {}
    for e, co in w.items():
        assert e.denominator == 1, f"non-integer Weyl exponent at {charge}"
        d[int(e)] = d.get(int(e), 0) + co
    return LaurentPoly(d)


# ---- the KAlgebra -----------------------------------------------------------


class SkeinA1D3KAlg(KAlgebra):
    """The stated-skein realization of `A_𝖖[[A₁,D₃]]` on the
    once-punctured triangle, on the intrinsic `A1D3KAlg`
    `(tile, a, b, k)` labels.  See the module docstring."""

    # q_int = lq^{ORI}: intrinsic q^k → lq^{-2k} (q_chart = lq^{-2})
    ORI = -2

    def __init__(self, intrinsic=None):
        if intrinsic is None:
            intrinsic = A1D3KAlg()
        self._intrinsic = intrinsic
        self.t = Triangulation.from_edge_data(
            n_punctures=4,
            edges=list(_EDGES),
            triangle_edges=list(_TRIANGLE_EDGES),
            allow_boundary=True,
        )
        self._sig = self.t.sigma()
        self._zero = (0,) * N_E
        self._certify_surface()
        self._build_generators()
        self._certify_relations()
        self._chi_cache = {0: {self._zero: HL_ONE}, 1: self._X1}
        self._fwd_cache = {}
        self._mult_cache = {}

    # ---- construction-time geometric certificates -----------------------

    def _certify_surface(self):
        """The once-punctured-triangle chart facts (all measured):
        boundary marked points, the single hole, the 3-cycle mutable
        block, and the puncture flavour charge in ker σ."""
        t = self.t
        assert t.marked_points == (0, 1, 2), "boundary marked points wrong"
        assert t.holes == ((3, 4, 5),), "hole (frozen boundary) wrong"
        assert t.internal_edge_ids == [0, 1, 2], "internal edge ids wrong"
        assert t.mutable_block() == MUTABLE_BLOCK, \
            "mutable block is not the 3-cycle quiver"
        f = t.puncture_flavour_charge(3)
        assert f == PUNCTURE_FLAVOUR, "puncture flavour charge wrong"
        sig = self._sig
        assert all(sum(sig[i][j] * f[j] for j in range(N_E)) == 0
                   for i in range(N_E)), \
            "puncture flavour charge not in ker(sigma)"

    def _build_generators(self):
        """Corner units G_v (unit monomials), the Weyl-1 dressed shorts
        D_i (measured images `D_GUARD`), the dressed longs T_i, and the
        peripheral loop χ₁ (measured image `LOOP_GUARD`)."""
        sig = self._sig
        G, D, T = {}, [None] * 3, [None] * 3
        for v in range(3):
            e1, e2 = ENDS[v]
            g = arc_image(self.t, LONG[v], {e1: -1, e2: 1})
            assert len(g) == 1, f"corner unit G_{v} is not a unit monomial"
            G[v] = g
        for i in range(3):
            e1, e2 = ENDS[i]
            x = skein_mul(
                sig,
                arc_image(self.t, SHORT[i], {e1: -1, e2: 1}),
                inverse_monomial(sig, G[i]),
            )
            exps = set()
            for ch, val in x.items():
                lp = weylize(self.t, ch, val)
                d = dict(lp._coeffs)
                assert len(d) == 1 and next(iter(d.values())) == 1, \
                    f"D_{i} weyl coefficient not a unit monomial at {ch}"
                exps.add(next(iter(d)))
            assert len(exps) == 1, \
                f"D_{i}: the two Weyl exponents differ ({sorted(exps)})"
            D[i] = torus_scale(x, Q(-next(iter(exps))))
            assert D[i] == D_GUARD[i], \
                f"D_{i} differs from the measured image"
        for i in range(3):
            v = (i + 1) % 3
            e1, e2 = ENDS[v]
            T[i] = skein_mul(
                sig, arc_image(self.t, LONG[v], {e1: 1, e2: -1}), G[v])
        X1 = arc_image(self.t, LOOP_CHARGE, {})
        assert X1 == LOOP_GUARD, \
            "peripheral loop differs from the measured image"
        self._G, self._D, self._T, self._X1 = G, D, T, X1

    def _certify_relations(self):
        """The exact algebra certificates at q_chart = lq^{-2}: the
        geometric T's equal the relation-derived ones, χ₁ = LOOP is
        relation-derived and central, and the full standalone relation
        set (18 interaction identities + 6 tile q-commutations) holds
        verbatim.  q^n ↦ lq^{-2n} throughout."""
        sig = self._sig
        gm = lambda x, y: skein_mul(sig, x, y)
        one = {self._zero: HL_ONE}
        D, T, X1 = self._D, self._T, self._X1
        for i in range(3):
            ip1, im1 = (i + 1) % 3, (i - 1) % 3
            # geometric T_i == relation-derived lq^{-2}(D_{i-1} D_i - 1)
            assert T[i] == torus_scale(
                torus_sub(gm(D[im1], D[i]), one), Q(-2)), \
                f"T_{i} geometric/relation-derived mismatch"
            # chi_1 relation-derived from T_i D_{i+1} == LOOP
            x1 = torus_sub(
                torus_sub(gm(T[i], D[ip1]), torus_scale(D[i], Q(2))),
                torus_scale(D[im1], Q(-2)))
            assert x1 == X1, f"relation-derived chi_1 (i={i}) != LOOP"
            # the 6 interaction identities of the i-th Z3 orbit slot
            assert gm(D[i], D[ip1]) == \
                torus_add(one, torus_scale(T[ip1], Q(2))), \
                f"D_{i} D_{ip1} = 1 + q^-1 T_{ip1} failed"
            assert gm(D[ip1], D[i]) == \
                torus_add(one, torus_scale(T[ip1], Q(-2))), \
                f"D_{ip1} D_{i} = 1 + q T_{ip1} failed"
            assert gm(T[i], D[ip1]) == torus_add(
                X1, torus_add(torus_scale(D[i], Q(2)),
                              torus_scale(D[im1], Q(-2)))), \
                f"T_{i} D_{ip1} = chi_1 + q^-1 D_{i} + q D_{im1} failed"
            assert gm(D[ip1], T[i]) == torus_add(
                X1, torus_add(torus_scale(D[i], Q(-2)),
                              torus_scale(D[im1], Q(2)))), \
                f"D_{ip1} T_{i} = chi_1 + q D_{i} + q^-1 D_{im1} failed"
            x1d = gm(X1, D[i])
            dd = gm(D[i], D[i])
            assert gm(T[i], T[ip1]) == torus_add(
                one, torus_add(torus_scale(x1d, Q(2)),
                               torus_scale(dd, Q(4)))), \
                f"T_{i} T_{ip1} = 1 + q^-1 chi_1 D_{i} + q^-2 D_{i}^2 failed"
            assert gm(T[ip1], T[i]) == torus_add(
                one, torus_add(torus_scale(x1d, Q(-2)),
                               torus_scale(dd, Q(-4)))), \
                f"T_{ip1} T_{i} = 1 + q chi_1 D_{i} + q^2 D_{i}^2 failed"
            # the 2 tile q-commutations at i
            assert gm(T[i], D[i]) == torus_scale(gm(D[i], T[i]), Q(4)), \
                f"T_{i} D_{i} = q^-2 D_{i} T_{i} failed"
            assert gm(T[i], D[im1]) == \
                torus_scale(gm(D[im1], T[i]), Q(-4)), \
                f"T_{i} D_{im1} = q^2 D_{im1} T_{i} failed"
        # chi_1 exactly central against every generator and unit
        for Z in list(D) + list(T) + list(self._G.values()):
            assert gm(X1, Z) == gm(Z, X1), "chi_1 = LOOP is not central"

    # ---- the dressed generators (torus elements) -------------------------

    def gmul(self, x, y):
        """The certified skein product on this chart's torus."""
        return skein_mul(self._sig, x, y)

    def unit(self, v: int) -> dict:
        """The corner unit monomial G_v = long-arc_v(−,+)."""
        return self._G[v % 3]

    def D(self, i: int) -> dict:
        """The Weyl-1 dressed short arc D_i (torus element; the skein
        image of the intrinsic generator D_i)."""
        return self._D[i % 3]

    def T(self, i: int) -> dict:
        """The dressed long arc T_i = long_{i+1}(+,−)·G_{i+1} (torus
        element; the skein image of the intrinsic generator T_i)."""
        return self._T[i % 3]

    def loop(self) -> dict:
        """The peripheral loop around the puncture = χ₁ (central)."""
        return self._X1

    def chi_image(self, k: int) -> dict:
        """The skein image of χ_k (Chebyshev: χ_{k+1} = χ₁χ_k − χ_{k-1})."""
        if k not in self._chi_cache:
            self._chi_cache[k] = torus_sub(
                self.gmul(self._X1, self.chi_image(k - 1)),
                self.chi_image(k - 2))
        return self._chi_cache[k]

    # ---- forward map (canonical label -> torus element) -------------------

    def forward(self, lbl) -> dict:
        """`forward(tile,a,b,k) = lq^{2ab·tw} · T_{i_T}^a · D_{i_D}^b ·
        χ_k` — the skein image of the intrinsic canonical label."""
        lbl = self._intrinsic.canonicalise(lbl)
        if lbl not in self._fwd_cache:
            tile, a, b, k = lbl
            (_, i_T), (_, i_D) = _TILE_LETTERS[tile]
            tw = _TILE_TWIST[tile]
            out = {self._zero: Q(Fraction(2 * a * b * tw))}
            for _ in range(a):
                out = self.gmul(out, self._T[i_T])
            for _ in range(b):
                out = self.gmul(out, self._D[i_D])
            if k:
                out = self.gmul(out, self.chi_image(k))
            self._fwd_cache[lbl] = out
        return self._fwd_cache[lbl]

    def _chart_to_hl(self, c: LaurentPoly) -> HalfLaurent:
        """The exact chart dictionary `q^k → lq^{ORI·k}` (= lq^{-2k})."""
        out = {}
        for k, v in c._coeffs.items():
            e = Fraction(self.ORI * k)
            out[e] = out.get(e, 0) + v
        return HalfLaurent(out)

    # ---- KAlgebra primitives -----------------------------------------------

    def coefficient_ring(self):
        """`R(SU(2))` — the intrinsic's `SU2ZPlusRing` instance (the
        first flavoured skein leg)."""
        return self._intrinsic.coefficient_ring()

    def identity(self):
        return self._intrinsic.identity()

    def canonicalise(self, x):
        return self._intrinsic.canonicalise(x)

    def multiply(self, a, b) -> Element:
        """Structure constants computed GENUINELY on the skein side,
        with the STANDING GUARD: the skein product of the forward
        images is ASSERTED verbatim equal (whole-element, δ ≡ 0 — no
        per-label table, never fitted) to the intrinsic expansion
        under `q^k → lq^{-2k}`; honest-fails on any mismatch."""
        A = self._intrinsic
        a = A.canonicalise(tuple(a))
        b = A.canonicalise(tuple(b))
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        got = self.gmul(self.forward(a), self.forward(b))
        hint = A.multiply(a, b)
        want = {}
        for lam, c in hint.terms.items():
            want = torus_add(
                want,
                torus_scale(self.forward(tuple(lam)), self._chart_to_hl(c)))
        assert got == want, (
            f"skein/intrinsic whole-element mismatch at {a}*{b} "
            f"(the standing delta == 0 guard): the skein-side product "
            f"does not equal Σ_l chart(c_l)·forward(l)")
        el = Element(dict(hint.terms))
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        """Transported from the intrinsic (geometrically ρ is the Z₃
        chart rotation r_i → r_{i+1}, s_i → s_{i+1}; all arc images
        are Z₃-equivariant)."""
        return self._intrinsic.rho(tuple(a))

    def rho_inverse(self, a):
        return self._intrinsic.rho_inverse(tuple(a))

    def trace(self, a, K: int = 20, **kwargs):
        """Transported from the intrinsic realization (as in the polygon
        siblings; the stated-side trace is not computed intrinsically)."""
        return self._intrinsic.trace(tuple(a), K, **kwargs)

    # ---- flavour lift (mirrors the intrinsic exactly) -----------------------

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate — the intrinsic's
        verbatim: peel the SU(2) spin index `k` off the gauge monomial
        `(tile, a, b)`."""
        tile, a, b, k = self._intrinsic.canonicalise(label)
        return (tile, a, b, 0), k

    def r_label_compose(self, section, r_basis_label):
        tile, a, b, _ = self._intrinsic.canonicalise(section)
        return self._intrinsic.canonicalise((tile, a, b, r_basis_label))

    def embed_R(self, r) -> Element:
        """Central embedding `R(SU(2)) ↪ A_𝖖` (transported: χ_k is the
        central canonical `L_{(0,0,0,k)}` — skein-side, the k-th
        Chebyshev polynomial of the peripheral loop)."""
        return self._intrinsic.embed_R(r)

    # ---- the iso witness -----------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({tuple(lbl): one})

        return KAlgebraIso(self, self._intrinsic, _id, _id,
                           name="a1d3[skein-pinned→standalone]")

    def __repr__(self) -> str:
        return ("SkeinA1D3KAlg(once-punctured triangle, "
                "q_chart=lq^-2, chi_1=peripheral loop)")
