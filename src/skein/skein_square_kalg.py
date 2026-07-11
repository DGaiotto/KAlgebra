"""`SkeinSquareKAlg` — SQED1 = A_𝖖[U(1)+N_f=1] extracted from the
PINNED quantum torus of the stated SQUARE (the even-marked / gauged
step of the pinned / glue programme).

In `PinnedPolygon(4)` (the quantum torus over Γ~ = Z ⊕ Z⁴ — ONE
internal edge, four boundary pins; `pinned_polygon.py`) the side arcs
are single-monomial UNITS and the two diagonals quantum-trace to
F(D₀) = {(-1|μ): 1, (1|μ): 1} (two terms) and F(D₁) = {(1|ν): 1}
(one term), μ = (1,0,1,0), ν = (0,1,0,1).  Measured:

* UNLIKE the odd polygons, pin-0 dressing is IMPOSSIBLE here: the
  boundary grading χ = (1,-1,1,-1) kills every side pin — the side-pin
  image is EXACTLY ker χ — while χ(pin D₀) = +2, χ(pin D₁) = -2 (the
  even-marked obstruction).  The dressed monopoles keep a residual
  pin ±μ:

      T₀ = F(D₀)                             (bare;  pin +μ)
      T₁ = lq⁻¹ F(D₁) s₁⁻¹ s₃⁻¹              (pin -μ; the two inverted
                                              sides commute)
      V  = lq⁻¹ s₀ s₁⁻¹ s₂ s₃⁻¹              (the charge-(-2|0) UNIT
                                              monomial — a torus unit,
                                              pin 0; in this interleaved
                                              order — c(s₁, s₂) = −1, so
                                              s₀ s₂ (s₁ s₃)⁻¹ differs by
                                              the phase and needs no
                                              prefactor)

  and the dressed algebra lives on the index-2 even sublattice
  {(k | m·μ) : k ≡ m mod 2} — the 't Hooft half-quantization shape.
  No strictly-central and no q-central χ = ±2 monomial exists that
  could complete the unpinning (scans certified in the tests).

* The SQED1 exchange relations hold EXACTLY in the pinned torus
  (q_chart = lq⁻²):

      T₀ T₁ = 1 + q_chart V,      T₁ T₀ = 1 + q_chart⁻¹ V

  with c(T₀, V) = -4 / c(T₁, V) = +4 uniformly across both T₀ terms
  (u₊v = q² v u₊, u₋v = q⁻² v u₋), side coupling c(s_r, T_p) =
  (-1)^{p+r}, and c(s_r, V) = 0 — the dictionary T₀ ↔ u₊ = L_{(1,0)},
  T₁ ↔ u₋ = L_{(-1,0)}, V ↔ v = L_{(0,1)}.

* FORWARD MAP.  Labels are the intrinsic `Sqed1KAlg` (m, n) labels
  verbatim;

      K(m, n) = lq^{2mn} · (T₀ if m ≥ 0 else T₁)^{|m|}
                        · (V if n ≥ 0 else V⁻¹)^{|n|}

  matches L_{m,n} = q^{-|m|·n·sign(m)} u_±^{|m|} v^n at q_chart = lq⁻²,
  with pin(K(m,n)) = m·μ exactly.

* 225/225 exact products on the 15×15 label grid vs `Sqed1KAlg` with
  normalization δ IDENTICALLY 0 — STRONGER than the pentagon (no
  per-label δ table): `multiply` is computed GENUINELY in the pinned
  torus and peeled against the forward images of the intrinsic-hinted
  LABEL SET (labels only), and each peeled coefficient is ASSERTED to
  equal the intrinsic coefficient verbatim under q^k → lq^{-2k} — a
  standing guard, never a fit.

`rho` / `rho_inverse` / `trace` are transported from the intrinsic
realization (documented, as in the pentagon siblings; the intrinsic
stated-side trace remains the separate work item).  `build_iso()`
returns the `KAlgebraIso` to the intrinsic (identity label maps;
verifying it tests the PINNED structure constants against the
intrinsic's — non-circular).  Registered as the ``'skein-pinned'``
realization of the SQED1 object.
"""

from __future__ import annotations

from fractions import Fraction

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import RPowerSeries, TrivialZPlusRing

from half_laurent import HalfLaurent
from pinned_polygon import PinnedPolygon
from skein_pentagon_kalg import SkeinPentagonKAlg

Q = HalfLaurent.monomial

MU = (1, 0, 1, 0)          # the residual monopole pin
CHI = (1, -1, 1, -1)       # the boundary obstruction grading


def _f4(x):
    return x % 4


class SkeinSquareKAlg(KAlgebra):
    """The pinned-torus realization of SQED1, on the `Sqed1KAlg`
    (m, n) labels."""

    def __init__(self, intrinsic=None):
        if intrinsic is None:
            from kalgebra_samples import Sqed1KAlg
            intrinsic = Sqed1KAlg()
        self._intrinsic = intrinsic
        self.pp = PinnedPolygon(4)
        pp = self.pp
        self._sides = [pp.arc_F(r, _f4(r + 1), 1, 1) for r in range(4)]
        self._sides_inv = [pp.inverse_monomial(s) for s in self._sides]
        self._chords = [pp.arc_F(0, 2, 1, 1), pp.arc_F(1, 3, 1, 1)]
        # the dressed monopoles (w0 = (0,0,0,0), x0 = 0;
        # w1 = (0,1,0,1), x1 = -1) and the torus unit V
        self._T0 = self._chords[0]
        t1 = pp.multiply(self._chords[1], self._sides_inv[1])
        t1 = pp.multiply(t1, self._sides_inv[3])
        self._T1 = pp.scale(t1, Q(-1))
        self._V = pp.monomial((-2, 0, 0, 0, 0))
        self._Vinv = pp.inverse_monomial(self._V)
        self._certify_geometry()
        self._fwd_cache = {}
        self._mult_cache = {}

    def _certify_geometry(self):
        """Construction-time geometric certificates (all cheap): the
        residual pins ±μ / 0, the V side-word identity, and the SQED1
        exchange pair — exact in the pinned torus."""
        pp = self.pp
        for r in range(4):
            assert len(self._sides[r]) == 1, \
                "side quantum trace is not a monomial"
        neg_mu = tuple(-x for x in MU)
        assert pp.pinning_vector(self._T0) == MU, "T0 pin is not +mu"
        assert pp.pinning_vector(self._T1) == neg_mu, "T1 pin is not -mu"
        assert pp.pinning_vector(self._V) == (0,) * 4, "V pin is not 0"
        # V = lq^{-1} s0 s1^{-1} s2 s3^{-1}  (interleaved order;
        # c(s1, s2) = -1, so the s0 s2 (s1 s3)^{-1} order carries no
        # prefactor)
        w = pp.multiply(self._sides[0], self._sides_inv[1])
        w = pp.multiply(w, self._sides[2])
        w = pp.multiply(w, self._sides_inv[3])
        assert pp.scale(w, Q(-1)) == self._V, "V side-word identity failed"
        # the exchange pair at q_chart = lq^{-2}:
        # T0 T1 = 1 + q_chart V,  T1 T0 = 1 + q_chart^{-1} V
        assert pp.multiply(self._T0, self._T1) == \
            pp.add(pp.one(), pp.scale(self._V, Q(-2))), \
            "exchange T0*T1 = 1 + q_chart*V failed"
        assert pp.multiply(self._T1, self._T0) == \
            pp.add(pp.one(), pp.scale(self._V, Q(2))), \
            "exchange T1*T0 = 1 + q_chart^{-1}*V failed"

    # ---- the dressed generators ---------------------------------------------

    def T(self, p: int) -> dict:
        """The dressed monopole T_p (residual pin ±μ), p ∈ {0, 1}."""
        return (self._T0, self._T1)[p]

    def V(self) -> dict:
        """The flux unit V (the charge-(-2|0) unit monomial, pin 0)."""
        return self._V

    def side(self, r: int) -> dict:
        """The pinned side monomial F(s_r(+,+))."""
        return self._sides[_f4(r)]

    # ---- forward map (canonical label -> pinned element) ---------------------

    def forward(self, lbl) -> dict:
        """K(m, n) = lq^{2mn} T_±^{|m|} V^{±|n|} — the pinned image of
        the canonical label (m, n)."""
        m, n = lbl
        lbl = (m, n)
        if lbl not in self._fwd_cache:
            pp = self.pp
            out = pp.scale(pp.one(), Q(Fraction(2 * m * n)))
            T = self._T0 if m >= 0 else self._T1
            for _ in range(abs(m)):
                out = pp.multiply(out, T)
            W = self._V if n >= 0 else self._Vinv
            for _ in range(abs(n)):
                out = pp.multiply(out, W)
            self._fwd_cache[lbl] = out
        return self._fwd_cache[lbl]

    # ---- KAlgebra primitives --------------------------------------------------

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return (0, 0)

    def multiply(self, a, b) -> Element:
        a, b = tuple(a), tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        pp = self.pp
        prod = pp.multiply(self.forward(a), self.forward(b))
        # candidates from the intrinsic expansion (LABELS only; every
        # coefficient below comes from the pinned torus)
        hint = self._intrinsic.multiply(a, b)
        pool = {tuple(lam): self.forward(tuple(lam)) for lam in hint.terms}
        rem = dict(prod)
        skein_co = {}
        while rem:
            # unique-charge peel (the PinnedPentagonKAlg loop), anchored
            # at a unit-monomial charge of the owner so the FULL
            # HalfLaurent coefficient peels genuinely (SQED1 structure
            # constants are multi-term, e.g. q + q^3 — the pentagon's
            # monomial ratio alone would not reach them)
            hit = None
            for u in sorted(rem):
                owners = [lam for lam, c in pool.items() if u in c]
                if len(owners) != 1:
                    continue
                lam = owners[0]
                anchor = pool[lam][u].items()
                if len(anchor) == 1 and anchor[0][1] in (1, -1):
                    hit = (u, lam, anchor[0])
                    break
                r = SkeinPentagonKAlg._mono_ratio_hl(rem[u], pool[lam][u])
                if r is not None:
                    hit = (u, lam, None, r)
                    break
            assert hit is not None, f"peel stuck for {a}*{b}"
            u, lam = hit[0], hit[1]
            if hit[2] is not None:
                (be, bv) = hit[2]
                h = rem[u] * Q(-be, bv)          # bv^{-1} = bv for ±1
            else:
                e, sg = hit[3]
                h = Q(e) * sg
            piece = pp.scale(pool.pop(lam), h)
            rem = pp.add(rem, {v: -c for v, c in piece.items()})
            skein_co[lam] = h
        assert not pool, (
            f"pinned/intrinsic label-set mismatch at {a}*{b}: hinted "
            f"labels {sorted(pool)} absent from the pinned product")
        # normalization: δ is IDENTICALLY 0 (no per-label table —
        # stronger than the pentagon), so every peeled coefficient must
        # equal the intrinsic coefficient verbatim under q^k -> lq^{-2k}.
        # ASSERTED on every product (a standing guard, never a fit).
        out = {}
        for lam, h in skein_co.items():
            cch = hint.terms[lam]
            assert h == SkeinPentagonKAlg._chart_to_hl(cch), (
                f"pinned coefficient differs from intrinsic at {lam} "
                f"in {a}*{b}: {h} vs chart {cch} (delta != 0)")
            out[lam] = cch
        el = Element({lam: c for lam, c in out.items()})
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        r = self._intrinsic.rho(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return tuple(r)

    def rho_inverse(self, a):
        r = self._intrinsic.rho_inverse(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return tuple(r)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """Transported from the intrinsic realization (the documented
        weak point, as in the pentagon siblings — the intrinsic
        stated-side trace is the separate work item)."""
        return self._intrinsic.trace(tuple(a), K)

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: unflavoured
        (the U(1)_F direction is specialized in the intrinsic)."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness --------------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        unit = LaurentPoly({0: 1})

        def fwd(lbl):
            return Element({tuple(lbl): unit})

        def inv(lbl):
            return Element({tuple(lbl): unit})

        return KAlgebraIso(self, self._intrinsic, fwd, inv,
                           name="sqed1[skein-pinned→intrinsic]")
