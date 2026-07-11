"""`SkeinAnnulusKAlg` — A_𝖖[pure SU(2)] extracted from the certified
bordered chart of the stated ANNULUS with one marked point per
boundary circle (the GMN pure-SU(2) surface = the Kronecker quiver's
cluster surface; realised as the once-punctured torus cut along an
ideal edge, `Triangulation.cut(0)` + the T-pin6/T-pin7 `CutReglue`
pipeline).  Labels are the intrinsic BPS lower tropical charges
`γ ∈ ℤ²` of `pure_ade_kalgebra([("A", 1)])` verbatim
(`abelianized_su2_bps_iso.gamma_of/me_of` give the `L_{m,e}`
dictionary).

Chart (measured; construction-asserted): the cut chart has boundary
edges [0, 3] (holes ((0,), (3,))), internal edges [1, 2], mutable
block [[0, 2], [-2, 0]] — the Kronecker quiver — and the extended
`BPSKAlgebra` `B` on the FULL frozen-extended lattice ℤ⁴ with the
embedded 2-state strong-coupling spec [(0,1,0,0), (0,0,1,0)].  The
flavour kernel of `B` is `K = ⟨K1 = (2,1,1,0), K2 = (-1,0,0,1)⟩`; the
reduction `P(u) = (u1 - u2, 2·u2 - u0 - u3)` has `ker P = K` EXACTLY
(the four rows Γ(1,0), Γ(0,1), K1, K2 are unimodular) and the section
`Γ(γ) = (-γ1, γ0, 0, 0)` satisfies `P∘Γ = id`.

Measured (exact):

* THE STATED-SECTOR DICTIONARY.  Each once-crossing curve of the
  closed torus opens to 4 endpoint-state sectors on the annulus; the
  8 stated-arc images sit as the exact bare part of the extended
  canonical F's, hosts by the leading-term law (asserted at
  construction):

      curve (1,1,0): (+,+)→(1,1,0,1),  (+,-)→(1,1,0,-1),
                     (-,+)→(-1,1,0,1), (-,-)→(-1,-1,0,-1)
      curve (1,0,1): (+,+)→(1,0,-1,1), (+,-)→(1,0,-1,-1),
                     (-,+)→(-1,0,-1,1), (-,-)→(-1,0,-1,-1)

* THE χ₂ SURPRISE.  The once-wrapping CORE multicurve (0,1,1) (the
  closed loop around the annulus) quantum-traces to
  `X_{(0,-1,-1,0)} + X_{(0,-1,1,0)} + X_{(0,1,1,0)}` (all unit
  coefficients) — the exact bare part of `F(L_{(0,-1,-1,0)})`, class
  `P = (0,-2)` = the ADJOINT Wilson line `χ₂ = L_{0,2}` — NOT the
  fundamental.  Certified three ways: (a) the F-inclusion verbatim;
  (b) the fusion `W² = W₂ + W + 1`
  (`multiply((0,-2),(0,-2)) = L_{(0,-4)} + L_{(0,-2)} + 1`);
  (c) the trace `-2q² + q⁴` at K=6.

* χ₁ AT THE BOUNDARY-UNIT CANONICAL.  `χ₁ = L_{0,1}` is the canonical
  at the boundary-edge unit label `e0 = (1,0,0,0)`:
  `F(L_{e0}) = X_{e0} + X_{e0+e2} + X_{e0+e1+e2}` (unit coefficients),
  `χ₁² = μ₁·(χ₂ + 1)` in the chart (`L_{e0}·L_{e0} = L_{2e0} + L_{K1}`
  with `F(L_{2e0}) = X_{K1}·F(L_{(0,-1,-1,0)})`), and zero trace.
  This identification is ALGEBRAICALLY certified (F, fusion, trace)
  but its GEOMETRIC stated realization is conjectural: the stated
  boundary arc itself is untested — the annulus analogue of the
  intrinsic-trace weak point of the closed-sphere realisations.

* THE PARITY / PIN OBSTRUCTION.  On the chart lattice the electric
  charge obeys `e ≡ u0 + u3 (mod 2)` (from `y = 2u2 - u0 - u3`), so
  odd-electric lines cannot unpin: the geometric closed-multicurve
  sector (`u0 = u3 = 0`) alone generates the even / SO(3)-like
  subalgebra, and the odd sector enters only through the
  boundary-unit canonicals (μ-unit torsor: `F(L_{±K1})`, `F(L_{±K2})`
  are single unit monomials, `μ₁·μ₁⁻¹ = 1`).

* THE BATTERY.  All 144 ordered products on the 12-line grid
  {(0,0)…(2,1)} (336 product terms) equal the intrinsic BPS multiply
  VERBATIM after reduction by `P` — normalization δ IDENTICALLY 0 and
  chart q == intrinsic q IDENTITY at this tier (the `lq^{-2}`
  convention of the pinned polygons lives one tier down and does NOT
  enter); no two chart terms of one product ever collide in a
  P-class (guarded loudly per product regardless).  `multiply` is the
  STANDING GUARD form: computed GENUINELY in the certified bordered
  chart, reduced, and asserted term-by-term equal to the intrinsic —
  honest-fail, never fitted.

* NATIVE ρ AND TRACE (unlike the pinned-polygon siblings, which
  transport them from the intrinsic): `ρ = P∘ρ_B∘Γ`, measured
  equivariant on the whole grid (ρ_B's image label carries a flavour
  unit `μ₁^c` twist which `P` strips); `trace = ε∘Tr_B∘Γ` where `ε`
  is the flavour augmentation `μ^w ↦ 1` (`zplus_ring.augmentation_hom`
  — exact, coefficient-wise).  The RAW chart trace carries the same
  `μ₁^c` unit twist in its R-valued coefficients (measured: each
  nonzero q-coefficient is a single μ-weight `(c, 0)`), and its
  ε-image equals the intrinsic trace verbatim on all 12 grid lines at
  K=6 (`1 → 1+q⁴`, `L_{1,0} → -q+q⁵`, `χ₂ → -2q²+q⁴`, `χ₁ → 0`,
  `L_{2,0} → q²`, odd/dyon lines → 0).

NON-CIRCULARITY CAVEAT (recorded, no overclaim): for general labels
`multiply` is computed in the certified bordered-chart `BPSKAlgebra`
with skein anchoring at the measured generators (the 8 stated
sectors, the core loop, the boundary units) — there is no intrinsic
bordered `SkeinAlgebra` to state-sum against yet, so the skein
content beyond the anchors is the certified chart itself; the
verbatim assert keeps every product honest-fail.

`build_iso()` returns the `KAlgebraIso` to the intrinsic BPS
realisation (identity label maps).  Registered as the ``'skein'``
realization of `pure_su2_object()`.
"""

from __future__ import annotations

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import RPowerSeries, TrivialZPlusRing, augmentation_hom

from cut_reglue import CutReglue
from triangulation import Triangulation

_ONE = LaurentPoly({0: 1})

K1 = (2, 1, 1, 0)           # the flavour-kernel units of the extended chart
K2 = (-1, 0, 0, 1)

# the measured stated-sector host table: (curve, eps_lo, eps_hi) ->
# canonical extended label a  (host gt = -a by the leading-term law)
HOSTS = {
    ((1, 1, 0), 1, 1): (1, 1, 0, 1),
    ((1, 1, 0), 1, -1): (1, 1, 0, -1),
    ((1, 1, 0), -1, 1): (-1, 1, 0, 1),
    ((1, 1, 0), -1, -1): (-1, -1, 0, -1),
    ((1, 0, 1), 1, 1): (1, 0, -1, 1),
    ((1, 0, 1), 1, -1): (1, 0, -1, -1),
    ((1, 0, 1), -1, 1): (-1, 0, -1, 1),
    ((1, 0, 1), -1, -1): (-1, 0, -1, -1),
}

# the measured core-loop arc image (χ₂ = L_{0,2}) and boundary-unit F
CORE_ARC = {(0, -1, -1, 0): _ONE, (0, -1, 1, 0): _ONE, (0, 1, 1, 0): _ONE}
F_E0 = {(1, 0, 0, 0): _ONE, (1, 0, 1, 0): _ONE, (1, 1, 1, 0): _ONE}


def P(u):
    """The flavour reduction ℤ⁴ → ℤ² (chart label → intrinsic γ):
    `P(u) = (u1 - u2, 2·u2 - u0 - u3)`; `ker P = ⟨K1, K2⟩` exactly."""
    return (u[1] - u[2], 2 * u[2] - u[0] - u[3])


def Gamma(g):
    """The section ℤ² → ℤ⁴ (`P∘Γ = id`): `Γ(γ) = (-γ1, γ0, 0, 0)`."""
    x, y = g
    return (-y, x, 0, 0)


def _det4(rows):
    """Integer determinant of a 4x4 matrix (Laplace, exact)."""
    def det(m):
        if len(m) == 1:
            return m[0][0]
        out = 0
        for j in range(len(m)):
            if m[0][j] == 0:
                continue
            minor = [r[:j] + r[j + 1:] for r in m[1:]]
            out += (-1) ** j * m[0][j] * det(minor)
        return out
    return det([list(r) for r in rows])


class SkeinAnnulusKAlg(KAlgebra):
    """The stated-annulus bordered-chart realization of pure SU(2), on
    the intrinsic BPS lower-tropical-charge labels γ ∈ ℤ²."""

    def __init__(self, bps=None):
        if bps is None:
            from pure_ade_lattice import pure_ade_kalgebra
            bps = pure_ade_kalgebra([("A", 1)])
        self._intrinsic = bps
        tt = Triangulation.from_edge_data(
            1, [(0, 0), (0, 0), (0, 0)], [(0, 1, 2), (0, 1, 2)])
        assert tt.genus == 1 and tt.n_punctures == 1
        self.cr = CutReglue(tt, 0)
        self.tc = self.cr.tc
        self.B = self.cr.B
        self._aug = augmentation_hom(self.B.coefficient_ring())
        self._certify_geometry()
        self._mult_cache = {}

    # ---- construction-time geometric certificates ---------------------------

    def _F(self, a):
        """The extended-chart canonical F(L_a) as a charge dict."""
        return dict(self.B.F(tuple(a)))

    def _certify_geometry(self):
        """Construction-time certificates (all ASSERT, all cheap): the
        chart shape, the P/Γ lattice split, the 8 stated-sector hosts,
        the χ₂ core loop, the χ₁ boundary-unit canonical, the μ-units,
        and the flagship products — every measured anchor of the fit."""
        tc, B, cr = self.tc, self.B, self.cr
        # chart shape
        assert tc.boundary_edge_ids == [0, 3], "boundary edges != [0, 3]"
        assert tc.holes == ((0,), (3,)), "holes != ((0,), (3,))"
        assert tc.internal_edge_ids == [1, 2], "internal edges != [1, 2]"
        assert tc.mutable_block() == [[0, 2], [-2, 0]], \
            "mutable block is not the Kronecker quiver"
        assert [tuple(s) for s in B.spec] == [(0, 1, 0, 0), (0, 0, 1, 0)], \
            "embedded 2-state spec mismatch"
        assert list(self._intrinsic.node_charges) == [(1, 0), (-1, 2)], \
            "intrinsic is not the Kronecker (1,0)/(-1,2) BPS chart"
        # the P/Γ lattice split: P∘Γ = id, K1/K2 ∈ ker P, and
        # {Γ(e1), Γ(e2), K1, K2} unimodular  ⇒  ker P = ⟨K1, K2⟩ exactly
        assert P(Gamma((1, 0))) == (1, 0) and P(Gamma((0, 1))) == (0, 1), \
            "P∘Γ != id"
        assert P(K1) == (0, 0) and P(K2) == (0, 0), "K1/K2 not in ker P"
        assert _det4([Gamma((1, 0)), Gamma((0, 1)), K1, K2]) in (1, -1), \
            "ker P != <K1, K2> (section+kernel not unimodular)"
        # (i) the 8 stated-sector hosts (leading-term law + bare
        # inclusion asserted inside cr.host)
        b_lo, b_hi = cr.BND
        for (curve, e1, e2), a in HOSTS.items():
            _A, found = cr.host(curve, {b_lo: e1, b_hi: e2})
            assert tuple(-x for x in found[0]) == a, (
                f"stated host mismatch for {curve} eps ({e1},{e2}): "
                f"{tuple(-x for x in found[0])} != {a}")
        # (ii) the core loop IS χ₂ = L_{0,2} (the adjoint Wilson):
        # arc image verbatim, inside F(L_{(0,-1,-1,0)}), class (0,-2)
        arc = cr.arc_image((0, 1, 1), {})
        assert arc == CORE_ARC, f"core-loop arc image mismatch: {arc}"
        f_aw = self._F((0, -1, -1, 0))
        assert all(ch in f_aw and f_aw[ch] == co for ch, co in arc.items()), \
            "core-loop arc is not the bare part of F(L_{(0,-1,-1,0)})"
        assert P((0, -1, -1, 0)) == (0, -2), "core-loop class != χ₂"
        # (iii) χ₁ = the boundary-unit canonical
        assert self._F((1, 0, 0, 0)) == F_E0, "F(L_{e0}) mismatch"
        # (iv) L_{2e0} = μ₁ · L_{(0,-1,-1,0)}  (F's shift by K1)
        f_2e0 = self._F((2, 0, 0, 0))
        shifted = {tuple(k + d for k, d in zip(ch, K1)): co
                   for ch, co in f_aw.items()}
        assert f_2e0 == shifted, "F(L_{2e0}) != X_{K1}·F(L_{(0,-1,-1,0)})"
        # (v) the μ-units: single unit monomials, μ₁·μ₁⁻¹ = 1
        neg = lambda u: tuple(-x for x in u)
        for u in (K1, neg(K1), K2, neg(K2)):
            assert self._F(u) == {u: _ONE}, f"F(L_{u}) is not the unit X_{u}"
        assert dict(B.multiply(K1, neg(K1)).terms) == {(0, 0, 0, 0): _ONE}, \
            "μ₁·μ₁⁻¹ != 1"
        # (vi) flagship products
        assert dict(B.multiply((1, 0, 0, 0), (1, 0, 0, 0)).terms) == \
            {(2, 0, 0, 0): _ONE, (2, 1, 1, 0): _ONE}, \
            "χ₁² != μ₁(χ₂ + 1) in the chart"
        assert dict(B.multiply((1, 1, 0, -1), (-1, 1, 0, 1)).terms) == \
            {(0, 2, 0, 0): _ONE}, "monopole-sector flagship failed"

    # ---- KAlgebra primitives --------------------------------------------------

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return (0, 0)

    def multiply(self, a, b) -> Element:
        """Computed GENUINELY in the certified bordered chart
        (`B.multiply` on the Γ-section labels), reduced by `P` with a
        class-collision guard, and ASSERTED verbatim equal to the
        intrinsic BPS product — the standing guard (δ == 0; never
        fitted, honest-fail on any mismatch)."""
        a, b = tuple(a), tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        ann = self.B.multiply(Gamma(a), Gamma(b))
        red = {}
        for lam, c in ann.terms.items():
            g = P(lam)
            assert g not in red, (
                f"class collision in {a}*{b}: two chart terms reduce to "
                f"{g} (none seen on the 336-term grid; guard loudly)")
            red[g] = c
        hint = self._intrinsic.multiply(a, b)
        assert red == dict(hint.terms), (
            f"chart product differs from intrinsic at {a}*{b}: "
            f"reduced {red} vs intrinsic {dict(hint.terms)} (delta != 0)")
        el = Element(red)
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        """NATIVE: `P(ρ_B(Γ(a)))` — measured equivariant with the
        intrinsic ρ on the 12-line grid (ρ_B's label carries a μ₁^c
        twist which `P` strips).  Not transported."""
        return P(self.B.rho(Gamma(tuple(a))))

    def rho_inverse(self, a):
        return P(self.B.rho_inverse(Gamma(tuple(a))))

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """NATIVE: the extended-chart trace at the Γ-section label,
        pushed through the flavour augmentation `ε : μ^w ↦ 1`
        (exact, coefficient-wise — the trace-side strip of the same
        μ₁^c unit twist that `P` strips from ρ's labels).  Measured
        equal to the intrinsic trace on all 12 grid lines at K=6."""
        raw = self.B.trace(Gamma(tuple(a)), K=K)
        return self._aug.apply_RPowerSeries(raw)

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: unflavoured
        (the chart's abelian flavour directions are the K1/K2 units,
        specialized away by `P`/ε)."""
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
                           name="pure-su2[skein-annulus→bps]")
