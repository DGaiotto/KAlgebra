"""`NativeLeEngine` — the native Lê construction + extensions as the
`SkeinKAlgebra` engine layer.

For a closed `Triangulation`, everything is built from the surface —
no BPS twin anywhere on the computational path (twins are certification
guards only):

  * **canonical F-supply** (`canonical_F`): the identity; ker-σ labels
    as central units (`F = X_m` — measured law on the flavour / Sk̃
    line-defect classes); simple-curve classes via the certified
    **cut-reglue** pipeline (`CutReglue` — the T-pin6 Lê-generalizing
    regluing rule, "forced, not fitted", = the graded RG solve run
    backwards) at an auto-chosen once-crossing cut edge.  Verified
    verbatim against the BPS F-solver on the tetrahedron channel
    curves (26/26 terms) — the native path never calls it.
  * **chart product** (`chart_mul`): the Chekhov–Fock torus
    `Q(σ_Δ)` cocycle arithmetic.
  * **canonical multiply** (`multiply`): product-and-peel — peel the
    chart product by the **leading-term law** (the lex-min support
    charge is the peeled label; its F has unit leading coefficient —
    asserted), with every candidate F built natively.  Honest-fail
    (never guess) when a candidate is out of the F-supply's certified
    scope.

THE RECORDED WALL (the gating build for the fully native multiply):
peel candidates with a normal coordinate ≥ 2 (e.g. the Dehn-twist
channel partners in the tetrahedron generator products, or any tower
class at the chart level) need the **multi-arc-per-triangle stated
primitive** — certified separately (elevation-ordered per-triangle
products, adversarially robust 1480/1480) but not yet productionized
into `curve_realization.realize_curve` / `CutReglue`.
Until that lands, `multiply` honest-fails on such products with a
diagnostic naming the wall; same-ray character towers do not need the
chart at all (the class computes them natively by Chebyshev).
"""

from __future__ import annotations

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from laurent_poly import LaurentPoly

from cut_reglue import CutReglue
from curve_realization import RealizationError


__all__ = ["NativeLeEngine", "NativeLeWallError"]

_ONE = LaurentPoly({0: 1})


class NativeLeWallError(NotImplementedError):
    """A native-Lê computation hit the multi-arc-per-triangle wall (the
    certified but not-yet-productionized stated primitive)."""


class NativeLeEngine:
    """The native Lê engine for a closed `Triangulation`.  See the
    module docstring."""

    def __init__(self, triangulation):
        self.tri = triangulation
        self.sigma = triangulation.sigma()
        self.n = len(self.sigma)
        from quantum_torus_kalgebra import QuantumTorusKAlg
        self.qt = QuantumTorusKAlg(self.sigma)
        self._cuts: dict = {}
        self._F_cache: dict = {}

    # ---- lattice helpers ----------------------------------------------------

    def in_kernel(self, m) -> bool:
        """Is `m` a flavour / line-defect-class charge (`σ·m = 0`)?"""
        return all(
            sum(self.sigma[i][j] * m[j] for j in range(self.n)) == 0
            for i in range(self.n)
        )

    # ---- the native canonical F-supply ---------------------------------------

    def canonical_F(self, label) -> dict:
        """`F(L_label)` as a chart-charge dict, built natively:

          * the identity → `{0: 1}`;
          * ker-σ labels → the central unit `X_label` (measured law:
            the flavour / spinor-class units are undressed);
          * labels whose negative is a simple-curve normal-coordinate
            vector with a once-crossing edge → the certified
            cut-reglue.

        Raises `NativeLeWallError` past the multi-arc wall and
        `ValueError` off the curve/unit scope — honest-fail, never a
        guess."""
        label = tuple(label)
        if label in self._F_cache:
            return self._F_cache[label]
        if not any(label):
            out = {label: _ONE}
        elif self.in_kernel(label):
            out = {label: _ONE}
        else:
            g = tuple(-x for x in label)
            if any(x < 0 for x in g):
                out = self._kernel_shifted_F(label)
                if out is None:
                    raise ValueError(
                        f"native F-supply: {label} is neither a ker-σ unit, "
                        f"nor (minus) a multicurve class, nor a bounded "
                        f"kernel shift of one — off the certified scope"
                    )
                self._F_cache[label] = out
                return out
            cut = next((i for i, x in enumerate(g) if x == 1), None)
            if cut is None:
                raise NativeLeWallError(
                    f"native F-supply: curve {g} has no once-crossing edge "
                    f"— multi-seam regluing is the recorded frontier"
                )
            if cut not in self._cuts:
                self._cuts[cut] = CutReglue(self.tri, cut)
            try:
                out, _hosts = self._cuts[cut].reglue(g)
            except RealizationError as exc:
                raise NativeLeWallError(
                    f"native F-supply at curve {g}: {exc} — the certified "
                    f"multi-arc-per-triangle primitive (scratch "
                    f"multiarc_*.py, 1480/1480) is not yet productionized "
                    f"into realize_curve/CutReglue; this is the gating "
                    f"build for the fully native Lê multiply"
                ) from exc
        self._F_cache[label] = out
        return out

    def _kernel_basis(self):
        if not hasattr(self, "_ker_b"):
            from snf_kernel import integer_kernel_and_section
            ker, _sec = integer_kernel_and_section(self.sigma)
            self._ker_b = [tuple(v) for v in ker]
        return self._ker_b

    def _kernel_shifted_F(self, label, radius: int = 2):
        """The flavour-orbit law `F(m) = X_k · F(m − k)` for `k ∈ ker σ`
        (kernel shifts are central chart units — the `BPSKAlgebra`
        F-cache orbit convention, here native): search kernel vectors
        with coefficients in `[-radius, radius]` for a shift making
        `−(m − k)` a multicurve class with a once-crossing edge; `None`
        if no bounded shift reaches one (the caller honest-fails)."""
        from itertools import product as iproduct
        kb = self._kernel_basis()
        for coeffs in iproduct(range(-radius, radius + 1), repeat=len(kb)):
            if not any(coeffs):
                continue
            k = tuple(sum(c * v[i] for c, v in zip(coeffs, kb))
                      for i in range(self.n))
            rest = tuple(m - x for m, x in zip(label, k))
            g = tuple(-x for x in rest)
            if any(x < 0 for x in g):
                continue
            if any(g) and not any(x == 1 for x in g):
                continue
            try:
                base = self.canonical_F(rest)
            except (NativeLeWallError, ValueError):
                continue
            return {tuple(gg + kk for gg, kk in zip(ch, k)): v
                    for ch, v in base.items()}
        return None

    # ---- chart arithmetic ------------------------------------------------------

    def chart_mul(self, P1: dict, P2: dict) -> dict:
        """The Chekhov–Fock torus product of two chart-charge dicts."""
        out: dict = {}
        e2 = Element({tuple(g): c for g, c in P2.items()})
        for g1, c1 in P1.items():
            prod = self.qt.multiply_elements(Element({tuple(g1): c1}), e2)
            for l, c in prod.terms.items():
                out[l] = (out[l] + c) if l in out else c
        return {k: v for k, v in out.items() if dict(v._coeffs)}

    # ---- the native canonical multiply ------------------------------------------

    def peel(self, P: dict) -> Element:
        """Peel a chart element into canonicals by the leading-term law
        (lex-min support charge; unit leading coefficient asserted),
        every candidate F built natively."""
        P = {tuple(k): v for k, v in P.items()}
        out: dict = {}
        guard = 0
        while P:
            guard += 1
            if guard > 10_000:
                raise RuntimeError("native peel: no convergence")
            cstar = min(P)
            F = self.canonical_F(cstar)
            # BOTH halves of the leading-term law: the label must be
            # the lex-min of its own F's
            # support (triangularity — otherwise the subtraction below
            # injects lex-smaller charges and mis-attributes them to
            # spurious canonicals), and the leading coefficient must be
            # the unit.
            assert cstar in F, (
                f"leading-term law violated at {cstar}: label absent "
                f"from its own F support {sorted(map(tuple, F))}")
            assert min(map(tuple, F)) == cstar, (
                f"leading-term law violated at {cstar}: F support "
                f"reaches lex-below the label "
                f"(min {min(map(tuple, F))})")
            lead = F[cstar]
            assert dict(lead._coeffs) == {0: 1}, (
                f"leading-term law violated at {cstar}: leading "
                f"coefficient {lead} != 1"
            )
            C = P[cstar]
            out[cstar] = C
            for l, v in F.items():
                l = tuple(l)
                P[l] = P.get(l, LaurentPoly.zero()) - C * v
            P = {k: v for k, v in P.items() if dict(v._coeffs)}
        return Element(out)

    def multiply(self, a, b) -> Element:
        """`L_a · L_b` natively: chart product of the native F's, then
        the native peel."""
        return self.peel(self.chart_mul(self.canonical_F(a),
                                        self.canonical_F(b)))
