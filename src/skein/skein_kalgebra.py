"""`SkeinKAlgebra(ConeKAlgebra)` — the unified skein realization on the
canonical surface.

Tier
----
The parent is `ConeKAlgebra` — *not* `RGKAlgebra`.  BPS / RG
auxiliaries are **produced** by factory
methods (`to_bpskalgebra()`), never inherited: the class stays
instantiable on surfaces with **no finite BPS chamber** (the N=2*
once-punctured torus is the honest boundary — there `to_bpskalgebra`
honest-fails while the skein surface itself exists).

Presentation
------------
The defining data is a `SkeinConeData` (`skein_cone_data.py`):

  * rays = simple non-self-intersecting curves/arcs;
  * Wilson / closed-curve rays are SU(2)-character generators, open-arc
    rays are monomial generators;
  * `cocycle` = the abstract σ̃ / Dirac pairing on ray charges;
  * `cross_product` = the certified skein engine's product, re-expressed
    as cone words (the standing guard lives in the engine).

`multiply` is inherited from `ConeKAlgebra` (the derived cone reducer
over engine-supplied cross-products) where the generic reducer serves
(pentagon); instances where it is known not to (pure SU(2)'s
Wilson-producing cross-cone Plückers — cf. `PureSU2KAlg`) override
`multiply` with the engine product directly, per the blessed
`SU2A1D3GammaKAlg` precedent ("a ConeKAlgebra need not use the Layer-1
trace" — nor, symmetrically, the derived multiply).

Trace
----------------------------------------------------------------------
`trace` is **delegated to a produced BPS auxiliary** (the
enforced-by-transport leg — a regression guard, not evidence).
`_trace_residual` carries the same delegation,
so the Layer-1 + Layer-2 factorization is in place for the
(N1)–(N3) / orthonormality bootstrap (the intrinsic closer) to land
seed-by-seed without changing the class shape.

First instances (both certified against their twins)
----------------------------------------------------------------------
  * `SkeinKAlgebra.pentagon()` — the A₂ pentagon on BPS chart labels
    ℤ²: five open-chord rays (all monomial), cones = the five
    compatible pairs, cocycle = the chart Dirac pairing, cross-products
    from the **pinned** engine (`PinnedPentagonKAlg`, ~17× faster than
    the localized route); `multiply` = the derived cone reducer.
  * `SkeinKAlgebra.annulus_pure_su2()` — A_𝖖[pure SU(2)] on the stated
    annulus (a *bordered*, non-sphere surface), on the
    intrinsic BPS lower-tropical labels γ ∈ ℤ²: the Wilson ray is the
    SU(2)-character generator χ₁ (its tower carries the closed core
    curve at level 2 — the measured χ₂ surprise), the 't Hooft rays
    H_n are open boundary-to-boundary arcs (monomial; the ρ-closed
    rank-3 H-cones of `pure_su2_h_cone_data`, an *infinite lazy* cone
    family); `multiply` = the `SkeinAnnulusKAlg` engine (standing
    guard); ρ and trace are the engine's native ones (the trace is the
    produced cut-chart `BPSKAlgebra` trace pushed through the flavour
    augmentation — itself an instance of the produced-auxiliary
    pattern).

Further instances: `SkeinKAlgebra.u1gauged_su2_nf1()`
(the verified ray dictionary as a cone; rgflow-anchored engine — the
honest exception while the stated μ/H arc engine is built) and
**`SkeinKAlgebra.sphere_su2_nf4()` — the fully NATIVE flagship**:
crossing-pair products from the intrinsic Kauffman bracelet engine and
the trace the intrinsic Schur–Askey–Wilson contour (no BPS transport on
any path of that instance).  Every instance carries structural
provenance flags (`engine_provenance` / `trace_provenance`) — making
auditable that the class owns a native skein engine; char towers are
always computed natively by the class (Chebyshev = Jones–Wenzl).

Flavour: Abelian by default (native kernel-Cartan flavour; KIsos to
flavour-reduced counterparts); the sphere instance carries its own
`R(Spin(8))` in from the absorbed `SU2Nf4SampleKAlgebra` (the tier
machinery is ring-agnostic; the flavour-in-labels lift passes through
`r_label_decompose_fn`).
"""

from __future__ import annotations

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Callable, Iterable, Optional

from bps_kalgebra import BPSKAlgebra
from cone_kalgebra import ConeKAlgebra
from kalgebra import Element, KAlgebra, Label
from zplus_ring import RPowerSeries, TrivialZPlusRing, ZPlusRing

from skein_cone_data import (IntrinsicBridgeSkeinConeData, SkeinConeData,
                              fraction_solve)


__all__ = [
    "SkeinKAlgebra",
    # the named instances (per-theory subclasses, repo-tier names — the
    # ConeKAlgebra-zoo convention U1Square / Pentagon / U1Hexagon /
    # Heptagon / Nonagon; engine route lives in the provenance flags,
    # never the name)
    "U1SquareSkeinKAlgebra",
    "PentagonSkeinKAlgebra",
    "U1HexagonSkeinKAlgebra",
    "HeptagonSkeinKAlgebra",
    "NonagonSkeinKAlgebra",
    "PureSU2AnnulusSkeinKAlgebra",
    "U1GaugedSU2Nf1SkeinKAlgebra",
    "SphereSU2Nf4SkeinKAlgebra",
    # the general family constructors: A1Aeven and U1A1Aodd, then
    # A1Dodd and U1A1Deven
    "A1AevenSkeinKAlgebra",
    "U1A1AoddSkeinKAlgebra",
    "A1DoddSkeinKAlgebra",
    "U1A1DevenSkeinKAlgebra",
    # new named polygon members over the general engines
    "U1OctagonSkeinKAlgebra",
    "U1DecagonSkeinKAlgebra",
    "HendecagonSkeinKAlgebra",
    # the SU(2)+N_f family over the certified bordered charts
    "SU2Nf1AnnulusSkeinKAlgebra",
    "SU2Nf2DiskSkeinKAlgebra",
    "SU2Nf3DiskSkeinKAlgebra",
    "SU2TwoPunctureDiskSkeinKAlgebra",
    "BorderedChartRaySkeinConeData",
    "ROSTER",
    # cone-data classes
    "PentagonSkeinConeData",
    "PureSU2AnnulusSkeinConeData",
    "U1GaugedSU2Nf1SkeinConeData",
    "SphereSU2Nf4SkeinConeData",
    "A1DoddSkeinConeData",
]


def _label_of(x) -> tuple:
    """Normalise a `rho`-style return (label or single-term Element) to a
    plain label tuple — the transported-ρ guard shared by the bridges.
    The coefficient IS part of the guard: a
    transported ρ returning ``q^n·L_l`` is exactly the mis-calibrated-frame
    failure mode, and truncating it to ``L_l`` would let the ρ-axiom
    verifiers pass with the wrong ρ."""
    if hasattr(x, "terms"):
        ts = dict(x.terms)
        assert len(ts) == 1, f"expected a single-label image, got {ts}"
        (lbl, coeff), = ts.items()
        assert str(coeff) == "1", (
            f"transported rho image carries a non-unit coefficient "
            f"{coeff} at {lbl} — a phase-dressed rho must not be "
            f"silently truncated")
        return tuple(lbl)
    return tuple(x)


# ===========================================================================
# The unified class
# ===========================================================================


class SkeinKAlgebra(ConeKAlgebra, BPSKAlgebra):
    """The unified skein realization: since `multiply` delegates to Lê's
    quantum-torus expression — native F's multiplied in `Q(σ_Δ)` and
    peeled — the class inherits both `ConeKAlgebra` and `BPSKAlgebra`.

    * The **cone parent** carries the presentation: rays = simple
      curves, char towers native (Chebyshev = Jones–Wenzl), the derived
      multiply where the reducer serves, the Layer-1/2 trace shape.
    * The **BPS parent** is the Lê-QT computational substrate in the
      *given-F* variant: when
      `chart=` data is supplied, `BPSKAlgebra.__init__` runs on
      `σ_Δ`-side data and the full chart machinery (F, Schur/Nahm
      trace, verifiers, caches) is live — with the spec OPTIONAL, so
      the chamber-free motivation survives: recipe/spec-free
      modes (or no chart at all) keep N=2*-type surfaces instantiable;
      a chartless instance is a cone-presented `SkeinKAlgebra` whose
      BPS-side methods honest-fail (`has_chart` is the flag, and
      `isinstance(·, BPSKAlgebra)` alone must not be read as "a
      functioning chart realisation").
    * The produced-auxiliary factories (`to_bpskalgebra`, cut
      auxiliaries) remain for the CUT charts (cut = node-drop RG).

    MRO: the cone side wins `multiply`/`trace` defaults (the
    `ConeRGKAlgebra` precedent); `multiply_via_chart` /
    `trace_via_chart` expose the BPS-side routes explicitly for
    cross-certification."""

    def __init__(
        self,
        cone_data: SkeinConeData,
        identity_label: Label,
        rho_fn: Callable[[Label], Label],
        rho_inverse_fn: Callable[[Label], Label],
        trace_fn: Callable[[Label, int], RPowerSeries],
        bps_factory: Optional[Callable[[], object]] = None,
        engine_multiply: Optional[Callable[[Label, Label], Element]] = None,
        rho2_rep_fn: Optional[Callable[[Label], Label]] = None,
        coefficient_ring: Optional[ZPlusRing] = None,
        name: str = "skein",
        r_label_decompose_fn: Optional[Callable] = None,
        r_label_compose_fn: Optional[Callable] = None,
        engine_provenance: str = "anchored",
        trace_provenance: str = "transported",
        chart: Optional[dict] = None,
        f_supply_fn: Optional[Callable[[Label], dict]] = None,
    ):
        self._cd = cone_data
        self._id = identity_label
        self._rho_fn = rho_fn
        self._rho_inv_fn = rho_inverse_fn
        self._trace_fn = trace_fn
        self._bps_factory = bps_factory
        self._bps_twin = None
        self._engine_multiply = engine_multiply
        self._rho2_rep_fn = rho2_rep_fn
        self._R = coefficient_ring or TrivialZPlusRing()
        self.name = name
        self._r_decomp_fn = r_label_decompose_fn
        self._r_comp_fn = r_label_compose_fn
        #: Structural honesty flags.
        #: `engine_provenance`: how crossing-pair products are certified —
        #: 'native-skein-kauffman' (the intrinsic geometric engine
        #: PRODUCES the data), 'stated-skein-pinned' /
        #: 'stated-skein-localized' / 'bordered-stated-chart' (the
        #: stated/pinned computation runs as a PER-CALL VERIFIER — the
        #: standing guard asserts every peeled coefficient against the
        #: intrinsic/BPS hint, whose objects are what is returned; read
        #: "verified against a stated computation", not "produced by
        #: one"), or '*-anchored' (an external
        #: certified twin — the honest exception, e.g. while a stated
        #: arc sector awaits its engine).  Char towers (colored parallel
        #: copies of one ray) are ALWAYS computed natively by the class
        #: (Chebyshev = the Jones–Wenzl skein calculus).
        #: `trace_provenance`: 'intrinsic-*' vs 'native-chart-*'
        #: (chart-native — the produced BPS auxiliary's trace, i.e. BPS
        #: transport, not skein-native) vs
        #: 'transported-*'.
        self.engine_provenance = engine_provenance
        self.trace_provenance = trace_provenance
        # ---- the optional BPS chart substrate: BPS inheritance with no
        # ---- spec is fine if F-finding is provided by cone+skein rays and
        # ---- S-finding can be delegated to the recursive S-builder from
        # ---- the BPS quiver ----
        self._f_supply = f_supply_fn
        self._f_supply_cache: dict = {}
        self.has_chart = chart is not None
        if chart is not None:
            BPSKAlgebra.__init__(self, **chart)

    # ---- KAlgebra primitives ---------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self) -> Label:
        return self._id

    def cone_data(self) -> SkeinConeData:
        return self._cd

    def multiply(self, a: Label, b: Label) -> Element:
        """Engine-overridden where the generic reducer is known not to
        serve; otherwise the inherited derived cone multiply (over
        engine-backed cross-products)."""
        if self._engine_multiply is not None:
            return self._engine_multiply(tuple(a), tuple(b))
        return super().multiply(tuple(a), tuple(b))

    def multiply_via_cone(self, a: Label, b: Label) -> Element:
        """The derived cone-reducer multiply, unconditionally — for
        certifying the cone presentation against the engine on
        instances whose `multiply` is engine-overridden."""
        return self._cd.derived_multiply(tuple(a), tuple(b))

    def _require_chart(self, what: str):
        if not self.has_chart:
            raise NotImplementedError(
                f"SkeinKAlgebra[{self.name}].{what}: no BPS chart substrate "
                f"was supplied (chart=None) — this instance is cone-presented "
                f"only; the BPS-side route needs chart data (spec optional)"
            )

    def inner_product(self, a: Label, b: Label, K: int = 20) -> RPowerSeries:
        """Pinned to the `KAlgebra` default (multiply-then-trace over
        THIS class's primitives): the BPS/RG parents' sharp Schur
        overrides need live chart attributes and must not capture
        chartless instances through the MRO.  The chart route is
        `inner_product_via_chart`."""
        return KAlgebra.inner_product(self, a, b, K)

    def _label_section_decompose(self, label):
        """Pinned to the `KAlgebra` derived bridge (reads this class's
        `r_label_decompose`) — the BPS parent's SNF-section override
        needs chart attributes."""
        return KAlgebra._label_section_decompose(self, label)

    def embed_R(self, r):
        """Pinned to the `KAlgebra` default — the BPS parent's override
        needs chart attributes."""
        return KAlgebra.embed_R(self, r)

    def cache_identity(self) -> str:
        """Pinned — the one remaining KAlgebra-API method that would
        otherwise resolve through the MRO to the BPS parent's
        chart-needing override, raising AttributeError on chartless
        instances.  The tag encodes the instance name (parametric —
        distinct instances must not share a cache file) and, when the
        chart substrate is live, the BPS quiver refinement.
        (The `hasattr` guard keeps this runnable against packages that
        predate the upstream caching layer.)"""
        tag = f"{type(self).__name__}[{self.name}]"
        if self.has_chart and hasattr(BPSKAlgebra, "cache_identity"):
            return tag + "|" + BPSKAlgebra.cache_identity(self)
        return tag

    def _F_internal(self, a):
        """The given-F hook (the F-finding half): when a native
        F-supply is wired (`f_supply_fn` — cone+skein rays: cut-reglue /
        char towers), the BPS chart machinery consumes IT instead of
        `solve_F`; conversion to the palindromic q-number form is exact
        and asserted.  Without a supply, the BPS parent's solve runs
        (needs the built/spec S)."""
        if self._f_supply is None:
            return BPSKAlgebra._F_internal(self, a)
        a = tuple(a)
        if a not in self._f_supply_cache:
            from q_number_poly import QNumberPoly
            nat = self._f_supply(a)
            # from_palindromic_laurent raises on non-palindromic input —
            # the bar-invariance guard on the supplied F, for free.
            self._f_supply_cache[a] = {
                tuple(g): QNumberPoly.from_palindromic_laurent(c)
                for g, c in nat.items()
            }
        return self._f_supply_cache[a]

    def inner_product_via_chart(self, a: Label, b: Label,
                                K: int = 20) -> RPowerSeries:
        """The BPS-parent Schur pairing (sharp single-Habiro path),
        exposed for chart-carrying instances."""
        self._require_chart("inner_product_via_chart")
        return BPSKAlgebra.inner_product(self, tuple(a), tuple(b), K)

    def multiply_via_chart(self, a: Label, b: Label) -> Element:
        """The BPS-parent multiply (the Lê-QT route through the chart
        machinery), exposed for cross-certification against the cone /
        engine routes."""
        self._require_chart("multiply_via_chart")
        return BPSKAlgebra.multiply(self, tuple(a), tuple(b))

    def trace_via_chart(self, a: Label, K: int = 20) -> RPowerSeries:
        """The BPS-parent trace (Schur/Nahm through the chart), exposed
        for cross-certification; needs a spec/recipe on the chart."""
        self._require_chart("trace_via_chart")
        return BPSKAlgebra.trace(self, tuple(a), K)

    def rho(self, a: Label) -> Label:
        return _label_of(self._rho_fn(tuple(a)))

    def rho_inverse(self, a: Label) -> Label:
        return _label_of(self._rho_inv_fn(tuple(a)))

    def _canonical_rho2_orbit_rep(self, label: Label) -> Label:
        if self._rho2_rep_fn is not None:
            return self._rho2_rep_fn(tuple(label))
        return super()._canonical_rho2_orbit_rep(label)

    # ---- trace: delegated to the produced auxiliary ------------------------

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """Delegated to the produced BPS auxiliary (enforced-by-transport
        — a regression guard, not evidence).  The
        intrinsic closer (the (N1)–(N3) / orthonormality bootstrap)
        replaces this seed-by-seed via `_trace_residual`
        when it lands."""
        return self._trace_fn(tuple(a), K)

    def _trace_residual(self, seed_label: Label, K: int) -> RPowerSeries:
        """Layer-2 seed trace — same delegation as `trace`, so the
        Layer-1 + Layer-2 path (`trace_via_cone`) is available and
        certifiable where the tagged-cyclicity reducer applies."""
        return self._trace_fn(tuple(seed_label), K)

    def trace_via_cone(self, a: Label, K: int = 20) -> RPowerSeries:
        """The `ConeKAlgebra` Layer-1 (tagged-cyclicity) + Layer-2 trace,
        unconditionally — for certifying the cone trace machinery
        against the delegated trace where Layer-1 applies (pentagon)."""
        return ConeKAlgebra.trace(self, tuple(a), K)

    # ---- produced auxiliaries (factories, never parents) --------------------

    def to_bpskalgebra(self):
        """The produced BPS auxiliary (memoized).  Honest-fails where no
        finite-chamber BPS twin exists (e.g. the N=2* once-punctured
        torus) — producing auxiliaries only where they exist is the
        point."""
        if self._bps_twin is None:
            if self._bps_factory is None:
                raise NotImplementedError(
                    f"SkeinKAlgebra[{self.name}]: no BPS auxiliary is "
                    f"registered for this surface (no finite BPS chamber "
                    f"is known — the honest boundary; cut auxiliaries / "
                    f"the chamber-free cut-reglue supply canonicals there)"
                )
            self._bps_twin = self._bps_factory()
        return self._bps_twin

    # ---- flavour lift (trivial default; flavour-in-labels instances pass
    # ---- their peel through the constructor) --------------------------------

    def r_label_decompose(self, label: Label):
        if self._r_decomp_fn is not None:
            return self._r_decomp_fn(label)
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        if self._r_comp_fn is not None:
            return self._r_comp_fn(section, r_basis_label)
        return tuple(section)

    # ---- iso witness ---------------------------------------------------------

    def build_iso(self, other=None):
        """Identity-label `KAlgebraIso` to the produced BPS auxiliary
        (or an explicit `other` sharing the label set)."""
        from kalgebra_iso import KAlgebraIso
        from laurent_poly import LaurentPoly
        unit = LaurentPoly({0: 1})
        target = other if other is not None else self.to_bpskalgebra()

        def fwd(lbl):
            return Element({tuple(lbl): unit})

        return KAlgebraIso(self, target, fwd, fwd,
                           name=f"{self.name}[cone→twin]")

    # ======================================================================
    # Named-instance constructors (thin aliases to the subclasses below —
    # every named instance is a subclass; the classmethods stay so
    # existing callers keep working)
    # ======================================================================

    @classmethod
    def pentagon(cls) -> "PentagonSkeinKAlgebra":
        """The A₂ pentagon — see `PentagonSkeinKAlgebra`."""
        return PentagonSkeinKAlgebra()

    @classmethod
    def sphere_su2_nf4(cls, channel: str = "a") -> "SphereSU2Nf4SkeinKAlgebra":
        """`Sk̃(S²₀,₄)` = SU(2)+N_f=4 — see `SphereSU2Nf4SkeinKAlgebra`."""
        return SphereSU2Nf4SkeinKAlgebra(channel=channel)

    @classmethod
    def u1gauged_su2_nf1(cls, engine=None) -> "U1GaugedSU2Nf1SkeinKAlgebra":
        """U(1)-gauged SU(2)+N_f=1 — see `U1GaugedSU2Nf1SkeinKAlgebra`."""
        return U1GaugedSU2Nf1SkeinKAlgebra(engine=engine)

    @classmethod
    def annulus_pure_su2(cls) -> "PureSU2AnnulusSkeinKAlgebra":
        """A_𝖖[pure SU(2)] on the stated annulus — see
        `PureSU2AnnulusSkeinKAlgebra`."""
        return PureSU2AnnulusSkeinKAlgebra()

    @classmethod
    def polygon(cls, n: int) -> "SkeinKAlgebra":
        """The n-marked polygon instance (the A1An roster).  Named
        members first: 4 → `U1SquareSkeinKAlgebra` (SQED1), 5 →
        `PentagonSkeinKAlgebra` ([A₁,A₂]), 6 → `U1HexagonSkeinKAlgebra`
        (gauged [A₁,A₃]), 7 → `HeptagonSkeinKAlgebra` ([A₁,A₄]), 8 →
        `U1OctagonSkeinKAlgebra` (gauged [A₁,A₅]), 9 →
        `NonagonSkeinKAlgebra` ([A₁,A₆]), 10 → `U1DecagonSkeinKAlgebra`
        (gauged [A₁,A₇]), 11 → `HendecagonSkeinKAlgebra` ([A₁,A₈]).
        Any other n ≥ 5 dispatches the GENERAL constructors (the
        generic-n unpin): odd n → `A1AevenSkeinKAlgebra`
        with k = (n−3)/2, even n → `U1A1AoddSkeinKAlgebra` with
        k = (n−4)/2 (even n beyond the frozen-table range k ≤ 3 builds
        its intrinsic through the RG oracle — constructible but slow)."""
        table = {4: U1SquareSkeinKAlgebra, 5: PentagonSkeinKAlgebra,
                 6: U1HexagonSkeinKAlgebra, 7: HeptagonSkeinKAlgebra,
                 8: U1OctagonSkeinKAlgebra, 9: NonagonSkeinKAlgebra,
                 10: U1DecagonSkeinKAlgebra, 11: HendecagonSkeinKAlgebra}
        if n in table:
            return table[n]()
        if n < 5:
            raise NotImplementedError(
                f"SkeinKAlgebra.polygon({n}): n must be >= 4"
            )
        if n % 2 == 1:
            return A1AevenSkeinKAlgebra((n - 3) // 2)
        return U1A1AoddSkeinKAlgebra((n - 4) // 2)

    @classmethod
    def a1dodd(cls, k: int = 0) -> "A1DoddSkeinKAlgebra":
        """`[A₁, D_{2k+3}]` (SU(2)-flavoured odd-D AD) on the
        `(2k+3)`-gon with a single regular interior puncture — see
        `A1DoddSkeinKAlgebra`."""
        return A1DoddSkeinKAlgebra(k)

    @classmethod
    def u1a1deven(cls, k: int = 1) -> "U1A1DevenSkeinKAlgebra":
        """The U(1)-gauged `[A₁, D_{2k+2}]` — see
        `U1A1DevenSkeinKAlgebra`."""
        return U1A1DevenSkeinKAlgebra(k)

    @classmethod
    def su2_nf(cls, nf: int) -> "SkeinKAlgebra":
        """The SU(2)+N_f family by flavour count (the surface
        dictionary): 0 → the (1,1) annulus
        (`PureSU2AnnulusSkeinKAlgebra`), 1 → the (1,2) annulus, 2 → the
        1-mark disk + 2 regular punctures, 3 → the 2-mark disk + 2
        regular punctures, 4 → the 4-punctured sphere
        (`SphereSU2Nf4SkeinKAlgebra`)."""
        table = {0: PureSU2AnnulusSkeinKAlgebra,
                 1: SU2Nf1AnnulusSkeinKAlgebra,
                 2: SU2Nf2DiskSkeinKAlgebra,
                 3: SU2Nf3DiskSkeinKAlgebra,
                 4: SphereSU2Nf4SkeinKAlgebra}
        if nf not in table:
            raise NotImplementedError(
                f"su2_nf({nf}): N_f must be 0..4 (conformal at 4)"
            )
        return table[nf]()

    @classmethod
    def two_puncture_disk(cls, n: int) -> "SkeinKAlgebra":
        """The disk with n boundary marks + 2 REGULAR interior
        punctures = SU(2) + 2 doublets + A1Dn (the family map):
        n = 1 → SU(2) N_f=2 (`SU2Nf2DiskSkeinKAlgebra`), n = 2 →
        SU(2) N_f=3 (`SU2Nf3DiskSkeinKAlgebra`), n ≥ 3 → the
        AD-matter-coupled `SU2TwoPunctureDiskSkeinKAlgebra(n)`."""
        if n < 1:
            raise NotImplementedError(
                f"two_puncture_disk({n}): n must be >= 1"
            )
        if n == 1:
            return SU2Nf2DiskSkeinKAlgebra()
        if n == 2:
            return SU2Nf3DiskSkeinKAlgebra()
        return SU2TwoPunctureDiskSkeinKAlgebra(n)

    @classmethod
    def punctured_polygon(cls, n: int) -> "SkeinKAlgebra":
        """The `[A₁, Dₙ]` family by its surface (A1Dn is a polygon with
        an extra regular puncture in the middle): the n-gon with a
        single regular interior
        puncture.  Odd n → `A1DoddSkeinKAlgebra((n−3)/2)` (SU(2)
        flavour at the puncture).  Even n honest-fails: only the
        **U(1)-gauged** even-D instance exists on the canonical surface
        (`u1a1deven((n−2)/2)` — puncture SU(2) kept, the extra U(1)
        gauged); the UNGAUGED rank-2-flavour `A1Deven` skein leg is
        open."""
        if n < 3:
            raise NotImplementedError(
                f"punctured_polygon({n}): [A₁, Dₙ] needs n >= 3"
            )
        if n % 2 == 1:
            return A1DoddSkeinKAlgebra((n - 3) // 2)
        raise NotImplementedError(
            f"punctured_polygon({n}): the ungauged [A₁, D_{n}] "
            f"(rank-2 flavour) has no skein leg yet — the U(1)-gauged "
            f"variant is SkeinKAlgebra.u1a1deven({(n - 2) // 2})"
        )


# ===========================================================================
# Pentagon cone data (five open-chord rays; finite)
# ===========================================================================


class PentagonSkeinConeData(SkeinConeData):
    """The pentagon's ray dictionary on BPS chart labels ℤ².

    Rays = the five chords' chart charges (the T̃-frame `h_p =
    ORBIT[p+1]` of `PinnedPentagonKAlg`), all open arcs ⇒ monomial.
    Cones = the five compatible pairs `{h_p, h_{p+2}}` (disjoint
    chords), whose spans are the five cluster cones covering ℤ².
    Cocycle = the chart Dirac pairing `⟨g, h⟩ = g₀h₁ − g₁h₀` (the
    abstract σ̃ pairing).  Cross-products (crossing chords)
    = the pinned engine's products (guarded against the BPS twin inside
    the engine)."""

    def __init__(self, bps=None):
        from skein_pentagon_kalg import ORBIT
        self._h = [tuple(ORBIT[(p + 1) % 5]) for p in range(5)]
        self._cone_sets = tuple(
            frozenset({self._h[p], self._h[(p + 2) % 5]}) for p in range(5)
        )
        if bps is None:
            from bps_kalgebra import BPSKAlgebra
            bps = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                              node_charges=[(1, 0), (0, 1)])
        self._twin = bps
        self._engine = None

    def bps_twin(self):
        return self._twin

    def _pinned_engine(self):
        if self._engine is None:
            from pinned_pentagon_kalg import PinnedPentagonKAlg
            self._engine = PinnedPentagonKAlg(bps=self._twin)
        return self._engine

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        return "monomial"                     # all five rays are open chords

    def mult_gens(self):
        """The global finite ray set (the `FiniteConeData` surface —
        object-layer sample harvesting reads this)."""
        return tuple(self._h)

    def cone_sets(self):
        return self._cone_sets

    def engine_product(self, g, h) -> Element:
        return self._pinned_engine().multiply(tuple(g), tuple(h))

    # -- ConeData surface -------------------------------------------------------

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h:
            return True
        pair = frozenset({g, h})
        return any(pair <= c for c in self._cone_sets)

    def cocycle(self, g, h) -> int:
        return g[0] * h[1] - g[1] * h[0]

    def to_cone_label(self, native_label):
        gamma = tuple(native_label)
        if gamma == (0, 0):
            return frozenset(), {}
        for p in range(5):
            g1, g2 = self._h[p], self._h[(p + 2) % 5]
            det = g1[0] * g2[1] - g1[1] * g2[0]
            if det * det != 1:
                continue
            a = (gamma[0] * g2[1] - gamma[1] * g2[0]) * det
            b = (g1[0] * gamma[1] - g1[1] * gamma[0]) * det
            if a >= 0 and b >= 0:
                powers = {}
                if a > 0:
                    powers[g1] = a
                if b > 0:
                    powers[g2] = b
                return frozenset(powers), powers
        raise ValueError(f"pentagon: label {gamma} not in any cluster cone")

    def from_cone_label(self, gens, powers):
        x = y = 0
        for g, p in powers.items():
            if p:
                x += g[0] * p
                y += g[1] * p
        return (x, y)


# ===========================================================================
# U(1)-gauged SU(2)+N_f=1 cone data (the verified ray dictionary)
# ===========================================================================


class U1GaugedSU2Nf1SkeinConeData(SkeinConeData):
    """The ray-sector cone data of U(1)-gauged SU(2)+N_f=1, on the
    `SU2Nf1PureSU2RGKAlgebra` labels `(su2_label, (a, b))`.

    Rays (the verified dictionary; fan (1,2)-annulus):
    `W = (w₁,(0,0))` char (core loop); `μ^{±1} = ((),(0,±1))` torus
    (formal monodromy — adjoined, inverse pair); `H/H⁻¹ = ((),(±1,0))`
    monomial (minuscule 't Hooft arcs).  Two cones
    `{W, μ^{±1}, H}` / `{W, μ^{±1}, H⁻¹}`; the non-q-commuting pairs
    are the torus cancellation `(μ, μ⁻¹) → 1` and the matter meson
    `(H, H⁻¹) → 1 + q·(w₁,μ) + q²·μ²`, both engine cross-products.
    Cocycle = the gauge-QT Dirac pairing `⟨(a,b),(a′,b′)⟩ = ab′ − ba′`
    (the su2 Wilson part is central — contributes 0).

    Certified scope of the label bijection: the ray sector — su2 part a
    Wilson `w_k` (or the identity), any gauge charge `(a, b)`.  Labels
    with a dressed su2-monopole part honest-fail (`ValueError`) — off
    the ray dictionary's certified scope."""

    def __init__(self):
        self._W = (((("W", 1), 1),), (0, 0))
        self._MU_P = ((), (0, 1))
        self._MU_N = ((), (0, -1))
        self._H = ((), (1, 0))
        self._F = ((), (-1, 0))
        self._cone_sets = (
            frozenset({self._W, self._MU_P, self._MU_N, self._H}),
            frozenset({self._W, self._MU_P, self._MU_N, self._F}),
        )
        self._rank = {self._W: 0, self._MU_P: 1, self._MU_N: 2,
                      self._H: 3, self._F: 4}
        self._engine_inst = None

    def engine(self):
        if self._engine_inst is None:
            from su2nf1_pure_su2_rgkalgebra import SU2Nf1PureSU2RGKAlgebra
            self._engine_inst = SU2Nf1PureSU2RGKAlgebra()
        return self._engine_inst

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        g = tuple(g)
        if g == self._W:
            return "char"
        if g in (self._MU_P, self._MU_N):
            return "torus"
        return "monomial"

    def cone_sets(self):
        return self._cone_sets

    def engine_product(self, g, h) -> Element:
        return self.engine().multiply(tuple(g), tuple(h))

    def _torus_inverse_letter(self, g):
        if tuple(g) == self._MU_P:
            return self._MU_N
        if tuple(g) == self._MU_N:
            return self._MU_P
        return None

    # -- ConeData surface -------------------------------------------------------

    @staticmethod
    def _su2_level(su2_part) -> int:
        """The Wilson χ-level of the su2 tensor factor; honest-fail on
        dressed su2-monopole content (off the ray sector)."""
        if su2_part == ():
            return 0
        if (len(su2_part) == 1 and isinstance(su2_part[0][0], tuple)
                and su2_part[0][0][0] == "W" and su2_part[0][1] == 1):
            return su2_part[0][0][1]
        raise ValueError(
            f"u1gauged ray sector: su2 part {su2_part!r} is not a Wilson "
            f"χ_k — off the certified ray-dictionary scope"
        )

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h:
            return True
        pair = {g, h}
        if pair == {self._MU_P, self._MU_N}:
            return False                       # μ·μ⁻¹ = 1: engine cross
        if pair == {self._H, self._F}:
            return False                       # the matter meson
        return True

    def cocycle(self, g, h) -> int:
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        (_, (a, b)), (_, (a2, b2)) = tuple(g), tuple(h)
        return a * b2 - b * a2

    def canonical_cone_order(self, gens):
        return tuple(sorted(gens, key=lambda g: self._rank[tuple(g)]))

    def to_cone_label(self, native_label):
        su2, (a, b) = native_label
        k = self._su2_level(tuple(su2))
        powers = {}
        if k > 0:
            powers[self._W] = k
        if a > 0:
            powers[self._H] = a
        elif a < 0:
            powers[self._F] = -a
        if b > 0:
            powers[self._MU_P] = b
        elif b < 0:
            powers[self._MU_N] = -b
        return frozenset(powers), powers

    def from_cone_label(self, gens, powers):
        k = powers.get(self._W, 0)
        a = powers.get(self._H, 0) - powers.get(self._F, 0)
        b = powers.get(self._MU_P, 0) - powers.get(self._MU_N, 0)
        su2 = () if k == 0 else ((("W", k), 1),)
        return (su2, (a, b))


# ===========================================================================
# S²₀,₄ = SU(2)+N_f=4 cone data (native Kauffman engine; intrinsic AW trace)
# ===========================================================================


class SphereSU2Nf4SkeinConeData(SkeinConeData):
    """The closed 4-punctured sphere's ray dictionary on the
    `SU2Nf4SampleKAlgebra` character labels.

    Rays = the primitive curve slopes (closed simple curves ⇒ every ray
    an SU(2)-character generator; an **infinite lazy family** — any two
    distinct essential simple closed curves on `S²₀,₄` intersect, so
    every cone is the singleton char cone of one slope, constructed
    directly from the label).  Cross-ray products are **native Kauffman
    resolutions** (the engine's bracelet peel).  Cone-label bijection
    scope: the flavour-section (undressed) sector; Spin(8)-dressed
    labels honest-fail here — their lift coordinate is
    `r_label_decompose`, and flavoured cone-words are recorded future
    work — so the derived cone path serves same-ray towers and
    honest-fails (never guesses) on cross-ray words."""

    _ZERO_W = (0, 0, 0, 0)

    def __init__(self, channel: str = "a"):
        self._channel = channel
        self._engine_inst = None

    def engine(self):
        if self._engine_inst is None:
            from su2_nf4_sample_kalgebra import SU2Nf4SampleKAlgebra
            self._engine_inst = SU2Nf4SampleKAlgebra(channel=self._channel)
        return self._engine_inst

    def _ray(self, slope):
        return ("C", tuple(slope), 1, self._ZERO_W)

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        return "char"                    # every ray is a closed simple curve

    def cone_sets(self):
        # A finite window of the infinite slope family (the six channel /
        # first-Dehn-twist slopes); `_cone_for_gens` constructs any other
        # slope's singleton cone directly from the label.
        for s in self.engine().SLOPES.values():
            yield frozenset({self._ray(s)})

    def engine_product(self, g, h) -> Element:
        return self.engine().multiply(tuple(g), tuple(h))

    def _cone_for_gens(self, gens_fs):
        gens_fs = frozenset(tuple(g) for g in gens_fs)
        if len(gens_fs) != 1:
            raise ValueError(
                f"S²₀,₄: distinct slopes always cross — no multi-ray cone "
                f"contains {set(gens_fs)}"
            )
        from cone_data import Cone
        return Cone(self, gens_fs, char_gens=gens_fs)

    def cone_of_label(self, native_label):
        gens_fs, _ = self.to_cone_label(native_label)
        if not gens_fs:
            gens_fs = frozenset({self._ray(next(iter(
                self.engine().SLOPES.values())))})
        return self._cone_for_gens(gens_fs)

    # -- ConeData surface -------------------------------------------------------

    def q_commute(self, g, h) -> bool:
        return tuple(g) == tuple(h)      # distinct slopes always cross

    def cocycle(self, g, h) -> int:
        if tuple(g) != tuple(h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        return 0

    def to_cone_label(self, native_label):
        lbl = tuple(native_label)
        if lbl == ("I",):
            return frozenset(), {}
        if lbl[0] == "C" and lbl[3] == self._ZERO_W:
            ray = self._ray(lbl[1])
            return frozenset({ray}), {ray: lbl[2]}
        raise ValueError(
            f"S²₀,₄ cone-label bijection: {lbl!r} is Spin(8)-dressed (or a "
            f"dressed identity) — off the flavour-section sector; its lift "
            f"coordinate is r_label_decompose (flavoured cone-words are "
            f"recorded future work)"
        )

    def from_cone_label(self, gens, powers):
        if not gens:
            return ("I",)
        (ray,) = tuple(gens)
        k = powers[ray]
        if k == 0:
            return ("I",)
        return ("C", ray[1], k, self._ZERO_W)


# ===========================================================================
# Pure-SU(2) annulus cone data (χ₁ character ray + infinite lazy H-cones)
# ===========================================================================


class PureSU2AnnulusSkeinConeData(SkeinConeData):
    """The pure-SU(2) ray dictionary on the intrinsic BPS
    lower-tropical labels γ ∈ ℤ² (the `SkeinAnnulusKAlg` labels).

    Rays: the Wilson χ₁ (a Wilson line ⇒ SU(2)-character generator;
    its geometric avatar is the conjectural stated boundary
    arc, with the honest closed core curve sitting at level 2 of its
    tower — the measured χ₂ surprise) and the 't Hooft `H_n = L_{1,n}`
    (open boundary-to-boundary arcs ⇒ monomial).  Cones: the singleton
    Wilson character cone plus the ρ-closed rank-3 H-cones
    `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}`, k ∈ ℤ — an **infinite, lazily
    enumerated** family (`cone_of_label` constructs the cone directly
    from the label's own rays).  Cocycle = the (m,e) Dirac pairing
    `m·e′ − e·m′` (abstract σ̃).  Label bijection = the
    max-diagonal convention of `pure_su2_h_cone_data`."""

    def __init__(self):
        from abelianized_su2_bps_iso import gamma_of, me_of
        from pure_su2_h_cone_data import (
            _native_to_psu2, _psu2_to_native, cone_index_for,
        )
        # plain functions on instance attributes stay unbound
        self.gamma_of = gamma_of
        self.me_of = me_of
        self._to_native = _psu2_to_native
        self._from_native = _native_to_psu2
        self._cone_index_for = cone_index_for
        self._W = gamma_of(0, 1)
        self._engine_inst = None

    def engine(self):
        if self._engine_inst is None:
            from skein_annulus_kalg import SkeinAnnulusKAlg
            self._engine_inst = SkeinAnnulusKAlg()
        return self._engine_inst

    def _H(self, n: int):
        return self.gamma_of(1, n)

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        return "char" if tuple(g) == self._W else "monomial"

    def cone_sets(self):
        yield frozenset({self._W})
        yield self._h_cone_set(0)
        k = 1
        while True:
            yield self._h_cone_set(k)
            yield self._h_cone_set(-k)
            k += 1

    def _h_cone_set(self, k: int) -> frozenset:
        return frozenset({self._H(2 * k), self._H(2 * k + 1),
                          self._H(2 * k + 2)})

    def engine_product(self, g, h) -> Element:
        return self.engine().multiply(tuple(g), tuple(h))

    def _cone_for_gens(self, gens_fs):
        """Direct construction — the infinite lazy family must never be
        scanned.  The label knows its own rays."""
        from cone_data import Cone
        gens_fs = frozenset(tuple(g) for g in gens_fs)
        if gens_fs == frozenset({self._W}):
            return Cone(self, gens_fs, char_gens=gens_fs)
        mes = sorted(self.me_of(*g) for g in gens_fs)
        assert all(m == 1 for m, _ in mes), (
            f"annulus cone lookup: mixed / non-ray gens {set(gens_fs)}"
        )
        ns = [e for _, e in mes]
        k = self._cone_index_for(ns[0], ns[-1])
        if k is None:
            raise ValueError(
                f"annulus: gens {set(gens_fs)} fit no single H-cone"
            )
        return Cone(self, self._h_cone_set(k))

    # -- ConeData surface -------------------------------------------------------

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h:
            return True
        mg, eg = self.me_of(*g)
        mh, eh = self.me_of(*h)
        if mg == 0 and mh == 0:
            return True                        # Wilson–Wilson
        if mg == 0 or mh == 0:
            return False                       # Wilson–H: crossing
        return self._cone_index_for(min(eg, eh), max(eg, eh)) is not None

    def cocycle(self, g, h) -> int:
        mg, eg = self.me_of(*tuple(g))
        mh, eh = self.me_of(*tuple(h))
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        return mg * eh - eg * mh

    def canonical_cone_order(self, gens):
        return tuple(sorted(gens, key=lambda g: self.me_of(*tuple(g))))

    def to_cone_label(self, native_label):
        m, e = self.me_of(*tuple(native_label))
        if m == 0:
            if e == 0:
                return frozenset(), {}
            if e < 0:
                raise ValueError(
                    f"annulus: no canonical label at (m, e) = (0, {e})"
                )
            return frozenset({self._W}), {self._W: e}
        powers = {}
        for (n, exp) in self._to_native(m, e):
            powers[self._H(n)] = exp
        return frozenset(powers), powers

    def from_cone_label(self, gens, powers):
        if not gens:
            return (0, 0)
        gens = frozenset(tuple(g) for g in gens)
        if gens == frozenset({self._W}):
            return self.gamma_of(0, powers[self._W])
        m = e = 0
        for g, p in powers.items():
            if not p:
                continue
            mg, eg = self.me_of(*tuple(g))
            assert mg == 1, f"annulus from_cone_label: non-ray gen {g}"
            m += p
            e += eg * p
        return self.gamma_of(m, e)


# ===========================================================================
# The named-instance subclasses; naming follows the repo-tier
# convention U1Square / Pentagon / U1Hexagon / Heptagon /
# Nonagon — the ConeKAlgebra-zoo names; the engine route stays in the
# provenance flags, never the class name
# ===========================================================================


class PentagonSkeinKAlgebra(SkeinKAlgebra):
    """The A₂ pentagon on BPS chart labels ℤ² — five open-chord rays,
    derived cone multiply over pinned-engine cross-products
    (`PinnedPentagonKAlg`, ~17× faster than the localized route);
    ρ/trace transported from the produced BPS twin; carries the A₂
    chart substrate (spec mode).

    `bps`: optionally supply the `BPSKAlgebra` twin to anchor on (so
    object-layer iso endpoints share instance identity)."""

    def __init__(self, bps=None):
        cd = PentagonSkeinConeData(bps=bps)
        twin = cd.bps_twin()
        self.intrinsic = twin
        super().__init__(
            cone_data=cd,
            identity_label=(0, 0),
            rho_fn=twin.rho,
            rho_inverse_fn=twin.rho_inverse,
            trace_fn=lambda a, K: twin.trace(a, K),
            bps_factory=lambda: twin,
            engine_multiply=None,          # derived cone multiply is the path
            name="skein-pentagon-cone",
            engine_provenance="stated-skein-pinned",
            trace_provenance="transported-bps",
            chart=dict(pairing=[[0, 1], [-1, 0]],
                       node_charges=[(1, 0), (0, 1)]),
        )


class _BridgedPolygonSkeinKAlgebra(SkeinKAlgebra):
    """Shared shell for the polygon instances riding
    `IntrinsicBridgeSkeinConeData`: cone combinatorics from the
    intrinsic realisation's certified `ConeData`, cross-products from
    the certified skein bridge engine, ρ/trace transported from the
    intrinsic (the registered object-layer instance where supplied, so
    iso witness endpoints match).  `multiply` is the derived cone
    reducer — the certification content is exactly "cone reducer over
    SKEIN cross-products ≡ intrinsic"."""

    def __init__(self, intrinsic, engine, name: str,
                 engine_provenance: str, rho2_rep_fn=None):
        from skein_cone_data import IntrinsicBridgeSkeinConeData
        self.intrinsic = intrinsic
        self.engine = engine
        cd = IntrinsicBridgeSkeinConeData(intrinsic.cone_data(),
                                          engine.multiply)
        super().__init__(
            cone_data=cd,
            identity_label=intrinsic.identity(),
            rho_fn=intrinsic.rho,
            rho_inverse_fn=intrinsic.rho_inverse,
            trace_fn=lambda a, K: intrinsic.trace(a, K),
            bps_factory=None,              # BPS twins live on the object
                                           # layer with their own frames
            engine_multiply=None,          # derived cone multiply is the path
            rho2_rep_fn=rho2_rep_fn,
            name=name,
            engine_provenance=engine_provenance,
            trace_provenance="transported-intrinsic",
        )


class U1SquareSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """SQED1 = the U(1)-gauged square, on the `(m, n) ∈ ℤ²` labels —
    the even-marked / gauged corner of the A1An roster (k = 0 of
    `U1A1AoddKAlg`).  Cone combinatorics from `U1SquareKAlg`'s QTCone
    presentation; cross-products from the PINNED square engine
    (`SkeinSquareKAlg` — the dressed diagonals keep residual pins ±μ,
    δ ≡ 0 against the intrinsic on every product).  ρ² is the shear
    `(m, n) ↦ (m, n + m)` — closed-form drift quotient `n mod |m|`."""

    def __init__(self, cone=None, engine=None):
        if cone is None:
            from u1_square_kalg import U1SquareKAlg
            cone = U1SquareKAlg()
        if engine is None:
            from skein_square_kalg import SkeinSquareKAlg
            engine = SkeinSquareKAlg()
        super().__init__(
            intrinsic=cone,
            engine=engine,
            name="skein-u1square-cone",
            engine_provenance="stated-skein-pinned",
            rho2_rep_fn=cone._canonical_rho2_orbit_rep,
        )


class U1HexagonSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """The U(1)-gauged hexagon (gauged [A₁,A₃]) on the
    `U1A1AoddKAlg(1)` labels `((factors), e_E)` — 9 chord rays
    (monomial) + the gauge unit `E`/`E⁻¹` (torus pair, read off the
    intrinsic's `_torus_inverse_letter`).  Cross-products from the
    localized stated engine (`SkeinU1HexagonKAlg` — zero-remainder
    engine identity, no fitted normalization).  ρ² drifts the E-power
    on chords — the intrinsic's closed-form drift quotient is passed
    through."""

    def __init__(self, intrinsic=None, engine=None):
        if intrinsic is None:
            from u1a1aodd_kalg import U1A1AoddKAlg
            intrinsic = U1A1AoddKAlg(1)
        if engine is None:
            from skein_u1hexagon_kalg import SkeinU1HexagonKAlg
            engine = SkeinU1HexagonKAlg(intrinsic=intrinsic)
        super().__init__(
            intrinsic=intrinsic,
            engine=engine,
            name="skein-u1hexagon-cone",
            engine_provenance="stated-skein-localized",
            rho2_rep_fn=intrinsic._canonical_rho2_orbit_rep,
        )


class HeptagonSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """The [A₁,A₄] heptagon on the `A1A2kKAlg(2)` labels — 14 open-chord
    rays (two families: `(1,i)` shorts / `(2,i)` longs), all monomial;
    cross-products from the localized stated engine
    (`SkeinHeptagonKAlg`; its one-time δ-calibration ≈ 1 min runs on
    the first crossing product).  ρ has finite order (i ↦ i+1 mod 7),
    so the default ρ²-orbit walk serves.  Trace = the intrinsic
    two-layer closed form (M(2,7) Andrews–Gordon characters)."""

    def __init__(self, intrinsic=None, engine=None):
        if intrinsic is None:
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(2)
        if engine is None:
            from skein_heptagon_kalg import SkeinHeptagonKAlg
            engine = SkeinHeptagonKAlg(intrinsic=intrinsic)
        super().__init__(
            intrinsic=intrinsic,
            engine=engine,
            name="skein-heptagon-cone",
            engine_provenance="stated-skein-localized",
        )


class NonagonSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """The [A₁,A₆] nonagon on the `A1A2kKAlg(3)` labels — three chord
    families, all monomial rays; cross-products from the PINNED stated
    engine (`SkeinNonagonKAlg` — pinned-torus arithmetic, fast).
    ρ finite order (mod 9).  Trace = the intrinsic two-layer closed
    form (M(2,9) characters)."""

    def __init__(self, intrinsic=None, engine=None):
        if intrinsic is None:
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(3)
        if engine is None:
            from skein_nonagon_kalg import SkeinNonagonKAlg
            engine = SkeinNonagonKAlg(intrinsic=intrinsic)
        super().__init__(
            intrinsic=intrinsic,
            engine=engine,
            name="skein-nonagon-cone",
            engine_provenance="stated-skein-pinned",
        )


class SphereSU2Nf4SkeinKAlgebra(SkeinKAlgebra):
    """`Sk̃(S²₀,₄)` = SU(2)+N_f=4 (conformal) — the **fully native**
    flagship instance: crossing-pair products from the intrinsic
    Kauffman engine (`SU2Nf4SampleKAlgebra`'s bracelet peel — total:
    any slopes/levels, `𝖖 = A²` with the `A² = −𝖖` sign), and the trace
    the **intrinsic Schur–Askey–Wilson contour functional** (the first
    instance whose trace is NOT transported — the trace weak point
    closed *for this instance*).

    Rays = the primitive curve slopes (infinite lazy family; every ray
    a closed simple curve ⇒ SU(2)-character generator, one singleton
    char cone per slope).  Labels = `("C", slope, k, dynkin)` with the
    Spin(8) flavour dressing in the label; coefficient ring =
    `R(Spin(8))`.  `ρ = id` (conformal).  No BPS twin on these labels
    (`to_bpskalgebra` honest-fails; chart twins live on `s2_04_object()`
    with their own frames)."""

    def __init__(self, channel: str = "a"):
        cd = SphereSU2Nf4SkeinConeData(channel=channel)
        eng = cd.engine()
        self.intrinsic = eng
        self.engine = eng
        super().__init__(
            cone_data=cd,
            identity_label=("I",),
            rho_fn=eng.rho,
            rho_inverse_fn=eng.rho,          # ρ² = id (self-dual curves)
            trace_fn=lambda a, K: eng.trace(a, K),
            bps_factory=None,
            engine_multiply=eng.multiply,    # the native Kauffman engine
            rho2_rep_fn=lambda l: tuple(l),  # ρ² = id
            coefficient_ring=eng.coefficient_ring(),
            name="skein-sphere-su2-nf4",
            r_label_decompose_fn=eng.r_label_decompose,
            r_label_compose_fn=eng.r_label_compose,
            engine_provenance="native-skein-kauffman",
            trace_provenance="intrinsic-schur-aw",
        )


class U1GaugedSU2Nf1SkeinKAlgebra(SkeinKAlgebra):
    """U(1)-gauged SU(2)+N_f=1 on the verified ray dictionary of the
    fan (1,2)-annulus.  Rays: the SU(2) Wilson `W` (char), the formal
    monodromy `μ^{±1}` (adjoined torus pair), the minuscule 't Hooft
    arcs `H`/`H⁻¹` (monomial; the matter meson is an engine
    cross-product).  Engine + ρ + trace = the
    `SU2Nf1PureSU2RGKAlgebra` anchor (the honest exception while the
    stated μ/H arc engine is built).  `to_bpskalgebra()` honest-fails:
    no BPS chart is registered for this global form."""

    def __init__(self, engine=None):
        cd = U1GaugedSU2Nf1SkeinConeData()
        if engine is not None:
            cd._engine_inst = engine
        eng = cd.engine()
        self.intrinsic = eng
        self.engine = eng

        def _rho2_rep(label):
            # ρ² is the gauge shear (a, b) ↦ (a, b + 2a): infinite orbits
            # for a ≠ 0 — closed-form drift quotient b mod 2|a|.
            s, (a, b) = label
            if a == 0:
                return (s, (a, b))
            return (s, (a, b % (2 * abs(a))))

        super().__init__(
            cone_data=cd,
            identity_label=((), (0, 0)),
            rho_fn=eng.rho,
            rho_inverse_fn=eng.rho_inverse,
            trace_fn=lambda a, K: eng.trace(a, K),
            bps_factory=None,              # the honest boundary: no BPS
                                           # chart registered for this
                                           # global form
            engine_multiply=eng.multiply,  # the RG anchor is the engine
            rho2_rep_fn=_rho2_rep,
            name="skein-u1gauged-su2-nf1-cone",
            engine_provenance="rgflow-anchored",   # the honest exception:
                                                   # the stated μ/H arc
                                                   # engine is the frontier
            trace_provenance="transported-rgflow",
        )


class PureSU2AnnulusSkeinKAlgebra(SkeinKAlgebra):
    """A_𝖖[pure SU(2)] on the stated annulus (bordered surface) — χ₁
    character ray + the infinite lazy H-cone family; multiply = the
    `SkeinAnnulusKAlg` engine (standing guard); ρ/trace native (the
    trace is the produced cut-chart BPS trace through the flavour
    augmentation)."""

    def __init__(self):
        cd = PureSU2AnnulusSkeinConeData()
        eng = cd.engine()
        self.intrinsic = eng
        self.engine = eng

        def _rho2_rep(label):
            # ρ²(m,e) = (m, e−8m): infinite orbits for m ≥ 1 — closed-form
            # drift quotient (contract requirement; cf. PureSU2KAlg).
            m, e = cd.me_of(*label)
            if m == 0:
                return tuple(label)
            return cd.gamma_of(m, e % (8 * m))

        super().__init__(
            cone_data=cd,
            identity_label=(0, 0),
            rho_fn=eng.rho,
            rho_inverse_fn=eng.rho_inverse,
            trace_fn=lambda a, K: eng.trace(a, K),
            bps_factory=lambda: eng._intrinsic,
            engine_multiply=eng.multiply,  # the generic reducer is known
                                           # not to serve pure SU(2)
            rho2_rep_fn=_rho2_rep,
            name="skein-annulus-cone",
            engine_provenance="bordered-stated-chart",
            trace_provenance="native-chart-augmentation",
        )


# ===========================================================================
# The GENERAL family constructors.  The polygon
# families ride the generic-n PINNED engines (`skein_oddgon_kalg` /
# `skein_evengon_kalg` — unpin words closed-form/solved, gauges
# calibrated/solved at construction, standing guard verbatim); the
# D-families are cone bridges over the self-contained D-frame intrinsics
# with the χ-tower native (Chebyshev = Jones–Wenzl) and the arc
# cross-products anchored on the intrinsic closed-form Ptolemy engines
# (the honest exception, per the `U1GaugedSU2Nf1SkeinKAlgebra`
# precedent — the stated punctured-disk arc engine is the recorded
# frontier).
# ===========================================================================


class A1AevenSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """`[A₁, A_{2k}]` — the general odd `(2k+3)`-gon member of the A1An
    roster, on the `A1A2kKAlg(k)` labels, over the generic-n PINNED
    engine (`SkeinOddPolygonKAlg`: closed-form unpin words,
    meson-calibrated gauges — the per-polygon fit eliminated).  k = 1,
    2, 3 are the pentagon/heptagon/nonagon siblings (the named roster
    instances keep their dedicated engines); k ≥ 4 are new (k = 4 =
    the hendecagon [A₁, A₈]).  ρ has finite order (mod 2k+3) — the
    default ρ²-orbit walk serves.  Trace = the intrinsic two-layer
    closed form (M(2, 2k+3) Andrews–Gordon characters)."""

    def __init__(self, k: int, intrinsic=None, engine=None):
        if intrinsic is None:
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(k)
        if engine is None:
            from skein_oddgon_kalg import SkeinOddPolygonKAlg
            engine = SkeinOddPolygonKAlg(intrinsic=intrinsic)
        self.k = intrinsic.k
        self.n = 2 * self.k + 3
        super().__init__(
            intrinsic=intrinsic,
            engine=engine,
            name=f"skein-a1a{2 * self.k}-cone",
            engine_provenance="stated-skein-pinned",
        )


class U1A1AoddSkeinKAlgebra(_BridgedPolygonSkeinKAlgebra):
    """The U(1)-gauged `[A₁, A_{2k+1}]` — the general even `(2k+4)`-gon
    member of the A1An roster, on the `U1A1AoddKAlg(k)` labels
    `(factors, e_E)`, over the generic-n PINNED engine
    (`SkeinEvenPolygonKAlg`: parity-covariant dressing with the ±ν
    residual pins, kernel offsets + per-letter gauges solved at
    construction — the per-polygon fit eliminated).  k = 1 is the
    hexagon sibling (the named roster instance keeps its dedicated
    localized engine; this class gives it an independent pinned route);
    k = 2, 3 are the octagon/decagon (NEW); k ≥ 4 builds its intrinsic
    through the RG oracle (constructible but slow).  ρ² drifts the
    E-power (infinite orbits) — the intrinsic closed-form drift
    quotient is passed through."""

    def __init__(self, k: int, intrinsic=None, engine=None):
        if intrinsic is None:
            from u1a1aodd_kalg import U1A1AoddKAlg
            intrinsic = U1A1AoddKAlg(k)
        if engine is None:
            from skein_evengon_kalg import SkeinEvenPolygonKAlg
            engine = SkeinEvenPolygonKAlg(intrinsic=intrinsic)
        self.k = intrinsic.k
        self.n = 2 * self.k + 4
        super().__init__(
            intrinsic=intrinsic,
            engine=engine,
            name=f"skein-u1a1a{2 * self.k + 1}-cone",
            engine_provenance="stated-skein-pinned",
            rho2_rep_fn=intrinsic._canonical_rho2_orbit_rep,
        )


class U1OctagonSkeinKAlgebra(U1A1AoddSkeinKAlgebra):
    """The U(1)-gauged octagon (gauged [A₁,A₅]) — k = 2 of the general
    even-gon family (named per the `U1OctagonKAlg` zoo convention)."""

    def __init__(self, intrinsic=None, engine=None):
        super().__init__(2, intrinsic=intrinsic, engine=engine)
        self.name = "skein-u1octagon-cone"


class U1DecagonSkeinKAlgebra(U1A1AoddSkeinKAlgebra):
    """The U(1)-gauged decagon (gauged [A₁,A₇]) — k = 3 of the general
    even-gon family (named per the `U1DecagonKAlg` zoo convention)."""

    def __init__(self, intrinsic=None, engine=None):
        super().__init__(3, intrinsic=intrinsic, engine=engine)
        self.name = "skein-u1decagon-cone"


class HendecagonSkeinKAlgebra(A1AevenSkeinKAlgebra):
    """The [A₁,A₈] hendecagon (11-gon) — k = 4 of the general odd-gon
    family; the first NEW odd member beyond the fitted roster."""

    def __init__(self, intrinsic=None, engine=None):
        super().__init__(4, intrinsic=intrinsic, engine=engine)
        self.name = "skein-hendecagon-cone"


# ===========================================================================
# A1Dodd cone data (once-punctured (2k+3)-gon: arc rays + the central
# peripheral-loop χ char ray)
# ===========================================================================

#: The peripheral-loop char-ray letter.  Word letters are (a, p, i)
#: triples with a ≥ 1, so (0, 0, 0) cannot collide (and sorts cleanly).
CHI_RAY = (0, 0, 0)


class A1DoddSkeinConeData(SkeinConeData):
    """The `[A₁, D_{2k+3}]` ray dictionary on the `A1DoddConeKAlg(k)`
    labels `(word, κ)`.

    **Surface (the A1Dn family map):** `[A₁, Dₙ]`
    is the **n-gon with a single REGULAR interior puncture** — the n
    boundary marks are the irregular-puncture (Stokes) data, and the
    regular puncture in the middle carries the SU(2) flavour.  Here
    n = 2k+3 (odd); the chart is the FST fan of
    `skein_a1dn_disk.bordered_a1dn(n)` (the certified `SkeinA1D3KAlg`
    geometry at general k; "once-punctured n-gon" below always means
    this regular interior puncture, never a boundary/irregular one).

    Rays: the `(a, p, i)` arc letters of the genuine D-type cluster
    frame (open arcs ⇒ monomial), plus the **peripheral loop around the
    regular puncture as the central SU(2)-character ray** `CHI_RAY` —
    the flavour χ₁ IS the Kauffman peripheral loop (the measured
    `SkeinA1D3KAlg` dictionary), so its tower `χ_κ` is computed
    NATIVELY by the char machinery (Chebyshev = Jones–Wenzl) and the
    Clebsch–Gordan fusion `χ_a·χ_b` is the tier's own char route —
    never delegated.  Word combinatorics (cones, cocycle, order,
    phases, canonicalisation) delegate to the intrinsic's certified
    `a1dodd_cone_data`; arc CROSS-products (the Ptolemy exchanges,
    which genuinely create χ — e.g. the D₃ meson `D₀·D₁ ∋ q⁻¹·T·χ₁`)
    come from the intrinsic closed-form engine (the anchored honest
    exception; the stated regular-punctured-disk arc engine is the
    recorded frontier)."""

    def __init__(self, intrinsic):
        self._alg = intrinsic                  # A1DoddConeKAlg
        self._int = intrinsic.cone_data()      # χ-stripped word frame

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        return "char" if tuple(g) == CHI_RAY else "monomial"

    def cone_sets(self):
        yield frozenset({CHI_RAY})
        for c in self._int.iter_cones():
            yield frozenset(c.mult_gens()) | {CHI_RAY}

    def mult_gens(self):
        return tuple(self._int.mult_gens()) + (CHI_RAY,)

    def engine_product(self, g, h) -> Element:
        """Arc-pair Ptolemy from the intrinsic engine (χ never crosses:
        it is central, so the reducer only asks for arc pairs)."""
        assert tuple(g) != CHI_RAY and tuple(h) != CHI_RAY, (g, h)
        return self._alg.multiply(self._one_letter(g), self._one_letter(h))

    def _one_letter(self, g):
        word = self._int.from_cone_label(frozenset({tuple(g)}),
                                         {tuple(g): 1})
        return (word, 0)

    # -- ConeData surface -------------------------------------------------------

    def coefficient_ring(self):
        # Z-form: the SU(2) flavour rides in the κ LABEL coordinate
        # (single-irrep lift), so reducer scalars are plain Z[q^±] —
        # the `BPSKAlgebra` flavour-in-labels pattern.
        return TrivialZPlusRing()

    def to_cone_label(self, native_label):
        word, kappa = native_label
        gens, powers = self._int.to_cone_label(tuple(word))
        gens = set(gens)
        powers = dict(powers)
        if kappa:
            gens.add(CHI_RAY)
            powers[CHI_RAY] = int(kappa)
        return frozenset(gens), powers

    def from_cone_label(self, gens, powers):
        kappa = powers.get(CHI_RAY, 0)
        wgens = frozenset(g for g in gens if tuple(g) != CHI_RAY)
        wpow = {g: p for g, p in powers.items() if tuple(g) != CHI_RAY}
        word = self._int.from_cone_label(wgens, wpow)
        return (word, kappa)

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h or g == CHI_RAY or h == CHI_RAY:
            return True                       # χ is central
        return self._int.q_commute(g, h)

    def cocycle(self, g, h) -> int:
        g, h = tuple(g), tuple(h)
        if g == CHI_RAY or h == CHI_RAY:
            return 0                          # χ is central
        return self._int.cocycle(g, h)

    def canonical_cone_order(self, gens):
        word_gens = frozenset(g for g in gens if tuple(g) != CHI_RAY)
        order = tuple(self._int.canonical_cone_order(word_gens)) \
            if word_gens else ()
        if CHI_RAY in gens:
            order = order + (CHI_RAY,)        # central: position immaterial
        return order

    def cone_label_phase(self, gens, powers) -> int:
        word_gens = frozenset(g for g in gens if tuple(g) != CHI_RAY)
        wpow = {g: p for g, p in powers.items() if tuple(g) != CHI_RAY}
        if not word_gens:
            return 0
        return self._int.cone_label_phase(word_gens, wpow)

    def canonicalize_cone_label(self, cone, gens, powers):
        kappa = powers.get(CHI_RAY, 0)
        word_gens = frozenset(g for g in gens if tuple(g) != CHI_RAY)
        wpow = {g: p for g, p in powers.items() if tuple(g) != CHI_RAY}
        wcone = frozenset(g for g in cone if tuple(g) != CHI_RAY)
        g2, p2, ph = self._int.canonicalize_cone_label(wcone, word_gens,
                                                       wpow)
        g2 = set(g2)
        p2 = dict(p2)
        if kappa:
            g2.add(CHI_RAY)
            p2[CHI_RAY] = kappa
        return frozenset(g2), p2, ph

    def cycle_period_bound(self) -> int:
        return self._int.cycle_period_bound()


class A1DoddSkeinKAlgebra(SkeinKAlgebra):
    """`[A₁, D_{2k+3}]` (odd-D AD, SU(2) flavour) — the D-side general
    constructor (k = 0 → D₃, the certified `SkeinA1D3KAlg` chart;
    k = 1 → D₅; …).

    **Surface**: the `(2k+3)`-gon with a single **regular interior
    puncture** (the A1Dn family map: a polygon with an extra regular
    puncture in the middle); the SU(2)
    flavour sits at the regular puncture and its peripheral loop is
    χ₁.

    Rays = the D-frame arc letters (monomial) + the peripheral loop χ
    (central char ray; **the χ-tower and all Clebsch–Gordan fusion are
    computed natively** — Chebyshev = Jones–Wenzl); `multiply` is the
    derived char-aware cone reducer over intrinsic-anchored arc
    cross-products (`A1DoddConeKAlg`'s closed-form Ptolemy — the honest
    exception while the stated regular-punctured-disk multi-arc engine
    is built; certification content: char-aware reducer + native CG ≡
    the intrinsic).  ρ = the clean `Z_{2k+3}` rotation (finite order —
    default ρ²-orbit walk).  Trace/flavour lift transported from the
    intrinsic (`r_label_decompose`: `L_{(word,κ)} = χ_κ·L_{(word,0)}`).
    No BPS twin registered on these labels."""

    def __init__(self, k: int = 0, intrinsic=None):
        if intrinsic is None:
            from a1dodd_kalg import A1DoddKAlg
            intrinsic = A1DoddKAlg(k, presentation="cone")
        self.k = getattr(intrinsic, "k", k)
        self.intrinsic = intrinsic
        cd = A1DoddSkeinConeData(intrinsic)
        super().__init__(
            cone_data=cd,
            identity_label=intrinsic.identity(),
            rho_fn=intrinsic.rho,
            rho_inverse_fn=intrinsic.rho_inverse,
            trace_fn=lambda a, K: intrinsic.trace(a, K),
            bps_factory=None,
            engine_multiply=None,          # derived char-aware reducer
            coefficient_ring=intrinsic.coefficient_ring(),
            name=f"skein-a1d{2 * self.k + 3}-cone",
            r_label_decompose_fn=intrinsic.r_label_decompose,
            r_label_compose_fn=intrinsic.r_label_compose,
            engine_provenance="cone-anchored",     # the honest exception:
                                                   # stated punctured-disk
                                                   # arc engine = frontier
            trace_provenance="transported-intrinsic",
        )


class _DevenSkeinConeData(A1DoddSkeinConeData):
    """The ray dictionary of `U1A1DevenSkeinKAlgebra` on the
    `U1A1DevenConeKAlgebra(k)` labels `(curves, e, κ)`: the curve rays of the
    once-punctured `(2k+2)`-gon (open arcs ⇒ monomial; a loop around the
    puncture is a curve `(x, 2k + 2)` too), the gauge letter `E^{±1}` as the
    adjoined torus pair, and the peripheral loop as the central `SU(2)`
    character ray `CHI_RAY` — the `A1DoddSkeinConeData` construction with the
    torus pair added.  Word combinatorics delegate to the intrinsic's
    `χ`-stripped cone data (native labels `(curves, e)`); crossing products
    come from the intrinsic's closed-form engine (the analytic skein rule,
    `u1a1deven_geometric_frame` docstring D)."""

    def ray_kind(self, g) -> str:
        if tuple(g) == CHI_RAY:
            return "char"
        if self._int._torus_inverse_letter(g) is not None:
            return "torus"
        return "monomial"

    def _one_letter(self, g):
        curves, e = self._int.from_cone_label(frozenset({tuple(g)}),
                                              {tuple(g): 1})
        return (curves, e, 0)

    def to_cone_label(self, native_label):
        curves, e, kappa = native_label
        gens, powers = self._int.to_cone_label((tuple(curves), e))
        gens = set(gens)
        powers = dict(powers)
        if kappa:
            gens.add(CHI_RAY)
            powers[CHI_RAY] = int(kappa)
        return frozenset(gens), powers

    def from_cone_label(self, gens, powers):
        kappa = powers.get(CHI_RAY, 0)
        wgens = frozenset(g for g in gens if tuple(g) != CHI_RAY)
        wpow = {g: p for g, p in powers.items() if tuple(g) != CHI_RAY}
        curves, e = self._int.from_cone_label(wgens, wpow)
        return (curves, e, kappa)

    def _torus_inverse_letter(self, g):
        if tuple(g) == CHI_RAY:
            return None
        return self._int._torus_inverse_letter(g)


class U1A1DevenSkeinKAlgebra(SkeinKAlgebra):
    """The U(1)-gauged `[A₁, D_{2k+2}]` (k = 1 → D₄, …) — the D-side
    even general constructor, on the `U1A1DevenConeKAlgebra(k)` labels
    `(curves, e, κ)` (since 2026-09-24).

    **Surface**: the underlying `[A₁, D_{2k+2}]` is the `(2k+2)`-gon
    with a single **regular interior puncture** (the A1Dn family map);
    the even-D flavour is rank 2 — the regular puncture's SU(2) stays
    a flavour here (its peripheral loop is `χ₁`, the κ slot of the
    labels), while the extra U(1) is the direction being gauged (the
    torus pair `E^{±1}`).  The rays are the simple curves of that
    punctured polygon, in `A1DnKAlg`'s `(x, ℓ)` convention —
    the ray↔curve dictionary this docstring used to list as open is the
    intrinsic's own labelling since the curve frame became the public
    class.

    Cone combinatorics delegate to the intrinsic's cone data (with the
    central χ ray, `_DevenSkeinConeData`); cross-products anchored on the
    intrinsic engine (the honest exception, as `A1DoddSkeinKAlgebra`).
    ρ² has infinite orbits (torus drift) — the intrinsic closed-form rep
    is passed through.  Trace = the intrinsic's: the exact closed form on
    the gauge sector, the exact transport of the closed-form RG image
    otherwise."""

    def __init__(self, k: int = 1, intrinsic=None):
        if intrinsic is None:
            from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
            intrinsic = U1A1DevenConeKAlgebra(k)
        self.k = getattr(intrinsic, "k", k)
        self.intrinsic = intrinsic
        cd = _DevenSkeinConeData(intrinsic)
        super().__init__(
            cone_data=cd,
            identity_label=intrinsic.identity(),
            rho_fn=intrinsic.rho,
            rho_inverse_fn=intrinsic.rho_inverse,
            trace_fn=lambda a, K: intrinsic.trace(a, K),
            bps_factory=None,
            engine_multiply=None,          # derived char-aware reducer
            rho2_rep_fn=intrinsic._canonical_rho2_orbit_rep,
            coefficient_ring=intrinsic.coefficient_ring(),
            name=f"skein-u1a1d{2 * self.k + 2}-cone",
            r_label_decompose_fn=intrinsic.r_label_decompose,
            r_label_compose_fn=intrinsic.r_label_compose,
            engine_provenance="cone-anchored",     # the honest exception
            trace_provenance="transported-intrinsic",
        )


# ===========================================================================
# The SU(2)+N_f family.
# SURFACE DICTIONARY: the SU(2)+N_f
# class-S surfaces are spheres with a mix of REGULAR (interior full)
# punctures and IRREGULAR punctures (rank-r irregular ↔ 2r marks on its
# boundary circle):
#   N_f=0 — the (1,1) annulus            (PureSU2AnnulusSkeinKAlgebra)
#   N_f=1 — the (1,2) annulus            (SU2Nf1AnnulusSkeinKAlgebra)
#   N_f=2 — disk: 1-mark boundary + 2 REGULAR punctures
#                                        (SU2Nf2DiskSkeinKAlgebra)
#   N_f=3 — disk: 2-mark boundary + 2 REGULAR punctures
#                                        (SU2Nf3DiskSkeinKAlgebra)
#   N_f=4 — the 4-punctured sphere       (SphereSU2Nf4SkeinKAlgebra)
# The N_f=1..3 instances ride the CERTIFIED bordered FST charts
# (`skein_su2_nf1_annulus` / `skein_su2_nf2_disk` / `skein_su2_nf3_disk`:
# minuscule flavour bases, vacuum indices ==
# `build_bps_su2_nf*`, cut = RG chain).  Flavour stays native
# Abelian/Cartan (KIsos to flavour-reduced
# counterparts are the witnesses).
# ===========================================================================


class BorderedChartRaySkeinConeData(SkeinConeData):
    """Ray-sector cone data over a certified `BorderedSkeinKAlg` chart.

    Rays (all charges in the chart's mutable-block label lattice):

      * **arc rays** — the FST interior-edge node charges (simple open
        arcs of the triangulation ⇒ monomial generators);
      * **torus rays** — the flavour units of `ker(B)` as inverse pairs
        (group-like central invertibles, `L_μ·L_{μ⁻¹} = 1` — asserted
        at construction).  Geometric attribution (measured, the a1d3
        `f = Σ incident radials` mechanism): a REGULAR puncture's unit
        is the **sum of its incident interior arcs** — the peripheral
        flavour unit whose Kauffman loop is the virtual character
        `μ + μ⁻¹`; an even-marked irregular boundary contributes its
        monodromy unit (the alternating word — the square's V/μ
        mechanism).

    Cones = the maximal q-commuting cliques of the arc rays (measured
    at construction from the chart: single-term products both ways),
    with every torus unit adjoined (central).  The cocycle is measured
    (`L_g·L_h = q^{c}·L_{g+h}` on q-commuting pairs; antisymmetry
    asserted).  Scope (honest): the bijection covers the
    **compatible (disjoint-curve) sector** — labels in a single cone's
    non-negative arc span + integer torus span; everything else
    honest-fails (`ValueError`).  Crossing-arc products land on dressed
    canonicals OUTSIDE this sector (measured), so the instance
    `multiply` is the chart engine (the Lê-QT route), never the
    reducer — the cone/chart split, the `PureSU2Annulus` precedent."""

    def __init__(self, chart, torus_units, unit_names=None,
                 arc_rays=None):
        self._B = chart
        # arc rays default to all interior-edge node charges; instances
        # exclude arcs whose parallel copies BUBBLE (measured: the arc
        # joining two REGULAR punctures has the 2-term tower
        # L_a² = L_{2a} + μ_{I1}·μ_{I2} — not a monomial generator; its
        # canonicals route through the chart engine)
        self._arcs = [tuple(c) for c in (arc_rays if arc_rays is not None
                                         else chart.node_charges)]
        self._units = [tuple(u) for u in torus_units]
        self._unit_names = tuple(unit_names or
                                 [f"mu{i}" for i in range(len(self._units))])
        self._n = len(self._arcs[0])
        # letters: arcs by charge; torus letters ("T", i, ±1)
        self._torus_letters = {}
        for i, u in enumerate(self._units):
            self._torus_letters[("T", i, 1)] = u
            self._torus_letters[("T", i, -1)] = tuple(-x for x in u)
        # measured q-commutation + cocycle on the ray letters
        self._qc = {}
        self._cc = {}
        letters = self._all_letters()
        for g in letters:
            for h in letters:
                if g == h:
                    continue
                e = self._pair_exponent(g, h)
                if e is None:
                    self._qc[(g, h)] = False
                else:
                    self._qc[(g, h)] = True
                    self._cc[(g, h)] = e
        for (g, h), e in self._cc.items():
            back = self._cc.get((h, g))
            assert back == -e, (g, h, e, back)
        # torus units: group-like central invertibles
        for i, u in enumerate(self._units):
            inv = tuple(-x for x in u)
            prod = dict(self._B.multiply(u, inv).terms)
            assert list(prod) == [(0,) * self._n] and \
                str(prod[(0,) * self._n]) == "1", (u, prod)
            for a in self._arcs:
                assert self._qc[(("T", i, 1), a)], (u, a, "unit not central")
        # arc rays must have MONOMIAL towers: the square is a single
        # unit-coefficient term at 2·arc (this loudly excludes bubbling
        # arcs — measured: the puncture–puncture arc has the 2-term
        # tower L² = L_{2a} + L_{μμ'} and is NOT a monomial ray)
        for a in self._arcs:
            two = tuple(2 * x for x in a)
            sq = dict(self._B.multiply(a, a).terms)
            assert list(sq) == [two], (a, "arc ray tower is not monomial "
                                          "(bubbling arc?)", sq)
            items = list(sq[two]._coeffs.items())
            assert len(items) == 1 and items[0][1] == 1, (a, sq)
        # maximal arc cliques (arcs are few — simple recursion)
        self._cliques = self._max_cliques()

    # -- construction helpers ---------------------------------------------

    def _all_letters(self):
        return list(self._arcs) + list(self._torus_letters)

    def _charge(self, g):
        if isinstance(g, tuple) and len(g) == 3 and g[0] == "T":
            return self._torus_letters[g]
        return tuple(g)

    def _pair_exponent(self, g, h):
        """q-exponent c with L_g·L_h = q^c·L_{g+h}, or None if the pair
        crosses (multi-term / off-sum support)."""
        cg, ch = self._charge(g), self._charge(h)
        x = dict(self._B.multiply(cg, ch).terms)
        y = dict(self._B.multiply(ch, cg).terms)
        s = tuple(a + b for a, b in zip(cg, ch))
        if list(x) != [s] or list(y) != [s]:
            return None
        items = list(x[s]._coeffs.items())
        if len(items) != 1 or items[0][1] != 1:
            return None
        return items[0][0]

    def _max_cliques(self):
        arcs = self._arcs
        out = []

        def bk(r, p, x):
            if not p and not x:
                out.append(frozenset(r))
                return
            for v in list(p):
                nb = {u for u in arcs
                      if u != v and self._qc.get((v, u), False)}
                bk(r | {v}, p & nb, x & nb)
                p = p - {v}
                x = x | {v}

        bk(set(), set(arcs), set())
        return tuple(out)

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        if isinstance(g, tuple) and len(g) == 3 and g[0] == "T":
            return "torus"
        return "monomial"

    def cone_sets(self):
        tl = frozenset(self._torus_letters)
        for c in self._cliques:
            yield c | tl

    def mult_gens(self):
        return tuple(self._arcs) + tuple(self._torus_letters)

    def engine_product(self, g, h) -> Element:
        return self._B.multiply(self._charge(g), self._charge(h))

    def _torus_inverse_letter(self, g):
        if isinstance(g, tuple) and len(g) == 3 and g[0] == "T":
            return ("T", g[1], -g[2])
        return None

    # -- ConeData surface -------------------------------------------------------

    def coefficient_ring(self):
        from zplus_ring import TrivialZPlusRing
        return TrivialZPlusRing()      # Z-form: flavour rides in the labels

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        pair = {g, h}
        if any(isinstance(z, tuple) and len(z) == 3 and z[0] == "T"
               for z in pair):
            # torus units are central; the μ·μ⁻¹ cancellation is handled
            # by `_word_to_gens_powers`' inverse-pair collapse
            return True
        return self._qc.get((g, h), False)

    def cocycle(self, g, h) -> int:
        if g == h:
            return 0
        return self._cc[(g, h)] if (g, h) in self._cc else 0

    def canonical_cone_order(self, gens):
        arcs = sorted(g for g in gens if self.ray_kind(g) == "monomial")
        tor = sorted(g for g in gens if self.ray_kind(g) == "torus")
        return tuple(arcs) + tuple(tor)

    @staticmethod
    def _solve_exact(cols, gamma, n):
        """gamma = cols·x via the layer's shared `fraction_solve`
        (greedy rank-profile pivoting in column order, free columns
        := 0 — the canonical choice where the column set is
        rank-deficient; the kernel units can overlap a clique's arc
        span — the non-simplicial `canonicalize_cone_label` situation,
        resolved by that fixed pivot rule).  Returns the x list or
        None."""
        got = fraction_solve(cols, list(gamma), dim=n)
        return None if got is None else got[0]

    def to_cone_label(self, native_label):
        """Decompose a chart charge into the compatible sector: a pure
        torus-unit word, or ONE clique's non-negative arc span +
        integer torus span (exact solve, canonical greedy pivots;
        honest-fail off the sector).  Every accepted decomposition is a
        TRUE identity — flavour-unit shifts are exact in the chart
        (flavour-in-labels) and monomial arc towers are certified at
        construction; `verify_consistent_with_multiply` is the battery
        guard."""
        gamma = tuple(native_label)
        if gamma == (0,) * self._n:
            return frozenset(), {}
        units = self._units
        # pure torus-unit words first (units are jointly independent)
        x = self._solve_exact(list(units), gamma, self._n)
        if x is not None and all(v.denominator == 1 for v in x):
            gens = set()
            powers = {}
            for i, v in enumerate(x):
                if v:
                    lt = ("T", i, 1 if v > 0 else -1)
                    gens.add(lt)
                    powers[lt] = abs(int(v))
            if gens:
                return frozenset(gens), powers
        for cl in self._cliques:
            cols = sorted(cl) + list(units)
            x = self._solve_exact(cols, gamma, self._n)
            if x is None or any(v.denominator != 1 for v in x):
                continue
            arcs_part = x[:len(cl)]
            tor_part = x[len(cl):]
            if any(v < 0 for v in arcs_part) or all(v == 0
                                                    for v in arcs_part):
                continue
            gens = set()
            powers = {}
            for a, v in zip(sorted(cl), arcs_part):
                if v:
                    gens.add(a)
                    powers[a] = int(v)
            for i, v in enumerate(tor_part):
                if v:
                    lt = ("T", i, 1 if v > 0 else -1)
                    gens.add(lt)
                    powers[lt] = abs(int(v))
            return frozenset(gens), powers
        raise ValueError(
            f"bordered-chart ray sector: {gamma} is not in any "
            f"compatible (disjoint-curve) cone — off the certified "
            f"bijection scope (dressed canonicals route through the "
            f"chart engine)"
        )

    def from_cone_label(self, gens, powers):
        out = [0] * self._n
        for g, p in powers.items():
            if not p:
                continue
            c = self._charge(g)
            for i in range(self._n):
                out[i] += c[i] * p
        return tuple(out)


class _SU2NfBorderedSkeinKAlgebra(SkeinKAlgebra):
    """Shared shell for the SU(2)+N_f = 1..3 instances over the
    certified bordered FST charts: `multiply`/ρ/trace = the chart
    engine (the Lê-QT substrate — `bordered-chart-anchored`: the
    stated multi-arc engine that would make crossing products
    stated-native is the recorded frontier), cone data = the measured
    ray-sector dictionary, `to_bpskalgebra()` = the SHARED chart
    instance (identity-label iso certifiable, no duplicate spec
    search)."""

    def __init__(self, chart, torus_units, unit_names, name,
                 arc_rays=None):
        self.chart = chart
        cd = BorderedChartRaySkeinConeData(chart, torus_units, unit_names,
                                           arc_rays=arc_rays)
        n = len(chart.node_charges[0])
        super().__init__(
            cone_data=cd,
            identity_label=(0,) * n,
            rho_fn=chart.rho,
            rho_inverse_fn=chart.rho_inverse,
            trace_fn=lambda a, K: chart.trace(a, K),
            bps_factory=lambda: chart,
            engine_multiply=chart.multiply,   # crossing arcs land on
                                              # dressed canonicals — the
                                              # reducer is not the path
            coefficient_ring=chart.coefficient_ring(),
            name=name,
            engine_provenance="bordered-chart-anchored",
            trace_provenance="native-chart-schur",
        )

    def torus_unit(self, i: int = 0):
        """The i-th flavour unit's charge (see the cone data's
        geometric attribution)."""
        return self._cd._units[i]


class SU2Nf1AnnulusSkeinKAlgebra(_SU2NfBorderedSkeinKAlgebra):
    """SU(2)+N_f=1 on the **(1,2) annulus** (outer 1-mark + inner
    2-mark irregular boundaries; no regular puncture) — the certified
    `bordered_annulus_su2_nf1` chart (minuscule / U(1)-gaugeable
    flavour ±1; vacuum index == `build_bps_su2_nf1` up to the flavour
    normalization).  Rays: the 3 interior fan arcs (monomial) + the
    inner-boundary monodromy unit μ = L_{(1,−1,1)} (torus; the
    even-marked boundary's alternating-arc unit — the square's V
    mechanism).  Pinning the inner 2-boundary is the U(1)-gauging (→
    `U1GaugedSU2Nf1SkeinKAlgebra`)."""

    def __init__(self, chart=None):
        if chart is None:
            from skein_su2_nf1_annulus import bordered_annulus_su2_nf1
            chart = bordered_annulus_su2_nf1()
        super().__init__(chart, [(1, -1, 1)], ("mu-inner",),
                         "skein-su2-nf1-annulus")


class SU2Nf2DiskSkeinKAlgebra(_SU2NfBorderedSkeinKAlgebra):
    """SU(2)+N_f=2 on the **disk with a 1-mark boundary + 2 REGULAR
    interior punctures** (NOT an (a,b) annulus) — the
    certified `bordered_disk_su2_nf2` chart.  Rays: the 3
    boundary-to-puncture arcs (monomial) + the two REGULAR punctures'
    peripheral flavour units (torus), **measured = the incident-arc
    sums** (the a1d3 `f = Σ radials` mechanism): μ_{I1} = L_{(1,0,1,1)},
    μ_{I2} = L_{(0,1,0,1)}; the Kauffman peripheral loop is the
    virtual character μ + μ⁻¹ (Abelian-flavour presentation).  The
    puncture–puncture arc `n₃`
    (I1–I2) is NOT a monomial ray — **measured bubbled tower
    `L_{n₃}² = L_{2n₃} + L_{μ_{I1}+μ_{I2}}`** (the arc analogue of the
    GNO tower; its canonicals route through the chart engine).
    Cutting the I1–I2 edge is the RG flow to the N_f=1 annulus
    (`cut_to_nf1`)."""

    def __init__(self, chart=None):
        if chart is None:
            from skein_su2_nf2_disk import bordered_disk_su2_nf2
            chart = bordered_disk_su2_nf2()
        super().__init__(chart, [(1, 0, 1, 1), (0, 1, 0, 1)],
                         ("mu-I1", "mu-I2"), "skein-su2-nf2-disk",
                         arc_rays=[(1, 0, 0, 0), (0, 1, 0, 0),
                                   (0, 0, 1, 0)])


class SU2Nf3DiskSkeinKAlgebra(_SU2NfBorderedSkeinKAlgebra):
    """SU(2)+N_f=3 on the **disk with a 2-mark boundary (rank-1
    irregular) + 2 REGULAR interior punctures** (NOT
    the (2,3) annulus) — the certified `bordered_disk_su2_nf3` chart.
    Rays: the 4 boundary-to-puncture arcs (monomial) + three flavour
    units (torus): the two REGULAR punctures' peripheral units,
    **measured = the incident-arc sums** μ_{I1} = L_{(1,1,0,0,1)},
    μ_{I2} = L_{(0,0,1,1,1)}, plus the outer-bigon boundary unit
    μ_∂ = L_{(0,1,−1,0,0)} (a kernel rep transverse to the puncture
    units; the irregular-boundary direction).  The puncture–puncture
    arc `n₄` (I1–I2) is NOT a monomial ray — the same **measured
    bubbled tower `L_{n₄}² = L_{2n₄} + L_{μ_{I1}+μ_{I2}}`** as the
    N_f=2 disk (chart-engine sector)."""

    def __init__(self, chart=None):
        if chart is None:
            from skein_su2_nf3_disk import bordered_disk_su2_nf3
            chart = bordered_disk_su2_nf3()
        super().__init__(chart,
                         [(1, 1, 0, 0, 1), (0, 0, 1, 1, 1),
                          (0, 1, -1, 0, 0)],
                         ("mu-I1", "mu-I2", "mu-boundary"),
                         "skein-su2-nf3-disk",
                         arc_rays=[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                                   (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)])


def _puncture_incident_unit(tri, vertex):
    """The incident-interior-arc sum of a vertex of a
    `BorderedTriangulation` — the measured peripheral flavour unit of a
    REGULAR puncture (the a1d3 `f = Σ radials` mechanism; role
    convention a = v0→v1, b = v1→v2, c = v2→v0)."""
    slot_to_edge, internal_ids, _ = tri.edge_numbering()
    pos = {e: i for i, e in enumerate(internal_ids)}
    seen = set()
    out = [0] * len(internal_ids)
    for (t, role), e in slot_to_edge.items():
        if e not in pos or e in seen:
            continue
        v0, v1, v2 = tri.triangle_verts[t]
        pair = ((v0, v1), (v1, v2), (v2, v0))[role]
        if vertex in pair:
            out[pos[e]] += pair.count(vertex)
            seen.add(e)
    return tuple(out)


class SU2TwoPunctureDiskSkeinKAlgebra(_SU2NfBorderedSkeinKAlgebra):
    """SU(2) + 2 doublets + A1Dn on the **disk with n boundary marks +
    2 REGULAR interior punctures** (the family map) —
    the general two-puncture-disk rung over the certified
    `bordered_su2_2punct(n)` charts.  n = 1, 2 are SU(2) N_f = 2, 3
    (the named `SU2Nf2Disk`/`SU2Nf3Disk` instances); **n ≥ 3 is
    genuinely AD-matter-coupled SU(2)** (interacting A1Dn matter — NOT
    SU(2) N_f = n+1), which is this class's domain.

    Rays: the interior arcs except the P–Q arc (monomial) + the
    flavour units (torus): the two REGULAR punctures' peripheral
    units, **computed as the incident-arc sums**
    (`_puncture_incident_unit` — the a1d3 mechanism, certified
    group-like/central at construction), plus — for EVEN n only — the
    extra A1D-even flavour direction (the D-parity feature of the
    A1Dn family: measured kernel rank 2 at n = 3, rank 3 at n = 2, 4;
    derived as the smallest kernel vector transverse to the puncture
    units).  The P–Q arc bubbles with the SAME law as the N_f = 2/3
    disks, `L² = L_{2a} + L_{μ_P+μ_Q}` — measured at n = 2, 3, 4
    (excluded from the monomial rays; the stated multi-arc engine's
    sharp target)."""

    def __init__(self, n: int = 3, chart=None):
        if n < 3:
            raise ValueError(
                f"SU2TwoPunctureDiskSkeinKAlgebra needs n >= 3 (the "
                f"AD-coupled regime); n=1,2 are the named "
                f"SU2Nf2Disk/SU2Nf3Disk instances"
            )
        from skein_su2_2punct_disk import (
            bordered_su2_2punct, su2_2punct_triangulation)
        if chart is None:
            chart = bordered_su2_2punct(n)
        self.n = n
        tri = su2_2punct_triangulation(n)
        mu_P = _puncture_incident_unit(tri, 0)          # the A1Dn puncture
        mu_Q = _puncture_incident_unit(tri, n + 1)      # the inserted one
        units = [mu_P, mu_Q]
        names = ["mu-P", "mu-Q"]
        if n % 2 == 0:
            units.append(self._extra_kernel_unit(tri, [mu_P, mu_Q]))
            names.append("mu-deven")
        # arc rays: interior arcs except the P–Q edge (the bubbling arc
        # — it is exactly the edge incident to BOTH punctures)
        rank = len(chart.node_charges[0])
        pq = tuple(min(a, b) for a, b in zip(mu_P, mu_Q))
        arcs = [tuple(c) for c in chart.node_charges
                if tuple(c) != pq]
        assert sum(pq) == 1 and len(arcs) == rank - 1, (mu_P, mu_Q)
        super().__init__(chart, units, tuple(names),
                         f"skein-su2-2punct-disk-a1d{n}",
                         arc_rays=arcs)

    @staticmethod
    def _extra_kernel_unit(tri, known):
        """The smallest (by |coords|, then lex) ker(B) vector transverse
        to the known puncture units — the A1D-even extra flavour
        direction (even n only).  Group-like/centrality is certified
        downstream by the cone-data construction."""
        import itertools as it
        Bm = tri.mutable_block()
        N = len(Bm)

        def in_span(v, vs):
            # v ∈ span(vs) ⇔ the system Σ x_j·vs_j = v is consistent
            # (the shared solver returns None exactly on inconsistency)
            return fraction_solve(list(vs), list(v), dim=N) is not None

        cands = sorted(
            (v for v in it.product((-1, 0, 1), repeat=N)
             if any(v) and all(sum(Bm[i][j] * v[j] for j in range(N)) == 0
                               for i in range(N))),
            key=lambda v: (sum(abs(x) for x in v), v))
        for v in cands:
            if not in_span(v, known):
                return tuple(v)
        raise ValueError("no transverse kernel unit found (even-n "
                         "A1D flavour direction expected)")


#: The named-instance roster (the "listing" surface): short id → subclass.
#: Polygon (A1An) family first, then the bordered / sphere instances,
#: then the general-family defaults.
ROSTER = {
    "u1square": U1SquareSkeinKAlgebra,
    "pentagon": PentagonSkeinKAlgebra,
    "u1hexagon": U1HexagonSkeinKAlgebra,
    "heptagon": HeptagonSkeinKAlgebra,
    "u1octagon": U1OctagonSkeinKAlgebra,
    "nonagon": NonagonSkeinKAlgebra,
    "u1decagon": U1DecagonSkeinKAlgebra,
    "hendecagon": HendecagonSkeinKAlgebra,
    "annulus-pure-su2": PureSU2AnnulusSkeinKAlgebra,
    "u1gauged-su2-nf1": U1GaugedSU2Nf1SkeinKAlgebra,
    "sphere-su2-nf4": SphereSU2Nf4SkeinKAlgebra,
    "a1d3": A1DoddSkeinKAlgebra,               # default k = 0 → D₃
    "u1a1d4": U1A1DevenSkeinKAlgebra,          # default k = 1 → D₄
    "annulus-su2-nf1": SU2Nf1AnnulusSkeinKAlgebra,
    "disk-su2-nf2": SU2Nf2DiskSkeinKAlgebra,
    "disk-su2-nf3": SU2Nf3DiskSkeinKAlgebra,
    "su2-2punct-a1d3": SU2TwoPunctureDiskSkeinKAlgebra,  # default n = 3
}
