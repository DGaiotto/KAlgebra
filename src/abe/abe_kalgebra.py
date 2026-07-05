"""`AbeKAlgebra` — `KAlgebra` presented on the abelianized (enriched-torus)
chart: the **completed contract** (Plan 30; rulings D1–D4, D7, D8).

`AbeKAlgebra(KAlgebra)` is the chart/torus-presentation tier, beside
`RGKAlgebra` (the flow presentation):

    KAlgebra
    ├── RGKAlgebra        presented by an RG flow over a graded auxiliary
    └── AbeKAlgebra       presented faithfully on an enriched rational
          │               quantum torus (THIS TIER)
          ├── PureUNKAlgebra        (retrofit: Plan 30 T3)
          ├── UNNfKAlgebra          (born from this contract: T4)
          └── UNQuiverKAlgebra      (born from this contract: T4)

The presentation (repo-pinned definitions — read this, not the literature)
--------------------------------------------------------------------------
Elements live in the **f-presentation** on the product cocharacter lattice

    x  =  Σ_{m⃗}  f_{m⃗}(𝔮^{m⃗} v) · U_{m⃗},      f_{m⃗} = Σ_{k⃗} f^{(k⃗)}_{m⃗}·μ^{k⃗},

with `U_{m⃗}` the enriched atoms of `quiver_urq_torus.QuiverURQTorus`
(per-node pure normalization × the per-cell matter rung products), labels
in **lower Kapustin** convention per node, ρ the √(full measure)
G-cocycle, the trace the (product) Schur-measure residue with the matter
M-factors, bar the componentwise `𝔮 ↦ 𝔮⁻¹` (so bar-invariance ⟺ W1
palindromicity), and the canonical basis characterized executably by
`well_formed` = W1 + the single-leading-orbit q-extreme (W2) — the
Kazhdan–Lusztig read.  The literature's abelianization (BFN et al.) is a
*related* construction and is **non-load-bearing here**: no contract
semantics may be imported from it (the "false friends" hazard; ruling D2).

**No DOp on this tier** (ruling D4): the f-presentation is the single
public language; chart-operator machinery (`abelianized_torus.DOp`) is
engine-internal only.

The contract (ruling D3 — the minimal triple)
---------------------------------------------
Concrete realisations supply exactly three primitives:

  * `torus_shape()` — the `TorusShape` of the enriched torus (per-node
    `RootDatum`s, matter multiplicities, links; type-A chains via
    `TorusShape.from_ranks_nf`).  This is the D6 shape surface (re-ruled
    2026-07-02 to the root-datum form).
  * `chart(label)` — the faithful f-presentation image of the canonical
    `L_label` (a `QuiverURQTorus` element).
  * `decompose(x)` — a torus element as `Element({label: C(q)})`: the
    **no-target**, level-ascending canonical read.  It must honest-fail
    (raise) off its certified scope — never guess.

Everything else is derived here: `multiply`, `rho`, `rho_inverse`,
`trace`, `inner_product`, the chart-level bar verifier, and the
`certify_canonical` acceptance.

constructive-build rule (contract-enforced)
---------------------------------
A canonical may be constructed **only** as a polynomial in basis
generators minus already-built lower canonicals; acceptance is
`well_formed` equality with the intended label.  This tier exposes **no
solve entry point**, and `decompose` is a read (it has no target to
fabricate toward).  Where a build does not reach, the realisation must
honest-fail — never fit, never solve.  (The constructive-build rule.)

Flavour rings (ruling D8)
-------------------------
The faithful flavour symmetry of a U(N) node with `M` fundamentals is
**SU(M)** (the central U(1) is absorbed by the gauge centre); link μ's
are formal solving gradings, not flavour.  The torus levels `k⃗` are the
internal torus-refined bookkeeping; the contract packages them through
`_flavour_element` — abelian identification by default, with the
`R(SU(M))` packaging supplied by the T4 shells (which also certify the
central-direction identification against gauge det-Wilson shifts).
"""
from __future__ import annotations

import os
import sys
from abc import abstractmethod

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element, Label
from laurent_poly import LaurentPoly
from root_datum import u_n
from zplus_ring import RElement, RPowerSeries


__all__ = ["AbeKAlgebra", "TorusShape"]


class TorusShape:
    """The declared shape of the enriched torus — the tier's "which theory am
    I" surface (ruling D6, re-ruled 2026-07-02 to the root-datum form, option
    (b): one honest object).

    Fields: `data` — the per-node `RootDatum`s (gauge factors); `matter` —
    per-node fundamental multiplicities; `links` — bifundamental edges as
    node-index pairs.  Type-A chains construct via `from_ranks_nf` (u_n data,
    consecutive links — the legacy `(ranks, Nf)` reading); group-general
    realisations via `from_root_data`.  Equality compares datum names +
    matter + links (factory data are name-canonical)."""

    __slots__ = ("data", "matter", "links")

    def __init__(self, data, matter, links=()):
        self.data = tuple(data)
        self.matter = tuple(int(x) for x in matter)
        self.links = tuple(tuple(l) for l in links)
        if len(self.matter) != len(self.data):
            raise ValueError("TorusShape: one matter multiplicity per node")

    @classmethod
    def from_ranks_nf(cls, ranks, nf, links=None):
        """The type-A chain shape — per-node `u_n` data, consecutive links."""
        ranks = tuple(int(r) for r in ranks)
        if links is None:
            links = tuple((i, i + 1) for i in range(len(ranks) - 1))
        return cls(tuple(u_n(r) for r in ranks), tuple(nf), links)

    @classmethod
    def from_root_data(cls, data, matter=None, links=()):
        data = tuple(data)
        if matter is None:
            matter = (0,) * len(data)
        return cls(data, matter, links)

    @property
    def ranks(self):
        """Per-node unitary ranks — honest-fails on non-U(N) nodes."""
        out = []
        for d in self.data:
            if not d.name.startswith("U("):
                raise NotImplementedError(
                    f"TorusShape.ranks: node {d.name} is not a U(N) node")
            out.append(d.dim)
        return tuple(out)

    @property
    def nf(self):
        return self.matter

    def _key(self):
        return (tuple(d.name for d in self.data), self.matter, self.links)

    def __eq__(self, other):
        if not isinstance(other, TorusShape):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        nodes = " × ".join(d.name for d in self.data)
        return f"TorusShape({nodes}; matter={self.matter}; links={self.links})"


class AbeTorus:
    """The enriched **WRQTorus** substrate of an `AbeKAlgebra`, SELECTED by its
    `TorusShape` (D9/D10: the group-general WRQTorus is the *universal*
    substrate — the pure `WRQTorus` / matter `MatterWRQTorus` / quiver
    `QuiverWRQTorus` element type falls out of the shape, so there is no
    per-tier substrate/algebra class).  A realisation picks its substrate purely
    by what `torus_shape()` returns; `AbeKAlgebra.torus()` builds this from it."""

    __slots__ = ("shape", "n", "nf", "kind", "datum")

    def __init__(self, shape: "TorusShape"):
        self.shape = shape
        self.n = len(shape.data)
        self.nf = shape.matter
        if self.n == 1 and shape.matter[0] == 0:
            self.kind = "pure"
            self.datum = shape.data[0]
        elif self.n == 1:
            self.kind = "matter"
            self.datum = shape.data[0]
        else:
            from quiver_wrq_torus import quiver_datum
            self.kind = "quiver"
            self.datum = quiver_datum(shape.ranks)

    @property
    def ranks(self):
        return self.shape.ranks

    def element(self, residuals):
        """Build a substrate element (the WRQTorus type selected by the shape)
        from `{atom: …}` residuals."""
        if self.kind == "pure":
            from wrq_torus import WRQTorus
            return WRQTorus(self.datum, residuals)
        if self.kind == "matter":
            from matter_wrq_torus import MatterWRQTorus
            return MatterWRQTorus(self.datum, self.nf[0], residuals)
        from quiver_wrq_torus import QuiverWRQTorus
        return QuiverWRQTorus(self.shape.ranks, self.nf, residuals,
                              datum=self.datum)

    def __repr__(self):
        return f"AbeTorus({self.kind}, {self.datum.name}, matter={self.nf})"


class AbeKAlgebra(KAlgebra):
    """`KAlgebra` presented faithfully on the enriched rational quantum
    torus — supply `torus_shape` / `chart` / `decompose`, inherit the
    algebra.  See the module docstring for the pinned conventions.

    The substrate is the **group-general WRQTorus**, selected by `torus_shape()`
    and exposed as the `torus()` property (D9/D10, user 2026-07-04: use WRQ
    universally, no extra WRQ-named classes — the substrate is a property of
    `AbeKAlgebra` picked by the subclass's shape)."""

    def torus(self) -> "AbeTorus":
        """The WRQTorus substrate selected by `torus_shape()` (cached)."""
        t = getattr(self, "_abe_torus", None)
        if t is None:
            t = AbeTorus(self.torus_shape())
            self._abe_torus = t
        return t

    # ------------------------------------------------------------------
    # The contract: three abstract primitives (D3).
    # ------------------------------------------------------------------

    @abstractmethod
    def torus_shape(self) -> "TorusShape":
        """The `TorusShape` of the enriched torus — per-node `RootDatum`s,
        matter multiplicities, and links.  The D6 shape surface (re-ruled
        2026-07-02 to the root-datum form); everything geometric (lattice
        rank, Weyl blocks, dressing cells, measure) derives from it.
        Type-A chains: `TorusShape.from_ranks_nf(ranks, Nf)`."""

    @abstractmethod
    def chart(self, label: Label):
        """`L_label` as a `QuiverURQTorus` element (the f-presentation
        image; faithful for all derived operations)."""

    @abstractmethod
    def decompose(self, x) -> Element:
        """A torus element in the canonical basis: `Element({label: C(q)})`.
        The no-target, level-ascending read; **honest-fail off-scope**
        (raise `NotImplementedError`) — never guess, never solve."""

    # ------------------------------------------------------------------
    # Derived KAlgebra primitives (the whole algebra).
    # ------------------------------------------------------------------

    def multiply(self, a: Label, b: Label) -> Element:
        """`L_a · L_b = Σ_c C^c_{ab}(q)·L_c` — chart-multiply (the exact
        torus cocycle product) then decompose."""
        return self.decompose(self.chart(a) * self.chart(b))

    def _single_label(self, x, what: str) -> Label:
        d = self.decompose(x)
        terms = {lab: c for lab, c in d.terms.items()
                 if not (hasattr(c, "is_zero") and c.is_zero())}
        if len(terms) != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.{what}: image is not a single "
                f"canonical ({len(terms)} terms)")
        (lab, c), = terms.items()
        cl = c if isinstance(c, LaurentPoly) else LaurentPoly({0: int(c)})
        if cl != LaurentPoly({0: 1}):
            raise NotImplementedError(
                f"{type(self).__name__}.{what}: image coefficient {c} ≠ 1 "
                f"(the basis map must be coefficient-free)")
        return lab

    def rho(self, a: Label) -> Label:
        """ρ on labels — the torus √measure conjugation read back through
        `decompose` (a sign-free basis permutation, coefficient 1)."""
        return self._single_label(self.chart(a).rho(), "rho")

    def rho_inverse(self, a: Label) -> Label:
        return self._single_label(self.chart(a).rho_inverse(), "rho_inverse")

    def r_label_decompose(self, label):
        """The flavour-lift coordinate (Plan 32), **defaulted for the
        abelianized contract**: this tier packages flavour into the
        *coefficient ring* (`_flavour_element` — the torus levels become
        `R(SU(M))` characters, "free over `R(G_f)`"), so the canonical basis is
        flavour-neutral and the lift is trivial — section = `label`, single
        irrep = χ₀ (`coefficient_ring().one_basis()`).

        Implemented **directly** (independent of `_label_section_decompose`,
        which now *derives* from this by the KAlgebra forward bridge) so that
        method is obsoletable, and `forget()` / ring-hom flavour reduction
        (`base_change(restriction)`) + promotion (`base_change(unit_hom)`) read
        this coordinate cleanly.  A realisation that instead carries the flavour
        irrep in a **label slot** (e.g. `UNNfKAlgebra`'s `((m, λ), w)`) overrides
        this with its gauge/weight split."""
        return label, self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        """Inverse of the trivial default `r_label_decompose`: the label is the
        section (the flavour is central in the coefficient ring, so there is no
        `embed_R`/`multiply` round-trip).  Overridden alongside
        `r_label_decompose` by flavour-in-a-label-slot realisations."""
        return section

    def _flavour_element(self, ring, lev: tuple, c):
        """Package one torus μ-level coefficient as a coefficient-ring
        element.  Default: the abelian identification (level vector =
        ring basis key); rank-0 shapes use the bare integer convention.
        The D8 SU(M) packaging overrides this in the flavoured shells."""
        if not lev:
            return c
        return RElement(ring, {tuple(lev): c})

    @staticmethod
    def _trace_levels(tr):
        """Normalize a chart trace to the per-μ-level form `{lev: LaurentPoly}`.
        Matter / quiver charts return that dict directly; a **pure-gauge**
        `WRQTorus` trace is a bare `LaurentPoly` (no matter levels) — treat it as
        the single trivial level `()`."""
        return tr if isinstance(tr, dict) else {(): tr}

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_a)` — the torus trace (product Schur-measure residue with
        the matter M-factors), packaged over the coefficient ring."""
        R = self.coefficient_ring()
        acc: dict = {}
        for lev, lp in self._trace_levels(self.chart(a).trace(K=K)).items():
            for e, c in lp._coeffs.items():
                if not (0 <= e <= K) or not c:
                    continue
                cur = acc.get(e)
                fe = self._flavour_element(R, tuple(lev), c)
                acc[e] = fe if cur is None else (cur + fe)
        return RPowerSeries(R, acc, K)

    def inner_product(self, a: Label, b: Label, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — evaluated entirely chart-side
        (torus ρ, torus product, torus trace): exact, no decompose."""
        R = self.coefficient_ring()
        prod = self.chart(a).rho() * self.chart(b)
        acc: dict = {}
        for lev, lp in self._trace_levels(prod.trace(K=K)).items():
            for e, c in lp._coeffs.items():
                if not (0 <= e <= K) or not c:
                    continue
                cur = acc.get(e)
                fe = self._flavour_element(R, tuple(lev), c)
                acc[e] = fe if cur is None else (cur + fe)
        return RPowerSeries(R, acc, K)

    # ------------------------------------------------------------------
    # Chart-level certificates / verifiers.
    # ------------------------------------------------------------------

    def verify_chart_bar(self, a: Label) -> bool:
        """Bar-invariance of `L_a` in chart form — W1: every residual
        component q-palindromic (`bar(chart) == chart`)."""
        return bool(self.chart(a).well_formed_w1())

    def certify_canonical(self, a: Label):
        """The executable KL acceptance of `L_a`'s chart: `well_formed()`
        = W1 + single-leading-orbit q-extreme (W2).  Returns the torus's
        `(joint lower-Kapustin label, base level)` read — the realisation's
        tests assert it against their own label map — or ``False``."""
        return self.chart(a).well_formed()
