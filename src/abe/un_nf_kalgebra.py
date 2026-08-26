"""`UNNfKAlgebra` — the native `A_𝖖[U(N)+N_f]` on the `AbeKAlgebra`
contract (ruling T4, increment 1: N_f = 1).

Labels are **`(m, λ)` only** — per the D8b ruling (user, 2026-06-12) the
matter is organized as (fundamental of U(N)) ⊗ (anti-fundamental of
SU(N_f)): there is no U(1) flavour factor at any stage, so at N_f = 1 the
flavour group is trivial and the canonical basis carries no flavour
coordinate.  The torus μ-levels survive only as **internal refinement
bookkeeping** (they make the de-dressing triangular); `decompose` projects
them away at the label boundary, and the trace sums them (the SU(1)
character).  N_f ≥ 2 (labels `(m, λ; w)` with `w` an SU(N_f) weight,
coefficient ring `R(SU(N_f))`) follows once the `SUNZPlusRing` packaging
lands.

The substrate is the group-general **`MatterWRQTorus`** (D9/D10: the WRQTorus
selected by `torus_shape()` — one U(N) node with `N_f` fundamentals).  The
contract triple:

  * `chart(label)` — the canonical as a `MatterWRQTorus`: the certified
    Route-A flow image (`_urq_chart`, the matter flow de-dressed onto the
    enriched atoms) **transported** onto the WRQ substrate (`vr_to_tr`).  The
    SU(N_f) character `χ_w` enters as μ-level shifts (`L_{(g,w)} = χ_w(μ)·
    L_{(g,())}`).  Cached.
  * `decompose(x)` — level-ascending attribution: at the global lowest μ-level
    the slice is the base of some canonical (no target), read by the shared
    pure-U(N) WRQ engine (`PureUNWRQ.decompose`); subtract the level-shifted
    base chart; recurse; the flavour charge is projected into SU(N_f)
    characters (D8b).  Honest-fails off scope.
  * `torus_shape()` = the one-node `TorusShape` with `N_f` fundamentals.

Derived through the contract: multiply, ρ/ρ⁻¹ (the matter twist reads
back as a label permutation after projection), trace / inner product
(matter-measure residue, flavour-summed), W1 + `certify_canonical`.

Internal machinery: a private `UNNfOverPure` instance supplies the pure
substrate and the certified flow images (`_urq_chart` transport source) — the
flow remains the cross-validation oracle in the tests.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abe_kalgebra import AbeKAlgebra
from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing

__all__ = ["UNNfKAlgebra"]


class UNNfKAlgebra(AbeKAlgebra):
    """Native U(N)+N_f on the abelianized contract (N_f = 1 increment)."""

    def __init__(self, N: int, Nf: int = 1, **flow_kwargs) -> None:
        from un_nf_over_pure_rgflow import UNNfOverPure
        from zplus_ring import SUNZPlusRing
        self._N = int(N)
        self._Nf = int(Nf)
        self._flow = UNNfOverPure(self._N, self._Nf, **flow_kwargs)
        self._P = self._flow.pure()
        self._R = (TrivialZPlusRing() if self._Nf == 1
                   else SUNZPlusRing(self._Nf))
        self._charts: dict = {}        # URQ transport-source cache
        self._wrq_charts: dict = {}    # MatterWRQTorus cache (the live substrate)
        self._wrq_pure = None          # lazy PureUNWRQ (per-level pure decompose)

    # ----- KAlgebra basics -------------------------------------------

    def coefficient_ring(self):
        """`R(SU(N_f))` (D8/D8b); trivial at N_f = 1 (SU(1))."""
        return self._R

    def identity(self):
        g = ((0,) * self._N, (0,) * self._N)
        return g if self._Nf == 1 else (g, ())

    def _gauge_w(self, label):
        """Split a label into (gauge `(m, λ)`, SU weight `w`)."""
        if self._Nf == 1:
            return (tuple(label[0]), tuple(label[1])), ()
        g, w = label
        return (tuple(g[0]), tuple(g[1])), self._R.reduce(w)

    def _mk_label(self, g, w):
        return g if self._Nf == 1 else (g, tuple(w))

    def _rho_closed_form(self, label, inverse: bool):
        """The explicit label-level ρ^{±1} (promoted 2026-08-23; the twist
        route is the verifier `verify_rho_via_twist`).  Gauge part: the pure
        U(N) class's own certified label maps (`PureUNKAlgebra.rho` /
        `rho_inverse` — the Witten-shift closed form, so the lower-Kapustin
        frame stays that class's business) followed by the matter shift of
        the general closed form (`wrq_torus.rho_label`) expressed at the
        image magnetic `n`:

            λ'_i −= N_f · max(0, −n_i)   for ρ      (the image-atom ladder)
            λ'_i −= N_f · max(0, +n_i)   for ρ⁻¹    (the source-atom ladder,
                                                     `n = −m₋` slot-aligned)

        (`N_f` defining slots; the shift is constant on equal-`n` blocks, so
        the stored Levi-dominant sorting is untouched).  Flavour: `w ↦ w^⋆`
        (the ring's rep-ring duality `star_basis`; the det twist dies under
        the D8b central specialization)."""
        (m, lam), w = self._gauge_w(label)
        Nf = self._Nf
        if inverse:
            n, lam2 = self._P.rho_inverse((tuple(m), tuple(lam)))
            shift = [Nf * max(0, x) for x in n]
        else:
            n, lam2 = self._P.rho((tuple(m), tuple(lam)))
            shift = [Nf * max(0, -x) for x in n]
        g = (tuple(n), tuple(x - s for x, s in zip(lam2, shift)))
        wf = w if self._Nf == 1 else self._R.star_basis(w)
        return self._mk_label(g, wf)

    def rho(self, label):
        return self._rho_closed_form(label, inverse=False)

    def rho_inverse(self, label):
        return self._rho_closed_form(label, inverse=True)

    def _label_section_decompose(self, label):
        """**Aspirationally obsolete** — superseded by
        `r_label_decompose` (same gauge section + the single SU(N_f) irrep,
        returned as the bare weight key `w` rather than wrapped as `χ_w`).
        Kept, not retired: `to_R_form` still routes through it (it derives from
        `r_label_decompose` by the KAlgebra forward bridge)."""
        if self._Nf == 1:
            return (label, self.coefficient_ring().one())
        from zplus_ring import RElement
        g, w = self._gauge_w(label)
        return (self._mk_label(g, ()), RElement(self._R, {w: 1}))

    def r_label_decompose(self, label):
        """Flavour-lift coordinate `(gauge section, SU(N_f) weight w)`: the
        gauge `(m, λ)` carried at neutral flavour is the section, the single
        irrep is the reduced dominant weight key `w` (the highest weight of
        `χ_w`).  `N_f = 1` (SU(1), unflavoured): the trivial lift `(label, χ₀)`.

        Implemented **directly** from the gauge/weight split (`_gauge_w` /
        `_mk_label`), independent of `_label_section_decompose` — so the latter
        is obsoletable, and `forget()` / ring-hom flavour reduction (SU(N_f) →
        SU(M)) / promotion read this coordinate cleanly."""
        if self._Nf == 1:
            return (label, self.coefficient_ring().one_basis())
        g, w = self._gauge_w(label)
        return self._mk_label(g, ()), w

    def r_label_compose(self, section, r_basis_label):
        """Inverse: re-attach the SU(N_f) weight `r_basis_label` to the gauge
        section (a label rebuild — no `embed_R` round-trip)."""
        if self._Nf == 1:
            return section
        g, _ = self._gauge_w(section)
        return self._mk_label(g, r_basis_label)

    # ----- the contract triple ---------------------------------------

    def torus_shape(self):
        from abe_kalgebra import TorusShape
        return TorusShape.from_ranks_nf((self._N,), (self._Nf,))

    def _su_weights(self, w):
        """The weight multiset of the SU(N_f) irrep `w` as level vectors
        (Kostka multiplicities) — the flavour content of `χ_w`."""
        if not w:
            return {(0,) * self._Nf: 1}
        return self._R._mono(tuple(w))

    def _urq_chart(self, label):
        """The canonical as a `MatterURQTorus`: the **certified flow
        image** de-dressed onto the enriched atoms (Route-A provenance,
        as in `UNQuiverKAlgebra`), with the SU(N_f) character `χ_w`
        attached as level shifts (`L_{(g,w)} = χ_w(μ)·L_{(g,())}`).

        PROVENANCE NOTE (2026-06-13): charts were originally the
        constructive `matter_image` canonicals; the pure-SU(2)-sector
        cross-check (`su2_nf1_object`) caught a **silent constructive
        gap** at mixed-sign dyonic labels — `matter_image((2,−2),(1,0))`
        drops the magnetic-bubbling term `((1,−1),(1,1))·μ` (flow
        coefficient 1) without honest-failing, and the native build
        honest-fails there (the sibling-dressing-cycle gap).  Until the
        engine closes that shape, the shell takes the certified flow
        images; `_constructive_chart` keeps the engine path for
        cross-checks on its certified scope."""
        g, w = self._gauge_w(label)
        key = (g, w)
        x = self._charts.get(key)
        if x is None:
            base = self._urq_base_chart(g)
            if not w:
                x = base
            else:
                from matter_urq_torus import MatterURQTorus
                from laurent_poly import LaurentPoly as _LP
                out: dict = {}
                for kv, mult in self._su_weights(w).items():
                    for atom, row in base.residuals().items():
                        dst = out.setdefault(atom, {})
                        for k0, vr in row.items():
                            kk = tuple(a + b for a, b in zip(k0, kv))
                            term = vr if mult == 1 else (
                                vr * mult)
                            dst[kk] = term if kk not in dst else (
                                dst[kk] + term).simplify()
                x = MatterURQTorus(out, self._N, self._Nf)
            self._charts[key] = x
        return x

    def _constructive_chart(self, g):
        """The engine (`matter_image`) chart — the constructive path,
        kept for cross-checks on its certified scope (see the chart()
        provenance note)."""
        from abelianized_torus import VRational
        from urq_torus import URQTorus
        from matter_urq_torus import MatterURQTorus
        from un_nf_dressed_generators import matter_image
        img = matter_image(self._flow, g[0], g[1])
        fam: dict = {}
        for (plab, k), c in img.terms.items():
            lp = c if isinstance(c, LaurentPoly) else c.expand(48)
            cv = VRational.from_scalar(lp, n=self._N)
            add = URQTorus.from_f(
                {a: (f * cv).simplify()
                 for a, f in self._P.urqt(plab).residuals().items()},
                self._N)
            kk = tuple(k)
            fam[kk] = add if kk not in fam else (fam[kk] + add)
        return MatterURQTorus.from_family(
            {k: u for k, u in fam.items() if u.residuals()},
            self._N, self._Nf)

    def _urq_base_chart(self, g):
        key = (g, None)
        x = self._charts.get(key)
        if x is None:
            from abelianized_torus import VRational
            from urq_torus import URQTorus
            from matter_urq_torus import MatterURQTorus
            img = self._flow.RG((g, (0,) * self._Nf))
            fam: dict = {}
            for (plab, k), c in img.terms.items():
                lp = c if isinstance(c, LaurentPoly) else c.expand(48)
                cv = VRational.from_scalar(lp, n=self._N)
                add = URQTorus.from_f(
                    {a: (f * cv).simplify()
                     for a, f in self._P.urqt(plab).residuals().items()},
                    self._N)
                kk = tuple(k)
                fam[kk] = add if kk not in fam else (fam[kk] + add)
            x = MatterURQTorus.from_family(
                {k: u for k, u in fam.items() if u.residuals()},
                self._N, self._Nf)
            self._charts[key] = x
        return x

    # ----- WRQ substrate: transport the Route-A flow charts onto the
    #       group-general MatterWRQTorus (D9/D10; the live substrate) -----

    def _transport(self, mturq):
        """A `MatterURQTorus` flow chart (the URQ transport source) → the
        group-general `MatterWRQTorus` over `self.torus().datum` (`u_n(N)`).
        The transport is an isomorphism (the suite in the source repository)."""
        from matter_wrq_torus import MatterWRQTorus, vr_to_tr
        D = self.torus().datum
        return MatterWRQTorus(D, self._Nf, {
            m: {k: vr_to_tr(D, vr) for k, vr in row.items()}
            for m, row in mturq.residuals().items()})

    def chart(self, label):
        """`L_label` as a `MatterWRQTorus` — the certified `_urq_chart` flow
        image transported onto the WRQ substrate (Route-A provenance)."""
        g, w = self._gauge_w(label)
        key = (g, w)
        x = self._wrq_charts.get(key)
        if x is None:
            x = self._transport(self._urq_chart(label))
            self._wrq_charts[key] = x
        return x

    def _wrq_base(self, g):
        """The WRQ base gauge chart (transport of `_urq_base_chart`) — used by
        `_attribute`'s subtract step (`_urq_base_chart` itself stays URQ so the
        transport source keeps building on the URQ side)."""
        key = (g, "wrq-base")
        x = self._wrq_charts.get(key)
        if x is None:
            x = self._transport(self._urq_base_chart(g))
            self._wrq_charts[key] = x
        return x

    def _attribute(self, x, max_rounds: int = 512) -> dict:
        """Flow-basis attribution `{(g, k⃗): C}` on the WRQ substrate by
        level-ascending subtraction of the shell's OWN charts (the certified
        flow images): at the global lowest μ-level every term is the base of
        some canonical (no target — unambiguous), read by the shared pure-U(N)
        WRQ engine (`PureUNWRQ.decompose`); subtract the full level-shifted
        base chart; recurse.  Self-consistent by construction — the same
        images that `chart` serves (the constructive-engine subtraction this
        replaces was silently gapped at mixed-sign dyonic labels; see the
        `_urq_chart` provenance note)."""
        if self._wrq_pure is None:
            from wrq_torus import PureUNWRQ
            self._wrq_pure = PureUNWRQ(self.torus().datum)
        cur = x
        out: dict = {}
        for _ in range(max_rounds):
            fam = cur.to_family()
            if not fam:
                return out
            k0 = min(fam, key=lambda k: (sum(k), k))
            for plab, C in self._wrq_pure.decompose(fam[k0]).items():
                g = (tuple(plab[0]), tuple(plab[1]))
                key = (g, tuple(k0))
                out[key] = out.get(key, LaurentPoly.zero()) + C
                cur = cur + self._wrq_base(g)._scaled(
                    C * LaurentPoly({0: -1}), k0)
        raise NotImplementedError(
            "UNNfKAlgebra._attribute: attribution did not terminate")

    def decompose(self, x) -> Element:
        """Per-level pure registry peel + level-ascending attribution
        against the shell's own charts → flavour projection (D8b)."""
        attributed = {(lab, k): C
                      for (lab, k), C in self._attribute(x).items()
                      if not C.is_zero()}
        if self._Nf == 1:
            out: dict = {}
            for (lab, _k), C in attributed.items():
                key = (tuple(lab[0]), tuple(lab[1]))
                out[key] = out.get(key, LaurentPoly.zero()) + C
            return Element({lab: C for lab, C in out.items()
                            if not C.is_zero()})
        # N_f >= 2: per gauge label, decompose the flavour content into
        # SU(N_f) characters (central fugacity specialized to 1 — an
        # algebra homomorphism, so associativity/bar survive; D8b).
        by_gauge: dict = {}
        for (lab, k), C in attributed.items():
            g = (tuple(lab[0]), tuple(lab[1]))
            by_gauge.setdefault(g, {})[tuple(k)] = \
                by_gauge.get(g, {}).get(tuple(k), LaurentPoly.zero()) + C
        out = {}
        for g, levels in by_gauge.items():
            for w, C in self._su_peel(levels).items():
                if not C.is_zero():
                    out[self._mk_label(g, w)] = C
        return Element(out)

    def _su_norm(self, kv):
        """The canonical representative of `kv` mod the central direction
        (subtract the min — entries ≥ 0 with at least one zero)."""
        m = min(kv)
        return tuple(x - m for x in kv)

    def _su_peel(self, levels: dict) -> dict:
        """`{k⃗: C(q)} → {w: C_w(q)}`: collapse the central direction and
        decompose into SU(N_f) characters by the greedy dominance peel
        (χ_w linearly independent on the SU torus — exact, signs free)."""
        work: dict = {}
        for kv, C in levels.items():
            nk = self._su_norm(kv)
            work[nk] = work.get(nk, LaurentPoly.zero()) + C
        work = {k: C for k, C in work.items() if not C.is_zero()}
        out: dict = {}
        guard = 0
        while work:
            guard += 1
            if guard > 10000:
                raise NotImplementedError("UNNfKAlgebra: SU peel stuck")
            nu = max(work, key=lambda v: tuple(sorted(v, reverse=True)))
            nu_dom = tuple(sorted(nu, reverse=True))
            C = work.get(nu_dom)
            if C is None:
                raise NotImplementedError(
                    f"UNNfKAlgebra: SU peel — dominant rep {nu_dom} absent")
            w = self._R.reduce(nu_dom)
            out[w] = out.get(w, LaurentPoly.zero()) + C
            for kv, mult in self._su_weights(w).items():
                nk = self._su_norm(kv)
                cur = work.get(nk, LaurentPoly.zero()) + C * (-mult)
                if cur.is_zero():
                    work.pop(nk, None)
                else:
                    work[nk] = cur
        return out

    # ----- flavour packaging (D8b) ------------------------------------

    def _flavour_element(self, ring, lev, c):
        """Per-μ-level flavour packaging — HONEST-FAIL at `N_f ≥ 2`.

        The SU(N_f) refinement cannot be expressed one level at a time (the
        un-branching into SU characters needs the whole weight diagram per
        q-order — `_package_levels`), so this class's traces/pairings go
        through the overrides below.  This hook used to silently return the
        bare coefficient, which made the TIER's chart-side `trace` /
        `inner_product` fabricate the flavour-collapsed (dimension) series on
        this class — caught 2026-08-23 when `verify_inner_product_consistent`
        compared the SU-refined pairing against that collapsed reference.
        Fabrication is never acceptable; a route that cannot refine must
        raise."""
        if self._Nf == 1:
            return c
        raise NotImplementedError(
            "UNNfKAlgebra: per-level flavour packaging cannot express the "
            "SU(N_f) refinement — use this class's trace/inner_product "
            "(the _package_levels route)")

    def _package_levels(self, levels: dict, K: int):
        from zplus_ring import RElement, RPowerSeries
        if self._Nf == 1:
            acc0: dict = {}
            for lev, lp in levels.items():
                for e, c in lp._coeffs.items():
                    if 0 <= e <= K and c:
                        acc0[e] = acc0.get(e, 0) + c
            return RPowerSeries(self._R, acc0, K)
        # N_f >= 2: un-branch the Cartan trace into SU(N_f) characters with the
        # robust Weyl-denominator decomposition (`sun_characters.decompose`),
        # per q-order, dropping the gauge-centre level (`self._R.reduce`).  This
        # replaces the former hand-rolled highest-weight `_su_peel`, which was
        # incomplete at N_f >= 4 (it raised on / fabricated spurious dominant
        # reps; see the suite in the source repository).
        import sun_characters as SC
        byq: dict = {}                       # {qexp: {Cartan weight: int}}
        for kv, lp in levels.items():
            wt = tuple(int(x) for x in kv)
            for e, c in lp._coeffs.items():
                if not (0 <= e <= K) or not c:
                    continue
                row = byq.setdefault(e, {})
                row[wt] = row.get(wt, 0) + int(c)
        acc: dict = {}
        for e, poly in byq.items():
            poly = {w: c for w, c in poly.items() if c}
            if not poly:
                continue
            chars: dict = {}
            for wdom, mult in SC.decompose(self._Nf, poly).items():
                if not mult:
                    continue
                # split_su_level normalises dominant U(N_f) weights (incl.
                # negative-level / anti-fundamental ones, which `reduce`
                # rejects) to the SU(N_f) partition, dropping the level.
                part, _lvl = SC.split_su_level(tuple(wdom))
                w = tuple(part)
                chars[w] = chars.get(w, 0) + int(mult)
            chars = {w: c for w, c in chars.items() if c}
            if chars:
                acc[e] = RElement(self._R, chars)
        return RPowerSeries(self._R, acc, K)

    def trace(self, a, K: int = 20):
        """`Tr(L_a)` — the matter-measure torus trace, flavour content
        assembled into SU(N_f) characters (central specialized; D8b)."""
        # W >= K: the mu-window lesson (a product path at partial flavour
        # level p costs at least q^p) — the default W=4 under-resolves K > 4.
        return self._package_levels(self.chart(a).trace(K=K, W=K + 2), K)

    def inner_product(self, a, b, K: int = 20):
        prod = self.chart(a).rho() * self.chart(b)
        return self._package_levels(prod.trace(K=K, W=K + 2), K)


if __name__ == "__main__":
    A = UNNfKAlgebra(2)
    E, F = ((1, 0), (0, 0)), ((0, -1), (0, 0))
    print("U(2)+N_f=1 native on the contract; labels (m, λ) only")
    print("  E·F =", {k: str(v) for k, v in A.multiply(E, F).terms.items()})
    print("  ρ(E) =", A.rho(E))
    print("  Tr(1, 4) =", dict(A.trace(A.identity(), 4).coeffs))
